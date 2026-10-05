// examples/roofline_probe_bisect -- the SECOND half of the M=1 INT2 GEMV bisect.
//
// WHY THIS IS A SEPARATE BINARY. examples/roofline_probe established that the
// shipped INT2 kernel runs at 13.8-41.3 GB/s while a faithful clone of the same
// algorithm reaches 78-261 GB/s -- a 5.6-6.4x gap, reproduced interleaved so
// contention cannot explain it. Two candidate causes were then measured and FALSIFIED:
//
//   module size   the shipped source in a SMALL TU bought only 1.26x, against a
//                 pre-committed >=2x kill criterion. The module theory is dead.
//   runtime branch  making use_int2_tab a template parameter (so `if constexpr`
//                 deletes the branch and one loop body) measured 1.007x.
//
// What remained was ~4.5-5x in the SHARED TEXT of the two functions, and the next two
// single-change steps were written and never measured, because adding them pushed the
// probe's translation unit past the level-zero UR_RESULT_ERROR_OUT_OF_RESOURCES (40)
// limit on submit. One probe TU holds roughly six to eight INT2 kernel bodies.
//
// So the ladder is SPLIT across two binaries rather than shrunk further. THIS one
// carries the two blocked arms; roofline_probe keeps the module/branch arms. The two
// binaries share the `table` control and the shipped kernel, so each can report a
// within-binary ratio without cross-binary absolutes.
//
// THE ARMS, each differing from `VERBATIM` in exactly ONE place:
//
//   table        a TIDY rewrite of the shipped inner loop. The fast end of the
//                ladder, and the control both binaries share.
//   VERBATIM     the SHIPPED source text, transcribed, compiled in this small TU.
//                Every other arm is one single change away from this one, which is
//                what makes the ladder attributable.
//   VERB+HALF    ONE change: `from_half_bits` (the engine's branchy SCALAR FP16->FP32
//                twiddle, 3 branches per call) replaced by `sycl::bit_cast<sycl::half>`.
//                Called once per GROUP, so 320x per work-item at MLP_GATE.
//   VERB+MP      ONE change: the weight side read through a `multi_ptr` instead of a
//                raw `code_u8[b0]`. A multi_ptr is what lets the compiler keep the
//                base in a register and fold the addressing.
//   SHIPPED      the real `intfer::gemm::quant_gemm_forward_usm`, same buffers, same
//                shape, same instrument.
//
// THE TWO ALWAYS-RED REQUIREMENTS, because this is a measurement and not a report:
//
//   * INTERLEAVING. Every rep runs every arm back to back and only RATIOS are
//     reported. Run separately, the shipped arm read 17.2, 39.3 and 41.3 GB/s on three
//     consecutive runs of the SAME binary on this contended box -- a number measured
//     minutes away from its comparator is not a comparison.
//   * THE TWIN. A quarter of the bytes must take about a quarter of the time, with
//     the same launch shape and disjoint quarters. If that does not hold, the GB/s
//     figures are not tracking traffic and the run exits 1. It has already caught
//     three bugs in the sibling probe, including a "ceiling" that measured 54.6 GB/s
//     while a real kernel reached 554.
//
// EXIT: 0 = the twin held and every arm ran; 1 = the twin failed, an arm did not run,
//        or device event profiling is unavailable (in which case NO number is
//        reported -- a host timer would be load-sensitive and is not a substitute).

#include "intfer/common/logger.hpp"
#include "intfer/device/device_context.hpp"
#include "intfer/gemm/quant_gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// CITATION(techpowerup arc-b580.c4244 / ARK 241598). Named constants so every
// printed percentage is against a figure with a source.
constexpr double kPeakBandwidthGBs = 456.0;
constexpr const char* kPeakCitation =
    "CITATION(techpowerup arc-b580.c4244 / ARK 241598: 456.0 GB/s, 20 Xe cores)";

struct Arm {
  std::string name;
  double ms = 0.0;
  std::uint64_t bytes = 0;
  double gbs = 0.0;
  double ratio_to_table = 1.0;
  double ratio_to_shipped = 1.0;
};

// Device event timestamps -- the same instrument the engine's `kernel` bucket uses,
// so these numbers and that bucket mean the same thing.
double timed_ms(sycl::queue& q, const sycl::event& e) {
  try {
    q.wait_and_throw();
    const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(t1 - t0) / 1e6;
  } catch (...) {
    return -1.0;
  }
}

// ---- the ceiling arm and its twin -------------------------------------------
// Sequential read, WORD at a time, FOUR independent accumulators. The v1 ceiling was
// byte-at-a-time into ONE register -- a 26M-long serial dependency chain -- and
// measured 54.6 GB/s while a real kernel reached 554. Four accumulators is the whole
// difference between a latency chain and a measurement.
sycl::event stream_kernel(sycl::queue& q, const std::uint16_t* src,
                          std::uint64_t row_words, std::uint32_t n_rows,
                          std::uint32_t* sink) {
  constexpr int nWG = 16;
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_rows), sycl::range<1>(nWG)),
                   [=](sycl::nd_item<1> it) {
      const int ng = static_cast<int>(it.get_global_id(0));
      if (ng >= static_cast<int>(n_rows)) return;
      const auto* row = src + static_cast<std::uint64_t>(ng) * row_words;
      std::uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      std::uint64_t i = 0;
      for (; i + 4 <= row_words; i += 4) {
        a0 += row[i];     a1 += row[i + 1];
        a2 += row[i + 2]; a3 += row[i + 3];
      }
      for (; i < row_words; ++i) a0 += row[i];
      if (it.get_local_id(0) == 0) sink[ng] = a0 ^ a1 ^ a2 ^ a3;
    });
  });
}

// ---- the INT2 ladder ---------------------------------------------------------
constexpr int kChunk = 8;   // the shipped chunk. 16 measured 0.25-0.35x -- a regression.

inline float half_to_float(std::uint16_t h) {
  const std::uint32_t s = (std::uint32_t(h & 0x8000u)) << 16;
  const std::uint32_t e = (h >> 10) & 0x1Fu;
  const std::uint32_t m = h & 0x3FFu;
  if (e == 0) return std::bit_cast<float>(s);
  if (e == 31) return std::bit_cast<float>(s | 0x7F800000u | (m << 13));
  return std::bit_cast<float>(s | ((e - 15u + 127u) << 23) | (m << 13));
}
inline float canon2(int code) {
  static const float c2[4] = {-1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f};
  return c2[code & 3];
}

struct Ctx {
  const std::uint8_t* code;
  const std::uint16_t* act;
  const std::uint16_t* scale;
  std::uint16_t* out;
  std::uint32_t N, K, Gw;
  int groups;
};

// `table`: the tidy rewrite. Same arithmetic, no branch, chunk at 8.
sycl::event arm_table(sycl::queue& q, const Ctx& c) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(c.N), sycl::range<1>(16)),
                   [=](sycl::nd_item<1> it) {
      const int ng = static_cast<int>(it.get_global_id(0));
      if (ng >= static_cast<int>(c.N)) return;
      float acc = 0.0f;
      const std::uint16_t* sptr = c.scale + static_cast<std::size_t>(ng) * c.groups;
      const std::size_t rb = static_cast<std::size_t>(ng) * c.K;
      sycl::multi_ptr<const std::uint16_t,
          sycl::access::address_space::global_space> act_mp(c.act);
      for (int g = 0; g < c.groups; ++g) {
        const float sc =
            static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g]));
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw),
                                 static_cast<int>(c.K));
        const float tab[4] = {sc * -1.0f, sc * (-1.0f / 3.0f),
                              sc * (1.0f / 3.0f), sc * 1.0f};
        int kg = kg0;
        std::size_t b0 = (rb + static_cast<std::size_t>(kg)) / 4u;
        for (; kg + kChunk <= kg1; kg += kChunk) {
          sycl::vec<std::uint16_t, 8> a16;
          a16.load(0, act_mp + kg);
          const std::uint8_t v0 = c.code[b0], v1 = c.code[b0 + 1];
          b0 += 2;
          for (int j = 0; j < kChunk; ++j) {
            const float av =
                static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? v0 : v1;
            acc += av * tab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = static_cast<float>(sycl::bit_cast<sycl::half>(c.act[kg]));
          const std::size_t idx = rb + static_cast<std::size_t>(kg);
          acc += av * sc * canon2(static_cast<int>((c.code[idx / 4] >>
                                                   ((idx % 4) * 2)) & 3));
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

// `VERBATIM`: the SHIPPED source text. Everything else is one change from this.
sycl::event arm_verbatim(sycl::queue& q, const Ctx& c) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(c.N), sycl::range<1>(16)),
                   [=](sycl::nd_item<1> it) {
      const int ng = static_cast<int>(it.get_global_id(0));
      if (ng >= static_cast<int>(c.N)) return;
      float acc = 0.0f;
      const std::uint16_t* sptr = c.scale + static_cast<std::size_t>(ng) * c.groups;
      const std::size_t row_base = static_cast<std::size_t>(ng) * c.K;
      sycl::multi_ptr<const std::uint16_t,
          sycl::access::address_space::global_space> act_mp(c.act);
      for (int g = 0; g < c.groups; ++g) {
        const float scale = half_to_float(sptr[g]);
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw),
                                 static_cast<int>(c.K));
        const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                               scale * (1.0f / 3.0f), scale * 1.0f};
        int kg = kg0;
        std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
        for (; kg + kChunk <= kg1; kg += kChunk) {
          sycl::vec<std::uint16_t, 8> a16;
          a16.load(0, act_mp + kg);
          const std::uint8_t b0v = c.code[b0];
          const std::uint8_t b1v = c.code[b0 + 1];
          b0 += 2;
          for (int j = 0; j < kChunk; ++j) {
            const float av =
                static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? b0v : b1v;
            acc += av * wtab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = half_to_float(c.act[kg]);
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const int code = static_cast<int>(
              (c.code[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * scale * canon2(code);
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

// `VERB+HALF`: ONE change from VERBATIM -- hardware convert for the group scale.
sycl::event arm_verb_half(sycl::queue& q, const Ctx& c) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(c.N), sycl::range<1>(16)),
                   [=](sycl::nd_item<1> it) {
      const int ng = static_cast<int>(it.get_global_id(0));
      if (ng >= static_cast<int>(c.N)) return;
      float acc = 0.0f;
      const std::uint16_t* sptr = c.scale + static_cast<std::size_t>(ng) * c.groups;
      const std::size_t row_base = static_cast<std::size_t>(ng) * c.K;
      sycl::multi_ptr<const std::uint16_t,
          sycl::access::address_space::global_space> act_mp(c.act);
      for (int g = 0; g < c.groups; ++g) {
        // THE ONLY DIFFERENCE FROM VERBATIM.
        const float scale = static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g]));
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw),
                                 static_cast<int>(c.K));
        const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                               scale * (1.0f / 3.0f), scale * 1.0f};
        int kg = kg0;
        std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
        for (; kg + kChunk <= kg1; kg += kChunk) {
          sycl::vec<std::uint16_t, 8> a16;
          a16.load(0, act_mp + kg);
          const std::uint8_t b0v = c.code[b0];
          const std::uint8_t b1v = c.code[b0 + 1];
          b0 += 2;
          for (int j = 0; j < kChunk; ++j) {
            const float av =
                static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? b0v : b1v;
            acc += av * wtab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = static_cast<float>(sycl::bit_cast<sycl::half>(c.act[kg]));
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const int code = static_cast<int>(
              (c.code[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * scale * canon2(code);
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

// `VERB+MP`: ONE change from VERBATIM -- the weight side through a multi_ptr.
sycl::event arm_verb_mp(sycl::queue& q, const Ctx& c) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(c.N), sycl::range<1>(16)),
                   [=](sycl::nd_item<1> it) {
      const int ng = static_cast<int>(it.get_global_id(0));
      if (ng >= static_cast<int>(c.N)) return;
      float acc = 0.0f;
      const std::uint16_t* sptr = c.scale + static_cast<std::size_t>(ng) * c.groups;
      const std::size_t row_base = static_cast<std::size_t>(ng) * c.K;
      sycl::multi_ptr<const std::uint16_t,
          sycl::access::address_space::global_space> act_mp(c.act);
      // THE ONLY DIFFERENCE FROM VERBATIM.
      sycl::multi_ptr<const std::uint8_t,
          sycl::access::address_space::global_space> wmp(c.code + row_base);
      for (int g = 0; g < c.groups; ++g) {
        const float scale = half_to_float(sptr[g]);
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw),
                                 static_cast<int>(c.K));
        const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                               scale * (1.0f / 3.0f), scale * 1.0f};
        int kg = kg0;
        std::size_t b0 = static_cast<std::size_t>(kg) / 4u;
        for (; kg + kChunk <= kg1; kg += kChunk) {
          sycl::vec<std::uint16_t, 8> a16;
          a16.load(0, act_mp + kg);
          const std::uint8_t b0v = wmp[b0];
          const std::uint8_t b1v = wmp[b0 + 1];
          b0 += 2;
          for (int j = 0; j < kChunk; ++j) {
            const float av =
                static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? b0v : b1v;
            acc += av * wtab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = half_to_float(c.act[kg]);
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const int code = static_cast<int>(
              (wmp[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * scale * canon2(code);
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

struct Shape { const char* role; std::uint32_t N, K, G; };
// The two shapes that bracket the INT2 spread. Derived, not remembered:
const Shape kShapes[] = {{"MLP_GATE(INT2)", 8704, 5120, 16},
                         {"MLP_DOWN(INT2)", 2560, 17408, 16}};

}  // namespace

int main(int argc, char** argv) {
  int dev_idx = 0;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc)
      dev_idx = std::atoi(argv[++i]);

  auto mgr = intfer::device::DeviceManager::discover("Arc");
  if (mgr.size() == 0 || static_cast<std::size_t>(dev_idx) >= mgr.size()) {
    std::fprintf(stderr, "FAIL: no usable Intel Arc device (is setvars.sh sourced?)\n");
    return 1;
  }
  auto& dev = mgr[dev_idx];
  sycl::queue q = dev.queue;
  q.wait_and_throw();
  std::fprintf(stderr, "[bisect] device %d: %s  (max_compute_units=%u)\n",
               dev_idx, dev.name.c_str(), dev.max_compute_units);

  constexpr int kReps = 12;   // medians over interleaved reps, never first-run noise

  // ---- the ceiling and its twin --------------------------------------------
  double ceil_ms = 0.0, quarter_ms = 0.0, ceil_gbs = 0.0;
  {
    const std::uint32_t N = 5120, K = 5120;
    const std::uint64_t words = std::uint64_t(N) * K, bytes = words * 2;
    void* p = sycl::malloc_shared(bytes, dev.sycl_device, dev.sycl_context);
    auto* sink = static_cast<std::uint32_t*>(
        sycl::malloc_shared(std::uint64_t(N) * 4, dev.sycl_device, dev.sycl_context));
    if (!p || !sink) { std::fprintf(stderr, "FAIL: alloc\n"); return 1; }
    {
      std::vector<std::uint16_t> host(words);
      for (std::uint64_t i = 0; i < words; ++i)
        host[i] = static_cast<std::uint16_t>((i * 2654435761u) >> 20);
      std::memcpy(p, host.data(), bytes);
    }
    std::vector<double> t;
    for (int r = 0; r < kReps; ++r) {
      const double ms = timed_ms(q, stream_kernel(
          q, static_cast<const std::uint16_t*>(p), K, N, sink));
      if (ms < 0) {
        std::fprintf(stderr,
            "FAIL: device event profiling unavailable, so NO number this binary\n"
            "      prints would be trustworthy. Refusing rather than substituting a\n"
            "      host timer (load-sensitive).\n");
        return 1;
      }
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    ceil_ms = t[t.size() / 2];
    ceil_gbs = double(bytes) / (ceil_ms * 1e-3) / 1e9;

    // THE TWIN: same kernel, same launch shape, all N rows kept (so occupancy is
    // unchanged) but each row a QUARTER as long, and the offset advances per rep so
    // the reps read four DISJOINT quarters of the same span. A quarter of the bytes
    // must take about a quarter of the time.
    std::vector<double> tq;
    const std::uint64_t q4 = K / 4;
    for (int r = 0; r < kReps; ++r) {
      const std::uint32_t off =
          static_cast<std::uint32_t>(r % 4) * static_cast<std::uint32_t>(q4);
      tq.push_back(timed_ms(q, stream_kernel(
          q, static_cast<const std::uint16_t*>(p) + off, q4, N, sink)));
    }
    std::sort(tq.begin(), tq.end());
    quarter_ms = tq[tq.size() / 2];

    std::printf("[bisect] ceiling %8.3f ms  %7.1f GB/s   %5.1f%% of peak %s\n",
                ceil_ms, ceil_gbs, 100.0 * ceil_gbs / kPeakBandwidthGBs,
                kPeakCitation);
    const double ratio = quarter_ms / ceil_ms;
    std::printf("[bisect] twin     %8.3f ms  ratio to full = %.2f "
                "(linear would be 0.25)\n", quarter_ms, ratio);
    if (ratio < 0.10 || ratio > 0.55) {
      std::fprintf(stderr,
          "FAIL VACUOUS: a QUARTER of the bytes took %.0f%% of the time (%.2fx,\n"
          "        expected ~0.25x) with the SAME launch shape. Time is not scaling\n"
          "        with traffic, so the ratios below are not measuring a memory\n"
          "        system and none of them is readable.\n",
          100.0 * ratio, ratio);
      return 1;
    }
    sycl::free(sink, q); sycl::free(p, q);
  }

  // ---- the ladder ------------------------------------------------------------
  for (const Shape& sh : kShapes) {
    const std::uint64_t cb = std::uint64_t(sh.N) * sh.K / 4u;
    const std::uint32_t ng = (sh.K + sh.G - 1) / sh.G;
    void* c = sycl::malloc_shared(cb, dev.sycl_device, dev.sycl_context);
    void* sc = sycl::malloc_shared(std::uint64_t(sh.N) * ng * 2,
                                   dev.sycl_device, dev.sycl_context);
    void* a = sycl::malloc_shared(std::uint64_t(sh.K) * 2,
                                   dev.sycl_device, dev.sycl_context);
    void* o = sycl::malloc_shared(std::uint64_t(sh.N) * 2,
                                   dev.sycl_device, dev.sycl_context);
    if (!c || !sc || !a || !o) { std::fprintf(stderr, "FAIL: alloc\n"); return 1; }
    {
      // Deliberately NON-constant weights: a constant weight lets the compiler hoist
      // the dequant, and then an arm measures nothing.
      auto* cp = static_cast<std::uint8_t*>(c);
      for (std::uint64_t i = 0; i < cb; ++i)
        cp[i] = static_cast<std::uint8_t>((i * 40503u) >> 3);
      auto* sp = static_cast<std::uint16_t*>(sc);
      for (std::uint32_t i = 0; i < sh.N * ng; ++i)
        sp[i] = static_cast<std::uint16_t>(0x2C00 + (i % 7) * 37);
      auto* ap = static_cast<std::uint16_t*>(a);
      for (std::uint32_t i = 0; i < sh.K; ++i)
        ap[i] = static_cast<std::uint16_t>(0x3400 + (i % 13) * 64);
    }

    std::printf("[bisect] -- %s (N_local=%u K=%u g%u) -- %s\n", sh.role, sh.N,
                sh.K, sh.G,
                "CITATION(roofline_probe_bisect: the SHIPPED arm shares this "
                "buffer with every clone, interleaved per rep)");
    const Ctx ctx{static_cast<const std::uint8_t*>(c),
                  static_cast<const std::uint16_t*>(a),
                  static_cast<const std::uint16_t*>(sc),
                  static_cast<std::uint16_t*>(o), sh.N, sh.K, sh.G,
                  static_cast<int>(ng)};

    // INTERLEAVED: every rep runs every arm back to back.
    const char* names[5] = {"table", "VERBATIM", "VERB+HALF", "VERB+MP", "SHIPPED"};
    std::vector<std::vector<double>> tt(5);
    for (int r = 0; r < kReps; ++r) {
      for (int v = 0; v < 5; ++v) {
        sycl::event e;
        switch (v) {
          case 0: e = arm_table(q, ctx); break;
          case 1: e = arm_verbatim(q, ctx); break;
          case 2: e = arm_verb_half(q, ctx); break;
          case 3: e = arm_verb_mp(q, ctx); break;
          default:
            // ctx.ACT, not ctx.out. The first draft passed the OUTPUT buffer as
            // the activation buffer. It hung, and the natural reading -- "the
            // shipped kernel is broken here too" -- was neither proven nor right:
            // testing each arm ALONE showed all five hang, so the fault was never
            // in the shipped call at all. Recorded because the wrong reading is
            // the one a hang invites.
            e = intfer::gemm::quant_gemm_forward_usm(
                q, const_cast<std::uint16_t*>(ctx.act), ctx.code, ctx.scale, 1.0f,
                const_cast<std::uint16_t*>(ctx.out), /*M=*/1, ctx.N, ctx.K, ctx.Gw,
                intfer::gemm::WeightScheme::INT2, sycl::event{});
            break;
        }
        const double ms = timed_ms(q, e);
        if (ms < 0) { std::fprintf(stderr, "FAIL: profiling lost mid-run\n"); return 1; }
        tt[v].push_back(ms);
      }
    }
    double med[5];
    for (int v = 0; v < 5; ++v) {
      std::sort(tt[v].begin(), tt[v].end());
      med[v] = tt[v][tt[v].size() / 2];
    }
    for (int v = 0; v < 5; ++v) {
      const double gbs = double(cb) / (med[v] * 1e-3) / 1e9;
      char nm[64];
      std::snprintf(nm, sizeof nm, "%s/%s", sh.role, names[v]);
      std::printf("[bisect]   %-24s %8.3f ms  %7.1f GB/s  %5.2fx vs table  "
                  "%5.2fx vs SHIPPED\n",
                  nm, med[v], gbs, med[0] / med[v], med[4] / med[v]);
      Arm arm;
      arm.name = nm; arm.ms = med[v]; arm.bytes = cb; arm.gbs = gbs;
      arm.ratio_to_table = med[0] / med[v];
      arm.ratio_to_shipped = med[4] / med[v];
      if (v == 0 || v == 4) (void)arm;
    }
    sycl::free(o, q); sycl::free(a, q); sycl::free(sc, q); sycl::free(c, q);
  }

  std::printf("\n== WHAT THIS RUN DOES AND DOES NOT ESTABLISH ==\n");
  std::printf("  ESTABLISHED by roofline_probe (the OTHER binary): the shipped INT2\n");
  std::printf("    kernel is 5.6-6.4x slower than a clone of its own algorithm; moving\n");
  std::printf("    it to a small TU bought only 1.26x (module theory DEAD); making\n");
  std::printf("    use_int2_tab a template parameter bought 1.007x (branch theory DEAD).\n");
  std::printf("  ESTABLISHED BY THIS RUN: whatever the VERB+HALF and VERB+MP columns say.\n");
  std::printf("  NOT ESTABLISHED unless they beat VERBATIM: the mechanism. Two candidates\n");
  std::printf("    are already falsified, and ~4.5-5x remains in the shared text.\n");
  std::printf("\nRESULT: PASS\n");
  std::printf("CITATION(roofline_probe_bisect, device %d: device event timestamps, "
              "medians over %d interleaved reps. RATIOS are the claim -- the\n"
              "        absolute GB/s move with box load, the interleaved ratio does "
              "not.)\n", dev_idx, kReps);
  return 0;
}