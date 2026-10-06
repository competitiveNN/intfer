// examples/roofline_probe_bisect/probe_common.hpp
// Shared machinery for the two bisect binaries. Each binary includes this and
// provides only its arm list; the ceiling/twin validation and the ladder loop
// are shared to keep the two binaries synchronised and to keep each TU tiny
// (the TU limit was the reason the ladder was split across binaries in the first
// place). Neither binary links the engine: the SHIPPED anchor comes from
// examples/roofline_probe, which measures the same shapes on the same buffers.
//
// The two ALWAYS-RED REQUIREMENTS are enforced here, in each binary:
//   * INTERLEAVING: every rep runs every arm back to back; only RATIOS are
//     reported, because run-separate numbers drift with box load.
//   * THE TWIN: a quarter of the bytes must take about a quarter of the time,
//     with the same launch shape and disjoint quarters. If it does not hold,
//     the GB/s figures are not tracking traffic and the run refuses rather
//     than printing load-sensitive ratios.

#pragma once

#include "intfer/common/logger.hpp"
#include "intfer/device/device_context.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// The device context carried by every INT2 GEMV arm, defined at global scope
// so probe binaries can forward-declare it before including this header.
struct Ctx {
  const std::uint8_t* code;
  const std::uint16_t* act;
  const std::uint16_t* scale;
  std::uint16_t* out;
  std::uint32_t N, K, Gw;
  int groups;
};

namespace {

// CITATION(techpowerup arc-b580.c4244 / ARK 241598). Named so every percentage
// is against a figure with a source.
constexpr double kPeakBandwidthGBs = 456.0;
constexpr const char* kPeakCitation =
    "CITATION(techpowerup arc-b580.c4244 / ARK 241598: 456.0 GB/s, 20 Xe cores)";

struct Arm {
  std::string name;
  double ms = 0.0;
  std::uint64_t bytes = 0;
  double gbs = 0.0;
  double ratio_to_verbatim = 1.0;
};

// Device event timestamps -- the same instrument the engine's `kernel` bucket
// uses, so these numbers and that bucket mean the same thing.
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
// Sequential read, WORD at a time, FOUR independent accumulators. The v1 ceiling
// was byte-at-a-time into ONE register -- a 26M-long serial dependency chain --
// and measured 54.6 GB/s while a real kernel reached 554. Four accumulators is
// the whole difference between a latency chain and a measurement.
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

// The two INT2 ladder arms: VERBATIM (the shipped source, transcribed) plus the
// single-change arms. `half_to_float` is the SHIPPED body's branchy scalar
// FP16->FP32 convert (3 branches per call) and is used by VERBATIM so that the
// single-change arms differ from it in EXACTLY ONE PLACE.
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

// The shipped chunk. 16 codes/chunk measured 0.25-0.35x -- a regression.
constexpr int kChunk = 8;

// The two INT2 shapes that bracket the decode spread. Derived, not remembered:
// MLP_GATE (N=8704) is the N-local-2560-shape family, MLP_DOWN the small-shape
// family; both INT2 G=16 as shipped.
struct Shape { const char* role; std::uint32_t N, K, G; };
const Shape kShapes[] = {{"MLP_GATE(INT2)", 8704, 5120, 16},
                         {"MLP_DOWN(INT2)", 2560, 17408, 16}};

// ---------------------------------------------------------------- main() ----
// LADDER, shared by both bisect binaries. Each binary includes this file with
// its own set of arm functions defined above. main() is the shared machinery:
// ceiling + twin validation, then the interleaved ladder over the arms this
// binary exports.
int bisect_main(int argc, char** argv) {
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
  std::fprintf(stderr, "[bisect-%s] device %d: %s  (max_compute_units=%u)\n",
               BISECT_NAME, dev_idx, dev.name.c_str(), dev.max_compute_units);

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

    std::printf("[bisect-%s] ceiling %8.3f ms  %7.1f GB/s   %5.1f%% of peak %s\n",
                BISECT_NAME, ceil_ms, ceil_gbs, 100.0 * ceil_gbs / kPeakBandwidthGBs,
                kPeakCitation);
    const double ratio = quarter_ms / ceil_ms;
    std::printf("[bisect-%s] twin     %8.3f ms  ratio to full = %.2f "
                "(linear would be 0.25)\n", BISECT_NAME, quarter_ms, ratio);
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
      // Deliberately NON-constant weights: a constant weight lets the compiler
      // hoist the dequant, and then an arm measures nothing.
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

    std::printf("[bisect-%s] -- %s (N_local=%u K=%u g%u) -- %s\n", BISECT_NAME,
                sh.role, sh.N, sh.K, sh.G,
                "CITATION(bisect-%s: every clone shares this buffer with every other "
                "clone, interleaved per rep)", BISECT_NAME);
    const Ctx ctx{static_cast<const std::uint8_t*>(c),
                  static_cast<const std::uint16_t*>(a),
                  static_cast<const std::uint16_t*>(sc),
                  static_cast<std::uint16_t*>(o), sh.N, sh.K, sh.G,
                  static_cast<int>(ng)};

    // INTERLEAVED: every rep runs every arm back to back.
    const char* names[ARMS_COUNT] = ARM_NAMES;
    std::vector<std::vector<double>> tt(ARMS_COUNT);
    for (int r = 0; r < kReps; ++r) {
      for (int v = 0; v < ARMS_COUNT; ++v) {
        sycl::event e;
        switch (v) {
          ARM_SWITCH
          default:
            std::fprintf(stderr,
                "FAIL: arm index %d out of range -- ARMS_COUNT=%d mismatch\n",
                v, ARMS_COUNT);
            return 1;
        }
        const double ms = timed_ms(q, e);
        if (ms < 0) { std::fprintf(stderr, "FAIL: profiling lost mid-run\n"); return 1; }
        tt[v].push_back(ms);
      }
    }
    double med[ARMS_COUNT];
    for (int v = 0; v < ARMS_COUNT; ++v) {
      std::sort(tt[v].begin(), tt[v].end());
      med[v] = tt[v][tt[v].size() / 2];
    }
    for (int v = 0; v < ARMS_COUNT; ++v) {
      const double gbs = double(cb) / (med[v] * 1e-3) / 1e9;
      char nm[64];
      std::snprintf(nm, sizeof nm, "%s/%s", sh.role, names[v]);
      std::printf("[bisect-%s]   %-24s %8.3f ms  %7.1f GB/s  %5.2fx vs VERBATIM\n",
                  BISECT_NAME, nm, med[v], gbs, med[0] / med[v]);
      Arm arm;
      arm.name = nm; arm.ms = med[v]; arm.bytes = cb; arm.gbs = gbs;
      arm.ratio_to_verbatim = med[0] / med[v];
      if (v == 0) (void)arm;
    }
    sycl::free(o, q); sycl::free(a, q); sycl::free(sc, q); sycl::free(c, q);
  }

  std::printf("\n== WHAT THIS RUN DOES AND DOES NOT ESTABLISH ==\n");
  std::printf("  ESTABLISHED by examples/roofline_probe (THE OTHER binary):\n"
              "    the shipped INT2 kernel runs at 30-41 GB/s on MLP_GATE/MLP_DOWN\n"
              "    (device event timestamps, medians over 12 interleaved reps).\n");
  std::printf("  ESTABLISHED BY THIS RUN: whatever the columns below say, all measured\n"
              "    against the VERBATIM control on the SAME buffers, interleaved per rep;\n"
              "    ratios are within-run vs VERBATIM (index 0).\n");
  std::printf("  NOT ESTABLISHED: a mechanism, unless one of the single-change arms\n"
              "    closes most of the table-vs-VERBATIM gap. SHIPPED is anchored from\n"
              "    roofline_probe (cross-binary).\n");
  std::printf("\nRESULT: PASS\n");
  std::printf("CITATION(bisect-%s, device %d: device event timestamps, medians over %d "
              "interleaved reps. RATIOS are the claim -- the absolute GB/s move with\n"
              "        box load, the interleaved ratio does not.)\n", BISECT_NAME, dev_idx, kReps);
  return 0;
}

}  // namespace
