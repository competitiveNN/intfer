// examples/roofline_probe_bisect -- bisect-2 of the M=1 INT2 GEMV probe ladder.
//
// WHY THIS IS A SEPARATE BINARY. The shipped INT2 kernel runs at 13.8-41.3 GB/s
// while a faithful clone of the same algorithm reaches 78-261 GB/s (examples/
// roofline_probe, interleaved, device event timestamps) -- a 5.6-6.4x gap whose
// mechanism is not located. Two candidate causes were measured and FALSIFIED:
//
//   module size   the shipped source in a SMALL TU bought only 1.26x, against a
//                 pre-committed >=2x kill criterion. The module theory is dead.
//   runtime branch  making use_int2_tab a template parameter (so `if constexpr`
//                 deletes the branch and one loop body) measured 1.007x.
//
// ~4.5-5x remains in the shared text, and the next steps were SINGLE-CHANGE arms
// off `VERBATIM`. This binary holds the second (and runnable) half of the split
// ladder. It carries no `SHIPPED` arm -- that anchor is measured by
// examples/roofline_probe on the same shapes/buffers and is cited from there, so
// the TU stays small enough to run on this box and there is no engine link that
// can trip the level-zero resource limit.
//
// THE ARMS, each differing from `VERBATIM` in EXACTLY ONE PLACE:
//
//   VERBATIM    the SHIPPED source text, transcribed. Baseline for all arms here.
//   VERB+HALF   ONE change: hardware `bit_cast<half>` for the GROUP scale instead
//               of `half_to_float`, the engine's branchy scalar FP16->FP32 twiddle
//               (3 branches per call). Called once per GROUP (320x per work-item
//               at MLP_GATE). Isolated: the TAIL keeps the base kernel's convert
//               so this arm measures the group-scale change alone.
//   VERB+TAIL   ONE change: hardware `bit_cast<half>` for the TAIL activation
//               scale instead of `half_to_float`. The group scale keeps the base
//               kernel's convert so this arm measures the tail-change alone. The
//               tail convert is called 320x per work-item at MLP_GATE (vs 1x for
//               the group scale), so it is the stronger candidate for the gap.
//               NUMERICS: for this probe's normal-range synthetic data the two
//               convert paths are bit-identical; the engine variant would need a
//               stream-identity gate like INTFER_DECODE1_VERBHALF.
//   VERB+MP     ONE change: the weight side read through a row-base `multi_ptr`
//               instead of indexing raw `code_u8` with an absolute byte offset.
//               (chunk loop: row-relative `b0`; tail: unoffset `code_u8` + absolute
//               indices, exactly as the base kernel -- the same double-offset class
//               of defect that lost the device in the original INTFER_DECODE1_ROWPAIR
//               is deliberately avoided here). BIT-EXACT BY CONSTRUCTION.
//
// THE TWO ALWAYS-RED REQUIREMENTS, because this is a measurement and not a report:
//
//   * INTERLEAVING. Every rep runs every arm back to back and only RATIOS are
//     reported. Run separately, the shipped arm read 17.2, 39.3 and 41.3 GB/s on
//     three consecutive runs of the SAME binary on this contended box -- a number
//     measured minutes away from its comparator is not a comparison.
//   * THE TWIN. A quarter of the bytes must take about a quarter of the time, with
//     the same launch shape and disjoint quarters. If that does not hold, the GB/s
//     figures are not tracking traffic and the run exits 1. It has already caught
//     three bugs in the sibling probe, including a "ceiling" that measured 54.6
//     GB/s while a real kernel reached 554.
//
// EXIT: 0 = the twin held and every arm ran; 1 = the twin failed, an arm did not
//      run, or device event profiling is unavailable (in which case NO number is
//      reported -- a host timer would be load-sensitive and is not a substitute).

// sycl headers first so the arm function declarations below are type-complete
// before probe_common.hpp (which defines Ctx) is included.
#include <sycl/sycl.hpp>

// sycl/Ctx forward declarations: the arm names must be visible to
// probe_common.hpp's bisect_main (compiled at the include below), while Ctx
// is defined by probe_common.hpp.
struct Ctx;
sycl::event arm_verbatim(sycl::queue& q, const Ctx& c);
sycl::event arm_verb_half(sycl::queue& q, const Ctx& c);
sycl::event arm_verb_tail(sycl::queue& q, const Ctx& c);
sycl::event arm_verb_mp(sycl::queue& q, const Ctx& c);

#define BISECT_NAME "bisect2"
#define ARMS_COUNT 4
#define ARM_NAMES {"VERBATIM", "VERB+HALF", "VERB+TAIL", "VERB+MP"}
#define ARM_SWITCH \
  case 0: e = arm_verbatim(q, ctx); break; \
  case 1: e = arm_verb_half(q, ctx); break; \
  case 2: e = arm_verb_tail(q, ctx); break; \
  case 3: e = arm_verb_mp(q, ctx); break; \

#include "probe_common.hpp"

// ---- the INT2 ladder ---------------------------------------------------------

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

// `VERB+HALF`: ONE change from VERBATIM -- hardware half->float for the GROUP
// scale. The tail keeps the base kernel's convert so this arm isolates the
// group-scale change alone.
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
        // THE ONLY DIFFERENCE FROM VERBATIM: hardware convert for the group scale.
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

// `VERB+TAIL`: ONE change from VERBATIM -- hardware half->float for the TAIL
// activation scale. The group scale keeps the base kernel's convert so this arm
// isolates the tail-change alone. The tail convert is called 320x per work-item
// at MLP_GATE, versus once per GROUP for the group scale, so it is the stronger
// candidate for the ~5x gap.
sycl::event arm_verb_tail(sycl::queue& q, const Ctx& c) {
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
          // THE ONLY DIFFERENCE FROM VERBATIM: hardware convert for the tail.
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

// `VERB+MP`: ONE change from VERBATIM -- the weight side through a row-base
// `multi_ptr`. The chunk loop uses a row-relative `b0`, the tail uses the
// unoffset `code_u8` with absolute indices (exactly as the base kernel).
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
      // THE ONLY DIFFERENCE FROM VERBATIM: the weight side through a row-base
      // multi_ptr. row_base is an element offset, packed four INT2 codes per
      // byte, so the multi_ptr base is converted to bytes; the chunk loop then
      // uses a row-RELATIVE b0, and the tail uses the unoffset base.
      sycl::multi_ptr<const std::uint8_t,
          sycl::access::address_space::global_space> wmp(c.code + row_base / 4u);
      for (int g = 0; g < c.groups; ++g) {
        const float scale = half_to_float(sptr[g]);
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw),
                                 static_cast<int>(c.K));
        const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                               scale * (1.0f / 3.0f), scale * 1.0f};
        int kg = kg0;
        // RELATIVE to the row base, because `wmp` already carries it.
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
              (c.code[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * scale * canon2(code);
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

int main(int argc, char** argv) {
  return bisect_main(argc, argv);
}
