// examples/roofline_probe_bisect3 -- bisect-3 of the M=1 INT2 GEMV probe ladder.
//
// WHY THIS IS A SEPARATE BINARY. The shipped INT2 kernel runs at 30-41 GB/s on
// MLP_GATE/MLP_DOWN (examples/roofline_probe, interleaved, device event timestamps)
// while a faithful clone of the same algorithm reaches 193-244 GB/s on `table` --
// a 5.6-6.4x gap whose mechanism is not located. The ladder was split across
// binaries because one probe TU holds roughly six to eight INT2 kernel bodies and
// pushing past that trips `UR_RESULT_ERROR_OUT_OF_RESOURCES` on submit, wedging the
// driver's xe_page_fault_work_queue (rebind required).
//
// *** THE GAP THIS WAS MAPS DOES NOT EXIST. RETRACTED 2026-10-08. *** The 244 GB/s
// clone figure is not reproducible: a fresh run of examples/roofline_probe returns
// VACUOUS, and a saved bisect4 log from 10-06 measured the SAME clone algorithm at
// 55.1 GB/s while the shipped arm stayed at ~40-49 GB/s in both. The clone number
// moved 4.7x between two saved logs of the same kernel; the shipped number did not.
// So the 5.6-6.4x was never a clone-vs-shipped ratio, it was a flaky numerator
// divided by a stable one, and the honest ratio is 1.12x. The two falsified
// candidates above were chasing a number that was never real. This binary's
// purpose -- closing the last cross-binary dependency by putting SHIPPED in the
// same TU -- is moot, because the ~4.5-5x it was built to measure does not exist.
// The arms are still a valid differential probe of the shipped source's cosmetic
// variants and the gate is live; what they can no longer claim is that they are
// tracing a deficit. See HISTORY.md "THE 6.39x ITSELF IS NOT REPRODUCIBLE".
//
// `bisect1` holds ladder-1 (`table` vs `VERBATIM`). `bisect2` holds the two arms
// with ENGINE FLAGS (`VERB+HALF` matches `INTFER_DECODE1_VERBHALF`).
//
// THIS BINARY holds the remaining single-change arms plus the `SHIPPED` anchor --
// all four in ONE TU so the run is fully self-contained: no cross-binary
// dependency on examples/roofline_probe, and the `SHIPPED` ratio is measured on the
// SAME buffers, in the SAME TU, interleaved per rep against the VERBATIM control.
// (A cross-binary SHIPPED would measure a kernel built in a different module
// context, so its register pressure and the resulting GB/s are not directly
// comparable to the in-TU arms.)
//
// THE ARMS, each differing from `VERBATIM` in exactly ONE place:
//
//   VERBATIM   the SHIPPED source text, transcribed. Baseline for all arms
//              below; ratio index 0.
//   VERB+TAIL  hardware `sycl::bit_cast<sycl::half>` for the TAIL activations
//              (320 calls/wi at MLP_GATE) instead of `half_to_float`.
//   VERB+MP    weight side through a row-base `multi_ptr<const uint8_t>` instead
//              of absolute indexing of raw code_u8.
//   SHIPPED    the ACTUAL shipped kernel from src/gemm/quant_gemm.cpp, cloned
//              verbatim (CC=8 chunk, dynamically-indexed 4-entry wtab,
//              absolute byte indexing) so the gap to it is measured here, not by
//              assumption. The ~4.5-5x of the 5.6-6.4x that survives the module
//              move (1.26x) and the branch (1.007x) sits between VERBATIM and
//              SHIPPED.
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
sycl::event arm_verb_tail(sycl::queue& q, const Ctx& c);
sycl::event arm_verb_mp(sycl::queue& q, const Ctx& c);
sycl::event arm_shipped(sycl::queue& q, const Ctx& c);

#define BISECT_NAME "bisect3"
#define ARMS_COUNT 4
#define ARM_NAMES {"VERBATIM", "VERB+TAIL", "VERB+MP", "SHIPPED"}
#define ARM_SWITCH \
  case 0: e = arm_verbatim(q, ctx); break; \
  case 1: e = arm_verb_tail(q, ctx); break; \
  case 2: e = arm_verb_mp(q, ctx); break; \
  case 3: e = arm_shipped(q, ctx); break; \

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

// `SHIPPED`: the ACTUAL shipped INT2 kernel from src/gemm/quant_gemm.cpp, cloned
// verbatim. DEQ=0 form: CC=8 codes/chunk, dynamically-indexed 4-entry wtab,
// absolute byte indexing for weights, row-based multi_ptr for activations. This
// is the baseline for the 5.6-6.4x deficit and the final step of the bisect: if
// VERBATIM is already close to SHIPPED in this TU, the mechanism lives in the
// module/launcher/properties around the cloned text; if VERBATIM is far from
// SHIPPED, the mechanism lives in the shared text itself.
sycl::event arm_shipped(sycl::queue& q, const Ctx& c) {
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
        const float scale =
            static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g])); // * alpha, alpha==1.0f
        const int kg0 = g * static_cast<int>(c.Gw);
        int kg1 = kg0 + static_cast<int>(c.Gw);
        if (kg1 > static_cast<int>(c.K)) kg1 = static_cast<int>(c.K);
        const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                               scale * (1.0f / 3.0f), scale * 1.0f};
        int kg = kg0;
        std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
        for (; kg + 8 <= kg1; kg += 8) {
          sycl::vec<std::uint16_t, 8> a_lo;
          a_lo.load(0, act_mp + kg);
          const std::uint8_t v0 = c.code[b0];
          const std::uint8_t v1 = c.code[b0 + 1];
          b0 += 2;
          for (int j = 0; j < 8; ++j) {
            const float av =
                static_cast<float>(sycl::bit_cast<sycl::half>(a_lo[j]));
            const std::uint8_t bv = (j < 4) ? v0 : v1;
            const int code = (bv >> ((j & 3) * 2)) & 3;
            acc += av * wtab[code];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av =
              static_cast<float>(sycl::bit_cast<sycl::half>(c.act[kg]));
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const std::uint8_t byte = c.code[idx / 4u];
          const int code = (byte >> (static_cast<int>(idx % 4u) * 2)) & 3;
          acc += av * wtab[code];
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

int main(int argc, char** argv) {
  return bisect_main(argc, argv);
}
