// examples/roofline_probe_bisect1 -- bisect-1 of the M=1 INT2 GEMV probe ladder.
//
// WHY THIS IS A SEPARATE BINARY. The shipped INT2 kernel runs at 13.8-41.3 GB/s
// while a faithful clone of the same algorithm reaches 78-261 GB/s (examples/
// roofline_probe, interleaved, device event timestamps) -- a 5.6-6.4x gap whose
// mechanism is not located. Two candidate causes were then measured and FALSIFIED:
//
//   module size   the shipped source in a SMALL TU bought only 1.26x, against a
//                 pre-committed >=2x kill criterion. The module theory is dead.
//   runtime branch  making use_int2_tab a template parameter (so `if constexpr`
//                 deletes the branch and one loop body) measured 1.007x.
//
// This binary holds the FIRST of the two split ladders: the `table` control (the
// tidy rewrite that sets the fast end) vs `VERBATIM` (the SHIPPED source text,
// transcribed) -- both in a single tiny TU that cannot possibly trip the
// level-zero resource limit that blocked the full ladder. It establishes two
// things the sibling roofline_probe cannot:
//
//   * the VERBATIM baseline in a small TU (roofline_probe has `table` and `SHIPPED`
//     but not a verbatim transcription, so the 1.26x module move could not be
//     re-verified here);
//   * the within-run table/VERBATIM ratio, which is not load-contended because
//     both arms are interleaved in the same reps.
//
// THE ARMS, each differing from `VERBATIM` in exactly ZERO or ONE place:
//
//   table       a TIDY rewrite of the shipped inner loop. The fast end of the
//               ladder and the control both binaries share.
//   VERBATIM    the SHIPPED source text, transcribed. Every other arm is one
//               single change away from this one, which is what makes the ladder
//               attributable.
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
sycl::event arm_table(sycl::queue& q, const Ctx& c);

#define BISECT_NAME "bisect1"
#define ARMS_COUNT 2
#define ARM_NAMES {"table", "VERBATIM"}
#define ARM_SWITCH \
  case 0: e = arm_verbatim(q, ctx); break; \
  case 1: e = arm_table(q, ctx); break; \

#include "probe_common.hpp"

// ---- the INT2 ladder ---------------------------------------------------------

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

int main(int argc, char** argv) {
  return bisect_main(argc, argv);
}
