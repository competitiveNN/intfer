// examples/roofline_probe_bisect4 -- bisect-4: differential between the
// bisect-2 VERBATIM body and a faithful transcription of the roofline_probe
// int2_variant<0> (DEQ=table) arm. Both share the same shape
// (MLP_GATE 8704x5120 g16, MLP_DOWN 2560x17408 g16) and run through
// probe_common.hpp's bisect_main, so the within-run RATIOS are trustworthy on
// this contended box.
//
// WHY. The clone in examples/roofline_probe measured the shipped-ish INT2 code
// at 262 GB/s while the bisect-2 VERBATIM (transcribed shipped source, same
// chunk loop, `wtab[code]`, bit_cast activation) measured 32 GB/s on the SAME
// box, SAME shapes, SAME instrument, fresh builds. Every verified single-change
// arm in the ladder (VERB+HALF, VERB+TAIL, VERB+MP) is a wash or a small win,
// so this differential isolates which of the remaining cosmetic differences
// (bit_cast vs half_to_float at group+tail, wtab vs canon2 at the tail, 4-byte
// vs 2-byte code loads, the probe's 3-ternary bv selector) actually closes the
// gap. No claim about a mechanism is made; this is a map of where the ~7x sits.
//
// ARMS:
//   VERBATIM       bisect-2 VERBATIM control.
//   VERB+bitcast   VERBATIM with group scale + tail activation converted by
//                  bit_cast<half> and the tail changed to av * wtab[code].
//   VERB+bitcast+4 VERB+bitcast with the probe's 4-byte code loads (v0..v3).
//   VERB+bitcast+4+vsel  + probe's 3-ternary bv selector (a_lo[a_hi for CC=16]).
//   CLONE          full transcription of int2_variant<0> (DEQ=table).

#include <sycl/sycl.hpp>

// Ctx is forward-declared here so the arm function signatures are type-complete
// before probe_common.hpp's bisect_main compiles at the include below; Ctx is
// defined by probe_common.hpp itself, and the arm bodies are defined afterwards,
// where the type is complete. BISECT_NAME / ARMS_COUNT / ARM_NAMES / ARM_SWITCH
// are defined before the include so bisect_main sees them.
struct Ctx;
sycl::event arm_verbatim(sycl::queue& q, const Ctx& c);
sycl::event arm_verb_bitcast(sycl::queue& q, const Ctx& c);
sycl::event arm_verb_bitcast4(sycl::queue& q, const Ctx& c);
sycl::event arm_verb_bitcast4vsel(sycl::queue& q, const Ctx& c);
sycl::event arm_clone(sycl::queue& q, const Ctx& c);

#define BISECT_NAME "bisect4"
#define ARMS_COUNT 5
#define ARM_NAMES {"VERBATIM", "VERB+bitcast", "VERB+bitcast+4", \
                   "VERB+bitcast+4+vsel", "CLONE"}
#define ARM_SWITCH \
  case 0: e = arm_verbatim(q, ctx); break; \
  case 1: e = arm_verb_bitcast(q, ctx); break; \
  case 2: e = arm_verb_bitcast4(q, ctx); break; \
  case 3: e = arm_verb_bitcast4vsel(q, ctx); break; \
  case 4: e = arm_clone(q, ctx); break; \

#include "probe_common.hpp"

// ---- differential ladder ----------------------------------------------------

// VERBATIM: bisect-2 VERBATIM control.
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
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw), static_cast<int>(c.K));
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
            const float av = static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? b0v : b1v;
            acc += av * wtab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = half_to_float(c.act[kg]);
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const int code = static_cast<int>((c.code[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * scale * canon2(code);
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

// VERB+bitcast: group scale + tail activation via bit_cast<half>; tail uses
// av * wtab[code]. Both changes taken from the roofline_probe clone.
sycl::event arm_verb_bitcast(sycl::queue& q, const Ctx& c) {
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
        const float scale = static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g]));
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw), static_cast<int>(c.K));
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
            const float av = static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? b0v : b1v;
            acc += av * wtab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = static_cast<float>(sycl::bit_cast<sycl::half>(c.act[kg]));
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const int code = static_cast<int>((c.code[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * wtab[code];
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

// VERB+bitcast+4: same, but load the probe's 4 code bytes (v0..v3) per chunk.
sycl::event arm_verb_bitcast4(sycl::queue& q, const Ctx& c) {
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
        const float scale = static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g]));
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw), static_cast<int>(c.K));
        const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                               scale * (1.0f / 3.0f), scale * 1.0f};
        int kg = kg0;
        std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
        for (; kg + kChunk <= kg1; kg += kChunk) {
          sycl::vec<std::uint16_t, 8> a16;
          a16.load(0, act_mp + kg);
          const std::uint8_t v0 = c.code[b0];
          const std::uint8_t v1 = c.code[b0 + 1];
          const std::uint8_t v2 = c.code[b0 + 2];
          const std::uint8_t v3 = c.code[b0 + 3];
          b0 += 2;
          for (int j = 0; j < kChunk; ++j) {
            const float av = static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? v0 : v1;
            acc += av * wtab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = static_cast<float>(sycl::bit_cast<sycl::half>(c.act[kg]));
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const int code = static_cast<int>((c.code[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * wtab[code];
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

// VERB+bitcast+4+vsel: same, but with the probe's 3-ternary bv selector that
// would select a_hi for CC=16 (a_lo only for CC=8).
sycl::event arm_verb_bitcast4vsel(sycl::queue& q, const Ctx& c) {
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
        const float scale = static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g]));
        const int kg0 = g * static_cast<int>(c.Gw);
        const int kg1 = std::min(kg0 + static_cast<int>(c.Gw), static_cast<int>(c.K));
        const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                               scale * (1.0f / 3.0f), scale * 1.0f};
        int kg = kg0;
        std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
        for (; kg + kChunk <= kg1; kg += kChunk) {
          sycl::vec<std::uint16_t, 8> a16;
          a16.load(0, act_mp + kg);
          const std::uint8_t v0 = c.code[b0];
          const std::uint8_t v1 = c.code[b0 + 1];
          const std::uint8_t v2 = c.code[b0 + 2];
          const std::uint8_t v3 = c.code[b0 + 3];
          b0 += 2;
          for (int j = 0; j < kChunk; ++j) {
            const float av = static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
            const std::uint8_t bv = (j < 4) ? v0 : (j < 8) ? v1 : v2;
            acc += av * wtab[(bv >> ((j & 3) * 2)) & 3];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = static_cast<float>(sycl::bit_cast<sycl::half>(c.act[kg]));
          const std::size_t idx = row_base + static_cast<std::size_t>(kg);
          const int code = static_cast<int>((c.code[idx / 4] >> ((idx % 4) * 2)) & 3);
          acc += av * wtab[code];
        }
      }
      c.out[ng] = std::bit_cast<std::uint16_t>(sycl::half(acc));
    });
  });
}

// CLONE: VERBATIM transcription of examples/roofline_probe's int2_variant<0>
// (DEQ=table) body. The roofline clone uses `(CC == 16 && j >= 8) ? a_hi[j-8] :
// a_lo[j]`, i.e. for CC=8 it reads a_lo[0..7] in full; my earlier CLONE
// (a_lo[a_hi[j-4]] for j=4..7) read UNINITIALIZED private memory for half of
// every 8-code chunk and is therefore a DIFFERENT kernel. This version matches
// the roofline arm bit-for-bit except for field names (alpha is folded into
// wtab, so no multiply, as in the shipped path).
sycl::event arm_clone(sycl::queue& q, const Ctx& c) {
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
        const float scale = static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g]));
        const int kg0 = g * static_cast<int>(c.Gw);
        int kg1 = kg0 + static_cast<int>(c.Gw);
        if (kg1 > static_cast<int>(c.K)) kg1 = static_cast<int>(c.K);
        const float w_n1 = scale * -1.0f;
        const float w_n3 = scale * (-1.0f / 3.0f);
        const float w_p3 = scale * (1.0f / 3.0f);
        const float w_p1 = scale * 1.0f;
        const float wtab[4] = {w_n1, w_n3, w_p3, w_p1};
        int kg = kg0;
        std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
        for (; kg + 8 <= kg1; kg += 8) {
          sycl::vec<std::uint16_t, 8> a_lo, a_hi;
          a_lo.load(0, act_mp + kg);
          const std::uint8_t v0 = c.code[b0];
          const std::uint8_t v1 = c.code[b0 + 1];
          const std::uint8_t v2 = c.code[b0 + 2];
          const std::uint8_t v3 = c.code[b0 + 3];
          b0 += 2;
          for (int j = 0; j < 8; ++j) {
            // CC=8 here: same as the roofline clone's `(CC == 16 && j >= 8) ?
            // a_hi[j - 8] : a_lo[j]` — a_lo[j] alone is correct.
            const std::uint16_t abits = a_lo[j];
            const float av = static_cast<float>(sycl::bit_cast<sycl::half>(abits));
            const std::uint8_t bv =
                (j < 4) ? v0 : (j < 8) ? v1 : (j < 12) ? v2 : v3;
            const int code = (bv >> ((j & 3) * 2)) & 3;
            acc += av * wtab[code];
          }
        }
        for (; kg < kg1; ++kg) {
          const float av = static_cast<float>(
              sycl::bit_cast<sycl::half>(c.act[kg]));
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
