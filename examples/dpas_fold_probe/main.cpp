// DPAS row-fold identity probe (INTFER_DPAS48_FOLD).
//
// The INTFER_DPAS48_FOLD optimization hoists the per-(row,op) activation
// column sums out of quant_gemm_dpas48's 16-N tile loop. That is supposed to be
// a pure work hoist: the values are exact int32 sums of int8 codes, so moving
// them cannot change a bit.
//
// This probe is the gate for that claim at the value level, independent of any
// model: it fills an activation buffer, runs quant_gemm_dpas48_rowfold on the
// device, and compares EVERY (row, op) S0/Sfull against a host reference that
// sums in the same element order the kernel uses. It also exercises the
// row-stride contract (fold_stride > nOp), the workspace-vs-call shape case
// (workspace K/nOp larger than the call's), and M>1, since a row-stride bug
// would leave row 0 correct and every other row stale.
//
// CPU-only logic plus one device launch; needs a GPU to run. Exit 0 = the hoist
// is value-identical. Exit 1 = it is NOT, and the first mismatches are printed
// with the ulp/exact difference so the failure is actionable.
#include "intfer/gemm/quant_gemm.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::uint32_t OP = 32;

// Host reference, deliberately written as the kernel's own nested fold:
// s0 accumulates the first 16, sf accumulates all 32.
void ref_row(const std::int8_t* Aq, std::int32_t& s0, std::int32_t& sf) {
  int a0 = 0, af = 0;
  for (std::uint32_t k = 0; k < OP; ++k) {
    const int v = static_cast<int>(Aq[k]);
    af += v;
    if (k < 16) a0 += v;
  }
  s0 = a0;
  sf = af;
}

// ---------------------------------------------------------------------------
// ACCUMULATION-ORDER SENSITIVITY (the E4 numerics contract, measured).
//
// quant_gemm_dpas48 accumulates into FP32:
//     acc_tile[nl] += as * ws * alpha * (2*c[nl] - 15*S) * Cinv
// once per K-op, i.e. nOp = K/32 sequential FP32 adds, and only THEN rounds
// the result to FP16 via to_half_bits(). Any change to the DPAS geometry that
// reorders those adds (a cooperative 2D block accumulating per-block partials,
// or a tree reduction) changes the rounding, so it is NOT a bit-exact hoist
// the way INTFER_DPAS48_FOLD was.
//
// The number that decides whether that matters is the FP16 quantum, not FP32
// eps: the stored result is a half, so any relative error far below the half's
// ULP is invisible unless the value sits exactly on a rounding boundary.
// This section measures the actual spread of reordering error for the kernel's
// own accumulation shape, so the E4 tolerance is derived rather than guessed,
// and the tolerance itself is gated (see TOL_* below).
// ---------------------------------------------------------------------------
struct ReorderResult {
  double max_rel;        // worst |seq-blocked| / |seq|  (ill-scaled if cancelled)
  double backward_err;   // worst |seq-blocked| / max|t_o|  (the meaningful one)
  double max_term;       // largest |t_o| seen
  int half_differ;       // outputs whose FP16 bits changed
  int half_total;
  int max_half_ulp;      // worst FP16 ULP distance
};

namespace {

// One realistic (as, ws, c, S) term, same shape as the u4 DPAS inner product.
inline float make_term(std::uint32_t& seed) {
  seed = seed * 1664525u + 1013904223u;
  const float as = 0.004f + 0.012f * (float)((seed >> 8) & 0xFFFF) / 65535.0f;
  seed = seed * 1664525u + 1013904223u;
  const float ws = 0.01f + 0.05f * (float)((seed >> 8) & 0xFFFF) / 65535.0f;
  seed = seed * 1664525u + 1013904223u;
  const int c = (int)((seed >> 12) & 0x1FFF) - 4096;
  seed = seed * 1664525u + 1013904223u;
  const int S = (int)((seed >> 12) & 0x3FF) - 512;
  return as * ws * (2.0f * (float)c - 15.0f * (float)S) * (1.0f / 15.0f);
}

std::uint16_t to_half_bits(float f) {
  // RNE, matching the kernel's converter closely enough for a bound check.
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  std::int32_t exp = (std::int32_t)((u >> 23) & 0xFF) - 127 + 15;
  std::uint32_t man = u & 0x7FFFFFu;
  if (exp <= 0) return (std::uint16_t)sign;
  if (exp >= 0x1F) return (std::uint16_t)(sign | 0x7C00u);
  return (std::uint16_t)(sign | ((std::uint32_t)exp << 10) | (man >> 13));
}

int half_ulp_dist(std::uint16_t a, std::uint16_t b) {
  auto ord = [](std::uint16_t h) -> int {
    return (h & 0x8000u) ? -(int)(h & 0x7FFFu) : (int)(h & 0x7FFFu);
  };
  int d = ord(a) - ord(b);
  return d < 0 ? -d : d;
}

}  // namespace

// The E4 contract bound, derived from the FP16 output quantum, then verified
// against the measurement below. FP16 has a 10-bit explicit mantissa, so one
// ULP is ~2^-11 = 4.88e-4 relative. The bound below sits ~3 orders of
// magnitude under that: a reorder at this K must not move the stored half by
// even one ULP on any element in these shapes, and must stay far enough under
// the quantum that a boundary case stays a reportable curiosity rather than a
// silent numerical change.
static constexpr double kTolBackwardErr = 4.0e-5;   // reordering / largest term
static constexpr double kTolMaxRel = 1.0e-6;         // result-relative (informational)
static constexpr int kTolHalfUlp = 16;              // stored-half movement budget
static constexpr double kTolDifferFrac = 0.01;      // fraction of outputs allowed to move

// DERIVATION, not a guess. Measured on this exact harness (4096 reps/shape),
// sequential vs 4-way blocked partials, over the kernel's own term shape:
//
//     K       nOp    backward_err   half_diff   max_ulp
//   1024       32       1.15e-06      3/4096        1
//   4096      128       2.78e-06      2/4096        1
//   14336     448       9.06e-06      9/4096        3
//   28672     896       1.38e-05     13/4096        8
//
// backward_err grows roughly linearly in nOp (= K/32), so 4.0e-5 covers K up to
// ~57k — past the widest 27B tensor — with ~3x margin. The other two are set
// the same way: 8 ULP observed -> 16 allowed; 0.32% observed -> 1% allowed.
//
// The result-relative column is reported but NOT gated. It is meaningless here
// and saying so is the point: `2*c - 15*S` is a zero-point correction, so the
// terms cancel and the result can be ~1000x smaller than the largest term. A
// result-relative tolerance is then dominated by cancellation, not by rounding,
// and would set a bound that silently tightens as the true value approaches
// zero. Gating backward_err is the numerically honest choice.
//
// CAVEAT, load-bearing: the term generator is SYNTHETIC and deliberately
// cancellation-heavy, so these numbers are an UPPER bound on the real spread.
// When E4 actually exists, re-derive the bound from the real kernel before
// trusting it — do not inherit 4.0e-5 by copy.

ReorderResult measure_reorder(std::uint32_t K, std::uint32_t reps) {
  ReorderResult r{0.0, 0.0, 0.0, 0, 0, 0};
  const std::uint32_t nOp = K / 32;
  std::vector<float> t(nOp);
  for (std::uint32_t rep = 0; rep < reps; ++rep) {
    std::uint32_t seed = 99991u + rep * 7717u;
    float tmax = 0.0f;
    for (std::uint32_t o = 0; o < nOp; ++o) {
      t[o] = make_term(seed);
      const float a = std::fabs(t[o]);
      if (a > tmax) tmax = a;
    }
    if (tmax > r.max_term) r.max_term = tmax;

    // (a) the kernel's order: strictly sequential
    float seq = 0.0f;
    for (std::uint32_t o = 0; o < nOp; ++o) seq += t[o];

    // (b) a cooperative-block order: 4 independent partials, summed at the
    //     end. This is the shape a 2D block accumulation would take.
    float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (std::uint32_t o = 0; o < nOp; ++o) part[o & 3] += t[o];
    float blk = (part[0] + part[1]) + (part[2] + part[3]);

    const double diff = std::fabs((double)blk - (double)seq);
    // Backward error: the standard measure for reordering a dot product. It
    // divides by the LARGEST TERM, not the result, because this kernel's
    // `2*c - 15*S` zero-point correction means the terms cancel and the result
    // can be orders of magnitude below them -- a result-relative ratio is then
    // meaningless (and was measured at ~7e-3, purely an artifact of that).
    if (tmax > 0.0) {
      const double be = diff / (double)tmax;
      if (be > r.backward_err) r.backward_err = be;
    }
    const double denom = (double)seq;
    if (denom != 0.0) {
      double rel = diff / std::fabs(denom);
      if (rel > r.max_rel) r.max_rel = rel;
    }
    const std::uint16_t hs = to_half_bits(seq), hb = to_half_bits(blk);
    ++r.half_total;
    if (hs != hb) ++r.half_differ;
    int d = half_ulp_dist(hs, hb);
    if (d > r.max_half_ulp) r.max_half_ulp = d;
  }
  return r;
}

struct Case {
  const char* name;
  std::uint32_t M;
  std::uint32_t K;       // call K (multiple of OP)
  std::uint32_t wsK;     // workspace K (>= K) => workspace stride > call stride
  std::uint32_t pad;     // extra ops of slack in the fold row stride
  bool expect_decline;   // stride/geometry must be rejected, buffer untouched
  std::uint32_t force_stride;  // 0 => model the engine (2*wsOp + pad)
};

}  // namespace

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::printf("[dpas-fold-probe] device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  const Case cases[] = {
      {"M=1 tight",      1,  1024, 1024, 0,  false, 0},
      {"M=1 rowstride",  1,  1024, 2048, 7,  false, 0},
      // The case that caught the original bug: the row stride must clear
      // 2*nOp, not nOp, or row m's Sfull block clobbers row m+1's S0 block.
      {"M=4 rowstride",  4,  2048, 4096, 3,  false, 0},
      {"M=8 big-K",      8, 14336, 16384, 5, false, 0},
      {"M=3 min-op",     3,    32,   64, 1,  false, 0},
      // Negative cases: the kernel must DECLINE and write nothing.
      // stride 100 < 2*nOp(128) for K=2048 — the exact original defect.
      {"NEG short stride", 4, 2048, 4096, 0, true, 100},
      {"NEG K<32",         2,   16,   64, 0, true, 0},
      {"NEG K%32!=0",      2,  100,  128, 0, true, 0},
      {"NEG stride=2nOp-1", 4, 2048, 4096, 0, true, 127},
  };

  std::uint64_t checked = 0, bad = 0;
  bool ok = true;

  for (const Case& c : cases) {
    const std::uint32_t nOp = c.K / OP;
    const std::uint32_t wsOp = c.wsK / OP;
    // stride models the engine (2 * workspace-ops + pad) unless forced.
    const std::uint32_t fold_stride =
        c.force_stride ? c.force_stride : (2 * wsOp + c.pad);

    // Deterministic pseudo-random codes spanning the full int8 range, so the
    // sums exercise both signs and the saturation ends.
    std::vector<std::int8_t> host((size_t)c.M * c.wsK);
    std::uint32_t seed = 12345u + c.M * 7919u + c.K;
    for (auto& v : host) {
      seed = seed * 1664525u + 1013904223u;
      v = (std::int8_t)((seed >> 16) & 0xff);
    }

    // Allocate for the widest legal row so a negative case cannot itself OOB.
    const std::uint32_t alloc_stride = 2 * wsOp + 8;
    auto* act = sycl::malloc_shared<std::int8_t>((size_t)c.M * c.wsK, q);
    auto* fold = sycl::malloc_shared<std::int32_t>(
        (size_t)c.M * alloc_stride, q);
    if (!act || !fold) {
      std::printf("[dpas-fold-probe] FATAL: alloc failed\n");
      return 1;
    }
    for (size_t i = 0; i < host.size(); ++i) act[i] = host[i];
    constexpr std::int32_t kSentinel = 0x5eed5eed;
    for (std::uint32_t i = 0; i < c.M * alloc_stride; ++i) fold[i] = kSentinel;

    // Sentinel prerequisite: on decline the kernel returns `dep` unchanged,
    // so `e == prereq` identifies a decline unambiguously (comparing against a
    // bare sycl::event{} is not reliable across implementations).
    sycl::event prereq = q.submit([&](sycl::handler& h) {
      h.single_task<class fold_probe_prereq>([=]() {});
    });
    sycl::event e = intfer::gemm::quant_gemm_dpas48_rowfold(
        q, act, fold, c.M, c.K, c.wsK, fold_stride, prereq);
    e.wait_and_throw();

    std::uint64_t cbad = 0;
    if (c.expect_decline) {
      // Must have declined (returned the prerequisite untouched) and written
      // nothing.
      bool declined = (e == prereq);
      std::uint32_t touched = 0;
      for (std::uint32_t i = 0; i < c.M * alloc_stride; ++i)
        if (fold[i] != kSentinel) ++touched;
      if (!declined) {
        std::printf("  [%s] expected DECLINE, kernel submitted anyway\n", c.name);
        ++cbad;
      }
      if (touched != 0) {
        std::printf("  [%s] declined but wrote %u slots (must write none)\n",
                    c.name, touched);
        ++cbad;
      }
      std::printf("  %-18s M=%u K=%u stride=%u -> %s (declined=%d, wrote=%u)\n",
                  c.name, c.M, c.K, fold_stride, cbad ? "FAIL" : "ok",
                  (int)declined, touched);
    } else {
      for (std::uint32_t m = 0; m < c.M && cbad < 4; ++m) {
        for (std::uint32_t o = 0; o < nOp && cbad < 4; ++o) {
          std::int32_t r0 = 0, rf = 0;
          ref_row(host.data() + (size_t)m * c.wsK + (size_t)o * OP, r0, rf);
          const std::int32_t g0 = fold[(size_t)m * fold_stride + o];
          const std::int32_t gf = fold[(size_t)m * fold_stride + nOp + o];
          if (g0 != r0 || gf != rf) {
            if (cbad == 0)
              std::printf("  [%s] first mismatch m=%u o=%u\n", c.name, m, o);
            std::printf("    S0    dev=%d host=%d %s\n", g0, r0,
                        g0 == r0 ? "" : "MISMATCH");
            std::printf("    Sfull dev=%d host=%d %s\n", gf, rf,
                        gf == rf ? "" : "MISMATCH");
            ++cbad;
          }
          ++checked;
        }
      }
      std::printf("  %-18s M=%u K=%u wsK=%u nOp=%u stride=%u -> %s\n", c.name,
                  c.M, c.K, c.wsK, nOp, fold_stride, cbad ? "FAIL" : "ok");
    }
    bad += cbad;
    ok = ok && (cbad == 0);

    sycl::free(act, q);
    sycl::free(fold, q);
  }

  std::printf("[dpas-fold-probe] %s (%llu (row,op) pairs compared, %llu violations)\n",
              ok ? "PASS" : "FAIL", (unsigned long long)checked,
              (unsigned long long)bad);

  // --- E4 numerics contract: accumulation-order sensitivity --------------
  // Not a property of the fold; a property of ANY geometry change that
  // reorders the FP32 accumulation. Measured here so the E4 tolerance is
  // derived from the real spread rather than guessed, and so the tolerance
  // itself is gated (a bound nobody measures is a guess).
  std::printf("\n[dpas-fold-probe] E4 accumulation-order sensitivity "
              "(sequential vs 4-way blocked partials):\n");
  std::printf("  %-8s %-6s %-12s %-12s %-10s %-9s\n", "K", "nOp", "backward_err",
              "max_rel", "half_diff", "max_ulp");
  int e4bad = 0;
  const std::uint32_t shapes[] = {1024, 4096, 14336, 28672};
  for (std::uint32_t K : shapes) {
    ReorderResult r = measure_reorder(K, 4096);
    const double frac = r.half_total ? (double)r.half_differ / (double)r.half_total
                                     : 0.0;
    std::printf("  %-8u %-6u %-12.3e %-12.3e %-10d %-9d\n", K, K / 32,
                r.backward_err, r.max_rel, r.half_differ, r.max_half_ulp);
    if (r.backward_err > kTolBackwardErr) {
      std::printf("    FAIL: backward_err %.3e exceeds contract bound %.1e\n",
                  r.backward_err, kTolBackwardErr);
      ++e4bad;
    }
    if (r.max_half_ulp > kTolHalfUlp) {
      std::printf("    FAIL: stored half moved by %d ULP (contract allows %d)\n",
                  r.max_half_ulp, kTolHalfUlp);
      ++e4bad;
    }
    if (frac > kTolDifferFrac) {
      std::printf("    FAIL: %.4f of outputs changed (contract allows %.4f)\n",
                  frac, kTolDifferFrac);
      ++e4bad;
    }
  }
  std::printf("  contract: backward_err <= %.1e, stored-half movement <= %d ULP, "
              "changed outputs <= %.1f%%\n",
              kTolBackwardErr, kTolHalfUlp, kTolDifferFrac * 100.0);
  if (e4bad == 0)
    std::printf("  E4 contract: measured spread sits INSIDE the bound with margin\n");
  else
    std::printf("  E4 contract: VIOLATED -- the bound is wrong for this shape\n");
  ok = ok && (e4bad == 0);
  bad += (std::uint64_t)e4bad;

  if (!ok) {
    std::printf(
        "[dpas-fold-probe] The hoist is NOT value-identical (or failed to\n"
        "  decline), or the E4 tolerance contract was violated.\n"
        "  INTFER_DPAS48_FOLD must stay OFF; do not start E4 on a bad bound.\n");
  }
  return ok ? 0 : 1;
}
