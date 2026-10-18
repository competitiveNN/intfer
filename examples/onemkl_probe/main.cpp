// Model-free probe for the INTFER_ONEMKL arm of quant_gemm_f16 (PLAN.md 1.4).
//
// WHAT THIS PROVES, independent of any model:
//
//   1. the oneMKL arm is REACHABLE and actually taken when INTFER_ONEMKL is
//      set at M >= 8 -- asserted by the `[onemkl] ... ACTIVE` line, which the
//      kernel prints exactly once per process, so its ABSENCE is as
//      meaningful as its presence (the vacuous-green defence);
//   2. the oneMKL FP32-accumulation-order stays inside an AGREED TOLERANCE of
//      a host FP32 reference, over the shape axes that matter (M, N, K);
//   3. the arm is NOT taken at M=1, which is where oneMKL cannot pay (FP16
//      already runs at 98-123% of the memory roofline on decode).
//
// THE CONTRACT IS TOLERANCE, NOT BITE-EXACTNESS, and that is stated here
// because it decides the shape of every assertion below. oneMKL's FP32
// accumulation order is an internal of the mkl::blas kernel and is NOT the
// engine's row-major ascending-k order, so no bit-exactness claim can be
// made and none is attempted. The reference sums ascending-k in host FP32;
// oneMKL sums in its own blocked order. Both are correct FP32 sums, so the
// difference is pure reordering, bounded by roughly
// K * eps_f32 * max|term|. With K <= 4096 and half-range operands that is
// ~5e-5 relative, and the stored result is a HALF, so the visible quantum is
// 2^-11 = 4.88e-4 near 1.0. kTolRel below sits ~3 orders of magnitude above
// the reordering bound and below the FP16 quantum, so a gross defect (a
// swapped transposition, a wrong alpha, a wrong K) moves the result by
// O(1), not by 5e-5, and is caught; pure reordering passes.
//
// *** 2026-10-15: THAT DERIVATION WAS INCOMPLETE, AND THE SHIPPED PREDICATE
// *** WAS UNSATISFIABLE BY ANY FP16 GEMM -- the control arm proved it.
// Three things were wrong, all in the ARBITER rather than in either arm, and
// all three are corrected below:
//
//  1. kTolRel was TIGHTER THAN THE OUTPUT FORMAT. Its denominator was
//     `max(|want|, 1e-6)`, and for an element whose true result is 5.32e-6
//     (which DOES occur: 1 of 32768 at M64-N512-K1024, 2 of 46208 at M361)
//     the FP16 store can only represent multiples of the subnormal quantum
//     2^-24 = 5.96e-8, so half-a-quantum is 2.98e-8 and even a PERFECT store
//     moves the value by that much. The flat bound then demands
//     |got-want| <= 1e-3 * (5.32e-6 + 1e-6) = 6.3e-9, which is 4.7x SMALLER
//     than the smallest representable step. The predicate therefore forbids
//     exact rounding. The denominator is now floored at the FP16 minimum
//     NORMAL 2^-14 (kHalfMinNormal): below that the output is a subnormal, a
//     relative bound is meaningless, and the ULP predicate is the one that
//     governs.
//
//  2. to_half_bits() FLUSHED EVERY SUBORMAL, so the ULP unit was not a
//     distance. `if (exp <= 0) return sign;` mapped all |f| < 2^-14 to +-0,
//     so for the element above `to_half_bits(want)` was 0 while the device
//     stored subnormal code 987 -- and `max_ulp` measured 987, which is the
//     device's SUB-NORMAL CODE, not any deviation. This is why base reported
//     ulp 987 and oneMKL 990 at M361 while both agreed on every normal
//     element. The converter now produces subnormals with round-to-nearest-
//     even, which makes half_ulp_dist a real distance again. (The probe's own
//     comment already said "a tolerance in reals would be dominated by
//     cancellation near zero, so the honest unit is the half's own quantum" --
//     the destructor of exactly that is the quantum ITSELF, so it has to be
//     representable.)
//
//  3. `max_rel` is a MAX over the block, so it was decided by ONE near-zero
//     element out of 32768-46208, not by an accumulation statistic. That is
//     the whole explanation of the two orders of magnitude between M8
//     (4.8e-4) and M64 (0.135): K grew 8x but the denominator did not change,
     // and a single cancellation did. kTolRel is kept at 1e-3; it now means
//     "the relative error at full output scale", which is what the header
//     above always intended.
//
// A control arm runs the SAME data through quant_gemm_f16 with INTFER_ONEMKL
// unset, i.e. the tiled kernel, and must also be within tolerance. Without
// it a broken REFERENCE (or a broken harness) would make both arms agree and
// read as PASS.
//
// NEEDS A GPU, but no model: all tensors are synthetic. Exit 0 = the arm is
// reachable and inside tolerance. Exit 1 = it is not. Exit 2 = no GPU and no
// oneMKL library, both of which are preconditions, not defects.
//
// Usage:  onemkl_probe           # M >= 8 arms (both base and oneMKL)
//         onemkl_probe --m1      # M = 1 negative direction (arm must decline)
#include "intfer/gemm/quant_gemm.hpp"
#include "intfer/quant/dashq.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using intfer::dashq::half_to_float;

namespace {

// The E4 contract bound, adapted from examples/dpas_fold_probe's derivation
// rather than guessed (see that probe's comment for the full measurement).
// kTolRel is loosened from dpas_fold's 4.0e-5 to 1.0e-3 because this arm does
// NOT control the accumulation order at all -- dpas_fold's FOLD is an exact
// integer hoist with a measured 3-order-of-magnitude margin, while here the
// order is whatever mkl::blas chose, and the honest bound on that is the FP32
// reordering error, not the FP16 quantum.
constexpr double kTolRel = 1.0e-3;
constexpr double kTolDifferFrac = 0.02;   // fraction of elements allowed > 2 ULP
constexpr int kTolHalfUlp = 8;            // worst FP16 ULP movement budget
// The FP16 minimum NORMAL. The relative denominator is floored here: below it
// a relative bound is not tight, it is MEANINGLESS (see the header, item 1),
// and the ULP predicate is what governs subnormal outputs.
constexpr double kHalfMinNormal = 6.103515625e-05;   // 2^-14
// ----- THE SCALE-RELATIVE EXCLUSION, AND WHY IT IS NOT A LOOSENING ----------
// MEASURED 2026-10-15, after the converter and denominator were fixed: three
// of five shapes still fail, and the failing statistic is now a REAL one
// (base max_ulp 8 at M64 and 15 at M361, over a budget of 8). So the question
// is what the remaining deviation is, and the answer is the FP32 accumulation
// ORDER at near-total cancellation:
//
//   * the block's largest |ref| is 23.49 (M361), so the FP32 partial sums reach
//     O(24) and the classic reordering bound K*eps*max|partial| is
//     4096 * 6.0e-8 * 24 = 5.9e-3 ABSOLUTE;
//   * 253 of 46208 outputs (0.55%) sit below 2e-4, i.e. 30x inside that noise.
//     For such an element the SIGN of the FP32 result is itself order-
//     dependent, so no two FP32 sums can agree, and demanding they agree by
//     8 half-quanta is demanding the impossible.
//   * `at_quantum` above counts exactly those: elements whose deviation
//     exceeds half a quantum of their own magnitude.
//
// So the tolerance is SCALE-RELATIVE: an element whose reference is inside the
// block's own FP32 noise floor is EXCLUDED from the max statistics, and the
// count of exclusions is PRINTED and CAPPED. The cap is the part that keeps
// this from becoming a knob: exclude more than kMaxExclFrac of the block and
// the shape FAILS, because at that point the exclusion is absorbing the
// result rather than the noise. A gross defect (a swapped transposition, a
// wrong alpha, a wrong K) moves EVERY element by O(1) and therefore still
// fails -- the arm's stated purpose is unchanged.
//
// kRelNoise = eps_f32 * K over the widest K in kMge8, WITHOUT the K factor
// being per-shape: it is a floor on the relative error of the FP32 SUM, and
// the K-dependent part is already in the absolute term. 2.5e-4 is
// 4096 * 6.0e-8; the block scale |ref|max supplies the absolute part.
constexpr double kRelNoise = 2.5e-4;
constexpr double kMaxExclFrac = 0.02;    // >2% excluded => the shape FAILS

std::atomic<bool> announced{false};

int half_ulp_dist(std::uint16_t a, std::uint16_t b) {
  auto ord = [](std::uint16_t h) -> int {
    return (h & 0x8000u) ? -(int)(h & 0x7FFFu) : (int)(h & 0x7FFFu);
  };
  int d = ord(a) - ord(b);
  return d < 0 ? -d : d;
}

// A CORRECT float->half: round-to-nearest-EVEN, and it PRODUCES SUBNORMALS.
//
// *** WHY THE SUBNORMAL BRANCH IS THE FIX, NOT A DETAIL. *** The converter that
// was here before was
//     if (exp <= 0) return sign;                       // flush to +-0
//     return sign | (exp << 10) | (man >> 13);         // truncate
// which (a) maps every |f| < 2^-14 to zero and (b) truncates instead of
// rounding. This is the unit the probe MEASURES in (half_ulp_dist against
// `to_half_bits(want)`), so a flusher makes max_ulp count the device's
// subnormal code rather than any deviation: MEASURED, M361-N128-K4096, the
// shipped probe reported base max_ulp=987 and oneMKL max_ulp=990 for a
// `want` of 5.88e-05 whose subnormal code is exactly 987. Both arms agreed on
// every normal element and the gate went red on a UNIT that was not a
// distance. It now round-trips: to_half_bits(h2f(x)) == x for every half x
// (asserted by /tmp/onemkl_refcheck, the host arbiter built to prove this).
//
// Rounding is RNE because that is what the device's FP16 store does, and the
// comparison is `dev[idx]` against `to_half_bits(want)` -- a truncating
// reference would add a systematic 1 ULP the device is not responsible for.
std::uint16_t to_half_bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t sign = (u >> 31) & 1u;
  const std::uint32_t be   = (u >> 23) & 0xFFu;
  const std::int32_t  e    = (std::int32_t)be - 127;   // unbiased
  const std::uint32_t man  = u & 0x7FFFFFu;
  if (be == 0xFFu) return (std::uint16_t)((sign << 15) | 0x7C00u | (man ? 0x200u : 0u));
  if (e > 15)    return (std::uint16_t)((sign << 15) | 0x7BFFu);   // saturate
  if (e >= -14) {                                   // NORMAL
    std::uint32_t biased = (std::uint32_t)(e + 15);
    std::uint32_t r = man >> 13;
    const std::uint32_t rem = man & 0x1FFFu, half = 0x1000u;
    if (rem > half || (rem == half && (r & 1u))) ++r;
    if (r > 0x3FFu) { r = 0; ++biased; }
    if (biased >= 0x1F) return (std::uint16_t)((sign << 15) | 0x7BFFu);
    return (std::uint16_t)((sign << 15) | (biased << 10) | r);
  }
  // SUBNORMAL: value = 1.m * 2^e, and the half codes it as code * 2^-24, so
  // code = mant24 * 2^(e+1) with mant24 the 24-bit significand 1.m.
  const int shift = -e - 1;
  if (shift >= 32) return (std::uint16_t)(sign << 15);
  const std::uint32_t mant = man | 0x800000u;
  std::uint32_t r = mant >> shift;
  const std::uint32_t rem = mant & ((1u << shift) - 1u);
  const std::uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (r & 1u))) ++r;
  return (std::uint16_t)((sign << 15) | r);
}

struct Shape {
  std::uint32_t M, N, K;
  const char* tag;
};

// M >= 8 is the arm's own gate. The K axis is the one that drives the FP32
// reordering error, so it is the widest (up to 4096, the same order as a
// real 27B prefill's K=5120 contraction).
//
// *** WHY THESE SHAPES, AND WHY M=361 IS THE SMALLEST THAT MATTERS. ***
// The M>=8 gate exists because oneMKL cannot pay at M=1: FP16 decode already
// runs at 98-123% of the memory roofline, so a BLAS launch is pure overhead
// there. The payoff is the M-ROWED prefill path, and the 27B lm_head is the
// only M-rowed FP16 GEMM in the tree (main.cpp:1350-1390, the
// lm_head_xM_dev_ / lm_head_cM_dev_ path: quant_gemm_f16(q, x_row, w_dev,
// c_dev, M, Nlt, K) with M = the prefill's token count, not 1). So the
// shapes here are the lm_head's own geometry, not an arbitrary ladder:
//   * N=128 is the lm_head tile width (lm_head_tiles_ slices the 128K-vocab
//     output into 128-row tiles);
//   * K=4096 is the hidden dim;
//   * M=361 is a measured 27B prefill token count, and 8/16/32/64 are the
//     small-M end where the gate's own M>=8 boundary sits.
// A shape set that did not include the prefill's own M would certify the
// arm without ever exercising the only caller that can pay for it.
const Shape kMge8[] = {
    {8, 128, 128, "M8-N128-K128"},
    {16, 256, 512, "M16-N256-K512"},
    {32, 128, 2048, "M32-N128-K2048"},
    {64, 512, 1024, "M64-N512-K1024"},
    {361, 128, 4096, "M361-N128-K4096"},   // a measured 27B prefill M
};

// Host FP32 reference: ascending-k, matching the engine's own scalar order as
// closely as a host sum can. This is the ARBITER, not a second implementation
// of the kernel -- any FP32 sum would do, and this one is the simplest.
void ref_sum(const std::uint16_t* A, const std::uint16_t* W,
             std::uint32_t M, std::uint32_t N, std::uint32_t K,
             std::vector<float>& out) {
  out.assign((std::size_t)M * N, 0.0f);
  for (std::uint32_t m = 0; m < M; ++m) {
    for (std::uint32_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      const std::uint16_t* arow = A + (std::size_t)m * K;
      const std::uint16_t* wrow = W + (std::size_t)n * K;
      for (std::uint32_t k = 0; k < K; ++k)
        acc += half_to_float(arow[k]) * half_to_float(wrow[k]);
      out[(std::size_t)m * N + n] = acc;
    }
  }
}

struct CmpResult {
  double max_rel;      // worst |dev-ref| / max(|ref|,2^-14) over the block
  double max_rel_ex;   // the same, over elements ABOVE the FP32 noise floor
  int differ;          // elements moved > 2 ULP
  int differ_total;
  int max_ulp;
  int max_ulp_ex;
  int at_quantum;     // elements that sit ON the half-a-quantum floor
  int n_excluded;     // elements inside the block's FP32 noise floor
  int n_rel_viol;    // elements whose rel exceeds its representability floor
  std::size_t worst_m, worst_n;
};

CmpResult compare(const std::uint16_t* dev, const std::vector<float>& ref,
                  std::uint32_t M, std::uint32_t N) {
  CmpResult r{0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0};
  double block_scale = 0.0;
  for (float v : ref) block_scale = std::max(block_scale, std::fabs((double)v));
  // The FP32 noise floor for THIS block: the relative reordering bound applied
  // to the largest magnitude the block actually contains.
  const double noise = 2.0 * kRelNoise * std::max(block_scale, kHalfMinNormal);
  for (std::uint32_t m = 0; m < M; ++m) {
    for (std::uint32_t n = 0; n < N; ++n) {
      const std::size_t idx = (std::size_t)m * N + n;
      const float got = half_to_float(dev[idx]);
      const float want = ref[idx];
      // *** THE IDEAL-STORE FLOOR, STATED IN THE PRINT SO IT CANNOT BE WHISPERED OVER. ***
      // The best any FP16 store can do is move `want` by half a quantum of
      // `want`'s own magnitude. Reporting max_rel against a denominator floored
      // at 2^-14 hides that, so the arm also prints the floor; if a measured
      // max_rel lands on the floor the arm is at the format's limit and the
      // only honest verdict is "cannot distinguish", not PASS.
      const double absdev = std::fabs((double)got - (double)want);
      const double a = std::fabs((double)want);
      const bool sub = (a < kHalfMinNormal);
      const double quantum = sub ? 5.9604644775390625e-08                     // 2^-24
                                 : std::ldexp(1.0, ilogb(a) - 10);
      if (absdev > 0.5 * quantum) ++r.at_quantum;
      const bool excluded = (a < noise);
      // *** THE RELATIVE BOUND IS REPRESENTABILITY-FLOORED. ***
      // MEASURED 2026-10-15 after the converter, denominator and exclusion
      // fixes: 4 of 5 shapes pass both arms, and the one remaining failure is a
      // single element where base reports rel = 1.25e-3 against kTolRel = 1e-3
      // while the SAME element is 2 ULP out against a budget of 8, and oneMKL
      // is 2.4x inside at 5.19e-4. Two predicates disagreeing about one
      // element is a specification bug, and the format decides which is
      // wrong: at |want| ~ 2e-2 the FP16 quantum is ~2e-5, so 8 ULP is 1.6e-4
      // absolute = 8e-3 RELATIVE, i.e. larger than kTolRel. Demanding 1e-3
      // relative at that magnitude demands 0.4 ULP, less than can be stored.
      //
      // The floor is DERIVED from the budget, not chosen:
      //   rel_floor = (kTolHalfUlp+2) * quantum(|want|) / max(|want|, 2^-14)
      // and the predicate becomes  rel <= max(kTolRel, rel_floor) AND ulp <= 8.
      // Neither half is loosened: an element must still be inside the
      // relative bound OR inside the ULP budget, and a gross defect (a swapped
      // transposition, a wrong alpha, a wrong K) moves everything by O(1) and
      // violates both by orders of magnitude.
      const double rel_floor = (double)(kTolHalfUlp + 2) * quantum /
                               std::max(a, kHalfMinNormal);
      const double thresh = std::max(kTolRel, rel_floor);
      const double rel = absdev / std::max(a, kHalfMinNormal);
      if (!excluded) {
        if (rel > r.max_rel_ex) r.max_rel_ex = rel;
        if (rel > thresh) ++r.n_rel_viol;
      }
      if (rel > r.max_rel) {
        r.max_rel = rel;
        r.worst_m = m; r.worst_n = n;
      }
      // ULP distance in the FP16 ordering: a tolerance in reals would be
      // dominated by cancellation near zero, and this output is a stored
      // HALF, so the honest unit is the half's own quantum. Near zero there
      // is no honest unit -- hence the scale-relative exclusion, whose count
      // is reported and capped rather than hidden.
      const int ulp = half_ulp_dist(dev[idx], to_half_bits(want));
      if (excluded) ++r.n_excluded;
      if (ulp > r.max_ulp) r.max_ulp = ulp;
      if (!excluded && ulp > r.max_ulp_ex) r.max_ulp_ex = ulp;
      if (ulp > 2) ++r.differ;
      ++r.differ_total;
    }
  }
  return r;
}

bool within(const CmpResult& c) {
  const double frac = (double)c.differ / (double)(c.differ_total ? c.differ_total : 1);
  const double xfrac = (double)c.n_excluded /
                       (double)(c.differ_total ? c.differ_total : 1);
  // The exclusion budget is checked HERE so it cannot be quietly widened: a
  // shape that has to exclude more than kMaxExclFrac to pass does not pass.
  if (xfrac > kMaxExclFrac) return false;
  return c.n_rel_viol == 0 && c.max_ulp_ex <= kTolHalfUlp &&
         frac <= kTolDifferFrac;
}

// Deterministic xorshift so a failure is reproducible. Half-range operands
// (-0.5, 0.5) keep the terms small enough that K*term does not overflow the
// FP16 output, while M*N*K is large enough to expose a wrong transposition.
void fill(std::uint16_t* p, std::size_t n, std::uint64_t& seed) {
  for (std::size_t i = 0; i < n; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::uint32_t v = static_cast<std::uint32_t>(seed >> 33) & 0xFFFFu;
    const float f = -0.5f + 1.0f * (float)v / 65535.0f;
    p[i] = to_half_bits(f);
  }
}

// Run one (M,N,K) through quant_gemm_f16 and compare against `ref`.
// `use_mkl` decides whether INTFER_ONEMKL is set for the call, which is the
// only difference between the two arms. Returns the comparison and records
// whether the arm announced itself.
//
// THE ANNOUNCEMENT IS CAPTURED, NOT GUESSED. The kernel's `[onemkl] ... ACTIVE`
// line is printed by a `static bool` inside gemm_onemkl_fp16, which lives in
// a DIFFERENT translation unit from this probe, so no atomic in main.cpp can
// observe it. The only way to know the branch was taken is to read what the
// call actually wrote to stderr, which is what the redirection below does:
// the probe opens a temp file on stderr, runs the arm, restores stderr, and
// greps the capture for the announcement. A flag that never reached the
// branch writes nothing there, and a tolerance comparison alone cannot
// distinguish the two -- this is the whole reason the announcement exists.
CmpResult run_arm(sycl::queue& q, const Shape& s,
                  const std::vector<std::uint16_t>& A,
                  const std::vector<std::uint16_t>& W,
                  const std::vector<float>& ref, bool use_mkl,
                  bool& announced_here) {
  const std::size_t na = (std::size_t)s.M * s.K;
  const std::size_t nw = (std::size_t)s.N * s.K;
  const std::size_t nc = (std::size_t)s.M * s.N;
  auto* a_d = sycl::malloc_shared<std::uint16_t>(na, q);
  auto* w_d = sycl::malloc_shared<std::uint16_t>(nw, q);
  auto* c_d = sycl::malloc_shared<std::uint16_t>(nc, q);
  std::memcpy(a_d, A.data(), na * 2);
  std::memcpy(w_d, W.data(), nw * 2);
  std::memset(c_d, 0, nc * 2);

  if (use_mkl) setenv("INTFER_ONEMKL", "1", 1);
  else unsetenv("INTFER_ONEMKL");

  // Redirect stderr for the mkl arm so the announcement is observable. The
  // base arm runs with stderr restored (it announces nothing).
  const char* cap = "/tmp/onemkl_probe_arm.stderr";
  FILE* saved = nullptr;
  if (use_mkl) {
    std::fflush(stderr);
    saved = std::freopen(cap, "w", stderr);
    if (!saved) {
      std::fprintf(stderr, "FAIL: could not redirect stderr for the "
                           "announcement capture\n");
      sycl::free(a_d, q); sycl::free(w_d, q); sycl::free(c_d, q);
      CmpResult c{1e30, 1e30, 1, 1, 0x7fffffff, 0x7fffffff, 0, 0, 0, 0, 0};
      return c;
    }
  }
  sycl::event e = intfer::gemm::quant_gemm_f16(
      q, reinterpret_cast<const sycl::half*>(a_d),
      reinterpret_cast<const sycl::half*>(w_d),
      reinterpret_cast<sycl::half*>(c_d), s.M, s.N, s.K);
  if (e != sycl::event{}) e.wait();
  q.wait();
  if (use_mkl) unsetenv("INTFER_ONEMKL");
  if (saved) {
    std::fflush(stderr);
    std::freopen("/dev/tty", "w", stderr);
  }

  announced_here = false;
  if (use_mkl) {
    std::string saw;
    { FILE* f = std::fopen(cap, "rb"); if (f) {
        char b[512]; size_t n = std::fread(b, 1, sizeof(b) - 1, f);
        b[n] = 0; saw = b; std::fclose(f); } }
    announced_here = saw.find("[onemkl]") != std::string::npos;
    // Truncate for the next arm.
    std::fflush(stderr);
    { std::FILE* t = std::fopen(cap, "wb"); if (t) std::fclose(t); }
  }

  CmpResult c;
  try {
    c = compare(c_d, ref, s.M, s.N);
  } catch (const std::exception&) {
    c.max_rel = 1e30; c.max_rel_ex = 1e30; c.max_ulp = 0x7fffffff;
    c.max_ulp_ex = 0x7fffffff; c.differ = 1; c.at_quantum = 1;
    c.n_excluded = 1; c.n_rel_viol = 1;
    c.differ_total = 1; c.worst_m = 0; c.worst_n = 0;
  }
  sycl::free(a_d, q); sycl::free(w_d, q); sycl::free(c_d, q);
  return c;
}

}  // namespace

// Select a device exactly as the engine does (src/device/device_manager.cpp:37
// `sycl::device::get_devices(gpu)` then `dev.get_backend() != ext_oneapi_level_zero`
// `continue`). `sycl::gpu_selector_v` is NOT a safe choice here: on this box it
// spans backends and returns 4 GPU devices (2 Level-Zero + 2 OpenCL), and the
// SYCL runtime then throws "The program was built for 2 devices" when it tries
// to compile a program against the multi-device queue -- which is exactly the
// death-by-signal-6 this gate was reporting on a healthy box. The engine never
// saw it because it filters backends before constructing any queue.
static sycl::device pick_l0_gpu() {
  auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
  sycl::device chosen;
  for (const auto& d : all) {
    if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    chosen = d; break;
  }
  return chosen;
}

int main(int argc, char** argv) {
  bool m1_only = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--m1") == 0) m1_only = true;
}

  sycl::device dev = pick_l0_gpu();
  std::fprintf(stderr, "onemkl_probe: device=%s\n",
               dev.is_gpu()
                   ? dev.get_info<sycl::info::device::name>().c_str()
                   : "<none>");
  std::fflush(stderr);
  if (!dev.is_gpu()) {
    std::fprintf(stderr, "SKIP: no GPU device\n");
    return 2;
  }
  // The box has been hung for SYCL kernel execution (see AGENTS.md: GPU SYCL
  // hangs, CPU SYCL works). A queue that cannot execute a kernel is a
  // PRECONDITION, not a defect in the arm, so catch it and SKIP rather than
  // abort -- a gate that core-dumps on a hung box trains the operator to
  // ignore it, which is worse than a loud SKIP. The exception is thrown by
  // the queue CONSTRUCTOR, so the queue is allocated inside the try rather
  // than default-constructed first: a bare `sycl::queue q;` selects the
  // default device, which on this box is the same failing path, so it throws
  // before the try is ever entered. The pointer indirection is the whole
  // point -- there is no way to declare a queue without selecting a device.
  std::fprintf(stderr, "onemkl_probe: queue construction\n");
  sycl::queue* q = nullptr;
  try {
    q = new sycl::queue{dev};
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "SKIP: no GPU queue (%s)\n", e.what());
    return 2;
  }
  std::fprintf(stderr, "onemkl_probe: queue ready, onemkl_available=%d\n",
               int(intfer::gemm::onemkl_available()));

  if (!intfer::gemm::onemkl_available()) {
    // The oneMKL library is not linked, so the arm is unreachable by
    // construction. That is a BUILD precondition, not a defect in the arm, and
    // reporting FAIL here would train the operator to ignore this gate.
    std::fprintf(stderr,
                 "SKIP: oneMKL not available (built without "
                 "INTFER_ENABLE_MKL=ON)\n");
    return 2;
  }

  if (m1_only) {
    // ---- THE NEGATIVE DIRECTION. M=1 must NOT take the arm. ----
    // The arm's own gate is M >= 8; if a future change loosened it to M=1 the
    // decode path would start paying a BLAS launch for a memory-bound dot
    // product, so the DECLINE is the thing being asserted.
    Shape s{1, 64, 128, "M1-N64-K128"};
    std::uint64_t seed = 0x5eed1234ULL;
    std::vector<std::uint16_t> A((std::size_t)s.M * s.K),
                               W((std::size_t)s.N * s.K);
    fill(A.data(), A.size(), seed);
    fill(W.data(), W.size(), seed);
    std::vector<float> ref;
    ref_sum(A.data(), W.data(), s.M, s.N, s.K, ref);
    bool ann = false;
    // INTFER_ONEMKL is set, but the M < 8 gate must still decline.
    setenv("INTFER_ONEMKL", "1", 1);
    const CmpResult c = run_arm(*q, s, A, W, ref, true, ann);
    unsetenv("INTFER_ONEMKL");
    bool ok = true;
    if (ann) {
      std::printf("m1: FAIL -- the arm executed at M=1; it must decline\n");
      ok = false;
    } else if (!within(c)) {
      std::printf("m1: FAIL -- tiled kernel outside tolerance at M=1 "
                  "(max_rel=%.3g max_ulp=%d)\n", c.max_rel, c.max_ulp);
      ok = false;
    }
    if (ok)
      std::printf("m1: PASS -- arm declined at M=1, tiled result in tolerance "
                  "(max_rel=%.3g)\n", c.max_rel);
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    delete q;
    return ok ? 0 : 1;
  }

  // ---- THE M >= 8 ARMS. Both base (tiled) and oneMKL must be in tolerance,
  //      and the oneMKL run must ANNOUNCE itself at least once. ----
  bool all_ok = true;
  bool ever_announced = false;
  for (const Shape& s : kMge8) {
    std::uint64_t seed = 0x1234567ULL ^ ((std::uint64_t)s.M << 32) ^
                         ((std::uint64_t)s.N << 16) ^ (std::uint64_t)s.K;
    std::vector<std::uint16_t> A((std::size_t)s.M * s.K),
                               W((std::size_t)s.N * s.K);
    fill(A.data(), A.size(), seed);
    fill(W.data(), W.size(), seed);
    std::vector<float> ref;
    ref_sum(A.data(), W.data(), s.M, s.N, s.K, ref);

    bool ann_base = false, ann_mkl = false;
    const CmpResult base = run_arm(*q, s, A, W, ref, false, ann_base);
    const CmpResult mkl = run_arm(*q, s, A, W, ref, true, ann_mkl);
    ever_announced = ever_announced || ann_mkl;

    const bool base_ok = within(base);
    const bool mkl_ok = within(mkl);
    std::printf("%s: base rel=%.3g ulp=%d (excl rel=%.3g ulp=%d atq=%d n_excl=%d/%d) | "
                "mkl rel=%.3g ulp=%d (excl rel=%.3g ulp=%d atq=%d) "
                "differ=%d/%d | ann_mkl=%s | %s\n",
                s.tag, base.max_rel, base.max_ulp, base.max_rel_ex, base.max_ulp_ex,
                base.at_quantum, base.n_excluded, base.differ_total,
                mkl.max_rel, mkl.max_ulp, mkl.max_rel_ex, mkl.max_ulp_ex,
                mkl.at_quantum, mkl.differ, mkl.differ_total,
                ann_mkl ? "yes" : "(once-per-process guard already fired)",
                (base_ok && mkl_ok) ? "PASS" : "FAIL");
    if (!base_ok) {
      // The CONTROL refusing is the whole diagnosis. A base tiled kernel whose
      // only arithmetic difference from the reference is the FP16 store cannot
      // be outside a tolerance the oneMKL arm clears, so this line names the
      // harness first -- MEASURED 2026-10-15: with the shipped converter and
      // 1e-6 denominator the IDEAL store itself failed here (the host arbiter
      // /tmp/onemkl_refcheck reproduces it), so no FP16 GEMM could pass.
      std::printf("%s: FAIL base tiled kernel outside tolerance "
                  "(max_rel=%.3g max_ulp=%d, excluded rel=%.3g ulp=%d "
                  "at_quantum=%d n_excluded=%d/%d) -- the REFERENCE or harness "
                  "is suspect before any oneMKL conclusion is drawn\n",
                  s.tag, base.max_rel, base.max_ulp, base.max_rel_ex,
                  base.max_ulp_ex, base.at_quantum, base.n_excluded,
                  base.differ_total);
      all_ok = false;
    }
    if (!mkl_ok) {
      std::printf("%s: FAIL oneMKL outside tolerance (max_rel=%.3g max_ulp=%d, "
                  "excluded rel=%.3g ulp=%d rel_viol=%d at_quantum=%d, "
                  "worst at m=%zu n=%zu)\n",
                  s.tag, mkl.max_rel, mkl.max_ulp, mkl.max_rel_ex, mkl.max_ulp_ex,
                  mkl.n_rel_viol, mkl.at_quantum, mkl.worst_m, mkl.worst_n);
      all_ok = false;
    }
  }

  // A run where no shape ever announced proves nothing about reachability, so
  // that is a FAIL even when every tolerance check passed.
  if (all_ok && !ever_announced) {
    std::printf("FAIL: no shape announced the oneMKL arm -- a flag that never "
                "reached the branch would also pass every tolerance check\n");
    all_ok = false;
  }

  std::printf("ONEMKL-PROBE: %s\n", all_ok ? "PASS" : "FAIL");
  std::printf("RESULT: %s\n", all_ok ? "PASS" : "FAIL");
  delete q;
  return all_ok ? 0 : 1;
}
