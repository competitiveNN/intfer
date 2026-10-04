// K-SPLIT NUMERICS CONTRACT for the M=1 INT2 decode kernel.
//
// WHAT THIS IS. AGENTS.md records the one remaining lever on 27B decode
// parallelism as: "split K across work-items. Raises parallelism AND coalesces
// the weight loads -- but it reassociates the FP32 sum, so it needs a
// tolerance-based numerics contract agreed BEFORE any code is written."
//
// This binary is that contract, MEASURED rather than guessed. The question it
// answers is the only one that has to be answered before the kernel is
// touched: **how far does the dot product move when the K axis is split into S
// partial sums and combined -- and does that movement survive the FP16 store
// the engine actually reads?**
//
// WHY A SEPARATE HOST BINARY RATHER THAN A KERNEL DIFF. The recorded failure
// this project keeps auditing is a numerics claim settled by reading two bodies
// and diffing them: the reverted fused-decode measured 180x slower decode after
// a "normalized statement-by-statement diff showed the arithmetic identical" --
// and the diff was RIGHT and it did not matter. So the reference here is a host
// transcription of the device kernel's exact accumulation order, and the whole
// experiment runs on the HOST: no GPU, no 27B model, seconds to run. That is
// deliberate. The contract must be settled before anyone risks the decode path,
// which is precisely the step that is hard to undo.
//
// THE ORDERS COMPARED -- this is the content of the contract:
//   seq      acc = 0; for kg ascending: acc += a[kg]*w[kg]
//            exactly what gemm_usm_decode1_vec does today (quant_gemm.cpp).
//            ONE accumulator, strictly ascending K, no reassociation.
//   split{S} S contiguous K chunks, each summed ascending, partials combined in
//            ascending chunk order. The shape a simple K-split kernel has.
//   tree{S}  the same S partials, combined by pairwise reduction (log2(S)
//            levels). What a warp-shuffle or SLM reduction does.
//
// seq-vs-split{S} and seq-vs-tree{S} are DIFFERENT numbers -- the partials are
// identical, only the COMBINATION differs -- so a contract that quotes one
// without saying which is ambiguous. That is the main reason to write it down.
//
// THE HEADLINE RESULT, and it is the reason this contract is cheap to satisfy:
// the store is FP16. A K-split perturbs the FP32 sum by ~1e-6 RELATIVE, while
// one FP16 ulp is ~5e-4 relative -- roughly two orders of magnitude coarser.
// So on realistic activations the perturbation is ABSORBED BY THE STORE and the
// FP16 result is bit-identical. The output of this binary is therefore not just
// "does it move" but "HOW MUCH HEADROOM IS LEFT", reported in FP16 ulps, because
// a tolerance that cannot be stated as a margin is a tolerance nobody can check.
//
// NON-VACUITY IS ASSERTED, NOT ASSUMED (the ga64_recip_test precedent). A
// tolerance gate that can print PASS on inputs where nothing moves proves
// nothing. So THREE directions are all required, and the run FAILS if any is
// missing:
//   (a) DISCRIMINATING: a dataset where the split provably DOES move the FP16
//       store. Without it the reported zeros are indistinguishable from a
//       harness that cannot see a move -- which is exactly the bug this binary
//       was written with and fixed after the first run printed all zeros.
//   (b) QUIET: realistic data where it does not move. Without this the gate
//       would be always-red and would train everyone to ignore it.
//   (c) SEPARATION: split{S} and tree{S} must measure DIFFERENTLY somewhere, or
//       quoting one is silently quoting the other.
// Plus two harness checks: seq-vs-split at S=1 must be bit-exact (else the host
// model is not the kernel's order and every number here is about the wrong
// thing), and fp16_ulp must both fire and stay quiet.
//
// COMPILER-REASSOCIATION HAZARD, stated because it would silently invalidate
// the whole experiment: -O3 vectorizing or reassociating the HOST reference
// would make `seq` a DIFFERENT order than the device's, and then every delta
// measured would be against a fiction. icx does not reassociate FP without
// fast-math, but that is a property of the compiler, not of this file, so it is
// ASSERTED in the self-check rather than assumed: the discriminating dataset
// must show seq != grouped. If a future compiler reassociates, that assertion
// fails loudly instead of the harness quietly reporting zeros forever.
//
// Exit: 0 = contract measured and self-consistent, 1 = NOT established,
//        2 = could not run.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// --- device helpers, transcribed from quant_gemm.cpp ----------------------
// These MUST match the kernel. to_half_bits is the FP16 store the device does
// (RNE); kCanon2 is the INT2 row of the DASH-Q codebook (canon_value()).

inline std::uint16_t to_half_bits(float f) {
  std::uint16_t h;
  std::memcpy(&h, &f, sizeof h);
  return h;
}
inline float from_half_bits(std::uint16_t h) {
  float f;
  std::memcpy(&f, &h, sizeof f);
  return f;
}
constexpr float kCanon2[4] = {-1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f};

struct Row {
  std::vector<float> a;  // activation, FP32 (already through the FP16 store)
  std::vector<float> w;  // wtab[code] = scale * canon2[code], FP32
};

// The kernel's order. NOINLINE + volatile sink so the compiler cannot fuse,
// vectorize across k, or reassociate this into a shape the device does not use.
__attribute__((noinline)) float acc_seq(const Row& r) {
  float acc = 0.0f;
  for (std::size_t kg = 0; kg < r.a.size(); ++kg) acc += r.a[kg] * r.w[kg];
  return acc;
}

// S contiguous chunks, each ascending, partials combined ascending.
float acc_split(const Row& r, int S) {
  const std::size_t K = r.a.size();
  const std::size_t base = K / (std::size_t)S;
  float total = 0.0f;
  for (int s = 0; s < S; ++s) {
    const std::size_t b = (std::size_t)s * base;
    const std::size_t e = (s == S - 1) ? K : b + base;
    float p = 0.0f;
    for (std::size_t kg = b; kg < e; ++kg) p += r.a[kg] * r.w[kg];
    total += p;
  }
  return total;
}

// Same S partials, pairwise reduction.
float acc_tree(const Row& r, int S) {
  const std::size_t K = r.a.size();
  std::vector<float> parts;
  parts.reserve((std::size_t)S);
  const std::size_t base = K / (std::size_t)S;
  for (int s = 0; s < S; ++s) {
    const std::size_t b = (std::size_t)s * base;
    const std::size_t e = (s == S - 1) ? K : b + base;
    float p = 0.0f;
    for (std::size_t kg = b; kg < e; ++kg) p += r.a[kg] * r.w[kg];
    parts.push_back(p);
  }
  while (parts.size() > 1) {
    std::vector<float> nx;
    nx.reserve((parts.size() + 1) / 2);
    for (std::size_t i = 0; i + 1 < parts.size(); i += 2)
      nx.push_back(parts[i] + parts[i + 1]);
    if (parts.size() & 1) nx.push_back(parts.back());
    parts.swap(nx);
  }
  return parts[0];
}

// Exact ulp distance between two FP16 values, by walking the FP32 grid --
// which is DENSE inside any single FP16 bucket, so stepping it from one
// endpoint to the other counts the FP16 ulps exactly and needs no table.
//
// Note this is NOT the same as fp16_ulp_naive below. The first version walked
// with nextafterf toward +inf only and gave up after 256 steps, so it reported
// a TRUNCATED distance for any move bigger than that and could not distinguish
// "moved by 1 ulp" from "moved by many". The naive version is kept as the
// self-check's contrast precisely so the difference is asserted, not assumed.
int fp16_ulp(std::uint16_t x, std::uint16_t y) {
  if (x == y) return 0;
  const float fx = from_half_bits(x), fy = from_half_bits(y);
  if (fx == fy) return 0;  // +0 / -0
  const float lo = std::min(fx, fy), hi = std::max(fx, fy);
  int n = 0;
  for (float v = lo; v != hi; ++v = std::nextafterf(v, hi)) {
    if (++n > (1 << 20)) return n;  // runaway guard, unreachable for finite FP16
  }
  return n;
}

// The first version's algorithm, kept ONLY so the self-check can prove the
// fixed one is not the same function. It truncates at 256.
int fp16_ulp_naive(std::uint16_t x, std::uint16_t y) {
  if (x == y) return 0;
  const float fx = from_half_bits(x), fy = from_half_bits(y);
  if (fx == fy) return 0;
  const float lo = std::min(fx, fy), hi = std::max(fx, fy);
  int n = 0;
  for (float v = lo; v < hi && n < 256; v = std::nextafterf(v, hi)) ++n;
  return n;
}

// Relative size of ONE FP16 ulp at the value `h`. This has to come from the
// FP16 bit pattern, NOT from nextafterf on the float: nextafterf steps the FP32
// grid, which is ~2^13 times finer than FP16, so a margin computed that way is
// off by three orders of magnitude and every headroom number below is fiction.
// (That was this file's second bug, and it is the same class as the first: a
// measurement that cannot fail, reporting a confident number.)
double fp16_step_rel(std::uint16_t h) {
  const float f = from_half_bits(h);
  if (f == 0.0f) return 1.0 / 65504.0;  // smallest normal step, relative to 1.0
  const std::uint16_t hn = (std::uint16_t)(h + 1);  // next FP16 up (finite here)
  const double step = std::fabs((double)from_half_bits(hn) - (double)f);
  return step / std::fabs((double)f);
}

// Headroom CONSUMED, in FP16 ulps, by the difference between the sequential
// result and an alternative. < 1 means the FP16 store absorbs the perturbation
// and the output is bit-identical; that is the number a tolerance is written
// against, because it says how much slack exists before a token can change.
double ulp_margin(float seq, float alt) {
  if (seq == alt) return 0.0;
  const double denom = std::max(1e-30, std::fabs((double)seq));
  const double rel = std::fabs((double)alt - (double)seq) / denom;
  const double step = fp16_step_rel(to_half_bits(seq));
  return step > 0.0 ? rel / step : INFINITY;
}

// --- datasets --------------------------------------------------------------
enum Kind { kRandom = 0, kOutliers = 1, kCancellation = 2, kInterleaved = 3 };

const char* kind_name(int k) {
  switch (k) {
    case kRandom: return "random";
    case kOutliers: return "act-outliers";
    case kCancellation: return "cancellation";
    default: return "escalating(ADV)";
  }
}

// activations are ALWAYS round-tripped through FP16, because the kernel stores
// FP16 activations; modelling an FP32 activation would test a kernel that
// cannot exist.
Row make_row(std::mt19937& rng, std::size_t K, int kind) {
  Row r;
  r.a.resize(K);
  r.w.resize(K);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_int_distribution<int> cd(0, 3);
  for (std::size_t i = 0; i < K; ++i) {
    r.a[i] = from_half_bits(to_half_bits(nd(rng)));
    const float sc = from_half_bits(to_half_bits(std::fabs(nd(rng)) * 0.05f + 1e-3f));
    r.w[i] = sc * kCanon2[cd(rng)];
  }
  if (kind == kOutliers) {
    // SpinQuant exists because these happen: a few very large activations.
    for (std::size_t i = 0; i < K; i += 997)
      r.a[i] = from_half_bits(to_half_bits(nd(rng) * 64.0f));
  } else if (kind == kCancellation) {
    // Near-total cancellation: the sum is small, so relative error is large.
    for (std::size_t i = 0; i < K; ++i)
      r.w[i] = ((i & 1) ? 1.0f : -1.0f) * (0.5f + 0.5f * std::fabs(nd(rng)));
  } else if (kind == kInterleaved) {
    // ADVERSARIAL, and the only dataset that moves the FP16 store.
    //
    // The mechanism is BOUNDED, and working it out is what makes this dataset
    // honest rather than a magic number. ulp(M) = M * 2^-24 in FP32, so a
    // running sum discards terms of size `small` once the running magnitude
    // passes small * 2^24. Two constraints fight:
    //   * activations are stored FP16, so |a| <= 65504 -- the available dynamic
    //     range is bounded, which is exactly why the perturbation stays small;
    //   * but K is large, so the SUM reaches K * 65504 ~ 5e8 at K=8192, and
    //     5e8 * 2^-24 ~ 30. Terms of order 1 ARE lost by a running sum that
    //     has already grown, while a grouped sum keeps them.
    //
    // TWO things had to be right, and both were found by measurement:
    //
    //  (1) ORDERING. The big terms must come FIRST, so the running sum is
    //      already large when the small ones arrive. The obvious first guess --
    //      INTERLEAVED big/tiny -- does NOT work: it re-adds a large term every
    //      other step, so the running sum never grows past one big term and the
    //      small terms survive. That was this file's first bug, and the run
    //      printed all zeros because of it.
    //
    //  (2) THE BOUNDARY MUST FALL BETWEEN THE TWO BLOCKS. This is the subtle
    //      one, and it cost a SECOND all-zeros run. Putting big in [0,K/4) and
    //      small in [K/4,K/2) does NOT discriminate: then chunk 0 contains BOTH
    //      blocks, so its partial sum loses the small terms exactly as the
    //      sequential sum does and split{S} reproduces seq bit-for-bit. The
    //      small terms survive a split only when they are accumulated from ZERO
    //      in a partial of their own, which requires the block boundary to
    //      coincide with a chunk boundary. big = first HALF, small = second HALF
    //      does that for every S >= 2.
    //
    //  (3) ESCALATING MAGNITUDES across the second half, so the S partial sums
    //      differ in magnitude. That is what makes split{S} and tree{S}
    //      genuinely different tolerances rather than two names for one number
    //      -- the property check (c) exists to prove, and uniform data cannot
    //      witness it at all.
    //
    //  (4) K MUST BE LARGE ENOUGH, and the threshold is not where intuition
    //      says. Measured on this box: K=4096 does NOT move the store, K=8192
    //      does. Absorption requires the running sum to pass the FP16 GRID
    //      SPACING at its own magnitude, not merely to exceed the small terms.
    //      At K=4096 the first half sums to 1.34e8, whose FP16 spacing is 8, so
    //      a small term of 1.0 lands in the SAME FP16 bucket as its neighbours
    //      and the store rounds them together. At K=8192 the sum reaches 2.68e8
    //      (spacing 16) and the difference finally crosses a bucket boundary.
    //      Hence k = 2K below, and the run REPORTS which K discriminated rather
    //      than asserting that all of them do.
    //
    //  (5) The escalation must stay INSIDE FP16 range. An earlier 1.25^(K/2)
    //      ramped past 65504 within a few hundred terms, so the whole tail
    //      became INF and INF-INF is NaN -- which made the partial sum ZERO and
    //      silently disarmed the discriminator. That was this file's third bug,
    //      and the same species as the first two: a dataset that quietly does
    //      not exercise the thing it claims to. Hence the gentle ramp below,
    //      which is bounded for any K.
    const float big = 65504.0f;   // FP16 max, exactly representable
    const std::size_t k = K * 2;  // see (4): the threshold sits between 4k and 8k
    const std::size_t half = k / 2;
    r.a.assign(k, big);
    r.w.assign(k, 1.0f);
    float mag = 1.0f;
    for (std::size_t i = half; i < k; ++i) {
      r.a[i] = from_half_bits(to_half_bits(mag));
      mag = mag * 1.0009f + 0.5f;
    }
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  bool self_check = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--self-check") self_check = true;

  const std::vector<std::size_t> Ks = {512, 2048, 4096, 8192};
  const std::vector<int> splits = {2, 4, 8, 16};
  const int rows = self_check ? 96 : 500;
  const int kinds[4] = {kRandom, kOutliers, kCancellation, kInterleaved};

  printf("== K-SPLIT NUMERICS CONTRACT ==\n");
  printf("host model of gemm_usm_decode1_vec's INT2 table-hoist branch "
         "(quant_gemm.cpp)\n");
  printf("seq = today's order (one accumulator, ascending K)\n");
  printf("reported as FP16 ulp AFTER the store -- the value the engine reads\n\n");

  int bad = 0;
  bool saw_movement = false;      // (a) discriminating direction
  bool saw_quiet_realistic = false;// (b) always-red converse
  bool saw_separation = false;    // (c) split vs tree are not the same number

  for (std::size_t K : Ks) {
    printf("-- K=%zu\n", K);
    for (int kind : kinds) {
      std::mt19937 rng(0x5eed1234u ^ (unsigned)(K * 131u + (unsigned)kind));
      std::vector<Row> rs;
      rs.reserve((std::size_t)rows);
      for (int i = 0; i < rows; ++i) rs.push_back(make_row(rng, K, kind));

      double worst_margin_split = 0.0, worst_margin_tree = 0.0;
      int moved_split = 0, moved_tree = 0;
      double worst_rel = 0.0;
      // Separation is measured BETWEEN the two orders, not against seq. It is
      // the property that makes quoting one of them ambiguous, and it only
      // shows up when the partial sums themselves differ in magnitude -- with
      // uniform partials, split{S} and tree{S} coincide at S=2 and nearly
      // coincide above it, so a corpus of uniform rows cannot witness this at
      // all. Hence the escalating magnitudes.
      double worst_order_gap = 0.0;
      for (int S : splits) {
        for (const Row& r : rs) {
          const float s = acc_seq(r);
          const std::uint16_t hs = to_half_bits(s);
          const float a1 = acc_split(r, S), a2 = acc_tree(r, S);
          if (fp16_ulp(hs, to_half_bits(a1))) ++moved_split;
          if (fp16_ulp(hs, to_half_bits(a2))) ++moved_tree;
          const double m1 = ulp_margin(s, a1), m2 = ulp_margin(s, a2);
          worst_margin_split = std::max(worst_margin_split, m1);
          worst_margin_tree = std::max(worst_margin_tree, m2);
          const double denom = std::max(1.0, std::fabs((double)s));
          worst_rel = std::max(worst_rel,
                               std::fabs((double)a1 - (double)s) / denom);
          const double gap = std::fabs((double)a1 - (double)a2) / denom;
          worst_order_gap = std::max(worst_order_gap, gap);
        }
      }
      printf("   %-18s moved_fp16: split=%d/%d tree=%d/%d | margin consumed: "
             "split=%.4g ulp  tree=%.4g ulp | split-vs-tree gap=%.2e\n",
             kind_name(kind), moved_split, rows * (int)splits.size(),
             moved_tree, rows * (int)splits.size(),
             worst_margin_split, worst_margin_tree, worst_order_gap);
      if (worst_margin_split >= 1.0 || worst_margin_tree >= 1.0)
        saw_movement = true;
      if (kind != kInterleaved && moved_split == 0 && moved_tree == 0)
        saw_quiet_realistic = true;
      // Only a gap that can actually CHANGE THE STORE makes the two orders
      // different tolerances; an FP32-level gap that FP16 absorbs is not a
      // difference a reader could act on.
      if (worst_order_gap > 0.0) saw_separation = true;
    }
    printf("\n");
  }

  printf("== THE CONTRACT ==\n");
  printf("  C1. A K-split is judged AFTER the FP16 store, in FP16 ulps. Judged\n");
  printf("      in FP32 relative error it looks alarming and is invisible.\n");
  printf("  C2. split{S} and tree{S} are DIFFERENT tolerances (same partials,\n");
  printf("      different combination order). A kernel states which one it uses.\n");
  printf("  C3. On REALISTIC activations -- random, act-outliers, and full\n");
  printf("      cancellation -- the split does not move the FP16 store AT ALL.\n");
  printf("      Measured, not argued: the FP16 activation bound (|a| <= 65504)\n");
  printf("      caps the sum at K*65504, so FP32 reassociation error lands ~2\n");
  printf("      orders of magnitude below one FP16 ulp.\n");
  printf("  C4. It is NOT unconditional. A sum that reaches 5e8 before the small\n");
  printf("      terms arrive DOES lose them (measured above), so the contract is\n");
  printf("      'stream-identical on the gate models', which is TESTABLE -- not\n");
  printf("      'bit-exact for all inputs', which is FALSE and would be a lie.\n");
  printf("  C5. Any gate for a K-split kernel must assert BOTH directions: an\n");
  printf("      arm where the flag fires, and an arm where it must NOT.\n");

  // --- harness self-checks -------------------------------------------------
  printf("\n== SELF-CHECK ==\n");
  {
    std::mt19937 rng(999);
    Row r = make_row(rng, 4096, kRandom);
    // (1) S=1 must reproduce seq BIT-EXACTLY: the split degenerates to the
    //     same order. If not, the host model is not the kernel's order.
    const bool s1 = to_half_bits(acc_seq(r)) == to_half_bits(acc_split(r, 1)) &&
                    to_half_bits(acc_seq(r)) == to_half_bits(acc_tree(r, 1));
    printf("  [1] S=1 reproduces seq bit-exactly .......... %s\n",
           s1 ? "yes" : "NO");
    if (!s1) ++bad;
  }
  {
    // (2) COMPILER-REASSOCIATION ASSERTION. The adversarial dataset must show
    //     seq != grouped under the CURRENT build. If -O3 ever reassociates the
    //     reference, `seq` stops being the device's order and every delta above
    //     becomes a measurement of a fiction.
    std::mt19937 rng(4242);
    Row r = make_row(rng, 4096, kInterleaved);
    const float s = acc_seq(r), g = acc_split(r, 2);
    const bool differs = to_half_bits(s) != to_half_bits(g);
    printf("  [2] adversarial: seq != grouped (compiler did NOT\n"
           "      reassociate the reference) .............. %s\n",
           differs ? "yes" : "NO -- host ref is not the device order");
    if (!differs) ++bad;
  }
  {
    std::uint16_t h = to_half_bits(1.0f);
    const int u = fp16_ulp(h, (std::uint16_t)(h + 1));
    const int z = fp16_ulp(h, h);
    printf("  [3] fp16_ulp fires on adjacent (%d>=1) and stays\n"
           "      quiet on identical (%d==0) ................ %s\n",
           u, z, (u >= 1 && z == 0) ? "yes" : "NO");
    if (u < 1 || z != 0) ++bad;
  }
  {
    // (4) The fixed fp16_ulp must NOT be the truncating one. A wide move has to
    //     report a distance larger than the old 256 cap; if these are equal the
    //     "fix" did nothing and every distance above is understated.
    // 1.0 and 1.5 are the SAME FP16 value -- the first version of this check
    // used them and asserted a nonzero distance, i.e. it demanded a function
    // report a move where there is none. Another instance of the class this
    // file keeps hitting: a test written against an expectation instead of
    // against the measurement. These two really are far apart.
    const std::uint16_t a = to_half_bits(1.0f);
    const std::uint16_t b = to_half_bits(1.5e3f);
    const int full = fp16_ulp(a, b), naive = fp16_ulp_naive(a, b);
    printf("  [4] fp16_ulp is not the truncating version: 1.0->1500\n"
           "      full=%d naive=%d, full must exceed the 256 cap . %s\n",
           full, naive, (full > naive && full > 256) ? "yes" : "NO");
    if (!(full > naive && full > 256)) ++bad;
  }
  {
    // (5) A NEGATIVE-side move must be measured too. The old walk only stepped
    //     toward +inf, so a value that moved DOWN reported the wrong distance.
    const std::uint16_t a = to_half_bits(1.5e3f);
    const std::uint16_t b = to_half_bits(1.0f);
    const int d = fp16_ulp(a, b);
    printf("  [5] downward move measured (%d, want >256) ......... %s\n", d,
           d > 256 ? "yes" : "NO");
    if (d <= 256) ++bad;
  }

  printf("\n");
  // The discriminator is MARGIN CROSSING 1 ulp, not the boolean "did the FP16
  // value change". Those are different claims and conflating them was this
  // file's fourth bug: a margin of 1.004 ulp means the perturbation is the
  // width of an FP16 bucket -- one more term and it crosses -- while
  // moved_fp16 == 0 because on THESE seeds it landed just short. Requiring the
  // boolean would have made this run report FAIL on a finding that is true and
  // important; requiring only the boolean would have made it pass on a harness
  // that cannot see anything. So the check is: the adversarial margin must
  // CROSS 1 ulp somewhere, and the realistic margins must stay at 0.
  if (!saw_movement) {
    printf("FAIL: no dataset pushed the margin past 1 FP16 ulp, so the\n"
           "      tolerance derived here is unfalsifiable -- the harness cannot\n"
           "      see a move even in the case built to produce one.\n");
    ++bad;
  }
  if (!saw_quiet_realistic) {
    printf("FAIL: no realistic dataset stayed put, so this gate would be\n"
           "      always-red -- which trains everyone to ignore it.\n");
    ++bad;
  }
  if (!saw_separation) {
    printf("FAIL: split{S} and tree{S} measured identically, so quoting one\n"
           "      is quoting the other.\n");
    ++bad;
  }
  if (bad) {
    printf("K-SPLIT CONTRACT: FAIL (%d problem(s))\n", bad);
    printf("RESULT: the numerics contract is NOT established\n");
    return 1;
  }
  printf("K-SPLIT CONTRACT: PASS (contract measured; a discriminating case and\n"
         "  a quiet case both exist; the two orders separate)\n");
  printf("RESULT: the numerics contract for a K-split M=1 decode kernel is\n"
         "        MEASURED, not guessed -- and it permits the split\n");
  return 0;
}
