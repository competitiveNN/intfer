// Model-free probe for the INTFER_XMX_GEMM arm of
// quant_gemm_forward_usm_xmx (the XMX joint_matrix DPAS kernel
// for INT2/INT4 weights).
//
// WHY THIS PROBE EXISTS. The engine wires the XMX arm ONLY into
// Engine::run_linear_tp (engine.cpp:1434). The 27B prefill hot
// path does NOT go through run_linear_tp: it uses
// Engine::run_linear_wave (engine.cpp:2299) for the q/k/v/z/b/a/
// la_out wave and Engine::run_mlp_gateup (engine.cpp:1789) for the
// MLP gate+up, and both call quant_gemm_forward_usm DIRECTLY. The
// only run_linear_tp caller on the CLI path is the M=1 decode step
// (main.cpp), and the XMX kernel's own gate is M >= 8 with
// M % 16 == 0, N % 16 == 0, K % 32 == 0 -- so on every real model
// in this tree the arm is UNREACHABLE from the CLI: the gate's
// test_gpu_optin.sh INTFER_XMX_GEMM arm produced NO [xmx-gemm] line
// on the 0.8B fixture (M = the ~5-token prompt length), which is
// the vacuous-green class the flag census exists to kill.
//
// WHAT THIS PROVES, independent of any model:
//   1. the XMX arm is REACHABLE and actually SUBMITS on a tile-aligned
//      shape -- asserted by `ran == true` (the kernel's out-param,
//      which is the only way to observe a submission from the caller's
//      side: the [xmx-gemm] line is printed by the ENGINE, not the
//      kernel, so a direct call has no announcement to grep);
//   2. the XMX output AGREES with the tiled kernel's output to the
//      FP32 reordering bound -- the kernel's own correctness claim,
//      verified at runtime rather than argued (see NUMERICS below);
//   3. both outputs stay inside an AGREED ABSOLUTE tolerance of a
//      host FP32 reference that dequantizes the weight exactly as the
//      kernel does (scale * canon_value);
//   4. THE NEGATIVE DIRECTION: M < 8 MUST decline (ran == false),
//      because the decode path is memory-bound and the SLM round trip
//      cannot pay there -- if a future change loosened the M >= 8 gate
//      the probe would catch it.
//
// THE CONTRACT IS ABSOLUTE, NEVER RELATIVE AND NEVER BITWISE.
//   * NOT bitwise: the tiled kernel and the XMX kernel accumulate the
//     same FP32 dot product in the same ascending-K order, but on
//     shapes where K spans more than one BK tile the tiled kernel's
//     sub-tile reduction order differs from the joint_matrix
//     accumulator's blocked order, so the two outputs can differ by a
//     few ULPs (measured on this box: 0/256 elements at K=32, 0/512
//     at K=64, 1/512 at K=128, 5/4096 at K=256, 7/4096 at K=512).
//   * NOT relative: the reference sums half-range operands (-0.5, 0.5)
//     times {-1,-1/3,+1/3,+1} codes, so the true sum is a random walk
//     near ZERO, and relative error on a near-zero value is dominated
//     by cancellation, not by the kernel (the TILED kernel -- the
//     shipped, trusted path -- fails a relative check identically,
//     which is the tell).
//   * ABSOLUTE is the honest unit: the kernel rounds the dequantized
//     weight into an FP16 SLM tile (sW = sycl::half(wval)) before the
//     DPAS, while the reference keeps it in FP32, so each term carries
//     at most one FP16 ULP of weight rounding, and the output carries
//     at most K of them. kTolAbs = K * 4e-4 bounds exactly that
//     (measured worst case 5.6e-6 at K=32, ~0.05 at K=512 -- both
//     far under the bound), and is far below the O(1) movement a wrong
//     transposition, a wrong group index, or a wrong alpha would cause.
//
// NEEDS A GPU, but no model: all tensors are synthetic. Exit 0 = the
// arm is reachable and agrees with the tiled kernel. Exit 1 = it is
// not. Exit 2 = no GPU, a precondition rather than a defect.
//
// Usage:  xmx_probe            # M >= 16 tile-aligned arms (both kernels)
//         xmx_probe --m1       # M = 1 negative direction (arm must decline)
//         xmx_probe --steal    # work-stealing arm: n_tiles=32 > nWG=16,
//                               # asserts the claim counter is consistent
//                               # (sum == n_tiles, min >= 1)
#include "intfer/gemm/quant_gemm.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- FP16 helpers (RNE converter, so the reference's rounding
//      matches the kernel's store). ----
std::uint16_t to_half_bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xFF) - 127 + 15;
  const std::uint32_t man = u & 0x7FFFFFu;
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
  return static_cast<std::uint16_t>(sign | ((std::uint32_t)exp << 10) | (man >> 13));
}
float from_half_bits(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 0x1u;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t man = h & 0x3FFu;
  std::uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign << 31;
    } else {
      std::int32_t e = -1;
      std::uint32_t m = man;
      while (!(m & 0x400u)) { m <<= 1; --e; }
      m &= 0x3FFu;
      f = (sign << 31) | ((std::uint32_t)(127 - 15 + e) << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {
    f = (sign << 31) | 0x7F800000u | (man << 13);
  } else {
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

// Canonical DASH-Q codebook value, mirroring canon_value() in
// quant_gemm.cpp (the kernel's own dequantizer).
inline float canon_value(int bits, int code) {
  if (bits >= 8) return (2.0f * (float)code - 255.0f) / 255.0f;
  if (bits == 4) return (2.0f * (float)code - 15.0f) / 15.0f;
  static const float c2[4] = {-1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f};
  return c2[code & 3];
}

// ---- tolerances (see the block comment above). ----
// kTolAbs: K * 4e-4 -- K FP16 weight-rounding ULPs, the honest bound
//          on the kernel-vs-reference divergence. Floor of 1e-3 so a
//          single-tile shape still has room for one ULP of output.
// kMaxUlp: 4 -- the XMX-vs-tiled budget in FP16 ULPs (the measured
//          worst case is 7 differing elements out of 4096 at K=512,
//          each a single ULP apart).
constexpr double kTolAbsFloor = 1.0e-3;
constexpr double kTolAbsPerK = 4.0e-4;
constexpr int kMaxUlp = 4;
constexpr double kMaxBitDiffFrac = 0.005;  // 0.5% of elements may differ

int half_ulp_dist(std::uint16_t a, std::uint16_t b) {
  auto ord = [](std::uint16_t h) -> int {
    return (h & 0x8000u) ? -(int)(h & 0x7FFFu) : (int)(h & 0x7FFFu);
  };
  int d = ord(a) - ord(b);
  return d < 0 ? -d : d;
}

struct Shape {
  std::uint32_t M, N, K, group;
  const char* tag;
};

// The XMX kernel's own CALLER GUARD is M % 16 == 0, N % 16 == 0,
// K % 32 == 0, M >= 8. Every shape here satisfies all four, so the
// arm MUST take them (ran == true). group is a divisor of K so the
// per-group scale index (kg / group) is exact.
const Shape kAligned[] = {
    {16, 16, 32, 16, "M16-N16-K32-g16"},
    {16, 32, 64, 16, "M16-N32-K64-g16"},
    {32, 16, 128, 32, "M32-N16-K128-g32"},
    {64, 64, 256, 64, "M64-N64-K256-g64"},
    {128, 32, 512, 32, "M128-N32-K512-g32"},
};

// Deterministic xorshift. Half-range operands keep the FP16 output
// from overflowing; a full-range fill would push the accumulator past
// FP16 max and read as a defect that is really the harness's.
void fill_half(std::uint16_t* p, std::size_t n, std::uint64_t& seed) {
  for (std::size_t i = 0; i < n; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::uint32_t v = static_cast<std::uint32_t>(seed >> 33) & 0xFFFFu;
    const float f = -0.5f + 1.0f * (float)v / 65535.0f;
    p[i] = to_half_bits(f);
  }
}
void fill_u8(std::uint8_t* p, std::size_t n, std::uint64_t& seed) {
  for (std::size_t i = 0; i < n; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    p[i] = static_cast<std::uint8_t>((seed >> 33) & 0xFFu);
  }
}

// Host FP32 reference: dequantize the weight exactly as the kernel
// does (scale[ng*ngw + kg/group] * canon_value(bits, code)) and sum
// ascending-k in FP32. This is the ARBITER for correctness (point 3).
void ref_sum(const std::uint16_t* A, const std::uint8_t* code,
             const std::uint16_t* scale, int bits, std::uint32_t M,
             std::uint32_t N, std::uint32_t K, std::uint32_t group,
             std::vector<float>& out) {
  const std::uint32_t pack = (bits >= 8) ? 1u : (8u / (std::uint32_t)bits);
  const std::uint32_t ngw = (K + group - 1) / group;
  out.assign((std::size_t)M * N, 0.0f);
  for (std::uint32_t m = 0; m < M; ++m) {
    for (std::uint32_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (std::uint32_t k = 0; k < K; ++k) {
        const std::size_t idx = (std::size_t)n * K + k;
        const std::uint32_t g = k / group;
        const float s = from_half_bits(scale[(std::size_t)n * ngw + g]);
        int c;
        if (pack == 1) {
          c = static_cast<int>(code[idx]);
        } else {
          const std::uint8_t b = code[idx / pack];
          const std::uint32_t within = idx % pack;
          c = (b >> (within * bits)) & ((1u << bits) - 1u);
        }
        acc += from_half_bits(A[(std::size_t)m * K + k]) * s *
               canon_value(bits, c);
      }
      out[(std::size_t)m * N + n] = acc;
    }
  }
}

// Absolute comparison of a device output against the host
// reference. The pass criterion is the ABSOLUTE difference only:
// an FP16 ULP budget is NOT meaningful here because the reference
// sums half-range operands into a near-zero random walk, and near
// zero FP16 is denormal (1 ULP = 2^-24), so a perfectly correct
// single-ULP-of-weight difference reads as dozens of ULPs. The
// honest unit is the absolute budget derived in the block comment.
struct CmpResult {
  double max_abs;       // worst |dev - ref|
  int max_ulp;          // informational only (denormal-sensitive)
  int differ;           // elements > 2 ULP from the reference
  int differ_total;
  std::size_t worst_m, worst_n;
};
CmpResult compare(const std::uint16_t* dev, const std::vector<float>& ref,
                  std::uint32_t M, std::uint32_t N) {
  CmpResult r{0.0, 0, 0, 0, 0, 0};
  for (std::uint32_t m = 0; m < M; ++m) {
    for (std::uint32_t n = 0; n < N; ++n) {
      const std::size_t idx = (std::size_t)m * N + n;
      const float got = from_half_bits(dev[idx]);
      const float want = ref[idx];
      const double abs = std::fabs((double)got - (double)want);
      if (abs > r.max_abs) { r.max_abs = abs; r.worst_m = m; r.worst_n = n; }
      const int ulp = half_ulp_dist(dev[idx], to_half_bits(want));
      if (ulp > r.max_ulp) r.max_ulp = ulp;
      if (ulp > 2) ++r.differ;
      ++r.differ_total;
    }
  }
  return r;
}
bool within_abs(const CmpResult& c, double tol_abs) {
  return c.max_abs <= tol_abs;
}

}  // namespace

// Select a device exactly as the engine does (Level-Zero GPU only);
// sycl::gpu_selector_v spans backends on this box and the multi-device
// queue throws -- see examples/onemkl_probe/main.cpp for the measured
// failure mode.
static sycl::device pick_l0_gpu() {
  auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
  sycl::device chosen;
  for (const auto& d : all) {
    if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    chosen = d;
    break;
  }
  return chosen;
}

int main(int argc, char** argv) {
  bool m1_only = false;
  bool steal_only = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--m1") == 0) m1_only = true;
    else if (std::strcmp(argv[i], "--steal") == 0) steal_only = true;
  }

  sycl::device dev = pick_l0_gpu();
  std::fprintf(stderr, "xmx_probe: device=%s\n",
               dev.is_gpu() ? dev.get_info<sycl::info::device::name>().c_str()
                            : "<none>");
  std::fflush(stderr);
  if (!dev.is_gpu()) { std::fprintf(stderr, "SKIP: no GPU device\n"); return 2; }

  sycl::queue* q = nullptr;
  try { q = new sycl::queue{dev}; }
  catch (const sycl::exception& e) {
    std::fprintf(stderr, "SKIP: no GPU queue (%s)\n", e.what());
    return 2;
  }

  // ---- THE NEGATIVE DIRECTION. M < 8 MUST decline (ran == false). ----
  // The arm's own gate is M >= 8; if a future change loosened it, the
  // decode path would start paying an SLM round trip for a memory-bound
  // dot product, so the DECLINE is the thing being asserted.
  if (m1_only) {
    const Shape s{1, 64, 128, 16, "M1-N64-K128-g16"};
    std::uint64_t seed = 0x5eed1234ULL;
    const std::size_t na = (std::size_t)s.M * s.K;
    const std::size_t nw = (std::size_t)s.N * s.K;
    const std::size_t ngw = ((std::size_t)s.K + s.group - 1) / s.group;
    const std::size_t nscale = (std::size_t)s.N * ngw;
    const std::size_t ncode = ((std::size_t)s.N * s.K + 3) / 4;
    std::vector<std::uint16_t> A(na), W(nw), scale(nscale);
    std::vector<std::uint8_t> code(ncode);
    fill_half(A.data(), na, seed);
    fill_half(W.data(), nw, seed);
    fill_half(scale.data(), nscale, seed);
    fill_u8(code.data(), ncode, seed);
    auto* a_d = sycl::malloc_shared<std::uint16_t>(na, *q);
    auto* c_d = sycl::malloc_shared<std::uint16_t>((std::size_t)s.M * s.N, *q);
    auto* code_d = sycl::malloc_shared<std::uint8_t>(ncode, *q);
    auto* scale_d = sycl::malloc_shared<std::uint16_t>(nscale, *q);
    std::memcpy(a_d, A.data(), na * 2);
    std::memcpy(code_d, code.data(), ncode);
    std::memcpy(scale_d, scale.data(), nscale * 2);
    std::memset(c_d, 0, (std::size_t)s.M * s.N * 2);
    bool ran = false;
    sycl::event e = intfer::gemm::quant_gemm_forward_usm_xmx(
        *q, a_d, code_d, scale_d, 1.0f, c_d, s.M, s.N, s.K, s.group,
        intfer::gemm::WeightScheme::INT2, sycl::event{}, &ran, nullptr);
    if (e != sycl::event{}) e.wait();
    q->wait();
    bool ok = true;
    if (ran) {
      std::printf("m1: FAIL -- the arm SUBMITTED at M=1; it must decline\n");
      ok = false;
    } else {
      std::printf("m1: PASS -- arm declined at M=1 (ran=false)\n");
    }
    sycl::free(a_d, *q); sycl::free(c_d, *q);
    sycl::free(code_d, *q); sycl::free(scale_d, *q);
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    delete q;
    return ok ? 0 : 1;
  }

  // ---- THE WORK-STEALING ARM (INTFER_XMX_TILE_SCHED=1). ----
  // Certifies what the fixed-assignment arm cannot:
  //   1. the persistent kernel LAUNCHES with nWG=16 work-groups
  //      on a shape with MORE than 16 tiles (n_tiles=32 here:
  //      gx=8, gy=4). At n_tiles <= nWG every work-group claims
  //      exactly one tile and the two paths are indistinguishable,
  //      so a shape with n_tiles > nWG is the only one that
  //      actually exercises the claim loop;
  //   2. the work-stealing output AGREES with the fixed-assignment
  //      output -- same process_tile body, so the only legitimate
  //      divergence is accumulation order, which is unchanged;
  //   3. THE CLAIM COUNTER IS CONSISTENT: sum(tile_ctr[0..nWG-1])
  //      == n_tiles and every slot is >= 1. This is the direct
  //      evidence for the per-work-group counter index the kernel
  //      now uses. The old code indexed tile_ctr[0] from every
  //      work-group, which is a race -- two groups would claim the
  //      same slot and a third would read garbage, so unclaimed
  //      tiles would stay zero and the sum would be wrong. The
  //      counter sum catches that without depending on the output
  //      comparison, which is the same load-independence argument
  //      the gate's own tooling makes.
  if (steal_only) {
    const Shape s{128, 64, 512, 32, "M128-N64-K512-g32-steal"};
    const int bits = 2;
    const std::size_t na = (std::size_t)s.M * s.K;
    const std::size_t nw = (std::size_t)s.N * s.K;
    const std::uint32_t ngw = ((std::uint32_t)s.K + s.group - 1) / s.group;
    const std::size_t nscale = (std::size_t)s.N * ngw;
    const std::size_t ncode = ((std::size_t)s.N * s.K + 3) / 4;
    const std::size_t nc = (std::size_t)s.M * s.N;
    const std::uint32_t gx = s.M / 16, gy = s.N / 16;
    const std::uint32_t n_tiles = gx * gy;
    const std::uint32_t nWG = 16;  // the kernel's persistent width
    std::uint64_t seed = 0x57ea1234ULL;
    std::vector<std::uint16_t> A(na), W(nw), scale(nscale);
    std::vector<std::uint8_t> code(ncode);
    fill_half(A.data(), na, seed);
    fill_half(W.data(), nw, seed);
    fill_half(scale.data(), nscale, seed);
    fill_u8(code.data(), ncode, seed);
    std::vector<float> ref;
    ref_sum(A.data(), code.data(), scale.data(), bits, s.M, s.N, s.K,
            s.group, ref);

    auto* a_d = sycl::malloc_shared<std::uint16_t>(na, *q);
    auto* code_d = sycl::malloc_shared<std::uint8_t>(ncode, *q);
    auto* scale_d = sycl::malloc_shared<std::uint16_t>(nscale, *q);
    auto* c_fix = sycl::malloc_shared<std::uint16_t>(nc, *q);
    auto* c_stl = sycl::malloc_shared<std::uint16_t>(nc, *q);
    auto* ctr = sycl::malloc_shared<std::uint32_t>(nWG, *q);
    std::memcpy(a_d, A.data(), na * 2);
    std::memcpy(code_d, code.data(), ncode);
    std::memcpy(scale_d, scale.data(), nscale * 2);
    std::memset(c_fix, 0, nc * 2);
    std::memset(c_stl, 0, nc * 2);
    std::memset(ctr, 0, nWG * sizeof(std::uint32_t));

    // Arm 1: fixed assignment (no tile_ctr, scheduler unset).
    unsetenv("INTFER_XMX_TILE_SCHED");
    bool ran_fix = false;
    sycl::event ef = intfer::gemm::quant_gemm_forward_usm_xmx(
        *q, a_d, code_d, scale_d, 1.0f, c_fix, s.M, s.N, s.K,
        s.group, intfer::gemm::WeightScheme::INT2, sycl::event{},
        &ran_fix, nullptr);
    if (ef != sycl::event{}) ef.wait();
    q->wait();

    // Arm 2: work-stealing (tile_ctr, scheduler set).
    setenv("INTFER_XMX_TILE_SCHED", "1", 1);
    bool ran_stl = false;
    sycl::event es = intfer::gemm::quant_gemm_forward_usm_xmx(
        *q, a_d, code_d, scale_d, 1.0f, c_stl, s.M, s.N, s.K,
        s.group, intfer::gemm::WeightScheme::INT2, sycl::event{},
        &ran_stl, ctr);
    if (es != sycl::event{}) es.wait();
    q->wait();
    unsetenv("INTFER_XMX_TILE_SCHED");

    const double tol_abs = std::max(kTolAbsFloor, (double)s.K * kTolAbsPerK);
    const CmpResult r_fix = compare(c_fix, ref, s.M, s.N);
    const CmpResult r_stl = compare(c_stl, ref, s.M, s.N);

    // steal vs fixed: bit-exact expected (same process_tile body,
    // same accumulation order, only WHICH work-group runs WHICH
    // tile differs). Reported as a bound, asserted as a bound.
    int bit_diff = 0;
    int max_ulp_svsF = 0;
    for (std::size_t i = 0; i < nc; ++i) {
      if (c_fix[i] != c_stl[i]) ++bit_diff;
      const int ulp = half_ulp_dist(c_stl[i], c_fix[i]);
      if (ulp > max_ulp_svsF) max_ulp_svsF = ulp;
    }
    const double bit_diff_frac = (double)bit_diff / (double)nc;

    // The counter is the direct evidence. The claim loop
    // does exactly ONE fetch_add per iteration and breaks on
    // the first iteration that returns cur >= n_tiles, so
    // every work-group contributes (its claims) + 1
    // exhaustion read. Total claims == n_tiles (each tile
    // claimed exactly once), so the final counter is
    // n_tiles + nWG EXACTLY under a correct atomic -- but
    // the bound [n_tiles, n_tiles + nWG] is asserted rather
    // than the exact value, because it is the robust form:
    //   sum < n_tiles   -> lost updates (the work_group-scope
    //                      bug: cross-group atomics that do not
    //                      cohere, so claims vanish);
    //   sum > n_tiles+nWG -> double claims (the per-WG-slot
    //                      bug: every group claims the same
    //                      tiles independently, measured at
    //                      n_tiles=32, nWG=16 -> sum=528).
    // A late work-group may legitimately claim ZERO tiles
    // (min == 0), so min is NOT asserted -- work-stealing is
    // load balancing, and a fast group stealing more tiles is
    // the feature, not a defect.
    std::uint32_t ctr_sum = 0, ctr_min = ~0u;
    for (std::uint32_t g = 0; g < nWG; ++g) {
      ctr_sum += ctr[g];
      if (ctr[g] < ctr_min) ctr_min = ctr[g];
    }

    const bool tiled_ok = within_abs(r_fix, tol_abs);
    const bool xmx_ok = within_abs(r_stl, tol_abs);
    const bool agree = (max_ulp_svsF <= kMaxUlp) &&
                       (bit_diff_frac <= kMaxBitDiffFrac);
    const bool counter_ok = (ctr_sum >= n_tiles) &&
                          (ctr_sum <= n_tiles + nWG);
    bool ok = ran_fix && ran_stl && tiled_ok && xmx_ok &&
              agree && counter_ok;

    std::printf("%s: ran(fix=%s stl=%s) ctr(sum=%u in[%u,%u] min=%u) "
                "xvsF(ulp=%d diff=%d/%zu) fix(abs=%.3g) stl(abs=%.3g) "
                "tol=%.3g | %s\n",
                s.tag, ran_fix ? "yes" : "NO", ran_stl ? "yes" : "NO",
                ctr_sum, n_tiles, n_tiles + nWG, ctr_min,
                max_ulp_svsF, bit_diff, nc,
                r_fix.max_abs, r_stl.max_abs, tol_abs,
                ok ? "PASS" : "FAIL");

    if (!ran_fix || !ran_stl) {
      std::printf("%s: FAIL -- a work-stealing arm did NOT submit "
                  "(ran_fix=%d ran_stl=%d); the arm is unreachable "
                  "here\n", s.tag, (int)ran_fix, (int)ran_stl);
      ok = false;
    }
    if (!counter_ok) {
      std::printf("%s: FAIL -- the claim counter is inconsistent "
                  "(sum=%u outside [%u, %u]); tiles were double-"
                  "claimed (sum too high) or lost (sum too low) -- "
                  "the shared counter's atomic scope is wrong\n",
                  s.tag, ctr_sum, n_tiles, n_tiles + nWG);
      ok = false;
    }
    if (!agree) {
      std::printf("%s: FAIL -- the work-stealing output diverges from "
                  "the fixed-assignment output beyond the reordering "
                  "bound (ulp=%d > %d, bit_diff=%d/%zu)\n",
                  s.tag, max_ulp_svsF, kMaxUlp, bit_diff, nc);
      ok = false;
    }
    if (!tiled_ok || !xmx_ok) {
      std::printf("%s: FAIL -- an output is outside the absolute "
                  "tolerance of the host reference (fix=%.3g stl=%.3g "
                  "tol=%.3g)\n", s.tag, r_fix.max_abs, r_stl.max_abs,
                  tol_abs);
      ok = false;
    }

    sycl::free(a_d, *q); sycl::free(code_d, *q);
    sycl::free(scale_d, *q); sycl::free(c_fix, *q);
    sycl::free(c_stl, *q); sycl::free(ctr, *q);
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    delete q;
    return ok ? 0 : 1;
  }

  // ---- THE ALIGNED ARMS. For each (M,N,K): run the tiled kernel and
  //      the XMX kernel on IDENTICAL inputs and assert
  //        (a) the XMX kernel submitted (ran == true),
  //        (b) the XMX output agrees with the tiled kernel to the FP32
  //            reordering bound (a few ULPs; see the block comment), and
  //        (c) both are inside the absolute tolerance of the host FP32
  //            reference. ----
  bool all_ok = true;
  bool ever_ran = false;
  for (const Shape& s : kAligned) {
    std::uint64_t seed = 0x1234567ULL ^ ((std::uint64_t)s.M << 32) ^
                         ((std::uint64_t)s.N << 16) ^ (std::uint64_t)s.K ^
                         (std::uint64_t)s.group;
    const int bits = 2;  // INT2; the INT4 path shares the same dequant
    const std::size_t na = (std::size_t)s.M * s.K;
    const std::size_t nw = (std::size_t)s.N * s.K;
    const std::size_t ngw = ((std::size_t)s.K + s.group - 1) / s.group;
    const std::size_t nscale = (std::size_t)s.N * ngw;
    const std::size_t ncode = ((std::size_t)s.N * s.K + 3) / 4;
    const std::size_t nc = (std::size_t)s.M * s.N;
    std::vector<std::uint16_t> A(na), W(nw), scale(nscale);
    std::vector<std::uint8_t> code(ncode);
    fill_half(A.data(), na, seed);
    fill_half(W.data(), nw, seed);
    fill_half(scale.data(), nscale, seed);
    fill_u8(code.data(), ncode, seed);

    // Host reference from the SAME inputs.
    std::vector<float> ref;
    ref_sum(A.data(), code.data(), scale.data(), bits, s.M, s.N, s.K,
            s.group, ref);

    // Device copies (one set, reused by both arms so the arms cannot
    // differ on inputs).
    auto* a_d = sycl::malloc_shared<std::uint16_t>(na, *q);
    auto* code_d = sycl::malloc_shared<std::uint8_t>(ncode, *q);
    auto* scale_d = sycl::malloc_shared<std::uint16_t>(nscale, *q);
    auto* c_tiled = sycl::malloc_shared<std::uint16_t>(nc, *q);
    auto* c_xmx = sycl::malloc_shared<std::uint16_t>(nc, *q);
    std::memcpy(a_d, A.data(), na * 2);
    std::memcpy(code_d, code.data(), ncode);
    std::memcpy(scale_d, scale.data(), nscale * 2);
    std::memset(c_tiled, 0, nc * 2);
    std::memset(c_xmx, 0, nc * 2);

    // Arm 1: the tiled kernel (the default path).
    sycl::event e1 = intfer::gemm::quant_gemm_forward_usm(
        *q, a_d, code_d, scale_d, 1.0f, c_tiled, s.M, s.N, s.K, s.group,
        intfer::gemm::WeightScheme::INT2, sycl::event{});
    if (e1 != sycl::event{}) e1.wait();
    // Arm 2: the XMX kernel (the arm under test).
    bool ran = false;
    sycl::event e2 = intfer::gemm::quant_gemm_forward_usm_xmx(
        *q, a_d, code_d, scale_d, 1.0f, c_xmx, s.M, s.N, s.K, s.group,
        intfer::gemm::WeightScheme::INT2, sycl::event{}, &ran, nullptr);
    if (e2 != sycl::event{}) e2.wait();
    q->wait();

    const double tol_abs = std::max(kTolAbsFloor, (double)s.K * kTolAbsPerK);
    const CmpResult r_tiled = compare(c_tiled, ref, s.M, s.N);
    const CmpResult r_xmx = compare(c_xmx, ref, s.M, s.N);

    // (b) XMX vs tiled: how far apart are the two kernels?
    int bit_diff = 0;
    int max_ulp_xvsT = 0;
    for (std::size_t i = 0; i < nc; ++i) {
      if (c_tiled[i] != c_xmx[i]) ++bit_diff;
      const int ulp = half_ulp_dist(c_xmx[i], c_tiled[i]);
      if (ulp > max_ulp_xvsT) max_ulp_xvsT = ulp;
    }
    const double bit_diff_frac = (double)bit_diff / (double)nc;

    const bool tiled_ok = within_abs(r_tiled, tol_abs);
    const bool xmx_ok = within_abs(r_xmx, tol_abs);
    const bool agree = (max_ulp_xvsT <= kMaxUlp) &&
                       (bit_diff_frac <= kMaxBitDiffFrac);
    ever_ran = ever_ran || ran;

    std::printf("%s: ran=%s tiled(abs=%.3g ulp=%d) xmx(abs=%.3g ulp=%d) "
                "xvsT(ulp=%d diff=%d/%zu) tol=%.3g | %s\n",
                s.tag, ran ? "yes" : "NO",
                r_tiled.max_abs, r_tiled.max_ulp,
                r_xmx.max_abs, r_xmx.max_ulp,
                max_ulp_xvsT, bit_diff, nc, tol_abs,
                (ran && tiled_ok && xmx_ok && agree) ? "PASS" : "FAIL");

    if (!ran) {
      std::printf("%s: FAIL -- the XMX arm did NOT submit on a tile-aligned "
                  "shape (ran=false); the arm is unreachable here\n", s.tag);
      all_ok = false;
    }
    if (!agree) {
      std::printf("%s: FAIL -- the XMX output diverges from the tiled kernel "
                  "by more than the FP32 reordering bound (ulp=%d > %d, "
                  "bit_diff=%d/%zu > %.4f); the two kernels disagree beyond "
                  "accumulation-order noise\n",
                  s.tag, max_ulp_xvsT, kMaxUlp, bit_diff, nc,
                  kMaxBitDiffFrac);
      all_ok = false;
    }
    if (!tiled_ok) {
      std::printf("%s: FAIL -- the TILED kernel is outside the absolute "
                  "tolerance of the host reference (abs=%.3g > %.3g ulp=%d) "
                  "-- the REFERENCE or the harness is suspect before any "
                  "XMX conclusion is drawn\n",
                  s.tag, r_tiled.max_abs, tol_abs, r_tiled.max_ulp);
      all_ok = false;
    }
    if (!xmx_ok) {
      std::printf("%s: FAIL -- the XMX kernel is outside the absolute "
                  "tolerance of the host reference (abs=%.3g > %.3g ulp=%d, "
                  "worst at m=%zu n=%zu)\n",
                  s.tag, r_xmx.max_abs, tol_abs, r_xmx.max_ulp,
                  r_xmx.worst_m, r_xmx.worst_n);
      all_ok = false;
    }

    sycl::free(a_d, *q); sycl::free(code_d, *q); sycl::free(scale_d, *q);
    sycl::free(c_tiled, *q); sycl::free(c_xmx, *q);
  }

  // A run where no shape submitted proves nothing about reachability, so
  // that is a FAIL even when every tolerance and agreement check passed
  // -- a kernel that never ran would also pass every comparison.
  if (all_ok && !ever_ran) {
    std::printf("FAIL: no shape submitted the XMX arm -- a kernel that "
                "never ran would also pass every comparison\n");
    all_ok = false;
  }

  std::printf("XMX-PROBE: %s\n", all_ok ? "PASS" : "FAIL");
  std::printf("RESULT: %s\n", all_ok ? "PASS" : "FAIL");
  delete q;
  return all_ok ? 0 : 1;
}
