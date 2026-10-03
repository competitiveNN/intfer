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
#include <algorithm>
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

// --- Reader/writer layout contract + device-level readback identity ----
//
// The value cases in main() gate the WRITER (quant_gemm_dpas48_rowfold). They
// cannot reach the READERS' stride plumbing, and neither can the end-to-end
// arms in test_gpu_optin.sh: the engine always sets fold_usm AND fold_stride
// together, so "fold_stride left at its 0 default" had NO coverage anywhere in
// the tree. That is the vacuous-gate shape -- a path nothing reaches is a path
// nothing can break.
//
// Two checks close it:
//
//  (1) CONTRACT. For every (nOp, stride), the reader must accept EXACTLY the
//      strides the writer accepts, resolving them to the same value. The
//      expected values here restate the writer's own rule independently
//      (0 => 2*nOp, then refuse anything below 2*nOp), so regressing the
//      reader's default to nOp fails immediately.
//
//  (2) DEVICE READBACK. Run the real quant_gemm_dpas48 twice on identical
//      inputs -- once with a hoisted fold buffer, once with the inline fold --
//      and require the stored FP16 outputs to be BIT-IDENTICAL. Covers three
//      stride regimes: the correct one, the 0 default, and a deliberately
//      too-small one (must fail CLOSED to the inline fold, not read garbage).
//
// The activation codes are built so S0 and Sfull are deterministically
// DISTINCT (lanes 0..15 = +(m+1), lanes 16..31 = -(m+1)), so a mis-strided read
// is caught by construction rather than by luck: with nOp=1 a reader wrongly
// resolving stride=nOp reads fold_d[m+1], which alternates Sfull/S0 between
// rows and so flips S for rows 1 and 3 by large, known amounts.
bool check_reader_contract(sycl::queue& q, std::uint64_t& pairs,
                           std::uint64_t& devs, std::uint64_t& bad) {
  bool ok = true;

  // ---- (1) writer/reader stride agreement ---------------------------
  const std::uint32_t nOps[] = {1, 2, 3, 4, 8, 16, 32, 64, 128, 448};
  const std::uint32_t kDummy = 1;
  const std::int32_t* dummy = reinterpret_cast<const std::int32_t*>(&kDummy);
  std::uint64_t cbad = 0;
  for (std::uint32_t nOp : nOps) {
    const std::uint32_t need = 2u * nOp;
    const std::uint32_t strides[] = {0u,      1u,      nOp,  need - 1u,
                                     need,    need + 1u, need + 7u, 2u * need};
    for (std::uint32_t s : strides) {
      // Independent restatement of the WRITER's rule (quant_gemm.cpp).
      const std::uint32_t exp_str = (s == 0u) ? need : s;
      const bool exp_ok = (exp_str >= need);
      ++pairs;
      const bool r_ok =
          (intfer::gemm::fold_readable(dummy, s, nOp) != nullptr);
      if (r_ok != exp_ok) {
        std::printf("  FAIL contract nOp=%-4u stride=%-6u reader accepts=%d but "
                    "the writer would accept=%d\n",
                    nOp, s, (int)r_ok, (int)exp_ok);
        ++cbad;
      }
      if (intfer::gemm::fold_row_stride(s, nOp) != exp_str) {
        std::printf("  FAIL contract nOp=%-4u stride=%-6u resolved %u, writer "
                    "resolves %u\n",
                    nOp, s, intfer::gemm::fold_row_stride(s, nOp), exp_str);
        ++cbad;
      }
      if (intfer::gemm::fold_readable(nullptr, s, nOp) != nullptr) {
        std::printf("  FAIL contract nOp=%-4u stride=%-6u null usm accepted\n",
                    nOp, s);
        ++cbad;
      }
    }
  }
  std::printf("  contract: %llu (nOp,stride) pairs, writer and reader agree on "
              "every one%s\n",
              (unsigned long long)pairs, cbad ? " -- NO" : "");
  ok = ok && (cbad == 0);

  // ---- (1b) INDEX rule, the cheap half of the S4 coverage -------------
  // The stride contract above is structurally blind to the kernel's index
  // arithmetic: mutation S4 (scripts/mutate_fold_matrix.sh) drops the `+ nOp`
  // that selects the Sfull half and the stride check still reports 80/80
  // agreement, because the index is a separate expression. Now that every read
  // goes through fold_index_sfull/fold_index_s0, that expression has ONE home
  // and can be gated without a device launch. Three properties, all of which S4
  // breaks: the two halves are exactly nOp apart; S0 lands at the row's base;
  // neither half ever escapes into the next row's block.
  std::uint64_t ibad = 0, ipairs = 0;
  for (std::uint32_t nOp : nOps) {
    const std::uint32_t need = 2u * nOp;
    for (std::uint32_t st : {need, need + 7u, 2u * need}) {
      for (std::size_t m : {std::size_t(0), std::size_t(1), std::size_t(7),
                            std::size_t(4095)}) {
        for (std::uint32_t o : {0u, nOp / 2u, nOp - 1u}) {
          const std::size_t s0 = intfer::gemm::fold_index_s0(m, st, o);
          const std::size_t sf = intfer::gemm::fold_index_sfull(m, st, nOp, o);
          const std::size_t base = m * (std::size_t)st;
          ++ipairs;
          if (sf - s0 != (std::size_t)nOp) {
            std::printf("  FAIL index nOp=%-4u stride=%-5u m=%-5zu o=%-4u "
                        "Sfull-S0=%zu, expected %u\n",
                        nOp, st, m, o, sf - s0, nOp);
            ++ibad;
          }
          if (s0 != base + (std::size_t)o || sf != base + nOp + (std::size_t)o) {
            std::printf("  FAIL index nOp=%-4u stride=%-5u m=%-5zu o=%-4u "
                        "S0=%zu Sfull=%zu expected base=%zu\n",
                        nOp, st, m, o, s0, sf, base);
            ++ibad;
          }
          // Must stay inside this row's 2*nOp block, never spill into the next.
          if (s0 - base >= (std::size_t)nOp || sf - base >= (std::size_t)need) {
            std::printf("  FAIL index nOp=%-4u stride=%-5u m=%-5zu o=%-4u "
                        "escapes its row block\n",
                        nOp, st, m, o);
            ++ibad;
          }
        }
      }
    }
  }
  std::printf("  index:    %llu (m,stride,nOp,o) tuples, halves %s%s\n",
              (unsigned long long)ipairs, ibad ? "WRONG -- NO" : "correct",
              ibad ? "" : " (Sfull-S0 == nOp, neither escapes its row)");
  ok = ok && (ibad == 0);
  bad += ibad;

  // ---- (2) device-level readback identity --------------------------
  // M=4, N=16, K=32, INT4/G64 -> nOp=1, so each row owns exactly one S0/Sfull
  // pair and the default-stride case is the regression under test.
  constexpr std::uint32_t M = 4, N = 16, K = 32, G = 64, GA = 64;
  const std::uint32_t nOp = K / OP, nGA = (K + GA - 1) / GA;
  const std::uint32_t nG = (K + G - 1) / G;

  // repack_dpas412 reads u4 codes PACKED (two per byte, low nibble first).
  std::vector<std::uint8_t> row_codes((size_t)N * K / 2);
  std::uint32_t seed = 24680u;
  for (auto& b : row_codes) {
    seed = seed * 1664525u + 1013904223u;
    b = (std::uint8_t)((seed >> 16) & 0xFF);
  }
  std::vector<std::uint8_t> code;
  if (!intfer::gemm::repack_dpas412(row_codes.data(), N, K, 4, code)) {
    std::printf("  FAIL readback: repack_dpas412 refused N=%u K=%u\n", N, K);
    ++bad;
    return false;
  }

  auto* code_d = sycl::malloc_shared<std::uint8_t>(code.size(), q);
  auto* scale_d = sycl::malloc_shared<std::uint16_t>((size_t)N * nG, q);
  auto* act_d = sycl::malloc_shared<std::int8_t>((size_t)M * K, q);
  auto* as_d = sycl::malloc_shared<float>((size_t)M * nGA, q);
  auto* fold_d = sycl::malloc_shared<std::int32_t>((size_t)M * 2 * nOp, q);
  auto* out_base = sycl::malloc_shared<std::uint16_t>((size_t)M * N, q);
  auto* out_fold = sycl::malloc_shared<std::uint16_t>((size_t)M * N, q);
  if (!code_d || !scale_d || !act_d || !as_d || !fold_d || !out_base ||
      !out_fold) {
    std::printf("  FAIL readback: alloc failed\n");
    ++bad;
    return false;
  }
  std::memcpy(code_d, code.data(), code.size());
  for (std::uint32_t i = 0; i < N * nG; ++i) {
    seed = seed * 1664525u + 1013904223u;
    scale_d[i] = (std::uint16_t)((seed >> 16) & 0xFFFF);  // raw FP16 bits
  }
  for (std::uint32_t m = 0; m < M; ++m)
    for (std::uint32_t k = 0; k < OP; ++k)
      act_d[(size_t)m * K + k] =
          (std::int8_t)(k < 16 ? (int)(m + 1) : -(int)(m + 1));
  for (std::uint32_t m = 0; m < M * nGA; ++m)
    as_d[m] = 0.01f + 0.002f * (float)m;
  for (std::uint32_t m = 0; m < M; ++m) {
    fold_d[2 * m + 0] = (std::int32_t)(16 * (m + 1));  // S0 = 16*(m+1)
    fold_d[2 * m + 1] = 0;                            // Sfull = 0
  }

  auto run = [&](std::uint16_t* out, const std::int32_t* fold,
                 std::uint32_t fstride) {
    intfer::gemm::Dpas48Params p;
    p.code_dpas = code_d;
    p.scale_usm = scale_d;
    p.act_int8 = act_d;
    p.act_scale = as_d;
    p.alpha = 1.0f / 100.0f;
    p.out_usm = out;
    p.M = M; p.N = N; p.K = K;
    p.group_size = G;
    p.act_group = GA;
    p.bits = 4;
    p.act_stride = K;
    p.as_stride = nGA;
    p.fold_usm = fold;
    p.fold_stride = fstride;
    intfer::gemm::quant_gemm_dpas48(q, p).wait_and_throw();
  };

  run(out_base, nullptr, 0);  // reference: the inline fold
  std::uint64_t rbad = 0;
  struct Regime { const char* name; std::uint32_t stride; };
  const Regime regimes[] = {
      {"correct stride 2*nOp", 2 * nOp},
      {"fold_stride left at 0", 0},
      {"too-small stride 1 (fail closed)", 1},
  };
  for (const Regime& r : regimes) {
    run(out_fold, fold_d, r.stride);
    std::uint32_t diff = 0;
    for (std::uint32_t i = 0; i < M * N; ++i)
      if (out_base[i] != out_fold[i]) ++diff;
    ++devs;
    std::printf("  readback %-34s stride=%-4u %s (%u/%u outputs differ)\n",
                r.name, r.stride, diff ? "FAIL" : "ok", diff, M * N);
    if (diff) ++rbad;
  }

  sycl::free(code_d, q); sycl::free(scale_d, q); sycl::free(act_d, q);
  sycl::free(as_d, q); sycl::free(fold_d, q);
  sycl::free(out_base, q); sycl::free(out_fold, q);
  // Fold these into the caller's violation count. Without this the one-line
  // verdict could read "FAIL (... 0 violations)" — a self-contradictory line
  // that makes a real failure look like a counting bug.
  bad += cbad + rbad;
  return ok && (rbad == 0);
}

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

  std::uint64_t checked = 0, bad = 0, devs = 0;
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

  // The writer cases above cannot reach the readers' stride plumbing, and no
  // end-to-end arm can (the engine always sets both fields). Gate it here.
  std::printf("\n[dpas-fold-probe] reader/writer layout contract:\n");
  std::uint64_t rc_pairs = 0, rc_devs = 0;
  if (!check_reader_contract(q, rc_pairs, rc_devs, bad)) ok = false;
  checked += rc_pairs;
  devs += rc_devs;

  // --- E4 REAL-KERNEL READBACK: quant_gemm_dpas48 vs _2d ------------------
  // The section above measures the contract's tolerance from a SYNTHETIC term
  // generator, and the contract itself says: "When E4 exists, re-derive from the
  // real kernel. Do NOT inherit 4.0e-5 by copy." This is that re-derivation --
  // the two geometries are the SAME template, so any difference is a real
  // difference in the kernel and nothing else.
  //
  // EXPECTED: bit-identical. Each (row, tile) keeps one accumulator and still
  // sums over ops in the same order, so there is nothing to reorder. That is a
  // property of the geometry, and this arm is what turns it from a claim into a
  // measurement -- in particular it would catch a future edit that hoisted a
  // partial sum out of the o loop.
  //
  // RUN OVER EVERY (bits, group) BRANCH, not just u4/G64. bc0b008 re-derived the
  // u4/u8/u2 bodies inside the R template -- they were mechanically moved into a
  // `for r` loop, which is exactly the "rule fixed at one site and re-derived at
  // another" shape this repo keeps auditing. Only u4/G64 has a device identity
  // readback here; the u8 body and the u2 body -- and especially the u2 **G16
  // split** (two DPAS calls plus separate S0/Sf folds per op) -- would otherwise
  // be untested branches inside a template that is on by default in the E4
  // experiment. Each combo is a separate labelled line, so a failure names the
  // branch instead of reporting one aggregate.
  std::printf("\n[dpas-fold-probe] E4 real-kernel readback "
              "(dpas48 R=1 vs dpas48_2d R=2, bit-exact expected):\n");
  std::uint64_t e4_dev_pairs = 0;
  auto e4_readback = [&](std::uint32_t bits, std::uint32_t G) {
    // Counted HERE, not at the call site: the verdict line prints `devs` as
    // "device readback regimes", and a total written beside the calls is a
    // hand-count that goes stale the moment a combo is added -- which is the
    // AUDIT #30 shape (a number in prose/behaviour that nothing re-derives).
    ++devs;
    const std::uint32_t M = 4, N = 64, K = 512, GA = 64;
    const std::uint32_t nOp = K / 32, nG = K / G, nGA = K / GA;
    const std::uint32_t op_bytes = 32 * 16 * bits / 8;
    // The kernel indexes B[(t*nOp + o)*op_bytes] with t < nT = N/TN, so the buffer
    // is nT tiles wide. Sizing it by N over-allocates by TN (16x here).
    const std::uint32_t nT = N / 16;
    const std::size_t code_sz = (std::size_t)nT * nOp * op_bytes;
    std::vector<std::uint8_t> code(code_sz);
    std::uint32_t seed = 12345u;
    for (std::size_t i = 0; i < code_sz; ++i) {
      seed = seed * 1664525u + 1013904223u;
      // Keep the code bytes inside the range the DPAS path expects for `bits`.
      code[i] = (std::uint8_t)((seed >> 24) & ((1u << bits) - 1u));
    }
    std::vector<std::uint16_t> scale((std::size_t)N * nG);
    for (std::uint32_t i = 0; i < N * nG; ++i) {
      seed = seed * 1664525u + 1013904223u;
      scale[i] = (std::uint16_t)((seed >> 16) & 0xFFFF);
    }
    std::vector<std::int8_t> act((std::size_t)M * K);
    for (std::uint32_t m = 0; m < M; ++m)
      for (std::uint32_t k = 0; k < K; ++k) {
        seed = seed * 1664525u + 1013904223u;
        act[(std::size_t)m * K + k] =
            (std::int8_t)((int)((seed >> 24) & 0xFF) - 128);
      }
    std::vector<float> as((std::size_t)M * nGA);
    for (std::uint32_t i = 0; i < M * nGA; ++i) {
      seed = seed * 1664525u + 1013904223u;
      as[i] = 0.01f + 0.0005f * (float)((seed >> 16) & 0x3FF);
    }

    // malloc_SHARED, not malloc_device: every other buffer in this probe is
    // shared, and on this Level Zero driver device-only USM is not host-
    // accessible -- memcpy() into it segfaults. Measured, not guessed: the
    // first run of this arm died exactly at the code memcpy.
    auto *code_d = sycl::malloc_shared<std::uint8_t>(code_sz, q);
    auto *scale_d = sycl::malloc_shared<std::uint16_t>(scale.size(), q);
    auto *act_d = sycl::malloc_shared<std::int8_t>(act.size(), q);
    auto *as_d = sycl::malloc_shared<float>(as.size(), q);
    auto *o1 = sycl::malloc_shared<std::uint16_t>((std::size_t)M * N, q);
    auto *o2 = sycl::malloc_shared<std::uint16_t>((std::size_t)M * N, q);
    std::memcpy(code_d, code.data(), code_sz);
    std::memcpy(scale_d, scale.data(), scale.size() * 2);
    std::memcpy(act_d, act.data(), act.size());
    std::memcpy(as_d, as.data(), as.size() * 4);

    intfer::gemm::Dpas48Params p;
    p.code_dpas = code_d; p.scale_usm = scale_d; p.act_int8 = act_d;
    p.act_scale = as_d; p.alpha = 1.0f / 64.0f; p.M = M; p.N = N; p.K = K;
    p.group_size = G; p.act_group = GA; p.bits = bits;
    p.act_stride = K; p.as_stride = nGA;
    p.out_usm = o1;
    intfer::gemm::quant_gemm_dpas48(q, p).wait_and_throw();
    p.out_usm = o2;
    intfer::gemm::quant_gemm_dpas48_2d(q, p).wait_and_throw();

    std::vector<std::uint16_t> h1((std::size_t)M * N), h2((std::size_t)M * N);
    std::memcpy(h1.data(), o1, h1.size() * 2);
    std::memcpy(h2.data(), o2, h2.size() * 2);
    std::uint64_t diff = 0;
    for (std::size_t i = 0; i < h1.size(); ++i)
      if (h1[i] != h2[i]) {
        if (diff < 4)
          std::printf("  FAIL  E4[bits=%u,G=%u] readback differs at %zu: 0x%04x vs "
                      "0x%04x\n", bits, G, i, h1[i], h2[i]);
        ++diff;
      }
    e4_dev_pairs += (std::uint64_t)M * N;
    // NON-TRIVIALITY. "256 outputs bit-identical" is ALSO what this arm would
    // print if BOTH kernels wrote nothing, or wrote a constant field: two
    // buffers of untouched shared memory compare equal forever. The comparison
    // needs a subject before its verdict means anything -- AUDIT #20's rule
    // ("found nothing to look at" and "found nothing wrong" are different
    // claims), applied to a device readback. Count the distinct outputs of the
    // R=1 side and require a real spread of values; the count is printed on the
    // ok line so the evidence is on the record rather than asserted in prose.
    std::vector<std::uint16_t> uniq(h1);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    const std::size_t e4_distinct = uniq.size();
    const bool e4_no_subject = (e4_distinct < 8);
    const std::uint64_t pairs = (std::uint64_t)M * N;
    if (e4_no_subject) {
      std::printf("  FAIL  E4[bits=%u,G=%u] readback has NO SUBJECT: only %zu "
                  "distinct value(s) across %llu outputs, so 'bit-identical' is "
                  "vacuous -- both kernels may have written a constant field\n",
                  bits, G, e4_distinct, (unsigned long long)pairs);
      ok = false;
    } else if (diff == 0) {
      std::printf("  ok    [bits=%u,G=%u] %llu outputs bit-identical between R=1 "
                  "and R=2 (M=%u, N=%u, K=%u, %zu distinct values -- the "
                  "comparison HAS a subject)\n",
                  bits, G, (unsigned long long)pairs, M, N, K, e4_distinct);
    } else {
      std::printf("  FAIL  E4[bits=%u,G=%u] readback: %llu of %llu outputs differ "
                  "-- the row block changed the arithmetic\n",
                  bits, G, (unsigned long long)diff, (unsigned long long)pairs);
      ok = false;
    }
    // The M<2 decline is a PRECONDITION, so ASSERT it rather than describe it.
    // ONCE, on the first combo -- it does not depend on bits or G, and repeating
    // it four times would print four lines describing one fact.
    // a 1-row call must not submit, which is what makes E4 inert at decode and
    // therefore attributable. Declining is inert on the M=1 path (the kernel is
    // mathematically defined there -- it is just not a row BLOCK), so it cannot
    // be distinguished by its output; the sentinel prerequisite is the only
    // reliable signal, and the bare `sycl::event{}` comparison this first used
    // is exactly the trap the NEG cases above already record.
    if (bits == 4 && G == 64) {
      const std::uint16_t kOutSentinel = 0xBEEF;
      std::vector<std::uint16_t> fill((std::size_t)M * N, kOutSentinel);
      std::memcpy(o1, fill.data(), fill.size() * 2);
      sycl::event prereq = q.submit([&](sycl::handler& h) {
        h.single_task<class e4_prereq_probe>([=]() {});
      });
      p.M = 1; p.out_usm = o1;
      sycl::event e1 = intfer::gemm::quant_gemm_dpas48_2d(q, p, prereq);
      e1.wait_and_throw();
      std::vector<std::uint16_t> chk((std::size_t)M * N);
      std::memcpy(chk.data(), o1, chk.size() * 2);
      std::uint32_t touched = 0;
      for (std::uint16_t v : chk)
        if (v != kOutSentinel) ++touched;
      if (e1 != prereq) {
        std::printf("  FAIL  E4 did not DECLINE at M=1 (it submitted anyway); it must"
                    " be inert across all of decode\n");
        ok = false;
      } else if (touched != 0) {
        std::printf("  FAIL  E4 declined at M=1 but wrote %u slots; a decline must"
                    " write nothing\n", touched);
        ok = false;
      } else {
        std::printf("  ok    DECLINES at M=1 (sentinel prerequisite returned"
                    " untouched, 0 slots written) -- inert across all of decode\n");
      }
    }

    sycl::free(code_d, q); sycl::free(scale_d, q); sycl::free(act_d, q);
    sycl::free(as_d, q); sycl::free(o1, q); sycl::free(o2, q);
  };
  // Every branch the template carries: u4/G64, u8/G64, u2/G64 (no split) and
  // u2/G16 (the masked-A split with separate S0/Sf folds). Read in full off the
  // kernel, not off this list: quant_gemm.cpp sets `need_split = (bits==2 &&
  // G==16)` and picks the dpas_argument_type from `bits`. If a combo is added to
  // the kernel and not here, it is untested -- say so rather than assume.
  //
  // SCOPE, stated rather than implied: R=1 and R=2 are the SAME template, so an
  // edit applied symmetrically to both cancels in this comparison. What the arm
  // proves is exactly what E4 can break: the per-ROW addressing (`m0 = mg*R`,
  // `m = m0 + r`, the `As[r]`/`Aq[r]` reads and the store's row index) and the
  // `m >= M` mask. It is NOT a correctness oracle for the arithmetic -- that is
  // the shipped kernel's business, and mutating `need_split` (which changes both
  // geometries identically) correctly leaves this arm green.
  e4_readback(4u, 64u);
  e4_readback(8u, 64u);
  e4_readback(2u, 64u);
  e4_readback(2u, 16u);
  checked += e4_dev_pairs;

  // --- THE OTHER DECLINE CONVENTION, ASSERTED -------------------------------
  // engine.cpp's decline test is `(kernel_evt == sycl::event{}) ||
  // (kernel_evt == dep)`, and it has to be a UNION because the two callees in
  // that ternary use DIFFERENT conventions:
  //   quant_gemm_dpas48_2d  -> `return dep`        (proved above, sentinel test)
  //   quant_gemm_dpas4_row  -> `return sycl::event{}`
  // Until now only the first was a MEASURED fact and the second was a comment,
  // which is the recorded "a rule that exists only in a document is not a check"
  // shape. So measure it, both ways:
  //   decline  -> returns an EMPTY event (and does NOT submit: the output buffer
  //               keeps its sentinel), which is the shape the engine's first test
  //               catches;
  //   accepted -> returns a FRESH event that is neither `dep` nor empty, which is
  //               what makes the union safe: a real submit can satisfy neither
  //               side of it, so the fallback cannot fire spuriously.
  // A one-sided version of this is worthless -- a test that only proves the
  // kernel declines would also pass if it returned `dep`.
  std::printf("\n[dpas-fold-probe] decline conventions (both, so the engine's union"
              " test is a fact and not a comment):\n");
  {
    constexpr std::uint32_t M = 4, N = 64, K = 512;
    const std::uint32_t nOp = K / 32, nG = K / 64, nGA = K / 64;
    const std::uint32_t nT = N / 16, op_bytes = 32 * 16 * 4 / 8;
    const std::size_t code_sz = (std::size_t)nT * nOp * op_bytes;
    std::vector<std::uint8_t> code(code_sz);
    std::vector<std::uint16_t> scale((std::size_t)N * nG);
    std::vector<std::int8_t> act((std::size_t)M * K);
    std::vector<float> as((std::size_t)M * nGA);
    std::uint32_t seed = 987654321u;
    for (std::size_t i = 0; i < code_sz; ++i) {
      seed = seed * 1664525u + 1013904223u;
      code[i] = (std::uint8_t)((seed >> 24) & 0x0F);
    }
    for (std::size_t i = 0; i < scale.size(); ++i) {
      seed = seed * 1664525u + 1013904223u;
      scale[i] = (std::uint16_t)((seed >> 16) & 0xFFFF);
    }
    for (std::size_t i = 0; i < act.size(); ++i) {
      seed = seed * 1664525u + 1013904223u;
      act[i] = (std::int8_t)((int)((seed >> 24) & 0xFF) - 128);
    }
    for (std::size_t i = 0; i < as.size(); ++i) as[i] = 0.02f;

    auto *code_d = sycl::malloc_shared<std::uint8_t>(code_sz, q);
    auto *scale_d = sycl::malloc_shared<std::uint16_t>(scale.size(), q);
    auto *act_d = sycl::malloc_shared<std::int8_t>(act.size(), q);
    auto *as_d = sycl::malloc_shared<float>(as.size(), q);
    auto *o1 = sycl::malloc_shared<std::uint16_t>((std::size_t)M * N, q);
    std::memcpy(code_d, code.data(), code_sz);
    std::memcpy(scale_d, scale.data(), scale.size() * 2);
    std::memcpy(act_d, act.data(), act.size());
    std::memcpy(as_d, as.data(), as.size() * 4);

    intfer::gemm::Dpas48Params p;
    p.code_dpas = code_d; p.scale_usm = scale_d; p.act_int8 = act_d;
    p.act_scale = as_d; p.alpha = 1.0f / 64.0f; p.M = M; p.N = N; p.K = K;
    p.act_group = 64; p.act_stride = K; p.as_stride = nGA; p.out_usm = o1;

    sycl::event dep = q.submit([&](sycl::handler& h) {
      h.single_task<class dc_prereq_probe>([=]() {});
    });

    // (a) DECLINE: bits=8 is outside the row kernel's contract (it requires
    // bits==4 and G==64), so it must return an empty event and write nothing.
    const std::uint16_t kSent = 0xC0DE;
    std::vector<std::uint16_t> fill((std::size_t)M * N, kSent);
    std::memcpy(o1, fill.data(), fill.size() * 2);
    p.bits = 8; p.group_size = 64;
    sycl::event e_dec = intfer::gemm::quant_gemm_dpas4_row(q, p, dep);
    e_dec.wait_and_throw();
    std::uint32_t touched_dec = 0;
    {
      std::vector<std::uint16_t> chk((std::size_t)M * N);
      std::memcpy(chk.data(), o1, chk.size() * 2);
      for (std::uint16_t v : chk)
        if (v != kSent) ++touched_dec;
    }
    // (b) ACCEPT: bits=4/G=64 is the row kernel's contract, so it must submit
    // and return a FRESH event -- neither `dep` and not empty. This is the half
    // that makes the union safe.
    p.bits = 4; p.group_size = 64;
    sycl::event e_ok = intfer::gemm::quant_gemm_dpas4_row(q, p, dep);
    e_ok.wait_and_throw();

    // The MEASURED, implementation-dependent fact, printed rather than asserted:
    // whether an event a callee produced by `return sycl::event{}` compares equal
    // to a bare one built HERE. It was written out of this section because the
    // answer is NO on this toolchain, and a probe that quietly required "yes"
    // would go red on the next compiler. The consequence is what matters and is
    // asserted below: engine.cpp must not depend on it.
    const bool null_cmp_works = (e_dec == sycl::event{});
    std::printf("  ok    MEASURED: a callee's `return sycl::event{}` %s compare "
                "equal to a bare sycl::event{} built here. engine.cpp therefore "
                "may NOT rely on that test.\n",
                null_cmp_works ? "DOES" : "does NOT");
    // What IS asserted, both directions:
    //   decline -> wrote nothing at all (the only reliable, side-effect-shaped
    //              signal), and returned `dep` unchanged;
    //   accept  -> returned an event that is neither `dep` nor empty, so the
    //              engine's `== dep` test cannot fire on a real submit.
    // And the rule itself is checked against OBSERVED behaviour in both
    // directions, so the engine's eligibility predicate can never drift from the
    // kernel without this going red -- that is the whole point of giving the
    // predicate one home.
    const bool dec_ok = (touched_dec == 0) && (e_dec == dep);
    const bool acc_ok = (e_ok != dep) && (e_ok != sycl::event{});
    const bool rule_ok = intfer::gemm::dpas4_row_eligible(8u, 64u) == false &&
                         intfer::gemm::dpas4_row_eligible(4u, 64u) == true &&
                         intfer::gemm::dpas4_row_eligible(4u, 32u) == false;
    std::printf("  %s  quant_gemm_dpas4_row DECLINES (bits=8): wrote %u slot(s) and "
                "returned `dep` UNCHANGED -- the shape engine.cpp's `== dep` "
                "catches\n", dec_ok ? "ok   " : "FAIL ", touched_dec);
    std::printf("  %s  quant_gemm_dpas4_row ACCEPTS  (bits=4/G=64): returned an "
                "event that is neither `dep` nor empty, so the test cannot fire "
                "spuriously\n", acc_ok ? "ok   " : "FAIL ");
    std::printf("  %s  dpas4_row_eligible() agrees with OBSERVED behaviour in both "
                "directions (8/64 declines, 4/64 accepts), so the engine's copy "
                "cannot drift\n", rule_ok ? "ok   " : "FAIL ");
    if (!dec_ok) ok = false;
    if (!acc_ok) ok = false;
    if (!rule_ok) ok = false;

    sycl::free(code_d, q); sycl::free(scale_d, q); sycl::free(act_d, q);
    sycl::free(as_d, q); sycl::free(o1, q);
  }

  // NOTE: the one-line verdict is printed at the END, after E4. Printing it
  // here would announce PASS and then let a later section flip it to FAIL, so
  // the line a human reads would disagree with the exit code.

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

  std::printf("\n[dpas-fold-probe] %s (%llu (row,op)+contract pairs compared, "
              "%llu device readback regimes, %llu violations)\n",
              ok ? "PASS" : "FAIL", (unsigned long long)checked,
              (unsigned long long)devs, (unsigned long long)bad);

  if (!ok) {
    std::printf(
        "[dpas-fold-probe] The hoist is NOT value-identical (or failed to\n"
        "  decline), or the E4 tolerance contract was violated.\n"
        "  INTFER_DPAS48_FOLD must stay OFF; do not start E4 on a bad bound.\n");
  }
  return ok ? 0 : 1;
}
