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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  if (!ok) {
    std::printf(
        "[dpas-fold-probe] The hoist is NOT value-identical (or failed to\n"
        "  decline). INTFER_DPAS48_FOLD must stay OFF.\n");
  }
  return ok ? 0 : 1;
}
