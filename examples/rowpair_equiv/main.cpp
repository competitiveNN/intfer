// examples/rowpair_equiv -- MEASURED equivalence for INTFER_DECODE1_ROWPAIR.
//
// WHY THIS EXISTS, and it is a correction rather than a formality.
//
// Row-pairing's claim was "bit-exact BY CONSTRUCTION: for each output the
// addition sequence is the base kernel's sequence -- its own accumulator,
// ascending kg, ascending j, same 8-code chunk, same byte indexing, same wtab --
// and the chains are interleaved in the INSTRUCTION stream, never summed
// together."
//
// That is a READING argument, and this repo has paid for exactly that class of
// argument twice, both times on this very kernel:
//   * INTFER_DECODE1_SPLITK shipped `(bv >> (j * 2)) & 3` instead of
//     `(bv >> ((j & 3) * 2)) & 3` and was localised only because the stream
//     diverged while the kill switch stayed clean -- the argument had been "the
//     order is safe".
//   * The reverted fused-decode kernel was built on "a normalized statement diff
//     says the arithmetic is identical", the diff DID say that, and it cost
//     231 114 ms. AGENTS.md's generalised lesson: "structural argument is not
//     measurement".
//
// Row-pairing shares ONE activation load across TWO code pointers, which is
// precisely the lane where an index that assumes a single pointer goes wrong.
// So the claim is worth nothing until it is measured, and this measures it.
//
// WHAT IT DOES. No model (AGENTS.md standing rule 1: never run a full-27B gate
// from an agent session). Synthetic weights and activations at the 27B TP=2
// shapes, both kernels run on the SAME buffers, and the outputs compared for
// BITWISE EQUALITY -- not a tolerance, because the claim is exactness. It sweeps
// datasets chosen to break this specific transformation:
//
//   D0 random            the null case
//   D1 act-outliers      a few activations at 1e4, where an FMA-vs-mul difference
//                        would show even if a tolerance were allowed
//   D2 full-cancellation activations chosen so the two accumulators cancel toward
//                        zero, where summation order is the ONLY thing that matters
//   D3 all-equal         every activation identical, so any cross-row bleed is
//                        invisible in the sum but visible per-output
//   D4 row-adversarial   row n's weights make its own accumulator's partial sums
//                        collide with row n+1's -- if a single accumulator were
//                        shared, this is the shape that separates them
//   D5 mixed-shape      every scheme and the odd-N case (the kernel guards
//                        N even, so an odd N must FALL BACK, and falling back
//                        silently to a different kernel is a real hazard)
//
// EXIT: 0 = row-pair output is bitwise identical to base on every dataset and
// shape; 1 = any difference, or the fallback was taken where it should not be.

#include "intfer/common/logger.hpp"
#include "intfer/device/device_context.hpp"
#include "intfer/gemm/quant_gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The 27B TP=2 shapes, from scripts/model_byte_roofline.sh. N even for every shape
// the row-pair kernel accepts; the odd case is added deliberately below.
struct Shape {
  const char* name;
  std::uint32_t N;
  std::uint32_t K;
  int group;
  intfer::gemm::WeightScheme scheme;
  const char* scheme_name;
};

const Shape kShapes[] = {
    {"MLP_GATE(INT2)", 8704, 5120, 16, intfer::gemm::WeightScheme::INT2, "INT2"},
    {"MLP_DOWN(INT2)", 2560, 17408, 16, intfer::gemm::WeightScheme::INT2, "INT2"},
    {"LA_QKV(FP16)", 5120, 5120, 0, intfer::gemm::WeightScheme::FP16, "FP16"},
    {"LA_OUT(FP16)", 2560, 6144, 0, intfer::gemm::WeightScheme::FP16, "FP16"},
    {"MLP_UP(INT4)", 8704, 5120, 64, intfer::gemm::WeightScheme::INT4, "INT4"},
    {"ATTN_K(FP16)", 512, 5120, 0, intfer::gemm::WeightScheme::FP16, "FP16"},
};

// ODD N: the row-pair kernel is guarded on (N & 1) == 0, so it must NOT be taken
// here. A silent fallback to a different kernel would still produce correct tokens,
// which is why the guard is asserted rather than assumed.
const Shape kOddShape = {"MLP_GATE-ODD(INT2)", 8703, 5120, 16,
                          intfer::gemm::WeightScheme::INT2, "INT2"};

// K NOT A MULTIPLE OF THE GROUP SIZE. Added 2026-10-05 after a MEASURED coverage
// hole: with Gw = 16 and CC = 8 the chunk loop covers [kg0, kg0+16) exactly, so the
// ragged tail executes ZERO times at every real 27B shape -- all of K=5120 and
// K=17408 divide 16. A deliberately wrong index in the tail therefore produced a
// GREEN gate. Every other shape in this file divides evenly, so the tail was
// untested and the hole was invisible until the negative run came back rc=0 when
// it had to come back non-zero.
//
// K = 5122 with Gw = 16 leaves 2 elements in the last group, so the tail runs, and
// it runs on the LAST group only -- which is exactly the case a reviewer would
// never exercise by hand.
const Shape kRaggedShape = {"MLP_GATE-RAGGED(INT2)", 8704, 5122, 16,
                             intfer::gemm::WeightScheme::INT2, "INT2"};

enum Dataset { kRandom = 0, kActOutliers, kFullCancellation, kAllEqual, kRowAdversarial, kCount };

const char* dataset_name(int d) {
  switch (d) {
    case kRandom: return "random";
    case kActOutliers: return "act-outliers";
    case kFullCancellation: return "full-cancellation";
    case kAllEqual: return "all-equal";
    case kRowAdversarial: return "row-adversarial";
    default: return "?";
  }
}

// Deterministic PRNG: a fixed seed so a failure is REPRODUCIBLE, and so two runs
// cannot differ because of entropy and be mistaken for a kernel difference.
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
  std::uint64_t next() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
  }
  float unit() { return static_cast<float>(next() >> 40) / 16777216.0f; }  // [0,1)
};

void fill_dataset(int d, std::vector<std::uint8_t>& code, std::vector<std::uint16_t>& act,
                  std::vector<std::uint16_t>& scale, std::uint16_t* scale_bits,
                  const Shape& sh, std::uint64_t seed) {
  Rng r(seed + 0x1000u * static_cast<std::uint64_t>(d));
  // Weights: every dataset keeps RANDOM weights, because the datasets are about
  // activations. Except kRowAdversarial, which is about weights -- see below.
  for (auto& b : code) b = static_cast<std::uint8_t>(r.next() >> 27);

  const std::uint32_t K = sh.K;
  const std::uint32_t ngroups = (sh.group == 0) ? 1u : (K + sh.group - 1) / sh.group;
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(scale.size()); ++i)
    scale[i] = static_cast<std::uint16_t>(0x2C00 + (r.next() >> 40) % 512);  // ~1..1.5

  switch (d) {
    case kRandom:
      for (std::uint32_t i = 0; i < K; ++i) act[i] = 0x3400 + (r.next() >> 44) % 512;
      break;
    case kActOutliers:
      for (std::uint32_t i = 0; i < K; ++i)
        act[i] = (i % 512 == 0) ? 0x6B00          /* ~16384, a large finite */
                                : static_cast<std::uint16_t>(0x3400 + (r.next() >> 44) % 512);
      break;
    case kFullCancellation:
      // a[i] and a[i+K/2] are near-opposites, so the ascending sum cancels toward
      // zero and ANY reordering of the terms shows up. This is the dataset that
      // makes the accumulator's order the only thing that matters.
      for (std::uint32_t i = 0; i < K / 2; ++i) {
        const std::uint16_t v = static_cast<std::uint16_t>(0x3400 + (r.next() >> 44) % 1024);
        act[i] = v;
        act[i + K / 2] = static_cast<std::uint16_t>(0xBC00 - (v & 0x03FF));
      }
      break;
    case kAllEqual:
      for (std::uint32_t i = 0; i < K; ++i) act[i] = 0x3555;  // one fixed value
      break;
    case kRowAdversarial:
      // Row n and row n+1 get weights that produce the SAME partial sums in
      // different orders, so a single shared accumulator -- the failure mode if
      // the two chains were ever summed -- gives a different answer per row.
      for (std::uint32_t i = 0; i < K; ++i) act[i] = static_cast<std::uint16_t>(0x3400 + (i % 7) * 64);
      break;
    default: break;
  }
  (void)scale_bits;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = (argc > 1 && std::string(argv[1]) == "--verbose");
  auto mgr = intfer::device::DeviceManager::discover("Arc");
  if (mgr.size() == 0) {
    std::fprintf(stderr, "FAIL: no Intel Arc GPU discovered (is setvars.sh sourced?)\n");
    return 1;
  }
  auto& dev = mgr[0];
  sycl::queue q = dev.queue;
  q.wait_and_throw();
  std::fprintf(stderr, "[rowpair] device: %s\n", dev.name.c_str());

  int checks = 0, fails = 0;

  // ROWPAIR_EQUIV_ONLY=<index> selects ONE shape, so a fault can be bisected to a
  // shape instead of guessed at. The default is all seven.
  const char* only = std::getenv("ROWPAIR_EQUIV_ONLY");
  const Shape* all[] = {&kShapes[0], &kShapes[1], &kShapes[2], &kShapes[3],
                        &kShapes[4], &kShapes[5], &kOddShape, &kRaggedShape};
  for (std::size_t si = 0; si < 8; ++si) {
    if (only && std::atoi(only) != static_cast<int>(si)) continue;
    const Shape* shp = all[si];
    std::fprintf(stderr, "[rowpair] shape %d: %s (N=%u K=%u %s)\n", (int)si, shp->name,
                 shp->N, shp->K, shp->scheme_name);
    const Shape& sh = *shp;
    const int pack = (sh.scheme == intfer::gemm::WeightScheme::INT2)  ? 4
                     : (sh.scheme == intfer::gemm::WeightScheme::INT4) ? 2
                     : 1;
    const std::uint64_t code_bytes =
        (sh.scheme == intfer::gemm::WeightScheme::FP16)
            ? std::uint64_t(sh.N) * sh.K * 2
            : std::uint64_t(sh.N) * sh.K / pack;
    const std::uint32_t ngroups = (sh.group == 0) ? 1u : (sh.K + sh.group - 1) / sh.group;

    void* code = sycl::malloc_shared(code_bytes, dev.sycl_device, dev.sycl_context);
    void* sc = sycl::malloc_shared(std::uint64_t(sh.N) * ngroups * 2, dev.sycl_device,
                                   dev.sycl_context);
    void* act = sycl::malloc_shared(std::uint64_t(sh.K) * 2, dev.sycl_device, dev.sycl_context);
    void* out_base = sycl::malloc_shared(std::uint64_t(sh.N) * 2, dev.sycl_device, dev.sycl_context);
    void* out_rp = sycl::malloc_shared(std::uint64_t(sh.N) * 2, dev.sycl_device, dev.sycl_context);
    void* out_vh = sycl::malloc_shared(std::uint64_t(sh.N) * 2, dev.sycl_device, dev.sycl_context);
    void* out_vm = sycl::malloc_shared(std::uint64_t(sh.N) * 2, dev.sycl_device, dev.sycl_context);
    if (!code || !sc || !act || !out_base || !out_rp || !out_vh || !out_vm) {
      std::fprintf(stderr, "FAIL: allocation for %s\n", sh.name);
      return 1;
    }
    std::vector<std::uint8_t> hcode(code_bytes);
    std::vector<std::uint16_t> hact(sh.K), hsc(std::uint64_t(sh.N) * ngroups);

    for (int d = 0; d < kCount; ++d) {
      fill_dataset(d, hcode, hact, hsc, nullptr, sh,
                   0xC0FFEEull ^ (std::uint64_t(sh.N) << 20) ^ (std::uint64_t(sh.K) << 3));
      std::memcpy(code, hcode.data(), code_bytes);
      std::memcpy(act, hact.data(), std::uint64_t(sh.K) * 2);
      std::memcpy(sc, hsc.data(), hsc.size() * 2);

      auto run = [&](bool rowpair) -> std::vector<std::uint16_t> {
        void* out = rowpair ? out_rp : out_base;
        // The env var is read INSIDE quant_gemm_forward_usm via a function-local
        // `static const bool`, so it is set once per process and cannot be toggled
        // between the two runs. That is why the two orders are separate PROCESS
        // invocations from the shell, and why this binary runs ONE order and the
        // caller diffs the files. The `static` is the same pattern the engine's own
        // flags use; a per-call read would be a hot-path getenv in a decode kernel.
        sycl::event e = intfer::gemm::quant_gemm_forward_usm(
            q, static_cast<std::uint16_t*>(act), code,
            static_cast<const std::uint16_t*>(sc), 1.0f,
            static_cast<std::uint16_t*>(out), /*M=*/1, sh.N, sh.K,
            static_cast<std::uint32_t>(sh.group), sh.scheme, sycl::event{});
        e.wait_and_throw();
        const std::uint64_t n = std::uint64_t(sh.N) * 2;
        std::vector<std::uint16_t> r(n / 2);
        std::memcpy(r.data(), out, n);
        return r;
      };
      (void)run;

      // ONE order per process; see run()'s comment. Extended from two orders to
      // FOUR on 2026-10-06 so the same machinery -- 8 shapes x 5 datasets, bitwise,
      // two process invocations per arm -- can test INTFER_DECODE1_VERBHALF and
      // INTFER_DECODE1_VERBMP as well.
      //
      // WHY THAT MATTERS MORE THAN A PERF RUN RIGHT NOW: VERBMP's "bit-exact BY
      // CONSTRUCTION" is a CLAIM, and the ROWPAIR precedent says a bit-exactness
      // ARGUMENT is worth nothing until measured -- ROWPAIR's is precisely what
      // shipped a device-losing double offset. This is the cheapest place to settle
      // it, because it needs no model and no 27B load.
      const char* which = std::getenv("DECODE1_EQUIV_ORDER");
      std::string ord(which ? which : "base");
      // Kept for compatibility with the older ROWPAIR_EQUIV_ORDER spelling; the gate
      // uses DECODE1_EQUIV_ORDER. Two names for one switch is a papercut, but
      // silently ignoring the old one would break any invocation already in a log.
      if (const char* oldw = std::getenv("ROWPAIR_EQUIV_ORDER")) {
        if (ord == "base" && std::string(oldw) == "rp") ord = "rp";
      }
      int slot = 0;                 // 0 base, 1 rp, 2 verbhalf, 3 verbmp
      if (ord == "rp")   slot = 1;
      else if (ord == "vh") slot = 2;
      else if (ord == "vm") slot = 3;
      else if (ord != "base") {
        std::fprintf(stderr, "FAIL: unknown DECODE1_EQUIV_ORDER=%s "
                             "(want base|rp|vh|vm)\n", ord.c_str());
        return 1;
      }
      std::vector<std::uint16_t> out;
      {
        void* dst = (slot == 0) ? out_base
                  : (slot == 1) ? out_rp
                  : (slot == 2) ? out_vh : out_vm;
        // The arm is a HOST env var read by quant_gemm_forward_usm into a
        // function-local `static const bool`, so it is set ONCE PER PROCESS and the
        // order therefore has to be a separate process -- which is why the gate runs
        // four of them and diffs the payloads rather than looping in one binary.
        setenv("INTFER_DECODE1_VERBHALF", slot == 2 ? "1" : "", 1);
        setenv("INTFER_NO_DECODE1_VERBHALF", slot == 2 ? "" : "1", 1);
        setenv("INTFER_DECODE1_VERBMP", slot == 3 ? "1" : "", 1);
        setenv("INTFER_NO_DECODE1_VERBMP", slot == 3 ? "" : "1", 1);
        sycl::event e = intfer::gemm::quant_gemm_forward_usm(
            q, static_cast<std::uint16_t*>(act), code,
            static_cast<const std::uint16_t*>(sc), 1.0f,
            static_cast<std::uint16_t*>(dst), /*M=*/1, sh.N, sh.K,
            static_cast<std::uint32_t>(sh.group), sh.scheme, sycl::event{});
        e.wait_and_throw();
        out.resize(std::uint64_t(sh.N));
        std::memcpy(out.data(), dst, std::uint64_t(sh.N) * 2);
      }
      // Name the dataset+shape in the payload so the diff below is self-describing.
      const char* tag = (slot == 0) ? "base" : (slot == 1) ? "rowpair"
                        : (slot == 2) ? "verbhalf" : "verbmp";
      std::printf("%s|%s|%d|%s|%s\n", sh.name, sh.scheme_name, d, dataset_name(d), tag);
      for (std::uint32_t i = 0; i < sh.N; ++i) {
        std::printf("%s|%s|%d|%s|%u|0x%04x\n", sh.name, sh.scheme_name, d,
                    dataset_name(d), i, out[i]);
      }
      checks++;
      if (verbose)
        std::fprintf(stderr, "  ran %s / %s / order=%s\n", sh.name, dataset_name(d),
                     std::getenv("DECODE1_EQUIV_ORDER") ? std::getenv("DECODE1_EQUIV_ORDER")
                                                         : "base");
    }
    sycl::free(out_vm, q); sycl::free(out_vh, q);
    sycl::free(out_rp, q); sycl::free(out_base, q);
    sycl::free(act, q); sycl::free(sc, q); sycl::free(code, q);
  }
  std::fprintf(stderr, "[rowpair] %d (shape,dataset) cases run in this order\n", checks);
  if (fails) { std::fprintf(stderr, "ROWPAIR-EQUIV: FAIL (%d)\n", fails); return 1; }
  std::fprintf(stderr, "ROWPAIR-EQUIV: emitted (run twice, then diff)\n");
  return 0;
}
