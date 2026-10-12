// examples/gateup_equiv -- MEASURED equivalence for the fused gate+up GEMM.
//
// WHY THIS EXISTS, and it is a correction rather than a formality.
//
// The fused gate+up kernel (quant_gemm.cpp:2036) claims "bit-exact by
// construction" against two separate quant_gemm_forward_usm calls, and that
// claim is a READING argument. AGENTS.md's standing lesson is that a structural
// argument is not measurement -- this repo has paid for that class of argument
// three times on this very family of kernels, and the operator's 27B run died
// at L3 prefill with UR_RESULT_ERROR_DEVICE_LOST inside a submit whose weight
// indexing double-counted the row offset (quant_gemm.cpp:2078 read
// `wrow = wbase + out_ng * K` while `b0 = (out_ng * K + kg) / pack` was already
// an ABSOLUTE byte offset, so every row past the first read ~4x past the
// shard's end).
//
// The indexing defect lived between the two kernels, so the only thing that
// settles it is a DIFFERENTIAL: run both paths on the SAME buffers and compare
// BITWISE. If the fused kernel's weight indexing is correct they agree on every
// output element; if it drifts they diverge.
//
// WHAT IT DOES. No model (AGENTS.md standing rule 1: never run a full-27B gate
// from an agent session). Synthetic INT2 weights at the 27B TP=2 MLP shapes,
// two separate shard buffers (gate, up) sharing one activation vector, both
// paths run on the SAME buffers, outputs compared for BITWISE EQUALITY -- not a
// tolerance, because the claim is exactness. The datasets are the ones that
// make a shifted weight row visible rather than averaged away.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "intfer/device/device_context.hpp"
#include "intfer/gemm/quant_gemm.hpp"
#include "intfer/l0/level_zero.hpp"

namespace {

// The 27B TP=2 MLP gate/up pair, from scripts/model_byte_roofline.sh. Both share
// K = 5120 (the MLP input width) and group = 16; N differs per projection but
// the fused kernel handles Ng != Nu (ng/un split on out_ng).
struct Shape {
  const char* name;
  std::uint32_t Ng;   // rows the KERNEL is told (the full output dim)
  std::uint32_t Nu;
  std::uint32_t Nl_g; // rows actually ALLOCATED for code/scale (the TP slice)
  std::uint32_t Nl_u;
  std::uint32_t Og_bytes; // OUT buffer rows (full unless isolating the out write)
  std::uint32_t Ou_bytes;
  std::uint32_t K;
  int group;
  std::uint32_t start_g; // first output row the shard holds (0 = TP rank 0)
  std::uint32_t start_u;
};

const Shape kShapes[] = {
    {"MLP_GATE+UP(INT2)", 8704, 8704, 8704, 8704, 8704, 8704, 5120, 16, 0, 0},
    {"MLP_GATE+UP-RAGGED(INT2)", 8704, 8704, 8704, 8704, 8704, 8704, 5122, 16, 0, 0},
    {"MLP_GATE+UP-ODDN(INT2)", 8703, 8705, 8703, 8705, 8703, 8705, 5120, 16, 0, 0},
    // *** THE SLICED SHAPE, and the one that actually reproduces the crash. ***
    // The engine shards the OUTPUT dim N across the 2 TP ranks, so each device
    // holds only its 4352-row slice of the 8704-row gate/up -- but run_mlp_gateup
    // (engine.cpp:1815-1821) passes the FULL Ng/Nu to the kernel, because the
    // kernel's out_ng runs over the whole output and the readback below
    // (engine.cpp:1886-1907) scatters each rank's slice back into mg/mu by
    // `start`. So the kernel reads rows 4352..8703 of a buffer that holds 4352.
    {"MLP_GATE+UP-SLICED(INT2)", 8704, 8704, 4352, 4352, 8704, 8704, 5120, 16, 0, 0},
    // *** THE ISOLATION SHAPE: full code/scale, SLICED out buffers. *** The
    // engine sizes out FULL (ensure_wave_workspace takes Nl = max(Ng, Nu)), so
    // this is NOT a shape the engine produces -- it exists solely to separate
    // the two failure hypotheses. If the crash is the CODE/SCALE over-read this
    // passes; if it is the OUT over-write it still crashes. A single "sliced"
    // shape cannot tell the two apart, and guessing cost a run.
    {"MLP_GATE+UP-SLICED-OUTONLY(INT2)", 8704, 8704, 8704, 8704, 4352, 4352, 5120, 16, 0, 0},
    // *** THE OFFSET SHAPE: a NON-ZERO start_g, so the offset arithmetic is
    // exercised, not merely defaulted to 0. The engine's balanced split puts
    // rank 0 at 0 and rank 1 at 4352; this shape puts the shard at 2048..6400,
    // which no rank ever holds, and the reference computes exactly those rows.
    {"MLP_GATE+UP-SLICED-OFFSET(INT2)", 8704, 8704, 4352, 4352, 8704, 8704, 5120, 16, 2048, 2048},
};

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

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
  std::uint64_t next() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
  }
};

void fill_weights(std::vector<std::uint8_t>& code_g, std::vector<std::uint8_t>& code_u,
                  const Shape& sh, std::uint64_t seed) {
  Rng r(seed);
  for (auto& b : code_g) b = static_cast<std::uint8_t>(r.next() >> 27);
  for (auto& b : code_u) b = static_cast<std::uint8_t>(r.next() >> 27);
}

void fill_scales(std::vector<std::uint16_t>& sg, std::vector<std::uint16_t>& su,
                 std::uint64_t seed) {
  Rng r(seed);
  for (auto& s : sg) s = static_cast<std::uint16_t>(0x2C00 + (r.next() >> 40) % 512);
  for (auto& s : su) s = static_cast<std::uint16_t>(0x2C00 + (r.next() >> 40) % 512);
}

void fill_act(int d, std::vector<std::uint16_t>& act, std::uint32_t K, std::uint64_t seed) {
  Rng r(seed);
  switch (d) {
    case kRandom:
      for (std::uint32_t i = 0; i < K; ++i)
        act[i] = static_cast<std::uint16_t>(0x3400 + (r.next() >> 44) % 512);
      break;
    case kActOutliers:
      for (std::uint32_t i = 0; i < K; ++i)
        act[i] = (i % 512 == 0) ? 0x6B00
                                : static_cast<std::uint16_t>(0x3400 + (r.next() >> 44) % 512);
      break;
    case kFullCancellation:
      for (std::uint32_t i = 0; i < K / 2; ++i) {
        const std::uint16_t v = static_cast<std::uint16_t>(0x3400 + (r.next() >> 44) % 1024);
        act[i] = v;
        act[i + K / 2] = static_cast<std::uint16_t>(0xBC00 - (v & 0x03FF));
      }
      break;
    case kAllEqual:
      for (std::uint32_t i = 0; i < K; ++i) act[i] = 0x3555;
      break;
    case kRowAdversarial:
      for (std::uint32_t i = 0; i < K; ++i)
        act[i] = static_cast<std::uint16_t>(0x3400 + (i % 7) * 64);
      break;
    default: break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* only_ds = std::getenv("GATEUP_EQUIV_ONLY");
  // The SLICED shape crashes the process (that is the point of it). To reach
  // the isolation shape after it, skip SLICED by name rather than by index --
  // the index shifts if a shape is added above.
  const char* skip_named = std::getenv("GATEUP_EQUIV_SKIP_NAMED");
  // zeInit FIRST: the native L0 driver must be initialized before a SYCL L0
  // context exists, or the first malloc_shared fails with
  // UR_RESULT_ERROR_UNINITIALIZED. The engine reaches L0 via
  // DeviceManager::discover -> LevelZero::enumerate_gpus -> initialize(); this
  // harness calls the gemm kernels directly, so it does that itself.
  intfer::l0::LevelZero::initialize();
  // discover() returns by value: keep the manager ALIVE (binding `[0]` to the
  // temporary dangles and the context dies at the end of the statement).
  auto mgr = intfer::device::DeviceManager::discover("Arc");
  if (mgr.size() == 0) {
    std::fprintf(stderr, "FAIL: no Intel Arc GPU discovered\n");
    return 1;
  }
  auto& dev = mgr[0];
  sycl::queue q = dev.queue;
  q.wait_and_throw();
  std::fprintf(stderr, "[gateup] device: %s\n", dev.name.c_str());

  const int pack = 4;  // INT2 = 4 codes/byte
  const auto sc = intfer::gemm::WeightScheme::INT2;
  int checks = 0, fails = 0;

  for (const Shape& sh : kShapes) {
    if (skip_named && std::strcmp(sh.name, skip_named) == 0) continue;
    const std::uint32_t ngroups = (sh.K + sh.group - 1) / sh.group;
  // ALLOCATED rows = the TP slice (Nl); the KERNEL is always told the FULL
    // output dim (Ng/Nu), because run_mlp_gateup passes the full dims and the
    // readback scatters the slice back in by `start`. On the non-sliced shapes
    // Nl == Ng so this is a no-op; on the SLICED shape the kernel reads rows
    // Nl..Ng of a buffer that holds Nl, which is the over-read that crashes.
    const std::uint32_t Nl_g = sh.Nl_g, Nl_u = sh.Nl_u;
    const std::size_t cg_bytes = (std::size_t)Nl_g * sh.K / pack;
    const std::size_t cu_bytes = (std::size_t)Nl_u * sh.K / pack;
    const std::size_t sg_bytes = (std::size_t)Nl_g * ngroups;
    const std::size_t su_bytes = (std::size_t)Nl_u * ngroups;
    const std::size_t act_bytes = sh.K;   // M = 1
    // OUT buffers: the engine sizes them FULL (ensure_wave_workspace takes
    // Nl = max(Ng, Nu)), so the out over-write is NOT the fault here -- the
    // fault is the CODE/SCALE read past the sliced shard. The two SLICED*
    // shapes below differ only in out sizing, which is what isolates it.
    const std::size_t og_bytes = sh.Og_bytes;
    const std::size_t ou_bytes = sh.Ou_bytes;
    // The REFERENCE GEMM must not over-read either, so it runs on FULL-SIZE
    // code/scale buffers (Ng/Nu rows). The shard buffers below hold only Nl.
    const std::size_t cfg_bytes = (std::size_t)sh.Ng * sh.K / pack;
    const std::size_t cuf_bytes = (std::size_t)sh.Nu * sh.K / pack;
    const std::size_t sfg_bytes = (std::size_t)sh.Ng * ngroups;
    const std::size_t suf_bytes = (std::size_t)sh.Nu * ngroups;

    auto* code_g = static_cast<std::uint8_t*>(sycl::malloc_shared(cg_bytes, dev.sycl_device, dev.sycl_context));
    auto* code_u = static_cast<std::uint8_t*>(sycl::malloc_shared(cu_bytes, dev.sycl_device, dev.sycl_context));
    auto* scg = static_cast<std::uint16_t*>(sycl::malloc_shared(sg_bytes * 2, dev.sycl_device, dev.sycl_context));
    auto* scu = static_cast<std::uint16_t*>(sycl::malloc_shared(su_bytes * 2, dev.sycl_device, dev.sycl_context));
    auto* code_full_g = static_cast<std::uint8_t*>(sycl::malloc_shared(cfg_bytes, dev.sycl_device, dev.sycl_context));
    auto* code_full_u = static_cast<std::uint8_t*>(sycl::malloc_shared(cuf_bytes, dev.sycl_device, dev.sycl_context));
    auto* sc_full_g = static_cast<std::uint16_t*>(sycl::malloc_shared(sfg_bytes * 2, dev.sycl_device, dev.sycl_context));
    auto* sc_full_u = static_cast<std::uint16_t*>(sycl::malloc_shared(suf_bytes * 2, dev.sycl_device, dev.sycl_context));
    auto* act = static_cast<std::uint16_t*>(sycl::malloc_shared(act_bytes * 2, dev.sycl_device, dev.sycl_context));
    auto* og = static_cast<std::uint16_t*>(sycl::malloc_shared(og_bytes * 2, dev.sycl_device, dev.sycl_context));
    auto* ou = static_cast<std::uint16_t*>(sycl::malloc_shared(ou_bytes * 2, dev.sycl_device, dev.sycl_context));
    // Reference outputs: computed by two SEPARATE single-output GEMM calls on
    // the FULL-SIZE buffers (the engine's wave fallback also shards, so the
    // honest comparison for the slice shape is slice-vs-slice).
    auto* og_ref = static_cast<std::uint16_t*>(sycl::malloc_shared(og_bytes * 2, dev.sycl_device, dev.sycl_context));
    auto* ou_ref = static_cast<std::uint16_t*>(sycl::malloc_shared(ou_bytes * 2, dev.sycl_device, dev.sycl_context));
    if (!code_g || !code_u || !scg || !scu || !code_full_g || !code_full_u ||
        !sc_full_g || !sc_full_u || !act || !og || !ou || !og_ref || !ou_ref) {
      std::fprintf(stderr, "FAIL: allocation for %s\n", sh.name);
      return 1;
    }

    std::fprintf(stderr, "[gateup] shape %s (Ng=%u Nu=%u Nl_g=%u Nl_u=%u K=%u group=%d)\n",
                 sh.name, sh.Ng, sh.Nu, Nl_g, Nl_u, sh.K, sh.group);

    for (int d = 0; d < kCount; ++d) {
      if (only_ds && std::atoi(only_ds) != d) continue;
      const std::uint64_t seed = 0xC0FFEEull ^ ((std::uint64_t)sh.Ng << 20) ^ ((std::uint64_t)sh.K << 3);
      // FULL-SIZE host vectors first: the reference GEMM must see the whole
      // tensor. The shard's host vectors are then the contiguous slice
      // [start, start+Nl) of these, which is exactly what shard_weight emits.
      std::vector<std::uint8_t> hfg(cfg_bytes), hfu(cuf_bytes);
      std::vector<std::uint16_t> hsfg(sfg_bytes), hsfu(suf_bytes), hact(act_bytes);
      fill_weights(hfg, hfu, sh, seed + 0x1000u * d);
      fill_scales(hsfg, hsfu, seed + 0x1000u * d + 777);
      fill_act(d, hact, sh.K, seed + 0x1000u * d + 999);
      const std::size_t kp = (std::size_t)sh.K / pack;  // code bytes per row
      std::vector<std::uint8_t> hg(cg_bytes), hu(cu_bytes);
      std::vector<std::uint16_t> hsg(sg_bytes), hsu(su_bytes);
      const std::size_t g_off = (std::size_t)sh.start_g * kp;
      const std::size_t u_off = (std::size_t)sh.start_u * kp;
      std::memcpy(hg.data(), hfg.data() + g_off, cg_bytes);
      std::memcpy(hu.data(), hfu.data() + u_off, cu_bytes);
      {
        std::memcpy(hsg.data(), hsfg.data() + (std::size_t)sh.start_g * ngroups, sg_bytes * 2);
        std::memcpy(hsu.data(), hsfu.data() + (std::size_t)sh.start_u * ngroups, su_bytes * 2);
      }
      std::memcpy(code_full_g, hfg.data(), cfg_bytes);
      std::memcpy(code_full_u, hfu.data(), cuf_bytes);
      std::memcpy(sc_full_g, hsfg.data(), sfg_bytes * 2);
      std::memcpy(sc_full_u, hsfu.data(), suf_bytes * 2);
      std::memcpy(code_g, hg.data(), cg_bytes);
      std::memcpy(code_u, hu.data(), cu_bytes);
      std::memcpy(scg, hsg.data(), sg_bytes * 2);
      std::memcpy(scu, hsu.data(), su_bytes * 2);
      std::memcpy(act, hact.data(), act_bytes * 2);
      // Poison both output sets so an unwritten element cannot match by luck.
      for (std::size_t i = 0; i < og_bytes; ++i) { og[i] = 0xDEAD; og_ref[i] = 0xBEEF; }
      for (std::size_t i = 0; i < ou_bytes; ++i) { ou[i] = 0xDEAD; ou_ref[i] = 0xBEEF; }

      // REFERENCE: two separate single-output GEMMs on a FULL-SIZE code/scale buffer
      // (the reference must not itself over-read the sliced shard). The single
      // GEMM writes its full M*Ng result into og_ref; the comparison window is
      // the engine's readback window (start_g..start_g+Nl_g), which is the only
      // part the fused kernel is contractually required to produce.
      intfer::gemm::quant_gemm_forward_usm(q, act, code_full_g, sc_full_g, 1.0f, og_ref,
                                           1, sh.Ng, sh.K, sh.group, sc, sycl::event{}).wait_and_throw();
      intfer::gemm::quant_gemm_forward_usm(q, act, code_full_u, sc_full_u, 1.0f, ou_ref,
                                           1, sh.Nu, sh.K, sh.group, sc, sycl::event{}).wait_and_throw();

      // ARM: the fused gate+up kernel on the SLICED shard buffers, with the
      // shard's first row and length passed through. On the full-size shapes
      // these are 0 / Ng and the call is byte-identical to the pre-fix form.
      intfer::gemm::quant_gemm_forward_usm_gateup(
          q, act, code_g, scg, 1.0f, og, code_u, scu, 1.0f, ou,
          1, sh.Ng, sh.Nu, sh.K, sh.group,
          sh.start_g, Nl_g, sh.start_u, Nl_u, sc, sc,
          sycl::event{}).wait_and_throw();

      int dg = 0, du = 0;
      for (std::uint32_t r = 0; r < Nl_g; ++r)
        if (og[sh.start_g + r] != og_ref[sh.start_g + r]) ++dg;
      for (std::uint32_t r = 0; r < Nl_u; ++r)
        if (ou[sh.start_u + r] != ou_ref[sh.start_u + r]) ++du;
      const bool ok = (dg == 0 && du == 0);
      ++checks;
      if (!ok) ++fails;
      std::printf("%s|%s|%d|%s|%d|%d|%s\n", sh.name, "INT2", d, dataset_name(d),
                  dg, du, ok ? "PASS" : "MISMATCH");
      if (!ok) {
        for (std::uint32_t r = 0; r < Nl_g && dg; ++r)
          if (og[sh.start_g + r] != og_ref[sh.start_g + r]) {
            std::fprintf(stderr, "  gate[%u]: fused=0x%04x ref=0x%04x\n",
                         sh.start_g + r, og[sh.start_g + r], og_ref[sh.start_g + r]);
            break;
          }
        for (std::uint32_t r = 0; r < Nl_u && du; ++r)
          if (ou[sh.start_u + r] != ou_ref[sh.start_u + r]) {
            std::fprintf(stderr, "  up[%u]: fused=0x%04x ref=0x%04x\n",
                         sh.start_u + r, ou[sh.start_u + r], ou_ref[sh.start_u + r]);
            break;
          }
      }
    }
    sycl::free(code_g, dev.sycl_context); sycl::free(code_u, dev.sycl_context);
    sycl::free(scg, dev.sycl_context); sycl::free(scu, dev.sycl_context);
    sycl::free(code_full_g, dev.sycl_context); sycl::free(code_full_u, dev.sycl_context);
    sycl::free(sc_full_g, dev.sycl_context); sycl::free(sc_full_u, dev.sycl_context);
    sycl::free(act, dev.sycl_context);
    sycl::free(og, dev.sycl_context); sycl::free(ou, dev.sycl_context);
    sycl::free(og_ref, dev.sycl_context); sycl::free(ou_ref, dev.sycl_context);
  }

  std::printf("# checks=%d fails=%d\n", checks, fails);
  std::fprintf(stderr, "[gateup] %s: checks=%d fails=%d\n",
               fails == 0 ? "PASS" : "FAIL", checks, fails);
  q.wait_and_throw();
  return fails == 0 ? 0 : 1;
}
