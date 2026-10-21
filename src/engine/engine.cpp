// src/engine/engine.cpp
// Stage-2 engine implementation: loads the .intfer, shards Linear weights across
// the 2x Arc B580 (tensor parallel), and runs the per-token decode step through
// the quantized GEMM (XMX/DPAS) + fused FlashAttention, with SYCL command-graph
// capture of the decode loop.
//
// NOTE: this is the engine skeleton. It compiles against oneAPI DPC++ and runs on
// real Arc hardware; in this sandbox no GPU/SYCL platform is present so it is
// build-validated only. The hybrid Qwen3.5 forward (RMSNorm, mRoPE, the
// gated-delta-rule SSM, KV cache, tokenizer, sampling) plugs into these
// primitives as the remaining integration.
// ... (previous imports)
#include "intfer/engine.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <vector>
// ...


#include "intfer/output.hpp"
#include "intfer/safetensors.hpp"
#include "intfer/gemm/quant_gemm.hpp"
#include "intfer/attention/flash_attention.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/device/device_manager.hpp"

#include "intfer/spinquant.hpp"  // for canonical_codebook + unpack_nibble

namespace intfer {

// ---------------------------------------------------------------------------
// S1 -- uncached host-USM OUTPUT buffers (Strata, informed-from-scratch).
//
// WHAT THIS IS. Upstream Strata allocates its doorbell buffer with
// zeMemAllocHost + ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED
// (sycl/include/strata/sycl_queue.hpp:63-79), so the HOST's read of a word
// the DEVICE wrote is served from memory instead of the host's stale cache
// line. CITATION(https://github.com/Niko1221/Strata -- docs/engines/strata.txt, the
// S1 row: informed-from-scratch, nothing copied as-is).
//
// WHY IT IS SAFE HERE, AND THE CONSTRAINT THAT MAKES IT SAFE.
//   * The buffer is DEVICE-WRITES / HOST-READS. The host never writes a .out
//     buffer (the only host writes into a workspace are the std::memcpy calls
//     into .act), so the buffer is strictly one-directional.
//   * .act IS EXCLUDED BY CONSTRUCTION AND MUST STAY THAT WAY. .act is
//     HOST-WRITES / DEVICE-READS, and BIAS_UNCACHED gives no DEVICE-side
//     guarantee -- the Level Zero spec says "No other agent is guaranteed to
//     see the memory as uncached" -- so an uncached .act could hand the device
//     a stale line. The helper below is therefore reachable ONLY from the .out
//     allocation sites; .act keeps sycl::malloc_shared unconditionally.
//
// PRECONDITION, NOT MEASURED ON THIS BOX. Whether B580's xe driver keeps the
// host's uncached read coherent with a DEVICE KERNEL STORE is exactly what
// examples/uncached_doorbell_probe/ measures: its one_round() submits a
// parallel_for that writes the buffer and then verifies EVERY BYTE on the
// host, which is the same shape as the .out readback. UNMEASURED here
// (operator-only). The flag stays DEFAULT OFF until that probe is run.
//
// DEFAULT OFF. INTFER_UNCACHED_HOST_USM is PRESENCE-spelled (any value
// enables; unset disables).
// ---------------------------------------------------------------------------
namespace uncached_out {

// Presence-spelled, read once per process.
inline bool s1_enabled() {
  static const bool on = (std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr);
  return on;
}

// Allocate `bytes` of device-accessible host memory with the uncached bias.
// Returns nullptr when the uncached path is unavailable or refused; the
// caller then falls back to sycl::malloc_shared, which is the shipped path.
inline void* alloc(std::size_t bytes, const device::DeviceContext& dev) {
#ifdef INTFER_HAS_L0
  if (s1_enabled() && dev.has_l0() && dev.ze_context != nullptr) {
    ze_host_mem_alloc_desc_t d{};
    d.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    d.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    void* p = nullptr;
    if (zeMemAllocHost(dev.ze_context, &d, bytes, 64, &p) == ZE_RESULT_SUCCESS &&
        p != nullptr) {
      return p;
    }
  }
#else
  (void)bytes;
  (void)dev;
#endif
  return nullptr;
}

// Release a pointer obtained from alloc(). Only ever called for a pointer
// whose out_uncached flag is set, i.e. one that came from alloc() and not
// from the sycl::malloc_shared fallback.
inline void release(void* p, const device::DeviceContext& dev) {
#ifdef INTFER_HAS_L0
  if (p != nullptr && dev.ze_context != nullptr) {
    (void)zeMemFree(dev.ze_context, p);
  }
#else
  (void)p;
  (void)dev;
#endif
}

// Once-per-process announcement. It is emitted ONLY when the flag is set, so
// the shipped (flag-unset) run prints nothing -- the same contract as the
// [decode1-wg] line, whose un-gated form once broke a shipped gate by
// printing a marker on the BASE run.
inline void announce(const char* site, bool took_uncached) {
  // Gated on the env read INSIDE the announcement, not only at the call site:
  // the shipped (flag-unset) run must print nothing. The [decode1-wg] line once
  // printed on the BASE run and broke a shipped gate, so the gate belongs here.
  if (!s1_enabled()) return;
  static std::atomic<bool> done{false};
  bool expected = false;
  if (!done.compare_exchange_strong(expected, true)) return;
  std::fprintf(stderr,
               "[uncached-host-usm] %s (INTFER_UNCACHED_HOST_USM=%s) %s "
               "buffers=out-only act=excluded\n",
               site, std::getenv("INTFER_UNCACHED_HOST_USM"),
               took_uncached ? "ACTIVE alloc=zeMemAllocHost|BIAS_UNCACHED"
                             : "FALLBACK alloc=sycl::malloc_shared");
  std::fflush(stderr);
}

}  // namespace uncached_out

Engine::~Engine() {
  if (backing_.base && backing_.size) ::munmap(backing_.base, backing_.size);
  backing_.base = nullptr;
  backing_.size = 0;
  // Drain the USM pool (body runs before member destruction, so queues live).
  for (size_t d = 0; d < usm_pool_.size() && d < devices_.size(); ++d)
    for (auto& [bytes, vec] : usm_pool_[d]) {
      (void)bytes;
      for (void* p : vec) sycl::free(p, devices_[d].queue);
      vec.clear();
    }
  // S1: DRAIN BEFORE RELEASE. An uncached .out buffer is a HOST ALLOCATION
  // (zeMemAllocHost), and zeMemFree UNMAPS it -- so a device kernel still
  // writing to it would fault. Strata's own note on this driver: "an Arc Pro
  // B70 (xe) page-faults on it (ccs0, FaultType 0, address in the CPU's mmap
  // range) and the card times out the job". The two GROWTH paths already
  // drain (ensure_wave_workspace and ensure_workspace both queue.wait() before
  // freeing the old buffer), but the DESTRUCTOR did not, so an Engine torn
  // down with work in flight could unmap a buffer the device was still
  // writing. THE DRAIN IS CONDITIONAL on an uncached buffer existing, so the
  // shipped (flag-unset) path is unchanged in both behaviour and cost.
  {
    bool any_uncached = false;
    for (size_t d = 0; d < ws_.size() && !any_uncached; ++d)
      if (ws_[d].out_uncached) any_uncached = true;
    for (size_t i = 0; i < wv_.size() && !any_uncached; ++i)
      if (wv_[i].out_uncached) any_uncached = true;
    if (any_uncached)
      for (size_t d = 0; d < devices_.size(); ++d) devices_[d].queue.wait();
  }
  // Free the shared act/out workspaces (2 live allocs per device max).
  for (size_t d = 0; d < ws_.size() && d < devices_.size(); ++d) {
    act_cache_ptr_ = nullptr; act_cache_M_ = 0; act_cache_K_ = 0;
    if (ws_[d].act) sycl::free(ws_[d].act, devices_[d].queue);
    if (ws_[d].out) {
      if (ws_[d].out_uncached) uncached_out::release(ws_[d].out, devices_[d]);
      else sycl::free(ws_[d].out, devices_[d].queue);
    }
    ws_[d].act = nullptr; ws_[d].out = nullptr; ws_[d].out_uncached = false;
  }
  // Free the DPAS act workspaces (3 live allocs per device max).
  for (size_t d = 0; d < ws48_.size() && d < devices_.size(); ++d) {
    if (ws48_[d].act) sycl::free(ws48_[d].act, devices_[d].queue);
    if (ws48_[d].as) sycl::free(ws48_[d].as, devices_[d].queue);
    if (ws48_[d].f32) sycl::free(ws48_[d].f32, devices_[d].queue);
    if (ws48_[d].si) sycl::free(ws48_[d].si, devices_[d].queue);
    if (ws48_[d].fold) sycl::free(ws48_[d].fold, devices_[d].queue);
    ws48_[d].act = nullptr; ws48_[d].as = nullptr; ws48_[d].f32 = nullptr;
    ws48_[d].si = nullptr; ws48_[d].fold = nullptr;
  }
  // Free the wave workspaces (2*W live allocs per device max).
  for (size_t i = 0; i < wv_.size(); ++i) {
    const size_t dd = wv_W_ > 0 ? i / wv_W_ : 0;
    if (dd < devices_.size()) {
      if (wv_[i].act) sycl::free(wv_[i].act, devices_[dd].queue);
      if (wv_[i].out) {
        if (wv_[i].out_uncached) uncached_out::release(wv_[i].out, devices_[dd]);
        else sycl::free(wv_[i].out, devices_[dd].queue);
      }
      wv_[i].act = nullptr; wv_[i].out = nullptr;
      wv_[i].out_uncached = false;
    }
  }
  wv_.clear(); wv_W_ = 0; wv_M_ = wv_K_ = wv_Nl_ = 0;
}

// INTFER_WEIGHTS_DEVICE -- allocate the WEIGHT shards as device USM instead of
// shared USM. Default OFF; kill switch INTFER_NO_WEIGHTS_DEVICE=1.
//
// WHY. Every weight shard is allocated here and filled by a host memcpy
// (ensure_shard_built), and the shards are 9.75 GB/device at 27B TP=2 -- read once
// per token, and 100 % of decode's traffic
// (scripts/model_byte_roofline.sh). So how this allocator is backed decides how
// fast the entire engine can read weights.
//
// MEASURED, by examples/roofline_probe on the identical buffer and access pattern
// (device event timestamps, so load-insensitive; the gate is
// scripts/test_roofline_probe.sh):
//
//     sequential read, malloc_shared   206-231 GB/s   45-51 % of the 456 spec
//     sequential read, malloc_device   231-280 GB/s   51-61 %  -> +11 % to +16 %
//
// Shared USM on a discrete card is host-accessible, so the runtime cannot use it
// exactly as it would a device allocation; that the gap is small is the good
// outcome -- it means shared USM is NOT the pathology this could have been, and
// the same probe puts decode's measured 41.1 GB/s at 19.9 % of the shared ceiling.
// The deficit is kernel-side, not allocator-side. That is a MEASURED negative for
// "the allocator is the fix" and a MEASURED positive for "+11-16 % is free".
//
// BIT-EXACT, and that is why this is the cheapest change in the batch: same bytes,
// same kernels, same order. It needs a stream-identity gate and NO numerics
// contract. The one thing it does change is that device USM is not host-readable,
// so every host-side read of code_usm must go through code_host instead. Audited:
// the only host writes are the memcpys in ensure_shard_built, and the only
// host-visible pointer handed out is `weights_for()`'s `v.code`, which the CLI
// passes to the KERNEL and never dereferences (grepped: examples/intfer_cli
// reads .code/.scale only to forward them).
static bool weights_device_usm() {
  static const bool on = [] {
    const char* e = std::getenv("INTFER_WEIGHTS_DEVICE");
    const bool no = (std::getenv("INTFER_NO_WEIGHTS_DEVICE") != nullptr);
    return (e != nullptr) && !no;
  }();
  return on;
}

void* Engine::pool_acquire(size_t device, size_t bytes) {
  if (device >= devices_.size() || bytes == 0) return nullptr;
  if (device >= usm_pool_.size()) usm_pool_.resize(device + 1);
  pool_acquires_++;
  auto& vec = usm_pool_[device][bytes];
  if (!vec.empty()) {
    void* p = vec.back();
    vec.pop_back();
    pool_hits_++;
    return p;
  }
  auto& dev = devices_[device];
  void* p = weights_device_usm()
               ? sycl::malloc_device(bytes, dev.sycl_device, dev.sycl_context)
               : sycl::malloc_shared(bytes, dev.sycl_device, dev.sycl_context);
  if (!p) {
    // Device OOM under preload-all: fail loudly with the exact shortfall
    // instead of a downstream nullptr-memcpy SIGSEGV. Steady-state VRAM is
    // ~10.9GB weights/dev (27B-sc TP=2) vs the 12216MiB B580 budget.
    std::fprintf(stderr,
                 "[intfer] FATAL: USM alloc of %zu bytes failed on device %zu "
                 "(VRAM budget exceeded?)\n",
                 bytes, device);
    std::fflush(stderr);
    std::abort();
  }
  return p;
}

void Engine::pool_release(size_t device, void* ptr, size_t bytes) {
  if (!ptr || bytes == 0) return;
  if (device >= usm_pool_.size()) usm_pool_.resize(device + 1);
  usm_pool_[device][bytes].push_back(ptr);
}

#ifdef INTFER_PROFILE_GPU
  struct GpuProfile {
    double total_kernel_us  = 0.0;
    double total_wait_us    = 0.0;
    double total_copy_us    = 0.0;
    double total_readback_us = 0.0;
    double total_usm_write_us = 0.0;  // host USM activation write loop
    double total_submit_us = 0.0;     // q.submit (launch) host overhead
    double total_waitblock_us = 0.0;  // event.wait() blocking host time
    double total_rebuild_us = 0.0;    // ensure_shard_built (host repack + malloc_shared + memcpy)
    uint64_t rebuild_count = 0;
    uint64_t peak_wait_depth = 0;
    uint64_t kernel_count   = 0;
    uint64_t wait_count     = 0;
    uint64_t copy_count     = 0;
    uint64_t readback_count = 0;
    uint64_t submit_count   = 0;   // q.submit host overhead calls
    uint64_t usm_write_count = 0;   // host USM activation write memcpy calls
    // Device-timeline markers (INTFER_TIMELINE=1): count of 1-work-item
    // empty kernels submitted as region boundaries. See timeline_mark().
    uint64_t timeline_marker_count = 0;
    // Matrix-engine (XMX joint_matrix) launch accounting (2026-10-19):
    // engine-side dispatch observations, NOT hardware XVE counters. See
    // GpuProfileSnapshot's field comment in include/intfer/engine.hpp -- the
    // attempt/launch/decline triple exists because a bare launch count cannot
    // distinguish "never requested" from "requested, every tile declined".
    uint64_t xmx_attempt_count = 0;
    uint64_t xmx_launch_count = 0;
    uint64_t xmx_decline_count = 0;
  };
  static GpuProfile g_gpu_profile;

  // ---- Matrix-engine (XMX) launch accounting (2026-10-19) ----
  // ONE call per XMX dispatch attempt, carrying the kernel's own `ran`
  // verdict. A helper rather than the counter logic open-coded at the four
  // dispatch sites (run_linear_tp, run_mlp_gateup's gate and up, la_out_tp2,
  // run_linear_wave) so the triple cannot drift between them: a site that
  // counted a DECLINE as a launch, or an attempt that never happened as one,
  // would make the [xmx-attrib] three-way verdict wrong in exactly the
  // direction the AGENTS.md XMX verdict depends on.
  void xmx_note_dispatch(bool ran) {
    g_gpu_profile.xmx_attempt_count += 1;
    if (ran) g_gpu_profile.xmx_launch_count += 1;
    else     g_gpu_profile.xmx_decline_count += 1;
  }
#endif

using namespace sycl::ext::oneapi::experimental;

namespace {

gemm::WeightScheme to_ws(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return gemm::WeightScheme::INT2;
    case QuantScheme::INT4: return gemm::WeightScheme::INT4;
    case QuantScheme::INT8: return gemm::WeightScheme::INT8;
    default:                return gemm::WeightScheme::FP16;
  }
}

// TODO 6 (W4A8/W2A8): DPAS eligibility for a shard. Requirements:
//   * low-bit scheme with supported (bits, weight-group) combo:
//     INT4/G64, INT8/G32, INT2/G32, INT2/G16 (G16 = 2 DPAS/op split)
//   * N_local % 16 == 0, K % 32 == 0 (all 27B/0.8B tensors post-TP-split)
//   * per-tensor global divisor only via host alpha (already the case)
// Activation quant (INT8/GA64) and repack happen at shard build; dispatch
// is then a flag check in run_linear_tp. Default OFF (2026-09-18: the DPAS
// path is a net loss on 0.8B at every tested shape — host GA64 act-quant
// ~6ms/token dominates, XMX kernel no faster than the tiled legacy kernel
// at M=1/128; greedy streams bit-identical either way). INTFER_DPAS48=1
// opts back in (bisection / future shapes where XMX wins).
bool dpas48_eligible(QuantScheme scheme, uint32_t group_size,
                     uint32_t N_local, uint32_t K) {
  const char* env = std::getenv("INTFER_DPAS48");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  if (!enabled) return false;
  int bits = (scheme == QuantScheme::INT2)    ? 2
             : (scheme == QuantScheme::INT4)  ? 4
             : (scheme == QuantScheme::INT8)  ? 8
                                              : 16;
  if (bits == 16) return false;
  const uint32_t G = (group_size == 0) ? K : group_size;
  const bool combo_ok =
      (bits == 4 && G == 64) || (bits == 8 && G == 32) ||
      (bits == 2 && (G == 32 || G == 16));
  if (!combo_ok) return false;
  return (N_local % 16 == 0) && (K % 32 == 0);
}

// Row-cooperative DPAS experiment opt-in (default OFF). Host-side read;
// getenv/static are illegal in device code, so the flag is read once.
bool dpas_row_enabled() {
  const char* env = std::getenv("INTFER_DPAS_ROW");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled;
}

// E4: cooperative 2D row-block DPAS (default OFF). Same getenv/static rule.
//
// Deliberately MUTUALLY EXCLUSIVE with INTFER_DPAS_ROW: that experiment already
// owns the "one WI owns many tiles" geometry and measured ~460x slower by
// collapsing occupancy. Letting both flags be set would leave it undecidable
// which geometry ran, which is the "unattributable experiment" failure the
// report warns about. Declining the combination is loud in the sense that
// dpas_row wins and E4 does not fire.
bool dpas_tiled_enabled() {
  const char* env = std::getenv("INTFER_DPAS48_TILED");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled && !dpas_row_enabled();
}

// Device act-quant pre-pass opt-in (default OFF). Host-side read, same
// getenv/static rule as dpas_row_enabled(). The pre-pass is bit-exact (same
// GA64 HalfAway contract as dashq::act_quant_ga64_batch) — it moves the
// quant from host to device, it does not change the arithmetic.
bool dpas_preq_enabled() {
  const char* env = std::getenv("INTFER_DPAS_PREQ");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled;
}

// DPAS row-fold hoist opt-in (default OFF). Same getenv/static rule as the
// flags above. Pure work hoist: the values moved out of the tile loop are
// exact int32 sums of the int8 activation codes, so this cannot change a
// result — it only stops every 16-N tile WI from recomputing them.
bool dpas_fold_enabled() {
  const char* env = std::getenv("INTFER_DPAS48_FOLD");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled;
}

// INTFER_WAVE_LAUNCH_FUSE: fuse a wave's independent GEMMs into one launch per
// device (see run_linear_wave). Default OFF; PRESENCE enables, matching the
// opt-in convention.
bool wave_launch_fuse_enabled() {
  return std::getenv("INTFER_WAVE_LAUNCH_FUSE") != nullptr;
}

// Build a per-device GemmWeightBuffers view over rows [start,end) of a tensor's
// packed payload (column tensor-parallel sharding of the output dim N).
//
// The Stage-1 DASH-Q pipeline emits codes in *group-major* order
// (idx[i] = a[g*N*Glocal + r*Glocal + c]) and scales per-group only, but the
// dequant-on-the-fly GEMM kernel expects codes in row-major (n*K+k) and scales
// per-(row,group). To bridge the two without rewriting the GEMM kernel, we
// re-unpack and re-pack on the host into the kernel's layout ONCE at load
// time, then use simple row-major slices for TP. The shard covers the
// contiguous range [start,end) of the N dimension.
//
// k_off/k_len (INTFER_MLP_PARTIAL_DOWN): instead of a contiguous N-slice, take
// the FULL N (start=0, end=t.N) and a K-slice [k_off, k_off+k_len) of every
// row. The kernel reads code[ng*K+kg] with K = k_len, so the shard is a
// full-N x K-slice buffer; the group-major source is sampled at the columns
// that fall inside the slice (a group may contribute 0..G columns). This is
// the partial-down shard: device d computes partial_d[n] over its K-slice for
// all n, and the host sums partial_0 + partial_1 in FP32. Default k_off=0,
// k_len=0 => the full-K contiguous slice every existing caller uses.
gemm::GemmWeightBuffers shard_weight(const ModelIR& m, const TensorIR& t,
                                     uint32_t start, uint32_t end,
                                     sycl::queue& q,
                                     uint32_t k_off = 0, uint32_t k_len = 0) {
  gemm::GemmWeightBuffers w;
  w.scheme = to_ws(t.qcfg.scheme);
  w.group_size = t.qcfg.group_size;
  w.has_global_divisor = t.qcfg.has_global_divisor;
  w.N = end - start;
  w.K = (k_len == 0) ? t.K : k_len;
  const uint32_t N = t.N, K = t.K;
  const uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : K;
  const uint32_t n_groups = (K + G - 1) / G;
  const int bits = (w.scheme == gemm::WeightScheme::INT2) ? 2
                 : (w.scheme == gemm::WeightScheme::INT4) ? 4
                 : (w.scheme == gemm::WeightScheme::FP16) ? 16 : 8;
  const int pack = (bits >= 8) ? 1 : (8 / bits);

  // 1) Build the per-group scale table (FP16 bits) and a per-row broadcast
  //    view. The kernel reads scale[ng*n_groups + g], so we emit (N*n_groups)
  //    FP16 values (broadcast the per-group scale to every row).
  //    FP16-scheme tensors carry no meaningful scale (CPU dequant is a raw
  //    half->float copy); the kernel still multiplies by the table, so emit
  //    all-ones (0x3C00) instead of copying scale_src (which holds a small
  //    fallback value that would attenuate the output ~50x).
  const uint16_t* scale_src =
      reinterpret_cast<const uint16_t*>(m.scale_data() + t.scale_offset);
  const bool fp16_passthrough = (w.scheme == gemm::WeightScheme::FP16);
  // K-slice shard: the groups that fall inside [k_off, k_off+w.K). Requires
  // w.K to be group-aligned (caller's responsibility) so a group contributes
  // either wholly or not at all.
  const uint32_t g0 = k_off / G;
  const uint32_t g1 = (k_len == 0) ? n_groups : g0 + w.K / G;
  const uint32_t n_groups_s = g1 - g0;
  std::vector<uint16_t> scale_slice((size_t)w.N * n_groups_s);
  for (uint32_t r = 0; r < w.N; ++r)
    for (uint32_t g = 0; g < n_groups_s; ++g)
      scale_slice[(size_t)r * n_groups_s + g] =
          fp16_passthrough ? 0x3C00 : scale_src[g0 + g];
  // Keep the packed vectors alive in the returned struct. The buffer is built
  // from the vector's iterators (eager copy, like gemm_demo); the vector stays
  // alive in `w` until the GEMM runs, so the payload is never lost.
  w.scale_store = std::move(scale_slice);
  w.scale = sycl::buffer<std::uint16_t>(w.scale_store.begin(), w.scale_store.end());

  // 2) Unpack group-major codes into row-major INT8 codes, then take the TP
  //    slice and re-pack (only for INT2/INT4). For INT8/FP16 the code bytes
  //    are identity so this is a plain memcpy. The kernel reads code[ng*K+kg]
  //    and unpacks bits-within-byte using idx%pack, so a row-major byte stream
  //    is exactly what it expects.
  if (w.scheme == gemm::WeightScheme::FP16) {
    // FP16: pass-through the N*K*2 byte slice (identity for the kernel).
    std::vector<uint8_t> code_slice((size_t)w.N * w.K * 2);
    const uint8_t* base = m.code_data() + t.code_offset
                        + (size_t)(start * K + k_off) * 2;
    std::memcpy(code_slice.data(), base, code_slice.size());
    w.code_store = std::move(code_slice);
    w.code = sycl::buffer<std::uint8_t>(w.code_store.data(),
                                         sycl::range<1>(w.code_store.size()));
  } else {
    // Group-major unpack -> row-major INT8 -> pack to bits. For a K-slice
    // shard the destination K is w.K (the slice length) and the source
    // columns are [k_off, k_off+w.K); a group that straddles the slice
    // boundary contributes only its in-slice columns (Glocal is recomputed
    // per group from the intersection).
    std::vector<uint8_t> row_codes((size_t)w.N * w.K);
    for (uint32_t g = g0; g < g1; ++g) {
      const uint32_t k0 = g * G;
      const uint32_t k1 = std::min(k0 + G, K);
      const uint32_t c0 = std::max(k0, k_off);
      const uint32_t c1 = std::min(k1, k_off + w.K);
      if (c0 >= c1) continue;
      const uint32_t Glocal = k1 - k0;
      for (uint32_t r = 0; r < w.N; ++r) {
        const uint32_t src_row = r + start;
        for (uint32_t c = c0; c < c1; ++c) {
          const size_t src_idx = (size_t)g * (N * Glocal)
                               + (size_t)src_row * Glocal + (c - k0);
          const uint8_t b = m.code_data()[t.code_offset + src_idx / pack];
          const int within = (int)(src_idx % pack);
          const uint8_t val = (b >> (within * bits)) & ((1u << bits) - 1u);
          row_codes[(size_t)r * w.K + (c - k_off)] = val;
        }
      }
    }
    // Repack row-major INT8 to packed bytes (kernel layout).
    const uint32_t per_row_code_bytes_s = (w.K + pack - 1) / pack;
    std::vector<uint8_t> code_slice((size_t)w.N * per_row_code_bytes_s, 0);
    for (size_t i = 0; i < (size_t)w.N * w.K; ++i) {
      const size_t pack_idx = i / pack;
      const int within = (int)(i % pack);
      code_slice[pack_idx] |= (uint8_t)((row_codes[i] & ((1u << bits) - 1u))
                                        << (within * bits));
    }
    w.code_store = std::move(code_slice);
    w.code = sycl::buffer<std::uint8_t>(w.code_store.data(),
                                         sycl::range<1>(w.code_store.size()));
  }

  if (t.global_bytes > 0) {
    // global_offset is a *byte* offset into the uint8_t global_buf (see
    // dashq.cpp: r.globals assigned from begin()+global_offset). Copy the 4
    // bytes into an aligned float (avoids the previous 4x mis-offset read that
    // yielded garbage alpha and zeroed/NaN'd the GEMM output).
    float alpha_val = 0.f;
    std::memcpy(&alpha_val, m.global_data() + t.global_offset,
                std::min<uint32_t>(t.global_bytes, 4u));
    w.global_store.assign(1, alpha_val);
    w.global = sycl::buffer<float>(w.global_store.begin(), w.global_store.end());
  }
  return w;
}

}  // namespace

std::unique_ptr<Engine> Engine::load(const std::string& intfer_path,
                                    size_t tp_devices) {
  auto e = std::unique_ptr<Engine>(new Engine());

  // Zero-copy load: mmap the file and borrow the payload views instead of
  // copying multi-GB payloads into RAM (a 22GB .intifer would otherwise need
  // ~44GB transient RSS: file blob + deserialized vectors). The mapping is
  // kept in e->backing_ for the Engine's lifetime. Small files (<1GB) use
  // the old read+copy path so short-lived tools don't pin mappings.
  RotationSet rs;
  {
    FILE* f = std::fopen(intfer_path.c_str(), "rb");
    if (!f) { std::cerr << "Engine: cannot read " << intfer_path << "\n"; return nullptr; }
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fclose(f);
    if (fsize <= 0) { std::cerr << "Engine: cannot read " << intfer_path << "\n"; return nullptr; }
    if ((size_t)fsize < (1ULL << 30)) {
      std::vector<uint8_t> blob;
      if (!read_file(intfer_path, blob)) {
        std::cerr << "Engine: cannot read " << intfer_path << "\n";
        return nullptr;
      }
      if (!deserialize_intfer(blob.data(), blob.size(), e->model_, rs)) {
        std::cerr << "Engine: cannot deserialize " << intfer_path << "\n";
        return nullptr;
      }
    } else {
      int fd = ::open(intfer_path.c_str(), O_RDONLY);
      if (fd < 0) { std::cerr << "Engine: cannot mmap " << intfer_path << "\n"; return nullptr; }
      void* m = ::mmap(nullptr, (size_t)fsize, PROT_READ, MAP_PRIVATE, fd, 0);
      ::close(fd);
      if (m == MAP_FAILED) { std::cerr << "Engine: cannot mmap " << intfer_path << "\n"; return nullptr; }
      e->backing_.base = m;
      e->backing_.size = (size_t)fsize;
      if (!deserialize_intfer_borrowed((const uint8_t*)m, (size_t)fsize, e->model_, rs)) {
        ::munmap(m, (size_t)fsize);
        e->backing_ = Engine::MappedFile{};
        std::cerr << "Engine: cannot deserialize " << intfer_path << "\n";
        return nullptr;
      }
    }
  }
  e->rs_ = std::move(rs);
  for (auto& R : e->rs_.rotations){
    // The serializer does not persist is_hadamard: re-detect empirically so
    // online replay (via rotation_for) can use the O(n log n) FWHT path.
    R.is_hadamard = rotation_is_hadamard(R);
    e->rid_[R.id] = &R;
  }

  // Discover the 2x Arc B580 fleet and pick `tp_devices` of them.
  auto dm = device::DeviceManager::discover("Arc");
  size_t n = (tp_devices == 0) ? dm.size() : std::min(tp_devices, dm.size());
  if (n == 0) { std::cerr << "Engine: no Arc GPU discovered\n"; return nullptr; }
  // AUDIT #9/#11: the min() clamp above is correct but SILENT. When the caller
  // asked for more ranks than the box actually has, the run continues at the
  // smaller TP and, if the model does not fit that many devices, dies later in
  // an opaque place — for 27B on one B580 the failure surfaces as a Level Zero
  // CPU-page-fault-memory-manager abort (signal 6) during prefill, which reads
  // like a driver bug and buries the real cause. Loud, NON-fatal: a legitimate
  // deliberate TP=1 run must not be blocked by this, so warn and continue.
  if (tp_devices != 0 && dm.size() < tp_devices) {
    std::cerr << "Engine: WARNING: TP=" << tp_devices << " requested but only "
              << dm.size() << " Arc GPU(s) discovered — continuing with TP=" << n
              << ". A model that needs more VRAM than one device holds (27B "
                 "needs ~19GB; a B580 has ~12GB) will fail below.\n"
              << std::flush;
  }
  for (size_t i = 0; i < n; ++i) e->devices_.push_back(dm.at(i));
  std::cerr << "Engine: " << n << " Arc device(s): ";
  for (auto& d : e->devices_) std::cerr << d.name << " ";
  std::cerr << "\n";

  e->build_shards();
  return e;
}

#ifdef INTFER_HAS_L0
#include <level_zero/zes_api.h>
#endif

// Free VRAM bytes on device i via sysman (see header). Sysman device order
// is NOT SYCL order (measured reversed on 2x B580), so match by native
// UUID: sycl::get_native<L0> handle -> zeDeviceGetProperties uuid vs
// zesDeviceGetProperties core.uuid (same UUID space; never call the core
// getter on a sysman handle — it segfaults). Returns 0 on any failure;
// the caller falls back to an even split.
uint64_t Engine::device_free_bytes(size_t i) {
  if (i >= devices_.size()) return 0;
#ifdef INTFER_HAS_L0
  if (!devices_[i].has_l0()) return 0;
  ze_device_properties_t want{};
  want.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (zeDeviceGetProperties(devices_[i].ze_device, &want) != ZE_RESULT_SUCCESS)
    return 0;
  if (zesInit(0) != ZE_RESULT_SUCCESS) return 0;
  uint64_t free_total = 0;
  bool matched = false;
  uint32_t ndrv = 0;
  if (zesDriverGet(&ndrv, nullptr) != ZE_RESULT_SUCCESS) return 0;
  std::vector<zes_driver_handle_t> drvs(ndrv);
  if (zesDriverGet(&ndrv, drvs.data()) != ZE_RESULT_SUCCESS) return 0;
  for (auto drv : drvs) {
    uint32_t ndev = 0;
    if (zesDeviceGet(drv, &ndev, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<zes_device_handle_t> sdevs(ndev);
    if (zesDeviceGet(drv, &ndev, sdevs.data()) != ZE_RESULT_SUCCESS) continue;
    for (auto sd : sdevs) {
      zes_device_properties_t sp{};
      sp.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      if (zesDeviceGetProperties(sd, &sp) != ZE_RESULT_SUCCESS) continue;
      bool eq = true;
      for (int b = 0; b < 16; ++b)
        if (sp.core.uuid.id[b] != want.uuid.id[b]) { eq = false; break; }
      if (!eq) continue;
      matched = true;
      uint32_t nmem = 0;
      if (zesDeviceEnumMemoryModules(sd, &nmem, nullptr) != ZE_RESULT_SUCCESS)
        return 0;
      std::vector<zes_mem_handle_t> mems(nmem);
      if (zesDeviceEnumMemoryModules(sd, &nmem, mems.data()) != ZE_RESULT_SUCCESS)
        return 0;
      for (auto mh : mems) {
        zes_mem_state_t st{};
        st.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
        if (zesMemoryGetState(mh, &st) == ZE_RESULT_SUCCESS) free_total += st.free;
      }
    }
  }
  return matched ? free_total : 0;
#else
  return 0;
#endif
}

void Engine::build_shards() {
  // Balanced contiguous sharding of the output dim N across the devices
  // (tensor-parallel column split). Mirrors DeviceManager::shard.
  auto split = [this](uint32_t total) {
    std::vector<std::pair<uint32_t, uint32_t>> r;
    size_t n = devices_.size();
    for (size_t i = 0; i < n; ++i) {
      uint32_t start = (uint32_t)(total * i / n);
      uint32_t end = (uint32_t)(total * (i + 1) / n);
      r.push_back({start, end});
    }
    return r;
  };
  shards_.resize(model_.tensors.size());
  shard_cache_.resize(model_.tensors.size());
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti) {
    const auto& t = model_.tensors[ti];
    shards_[ti].ranges = (devices_.size() > 1) ? split(t.N)
                                            : std::vector<std::pair<uint32_t, uint32_t>>{{0, t.N}};
  }
  // Identify layer blocks for the decode loop.
  for (const auto& t : model_.tensors)
    if (t.role == TensorRole::ATTN_Q) {
      std::string p = t.name;
      auto pos = p.find("layers.");
      if (pos != std::string::npos) layer_names_.push_back(p.substr(0, p.find('.', pos + 7) + 1));
    }
  // Per-layer tensor index (layer-wise VRAM streaming). Unlayered tensors
  // (lm_head, ...) get -1 and are never evicted by begin_layer().
  layer_of_.assign(model_.tensors.size(), -1);
  int maxL = -1;
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti) {
    int L = layer_of_name(model_.tensors[ti].name);
    layer_of_[ti] = L;
    if (L > maxL) maxL = L;
  }
  layer_tensors_.assign(maxL + 1, {});
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti)
    if (layer_of_[ti] >= 0) layer_tensors_[(size_t)layer_of_[ti]].push_back((int)ti);
  // Auto-disable layer-wise streaming when the full model fits comfortably
  // in VRAM. Streaming evicts + rebuilds every layer on each advance, so a
  // small model (0.8B: ~0.2GB) pays a full malloc_shared + host repack per
  // tensor per token for no benefit. begin_layer() early-returns on an
  // empty layer_tensors_, so clearing it disables eviction entirely.
  // Threshold 4GB: far above small models, far below 27B (~23GB).
  {
    uint64_t total_w = 0;
    for (const auto& t : model_.tensors) total_w += t.weight_bytes();
    // Per-device footprint ≈ weights + act/out working buffers (small).
    if (total_w < 4ULL * 1024 * 1024 * 1024) {
      layer_tensors_.clear();
      std::cerr << "Engine: streaming disabled (model "
                << total_w / (1024 * 1024) << " MB fits in VRAM)\n";
    }
  }
}

int Engine::layer_of_name(const std::string& name) {
  size_t p = name.find("model.language_model.layers.");
  if (p == std::string::npos) return -1;
  int L = std::atoi(name.c_str() + p + 28);
  if (L < 0) return -1;
  // Guard against atoi garbage on malformed names: require digits.
  const char* s = name.c_str() + p + 28;
  if (*s < '0' || *s > '9') return -1;
  return L;
}

void Engine::release_shard(int tensor_index) {
  if (pinned_) return;  // preload_all(): everything stays resident.
  if (tensor_index < 0 || (size_t)tensor_index >= shard_cache_.size()) return;
  auto& sc = shard_cache_[(size_t)tensor_index];
  if (!sc.built) return;
  // Park buffers in the shape-keyed pool instead of freeing (Phase 3): the
  // same shapes recur every layer, so steady-state rebuild is acquire+memcpy
  // with no L0 alloc or page re-pinning. Sizes recomputed exactly as in
  // ensure_shard_built (host-cache sizes when present, else model geometry).
  const auto& t = model_.tensors[tensor_index];
  // run_linear_tp waits on kernel events before readback, so by the time a
  // later begin_layer() releases this shard no kernel can still reference
  // it. Belt-and-braces queue sync (already complete => ~free).
  for (size_t d = 0; d < devices_.size(); ++d) {
    devices_[d].queue.wait();
    size_t code_bytes = sc.has_host_cache && d < sc.code_host.size()
        ? sc.code_host[d].size() : 0;
    size_t scale_bytes = sc.has_host_cache && d < sc.scale_host.size()
        ? sc.scale_host[d].size() * sizeof(uint16_t) : 0;
    // TODO 6: DPAS shards hold repacked tiles in USM (dpas_bytes), not the
    // row-major code_host bytes — pool/release by the USM size.
    if (sc.dpas48 && d < sc.dpas_bytes.size() && sc.dpas_bytes[d] > 0)
      code_bytes = sc.dpas_bytes[d];
    if (code_bytes == 0 || scale_bytes == 0) {
      // No host cache (e.g. FP16 first-build): derive from model geometry.
      // FP16 code = full N_local*K halves; INT* packed = via group math.
      // These shards are resident in practice; this is a safe fallback.
      auto& rng = shards_[tensor_index].ranges;
      uint32_t Nl = (d < rng.size()) ? rng[d].second - rng[d].first : 0;
      uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
      uint32_t ng = (t.K + G - 1) / G;
      const int bits = (sc.scheme == gemm::WeightScheme::INT2) ? 2
                     : (sc.scheme == gemm::WeightScheme::INT4) ? 4
                     : (sc.scheme == gemm::WeightScheme::FP16) ? 16 : 8;
      const int pack = (bits >= 8) ? 1 : (8 / bits);
      code_bytes = (sc.scheme == gemm::WeightScheme::FP16)
          ? (size_t)Nl * t.K * 2
          : (size_t)Nl * ((t.K + pack - 1) / pack);
      scale_bytes = (size_t)Nl * ng * sizeof(uint16_t);
    }
    if (sc.code_usm.size() > d && sc.code_usm[d]) {
      pool_release(d, sc.code_usm[d], code_bytes);
      sc.code_usm[d] = nullptr;
    }
    if (sc.scale_usm.size() > d && sc.scale_usm[d]) {
      pool_release(d, sc.scale_usm[d], scale_bytes);
      sc.scale_usm[d] = nullptr;
    }
    // *** INTFER_MLP_PARTIAL_DOWN: K-split shard release. *** Same shape-keyed
    // pool, different layout (full N x K-slice). Sizes come from the host
    // cache when present, else from the ksplit geometry (N=t.N rows, K=ksplit_K).
    if (sc.ksplit) {
      size_t code_bytes_k = sc.has_host_cache && d < sc.ksplit_code_host.size()
          ? sc.ksplit_code_host[d].size() : 0;
      size_t scale_bytes_k = sc.has_host_cache && d < sc.ksplit_scale_host.size()
          ? sc.ksplit_scale_host[d].size() * sizeof(uint16_t) : 0;
      if (code_bytes_k == 0 || scale_bytes_k == 0) {
        const uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
        const uint32_t ng = (sc.ksplit_K + G - 1) / G;
        const int bits = (sc.scheme == gemm::WeightScheme::INT2) ? 2
                       : (sc.scheme == gemm::WeightScheme::INT4) ? 4
                       : (sc.scheme == gemm::WeightScheme::FP16) ? 16 : 8;
        const int pack = (bits >= 8) ? 1 : (8 / bits);
        code_bytes_k = (sc.scheme == gemm::WeightScheme::FP16)
            ? (size_t)t.N * sc.ksplit_K * 2
            : (size_t)t.N * ((sc.ksplit_K + pack - 1) / pack);
        scale_bytes_k = (size_t)t.N * ng * sizeof(uint16_t);
      }
      if (sc.ksplit_code_usm.size() > d && sc.ksplit_code_usm[d]) {
        pool_release(d, sc.ksplit_code_usm[d], code_bytes_k);
        sc.ksplit_code_usm[d] = nullptr;
      }
      if (sc.ksplit_scale_usm.size() > d && sc.ksplit_scale_usm[d]) {
        pool_release(d, sc.ksplit_scale_usm[d], scale_bytes_k);
        sc.ksplit_scale_usm[d] = nullptr;
      }
    }
    // NOTE: alpha is host-side only (global_host); no device global buffer.
    // NOTE: no per-shard act/out — the shared workspaces (ws_) serve all
    // shards, so release only the weight buffers (code/scale).
  }
  sc.code_usm.clear(); sc.scale_usm.clear();
  sc.ksplit_code_usm.clear(); sc.ksplit_scale_usm.clear();
  sc.ksplit = false; sc.ksplit_K = 0;
  sc.built = false;
}

// Wave workspaces for run_linear_wave (see engine.hpp): per-device slot
// pairs, one slot per live wave item. Grow-only; queue-wait before
// realloc (same aliasing contract as ensure_workspace: slots never live
// across calls, every wave event is waited before readback).
void Engine::ensure_wave_workspace(uint32_t M, uint32_t W, uint32_t K,
                                   uint32_t Nl) {
  if (M == 0) M = 1;
  if (W == 0) W = 1;
  if (wv_.size() == devices_.size() * (size_t)wv_W_ && wv_W_ > 0 &&
      M <= wv_M_ && W <= wv_W_ && K <= wv_K_ && Nl <= wv_Nl_)
    return;
  for (size_t d = 0; d < devices_.size(); ++d) devices_[d].queue.wait();
  const uint32_t nM = std::max(M, wv_M_);
  const uint32_t nW = std::max(W, wv_W_);
  const uint32_t nK = std::max(K, wv_K_);
  const uint32_t nNl = std::max(Nl, wv_Nl_);
  std::vector<WaveSlot> nwv(devices_.size() * (size_t)nW);
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    for (uint32_t i = 0; i < nW; ++i) {
      WaveSlot s;
      s.act = static_cast<std::uint16_t*>(sycl::malloc_shared(
          (size_t)nM * nK * sizeof(uint16_t), dev.sycl_device, dev.sycl_context));
      // S1: the .out slot is DEVICE-WRITES / HOST-READS, so it is the only
      // buffer eligible for the uncached-host allocation. .act stays
      // malloc_shared (host-writes / device-reads -- see uncached_out).
      {
        void* uc = uncached_out::alloc(
            (size_t)nM * nNl * sizeof(uint16_t), dev);
        if (uc != nullptr) {
          s.out = static_cast<std::uint16_t*>(uc);
          s.out_uncached = true;
          uncached_out::announce("wave", true);
        } else {
          s.out = static_cast<std::uint16_t*>(sycl::malloc_shared(
              (size_t)nM * nNl * sizeof(uint16_t), dev.sycl_device,
              dev.sycl_context));
          uncached_out::announce("wave", false);
        }
      }
      if (!s.act || !s.out) {
        std::fprintf(stderr,
                     "[intfer] FATAL: wave workspace alloc failed on device %zu "
                     "(M=%u W=%u K=%u Nl=%u)\n", d, nM, nW, nK, nNl);
        std::fflush(stderr);
        std::abort();
      }
      nwv[d * nW + i] = s;
    }
  }
  // Free old slots after the new ones are live (queues already waited).
  for (size_t i = 0; i < wv_.size(); ++i) {
    const size_t dd = wv_W_ > 0 ? i / wv_W_ : 0;
    if (dd < devices_.size()) {
      if (wv_[i].act) sycl::free(wv_[i].act, devices_[dd].queue);
      if (wv_[i].out) {
        if (wv_[i].out_uncached)
          uncached_out::release(wv_[i].out, devices_[dd]);
        else
          sycl::free(wv_[i].out, devices_[dd].queue);
      }
    }
  }
  wv_ = std::move(nwv);
  wv_W_ = nW; wv_M_ = nM; wv_K_ = nK; wv_Nl_ = nNl;
}

// Shared per-device act/out workspaces (replaces per-shard act/out USM).
// One buffer pair per device serves all shards: a shard's act/out are
// written+consumed within one run_linear_tp call and never live across calls
// (single-threaded CLI), and every kernel event is waited before readback,
// so aliasing across shards is safe. Grow-only, direct malloc_shared (NOT
// the shape pool — pool entries stay live and would pin the old size
// resident). K/Nl vary per tensor, so the workspace covers the max seen.
void Engine::ensure_workspace(uint32_t M, uint32_t K, uint32_t Nl_max) {
  if (M == 0) M = 1;
  if (ws_.size() != devices_.size()) ws_.resize(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& w = ws_[d];
    if (w.act && w.out && M <= w.M && K <= w.K && Nl_max <= w.Nl) continue;
    devices_[d].queue.wait();
    uint32_t nM = std::max(M, w.M);
    uint32_t nK = std::max(K, w.K);
    uint32_t nNl = std::max(Nl_max, w.Nl);
    // The cached activation bytes lived in the buffer being freed. Clearing
    // here is MANDATORY, not tidy: a surviving cache would let the next call
    // skip an upload into memory that no longer exists.
    act_cache_ptr_ = nullptr; act_cache_M_ = 0; act_cache_K_ = 0;
    if (w.act) { sycl::free(w.act, devices_[d].queue); w.act = nullptr; }
    if (w.out) {
      if (w.out_uncached) uncached_out::release(w.out, devices_[d]);
      else sycl::free(w.out, devices_[d].queue);
      w.out = nullptr;
    }
    w.out_uncached = false;
    auto& dev = devices_[d];
    w.act = static_cast<std::uint16_t*>(sycl::malloc_shared(
        (size_t)nM * nK * sizeof(uint16_t), dev.sycl_device, dev.sycl_context));
    // S1: .out is DEVICE-WRITES / HOST-READS, so it is the only buffer
    // eligible for the uncached-host allocation. This one buffer is what the
    // three named seams read back through -- run_linear_tp (ws_[d].out),
    // run_mlp_partial_down (ws_[d].out) and la_out_tp2 (ws_[0].out) -- so one
    // allocation site covers all three. .act stays malloc_shared (host-writes
    // / device-reads -- see uncached_out).
    {
      void* uc = uncached_out::alloc((size_t)nM * nNl * sizeof(uint16_t), dev);
      if (uc != nullptr) {
        w.out = static_cast<std::uint16_t*>(uc);
        w.out_uncached = true;
        uncached_out::announce("device", true);
      } else {
        w.out = static_cast<std::uint16_t*>(sycl::malloc_shared(
            (size_t)nM * nNl * sizeof(uint16_t), dev.sycl_device,
            dev.sycl_context));
        uncached_out::announce("device", false);
      }
    }
    if (!w.act || !w.out) {
      std::fprintf(stderr,
                   "[intfer] FATAL: workspace alloc failed on device %zu "
                   "(M=%u K=%u Nl=%u)\n", d, nM, nK, nNl);
      std::fflush(stderr);
      std::abort();
    }
    w.M = nM; w.K = nK; w.Nl = nNl;
  }
}

// TODO 6 (W4A8/W2A8): grow-only per-device INT8-act + act-scale workspaces.
// GA=64 fixed (matches model qcfg act INT8_G64); nGA covers max K seen.
void Engine::ensure_workspace48(uint32_t M, uint32_t K) {
  constexpr uint32_t GA = 64;
  if (M == 0) M = 1;
  if (ws48_.size() != devices_.size()) ws48_.resize(devices_.size());
  const uint32_t nGA = (K + GA - 1) / GA;
  const uint32_t nOp = K / 32;  // DPAS K-op = 32
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& w = ws48_[d];
    if (w.act && w.as && w.f32 && w.si && w.fold && M <= w.M && K <= w.K &&
        nGA <= w.nGA && nOp <= w.nOp)
      continue;
    devices_[d].queue.wait();
    uint32_t nM = std::max(M, w.M);
    uint32_t nK = std::max(K, w.K);
    uint32_t nnGA = std::max(nGA, w.nGA);
    uint32_t nnOp = std::max(nOp, w.nOp);
    if (w.act) { sycl::free(w.act, devices_[d].queue); w.act = nullptr; }
    if (w.as) { sycl::free(w.as, devices_[d].queue); w.as = nullptr; }
    if (w.f32) { sycl::free(w.f32, devices_[d].queue); w.f32 = nullptr; }
    if (w.si) { sycl::free(w.si, devices_[d].queue); w.si = nullptr; }
    if (w.fold) { sycl::free(w.fold, devices_[d].queue); w.fold = nullptr; }
    auto& dev = devices_[d];
    w.act = static_cast<std::int8_t*>(sycl::malloc_shared(
        (size_t)nM * nK, dev.sycl_device, dev.sycl_context));
    w.as = static_cast<float*>(sycl::malloc_shared(
        (size_t)nM * nnGA * sizeof(float), dev.sycl_device, dev.sycl_context));
    w.f32 = static_cast<float*>(sycl::malloc_shared(
        (size_t)nM * nK * sizeof(float), dev.sycl_device, dev.sycl_context));
    w.si = static_cast<float*>(sycl::malloc_shared(
        2 * (size_t)nM * nnGA * sizeof(float), dev.sycl_device, dev.sycl_context));
    // 2 * M * nOp INT32 (S0 + Sfull per op). Guard the multiply: nnOp is
    // derived from nK, and the count is allocated 64-bit.
    w.fold = static_cast<std::int32_t*>(sycl::malloc_shared(
        (size_t)2 * nM * nnOp * sizeof(std::int32_t), dev.sycl_device,
        dev.sycl_context));
    if (!w.act || !w.as || !w.f32 || !w.si || !w.fold) {
      std::fprintf(stderr,
                   "[intfer] FATAL: workspace48 alloc failed on device %zu "
                   "(M=%u K=%u nGA=%u nOp=%u)\n", d, nM, nK, nnGA, nnOp);
      std::fflush(stderr);
      std::abort();
    }
    w.M = nM; w.K = nK; w.nGA = nnGA; w.nOp = nnOp;
  }
}

void Engine::begin_layer(int L) {
  // *** INTFER_STREAM_F16 ACCEPTANCE ANNOUNCEMENT. *** The flag moves WHEN a
  // shard is released, never WHAT: the eviction predicate is unchanged, so the
  // two streams are byte-identical by construction and a stream comparison is
  // satisfiable by a flag that never reached begin_layer. The line names the
  // policy actually in force (evict-FP16 vs keep-FP16-resident), once per
  // process, so the arm certifies the policy rather than a constant.
  //
  // It goes BEFORE the pinned_ early return on purpose: preload_all() pins
  // every shard and begin_layer then returns without touching a single one,
  // so the steady-state forward pass never evicts anything regardless of the
  // flag. A line placed after the guard would only ever print on the warmup
  // request -- and on a box where the first request is the only one that
  // evicts, never. The policy is a property of the engine, not of the
  // request, so it announces before the short-circuit.
  static const bool stream_f16 = (std::getenv("INTFER_STREAM_F16") != nullptr);
  static bool announced_stream_f16 = false;
  if (stream_f16 && !announced_stream_f16) {
    announced_stream_f16 = true;
    std::fprintf(stderr,
                 "[stream-f16] FP16 shard eviction policy ACTIVE "
                 "(policy=evict-everything; INTFER_STREAM_F16=1)\n");
    std::fflush(stderr);
  }
  if (pinned_) return;  // preload_all(): steady-state zero evict+rebuild.
  if (layer_tensors_.empty()) return;
  // Phase 1: FP16-scheme shards are memcpy-identity slices (no unpack work)
  // and small enough to stay resident: layered FP16 is ~13.5GB total, i.e.
  // ~6.7GB per device under TP=2, well under the 12GB B580 budget
  // (plus ~2.5GB cached lm_head and the INT4 streaming working set).
  // Skipping their eviction kills ~75% of per-token evict+rebuild traffic.
  // INTFER_STREAM_F16=1 restores the old evict-everything behavior.
  for (size_t ti = 0; ti < shard_cache_.size(); ++ti) {
    if (!shard_cache_[ti].built) continue;
    if (!stream_f16 && shard_cache_[ti].scheme == gemm::WeightScheme::FP16)
      continue;
    int tl = (ti < layer_of_.size()) ? layer_of_[ti] : -1;
    if (tl >= 0 && tl != L) release_shard((int)ti);
  }
}

// Full-VRAM residency: build every runnable Linear shard once and pin it.
// Only GEMM-runnable roles are built (ATTN Q/K/V/O, MLP gate/up/down, linear
// attention QKV/Z/B/A/OUT). Skipped: lm_head-named tensors (the CLI serves
// lm_head through its own TP-sharded run_lm_head_f16 upload, never through
// engine shards — building the slot would waste 1.27GB/dev), norms/vectors
// (K=1), conv1d and OTHER roles (host-side; ~1.27GB/dev of dead shards).
// Anything missed lazy-builds on first run_linear_tp (correct by
// construction: pinning blocks eviction, never construction).
// Afterwards the INT4 host repack cache (~9.4GB host RAM on 27B-sc) is
// dropped — with eviction disabled no rebuild can ever need it. Bit-identical:
// same bytes as lazy builds.
void Engine::preload_all() {
  if (pinned_) return;
  // No rebuilds will ever happen: skip the host repack cache entirely
  // (saves ~9.4GB host RAM; without this the OOM killer takes the preload).
  skip_host_cache_ = true;
  auto is_runnable = [](TensorRole r) {
    switch (r) {
      case TensorRole::ATTN_Q: case TensorRole::ATTN_K:
      case TensorRole::ATTN_V: case TensorRole::ATTN_O:
      case TensorRole::MLP_GATE: case TensorRole::MLP_UP:
      case TensorRole::MLP_DOWN:
      case TensorRole::LINEAR_ATTN_QKV: case TensorRole::LINEAR_ATTN_Z:
      case TensorRole::LINEAR_ATTN_B: case TensorRole::LINEAR_ATTN_A:
      case TensorRole::LINEAR_ATTN_OUT:
        return true;
      default:
        return false;
    }
  };
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t n_built = 0, n_skip = 0;
  uint64_t dev_bytes = 0;
  // Bisection knob (DEVICE_LOST triage): cap the number of preloaded shards.
  // Anything not preloaded lazy-builds on first use (correct by construction).
  size_t preload_max = (size_t)-1;
  if (const char* pm = std::getenv("INTFER_PRELOAD_MAX")) {
    long v = std::atol(pm); if (v > 0) preload_max = (size_t)v;
  }
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti) {
    const auto& t = model_.tensors[ti];
    if (t.name.find("lm_head") != std::string::npos) { n_skip++; continue; }
    if (!is_runnable(t.role)) { n_skip++; continue; }
    if (t.N == 0 || t.K == 0) { n_skip++; continue; }
    if (shard_cache_[ti].built) continue;
    if (n_built >= preload_max) continue;
    ensure_shard_built((int)ti);
    n_built++;
    if ((n_built % 100) == 0) {
      std::cerr << "Engine: preload " << n_built << " shards...\n";
      std::cerr.flush();
    }
    if (const char* pt = std::getenv("INTFER_PRELOAD_TRACE")) {
      // Per-shard trace: cumulative geometry bytes per device + tensor name.
      // Correlates engine accounting with xpu-smi to find device-asymmetric
      // growth (dev0-only jumps). Enable with INTFER_PRELOAD_TRACE=1.
      static uint64_t trace_acc[8] = {0,0,0,0,0,0,0,0};
      auto& rng = shards_[ti].ranges;
      for (size_t d = 0; d < devices_.size() && d < 8; ++d) {
        uint32_t Nl = (d < rng.size()) ? rng[d].second - rng[d].first : 0;
        uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
        uint32_t ng = (t.K + G - 1) / G;
        const int tbits = (t.qcfg.scheme == QuantScheme::INT2) ? 2
                        : (t.qcfg.scheme == QuantScheme::INT4) ? 4
                        : (t.qcfg.scheme == QuantScheme::FP16) ? 16 : 8;
        const int tpack = (tbits >= 8) ? 1 : (8 / tbits);
        trace_acc[d] += (tbits == 16)
            ? (uint64_t)Nl * t.K * 2 + (uint64_t)Nl * ng * 2
            : (uint64_t)Nl * ((t.K + tpack - 1) / tpack)
              + (uint64_t)Nl * ng * 2;
      }
      auto now = std::chrono::high_resolution_clock::now();
      double el = std::chrono::duration<double>(now - t0).count();
      std::cerr << "Engine: preload #" << n_built << " t=" << (long)el
                << "s dev0=" << trace_acc[0] / (1024*1024) << "MiB dev1="
                << (devices_.size() > 1 ? trace_acc[1] / (1024*1024) : 0)
                << "MiB " << t.name << "\n";
      std::cerr.flush();
    }
  }
  // Pin first: any concurrent begin_layer() (none expected — single-threaded
  // CLI, but belt-and-braces) must not evict what we just built.
  // (skip_host_cache_ stays on: pinned shards are never rebuilt, and any
  // lazy-built straggler is single-use — caching it would be dead RAM.)
  pinned_ = true;
  // NOTE (2026-09-21, measured on 27B TP=2): shard USM is malloc_shared and
  // pages stay host-resident until first kernel touch — xpu-smi shows
  // dev0 854MB/dev1 115MB at startup, jumping to ~11GB only after the first
  // request. A q.prefetch(code)+q.prefetch(scale) pass over all shards here
  // was tried and REVERTED: 18.7GB "migrated" in 59ms per the log, but
  // xpu-smi was unchanged and the first-request jump identical — prefetch
  // is a no-op hint for shared allocations on this Level Zero driver.
  // First-request migration stalls stand; do NOT re-add prefetch as-is.
  // (A real touch kernel over every shard at startup might migrate pages,
  // but that is 496 extra launches + waits in the startup path — unscoped.)
  for (size_t d = 0; d < devices_.size(); ++d) {
    devices_[d].queue.wait();
    // Tally resident weight bytes per device (code+scale+global).
    uint64_t per_dev = 0;
    for (size_t ti = 0; ti < shard_cache_.size(); ++ti) {
      const auto& sc = shard_cache_[ti];
      if (!sc.built || d >= sc.code_usm.size() || !sc.code_usm[d]) continue;
      const auto& t = model_.tensors[ti];
      auto& rng = shards_[ti].ranges;
      uint32_t Nl = (d < rng.size()) ? rng[d].second - rng[d].first : 0;
      uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
      uint32_t ng = (t.K + G - 1) / G;
      const int bits = (sc.scheme == gemm::WeightScheme::INT2) ? 2
                     : (sc.scheme == gemm::WeightScheme::INT4) ? 4
                     : (sc.scheme == gemm::WeightScheme::FP16) ? 16 : 8;
      const int pack = (bits >= 8) ? 1 : (8 / bits);
      per_dev += (sc.scheme == gemm::WeightScheme::FP16)
          ? (uint64_t)Nl * t.K * 2
          : (uint64_t)Nl * ((t.K + pack - 1) / pack);
      per_dev += (uint64_t)Nl * ng * sizeof(uint16_t);
      // Alpha is host-side (no device global buffer).
    }
    if (d == 0) dev_bytes = per_dev;
    std::cerr << "Engine: preload device " << d << " resident weights: "
              << per_dev / (1024 * 1024) << " MiB\n";
    std::cerr.flush();
  }
  // Allocator audit: pool contents (should be empty post-preload) + total
  // acquire count, to separate geometry bytes from runtime overhead.
  {
    size_t pool_n = 0; uint64_t pool_bytes = 0;
    for (size_t d = 0; d < usm_pool_.size(); ++d)
      for (auto& [bytes, vec] : usm_pool_[d]) {
        pool_n += vec.size(); pool_bytes += bytes * vec.size();
      }
    std::cerr << "Engine: preload audit: pool holds " << pool_n
              << " bufs (" << pool_bytes / (1024 * 1024)
              << " MiB), pool_acquire calls=" << pool_acquires_
              << " pool_hits=" << pool_hits_ << "\n";
    std::cerr.flush();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cerr << "Engine: preload_all built " << n_built << " shards (skipped "
            << n_skip << ", host cache bypassed), dev0 weights "
            << dev_bytes / (1024 * 1024)
            << " MiB in " << (long)ms << " ms; shards pinned (no evict)\n";
  std::cerr.flush();
}

std::vector<float> Engine::decode_step(const std::vector<float>& token_hidden,
                                       int tensor_index) {
  if (model_.tensors.empty()) return {};
  const int ti = (tensor_index >= 0 &&
                  (size_t)tensor_index < model_.tensors.size())
                     ? tensor_index
                     : 0;
  std::vector<float> out;
  run_linear_tp(ti, token_hidden, out);
  return out;
}
 
std::vector<std::vector<float>> Engine::decode_step_batch(
    const std::vector<std::vector<float>>& batch_hidden,
    int tensor_index) {
  std::vector<std::vector<float>> out;
  if (batch_hidden.empty()) return out;
  const uint32_t M = static_cast<uint32_t>(batch_hidden.size());
  const int ti = (tensor_index >= 0 &&
                  (size_t)tensor_index < model_.tensors.size())
                     ? tensor_index
                     : 0;
  const auto& t = model_.tensors[ti];
  std::vector<float> x;
  x.reserve((size_t)M * t.K);
  for (const auto& row : batch_hidden) x.insert(x.end(), row.begin(), row.end());
  std::vector<float> y;
  run_linear_tp(ti, x, y, M);
  out.reserve(M);
  for (uint32_t i = 0; i < M; ++i)
    out.emplace_back(y.begin() + i * t.N, y.begin() + (i + 1) * t.N);
  return out;
}
 
void Engine::run_linear_tp(int tensor_index, const std::vector<float>& x,
                           std::vector<float>& out) {
  run_linear_tp(tensor_index, x, out, 1);
}

// TODO 6 (W4A8/W2A8): DPAS dispatch. Fused host act-quant (FP32 -> INT8,
// per-row GA=64 slices, symmetric) writes the shared ws48_ workspaces;
// the kernel then runs fully from device-resident data. Readback reuses
// ws_[d].out (M*N_local FP16). Same event-wait contract as run_linear_tp.
void Engine::run_linear_tp_dpas48(int tensor_index,
                                  const std::vector<float>& x,
                                  std::vector<float>& out, uint32_t M) {
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (M == 0) M = 1;
  const uint32_t out_elems = M * t.N;
  if (out.size() != out_elems) out.resize(out_elems);
  const auto& sc = shard_cache_[tensor_index];
  constexpr uint32_t GA = 64;
  const uint32_t K = t.K;
  const uint32_t nGA = (K + GA - 1) / GA;
  ensure_workspace48(M, K);

  const int bits = (sc.scheme == gemm::WeightScheme::INT2)    ? 2
                   : (sc.scheme == gemm::WeightScheme::INT4)  ? 4
                                                              : 8;
  const uint32_t Gw = (sc.group_size == 0) ? K : sc.group_size;
  // EXPERIMENTAL row-cooperative geometry (default-off INTFER_DPAS_ROW=1;
  // measured ~460x slower — experiment base only). Eligibility is the KERNEL'S
  // OWN predicate (dpas4_row_eligible), not a second copy of it: this copy used
  // to live here and to agree only by coincidence, and it is the copy that makes
  // the kernel's decline path unreachable. That guarantee cannot rest on the
  // event-identity test below -- measured 2026-10-03, an event a callee returns
  // as a bare sycl::event{} does NOT compare equal to one built here, so the
  // `== sycl::event{}` half of the fallback cannot be relied on (dpas_fold_probe
  // asserts both directions).
  const bool use_row = dpas_row_enabled() && gemm::dpas4_row_eligible(bits, Gw);
  // Device act-quant pre-pass (INTFER_DPAS_PREQ=1, default OFF): per device
  // one FP32 H2D + one GA64 quant kernel, dep-chained into that rank's DPAS
  // launch. Kills the host quant loop and adds no host wait. Per device
  // (NOT one launch for both): each rank owns its sycl::context, so a kernel
  // cannot legally write the other rank's malloc_shared buffer.
  bool preq_ok = dpas_preq_enabled() && !use_row;
  for (size_t d = 0; d < devices_.size() && preq_ok; ++d)
    preq_ok = ws48_[d].f32 && ws48_[d].act && ws48_[d].as && ws48_[d].si;
  // EXPERIMENTAL DPAS row-fold hoist (default-off INTFER_DPAS48_FOLD=1).
  // Composes with the pre-pass above (the fold dep-chains on it) and with the
  // row-cooperative geometry (which has its own fold handling).
  const bool use_fold = dpas_fold_enabled();
// E4: cooperative 2D row block. Requires M >= 2, which quant_gemm_dpas48_2d
  // ALSO enforces and DECLINES on -- the flag is checked here only so the
  // experiment is inert (and attributable) rather than attempted and declined on
  // every decode call.
  const bool use_tiled = dpas_tiled_enabled() && M >= 2 && !use_row;

  // 1) Activation quant into ws48_ (per row, per GA-slice max-abs scale).
  //    Device pre-pass first; on ANY decline fall through to the host loop,
  //    which rewrites every device's rows — never partial work.
  std::vector<sycl::event> preq_events;
#ifdef INTFER_PROFILE_GPU
  auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
  if (preq_ok) {
    preq_events.resize(devices_.size());
    // (scale, inverse) pairs on the HOST, verbatim. The device must neither
    // divide nor re-reduce: q = trunc(x*inv +- 0.5) turns a 1-ulp reciprocal
    // difference into a whole +-1 code error at .5 ties, and `1.0f/s` is not a
    // portable bit-source (device lowers it to rcp.approx). Shipping the host's
    // bits -- all produced by dashq's single noinline ga64_recip site, gated
    // by scripts/test_ga64_recip.sh -- makes the pre-pass bit-exact by
    // construction. O(M*K) work + a tiny H2D; this replaces the O(M*K) host
    // quant loop it was introduced to delete (but not its max-abs reduction),
    // and stays off the GEMM's critical path (dep-chained, no added wait).
    if (si48_staging_.size() != 2 * (size_t)M * nGA)
      si48_staging_.resize(2 * (size_t)M * nGA);
    // nGA is re-derived inside the helper from K (identical expression to the
    // kernel's), so the staging size above and the kernel's slice count cannot
    // drift apart.
    dashq::act_quant_ga64_scale_inv_batch(x.data(), M, K,
                                          si48_staging_.data());
    // Stop submitting once any device declines: the remaining devices are
    // about to be overwritten by the host fallback anyway.
    for (size_t d = 0; d < devices_.size() && preq_ok; ++d) {
      sycl::queue& q = devices_[d].queue;
      float* xd = ws48_[d].f32;
      // Unconditional H2D. `x` is a host std::vector; `xd` is device USM,
      // so an address comparison can never legitimately match (and must not
      // be used to skip the copy — the source is host memory either way).
      q.memcpy(xd, x.data(), (size_t)M * K * sizeof(float));
      gemm::ActQuantDevParams ap;
      q.memcpy(ws48_[d].si, si48_staging_.data(),
               2 * (size_t)M * nGA * sizeof(float));
      ap.x_f32 = xd;
      ap.scale_inv_in = ws48_[d].si;
      ap.q_out = ws48_[d].act;
      ap.s_out = ws48_[d].as;
      ap.M = M;
      ap.K = K;
      ap.Kstride = ws48_[d].K;
      ap.nGAstride = ws48_[d].nGA;
      preq_events[d] = gemm::act_quant_ga64_dev(q, ap);
      if (preq_events[d] == sycl::event{}) preq_ok = false;
    }
    // PARTIAL-DECLINE GUARD: a pre-pass kernel already submitted for device
    // d' < d is still queued and WILL write ws48_[d'].act asynchronously. If
    // we now fall through to the host fallback, its std::memcpy into the same
    // buffer races that in-flight kernel -> nondeterministic act codes. Drain
    // every already-submitted pre-pass event first (the host path then owns
    // the buffer exclusively and overwrites all of it).
    if (!preq_ok) {
      for (auto& e : preq_events)
        if (e != sycl::event{}) e.wait_and_throw();
      preq_events.clear();
    }
  } else {
    // Host fallback: vectorized GA=64 quant into HOST staging (bit-exact vs
    // the old scalar loop, incl. the trunk's original lround half-away-from-
    // zero); then one memcpy per device row into the shared workspace. Scalar
    // stores straight into malloc_shared are ~100x slower (copy bucket
    // 96ms/7tok); memcpy emits wide vectorized stores. Staging is grow-only
    // (zero steady-state allocs); rows are tight here, workspace rows strided
    // — copy per row.
    //
    // This branch ALSO owns the (scale, inverse) pairs the pre-pass path
    // shipped from the host: both derive s and inv from the same shared
    // ga64_slice_max/ga64_recip sites, so a decline costs time, not codes.
    for (uint32_t m = 0; m < M; ++m) {
      const float* xr = x.data() + (size_t)m * K;
      if (act48_staging_.size() != (size_t)M * K)
        act48_staging_.resize((size_t)M * K);
      if (as48_staging_.size() != (size_t)M * nGA)
        as48_staging_.resize((size_t)M * nGA);
      std::int8_t* qrow = act48_staging_.data() + (size_t)m * K;
      float* srow = as48_staging_.data() + (size_t)m * nGA;
      // Trunk's original loop used std::lround (half away from zero).
      dashq::act_quant_ga64_batch(xr, qrow, srow, K,
                                  dashq::ActQuantRound::HalfAway);
      for (size_t d = 0; d < devices_.size(); ++d) {
        std::memcpy(ws48_[d].act + (size_t)m * ws48_[d].K, qrow, K);
        std::memcpy(ws48_[d].as + (size_t)m * ws48_[d].nGA, srow,
                    (size_t)nGA * sizeof(float));
      }
    }
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
    g_gpu_profile.copy_count++;
  }
#endif

  // 2) Launch per device.
  float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    uint32_t N_local = end - start;
    gemm::Dpas48Params p;
    p.code_dpas = sc.code_usm[d];
    p.scale_usm = sc.scale_usm[d];
    p.act_int8 = ws48_[d].act;
    p.act_scale = ws48_[d].as;
    p.alpha = alpha;
    p.out_usm = ws_[d].out;
    p.M = M;
    p.N = N_local;
    p.K = K;
    p.group_size = sc.group_size;
    p.act_group = GA;
    p.bits = bits;
    p.act_stride = ws48_[d].K;
    p.as_stride = ws48_[d].nGA;
    sycl::event dep = preq_ok ? preq_events[d] : sycl::event{};
    // INTFER_DPAS48_WG (default 0 => 16, shipped): widen the tiled DPAS
    // work-group. Pure partition (each (m,n) written by one block), so bit-exact.
    if(const char* wg_env = std::getenv("INTFER_DPAS48_WG")){
      long v = std::atol(wg_env);
      if(v > 0) p.wg = (uint32_t)v;
    }
#ifdef INTFER_PROFILE_GPU
    auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
    sycl::event kernel_evt = use_row ? gemm::quant_gemm_dpas4_row(q, p, dep)
                          : use_tiled ? gemm::quant_gemm_dpas48_2d(q, p, dep)
                                      : gemm::quant_gemm_dpas48(q, p, dep);
    // THE PATH ANNOUNCES ITSELF, ONCE. The work-group width is a pure
    // partition (each output element (m,n) is written by exactly one block:
    // t = n/16, mg = m/R), so a gate comparing the token stream against the
    // base run cannot tell whether the group widened -- the two streams are
    // identical BY CONSTRUCTION, and a flag that never reached the kernel
    // would pass every stream check while certifying nothing. This one line,
    // host-side, one per process, names the width the kernel actually used, so
    // the arm asserts the thing that changed rather than a constant. Same shape
    // as the [mlp-gateup] ACTIVE line and the [lmhead-tiles] layout line.
    //
    // Scope, stated because it is the difference between a live flag and a
    // no-op: p.wg is read by ALL THREE DPAS kernels (quant_gemm_dpas48,
    // quant_gemm_dpas48_2d and quant_gemm_dpas4_row), so the flag is live on
    // every path and this announcement fires whenever the DPAS path runs.
    static bool announced_wg = false;
    if(!announced_wg){
      announced_wg = true;
      std::fprintf(stderr,
                   "[dpas48-wg] work-group=%u (INTFER_DPAS48_WG=%s) geometry=%s "
                   "M=%u N=%u K=%u\n",
                   (p.wg == 0) ? 16u : p.wg,
                   std::getenv("INTFER_DPAS48_WG") ?
                       std::getenv("INTFER_DPAS48_WG") : "(default 16)",
                   use_tiled ? "tiled2d" : (use_row ? "row" : "tiled1d"),
                   M, N_local, K);
      std::fflush(stderr);
    }
    // Decline detection. BOTH DPAS48 callees now decline the SAME way -- they return
    // `dep` UNCHANGED (quant_gemm.cpp) -- so `kernel_evt == dep` is the reliable
    // test. The `sycl::event{}` half is kept as defence in depth and NOT relied
    // on: measured on this toolchain, an event produced by `return sycl::event{}`
    // does not compare equal to a bare one built here, so it cannot be the
    // mechanism (dpas_fold_probe asserts that measurement in both directions).
    // What actually keeps a decline unreachable is the eligibility predicate the
    // caller checks BEFORE launching (gemm::dpas4_row_eligible, use_tiled's
    // M >= 2) -- one rule, one home, consulted before the launch rather than
    // inferred from a return value.
    const bool declined =
        (kernel_evt == sycl::event{}) || (kernel_evt == dep);
    if (declined) kernel_evt = gemm::quant_gemm_dpas48(q, p, dep);
    kernel_events.push_back(kernel_evt);
    // INTFER_TILE_SCHED (PLAN.md 1.3, default OFF): the prefill tiled kernel
    // can run its n_blocks tiles from a per-device SLM work-stealing claim
    // queue instead of the fixed block%WG assignment. It is a pure dispatch
    // change (each tile computes the same dot products in the same order),
    // so the token stream is bit-identical to the base run and a stream
    // comparison is satisfiable by a flag that never reached the kernel --
    // the nested-flag vacuity the wg and gateup arms defeat with an
    // announcement line. This one line names the dispatch mode the kernel
    // actually used, host-side and once per process, so the arm asserts the
    // thing that changed rather than a constant.
    //
    // n_blocks is the per-device tile count, nMG * nT, where nMG = (M + R - 1)
    // / R and nT = N_local / 16. R is the row block: 1 for the shipped
    // quant_gemm_dpas48 path and 2 for E4's quant_gemm_dpas48_2d, so it is
    // derived from the geometry the launch actually used rather than assumed.
    {
      const uint32_t R_path = use_tiled ? 2u : 1u;
      const uint32_t nT_path = N_local / 16u;
      const uint32_t nMG_path = (M + R_path - 1) / R_path;
      const uint32_t n_blocks = nMG_path * nT_path;
      const uint32_t nWG_path = (p.wg == 0) ? 16u : p.wg;
      static bool announced_ts = false;
      if (!announced_ts) {
        announced_ts = true;
        std::fprintf(stderr,
                     "[tile-sched] INTFER_TILE_SCHED=%s n_blocks=%u nWG=%u "
                     "(per-device SLM work-stealing claim queue)\n",
                     (std::getenv("INTFER_TILE_SCHED") != nullptr) ? "on"
                                                                  : "off",
                     n_blocks, nWG_path);
        std::fflush(stderr);
      }
    }
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
    }
    if (kernel_events.size() > g_gpu_profile.peak_wait_depth)
      g_gpu_profile.peak_wait_depth = kernel_events.size();
#endif
  }
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  for (auto& e : kernel_events) e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += devices_.size();
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif

  // 3) Read back FP16 -> FP32 (same as run_linear_tp tail).
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto [start, end] = ranges[d];
    const std::uint16_t* out_ptr = ws_[d].out;
    const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
    for (uint32_t m = 0; m < M; ++m)
      dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                 out.data() + (size_t)m * t.N + start,
                                 N_local_rb);
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    // Per (item, device), matching run_linear_wave and run_mlp_gateup: this
    // path does devices_.size() separate D2H readbacks (one per rank's
    // shard) and counted them as 1, which made the counter non-comparable
    // with the wave path it feeds into.
    g_gpu_profile.readback_count += (int)devices_.size();
  }
#endif
}

void Engine::run_linear_tp(int tensor_index, const std::vector<float>& x,
                           std::vector<float>& out, uint32_t M) {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size()) return;
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (M == 0) M = 1;
  const uint32_t out_elems = M * t.N;
  if (out.size() != out_elems) out.resize(out_elems);
 
  // Lazy-build the USM shard cache for this tensor on first call.
  if (!shard_cache_[tensor_index].built)
    ensure_shard_built(tensor_index);
  // Shared workspaces cover this call's (M, K, max-Nl) shape (grow-only;
  // the M>1 batched path writes M*K / M*N_local elements, so M=1 sizing
  // would OOB-write). Nl_max is over TP ranks (column sharding of N).
  {
    uint32_t Nl_max = 0;
    for (size_t d = 0; d < ranges.size(); ++d)
      Nl_max = std::max(Nl_max, ranges[d].second - ranges[d].first);
    ensure_workspace(M, t.K, Nl_max);
  }
  const auto& sc = shard_cache_[tensor_index];

  // TODO 6 (W4A8/W2A8) DPAS fast path: fused FP32->INT8 activation quant
  // (per-row, GA=64 slices) + native sub-byte DPAS kernel. Weights stay
  // bit-exact (affine fold); only the act quant is new error vs the
  // FP16-act path. INTFER_DPAS48=0 or ineligible shapes use the legacy path.
  if (sc.dpas48) {
    run_linear_tp_dpas48(tensor_index, x, out, M);
    return;
  }
 
  // INTFER_FUSE_ACT (default OFF): an MLP's gate and up consume the SAME host
  // activation vector, so the sequential path converts and uploads identical bits
  // twice. Skipping the second is BIT-EXACT by construction -- the destination
  // already holds exactly the bytes this call would write -- and it removes a
  // full M*K host convert plus one M*K*2 upload per device, per repeat.
  //
  // It removes HOST work only: no launch is added or removed, and no arithmetic
  // is touched, so it is the "exact" branch of the fusion directive rather than
  // a tolerance question. With the flag OFF the cache is never consulted and
  // never written, so the default path is byte-for-byte the old one.
  const char* fa = std::getenv("INTFER_FUSE_ACT");
  static const bool fuse_act =
      (fa && fa[0] == '1' && fa[1] == '\0') &&
      (std::getenv("INTFER_NO_FUSE_ACT") == nullptr);
  // A hit needs THREE things to agree, and the third is the one the first
  // version was missing. The key alone is an address; `x` is a
  // std::vector<float>& and a vector that is assigned into keeps its heap
  // buffer, so consecutive decode steps hit the same key with different bytes.
  // Comparing the source against the exact FP32 that produced the staged FP16
  // makes the key CONTENT, which is what makes "the destination already holds
  // exactly the bytes this call would write" true rather than merely plausible.
  const size_t act_need = (size_t)M * t.K;
  bool act_hit = fuse_act && act_cache_ptr_ == x.data() &&
                 act_cache_M_ == M && act_cache_K_ == (uint32_t)t.K &&
                 act_cache_gen_ == act_staging_gen_ &&
                 act_src_.size() == act_need &&
                 std::memcmp(act_src_.data(), x.data(),
                             act_need * sizeof(float)) == 0;
  // Pre-convert FP32 activation to FP16 once (shared by all devices), unless
  // the identical bits are already staged in every device's workspace.
  if (!act_hit) {
#ifdef INTFER_PROFILE_GPU
    auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
    if (act_staging_.size() != act_need) act_staging_.resize(act_need);
    ++act_staging_gen_;
    dashq::float_to_half_batch(x.data(), act_staging_.data(), act_need);
    // Shadow the SOURCE, not the staging: staging is regenerated on every miss,
    // so shadowing it would make the comparison a tautology.
    if (fuse_act) {
      if (act_src_.size() != act_need) act_src_.resize(act_need);
      std::memcpy(act_src_.data(), x.data(), act_need * sizeof(float));
    }
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
      g_gpu_profile.copy_count++;
    }
#endif
  }  // !act_hit -- the convert above is skipped entirely on a cache hit
 
  // Issue all device work in parallel and collect kernel events.
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(devices_.size());
 
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    auto N_local = end - start;
 
    // 1) Write FP16 activation into the shared per-device workspace
    // (one wide memcpy, not scalar half-stores into malloc_shared).
    std::uint16_t* const act_ptr = ws_[d].act;
#ifdef INTFER_PROFILE_GPU
    auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
    if (!act_hit)
      std::memcpy(act_ptr, act_staging_.data(),
                  (size_t)M * t.K * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
      g_gpu_profile.usm_write_count += 1;
    }
#endif

    // 2) Launch the USM kernel on the persistent weight buffers.
    // Alpha comes from the host copy (kernel takes it by value; no device
    // alloc for 4 bytes).
    float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
    auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
    sycl::event kernel_evt;
    // INTFER_XMX_GEMM (default OFF, presence enables): the XMX joint_matrix
    // GEMM path. The kernel declines (returns {}) on FP16/INT8/M<8 or
    // non-tile shapes; the caller's fallback is the same as before.
    //
    // Work-stealing claim counter: ONE uint32_t per device, zero-init
    // by the caller. The kernel claims tiles on a SINGLE shared
    // address (tile_ctr[0]) with a device-scope atomic -- not one
    // slot per work-group, which would make every group claim the
    // same tiles independently. The pointer handed to the kernel is
    // this device's slot, so one slot per device is exactly enough.
    const char* xmx_env = std::getenv("INTFER_XMX_GEMM");
    static std::vector<std::uint32_t> xmx_tile_ctr;
    static bool xmx_announced = false;
    if (xmx_env) {
      const uint32_t gx = (M / 16);
      const uint32_t gy = (N_local / 16);
      const uint32_t n_tiles = gx * gy;
      const bool ts_env = (std::getenv("INTFER_XMX_TILE_SCHED") != nullptr);
      const uint32_t nWG = ts_env ? 16u : n_tiles;
      if (xmx_tile_ctr.size() < devices_.size())
        xmx_tile_ctr.resize(devices_.size(), 0);
      xmx_tile_ctr[d] = 0;
      bool xmx_ran = false;
      kernel_evt = gemm::quant_gemm_forward_usm_xmx(
          q, ws_[d].act, sc.code_usm[d], sc.scale_usm[d],
          alpha, ws_[d].out, M, N_local, t.K,
          sc.group_size, sc.scheme, sycl::event{},
          &xmx_ran, xmx_tile_ctr.data() + d);
      // A null event means the kernel DECLINED this shape (M<8, non-tile
      // residual, FP16/INT8 scheme) and wrote nothing to ws_[d].out. The
      // caller MUST fall back to the tiled kernel in that case -- a null
      // event is not a no-op here, it is an unwritten buffer. The old code
      // pushed the null event and read back garbage on every declined tile.
      if (!xmx_ran) {
        kernel_evt = gemm::quant_gemm_forward_usm(q,
                                  ws_[d].act,
                                  sc.code_usm[d],
                                  sc.scale_usm[d],
                                  alpha,
                                  ws_[d].out,
                                  M, N_local, t.K,
                                  sc.group_size,
                                  sc.scheme,
                                  sycl::event{});
      }
      xmx_note_dispatch(xmx_ran);
      // Announce the tile scheduler state ONCE per process, ONLY when the
      // XMX path actually ran. The line prints n_tiles and nWG so the gate
      // can verify the work-stealing path was taken. Gating on `xmx_ran`
      // makes the announcement a proof of execution rather than a proof
      // that the env var was read (the engine used to print it
      // unconditionally, so a gate asserting it would pass on a fixture
      // where the kernel declined every tile and the fallback wrote
      // nothing -- a vacuous-green defect).
      if (xmx_ran && !xmx_announced) {
        xmx_announced = true;
        std::fprintf(stderr,
                     "[xmx-gemm] ACTIVE on eligible INT2/INT4 tiles "
                     "(INTFER_XMX_GEMM=1)\n"
                     "[xmx-tile-sched] %s n_tiles=%u nWG=%u "
                     "(%s work-stealing)\n",
                     ts_env ? "ACTIVE" : "OFF",
                     n_tiles, nWG,
                     ts_env ? "persistent" : "fixed-assignment");
      }
    } else {
      kernel_evt = gemm::quant_gemm_forward_usm(q,
                                ws_[d].act,
                                sc.code_usm[d],
                                sc.scale_usm[d],
                                alpha,
                                ws_[d].out,
                                M, N_local, t.K,
                                sc.group_size,
                                sc.scheme,
                                sycl::event{});
    }
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
      g_gpu_profile.submit_count += 1;
    }
#endif
#ifdef INTFER_PROFILE_GPU
    if (kernel_events.size() + 1 > g_gpu_profile.peak_wait_depth)
      g_gpu_profile.peak_wait_depth = kernel_events.size() + 1;
#endif
    kernel_events.push_back(kernel_evt);
  }
 
  // The bytes are now in EVERY device's ws_[d].act, so the cache may record
  // them. Recorded AFTER the per-device loop, never before: recording early
  // would let a hit skip an upload that has not happened yet.
  if (fuse_act) {
    act_cache_ptr_ = x.data();
    act_cache_M_ = M;
    act_cache_K_ = (uint32_t)t.K;
    act_cache_gen_ = act_staging_gen_;
    static bool announced_act = false;
    // Gated on act_hit, not on fuse_act. The flag only says the cache is
    // CONSULTED; a line printed on the flag announces "the knob was read",
    // which is exactly what the arm's stream comparison already cannot
    // distinguish from a flag that did nothing. MEASURED: with -n 1 there is
    // a single decode step after an M=5 prefill, so act_cache_M_ (5) can
    // never equal M (1) and act_hit is unreachable -- the old flag-gated line
    // still printed, so the gate's "announced a staging HIT" assertion was
    // satisfiable with zero hits on the whole run.
    if (act_hit && !announced_act) {
      announced_act = true;
      std::fprintf(stderr,
                   "[fuse-act] activation staging reuse ACTIVE "
                   "(identical FP16 bits already in every device workspace)\n");
    }
  }

  // Wait for all kernels, then read back from the shared workspace.
  // NOTE: the wait is load-bearing, not profiling — without it the host
  // reads ws out before the kernel runs (fresh zero pages => all-zero
  // GEMM outputs). Only the timing accumulation is profiling-gated.
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
  // Device-timeline marker: the region between this and the readback
  // marker below is the kernel+wait region. On this box the per-event
  // kernel timestamps are dead (submit==start==end), so a marker pair is
  // the only way to attribute wall-clock to the compute region rather
  // than to the host. See timeline_mark in engine.cpp.
  intfer::timeline_mark(devices_[0].queue, "run_linear_tp:compute");
#endif
  for (size_t d = 0; d < devices_.size(); ++d) {
    kernel_events[d].wait();
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += devices_.size();
  // Device-side kernel time from event profiling timestamps (the queue is
  // created with enable_profiling). This separates actual XMX/compute time
  // from host-side launch + wait overhead.
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }

  auto t_rb0 = std::chrono::high_resolution_clock::now();
  intfer::timeline_mark(devices_[0].queue, "run_linear_tp:readback");
#endif
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto [start, end] = ranges[d];
    const std::uint16_t* out_ptr = ws_[d].out;
    const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
    for (uint32_t m = 0; m < M; ++m)
      dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                 out.data() + (size_t)m * t.N + start,
                                 N_local_rb);
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
  intfer::timeline_mark(devices_[0].queue, "run_linear_tp:done");
#endif
}

// Fused MLP down-projection (TP=1 only): gate/up convert + submit
// back-to-back, chain on-device through the fused silu-mul + FP16 convert
// straight into the down GEMM, then ONE host wait + ONE D2H. Saves vs the
// wave + host path: 1 wait, 2 D2H readbacks (gate/up), the host silu-mul
// + convert, and 1 H2D (down act). Bit-contract: the fused kernel is the
// same FP32 op order as the host loop; sycl::exp may differ from host
// expf by ~1ulp on rare elements, so the oracle token stream is the
// arbiter (same standing as the AVX2 host-math change). The FP16
// down-convert is RNE bit-identical to dashq::float_to_half by
// construction (dev_float_to_half); the CPU parity gate verifies it, the
// oracle gate otherwise. False (caller keeps host path) on TP>1, DPAS48
// items, bad shapes, or any missing index — never partial work.
bool Engine::run_mlp_fused(int gate_ti, int up_ti, int down_ti,
                           const std::vector<float>& rin,
                           std::vector<float>& md, uint32_t M) {
  if (devices_.size() != 1) return false;
  if (M == 0) M = 1;
  for (int ti : {gate_ti, up_ti, down_ti}) {
    if (ti < 0 || (size_t)ti >= model_.tensors.size()) return false;
    if (!shard_cache_[(size_t)ti].built && !ensure_shard_built(ti))
      return false;
    if (shard_cache_[(size_t)ti].dpas48) return false;
  }
  const auto& tg = model_.tensors[(size_t)gate_ti];
  const auto& tu = model_.tensors[(size_t)up_ti];
  const auto& td = model_.tensors[(size_t)down_ti];
  const auto& scg = shard_cache_[(size_t)gate_ti];
  const auto& scu = shard_cache_[(size_t)up_ti];
  const auto& scd = shard_cache_[(size_t)down_ti];
  // Geometry: gate/up share rin (K) and inter (N); down maps inter -> H.
  // Range-split sanity: TP=1 so each shard covers the full N.
  const uint32_t inter = tg.N;
  if (tu.N != inter || tu.K != tg.K) return false;
  if (td.K != inter) return false;
  if (rin.size() != (size_t)M * tg.K) return false;
  if (shards_[gate_ti].ranges[0].first != 0 ||
      shards_[gate_ti].ranges[0].second != inter) return false;
  if (shards_[up_ti].ranges[0].first != 0 ||
      shards_[up_ti].ranges[0].second != inter) return false;
  if (shards_[down_ti].ranges[0].first != 0 ||
      shards_[down_ti].ranges[0].second != td.N) return false;
  if (md.size() != (size_t)M * td.N) md.resize((size_t)M * td.N);

  sycl::queue& q = devices_[0].queue;
  // Phase 1: convert rin once per item into the wave slots (same convert as
  // run_linear_wave phase 1; two slots, K_max = tg.K).
  ensure_wave_workspace(M, 2, tg.K, std::max(inter, td.N));
#ifdef INTFER_PROFILE_GPU
  auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
  const size_t need = (size_t)M * tg.K;
  if (act_staging_.size() != need) act_staging_.resize(need);
  // act_staging_ is SHARED: this wave phase rewrites it from rin, so any
  // INTFER_FUSE_ACT hit recorded before this point is now describing bytes that
  // are no longer in it. Bumping the generation is what makes the recorded cache
  // expire; without it the source-content check alone would still admit a hit
  // whose staged bytes belong to a different tensor.
  ++act_staging_gen_;
  dashq::float_to_half_batch(rin.data(), act_staging_.data(), need);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
    g_gpu_profile.copy_count++;
  }
  auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
  std::memcpy(wv_[0].act, act_staging_.data(), need * sizeof(std::uint16_t));
  std::memcpy(wv_[1].act, act_staging_.data(), need * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
    g_gpu_profile.usm_write_count += 2;
  }
#endif

  // Phase 2: submit gate + up back-to-back (in-order queue serializes;
  // explicit deps keep out-of-order queues correct too).
  const float ag = scg.has_global_divisor ? scg.global_host : 1.0f;
  const float au = scu.has_global_divisor ? scu.global_host : 1.0f;
  const float ad = scd.has_global_divisor ? scd.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
  auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
  sycl::event e_gate = gemm::quant_gemm_forward_usm(q, wv_[0].act,
      scg.code_usm[0], scg.scale_usm[0], ag, wv_[0].out,
      M, inter, tg.K, scg.group_size, scg.scheme, sycl::event{});
  // *** NO XMX DISPATCH HERE, DELIBERATELY (2026-10-14 coverage audit). ***
  // run_mlp_fused is TP=1-locked (devices_.size() != 1 => false, above), so
  // at the 27B --tp 2 target this whole path is dead code; wiring
  // INTFER_XMX_GEMM into it would create the "opt-in flag no gate can
  // reach" class the flag census exists to kill. The TP=2 MLP XMX dispatch
  // lives in run_mlp_gateup (per-output, with the compact/global readback
  // branch); if this seam is ever promoted to TP=2, port that dispatch
  // contract with it, including the per-arm layout rule.
  sycl::event e_up = gemm::quant_gemm_forward_usm(q, wv_[1].act,
      scu.code_usm[0], scu.scale_usm[0], au, wv_[1].out,
      M, inter, tu.K, scu.group_size, scu.scheme, e_gate);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
    g_gpu_profile.submit_count += 2;
  }
#endif

  // Phase 3: fused silu-mul + convert into the shared down-act workspace.
  ensure_workspace(M, td.K, td.N);
  gemm::SiluMulParams sp;
  sp.mg_fp16 = wv_[0].out;
  sp.mu_fp16 = wv_[1].out;
  sp.act_fp16 = ws_[0].act;
  sp.n = M * inter;
  sycl::event e_fused = gemm::silumul_to_half(q, sp, e_up);

  // Phase 4: down GEMM straight from the device act buffer, then the
  // single host wait + D2H of this whole chain.
  sycl::event e_down = gemm::quant_gemm_forward_usm(q, ws_[0].act,
      scd.code_usm[0], scd.scale_usm[0], ad, ws_[0].out,
      M, td.N, td.K, scd.group_size, scd.scheme, e_fused);
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  e_down.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += 1;
  for (sycl::event* e : {&e_gate, &e_up, &e_fused, &e_down}) {
    try {
      auto t0 = e->get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e->get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  const std::uint16_t* out_ptr = ws_[0].out;
  for (uint32_t m = 0; m < M; ++m)
    dashq::half_to_float_batch(out_ptr + (size_t)m * td.N,
                               md.data() + (size_t)m * td.N, td.N);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
#endif
  return true;
}

// Fused gate+up GEMM, TP=2 (see engine.hpp). Gate and up are both
// column-parallel over the SAME split and device-local, so one parallel_for
// over Ng+Nu outputs is bit-exact vs two separate quant_gemm_forward_usm
// calls: each output keeps its own accumulator, ascending kg, ascending j,
// its own group scale and wtab. Saves one host USM activation write and
// one q.submit per device per token vs the wave path. It does NOT save
// the event.wait (the wave already waits once) or the D2H readback: the
// kernel writes gate and up into TWO separate out buffers (wv_[d*wv_W_+1]
// and +2), so the host still does two half_to_float_batch loops. TP=2
// cannot fuse the readback the way run_mlp_fused (TP=1) does -- it writes
// into ONE ws_[0].out through the silu-mul->down device chain -- because
// gate/up are column-parallel and device d owns only its slice of Ng/Nu.
// M>1 IS supported: the kernel's outer m loop iterates the rows, so the
// same launch covers prefill and decode.
//
// NK ACCOUNTING, and it is the whole point of the counter. The gateup
// kernel is ONE launch per device, so on TP=2 a fused token contributes
// nk=2. The wave fallback the MLP callers actually use is
// run_linear_wave(items, 2) -- gate and up as two wave items, each
// submitted on both devices -- which is FOUR launches, so the fusion
// halves nk (4 -> 2) exactly as it halves submit_count and
// usm_write_count. It is NOT flat: the a119227 revision wrote "nk is
// flat and MUST stay flat" because it compared this path against
// run_mlp_fused (TP=1, two quant_gemm_forward_usm calls on device 0),
// which is not the TP=2 fallback. Both paths count launches through the
// SAME post-wait get_profiling_info loop, which makes the two counts
// comparable -- it does not make them equal, and asserting flatness
// here would have hidden a real delta behind a false one.
//
// An earlier revision also called gemm::profile_note_launch() in the
// submit loop, which double-counted the same two launches (nk=4 for the
// fused path against a wave path that also reported 4, so the delta was
// 0/0 and "the fusion saves a submit" was unfalsifiable). Removed; see
// the submit block for the reason. profile_note_launch exists for the
// free-function SSM/norm kernels (silumul_to_half, fwht_rot_to_half, ...)
// which have no post-wait profiling loop in their caller; the gateup
// kernel has one here, so it must not also take the launch-note path.
//
// The same inversion class, the other way round: submit_count and
// usm_write_count were ONLY ever incremented by THIS path. run_linear_tp,
// run_mlp_fused and run_linear_wave all accumulated the matching
// total_*_us microsecond sums but never bumped the count -- so a fused
// token reported submit_count=2 / usm_write_count=2 against a wave token
// that reported 0, i.e. the "halved" claim was a numerator against a zero
// denominator, unfalsifiable in exactly the way the nk double-count was.
// Every launch site and every memcpy site in those three paths now bumps
// its count alongside its microsecond sum (run_mlp_fused counts 2 for
// its two memcpys and 2 for its two submits; the wave's two memcpy arms
// count devices_.size() each, since they write one slot per device), so
// every delta has a real denominator. INTFER_PROFILE_GPU is ON in this
// build (CMakeCache), so the counters are live.
bool Engine::run_mlp_gateup(int gate_ti, int up_ti,
                            const std::vector<float>& rin,
                            std::vector<float>& mg, std::vector<float>& mu,
                            uint32_t M, bool readback,
                            std::vector<sycl::event>* chain_out) {
  // *** readback=false + chain_out: run_mlp_partial_down reuses this launch
  // loop to put gate/up outputs ON DEVICE (wv_ slots 1/2) and chains the silu
  // mul + partial down off the returned events, so the inter never reaches the
  // host. The host readback/wait/profiling/announcement blocks below are gated
  // by readback; the launch-loop submit/usm_write accounting is NOT (it must
  // stay true in both modes). chain_out, if set, absorbs the unsignaled events
  // for the caller to chain; the wait loop is skipped so the caller owns the
  // single wait for the whole device chain.
  if (devices_.size() != 2) return false;
  if (M == 0) M = 1;
  for (int ti : {gate_ti, up_ti}) {
    if (ti < 0 || (size_t)ti >= model_.tensors.size()) return false;
    if (!shard_cache_[(size_t)ti].built && !ensure_shard_built(ti))
      return false;
    if (shard_cache_[(size_t)ti].dpas48) return false;
  }
  const auto& tg = model_.tensors[(size_t)gate_ti];
  const auto& tu = model_.tensors[(size_t)up_ti];
  if (tu.K != tg.K) return false;
  if (tg.qcfg.group_size != tu.qcfg.group_size) return false;
  if (tg.qcfg.scheme != tu.qcfg.scheme) return false;
  if (rin.size() != (size_t)M * tg.K) return false;
  // M>1 IS supported: the kernel's outer m loop iterates the rows, so the
  // same launch covers prefill and decode. (The M=1-only version of this
  // kernel returned a NULL event for M>1 and this guard sent the prefill
  // arms back to run_linear_wave_batch; that is gone now.)
  // *** SCHEME TWIN OF THE M>1 DECLINE, FOUND 2026-10-10 ON A LIVE 27B RUN.
  // The fused kernel quant_gemm_forward_usm_gateup has a body ONLY for
  // INT2+INT2 (quant_gemm.cpp:2036); every other same-scheme pair falls
  // through to `return sycl::event();` -- a NULL event. A same-scheme
  // INT4+INT4 pair passes every guard above (same K, same group, same
  // scheme) and then runs NO kernel, while this function returns true and
  // the caller reads mg/mu back out of unwritten wave slots -- the exact
  // "memcpy out of an unwritten buffer" failure e497325 guarded against for
  // M>1, reached through the scheme dimension instead. On the 27B that is
  // 18 of 64 layers (L3, L6, L7, L10, ...); the operator's run died at
  // L3 with UR_RESULT_ERROR_OUT_OF_RESOURCES inside a submit whose kernel
  // body never existed. The guard must mirror the kernel's body condition,
  // not the weaker "schemes match" one.
  if (tg.qcfg.scheme != QuantScheme::INT2) return false;
  const auto& scg = shard_cache_[(size_t)gate_ti];
  const auto& scu = shard_cache_[(size_t)up_ti];
  if (scg.code_usm.size() != 2 || scu.code_usm.size() != 2) return false;
  for (size_t d = 0; d < 2; ++d) {
    if (!scg.code_usm[d] || !scg.scale_usm[d] ||
        !scu.code_usm[d] || !scu.scale_usm[d]) return false;
  }
  const uint32_t Ng = tg.N, Nu = tu.N;
  if (mg.size() != (size_t)M * Ng) mg.resize((size_t)M * Ng);
  if (mu.size() != (size_t)M * Nu) mu.resize((size_t)M * Nu);

  // Shared FP32->FP16 staging (gate and up read the SAME rin).
  const size_t need = (size_t)M * tg.K;
  if (act_staging_.size() != need) act_staging_.resize(need);
  ++act_staging_gen_;
  dashq::float_to_half_batch(rin.data(), act_staging_.data(), need);

  // Per-device: one act buffer shared by both outputs, two out buffers
  // (gate's slice, up's slice). The wave workspace gives us one slot per
  // device; reuse slot 0 for the shared act and slot 1 for the gate out,
  // slot 2 for the up out. Grow the wave workspace to W=3 if needed.
  // Transfer accounting (counted, not measured -- see the header): vs the
  // 2-item wave this saves one host USM activation write and one q.submit
  // per device, and it saves one launch per device on nk too (the wave
  // submits gate and up as two separate items, each on both devices).
  // The one remaining usm_write per device (same act_staging_ bytes to both
  // devices' slot 0) is NOT collapsible (2026-10-14 fusion test): each
  // device's slot is distinct USM and there is no P2P on this box, so the
  // bytes must reach each slot by a separate host write. Round trips
  // removed 0, launch delta 0. Recorded, not attempted.
  // The event.wait is already one per wave and the D2H readback is NOT
  // saved (two separate out buffers, see header).
  ensure_wave_workspace(M, 3, tg.K, std::max(Ng, Nu));
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(2);
  // Per-device XMX disposition, consumed by the readback loop below (the
  // launch loop's own locals are out of scope by then):
  //   xmx_ran_g[d]  -- gate's slice ran quant_gemm_forward_usm_xmx on
  //                      device d => the gate slot is COMPACT (M x Nl_g).
  //   up_compact[d] -- up's slot was written by a compact-layout kernel
  //                      on device d (XMX, or the plain tiled fallback).
  // Both false == the scalar gateup kernel wrote BOTH slots GLOBAL-SCATTER.
  bool xmx_ran_g[2] = {false, false};
  bool up_compact[2] = {false, false};
  // Per-device work-stealing counters for the XMX kernel -- the same
  // pattern as run_linear_tp (engine.cpp:1445) and run_linear_wave
  // (engine.cpp:2585), but TWO SLOTS PER DEVICE, not one: gate and up
  // are separate XMX launches on the same device and the kernel reads its
  // counter DURING execution (quant_gemm.cpp:2357-2360), so a single
  // slot would leave up's claims starting from gate's final count --
  // cur >= n_tiles breaks the claim loop immediately and up writes
  // nothing. Each slot is host-zeroed immediately before ITS OWN launch,
  // and this function waits for every event before returning (the loop
  // below), so no launch ever reads a slot being zeroed. A null or
  // invalid tile_ctr with INTFER_XMX_TILE_SCHED=ON is a device fault,
  // not a decline: the kernel dereferences it unconditionally in
  // work-stealing mode.
  static std::vector<std::uint32_t> xmx_tile_ctr;
  if (std::getenv("INTFER_XMX_GEMM") &&
      xmx_tile_ctr.size() < 2 * devices_.size())
    xmx_tile_ctr.resize(2 * devices_.size(), 0);
  for (size_t d = 0; d < 2; ++d) {
    auto& dev = devices_[d];
    sycl::queue& q = dev.queue;
    std::uint16_t* act_ptr = wv_[d * wv_W_ + 0].act;
#ifdef INTFER_PROFILE_GPU
    auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
    std::memcpy(act_ptr, act_staging_.data(), need * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_usm_write_us +=
          std::chrono::duration<double, std::micro>(t1 - t_w0).count();
      g_gpu_profile.usm_write_count += 1;
    }
#endif
    const float ag = scg.has_global_divisor ? scg.global_host : 1.0f;
    const float au = scu.has_global_divisor ? scu.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
    auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
    // *** INTFER_XMX_GEMM ON THE GATE+UP PATH (2026-10-14). *** The gate+up
    // launch is the largest INT2 GEMM population in prefill that the XMX
    // joint_matrix kernel had never reached: run_linear_tp and (since the
    // wave wiring) run_linear_wave both try XMX first, but this path
    // submitted the scalar INT2 kernel unconditionally. The 27B shape is
    // XMX-eligible on both ranks: chunk_M = 128 (%16==0), Nl = Ng/2 = 4352
    // (%16==0), K = 5120 (%32==0). Each output keeps its own accumulator and
    // its own launch, so this is the SAME try-XMX-then-fallback contract the
    // other two paths use -- XMX first per output, tiled scalar kernel as the
    // mandatory fallback when the kernel declines (M<8, non-tile residual,
    // FP16/INT8 scheme, or any % condition). A decline submits NOTHING, so
    // the fallback is not cosmetic: pushing a null event would leave the
    // wave slot unwritten and the caller would read garbage into mg/mu --
    // the same unwritten-slot class the scheme guard at 1830 refuses.
    //
    // *** THE SLOT LAYOUT IS PER-ARM, AND THE READBACK MUST FOLLOW IT. ***
    // Three writers can touch these two slots, and they use two different
    // layouts (derived from the kernel bodies, not remembered):
    //   * quant_gemm_forward_usm_gateup writes GLOBAL-SCATTER,
    //     outp[m * (is_up ? Nu : Ng) + (start + local)]
    //     (quant_gemm.cpp:2506, f0192ea);
    //   * quant_gemm_forward_usm_xmx writes COMPACT, out[m*N + n]
    //     (quant_gemm.cpp:2304, joint_matrix_store stride N); and
    //   * quant_gemm_forward_usm (the up fallback) writes COMPACT,
    //     out_usm[m*N + n] -- the layout the wave path reads back at
    //     engine.cpp:2841 (out_ptr + m*N_local, no offset).
    // So whenever ANY XMX dispatch ran on this device, both slots are
    // compact; the gateup fallback is the only global-scatter writer.
    // xmx_ran_g[d] / up_compact[d] (declared above the loop) record
    // which, and the readback loops branch on them.
    //
    // *** THREE XMX ANNOUNCEMENT SITES, THREE FUNCTION STATICS. *** The
    // [xmx-gemm] line is printed from run_linear_tp (engine.cpp:1446),
    // from here, and from run_linear_wave (engine.cpp:2682, named
    // xmx_wave_announced). Each is a function-local static, so the three
    // never collide -- but each fires at most once per process, IN CALL
    // ORDER. A run that reaches run_linear_tp's XMX dispatch first prints
    // the tp-flavoured line; the gateup-flavoured line then still fires
    // here (its own static is unset), so a gate grepping EITHER flavour
    // still sees its path. Do NOT merge them into one shared static: the
    // first path to run would silence the other two and a gate asserting
    // a silenced flavour would go red for an ordering reason.
    static bool xmx_announced = false;
    const uint32_t Nl_g = static_cast<uint32_t>(
        shards_[gate_ti].ranges[d].second - shards_[gate_ti].ranges[d].first);
    const uint32_t Nl_u = static_cast<uint32_t>(
        shards_[up_ti].ranges[d].second - shards_[up_ti].ranges[d].first);
    bool xmx_ran_u = false;
    sycl::event e;
    if (std::getenv("INTFER_XMX_GEMM")) {
      xmx_tile_ctr[2 * d] = 0;
      e = gemm::quant_gemm_forward_usm_xmx(
          q, act_ptr,
          scg.code_usm[d], scg.scale_usm[d], ag,
          wv_[d * wv_W_ + 1].out,
          M, Nl_g, tg.K, scg.group_size, scg.scheme,
          sycl::event{}, &xmx_ran_g[d], xmx_tile_ctr.data() + 2 * d);
      // Count the gate dispatch BEFORE the if/else below: its verdict is
      // already valid, and the note must fire on BOTH branches (the decline
      // branch and the XMX-ran branch are one attempt, not two). Placing it
      // after the fallback `}` would split the `}` / `else` chain.
      xmx_note_dispatch(xmx_ran_g[d]);
      if (!xmx_ran_g[d]) {
        e = gemm::quant_gemm_forward_usm_gateup(
            q, act_ptr,
            scg.code_usm[d], scg.scale_usm[d], ag,
            wv_[d * wv_W_ + 1].out,
            scu.code_usm[d], scu.scale_usm[d], au,
            wv_[d * wv_W_ + 2].out,
            M, Ng, Nu, tg.K, scg.group_size,
            (uint32_t)shards_[gate_ti].ranges[d].first, Nl_g,
            (uint32_t)shards_[up_ti].ranges[d].first, Nl_u,
            scg.scheme, scu.scheme,
            sycl::event{});
      } else {
        // Gate's slice went through XMX (compact slot). Up still tries
        // its own XMX; on a decline it falls back to the plain
        // single-output tiled kernel, which writes the SAME compact
        // layout -- so up's slot is compact on this device EITHER WAY,
        // and up_compact records it. The scalar gateup kernel is NOT
        // re-run here: it writes BOTH outputs and would double-write
        // gate's slice.
        up_compact[d] = true;
        xmx_tile_ctr[2 * d + 1] = 0;
        sycl::event e_u = gemm::quant_gemm_forward_usm_xmx(
            q, act_ptr,
            scu.code_usm[d], scu.scale_usm[d], au,
            wv_[d * wv_W_ + 2].out,
            M, Nl_u, tu.K, scu.group_size, scu.scheme,
            sycl::event{}, &xmx_ran_u, xmx_tile_ctr.data() + 2 * d + 1);
        if (!xmx_ran_u) {
          e_u = gemm::quant_gemm_forward_usm(
              q, act_ptr, scu.code_usm[d], scu.scale_usm[d], au,
              wv_[d * wv_W_ + 2].out,
              M, Nl_u, tu.K, scu.group_size, scu.scheme,
              sycl::event{});
        }
        xmx_note_dispatch(xmx_ran_u);
        // Push up's event instead of waiting it inline: the shared wait
        // loop and the post-wait profiling loop below then cover it, so
        // the kernel/submit/wait counters stay true in the mixed arm (an
        // inline wait would be an UNCOUNTED host wait and up's kernel
        // time would vanish from the kernel bucket).
        kernel_events.push_back(e_u);
      }
      if ((xmx_ran_g[d] || xmx_ran_u) && !xmx_announced) {
        xmx_announced = true;
        const bool ts = (std::getenv("INTFER_XMX_TILE_SCHED") != nullptr);
        std::fprintf(stderr,
                     "[xmx-gemm] ACTIVE on the fused gate+up launch "
                     "(INTFER_XMX_GEMM=1, M=%u Nl_g=%u Nl_u=%u K=%u)\n"
                     "[xmx-gateup-tiles] gate_tiles=%u up_tiles=%u "
                     "(%s work-stealing)\n",
                     M, Nl_g, Nl_u, tg.K,
                     (M / 16) * (Nl_g / 16), (M / 16) * (Nl_u / 16),
                     ts ? "persistent" : "fixed-assignment");
        std::fflush(stderr);
      }
    } else {
      e = gemm::quant_gemm_forward_usm_gateup(
          q, act_ptr,
          scg.code_usm[d], scg.scale_usm[d], ag,
          wv_[d * wv_W_ + 1].out,
          scu.code_usm[d], scu.scale_usm[d], au,
          wv_[d * wv_W_ + 2].out,
          M, Ng, Nu, tg.K, scg.group_size,
          (uint32_t)shards_[gate_ti].ranges[d].first, Nl_g,
          (uint32_t)shards_[up_ti].ranges[d].first, Nl_u,
          scg.scheme, scu.scheme,
          sycl::event{});
    }
    kernel_events.push_back(e);
    // nk accounting: ONE increment per launch, and it comes from the SAME
    // mechanism every other engine path uses -- the post-wait
    // get_profiling_info loop below (lines 1700-1707), which reads the
    // device's own event timestamps. An earlier version also called
    // gemm::profile_note_launch() here, which bumped kernel_count a SECOND
    // time for the same two launches: on TP=2 the fused token reported nk=4
    // where the wave path reports nk=2, so "the fusion saves a submit" was
    // comparing a double-counted numerator against a single-counted one and
    // was unfalsifiable. profile_note_launch exists for the free-function
    // SSM/norm kernels (silumul_to_half, fwht_rot_to_half, ...) which have
    // no post-wait profiling loop in their caller; the gateup kernel has one
    // here, so it must not also take the launch-note path.
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_submit_us +=
          std::chrono::duration<double, std::micro>(t1 - t_s0).count();
      // submit_count is the number of q.submit CALLS. In the mixed XMX arm
      // up took its own launch on this device (its event was pushed into
      // kernel_events, not waited inline), so that device contributes 2;
      // every other arm submits exactly one per device. This matches what
      // the post-wait profiling loop counts into kernel_count below.
      g_gpu_profile.submit_count += (up_compact[d] ? 2 : 1);
    }
#endif
  }
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  // readback=false (run_mlp_partial_down): the caller chains silu-mul + the
  // partial-down GEMM off these SAME events and owns the single wait for the
  // whole device chain, so this path does NOT wait here. It absorbs the
  // unsignaled events into chain_out for the caller to chain; the caller's
  // own post-wait profiling loop counts them into nk, so the device launches
  // are described identically in both modes -- only the host wait/readback
  // moves.
  if (!readback) {
    if (chain_out) *chain_out = std::move(kernel_events);
    return true;
  }
  for (auto& e : kernel_events) e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us +=
        std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
    // wait_count is the number of HOST .wait() CALLS. This path loops over
    // kernel_events (2, one per device -- 4 in the mixed XMX arm, where up
    // takes its own launch on each device) and calls e.wait() on each, so
    // it contributes 2 (4 in that arm). The wave fallback loops over 4
    // events (2 items x 2 devices) and contributes 4. So the fusion DOES
    // halve wait_count (4 -> 2), exactly like nk, submit_count and
    // usm_write_count -- it is not flat and must not be written off as
    // "already one per wave".
    // The one thing it does NOT do is cut the number of HOST WAITS below
    // one per device: gate and up go into two separate out buffers, so the
    // readback cannot be fused the way run_mlp_fused (TP=1) does, and the
    // host still has to wait for each device's launch to finish before
    // reading. Counting it per event keeps this counter consistent with
    // every other engine path; dropping it would make a fused token report
    // wait_count=0 and read as "the fusion removed every wait", which is
    // not true.
    g_gpu_profile.wait_count += (int)kernel_events.size();
  }
#endif
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  // *** THE READBACK MUST MATCH THE KERNEL'S WRITE LAYOUT, AND THERE ARE
  // TWO OF THEM (both derived from the kernel bodies, not remembered). ***
  // The scalar gateup kernel writes GLOBAL-SCATTER:
  // outp[m * (is_up ? Nu : Ng) + (start + local)] (quant_gemm.cpp:2506,
  // f0192ea). The slot is sized M x max(Ng, Nu) (ensure_wave_workspace
  // takes Nl = max(Ng, Nu)), so the row stride IS Ng/Nu and the shard
  // occupies COLUMNS [start, start+Nl) of every row; the source index is
  // m * Ng + start + local. A compact source (m * Nl_g) would read columns
  // [0, Nl_g) -- columns the kernel never wrote at TP=2 -- and the two
  // layouts agree only when start == 0 && Nl == Ng, i.e. TP=1. That defect
  // was invisible to the gateup arm (and to examples/gateup_equiv, which
  // compares the kernel's output against the reference directly): the fused
  // path IS the default TP=2 path, so base and fused produce the SAME bytes
  // and the stream-identity predicate cannot see a wrong-but-consistent
  // readback.
  //
  // The XMX arm and the plain tiled up fallback write COMPACT:
  // out[m * Nl + n] -- the layout the wave path reads back at
  // engine.cpp:2841 (out_ptr + m*N_local, no offset). Each device's
  // readback therefore branches on the disposition flags recorded in the
  // launch loop (xmx_ran_g[d] for gate, up_compact[d] for up). On this
  // architecture Ng == Nu == inter (gate and up both project hidden ->
  // inter), so the global-strided rows never collide.
  // *** THE TWO READBACK LOOPS ARE NOT FUSIBLE (2026-10-14 fusion test). ***
  // They read two DIFFERENT device slots (wv slot 1 = gate out, slot 2 =
  // up out) into two DIFFERENT caller buffers (mg, mu). One "fused" pass
  // would still iterate both (slot, buffer) pairs -- same D2H transfers,
  // same bytes, launch delta 0, round trips removed 0. The readback is
  // host-side; there is no launch to fuse and no producer/consumer pair
  // to collapse. readback_count stays 2 per device (4 total), which matches
  // the wave fallback's granularity and keeps the counters comparable.
  // Collapsing to ONE out slot would change the kernel's write layout (the
  // f0192ea shard contract), the XMX compact precedent and the wave slot
  // sharing -- a layout migration with zero launch delta, paid only in the
  // load-sensitive host buckets. Recorded, not attempted.
  // *** readback=false skips this block ENTIRELY: run_mlp_partial_down keeps
  // gate/up outputs on device (wv_ slots 1/2) and fuses silu-mul + the partial
  // down off the absorbed events, so there is no mg/mu to fill and no D2H.
  // The launch-loop submit/usm_write accounting above is NOT gated by
  // readback -- it stays true in both modes, so the counters describe the
  // device work the same way whether the host reads the results or not.
  if (readback) {
  for (size_t d = 0; d < 2; ++d) {
    auto [start, end] = shards_[gate_ti].ranges[d];
    const uint32_t Nl_g = static_cast<uint32_t>(end - start);
    const std::uint16_t* g_out = wv_[d * wv_W_ + 1].out;
#ifdef INTFER_PROFILE_GPU
    auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
    // Source layout follows which kernel ran on THIS device: global-scatter
    // (row stride Ng, column start+local) for the gateup kernel, compact
    // (row stride Nl_g, column 0) for XMX. The DESTINATION is the same
    // either way (the caller's mg row, columns [start, start+Nl_g)).
    for (uint32_t m = 0; m < M; ++m)
      dashq::half_to_float_batch(
          g_out + (xmx_ran_g[d] ? (size_t)m * Nl_g
                                : (size_t)m * Ng + start),
          mg.data() + (size_t)m * tg.N + start,
          Nl_g);
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_readback_us +=
          std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
      g_gpu_profile.readback_count += 1;
    }
#endif
    auto [s2, e2] = shards_[up_ti].ranges[d];
    const uint32_t Nl_u = static_cast<uint32_t>(e2 - s2);
    const std::uint16_t* u_out = wv_[d * wv_W_ + 2].out;
#ifdef INTFER_PROFILE_GPU
    t_rb0 = std::chrono::high_resolution_clock::now();
#endif
    // Same branch for up: compact (up_compact[d]) when an XMX-dispatch
    // kernel wrote the slot, global-scatter (row stride Nu, column s2)
    // when the gateup kernel wrote both.
    for (uint32_t m = 0; m < M; ++m)
      dashq::half_to_float_batch(
          u_out + (up_compact[d] ? (size_t)m * Nl_u
                                 : (size_t)m * Nu + s2),
          mu.data() + (size_t)m * tu.N + s2,
          Nl_u);
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_readback_us +=
          std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
      g_gpu_profile.readback_count += 1;
    }
#endif
  }
  }  // if (readback)
  // THE PATH ANNOUNCES ITSELF, ONCE. Same shape as the decode1 arms and the
  // L4 la_out seam: without it, a gate that compares this run's token stream
  // against the base run's cannot tell whether the fused path ran -- the two
  // streams are bit-identical by construction, so a vacuous arm (the wave
  // fallback, which produces the same bytes) passes every stream check and
  // the arm certifies nothing. This one line is what turns "the runs
  // matched" into "the fusion ran". It is host-side, one per process, so it
  // does not flood a 794-launches-per-token log.
  //
  // This point is reached ONLY for INT2+INT2 pairs (the scheme guard above
  // returns false for everything else), so the line cannot be printed by a
  // layer that ran no kernel. The operator's 2026-10-10 run printed it with
  // scheme=2 (INT4) -- that was the bug, not the announcement: the guard let
  // a NULL-event layer through and the fprintf still fired. The line is now
  // a true statement of "a real fused launch ran".
  static bool announced_gu = false;
  if (!announced_gu) {
    announced_gu = true;
    std::fprintf(stderr,
                 "[mlp-gateup] fused gate+up GEMM ACTIVE "
                 "(Ng=%u Nu=%u K=%u group=%u scheme=%d, one launch for both)\n",
                 Ng, Nu, tg.K, scg.group_size, (int)scg.scheme);
  }
  return true;
}

// Device-act GEMM seam (see engine.hpp): same FP16-act kernel as
// run_linear_tp but the M*K activation bits come from a caller-owned device
// buffer. Skips host FP32->FP16 convert + H2D; the dep chains after the
// producer kernel so a device-side chain pays one host wait total.
bool Engine::run_linear_tp_devact(int tensor_index,
                                  const std::uint16_t* dev_act,
                                  std::vector<float>& out, uint32_t M,
                                  const sycl::event& dep) {
  if (devices_.size() != 1) return false;
  if (!dev_act || M == 0) return false;
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return false;
  // *** NO XMX DISPATCH HERE, DELIBERATELY (2026-10-14 coverage audit). ***
  // TP=1-locked (devices_.size() != 1 => false, above) — dead code at the
  // 27B --tp 2 target, so an INTFER_XMX_GEMM dispatch here would be
  // unreachable from any gate. See run_mlp_fused's note for the same
  // decision at the other TP=1 seam.
  const auto& t = model_.tensors[(size_t)tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (ranges.size() != 1) return false;
  if (ranges[0].first != 0) return false;
  if (t.N == 0 || t.K == 0) return false;
  // No mutation on failure: the caller reuses its output buffer on the
  // host fallback path, so a shape mismatch must not resize it.
  if (out.size() != (size_t)M * t.N) return false;
  if (!shard_cache_[(size_t)tensor_index].built &&
      !ensure_shard_built(tensor_index))
    return false;
  const auto& sc = shard_cache_[(size_t)tensor_index];
  if (sc.dpas48) return false;  // FP16-act path only; host path covers DPAS
  if (sc.code_usm.empty() || !sc.code_usm[0]) return false;
  if (sc.scale_usm.empty() || !sc.scale_usm[0]) return false;
  {
    uint32_t Nl = ranges[0].second - ranges[0].first;
    if (Nl != t.N) return false;  // TP=1 full-row sanity
    ensure_workspace(M, t.K, Nl);
  }
  // Output staging reuses ws_[0].out: reject a dev_act that aliases it
  // (a caller chaining through the shared workspace would corrupt its own
  // input; dedicated buffers — the L4 pattern — never hit this). Checked
  // after ensure_workspace so the pointer is final.
  if (ws_[0].out && dev_act >= ws_[0].out &&
      dev_act < ws_[0].out + (size_t)ws_[0].M * ws_[0].Nl)
    return false;
  sycl::queue& q = devices_[0].queue;
  float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
  auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
  sycl::event e = gemm::quant_gemm_forward_usm(q,
      const_cast<std::uint16_t*>(dev_act),
      sc.code_usm[0], sc.scale_usm[0], alpha, ws_[0].out,
      M, t.N, t.K, sc.group_size, sc.scheme, dep);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
  }
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += 1;
  try {
    auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
    g_gpu_profile.kernel_count++;
  } catch (...) {}
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  const std::uint16_t* out_ptr = ws_[0].out;
  for (uint32_t m = 0; m < M; ++m)
    dashq::half_to_float_batch(out_ptr + (size_t)m * t.N,
                               out.data() + (size_t)m * t.N, t.N);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
#endif
  return true;
}

// TP=2 twin of run_linear_tp_devact (INTFER_LA_OUT_TP2). The activation is
// a PER-DEVICE array because the SSM producer lives on device 0 and there is
// no P2P on this box: device 0 hands the engine the norm kernel's own
// device buffer, device 1 gets its own copy (the caller stages it -- see
// run_mlp_gateup's documented "bytes must reach each slot by a separate
// host write" rule at the gateup launch loop). Each rank then runs its own
// N-slice with the SAME kernel run_linear_tp uses, so the arithmetic is
// unchanged and the readback assembly is the one run_linear_tp already does.
// Everything else (XMX try-first + mandatory fallback, the once-per-process
// announcement, the shared wait, the profiling buckets) mirrors it so the
// counters stay comparable rather than a second accounting path.
bool Engine::run_linear_tp_devact_tp2(int tensor_index,
                                      const std::uint16_t* const* dev_act,
                                      std::vector<float>& out, uint32_t M,
                                      const sycl::event& dep) {
  if (devices_.size() != 2) return false;  // TP=2 only
  if (!dev_act || M == 0) return false;
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return false;
  const auto& t = model_.tensors[(size_t)tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (ranges.size() != 2) return false;  // must be a 2-rank column split
  if (ranges[0].first != 0 || ranges[1].second != t.N) return false;
  for (size_t d = 0; d < 2; ++d)
    if (ranges[d].second <= ranges[d].first) return false;
  if (t.N == 0 || t.K == 0) return false;
  // No mutation on failure: the caller reuses its output buffer on the host
  // fallback path, so a shape mismatch must not resize it.
  if (out.size() != (size_t)M * t.N) return false;
  if (!shard_cache_[(size_t)tensor_index].built &&
      !ensure_shard_built(tensor_index))
    return false;
  const auto& sc = shard_cache_[(size_t)tensor_index];
  if (sc.dpas48) return false;  // FP16-act path only; host path covers DPAS
  if (sc.code_usm.size() != 2) return false;
  for (size_t d = 0; d < 2; ++d) {
    if (!sc.code_usm[d] || !sc.scale_usm[d]) return false;
    if (!dev_act[d]) return false;
  }
  // Shared workspaces grow-only across ranks; Nl_max is over TP ranks.
  {
    uint32_t Nl_max = std::max(ranges[0].second - ranges[0].first,
                               ranges[1].second - ranges[1].first);
    ensure_workspace(M, t.K, Nl_max);
  }
  // Output staging reuses ws_[d].out, so a dev_act aliasing it would corrupt
  // its own input -- checked after ensure_workspace so the pointer is final.
  for (size_t d = 0; d < 2; ++d) {
    if (ws_[d].out && dev_act[d] >= ws_[d].out &&
        dev_act[d] < ws_[d].out + (size_t)ws_[d].M * ws_[d].Nl)
      return false;
  }
  const float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
  auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(2);
  // Per-device work-stealing counters for the XMX kernel (same 2-slot rule
  // as run_mlp_gateup: the kernel reads its counter DURING execution, so one
  // slot per device is the minimum and each is host-zeroed before its own
  // launch).
  static std::vector<std::uint32_t> xmx_tile_ctr;
  if (std::getenv("INTFER_XMX_GEMM") &&
      xmx_tile_ctr.size() < 2 * devices_.size())
    xmx_tile_ctr.resize(2 * devices_.size(), 0);
  bool xmx_ran_any = false;
  for (size_t d = 0; d < 2; ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    uint32_t N_local = end - start;
    sycl::event kernel_evt;
    const char* xmx_env = std::getenv("INTFER_XMX_GEMM");
    if (xmx_env) {
      bool xmx_ran = false;
      xmx_tile_ctr[d] = 0;
      kernel_evt = gemm::quant_gemm_forward_usm_xmx(
          q, const_cast<std::uint16_t*>(dev_act[d]), sc.code_usm[d],
          sc.scale_usm[d], alpha, ws_[d].out, M, N_local, t.K,
          sc.group_size, sc.scheme, sycl::event{}, &xmx_ran,
          xmx_tile_ctr.data() + d);
      // A declined XMX attempt writes NOTHING, so the tiled fallback is
      // mandatory, not cosmetic (same contract as run_linear_tp:1505).
      if (!xmx_ran) {
        kernel_evt = gemm::quant_gemm_forward_usm(
            q, const_cast<std::uint16_t*>(dev_act[d]), sc.code_usm[d],
            sc.scale_usm[d], alpha, ws_[d].out, M, N_local, t.K,
            sc.group_size, sc.scheme, sycl::event{});
      } else {
        xmx_ran_any = true;
      }
      xmx_note_dispatch(xmx_ran);
    } else {
      kernel_evt = gemm::quant_gemm_forward_usm(
          q, const_cast<std::uint16_t*>(dev_act[d]), sc.code_usm[d],
          sc.scale_usm[d], alpha, ws_[d].out, M, N_local, t.K,
          sc.group_size, sc.scheme, sycl::event{});
      (void)dep;
    }
    kernel_events.push_back(kernel_evt);
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
    }
    if (kernel_events.size() > g_gpu_profile.peak_wait_depth)
      g_gpu_profile.peak_wait_depth = kernel_events.size();
#endif
  }
  // THE PATH ANNOUNCES ITSELF, ONCE, gated on xmx_ran_any rather than on the
  // env var: a line printed for a flag the kernel declined would be exactly
  // the vacuous-green shape the [xmx-gemm] arm in run_linear_tp gates
  // against. Its own function-local static keeps it from colliding with the
  // tp/gateup/wave flavours (the three-statics rule at run_mlp_gateup:2009).
  if (xmx_ran_any) {
    static bool announced_l4_tp2 = false;
    if (!announced_l4_tp2) {
      announced_l4_tp2 = true;
      std::fprintf(stderr,
                   "[xmx-gemm] ACTIVE on eligible INT2/INT4 tiles "
                   "(INTFER_XMX_GEMM=1)\n");
    }
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t_wb0 = std::chrono::high_resolution_clock::now();
    for (auto& e : kernel_events) e.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
    g_gpu_profile.wait_count += devices_.size();
  }
  // t_rb0 MUST live at this scope (not inside the wait block above): the
  // readback accumulator below, still inside its own INTFER_PROFILE_GPU
  // block, reads it -- the first revision declared it in the wait block's
  // scope and the engine TU failed to compile.
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1e = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1e - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#else
  for (auto& e : kernel_events) e.wait();
#endif
  // Read back FP16 -> FP32 per rank into out.data() + m*t.N + start. The
  // wait above is load-bearing, not profiling: without it the host reads
  // ws_[d].out before the kernel runs.
  for (size_t d = 0; d < 2; ++d) {
    auto [start, end] = ranges[d];
    const std::uint16_t* out_ptr = ws_[d].out;
    const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
    for (uint32_t m = 0; m < M; ++m)
      dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                 out.data() + (size_t)m * t.N + start,
                                 N_local_rb);
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count += (int)devices_.size();
  }
#endif
  return true;
}

// Static pre-flight for run_linear_tp_devact_tp2 (see the header). Every
// precondition the seam can check without touching device state: the tensor
// exists, is column-sharded across exactly 2 ranks, is FP16-act (not DPAS48),
// the shard is built, and the caller's output buffer is the right size. The
// caller MUST run this BEFORE submitting the SSM chain kernel, because the
// chain advances the recurrent state in place and a decline after that would
// corrupt it (the fusion would have to re-run the recurrence to produce
// c_ncore for the host path). Same "eligibility predicate consulted before
// the launch" idiom as run_linear_tp_dpas48:1270.
bool Engine::la_out_tp2_eligible(int tensor_index, uint32_t M,
                                 size_t out_elems) const {
  if (devices_.size() != 2) return false;
  if (M == 0) return false;
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return false;
  const auto& t = model_.tensors[(size_t)tensor_index];
  const auto& ranges = shards_[(size_t)tensor_index].ranges;
  if (ranges.size() != 2) return false;
  if (ranges[0].first != 0 || ranges[1].second != t.N) return false;
  for (size_t d = 0; d < 2; ++d)
    if (ranges[d].second <= ranges[d].first) return false;
  if (t.N == 0 || t.K == 0) return false;
  if (out_elems != (size_t)M * t.N) return false;
  const auto& sc = shard_cache_[(size_t)tensor_index];
  if (!sc.built) return false;
  if (sc.dpas48) return false;  // FP16-act path only; host path covers DPAS
  if (sc.code_usm.size() != 2) return false;
  for (size_t d = 0; d < 2; ++d)
    if (!sc.code_usm[d] || !sc.scale_usm[d]) return false;
  return true;
}

// Wave variant: one independent GEMM per item (see engine.hpp). Phases:
// (1) per item: lazy shard build + FP32->FP16 convert + per-device act
// memcpy into that item's wave slot; (2) per device: submit all items'
// kernels back-to-back; (3) wait once for the whole wave; (4) per item:
// read back. Bit-identical to sequential run_linear_tp calls: same shard
// bytes, same convert, same kernel args, same output assembly — only the
// submit/wait grouping changes, so the device runs the wave back-to-back
// and the host pays one round-trip per wave instead of per GEMM.
// DPAS48-flagged items fall back to sequential run_linear_tp (rare:
// DPAS is default-off; the wave's shared-slot layout targets the FP16-act
// path). Items with tensor_index < 0 are skipped.
// INTFER_WAVE_LAUNCH_FUSE implementation. One parallel_for per device over
// the union of the wave's output elements; each item's accumulator is
// untouched, so the launch is bit-exact vs nl separate calls. Returns true
// on success (the caller's Phase 3/4 wait+readback then apply to the whole
// wave); false means the caller must fall through to the default path.
bool Engine::run_linear_wave_fused(const LinearWaveItem** items, size_t nl,
                                   uint32_t M) {
  std::vector<const std::uint16_t*> act(nl);
  std::vector<const void*> code(nl);
  std::vector<const std::uint16_t*> scale(nl);
  std::vector<float> alpha(nl);
  std::vector<std::uint16_t*> outp(nl);
  std::vector<uint32_t> Mi(nl), Ni(nl), Ki(nl), Gi(nl);
  std::vector<gemm::WeightScheme> ws(nl);
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)items[i]->tensor_index];
    const auto& sc = shard_cache_[(size_t)items[i]->tensor_index];
    // Per-device pointers are filled in the device loop below; the first
    // pass seeds them with device 0 so the vectors are never empty.
    code[i] = sc.code_usm[0];
    scale[i] = sc.scale_usm[0];
    alpha[i] = sc.has_global_divisor ? sc.global_host : 1.0f;
    // *** Ni IS THE LOCAL COLUMN COUNT, NOT t.N. *** The kernel writes
    // outp[m * Ni_arr[i] + n] for n in [0, Ni) and its nd_range total is
    // sum(M_i * Ni_i), so Ni MUST be THIS DEVICE's column range, matching the
    // wv_[...].out buffer (engine.cpp:669 allocates M * Nl_max). Passing
    // t.N here made every fused wave at TP=2 compute m = local / t.N and
    // write M * t.N halfs into an M * (t.N / 2) buffer -- a 2x OOB write whose
    // data the Phase 4 readback (local N_local_rb stride) never reads. Seeded
    // per device below; this pass only fills the device-0 arrays so they are
    // never empty.
    Mi[i] = M; Ni[i] = t.N; Ki[i] = t.K; Gi[i] = sc.group_size;
    ws[i] = sc.scheme;
  }
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    sycl::queue& q = dev.queue;
    for (size_t i = 0; i < nl; ++i) {
      const auto& sc = shard_cache_[(size_t)items[i]->tensor_index];
      const auto& ranges = shards_[(size_t)items[i]->tensor_index].ranges;
      code[i] = sc.code_usm[d];
      scale[i] = sc.scale_usm[d];
      // Ni must be THIS DEVICE's column range: the kernel's linear -> (m, n)
      // decode and its output stride are both Ni, and the out buffer slot is
      // M * Nl_local. Using t.N (the sharded width) is a 2x OOB write at
      // TP=2 -- see the note in the setup loop above.
      Ni[i] = static_cast<uint32_t>(ranges[d].second - ranges[d].first);
      // act AND out are per (item, device): slot i on device d is
      // wv_[d*wv_W_ + i]. The default path below writes the same slots in
      // Phase 1/2 and reads them back in Phase 4, so the workspace is
      // shared and the indexing is identical.
      act[i] = wv_[d * wv_W_ + i].act;
      outp[i] = wv_[d * wv_W_ + i].out;
    }
#ifdef INTFER_PROFILE_GPU
    auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
    sycl::event e = gemm::quant_gemm_forward_usm_wave(
        q, act.data(), code.data(), scale.data(), alpha.data(),
        outp.data(), (uint32_t)nl, Mi.data(), Ni.data(), Ki.data(),
        Gi.data(), ws.data(), sycl::event{});
    kernel_events.push_back(e);
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_submit_us +=
          std::chrono::duration<double, std::micro>(t1 - t_s0).count();
      g_gpu_profile.submit_count += 1;
    }
#endif
  }
  // *** PHASE 3 + 4 ARE MINE, NOT THE CALLER'S. *** The caller does
  // `if (run_linear_wave_fused(...)) return;`, which skips the default
  // path's Phase 3 (wait) and Phase 4 (readback). If this function returned
  // without doing them, every wave taking the fused path would leave
  // live[i]->out stale and never drain the device work -- a silent wrong
  // output with no crash, and exactly the defect the default path's
  // Phase 3/4 comment ("host must not read wave outputs before the
  // kernels run") exists to prevent. The fused path writes the SAME slots
  // (wv_[d*wv_W_ + i].out) the default path reads back, so the wait and
  // the half_to_float_batch pass below are byte-for-byte the same code.
  // Phase 3: wait once for the whole wave.
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  for (auto& e : kernel_events) e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us +=
        std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += (int)kernel_events.size();
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  // Phase 4: per item — read back from that item's wave slot. Same slots and
  // the same `out.data() + m*t.N + start` gather the default path uses, so
  // the two paths produce byte-identical caller-visible output.
  uint32_t rb_items = 0;
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)items[i]->tensor_index];
    std::vector<float>& out = *items[i]->out;
    const auto& ranges = shards_[(size_t)items[i]->tensor_index].ranges;
    for (size_t d = 0; d < devices_.size(); ++d) {
      auto [start, end] = ranges[d];
      const std::uint16_t* out_ptr = wv_[d * wv_W_ + i].out;
      const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
      for (uint32_t m = 0; m < M; ++m)
        dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                   out.data() + (size_t)m * t.N + start,
                                   N_local_rb);
      ++rb_items;
    }
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us +=
        std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count += rb_items;
  }
#endif

  // THE PATH ANNOUNCES ITSELF, ONCE. Same shape as the [mlp-gateup] line:
  // without it a gate that compares this run's token stream against the
  // base run's cannot tell whether the fused path ran -- the two streams
  // are bit-identical by construction, so a flag that never reached the
  // kernel would pass every stream check while certifying nothing. The
  // line is printed here, AFTER the wait+readback, so it cannot fire for a
  // wave that fell through to the default path.
  static bool announced_wf = false;
  if (!announced_wf) {
    announced_wf = true;
    const auto& t0 = model_.tensors[(size_t)items[0]->tensor_index];
    const auto& sc0 = shard_cache_[(size_t)items[0]->tensor_index];
    // WHICH wave fused. The launch count alone cannot say it: the q/k/v
    // wave-3 and the z/b/a wave-4 both print items=3/4, so a gate reading
    // only `items=` would certify "the q/k/v wave fused" on a log where only
    // the z/b/a wave did. Classify from the tensors' own names -- the same
    // suffixes the caller groups on (main.cpp: "q/k/v are independent",
    // "z/b/a are independent", "gate/up share rin"). An item whose name the
    // caller did not carry (name==nullptr) drops out of the classification
    // and the tag falls back to "items", which is still a real distinction
    // from the default path (nl separate launches).
    bool has_q=false, has_k=false, has_v=false, has_z=false, has_b=false,
         has_a=false, has_gate=false, has_up=false, has_qkv=false;
    size_t named = 0;
    for (size_t i = 0; i < nl; ++i) {
      const std::string* nm = items[i]->name;
      if (!nm) continue;
      ++named;
      if (nm->find("in_proj_qkv") != std::string::npos) has_qkv = true;
      else {
        if (nm->find("q_proj") != std::string::npos) has_q = true;
        if (nm->find("k_proj") != std::string::npos) has_k = true;
        if (nm->find("v_proj") != std::string::npos) has_v = true;
      }
      // in_proj_z.weight / in_proj_b.weight / in_proj_a.weight carry the
      // letter right before ".weight", so "_z." / "_b." / "_a." matches.
      if (nm->find("in_proj_z") != std::string::npos ||
          nm->find("_z.") != std::string::npos) has_z = true;
      if (nm->find("in_proj_b") != std::string::npos ||
          nm->find("_b.") != std::string::npos) has_b = true;
      if (nm->find("in_proj_a") != std::string::npos ||
          nm->find("_a.") != std::string::npos) has_a = true;
      if (nm->find("gate_proj") != std::string::npos) has_gate = true;
      if (nm->find("up_proj") != std::string::npos) has_up = true;
    }
    const char* wave = "items";
    if (has_qkv && has_z && has_b && has_a) wave = "qkvzba";
    else if (has_q && has_k && has_v && has_z && has_b && has_a) wave = "qkvzba";
    else if (has_q && has_k && has_v) wave = "qkv";
    else if (has_gate && has_up) wave = "gateup";
    else if (has_z && has_b && has_a) wave = "zba";
    else if (named == nl) wave = "named";
    std::fprintf(stderr,
                 "[wave-launch-fuse] fused wave GEMM ACTIVE "
                 "(wave=%s items=%zu M=%u K=%u group=%u, one launch per device)\n",
                 wave, nl, M, t0.K, sc0.group_size);
  }
  return true;
}

void Engine::run_linear_wave(const LinearWaveItem* items, size_t n,
                             uint32_t M) {
  if (!items || n == 0) return;
  if (M == 0) M = 1;
  // Compact out skipped items (stable order = call order).
  const LinearWaveItem* live[16];
  size_t nl = 0;
  for (size_t i = 0; i < n && nl < 16; ++i)
    if (items[i].tensor_index >= 0 &&
        (size_t)items[i].tensor_index < model_.tensors.size() &&
        items[i].x && items[i].out)
      live[nl++] = &items[i];
  if (nl == 0) return;
  // DPAS48 fallback: keep the wave contract (same outputs) via the
  // sequential path. Bit-identical by construction.
  for (size_t i = 0; i < nl; ++i) {
    if (!shard_cache_[(size_t)live[i]->tensor_index].built)
      ensure_shard_built(live[i]->tensor_index);
    if (shard_cache_[(size_t)live[i]->tensor_index].dpas48) {
      for (size_t j = 0; j < nl; ++j)
        run_linear_tp(live[j]->tensor_index, *live[j]->x, *live[j]->out, M);
      return;
    }
  }

  // Phase 0: the wave workspace must cover every item's (M, K, Nl) shape
  // plus one slot per live item. Nl_max is over TP ranks AND items; K is
  // the max over items (K varies per tensor: H vs inter vs conv_dim...).
  // Sized BEFORE the fuse check: the fused path writes item i's output into
  // slot i on each device (wv_[d*wv_W_ + i].out) and the default path's
  // Phase 1/2 writes the SAME slots it reads back in Phase 4, so the
  // workspace must exist for either. Sizing it here, once, covers both.
  uint32_t Nl_max = 0, K_max = 0;
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
    K_max = std::max(K_max, t.K);
    const auto& ranges = shards_[live[i]->tensor_index].ranges;
    for (size_t d = 0; d < ranges.size(); ++d)
      Nl_max = std::max(Nl_max, ranges[d].second - ranges[d].first);
  }
  ensure_wave_workspace(M, (uint32_t)nl, K_max, Nl_max);

  // INTFER_WAVE_LAUNCH_FUSE: one parallel_for over the union of the wave's
  // output elements, instead of nl separate launches per device. Bit-exact
  // by construction (each item's accumulator is untouched), so the default
  // path below is byte-for-byte the old one when the flag is unset. Decline
  // on anything the fused kernel does not support -- a partial fuse is worse
  // than none, because the caller's Phase 3/4 (one wait, one readback pass)
  // already assumes every item ran.
  //
  // *** WHAT THE PRECONDITION REQUIRES, AND WHAT IT NO LONGER DOES. ***
  // The kernel derives each item's group width and group count from its OWN
  // K and group_size (quant_gemm.cpp:2211-2212), and reads each item's own
  // K, N, scheme, code, scale, alpha and output pointer. So the wave does
  // NOT need a shared K or a shared group_size -- only a shared M (the
  // workspace slot is M*K per item, and the activation staging below is
  // M*K per item, so differing K is safe on both sides). Requiring equal K
  // and group_size used to refuse perfectly fusible waves where, say, the
  // q/k/v tensors share M and scheme but differ in width; nothing in the
  // arithmetic depends on them being equal.
  //
  // The one genuine refusal is the INT8 DPAS branch: the base path's
  // quant_gemm_forward_usm takes it for exactly
  //   scheme == INT8 && M >= 8 && (K % 32 == 0) && (N % 16 == 0) && group_size == 0
  // (quant_gemm.cpp:337) -- a different kernel (quant_gemm_dpas_int8), so
  // fusing an item that would take it would not be bit-exact vs the default.
  // Refusing anything wider (e.g. "all INT8 at M >= 8") would be safe but
  // over-broad: an INT8 item whose shape misses the DPAS conditions takes
  // the SAME generic dequant the fused kernel replicates, so it is fusible.
  // The check is per item, because K, N and group_size vary per tensor.
  if (wave_launch_fuse_enabled()) {
    bool ok = true;
    for (size_t i = 0; i < nl; ++i) {
      const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
      const auto& sc = shard_cache_[(size_t)live[i]->tensor_index];
      if (sc.scheme == gemm::WeightScheme::INT8 &&
          M >= 8 && (t.K % 32 == 0) && (t.N % 16 == 0) && sc.group_size == 0) {
        ok = false; break;
      }
    }
    if (ok && run_linear_wave_fused(live, nl, M)) {
      // *** INTFER_WAVE_LAUNCH_FUSE PRE-EMPTS THE PER-ITEM XMX DISPATCH
      // (2026-10-14). *** run_linear_wave_fused submits
      // quant_gemm_forward_usm_wave directly (engine.cpp:2336) -- ONE
      // parallel_for over the union of the wave's items -- so a wave that
      // takes it never reaches the per-item try-XMX-first dispatch below.
      // With INTFER_XMX_GEMM also set, the matrix engines therefore run
      // ONLY the non-fused waves (every wave that misses a fused-launch
      // precondition: INT8 items, mismatched M, ...), and a gate arm that
      // sets both flags and asserts the [xmx-gemm] wave announcement
      // would go red for a reason that is precedence, not a wiring break.
      // Correctness is unaffected: both kernels write the same compact
      // out[m*Ni+n] layout into the same slots and the Phase 4 readback is
      // shared, so the bytes are the same either way (XMX is a tolerance
      // change, not a bit change -- see the XMX wave arm in
      // scripts/test_tp2_27b.sh). This line names the precedence once per
      // process so the interaction is greppable instead of silent.
      static bool announced_wf_xmx = false;
      if (!announced_wf_xmx && std::getenv("INTFER_XMX_GEMM")) {
        announced_wf_xmx = true;
        std::fprintf(stderr,
                     "[wave-launch-fuse] fused wave took the EU/ALU "
                     "wave kernel; per-item INTFER_XMX_GEMM dispatch "
                     "skipped for this wave (n_items=%zu M=%u)\n",
                     nl, M);
        std::fflush(stderr);
      }
      return;
    }
  }

  // Phase 1: per item — convert + write each device's wave slot.
  // Shared-input fast path: wave items often share one input vector
  // (gate/up share rin; M-row batch gate/up share X). The FP32->FP16
  // convert is a pure function of immutable wave inputs, so convert each
  // DISTINCT (pointer, bytes) input once and reuse the bytes for
  // duplicates — bit-identical (same input bytes by construction), one
  // convert + one host-side staging copy saved per duplicate. Slots stay
  // per-item (kernels read their own slot); only the convert dedupes.
  // Scope is this call only (the shared_valid flag — no cross-call
  // staleness). The staging buffer is the PERSISTENT member
  // wave_shared_staging_ (resized on grow): the local vector it replaced
  // heap-allocated M*K halves (~1 MB on 27B prefill chunks, above glibc's
  // mmap threshold) on every wave call that had a reused input.
  const float* shared_ptr = nullptr;
  size_t shared_need = 0;
  bool shared_valid = false;
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
    const std::vector<float>& x = *live[i]->x;
    std::vector<float>& out = *live[i]->out;
    const uint32_t out_elems = M * t.N;
    if (out.size() != out_elems) out.resize(out_elems);
    const size_t need = (size_t)M * t.K;
    const bool dup = shared_valid && x.data() == shared_ptr && need == shared_need;
    if (!dup) {
      // Is this input reused by a later item? Only then is staging worth saving.
      bool reused = false;
      for (size_t j = i + 1; j < nl; ++j) {
        const auto& tj = model_.tensors[(size_t)live[j]->tensor_index];
        if (live[j]->x->data() == x.data() && (size_t)M * tj.K == need) { reused = true; break; }
      }
#ifdef INTFER_PROFILE_GPU
      auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
      if (act_staging_.size() != need) act_staging_.resize(need);
      ++act_staging_gen_;   // shared buffer: see run_linear_tp_devact
      dashq::float_to_half_batch(x.data(), act_staging_.data(), need);
      if (reused) {
        if (wave_shared_staging_.size() != need) wave_shared_staging_.resize(need);
        std::memcpy(wave_shared_staging_.data(), act_staging_.data(), need * sizeof(std::uint16_t));
        shared_ptr = x.data();
        shared_need = need;
        shared_valid = true;
      } else {
        shared_valid = false;
      }
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
        g_gpu_profile.copy_count++;
      }
      auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
      for (size_t d = 0; d < devices_.size(); ++d)
        std::memcpy(wv_[d * wv_W_ + i].act, act_staging_.data(),
                    (size_t)M * t.K * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
        g_gpu_profile.usm_write_count += (int)devices_.size();
      }
#endif
    } else {
      // Duplicate: slots get the saved staging bytes (no convert).
#ifdef INTFER_PROFILE_GPU
      auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
      for (size_t d = 0; d < devices_.size(); ++d)
        std::memcpy(wv_[d * wv_W_ + i].act, wave_shared_staging_.data(),
                    (size_t)M * t.K * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
        g_gpu_profile.usm_write_count += (int)devices_.size();
      }
#endif
    }
  }

  // Phase 2: per device — submit all items' kernels back-to-back.
  // *** INTFER_XMX_GEMM WAVE WIRING (2026-10-14). *** The same
  // try-XMX-first-then-fallback dispatch run_linear_tp applies
  // (engine.cpp:1444-1498), applied to the wave's per-item
  // launch. Without this the XMX joint_matrix kernel was
  // reachable ONLY through run_linear_tp (the down-projection and
  // run_linear_batch callers) and never through the q/k/v/z/b/a/
  // la_out wave -- the majority of layer-GEMM launches at prefill
  // (chunk_M=128, so M%16==0 holds on every full chunk; the
  // ragged tail chunk and every M<8 shape decline inside the
  // kernel and fall back to the tiled kernel below, byte-
  // identically). NOT bit-exact vs the tiled kernel: the DPAS
  // accumulation order differs (the xmx_probe measured 1-7
  // differing elements per shape at K>=128), so this is a
  // TOLERANCE-CLASS flag like INTFER_ONEMKL -- the announcement
  // is the reachability proof and a stream comparison against
  // base is the CONTRACT, not a check. Declined attempts submit
  // nothing (every decline path returns before q.submit), so the
  // submit accounting below still counts exactly one launch per
  // item.
  const char* xmx_env = std::getenv("INTFER_XMX_GEMM");
  static std::vector<std::uint32_t> xmx_tile_ctr;
  static bool xmx_wave_announced = false;
  if (xmx_env && xmx_tile_ctr.size() < devices_.size())
    xmx_tile_ctr.resize(devices_.size(), 0);
  const bool xmx_ts_env =
      (xmx_env != nullptr) &&
      (std::getenv("INTFER_XMX_TILE_SCHED") != nullptr);
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(devices_.size() * nl);
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    sycl::queue& q = dev.queue;
    for (size_t i = 0; i < nl; ++i) {
      const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
      const auto& sc = shard_cache_[(size_t)live[i]->tensor_index];
      const auto& ranges = shards_[live[i]->tensor_index].ranges;
      auto [start, end] = ranges[d];
      auto N_local = end - start;
      float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
      auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
      sycl::event kernel_evt;
      if (xmx_env) {
        // Try the XMX joint_matrix kernel first. The shape gates
        // (INT2/INT4 only, M>=8, M%16==0, N%16==0, K%32==0) are
        // the kernel's own; a decline (xmx_ran == false) submits
        // NOTHING and wrote nothing, so the fallback below is
        // mandatory, not cosmetic -- pushing the null event would
        // leave this item's wave slot unwritten.
        const uint32_t gx = (M / 16);
        const uint32_t gy = (N_local / 16);
        const uint32_t n_tiles = gx * gy;
        const uint32_t nWG = xmx_ts_env ? 16u : n_tiles;
        xmx_tile_ctr[d] = 0;
        bool xmx_ran = false;
        kernel_evt = gemm::quant_gemm_forward_usm_xmx(
            q, wv_[d * wv_W_ + i].act, sc.code_usm[d],
            sc.scale_usm[d], alpha, wv_[d * wv_W_ + i].out,
            M, N_local, t.K, sc.group_size, sc.scheme,
            sycl::event{}, &xmx_ran, xmx_tile_ctr.data() + d);
        if (!xmx_ran) {
          kernel_evt = gemm::quant_gemm_forward_usm(q,
                                        wv_[d * wv_W_ + i].act,
                                        sc.code_usm[d],
                                        sc.scale_usm[d],
                                        alpha,
                                        wv_[d * wv_W_ + i].out,
                                        M, N_local, t.K,
                                        sc.group_size,
                                        sc.scheme,
                                        sycl::event{});
        }
        xmx_note_dispatch(xmx_ran);
        // Announce once per process, ONLY when the kernel actually
        // ran -- the same xmx_ran-gated polarity as run_linear_tp's
        // announcement (engine.cpp:1487): a log where the kernel
        // declined every tile must NOT print the line, or the gate
        // arm's announcement-absent assertion is vacuous.
        if (xmx_ran && !xmx_wave_announced) {
          xmx_wave_announced = true;
          std::fprintf(stderr,
                       "[xmx-gemm] ACTIVE on eligible INT2/INT4 tiles "
                       "(INTFER_XMX_GEMM=1)\n"
                       "[xmx-tile-sched] %s n_tiles=%u nWG=%u "
                       "(%s work-stealing)\n",
                       xmx_ts_env ? "ACTIVE" : "OFF",
                       n_tiles, nWG,
                       xmx_ts_env ? "persistent" : "fixed-assignment");
        }
      } else {
        kernel_evt = gemm::quant_gemm_forward_usm(q,
                                      wv_[d * wv_W_ + i].act,
                                      sc.code_usm[d],
                                      sc.scale_usm[d],
                                      alpha,
                                      wv_[d * wv_W_ + i].out,
                                      M, N_local, t.K,
                                      sc.group_size,
                                      sc.scheme,
                                      sycl::event{});
      }
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
        g_gpu_profile.submit_count += 1;
      }
      if (kernel_events.size() + 1 > g_gpu_profile.peak_wait_depth)
        g_gpu_profile.peak_wait_depth = kernel_events.size() + 1;
#endif
      kernel_events.push_back(kernel_evt);
    }
  }

  // Phase 3: wait once for the whole wave (load-bearing: host must not
  // read wave outputs before the kernels run).
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  for (auto& e : kernel_events) e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += kernel_events.size();
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif

  // Phase 4: per item — read back from that item's wave slot.
  // readback_count is per (item, device), matching run_mlp_gateup's
  // granularity (two outputs x two devices = 4). Counting it once per
  // call made the counter non-comparable between the two paths the same
  // way submit_count and usm_write_count were: a fused token would have
  // reported readback_count=4 against a wave token reporting 1, and the
  // "the readback is unchanged" claim would have been a ratio of 4:1
  // rather than a delta of 0.
  uint32_t rb_items = 0;
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
    std::vector<float>& out = *live[i]->out;
    const auto& ranges = shards_[live[i]->tensor_index].ranges;
    for (size_t d = 0; d < devices_.size(); ++d) {
      auto [start, end] = ranges[d];
      const std::uint16_t* out_ptr = wv_[d * wv_W_ + i].out;
      const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
      for (uint32_t m = 0; m < M; ++m)
        dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                   out.data() + (size_t)m * t.N + start,
                                   N_local_rb);
      ++rb_items;
    }
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count += rb_items;
  }
#endif
}


#ifdef INTFER_PROFILE_GPU
  GpuProfileSnapshot get_gpu_profile() {
    return {g_gpu_profile.total_kernel_us, g_gpu_profile.total_wait_us,
            g_gpu_profile.total_copy_us, g_gpu_profile.total_readback_us,
            g_gpu_profile.total_usm_write_us, g_gpu_profile.total_submit_us,
            g_gpu_profile.total_waitblock_us,
            g_gpu_profile.total_rebuild_us, g_gpu_profile.rebuild_count,
            g_gpu_profile.peak_wait_depth,
            g_gpu_profile.kernel_count, g_gpu_profile.wait_count,
            g_gpu_profile.copy_count, g_gpu_profile.readback_count,
            g_gpu_profile.submit_count, g_gpu_profile.usm_write_count,
            g_gpu_profile.timeline_marker_count,
            g_gpu_profile.xmx_attempt_count,
            g_gpu_profile.xmx_launch_count,
            g_gpu_profile.xmx_decline_count};
  }
  void reset_gpu_profile() { g_gpu_profile = GpuProfile{}; }

  // ---- [xmx-attrib]: matrix-engine attribution for the counter dump ----
  // WHY IT PRINTS WHEN THE ANSWER IS UNKNOWN. AGENTS.md's standing verdict is
  // "the matrix engines are idle on the hot path", derived from measured runs.
  // A verdict a future session must re-derive is only useful if the dump SAYS
  // which of the three states this run was in, so the line always carries the
  // engine-side triple and NEVER substitutes a hardware figure it did not
  // read. xmxmon is an external Level Zero metric streamer; this box runs
  // dev.xe.observation_paranoid=1 with no sudo, so the observation session
  // cannot open from the engine, and scripts/xmx_profile.sh is the path that
  // joins XVE_INST_EXECUTED_XMX_* with these buckets.
  std::string xmx_attrib_line(const GpuProfileSnapshot& p) {
    const char* verdict = "not-requested";
    if (p.xmx_attempt_count > 0)
      verdict = (p.xmx_launch_count > 0) ? "active"
                                         : "requested-all-declined";
    // The absence is NAMED. Printing nothing (or an XVE-shaped zero) would be
    // the vacuous-green shape: a reader would read "0 XMX" as a measurement.
    std::string found;
    if (const char* e = std::getenv("XMXMON")) {
      struct stat st;
      if (::stat(e, &st) == 0 && (st.st_mode & S_IXUSR)) found = e;
    }
    if (found.empty()) {
      const char* cands[] = {"/tmp/xmxmon", "./xmxmon"};
      for (const char* c : cands) {
        struct stat st;
        if (::stat(c, &st) == 0 && (st.st_mode & S_IXUSR)) { found = c; break; }
      }
    }
    std::string s = "[xmx-attrib] attempts=";
    s += std::to_string(p.xmx_attempt_count);
    s += " launches=";
    s += std::to_string(p.xmx_launch_count);
    s += " declined=";
    s += std::to_string(p.xmx_decline_count);
    s += " matrix_engines=";
    s += verdict;
    if (!found.empty()) {
      s += " | xmxmon=";
      s += found;
      s += " (engine does NOT read XVE_INST_EXECUTED_XMX_*; run "
           "scripts/xmx_profile.sh to join the hardware counters with these "
           "buckets)";
    } else {
      s += " | XVE_INST_EXECUTED_XMX_*: NOT READ -- no xmxmon binary "
           "(XMXMON=/path/to/xmxmon, /tmp/xmxmon or ./xmxmon; build: g++ -O2 "
           "-o xmxmon xmxmon.cpp -lze_loader). Hardware XMX counters "
           "UNKNOWN on this box (dev.xe.observation_paranoid=1); the counts "
           "above are engine-side dispatch observations only";
    }
    return s;
  }
  // ---- Device-timeline markers (INTFER_TIMELINE=1) ----
  // GRIMOIRE's mark() pattern: submit a 1-work-item empty kernel and record
  // its event. On an in-order queue the gap between marker i's end and
  // marker i+1's start IS the device time of everything between them, with
  // no per-launch instrumentation and no dependence on
  // get_profiling_info<command_start/command_end> returning sane durations.
  // This is the ONLY way to attribute wall-clock to a region on this box,
  // where the driver records submit==start==end for real kernels while they
  // still execute and write the sink (see AGENTS.md "The profiling defect").
  //
  // The marker itself is a submit + a wait on the NEXT marker, so it costs
  // 1 submit + 1 wait per region. It is a DIAGNOSTIC, off by default, and
  // must never be used as a decision input: it measures host wall-clock
  // between two in-order points, which is load-sensitive. Decide on
  // nk/nw/nr, never on a timeline figure.
  namespace {
    struct TimelineMarker { const char* tag; sycl::event ev; };
    std::vector<TimelineMarker> g_timeline;
    bool g_timeline_on = false;
    bool g_timeline_init = false;
  }
  void timeline_mark(sycl::queue& q, const char* tag) {
    if(!g_timeline_init){
      g_timeline_init = true;
      g_timeline_on = (std::getenv("INTFER_TIMELINE") != nullptr);
      if(g_timeline_on) std::fprintf(stderr, "[timeline] device-timeline markers ON\n");
    }
    if(!g_timeline_on) return;
    sycl::event ev = q.submit([&](sycl::handler& h){
      h.parallel_for(sycl::range<1>(1), [](sycl::item<1>){});
    });
    g_timeline.push_back({tag, ev});
    g_gpu_profile.timeline_marker_count++;
  }
  void timeline_dump(){
    if(!g_timeline_on) return;
    if(g_timeline.size() < 2) return;
    // Drain every marker so the last one's end is known.
    for(auto& m : g_timeline) m.ev.wait();
    auto ns = [](const sycl::event& e){
      return (long long)(e.get_profiling_info<sycl::info::event_profiling::command_end>()
               - e.get_profiling_info<sycl::info::event_profiling::command_start>());
    };
    std::fprintf(stderr, "[timeline] %zu markers:\n", g_timeline.size());
    for(size_t i = 0; i + 1 < g_timeline.size(); ++i){
      long long gap = ns(g_timeline[i+1].ev) - ns(g_timeline[i].ev);
      // The gap between marker i's END and marker i+1's START is the device
      // time of the region between them. command_end of i minus
      // command_start of i+1 is negative when the queue is busy; the
      // region time is the complement. Both forms are reported: the
      // in-order interpretation is that the queue was busy for the whole
      // gap, so the region's device time is (start[i+1] - end[i]) when
      // positive, i.e. the idle gap. The useful figure is the total busy
      // span: end[i+1] - start[i].
      long long start_i = (long long)g_timeline[i].ev.get_profiling_info<sycl::info::event_profiling::command_start>();
      long long end_i   = (long long)g_timeline[i].ev.get_profiling_info<sycl::info::event_profiling::command_end>();
      long long start_n = (long long)g_timeline[i+1].ev.get_profiling_info<sycl::info::event_profiling::command_start>();
      long long end_n   = (long long)g_timeline[i+1].ev.get_profiling_info<sycl::info::event_profiling::command_end>();
      std::fprintf(stderr, "[timeline]   %s -> %s: region=%lld ns (end[%s]=%lld start[%s]=%lld)\n",
                   g_timeline[i].tag, g_timeline[i+1].tag,
                   start_n - end_i, g_timeline[i].tag, end_i, g_timeline[i+1].tag, start_n);
      (void)gap; (void)end_n; (void)start_i;
    }
  }
  namespace gemm {
  void profile_note_launch() { g_gpu_profile.kernel_count++; }
  }  // namespace gemm
#endif

void Engine::run_linear_f16_tp(const sycl::half* x, const sycl::half* w,
                               float* out, uint32_t N, uint32_t K,
                               const sycl::event& copy_dep) {
  // One-off per-device weight buffer. We copy the full W to each device;
  // each device computes the full N rows; the host reads back halves from
  // device 0 and converts to float on the host (c_dev holds N halves, not
  // N floats — a direct N*sizeof(float) device read is a 2x OOB read).
  // Device frees are deferred until after all events complete (sycl::free
  // acts on the host immediately; freeing in-loop races the kernel).
  struct DevBufs { sycl::half *w_dev, *x_dev, *c_dev; sycl::queue* q; };
  std::vector<DevBufs> bufs;
  bufs.reserve(devices_.size());
  std::vector<sycl::event> evts;
  evts.reserve(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    sycl::queue& q = dev.queue;
    size_t w_bytes = (size_t)N * K * sizeof(sycl::half);
    sycl::half* w_dev = static_cast<sycl::half*>(
        sycl::malloc_device(w_bytes, dev.sycl_device, dev.sycl_context));
    sycl::half* x_dev = static_cast<sycl::half*>(
        sycl::malloc_device((size_t)K * sizeof(sycl::half), dev.sycl_device, dev.sycl_context));
    sycl::half* c_dev = static_cast<sycl::half*>(
        sycl::malloc_device((size_t)N * sizeof(sycl::half), dev.sycl_device, dev.sycl_context));
    bufs.push_back({w_dev, x_dev, c_dev, &q});

    q.memcpy(w_dev, w, w_bytes);
    q.memcpy(x_dev, x, (size_t)K * sizeof(sycl::half)).wait();
    sycl::event evt = gemm::quant_gemm_f16(q, x_dev, w_dev, c_dev, 1, N, K);
    evts.push_back(evt);
  }
  for (auto& e : evts) e.wait();
  // Read back N halves from device 0 and convert on the host.
  std::vector<sycl::half> tmp(N);
  bufs[0].q->memcpy(tmp.data(), bufs[0].c_dev, (size_t)N * sizeof(sycl::half)).wait();
  for (uint32_t i = 0; i < N; ++i) out[i] = static_cast<float>(tmp[i]);
  for (auto& b : bufs) {
    sycl::free(b.w_dev, *b.q);
    sycl::free(b.x_dev, *b.q);
    sycl::free(b.c_dev, *b.q);
  }
}

const Rotation* Engine::rotation_for(int tensor_index) const {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return nullptr;
  const auto& t = model_.tensors[tensor_index];
  auto it = rid_.find(t.qcfg.rotation_id);
  return it == rid_.end() ? nullptr : it->second;
}

Engine::ShardView Engine::shard_view(int tensor_index) {
  ShardView v;
  if (devices_.size() != 1) return v;  // TP=1 full-row only
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return v;
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (ranges.size() != 1 || ranges[0].first != 0) return v;
  if (ranges[0].second - ranges[0].first != t.N) return v;  // full-row sanity
  if (t.N == 0 || t.K == 0) return v;
  if (!shard_cache_[(size_t)tensor_index].built &&
      !ensure_shard_built(tensor_index))
    return v;
  const auto& sc = shard_cache_[(size_t)tensor_index];
  if (sc.code_usm.empty() || !sc.code_usm[0]) return v;
  if (sc.scale_usm.empty() || !sc.scale_usm[0]) return v;
  v.ok = true;
  v.code = sc.code_usm[0];
  v.scale = sc.scale_usm[0];
  v.alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
  v.N = t.N;
  v.K = t.K;
  v.group_size = sc.group_size;
  v.scheme = sc.scheme;
  v.dpas48 = sc.dpas48;
  return v;
}

// Fill a freshly-acquired weight shard. Device USM is not host-writable, so the
// copy has to go through the queue there; shared USM keeps the plain memcpy it
// always used (that path is what preload timing was measured against, and changing
// it silently would move the 78 s preload number for no reason).
static void shard_fill(sycl::queue& q, void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (weights_device_usm()) {
    q.memcpy(dst, src, bytes).wait();
  } else {
    std::memcpy(dst, src, bytes);
  }
}

bool Engine::ensure_shard_built(int tensor_index) {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size()) return false;
  auto& sc = shard_cache_[tensor_index];
  if (sc.built) return true;

#ifdef INTFER_PROFILE_GPU
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  sc.built = true;
  sc.scheme = to_ws(t.qcfg.scheme);
  sc.group_size = t.qcfg.group_size;
  sc.has_global_divisor = t.qcfg.has_global_divisor;
  sc.K = t.K;
  sc.code_usm.resize(devices_.size());
  sc.scale_usm.resize(devices_.size());
  // Host-cache gate (see below): FP16 is memcpy-identity + resident, so only
  // cache the schemes whose shard_weight() does scalar unpack work (INT4/8/2).
  // skip_host_cache_ (preload_all): pinned shards are never rebuilt, so the
  // cache would be dead host RAM (~9.4GB on 27B-sc) — skip populating it.
  const bool cacheable =
      (sc.scheme != gemm::WeightScheme::FP16) && !skip_host_cache_;
  // *** INTFER_MLP_PARTIAL_DOWN: K-split full-N shard. *** Only the DOWN
  // tensor needs it (the partial-sum gather requires full-N partials), and
  // only when the flag is on, K splits evenly across devices, and the split
  // lands on a group boundary. The shard is built per device: device d gets
  // K-slice d (full N). It is INDEPENDENT of the column-split shard (which
  // the default path still uses), so both coexist — ~2x the down weight
  // memory while the flag is on (~100MB on 27B), acceptable for an opt-in.
  // DPAS48 shards decline inside run_mlp_partial_down, so never build the
  // ksplit shard for them (the repack layout would differ).
  const bool ksplit_ok =
      (std::getenv("INTFER_MLP_PARTIAL_DOWN") != nullptr) &&
      (t.role == TensorRole::MLP_DOWN) &&
      (devices_.size() == 2) &&
      (t.K % devices_.size() == 0) &&
      (t.qcfg.group_size == 0 || t.K % (t.qcfg.group_size * devices_.size()) == 0);
  sc.ksplit = ksplit_ok;
  sc.ksplit_K = sc.ksplit ? (t.K / devices_.size()) : 0;
  if (sc.ksplit) {
    sc.ksplit_code_usm.resize(devices_.size());
    sc.ksplit_scale_usm.resize(devices_.size());
  }

  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    uint32_t N_local = end - start;

    // 1) Host-side shard data: repack-once cache for non-FP16 schemes.
    //    shard_weight() is a scalar unpack+repack, immutable per (tensor,
    //    rank) — run it only on the first build, then memcpy from the cache
    //    (~8.8GB for the 27B INT4 set). FP16 is memcpy-identity and stays
    //    resident since Phase 1, so caching it would only waste host RAM
    //    (it OOMs: +13.5GB). The cacheable gate is computed before the loop.
    const std::uint8_t* code_ptr;
    size_t code_bytes;
    const std::uint16_t* scale_ptr;
    size_t scale_bytes;
    const float* global_ptr = nullptr;
    gemm::GemmWeightBuffers w;
    if (sc.has_host_cache && cacheable) {
      code_ptr = sc.code_host[d].data();
      code_bytes = sc.code_host[d].size();
      scale_ptr = sc.scale_host[d].data();
      scale_bytes = sc.scale_host[d].size() * sizeof(uint16_t);
      if (t.global_bytes > 0) global_ptr = &sc.global_host;
    } else {
      w = shard_weight(model_, t, start, end, q);
      code_ptr = w.code_store.data();
      code_bytes = w.code_store.size();
      scale_ptr = w.scale_store.data();
      scale_bytes = w.scale_store.size() * sizeof(uint16_t);
      if (t.global_bytes > 0) {
        float alpha_val = 0.f;
        std::memcpy(&alpha_val, model_.global_data() + t.global_offset,
                    std::min<uint32_t>(t.global_bytes, 4u));
        sc.global_host = alpha_val;
        global_ptr = &sc.global_host;
      }
    }

    // 2) Allocate persistent USM on each device and copy weight data.
    //    On first build, retain the shard_weight() output in the host cache.
    //    TODO 6 (W4A8/W2A8): DPAS-eligible shards repack row-major codes to
    //    vertical-B tiles ONCE here (repack_dpas412); the kernel then reads
    //    code_usm[d] directly. dpas_bytes[d] records the USM size so
    //    release_shard pools by exact size (code_host keeps the row-major
    //    bytes for rebuilds — repack re-runs from them, still no scalar
    //    unpack repetition of the group-major payload).
    sc.N_local = N_local;
    if (!sc.has_host_cache && cacheable) {
      if (sc.code_host.size() != devices_.size()) {
        sc.code_host.resize(devices_.size());
        sc.scale_host.resize(devices_.size());
      }
      sc.code_host[d] = w.code_store;
      sc.scale_host[d] = w.scale_store;
    }
    if (d == 0) {
      sc.dpas48 = dpas48_eligible(t.qcfg.scheme, t.qcfg.group_size,
                                  N_local, t.K) &&
                  (ranges.size() == devices_.size());
      // All ranks share N_local only for even splits; re-check per rank
      // below (N_local varies per d on asymmetric lm_head-style ranges —
      // run_linear shards are even, but verify anyway).
      for (size_t dd = 0; dd < devices_.size(); ++dd) {
        uint32_t Nl_dd = ranges[dd].second - ranges[dd].first;
        if (!dpas48_eligible(t.qcfg.scheme, t.qcfg.group_size, Nl_dd, t.K)) {
          sc.dpas48 = false;
          break;
        }
      }
      if (sc.dpas48) sc.dpas_bytes.assign(devices_.size(), 0);
    }

    // Code: packed bytes (pool: same shapes recur every layer).
    if (sc.dpas48) {
      const int bits = (sc.scheme == gemm::WeightScheme::INT2)    ? 2
                       : (sc.scheme == gemm::WeightScheme::INT4)  ? 4
                                                                 : 8;
      std::vector<std::uint8_t> dpas_tiles;
      if (!gemm::repack_dpas412(code_ptr, N_local, t.K, bits, dpas_tiles)) {
        // Shape refused (should not happen after dpas48_eligible, but
        // safe): fall back to the row-major path for this shard.
        sc.dpas48 = false;
        sc.code_usm[d] =
            static_cast<std::uint8_t*>(pool_acquire(d, code_bytes));
        shard_fill(q, sc.code_usm[d], code_ptr, code_bytes);
      } else {
        sc.dpas_bytes[d] = dpas_tiles.size();
        sc.code_usm[d] = static_cast<std::uint8_t*>(
            pool_acquire(d, dpas_tiles.size()));
        shard_fill(q, sc.code_usm[d], dpas_tiles.data(), dpas_tiles.size());
      }
    } else {
      sc.code_usm[d] =
          static_cast<std::uint8_t*>(pool_acquire(d, code_bytes));
      shard_fill(q, sc.code_usm[d], code_ptr, code_bytes);
    }

    // Scale: N_local * n_groups FP16
    sc.scale_usm[d] = static_cast<std::uint16_t*>(pool_acquire(d, scale_bytes));
    // Same allocator decision as the code bytes above: scales go through shard_fill
    // so a device-USM scale buffer is filled by the queue, not by the host. They are
    // 21.5 % of nothing on their own but they are read in the GEMV's innermost
    // group loop, so their residency is the same question as the weights'.
    shard_fill(q, sc.scale_usm[d], scale_ptr, scale_bytes);
    // NOTE: no global_usm alloc — alpha stays host-side (global_host) and is
    // passed to the kernel by value (see run_linear_tp).

    // NOTE: no per-shard act/out — run_linear_tp writes into the shared
    // per-device workspaces (ensure_workspace, grow-only).
    //
    // *** INTFER_MLP_PARTIAL_DOWN: K-split shard build. *** Device d's shard
    // covers full N (start=0, end=t.N) x K-slice [d*ksplit_K, (d+1)*ksplit_K)
    // -- the strided pack shard_weight(..., k_off=d*ksplit_K, k_len=ksplit_K)
    // now emits row-major codes + per-(row,group) scales for that slice. The
    // host cache is retained across evict+rebuild exactly like the
    // column-split shard. This is a DIFFERENT layout from code_usm[d], so it
    // needs its own host cache vectors (ksplit_code_host/scale_host).
    if (sc.ksplit) {
      const uint32_t k_off = sc.ksplit_K * (uint32_t)d;
      gemm::GemmWeightBuffers wk =
          shard_weight(model_, t, 0, t.N, q, k_off, sc.ksplit_K);
      const size_t code_bytes_k = wk.code_store.size();
      const size_t scale_bytes_k = wk.scale_store.size() * sizeof(uint16_t);
      if (!cacheable) {
        sc.ksplit_code_usm[d] =
            static_cast<std::uint8_t*>(pool_acquire(d, code_bytes_k));
        sc.ksplit_scale_usm[d] =
            static_cast<std::uint16_t*>(pool_acquire(d, scale_bytes_k));
        shard_fill(q, sc.ksplit_code_usm[d], wk.code_store.data(), code_bytes_k);
        shard_fill(q, sc.ksplit_scale_usm[d], wk.scale_store.data(), scale_bytes_k);
      } else {
        if (sc.ksplit_code_host.size() != devices_.size()) {
          sc.ksplit_code_host.resize(devices_.size());
          sc.ksplit_scale_host.resize(devices_.size());
        }
        sc.ksplit_code_host[d] = wk.code_store;
        sc.ksplit_scale_host[d] = wk.scale_store;
        sc.ksplit_code_usm[d] =
            static_cast<std::uint8_t*>(pool_acquire(d, code_bytes_k));
        sc.ksplit_scale_usm[d] =
            static_cast<std::uint16_t*>(pool_acquire(d, scale_bytes_k));
        shard_fill(q, sc.ksplit_code_usm[d],
                 sc.ksplit_code_host[d].data(), code_bytes_k);
        shard_fill(q, sc.ksplit_scale_usm[d],
                 sc.ksplit_scale_host[d].data(), scale_bytes_k);
      }
    }
  }
  if (cacheable) sc.has_host_cache = true;
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_rebuild_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.rebuild_count++;
  }
#endif
  return true;
}

// *** INTFER_MLP_PARTIAL_DOWN (2026-10-14). ***
// TP=2 MLP down-projection as a partial-sum gather, behind the default-OFF
// presence-spelled INTFER_MLP_PARTIAL_DOWN. Device d keeps its gate/up
// outputs ON DEVICE (wv_ slots 1/2), fuses silu(gate)*up via the existing
// silumul_to_half into a per-device M*inter FP16 buffer, then runs the down
// GEMM over its K-sliced FULL-N shard (N=td.N, K=td.K/devices_.size()),
// writing an FP16 partial. The host waits once and sums the two partials in
// FP32 into md. Removes the full-inter D2H + the host silu loop + the full
// inter H2D + the separate down GEMM round trips that the default TP=2 path
// pays; adds 1 silumul launch + 1 partial-down launch + 1 wait/device vs the
// default's 2. NOT bit-exact (two FP16 partials summed in FP32 vs the tiled
// kernel's single FP32-accumulate-then-FP16-store) -> tolerance contract,
// never stream identity. Returns false (caller keeps the host gateup path) on
// the flag unset, TP!=2, DPAS48 shards, bad shapes, or K not splittable into
// a group-aligned K-slice.
bool Engine::run_mlp_partial_down(int gate_ti, int up_ti, int down_ti,
                                  const std::vector<float>& rin,
                                  std::vector<float>& md, uint32_t M) {
  if (devices_.size() != 2) return false;
  if (!std::getenv("INTFER_MLP_PARTIAL_DOWN")) return false;
  if (M == 0) M = 1;
  for (int ti : {gate_ti, up_ti, down_ti}) {
    if (ti < 0 || (size_t)ti >= model_.tensors.size()) return false;
    if (!shard_cache_[(size_t)ti].built && !ensure_shard_built(ti))
      return false;
    if (shard_cache_[(size_t)ti].dpas48) return false;
  }
  const auto& tg = model_.tensors[(size_t)gate_ti];
  const auto& tu = model_.tensors[(size_t)up_ti];
  const auto& td = model_.tensors[(size_t)down_ti];
  const auto& scg = shard_cache_[(size_t)gate_ti];
  const auto& scu = shard_cache_[(size_t)up_ti];
  const auto& scd = shard_cache_[(size_t)down_ti];
  const uint32_t inter = tg.N;
  if (tu.N != inter || tu.K != tg.K) return false;
  if (td.K != inter) return false;
  // Only INT2+INT2 gate/up are supported (the fused gateup kernel has no
  // other body); the down tensor must also be INT2 for the ksplit shard to
  // match the scheme.
  if (tg.qcfg.scheme != QuantScheme::INT2) return false;
  if (scg.code_usm.size() != 2 || scu.code_usm.size() != 2) return false;
  if (!scd.ksplit || scd.ksplit_code_usm.size() != 2) return false;
  for (size_t d = 0; d < 2; ++d) {
    if (!scg.code_usm[d] || !scg.scale_usm[d] ||
        !scu.code_usm[d] || !scu.scale_usm[d]) return false;
    if (!scd.ksplit_code_usm[d] || !scd.ksplit_scale_usm[d]) return false;
  }
  if (rin.size() != (size_t)M * tg.K) return false;
  if (md.size() != (size_t)M * td.N) md.resize((size_t)M * td.N);
  // Zero md for the FP32 partial sum.
  for (size_t i = 0; i < (size_t)M * td.N; ++i) md[i] = 0.0f;

  // Shared FP32->FP16 staging for the gate/up input.
  const size_t need = (size_t)M * tg.K;
  if (act_staging_.size() != need) act_staging_.resize(need);
  ++act_staging_gen_;
  dashq::float_to_half_batch(rin.data(), act_staging_.data(), need);

  // Wave workspace: slot 0 = shared act, slot 1 = gate out (compact M*Nl_g),
  // slot 2 = up out (compact M*Nl_u). Nl_g/Nl_u <= inter, and we need M*inter
  // for silumul output. The down workspace ws_ is sized for M x td.K (=inter)
  // act and M x td.N out.
  ensure_wave_workspace(M, 3, tg.K, std::max(tg.N, tu.N));
  ensure_workspace(M, td.K, td.N);

  const float ag = scg.has_global_divisor ? scg.global_host : 1.0f;
  const float au = scu.has_global_divisor ? scu.global_host : 1.0f;
  const float ad = scd.has_global_divisor ? scd.global_host : 1.0f;

  std::vector<sycl::event> all_events;
  all_events.reserve(6);

  for (size_t d = 0; d < 2; ++d) {
    auto& dev = devices_[d];
    sycl::queue& q = dev.queue;
    std::uint16_t* act_ptr = wv_[d * wv_W_ + 0].act;

    // Upload shared activation to device d (slot 0).
#ifdef INTFER_PROFILE_GPU
    auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
    std::memcpy(act_ptr, act_staging_.data(), need * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_usm_write_us +=
          std::chrono::duration<double, std::micro>(t1 - t_w0).count();
      g_gpu_profile.usm_write_count += 1;
    }
#endif

    const uint32_t Nl_g = static_cast<uint32_t>(
        shards_[gate_ti].ranges[d].second - shards_[gate_ti].ranges[d].first);
    const uint32_t Nl_u = static_cast<uint32_t>(
        shards_[up_ti].ranges[d].second - shards_[up_ti].ranges[d].first);
    const uint32_t start_g = shards_[gate_ti].ranges[d].first;
    const uint32_t start_u = shards_[up_ti].ranges[d].first;

    // *** Phase 1+2 (INTFER_MLP_GATEUP_SILUMUL_FUSE, default OFF). ***
    //
    // FUSE SET: ONE launch computes gate+up AND silu(gate)*up, writing the
    // compact [M*(Nl_g+Nl_u)] FP16 activation straight into ws_[d].act.
    // The separate gateup launch below must NOT also run -- the fused
    // kernel re-computes the same dot products into the same slots, so
    // running both would DOUBLE the gate+up GEMM work per device while
    // writing identical bytes. (Found 2026-10-14: the original wiring ran
    // Phase 1 unconditionally and Phase 2's fused kernel on top, so the
    // "fusion" ADDED a launch instead of replacing one -- the exact
    // vacuous-gate shape the audit log warns about. The announcement
    // fired, the launch count went UP, and nothing in the gate arm could
    // see it because the stream was still correct.)
    //
    // FUSE UNSET: the two-launch chain -- compact gate+up GEMM into
    // slots 1/2, then silumul_to_half chained off its event into
    // ws_[d].act. Bit-exact in both branches: the fused kernel's
    // accumulators, kg/j element order and wtab are the compact gateup
    // kernel's, and the silu-mul epilogue is silumul_to_half's FP32 order
    // and RNE FP16 store.
    sycl::event e_silu;
    if (std::getenv("INTFER_MLP_GATEUP_SILUMUL_FUSE")) {
      e_silu = gemm::quant_gemm_forward_usm_gateup_compact_silumul(
          q, act_ptr,
          scg.code_usm[d], scg.scale_usm[d], ag,
          wv_[d * wv_W_ + 1].out,
          scu.code_usm[d], scu.scale_usm[d], au,
          wv_[d * wv_W_ + 2].out,
          M, tg.N, tu.N, tg.K, scg.group_size,
          start_g, Nl_g, start_u, Nl_u,
          scg.scheme, scu.scheme,
          sycl::event{},
          ws_[d].act);
      all_events.push_back(e_silu);
    } else {
      sycl::event e_gu = gemm::quant_gemm_forward_usm_gateup_compact(
          q, act_ptr,
          scg.code_usm[d], scg.scale_usm[d], ag,
          wv_[d * wv_W_ + 1].out,
          scu.code_usm[d], scu.scale_usm[d], au,
          wv_[d * wv_W_ + 2].out,
          M, tg.N, tu.N, tg.K, scg.group_size,
          start_g, Nl_g, start_u, Nl_u,
          scg.scheme, scu.scheme,
          sycl::event{});
      all_events.push_back(e_gu);
#ifdef INTFER_PROFILE_GPU
      g_gpu_profile.submit_count += 1;
#endif
      gemm::SiluMulParams sp;
      sp.mg_fp16 = wv_[d * wv_W_ + 1].out;
      sp.mu_fp16 = wv_[d * wv_W_ + 2].out;
      sp.act_fp16 = ws_[d].act;
      sp.n = M * inter;
      e_silu = gemm::silumul_to_half(q, sp, e_gu);
      all_events.push_back(e_silu);
    }
#ifdef INTFER_PROFILE_GPU
    // Launch counting: the non-FUSE branch's gateup launch adds its own
    // submit_count above (the gateup kernel submits explicitly, so the
    // post-wait profiling loop would miss it); the FUSE branch's single
    // launch is counted in the post-wait loop below like every other
    // event-pushed launch. Neither branch double-counts.
#endif

    // Phase 3: partial down GEMM over K-slice (ws_[d].act -> ws_[d].out).
    sycl::event e_down = gemm::quant_gemm_forward_usm(
        q, ws_[d].act,
        scd.ksplit_code_usm[d], scd.ksplit_scale_usm[d], ad,
        ws_[d].out,
        M, td.N, scd.ksplit_K, scd.group_size, scd.scheme,
        e_silu);
    all_events.push_back(e_down);
#ifdef INTFER_PROFILE_GPU
    g_gpu_profile.submit_count += 1;
#endif
  }

  // Wait for all device chains.
  for (auto& e : all_events) e.wait();
#ifdef INTFER_PROFILE_GPU
  for (auto& e : all_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  g_gpu_profile.wait_count += all_events.size();
#endif

  // Host FP32 partial sum: md += partial_0 + partial_1.
  // NOTE: there is NO D2H transfer here. ws_[d].out is USM (malloc_shared),
  // so the host reads the partials straight from device memory -- this is
  // half of the win (the default path's full-inter D2H is gone). The loop is
  // host-side FP16->FP32, not a device transfer, so it is NOT counted as a
  // readback (readback_count counts D2H transfers, and there are none here).
  // It is load-sensitive and is deliberately not bucketed into any host
  // accumulator -- the counters that matter are nk/nw/nr and the submit
  // buckets, which are device-driven.
  //
  // *** BATCHED, NOT SCALAR (2026-10-15). *** The original loop called
  // dashq::half_to_float once per element -- the last scalar conversion site
  // in the tree (every default readback at engine.cpp:1346/1632/1788/2231/
  // 2254/2372/2513/2934 uses half_to_float_batch). half_to_float_batch takes
  // the F16C path (_mm256_cvtph_ps, 8 lanes) when available; FP16->FP32 is an
  // EXACT conversion, so the FP32 bytes are identical to the scalar twiddle's
  // and the md values are bit-for-bit what the scalar loop produced. The
  // +0.0f seed above is KEPT rather than fused into d==0's store: +0.0f +
  // (-0.0f) is +0.0f while a direct seed would propagate -0.0f, and the seed
  // costs one FP32 pass against a gather that converts 2*M*td.N halves.
  if (pd_gather_.size() < (size_t)M * td.N) pd_gather_.resize((size_t)M * td.N);
  for (size_t d = 0; d < 2; ++d) {
    dashq::half_to_float_batch(ws_[d].out, pd_gather_.data(),
                               (size_t)M * td.N);
    const float* part = pd_gather_.data();
    for (size_t i = 0; i < (size_t)M * td.N; ++i) md[i] += part[i];
  }

  // Once-per-process announcement.
  static bool announced_pd = false;
  if (!announced_pd) {
    announced_pd = true;
    const uint32_t Nl_g_ann =
        shards_[gate_ti].ranges[0].second - shards_[gate_ti].ranges[0].first;
    const uint32_t Nl_u_ann =
        shards_[up_ti].ranges[0].second - shards_[up_ti].ranges[0].first;
    std::fprintf(stderr,
                 "[mlp-partial-down] ACTIVE (INTFER_MLP_PARTIAL_DOWN=1, "
                 "Ng=%u Nu=%u inter=%u K=%u group=%u scheme=%d, "
                 "two FP16 partials per device -> host FP32 sum)\n",
                 tg.N, tu.N, inter, tg.K, scg.group_size,
                 (int)scg.scheme);
    if (std::getenv("INTFER_MLP_GATEUP_SILUMUL_FUSE")) {
      std::fprintf(stderr,
                   "[mlp-gateup-silumul-fuse] ACTIVE (INTFER_MLP_GATEUP_SILUMUL_FUSE=1, "
                   "Ng=%u Nu=%u Nl_g=%u Nl_u=%u K=%u group=%u): gate+up GEMM and "
                   "silu(gate)*up epilogue in ONE launch per device\n",
                   tg.N, tu.N, Nl_g_ann, Nl_u_ann, tg.K, scg.group_size);
    }
    std::fflush(stderr);
  }
  return true;
}


bool Engine::load_language_weights(const std::string& safetensors_path) {
  SafetensorsReader r;
  if (!r.open(safetensors_path)) {
    std::cerr << "Engine: cannot open safetensors " << safetensors_path << "\n";
    return false;
  }
  for (const auto& t : r.tensors()) {
    std::vector<float> v;
    if (!r.read(t.name, v)) { std::cerr << "Engine: read fail " << t.name << "\n"; return false; }
    lang_[t.name] = std::move(v);
  }
  std::cerr << "Engine: loaded " << lang_.size() << " safetensors tensors\n";
  return true;
}

}  // namespace intfer


