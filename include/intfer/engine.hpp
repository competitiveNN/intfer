// include/intfer/engine.hpp
// Stage-2 inference engine (Intel Arc / SYCL / Level-Zero).
//
// Wires the Stage-1 quantized ModelIR (from serialize_intfer) into a runnable
// engine that follows the AGENTS.md Stage-2 requirements:
//   * Low-bit quantized GEMM via intfer::gemm::quant_gemm_forward, which lowers
//     the INT8 x INT8 case onto the Intel XMX DPAS (Dot Product Accumulate
//     Systolic) instruction (use_dpas = true).
//   * Fused FlashAttention via intfer::attention::flash_attention_forward_f16.
//   * Tensor-parallel sharding of the Linear projections across the 2x B580
//     (column sharding of the output dimension N; one shard per GPU).
//   * SYCL command-graph capture of the per-token decode step so the CPU launch
//     overhead is ~zero during generation (replaces CUDA graphs).
//
// This is the engine skeleton: it builds the shards, the GEMM/attention ops and
// the command graph. The full Qwen3.5 hybrid forward (RMSNorm, mRoPE, the
// gated-delta-rule linear-attention SSM, KV cache, tokenizer, sampling) is the
// remaining integration that plugs into these primitives.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include "intfer/quant_spec.h"
#include "intfer/spinquant.hpp"
#include "intfer/gemm/quant_gemm.hpp"
#include "intfer/device/device_context.hpp"

namespace intfer {

class Engine {
 public:
  // Load a native-SYCL (.intfer) artifact and shard its Linear weights across
  // every discovered Arc GPU. `tp_axis` = how many devices to use (default: all).
  static std::unique_ptr<Engine> load(const std::string& intfer_path,
                                      size_t tp_devices = 0);

  size_t num_devices() const { return devices_.size(); }
  size_t num_layers() const { return layer_names_.size(); }

  // Run ONE autoregressive decode step for a single token's embedding vector
  // (length = hidden). Returns the post-MLP hidden of the last layer (length
  // hidden). This submits (or replays the captured command graph for) the
  // quantized GEMM + FlashAttention compute across all shards/layers.
  // `tensor_index` selects which Linear tensor is exercised (defaults to 0).
  std::vector<float> decode_step(const std::vector<float>& token_hidden,
                                 int tensor_index = 0);
  // Batched decode: M tokens, each hidden length K. Returns M x hidden.
  std::vector<std::vector<float>> decode_step_batch(const std::vector<std::vector<float>>& batch_hidden,
                                 int tensor_index = 0);

  // --- Full hybrid forward (Stage-2 integration) -----------------------------
  // Get the model IR (readonly).
  const ModelIR& model() const { return model_; }
  
  // Load the non-quantized language weights (embed, norms, SSM) from safetensors.
  bool load_language_weights(const std::string& safetensors_path);
  
  // Run ONE quantized Linear tensor-parallel across the fleet.
  void run_linear_tp(int tensor_index, const std::vector<float>& x,
                     std::vector<float>& out);
  // Batched variant: x is M*K row-major; out is M*N row-major.
  void run_linear_tp(int tensor_index, const std::vector<float>& x,
                     std::vector<float>& out, uint32_t M);

  // Wave variant: one independent GEMM per item (distinct tensors, distinct
  // inputs/outputs, no data dependencies between items). Converts + writes
  // every item's activation, submits every kernel, waits once, then reads
  // back every output — so the device runs the wave back-to-back and pays
  // one host round-trip per wave instead of per GEMM. Bit-identical to
  // sequential run_linear_tp calls (same kernels, same per-item buffers).
  // Items with tensor_index < 0 are skipped (lets callers pass optional
  // slots without branching). M is the row count shared by all items.
  struct LinearWaveItem {
    int tensor_index = -1;
    const std::vector<float>* x = nullptr;
    std::vector<float>* out = nullptr;
    // The tensor's HF name (model_.tensors[tensor_index].name), carried by the
    // caller so the fused-wave announcement can name WHICH wave fused. The
    // launch count alone cannot: a wave of 4 items is 4 separate launches in
    // the default path and 1 in the fused path, so `items=N` distinguishes
    // fused from unfused but says nothing about which tensors were in it --
    // the q/k/v wave-3 and the z/b/a wave-4 both print items=3/4 and a gate
    // reading only that field would certify "the q/k/v wave fused" on a log
    // where only the z/b/a wave did. Optional (nullptr = caller did not set
    // it); the announcement falls back to the count-only form rather than
    // crashing on a null deref.
    const std::string* name = nullptr;
  };
  void run_linear_wave(const LinearWaveItem* items, size_t n, uint32_t M);

  // Fused wave launch (INTFER_WAVE_LAUNCH_FUSE): one parallel_for per device
  // over the UNION of the wave's output elements, instead of nl separate
  // launches. Bit-exact by construction (each item's accumulator is
  // untouched), so the default path is byte-for-byte the old one when the
  // flag is unset. Returns true on success (the caller's Phase 3/4
  // wait+readback then apply to the whole wave); false means the caller must
  // fall through to the default path. Declines on a mismatched M/K/group
  // across items, on INT8 items (which take the host-quant DPAS path), or on
  // any item the fused kernel does not support.
  bool run_linear_wave_fused(const LinearWaveItem** items, size_t nl,
                             uint32_t M);

  // Fused MLP down-projection (TP=1 only): gate/up wave-submit chained
  // on-device through silu-mul + FP16 convert straight into the down
  // GEMM — one host wait + one D2H per MLP instead of wave-wait +
  // readbacks + host math + convert + H2D + down-wait. Returns false
  // (caller keeps host path) on TP>1, DPAS48 items, or bad shapes.
  // md is resized to M*down.N. mg/mu host vectors are NOT filled.
  bool run_mlp_fused(int gate_ti, int up_ti, int down_ti,
                     const std::vector<float>& rin,
                     std::vector<float>& md, uint32_t M);

  // Fused gate+up GEMM, TP=2 (the only fusion that survives column sharding).
  // Gate and up are both column-parallel over the SAME split and device-local,
  // so one parallel_for over Ng+Nu outputs is bit-exact vs two separate
  // quant_gemm_forward_usm calls: each output keeps its own accumulator,
  // ascending kg, ascending j, its own group scale and wtab. Saves one submit
  // + one wait + one D2H readback per device per token vs the wave path.
  // Returns false (caller keeps the wave) on TP!=2, DPAS48 items, or
  // mismatched K/group/scheme. mg/mu are filled; md is NOT (the caller's down
  // path consumes mg/mu via silu-mul, same contract as run_mlp_fused's host
  // fallback).
  //
  // M>1 IS SUPPORTED, not a decline. The kernel's outer m loop iterates the
  // rows, so the SAME launch covers prefill and decode (783e56f). An earlier
  // version had no tiled M>1 body and returned a NULL event for M>1; the
  // engine then declined with `if (M > 1) return false;` and the M>1 prefill
  // arms took run_linear_wave_batch. That guard and the NULL-event path are
  // gone -- the M>1 indexing bug they papered over (reading the wrong weight
  // row and the wrong activation row) is fixed in the kernel, so the decline
  // would now be a live performance loss, not a safety net.
  bool run_mlp_gateup(int gate_ti, int up_ti,
                      const std::vector<float>& rin,
                      std::vector<float>& mg, std::vector<float>& mu,
                      uint32_t M, bool readback = true,
                       std::vector<sycl::event>* chain_out = nullptr);

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
  bool run_mlp_partial_down(int gate_ti, int up_ti, int down_ti,
                            const std::vector<float>& rin,
                            std::vector<float>& md, uint32_t M);

  // Device-act GEMM seam (L4 la_out fusion): run tensor ti reading M*K FP16
  // activation bits from a caller-owned device buffer (no host convert, no
  // H2D), dep-chained after the producer kernel; waits once and reads back
  // M*N FP32 rows to out. Returns false (caller keeps host path) on TP>1
  // (device acts are not cross-visible), DPAS48 shards, or bad shapes —
  // never partial work. The buffer must outlive the call (the wait is
  // inside). Output staging reuses ws_[0].out, so dev_act must not alias
  // it; otherwise any dedicated caller-owned device buffer works (same
  // pattern as run_mlp_fused feeding ws_[0].act into the down GEMM).
  bool run_linear_tp_devact(int tensor_index, const std::uint16_t* dev_act,
                            std::vector<float>& out, uint32_t M,
                            const sycl::event& dep = {});

  // Pre-flight for run_linear_tp_devact_tp2: every STATIC precondition the
  // seam can check without touching device state. The TP=2 la_out fusion
  // calls this BEFORE submitting the SSM chain kernel, because the chain
  // advances the recurrent state in place and a decline after that would
  // corrupt it (the fusion would have to re-run the recurrence to produce
  // c_ncore for the host path). This is the shipped "eligibility predicate
  // consulted before the launch" pattern (run_linear_tp_dpas48 at
  // engine.cpp:1270) -- the kernel re-checks the same facts, so this is
  // defence in depth, not the thing being certified.
  bool la_out_tp2_eligible(int tensor_index, uint32_t M, size_t out_elems) const;

  // TP=2 twin of run_linear_tp_devact (INTFER_LA_OUT_TP2). Same device-act
  // contract (caller-owned FP16 activation per device, no host convert, no
  // H2D, dep-chained after the producer kernel) but every rank runs its OWN
  // N-slice: dev_act[d] is device d's copy of the M*K FP16 act, ws_[d].out is
  // device d's workspace, and the readback scatters each rank back into
  // out.data() + m*t.N + start by ranges[d] -- byte-identical to the
  // run_linear_tp tail, so the only difference from the host path is WHERE
  // the activation bits came from. Requires the tensor to be column-sharded
  // across exactly 2 ranks; every other shape returns false so the caller
  // keeps its host path. No P2P is needed or assumed: the caller is
  // responsible for placing the act bytes in each device's context (the
  // same contract run_mlp_gateup documents -- distinct USM per rank).
  bool run_linear_tp_devact_tp2(int tensor_index,
                                const std::uint16_t* const* dev_act,
                                std::vector<float>& out, uint32_t M,
                                const sycl::event& dep = {});

  // FP16 x FP16 GEMM for non-quantized final heads (lm_head/embed).
  // C[1,N] = x[1,K] * W[N,K]^T. All row-major. Outputs FP32 logits.
  void run_linear_f16_tp(const sycl::half* x, const sycl::half* w,
                         float* out, uint32_t N, uint32_t K,
                         const sycl::event& copy_dep = {});
  
  // Which Rotation (SpinQuant) belongs to tensor ti (by its rotation_id).
  const Rotation* rotation_for(int tensor_index) const;

  // Read-only weight-shard view (L5 front chain): lets main.cpp drive
  // gemm::quant_gemm_forward_usm directly into caller-owned device buffers
  // (FWHT acts -> GEMM -> la_conv_prep) without a host round-trip. Building
  // the shard has no numeric effect (same bytes run_linear_tp would use).
  // TP=1 full-row only (device buffers are not cross-visible); the caller
  // re-checks num_devices()==1 and declines otherwise. dpas48 shards are
  // reported with dpas48=true (caller keeps the host path).
  struct ShardView {
    bool ok = false;
    const void* code = nullptr;          // packed codes (device USM)
    const std::uint16_t* scale = nullptr;// FP16 scales (device USM)
    float alpha = 1.0f;                  // global divisor (host value)
    uint32_t N = 0, K = 0;               // full-tensor geometry
    uint32_t group_size = 0;
    gemm::WeightScheme scheme = gemm::WeightScheme::INT8;
    bool dpas48 = true;
  };
  ShardView shard_view(int tensor_index);
  
  // Convenience: run a Linear TP GEMM and return the output vector.
  std::vector<float> run_linear_tp_ret(int tensor_index, const std::vector<float>& x) {
    std::vector<float> out; run_linear_tp(tensor_index, x, out); return out;
  }

  // --- Layer-wise VRAM streaming (27B on 12GB GPUs) -------------------------
  // The full-model shard cache (~23GB for 27B) exceeds B580 VRAM. Call
  // begin_layer(L) when the caller advances to layer L: it frees every built
  // shard that does not belong to L (unlayered tensors like lm_head stay
  // resident). The next run_linear_tp for a freed tensor rebuilds its shard
  // on demand, so streaming is numerically transparent. Peak VRAM drops to
  // one layer + lm_head (~1.1GB on 27B).
  //
  // Full-VRAM residency: preload_all() builds every runnable Linear shard
  // once (except lm_head-named tensors, which the CLI uploads via its own
  // TP-sharded run_lm_head_f16 path) and pins them: begin_layer() and
  // release_shard() become no-ops, so steady-state prefill/decode performs
  // zero evict+rebuild. Measured 27B-sc TP=2: ~10.9GB weights/dev + 2.5GB
  // sharded lm_head vs 12216MiB budget (dev0 keeps ~384MiB display
  // reservation). INTFER_PRELOAD=0 restores layer streaming.
  void begin_layer(int L);
  // Build all runnable Linear shards now and pin them resident (no evict).
  // Frees the 9.4GB INT4 host repack cache afterwards (no rebuilds remain).
  void preload_all();
  // True when preload_all() pinned the shards (streaming disabled).
  bool preloaded() const { return pinned_; }
  // Shared per-device act/out workspaces (replaces per-shard act/out USM).
  // A shard's act/out buffers are written+consumed within one run_linear_tp
  // call and never live across calls (single-threaded CLI), so one
  // (maxM × maxK / maxM × maxNl) buffer pair per device serves all shards.
  // This kills ~2000 live USM allocs on 27B-sc (~460MB L0 overhead) — the
  // margin that makes full residency + lm_head fit the 12GB budget.
  void ensure_workspace(uint32_t M, uint32_t K, uint32_t Nl_max);
  // Free one tensor's device shard (no-op if not built).
  void release_shard(int tensor_index);
  // Layer index parsed from a tensor name ("...layers.<L>..."), or -1 when
  // the tensor is not per-layer (lm_head, norms).
  static int layer_of_name(const std::string& name);
  
  // SYCL queue for device i (for attention kernels etc.).
  sycl::queue& device_queue(size_t i) { return devices_.at(i).queue; }
  // Free VRAM bytes currently visible on device i (sysman zesMemoryGetState,
  // matched by native-handle UUID; 0 when unavailable). Used to size the
  // asymmetric lm_head TP split after preload_all() — never assume index
  // equality across APIs (sysman enumerates in reverse vs SYCL/xpu-smi).
  uint64_t device_free_bytes(size_t i);


 private:
  Engine() = default;
 public:
  // Releases the borrowed mmap backing (if any). Defined in engine.cpp.
  ~Engine();
 private:
  void build_shards();

  ModelIR model_;
  std::vector<intfer::device::DeviceContext> devices_;
  // Borrowed mmap mapping backing model_ when loaded via deserialize_borrowed
  // (zero-copy for multi-GB files). Must outlive model_. Empty when the model
  // was deserialized from an owned blob (small files).
  struct MappedFile {
    void* base = nullptr;
    size_t size = 0;
  };
  MappedFile backing_;

  // One contiguous shard range [start,end) of the output dim N per device, for
  // each quantized Linear tensor (indexed parallel to model_.tensors).
  struct TensorShards {
    std::vector<std::pair<uint32_t, uint32_t>> ranges;  // per device
  };
  std::vector<TensorShards> shards_;

  std::vector<std::string> layer_names_;  // "model.language_model.layers.<L>"

  RotationSet rs_;
  std::map<uint8_t, const Rotation*> rid_;
  std::map<std::string, std::vector<float>> lang_;  // safetensors weights (CPU)

  // Cached tensor indices for the decode loop (attn q/k/v/o, mlp gate/up/down per layer)
  std::vector<int> tq_, tk_, tv_, to_, tg_, tu_, td_;
  std::vector<bool> is_full_;

  // -----------------------------------------------------------------------
  // Per-tensor shard cache (used by the per-token `run_linear_tp` fast path).
  // Built lazily on the first call to run_linear_tp for a given tensor_index.
  // Eliminates the per-step unpack+repack+SYCL buffer construction overhead
  // (the previous slow path was O(N*K) host work + host->device copy every
  // call, dominating per-token latency).
  // -----------------------------------------------------------------------
  struct CachedShard {
    bool built = false;
    gemm::WeightScheme scheme = gemm::WeightScheme::INT8;
    gemm::ActScheme act_scheme = gemm::ActScheme::FP16;
    uint32_t group_size = 0;
    bool has_global_divisor = false;
    bool use_dpas = true;  // TODO 6: enable XMX DPAS path when eligible
    uint32_t N_local = 0;   // N rows on this device (post-shard)
    uint32_t K = 0;
    // *** INTFER_MLP_PARTIAL_DOWN (2026-10-14): K-split full-N shard. ***
    // The down-projection at TP=2 cannot fuse the way run_mlp_fused (TP=1)
    // does: gate/up are column-parallel, so device d owns only its slice of
    // the inter, while down has K = the FULL inter and therefore needs the
    // other device's half too. There is no P2P on this box (grep returns
    // nothing), so the intermediate must cross the host -- that is the
    // structural limit recorded in AGENTS.md, and it is real.
    //
    // The workaround is to split the OTHER dimension. Instead of device d
    // owning rows [H*d/2, H*(d+1)/2) x full K (the column-split shard), give
    // it the full N (5120 rows) x a K-slice (8704 cols) shard: device d
    // computes partial_d[n] = sum_{k in K-slice d} W[n,k]*act[k] for ALL n,
    // and the host sums partial_0 + partial_1 in FP32. Same total bytes as
    // the column-split shard (5120*8704 == 2560*17408), g16-aligned on the
    // 27B (K/2 = 8704 = 544 groups of 16), so scales/codes pack exactly as
    // shard_weight already does. The shard is built by a STRIDED pack from
    // the model tensor (shard_weight's k_off parameter) -- NOT by slicing the
    // column-split shard, which would be rows [2560d,2560(d+1)) x K-slice d
    // and would never compute the cross terms device 0's n-range needs.
    //
    // ksplit_code_usm[d] holds K-slice d's shard ON device d (USM is
    // per-device; device d reads its own K-slice from its own buffer). The
    // host cache ksplit_code_host[d] is the strided payload for K-slice d.
    bool ksplit = false;
    uint32_t ksplit_K = 0;             // K per ksplit shard (= t.K / n_devices)
    std::vector<std::uint8_t*>  ksplit_code_usm;   // [device]
    std::vector<std::uint16_t*> ksplit_scale_usm;  // [device]
    std::vector<std::vector<std::uint8_t>>  ksplit_code_host;   // [device]
    std::vector<std::vector<std::uint16_t>> ksplit_scale_host;  // [device]
    // Per-device persistent USM pointers (size = devices_.size()).
    std::vector<std::uint8_t*>  code_usm;   // packed codes (FP16: 2*N*K bytes; INT*: packed)
    std::vector<std::uint16_t*> scale_usm;  // FP16 scale broadcast, N_local*n_groups
    // NOTE: no global_usm — the alpha divisor already lives on the host
    // (global_host) and the kernel takes it by value, so a 4-byte device
    // alloc per shard per rank (~400 USM allocs of pure overhead) is waste.
    // NOTE: no per-shard act/out — run_linear_tp uses the shared per-device
    // workspaces (ws_): one (M x K / M x Nl) pair per device serves all
    // shards, killing ~2000 live USM allocs (~460MB L0 overhead on 27B-sc).
    // SYCL event tracking the latest GEMM launch (so host can wait on it).
    sycl::event last_kernel_event;
    // Phase 2: repack-once host cache. shard_weight() output is immutable
    // per (tensor, TP rank); retaining it makes evict+rebuild memcpy-speed
    // (no scalar unpack repetition). ~8.8GB for the 27B INT4 set; populated
    // on first build, preserved across release_shard(). Bit-identical: the
    // same bytes feed the same USM buffers.
    std::vector<std::vector<std::uint8_t>> code_host;    // per device
    std::vector<std::vector<std::uint16_t>> scale_host;  // per device
    float global_host = 0.0f;
    bool has_host_cache = false;
    // TODO 6 (W4A8/W2A8): DPAS fast path. When dpas48 is true, code_usm[d]
    // holds the repacked vertical-B tiles (repack_dpas412 layout, NOT the
    // row-major packed codes) and dpas_bytes[d] is the USM allocation size
    // (release_shard pools by exact size). Act quant + scales live in the
    // shared workspaces (ws_act48_/ws_as48_), sized by ensure_workspace48.
    bool dpas48 = false;
    std::vector<size_t> dpas_bytes;  // per device USM bytes for code_usm
  };
  std::vector<CachedShard> shard_cache_;
  bool ensure_shard_built(int tensor_index);
  // Phase 3: shape-keyed USM pool. Layer streaming frees + reallocs the same
  // few buffer shapes every layer (INT4 gate/up/down per rank); parking freed
  // buffers here turns rebuild into acquire + memcpy (no L0 alloc, no page
  // re-pinning). Keyed by byte size; per device; u8-typed (aliased by callers).
  // Correctness: buffers re-enter the pool only via release_shard(), which
  // runs after run_linear_tp's event waits — no kernel can reference them.
  std::vector<std::map<size_t, std::vector<void*>>> usm_pool_;  // [device][bytes]
  void* pool_acquire(size_t device, size_t bytes);
  void pool_release(size_t device, void* ptr, size_t bytes);
  uint64_t pool_acquires_ = 0, pool_hits_ = 0;  // allocator audit counters
  // Reusable FP16 activation staging for run_linear_tp (M*K). Resized only
  // when the shape grows: steady-state zero mallocs per GEMM call.
  std::vector<std::uint16_t> act_staging_;
  // Reusable FP32 staging for run_mlp_partial_down's host partial-sum gather
  // (M*td.N). The gather converts each device's FP16 partial with
  // half_to_float_batch (the DEFAULT readback path's converter, engine.cpp
  // :1346 etc. — F16C cvtph is exact, so the bytes are identical to the
  // scalar per-element form it replaced, which was the last scalar
  // half_to_float site in the tree). Resized only when the shape grows:
  // steady-state zero mallocs per MLP call.
  std::vector<float> pd_gather_;
  // Reusable staging that preserves ONE (pointer, bytes) input pair across
  // run_linear_wave's Phase-1 dedup. The local it replaces was a fresh
  // std::vector per call, so every wave call with a reused input paid a
  // heap malloc of M*K halves (~1 MB on 27B prefill chunks — above glibc's
  // mmap threshold, so mmap+munmap plus first-touch page faults on every
  // call). Resized only when the shape grows. Staleness: content is read
  // only under `dup`, and `shared_valid` is false at every call start, so
  // a previous call's bytes can never be admitted as a dup.
  std::vector<std::uint16_t> wave_shared_staging_;
  // INTFER_FUSE_ACT (default OFF). Identity of the FP16 activation currently
  // sitting in ws_[d].act for EVERY device: the host vector pointer it was
  // converted from, plus the shape it was converted for.
  //
  // *** THE POINTER IS NOT AN IDENTITY, AND THE FIRST VERSION OF THIS WAS A
  // *** SHIPPED STREAM-DIVERGING BUG (found 2026-10-05, gate test_gpu_optin.sh).
  // The original comment here argued the cache was safe because ws_[d].act has
  // exactly three classes of writer, two of which write DIFFERENT buffers, and
  // listed "(b) a different input vector" without ever asking how a different
  // vector is DETECTED. It is not. run_linear_tp takes
  // `const std::vector<float>& x` and a std::vector that is assigned into keeps
  // its heap buffer, so the CLI's per-step hidden vector arrives at the SAME
  // x.data() every decode step with DIFFERENT contents. M and K are constant too.
  // The key therefore matched from step 2 onward, the convert and the upload
  // were skipped, and the GEMM read the previous step's FP16 bytes.
  // Measured: 0.8B TP=1, -n4, base '369 264 3177 13' vs flag
  // '11 11 40252 306'. A repeated '11 11' is what stale workspace looks like --
  // not a subtly different arithmetic result.
  //
  // The lesson generalises past this flag: an argument that enumerates the
  // WRITERS of a destination and never examines the SOURCES feeding the lookup
  // has not closed the case, and check_act_cache_invalidation.sh could not see
  // it either because the defect is not a missing invalidation -- it is a lookup
  // that cannot fail.
  const float* act_cache_ptr_ = nullptr;
  uint32_t act_cache_M_ = 0;
  uint32_t act_cache_K_ = 0;
  // The exact FP32 bytes that produced act_staging_. A hit is admitted only if
  // x still equals these, so the key is CONTENT and the guarantee holds for any
  // caller that refills its buffer in place -- no invalidation site to forget.
  // memcmp is the cheap half of the comparison it replaces: one vectorised,
  // early-exit pass over contiguous bytes against a branchy scalar
  // float->half convert, and it decides the same thing exactly. A sampled
  // fingerprint would be cheaper and would be a probabilistic cache, which is a
  // different thing than the bit-exact one the flag claims.
  std::vector<float> act_src_;
  // Monotonic generation of act_staging_, bumped by EVERY writer of it (three
  // sites: this path, run_linear_tp_devact's wave phase, and run_linear_f16_tp's
  // shared-staging phase). The cache records the generation it converted at, so
  // a hit also requires that nothing has rewritten the staging since.
  //
  // This is the SECOND instance of the same family and it was latent, not
  // observed: at TP=1 with the fused MLP ON, the down projection runs through
  // run_linear_tp_devact, which rewrites act_staging_ from a different source.
  // Today that is masked because the next call always misses on its content, but
  // the content check is only half the invariant -- the cache claims to know what
  // act_staging_ holds, and act_staging_ is shared with two other writers. A
  // cache that caches a buffer without tracking that buffer's writers is the same
  // defect one level down.
  uint64_t act_staging_gen_ = 0;
  uint64_t act_cache_gen_ = 0;
  // Reusable INT8-act + scale staging for run_linear_tp_dpas48 (M*K int8,
  // M*nGA fp32, tight rows). The quant writes here (host memory), then one
  // memcpy per workspace row lands wide vectorized stores into the shared
  // ws48_ buffers — scalar byte-stores straight into malloc_shared cost
  // ~100x (each store its own transaction: copy bucket 96ms -> ~25ms).
  std::vector<std::int8_t> act48_staging_;
  std::vector<float> as48_staging_;
  // Host-side GA64 (scale, inverse) pairs (2*M*nGA) for the device act-quant
  // pre-pass. Computed here, on the reference side, and shipped to the device
  // so the pre-pass never divides or re-reduces (see
  // ActQuantDevParams::scale_inv_in).
  std::vector<float> si48_staging_;
  // Shared per-device act/out workspaces (replaces per-shard act/out USM).
  // One buffer pair per device serves all shards: a shard's act/out are
  // written+consumed within a single run_linear_tp call and never live
  // across calls (single-threaded CLI), so aliasing is safe. Grow-only;
  // steady-state zero allocs. Direct malloc_shared (NOT the shape pool —
  // pool entries stay live and would keep the old size resident).
  struct Workspace {
    std::uint16_t* act = nullptr;  // M * K FP16
    std::uint16_t* out = nullptr;  // M * Nl FP16
    // S1 (uncached host-USM output, INTFER_UNCACHED_HOST_USM): true when `out`
    // was allocated with zeMemAllocHost + ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED
    // instead of sycl::malloc_shared, in which case it is released with
    // zeMemFree rather than sycl::free. `act` NEVER carries this flag -- see
    // the direction constraint in the engine.cpp helper.
    bool out_uncached = false;
    uint32_t M = 0, K = 0, Nl = 0;
  };
  std::vector<Workspace> ws_;
  // Wave workspaces for run_linear_wave: one (M*K / M*Nl) slot pair per
  // live wave item per device, so independent GEMMs can be converted,
  // submitted back-to-back, and read back without aliasing. Same
  // grow-only single-threaded contract as ws_: slots are written+consumed
  // within one run_linear_wave call. Sized for the max wave width seen
  // (steady state: LA qkv/z/b/a = 4 slots).
  struct WaveSlot {
    std::uint16_t* act = nullptr;
    std::uint16_t* out = nullptr;
    // S1 provenance, same contract as Workspace::out_uncached.
    bool out_uncached = false;
  };
  std::vector<WaveSlot> wv_;  // [device * wv_W_ + slot]
  uint32_t wv_M_ = 0, wv_K_ = 0, wv_Nl_ = 0, wv_W_ = 0;
  void ensure_wave_workspace(uint32_t M, uint32_t W, uint32_t K, uint32_t Nl);
  // TODO 6 (W4A8/W2A8): per-device INT8-act + act-scale workspaces for the
  // DPAS path (M*K int8, M*nGA fp32). Same grow-only aliasing contract as
  // ws_: written+consumed within one run_linear_tp call.
  struct Workspace48 {
    std::int8_t* act = nullptr;  // M * K INT8
    float* as = nullptr;         // M * nGA FP32 act scales (nGA = max over K)
    // FP32 act landing zone for the device act-quant pre-pass
    // (INTFER_DPAS_PREQ): per-device because every rank has its own
    // sycl::context. The H2D is skipped when the caller's buffer already
    // aliases this one (devact callers); with the current const-vector
    // signature the copy is unconditional.
    float* f32 = nullptr;        // M * K FP32
    // Host-computed GA64 (scale, inverse) pairs for the device act-quant
    // pre-pass (2 * M * nGA FP32, tight rows). Sent from the host so the
    // device neither divides nor re-reduces max-abs: see
    // ActQuantDevParams::scale_inv_in for why the reciprocal must come from
    // the host.
    float* si = nullptr;
    // INTFER_DPAS48_FOLD: per-device landing zone for the DPAS row-fold
    // pre-pass — 2 * M * nOp INT32 (S0 and Sfull per (row, op), op = 32).
    // Tiny (K/32 entries), grow-only, same contract as the buffers above.
    // Per-device for the same reason as the rest: each rank owns its own
    // sycl::context, so rank 0's queue cannot legally write rank 1's USM.
    std::int32_t* fold = nullptr;
    uint32_t M = 0, K = 0, nGA = 0;
    uint32_t nOp = 0;  // max K/32 seen (DPAS row-fold stride)
  };
  std::vector<Workspace48> ws48_;
  void ensure_workspace48(uint32_t M, uint32_t K);
  void run_linear_tp_dpas48(int tensor_index, const std::vector<float>& x,
                            std::vector<float>& out, uint32_t M);
  // Per-layer tensor index lists, built by build_shards() from tensor names.
  // layer_tensors_[L] = indices of "...layers.<L>..." tensors; layer_of_[i]
  // mirrors it per tensor index (-1 = unlayered: lm_head, norms, ...).
  // pinned_ (set by preload_all()) disables all eviction: begin_layer() and
  // release_shard() early-return, keeping every shard resident.
  // skip_host_cache_ (set by preload_all()): with eviction disabled no
  // rebuild can ever need the INT4 repack cache, so ensure_shard_built()
  // skips populating it (~9.4GB host RAM on 27B-sc). Streaming mode keeps
  // the cache (rebuilds need it).
  std::vector<std::vector<int>> layer_tensors_;
  std::vector<int> layer_of_;
  bool pinned_ = false;
  bool skip_host_cache_ = false;
};


}  // namespace intfer

#ifdef INTFER_PROFILE_GPU
// GPU profiling helpers (enabled via -DINTFER_PROFILE_GPU).
// Device-timeline marker API (INTFER_TIMELINE=1): submit a 1-work-item
// empty kernel on `q` and record its event, so the gap between consecutive
// markers measures the device time of the region between them without
// trusting per-event timestamps. See engine.cpp:timeline_mark.
namespace intfer {
  void timeline_mark(sycl::queue& q, const char* tag);
  void timeline_dump();
}
struct GpuProfileSnapshot {
  // Aggregate accumulated times (host-side chrono + device timestamps where available).
  double kernel_us;
  double wait_us;
  double copy_us;
  double readback_us;
  double usm_write_us;
  double submit_us;
  double waitblock_us;
  double rebuild_us;
  uint64_t rebuild_count;
  // Peak instantaneous queue depth observed during `run_linear_tp`.
  uint64_t peak_wait_depth;
  // Counters.
  uint64_t kernel_count, wait_count, copy_count, readback_count;
  // Per-phase launch/transfer counts (host USM activation writes, q.submit
  // host overhead calls). These are what make the fusion's "one submit +
  // one wait + one D2H readback saved" claim MEASURABLE rather than prose:
  // the wave path emits one usm_write + one submit + one readback PER
  // GEMM, the fused path emits them once for both.
  uint64_t submit_count, usm_write_count;
  // Device-timeline markers (INTFER_TIMELINE=1): a 1-work-item empty kernel
  // per region. On an in-order queue the gap between marker i's end and
  // marker i+1's start IS the device time of everything between them, with
  // no per-launch instrumentation and no dependence on
  // get_profiling_info<command_start/command_end> returning sane durations.
  // This exists because the driver records submit==start==end for real
  // kernels on this box (see AGENTS.md "The profiling defect"), so the
  // per-event kernel bucket is unmeasurable and the only way to attribute
  // wall-clock to a region is a marker pair. See engine.cpp:timeline_mark.
  uint64_t timeline_marker_count;
  // Matrix-engine (XMX joint_matrix) launch accounting (2026-10-19), so the
  // "the matrix engines are idle on the hot path" verdict is RE-DERIVABLE from
  // a run's own counter dump instead of remembered. These are ENGINE-SIDE
  // dispatch observations, NOT hardware XVE counters: the XMX kernel writes
  // its own `ran` verdict, so a declined attempt is counted separately and the
  // triple below distinguishes three states that a bare launch count conflates
  // -- "never requested" (no attempt), "requested but every tile declined"
  // (attempted, nothing submitted), and "actually ran". The hardware counters
  // (XVE_INST_EXECUTED_XMX_*) stay UNKNOWN here: xmxmon is an external Level
  // Zero metric streamer and this box runs dev.xe.observation_paranoid=1, so
  // scripts/xmx_profile.sh is the path that joins them, not the engine.
  uint64_t xmx_attempt_count, xmx_launch_count, xmx_decline_count;
};
namespace intfer {
  GpuProfileSnapshot get_gpu_profile();
  void reset_gpu_profile();
  // One-line matrix-engine attribution for the [GPU profile] dump, printed
  // beside nk/nw/nr. It ALWAYS names the absence of the hardware counters
  // rather than printing a zero that reads as a measurement -- see the field
  // comment above.
  std::string xmx_attrib_line(const GpuProfileSnapshot& p);
}
#endif
