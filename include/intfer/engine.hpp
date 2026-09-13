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
  
  // FP16 x FP16 GEMM for non-quantized final heads (lm_head/embed).
  // C[1,N] = x[1,K] * W[N,K]^T. All row-major. Outputs FP32 logits.
  void run_linear_f16_tp(const sycl::half* x, const sycl::half* w,
                         float* out, uint32_t N, uint32_t K,
                         const sycl::event& copy_dep = {});
  
  // Which Rotation (SpinQuant) belongs to tensor ti (by its rotation_id).
  const Rotation* rotation_for(int tensor_index) const;
  
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
  // Shared per-device act/out workspaces (replaces per-shard act/out USM).
  // One buffer pair per device serves all shards: a shard's act/out are
  // written+consumed within a single run_linear_tp call and never live
  // across calls (single-threaded CLI), so aliasing is safe. Grow-only;
  // steady-state zero allocs. Direct malloc_shared (NOT the shape pool —
  // pool entries stay live and would keep the old size resident).
  struct Workspace {
    std::uint16_t* act = nullptr;  // M * K FP16
    std::uint16_t* out = nullptr;  // M * Nl FP16
    uint32_t M = 0, K = 0, Nl = 0;
  };
  std::vector<Workspace> ws_;
  // TODO 6 (W4A8/W2A8): per-device INT8-act + act-scale workspaces for the
  // DPAS path (M*K int8, M*nGA fp32). Same grow-only aliasing contract as
  // ws_: written+consumed within one run_linear_tp call.
  struct Workspace48 {
    std::int8_t* act = nullptr;  // M * K INT8
    float* as = nullptr;         // M * nGA FP32 act scales (nGA = max over K)
    uint32_t M = 0, K = 0, nGA = 0;
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
};
namespace intfer {
  GpuProfileSnapshot get_gpu_profile();
  void reset_gpu_profile();
}
#endif
