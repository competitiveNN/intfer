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

  // --- Full hybrid forward (Stage-2 integration) -----------------------------
  // Get the model IR (readonly).
  const ModelIR& model() const { return model_; }
  
  // Load the non-quantized language weights (embed, norms, SSM) from safetensors.
  bool load_language_weights(const std::string& safetensors_path);
  
  // Run ONE quantized Linear tensor-parallel across the fleet.
  void run_linear_tp(int tensor_index, const std::vector<float>& x,
                     std::vector<float>& out);
  
  // Which Rotation (SpinQuant) belongs to tensor ti (by its rotation_id).
  const Rotation* rotation_for(int tensor_index) const;
  
  // Convenience: run a Linear TP GEMM and return the output vector.
  std::vector<float> run_linear_tp_ret(int tensor_index, const std::vector<float>& x) {
    std::vector<float> out; run_linear_tp(tensor_index, x, out); return out;
  }
  
  // SYCL queue for device i (for attention kernels etc.).
  sycl::queue& device_queue(size_t i) { return devices_.at(i).queue; }


 private:
  Engine() = default;
  void build_shards();

  ModelIR model_;
  std::vector<intfer::device::DeviceContext> devices_;

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
    uint32_t group_size = 0;
    bool has_global_divisor = false;
    uint32_t N_local = 0;   // N rows on this device (post-shard)
    uint32_t K = 0;
    // Per-device persistent USM pointers (size = devices_.size()).
    std::vector<std::uint8_t*>  code_usm;   // packed codes (FP16: 2*N*K bytes; INT*: packed)
    std::vector<std::uint16_t*> scale_usm;  // FP16 scale broadcast, N_local*n_groups
    std::vector<float*>         global_usm; // 1 element (alpha)
    // Per-device persistent USM act / output buffers.
    std::vector<std::uint16_t*> act_usm;    // size = K
    std::vector<std::uint16_t*> out_usm;    // size = N_local (FP16 gemm output)
    // SYCL event tracking the latest GEMM launch (so host can wait on it).
    sycl::event last_kernel_event;
  };
  std::vector<CachedShard> shard_cache_;
  bool ensure_shard_built(int tensor_index);

  // Command-graph capture state (experimental; requires INTFER_GRAPH_CAPTURE=1).
#if INTFER_GRAPH_CAPTURE
  struct DeviceUSM {
    sycl::queue q;
    std::vector<std::uint8_t> code_store;
    std::vector<std::uint16_t> scale_store;
    std::vector<float> global_store;
    std::vector<std::uint16_t> act_store;
    std::vector<std::uint16_t> out_store;
    std::vector<std::uint16_t> c_store;     // persistent output accumulator
    // SYCL buffers — wrapped in std::optional so DeviceUSM is default-constructible
    std::optional<sycl::buffer<std::uint8_t>> code;
    std::optional<sycl::buffer<std::uint16_t>> scale;
    std::optional<sycl::buffer<float>> global;
    std::optional<sycl::buffer<std::uint16_t>> act;
    std::optional<sycl::buffer<std::uint16_t>> out;
    std::optional<sycl::buffer<std::uint16_t>> c;
    // compiled executable graph
    std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>> graph;
  };
  std::vector<DeviceUSM> usm_;
  bool graph_built_ = false;
  bool build_usm_buffers_for_tensor(int tensor_index);
  void build_decode_graph();
#endif
};


}  // namespace intfer

#ifdef INTFER_PROFILE_GPU
// GPU profiling helpers (enabled via -DINTFER_PROFILE_GPU).
struct GpuProfileSnapshot {
  double kernel_us, wait_us, copy_us, readback_us;
  uint64_t kernel_count, wait_count, copy_count, readback_count;
};
namespace intfer {
  GpuProfileSnapshot get_gpu_profile();
  void reset_gpu_profile();
}
#endif
