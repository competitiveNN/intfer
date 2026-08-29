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
#include <memory>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include "intfer/quant_spec.h"
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
  std::vector<float> decode_step(const std::vector<float>& token_hidden);

  // Build + finalize the per-token command graph (call once after load).
  void capture_decode_graph();

  const ModelIR& model() const { return model_; }

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

  bool graph_captured_ = false;
  std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::executable>> graph_;
};

}  // namespace intfer
