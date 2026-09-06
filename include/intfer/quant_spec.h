// include/intfer/quant_spec.h
// Authoritative Tensor-IR contract for Stage 1 (t0 -> t3/t4/t5/t6).
// Consumed by: t3 SpinQuant, t4 DASH-Q, t5 RCO, t6 pipeline/output.
// Model-agnostic; applies to any HF transformer with Linear attn/MLP.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace intfer {

// ---- Enumerations (see docs/quant_spec.md §6.1) -------------------------
enum class QuantScheme : uint8_t { FP16 = 0, INT8 = 1, INT4 = 2, INT2 = 3 };

enum class TensorRole : uint8_t {
  EMBED = 0, LM_HEAD = 1,
  ATTN_Q = 2, ATTN_K = 3, ATTN_V = 4, ATTN_O = 5,
  MLP_GATE = 6, MLP_UP = 7, MLP_DOWN = 8,
  NORM = 9, ROPE = 10,
  // linear_attn (SSM) — Qwen3.5 hybrid model. Mamba-style tensors have
  // unique shapes (no attention QKV split), so we map them to dedicated
  // roles so DASH-Q can target them with the same INT2/INT4/INT8
  // candidate ladder as self_attn/MLP. They get rotated at the same
  // sites (QKV/FFN_IN/O) so the Stage-2 engine can apply the matching
  // online replay.
  LINEAR_ATTN_QKV = 11,   // in_proj_qkv: H -> conv_dim
  LINEAR_ATTN_Z   = 12,   // in_proj_z:   H -> n_v*v_head_dim
  LINEAR_ATTN_B   = 13,   // in_proj_b:   H -> n_v_heads
  LINEAR_ATTN_A   = 14,   // in_proj_a:   H -> n_v_heads
  LINEAR_ATTN_CONV= 15,   // conv1d:      (conv_k, conv_dim)
  LINEAR_ATTN_OUT = 16,   // out_proj:    n_v*v_head_dim -> H
  OTHER = 17
};

enum class ActScheme : uint8_t { FP16 = 0, BF16 = 1, INT8_G64 = 2 };

enum class ScaleDtype : uint8_t { FP16 = 0, FP8_E4M3 = 1 };

// ---- QuantConfig (docs/quant_spec.md §6.2) ------------------------------
struct QuantConfig {
  QuantScheme scheme = QuantScheme::FP16;
  uint16_t group_size = 0;        // K-group for scales (0 = per-tensor)
  ScaleDtype scale_dtype = ScaleDtype::FP16;
  bool has_global_divisor = false; // emit per-tensor FP32 alpha
  ActScheme act = ActScheme::FP16;
  uint8_t rotation_id = 0;         // SpinQuant R applied (t3 sets, 0 = none)
};

// ---- TensorIR (docs/quant_spec.md §6.3) ---------------------------------
struct TensorIR {
  std::string name;                // HF key, e.g. model.layers.0.self_attn.q_proj.weight
  TensorRole role = TensorRole::OTHER;
  uint32_t N = 0;                  // logical rows
  uint32_t K = 0;                  // logical cols (contraction dim, padded %64 / %128)
  QuantConfig qcfg;                // chosen config (t5 finalizes)

  uint64_t code_offset = 0;        // slice in ModelIR::code_buf
  uint64_t code_bytes = 0;
  uint64_t scale_offset = 0;       // slice in ModelIR::scale_buf
  uint64_t scale_bytes = 0;
  uint64_t global_offset = 0;      // slice in ModelIR::global_buf (0 = none)
  uint64_t global_bytes = 0;

  // FP16 weight storage used by the SpinQuant preprocessing stage (t3) and
  // consumed by DASH-Q (t4). Empty for already-quantized tensors.
  uint64_t fp16_offset = 0;       // slice in ModelIR::fp16_buf
  uint64_t fp16_bytes = 0;        // bytes (== N*K*2)

  float est_error = 0.0f;          // DASH-Q rel-L2 reconstruction error (t4)
  std::vector<TensorIR> candidates; // DASH-Q variant list for RCO (t4 -> t5)

  // Convenience: bytes this tensor contributes to the weight budget.
  uint64_t weight_bytes() const { return code_bytes + scale_bytes + global_bytes; }
  // Effective bits/weight incl. scale overhead (docs §3.3).
  float eff_bits_per_weight() const;
};

// ---- ModelIR : the single interchange object for t3->t4->t5->t6 ---------
struct ModelIR {
  std::string model_id = "Qwen3-32B"; // primary target (docs §2)
  uint32_t n_layers = 0;
  std::vector<TensorIR> tensors;

  // Packed payload buffers (offsets above index into these).
  std::vector<uint8_t> code_buf;    // INT2/4/8 packed codes
  std::vector<uint8_t> scale_buf;   // per-group FP16 / FP8 scales
  std::vector<uint8_t> global_buf;  // per-tensor FP32 global divisors (alpha)
  std::vector<uint16_t> fp16_buf;   // FP16 (binary16) weights, used by t3/t4 stage

  uint64_t total_weight_bytes() const; // sum of tensor.weight_bytes()
};

// ---- Helpers ------------------------------------------------------------
// Effective bits/weight for a config (docs §3.3 table).
float effective_bits(QuantScheme scheme, uint16_t group_size, ScaleDtype sd);

// Default candidate set emitted by DASH-Q (t4) for a given role (docs §5).
std::vector<QuantConfig> default_candidates(TensorRole role);

} // namespace intfer
