// src/quant/quant_spec.cpp
// Definitions for the Tensor-IR contract declared in include/intfer/quant_spec.h
// (t0). Implements effective-bits math (docs quant_spec.md §3.3), the per-role
// default DASH-Q candidate set (§5), and the ModelIR/TensorIR size helpers.
#include "intfer/quant_spec.h"

namespace intfer {

float effective_bits(QuantScheme scheme, uint16_t group_size, ScaleDtype sd) {
  if (scheme == QuantScheme::FP16) return 16.0f;
  const float code_bits = (scheme == QuantScheme::INT2) ? 2.0f
                        : (scheme == QuantScheme::INT4) ? 4.0f : 8.0f;
  const float scale_bits = (sd == ScaleDtype::FP16) ? 16.0f : 8.0f; // FP8 variant = 0.5 bit/w saved
  const float g = group_size > 0 ? static_cast<float>(group_size) : 1.0f;
  return code_bits + scale_bits / g;
}

float TensorIR::eff_bits_per_weight() const {
  return effective_bits(qcfg.scheme, qcfg.group_size, qcfg.scale_dtype);
}

uint64_t ModelIR::total_weight_bytes() const {
  uint64_t s = 0;
  for (const auto& t : tensors) s += t.weight_bytes();
  return s;
}

std::vector<QuantConfig> default_candidates(TensorRole role) {
  std::vector<QuantConfig> out;
  auto mk = [](QuantScheme sch, uint16_t g, bool glob = true,
               ActScheme act = ActScheme::INT8_G64) {
    QuantConfig c;
    c.scheme = sch;
    c.group_size = g;
    c.scale_dtype = ScaleDtype::FP16;
    c.has_global_divisor = glob;
    c.act = act;
    c.rotation_id = 0;
    return c;
  };
  switch (role) {
    case TensorRole::EMBED:
    case TensorRole::LM_HEAD:
      out.push_back(mk(QuantScheme::INT4, 64));
      out.push_back(mk(QuantScheme::INT8, 32));
      out.push_back(mk(QuantScheme::FP16, 0, false, ActScheme::FP16));
      break;
    case TensorRole::ATTN_Q:
    case TensorRole::ATTN_K:
    case TensorRole::ATTN_V:
    case TensorRole::ATTN_O:
    case TensorRole::MLP_GATE:
    case TensorRole::MLP_UP:
    case TensorRole::MLP_DOWN:
    case TensorRole::LINEAR_ATTN_QKV:
    case TensorRole::LINEAR_ATTN_Z:
    case TensorRole::LINEAR_ATTN_B:
    case TensorRole::LINEAR_ATTN_A:
    case TensorRole::LINEAR_ATTN_OUT:
      // Bulk of params -> INT2 target; FP16 escape at the end.
      // LINEAR_ATTN_B and _A have tiny N (n_v_heads ~16-64) and
      // unrotated scalars, so we keep them at INT4/INT8 floor; DASH-Q
      // will still try the lower bit-widths in the candidate set.
      out.push_back(mk(QuantScheme::INT2, 16));
      out.push_back(mk(QuantScheme::INT2, 32));
      out.push_back(mk(QuantScheme::INT4, 64));
      out.push_back(mk(QuantScheme::INT8, 32));
      out.push_back(mk(QuantScheme::FP16, 0, false, ActScheme::FP16));
      break;
    case TensorRole::NORM:
    case TensorRole::ROPE:
    default:
      // Elementwise / precision-critical -> never quantized.
      out.push_back(mk(QuantScheme::FP16, 0, false, ActScheme::FP16));
      break;
  }
  return out;
}

} // namespace intfer
