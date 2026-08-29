// src/quant_spec.cpp
// Definitions for the Stage-1 Tensor-IR contract (docs/quant_spec.md).
#include "intfer/quant_spec.h"

#include <cmath>

namespace intfer {

float TensorIR::eff_bits_per_weight() const {
  return effective_bits(qcfg.scheme, qcfg.group_size, qcfg.scale_dtype);
}

uint64_t ModelIR::total_weight_bytes() const {
  uint64_t s = 0;
  for (const auto& t : tensors) s += t.weight_bytes();
  return s;
}

float effective_bits(QuantScheme scheme, uint16_t group_size, ScaleDtype sd) {
  // code bits + scale overhead (docs/quant_spec.md §3.3).
  float code_bits = 0.0f;
  switch (scheme) {
    case QuantScheme::FP16: code_bits = 16.0f; break;
    case QuantScheme::INT8: code_bits = 8.0f;  break;
    case QuantScheme::INT4: code_bits = 4.0f;  break;
    case QuantScheme::INT2: code_bits = 2.0f;  break;
  }
  if (scheme == QuantScheme::FP16) return 16.0f;  // scales are never stored for FP16

  // scale bits per weight = scale_bits / group_size.
  float scale_bits = (sd == ScaleDtype::FP8_E4M3) ? 8.0f : 16.0f;
  float g = static_cast<float>(group_size > 0 ? group_size : 1);
  return code_bits + scale_bits / g;
}

std::vector<QuantConfig> default_candidates(TensorRole role) {
  // Per docs/quant_spec.md §5 bit-allocation policy.
  auto mk = [](QuantScheme s, uint16_t g, ActScheme act,
               bool global = true, ScaleDtype sd = ScaleDtype::FP16) {
    QuantConfig c;
    c.scheme = s;
    c.group_size = g;
    c.scale_dtype = sd;
    c.has_global_divisor = global;
    c.act = act;
    return c;
  };
  const ActScheme a8 = ActScheme::INT8_G64;
  switch (role) {
    case TensorRole::EMBED:
    case TensorRole::LM_HEAD: {
      // Floor >= INT4 (sensitive). Offer INT4/INT8/FP16.
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
    case TensorRole::ATTN_Q:
    case TensorRole::ATTN_K:
    case TensorRole::ATTN_V: {
      // first layer forced >= INT4 by RCO; candidates include INT2/INT4/INT8/FP16.
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT2, 16, a8));
      v.push_back(mk(QuantScheme::INT2, 32, a8));
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
    case TensorRole::ATTN_O:
    case TensorRole::MLP_GATE:
    case TensorRole::MLP_UP:
    case TensorRole::MLP_DOWN: {
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT2, 16, a8));
      v.push_back(mk(QuantScheme::INT2, 32, a8));
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      return v;
    }
    case TensorRole::NORM:
    case TensorRole::ROPE:
    default: {
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
  }
}

} // namespace intfer
