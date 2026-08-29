// include/intfer/quant/dashq.hpp
// DASH-Q: Diagonal-Hessian & Iterative Least-Squares PTQ solver (Stage 1, step 2).
//
// Core idea (docs/quant_spec.md §3, AGENTS.md Stage 1 [2]): instead of GPTQ's
// full off-diagonal Hessian — which is corrupted by sampling noise at
// ultra-low bits — DASH-Q keeps ONLY the diagonal H_ii. This decouples
// per-group quantization into INDEPENDENT weighted least-squares problems:
//
//     minimize_{s, codes}  Σ_i  H_ii · (w_i − s · c_{a_i})²
//
// solved by alternating (coordinate descent):
//   * assignment:  a_i = argmin_j  (w_i − s·c_j)²          (H_ii > 0 factors out)
//   * scale (LS):  s   = Σ_i H_ii·w_i·c_a_i / Σ_i H_ii·c_a_i²
//   * codebook:    c_j = Σ_{i:a_i=j} H_ii·w_i / (s·Σ_{i:a_i=j} H_ii)   [learned variant]
//
// The diagonal Hessian is estimated from calibration activations as the
// diagonal of the layer Hessian H = 2·X·Xᵀ (X = [n_tokens, K] inputs).
//
// Runs independently of t3 (SpinQuant): it takes a plain FP16/fp32 weight
// tensor + calibration data and emits one or more candidate quantizations
// (INT2/INT4/INT8/FP16) with a reported reconstruction error.
#pragma once

#include <cstdint>
#include <vector>

#include "intfer/quant_spec.h"

namespace intfer {
namespace dashq {

// Number of distinct levels for a low-bit scheme (0 for FP16).
inline int n_levels(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return 4;
    case QuantScheme::INT4: return 16;
    case QuantScheme::INT8: return 256;
    default:                 return 0;
  }
}

// float <-> IEEE-754 binary16 (round-to-nearest-even). Returns/stores raw u16.
uint16_t float_to_half(float v);
float    half_to_float(uint16_t h);

// ---------------------------------------------------------------------------
// Diagonal Hessian
// ---------------------------------------------------------------------------
// Only the diagonal H_ii of the layer Hessian is retained. For a linear layer
// Y = W·X the (uncentered) Hessian is H = 2 X Xᵀ, so H_ii = 2·Σ_t X_{t,i}².
// Keeping just the diagonal is what makes every group an *independent* weighted
// least-squares problem (the DASH-Q speed/robustness win over GPTQ at int2).
class DiagHessian {
public:
  // Estimate H_ii from calibration activations X [n_tokens * K] (row-major).
  static DiagHessian from_activations(const float* X, uint32_t n_tokens, uint32_t K);
  // Uniform weights (no calibration available -> plain L2).
  static DiagHessian uniform(uint32_t K);
  // Explicit per-column (length K) or per-element (length N*K) weights.
  DiagHessian(std::vector<float> weights, bool per_element = false)
      : h_(std::move(weights)), per_element_(per_element),
        K_(per_element_ ? 0u : static_cast<uint32_t>(h_.size())) {}

  const std::vector<float>& data() const { return h_; }
  bool    per_element() const { return per_element_; }
  uint32_t K() const { return K_; }

private:
  std::vector<float> h_;
  bool    per_element_ = false;
  uint32_t K_ = 0;
};

// ---------------------------------------------------------------------------
// Quantization result for a single tensor at a single config
// ---------------------------------------------------------------------------
struct QuantResult {
  QuantConfig qcfg;
  std::vector<uint8_t> codes;   // packed: INT2=4/byte, INT4=2/byte, INT8=1/byte, FP16=raw half
  std::vector<uint8_t> scales;  // per-group FP16 scale, 2 bytes each (little-endian)
  std::vector<uint8_t> globals; // per-tensor FP32 global divisor α (4 bytes) if has_global_divisor
  std::vector<float>    codebook; // learned codebook (n_levels); empty => use canonical
  float est_error = 0.f;        // weighted relative-L2 reconstruction error (docs §6.4)
  float weighted_mse = 0.f;     // Σ H_ii (w_i − dequant)²
  float weighted_norm = 0.f;    // Σ H_ii w_i²

  uint64_t weight_bytes() const { return codes.size() + scales.size() + globals.size(); }
  // Dequantize to a [N,K] fp32 buffer (for validation against the FP16 reference).
  void dequant(std::vector<float>& out, uint32_t N, uint32_t K) const;
  // Effective stored bits/weight = 8·(codes+scales+globals)/(N·K).
  float eff_bits(uint32_t N, uint32_t K) const;
};

// Canonical (fixed) symmetric codebook with |max| = 1 for a scheme.
std::vector<float> canonical_codebook(QuantScheme s);

// ---------------------------------------------------------------------------
// Core solver
// ---------------------------------------------------------------------------
// Quantize a single fp32 [N,K] tensor to `cfg` using diagonal Hessian `H`.
//   * Per-group scale is always solved by weighted least squares (the
//     "iterative least squares" step).
//   * If `learn_codebook` and n_levels<=16, a single non-uniform symmetric
//     codebook is learned for the whole tensor via iterative weighted LS
//     (coordinate descent on scale + codebook); otherwise the canonical fixed
//     codebook is used.
QuantResult quantize(const float* W, uint32_t N, uint32_t K,
                     const QuantConfig& cfg, const DiagHessian& H,
                     bool learn_codebook = false, int iters = 24);

// Dequantize one *finalized* tensor (post-RCO: t.qcfg + the slices it indexes in
// `m.code_buf/scale_buf/global_buf`) into a [N,K] fp32 buffer. Used by validation
// and by the Stage-2 engine. Mirrors QuantResult::dequant on the stored payload.
void dequant_tensor(const ModelIR& m, const TensorIR& t, std::vector<float>& out);

// Emit the default candidate set (docs quant_spec.md §5) for a tensor role.
// Each candidate is appended to `parent.candidates` with its QuantConfig,
// estimated error, and payload slices written into `model`
// (code_buf / scale_buf / global_buf). This is exactly the object RCO (t5)
// consumes.
void quantize_candidates(TensorIR& parent, const float* W, uint32_t N, uint32_t K,
                         const DiagHessian& H, ModelIR& model,
                         bool learn_codebook = false);

} // namespace dashq
} // namespace intfer
