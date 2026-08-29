// src/quant/pipeline.cpp
// t6 — INT2 quantization pipeline: SpinQuant (t3) -> DASH-Q candidates (t4) ->
// RCO bit-budget selection (t5). Pure host C++ (no SYCL). Consumes a ModelIR
// carrying FP16 weights and produces the finalized ModelIR (+ RotationSet) that
// output.hpp serializes to the native SYCL loader / GGUF formats.
#include "intfer/pipeline.hpp"

#include <stdexcept>
#include <vector>

#include "intfer/quant/dashq.hpp"
#include "intfer/spinquant.hpp"

namespace intfer {

PipelineResult run_pipeline(const ModelIR& input, const PipelineConfig& cfg) {
  PipelineResult res;

  // ---- t3: SpinQuant orthogonal rotation preprocessing ----
  SpinQuantResult sqr = spinquant_preprocess(input, cfg.sq, cfg.use_calibration ? &cfg.calib : nullptr);
  res.rotations = sqr.rotations;

  // take ownership of the rotated FP16 model
  ModelIR& m = res.model;
  m = std::move(sqr.model);
  m.model_id = cfg.model_id;
  m.n_layers = cfg.n_layers;

  // unquantized footprint for compression reporting
  for (const auto& t : input.tensors)
    res.fp16_baseline_bytes += (uint64_t)t.N * t.K * 2;

  // ---- t4: DASH-Q candidate quantization (per tensor/role) ----
  for (auto& t : m.tensors) {
    std::vector<float> W = load_tensor_fp16(m, t);
    if (W.empty()) continue;  // tensor has no FP16 payload -> RCO treats as fixed

    // Diagonal Hessian: from calibration activations if provided, else uniform.
    dashq::DiagHessian H = dashq::DiagHessian::uniform(t.K);
    if (cfg.use_calibration) {
      for (const auto& kv : cfg.calib.activations) {
        if (kv.first == t.name && !kv.second.empty()) {
          const uint32_t nTok = (uint32_t)(kv.second.size() / (size_t)t.K);
          H = dashq::DiagHessian::from_activations(kv.second.data(), nTok, t.K);
          break;
        }
      }
    }
    dashq::quantize_candidates(t, W.data(), t.N, t.K, H, m, cfg.learn_codebook,
                               cfg.skip_int8_above);
  }

  // ---- t5: RCO bit-budget search (mutates m to single canonical config) ----
  res.rco_report = rco::run_rco(m, cfg.rco);
  if (!res.rco_report.feasible && res.rco_report.total_bytes > cfg.rco.budget_bytes) {
    throw std::runtime_error("RCO produced an infeasible allocation (exceeds VRAM budget)");
  }
  return res;
}

} // namespace intfer
