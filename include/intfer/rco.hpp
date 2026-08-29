// include/intfer/rco.hpp
// RCO: Runtime-Constrained Optimization — bit-budget search (Stage 1, step 3).
//
// After t4 (DASH-Q) every rotatable tensor carries a list of candidate
// quantizations in `TensorIR::candidates` (each with a QuantConfig, a stored
// byte cost, and a DASH-Q relative-L2 `est_error`, docs quant_spec.md §6.3/§6.4).
//
// RCO consumes that candidate set and picks EXACTLY ONE candidate per tensor so
// that the total stored weight bytes fit an exact VRAM budget while minimizing a
// global accuracy proxy (docs §5):
//
//     minimize  Σ_i  w_role(i) · est_error_i        (total reconstruction error)
//     subject to  Σ_i  weight_bytes_i  ≤  budget_bytes
//
// This is a Multiple-Choice Knapsack Problem (one choice per group). The budget
// is the §1 "effective total weight budget" (~36 GB for 2×B580), passed in bytes.
//
// RCO also enforces the per-role *min-bit floors* from §5:
//   embed / lm_head        ≥ INT4
//   attn_q/k/v layer 0     ≥ INT4   (ninfer-style early-outlier escape)
//   everything else        ≥ INT2   (FP16 only candidate for norm/rope/other)
//
// Solver strategy (cheap, always feasible, near-optimal):
//   1. filter each tensor's candidates to those meeting its floor,
//   2. Lagrangian relaxation: for a penalty λ on bytes, each tensor independently
//      picks the option minimizing (value + λ·cost); C(λ) (total bytes) is
//      non-increasing in λ. Bisect λ to the smallest value with C(λ) ≤ budget →
//      a feasible allocation sitting on the budget boundary (best accuracy),
//   3. local-search polish: coordinate-descent over tensors, each step taking the
//      feasible option that most reduces total error, until convergence.
//
// The chosen config is written back into the parent TensorIR (qcfg / est_error /
// payload slice offsets) and the ModelIR code/scale/global buffers are repacked
// to contain ONLY the selected slices — exactly what t6 (pipeline/output) reads.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "intfer/quant_spec.h"

namespace intfer {
namespace rco {

// ---------------------------------------------------------------------------
// Min-bit floors (docs §5)
// ---------------------------------------------------------------------------
inline int scheme_rank(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return 2;
    case QuantScheme::INT4: return 4;
    case QuantScheme::INT8: return 8;
    case QuantScheme::FP16: return 16;
  }
  return 0;
}

// Lowest scheme allowed for a (role, layer) under the §5 policy.
QuantScheme floor_scheme(TensorRole role, int layer_idx);
// Convenience: parse the layer index from `model.layers.N...` and apply the rule.
QuantScheme floor_scheme(const TensorIR& t);
// -1 when the name does not encode a layer number.
int parse_layer_index(const std::string& name);

// ---------------------------------------------------------------------------
// Options / report
// ---------------------------------------------------------------------------
struct RcoOptions {
  // Exact VRAM byte budget the selected weights must fit (docs §1: ~36 GB for
  // 2×B580). This is the hard constraint RCO optimizes against.
  uint64_t budget_bytes = 36ULL * 1024 * 1024 * 1024;
  // Drop the source FP16 buffer from the finalized ModelIR (weights are now
  // fully quantized; t6 reads code/scale/global only). Saves artifact memory.
  bool clear_fp16_after = true;
  // Local-search coordinate-descent polish after the Lagrangian initialization.
  bool greedy_refine = true;
  int lagrange_iters = 60;   // bisection steps on the Lagrange multiplier
  int refine_passes = 24;    // max local-search passes

  // Optional per-role accuracy weights w_role for the §5 proxy
  // Σ w_role·(−est_error). Indexed by (uint8)TensorRole; unset roles default 1.0.
  std::vector<float> role_weights;
};

// One finalized selection per tensor.
struct RcoSelection {
  size_t tensor_index = 0;     // index into ModelIR::tensors
  size_t candidate_index = 0;  // index into TensorIR::candidates
  TensorRole role = TensorRole::OTHER;
  QuantScheme scheme = QuantScheme::FP16;
  uint16_t group_size = 0;
  uint64_t bytes = 0;          // weight_bytes() of the chosen candidate
  float est_error = 0.f;       // DASH-Q relative-L2 reconstruction error
  float eff_bits = 0.f;        // effective bits/weight (incl. scale overhead)
  uint64_t params = 0;         // N*K, for weighting aggregates
};

struct RcoReport {
  bool feasible = false;             // selected weights fit the budget
  uint64_t budget_bytes = 0;
  uint64_t total_bytes = 0;          // Σ chosen weight_bytes()
  double total_weighted_error = 0.0; // Σ w_role·est_error
  double total_unweighted_error = 0.0; // Σ est_error
  float mean_eff_bits = 0.f;         // params-weighted mean effective bits/weight
  std::vector<RcoSelection> selections;

  void print(const ModelIR* model = nullptr) const;
};

// ---------------------------------------------------------------------------
// Core entry point
// ---------------------------------------------------------------------------
// Select one candidate per tensor under `opt.budget_bytes`, mutate `model` to
// the finalized single-config form (candidates cleared, buffers repacked), and
// return a report. Tensors with an empty candidate list are left untouched
// (their current qcfg is assumed canonical, e.g. FP16-only norm/rope/other).
RcoReport run_rco(ModelIR& model, const RcoOptions& opt = {});

// Non-mutating variant: compute the selection only (no ModelIR rewrite). Useful
// for callers that want to inspect the plan before committing.
RcoReport plan_rco(const ModelIR& model, const RcoOptions& opt = {});

} // namespace rco
} // namespace intfer
