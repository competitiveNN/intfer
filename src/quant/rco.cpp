// src/quant/rco.cpp
// RCO bit-budget search (Stage 1, step 3). Pure host C++ (no SYCL); the Stage-2
// SYCL path (t7) consumes the finalized ModelIR emitted here.
#include "intfer/rco.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace intfer {
namespace rco {

// ---------------------------------------------------------------------------
// Floors
// ---------------------------------------------------------------------------
int parse_layer_index(const std::string& name) {
  // Match the canonical HF prefix "model.language_model.layers." exactly.
  // We intentionally do NOT match the speculative "mtp.layers." decoder
  // (a separate model in Qwen3.5+) nor "model.layers." (older formats).
  static const std::string key = "model.language_model.layers.";
  size_t p = name.find(key);
  if (p == std::string::npos) return -1;
  p += key.size();
  int v = 0;
  bool any = false;
  while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
    v = v * 10 + (name[p] - '0');
    any = true;
    ++p;
  }
  return any ? v : -1;
}

QuantScheme floor_scheme(TensorRole role, int layer_idx) {
  switch (role) {
    case TensorRole::EMBED:
    case TensorRole::LM_HEAD:
      return QuantScheme::INT4;
    case TensorRole::ATTN_Q:
    case TensorRole::ATTN_K:
    case TensorRole::ATTN_V:
    case TensorRole::LINEAR_ATTN_QKV:
      // ninfer-style early-outlier escape: first layer forced >= INT4.
      return (layer_idx == 0) ? QuantScheme::INT4 : QuantScheme::INT2;
    default:
      // attn_o / mlp_* / linear_attn_* floor INT2; norm/rope/other only
      // expose an FP16 candidate anyway, so INT2 floor keeps it selectable.
      return QuantScheme::INT2;
  }
}

QuantScheme floor_scheme(const TensorIR& t) {
  return floor_scheme(t.role, parse_layer_index(t.name));
}

namespace {

float role_weight_of(TensorRole r, const RcoOptions& opt) {
  const size_t idx = static_cast<size_t>(r);
  if (!opt.role_weights.empty() && idx < opt.role_weights.size())
    return opt.role_weights[idx];
  return 1.0f;
}

// Per-tensor option after floor filtering.
struct Option {
  size_t cand_idx;     // index into TensorIR::candidates
  uint64_t cost;       // weight_bytes()
  double value;        // w_role * est_error  (to be minimized)
  float eff_bits;      // effective bits/weight for reporting
};

// Build the filtered option list for every tensor. Tensors without candidates
// are recorded as "fixed" (no options) and skipped by the optimizer.
struct Problem {
  std::vector<std::vector<Option>> opts;  // per tensor (empty => fixed)
  std::vector<size_t> fixed_cfg;          // chosen candidate idx for fixed tensors
  bool any_options = false;
};

Problem build_problem(const ModelIR& model, const RcoOptions& opt) {
  Problem P;
  P.opts.resize(model.tensors.size());
  P.fixed_cfg.resize(model.tensors.size(), 0);
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    const TensorIR& t = model.tensors[i];
    if (t.candidates.empty()) continue;  // fixed tensor (e.g. FP16-only)
    const QuantScheme floor = floor_scheme(t);
    const float w = role_weight_of(t.role, opt);
    std::vector<Option> local;
    for (size_t j = 0; j < t.candidates.size(); ++j) {
      const TensorIR& c = t.candidates[j];
      if (scheme_rank(c.qcfg.scheme) < scheme_rank(floor)) continue;  // floor
      const uint64_t cost = c.weight_bytes();
      const double value = w * static_cast<double>(c.est_error);
      local.push_back({j, cost, value, effective_bits(c.qcfg.scheme, c.qcfg.group_size, c.qcfg.scale_dtype)});
    }
    if (local.empty()) {
      // Floor excluded everything (shouldn't happen — FP16 always satisfies);
      // fall back to the FP16 candidate so the tensor stays selectable.
      for (size_t j = 0; j < t.candidates.size(); ++j) {
        if (t.candidates[j].qcfg.scheme == QuantScheme::FP16) {
          const uint64_t cost = t.candidates[j].weight_bytes();
          local.push_back({j, cost, w * (double)t.candidates[j].est_error,
                           effective_bits(QuantScheme::FP16, 0, t.candidates[j].qcfg.scale_dtype)});
          break;
        }
      }
    }
    if (!local.empty()) P.any_options = true;
    P.opts[i] = std::move(local);
  }
  return P;
}

// Evaluate the Lagrangian choice for a given multiplier λ: returns total bytes
// and the per-tensor pick indices (argmin of value + λ·cost per tensor).
struct Eval {
  uint64_t total_cost = 0;
  double total_value = 0.0;
  std::vector<size_t> pick;            // option index within opts[i] (0 if none)
};

Eval evaluate(const Problem& P, double lambda) {
  Eval e;
  e.pick.assign(P.opts.size(), 0);
  for (size_t i = 0; i < P.opts.size(); ++i) {
    const std::vector<Option>& o = P.opts[i];
    if (o.empty()) continue;  // fixed tensor
    size_t best = 0;
    double best_r = std::numeric_limits<double>::infinity();
    uint64_t best_cost = 0;
    for (size_t k = 0; k < o.size(); ++k) {
      const double r = o[k].value + lambda * (double)o[k].cost;
      // tie-break toward lower cost (saves budget for equal accuracy)
      if (r < best_r - 1e-12 || (std::fabs(r - best_r) <= 1e-12 && o[k].cost < best_cost)) {
        best_r = r;
        best = k;
        best_cost = o[k].cost;
      }
    }
    e.pick[i] = best;
    e.total_cost += o[best].cost;
    e.total_value += o[best].value;
  }
  return e;
}

// Cheapest possible allocation (all tensors at min cost), used to detect
// infeasibility and as the local-search starting anchor when needed.
Eval cheapest(const Problem& P) {
  Eval e;
  e.pick.assign(P.opts.size(), 0);
  for (size_t i = 0; i < P.opts.size(); ++i) {
    const std::vector<Option>& o = P.opts[i];
    if (o.empty()) continue;
    size_t best = 0;
    uint64_t bc = o[0].cost;
    for (size_t k = 1; k < o.size(); ++k)
      if (o[k].cost < bc) { bc = o[k].cost; best = k; }
    e.pick[i] = best;
    e.total_cost += o[best].cost;
    e.total_value += o[best].value;
  }
  return e;
}

// Lagrangian bisection: smallest λ with C(λ) ≤ budget (best-accuracy feasible).
Eval lagrange_feasible(const Problem& P, uint64_t budget, int iters) {
  const Eval e0 = evaluate(P, 0.0);
  if (e0.total_cost <= budget) return e0;          // most-accurate fits
  // doubling search for an upper bound where cost drops to ≤ budget
  double hi = 1.0;
  Eval ehi = evaluate(P, hi);
  while (ehi.total_cost > budget && hi < 1e30) {
    hi *= 2.0;
    ehi = evaluate(P, hi);
  }
  if (ehi.total_cost > budget) return cheapest(P);  // infeasible: least-cost fallback
  double lo = 0.0;
  for (int it = 0; it < iters; ++it) {
    const double mid = 0.5 * (lo + hi);
    const Eval em = evaluate(P, mid);
    if (em.total_cost <= budget) hi = mid;  // feasible side -> tighten up
    else lo = mid;
  }
  return evaluate(P, hi);
}

// Coordinate-descent local search: from `init`, repeatedly move each tensor to
// the feasible option that most reduces total error, until no improvement.
Eval local_search(const Problem& P, Eval cur, uint64_t budget, int passes) {
  for (int pass = 0; pass < passes; ++pass) {
    bool improved = false;
    for (size_t i = 0; i < P.opts.size(); ++i) {
      const std::vector<Option>& o = P.opts[i];
      if (o.empty()) continue;
      const uint64_t cur_cost = o[cur.pick[i]].cost;
      const double cur_val = o[cur.pick[i]].value;
      size_t best_k = cur.pick[i];
      double best_val = cur_val;
      uint64_t best_cost = cur_cost;
      for (size_t k = 0; k < o.size(); ++k) {
        const uint64_t new_cost = cur.total_cost - cur_cost + o[k].cost;
        if (new_cost > budget) continue;                 // must stay feasible
        if (o[k].value < best_val - 1e-12 ||
            (std::fabs(o[k].value - best_val) <= 1e-12 && o[k].cost < best_cost)) {
          best_val = o[k].value;
          best_k = k;
          best_cost = o[k].cost;
        }
      }
      if (best_k != cur.pick[i]) {
        cur.total_cost += o[best_k].cost - cur_cost;
        cur.total_value += o[best_k].value - cur_val;
        cur.pick[i] = best_k;
        improved = true;
      }
    }
    if (!improved) break;
  }
  return cur;
}

RcoReport build_report(const ModelIR& model, const Problem& P, const Eval& sol,
                       const RcoOptions& opt, bool feasible) {
  RcoReport rep;
  rep.feasible = feasible;
  rep.budget_bytes = opt.budget_bytes;
  rep.total_bytes = sol.total_cost;
  rep.total_weighted_error = sol.total_value;

  double unweighted = 0.0;
  double weighted_bits = 0.0;
  uint64_t total_params = 0;
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    const TensorIR& t = model.tensors[i];
    if (P.opts[i].empty()) {
      // fixed tensor (no candidates): report its current canonical config
      if (t.candidates.empty()) {
        RcoSelection s;
        s.tensor_index = i;
        s.role = t.role;
        s.scheme = t.qcfg.scheme;
        s.group_size = t.qcfg.group_size;
        s.bytes = t.weight_bytes();
        s.est_error = t.est_error;
        s.eff_bits = effective_bits(t.qcfg.scheme, t.qcfg.group_size, t.qcfg.scale_dtype);
        s.params = (uint64_t)t.N * t.K;
        rep.selections.push_back(s);
        unweighted += (double)t.est_error;
        weighted_bits += (double)s.eff_bits * (double)s.params;
        total_params += s.params;
      }
      continue;
    }
    const Option& o = P.opts[i][sol.pick[i]];
    const TensorIR& c = t.candidates[o.cand_idx];
    RcoSelection s;
    s.tensor_index = i;
    s.candidate_index = o.cand_idx;
    s.role = t.role;
    s.scheme = c.qcfg.scheme;
    s.group_size = c.qcfg.group_size;
    s.bytes = o.cost;
    s.est_error = c.est_error;
    s.eff_bits = o.eff_bits;
    s.params = (uint64_t)t.N * t.K;
    rep.selections.push_back(s);
    unweighted += (double)c.est_error;
    const float w = role_weight_of(t.role, opt);
    weighted_bits += (double)o.eff_bits * (double)s.params;
    total_params += s.params;
  }
  rep.total_unweighted_error = unweighted;
  rep.mean_eff_bits = total_params ? (float)(weighted_bits / (double)total_params) : 0.f;
  return rep;
}

// Apply a solution to the ModelIR: set canonical config + repack payload buffers.
void apply_solution(ModelIR& model, const Problem& P, const Eval& sol,
                    const RcoOptions& opt) {
  std::vector<uint8_t> new_code, new_scale, new_global;
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    TensorIR& t = model.tensors[i];
    if (P.opts[i].empty()) {
      t.candidates.clear();   // fixed tensors: drop any candidate list
      continue;
    }
    const Option& o = P.opts[i][sol.pick[i]];
    const TensorIR& c = t.candidates[o.cand_idx];

    // code slice
    if (c.code_bytes) {
      t.code_offset = new_code.size();
      new_code.insert(new_code.end(), model.code_buf.begin() + c.code_offset,
                      model.code_buf.begin() + c.code_offset + c.code_bytes);
    } else {
      t.code_offset = 0;
    }
    t.code_bytes = c.code_bytes;

    // scale slice
    if (c.scale_bytes) {
      t.scale_offset = new_scale.size();
      new_scale.insert(new_scale.end(), model.scale_buf.begin() + c.scale_offset,
                       model.scale_buf.begin() + c.scale_offset + c.scale_bytes);
    } else {
      t.scale_offset = 0;
    }
    t.scale_bytes = c.scale_bytes;

    // global slice
    if (c.global_bytes) {
      t.global_offset = new_global.size();
      new_global.insert(new_global.end(), model.global_buf.begin() + c.global_offset,
                       model.global_buf.begin() + c.global_offset + c.global_bytes);
    } else {
      t.global_offset = 0;
    }
    t.global_bytes = c.global_bytes;

    t.qcfg = c.qcfg;            // carries rotation_id from t3/t4
    t.est_error = c.est_error;
    t.candidates.clear();       // exactly one config chosen -> list consumed
  }
  model.code_buf = std::move(new_code);
  model.scale_buf = std::move(new_scale);
  model.global_buf = std::move(new_global);
  if (opt.clear_fp16_after) model.fp16_buf.clear();  // weights now quantized
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
RcoReport plan_rco(const ModelIR& model, const RcoOptions& opt) {
  Problem P = build_problem(model, opt);
  if (!P.any_options) {
    // Nothing to allocate: treat current configs as the plan (still report).
    Eval sol;
    sol.pick.assign(model.tensors.size(), 0);
    for (size_t i = 0; i < model.tensors.size(); ++i) {
      uint64_t cost = model.tensors[i].weight_bytes();
      sol.total_cost += cost;
      sol.total_value += (double)model.tensors[i].est_error;
    }
    const bool feasible = sol.total_cost <= opt.budget_bytes;
    return build_report(model, P, sol, opt, feasible);
  }
  Eval sol = lagrange_feasible(P, opt.budget_bytes, opt.lagrange_iters);
  const bool lag_feasible = sol.total_cost <= opt.budget_bytes;
  if (opt.greedy_refine) sol = local_search(P, sol, opt.budget_bytes, opt.refine_passes);
  const bool feasible = sol.total_cost <= opt.budget_bytes;
  // If even the cheapest allocation exceeds budget, mark infeasible.
  if (!feasible && !lag_feasible) {
    Eval ch = cheapest(P);
    sol = ch;
  }
  return build_report(model, P, sol, opt, feasible);
}

RcoReport run_rco(ModelIR& model, const RcoOptions& opt) {
  RcoReport rep = plan_rco(model, opt);
  // Rebuild the problem/solution against the live model to apply it. plan_rco
  // is non-mutating, so recompute here.
  Problem P = build_problem(model, opt);
  Eval sol;
  if (!P.any_options) {
    return rep;  // nothing to apply
  }
  sol = lagrange_feasible(P, opt.budget_bytes, opt.lagrange_iters);
  if (opt.greedy_refine) sol = local_search(P, sol, opt.budget_bytes, opt.refine_passes);
  if (!(sol.total_cost <= opt.budget_bytes)) sol = cheapest(P);
  apply_solution(model, P, sol, opt);
  return rep;
}

// ---------------------------------------------------------------------------
// Report printing
// ---------------------------------------------------------------------------
const char* scheme_name(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return "INT2";
    case QuantScheme::INT4: return "INT4";
    case QuantScheme::INT8: return "INT8";
    case QuantScheme::FP16: return "FP16";
  }
  return "?";
}
const char* role_name(TensorRole r) {
  switch (r) {
    case TensorRole::EMBED: return "EMBED";
    case TensorRole::LM_HEAD: return "LM_HEAD";
    case TensorRole::ATTN_Q: return "ATTN_Q";
    case TensorRole::ATTN_K: return "ATTN_K";
    case TensorRole::ATTN_V: return "ATTN_V";
    case TensorRole::ATTN_O: return "ATTN_O";
    case TensorRole::MLP_GATE: return "MLP_GATE";
    case TensorRole::MLP_UP: return "MLP_UP";
    case TensorRole::MLP_DOWN: return "MLP_DOWN";
    case TensorRole::NORM: return "NORM";
    case TensorRole::ROPE: return "ROPE";
    case TensorRole::LINEAR_ATTN_QKV: return "LINEAR_ATTN_QKV";
    case TensorRole::LINEAR_ATTN_Z:   return "LINEAR_ATTN_Z";
    case TensorRole::LINEAR_ATTN_B:   return "LINEAR_ATTN_B";
    case TensorRole::LINEAR_ATTN_A:   return "LINEAR_ATTN_A";
    case TensorRole::LINEAR_ATTN_CONV:return "LINEAR_ATTN_CONV";
    case TensorRole::LINEAR_ATTN_OUT: return "LINEAR_ATTN_OUT";
    default: return "OTHER";
  }
}

void RcoReport::print(const ModelIR* model) const {
  const double budget_gb = (double)budget_bytes / (1024.0 * 1024.0 * 1024.0);
  const double total_gb = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);
  printf("=== RCO bit-budget search report ===\n");
  printf("feasible            : %s\n", feasible ? "YES" : "NO");
  printf("budget              : %.3f GiB (%llu bytes)\n", budget_gb,
         (unsigned long long)budget_bytes);
  printf("selected weight bytes: %.3f GiB (%llu bytes)  [%.2f%% of budget]\n",
         total_gb, (unsigned long long)total_bytes,
         budget_bytes ? 100.0 * (double)total_bytes / (double)budget_bytes : 0.0);
  printf("total weighted error : %.6f\n", total_weighted_error);
  printf("total raw error      : %.6f\n", total_unweighted_error);
  printf("mean eff bits/weight : %.3f\n", mean_eff_bits);
  printf("\n%-26s %-9s %-5s %-9s %-10s %-10s %s\n",
         "tensor", "role", "bits", "scheme", "est_err", "bytes", "name");
  printf("--------------------------------------------------------------------------------\n");
  for (const auto& s : selections) {
    const char* nm = (model && s.tensor_index < model->tensors.size())
                         ? model->tensors[s.tensor_index].name.c_str() : "";
    printf("%-26s %-9s %-5.2f %-9s %-10.5f %-10llu %s\n",
           nm, role_name(s.role), s.eff_bits, scheme_name(s.scheme),
           s.est_error, (unsigned long long)s.bytes, nm);
  }
}

} // namespace rco
} // namespace intfer
