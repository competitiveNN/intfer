// examples/rco_demo/main.cpp
// Standalone validation of the RCO bit-budget search (t5).
//
// Builds a synthetic multi-layer transformer with tensors across every role,
// runs t4 (DASH-Q) to emit candidate quantizations into each TensorIR, then:
//   1. runs RCO under a LOOSE budget (everything fits) and checks it selects the
//      most-accurate allocation while staying within budget,
//   2. runs RCO under a TIGHT budget and checks the §5 per-role min-bit floors
//      are honored (embed/lm_head >= INT4; layer-0 attn_q/k/v >= INT4), that the
//      bulk params fall to INT2 under pressure, and total bytes stay <= budget,
//   3. shows the search actually buys accuracy: RCO's weighted error is <= the
//      error of an "all-cheapest (INT2-G16)" baseline under the same budget,
//   4. verifies run_rco() rewrites the ModelIR to a single canonical config per
//      tensor (candidates cleared, payload buffers repacked, sizes consistent).
//
// Pure host C++ — run with:
//   cmake -S examples/rco_demo -B build_rco && cmake --build build_rco && ./build_rco/rco_demo
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "intfer/quant_spec.h"
#include "intfer/quant/dashq.hpp"
#include "intfer/rco.hpp"

using namespace intfer;
using namespace intfer::dashq;
using namespace intfer::rco;

static int srank(QuantScheme s) {
  return (s == QuantScheme::INT2) ? 2 : (s == QuantScheme::INT4) ? 4
       : (s == QuantScheme::INT8) ? 8 : 16;
}

// Build a synthetic weight tensor [N,K] (random + an outlier channel) and emit
// the DASH-Q candidate set into the model under `parent` (name/role/N/K only).
static void make_tensor(ModelIR& model, const TensorIR& parent, uint32_t N, uint32_t K,
                        std::mt19937& gen) {
  std::vector<float> W((size_t)N * K);
  std::normal_distribution<float> nrm(0.0f, 0.05f);
  for (uint32_t i = 0; i < N * K; ++i) W[i] = nrm(gen);
  for (uint32_t r = 0; r < N; ++r) W[r * K + (K / 7)] += (r % 2 ? 1.f : -1.f) * 1.5f;
  const DiagHessian H = DiagHessian::uniform(K);
  TensorIR p = parent;
  quantize_candidates(p, W.data(), N, K, H, model, /*learn_codebook=*/false);
  model.tensors.push_back(p);
}

static TensorIR spec(const char* name, TensorRole role, uint32_t N, uint32_t K) {
  TensorIR t;
  t.name = name;
  t.role = role;
  t.N = N; t.K = K;
  return t;
}

// Build a fresh 4-layer model (30 weight tensors) with DASH-Q candidates.
static void build_model(ModelIR& model) {
  model = ModelIR{};
  model.model_id = "rco-demo";
  std::mt19937 gen(20240829);
  const uint32_t HID = 256;
  const char* attn[3] = {"q_proj", "k_proj", "v_proj"};
  const char* mlp[3] = {"gate_proj", "up_proj", "down_proj"};
  make_tensor(model, spec("model.embed_tokens.weight", TensorRole::EMBED, 1024, HID), HID, 1024, gen);
  make_tensor(model, spec("model.lm_head.weight", TensorRole::LM_HEAD, 1024, HID), HID, 1024, gen);
  for (int L = 0; L < 4; ++L) {
    char nm[128];
    for (int a = 0; a < 3; ++a) {
      snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.%s.weight", L, attn[a]);
      TensorRole r = (a == 0) ? TensorRole::ATTN_Q : (a == 1) ? TensorRole::ATTN_K : TensorRole::ATTN_V;
      make_tensor(model, spec(nm, r, HID, HID), HID, HID, gen);
    }
    snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", L);
    make_tensor(model, spec(nm, TensorRole::ATTN_O, HID, HID), HID, HID, gen);
    for (int m = 0; m < 3; ++m) {
      snprintf(nm, sizeof(nm), "model.layers.%d.mlp.%s.weight", L, mlp[m]);
      TensorRole r = (m == 0) ? TensorRole::MLP_GATE : (m == 1) ? TensorRole::MLP_UP : TensorRole::MLP_DOWN;
      make_tensor(model, spec(nm, r, HID, HID), HID, HID, gen);
    }
  }
}

int main() {
  printf("=== RCO bit-budget search (t5) ===\n\n");
  ModelIR model;
  build_model(model);
  uint64_t full_fp16 = 0;
  for (const auto& t : model.tensors) full_fp16 += (uint64_t)t.N * t.K * 2;
  printf("model: %zu tensors, full FP16 = %.2f MiB\n\n",
         model.tensors.size(), (double)full_fp16 / (1024.0 * 1024.0));
  bool ok = true;

  // ---- 1. LOOSE budget: everything fits -> most-accurate allocation --------
  {
    RcoOptions opt;
    opt.budget_bytes = full_fp16 * 2;
    RcoReport rep = run_rco(model, opt);
    printf("[LOOSE budget = 2x FP16]\n");
    rep.print(&model);
    bool fits = rep.total_bytes <= opt.budget_bytes;
    bool all_fp16 = true;
    for (const auto& s : rep.selections)
      if (s.scheme != QuantScheme::FP16) all_fp16 = false;
    printf("  fits budget: %s | all-FP16 (most accurate): %s\n", fits ? "PASS" : "FAIL",
           all_fp16 ? "PASS" : "FAIL");
    if (!fits || !all_fp16) ok = false;
  }

  // ---- 2. TIGHT budget: floors honored, bulk drops to INT2, budget fit -----
  ModelIR tight;
  build_model(tight);
  uint64_t tight_budget = (uint64_t)(0.22 * (double)full_fp16);
  {
    // Baseline (all-cheapest) error computed BEFORE run_rco consumes candidates.
    double base_err = 0.0;
    for (const auto& t : tight.tensors) {
      uint64_t mc = ~0ULL; size_t mi = 0;
      for (size_t j = 0; j < t.candidates.size(); ++j)
        if (t.candidates[j].weight_bytes() < mc) { mc = t.candidates[j].weight_bytes(); mi = j; }
      base_err += (double)t.candidates[mi].est_error;
    }

    RcoOptions opt;
    opt.budget_bytes = tight_budget;
    RcoReport rep = run_rco(tight, opt);
    printf("\n[TIGHT budget = %.2f MiB (22%% of FP16)]\n", (double)tight_budget / (1024.0 * 1024.0));
    rep.print(&tight);

    bool fits = rep.total_bytes <= opt.budget_bytes;
    bool floors_ok = true;
    bool any_int2 = false;
    for (const auto& s : rep.selections) {
      const std::string& nm = tight.tensors[s.tensor_index].name;
      if (s.scheme == QuantScheme::INT2) any_int2 = true;
      if (s.role == TensorRole::EMBED || s.role == TensorRole::LM_HEAD) {
        if (srank(s.scheme) < 4) { floors_ok = false; printf("  FLOOR VIOLATION: %s\n", nm.c_str()); }
      }
      int layer = parse_layer_index(nm);
      if ((s.role == TensorRole::ATTN_Q || s.role == TensorRole::ATTN_K ||
           s.role == TensorRole::ATTN_V) && layer == 0) {
        if (srank(s.scheme) < 4) { floors_ok = false; printf("  FLOOR VIOLATION: layer0 %s\n", nm.c_str()); }
      }
    }
    bool buys = rep.total_unweighted_error <= base_err + 1e-9;
    printf("  fits budget: %s | §5 floors honored: %s | bulk reached INT2: %s\n",
           fits ? "PASS" : "FAIL", floors_ok ? "PASS" : "FAIL", any_int2 ? "PASS" : "FAIL");
    printf("  RCO error=%.4f vs all-cheapest baseline=%.4f (search buys accuracy: %s)\n",
           rep.total_unweighted_error, base_err, buys ? "PASS" : "FAIL");
    if (!fits || !floors_ok || !any_int2 || !buys) ok = false;

    // ---- 4. ModelIR rewrite consistency ------------------------------------
    bool consistent = true;
    uint64_t sum_code = 0, sum_scale = 0, sum_global = 0;
    for (const auto& t : tight.tensors) {
      if (!t.candidates.empty()) consistent = false;  // candidates must be cleared
      sum_code += t.code_bytes;
      sum_scale += t.scale_bytes;
      sum_global += t.global_bytes;
    }
    if (tight.code_buf.size() != sum_code) consistent = false;
    if (tight.scale_buf.size() != sum_scale) consistent = false;
    if (tight.global_buf.size() != sum_global) consistent = false;
    if (tight.total_weight_bytes() != rep.total_bytes) consistent = false;
    printf("  ModelIR rewrite consistent (candidates cleared, buffers repacked): %s\n",
           consistent ? "PASS" : "FAIL");
    if (!consistent) ok = false;
  }

  printf("\nACCEPTANCE: RCO selects one candidate/tensor under budget with §5 floors: %s\n",
         ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
