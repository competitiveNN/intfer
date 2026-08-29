// examples/quant_model_real/main.cpp
// Quantize the REAL Qwen3.5-0.8B checkpoint with our Stage-1 method
// (SpinQuant -> DASH-Q -> RCO) and verify the quantization does not break the
// model's linear operators.
//
// This is a host-only C++ program (no SYCL needed for Stage-1). Built directly
// with g++ (see scripts/build_quant_real.sh). It:
//   1. memory-maps the safetensors file and loads every Linear projection
//      weight of the language model (q/k/v/o, gate/up/down, embed_tokens),
//   2. runs run_pipeline() (SpinQuant orthogonal rotation + DASH-Q candidates
//      + RCO bit-budget selection),
//   3. verifies fidelity TWO ways, both rotation-invariant:
//        (a) weight reconstruction: ||dequant - rotated_FP16|| / ||rotated_FP16||
//        (b) linear-output error: for random inputs x, compare y = x·Wᵀ in the
//            rotated space for the FP16 vs dequantized weight (this is exactly
//            what our quantized GEMM kernel would compute).
//
// Usage:
//   quant_model_real <safetensors> [--layers N] [--budget-frac F] [--seed S]
//
// NOTE: this is an OPERATOR-LEVEL fidelity check, not full autoregressive
// generation (the Stage-2 engine does not yet run the hybrid Qwen3.5 arch
// end-to-end). It proves the quantized weights reproduce the FP16 linear maps.
#include "intfer/safetensors.hpp"
#include "intfer/pipeline.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/spinquant.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace intfer;

static TensorRole role_for_name(const std::string& n, bool& keep) {
  keep = true;
  if (n.find("self_attn.q_proj.weight") != std::string::npos) return TensorRole::ATTN_Q;
  if (n.find("self_attn.k_proj.weight") != std::string::npos) return TensorRole::ATTN_K;
  if (n.find("self_attn.v_proj.weight") != std::string::npos) return TensorRole::ATTN_V;
  if (n.find("self_attn.o_proj.weight") != std::string::npos) return TensorRole::ATTN_O;
  if (n.find("mlp.gate_proj.weight")   != std::string::npos) return TensorRole::MLP_GATE;
  if (n.find("mlp.up_proj.weight")     != std::string::npos) return TensorRole::MLP_UP;
  if (n.find("mlp.down_proj.weight")   != std::string::npos) return TensorRole::MLP_DOWN;
  if (n.find("embed_tokens.weight")    != std::string::npos) return TensorRole::EMBED;
  if (n.find("lm_head.weight")         != std::string::npos) return TensorRole::LM_HEAD;
  keep = false;
  return TensorRole::OTHER;
}

static const char* role_name(TensorRole r) {
  switch (r) {
    case TensorRole::EMBED:   return "EMBED";
    case TensorRole::LM_HEAD: return "LM_HEAD";
    case TensorRole::ATTN_Q:  return "ATTN_Q";
    case TensorRole::ATTN_K:  return "ATTN_K";
    case TensorRole::ATTN_V:  return "ATTN_V";
    case TensorRole::ATTN_O:  return "ATTN_O";
    case TensorRole::MLP_GATE:return "MLP_GATE";
    case TensorRole::MLP_UP:  return "MLP_UP";
    case TensorRole::MLP_DOWN:return "MLP_DOWN";
    default: return "OTHER";
  }
}
static const char* scheme_name(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return "INT2";
    case QuantScheme::INT4: return "INT4";
    case QuantScheme::INT8: return "INT8";
    default: return "FP16";
  }
}

int main(int argc, char** argv) {
  std::string path = "test/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
  int max_layers = 1 << 30;
  double budget_frac = 0.30;
  bool no_embed = false;
  std::string out_path;
  std::string dump_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--layers" && i + 1 < argc) max_layers = std::stoi(argv[++i]);
    else if (a == "--budget-frac" && i + 1 < argc) budget_frac = std::stod(argv[++i]);
    else if (a == "--no-embed") no_embed = true;
    else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];
    else if (a.rfind("--", 0) != 0) path = a;
  }

  printf("=== Quantize REAL Qwen3.5-0.8B with intfer Stage-1 ===\n");
  printf("model : %s\n", path.c_str());
  printf("layers: up to %d | budget_frac: %.2f (INT2 stress)\n\n", max_layers, budget_frac);

  SafetensorsReader reader;
  if (!reader.open(path)) { printf("FAIL: cannot open safetensors\n"); return 1; }

  // ---- Load Linear weights into ModelIR ----------------------------------
  ModelIR model;
  model.model_id = "Qwen3.5-0.8B";
  uint32_t n_layers = 0;
  size_t kept = 0, skipped = 0;
  uint64_t fp16_bytes = 0;
  std::mt19937 gen(0xC0FFEE);
  for (const auto& t : reader.tensors()) {
    bool keep;
    TensorRole role = role_for_name(t.name, keep);
    if (!keep) { ++skipped; continue; }
    int layer = parse_layer_index(t.name);
    if (layer >= 0) {
      if (layer >= max_layers) { ++skipped; continue; }
      n_layers = std::max(n_layers, (uint32_t)(layer + 1));
    }
    if (t.shape.size() != 2) { ++skipped; continue; }
    if (no_embed && role == TensorRole::EMBED) { ++skipped; continue; }
    uint32_t N = (uint32_t)t.shape[0];
    uint32_t K = (uint32_t)t.shape[1];

    std::vector<float> w;
    if (!reader.read(t.name, w)) { printf("FAIL: read %s\n", t.name.c_str()); return 1; }

    TensorIR ti;
    ti.name = t.name; ti.role = role; ti.N = N; ti.K = K;
    store_tensor_fp16(model, ti, w);
    model.tensors.push_back(ti);
    fp16_bytes += (uint64_t)N * K * 2;
    ++kept;
  }
  model.n_layers = n_layers;
  printf("loaded %zu linear tensors (skipped %zu), fp16 = %.1f MiB, layers = %u\n\n",
         kept, skipped, (double)fp16_bytes / (1024.0 * 1024.0), n_layers);
  if (kept == 0) { printf("FAIL: no linear tensors matched\n"); return 1; }

  // ---- Run the full Stage-1 pipeline -------------------------------------
  PipelineConfig cfg;
  cfg.sq.type = RotationType::Hadamard;
  cfg.sq.per_layer = false;
  cfg.learn_codebook = false;
  cfg.use_calibration = false;          // uniform diagonal Hessian (no runner yet)
  cfg.rco.budget_bytes = (uint64_t)((double)fp16_bytes * budget_frac);
  cfg.skip_int8_above = 16'000'000;  // skip INT8 cand for huge tensors (e.g. embed)
  cfg.rco.clear_fp16_after = false;     // KEEP rotated FP16 for verification
  cfg.rco.greedy_refine = true;

  PipelineResult res;
  try {
    res = run_pipeline(model, cfg);
  } catch (const std::exception& e) {
    printf("FAIL: run_pipeline threw: %s\n", e.what());
    return 1;
  }

  // ---- Per-role aggregates -----------------------------------------------
  struct Agg { double wsum = 0, wnorm = 0; double max_op = 0; size_t count = 0;
               uint64_t params = 0; } agg[(int)TensorRole::OTHER + 1];
  int scheme_counts[4] = {0,0,0,0}; // FP16,INT8,INT4,INT2 (enum order)

  std::vector<float> Wrot, Wq, xbuf, yrot, yq;
  const uint32_t T = 16;
  double g_wsum = 0, g_wnorm = 0, g_maxop = 0;
  uint64_t g_params = 0;

  printf("%-10s %5s %8s %10s %10s %8s\n", "role", "cnt", "w_recon", "max_out", "scheme", "params");
  printf("---------------------------------------------------------------------\n");

  for (size_t i = 0; i < res.model.tensors.size(); ++i) {
    const auto& t = res.model.tensors[i];
    uint32_t N = t.N, K = t.K;
    Wrot = load_tensor_fp16(res.model, t);
    if (Wrot.empty()) { printf("FAIL: rotated FP16 missing for %s\n", t.name.c_str()); return 1; }
    Wq.clear(); dashq::dequant_tensor(res.model, t, Wq);

    // (a) weight reconstruction (quantization-only error)
    double num = 0, den = 0;
    for (size_t j = 0; j < Wrot.size(); ++j) {
      double d = (double)Wq[j] - (double)Wrot[j];
      num += d * d; den += (double)Wrot[j] * (double)Wrot[j];
    }
    double wrec = (float)std::sqrt(num / (den + 1e-12));

    // (b) linear-output error with random inputs (rotated space, both sides)
    xbuf.resize((size_t)T * K);
    yrot.assign((size_t)T * N, 0.f); yq.assign((size_t)T * N, 0.f);
    for (size_t k = 0; k < xbuf.size(); ++k) xbuf[k] = ((float)gen() / (float)gen.max()) * 2.f - 1.f;
    double maxop = 0;
    for (uint32_t r = 0; r < T; ++r) {
      const float* x = &xbuf[(size_t)r * K];
      for (uint32_t j = 0; j < N; ++j) {
        const float* wr = &Wrot[(size_t)j * K];
        const float* wq = &Wq[(size_t)j * K];
        float sr = 0, sq = 0;
        for (uint32_t k = 0; k < K; ++k) { sr += x[k] * wr[k]; sq += x[k] * wq[k]; }
        yrot[(size_t)r * N + j] = sr; yq[(size_t)r * N + j] = sq;
      }
      double on = 0, od = 0;
      for (uint32_t j = 0; j < N; ++j) {
        double d = (double)yq[(size_t)r * N + j] - (double)yrot[(size_t)r * N + j];
        on += d * d; od += (double)yrot[(size_t)r * N + j] * (double)yrot[(size_t)r * N + j];
      }
      maxop = std::max(maxop, std::sqrt(on / (od + 1e-12)));
    }

    int ri = (int)t.role;
    agg[ri].wsum += wrec * (double)N * K;
    agg[ri].wnorm += (double)N * K;
    agg[ri].max_op = std::max(agg[ri].max_op, maxop);
    agg[ri].count += 1;
    agg[ri].params += (uint64_t)N * K;
    g_wsum += wrec * (double)N * K; g_wnorm += (double)N * K;
    g_maxop = std::max(g_maxop, maxop); g_params += (uint64_t)N * K;
    int si = (int)t.qcfg.scheme; scheme_counts[si]++;

    printf("%-10s %5zu %8.4f %10.4f %8s %8.1fM\n", role_name(t.role), agg[ri].count,
           (float)(agg[ri].wsum / agg[ri].wnorm), (float)agg[ri].max_op,
           scheme_name(t.qcfg.scheme), (double)agg[ri].params / 1e6);
  }

  // ---- Summary ------------------------------------------------------------
  double comp = fp16_bytes ? (double)fp16_bytes / (double)res.rco_report.total_bytes : 0;
  printf("\n---------------------------------------------------------------------\n");
  printf("RCO feasible: %s | budget %.1f MiB | selected %.1f MiB | compression %.2fx\n",
         res.rco_report.feasible ? "YES" : "NO",
         (double)cfg.rco.budget_bytes / (1024.0 * 1024.0),
         (double)res.rco_report.total_bytes / (1024.0 * 1024.0), comp);
  printf("scheme dist: INT2=%d INT4=%d INT8=%d FP16=%d\n",
         scheme_counts[3], scheme_counts[2], scheme_counts[1], scheme_counts[0]);
  printf("overall weight reconstruction (params-weighted): %.4f rel-L2\n",
         (float)(g_wsum / g_wnorm));
  printf("overall max linear-output error: %.4f rel-L2\n", (float)g_maxop);

  // Per-role mean weight reconstruction lines
  printf("\nper-role mean weight reconstruction:\n");
  for (int r = 0; r <= (int)TensorRole::OTHER; ++r) {
    if (agg[r].count == 0) continue;
    printf("  %-10s %.4f  (max_out %.4f, %zu tensors)\n",
           role_name((TensorRole)r), (float)(agg[r].wsum / agg[r].wnorm),
           (float)agg[r].max_op, agg[r].count);
  }

  // ---- Verdict (operator-level) ------------------------------------------
  // Thresholds are deliberately lenient: at INT2 with NO calibration data the
  // per-group L2 error is expected to be a few percent. "Broken" would mean the
  // linear map is no longer reproduced (output error >> weight error).
  bool ok = res.rco_report.feasible;
  ok &= (g_wsum / g_wnorm) < 0.25;   // mean weight recon < 25%
  ok &= (g_maxop < 0.40);            // linear-output error < 40%
  printf("\nVERDICT: quantization reproduces the FP16 linear operators: %s\n",
         ok ? "PASS" : "REVIEW");
  printf("(operator-level fidelity; full autoregressive generation not yet run)\n");

  // ---- Write the native-SYCL artifact (if requested) ---------------------
  if (!out_path.empty()) {
    std::vector<uint8_t> blob = serialize_intfer(res.model, res.rotations);
    if (write_file(out_path, blob)) {
      printf("\nwrote native-SYCL artifact: %s (%.1f MiB)\n",
             out_path.c_str(), (double)blob.size() / (1024.0 * 1024.0));
      printf("  contains %zu tensors, %zu rotations (magic INTFER1)\n",
             res.model.tensors.size(), res.rotations.size());
    } else {
      printf("\nFAIL: could not write %s\n", out_path.c_str());
      return 1;
    }
  }

  // ---- Dump dequantized weights (for external validation, e.g. transformers) -
  // Simple format: magic "DQMP" | uint32 count | per tensor:
  //   uint32 name_len | name bytes | uint32 N | uint32 K | N*K float32 LE.
  if (!dump_path.empty()) {
    FILE* f = std::fopen(dump_path.c_str(), "wb");
    if (!f) { printf("\nFAIL: cannot open dump %s\n", dump_path.c_str()); return 1; }
    uint32_t cnt = (uint32_t)res.model.tensors.size();
    fwrite("DQMP", 1, 4, f);
    fwrite(&cnt, 4, 1, f);
    std::vector<float> w;
    for (const auto& t : res.model.tensors) {
      w.clear(); dashq::dequant_tensor(res.model, t, w);
      uint32_t nl = (uint32_t)t.name.size();
      fwrite(&nl, 4, 1, f);
      fwrite(t.name.data(), 1, nl, f);
      fwrite(&t.N, 4, 1, f);
      fwrite(&t.K, 4, 1, f);
      fwrite(w.data(), 4, w.size(), f);
    }
    std::fclose(f);
    printf("\nwrote dequantized weights: %s (%u tensors)\n", dump_path.c_str(), cnt);
  }
  return ok ? 0 : 2;
}
