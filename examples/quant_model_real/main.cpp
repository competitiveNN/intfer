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
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace intfer;

static std::string model_id = "Qwen3.5-0.8B";

// ---------------------------------------------------------------------------
// Memory budget enforcement
// ---------------------------------------------------------------------------
// Read /proc/meminfo to get MemAvailable. Returns 0 on failure (e.g. non-Linux).
static uint64_t meminfo_available_bytes() {
  std::ifstream f("/proc/meminfo");
  if (!f) return 0;
  std::string key;
  uint64_t kb = 0;
  while (f >> key) {
    if (key == "MemAvailable:") {
      f >> kb;
      return kb * 1024ULL;
    }
  }
  return 0;
}

// Estimate the FP16 weight bytes per layer by walking the safetensors tensor
// list once (only metadata, no decode). Used to derive a safe --layer-count
// when --max-mem-frac / --max-mem-mib are set.
struct LayerFootprint {
  uint64_t per_layer_bytes = 0;   // average FP16 size of linear tensors per layer
  uint64_t embed_bytes = 0;       // embed/lm_head (kept once across the whole run)
  int      layers = 0;            // number of distinct model layers found
};
static LayerFootprint scan_layer_footprint(const SafetensorsReader& sr) {
  LayerFootprint fp;
  std::set<int> seen;
  for (const auto& t : sr.tensors()) {
    if (t.shape.size() != 2) continue;
    bool is_embed = (t.name.find("embed_tokens.weight") != std::string::npos ||
                     t.name.find("lm_head.weight") != std::string::npos);
    uint64_t bytes = 2ULL;
    for (size_t d : t.shape) bytes *= d;
    if (is_embed) {
      fp.embed_bytes += bytes;
    } else {
      int L = parse_layer_index(t.name);
      if (L < 0) continue;
      if (seen.insert(L).second) ++fp.layers;
      fp.per_layer_bytes += bytes;
    }
  }
  if (fp.layers > 0) fp.per_layer_bytes /= size_t(fp.layers);
  return fp;
}

static TensorRole role_for_name(const std::string& n, bool& keep) {
  // Skip speculative-decoder / MTP tensors entirely. They live in
  // "mtp.layers.*" or "mtp.*" and are a separate model that the
  // Stage-2 engine doesn't consume. Skipping them here also prevents
  // them from being duplicated into every layer-chunk of the output.
  if (n.rfind("mtp.", 0) == 0 || n.find("/mtp.") != std::string::npos) {
    keep = false;
    return TensorRole::OTHER;
  }
  keep = true;
  if (n.find("self_attn.q_proj.weight") != std::string::npos) return TensorRole::ATTN_Q;
  if (n.find("self_attn.k_proj.weight") != std::string::npos) return TensorRole::ATTN_K;
  if (n.find("self_attn.v_proj.weight") != std::string::npos) return TensorRole::ATTN_V;
  if (n.find("self_attn.o_proj.weight") != std::string::npos) return TensorRole::ATTN_O;
  if (n.find("self_attn.q_norm.weight") != std::string::npos) return TensorRole::NORM;
  if (n.find("self_attn.k_norm.weight") != std::string::npos) return TensorRole::NORM;
  if (n.find("mlp.gate_proj.weight")   != std::string::npos) return TensorRole::MLP_GATE;
  if (n.find("mlp.up_proj.weight")     != std::string::npos) return TensorRole::MLP_UP;
  if (n.find("mlp.down_proj.weight")   != std::string::npos) return TensorRole::MLP_DOWN;
  // linear_attn (SSM, Qwen3.5 hybrid) — bulk of the 27B params.
  // Each in/out Linear gets its own role so DASH-Q can target the
  // bulk-projection tensors (qkv, z, out_proj) with INT2/INT4/INT8 and
  // the small-projection tensors (a, b) with the same ladder. conv1d is
  // small (3-4 columns) and gets the FP16-only floor via OTHER.
  if (n.find("linear_attn.in_proj_qkv.weight") != std::string::npos) return TensorRole::LINEAR_ATTN_QKV;
  if (n.find("linear_attn.in_proj_z.weight")   != std::string::npos) return TensorRole::LINEAR_ATTN_Z;
  if (n.find("linear_attn.in_proj_b.weight")   != std::string::npos) return TensorRole::LINEAR_ATTN_B;
  if (n.find("linear_attn.in_proj_a.weight")   != std::string::npos) return TensorRole::LINEAR_ATTN_A;
  if (n.find("linear_attn.out_proj.weight")    != std::string::npos) return TensorRole::LINEAR_ATTN_OUT;
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
    case TensorRole::NORM:    return "NORM";
    case TensorRole::ROPE:    return "ROPE";
    case TensorRole::LINEAR_ATTN_QKV: return "LINEAR_ATTN_QKV";
    case TensorRole::LINEAR_ATTN_Z:   return "LINEAR_ATTN_Z";
    case TensorRole::LINEAR_ATTN_B:   return "LINEAR_ATTN_B";
    case TensorRole::LINEAR_ATTN_A:   return "LINEAR_ATTN_A";
    case TensorRole::LINEAR_ATTN_CONV:return "LINEAR_ATTN_CONV";
    case TensorRole::LINEAR_ATTN_OUT: return "LINEAR_ATTN_OUT";
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
  int layer_start = 0;   // inclusive lower bound on layer index
  int layer_count = -1;  // -1 = no upper bound (use max_layers as the inclusive upper bound)
  double budget_frac = 0.30;
  double max_mem_frac = 0.90;   // max fraction of available RAM for the fp16_buf
  bool no_lm_head = false;   // exclude lm_head from this chunk (see note at the loader)
  uint64_t max_mem_mib = 0;     // explicit cap in MiB; overrides max_mem_frac if nonzero
  bool no_embed = false;
  std::string out_path;
  std::string dump_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--layers" && i + 1 < argc) max_layers = std::stoi(argv[++i]);
    else if (a == "--layer-start" && i + 1 < argc) layer_start = std::stoi(argv[++i]);
    else if (a == "--layer-count" && i + 1 < argc) layer_count = std::stoi(argv[++i]);
    else if (a == "--budget-frac" && i + 1 < argc) budget_frac = std::stod(argv[++i]);
    else if (a == "--max-mem-frac" && i + 1 < argc) max_mem_frac = std::stod(argv[++i]);
    else if (a == "--max-mem-mib" && i + 1 < argc) max_mem_mib = std::stoull(argv[++i]);
    else if (a == "--no-embed") no_embed = true;
    else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];
    else if (a.rfind("--", 0) != 0) path = a;
    else if (a == "--no-lm-head") no_lm_head = true;
  }
  // --layer-start + --layer-count define a half-open range [start, start+count).
  // They take precedence over --layers (which is treated as the inclusive upper bound).
  const int layer_end_excl = (layer_count >= 0) ? (layer_start + layer_count)
                                                : max_layers;

  // Auto-detect model_id from the safetensors directory name.
  // e.g. .../Qwen3.8-27B/model.safetensors-... -> "Qwen3.8-27B"
  {
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) pos = path.rfind('\\');
    if (pos != std::string::npos) {
      std::string dir = path.substr(0, pos);
      size_t p2 = dir.rfind('/');
      if (p2 != std::string::npos) {
        std::string d = dir.substr(p2 + 1);
        // strip any ".safetensors" suffix info — we want the model dir name.
        if (d.find("Qwen3.") != std::string::npos)
          model_id = d;
        else if (d.find("Qwen3_5") != std::string::npos)
          model_id = "Qwen3.5-0.8B";
      }
    }
  }

  printf("=== Quantize %s with intfer Stage-1 ===\n", model_id.c_str());
  printf("model : %s\n", path.c_str());
  if (layer_count >= 0)
    printf("layers: [%d, %d) (%d layers) | budget_frac: %.2f (INT2 stress)\n\n",
           layer_start, layer_end_excl, layer_count, budget_frac);
  else
    printf("layers: up to %d | budget_frac: %.2f (INT2 stress)\n\n", max_layers, budget_frac);

  SafetensorsReader reader;
  if (!reader.open(path)) { printf("FAIL: cannot open safetensors\n"); return 1; }

  // ---- Memory budget enforcement ---------------------------------------
  // Compute how many layers fit into memory given max_mem_frac (default
  // 0.90) of available RAM.  The per-layer fp16 footprint is computed
  // from the safetensors metadata; embed/lm_head are counted once.
  // If --max-mem-mib is given it overrides max_mem_frac.
  {
    const uint64_t avail = meminfo_available_bytes();
    uint64_t cap_bytes = (max_mem_mib > 0)
                             ? max_mem_mib * 1024ULL * 1024ULL
                             : uint64_t(double(avail) * max_mem_frac);
    LayerFootprint fp = scan_layer_footprint(reader);
    // Guard against divide-by-zero if the model has no learnable layers.
    uint64_t per_layer = fp.per_layer_bytes;
    uint64_t extra = fp.embed_bytes;  // embed/lm_head loaded once total
    int safe_layers = max_layers;
    if (per_layer > 0) {
      // Budget must cover: (layers * per_layer) + extra
      safe_layers = int((cap_bytes - extra) / per_layer);
      if (safe_layers < 1) safe_layers = 1;
    }
    if (layer_count > safe_layers) {
      printf("[max-mem] cap=%" PRIu64 " MiB avail=%" PRIu64 " MiB "
             "per_layer=%" PRIu64 " MiB -> max %d layers (requested %d)\n",
             cap_bytes / (1024 * 1024), avail / (1024 * 1024),
             per_layer / (1024 * 1024), safe_layers, layer_count);
      layer_count = safe_layers;
    }
    printf("[max-mem] available=%" PRIu64 " MiB cap=%" PRIu64 " MiB "
           "(%.1f%%) per_layer_footprint=%" PRIu64 " MiB\n",
           avail / (1024 * 1024), cap_bytes / (1024 * 1024),
           max_mem_frac * 100.0, per_layer / (1024 * 1024));
  }

  // ---- Load Linear weights into ModelIR ----------------------------------
  ModelIR model;
  model.model_id = model_id;
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
      if (layer < layer_start || layer >= layer_end_excl) { ++skipped; continue; }
      n_layers = std::max(n_layers, (uint32_t)(layer + 1));
    }
    if (t.shape.size() != 2) { ++skipped; continue; }
    if (no_embed && role == TensorRole::EMBED) { ++skipped; continue; }
    // --no-lm-head: LM_HEAD (248320x5120 = 2428 MiB fp16) carries no layer
    // index, so the layer-range guard above never filters it. With only
    // --no-embed, EVERY chunk re-loads it, and both the chunk's `fp16_bytes`
    // (hence its RCO budget) and its cheapest allocation grow by its INT4 floor
    // (643 MiB). Measured on a 2-layer LIN+LIN chunk at 27B: loaded fp16
    // 3886.9 MiB = lm_head 2428 + layers 1462, so a --budget-frac of 0.20 gave
    // a 778 MiB budget against an 871 MiB floor and RCO correctly REFUSED
    // ("infeasible allocation"). Chunks >0 that only carry layers must exclude
    // it, or the frac needed to stay feasible is set by lm_head rather than by
    // the layers the chunk actually quantizes. Chunk 0 keeps it (it is the one
    // place the embedding/head pair is selected once).
    if (no_lm_head && role == TensorRole::LM_HEAD) { ++skipped; continue; }
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
