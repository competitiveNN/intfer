// examples/quant_pipeline/main.cpp
// End-to-end validation of the Stage-1 INT2 pipeline + GGUF / native-SYCL output
// (t6). Builds a synthetic 2-layer transformer in FP16, runs the full pipeline
// (SpinQuant -> DASH-Q -> RCO under a tight VRAM budget), then:
//   A. reconstruction: dequantized chosen configs closely match the rotated FP16
//   B. RCO report: §5 floors honored, bulk params reach INT2, fits the budget
//   C. native-SYCL (INTFER1) round-trip: buffers + dequant identical, rotations kept
//   D. GGUF v3 round-trip: buffers + dequant identical, rotations kept
//
// Pure host C++ — run with:
//   cmake -S examples/quant_pipeline -B build_pipeline && cmake --build build_pipeline && ./build_pipeline/quant_pipeline
#include "intfer/pipeline.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/spinquant.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace intfer;

static bool g_ok = true;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); g_ok = false; } } while (0)

static std::mt19937 g_gen(0x1234ABCD);
static float rnd() { return std::uniform_real_distribution<float>(-1.f, 1.f)(g_gen); }

static std::vector<float> matmul_TK_NK(const std::vector<float>& x, uint32_t T, uint32_t K,
                                       const std::vector<float>& W, uint32_t N) {
  std::vector<float> y(T * N, 0.f);
  for (uint32_t t = 0; t < T; ++t)
    for (uint32_t i = 0; i < N; ++i) {
      float a = 0.f;
      for (uint32_t k = 0; k < K; ++k) a += x[t * K + k] * W[i * K + k];
      y[t * N + i] = a;
    }
  return y;
}
static float rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  double n = 0.0, d = 0.0;
  for (size_t i = 0; i < a.size(); ++i) { n += (double)(a[i] - b[i]) * (a[i] - b[i]); d += (double)a[i] * a[i]; }
  return (float)std::sqrt(n / (d + 1e-12));
}

// Build a 2-layer model with FP16 weights stored in model.fp16_buf (as the
// pipeline expects on entry).
static ModelIR build_model() {
  ModelIR model;
  model.model_id = "intfer-t6-synthetic";
  model.n_layers = 2;
  const uint32_t d = 64, inter = 128;
  auto add = [&](const std::string& name, TensorRole role, uint32_t N, uint32_t K) {
    TensorIR t;
    t.name = name; t.role = role; t.N = N; t.K = K;
    std::vector<float> w((size_t)N * K);
    for (auto& v : w) v = rnd();
    store_tensor_fp16(model, t, w);
    model.tensors.push_back(t);
  };
  add("model.embed_tokens.weight", TensorRole::EMBED, 100, d);
  add("model.lm_head.weight",      TensorRole::LM_HEAD, 100, d);
  for (int L = 0; L < 2; ++L) {
    std::string p = "model.layers." + std::to_string(L) + ".";
    add(p + "self_attn.q_proj.weight", TensorRole::ATTN_Q, d, d);
    add(p + "self_attn.k_proj.weight", TensorRole::ATTN_K, d, d);
    add(p + "self_attn.v_proj.weight", TensorRole::ATTN_V, d, d);
    add(p + "self_attn.o_proj.weight", TensorRole::ATTN_O, d, d);
    add(p + "mlp.gate_proj.weight",     TensorRole::MLP_GATE, inter, d);
    add(p + "mlp.up_proj.weight",       TensorRole::MLP_UP,   inter, d);
    add(p + "mlp.down_proj.weight",     TensorRole::MLP_DOWN, d, inter);
  }
  return model;
}

static bool buffers_equal(const ModelIR& a, const ModelIR& b) {
  return a.code_buf == b.code_buf && a.scale_buf == b.scale_buf && a.global_buf == b.global_buf;
}
static bool dequant_equal(const ModelIR& a, const ModelIR& b) {
  if (a.tensors.size() != b.tensors.size()) return false;
  for (size_t i = 0; i < a.tensors.size(); ++i) {
    std::vector<float> da, db;
    dashq::dequant_tensor(a, a.tensors[i], da);
    dashq::dequant_tensor(b, b.tensors[i], db);
    if (da != db) return false;
  }
  return true;
}
static bool rotations_equal(const RotationSet& a, const RotationSet& b) {
  if (a.rotations.size() != b.rotations.size()) return false;
  for (size_t i = 0; i < a.rotations.size(); ++i) {
    if (a.rotations[i].id != b.rotations[i].id) return false;
    if (a.rotations[i].site != b.rotations[i].site) return false;
    if (a.rotations[i].dim != b.rotations[i].dim) return false;
    if (a.rotations[i].mat != b.rotations[i].mat) return false;
  }
  return true;
}

int main() {
  printf("=== Stage-1 INT2 pipeline + GGUF / native-SYCL output (t6) ===\n\n");

  ModelIR input = build_model();
  uint64_t full_fp16 = 0;
  for (const auto& t : input.tensors) full_fp16 += (uint64_t)t.N * t.K * 2;
  printf("model: %zu tensors, full FP16 = %.2f MiB\n\n",
         input.tensors.size(), (double)full_fp16 / (1024.0 * 1024.0));

  // ---- Run the full pipeline under a TIGHT budget --------------------------
  PipelineConfig cfg;
  cfg.sq.type = RotationType::Hadamard;
  cfg.sq.per_layer = false;
  cfg.rco.budget_bytes = (uint64_t)(0.22 * (double)full_fp16);  // tight -> bulk reaches INT2 but feasible
  cfg.rco.clear_fp16_after = false;  // keep rotated FP16 for reconstruction check

  PipelineResult res;
  try {
    res = run_pipeline(input, cfg);
  } catch (const std::exception& e) {
    printf("  FAIL: run_pipeline threw: %s\n", e.what());
    g_ok = false;
  }
  if (!g_ok) return 1;
  res.rco_report.print(&res.model);
  CHECK(res.rco_report.feasible, "RCO selection fits the VRAM budget");
  CHECK(res.rco_report.total_bytes <= cfg.rco.budget_bytes, "total bytes <= budget");

  // ---- B. §5 floors + INT2 bulk -------------------------------------------
  bool floors_ok = true, any_int2 = false;
  for (const auto& s : res.rco_report.selections) {
    if (s.scheme == QuantScheme::INT2) any_int2 = true;
    if (s.role == TensorRole::EMBED || s.role == TensorRole::LM_HEAD)
      if (s.scheme == QuantScheme::INT2) floors_ok = false;
    int layer = rco::parse_layer_index(res.model.tensors[s.tensor_index].name);
    if ((s.role == TensorRole::ATTN_Q || s.role == TensorRole::ATTN_K ||
         s.role == TensorRole::ATTN_V) && layer == 0)
      if (s.scheme == QuantScheme::INT2) floors_ok = false;
  }
  CHECK(floors_ok, "§5 min-bit floors honored (embed/lm_head & layer0 attn >= INT4)");
  CHECK(any_int2, "bulk params reach INT2 under tight budget");

  // ---- A. reconstruction quality (dequant vs rotated FP16) -----------------
  float max_rel = 0.f;
  for (const auto& t : res.model.tensors) {
    std::vector<float> dq, fp16;
    dashq::dequant_tensor(res.model, t, dq);
    fp16 = load_tensor_fp16(res.model, t);   // rotated FP16 (clear_fp16_after=false)
    max_rel = std::max(max_rel, rel_err(dq, fp16));
  }
  printf("\n[A] max reconstruction rel-L2 vs rotated FP16 = %.4f %s\n",
         max_rel, max_rel < 0.5f ? "(PASS)" : "(FAIL)");
  CHECK(max_rel < 0.5f, "dequantized weights reconstruct the rotated FP16 input");

  // ---- C. native-SYCL (INTFER1) round-trip --------------------------------
  std::vector<uint8_t> native_blob = serialize_intfer(res.model, res.rotations);
  ModelIR m_native; RotationSet r_native;
  CHECK(deserialize_intfer(native_blob.data(), native_blob.size(), m_native, r_native),
        "native: deserialized OK");
  printf("[C] INTFER1 blob = %.2f KiB | buffers: %s | dequant: %s | rotations: %s\n",
         (double)native_blob.size() / 1024.0,
         buffers_equal(res.model, m_native) ? "PASS" : "FAIL",
         dequant_equal(res.model, m_native) ? "PASS" : "FAIL",
         rotations_equal(res.rotations, r_native) ? "PASS" : "FAIL");
  CHECK(buffers_equal(res.model, m_native), "native: payload buffers byte-identical");
  CHECK(dequant_equal(res.model, m_native), "native: dequant identical");
  CHECK(rotations_equal(res.rotations, r_native), "native: rotation set round-trips");

  // ---- D. GGUF v3 round-trip ----------------------------------------------
  std::vector<uint8_t> gguf_blob = serialize_gguf(res.model, res.rotations);
  ModelIR m_gguf; RotationSet r_gguf;
  CHECK(deserialize_gguf(gguf_blob.data(), gguf_blob.size(), m_gguf, r_gguf),
        "gguf: deserialized OK");
  printf("[D] GGUF v3 blob = %.2f KiB | buffers: %s | dequant: %s | rotations: %s\n",
         (double)gguf_blob.size() / 1024.0,
         buffers_equal(res.model, m_gguf) ? "PASS" : "FAIL",
         dequant_equal(res.model, m_gguf) ? "PASS" : "FAIL",
         rotations_equal(res.rotations, r_gguf) ? "PASS" : "FAIL");
  CHECK(buffers_equal(res.model, m_gguf), "gguf: payload buffers byte-identical");
  CHECK(dequant_equal(res.model, m_gguf), "gguf: dequant identical");
  CHECK(rotations_equal(res.rotations, r_gguf), "gguf: rotation set round-trips");

  // ---- disk IO -------------------------------------------------------------
  CHECK(write_file("/tmp/intfer_t6.gguf", gguf_blob), "write GGUF to disk");
  CHECK(write_file("/tmp/intfer_t6.intfer", native_blob), "write INTFER1 to disk");
  std::vector<uint8_t> disk;
  CHECK(read_file("/tmp/intfer_t6.gguf", disk), "read GGUF from disk");
  ModelIR m_disk; RotationSet r_disk;
  CHECK(deserialize_gguf(disk.data(), disk.size(), m_disk, r_disk), "gguf disk round-trip");
  CHECK(buffers_equal(res.model, m_disk), "gguf disk: payload buffers byte-identical");

  printf("\nACCEPTANCE: INT2 pipeline produces GGUF + native-SYCL artifacts that round-trip: %s\n",
         g_ok ? "PASS" : "FAIL");
  return g_ok ? 0 : 1;
}
