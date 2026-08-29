// examples/spinquant_demo/main.cpp
// Self-test for the Stage-1 SpinQuant orthogonal rotation preprocessing (t3).
// Builds a synthetic transformer, rotates its FP16 weights, and verifies:
//   A. network equivalence (transparency) for right- and left-site rotations
//   B. rotation orthogonality (R R^T = I)
//   C. outlier spreading on a synthetic outlier activation
//   D. rotation serialization round-trip
//   E. learned-rotation path keeps the network equivalent
//
// Pure host code (no SYCL / GPU required) so it can run anywhere.
#include "intfer/spinquant.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>

using namespace intfer;

static std::mt19937 g_gen(12345);
static float rnd() { return std::uniform_real_distribution<float>(-1.f, 1.f)(g_gen); }

// y = x * W^T, x:[T,K], W:[N,K] -> y:[T,N]
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
// y = x * R, x:[T,d], R:[d,d] -> y:[T,d]
static std::vector<float> matmul_Td_dd(const std::vector<float>& x, uint32_t T, uint32_t d,
                                       const std::vector<float>& R) {
  std::vector<float> y(T * d, 0.f);
  for (uint32_t t = 0; t < T; ++t)
    for (uint32_t j = 0; j < d; ++j) {
      float a = 0.f;
      for (uint32_t k = 0; k < d; ++k) a += x[t * d + k] * R[k * d + j];
      y[t * d + j] = a;
    }
  return y;
}
// Relative L2 error (FP16-aware: weights are stored as binary16).
static float rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  float n = 0.f, d = 0.f;
  for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (a[i] - b[i]); d += a[i] * a[i]; }
  return std::sqrt(n) / (std::sqrt(d) + 1e-12f);
}

static bool g_ok = true;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); g_ok = false; } } while(0)

int main() {
  const uint32_t d = 64;       // hidden (power of two -> exact Hadamard)
  const uint32_t inter = 128;  // ffn intermediate
  const uint32_t T = 16;       // tokens

  // ---- Build a synthetic 2-layer model with FP16 weights ----------------
  ModelIR model;
  model.model_id = "synthetic";
  model.n_layers = 2;
  auto add = [&](const std::string& name, TensorRole role, uint32_t N, uint32_t K) {
    TensorIR t;
    t.name = name; t.role = role; t.N = N; t.K = K;
    std::vector<float> w(static_cast<size_t>(N) * K);
    for (auto& v : w) v = rnd();
    store_tensor_fp16(model, t, w);
    model.tensors.push_back(t);
  };
  add("model.embed_tokens.weight", TensorRole::EMBED, 100, d);
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

  // ---- Run SpinQuant (Hadamard) ----------------------------------------
  SpinQuantConfig cfg;
  cfg.type = RotationType::Hadamard;
  cfg.per_layer = false;
  SpinQuantResult res = spinquant_preprocess(model, cfg);

  // ---- B. orthogonality of every rotation ------------------------------
  for (const auto& R : res.rotations.rotations) {
    float off = 0.f;
    for (uint32_t i = 0; i < R.dim; ++i)
      for (uint32_t j = 0; j < R.dim; ++j) {
        float dot = 0.f;
        for (uint32_t k = 0; k < R.dim; ++k) dot += R.mat[i * R.dim + k] * R.mat[j * R.dim + k];
        float expect = (i == j) ? 1.f : 0.f;
        off = std::max(off, std::fabs(dot - expect));
      }
    CHECK(off < 1e-4f, ("rotation id " + std::to_string(R.id) + " not orthogonal").c_str());
  }

  // helper: get rotated W for a tensor name
  auto get_W = [&](const std::string& name) -> std::vector<float> {
    for (const auto& t : res.model.tensors)
      if (t.name == name) return load_tensor_fp16(res.model, t);
    return {};
  };

  // ---- A. transparency: RIGHT site (q_proj) ----------------------------
  {
    std::vector<float> x(T * d); for (auto& v : x) v = rnd();
    std::vector<float> W0 = load_tensor_fp16(model, *std::find_if(model.tensors.begin(), model.tensors.end(),
        [](const TensorIR& t){ return t.name.find("layers.0.self_attn.q_proj") != std::string::npos; }));
    std::vector<float> Wr = get_W("model.layers.0.self_attn.q_proj.weight");
    const Rotation* R = res.rotations.find(
        std::find_if(res.model.tensors.begin(), res.model.tensors.end(),
          [](const TensorIR& t){ return t.name.find("layers.0.self_attn.q_proj") != std::string::npos; })->qcfg.rotation_id);
    std::vector<float> y_ref = matmul_TK_NK(x, T, d, W0, d);
    std::vector<float> xR = matmul_Td_dd(x, T, d, R->mat);
    std::vector<float> y_rot = matmul_TK_NK(xR, T, d, Wr, d);
    CHECK(rel_err(y_ref, y_rot) < 0.02f, "right-site transparency failed");
  }
  // ---- A. transparency: LEFT site (o_proj) -----------------------------
  {
    std::vector<float> x(T * d); for (auto& v : x) v = rnd();
    TensorIR t0 = *std::find_if(model.tensors.begin(), model.tensors.end(),
        [](const TensorIR& t){ return t.name.find("layers.0.self_attn.o_proj") != std::string::npos; });
    std::vector<float> W0 = load_tensor_fp16(model, t0);
    std::vector<float> Wr = get_W("model.layers.0.self_attn.o_proj.weight");
    const Rotation* R = res.rotations.find(
        std::find_if(res.model.tensors.begin(), res.model.tensors.end(),
          [](const TensorIR& t){ return t.name.find("layers.0.self_attn.o_proj") != std::string::npos; })->qcfg.rotation_id);
    // o_proj is left site: W' = R W, online recovery multiplies output by R.
    std::vector<float> y_ref = matmul_TK_NK(x, T, d, W0, d);
    std::vector<float> y_rot = matmul_TK_NK(x, T, d, Wr, d);   // y_rot = x (W')^T
    std::vector<float> y_rec = matmul_Td_dd(y_rot, T, d, R->mat); // recover: y_rec = y_rot R
    CHECK(rel_err(y_ref, y_rec) < 0.02f, "left-site transparency failed");
  }
  // ---- A. transparency: LEFT site (down_proj, N=d, K=inter) ------------
  {
    std::vector<float> x(T * inter); for (auto& v : x) v = rnd();
    TensorIR t0 = *std::find_if(model.tensors.begin(), model.tensors.end(),
        [](const TensorIR& t){ return t.name.find("layers.1.mlp.down_proj") != std::string::npos; });
    std::vector<float> W0 = load_tensor_fp16(model, t0);
    std::vector<float> Wr = get_W("model.layers.1.mlp.down_proj.weight");
    const Rotation* R = res.rotations.find(
        std::find_if(res.model.tensors.begin(), res.model.tensors.end(),
          [](const TensorIR& t){ return t.name.find("layers.1.mlp.down_proj") != std::string::npos; })->qcfg.rotation_id);
    std::vector<float> y_ref = matmul_TK_NK(x, T, inter, W0, d);
    std::vector<float> y_rot = matmul_TK_NK(x, T, inter, Wr, d);
    std::vector<float> y_rec = matmul_Td_dd(y_rot, T, d, R->mat);
    CHECK(rel_err(y_ref, y_rec) < 0.02f, "down(left)-site transparency failed");
  }

  // ---- C. outlier spreading micro-test ---------------------------------
  {
    // Activation with a single huge outlier channel -> rotation should spread it.
    uint32_t d2 = 32;
    std::vector<float> xact(T * d2, 0.01f);
    xact[3 * d2 + 7] = 50.0f;  // strong outlier at token 3, channel 7
    Rotation R = make_hadamard(d2);
    std::vector<float> xr = matmul_Td_dd(xact, T, d2, R.mat);
    float max0 = 0.f, max1 = 0.f;
    for (float v : xact) max0 = std::max(max0, std::fabs(v));
    for (float v : xr)   max1 = std::max(max1, std::fabs(v));
    float var0 = 0.f, var1 = 0.f;
    for (float v : xact) var0 += v * v;
    for (float v : xr)   var1 += v * v;
    // total energy preserved (orthogonal), but peak outlier should shrink.
    printf("  [C] outlier peak %.2f -> %.2f (energy %.2f -> %.2f)\n", max0, max1, var0, var1);
    CHECK(max1 < max0 * 0.9f, "outlier not spread by rotation");
    CHECK(std::fabs(var0 - var1) / var0 < 1e-3f, "rotation changed total energy");
  }

  // ---- D. serialization round-trip -------------------------------------
  {
    std::vector<uint8_t> blob = serialize_rotations(res.rotations);
    RotationSet rs2 = deserialize_rotations(blob.data(), blob.size());
    CHECK(rs2.size() == res.rotations.size(), "serialization lost rotations");
    bool eq = true;
    for (const auto& r : res.rotations.rotations) {
      const Rotation* f = rs2.find(r.id);
      if (!f || f->mat.size() != r.mat.size()) { eq = false; break; }
      for (size_t i = 0; i < r.mat.size(); ++i)
        if (std::fabs(f->mat[i] - r.mat[i]) > 1e-5f) { eq = false; break; }
    }
    CHECK(eq, "serialization mismatch");
  }

  // ---- E. learned path runs & stays transparent ------------------------
  {
    CalibrationData calib;
    for (const auto& t : model.tensors) {
      if (site_for_role(t.role) == RotationSite::NONE) continue;
      uint32_t dim = (side_for_role(t.role) == RotationSide::Right) ? t.K : t.N;
      std::vector<float> acts(T * dim);
      for (auto& v : acts) v = rnd() * (1.0f + 5.0f * (g_gen() % 5 == 0 ? 1.0f : 0.0f));
      calib.activations.emplace_back(t.name, acts);
    }
    SpinQuantConfig lcfg; lcfg.type = RotationType::Learned; lcfg.per_layer = false;
    SpinQuantResult lr = spinquant_preprocess(model, lcfg, &calib);
    // orthogonality
    bool ortho = true;
    for (const auto& R : lr.rotations.rotations) {
      float off = 0.f;
      for (uint32_t i = 0; i < R.dim; ++i)
        for (uint32_t j = 0; j < R.dim; ++j) {
          float dot = 0.f;
          for (uint32_t k = 0; k < R.dim; ++k) dot += R.mat[i*R.dim+k] * R.mat[j*R.dim+k];
          off = std::max(off, std::fabs(dot - (i==j?1.f:0.f)));
        }
      if (off > 1e-3f) ortho = false;
    }
    CHECK(ortho, "learned rotation not orthogonal");
    // transparency for q_proj using learned R
    std::vector<float> x(T * d); for (auto& v : x) v = rnd();
    TensorIR tq = *std::find_if(lr.model.tensors.begin(), lr.model.tensors.end(),
        [](const TensorIR& t){ return t.name.find("layers.0.self_attn.q_proj") != std::string::npos; });
    std::vector<float> W0 = load_tensor_fp16(model, *std::find_if(model.tensors.begin(), model.tensors.end(),
        [](const TensorIR& t){ return t.name.find("layers.0.self_attn.q_proj") != std::string::npos; }));
    std::vector<float> Wr = load_tensor_fp16(lr.model, tq);
    const Rotation* R = lr.rotations.find(tq.qcfg.rotation_id);
    std::vector<float> y_ref = matmul_TK_NK(x, T, d, W0, d);
    std::vector<float> xR = matmul_Td_dd(x, T, d, R->mat);
    std::vector<float> y_rot = matmul_TK_NK(xR, T, d, Wr, d);
    CHECK(rel_err(y_ref, y_rot) < 0.02f, "learned right-site transparency failed");
  }

  printf(g_ok ? "[spinquant_demo] PASS\n" : "[spinquant_demo] FAIL\n");
  return g_ok ? 0 : 1;
}
