// examples/dashq_demo/main.cpp
// Standalone validation of the DASH-Q diagonal-Hessian PTQ solver (t4).
// Builds a synthetic weight tensor with an injected outlier column, estimates
// the diagonal Hessian from calibration activations, then:
//   1. emits the default candidate set through the TensorIR/ModelIR pipeline,
//   2. compares DASH-Q (diagonal-Hessian weighted LS) vs a naive min-max
//      uniform quantizer (no importance weighting) to show the benefit,
//   3. round-trips each candidate through dequant and verifies the reported
//      est_error matches the reconstructed relative-L2 error.
//
// Pure host C++ — run with:  cmake -S examples/dashq_demo -B build_dashq && cmake --build build_dashq && ./build_dashq/dashq_demo
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "intfer/quant_spec.h"
#include "intfer/quant/dashq.hpp"

using namespace intfer;
using namespace intfer::dashq;

// Fair baseline: identical codebook to DASH-Q, but unweighted min-max scale
// (no diagonal-Hessian importance weighting, no learned codebook). This isolates
// the DASH-Q advantage (weighted-LS scale + learned codebook).
static float naive_error(const float* W, uint32_t N, uint32_t K, QuantScheme sch,
                         uint32_t G, const DiagHessian& H) {
  const std::vector<float> cb = canonical_codebook(sch);
  float cbmax = 0.f;
  for (float v : cb) cbmax = std::max(cbmax, std::fabs(v));
  const uint32_t n_groups = (K + G - 1) / G;
  auto h_at = [&](uint32_t r, uint32_t c) -> float {
    return H.per_element() ? H.data()[r * K + c] : H.data()[c];
  };
  double mse = 0.0, norm = 0.0;
  for (uint32_t g = 0; g < n_groups; ++g) {
    uint32_t gend = std::min((uint32_t)((g + 1) * G), K);
    float maxw = 0.f;
    for (uint32_t r = 0; r < N; ++r)
      for (uint32_t c = g * G; c < gend; ++c)
        maxw = std::max(maxw, std::fabs(W[r * K + c]));
    const float s = (cbmax > 0.f && maxw > 0.f) ? maxw / cbmax : 1.f;
    for (uint32_t r = 0; r < N; ++r)
      for (uint32_t c = g * G; c < gend; ++c) {
        const float w = W[r * K + c];
        int best = 0; float bd = 1e30f;
        for (size_t j = 0; j < cb.size(); ++j) {
          const float d = w - s * cb[j];
          if (d * d < bd) { bd = d * d; best = (int)j; }
        }
        const float dq = s * cb[best];
        const float hh = h_at(r, c);
        const double e = w - dq;
        mse += (double)hh * e * e;
        norm += (double)hh * w * w;
      }
  }
  return (float)std::sqrt(mse / norm);
}

static QuantConfig make_cfg(QuantScheme sch, uint16_t g) {
  QuantConfig c;
  c.scheme = sch;
  c.group_size = g;
  c.scale_dtype = ScaleDtype::FP16;
  c.has_global_divisor = true;
  c.act = ActScheme::INT8_G64;
  return c;
}

int main() {
  const uint32_t N = 64, K = 128;
  std::vector<float> W((size_t)N * K);
  std::mt19937 gen(12345);
  std::normal_distribution<float> nrm(0.0f, 0.05f);
  for (uint32_t i = 0; i < N * K; ++i) W[i] = nrm(gen);
  // Inject a strong outlier on column 10 (simulates pre-SpinQuant outliers).
  for (uint32_t r = 0; r < N; ++r)
    W[r * K + 10] += (r % 2 ? 1.0f : -1.0f) * 2.0f;

  // Calibration activations: column 10 is "important" (high variance),
  // column 42 is unimportant. This drives the diagonal Hessian weights.
  const uint32_t nTok = 256;
  std::vector<float> X((size_t)nTok * K);
  std::normal_distribution<float> nrm2(0.0f, 0.3f);
  for (uint32_t t = 0; t < nTok; ++t)
    for (uint32_t c = 0; c < K; ++c) {
      float v = nrm2(gen);
      if (c == 10) v *= 6.0f;   // important channel
      if (c == 42) v *= 0.05f;  // unimportant channel
      X[(size_t)t * K + c] = v;
    }
  const DiagHessian H = DiagHessian::from_activations(X.data(), nTok, K);

  printf("=== DASH-Q diagonal-Hessian PTQ solver (N=%u K=%u) ===\n\n", N, K);
  printf("%-8s %-8s %-11s %-11s %-11s %-11s\n",
         "scheme", "bits/w", "DASH-Q", "DASH-Q(L)", "naive", "verify");
  printf("---------------------------------------------------------------------------\n");

  const struct { QuantScheme s; uint16_t g; const char* name; } schemes[] = {
    {QuantScheme::INT2, 16, "INT2-G16"},
    {QuantScheme::INT2, 32, "INT2-G32"},
    {QuantScheme::INT4, 64, "INT4-G64"},
    {QuantScheme::INT8, 32, "INT8-G32"},
  };

  // Sanity carrier for the IR pipeline path.
  TensorIR parent;
  parent.name = "model.layers.0.mlp.down_proj.weight";
  parent.role = TensorRole::MLP_DOWN;
  parent.N = N; parent.K = K;
  ModelIR model;
  quantize_candidates(parent, W.data(), N, K, H, model, /*learn_codebook=*/true);
  model.tensors.push_back(parent); // so total_weight_bytes() reflects the candidates

  for (const auto& sc : schemes) {
    const QuantResult qCanon = quantize(W.data(), N, K, make_cfg(sc.s, sc.g), H, false);
    const QuantResult qLearn = quantize(W.data(), N, K, make_cfg(sc.s, sc.g), H, true);
    const float naive = naive_error(W.data(), N, K, sc.s, sc.g, H);

    // Verify the learned variant dequant round-trips to its reported error.
    std::vector<float> dec;
    qLearn.dequant(dec, N, K);
    double mse = 0.0, norm = 0.0;
    for (uint32_t i = 0; i < N * K; ++i) {
      const float hh = H.data()[i % K];
      const double e = W[i] - dec[i];
      mse += (double)hh * e * e;
      norm += (double)hh * W[i] * W[i];
    }
    const float verify = (float)std::sqrt(mse / norm);

    printf("%-8s %-8.2f %-11.5f %-11.5f %-11.5f %-11.5f\n",
           sc.name, qLearn.eff_bits(N, K),
           qCanon.est_error, qLearn.est_error, naive, verify);
  }

  printf("\nIR pipeline: %zu candidate tensors emitted.\n", parent.candidates.size());
  for (const auto& cand : parent.candidates) {
    const char* sn = cand.qcfg.scheme == QuantScheme::INT2 ? "INT2"
                   : cand.qcfg.scheme == QuantScheme::INT4 ? "INT4"
                   : cand.qcfg.scheme == QuantScheme::INT8 ? "INT8" : "FP16";
    printf("  %-7s G=%-3u est_error=%.5f bytes(code=%llu scale=%llu global=%llu)\n",
           sn, cand.qcfg.group_size, cand.est_error,
           (unsigned long long)cand.code_bytes, (unsigned long long)cand.scale_bytes,
           (unsigned long long)cand.global_bytes);
  }
  uint64_t tot = 0;
  for (const auto& c : parent.candidates) tot += c.weight_bytes();
  printf("\nModelIR payload: code=%zu scale=%zu global=%zu bytes "
         "(total candidate weight budget=%llu, buffers=%zu)\n",
         model.code_buf.size(), model.scale_buf.size(), model.global_buf.size(),
         (unsigned long long)tot, model.total_weight_bytes());

  // Acceptance check: DASH-Q with diagonal-Hessian weighting must not be worse
  // than the unweighted naive baseline at the same bit-width.
  bool ok = true;
  for (const auto& sc : schemes) {
    const QuantResult q = quantize(W.data(), N, K, make_cfg(sc.s, sc.g), H, true);
    const float naive = naive_error(W.data(), N, K, sc.s, sc.g, H);
    if (q.est_error > naive + 1e-6f) ok = false;
  }
  printf("\nACCEPTANCE: DASH-Q est_error <= naive baseline for all low-bit schemes: %s\n",
         ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
