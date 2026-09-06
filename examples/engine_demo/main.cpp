// examples/engine_demo/main.cpp
// Stage-2 engine correctness test on the 2x Arc B580:
//   1. load the .intfer and shard its Linear weights across the discovered GPUs,
//   2. run one decode step (one real quantized GEMM, tensor-parallel),
//   3. compare the GPU result against a CPU dequant reference for the same
//      tensor (proves the quantized GEMM + TP sharding are correct on real HW).
#include "intfer/engine.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s <model.intfer>\n", argv[0]); return 1; }
  int ndev = (argc > 3) ? std::atoi(argv[3]) : 0;
  auto eng = intfer::Engine::load(argv[1], ndev);
  if (!eng) { printf("engine: load failed (no Arc GPU / bad file)\n"); return 1; }
  printf("engine: %zu devices, %zu layers\n", eng->num_devices(), eng->num_layers());

  // CPU reference for tensor 0 (rotated weights dequantized to FP32).
  std::vector<uint8_t> blob;
  if (!intfer::read_file(argv[1], blob)) { printf("ref: read fail\n"); return 1; }
  intfer::ModelIR model; intfer::RotationSet rs;
  if (!intfer::deserialize_intfer(blob.data(), blob.size(), model, rs)) {
    printf("ref: deserialize fail\n"); return 1;
  }
  if (model.tensors.empty()) { printf("ref: no tensors\n"); return 1; }
  int tidx = (argc > 2) ? std::atoi(argv[2]) : 0;
  if (tidx < 0 || (size_t)tidx >= model.tensors.size()) tidx = 0;
  const auto& t = model.tensors[tidx];
  std::vector<float> W;
  intfer::dashq::dequant_tensor(model, t, W);   // [N,K], rotated space

  // Varied (non-constant) activation so the zero-mean down_proj produces a
  // well-conditioned, non-trivial output (a constant activation collapses it to
  // ~0 and makes cosine undefined).
  std::vector<float> tok(t.K);
  for (uint32_t k = 0; k < t.K; ++k) tok[k] = 0.02f * (float)((int)(k % 13) - 6);
  std::vector<std::uint16_t> tok_fp16(t.K);
  std::vector<float> A(t.K);
  for (uint32_t k = 0; k < t.K; ++k) {
    tok_fp16[k] = intfer::dashq::float_to_half(tok[k]);
    A[k] = intfer::dashq::half_to_float(tok_fp16[k]);
  }

  // CPU reference: C[n] = sum_k A[k] * W[n,k]
  std::vector<float> ref((size_t)1 * t.N, 0.f);
  for (uint32_t n = 0; n < t.N; ++n)
    for (uint32_t k = 0; k < t.K; ++k)
      ref[n] += A[k] * W[(size_t)n * t.K + k];

  // Engine GPU path (tensor-parallel across the fleet).
  auto out = eng->decode_step(tok, tidx);
  if (out.size() != t.N) { printf("engine: unexpected out size %zu (expect %u)\n", out.size(), t.N); return 1; }

  // ---- CPU replica of shard_weight (device 0, start=0) to localize the NaN ----
  {
    const int bits = (t.qcfg.scheme == intfer::QuantScheme::INT2) ? 2
                   : (t.qcfg.scheme == intfer::QuantScheme::INT4) ? 4 : 8;
    const int pack = (bits >= 8) ? 1 : (8 / bits);
    const uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
    const uint32_t n_groups = (t.K + G - 1) / G;
    const uint16_t* scale_src = reinterpret_cast<const uint16_t*>(model.scale_buf.data() + t.scale_offset);
    float alpha = 1.0f;
    if (t.qcfg.has_global_divisor && t.global_bytes > 0)
      alpha = *reinterpret_cast<const float*>(model.global_buf.data() + t.global_offset);
    auto canon = [&](int code) -> float {
      if (bits >= 8) return (2.f * code - 255.f) / 255.f;
      if (bits == 4) return (2.f * code - 15.f) / 15.f;
      static const float c2[4] = {-1.f, -1.f/3.f, 1.f/3.f, 1.f};
      return c2[code & 3];
    };
    std::vector<float> wscale((size_t)512 * n_groups);
    for (uint32_t r = 0; r < 512; ++r)
      for (uint32_t g = 0; g < n_groups; ++g)
        wscale[(size_t)r * n_groups + g] = intfer::dashq::half_to_float(scale_src[g]);
    std::vector<int> rowcode((size_t)512 * t.K, 0);
    for (uint32_t g = 0; g < n_groups; ++g) {
      uint32_t k0 = g * G, k1 = std::min(k0 + G, t.K);
      uint32_t Glocal = k1 - k0;
      for (uint32_t r = 0; r < 512; ++r)
        for (uint32_t c = k0; c < k1; ++c) {
          size_t src_idx = (size_t)g * (t.N * Glocal) + (size_t)r * Glocal + (c - k0);
          const uint8_t b = model.code_buf[t.code_offset + src_idx / pack];
          const int within = (int)(src_idx % pack);
          rowcode[(size_t)r * t.K + c] = (b >> (within * bits)) & ((1u << bits) - 1);
        }
    }
    float s_src0 = intfer::dashq::half_to_float(scale_src[0]);
    printf("DEBUG scale_src[0]=%.6f alpha=%.6f bits=%d n_groups=%u G=%u code[0*K+0..3]=%d,%d,%d,%d\n",
           s_src0, alpha, bits, n_groups, G,
           rowcode[0], rowcode[1], rowcode[2], rowcode[3]);
    for (int n = 0; n < 4; ++n) {
      float acc = 0;
      for (uint32_t k = 0; k < t.K; ++k) {
        uint32_t gg = k / G;
        float s = wscale[(size_t)n * n_groups + gg] / alpha;
        acc += A[k] * s * canon(rowcode[(size_t)n * t.K + k]);
      }
      printf("DEBUG sharded-CPU C[%d]=%.6f  gpu=%.6f  ref=%.6f\n", n, acc, out[n], ref[n]);
    }
    // Repack into code_slice exactly like shard_weight, then read back via the
    // kernel's exact index formula. Isolates the repack/relayout bug.
    std::vector<uint8_t> code_slice((size_t)512 * ((t.K + pack - 1) / pack), 0);
    for (size_t i = 0; i < rowcode.size(); ++i) {
      size_t pidx = i / pack;
      int within = (int)(i % pack);
      code_slice[pidx] |= (uint8_t)((rowcode[i] & ((1u << bits) - 1u)) << (within * bits));
    }
    for (int n = 0; n < 4; ++n) {
      float acc = 0; float bad = 0;
      for (uint32_t k = 0; k < t.K; ++k) {
        size_t idx = (size_t)n * t.K + k;
        const uint8_t b = code_slice[idx / pack];
        int within = (int)(idx % pack);
        int code = (b >> (within * bits)) & ((1u << bits) - 1u);
        if (code < 0 || code >= (1 << bits)) bad = 1;
        uint32_t gg = k / G;
        float s = wscale[(size_t)n * n_groups + gg] / alpha;
        acc += A[k] * s * canon(code);
      }
      printf("DEBUG via-code_slice C[%d]=%.6f  (code-range-ok=%d)\n", n, acc, bad == 0);
    }
  }

  double cos = cosine(ref, out);
  double maxerr = 0;
  size_t non = 0, nref = 0; double on = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::isnan(out[i])) ++non;
    if (std::isnan(ref[i])) ++nref;
    on += (double)out[i] * (double)out[i];
    maxerr = std::max<double>(maxerr, std::fabs(ref[i] - out[i]));
  }
  // Relative error normalized by the reference output magnitude.
  double refn = 0; for (float v : ref) refn += (double)v * v; refn = std::sqrt(refn);
  double relerr = maxerr / (refn + 1e-6);
  printf("tensor0 : %s  N=%u K=%u scheme=%d\n", t.name.c_str(), t.N, t.K, (int)t.qcfg.scheme);
  printf("|out|=%.4f  nout_nan=%zu  nref_nan=%zu\n", std::sqrt(on), non, nref);
  for (size_t i = 0; i < out.size(); ++i)
    if (std::isnan(out[i])) {
      printf("FIRST NAN at out[%zu]; ref=%.5f; out[%zu..%zu]=%.4f,%.4f,%.4f,%.4f\n",
             i, ref[i], (i>1?i-2:0), (i+2<out.size()?i+2:i),
             out[i>1?i-2:0], out[i>0?i-1:0],
             (i+1<out.size()?out[i+1]:0.f), (i+2<out.size()?out[i+2]:0.f));
      break;
    }
  printf("cosine(ref, gpu) = %.6f   max|err| = %.6f   |ref| = %.4f   rel-err = %.4e\n",
         cos, maxerr, refn, relerr);
  bool pass = (cos > 0.99) && (relerr < 0.05);
  printf("%s\n", pass ? "PASS: engine quantized GEMM matches CPU reference on Arc B580"
                      : "FAIL: mismatch");
  return pass ? 0 : 1;
}
