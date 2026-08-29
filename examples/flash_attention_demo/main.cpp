// flash_attention_demo — validates the intfer fused FlashAttention SYCL
// kernels on the local Intel Arc fleet (B580) against an FP32 reference and
// reports kernel latency.
//
// Two kernels are exercised:
//   * flash_attention_forward_f16  (FP16 Q/K/V, FlashAttention-2 online softmax)
//   * flash_attention_forward_int8_kv (groupwise-INT8 KV cache, dequant on the fly)
// Both are compared to a host FP32 reference.

#include "intfer/attention/flash_attention.hpp"
#include "intfer/common/logger.hpp"
#include "intfer/device/device_context.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

std::uint16_t to_half(float f) {
  return sycl::bit_cast<std::uint16_t>(sycl::half(f));
}
float from_half(std::uint16_t h) {
  return static_cast<float>(sycl::bit_cast<sycl::half>(h));
}

// Symmetric per-(head, token-group) INT8 quantization of a K/V tensor.
// group_size tokens share one FP32 scale = max_abs/127 over the group.
void quantize_int8_kv(const std::vector<float>& kv, int B, int H, int S, int D,
                      int group_size, std::vector<std::int8_t>& out,
                      std::vector<float>& scales) {
  const int num_groups = (S + group_size - 1) / group_size;
  out.assign(B * H * S * D, 0);
  scales.assign(B * H * num_groups, 0.0f);

  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int g = 0; g < num_groups; ++g) {
        const int t0 = g * group_size;
        const int t1 = std::min(t0 + group_size, S);
        float max_abs = 1e-20f;
        for (int t = t0; t < t1; ++t)
          for (int d = 0; d < D; ++d) {
            const float v = kv[((b * H + h) * S + t) * D + d];
            max_abs = std::max(max_abs, std::fabs(v));
          }
        const float scale = max_abs / 127.0f;
        scales[(b * H + h) * num_groups + g] = scale;
        for (int t = t0; t < t1; ++t)
          for (int d = 0; d < D; ++d) {
            const float v = kv[((b * H + h) * S + t) * D + d];
            int q = static_cast<int>(std::round(v / scale));
            q = std::max(-127, std::min(127, q));
            out[((b * H + h) * S + t) * D + d] = static_cast<std::int8_t>(q);
          }
      }
}

double max_abs_error(const std::vector<float>& ref,
                     const std::vector<std::uint16_t>& got, int n) {
  double e = 0.0;
  for (int i = 0; i < n; ++i)
    e = std::max(e, (double)std::fabs((double)ref[i] - (double)from_half(got[i])));
  return e;
}

double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

template <typename Fn>
double timed_ms(int iters, Fn&& fn) {
  // Warmup.
  fn();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

bool test_f16(intfer::device::DeviceContext& ctx, int S) {
  using namespace intfer::attention;
  constexpr int B = 1, H = 4, D = 64;
  const int N = B * H * S * D;

  std::mt19937 gen(1234);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> qf(N), kf(N), vf(N);
  for (int i = 0; i < N; ++i) {
    qf[i] = dist(gen);
    kf[i] = dist(gen);
    vf[i] = dist(gen);
  }

  std::vector<std::uint16_t> qh(N), kh(N), vh(N);
  for (int i = 0; i < N; ++i) {
    qh[i] = to_half(qf[i]);
    kh[i] = to_half(kf[i]);
    vh[i] = to_half(vf[i]);
  }

  FlashAttentionParams p{B, H, S, S, D, 1.0f / std::sqrt(float(D)), true};

  std::vector<std::uint16_t> out(N);
  {
    sycl::buffer<std::uint16_t> qb(qh), kb(kh), vb(vh), ob(out);
    flash_attention_forward_f16(ctx.queue, qb, kb, vb, ob, p);
  }

  std::vector<float> ref;
  flash_attention_reference_f32(qf, kf, vf, ref, p);

  const double err = max_abs_error(ref, out, N);
  {
    std::vector<float> out_f(out.size());
    for (size_t i = 0; i < out.size(); ++i) out_f[i] = from_half(out[i]);
    const double cos = cosine_sim(ref, out_f);
    const double lat = timed_ms(50, [&] {
      sycl::buffer<std::uint16_t> qb(qh), kb(kh), vb(vh), ob(out);
      flash_attention_forward_f16(ctx.queue, qb, kb, vb, ob, p);
    });
    INTFER_LOG_INFO("  [f16 S=%d]  max|err|=%.5f  cosine=%.6f  latency=%.3f ms", S, err, cos, lat);
    if (!(err < 0.05 && cos > 0.999)) return false;
  }
  return true;
}

bool run_f16(intfer::device::DeviceContext& ctx) {
  // Aligned (S multiple of tile 64) and non-aligned (exercises padding/mask).
  return test_f16(ctx, 256) && test_f16(ctx, 200);
}

bool run_int8_kv(intfer::device::DeviceContext& ctx) {
  using namespace intfer::attention;
  constexpr int B = 1, H = 4, S = 256, D = 64, G = 32;
  const int N = B * H * S * D;
  const int num_groups = (S + G - 1) / G;

  std::mt19937 gen(5678);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> qf(N), kf(N), vf(N);
  for (int i = 0; i < N; ++i) {
    qf[i] = dist(gen);
    kf[i] = dist(gen);
    vf[i] = dist(gen);
  }

  // Q stays FP16; K/V become groupwise-INT8.
  std::vector<std::uint16_t> qh(N);
  for (int i = 0; i < N; ++i) qh[i] = to_half(qf[i]);
  std::vector<std::int8_t> kq, vq;
  std::vector<float> ks, vs;
  quantize_int8_kv(kf, B, H, S, D, G, kq, ks);
  quantize_int8_kv(vf, B, H, S, D, G, vq, vs);

  FlashAttentionParams p{B, H, S, S, D, 1.0f / std::sqrt(float(D)), true};

  std::vector<std::uint16_t> out(N);
  {
    sycl::buffer<std::uint16_t> qb(qh);
    sycl::buffer<std::int8_t> kb(kq), vb(vq);
    sycl::buffer<float> ksb(ks), vsb(vs);
    sycl::buffer<std::uint16_t> ob(out);
    flash_attention_forward_int8_kv(ctx.queue, qb, kb, vb, ksb, vsb, ob, p, G);
  }

  // Reference on the *dequantized* K/V to isolate kernel correctness.
  std::vector<float> kdq(N), vdq(N);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < S; ++t) {
        const float s_k = ks[(b * H + h) * num_groups + t / G];
        const float s_v = vs[(b * H + h) * num_groups + t / G];
        for (int d = 0; d < D; ++d) {
          kdq[((b * H + h) * S + t) * D + d] =
              s_k * kq[((b * H + h) * S + t) * D + d];
          vdq[((b * H + h) * S + t) * D + d] =
              s_v * vq[((b * H + h) * S + t) * D + d];
        }
      }
  std::vector<float> ref;
  flash_attention_reference_f32(qf, kdq, vdq, ref, p);

  const double err = max_abs_error(ref, out, N);
  std::vector<float> out_f(out.size());
  for (size_t i = 0; i < out.size(); ++i) out_f[i] = from_half(out[i]);
  const double cos = cosine_sim(ref, out_f);
  const double lat = timed_ms(50, [&] {
    sycl::buffer<std::uint16_t> qb(qh);
    sycl::buffer<std::int8_t> kb(kq), vb(vq);
    sycl::buffer<float> ksb(ks), vsb(vs);
    sycl::buffer<std::uint16_t> ob(out);
    flash_attention_forward_int8_kv(ctx.queue, qb, kb, vb, ksb, vsb, ob, p, G);
  });

  INTFER_LOG_INFO("  [int8kv] max|err|=%.5f  cosine=%.6f  latency=%.3f ms", err, cos, lat);
  // INT8 KV introduces quantization noise; allow a looser tolerance.
  return err < 0.08 && cos > 0.99;
}

}  // namespace

int main() {
  auto mgr = intfer::device::DeviceManager::discover("Arc");
  if (mgr.size() == 0) {
    INTFER_LOG_ERROR("No Intel Arc GPUs discovered.");
    return 1;
  }
  mgr.print_summary();
  auto& ctx = mgr[0];
  INTFER_LOG_INFO("Running fused FlashAttention on GPU[0] '%s' ...",
                  ctx.name.c_str());

  bool ok_f16 = run_f16(ctx);
  bool ok_int8 = run_int8_kv(ctx);

  if (ok_f16 && ok_int8) {
    INTFER_LOG_INFO("FlashAttention SYCL kernels: PASS (both variants match reference).");
    return 0;
  }
  INTFER_LOG_ERROR("FlashAttention SYCL kernels: FAIL (f16=%s int8=%s).",
                   ok_f16 ? "ok" : "FAIL", ok_int8 ? "ok" : "FAIL");
  return 1;
}
