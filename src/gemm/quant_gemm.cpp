// src/gemm/quant_gemm.cpp
//
// Low-level SYCL quantized GEMM kernels for Intel Arc (Xe-core / XMX).
// See include/intfer/gemm/quant_gemm.hpp for the interface contract.
//
//   (1) quant_gemm_forward       - dequant-on-the-fly GEMM (portable SYCL,
//                                  FP32 accum via FP16 MMA -> XMX). Handles
//                                  INT2/INT4/INT8 weights, FP16 or groupwise
//                                  INT8 activations, per-group FP16 scales and
//                                  an optional per-tensor global divisor.
//   (2) quant_gemm_dpas_int8     - ESIMD kernel issuing the Intel XMX DPAS
//                                  (Dot Product Accumulate Systolic) instruction
//                                  directly for the INT8 x INT8 case.

#include "intfer/gemm/quant_gemm.hpp"

#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

#include <vector>

// ESIMD / XMX DPAS intrinsics (header-only; only used by the DPAS path below).
#include <sycl/ext/intel/esimd.hpp>

namespace intfer::gemm {
namespace {

inline float lf(sycl::half h) { return static_cast<float>(h); }
inline std::uint16_t to_half_bits(float f) {
  return sycl::bit_cast<std::uint16_t>(sycl::half(f));
}
inline float from_half_bits(std::uint16_t h) {
  return static_cast<float>(sycl::bit_cast<sycl::half>(h));
}

// Canonical DASH-Q codebook value for an unsigned code index `code`.
// The Stage-1 pipeline stores LOW-BIT codes as *indices* into this codebook
// (canonical_codebook() in dashq.cpp); the dequantized weight is scale*cb[code].
//   INT2 : {-1, -1/3, +1/3, +1}
//   INT4 : uniform symmetric 15 levels in [-1, 1]:  (2j-15)/15
//   INT8 : uniform symmetric 255 levels in [-1, 1]: (2j-255)/255
inline float canon_value(int bits, int code) {
  if (bits >= 8) return (2.0f * (float)code - 255.0f) / 255.0f;
  if (bits == 4) return (2.0f * (float)code - 15.0f) / 15.0f;
  static const float c2[4] = {-1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f};
  return c2[code & 3];
}

// Tile geometry (subgroup = 16x16 work-items, each covers a 4x4 output sub-tile).
constexpr int kBM = 64;   // output rows per work-group
constexpr int kBN = 64;   // output cols per work-group
constexpr int kBK = 64;   // K per iteration
constexpr int kThr = 16;  // threads per tile dimension
constexpr int kSub = 4;   // sub-tile per thread

}  // namespace

// ---------------------------------------------------------------------------
// Host FP32 reference.
// ---------------------------------------------------------------------------
void gemm_reference_f32(const std::vector<float>& A,
                        const std::vector<float>& W,
                        std::vector<float>& C,
                        uint32_t M, uint32_t N, uint32_t K) {
  C.assign(static_cast<size_t>(M) * N, 0.0f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k)
        acc += A[static_cast<size_t>(m) * K + k] *
               W[static_cast<size_t>(n) * K + k];
      C[static_cast<size_t>(m) * N + n] = acc;
    }
}

// ---------------------------------------------------------------------------
// (1) Dequant-on-the-fly SYCL GEMM.
// ---------------------------------------------------------------------------
void quant_gemm_forward(sycl::queue& q,
                        GemmActBuffers& a,
                        GemmWeightBuffers& w,
                        sycl::buffer<std::uint16_t>& c,
                        const QuantGemmOptions& opt) {
  const uint32_t M = a.M, N = w.N, K = w.K;
  const int bits = (w.scheme == WeightScheme::INT2) ? 2
                 : (w.scheme == WeightScheme::INT4) ? 4
                 : (w.scheme == WeightScheme::FP16) ? 16 : 8;
  const int Gw = (w.group_size == 0) ? static_cast<int>(K) : w.group_size;
  const int Ga = (a.group_size == 0) ? static_cast<int>(K)
                                     : static_cast<int>(a.group_size);
  const int num_groups_w = (static_cast<int>(K) + Gw - 1) / Gw;
  const int num_groups_a = (static_cast<int>(K) + Ga - 1) / Ga;
  const float alpha = w.has_global_divisor
                          ? w.global.get_access<sycl::access::mode::read>()[0]
                          : 1.0f;

  // Eligible for the literal DPAS path?
  const bool dpas_eligible =
      opt.use_dpas && w.scheme == WeightScheme::INT8 &&
      a.scheme == ActScheme::INT8 && (K % 32 == 0) && (N % 16 == 0) &&
      (M % 8 == 0) && w.group_size == 0 && a.group_size == 0;

  if (dpas_eligible) {
    // Extract INT8 payload into USM and dispatch the DPAS kernel.
    auto code_h = w.code.get_access<sycl::access::mode::read>();
    auto ascale_h = a.scale.get_access<sycl::access::mode::read>();
    const float a_scale = from_half_bits(ascale_h[0]);  // per-tensor activation

    auto wscale_h = w.scale.get_access<sycl::access::mode::read>();
    std::vector<float> w_scale_f(N);
    for (uint32_t n = 0; n < N; ++n)
      w_scale_f[n] = from_half_bits(wscale_h[n]);

    auto aint8_h = a.int8.get_access<sycl::access::mode::read>();
    std::vector<std::int8_t> a_int8(M * K);
    for (size_t i = 0; i < a_int8.size(); ++i) a_int8[i] = aint8_h[i];

    std::vector<std::int8_t> w_int8(N * K);
    for (size_t i = 0; i < w_int8.size(); ++i) w_int8[i] = code_h[i];
    std::vector<std::int8_t> w_vnni(N * K);
    build_vnni_int8(w_int8.data(), w_vnni.data(), N, K);

    auto* a_dev = sycl::malloc_device<std::int8_t>(M * K, q);
    auto* w_dev = sycl::malloc_device<std::int8_t>(N * K, q);
    auto* ws_dev = sycl::malloc_device<float>(N, q);
    auto* c_dev = sycl::malloc_device<std::uint16_t>(M * N, q);
    q.memcpy(a_dev, a_int8.data(), M * K * sizeof(std::int8_t)).wait();
    q.memcpy(w_dev, w_vnni.data(), N * K * sizeof(std::int8_t)).wait();
    q.memcpy(ws_dev, w_scale_f.data(), N * sizeof(float)).wait();

    quant_gemm_dpas_int8(q, a_dev, a_scale, w_dev, ws_dev, c_dev, M, N, K);

    std::vector<std::uint16_t> c_tmp(M * N);
    q.memcpy(c_tmp.data(), c_dev, M * N * sizeof(std::uint16_t)).wait();
    auto c_h = c.get_access<sycl::access::mode::write>();
    for (size_t i = 0; i < c_tmp.size(); ++i) c_h[i] = c_tmp[i];
    sycl::free(a_dev, q);
    sycl::free(w_dev, q);
    sycl::free(ws_dev, q);
    sycl::free(c_dev, q);
    return;
  }

  const bool act_int8 = (a.scheme == ActScheme::INT8);

  // Dummy buffers back the inactive operand path so every accessor is valid.
  sycl::buffer<std::uint16_t> dummy_h(1);
  sycl::buffer<std::int8_t> dummy_i(1);
  sycl::buffer<std::uint16_t> dummy_s(1);

  const int gx = (static_cast<int>(M) + kBM - 1) / kBM;
  const int gy = (static_cast<int>(N) + kBN - 1) / kBN;

  sycl::nd_range<2> nd(sycl::range<2>(gx * kThr, gy * kThr),
                       sycl::range<2>(kThr, kThr));

  q.submit([&](sycl::handler& h) {
    auto a16k = (a.scheme == ActScheme::FP16)
                    ? a.fp16.get_access<sycl::access::mode::read>(h)
                    : dummy_h.get_access<sycl::access::mode::read>(h);
    auto ai8k = (act_int8) ? a.int8.get_access<sycl::access::mode::read>(h)
                           : dummy_i.get_access<sycl::access::mode::read>(h);
    auto asclk = (act_int8) ? a.scale.get_access<sycl::access::mode::read>(h)
                            : dummy_s.get_access<sycl::access::mode::read>(h);
    auto wck = w.code.get_access<sycl::access::mode::read>(h);
    auto wsk = w.scale.get_access<sycl::access::mode::read>(h);
    auto ock = c.get_access<sycl::access::mode::write>(h);

    sycl::local_accessor<std::uint16_t, 1> sA(sycl::range<1>(kBM * kBK), h);
    sycl::local_accessor<std::uint16_t, 1> sW(sycl::range<1>(kBN * kBK), h);

    h.parallel_for(nd, [=](sycl::nd_item<2> it) {
      const int itm = it.get_global_id(0);
      const int itn = it.get_global_id(1);
      const int bm = itm / kThr, bn = itn / kThr;
      const int ty = itm % kThr, tx = itn % kThr;
      const int lid = ty * kThr + tx;

      float acc[kSub * kSub] = {0};

      const int nk_blocks = (static_cast<int>(K) + kBK - 1) / kBK;
      for (int bk = 0; bk < nk_blocks; ++bk) {
        for (int e = lid; e < kBM * kBK; e += kThr * kThr) {
          const int r = e / kBK, c = e % kBK;
          const int mg = bm * kBM + r;
          const int kg = bk * kBK + c;
          std::uint16_t val = 0;
          if (mg < static_cast<int>(M) && kg < static_cast<int>(K)) {
            if (act_int8) {
              const int g = kg / Ga;
              const float s = from_half_bits(asclk[mg * num_groups_a + g]);
              val = to_half_bits(s * static_cast<float>(ai8k[mg * K + kg]));
            } else {
              val = a16k[mg * K + kg];
            }
          }
          sA[e] = val;
        }
        for (int e = lid; e < kBN * kBK; e += kThr * kThr) {
          const int r = e / kBK, c = e % kBK;
          const int ng = bn * kBN + r;
          const int kg = bk * kBK + c;
          std::uint16_t val = 0;
          if (ng < static_cast<int>(N) && kg < static_cast<int>(K)) {
            const std::size_t idx = static_cast<std::size_t>(ng) * K + kg;
            const int g = kg / Gw;
            const float s = from_half_bits(wsk[ng * num_groups_w + g]) * alpha;
            float wval;
            if (bits >= 16) {
              // FP16 weight: raw half bits.
              const std::uint16_t hv = (std::uint16_t)(
                  wck[idx * 2] | ((std::uint16_t)wck[idx * 2 + 1] << 8));
              wval = from_half_bits(hv);
            } else {
              int code;
              if (bits >= 8) {
                code = static_cast<int>(wck[idx]);  // unsigned 0..255 index
              } else {
                const int pack = 8 / bits;
                const std::uint8_t b = wck[idx / pack];
                const int within = static_cast<int>(idx % pack);
                code = (b >> (within * bits)) & ((1 << bits) - 1);  // unsigned idx
              }
              wval = s * canon_value(bits, code);
            }
            val = to_half_bits(wval);
          }
          sW[e] = val;
        }
        it.barrier();

        for (int kk = 0; kk < kBK; ++kk) {
          for (int i = 0; i < kSub; ++i)
            for (int j = 0; j < kSub; ++j) {
              const float av = lf(sycl::bit_cast<sycl::half>(sA[(ty * kSub + i) * kBK + kk]));
              const float wv = lf(sycl::bit_cast<sycl::half>(sW[(tx * kSub + j) * kBK + kk]));
              acc[i * kSub + j] += av * wv;
            }
        }
        it.barrier();
      }

      for (int i = 0; i < kSub; ++i)
        for (int j = 0; j < kSub; ++j) {
          const int mg = bm * kBM + ty * kSub + i;
          const int ng = bn * kBN + tx * kSub + j;
          if (mg < static_cast<int>(M) && ng < static_cast<int>(N))
            ock[static_cast<std::size_t>(mg) * N + ng] = to_half_bits(acc[i * kSub + j]);
        }
    });
  });
  q.wait();
}

// ---------------------------------------------------------------------------
// USM-based dequant-on-the-fly GEMM (no buffer construction overhead).
// All data lives in device USM; the activation is copied in via q.copy.
//
// Returns a SYCL event that signals when the output USM buffer (out_usm)
// has been written. Callers can wait on the returned event to read the output.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_forward_usm(sycl::queue& q,
                                    std::uint16_t* act_usm,
                                    const void* code_usm,
                                    const std::uint16_t* scale_usm,
                                    float alpha,
                                    std::uint16_t* out_usm,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    uint32_t group_size,
                                    gemm::WeightScheme scheme,
                                    const sycl::event& copy_dep) {
  const int bits = (scheme == WeightScheme::INT2) ? 2
                 : (scheme == WeightScheme::INT4) ? 4
                 : (scheme == WeightScheme::FP16) ? 16 : 8;
  const int Gw = (group_size == 0) ? static_cast<int>(K) : static_cast<int>(group_size);
  const int num_groups_w = (static_cast<int>(K) + Gw - 1) / Gw;
  const int nk_blocks = (static_cast<int>(K) + kBK - 1) / kBK;
  const auto* code_u8 = static_cast<const std::uint8_t*>(code_usm);

  // TODO 6: INT8 DPAS fast path. Quantize FP16 activations to INT8 and
  // dispatch the XMX DPAS systolic path when shapes are compatible.
  if (scheme == WeightScheme::INT8 && M >= 8 && (K % 32 == 0) && (N % 16 == 0) && group_size == 0) {
    std::vector<std::int8_t> a_int8(M * K);
    float a_scale = 0.0f;
    for (uint32_t i = 0; i < M * K; ++i) {
      const float v = static_cast<float>(sycl::bit_cast<sycl::half>(act_usm[i]));
      a_scale = std::max(a_scale, std::abs(v));
    }
    a_scale = std::max(a_scale, 1e-6f) / 127.0f;
    for (uint32_t i = 0; i < static_cast<uint32_t>(M * K); ++i) {
      const float v = static_cast<float>(sycl::bit_cast<sycl::half>(act_usm[i]));
      a_int8[i] = static_cast<std::int8_t>(
          std::round(std::clamp(v / a_scale, -127.0f, 127.0f)));
    }

    std::vector<std::int8_t> w_int8(static_cast<size_t>(N) * K);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k)
        w_int8[static_cast<size_t>(k) * N + n] =
            static_cast<std::int8_t>(code_u8[static_cast<size_t>(n) * K + k]);
    std::vector<std::int8_t> w_vnni(static_cast<size_t>(N) * K);
    build_vnni_int8(w_int8.data(), w_vnni.data(), N, K);

    std::vector<float> w_scale_f(N);
    for (uint32_t n = 0; n < N; ++n)
      w_scale_f[n] = from_half_bits(scale_usm[n]) * alpha;

    auto* a_dev = sycl::malloc_device<std::int8_t>(static_cast<size_t>(M) * K, q);
    auto* w_dev = sycl::malloc_device<std::int8_t>(static_cast<size_t>(N) * K, q);
    auto* ws_dev = sycl::malloc_device<float>(N, q);
    q.memcpy(a_dev, a_int8.data(), static_cast<size_t>(M) * K * sizeof(std::int8_t)).wait();
    q.memcpy(w_dev, w_vnni.data(), static_cast<size_t>(N) * K * sizeof(std::int8_t)).wait();
    q.memcpy(ws_dev, w_scale_f.data(), N * sizeof(float)).wait();

    auto evt = quant_gemm_dpas_int8(q, a_dev, a_scale, w_dev, ws_dev,
                                      out_usm, M, N, K);
    sycl::free(a_dev, q);
    sycl::free(w_dev, q);
    sycl::free(ws_dev, q);
    return evt;
  }

  // Fast path for M=1 (single-token decode): eliminate tiling overhead.
  // The general kernel uses kBM=64 tiles and a 16x16 work-group (256 threads
  // per group, 16384 total for N=4096), but only 16 work-items produce
  // output for a single row. A flat 1D kernel with one output element per
  // work-item is far more efficient for decode.
  if (M == 1) {
    constexpr int nWG = kThr;
    const int nBlocks = (N + nWG - 1) / nWG;
    const int global = nBlocks * nWG;

    // FP16 fast path: load activations and weights as 8-wide vectors.
    // This halves the LSU pressure compared to scalar loads.
    if (bits == 16 && Gw == static_cast<int>(K)) {
      return q.submit([&](sycl::handler& h) {
        h.depends_on(copy_dep);
        h.parallel_for<class gemm_usm_decode1_f16>(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
            [=](sycl::nd_item<1> it) {
              const int ng = it.get_global_id(0);
              if (ng >= N) return;

              const int ng_g = ng * num_groups_w;
              const std::uint16_t* sptr = scale_usm + ng_g;
              const float alpha_f = alpha;
              float acc = 0.0f;

              const auto* w8 = reinterpret_cast<const std::uint16_t*>(code_u8);
              const auto* a8 = act_usm;

              for (int bk = 0; bk < nk_blocks; ++bk) {
                const int kg_base = bk * kBK;
                const std::uint16_t* aptr = a8 + kg_base;
                const std::uint16_t* wptr = w8 + (std::size_t)ng * K + kg_base;

                // Accumulate 8 elements at a time using SYCL vec loads.
                for (int i = 0; i < kBK; i += 8) {
                  sycl::vec<sycl::half, 8> av8;
                  sycl::vec<sycl::half, 8> wv8;
                  for (int j = 0; j < 8; ++j) {
                    av8[j] = sycl::bit_cast<sycl::half>(aptr[i + j]);
                    wv8[j] = sycl::bit_cast<sycl::half>(wptr[i + j]);
                  }
                  for (int j = 0; j < 8; ++j) {
                    acc += static_cast<float>(av8[j]) *
                           static_cast<float>(wv8[j]) * alpha_f *
                           from_half_bits(sptr[0]);
                  }
                }
              }
              out_usm[ng] = to_half_bits(acc);
            });
      });
    }

    // Vectorized M=1 variant: load 8 activations at a time to halve LSU
    // pressure while keeping weight dequant scalar.
    if (bits < 16) {
      return q.submit([&](sycl::handler& h) {
        h.depends_on(copy_dep);
        h.parallel_for<class gemm_usm_decode1_vec>(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
            [=](sycl::nd_item<1> it) {
              const int ng = it.get_global_id(0);
              if (ng >= N) return;

              float acc = 0.0f;
              const int ng_g = ng * num_groups_w;
              const std::uint16_t* sptr = scale_usm + ng_g;

              for (int bk = 0; bk < nk_blocks; ++bk) {
                const int kg_base = bk * kBK;
                // Load 8 activations as a vector per K-step.
                for (int kk = 0; kk < kBK; kk += 8) {
                  sycl::vec<sycl::half, 8> av8;
                  for (int j = 0; j < 8; ++j) {
                    const int kg = kg_base + kk + j;
                    if (kg >= static_cast<int>(K)) break;
                    av8[j] = sycl::bit_cast<sycl::half>(act_usm[kg]);
                  }
                  for (int j = 0; j < 8; ++j) {
                    const int kg = kg_base + kk + j;
                    if (kg >= static_cast<int>(K)) break;
                    const std::size_t idx = static_cast<std::size_t>(ng) * K + kg;
                    const int g = kg / Gw;
                    const float scale = from_half_bits(sptr[g]) * alpha;
                    int code;
                    if (bits >= 8) {
                      code = static_cast<int>(code_u8[idx]);
                    } else {
                      const int pack = 8 / bits;
                      const std::uint8_t b = code_u8[idx / pack];
                      const int within = static_cast<int>(idx % pack);
                      code = (b >> (within * bits)) & ((1 << bits) - 1);
                    }
                    const float wval = scale * canon_value(bits, code);
                    acc += static_cast<float>(av8[j]) * wval;
                  }
                }
              }
              out_usm[ng] = to_half_bits(acc);
            });
      });
    }

    // Scalar M=1 path (kept for groupwise FP16 activations).
    return q.submit([&](sycl::handler& h) {
      h.depends_on(copy_dep);
      h.parallel_for<class gemm_usm_decode1>(
          sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
          [=](sycl::nd_item<1> it) {
            const int ng = it.get_global_id(0);
            if (ng >= N) return;

            float acc = 0.0f;
            const int ng_g = ng * num_groups_w;
            for (int bk = 0; bk < nk_blocks; ++bk) {
              const int kg_base = bk * kBK;
              for (int kk = 0; kk < kBK; ++kk) {
                const int kg = kg_base + kk;
                if (kg >= K) break;

                const float av = from_half_bits(act_usm[kg]);
                const std::size_t idx = static_cast<std::size_t>(ng) * K + kg;
                const int g = kg / Gw;
                const float s = from_half_bits(scale_usm[ng_g + g]) * alpha;

                float wval;
                if (bits >= 16) {
                  const std::uint16_t hv = code_u8[idx * 2] |
                      (static_cast<std::uint16_t>(code_u8[idx * 2 + 1]) << 8);
                  wval = from_half_bits(hv);
                } else {
                  int code;
                  if (bits >= 8) {
                    code = static_cast<int>(code_u8[idx]);
                  } else {
                    const int pack = 8 / bits;
                    const std::uint8_t b = code_u8[idx / pack];
                    const int within = static_cast<int>(idx % pack);
                    code = (b >> (within * bits)) & ((1 << bits) - 1);
                  }
                  wval = s * canon_value(bits, code);
                }
                acc += av * wval;
              }
            }
            out_usm[ng] = to_half_bits(acc);
          });
    });
  }

  const int gx = (static_cast<int>(M) + kBM - 1) / kBM;
  const int gy = (static_cast<int>(N) + kBN - 1) / kBN;
  sycl::nd_range<2> nd(sycl::range<2>(gx * kThr, gy * kThr),
                       sycl::range<2>(kThr, kThr));

  return q.submit([&](sycl::handler& h) {
    h.depends_on(copy_dep);
    sycl::local_accessor<std::uint16_t, 1> sA(sycl::range<1>(kBM * kBK), h);
    sycl::local_accessor<std::uint16_t, 1> sW(sycl::range<1>(kBN * kBK), h);
    h.parallel_for(nd, [=](sycl::nd_item<2> it) {
      const int itm = it.get_global_id(0);
      const int itn = it.get_global_id(1);
      const int bm = itm / kThr, bn = itn / kThr;
      const int ty = itm % kThr, tx = itn % kThr;

      float acc[kSub * kSub] = {0};

      const int nk_blocks = (static_cast<int>(K) + kBK - 1) / kBK;
      for (int bk = 0; bk < nk_blocks; ++bk) {
        // Load activation tile (FP16, pass-through from USM).
        for (int e = ty * kThr + tx; e < kBM * kBK; e += kThr * kThr) {
          const int r = e / kBK, c = e % kBK;
          const int mg = bm * kBM + r;
          const int kg = bk * kBK + c;
          std::uint16_t val = 0;
          if (mg < static_cast<int>(M) && kg < static_cast<int>(K))
            val = act_usm[mg * K + kg];
          sA[e] = val;
        }
        // Load weight tile (dequant-on-the-fly from USM).
        for (int e = ty * kThr + tx; e < kBN * kBK; e += kThr * kThr) {
          const int r = e / kBK, c = e % kBK;
          const int ng = bn * kBN + r;
          const int kg = bk * kBK + c;
          std::uint16_t val = 0;
          if (ng < static_cast<int>(N) && kg < static_cast<int>(K)) {
            const std::size_t idx = static_cast<std::size_t>(ng) * K + kg;
            const int g = kg / Gw;
            const float s = from_half_bits(scale_usm[ng * num_groups_w + g]) * alpha;
            float wval;
            if (bits >= 16) {
              const std::uint16_t hv = code_u8[idx * 2] | ((std::uint16_t)code_u8[idx * 2 + 1] << 8);
              wval = from_half_bits(hv);
            } else {
              int code;
              if (bits >= 8) {
                code = static_cast<int>(code_u8[idx]);
              } else {
                const int pack = 8 / bits;
                const std::uint8_t b = code_u8[idx / pack];
                const int within = static_cast<int>(idx % pack);
                code = (b >> (within * bits)) & ((1 << bits) - 1);
              }
              wval = s * canon_value(bits, code);
            }
            val = to_half_bits(wval);
          }
          sW[e] = val;
        }
        it.barrier();

        for (int kk = 0; kk < kBK; ++kk) {
          for (int i = 0; i < kSub; ++i)
            for (int j = 0; j < kSub; ++j) {
              const sycl::half av = sycl::bit_cast<sycl::half>(sA[(ty * kSub + i) * kBK + kk]);
              const sycl::half wv = sycl::bit_cast<sycl::half>(sW[(tx * kSub + j) * kBK + kk]);
              acc[i * kSub + j] += static_cast<float>(av) * static_cast<float>(wv);
            }
        }
        it.barrier();
      }

      for (int i = 0; i < kSub; ++i)
        for (int j = 0; j < kSub; ++j) {
          const int mg = bm * kBM + ty * kSub + i;
          const int ng = bn * kBN + tx * kSub + j;
          if (mg < static_cast<int>(M) && ng < static_cast<int>(N))
            out_usm[mg * N + ng] = to_half_bits(acc[i * kSub + j]);
        }
    });
  });
}

// ---------------------------------------------------------------------------
// (2) ESIMD DPAS INT8 x INT8 GEMM (XMX systolic array).
// ---------------------------------------------------------------------------
void build_vnni_int8(const std::int8_t* w, std::int8_t* out,
                     uint32_t N, uint32_t K) {
  // B (K x N) operand in plain row-major: out[k*N + n] = W[n, k] (B = W^T).
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      out[static_cast<std::size_t>(k) * N + n] =
          w[static_cast<std::size_t>(n) * K + k];
}

// ---------------------------------------------------------------------------
// FP16 x FP16 GEMM (non-quantized path for lm_head etc.).
// C[M,N] = A[M,K] * W[N,K]^T. All row-major. FP16 inputs, FP32 accum,
// FP16 output. The compiler lowers this onto XMX FP16 matrix engines on Arc.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_f16(sycl::queue& q,
                    const sycl::half* a,  // M*K row-major
                    const sycl::half* w,  // N*K row-major
                    sycl::half* c,        // M*N row-major
                    uint32_t M, uint32_t N, uint32_t K) {
  const int gx = (static_cast<int>(M) + kBM - 1) / kBM;
  const int gy = (static_cast<int>(N) + kBN - 1) / kBN;
  sycl::nd_range<2> nd(sycl::range<2>(gx * kThr, gy * kThr),
                       sycl::range<2>(kThr, kThr));

  auto evt = q.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint16_t, 1> sA(sycl::range<1>(kBM * kBK), h);
    sycl::local_accessor<std::uint16_t, 1> sW(sycl::range<1>(kBN * kBK), h);
    h.parallel_for(nd, [=](sycl::nd_item<2> it) {
      const int itm = it.get_global_id(0);
      const int itn = it.get_global_id(1);
      const int bm = itm / kThr, bn = itn / kThr;
      const int ty = itm % kThr, tx = itn % kThr;

      float acc[kSub * kSub] = {0};

      const int nk_blocks = (K + kBK - 1) / kBK;
      for (int bk = 0; bk < nk_blocks; ++bk) {
        for (int e = ty * kThr + tx; e < kBM * kBK; e += kThr * kThr) {
          const int r = e / kBK, cc = e % kBK;
          const int mg = bm * kBM + r;
          const int kg = bk * kBK + cc;
          std::uint16_t val = 0;
          if (mg < static_cast<int>(M) && kg < static_cast<int>(K))
            val = sycl::bit_cast<std::uint16_t>(a[mg * K + kg]);
          sA[e] = val;
        }
        for (int e = ty * kThr + tx; e < kBN * kBK; e += kThr * kThr) {
          const int r = e / kBK, cc = e % kBK;
          const int ng = bn * kBN + r;
          const int kg = bk * kBK + cc;
          std::uint16_t val = 0;
          if (ng < static_cast<int>(N) && kg < static_cast<int>(K))
            val = sycl::bit_cast<std::uint16_t>(w[ng * K + kg]);
          sW[e] = val;
        }
        it.barrier();

        for (int kk = 0; kk < kBK; ++kk) {
          for (int i = 0; i < kSub; ++i)
            for (int j = 0; j < kSub; ++j) {
              const float av = static_cast<float>(
                  sycl::bit_cast<sycl::half>(sA[(ty * kSub + i) * kBK + kk]));
              const float wv = static_cast<float>(
                  sycl::bit_cast<sycl::half>(sW[(tx * kSub + j) * kBK + kk]));
              acc[i * kSub + j] += av * wv;
            }
        }
        it.barrier();
      }

      for (int i = 0; i < kSub; ++i)
        for (int j = 0; j < kSub; ++j) {
          const int mg = bm * kBM + ty * kSub + i;
          const int ng = bn * kBN + tx * kSub + j;
          if (mg < static_cast<int>(M) && ng < static_cast<int>(N))
            c[mg * N + ng] = to_half_bits(acc[i * kSub + j]);
        }
    });
  });
  return evt;
}

sycl::event quant_gemm_dpas_int8(sycl::queue& q,
                          const std::int8_t* a_int8,
                          float a_scale,
                          const std::int8_t* w_vnni,
                          const float* w_scale,
                          std::uint16_t* c,
                          uint32_t M, uint32_t N, uint32_t K) {
  using namespace sycl;
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::esimd::xmx;
  using namespace intfer::gemm;

  constexpr int SD = 8;  // systolic depth
  constexpr int RC = 8;  // repeat count -> _M = 8
  // _M = RC = 8, _K = SD * 4 (s8 ops/chan) = 32, _N = 8 (Xe2 exec size 8).
  constexpr int TM = 8;
  constexpr int TK = 32;
  constexpr int TN = 8;

  const uint32_t M8 = (M + TM - 1) / TM;
  const uint32_t N16 = (N + TN - 1) / TN;
  const uint32_t Ksteps = K / TK;
  const uint32_t n_blocks = M8 * N16;
  const uint32_t nWG = 16;  // work-group size == SIMD width
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;

  auto evt = q.submit([&](handler& h) {
    h.parallel_for<class gemm_dpas_int8_kernel>(
        nd_range<1>(range<1>(global), range<1>(nWG)),
        [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
          const uint32_t block = it.get_global_id(0);
          if (block >= n_blocks) return;
          const uint32_t bm = block / N16;
          const uint32_t bn = block % N16;

          simd<int, TM * TN> acc(0);
          for (uint32_t bk = 0; bk < Ksteps; ++bk) {
            const std::int8_t* a_ptr = a_int8 + (bm * TM) * K + bk * TK;
            const std::int8_t* w_ptr = w_vnni + (bk * TK) * N + bn * TN;
            simd<std::int8_t, TM * TK> va;
            va.copy_from(a_ptr);
            simd<std::int8_t, TK * TN> vb;
            vb.copy_from(w_ptr);
            acc = dpas<SD, RC, int, int, std::int8_t, std::int8_t>(acc, va, vb);
          }

          // Apply per-tensor activation scale * per-row weight scale.
          simd<float, TN> wsf;
          wsf.copy_from(w_scale + bn * TN);
          std::uint16_t out[TM * TN];
          for (int mm = 0; mm < TM; ++mm)
            for (int nn = 0; nn < TN; ++nn) {
              const float v = a_scale * wsf[nn] *
                              static_cast<float>(acc[mm * TN + nn]);
              out[mm * TN + nn] = to_half_bits(v);
            }
          simd<std::uint16_t, TM * TN> oc;
          oc.copy_from(out);
          oc.copy_to(c + (bm * TM) * N + bn * TN);
        });
  });
  q.wait();
  return evt;
}

}  // namespace intfer::gemm
