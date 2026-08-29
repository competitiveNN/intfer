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

// Sign-extend `bits`-wide value stored in the low bits of raw at byte position.
inline int unpack_nibble(std::uint8_t raw, int bits, int shift) {
  int v = (raw >> shift) & ((1u << bits) - 1u);
  int s = 32 - bits;
  return (v << s) >> s;
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
            int code = 0;
            if (bits >= 8) {
              code = static_cast<int>(wck[idx]);
            } else {
              const int pack = 8 / bits;
              const std::uint8_t b = wck[idx / pack];
              code = unpack_nibble(b, bits, static_cast<int>(idx % pack) * bits);
            }
            const int g = kg / Gw;
            const float s = from_half_bits(wsk[ng * num_groups_w + g]) / alpha;
            val = to_half_bits(s * static_cast<float>(code));
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

void quant_gemm_dpas_int8(sycl::queue& q,
                          const std::int8_t* a_int8,
                          float a_scale,
                          const std::int8_t* w_vnni,
                          const float* w_scale,
                          std::uint16_t* c,
                          uint32_t M, uint32_t N, uint32_t K) {
  using namespace sycl;
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::esimd::xmx;

  constexpr int SD = 8;  // systolic depth
  constexpr int RC = 8;  // repeat count -> _M = 8
  // _M = RC = 8, _K = SD * 4 (s8 ops/chan) = 32, _N = 8 (Xe2 exec size 8).
  constexpr int TM = 8;
  constexpr int TK = 32;
  constexpr int TN = 8;

  const uint32_t M8 = M / TM;
  const uint32_t N16 = N / TN;
  const uint32_t Ksteps = K / TK;
  const uint32_t n_blocks = M8 * N16;
  const uint32_t nWG = 16;  // work-group size == SIMD width
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;

  q.submit([&](handler& h) {
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
}

}  // namespace intfer::gemm
