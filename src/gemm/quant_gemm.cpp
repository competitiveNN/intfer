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
#include <cstdlib>
#include <sycl/sycl.hpp>

#include <vector>

// ESIMD / XMX DPAS intrinsics (header-only; only used by the DPAS path below).
#include <sycl/ext/intel/esimd.hpp>

namespace intfer::gemm {

#ifdef INTFER_PROFILE_GPU
// Weak default: targets linking intfer_gemm without intfer_engine (e.g.
// gemm_demo) still link. intfer_engine provides the strong definition that
// counts into g_gpu_profile; the linker prefers it whenever both are linked.
__attribute__((weak)) void profile_note_launch() {}
#endif

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

    // Vectorized M=1 variant: group-outer loop hoists the per-group scale
    // (was recomputed per K element), vector LSU loads for activations and
    // INT8 codes, and constant-trip-count inner loops so the compiler can
    // unroll/interleave the dequant+MAC chains. Same numerics as scalar path.
    if (bits < 16) {
      const int num_groups = (static_cast<int>(K) + Gw - 1) / Gw;
      const bool is_int8 = (bits >= 8);
      const int pack = is_int8 ? 1 : 8 / bits;
      // Kill switch INTFER_NO_INT2_TAB=1 restores the generic path
      // (bisection). Host-side read; getenv/static are illegal in device.
      const bool use_int2_tab =
          (std::getenv("INTFER_NO_INT2_TAB") == nullptr);
      return q.submit([&](sycl::handler& h) {
        h.depends_on(copy_dep);
        h.parallel_for<class gemm_usm_decode1_vec>(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
            [=](sycl::nd_item<1> it) {
              const int ng = it.get_global_id(0);
              if (ng >= N) return;

              float acc = 0.0f;
              const std::uint16_t* sptr = scale_usm + ng * num_groups_w;
              const std::uint8_t* wrow =
                  code_u8 + static_cast<std::size_t>(ng) * K;
              sycl::multi_ptr<const std::uint16_t,
                  sycl::access::address_space::global_space> act_mp(act_usm);
              sycl::multi_ptr<const std::uint8_t,
                  sycl::access::address_space::global_space> w_mp(wrow);

              for (int g = 0; g < num_groups; ++g) {
                const float scale = from_half_bits(sptr[g]) * alpha;
                const int kg0 = g * Gw;
                int kg1 = kg0 + Gw;
                if (kg1 > static_cast<int>(K)) kg1 = (int)K;
                if (is_int8) {
                  int kg = kg0;
                  for (; kg + 8 <= kg1; kg += 8) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    sycl::vec<std::uint8_t, 8> cv;
                    cv.load(0, w_mp + kg);
                    for (int j = 0; j < 8; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const float wval =
                          scale * ((2.0f * (float)cv[j] - 255.0f) / 255.0f);
                      acc += av * wval;
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    const float wval =
                        scale * ((2.0f * (float)wrow[kg] - 255.0f) / 255.0f);
                    acc += av * wval;
                  }
                } else {
                  // Packed codes (INT4/INT2): 8-code chunks; byte index and
                  // bit position derive from j with shifts (no div/mod).
                  const int codes_per_chunk = 8;
                  const int bytes_per_chunk = codes_per_chunk / pack;
                  const int code_mask = (1 << bits) - 1;
                  const std::size_t row_base =
                      static_cast<std::size_t>(ng) * K;
                  const int shr = (bits == 4) ? 1 : 2;  // j -> byte index
                  // INT2 table-hoist path (default ON; kill switch
                  // INTFER_NO_INT2_TAB=1): fold the
                  // per-group scale into a 4-entry register weight table so
                  // the inner element costs one table lookup + one FMA
                  // instead of canon_value (branch + static-table load) plus
                  // two scale multiplies. Op order changes from
                  // (av*scale)*c2 to av*(scale*c2): ~1ulp on rare elements,
                  // oracle-gated (same standing as the silu-exp precedent).
                  // INT4 keeps the generic path (linear formula vectorizes).
                  // (use_int2_tab is a host-read flag captured by value.)
                  // PROMOTED 2026-09-28 (was INTFER_INT2_TAB opt-in, 4917d8f):
                  // quiet-box 27B TP=2 decode 3 rounds +5.7/+6.0/+5.2%
                  // (2.63->2.78, 2.67->2.83, 2.68->2.82 tok/s), kernel bucket
                  // (load-immune) -12.8%, streams identical TP1/TP2,
                  // cpu parity + http batch + gpu-optin 10/10 PASS.
                  if (use_int2_tab && bits == 2) {
                    const float wtab[4] = {
                        scale * -1.0f, scale * (-1.0f / 3.0f),
                        scale * (1.0f / 3.0f), scale * 1.0f};
                    int kg = kg0;
                    std::size_t b0 =
                        (row_base + (std::size_t)kg) / (std::size_t)pack;
                    for (; kg + codes_per_chunk <= kg1;
                         kg += codes_per_chunk) {
                      sycl::vec<std::uint16_t, 8> a16;
                      a16.load(0, act_mp + kg);
                      const std::uint8_t b0v = code_u8[b0];
                      const std::uint8_t b1v = code_u8[b0 + 1];
                      b0 += 2;
                      for (int j = 0; j < codes_per_chunk; ++j) {
                        const float av = static_cast<float>(
                            sycl::bit_cast<sycl::half>(a16[j]));
                        const std::uint8_t bv = (j < 4) ? b0v : b1v;
                        const int code =
                            (bv >> ((j & 3) * 2)) & 3;
                        acc += av * wtab[code];
                      }
                    }
                    for (; kg < kg1; ++kg) {
                      const float av = from_half_bits(act_usm[kg]);
                      const std::size_t idx = row_base + kg;
                      const std::uint8_t byte = code_u8[idx / pack];
                      const int within = (int)(idx % pack);
                      const int code =
                          (byte >> (within * bits)) & ((1 << bits) - 1);
                      acc += av * scale * canon_value(bits, code);
                    }
                  } else {
                  int kg = kg0;
                  std::size_t b0 =
                      (row_base + (std::size_t)kg) / (std::size_t)pack;
                  for (; kg + codes_per_chunk <= kg1;
                       kg += codes_per_chunk) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    std::uint8_t b[4];
                    for (int t = 0; t < bytes_per_chunk; ++t)
                      b[t] = code_u8[b0 + t];
                    b0 += bytes_per_chunk;
                    for (int j = 0; j < codes_per_chunk; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const int code =
                          (b[j >> shr] >> ((j & (pack - 1)) * bits)) &
                          code_mask;
                      acc += av * scale * canon_value(bits, code);
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    const std::size_t idx = row_base + kg;
                    const std::uint8_t byte = code_u8[idx / pack];
                    const int within = (int)(idx % pack);
                    const int code =
                        (byte >> (within * bits)) & ((1 << bits) - 1);
                    acc += av * scale * canon_value(bits, code);
                  }
                  }  // end generic packed path (non-INT2-tab)
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
            c[mg * N + ng] = sycl::bit_cast<sycl::half>(to_half_bits(acc[i * kSub + j]));
        }
    });
  });
  return evt;
}

sycl::event quant_gemm_f16_int8w(sycl::queue& q,
                    const sycl::half* a,    // M*K row-major FP16
                    const std::int8_t* w,   // N*K row-major INT8 codes
                    const sycl::half* ws,   // N per-row FP16 scales
                    sycl::half* c,          // M*N row-major FP16 out
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
          if (ng < static_cast<int>(N) && kg < static_cast<int>(K)) {
            // Per-row symmetric dequant: w = scale[ng] * int8.
            const float s = static_cast<float>(ws[ng]);
            const float wv = s * static_cast<float>(w[(std::size_t)ng * K + kg]);
            val = to_half_bits(wv);
          }
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
            c[mg * N + ng] = sycl::bit_cast<sycl::half>(to_half_bits(acc[i * kSub + j]));
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
  using namespace gemm;

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

// ---------------------------------------------------------------------------
// W4A8 / W2A8 native sub-byte DPAS (M=1 decode). See header for math/layout.
// ---------------------------------------------------------------------------
bool repack_dpas412(const std::uint8_t* row_codes, uint32_t N, uint32_t K,
                    int bits, std::vector<std::uint8_t>& out) {
  constexpr uint32_t TN = 16, OP = 32;
  if (N % TN != 0 || K % OP != 0) return false;
  if (bits != 8 && bits != 4 && bits != 2) return false;
  const uint32_t nT = N / TN, nOp = K / OP;
  const uint32_t op_bytes = OP * TN * (uint32_t)bits / 8;
  out.assign((size_t)nT * nOp * op_bytes, 0);
  const int pack = (bits >= 8) ? 1 : 8 / bits;  // codes per row-major byte
  const int E = 32 / bits;                       // rows per dword
  const uint32_t mask = (bits >= 8) ? 0xFFu : ((1u << bits) - 1u);
  auto get_idx = [&](uint32_t n, uint32_t k) -> uint32_t {
    const size_t flat = (size_t)n * K + k;
    if (bits >= 8) return row_codes[flat];
    const uint8_t b = row_codes[flat / (size_t)pack];
    return (b >> ((flat % (size_t)pack) * (uint32_t)bits)) & mask;
  };
  for (uint32_t t = 0; t < nT; ++t)
    for (uint32_t o = 0; o < nOp; ++o) {
      uint32_t* d = reinterpret_cast<uint32_t*>(
          out.data() + ((size_t)t * nOp + o) * op_bytes);
      for (uint32_t kl = 0; kl < OP; ++kl)
        for (uint32_t nl = 0; nl < TN; ++nl) {
          const uint32_t v = get_idx(t * TN + nl, o * OP + kl);
          d[(size_t)(kl / E) * TN + nl] |= v << ((kl % E) * bits);
        }
    }
  return true;
}

sycl::event quant_gemm_dpas48(sycl::queue& q, const Dpas48Params& p) {
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::esimd::xmx;
  constexpr uint32_t TN = 16, OP = 32;
  const uint32_t M = p.M, N = p.N, K = p.K;
  const uint32_t G = (p.group_size == 0) ? K : p.group_size;
  const uint32_t GA = (p.act_group == 0) ? K : p.act_group;
  const uint32_t nT = N / TN, nOp = K / OP;
  const uint32_t nGA = (K + GA - 1) / GA;
  const uint32_t nG = (K + G - 1) / G;
  const uint32_t act_stride = (p.act_stride == 0) ? K : p.act_stride;
  const uint32_t as_stride = (p.as_stride == 0) ? nGA : p.as_stride;
  const uint32_t op_bytes = OP * TN * (uint32_t)p.bits / 8;
  const uint8_t* B = static_cast<const uint8_t*>(p.code_dpas);
  const auto BPrec = (p.bits == 4) ? dpas_argument_type::u4
                     : (p.bits == 2) ? dpas_argument_type::u2
                                     : dpas_argument_type::u8;
  // INT2/G16 needs a masked-A second DPAS per op (linearity split);
  // every other (bits, G) combo is 1 DPAS/op.
  const bool need_split = (p.bits == 2 && G == 16);
  const float C = (p.bits == 4) ? 15.0f : (p.bits == 2) ? 3.0f : 255.0f;
  const float Cinv = 1.0f / C;
  const float alpha = p.alpha;

  // One WI per (row m, 16-N tile t). B op-tiles reload per WI (L1-resident
  // across the M rows of a tile); A is per-row. Same fold math as the M=1
  // probe (probe_dpas_fold.cpp T5/T6, 1e-16 exact).
  const uint32_t nWG = 16;
  const uint32_t n_blocks = M * nT;
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;

  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class gemm_dpas48_m1_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const uint32_t block = it.get_global_id(0);
          if (block >= n_blocks) return;
          const uint32_t m = block / nT;
          const uint32_t t = block % nT;
          const std::int8_t* Aq = p.act_int8 + (size_t)m * act_stride;
          const float* As = p.act_scale + (size_t)m * as_stride;
          float acc_tile[TN];
          for (uint32_t nl = 0; nl < TN; ++nl) acc_tile[nl] = 0.0f;
          for (uint32_t o = 0; o < nOp; ++o) {
            const uint32_t k0 = o * OP;
            simd<std::int8_t, OP> va;
            va.copy_from(Aq + k0);
            const float as = As[k0 / GA];
            if (BPrec == dpas_argument_type::u4) {
              simd<uint8_t, 256> vb;  // op_bytes(u4) = 32*16*4/8
              vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes);
              simd<int, TN> c;
              c = dpas<8, 1, int, uint8_t, std::int8_t,
                       dpas_argument_type::u4, dpas_argument_type::s8>(vb, va);
              int S = 0;
              for (uint32_t k = 0; k < OP; ++k) S += va[k];
              const uint32_t g = k0 / G;
              for (uint32_t nl = 0; nl < TN; ++nl) {
                const uint32_t n = t * TN + nl;
                const float ws = static_cast<float>(
                    sycl::bit_cast<sycl::half>(
                        p.scale_usm[(size_t)n * nG + g]));
                acc_tile[nl] += as * ws * alpha *
                                (2.0f * (float)c[nl] - 15.0f * (float)S) *
                                Cinv;
              }
            } else if (BPrec == dpas_argument_type::u8) {
              simd<uint8_t, 512> vb;  // op_bytes(u8) = 32*16*8/8
              vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes);
              simd<int, TN> c;
              c = dpas<8, 1, int, uint8_t, std::int8_t,
                       dpas_argument_type::u8, dpas_argument_type::s8>(vb, va);
              int S = 0;
              for (uint32_t k = 0; k < OP; ++k) S += va[k];
              const uint32_t g = k0 / G;
              for (uint32_t nl = 0; nl < TN; ++nl) {
                const uint32_t n = t * TN + nl;
                const float ws = static_cast<float>(
                    sycl::bit_cast<sycl::half>(
                        p.scale_usm[(size_t)n * nG + g]));
                acc_tile[nl] += as * ws * alpha *
                                (2.0f * (float)c[nl] - 255.0f * (float)S) *
                                Cinv;
              }
            } else {
              // u2: K=32 op, E=16 rows/dword, 32*16*2/8 = 128B.
              simd<uint8_t, 128> vb;
              vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes);
              if (!need_split) {
                simd<int, TN> c;
                c = dpas<8, 1, int, uint8_t, std::int8_t,
                         dpas_argument_type::u2, dpas_argument_type::s8>(vb, va);
                int S = 0;
                for (uint32_t k = 0; k < OP; ++k) S += va[k];
                const uint32_t g = k0 / G;
                for (uint32_t nl = 0; nl < TN; ++nl) {
                  const uint32_t n = t * TN + nl;
                  const float ws = static_cast<float>(
                      sycl::bit_cast<sycl::half>(
                          p.scale_usm[(size_t)n * nG + g]));
                  acc_tile[nl] += as * ws * alpha *
                                  (2.0f * (float)c[nl] - 3.0f * (float)S) *
                                  Cinv;
                }
              } else {
                // G16 split: T_full on va, T_0 on va masked to first 16.
                simd<int, TN> cf, c0;
                cf = dpas<8, 1, int, uint8_t, std::int8_t,
                          dpas_argument_type::u2, dpas_argument_type::s8>(vb, va);
                simd<std::int8_t, OP> va0(0);
                int S0 = 0;
                for (uint32_t k = 0; k < 16; ++k) {
                  va0[k] = va[k];
                  S0 += va[k];
                }
                c0 = dpas<8, 1, int, uint8_t, std::int8_t,
                          dpas_argument_type::u2, dpas_argument_type::s8>(vb, va0);
                int Sf = S0;
                for (uint32_t k = 16; k < OP; ++k) Sf += va[k];
                const uint32_t g0 = k0 / G, g1 = g0 + 1;
                for (uint32_t nl = 0; nl < TN; ++nl) {
                  const uint32_t n = t * TN + nl;
                  const float ws0 = static_cast<float>(
                      sycl::bit_cast<sycl::half>(
                          p.scale_usm[(size_t)n * nG + g0]));
                  const float ws1 = static_cast<float>(
                      sycl::bit_cast<sycl::half>(
                          p.scale_usm[(size_t)n * nG + g1]));
                  const float T0 = (float)c0[nl], Tf = (float)cf[nl];
                  const float T1 = Tf - T0;
                  const float S1 = (float)(Sf - S0);
                  acc_tile[nl] += as * alpha *
                                  ((2.0f * T0 - 3.0f * (float)S0) * ws0 +
                                   (2.0f * T1 - 3.0f * S1) * ws1) *
                                  Cinv;
                }
              }
            }
          }
          std::uint16_t oh[TN];
          for (uint32_t nl = 0; nl < TN; ++nl)
            oh[nl] = to_half_bits(acc_tile[nl]);
          simd<std::uint16_t, TN> oc;
          oc.copy_from(oh);
          oc.copy_to(p.out_usm + (size_t)m * N + (size_t)t * TN);
        });
  });
}

// ---------------------------------------------------------------------------
// Row-cooperative INT4/G64 DPAS (EXPERIMENTAL, default-off INTFER_DPAS_ROW).
// Same Dpas48Params contract + same u4 fold math as quant_gemm_dpas48, but
// one WI per ROW loops over the 16-N tiles (strips of RSTRIP resident)
// instead of one WI per (row, tile). Per-(row,op) A work (va load, S sum,
// as load) runs once instead of nT x.
//
// MEASURED 2026-09-29 (build_tmp probe, since removed): BIT-EXACT vs
// legacy (0 diffs) but ~460x SLOWER device-side (260us -> 119ms on the
// 27B up shape N=8704/K=5120 M=1). Cause: serializing all tiles x ops
// through one WI collapses occupancy AND the acc[strip][16] private
// array + runtime-bound inner tile loop defeats ESIMD DPAS pipelining.
// The legacy WI-per-(m,t) geometry is already right for DPAS issue.
// Kept ONLY as the experiment base for subgroup-cooperative sharing
// (shuffle/SLM broadcast of va/S/as, WI-per-tile parallelism kept).
// Do NOT promote without a quiet-box A/B win. Declines (null event)
// unless bits==4 and weight group==64.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_dpas4_row(sycl::queue& q, const Dpas48Params& p) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::esimd::xmx;
  constexpr uint32_t TN = 16, OP = 32;
  constexpr uint32_t RSTRIP = 4;
  const uint32_t M = p.M, N = p.N, K = p.K;
  const uint32_t G = (p.group_size == 0) ? K : p.group_size;
  if (p.bits != 4 || G != 64) return sycl::event{};
  const uint32_t GA = (p.act_group == 0) ? K : p.act_group;
  const uint32_t nT = N / TN, nOp = K / OP;
  const uint32_t nGA = (K + GA - 1) / GA;
  const uint32_t nG = (K + G - 1) / G;
  const uint32_t act_stride = (p.act_stride == 0) ? K : p.act_stride;
  const uint32_t as_stride = (p.as_stride == 0) ? nGA : p.as_stride;
  const uint32_t op_bytes = OP * TN * 4 / 8;  // u4
  const uint8_t* B = static_cast<const uint8_t*>(p.code_dpas);
  const float alpha = p.alpha;

  const uint32_t nWG = 16;
  const uint32_t global = ((M + nWG - 1) / nWG) * nWG;

  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class gemm_dpas4_row_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const uint32_t m = it.get_global_id(0);
          if (m >= M) return;
          const std::int8_t* Aqr = p.act_int8 + (size_t)m * act_stride;
          const float* Asr = p.act_scale + (size_t)m * as_stride;
          for (uint32_t ts = 0; ts < nT; ts += RSTRIP) {
            const uint32_t tlim = (ts + RSTRIP < nT) ? ts + RSTRIP : nT;
            float acc[RSTRIP][TN];
            for (uint32_t tt = ts; tt < tlim; ++tt)
              for (uint32_t nl = 0; nl < TN; ++nl) acc[tt - ts][nl] = 0.0f;
            for (uint32_t o = 0; o < nOp; ++o) {
              const uint32_t k0 = o * OP;
              simd<std::int8_t, OP> va;
              va.copy_from(Aqr + k0);
              const float as = Asr[k0 / GA];
              int S = 0;
              for (uint32_t k = 0; k < OP; ++k) S += va[k];
              const uint32_t g = k0 / G;
              for (uint32_t tt = ts; tt < tlim; ++tt) {
                simd<uint8_t, 256> vb;  // op_bytes(u4) = 32*16*4/8
                vb.copy_from(B + ((size_t)tt * nOp + o) * op_bytes);
                simd<int, TN> c;
                c = dpas<8, 1, int, uint8_t, std::int8_t,
                         dpas_argument_type::u4, dpas_argument_type::s8>(vb, va);
                for (uint32_t nl = 0; nl < TN; ++nl) {
                  const uint32_t n = tt * TN + nl;
                  const float ws = static_cast<float>(
                      sycl::bit_cast<sycl::half>(
                          p.scale_usm[(size_t)n * nG + g]));
                  acc[tt - ts][nl] += as * ws * alpha *
                                      (2.0f * (float)c[nl] - 15.0f * (float)S) /
                                      15.0f;
                }
              }
            }
            for (uint32_t tt = ts; tt < tlim; ++tt) {
              std::uint16_t oh[TN];
              for (uint32_t nl = 0; nl < TN; ++nl)
                oh[nl] = to_half_bits(acc[tt - ts][nl]);
              simd<std::uint16_t, TN> oc;
              oc.copy_from(oh);
              oc.copy_to(p.out_usm + (size_t)m * N + (size_t)tt * TN);
            }
          }
        });
  });
}

// ---------------------------------------------------------------------------
// FP16-act-equivalent s8-weight DPAS GEMM (lm_head decode).
// ---------------------------------------------------------------------------
sycl::event quant_gemm_dpas_s8w(sycl::queue& q, const Dpas48Params& p) {
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::esimd::xmx;
  constexpr uint32_t TN = 16, OP = 32;
  const uint32_t M = p.M, N = p.N, K = p.K;
  const uint32_t GA = (p.act_group == 0) ? K : p.act_group;
  const uint32_t nT = N / TN, nOp = K / OP;
  const uint32_t nGA = (K + GA - 1) / GA;
  const uint32_t act_stride = (p.act_stride == 0) ? K : p.act_stride;
  const uint32_t as_stride = (p.as_stride == 0) ? nGA : p.as_stride;
  const uint32_t op_bytes = OP * TN;  // s8: 1 code/byte
  const int8_t* B = static_cast<const int8_t*>(p.code_dpas);
  const float alpha = p.alpha;

  // One WI per (row m, 16-N tile t). s8 x s8 DPAS: exact integer dot of the
  // raw symmetric codes; fold is a single scale multiply. Same launch
  // shape as quant_gemm_dpas48.
  const uint32_t nWG = 16;
  const uint32_t n_blocks = M * nT;
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;

  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class gemm_dpas_s8w_lmhead_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const uint32_t block = it.get_global_id(0);
          if (block >= n_blocks) return;
          const uint32_t m = block / nT;
          const uint32_t t = block % nT;
          const std::int8_t* Aq = p.act_int8 + (size_t)m * act_stride;
          const float* As = p.act_scale + (size_t)m * as_stride;
          float acc_tile[TN];
          for (uint32_t nl = 0; nl < TN; ++nl) acc_tile[nl] = 0.0f;
          for (uint32_t o = 0; o < nOp; ++o) {
            const uint32_t k0 = o * OP;
            simd<std::int8_t, OP> va;
            va.copy_from(Aq + k0);
            const float as = As[k0 / GA];
            // Vertical-B layout: op-tile (t, o) holds 32 K of 16-N rows;
            // raw s8 codes are VNNI-compatible for the s8 DPAS B operand.
            const int8_t* bop = B + ((size_t)t * nOp + o) * op_bytes;
            simd<int8_t, 512> vb;
            vb.copy_from(bop);
            simd<int, TN> c;
            c = dpas<8, 1, int, int8_t, std::int8_t,
                     dpas_argument_type::s8, dpas_argument_type::s8>(vb, va);
            for (uint32_t nl = 0; nl < TN; ++nl) {
              const uint32_t n = t * TN + nl;
              const float ws = static_cast<float>(
                  sycl::bit_cast<sycl::half>(p.scale_usm[n]));
              acc_tile[nl] += as * ws * alpha * (float)c[nl];
            }
          }
          std::uint16_t oh[TN];
          for (uint32_t nl = 0; nl < TN; ++nl)
            oh[nl] = to_half_bits(acc_tile[nl]);
          simd<std::uint16_t, TN> oc;
          oc.copy_from(oh);
          oc.copy_to(p.out_usm + (size_t)m * N + (size_t)t * TN);
        });
  });
}

// ---------------------------------------------------------------------------
// Device-side Gated-DeltaNet SSM recurrent update (Stage 3 GPU offload).
// ---------------------------------------------------------------------------
sycl::event ssm_forward(sycl::queue& q, const SsmParams& p) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const uint32_t M = p.M, n_v = p.n_v, n_k = p.n_k;  const uint32_t kd = p.kd, vd = p.vd;
  const uint32_t nWG = 32;
  const uint32_t n_blocks = M * n_v * vd;
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;
  // Capture row-pointer arrays by value (device USM pointers; the arrays
  // themselves live in device USM and are indexed on-device).
  const float* const* qn_rows = p.qn_rows;
  const float* const* kn_rows = p.kn_rows;
  const float* const* v_rows = p.v_rows;
  const float* const* g_rows = p.g_rows;
  const float* const* b_rows = p.b_rows;
  float* const* s_rows = p.s_rows;
  float* const* core_rows = p.core_rows;

  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class ssm_forward_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
          const uint32_t block = it.get_global_id(0);
          if (block >= n_blocks) return;
          const uint32_t c = block % vd;
          const uint32_t hh = (block / vd) % n_v;
          const uint32_t m = block / (vd * n_v);
          // Key-head grouping (matches host): value head hh shares key
          // head kh when n_v > n_k. kh selects the qn/kn row section;
          // value/g/b/S/core rows are indexed by hh directly.
          const uint32_t kh = (n_v > n_k)
              ? (uint32_t)(((uint64_t)hh * n_k) / n_v) : hh;
          const float* qn = qn_rows[m] + (size_t)kh * kd;
          const float* kn = kn_rows[m] + (size_t)kh * kd;
          const float* vv = v_rows[m] + (size_t)hh * vd;
          const float g_t = sycl::exp(g_rows[m][hh]);
          // sigmoid(b) = 1/(1+exp(-b)), same as host sigmoidf.
          const float bb = b_rows[m][hh];
          const float beta = 1.0f / (1.0f + sycl::exp(-bb));
          float* S = s_rows[m] + ((size_t)hh * kd) * vd;
          // 1) decay this head's state column block: S[d,c] *= g_t.
          for (uint32_t d = 0; d < kd; ++d) S[(size_t)d * vd + c] *= g_t;
          // 2) kv[c] = sum_d S[d,c]*kn[d]; delta = (v-kv)*beta.
          float kv = 0.0f;
          for (uint32_t d = 0; d < kd; ++d) kv += S[(size_t)d * vd + c] * kn[d];
          const float delta = (vv[c] - kv) * beta;
          // 3) rank-1 update: S[d,c] += kn[d]*delta.
          for (uint32_t d = 0; d < kd; ++d)
            S[(size_t)d * vd + c] += kn[d] * delta;
          // 4) readout: core = sum_d S[d,c]*qn[d].
          float s = 0.0f;
          for (uint32_t d = 0; d < kd; ++d) s += S[(size_t)d * vd + c] * qn[d];
          core_rows[m][(size_t)hh * vd + c] = s;
        });
  });
}

// Chained variant: one WI per (head hh, v-col c); the M rows share one
// state and run sequentially in row order (order-dependent recurrence).
sycl::event ssm_forward_chain(sycl::queue& q, const SsmChainParams& p) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const uint32_t M = p.M, n_v = p.n_v, n_k = p.n_k;
  const uint32_t kd = p.kd, vd = p.vd, rs = p.row_stride;
  const uint32_t nWG = 32;
  const uint32_t n_blocks = n_v * vd;
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;
  // Packed row-major inputs (single H2D upload); captured by value.
  const float* qn_all = p.qn_rows;
  const float* kn_all = p.kn_rows;
  const float* v_all = p.v_rows;
  const float* g_all = p.g_rows;
  const float* b_all = p.b_rows;
  float* S0 = p.state;
  float* core_all = p.core_rows;

  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class ssm_forward_chain_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
          const uint32_t block = it.get_global_id(0);
          if (block >= n_blocks) return;
          const uint32_t c = block % vd;
          const uint32_t hh = block / vd;
          const uint32_t kh = (n_v > n_k)
              ? (uint32_t)(((uint64_t)hh * n_k) / n_v) : hh;
          float* S = S0 + ((size_t)hh * kd) * vd;
          for (uint32_t m = 0; m < M; ++m) {
            const float* qn = qn_all + (size_t)m * rs + (size_t)kh * kd;
            const float* kn = kn_all + (size_t)m * rs + (size_t)kh * kd;
            const float vv = v_all[(size_t)m * rs + (size_t)hh * vd + c];
            const float g_t = sycl::exp(g_all[(size_t)m * rs + hh]);
            const float bb = b_all[(size_t)m * rs + hh];
            const float beta = 1.0f / (1.0f + sycl::exp(-bb));
            for (uint32_t d = 0; d < kd; ++d) S[(size_t)d * vd + c] *= g_t;
            float kv = 0.0f;
            for (uint32_t d = 0; d < kd; ++d)
              kv += S[(size_t)d * vd + c] * kn[d];
            const float delta = (vv - kv) * beta;
            for (uint32_t d = 0; d < kd; ++d)
              S[(size_t)d * vd + c] += kn[d] * delta;
            float s = 0.0f;
            for (uint32_t d = 0; d < kd; ++d)
              s += S[(size_t)d * vd + c] * qn[d];
            core_all[((size_t)m * n_v + hh) * vd + c] = s;
          }
        });
  });
}

// Bit-exact FP32->FP16 (round-to-nearest-even) matching
// dashq::float_to_half and the F16C hardware path, so device-fused down
// activations carry the same bits as the host convert path (parity-gated,
// not assumed).
static inline std::uint16_t dev_float_to_half(float v) {
  std::uint32_t x;
  static_assert(sizeof(x) == sizeof(v), "u32/f32 size");
  __builtin_memcpy(&x, &v, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::uint32_t exp = (x >> 23) & 0xFFu;
  const std::uint32_t mant = x & 0x7FFFFFu;
  if (exp == 0xFFu)
    return (std::uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
  const int e = (int)exp - 127 + 15;
  if (e >= 0x1F) return (std::uint16_t)(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return (std::uint16_t)sign;
    std::uint32_t m2 = mant | 0x800000u;
    const int shift = 14 - e;
    std::uint32_t half = m2 >> shift;
    const std::uint32_t rem = m2 & ((1u << shift) - 1u);
    const std::uint32_t hbit = 1u << (shift - 1);
    if (rem > hbit || (rem == hbit && (half & 1u))) half += 1u;
    if (half & 0x400u) return (std::uint16_t)(sign | 0x0400u);  // to min-normal
    return (std::uint16_t)(sign | (half & 0x3FFu));
  }
  std::uint32_t half = ((std::uint32_t)e << 10) | (mant >> 13);
  const std::uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half += 1u;
  return (std::uint16_t)(sign | (half & 0xFFFFu));
}

// Bit-exact FP16->FP32 matching dashq::half_to_float.
static inline float dev_half_to_float(std::uint16_t h) {
  const std::uint32_t sign = ((std::uint32_t)h >> 15) & 1u;
  const std::uint32_t exp = ((std::uint32_t)h >> 10) & 0x1Fu;
  const std::uint32_t mant = (std::uint32_t)h & 0x3FFu;
  std::uint32_t x;
  if (exp == 0) {
    if (mant == 0) { x = sign << 31; }
    else {
      int e = 0;
      std::uint32_t m2 = mant;
      while (!(m2 & 0x400u)) { m2 <<= 1; ++e; }
      m2 &= 0x3FFu;
      x = (sign << 31) | ((127 - 14 - e) << 23) | (m2 << 13);
    }
  } else if (exp == 0x1Fu) {
    x = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    x = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  __builtin_memcpy(&f, &x, 4);
  return f;
}

sycl::event silumul_to_half(sycl::queue& q, const SiluMulParams& p,
                            const sycl::event& dep) {
  const std::uint16_t* mg = p.mg_fp16;
  const std::uint16_t* mu = p.mu_fp16;
  std::uint16_t* act = p.act_fp16;
  const std::uint32_t n = p.n;
  if (!mg || !mu || !act || n == 0) return sycl::event{};
  const std::uint32_t nWG = 32;
  const std::uint32_t global = ((n + nWG - 1) / nWG) * nWG;
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for<class silumul_to_half_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
          const std::uint32_t i = it.get_global_id(0);
          if (i >= n) return;
          const float g = dev_half_to_float(mg[i]);
          const float u = dev_half_to_float(mu[i]);
          // silu(g) = g / (1 + exp(-g)), same order as host sigmoidf path.
          const float s = g / (1.0f + sycl::exp(-g));
          act[i] = dev_float_to_half(s * u);
        });
  });
}

sycl::event ssm_norm_silu_to_half(sycl::queue& q, const SsmNormParams& p,
                                   const sycl::event& dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const float* core = p.core_fp32;
  const float* z = p.z_fp32;
  const float* w = p.norm_w_fp32;
  std::uint16_t* act = p.act_fp16;
  const std::uint32_t n_v = p.n_v, vd = p.vd;
  const float eps = p.rms_eps;
  if (!core || !z || !w || !act || n_v == 0 || vd == 0)
    return sycl::event{};
  const std::uint32_t nWG = 32;
  const std::uint32_t global = ((n_v + nWG - 1) / nWG) * nWG;
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for<class ssm_norm_silu_to_half_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
          const std::uint32_t hh = it.get_global_id(0);
          if (hh >= n_v) return;
          const float* xx = core + (size_t)hh * vd;
          const float* zg = z + (size_t)hh * vd;
          std::uint16_t* oo = act + (size_t)hh * vd;
          // Same order as the host LA norm loop: sum squares, then
          // elementwise normalize * norm_w * silu(z).
          float v = 0.0f;
          for (std::uint32_t i = 0; i < vd; ++i) v += xx[i] * xx[i];
          const float inv = 1.0f / sycl::sqrt(v / (float)vd + eps);
          for (std::uint32_t i = 0; i < vd; ++i) {
            const float s = zg[i] / (1.0f + sycl::exp(-zg[i]));
            oo[i] = dev_float_to_half((xx[i] * inv) * w[i] * s);
          }
        });
  });
}

sycl::event ssm_norm_silu_to_half_m(sycl::queue& q, const SsmNormMParams& p,
                                    const sycl::event& dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const float* core = p.core_fp32;
  const float* z = p.z_fp32;
  const float* w = p.norm_w_fp32;
  std::uint16_t* act = p.act_fp16;
  const std::uint32_t M = p.M, n_v = p.n_v, vd = p.vd;
  const float eps = p.rms_eps;
  if (!core || !z || !w || !act || M == 0 || n_v == 0 || vd == 0)
    return sycl::event{};
  const std::uint32_t nWG = 32;
  const std::uint32_t n_blocks = M * n_v;
  const std::uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;
  const size_t zvw = (size_t)n_v * vd;
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for<class ssm_norm_silu_to_half_m_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
          const std::uint32_t b = it.get_global_id(0);
          if (b >= n_blocks) return;
          const std::uint32_t hh = b % n_v;
          const std::uint32_t m = b / n_v;
          // Same order as the host LA norm loop: sum squares, then
          // elementwise normalize * norm_w * silu(z). Row m's slice of
          // the contiguous [M*zvw] core/z/act buffers.
          const float* xx = core + (size_t)m * zvw + (size_t)hh * vd;
          const float* zg = z + (size_t)m * zvw + (size_t)hh * vd;
          std::uint16_t* oo = act + (size_t)m * zvw + (size_t)hh * vd;
          float v = 0.0f;
          for (std::uint32_t i = 0; i < vd; ++i) v += xx[i] * xx[i];
          const float inv = 1.0f / sycl::sqrt(v / (float)vd + eps);
          for (std::uint32_t i = 0; i < vd; ++i) {
            const float s = zg[i] / (1.0f + sycl::exp(-zg[i]));
            oo[i] = dev_float_to_half((xx[i] * inv) * w[i] * s);
          }
        });
  });
}

// L5 kernel 1: FWHT online rotations + RNE FP16 convert. One WI per tensor
// (sequential FWHT = exact host loop order in fwht(), bit-identical, no
// SLM/ordering risk). kind 0 = null rotation (copy); kind 1 = hadamard
// (FWHT over prefix p + /sqrt scale, tail passthrough — matches
// apply_rotation_fwht incl. the blockdiag path); kind 2 = dense (WI writes
// nothing; the caller must decline to the host path). N rows (qkv/z/b/a
// widths) may exceed H (packed-gate qkv: N=2*n_heads*hd); the rotation
// covers the first H elements, the tail copies through (host GEMM reads
// min(x.size,K)... no: the GEMM reads K=H inputs; outputs beyond H are the
// packed gate rows produced BY the GEMM. So the act row IS H wide —
// N here is informational; each act row is exactly H FP16 out).
sycl::event fwht_rot_to_half(sycl::queue& q, const FwhtRotParams& p,
                             const sycl::event& dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const float* xn = p.normed_fp32;
  const uint8_t* kind = p.rot_kind;
  const std::uint32_t H = p.H, pr = p.rot_p;
  const float inv = p.rot_inv_sqrt;
  if (!xn || !kind || H == 0 || pr == 0) return sycl::event{};
  std::uint16_t* a0 = p.acts_fp16[0];
  std::uint16_t* a1 = p.acts_fp16[1];
  std::uint16_t* a2 = p.acts_fp16[2];
  std::uint16_t* a3 = p.acts_fp16[3];
  if (!a0 || !a1 || !a2 || !a3) return sycl::event{};
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for<class fwht_rot_to_half_kernel>(
        sycl::range<1>(4), [=](sycl::item<1> it) {
          const std::uint32_t t = (std::uint32_t)it.get_id(0);
          const uint8_t k = kind[t];
          if (k == 2) return;  // dense: caller declines
          std::uint16_t* oo = (t == 0) ? a0 : (t == 1) ? a1
                             : (t == 2)            ? a2
                                                   : a3;
          if (k == 0) {
            for (std::uint32_t i = 0; i < H; ++i)
              oo[i] = dev_float_to_half(xn[i]);
            return;
          }
          // Hadamard: exact host order (fwht len=2..p ascending, then
          // scale; blockdiag tail passthrough). Private array sized to the
          // max H (27B H=5120); larger H declines at the caller.
          float tmp[5120];
          if (H > 5120) return;  // caller guards; never partial work
          for (std::uint32_t i = 0; i < H; ++i) tmp[i] = xn[i];
          for (std::uint32_t len = 2; len <= pr; len <<= 1) {
            const std::uint32_t half = len >> 1;
            for (std::uint32_t i = 0; i < pr; i += len)
              for (std::uint32_t j = 0; j < half; ++j) {
                const float u = tmp[i + j], v = tmp[i + j + half];
                tmp[i + j] = u + v;
                tmp[i + j + half] = u - v;
              }
          }
          for (std::uint32_t i = 0; i < pr; ++i)
            oo[i] = dev_float_to_half(tmp[i] * inv);
          for (std::uint32_t i = pr; i < H; ++i)
            oo[i] = dev_float_to_half(tmp[i]);
        });
  });
}

// L5 kernel 2: conv FIR + silu + q/k/v/g/b prep pack. One WI, sequential,
// host loop order. conv (host step order: c outer over conv_dim, k inner
// over conv_k taps against conv_state[c*3+k...] layout with state shift);
// silu via sycl::exp (~1ulp vs host expf — same standing as L4); q/k
// l2norm reductions sequential (same order as host l2norm — check its loop
// before claiming exact); g via sycl exp/log1p analogs of host
// expf/log1pf (~1ulp). v/b are memcpy-order packs (exact).
sycl::event la_conv_prep(sycl::queue& q, const LaConvPrepParams& p,
                         const sycl::event& dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const float* mqkv = p.mqkv_fp32;
  float* cst = p.conv_state;
  const float* cw = p.conv_w;
  const float* brow = p.b_row;
  const float* arow = p.a_row;
  float* packed = p.packed_in;
  const float* dtb = p.dt_bias;
  const float* Alog = p.A_log;
  const std::uint32_t cdim = p.conv_dim, ck = p.conv_k;
  const std::uint32_t nv = p.nv, nk = p.nk, kd = p.kd, vd = p.vd;
  const float qscale = p.qscale;
  if (!mqkv || !cst || !cw || !packed || !dtb || !Alog) return sycl::event{};
  if (!brow || !arow) return sycl::event{};
  if (cdim == 0 || ck == 0 || nv == 0 || nk == 0 || kd == 0 || vd == 0)
    return sycl::event{};
  const size_t zvw = (size_t)nv * vd;
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.single_task<class la_conv_prep_kernel>([=]() {
      // conv section layout inside mqkv (host step order, conv_k==4):
      // query [0,nk*kd), key [nk*kd,2*nk*kd), value [2*nk*kd,2*nk*kd+zvw).
      // Taps 0..2 read conv_state[c*3+0..2], tap 3 reads mqkv[c] — exactly
      // the 4 host lines. conv_state shifts in place (host order).
      const size_t qk = (size_t)nk * kd;
      // query/key/value section base offsets in conv_out space
      for (std::uint32_t c = 0; c < cdim; ++c) {
        float s = 0.0f;
        if (ck == 4) {
          s = cw[(size_t)c * ck + 0] * cst[(size_t)c * 3 + 0] +
              cw[(size_t)c * ck + 1] * cst[(size_t)c * 3 + 1] +
              cw[(size_t)c * ck + 2] * cst[(size_t)c * 3 + 2] +
              cw[(size_t)c * ck + 3] * mqkv[c];
        } else {
          for (std::uint32_t k = 0; k + 1 < ck; ++k)
            s += cw[(size_t)c * ck + k] * cst[(size_t)c * 3 + k];
          s += cw[(size_t)c * ck + ck - 1] * mqkv[c];
        }
        const float es = sycl::exp(-s);
        const float silu = s / (1.0f + es);
        // shift history, append mqkv (host order)
        cst[(size_t)c * 3 + 0] = cst[(size_t)c * 3 + 1];
        cst[(size_t)c * 3 + 1] = cst[(size_t)c * 3 + 2];
        cst[(size_t)c * 3 + 2] = mqkv[c];
        // stash conv_out[c] = silu into packed's v section layout later;
        // first write to a scratch area past the packed tail (caller sizes
        // packed_in + cdim floats of scratch).
        packed[(2 * qk + zvw + 2 * (size_t)nv) + c] = silu;
      }
      const float* cout = packed + (2 * qk + zvw + 2 * (size_t)nv);
      const float* query = cout;
      const float* key = cout + qk;
      const float* value = cout + 2 * qk;
      float* hqn = packed;
      float* hkn = packed + qk;
      float* hv = packed + 2 * qk;
      float* hg = hv + zvw;
      float* hb = hg + nv;
      for (std::uint32_t kh = 0; kh < nk; ++kh) {
        // l2norm sequential (host l2norm order: sum squares ascending,
        // inv=1/sqrt(s+1e-6), out=x*inv — matched exactly, then q*qscale
        // in the same two-rounding order as the host's separate loop).
        float sq = 0.0f, sk = 0.0f;
        for (std::uint32_t d = 0; d < kd; ++d) {
          sq += query[(size_t)kh * kd + d] * query[(size_t)kh * kd + d];
          sk += key[(size_t)kh * kd + d] * key[(size_t)kh * kd + d];
        }
        const float iq = 1.0f / sycl::sqrt(sq + 1e-6f);
        const float ik = 1.0f / sycl::sqrt(sk + 1e-6f);
        for (std::uint32_t d = 0; d < kd; ++d) {
          hqn[(size_t)kh * kd + d] = query[(size_t)kh * kd + d] * iq * qscale;
          hkn[(size_t)kh * kd + d] = key[(size_t)kh * kd + d] * ik;
        }
      }
      for (size_t i = 0; i < zvw; ++i) hv[i] = value[i];
      for (std::uint32_t i = 0; i < nv; ++i) hb[i] = brow[i];
      for (std::uint32_t hh = 0; hh < nv; ++hh) {
        // host: sp=log1pf(expf(a+dt)); g=-expf(A_log)*sp. sycl exp/log1p
        // analogs (~1ulp vs host expf/log1pf — oracle arbitrates, L4 norm).
        const float sp =
            sycl::log1p(sycl::exp(arow[hh] + dtb[hh]));
        hg[hh] = -sycl::exp(Alog[hh]) * sp;
      }
    });
  });
}

// L5 helper: FP16->FP32 convert (bit-exact dev_half_to_float per element).
sycl::event half_to_float(sycl::queue& q, const HalfToFloatParams& p,
                          const sycl::event& dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const std::uint16_t* xin = p.in_fp16;
  float* xout = p.out_fp32;
  const std::uint32_t n = p.n;
  if (!xin || !xout || n == 0) return sycl::event{};
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for<class half_to_float_kernel>(
        sycl::range<1>(n), [=](sycl::item<1> it) {
          const std::uint32_t i = (std::uint32_t)it.get_id(0);
          xout[i] = dev_half_to_float(xin[i]);
        });
  });
}

sycl::event lm_head_argmax_tile(sycl::queue& q, const LmArgmaxTileParams& p,
                                const sycl::event& dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  const std::uint16_t* tile = p.tile_fp16;
  int* out_idx = p.out_idx;
  float* out_val = p.out_val;
  const std::uint8_t* greedy = p.is_greedy;
  const std::uint32_t n = p.n, M = p.M;
  const std::uint32_t stride = p.row_stride == 0 ? n : p.row_stride;
  const std::uint32_t col0 = p.global_col0;
  if (!tile || !out_idx || !out_val || n == 0 || M == 0) return sycl::event{};
  const std::uint32_t nWG = 32;
  const std::uint32_t global = ((M + nWG - 1) / nWG) * nWG;
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for<class lm_head_argmax_tile_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
          const std::uint32_t m = it.get_global_id(0);
          if (m >= M) return;
          if (greedy && !greedy[m]) {
            out_idx[m] = -1;
            out_val[m] = -1e30f;
            return;
          }
          const std::uint16_t* row = tile + (size_t)m * stride;
          int best = -1;
          float bestv = -1e30f;
          for (std::uint32_t i = 0; i < n; ++i) {
            const float s = dev_half_to_float(row[i]);
            if (s > bestv) {
              bestv = s;
              best = (int)(col0 + i);
            }
          }
          out_idx[m] = best;
          out_val[m] = bestv;
        });
  });
}

}  // namespace intfer::gemm
