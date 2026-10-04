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
#include <cstdio>
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
      // INTFER_DECODE1_MR=1 (EXPERIMENTAL, default OFF): process MR output
      // elements per work-item instead of one. Kill switch
      // INTFER_NO_DECODE1_MR=1 additionally forces it off.
      //
      // *** COVERAGE, AND IT IS A SHORT LIST. READ BEFORE ADDING AN ARM. ***
      // The branch below is `bits == 2`, and the scheme census of the two gate
      // models is decisive:
      //     Qwen3.5-0.8B.intifer        INT4 x76, FP16 x93, INT8 x18
      //                                -> ZERO INT2, so 9 of the 10 correctness
      //                                   gates CANNOT reach this code at all.
      //     Qwen3.8-27B-int2.intifer   INT4 x116, FP16 x256, INT2 x110, INT8 x16
      // The MR path's safety claim -- each output accumulates its own K in
      // ascending kg then ascending j, in its own register, so no FP32 add is
      // reordered -- is a CODE-READ argument, which is not evidence. The one arm
      // that actually tests it therefore lives in scripts/test_tp2_27b.sh, the
      // only gate whose model carries an INT2 tensor, and it asserts the
      // DECODE1_MR stream against the same fixed oracle the base run uses.
      // DO NOT add stream-identity arms for this flag to test_gpu_optin.sh or any
      // other 0.8B gate: they would be green because the path never runs, which
      // is audit #3's nested-flag vacuity in a new place -- a gate that cannot
      // reach its subject must not be allowed to certify it.
      //
      // *** MEASURED 2026-10-04, 27B --tp 2, decode M=1, n=8: 2.3x SLOWER. ***
      // kernel 2055.1 ms -> 4698.7 ms, waitblock 1037.7 -> 2362.5 ms.
      // DEFAULT OFF, and the reason is the most useful thing this experiment
      // produced, so it is recorded here rather than only in a report:
      //
      // At M=1 the total work-item count of this kernel is EXACTLY N -- one
      // output element per work-item -- and that is invariant to the work-group
      // shape. So MR=4 did not add memory parallelism, it DIVIDED an already
      // starved parallelism budget by four. At 27B TP=2 a layer GEMM has
      // N~2048 per rank over 512 Xe cores = ~4 threads/EU, and kThr=16 cannot
      // even fill one SIMD. The kernel is PARALLELISM-bound, not bandwidth-
      // bound, and not issue/MLP-bound.
      //
      // That reframes the whole M=1 decode target: the only ways to raise
      // parallelism are (a) split K across work-items, which reassociates the
      // FP32 sum and is therefore NOT bit-exact, or (b) fuse INDEPENDENT GEMMs
      // into one nd_range so one launch carries several N's worth of
      // work-items, which IS bit-exact because each output keeps its own
      // work-item and its own K order. (b) is the live direction.
      const bool use_mr = (std::getenv("INTFER_DECODE1_MR") != nullptr) &&
                          (std::getenv("INTFER_NO_DECODE1_MR") == nullptr);

      // --- WHY (measured 2026-10-04, 27B --tp 2, decode M=1) ---------------
      // The one-output-per-work-item kernel gives lane i the row
      // `code_u8 + i*K`, so at any instant the subgroup's lanes sit K bytes
      // apart: a stride-K gather, not a coalesced stream. Worse, EVERY work-item
      // re-loads the ENTIRE activation row (`a16.load` inside the K loop), so
      // one GEMM re-reads K halves N times.
      // Measured effect at 27B: decode reads 9.16 GiB of weights per device per
      // token in 257.7 ms of kernel = 35.5 GiB/s, which is ~21 % of the B580's
      // ~179 GB/s. So the kernel is issue/MLP-bound, not bandwidth-bound.
      //
      // MR attacks BOTH causes without touching the arithmetic:
      //   * MR independent weight streams per lane => more memory-level
      //     parallelism to cover the same latency;
      //   * the activation chunk is loaded ONCE per K step and reused across all
      //     MR outputs => MR-fold fewer activation loads.
      //
      // BIT-EXACT, and that is the whole reason this shape was chosen over the
      // obvious alternative. Each output still accumulates its own K in ascending
      // kg, ascending j, into its own register. The tempting fix -- split K
      // across lanes and tree-reduce -- would make the weight loads coalesced
      // too, but it reassociates the FP32 sum and is therefore NOT bit-exact.
      // This changes only WHICH work-item computes an output and how many loads
      // it shares, never the order of any accumulation.
      if (use_mr && bits == 2 && use_int2_tab && num_groups > 0) {
        constexpr int MR = 4;
        const int per_group = nWG * MR;
        const int nB = (N + per_group - 1) / per_group;
        const int global_mr = nB * nWG;
        // *** THE PATH ANNOUNCES ITSELF, ONCE. *** Same shape as INTFER_L4_DBG
        // (main.cpp:3115) and for the same reason. MEASURED 2026-10-04: with
        // nothing here, scripts/test_tp2_27b.sh's DECODE1_MR arm compares the
        // MR run's stream against the base run's stream, and if the flag never
        // reached this branch -- renamed, typo'd, dropped by a shell wrapper, or
        // a `bits == 2` tensor that did not occur on this model -- the two runs
        // are IDENTICAL BY CONSTRUCTION and the arm certifies nothing while
        // printing nothing wrong. That is audit #3's nested-flag vacuity, one
        // level below the one that commit fixed: it cannot see whether its own
        // subject ran. A one-time host-side stderr line makes it seeable, and
        // the gate asserts the line is present.
        static bool announced = false;
        if (!announced) {
          announced = true;
          std::fprintf(stderr,
                       "[decode1-mr] M=1 INT2 decode: MR=%d kernel ACTIVE "
                       "(N=%d K=%d bits=%d)\n",
                       MR, N, K, bits);
        }
        return q.submit([&](sycl::handler& h) {
          h.depends_on(copy_dep);
          h.parallel_for<class gemm_usm_decode1_vec_mr>(
              sycl::nd_range<1>(sycl::range<1>(global_mr), sycl::range<1>(nWG)),
              [=](sycl::nd_item<1> it) {
                const int ng0 = static_cast<int>(it.get_global_id(0)) * MR;
                if (ng0 >= N) return;
                const int nvalid =
                    (N - ng0) < MR ? (N - ng0) : MR;

                float acc[MR];
                const std::uint16_t* sptr[MR];
                const std::uint8_t* wrow[MR];
                for (int r = 0; r < MR; ++r) {
                  acc[r] = 0.0f;
                  sptr[r] = scale_usm + (ng0 + r) * num_groups_w;
                  wrow[r] = code_u8 + static_cast<std::size_t>(ng0 + r) * K;
                }
                sycl::multi_ptr<const std::uint16_t,
                    sycl::access::address_space::global_space> act_mp(act_usm);

                const float alpha_f = alpha;
                const std::size_t pack_sz = 4;  // 8 codes / 4 bytes at INT2

                for (int g = 0; g < num_groups; ++g) {
                  const int kg0 = g * Gw;
                  int kg1 = kg0 + Gw;
                  if (kg1 > static_cast<int>(K)) kg1 = (int)K;
                  // Per-output group scale folded into a 4-entry register table
                  // (identical algebra and identical op order to the 1-output
                  // path's wtab -- see the note there).
                  float wtab[MR][4];
                  for (int r = 0; r < nvalid; ++r) {
                    const float sc = from_half_bits(sptr[r][g]) * alpha_f;
                    wtab[r][0] = sc * -1.0f;
                    wtab[r][1] = sc * (-1.0f / 3.0f);
                    wtab[r][2] = sc * (1.0f / 3.0f);
                    wtab[r][3] = sc * 1.0f;
                  }
                  int kg = kg0;
                  std::size_t b0[MR];
                  for (int r = 0; r < nvalid; ++r)
                    b0[r] = (static_cast<std::size_t>(ng0 + r) * K +
                             static_cast<std::size_t>(kg)) / pack_sz;
                  for (; kg + 8 <= kg1; kg += 8) {
                    // ONE activation load, reused by every output.
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    for (int r = 0; r < nvalid; ++r) {
                      const std::uint8_t b0v = code_u8[b0[r]];
                      const std::uint8_t b1v = code_u8[b0[r] + 1];
                      b0[r] += 2;
                      for (int j = 0; j < 8; ++j) {
                        const float av = static_cast<float>(
                            sycl::bit_cast<sycl::half>(a16[j]));
                        const std::uint8_t bv = (j < 4) ? b0v : b1v;
                        const int code = (bv >> ((j & 3) * 2)) & 3;
                        acc[r] += av * wtab[r][code];
                      }
                    }
                  }
                  // Scalar tail: same order, same table, same products.
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    for (int r = 0; r < nvalid; ++r) {
                      const std::size_t idx =
                          static_cast<std::size_t>(ng0 + r) * K +
                          static_cast<std::size_t>(kg);
                      const std::uint8_t byte = code_u8[idx / pack_sz];
                      const int within = (int)(idx % pack_sz);
                      const int code = (byte >> (within * 2)) & 3;
                      acc[r] += av * wtab[r][code];
                    }
                  }
                }
                for (int r = 0; r < nvalid; ++r)
                  out_usm[ng0 + r] = to_half_bits(acc[r]);
              });
        });
      }

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

// Device-side GA64 act-quant pre-pass (see header for bit-contract).
sycl::event act_quant_ga64_dev(sycl::queue& q, const ActQuantDevParams& p) {
  constexpr uint32_t GA = 64;
  const uint32_t M = p.M, K = p.K;
  if (M == 0 || K == 0 || !p.x_f32 || !p.scale_inv_in || !p.q_out || !p.s_out)
    return sycl::event{};
#ifdef INTFER_PROFILE_GPU
  // AFTER the decline check: a declined call submits no kernel, so counting
  // it here would inflate nk relative to actual device work.
  profile_note_launch();
#endif
  const uint32_t nGA = (K + GA - 1) / GA;
  const uint32_t Ks = (p.Kstride == 0) ? K : p.Kstride;
  const uint32_t Ss = (p.nGAstride == 0) ? nGA : p.nGAstride;
  const uint32_t nWG = 32;
  const uint32_t n_blocks = M * nGA;
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;
  const float* X = p.x_f32;
  const float* SI = p.scale_inv_in;
  std::int8_t* Q0 = p.q_out;
  float* S0 = p.s_out;
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class act_quant_ga64_dev_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
          const uint32_t b = it.get_global_id(0);
          if (b >= n_blocks) return;
          const uint32_t m = b / nGA;
          const uint32_t g = b % nGA;
          const uint32_t k0 = g * GA;
          uint32_t k1 = k0 + GA;
          if (k1 > K) k1 = K;
          const float* xr = X + (size_t)m * K;
          // Scale AND reciprocal come from the HOST (see
          // ActQuantDevParams::scale_inv_in), which is why there is no
          // per-slice max-abs reduction here: the host already did it, and
          // doing it twice only created a way for the scale we WRITE to
          // disagree with the inverse we multiply by.
          //
          // Do NOT recompute `1.0f / s` here: q = trunc(x*inv +- 0.5) turns a
          // 1-ulp reciprocal difference into a whole +-1 code error at .5
          // ties, and `1.0f/s` is not a portable bit-source (a device
          // division lowers to rcp.approx, 1 ulp off on ~10% of legal
          // scales). An earlier revision "fixed" this with an fma Newton
          // step on the device, which made the device correctly rounded and
          // therefore moved it AWAY from the host — that is what perturbed the
          // greedy stream at decode step 14. The host side now routes every
          // GA64 reciprocal through one noinline site (dashq::ga64_recip);
          // this kernel consumes those exact bits.
          const float* si = SI + 2 * ((size_t)m * nGA + g);
          const float s = si[0];
          const float inv = si[1];
          // Row-strided workspace write; each element is quanted ONCE and
          // stored ONCE (the fused-quant failure mode replicated the O(K)
          // quant work nT x, once per 16-N tile).
          std::int8_t* qo = Q0 + (size_t)m * Ks + k0;
          for (uint32_t k = k0; k < k1; ++k) {
            float v = xr[k] * inv;
            if (v > 127.0f) v = 127.0f;
            else if (v < -127.0f) v = -127.0f;
            // Truncation of (v +- 0.5) IS half-away-from-zero: same result
            // as the host std::lround and as the AVX2 path's
            // (v + (sign|0.5)) followed by cvttps, for every |v| <= 127.
            qo[k - k0] = (std::int8_t)(v >= 0 ? v + 0.5f : v - 0.5f);
          }
          S0[(size_t)m * Ss + g] = s;
        });
  });
}

sycl::event quant_gemm_dpas48_rowfold(sycl::queue& q,
                                      const std::int8_t* act_int8,
                                      std::int32_t* fold_usm,
                                      uint32_t M, uint32_t K,
                                      uint32_t act_stride,
                                      uint32_t fold_stride,
                                      sycl::event dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  constexpr uint32_t OP = 32, TN = 16;
  // Self-guarding: any geometry the DPAS kernel cannot consume exactly, this
  // declines and the caller keeps the inline fold. K%32!=0 is included because
  // quant_gemm_dpas48 assumes whole K-ops; folding a truncated op count would
  // silently agree with a wrong GEMM instead of refusing.
  if (M == 0 || K < OP || (K % OP) != 0) return dep;
  const uint32_t nOp = K / OP;
  const uint32_t blocks = M * nOp;
  // Accept/reject through the SAME predicate the readers use. The rule is
  // "stride must be >= 2*nOp" (each row owns 2*nOp ints, [S0][Sfull], so a
  // shorter stride lets row m's Sfull overwrite row m+1's S0 -- silent at
  // M=1). Declining is always safe: the caller keeps the inline fold.
  // AUDIT #13: this used to be an open-coded `fold_stride < 2*nOp` here, a
  // second copy of a rule the header says never to hand-roll. The null check
  // is the writer-side mirror of fold_readable()'s: the writer dereferences
  // fold_usm unguarded at dst = fold_usm + m*fold_stride, so a null buffer
  // would be a device-side null write rather than a decline.
  if (!fold_usm || !fold_stride_ok(fold_stride, nOp)) return dep;
  fold_stride = fold_row_stride(fold_stride, nOp);
  const uint32_t global = ((blocks + TN - 1) / TN) * TN;
  return q.submit([&](sycl::handler& h) {
    if (dep != sycl::event{}) h.depends_on(dep);
    h.parallel_for<class gemm_dpas48_rowfold_kernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(TN)),
        [=](sycl::nd_item<1> it) {
          const uint32_t b = it.get_global_id(0);
          if (b >= blocks) return;
          const uint32_t m = b / nOp;
          const uint32_t o = b - m * nOp;
          const std::int8_t* Aq =
              act_int8 + (size_t)m * act_stride + (size_t)o * OP;
          // Same element order as the inline fold in quant_gemm_dpas48; the
          // result is an exact int32 sum so the order cannot change it.
          int s0 = 0, sf = 0;
          for (uint32_t k = 0; k < OP; ++k) {
            const int v = (int)Aq[k];
            sf += v;
            if (k < 16) s0 += v;
          }
          std::int32_t* dst = fold_usm + (size_t)m * fold_stride;
          dst[o] = (std::int32_t)s0;
          dst[nOp + o] = (std::int32_t)sf;
        });
  });
}

// ---------------------------------------------------------------------------
// DPAS48 with a COOPERATIVE 2D WORK-ITEM BLOCK (E4, default OFF).
//
// R == 1 is the shipped geometry: one WI per (row m, 16-N tile t). R > 1 gives
// one WI a 2D block of R rows x ONE tile, so the B op-tile is fetched once per
// op and reused across R rows, instead of being re-fetched by every row's WI.
// This is exactly the load-replication half that E3 (INTFER_DPAS48_FOLD)
// deliberately left alone: E3 removed the ARITHMETIC replication, this removes
// the LOAD.
//
// WHY IT IS BIT-EXACT, and why that is a design property rather than luck:
// every row still accumulates acc_tile[r][nl] over ops o = 0..nOp-1 in the same
// order, with the same expression and the same operands. Sharing vb across rows
// reorders no FP32 add at all. The hazard E4's numerics contract was written
// for -- per-block partial sums combined in a different order -- DOES NOT ARISE
// in this geometry, because there are no partial sums: there is one
// accumulator per (row, tile), exactly as before.
//
// Per that contract, identity must still be CHECKED and REPORTED, never turned
// into the gate; the tolerance gate (backward_err / ULP / fraction-changed) stays
// the gate. That contract also warned that a real kernel "may need a different
// bound" because its numbers came from a SYNTHETIC term generator. Measured
// against the real kernel, this geometry is exactly identical, so the question
// is answered rather than assumed -- see the E4 readback arms in
// scripts/test_dpas_fold.sh.
//
// NOT the same experiment as INTFER_DPAS_ROW (quant_gemm_dpas4_row), which put
// the whole TILE LOOP inside one WI and measured ~460x SLOWER because it
// collapsed occupancy. Here tile parallelism is preserved exactly -- one 16-N
// tile per WI, WI-per-tile parallelism kept -- and only the ROW count per tile
// grows. That is what the "real cooperation" warning in that experiment's
// comment asks for, and it is why this design does not repeat its failure.
// ---------------------------------------------------------------------------
// A SYCL kernel NAME is a type, and a templated kernel name therefore has to
// exist as a declared (never defined, never instantiated) class template before
// it can be named in parallel_for<...>. Instantiating it for R==1 and R==2 is
// what gives the two geometries distinct kernel identities.
template <uint32_t R>
class gemm_dpas48_tiled_kernel;

namespace {
template <uint32_t R>
sycl::event dpas48_tiled(sycl::queue& q, const Dpas48Params& p,
                         sycl::event dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
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
  // INTFER_DPAS48_FOLD: when the caller ran quant_gemm_dpas48_rowfold, the
  // per-(row,op) sums are precomputed and every tile WI just reads them,
  // instead of redoing the identical O(K) integer fold once per 16-N tile.
  // Exact int32 values, so this is a pure work hoist -- not a numeric change.
  // One shared layout rule with the writer (fold_readable): a fold buffer
  // whose stride does not clear 2*nOp is IGNORED and the kernel falls back to
  // the inline fold, rather than reading Sfull out of the wrong row's S0.
  const uint32_t fold_stride = fold_row_stride(p.fold_stride, nOp);
  const std::int32_t* Fold = fold_readable(p.fold_usm, p.fold_stride, nOp);

  // One WI per (row-group of R, 16-N tile t). B op-tiles reload per WI (L1-
  // resident across the M rows of a tile); A is per-row. Same fold math as the
  // M=1 probe (probe_dpas_fold.cpp T5/T6, 1e-16 exact).
  const uint32_t nWG = 16;
  const uint32_t nMG = (M + R - 1) / R;
  const uint32_t n_blocks = nMG * nT;
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;

  return q.submit([&](sycl::handler& h) {
    if (dep != sycl::event{}) h.depends_on(dep);
    h.parallel_for<class gemm_dpas48_tiled_kernel<R>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const uint32_t block = it.get_global_id(0);
          if (block >= n_blocks) return;
          // NOTE the decode is (t = block % nT) FIRST, then the row group. At
          // R==1 that is bit-for-bit the shipped mapping (m = block / nT,
          // t = block % nT), so the default path is unchanged -- which is why
          // this can be one template instead of a second copy of the math.
          const uint32_t t = block % nT;
          const uint32_t mg = block / nT;
          const uint32_t m0 = mg * R;
          const std::int8_t* Aq[R];
          const float* As[R];
          float acc_tile[R][TN];
#pragma unroll
          for (uint32_t r = 0; r < R; ++r) {
            const uint32_t m = m0 + r;
            Aq[r] = p.act_int8 + (size_t)m * act_stride;
            As[r] = p.act_scale + (size_t)m * as_stride;
#pragma unroll
            for (uint32_t nl = 0; nl < TN; ++nl) acc_tile[r][nl] = 0.0f;
          }
          for (uint32_t o = 0; o < nOp; ++o) {
            const uint32_t k0 = o * OP;
            const uint32_t g = k0 / G;
            if (BPrec == dpas_argument_type::u4) {
              simd<uint8_t, 256> vb;  // op_bytes(u4) = 32*16*4/8
              vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes);
#pragma unroll
              for (uint32_t r = 0; r < R; ++r) {
                const uint32_t m = m0 + r;
                if (m >= M) continue;
                simd<std::int8_t, OP> va;
                va.copy_from(Aq[r] + k0);
                const float as = As[r][k0 / GA];
                simd<int, TN> c;
                c = dpas<8, 1, int, uint8_t, std::int8_t,
                         dpas_argument_type::u4, dpas_argument_type::s8>(vb, va);
                int S;
                if (Fold) S = Fold[fold_index_sfull(m, fold_stride, nOp, o)];
                else { S = 0; for (uint32_t k = 0; k < OP; ++k) S += va[k]; }
#pragma unroll
                for (uint32_t nl = 0; nl < TN; ++nl) {
                  const uint32_t n = t * TN + nl;
                  const float ws = static_cast<float>(
                      sycl::bit_cast<sycl::half>(
                          p.scale_usm[(size_t)n * nG + g]));
                  acc_tile[r][nl] += as * ws * alpha *
                                  (2.0f * (float)c[nl] - 15.0f * (float)S) *
                                  Cinv;
                }
              }
            } else if (BPrec == dpas_argument_type::u8) {
              simd<uint8_t, 512> vb;  // op_bytes(u8) = 32*16*8/8
              vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes);
#pragma unroll
              for (uint32_t r = 0; r < R; ++r) {
                const uint32_t m = m0 + r;
                if (m >= M) continue;
                simd<std::int8_t, OP> va;
                va.copy_from(Aq[r] + k0);
                const float as = As[r][k0 / GA];
                simd<int, TN> c;
                c = dpas<8, 1, int, uint8_t, std::int8_t,
                         dpas_argument_type::u8, dpas_argument_type::s8>(vb, va);
                int S;
                if (Fold) S = Fold[fold_index_sfull(m, fold_stride, nOp, o)];
                else { S = 0; for (uint32_t k = 0; k < OP; ++k) S += va[k]; }
#pragma unroll
                for (uint32_t nl = 0; nl < TN; ++nl) {
                  const uint32_t n = t * TN + nl;
                  const float ws = static_cast<float>(
                      sycl::bit_cast<sycl::half>(
                          p.scale_usm[(size_t)n * nG + g]));
                  acc_tile[r][nl] += as * ws * alpha *
                                  (2.0f * (float)c[nl] - 255.0f * (float)S) *
                                  Cinv;
                }
              }
            } else {
              // u2: K=32 op, E=16 rows/dword, 32*16*2/8 = 128B.
              simd<uint8_t, 128> vb;
              vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes);
              if (!need_split) {
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) {
                  const uint32_t m = m0 + r;
                  if (m >= M) continue;
                  simd<std::int8_t, OP> va;
                  va.copy_from(Aq[r] + k0);
                  const float as = As[r][k0 / GA];
                  simd<int, TN> c;
                  c = dpas<8, 1, int, uint8_t, std::int8_t,
                           dpas_argument_type::u2, dpas_argument_type::s8>(vb, va);
                  int S = 0;
                  for (uint32_t k = 0; k < OP; ++k) S += va[k];
#pragma unroll
                  for (uint32_t nl = 0; nl < TN; ++nl) {
                    const uint32_t n = t * TN + nl;
                    const float ws = static_cast<float>(
                        sycl::bit_cast<sycl::half>(
                            p.scale_usm[(size_t)n * nG + g]));
                    acc_tile[r][nl] += as * ws * alpha *
                                    (2.0f * (float)c[nl] - 3.0f * (float)S) *
                                    Cinv;
                  }
                }
              } else {
                // G16 split: T_full on va, T_0 on va masked to first 16.
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) {
                  const uint32_t m = m0 + r;
                  if (m >= M) continue;
                  simd<std::int8_t, OP> va;
                  va.copy_from(Aq[r] + k0);
                  const float as = As[r][k0 / GA];
                  simd<int, TN> cf, c0;
                  cf = dpas<8, 1, int, uint8_t, std::int8_t,
                            dpas_argument_type::u2, dpas_argument_type::s8>(vb, va);
                  simd<std::int8_t, OP> va0(0);
                  int S0, Sf;
                  if (Fold) {
                    S0 = Fold[fold_index_s0(m, fold_stride, o)];
                    Sf = Fold[fold_index_sfull(m, fold_stride, nOp, o)];
                  } else {
                    S0 = 0;
                    for (uint32_t k = 0; k < 16; ++k) {
                      va0[k] = va[k];
                      S0 += va[k];
                    }
                    Sf = S0;
                    for (uint32_t k = 16; k < OP; ++k) Sf += va[k];
                  }
                  c0 = dpas<8, 1, int, uint8_t, std::int8_t,
                            dpas_argument_type::u2, dpas_argument_type::s8>(vb, va0);
                  const uint32_t g0 = k0 / G, g1 = g0 + 1;
#pragma unroll
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
                    acc_tile[r][nl] += as * alpha *
                                    ((2.0f * T0 - 3.0f * (float)S0) * ws0 +
                                     (2.0f * T1 - 3.0f * S1) * ws1) *
                                    Cinv;
                  }
                }
              }
            }
          }
#pragma unroll
          for (uint32_t r = 0; r < R; ++r) {
            const uint32_t m = m0 + r;
            if (m >= M) continue;
            std::uint16_t oh[TN];
            for (uint32_t nl = 0; nl < TN; ++nl)
              oh[nl] = to_half_bits(acc_tile[r][nl]);
            simd<std::uint16_t, TN> oc;
            oc.copy_from(oh);
            oc.copy_to(p.out_usm + (size_t)m * N + (size_t)t * TN);
          }
        });
  });
}
}  // namespace

sycl::event quant_gemm_dpas48(sycl::queue& q, const Dpas48Params& p,
                              sycl::event dep) {
  return dpas48_tiled<1>(q, p, dep);
}

// E4 entry point. DECLINES below the block size rather than running a 2-row
// block over a 1-row call: a decline is observable (the caller keeps the
// shipped kernel and the stream is unchanged), whereas a "works at any M"
// variant would hide a real precondition and make the flag's effect
// unattributable. At M==1 -- i.e. all of decode -- E4 is inert BY CONSTRUCTION,
// which is exactly what the gate asserts.
sycl::event quant_gemm_dpas48_2d(sycl::queue& q, const Dpas48Params& p,
                                 sycl::event dep) {
  // Decline by RETURNING `dep` UNCHANGED, not a fresh event. That is what makes
  // a decline observable at all: the caller passes a sentinel prerequisite and
  // tests `e == dep`, because comparing against a bare sycl::event{} is NOT
  // reliable across implementations (recorded in this probe's own NEG cases).
  // Returning {} here instead would be undetectable whenever dep is non-empty.
  if (p.M < 2) return dep;
  return dpas48_tiled<2>(q, p, dep);
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
sycl::event quant_gemm_dpas4_row(sycl::queue& q, const Dpas48Params& p,
                                sycl::event dep) {
#ifdef INTFER_PROFILE_GPU
  profile_note_launch();
#endif
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::esimd::xmx;
  constexpr uint32_t TN = 16, OP = 32;
  constexpr uint32_t RSTRIP = 4;
  const uint32_t M = p.M, N = p.N, K = p.K;
  const uint32_t G = (p.group_size == 0) ? K : p.group_size;
  // Decline by RETURNING `dep` UNCHANGED, the same convention as
  // quant_gemm_dpas48_2d. It used to return a bare `sycl::event{}`, and that is
  // NOT reliably observable: on this toolchain an event produced by
  // `return sycl::event{}` does not compare equal to a bare `sycl::event{}`
  // built at the call site (measured; dpas_fold_probe asserts it). The rule is
  // dpas4_row_eligible() -- ONE home, shared with the engine -- so this branch is
  // defence in depth rather than the mechanism the caller depends on.
  if (!dpas4_row_eligible(p.bits, G)) return dep;
  const uint32_t GA = (p.act_group == 0) ? K : p.act_group;
  const uint32_t nT = N / TN, nOp = K / OP;
  const uint32_t nGA = (K + GA - 1) / GA;
  const uint32_t nG = (K + G - 1) / G;
  const uint32_t act_stride = (p.act_stride == 0) ? K : p.act_stride;
  const uint32_t as_stride = (p.as_stride == 0) ? nGA : p.as_stride;
  const uint32_t op_bytes = OP * TN * 4 / 8;  // u4
  const uint8_t* B = static_cast<const uint8_t*>(p.code_dpas);
  const float alpha = p.alpha;
  // Same INTFER_DPAS48_FOLD contract as quant_gemm_dpas48. This kernel already
  // evaluates the fold once per (row, op) (it sits outside the tile strip
  // loop), so it does not NEED the hoist — but reading the precomputed values
  // is still exact and keeps the flag uniform across both DPAS geometries.
  // One shared layout rule with the writer (fold_readable): a fold buffer
  // whose stride does not clear 2*nOp is IGNORED and the kernel falls back to
  // the inline fold, rather than reading Sfull out of the wrong row's S0.
  const uint32_t fold_stride = fold_row_stride(p.fold_stride, nOp);
  const std::int32_t* Fold = fold_readable(p.fold_usm, p.fold_stride, nOp);

  const uint32_t nWG = 16;
  const uint32_t global = ((M + nWG - 1) / nWG) * nWG;

  return q.submit([&](sycl::handler& h) {
    if (dep != sycl::event{}) h.depends_on(dep);
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
              int S;
              if (Fold) S = Fold[fold_index_sfull(m, fold_stride, nOp, o)];
              else { S = 0; for (uint32_t k = 0; k < OP; ++k) S += va[k]; }
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
