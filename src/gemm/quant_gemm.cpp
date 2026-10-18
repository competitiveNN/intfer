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
#include <cstring>
#include <mutex>
#include <set>
#include <string>
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

// ---- DECODE1 ARM ANNOUNCEMENT -------------------------------------------------
// A function-local `static const bool announced = [] { fprintf(...); }();` inside
// EACH arm's guard is, in principle, a per-arm latch -- and in practice it made
// `[decode1-rowpair]` appear for EVERY non-base order in examples/rowpair_equiv,
// including orders whose env the binary had just set to kill rowpair. Six such
// statics, six identical names, and a harness that could not tell which arm had run.
// This replaces all six with ONE explicitly-keyed function:
//
//   * the latch is keyed by the arm's TAG, held in a set, so it is per-arm BY
//     CONSTRUCTION rather than per-arm by accident of lexical scope. Two arms can
//     never suppress each other's line, and re-entering an arm cannot re-print.
//   * the line carries a CALL COUNTER, so "announced" and "how many launches took
//     this branch" are both observable, and a latch that fires late is visible.
//   * INTFER_DECODE1_ANNOUNCE_EVERY=1 disables the once-only latch and prints every
//     launch, which is how the harness can PROVE the latch is not what decides
//     which arm appears.
//
// It is deliberately host-side and NOT a device function: extracting arithmetic
// into a shared device function is what cost 231114 ms once already.
inline void decode1_announce(const char* tag) {
  static std::set<std::string> seen;
  static std::mutex mu;
  static int n_calls = 0;
  const bool every =
      (std::getenv("INTFER_DECODE1_ANNOUNCE_EVERY") != nullptr);
  {
    std::lock_guard<std::mutex> lk(mu);
    ++n_calls;
    if (!every && !seen.insert(tag).second) return;
    const int n = n_calls;
    if (every) {
      std::fprintf(stderr, "[%s] call %d ACTIVE\n", tag, n);
    } else {
      std::fprintf(stderr, "[%s] call %d ACTIVE (once; "
                           "set INTFER_DECODE1_ANNOUNCE_EVERY=1 for every launch)\n",
                   tag, n);
    }
  }
}

inline float lf(sycl::half h) { return static_cast<float>(h); }
inline std::uint16_t to_half_bits(float f) {
  return sycl::bit_cast<std::uint16_t>(sycl::half(f));
}
inline float from_half_bits(std::uint16_t h) {
  return static_cast<float>(sycl::bit_cast<sycl::half>(h));
}

// ---- SSM pass-fusion announcement -------------------------------------------
// INTFER_SSM_FUSE (default OFF, PRESENCE enables) folds the SSM recurrence's
// four kd-serial passes over the state S into two, by consuming each updated
// element in the same d-iteration that produced it:
//   pass 1+2: S[d,c] *= g_t; kv += S[d,c]*kn[d]   (decay result feeds kv dot)
//   pass 3+4: S[d,c] += kn[d]*delta; s += S[d,c]*qn[d]  (update feeds readout)
// Bit-exact vs the 4-pass base: pass 1 writes S[d,c] in place and pass 2 reads
// the SAME in-register value the store would have produced (the decay touches
// only S[d,c], never S[d',c]); same for passes 3+4. The accumulation order is
// unchanged (d ascending in both), so the FP results are identical and the
// token stream cannot move. The win is S memory traffic: one load+store pair
// replaces a load+store+load triple per (d, head, column).
//
// TP-agnostic: the SSM kernel runs on device 0 regardless of --tp, so this is
// the one fusion the PARAMOUNT DIRECTIVE's TP=2 target can actually reach --
// unlike the host<->device seams (run_mlp_fused / run_linear_tp_devact /
// L4/L5 fused), which all open with `num_devices_ != 1 => return false`.
//
// Same once-per-process latch shape as decode1_announce (keyed by tag, call
// counter, INTFER_SSM_FUSE_ANNOUNCE_EVERY=1 for every launch).
inline void ssm_fuse_announce(const char* tag) {
  static std::set<std::string> seen;
  static std::mutex mu;
  static int n_calls = 0;
  const bool every =
      (std::getenv("INTFER_SSM_FUSE_ANNOUNCE_EVERY") != nullptr);
  {
    std::lock_guard<std::mutex> lk(mu);
    ++n_calls;
    if (!every && !seen.insert(tag).second) return;
    const int n = n_calls;
    if (every) {
      std::fprintf(stderr, "[%s] call %d ACTIVE\n", tag, n);
    } else {
      std::fprintf(stderr, "[%s] call %d ACTIVE (once; "
                           "set INTFER_SSM_FUSE_ANNOUNCE_EVERY=1 for every "
                           "launch)\n",
                   tag, n);
    }
  }
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
  // *** LAUNCH-NOTE ACCOUNTING (2026-10-13 AUDIT). *** This entry point does
  // NOT call profile_note_launch(), and that is CORRECT -- it is counted by
  // its callers, which all drain the returned event and read its profiling
  // timestamps. Do not "fix" this by adding a note.
  //
  // The twelve kernels that DO take the note (act_quant_ga64_dev, the three
  // dpas48_* row variants, dpas48_tiled[1d/2d], the four SSM/norm kernels,
  // fwht_rot_to_half, la_conv_prep, half_to_float, lm_head_argmax_tile) are
  // all free functions whose callers (gpu_ssm_rows, the L4 fused path, the
  // pre-act-quant probe) do NOT loop over events or read timestamps.
  // profile_note_launch exists precisely for THAT class.
  //
  // quant_gemm_forward_usm is the OPPOSITE class: every caller drains and
  // counts it. Engine paths drain the returned event in the post-wait loop
  // (engine.cpp:1501-1508 run_linear_tp, 1878-1884 run_mlp_gateup,
  // 2140-2146 run_linear_wave, 2451-2458 run_linear_wave), and the lm_head
  // callers in main.cpp read command_start/command_end inline
  // (1934-1937, 2150-2153, 2438-2441). Adding a note here would DOUBLE-count
  // every engine launch of this kernel -- the engine's own comment at
  // engine.cpp:1831-1842 records the same prohibition for gateup.
  //
  // THE INT8-DPAS BRANCH (line 430) IS NOT a gap. It dispatches
  // quant_gemm_dpas_int8, which has no note -- but that branch is unreachable
  // from the engine: (1) it requires scheme==INT8 && group_size==0, and that
  // combination makes dpas48_eligible() true (engine.cpp:2687), which routes
  // to run_linear_tp_dpas48 -- which calls the dpas48_* kernels DIRECTLY, not
  // this branch; (2) the branch also needs ActScheme::INT8 activations, but
  // the shard cache hardcodes act_scheme=FP16 (engine.hpp:279) with ZERO
  // assignments anywhere in the tree, so it is dead code gated by "TODO 6".
  // The lm_head INT8 path is quant_gemm_dpas_s8w (INT8 *weights*, FP16 acts),
  // a different kernel that already carries profile_note_launch() (3153).
  // So there is no uncounted launch on the live engine path and no fix to
  // make here -- record the negative, do not add the note.
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
    // INTFER_DECODE1_WG (default 0 => kThr=16, the shipped value): work-group
    // width for the M=1 decode path. The scalar arms below (VERBHALF, VERBMP,
    // VERBDEAD, VERBSEL, VERBVEC, WVEC, MR -- every arm with NO local_accessor)
    // are a PURE MAP over the output element ng: one work-item per ng, each
    // work-item writes exactly out_usm[ng], no cross-WI reduction and no shared
    // state. So nWG is a pure SCHEDULING parameter -- every output element is
    // written by exactly one work-item regardless of the group width, and the
    // FP32 accumulation order inside each work-item is unchanged. BIT-EXACT BY
    // CONSTRUCTION, the same partition argument as INTFER_DPAS48_WG (quant_
    // gemm.cpp:3378) and INTFER_LMHEAD_TILE_ROWS (main.cpp:1667).
    //
    // The B580 has 20 Xe cores / 256 EU against the hardcoded 16; this is the
    // DECODE analogue of the prefill lever INTFER_DPAS48_WG, and it is a
    // dispatch knob with no numerics contract -- UNMEASURED, operator-only.
    //
    // *** SPLITK composition, and why it is NOT a bit-exact concern. *** The one
    // arm that uses a local_accessor reduction (SPLITK,
    // quant_gemm.cpp:1702) derives its partial count S from nWG, so composing
    // this flag with INTFER_DECODE1_SPLITK=1 changes S. That arm is already
    // tolerance-gated (NOT bit-exact by contract; the ascending-combine
    // contract is measured by examples/ksplit_contract) and already
    // default-OFF and MEASURED-LOSS, so the composition stays inside its
    // existing contract class -- more/fewer partials, same ascending combine.
    // The gate arm runs on the DEFAULT path (VERBHALF), where the claim holds
    // exactly, so the composition needs no separate certification.
    int nWG = kThr;
    if(const char* wg_env = std::getenv("INTFER_DECODE1_WG")){
      long v = std::atol(wg_env);
      if(v > 0) nWG = (int)v;
    }
    // THE PATH ANNOUNCES ITSELF, ONCE -- AND ONLY WHEN THE FLAG IS SET.
    // Same shape as the [dpas48-wg] line and every decode1_* latch: a
    // base-vs-flagged stream comparison is satisfiable by a flag that never
    // reached the branch (the partition is exact, so the two streams are
    // identical by construction), so the arm asserts the WIDTH the kernel
    // actually used rather than a constant.
    //
    // *** WHY THE ANNOUNCEMENT IS GATED ON THE ENV, NOT UNCONDITIONAL. ***
    // Unlike INTFER_DPAS48_WG (whose line lives INSIDE run_linear_tp's
    // dpas48 branch, reachable only under the INTFER_DPAS48=1 path selector,
    // so it cannot fire on a base run), THIS knob rides the DEFAULT decode
    // path with no selector -- so an unconditional once-per-process
    // announcement prints on the BASE run too. MEASURED DEFECT, 2026-10-15:
    // that was exactly what it did, and scripts/test_rowpair_equiv.sh went
    // red on all eight arms with "base printed a marker but no arm was
    // selected: [decode1-wg]" -- the gate's marker contract is that the base
    // order prints NO [decode1-*] marker, because a marker on base is an arm
    // that ran when none was selected. Gating on presence of the env restores
    // both halves at once: the line is now a genuine reachability marker (it
    // CANNOT print unless the flag exists), and base stays quiet.
    if(const char* wg_env = std::getenv("INTFER_DECODE1_WG")){
      static bool announced_wg = false;
      if(!announced_wg){
        announced_wg = true;
        std::fprintf(stderr,
                     "[decode1-wg] M=1 decode: work-group=%d "
                     "(INTFER_DECODE1_WG=%s) ACTIVE (N=%d K=%d bits=%d)\n",
                     nWG, wg_env, N, K, bits);
        std::fflush(stderr);
      }
    }
    const int nBlocks = (N + nWG - 1) / nWG;
    const int global = nBlocks * nWG;

    // ====================================================================
    // DECODE1-BISECT EXCLUSIVITY GUARD. Placed HERE, at the top of the M=1
    // block and BEFORE every arm, because it is the only position from which
    // it can observe anything.
    //
    // *** WHY IT WAS DEAD CODE WHERE IT USED TO BE (commit 30f3840). *** It
    // sat after the ROWPAIR / VERBHALF / VERBMP / VERBDEAD / VERBSEL blocks,
    // and EVERY one of those ends in `return q.submit(...)`. So control
    // never reached it from those five arms -- its `use_rowpair`,
    // `use_verbdead` and `use_verbsel` entries were unreachable, and it could
    // only ever police pairs among the three arms that dispatch AFTER it
    // (verbvec, wvec, mr). Worse, it never counted use_splitk, which
    // dispatches after it and IS reachable -- so INTFER_DECODE1_SPLITK=1
    // plus any later arm was a state it could not see. The predicate was
    // right; the position was not, and a guard that cannot see its subjects
    // certifies nothing.
    //
    // *** THE PREDICATE IS "HOW MANY OPT-IN ARMS CAN DISPATCH", NOT "HOW
    // MANY FLAGS ARE SET". *** VERBHALF and VERBMP are default-ON kill
    // switches (promoted 2026-10-07), so the production state is ZERO
    // opt-in flags set and BOTH defaults live. Counting them would refuse
    // production -- which is exactly the bug 30f3840 fixed -- so they are
    // excluded from the count. An arm whose dispatch condition is false on
    // this call cannot win anything and is not counted. Two eligible opt-in
    // arms still conflict (the earliest wins, the later one is silently
    // unreachable), and that is what this refuses.
    //
    // What this guard CANNOT police, by construction: an opt-in arm shadowed
    // by a LIVE DEFAULT (e.g. INTFER_DECODE1_MR=1 with the defaults unset
    // -- VERBHALF is checked earlier and wins). That state dispatches
    // VERBHALF and returns before this guard, so it is invisible here; the
    // gate scripts' announcement assertions are the mechanism that catches
    // it, which is why every opt-in gate arm must kill BOTH defaults in its
    // own env.
    //
    // *** ALL NINE OPT-IN FLAGS ARE READ HERE, ONCE, SO THE GUARD AND THE
    // ARMS CANNOT DRIFT. *** A new arm adds one line to this block and
    // nothing else; there is no second place where the flag can be declared,
    // so a guard that forgets to count an arm is a review-time visible miss
    // rather than a silent hole.
    const int num_groups = (static_cast<int>(K) + Gw - 1) / Gw;
    const bool is_int8 = (bits >= 8);
    const int pack = is_int8 ? 1 : 8 / bits;
    // Kill switch INTFER_NO_INT2_TAB=1 restores the generic path
    // (bisection). Host-side read; getenv/static are illegal in device.
    const bool use_int2_tab =
        (std::getenv("INTFER_NO_INT2_TAB") == nullptr);
    const bool use_rowpair =
        (std::getenv("INTFER_DECODE1_ROWPAIR") != nullptr) &&
        (std::getenv("INTFER_NO_DECODE1_ROWPAIR") == nullptr);
    const bool use_mr = (std::getenv("INTFER_DECODE1_MR") != nullptr) &&
                        (std::getenv("INTFER_NO_DECODE1_MR") == nullptr);
    const bool use_verbdead =
        (std::getenv("INTFER_DECODE1_VERBDEAD") != nullptr) &&
        (std::getenv("INTFER_NO_DECODE1_VERBDEAD") == nullptr);
    const bool use_verbsel =
        (std::getenv("INTFER_DECODE1_VERBSEL") != nullptr) &&
        (std::getenv("INTFER_NO_DECODE1_VERBSEL") == nullptr);
    const bool use_verbvec =
        (std::getenv("INTFER_DECODE1_VERBVEC") != nullptr) &&
        (std::getenv("INTFER_NO_DECODE1_VERBVEC") == nullptr);
    const bool use_wvec =
        (std::getenv("INTFER_DECODE1_WVEC") != nullptr) &&
        (std::getenv("INTFER_NO_DECODE1_WVEC") == nullptr);
    const char* sk_env = std::getenv("INTFER_DECODE1_SPLITK");
    const bool use_splitk =
        (sk_env && sk_env[0] == '1' && sk_env[1] == '\0') &&
        (std::getenv("INTFER_NO_DECODE1_SPLITK") == nullptr);
    {
      const bool tab_m1 = (bits == 2 && use_int2_tab && num_groups > 0);
      const bool rp_ok  = (bits == 2 && (N & 1) == 0);
      // *** THE ARRAY WAS SZED 2 AGAINST A POPULATION OF SEVEN. ***
      // note_o is called by SEVEN guards below -- rowpair, verbdead,
      // verbsel, verbvec, wvec, mr, splitk -- so a state with three or
      // more live opt-in arms wrote past on[1] (an out-of-bounds stack
      // store) AND the refusal named only the first two, hiding the
      // third from the operator the message exists to inform. The size is
      // therefore derived from the guards that feed it, note_o clamps
      // the index, and the message enumerates every live arm. The default
      // production state sets ZERO opt-in flags, so the refusal is
      // unreachable on the hot path and this changes no arithmetic.
      const char* on[8] = {nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr, nullptr, nullptr};
      int no = 0;
      auto note_o = [&](const char* n) {
        if (no < 8) on[no] = n;
        ++no;
      };
      if (use_rowpair  && rp_ok)  note_o("INTFER_DECODE1_ROWPAIR");
      if (use_verbdead && tab_m1) note_o("INTFER_DECODE1_VERBDEAD");
      if (use_verbsel  && tab_m1) note_o("INTFER_DECODE1_VERBSEL");
      if (use_verbvec  && tab_m1) note_o("INTFER_DECODE1_VERBVEC");
      if (use_wvec     && tab_m1) note_o("INTFER_DECODE1_WVEC");
      if (use_mr       && tab_m1) note_o("INTFER_DECODE1_MR");
      if (use_splitk   && tab_m1) note_o("INTFER_DECODE1_SPLITK");
      if (no > 1) {
        // Enumeration, not "%s and %s": with the OOB store fixed above,
        // more than two live arms is now a reachable state, and a
        // two-name message would under-report it -- the guard's whole
        // job is that the operator sees every flag that is live.
        std::fprintf(stderr,
                     "[decode1-bisect] REFUSING: %d opt-in arms are live and\n"
                     "  all dispatch on this shape. The M=1 decode1 arms each\n"
                     "  end in `return q.submit(...)`, so the EARLIEST one wins\n"
                     "  and the later ones are silently unreachable:\n",
                     no);
        for (int i = 0; i < no && i < 8; ++i)
          std::fprintf(stderr, "    %s\n", on[i]);
        std::fprintf(stderr, "  Exactly one opt-in arm at a time.\n");
        return sycl::event();
      }
    }

    // --------------------------------------------------------------------
    // INTFER_DECODE1_ROWPAIR -- default OFF, kill switch
    // INTFER_NO_DECODE1_ROWPAIR=1. BIT-EXACT BY CONSTRUCTION.
    //
    // WHAT IT IS. One work-item computes TWO consecutive output rows instead of
    // one: it loads the activation vector ONCE and runs TWO INDEPENDENT FMA chains
    // against it, so the shared activation load is amortised over twice the weight
    // traffic and -- the point -- there are twice as many weight loads in flight
    // per work-item. Copied from llama.cpp ggml-sycl/esimd.hpp's `mac_pair`, whose
    // own comment gives the reason: "the two FMA chains interleaved
    // (co-scheduled to hide FMA latency)".
    //
    // *** THIS IS NOT INTFER_DECODE1_MR, AND THE DISTINCTION IS THE WHOLE POINT.
    // MR gave each work-item MORE OUTPUTS and measured 2.3x SLOWER, because at M=1
    // the work-item count IS N and dividing a fixed budget by four is
    // anti-parallel. Row-pairing keeps the work-item count where it is
    // (ceil(N/2) work-items instead of N -- HALF, not a quarter) and instead
    // doubles the BYTES IN FLIGHT per work-item, which is the lever
    // examples/roofline_probe says is live: decode sits at 15.6-19.9 % of the
    // ceiling this card demonstrates with the same allocator and the same access
    // pattern, and sustaining 456 GB/s at ~600 ns latency needs ~274 KB in flight.
    //
    // WHY IT IS BIT-EXACT. For output row n, the sequence of additions is
    // unchanged: its own accumulator, ascending kg, ascending j, the same 8-code
    // chunk, the same byte indexing, the same wtab lookup. The second row's chain
    // is interleaved in the INSTRUCTION stream, not folded into the first row's
    // accumulator, so no floating-point value is ever combined with another row's.
    // So the claim is stream identity against the fixed oracle, and it needs NO
    // numerics contract -- which is exactly what distinguishes it from
    // INTFER_DECODE1_SPLITK and INTFER_DECODE1_SUBGROUP, both of which reassociate
    // and both of which need one.
    //
    // NUMERICS IS NOT AN ARGUMENT HERE, IT IS A CHECK: scripts/check_decode1_copies.sh
    // pins the two inline copies against each other (mutual agreement, NOT against
    // remembered literals -- the first version of that checker failed its own "both
    // copies changed together" state for exactly that reason), and
    // scripts/test_tp2_27b.sh asserts the flag's arm against the same fixed oracle
    // the base run uses, not a self-comparison.
    //
    // REACHABILITY: the branch is guarded on `bits == 2` exactly like the MR and
    // SPLITK arms, and the 0.8B gate model has ZERO INT2 tensors -- so this arm
    // MUST live in test_tp2_27b.sh, not test_gpu_optin.sh, or it would be green
    // because the path never ran.
    // (use_rowpair is declared at the top of the M=1 block, with the other
    // eight opt-in flags, so the exclusivity guard can see it.)
    // *** INTFER_DECODE1_BLOCKLOAD WAS WRITTEN AND THEN DELETED, and the reason
    // is the point. *** An earlier draft of this batch shipped a second flag doing
    // "one sub-group per output row, K split across lanes, register-only
    // reduction, no local_accessor and no barrier" -- and discovered on the way to
    // the gate that INTFER_DECODE1_SUBGROUP ALREADY DOES EXACTLY THAT, and does it
    // better on both counts:
    //
    //   * SUBGROUP reduces with `sycl::reduce_over_group(sg, acc, plus<float>())`.
    //     This draft hand-rolled a 4-step `permute_group_by_xor` butterfly. The
    //     collective is the idiomatic form, it is what llama.cpp's mmvq.cpp does,
    //     and it is convergent without the author having to prove it.
    //   * SUBGROUP's split is BLOCK-CYCLIC (lane l owns chunks l, l+S, l+2S, ...),
    //     which is the one that coalesces. This draft's was a CONTIGUOUS K/16
    //     slice per lane -- and AGENTS.md already records, from the SPLITK work,
    //     that contiguous chunks stay strided.
    //
    // Shipping both would be the "two copies and flag polarity" defect this repo
    // has a memory tag for: two flags naming one design, differing in the
    // reduction primitive, neither certifiable on its own because a reader cannot
    // tell which one a given measurement used. So the duplicate is deleted and the
    // REAL flag gets the gate arm instead -- which is where the value was: that arm
    // did not exist, and INTFER_DECODE1_SUBGROUP is default OFF and reassociating.
    //
    // The one genuinely NEW thing the deleted draft had was the esimd
    // `block_load<uint8_t,128>` (8 B per lane at the measured sg=16) in place of
    // scalar 2-byte `code_u8[]` reads. That is NOT lost, it is DOWNGRADED to what
    // it should have been from the start: a single-change arm on top of SUBGROUP,
    // measured by examples/roofline_probe, not a flag. And its prior is already
    // negative -- the probe measured a 16-code chunk at 0.25-0.35x, a 3-4x
    // REGRESSION, because register pressure dominates. If someone still wants the
    // number, the arm is: SUBGROUP's body with the two `code_u8[b0]`/`[b0+1]`
    // reads replaced by one `block_load<uint8_t,128>` per 32 codes. Kill criterion
    // is the probe's own convention: below 2x it closes llama.cpp's block_load
    // direction for intfer.

    if (use_rowpair && bits == 2 && (N & 1) == 0) {
      // Announce ONCE, on the host, not per launch: a 794-launches-per-token engine
      // printing per launch would flood the log the gate parses. The gate's arm
      // REQUIRES this line, because a stream comparison is satisfiable by a flag
      // that did nothing -- audit #3's nested-flag vacuity, and the one thing that
      // turns "the two runs matched" into "the row-pair path ran".
      decode1_announce("decode1-rowpair");
      const int npair = N / 2;
      const int pBlocks = (npair + nWG - 1) / nWG;
      const int pGlobal = pBlocks * nWG;
      return q.submit([&](sycl::handler& h) {
        h.depends_on(copy_dep);
        h.parallel_for<class gemm_usm_decode1_rowpair>(
            sycl::nd_range<1>(sycl::range<1>(pGlobal), sycl::range<1>(nWG)),
            [=](sycl::nd_item<1> it) {
              const int np = static_cast<int>(it.get_global_id(0));
              if (np >= npair) return;
              const int na = np * 2;      // row A = the even output
              const int nb = np * 2 + 1;  // row B = the odd output

              // INLINE AT EACH SITE, NEVER A SHARED DEVICE FUNCTION. The reverted
              // fused-decode kernel cost 231 114 ms (a ~180x regression) with
              // arithmetic that a normalized diff proved identical, and the
              // generalised cause is that the SYCL device compiler failed to
              // optimise the extracted body. So the two chains are written out
              // here, twice, and the copies are pinned by a checker rather than
              // shared.
              float accA = 0.0f, accB = 0.0f;
              const std::uint16_t* sA = scale_usm + na * num_groups_w;
              const std::uint16_t* sB = scale_usm + nb * num_groups_w;
              const std::size_t rbA = static_cast<std::size_t>(na) * K;
              const std::size_t rbB = static_cast<std::size_t>(nb) * K;
              sycl::multi_ptr<const std::uint16_t,
                  sycl::access::address_space::global_space> act_mp(act_usm);
              // *** NO wA/wB multi_ptrs, ON PURPOSE, AND THE COMMENT IS THE
              // POST-MORTEM. *** The first version of this arm built two
              // row-OFFSET multi_ptrs (`code_u8 + rbA`) and then indexed them with
              // ABSOLUTE byte offsets, `(rbA + kg) / 4` -- which is the base
              // kernel's own expression. Double offset: it read at
              // rbA + (rbA+kg)/4, far past the shard, and the result was
              // UR_RESULT_ERROR_DEVICE_LOST on both INT2 shapes. The FP16 and INT4
              // shapes passed, because the branch is guarded on bits == 2 and never
              // ran there -- so a smoke test on a shape the flag does not reach
              // would have called this green.
              //
              // The "bit-exact by construction" argument looked at the arithmetic
              // and said the indices matched the base kernel. They did not, and no
              // reading would have found it. Two rows sharing one activation load is
              // exactly where a base pointer you forgot to update lives. The fix is
              // to use the SAME unoffset `code_u8` with the SAME absolute
              // expressions the base kernel uses, so the only difference from the
              // reference is the two accumulators -- which is the claim being
              // certified.
              for (int g = 0; g < num_groups_w; ++g) {
                const float scA = from_half_bits(sA[g]) * alpha;
                const float scB = from_half_bits(sB[g]) * alpha;
                const int kg0 = g * Gw;
                int kg1 = kg0 + Gw;
                if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                const float wA4[4] = {scA * -1.0f, scA * (-1.0f / 3.0f),
                                      scA * (1.0f / 3.0f), scA * 1.0f};
                const float wB4[4] = {scB * -1.0f, scB * (-1.0f / 3.0f),
                                      scB * (1.0f / 3.0f), scB * 1.0f};
                constexpr int CC = 8;   // 16 codes/chunk measured 0.25-0.35x
                int kg = kg0;
                std::size_t b0A = (rbA + static_cast<std::size_t>(kg)) / 4u;
                std::size_t b0B = (rbB + static_cast<std::size_t>(kg)) / 4u;
                for (; kg + CC <= kg1; kg += CC) {
                  // THE SHARED LOAD. One activation vector, two weight pairs, two
                  // independent accumulators. This is the entire mechanism, and it
                  // is the ONLY structural difference from the base kernel.
                  sycl::vec<std::uint16_t, 8> a16;
                  a16.load(0, act_mp + kg);
                  const std::uint8_t a0 = code_u8[b0A], a1 = code_u8[b0A + 1];
                  const std::uint8_t c0 = code_u8[b0B], c1 = code_u8[b0B + 1];
                  b0A += 2; b0B += 2;
                  for (int j = 0; j < CC; ++j) {
                    const float av =
                        static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                    const std::uint8_t va = (j < 4) ? a0 : a1;
                    const std::uint8_t vb = (j < 4) ? c0 : c1;
                    // Two SEPARATE accumulators: the chains are co-scheduled in the
                    // instruction stream, never summed together. That is what makes
                    // each output's addition sequence the base kernel's sequence.
                    accA += av * wA4[(va >> ((j & 3) * 2)) & 3];
                    accB += av * wB4[(vb >> ((j & 3) * 2)) & 3];
                  }
                }
                // Ragged group tail, per row, with the base kernel's expressions.
                for (; kg < kg1; ++kg) {
                  const float av = from_half_bits(act_usm[kg]);
                  const std::size_t iA = rbA + kg, iB = rbB + kg;
                  accA += av * scA *
                          canon_value(2, (code_u8[iA / 4] >> ((iA % 4) * 2)) & 3);
                  accB += av * scB *
                          canon_value(2, (code_u8[iB / 4] >> ((iB % 4) * 2)) & 3);
                }
              }
              out_usm[na] = to_half_bits(accA);
              out_usm[nb] = to_half_bits(accB);
            });
      });
    }

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
      // *** num_groups / is_int8 / pack / use_int2_tab and ALL NINE
      // INTFER_DECODE1_* flags are declared at the TOP of the M=1 block
      // (above), so the exclusivity guard can see every arm. Nothing is
      // re-declared here; a duplicate declaration here would shadow the
      // hoisted one and re-open the hole the guard exists to close.
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
      // SUPERSEDED 2026-10-05: the occupancy model above is WRONG, because
      // "512 Xe cores" was the A770's count. The B580 has 20 Xe cores and ~2560
      // FP32 lanes, so a layer GEMM at N~2048-8704 is at or above 100% lane
      // occupancy and the kernel is NOT lane-starved. The MR MEASUREMENT (2.3x
      // slower) stands and MR stays closed; the EXPLANATION does not. See
      // AGENTS.md's SUPERSEDED banner and
      // reports/2026-10-05-batch-gemv-roofline.md.
      //
      // That reframes the whole M=1 decode target: the only ways to raise
      // parallelism are (a) split K across work-items, which reassociates the
      // FP32 sum and is therefore NOT bit-exact, or (b) fuse INDEPENDENT GEMMs
      // into one nd_range so one launch carries several N's worth of
      // work-items, which IS bit-exact because each output keeps its own
      // work-item and its own K order. (b) is the live direction.
      // (use_mr is declared at the top of the M=1 block, with the other
      // eight opt-in flags, so the exclusivity guard can see it.)
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
      // SUPERSEDED 2026-10-05: the "~21 % of ~179 GB/s" is wrong twice. Peak is
      // 456 GB/s, so the real figure is ~9 %. The B580 is not bandwidth-bound at
      // 9% -- something else is. Do not propagate the percentage above.
      // CITATION(techpowerup arc-b580.c4244 / ARK 241598: 456.0 GB/s, 20 Xe
      // cores. CITATION(scripts/model_byte_roofline.sh: 9.752 GB/device/token,
      // 41.1 GB/s at 237.5 ms = 9.0%.)
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
      // --------------------------------------------------------------------
      // INTFER_DECODE1_VERBHALF -- default ON 2026-10-07, kill switch
// INTFER_NO_DECODE1_VERBHALF=1 restores the old branchy scale twiddle.
      //
      // WHAT IT IS. EXACTLY ONE change to this kernel: the per-group scale is read
      // with `sycl::bit_cast<sycl::half>` (one hardware convert) instead of
      // `from_half_bits`, the engine's branchy SCALAR FP16->FP32 twiddle
      // (3 branches per call). The twiddle is called once per GROUP, and
      // num_groups = K/Gw = 5120/16 = 320 at MLP_GATE -- so 320 calls per
      // work-item, in the loop that also carries the dequant and the FMA chain.
      //
      // WHY IT IS HERE. examples/roofline_probe measured this kernel at 13.8-41.3
      // GB/s against 78-261 GB/s for a faithful clone of the SAME algorithm --
      // a 5.6-6.4x gap reproduced interleaved. Two causes were then measured and
      // FALSIFIED: module size (the shipped source in a small TU bought 1.26x
      // against a >=2x kill criterion) and the runtime branch (making use_int2_tab
      // a template parameter bought 1.007x). This arm is the first remaining
      // single-change step, and it was blocked only because one probe TU crosses
      // the level-zero OUT_OF_RESOURCES limit -- so it is landed HERE, where it can
      // be measured on real weights.
      //
      // *** NOT MEASURED. *** examples/roofline_probe_bisect exists to measure it
      // and has NEVER COMPLETED A RUN (every arm hangs at HEAD, including the
      // `table` control; scripts/test_roofline_bisect.sh reports rc=2 UNPROVEN).
      // So the direction is UNKNOWN, not positive.
      //
      // NUMERICS. NOT bit-exact by construction, and the two forms are not
      // interchangeable: `from_half_bits` returns 0 for a subnormal or zero
      // exponent and Inf/NaN for the saturated cases, while `bit_cast<half>` hands
      // the denormal to the FP32 convert, which flushes it to zero under the
      // default FTZ. A zero-group scale is the case that differs, and FP16 scale
      // values come from `to_half_bits` of a positive host float, so it should not
      // arise -- but "should not" is an argument, not a measurement, and this arm
      // therefore needs a stream-identity gate and NOT an arithmetic claim. It is
      // in the same class as INTFER_DECODE1_SPLITK, whose contract
      // (`examples/ksplit_contract`) measures 0 FP16 ulp on realistic activations
      // and 2 on an adversarial escalating set.
      //
      // INLINE AT THE CALL SITE. The reverted `quant_gemm_fused_decode1` cost
      // 231114 ms with arithmetic a normalized diff proved identical, and the
      // generalised cause is that the SYCL device compiler failed to optimise an
      // extracted body. So this is a SEPARATE parallel_for, not a helper.
      // PROMOTED TO DEFAULT-ON 2026-10-07 (was default OFF). Kill switch
      // INTFER_NO_DECODE1_VERBHALF=1 restores the old branchy form. The
      // clone's +14% MLP_GATE / +17% MLP_DOWN is RETRACTED (5bfd8dd:
      // every bisect arm read 0.000 ms / inf GB/s, so the ratios were
      // 0/0) -- the throughput direction is UNKNOWN, and the engine arm
      // in scripts/test_tp2_27b.sh is the only way to measure it on real
      // 27B weights. What is claimed: stream identity, NOT bit-exactness
      // (the FTZ denormal caveat below).
      //
      // *** THIS IS THE DEFAULT DECODE PATH NOW. *** The guard is inverted:
      // the feature is ON unless the kill switch is set, so the 3-branch scalar
      // `from_half_bits` twiddle no longer runs on the hot path. It was called
      // once per GROUP = K/Gw = 320 times per work-item at MLP_GATE, and the
      // branchy form is what the clone measured.
      //
      // NUMERICS, AND THE CAVEAT STAYS. NOT bit-exact by construction, and the two
      // forms are not interchangeable: `from_half_bits` returns 0 for a subnormal
      // or zero exponent and Inf/NaN when saturated, while `bit_cast<half>` hands
      // the denormal to the FP32 convert, which flushes to zero under the default
      // FTZ. A zero-group scale is the case that differs, and FP16 scale values
      // come from `to_half_bits` of a positive host float, so it should not arise
      // -- but "should not" is an argument, not a measurement. This arm is therefore
      // in the SAME CLASS AS INTFER_DECODE1_SPLITK: default-ON with a kill switch,
      // a gate arm that asserts STREAM IDENTITY (not arithmetic equality), and no
      // bit-exactness claim. scripts/test_rowpair_equiv.sh measures it model-free:
      // on the synthetic datasets it is bitwise identical to base, which does NOT
      // prove the FTZ concern away -- only that these scales never reach it.
      const bool use_verbhalf =
          (std::getenv("INTFER_NO_DECODE1_VERBHALF") == nullptr);
      if (use_verbhalf && bits == 2 && use_int2_tab && num_groups > 0) {
        // *** THE ANNOUNCEMENT WAS MISSING HERE AT FIRST. *** Round 1 added
        // announcement lines to the round-2 arms and not to these two, so the
        // differential's announcement requirement could never be satisfied for
        // them and the arms were indistinguishable from dead ones. A gate that
        // requires proof-of-execution cannot be satisfied by an arm that never
        // emits proof, and the failure mode is exactly the vacuous PASS.
        decode1_announce("decode1-verbhalf");
        return q.submit([&](sycl::handler& h) {
          h.depends_on(copy_dep);
          h.parallel_for<class gemm_usm_decode1_verbhalf>(
              sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
              [=](sycl::nd_item<1> it) {
                const int ng = static_cast<int>(it.get_global_id(0));
                if (ng >= N) return;
                float acc = 0.0f;
                const std::uint16_t* sptr = scale_usm + ng * num_groups_w;
                const std::size_t row_base = static_cast<std::size_t>(ng) * K;
                sycl::multi_ptr<const std::uint16_t,
                    sycl::access::address_space::global_space> act_mp(act_usm);
                for (int g = 0; g < num_groups; ++g) {
                  // *** THE ONE CHANGE. Everything else below is the base kernel. ***
                  const float scale =
                      static_cast<float>(sycl::bit_cast<sycl::half>(sptr[g])) * alpha;
                  const int kg0 = g * Gw;
                  int kg1 = kg0 + Gw;
                  if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                  const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                                         scale * (1.0f / 3.0f), scale * 1.0f};
                  constexpr int CC = 8;   // 16 codes/chunk measured 0.25-0.35x
                  int kg = kg0;
                  std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
                  for (; kg + CC <= kg1; kg += CC) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    const std::uint8_t b0v = code_u8[b0];
                    const std::uint8_t b1v = code_u8[b0 + 1];
                    b0 += 2;
                    for (int j = 0; j < CC; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const std::uint8_t bv = (j < 4) ? b0v : b1v;
                      // (j & 3), NOT j: INT2 packs FOUR codes per byte, so a shift
                      // of >= 8 on a uint8_t promotes to 0. That bug shipped once.
                      const int code = (bv >> ((j & 3) * 2)) & 3;
                      acc += av * wtab[code];
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    const std::size_t idx = row_base + static_cast<std::size_t>(kg);
                    const int code =
                        static_cast<int>((code_u8[idx / 4] >> ((idx % 4) * 2)) & 3);
                    acc += av * scale * canon_value(2, code);
                  }
                }
                out_usm[ng] = to_half_bits(acc);
              });
        });
      }

      // --------------------------------------------------------------------
      // INTFER_DECODE1_VERBMP -- default ON 2026-10-07, kill switch
      // INTFER_NO_DECODE1_VERBMP=1 restores the absolute-index form.
      //
      // WHAT IT IS. EXACTLY ONE change to this kernel: the weight side is read
      // through a `multi_ptr` built on the row base, instead of indexing the raw
      // `code_u8[b0]` with an absolute byte offset. A multi_ptr is what lets the
      // compiler keep the base in a register and fold the addressing rather than
      // re-materialising `code_u8 + b0` every iteration.
      //
      // *** AND THIS IS THE SHAPE OF AN ALREADY-FOUND BUG, WHICH IS WHY IT IS
      // WORTH A SEPARATE ARM AND WHY IT IS INLINE. *** INTFER_DECODE1_ROWPAIR
      // shipped row-OFFSET multi_ptrs indexed with the base kernel's ABSOLUTE byte
      // offsets -- a double offset that read past the shard and LOST THE DEVICE
      // (UR_RESULT_ERROR_DEVICE_LOST on both INT2 shapes). A CPU reproduction of the
      // indexing arithmetic, and a model-free bitwise differential
      // (scripts/test_rowpair_equiv.sh), are what caught it; a reading did not.
      // The indices below are therefore `b0` RELATIVE to the row base, and the tail
      // deliberately uses the UNOFFSET `code_u8` with absolute indices exactly as
      // the base kernel does -- so the only difference is which pointer type the
      // chunk loop uses.
      //
      // UNMEASURED, same as VERBHALF, and for the same reason. NUMERICS: this one
      // IS bit-exact by construction -- the arithmetic expressions and the index
      // VALUES are unchanged, only the pointer type the compiler folds them
      // through. So it needs a stream-identity gate and no numerics contract.
      // PROMOTED TO DEFAULT-ON 2026-10-07 (was default OFF). Kill switch
      // INTFER_NO_DECODE1_VERBMP=1 restores the absolute-index form. The
      // clone's 1.01x measurement is RETRACTED (5bfd8dd: every bisect arm
      // read 0.000 ms / inf GB/s), so the throughput direction is UNKNOWN;
      // what is MEASURED is the numerics -- bit-exact by construction,
      // measured model-free bitwise against the restored base by
      // scripts/test_rowpair_equiv.sh (8 shapes x 5 datasets).
      //
      // *** DEFAULT-ON, BUT NOT ON THE DEFAULT PATH -- SHADOWED BY
      // VERBHALF. *** The guard is inverted (the feature is ON unless the
      // kill switch is set), but the VERBHALF arm above is checked FIRST
      // and ends in `return`, so with both kill switches unset the
      // VERBHALF body is the hot path and this arm is unreachable. It
      // runs only with INTFER_NO_DECODE1_VERBHALF=1, where it is the
      // bit-exact alternative to the default. The composition of both
      // changes (the bit_cast scale AND this multi_ptr) is unreachable
      // by any env combination; whether it belongs on the default path
      // is an open 27B measurement, not a code decision.
      //
      // NUMERICS: this one IS bit-exact by construction -- the arithmetic
      // expressions and the index VALUES are unchanged, only the pointer TYPE the
      // compiler folds them through. scripts/test_rowpair_equiv.sh measures it
      // model-free: 227 835 element values bitwise identical to base over 8 shapes
      // x 5 datasets, so the claim is MEASURED, not argued.
      const bool use_verbmp =
          (std::getenv("INTFER_NO_DECODE1_VERBMP") == nullptr);
      if (use_verbmp && bits == 2 && use_int2_tab && num_groups > 0) {
        // *** THE ANNOUNCEMENT WAS MISSING HERE AT FIRST. *** Round 1 added
        // announcement lines to the round-2 arms and not to these two, so the
        // differential's announcement requirement could never be satisfied for
        // them and the arms were indistinguishable from dead ones. A gate that
        // requires proof-of-execution cannot be satisfied by an arm that never
        // emits proof, and the failure mode is exactly the vacuous PASS.
        decode1_announce("decode1-verbmp");
        return q.submit([&](sycl::handler& h) {
          h.depends_on(copy_dep);
          h.parallel_for<class gemm_usm_decode1_verbmp>(
              sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
              [=](sycl::nd_item<1> it) {
                const int ng = static_cast<int>(it.get_global_id(0));
                if (ng >= N) return;
                float acc = 0.0f;
                const std::uint16_t* sptr = scale_usm + ng * num_groups_w;
                const std::size_t row_base = static_cast<std::size_t>(ng) * K;
                sycl::multi_ptr<const std::uint16_t,
                    sycl::access::address_space::global_space> act_mp(act_usm);
                // *** THE ONE CHANGE: the weight side through a row-base multi_ptr. ***
                // row_base is an ELEMENT offset (ng * K); the weight side is packed
                // four INT2 codes per byte, so the multi_ptr base must be converted
                // to bytes. This is the same double-offset class of defect that made
                // the ORIGINAL INTFER_DECODE1_ROWPAIR lose the device (row-offset
                // multi_ptr indexed with the base kernel's ABSOLUTE byte offsets);
                // if the /4 is wrong, the arm reads past the shard and DEVICE_LOSTs.
                sycl::multi_ptr<const std::uint8_t,
                    sycl::access::address_space::global_space> wmp(code_u8 + row_base / 4u);
                for (int g = 0; g < num_groups; ++g) {
                  const float scale = from_half_bits(sptr[g]) * alpha;
                  const int kg0 = g * Gw;
                  int kg1 = kg0 + Gw;
                  if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                  const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                                         scale * (1.0f / 3.0f), scale * 1.0f};
                  constexpr int CC = 8;
                  int kg = kg0;
                  // RELATIVE to the row base, because `wmp` already carries it.
                  std::size_t b0 = static_cast<std::size_t>(kg) / 4u;
                  for (; kg + CC <= kg1; kg += CC) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    const std::uint8_t b0v = wmp[b0];
                    const std::uint8_t b1v = wmp[b0 + 1];
                    b0 += 2;
                    for (int j = 0; j < CC; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const std::uint8_t bv = (j < 4) ? b0v : b1v;
                      const int code = (bv >> ((j & 3) * 2)) & 3;
                      acc += av * wtab[code];
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    // UNOFFSET base with ABSOLUTE indices, exactly as the base
                    // kernel's tail -- NOT `wmp[idx / 4]`, which is the ROWPAIR
                    // double-offset that lost the device.
                    const std::size_t idx = row_base + static_cast<std::size_t>(kg);
                    const int code =
                        static_cast<int>((code_u8[idx / 4] >> ((idx % 4) * 2)) & 3);
                    acc += av * scale * canon_value(2, code);
                  }
                }
                out_usm[ng] = to_half_bits(acc);
              });
        });
      }

      // ====================================================================
      // BISECT ROUND 2. Three more single-change arms off the SHIPPED body, all
      // BIT-EXACT BY CONSTRUCTION so `examples/rowpair_equiv` can give a hard
      // verdict on them TODAY without a model and without a 27B load. Round 1's
      // VERBHALF is the exception and is stream-identity only.
      //
      // WHY THESE THREE AND NOT THE OBVIOUS ONES. Two candidates that sound
      // equivalent are not, and writing them would have been bugs:
      //   * "hoist the group scale out of the per-group loop". IMPOSSIBLE as stated:
      //     the scale is PER GROUP -- `from_half_bits(sptr[g])` with
      //     num_groups = K/Gw = 320 at MLP_GATE. Hoisting it out would reuse one
      //     group's scale for all 320 and change the arithmetic. What CAN be hoisted
      //     is the code AROUND it, which is VERBDEAD below.
      //   * "avoid the per-group multi_ptr re-derivation". The base kernel already
      //     constructs `act_mp` and `wmp` ONCE, outside the group loop; there is
      //     nothing per-group to avoid.
      // ====================================================================

      // ---- INTFER_DECODE1_VERBDEAD: dead per-group locals removed -----------
      // ONE change, and it is pure CODE MOTION: `bytes_per_chunk` and `code_mask`
      // are computed INSIDE the per-group loop and are dead on the `use_int2_tab`
      // path (they belong to the generic `else` branch, which this guard does not
      // take). The compiler must prove that per iteration. Hoisting them to the
      // function scope -- or deleting them, since this arm never takes the branch
      // that uses them -- is arithmetically a no-op, so this arm is BIT-EXACT BY
      // CONSTRUCTION and a difference means the compiler did something we did not
      // ask for. It is the cheapest arm in the batch and the first thing to run,
      // because if it pays the others are not needed.
      // (use_verbdead is declared at the top of the M=1 block, with the other
      // eight opt-in flags, so the exclusivity guard can see it.)
      if (use_verbdead && bits == 2 && use_int2_tab && num_groups > 0) {
        decode1_announce("decode1-verbdead");
        // Hoisted OUT of the group loop and, on this path, UNUSED -- which is the
        // point. `bytes_per_chunk`/`code_mask` feed only the generic packed branch
        // below, which this guard excludes.
        const int bytes_per_chunk = 8 / 4;
        const int code_mask = 3;
        (void)bytes_per_chunk; (void)code_mask;
        return q.submit([&](sycl::handler& h) {
          h.depends_on(copy_dep);
          h.parallel_for<class gemm_usm_decode1_verbdead>(
              sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
              [=](sycl::nd_item<1> it) {
                const int ng = static_cast<int>(it.get_global_id(0));
                if (ng >= N) return;
                float acc = 0.0f;
                const std::uint16_t* sptr = scale_usm + ng * num_groups_w;
                const std::size_t row_base = static_cast<std::size_t>(ng) * K;
                sycl::multi_ptr<const std::uint16_t,
                    sycl::access::address_space::global_space> act_mp(act_usm);
                sycl::multi_ptr<const std::uint8_t,
                    sycl::access::address_space::global_space> wmp(code_u8 + row_base);
                for (int g = 0; g < num_groups; ++g) {
                  const float scale = from_half_bits(sptr[g]) * alpha;
                  const int kg0 = g * Gw;
                  int kg1 = kg0 + Gw;
                  if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                  const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                                         scale * (1.0f / 3.0f), scale * 1.0f};
                  constexpr int CC = 8;
                  int kg = kg0;
                  std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
                  for (; kg + CC <= kg1; kg += CC) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    const std::uint8_t b0v = code_u8[b0];
                    const std::uint8_t b1v = code_u8[b0 + 1];
                    b0 += 2;
                    for (int j = 0; j < CC; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const std::uint8_t bv = (j < 4) ? b0v : b1v;
                      const int code = (bv >> ((j & 3) * 2)) & 3;
                      acc += av * wtab[code];
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    const std::size_t idx = row_base + static_cast<std::size_t>(kg);
                    const int code =
                        static_cast<int>((code_u8[idx / 4] >> ((idx % 4) * 2)) & 3);
                    acc += av * scale * canon_value(2, code);
                  }
                }
                out_usm[ng] = to_half_bits(acc);
              });
        });
      }

      // ---- INTFER_DECODE1_VERBSEL: the 4-float codebook by SELECT ------------
      // ONE change: `wtab[code]` (a 4-float array indexed by a runtime value)
      // becomes a select chain over the SAME four values held in named registers.
      //
      // *** AND THIS IS THE HYPOTHESIS THAT WAS FALSIFIED IN THE WRONG BODY. ***
      // examples/roofline_probe compared `table` against `select` and got 1.01x --
      // but BOTH of those were in the probe's own TIDY CLONE, where the compiler
      // had already normalised the array. This tests the SAME transformation in
      // the SHIPPED body, which is 5.6-6.4x slower than that clone and may not
      // normalise the same way. So the earlier result does not falsify this; it
      // falsifies the claim that the array is free *in a well-formed copy*.
      //
      // BIT-EXACT BY CONSTRUCTION: the four values are the same expressions, so the
      // selected value is bit-for-bit what the array held. (The arithmetic form
      // `(2*code-3)/3` is ALSO bit-identical and was verified exhaustively on the
      // CPU over all 4 codes x 64 scales -- but it is a different change and is not
      // conflated with this one.)
      // (use_verbsel is declared at the top of the M=1 block, with the other
      // eight opt-in flags, so the exclusivity guard can see it.)
      if (use_verbsel && bits == 2 && use_int2_tab && num_groups > 0) {
        decode1_announce("decode1-verbsel");
        return q.submit([&](sycl::handler& h) {
          h.depends_on(copy_dep);
          h.parallel_for<class gemm_usm_decode1_verbsel>(
              sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
              [=](sycl::nd_item<1> it) {
                const int ng = static_cast<int>(it.get_global_id(0));
                if (ng >= N) return;
                float acc = 0.0f;
                const std::uint16_t* sptr = scale_usm + ng * num_groups_w;
                const std::size_t row_base = static_cast<std::size_t>(ng) * K;
                sycl::multi_ptr<const std::uint16_t,
                    sycl::access::address_space::global_space> act_mp(act_usm);
                sycl::multi_ptr<const std::uint8_t,
                    sycl::access::address_space::global_space> wmp(code_u8 + row_base);
                for (int g = 0; g < num_groups; ++g) {
                  const float scale = from_half_bits(sptr[g]) * alpha;
                  const int kg0 = g * Gw;
                  int kg1 = kg0 + Gw;
                  if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                  // THE CHANGE: four named registers instead of an indexed array.
                  const float w_n1 = scale * -1.0f;
                  const float w_n3 = scale * (-1.0f / 3.0f);
                  const float w_p3 = scale * (1.0f / 3.0f);
                  const float w_p1 = scale * 1.0f;
                  constexpr int CC = 8;
                  int kg = kg0;
                  std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
                  for (; kg + CC <= kg1; kg += CC) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    const std::uint8_t b0v = code_u8[b0];
                    const std::uint8_t b1v = code_u8[b0 + 1];
                    b0 += 2;
                    for (int j = 0; j < CC; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const std::uint8_t bv = (j < 4) ? b0v : b1v;
                      const int code = (bv >> ((j & 3) * 2)) & 3;
                      const float w = (code == 0) ? w_n1
                                     : (code == 1) ? w_n3
                                     : (code == 2) ? w_p3 : w_p1;
                      acc += av * w;
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    const std::size_t idx = row_base + static_cast<std::size_t>(kg);
                    const int code =
                        static_cast<int>((code_u8[idx / 4] >> ((idx % 4) * 2)) & 3);
                    acc += av * scale * canon_value(2, code);
                  }
                }
                out_usm[ng] = to_half_bits(acc);
              });
        });
      }

      // ---- INTFER_DECODE1_VERBVEC: activations through a plain pointer --------
      // ONE change, ADDRESSING ONLY: the activation load goes through a raw
      // `const std::uint16_t*` + offset instead of a `multi_ptr`. The bytes read and
      // the arithmetic are unchanged, so this is bit-exact by construction, and it
      // is the LAST addressing difference between the shipped body and the probe's
      // tidy clone (which uses a multi_ptr too -- so this tests whether the
      // multi_ptr is costing anything in EITHER form).
      // (use_verbvec is declared at the top of the M=1 block, with the other
      // eight opt-in flags, so the exclusivity guard can see it.)

      // ---- INTFER_DECODE1_WVEC: the INT2 weight bytes via a vec load ----------
      // PLAN.md 1.7, the one static-analysis finding with a bit-exact
      // candidate. ONE change to the shipped INT2-tab body: the two weight
      // bytes are read through `sycl::vec<std::uint8_t,2>::load` from the
      // `w_mp` multi_ptr instead of two scalar `code_u8[b0]` / `code_u8[b0+1]`
      // reads. The multi_ptr is built RELATIVE to the row base (the VERBMP
      // idiom), so `b0` stays RELATIVE too -- and the tail deliberately keeps
      // the UNOFFSET `code_u8` with ABSOLUTE indices, exactly as the base
      // kernel does, because that is the ROWPAIR double-offset class that lost
      // the device. Identical bytes read, identical codes extracted,
      // identical accumulation order -> BYTE-IDENTICAL OUTPUT BY CONSTRUCTION,
      // no numerics contract. Moves only the `kernel` bucket.
      // (use_wvec is declared at the top of the M=1 block, with the other
      // eight opt-in flags, so the exclusivity guard can see it.)

      if (use_verbvec && bits == 2 && use_int2_tab && num_groups > 0) {
        decode1_announce("decode1-verbvec");
        return q.submit([&](sycl::handler& h) {
          h.depends_on(copy_dep);
          h.parallel_for<class gemm_usm_decode1_verbvec>(
              sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
              [=](sycl::nd_item<1> it) {
                const int ng = static_cast<int>(it.get_global_id(0));
                if (ng >= N) return;
                float acc = 0.0f;
                const std::uint16_t* sptr = scale_usm + ng * num_groups_w;
                const std::size_t row_base = static_cast<std::size_t>(ng) * K;
                const std::uint16_t* actp = act_usm;   // THE CHANGE
                sycl::multi_ptr<const std::uint8_t,
                    sycl::access::address_space::global_space> wmp(code_u8 + row_base);
                for (int g = 0; g < num_groups; ++g) {
                  const float scale = from_half_bits(sptr[g]) * alpha;
                  const int kg0 = g * Gw;
                  int kg1 = kg0 + Gw;
                  if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                  const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                                         scale * (1.0f / 3.0f), scale * 1.0f};
                  constexpr int CC = 8;
                  int kg = kg0;
                  std::size_t b0 = (row_base + static_cast<std::size_t>(kg)) / 4u;
                  for (; kg + CC <= kg1; kg += CC) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, actp + kg);          // THE CHANGE
                    const std::uint8_t b0v = code_u8[b0];
                    const std::uint8_t b1v = code_u8[b0 + 1];
                    b0 += 2;
                    for (int j = 0; j < CC; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const std::uint8_t bv = (j < 4) ? b0v : b1v;
                      const int code = (bv >> ((j & 3) * 2)) & 3;
                      acc += av * wtab[code];
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    const std::size_t idx = row_base + static_cast<std::size_t>(kg);
                    const int code =
                        static_cast<int>((code_u8[idx / 4] >> ((idx % 4) * 2)) & 3);
                    acc += av * scale * canon_value(2, code);
                  }
                }
                out_usm[ng] = to_half_bits(acc);
              });
        });
      }

      // ----------------------------------------------------------------
      // INTFER_DECODE1_WVEC -- bit-exact INT2 weight-load forming
      // (PLAN.md 1.7). EXACTLY ONE change to the shipped INT2-tab
      // body, and it is a LOAD FORMING change: the two INT2 weight
      // bytes of each 8-code chunk are read through
      // `sycl::vec<std::uint8_t,2>::load` from a row-base multi_ptr
      // (the VERBMP idiom) instead of the two scalar
      // `code_u8[b0]` / `code_u8[b0 + 1]` byte reads. The multi_ptr
      // is built on `code_u8 + row_base / 4u` -- row_base is an
      // ELEMENT offset (ng * K) and four INT2 codes pack per byte,
      // so the /4 is the same conversion VERBMP carries; get it
      // wrong and this is the ROWPAIR double-offset that lost the
      // device. `b0` is therefore RELATIVE to that base, and the
      // tail deliberately keeps the UNOFFSET `code_u8` with
      // ABSOLUTE indices, exactly as the base kernel does.
      //
      // *** WHY THIS IS BIT-EXACT BY CONSTRUCTION, AND ONLY THIS. ***
      // Identical bytes are read (the same two byte addresses the
      // scalar reads touch), the same 4 codes are extracted from each
      // byte with the same `(bv >> ((j & 3) * 2)) & 3`, the same
      // `wtab` is consulted, and the accumulation order is untouched
      // -- only the LOAD IDIOM differs. So the output is
      // BYTE-IDENTICAL to the base body, no numerics contract applies,
      // and `scripts/test_rowpair_equiv.sh` measures it model-free
      // (its `wv` order) while `scripts/test_tp2_27b.sh` asserts the
      // stream identity on real 27B weights.
      //
      // *** MEASURED CLAIMS, SEPARATED FROM THE ARGUMENT. *** The
      // bit-exactness is what the differential below can decide
      // today; the PERFORMANCE direction is UNKNOWN -- a
      // load-forming change moves only the `kernel` bucket, and that
      // bucket's liveness on this box is UNVERIFIED for the engine
      // (AGENTS.md "The profiling defect"). The 27B engine arm is
      // the only way to measure it on real weights (operator-only).
      //
      // *** REACHABILITY. *** This arm sits NINTH in the dispatch
      // chain, behind TWO default-ON arms (verbhalf first, verbmp
      // second), each ending in `return q.submit(...)`. It therefore
      // runs only with BOTH kill switches set AND the presence flag:
      // `INTFER_NO_DECODE1_VERBHALF=1 INTFER_NO_DECODE1_VERBMP=1
      // INTFER_DECODE1_WVEC=1`. Killing only verbhalf leaves verbmp
      // winning the run -- proven model-free at the M=1 INT2
      // MLP_GATE shape (the arm announces decode1-verbmp, not
      // decode1-wvec). The exclusivity guard sits at the TOP of the
      // M=1 block now, so it CAN police this pair: it counts
      // use_verbmp only if VERBMP's kill switch is unset, which is
      // exactly the state in which this arm would be unreachable, and
      // refuses it. (It used to sit after the four earliest arms and
      // could only see verbvec/wvec/mr -- that is what this commit
      // fixed.)
      if (use_wvec && bits == 2 && use_int2_tab && num_groups > 0) {
        decode1_announce("decode1-wvec");
        return q.submit([&](sycl::handler& h) {
          h.depends_on(copy_dep);
          h.parallel_for<class gemm_usm_decode1_wvec>(
              sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
              [=](sycl::nd_item<1> it) {
                const int ng = static_cast<int>(it.get_global_id(0));
                if (ng >= N) return;
                float acc = 0.0f;
                const std::uint16_t* sptr = scale_usm + ng * num_groups_w;
                const std::size_t row_base = static_cast<std::size_t>(ng) * K;
                sycl::multi_ptr<const std::uint16_t,
                    sycl::access::address_space::global_space> act_mp(act_usm);
                // Row-base multi_ptr, the VERBMP idiom: /4 converts the
                // ELEMENT row base to the BYTE base the packed weights
                // live at. RELATIVE byte indices from here on.
                sycl::multi_ptr<const std::uint8_t,
                    sycl::access::address_space::global_space> wmp(
                    code_u8 + row_base / 4u);
                for (int g = 0; g < num_groups; ++g) {
                  const float scale = from_half_bits(sptr[g]) * alpha;
                  const int kg0 = g * Gw;
                  int kg1 = kg0 + Gw;
                  if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                  const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                                         scale * (1.0f / 3.0f), scale * 1.0f};
                  constexpr int CC = 8;
                  int kg = kg0;
                  std::size_t b0 = static_cast<std::size_t>(kg) / 4u;
                  for (; kg + CC <= kg1; kg += CC) {
                    sycl::vec<std::uint16_t, 8> a16;
                    a16.load(0, act_mp + kg);
                    // *** THE ONE CHANGE: the two INT2 weight bytes of
                    // the 8-code chunk as ONE vec load from the
                    // multi_ptr, replacing the two scalar
                    // `code_u8[b0]` / `code_u8[b0 + 1]` reads. ***
                    sycl::vec<std::uint8_t, 2> wv;
                    wv.load(0, wmp + b0);
                    const std::uint8_t b0v = wv[0];
                    const std::uint8_t b1v = wv[1];
                    b0 += 2;
                    for (int j = 0; j < CC; ++j) {
                      const float av =
                          static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                      const std::uint8_t bv = (j < 4) ? b0v : b1v;
                      const int code = (bv >> ((j & 3) * 2)) & 3;
                      acc += av * wtab[code];
                    }
                  }
                  for (; kg < kg1; ++kg) {
                    const float av = from_half_bits(act_usm[kg]);
                    // UNOFFSET base with ABSOLUTE indices, exactly as
                    // the base kernel's tail -- NOT `wmp[idx / 4]`,
                    // which is the ROWPAIR double-offset that lost the
                    // device.
                    const std::size_t idx = row_base + static_cast<std::size_t>(kg);
                    const int code =
                        static_cast<int>((code_u8[idx / 4] >> ((idx % 4) * 2)) & 3);
                    acc += av * scale * canon_value(2, code);
                  }
                }
                out_usm[ng] = to_half_bits(acc);
              });
        });
      }

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

      // --------------------------------------------------------------------
      // INTFER_DECODE1_SPLITK -- split-K decode, default OFF, EXPERIMENTAL.
      //
      // WHY IT EXISTS. The M=1 kernel below launches EXACTLY N work-items, one
      // per output element, and that count is invariant to the work-group
      // shape. At 27B --tp 2 a layer GEMM has N ~ 2048 per rank across 512 Xe
      // cores (~4 threads/EU), so the launch is PARALLELISM-bound, not
      // bandwidth-bound: measured 9.16 GiB of weights per device per token in
      // 257.7 ms of kernel = 35.5 GiB/s, about 21% of the B580's ~179 GB/s.
      //
      // SUPERSEDED 2026-10-05: "512 Xe cores" is the A770's count; the B580 has
      // 20, so the "parallelism-bound" premise above is void. The +43.8%
      // MEASUREMENT this kernel produced stands and it stays default OFF; what
      // is re-examined is the SHAPE -- same 16-way K split, but reduced with a
      // sub-group xor butterfly and NO local_accessor / it.barrier, which is the
      // specific cause the bisection named. See
      // reports/2026-10-05-batch-gemv-roofline.md.
      // The sibling experiment INTFER_DECODE1_MR tried to fix that by giving
      // each work-item MORE outputs and came out 2.3x SLOWER, because dividing
      // an already-starved work-item budget by four is anti-parallel here.
      // This kernel goes the other way: N*S work-items, so parallelism RISES.
      //
      // BLOCK-CYCLIC, AND THAT IS THE WHOLE POINT. The other way to split K is
      // contiguous chunks per work-item, and it does NOT coalesce: with
      // K=2048, S=16, INT2 (8 codes per 2 bytes) lane L reads byte 32L+2j, so
      // the lanes sit 32 bytes apart -- the same stride-K gather the un-split
      // kernel has. Block-cyclic gives lane L the elements {L*8 + t*S*8 + j},
      // so at step j the lanes read bytes 0,2,4,...,30: one contiguous run.
      // Measured on the load pattern, not assumed. Parallelism AND coalescing,
      // which is the whole stated goal, and contiguous chunks would have
      // delivered only half of it.
      //
      // *** THE ARITHMETIC IS DELIBERATELY INLINED. DO NOT EXTRACT IT. ***
      // gemm::quant_gemm_fused_decode1 did exactly that -- hoisted this same
      // per-output body into a shared device function, and a normalized
      // statement-by-statement diff of the two bodies said the arithmetic was
      // identical. It WAS identical, and 27B decode still went 1284 ms ->
      // 231 114 ms, bisected to the extraction itself and not to the fusion it
      // was extracted for. Two homes for this arithmetic is the price of not
      // repeating that. If you are tempted to factor it, that commit is the
      // evidence.
      //
      // *** NUMERICS: NOT BIT-EXACT, BY CONTRACT. *** This reassociates the
      // FP32 sum into S partials combined ascending. The tolerance is not a
      // guess: examples/ksplit_contract measures it (gated by
      // scripts/test_ksplit_contract.sh). Measured margin CONSUMED, in FP16
      // ulps after the store the engine actually reads:
      //     random / act-outliers / full cancellation ... 0 ulp  (all K)
      //     adversarial escalating dataset ................ 2 ulp
      // So the honest claim is "stream-identical on the gate models", NOT
      // "bit-exact for all inputs". The store is FP16 and activations are FP16
      // (|a| <= 65504), which caps the dot product and leaves ~2 orders of
      // magnitude of slack; that slack is the entire safety argument and it is
      // measured, not asserted.
      //
      // Gate: the arm that can actually reach this branch lives in
      // scripts/test_tp2_27b.sh, the only gate whose model carries an INT2
      // tensor (0.8B has zero). An arm in a 0.8B gate would be green because
      // the path never runs.
      // (use_splitk is declared at the top of the M=1 block, with the other
      // eight opt-in flags, so the exclusivity guard can see it.)
      if (use_splitk && bits == 2 && num_groups > 0) {
        // S must divide nWG, which is what puts all S work-items for one output
        // inside a single work-group and makes the local reduction legal.
        constexpr int kVSplit = 8;   // codes per vector load (matches the kernel)
        int S = nWG;                  // full work-group == one output row
        if (S > (K / kVSplit)) S = (K / kVSplit) > 1 ? (K / kVSplit) : 1;
        while (S > 1 && (nWG % S) != 0) S >>= 1;
        if (S > 1) {
          const int nSk = N * S;
          const int globalSk = ((nSk + nWG - 1) / nWG) * nWG;
          const int step = S * kVSplit;   // block-cyclic period, in elements
          const int packSk = pack;         // == 4 for INT2
          // THE PATH ANNOUNCES ITSELF, ONCE -- same shape as INTFER_DECODE1_MR
          // above and for the same reason: without it, an arm that compares
          // this kernel's stream against the base run's cannot tell whether the
          // flag reached the branch. The gate asserts the line is present.
          static bool announced_sk = false;
          if (!announced_sk) {
            announced_sk = true;
            std::fprintf(stderr,
                         "[decode1-splitk] M=1 INT2 decode: split-K S=%d "
                         "ACTIVE (N=%d K=%d bits=%d)\n",
                         S, N, K, bits);
          }
          return q.submit([&](sycl::handler& h) {
            h.depends_on(copy_dep);
            auto lacc = sycl::local_accessor<float, 1>(nWG, h);
            h.parallel_for<class gemm_usm_decode1_vec_splitk>(
                sycl::nd_range<1>(sycl::range<1>(globalSk), sycl::range<1>(nWG)),
                [=](sycl::nd_item<1> it) {
                  const int g_id = static_cast<int>(it.get_global_id(0));
                  const int lid  = static_cast<int>(it.get_local_id(0));
                  const int n  = g_id / S;     // output element
                  const int sp = g_id % S;     // which K-partial
                  const bool live = (n < N);
                  const int row_base = live ? static_cast<int>(
                      static_cast<std::size_t>(n) * K) : 0;
                  const std::uint16_t* sptr =
                      live ? scale_usm + static_cast<std::size_t>(n) * num_groups_w
                           : nullptr;

                  float acc = 0.0f;
                  const float alpha_f = alpha;
                  sycl::multi_ptr<const std::uint16_t,
                      sycl::access::address_space::global_space> act_mp(act_usm);

                  // ---- INLINE arithmetic: do not factor this out. See above.
                  if (live) {
                    const int off = sp * kVSplit;
                    const int g0 = off / Gw;
                    const int g1last = (K - 1) / Gw;
                    for (int g = g0; g <= g1last; ++g) {
                      const int kg0 = g * Gw;
                      int kg1 = kg0 + Gw;
                      if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                      const float scale = from_half_bits(sptr[g]) * alpha_f;
                      const float wtab[4] = {scale * -1.0f,
                                             scale * (-1.0f / 3.0f),
                                             scale * (1.0f / 3.0f),
                                             scale * 1.0f};
                      // First k in [kg0,kg1) matching this split's phase.
                      int k = kg0 + ((off - (kg0 % step) + step) % step);
                      for (; k + kVSplit <= kg1; k += step) {
                        sycl::vec<std::uint16_t, kVSplit> a16;
                        a16.load(0, act_mp + k);
                        const std::size_t b0 =
                            (static_cast<std::size_t>(row_base) +
                             static_cast<std::size_t>(k)) / packSk;
                        const std::uint8_t b0v = code_u8[b0];
                        const std::uint8_t b1v = code_u8[b0 + 1];
                        for (int j = 0; j < kVSplit; ++j) {
                          const float av = static_cast<float>(
                              sycl::bit_cast<sycl::half>(a16[j]));
                          const std::uint8_t bv = (j < 4) ? b0v : b1v;
                          // (j & 3), NOT j. INT2 packs FOUR codes per byte, so
                          // j=4..7 must re-read bit 0 of b1v. Shifting by
                          // j*2 gives shifts of 8..14, which on a uint8_t are
                          // >= the width: the integer promotion makes every one
                          // of those codes 0. That bug shipped in the first
                          // version of this kernel and was caught by the 27B
                          // reachability probe -- stream diverged at step 0
                          // while the kill switch stayed clean, which is what
                          // localised it to this expression rather than to the
                          // guard, the reduction, or the phase arithmetic.
                          const int code = (bv >> ((j & 3) * 2)) & 3;
                          acc += av * wtab[code];
                        }
                      }
                      for (; k < kg1; ++k) {   // ragged tail of the group
                        const float av = from_half_bits(act_usm[k]);
                        const std::size_t idx =
                            static_cast<std::size_t>(row_base) +
                            static_cast<std::size_t>(k);
                        const std::uint8_t byte = code_u8[idx / packSk];
                        const int code = (byte >> ((idx % packSk) * 2)) & 3;
                        acc += av * wtab[code];
                      }
                    }
                  }

                  // ---- reduce the S partials, ascending (contract split{S})
                  lacc[lid] = acc;
                  it.barrier(sycl::access::fence_space::local_space);
                  if (live && sp == 0) {
                    const int lane0 = (n * S) % nWG;
                    float tot = 0.0f;
                    for (int t = 0; t < S; ++t) tot += lacc[lane0 + t];
                    out_usm[n] = to_half_bits(tot);
                  }
                });
          });
        }
      }      // *** INTFER_DECODE1_SUBGROUP WAS QUARANTINED 2026-10-05, AND THE REASON IS
      // A WRONG ANSWER, NOT A SLOW ONE. *** It ran, announced itself, and produced
      // ids '279 12 12' where the base produces '13 198 760' -- deterministically,
      // on every 27B TP=2 gate run. A numerically wrong INT2 decode path must not
      // sit in the tree behind a default-OFF flag with a certification arm that
      // fails; recording that in three documents does not make it acceptable.
      //
      // WHAT WAS RULED OUT, by measurement rather than by reading. SPLITK and
      // SUBGROUP were meant to share the identical 16-way block-cyclic K split and
      // differ ONLY in the reduction, and SPLITK is CORRECT, so the split is sound
      // and the reduction is the suspect. Two candidate causes were TESTED on the
      // CPU against the kernel's own mapping arithmetic:
      //
      //   * "step = row_lanes*kVSg = 128 but a group is only 16 wide, so the chunk
      //     loop overshoots and half of every group is dropped." FALSE. Reproducing
      //     the phase formula over 8 groups (the 128/16 period) showed 128 of 128
      //     k-values covered and all 16 lanes contributing: group g is served by
      //     lanes 2g and 2g+1, and the groups cycle with period 8. Complete.
      //   * A lane offset in `code_u8[b0]` -- the same shape as the ROWPAIR
      //     double-offset that LOST THE DEVICE. FALSE: SUBGROUP indexes the
      //     UNOFFSET `code_u8` with ABSOLUTE byte offsets in both the chunk loop
      //     and the tail, which is correct.
      //
      // SO THE CAUSE IS NOT LOCATED, and it is in one of: the reduce_over_group
      // collective's interaction with this launch's sub-group geometry, or the
      // sg.leader() write. Guessing there and shipping it would repeat the failure
      // mode this repo has paid for repeatedly (AGENTS.md: "structural argument is
      // not measurement").
      //
      // The flag and its kernel are removed rather than defaulted off, because a
      // certification arm that deterministically fails must not exist. Nothing is
      // lost: it had no measured performance, and the full account of what was
      // tried and what was ruled out is reports/2026-10-05-decode1-subgroup.md.
      // Reinstating it needs a MODEL-FREE differential (SUBGROUP vs the base M=1
      // body, bitwise, over shapes and datasets) so the fault is localised without
      // the 27B stack -- which is the tooling that does not exist yet.

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
// XMX joint_matrix FP16-act x INT2/4-weight GEMM.
// Ported from llama.cpp fused-gemm (ggml/src/ggml-sycl/fused-gemm.cpp)
// using the SYCL ext::oneapi::experimental::matrix API.
//
// The portable tiled kernel (quant_gemm_forward_usm) dequantizes weights
// into SLM and accumulates with FP16 FMA on EU/ALU -- the XMX matrix
// engines are idle there (xmxmon reads zero DPAS on this path). This
// kernel uses the SYCL joint_matrix API to issue the XMX DPAS directly
// for the FP16 x FP16 case, after dequantizing the INT2/4 weight tile
// into FP16 in SLM. The dequant is bit-identical to the tiled kernel's
// (same canon_value, same scale, same alpha fold); only the compute
// unit changes: XMX DPAS instead of EU/ALU FMA.
//
// INTFER_XMX_GEMM (default OFF, presence enables). M >= 8 only: XMX
// needs a full TM x TK x TN tile to amortize the SLM round trip; M=1
// decode is memory-bound and the tiled kernel is already at the
// memory roofline.
//
// Tile: TM=16, TN=16, TK=32, SG=16 (one sub-group per tile, native
// Xe2/3 sub-group size). Each sub-group owns a 16x16 output tile and
// walks K in 32-wide steps. Work-group = (N/TN, M/TM) * SG threads.
// Bit-exact by construction: the dequantized FP16 weight tile holds
// the same bits the tiled kernel would load into sW, the activation
// tile holds the same bits as sA, and the DPAS FP32 accumulation
// order is the same ascending-K order as the tiled kernel's inner
// kk loop -- so the output FP16 is bit-identical to the tiled kernel
// and the token stream cannot move.
//
// CALLER GUARD: M % TM == 0, N % TN == 0, K % BK == 0. The kernel
// only runs on full tiles; any residual is handled by the caller
// falling back to quant_gemm_forward_usm.
// ---------------------------------------------------------------------------
namespace {

// XMX tile configuration. TM=TN=16, SG=16 are the Xe2/3 native
// sub-group size and the only accumulator width the hardware accepts.
// TK is the K-width of each joint_matrix_mad step.
//
// *** TK=16, NOT 32. *** The first draft ported llama.cpp fused-gemm's
// fg_combo<fg_half,fg_half,16,16,32,16> verbatim (TK=32, BK=64), and
// the kernel threw on the very first launch on this box:
//   "joint_matrix with parameters matrix_type::fp16, use::a, Rows=16,
//    Cols=32 is not supported on this device"
// i.e. joint_matrix<sub_group,half,use::a,16,32,row_major> and its
// B-side twin <32,16,row_major> are both ILLEGAL on B580. Probed
// exhaustively (examples/xmx_probe's companion probe): the legal
// fp16 use::a combos are K=16 with M in {1,2,4,8,16,32} and K=32 with
// M=1 only; use::b is K=16 N=16 only. So the whole TK=32 axis is
// unsupported here -- it is a Xe2/Xe3-H (512 EU) shape, and B580 is
// Xe2-S (256 EU). TK=16, BK=32 (2 steps per BK tile) is the widest
// legal combo and is what ships.
struct xmx_tile_cfg {
static constexpr int TM = 16;  // output rows per sub-group
static constexpr int TN = 16;  // output cols per sub-group
static constexpr int TK = 16;  // K per inner step (B580 legal width)
static constexpr int SG = 16;  // sub-group size (Xe2/3 native)
static constexpr int BK = 32;  // K per SLM tile (2 x TK steps)
// Work-group COUNT the work-stealing scheduler caps the launch at when
// INTFER_XMX_TILE_SCHED is set. Named, not a literal, because
// examples/xmx_probe --steal asserts sum(tile_ctr) == n_tiles + this value
// and a silent change to the literal silently breaks the probe. Named the
// same magnitude as SG by design: one sub-group per WG, and 16 groups on a
// B580 (20 Xe cores / 160 CUs) is the saturation floor the scheduler exists
// to exceed.
static constexpr int kXmxSchedWG = 16;
};

}  // namespace

sycl::event quant_gemm_forward_usm_xmx(
    sycl::queue& q, std::uint16_t* act_usm,
    const void* code_usm, const std::uint16_t* scale_usm,
    float alpha, std::uint16_t* out_usm,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t group_size, gemm::WeightScheme scheme,
    const sycl::event& copy_dep,
    bool* ran,
    std::uint32_t* tile_ctr) {
  if (scheme == gemm::WeightScheme::FP16) return sycl::event{};
  const int bits = (scheme == gemm::WeightScheme::INT2) ? 2
                 : (scheme == gemm::WeightScheme::INT4) ? 4
                 : 8;
  if (bits == 8) return sycl::event{};  // INT8 uses the ESIMD DPAS path
  if (M < 8) return sycl::event{};      // XMX needs a full tile
  const int Gw = (group_size == 0) ? static_cast<int>(K) : static_cast<int>(group_size);
  const int num_groups_w = (static_cast<int>(K) + Gw - 1) / Gw;
  const auto* code_u8 = static_cast<const std::uint8_t*>(code_usm);

  constexpr int TM = xmx_tile_cfg::TM;
  constexpr int TN = xmx_tile_cfg::TN;
  constexpr int TK = xmx_tile_cfg::TK;
  constexpr int SG = xmx_tile_cfg::SG;
  constexpr int BK = xmx_tile_cfg::BK;

  // Caller must guard: M % TM == 0, N % TN == 0, K % BK == 0.
  if (M % TM != 0 || N % TN != 0 || K % BK != 0) return sycl::event{};

  const int gx = static_cast<int>(M) / TM;
  const int gy = static_cast<int>(N) / TN;
  const int n_tiles = gx * gy;

  // INTFER_XMX_TILE_SCHED (PLAN.md 1.3 port): work-stealing tile scheduler.
  // When set, launches only nWG work-groups (default 16) regardless of
  // n_tiles; each sub-group claims the next unclaimed tile via a global
  // atomic counter, so a sub-group that finishes early pulls the next
  // unfinished tile instead of idling. This is the persistent-kernel
  // pattern: fewer launches, better EU saturation at M=128 prefill.
  //
  // Unlike the DPAS scheduler (per-WI work-stealing via ESIMD
  // slm_atomic_update), the XMX kernel uses joint_matrix which requires
  // ALL SG lanes to participate. So work-stealing is per-SUBGROUP: lane 0
  // claims the next tile, broadcasts it via a local_accessor, and all
  // SG lanes process the claimed tile together.
  //
  // Bit-exact by construction: the process_tile body is shared by both
  // paths; fixed-assignment (bm = group, bn = group) and work-stealing
  // (bm = claimed_tile / gy, bn = claimed_tile % gy) compute the same
  // dot products in the same order, only differing in WHICH tile each
  // sub-group computes.
  // *** THE nWG FALLBACK MUST REMEMBER tile_sched (regression, fixed here). ***
  // INTFER_XMX_TILE_SCHED's contract is "launches ONLY nWG work-groups
  // (default 16) regardless of n_tiles" (the comment above), and examples/
  // xmx_probe --steal asserts that contract: sum(tile_ctr) == n_tiles + nWG
  // with nWG hardcoded to 16 the way the scheduler's own design states. An
  // earlier revision derived `nWG = xmx_wg_active ? nWG_raw : n_tiles`, so
  // with INTFER_XMX_TILE_SCHED set alone the launch widened to n_tiles groups
  // and every sub-group claimed exactly one tile -- the work-stealing
  // degenerated into fixed assignment plus n_tiles atomics, the measured
  // claim counter read 64 (= 32 + 32) against the bound [32, 48], and the
  // persistent-kernel pattern the flag exists for was silently defeated.
  // The output stayed bit-exact, so ONLY the counter could see it. Hence the
  // fallback: tile_sched still caps at 16, and xmx_wg_active keeps its own
  // override (a positive value below n_tiles) on top of it.
  const bool tile_sched = (std::getenv("INTFER_XMX_TILE_SCHED") != nullptr);
  // *** INTFER_XMX_WG (2026-10-15): work-group COUNT for the XMX kernel. ***
  // Default 0 => n_tiles (one sub-group per tile, the shipped fixed-assignment
  // geometry). A positive value caps the launch at that many work-groups and
  // switches on the work-stealing scheduler (fewer groups than tiles means
  // groups must claim multiple tiles, which only the scheduler path supports).
  // The scheduler is a pure DISPATCH change (each tile computes the same dot
  // products in the same order; fixed-assignment and work-stealing share the
  // same process_tile body, quant_gemm.cpp:2288), so this is BIT-EXACT BY
  // CONSTRUCTION -- no numerics contract. The B580 has 20 Xe cores / 160 CUs;
  // the shipped n_tiles is often far below that (prefill M=128, N=4352 =>
  // 8x272 = 2176 tiles, which is 136x the 16-group floor), so a cap is the
  // decode-analogue of INTFER_DECODE1_WG and the prefill analogue of
  // INTFER_DPAS48_WG. Unlike those two this rides the XMX path, so it needs
  // NO path selector -- the kernel declines non-eligible shapes before the
  // dispatch geometry is consulted.
  //
  // *** WHY IT IS NOT MERGED WITH INTFER_XMX_TILE_SCHED. *** Both knobs move
  // nWG, and either alone implies the other (nWG < n_tiles forces work-stealing
  // or tiles are re-run). They are kept separate because the gate arm asserts
  // the scheduler state the kernel actually used: INTFER_XMX_TILE_SCHED's arm
  // asserts "OFF / fixed-assignment" with nWG == n_tiles, and a combined knob
  // would make that arm unsatisfiable whenever the width flag was set. Two
  // knobs, two arms, each asserting its own observable.
  //
  // *** xmx_wg_active IS TRUE ONLY WHEN THE CAP ACTUALLY BITES. *** A value
  // >= n_tiles is a no-op (nWG falls back to n_tiles, the launch is the
  // default fixed-assignment geometry), so it must NOT switch the scheduler
  // on and must NOT announce -- otherwise an operator setting
  // INTFER_XMX_WG=999999 (a "I want everything" value) would get a
  // work-stealing launch and a misleading announcement for a run that is
  // byte-for-byte the default. The gate arm asks for a cap below n_tiles, so
  // this is the shape it certifies; the refusal direction is the no-op value.
  const uint32_t nWG_raw =
      (std::getenv("INTFER_XMX_WG") != nullptr)
          ? static_cast<uint32_t>(std::atol(std::getenv("INTFER_XMX_WG")))
          : 0u;
  const bool xmx_wg_active =
      (nWG_raw > 0 && nWG_raw < static_cast<uint32_t>(n_tiles));
  const uint32_t nWG = xmx_wg_active ? nWG_raw
                                     : (tile_sched
                                            ? xmx_tile_cfg::kXmxSchedWG
                                            : static_cast<uint32_t>(n_tiles));
  const bool tile_sched_effective = tile_sched || xmx_wg_active;
  const uint32_t n_tiles_u = static_cast<uint32_t>(n_tiles);

  // Work-group: one sub-group per tile (SG threads), SG lanes per sub-group.
  // joint_matrix_load/store/mad REQUIRE a full sub-group (SG=16 on Xe2/3).
  sycl::nd_range<1> nd(sycl::range<1>(nWG * SG),
                       sycl::range<1>(SG));

  // Capture the event rather than returning the submit() result directly,
  // so `ran` can be set: the caller must know whether a real submission
  // happened (a null event here means the output was never written).
  sycl::event xmx_evt = q.submit([&](sycl::handler& h) {
    if (copy_dep != sycl::event{}) h.depends_on(copy_dep);
    // SLM: dequantized weight tile [BK x TN] (k-major, so a
    // row_major joint_matrix load reads it as [TK x TN] per step).
    // One tile per work-group; the SLM is private to that group.
    sycl::local_accessor<sycl::half, 1> sW(sycl::range<1>(BK * TN), h);
    // Broadcast slot: lane 0 writes the claimed tile ID, all lanes read it.
    sycl::local_accessor<int32_t, 1> tile_buf(sycl::range<1>(1), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(SG)]] {
      const auto sg = it.get_sub_group();
      const int lane = sg.get_local_id()[0];

      // ---- Per-tile body (shared by fixed-assignment and work-stealing). ----
      auto process_tile = [&](int bm, int bn) {
        const int m0 = bm * TM;
        const int n0 = bn * TN;
        const bool tile_valid = (bm < gx) && (bn < gy);

        using namespace sycl::ext::oneapi::experimental::matrix;
        joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN,
                     layout::dynamic> acc;
        joint_matrix_fill(sg, acc, 0.0f);

        const int nk_steps = static_cast<int>(K) / BK;
        for (int bk = 0; bk < nk_steps; ++bk) {
          const int k0 = bk * BK;
          // Phase 1: dequantize the weight tile into SLM. Every lane
          // dequantizes a contiguous slice of the BK*TN tile so the
          // code loads coalesce (one lane per (k, n) pair, stride SG).
          // Bit-identical to the tiled kernel's sW load.
          for (int e = lane; e < BK * TN; e += SG) {
            const int k = e / TN;
            const int n = e % TN;
            const int ng = n0 + n;
            const int kg = k0 + k;
            float wval = 0.0f;
            if (tile_valid && ng < static_cast<int>(N) && kg < static_cast<int>(K)) {
              const std::size_t idx = static_cast<std::size_t>(ng) * K + kg;
              const int g = kg / Gw;
              const float s = from_half_bits(
                  scale_usm[ng * num_groups_w + g]) * alpha;
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
            sW[e] = sycl::half(wval);
          }
          it.barrier();

          // Phase 2: XMX DPAS over the BK-wide K tile in TK steps.
          // A is loaded from USM (FP16, row-major); B is loaded from
          // SLM (FP16, row-major, k-major layout).
          for (int kt = 0; kt < BK / TK; ++kt) {
            const int kg0 = k0 + kt * TK;
            // A tile: [TM x TK] from act_usm, row-major.
            joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK,
                         layout::row_major> sub_a;
            // B tile: [TK x TN] from sW, row-major (k-major in SLM).
            joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN,
                         layout::row_major> sub_b;
            // A load: act_usm[(m0 + m) * K + (kg0 + k)] for m in [0,TM), k in [0,TK).
            const auto a_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(
                reinterpret_cast<sycl::half*>(act_usm));
            // B load: sW[k * TN + n] for k in [kt*TK, kt*TK+TK), n in [0,TN).
            const auto b_ptr = sW.get_multi_ptr<sycl::access::decorated::no>();
            joint_matrix_load(sg, sub_a,
                a_ptr + static_cast<std::size_t>(m0) * K + kg0,
                static_cast<size_t>(K));
            joint_matrix_load(sg, sub_b,
                b_ptr + static_cast<std::size_t>(kt) * TK * TN,
                static_cast<size_t>(TN));
            joint_matrix_mad(sg, acc, sub_a, sub_b, acc);
          }
          it.barrier();
        }

        // Store: FP32 acc -> FP16 out_usm[(m0+m)*N + (n0+n)].
        // joint_matrix_copy converts float acc to half acc (RNE round),
        // then joint_matrix_store writes [TM x TN] row-major, stride N.
        joint_matrix<sycl::sub_group, sycl::half, use::accumulator, TM, TN,
                       layout::dynamic> acc_half;
        joint_matrix_copy(sg, acc, acc_half);
        const auto out_ptr = sycl::address_space_cast<
            sycl::access::address_space::global_space,
            sycl::access::decorated::no>(
            reinterpret_cast<sycl::half*>(out_usm));
        joint_matrix_store(sg, acc_half,
            out_ptr + static_cast<std::size_t>(m0) * N + n0,
            static_cast<size_t>(N), layout::row_major);
      };  // process_tile

      // ---- DISPATCH: work-stealing vs fixed-assignment. ----
      // tile_sched_effective is true when EITHER knob requested it
      // (INTFER_XMX_TILE_SCHED, or INTFER_XMX_WG capping the group count
      // below n_tiles). Both paths share process_tile, so the output is
      // bit-exact either way -- the predicate only changes WHICH tile each
      // sub-group computes.
      if (!tile_sched_effective) {
        // Fixed assignment: WG g does tile (bm, bn) where
        // g is the LINEAR work-group id. The range is nd_range<1>
        // (one sub-group per tile), so get_group() has exactly one
        // dimension -- the first draft read it.get_group(1), which
        // is an out-of-range dimension on a 1D range (undefined
        // behaviour) and faulted the device with
        // UR_RESULT_ERROR_DEVICE_LOST on the very first launch.
        // Derive (bm, bn) from the single group index instead.
        const int g = static_cast<int>(it.get_group(0));
        const int bm = g / gy;
        const int bn = g % gy;
        if (bm < gx && bn < gy) process_tile(bm, bn);
      } else {
        // Work-stealing: lane 0 claims the next tile, all SG lanes
        // process it via broadcast. The counter is ONE uint32_t
        // in global USM shared by every work-group, 0-init by
        // the caller. fetch_add returns the pre-increment value
        // -- "claimed" iff old < n_tiles.
        //
        // *** THE SCOPE IS DEVICE, NOT WORK_GROUP. *** This was
        // a real bug, and it is invisible in the output: the
        // counter is GLOBAL USM shared across all work-groups,
        // so a work_group-scoped atomic has NO cross-group
        // ordering guarantee. Every work-group reads the same
        // pre-inc value, writes the same post-inc value, and the
        // update is lost -- the claim degenerates and tiles are
        // re-run. The OUTPUT can still come out bit-exact (the
        // same tile computed twice is the same result), so the
        // race is detectable ONLY in the claim counter, which is
        // why the probe asserts sum == n_tiles.
        //
        // The counter is a single shared address (tile_ctr[0]),
        // NOT one slot per work-group: a per-WG slot would make
        // every work-group claim the same tiles independently
        // (measured: n_tiles=32, nWG=16 -> sum=528 = 16 x 33,
        // min=33, every tile claimed 16 times). The earlier
        // comment ("device-wide would serialize across groups")
        // mistook correctness for cost -- serialization on one
        // address is the whole point of a claim counter, and one
        // atomic per claimed tile is negligible next to a GEMM
        // tile.
        static constexpr auto RELAXED = sycl::memory_order::relaxed;
        static constexpr auto SCOPE = sycl::memory_scope::device;
        while (true) {
          tile_buf[0] = -1;
          it.barrier();
          if (lane == 0) {
            sycl::atomic_ref<std::uint32_t, RELAXED, SCOPE>
                ref(tile_ctr[0]);
            uint32_t cur = ref.fetch_add(1);
            if (cur < n_tiles_u) tile_buf[0] = static_cast<int32_t>(cur);
          }
          it.barrier();
          int tile_id = tile_buf[0];
          if (tile_id < 0) break;  // no more tiles
          const int bm = tile_id / gy;
          const int bn = tile_id % gy;
          process_tile(bm, bn);
        }
      }
    });
  });
  // *** THE PATH ANNOUNCES ITSELF, ONCE. *** Same shape as the [decode1-wg]
  // line (quant_gemm.cpp:494) and the [xmx-tile-sched] line the engine emits:
  // a base-vs-flagged stream comparison is satisfiable by a flag that never
  // reached the branch (the scheduler is a pure dispatch change, so the two
  // streams are bit-identical by construction), so the arm asserts the WIDTH
  // and the scheduler state the kernel ACTUALLY used rather than a constant.
  // Gated on xmx_wg_active (the flag was read AND the cap actually bit --
  // a value >= n_tiles is a no-op and must not announce), not on
  // tile_sched_effective: the scheduler may be on for its own reasons
  // (INTFER_XMX_TILE_SCHED), and this line must not fire then -- the
  // [xmx-tile-sched] arm asserts scheduler OFF with nWG==n_tiles, so a line
  // here would make that arm's "OFF" assertion unsatisfiable whenever the
  // width flag was also set. Two knobs, two announcements, each asserting
  // its own observable.
  if (xmx_wg_active) {
    static bool announced_xmx_wg = false;
    if (!announced_xmx_wg) {
      announced_xmx_wg = true;
      std::fprintf(stderr,
                   "[xmx-wg] XMX work-group count=%u "
                   "(INTFER_XMX_WG=%s) scheduler=%s n_tiles=%u "
                   "(%s work-stealing)\n",
                   nWG,
                   std::getenv("INTFER_XMX_WG") ?
                       std::getenv("INTFER_XMX_WG") : "0",
                   tile_sched_effective ? "ACTIVE" : "OFF",
                   n_tiles_u,
                   tile_sched_effective ? "persistent" : "fixed-assignment");
      std::fflush(stderr);
    }
  }
  if (ran) *ran = true;
  return xmx_evt;
}


// ---------------------------------------------------------------------------
// FUSED GATE+UP GEMM (INTFER_FUSED_GATE_UP). See quant_gemm.hpp.
sycl::event quant_gemm_forward_usm_gateup(
    sycl::queue& q, std::uint16_t* act_usm,
    const void* code_g, const std::uint16_t* scale_g, float alpha_g,
    std::uint16_t* out_g,
    const void* code_u, const std::uint16_t* scale_u, float alpha_u,
    std::uint16_t* out_u,
    uint32_t M, uint32_t Ng, uint32_t Nu, uint32_t K,
    uint32_t group_size,
    uint32_t start_g, uint32_t Nl_g, uint32_t start_u, uint32_t Nl_u,
    gemm::WeightScheme scheme_g, gemm::WeightScheme scheme_u,
    const sycl::event& copy_dep, bool compact) {
  if (M == 0 || K == 0 || (Ng == 0 && Nu == 0)) return sycl::event();
  if (group_size == 0) group_size = K;
  const int Gw = static_cast<int>(group_size);
  const int num_groups = (static_cast<int>(K) + Gw - 1) / Gw;
  if (num_groups <= 0) return sycl::event();

  // *** THE DISPATCH IS THE SLICE, NOT THE WHOLE OUTPUT. *** The shard holds
  // only rows [start, start+Nl), so those are the only rows this device can
  // compute. Dispatching the full Ng+Nu would have every rank compute rows it
  // holds no weights for, and (before the shard was sliced correctly) read
  // ~4x past the shard's end.
  constexpr int nWG = kThr;
  const int total = static_cast<int>(Nl_g) + static_cast<int>(Nl_u);
  const int nBlocks = (total + nWG - 1) / nWG;
  const int global = nBlocks * nWG;

  // INT2 + INT2 with the INT2-tab fast path: the 27B MLP gate/up are INT2.
  // ONE WI per (row m, output ng), each with its own accumulator ascending
  // kg then ascending j -- so the fused launch is bit-exact vs two separate
  // quant_gemm_forward_usm calls at ANY M, not just M=1. The M dimension is
  // the outer index: act_usm is M*K row-major, out_g/out_u are M*Ng/M*Nu.
  if (scheme_g == WeightScheme::INT2 && scheme_u == WeightScheme::INT2) {
    const int bits = 2;
    const int pack = 8 / bits;   // 4 codes per byte
    const auto* code_g8 = static_cast<const std::uint8_t*>(code_g);
    const auto* code_u8 = static_cast<const std::uint8_t*>(code_u);
    return q.submit([&](sycl::handler& h) {
      h.depends_on(copy_dep);
      h.parallel_for<class gemm_usm_decode1_gateup_int2tab>(
          sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
          [=](sycl::nd_item<1> it) {
            const int ng = static_cast<int>(it.get_global_id(0));
            if (ng >= total) return;
            // *** LOCAL ROW (SHARD) vs GLOBAL ROW (OUTPUT). *** The work item
            // ids run over THIS DEVICE'S SLICE (total = Nl_g + Nl_u), so
            // ng is a local row: local_g in [0, Nl_g) indexes the gate shard,
            // local_u in [0, Nl_u) indexes the up shard. The GLOBAL output row
            // is start + local, and only the output buffer (full M*Ng/M*Nu)
            // uses it; the shard (code bytes, scale rows) uses the local one.
            const bool is_up = ng >= static_cast<int>(Nl_g);
            const uint32_t local = is_up ? (uint32_t)(ng - (int)Nl_g) : (uint32_t)ng;
            const uint32_t out_ng = (is_up ? start_u : start_g) + local;
            // Index the scale shard by the LOCAL row (w.N = Nl rows, engine.cpp:312).
            const std::uint16_t* sptr =
                is_up ? (scale_u + (std::size_t)local * num_groups)
                      : (scale_g + (std::size_t)local * num_groups);
            const std::uint8_t* wbase =
                is_up ? code_u8 : code_g8;
            const float alpha_f = is_up ? alpha_u : alpha_g;
            std::uint16_t* outp = is_up ? out_u : out_g;
            // *** THE WEIGHT ROW IS A BYTE ROW, NOT AN ELEMENT ROW. *** The
            // shard is packed by shard_weight (engine.cpp:357: code_slice is
            // w.N * (K+pack-1)/pack bytes, four INT2 codes per byte) and the
            // USM copy is byte-for-byte of that buffer (engine.cpp:2621
            // sc.code_host[d] = w.code_store; shard_fill copies code_bytes).
            // So the row stride is K/pack BYTES, and the byte indices below
            // (b0, idx) are ABSOLUTE byte offsets into the shard already
            // divided by pack -- which is why wrow is the shard BASE, not
            // the shard base plus an element row offset.
            //
            // *** AND THE SHARD IS A TP SLICE, NOT THE FULL TENSOR. ***
            // shard_weight(model_, t, start, end, q) (engine.cpp:278) emits
            // LOCAL rows: w.N = end - start, row_codes[r] for r in 0..w.N,
            // scale_slice[r] for r in 0..w.N (engine.cpp:312-316). The shard
            // holds rows start..end of the tensor, indexed 0..Nl-1.
            //
            // On a single-device run (and every full-size shape in
            // examples/gateup_equiv) start=0, Nl=Ng, so `local` == the global
            // row and this is byte-identical to the TP=1 behaviour.
            for (int m = 0; m < static_cast<int>(M); ++m) {
              float acc = 0.0f;
              const std::size_t act_off = (std::size_t)m * K;
              const std::uint8_t* wrow = wbase;
              sycl::multi_ptr<const std::uint16_t,
                  sycl::access::address_space::global_space> act_mp(act_usm);
              for (int g = 0; g < num_groups; ++g) {
                const float scale = from_half_bits(sptr[g]) * alpha_f;
                const int kg0 = g * Gw;
                int kg1 = kg0 + Gw;
                if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
                const float wtab[4] = {scale * -1.0f, scale * (-1.0f / 3.0f),
                                       scale * (1.0f / 3.0f), scale * 1.0f};
                constexpr int CC = 8;
                int kg = kg0;
                // `local` is the row within THIS shard (0..Nl-1); the shard is packed
                                // row-major so the byte row is local * K, and the byte within
                                // it is kg / pack.
                                std::size_t b0 = ((std::size_t)local * K +
                                  (std::size_t)kg) /
                                 (std::size_t)pack;
                for (; kg + CC <= kg1; kg += CC) {
                  sycl::vec<std::uint16_t, 8> a16;
                  a16.load(0, act_mp + act_off + kg);
                  const std::uint8_t b0v = wrow[b0];
                  const std::uint8_t b1v = wrow[b0 + 1];
                  b0 += 2;
                  for (int j = 0; j < CC; ++j) {
                    const float av =
                        static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                    const std::uint8_t bv = (j < 4) ? b0v : b1v;
                    const int code = (bv >> ((j & 3) * 2)) & 3;
                    acc += av * wtab[code];
                  }
                }
                for (; kg < kg1; ++kg) {
                  const float av = from_half_bits(
                      act_usm[act_off + (std::size_t)kg]);
                  const std::size_t idx =
                      (std::size_t)local * K + (std::size_t)kg;
                  const std::uint8_t byte = wrow[idx / pack];
                  const int within = static_cast<int>(idx % pack);
                  const int code = (byte >> (within * bits)) &
                                   ((1 << bits) - 1);
                  acc += av * scale * canon_value(bits, code);
                }
              }
              // *** STORE LAYOUT IS PER-MODE, AND IT IS THE ONLY THING THAT
              // CHANGES. *** compact=true writes the device's own slice
              // compactly (outp[m*Nl + local], Nl = Nl_g for gate / Nl_u for
              // up) so the downstream silumul_to_half kernel can read it as one
              // contiguous [M*inter] buffer; compact=false (the default) writes
              // GLOBAL-SCATTER into a full-width buffer (outp[m*(is_up?Nu:Ng)
              // + out_ng]) so the host readback can pick this device's slice
              // out of it. The accumulator, the kg/j element order and the
              // wtab are identical in both modes, so the two forms agree on
              // every column this device owns.
              const uint32_t Nl = is_up ? Nl_u : Nl_g;
              const std::size_t store_off =
                  compact ? (std::size_t)m * Nl + local
                          : (std::size_t)m * (is_up ? Nu : Ng) + out_ng;
              outp[store_off] = to_half_bits(acc);
            }
          });
    });
  }

  return sycl::event();
}

// *** INTFER_MLP_GATEUP_SILUMUL_FUSE (2026-10-14). ***
// Fused gate+up GEMM + silu(gate)*up epilogue in ONE launch.
// Default OFF (INTFER_MLP_GATEUP_SILUMUL_FUSE, presence spells).
//
// Removes the separate silumul_to_half launch: with the flag set the
// per-device chain is ONE launch (this kernel) instead of TWO (compact
// gateup + silumul_to_half chained off its event), so 1 submit and 1
// wait are removed per device per MLP layer. (The partial-down path has
// NO D2H readback in either branch -- gate/up slots stay on device and
// the host sums the down partials straight from USM -- so the win is
// launch count, not transfer count.)
//
// BIT-IDENTITY CONTRACT. The accumulator, the kg/j element order and
// the wtab are identical to the compact gateup kernel — only the
// epilogue changes. silu(g) = g/(1+exp(-g)) is computed in FP32 with
// sycl::exp (the same call site silumul_to_half uses), and the store
// is to_half_bits (sycl::half RNE, bit-identical to silumul_to_half's
// dev_float_to_half). So every element is the same as if the compact
// gateup output had been fed through silumul_to_half. Stream-identity
// gate is warranted.
//
// TP=2-VALID: per-device slice, no cross-device dependency, no P2P.
// Only the INT2+INT2 body exists; any other scheme pair returns
// sycl::event() and the caller keeps the two-launch path.
//
// The fused kernel is placed AFTER the compact gateup kernel (which it
// is a strict superset of) so it can see to_half_bits, which is defined
// at the top of this file.
sycl::event quant_gemm_forward_usm_gateup_compact_silumul(
    sycl::queue& q, std::uint16_t* act_usm,
    const void* code_g, const std::uint16_t* scale_g, float alpha_g,
    std::uint16_t* out_g,
    const void* code_u, const std::uint16_t* scale_u, float alpha_u,
    std::uint16_t* out_u,
    uint32_t M, uint32_t Ng, uint32_t Nu, uint32_t K,
    uint32_t group_size,
    uint32_t start_g, uint32_t Nl_g, uint32_t start_u, uint32_t Nl_u,
    gemm::WeightScheme scheme_g, gemm::WeightScheme scheme_u,
    const sycl::event& copy_dep,
    std::uint16_t* act_fp16) {
  // *** INTFER_MLP_GATEUP_SILUMUL_FUSE dispatch. ***
  // Each WI computes ONE local column's gate AND up dot products (gate and
  // up are column-parallel over the SAME split on TP=2, so Nl_g == Nl_u and
  // the dispatch is Nl_g WIs). The compact gateup kernel computes one column
  // per WI too; this kernel does the same work twice in the same WI (one
  // accumulator per output, ascending kg, ascending j, its own group scale
  // and wtab), so the instruction stream is the compact kernel's plus a
  // silu-mul. Nothing is summed across WIs. Stream-identity gate is
  // warranted -- the compact gateup output is byte-identical and the fused
  // act_fp16 is byte-identical to silumul_to_half's output.
  //
  // act_fp16 is M x (Nl_g + Nl_u) compact: gate columns 0..Nl_g-1, then up
  // columns Nl_g..Nl_g+Nl_u-1, row-major. This is the SAME layout the
  // silumul_to_half kernel would have produced from the compact gateup
  // outputs (out_g is M*Nl_g, out_u is M*Nl_u, concatenated). The down GEMM
  // reads ws_[d].act as M*inter with inter = Nl_g + Nl_u (the full inter on
  // TP=2), so this is exactly the buffer it expects.
  //
  // Returns sycl::event() on any non-INT2+INT2 scheme pair; the caller keeps
  // the two-launch path. Default OFF (INTFER_MLP_GATEUP_SILUMUL_FUSE).
  if (scheme_g != WeightScheme::INT2 || scheme_u != WeightScheme::INT2) {
    return sycl::event();
  }
  if (M == 0 || K == 0 || (Ng == 0 && Nu == 0)) return sycl::event{};
  if (group_size == 0) group_size = K;
  const int Gw = static_cast<int>(group_size);
  const int num_groups = (static_cast<int>(K) + Gw - 1) / Gw;
  if (num_groups <= 0) return sycl::event{};

  // Dispatch same as compact gateup: one WI per (row m, local output ng).
  // Each WI computes BOTH gate and up for its local column (gate and up are
  // column-parallel over the SAME split on TP=2, so Nl_g == Nl_u).
  constexpr int nWG = kThr;
  const int total = static_cast<int>(Nl_g);
  const int nBlocks = (total + nWG - 1) / nWG;
  const int global = nBlocks * nWG;

  const int bits = 2;
  const int pack = 8 / bits;
  const auto* code_g8 = static_cast<const std::uint8_t*>(code_g);
  const auto* code_u8 = static_cast<const std::uint8_t*>(code_u);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(copy_dep);
    h.parallel_for<class gemm_usm_decode1_gateup_int2tab_silumul>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) {
const int ng = static_cast<int>(it.get_global_id(0));
          if (ng >= total) return;
          // *** ONE WI PER LOCAL COLUMN, COMPUTING BOTH GATE AND UP. ***
          // Gate and up are column-parallel over the SAME split (run_mlp_partial_down),
          // so Nl_g == Nl_u on every reachable call and the dispatch is Nl_g WIs, each
          // computing BOTH gate and up for its local column, sharing the activation
          // load. This is the compact gateup's per-column work (one accumulator, one
          // kg/j order, one wtab) done TWICE in the same WI, so the instruction
          // stream is the compact kernel's plus a silu-mul; nothing is summed across
          // WIs. Stream-identity gate is warranted.
          const uint32_t local = static_cast<uint32_t>(ng);
          const std::uint16_t* sptr_g =
              scale_g + (std::size_t)local * num_groups;
          const std::uint16_t* sptr_u =
              scale_u + (std::size_t)local * num_groups;
          const std::uint8_t* wrow_g = code_g8;
          const std::uint8_t* wrow_u = code_u8;

          for (int m = 0; m < static_cast<int>(M); ++m) {
            float acc_g = 0.0f;
            float acc_u = 0.0f;
            const std::size_t act_off = (std::size_t)m * K;
            sycl::multi_ptr<const std::uint16_t,
                sycl::access::address_space::global_space> act_mp(act_usm);
            for (int g = 0; g < num_groups; ++g) {
              const float scale_g_val = from_half_bits(sptr_g[g]) * alpha_g;
              const float wtab_g[4] = {scale_g_val * -1.0f,
                                       scale_g_val * (-1.0f / 3.0f),
                                       scale_g_val * (1.0f / 3.0f),
                                       scale_g_val * 1.0f};
              const float scale_u_val = from_half_bits(sptr_u[g]) * alpha_u;
              const float wtab_u[4] = {scale_u_val * -1.0f,
                                       scale_u_val * (-1.0f / 3.0f),
                                       scale_u_val * (1.0f / 3.0f),
                                       scale_u_val * 1.0f};
              const int kg0 = g * Gw;
              int kg1 = kg0 + Gw;
              if (kg1 > static_cast<int>(K)) kg1 = static_cast<int>(K);
              constexpr int CC = 8;
              int kg = kg0;
              std::size_t b0_g = ((std::size_t)local * K +
                                  (std::size_t)kg) / (std::size_t)pack;
              std::size_t b0_u = b0_g;
              for (; kg + CC <= kg1; kg += CC) {
                sycl::vec<std::uint16_t, 8> a16;
                a16.load(0, act_mp + act_off + kg);
                const std::uint8_t b0v_g = wrow_g[b0_g];
                const std::uint8_t b1v_g = wrow_g[b0_g + 1];
                const std::uint8_t b0v_u = wrow_u[b0_u];
                const std::uint8_t b1v_u = wrow_u[b0_u + 1];
                b0_g += 2;
                b0_u += 2;
                for (int j = 0; j < CC; ++j) {
                  const float av =
                      static_cast<float>(sycl::bit_cast<sycl::half>(a16[j]));
                  const std::uint8_t bv_g = (j < 4) ? b0v_g : b1v_g;
                  const int code_g_val = (bv_g >> ((j & 3) * 2)) & 3;
                  acc_g += av * wtab_g[code_g_val];
                  const std::uint8_t bv_u = (j < 4) ? b0v_u : b1v_u;
                  const int code_u_val = (bv_u >> ((j & 3) * 2)) & 3;
                  acc_u += av * wtab_u[code_u_val];
                }
              }
              for (; kg < kg1; ++kg) {
                const float av =
                    from_half_bits(act_usm[act_off + (std::size_t)kg]);
                const std::size_t idx =
                    (std::size_t)local * K + (std::size_t)kg;
                const std::uint8_t byte_g = wrow_g[idx / pack];
                const int within = static_cast<int>(idx % pack);
                const int code_g_val =
                    (byte_g >> (within * bits)) & ((1 << bits) - 1);
                acc_g += av * scale_g_val * canon_value(bits, code_g_val);
                const std::uint8_t byte_u = wrow_u[idx / pack];
                const int code_u_val =
                    (byte_u >> (within * bits)) & ((1 << bits) - 1);
                acc_u += av * scale_u_val * canon_value(bits, code_u_val);
              }
            }
            // Compact stores (same as the compact gateup kernel).
            const std::size_t store_off_g =
                (std::size_t)m * Nl_g + local;
            out_g[store_off_g] = to_half_bits(acc_g);
            const std::size_t store_off_u =
                (std::size_t)m * Nl_u + local;
            out_u[store_off_u] = to_half_bits(acc_u);
            // *** FUSED EPILOGUE. *** silu(gate)*up, same FP32 order and
            // same RNE FP16 store as silumul_to_half. The fused activation
            // buffer is compact [M*(Nl_g+Nl_u)] row-major: gate columns
            // 0..Nl_g-1 then up columns Nl_g..Nl_g+Nl_u-1, so this WI's
            // column is `local` in both halves.
            const float s = acc_g / (1.0f + sycl::exp(-acc_g));
            const std::size_t act_off_fused =
                (std::size_t)m * (Nl_g + Nl_u) + (std::size_t)local;
            act_fp16[act_off_fused] = to_half_bits(s * acc_u);
          }
        });
  });
}

// *** INTFER_MLP_PARTIAL_DOWN (2026-10-14): compact-write variant. ***
// *** INTFER_MLP_GATEUP_SILUMUL_FUSE (2026-10-14): fused gate+up + silu-mul. ***
// The fused kernel lives ABOVE the compact wrapper (which it is a strict
// superset of) so it can see to_half_bits, which is defined at the top
// of this file. The compact wrapper is kept as a named function so the
// partial-down path cannot accidentally pass compact=false and feed a
// global-scatter buffer into silumul_to_half -- the two layouts are
// not interchangeable, and a single spelling makes the call site
// self-documenting.
sycl::event quant_gemm_forward_usm_gateup_compact(
    sycl::queue& q, std::uint16_t* act_usm,
    const void* code_g, const std::uint16_t* scale_g, float alpha_g,
    std::uint16_t* out_g,
    const void* code_u, const std::uint16_t* scale_u, float alpha_u,
    std::uint16_t* out_u,
    uint32_t M, uint32_t Ng, uint32_t Nu, uint32_t K,
    uint32_t group_size,
    uint32_t start_g, uint32_t Nl_g, uint32_t start_u, uint32_t Nl_u,
    gemm::WeightScheme scheme_g, gemm::WeightScheme scheme_u,
    const sycl::event& copy_dep) {
  return quant_gemm_forward_usm_gateup(
      q, act_usm, code_g, scale_g, alpha_g, out_g,
      code_u, scale_u, alpha_u, out_u,
      M, Ng, Nu, K, group_size, start_g, Nl_g, start_u, Nl_u,
      scheme_g, scheme_u, copy_dep, /*compact=*/true);
}

// ---------------------------------------------------------------------------
// FUSED WAVE LAUNCH (INTFER_WAVE_LAUNCH_FUSE).
//
// One parallel_for over the UNION of the wave's output elements. Each item's
// accumulation is untouched (its own accumulator, ascending kg, ascending j,
// its own group scale, its own wtab), so the launch is bit-exact vs N separate
// quant_gemm_forward_usm calls. The M dimension is the outer index for every
// item (act_usm[i] is M*K row-major, out_usm[i] is M*N row-major), and the two
// offsets are computed separately and never combined -- the same guard the
// gate+up kernel uses, which is what makes the M>1 prefill path safe here.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_forward_usm_wave(
    sycl::queue& q,
    const std::uint16_t* const* act_usm,
    const void* const* code_usm,
    const std::uint16_t* const* scale_usm,
    const float* alpha,
    std::uint16_t* const* out_usm,
    uint32_t N_items,
    const uint32_t* M_arr, const uint32_t* N_arr, const uint32_t* K_arr,
    const uint32_t* group_arr,
    const gemm::WeightScheme* scheme,
    const sycl::event& copy_dep) {
  if (N_items == 0) return sycl::event();
  int total = 0;
  for (uint32_t i = 0; i < N_items; ++i) {
    if (M_arr[i] == 0 || N_arr[i] == 0 || K_arr[i] == 0) continue;
    total += static_cast<int>(M_arr[i]) * static_cast<int>(N_arr[i]);
  }
  if (total <= 0) return sycl::event();

  // Prefix sums over the items' output counts. N_items <= 16 (the wave
  // contract), so a fixed-size C array is device-copyable by value -- a
  // std::vector is not, and capturing it would fail the kernel's
  // is_device_copyable static_assert.
  int start[17];
  start[0] = 0;
  for (uint32_t i = 0; i < N_items; ++i)
    start[i + 1] = start[i] +
                   static_cast<int>(M_arr[i]) * static_cast<int>(N_arr[i]);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(copy_dep);
    h.parallel_for<class gemm_usm_decode1_wave_fuse>(
        sycl::nd_range<1>(sycl::range<1>(total), sycl::range<1>(kThr)),
        [=](sycl::nd_item<1> it) {
          const int gid = static_cast<int>(it.get_global_id(0));
          if (gid >= total) return;
          // Linear search over the prefix sums (N_items <= 16, so this is
          // cheap and keeps the kernel branch-free of any loop-carried
          // dependency on the item count).
          int item = 0;
          while (item < static_cast<int>(N_items) - 1 &&
                 gid >= start[item + 1])
            ++item;
          const int local = gid - start[item];
          const int m = local / static_cast<int>(N_arr[item]);
          const int n = local % static_cast<int>(N_arr[item]);
          const int K = static_cast<int>(K_arr[item]);
          const int Mi = static_cast<int>(M_arr[item]);
          const int Ni = static_cast<int>(N_arr[item]);
          const std::uint16_t* act = act_usm[item];
          const std::uint16_t* sptr = scale_usm[item];
          const float alpha_f = alpha[item];
          std::uint16_t* outp = out_usm[item];
          const int bits = (scheme[item] == WeightScheme::INT2) ? 2
                         : (scheme[item] == WeightScheme::INT4) ? 4
                         : (scheme[item] == WeightScheme::FP16) ? 16 : 8;
          const int pack = (bits >= 8) ? 1 : 8 / bits;
          const auto* code_u8 = static_cast<const std::uint8_t*>(code_usm[item]);
          const std::size_t act_off = (std::size_t)m * K;
          const std::uint8_t* wrow = code_u8 + (std::size_t)n * K;
          // *** SCALE IS PER ROW, AND THE GROUP WIDTH IS PER ITEM. ***
          // shard_weight emits scale[(row * n_groups + g)] (engine.cpp:306-316),
          // so the base kernel reads `scale_usm + ng * num_groups_w`
          // (quant_gemm.cpp:886) and loops g over THAT ROW's n_groups. The
          // group width is per item (t.qcfg.group_size), so derive both from
          // this item's own K and group width -- the wave contract does NOT
          // require them to be equal across items, and requiring it made the
          // engine refuse perfectly fusible waves.
          const int Gw_i = (group_arr[item] == 0) ? K : static_cast<int>(group_arr[item]);
          const int ngroups_i = (K + Gw_i - 1) / Gw_i;
          const std::uint16_t* sptr_row = sptr + (std::size_t)n * ngroups_i;

          float acc = 0.0f;
          for (int g = 0; g < ngroups_i; ++g) {
            const float scale = from_half_bits(sptr_row[g]) * alpha_f;
            const int kg0 = g * Gw_i;
            int kg1 = kg0 + Gw_i;
            if (kg1 > K) kg1 = K;
            if (bits >= 16) {
              for (int kg = kg0; kg < kg1; ++kg) {
                const std::uint16_t hv = code_u8[((std::size_t)n * K + kg) * 2] |
                    (static_cast<std::uint16_t>(code_u8[((std::size_t)n * K + kg) * 2 + 1]) << 8);
                acc += from_half_bits(act[act_off + kg]) *
                       from_half_bits(hv);
              }
            } else {
              for (int kg = kg0; kg < kg1; ++kg) {
                const float av = from_half_bits(act[act_off + kg]);
                const std::size_t idx = (std::size_t)n * K + kg;
                const std::uint8_t byte = wrow[idx / pack];
                const int within = static_cast<int>(idx % pack);
                const int code = (byte >> (within * bits)) & ((1 << bits) - 1);
                // Bit-exactness vs the base tiled GEMM (quant_gemm_forward_usm,
                // weight loop ~line 321): that path pre-rounds the scaled weight
                // to FP16 via `to_half_bits(s * canon_value(...))` before the
                // multiply, so the fused kernel MUST round the same way.  For
                // INT2/INT4/INT8 the canon_value divisors (3/15/255) are not
                // powers of two, so `av * scale * cv` held in FP32 diverges per
                // term from `av * (half)(scale*cv)`.  Gw_i == Gw (both from the
                // item's qcfg.group_size), the kg summation order is increasing
                // in both, and sptr_row[g] indexes the same scale[row,group]
                // buffer -- so this now reproduces the base's per-element
                // rounding exactly.  The FP16-weight branch above needs no
                // change: canon_value is the identity there and both paths
                // expand raw half bits.
                const float wval = scale * canon_value(bits, code);
                const float wv = from_half_bits(to_half_bits(wval));
                acc += av * wv;
              }
            }
          }
          outp[(std::size_t)m * Ni + n] = to_half_bits(acc);
        });
  });
}

// FP16 x FP16 GEMM (non-quantized path for lm_head etc.).
// C[M,N] = A[M,K] * W[N,K]^T. All row-major. FP16 inputs, FP32 accum,
// FP16 output. The compiler lowers this onto XMX FP16 matrix engines on Arc.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_f16(sycl::queue& q,
                    const sycl::half* a,  // M*K row-major
                    const sycl::half* w,  // N*K row-major
                    sycl::half* c,        // M*N row-major
                    uint32_t M, uint32_t N, uint32_t K) {
  // 1.4 oneMKL SYCL BLAS fallback for large contiguous FP16. Default OFF
  // (INTFER_ONEMKL=1) and only when the library is actually linked, so the
  // default build is byte-for-byte the tiled kernel below. The gate is M >=
  // 8: at M=1 (decode) FP16 already runs at 98-123% of the memory roofline,
  // so a BLAS call cannot pay there -- this is a prefill/large-M item.
  //
  // The arm is NOT bit-exact (oneMKL's FP32 accumulation order is its own),
  // so a stream comparison here is the CONTRACT, not a check: the gate
  // compares against base under an agreed tolerance, and this line is what
  // makes the run's having reached here observable. Without it a flag that
  // never reached the kernel would pass a tolerance-gated comparison just as
  // easily as one that did.
  if (onemkl_available() &&
      (std::getenv("INTFER_ONEMKL") != nullptr) &&
      M >= 8) {
    static bool announced_onemkl = false;
    if (!announced_onemkl) {
      announced_onemkl = true;
      std::fprintf(stderr,
                   "[onemkl] FP16 gemm via oneMKL SYCL BLAS ACTIVE "
                   "(M=%llu N=%llu K=%llu -- FP32 accumulation order is an "
                   "MKL internal, tolerance-gated not bit-exact)\n",
                   (unsigned long long)M, (unsigned long long)N,
                   (unsigned long long)K);
      std::fflush(stderr);
    }
    return gemm_onemkl_fp16(q, c, a, w, M, N, K, 1.0f, 0.0f,
                           sycl::event{});
  }
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
  // INTFER_DPAS48_WG: the shipped nWG=16 is hardcoded, not derived from the
  // device. Each output element (m,n) is written by exactly one block (t=n/16,
  // mg=m/R), so this is a pure partition -- a wider group reschedules WIs without
  // changing who writes what, which is why it is bit-exact and needs no
  // numerics contract. The B580 has 160 compute units; the prefill lever is to
  // widen the group to fill them (PLAN item 1.3). 0 => 16 (shipped).
  const uint32_t nWG = (p.wg == 0) ? 16u : p.wg;
  const uint32_t nMG = (M + R - 1) / R;
  const uint32_t n_blocks = nMG * nT;
  const uint32_t global = ((n_blocks + nWG - 1) / nWG) * nWG;

  // INTFER_TILE_SCHED (PLAN.md 1.3, default OFF): a per-device SLM work-
  // stealing claim queue. The fixed-assignment launch above maps block -> WI
  // by block % nWG, so when nWG < n_blocks the trailing WIs idle while
  // earlier blocks are still running, and the submit cost (~600 us/launch at
  // M=128) is paid per launch regardless. With the queue enabled a WI that
  // has drained its block pulls the next unclaimed one instead of idling:
  // fewer, better-filled launches, which is the prefill lever.
  //
  // It is a pure DISPATCH change, never an arithmetic one: each block still
  // computes the same dot products and accumulates in the same order
  // (row-major over the tile's K, ascending), so the tiled kernel output is
  // bit-identical to the fixed-assignment path and no numerics contract is
  // needed. The counter is one uint32_t per block (4 B * n_blocks -- a few KB
  // at 27B prefill, well inside the 192 KB/Xe-core SLM budget), and the
  // atomic is the ESIMD slm_atomic_update, the primitive the ESIMD context
  // supports (sycl::atomic_ref is not permitted in a SYCL_ESIMD_KERNEL).
  //
  // The queue is per-DEVICE: device d owns only its slice of N, so the
  // counter lives in that device's SLM and this is TP-compatible.
  //
  // The claim shape is the vllm/sgl TileScheduler shape: one atomic per
  // claimed block. A WI loops, claiming the next free block, and stops when
  // the counter reaches n_blocks; every block is processed by exactly the WI
  // that claimed it, so the arithmetic is unchanged.
  const bool tile_sched = (std::getenv("INTFER_TILE_SCHED") != nullptr);

  return q.submit([&](sycl::handler& h) {
    if (dep != sycl::event{}) h.depends_on(dep);
    // 4 B SLM per block, zero-initialized by SYCL. Declared unconditionally
    // so the capture set is the same for both paths; when the flag is off
    // it is never touched and costs nothing.
    sycl::local_accessor<std::uint32_t, 1> ctr(sycl::range<1>(n_blocks), h);
    h.parallel_for<class gemm_dpas48_tiled_kernel<R>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(nWG)),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const uint32_t wid = it.get_global_id(0);
          // Per-block body, shared by the work-stealing and fixed-assignment
          // paths. Kept as a device lambda so the two paths cannot drift:
          // the arithmetic here is the ONLY copy, and the dispatch change is
          // confined to WHICH block each WI runs, never HOW.
          auto process_block = [&](uint32_t block) __attribute__((sycl_explicit_simd)) {
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
        };  // process_block

        // ---- DISPATCH: work-stealing vs fixed-assignment. ----
        //
        // Both paths call the SAME process_block, so the arithmetic is
        // identical and the only difference is WHICH block each WI runs.
        // The fixed-assignment path (tile_sched off) is the shipped kernel:
        // WI i does block i, and blocks >= nWG are simply not launched.
        // The work-stealing path (tile_sched on) has every WI claim the
        // next free block via the SLM counter, so a WI that finishes early
        // pulls the next unfinished one instead of idling.
        //
        // The claim is an atomic inc returning the OLD value: a block is
        // claimed by the WI that sees 0 before the inc. ESIMD's
        // slm_atomic_update<atomic_op::inc> is the supported primitive
        // (sycl::atomic_ref is not permitted in a SYCL_ESIMD_KERNEL), and
        // it returns the pre-inc value, which is exactly what the claim
        // needs. The counter is one uint32_t per block, zero-initialized
        // by SYCL, so "old == block" means "unclaimed".
        if (!tile_sched) {
          if (wid < n_blocks) process_block(wid);
        } else {
          using namespace sycl::ext::intel::esimd;
          uint32_t cur = wid;
          while (cur < n_blocks) {
            // One uint32_t per block, zero-initialized by SYCL. The claim
            // is an atomic inc returning the OLD value: a block is claimed
            // by the WI that sees its own index before the inc. The
            // local_accessor form is the one ESIMD supports (sycl::atomic_ref
            // is not permitted in a SYCL_ESIMD_KERNEL). This call lives in
            // the ESIMD kernel body, NOT in process_block: atomic_update
            // carries the sycl_explicit_simd attribute and a plain helper
            // cannot call it.
            simd<uint32_t, 1> old =
                atomic_update<atomic_op::inc, uint32_t, 1>(
                    ctr, simd<uint32_t, 1>(cur * sizeof(uint32_t)));
            if (old[0] != cur) break;  // claimed by another WI
            process_block(cur);
            cur += nWG;
          }
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

  const uint32_t nWG = (p.wg == 0) ? 16u : p.wg;
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
  // INTFER_SSM_FUSE (default OFF, PRESENCE enables): fold the four kd-serial
  // passes over S into two by consuming each updated element in the same
  // d-iteration that produced it. Bit-exact vs the 4-pass base (the decay /
  // update touch only S[d,c] before the dot reads it; d ascending in both),
  // and TP-agnostic (device 0 regardless of --tp) -- the one fusion the
  // PARAMOUNT DIRECTIVE's TP=2 target can actually reach, unlike the
  // host<->device seams which all open with `num_devices_ != 1 => false`.
  const bool fuse = (std::getenv("INTFER_SSM_FUSE") != nullptr);
  if (fuse) ssm_fuse_announce("ssm-fuse");
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
          float kv = 0.0f;
          float s = 0.0f;
          if (fuse) {
            // 1+2) decay S[d,c], then read the decayed value into the kv
            // dot. The store and the read are the same value (the decay
            // touches only S[d,c], never S[d',c]); d ascending in both
            // forms, so the FP result is bit-identical to the 4-pass base
            // and the token stream cannot move.
            for (uint32_t d = 0; d < kd; ++d) {
              S[(size_t)d * vd + c] *= g_t;
              kv += S[(size_t)d * vd + c] * kn[d];
            }
            const float delta = (vv[c] - kv) * beta;
            // 3+4) rank-1 update S[d,c], then read the updated value into
            // the readout -- same argument, same accumulation order.
            for (uint32_t d = 0; d < kd; ++d) {
              S[(size_t)d * vd + c] += kn[d] * delta;
              s += S[(size_t)d * vd + c] * qn[d];
            }
          } else {
            // 1) decay this head's state column block: S[d,c] *= g_t.
            for (uint32_t d = 0; d < kd; ++d) S[(size_t)d * vd + c] *= g_t;
            // 2) kv[c] = sum_d S[d,c]*kn[d]; delta = (v-kv)*beta.
            for (uint32_t d = 0; d < kd; ++d) kv += S[(size_t)d * vd + c] * kn[d];
            const float delta = (vv[c] - kv) * beta;
            // 3) rank-1 update: S[d,c] += kn[d]*delta.
            for (uint32_t d = 0; d < kd; ++d)
              S[(size_t)d * vd + c] += kn[d] * delta;
            // 4) readout: core = sum_d S[d,c]*qn[d].
            for (uint32_t d = 0; d < kd; ++d) s += S[(size_t)d * vd + c] * qn[d];
          }
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
  // INTFER_SSM_FUSE: same 2-pass fold as ssm_forward. The chained kernel's
  // per-(head,col) loop over m is unchanged; only the inner d-loop changes.
  const bool fuse = (std::getenv("INTFER_SSM_FUSE") != nullptr);
  if (fuse) ssm_fuse_announce("ssm-fuse");
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
            float kv = 0.0f;
            float s = 0.0f;
            if (fuse) {
              for (uint32_t d = 0; d < kd; ++d) {
                S[(size_t)d * vd + c] *= g_t;
                kv += S[(size_t)d * vd + c] * kn[d];
              }
              const float delta = (vv - kv) * beta;
              for (uint32_t d = 0; d < kd; ++d) {
                S[(size_t)d * vd + c] += kn[d] * delta;
                s += S[(size_t)d * vd + c] * qn[d];
              }
            } else {
              for (uint32_t d = 0; d < kd; ++d) S[(size_t)d * vd + c] *= g_t;
              for (uint32_t d = 0; d < kd; ++d)
                kv += S[(size_t)d * vd + c] * kn[d];
              const float delta = (vv - kv) * beta;
              for (uint32_t d = 0; d < kd; ++d)
                S[(size_t)d * vd + c] += kn[d] * delta;
              for (uint32_t d = 0; d < kd; ++d)
                s += S[(size_t)d * vd + c] * qn[d];
            }
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
    if (half & 0x400u) return (std::uint16_t)(sign | 0x0400u);
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
      x = (sign << 31) | ((127 - 15 + e) << 23) | (m2 << 13);
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
  // Bit-exact dispatch knob (INTFER_LA_CONV_PREP_WG). The shipped body is a
  // single_task: ONE work-item runs every section serially (conv over
  // conv_dim, l2norm over 2*nk, the hv/hb copies, the g loop over nv). Each
  // section is a PURE MAP over an independent index -- each c touches only
  // cst[c*3..]/cw[c*ck..], each (kh,is_q) reduces only its OWN kd slice (the
  // inner kd loop stays sequential inside one work-item, so sqrt/sum order is
  // untouched), hv/hb are copies, each hh is independent -- so partitioning
  // those index spaces across work-items writes every element with exactly one
  // work-item and the SAME arithmetic in the SAME order. No cross-WI
  // reduction, no shared local_accessor: bit-exact by construction, the same
  // class as INTFER_DECODE1_WG / INTFER_XMX_WG / INTFER_DPAS48_WG (dispatch
  // partition, no numerics contract). The upstream analogue is
  // ggml-sycl/norm.cpp's sub-group-per-row geometry; its reduce_over_group
  // butterfly is deliberately NOT taken (it would reorder the kd sum and break
  // bit-exactness) -- the GEOMETRY ports, the reduction does not.
  //
  // The per-element bodies are shared lambdas so the serial and parallel
  // launches compute them byte-for-byte; the ONLY difference between arms is
  // the launch geometry, which is what makes the stream-identity claim
  // provable.
  //
  // *** THE CONV SECTION IS AN ORDERING BARRIER, NOT A SECTION. *** conv
  // writes packed[(2*qk+zvw+2*nv) + c] -- which is the `cout` region the
  // l2norm and hv sections READ. The shipped serial kernel works only because
  // section A finishes before section B starts. A single flat parallel launch
  // would race them, so the parallel arm submits TWO ordered launches (conv,
  // then tail chained off conv's event). Two launches still replaces one
  // work-item running conv_dim+nk+zvw+2*nv iterations with a partitioned map.
  const size_t qk = (size_t)nk * kd;
  const float* cout = packed + (2 * qk + zvw + 2 * (size_t)nv);
  // Section A: conv, one WI per channel c in [0, cdim).
  auto emit_conv = [=](std::uint32_t c) {
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
    cst[(size_t)c * 3 + 0] = cst[(size_t)c * 3 + 1];
    cst[(size_t)c * 3 + 1] = cst[(size_t)c * 3 + 2];
    cst[(size_t)c * 3 + 2] = mqkv[c];
    packed[(2 * qk + zvw + 2 * (size_t)nv) + c] = silu;
  };
  // The tail after conv: l2norm [0,2*nk), hv copy [0,zvw), hb copy [0,nv),
  // g [0,nv). One WI per flat index j over the concatenation.
  const std::uint32_t tail_total =
      (std::uint32_t)(2 * nk + zvw + 2 * (size_t)nv);
  auto emit_tail = [=](std::uint32_t j) {
    // l2norm: even => query, odd => key (matches the serial per-head order,
    // whose kd sum stays sequential inside one WI).
    if (j < 2 * nk) {
      const std::uint32_t kh = j >> 1;
      const bool is_q = ((j & 1u) == 0u);
      const float* src = (is_q ? cout : cout + qk) + (size_t)kh * kd;
      float* dst = (is_q ? packed : packed + qk) + (size_t)kh * kd;
      float ss = 0.0f;
      for (std::uint32_t d = 0; d < kd; ++d) ss += src[d] * src[d];
      const float inv = 1.0f / sycl::sqrt(ss + 1e-6f);
      if (is_q) {
        for (std::uint32_t d = 0; d < kd; ++d)
          dst[d] = src[d] * inv * qscale;
      } else {
        for (std::uint32_t d = 0; d < kd; ++d) dst[d] = src[d] * inv;
      }
      return;
    }
    j -= 2 * nk;
    if ((size_t)j < zvw) {
      float* hv = packed + 2 * qk;
      hv[j] = cout[2 * qk + j];
      return;
    }
    j -= (std::uint32_t)zvw;
    // Layout mirrors the host reference (main.cpp:4864-4866):
    //   hqn [0,nk*kd), hkn [nk*kd,2nk*kd), hv [2nk*kd, 2nk*kd+zvw),
    //   hg [2nk*kd+zvw, ...+nv), hb [..+nv, ..+2nv).
    // so hg sits BEFORE hb (hg = hv+zvw, hb = hg+nv).
    if (j < nv) {
      float* hg = packed + 2 * qk + zvw;
      const std::uint32_t hh = j;
      const float sp = sycl::log1p(sycl::exp(arow[hh] + dtb[hh]));
      hg[hh] = -sycl::exp(Alog[hh]) * sp;
      return;
    }
    j -= nv;
    float* hb = packed + 2 * qk + zvw + nv;
    hb[j] = brow[j];
  };
  const std::uint32_t wg_env = []() -> std::uint32_t {
    const char* e = std::getenv("INTFER_LA_CONV_PREP_WG");
    if (!e) return 0;
    int v = std::atoi(e);
    return v > 0 ? (std::uint32_t)v : 0;
  }();
  if (wg_env == 0) {
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep);
      h.single_task<class la_conv_prep_kernel>([=]() {
        for (std::uint32_t c = 0; c < cdim; ++c) emit_conv(c);
        for (std::uint32_t j = 0; j < tail_total; ++j) emit_tail(j);
      });
    });
  }
  // Parallel arm: the IDENTICAL per-element bodies, partitioned over
  // work-items, with conv and tail as two ordered launches (the cout region
  // makes conv a producer the tail consumes). The announcement is gated on the
  // env: the base arm must stay silent (a marker on base is an arm that ran
  // when none was selected -- the test_rowpair_equiv.sh contract).
  {
    static bool announced = false;
    if (!announced) {
      announced = true;
      std::fprintf(stderr,
                   "[la-conv-prep-wg] work-group=%u "
                   "(INTFER_LA_CONV_PREP_WG=%s) ACTIVE "
                   "(conv_dim=%u nk=%u nv=%u tail=%u)\n",
                   wg_env, std::getenv("INTFER_LA_CONV_PREP_WG"), cdim, nk, nv,
                   tail_total);
      std::fflush(stderr);
    }
  }
  const std::uint32_t wg = wg_env;
  const std::uint32_t gconv = ((cdim + wg - 1) / wg) * wg;
  const std::uint32_t gtail = ((tail_total + wg - 1) / wg) * wg;
  sycl::event e_conv = q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for<class la_conv_prep_kernel_par_conv>(
        sycl::nd_range<1>(sycl::range<1>(gconv), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
          const std::uint32_t i = (std::uint32_t)it.get_global_id(0);
          if (i >= cdim) return;
          emit_conv(i);
        });
  });
  return q.submit([&](sycl::handler& h) {
    h.depends_on(e_conv);
    h.parallel_for<class la_conv_prep_kernel_par_tail>(
        sycl::nd_range<1>(sycl::range<1>(gtail), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
          const std::uint32_t j = (std::uint32_t)it.get_global_id(0);
          if (j >= tail_total) return;
          emit_tail(j);
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
