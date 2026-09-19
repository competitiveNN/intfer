// include/intfer/gemm/quant_gemm.hpp
//
// Low-level SYCL quantized GEMM kernels for Intel Arc (Xe-core / XMX).
//
// This is the Stage-2 compute primitive that consumes the quantized weight
// payload produced by the Stage-1 DASH-Q / RCO pipeline (docs/quant_spec.md):
//   * Packed low-bit codes  (INT2 = 4 codes/byte, INT4 = 2/byte, INT8 = 1/byte)
//   * Per-group FP16 symmetric scales (the exact scale_buf layout from
//     TensorIR / ModelIR: scale[(row * num_groups + g)]).
//   * Optional per-tensor FP32 global divisor alpha (global_buf).
//
// Two execution paths are provided:
//
//   (1) quant_gemm_forward  - dequant-on-the-fly GEMM. Weights are dequantized
//       inside the kernel (in registers / SLM) and the inner product is
//       accumulated in FP32 via FP16 multiply-accumulate, which the compiler
//       lowers onto the XMX matrix engines for FP16 on Arc. This is the
//       general, bit-flexible path (INT2/INT4/INT8 weights, FP16 or groupwise
//       INT8 activations) and is always available.
//
//   (2) quant_gemm_dpas_int8 - a hand-written ESIMD kernel that issues the
//       Intel XMX DPAS (Dot Product Accumulate Systolic) instruction directly
//       for the INT8 weight x INT8 activation case (per-tensor symmetric
//       scales). This is the literal "DPAS" low-level path and is selected by
//       quant_gemm_forward(use_dpas=true) when both operands are INT8 and the
//       shapes are DPAS-friendly; otherwise (1) is used.
//
// Layout convention (matches the rest of intfer):
//   C[M, N] = A[M, K] * W[N, K]^T
//   i.e. C[m, n] = sum_k  A[m, k] * W[n, k].
// All matrices are row-major. C is stored as FP16 (binary16) bits.

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

namespace intfer::gemm {

// Quantized weight precision (mirrors quant_spec.h QuantScheme, renamed to
// avoid clashing with the Stage-1 enum in the same translation unit).
enum class WeightScheme : uint8_t { FP16 = 0, INT8 = 1, INT4 = 2, INT2 = 3 };

// Activation precision fed to the GEMM.
enum class ActScheme : uint8_t { FP16 = 0, INT8 = 1 };

// ---------------------------------------------------------------------------
// Buffer-based description of the (already quantized) weight tensor.
// ---------------------------------------------------------------------------
struct GemmWeightBuffers {
  WeightScheme scheme = WeightScheme::INT8;
  uint16_t group_size = 0;        // K-group for scales; 0 = per-tensor
  bool has_global_divisor = false;// per-tensor FP32 alpha present
  uint32_t N = 0;                 // output rows (channels)
  uint32_t K = 0;                 // contraction dim

  sycl::buffer<std::uint8_t>  code;   // packed codes (INT2:4/byte, INT4:2/byte,
                                      // INT8:1/byte, FP16:2 bytes/el)
  sycl::buffer<std::uint16_t> scale;  // FP16 per (row, group)
  sycl::buffer<float>         global; // FP32 per-tensor alpha (may be empty)

  // Backing storage. The buffer(ptr, range) / buffer(vector) constructors only
  // *reference* the payload, so the owning vector must outlive the buffer. We
  // move the packed vectors here so they stay alive for as long as `w` is used.
  std::vector<std::uint8_t>  code_store;
  std::vector<std::uint16_t> scale_store;
  std::vector<float>          global_store;  // owns the per-tensor alpha copy

  GemmWeightBuffers()
      : code(sycl::range<1>(1)), scale(sycl::range<1>(1)),
        global(sycl::range<1>(1)) {}
};

// ---------------------------------------------------------------------------
// Buffer-based description of the activation tensor.
// ---------------------------------------------------------------------------
struct GemmActBuffers {
  ActScheme scheme = ActScheme::FP16;
  uint16_t group_size = 0;        // K-group for INT8 activation scales; 0 = per-tensor
  uint32_t M = 0;
  uint32_t K = 0;

  sycl::buffer<std::uint16_t> fp16;  // M*K half  (used when scheme == FP16)
  sycl::buffer<std::int8_t>   int8;  // M*K int8  (used when scheme == INT8)
  sycl::buffer<std::uint16_t> scale; // FP16 per (row, group) for INT8 act

  GemmActBuffers()
      : fp16(sycl::range<1>(1)), int8(sycl::range<1>(1)),
        scale(sycl::range<1>(1)) {}
};

struct QuantGemmOptions {
  bool use_dpas = false;          // request the ESIMD DPAS INT8 path when legal
};

// Host FP32 reference: C[M,N] = A[M,K] * W[N,K]^T.
void gemm_reference_f32(const std::vector<float>& A,
                        const std::vector<float>& W,
                        std::vector<float>& C,
                        uint32_t M, uint32_t N, uint32_t K);

// Dequant-on-the-fly (and optionally DPAS) quantized GEMM.
//   c : output buffer, M*N FP16 bits.
// The DPAS path is taken only when opt.use_dpas && weight==INT8 && act==INT8
// and both K dimensions are multiples of the DPAS systolic depth (32);
// otherwise the portable dequant kernel runs.
void quant_gemm_forward(sycl::queue& q,
                        GemmActBuffers& a,
                        GemmWeightBuffers& w,
                        sycl::buffer<std::uint16_t>& c,
                        const QuantGemmOptions& opt = {});

// ---------------------------------------------------------------------------
// ESIMD DPAS INT8 x INT8 GEMM (raw USM pointers, per-tensor symmetric scales).
// Returns a SYCL event that signals when the kernel has completed.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_dpas_int8(sycl::queue& q,
                          const std::int8_t* a_int8,
                          float a_scale,
                          const std::int8_t* w_vnni,
                          const float* w_scale,
                          std::uint16_t* c,
                          uint32_t M, uint32_t N, uint32_t K);

// Build the VNNI (K x N) transpose of an INT8 weight matrix on the host.
// out[k*N + n] = w[n*K + k].
void build_vnni_int8(const std::int8_t* w, std::int8_t* out,
                     uint32_t N, uint32_t K);

// ---------------------------------------------------------------------------
// USM-based dequant-on-the-fly GEMM entry point.
//
// This is the "fast path" for the per-token decode loop: weights are
// pre-cached in device USM by Engine::ensure_shard_built, and this function
// accepts raw USM pointers so the kernel can read directly from device memory
// without the per-call SYCL buffer construction overhead.
//
// The activation buffer (size M*K FP16) and output buffer (size M*N FP16
// bits) are also pre-allocated in USM by the cache; this function copies the
// host activation into the act USM buffer, launches the kernel, and returns
// a SYCL event that the caller can wait on before reading out[0:N].
//
// Supported precisions: INT2 / INT4 / INT8 / FP16 weights with FP16
// activations. The kernel dequantizes weights in-register and accumulates in
// FP32, which the compiler lowers onto the XMX matrix engines for FP16 MMA
// on Intel Arc.
//
// For M=1 (decode), a dedicated flat 1D kernel is used (no tiling, no
// local memory, no barriers) to avoid the overhead of the tiled kernel.
// For M>1, a tiled 2D kernel with shared local memory is used.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_forward_usm(sycl::queue& q,
                                      // Activation: M*K FP16 bits (pre-allocated, updated every call)
                                      std::uint16_t* act_usm,
                                      const void* code_usm,    // packed codes
                                      const std::uint16_t* scale_usm, // FP16 per (row, group)
                                      float alpha,             // global divisor (1.0 if no global)
                                      std::uint16_t* out_usm,  // M*N FP16 bits output
                                      uint32_t M, uint32_t N, uint32_t K,
                                      uint32_t group_size,     // 0 = per-tensor
                                      gemm::WeightScheme scheme,
                                      const sycl::event& copy_dep);

// FP16 x FP16 GEMM for non-quantized tensors like lm_head.
// C[M,N] = A[M,K] * W[N,K]^T. A is row-major FP16, W is row-major FP16,
// C is row-major FP16. Uses the XMX DPAS path via the portable FP16 kernel.
sycl::event quant_gemm_f16(sycl::queue& q,
                    const sycl::half* a,  // M*K row-major
                    const sycl::half* w,  // N*K row-major
                    sycl::half* c,        // M*N row-major
                    uint32_t M, uint32_t N, uint32_t K);

// FP16 x INT8 GEMM for the VRAM-resident lm_head (27B TP=2): dev0 cannot
// hold the 1.27GB FP16 shard next to the preloaded engine (12647 > 12216
// hard), so lm_head uploads as per-row symmetric INT8 (0.64GB/dev) +
// FP16 per-row scales. W_int8[n,k] = round(W[n,k]/scale[n]),
// scale[n] = max|W[n,:]|/127. Same 64x64x64 tiling as quant_gemm_f16;
// dequant is one FMA (scale * int8) in the weight-load path.
sycl::event quant_gemm_f16_int8w(sycl::queue& q,
                    const sycl::half* a,    // M*K row-major FP16
                    const std::int8_t* w,   // N*K row-major INT8 codes
                    const sycl::half* ws,   // N per-row FP16 scales
                    sycl::half* c,          // M*N row-major FP16 out
                    uint32_t M, uint32_t N, uint32_t K);

// ---------------------------------------------------------------------------
// W4A8 / W2A8 native sub-byte DPAS GEMM (TODO 6, M=1 decode).
//
// Exact-weight path for grouped INT4/INT2 tensors: the DASH-Q codebooks are
// affine in the stored index (INT4: (2j-15)/15, INT2: (2j-3)/3), so
//   out = a_s * w_s * (2*T - C*S) / C,  T = DPAS(a_q, idx), S = sum(a_q),
// with C=15 (INT4) or C=3 (INT2). Weights stay bit-exact; the only error
// source vs the FP16-act engine is the per-tensor/per-GA activation quant
// to INT8 (argmax-gated, not bit-identical).
//
// Weight layout (built once at shard build by repack_dpas412, per device):
// per (16-N tile t, K-op o) contiguous dwords, vertical pack
//   dword[(k/E)*16+n] bit (k%E)*b, E=8/b... i.e. u4: E=8, u2: E=16.
// Per-op K: 32 (u4/u8: ops/chan 4), 32 (u2xs8: max bits 8 -> 4).
// Validity needs a_s and w_s constant over the op K-range:
//   INT4/G64: op K=32 divides 64 -> 2 ops per w-group. CLEAN (1 DPAS/op).
//   INT8/G32: op K=32 == group. CLEAN (1 DPAS/op).
//   INT2/G32: op K=32 == group. CLEAN (1 DPAS/op).
//   INT2/G16: op = 2 subgroups -> linearity split: T_full + masked T_0
//     (T_1 = T_full - T_0, 2 DPAS/op). GA=64 act scale: fold per op via
//     the op's GA slice (op K=32 divides GA=64 -> 2 slices/op).
// Shapes: N_local % 16 == 0, K % 32 == 0 required (all 27B/0.8B tensors
// satisfy this post-TP-split); otherwise returns a null event and the
// caller must use quant_gemm_forward_usm.
// ---------------------------------------------------------------------------
struct Dpas48Params {
  const void* code_dpas = nullptr;  // repacked vertical B (per-device USM)
  const std::uint16_t* scale_usm = nullptr;  // FP16 per (row, w-group)
  const std::int8_t* act_int8 = nullptr;     // M rows of INT8 acts (device USM)
  const float* act_scale = nullptr;          // per GA-slice FP32 (device USM)
  float alpha = 1.0f;               // global divisor (host, by value)
  std::uint16_t* out_usm = nullptr; // M*N FP16 bits
  uint32_t M = 0, N = 0, K = 0;
  uint32_t group_size = 0;          // weight K-group (64/32/16)
  uint32_t act_group = 64;          // activation quant group
  int bits = 4;                     // 8/4/2
  // Row strides (elements) for the act workspaces. The engine workspaces
  // are grow-only, so a call's K/nGA can be SMALLER than the workspace
  // dims: rows must be addressed by workspace stride, not call shape
  // (M=1 never noticed; M>1 read garbage scales/acts for rows >= 1).
  uint32_t act_stride = 0;          // 0 => K (tight); else workspace K
  uint32_t as_stride = 0;           // 0 => nGA; else workspace nGA
};

// Host-side repack: row-major packed codes (n*K+k indices) -> DPAS vertical
// B tiles. out size = (N/16) * (K/opK) * (opK*16*b/8) bytes; opK=32.
// Returns false when N%16 != 0 or K%32 != 0 (caller falls back).
bool repack_dpas412(const std::uint8_t* row_codes, uint32_t N, uint32_t K,
                    int bits, std::vector<std::uint8_t>& out);

// M=1 W4A8/W2A8 decode kernel. Returns the kernel event.
sycl::event quant_gemm_dpas48(sycl::queue& q, const Dpas48Params& p);

// ---------------------------------------------------------------------------
// FP16-act-equivalent s8-weight DPAS GEMM for the VRAM-resident lm_head.
//
// Same Dpas48Params contract (repacked vertical-B tiles from repack_dpas412
// with bits=8, FP16 per-row weight scales, INT8 acts + FP32 GA-slice scales),
// but the B operand is SIGNED int8 (raw symmetric codes, not DASH-Q codebook
// indices), so no (2*T - C*S)/C fold is needed: the s8 x s8 DPAS integer dot
// is exact and the output is
//   out = act_scale * w_scale * alpha * DPAS(a_q, w_q)
// accumulated in FP32 over K-ops, then rounded once to FP16.
//
// Validity: N % 16 == 0, K % 32 == 0 (caller guarantees via repack success;
// the kernel assumes it). M=1 decode; general M works via M*nT blocks.
// ---------------------------------------------------------------------------
sycl::event quant_gemm_dpas_s8w(sycl::queue& q, const Dpas48Params& p);

// ---------------------------------------------------------------------------
// Device-side Gated-DeltaNet SSM recurrent update (Stage 3 GPU offload).
//
// Offloads the per-token LA-branch recurrence (host scalar loop over
// n_v_heads x k_head_dim x v_head_dim, ~1M FLOP/token/layer on 0.8B) to a
// plain FP32 SYCL kernel with device-resident state. Per head hh (key head
// kh = hh*n_k/n_v grouping when n_v > n_k):
//   g_t = exp(g[hh]); beta = sigmoid(b[hh])
//   S[hh] *= g_t
//   kv = S[hh]^T @ kn; delta = (v - kv) * beta
//   S[hh] += kn outer delta
//   core[hh] = S[hh]^T @ qn
// All FP32, same element order as the host loop (bit-identical output up
// to FMA contraction differences; the oracle token stream is the arbiter).
// State S lives in caller-owned device USM (n_v*kd*vd floats), persistent
// across tokens; the kernel never allocates. qn/kn are L2-normalized +
// q-scaled on the host (cheap O(kd) per head) and uploaded per call with
// the other small per-token inputs; core (n_v*vd) is read back per call.
// M rows (batched requests x positions) share one launch: each row has its
// own S state pointer (rows may belong to different requests or layers),
// passed as a device-pointer array.
struct SsmParams {
  const float* const* qn_rows;  // [M] device pointers to [kd] qn rows
  const float* const* kn_rows;  // [M] device pointers to [kd] kn rows
  const float* const* v_rows;   // [M] device pointers to [vd] value rows
  const float* const* g_rows;   // [M] device pointers to [n_v] g logits
  const float* const* b_rows;   // [M] device pointers to [n_v] b logits
  float* const* s_rows;         // [M] device pointers to [n_v*kd*vd] states
  float* const* core_rows;      // [M] device pointers to [n_v*vd] outputs
  uint32_t M;                   // batch rows
  uint32_t n_v;                 // value heads
  uint32_t n_k;                 // key heads (grouping when n_v > n_k)
  uint32_t kd;                  // k head dim
  uint32_t vd;                  // v head dim
};

// One WI per (row m, head hh, v-col c): kd-serial inner reductions.
// State stays resident; only small per-token vectors transfer.
// Rows MUST have distinct states (decode rows of different requests, or
// single rows); rows sharing one state must use ssm_forward_chain.
sycl::event ssm_forward(sycl::queue& q, const SsmParams& p);

// Chained variant: all M rows share ONE state (single pointer, not an
// array) and run sequentially in row order (prefill-chunk rows of one
// request: the recurrence is order-dependent, so one WI per (head, col)
// loops rows innermost-to-outermost over m). Same math as the host
// per-row loop; the oracle token stream is the arbiter.
struct SsmChainParams {
  const float* qn_rows;  // row m qn at qn_rows + m*row_stride ([n_k*kd])
  const float* kn_rows;  // row m kn at kn_rows + m*row_stride ([n_k*kd])
  const float* v_rows;   // row m v at v_rows + m*row_stride ([n_v*vd])
  const float* g_rows;   // row m g at g_rows + m*row_stride ([n_v])
  const float* b_rows;   // row m b at b_rows + m*row_stride ([n_v])
  float* state;          // [n_v*kd*vd] resident, updated in place
  float* core_rows;      // [M*n_v*vd] outputs (contiguous)
  uint32_t M;
  uint32_t n_v;
  uint32_t n_k;
  uint32_t kd;
  uint32_t vd;
  uint32_t row_stride;   // float stride between consecutive rows in *_rows
};

// One WI per (head hh, v-col c), kd-serial, m-sequential inner loop.
sycl::event ssm_forward_chain(sycl::queue& q, const SsmChainParams& p);

// ---------------------------------------------------------------------------
// Device-side fused silu-mul + FP32->FP16 convert (MLP down-act fusion).
//
// Reads two FP16 gate/up GEMM outputs (mg_fp16[i], mu_fp16[i]), computes
//   fused[i] = silu((float)mg[i]) * (float)mu[i],  silu(x) = x/(1+exp(-x))
// in FP32, and writes round-to-nearest-even FP16 bits to act_fp16[i].
// One WI per element; no cross-element dependencies.
//
// Bit-identity contract: half->float is exact; the FP32 silu/mul uses the
// same element order as the host loop (sycl::exp may differ from host
// expf by ~1ulp on rare elements; the oracle token stream is the
// arbiter — same standing as the AVX2 host-math change). The FP16
// down-convert matches dashq::float_to_half RNE bit-for-bit (verified by
// the CPU parity gate when it passes; oracle gate otherwise).
// dep chains after the gate/up wave kernels (in-order queue also
// observes it; the explicit dep keeps out-of-order queues correct).
// ---------------------------------------------------------------------------
struct SiluMulParams {
  const std::uint16_t* mg_fp16;  // [n] gate GEMM output (device FP16 bits)
  const std::uint16_t* mu_fp16;  // [n] up GEMM output (device FP16 bits)
  std::uint16_t* act_fp16;       // [n] fused down activation (device FP16 bits)
  uint32_t n;                    // element count (M * inter)
};
sycl::event silumul_to_half(sycl::queue& q, const SiluMulParams& p,
                            const sycl::event& dep = {});

// ---------------------------------------------------------------------------
// Device-side fused SSM-norm + silu(z) + FP32->FP16 convert (L4 la_out seam).
//
// Reads one FP32 SSM core row (core[n_v*vd]), the z gate row (z[n_v*vd])
// and the per-dim norm weights (w[vd], shared across heads), computes per
// head hh the same math as the host LA norm loop:
//   v = sum_i core[hh,i]^2; inv = 1/sqrt(v/vd + eps)
//   out[hh,i] = RNE_FP16((core[hh,i]*inv) * w[i] * silu(z[hh,i]))
// One WI per head (vd-serial inner loops, same element order as host).
// silu uses sycl::exp (~1ulp vs host expf on rare elements; the oracle
// token stream is the arbiter — same standing as silumul_to_half). The
// FP16 convert is dev_float_to_half (RNE bit-identical to
// dashq::float_to_half). dep chains after the SSM kernel.
// ---------------------------------------------------------------------------
struct SsmNormParams {
  const float* core_fp32;      // [n_v*vd] SSM core row (device FP32)
  const float* z_fp32;         // [n_v*vd] z gate row (device FP32)
  const float* norm_w_fp32;    // [vd] ssm norm weights (device FP32)
  std::uint16_t* act_fp16;     // [n_v*vd] fused la_out activation (device FP16)
  uint32_t n_v;                // value heads
  uint32_t vd;                 // v head dim
  float rms_eps;
};
sycl::event ssm_norm_silu_to_half(sycl::queue& q, const SsmNormParams& p,
                                  const sycl::event& dep = {});

}  // namespace intfer::gemm
