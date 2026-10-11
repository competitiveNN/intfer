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

                                      // ---------------------------------------------------------------------------
                                      // FUSED GATE+UP GEMM (INTFER_FUSED_GATE_UP). TP=2-only fusion.
                                      // ---------------------------------------------------------------------------
                                      // One launch computes TWO independent GEMMs that share the SAME activation
                                      // vector (the MLP gate and up projections both read `rin`). In the current
                                      // TP=2 MLP path each is a separate wave item, so the host pays 2 submits +
                                      // 2 waits + 2 D2H readbacks per device per token just to run them. This
                                      // fuses them into ONE parallel_for over N = Ng + Nu output elements: the
                                      // activation is loaded once per work-item and two independent FMA chains
                                      // (one per output) run against it, so the work-item count is the UNION of
                                      // the two GEMMs' Ns rather than their sum and the launch count halves.
                                      //
                                      // WHY THIS IS THE ONLY FUSION THAT SURVIVES TP=2. The two existing fusion
                                      // seams (run_mlp_fused engine.cpp:1451, run_linear_tp_devact engine.cpp:1588)
                                      // are TP=1 ONLY, and at the target they are dead code. TP slices the OUTPUT
                                      // dim N, so gate/up produce only device d's slice of `inter` while `down` has
                                      // K = the FULL inter and needs the other device's slice; the intermediate
                                      // must cross the host and there is no P2P. Gate and up are BOTH
                                      // column-parallel over the SAME split, both device-local, and each output's
                                      // K accumulation is untouched, so this is bit-exact -- the one TP=2-valid
                                      // fusion in the tree.
                                      //
                                      // NUMERICS. Bit-exact by construction: each output keeps its own accumulator,
                                      // ascending kg, ascending j, its own group scale, its own wtab. The two
                                      // chains are interleaved in the INSTRUCTION STREAM, never summed together,
                                      // which is the same argument ROWPAIR uses (and ROWPAIR is measured, not
                                      // argued: scripts/test_rowpair_equiv.sh, 227 835 INT2 values bitwise).
                                      // No reassociation, no tolerance contract.
                                      //
                                      // CONTRACT. out_g and out_u are separate buffers (M*Ng and M*Nu FP16 bits);
                                      // the caller assembles them into the host vectors. gate/up MUST share K and
                                      // group_size; Ng and Nu may differ. Returns the event; caller waits once.
                                      sycl::event quant_gemm_forward_usm_gateup(
                                          sycl::queue& q,
                                          std::uint16_t* act_usm,                 // shared M*K FP16 activation
                                          const void* code_g, const std::uint16_t* scale_g, float alpha_g,
                                          std::uint16_t* out_g,                   // M*Ng FP16 output (gate)
                                          const void* code_u, const std::uint16_t* scale_u, float alpha_u,
                                          std::uint16_t* out_u,                   // M*Nu FP16 output (up)
                                          uint32_t M, uint32_t Ng, uint32_t Nu, uint32_t K,
                                          uint32_t group_size,
                                          gemm::WeightScheme scheme_g, gemm::WeightScheme scheme_u,
                                          const sycl::event& copy_dep);

                                      // ---------------------------------------------------------------------------
                                      // FUSED WAVE LAUNCH (INTFER_WAVE_LAUNCH_FUSE).
                                      // ---------------------------------------------------------------------------
                                      // A wave is a set of INDEPENDENT GEMMs (distinct tensors, distinct
                                      // activations, distinct outputs, no data dependency between items)
                                      // that the engine currently runs as N separate parallel_for
                                      // launches per device, each with its own submit + wait + D2H
                                      // readback. This fuses them into ONE parallel_for whose
                                      // global size is the UNION of the items' output elements:
                                      // one WI owns one (item, m, n) triple, so each output's K
                                      // accumulation is untouched and the launch count drops from
                                      // N to 1 per device.
                                      //
                                      // WHY THIS IS GENERAL, NOT ANOTHER TP=1-ONLY FUSION. Unlike
                                      // gate+up (which shares one activation vector, so it is
                                      // column-parallel over the same split and device-local), a
                                      // wave's items have DIFFERENT activations -- but they are all
                                      // FP16-act, all read from the SAME per-device wave slot
                                      // array, and TP slices the OUTPUT dim N per item, so every
                                      // item is still device-local. The activation for item i lives
                                      // in wv_[d*wv_W_ + i].act, its output in wv_[d*wv_W_ + i].out.
                                      // Nothing crosses the host; this is valid at TP=2.
                                      //
                                      // NUMERICS. Bit-exact by construction: each output keeps its
                                      // own accumulator, ascending kg, ascending j, its own group
                                      // scale, its own wtab. The items are interleaved in the
                                      // INSTRUCTION STREAM, never summed together -- the same
                                      // argument the gate+up kernel uses (and gate+up is measured,
                                      // not argued). No reassociation, no tolerance contract.
                                      //
                                      // CONTRACT. `items` carries N entries; every item MUST share
                                      // the same M, K and group_size (the wave workspace is sized
                                      // to the max, but the accumulation order below is per-item
                                      // and uses each item's own K and group_size, so differing K and
                                      // group_size are safe in principle -- the engine only calls this when
                                      // every item is non-INT8, which is the one scheme the fused kernel does
                                      // not support). Returns the event; the caller waits once and readbacks
                                      // once.
                                      sycl::event quant_gemm_forward_usm_wave(
                                          sycl::queue& q,
                                          const std::uint16_t* const* act_usm,  // N pointers, M*K FP16 each
                                          const void* const* code_usm,          // N pointers, packed codes
                                          const std::uint16_t* const* scale_usm,// N pointers, FP16 per (row, group)
                                          const float* alpha,                   // N per-tensor divisors
                                          std::uint16_t* const* out_usm,        // N pointers, M*N FP16 output
                                          uint32_t N,                           // number of items
                                          const uint32_t* M_arr, const uint32_t* N_arr,
                                          const uint32_t* K_arr, const uint32_t* group_arr,
                                          const gemm::WeightScheme* scheme,
                                          const sycl::event& copy_dep);

                                      // ---------------------------------------------------------------------------
                                      // FUSED INDEPENDENT-GEMM LAUNCH (INTFER_FUSED_DECODE1) -- REMOVED 2026-10-04.
                                      // ---------------------------------------------------------------------------
// THERE IS NO SUCH API HERE, AND THE HEADER PREVIOUSLY ADVERTISED ONE. The
// fused decode launch shipped in 0961035 and was reverted in 776ef33 because it
// broke 27B decode catastrophically (1284 ms -> 231,114 ms per token), isolated
// to EXTRACTING the M=1 INT2 body into a device function. The revert restored
// src/gemm/quant_gemm.cpp and src/engine/engine.cpp but NOT this header, so
// FusedDecode1Set, kFusedDecode1MaxSets, fused_decode1_eligible() and
// quant_gemm_fused_decode1() were left declared here with a 22-line comment
// asserting a bit-exactness argument -- an API with no definition anywhere.
//
// MEASURED, not inferred: the block below used to sit here and was deleted in
// this commit. `grep -rn 'FusedDecode1Set' src/ include/ examples/` returned 6
// hits, ALL in this header and NONE in any .cpp -- an unimplemented public API.
// A stale document is recoverable; a header that advertises a kernel nobody can
// link is not, and this file is exactly what a reader trusts over prose.
//
// WHY THE BLOCK IS REPLACED BY THIS NOTE RATHER THAN DELETED SILENTLY: the
// revert's real lesson -- that moving arithmetic into a device function is NOT
// a safe refactor on this toolchain -- lives in AGENTS.md and in the batch
// report, and this is the one place a future implementer of the same idea will
// look. If the fusion is ever re-attempted, it must go back to the SEAMS the
// revert established (keep each GEMM's arithmetic INLINE at its own call site;
// fuse by sharing a launch, not by extracting a body), and it must be gated
// again -- and it must NOT be declared here until a definition exists.
//
// The perf context that motivated it still stands and is not in question: M=1
// decode was EU-parallelism-bound, because this kernel's total work-item count is
// SUPERSEDED 2026-10-05: that premise rested on the A770's 512 Xe cores. The B580
// has 20 Xe cores / ~2560 FP32 lanes and 456 GB/s, so decode is at ~9 % of the
// memory roofline rather than ~21 %, and the GEMV is not lane-starved. The
// MR/SPLITK measurements stand; the model does not. AGENTS.md, SUPERSEDED banner.
// exactly N and that is invariant to the work-group shape. Splitting K across
// work-items would raise parallelism but reassociates the FP32 sum and needs a
// numerics contract agreed first; fusing INDEPENDENT GEMMs keeps each output's
// K order untouched and is the bit-exact direction.

// FP16 x FP16 GEMM for non-quantized tensors like lm_head.
// C[M,N] = A[M,K] * W[N,K]^T. A is row-major FP16, W is row-major FP16,
// C is row-major FP16. Uses the XMX DPAS path via the portable FP16 kernel.
sycl::event quant_gemm_f16(sycl::queue& q,
                    const sycl::half* a,  // M*K row-major
                    const sycl::half* w,  // N*K row-major
                    sycl::half* c,        // M*N row-major
                    uint32_t M, uint32_t N, uint32_t K);

// ---------------------------------------------------------------------------
// 1.4 oneMKL SYCL BLAS fallback for large contiguous FP16 GEMM.
//
// Built ONLY when INTFER_ENABLE_MKL=ON and libmkl_sycl_blas is on the linker
// path. When the library is absent this file compiles the empty-fallback
// spellings below, so the engine links either way and the arm is
// unreachable rather than wrong.
//
// THE NUMERICS CONTRACT, first, because it is the whole risk of linking a
// BLAS. oneMKL's FP32 accumulation order is an internal implementation detail
// of the mkl::blas kernel and is NOT the engine's row-major ascending-k
// order. So a bit-exactness guarantee CANNOT be made for this arm, and none
// is claimed. PLAN.md records why: "the repo has a measured matmul-precision
// divergence history, so the contract comes first." The arm therefore runs
// only on explicit opt-in (INTFER_ONEMKL=1), and any gate compares the token
// stream against base under an AGREED TOLERANCE -- never bitwise.
//
// WHERE IT APPLIES: large-M contiguous FP16 only. On this box FP16 already
// runs at 98-123% of the memory roofline on decode, so a BLAS call cannot pay
// at M=1 -- it is a prefill/large-M item.
// ---------------------------------------------------------------------------
bool onemkl_available();

sycl::event gemm_onemkl_fp16(sycl::queue& q, sycl::half* C,
                            const sycl::half* A, const sycl::half* B,
                            std::uint64_t M, std::uint64_t N,
                            std::uint64_t K, float alpha, float beta,
                            sycl::event dep);

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
  uint32_t act_stride = 0;  // 0 => K (tight); else workspace K
  uint32_t as_stride = 0;   // 0 => nGA; else workspace nGA
  // INTFER_DPAS48_FOLD=1: precomputed per-(row,op) activation column sums
  // from quant_gemm_dpas48_rowfold. nullptr (default) => the kernel recomputes
  // them inline, which is what makes it redo the same O(K) integer fold once
  // per 16-N tile. Bit-identical either way (exact int32 sums).
  const std::int32_t* fold_usm = nullptr;
  // Fold workspace row stride in INT32 elements. MUST be >= 2*nOp, because a
  // row owns [S0][Sfull]. 0 means "tight", which fold_row_stride resolves to
  // exactly 2*nOp. Do NOT resolve or validate this by hand at a call site --
  // go through fold_readable(), which is the same rule the writer uses.
  uint32_t fold_stride = 0;
  // INTFER_DPAS48_WG: work-group width for the tiled DPAS kernel. 0 (default)
  // => 16, the shipped value, hardcoded in quant_gemm.cpp. The kernel is one
  // WI per (row-group of R, 16-N tile), and each output element (m,n) is written
  // by exactly ONE block (t = n/16, mg = m/R), so the mapping is a pure
  // partition: changing nWG changes only which WI schedules which block, never
  // who writes what. Bit-exact by construction, same as the lm_head tile split.
  // The B580 has 160 compute units against the hardcoded 16; a wider group is
  // the prefill lever (item 1.3) and needs no numerics contract.
  uint32_t wg = 0;
};

// INTFER_DPAS48_FOLD layout rule -- ONE definition, consumed by the writer
// (quant_gemm_dpas48_rowfold) and by BOTH readers (quant_gemm_dpas48 and
// quant_gemm_dpas4_row).
//
// Each row owns 2*nOp ints, [S0][Sfull]. A stride below 2*nOp lets row m's
// Sfull block overwrite row m+1's S0 block, which is SILENT at M=1 -- the
// defect this hoist already shipped and then caught. The writer refuses such
// a stride, so a reader that resolved the stride any other way would index a
// layout the writer can never produce and read Sfull out of the *wrong row's*
// S0. Same lesson as the GA64 reciprocal site: a rule enforced at one site and
// merely assumed at another is not enforced.
inline uint32_t fold_row_stride(uint32_t fold_stride, uint32_t nOp) {
  return (fold_stride == 0) ? 2u * nOp : fold_stride;
}

// The ONE accept/reject DECISION for a fold row stride, consumed by the writer
// (quant_gemm_dpas48_rowfold) AND by fold_readable() below.
//
// AUDIT #13: the writer used to open-code this as a bare
// `if (fold_stride < 2*nOp) return dep;` -- a SECOND copy of a rule whose header
// comment explicitly says "do NOT resolve or validate this by hand at a call
// site". Two copies is the exact condition AUDIT #12 exists to eliminate, and
// it was invisible to the contract check (which only calls fold_readable) and to
// the whole mutation matrix (which never touched the writer's line). It was
// caught only by the probe's two NEG stride cases. Same semantics, one home.
inline bool fold_stride_ok(uint32_t fold_stride, uint32_t nOp) {
  return fold_row_stride(fold_stride, nOp) >= 2u * nOp;
}

// Fail CLOSED. A reader handed a fold buffer whose stride it cannot validate
// returns nullptr and recomputes the column sums inline: bit-identical to the
// pre-FOLD path, just without the hoist. Never returns a mis-strided row.
inline const std::int32_t* fold_readable(const std::int32_t* usm,
                                         uint32_t fold_stride,
                                         uint32_t nOp) {
  if (!usm) return nullptr;
  return fold_stride_ok(fold_stride, nOp) ? usm : nullptr;
}

// Fold INDEX rule -- likewise one definition, consumed by every read site in
// both readers. Kept separate from the stride rule on purpose: the stride check
// reasons about arithmetic agreement and is structurally BLIND to the kernel's
// indexing. Mutation S4 in scripts/mutate_fold_matrix.sh drops the `+ nOp`
// that selects the Sfull half from the reader's index expression and the stride
// check still reports 80/80 agreement -- only the device readback catches it,
// at 60/64 outputs. Routing every read through these two functions means the
// expression has exactly ONE home, so it can be unit-gated cheaply instead of
// only by a device launch.
//
// Sfull half: selected by `+ nOp` (the second nOp ints of the row block).
inline std::size_t fold_index_sfull(std::size_t m, uint32_t fold_stride,
                                   uint32_t nOp, uint32_t o) {
  return m * (std::size_t)fold_stride + (std::size_t)nOp + (std::size_t)o;
}
// S0 half: the first nOp ints of the row block (INT2/G16 linearity split).
//
// THIS ONE HAS NO DEVICE COVERAGE ANYWHERE (mutation S6, audit #13). It is
// called from exactly one place -- the u2/G16 split branch of quant_gemm_dpas48
// -- and every device readback in the tree runs bits=4, i.e. the u4 path, which
// reads only fold_index_sfull. Drop the row offset here and the device readback
// stays GREEN; the probe's host-side index check is the only thing that fails.
// Do not assume a device arm is covering this when you add one.
inline std::size_t fold_index_s0(std::size_t m, uint32_t fold_stride,
                                 uint32_t o) {
  return m * (std::size_t)fold_stride + (std::size_t)o;
}

// DPAS48 row-fold pre-pass (EXPERIMENTAL, default-off INTFER_DPAS48_FOLD).
//
// quant_gemm_dpas48 runs one WI per (row m, 16-N tile t) and, inside its K
// loop, recomputes the activation column sums
//     S  = sum_{k<32} va[k]                (u4/u8/u2 non-split)
//     S0 = sum_{k<16} va[k], Sf = S - S0   (INT2/G16 linearity split)
// Those depend ONLY on (row, op), never on the tile, so each of the
// nT = N/16 tiles per row recomputes an identical O(K) integer fold. That
// replication is the documented reason the wide-N DPAS kernel loses to the
// legacy tiled path even though the multiply itself is on XMX.
//
// This computes the sums once per (row, op). BIT-EXACT BY CONSTRUCTION: they
// are exact int32 sums of int8 codes (|sum| <= 32*128 = 4096, nowhere near
// overflow), and integer addition is exact and associative, so hoisting the
// sum out of the tile loop cannot change a single bit — unlike the FP32 folds
// elsewhere in this file, there is no rounding-order exposure. The DPAS math,
// the scales, and the accumulation order are all untouched.
//
// Layout (row stride fold_stride ints; MUST be >= 2*nOp, i.e. >= 2*K/32):
//     fold_usm[(m*fold_stride) + o]           = S0    (first 16 of the op)
//     fold_usm[(m*fold_stride) + nOp + o]     = Sfull (all 32; = Sf in split)
// The >= 2*nOp requirement is NOT cosmetic: each row occupies 2*nOp ints
// ([S0 block][Sfull block]), so a shorter stride makes row m's Sfull block
// overwrite row m+1's S0 block. That is silent at M=1 (only row 0 exists) and
// only bites once the workspace stride is smaller than the call's 2*nOp, which
// is exactly what a grow-only workspace hits when a smaller-K GEMM follows a
// larger one. Found by examples/dpas_fold_probe, which gates this exact value.
// `dep` is the optional prerequisite (the act-quant pre-pass event under
// INTFER_DPAS_PREQ=1), because this reads the same activation buffer it
// writes. Default {} means the activations are already on device.
sycl::event quant_gemm_dpas48_rowfold(sycl::queue& q,
                                      const std::int8_t* act_int8,
                                      std::int32_t* fold_usm,
                                      uint32_t M, uint32_t K,
                                      uint32_t act_stride,
                                      uint32_t fold_stride,
                                      sycl::event dep = sycl::event{});
// Declines (returns `dep` unchanged and writes NOTHING) when the geometry is
// unusable: M==0, K < 32, K % 32 != 0, or fold_stride < 2*(K/32). The caller
// must treat a decline as "fold_usm is not usable" and keep the inline fold —
// which is why the engine re-checks before setting Dpas48Params::fold_usm.
// NOTE: the device act-quant pre-pass dependency is NOT a Dpas48Params
// member — the struct is captured BY VALUE into the kernel lambda, and
// sycl::event is not device-copyable. It is a separate trailing parameter
// instead (default {} = no dependency).

// Host-side repack: row-major packed codes (n*K+k indices) -> DPAS vertical
// B tiles. out size = (N/16) * (K/opK) * (opK*16*b/8) bytes; opK=32.
// Returns false when N%16 != 0 or K%32 != 0 (caller falls back).
bool repack_dpas412(const std::uint8_t* row_codes, uint32_t N, uint32_t K,
                    int bits, std::vector<std::uint8_t>& out);

// M=1 W4A8/W2A8 decode kernel. Returns the kernel event. `dep` is an
// optional prerequisite (INTFER_DPAS_PREQ=1 passes the act-quant pre-pass
// event); default {} means the act buffers are already host-staged.
sycl::event quant_gemm_dpas48(sycl::queue& q, const Dpas48Params& p,
                              sycl::event dep = sycl::event{});

// E4 (INTFER_DPAS48_TILED=1, default OFF): the SAME kernel and the SAME math as
// quant_gemm_dpas48 -- literally the same template instantiated with 2 rows per
// work-item instead of 1 -- but each WI owns a 2D block of 2 rows x ONE 16-N
// tile, so the B op-tile is fetched once per op and reused across both rows.
//
// This is the load-replication half that E3/INTFER_DPAS48_FOLD left alone: E3
// hoisted the per-(row,op) ARITHMETIC replication, this hoists the LOAD.
//
// BIT-EXACT by construction, and that is a property of the geometry rather than
// a lucky tolerance: each (row, tile) keeps ONE accumulator and still sums over
// ops o = 0..nOp-1 in the same order with the same operands, so no FP32 add is
// reordered. E4's numerics contract (retired batch report; see AGENTS.md) was written for a design
// that combines per-block partial sums in a different order; this geometry has
// no partial sums, so it does not have that hazard. Per the contract, identity
// is REPORTED, never used as the gate.
//
// DECLINES (returns a null event) when M < 2, so the caller keeps the shipped
// kernel. E4 is therefore inert at M==1 -- i.e. across all of decode -- which is
// what the gate asserts rather than assumes.
sycl::event quant_gemm_dpas48_2d(sycl::queue& q, const Dpas48Params& p,
                                 sycl::event dep = sycl::event{});

// Device-side GA64 activation quant pre-pass (INTFER_DPAS_PREQ=1).
// Mirrors dashq::act_quant_ga64_batch(HalfAway) bit-exactly: per-64-slice
// max-abs (floor 1e-6), s=mx/127, clamp(x*inv, +-127), half-away-from-zero.
// One WI per (row, slice): each element is quanted and stored ONCE — the
// fused-quant failure mode replicated the O(K) quant work nT x, once per
// 16-N tile. Plain SYCL (no ESIMD: scalar FP32 + select only), so the
// arithmetic is plain IEEE-754 FP32 exactly like the host scalar loop. The
// caller dep-chains the DPAS launch on the returned event
// (Dpas48Params::dep), so the pre-pass costs no extra host wait.
//
// ONE LAUNCH PER DEVICE, never one for both ranks: every rank has its own
// sycl::context (device_manager.cpp), so a kernel on rank 0's queue cannot
// legally write rank 1's malloc_shared buffer.
//
// scale_inv_in IS REQUIRED (2*M*nGA FP32, row stride nGA, device USM, H2D'd),
// interleaved (s, inv) per slice from dashq::act_quant_ga64_scale_inv_batch.
// Both the scale AND its reciprocal are consumed verbatim and the per-slice
// max-abs reduction is therefore NOT redone on device. Rationale — the GA64
// rule q = trunc(x*inv +- 0.5) is pathologically ill-conditioned in `inv`: a
// 1-ulp difference in the reciprocal moves the product across a .5 tie and
// flips the code by one. `1.0f/s` is not a portable bit-source: the device
// lowers it to rcp.approx (1 ulp off on ~10% of legal scales) and host call
// sites in different contexts have been observed to disagree, so every GA64
// producer routes through ONE noinline reciprocal site in dashq.cpp
// (ga64_recip) and this kernel consumes those bits verbatim. Handing the bits
// across is the only way to be bit-exact by construction instead of by luck,
// and it also removes the only way the written scale could disagree with the
// inverse actually used. Gated by scripts/test_ga64_recip.sh.
struct ActQuantDevParams {
  const float* x_f32 = nullptr;      // M*K FP32 rows (device USM, H2D'd)
  const float* scale_inv_in = nullptr;  // 2*M*nGA FP32 (s, inv) pairs (tight)
  std::int8_t* q_out = nullptr;      // M*Ks INT8 out (row stride Ks)
  float* s_out = nullptr;            // M*Ss FP32 act scales (row stride Ss)
  uint32_t M = 0, K = 0;
  uint32_t Kstride = 0;              // workspace K (row stride, elements)
  uint32_t nGAstride = 0;            // workspace nGA (row stride)
};
sycl::event act_quant_ga64_dev(sycl::queue& q, const ActQuantDevParams& p);

// Row-cooperative INT4/G64 DPAS (EXPERIMENTAL, SLOWER — see .cpp note).
// Same Dpas48Params contract as quant_gemm_dpas48; one WI per row loops
// over 16-N tiles. Bit-exact vs the u4 path; ~460x slower device-side
// (occupancy collapse). Default-off INTFER_DPAS_ROW=1 opts in; DECLINES unless
// dpas4_row_eligible() says otherwise. Experiment base for subgroup-cooperative
// sharing only — do NOT promote without a quiet A/B.
sycl::event quant_gemm_dpas4_row(sycl::queue& q, const Dpas48Params& p,
                                sycl::event dep = sycl::event{});

// *** THE ROW KERNEL'S ELIGIBILITY RULE, IN ONE PLACE. ***
//
// It used to be written twice -- once in the kernel (`if (p.bits != 4 || G != 64)
// return sycl::event{}`) and once in the engine (`use_row = ... && bits == 4 &&
// Gw == 64`) -- and the two happened to agree, which is the audited-to-death shape
// of a rule fixed at one site and re-derived at another. The engine's copy is the
// dangerous one: it is the reason the kernel's decline path is unreachable today.
//
// It MUST stay unreachable, and it used to be unreachable only by coincidence.
// Measured 2026-10-03 on this toolchain: an event that a callee produced by
// `return sycl::event{}` does NOT compare equal to a bare `sycl::event{}`
// constructed at the call site (see dpas_fold_probe's "decline conventions"
// section, which asserts both directions). So engine.cpp's
// `kernel_evt == sycl::event{}` fallback CANNOT be relied on to catch this
// kernel's decline -- which is precisely why the guarantee has to be the
// eligibility check, not the event comparison.
//
// Call this from the engine BEFORE launching, and the identity test is defence in
// depth instead of the only thing standing between a decline and a silently
// skipped GEMM.
inline bool dpas4_row_eligible(uint32_t bits, uint32_t group_size) {
  return bits == 4 && group_size == 64;
}

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

// Fused-launch accounting: the quantized GEMM path (engine.cpp) counts its
// own launches, but SSM/norm kernels submit directly through these free
// functions, so their launches are invisible to the [GPU profile] nk
// counter. Any submitter calls this exactly once per q.submit it issues
// (guarded by INTFER_PROFILE_GPU at the call site; the definition lives in
// engine.cpp next to g_gpu_profile). Zero-cost when profiling is off.
#ifdef INTFER_PROFILE_GPU
void profile_note_launch();
#endif

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
// M-row variant (L4 M>1 serve fusion): M contiguous core/z rows in,
// M contiguous FP16 act rows out. One WI per (row m, head hh); same
// per-element math/order as the single-row kernel (host-loop identical,
// silu via sycl::exp, RNE convert). norm_w is shared across rows.
// dep chains after the M-row SSM kernel.
struct SsmNormMParams {
  const float* core_fp32;      // [M*n_v*vd] SSM core rows (device FP32)
  const float* z_fp32;         // [M*n_v*vd] z gate rows (device FP32)
  const float* norm_w_fp32;    // [vd] ssm norm weights (device FP32, shared)
  std::uint16_t* act_fp16;     // [M*n_v*vd] fused la_out acts (device FP16)
  uint32_t M;                  // batch rows
  uint32_t n_v;                // value heads
  uint32_t vd;                 // v head dim
  float rms_eps;
};
sycl::event ssm_norm_silu_to_half_m(sycl::queue& q, const SsmNormMParams& p,
                                    const sycl::event& dep = {});

// ---------------------------------------------------------------------------
// L5 LA-front chain (device-resident LA input path, TP=1 scalar decode):
// two kernels that replace the host wave-4 + conv + q/k prep ahead of the
// certified L4 tail (ssm_forward -> ssm_norm_silu_to_half -> devact la_out).
//
// Kernel 1 (fwht_rot_to_half): normed[H] FP32 in (one H2D per layer), 4x
// FWHT online rotations (null = copy, hadamard = exact host-order FWHT +
// /sqrt(p), dense = WI writes nothing, caller declines) + RNE FP16
// convert, 4x device act rows out. One WI per tensor (sequential FWHT =
// host loop order, bit-identical, no SLM/ordering risk).
// Kernel 2 (la_conv_prep): mqkv[conv_dim] FP32 (device GEMM out) +
// conv_state (device mirror, updated in place) + conv_w[conv_dim*conv_k]
// -> conv FIR + silu (host loop order, silu via sycl::exp ~1ulp) ->
// conv_out slices + q/k l2norm+scale (sycl::sqrt, IEEE exact) + g logits
// (sycl::exp/log1p analogs of host expf/log1pf ~1ulp) + v/b pack into the
// L4 sd->in packed layout [qn(nk*kd)|kn(nk*kd)|v(zvw)|g(nv)|b(nv)].
// One WI (sequential, host order). dep chains after the conv-input GEMM.
// ---------------------------------------------------------------------------
struct FwhtRotParams {
  const float* normed_fp32;     // [H] normalized hidden (device FP32)
  std::uint16_t* acts_fp16[4];  // 4x [Nmax] device FP16 act rows (qkv/z/b/a)
  const uint8_t* rot_kind;      // 4x kind: 0=null(copy) 1=hadamard(FWHT) 2=dense(decline)
  uint32_t rot_p;               // FWHT prefix (floor_pow2(dim); dim when pow2)
  float rot_inv_sqrt;           // 1/sqrt(p) FWHT scale (host-identical value)
  uint32_t H;                   // hidden dim (= tensor K)
};
sycl::event fwht_rot_to_half(sycl::queue& q, const FwhtRotParams& p,
                             const sycl::event& dep = {});
struct LaConvPrepParams {  const float* mqkv_fp32;    // [conv_dim] qkv GEMM out (device FP32)
  float* conv_state;         // [conv_dim*3] device mirror (updated in place)
  const float* conv_w;       // [conv_dim*conv_k] conv weights (device FP32)
  const float* b_row;        // [nv] b GEMM row (device FP32)
  const float* a_row;        // [nv] a GEMM row (device FP32)
  float* packed_in;          // [2*nk*kd+zvw+2*nv+conv_dim] L4 sd->in layout
                             // (device FP32) + conv_dim scratch tail
  const float* dt_bias;      // [nv] (device FP32)
  const float* A_log;        // [nv] (device FP32)
  uint32_t conv_dim, conv_k;
  uint32_t nv, nk, kd, vd;
  float qscale;              // 1/sqrt(kd) (host-identical value)
};
sycl::event la_conv_prep(sycl::queue& q, const LaConvPrepParams& p,
                         const sycl::event& dep = {});

// ---------------------------------------------------------------------------
// Device-side FP16->FP32 convert (L5 front chain): GEMM FP16 outputs that
// downstream device kernels consume as FP32 (qkv/b/a rows into la_conv_prep,
// z row into the L4 norm staging). Bit-exact dev_half_to_float per element
// (half->float is exact; matches dashq::half_to_float). One WI per element.
// ---------------------------------------------------------------------------
struct HalfToFloatParams {
  const std::uint16_t* in_fp16;  // [n] device FP16 bits
  float* out_fp32;               // [n] device FP32
  uint32_t n;
};
sycl::event half_to_float(sycl::queue& q, const HalfToFloatParams& p,
                         const sycl::event& dep = {});

// ---------------------------------------------------------------------------
// Device-side lm_head argmax (serve scoring seam): per-row argmax over one
// device-resident FP16 vocab tile [M x n], avoiding the D2H of full logits.
// Bit-contract: per-row single WI sequential scan, `float s =
// dev_half_to_float(bits); if (s > bestv)` strict->, init best=-1,
// bestv=-1e30f (first-max-wins, NaN ignored — identical to the host loops in
// main.cpp step()/score_row/step_batch). No tree reduction, no fast-math.
// out_idx gets the global vocab id (global_col0+i) or -1 for skipped rows;
// out_val gets the winning value or -1e30f.
// ---------------------------------------------------------------------------
struct LmArgmaxTileParams {
  const std::uint16_t* tile_fp16 = nullptr;  // [M x row_stride] device FP16
  std::uint32_t n = 0;                       // tile width (vocab cols)
  std::uint32_t M = 1;                       // rows
  std::uint32_t row_stride = 0;              // 0 => tight (== n)
  std::uint32_t global_col0 = 0;             // vocab offset of tile col 0
  const std::uint8_t* is_greedy = nullptr;   // [M] or nullptr (all greedy)
  int* out_idx = nullptr;                    // [M] device out
  float* out_val = nullptr;                  // [M] device out
};
sycl::event lm_head_argmax_tile(sycl::queue& q, const LmArgmaxTileParams& p,
                                const sycl::event& dep = {});

}  // namespace intfer::gemm
