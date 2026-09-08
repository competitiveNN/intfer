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

}  // namespace intfer::gemm
