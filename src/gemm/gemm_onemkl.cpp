// src/gemm/gemm_onemkl.cpp
//
// 1.4 oneMKL `gemm` fallback for large contiguous FP16 GEMMs.
//
// Built ONLY when CMake found libmkl_sycl_blas (INTFER_ENABLE_MKL=ON, default
// ON when the library is present). When the library is absent this file is
// not compiled at all and intfer_gemm exports the empty-fallback versions of
// gemm::onemkl_available() / gemm::gemm_onemkl_fp16() declared in
// quant_gemm.hpp, so the engine links either way.
//
// THE NUMERICS CONTRACT, first, because it is the whole risk of linking a
// BLAS. oneMKL's FP32 accumulation order is an internal implementation detail
// of the mkl::blas kernel and is NOT the engine's row-major ascending-k
// order. So a bit-exactness guarantee CANNOT be made for this arm, and none
// is claimed. PLAN §3 records why: "the repo has a measured matmul-precision
// divergence history, so the contract comes first." The arm therefore runs
// only on explicit opt-in (INTFER_ONEMKL=1), and the gate compares the token
// stream against base under an AGREED TOLERANCE -- never bitwise.
//
// WHERE IT APPLIES: large-M contiguous FP16 only. On this box FP16 already
// runs at 98-123% of the memory roofline on decode, so a BLAS call cannot pay
// at M=1 -- it is a prefill/large-M item. The engine gates on M >= 8 and a
// contiguous FP16 (non-quantized, non-sharded-split) tensor.

#include "intfer/gemm/quant_gemm.hpp"

#ifdef INTFER_ENABLE_MKL

#include <sycl/sycl.hpp>
// The composite header is the ONLY supported spelling: it defines
// MKL_BLAS_COMPUTE_MODE before pulling usm.hpp, and includes
// oneapi/mkl/blas/types.hpp (where `compute_mode` lives). Including
// blas/usm.hpp directly leaves MKL_BLAS_COMPUTE_MODE undefined and
// compute_mode undeclared -- a compile error that looks like a header bug.
#include <oneapi/mkl/blas.hpp>

namespace intfer::gemm {

bool onemkl_available() { return true; }

// sycl::event return, matching the engine's other GEMM arms' signature so the
// call site can queue dependency ordering.
//
// Row-major C[M,N] += A[M,K] * B[K,N], all FP16 in / FP32 accumulate out.
// The MKL SYCL BLAS is column-major, so the row-major product is computed as
// C^T = B^T A^T with transa/transb swapped -- the same trick llama.cpp uses.
sycl::event gemm_onemkl_fp16(sycl::queue& q, sycl::half* C,
                             const sycl::half* A, const sycl::half* B,
                             std::uint64_t M, std::uint64_t N,
                             std::uint64_t K, float alpha, float beta,
                             sycl::event dep) {
  // This header declares the SYCL-USM (unified shared memory) BLAS, which is
  // what the engine's USM kernel path uses. The buffer-based variant
  // (oneapi/mkl/blas/buffer.hpp) is the other spelling and needs a
  // sycl::buffer surface, which never reaches a USM kernel.
  if (dep != sycl::event{}) dep.wait();
  // Row-major C[M,N] = A[M,K] * B[K,N]^T. oneMKL's SYCL BLAS is COLUMN-major,
  // so this row-major product is computed as its own transpose:
  //   C^T[N,M] = (A*B^T)^T = B * A^T
  // i.e. in MKL terms, gemm(nontrans, trans, M=N, N=M, K, ...) over B (=W,
  // [N,K] row-major) and A (=x, [M,K] row-major), which is the same
  // transposition llama.cpp uses for its SYCL BLAS fallback.
  //
  // *** THE PATH ANNOUNCES ITSELF, ONCE. *** This arm is NOT bit-exact by
  // construction, so a stream comparison is not just a vacuity risk -- it is
  // the CONTRACT itself, and it needs the run to have actually reached here
  // for the comparison to mean anything. Prints the shape, and the tolerance
  // status is stated in the gate, not on this line.
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
  oneapi::mkl::blas::gemm(q, oneapi::mkl::transpose::nontrans,
                          oneapi::mkl::transpose::trans, N, M, K,
                          alpha, B, K, A, K, beta,
                          reinterpret_cast<sycl::half*>(C), N);
  return sycl::event{};
}

}  // namespace intfer::gemm

#else  // !INTFER_ENABLE_MKL

// Empty fallbacks so intfer_gemm links and the engine's onemkl_available()
// query returns false without any #ifdef at the call site. The engine then
// never calls gemm_onemkl_fp16(), so the tolerance-gated arm is unreachable
// rather than wrong.
namespace intfer::gemm {

bool onemkl_available() { return false; }

sycl::event gemm_onemkl_fp16(sycl::queue&, sycl::half*,
                             const sycl::half*, const sycl::half*,
                             std::uint64_t, std::uint64_t, std::uint64_t,
                             float, float, sycl::event) {
  return sycl::event{};
}

}  // namespace intfer::gemm

#endif  // INTFER_ENABLE_MKL
