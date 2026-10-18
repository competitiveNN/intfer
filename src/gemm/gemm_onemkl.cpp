// src/gemm/gemm_onemkl.cpp
//
// 1.4 oneMKL `gemm` fallback for large contiguous FP16 GEMMs.
//
// Built ALWAYS. Under #ifdef INTFER_ENABLE_MKL it carries the real oneMKL
// SYCL BLAS body; under #else it carries the empty-fallback spellings of
// onemkl_available() / gemm_onemkl_fp16(), so the engine links either way and
// the arm is unreachable rather than wrong when the library is absent.
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
// at M=1 -- it is a prefill/large-M item. quant_gemm_f16 gates on M >= 8.

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
  // i.e. in MKL terms, gemm(m=N, n=M, k=K) over B (=W, [N,K] row-major) and
  // A (=x, [M,K] row-major).
  //
  // *** THE TRANSPOSE FLAGS, AND WHY THE OBVIOUS SPELLING IS WRONG. ***
  // MEASURED 2026-10-15 by examples/onemkl_probe (M8-N128-K128), never before
  // because the gate's gate_require_build built into $ROOT/build while it ran
  // the binary from $GATE_BUILD_DIR -- so the probe it ran was the MKL-OFF one
  // and every arm exited SKIP. The shipped spelling was
  //   gemm(nontrans, trans, N, M, K, alpha, B, K, A, K, beta, C, N)
  // and it computed a DIFFERENT product: max_rel 64.6, 1023/1024 elements
  // differing from the host reference, worst at m=0 n=5.
  //
  // A row-major [R, Lc] buffer IS its own column-major transpose [Lc, R] when
  // read with leading dimension Lc: element (i, j) sits at buf[j*Lc + i]. For
  // the weight buffer W (=B, row-major [N, K]) that buffer is a column-major
  // [K, N] matrix holding W^T, so it can serve MKL's `a` when MKL wants
  // op(a) == W, i.e. when transa=T and a is declared k x m (K x N). Likewise
  // the activation A (row-major [M, K]) read as column-major [K, M] holds
  // A^T, so it serves MKL's `b` when transb=N and b is declared k x n (K x M).
  // The product is then W (N x K) * A^T (K x M) = C^T (N x M), stored with
  // ldc=N into the row-major C[M, N] buffer -- element (i, j) at C[j*N + i].
  //
  // With the flags the other way round (nontrans, trans) MKL reads
  // op(a) as the N x K matrix W^T and op(b) -- declared with ldb=K, which for
  // transb=T MKL reads as an n x k = M x K column-major buffer, i.e. only
  // M rows -- so the K index runs off the end of the activation buffer and the
  // arithmetic is a different product entirely. The ld arguments themselves are
  // unchanged; only the two flags were swapped.
  //
  // *** THE PATH ANNOUNCES ITSELF, ONCE. *** This arm is NOT bit-exact by
  // construction, so a stream comparison is not just a vacuity risk -- it is
  // the CONTRACT itself, and it needs the run to have actually reached here
  // for the comparison to mean anything. Prints the shape; the tolerance
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
  // transa=T, transb=N -- see the transpose derivation above. The WRONG
  // (nontrans, trans) spelling this replaces is recorded there with its
  // measured effect.
  oneapi::mkl::blas::gemm(q, oneapi::mkl::transpose::trans,
                          oneapi::mkl::transpose::nontrans, N, M, K,
                          alpha, B, K, A, K, beta,
                          reinterpret_cast<sycl::half*>(C), N);
  return sycl::event{};
}

}  // namespace intfer::gemm

#else  // !INTFER_ENABLE_MKL

// Empty fallbacks so intfer_gemm links and the engine's onemkl_available()
// query returns false without any #ifdef at the call site. The engine then
// never calls gemm_onemkl_fp16(), so the tolerance-gated arm is unreachable
// rather than wrong. quant_gemm_f16's own gate (onemkl_available() && M >= 8)
// is the dispatch point; this file is what makes that predicate real.
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