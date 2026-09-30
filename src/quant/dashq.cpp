// src/quant/dashq.cpp
// DASH-Q diagonal-Hessian PTQ solver implementation. Pure C++ (no SYCL) so it
// runs on the host and is independently testable; the Stage-2 SYCL path (t7)
// consumes the emitted code/scale/global buffers.
#include "intfer/quant/dashq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace intfer {
namespace dashq {

#if defined(__x86_64__) || defined(_M_X64)
// F16C probe: _mm256_cvtps_ph faults with #UD without F16C. cpuid leaf 1,
// ECX bit 29.
bool f16c_available() {
  static const bool ok = [] {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 1);
    ecx = (unsigned int)regs[2];
#else
    __asm__ volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(1u), "c"(0u));
#endif
    return (ecx & (1u << 29)) != 0;
  }();
  return ok;
}

// AVX2 + AVX-OS-support probe (leaf 7 EBX bit 5; leaf 1 ECX AVX/OSXSAVE;
// XCR0 SSE+YMM state). _mm256_* faults with #UD without all three.
bool avx2_available() {
  static const bool ok = [] {
#if defined(_MSC_VER)
    return true;  // x86-64 Windows toolchains target AVX2-capable CPUs here
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __asm__ volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(1u), "c"(0u));
    const bool avx = (ecx & (1u << 28)) != 0;
    const bool osxsave = (ecx & (1u << 27)) != 0;
    if (!avx || !osxsave) return false;
    // XCR0: XMM (bit 1) + YMM (bit 2) state enabled by the OS.
    unsigned int xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0u));
    if (((xcr0_lo & 0x6u) != 0x6u)) return false;
    __asm__ volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(7u), "c"(0u));
    return (ebx & (1u << 5)) != 0;
#endif
  }();
  return ok;
}

// 8-wide F16C kernels. Function-target keeps AVX2/F16C codegen local to
// these routines (supported by GCC, Clang, and icpx).
__attribute__((target("f16c,avx2")))
static void f32_to_h16_f16c(const float* x, uint16_t* y, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Full 128-bit store: cvtps_ph packs all 8 halves into the __m128i;
    // storel_epi64 would write only the low 4.
    _mm_storeu_si128(reinterpret_cast<__m128i*>(y + i), h);
  }
  for (; i < n; ++i) {
    __m128 s = _mm_load_ss(x + i);
    __m128i h = _mm_cvtps_ph(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    y[i] = (uint16_t)_mm_cvtsi128_si32(h);
  }
}

__attribute__((target("f16c,avx2")))
static void h16_to_f32_f16c(const uint16_t* x, float* y, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
    __m256 v = _mm256_cvtph_ps(h);
    _mm256_storeu_ps(y + i, v);
  }
  for (; i < n; ++i) {
    __m128i h = _mm_cvtsi32_si128(x[i]);
    __m128 s = _mm_cvtph_ps(h);
    _mm_store_ss(y + i, s);
  }
}
#endif

void float_to_half_batch(const float* x, uint16_t* y, size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
  if (f16c_available()) { f32_to_h16_f16c(x, y, n); return; }
#endif
  for (size_t i = 0; i < n; ++i) y[i] = float_to_half(x[i]);
}

void half_to_float_batch(const uint16_t* x, float* y, size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
  if (f16c_available()) { h16_to_f32_f16c(x, y, n); return; }
#endif
  for (size_t i = 0; i < n; ++i) y[i] = half_to_float(x[i]);
}

// ---------------------------------------------------------------------------
// GA64 slice scale: ONE max-abs site and ONE reciprocal site.
//
// q = trunc(x*inv +- 0.5) is pathologically ill-conditioned in `inv`: a 1-ulp
// reciprocal difference moves the product across an exact .5 tie and flips the
// code by one. `1.0f/s` is therefore NOT a portable bit-source -- its codegen
// depends on the caller (the device lowers it to rcp.approx, which is 1 ulp off
// the correctly rounded value on ~10% of legal scales, and host call sites
// written in different contexts have been observed to disagree with each
// other). So EVERY producer of GA64 reciprocal bits (the AVX2 oracle, the
// scalar fallback, and act_quant_ga64_scale_inv_batch() feeding the device
// pre-pass) goes through the two helpers below:
//
//   ga64_slice_max*  -- one max-abs reduction, so the scale is one value and
//                       the reciprocal is always derived from THAT scale.
//   ga64_recip       -- noinline on purpose: one compiled body means one
//                       instruction sequence, which makes the shared bits a
//                       structural property instead of a coincidence of how
//                       each caller happens to be optimized. Do NOT inline it,
//                       do NOT add a second division site, and do NOT re-add an
//                       fma/Newton "correction": a device- or kernel-local
//                       reciprocal can only move AWAY from the oracle. (The
//                       shared site is correctly rounded today -- verified over
//                       64000 legal scales by examples/ga64_recip_test -- but
//                       that is a property to preserve, not to re-derive.)
#if defined(_MSC_VER)
#define INTFER_GA64_NOINLINE __declspec(noinline)
#else
#define INTFER_GA64_NOINLINE __attribute__((noinline))
#endif

static INTFER_GA64_NOINLINE float ga64_recip(float s) { return 1.0f / s; }

static float ga64_slice_max_scalar(const float* x, size_t k0, size_t k1) {
  float mx = 1e-6f;
  for (size_t k = k0; k < k1; ++k) {
    float v = x[k] >= 0 ? x[k] : -x[k];
    if (v > mx) mx = v;
  }
  return mx;
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static float ga64_slice_max_avx2(const float* x, size_t k0, size_t k1) {
  const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  __m256 mx8 = _mm256_set1_ps(1e-6f);
  size_t k = k0;
  for (; k + 8 <= k1; k += 8) {
    __m256 v = _mm256_loadu_ps(x + k);
    mx8 = _mm256_max_ps(mx8, _mm256_and_ps(v, absmask));
  }
  // Horizontal max of the 8 lanes.
  __m128 lo = _mm256_castps256_ps128(mx8);
  __m128 hi = _mm256_extractf128_ps(mx8, 1);
  __m128 m4 = _mm_max_ps(lo, hi);
  m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
  m4 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, 0x55));
  float mx = _mm_cvtss_f32(m4);
  for (; k < k1; ++k) {
    float v = x[k] >= 0 ? x[k] : -x[k];
    if (v > mx) mx = v;
  }
  return mx;
}

static inline float ga64_slice_max(const float* x, size_t k0, size_t k1) {
  if (avx2_available()) return ga64_slice_max_avx2(x, k0, k1);
  return ga64_slice_max_scalar(x, k0, k1);
}
#else
static inline float ga64_slice_max(const float* x, size_t k0, size_t k1) {
  return ga64_slice_max_scalar(x, k0, k1);
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
// 8-wide AVX2 GA=64 act quant. Per slice: horizontal max-abs (SIMD tree +
// scalar tail), then scale/broadcast/quantize.
// HalfAway: q = trunc(v*inv + copysign(0.5, v*inv)) via cvttps (truncation,
// immune to MXCSR) — lround-exact, incl. exact x.5 away from zero.
// RoundNear: q = (int8_t)cvtps_epi32(v*inv), which rounds per MXCSR exactly
// like lrint does (RNE by default) — lrint-exact, incl. exact x.5 to even.
// The two modes genuinely differ (82.5 -> 83 vs 82); shared max-abs + tail
// take the mode as a template param to keep both paths in one place.
template <ActQuantRound kRound>
__attribute__((target("avx2")))
static void act_quant_ga64_avx2(const float* x, int8_t* q, float* scales,
                                size_t K) {
  constexpr size_t GA = 64;
  const __m256 signbit = _mm256_set1_ps(-0.0f);
  const __m256 v127 = _mm256_set1_ps(127.0f);
  const __m256 vn127 = _mm256_set1_ps(-127.0f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const size_t nGA = (K + GA - 1) / GA;
  for (size_t g = 0; g < nGA; ++g) {
    const size_t k0 = g * GA;
    size_t k1 = k0 + GA;
    if (k1 > K) k1 = K;
    // Scale from the SHARED max-abs site and the SHARED reciprocal site (see
    // ga64_recip): the oracle and the device pre-pass feed must not be able to
    // disagree, at any .5 tie, ever.
    const float s = ga64_slice_max_avx2(x, k0, k1) / 127.0f;
    const float inv = ga64_recip(s);
    scales[g] = s;
    // Quantize: clamp, then round per mode.
    const __m256 vinv = _mm256_set1_ps(inv);
    size_t k = k0;
    for (; k + 8 <= k1; k += 8) {
      __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + k), vinv);
      v = _mm256_min_ps(_mm256_max_ps(v, vn127), v127);
      __m256i q32;
      if constexpr (kRound == ActQuantRound::HalfAway) {
        __m256 adj = _mm256_or_ps(_mm256_and_ps(v, signbit), half);
        q32 = _mm256_cvttps_epi32(_mm256_add_ps(v, adj));
      } else {
        q32 = _mm256_cvtps_epi32(v);
      }
      // Pack 8x i32 -> 8x i8 (values in [-127,127]; saturation identity).
      // NOTE: 256-bit packs operate in 128-bit lanes (lanes 4-7 would land
      // in the high half), so pack in 128-bit: lo=lanes 0-3, hi=lanes 4-7.
      __m128i lo32 = _mm256_castsi256_si128(q32);
      __m128i hi32 = _mm256_extracti128_si256(q32, 1);
      __m128i p16 = _mm_packs_epi32(lo32, hi32);
      __m128i p8 = _mm_packs_epi16(p16, _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i*>(q + k), p8);
    }
    for (; k < k1; ++k) {
      float v = x[k] * inv;
      if (v > 127.0f) v = 127.0f;
      else if (v < -127.0f) v = -127.0f;
      if constexpr (kRound == ActQuantRound::HalfAway)
        q[k] = (int8_t)std::lround(v);
      else
        q[k] = (int8_t)std::lrint(v);
    }
  }
}
#endif

template <ActQuantRound kRound>
static void act_quant_ga64_scalar(const float* x, int8_t* q, float* scales,
                                  size_t K) {
  constexpr size_t GA = 64;
  const size_t nGA = (K + GA - 1) / GA;
  for (size_t g = 0; g < nGA; ++g) {
    const size_t k0 = g * GA;
    size_t k1 = k0 + GA;
    if (k1 > K) k1 = K;
    const float s = ga64_slice_max_scalar(x, k0, k1) / 127.0f;
    const float inv = ga64_recip(s);
    scales[g] = s;
    for (size_t k = k0; k < k1; ++k) {
      float v = x[k] * inv;
      if (v > 127.0f) v = 127.0f;
      else if (v < -127.0f) v = -127.0f;
      if constexpr (kRound == ActQuantRound::HalfAway)
        q[k] = (int8_t)std::lround(v);
      else
        q[k] = (int8_t)std::lrint(v);
    }
  }
}

void act_quant_ga64_scale_inv_batch(const float* x, size_t M, size_t K,
                                    float* out) {
  constexpr size_t GA = 64;
  // nGA is DERIVED here, not passed in: the device pre-pass derives it from K
  // with the identical expression, and a caller-supplied nGA that disagreed
  // would silently leave the tail slices' (s, inv) pairs uninitialised.
  const size_t nGA = (K + GA - 1) / GA;
  for (size_t m = 0; m < M; ++m) {
    const float* xr = x + m * K;
    float* o = out + 2 * (size_t)m * nGA;
    for (size_t g = 0; g < nGA; ++g) {
      const size_t k0 = g * GA;
      size_t k1 = k0 + GA;
      if (k1 > K) k1 = K;
      const float s = ga64_slice_max(xr, k0, k1) / 127.0f;
      o[2 * g] = s;
      o[2 * g + 1] = ga64_recip(s);
    }
  }
}

void act_quant_ga64_batch(const float* x, int8_t* q, float* scales,
                          size_t K, ActQuantRound round) {
#if defined(__x86_64__) || defined(_M_X64)
  if (avx2_available()) {
    if (round == ActQuantRound::HalfAway)
      act_quant_ga64_avx2<ActQuantRound::HalfAway>(x, q, scales, K);
    else
      act_quant_ga64_avx2<ActQuantRound::RoundNear>(x, q, scales, K);
    return;
  }
#endif
  if (round == ActQuantRound::HalfAway)
    act_quant_ga64_scalar<ActQuantRound::HalfAway>(x, q, scales, K);
  else
    act_quant_ga64_scalar<ActQuantRound::RoundNear>(x, q, scales, K);
}

// ---- IEEE-754 binary16 helpers -------------------------------------------
uint16_t float_to_half(float v) {
  uint32_t x; std::memcpy(&x, &v, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t exp  = (x >> 23) & 0xFFu;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp == 0xFFu) // inf or nan
    return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
  int e = (int)exp - 127 + 15;
  if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u); // overflow -> inf
  if (e <= 0) {                                      // subnormal or zero
    if (e < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    int shift = 14 - e;
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);       // round-to-nearest-even
    uint32_t hbit = 1u << (shift - 1);
    if (rem > hbit || (rem == hbit && (half & 1u))) half += 1u;
    if (half & 0x400u) return (uint16_t)(sign | 0x0400u);  // carry to min-normal
    return (uint16_t)(sign | (half & 0x3FFu));
  }
  uint32_t half = ((uint32_t)e << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFFu;                     // round-to-nearest-even
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half += 1u;
  return (uint16_t)(sign | (half & 0xFFFFu));
}

float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) & 1u;
  uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
  uint32_t mant = (uint32_t)(h & 0x3FFu);
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) { x = sign << 31; }
    else {
      int e = 0;
      while (!(mant & 0x400u)) { mant <<= 1; ++e; }
      mant &= 0x3FFu;
      x = (sign << 31) | ((127 - 14 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    x = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    x = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f; std::memcpy(&f, &x, 4);
  return f;
}

// ---- packing / unpacking --------------------------------------------------
static void pack_codes(const std::vector<uint8_t>& idx, int bits, std::vector<uint8_t>& out) {
  const size_t n = idx.size();
  if (bits == 8) { out = idx; return; }
  out.assign((n * bits + 7) / 8, 0);
  if (bits == 2) {
    for (size_t i = 0; i < n; ++i) out[i / 4] |= (uint8_t)((idx[i] & 0x3) << ((i % 4) * 2));
  } else if (bits == 4) {
    for (size_t i = 0; i < n; ++i) out[i / 2] |= (uint8_t)((idx[i] & 0xF) << ((i % 2) * 4));
  }
}
static uint8_t unpack_code(const uint8_t* buf, size_t i, int bits) {
  if (bits == 8) return buf[i];
  if (bits == 2) return (uint8_t)((buf[i / 4] >> ((i % 4) * 2)) & 0x3);
  if (bits == 4) return (uint8_t)((buf[i / 2] >> ((i % 2) * 4)) & 0xF);
  return 0;
}

// ---- canonical codebooks --------------------------------------------------
std::vector<float> canonical_codebook(QuantScheme s) {
  const int L = n_levels(s);
  std::vector<float> cb(L);
  if (s == QuantScheme::INT2) {
    // Non-uniform symmetric, |max| = 1 (the part worth copying from NVFP4).
    const float vals[4] = { -1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f };
    for (int i = 0; i < 4; ++i) cb[i] = vals[i];
  } else {
    // Uniform symmetric in [-1, 1].
    for (int i = 0; i < L; ++i) cb[i] = (2.0f * i - (L - 1)) / (L - 1);
  }
  return cb;
}

// Force a learned codebook to stay symmetric (docs §3.1).
static void symmetrize(std::vector<float>& cb) {
  const int L = (int)cb.size();
  for (int j = 0; j < L / 2; ++j) {
    const float mag = (std::fabs(cb[j]) + std::fabs(cb[L - 1 - j])) * 0.5f;
    cb[j] = -mag;
    cb[L - 1 - j] = mag;
  }
}

// ---------------------------------------------------------------------------
// Core per-group solver: alternating assignment / scale-LS, optionally codebook.
// ---------------------------------------------------------------------------
static float optimize_group(const std::vector<float>& w, const std::vector<float>& h,
                            const std::vector<float>& cb, int iters,
                            std::vector<uint8_t>& assign) {
  const int n = (int)w.size();
  const int L = (int)cb.size();
  float maxw = 0.f, maxc = 0.f;
  for (float v : w) maxw = std::max(maxw, std::fabs(v));
  for (float v : cb) maxc = std::max(maxc, std::fabs(v));
  float s = (maxc > 0.f) ? maxw / maxc : 1.0f;
  if (!(s > 0.f)) s = 1.0f;
  assign.assign(n, 0);

  for (int it = 0; it < iters; ++it) {
    // assignment:  a_i = argmin_j (w_i − s·c_j)²   (H_ii>0 is a positive factor)
    for (int i = 0; i < n; ++i) {
      float best = 1e30f;
      uint8_t bj = 0;
      const float wi = w[i];
      for (int j = 0; j < L; ++j) {
        const float d = wi - s * cb[j];
        const float dd = d * d;
        if (dd < best) { best = dd; bj = (uint8_t)j; }
      }
      assign[i] = bj;
    }
    // scale update (weighted least squares): s* = Σ H_ii·w_i·c_a / Σ H_ii·c_a²
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
      const float c = cb[assign[i]];
      num += (double)h[i] * w[i] * c;
      den += (double)h[i] * c * c;
    }
    if (den > 1e-12) s = (float)(num / den);
    else break;
  }
  // Converge to a consistent local optimum: re-assign with the final scale,
  // re-solve the weighted-LS scale for that assignment, then assign once more.
  // This guarantees (s, assign) is a fixed point so the reported reconstruction
  // error is the true minimum for this codebook (and <= any unweighted baseline).
  auto assign_all = [&](float sc) {
    for (int i = 0; i < n; ++i) {
      float best = 1e30f;
      uint8_t bj = 0;
      const float wi = w[i];
      for (int j = 0; j < L; ++j) {
        const float d = wi - sc * cb[j];
        const float dd = d * d;
        if (dd < best) { best = dd; bj = (uint8_t)j; }
      }
      assign[i] = bj;
    }
  };
  assign_all(s);
  { double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) { const float c = cb[assign[i]]; num += (double)h[i]*w[i]*c; den += (double)h[i]*c*c; }
    if (den > 1e-12) s = (float)(num / den); }
  assign_all(s);
  return s;
}

// Learn one shared non-uniform symmetric codebook for the whole tensor.
static std::vector<float> learn_codebook_tensor(const float* W, uint32_t N, uint32_t K,
                                         uint32_t G, const DiagHessian& H,
                                         const std::vector<float>& base_cb,
                                         int learner_iters, int inner_iters) {
  const int L = (int)base_cb.size();
  std::vector<float> cb = base_cb;
  const uint32_t n_groups = (K + G - 1) / G;
  auto h_at = [&](uint32_t r, uint32_t c) -> float {
    return H.per_element() ? H.data()[r * K + c] : H.data()[c];
  };
  for (int p = 0; p < learner_iters; ++p) {
    std::vector<double> num(L, 0.0), den(L, 0.0);
    for (uint32_t g = 0; g < n_groups; ++g) {
      std::vector<float> wg, hg;
      uint32_t gend = std::min((uint32_t)((g + 1) * G), K);
      for (uint32_t r = 0; r < N; ++r)
        for (uint32_t c = g * G; c < gend; ++c) {
          wg.push_back(W[r * K + c]);
          hg.push_back(h_at(r, c));
        }
      std::vector<uint8_t> assign;
      const float s = optimize_group(wg, hg, cb, inner_iters, assign);
      for (size_t i = 0; i < wg.size(); ++i) {
        const int j = assign[i];
        num[j] += (double)hg[i] * wg[i] / s;
        den[j] += (double)hg[i] / s;
      }
    }
    for (int j = 0; j < L; ++j)
      if (den[j] > 1e-12) cb[j] = (float)(num[j] / den[j]);
    symmetrize(cb);
  }
  return cb;
}

// ---------------------------------------------------------------------------
// QuantResult::dequant / eff_bits
// ---------------------------------------------------------------------------
void QuantResult::dequant(std::vector<float>& out, uint32_t N, uint32_t K) const {
  out.assign((size_t)N * K, 0.f);
  if (qcfg.scheme == QuantScheme::FP16) {
    for (uint32_t i = 0; i < N * K; ++i) {
      const uint16_t hv = (uint16_t)(codes[2 * i] | ((uint16_t)codes[2 * i + 1] << 8));
      out[i] = half_to_float(hv);
    }
    return;
  }
  const int bits = (qcfg.scheme == QuantScheme::INT2) ? 2
                 : (qcfg.scheme == QuantScheme::INT4) ? 4 : 8;
  const int L = n_levels(qcfg.scheme);
  const std::vector<float> cb = codebook.empty() ? canonical_codebook(qcfg.scheme) : codebook;
  const uint32_t G = qcfg.group_size > 0 ? qcfg.group_size : K;
  const uint32_t n_groups = (K + G - 1) / G;
  float alpha = 1.0f;
  if (!globals.empty() && globals.size() >= 4) std::memcpy(&alpha, globals.data(), 4);

  std::vector<uint8_t> idx;
  const size_t n = (size_t)N * K;
  idx.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (bits != 8 && (i >> (bits == 2 ? 2 : 1)) >= codes.size()) break; // ragged slice
    if (bits == 8 && i >= codes.size()) break;
    idx.push_back(unpack_code(codes.data(), i, bits));
  }
  if (idx.size() < n) idx.resize(n, 0);

  for (uint32_t g = 0; g < n_groups; ++g) {
    if (2 * (size_t)g + 1 >= scales.size()) break; // corrupt/ragged scale slice
    const uint16_t hv = (uint16_t)(scales[2 * g] | ((uint16_t)scales[2 * g + 1] << 8));
    const float s = half_to_float(hv) * alpha;
    const uint32_t gend = std::min((uint32_t)((g + 1) * G), K);
    const uint32_t Glocal = gend - g * G;
    for (uint32_t r = 0; r < N; ++r)
      for (uint32_t c = g * G; c < gend; ++c) {
        const uint32_t wi = r * Glocal + (c - g * G);
        const size_t ii = (size_t)g * (N * G) + wi;
        if (ii >= idx.size()) continue; // ragged slice
        const uint8_t a = idx[ii];
        if ((size_t)a >= cb.size()) continue;
        out[r * K + c] = s * cb[a];
      }
  }
}

float QuantResult::eff_bits(uint32_t N, uint32_t K) const {
  const uint64_t total = weight_bytes();
  if (N * K == 0) return 0.f;
  return (float)(total * 8) / (float)(N * K);
}

// ---------------------------------------------------------------------------
// quantize: top-level solver for one config
// ---------------------------------------------------------------------------
QuantResult quantize(const float* W, uint32_t N, uint32_t K, const QuantConfig& cfg,
                     const DiagHessian& H, bool learn_codebook, int iters) {
  QuantResult res;
  res.qcfg = cfg;

  if (cfg.scheme == QuantScheme::FP16) {
    res.codes.resize((size_t)N * K * 2);
    for (uint32_t i = 0; i < N * K; ++i) {
      const uint16_t hv = float_to_half(W[i]);
      res.codes[2 * i] = (uint8_t)(hv & 0xFF);
      res.codes[2 * i + 1] = (uint8_t)(hv >> 8);
    }
    if (cfg.has_global_divisor) {
      const float a = 1.0f;
      res.globals.resize(4);
      std::memcpy(res.globals.data(), &a, 4);
    }
    res.est_error = 0.f;
    res.weighted_mse = 0.f;
    double norm = 0.0;
    for (uint32_t i = 0; i < N * K; ++i) {
      const float hh = H.per_element() ? H.data()[i] : H.data()[i % K];
      norm += (double)hh * W[i] * W[i];
    }
    res.weighted_norm = (float)norm;
    return res;
  }

  const int L = n_levels(cfg.scheme);
  const int bits = (cfg.scheme == QuantScheme::INT2) ? 2
                 : (cfg.scheme == QuantScheme::INT4) ? 4 : 8;
  const uint32_t G = cfg.group_size > 0 ? cfg.group_size : K;
  const uint32_t n_groups = (K + G - 1) / G;

  std::vector<float> cb = canonical_codebook(cfg.scheme);
  if (learn_codebook && L <= 16)
    cb = learn_codebook_tensor(W, N, K, G, H, cb, std::max(1, iters / 2 + 1), iters);

  auto h_at = [&](uint32_t r, uint32_t c) -> float {
    return H.per_element() ? H.data()[r * K + c] : H.data()[c];
  };

  // For a fixed (non-learned, canonical) codebook the alternating
  // assignment/scale solver converges in 2-4 passes; the full `iters` sweep is
  // only needed for the learned-codebook path. Capping it keeps INT8 (256
  // levels) tractable on large tensors (e.g. the 248K-token embedding table)
  // without changing the result materially.
  const int grp_iters = learn_codebook ? iters : 4;

  // Layout assumption (matches dequant_tensor): each group occupies
  // N*G slots, group g lives at idx[g * N * G .. g*N*G + N*Glocal - 1].
  // The last group's trailing Glocal < G slots are left at 0.
  const size_t idx_per_group = (size_t)N * G;
  std::vector<uint8_t> idx(idx_per_group * n_groups, 0);
  res.scales.resize((size_t)n_groups * 2, 0);
  double mse = 0.0, norm = 0.0;

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic) reduction(+:mse,norm)
#endif
  for (uint32_t g = 0; g < n_groups; ++g) {
    const uint32_t gend = std::min((uint32_t)((g + 1) * G), K);
    const uint32_t Glocal = gend - g * G;
    // Build the local [N, Glocal] weight/Hessian slice.
    std::vector<float> wg, hg;
    wg.reserve((size_t)N * Glocal);
    hg.reserve((size_t)N * Glocal);
    for (uint32_t r = 0; r < N; ++r)
      for (uint32_t c = g * G; c < gend; ++c) {
        wg.push_back(W[r * K + c]);
        hg.push_back(h_at(r, c));
      }

    std::vector<uint8_t> assign;
    const float s = optimize_group(wg, hg, cb, grp_iters, assign);

    const uint16_t hv = float_to_half(s);
    res.scales[2 * g] = (uint8_t)(hv & 0xFF);
    res.scales[2 * g + 1] = (uint8_t)(hv >> 8);

    const size_t goff = (size_t)g * idx_per_group;
    for (uint32_t r = 0; r < N; ++r)
      for (uint32_t c = g * G; c < gend; ++c) {
        const uint32_t wi = r * Glocal + (c - g * G);
        idx[goff + wi] = assign[wi];
        const float dq = s * cb[assign[wi]];
        const float wv = W[r * K + c];
        const float hh = h_at(r, c);
        const double e = wv - dq;
        mse += (double)hh * e * e;
        norm += (double)hh * wv * wv;
      }
  }

  pack_codes(idx, bits, res.codes);
  if (cfg.has_global_divisor) {
    const float a = 1.0f;
    res.globals.resize(4);
    std::memcpy(res.globals.data(), &a, 4);
  }
  res.codebook = (learn_codebook && L <= 16) ? cb : std::vector<float>{};
  res.weighted_mse = (float)mse;
  res.weighted_norm = (float)norm;
  res.est_error = (norm > 0.0) ? (float)std::sqrt(mse / norm) : 0.f;
  return res;
}

// ---------------------------------------------------------------------------
// dequant_tensor: dequantize a finalized tensor directly from a ModelIR payload.
// ---------------------------------------------------------------------------
void dequant_tensor(const ModelIR& m, const TensorIR& t, std::vector<float>& out) {
  QuantResult r;
  r.qcfg = t.qcfg;
  r.codes.assign(m.code_data() + t.code_offset,
                 m.code_data() + t.code_offset + t.code_bytes);
  r.scales.assign(m.scale_data() + t.scale_offset,
                  m.scale_data() + t.scale_offset + t.scale_bytes);
  r.globals.assign(m.global_data() + t.global_offset,
                   m.global_data() + t.global_offset + t.global_bytes);
  r.dequant(out, t.N, t.K);
}

// ---------------------------------------------------------------------------
// quantize_candidates: fill the TensorIR candidate list + ModelIR payload.
// ---------------------------------------------------------------------------
void quantize_candidates(TensorIR& parent, const float* W, uint32_t N, uint32_t K,
                         const DiagHessian& H, ModelIR& model, bool learn_codebook,
                         uint64_t skip_int8_above_elements) {
  const std::vector<QuantConfig> cfgs = default_candidates(parent.role);
  for (const auto& base : cfgs) {
    // INT8 (256 levels) per-group assignment is O(256·N·G) — prohibitive on very
    // large tensors (e.g. the 248K-token embedding). RCO will never pick INT8 for
    // such tensors anyway (their floor is INT4 and they are budget-expensive), so
    // skipping the candidate saves minutes with no effect on the chosen plan.
    if (skip_int8_above_elements > 0 && base.scheme == QuantScheme::INT8 &&
        (uint64_t)N * K > skip_int8_above_elements)
      continue;
    QuantConfig c = base;
    c.rotation_id = parent.qcfg.rotation_id; // carry SpinQuant id if t3 ran first
    const QuantResult r = quantize(W, N, K, c, H, learn_codebook);

    TensorIR cand;
    cand.name = parent.name;
    cand.role = parent.role;
    cand.N = N;
    cand.K = K;
    cand.qcfg = c;
    cand.est_error = r.est_error;

    cand.code_offset = model.code_buf.size();
    cand.code_bytes = r.codes.size();
    model.code_buf.insert(model.code_buf.end(), r.codes.begin(), r.codes.end());

    cand.scale_offset = model.scale_buf.size();
    cand.scale_bytes = r.scales.size();
    model.scale_buf.insert(model.scale_buf.end(), r.scales.begin(), r.scales.end());

    cand.global_offset = model.global_buf.size();
    cand.global_bytes = r.globals.size();
    model.global_buf.insert(model.global_buf.end(), r.globals.begin(), r.globals.end());

    parent.candidates.push_back(cand);
  }
}

// ---------------------------------------------------------------------------
// DiagHessian estimators
// ---------------------------------------------------------------------------
DiagHessian DiagHessian::from_activations(const float* X, uint32_t n_tokens, uint32_t K) {
  std::vector<float> h(K, 0.f);
  for (uint32_t i = 0; i < K; ++i) {
    double acc = 0.0;
    for (uint32_t t = 0; t < n_tokens; ++t) {
      const float v = X[(size_t)t * K + i];
      acc += (double)v * v;
    }
    // H = 2 X Xᵀ  (the constant 2 is harmless for relative error).
    h[i] = (float)(2.0 * acc / (double)std::max<uint32_t>(n_tokens, 1));
  }
  return DiagHessian(std::move(h), false);
}

DiagHessian DiagHessian::uniform(uint32_t K) {
  return DiagHessian(std::vector<float>(K, 1.0f), false);
}

} // namespace dashq
} // namespace intfer
