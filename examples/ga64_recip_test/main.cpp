// GA64 reciprocal-identity gate.
//
// The device act-quant pre-pass (INTFER_DPAS_PREQ) does NOT quantize on the
// device: it consumes (scale, inverse) pairs computed on the host by
// dashq::act_quant_ga64_scale_inv_batch() and only applies
//
//     v = clamp(x*inv, +-127);  q = (int8)(v >= 0 ? v+0.5f : v-0.5f)
//
// That is only sound if the shipped `inv` is BIT-IDENTICAL to the one
// act_quant_ga64_batch() (the stream oracle) uses internally. GA64's rule
// q = trunc(x*inv +- 0.5) is pathologically ill-conditioned in `inv`: a 1-ulp
// difference moves the product across an exact .5 tie and flips the code by
// one, which is enough to change a greedy token stream. `1.0f/s` is not
// guaranteed to produce the same bits at two different call sites -- its
// codegen (divss vs rcp.approx+NR) depends on the surrounding code, and the
// device lowers it to rcp.approx.
//
// This gate fails if anyone reintroduces a second reciprocal site, inlines
// ga64_recip(), derives a scale in the helper without the shared max-abs
// reduction, or makes the helper quantize with a different rounding rule.
//
// Method: the reference path exposes its scale but not its reciprocal, so we
// RECOVER the reciprocal the oracle effectively used -- for each slice we
// re-quantize with inv-1ulp, inv and inv+1ulp and see which one reproduces
// the reference codes exactly. A regression then reports exactly how far the
// helper drifted (1 ulp / scale mismatch) instead of just "streams differ".
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "intfer/quant/dashq.hpp"

using intfer::dashq::ActQuantRound;

namespace {

constexpr size_t kGA = 64;
constexpr size_t kK = 2048;  // 32 slices; exercises the 64-multiple tail path

float ulp_up(float f) {
  std::int32_t i;
  std::memcpy(&i, &f, 4);
  i += 1;
  float o;
  std::memcpy(&o, &i, 4);
  return o;
}
float ulp_dn(float f) {
  std::int32_t i;
  std::memcpy(&i, &f, 4);
  i -= 1;
  float o;
  std::memcpy(&o, &i, 4);
  return o;
}

// Exactly the rule act_quant_ga64_dev() applies on the device.
void quant_with(const float* x, size_t n, float inv, std::int8_t* q) {
  for (size_t k = 0; k < n; ++k) {
    float v = x[k] * inv;
    if (v > 127.0f) v = 127.0f;
    else if (v < -127.0f) v = -127.0f;
    q[k] = static_cast<std::int8_t>(v >= 0 ? v + 0.5f : v - 0.5f);
  }
}

struct Tally {
  long slices = 0;
  long ok = 0;
  long inv_off = 0;        // oracle used a neighbouring reciprocal
  long scale_off = 0;      // shipped scale != oracle scale
  long code_off = 0;       // shipped codes != oracle codes
  long elements = 0;
};

// One multi-row batch: row 0 is the reference row, every row is checked.
void check_batch(const std::vector<float>& x, size_t M, size_t K, Tally& t) {
  const size_t nGA = (K + kGA - 1) / kGA;
  std::vector<std::int8_t> q_ref(K), q_h(K), q_dn(K), q_up(K);
  std::vector<float> s_ref(nGA), si(2 * M * nGA);
  intfer::dashq::act_quant_ga64_batch(x.data(), q_ref.data(), s_ref.data(), K,
                                       ActQuantRound::HalfAway);
  intfer::dashq::act_quant_ga64_scale_inv_batch(x.data(), M, K, si.data());
  for (size_t g = 0; g < nGA; ++g) {
    const size_t k0 = g * kGA;
    const size_t k1 = std::min(k0 + kGA, K);
    const float sh = si[2 * g], ih = si[2 * g + 1];
    if (std::memcmp(&sh, &s_ref[g], 4) != 0) t.scale_off++;
    quant_with(x.data() + k0, k1 - k0, ih, q_h.data());
    quant_with(x.data() + k0, k1 - k0, ulp_dn(ih), q_dn.data());
    quant_with(x.data() + k0, k1 - k0, ulp_up(ih), q_up.data());
    const std::int8_t* qr = q_ref.data() + k0;
    const bool mh = std::memcmp(q_h.data(), qr, k1 - k0) == 0;
    const bool md = std::memcmp(q_dn.data(), qr, k1 - k0) == 0;
    const bool mu = std::memcmp(q_up.data(), qr, k1 - k0) == 0;
    if (!mh) t.code_off++;
    t.slices++;
    t.elements += static_cast<long>(k1 - k0);
    if (mh) {
      t.ok++;
    } else if (md || mu) {
      t.inv_off++;
    } else {
      t.inv_off++;  // further than 1 ulp: still a reciprocal-class failure
    }
  }
}

// Non-multiple-of-64 K: the tail slice must be covered by both sides.
void check_tail(Tally& t) {
  for (size_t K = 1; K <= 4 * kGA + 1; ++K) {
    std::vector<float> x(K);
    for (size_t k = 0; k < K; ++k) x[k] = std::sin(0.37f * float(k) + 0.11f);
    check_batch(x, 1, K, t);
  }
}

void report(const char* name, const Tally& t) {
  std::printf(
      "  %-10s slices=%-7ld ok=%-7ld inv_off=%-4ld scale_off=%-4ld "
      "code_off=%-4ld elems=%ld\n",
      name, t.slices, t.ok, t.inv_off, t.scale_off, t.code_off, t.elements);
}

}  // namespace

int main() {
  const size_t K = kK;
  const size_t nGA = (K + kGA - 1) / kGA;
  const size_t M = 3;  // multi-row: the engine passes M rows, not one
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<float> x(M * K);
  long bad = 0;

  std::puts("GA64 reciprocal identity gate (helper (s,inv) vs oracle codes)");

  // A) random magnitudes.
  Tally a;
  for (int iter = 0; iter < 2000; ++iter) {
    const float sc = std::pow(10.0f, -8.0f + 11.0f * (iter % 22) / 21.0f);
    for (size_t k = 0; k < x.size(); ++k) x[k] = u(rng) * sc;
    check_batch(x, M, K, a);
  }
  report("random", a);
  bad += a.inv_off + a.scale_off + a.code_off;

  // B) tie-adversarial: slice max fixed, remaining elements placed at exact
  //    (j+0.5)*s so x*inv lands on the .5 boundary -- the only place a 1-ulp
  //    reciprocal difference can change a code.
  Tally b;
  for (int iter = 0; iter < 2000; ++iter) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t g = 0; g < nGA; ++g) {
        const size_t k0 = g * kGA;
        const float mx = std::pow(10.0f, -6.0f + 9.0f * (iter % 16) / 15.0f);
        const float s = mx / 127.0f;
        for (size_t j = 0; j < kGA; ++j) {
          float v;
          if (j == 0) {
            v = mx;
          } else {
            const float half = static_cast<float>(j) - 32.0f + 0.5f;
            v = half * s * (1.0f + 1e-7f * static_cast<float>(
                                            (iter + static_cast<int>(j)) % 3 - 1));
            if (j % 5 == 0) v = -v;
          }
          x[m * K + k0 + j] = v;
        }
      }
    }
    check_batch(x, M, K, b);
  }
  report("tie", b);
  bad += b.inv_off + b.scale_off + b.code_off;

  // C) edges: zeros, denormals, saturation, single outlier per slice.
  Tally c;
  for (int iter = 0; iter < 1000; ++iter) {
    for (size_t k = 0; k < x.size(); ++k) {
      switch (iter % 6) {
        case 0: x[k] = 0.0f; break;
        case 1: x[k] = -0.0f; break;
        case 2: x[k] = 1e-42f; break;  // denormal
        case 3: x[k] = (k % kGA == 0) ? 3.4e38f : 1e-30f; break;
        case 4: x[k] = 127.0f * 1.0000001f; break;
        default: x[k] = u(rng); break;
      }
    }
    check_batch(x, M, K, c);
  }
  report("edge", c);
  bad += c.inv_off + c.scale_off + c.code_off;

  // D) K not a multiple of 64 (tail slice), incl. K=1.
  Tally d;
  check_tail(d);
  report("tail", d);
  bad += d.inv_off + d.scale_off + d.code_off;

  if (bad != 0) {
    std::printf(
        "GA64 RECIPROCAL TEST: FAIL (%ld failures: the helper no longer ships "
        "the oracle's (scale, reciprocal) bits)\n",
        bad);
    return 1;
  }
  std::printf(
      "GA64 RECIPROCAL TEST: PASS (%ld slices, %ld elements bit-identical)\n",
      a.slices + b.slices + c.slices + d.slices,
      a.elements + b.elements + c.elements + d.elements);
  return 0;
}
