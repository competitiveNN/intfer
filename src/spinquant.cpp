// src/spinquant.cpp
// SpinQuant orthogonal rotation preprocessing (Stage 1, t3).
// See include/intfer/spinquant.hpp for the algorithm and role/site mapping.
#include "intfer/spinquant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <numeric>
#include <random>

#include "intfer/common/logger.hpp"

namespace intfer {

bool is_pow2(uint32_t x) { return x != 0 && (x & (x - 1)) == 0; }

// Per-row symmetric per-group INT8 (maxabs) sum of squared errors, group size G.
static double sym_group_sse(const float* row, int len, int G) {
    if (len <= 0) return 0.0;
    double err = 0.0;
    int ng = (len + G - 1) / G;
    for (int g = 0; g < ng; ++g) {
        int s = g * G, e = std::min(len, s + G);
        float mx = 0.0f;
        for (int i = s; i < e; ++i) mx = std::max(mx, std::fabs(row[i]));
        if (mx == 0.0f) continue;
        float scale = mx / 127.0f;
        for (int i = s; i < e; ++i) {
            float q = std::round(row[i] / scale);
            q = std::max(-127.0f, std::min(127.0f, q));
            float d = row[i] - q * scale;
            err += static_cast<double>(d) * d;
        }
    }
    return err;
}
uint32_t floor_pow2(uint32_t x) {
  uint32_t p = 1;
  while (p * 2 <= x) p *= 2;
  return p;
}

// --- Random orthogonal via column Gram-Schmidt ---------------------------
Rotation make_random_orthogonal(uint32_t dim, uint32_t seed) {
  Rotation R;
  R.dim = dim;
  R.mat.assign(static_cast<size_t>(dim) * dim, 0.0f);
  std::mt19937 gen(seed ? seed : 1u);
  std::normal_distribution<float> g(0.0f, 1.0f);
  std::vector<float> col(dim);
  for (uint32_t j = 0; j < dim; ++j) {
    for (uint32_t i = 0; i < dim; ++i) col[i] = g(gen);
    // orthogonalize against previous columns
    for (uint32_t k = 0; k < j; ++k) {
      float dot = 0.0f;
      for (uint32_t i = 0; i < dim; ++i) dot += col[i] * R.mat[static_cast<size_t>(k) * dim + i];
      for (uint32_t i = 0; i < dim; ++i)
        col[i] -= dot * R.mat[static_cast<size_t>(k) * dim + i];
    }
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) norm += col[i] * col[i];
    norm = std::sqrt(norm) + 1e-12f;
    for (uint32_t i = 0; i < dim; ++i)
      R.mat[static_cast<size_t>(j) * dim + i] = col[i] / norm;  // store column-major in slot j
  }
  // R.mat currently column-major: convert to row-major.
  std::vector<float> row(dim * dim);
  for (uint32_t i = 0; i < dim; ++i)
    for (uint32_t j = 0; j < dim; ++j)
      row[static_cast<size_t>(i) * dim + j] = R.mat[static_cast<size_t>(j) * dim + i];
  R.mat = std::move(row);
  return R;
}

// --- Walsh-Hadamard (exact for pow2; blockdiag(H_p, I) otherwise) --------
void hadamard_pow2(uint32_t n, std::vector<float>& H) {
  H.assign(static_cast<size_t>(n) * n, 0.0f);
  if (n == 1) { H[0] = 1.0f; return; }
  std::vector<float> Hh;
  hadamard_pow2(n / 2, Hh);
  const float inv = 1.0f;  // we normalize at the end
  for (uint32_t i = 0; i < n / 2; ++i)
    for (uint32_t j = 0; j < n / 2; ++j) {
      float v = Hh[static_cast<size_t>(i) * (n / 2) + j];
      H[static_cast<size_t>(i) * n + j]         = v;
      H[static_cast<size_t>(i) * n + (j + n / 2)] = v;
      H[static_cast<size_t>(i + n / 2) * n + j]     = v;
      H[static_cast<size_t>(i + n / 2) * n + (j + n / 2)] = -v;
    }
  (void)inv;
}

Rotation make_hadamard(uint32_t dim) {
  Rotation R;
  R.dim = dim;
  R.mat.assign(static_cast<size_t>(dim) * dim, 0.0f);
  if (is_pow2(dim)) {
    std::vector<float> H;
    hadamard_pow2(dim, H);
    const float s = 1.0f / std::sqrt(static_cast<float>(dim));
    for (size_t i = 0; i < H.size(); ++i) R.mat[i] = H[i] * s;
  } else {
    uint32_t p = floor_pow2(dim);
    std::vector<float> H;
    hadamard_pow2(p, H);
    const float s = 1.0f / std::sqrt(static_cast<float>(p));
    for (uint32_t i = 0; i < dim; ++i) {
      if (i < p) {
        for (uint32_t j = 0; j < p; ++j)
          R.mat[static_cast<size_t>(i) * dim + j] = H[static_cast<size_t>(i) * p + j] * s;
      } else {
        R.mat[static_cast<size_t>(i) * dim + i] = 1.0f;  // identity tail
      }
    }
    INTFER_LOG_WARN("SpinQuant: dim %u not power-of-two; used blockdiag(Hadamard_%u, I_%u).",
                    dim, p, dim - p);
  }
  R.is_hadamard = true;
  return R;
}

// --- Learned rotation: weighted Jacobi outlier-suppression ----------------
// Minimizes the energy of large-magnitude activation tokens after an orthogonal
// transform (a lightweight, data-driven stand-in for full SpinQuant learned
// rotations). Returns an orthogonal dim x dim matrix.
Rotation learn_rotation(uint32_t dim, const std::vector<float>& acts,
                        const SpinQuantConfig& cfg) {
  Rotation R;
  R.dim = dim;
  R.mat.assign(static_cast<size_t>(dim) * dim, 0.0f);
  for (uint32_t i = 0; i < dim; ++i) R.mat[static_cast<size_t>(i) * dim + i] = 1.0f;

  if (acts.empty() || dim == 0) {
    INTFER_LOG_WARN("SpinQuant learned rotation: no calibration; falling back to Hadamard.");
    return make_hadamard(dim);
  }
  if (dim > cfg.learn_max_dim) {
    INTFER_LOG_WARN("SpinQuant learned rotation: dim %u > learn_max_dim %u; falling back to Hadamard.",
                    dim, cfg.learn_max_dim);
    return make_hadamard(dim);
  }

  const uint32_t T = static_cast<uint32_t>(acts.size() / dim);
  // Working rotated activations (copy; we rotate columns in place).
  std::vector<float> A = acts;
  // Outlier token weights: emphasize rows whose largest entry is large.
  std::vector<float> w(T, 0.0f);
  for (uint32_t t = 0; t < T; ++t) {
    float mx = 0.0f;
    for (uint32_t j = 0; j < dim; ++j) {
      float a = std::fabs(A[static_cast<size_t>(t) * dim + j]);
      if (a > mx) mx = a;
    }
    w[t] = mx * mx;
  }

  const size_t D = static_cast<size_t>(dim);
  for (size_t sweep = 0; sweep < cfg.learn_iters; ++sweep) {
    for (uint32_t i = 0; i < dim; ++i) {
      for (uint32_t j = i + 1; j < dim; ++j) {
        double s00 = 0, s11 = 0, s01 = 0;
        for (uint32_t t = 0; t < T; ++t) {
          float u = A[static_cast<size_t>(t) * D + i];
          float v = A[static_cast<size_t>(t) * D + j];
          float wt = w[t];
          s00 += wt * u * u;
          s11 += wt * v * v;
          s01 += wt * u * v;
        }
        // Jacobi angle minimizing weighted covariance off-diagonal.
        double theta = 0.5 * std::atan2(2.0 * s01, s00 - s11);
        float c = static_cast<float>(std::cos(theta)) * cfg.learn_lr;
        // ensure orthonormal: renormalize if learn_lr<1 handled by clamp below
        float s = static_cast<float>(std::sin(theta)) * cfg.learn_lr;
        float r = std::sqrt(c * c + s * s);
        if (r < 1e-6f) continue;
        c /= r; s /= r;
        // rotate columns i,j of A and of R (right multiply by G(i,j)).
        for (uint32_t t = 0; t < T; ++t) {
          size_t bi = static_cast<size_t>(t) * D + i;
          size_t bj = static_cast<size_t>(t) * D + j;
          float u = A[bi], v = A[bj];
          A[bi] = c * u + s * v;
          A[bj] = -s * u + c * v;
        }
        for (uint32_t row = 0; row < dim; ++row) {
          size_t bi = static_cast<size_t>(row) * D + i;
          size_t bj = static_cast<size_t>(row) * D + j;
          float u = R.mat[bi], v = R.mat[bj];
          R.mat[bi] = c * u + s * v;
          R.mat[bj] = -s * u + c * v;
        }
      }
    }
  }

  // Never regress: keep the rotation with the smaller symmetric-INT8 quantization
  // proxy on the (rotated) calibration activations. Learned may tie or beat
  // Hadamard; it must not be worse than the cheap baseline.
  auto proxy = [&](const std::vector<float>& M) -> double {
    double tot = 0.0;
    for (uint32_t t = 0; t < T; ++t)
      tot += sym_group_sse(&M[static_cast<size_t>(t) * D], static_cast<int>(dim), 64);
    return tot;
  };
  double learned_proxy = proxy(A);
  Rotation Rh = make_hadamard(dim);
  std::vector<float> Ah = acts;
  for (uint32_t t = 0; t < T; ++t)
    for (uint32_t j = 0; j < dim; ++j) {
      float s = 0.0f;
      for (uint32_t k = 0; k < dim; ++k) s += Ah[static_cast<size_t>(t) * D + k] * Rh.mat[static_cast<size_t>(j) * dim + k];
      Ah[static_cast<size_t>(t) * D + j] = s;
    }
  if (proxy(Ah) < learned_proxy) return Rh;
  return R;
}

// --- IEEE 754 binary16 <-> float32 (public, matches spinquant.hpp) -------
uint16_t f32_to_f16(float v) {
  // Fast, correct-enough conversion (round-to-nearest-even via bit tricks).
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  uint32_t sign = (f >> 16) & 0x8000u;
  int32_t  e    = static_cast<int32_t>((f >> 23) & 0xff) - 127 + 15;
  uint32_t m    = f & 0x7fffffu;
  if (e <= 0) {  // subnormal or zero / underflow
    if (e < -10) return static_cast<uint16_t>(sign);
    m |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - e);
    uint32_t out = m >> shift;
    uint32_t rem = m & ((1u << shift) - 1);
    if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (out & 1)))
      out += 1;  // round to nearest even
    return static_cast<uint16_t>(sign | out);
  }
  if (e >= 31) {  // overflow / inf / nan
    if ((f & 0x7fffffu) == 0) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | 0x7c00u | (m >> 13));
  }
  uint32_t out = (static_cast<uint32_t>(e) << 10) | (m >> 13);
  uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (out & 1))) out += 1;  // nearest even
  return static_cast<uint16_t>(sign | out);
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t e    = (h >> 10) & 0x1fu;
  uint32_t m    = h & 0x3ffu;
  uint32_t f;
  if (e == 0) {
    if (m == 0) { f = sign; }
    else {
      uint32_t em = m;
      int32_t sh = 1;
      while ((em & 0x400u) == 0) { em <<= 1; --sh; }
      em &= 0x3ffu;
      f = sign | ((127 - 15 - sh) << 23) | (em << 13);
    }
  } else if (e == 31) {
    f = sign | 0x7f800000u | (m << 13);
  } else {
    f = sign | ((e + 127 - 15) << 23) | (m << 13);
  }
  float v;
  std::memcpy(&v, &f, sizeof(v));
  return v;
}

// --- public: role mappings ----------------------------------------------
RotationSite site_for_role(TensorRole role) {
  switch (role) {
    case TensorRole::ATTN_Q:
    case TensorRole::ATTN_K:
    case TensorRole::ATTN_V:
    case TensorRole::LINEAR_ATTN_QKV:
    case TensorRole::LINEAR_ATTN_Z:
    case TensorRole::LINEAR_ATTN_B:
    case TensorRole::LINEAR_ATTN_A:   return RotationSite::QKV;
    case TensorRole::ATTN_O:
    case TensorRole::LINEAR_ATTN_OUT: return RotationSite::O;
    case TensorRole::MLP_GATE:
    case TensorRole::MLP_UP:   return RotationSite::FFN_IN;
    case TensorRole::MLP_DOWN: return RotationSite::FFN_OUT;
    default:                   return RotationSite::NONE;  // no rotation (embed/lm_head/norm/rope/conv/other)
  }
}
RotationSide side_for_role(TensorRole role) {
  RotationSite s = site_for_role(role);
  return (s == RotationSite::O || s == RotationSite::FFN_OUT) ? RotationSide::Left : RotationSide::Right;
}

// --- public: weight rotation --------------------------------------------
namespace {
// In-place unscaled Walsh-Hadamard transform (+-1 entries), dim must be pow2.
void fwht(float* x, uint32_t n) {
  for (uint32_t len = 2; len <= n; len <<= 1) {
    uint32_t half = len >> 1;
    for (uint32_t i = 0; i < n; i += len)
      for (uint32_t j = 0; j < half; ++j) {
        float u = x[i + j], v = x[i + j + half];
        x[i + j] = u + v; x[i + j + half] = u - v;
      }
  }
}
} // namespace

void rotate_weight(std::vector<float>& W, uint32_t N, uint32_t K,
                   const Rotation& R, RotationSide side) {
  const uint32_t dim = (side == RotationSide::Right) ? K : N;
  if (R.dim != dim) {
    INTFER_LOG_ERROR("SpinQuant rotate_weight: rotation dim %u != expected %u", R.dim, dim);
    return;
  }
  // Fast path: Hadamard rotations. For power-of-two dims use the
  // Walsh-Hadamard transform (O(dim log dim)). For the blockdiag
  // fallback (make_hadamard splits off a power-of-two tail into I),
  // apply FWHT to the power-of-two prefix and leave the tail as-is.
  if (R.is_hadamard) {
    const uint32_t p = floor_pow2(dim);
    const bool is_blockdiag = !is_pow2(dim) && p < dim;
    const float inv = 1.0f / std::sqrt(static_cast<float>(p));
    if (side == RotationSide::Right) {
      #pragma omp parallel for schedule(static) if((size_t)N * p >= 65536)
      for (int64_t i = 0; i < (int64_t)N; ++i) {
        float* row = &W[static_cast<size_t>(i) * K];
        if (is_blockdiag) {
          // blockdiag(Hadamard_p, I_{dim-p}): FWHT on first p cols,
          // leave cols [p..dim) untouched.
          std::vector<float> tmp(p);
          std::copy_n(row, p, tmp.data());
          fwht(tmp.data(), p);
          for (uint32_t j = 0; j < p; ++j) row[j] = tmp[j] * inv;
        } else {
          std::vector<float> tmp(dim);
          std::copy_n(row, dim, tmp.data());
          fwht(tmp.data(), dim);
          for (uint32_t j = 0; j < dim; ++j) row[j] = tmp[j] * inv;
        }
      }
    } else {
      #pragma omp parallel for schedule(static) if((size_t)K * p >= 65536)
      for (int64_t k = 0; k < (int64_t)K; ++k) {
        if (is_blockdiag) {
          std::vector<float> tmp(p);
          for (uint32_t i = 0; i < p; ++i) tmp[i] = W[static_cast<size_t>(i) * K + k];
          fwht(tmp.data(), p);
          for (uint32_t i = 0; i < p; ++i) W[static_cast<size_t>(i) * K + k] = tmp[i] * inv;
        } else {
          std::vector<float> tmp(dim);
          for (uint32_t i = 0; i < dim; ++i) tmp[i] = W[static_cast<size_t>(i) * K + k];
          fwht(tmp.data(), dim);
          for (uint32_t i = 0; i < dim; ++i) W[static_cast<size_t>(i) * K + k] = tmp[i] * inv;
        }
      }
    }
  } else {
    // General (non-Hadamard) path: naive matmul.
    // Parallelize over the OUTER i dimension (rows of W). With 5k+ rows and
    // 5120-element inner accumulation, this gives near-linear scaling up to the
    // core count. Each thread writes only into its own `out[i*K .. i*K+K]`,
    // so no synchronization is needed.
    std::vector<float> out(W.size(), 0.0f);
    if (side == RotationSide::Right) {
      // W' = W R^T : out[i,k] = sum_j W[i,j] * R[k,j]
      #pragma omp parallel for schedule(static) if((size_t)N * K >= 65536)
      for (int64_t i = 0; i < (int64_t)N; ++i) {
        for (uint32_t k = 0; k < K; ++k) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < K; ++j)
          acc += W[static_cast<size_t>(i) * K + j] * R.mat[static_cast<size_t>(k) * K + j];
          out[static_cast<size_t>(i) * K + k] = acc;
        }
      }
    } else {
      // W' = R W : out[i,k] = sum_j R[i,j] * W[j,k]
      #pragma omp parallel for schedule(static) if((size_t)N * K >= 65536)
      for (int64_t i = 0; i < (int64_t)N; ++i) {
        for (uint32_t k = 0; k < K; ++k) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < N; ++j)
          acc += R.mat[static_cast<size_t>(i) * N + j] * W[static_cast<size_t>(j) * K + k];
          out[static_cast<size_t>(i) * K + k] = acc;
        }
      }
    }
    W = std::move(out);
  }
}

// --- public: fp16 helpers ------------------------------------------------
std::vector<float> load_tensor_fp16(const ModelIR& m, const TensorIR& t) {
  std::vector<float> out;
  if (t.fp16_bytes == 0) return out;
  size_t n = t.fp16_bytes / 2;
  out.resize(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = f16_to_f32(m.fp16_buf[t.fp16_offset + i]);
  return out;
}
void store_tensor_fp16(ModelIR& m, TensorIR& t, const std::vector<float>& data) {
  const size_t n = data.size();
  const uint64_t need = static_cast<uint64_t>(n) * 2;
  if (t.fp16_bytes == need && t.fp16_offset + need <= m.fp16_buf.size()) {
    for (size_t i = 0; i < n; ++i) m.fp16_buf[t.fp16_offset + i] = f32_to_f16(data[i]);
  } else {
    t.fp16_offset = m.fp16_buf.size();
    t.fp16_bytes = need;
    m.fp16_buf.resize(t.fp16_offset + n);
    for (size_t i = 0; i < n; ++i) m.fp16_buf[t.fp16_offset + i] = f32_to_f16(data[i]);
  }
}

int parse_layer_index(const std::string& name) {
  // Match the canonical HF prefix "model.language_model.layers." exactly.
  // We intentionally do NOT match the speculative "mtp.layers." decoder
  // (a separate model in Qwen3.5+) nor "model.layers." (older formats).
  static const std::string key = "model.language_model.layers.";
  size_t pos = name.find(key);
  if (pos == std::string::npos) return -1;
  pos += key.size();
  size_t end = name.find('.', pos);
  if (end == std::string::npos) end = name.size();
  try {
    return std::stoi(name.substr(pos, end - pos));
  } catch (...) { return -1; }
}

const Rotation* RotationSet::find(uint8_t id) const {
  for (const auto& r : rotations)
    if (r.id == id) return &r;
  return nullptr;
}

// --- main entry ----------------------------------------------------------
SpinQuantResult spinquant_preprocess(const ModelIR& model,
                                     const SpinQuantConfig& cfg,
                                     const CalibrationData* calib) {
  SpinQuantResult res;
  res.model = model;  // copy metadata + fp16 payload

  // name -> role, for resolving calibration activations.
  std::map<std::string, TensorRole> name_role;
  for (const auto& t : model.tensors) name_role[t.name] = t.role;

  // Decide shared vs per-layer, and assign ids within uint8 (0 = none).
  bool per_layer = cfg.per_layer;
  int max_layer = 0;
  for (const auto& t : model.tensors) {
    int l = parse_layer_index(t.name);
    if (l > max_layer) max_layer = l;
  }
  if (per_layer) {
    uint32_t needed = static_cast<uint32_t>(max_layer + 1) * 4 + 1;  // +1 for id 0
    if (needed > 255) {
      INTFER_LOG_WARN("SpinQuant: per-layer needs %u ids (>255); falling back to shared-per-site.",
                      needed);
      per_layer = false;
    }
  }

  // key -> rotation builder.
  struct Key { RotationSite site; int layer; bool operator<(const Key& o) const {
    return site < o.site || (site == o.site && layer < o.layer); } };

  std::map<Key, Rotation> built;
  // Aggregate calibration activations per key (for Learned).
  std::map<Key, std::vector<float>> calib_acc;
  if (calib) {
    for (const auto& [name, acts] : calib->activations) {
      auto it = name_role.find(name);
      if (it == name_role.end()) continue;
      RotationSite site = site_for_role(it->second);
      if (site == RotationSite::NONE) continue;
      int layer = parse_layer_index(name);
      if (layer < 0) layer = 0;
      Key k{site, per_layer ? layer : 0};
      auto& v = calib_acc[k];
      v.insert(v.end(), acts.begin(), acts.end());
    }
  }

  auto get_rotation = [&](RotationSite site, int layer, uint32_t dim) -> const Rotation& {
    Key k{site, per_layer ? layer : 0};
    auto it = built.find(k);
    if (it != built.end()) {
      if (it->second.dim != dim)
        INTFER_LOG_ERROR("SpinQuant: inconsistent rotation dim %u vs %u for site %d",
                        it->second.dim, dim, (int)site);
      return it->second;
    }
    Rotation R;
    switch (cfg.type) {
      case RotationType::Random:  R = make_random_orthogonal(dim, cfg.seed + (per_layer ? (uint32_t)(layer*4+(int)site) : (uint32_t)site)); break;
      case RotationType::Hadamard: R = make_hadamard(dim); break;
      case RotationType::Learned:  R = learn_rotation(dim, calib_acc.count(k) ? calib_acc[k] : std::vector<float>{}, cfg); break;
    }
    // The generators reset R; restore the metadata the caller relies on.
    R.site = site;
    R.dim = dim;
    // assign id
    uint8_t id;
    if (per_layer) id = static_cast<uint8_t>(1 + layer * 4 + static_cast<int>(site));
    else           id = static_cast<uint8_t>(1 + static_cast<int>(site));
    R.id = id;
    auto ins = built.emplace(k, std::move(R));
    return ins.first->second;
  };

  // Apply to each rotatable tensor.
  for (auto& t : res.model.tensors) {
    RotationSite site = site_for_role(t.role);
    if (site == RotationSite::NONE) continue;  // pass through (embed/lm_head/norm/rope/other)
    RotationSide side = side_for_role(t.role);
    uint32_t dim = (side == RotationSide::Right) ? t.K : t.N;
    int layer = parse_layer_index(t.name);
    if (layer < 0) layer = 0;
    const Rotation& R = get_rotation(site, layer, dim);
    std::vector<float> W = load_tensor_fp16(res.model, t);
    if (W.empty()) {
      INTFER_LOG_WARN("SpinQuant: tensor '%s' has no FP16 payload; skipped.", t.name.c_str());
      continue;
    }
    rotate_weight(W, t.N, t.K, R, side);
    store_tensor_fp16(res.model, t, W);
    t.qcfg.rotation_id = R.id;  // recorded for Stage-2 online replay
  }

  // Build RotationSet.
  for (auto& kv : built) res.rotations.rotations.push_back(std::move(kv.second));
  return res;
}

// --- serialization -------------------------------------------------------
std::vector<uint8_t> serialize_rotations(const RotationSet& rs) {
  std::vector<uint8_t> out;
  auto put32 = [&](uint32_t x) { for (int i = 0; i < 4; ++i) out.push_back((x >> (8*i)) & 0xff); };
  auto putf  = [&](float f) { uint32_t u; std::memcpy(&u, &f, 4); put32(u); };
  out.push_back('R'); out.push_back('O'); out.push_back('T'); out.push_back('1');
  put32(static_cast<uint32_t>(rs.rotations.size()));
  for (const auto& r : rs.rotations) {
    out.push_back(r.id);
    out.push_back(static_cast<uint8_t>(r.site));
    put32(r.dim);
    for (uint32_t i = 0; i < r.dim * r.dim; ++i) putf(r.mat[i]);
  }
  return out;
}
RotationSet deserialize_rotations(const uint8_t* data, size_t n) {
  RotationSet rs;
  if (n < 8 || data[0] != 'R' || data[1] != 'O' || data[2] != 'T' || data[3] != '1')
    return rs;
  auto get32 = [&](size_t o) -> uint32_t {
    uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(data[o + i]) << (8 * i);
    return v; };
  size_t off = 4;
  uint32_t cnt = get32(off); off += 4;
  for (uint32_t c = 0; c < cnt; ++c) {
    Rotation r;
    r.id = data[off++];
    r.site = static_cast<RotationSite>(data[off++]);
    r.dim = get32(off); off += 4;
    r.mat.resize(static_cast<size_t>(r.dim) * r.dim);
    for (uint32_t i = 0; i < r.dim * r.dim; ++i) {
      uint32_t u = get32(off); off += 4;
      std::memcpy(&r.mat[i], &u, 4);
    }
    rs.rotations.push_back(std::move(r));
  }
  return rs;
}

} // namespace intfer
