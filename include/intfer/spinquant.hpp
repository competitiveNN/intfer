// include/intfer/spinquant.hpp
// SpinQuant orthogonal rotation preprocessing (Stage 1, t3).
//
// SpinQuant (Ashkboos et al., "LLM quantization with learned rotations", 2024)
// suppresses weight/activation outliers by multiplying weight matrices and
// activations by orthogonal rotation matrices. Concretely, a linear layer
//   y = x W^T                         (x:[T,in], W:[out,in], row-major)
// is replaced, transparently, by
//   right site (QKV / FFN_in):  W' = W  R^T,   online:  x' = x R
//   left  site (O   / FFN_out): W' = R  W,      online:  y  = y' R
// because R is orthogonal (R R^T = I) the network output is *unchanged* while
// the rotated weight/activation distributions become far more quantization
// friendly (outliers spread across dimensions).
//
// This module implements the OFFLINE preprocessing part of t3:
//   - generate the rotation matrices R (random orthogonal / Hadamard /
//     data-driven learned),
//   - fold them into the FP16 weight tensors in-place,
//   - record `rotation_id` on each TensorIR so the Stage-2 engine can replay
//     the matching online rotation during inference.
//
// It does NOT quantize (that is t4/DASH-Q) and keeps scheme == FP16.
#pragma once

#include "intfer/quant_spec.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace intfer {

// How a rotation matrix is produced.
enum class RotationType : uint8_t { Random = 0, Hadamard = 1, Learned = 2 };

// Which side of the weight the rotation multiplies.
enum class RotationSide : uint8_t { Right = 0, Left = 1 };

// A logically-shared rotation "site" in the transformer.
//   QKV    -> applied to q/k/v projections (right side, dim = hidden)
//   O      -> applied to attention output projection (left side, dim = hidden)
//   FFN_IN -> applied to gate/up projections (right side, dim = hidden)
//   FFN_OUT-> applied to down projection (left side, dim = hidden)
enum class RotationSite : uint8_t { QKV = 0, O = 1, FFN_IN = 2, FFN_OUT = 3, NONE = 4 };

// A single orthogonal rotation matrix, row-major (dim x dim).
struct Rotation {
  uint8_t     id   = 0;            // links to TensorIR::qcfg.rotation_id (0 = none)
  RotationSite site = RotationSite::NONE;
  uint32_t    dim  = 0;
  std::vector<float> mat;           // row-major dim*dim, orthogonal
};

// The full set of rotations produced by a preprocessing run.
struct RotationSet {
  std::vector<Rotation> rotations;
  const Rotation* find(uint8_t id) const;
  size_t size() const { return rotations.size(); }
};

// Optional calibration activations for the data-driven (Learned) path.
// Each entry is one tensor name paired with its row-major [n_tokens * dim] fp32
// activations (the input activation of that tensor's linear layer).
struct CalibrationData {
  std::vector<std::pair<std::string, std::vector<float>>> activations;
};

struct SpinQuantConfig {
  RotationType type = RotationType::Hadamard;  // default: fast Hadamard (pow2 dims)
  uint32_t     seed = 0x5EEDU;                 // RNG seed for Random
  bool         per_layer = false;              // distinct R per (layer, site) vs shared per site
  // Learned-rotation hyperparameters (only used when type == Learned).
  size_t learn_iters = 12;        // Jacobi/Givens sweeps over pairs
  float  learn_lr = 1.0f;         // step scale (rotation is exact, mostly cosmetic)
  float  learn_outlier_frac = 0.05f; // fraction of top-magnitude tokens weighted in proxy
  uint32_t learn_max_dim = 4096;  // beyond this, Learned falls back to Hadamard
};

// --- Rotation generators -------------------------------------------------
Rotation make_random_orthogonal(uint32_t dim, uint32_t seed);
Rotation make_hadamard(uint32_t dim);     // pow2 exact; else blockdiag(H_p, I)
Rotation learn_rotation(uint32_t dim, const std::vector<float>& activations,
                        const SpinQuantConfig& cfg);

// --- Role -> site/side mapping (see header diagram) ----------------------
RotationSite site_for_role(TensorRole role);
RotationSide side_for_role(TensorRole role);

// --- Weight application --------------------------------------------------
// W row-major [N, K]. Right: W' = W R^T.  Left: W' = R W.  In-place.
void rotate_weight(std::vector<float>& W, uint32_t N, uint32_t K,
                   const Rotation& R, RotationSide side);

// --- FP16 <-> float helpers (binary16, IEEE 754) -------------------------
uint16_t f32_to_f16(float v);
float    f16_to_f32(uint16_t h);

// Load / store a tensor's FP16 payload (in ModelIR::fp16_buf) as float.
std::vector<float> load_tensor_fp16(const ModelIR& m, const TensorIR& t);
void store_tensor_fp16(ModelIR& m, TensorIR& t, const std::vector<float>& data);

// --- Main preprocessing entry -------------------------------------------
// Consumes an FP16 ModelIR (tensors carry FP16 payloads in model.fp16_buf),
// rotates the rotatable weight tensors in place, sets qcfg.rotation_id on each
// tensor, and returns the RotationSet used (for serialization / Stage-2 online
// replay). Non-rotatable roles (EMBED/LM_HEAD/NORM/ROPE/OTHER) are passed
// through untouched.
struct SpinQuantResult {
  ModelIR     model;     // updated: fp16_buf rotated + rotation_id set
  RotationSet rotations; // id -> R
};
SpinQuantResult spinquant_preprocess(const ModelIR& model,
                                     const SpinQuantConfig& cfg,
                                     const CalibrationData* calib = nullptr);

// --- Serialization (little-endian binary) -------------------------------
// Layout: magic 'R','O','T','1' | uint32 n | per rotation:
//   uint8 id | uint8 site | uint32 dim | dim*dim float32 row-major.
std::vector<uint8_t> serialize_rotations(const RotationSet& rs);
RotationSet deserialize_rotations(const uint8_t* data, size_t n);

// Parse the layer index out of a HF tensor name ("...layers.<N>...").
// Returns -1 if no layer component is found.
int parse_layer_index(const std::string& name);

} // namespace intfer
