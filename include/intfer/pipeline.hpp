// include/intfer/pipeline.hpp
// t6 — INT2 quantization pipeline (orchestration of Stage-1 steps).
//
// Chains the three Stage-1 transforms into one call:
//   t3 SpinQuant  -> rotate FP16 weights, emit RotationSet
//   t4 DASH-Q     -> emit per-tensor candidate quantizations (INT2/4/8/FP16)
//   t5 RCO        -> pick one candidate/tensor under the VRAM bit-budget
// producing the finalized ModelIR (+ RotationSet + RCO report) that
// output.hpp (t6) serializes to the native SYCL loader / GGUF formats.
#pragma once

#include <cstdint>
#include <string>

#include "intfer/quant_spec.h"
#include "intfer/rco.hpp"
#include "intfer/spinquant.hpp"

namespace intfer {

// Full-pipeline configuration (defaults match the Stage-1 "canonical INT2"
// bring-up: Hadamard SpinQuant, INT2-G16 target via RCO, ~36 GiB budget).
struct PipelineConfig {
  SpinQuantConfig sq;                 // t3 rotation generator
  rco::RcoOptions rco;               // t5 bit-budget + floors
  bool learn_codebook = false;       // t4 DASH-Q learned non-uniform codebook
  bool use_calibration = false;      // use cfg.calib activations for the Hessian
  CalibrationData calib;             // optional calibration X (t4 diagonal-Hessian)
  std::string model_id = "Qwen3-32B";
  uint32_t n_layers = 0;
};

struct PipelineResult {
  ModelIR model;                     // finalized: one QuantConfig/tensor, buffers repacked
  RotationSet rotations;             // t3 emit (for Stage-2 online replay)
  rco::RcoReport rco_report;         // t5 selection report
  uint64_t fp16_baseline_bytes = 0;  // unquantized weight footprint (for compression stats)
};

// Run the full Stage-1 pipeline on a ModelIR that carries FP16 weights in
// ModelIR::fp16_buf (role + shape + rotation_id=0). Returns the finalized
// ModelIR ready for serialize_intfer()/serialize_gguf(). Throws std::runtime_error
// if a stage fails.
PipelineResult run_pipeline(const ModelIR& input, const PipelineConfig& cfg);

} // namespace intfer
