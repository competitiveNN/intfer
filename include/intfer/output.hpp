// include/intfer/output.hpp
// t6 — final Stage-1 serialization: the finalized (post-RCO) ModelIR + the
// SpinQuant RotationSet emitted by t3 are written to disk in two interchange
// formats (docs quant_spec.md §6.5):
//
//   1. Native SYCL loader  (magic "INTFER1"): header + JSON metadata + raw
//      code/scale/global buffers + appended rotation matrices. This is what the
//      Stage-2 SYCL engine (t7+) mmaps directly; layout matches the XMX DPAS
//      expectations (codes packed, per-group FP16 scale, per-tensor FP32 alpha).
//
//   2. GGUF interop (magic "GGUF", ggml-compatible container): every TensorIR
//      becomes a GGUF tensor carrying a custom intfer quant tag
//      (INT2_G16 / INT2_G32 / INT4_G64 / INT8_G32) that encodes scheme +
//      group + scale_dtype + rotation_id + act into the ggml_type field, plus
//      auxiliary `.scale` (FP16) and `.global` (FP32 alpha) tensors and
//      `intfer.rotation.<id>.<SITE>` F32 tensors for the Stage-2 online replay.
//
// Both paths are fully round-trippable back into a ModelIR + RotationSet so the
// dequant path (t7) can be validated against the host reference.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "intfer/quant_spec.h"
#include "intfer/spinquant.hpp"

namespace intfer {

// ---------------------------------------------------------------------------
// Native SYCL loader (magic "INTFER1")
// ---------------------------------------------------------------------------
// Layout (little-endian):
//   magic[8]="INTFER1\0" | u32 version | u32 n_tensors |
//   u64 code_len | u64 scale_len | u64 global_len | u32 n_rotations | u16 reserved |
//   u64 json_len | json (UTF-8) |
//   code_buf[code_len] | scale_buf[scale_len] | global_buf[global_len] |
//   rotation region: for each rotation, dim*dim float32 (offset recorded in JSON)
std::vector<uint8_t> serialize_intfer(const ModelIR& model, const RotationSet& rs);
bool deserialize_intfer(const uint8_t* data, size_t n, ModelIR& out_model, RotationSet& out_rs);
// Zero-copy variant: payload views point into [data, data+n) instead of
// copying (saves ~22GB RAM on the 27B model when loading from mmap).
// Caller must keep the source mapping alive as long as out_model lives.
bool deserialize_intfer_borrowed(const uint8_t* data, size_t n, ModelIR& out_model,
                                 RotationSet& out_rs);

// ---------------------------------------------------------------------------
// GGUF interop (magic "GGUF", ggml-compatible container)
// ---------------------------------------------------------------------------
std::vector<uint8_t> serialize_gguf(const ModelIR& model, const RotationSet& rs);
bool deserialize_gguf(const uint8_t* data, size_t n, ModelIR& out_model, RotationSet& out_rs);

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------
bool write_file(const std::string& path, const std::vector<uint8_t>& data);
bool read_file(const std::string& path, std::vector<uint8_t>& out);

} // namespace intfer
