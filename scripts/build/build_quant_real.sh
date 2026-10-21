#!/usr/bin/env bash
# Build the host-only Stage-1 quantization harness that runs on a REAL model
# (no SYCL / oneAPI toolchain needed — Stage-1 is pure C++).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
OUT="${INTFER_QUANT_REAL_BIN:-$ROOT/build_quant_real/quant_model_real}"

SRC=(
  src/common/logger.cpp
  src/quant_spec.cpp
  src/spinquant.cpp
  src/quant/dashq.cpp
  src/quant/rco.cpp
  src/quant/pipeline.cpp
  src/quant/output.cpp
  src/quant/safetensors.cpp
  examples/quant_model_real/main.cpp
)

mkdir -p "$(dirname "$OUT")"

# -fopenmp enables the multi-threaded DASH-Q group/tensor loops
g++ -std=c++20 -O2 -pthread -fopenmp -I "$ROOT/include" "${SRC[@]}" -o "$OUT"
echo "Built: $OUT"
