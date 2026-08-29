#!/usr/bin/env bash
# Build script for intfer (oneAPI DPC++ / SYCL + Intel Level-Zero).
#
# Locates the oneAPI DPC++ compiler via the install tree (avoids sourcing
# setvars.sh, which calls exit and would terminate this script) and builds.
# Extra CMake arguments can be appended on the command line.
#
#   ./scripts/build.sh                 # configure + build
#   ./scripts/build.sh -DCMAKE_BUILD_TYPE=Debug
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"

# Locate the oneAPI DPC++ compiler without sourcing setvars.sh.
_COMPILER_BIN="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/bin 2>/dev/null | sort -V | tail -1)"
if [ -n "$_COMPILER_BIN" ]; then
  export PATH="$_COMPILER_BIN:$PATH"
  _COMPILER_LIB="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/lib 2>/dev/null | sort -V | tail -1)"
  export LD_LIBRARY_PATH="$_COMPILER_LIB:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-icpx}" \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER:-icx}" \
  -DINTFER_ENABLE_L0=ON \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  "$@"

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "Build complete. Binaries in: $BUILD_DIR"
