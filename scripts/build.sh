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
_TOOLKIT_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
_COMPILER_BIN="$(ls -dv "$_TOOLKIT_ROOT"/compiler/[0-9]*/bin 2>/dev/null | sort -V | tail -1)"
if [ -n "$_COMPILER_BIN" ]; then
  export PATH="$_COMPILER_BIN:$PATH"
  _COMPILER_LIB="$(ls -dv "$_TOOLKIT_ROOT"/compiler/[0-9]*/lib 2>/dev/null | sort -V | tail -1)"
  export LD_LIBRARY_PATH="$_COMPILER_LIB:${LD_LIBRARY_PATH:-}"
fi

# Intel Level-Zero driver (libze_loader) lives outside the toolkit tree on this
# host — find it so CMake can link the L0 layer.
_CMAKE_EXTRA=()
if [ "${INTFER_ENABLE_L0:-ON}" = "ON" ]; then
  if [ -z "${ZE_LOADER:-}" ]; then
    for _d in /usr/lib64 /usr/lib/x86_64-linux-gnu; do
      _zl="$(ls "$_d"/libze_loader.so* 2>/dev/null | head -1)"
      [ -n "$_zl" ] && _CMAKE_EXTRA+=(-DZE_LOADER="$_zl") && break
    done
  fi
fi

mkdir -p "$BUILD_DIR"

# Intel Level-Zero headers: the toolkit overlay carries them but the host /usr is
# read-only, so locate the tree wherever it lives and copy it under
# <build>/include/level_zero so <level_zero/ze_api.h> resolves via
# CMAKE_INCLUDE_PATH / LEVEL_ZERO_INCLUDE_DIR.  Sourced first so the copy is
# present before the first cmake configure; the copy is rebuilt each build so a
# stale tree cannot be read.
_LZ_INC="$BUILD_DIR/include"
mkdir -p "$_LZ_INC"
_LZ_SRC="$(find / -maxdepth 8 -path "*/include/level_zero/ze_api.h" 2>/dev/null | head -1 || true)"
if [ -n "$_LZ_SRC" ]; then
  _LZ_SRC="$(dirname "$(dirname "$_LZ_SRC")")"
  cp -a "$_LZ_SRC"/. "$_LZ_INC"/
  _CMAKE_EXTRA+=(-DCMAKE_INCLUDE_PATH="$_LZ_INC")
fi
if [ -d "$_LZ_INC" ]; then
  _CMAKE_EXTRA+=(-DLEVEL_ZERO_INCLUDE_DIR="$_LZ_INC")
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-icpx}" \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER:-icx}" \
  -DINTFER_ENABLE_L0=ON \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  "${_CMAKE_EXTRA[@]}" \
  "$@"

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "Build complete. Binaries in: $BUILD_DIR"
