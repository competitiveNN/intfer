#!/usr/bin/env bash
# Run the intfer hello-SYCL binary.
#
# The Level-Zero SYCL plugin and its dependencies (libze_loader, umf, tbb, ...)
# must be resolvable at runtime. We build LD_LIBRARY_PATH from the oneAPI
# library directories, but only the ACTIVE compiler version's lib dir (to avoid
# a libur_loader/libsycl version clash when several compiler versions are
# installed). We deliberately do NOT source setvars.sh: it ends with `exit`,
# which would terminate this script.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/hello_sycl/hello_sycl"

# Resolve the active oneAPI compiler version (same logic as scripts/build.sh).
_ACTIVE_BIN="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/bin/icpx 2>/dev/null | sort -V | tail -1)"
_ACTIVE_LIB="$(dirname "$(dirname "$_ACTIVE_BIN")")/lib"

_new_ld=""
for d in /opt/intel/oneapi/*/*/lib /opt/intel/oneapi/*/lib; do
  [ -d "$d" ] || continue
  # Only the active compiler version's lib dir; skip the others.
  case "$d" in
    */compiler/*) [ "$d" = "$_ACTIVE_LIB" ] || continue ;;
  esac
  _new_ld="${_new_ld:+$_new_ld:}$d"
done
export LD_LIBRARY_PATH="${_new_ld:+$_new_ld:}${LD_LIBRARY_PATH:-}"

exec "$BIN" "$@"
