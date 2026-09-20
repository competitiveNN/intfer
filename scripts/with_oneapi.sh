#!/usr/bin/env bash
# L5 session helper: full oneAPI LD_LIBRARY_PATH + run intfer_cli.
set -uo pipefail
_ACTIVE_BIN="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/bin/icpx 2>/dev/null | sort -V | tail -1)"
_ACTIVE_LIB="$(dirname "$(dirname "$_ACTIVE_BIN")")/lib"
_new_ld=""
for d in /opt/intel/oneapi/*/*/lib /opt/intel/oneapi/*/lib; do
  [ -d "$d" ] || continue
  case "$d" in
    */compiler/*) [ "$d" = "$_ACTIVE_LIB" ] || continue ;;
  esac
  _new_ld="${_new_ld:+$_new_ld:}$d"
done
export LD_LIBRARY_PATH="${_new_ld:+$_new_ld:}${LD_LIBRARY_PATH:-}"
export PATH="$(dirname "$_ACTIVE_BIN"):$PATH"
exec "$@"
