#!/usr/bin/env bash
# Generic runner: sets up oneAPI LD_LIBRARY_PATH (matching run.sh) and execs its args.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
_ACTIVE_LIB="/opt/intel/oneapi/compiler/2026.1/lib"
_new_ld=""
for d in /opt/intel/oneapi/*/*/lib /opt/intel/oneapi/*/lib; do
  [ -d "$d" ] || continue
  case "$d" in
    */compiler/*) [ "$d" = "$_ACTIVE_LIB" ] || continue ;;
  esac
  _new_ld="${_new_ld:+$_new_ld:}$d"
done
export LD_LIBRARY_PATH="${_new_ld:+$_new_ld:}${LD_LIBRARY_PATH:-}"
exec "$@"
