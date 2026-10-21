#!/usr/bin/env bash
# Run any command with a COMPLETE oneAPI runtime environment.
#
# NOT an L5 artifact any more, despite the header it used to carry: L5 is a
# recorded dead end, but the env bootstrap is the single most useful thing in
# scripts/ for a bare shell. `source /opt/intel/oneapi/setvars.sh` also works and
# IS what every gate does -- but it ends in `exit`, so it cannot be used to prefix
# a command, and a plain non-sourced shell cannot enumerate ANY GPU (libumf is not
# on the default LD_LIBRARY_PATH, so every UR adapter fails to load; see
# memory sycl_env_libumf).
#
# So this is the "prefix one command" form. It resolves the ACTIVE compiler
# version's lib dir only, because globbing every installed compiler is what
# produces a libur_loader/libsycl version clash -- the same reasoning, and nearly
# the same code, as scripts/run/run.sh (which execs hello_sycl) and scripts/build.sh.
# That duplication is deliberate and bounded; if you change the resolution rule,
# change all three together.
#
# Usage: scripts/with_oneapi.sh <cmd> [args...]
#   e.g. scripts/with_oneapi.sh build/examples/intfer_cli/intfer_cli --help
set -uo pipefail
ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
_ACTIVE_BIN="$(ls -dv "$ROOT"/compiler/[0-9]*/bin/icpx 2>/dev/null | sort -V | tail -1)"
_ACTIVE_LIB="$(dirname "$(dirname "$_ACTIVE_BIN")")/lib"
_new_ld=""
for d in "$ROOT"/*/*/lib "$ROOT"/*/lib; do
  [ -d "$d" ] || continue
  case "$d" in
    */compiler/*) [ "$d" = "$_ACTIVE_LIB" ] || continue ;;
  esac
  _new_ld="${_new_ld:+$_new_ld:}$d"
done
export LD_LIBRARY_PATH="${_new_ld:+$_new_ld:}${LD_LIBRARY_PATH:-}"
export PATH="$(dirname "$_ACTIVE_BIN"):$PATH"
exec "$@"
