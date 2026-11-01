#!/usr/bin/env bash
# Shared build-first guard for gate scripts. Sourced, never executed.
#
# Usage, after the gate has defined SCRIPT_DIR/ROOT:
#     . "$SCRIPT_DIR/_gate_build.sh"
#     gate_require_build intfer_cli || exit 1
#
# gate_require_build builds $1 in $BUILD_DIR and exits non-zero if the build
# fails, so a gate can never report PASS for a stale on-disk binary. A failed
# build leaves the PREVIOUS binary in place and runnable, which is the whole
# point: without this step the gate silently tests code that no longer exists.
# Observed live in audit #4c, then found in EVERY gate that consumes a build
# artifact (audits #6/#7) -- hence one copy here, mutation-tested once, instead
# of a hand-rolled block per gate that can drift or be forgotten.
#
# Env:
#   INTFER_BUILD_DIR      build tree (default $ROOT/build)
#   INTFER_GATE_NO_BUILD  1 = skip the build; only when the binary is known current
#
# Each gate gets its own log ($BUILD_DIR/gate_<gate>_build.log): eight gates
# build intfer_cli, and a shared log file meant a concurrent pair overwrote each
# other's failure output, so `tail -20` could show the wrong compiler error.

gate_require_build() { # $1 = cmake target name
  local target="${1:-}"
  local dir="${BUILD_DIR:-${INTFER_BUILD_DIR:-$ROOT/build}}"
  local gate log
  if [ -z "$target" ]; then
    echo "FAIL: gate_require_build called with no target name" >&2
    return 1
  fi
  if [ "${INTFER_GATE_NO_BUILD:-0}" = "1" ]; then
    return 0
  fi
  if [ ! -f "$dir/build.ninja" ] && [ ! -f "$dir/Makefile" ]; then
    echo "FAIL: no build system in $dir (configure it, or set INTFER_BUILD_DIR)" >&2
    return 1
  fi
  # BASH_SOURCE[1] is the file that called us, i.e. the gate itself.
  gate="$(basename "${BASH_SOURCE[1]:-$0}" .sh)"
  log="$dir/gate_${gate}_build.log"
  if ! cmake --build "$dir" --target "$target" >"$log" 2>&1; then
    echo "FAIL: $target did not build (refusing to run a stale binary)" >&2
    tail -20 "$log" >&2
    return 1
  fi
  return 0
}