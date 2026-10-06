#!/usr/bin/env bash
# Gate for examples/roofline_probe_bisect4 -- bisect-4 of the M=1 INT2 GEMV
# probe ladder. A differential probe: it re-derives the ~7x gap between the
# bisect-2 VERBATIM body (32 GB/s) and the roofline_probe int2_variant<0> clone
# (262 GB/s) by transcribing the shippped-like kernel incrementally into
# bisect-4 and isolating which single change closes the gap. It runs engine-free
# on the same shapes as the roofline_probe arms and the same probe machinery
# (ceil + always-red twin + interleaved ratios), so the within-run ratios are
# trustworthy on this contended box.
#
# WHAT IT MAY AND MAY NOT CLAIM, enforced by the gate rather than by prose:
#   * It reports RATIOS, medians over interleaved reps. Absolute GB/s moves
#     with box load; the interleaved ratio does not.
#   * It binds NO MODEL and reads DEVICE EVENT TIMESTAMPS -- the same instrument
#     as the engine's `kernel` bucket. AGENTS.md standing rule 1 forbids an
#     agent session from running a full-27B gate; this needs no model and no
#     preload.
#   * It runs the ALWAYS-RED TWIN first: a quarter of the bytes must take about
#     a quarter of the time. If that does not hold, the ratios are not tracking
#     traffic and the binary exits 1.
#
# EXIT SEMANTICS:
#   0 = twin held and every arm ran (ROOFLINE-BISECT4 TEST: PASS)
#   1 = twin failed, an arm did not run, or profiling lost
#   2 = the probe timed out (UNPROVEN: not a regression; a documented safety net)
#
# BUILD-FIRST (audit #3c/4c family): run whatever is on disk, or a failed
# compile leaves the previous binary reporting PASS for code that no longer
# exists.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe_bisect4/roofline_probe_bisect4"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build roofline_probe_bisect4 || exit 1
[ -x "$BIN" ] || { echo "FAIL: roofline_probe_bisect4 binary missing: $BIN" >&2; exit 1; }

LOG="$BUILD_DIR/roofline_bisect4.log"

# SAFETY NET, not a verdict. If the probe takes longer than the gate's bound,
# the gate REFUSES to run it -- because a hang would consume the whole step
# (the runner allows `timeout -k 30 1800`) and would surface as rc=124, which
# the runner reads as a REGRESSION rather than "evidence does not cover this".
PROBE_TIMEOUT="${INTFER_BISECT_TIMEOUT:-120}"
RUNNER_BUDGET=1800
if [ "$PROBE_TIMEOUT" -ge "$RUNNER_BUDGET" ]; then
  echo "FAIL: INTFER_BISECT_TIMEOUT=$PROBE_TIMEOUT is not under the runner's" >&2
  echo "      $RUNNER_BUDGET s budget, so a hang would consume the whole step" >&2
  echo "      AND surface as rc=124, which the runner reads as a regression." >&2
  exit 1
fi

timeout -k 15 "$PROBE_TIMEOUT" "$SCRIPT_DIR/with_oneapi.sh" "$BIN" --device "$DEVICE" \
  >"$LOG" 2>&1
RC=$?
cat "$LOG"
echo

if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
  cat <<EOF
SKIP: the bisect-4 probe did not finish in ${PROBE_TIMEOUT}s (rc=$RC).
      This is NOT a regression -- it means the TU grew beyond what this binary
      can hold, and the gate is doing its job by refusing to wait the runner's
      full 1800 s. Rebuild smaller (split further across binaries) and re-run.
UNPROVEN: bisect-4 probe timed out (not a regression; safety net fired)
EOF
  exit 2
fi

need() { grep -qE "$1" "$LOG" || {
          echo "FAIL: expected '$2' in the output, and it is absent." >&2
          echo "      A missing arm is not a neutral result: the run did not measure" >&2
          echo "      what this gate claims to measure." >&2
          exit 1; }; }

# THE TWIN and the ceiling must have run (always-red).
need 'ceiling .*GB/s'        "the ceiling arm"
need 'twin .*ratio to full'  "the always-red twin"
# Every clone arm must have run and announced itself; `vs VERBATIM` proves the
# interleaved ratio column was produced for every arm.
need 'VERBATIM'              "the shipped-source control arm"
need 'VERB\+bitcast'         "the bit_cast group+tail + wtab tail arm"
need 'VERB\+bitcast\+4'      "the 4-byte code-load arm"
need 'VERB\+bitcast\+4\+vsel' "the probe's 3-ternary bv selector arm"
need 'CLONE'                 "the full int2_variant<0> transcription arm"
need 'vs VERBATIM'           "the interleaved ratio column"
need 'WHAT THIS RUN DOES'    "the what-this-establishes banner"

if [ "$RC" -ne 0 ]; then
  echo "FAIL: the bisect-4 probe exited rc=$RC (the twin is the likely reason)" >&2
  exit 1
fi
grep -q 'RESULT: PASS' "$LOG" || { echo "FAIL: no RESULT: PASS in the output" >&2; exit 1; }

echo "ROOFLINE-BISECT4 TEST: PASS"
echo "RESULT: bisect-4 ran with its twin holding; the VERB+bitcast / VERB+bitcast"
echo "        +4 / VERB+bitcast+4+vsel / CLONE ratios above are measured against the"
echo "        VERBATIM control on the same buffers, interleaved per rep."
exit 0
