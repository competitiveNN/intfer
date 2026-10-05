#!/usr/bin/env bash
# Gate for examples/roofline_probe_bisect -- the SECOND half of the M=1 INT2 GEMV
# bisect, split into its own translation unit.
#
# WHY A SEPARATE BINARY. examples/roofline_probe measured the 5.6-6.4x
# clone-vs-shipped gap and then FALSIFIED two candidate causes. The two remaining
# single-change arms (VERB+HALF, VERB+MP) were written and never measured, because
# adding them pushed that probe's TU past the level-zero
# UR_RESULT_ERROR_OUT_OF_RESOURCES (40) limit on submit. ONE PROBE TU HOLDS ROUGHLY
# SIX TO EIGHT INT2 KERNEL BODIES. The fix is to split the ladder across two binaries
# with the `table` control and the shipped kernel shared, not to shrink the ladder
# further -- a smaller ladder answers fewer questions.
#
# WHAT IT MAY AND MAY NOT CLAIM, enforced by the gate rather than by prose:
#   * It reports RATIOS, medians over interleaved reps. Run separately, the shipped
#     arm read 17.2 / 39.3 / 41.3 GB/s on three consecutive runs of the SAME binary
#     on this contended box, so an absolute GB/s is not a result.
#   * It binds NO MODEL and reads DEVICE EVENT TIMESTAMPS -- the same instrument as
#     the engine's `kernel` bucket. AGENTS.md standing rule 1 forbids an agent
#     session from running a full-27B gate; this needs no model and no preload.
#   * It runs the ALWAYS-RED TWIN first: a quarter of the bytes must take about a
#     quarter of the time, with the same launch shape and disjoint quarters. If that
#     does not hold, the ratios are not tracking traffic and the binary exits 1.
#
# BUILD-FIRST (audit #3c/4c family): run whatever is on disk, or a failed compile
# leaves the previous binary reporting PASS for code that no longer exists.
#
# Usage: scripts/test_roofline_bisect.sh [--device N]
# Exit:  0 = twin held and every arm ran; 1 = twin failed or an arm did not run.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe_bisect/roofline_probe_bisect"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build roofline_probe_bisect || exit 1
[ -x "$BIN" ] || { echo "FAIL: roofline_probe_bisect binary missing: $BIN" >&2; exit 1; }

LOG="$BUILD_DIR/roofline_bisect.log"
timeout -k 30 1800 "$SCRIPT_DIR/with_oneapi.sh" "$BIN" --device "$DEVICE" >"$LOG" 2>&1
RC=$?
cat "$LOG"
echo

need() { grep -qE "$1" "$LOG" || {
          echo "FAIL: expected '$2' in the output, and it is absent." >&2
          echo "      A missing arm is not a neutral result: the run did not measure" >&2
          echo "      what this gate claims to measure." >&2
          exit 1; }; }

need 'twin .*ratio to full'   "the always-red twin"
need 'ceiling .*GB/s'         "the ceiling arm"
need 'vs table'               "the interleaved ratio column"
need 'VERBATIM'               "the shipped-source control arm"
need 'VERB\+HALF'             "the hardware half->float arm (VERB+HALF)"
need 'VERB\+MP'               "the multi_ptr weight-side arm (VERB+MP)"
need 'SHIPPED'                "the shipped kernel"
need 'WHAT THIS RUN DOES'     "the what-this-establishes banner"

if [ "$RC" -ne 0 ]; then
  echo "FAIL: the bisect probe exited rc=$RC (the twin is the likely reason)" >&2
  exit 1
fi
grep -q 'RESULT: PASS' "$LOG" || { echo "FAIL: no RESULT: PASS in the output" >&2; exit 1; }

echo "ROOFLINE-BISECT TEST: PASS"
echo "RESULT: the split ladder ran with its twin holding; the VERB+HALF and VERB+MP"
echo "        ratios above are the first measurements of those two arms, which the"
echo "        single-TU probe could not hold."
exit 0
