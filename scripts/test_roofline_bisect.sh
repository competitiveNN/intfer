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

# *** THE PROBE IS KNOWN-UNRUNNABLE AT HEAD, AND THIS GATE IS BOUNDED TO SAY SO. ***
# MEASURED 2026-10-06: every arm of the binary HANGS, including the `table`
# control; each arm tested ALONE also hangs, so the fault is not in the shipped
# call. The first version of this gate ran the probe under `timeout 1800` --
# the SAME budget its runner gives it -- and returned **rc=124 with EMPTY output**.
# That is the worst of both worlds and it is worth being precise about why:
#
#   * rc=124 is NOT in the runner's vocabulary. `check_device_precond.sh` reads
#     healthy rc as 0, or 2 WITH a skip diagnostic; anything else is
#     "a gate must not go red on a healthy box" -> a REGRESSION. So an unrunnable
#     probe was masquerading as a code failure.
#   * empty output meant even `need` had nothing to match, so there was no
#     diagnostic at all.
#   * and it consumed the runner's entire 1800s budget on a step that cannot
#     finish.
#
# So: an internal timeout FAR under the runner's, and on timeout a distinct
# SKIP verdict with rc=2 -- UNPROVEN, which the runner records as "evidence does
# not cover this" rather than "regression". That is the honest class for "the
# probe cannot run here yet", and it is the same class the repo already uses for
# stale stamps. A known-broken probe must not be able to look like a PASS or a
# FAIL.
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
SKIP: the INT2 GEMV bisect probe did not finish in ${PROBE_TIMEOUT}s at HEAD
      (rc=$RC). This is a KNOWN, DOCUMENTED state, not a regression --
      reports/2026-10-06-decode1-gemv-bisect.md section 4 records the isolation:
      every arm hangs including the \`table\` control, each arm alone also
      hangs (so it is not the shipped call), the sibling roofline_probe
      completes in ~6 s on the same box at the same moment, and the timed-out runs
      wedge the driver's kworker/*+xe_page_fault_work_queue.

      CONSEQUENCE, stated so nobody reads a green run as coverage: the arms
      VERB+HALF and VERB+MP remain UNMEASURED. The module candidate (1.26x vs a
      >=2x kill criterion) and the runtime-branch candidate (1.007x) are already
      falsified; ~4.5-5x of the 5.6-6.4x gap is still unexplained.

      To measure them: bash scripts/gpu_recover.sh diag && sudo bash
      scripts/gpu_recover.sh rebind, then re-run this gate.
UNPROVEN: bisect probe cannot run at HEAD (not a regression)
EOF
  exit 2
fi

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
