#!/usr/bin/env bash
# Gate for examples/roofline_probe_bisect1 -- bisect-1 of the M=1 INT2 GEMV probe
# ladder (the `table` control vs `VERBATIM`, the shipped-source transcription).
# See scripts/gates/test_roofline_bisect.sh for the full context: the ladder was split
# across two binaries because one probe TU holds roughly six to eight INT2 kernel
# bodies, and bisect1/ bisect-2 each run engine-free with the ceiling+main
# machinery shared in probe_common.hpp. bisect1 establishes the VERBATIM baseline
# and the within-run table/VERBATIM ratio; bisect-2 measures the single-change
# arms (VERB+HALF / VERB+TAIL / VERB+MP) against VERBATIM.
#
# EXIT SEMANTICS:
#   0 = twin held and every arm ran (ROOFLINE-BISECT1 TEST: PASS)
#   1 = twin failed, an arm did not run, or profiling lost
#   2 = the probe timed out (UNPROVEN: not a regression; safety net)
#
# BUILD-FIRST (audit #3c/4c family): run whatever is on disk, or a failed compile
# leaves the previous binary reporting PASS for code that no longer exists.
#
# Usage: scripts/gates/test_roofline_bisect1.sh [--device N]
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe_bisect1/roofline_probe_bisect1"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/../lib/_gate_build.sh"
gate_require_build roofline_probe_bisect1 || exit 1
[ -x "$BIN" ] || { echo "FAIL: roofline_probe_bisect1 binary missing: $BIN" >&2; exit 1; }

LOG="$BUILD_DIR/roofline_bisect1.log"

# SAFETY NET: refuse to wait the runner's full budget if the TU ever grows too big.
PROBE_TIMEOUT="${INTFER_BISECT1_TIMEOUT:-120}"
RUNNER_BUDGET=1800
if [ "$PROBE_TIMEOUT" -ge "$RUNNER_BUDGET" ]; then
  echo "FAIL: INTFER_BISECT1_TIMEOUT=$PROBE_TIMEOUT is not under the runner's" >&2
  echo "      $RUNNER_BUDGET s budget, so a hang would consume the whole step" >&2
  echo "      AND surface as rc=124, which the runner reads as a regression." >&2
  exit 1
fi

timeout -k 15 "$PROBE_TIMEOUT" "\"$SCRIPT_DIR/../with_oneapi.sh" "$BIN" --device "$DEVICE" \
  >"$LOG" 2>&1
RC=$?
cat "$LOG"
echo

if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
  cat <<EOF
SKIP: the bisect-1 probe did not finish in ${PROBE_TIMEOUT}s (rc=$RC).
      This is NOT a regression -- it means the TU grew beyond what this binary
      can hold, and the gate is doing its job by refusing to wait the runner's
      full 1800 s. Rebuild smaller (split further across binaries) and re-run.
UNPROVEN: bisect-1 probe timed out (not a regression; safety net fired)
EOF
  exit 2
fi

if [ "$RC" -ne 0 ]; then
  echo "FAIL: the bisect-1 probe exited rc=$RC -- it refused to run (the probe's" >&2
  echo "      own vacuity floor fired, so no arm line exists to be 'missing')." >&2
  exit 1
fi

need() { grep -qE "$1" "$LOG" || {
          echo "FAIL: expected '$2' in the output, and it is absent." >&2
          echo "      A missing arm is not a neutral result: the run did not measure" >&2
          echo "      what this gate claims to measure." >&2
          exit 1; }; }

# THE TWIN and the ceiling must have run (always-red).
need 'ceiling .*GB/s'        "the ceiling arm"
need 'twin .*ratio to full'  "the always-red twin"
# Each arm must have run; `vs VERBATIM` proves the interleaved ratio column ran
# for both arms. The ratio is computed against VERBATIM (index 0) within the run.
need 'table'                 "the tidy-rewrite control arm"
need 'VERBATIM'              "the shipped-source transcription arm"
need 'vs VERBATIM'           "the interleaved ratio column"
need 'WHAT THIS RUN DOES'    "the what-this-establishes banner"

# *** THE VACUITY FLOOR. *** `need()` asserts that the arm strings EXIST, and on
# this box bisect-1 prints every arm at `0.000 ms / inf GB/s / -nanx vs VERBATIM`
# -- including the 52 MB ceiling -- and still prints RESULT: PASS. A string
# assertion cannot see that the numbers are NaN: it certifies that the probe
# RAN, not that it MEASURED. This floor reads the actual numeric columns and
# refuses any run whose VERBATIM median is not a finite positive millisecond,
# which is the one number every ratio in the table is divided by. Without it the
# gate reports PASS for a probe that produced no measurement.
#
# *** THE ORIGINAL refusal was written for the VACUOUS all-zero era and encoded
# "positive" as `[1-9][0-9]*\.` -- an INTEGER PART of at least 1. MEASURED
# 2026-10-12: the probe now really measures on this box (device timestamps are
# live again), with VERBATIM medians of 0.249-0.622 ms across all four bisect
# arms, so the floor refused every REAL run with "no finite positive VERBATIM
# median" -- and bisect-2's own comment below cites the real value as "~0.225
# ms", i.e. the floor contradicted the reference quoted beside it. A floor that
# refuses a real measurement is the always-red twin: it cannot be satisfied by
# anything that runs, so it certifies nothing. Both predicates are now written
# from the intended MEANING -- "finite" = a decimal number is present
# (nan/inf/absent refuse), "positive" = not all zeroes -- and the vacuous case
# is caught by the zero branch below, which was and stays the loud, specific
# refusal. Both directions are exercised by mutate_roofline_bisect1.sh: RB1-8
# is the vacuous stub (must REFUSE) and RB1-9 is a REAL sub-millisecond stub
# (must PASS); without RB1-9 this defect could not have been observed here at
# all, because the matrix only ever proved the floor could still fire.
if grep -qE 'VERBATIM +0+\.0* ms' "$LOG"; then
  echo "FAIL: the bisect-1 probe printed a ZERO VERBATIM median -- the device" >&2
  echo "      timer reported command_start == command_end, so every GB/s and" >&2
  echo "      ratio in the table is inf/-nan and the run measured nothing. The" >&2
  echo "      probe itself now refuses this (zero is a failure, not a result);" >&2
  echo "      this floor catches the case where it does not." >&2
  exit 1
fi
if ! grep -qE 'VERBATIM +[0-9]+\.[0-9]+ ms' "$LOG"; then
  echo "FAIL: no finite positive VERBATIM median in the bisect-1 output -- the" >&2
  echo "      ratios are not anchored to a real measurement." >&2
  exit 1
fi

grep -q 'RESULT: PASS' "$LOG" || { echo "FAIL: no RESULT: PASS in the output" >&2; exit 1; }

echo "ROOFLINE-BISECT1 TEST: PASS"
echo "RESULT: bisect-1 ran with its twin holding; the table/VERBATIM ratio above"
echo "        is the within-run (interleaved) measurement of the tidy-rewrite control"
echo "        against the transcribed SHIPPED source, anchored on the VERBATIM control"
echo "        at index 0 (SHIPPED GB/s anchored from examples/roofline_probe)."
exit 0
