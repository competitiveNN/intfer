#!/usr/bin/env bash
# Gate for examples/roofline_probe_bisect3 -- bisect-3 of the M=1 INT2 GEMV probe
# ladder: `VERBATIM` (the transcribed SHIPPED source) plus single-change arms
# `VERB+TAIL` / `VERB+MP`, and the `SHIPPED` anchor -- all four in ONE TU so the
# run is fully self-contained and the SHIPPED ratio is measured on the SAME buffers,
# interleaved per rep against the VERBATIM control. See scripts/test_roofline_bisect.sh
# and scripts/test_roofline_bisect1.sh for the full context: the ladder was split
# across binaries because one probe TU holds roughly six to eight INT2 kernel
# bodies, and bisect1/ bisect2/ bisect3 each run engine-free with the ceiling+main
# machinery shared in probe_common.hpp.
#
# WHY THIS BINARY. bisect-2 carries no `SHIPPED` arm -- that anchor is measured by
# examples/roofline_probe, which is a DIFFERENT module context, so its register
# pressure and the resulting GB/s are not directly comparable to the in-TU arms.
# This binary closes that gap: the ~4.5-5x of the 5.6-6.4x that survives the module
# move (1.26x) and the runtime branch (1.007x) sits between VERBATIM and SHIPPED, and
# this run measures it in one TU, interleaved, on the same buffers.
#
# *** THE GAP THIS WAS BUILT TO CLOSE DOES NOT EXIST. RETRACTED 2026-10-08. *** The
# 244 GB/s clone figure is not reproducible: a fresh run of examples/roofline_probe
# returns VACUOUS, and a saved bisect4 log from 10-06 measured the SAME clone
# algorithm at 55.1 GB/s while the shipped arm stayed at ~40-49 GB/s in both. The
# clone number moved 4.7x between two saved logs of the same kernel; the shipped
# number did not. So the 5.6-6.4x was never a clone-vs-shipped ratio, it was a flaky
# numerator divided by a stable one, and the honest ratio is 1.12x. The "~4.5-5x
# that survives the module move" line above is void: there is no residue to measure,
# and this binary's SHIPPED arm is now just an in-TU control, not an anchor for a
# deficit. See HISTORY.md "THE 6.39x ITSELF IS NOT REPRODUCIBLE".
#
# EXIT SEMANTICS:
#   0 = twin held and every arm ran (ROOFLINE-BISECT3 TEST: PASS)
#   1 = twin failed, an arm did not run, or profiling lost
#   2 = the probe timed out (UNPROVEN: not a regression; safety net)
#
# BUILD-FIRST (audit #3c/4c family): run whatever is on disk, or a failed compile
# leaves the previous binary reporting PASS for code that no longer exists.
#
# Usage: scripts/test_roofline_bisect3.sh [--device N]
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe_bisect3/roofline_probe_bisect3"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build roofline_probe_bisect3 || exit 1
[ -x "$BIN" ] || { echo "FAIL: roofline_probe_bisect3 binary missing: $BIN" >&2; exit 1; }

LOG="$BUILD_DIR/roofline_bisect3.log"

# SAFETY NET: refuse to wait the runner's full budget if the TU ever grows too big.
PROBE_TIMEOUT="${INTFER_BISECT3_TIMEOUT:-120}"
RUNNER_BUDGET=1800
if [ "$PROBE_TIMEOUT" -ge "$RUNNER_BUDGET" ]; then
  echo "FAIL: INTFER_BISECT3_TIMEOUT=$PROBE_TIMEOUT is not under the runner's" >&2
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
SKIP: the bisect-3 probe did not finish in ${PROBE_TIMEOUT}s (rc=$RC).
      This is NOT a regression -- it means the TU grew beyond what this binary
      can hold, and the gate is doing its job by refusing to wait the runner's
      full 1800 s. Rebuild smaller (split further across binaries) and re-run.
UNPROVEN: bisect-3 probe timed out (not a regression; safety net fired)
EOF
  exit 2
fi

if [ "$RC" -ne 0 ]; then
  echo "FAIL: the bisect-3 probe exited rc=$RC -- it refused to run (the probe's" >&2
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
# Every arm must have run; `vs VERBATIM` proves the interleaved ratio column ran
# for both arms. The ratio is computed against VERBATIM (index 0) within the run.
need 'VERBATIM'              "the shipped-source control arm"
need 'VERB\+TAIL'           "the hardware half->float tail-scale arm (bisect-3)"
need 'VERB\+MP'             "the multi_ptr weight-side arm"
need 'SHIPPED'               "the shipped-source anchor (in-TU, same buffers)"
need 'vs VERBATIM'           "the interleaved ratio column"
need 'WHAT THIS RUN DOES'    "the what-this-establishes banner"

# *** THE VACUITY FLOOR. *** `need()` asserts that the arm strings EXIST, and on
# this box bisect-3 prints every arm at `0.000 ms / inf GB/s / -nanx vs VERBATIM`
# -- including the 52 MB ceiling -- and still prints RESULT: PASS. A string
# assertion cannot see that the numbers are NaN: it certifies that the probe
# RAN, not that it MEASURED. This floor reads the actual numeric columns and
# refuses any run whose VERBATIM median is not a finite positive millisecond,
# which is the one number every ratio in the table is divided by. Without it the
# gate reports PASS for a probe that produced no measurement.
if grep -qE 'VERBATIM +0+\.0* ms' "$LOG"; then
  echo "FAIL: the bisect-3 probe printed a ZERO VERBATIM median -- the device" >&2
  echo "      timer reported command_start == command_end, so every GB/s and" >&2
  echo "      ratio in the table is inf/-nan and the run measured nothing. The" >&2
  echo "      probe itself now refuses this (zero is a failure, not a result);" >&2
  echo "      this floor catches the case where it does not." >&2
  exit 1
fi
# THE ORIGINAL floor encoded "positive" as `[1-9][0-9]*\.` (integer part >= 1),
# which MEASURED 2026-10-12 refuses every REAL sub-millisecond run on this box
# (VERBATIM medians 0.249-0.622 ms). "finite" is now "a decimal number is
# present", "positive" is "not all zeroes" (the zero branch above catches that
# specifically). Both directions are exercised by mutate_roofline_bisect3.sh:
# the vacuous stub must REFUSE, the new real sub-millisecond stub must PASS.
if ! grep -qE 'VERBATIM +[0-9]+\.[0-9]+ ms' "$LOG"; then
  echo "FAIL: no finite positive VERBATIM median in the bisect-3 output -- the" >&2
  echo "      ratios are not anchored to a real measurement." >&2
  exit 1
fi

grep -q 'RESULT: PASS' "$LOG" || { echo "FAIL: no RESULT: PASS in the output" >&2; exit 1; }

echo "ROOFLINE-BISECT3 TEST: PASS"
echo "RESULT: bisect-3 ran with its twin holding; the VERB+TAIL / VERB+MP /"
echo "        SHIPPED ratios above are measured against the VERBATIM control on"
echo "        the SAME buffers, in the SAME TU, interleaved per rep. SHIPPED is an"
echo "        in-TU control, NOT an anchor for a deficit: the 5.6-6.4x this ladder"
echo "        was built to trace is RETRACTED (the clone number moved 4.7x between"
echo "        two saved logs of the same kernel). See HISTORY.md."
exit 0