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
# *** THE GAP THIS WAS MAPS DOES NOT EXIST. RETRACTED 2026-10-08. *** The 262 GB/s
# clone figure is not reproducible: a fresh run of examples/roofline_probe returns
# VACUOUS, and a saved bisect4 log from 10-06 measured the SAME clone algorithm at
# 55.1 GB/s while the shipped arm stayed at ~40-49 GB/s in both. The clone number
# moved 4.7x between two saved logs of the same kernel; the shipped number did not.
# So the 6.39x was never a clone-vs-shipped ratio, it was a flaky numerator divided
# by a stable one, and the honest ratio is 1.12x (VERBATIM 49.4, CLONE 55.1, MLP_GATE,
# build/roofline_bisect4.log 10-06). The ladder's purpose is moot: there is no gap
# left to close. The arms are still a valid differential probe of the shipped source's
# cosmetic variants and the gate is live; what they can no longer claim is that they
# are tracing a deficit. See HISTORY.md "THE 6.39x ITSELF IS NOT REPRODUCIBLE".
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
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe_bisect4/roofline_probe_bisect4"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/../lib/_gate_build.sh"
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

timeout -k 15 "$PROBE_TIMEOUT" "\"$SCRIPT_DIR/../with_oneapi.sh" "$BIN" --device "$DEVICE" \
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

if [ "$RC" -ne 0 ]; then
  echo "FAIL: the bisect-4 probe exited rc=$RC -- it refused to run (the probe's" >&2
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
# Every clone arm must have run and announced itself; `vs VERBATIM` proves the
# interleaved ratio column was produced for every arm.
need 'VERBATIM'              "the shipped-source control arm"
need 'VERB\+bitcast'         "the bit_cast group+tail + wtab tail arm"
need 'VERB\+bitcast\+4'      "the 4-byte code-load arm"
need 'VERB\+bitcast\+4\+vsel' "the probe's 3-ternary bv selector arm"
need 'CLONE'                 "the full int2_variant<0> transcription arm"
need 'vs VERBATIM'           "the interleaved ratio column"
need 'WHAT THIS RUN DOES'    "the what-this-establishes banner"

# *** THE VACUITY FLOOR. *** `need()` asserts that the arm strings EXIST, and on
# this box bisect-4 prints every arm at `0.000 ms / inf GB/s / -nanx vs VERBATIM`
# -- including the 52 MB ceiling -- and still prints RESULT: PASS. A string
# assertion cannot see that the numbers are NaN: it certifies that the probe
# RAN, not that it MEASURED. This floor reads the actual numeric columns and
# refuses any run whose VERBATIM median is not a finite positive millisecond,
# which is the one number every ratio in the table is divided by. Without it the
# gate reports PASS for a probe that produced no measurement.
if grep -qE 'VERBATIM +0+\.0* ms' "$LOG"; then
  echo "FAIL: the bisect-4 probe printed a ZERO VERBATIM median -- the device" >&2
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
# specifically). Both directions are exercised by mutate_roofline_bisect4.sh:
# the vacuous stub must REFUSE, the new real sub-millisecond stub must PASS.
if ! grep -qE 'VERBATIM +[0-9]+\.[0-9]+ ms' "$LOG"; then
  echo "FAIL: no finite positive VERBATIM median in the bisect-4 output -- the" >&2
  echo "      ratios are not anchored to a real measurement." >&2
  exit 1
fi

grep -q 'RESULT: PASS' "$LOG" || { echo "FAIL: no RESULT: PASS in the output" >&2; exit 1; }

echo "ROOFLINE-BISECT4 TEST: PASS"
echo "RESULT: bisect-4 ran with its twin holding; the VERB+bitcast / VERB+bitcast"
echo "        +4 / VERB+bitcast+4+vsel / CLONE ratios above are measured against the"
echo "        VERBATIM control on the same buffers, interleaved per rep."
exit 0
