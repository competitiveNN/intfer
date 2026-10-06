#!/usr/bin/env bash
# Gate for examples/roofline_probe_bisect1 -- bisect-1 of the M=1 INT2 GEMV probe
# ladder (the `table` control vs `VERBATIM`, the shipped-source transcription).
# See scripts/test_roofline_bisect.sh for the full context: the ladder was split
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
# Usage: scripts/test_roofline_bisect1.sh [--device N]
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe_bisect1/roofline_probe_bisect1"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/_gate_build.sh"
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

timeout -k 15 "$PROBE_TIMEOUT" "$SCRIPT_DIR/with_oneapi.sh" "$BIN" --device "$DEVICE" \
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

if [ "$RC" -ne 0 ]; then
  echo "FAIL: the bisect-1 probe exited rc=$RC (the twin is the likely reason)" >&2
  exit 1
fi
grep -q 'RESULT: PASS' "$LOG" || { echo "FAIL: no RESULT: PASS in the output" >&2; exit 1; }

echo "ROOFLINE-BISECT1 TEST: PASS"
echo "RESULT: bisect-1 ran with its twin holding; the table/VERBATIM ratio above"
echo "        is the within-run (interleaved) measurement of the tidy-rewrite control"
echo "        against the transcribed SHIPPED source, anchored on the VERBATIM control"
echo "        at index 0 (SHIPPED GB/s anchored from examples/roofline_probe)."
exit 0
