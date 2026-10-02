#!/usr/bin/env bash
# Wait for a genuinely quiet box, then run the RESULT-filling benchmark block.
#
# WHY. The block's own require_quiet() gate aborts when load1 >= 2, which is
# correct but leaves the work as a manual "check the box, then paste this". This
# turns the one open item into a single command: start it, and it runs when the
# box is free, with every round still gated.
#
# The julia job at 99.7% CPU has been the blocker for the whole session. Nothing
# here stops it -- only you can -- but this will be waiting when you do.
#
# Usage: scripts/bench_when_quiet.sh [--max-wait SECONDS]
# Exit: 0 = the block ran; 2 = gave up waiting (nothing was measured);
#       the block's own rc otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
BACKLOG="reports/2026-09-30-pending-bench-backlog.md"
LIMIT="${INTFER_QUIET_WAIT:-3600}"
[ "${1:-}" = "--max-wait" ] && LIMIT="${2:-3600}"

QLOAD="${INTFER_QUIET_LOAD:-2}"
load1() { awk '{print $1}' /proc/loadavg; }
quiet() { awk -v l="$(load1)" -v q="$QLOAD" 'BEGIN{exit !(l < q)}'; }

echo "waiting for load1 < $QLOAD (max ${LIMIT}s) -- Ctrl-C to stop"
waited=0
while [ "$waited" -lt "$LIMIT" ]; do
  if quiet; then
    echo "quiet after ${waited}s: load1=$(load1)"
    break
  fi
  [ $((waited % 300)) -eq 0 ] && echo "  ...load1=$(load1) after ${waited}s"
  sleep 30; waited=$((waited+30))
done

if ! quiet; then
  echo
  echo "GAVE UP after ${waited}s: load1=$(load1) is not < $QLOAD." >&2
  echo "  NOTHING WAS MEASURED and no figure from this session is usable." >&2
  echo "  Top CPU consumers right now:" >&2
  ps -eo pcpu,comm --sort=-pcpu 2>/dev/null | head -5 >&2
  echo "RESULT: bench_when_quiet TIMED OUT (no measurement taken)" >&2
  exit 2
fi

echo
echo "=== box is quiet; running the block from $BACKLOG ==="
# Extract the fenced bash block under the ONE COMMAND heading and run it
# verbatim, so what runs is exactly what the report documents.
BLOCK="$(awk '/^## ONE COMMAND FILLS/{f=1} f&&/^```bash$/{p=1;next} p&&/^```$/{exit} p' "$BACKLOG")"
if [ -z "$BLOCK" ]; then
  echo "FAIL: could not extract the block from $BACKLOG" >&2
  exit 1
fi
printf '%s\n' "$BLOCK" > /tmp/intfer_result_block.sh
bash -n /tmp/intfer_result_block.sh || { echo "FAIL: block does not parse" >&2; exit 1; }
echo "block extracted ($(printf '%s\n' "$BLOCK" | grep -c .) lines), syntax OK"
bash /tmp/intfer_result_block.sh
rc=$?
rm -f /tmp/intfer_result_block.sh
echo
echo "block exit=$rc -- paste the 'decode: N tokens in X ms (Y tok/s)' lines"
echo "into the RESULT: lines it names. Do NOT copy the bucket numbers across."
exit "$rc"
