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

# AUDIT #22 root cause 2 was "no arm wrote a log", and this script was the half of
# it that was OPERATOR-enforced: its own stdout only reached a file if someone
# remembered `nohup ... > log 2>&1 &`. On arrival the only bench_when_quiet log
# on the box held no measurement at all, and the figures that had been published
# had no artifact anywhere. So the run captures ITSELF, to a per-run path, and
# says so -- a figure is only quotable if it can be pointed at.
# (Per-arm logs are the block's job, via arm_run; this is the top-level capture.)
RUNLOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/bench_logs"
mkdir -p "$RUNLOG_DIR" || { echo "FAIL: cannot create $RUNLOG_DIR" >&2; exit 2; }
# The name carries the short HEAD and the PID, not just a second-resolution
# timestamp. Measured: two runs inside the same second produced the SAME path and
# the second APPENDED to the first, so the artifact could no longer say which run
# produced which line -- an ambiguous log is the artifact problem all over again.
# INTFER_RUNLOG overrides the artifact path. That exists so the COLLISION GUARD
# below is TESTABLE: the guard's whole job is to refuse an existing path, and the
# only way to exercise that is to hand the script a path that already exists --
# which is the same rationale as INTFER_SYCL_LS and INTFER_BUILD_DIR. An
# assumption a harness cannot redirect is one it cannot be tested against.
RUNLOG="${INTFER_RUNLOG:-$RUNLOG_DIR/when_quiet_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD 2>/dev/null || echo unknown)_$$.log}"
# REFUSE to reuse an existing artifact. The name carries second-resolution time,
# HEAD and the PID, which makes a collision unlikely inside one session -- but a
# relaunch on the SAME second, or a PID recycled by the kernel, produces the same
# path, and then the second run APPENDS to the first and the artifact can no longer
# say which run produced which line. Measured this session: an outer capture file
# written with a FIXED name (`QUEUED_when_quiet.out`) was overwritten by three
# successive launches, so the "which run produced the four USABLE arms" question
# could not be answered from it. An ambiguous log is the artifact problem all over
# again, and it is cheaper to refuse than to reconstruct.
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing to append to an existing artifact." >&2
  echo "      Move it aside, or wait a second and re-run." >&2
  exit 2
fi
echo "run log: $RUNLOG"
echo
echo "NOTE: if you are QUEUEING this (\"&\" or nohup/setsid), give the OUTER capture a"
echo "  UNIQUE name too -- e.g. > build/bench_logs/wq_\$(date -u +%Y%m%dT%H%M%SZ).out"
echo "  A fixed name there silently overwrites the previous launch's record, and"
echo "  then no one can say which run produced which arm."
echo
echo "NOTE: do not EDIT this file while it is running. bash re-reads a script by"
echo "  byte offset, so a write-then-mv replacement leaves the live run executing"
echo "  the OLD inode -- which is the safer of the two, but means your edit does"
echo "  not take effect until the next run."

printf '%s\n' "$BLOCK" > "$RUNLOG_DIR/intfer_result_block.sh"
bash -n "$RUNLOG_DIR/intfer_result_block.sh" || {
  echo "FAIL: block does not parse" >&2; exit 1; }
echo "block extracted ($(printf '%s\n' "$BLOCK" | grep -c .) lines), syntax OK"

# Self-describing header, so the artifact records WHICH tree produced it. A log
# with no sha is a citation to nothing -- the AUDIT #22 failure mode.
{
  echo "=== when_quiet run $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "threshold_load1=$QLOAD  load1_at_start=$(load1)"
  echo "=== block output ==="
} >> "$RUNLOG"

# tee rather than `exec > >(tee)`: process substitution is the fragile way to do
# this and PIPESTATUS[0] keeps the BLOCK's exit code rather than tee's.
bash "$RUNLOG_DIR/intfer_result_block.sh" 2>&1 | tee -a "$RUNLOG"
rc="${PIPESTATUS[0]}"
rm -f "$RUNLOG_DIR/intfer_result_block.sh"

# Anti-vacuity: a capture that produced no file, or an empty one, is not evidence.
# "Found nothing to look at" and "found nothing wrong" are different claims
# (AUDIT #20). Without this the tee could silently no-op and the run would look
# measured exactly as it did on arrival.
if [ ! -s "$RUNLOG" ]; then
  echo "FAIL: run log $RUNLOG is missing or empty -- NOTHING WAS MEASURED" >&2
  exit 2
fi
{
  echo "=== end $(date -u +%Y-%m-%dT%H:%M:%SZ)  block_exit=$rc ==="
  echo "Usable figures are the [USABLE] lines only. [CONTENDED -- NOT quotable]"
  echo "means the load crossed the threshold while that arm ran."
} >> "$RUNLOG"

echo
echo "block exit=$rc"
echo "run artifact: $RUNLOG   (per-arm logs: build/bench_logs/<run>/<arm>.log)"
echo "Only arms this run tagged [USABLE] may be pasted into the RESULT: lines."
echo "A [CONTENDED -- NOT quotable] line has NO figure: the load crossed the"
echo "threshold while that arm ran. Do NOT copy the bucket numbers across."
exit "$rc"
