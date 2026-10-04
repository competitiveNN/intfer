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
#        scripts/bench_when_quiet.sh --guard-only   (test BOTH setup guards in
#        seconds: the concurrency lock and the artifact collision guard)
# Exit: 0 = the block ran; 2 = gave up waiting, or a guard refused;
#       the block's own rc otherwise.
#
# A second concurrent run is REFUSED, not accommodated: the guards below run before
# the wait, so a second launch costs one second to reject rather than an hour.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
BACKLOG="reports/2026-09-30-pending-bench-backlog.md"
LIMIT="${INTFER_QUIET_WAIT:-3600}"
[ "${1:-}" = "--max-wait" ] && LIMIT="${2:-3600}"

# SEAMS, for the same reason scripts/_gate_precond.sh honours INTFER_SYCL_LS and
# scripts/measure_load_attribution.sh honours $LOADAVG: an assumption a harness
# cannot redirect is one it cannot be tested against. Without $LOADAVG the
# collision guard below is UNREACHABLE on a busy box -- the wait loop would give up
# first and exit 2 for a completely different reason, so the guard could never be
# observed failing. Measured: that is exactly what happened the first time it was
# checked, and the "proof" that came back was the give-up path, not the guard.
#   $LOADAVG          loadavg source (default /proc/loadavg)
#   INTFER_QUIET_LOAD threshold (default 2)
QLOAD="${INTFER_QUIET_LOAD:-2}"
LOADAVG="${LOADAVG:-/proc/loadavg}"
[ -r "$LOADAVG" ] || { echo "FAIL: cannot read \$LOADAVG=$LOADAVG" >&2; exit 2; }
load1() { awk '{print $1}' "$LOADAVG"; }
quiet() { awk -v l="$(load1)" -v q="$QLOAD" 'BEGIN{exit !(l < q)}'; }

RUNLOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/bench_logs"
mkdir -p "$RUNLOG_DIR" || { echo "FAIL: cannot create $RUNLOG_DIR" >&2; exit 2; }

# *** A PID-OWNING LOCK, BECAUSE THE COLLISION GUARD BELOW CANNOT SEE A CONCURRENT RUN. ***
# The guard refuses an EXISTING ARTIFACT PATH, and the artifact name carries $$ -- so
# two LIVE runs get DIFFERENT names and neither one sees the other. That is not a
# cosmetic gap. These are 27B --tp 2 batches: a second one loads the same CPU and the
# same GPU, so BOTH runs' arms cross the load threshold and BOTH artifacts withhold
# every figure. Two runs cannot each obtain the quiet window they are both waiting for,
# and the artifacts stay perfectly clean while both are ruined -- so this is the same
# concern as the collision guard one level up, and the guard's own name is the reason
# it is missed.
#
# It is the discipline scripts/run_all_checks.sh already uses, for the same reason and
# with the same two directions, both of which must hold:
#   owner ALIVE -> refuse LOUDLY, and never instruct deleting the lock
#   owner DEAD  -> self-heal, with a loud line naming the dead pid
# Fail-closed on a RECYCLED pid (kill -0 succeeds, so the lock reads live and we
# refuse): annoying, never dangerous.
# INTFER_WQ_LOCK is a seam so both directions are testable in seconds. The
# INTFER_SYCL_LS / INTFER_BUILD_DIR rationale: an assumption no harness can redirect
# is an assumption no harness can test.
WQ_LOCK="${INTFER_WQ_LOCK:-$RUNLOG_DIR/.bench_when_quiet.lock}"
if ! mkdir "$WQ_LOCK" 2>/dev/null; then
  wq_owner="$(cat "$WQ_LOCK/pid" 2>/dev/null || true)"
  if [ -n "$wq_owner" ] && ! kill -0 "$wq_owner" 2>/dev/null; then
    echo "STALE LOCK: $WQ_LOCK was held by pid $wq_owner, which is gone. Removing" >&2
    echo "  it and continuing. Loud on purpose: a silent takeover would hide a run that" >&2
    echo "  died mid-flight, leaving a truncated artifact that looks complete." >&2
    rm -rf "$WQ_LOCK"
    mkdir "$WQ_LOCK" 2>/dev/null || {
      echo "FAIL: could not re-create $WQ_LOCK after removing a stale one." >&2; exit 2; }
  else
    echo "FAIL: another bench_when_quiet.sh holds $WQ_LOCK${wq_owner:+ (pid $wq_owner)}." >&2
    echo "  These are 27B --tp 2 batches: a second one loads the same CPU and GPU, so" >&2
    echo "  BOTH runs' arms would cross the load threshold and BOTH artifacts would" >&2
    echo "  withhold every figure. Wait for it. Check the holder with:" >&2
    echo "    kill -0 ${wq_owner:-<pid in $WQ_LOCK/pid>}" >&2
    echo "  Do NOT delete the lock to get past this -- the artifact collision guard" >&2
    echo "  cannot see a concurrent run, so this lock is the ONLY thing that can." >&2
    exit 2
  fi
fi
echo $$ > "$WQ_LOCK/pid" 2>/dev/null || true
trap 'rm -rf "$WQ_LOCK" 2>/dev/null || true' EXIT
trap 'rm -rf "$WQ_LOCK" 2>/dev/null || true; exit 130' INT
trap 'rm -rf "$WQ_LOCK" 2>/dev/null || true; exit 143' TERM

# *** THE ARTIFACT PATH AND ITS COLLISION GUARD ARE SETUP, NOT A WAIT RESULT. ***
# Both used to sit AFTER the quiet-wait loop, which made the guard unreachable on a
# busy box: the wait gave up and exited 2 for an entirely different reason, so the
# refusal could never be observed -- and the first "proof" of it was that give-up
# path. They now run first, which also means the name records when the run was
# QUEUED rather than when the box happened to go quiet, which is the more honest
# of the two.
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

# --guard-only: evaluate the two setup decisions and stop. This exists so BOTH
# directions of BOTH guards are testable in seconds:
#   live lock holder   -> rc=2, refusal, holder's lock untouched
#   dead lock holder   -> rc=0, self-heal announced, run proceeds
#   existing artifact  -> rc=2, refusal message, artifact untouched
#   new artifact       -> rc=0, NO refusal message, nothing written
# Without it the "must stay quiet" directions require running the whole 27B block,
# which is why they could only be OBSERVED (in the artifacts this script wrote) and
# never staged. An assumption a harness cannot redirect is one it cannot be tested
# against -- and a direction that costs 20 minutes of GPU is one nobody re-runs.
# The EXIT trap releases the lock on the way out, so this mode takes and frees it.
case "${1:-}" in
  --guard-only)
    echo "GUARD-ONLY: lock $WQ_LOCK acquired (no refusal)"
    echo "GUARD-ONLY: artifact $RUNLOG is available (no refusal)"
    exit 0 ;;
esac

echo "waiting for load1 < $QLOAD (max ${LIMIT}s) -- Ctrl-C to stop"
# *** THE SLEEP MUST BE INTERRUPTIBLE, OR THE TRAP IS DEFERRED AND NEVER RUNS. ***
# A trap for a signal arriving while a FOREGROUND command runs does not execute
# until that command returns -- so a plain `sleep 30` here meant SIGTERM sat
# unhandled for up to 30 seconds. Measured: 10s after SIGTERM the lock was still
# present and the process still alive, so `timeout` could not stop it at all; and
# since GNU timeout WAITS for the child, a step that ignored its signal would hang
# the whole stack rather than fail it. `wait` IS interruptible, so the sleep is
# backgrounded and waited on: the trap fires at once, releases the lock and exits
# 143. The orphaned `sleep` then ends by itself within 30s and touches nothing.
waited=0
while [ "$waited" -lt "$LIMIT" ]; do
  if quiet; then
    echo "quiet after ${waited}s: load1=$(load1)"
    break
  fi
  [ $((waited % 300)) -eq 0 ] && echo "  ...load1=$(load1) after ${waited}s"
  sleep 30 &
  _wq_sleep=$!
  wait "$_wq_sleep" 2>/dev/null || true
  kill "$_wq_sleep" 2>/dev/null || true
  waited=$((waited+30))
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
