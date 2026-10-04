#!/usr/bin/env bash
# HOW MUCH OF `load1` DURING A BENCH ARM IS THE BENCHMARK ITSELF?
#
# WHY THIS EXISTS. The quiet-box rule is `load1 < 2`, and the standing explanation
# for why this box never satisfies it has been "other people's work" ever since the
# julia job turned out to be gone. That explanation was never MEASURED, and it
# decides the advice: if the harness's own CPU keeps the 1-minute average over the
# line, then waiting for a quiet box is necessary and NOT sufficient, and no amount
# of waiting will ever help. If the arm is cheap and the crossings come from
# elsewhere, waiting is the whole answer.
#
# MEASURED 2026-10-03 (this script's output, quoted in the commit and in
# reports/2026-09-30-pending-bench-backlog.md): on a box settled to load1=0.70,
# across one 27B `--tp 2` arm, load1 had median 0.62 and max 2.82, and the CLI's own
# process tree used a mean of 0.94 cores. So the arm is NOT sustainably CPU-bound --
# its median load is BELOW the reading taken before it started -- and the crossings
# come from short bursts plus whatever else is running. Hence: wait for a box with
# no desktop compositor and no sibling agent session.
#
# *** TWO MEASUREMENT PATHS THAT DO NOT WORK, RECORDED SO NEITHER IS RETRIED. ***
#   * /proc/stat's machine-wide idle delta. Measured: the idle counter went UP
#     across an arm in this container, so the delta is negative and the derived
#     "runnable tasks" figure is nonsense. Per-process /proc/<pid>/stat utime+stime
#     is readable and is what this script uses instead.
#   * inferring causation from "load1 rose during the arm". That is the AUDIT #9/#11
#     mistake; it needs the per-process number beside it, which is why both are
#     sampled here together.
#
# SEAMS, for the same reason scripts/_gate_precond.sh honours INTFER_SYCL_LS: an
# assumption this script cannot redirect is one it cannot be tested against.
#   $LOADAVG            loadavg source (default /proc/loadavg)
#   INTFER_QUIET_LOAD   threshold (default 2)
#   INTFER_BUILD_DIR    output dir (default $ROOT/build)
#   INTFER_ARM_NTOK     decode steps per arm (default 16 -- short; this measures
#                       load, not throughput, and a long arm only wastes a window)
#   INTFER_ARM_SECONDS  seconds to sample (default 0 = one arm's own duration)
#
# NOT a benchmark and NOT a performance harness: it publishes NO rate and NO
# wall-time figure, only the ratio of the CLI's own CPU to the loadavg the gate
# reads. Exit: 0 = measured; 1 = could not measure; 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

# The binary's tree is the SAME override as the output dir: an artifact consumer
# that ignores it measures the wrong build, which is the AUDIT #8 defect. Caught by
# scripts/check_build_dir_consumers.sh on this script's first run -- the checker
# flagged the hardcoded $ROOT/build here while the $OUT line beside it was already
# correct, which is exactly the "one rule, one site" case that checker exists for.
BIN="${INTFER_BIN:-${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli}"
MODEL="${INTFER_MODEL:-/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer}"
OUT="${INTFER_BUILD_DIR:-$ROOT/build}/load_attribution"
LOADAVG="${LOADAVG:-/proc/loadavg}"
QLOAD="${INTFER_QUIET_LOAD:-2}"
NTOK="${INTFER_ARM_NTOK:-16}"
HZ="$(getconf CLK_TCK 2>/dev/null || echo 100)"

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_loadattr.XXXXXX")"
# EXIT alone does not fire on an untrapped signal, and this script gets wrapped in
# `timeout` by callers -- the recorded /tmp rule, which cost 430 MB once.
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

[ -x "$BIN" ]  || { echo "SKIP: no intfer_cli at $BIN (build it, or set INTFER_BIN)"; exit 2; }
[ -f "$MODEL" ] || { echo "SKIP: no model at $MODEL (set INTFER_MODEL)"; exit 2; }
[ -r "$LOADAVG" ] || { echo "FAIL: cannot read \$LOADAVG=$LOADAVG"; exit 1; }
mkdir -p "$OUT" || { echo "FAIL: cannot create $OUT"; exit 2; }

set +u; . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"

load1() { awk '{print $1}' "$LOADAVG"; }
# CUMULATIVE CPU seconds consumed by a process AND its descendants, from
# /proc/<pid>/stat. Summed over the whole tree as it stands at this instant -- this
# is a monotone-ish total, NOT a rate. The caller differences it against the clock
# to obtain cores.
tree_cpu() { # $1 = root pid -> CPU seconds on stdout
  local pids="$1" front="$1" tot=0 p f pp next
  while [ -n "$front" ]; do
    next=""
    for p in $front; do
      [ -r "/proc/$p/stat" ] || continue
      tot=$(( tot + $(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null || echo 0) ))
      for f in /proc/[0-9]*/status; do
        [ -r "$f" ] || continue
        pp="$(awk '/^PPid:/{print $2; exit}' "$f" 2>/dev/null || echo 0)"
        [ "$pp" = "$p" ] && next="$next $(basename "$(dirname "$f")")"
      done
    done
    front="$next"
  done
  awk -v t="$tot" -v h="$HZ" 'BEGIN{printf "%.2f", t/h}'
}

run_arm() {
  "$BIN" --gpu --tp 2 -m "$MODEL" -p "The capital of France is" -n "$NTOK" \
    --no-stream --tokenizer-py scripts/bpe_tokenize.py \
    --detok-py scripts/bpe_detokenize.py
}

echo "== LOAD ATTRIBUTION: is the quiet-box gate measuring the harness? =="
echo "   threshold: load1 < $QLOAD   (INTFER_QUIET_LOAD)"
echo "   loadavg source: $LOADAVG   (LOADAVG)"
echo "   arm: 27B --tp 2, $NTOK decode steps   (this measures LOAD, not speed)"
echo
echo "load1 before the arm: $(load1)"
echo "settling so the reading means something (the loadavg is a 1-minute average,"
echo "so a reading taken straight after another run describes THAT run) ..."
settled=0
for _ in $(seq 1 40); do
  l="$(load1)"
  if awk -v l="$l" 'BEGIN{exit !(l < 1.0)}'; then settled=1; break; fi
  sleep 15
done
sleep 15
if [ "$settled" != "1" ]; then
  # *** REFUSE, do not measure anyway. *** The first version printed "settled at"
  # regardless and ran the arm on a busy box, which is the whole discipline this
  # repo exists to enforce applied to my own new script -- and the number it would
  # have produced would have been about a box in the middle of something else.
  # Found by running it: the loop timed out and the label still said "settled".
  echo "FAIL: the box never settled to load1 < 1.0 within 600s (now $(load1))."
  echo "  Attribution measured on a box that is already busy answers a different"
  echo "  question, and this repo has retracted figures for exactly that. Nothing"
  echo "  was measured. Re-run when the box is free."
  exit 1
fi
echo "load1 settled at    : $(load1)"
echo

SAMPLES="$OUT/samples.txt"
: > "$SAMPLES"
SECS="${INTFER_ARM_SECONDS:-0}"
[ "$SECS" -gt 0 ] 2>/dev/null || SECS=0
run_arm > "$SB/arm.log" 2>&1 &
CLI=$!
t0="$(date +%s)"
while kill -0 "$CLI" 2>/dev/null; do
  # THREE fields: the timestamp is REQUIRED. `tree_cpu` returns CUMULATIVE CPU
  # seconds, and labelling that "cores" is a unit error I made on the first run --
  # it printed "30.89 cores" for an arm that used about one. Cores are a RATE, so
  # they need the interval: (cpu[i]-cpu[i-1]) / (t[i]-t[i-1]).
  echo "$(load1) $(tree_cpu "$CLI") $(date +%s)" >> "$SAMPLES"
  [ "$SECS" -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge "$SECS" ] && break
  sleep 1
done
wait "$CLI"; rc=$?
cp "$SB/arm.log" "$OUT/arm.log" 2>/dev/null || true

n="$(grep -c . "$SAMPLES" 2>/dev/null || echo 0)"
if [ "$n" -lt 5 ]; then
  echo "FAIL: only $n sample(s) -- the arm finished too fast to attribute anything."
  echo "  Re-run with INTFER_ARM_NTOK=64 (and a quiet box)."
  exit 1
fi

awk -v n="$n" -v q="$QLOAD" '
  { l[NR]=$1; cu[NR]=$2; ts[NR]=$3
    if ($1>ml) ml=$1 }
  # CORES = d(cpu seconds)/d(seconds). Never print the cumulative number under the
  # word "cores": the first version did, and it read 30.89 for an arm using ~1.
  END {
    for (i = 2; i <= n; i++) {
      dt = ts[i] - ts[i-1]
      if (dt <= 0) continue
      r[i] = (cu[i] - cu[i-1]) / dt
      if (r[i] < 0) r[i] = 0          # a process EXITED; its counter stops growing
      k++; rr[k] = r[i]; sr += r[i]
      if (r[i] > mr) mr = r[i]
    }
    med_l = l[int(n/2)+1]
    mean_r = (k > 0) ? sr/k : 0
    asort_r = 0
    # median of the rate samples, via a simple insertion sort (n is small)
    for (i = 2; i <= k; i++) { v = rr[i]; j = i-1
      while (j >= 1 && rr[j] > v) { rr[j+1] = rr[j]; j-- }
      rr[j+1] = v }
    med_r = (k > 0) ? rr[int(k/2)+1] : 0
    printf "samples                 : %d load readings, %d rate intervals\n", n, k
    printf "load1  median / max     : %.2f / %.2f   (threshold %.2f)\n", med_l, ml, q
    printf "CLI tree  median / mean : %.2f / %.2f cores\n", med_r, mean_r
    printf "CLI tree  max rate     : %.2f cores\n", mr
    crosses = (ml >= q) ? "yes" : "no"
    printf "did load1 cross %-2s      : %s\n", q, crosses
    print ""
    # NOTE THE BRACED `} else {`. The first version wrote the branches as bare
    # multi-line `print` sequences under a newline-separated `else`, which awk
    # parses as a statement ending at the newline -- it died with a syntax error at
    # the word `else`. Found by RUNNING the script, not by reading it, which is the
    # only reason this is fixed rather than shipped.
    if (med_l < q) {
      print "VERDICT: the arm is NOT what keeps the box over the threshold -- its"
      print "         MEDIAN load is below it. Waiting for a quiet box IS the right"
      print "         advice; on this host it means no desktop compositor and no"
      print "         sibling session, not merely a low reading."
      print ""
      printf "         the arm itself uses ~%.1f cores (mean), so it cannot account\n", mean_r
      print "         for a box sitting far above the threshold; the rest is"
      print "         whatever else is running."
    } else {
      print "VERDICT: the arm itself holds load1 at or above the threshold, so"
      print "         \"wait for a quiet box\" is NECESSARY AND NOT SUFFICIENT."
      print "         Do NOT respond by lowering the threshold."
    }
  }' "$SAMPLES"

echo
echo "artefacts: $SAMPLES   $OUT/arm.log   (arm rc=$rc)"
echo "NOTE: this publishes no rate and no wall-time figure. It answers one"
echo "      question -- who owns the load -- and nothing else."
exit 0