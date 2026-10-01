#!/usr/bin/env bash
# Benchmark ONE batch of report experiments (E1/E2/E3) and emit a keep/kill table.
#
# Protocol (deliberately strict, follows scripts/bench_quiet_ab.sh):
#   1. device preconditions, quiet gate (load1 < INTFER_QUIET_LOAD)
#   2. STREAM IDENTITY FIRST, on every arm, before any timing is believed
#   3. interleaved rounds (round-major, arm-minor), every round reported
#   4. wall time AND the load-insensitive [GPU profile] buckets side by side
#
# Usage: scripts/bench_reports_batch.sh <E1|E2|E3> [ROUNDS]
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
TP="${INTFER_TP:-1}"
BATCH="${1:-}"
ROUNDS="${2:-3}"
QLOAD="${INTFER_QUIET_LOAD:-2.0}"
QWAIT="${INTFER_QUIET_WAIT:-3600}"
LOG="${INTFER_BENCH_LOG:-$ROOT/build/bench_${BATCH}.log}"

[ -n "$BATCH" ] || { echo "usage: $0 <E1|E2|E3> [ROUNDS]"; exit 1; }
[ -x "$BIN" ] || { echo "FAIL: cli missing: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL"; exit 1; }
python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null \
  || { echo "SKIP: no render-node access from this shell."; exit 2; }
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"

NDEV="$(sycl-ls 2>/dev/null | grep -c '\[level_zero:gpu\]')"
echo "== batch $BATCH  model=$(basename "$MODEL")  tp=$TP  devices=$NDEV =="

# ---- arm table: name -> env assignments. env -u is used at run time so no
# arm can inherit another arm's flags.
case "$BATCH" in
  E1) ARMS="BASE DPAS48 PREQ"
      arm_env_BASE=""; arm_env_DPAS48="INTFER_DPAS48=1"; arm_env_PREQ="INTFER_DPAS48=1 INTFER_DPAS_PREQ=1"
      # invariant: PREQ must equal DPAS48 (both quantize); BASE is EXPECTED to differ
      REF_ARM="DPAS48"; ;;
  E2) ARMS="BASE KV KVQ8"
      arm_env_BASE=""; arm_env_KV="INTFER_INT8_KV=1"
      arm_env_KVQ8="INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1"
      REF_ARM="BASE" ;;   # all three must be byte-identical
  E3) ARMS="BASE DPAS48 FOLD FOLDPREQ"
      arm_env_BASE=""; arm_env_DPAS48="INTFER_DPAS48=1"
      arm_env_FOLD="INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1"
      arm_env_FOLDPREQ="INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1 INTFER_DPAS_PREQ=1"
      REF_ARM="DPAS48" ;;  # FOLD/FOLDPREQ must equal DPAS48
  *) echo "FAIL: unknown batch '$BATCH'"; exit 1 ;;
esac

PROMPT="${INTFER_PROMPT:-The capital of France is}"
run_arm() {  # $1=arm $2=ntok  -> stdout = raw CLI output
  local arm="$1" ntok="$2" e
  e="$(eval echo "\$arm_env_$arm")"
  # shellcheck disable=SC2086
  env $e timeout 900 "$BIN" --gpu --tp "$TP" -m "$MODEL" \
    -p "$PROMPT" -n "$ntok" --no-stream $TOK 2>&1
}

load1() { awk '{print $1}' /proc/loadavg; }

# ---------------------------------------------------------------- 2. identity
echo; echo "== STREAM IDENTITY (before any timing) =="
declare -A IDS
for arm in $ARMS; do
  out="$(run_arm "$arm" 8)"
  ids="$(printf '%s\n' "$out" | grep -ao 'decode step [0-9]*: token=[0-9]*' | grep -o '[0-9]*$' | tr '\n' ' ')"
  IDS[$arm]="$ids"
  printf '  %-9s %s\n' "$arm" "${ids:-<no ids>}"
done
REF="${IDS[$REF_ARM]}"
IDENT=0
# ANTI-VACUITY: an arm that produced NO ids must never compare equal. An empty
# stream used to satisfy `"" == ""` and print "OK", i.e. the check passed
# precisely when the run had failed (e.g. KV capacity abort). Every arm must
# produce a non-empty id list before any comparison is meaningful.
EMPTY=""
for arm in $ARMS; do
  if [ -z "${IDS[$arm]}" ]; then
    echo "  FAIL  $arm produced NO token ids (run failed - see output below)"
    EMPTY="$EMPTY $arm"
  fi
done
if [ -n "$EMPTY" ]; then
  echo
  echo "RESULT: identity check is VACUOUS for:$EMPTY - the run produced nothing."
  echo "RESULT: (typical cause: prompt exceeds KV capacity; raise INTFER_KV_CAP)"
  exit 1
fi
for arm in $ARMS; do
  [ "$arm" = "$REF_ARM" ] && continue
  if [ "${IDS[$arm]}" = "$REF" ]; then
    echo "  OK    $arm == $REF_ARM"
  elif [ "$arm" = "BASE" ] && [ "$REF_ARM" != "BASE" ]; then
    # Derived, not batch-named: whenever the reference arm is an activating
    # (quantizing) arm, the unquantized BASE is EXPECTED to differ from it.
    # Hard-coding this to one batch name is how E3 wrongly FAILED on this line.
    echo "  OK    BASE differs from $REF_ARM (EXPECTED: $REF_ARM quantizes acts)"
  else
    echo "  FAIL  $arm != $REF_ARM  <-- stream divergence, timings below are meaningless"
    IDENT=1
  fi
done
[ "$IDENT" -eq 0 ] || { echo; echo "RESULT: STREAM DIVERGENCE - stop, do not report timings."; exit 1; }

# ---------------------------------------------------------------- quiet gate
if [ "${INTFER_BENCH_TIMING:-1}" != "1" ]; then
  echo
  echo "== COUNTERS-ONLY MODE (INTFER_BENCH_TIMING=0) =="
  echo "Wall-time tok/s is DELIBERATELY NOT produced: the box is contended, so it"
  echo "would be variance, not signal. Reported below are load-insensitive counters"
  echo "(device-side kernel ms, nk/nw/nr) plus the stream-identity verdict above."
  echo "load1 at start = $(load1)"
  CNTLOG="$LOG.counters"
  : > "$CNTLOG"
  for arm in $ARMS; do
    out="$(run_arm "$arm" 64)"
    line="$(printf '%s\n' "$out" | grep -aoE '@decode\] .*' | head -1)"
    printf 'ARM=%s TPS=%s LOAD1=%s BUCKETS=%s\n' \
      "$arm" "SKIPPED-CONTENDED" "$(load1)" "$line" >> "$CNTLOG"
    echo "  $arm  $line"
  done
  LOG="$CNTLOG"
else
echo; echo "== quiet gate: waiting for load1 < $QLOAD (max ${QWAIT}s) =="
waited=0
while [ "$waited" -lt "$QWAIT" ]; do
  L="$(load1)"
  if awk -v l="$L" -v q="$QLOAD" 'BEGIN{exit !(l<q)}'; then echo "quiet: load1=$L after ${waited}s"; break; fi
  [ $((waited % 300)) -eq 0 ] && echo "... load1=$L after ${waited}s"
  sleep 30; waited=$((waited+30))
done
L="$(load1)"
if ! awk -v l="$L" -v q="$QLOAD" 'BEGIN{exit !(l<q)}'; then
  echo "WARN: box never quieted (load1=$L >= $QLOAD) - wall deltas are VARIANCE, not signal."
fi

# ---------------------------------------------------------------- 3. timing
fi
if [ "${INTFER_BENCH_TIMING:-1}" = "1" ]; then
: > "$LOG"
echo; echo "== TIMING: decode -n64, interleaved, $ROUNDS rounds =="
for r in $(seq 1 "$ROUNDS"); do
  for arm in $ARMS; do
    out="$(run_arm "$arm" 64)"
    tps="$(printf '%s\n' "$out" | grep -aoE '\([0-9.]+ tok/s\)' | head -1 | tr -d '() tok/s')"
    line="$(printf '%s\n' "$out" | grep -aoE '@decode\] .*' | head -1)"
    # one machine-readable record per run: ARM, round, tok/s, load1, buckets
    printf 'ARM=%s ROUND=%s TPS=%s LOAD1=%s BUCKETS=%s\n' \
      "$arm" "$r" "${tps:-NA}" "$(load1)" "$line" >> "$LOG"
    echo "  r$r $arm  load1=$(load1)  ${tps:-NA} tok/s"
    echo "     $line"
  done
done

# ---------------------------------------------------------------- 4. summary
echo; echo "== SUMMARY (wall tok/s per round, then median; buckets are load-insensitive) =="
python3 - "$LOG" "$ARMS" <<'PY'
import re, sys, statistics
log, arms = sys.argv[1], sys.argv[2].split()
recs = []
for line in open(log, errors="replace"):
    if not line.startswith("ARM="):
        continue
    d = dict(kv.split("=", 1) for kv in line.split())
    recs.append(d)

def num(rec, key, pat):
    m = re.search(pat, rec.get(key, "") or "")
    return float(m.group(1)) if m else None

for a in arms:
    rs = [r for r in recs if r["ARM"] == a]
    tps = []
    for r in rs:
        try:
            tps.append(float(r["TPS"]))
        except (ValueError, KeyError):
            pass
    # bucket medians: nk/nw/nr are exact counters, kernel is device-side ms
    def medb(pat):
        v = [num(r, "BUCKETS", pat) for r in rs]
        v = [x for x in v if x is not None]
        return statistics.median(v) if v else None
    nk, nw, nr = medb(r"\bnk=(\d+)"), medb(r"\bnw=(\d+)"), medb(r"\bnr=(\d+)")
    kern, subm, wait, copy, rb, usm = (medb(r"\bkernel=([\d.]+)ms"), medb(r"\bsubmit=([\d.]+)ms"),
                                       medb(r"\bwaitblock=([\d.]+)ms"), medb(r"\bcopy=([\d.]+)ms"),
                                       medb(r"\breadback=([\d.]+)ms"), medb(r"\busm_write=([\d.]+)ms"))
    f = lambda x, w=6: ("%*.*f" % (w, w, x)) if x is not None else " " * (w - 1) + "-"
    g = lambda x: ("%d" % x) if x is not None else "-"
    if not rs:
        print(f"{a:9s} (no data)"); continue
    if tps:
        print(f"{a:9s} tok/s {'  '.join('%6.2f' % t for t in tps)}  median={statistics.median(tps):6.2f}")
    else:
        print(f"{a:9s} (wall-time withheld: box contended)")
    print(f"{'':9s}   nk={g(nk):>6} nw={g(nw):>6} nr={g(nr):>6} kernel={f(kern)}ms submit={f(subm)}ms "
          f"wait={f(wait)}ms copy={f(copy)}ms rb={f(rb)}ms usmw={f(usm)}ms")
if recs:
    print(f"\n(n={len(recs)} runs, load1 at each run is in {log})")
PY
echo
echo "RESULT ($BATCH, $(basename "$MODEL") tp=$TP, devices=$NDEV) - NO wall-time claimed:"
echo "RESULT:"
fi
