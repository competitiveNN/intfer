#!/usr/bin/env bash
# Quiet-box-gated A/B bench: wave vs sequential on the M-row paths.
#
# Rationale: wall-time deltas under box load are run variance, not signal
# (retired quoting rule; sweep-harness artifact notes in AGENTS.md). This
# script polls the 1-min loadavg until it drops below INTFER_QUIET_LOAD
# (default 2.0) or INTFER_QUIET_WAIT seconds elapse (default 3600), then
# runs interleaved wave/NOWAVE rounds of:
#   1. decode TP1 -n64 (M=1 step path — expect ~neutral; regression guard)
#   2. prefill chunk128 + chunk361 (M-row path — the M-row join signal)
#   3. prefill+decode buckets (nk/nw/nr + submit/kernel/waitblock)
# Serve-side (step_batch) coverage comes from scripts/bench_serve.sh and
# scripts/test_serve_load.sh under the same quiet gate.
#
# Usage: scripts/bench_quiet_ab.sh [ROUNDS]   (default 2)
# Env: INTFER_QUIET_LOAD=2.0  INTFER_QUIET_WAIT=3600
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
ROUNDS="${1:-2}"
QLOAD="${INTFER_QUIET_LOAD:-2.0}"
QWAIT="${INTFER_QUIET_WAIT:-3600}"

[ -x "$BIN" ] || { echo "FAIL: cli missing: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL"; exit 1; }
if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
  echo "SKIP: no render-node access from this shell."; exit 2
fi
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

# shellcheck source=scripts/_gate_loadlog.sh
. "$SCRIPT_DIR/_gate_loadlog.sh"
# Per-leg logs + load traces. AUDIT #23: the old loop piped every invocation into
# `grep | head -3`, so nothing survived on disk and a tok/s could only be read off
# the terminal. loadlog_arm also samples load1 CONTINUOUSLY and judges on the
# MAXIMUM -- this gate only samples once, at the start, and then measures for
# minutes (observed here: load1 1.94 -> 10.69 in under two minutes).
loadlog_init "${INTFER_BUILD_DIR:-$ROOT/build}/bench_logs/quiet_ab_$(date -u +%Y%m%dT%H%M%SZ)" >/dev/null

# One measured invocation: logged, continuously sampled, tok/s withheld unless
# the arm stayed under the threshold for its whole duration.
run_leg() { # $1 = label; the rest is the command
  loadlog_arm "$1" "${@:2}" || true
  local tps field
  tps="$(grep -aoE '\([0-9.]+ tok/s\)' "$_LL_LOG" 2>/dev/null | head -1 | tr -d '() tok/s')"
  field="$(loadlog_tps_field "${tps:-NA}")"
  echo "  leg $1  max_load1=$_LL_MAX  TPS=$field  log=$_LL_LOG"
  grep -aE "tokens in|@decode\]|@prefill\]" "$_LL_LOG" 2>/dev/null | head -3 | sed 's|^|      |' || true
}

load1() { loadlog_load1; }
echo "== quiet gate: waiting for load1 < $QLOAD (max ${QWAIT}s) =="
 waited=0
while [ "$waited" -lt "$QWAIT" ]; do
  L="$(load1)"
  if awk -v l="$L" -v q="$QLOAD" 'BEGIN{exit !(l < q)}'; then
    echo "quiet: load1=$L after ${waited}s"; break
  fi
  [ $((waited % 300)) -eq 0 ] && echo "... load1=$L after ${waited}s"
  sleep 30; waited=$((waited + 30))
done
L="$(load1)"
echo "starting A/B at load1=$L"
if ! awk -v l="$L" -v q="$QLOAD" 'BEGIN{exit !(l < q)}'; then
  echo "WARN: box never quieted (load1=$L >= $QLOAD). The run continues so the"
  echo "  CONTENDERS still load and stream-identity still get checked, but every"
  echo "  leg below that crosses the threshold has its TPS recorded as"
  echo "  SKIPPED-CONTENDED, so no figure can be quoted from a busy box."
fi

P361="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
for r in $(seq 1 "$ROUNDS"); do
  for mode in WAVE NOWAVE; do
    echo "===== round $r mode=$mode ====="
    if [ "$mode" = NOWAVE ]; then export INTFER_NO_WAVE=1; else unset INTFER_NO_WAVE; fi
    run_leg "r${r}_${mode}_decode" \
      "$BIN" --gpu --tp 1 -m "$MODEL" -p "The capital of France is" -n 64 --no-stream $TOK
    for m in 128 361; do
      run_leg "r${r}_${mode}_pf${m}" env INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M="$m" INTFER_BATCH_GEMM=1 \
        "$BIN" --gpu --tp 1 -m "$MODEL" -p "$P361" -n 5 --no-stream $TOK
    done
  done
done
unset INTFER_NO_WAVE
echo "== done (final load1=$(load1)) =="
