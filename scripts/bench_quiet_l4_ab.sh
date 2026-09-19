#!/usr/bin/env bash
# Quiet-box-gated A/B bench: L4 la_out fusion (INTFER_L4_FUSE=1) vs base.
#
# Rationale: wall-time deltas under box load are run variance, not signal
# (retired quoting rule; sweep-harness artifact notes in AGENTS.md). This
# script polls the 1-min loadavg until it drops below INTFER_QUIET_LOAD
# (default 2.0) or INTFER_QUIET_WAIT seconds elapse (default 3600), then
# runs interleaved L4/BASE rounds of:
#   1. decode TP1 -n64 (THE signal: L4 fires on TP=1 scalar decode only,
#      ~+10% est = ~4-5ms/token of ~41ms over 18 LA layers)
#   2. prefill chunk361 (regression guard: L4 must be a silent no-op there)
#
# Correctness was gated separately, load-insensitive (AGENTS.md L4-lite):
# L4-vs-base streams identical TP1 n=4/16/32, TP2 no-op, cpu parity +
# http batch + gpu-optin PASS. Do NOT re-claim correctness here.
#
# Usage: scripts/bench_quiet_l4_ab.sh [ROUNDS]   (default 3)
# Env: INTFER_QUIET_LOAD=2.0  INTFER_QUIET_WAIT=3600
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/build/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
ROUNDS="${1:-3}"
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

load1() { awk '{print $1}' /proc/loadavg; }
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
echo "starting L4 A/B at load1=$L"
if ! awk -v l="$L" -v q="$QLOAD" 'BEGIN{exit !(l < q)}'; then
  echo "WARN: box never quieted (load1=$L >= $QLOAD); results are variance, not signal."
fi

P361="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
for r in $(seq 1 "$ROUNDS"); do
  for mode in L4 BASE; do
    echo "===== round $r mode=$mode (load1=$(load1)) ====="
    if [ "$mode" = L4 ]; then export INTFER_L4_FUSE=1; else unset INTFER_L4_FUSE; fi
    echo "--- decode TP1 -n64 (L4 signal path) ---"
    "$BIN" --gpu --tp 1 -m "$MODEL" -p "The capital of France is" -n 64 \
      --no-stream $TOK 2>&1 | grep -aE "tokens in|@decode" | head -3
    echo "--- prefill chunk361 (L4 no-op regression guard) ---"
    INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 INTFER_BATCH_GEMM=1 \
      "$BIN" --gpu --tp 1 -m "$MODEL" -p "$P361" -n 5 --no-stream $TOK 2>&1 \
      | grep -aE "tokens in|@prefill" | head -3
  done
done
unset INTFER_L4_FUSE
echo "== done (final load1=$(load1)) =="
