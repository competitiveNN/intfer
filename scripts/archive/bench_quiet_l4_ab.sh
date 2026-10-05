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
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
ROUNDS="${1:-3}"
QLOAD="${INTFER_QUIET_LOAD:-2.0}"
# QWAIT / INTFER_QUIET_WAIT are GONE, not defaulted: the blocking wait they
# configured was removed (see the load-notification block below), and leaving a
# dead "QWAIT=0" behind would be a variable that reads as a knob and does nothing.
# scripts/bench_when_quiet.sh is now the single place that waits. Old command lines
# that still export INTFER_QUIET_WAIT are harmless -- an unread env var is not an
# error.

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
# AUDIT #23: per-leg logs + CONTINUOUS load sampling. The old loop piped each
# invocation into `grep | head -3`, so nothing survived on disk and a tok/s could
# only be read off the terminal; and the quiet gate sampled load1 once, at the
# start, then measured for minutes.
loadlog_init "${INTFER_BUILD_DIR:-$ROOT/build}/bench_logs/quiet_l4_$(date -u +%Y%m%dT%H%M%SZ)" >/dev/null

run_leg() { # $1 = label; the rest is the command
  loadlog_arm "$1" "${@:2}" || true
  local tps field
  tps="$(grep -aoE '\([0-9.]+ tok/s\)' "$_LL_LOG" 2>/dev/null | head -1 | tr -d '() tok/s')"
  field="$(loadlog_tps_field "${tps:-NA}")"
  echo "  leg $1  max_load1=$_LL_MAX  TPS=$field  log=$_LL_LOG"
  grep -aE "tokens in|@decode\]|@prefill\]" "$_LL_LOG" 2>/dev/null | head -3 | sed 's|^|      |' || true
}

load1() { loadlog_load1; }
# ---- load: NOTIFY ONLY, NEVER WAIT (the blocking wait was removed 2026-10-02) --
# WHY IT WENT. This loop blocked for up to QWAIT (default 3600s) and then measured
# ANYWAY, so on a box held busy by a long-running neighbour it cost an hour and
# bought nothing. It was also a hand-rolled copy in three harnesses -- the AUDIT #7
# shape. There is now ONE home for "run this when the box is quiet":
# scripts/bench_when_quiet.sh.
#
# *** THE PROTECTION IS A DIFFERENT MECHANISM AND IS UNTOUCHED. *** Whether a figure
# is quotable is decided per-arm by loadlog_arm, on the MAXIMUM load over the whole
# arm -- not here, and not by this print. Removing the wait does not make a contended
# measurement quotable; it makes it visible sooner.
L="$(load1)"
echo "starting L4 A/B at load1=$L (not waiting)"
if awk -v l="$L" -v q="$QLOAD" 'BEGIN{exit !(l<q)}'; then
  echo "load1=$L is under $QLOAD"
else
  echo "load1=$L is >= $QLOAD. NOT waiting -- running anyway."
  echo "  Each leg whose MAXIMUM load crosses $QLOAD has its figure withheld as"
  echo "  SKIPPED-CONTENDED, so a busy box cannot produce a quotable number."
  echo "  For a quiet-window run use scripts/bench_when_quiet.sh."
fi

P361="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
for r in $(seq 1 "$ROUNDS"); do
  for mode in L4 BASE; do
    echo "===== round $r mode=$mode ====="
    if [ "$mode" = L4 ]; then export INTFER_L4_FUSE=1; else unset INTFER_L4_FUSE; fi
    run_leg "r${r}_${mode}_decode" \
      "$BIN" --gpu --tp 1 -m "$MODEL" -p "The capital of France is" -n 64 --no-stream $TOK
    run_leg "r${r}_${mode}_pf361" env INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 INTFER_BATCH_GEMM=1 \
      "$BIN" --gpu --tp 1 -m "$MODEL" -p "$P361" -n 5 --no-stream $TOK
  done
done
unset INTFER_L4_FUSE
echo "== done (final load1=$(load1)) =="
