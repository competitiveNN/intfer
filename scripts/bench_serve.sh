#!/usr/bin/env bash
# Serve throughput benchmark (Stage 3): concurrent requests through --serve.
#
# Feeds N concurrent JSON-lines requests (continuous batching) and reports:
#   - wall time, total output tokens, tok/s
#   - per-request latency (first-to-last token span is not observable from
#     token stream alone, so we report mean tokens/req + throughput)
# Gates: all N requests complete with the expected token count, and
# throughput meets a floor (regression guard, not a record attempt).
#
# Reference numbers (2026-09-13, 0.8B INT2, N=8/4 max_new=5):
#   CPU (--no-gpu): 2.7 tok/s wall, 8 reqs concurrent at ~single-stream rate
#   GPU (--gpu TP=2): 2.4 tok/s (tiny model: launch overhead dominates;
#     tensor-parallelism pays off on 27B, see scripts/test_tp2_27b.sh).
#
# Usage: scripts/bench_serve.sh [N] [MAX_NEW]
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intfer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
N="${1:-8}"
MAX_NEW="${2:-5}"

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}"
REQ="$(mktemp --tmpdir="$TMPDIR" bench_req.XXXX.jsonl)"
OUT="$(mktemp --tmpdir="$TMPDIR" bench_out.XXXX.jsonl)"
trap 'rm -f "$REQ" "$OUT"' EXIT
trap 'rm -f "$REQ" "$OUT"; exit 130' INT
trap 'rm -f "$REQ" "$OUT"; exit 143' TERM

PROMPTS=("The capital of France is" "The quick brown fox jumps over the lazy dog." "Hi" "Hello world")
for ((i = 0; i < N; i++)); do
  P="${PROMPTS[$((i % 4))]}"
  printf '{"id":%d,"prompt":"%s","max_new":%d}\n' "$((i + 1))" "$P" "$MAX_NEW" >>"$REQ"
done

# shellcheck source=scripts/_gate_loadlog.sh
. "$SCRIPT_DIR/_gate_loadlog.sh"
loadlog_init "${INTFER_BUILD_DIR:-$ROOT/build}/bench_logs/serve_$(date -u +%Y%m%dT%H%M%SZ)" >/dev/null

T0=$(date +%s.%N)
# stdout is the SSE stream the token counts below are read from, so it must stay
# in $OUT; stderr and everything else goes to the per-run artifact via loadlog.
loadlog_arm "serve_n${N}" sh -c '"$1" --no-gpu -m "$2" --serve \
  --tokenizer-py "$3" --detok-py "$4" <"$5"' _ \
  "$BIN" "$MODEL" "$SCRIPT_DIR/bpe_tokenize.py" "$SCRIPT_DIR/bpe_detokenize.py" "$REQ" \
  >"$OUT" 2>/dev/null
RC=$?
T1=$(date +%s.%N)
MAXL="$_LL_MAX"
[ "$RC" = "0" ] || { echo "BENCH FAIL: serve exited rc=$RC (log: $_LL_LOG)" >&2; exit 1; }

WALL=$(echo "$T1 $T0" | awk '{printf "%.2f", $1 - $2}')
TOKS=$(grep -a -c '"token"' "$OUT")
IDS_DONE=$(grep -a -oE '"id":[0-9]+' "$OUT" | sort -u | wc -l)
TPS=$(echo "$TOKS $WALL" | awk '{printf "%.2f", $1 / $2}')
TPS_FIELD="$(loadlog_tps_field "$TPS")"
PER_REQ_MS=$(echo "$WALL $N" | awk '{printf "%.0f", $1 * 1000 / $2}')

echo "bench_serve: N=$N max_new=$MAX_NEW wall=${WALL}s tokens=$TOKS ids_done=$IDS_DONE tok/s=$TPS_FIELD max_load1=${MAXL:-NA} mean_lat/req=${PER_REQ_MS}ms"
[ "$IDS_DONE" = "$N" ] || { echo "BENCH FAIL: only $IDS_DONE/$N requests completed" >&2; exit 1; }
EXP_TOKS=$((N * MAX_NEW))
[ "$TOKS" = "$EXP_TOKS" ] || { echo "BENCH FAIL: $TOKS tokens != expected $EXP_TOKS" >&2; exit 1; }

# AUDIT #23: a tok/s measured while the box is busy is run variance, and this
# script's headline IS a tok/s. The serve call is now wrapped so the load is
# sampled for its whole duration and judged on the MAXIMUM -- not sampled once
# before a run that can take minutes. A contended run reports
# SKIPPED-CONTENDED and the throughput floor is NOT applied to it, because the
# floor is a stall guard and on a loaded box it can only produce a false alarm.
if [ "$TPS_FIELD" = "SKIPPED-CONTENDED" ]; then
  echo "BENCH: wall-time WITHHELD (load1 max ${MAXL:-NA} >= $INTFER_QUIET_LOAD)." >&2
  echo "  The stall/token-count assertions above still ran and still mean something;" >&2
  echo "  only the throughput figure is withheld. Re-run when the box is quiet." >&2
  echo "SERVE-BENCH: PASS (counters only; no throughput figure)"
  exit 0
fi

# Throughput floor: 0.8B CPU greedy must sustain >= 1 tok/s (far below the
# ~3-8 tok/s single-stream rate; this guards against scheduler stalls,
# not for records — see REMAINING_TODOS for headroom notes).
echo "$TPS" | awk '{exit ($1 >= 1.0) ? 0 : 1}' \
  || { echo "BENCH FAIL: tok/s=$TPS < floor 1.0" >&2; exit 1; }
echo "SERVE-BENCH: PASS"
