#!/usr/bin/env bash
# Oracle-equivalence gate for non-bit-identical CPU math (FMA dot, FP16-W).
# Bit-parity (test_cpu_parity.sh) cannot apply when FP rounding order changes;
# the arbiter is the greedy token stream: prefill top-1 + full decode ids must
# match the HF oracle (11751 / 13 198 760) AND the scalar baseline on a long
# prompt (prefill top-1 + 8 decode ids).
# MODEL: the canonical .intfer file (NOT .intifer — see
# memories/model_files_intfer_vs_intifer.md; one-letter name difference,
# different greedy streams).
# Usage: VAR_ENV="K=V ..." scripts/test_oracle_equiv.sh
#   e.g. VAR_ENV="INTFER_CPU_FMA_DOT=1" scripts/test_oracle_equiv.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intfer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PROMPT="The capital of France is"
PLONG="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
export TMPDIR="${TMPDIR:-/tmp}"
PASS=1
fail() { echo "GATE FAIL: $1" >&2; PASS=0; }
ids_of() { # $1=prompt $2=n -- prints "prefill decode..." ids
  "$BIN" --no-gpu -m "$MODEL" -p "$1" -n "$2" --no-stream --ids \
    --tokenizer-py "$SCRIPT_DIR/bpe_tokenize.py" \
    --detok-py "$SCRIPT_DIR/bpe_detokenize.py" 2>&1 >/dev/null \
    | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' | tr '\n' ' '
}
# Variant runs cannot go through `env` (ids_of is a shell function), so the
# caller passes VAR_ENV="K=V K2=V2" and we eval-apply it inside this shell.
VAR_ENV="${VAR_ENV:-}"
BASE_SHORT="$(ids_of "$PROMPT" 4)"
BASE_LONG="$(INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 INTFER_BATCH_GEMM=1 ids_of "$PLONG" 9)"
echo "baseline short: $BASE_SHORT"
echo "baseline long : $BASE_LONG"
[ "$BASE_SHORT" = "11751 13 198 760 " ] || fail "baseline short drift: '$BASE_SHORT'"
# shellcheck disable=SC2086
if [ -n "$VAR_ENV" ]; then eval "export $VAR_ENV"; fi
echo "variant env : ${VAR_ENV:-(none)}"
VAR_SHORT="$(ids_of "$PROMPT" 4)"
VAR_LONG="$(INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 INTFER_BATCH_GEMM=1 ids_of "$PLONG" 9)"
echo "variant  short: $VAR_SHORT"
echo "variant  long : $VAR_LONG"
[ "$VAR_SHORT" = "$BASE_SHORT" ] || fail "short stream diverged: '$VAR_SHORT' vs '$BASE_SHORT'"
[ "$VAR_LONG" = "$BASE_LONG" ] || fail "long stream diverged: '$VAR_LONG' vs '$BASE_LONG'"
if [ "$PASS" = "1" ]; then echo "ORACLE-EQUIV TEST: PASS ($*)"; else echo "ORACLE-EQUIV TEST: FAIL ($*)" >&2; exit 1; fi
