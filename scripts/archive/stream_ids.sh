#!/usr/bin/env bash
# Compare decode token streams: usage stream_ids.sh [TP] [N] [ENV...]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TP="${1:-1}"; shift
N="${1:-8}"; shift
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
PROMPT="${INTFER_PROMPT:-The capital of France is}"
# shellcheck disable=SC2086
env "$@" "$BIN" --gpu --tp "$TP" -m "$MODEL" -p "$PROMPT" -n "$N" \
  --no-stream --ids $TOK 2>&1 >/dev/null \
  | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*step ([0-9]+): token=([0-9]+)/\1:\2/' \
  | tr '\n' ' '; echo
