#!/usr/bin/env bash
# GPU A/B bench: decode + prefill throughput on 2x Arc Pro B580.
# Fails fast with the exact remedy when render-node perms are missing.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"

# --- 0. access gate: L0 render node must open, else Level Zero init fails ---
if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
  echo "SKIP: no render-node access from this shell."
  echo "  groups now: $(python3 -c 'import os; print(os.getgroups())') (need 105/render or 39/video)"
  echo "  remedy: fresh login (picks up /etc/group), or root: chgrp render /dev/dri/renderD* && chmod 0660"
  exit 2
fi
if ! /usr/bin/zello_world >/dev/null 2>&1; then
  echo "SKIP: zello_world failed despite node access (driver/runtime issue)."
  exit 2
fi
# NOTE: setvars.sh is not `set -u`-clean; drop nounset across the source
# (it dies on unset-var expansion otherwise and kills this script).
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
[ -x "$BIN" ] || { echo "FAIL: cli missing: $BIN"; exit 1; }

P361="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
echo "== decode tok/s (5-tok prompt, -n 64) =="
for tp in 1 2; do
  echo "--- TP=$tp ---"
  "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "The capital of France is" -n 64 \
    --no-stream $TOK 2>&1 | grep -aE "tokens in|topology|decode:" | head -6
done
echo "== prefill tok/s (361-tok prompt, -n 5) =="
for tp in 1 2; do
  # 361 = whole prompt in one chunk (no chunking overhead; best throughput,
  # worst TTFT — chunked widths stay for the serving tradeoff).
  for m in 8 32 128 361; do
    echo "--- TP=$tp chunk_M=$m ---"
    INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M="$m" INTFER_BATCH_GEMM=1 \
      "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$P361" -n 5 --no-stream $TOK 2>&1 \
      | grep -aE "tokens in" | head -2
  done
done
echo "== GPU sanity spot-check (expect prefill 279 for the .intifer file) =="
# NOTE: this bench uses the self-contained .intifer model, whose greedy
# stream (279/6511/314/279) differs from the HF oracle by design — the
# 11751 oracle lives in the legacy .intfer file (see memories/
# model_files_intfer_vs_intifer.md). This check only guards against a
# broken GPU run (no output / crash), not oracle equivalence.
"$BIN" --gpu --tp 1 -m "$MODEL" -p "The capital of France is" -n 1 --no-stream --ids $TOK 2>&1 \
  | grep -a -oE "decode step 0: token=[0-9]+" | head -1
