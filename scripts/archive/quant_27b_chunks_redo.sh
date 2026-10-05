#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd /var/home/fra/dev/intfer
BIN="${INTFER_QUANT_REAL_BIN:-$ROOT/build_quant_real/quant_model_real}"
SRC=/var/home/fra/data/models/safetensors/Qwen3.8-27B/model.safetensors.index.json
OUT=test/quant_27b
export OMP_NUM_THREADS=4
mkdir -p "$OUT"

for start in 0 2 4 6 8 10 12 14; do
  end=$((start + 2))
  if [ -f "$OUT/${start}_${end}.intifer" ] && [ "$(stat -c%s "$OUT/${start}_${end}.intifer")" -gt 1000000000 ]; then
    echo "[skip] ${start}_${end} exists"
    continue
  fi
  echo "[start] chunk $start..$end"
  if [ "$start" -eq 0 ]; then
    "$BIN" "$SRC" --out "$OUT/${start}_${end}.intifer" --layer-start $start --layer-count 2 --max-mem-frac 0.80 > "$OUT/${start}_${end}.log" 2>&1
  else
    "$BIN" "$SRC" --out "$OUT/${start}_${end}.intifer" --layer-start $start --layer-count 2 --max-mem-frac 0.80 --no-embed > "$OUT/${start}_${end}.log" 2>&1
  fi
  echo "[done] $start..$end rc=$? size=$(du -h "$OUT/${start}_${end}.intifer" 2>/dev/null | cut -f1)"
done
echo "all done"
