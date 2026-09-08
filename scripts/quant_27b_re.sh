#!/usr/bin/env bash
# Re-quantize Qwen3.8-27B in 2-layer chunks (embed handled separately).
# Peak RSS ~23GB per chunk; run serially. Skips chunks whose output exists
# and exceeds 1GB.
set -o pipefail
cd /var/home/fra/dev/intfer
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
# OMP=2: peak RSS ~16GB per 2-layer chunk (fits alongside desktop usage).
# OMP=4 peaked at ~23GB and OOMs when available RAM < ~24GB.
export OMP_NUM_THREADS=2
BIN=./build_quant_real/quant_model_real
SRC=/var/home/fra/data/models/safetensors/Qwen3.8-27B/model.safetensors.index.json
OUT=test/quant_27b
mkdir -p "$OUT"
for start in 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62; do
  end=$((start + 2))
  out="$OUT/${start}_${end}.intifer"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 1000000000 ]; then
    echo "[skip] ${start}_${end} exists ($(du -h "$out" | cut -f1))"
    continue
  fi
  echo "[start] chunk ${start}_${end} at $(date +%H:%M:%S)"
  "$BIN" "$SRC" --out "$out" --layer-start $start --layer-count 2 \
    --max-mem-mib 8192 --budget-frac 0.40 --no-embed > "$OUT/${start}_${end}.log" 2>&1
  rc=$?
  echo "[done] ${start}_${end} rc=$rc size=$(du -h "$out" 2>/dev/null | cut -f1) at $(date +%H:%M:%S)"
  if [ $rc -ne 0 ]; then echo "FAIL chunk ${start}_${end}"; tail -5 "$OUT/${start}_${end}.log"; exit 1; fi
done
echo "ALL CHUNKS DONE"
ls -la "$OUT"/*.intifer | wc -l
