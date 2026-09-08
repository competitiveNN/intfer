#!/usr/bin/env bash
set -o pipefail
cd /var/home/fra/dev/intfer
MERGE=./build/examples/quant_merge/quant_merge
OUT=/var/home/fra/data/models/intfer/Qwen3.8-27B.intifer
CHUNKS=test/quant_27b
mkdir -p "$CHUNKS"

# Generate 16 chunk commands for layers 0-31 (8 pairs)
# Re-quantize the missing first half (0..15)
for start in $(seq 0 2 14); do
  end=$((start + 2))
  if [ ! -f "$CHUNKS/${start}_${end}.intifer" ]; then
    echo "Quantizing $start..$end"
    OMP_NUM_THREADS=4 nohup setsid ./build_quant_real/quant_model_real \
      /var/home/fra/data/models/safetensors/Qwen3.8-27B/model.safetensors.index.json \
      --out "$CHUNKS/${start}_${end}.intifer" \
      --layer-start $start --layer-count 2 --max-mem-frac 0.80 > "$CHUNKS/${start}_${end}.log" 2>&1 < /dev/null &
  fi
done
