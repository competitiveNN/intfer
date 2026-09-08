#!/usr/bin/env bash
set -o pipefail
cd /var/home/fra/dev/intfer

MERGE=./build/examples/quant_merge/quant_merge
CHUNKS=test/quant_27b
EXISTING=/var/home/fra/data/models/intfer/Qwen3.8-27B.intifer
OUT=/var/home/fra/data/models/intfer/Qwen3.8-27B.intifer.new

# Verify chunks 0-15 exist (newly re-quantized)
for c in 0_2 2_4 4_6 6_8 8_10 10_12 12_14 14_16; do
  f="$CHUNKS/${c}.intifer"
  if [ ! -f "$f" ]; then
    echo "FAIL: $f missing"
    exit 1
  fi
done
if [ ! -f "$EXISTING" ]; then
  echo "FAIL: $EXISTING missing"
  exit 1
fi

# Args: new chunks 0-15 (in layer order) + existing 16-31 file
ARGS=""
for c in 0_2 2_4 4_6 6_8 8_10 10_12 12_14 14_16; do
  ARGS="$ARGS $CHUNKS/${c}.intifer"
done
ARGS="$ARGS $EXISTING"

echo "Streaming merge: 8 new chunks (0-15) + existing 16-31 final (16 inputs)..."

/usr/bin/time -v "$MERGE" "$OUT" $ARGS 2>&1 | tee test/quant_27b/merge_final.log

echo "=== FINAL ==="
ls -lh "$OUT"
