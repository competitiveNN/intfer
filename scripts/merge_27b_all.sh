#!/usr/bin/env bash
# Streaming merge of all 32 2-layer chunks into the final 27B .intifer.
# Uses quant_merge's dedup + streaming design (3.5 GB peak vs 57 GB naive).
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd /var/home/fra/dev/intfer

MERGE="${INTFER_BUILD_DIR:-$ROOT/build}/examples/quant_merge/quant_merge"
CHUNKS=test/quant_27b
OUT=/var/home/fra/data/models/intfer/Qwen3.8-27B.intifer

# Verify all 32 chunks exist
MISSING=0
for ((s=0; s<64; s+=2)); do
  e=$((s+2))
  f="$CHUNKS/${s}_${e}.intifer"
  if [ ! -f "$f" ]; then
    echo "FAIL: $f missing"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo "FATAL: some chunks missing. Run scripts/run_27b_chunks.sh first."
  exit 1
fi

# Args: all 32 chunks in layer order
ARGS=""
for ((s=0; s<64; s+=2)); do
  e=$((s+2))
  ARGS="$ARGS $CHUNKS/${s}_${e}.intifer"
done

echo "Streaming merge: 32 chunks (layers 0..63)..."
echo "Output: $OUT"
/usr/bin/time -v "$MERGE" "$OUT" $ARGS 2>&1 | tee test/quant_27b/merge_final.log

echo "=== FINAL ==="
ls -lh "$OUT"

# Verify deserialization
echo "=== VERIFY ==="
g++ -std=c++17 -O2 -fno-builtin -I include -c src/quant/output.cpp -o /tmp/output_fresh.o 2>/dev/null
g++ -std=c++17 -O2 -fno-builtin -I include /tmp/verify_merge.cpp -o /tmp/verify_merge \
  /tmp/output_fresh.o build/src/libintfer_quant.a build/src/libintfer_common.a build/src/libintfer_l0.a \
  -lze_loader -lrt -lpthread 2>/dev/null
/tmp/verify_merge "$OUT" 2>&1 | head -5
