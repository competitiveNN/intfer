#!/usr/bin/env bash
set -o pipefail
cd /var/home/fra/dev/intfer

MERGE=./build/examples/quant_merge/quant_merge
CHUNKS=test/quant_27b
OUT=/var/home/fra/data/models/intfer/Qwen3.8-27B.intifer
INTER=/var/home/fra/dev/intfer/test/quant_27b/intermediate
mkdir -p "$INTER"

# All 32 chunks in layer order
ALL="0_2 2_4 4_6 6_8 8_10 10_12 12_14 14_16 16_18 18_20 20_22 22_24 24_26 26_28 28_30 30_32 32_34 34_36 36_38 38_40 40_42 42_44 44_46 46_48 48_50 50_52 52_54 54_56 56_58 58_60 60_62 62_64"

# Verify all chunks exist
for c in $ALL; do
  f="$CHUNKS/${c}.intifer"
  if [ ! -f "$f" ]; then
    echo "FAIL: $f missing"
    exit 1
  fi
done

# Split into 4 groups of 8
GROUPS=(
  "0_2 2_4 4_6 6_8 8_10 10_12 12_14 14_16"
  "16_18 18_20 20_22 22_24 24_26 26_28 28_30 30_32"
  "32_34 34_36 36_38 38_40 40_42 42_44 44_46 46_48"
  "48_50 50_52 52_54 54_56 56_58 58_60 60_62 62_64"
)

# Pass 1: merge each group into an intermediate file
for i in "${!GROUPS[@]}"; do
  group="${GROUPS[$i]}"
  inter="$INTER/intfer_pass1_$i.intifer"
  ARGS=""
  for c in $group; do
    ARGS="$ARGS $CHUNKS/${c}.intifer"
  done
  echo "Pass 1 group $i: $group"
  /usr/bin/time -v "$MERGE" "$inter" $ARGS 2>&1 | tee "$INTER/group_$i.log"
  echo "Pass 1 group $i done."
done

# Pass 2: merge all 4 intermediates into final
echo "Pass 2: merging 4 intermediates"
INTR_ARGS=""
for i in "${!GROUPS[@]}"; do
  INTR_ARGS="$INTR_ARGS $INTER/intfer_pass1_$i.intifer"
done
/usr/bin/time -v "$MERGE" "$OUT" $INTR_ARGS 2>&1 | tee "$INTER/pass2.log"

echo "=== FINAL ==="
ls -lh "$OUT"
