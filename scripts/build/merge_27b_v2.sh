#!/usr/bin/env bash
# Merge the 32 chunks from scripts/run/run_27b_int2_v2.sh into the final .intifer.
#
# quant_merge dedups by tensor name (examples/quant_merge/main.cpp:431 prints
# "dropped N duplicate names"), which is what makes chunk 0 carrying
# EMBED + LM_HEAD safe: no other chunk has them (--no-embed --no-lm-head), so
# there is exactly one copy of each in the output.
#
# Output is a NEW filename so the 19 GB Qwen3.8-27B-int2.intifer is preserved
# for comparison.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd /var/home/fra/dev/intfer

MERGE="${INTFER_BUILD_DIR:-$ROOT/build}/examples/quant_merge/quant_merge"
CHUNKS="${INTFER_CHUNK_DIR:-/var/home/fra/data/models/intfer/chunks_v2}"
OUT="${INTFER_MERGE_OUT:-/var/home/fra/data/models/intfer/Qwen3.8-27B-int2-v2.intifer}"

[ -x "$MERGE" ] || { echo "FAIL: merge binary missing: $MERGE" >&2; exit 1; }

ARGS=""
missing=0
for ((s=0; s<64; s+=2)); do
  f="$CHUNKS/${s}_$((s+2)).intifer"
  if [ ! -f "$f" ]; then echo "FAIL: missing $f" >&2; missing=1; fi
  ARGS="$ARGS $f"
done
[ "$missing" -eq 0 ] || { echo "FAIL: chunks incomplete; not merging" >&2; exit 1; }

echo "Merging 32 chunks from $CHUNKS"
echo "Output: $OUT"
"$MERGE" "$OUT" $ARGS
rc=$?
echo "MERGE_RC=$rc"
[ $rc -eq 0 ] || exit $rc

echo "=== FINAL ==="
ls -lh "$OUT"
echo "size_bytes=$(stat -c%s "$OUT")"
