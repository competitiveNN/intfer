#!/usr/bin/env bash
# Merge 32 2-layer chunks into the final Qwen3.8-27B.intifer
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd /var/home/fra/dev/intfer
MERGE="${INTFER_BUILD_DIR:-$ROOT/build}/examples/quant_merge/quant_merge"
OUT=/var/home/fra/data/models/intfer/Qwen3.8-27B.intifer
CHUNKS=test/quant_27b

# Find all chunks sorted
chunks=($(ls -1 "$CHUNKS"/*.intifer | sort -V))

# Filter only complete chunks (> 800 MB)
final=()
for c in "${chunks[@]}"; do
  sz=$(stat -c%s "$c" 2>/dev/null || echo 0)
  if [ "$sz" -gt 800000000 ]; then
    final+=("$c")
  else
    echo "[warn] skipping incomplete $(basename $c) ($sz bytes)"
  fi
done

echo "Found ${#final[@]} complete chunks (need 32)"
if [ "${#final[@]}" -ne 32 ]; then
  echo "ERROR: expected 32 complete chunks, got ${#final[@]}"
  echo "Chunks: ${final[@]}"
  exit 1
fi

# Make sure target dir exists
mkdir -p "$(dirname "$OUT")"

# Free up disk space: remove chunks after successful merge
echo "Merging ${#final[@]} chunks -> $OUT"
"$MERGE" "$OUT" "${final[@]}"
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAIL: merge rc=$rc"
  exit $rc
fi

echo "Done. Final size:"
ls -la "$OUT"

# Cleanup chunks
echo "Removing chunks..."
rm -f "${final[@]}"
ls -la "$CHUNKS"/
