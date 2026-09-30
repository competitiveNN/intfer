#!/usr/bin/env bash
# Run 32 2-layer chunks of 27B serially (peak ~25GB per chunk).
# Skips MTP tensors, keeps embed/lm_head only in chunk 0.
# As of 2026-09-03: includes linear_attn roles (QKV/Z/B/A/OUT) so the
# SSM tensors (the bulk of the 27B) are quantized alongside self_attn/MLP.
set -o pipefail

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd /var/home/fra/dev/intfer
BIN="${INTFER_QUANT_REAL_BIN:-$ROOT/build_quant_real/quant_model_real}"
SRC=/var/home/fra/data/models/safetensors/Qwen3.8-27B/model.safetensors.index.json
OUT=test/quant_27b
LAYER_PER_CHUNK=2
TOTAL_LAYERS=64

export OMP_NUM_THREADS=4

mkdir -p "$OUT"

run_chunk() {
  local start=$1
  local end=$2
  local out="$OUT/${start}_${end}.intifer"
  local log="$OUT/${start}_${end}.log"
  if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 1000000000 ]; then
    echo "[skip] chunk $start..$end already exists"
    return 0
  fi
  echo "[start] chunk $start..$end (pid=$$) at $(date +%H:%M:%S)"
  local rc=0
  if [ "$start" -eq 0 ]; then
    "$BIN" "$SRC" --out "$out" --layer-start "$start" --layer-count $((end - start)) --max-mem-frac 0.80 --budget-frac 0.40 >"$log" 2>&1
    rc=$?
  else
    "$BIN" "$SRC" --out "$out" --layer-start "$start" --layer-count $((end - start)) --max-mem-frac 0.80 --budget-frac 0.40 --no-embed >"$log" 2>&1
    rc=$?
  fi
  if [ $rc -ne 0 ]; then
    echo "[FAIL ] chunk $start..$end rc=$rc at $(date +%H:%M:%S) (log: $log)"
    tail -10 "$log"
    return $rc
  fi
  local sz=$(du -h "$out" 2>/dev/null | cut -f1)
  echo "[done ] chunk $start..$end -> $sz at $(date +%H:%M:%S)"
}
export -f run_chunk
export BIN SRC OUT

JOBS=()
for ((start=0; start<TOTAL_LAYERS; start+=LAYER_PER_CHUNK)); do
  end=$((start + LAYER_PER_CHUNK))
  JOBS+=("$start $end")
done

printf '%s\n' "${JOBS[@]}" | xargs -n2 -P1 bash -c 'run_chunk "$@"' _

echo "==== All chunks done ===="
ls -la "$OUT"/*.intifer 2>/dev/null
