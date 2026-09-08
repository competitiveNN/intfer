#!/usr/bin/env bash
# Full 32-chunk INT2 re-quant of Qwen3.8-27B at uniform budget-frac 0.29.
#
# Evidence for 0.29 (test/quant_int2/probe_*.log):
#   chunk 0 (hardest: EMBED+LM_HEAD, fp16 6311.9 MiB):
#     0.27 infeasible | 0.28 infeasible | 0.285 feasible REVIEW (rc=2)
#     0.29 feasible PASS rc=0 (INT2=4) | 0.30 feasible PASS (INT2=1)
#   chunk 30-32 @0.27: feasible PASS (INT2=5)
# Chunks go to /data scratch (home disk is full); serial (-P1, ~2.5min/chunk).
set -o pipefail

cd /var/home/fra/dev/intfer
BIN=./build_quant_real/quant_model_real
SRC=/var/home/fra/data/models/safetensors/Qwen3.8-27B/model.safetensors.index.json
OUT=/var/home/fra/data/models/intfer/chunks_int2
LAYER_PER_CHUNK=2
TOTAL_LAYERS=64
FRAC=0.29

export OMP_NUM_THREADS=4
export TMPDIR=/var/home/fra/build_tmp

mkdir -p "$OUT"

run_chunk() {
  local start=$1
  local end=$2
  local out="$OUT/${start}_${end}.intifer"
  local log="$OUT/${start}_${end}.log"
  if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 500000000 ]; then
    echo "[skip] chunk $start..$end already exists"
    return 0
  fi
  echo "[start] chunk $start..$end (frac=$FRAC) at $(date +%H:%M:%S)"
  local rc=0
  if [ "$start" -eq 0 ]; then
    "$BIN" "$SRC" --out "$out" --layer-start "$start" --layer-count $((end - start)) --max-mem-frac 0.80 --budget-frac $FRAC >"$log" 2>&1
    rc=$?
  else
    "$BIN" "$SRC" --out "$out" --layer-start "$start" --layer-count $((end - start)) --max-mem-frac 0.80 --budget-frac $FRAC --no-embed >"$log" 2>&1
    rc=$?
  fi
  if [ $rc -ne 0 ]; then
    echo "[FAIL ] chunk $start..$end rc=$rc at $(date +%H:%M:%S) (log: $log)"
    tail -5 "$log"
    return $rc
  fi
  local verdict
  verdict=$(grep -a -oE "VERDICT.*" "$log" | head -1)
  local dist
  dist=$(grep -a -oE "scheme dist.*" "$log" | head -1)
  echo "[done ] chunk $start..$end -> $(du -h "$out" | cut -f1) | $dist | $verdict"
}
export -f run_chunk
export BIN SRC OUT FRAC

JOBS=()
for ((start=0; start<TOTAL_LAYERS; start+=LAYER_PER_CHUNK)); do
  end=$((start + LAYER_PER_CHUNK))
  JOBS+=("$start $end")
done

printf '%s\n' "${JOBS[@]}" | xargs -n2 -P1 bash -c 'run_chunk "$@"' _

echo "==== All chunks done ===="
ls -la "$OUT"/*.intifer 2>/dev/null
grep -a -h -oE "scheme dist.*" "$OUT"/*.log 2>/dev/null
