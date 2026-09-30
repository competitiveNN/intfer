#!/usr/bin/env bash
# Full 32-chunk INT2 re-quant of Qwen3.8-27B at uniform budget-frac 0.29.
#
# Evidence for per-class frac (test/quant_int2/probe_*.log):
#   chunk 0 (EMBED+LM_HEAD, 18 tensors): 0.28 infeas, 0.285 REVIEW rc=2,
#     0.29 PASS rc=0 (INT2=4)
#   LIN+SELF chunks (16 tensors, start%4==2): 0.29 PASS (INT2=1..5)
#   LIN+LIN chunks (17 tensors, start%4==0, start>0): 0.302 infeas,
#     0.305 PASS (INT2=6), 0.32 PASS (INT2=3). Arch: layer%4==3 uses
#     self-attn, all other layers linear_attn.
# Chunks go to /data scratch (home disk is full); serial (-P1, ~2.5min/chunk).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd /var/home/fra/dev/intfer
BIN="${INTFER_QUANT_REAL_BIN:-$ROOT/build_quant_real/quant_model_real}"
SRC=/var/home/fra/data/models/safetensors/Qwen3.8-27B/model.safetensors.index.json
OUT=/var/home/fra/data/models/intfer/chunks_int2
LAYER_PER_CHUNK=2
TOTAL_LAYERS=64
FRAC_DEFAULT=0.29   # chunk 0 + LIN+SELF chunks
FRAC_LINLIN=0.305   # LIN+LIN chunks (start%4==0, start>0): edge 0.302/0.305

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
  # Per-class budget: LIN+LIN chunks (start%4==0, start>0) need 0.305.
  local frac=$FRAC_DEFAULT
  if [ "$start" -gt 0 ] && [ $((start % 4)) -eq 0 ]; then
    frac=$FRAC_LINLIN
  fi
  echo "[start] chunk $start..$end (frac=$frac) at $(date +%H:%M:%S)"
  local rc=0
  if [ "$start" -eq 0 ]; then
    "$BIN" "$SRC" --out "$out" --layer-start "$start" --layer-count $((end - start)) --max-mem-frac 0.80 --budget-frac $frac >"$log" 2>&1
    rc=$?
  else
    "$BIN" "$SRC" --out "$out" --layer-start "$start" --layer-count $((end - start)) --max-mem-frac 0.80 --budget-frac $frac --no-embed >"$log" 2>&1
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
export BIN SRC OUT FRAC_DEFAULT FRAC_LINLIN

JOBS=()
for ((start=0; start<TOTAL_LAYERS; start+=LAYER_PER_CHUNK)); do
  end=$((start + LAYER_PER_CHUNK))
  JOBS+=("$start $end")
done

printf '%s\n' "${JOBS[@]}" | xargs -n2 -P1 bash -c 'run_chunk "$@"' _

echo "==== All chunks done ===="
ls -la "$OUT"/*.intifer 2>/dev/null
grep -a -h -oE "scheme dist.*" "$OUT"/*.log 2>/dev/null
