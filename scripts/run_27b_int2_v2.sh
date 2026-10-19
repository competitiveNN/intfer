#!/usr/bin/env bash
# 32-chunk INT2 re-quant of Qwen3.8-27B at ~10 GB total, using the FIXED
# LINEAR_ATTN_* ladder (src/quant_spec.cpp, 2026-10-18) and --no-lm-head for
# chunks >0.
#
# WHY THE BUDGETS ARE PER-CLASS.
#   The shipped 19 GB artifact was built by scripts/archive/run_27b_int2_chunks.sh
#   with --no-embed only. LM_HEAD carries no layer index, so the layer-range
#   guard never filtered it out and EVERY chunk re-loaded it: a 2-layer LIN+LIN
#   chunk measured fp16 = 3886.9 MiB = lm_head 2428 + layers 1462. Both the
#   chunk's RCO budget and its cheapest allocation therefore grew by lm_head's
#   INT4 floor (643 MiB), so the frac needed to stay feasible was set by lm_head
#   rather than by the layers the chunk actually quantizes. Combined with the
#   FP16-by-omission bug (240 of 498 tensors FP16 by role, no INT2 candidate
#   ever offered), every LIN+LIN chunk refused with "RCO produced an infeasible
#   allocation".
#
#   Chunk 0 keeps EMBED + LM_HEAD (selected exactly once, merged with dedup) and
#   therefore needs frac >= 0.2366 to clear its own INT4 floor:
#     cheapest = embed+lm_head @INT4g64 (1320 MiB) + 2 layers @INT2g32 (228 MiB)
#              = 1548 MiB over a chunk fp16 of 6542 MiB.
#   Chunks >0 exclude both, so a uniform layer-only frac applies.
#
# MEASURED CURVE on chunk 4..6 (2 LIN layers, 1462 MiB fp16, --no-lm-head):
#   frac 0.186 -> selected 259.7 MiB, mean recon 0.2790, REVIEW
#   frac 0.240 -> selected 344.7 MiB, mean recon 0.1995, PASS
#   frac 0.300 -> selected 365.9 MiB, mean recon 0.1811, PASS (saturates)
# FRAC_LAYER 0.19 is chosen to land the whole model at ~10.2 GB, the operator's
# stated budget. Extrapolated total = 32 x ~266 MiB (layers) + 1250 MiB
# (embed+lm_head @INT4) = ~9.7 GiB = ~10.2 GB. Raising FRAC_LAYER to 0.206
# buys back the 0.25 mean-recon PASS threshold at ~11 GB.
#
# Chunks go to /var/home/fra/data (5.5T free); each .intifer keeps the rotated
# FP16 verification copy, so 32 chunks occupy ~21 GB and the merge drops it.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd /var/home/fra/dev/intfer

BIN="${INTFER_QUANT_REAL_BIN:-$ROOT/build_quant_real/quant_model_real}"
SRC=/var/home/fra/data/models/safetensors/Qwen3.8-27B/model.safetensors.index.json
OUT="${INTFER_CHUNK_DIR:-/var/home/fra/data/models/intfer/chunks_v2}"
LAYER_PER_CHUNK=2
TOTAL_LAYERS=64
FRAC_LAYER="${INTFER_FRAC_LAYER:-0.24}"   # chunks 2..64 (layers only) -- knee of the recon curve
FRAC_HEAD="${INTFER_FRAC_HEAD:-0.24}"     # chunk 0 (embed + lm_head + layers 0,1)
MAXMEM="${INTFER_MAXMEM:-0.30}"
JOBS="${INTFER_JOBS:-4}"

export OMP_NUM_THREADS="${INTFER_OMP:-3}"
export TMPDIR=/var/home/fra/build_tmp
mkdir -p "$OUT" "$TMPDIR"

run_chunk() {
  local start=$1 end=$2
  local out_f="$OUT/${start}_${end}.intifer" log="$OUT/${start}_${end}.log"
  if [ -f "$out_f" ] && [ "$(stat -c%s "$out_f" 2>/dev/null || echo 0)" -gt 500000000 ]; then
    echo "[skip] chunk $start..$end already exists"
    return 0
  fi
  local frac="$FRAC_LAYER" extra="--no-embed --no-lm-head"
  if [ "$start" -eq 0 ]; then frac="$FRAC_HEAD"; extra=""; fi  # chunk 0 carries EMBED + LM_HEAD
  echo "[start] chunk $start..$end (frac=$frac) at $(date +%H:%M:%S)"
  local rc=0
  timeout 3600 "$BIN" "$SRC" --layer-start "$start" --layer-count $((end - start)) \
    $extra --budget-frac "$frac" --max-mem-frac "$MAXMEM" \
    --out "$out_f" >"$log" 2>&1
  rc=$?
  # quant_model_real's tail is `return ok ? 0 : 2;` (main.cpp:452) where `ok` is
  # the operator-fidelity verdict. rc=2 means REVIEW (that chunk's mean weight
  # recon >= 0.25) -- the artifact is still WRITTEN and VALID, so rc=2 must not
  # read as a failed chunk. A missing/short artifact is the real failure and the
  # existence check below is its authority.
  if [ $rc -ne 0 ] && [ $rc -ne 2 ]; then
    echo "[FAIL ] chunk $start..$end rc=$rc at $(date +%H:%M:%S) | $(tail -1 "$log")"
    return $rc
  fi
  if [ ! -f "$out_f" ] || [ "$(stat -c%s "$out_f" 2>/dev/null || echo 0)" -lt 1000000 ]; then
    echo "[FAIL ] chunk $start..$end rc=$rc wrote no usable artifact"
    return 1
  fi
  local dist verdict
  dist=$(grep -a -oE "scheme dist.*" "$log" | head -1)
  verdict=$(grep -a -oE "VERDICT.*" "$log" | head -1)
  echo "[done ] chunk $start..$end frac=$frac | $dist | $verdict | $(du -h "$out_f" | cut -f1)"
  return 0
}
export -f run_chunk
export BIN SRC OUT FRAC_LAYER FRAC_HEAD MAXMEM

JOBS_ARR=()
for ((start=0; start<TOTAL_LAYERS; start+=LAYER_PER_CHUNK)); do
  JOBS_ARR+=("$start $((start + LAYER_PER_CHUNK))")
done
printf '%s\n' "${JOBS_ARR[@]}" | xargs -n2 -P"$JOBS" bash -c 'run_chunk "$@"' _

echo "==== chunks done: $(ls "$OUT"/*.intifer 2>/dev/null | wc -l)/32 ===="
ls -la "$OUT"/*.intifer 2>/dev/null
