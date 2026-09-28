#!/usr/bin/env bash
# Interleaved base/TAB microbench A/B on 27B INT2 down-proj (load-immune kernel_us).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/build/examples/gemm_bench/gemm_bench"
MODEL="/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer"
TENSOR="layers.0.mlp.down_proj"
REPS="${1:-6}"
ROUNDS="${2:-3}"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="BASE TAB"; else order="TAB BASE"; fi
  for mode in $order; do
    echo "===== round $r mode=$mode ====="
    if [ "$mode" = TAB ]; then export INTFER_INT2_TAB=1; else unset INTFER_INT2_TAB; fi
    "$BIN" "$MODEL" "$TENSOR" "$REPS" 2>&1 | grep -aE 'tensor|steady/call|steady2/call|y\[0'
  done
done
unset INTFER_INT2_TAB
