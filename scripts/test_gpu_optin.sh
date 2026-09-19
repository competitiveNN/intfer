#!/usr/bin/env bash
# GPU opt-in gate: KV-mirror and SSM_GPU paths are stream-identical to the
# base GPU path on the self-contained .intifer model.
#
# Rationale (2026-09-19): the CPU parity gate already asserts INT8-KV/Q8
# vs the HF oracle on --no-gpu, and AGENTS.md records SSM_GPU M=1 as -9%
# (default-off). What was unverified is the GPU combination: KV mirrors
# (INTFER_INT8_KV / INTFER_KV_Q8 / INTFER_KV_HAD) and device SSM
# (INTFER_SSM_GPU=1) under --gpu x TP=1/2. This gate certifies the baseline
# so the next quiet-box session can measure wall-time with a trusted setup.
#
# NOTE: this uses the self-contained .intifer model, whose greedy stream
# (279/6511/314/279) differs from the HF oracle by design — the 11751
# oracle lives in the legacy .intfer file (see memories/
# model_files_intfer_vs_intifer.md). This gate asserts cross-knob identity
# on the .intifer stream, not oracle equivalence.
# Usage: scripts/test_gpu_optin.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PROMPT="The capital of France is"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
  echo "SKIP: no render-node access from this shell."
  exit 2
fi
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

PASS=1
fail() { echo "GATE FAIL: $1" >&2; PASS=0; }
ids_of() { # $1=tp $2=env... -- prints "t0 t1 t2 t3" decode ids
  local tp="$1"; shift
  env "$@" "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n 4 \
    --no-stream --ids $TOK 2>&1 >/dev/null \
    | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
    | tr '\n' ' ' | sed 's/ *$//'
}

for tp in 1 2; do
  BASE="$(ids_of "$tp")"
  echo "TP=$tp base: '$BASE'"
  [ -n "$BASE" ] || fail "TP=$tp base empty (broken GPU run)"
  for mode in "INTFER_INT8_KV=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1" "INTFER_SSM_GPU=1"; do
    # shellcheck disable=SC2086
    GOT="$(env $mode "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n 4 \
      --no-stream --ids $TOK 2>&1 >/dev/null \
      | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
      | tr '\n' ' ' | sed 's/ *$//')"
    echo "TP=$tp [$mode]: '$GOT'"
    [ "$GOT" = "$BASE" ] || fail "TP=$tp [$mode] '$GOT' != base '$BASE'"
  done
done

if [ "$PASS" = 1 ]; then echo "GPU-OPTIN TEST: PASS"; else echo "GPU-OPTIN TEST: FAIL" >&2; exit 1; fi
