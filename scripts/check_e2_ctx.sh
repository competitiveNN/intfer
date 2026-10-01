#!/usr/bin/env bash
# Per-arm EVIDENCE of the context length actually processed, for every E2 arm.
# Not an inference from a command that died: each arm prints its own
# "prompt: N tokens" line, and the run must reach the decode steps.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="$ROOT/test/models/Qwen3.5-0.8B.intifer"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"

LONG="$(python3 -c "print('The history of artificial intelligence and machine learning. '*400)")"
echo "INTFER_KV_CAP=${INTFER_KV_CAP:-<unset, engine default 1024>}"
echo
for arm in BASE KV KVQ8; do
  case $arm in
    BASE) e="" ;;
    KV)   e="INTFER_INT8_KV=1" ;;
    KVQ8) e="INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1" ;;
  esac
  # shellcheck disable=SC2086
  out="$(env $e timeout 1800 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$LONG" -n 8 \
         --no-stream $TOK 2>&1)"
  rc=$?
  ptok="$(printf '%s\n' "$out" | grep -aoE 'prompt: [0-9]+ tokens' | head -1)"
  ids="$(printf '%s\n' "$out" | grep -ao 'decode step [0-9]*: token=[0-9]*' | grep -o '[0-9]*$' | tr '\n' ' ')"
  steps="$(printf '%s\n' "$out" | grep -ac 'decode step')"
  fatal="$(printf '%s\n' "$out" | grep -aoE 'FATAL: [^\r]*' | head -1)"
  printf '%-6s rc=%s  %-22s decode_steps=%s\n' "$arm" "$rc" "${ptok:-<no prompt line>}" "$steps"
  printf '       ids: %s\n' "${ids:-<none>}"
  [ -n "$fatal" ] && printf '       FATAL: %s\n' "$fatal"
done
