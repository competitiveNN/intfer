#!/usr/bin/env bash
# CPU parity gate: 0.8B INT2 model on the CPU backend (--no-gpu).
#
# Encodes the post-3eb4e34 invariants (exact per-slot rot_for fallback):
#   1. HF-oracle agreement: prefill top-1 for "The capital of France is"
#      is 11751, and the decode prefix is 13 198 760 (matches the stored
#      HF oracle test/models/qwen35_oracle.npz generated[:4]).
#   2. Cross-path identity: M=1 step loop == chunked prefill (M=1/2/8) ==
#      serve split == serve joint at every chunk width. The serve
#      split-vs-joint comparison is the regression test for the lifted
#      joint P>1 cap (pre-fix, wide joint chunks diverged: 271 vs 561).
#
# Usage: scripts/test_cpu_parity.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intfer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PROMPT="The capital of France is"
PROMPT2="The quick brown fox jumps over the lazy dog."
PROMPT3="Hi"

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}"
PASS=1
ROTLOG="$(mktemp --tmpdir="$TMPDIR" rotcheck.XXXX.log)"
fail() { echo "GATE FAIL: $1" >&2; PASS=0; }

# --- 0. Rotation invariant: cold==warm for every layer x slot ---
# Unit check behind the 3eb4e34 fix (old first-hit fallback violated it
# for LA/o/down slots). Runs inside the CLI at startup; exits nonzero
# (rc=2) with [rotcheck] MISMATCH lines on violation.
echo "--- rotcheck ---"
INTFER_CHECK_ROT=1 "$BIN" --no-gpu -m "$MODEL" -p "Hi" -n 1 --no-stream \
  --tokenizer-py "$SCRIPT_DIR/bpe_tokenize.py" \
  --detok-py "$SCRIPT_DIR/bpe_detokenize.py" >/dev/null 2>"$ROTLOG"; RC=$?
grep -a "rotcheck" "$ROTLOG"
[ "$RC" = "0" ] || fail "rotcheck exited rc=$RC (see $ROTLOG)"
grep -aq "rotcheck. OK" "$ROTLOG" || fail "rotcheck did not report OK"
rm -f "$ROTLOG"

# --- 1. M=1 step loop: prefill top-1 + decode prefix vs HF oracle ---
# NOTE: -n counts the prefill token, so -n 4 yields 3 decode steps.
OUT="$("$BIN" --no-gpu -m "$MODEL" -p "$PROMPT" -n 4 --no-stream --ids \
  --tokenizer-py "$SCRIPT_DIR/bpe_tokenize.py" \
  --detok-py "$SCRIPT_DIR/bpe_detokenize.py" 2>&1 >/dev/null \
  | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//')"
PREFILL="$(echo "$OUT" | head -1)"
DECODE="$(echo "$OUT" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
echo "M=1: prefill=$PREFILL (expect 11751) decode='$DECODE' (expect '13 198 760')"
[ "$PREFILL" = "11751" ] || fail "M=1 prefill $PREFILL != 11751"
[ "$DECODE" = "13 198 760" ] || fail "M=1 decode '$DECODE' != '13 198 760'"

# --- 2. Chunked prefill agrees with M=1 at every width ---
for m in 1 2 8; do
  T="$(INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M="$m" \
    "$BIN" --no-gpu -m "$MODEL" -p "$PROMPT" -n 1 --no-stream --ids \
    --tokenizer-py "$SCRIPT_DIR/bpe_tokenize.py" \
    --detok-py "$SCRIPT_DIR/bpe_detokenize.py" 2>&1 >/dev/null \
    | grep -a -oE "decode step 0: token=[0-9]+" | sed -E 's/.*token=//')"
  echo "chunked M=$m: prefill=$T"
  [ "$T" = "11751" ] || fail "chunked M=$m prefill $T != 11751"
done

# --- 3. Serve: split == joint at every chunk width (P>1 cap-lift gate) ---
serve_run() { # $1=chunk_M $2=extra env assignments
  printf '%s\n%s\n%s\n' \
    "{\"prompt\":\"$PROMPT\",\"max_new\":2}" \
    "{\"prompt\":\"$PROMPT2\",\"max_new\":2}" \
    "{\"prompt\":\"$PROMPT3\",\"max_new\":2}" \
  | env INTFER_PREFILL_CHUNK_M="$1" $2 \
    "$BIN" --no-gpu -m "$MODEL" --serve \
    --tokenizer-py "$SCRIPT_DIR/bpe_tokenize.py" \
    --detok-py "$SCRIPT_DIR/bpe_detokenize.py" 2>/dev/null | sort | md5sum | cut -d' ' -f1
}
for m in 1 2 8; do
  S="$(serve_run "$m" "INTFER_SPLIT_ITER=1")"
  J="$(serve_run "$m" "")"
  echo "serve chunk_M=$m: split=$S joint=$J"
  [ "$S" = "$J" ] || fail "serve chunk_M=$m split != joint"
done

# --- 4. Serve streams carry the oracle prefill continuation ---
GOT="$(printf '%s\n' "{\"prompt\":\"$PROMPT\",\"max_new\":3}" \
  | "$BIN" --no-gpu -m "$MODEL" --serve \
    --tokenizer-py "$SCRIPT_DIR/bpe_tokenize.py" \
    --detok-py "$SCRIPT_DIR/bpe_detokenize.py" 2>/dev/null \
  | grep -a -oE '"token":[0-9]+' | sed -E 's/.*://' | tr '\n' ' ' | sed 's/ *$//')"
echo "serve decode: '$GOT' (expect '13 198 760')"
[ "$GOT" = "13 198 760" ] || fail "serve decode '$GOT' != '13 198 760'"

# --- 5. Groupwise-INT8 KV cache (INTFER_INT8_KV=1) is oracle-identical ---
# The INT8 mirror (per kv_head x 32-token group FP32 scale, kernel-compatible
# layout) must not change a single token: same prefill + decode as FP32.
IOUT="$(INTFER_INT8_KV=1 "$BIN" --no-gpu -m "$MODEL" -p "$PROMPT" -n 4 --no-stream --ids \
  --tokenizer-py "$SCRIPT_DIR/bpe_tokenize.py" \
  --detok-py "$SCRIPT_DIR/bpe_detokenize.py" 2>&1 >/dev/null \
  | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//')"
IPREFILL="$(echo "$IOUT" | head -1)"
IDECODE="$(echo "$IOUT" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
echo "INT8-KV: prefill=$IPREFILL (expect 11751) decode='$IDECODE' (expect '13 198 760')"
[ "$IPREFILL" = "11751" ] || fail "INT8-KV prefill $IPREFILL != 11751"
[ "$IDECODE" = "13 198 760" ] || fail "INT8-KV decode '$IDECODE' != '13 198 760'"

if [ "$PASS" = 1 ]; then echo "CPU-PARITY TEST: PASS"; else echo "CPU-PARITY TEST: FAIL" >&2; exit 1; fi
