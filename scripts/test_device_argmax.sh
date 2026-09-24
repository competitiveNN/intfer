#!/usr/bin/env bash
# Device-argmax v1 gate (audit B lever (a), d460eb7): INTFER_DEVICE_ARGMAX=1
# must be stream-identical to the full-D2H path everywhere greedy rows flow.
#
# Bit-contract (proven on CPU, seam_audit_b.md §6 — harness since deleted):
#   (a) dev_half_to_float == sycl::half over ALL 65536 half patterns;
#   (b) kernel scan == host strict-> loop on adversarial tiles + 200k fuzz;
#   (c) multi-tile ts-order host reduce == flat loop incl. cross-tile ties.
# This script gates the DEVICE side on the fleet box (needs Arc L0).
#
# Sections (all load-insensitive stream-identity; wall-time NOT claimed):
#   1. TP1/TP2 8-way matrix (-n6 CLI decode): base==flag==HOSTPREP==SSM==
#      SSMxflag==FLAG+SSMxHOSTPREP==TP2-base==TP2-flag. [argmax] counter must
#      fire on FLAG runs (non-triviality: proves the fused path ran, not a
#      silent fallback) and stay silent on CPU-only sanity.
#   2. Mixed-temp CLI sampling smoke: -t 0.7 run completes identical token
#      COUNT base-vs-flag (values differ by RNG by design; count + finish
#      must match — sampling rows keep the full-D2H path).
#   3. Spec-verify end-to-end: INTFER_SPEC_K=4 greedy run identical
#      base-vs-flag. NOTE (code-read 2026-09-23): the verify rows in
#      prefill_layer_outer/score_row take the full-D2H path BY DESIGN
#      (StepOut carries full logits; the *_best hooks are never called
#      there), so the flag is a silent no-op on verify rows themselves.
#      This section only proves the end-to-end spec LOOP (whose accepted /
#      bonus tokens flow through scalar step() rounds that DO fire the
#      device path) is unperturbed — it is NOT a verify-row gate.
#   4. Standard gates: cpu parity + http batch + gpu-optin 10/10.
#
# Usage: scripts/test_device_argmax.sh
# Env: INTFER_08B (model), INTFER_BUILD_DIR (build dir).
# Exit 0 PASS, 1 FAIL, 2 SKIP (no render node).
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
ids_of() { # $1=tp $2...=env -- prints "t0 t1 t2 t3 t4 t5" decode ids
  local tp="$1"; shift
  # shellcheck disable=SC2086
  env "$@" "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n 6 \
    --no-stream --ids $TOK 2>&1 \
    | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
    | tr '\n' ' ' | sed 's/ *$//'
}
count_of() { # $1=tp $2...=env -- prints decode-token count (sampling: values vary, count must not)
  local tp="$1"; shift
  # shellcheck disable=SC2086
  env "$@" "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n 6 -t 0.7 -S 42 \
    --no-stream --ids $TOK 2>&1 \
    | grep -ac -E "decode step [0-9]+: token=[0-9]+"
}

echo "--- 1. TP1/TP2 8-way stream matrix (-n6 greedy) ---"
TP1_BASE="$(ids_of 1)"
echo "TP1 base: '$TP1_BASE'"
[ -n "$TP1_BASE" ] || fail "TP1 base empty (broken GPU run)"
TP1_FLAG="$(ids_of 1 INTFER_DEVICE_ARGMAX=1)"
echo "TP1 flag: '$TP1_FLAG'"
[ "$TP1_FLAG" = "$TP1_BASE" ] || fail "TP1 flag '$TP1_FLAG' != base '$TP1_BASE'"
for mode in "INTFER_L5_HOSTPREP=1" "INTFER_SSM_GPU=1" \
            "INTFER_DEVICE_ARGMAX=1 INTFER_SSM_GPU=1" \
            "INTFER_DEVICE_ARGMAX=1 INTFER_L5_HOSTPREP=1 INTFER_SSM_GPU=1"; do
  # shellcheck disable=SC2086
  GOT="$(ids_of 1 $mode)"
  echo "TP1 [$mode]: '$GOT'"
  [ "$GOT" = "$TP1_BASE" ] || fail "TP1 [$mode] '$GOT' != base '$TP1_BASE'"
done
TP2_BASE="$(ids_of 2)"
echo "TP2 base: '$TP2_BASE'"
[ -n "$TP2_BASE" ] || fail "TP2 base empty (broken GPU run)"
TP2_FLAG="$(ids_of 2 INTFER_DEVICE_ARGMAX=1)"
echo "TP2 flag: '$TP2_FLAG'"
[ "$TP2_FLAG" = "$TP2_BASE" ] || fail "TP2 flag '$TP2_FLAG' != base '$TP2_BASE'"

echo "--- 1b. non-triviality: [argmax] counter fires on GPU flag runs ---"
DBG_N="$(INTFER_DEVICE_ARGMAX=1 INTFER_ARGMAX_DBG=1 "$BIN" --gpu --tp 1 \
  -m "$MODEL" -p "$PROMPT" -n 6 --no-stream --ids $TOK 2>&1 >/dev/null \
  | grep -ac "\[argmax\]")"
echo "TP1 flag [argmax] lines: $DBG_N (expect >=1)"
[ "$DBG_N" -ge 1 ] || fail "[argmax] counter silent on GPU flag run (fallback? path never fired?)"
DBG_OFF="$(INTFER_ARGMAX_DBG=1 "$BIN" --gpu --tp 1 \
  -m "$MODEL" -p "$PROMPT" -n 6 --no-stream --ids $TOK 2>&1 >/dev/null \
  | grep -ac "\[argmax\]")"
echo "TP1 base [argmax] lines: $DBG_OFF (expect 0)"
[ "$DBG_OFF" = 0 ] || fail "[argmax] counter fired without the flag ($DBG_OFF lines)"

echo "--- 2. mixed-temp sampling smoke (count parity, -t 0.7 -S 42) ---"
S_BASE="$(count_of 1)"
S_FLAG="$(count_of 1 INTFER_DEVICE_ARGMAX=1)"
echo "TP1 sampling count base=$S_BASE flag=$S_FLAG"
[ -n "$S_BASE" ] && [ "$S_BASE" = "$S_FLAG" ] || \
  fail "sampling count base=$S_BASE flag=$S_FLAG (rows must still complete)"

echo "--- 3. spec-verify end-to-end (INTFER_SPEC_K=4 greedy) ---"
SPEC_BASE="$(ids_of 1 INTFER_SPEC_K=4)"
SPEC_FLAG="$(ids_of 1 INTFER_DEVICE_ARGMAX=1 INTFER_SPEC_K=4)"
echo "TP1 spec base: '$SPEC_BASE'"
echo "TP1 spec flag: '$SPEC_FLAG'"
[ -n "$SPEC_BASE" ] || fail "TP1 spec base empty"
[ "$SPEC_FLAG" = "$SPEC_BASE" ] || fail "TP1 spec flag != base"
SPEC_DBG="$(INTFER_DEVICE_ARGMAX=1 INTFER_ARGMAX_DBG=1 INTFER_SPEC_K=4 "$BIN" \
  --gpu --tp 1 -m "$MODEL" -p "$PROMPT" -n 6 --no-stream --ids $TOK 2>&1 >/dev/null \
  | grep -ac "\[argmax\]")"
echo "TP1 spec flag [argmax] lines: $SPEC_DBG (expect >=1: scalar rounds fire)"
[ "$SPEC_DBG" -ge 1 ] || fail "[argmax] silent on spec run (scalar rounds fell back?)"

echo "--- 3b. spec-verify ROWS ride the hook (repetitive prompt, K=4) ---"
# N-gram drafts need repeated tokens: the repeated 8-word prompt guarantees
# verify rounds fire (step jumps >1). Verify consumers read .best only, so
# with the flag every verify row takes the scalar hook (all_greedy=true):
# base==flag streams must hold AND [argmax-part] lines must grow by exactly
# 4/verify-row (TP1 0.8B = 4 lm_head tiles) vs the no-spec flag run.
SPEC_PROMPT="one two three four five six seven eight one two three four five six seven eight"
spec_steps() { # $1...=env -- prints decode-step lines (round starts, not tokens)
  # shellcheck disable=SC2086
  env "$@" "$BIN" --gpu --tp 1 -m "$MODEL" -p "$SPEC_PROMPT" -n 8 \
    --no-stream --ids $TOK 2>&1 \
    | grep -a -E "decode step [0-9]+: token=[0-9]+" | tr '\n' '|'
}
spec_parts() { # $1...=env -- prints [argmax-part] line count
  # shellcheck disable=SC2086
  env INTFER_ARGMAX_PARTS=1 "$@" "$BIN" --gpu --tp 1 -m "$MODEL" -p "$SPEC_PROMPT" -n 8 \
    --no-stream --ids $TOK 2>&1 >/dev/null | grep -ac "argmax-part"
}
SP_BASE="$(spec_steps INTFER_SPEC_K=4)"
SP_FLAG="$(spec_steps INTFER_DEVICE_ARGMAX=1 INTFER_SPEC_K=4)"
echo "spec verify-round starts base: '$SP_BASE'"
echo "spec verify-round starts flag: '$SP_FLAG'"
[ -n "$SP_BASE" ] || fail "spec repetitive base empty"
[ "$SP_FLAG" = "$SP_BASE" ] || fail "spec repetitive flag != base"
P_NOSPEC="$(spec_parts INTFER_DEVICE_ARGMAX=1)"
P_SPEC="$(spec_parts INTFER_DEVICE_ARGMAX=1 INTFER_SPEC_K=4)"
P_SPEC2="$(spec_parts INTFER_DEVICE_ARGMAX=1 INTFER_SPEC_K=4)"
echo "PARTS no-spec=$P_NOSPEC spec=$P_SPEC spec-rerun=$P_SPEC2"
[ "$P_SPEC" -gt "$P_NOSPEC" ] || \
  fail "verify rows did not ride the hook (spec PARTS $P_SPEC <= no-spec $P_NOSPEC)"
[ "$P_SPEC2" = "$P_SPEC" ] || \
  fail "verify PARTS not deterministic ($P_SPEC vs rerun $P_SPEC2)"
P_TP2FLAG="$(env INTFER_DEVICE_ARGMAX=1 INTFER_ARGMAX_PARTS=1 INTFER_SPEC_K=4 "$BIN" \
  --gpu --tp 2 -m "$MODEL" -p "$SPEC_PROMPT" -n 8 --no-stream --ids $TOK 2>&1 >/dev/null \
  | grep -ac "argmax-part")"
echo "PARTS TP2 spec flag: $P_TP2FLAG (expect 0: TP1-only guard)"
[ "$P_TP2FLAG" = 0 ] || fail "hook fired on TP2 ($P_TP2FLAG lines)"
SP2_BASE="$(env INTFER_SPEC_K=4 "$BIN" --gpu --tp 2 -m "$MODEL" -p "$SPEC_PROMPT" \
  -n 8 --no-stream --ids $TOK 2>&1 | grep -a -E "decode step [0-9]+: token=[0-9]+" | tr '\n' '|')"
SP2_FLAG="$(env INTFER_DEVICE_ARGMAX=1 INTFER_SPEC_K=4 "$BIN" --gpu --tp 2 -m "$MODEL" \
  -p "$SPEC_PROMPT" -n 8 --no-stream --ids $TOK 2>&1 | grep -a -E "decode step [0-9]+: token=[0-9]+" | tr '\n' '|')"
[ "$SP2_FLAG" = "$SP2_BASE" ] || fail "TP2 spec repetitive flag != base"

echo "--- 4. standard gates ---"
if "$SCRIPT_DIR/test_cpu_parity.sh" >/dev/null 2>&1; then echo "cpu parity: PASS"; else fail "cpu parity"; fi
if "$SCRIPT_DIR/test_http_batch.sh" >/dev/null 2>&1; then echo "http batch: PASS"; else fail "http batch"; fi
if "$SCRIPT_DIR/test_gpu_optin.sh" >/dev/null 2>&1; then echo "gpu-optin: PASS"; else fail "gpu-optin"; fi

if [ "$PASS" = 1 ]; then echo "DEVICE-ARGMAX TEST: PASS"; else echo "DEVICE-ARGMAX TEST: FAIL" >&2; exit 1; fi
