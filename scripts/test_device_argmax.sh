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
#
# BUILD-FIRST (audit #7): gate_require_build in scripts/_gate_build.sh. Audit
# #6 fixed four gates against the stale-artifact class; this one consumed the
# intfer_cli binary without ever building it, so a source edit that failed to
# compile left the previous binary in place and the gate judged code that no
# longer existed. Opt out with INTFER_GATE_NO_BUILD=1 only when the build is
# known current.
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

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build intfer_cli || exit 1

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
  echo "SKIP: no render-node access from this shell."
  exit 2
fi
# 2-GPU precondition, PER ARM (AUDIT #13). Engine::load clamps
# min(tp_devices, dm.size()), so on a 1-GPU box every `--tp 2` arm silently runs
# as TP=1 and asserts against a path it never exercised.
#
# This gate is MOSTLY 1-GPU, though: sections 1, 1b, 2, 3, 4 and 5 are TP=1 (and
# 4's sub-gates are 1-GPU). Only five arms need a second device. So the guard is
# applied to those five arms, NOT to the script.
#
# That distinction is the whole point. An earlier revision put a top-level
# `gate_require_gpus 2 || exit 2` here, which on a 1-GPU box skipped the ENTIRE
# gate -- silently dropping the serve-fence regression test (section 5), which
# exists precisely to catch a crash that "only manifests on a LATER sampling
# request" and so is invisible to every stream gate. A gate that always skips is
# indistinguishable from a gate that is not run, and the path it protects is a
# known-dangerous one. Restoring per-arm granularity is the fix; exit 2 still
# reports the lost coverage at the end (see the verdict block).
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
. "$SCRIPT_DIR/_gate_precond.sh"
N_GPUS="$(gate_gpu_count || echo 0)"
HAVE_TP2=0; [ "${N_GPUS:-0}" -ge 2 ] && HAVE_TP2=1
SKIPPED_TP2=0
if [ "$HAVE_TP2" = 1 ]; then
  echo "precondition: $N_GPUS GPU(s) visible (need 2 for the TP=2 arms)" >&2
else
  echo "SKIPPING the 5 TP=2 arms: need 2 GPUs, ${N_GPUS:-0} visible." >&2
  echo "  BOX STATE, not a code failure. The TP=1 arms below still run and still assert." >&2
  SKIPPED_TP2=1
fi

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
if [ "$HAVE_TP2" = 1 ]; then
  TP2_BASE="$(ids_of 2)"
  echo "TP2 base: '$TP2_BASE'"
  [ -n "$TP2_BASE" ] || fail "TP2 base empty (broken GPU run)"
  TP2_FLAG="$(ids_of 2 INTFER_DEVICE_ARGMAX=1)"
  echo "TP2 flag: '$TP2_FLAG'"
  [ "$TP2_FLAG" = "$TP2_BASE" ] || fail "TP2 flag '$TP2_FLAG' != base '$TP2_BASE'"
fi

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
flag_parts() { # $1...=env -- PARTS lines on the STANDARD prompt (-n6 decode)
  # shellcheck disable=SC2086
  env INTFER_ARGMAX_PARTS=1 "$@" "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" -n 6 \
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
# Exact-count lock (TP1 0.8B = 4 lm_head tiles; -n8 + one 2-row verify round):
# any silent fallback (fence erosion, first-use-probe drift, hook decline)
# changes these numbers. If the prompt/model changes, re-baseline here.
[ "$P_NOSPEC" = 32 ] || fail "PARTS no-spec=$P_NOSPEC != 32 (hook coverage changed?)"
[ "$P_SPEC" = 40 ] || fail "PARTS spec=$P_SPEC != 40 (verify-row coverage changed?)"
[ "$P_SPEC" -gt "$P_NOSPEC" ] || \
  fail "verify rows did not ride the hook (spec PARTS $P_SPEC <= no-spec $P_NOSPEC)"
[ "$P_SPEC2" = "$P_SPEC" ] || \
  fail "verify PARTS not deterministic ($P_SPEC vs rerun $P_SPEC2)"
if [ "$HAVE_TP2" = 1 ]; then
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
else
  echo "  (skipped 3 TP=2 spec/PARTS arms -- need 2 GPUs)"
fi

echo "--- 4. standard gates ---"
# Sub-gate rc map: 0 = PASS, 2 = PARTIAL/SKIP (box state, NOT a failure),
# anything else = FAIL. Collapsing 2 into "not 0" would turn an honest skip
# report from gpu-optin (which now returns 2 on a 1-GPU box) into a red line
# here -- the same false-FAIL class AUDIT #13 removed one level up.
sub_gate() { # $1 = label, rest = script
  local label="$1"; shift
  "$@" >/dev/null 2>&1
  local rc=$?
  case "$rc" in
    0) echo "$label: PASS" ;;
    2) echo "$label: PARTIAL/SKIP (rc=2, box state -- not counted as a failure)"
       SKIPPED_SUB=$((SKIPPED_SUB + 1)) ;;
    *) fail "$label (rc=$rc)"; echo "$label: FAIL (rc=$rc)" >&2 ;;
  esac
}
SKIPPED_SUB=0
sub_gate "cpu parity" "$SCRIPT_DIR/test_cpu_parity.sh"
sub_gate "http batch" "$SCRIPT_DIR/test_http_batch.sh"
sub_gate "gpu-optin" "$SCRIPT_DIR/test_gpu_optin.sh"

echo "--- 5. serve fence lock (flag ON server: no prefill-chunk hook firing) ---"
# The decode_only fence (batch hook declines prefill chunks — serve-poison
# fix) erodes silently if a refactor re-fires the hook on prefill: the crash
# only manifests on a LATER sampling request, so stream gates can't catch it.
# This section runs the concurrent mixed-temp serve load WITH the flag and
# asserts (a) no FATAL/crash, (b) greedy parity vs the known-good text.
# Load-insensitive: finish_reason + text equality only, no timing claims.
SERVE_PORT="${INTFER_ARGMAX_SERVE_PORT:-18321}"
SERVE_LOG="${TMPDIR:-/tmp}/argmax_serve_fence.log"
SERVE_PID=""
serve_cleanup() {
  # NOTE: the server may ignore SIGTERM while blocked in GPU calls; fall
  # back to KILL so the gate can't hang in `wait` (bare `wait` below would
  # also catch the session-detached server child — always wait on PIDs).
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
  [ -n "$SERVE_PID" ] && sleep 3 && kill -9 "$SERVE_PID" 2>/dev/null
  [ -n "$SERVE_PID" ] && wait "$SERVE_PID" 2>/dev/null
  SERVE_PID=""
}
trap serve_cleanup EXIT
setsid nohup env INTFER_DEVICE_ARGMAX=1 "$BIN" -m "$MODEL" --gpu --tp 1 \
  --serve-http --port "$SERVE_PORT" --host 127.0.0.1 >"$SERVE_LOG" 2>&1 < /dev/null &
SERVE_PID=$!
SERVE_UP=0
for _ in $(seq 1 40); do
  sleep 5
  curl -s -m 3 "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1 && { SERVE_UP=1; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || break
done
if [ "$SERVE_UP" != 1 ]; then
  fail "flag-ON server never came up (see $SERVE_LOG)"
else
  # Sequential greedy-then-sampling (the original 8/8 crash repro shape),
  # then a concurrent mixed batch (greedy + sampling in flight together).
  GREEDY_OUT="$(curl -s -m 120 -X POST "http://127.0.0.1:$SERVE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8,"temperature":0}')"
  echo "$GREEDY_OUT" | grep -q '"finish_reason"' || fail "greedy serve request failed: $GREEDY_OUT"
  SAMPLING_OUT="$(curl -s -m 120 -X POST "http://127.0.0.1:$SERVE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8,"temperature":0.7}')"
  echo "$SAMPLING_OUT" | grep -q '"finish_reason"' || fail "post-greedy sampling request failed (fence erosion?): $SAMPLING_OUT"
  MIXED_OK=1
  CURLPIDS=""
  for _ in 1 2 3; do
    curl -s -m 120 -X POST "http://127.0.0.1:$SERVE_PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8,"temperature":0}' >/dev/null 2>&1 &
    CURLPIDS="$CURLPIDS $!"
  done
  curl -s -m 120 -X POST "http://127.0.0.1:$SERVE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a short poem about the sea"}],"max_tokens":8,"temperature":0.7}' \
    | grep -q '"finish_reason"' || MIXED_OK=0
  # shellcheck disable=SC2086
  wait $CURLPIDS # PID-scoped: bare `wait` would also catch the detached
                 # server child and hang until it exits (gate stuck 11h).
  [ "$MIXED_OK" = 1 ] || fail "concurrent mixed serve batch failed under flag"
  if kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "serve fence: server alive after greedy+sampling+mixed (no SIGSEGV)"
  else
    fail "flag-ON server DIED during mixed serve load (poison fence broken?)"
  fi
  grep -aiE "FATAL|SIGSEGV|segment" "$SERVE_LOG" | grep -av "no FATAL" | head -3 | \
    while read -r line; do fail "server log fatal: $line"; done
fi
serve_cleanup
trap - EXIT

if [ "$PASS" != 1 ]; then
  echo "DEVICE-ARGMAX TEST: FAIL" >&2
  exit 1
fi
if [ "$SKIPPED_TP2" = 1 ] || [ "${SKIPPED_SUB:-0}" -gt 0 ]; then
  # Everything runnable here passed, but some coverage never executed (TP=2
  # arms and/or a sub-gate that returned 2). Exit 2, never 0: a silent skip
  # reads as coverage we do not have.
  echo "DEVICE-ARGMAX TEST: PARTIAL PASS (TP=1 arms all green; skipped: TP2 arms=$SKIPPED_TP2, sub-gates=$SKIPPED_SUB -- need 2 GPUs)" >&2
  exit 2
fi
echo "DEVICE-ARGMAX TEST: PASS"
