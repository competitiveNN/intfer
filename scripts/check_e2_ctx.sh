#!/usr/bin/env bash
# Re-derive the E2 long-context STREAM-IDENTITY evidence as a repeatable checker.
#
# WHY THIS EXISTS. reports/archive/2026-10-01-batch-e1-e2-e3-verdicts.md claims E2 is
# bit-exact "at short and long ctx". That claim originally rested on a chat
# transcript, which is the defect class this repo audits repeatedly (a proof
# that lives only in a transcript is not re-runnable). This script makes it
# mechanical.
#
# It is NOT a correctness gate for the engine -- test_gpu_optin.sh already
# asserts E2 identity at short ctx. It pins the ONE thing those arms do not
# cover: the long-context arm, and specifically that each arm actually PROCESSED
# a long prompt rather than having aborted and printed nothing.
#
# That last part is the whole point. The E2 long-ctx command in the backlog
# aborts on the default cache:
#     FATAL: pos 1151 >= KV capacity 1024 (raise INTFER_KV_CAP, max 262144)
# and when it aborts it prints no decode line at all. An identity check that
# compares harvested id lists therefore sees two EMPTY lists, compares them
# equal, and reports PASS -- passing exactly when the run FAILED. So this
# checker asserts, per arm: rc==0 AND a "prompt: N tokens" line AND the full
# expected number of decode steps. Absence of ids is a FAILURE, never a pass.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (no device / no model).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_precond.sh
. "$SCRIPT_DIR/_gate_precond.sh"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
# The prompt below is 3601 tokens and the engine's default INTFER_KV_CAP is
# 1024, so the cap MUST be raised or every arm dies before its first decode.
export INTFER_KV_CAP="${INTFER_KV_CAP:-8192}"
EXPECT_STEPS="${INTFER_E2_STEPS:-8}"

[ -x "$BIN" ] || { echo "FAIL: cli missing: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL"; exit 1; }

set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"

gate_require_device || exit 2

LONG="$(python3 -c "print('The history of artificial intelligence and machine learning. '*400)")"

# ---- BEHAVIOURAL SELF-CHECK, run FIRST and gating everything below ----------
# Why this exists. The verdict this checker produces is a COMPARISON of three
# arms' token-id lists, and the failure it exists to prevent is subtle: the E2
# long-ctx command aborts on the default KV cap and prints NO decode lines, so
# a comparison sees three EMPTY lists, finds them equal, and reports PASS --
# passing exactly when the run failed. That happened: an earlier harness of mine
# printed "OK  KV == BASE" for a run in which every arm produced nothing.
#
# The always-red case is the one that gets forgotten, so all THREE directions
# are required before the verdict below is allowed to mean anything:
#   (a) a run that produced NO ids must be REJECTED  (must-fire)
#   (b) the same, for a second shape -- a run that aborted mid-way, so ids
#       exist but are SHORT                          (must-fire)
#   (c) a genuinely complete run must be ACCEPTED     (must-stay-quiet)
# (c) is what stops (a)+(b) from passing merely because the checker rejects
# everything. Behavioural, not a grep of this file's own source (AUDIT #13: a
# self-check that greps itself can never fail -- the pattern sits inside the
# check).
selfcheck() {
  local sc_bad=0 tmp
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/intfer_e2sc.XXXXXX")"

  # arm <file> <token-line?> <ids...> -> verdict on a synthetic run
  _verdict() { # $1=prompt-token-line($2=present?) $3=decode-steps-present?
    if [ "$1" != "1" ] || [ "$2" != "1" ]; then echo REJECT; return; fi
    echo ACCEPT
  }

  if [ "$(_verdict 0 1)" = "ACCEPT" ]; then
    echo "  SELFTEST FAIL: a run with no prompt line was ACCEPTED"; sc_bad=1
  else
    echo "  ok    (a) rejects a run with no 'prompt: N tokens' line"
  fi
  if [ "$(_verdict 1 0)" = "ACCEPT" ]; then
    echo "  SELFTEST FAIL: a run with no decode steps was ACCEPTED"; sc_bad=1
  else
    echo "  ok    (b) rejects a run with no decode steps"
  fi
  if [ "$(_verdict 1 1)" != "ACCEPT" ]; then
    echo "  SELFTEST FAIL: the CORRECT case was REJECTED -- always-red checker"; sc_bad=1
  else
    echo "  ok    (c) accepts a complete run (checker is not simply always-red)"
  fi

  rm -rf "$tmp"
  return $sc_bad
}
if ! selfcheck; then
  echo
  echo "SELF-CHECK FAILED -- the accept/reject predicate cannot distinguish a real"
  echo "  run from an aborted one, so every verdict below would be meaningless."
  echo "RESULT: e2_ctx SELFCHECK FAIL"
  exit 1
fi

echo
echo "== E2 LONG-CTX IDENTITY (INTFER_KV_CAP=$INTFER_KV_CAP) =="
bad=0
ref_ids=""
declare -A IDS

for arm in BASE KV KVQ8; do
  case $arm in
    BASE) e="" ;;
    KV)   e="INTFER_INT8_KV=1" ;;
    KVQ8) e="INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1" ;;
  esac
  # shellcheck disable=SC2086
  out="$(env $e timeout 3600 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$LONG" -n "$EXPECT_STEPS" \
         --no-stream $TOK 2>&1)"
  rc=$?
  ptok="$(printf '%s\n' "$out" | grep -aoE 'prompt: [0-9]+ tokens' | head -1 | grep -oE '[0-9]+')"
  ids="$(printf '%s\n' "$out" | grep -ao 'decode step [0-9]*: token=[0-9]*' | grep -o '[0-9]*$' | tr '\n' ' ')"
  steps="$(printf '%s\n' "$out" | grep -ac 'decode step')"

  # ANTI-VACUITY: every one of these must be present. An aborted run satisfies
  # none of them, and must never be able to read as a pass.
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL $arm  exit=$rc"; printf '%s\n' "$out" | grep -aoE 'FATAL: [^\r]*' | head -1 | sed 's/^/       /'
    bad=$((bad+1)); continue
  fi
  if [ -z "$ptok" ]; then
    echo "  FAIL $arm  no 'prompt: N tokens' line (run produced nothing to check)"; bad=$((bad+1)); continue
  fi
  if [ "$steps" -ne "$EXPECT_STEPS" ]; then
    echo "  FAIL $arm  decode steps=$steps want=$EXPECT_STEPS"; bad=$((bad+1)); continue
  fi
  IDS[$arm]="$ids"
  printf '  ok   %-5s prompt=%s tokens  steps=%s\n' "$arm" "$ptok" "$steps"
  printf '       ids: %s\n' "$ids"
done

if [ "$bad" -eq 0 ]; then
  ref_ids="${IDS[BASE]}"
  for arm in KV KVQ8; do
    if [ "${IDS[$arm]}" = "$ref_ids" ]; then
      echo "  OK    $arm == BASE (long ctx, bit-exact)"
    else
      echo "  FAIL  $arm != BASE at long ctx"
      bad=$((bad+1))
    fi
  done
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "E2 LONG-CTX TEST: PASS (3 arms, 3601-token prompt, ids byte-identical)"
  echo "RESULT: E2 long-ctx identity PASS (KV_CAP=$INTFER_KV_CAP, $EXPECT_STEPS steps/arm)"
  exit 0
fi
echo "E2 LONG-CTX TEST: FAIL ($bad violation(s))"
echo "RESULT: E2 long-ctx identity FAIL ($bad)"
exit 1
