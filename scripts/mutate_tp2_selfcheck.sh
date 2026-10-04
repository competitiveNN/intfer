#!/usr/bin/env bash
# Mutation matrix for the DECODE1_MR arm's --self-check in scripts/test_tp2_27b.sh.
#
# WHY A MATRIX FOR A test_*.sh SELF-CHECK, WHEN run_ten_gates.sh HAS NONE (AUDIT #24):
#   run_ten_gates.sh's subject is the ENGINE and "does it classify correctly" is
#   answered by its own self-check. This is the same trade. The self-check exists
#   because the arm's real-run half needs 2 GPUs and an 18 GiB model load, which
#   is the check nobody re-runs -- so the stubbed half must be provably
#   load-bearing, and "provably" needs re-runnable evidence, not a transcript.
#
# WHAT IT PROTECTS, AND WHY IT HAD TO BE BUILT (AUDIT #31, measured):
#   The self-check as first committed defined its own `arm_ok` and called it "the
#   real arm's guard, verbatim". It was a COPY. MEASURED on that version: deleting
#   the real arm's stream comparison, its prefill oracle and its TP=sharded grep
#   each left --self-check at rc=0 PASS. Three of the arm's assertions were
#   UNOBSERVABLE to the check written to cover them -- audit #3's shape (a rule
#   fixed at one site and re-derived at another), committed in the code written to
#   avoid exactly that. Fixed by giving the arm ONE home (mr_arm_assert) that both
#   the real arm and the self-check call; these states are what hold that honest.
#
# RUN: scripts/mutate_tp2_selfcheck.sh            (all states, seconds, no GPU)
#      scripts/mutate_tp2_selfcheck.sh --list     (states + expectations)
#
# EXIT CODES
#   0  every state was caught (and the baseline was correct)
#   1  a state was MISSED, a baseline was wrong, or the pre-flight refused
#
# SCOPE, STATED NOT IMPLIED: this matrix proves the SELF-CHECK can fail. It does
#   NOT prove the arm's real-run half is correct -- that needs the box and the
#   model, and is reported UNRUN in AGENTS.md rather than claimed.
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="$SCRIPT_DIR/test_tp2_27b.sh"
BUILD="${INTFER_BUILD_DIR:-$ROOT/build}/tp2_selfcheck_mutate"
STAMP="$BUILD/last_pass.stamp"

# Dirty-tree guard: this script edits a TRACKED file. Same reason the other
# matrices have it -- mutate_fold_matrix.sh's `git checkout --` destroyed
# uncommitted work once (recorded in AGENTS.md).
if ! git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null; then
  echo "MUTATE: refusing to start on a dirty scripts/ tree -- this matrix edits" >&2
  echo "  a TRACKED file and restores it with git checkout." >&2
  exit 1
fi
[ -f "$TARGET" ] || { echo "MUTATE: no target: $TARGET" >&2; exit 1; }
mkdir -p "$BUILD"
rm -f "$STAMP"

SB="$(mktemp -d)"
# EXIT INT TERM, not EXIT only: an EXIT trap does not run when the shell is
# killed by an untrapped signal, and these matrices get wrapped in timeout.
# *** AND NO BACKSLASHES INSIDE THE SINGLE QUOTES. *** An earlier revision of
# another matrix in this repo shipped trap 'rm -rf \"$SB\"', which removes a
# filename that does not exist -- cleanup got WORSE than EXIT-only and bash -n
# stayed green, because it is syntactically valid and semantically wrong.
trap 'rm -rf "$SB"' EXIT INT TERM
SB_DIGEST_BEFORE="$(cat "$TARGET" | cksum)"

total=0; caught=0; bad=0; baselines=0
declare -a RESULTS

# patch_literal: exact-literal replacement, EXACTLY THREE ARGS, refuses unless
# the anchor occurs exactly once. A drifted anchor must be a hard stop, never a
# silent wrong-site patch. Shared with the other matrices via _gate_patch.sh.
. "$SCRIPT_DIR/_gate_patch.sh"

# --- pre-flight: every anchor must occur exactly once, BEFORE any state runs ---
# Runs in the first second. An anchor that has drifted should fail here, not
# after nine baselines have each executed a full mutant run (AUDIT #28).
preflight() {
  local n=0 f
  for f in "$SB"/anchor_*.txt; do
    : # files are created by anchor()
  done
  local anchors=(
    'MR_ARM_ASSERT_CALL'
    'PRED_STREAMS_FULL'
    'PRED_PREFILL'
    'PRED_TP'
    'ARM_RETURN'
  )
  local a
  for a in "${anchors[@]}"; do
    local f="$SB/anchor_$a.txt"
    [ -f "$f" ] || { echo "ANCHOR PREFLIGHT: $a has no file"; return 1; }
    local c
    c="$(grep -cF -- "$(cat "$f")" "$TARGET" 2>/dev/null || true)"
    c="${c:-0}"
    if [ "$c" -ne 1 ]; then
      echo "ANCHOR STALE in test_tp2_27b.sh: $a found $c occurrence(s), want 1." >&2
      echo "  first line:     $(head -1 "$f")" >&2
      return 1
    fi
  done
  echo "anchor pre-flight: ok (${#anchors[@]}/${#anchors[@]} targets located exactly once)"
  return 0
}

anchor() { printf '%s' "$2" >"$SB/anchor_$1.txt"; }

# --- baseline: the UNMUTATED self-check must PASS, and must say so ------------
baseline() {
  local out rc
  out="$("$TARGET" --self-check 2>&1)"; rc=$?
  baselines=$((baselines+1))
  if [ "$rc" -ne 0 ]; then
    echo "BASELINE FAIL: unmutated --self-check exited rc=$rc -- every state below" >&2
    echo "  would be scored against a broken baseline." >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  # The converse half: a healthy run must not print any FAIL line. Without this
  # a baseline could be "correct" for the wrong reason (e.g. a case that only
  # ever fires).
  case "$out" in
    *"SELF-CHECK FAIL"*) echo "BASELINE FAIL: healthy run printed a FAIL line" >&2; exit 1 ;;
  esac
  case "$out" in
    *"SELF-CHECK: PASS"*) ;;
    *) echo "BASELINE FAIL: healthy run did not print its PASS verdict" >&2; exit 1 ;;
  esac
  echo "  ok baseline rc=0, PASS verdict present, no FAIL line"
}

# run_state <id> <expected-regex> <mutation-fn>
# A state is CAUGHT when the mutant's --self-check exits NON-ZERO **and** prints
# the expected marker: rc alone is not enough (a state can fail for an unrelated
# reason, which is how a contaminated hand-run once scored a false catch).
# The mutant runs on a TEMP COPY, so the tracked file is never left mutated.
run_state() {
  local id="$1" expect="$2" fn="$3" m out rc
  m="$SB/$id.sh"
  cp "$TARGET" "$m"
  "$fn" "$m" || { record "$id" MISSED "(mutation fn failed)"; return; }
  if ! bash -n "$m" 2>/dev/null; then
    # A syntax error is NOT evidence a mechanism is load-bearing.
    record "$id" MISSED "(mutation broke syntax -- not evidence)"
    return
  fi
  out="$(bash "$m" --self-check 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$BUILD/$id.log"
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE -- "$expect"; then
    caught=$((caught+1)); record "$id" CAUGHT ""
  elif [ "$rc" -eq 0 ]; then
    record "$id" MISSED "(self-check still PASS -- unobservable)"
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    record "$id" MISSED "(caught for the WRONG reason; wanted /$expect/)"
  else
    record "$id" MISSED "(rc=$rc, marker absent)"
  fi
}

record() { total=$((total+1)); RESULTS+=("$1 $2 $3"); }

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  MR-1  arm's mr_arm_assert call deleted"
  echo "  MR-2  mr_streams_agree non-empty half deleted"
  echo "  MR-3  mr_streams_agree equality half deleted"
  echo "  MR-4  mr_streams_agree always accepts"
  echo "  MR-5  mr_prefill_ok always accepts"
  echo "  MR-6  mr_prefill_ok inverted"
  echo "  MR-7  mr_tp_sharded always accepts"
  echo "  MR-8  mr_tp_sharded pattern never matches"
  echo "  MR-9  mr_arm_assert always certifies"
  echo "  MR-10 cli_ids returns nothing"
  echo "  MR-11 cli_prefill returns nothing"
  exit 0
fi

echo "== anchor pre-flight =="
anchor MR_ARM_ASSERT_CALL '  mr_arm_assert "$CLI_ERR" "$MR_ERR" "GATE FAIL: " || PASS=0'
anchor PRED_STREAMS_FULL '  [ -n "$1" ] && [ -n "$2" ] && [ "$1" = "$2" ]'
anchor PRED_PREFILL '  [ "$1" = "11751" ]'
anchor PRED_TP '  grep -aq "TP=sharded" "$1"'
anchor ARM_RETURN '  return "$ok"'
preflight || exit 1

echo "== baseline =="
baseline

echo "== states =="

m_mr1() { patch_literal "$1" "$(cat "$SB/anchor_MR_ARM_ASSERT_CALL.txt")" \
  '  : #MR1 the arm never calls its own assertion function'; }
m_mr2() { patch_literal "$1" "$(cat "$SB/anchor_PRED_STREAMS_FULL.txt")" \
  '  [ "$1" = "$2" ] #MR2 non-empty half deleted'; }
m_mr3() { patch_literal "$1" "$(cat "$SB/anchor_PRED_STREAMS_FULL.txt")" \
  '  [ -n "$1" ] && [ -n "$2" ] #MR3 equality half deleted'; }
m_mr4() { patch_literal "$1" "$(cat "$SB/anchor_PRED_STREAMS_FULL.txt")" \
  '  true #MR4 always accepts'; }
m_mr5() { patch_literal "$1" "$(cat "$SB/anchor_PRED_PREFILL.txt")" \
  '  true #MR5 always accepts'; }
m_mr6() { patch_literal "$1" "$(cat "$SB/anchor_PREFILL.txt" 2>/dev/null || cat "$SB/anchor_PRED_PREFILL.txt")" \
  '  [ "$1" = "4242" ] #MR6 inverted'; }
m_mr7() { patch_literal "$1" "$(cat "$SB/anchor_PRED_TP.txt")" \
  '  true #MR7 always accepts'; }
m_mr8() { patch_literal "$1" "$(cat "$SB/anchor_PRED_TP.txt")" \
  '  grep -aq "TP=shardedX" "$1" #MR8 never matches'; }
m_mr9() { patch_literal "$1" "$(cat "$SB/anchor_ARM_RETURN.txt")" \
  '  return 0 #MR9 always certifies'; }
m_mr10() { patch_literal "$1" \
  'cli_ids() { # $1 = log -- every decode-step token id, prefill included
  grep -a -oE "decode step [0-9]+: token=[0-9]+" "$1" | sed -E '"'"'s/.*token=//'"'"'
}' 'cli_ids() { return 0; } #MR10'; }
m_mr11() { patch_literal "$1" \
  'cli_prefill() { # $1 = log -- the step-0 id alone
  grep -a -oE "decode step 0: token=[0-9]+" "$1" | sed -E '"'"'s/.*=//'"'"'
}' 'cli_prefill() { return 0; } #MR11'; }

run_state MR-1  'SELF-CHECK FAIL'   m_mr1
run_state MR-2  'SELF-CHECK FAIL \(4\)' m_mr2
run_state MR-3  'SELF-CHECK FAIL \(3\)' m_mr3
run_state MR-4  'SELF-CHECK FAIL \(3\)' m_mr4
run_state MR-5  'SELF-CHECK FAIL \(5\).*prefill' m_mr5
run_state MR-6  'SELF-CHECK FAIL \(5\).*prefill' m_mr6
run_state MR-7  'SELF-CHECK FAIL \(5\).*TP'     m_mr7
run_state MR-8  'SELF-CHECK FAIL \(5\).*TP'     m_mr8
run_state MR-9  'SELF-CHECK FAIL \(3\)' m_mr9
run_state MR-10 'SELF-CHECK FAIL \(1\)' m_mr10
run_state MR-11 'SELF-CHECK FAIL \(1\)|SELF-CHECK FAIL \(4\)|SELF-CHECK FAIL \(5\).*prefill' m_mr11

# --- residue: the tracked file must be byte-identical to HEAD ----------------
if [ "$SB_DIGEST_BEFORE" != "$(cat "$TARGET" | cksum)" ]; then
  echo "RESIDUE: $TARGET was modified by the matrix" >&2
  bad=$((bad+1))
else
  echo "residue: ok (target byte-identical to its pre-run state)"
fi

# --- verdict. Printed LAST, after the table, never before it -----------------
{
  echo
  for r in "${RESULTS[@]}"; do echo "  $r"; done
  echo
  echo "states=$total caught=$caught baselines=$baselines"
  if [ "$caught" -eq "$total" ] && [ "$baselines" -eq "$total" ] && [ "$bad" -eq 0 ]; then
    echo "RESULT: PASS ($caught/$total states caught, each for its own reason)"
  else
    echo "RESULT: FAIL ($caught/$total states caught)"
  fi
} | tee "$BUILD/summary.txt"

if [ "$caught" -eq "$total" ] && [ "$baselines" -eq "$total" ] && [ "$bad" -eq 0 ]; then
  { echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"
    echo "caught=$caught"
    echo "baselines=$baselines"
  } >"$STAMP.tmp" && mv "$STAMP.tmp" "$STAMP"
  exit 0
fi
exit 1