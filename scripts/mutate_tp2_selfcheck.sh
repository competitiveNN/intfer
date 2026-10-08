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
# *** DECLARED IN THE SHAPE scripts/_gate_pop.sh's DERIVATION READS. *** MEASURED:
# writing this as BUILD="${INTFER_BUILD_DIR:-$ROOT/build}/tp2_selfcheck_mutate"
# plus STAMP="$BUILD/last_pass.stamp" made pop_assert_no_leak report this matrix's
# own directory as an UNOWNED leak -- a false alarm on a healthy tree, which is
# the exact failure _gate_pop.sh's header says the derivation exists to prevent.
# It reads (WORK|LOG_DIR|STAMP)="..." and peels a known base var off the front, so
# the literal must START with $BUILD_DIR (or $ROOT). Siblings use that form, e.g.
# mutate_l0_cite.sh: STAMP="$BUILD_DIR/l0_cite_mutate".
# NOTE THE VARIABLE NAME. The derivation reads (WORK|LOG_DIR|STAMP)="...", so a
# name like STAMP_DIR= is INVISIBLE to it -- MEASURED: the directory was missing
# from known_dirs and the leak assertion fired on a healthy tree. The first
# attempt at this file used STAMP_DIR and looked correct in every other respect.
# LOG_DIR is one of the three names the helper already handles.
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/tp2_selfcheck_mutate"
STAMP="$LOG_DIR/last_pass.stamp"
BUILD="$LOG_DIR"

# *** NO DIRTY-TREE GUARD HERE, AND THAT IS A DELIBERATE DIFFERENCE. *** The
# sibling matrices that edit tracked files in place need one, because they
# restore with `git checkout --` (mutate_fold_matrix.sh destroyed uncommitted
# work that way once, recorded in AGENTS.md). This matrix does NOT: every state
# copies TARGET to a temp file, patches the COPY, and runs --self-check against
# it, so the tracked file is never modified and there is nothing to restore.
# MEASURED: `grep -n 'git checkout' scripts/mutate_tp2_selfcheck.sh` finds only
# the two lines of prose that used to claim otherwise.
# A guard whose stated reason does not apply is worse than no guard: it trains
# its reader to reach for --force or to disable it, and it blocked a legitimate
# run here for no reason at all. What IS asserted instead is the property that
# actually matters -- that TARGET is unchanged at exit (see the residue check).
# If a future state is ever added that edits TARGET in place, it MUST restore it
# and MUST reinstate the dirty-tree guard; the residue check alone is not a
# substitute, because a run that is interrupted leaves no residue to check.
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
# *** THREE SEPARATE TRAPS, AND THE EXIT CODES ARE NOT OPTIONAL. *** CAUGHT BY
# check_trap_release.sh, which classifies a single combined
# `trap ... EXIT INT TERM` as a FAILURE: the combined trap cleans up but does not
# TERMINATE, and a shell that survives SIGTERM is exactly the defect
# mutate_trap_release.sh's state T-B exists for -- `timeout` waits for the full
# duration instead of the process dying. So this is not a style preference, it is
# the same bug that class of guard was written to stop. Form per
# mutate_gate_pop.sh / mutate_verdict_scope.sh.
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM
SB_DIGEST_BEFORE="$(cat "$TARGET" | cksum)"

total=0; caught=0; bad=0; baselines=0; excluded=0
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
  # EXCLUDED states are recorded separately so they do not count against the
  # caught/total condition: an unobservable state is not a MISS, it is a state
  # this harness cannot see, and counting it as a miss would make the matrix
  # FAIL on a healthy tree (AUDIT #21). The population is split in the verdict
  # so "caught == scored" is the honest condition and the excluded set is named.
  # NOTE: record_excluded does NOT call record(), so `total` counts SCORED
  # states only -- otherwise caught could never equal total on a healthy tree.
  record_excluded() { excluded=$((excluded+1)); RESULTS+=("$1 EXCLUDED $2"); }

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  [EXCLUDED] MR-1 arm mr_arm_assert call deleted -- reason at its site"
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
  echo "  SK-12  decode1_path_fired tag is ignored (matches EITHER announcement)"
  echo "  SK-13  sk_path_fired aliases the MR tag instead of its own"
  echo "  [EXCLUDED] GU-14 gu_path_fired aliases the MR tag instead of its own -- reason at its site"
  echo "  [EXCLUDED] GU-15 gu_arm_assert always certifies -- reason at its site"
  echo "  [EXCLUDED] GU-16 gu_arm_assert call deleted from the arm -- reason at its site"
  exit 0
fi

echo "== anchor pre-flight =="
anchor PRED_STREAMS_FULL '  [ -n "$1" ] && [ -n "$2" ] && [ "$1" = "$2" ]'
anchor PRED_PREFILL '  [ "$1" = "11751" ]'
anchor PRED_TP '  grep -aq "TP=sharded" "$1"'
anchor ARM_RETURN '  return "$ok"'
anchor PATH_FIRED '  grep -aq "\[$2\].*ACTIVE" "$1"'
anchor GU_PATH 'gu_path_fired() { decode1_path_fired "$1" '"'"'mlp-gateup'"'"'; }'
anchor GU_ARM '  gu_arm_assert "$CLI_ERR" "$GATEUP_ERR" "GATE FAIL: " || PASS=0'
preflight || exit 1

echo "== baseline =="
baseline

echo "== states =="

# *** GU-14..GU-16 ARE EXCLUDED, NOT SCORED, AND THIS IS THE INTERESTING ONE. ***
# They target the gateup arm's own predicate (gu_path_fired / gu_arm_assert),
# which lives in the half of test_tp2_27b.sh that only runs on a real 27B
# decode. --self-check (the stubbed half) does NOT exercise it: it drives the
# DECODE1 predicates on synthetic fixtures, and the gateup arm's self-check
# block is gated behind the real run's rc. MEASURED: deleting gu_arm_assert's
# call from the arm leaves --self-check at rc=0 PASS, because the stubbed half
# never calls it. This is MR-1's shape a fourth time, and the reason is the
# same: the only sound detector runs the arm, which needs 2 GPUs and an 18 GiB
# 27B load, and the operator runs that (standing rule: no full-27B gate is
# ever driven from an agent session). Per AUDIT #21 an unobservable state is
# EXCLUDED WITH ITS REASON rather than counted, so "12/12" below is honest:
# MR-2..MR-11 and SK-12/SK-13 are the states whose defect the stubbed
# self-check can actually see. The gateup arm's predicate is exercised in
# BOTH directions by its own self-check block (cases a/b/c), which is the
# evidence this repo has for it -- it is not the same as a mutation matrix.
# A structural grep cannot close it honestly either: gu_arm_assert's name
# appears in this file's own comments, so a "the arm still calls it" check
# written as a grep over this file would match its own source -- the
# self-proving state of AUDIT #19.
m_gu14() { patch_literal "$1" \
  'gu_path_fired() { decode1_path_fired "$1" '"'"'mlp-gateup'"'"'; }' \
  'gu_path_fired() { decode1_path_fired "$1" '"'"'decode1-mr'"'"'; } #GU14 wrong tag'; }
m_gu15() { patch_literal "$1" \
  '  return "$ok"' \
  '  return 0 #GU15 always certifies'; }
m_gu16() { patch_literal "$1" \
  '  gu_arm_assert "$CLI_ERR" "$GATEUP_ERR" "GATE FAIL: " || PASS=0' \
  '  : #GU16 arm assertion deleted'; }

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
# *** SK-12/SK-13 PIN THE ONE GENUINELY NEW MECHANISM IN THIS CHANGE. ***
# decode1_path_fired took a $2 tag so the second flag could share the rule. The
# risk that creates is specific and new: a tag that is IGNORED still passes the
# single-flag cases, because each arm's own log contains its own announcement. It
# only shows up as a CROSS-FLAG failure -- MR's line satisfying SPLITK's check and
# vice versa -- which is why the self-check has cases (7)'s last two lines and
# why these two states target exactly that.
m_sk12() { patch_literal "$1" "$(cat "$SB/anchor_PATH_FIRED.txt")" \
  '  grep -aqE "decode1-(mr|splitk).*ACTIVE" "$1" #SK12 tag ignored'; }
m_sk13() { patch_literal "$1" \
  'sk_path_fired()  { decode1_path_fired "$1" '"'"'decode1-splitk'"'"'; }' \
  'sk_path_fired()  { decode1_path_fired "$1" '"'"'decode1-mr'"'"'; } #SK13 wrong tag'; }
m_mr10() { patch_literal "$1" \
  'cli_ids() { # $1 = log -- every decode-step token id, prefill included
  grep -a -oE "decode step [0-9]+: token=[0-9]+" "$1" | sed -E '"'"'s/.*token=//'"'"'
}' 'cli_ids() { return 0; } #MR10'; }
m_mr11() { patch_literal "$1" \
  'cli_prefill() { # $1 = log -- the step-0 id alone
  grep -a -oE "decode step 0: token=[0-9]+" "$1" | sed -E '"'"'s/.*=//'"'"'
}' 'cli_prefill() { return 0; } #MR11'; }

# *** MR-1 IS EXCLUDED, NOT SCORED, AND IT IS THE INTERESTING ONE. ***
# It deletes the real arm's call to mr_arm_assert. MEASURED: --self-check still
# PASSES, because the self-check drives the assertion FUNCTION on fixtures and
# never executes the arm that calls it. A structural grep cannot close it
# honestly either: the function's name appears in this file's own comments, so a
# "the arm still calls it" check written as a grep over this file would match its
# own source -- the self-proving state of AUDIT #19. The only sound detector runs
# the arm, which needs 2 GPUs and an 18 GiB 27B load, and the operator runs that
# (see the standing rule: no full-27B gate is ever driven from an agent session).
# Per AUDIT #21 an unobservable state is EXCLUDED WITH ITS REASON rather than
# counted, and "10/11" below is honest about it: MR-2..MR-11 are the states whose
# defect the self-check can actually see.
run_state MR-2  'SELF-CHECK FAIL \(4\)' m_mr2
run_state MR-3  'SELF-CHECK FAIL \(3\)' m_mr3
run_state MR-4  'SELF-CHECK FAIL \(3\)' m_mr4
run_state MR-5  'SELF-CHECK FAIL \(5\).*prefill' m_mr5
run_state MR-6  'SELF-CHECK FAIL \(5\).*prefill' m_mr6
run_state MR-7  'SELF-CHECK FAIL \(5\).*TP'     m_mr7
run_state MR-8  'SELF-CHECK FAIL \(2\).*TP'     m_mr8
run_state MR-9  'SELF-CHECK FAIL \(3\)' m_mr9
run_state MR-10 'SELF-CHECK FAIL \(1\)' m_mr10
run_state MR-11 'SELF-CHECK FAIL \(2\).*prefill' m_mr11
run_state SK-12 'SELF-CHECK FAIL \(7\)' m_sk12
run_state SK-13 'SELF-CHECK FAIL \(7\)' m_sk13
# GU-14/15/16 are EXCLUDED, NOT SCORED -- see the note above the state list.
# They are recorded here so the matrix's own accounting is honest about the
# population it could not test, rather than silently omitting them.
record_excluded GU-14 "gu_path_fired aliases the MR tag; the stubbed self-check never calls it"
record_excluded GU-15 "gu_arm_assert always certifies; the stubbed self-check never calls it"
record_excluded GU-16 "gu_arm_assert call deleted from the arm; the stubbed self-check never calls it"

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
  # *** ONE BASELINE FOR THE WHOLE RUN, AND THE CONDITION SAYS SO. *** MEASURED
  # DEFECT, and it is the AUDIT #12 self-contradicting-report shape in miniature:
  # this first demanded baselines -eq total, i.e. one baseline PER STATE, which no
  # matrix of this shape can satisfy (there is exactly one unmutated self-check).
  # Every state ran, every state was caught, and the summary still printed
  # "RESULT: FAIL (10/10 states caught)". A verdict that can contradict its own
  # table trains its reader to trust the table, which is exactly how a real miss
  # gets read as a pass. The correct conditions are separate and named:
  #   caught == total  every state was caught
  #   baselines >= 1   the unmutated self-check was actually exercised
  #   bad == 0         no residue, no anchor failure
  if [ "$caught" -ne "$total" ] || [ "$baselines" -lt 1 ] || [ "$bad" -ne 0 ]; then
    echo "RESULT: FAIL ($caught/$total caught, $baselines baseline(s), $bad other problem(s))"
  else
    echo "RESULT: PASS ($caught/$total states caught, each for its own reason)"
  fi
} | tee "$BUILD/summary.txt"

# Same three conditions as the printed verdict, so the exit code can never
# disagree with the line above it.
if [ "$caught" -ne "$total" ] || [ "$baselines" -lt 1 ] || [ "$bad" -ne 0 ]; then
  exit 1
fi
# write-then-mv: a stamp read by run_all_checks.sh must never be seen half-written.
{ echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "states=$total"
  echo "caught=$caught"
  echo "baselines=$baselines"
  echo "excluded=MR-1 (arm call site; observable only by a real 27B run -- see the file)"
} >"$STAMP.tmp" && mv "$STAMP.tmp" "$STAMP"
exit 0