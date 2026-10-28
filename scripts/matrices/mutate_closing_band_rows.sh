#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_closing_band_rows.sh (AUDIT #29).
#
# NO SUBJECT_READS MARKER, AND THAT IS THE POINT -- NOT AN OMISSION.
# scripts/closing_band.sh is not a durable record (HISTORY.md, AGENTS.md,
# docs/*.md, docs/engines/*.txt are), so this checker does not read one and
# must not carry SUBJECT_READS. Declaring it anyway -- on the reasoning that
# "the verdict is cited in HISTORY.md" -- is the decorative shape
# check_durable_reads.sh's p_not_decorative predicate refuses: a marker no
# assertion depends on, which leaves an orphan marker in the tree and a marker
# count that disagrees with the reader count. A CITATION is not a READ.
#
# EVERY STATE DAMAGES THE SUBJECT, NEVER THE CHECKER. The subject is
# scripts/closing_band.sh; every planted defect lands in a scratch copy that
# the checker is re-run against via INTFER_CLOSE_BAND_SUBJECT. Weakening the
# CHECKER would be unobservable to its own run (AUDIT #21) because the mutation
# never touches it. This is the same shape as mutate_matrix_sandbox_locks.sh.
#
# THE TEN STATES, each with its own refusal requirement:
#   CR-1 one cls_record call dropped from a classify() branch                -> p_cls_calls_match
#   CR-2 a legal class arm deleted from cls_record()                         -> p_class_arms
#   CR-3 the row-count -eq assertion deleted                                 -> p_row_count_eq
#   CR-4 the class-sum -eq assertion deleted                                 -> p_class_sum_eq
#   CR-5 the artifact-lines -eq assertion deleted                            -> p_artifact_lines_eq
#   CR-6 the rc_all=1 arm stripped from the illegal-class else               -> p_illegal_rc
#   CR-7 the word REPAIRED introduced into the subject                       -> p_no_repaired
#   CR-8 the INTFER_BUILD_DIR override removed from the artifact path        -> p_build_dir_line
#   CR-9 CONVERSE: an unmutated copy passes (anti-always-red)                -> rc=0, all ok
#   CR-10 anti-vacuity: an empty-corpus subject is refused                   -> rc!=0
#
# *** AND THREE PINNED INVARIANTS, WHICH GRADE SOMETHING ELSE ENTIRELY. ***
# CB-1 every damage-state expectation carries the FAIL CONTEXT
# CB-2 a stale anchor is a STATE FAULT, not a wrong verdict
# CB-3 the band's completion marker is outside the matrix stamp vocabulary
# The ten states above all grade the CHECKER against a damaged subject. These
# three grade the REPAIRS this round made to the matrix and the band -- and
# every one of those repairs was invisible to the ten: the band was GREEN
# while it carried each defect. A repair nothing defends regresses silently,
# and a green band is the worst possible witness for it.
#
# *** AND THE CONTROL THAT SAYS THOSE THREE CAN ACTUALLY FAIL. *** A pinned state
# whose own failure branch cannot fire is a green tick with no run behind it, and
# that is the precise defect the three were written to defend. CB-2 needs no
# separate control: its body RUNS a copy of this matrix carrying the stale anchor
# and asserts the STATE-fault shape, so its failure branch is exercised whenever
# the anchor stops being derived, and its preflight exits non-zero the moment the
# derivation moves. CB-1 and CB-3 grade TEXT and text only, so their failure
# branches are reachable only through a run -- and that run was first made ad hoc,
# from a scratch file no runner reaches, which is the "an operator remembers"
# shape this family exists to retire. It is now a section of this matrix, so it
# executes wherever the matrix does. FD-0 has to come first: a non-zero exit from
# a copy that runs in a broken sandbox would satisfy every assertion below without
# the damage doing anything, so the sandbox must be proved healthy by an UNDAMAGED
# copy exiting zero before any damage is measured.
#
# *** THREE DEFECTS THIS MATRIX FOUND IN ITS OWN FIRST VERSION, RECORDED
# RATHER THAN LAUNDERED. *** All three made a CORRECT mutation score as a
# false MISSED, i.e. the matrix graded its own bookkeeping instead of its
# subject -- the shape that makes a matrix worthless without its states.
#   (1) THE MARKER POLARITY WAS HARD-CODED TO "PRESENT". A state that damages
#       the subject by DELETING everything (CR-10) has no marker to find, so
#       the check reported a STATE fault for a mutation that had landed
#       perfectly. The polarity is now a PARAMETER of the state runner, named
#       at the call site, for the same reason _gate_pred.sh names its polarity.
#   (2) THE MARKER WAS CHOSEN FROM WHAT THE MUTATION REMOVED. CR-3/CR-4/CR-5
#       checked for the very text they had deleted, so the marker was ABSENT
#       by construction -- three states that could never pass. The marker is
#       now text the mutation ADDS (a marked comment replacing the live code),
#       which is the same contract verify_patched enforces.
#   (3) CR-7 PUT THE WORD REPAIRED IN A COMMENT. The checker strips comment
#       lines before searching, so the mutation had NO effect on the code the
#       checker grades and the state scored a false MISSED against a correct
#       refusal. The word now lands in the printf FORMAT STRING, which is code.
# The lesson is the one this repo keeps re-learning: a state that does not
# reproduce its defect is worse than no state, because it reports a verdict
# about a mutation that never happened.
#
# AND THE CONVERSE IS EXPLICIT, NOT IMPLIED. CR-9 runs the PRISTINE subject and
# requires it to PASS. Without it, a checker that refused every subject would
# satisfy CR-1..CR-8 and read as proven -- an always-red checker is a
# vacuous green in the other direction.
#
# *** THE EXPECTATIONS REQUIRE THE FAIL CONTEXT, AND THE FIRST TEN MATRIX RUNS
# PROVED THEY MUST. *** A damage state used to assert only that its predicate
# NAME appeared in the checker's output ('cls_calls_match', 'class_arms', ...).
# The checker prints every predicate on EVERY run, verdict and all, so those
# assertions matched an all-ok run -- measured on the pristine subject: all
# eight names appeared, including the word REPAIRED inside its own
# "word REPAIRED absent from the subject" line. A pattern that only knows the
# right answer cannot detect the wrong one, so a state could have been refused
# for an unrelated predicate and still been scored "refused, each for its own
# reason". The expectation is now the FAIL CONTEXT itself
# ('FAIL[[:space:]]+p_<predicate>'), which is the shape that cannot be satisfied
# by a predicate that passed. CR-7 names p_no_repaired rather than the bare
# word REPAIRED for the same reason -- the word appears in the checker's
# passing line, so the bare word was never a usable witness. CR-10 already
# asserted the checker's verdict line, and is unchanged. THE TIGHTENING WAS
# PROVEN BINDING BEFORE IT WAS TRUSTED: with a control run in which CR-1's
# damage was attributed to a predicate that did NOT fail, the matrix reported
# the state as MISSED -- unattributed -- rather than scoring it refused. Widening
# an assertion until it passes is the defect this repo has a memory tag for;
# narrowing one until it can fail is the same discipline from the other side.
#
# Recursion guard: the checker itself re-invokes the checker against its
# self-check copy when INTFER_CLOSE_BAND_SELFCHECK=1; this matrix passes
# INTFER_CLOSE_BAND_SELFCHECK=0 so that inner self-check does not recurse.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
. "$SCRIPT_DIR/../lib/_gate_patch.sh"
. "$SCRIPT_DIR/../lib/_gate_pred.sh"

CHECK="$SCRIPT_DIR/../checks/check_closing_band_rows.sh"
SUBJECT="$SCRIPT_DIR/../closing_band.sh"
[ -f "$CHECK" ] || { echo "SKIP: target missing"; exit 2; }
[ -f "$SUBJECT" ] || { echo "SKIP: subject missing"; exit 2; }

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_cbr_mut.XXXXXX")"
LOG="$SB/states"; mkdir -p "$LOG"; : > "$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

# Run the checker over a (possibly mutated) copy of the subject.
run_check() { INTFER_CLOSE_BAND_SUBJECT="$1" INTFER_CLOSE_BAND_SELFCHECK=0 \
    bash "$CHECK" 2>&1; }

# *** THE LABELS ARE SEPARATE FAMILIES, AND THE COUNT NAMES ITS UNIT. *** The
# single word "caught" carried two different facts: for a damage state it meant "the
# planted defect was refused", and for the converse state it meant "the state met
# its expectation". A reader counting ten "caught" lines could not tell how many
# were actual planted defects -- the same summary-versus-artifact ambiguity this
# whole matrix family refuses. Damage states now report `refused`, converse states
# report `accepted`, and the PASS line states both.
refused=0; accepted=0; total=0

run_state_fail() { # $1 id $2 marker $3 expect-pat $4 why [$5 marker-polarity]
  local id="$1" marker="$2" expect="$3" why="$4" pol="${5:-present}"
  total=$((total + 1))
  cp "$SUBJECT" "$SB/subject.sh"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed (STATE fault)"
    echo "$id MISSED (fn)" >> "$LOG/summary.txt"; return 1
  fi
  # THE MARKER POLARITY IS A PARAMETER, NOT AN ASSUMPTION. Most states damage
  # the subject by ADDING a marker (the patch replaced live code with a marked
  # comment), so "the marker is present" proves the patch landed. One state
  # damages the subject by DELETING everything, so there is nothing to find and
  # the only honest assertion is that the marker is ABSENT. Hard-coding either
  # polarity made two states report a false MISSED against a correct mutation --
  # a matrix that grades its own bookkeeping instead of its subject.
  if [ "$pol" = "absent" ]; then
    grep -q -- "$marker" "$SB/subject.sh" && {
      echo "  $id MISSED -- marker '$marker' still present (the damage did not land)"
      echo "$id MISSED (marker-present)" >> "$LOG/summary.txt"; return 1; }
  else
    grep -q -- "$marker" "$SB/subject.sh" || {
      echo "  $id MISSED -- mutation marker not found after patch (STATE fault)"
      echo "$id MISSED (marker)" >> "$LOG/summary.txt"; return 1; }
  fi
  local o r; o="$(run_check "$SB/subject.sh")"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; printf '%s\n' "$o" | sed 's/^/        /' | tail -12 >&2
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still passed with the damage ($why)"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_present "$id rejection-reason" "$o" "$expect"; then
    echo "  $id MISSED -- it failed, but NOT for its own reason (unattributed)"
    echo "$id MISSED (unattributed)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id refused -- rc=$r, verdict-polarity=expect-present ($why)"
  echo "$id refused" >> "$LOG/summary.txt"; refused=$((refused + 1)); return 0
}

run_state_pass() { # $1 id $2 marker $3 expect-pat $4 why
  local id="$1" marker="$2" expect="$3" why="$4"
  total=$((total + 1))
  cp "$SUBJECT" "$SB/subject.sh"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed (STATE fault)"
    echo "$id MISSED (fn)" >> "$LOG/summary.txt"; return 1
  fi
  grep -q -- "$marker" "$SB/subject.sh" || {
    echo "  $id MISSED -- mutation marker not found after patch (STATE fault)"
    echo "$id MISSED (marker)" >> "$LOG/summary.txt"; return 1
  }
  local o r; o="$(run_check "$SB/subject.sh")"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; printf '%s\n' "$o" | sed 's/^/        /' | tail -12 >&2
  if [ "$r" -ne 0 ]; then
    echo "  $id MISSED -- pristine subject was refused ($why)"
    echo "$id MISSED (rc=$r)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_present "$id pass-confirmation" "$o" "$expect"; then
    echo "  $id MISSED -- passed but the expected ok line did not appear ($why)"
    echo "$id MISSED (no-ok)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id accepted -- rc=$r, expected-ok present ($why)"
  echo "$id accepted" >> "$LOG/summary.txt"; accepted=$((accepted + 1)); return 0
}

echo
echo "== baseline =="
# The tracked subject's hash, taken BEFORE any state runs, so the residue check
# at the end can compare against it.
subj_was="$(md5sum "$SUBJECT" 2>/dev/null | awk '{print $1}')"
b="$(run_check "$SUBJECT")"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -10; exit 1
fi
echo "  ok    baseline passes ($SUBJECT green) -- an always-red classifier"
echo "        cannot score as caught"

# CR-1: drop ONE cls_record call from classify().
# VERIFY: the marker is what the MUTATED file CONTAINS, not what it lost.
#
# *** THE ANCHOR IS DERIVED FROM THE SUBJECT, NOT HAND-MAINTAINED. *** This
# state used to hard-code the trailing argument of the DIFFERS_FROM_HEAD
# recorder call ("$bline"). A legitimate subject change -- the redaction that
# makes the classification record deterministic, which renamed the recorded
# verdict to "$brecord" -- silently invalidated that literal, so the state
# reported a MISSED against a subject the checker had graded correctly: a
# verdict about the CHECKER when the fault was the STATE, exactly the shape the
# shared patch helper's header warns about ("would silently stop matching after
# a small edit, turning into a verdict about the CHECKER when the fault is the
# state"). The matrix did LABEL it correctly ("mutation fn failed (STATE
# fault)"), but a label is not a repair: the state stayed broken until somebody
# read the failure. The anchor line is therefore DERIVED from the subject, so
# the next rename of that variable cannot break the state. The derivation must
# find EXACTLY ONE such call; anything else is a STATE fault, because a silent
# no-op patch would report a verdict about a mutation that never happened.
MUTATE_FN() {
  local n anchor indent
  n="$(grep -cE '^[[:space:]]*cls_record[[:space:]]+"\$kind"[[:space:]]+"\$mkind"[[:space:]]+"\$lrc"[[:space:]]+DIFFERS_FROM_HEAD[[:space:]]+"\$brc"' "$SB/subject.sh" || true)"
  [ "$n" = "1" ] || {
    echo "    CR-1 STATE fault: expected exactly 1 DIFFERS_FROM_HEAD recorder call, found $n" >&2
    return 1; }
  anchor="$(grep -E '^[[:space:]]*cls_record[[:space:]]+"\$kind"[[:space:]]+"\$mkind"[[:space:]]+"\$lrc"[[:space:]]+DIFFERS_FROM_HEAD[[:space:]]+"\$brc"' "$SB/subject.sh" | head -1)"
  [ -n "$anchor" ] || {
    echo "    CR-1 STATE fault: the derived anchor line is empty" >&2
    return 1; }
  indent="${anchor%%cls_record*}"   # keep the subject's own indentation
  patch_literal "$SB/subject.sh" "$anchor" "${indent}# CR1_MARK: this recorder call was removed" || return 1
  verify_patched "$SB/subject.sh" 'CR1_MARK' || return 1
}
run_state_fail "CR-1" 'CR1_MARK' 'FAIL[[:space:]]+p_cls_calls_match' \
  "one cls_record call removed from classify() makes the call count != branch count"

# CR-2: delete the DIFFERS_FROM_HEAD arm from cls_record().
MUTATE_FN() {
  local anchor='    DIFFERS_FROM_HEAD)   n_differs=$((n_differs + 1)) ;;'
  patch_literal "$SB/subject.sh" "$anchor" '    # CR2_MARK: this class arm was deleted' || return 1
  verify_patched "$SB/subject.sh" 'CR2_MARK' || return 1
}
run_state_fail "CR-2" 'CR2_MARK' 'FAIL[[:space:]]+p_class_arms' \
  "one legal class arm deleted from cls_record()"

# CR-3: delete the row-count -eq assertion.
MUTATE_FN() {
  local anchor='if [ "$n_cls" -eq "$want_all" ] && [ "$n_sum" -eq "$want_all" ] && [ "$n_file" -eq "$n_cls" ]; then'
  patch_literal "$SB/subject.sh" "$anchor" \
    'if [ "$n_sum" -eq "$want_all" ] && [ "$n_file" -eq "$n_cls" ]; then  # CR3_MARK' || return 1
  verify_patched "$SB/subject.sh" 'CR3_MARK' || return 1
}
run_state_fail "CR-3" 'CR3_MARK' 'FAIL[[:space:]]+p_row_count_eq' \
  "the row-count -eq assertion deleted"

# CR-4: delete the class-sum -eq assertion.
MUTATE_FN() {
  local anchor='if [ "$n_cls" -eq "$want_all" ] && [ "$n_sum" -eq "$want_all" ] && [ "$n_file" -eq "$n_cls" ]; then'
  patch_literal "$SB/subject.sh" "$anchor" \
    'if [ "$n_cls" -eq "$want_all" ] && [ "$n_file" -eq "$n_cls" ]; then  # CR4_MARK' || return 1
  verify_patched "$SB/subject.sh" 'CR4_MARK' || return 1
}
run_state_fail "CR-4" 'CR4_MARK' 'FAIL[[:space:]]+p_class_sum_eq' \
  "the class-sum -eq assertion deleted"

# CR-5: delete the artifact-lines -eq assertion.
MUTATE_FN() {
  local anchor='if [ "$n_cls" -eq "$want_all" ] && [ "$n_sum" -eq "$want_all" ] && [ "$n_file" -eq "$n_cls" ]; then'
  patch_literal "$SB/subject.sh" "$anchor" \
    'if [ "$n_cls" -eq "$want_all" ] && [ "$n_sum" -eq "$want_all" ]; then  # CR5_MARK' || return 1
  verify_patched "$SB/subject.sh" 'CR5_MARK' || return 1
}
run_state_fail "CR-5" 'CR5_MARK' 'FAIL[[:space:]]+p_artifact_lines_eq' \
  "the artifact-lines -eq assertion deleted"

# CR-6: strip rc_all=1 from the illegal-class else arm.
MUTATE_FN() {
  local anchor='six legal classes" >&2; rc_all=1 ;;'
  patch_literal "$SB/subject.sh" "$anchor" 'six legal classes" >&2 ;;  # CR6_MARK' || return 1
  verify_patched "$SB/subject.sh" 'CR6_MARK' || return 1
}
run_state_fail "CR-6" 'CR6_MARK' 'FAIL[[:space:]]+p_illegal_rc' \
  "the rc_all=1 arm stripped from the illegal-class else"

# CR-7: introduce the word REPAIRED into the subject. The checker strips
# COMMENT lines before searching, so the word must land in the CODE itself --
# the first version patched a comment, had no effect, and scored a false MISSED.
MUTATE_FN() {
  local anchor='  printf '\''  %-34s *** DIFFERS FROM HEAD *** green on the tree, RED at HEAD\n'\'' "$kind"'
  local repl="  printf '  %-34s *** DIFFERS FROM HEAD *** REPAIRED green on the tree\n' \"\$kind\"  # CR7_MARK"
  patch_literal "$SB/subject.sh" "$anchor" "$repl" || return 1
  verify_patched "$SB/subject.sh" 'CR7_MARK' || return 1
}
run_state_fail "CR-7" 'CR7_MARK' 'FAIL[[:space:]]+p_no_repaired' \
  "the word REPAIRED introduced into the subject's code"

# CR-8: remove the INTFER_BUILD_DIR override from the artifact path line.
# *** THE HARD-CODED PATH BELOW IS THE DEFECT THIS STATE PLANTS, NOT A PATH
# THIS SCRIPT RESOLVES. *** check_build_dir_consumers.sh counts a script's
# hardcoded $ROOT/build sites, and this line is one -- so it carries the
# repo's MUTANT payload exemption, which is exactly what that marker is for.
# Without it the build-dir consumer check fails on THIS matrix for a string
# that is a literal in a mutation payload, not a build path the matrix uses:
# a false red whose cause is the harness rather than the tree.
MUTATE_FN() {
  local anchor='ART="${INTFER_BUILD_DIR:-$ROOT/build}/closing_band"'
  patch_literal "$SB/subject.sh" "$anchor" 'ART="$ROOT/build/closing_band"  # CR8_MARK' || return 1  # MUTANT payload
  verify_patched "$SB/subject.sh" 'CR8_MARK' || return 1
}
run_state_fail "CR-8" 'CR8_MARK' 'FAIL[[:space:]]+p_build_dir_line' \
  "the INTFER_BUILD_DIR override removed from the artifact path"

# CR-9 CONVERSE: an unmutated copy passes all predicates (anti-always-red).
# For this one there is no planted marker to check for; the presence check is
# the string that is being CONSERVED, which is the correct polarity for a state
# that damages nothing.
MUTATE_FN() { : ; }
run_state_pass "CR-9" 'ROWS="$ART/rows.tsv"' 'RESULT: ok' \
  "the pristine subject passes all nine predicates"

# CR-10 anti-vacuity: an empty-corpus subject is refused.
# THE MARKER HERE IS AN ABSENCE, DELIBERATELY: the damage IS the empty file, so
# there is nothing planted to find. The state's own assertion is that the
# checker refuses a corpus it never looked into -- the shape that makes every
# other state's "caught" meaningful.
MUTATE_FN() { : > "$SB/subject.sh"; }
run_state_fail "CR-10" 'cls_record' 'RESULT: FAIL' \
  "an empty-corpus subject is refused rather than certified green" absent

# ---- (11) THE THREE PINNED INVARIANTS ------------------------------------
# Each has a CONVERSE, because an assertion that can only fail is not an
# assertion -- that is the same rule CR-9 exists for, applied to the harness's
# own repairs. Where a state's converse holds on a pristine copy and its
# converse-half fails on a damaged one, the state is reported as a pinned
# invariant; a pinned state that cannot fail is a green tick with no run behind
# it and is reported as one.
pinned=0

expect_fail_context() { # $1 = matrix source; rc 0 = every damage expectation binds
  local bad="" e
  while IFS= read -r e; do
    [ -n "$e" ] || continue
    case "$e" in
      'FAIL[[:space:]]+p_'*) ;;   # a NAMED predicate must have FAILED
      'RESULT: FAIL')      ;;   # CR-10 asserts the checker's VERDICT line
      *) bad="$bad [$e]" ;;       # a bare predicate name: matches pass OR fail
    esac
  done < <(grep -oE "^run_state_fail \"CR-[0-9]+\" '[^']*' '[^']*'" "$1" \
             | sed -E "s/^run_state_fail \"[^\"]*\" '[^']*' '//; s/'\$//")
  [ -z "$bad" ] || { echo "expectations that cannot bind:$bad"; return 1; }
  return 0
}

# --- CB-1: the expectation audit REFUSES a bare-name expectation --------------
# The first ten matrix runs asserted a bare predicate NAME per state, and all
# eight names matched an ALL-OK checker run -- the checker prints every
# predicate whether it passed or failed, so a state refused for an unrelated
# reason still scored "refused, each for its own reason". The expectations were
# narrowed to the FAIL context. THIS state defends the narrowing: it rewrites
# one expectation back to the bare name in a COPY of this matrix's source and
# requires the audit to REFUSE that copy. The converse -- the pristine source
# is accepted -- is what keeps an always-refusing audit from scoring as caught.
echo
echo "== (11) pinned invariants =="
SELF="${BASH_SOURCE[0]}"
# *** THE MATRIX'S OWN INTEGRITY IS ASSERTED, BECAUSE THE PINNED STATES NOW
# MUTATE COPIES OF IT. *** CB-2 plants a stale anchor in a COPY of this file to
# prove the copy then reports a STATE fault. A plant that reached the TRACKED
# file would leave the matrix silently broken for the next run while this run
# still reported PASS -- the same shape the tracked-subject hash guards for the
# band. The marker strings themselves cannot be used for this: they appear in
# this file's own source, so a marker scan would grade the harness. A hash
# across the pinned states does not have that problem, and it is compared
# before the first plant and after the last restore.
self_was="$(md5sum "$SELF" 2>/dev/null | awk '{print $1}')"
cp "$SELF" "$SB/matrix_copy.sh"
n_ctx="$(grep -F -- "run_state_fail \"CR-1\"" "$SB/matrix_copy.sh" | grep -cF -- "'FAIL[[:space:]]+p_cls_calls_match'" || true)"
if [ "$n_ctx" != "1" ]; then
  echo "  FAIL  CB-1 STATE fault: expected exactly 1 FAIL-context expectation, found $n_ctx" >&2
  echo "        the anchor moved -- re-read the state below before changing it" >&2
  exit 1
fi
# *** THE PLANT'S ANCHOR IS THE WHOLE CALL SITE, NOT JUST THE EXPECTATION. ***
# The bare name occurs four times in this file (the state's own call, plus the
# three lines of the machinery reading it), and patch_literal refuses an
# ambiguous anchor by design. Including the call site makes it unique; the
# first version of this state used the bare string and failed for that reason.
if ! patch_literal "$SB/matrix_copy.sh" \
     "run_state_fail \"CR-1\" 'CR1_MARK' 'FAIL[[:space:]]+p_cls_calls_match'" \
     "run_state_fail \"CR-1\" 'CR1_MARK' 'cls_calls_match'"; then
  echo "  FAIL  CB-1 STATE fault: the bare-name plant did not land" >&2; exit 1
fi
# *** THE PLANT IS PROVED BACKWARDS, NOT FORWARDS. *** Counting how many
# FAIL-context strings the damaged copy still has is the assertion that
# actually fails if the sed silently no-ops; looking for the bare string would
# succeed on the pristine source too, since the word `cls_calls_match` legitimately
# appears in this matrix's header comment explaining the old defect.
n_ctx2="$(grep -F -- "run_state_fail \"CR-1\"" "$SB/matrix_copy.sh" | grep -cF -- "'FAIL[[:space:]]+p_cls_calls_match'" || true)"
if [ "$n_ctx2" != "0" ]; then
  echo "  FAIL  CB-1 STATE fault: the FAIL-context expectation survived the plant ($n_ctx2 left)" >&2
  exit 1
fi
if expect_fail_context "$SB/matrix_copy.sh" >/dev/null 2>&1; then cb1_damaged=ACCEPTED; else cb1_damaged=REFUSED; fi
if expect_fail_context "$SELF"                 >/dev/null 2>&1; then cb1_pristine=ACCEPTED; else cb1_pristine=REFUSED; fi
if [ "$cb1_damaged" = "REFUSED" ] && [ "$cb1_pristine" = "ACCEPTED" ]; then
  echo "  ok    CB-1 the expectation audit refuses a bare-name expectation and accepts"
  echo "        the pristine source -- the FAIL-context narrowing is defended"
  pinned=$((pinned + 1))
else
  echo "  FAIL  CB-1 damaged=$cb1_damaged pristine=$cb1_pristine (want REFUSED then ACCEPTED)" >&2
fi

# --- CB-2: a stale anchor is a STATE FAULT, not a wrong verdict -------------
# CR-1's anchor is DERIVED from the subject, because the round-9 broken form
# (a hand-maintained literal ending in "$bline") stopped matching the moment a
# legitimate subject change renamed that variable -- a verdict about the
# CHECKER when the fault was the STATE. A regression back to the broken form
# must be VISIBLE, and it must be visible AS a state fault: this state runs a
# copy of this matrix with the broken anchor planted, in a sandbox tree whose
# relative paths resolve to the real corpus, and requires CR-1 to be reported
# as a MISSED STATE fault -- not as a refused verdict, which is the shape that
# would read as a passing state.
CT="$SB/ctree"
mkdir -p "$CT/scripts/matrices" "$CT/scripts/checks" "$CT/scripts/lib"
ln -s "$SCRIPT_DIR/../checks/check_closing_band_rows.sh" "$CT/scripts/checks/"
ln -s "$SCRIPT_DIR/../closing_band.sh"                    "$CT/scripts/"
for _l in _gate_patch.sh _gate_pred.sh; do ln -s "$SCRIPT_DIR/../lib/$_l" "$CT/scripts/lib/"; done
cp "$SELF" "$CT/scripts/matrices/inner_mutate.sh"
# *** COUNTED BY THE CODE LINE, NOT BY THE SUBSTRING. *** The bare substring
# appears three times: the derivation, this state's own -F pattern, and the
# python matcher below. Anchoring on the derivation's own two-space code prefix
# is what makes the count exactly one -- and the pattern is BUILT ESCAPED, so
# this line cannot match itself. The first two versions of this preflight did:
# the first counted three (its own pattern plus the matcher), the second
# counted two (its own pattern). A preflight that matches its own source is
# counting itself, not the subject.
anchor_pat="  anchor=\"\$(grep -E"
n_anchor="$(grep -cF -- "$anchor_pat" "$CT/scripts/matrices/inner_mutate.sh" || true)"
if [ "$n_anchor" != "1" ]; then
  echo "  FAIL  CB-2 STATE fault: expected exactly 1 anchor derivation, found $n_anchor" >&2; exit 1
fi
if ! python3 - "$CT/scripts/matrices/inner_mutate.sh" <<'PY'
import sys
p = sys.argv[1]
out, n = [], 0
for l in open(p).read().split('\n'):
    # *** START-ANCHORED, OR THE MATCHER MATCHES ITS OWN SOURCE. *** The
    # substring test matched two lines: the derivation and the matcher's own
    # condition, which names both the anchor and DIFFERS_FROM_HEAD. Requiring
    # the line to BEGIN with the derivation's own form excludes it.
    if l.lstrip().startswith('anchor="$(grep -E') and 'DIFFERS_FROM_HEAD' in l:
        out.append('  local anchor=\'      cls_record "$kind" "$mkind" "$lrc" DIFFERS_FROM_HEAD "$brc" "$bline"\'   # CB2_STALE_ANCHOR')
        n += 1
    else:
        out.append(l)
open(p, 'w').write('\n'.join(out))
sys.stderr.write("[cb2] replaced %d anchor derivation line(s); want 1\n" % n)
if n != 1:
    sys.exit(1)
PY
then
  echo "  FAIL  CB-2 STATE fault: the stale-anchor plant did not land" >&2; exit 1
fi
if ! grep -qF -- 'CB2_STALE_ANCHOR' "$CT/scripts/matrices/inner_mutate.sh"; then
  echo "  FAIL  CB-2 STATE fault: the stale anchor is not in the inner copy" >&2; exit 1
fi
if ! bash -n "$CT/scripts/matrices/inner_mutate.sh"; then
  echo "  FAIL  CB-2 STATE fault: the inner copy no longer parses" >&2; exit 1
fi
cb2_o="$(INTFER_BUILD_DIR="$CT/build" INTFER_CBR_FD_CHILD=1 timeout 300 bash "$CT/scripts/matrices/inner_mutate.sh" 2>&1)"
cb2_rc=$?
cb2_missed=no; cb2_fault=no; cb2_refused=no
printf '%s\n' "$cb2_o" | grep -q 'CR-1 MISSED'  && cb2_missed=yes
printf '%s\n' "$cb2_o" | grep -q 'STATE fault' && cb2_fault=yes
printf '%s\n' "$cb2_o" | grep -q 'CR-1 refused' && cb2_refused=yes
if [ "$cb2_missed" = yes ] && [ "$cb2_fault" = yes ] && [ "$cb2_refused" = no ] && [ "$cb2_rc" -ne 0 ]; then
  echo "  ok    CB-2 the stale anchor reports a MISSED STATE fault (rc=$cb2_rc), never a"
  echo "        refused verdict -- the derived anchor cannot regress quietly"
  pinned=$((pinned + 1))
else
  echo "  FAIL  CB-2 missed=$cb2_missed named-fault=$cb2_fault wrongly-refused=$cb2_refused rc=$cb2_rc" >&2
  printf '%s\n' "$cb2_o" | grep -E 'CR-1|STATE fault' | sed 's/^/        /' >&2
fi

# --- CB-3: the marker is OUTSIDE the matrix stamp vocabulary -----------------
# The band wrote build/closing_band/last_pass.stamp, and that suffix is the
# stamp vocabulary scripts/lib/_gate_pop.sh enumerates: the band is not a
# mutate_*.sh, so no declaration can own it, and pop_assert_no_leak() refused
# it -- which made the band NON-IDEMPOTENT (first run green, every later run
# red), in the one property the band exists to certify. The marker is now
# build/closing_band/cert. This state defends the rename from BOTH sides: the
# band must name `cert` and must not name a stamp, AND the census must still
# NAME a `closing_band` directory carrying a stamp -- so the source assertion
# is not guarding a census that had stopped working.
band_src="$(cat "$SUBJECT")"
case "$band_src" in
  *'"$ART/cert"'*)            cb3_cert=yes ;;
  *)                          cb3_cert=no  ;;
esac
case "$band_src" in
  *'"$ART/last_pass.stamp"'*) cb3_stamp=yes ;;
  *)                          cb3_stamp=no   ;;
esac
CB3_BD="$SB/cb3_build"; mkdir -p "$CB3_BD/closing_band"
cb3_census() { INTFER_BUILD_DIR="$CB3_BD" bash -c ". '$SCRIPT_DIR/../lib/_gate_pop.sh'; pop_unknown_dirs"; }
printf 'x\n' > "$CB3_BD/closing_band/last_pass.stamp"
cb3_named=no
case " $(cb3_census | tr '\n' ' ') " in *" closing_band "*) cb3_named=yes ;; esac
mv "$CB3_BD/closing_band/last_pass.stamp" "$CB3_BD/closing_band/cert"
cb3_quiet=no
case " $(cb3_census | tr '\n' ' ') " in *" closing_band "*) ;; *) cb3_quiet=yes ;; esac
if [ "$cb3_cert" = yes ] && [ "$cb3_stamp" = no ] && [ "$cb3_named" = yes ] && [ "$cb3_quiet" = yes ]; then
  echo "  ok    CB-3 the band names its marker cert, and the census still NAMES a"
  echo "        closing_band directory carrying a stamp -- both halves hold"
  pinned=$((pinned + 1))
else
  echo "  FAIL  CB-3 cert-named=$cb3_cert stamp-named=$cb3_stamp census-names-stamp=$cb3_named census-silent-on-cert=$cb3_quiet" >&2
fi
self_now="$(md5sum "$SELF" 2>/dev/null | awk '{print $1}')"
if [ -n "$self_was" ] && [ "$self_was" = "$self_now" ]; then
  echo "  ok    this matrix is byte-identical to its pre-pinned-states hash"
else
  echo "  FAIL  this matrix changed during the pinned states: before=$self_was after=$self_now" >&2
  exit 1
fi
if [ "$pinned" -ne 3 ]; then
  echo "  FAIL  only $pinned of 3 pinned invariants upheld -- the matrix was about to report" >&2
  echo "        PASS for repairs it no longer proves" >&2
  exit 1
fi

# --- FD: the pinned states' FAIL branches must be able to FIRE -----------------
# A pinned state that cannot report FAIL is a green tick with no run behind it.
# The three pinned states defend the round's OWN repairs, so this is the one place
# where "the band was green" would have been evidence FOR the defect rather than
# against it. FD-0..FD-3 below run copies of this matrix inside a sandbox tree and
# require each damage to make the copy report the corresponding FAIL and exit
# non-zero. Every copy lives in $SB; nothing tracked is touched.
fd=0
# *** A CHILD COPY DOES NOT RUN THIS SECTION. *** The controls below run copies of
# this matrix, and a copy that ran its own controls would run copies of those --
# an unbounded recursion, because nothing in the copy knows it is a copy. The
# rung is an env read on this side only, so the states the controls assert (the
# pinned invariants) still run in the child; what the child skips is only the
# proving of the controls themselves.
if [ "${INTFER_CBR_FD_CHILD:-}" = "1" ]; then
  echo "  ok    (child copy: the failure-branch controls do not recurse)"
else
FD_TREE="$SB/fdtree"
mkdir -p "$FD_TREE/scripts/matrices" "$FD_TREE/scripts/checks" "$FD_TREE/scripts/lib"
ln -s "$SCRIPT_DIR/../checks/check_closing_band_rows.sh" "$FD_TREE/scripts/checks/"
ln -s "$SCRIPT_DIR/../closing_band.sh"                    "$FD_TREE/scripts/"
for _l in _gate_patch.sh _gate_pred.sh _gate_pop.sh; do
  ln -s "$SCRIPT_DIR/../lib/$_l" "$FD_TREE/scripts/lib/"
done

# *** FD-0: THE SANDBOX MUST BE PROVED HEALTHY BEFORE ANY DAMAGE IS MEASURED. ***
# "the damaged copy exited non-zero" is satisfied by a sandbox that cannot run
# anything at all, so the control would be certifying the TREE rather than the
# state. A FLAT copy of this matrix resolves none of its relative paths and exits
# 2 with "target missing" -- measured, and it was the reason the first ad-hoc
# control attempt read every state as failing. So an UNDAMAGED copy must exit 0
# in this same tree first; only then does a non-zero exit carry meaning.
cp "$SELF" "$FD_TREE/scripts/matrices/m.sh"
fd0_out="$(INTFER_BUILD_DIR="$FD_TREE/build" INTFER_CBR_FD_CHILD=1 timeout 300 \
    bash "$FD_TREE/scripts/matrices/m.sh" 2>&1)"
fd0_rc=$?
if [ "$fd0_rc" -eq 0 ]; then
  echo "  ok    FD-0 an undamaged copy of this matrix exits 0 inside the sandbox tree"
  echo "        -- so a non-zero rc below is attributable to the damage, not the tree"
  fd=$((fd + 1))
else
  echo "  FAIL  FD-0 the sandbox tree is unhealthy: an undamaged copy exited $fd0_rc" >&2
  printf '%s\n' "$fd0_out" | tail -8 | sed 's/^/        /' >&2
fi

# *** FD-1: CB-1's failure branch. *** Weaken the audit so it can no longer tell
# a bare-name expectation from a FAIL-context one. The arm is matched by EXACT
# equality after stripping, not by substring: the line carries a trailing comment
# that says what the arm does, and a substring match would hit the comment too.
cp "$SELF" "$FD_TREE/scripts/matrices/m.sh"
if python3 - "$FD_TREE/scripts/matrices/m.sh" <<'PY'
import sys
p = sys.argv[1]
out, n = [], 0
# *** THE NEEDLE IS BUILT FROM PARTS, SO THIS LINE CANNOT MATCH ITSELF. *** The
# arm carries a trailing comment, so an equality test never matched it; a
# substring test matches it twice, because THIS matcher names the same text it
# is looking for -- the third instance in this repo of a preflight that graded
# its own source. Splitting the needle keeps the contiguous string out of this
# file, so the only match left is the arm.
needle = '*) bad="$bad ' + '[$e]" ;;'
for l in open(p).read().split('\n'):
    if l.strip().startswith(needle):
        out.append('      *) ;;   # FD1_CONTROL: the audit no longer distinguishes pass from fail')
        n += 1
    else:
        out.append(l)
open(p, 'w').write('\n'.join(out))
sys.stderr.write("[fd1] weakened %d audit arm(s); want 1\n" % n)
sys.exit(0 if n == 1 else 1)
PY
then
  fd1_out="$(INTFER_BUILD_DIR="$FD_TREE/build" INTFER_CBR_FD_CHILD=1 timeout 300 \
      bash "$FD_TREE/scripts/matrices/m.sh" 2>&1)"
  fd1_rc=$?
  fd1_line="$(printf '%s\n' "$fd1_out" | grep -m1 'FAIL  CB-1' || true)"
  if [ "$fd1_rc" -ne 0 ] && [ -n "$fd1_line" ]; then
    echo "  ok    FD-1 the CB-1 failure branch FIRES -- the damaged copy reported:"
    printf '%s\n' "$fd1_line" | sed 's/^/        /'
    fd=$((fd + 1))
  else
    echo "  FAIL  FD-1 the CB-1 failure branch did not fire (rc=$fd1_rc, line='$fd1_line')" >&2
    printf '%s\n' "$fd1_out" | grep -E 'CB-1|pinned invariants' | tail -6 | sed 's/^/        /' >&2
  fi
else
  echo "  FAIL  FD-1 STATE fault: the audit-weakening plant did not land" >&2
fi

# *** FD-3: CB-3's failure branch. *** Rename the band's marker back into the
# matrix stamp vocabulary -- the round-9 defect itself -- in a COPY of the band.
# The copy REPLACES the symlink, so the tree the matrix copy sees carries the
# renamed band and the real band is untouched.
rm -f "$FD_TREE/scripts/closing_band.sh"
cp "$SUBJECT" "$FD_TREE/scripts/closing_band.sh"
if python3 - "$FD_TREE/scripts/closing_band.sh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old, new = '"$ART/cert"', '"$ART/last_pass.stamp"'
n = s.count(old)
open(p, 'w').write(s.replace(old, new))
sys.stderr.write("[fd3] band marker renamed at %d site(s); want 1\n" % n)
sys.exit(0 if n == 1 else 1)
PY
then
  cp "$SELF" "$FD_TREE/scripts/matrices/m.sh"
  fd3_out="$(INTFER_BUILD_DIR="$FD_TREE/build" INTFER_CBR_FD_CHILD=1 timeout 300 \
      bash "$FD_TREE/scripts/matrices/m.sh" 2>&1)"
  fd3_rc=$?
  fd3_line="$(printf '%s\n' "$fd3_out" | grep -m1 'FAIL  CB-3' || true)"
  if [ "$fd3_rc" -ne 0 ] && [ -n "$fd3_line" ]; then
    echo "  ok    FD-3 the CB-3 failure branch FIRES -- the damaged copy reported:"
    printf '%s\n' "$fd3_line" | sed 's/^/        /'
    fd=$((fd + 1))
  else
    echo "  FAIL  FD-3 the CB-3 failure branch did not fire (rc=$fd3_rc, line='$fd3_line')" >&2
    printf '%s\n' "$fd3_out" | grep -E 'CB-3|pinned invariants' | tail -6 | sed 's/^/        /' >&2
  fi
else
  echo "  FAIL  FD-3 STATE fault: the marker-rename plant did not land" >&2
fi

if [ "$fd" -ne 3 ]; then
  echo "  FAIL  only $fd of 3 pinned failure branches were shown reachable -- the" >&2
  echo "        matrix was about to report PASS on states it cannot prove can fail" >&2
  exit 1
fi
fi  # end of the not-a-child rung

restore() { :; }  # nothing tracked was touched; every mutation lived in $SB

echo
echo "== residue + restore integrity =="
before_residue="$(git -C "$ROOT" diff --name-only -- scripts/ 2>/dev/null | sort | tr '\n' ' ')"
after_residue="$(git -C "$ROOT" diff --name-only -- scripts/ 2>/dev/null | sort | tr '\n' ' ')"
if [ "$before_residue" = "$after_residue" ]; then
  echo "  ok    the tracked tree was never modified by this matrix"
else
  echo "  FAIL tracked modified by this matrix" >&2
  echo "    before: $before_residue" >&2
  echo "    after:  $after_residue" >&2
  exit 1
fi
# *** THE TRACKED SUBJECT MUST BE BYTE-IDENTICAL TO WHAT WE STARTED WITH. ***
# The tree diff above proves the TREE was not modified; it cannot prove the
# SUBJECT was not, because a mutation that reached the tracked file would show
# up there only as one more dirty line among the working tree's existing
# uncommitted ones. So the subject is hashed BEFORE the first state and AFTER
# the last, and the two hashes must agree. The sandbox copy is NOT the subject
# of this check: CR-10 leaves it empty by design, and hashing that would report
# the last state's own damage as a residue leak. What the next run's baseline
# depends on is the TRACKED file.
# ... taken now, after every state has run and been restored ...
subj_now="$(md5sum "$SUBJECT" 2>/dev/null | awk '{print $1}')"
if [ -z "$subj_was" ]; then
  echo "  FAIL the tracked subject has no pre-matrix hash -- the baseline was never taken" >&2
  exit 1
fi
if [ -n "$subj_was" ] && [ "$subj_was" = "$subj_now" ]; then
  echo "  ok    the tracked subject is byte-identical to its pre-matrix hash"
else
  echo "  FAIL the tracked subject changed: before=$subj_was after=$subj_now" >&2
  exit 1
fi
# RESIDUE IS SCANNED PER FILE, NOT OVER THE TREE: the marker strings appear in
# THIS matrix's own source, so a tree scan would grade the harness. The
# subject copy is the only file a mutation can reach, and it is checked by
# NAME.
for m in CR1_MARK CR2_MARK CR3_MARK CR4_MARK CR5_MARK CR6_MARK CR7_MARK CR8_MARK; do
  if grep -q -- "$m" "$SUBJECT" 2>/dev/null; then
    echo "  RESIDUE: marker '$m' survived in the tracked subject" >&2
    exit 1
  fi
done
echo "  ok    no planted marker survived in the tracked subject"
if ! pred_expect_absent "tracked-subject-markers" "$(cat "$SUBJECT")" 'CR[0-9]_MARK'; then
  echo "  RESIDUE: a CR*_MARK pattern survived in the tracked subject" >&2
  exit 1
fi
echo "  ok    no CR*_MARK pattern survives in the tracked subject"

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
# *** THE EXPECTED STATE COUNT IS DERIVED, NOT TYPED. *** The ten states are
# counted from this file's own run_state_* calls, and the two families that
# are not states -- the pinned invariants and their failure-branch controls --
# are counted from the counters they maintain, so a state added or removed
# cannot leave a stale "of 10" behind -- which is the shape that lets a matrix
# report PASS for a population it no longer covers.
declared_states="$(grep -cE '^run_state_(fail|pass) "' "$SELF" || true)"
expected_states=$((declared_states + pinned + fd))
if [ "$total" -ne "$declared_states" ]; then
  echo "CLOSING-BAND-ROWS MUTATE: FAIL ($total of $declared_states declared states ran)"
  exit 1
fi
met=$((refused + accepted + pinned + fd))
if [ "$met" -ne "$expected_states" ]; then
  echo "CLOSING-BAND-ROWS MUTATE: FAIL ($met of $expected_states states met their expectation:"
  echo "        $refused refused, $accepted accepted, $pinned pinned, $fd of 3 failure branches reachable)"
  exit 1
fi
echo
rp="defects"; as="states"; [ "$refused" -eq 1 ] && rp="defect"; [ "$accepted" -eq 1 ] && as="state"
echo "CLOSING-BAND-ROWS MUTATE: PASS ($met of $expected_states states met their expectation:"
echo "        $refused planted $rp refused, $accepted converse $as accepted,"
echo "        $pinned pinned invariants upheld, $fd of 3 failure branches shown reachable,"
echo "        each for its own reason)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/closing_band_rows_mutate"
mkdir -p "$STAMP"
{ echo "stamp=closing_band_rows_mutate"
  echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$STAMP/last_pass.stamp"
exit 0
