#!/usr/bin/env bash
# Mutation matrix for scripts/check_bench_load_gate.sh (AUDIT #23).
#
# THE MODEL IS AUDIT #21 APPLIED TO THE ARTIFACT. The harnesses are CORRECT after
# this change, so weakening the CHECKER is unobservable -- it would keep passing.
# Each state therefore damages a HARNESS in a sandbox and requires the checker to
# reject the damage.
#
# The whole scripts/ directory is COPIED into the sandbox rather than symlinked:
# the checker derives its population by globbing scripts/*.sh and then greps those
# files, so a symlink would make every mutation edit the REAL tree. Copying is
# what makes the sandbox safe, and it also makes the derived population faithful
# instead of hand-listed.
#
# Usage: scripts/mutate_bench_load_gate.sh
# Exit: 0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_patch.sh
. "$SCRIPT_DIR/_gate_patch.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/bench_load_gate_mutate"
LOG="$STAMP/states"
CHECK="$SCRIPT_DIR/check_bench_load_gate.sh"
TARGET="$SCRIPT_DIR/bench_quiet_ab.sh"

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null || {
  echo "FAIL: scripts/ is dirty; this matrix edits a sandbox copy."; exit 2; }

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_blg.XXXXXX")"
trap 'rm -rf "$SB"' EXIT
mkdir -p "$SB/scripts" "$LOG"
# Reset the summary. Without this it ACCUMULATES across runs and the printed
# block contradicts its own verdict -- it listed "B-A MISSED (fn)" from the
# previous run directly above "B-A caught", under a "PASS (5/5)" line. That is
# the self-contradicting-report class of AUDIT #22 (a summary listing 4 entries
# beside a verdict claiming 8), and the in-memory counters are what make the
# verdict correct, so only the EVIDENCE was wrong -- which is the worst place for
# it to be wrong.
: > "$LOG/summary.txt"
cp "$SCRIPT_DIR"/*.sh "$SB/scripts/"          # copies, NOT symlinks -- see header
# *** .py TOO, SINCE THE SUMMARY MOVED OUT OF THE HARNESS. *** The sandbox mirrored
# only `*.sh`, so when the summary became scripts/_bench_summary.py -- a real file,
# precisely so the failed-arm guard could be driven against fixtures with no GPU --
# the sandbox checker reported "summary missing" and would have failed the BASELINE
# and every state for a reason unrelated to any mutation. A sandbox that does not
# contain the files under test tests nothing.
( shopt -s nullglob; cp "$SCRIPT_DIR"/*.py "$SB/scripts/" )
TGT="$SB/scripts/check_bench_load_gate.sh"
ORIG_T="$SB/.orig_target"
ORIG_H="$SB/.orig_helper"
cp "$TARGET" "$ORIG_T"
cp "$SCRIPT_DIR/_gate_loadlog.sh" "$ORIG_H"
# *** RESTORE THE DIRECTORY, NOT A LIST OF FILES. ***
# Two rounds of this exact defect, and the second one is why the first fix was
# wrong. Round 1: restore() covered the target and the checker but not the HELPER,
# and B-E mutates the helper -- so B-E's damage leaked into B-F..B-H and B-H
# reported its anchor missing when the anchor was fine. Round 2, after adding a
# state that damages a SECOND harness: the helper was restored and
# bench_chat.sh was not, so B-F's damage leaked into B-I and B-I died with
# "anchor found 0 times" -- on an anchor that was present and correct in the real
# tree. Extending a hand-list once per new state guarantees the third round.
# A pristine copy of the whole sandbox scripts/ directory is snapshotted instead,
# so any file any state touches is covered without knowing which.
PRISTINE="$SB/.pristine_scripts"
cp -a "$SB/scripts" "$PRISTINE"
restore() { rm -rf "$SB/scripts"; cp -a "$PRISTINE" "$SB/scripts"
            TGT="$SB/scripts/check_bench_load_gate.sh"; }
run_check() { ( cd "$SB" && bash "$TGT" 2>&1 ); }
# The harness a state damages, as a sandbox path.
H="$SB/scripts/$(basename "$TARGET")"
# The latency-unit harness, for the states that target the widened derivation.
HC="$SB/scripts/bench_chat.sh"

echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

# *** ANCHOR PRE-FLIGHT, BEFORE THE BASELINE (AUDIT #17's rule, added in #28). ***
# B-J scored MISSED (fn) after AUDIT #28's rewrite deleted its anchor text: nine
# baselines had already executed before the drift surfaced. patch_literal refuses
# a wrong or ambiguous anchor, which is loud, but it is loud LATE. Every state's
# anchor is asserted here -- present, and occurring EXACTLY ONCE (once is not a
# nicety: an anchor matching twice means patch_literal would patch whichever comes
# first, which is a different state than the one described).
check_anchor() { # $1 file, $2 anchor text (multi-line, as one quoted arg)
  local n
  n="$(python3 -c 'import sys;print(open(sys.argv[1]).read().count(sys.argv[2]))' \
        "$1" "$2")"
  if [ "$n" != "1" ]; then
    echo "ANCHOR STALE in $(basename "$1"): found $n occurrence(s), want 1."
    echo "  Re-point the state; do NOT loosen the check -- a loosened anchor is how"
    echo "  a mutation silently stops mutating."
    printf '  first line: %s\n' "$(printf '%s' "$2" | head -1)"
    return 1
  fi
}
preflight_anchors() {
  local f
  for f in "$SB"/scripts/*.sh "$SB"/scripts/*.py; do :; done
  check_anchor "$SB/scripts/_bench_summary.py" \
    '        if not b:
            sig = "no_bucket_data_in_record"' || return 1
  check_anchor "$SB/scripts/_bench_summary.py" \
    '    if len(common) < 99:' || return 1
  check_anchor "$SB/scripts/_bench_summary.py" \
    '            miss = [k for k in REQUIRED if field(b, k) is None]' || return 1
  check_anchor "$SB/scripts/_bench_summary.py" \
    '        else:
            # Not a crash. Recorded so the refusal below can NAME the defect
            # instead of aborting on it.
            d.setdefault("MALFORMED", kv)' || return 1
  check_anchor "$SB/scripts/bench_reports_batch.sh" 'SUM_RC=$?' || return 1
  echo "  ok    every AUDIT-#28 anchor occurs exactly once in its sandbox file"
}
echo
echo "== anchor pre-flight =="
preflight_anchors || exit 1

echo
echo "== baseline =="
restore
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -8; exit 1
fi
n_ok="$(printf '%s\n' "$b" | grep -cE '^\s+ok' || true)"
[ "$n_ok" -ge 10 ] || { echo "BASELINE FAIL -- only $n_ok ok lines"; exit 1; }
echo "  ok    baseline passes with $n_ok ok lines"
echo "  ok    CONVERSE: the gated harnesses verify clean, so an always-red"
echo "        checker cannot score as caught"

caught=0; total=0
run_state() { # $1 id, $2 marker, $3 expected-RE (unique to this state), $4 why
  local id="$1" marker="$2" expect="$3" why="$4"
  total=$((total+1)); restore
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed (STATE fault)"
    echo "$id MISSED (fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  bash -n "$SB/scripts/$(basename "$TARGET")" 2>/dev/null || {
    echo "  $id MISSED -- harness left broken (STATE fault)"; restore; return 1; }
  local o r; o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still passed with the damage ($why)"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_present "$id property" "$o" "$expect"; then
    echo "$id MISSED (rejection marker absent)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-present ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states (each damages a HARNESS, not the checker) =="

# B-A: the harness stops routing through the shared gate -- the family reopening.
# The anchor is UNINDENTED on purpose. An earlier version of this state carried
# two leading spaces and matched 0 times, so B-A scored MISSED with the message
# "mutation fn failed" -- which reads as "the harness cannot be damaged" when it
# actually means "the anchor is wrong". patch_literal reports the count, so this
# is loud, but it cost a full matrix run. Same lesson as the AUDIT #17 anchor
# parser: confirm the anchor form mechanically, never by eye.
MUTATE_FN() {
  patch_literal "$H" '. "$SCRIPT_DIR/_gate_loadlog.sh"' ':   # B_A removed the shared gate' || return 1
  grep -q 'B_A' "$H"
}
run_state "B-A" 'B_A' 'NO -- it can publish a figure without the shared load gate' \
  "a harness no longer sources the one shared gate"

# B-B: the harness builds its own figure, bypassing loadlog_tps_field, which is
# the ONLY thing that withholds. This is the retracted-defect shape.
MUTATE_FN() {
  patch_literal "$H" '  field="$(loadlog_tps_field "${tps:-NA}")"' '  field="${tps:-NA}"   # B_B raw' || return 1
  grep -q 'B_B' "$H"
}
run_state "B-B" 'B_B' 'NO -- it emits a figure without consulting the gate that withholds' \
  "a harness publishes a figure without passing the gate"

# B-C: the criterion is redefined in a harness, so property (1) -- "does it source
# the helper" -- still says yes while the behaviour has a second home. The
# copy-drift family, caught at the DEFINITION rather than at the call.
MUTATE_FN() {
  patch_literal "$H" 'load1() { loadlog_load1; }' 'loadlog_tps_field() { printf "%s" "$1"; }   # B_C second home || return 1
load1() { loadlog_load1; }'
  grep -q 'B_C' "$H"
}
run_state "B-C" 'B_C' 'REDEFINES loadlog_arm/loadlog_tps_field' \
  "a harness redefines the criterion instead of using it"

# B-D: the warn-and-continue line comes back. All three original harnesses
# printed exactly this and then measured anyway.
#
# *** RE-ANCHORED, AND THAT WAS FORCED BY A REAL EDIT. *** This state used to patch
# the literal line the harnesses shipped with -- `echo "WARN: box never quieted
# ... The run continues so the"`. Removing the blocking wait (2026-10-02) deleted
# that line and the warn block with it, so the anchor became STALE and
# patch_literal (which requires EXACTLY ONE match) would have refused to run.
#
# A stale anchor is not cosmetic: a mutation that silently stops mutating reports a
# confident verdict about a mechanism that is fine (the AUDIT #3c / E-D lesson, and
# the same trap this repo records for mutate_fold_matrix.sh's apply_m4). Re-anchored
# onto the line that REPLACED it, which is the honest formulation of the same
# defect: the harness announces a busy box and then measures anyway, licensing a
# figure the per-arm gate would otherwise withhold. The replacement text is
# deliberately still a warning -- so the state's meaning did not have to change.
MUTATE_FN() {
  patch_literal "$H" '  echo "load1=$L is >= $QLOAD. NOT waiting -- running anyway."' \
    '  echo "WARN: box never quieted (load1=$L >= $QLOAD); results are variance, not signal."   # B_D' || return 1
  grep -q 'B_D' "$H"
}
run_state "B-D" 'B_D' 'still prints the warn-and-continue line' \
  "the warn-and-continue quiet gate restored"

# B-E: THE SHARPEST ONE. The helper can no longer withhold -- it always says
# USABLE. Every TEXT property still passes (the harness still sources it, still
# routes through loadlog_tps_field, does not redefine it, has no warn line), so
# only the behavioural probe can see it. This is the AUDIT #12 S4 shape and it is
# the always-red converse: a gate that can only say USABLE is how a load1-4.67 arm
# gets recorded as a figure.
MUTATE_FN() {
  patch_literal "$SB/scripts/_gate_loadlog.sh" \
    '  if [ "$_LL_USABLE" = 1 ]; then
    printf '"'"'%s'"'"' "$1"
  else
    printf '"'"'SKIPPED-CONTENDED'"'"'
  fi' \
    '  printf '"'"'%s'"'"' "$1"   # B_E never withholds' || return 1
  grep -q 'B_E' "$SB/scripts/_gate_loadlog.sh"
}
run_state "B-E" 'B_E' 'published a figure for a drifting arm' \
  "the helper stops withholding; only the behavioural probe can see it"

# B-F: the newly-wired LATENCY harness regresses to publishing unconditionally --
# i.e. exactly the defect it was wired to remove. Two hunks, because removing only
# the branch would leave the helper sourced and the state invisible, which is the
# AUDIT #17 mutual-masking shape. Marked on the harness NAME, which is unique to
# this state: B-A trips the same property on a different file, and a shared
# marker would let one state's catch be scored as another's.
MUTATE_FN() {
  # EVERY hunk is `|| return 1`. Without that the function's exit status is its
  # LAST command's, so a hunk that failed to apply was masked by the succeeding
  # grep and the state was scored CAUGHT on the surviving hunk alone -- B-F did
  # exactly that, and the checker caught the damage for the wrong reason. The
  # repo has hit this shape before ("an argument accepted and never forwarded");
  # patch_literal refusing to patch is only useful if the caller BELIEVES it.
  patch_literal "$HC" 'if loadlog_usable; then' 'if true; then   # B_F always publishes' || return 1
  patch_literal "$HC" '. "$SCRIPT_DIR/_gate_loadlog.sh"' ':   # B_F removed the shared gate' || return 1
  grep -q 'B_F' "$HC"
}
run_state "B-F" 'B_F' 'bench_chat\.sh' \
  "the latency harness stops gating its TTFT/TPOT/E2E figures"

# B-G: *** PROVES THE WIDENED DERIVATION CAN FAIL. *** A plant that matches ONLY
# the latency arm of the criterion -- TTFT/TPOT/E2E, and not one rate token. If
# the population still keyed on `tok/s|tokens in` this file would be INVISIBLE to
# the checker and the state could never pass, so B-G is the state that makes the
# widening load-bearing rather than decorative. It is also the audit's own lesson
# closing on itself: bench_chat.sh was missed by the original derivation for
# exactly this reason, so the miss is now the test.
MUTATE_FN() {
  cat > "$SB/scripts/bench_fake_latency.sh" <<'PLANT'
#!/usr/bin/env bash
# B_G plant: a wall-time harness in LATENCY units only -- no tok/s, no
# "tokens in". If the derivation still keyed on rates alone this file would be
# invisible, and this state could never pass.
echo "turn0: TTFT=1.2s TPOT=0.05s/tok E2E=3.4s toks=64"
PLANT
  grep -q 'TTFT' "$SB/scripts/bench_fake_latency.sh"
}
run_state "B-G" 'B_G' 'bench_fake_latency' \
  "an ungated latency-only harness is INVISIBLE to the derivation"

# B-H: the helper's fail-loud guard removed. Without it an unset verdict compared
# unequal to "1" and loadlog_field returned SKIPPED-CONTENDED -- the SAFE
# direction but a SILENT one, so a harness wired wrong withholds every figure
# forever and never says why. Every other property still passes: the helper is
# still sourced, still not redefined, still withholds a drifting arm correctly
# (loadlog_arm sets the verdict, so the guard is never consulted there). Only the
# no-arm probe can see this, which is why it is a separate state rather than
# folded into B-E.
MUTATE_FN() {
  patch_literal "$SB/scripts/_gate_loadlog.sh" \
    '  _ll_need _LL_USABLE || return 1
  if [ "$_LL_USABLE" = 1 ]; then' \
    '  if [ "${_LL_USABLE-}" = 1 ]; then   # B_H guard removed' || return 1
  grep -q 'B_H' "$SB/scripts/_gate_loadlog.sh"
}
run_state "B-H" 'B_H' 'did not fail loudly' \
  "the helper silently withholds instead of refusing when no arm has run"

# B-I: the gate is consulted BEFORE the arm. Every figure from the harness would be
# REFUSED at runtime by _ll_need -- a loud, total failure, which is why the
# invariant is checked rather than trusted. _ll_need itself is fixture-only
# coverage (no real run reaches it, which is the POINT of this property), so this
# state is the only thing standing between that fact and a production outage.
MUTATE_FN() {
  patch_literal "$HC" 'if loadlog_usable; then' 'if loadlog_usable; then   # B_I queried too early' || return 1
  # move the query above the arm: the verdict is unset at that point
  patch_literal "$HC" 'loadlog_arm chat_runs python3 - <<'"'"'EOF'"'"'' \
    'if loadlog_usable; then   # B_I queried BEFORE the arm
loadlog_arm chat_runs python3 - <<'"'"'EOF'"'"'' || return 1
  patch_literal "$HC" 'echo "  FIGURES USABLE (max_load1=$_LL_MAX < $INTFER_QUIET_LOAD)"
else' \
    'echo "  (gate moved above)"' || return 1
  grep -q 'B_I' "$HC"
}
run_state "B-I" 'B_I' 'GATE-BEFORE-ARM' \
  "the harness consults the load verdict before running its arm"

# B-J: THE FAILED-ARM GUARD IS REMOVED FROM THE SUMMARY. *** AN ARTIFACT-DEFECT
# STATE, NOT A CHECKER-WEAKENING ONE. *** The summary is correct by construction
# once written, so weakening the checker that tests it is UNOBSERVABLE -- which is
# AUDIT #21's wasted 0/4 attempt, and its rule: mutate the ARTIFACT and require the
# checker to REJECT it. The patch drops the clause that treats a record with no
# bucket data as a failure, so the OLD log shape (`BUCKETS=""`, no FAILED tag --
# exactly what the lost E2 batch left on disk) summarises cleanly again and the
# batch reports itself as a completed measurement.
MUTATE_FN() {
  # ANCHOR RE-POINTED (AUDIT #28). It read
  #   `if r.get("FAILED") or not (r.get("BUCKETS") or "").strip():`
  # which AUDIT #28's rewrite of the record parse deleted, so this state scored
  # MISSED (fn) -- the stale-anchor class of AUDIT #17, one level down. The rule
  # from there is explicit: NEVER loosen the check, RE-POINT the anchor, because a
  # loosened anchor is how a mutation silently stops mutating.
  patch_literal "$SB/scripts/_bench_summary.py" \
    '        if not b:
            sig = "no_bucket_data_in_record"' \
    '        # B_J untagged dead records are no longer diagnosed' || return 1
  grep -q 'B_J' "$SB/scripts/_bench_summary.py"
}
run_state "B-J" 'B_J' 'summary-dead-rc3' \
  "a batch whose arms produced no data is summarised as a result again"

# ---------------------------------------------------------------- AUDIT #28 states
# Four more defects were found by reading the code AUDIT #27 shipped, and each is
# measured rather than argued (the measurements are quoted in the state). They are
# ARTIFACT-defect states for the same reason as B-J: the summariser and the harness
# are correct after this commit, so weakening the checker is unobservable.

# B-K: THE VERDICT IS GATED ON THE BATCH'S ROUND COUNT AGAIN. Reverts the gate from
# the rounds the two arms hold IN COMMON to the batch-wide round count (the max
# over arms), which is what let a one-round delta be scored against a spread of
# 0.0 and print "CLEARS the spread ... every round agrees in sign: yes".
MUTATE_FN() {
  patch_literal "$SB/scripts/_bench_summary.py" \
    '    if len(common) < 99:' \
    '    if rounds < 2:   # B_K gates on the batch, not on the paired rounds' || return 1
  grep -q 'B_K' "$SB/scripts/_bench_summary.py"
}
run_state "B-K" 'B_K' 'summary-oneround-refused' \
  "a one-round delta is scored against a zero noise floor again"

# B-L: A PARTIALLY-WRITTEN RECORD IS DATA AGAIN. Drops the required-field test, so
# a record whose bucket line lost nk/nw/nr counts as a good round and its
# exact-counter columns print '-' -- the silent-dash table AUDIT #27 closed.
MUTATE_FN() {
  patch_literal "$SB/scripts/_bench_summary.py" \
    '            miss = [k for k in REQUIRED if field(b, k) is None]' \
    '            miss = []  # B_L a partial bucket line counts as a measurement' || return 1
  grep -q 'B_L' "$SB/scripts/_bench_summary.py"
}
run_state "B-L" 'B_L' 'summary-trunc-rc3' \
  "a partially-written record summarises into dashes again"

# B-M: THE READER CRASHES ON A MALFORMED RECORD INSTEAD OF NAMING IT. Reverts to
# splitting the whole record and building a dict, which raises ValueError on the
# first token without '='. MEASURED on the pre-fix file: traceback, no summary at
# all, exit 1 -- and the harness discarded the rc, so the batch exited 0.
MUTATE_FN() {
  patch_literal "$SB/scripts/_bench_summary.py" \
    '        else:
            # Not a crash. Recorded so the refusal below can NAME the defect
            # instead of aborting on it.
            d.setdefault("MALFORMED", kv)' \
    '        else:
            raise ValueError("B_M malformed record token: %s" % kv)   # B_M' || return 1
  grep -q 'B_M' "$SB/scripts/_bench_summary.py"
}
run_state "B-M" 'B_M' 'summary-malformed-rc3' \
  "one malformed record aborts the whole summary instead of naming itself"

# B-N: THE HARNESS DISCARDS THE SUMMARY'S VERDICT AGAIN. The harness captured the
# summariser's rc and refused on it; this drops both, which is how a summary that
# raised a traceback produced no table while the batch still exited 0.
MUTATE_FN() {
  patch_literal "$SB/scripts/bench_reports_batch.sh" \
    'SUM_RC=$?' \
    'SUM_RC=0   # B_N the summary'"'"'s verdict is discarded' || return 1
  bash -n "$SB/scripts/bench_reports_batch.sh" || return 1
  grep -q 'B_N' "$SB/scripts/bench_reports_batch.sh"
}
run_state "B-N" 'B_N' 'neither captures nor refuses' \
  "the harness no longer propagates the summary's refusal"

restore
echo
echo "== residue + restore integrity =="
git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null \
  && echo "  ok    the tracked tree was never modified" || { echo "  FAIL tracked modified"; exit 1; }
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "BENCH LOAD-GATE MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=bench_load_gate_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "BENCH LOAD-GATE MUTATE: FAIL ($caught/$total caught)"
exit 1
