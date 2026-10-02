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
MUTATE_FN() {
  patch_literal "$H" '  echo "WARN: box never quieted (load1=$L >= $QLOAD). The run continues so the"' \
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
