#!/usr/bin/env bash
# Mutation matrix for scripts/check_fold_report_class.sh -- the newest checker in
# the tree, and (at the time of writing) the only one with no matrix.
#
# *** AUDIT #21 APPLIES DIRECTLY, AND CHANGES THE MODEL. *** Every artifact this
# checker drives is CORRECT BY CONSTRUCTION: it synthesises the summary contents
# and stamp sha itself. So checker-weakening states -- drop an assertion, replace
# a condition with `if true` -- would score MISSED, because the checker would
# still be right about a correct input. That is the 0/4 lesson from
# mutate_gpu_recover.sh's first version.
#
# So each state here breaks the EXPECTED DIRECTION rather than the checker: it
# turns a case into a DIFFERENT case, so a checker that has quietly stopped
# testing a direction must fail. Concretely: expect rc=1 for a damaged summary
# on CURRENT evidence, and the state makes the stamp stale -- if the checker
# returns 2 there, it is laundering a real regression.
#
# Fixtures are written into the real build tree and restored in a trap; the
# stamp-population assertion compares the SET of stamp directories, not a count,
# because mutate_device_precond.sh writes two stamps into one directory.
#
# Usage: scripts/mutate_fold_report_class.sh
# Exit: 0 = every state caught, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_patch.sh
. "$SCRIPT_DIR/_gate_patch.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"

CHECK="$SCRIPT_DIR/check_fold_report_class.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP_DIR="$BUILD_DIR/fold_matrix"
STAMP="$BUILD_DIR/fold_report_class_mutate"
LOG="$STAMP/states"
SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_mfrc.XXXXXX")"   # OUTSIDE $BUILD_DIR
mkdir -p "$LOG"; : > "$LOG/summary.txt"
SB_SUM="$(mktemp)"; SB_STAMP="$(mktemp)"
cleanup() { cp "$SB_SUM" "$STAMP_DIR/summary.txt" 2>/dev/null
            cp "$SB_STAMP" "$STAMP_DIR/last_pass.stamp" 2>/dev/null
            rm -rf "$SB" "$SB_SUM" "$SB_STAMP"; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
[ -d "$STAMP_DIR" ] || { echo "SKIP: $STAMP_DIR absent; run mutate_fold_matrix.sh first"; exit 2; }
git -C "$ROOT" diff --quiet -- "$CHECK" 2>/dev/null || {
  echo "FAIL: $CHECK is dirty; this matrix edits a sandbox copy."; exit 2; }
cp "$STAMP_DIR/summary.txt" "$SB_SUM"; cp "$STAMP_DIR/last_pass.stamp" "$SB_STAMP"

# The sandbox must LOOK like the repo to the scripts it runs: they resolve
# ROOT from their own location and shell out to `git rev-parse HEAD`. Without a
# .git and the real build tree linked in, every case read STALE and the baseline
# reported 2 directions failing -- an artefact of the sandbox, not of the checker.
ln -sfn "$ROOT/.git" "$SB/.git"
ln -sfn "$BUILD_DIR" "$SB/build"
# The checker resolves its helper sibling through $SCRIPT_DIR, so link it in.
mkdir -p "$SB/scripts"
for f in "$CHECK" "$SCRIPT_DIR/fold_report_counts.sh" "$SCRIPT_DIR/_gate_pop.sh" \
         "$SCRIPT_DIR/_gate_pred.sh" "$SCRIPT_DIR/_gate_patch.sh" "$SCRIPT_DIR/_gate_enum.sh"; do
  [ -e "$f" ] && cp "$f" "$SB/scripts/"
done
ORIG="$SB/.orig"; cp "$SB/scripts/check_fold_report_class.sh" "$ORIG"
restore() { cp "$ORIG" "$SB/scripts/check_fold_report_class.sh"; }

echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

echo "== baseline: the checker must PASS and print 5 directions =="
restore
b="$(INTFER_BUILD_DIR="$BUILD_DIR" bash "$SB/scripts/check_fold_report_class.sh" 2>&1)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -8; exit 1
fi
n_ok="$(printf '%s\n' "$b" | grep -c '^  ok ')"
if [ "$n_ok" -lt 5 ]; then
  echo "BASELINE FAIL -- only $n_ok of 5 directions green"; exit 1
fi
echo "  ok    baseline passes with $n_ok/5 directions"
echo "  ok    CONVERSE: a healthy run asserts every direction, so a dropped"
echo "        direction below shows up as a missing ok line, not a silent pass"

caught=0; total=0
run_state() { # $1 id, $2 marker, $3 absent-RE
  local id="$1" marker="$2" expect="$3"
  total=$((total+1)); restore
  local before; before="$(cat "$SB/scripts/check_fold_report_class.sh")"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed"; echo "$id MISSED (fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  [ "$before" = "$(cat "$SB/scripts/check_fold_report_class.sh")" ] && {
    echo "  $id MISSED -- mutation did not change the file"
    echo "$id MISSED (no-op)" >> "$LOG/summary.txt"; restore; return 1; }
  grep -q -- "$marker" "$SB/scripts/check_fold_report_class.sh" || {
    echo "  $id MISSED -- marker absent (stale anchor)"
    echo "$id MISSED (anchor)" >> "$LOG/summary.txt"; restore; return 1; }
  bash -n "$SB/scripts/check_fold_report_class.sh" 2>/dev/null || {
    echo "  $id MISSED -- patch left the checker syntactically broken (STATE fault)"
    echo "$id MISSED (syntax)" >> "$LOG/summary.txt"; restore; return 1; }
  local o r
  o="$(INTFER_BUILD_DIR="$BUILD_DIR" bash "$SB/scripts/check_fold_report_class.sh" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore
  # POLARITY NAMED (AUDIT #20): the dropped direction's ok line must be ABSENT.
  if ! pred_expect_absent "$id direction" "$o" "$expect"; then
    echo "$id MISSED (polarity or marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- verdict-polarity=expect-absent, direction dropped and reported"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states (each removes one EXPECTED DIRECTION) =="

# M-A: the "damaged summary + CURRENT stamp -> 1 (a real regression)" case
# deleted. If the checker stops asserting the case where rc=1 is REQUIRED, it
# can start laundering a real regression as stale evidence and still look green.
MUTATE_FN() {
  patch_literal "$SB/scripts/check_fold_report_class.sh" \
    'expect "damaged summary + current stamp = REAL regression" 1 "$(run_rc)"' \
    ': # MUT_A -- the real-regression direction is no longer asserted' \
    && verify_patched "$SB/scripts/check_fold_report_class.sh" 'MUT_A'
}
run_state "M-A" 'MUT_A' 'ok    damaged summary \+ current stamp = REAL regression -> rc=1'

# M-B: the stale/damaged direction dropped -- the one that WAS misclassified.
MUTATE_FN() {
  patch_literal "$SB/scripts/check_fold_report_class.sh" \
    'expect "damaged summary + STALE stamp (was rc=1)"     2 "$(run_rc)"' \
    ': # MUT_B -- the stale-evidence direction is no longer asserted' \
    && verify_patched "$SB/scripts/check_fold_report_class.sh" 'MUT_B'
}
run_state "M-B" 'MUT_B' 'ok    damaged summary \+ STALE stamp \(was rc=1\) -> rc=2'

# M-C: the "no stamp at all" direction dropped.
MUTATE_FN() {
  patch_literal "$SB/scripts/check_fold_report_class.sh" \
    'expect "no stamp at all"                              2 "$(run_rc)"' \
    ': # MUT_C -- the missing-stamp direction is no longer asserted' \
    && verify_patched "$SB/scripts/check_fold_report_class.sh" 'MUT_C'
}
run_state "M-C" 'MUT_C' 'ok    no stamp at all -> rc=2'

# M-D: the clean-tree direction dropped. This is the CONVERSE, and the one that
# gets forgotten: without it, a checker that only ever expects rc=2 would score
# M-A/M-B/M-C "correctly" while never confirming the healthy case passes.
MUTATE_FN() {
  patch_literal "$SB/scripts/check_fold_report_class.sh" \
    'expect "clean summary + current stamp" 0 "$(run_rc)"' \
    ': # MUT_D -- the healthy direction is no longer asserted' \
    && verify_patched "$SB/scripts/check_fold_report_class.sh" 'MUT_D'
}
run_state "M-D" 'MUT_D' 'ok    clean summary \+ current stamp -> rc=0'

restore
echo
echo "== stamp-population leak assertion (after states) =="
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "FOLD REPORT CLASS MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=fold_report_class_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "FOLD REPORT CLASS MUTATE: FAIL ($caught/$total caught)"
exit 1