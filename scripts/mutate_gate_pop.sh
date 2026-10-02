#!/usr/bin/env bash
# Mutation matrix for scripts/check_gate_pop.sh.
#
# *** WHICH AUDIT #21 MODEL APPLIES HERE, AND WHY IT IS NOT THE STUB ONE. ***
# mutate_gpu_recover.sh and mutate_fold_report_class.sh drive fixtures that are
# correct BY CONSTRUCTION, so weakening their CHECKERS was unobservable (0/4).
# check_gate_pop.sh is the opposite case: its five cases are synthetic and
# SELF-CONTAINED -- each builds its own tree and asserts the helper's answer --
# so removing a case removes a MEASUREMENT and is directly observable. That is
# the distinction, and getting it wrong is what cost mutate_gpu_recover.sh a
# wasted pass.
#
# Each state removes or inverts ONE case; the checker must then FAIL, either
# through its own ok-count baseline or through the case's own assertion.
#
# AUDIT #19 leak discipline, and it is proven by a hand-planted fixture rather
# than a self-proving state: the matrix plants a stamp in a directory no matrix
# owns and REQUIRES its own as-found assertion to reject it, before any state.
#
# Usage: scripts/mutate_gate_pop.sh
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

CHECK="$SCRIPT_DIR/check_gate_pop.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/gate_pop_mutate"
LOG="$STAMP/states"
SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_mgp.XXXXXX")"   # OUTSIDE $BUILD_DIR
mkdir -p "$LOG"; : > "$LOG/summary.txt"
trap 'rm -rf \"$SB\"' EXIT INT TERM

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
git -C "$ROOT" diff --quiet -- "$CHECK" 2>/dev/null || {
  echo "FAIL: $CHECK is dirty; this matrix edits a sandbox copy."; exit 2; }

# The checker resolves its helper as "$SCRIPT_DIR/_gate_pop.sh", and a
# sandboxed run has $SCRIPT_DIR == the sandbox -- so without this copy it found
# no helper at all and read "exported=0 local=0", i.e. a PHANTOM baseline
# failure caused by the sandbox rather than by the code.
cp "$SCRIPT_DIR/_gate_pop.sh" "$SB/_gate_pop.sh"
cp "$CHECK" "$SB/check_gate_pop.sh"; ORIG="$SB/.orig"; cp "$SB/check_gate_pop.sh" "$ORIG"
restore() { cp "$ORIG" "$SB/check_gate_pop.sh"; }
run_check() { "$SB/check_gate_pop.sh" 2>&1; }

# ---- anchors must occur EXACTLY ONCE before any state runs ----------------
check_anchor() { # $1 file, $2 exact text, $3 description
  local n
  n="$(grep -cF -- "$2" "$1" 2>/dev/null || true)"
  if [ "$n" != "1" ]; then
    echo "ANCHOR STALE: '$3' occurs $n time(s) in $(basename "$1"), want exactly 1"
    echo "  Re-point the state; do NOT loosen the check -- an anchor that matches"
    echo "  nothing proves nothing (AUDIT #17)."
    exit 1
  fi
  echo "  anchor ok  $3"
}

# ---- LEAK PROOF by hand-planting, NOT by a self-proving state -------------
echo "== stamp-population leak assertion =="
PROBE_DIR="gate_pop_matrix_leakprobe"
PROBE="$BUILD_DIR/$PROBE_DIR/last_pass.stamp"
mkdir -p "$BUILD_DIR/$PROBE_DIR"; echo fixture > "$PROBE"
probe_rc=0
INTFER_BUILD_DIR="$BUILD_DIR" pop_assert_no_leak "planted-by-hand" >/dev/null 2>&1 || probe_rc=1
rm -rf "$BUILD_DIR/$PROBE_DIR"
if [ "$probe_rc" -eq 0 ]; then
  echo "  FAIL the population assertion ACCEPTED a hand-planted fixture stamp"
  echo "RESULT: gate_pop_mutate FAIL (leak assertion is not falsifiable)"
  exit 1
fi
echo "  ok    converse: a hand-planted fixture IS rejected (can go red)"
echo "  ok    CONVERSE: on a clean population the same assertion stays silent"
INTFER_BUILD_DIR="$BUILD_DIR" pop_assert_no_leak "clean" || exit 1

echo
echo "== anchors (exactly once, before any state) =="
check_anchor "$SB/check_gate_pop.sh" 'ok    all four prefix forms recognised' "M-A prefix forms"
check_anchor "$SB/check_gate_pop.sh" 'ok    ok -- the assertion stays SILENT on a clean tree' "M-B converse"
check_anchor "$SB/check_gate_pop.sh" 'ok    rejected, and it named' "M-C foreign dir"
check_anchor "$SB/check_gate_pop.sh" 'ok    both shapes enumerate the same' "M-D variable shape"
check_anchor "$SB/check_gate_pop.sh" 'ok    refused (rc=' "M-E fail-closed"

echo
echo "== baseline =="
restore
b="$(run_check)"; brc=$?
[ "$brc" -eq 0 ] || { echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -6; exit 1; }
n_ok="$(printf '%s\n' "$b" | grep -c '^  ok ')"
[ "$n_ok" -ge 5 ] || { echo "BASELINE FAIL -- only $n_ok ok lines (want >=5)"; exit 1; }
echo "  ok    baseline passes with $n_ok ok lines"

caught=0; total=0
run_state() { # $1 id, $2 marker, $3 absent-RE
  local id="$1" marker="$2" expect="$3"
  total=$((total+1)); restore
  local before; before="$(cat "$SB/check_gate_pop.sh")"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed"; echo "$id MISSED (fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  cmp -s "$before" "$SB/check_gate_pop.sh" && {
    echo "  $id MISSED -- no-op"; echo "$id MISSED (no-op)" >> "$LOG/summary.txt"; restore; return 1; }
  grep -q -- "$marker" "$SB/check_gate_pop.sh" || {
    echo "  $id MISSED -- marker absent (stale anchor)"
    echo "$id MISSED (anchor)" >> "$LOG/summary.txt"; restore; return 1; }
  bash -n "$SB/check_gate_pop.sh" 2>/dev/null || {
    echo "  $id MISSED -- patch left the checker broken (STATE fault)"
    echo "$id MISSED (syntax)" >> "$LOG/summary.txt"; restore; return 1; }
  local o r; o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still exited 0 with the measurement removed"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_absent "$id ok-line" "$o" "$expect"; then
    echo "$id MISSED (polarity or marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-absent, case dropped and reported"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states =="

# *** patch_literal takes THREE arguments: file, exact-old, exact-new. ***
# The first version of this matrix passed FIVE -- one quoted string per line --
# because each quoted string is a separate ARGUMENT, not a concatenation. It
# therefore replaced the echo line with the counter increment, never wrote the
# marker, and all five states reported "PATCH VERIFY FAIL: NOT present" while
# patch_literal itself succeeded. Found with `bash -x`, not by guessing.
# A single-quoted bash string MAY span lines, so old and new each carry the
# WHOLE case as one argument. That is the form below.
#
# Each state removes ONE case: its echo line and its own counter increment.
# The counters are n_a..n_e precisely so each removal has a UNIQUE anchor; a
# shared counter cannot be targeted (five identical increments).

# M-A: the prefix-form case -- the FALSE LEAK this helper shipped ($PWD/build assumed
# where the matrices write $ROOT/build)
MUTATE_FN() {
  patch_literal "$SB/check_gate_pop.sh" \
    'echo "  ok    all four prefix forms recognised: $got"
  n_a=$((n_a+1))' \
    ': # MUT_A -- this case is no longer measured
  n_a=$((n_a))' \
  && verify_patched "$SB/check_gate_pop.sh" 'MUT_A'
}
run_state "M-A" 'MUT_A' 'ok    all four prefix forms recognised'

# M-B: the CONVERSE -- without it a checker that rejects everything passes the rest
MUTATE_FN() {
  patch_literal "$SB/check_gate_pop.sh" \
    'echo "  ok    ok -- the assertion stays SILENT on a clean tree"
  n_b=$((n_b+1))' \
    ': # MUT_B -- this case is no longer measured
  n_b=$((n_b))' \
  && verify_patched "$SB/check_gate_pop.sh" 'MUT_B'
}
run_state "M-B" 'MUT_B' 'stays SILENT on a clean tree'

# M-C: the can-go-red direction -- without it a stub always returning 0 would pass
MUTATE_FN() {
  patch_literal "$SB/check_gate_pop.sh" \
    'echo "  ok    rejected, and it named fixture_leak"
  n_c=$((n_c+1))' \
    ': # MUT_C -- this case is no longer measured
  n_c=$((n_c))' \
  && verify_patched "$SB/check_gate_pop.sh" 'MUT_C'
}
run_state "M-C" 'MUT_C' 'named fixture_leak'

# M-D: the exported-vs-local variable shape -- the defect that made an assertion
# report itself vacuous
MUTATE_FN() {
  patch_literal "$SB/check_gate_pop.sh" \
    'echo "  ok    both shapes enumerate the same $r1 dir(s)"
  n_d=$((n_d+1))' \
    ': # MUT_D -- this case is no longer measured
  n_d=$((n_d))' \
  && verify_patched "$SB/check_gate_pop.sh" 'MUT_D'
}
run_state "M-D" 'MUT_D' 'both shapes enumerate the same'

# M-E: the fail-closed case -- the third defect, an empty path enumerated at rc=0
MUTATE_FN() {
  patch_literal "$SB/check_gate_pop.sh" \
    'echo "  ok    refused (rc=$rc) instead of reporting an empty population as ok"
  n_e=$((n_e+1))' \
    ': # MUT_E -- this case is no longer measured
  n_e=$((n_e))' \
  && verify_patched "$SB/check_gate_pop.sh" 'MUT_E'
}
run_state "M-E" 'MUT_E' 'refused \(rc='
restore
echo
echo "== residue + restore integrity =="
if residue_scan "$SB" MUT_A MUT_B MUT_C MUT_D MUT_E; then
  echo "  ok    no mutation marker survived in the sandbox"
else
  echo "  FAIL a mutation marker survived"; exit 1
fi
if cmp -s "$ORIG" "$SB/check_gate_pop.sh"; then
  echo "  ok    sandbox copy is byte-identical to the pristine source"
else
  echo "  FAIL sandbox copy was left mutated"; exit 1
fi
# The REAL source must be untouched, byte for byte.
if git -C "$ROOT" diff --quiet -- "$CHECK" 2>/dev/null; then
  echo "  ok    the tracked checker was never modified"
else
  echo "  FAIL the tracked checker was modified"; exit 1
fi

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "GATE POP MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=gate_pop_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "GATE POP MUTATE: FAIL ($caught/$total caught)"
exit 1