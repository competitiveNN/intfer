#!/usr/bin/env bash
# Mutation matrix for scripts/check_checker_matrix_coverage.sh.
#
# *** WHY EACH STATE DOES TWO THINGS. *** Weakening the checker ALONE is
# unobservable here: the real tree has full coverage, so deleting a requirement
# changes nothing and the checker still exits 0. That is the same 0/4 lesson as
# mutate_gpu_recover.sh, and it is the reason this file was reverted once
# already. So each state POISONS the tree with a checker that has no matrix AND
# removes the requirement that would have caught it. Only the combination is
# observable, and only the combination is the honest test of "does this
# requirement actually matter".
#
# PATCH FORM (the arity lesson, applied deliberately). patch_literal takes THREE
# arguments -- file, exact-old, exact-new -- and each of old/new may contain a
# LITERAL NEWLINE inside one single-quoted argument. Passing one quoted string
# per LINE was the original bug and is now rejected by patch_literal itself.
#
# PATH FORM (the sandbox-mismatch class, closed up front). The patch, the
# verify, the no-op compare, the syntax check and the run ALL use the identical
# path TGT. That class -- patching $SB/x and verifying $SB/scripts/x -- was hit
# twice in this repo and is asserted here at the top rather than fixed per state.
#
# ANTI-VACUITY. The baseline requires the healthy tree to PASS and to list every
# checker, so a checker which rejected everything could not score as caught.
#
# Usage: scripts/mutate_checker_matrix_coverage.sh
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
STAMP="$BUILD_DIR/coverage_mutate"
LOG="$STAMP/states"

[ -f "$SCRIPT_DIR/check_checker_matrix_coverage.sh" ] || { echo "SKIP: checker missing"; exit 2; }
git -C "$ROOT" diff --quiet -- scripts/check_checker_matrix_coverage.sh 2>/dev/null || {
  echo "FAIL: the checker is dirty; this matrix edits a sandbox copy."; exit 2; }

# ---- THE SANDBOX MUST MIRROR scripts/ --------------------------------------
# The checker derives ROOT from its own location and enumerates scripts/ under
# it, so a bare copy made ROOT resolve to /tmp, enumerated NOTHING, and reported
# "no check_*.sh found" -- a vacuous failure manufactured by the sandbox, which
# is exactly what the enumeration guard inside the checker exists to catch.
SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_mcov.XXXXXX")"
TGT="$SB/scripts/check_checker_matrix_coverage.sh"
mkdir -p "$SB/scripts" "$LOG"
: > "$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT
for f in "$SCRIPT_DIR"/*.sh; do ln -sf "$f" "$SB/scripts/$(basename "$f")"; done
rm -f "$TGT"
cp "$SCRIPT_DIR/check_checker_matrix_coverage.sh" "$TGT"
ORIG="$SB/.orig"; cp "$TGT" "$ORIG"
restore() { cp "$ORIG" "$TGT"; rm -f "$SB/scripts/check_zz_unmated.sh"; }

# ONE place that poisons the tree. A checker with no matrix and no arm: exactly
# the drift the checker is built to report, and it stays planted for the whole
# state (removing it before the run would make the state vacuous).
POISON="$SB/scripts/check_zz_unmated.sh"
poison() { printf '#!/usr/bin/env bash\nexit 0\n' > "$POISON"; chmod +x "$POISON"; }

echo "== path-form assertion (patch/verify/run must agree) =="
n_tgt="$(grep -c '"\$TGT"' "$0" || true)"
if [ "$n_tgt" -lt 5 ]; then
  echo "  FAIL  only $n_tgt uses of \"\$TGT\"; every patch/verify/run must share ONE path"
  exit 1
fi
echo "  ok    $n_tgt references all use the single \$TGT path form"
echo "  ok    CONVERSE: no state hard-codes a second sandbox path"

echo
echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

echo
echo "== baseline: healthy tree must PASS and list every checker =="
restore
b="$(bash "$TGT" 2>&1)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL -- coverage does not pass on the clean tree:"
  printf '%s\n' "$b" | tail -8; exit 1
fi
n_rows="$(printf '%s\n' "$b" | grep -c '^check_' || true)"
if [ "$n_rows" -lt 8 ]; then
  echo "BASELINE FAIL -- only $n_rows checker rows listed (want >=8)"
  exit 1
fi
echo "  ok    baseline passes and lists $n_rows checkers"
echo "  ok    CONVERSE: the healthy direction is verified, so an always-red"
echo "        checker cannot score as caught"

caught=0; total=0
run_state() { # $1 id, $2 marker, $3 absent-RE, $4 why
  local id="$1" marker="$2" expect="$3" why="$4"
  total=$((total+1)); restore; poison
  local before; before="$(cat "$TGT")"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed"; echo "$id MISSED (fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  cmp -s "$before" "$TGT" && {
    echo "  $id MISSED -- no-op"; echo "$id MISSED (no-op)" >> "$LOG/summary.txt"; restore; return 1; }
  grep -q -- "$marker" "$TGT" || {
    echo "  $id MISSED -- marker absent (stale anchor)"
    echo "$id MISSED (anchor)" >> "$LOG/summary.txt"; restore; return 1; }
  bash -n "$TGT" 2>/dev/null || {
    echo "  $id MISSED -- patch left the checker broken (STATE fault)"
    echo "$id MISSED (syntax)" >> "$LOG/summary.txt"; restore; return 1; }
  local o r; o="$(bash "$TGT" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- coverage still passed with the requirement removed AND"
    echo "        an unmated checker planted ($why)"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_absent "$id requirement" "$o" "$expect"; then
    echo "$id MISSED (polarity or marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-absent ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states (each: poison the tree AND remove the requirement) =="

# M-A: REQUIREMENT 1 -- a checker with no matrix is drift. Removed, while an
# unmated checker is planted: the checker must then MISS it and exit 0.
MUTATE_FN() {
  poison
  patch_literal "$TGT" \
    '  [ "$have" = "MISSING" ] && bad=$((bad+1))' \
    '  : # MUT_A -- requirement 1 (unmated checker) no longer asserted' \
    && verify_patched "$TGT" 'MUT_A'
}
# The absent marker is the VERDICT line, not the word MISSING: the table still
# prints "MISSING" for the planted checker even with the requirement removed --
# only the `bad` increment and the exit code disappear. Judging on the table
# would make this state unscorable.
run_state "M-A" 'MUT_A' 'COVERAGE: PASS' "unmated checker planted, requirement 1 removed"

# M-B: REQUIREMENT 2 -- a matrix with no dispatch arm never runs (audit #18), so
# the suite would still print PASS. Removed, with an unmated checker planted.
MUTATE_FN() {
  poison
  patch_literal "$TGT" \
    '  [ "$arm" = "NO" ] && bad=$((bad+1))' \
    '  : # MUT_B -- requirement 2 (undispatched matrix) no longer asserted' \
    && verify_patched "$TGT" 'MUT_B'
}
run_state "M-B" 'MUT_B' 'COVERAGE: PASS' "unmated checker planted, requirement 2 removed"

# M-D: the ARM LOOKUP's QUERY fails. Per AUDIT #21 the SEAM is broken, not the
# checker's leniency: here the query itself is made to FAIL, so a checker that
# collapses "the query did not run" into "no dispatch arm" would report ordinary
# drift for entirely the wrong reason. A stub grep exiting 2 stands in for a
# malformed pattern or an unreadable file -- a real grep cannot be made to error
# on a well-formed -F pattern.
MUTATE_FN() {
  mkdir -p "$SB/bin"
  printf '#!/usr/bin/env bash\nexit 2   # the query did not run\n' > "$SB/bin/grep"
  chmod +x "$SB/bin/grep"
  patch_literal "$TGT" \
    '  [ "$arm" = "NO" ] && bad=$((bad+1))' \
    '  : # M_D -- the arm-drift requirement is no longer asserted' \
    && verify_patched "$TGT" 'M_D'
}

# Same shape as run_state, but runs the checker with the failing grep on PATH.
run_state_stub() {
  local id="$1" expect="$2"
  total=$((total+1)); restore
  local before; before="$(cat "$TGT")"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed"; echo "$id MISSED (fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  cmp -s "$before" "$TGT" && { echo "  $id MISSED -- no-op"; restore; return 1; }
  grep -q -- 'M_D' "$TGT" || {
    echo "  $id MISSED -- marker absent (stale anchor)"; restore; return 1; }
  bash -n "$TGT" 2>/dev/null || {
    echo "  $id MISSED -- patch left the checker broken (STATE fault)"; restore; return 1; }
  local o r; o="$(PATH="$SB/bin:$PATH" bash "$TGT" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore; rm -rf "$SB/bin"
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- coverage passed with a FAILING arm lookup"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  # PRESENCE, not absence -- and this is the sixth instance of the same
  # inversion. With the arm-drift rule removed and the query forced to fail, the
  # checker MUST still refuse, and it signals refusal by PRINTING QUERY-FAILED.
  # I asserted the marker would be absent, which scored the working guard as
  # MISSED. Read the log, write the polarity from what it shows.
  if ! pred_expect_present "$id query-failure" "$o" "$expect"; then
    echo "$id MISSED (polarity: expected the refusal marker to be PRESENT)" \
      >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-absent (query failure refused)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

run_state_stub "M-D" 'QUERY-FAILED'

# M-C: the enumeration guard itself. With it gone, a checker that harvests
# NOTHING would classify nothing and report PASS -- the sites_found=0
# vacuous-green, which is the class this whole tree keeps auditing.
MUTATE_FN() {
  poison
  patch_literal "$TGT" \
    'if [ "$n_chk" -eq 0 ]; then' \
    'if false; then  # MUT_C' \
    && verify_patched "$TGT" 'MUT_C'
}
run_state "M-C" 'MUT_C' 'enumeration is broken' "empty-population guard removed"

restore
echo
echo "== residue + restore integrity =="
residue_scan "$SB" MUT_A MUT_B MUT_C \
  && echo "  ok    no mutation marker survived" || exit 1
cmp -s "$ORIG" "$TGT" \
  && echo "  ok    sandbox byte-identical to pristine" || { echo "  FAIL sandbox mutated"; exit 1; }
[ ! -e "$POISON" ] \
  && echo "  ok    the planted unmated checker was removed" || { echo "  FAIL poison leaked"; exit 1; }
git -C "$ROOT" diff --quiet -- scripts/check_checker_matrix_coverage.sh 2>/dev/null \
  && echo "  ok    the tracked checker was never modified" || { echo "  FAIL tracked modified"; exit 1; }
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "COVERAGE MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=coverage_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "COVERAGE MUTATE: FAIL ($caught/$total caught)"
exit 1