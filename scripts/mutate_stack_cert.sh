#!/usr/bin/env bash
# Mutation matrix for scripts/check_stack_cert.sh (AUDIT #29).
#
# WHAT THIS IS FOR. The checker has a seven-case SELF-CHECK, and those cases all
# pass with the checker BROKEN in a specific way: every fixture derives its census
# from the SAME expression the verdict uses, so a wrong expression satisfies both.
# That is the AUDIT #12 S4 shape -- a contract that reasons about its own inputs
# and is structurally blind to how it obtains them -- and it is invisible to a
# self-check by construction. So this matrix breaks the DERIVATION and the
# TREE-TEST, which the self-check cannot see.
#
#   S-A  the census derivation drops check_* and counts only mutate_* -- invisible
#        to every self-check case (they compare a fixture against the same wrong
#        number), so the live verdict is the only possible witness.
#   S-B  the dirty-tree test is neutered (`git diff --quiet || true`) and a sandbox
#        repo with a MODIFIED tracked file is planted. Needs INTFER_STACK_CERT_ROOT,
#        without which the branch judges the real tree and can never be exercised.
#   S-C  the census comparison is widened so a mismatched count is accepted.
#
# *** ARTIFACT-DEFECT STATES, PER AUDIT #21. *** The artifact is the TREE and the
# certificate, so each state plants one; the checker is only redirected, never
# weakened. And the converse is required: the baseline must show a GOOD tree
# certified (rc=0), or every state below is also satisfied by a checker that
# refuses everything.
#
# Usage: scripts/mutate_stack_cert.sh
# Exit: 0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/stack_cert_mutate"
LOG="$STAMP/states"
CHECK="$SCRIPT_DIR/check_stack_cert.sh"
[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
mkdir -p "$LOG"
: > "$LOG/summary.txt"

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_scmut.XXXXXX")"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM
REPO="$SB/repo"
BD="$SB/build"
mkdir -p "$REPO/scripts" "$BD/stack_cert"
cp "$SCRIPT_DIR"/*.sh "$REPO/scripts/" 2>/dev/null
git init -q "$REPO" || { echo "FAIL: could not init sandbox repo"; exit 2; }
git -C "$REPO" -c user.email=m@x -c user.name=matrix add -A >/dev/null 2>&1
git -C "$REPO" -c user.email=m@x -c user.name=matrix commit -q -m sandbox >/dev/null 2>&1
TARGET="$REPO/scripts/check_stack_cert.sh"

# The census the fixture cert must claim, derived from the SANDBOX (the nested run
# grades the sandbox, so the numbers have to be the sandbox's).
ns=$(find "$REPO/scripts" -maxdepth 1 \( -name 'check_*.sh' -o -name 'mutate_*.sh' \) -type f | wc -l | tr -d ' ')
nc=$(find "$REPO/scripts" -maxdepth 1 -name 'check_*.sh' -type f | wc -l | tr -d ' ')
nm=$(find "$REPO/scripts" -maxdepth 1 -name 'mutate_*.sh' -type f | wc -l | tr -d ' ')
mkdir -p "$BD/m1" "$BD/m2"
echo "sha=deadbeef" > "$BD/m1/last_pass.stamp"
echo "sha=deadbeef" > "$BD/m2/last_pass.stamp"

good_cert() { # writes a cert that matches the sandbox HEAD and the sandbox census
  local h; h="$(git -C "$REPO" rev-parse HEAD)"
  { echo "stamp=stack_cert"; echo "sha=$h"; echo "short=${h:0:7}"; echo "rc=0"
    echo "verdict=PASS"; echo "failed=0"; echo "stale=0"; echo "unproven=0"
    echo "stamps=2"; echo "scripts=$ns"; echo "checks=$nc"; echo "matrices=$nm"
    echo "stamp_dirs=2"; echo "quick=0"; echo "date=1970-01-01T00:00:00Z"
  } > "$BD/stack_cert/cert"
}
# The nested run must not recurse, and must be told the build tree to grade.
run_check() {
  INTFER_BUILD_DIR="$BD" INTFER_STACK_CERT_ROOT="$REPO" \
    INTFER_STACK_CERT_NO_SELFCHECK=1 \
    bash "$TARGET" >"$WORK_OUT" 2>&1
}

echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

WORK_OUT="$SB/out.log"
echo
echo "== baseline: a GOOD tree must CERTIFY (the converse half) =="
good_cert
run_check; brc=$?
if [ "$brc" -eq 0 ] && grep -q 'STACK CERT CHECK: PASS' "$WORK_OUT"; then
  echo "  ok    baseline: a matching rc=0 cert on a clean tree is CERTIFIED (rc=0)"
  nb=1
else
  echo "FAIL: baseline rc=$brc -- every firing state below would then pass for the" >&2
  echo "  wrong reason (a checker that refuses everything)." >&2
  sed 's/^/        /' "$WORK_OUT" >&2; exit 1
fi

reset_tree() { git -C "$REPO" checkout -q -- . ; }
caught=0; total=0
run_state() { # $1 id, $2 desc, $3 expected-RE, $4 why
  local id="$1" desc="$2" expect="$3" why="$4"
  total=$((total + 1))
  reset_tree
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed (STATE fault)"
    echo "$id MISSED (fn)" >> "$LOG/summary.txt"; reset_tree; return 1
  fi
  bash -n "$TARGET" 2>/dev/null || {
    echo "  $id MISSED -- the checker was left syntactically broken (STATE fault)"
    reset_tree; return 1; }
  run_check; local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  $id MISSED -- the checker still passed with the damage ($why)"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; reset_tree; return 1
  fi
  if ! printf '%s' "$(cat "$WORK_OUT")" | grep -qE "$expect"; then
    echo "  $id MISSED -- it failed, but NOT for its own reason (unattributed)"
    sed 's/^/        /' "$WORK_OUT"
    echo "$id MISSED (unattributed)" >> "$LOG/summary.txt"; reset_tree; return 1
  fi
  echo "  $id caught -- rc=$rc, marker matched: $desc"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught + 1))
  reset_tree; return 0
}

echo
echo "== states (each damages the DERIVATION or the TREE-TEST, never the verdicts) =="

# S-A: the census derivation drops check_* files. INVISIBLE to every self-check
# case, because each one compares a fixture against the same wrong number.
MUTATE_FN() {
  sed -i "s|-name 'check_\*\.sh' -o -name 'mutate_\*\.sh'|-name 'mutate_*.sh'|" "$TARGET" \
    || return 1
  grep -q "n_scripts=\$(find" "$TARGET"
}
run_state "S-A" "the census stops counting check_*.sh" 'the census is stale' \
  "a census that only counts matrices is invisible to the self-check"

# S-B: THE DIRTY-TREE GUARD REMOVED. *** IT CANNOT USE run_state, AND THE REASON IS
# THE INVERTED-POLARITY LESSON (measured, not argued). *** run_state judges a state
# by "rc != 0", which is right for every other state HERE -- and exactly wrong for
# this one. Neutering the guard turns the checker from REJECTING a dirty tree into
# ACCEPTING it, i.e. rc 1 -> rc 0: the mutation makes it exit ZERO where it should
# exit non-zero. A `rc != 0` predicate can never see that, and the first run of
# this matrix reported S-B MISSED (rc=0) for a state whose defect had reproduced
# perfectly. This is the same trap as the three recorded inverted predicates, in the
# opposite direction: there the marker was scored as absent when it was present,
# here success was scored as failure.
#
# So BOTH halves are asserted here, and the first one is the case's real proof:
#   (1) unmutated + DIRTY tree   -> rc=1 and it says DIRTY. If this does not hold,
#       the CASE is broken, not the mutation, and the matrix stops.
#   (2) guard removed + DIRTY    -> rc=0 and it says PASS. The defect reproduced.
# The planting is the ROOT seam's only use: without it this branch judges the REAL
# tree and can never be exercised at all.
total=$((total + 1))
reset_tree
printf 'dirty\n' >> "$REPO/scripts/mutate_result_block.sh"
run_check; s_b_base=$?
if [ "$s_b_base" -ne 1 ] || ! grep -q 'DIRTY' "$WORK_OUT"; then
  echo "  S-B MISSED (case) -- a DIRTY tree was not refused (rc=$s_b_base). The" >&2
  echo "        guard under test does not reject its subject, so removing it" >&2
  echo "        would change nothing observable." >&2
  sed 's/^/        /' "$WORK_OUT" >&2; exit 1
fi
echo "  ok    S-B case: the UNMUTATED checker refuses a dirty tree (rc=1, DIRTY)"
sed -i 's|if git diff --quiet \&\& git diff --cached --quiet; then|if true; then   # S_B dirty test removed|' "$TARGET" \
  || { echo "S-B MISSED (fn)" >&2; exit 1; }
run_check; s_b_rc=$?
if [ "$s_b_rc" -eq 0 ] && grep -q 'STACK CERT CHECK: PASS' "$WORK_OUT"; then
  caught=$((caught + 1))
  echo "  S-B caught -- rc=0 where it must be non-zero: a DIRTY tree is certified"
  echo "S-B caught" >> "$LOG/summary.txt"
else
  echo "  S-B MISSED -- rc=$s_b_rc; the guard removal was not observable"
  echo "S-B MISSED" >> "$LOG/summary.txt"
fi
reset_tree

# S-C: the census comparison widened, so a mismatched count is accepted.
MUTATE_FN() {
  sed -i 's|if \[ "\$claimed" != "\$live" \]; then|if [ "$claimed" != "ALWAYS_MISMATCH" ]; then   # S_C|' "$TARGET" \
    || return 1
  grep -q 'S_C' "$TARGET"
}
run_state "S-C" "a mismatched census is accepted" 'the census is stale' \
  "the census comparison can no longer fail"

# S-D: THE EXEC-BIT PROPERTY, damaged on the ARTIFACT (AUDIT #21). The new
# executability check is correct by construction on a clean tree, so weakening
# the CHECKER would be unobservable -- every script is 755, so deleting the
# assertion changes nothing observable. The defect must therefore be planted in
# the TREE: clear one script's execute bit and require the checker to reject it.
# This is the exact shape of the 2026-10-03 defect (five shebanged scripts at
# mode 644 that the stack never noticed), which is why it is the state that
# makes the new property mean anything.
#
# NOTE the mode IS restored between states: git records the exec bit, so
# `reset_tree` (git checkout -- .) puts it back. Verified by the residue check
# below, which fails loudly if the sandbox repo is not clean -- a leaked 644
# would otherwise be inherited by every state after this one, exactly the
# incomplete-restore defect that hit mutate_bench_load_gate.sh twice.
MUTATE_FN() {
  chmod 644 "$REPO"/scripts/check_stack_cert.sh || return 1
  [ -x "$REPO"/scripts/check_stack_cert.sh ] && return 1   # it must NOT be executable
  return 0
}
run_state "S-D" "a non-executable entry point is accepted" \
  'is not executable' \
  "a shebang that cannot be run is invisible again"

reset_tree
echo
echo "== residue + restore integrity =="
if git -C "$REPO" status --porcelain | grep -q .; then
  echo "  FAIL  the sandbox repo was not restored:"; git -C "$REPO" status --porcelain | sed 's/^/        /'
  exit 1
fi
echo "  ok    the sandbox repo is clean after the states"
git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null \
  && echo "  ok    the tracked tree was never modified" \
  || { echo "  FAIL tracked modified"; exit 1; }
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "STACK-CERT MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=stack_cert_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "STACK-CERT MUTATE: FAIL ($caught/$total caught)"
exit 1
