#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_matrix_sandbox_locks.sh.
#
# NO SUBJECT_READS MARKER, AND THAT IS THE POINT -- NOT AN OMISSION.
# Every assertion this matrix makes reads its own mktemp corpus (the fixtures
# under $CROOT) and the checker's source. Nothing in it reads HISTORY.md, AGENTS.md
# or docs/, so the durable-read population must NOT count it. Declaring the marker
# anyway -- on the reasoning that "the verdict is CITED in HISTORY.md" -- is the
# decorative shape check_durable_reads.sh's p_not_decorative predicate refuses:
# a marker no assertion depends on, which leaves an orphan marker in the tree and
# a marker count that disagrees with the reader count. A CITATION is not a READ.
# (The first version of this file DID declare the marker and DID fail the band;
# see the HISTORY.md record for the measurement.) The citation of this matrix's
# verdict is protected by the record, not by this matrix.
#
# EVERY STATE DAMAGES THE CORPUS, NEVER THE CHECKER. The corpus the checker grades
# is handed to it as $1 ("$ROOT"), and every planted defect lands in a fixture
# under that corpus root. Weakening the CHECKER would be unobservable to the
# checker's own run -- the defect would move with the predicate (AUDIT #21), which
# is why mutate_pipefail_grep_q.sh uses the same INTFER_SCRIPTS_DIR shape.
#
# The corpus is a MKTEMP sandbox, so this matrix is itself immune to the class it
# grades: a mktemp path is unique per invocation, nothing is shared, and there is
# nothing to lock. That is the checker's own condition (1), stated in code rather
# than in a comment.
#
# THE EIGHT STATES, each with its own refusal requirement:
#   MS-1 the lock is never declared            -> LOCK-NOT-DECLARED
#   MS-2 the lock is declared but never taken  -> LOCK-NOT-ACQUIRED
#   MS-3 the lock is taken but never released  -> LOCK-NOT-RELEASED
#   MS-4 the lock sits INSIDE the wiped tree   -> LOCK-INSIDE-WIPED-TREE
#   MS-5 the lock is taken AFTER the wipe      -> LOCK-AFTER-WIPE
#   MS-6 the population is below the floor     -> "derived only N matrix(es)"
#   MS-7 CONVERSE: an mktemp sandbox needs no lock, and must not be red
#   MS-8 the EXIT trap is DISARMED and the lock is released AFTER the disarm
#                                              -> LOCK-DISARM-BEFORE-RELEASE
# MS-7 is the anti-always-red half. Without it a checker that flagged EVERY matrix
# -- including the mktemp-immune ones the checker's own condition (1) excludes --
# would satisfy MS-1..MS-6 and read as proven.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
. "$SCRIPT_DIR/../lib/_gate_patch.sh"
. "$SCRIPT_DIR/../lib/_gate_pred.sh"

CHECK="$SCRIPT_DIR/../checks/check_matrix_sandbox_locks.sh"
[ -f "$CHECK" ] || { echo "SKIP: target missing"; exit 2; }

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_mslock.XXXXXX")"
LOG="$SB/states"; mkdir -p "$LOG"; : > "$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

# *** THE CHECKER IS READ, NEVER WRITTEN, SO AN UNCOMMITTED CHECKER IS AN INPUT,
# NOT A HAZARD. *** The original guard refused when $CHECK was dirty, which is the
# right rule for a matrix that MUTATES its target and restores it with
# `git checkout --` (that is how uncommitted work gets destroyed). This matrix
# only READS the checker's source and grades a mktemp corpus, so refusing here
# blocked every session that improves the checker -- measured 2026-10-30, the MS-8
# campaign. The refusal is replaced by a NAMED INPUT: the checker's blob hash is
# printed and stamped, so the verdict says exactly which source was graded even
# when that source is uncommitted, and a verdict cannot be silently attributed to
# a different revision.
CHECK_BLOB="$(git -C "$ROOT" hash-object "$CHECK" 2>/dev/null || echo unknown)"
if ! git -C "$ROOT" diff --quiet -- "$CHECK" 2>/dev/null; then
  CHECK_DIRTY=yes
  echo "  ..    INPUT: $CHECK is UNCOMMITTED (blob $CHECK_BLOB); that is the source"
  echo "        graded below, and the stamp names it -- this matrix never writes it."
else
  CHECK_DIRTY=no
fi

# --- the corpus -------------------------------------------------------------
# $CROOT/scripts/matrices/ holds FIXTURES whose declarations are the subject. The
# checker reads their source, never runs them, so a fixture need only carry the
# assignment shape the derivation looks for.
CROOT="$SB/root"
MATS="$CROOT/scripts/matrices"

# The baseline fixture: a fixed sandbox under $BUILD that is wiped, and a lock
# OUTSIDE it, taken before the wipe and released. Every planted state is a
# single-line edit of this one file, so no state can also change anything else.
fixture() {
  cat >"$1" <<'EOF'
#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/fixture_matrix"
LOCK="$BUILD/.fixture.lock"
mkdir -p "$BUILD" 2>/dev/null || true
  # *** THE LOCK'S PARENT MUST EXIST, OR A MISSING DIRECTORY READS AS A
  # COLLISION. *** `mkdir "$LOCK"` fails for two structurally different
  # reasons -- EEXIST, a real second instance, and ENOENT, its parent does
  # not exist -- and the message below names only the first. Measured
  # 2026-10-25: with a fresh INTFER_BUILD_DIR the parent does not exist,
  # so the mkdir died with ENOENT and the harness reported "another
  # instance is running" for a collision that never happened. The stack
  # hides it (run_all_checks.sh:62 creates its build dir first), so it only
  # fires on a standalone run -- a fresh clone, or an operator running one
  # matrix by hand, which is the documented way to run one. With the parent
  # created, mkdir can fail ONLY with EEXIST, so the existing message is
  # true without being rewritten.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another fixture is running" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
rm -rf "$SB"
mkdir -p "$SB"
echo state >"$SB/stamp"
EOF
}

# The mktemp fixture: immune by the checker's own condition (1). MS-7 asserts the
# checker leaves it alone.
mktemp_fixture() {
  cat >"$1" <<'EOF'
#!/usr/bin/env bash
set -u
SB="$(mktemp -d "${TMPDIR:-/tmp}/mfix.XXXXXX")"
rm -rf "$SB"
mkdir -p "$SB"
EOF
}

corpus() { # $1 = number of locked fixtures
  rm -rf "$CROOT"; mkdir -p "$MATS"
  local i
  for i in $(seq 1 "$1"); do
    fixture "$MATS/mutate_fx$i.sh"
  done
}

echo "== corpus-based population derivation =="
echo "-- sandbox corpus: $CROOT (mktemp, so this matrix is itself immune)"
echo

caught=0; total=0

state() { # $1 id, $2 corpus size, $3 sed-expr or '' , $4 marker, $5 expect-ERE, $6 desc
  local id="$1" n="$2" sedexpr="$3" marker="$4" ere="$5" desc="$6"
  local out rc
  total=$((total + 1))
  corpus "$n"
  if [ -n "${DEBUG:-}" ]; then
    echo "--- corpus for $id ($n locked fixtures) ---" >&2
    grep -nE '^[A-Z_]+=|^rm |^mkdir' "$MATS"/mutate_fx1.sh >&2
  fi
  if [ -n "$sedexpr" ]; then
    # every state mutates the FIRST fixture; the rest stay pristine, which is
    # what keeps a state's refusal attributable to that one line.
    sed -i "$sedexpr" "$MATS/mutate_fx1.sh"
  fi
  verify_ok=1
  [ -z "$marker" ] || grep -qF -- "$marker" "$MATS/mutate_fx1.sh" || verify_ok=0
  out="$(bash "$CHECK" "$CROOT" 2>&1)"; rc=$?
  if [ "$verify_ok" != 1 ]; then
    printf '  %s MISSED -- the planted mutation did not land (sed no-op)\n' "$id"
    printf '%s\n' "  $id MISSED (mutation never landed)" >>"$LOG/summary.txt"
    return
  fi
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qE -- "$ere"; then
    caught=$((caught + 1))
    printf '  %s caught -- rc=1 and the refusal named it: %s\n' "$id" "$desc"
    printf '%s\n' "  $id caught" >>"$LOG/summary.txt"
  else
    printf '  %s MISSED -- rc=%s and the expected refusal did not name: %s\n' \
      "$id" "$rc" "$desc"
    printf '%s\n' "$out" | sed -n '2,6p' >&2
    printf '%s\n' "  $id MISSED (rc=$rc)" >>"$LOG/summary.txt"
  fi
  # Residue: the planted mutation must not survive into the next state, or every
  # later state is graded against a corpus the previous one already broke. The
  # rebuild is always 3 fixtures -- the floor is 3, so a corpus of 2 would fail
  # for the WRONG reason and mask a real residue.
  corpus 3
  local after afterrc
  after="$(bash "$CHECK" "$CROOT" 2>&1)"; afterrc=$?
  if [ "$afterrc" -ne 0 ]; then
    echo "FAIL: state $id left the corpus dirty -- a later state would be graded against a broken baseline" >&2
    printf '%s\n' "$out" | tail -3 >&2
    exit 1
  fi
}

# --- baseline: a clean corpus MUST pass (the converse of always-red) --------
corpus 3
base="$(bash "$CHECK" "$CROOT" 2>&1)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL -- the checker refuses a healthy corpus, so every state below"
  echo "would be scoring against a broken predicate:"
  printf '%s\n' "$base" | tail -5
  exit 1
fi
echo "  ok    baseline: 3 locked fixtures + an immune one => PASS"

pred_reset
pred_expect_present "baseline verdict" "$base" 'RESULT: PASS \(3 matrix\(es\)'
pred_bad | grep -qx 0 || { echo "BASELINE FAIL -- no PASS verdict line" >&2; exit 1; }
echo

echo "== states =="
# MS-1 the lock is never declared at all
state MS-1 3 's@^LOCK="\$BUILD/\.fixture\.lock"$@TRUE=1 #MUT_MS1@' MUT_MS1 'LOCK-NOT-DECLARED' \
  'the lock is never declared'
# MS-2 declared but never acquired. The acquire is the `mkdir` in the CONDITION of
# the guard, so neutering the condition removes the acquire with it.
state MS-2 3 's@^if ! mkdir "\$LOCK" 2>/dev/null; then$@if false; then #MUT_MS2@' MUT_MS2 'LOCK-NOT-ACQUIRED' \
  'the lock is declared but never taken'
# MS-3 acquired but never released (the release is the trap body)
state MS-3 3 "s@trap 'rmdir \"\$LOCK\" 2>/dev/null || true' EXIT@trap 'true' EXIT #MUT_MS3@" MUT_MS3 'LOCK-NOT-RELEASED' \
  'the lock is taken but never released'
# MS-4 the lock lives inside the directory the wipe removes
state MS-4 3 's@^LOCK="\$BUILD/\.fixture\.lock"$@# MUT_MS4\nLOCK="\$SB/\.fixture\.lock"@' MUT_MS4 'LOCK-INSIDE-WIPED-TREE' \
  'the lock sits inside the tree the wipe removes'
# MS-5 the lock is taken AFTER the wipe has already run. The acquire is MOVED,
# not deleted: deleting it would fire LOCK-NOT-ACQUIRED instead, and a state that
# proves a different predicate than it claims proves nothing.
state MS-5 3 's@^if ! mkdir "\$LOCK" 2>/dev/null; then$@@; s@^rm -rf "\$SB"$@rm -rf "$SB"\nif ! mkdir "$LOCK" 2>/dev/null; then echo "FAIL" >\&2; exit 1; fi #MUT_MS5@' MUT_MS5 'LOCK-AFTER-WIPE' \
  'the lock is acquired after the wipe'
# MS-6 the population is below the checker's own floor
state MS-6 2 '' '' 'derived only 2 matrix\(es\)' \
  'a population below the floor must not read as PASS'
# MS-8 the EXIT handler is DISARMED and the release moved AFTER the disarm. This is
# the subtle half of the 2026-10-30 campaign: a static `rmdir appears somewhere`
# test reads it as released, but `trap - EXIT` never RUNS the handler, so the lock
# leaks on exactly the path that runs the disarm. The acquire stays where it was,
# so LOCK-NOT-ACQUIRED/LOCK-AFTER-WIPE cannot fire and mask this predicate.
state MS-8 3 "s@^trap 'rmdir \"\$LOCK\" 2>/dev/null || true' EXIT\$@trap - EXIT #MUT_MS8\nrmdir \"\$LOCK\" 2>/dev/null || true@" MUT_MS8 'LOCK-DISARM-BEFORE-RELEASE' \
  'a disarm drops the EXIT handler without running it, so a release after it never runs'
# MS-7 CONVERSE: an mktemp sandbox is immune and must be left alone. This state
# ADDS the immune fixture to a corpus that already passes and re-requires PASS.
corpus 3
mktemp_fixture "$MATS/mutate_fxm.sh"
mt="$(bash "$CHECK" "$CROOT" 2>&1)"; mtrc=$?
total=$((total + 1))
if [ "$mtrc" -eq 0 ] && printf '%s\n' "$mt" | grep -qE 'RESULT: PASS'; then
  caught=$((caught + 1))
  echo "  MS-7 caught (converse) -- an mktemp sandbox stays immune, rc=0 and PASS"
  printf '%s\n' "  MS-7 caught" >>"$LOG/summary.txt"
else
  echo "  MS-7 MISSED -- the checker reddened an immune matrix"
  printf '%s\n' "$mt" | tail -4 >&2
  printf '%s\n' "  MS-7 MISSED" >>"$LOG/summary.txt"
fi

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
echo
echo "SANDBOX-LOCK MUTATION MATRIX: $([ "$caught" -eq "$total" ] && echo PASS || echo FAIL) ($caught/$total states caught, each for its own reason)"
if [ "$caught" -ne "$total" ]; then
  echo "RESULT: a planted defect was NOT caught -- the checker is weaker than it claims" >&2
  exit 1
fi
mkdir -p "$SB/stamp"
{ echo "stamp=matrix_sandbox_locks_mutate"
  echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "checker_blob=$CHECK_BLOB"; echo "checker_dirty=$CHECK_DIRTY"
  echo "states=$total"; echo "caught=$caught"; } >"$SB/stamp/last_pass.stamp"
echo "RESULT: PASS (every lock-class defect observed for its own named predicate)"
exit 0
