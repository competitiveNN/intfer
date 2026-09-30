#!/usr/bin/env bash
# Reproducible non-vacuity proof for scripts/run_all_checks.sh.
#
# WHY. run_all_checks.sh is the ENTRY POINT for the whole verification stack, and
# it was the one script in it with neither a mutation matrix nor a self-check --
# so a state that made it always-red, or that broke its verdict, would have
# looked healthy. Audit #16's rule: a new checker is a new gate.
#
# It cannot be matrixed naively: it runs the full ~45 minute stack, so four
# states would cost three hours. That is why run_all_checks.sh grew
# INTFER_RUN_ALL_STEPS / INTFER_RUN_ALL_EXTRA_STEPS_DIR / INTFER_RUN_ALL_STAMP_DIR
# seams -- the same rationale as INTFER_SYCL_LS and INTFER_BUILD_DIR: an
# assumption the harness cannot redirect is an assumption the harness cannot test.
# Every case here runs in seconds against synthetic steps and stamps, and no case
# runs another matrix, so there is no recursion.
#
# Each state breaks ONE mechanism. A state is CAUGHT when its DEFECT PREDICATE
# reports the defect present -- NOT when the rc is non-zero, because three of the
# four breaks produce rc=0 or 2, which is what a HEALTHY run also produces. A
# matrix that only watched the exit code would score those as passes forever.
#
# Every case is BASELINED first: the unmutated script is run in the same setup and
# must reach its declared good rc. Without that, a state could be "caught" by an
# unrelated breakage and never reproduce its own defect -- and a mutation that
# fails to reproduce is worse than none, because it launders red green.
#
#   N-A  classify rc=2 as a failure (drop the UNPROVEN verdict)
#        Re-collapses "evidence is missing" into "the code is broken", the exact
#        doctrine violation this repo audits for. Detected by the classifier
#        self-check, which exists for no other purpose.
#   N-B  delete the 2b re-run block
#        The verdict goes back to being computed from a snapshot the run itself
#        invalidated: a step that is unproven on the first pass and proven on the
#        re-run leaves the run reporting evidence missing when it is present.
#   N-C  replace the find-based stamp enumeration with a hand-written 2-item list
#        The failure audits #7 and #8 were about: a table that quietly stops
#        covering a matrix when one is added. Here a THIRD stamp becomes
#        invisible, so a stale one goes unreported and the run passes.
#   N-D  delete the dirty-tree guard
#        The mutation matrices restore with `git checkout --`, so without this
#        guard a run over uncommitted work DESTROYS it. Detected behaviourally: a
#        step that writes a sentinel must not run at all.
#
# Env: INTFER_BUILD_DIR  build tree (default $ROOT/build) for logs and the stamp.
# Exit: 0 every state caught; 1 otherwise.
#
# GOTCHA, learned the first time this was run: THIS SCRIPT MUST BE COMMITTED
# BEFORE IT CAN RUN. It drives run_all_checks.sh, which fails closed on ANY dirty
# tracked tree -- so an uncommitted edit to this file makes every baseline read
# rc=1 and the matrix reports "the CASE is broken, not the mutation". That is the
# guard behaving correctly; the fix is to commit, not to weaken the guard.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
TARGET="scripts/run_all_checks.sh"
WORK="$BUILD_DIR/run_all_mutate"
STAMP="$WORK/last_pass.stamp"
LOCK="$WORK/.lock"
rm -rf "$WORK"
mkdir -p "$WORK" || exit 1

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_run_all_checks.sh is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

if ! git diff --quiet -- "$TARGET" || ! git diff --cached --quiet -- "$TARGET"; then
  echo "FAIL: $TARGET has uncommitted changes; this script edits and restores it" >&2
  echo "  with 'git checkout --', which would destroy that work." >&2
  exit 1
fi

# --- the synthetic world ---------------------------------------------------
# Steps. step_flaky2 is unproven on its FIRST run and proven on the next, which
# is what makes the 2b re-run observable; step_sentinel records that it ran.
STEPS="$WORK/steps"; mkdir -p "$STEPS"
printf '#!/usr/bin/env bash\nexit 0\n' >"$STEPS/step_ok"
printf '#!/usr/bin/env bash\nC="%s/step_flaky2.count"\nn=$(cat "$C" 2>/dev/null || echo 0)\necho $((n+1)) >"$C"\n[ "$n" -eq 0 ] && exit 2\nexit 0\n' "$WORK" >"$STEPS/step_flaky2"
printf '#!/usr/bin/env bash\n: >"%s/step_sentinel.ran"\nexit 0\n' "$WORK" >"$STEPS/step_sentinel"
chmod +x "$STEPS"/*

# Stamps: a/ and b/ current, c/ STALE. c/ is the one a hand-written list misses.
mk_stamps() { # $1 = dir
  mkdir -p "$1/a" "$1/b" "$1/c"
  printf 'sha=%s\n' "$(git rev-parse HEAD)" >"$1/a/last_pass.stamp"
  printf 'sha=%s\n' "$(git rev-parse HEAD)" >"$1/b/last_pass.stamp"
  printf 'sha=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n' >"$1/c/last_pass.stamp"
}
ALL_CURRENT="$WORK/stamps_all_current"; mk_stamps "$ALL_CURRENT"
rm -f "$ALL_CURRENT/c/last_pass.stamp"
HAS_STALE="$WORK/stamps_with_stale"; mk_stamps "$HAS_STALE"

run_rac() { # $1 = case id, $2 = steps, $3 = stamp dir, $4 = extra env assignments
  local id="$1" steps="$2" sdir="$3" extra="${4:-}"
  rm -f "$WORK/step_sentinel.ran" "$WORK/step_flaky2.count"
  env INTFER_RUN_ALL_STEPS="$steps" \
      INTFER_RUN_ALL_EXTRA_STEPS_DIR="$STEPS" \
      INTFER_RUN_ALL_STAMP_DIR="$sdir" \
      $extra \
      timeout 600 bash "$TARGET" >"$WORK/$id.log" 2>&1
  return $?
}

# --- defect predicates: return 0 when the DEFECT IS PRESENT ----------------
def_n_a() { grep -q 'SELFCHECK: sc_unproven' "$WORK/N-A.log"; }
def_n_b() { # unproven never cleared -> the re-run did not happen
  grep -q '2 step(s) unproven\|1 step(s) unproven' "$WORK/N-B.log" && ! grep -q '(re-run)' "$WORK/N-B.log"; }
def_n_c() { # the stale stamp c/ was not reported
  ! grep -q 'STALE' "$WORK/N-C.log"; }
def_n_d() { [ -f "$WORK/step_sentinel.ran" ]; }   # a step ran on a dirty tree

# --- baselines: prove each case PASSES before it is broken -----------------
# $1 id, $2 steps, $3 stampdir, $4 want-good rc, $5 extra env
baseline() {
  local id="$1" steps="$2" sdir="$3" want="$4" extra="${5:-}" rc
  run_rac "$id" "$steps" "$sdir" "$extra"; rc=$?
  if [ "$rc" != "$want" ]; then
    echo "FAIL: baseline $id reached rc=$rc, want $want -- the CASE is broken, not" >&2
    echo "  the mutation. A state that cannot demonstrate correct behaviour cannot" >&2
    echo "  demonstrate its defect either. ($WORK/$id.log)" >&2
    exit 1
  fi
  printf '  ok    baseline %-4s rc=%s\n' "$id" "$rc"
}
echo "== BASELINES (unmutated run_all_checks must be CORRECT in each setup) =="
baseline "N-A" "step_ok" "$ALL_CURRENT" 0
baseline "N-B" "step_flaky2" "$ALL_CURRENT" 0
baseline "N-C" "step_ok" "$HAS_STALE" 2
# N-D's baseline needs a dirty tracked tree, which the guard must refuse.
DIRTY_FILE="scripts/_run_all_dirty_probe.md"
printf 'probe\n' >"$DIRTY_FILE"; git add -N "$DIRTY_FILE" 2>/dev/null
printf 'uncommitted\n' >>"$DIRTY_FILE"
baseline "N-D" "step_sentinel" "$ALL_CURRENT" 1
git checkout -- "$DIRTY_FILE" 2>/dev/null; rm -f "$DIRTY_FILE"

# --- anchor pre-flight -----------------------------------------------------
check_anchor() { # $1 = literal, $2 = description
  local n; n="$(grep -cF -- "$1" "$TARGET" 2>/dev/null || echo 0)"
  if [ "$n" != "1" ]; then
    echo "FAIL: anchor for $2 found $n times in $TARGET, want 1" >&2
    echo "  Target: $1" >&2
    exit 1
  fi
}
A_CLASSIFY="2) printf 'unproven\\n' ;;"
A_RERUN='if [ -n "$UNPROVEN_LABELS" ]; then'
A_FIND="find \"\$STAMP_DIR\" -name '*last_pass.stamp' -type f 2>/dev/null | sort"
A_GUARD='if ! git diff --quiet || ! git diff --cached --quiet; then'
check_anchor "$A_CLASSIFY" "N-A: the rc=2 UNPROVEN classification"
check_anchor "$A_RERUN"   "N-B: the 2b re-run guard"
check_anchor "$A_FIND"    "N-C: the find-based stamp enumeration"
check_anchor "$A_GUARD"   "N-D: the dirty-tree guard"
echo "anchor pre-flight: ok (4/4 targets located exactly once)"

# Exact literal replacement, no delimiter to collide with (audit #17b: a sed
# delimiter that occurs inside the pattern killed a state, and a state that
# silently does nothing is reported as "NOT load-bearing", a false verdict).
patch_literal() { # $1 = file, $2 = old, $3 = new
  python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
if n != 1:
    sys.exit("patch_literal: anchor found %d times, want 1" % n)
open(path, "w").write(src.replace(old, new))
PY
}
verify_patched() { # $1 = state id
  if git diff --quiet -- "$TARGET"; then
    echo "  ABORT $1  the patch did NOT change $TARGET. The state would score as" >&2
    echo "         'defect not reproduced', which is a fault in the STATE." >&2
    exit 1
  fi
}

CAUGHT=0; TOTAL=0
run_state() { # $1 id, $2 desc, $3 predicate fn, $4 steps, $5 stampdir, $6 extra env, $7 dirty?
  local id="$1" desc="$2" pred="$3" steps="$4" sdir="$5" extra="${6:-}" dirty="${7:-}"
  TOTAL=$((TOTAL + 1))
  if [ "$dirty" = "dirty" ]; then
    printf 'probe\n' >"$DIRTY_FILE"; git add -N "$DIRTY_FILE" 2>/dev/null
    printf 'uncommitted\n' >>"$DIRTY_FILE"
  fi
  run_rac "$id" "$steps" "$sdir" "$extra"
  git checkout -- "$TARGET" 2>/dev/null
  [ "$dirty" = "dirty" ] && { git checkout -- "$DIRTY_FILE" 2>/dev/null; rm -f "$DIRTY_FILE"; }
  if $pred; then
    CAUGHT=$((CAUGHT + 1))
    printf '  ok    %-4s defect reproduced: %s\n' "$id" "$desc"
  else
    printf '  MISSED %-4s no defect observed: %s\n' "$id" "$desc"
  fi
  if git diff --quiet -- "$TARGET"; then
    printf '         restored: %s clean\n' "$TARGET"
  else
    printf '  MISSED %-4s RESTORE FAILED -- manual repair needed\n' "$id"
    CAUGHT=$((CAUGHT - 1))
  fi
}

echo "== MUTATION MATRIX: run_all_checks.sh =="
# N-A: an unproven step is no longer unproven, it is a failure. The self-check
# is the only thing that can see this, which is why it exists.
assert_clean_target() { git diff --quiet -- "$TARGET" || { echo "  ABORT: $TARGET dirty" >&2; exit 1; }; }
assert_clean_target
patch_literal "$TARGET" "$A_CLASSIFY" "2) printf 'fail\\n' ;;"
verify_patched N-A
run_state "N-A" "rc=2 no longer classified UNPROVEN" def_n_a "step_ok" "$ALL_CURRENT" ""

# N-B: the 2b re-run is gone, so UNPROVEN is never cleared and the verdict is
# computed from the snapshot the run itself invalidated.
assert_clean_target
patch_literal "$TARGET" "$A_RERUN" "if false; then"
verify_patched N-B
run_state "N-B" "2b re-run deleted; unproven never cleared" def_n_b "step_flaky2" "$ALL_CURRENT" ""

# N-C: enumeration replaced by a hand-written two-item list. c/ is the third
# stamp, and a stale one it does not name goes unreported.
assert_clean_target
patch_literal "$TARGET" "$A_FIND" "printf '%s\\n' \"\$STAMP_DIR/a/last_pass.stamp\" \"\$STAMP_DIR/b/last_pass.stamp\""
verify_patched N-C
run_state "N-C" "find replaced by a 2-item list; 3rd stamp invisible" def_n_c "step_ok" "$HAS_STALE" ""

# N-D: the dirty-tree guard is gone, so steps run over uncommitted work that the
# matrices would then restore away.
assert_clean_target
patch_literal "$TARGET" "$A_GUARD" "if false; then"
verify_patched N-D
run_state "N-D" "dirty-tree guard deleted; a step ran over uncommitted work" def_n_d \
  "step_sentinel" "$ALL_CURRENT" "" "dirty"

echo
if [ "$CAUGHT" -eq "$TOTAL" ]; then
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=N-A,N-B,N-C,N-D"
    echo "caught=$CAUGHT/$TOTAL"
    echo "baselines=4/4 correct-before-broken"
  } >"$STAMP"
  echo "RUN-ALL MUTATION: PASS ($CAUGHT/$TOTAL caught, 4/4 baselines correct)"
  echo "  run_all_checks.sh can fail: its rc=2 verdict, its 2b re-run, its stamp"
  echo "  enumeration and its dirty-tree guard are each observable to break."
  echo "  stamp: $STAMP"
  exit 0
fi
echo "RUN-ALL MUTATION: FAIL ($CAUGHT/$TOTAL caught)" >&2
echo "  Check the baseline for that case first: a state whose baseline is wrong" >&2
echo "  never reproduced its own defect, so it is evidence of nothing." >&2
exit 1
