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
# Every case runs in seconds against synthetic steps and stamps, and no case runs
# another matrix, so there is no recursion.
#
# ==== THE FIRST VERSION OF THIS MATRIX WAS WRONG IN AN IMPORTANT WAY ====
# It mutated scripts/run_all_checks.sh IN PLACE and then ran it. But
# run_all_checks.sh fails closed on any dirty tracked tree -- so the MUTATION ITSELF
# tripped its own target's guard, and every run aborted at the top before the
# defect under test could express itself. Consequences, both bad:
#   * N-A and N-B scored MISSED, when in fact the states never ran.
#   * N-C scored CAUGHT, and that was WORSE: its predicate was "no STALE line in
#     the output", which a top-of-run abort trivially satisfies. A false catch.
# A mask that hides the defect under test is worse than no state at all, and a
# predicate loose enough to be satisfied by an early abort is a check that cannot
# fail.
#
# The SECOND version used an UNTRACKED copy at scripts/_rac_mutant.sh, which fixed
# the masking -- and then immediately collided with the tree-hygiene assertion
# this matrix also introduced, because that copy IS untracked residue in the very
# directory the assertion polices. Exempting it by name would be a hand-maintained
# list, i.e. the copy-drift family again.
#
# So the mutant now lives in a THROWAWAY GIT REPO (see $SB below). Its git diff
# and git rev-parse resolve inside the sandbox, its stamps are built from the
# sandbox head, and the real repository is never touched at all. Each patched state
# is COMMITTED in the sandbox, so the mutant's own start-of-run guard stays armed
# against what it is meant to catch (N-D's deliberate dirtiness) rather than
# against the harness.
#
# Each state is judged by a DEFECT PREDICATE, and every predicate is a POSITIVE
# assertion about the defect ("this specific evidence is ABSENT/BAD"), never the
# mere absence of a good signal. Each case is also BASELINED first, and the
# baseline must show the CORRECT marker PRESENT as well as the right rc -- so a
# state that cannot demonstrate correct behaviour cannot demonstrate its defect
# either, and a mutation that fails to reproduce is worse than none.
#
#   N-A  classify rc=2 as a failure (drop the UNPROVEN verdict)
#        Re-collapses "evidence is missing" into "the code is broken", the exact
#        doctrine violation this repo audits for. Only the classifier self-check
#        can see it, which is why it exists.
#   N-B  delete the 2b re-run block
#        The verdict goes back to being computed from a snapshot the run itself
#        invalidated: a step unproven on the first pass and proven on the re-run
#        leaves the run reporting evidence missing when it is present.
#   N-C  replace the find-based stamp enumeration with a hand-written 2-item list
#        A THIRD stamp becomes invisible, so a stale one goes unreported and the
#        run passes. The audits #7/#8 shape.
#   N-D  delete the dirty-tree guard
#        The mutation matrices restore with `git checkout --`, so without this
#        guard a run over uncommitted work DESTROYS it. Detected behaviourally: a
#        step that writes a sentinel must not run at all.
#   N-E  neutralise the SELF-CHECK, compound with the N-A classification break
#        The matrix proves the step CLASSIFIER can fail, but nothing yet proved
#        the self-check that watches it can fail -- the always-red lesson applied
#        one level in. N-A alone cannot answer that: its detection mechanism IS
#        the self-check, so neutralising the self-check is invisible while
#        classify_rc is correct. So N-E breaks BOTH: a misclassification the
#        self-check must catch, and a self-check that cannot. If the run reports
#        no SELFCHECK line, the self-check is not load-bearing.
#   N-F  break the 2b re-run's DERIVATION, not its guard
#        N-B deletes the re-run block. This keeps the block and breaks the
#        accumulated $UNPROVEN_LABELS set it iterates, so the re-run finds nothing
#        to do. Same observable as N-B, different mechanism: the audits #7/#8
#        "enumerate, never remember" discipline proven load-bearing in the re-run
#        path too, not merely present.
#   N-G  delete the end-of-run tree-hygiene assertion
#        run_all_checks refuses to START dirty, but that does not prove a step
#        left the tree as it found it. Each matrix only verifies the files it
#        declared as targets, so a step that dirtied something else leaves no
#        trace in its own report. With the assertion gone, untracked residue
#        anywhere in the work tree goes unnoticed and the run passes.
#   N-H  make the 2b re-run's rc=1 arm a no-op (the pre-fix shape)
#        The re-run is where stale evidence is resolved, so it is also where a
#        REAL failure can appear. Patched back to the first version -- branch on
#        success/failure only, leave FAILED at 0 either way -- a step answering
#        rc=2 then rc=1 prints "still rc=1" and the run exits 2 with "NOT a
#        regression: Nothing is wrong with the code", i.e. a checker failure is
#        reported as healthy. Measured before the fix.
#   N-I  narrow the residue diff back to scripts/
#        The bug the diff replaced: the assertion ran `git status -- scripts/`
#        and called it the whole tree, because scripts/ is where an earlier
#        version of this matrix kept its mutant. A step that litters the
#        repository ROOT then leaves the run printing "no untracked residue in
#        scripts/" and exiting 0 = PASS. Measured before the fix.
#   N-J  neuter the dispatch-coverage enumeration
#        dispatch() and STEPS are hand-written, so a new check_*/mutate_* nobody
#        remembers to add is not a step that fails, it is a step that NEVER
#        RUNS -- and this entry point would still print PASS. The state drops an
#        undispatched mutate_dummy.sh into the sandbox's scripts/ and proves the
#        coverage check is what notices.
#
# GOTCHA, learned the first time this was run: THIS SCRIPT MUST BE COMMITTED
# BEFORE IT CAN RUN. It drives run_all_checks.sh, which fails closed on ANY dirty
# tracked tree -- so an uncommitted edit to this file makes every baseline read
# rc=1 and the matrix reports "the CASE is broken, not the mutation". That is the
# guard behaving correctly; the fix is to commit, not to weaken the guard.
#
# Env: INTFER_BUILD_DIR  build tree (default $ROOT/build) for logs and the stamp.
# Exit: 0 every state caught; 1 otherwise.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
TARGET="scripts/run_all_checks.sh"
# The mutant lives in a SANDBOX GIT REPO, not in the real tree. It used to be an
# untracked copy at scripts/_rac_mutant.sh, which put it inside the very
# directory run_all_checks.sh now asserts has no untracked residue -- so every
# baseline read the harness's own file as leftover residue. Exempting it by name
# would be a hand-maintained list, i.e. the copy-drift family again.
#
# A sandbox removes the need for an exemption AND stops the matrix touching the
# real repo at all: its own `git diff`/`git rev-parse` resolve inside the sandbox,
# whose HEAD is a throwaway commit. The stamps are built from the SANDBOX head, so
# they are self-consistent with the code under test.
# The SYNTHETIC world deliberately lives OUTSIDE the build tree. Two reasons, both
# learned the hard way:
#   * run_all_checks.sh enumerates stamps with `find "$BUILD_DIR" -name
#     '*last_pass.stamp'`. Synthetic stamp files inside the build tree were picked
#     up by the REAL enumeration, so the stack's stamp population grew by seven
#     fixture files -- the enumerator silently measuring its own test data.
#     Adding this matrix is exactly the population change that exposed it.
#   * The inner runs get their own INTFER_BUILD_DIR (see run_rac), so the mutant's
#     lock is not the one the OUTER run_all_checks.sh already holds. Sharing it
#     made every inner run exit 1 with "another run_all_checks.sh is running", and
#     the matrix correctly reported that as a broken CASE.
# Assigned BEFORE the sandbox paths below, which are expressed relative to it.
SYNTH="$(mktemp -d "${TMPDIR:-/tmp}/rac_mut.XXXXXX")" || exit 1
mkdir -p "$SYNTH/inner"
SB="$SYNTH/root"
MUTANT="$SB/scripts/run_all_checks.sh"
WORK="$BUILD_DIR/run_all_mutate"
STAMP="$WORK/last_pass.stamp"
LOCK="$WORK/.lock"
PROBE="$SB/scripts/_rac_dirty_probe.md"
RESIDUE_PROBE="$SB/scripts/_rac_residue_probe.md"
# A SECOND litter site, at the sandbox repo ROOT. The scripts/-only check that
# N-I restores cannot see this one; the whole-work-tree diff can.
ROOT_PROBE="$SB/_rac_root_litter.md"
DUMMY="$SB/scripts/mutate_dummy.sh"
mkdir -p "$SB/scripts"
rm -rf "$WORK"; mkdir -p "$WORK" || exit 1
cp "$TARGET" "$MUTANT"; chmod +x "$MUTANT"
git init -q "$SB" || { echo "FAIL: could not init sandbox repo $SB" >&2; exit 1; }
git -C "$SB" -c user.email=m@x -c user.name=matrix add -A >/dev/null 2>&1
git -C "$SB" -c user.email=m@x -c user.name=matrix commit -q -m sandbox >/dev/null 2>&1
SB_HEAD="$(git -C "$SB" rev-parse HEAD)" || { echo "FAIL: sandbox has no HEAD" >&2; exit 1; }
export GIT_CONFIG_NOSYSTEM=1

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_run_all_checks.sh is running (lock: $LOCK)" >&2
  exit 1
fi
# Always clean up the mutant, even on an abort: it lives in scripts/, and leaving
# a mutated copy there would poison every later check that globs that directory.
cleanup() {
  rm -f "$PROBE" "$RESIDUE_PROBE" "$ROOT_PROBE" "$DUMMY"
  rmdir "$LOCK" 2>/dev/null || true
  rm -rf "$SYNTH"
}
trap cleanup EXIT

[ -e "$TARGET" ] || { echo "FAIL: $TARGET missing" >&2; exit 1; }
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "FAIL: the tracked tree has uncommitted changes." >&2
  echo "  The mutant inherits that dirtiness, so every case would read the dirty" >&2
  echo "  guard instead of the defect. Commit or stash first." >&2
  git status --porcelain --untracked-files=no >&2
  exit 1
fi

# --- the synthetic world ---------------------------------------------------
STEPS="$SYNTH/steps"; mkdir -p "$STEPS"
printf '#!/usr/bin/env bash\nexit 0\n' >"$STEPS/step_ok"
printf '#!/usr/bin/env bash\nC="%s/step_flaky2.count"\nn=$(cat "$C" 2>/dev/null || echo 0)\necho $((n+1)) >"$C"\n[ "$n" -eq 0 ] && exit 2\nexit 0\n' "$SYNTH" >"$STEPS/step_flaky2"
printf '#!/usr/bin/env bash\n: >"%s/step_sentinel.ran"\nexit 0\n' "$SYNTH" >"$STEPS/step_sentinel"
# N-G/N-I: a step that leaves residue DURING the run. It has to be the STEP that
# litters, not the harness: the end-of-run check is a diff against a snapshot
# taken before step 1, so a file the harness created beforehand is present in
# both snapshots and correctly reads as "not left behind". That first version
# of the N-G baseline scored rc=0 for exactly that reason -- the case was broken,
# and the baseline caught it.
#
# A heredoc, not a printf format. The format was written with '\${...}' inside
# single quotes, which printf emitted VERBATIM, so the step created a file
# literally named '${INTFER_RAC_LITTER:?no litter path}' in the sandbox root
# instead of the intended path. N-G still "caught" (the assertion fires on ANY
# new untracked file) and N-I then read rc=0 -- because that stray survived
# run_rac's cleanup and was therefore already present in the NEXT run's
# before-snapshot. A false catch on one state silently disarming the next; the
# N-I baseline caught it.
cat >"$STEPS/step_litter" <<EOF
#!/usr/bin/env bash
[ -n "\${INTFER_RAC_LITTER:-}" ] || exit 3
printf 'litter\n' >"\$INTFER_RAC_LITTER"
exit 0
EOF
# N-H: unproven on the FIRST pass, a REAL failure on the re-run. The only
# shape that distinguishes "missing evidence" from "a regression appeared while
# we were re-measuring", and therefore the only shape that exercises the
# re-run's rc=1 arm.
printf '#!/usr/bin/env bash\nC="%s/step_escalate.count"\nn=$(cat "$C" 2>/dev/null || echo 0)\necho $((n+1)) >"$C"\n[ "$n" -eq 0 ] && exit 2\nexit 1\n' "$SYNTH" >"$STEPS/step_escalate"
chmod +x "$STEPS"/*

mk_stamps() { # $1 = dir, $2 = "stale" to make c/ stale
  mkdir -p "$1/a" "$1/b" "$1/c"
  # The sandbox head ADVANCES with every patch, so the stamps are (re)written
  # per run from the head in force NOW. Baking them once at setup made every
  # state after the first read as STALE -- the evidence describing older code,
  # which is the family-3 lesson in miniature.
  local h; h="$(git -C "$SB" rev-parse HEAD)"
  printf 'sha=%s\n' "$h" >"$1/a/last_pass.stamp"
  printf 'sha=%s\n' "$h" >"$1/b/last_pass.stamp"
  printf 'sha=%s\n' "$h" >"$1/c/last_pass.stamp"
  case "$1" in *stale*)
    printf 'sha=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n' >"$1/c/last_pass.stamp" ;;
  esac
  return 0
}
ALL_CURRENT="$SYNTH/stamps_all_current"; mk_stamps "$ALL_CURRENT" current
HAS_STALE="$SYNTH/stamps_with_stale";    mk_stamps "$HAS_STALE" stale

# Path-limited on purpose. 'git commit -a' would ALSO commit the deliberately
# dirty probe file, which silently disarms N-D: the tree came back clean, the
# start-of-run guard never fired, and the state scored as a broken CASE rather
# than as the defect it tests.
commit_mutant() { # $1 = message
  git -C "$SB" -c user.email=m@x -c user.name=matrix add -- "$MUTANT" >/dev/null 2>&1
  git -C "$SB" -c user.email=m@x -c user.name=matrix commit -q -m "$1" -- "$MUTANT" >/dev/null 2>&1 || true
}
reset_mutant() { cp "$TARGET" "$MUTANT"; chmod +x "$MUTANT"; commit_mutant mutant; }
run_rac() { # $1 = case id, $2 = steps, $3 = stamp dir, $4 = leave residue?
            # $5 = "root" to litter the sandbox repo ROOT instead of scripts/
  local id="$1" steps="$2" sdir="$3" residue="${4:-}" where="${5:-}"
  local litter="$RESIDUE_PROBE"
  [ "$where" = "root" ] && litter="$ROOT_PROBE"
  mk_stamps "$sdir"
  rm -f "$SYNTH/step_sentinel.ran" "$SYNTH/step_flaky2.count" \
        "$SYNTH/step_escalate.count" "$RESIDUE_PROBE" "$ROOT_PROBE"
  # Deliberately NOT staged. The end-of-run assertion flags UNTRACKED residue,
  # and `git add -N` makes the file intent-to-add, which counts as staged dirt --
  # so the START guard rejected it first and the residue assertion was never
  # reached. Untracked means untracked.
  # INTFER_BUILD_DIR is redirected so the mutant's lock is not the outer run's.
  INTFER_BUILD_DIR="$SYNTH/inner" \
  INTFER_RUN_ALL_STEPS="$steps" INTFER_RUN_ALL_EXTRA_STEPS_DIR="$STEPS" \
    INTFER_RUN_ALL_STAMP_DIR="$sdir" INTFER_RAC_LITTER="$litter" \
    timeout 600 bash "$MUTANT" >"$WORK/$id.log" 2>&1
  local rc=$?
  rm -f "$RESIDUE_PROBE" "$ROOT_PROBE"
  return $rc
}

# --- baselines: correct behaviour must be observable BEFORE any mutation ----
# $1 id, $2 steps, $3 stampdir, $4 want rc, $5 marker that MUST be present
baseline() {
  local id="$1" steps="$2" sdir="$3" want="$4" marker="$5" residue="${6:-}" rc
  local where="${7:-}"   # litter site: "" = scripts/, "root" = sandbox repo root
  NBASE=$((NBASE + 1))
  reset_mutant
  run_rac "$id" "$steps" "$sdir" "$residue" "$where"; rc=$?
  if [ "$rc" != "$want" ]; then
    echo "FAIL: baseline $id reached rc=$rc, want $want -- the CASE is broken, not" >&2
    echo "  the mutation. ($WORK/$id.log)" >&2; exit 1
  fi
  if [ -n "$marker" ] && ! grep -qF -- "$marker" "$WORK/$id.log"; then
    echo "FAIL: baseline $id did not show the CORRECT marker '$marker'." >&2
    echo "  Without this a predicate can be satisfied by an early abort and score" >&2
    echo "  a false catch. ($WORK/$id.log)" >&2; exit 1
  fi
  printf '  ok    baseline %-4s rc=%s, correct marker present\n' "$id" "$rc"
}

# --- defect predicates: return 0 ONLY when the defect is really present -----
# Each names a specific piece of evidence that must be MISSING when broken.
def_n_a() { grep -q 'SELFCHECK: sc_unproven' "$WORK/N-A.log"; }
def_n_b() { grep -q '1 step(s) unproven' "$WORK/N-B.log" && ! grep -q '(re-run)' "$WORK/N-B.log"; }
def_n_c() { ! grep -q 'c/last_pass.stamp' "$WORK/N-C.log"; }   # 3rd stamp not enumerated
def_n_d() { [ -f "$SYNTH/step_sentinel.ran" ]; }                  # a step ran on a dirty tree
# The defect is not "no SELFCHECK line": those are printed DURING the loop, so
# they appear even when the self-check's return value is neutralised. The defect
# is that the run PROCEEDS past the self-check while classify_rc is wrong. That
# only means anything because N-A shows the converse -- with the self-check armed,
# a misclassification never reaches "1/2 checkers". That converse is asserted
# below, so this predicate cannot pass vacuously.
def_n_e() { grep -q '1/2 checkers' "$WORK/N-E.log"; }
def_n_f() { grep -q 'step(s) unproven' "$WORK/N-F.log" && ! grep -q '(re-run)' "$WORK/N-F.log"; }
# Anchored on the FAIL line specifically. A bare grep for 'untracked residue'
# also matches the SUCCESS branch ("ok  no new untracked files ..."), so the
# predicate was never true -- the first version of this state could not be caught.
def_n_g() { ! grep -qE 'FAIL +untracked residue' "$WORK/N-G.log"; }
# N-H: the escalating step must be announced as a REGRESSION. Absence of the
# line is the defect; the baseline below requires the line PRESENT, so the
# predicate cannot be satisfied by a run that never got that far.
def_n_h() { ! grep -q 'REGRESSED' "$WORK/N-H.log"; }
# N-I: same shape as N-G, but the litter is at the sandbox repo ROOT, which the
# pre-fix scripts/-only scope cannot see.
def_n_i() { ! grep -qE 'FAIL +untracked residue' "$WORK/N-I.log"; }
def_n_j() { ! grep -q 'no dispatch() arm for' "$WORK/N-J.log"; }

# N-H/N-I/N-J need their litter/counter state cleared before each run, which
# run_rac does; N-H also needs its own counter file reset because a leftover
# count would make the first pass answer rc=1 (a plain failure) and never reach
# the re-run. run_rac removes it, which is asserted by the N-H baseline marker.

NBASE=0
echo "== BASELINES (unmutated run_all_checks must be CORRECT in each setup) =="
baseline "N-A" "step_ok"     "$ALL_CURRENT" 0 ""
baseline "N-B" "step_flaky2" "$ALL_CURRENT" 0 "ok (evidence now current)"   # 2b re-run visible
baseline "N-C" "step_ok"     "$HAS_STALE"   2 "STALE c/last_pass.stamp"      # 3rd stamp enumerated
printf 'probe\n' >"$PROBE"; git -C "$SB" add -- "$PROBE"   # a genuinely TRACKED, dirty file
baseline "N-D" "step_sentinel" "$ALL_CURRENT" 1 "uncommitted changes"
git -C "$SB" reset -q -- "$PROBE"; rm -f "$PROBE"
baseline "N-E" "step_ok"     "$ALL_CURRENT" 0 ""                              # self-check passes
baseline "N-F" "step_flaky2" "$ALL_CURRENT" 0 "ok (evidence now current)"       # re-run works
baseline "N-G" "step_litter" "$ALL_CURRENT" 1 "untracked residue left behind" yes
baseline "N-H" "step_escalate" "$ALL_CURRENT" 1 "REGRESSED rc=1"
# The litter goes to the sandbox ROOT, and the baseline must prove the check
# still sees it there -- otherwise N-I's predicate would be satisfied by a run
# that never littered anything.
baseline "N-I" "step_litter" "$ALL_CURRENT" 1 "untracked residue left behind" yes root
# N-J: an undispatched matrix dropped into scripts/. COMMITTED, so the run's own
# residue diff cannot be what fails -- only the coverage check can.
printf '#!/usr/bin/env bash\nexit 0\n' >"$DUMMY"; chmod +x "$DUMMY"
git -C "$SB" -c user.email=m@x -c user.name=matrix add -- "$DUMMY" >/dev/null 2>&1
git -C "$SB" -c user.email=m@x -c user.name=matrix commit -q -m dummy -- "$DUMMY" >/dev/null 2>&1
baseline "N-J" "step_ok"     "$ALL_CURRENT" 1 "no dispatch() arm for: mutate_dummy.sh"
rm -f "$DUMMY"

# --- anchor pre-flight (checked against the PRISTINE target) ---------------
check_anchor() { # $1 = literal, $2 = description
  # `grep -c` prints "0" AND exits 1 when nothing matches, so `|| echo 0` used to
  # append a SECOND "0" and every miss reported "found 0\n0 times". It failed
  # closed either way, but the diagnostic IS the product of a mutation harness --
  # three sibling sites already use `|| true`, which is the correct form.
  local n; n="$(grep -cF -- "$1" "$TARGET" 2>/dev/null || true)"
  if [ "$n" != "1" ]; then
    echo "FAIL: anchor for $2 found $n times in $TARGET, want 1" >&2
    echo "  Target: $1" >&2; exit 1
  fi
}
A_CLASSIFY="2) printf 'unproven\\n' ;;"
A_RERUN='if [ -n "$UNPROVEN_LABELS" ]; then'
A_FIND="find \"\$STAMP_DIR\" -name '*last_pass.stamp' -type f 2>/dev/null | sort"
A_GUARD='if ! git diff --quiet || ! git diff --cached --quiet; then'
A_SELFCHK='return $bad'
A_DERIVE='UNPROVEN_LABELS="$UNPROVEN_LABELS $label"'
A_RESIDUE='RESIDUE="$(LC_ALL=C comm -13 \'
A_ESCALATE="printf 'REGRESSED rc=%s (%s) -- was unproven, is now a real failure\\n' \"\$rc\" \"\$log\""
A_COVER='grep -qxF "$b" || UNDISPATCHED="$UNDISPATCHED $b"'
check_anchor "$A_CLASSIFY" "N-A: the rc=2 UNPROVEN classification"
check_anchor "$A_RERUN"   "N-B: the 2b re-run guard"
check_anchor "$A_FIND"    "N-C: the find-based stamp enumeration"
check_anchor "$A_GUARD"   "N-D: the dirty-tree guard"
check_anchor "$A_SELFCHK" "N-E: the self-check verdict"
check_anchor "$A_DERIVE"  "N-F: the re-run set derivation"
check_anchor "$A_RESIDUE" "N-G: the tree-hygiene residue diff"
check_anchor "$A_ESCALATE" "N-H: the 2b re-run's REGRESSED announcement"
check_anchor "$A_COVER"   "N-J: the dispatch-coverage enumeration"
echo "anchor pre-flight: ok (9/9 targets located exactly once)"

# Exact literal replacement -- no delimiter that can occur inside the pattern
# (audit #17b: a sed delimiter inside an ERE alternation killed a state, and a
# state that silently does nothing is reported as "defect not reproduced").
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
  # Commit the patch. In the sandbox the mutant is a TRACKED file, so patching it
  # leaves the tree dirty -- and the mutant's own start-of-run guard would then
  # abort every case before the defect could appear. That is precisely the masking
  # that broke the FIRST version of this matrix, reappearing one level down; the
  # sandbox removes it only if each patched state is committed.
  commit_mutant patched
}
verify_patched() { # $1 = state id -- the MUTANT copy must differ from the target
  if cmp -s "$MUTANT" "$TARGET"; then
    echo "  ABORT $1  the patch did NOT change $MUTANT. The state would score as" >&2
    echo "         'defect not reproduced', which is a fault in the STATE, not the" >&2
    echo "         checker. Fix the patch; do not touch the predicate." >&2
    exit 1
  fi
  bash -n "$MUTANT" || { echo "  ABORT $1  the patch broke the mutant's SYNTAX." >&2
                          echo "         A syntax error is not evidence the mechanism" >&2
                          echo "         is load-bearing -- it just fails earlier." >&2; exit 1; }
}

CAUGHT=0; TOTAL=0
run_state() { # $1 id, $2 desc, $3 predicate, $4 steps, $5 stampdir, $6 dirty?
              # $7 residue?, $8 litter site ("root"), $9/$10 optional SECOND
              # (old,new) patch pair, for N-E
  local id="$1" desc="$2" pred="$3" steps="$4" sdir="$5" dirty="${6:-}"
  local residue="${7:-}" where="${8:-}" old2="${9:-}" new2="${10:-}"
  TOTAL=$((TOTAL + 1))
  [ "$dirty" = "dirty" ] && { printf 'probe\n' >"$PROBE"; git -C "$SB" add -- "$PROBE"; }
  run_rac "$id" "$steps" "$sdir" "$residue" "$where"
  [ "$dirty" = "dirty" ] && { git -C "$SB" reset -q -- "$PROBE"; rm -f "$PROBE"; }
  if $pred; then
    CAUGHT=$((CAUGHT + 1))
    printf '  ok    %-4s defect reproduced: %s\n' "$id" "$desc"
  else
    printf '  MISSED %-4s defect NOT reproduced: %s\n' "$id" "$desc"
  fi
  # Printed unconditionally, so it must not carry an "ok" verdict: the first
  # version prefixed it "ok", which printed "ok" directly under a MISSED line and
  # read as a second verdict for a state that had just failed.
  printf '       %-4s sandbox mutant only; the real tracked tree is never written\n' "$id"
}

echo "== MUTATION MATRIX: run_all_checks.sh (sandbox mutant) =="
reset_mutant
patch_literal "$MUTANT" "$A_CLASSIFY" "2) printf 'fail\\n' ;;"
verify_patched N-A
run_state "N-A" "rc=2 no longer classified UNPROVEN" def_n_a "step_ok" "$ALL_CURRENT"
# The converse def_n_e depends on: with the self-check ARMED, a misclassification
# must stop the run BEFORE the steps. Without this assertion, def_n_e ("reached the
# steps") would be satisfied by a healthy run and prove nothing.
if grep -q '1/2 checkers' "$WORK/N-A.log"; then
  echo "  MISSED N-A  the misclassification did NOT gate the run -- the run reached" >&2
  echo "             the steps anyway, so the self-check is not load-bearing and" >&2
  echo "             N-E's predicate would pass vacuously." >&2
  exit 1
fi
echo "  ok    N-A    misclassification gated BEFORE the steps (makes def_n_e sound)"

reset_mutant
patch_literal "$MUTANT" "$A_RERUN" "if false; then"
verify_patched N-B
run_state "N-B" "2b re-run deleted; unproven never cleared" def_n_b "step_flaky2" "$ALL_CURRENT"

reset_mutant
patch_literal "$MUTANT" "$A_FIND" "printf '%s\\n' \"\$STAMP_DIR/a/last_pass.stamp\" \"\$STAMP_DIR/b/last_pass.stamp\""
verify_patched N-C
run_state "N-C" "find replaced by a 2-item list; 3rd stamp invisible" def_n_c "step_ok" "$HAS_STALE"

reset_mutant
patch_literal "$MUTANT" "$A_GUARD" "if false; then"
verify_patched N-D
run_state "N-D" "dirty-tree guard deleted; a step ran over uncommitted work" def_n_d \
  "step_sentinel" "$ALL_CURRENT" "dirty"

reset_mutant
patch_literal "$MUTANT" "$A_CLASSIFY" "2) printf 'fail\\n' ;;"
patch_literal "$MUTANT" "$A_SELFCHK" "return 0  # MUTATION: self-check disarmed"
verify_patched N-E
run_state "N-E" "self-check disarmed while classify_rc is wrong" def_n_e "step_ok" "$ALL_CURRENT"

reset_mutant
patch_literal "$MUTANT" "$A_DERIVE" 'UNPROVEN_LABELS=""'
verify_patched N-F
run_state "N-F" "re-run block present but its set derivation is broken" def_n_f "step_flaky2" "$ALL_CURRENT"

reset_mutant
patch_literal "$MUTANT" "$A_RESIDUE" 'RESIDUE=""'
verify_patched N-G
run_state "N-G" "end-of-run tree-hygiene assertion deleted" def_n_g "step_litter" "$ALL_CURRENT" "" "yes"

# N-H: the exact pre-fix 2b fail arm -- success/failure only, FAILED left at 0.
reset_mutant
patch_literal "$MUTANT" "$A_ESCALATE" "printf 'still rc=%s (%s)\\n' \"\$rc\" \"\$log\" ;;"
verify_patched N-H
run_state "N-H" "2b re-run's rc=1 arm is a no-op (pre-fix shape)" def_n_h "step_escalate" "$ALL_CURRENT"

# N-I: the exact pre-fix residue scope -- one hand-picked directory.
reset_mutant
patch_literal "$MUTANT" "$A_RESIDUE" \
  "RESIDUE=\"\$(git status --porcelain --untracked-files=all -- scripts/ | grep -v '^ M' || true)\""
verify_patched N-I
run_state "N-I" "residue diff narrowed back to scripts/" def_n_i "step_litter" "$ALL_CURRENT" "" "yes" "root"

# N-J: the coverage enumeration neutered; the undispatched dummy must be missed.
reset_mutant
patch_literal "$MUTANT" "$A_COVER" 'grep -qxF "$b" >/dev/null'
verify_patched N-J
printf '#!/usr/bin/env bash\nexit 0\n' >"$DUMMY"; chmod +x "$DUMMY"
git -C "$SB" -c user.email=m@x -c user.name=matrix add -- "$DUMMY" >/dev/null 2>&1
git -C "$SB" -c user.email=m@x -c user.name=matrix commit -q -m dummy -- "$DUMMY" >/dev/null 2>&1
run_state "N-J" "dispatch-coverage enumeration neutered" def_n_j "step_ok" "$ALL_CURRENT"
rm -f "$DUMMY"

rm -f "$MUTANT"
# The synthetic population must never have leaked into the real one, or the real
# enumeration counts fixture stamps as matrix evidence.
if find "$BUILD_DIR" \( -path '*stamps_all_current*' -o -path '*stamps_with_stale*' \) 2>/dev/null | grep -q .; then
  echo "FAIL: synthetic stamp files leaked into the build tree." >&2
  echo "  run_all_checks.sh enumerates '*last_pass.stamp' under BUILD_DIR, so" >&2
  echo "  fixtures living there are counted as matrix evidence." >&2
  exit 1
fi
echo
if [ "$CAUGHT" -eq "$TOTAL" ]; then
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=N-A,N-B,N-C,N-D,N-E,N-F,N-G,N-H,N-I,N-J"
    echo "caught=$CAUGHT/$TOTAL"
    echo "baselines=$NBASE/$NBASE rc+marker correct before mutation"
  } >"$STAMP"
  echo "RUN-ALL MUTATION: PASS ($CAUGHT/$TOTAL caught, $NBASE baselines correct)"
  echo "  run_all_checks.sh can fail: its rc=2 verdict, its 2b re-run (guard,"
  echo "  derivation AND rc=1 escalation), its stamp enumeration, its dirty-tree"
  echo "  guard, its classifier self-check, its end-of-run tree-hygiene diff over"
  echo "  the WHOLE work tree, and its derived dispatch-coverage enumeration."
  echo "  The real repository is never modified: the mutant lives in a throwaway"
  echo "  git repo and each patched state is committed there, so the target's own"
  echo "  dirty-tree guard stays armed against N-D's deliberate dirtiness."
  echo "  stamp: $STAMP"
  exit 0
fi
echo "RUN-ALL MUTATION: FAIL ($CAUGHT/$TOTAL caught)" >&2
echo "  Check that case's baseline first: a state whose baseline is wrong never" >&2
echo "  reproduced its own defect, so it is evidence of nothing." >&2
exit 1
