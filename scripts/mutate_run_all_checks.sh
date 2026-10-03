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
#   N-K  delete the mid-run HEAD-moved detector
#        A second writer committing while this ~45 min stack runs moved HEAD
#        between the start-of-run snapshot and the stamp assertion, so stamps
#        naming the NEW commit were compared against the OLD $HEAD_SHA and
#        reported STALE -- a verdict about a race wearing the costume of a
#        verdict about coverage, with "re-run the matrix" as the wrong remedy.
#        Measured: it invalidated the stamp set three times in one session. The
#        state has a step that commits mid-run, so the move is reproducible.
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
# The mutant SOURCES scripts/_gate_enum.sh (grep rc-classification + the
# anti-vacuity guard in section 0b). Both here AND in reset_mutant, because
# reset_mutant re-commits: populating only one of the two left the sandbox
# dirty after the baselines and the matrix refused to run its states.
for sib in _gate_enum.sh _gate_precond.sh; do
  [ -e "$SCRIPT_DIR/$sib" ] && cp "$SCRIPT_DIR/$sib" "$SB/scripts/$sib"
done
# *** THE SANDBOX NEEDS A POPULATION FOR 0b/0c TO GRADE. *** Measured: the sandbox
# held only run_all_checks.sh and two sourced helpers, so `$SCRIPT_DIR/check_*.sh`
# matched NOTHING and both coverage sections printed their green line over an empty
# set -- "0 check_*/mutate_* script(s), all dispatched". That is the AUDIT #20
# vacuous-harvest class inside the harness that certifies every other checker, and
# it surfaced only because state N-P needed the population to exist. Every real
# check_*/mutate_* is copied in (they are never EXECUTED here -- section 0 stops
# the run), and committed, so the residue diff stays quiet.
cp "$SCRIPT_DIR"/check_*.sh "$SCRIPT_DIR"/mutate_*.sh "$SB/scripts/" 2>/dev/null || true
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
# N-K: a step that COMMITS to the sandbox, i.e. a concurrent writer landing
# mid-run. This is the race that forced three wasted ~45 minute re-runs in one
# session: HEAD moved between the start-of-run snapshot and the stamp assertion,
# so stamps naming the NEW commit compared against the OLD $HEAD_SHA and were
# reported STALE -- a verdict about a race dressed as a verdict about coverage.
# The commit is a NEW untracked file, committed path-limited, so it moves HEAD
# without dirtying the mutant (the dirty-tree guard must stay armed).
printf '#!/usr/bin/env bash\ncd "%s" || exit 1\nprintf "concurrent\\n" > concurrent_writer.txt\ngit -c user.email=m@x -c user.name=matrix add -- concurrent_writer.txt >/dev/null 2>&1\ngit -c user.email=m@x -c user.name=matrix commit -q -m concurrent -- concurrent_writer.txt >/dev/null 2>&1\nexit 0\n' "$SB" >"$STEPS/step_commit"
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
# *** AND FOR A SIBLING FILE, WHICH commit_mutant DOES NOT STAGE. *** The POST-STACK
# states edit the SANDBOX's check_stack_cert.sh, not the mutant. The mutant's own
# start-of-run guard refuses a dirty TRACKED tree, so an uncommitted sibling edit
# aborts the run before 0c exists -- and N-T scored MISSED (unattributed) for
# exactly that reason on its first run. This is the same masking as the N-Q fix, in
# a third place: a sandbox mutation that is not committed stops the CASE, not the
# MUTANT, from being exercised.
commit_sandbox() { # $1 = message, $2 = path (default: the mutant)
  local msg="$1" path="${2:-$MUTANT}"
  git -C "$SB" -c user.email=m@x -c user.name=matrix add -- "$path" >/dev/null 2>&1
  git -C "$SB" -c user.email=m@x -c user.name=matrix commit -q -m "$msg" -- "$path" >/dev/null 2>&1 || true
}
# Every POST-STACK state also needs the SANDBOX's check_stack_cert.sh restored, or
# one state's declaration edit survives into the next.
# *** ORDER MATTERS, AND GETTING IT WRONG POISONS THE NEXT TWO STATES. *** The first
# version copied the pristine file and THEN ran 'git checkout -- .', which
# overwrote it with whatever the PREVIOUS state had committed -- so N-V ran against
# an indented (non-declaring) check_stack_cert.sh and saw no POST-STACK file at all,
# and N-R then inherited a dirty sandbox and failed on the mutant's start guard
# instead of on its lock. Two states, two symptoms, one cause, and neither message
# pointed at it. So: restore the sandbox's tracked state FIRST, copy the pristine
# file over it, then COMMIT that copy so the mutant's dirty-tree guard stays armed.
reset_post_stack() {
  git -C "$SB" -c user.email=m@x -c user.name=matrix checkout -q -- . 2>/dev/null
  if [ -e "$SCRIPT_DIR/check_stack_cert.sh" ]; then
    cp "$SCRIPT_DIR/check_stack_cert.sh" "$SB/scripts/check_stack_cert.sh"
    commit_sandbox reset-post-stack "$SB/scripts/check_stack_cert.sh"
  fi
}
reset_mutant() {
  cp "$TARGET" "$MUTANT"; chmod +x "$MUTANT"
  # The mutant SOURCES scripts/_gate_enum.sh (grep rc-classification +
  # anti-vacuity guard in section 0b), so the sandbox needs that sibling.
  # Without it every baseline died with "_gate_enum.sh: No such file or
  # directory" and the matrix reported a broken CASE rather than its own
  # defect -- the same missing-sibling class that has cost three detours.
  for sib in _gate_enum.sh _gate_precond.sh; do
    [ -e "$SCRIPT_DIR/$sib" ] && cp "$SCRIPT_DIR/$sib" "$SB/scripts/$sib"
  done
  commit_mutant mutant
}
run_rac() { # $1 = case id, $2 = steps, $3 = stamp dir, $4 = leave residue?
            # $5 = "root" to litter the sandbox repo ROOT instead of scripts/
            # $6 = log suffix. A baseline and the state of the same id used to
            # share "$WORK/$id.log", so the STATE overwrote the baseline's log:
            # the check still ran (baseline() greps before the state starts) but
            # the evidence for "the baseline showed the correct marker" was gone
            # by the time anyone looked. Two phases, two files.
  local id="$1" steps="$2" sdir="$3" residue="${4:-}" where="${5:-}" logsuf="${6:-}"
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
    timeout 600 bash "$MUTANT" >"$WORK/$id$logsuf.log" 2>&1
  local rc=$?
  rm -f "$RESIDUE_PROBE" "$ROOT_PROBE"
  return $rc
}

# --- baselines: correct behaviour must be observable BEFORE any mutation ----
# $1 id, $2 steps, $3 stampdir, $4 want rc, $5 marker that MUST be present
#                                       $6 litter  $7 litter site  $8 FORBIDDEN marker
# The forbidden marker is the CONVERSE half of the proof. Every other baseline
# asserts a marker is present, which cannot distinguish a check that fires on the
# right condition from one that ALWAYS fires -- the always-red twin of the vacuous
# check (audit #17). For N-K that mattered: the detector's whole job is to stay
# SILENT when HEAD holds still, and every other baseline in this file runs on a
# tree that never moves, so "the detector is quiet on a healthy run" was asserted
# nowhere. A detector that always reported HEAD MOVED would have passed N-K and
# turned every future stack run red.
baseline() {
  local id="$1" steps="$2" sdir="$3" want="$4" marker="$5" residue="${6:-}" rc
  local where="${7:-}"   # litter site: "" = scripts/, "root" = sandbox repo root
  local forbidden="${8:-}"
  NBASE=$((NBASE + 1))
  reset_mutant
  run_rac "$id" "$steps" "$sdir" "$residue" "$where" ".baseline"; rc=$?
  if [ "$rc" != "$want" ]; then
    echo "FAIL: baseline $id reached rc=$rc, want $want -- the CASE is broken, not" >&2
    echo "  the mutation. ($WORK/$id.baseline.log)" >&2; exit 1
  fi
  if [ -n "$marker" ] && ! grep -qF -- "$marker" "$WORK/$id.baseline.log"; then
    echo "FAIL: baseline $id did not show the CORRECT marker '$marker'." >&2
    echo "  Without this a predicate can be satisfied by an early abort and score" >&2
    echo "  a false catch. ($WORK/$id.baseline.log)" >&2; exit 1
  fi
  # The baseline's proof is that the CORRECT behaviour is observable -- for the
  # residue states that includes WHICH file was littered, so a baseline that
  # quietly littered scripts/ (as an earlier version of baseline() did, by
  # accepting a litter-site argument and never forwarding it) is visibly
  # indistinguishable from N-G. The marker cannot express that, so the litter
  # site is echoed rather than left in the reader's imagination.
  # Converse assertion: this marker must be ABSENT on a correct run.
  if [ -n "$forbidden" ] && grep -qF -- "$forbidden" "$WORK/$id.baseline.log"; then
    echo "FAIL: baseline $id emitted '$forbidden', which must be ABSENT." >&2
    echo "  A check that fires on every run is an always-red gate, not a" >&2
    echo "  detector. ($WORK/$id.baseline.log)" >&2; exit 1
  fi
  local lit="(none)"
  [ -n "$residue" ] && lit="$([ "$where" = root ] && echo "sandbox ROOT" || echo "scripts/")"
  printf '  ok    baseline %-4s rc=%s, correct marker present, litter=%s%s\n' \
    "$id" "$rc" "$lit" "$([ -n "$forbidden" ] && echo ', converse marker absent' || true)"
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
# N-K: the mid-run HEAD-change detector deleted. Predicate is POSITIVE ("this
# specific evidence is ABSENT"), and the converse is asserted by the baseline: a
# run WITH the detector prints the FAIL line and exits 1, so the state cannot be
# scored by a top-of-run abort.
def_n_k() { ! grep -q 'HEAD MOVED under this run' "$WORK/N-K.log"; }

# N-H/N-I/N-J need their litter/counter state cleared before each run, which
# run_rac does; N-H also needs its own counter file reset because a leftover
# count would make the first pass answer rc=1 (a plain failure) and never reach
# the re-run. run_rac removes it, which is asserted by the N-H baseline marker.

# --- anchor pre-flight (checked against the PRISTINE target) ---------------
# Counted with the SAME mechanism patch_literal uses, deliberately. Two earlier
# versions of this helper were wrong in ways that mattered:
#   * `grep -cF ... || echo 0` -- grep -c prints "0" AND exits 1 on no match, so
#     every miss reported "found 0\n0 times".
#   * `grep -cF` itself counts LINES, and one anchor (A_RESIDUE) is a four-line
#     continuation, so it reported 4 occurrences and refused to start.
# The pre-flight and the patch must agree on what "exactly once" means, or the
# guard and the thing it guards drift -- so both count SUBSTRING occurrences in
# python, here and in patch_literal below.
check_anchor() { # $1 = literal, $2 = description
  local n
  n="$(python3 -c 'import sys
src = open(sys.argv[1]).read()
print(src.count(sys.argv[2]))' "$TARGET" "$1" 2>/dev/null || echo COUNT-FAILED)"
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
# The WHOLE four-line block, not just its first line: the first attempt anchored
# on line 1 alone, the patch replaced it with a one-liner, and the three
# continuation lines were left dangling -- verify_patched aborted with a syntax
# error, correctly, before the state could score.
A_RESIDUE='RESIDUE="$(LC_ALL=C comm -13 \
             <(LC_ALL=C sort "$UNTRACKED_BEFORE") \
             <(git status --porcelain --untracked-files=all | LC_ALL=C sort) \
           | grep -v '"'"'^ M'"'"' || true)"'
# A three-LINE block, not the announcement line alone: the replacement has to
# absorb the two statements that follow it, or they are left dangling and
# verify_patched aborts on a syntax error (which is correct, and is why the
# state failed loudly rather than scoring a false catch).
A_ESCALATE="printf 'REGRESSED rc=%s (%s) -- was unproven, is now a real failure\\n' \"\$rc\" \"\$log\"
          UNPROVEN=\$((UNPROVEN - 1))
          FAILED=\$((FAILED + 1)) ;;"
# REPOINTED: the arm lookup now classifies grep's rc (1 = no match,
# >=2 = the query failed) instead of collapsing both into "no dispatch
# arm", so the old unclassified anchor no longer exists. The pre-flight
# caught that, which is exactly what it is for.
A_COVER='elif [ "$grc" -eq 1 ]; then'
# N-K: the WHOLE detector block -- guard, assignment, report, increment and the
# closing fi. Removing only the announcement leaves the increment dangling;
# removing only the guard leaves a detector that never fires. Both are
# harness-reported failures, not a scored state, so the anchor must cover the
# unit that can actually be removed as ONE piece. (First version stopped at
# "HEAD_MOVED=1" and orphaned the fi; verify_patched caught the syntax error,
# which is the mechanism working.)
#
# Built by HEREDOC, not by wrapping the block in single quotes. The block
# contains `'STALE'`, and hand-escaping those quotes inside a single-quoted
# shell literal produced a syntax error in THIS script -- the same
# anchor-covers-too-much failure as above, one level out. A quoted heredoc has no
# escaping rules to get wrong, so the literal is exactly the target's bytes.
read -r -d '' A_HEADMOVE <<'A_HEADMOVE_EOF' || true
HEAD_MOVED=0
if [ "$HEAD_NOW" != "$HEAD_SHA" ]; then
  HEAD_MOVED=1
  echo "== 2c HEAD MOVED under this run =="
  echo "  FAIL  HEAD changed during the run: started at $HEAD_SHORT, now ${HEAD_NOW:0:7}." >&2
  echo "         Another writer committed to this clone while the stack was running." >&2
  echo "         The stamps just written name the NEW commit, so 'STALE' below would" >&2
  echo "         be an artifact of this race, not missing evidence -- and re-running" >&2
  echo "         is NOT the fix. Either finish the concurrent commit first (single-" >&2
  echo "         writer convention, AGENTS.md) or re-run now that the tree is settled." >&2
  FAILED=$((FAILED + 1))
fi
A_HEADMOVE_EOF
check_anchor "$A_CLASSIFY" "N-A: the rc=2 UNPROVEN classification"
check_anchor "$A_RERUN"   "N-B: the 2b re-run guard"
check_anchor "$A_FIND"    "N-C: the find-based stamp enumeration"
check_anchor "$A_GUARD"   "N-D: the dirty-tree guard"
check_anchor "$A_SELFCHK" "N-E: the self-check verdict"
check_anchor "$A_DERIVE"  "N-F: the re-run set derivation"
check_anchor "$A_RESIDUE" "N-G: the tree-hygiene residue diff"
check_anchor "$A_ESCALATE" "N-H: the 2b re-run's REGRESSED announcement"
check_anchor "$A_COVER"   "N-J: the dispatch-coverage enumeration"
check_anchor "$A_HEADMOVE" "N-K: the mid-run HEAD-moved detector"
echo "anchor pre-flight: ok (10/10 targets located exactly once)"
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
# N-K baseline: a step commits mid-run, so the DETECTOR must fire. rc=1, and the
# marker must be the move notice -- without it, N-K's predicate is satisfied by
# any run that happens to not print that line, including one that aborted early.
baseline "N-K" "step_ok step_commit" "$ALL_CURRENT" 1 "HEAD MOVED under this run"
# N-K converse: with NO committing step, HEAD holds still, so the detector must
# stay SILENT and the run must reach PASS. This is the half that makes N-K a
# detector rather than an always-red gate, and every other baseline in this file
# runs on a tree that never moves -- so without this line "the detector is quiet
# on a healthy run" was asserted nowhere. A detector hard-wired to always report a
# move would pass N-K and turn every future stack run red.
# `rm -f` alone leaves a DELETION in the sandbox's working tree, so every
# subsequent state aborted at the top on the mutant's own dirty-tree guard and
# scored MISSED -- four states (N-A, N-B, N-E, N-F) disarmed by one stray `rm`.
# The removal is committed, and the sandbox-clean assertion below catches that
# class independently of this comment.
rm -f "$DUMMY"
git -C "$SB" -c user.email=m@x -c user.name=matrix commit -q -m "drop dummy" -- "$DUMMY" >/dev/null 2>&1
baseline "N-Kq" "step_ok" "$ALL_CURRENT" 0 "" "" "HEAD MOVED under this run"

# The baselines are where the sandbox is set up, so this is the last point at
# which a stray staged change can be seen before the states start aborting on
# it. Derived (git status), not a list of files to remember to clean.
if [ -n "$(git -C "$SB" status --porcelain)" ]; then
  echo "FAIL: the sandbox git repo is dirty after the baselines." >&2
  echo "  Every state below would abort at the top on the mutant's own" >&2
  echo "  dirty-tree guard and score MISSED -- a broken HARNESS that reads as" >&2
  echo "  ten working states." >&2
  git -C "$SB" status --porcelain >&2
  exit 1
fi
echo "sandbox hygiene: ok (clean after $NBASE baselines)"


# Exact literal replacement -- no delimiter that can occur inside the pattern
# (audit #17b: a sed delimiter inside an ERE alternation killed a state, and a
# state that silently does nothing is reported as "defect not reproduced").
patch_literal() { # $1 = file, $2 = old, $3 = new -- delegates, then commits
  patch_literal_rc "$1" "$2" "$3" || return 1
  # Commit the patch. In the sandbox the mutant is a TRACKED file, so patching it
  # leaves the tree dirty -- and the mutant's own start-of-run guard would then
  # abort every case before the defect could appear. That is precisely the masking
  # that broke the FIRST version of this matrix, reappearing one level down; the
  # sandbox removes it only if each patched state is committed.
  commit_mutant patched
}
# *** AND THE FUNCTION NOW FAILS CLOSED. *** Its last command was commit_mutant, so
# it exited with COMMIT's status and python's non-zero exit was discarded.
# MEASURED consequence: state N-P's anchor drifted (a later STEPS edit appended
# another matrix name), patch_literal refused correctly -- "anchor found 0 times,
# want 1" -- and the `|| fail` guard NEVER FIRED. The run continued and N-P reported
# "defect NOT reproduced", which reads as "this state is weak" and is completely
# wrong: nothing had been mutated at all.
#
# *** THAT AFFECTED EVERY STATE IN THIS MATRIX, not just N-P: *** a silent patch
# failure anywhere here was reported as an un-reproduced defect. This is the
# AGENTS.md patch_literal-duplication divergence TRIGGER firing for real -- the
# recorded reason to keep the copies was that each is exercised whenever its matrix
# runs, which presumes they behave like the shared helper. This one did not. It
# keeps its own copy (it must commit after patching, which the shared helper does
# not) and now fails closed like the shared one.
patch_literal_rc() { # $1 = file, $2 = old, $3 = new -- NO commit, honest rc
  python3 - "$1" "$2" "$3" <<'PYX'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
if n != 1:
    sys.exit("patch_literal: anchor found %d times, want 1" % n)
open(path, "w").write(src.replace(old, new))
PYX
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
# The converse, asserted in BOTH directions on purpose. The first version only
# checked that '1/2 checkers' was ABSENT -- which a top-of-run abort (a dirty
# sandbox, a bad anchor) satisfies trivially. It printed 'ok' while four states
# were scoring MISSED for exactly that reason. Requiring the self-check to have
# actually RUN first pins the absence to the gating decision and nothing else.
if ! grep -q 'SELFCHECK' "$WORK/N-A.log"; then
  echo "  ABORT N-A  the mutant never reached the self-check, so the converse" >&2
  echo "           assertion below would be satisfied by an early abort." >&2
  exit 1
fi
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
patch_literal "$MUTANT" "$A_ESCALATE" "printf 'still rc=%s (%s) -- still unproven\\n' \"\$rc\" \"\$log\" ;;"
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
git -C "$SB" -c user.email=m@x -c user.name=matrix commit -q -m "drop dummy" -- "$DUMMY" >/dev/null 2>&1

# N-K: the mid-run HEAD-moved detector deleted. The concurrent commit must then go
# UNREPORTED -- the run silently compares stamps against the commit it started on,
# which is the race this state exists to prove is now loud.
reset_mutant
patch_literal "$MUTANT" "$A_HEADMOVE" 'HEAD_MOVED=0  # detector removed; move now silent'
verify_patched N-K
run_state "N-K" "mid-run HEAD-moved detector deleted" def_n_k "step_ok step_commit" "$ALL_CURRENT"

# ---------------------------------------------------------------------------
# N-P: SECTION 0c (every check_*/mutate_* is SCHEDULED) NEVER RUNS. *** THE STATE
# THAT PROVES 0c IS A DETECTOR RATHER THAN AN ASSUMPTION. ***
#
# 0b only compares against dispatch(), and STEPS -- the list the run ITERATES -- was
# hand-maintained beside it. Measured 2026-10-02: a checker and a matrix were added
# with their arms, 0b reported "all dispatched", the matrix NEVER RAN, and step 3/3
# reported its stamp STALE, which reads as "the evidence is older than the tree"
# with "re-run the named matrix" as the remedy. Re-running changes nothing.
#
# *** IT NEEDS ITS OWN RUNNER, because run_rac always sets INTFER_RUN_ALL_STEPS --
# which is exactly the variable that makes 0c SKIP. *** The state therefore drives
# the mutant with INTFER_RUN_ALL_COVERAGE_ONLY=1 and NO steps override, so 0c runs
# for real against the sandbox's scripts/ and the run stops after section 0c (no
# step executes, and no stamp dir is needed). That seam is the reason 0c is
# testable at all: proving it costs seconds instead of a 45-minute stack.
run_rac_cov() { # $1 = case id, $2 = log suffix
  INTFER_BUILD_DIR="$SYNTH/inner" INTFER_RUN_ALL_COVERAGE_ONLY=1 \
    timeout 300 bash "$MUTANT" >"$WORK/$1$2.log" 2>&1
}

# The baseline is the CONVERSE half, and it is required: without it a state that
# requires "0c printed FAIL" could be scored by a run that aborts before 0c. It
# must show 0c's OK line, i.e. that 0c stays SILENT on a healthy tree.
NBASE=$((NBASE + 1))
reset_mutant
run_rac_cov "N-P" ".baseline"; npb=$?
if [ "$npb" -ne 0 ] \
   || ! grep -q 'every check_\*/mutate_\* is in STEPS as well as dispatch()' \
        "$WORK/N-P.baseline.log" \
   || grep -q 'never scheduled' "$WORK/N-P.baseline.log" \
   || grep -qF 'no check_*/mutate_* files were found' "$WORK/N-P.baseline.log" \
   || grep -qF 'over an empty set proves nothing' "$WORK/N-P.baseline.log"; then
  echo "FAIL: baseline N-P rc=$npb -- 0c must be SILENT and green on a healthy" >&2
  echo "  tree, or the state below cannot distinguish '0c fired' from 'the run" >&2
  echo "  died'. ($WORK/N-P.baseline.log)" >&2; exit 1
fi
echo "  ok    baseline N-P: 0c is silent and rc=0 on a healthy tree"
echo "  ok    CONVERSE for N-Q: the EMPTY-CENSUS guards stayed quiet on a"
echo "        populated census (asserted as ABSENCE above, not just rc=0)"

# RE-POINTED (AUDIT #29 round 4): this anchor ENDED the STEPS matrices line, and
# appending mutate_stack_cert.sh moved the end. The refusal was CORRECT -- and the
# guard that should have surfaced it was masked by the patch_literal return-value
# defect above, so N-P reported itself as a weak state instead of a stale anchor.
# Re-point, never loosen -- and PRE-FLIGHT it, so a future drift is caught in the
# first second instead of after thirteen baselines.
A_STEPS_ENTRY='mutate_checker_matrix_coverage.sh mutate_verdict_scope.sh mutate_stack_cert.sh"'
check_anchor() { # $1 file, $2 anchor -- exactly once, not merely present
  local n
  n="$(python3 -c 'import sys;print(open(sys.argv[1]).read().count(sys.argv[2]))' "$1" "$2")"
  [ "$n" = "1" ] && return 0
  echo "ANCHOR STALE in $(basename "$1"): found $n occurrence(s), want 1." >&2
  echo "  Re-point the state; do NOT loosen the check." >&2
  return 1
}
check_anchor "$TARGET" "$A_STEPS_ENTRY" || {
  echo "PREFLIGHT FAIL -- no state was run and nothing was mutated." >&2; exit 1; }
echo "  ok    anchor pre-flight: the N-P STEPS anchor occurs exactly once"
reset_mutant
patch_literal "$MUTANT" "$A_STEPS_ENTRY" 'mutate_checker_matrix_coverage.sh"' || {
  echo "N-P MISSED (fn) -- the STEPS anchor did not match" >&2; exit 1; }
# The arm must SURVIVE, or this state would be N-J again (a missing arm) and
# would prove nothing about scheduling. Assert it rather than assume it.
grep -q '^    mutate_verdict_scope.sh)' "$MUTANT" \
  || { echo "N-P MISSED (fn) -- the dispatch arm is gone; this is N-J, not 0c" >&2; exit 1; }
grep -q 'N-P' "$MUTANT" || true
TOTAL=$((TOTAL + 1))
run_rac_cov "N-P" ""
# *** THE PREDICATE GREPS FOR THE NAME, NOT FOR ITS POSITION IN THE LIST. *** The
# first version matched the whole line including the name, and 0c prints the
# unscheduled scripts in GLOB order, so the message reads
# "...: mutate_stack_cert.sh mutate_verdict_scope.sh" -- the name is present but not
# where the exact-match string expected. That is the AGENTS.md rule one level down
# (write the predicate from what is ACTUALLY printed, not from what you expect),
# and it is also why the two halves are asserted separately: the header proves 0c
# refused an unscheduled step, and the name proves it was THIS step.
if [ "$?" -ne 0 ] \
   && grep -q 'never scheduled, so this run will NOT execute:' "$WORK/N-P.log" \
   && grep -q 'mutate_verdict_scope.sh' "$WORK/N-P.log"; then
  CAUGHT=$((CAUGHT + 1))
  printf '  ok    %-4s defect reproduced: a matrix with an ARM but no STEPS entry never runs\n' "N-P"
else
  printf '  MISSED %-4s defect NOT reproduced: 0c accepted an unscheduled matrix\n' "N-P"
  printf '       %-4s (rc and marker are both required; see %s)\n' "N-P" "$WORK/N-P.log"
fi
printf '       %-4s sandbox mutant only; the real tracked tree is never written\n' "N-P"
reset_mutant

# ---------------------------------------------------------------------------
# N-Q: THE EMPTY-CENSUS GUARD MUST FIRE. *** THE STATE THAT KEEPS A FIXED BRANCH
# FROM SHIPPING UNEXERCISED. ***
#
# N-P DISCOVERED that both coverage sections printed their green line over an empty
# population ("0 check_*/mutate_* script(s), all dispatched"), and the fix did TWO
# things: added the guards, and gave the sandbox a real scripts/ so the scenario
# stops happening. *** The second half made the new guards unreachable. *** A guard
# nothing can exercise is the dead-branch class this repo audits constantly, and it
# would have shipped with a plausible-looking message and no proof it works.
#
# So the scenario is rebuilt deliberately, as an ARTIFACT defect rather than a
# mutation: a checkout whose scripts/ holds ONLY the mutant and the two helpers it
# sources. SCRIPT_DIR is derived from $0, so running the mutant from THAT directory
# gives 0b and 0c an empty census -- the exact condition they must refuse.
# Committed, not mutated: the defect is the tree, and there is no patch to apply.
#
# *** AND IT MUST LIVE INSIDE THE SANDBOX GIT REPO. *** The first version put it at
# $SYNTH/empty/scripts, which is not a work tree, so the harness's dirty-tree guard
# died at its own `git diff --quiet` and the state scored rc=1 for a reason that had
# nothing to do with the census -- a state failing for an unrelated cause, which the
# predicate below correctly refused to score as a catch. ROOT is dirname(SCRIPT_DIR),
# so the directory has to be a checkout; $SB/emptysb is inside the sandbox repo, and
# UNTRACKED files there are fine (the guard rejects tracked changes only).
#
# *** BOTH GUARDS ARE REQUIRED, not just one. *** 0b harvests the per-script
# population and 0c harvests the scheduled one; they are separate loops with
# separate counters, and fixing only the first is exactly what happened in the
# first commit. N-P's baseline is the converse -- the same two guards must stay
# SILENT on a populated census -- and it now asserts that as ABSENCE, so neither
# half of this can be satisfied by a guard that always fires.
EQ="$SB/emptysb/scripts"
mkdir -p "$EQ"
cp "$MUTANT" "$EQ/run_all_checks.sh"; chmod +x "$EQ/run_all_checks.sh"
for sib in _gate_enum.sh _gate_precond.sh; do
  [ -e "$SCRIPT_DIR/$sib" ] && cp "$SCRIPT_DIR/$sib" "$EQ/$sib"
done
TOTAL=$((TOTAL + 1))
INTFER_BUILD_DIR="$SYNTH/inner3" INTFER_RUN_ALL_COVERAGE_ONLY=1 \
  timeout 300 bash "$EQ/run_all_checks.sh" >"$WORK/N-Q.log" 2>&1
nq_rc=$?
nq_0b=0; nq_0c=0
grep -qF 'no check_*/mutate_* files were found' "$WORK/N-Q.log" && nq_0b=1
grep -qF 'over an empty set proves nothing' "$WORK/N-Q.log" && nq_0c=1
if [ "$nq_rc" -ne 0 ] && [ "$nq_0b" -eq 1 ] && [ "$nq_0c" -eq 1 ]; then
  CAUGHT=$((CAUGHT + 1))
  printf '  ok    %-4s defect reproduced: an EMPTY census is refused by 0b AND 0c\n' "N-Q"
  printf '       %-4s rc=%s, both guards named the empty set (not just one)\n' "N-Q" "$nq_rc"
else
  printf '  MISSED %-4s defect NOT reproduced: rc=%s 0b-guard=%s 0c-guard=%s\n' \
    "N-Q" "$nq_rc" "$nq_0b" "$nq_0c"
  printf '       %-4s both guards are required; see %s\n' "N-Q" "$WORK/N-Q.log"
fi

# ---------------------------------------------------------------------------
# N-T / N-U: THE POST-STACK EXEMPTION MUST NOT BE AVAILABLE BY ACCIDENT. *** A
# MECHANISM THAT DECIDES WHO MAY BE UNSCHEDULED, WITH NOTHING TESTING IT. ***
#
# Section 0c exempts a script from the STEPS requirement when its own header
# carries `# POST-STACK:` within the FIRST 40 LINES. That marker is a comment, so it
# cannot be verified by running the checker on its normal fixtures -- and the whole
# point of the window is to reject a mention that does not count. Two states:
#
#   N-T  the declaration is pushed PAST line 40 -> the exemption must NOT apply, so
#         0c must demand the file be scheduled (rc=1).
#   N-U  the declaration survives but is demoted to a BODY mention (indented, so the
#         anchor `^# POST-STACK:` cannot match) -> the exemption must NOT apply
#         either. This is the "a comment that merely mentions it" case, the same
#         idea as check_bench_load_gate's case (7c) and check_verdict_scope's (d).
#
# *** AND THE CONVERSE IS N-P's baseline plus the assertion below: *** with the
# declaration intact, 0c must print the exemption and stay green. Without it, N-T
# and N-U would both be satisfied by a checker that exempts NOTHING and therefore
# demands everything -- an always-red exemption is the mirror defect.
CSC="$SB/scripts/check_stack_cert.sh"
conj_run() { INTFER_BUILD_DIR="$SYNTH/inner" INTFER_RUN_ALL_COVERAGE_ONLY=1 \
               timeout 300 bash "$MUTANT" >"$WORK/$1" 2>&1; }
# The converse, asserted first: the declaration as shipped must be honoured.
reset_mutant
conj_run "N-T.converse.log"; cv=$?
if [ "$cv" -eq 0 ] && grep -q 'POST-STACK, exempt from STEPS' "$WORK/N-T.converse.log" \
   && ! grep -q 'never scheduled' "$WORK/N-T.converse.log"; then
  echo "  ok    baseline N-T: a header declaration IS honoured, and 0c stays green"
  nb_conv=1
else
  echo "FAIL: baseline N-T rc=$cv -- an in-window declaration must exempt the file." >&2
  echo "  Without this, N-T and N-U are also satisfied by an exemption that never" >&2
  echo "  fires (which would demand every file be scheduled). ($WORK/N-T.converse.log)" >&2
  exit 1
fi
# A helper that moves the declaration below the window by padding above it.
pad_above_decl() { # $1 = how many filler comment lines to insert
  python3 - "$CSC" "$1" <<'PYX'
import sys
p, n = sys.argv[1], int(sys.argv[2])
lines = open(p).read().split("\n")
out = []
for l in lines:
    if l.startswith("# POST-STACK:") and not any(x.startswith("# POST-STACK:") for x in out):
        out.extend(["# filler %d" % i for i in range(n)])
    out.append(l)
open(p, "w").write("\n".join(out))
PYX
}
run_state_post() { # $1 id, $2 desc, $3 expected-RE
  local id="$1" desc="$2" expect="$3" o r
  TOTAL=$((TOTAL + 1)); reset_mutant
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed (STATE fault)"; reset_mutant; reset_post_stack; return 1
  fi
  conj_run "$id.log"; r=$?
  o="$(cat "$WORK/$id.log")"
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- 0c accepted a declaration it must not honour ($desc)"
    echo "$id MISSED (rc=0)" >> "$WORK/summary.txt"; reset_mutant; reset_post_stack; return 1
  fi
  if ! printf '%s' "$o" | grep -qE "$expect"; then
    echo "  $id MISSED -- failed for the wrong reason (unattributed)"; sed 's/^/        /' "$WORK/$id.log"
    echo "$id MISSED (unattributed)" >> "$WORK/summary.txt"; reset_mutant; reset_post_stack; return 1
  fi
  echo "  $id caught -- rc=$r, marker matched: $desc"
  echo "$id caught" >> "$WORK/summary.txt"; CAUGHT=$((CAUGHT + 1)); reset_mutant; reset_post_stack; return 0
}
# N-T: pushed past line 40.
MUTATE_FN() { pad_above_decl 45; grep -q '^# POST-STACK:' "$CSC" || return 1; commit_sandbox post-stack "$CSC"; }
run_state_post "N-T" "the declaration sits past the 40-line window" \
  'never scheduled.*check_stack_cert|declares POST-STACK'
# N-U: demoted to an indented body mention.
MUTATE_FN() { sed -i 's|^# POST-STACK: run this AFTER|  # POST-STACK: run this AFTER|' "$CSC" || return 1; commit_sandbox post-stack "$CSC"; }
run_state_post "N-U" "the declaration is an indented body mention" \
  'never scheduled.*check_stack_cert|declares POST-STACK'

# N-V: THE EXEMPTION IS SELF-POLICING -- a POST-STACK file must NOT also be in
# STEPS, because that is the exact wiring that made the stack FAIL. The converse is
# that adding it back to STEPS while keeping the declaration must be refused.
reset_mutant
python3 - "$MUTANT" <<'PYX'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'check_doc_counts.sh check_verdict_scope.sh"'
new = 'check_doc_counts.sh check_verdict_scope.sh check_stack_cert.sh"'
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
PYX
commit_mutant n-v
conj_run "N-V.log"; nv=$?
if [ "$nv" -ne 0 ] && grep -q 'declares POST-STACK \*and\* is in STEPS' "$WORK/N-V.log"; then
  TOTAL=$((TOTAL + 1)); CAUGHT=$((CAUGHT + 1))
  echo "  N-V caught -- rc=$nv: an EXEMPT-AND-SCHEDULED file is refused"
  echo "N-V caught" >> "$WORK/summary.txt"
else
  TOTAL=$((TOTAL + 1))
  echo "  N-V MISSED -- rc=$nv; a file that is both exempt and scheduled was accepted"
  echo "N-V MISSED" >> "$WORK/summary.txt"
fi
reset_mutant

# ---------------------------------------------------------------------------
# N-R / N-S: THE STALE-LOCK SELF-HEAL, IN BOTH DIRECTIONS. *** AN UNTESTED
# SELF-HEAL IS AN UNCONDITIONAL DELETE WITH EXTRA STEPS. ***
#
# MEASURED HAZARD, twice this session: a run killed mid-stack left the lock behind,
# and the documented remedy was `rm -f $LOCK`, which will happily delete a LIVE
# run's lock and let a second stack restore another run's mutations over its own.
# The lock now records its owner's pid and self-heals only after `kill -0` proves
# the owner is dead.
#
# Both directions are states, and both plant the LOCK rather than patching anything
# -- the defect is the filesystem, not the text. N-R plants a DEAD owner and
# requires the self-heal; N-S plants a LIVE owner (this very matrix process) and
# requires a REFUSAL. N-S is the half that matters: without it, a self-heal that
# fires unconditionally would score as a pass and delete a live run's lock.
LOCKP="$SYNTH/lockprobe/.run_all_checks.lock"
cov_run() { # $1 = log name  -> runs the mutant with COVERAGE_ONLY in a probe dir
  INTFER_BUILD_DIR="$SYNTH/lockprobe" INTFER_RUN_ALL_COVERAGE_ONLY=1 \
    timeout 300 bash "$MUTANT" >"$WORK/$1" 2>&1
}

TOTAL=$((TOTAL + 1)); reset_mutant
rm -rf "$SYNTH/lockprobe"; mkdir -p "$LOCKP"; echo 999999 > "$LOCKP/pid"
cov_run "N-R.log"; nr_rc=$?
if [ "$nr_rc" -eq 0 ] && grep -qF 'STALE LOCK' "$WORK/N-R.log"; then
  CAUGHT=$((CAUGHT + 1))
  printf '  ok    %-4s defect reproduced: a lock whose owner is DEAD self-heals\n' "N-R"
else
  printf '  MISSED %-4s defect NOT reproduced: rc=%s (a stale lock must be taken over)\n' \
    "N-R" "$nr_rc"
fi

TOTAL=$((TOTAL + 1)); reset_mutant
rm -rf "$SYNTH/lockprobe"; mkdir -p "$LOCKP"; echo $$ > "$LOCKP/pid"
cov_run "N-S.log"; ns_rc=$?
if [ "$ns_rc" -eq 1 ] && grep -qF 'another run_all_checks.sh holds' "$WORK/N-S.log" \
   && ! grep -qF 'STALE LOCK' "$WORK/N-S.log"; then
  CAUGHT=$((CAUGHT + 1))
  printf '  ok    %-4s defect reproduced: a lock whose owner is LIVE is REFUSED, not taken\n' "N-S"
else
  printf '  MISSED %-4s defect NOT reproduced: rc=%s (a LIVE lock must refuse; deleting it is\n' \
    "N-S" "$ns_rc"
  printf '       %-4s the one action that corrupts a concurrent run'"'"'s tree)\n' "N-S"
fi
rm -rf "$SYNTH/lockprobe"

# N-M: a dispatch() arm given a STRAY ARGUMENT. This is the class the full
# stack exposed on 2026-10-01: a STEPS-line edit that added a matrix name also
# matched inside the mutate_gate_pop.sh dispatch arm, so it ran as
#   bash $SCRIPT_DIR/mutate_gate_pop.sh mutate_checker_matrix_coverage.sh
# Every DIRECT invocation of that matrix passed; only the end-to-end stack
# failed (rc=127 plus a stale stamp). A matrix author fixing it by eye is not a
# guard -- the same edit shape recurs.
#
# Under this mutation a dispatch arm is handed an extra word. Nothing about the
# arm's logic changes, so 0b's coverage enumeration and step 3/3 must not notice:
# the arm exists and is dispatched either way. What must catch it is the arm
# EXECUTING, which dies with the argument as a path. So the observable is a
# non-zero run with the "No such file or directory" shape -- i.e. the harness
# already fails loudly, and this state proves it rather than assuming it.
reset_mutant
patch_literal "$MUTANT" \
  'timeout 1800 bash "$SCRIPT_DIR/mutate_gate_pop.sh"; return $? ;;' \
  'timeout 1800 bash "$SCRIPT_DIR/mutate_gate_pop.sh" stray_arg; return $? ;;' \
  && verify_patched N-M

# The predicate is POSITIVE: the stray-argument run must FAIL, and it must fail
# with the path-shaped error rather than for an unrelated reason. Asserting only
# "rc != 0" would let this pass because the matrix broke some other way.
# The predicate had to be written from what the STACK LOG actually contains.
# My first version grepped for "No such file or directory" -- which is printed by
# the STEP's own log, not by the stack, so the state reported MISSED even though
# the defect reproduced perfectly (step_current FAIL rc=97, and the run ended
# "FAIL (1 step(s) FAILED)"). Write the predicate from the observed output, not
# from the output you expect (the same lesson as M-A's polarity).
def_n_m() { grep -qE 'step_current +FAIL rc=|RUN ALL CHECKS: FAIL \(1 step' "$WORK/N-M.log"; }
run_state "N-M" "dispatch() arm given a stray argument" def_n_m "step_ok step_current" "$ALL_CURRENT"

# N-L: the detector hard-wired to ALWAYS fire. This state exists so the N-Kq
# converse baseline is not another vacuous check -- without it, "the detector
# stays silent on a healthy run" is a line nobody has ever seen fail. Under this
# mutation the N-Kq baseline (no committing step, HEAD still) MUST report the
# forbidden marker and abort; N-K itself is unaffected, since an always-firing
# detector still catches the real move. That asymmetry is the point: N-K alone
# cannot tell a detector from an alarm, only N-L can.
#
# Patched by forcing the GUARD to true rather than deleting the block, because
# deleting it is N-K. Same always-red twin, opposite direction.
reset_mutant
patch_literal "$MUTANT" 'if [ "$HEAD_NOW" != "$HEAD_SHA" ]; then' \
  'if [ "$HEAD_NOW" != "$HEAD_SHA" ] || true; then  # N-L: always fire'
verify_patched N-L
# run_rac's return code is the MUTANT's rc and says nothing about whether the
# converse REJECTED it, so the evidence is the LOG: an always-firing detector must
# print the notice, which is precisely what N-Kq's forbidden-marker check
# consumes. Assert on that, not on a rc that a direct call cannot carry.
# NOTE the argument shape: run_rac reads $6 as the log suffix, and $4/$5 are
# the residue flags. Passing 0 "" "" ".baseline" would put ".baseline" in $7 and
# write N-L.log instead -- which is exactly what the first version of this state
# did, and the missing N-L.baseline.log is how it surfaced. Six args, not seven.
run_rac "N-L" "step_ok" "$ALL_CURRENT" "" "" ".baseline" >/dev/null
if ! grep -q 'HEAD MOVED under this run' "$WORK/N-L.baseline.log"; then
  echo "FAIL: the N-L converse baseline did not REJECT an always-firing detector." >&2
  echo "  An always-firing detector must make N-Kq's forbidden-marker check abort." >&2
  echo "  Note this calls run_rac directly, NOT baseline(): baseline() exits the" >&2
  echo "  whole matrix on a forbidden marker, which is correct for N-Kq (a real" >&2
  echo "  defect in the CASE) but wrong here, where the always-firing mutant IS" >&2
  echo "  the thing under test and the rejection is the expected outcome." >&2
  echo "  ($WORK/N-L.baseline.log)" >&2
  exit 1
fi
echo "  ok    baseline N-L   converse REJECTED an always-firing detector (as it must)"

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
    echo "states=N-A,N-B,N-C,N-D,N-E,N-F,N-G,N-H,N-I,N-J,N-K,N-M"
    echo "caught=$CAUGHT/$TOTAL"
    echo "baselines=$NBASE/$NBASE rc+marker correct before mutation"
  } >"$STAMP"
  echo "RUN-ALL MUTATION: PASS ($CAUGHT/$TOTAL caught, $NBASE baselines correct)"
  echo "  run_all_checks.sh can fail: its rc=2 verdict, its 2b re-run (guard,"
  echo "  derivation AND rc=1 escalation), its mid-run HEAD-moved detector (fires on a"
   echo "  move AND proven silent when HEAD holds still -- N-K/N-Kq/N-L), its"
  echo "  stamp enumeration, its dirty-tree guard, its classifier self-check, its"
  echo "  end-of-run tree-hygiene diff over the WHOLE work tree, and its derived"
  echo "  dispatch-coverage enumeration."
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
