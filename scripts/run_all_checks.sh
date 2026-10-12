#!/usr/bin/env bash
# One command for the whole gate-hygiene verification stack, ending in a stamp
# assertion.
#
# WHY. "All stamps equal HEAD" is a standard this project holds itself to, and
# until now satisfying it was a SESSION RITUAL: a human re-ran three checkers and
# three mutation matrices serially, in the right order, and then read four stamp
# files. That is exactly the "re-derive by enumeration, never from a remembered
# list" rule, performed by memory, and it fails silently in the two ways that
# matter: a skipped matrix leaves a stale stamp that still looks plausible, and a
# matrix run against a dirty tree restores over real work.
#
# So: this script does the enumeration, the ordering, the locking, and the
# staleness assertion. A human supplies only "which commit".
#
# WHAT IT RUNS, and why this order:
#   1. the three CHECKERS (fast, fail the run early if one is already broken)
#   2. the three MUTATION MATRICES (slow; they edit tracked source and restore)
#   3. the STAMP ASSERTION over every stamp found by enumeration
# Checkers first because a matrix run against a broken checker wastes the most
# expensive minutes available, and because a matrix MUTATES the checker's own
# source: a checker that cannot be trusted should be reported before anything
# starts rewriting files.
#
# THE STAMP LIST IS ENUMERATED, NEVER REMEMBERED: find(1) over the build tree for
# *last_pass.stamp. A hand-written list is a table that silently stops covering a
# matrix when one is added -- the same drift the gate inventory audits spent two
# rounds closing (audit #7: "closed" was an overclaim; audit #8: re-derived).
#
# EXIT CODES, and why a stale stamp is 2 and not 1:
#   0  every checker and matrix passed, and every stamp names HEAD
#   1  a checker or matrix FAILED -- a real regression
#   2  a stamp does not name HEAD, i.e. the proof is UNPROVEN for this code
# The 1-vs-2 split is the project's core doctrine: a bare FAIL trains everyone to
# ignore red, and "this proof does not apply to this tree" is not a regression,
# it is missing evidence. It is loud either way. (A stale stamp is exactly the
# "citation vs guess" line -- see check_fold_matrix.sh's rc=2 handling.)
#
# Env:
#   INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1  forward to the device-precond checker to skip its
#                     healthy column (saves ~6.5 min; the stamps then record the
#                     column as NOT RUN, which the assertion below still sees as
#                     a valid stamp for that sha). NOTE: this is the ONLY flag
#                     that silences the healthy column, and it is honoured in
#                     exactly one place (the dispatch() arm below). An earlier
#                     spelling `INTFER_SKIP_HEALTHY` existed in this file and
#                     nowhere else -- a bypass nobody could have known about,
#                     and the healthy column is the only thing in the stack that
#                     catches the 27B gateup crash. Renamed, not deleted.
#   INTFER_QUICK=1    checkers only, no mutation matrices (~8 min vs ~45)
#
# Locked: the mutation matrices edit tracked source, so two concurrent runs would
# cross-contaminate each other's verdicts -- a failure already observed by hand.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
LOCK_DIR="$BUILD_DIR"
LOCK="$LOCK_DIR/.run_all_checks.lock"
mkdir -p "$BUILD_DIR" || exit 1

# *** THE LOCK RECORDS ITS OWNER, AND SELF-HEALS ONLY IF THAT OWNER IS DEAD. ***
# MEASURED HAZARD, and it was mine twice in this session: a run killed mid-stack
# left the lock behind, and the operator's remedy was `rm -f $LOCK` -- a command
# that will happily delete a LIVE run's lock and let a second stack restore
# another run's mutations over its own. The message used to even instruct that. So
# the lock now carries the owner's pid, and a stale lock is removed only after
# `kill -0` proves the owner is gone, with a loud line naming the dead pid.
#
# FAIL-CLOSED ON RECYCLED PIDS, which is the right direction: if a new process has
# inherited the dead run's pid, `kill -0` SUCCEEDS, the lock is treated as live, and
# the run refuses -- annoying, never dangerous. States N-R and N-S in
# mutate_run_all_checks.sh drive both directions (dead owner -> self-heal, live
# owner -> refuse), because a self-heal that is not tested is an unconditional
# delete with extra steps.
if ! mkdir "$LOCK" 2>/dev/null; then
  lock_owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$lock_owner" ] && ! kill -0 "$lock_owner" 2>/dev/null; then
    echo "STALE LOCK: $LOCK was held by pid $lock_owner, which is gone. Removing" >&2
    echo "  it and continuing. (Recorded, because a silent takeover would hide a" >&2
    echo "  killed run; check build/ for a truncated stack log if that matters.)" >&2
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || {
      echo "FAIL: could not re-create $LOCK after removing a stale one." >&2; exit 1; }
  else
    echo "FAIL: another run_all_checks.sh holds $LOCK${lock_owner:+ (pid $lock_owner)}." >&2
    echo "  The mutation matrices edit tracked source, so a concurrent run would" >&2
    echo "  restore another run's mutations over its own." >&2
    echo "  Do NOT delete the lock to get past this -- that is the one thing that" >&2
    echo "  turns a refusal into a corrupted tree. Wait for it, or check whether" >&2
    echo "  the holder is alive: kill -0 ${lock_owner:-<pid in $LOCK/pid>}." >&2
    exit 1
  fi
fi
echo $$ > "$LOCK/pid" 2>/dev/null || true
trap 'rm -rf "$LOCK" 2>/dev/null || true' EXIT
trap 'rm -rf "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rm -rf "$LOCK" 2>/dev/null || true; exit 143' TERM

# --- fail closed on a dirty tracked tree ------------------------------------
# The matrices restore with `git checkout --`, so ANY uncommitted tracked change
# is work they would destroy. Checking the whole tracked tree is derived, so a new
# matrix target does not have to be added to a list here to stay safe.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "FAIL: the tracked tree has uncommitted changes." >&2
  echo "  The mutation matrices restore with 'git checkout --' and would destroy" >&2
  echo "  that work. Commit or stash first, then re-run." >&2
  echo "  (Untracked files are fine; only tracked changes are at risk.)" >&2
  git status --porcelain --untracked-files=no >&2
  exit 1
fi
HEAD_SHA="$(git rev-parse HEAD)"
HEAD_SHORT="${HEAD_SHA:0:7}"
echo "run_all_checks: tree clean, HEAD=$HEAD_SHORT"

# --- snapshot the UNTRACKED set, so "did a step leave something behind" is a
# DIFF rather than a hand-picked directory -------------------------------------
# The end-of-run hygiene check used to run `git status ... -- scripts/` and call
# that the whole tree. It was not: scripts/ was the directory where an EARLIER
# version of mutate_run_all_checks.sh kept its mutant, and the scope was never
# revisited after the sandbox removed that file. Measured, before the fix: a
# step that littered the repository ROOT left the run printing "no untracked
# residue in scripts/" and exiting 0 = PASS. A matrix (or any step) that wrote
# into src/ or examples/ was equally invisible.
#
# So the whole work tree is snapshotted here and compared at the end. A DIFF, not
# a tally: pre-existing untracked files are the developer's business and must not
# fail the run, which is also what the start-of-run message promises ("Untracked
# files are fine"). Ignored paths (build/, *.intfer, ...) are excluded by git,
# so the build tree and the lock directory never appear.
UNTRACKED_BEFORE="$BUILD_DIR/.run_all_untracked_before"
git status --porcelain --untracked-files=all >"$UNTRACKED_BEFORE" || {
  echo "FAIL: could not snapshot the untracked set ($UNTRACKED_BEFORE)." >&2
  echo "  Without it the end-of-run residue check cannot be a diff, and would" >&2
  echo "  silently degrade to no check at all." >&2
  exit 1
}
[ "${INTFER_QUICK:-0}" = "1" ] && echo "run_all_checks: INTFER_QUICK=1 -- checkers only, NO mutation matrices"

# --- run a step, record its rc ---------------------------------------------
# rc 2 from a step means UNPROVEN, not broken: "Nothing is wrong with the code;
# the EVIDENCE is about an older tree." Reporting that as FAIL would be the exact doctrine violation this
# repo audits for -- conflating missing evidence with a regression, so a red that
# means "re-run the matrix" is indistinguishable from a red that means "fix the
# code", and people learn to ignore both.
RC_SUMMARY=""
FAILED=0
UNPROVEN=0
UNPROVEN_LABELS=""

# The 0 / 2 / other decision lives in ONE place, used by step() and by the
# self-check below. A classification that exists twice is one that can disagree
# with itself, and the self-check would then be testing a copy rather than the
# code that actually runs.
#
# *** ONE STEP'S rc=1 MEANS "EVIDENCE IS OLDER THAN THE TREE", NOT "THE CODE IS
# BROKEN". *** check_engine_cert.sh documents its own rc=1 as `stale`: an engine
# input moved since the 45-minute gate run, so the certification no longer
# describes HEAD. Measured 2026-10-03: an audit that legitimately touched
# src/engine/engine.cpp made this step answer rc=1, and the stack printed
# "RUN ALL CHECKS: FAIL -- a real regression" with the remedy "Fix, re-run".
# Those are two OPPOSITE actions -- one of them is "change nothing and re-run
# scripts/run_ten_gates.sh" -- and conflating them is exactly the recorded
# AUDIT #18 defect (a red line that means the evidence, rather than the code, is
# how people learn to ignore red lines). AUDIT #29 hit the identical shape when
# gate_inventory called the documents stale on a 1-GPU box; that one got a BOX
# category, this one needs the same treatment.
#
# Per-STEP, derived where the step is named, not a global reinterpretation of 1:
# for every other step rc=1 still means a real failure, and it must keep doing
# so. check_engine_cert.sh is the only checker in the tree whose rc=1 is a
# statement about evidence rather than about code.
classify_rc() { # $1 = rc [$2 = step label] -> prints ok | stale | unproven | fail
  if [ "${2:-}" = "check_engine_cert.sh" ] && [ "$1" = "1" ]; then
    printf 'stale\n'; return 0
  fi
  case "$1" in
    0) printf 'ok\n' ;;
    2) printf 'unproven\n' ;;
    *) printf 'fail\n' ;;
  esac
}

# One place that knows what each step's command is, so the post-matrix re-run
# dispatches through the SAME definition instead of a second hand-maintained list.
# A duplicated table is exactly how a step silently stops being re-run.
dispatch() { # $1 = label -> runs it
  case "$1" in
    gate_inventory.sh)
      timeout -k 30 1800 bash "$SCRIPT_DIR/gate_inventory.sh"; return $? ;;
    check_device_precond.sh)
      # *** THE SKIP FLAG IS INTFER_DEVICE_PRECOND_SKIP_HEALTHY, and this branch
      # is the ONLY place it is honoured. *** An earlier revision read
      # `INTFER_SKIP_HEALTHY` here -- a flag that exists nowhere else, so it was
      # a bypass nobody could have known about, and the healthy column is the
      # only thing in the stack that catches the 27B gateup crash. The mutation
      # matrix sets the correct flag by name; nothing else may silence this
      # column, and the default (unset) runs it in full.
      if [ "${INTFER_DEVICE_PRECOND_SKIP_HEALTHY:-0}" = "1" ]; then
        INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1 timeout -k 30 1800 bash "$SCRIPT_DIR/check_device_precond.sh"
      else
        timeout -k 30 3600 bash "$SCRIPT_DIR/check_device_precond.sh"
      fi
      return $? ;;
    check_e2_ctx.sh)
      timeout -k 30 3600 bash "$SCRIPT_DIR/check_e2_ctx.sh"; return $? ;;
    check_l0_cite.sh)
      # Needs network on first run (clones compute-runtime sparse, ~1 min) and
      # exits 2 if the pin cannot be resolved, which reads as UNPROVEN rather
      # than as a pass.
      timeout -k 30 1800 bash "$SCRIPT_DIR/check_l0_cite.sh"; return $? ;;
    check_gate_pop.sh)
      # Self-check for the SHARED stamp-population helper. Synthetic tree, no
      # build state touched. Gated on the two defects the helper actually
      # produced during conversion (false LEAK, vacuous assertion) plus the
      # fail-closed case its own check then found.
      timeout -k 30 300 bash "$SCRIPT_DIR/check_gate_pop.sh"; return $? ;;
    check_verdict_scope.sh)
      # AUDIT #29. Closes AUDIT #28 defect #1 as a FAMILY: every non-verifier in
      # scripts/ that emits a spread-bounded verdict must be the one owner or a
      # declared site, and the owner's gate must be over the rounds the two arms
      # hold IN COMMON. Read-only, no GPU, no build; plants its own counter-
      # examples in a temp tree.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_verdict_scope.sh"; return $? ;;
    check_act_cache_invalidation.sh)
      # INTFER_FUSE_ACT caches FP16 activation bytes in ws_[d].act and skips a
      # host convert + upload per device when the same vector arrives twice. That
      # is bit-exact ONLY while the cached bytes are the bytes the GEMM reads, so
      # every free of that buffer must clear the cache. Function-scoped, because
      # the alias `w` names ws_ in one function and ws48_ in another.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_act_cache_invalidation.sh"; return $? ;;
    check_hw_constants.sh)
      # AUDIT #33 (2026-10-05). A hardware capability figure in a document must
      # name its source. Written because "512 Xe cores / ~179 GB/s" sat in
      # AGENTS.md and in three code comments describing the A770, and an entire
      # session's optimization strategy was built on it. Plants its own
      # counter-examples, so a vacuous green is impossible. Read-only, no GPU,
      # no build.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_hw_constants.sh"; return $? ;;
    check_dpas_shape_audit.sh)
      # DPAS SHAPE. The three ESIMD DPAS kernels derive nT = N/16 and nOp =
      # K/32 with no bounds check inside the kernel, so a shape that does not
      # divide evenly is an out-of-bounds launch, not a declined one. The
      # engine's own guard (engine.cpp:201) makes the DPAS path unreachable
      # for any non-dividing shape, so this certifies the GUARD: it re-derives
      # eligibility from the model's own tensor table (brace-matched header,
      # never the weights -- load-insensitive) and checks it against the
      # predicate the engine consults, so a future model or shard split that
      # violates a constraint is caught here rather than at first 27B run.
      # A 14-case behavioural self-check gates the run, and the negative
      # direction is proven by planting a violating model. Read-only, no GPU,
      # no build, no model loading.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_dpas_shape_audit.sh"; return $? ;;
    check_nullglob_sites.sh)
      # AUDIT #32 by ENUMERATION: derives every file that enables nullglob and
      # every glob-bearing command in those regions, so the next site is caught
      # when it is written rather than when it misleads. Finding only the site you
      # remember is exactly how the original survived a session that diagnosed it
      # in detail.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_nullglob_sites.sh"; return $? ;;
    check_dispatch_steps_consistency.sh)
      # The dispatch case labels and the STEPS lists must AGREE. 0b asks whether
      # every script HAS an arm; 0c asks whether every script is SCHEDULED;
      # neither asks whether the two sets match, and both can pass while a step
      # names a script that cannot run. Also asserts each arm executes the script
      # its LABEL names -- the spaced-filename bug that shipped once and that
      # `bash -n` cannot see.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_dispatch_steps_consistency.sh" check_pending_operator.sh; return $? ;;
    check_pending_operator.sh)
      # Splits the stack's unproven steps into OPERATOR-DEFERRED (needs the 27B
      # gate run, which an agent session must not drive) vs locally clearable,
      # read from each step's own RESULT line. Makes rc=2 actionable rather than
      # an undifferentiated "something is unproven".
      timeout -k 30 600 bash "$SCRIPT_DIR/check_pending_operator.sh"; return $? ;;
    check_destructive_git.sh)
      # No script may destroy uncommitted work: any script performing
      # `git reset --hard` / `git clean -f` / `git checkout --` must refuse to run
      # on a dirty tree. The harvest strips COMMENTS first, because
      # mutate_tp2_selfcheck.sh mentions `git checkout --` exactly once inside a
      # comment explaining that it deliberately needs no guard. Also asserts the
      # AGENTS.md standing rule exists, which is the only cover for the case that
      # actually destroyed work: a destructive command typed at a prompt.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_destructive_git.sh"; return $? ;;
    check_env_not_box_state.sh)
      # The device preconditions must name an unprepared shell as an ENVIRONMENT
      # problem, never as box state. This session misattributed a missing
      # `source setvars.sh` to a driver mismatch and told the operator a 27B
      # handoff could not run; a second time the same omission hung a checker to
      # a timeout. Its final assertion is a LIVE probe -- it runs the helper with
      # a PATH that has no sycl-ls -- so the banner is proved to print rather than
      # merely to exist.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_env_not_box_state.sh"; return $? ;;
    check_flag_gate_coverage.sh)
      # Classifies the ungated-flag residue: only the OPT-IN class is a
      # certification gap. No GPU, no model, no build. Exists because AGENTS.md's
      # "UNCERTIFIED" claim was undifferentiated and so read as a stronger claim
      # than it supported -- the residue also holds kill switches, diagnostics and
      # operational knobs, and a gate arm for one of those certifies the DEFAULT
      # path, which the gates already test.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_flag_gate_coverage.sh"; return $? ;;
    check_decode1_copies.sh)
      # The two inline copies of the M=1 decode arithmetic must AGREE. No GPU,
      # no model, no build: it reads src/gemm/quant_gemm.cpp and compares
      # arithmetic atoms, so it is an ordinary step. It is here because the
      # duplication is deliberate (quant_gemm_fused_decode1 measured that the
      # shared-device-function alternative cost 231 114 ms) and deliberate
      # duplication is only acceptable while it is pinned.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_decode1_copies.sh"; return $? ;;
    check_decode1_shipped_anchor.sh)
      # Pins the bisect-3 `SHIPPED` arm (examples/roofline_probe_bisect3/main.cpp)
      # to the engine's INT2-tab path in src/gemm/quant_gemm.cpp::
      # gemm_usm_decode1_vec. The bisect-3 ratio is only meaningful if the SHIPPED
      # body is the SAME code the engine runs; this derives the anchor's fidelity
      # from five structural invariants (2-bytes-per-8-code window, absolute per-
      # group indexing, 4-code-packed-per-byte shift, the 4-entry weight table,
      # absolute-tail indexing) with EXACTLY-ONE matching, so a drifted anchor
      # fails loudly instead of measuring against a phantom. Read-only, no GPU, no
      # build. Runs BEFORE check_scripts_live.sh so its presence does not count as
      # an unreachable script.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_decode1_shipped_anchor.sh"; return $? ;;
    check_scripts_live.sh)
      # MEASURED 2026-10-05, not predicted: a full reference-graph derivation of
      # scripts/ found SIXTEEN entries that no code referenced -- a superseded
      # generation of one-shot 27B quantize/merge drivers, four wall-time A/B
      # harnesses for experiments already recorded as dead ends or withdrawn, and
      # three probes for the one-GPU driver window. Every one survived on a `docs/`
      # or `reports/` citation, which is why a cleanup by eye does not find them.
      # This derives the rule (NAMED by a live source, or matched by the gate
      # runner's own GATE_GLOB parsed out of run_ten_gates.sh) and proves both
      # polarities on planted trees, so scripts/ cannot drift back. No GPU, no
      # build, no model.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_scripts_live.sh"; return $? ;;
    check_dri_probe.sh)
      # Closes the /dev/dri family: no script may identify a GPU card by its DRM
      # minor, which is not stable (the 2nd B580 became card2/renderD130). The
      # rule is derived from scripts/, and a four-case planted self-check gates the
      # verdict -- two earlier versions of the rule passed while a defect was
      # planted. Read-only, no GPU, no build.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_dri_probe.sh"; return $? ;;
    check_trap_release.sh)
      # AUDIT #31. A cleanup trap must release on EXIT/INT/TERM *and* the INT/TERM
      # handlers must TERMINATE. `trap ... EXIT INT TERM` -- the AUDIT #29 remedy --
      # cleans up and then resumes, so the process survives the signal; every
      # dispatch arm wraps its target in `timeout`, which waits for the child, so
      # one stubborn step HANGS the stack instead of failing it. And bash DEFERS a
      # trap behind a foreground command, so a plain `sleep 30` delays it further.
      # The population is derived from scripts/, and a five-case planted self-check
      # gates the verdict. Read-only, no GPU, no build.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_trap_release.sh"; return $? ;;
    check_guard_identity.sh)
      # AUDIT #31. A guard keyed on a resource NAME cannot see duplicates when the
      # name carries the per-instance discriminator: an earlier RESULT-filler
      # (now retired) guarded `$RUNLOG` and the name contains $$, so two LIVE runs get different names and
      # neither saw the other -- while both 27B runs lost every figure and both
      # artifacts stayed clean. Requires a shared lock whose own name does NOT carry
      # the discriminator. The converse is load-bearing: a FIXED-name refusal is
      # self-protecting and needs none. Read-only, no GPU, no build.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_guard_identity.sh"; return $? ;;
    check_stack_cert.sh)
      # AUDIT #29. Re-derives build/stack_cert/cert against THIS tree: sha, rc,
      # the census, and a clean tree. A seven-case behavioural self-check runs
      # FIRST and gates the verdict; a nested run (the self-check itself) skips it
      # and SAYS SO. Read-only, no GPU, no build.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_stack_cert.sh"; return $? ;;
    check_engine_cert.sh)
      # AUDIT #24. run_all_checks certifies the TOOLING; scripts/run_ten_gates.sh
      # certifies the ENGINE. This is what stops the two being reconciled by
      # hand -- it mechanises the "nothing the gates read has changed" argument,
      # so a 45-minute engine certification is not silently carried over an
      # engine edit, nor needlessly invalidated by a docs-only one.
      # Read-only, no GPU, no build. rc=2 (no artifact) is UNPROVEN, not a
      # regression, and the 2b re-run path handles it.
      timeout -k 30 300 bash "$SCRIPT_DIR/check_engine_cert.sh"; return $? ;;
    test_roofline_bisect.sh)
      # Bisect-2 of the M=1 INT2 GEMV probe ladder: the four clone arms (VERBATIM
      # control + VERB+HALF / VERB+TAIL / VERB+MP single-change arms) in one
      # engine-free TU, running interleaved with its own always-red twin. No model
      # bound; device event timestamps; claims only ITS OWN ratios -- SHIPPED is
      # anchored from examples/roofline_probe. The TU is bounded below the runner's
      # budget so a hang can never surface as rc=124 (a phantom regression).
      timeout -k 30 1800 bash "$SCRIPT_DIR/test_roofline_bisect.sh"; return $? ;;
    test_roofline_bisect1.sh)
      # Bisect-1 of the M=1 INT2 GEMV probe ladder: the `table` control vs the
      # transcribed SHIPPED source (`VERBATIM`) in a tiny engine-free TU, interleaved
      # with its own always-red twin. Establishes the VERBATIM baseline and the
      # within-run table/VERBATIM ratio. No model bound; device event timestamps.
      timeout -k 30 1800 bash "$SCRIPT_DIR/test_roofline_bisect1.sh"; return $? ;;
    test_roofline_bisect3.sh)
      # Bisect-3 of the M=1 INT2 GEMV probe ladder: `VERBATIM` plus single-change
      # arms `VERB+TAIL` / `VERB+MP` and the `SHIPPED` anchor, all four in one TU
      # so the SHIPPED ratio is measured on the SAME buffers in the SAME TU as the
      # arms (not cross-binary, where register pressure would differ). The ~4.5-5x
      # of the 5.6-6.4x gap that survives the module move (1.26x) and the branch
      # (1.007x) sits between VERBATIM and SHIPPED -- this is where it is measured.
      timeout -k 30 1800 bash "$SCRIPT_DIR/test_roofline_bisect3.sh"; return $? ;;
     test_roofline_bisect4.sh)
       # Bisect-4 of the M=1 INT2 GEMV probe ladder: the DIFFERENTIAL between the
       # bisect-2 VERBATIM body and the roofline_probe int2_variant<0> transcription.
       # Five incremental arms (VERBATIM / VERB+bitcast / VERB+bitcast+4 /
       # VERB+bitcast+4+vsel / CLONE) in one TU, interleaved on the same buffers.
       # This is the run that settled the question "does the ~7x sit in the
       # cosmetic single-changes" -- it does not (every arm lands at ~1.11x), so
       # the gap is in the shared text, not in any of these four changes. No model
       # bound; device event timestamps; the same always-red twin as bisect-2.
       timeout -k 30 1800 bash "$SCRIPT_DIR/test_roofline_bisect4.sh"; return $? ;;
    test_rowpair_equiv.sh)
      # The MEASURED bitwise-equivalence check for INTFER_DECODE1_ROWPAIR. Runs the
      # binary TWICE in two processes -- quant_gemm_forward_usm reads the flag into
      # a function-local `static const bool`, so one process cannot hold both
      # orders -- and diffs 200k+ element values. Binds no model. This gate exists
      # because the kernel's "bit-exact by construction" argument was WRONG: the
      # first version double-offset a row-relative multi_ptr and lost the device on
      # both INT2 shapes, while passing on the FP16 and INT4 shapes the branch
      # never reaches.
      timeout -k 30 1800 bash "$SCRIPT_DIR/test_rowpair_equiv.sh"; return $? ;;
    test_roofline_probe.sh)
      # AUDIT #33 follow-on. The allocator-vs-kernel discriminator for decode's
      # bandwidth deficit. Device event timestamps (load-insensitive), binds NO
      # model (AGENTS.md standing rule 1), and carries its own always-red twin: a
      # quarter of the bytes must take about a quarter of the time, or the GB/s
      # figures are not tracking traffic and the run exits 1. That twin caught two
      # real bugs in the probe itself.
      timeout -k 30 900 bash "$SCRIPT_DIR/test_roofline_probe.sh"; return $? ;;
    run_ten_gates.sh)
      # AUDIT #24 -- SELF-CHECK ONLY, never the gates themselves. run_ten_gates.sh
      # runs the engine on both GPUs for tens of minutes and is deliberately NOT a
      # step of this stack (see the "WHY IT IS NOT A STEP" block in that file: a
      # check slow enough to be skipped gets skipped). Its SELF-CHECK is seconds of
      # stubs and is what the stack enforces -- otherwise the one runner in the tree
      # whose failure path nobody has ever watched execute is exactly the always-green
      # harness the AUDIT #14 vacuity class describes.
      timeout -k 30 300 bash "$SCRIPT_DIR/run_ten_gates.sh" --self-check; return $? ;;
    check_checker_matrix_coverage.sh)
      # Derives "every checker has a dispatched matrix" from scripts/ rather than
      # trusting a hand-written total. Wired, and its own matrix is
      # mutate_checker_matrix_coverage.sh -- otherwise this checker would be the
      # very drift it reports.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_checker_matrix_coverage.sh"; return $? ;;
    check_gpu_recover.sh)
      # Stubbed probe; nothing real is modified. Pins the RECOVERY TOOL's exit
      # codes, especially 2=unmeasurable vs 1=still-broken -- the distinction
      # that keeps "the probe could not count" from being read as a device
      # count. Needs no GPU.
      timeout -k 30 600 bash "$SCRIPT_DIR/check_gpu_recover.sh"; return $? ;;
    check_gate_enum.sh)
      # Behavioural self-check for the SHARED vacuity guard. Four cases plus a
      # converse: must reject 0/empty/non-numeric/below-minimum, and must
      # ACCEPT well-formed input SILENTLY -- without that, a helper that
      # rejects everything would score 3/3 and prove nothing.
      timeout -k 30 300 bash "$SCRIPT_DIR/check_gate_enum.sh"; return $? ;;
    check_build_dir_consumers.sh)
      # Static: derives the artifact-consumer population from the CMake
      # add_executable list plus every script touching a build tree, then fails
      # on any that hardcodes one. No GPU, no build, seconds.
      timeout -k 30 900 bash "$SCRIPT_DIR/check_build_dir_consumers.sh"; return $? ;;
    mutate_build_dir_consumers.sh)
      # Proves check_build_dir_consumers.sh is non-vacuous: plants a consumer that
      # hardcodes "$ROOT/build" (and "${ROOT}/build"), rejects it, then verifies
      # a comment-only mention is NOT graded as a consumer and that a compliant
      # override is accepted. Sandbox OUTSIDE $BUILD_DIR, never touches scripts/.
      timeout -k 30 1200 bash "$SCRIPT_DIR/mutate_build_dir_consumers.sh"; return $? ;;
    mutate_e2_ctx.sh)
      # Proves check_e2_ctx.sh's SELF-CHECK half. The real-run half is not
      # mutation-testable (its artifact is correct by construction) and one
      # state is excluded as unobservable -- see the matrix's SCOPE block.
      timeout -k 30 1800 bash "$SCRIPT_DIR/mutate_e2_ctx.sh"; return $? ;;
    mutate_gpu_recover.sh)
      # Fixtures live in a mktemp sandbox OUTSIDE $BUILD_DIR, and the matrix
      # asserts the enumerated stamp-DIRECTORY set is unchanged, so its own
      # artefacts cannot join the population step 3/3 compares.
      timeout -k 30 1800 bash "$SCRIPT_DIR/mutate_gpu_recover.sh"; return $? ;;
    mutate_verdict_scope.sh)
      # AUDIT #29. Every state damages the TREE the checker grades, in a copy of
      # scripts/, and the checker that runs is the sandbox's OWN copy -- the
      # first version ran the real checker, which made its two checker-patching
      # states unobservable by construction. V-F is an isolation case, not a
      # mutation: it empties the harvest with the self-tests disabled so the
      # enumeration guard is the only thing that can fire.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_verdict_scope.sh"; return $? ;;
    mutate_dri_probe.sh)
      # Every state damages the CORPUS the rule grades, handed over through
      # INTFER_SCRIPTS_DIR; the checker that runs is the real one reading that
      # corpus -- weakening the CHECKER would be unobservable (AUDIT #21). D-E is
      # the CONVERSE: a single-minor liveness probe must stay clean, without which
      # a rule that flagged everything would satisfy D-A..D-D.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_dri_probe.sh"; return $? ;;
    mutate_trap_release.sh)
      # Every state damages a CORPUS script's cleanup trap via INTFER_SCRIPTS_DIR;
      # the checker is correct by construction, so weakening the CHECKER would be
      # unobservable (AUDIT #21). T-E is the CONVERSE: the compliant shape and a
      # `trap -` disarm must stay clean, without which a rule that flagged every
      # trap would satisfy T-A..T-D.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_trap_release.sh"; return $? ;;
    mutate_guard_identity.sh)
      # Same artifact-defect shape, on the guard-identity rule. G-E is the state
      # that gets forgotten and the one that would have caught the first version of
      # the rule: a FIXED-name refusal owes no lock, so demanding one there would
      # force a pointless lock onto every idempotent re-run.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_guard_identity.sh"; return $? ;;
    mutate_stack_cert.sh)
      # AUDIT #29. Breaks the census DERIVATION and the dirty-tree test, both of
      # which the self-check cannot see because every fixture derives its census
      # from the same expression the verdict uses (the AUDIT #12 S4 shape). S-B
      # cannot use the shared `rc != 0` predicate -- neutering the guard turns
      # rc 1 into rc 0 -- so both halves are asserted there explicitly.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_stack_cert.sh"; return $? ;;
    mutate_engine_cert.sh)
      # AUDIT #24. Plants ten_gates/HEAD into a SANDBOX build dir (via
      # INTFER_BUILD_DIR) so the real certification is never touched, and drives
      # the four verdict paths. Its own E-D state weakens the input filter while
      # the artifact is deliberately stale -- legitimate only because the planted
      # artifact is itself defective (the AUDIT #21 rule).
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_engine_cert.sh"; return $? ;;
    mutate_act_cache_invalidation.sh)
      # AC-1 removes an invalidation (a use-after-free no stream test finds);
      # AC-4 inverts the checker's own comparison -- the bug it shipped with.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_act_cache_invalidation.sh"; return $? ;;
    mutate_nullglob_sites.sh)
      # NS-2 covers a blind spot the scanner REALLY had: a same-line
      # ( shopt -s nullglob; cmd GLOB ) was invisible to a line-start regex, and
      # the scan reported POPULATION 0 over a tree that contained one.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_nullglob_sites.sh"; return $? ;;
    mutate_hw_constants.sh)
      # AUDIT #33 matrix. Degrades the RULE (each unit class, the CITATION
      # requirement, case-insensitivity, the corpus floor) and requires the
      # checker's self-check to fire; HW-1b is the always-red twin that neuters the
      # self-check gate too and requires the degraded rule to accept a defective
      # corpus, which is what makes HW-1's catch attributable. Runs entirely
      # against copies under build/ -- the tracked tree is never edited.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_hw_constants.sh"; return $? ;;
    mutate_dispatch_steps_consistency.sh)
      # DS-1 re-arms the bug that shipped: a dispatch arm whose path becomes a
      # filename containing a space. Valid shell, rc=127 at runtime.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_dispatch_steps_consistency.sh"; return $? ;;
    mutate_pending_operator.sh)
      # PO-1 empties the log directory: the classifier must report INCONCLUSIVE,
      # NOT "nothing pending" -- different claims, and only one is reassuring.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_pending_operator.sh"; return $? ;;
    mutate_destructive_git.sh)
      # DG-2 is the state that makes the comment-strip load-bearing: without it
      # the mutate_tp2_selfcheck.sh DECOY is flagged. DG-4 is the anti-vacuity
      # test -- a harvest matching nothing must be refused, not read as "all
      # guarded, nothing to check".
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_destructive_git.sh"; return $? ;;
    mutate_env_not_box_state.sh)
      # Mutates scripts/_gate_precond.sh. EN-2 is the state that keeps EN-1
      # honest: delete the banner TEXT while every call site survives, so a
      # call-site count alone cannot pass.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_env_not_box_state.sh"; return $? ;;
    mutate_flag_gate_coverage.sh)
      # FG-1/FG-4 are the two states that FOUND real holes in the checker: a
      # census query with no floor of its own, and an opt-in class with no
      # non-empty floor. FG-3 is the inverse that proves the floors are what does
      # the catching rather than the state passing for an unrelated reason.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_flag_gate_coverage.sh"; return $? ;;
    mutate_roofline_bisect.sh)
      # The gate for test_roofline_bisect.sh enforces EXIT SEMANTICS rather than
      # source correctness, and those semantics were themselves the defect: the
      # first version returned rc=124 with EMPTY OUTPUT, which the runner reads as a
      # REGRESSION and which cost it the whole step budget. 4 states, all caught,
      # against a STUB probe so no real GPU job is launched; 2 states excluded with
      # their reason and 1 dropped with its reason, all recorded in the file.
      timeout -k 30 1200 bash "$SCRIPT_DIR/mutate_roofline_bisect.sh"; return $? ;;
    mutate_roofline_bisect1.sh)
      # The gate for test_roofline_bisect1.sh (bisect-1: table vs VERBATIM) enforces
      # EXIT SEMANTICS via the probe-timeout bound; states run against a STUB probe
      # so no real GPU job is launched.
      timeout -k 30 1200 bash "$SCRIPT_DIR/mutate_roofline_bisect1.sh"; return $? ;;
    mutate_roofline_bisect3.sh)
      # The gate for test_roofline_bisect3.sh (bisect-3: VERBATIM / VERB+TAIL /
      # VERB+MP / SHIPPED) enforces EXIT SEMANTICS via the probe-timeout bound; states
      # run against a STUB probe so no real GPU job is launched.
      timeout -k 30 1200 bash "$SCRIPT_DIR/mutate_roofline_bisect3.sh"; return $? ;;
     mutate_roofline_bisect4.sh)
       # The gate for test_roofline_bisect4.sh enforces EXIT SEMANTICS via the
       # probe-timeout bound; states run against a STUB probe so no real GPU job is
       # launched.
       timeout -k 30 1200 bash "$SCRIPT_DIR/mutate_roofline_bisect4.sh"; return $? ;;
    mutate_roofline_probe.sh)
      # The gate for test_roofline_probe.sh enforces EXIT SEMANTICS (the probe
      # exits 1 on a vacuous run) AND a numeric ceiling floor. States run against
      # a pre-produced log via INTFER_PROBE_LOG/INTFER_PROBE_RC, so no real GPU
      # job is launched and the matrix grades the GATE, not the probe.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_roofline_probe.sh"; return $? ;;
    mutate_decode1_copies.sh)
      # AUDIT #11 direction: these states mutate the ARTIFACT (quant_gemm.cpp),
      # not the checker, because an artifact that is correct by construction
      # makes any weakening of the checker unobservable. DC-6 is the always-red
      # twin -- both copies drifting together must stay GREEN. Mutates a TEMP
      # COPY only and restores from a pre-matrix snapshot, never from git, so it
      # is safe on a legitimately dirty tree.
      #
      # REWRITTEN 2026-10-05: the checker's population is now DERIVED by
      # enumeration, so the matrix was rebuilt -- 5 states, all caught, plus two
      # EXCLUDED with their reason (the UNCOVERED rule and the required set share
      # one list, so they cannot be tested independently). It paid for itself
      # immediately: it caught the enumeration labelling row-pair
      # "COVERED-BY-ATOM-PIN" while only a FLOOR backed that label, and a real
      # shift-into-mask EQUALITY comparison had to be added before DC-2 could fire.
      # NOTE there is only ONE arm for this matrix; a second was added by mistake
      # and the stack's own 0c step reported it as "scheduled MORE THAN ONCE".
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_decode1_copies.sh"; return $? ;;
    mutate_decode1_shipped_anchor.sh)
      # Audit #21 direction: the checker is correct by construction on a clean
      # tree, so the matrix drifts the ARTIFACT instead of weakening the checker.
      # The bisect-3 SHIPPED arm (examples/roofline_probe_bisect3/main.cpp) is
      # copied into a git-archive sandbox and patched (one byte-offset drifts
      # off-by-one); the checker must REJECT the drifted anchor and PASS the
      # restored one. No change to scripts/ or src/; sandbox via mktemp.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_decode1_shipped_anchor.sh"; return $? ;;
    mutate_scripts_live.sh)
      # SL states weaken the reachability RULE (the corpus is a temp tree, never
      # the tracked scripts/), which is the audit #21 direction for a checker
      # whose artifact is its own classification: the rule must be shown to fire
      # AND to stay quiet before either version is believed. SL-C is the state
      # that would have caught the first version of the checker, whose own header
      # prose named real scripts and so satisfied the rule with itself.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_scripts_live.sh"; return $? ;;
    mutate_tp2_selfcheck.sh)
      # AUDIT #31. The DECODE1_MR arm's --self-check in test_tp2_27b.sh. It runs
      # ONLY the stubbed --self-check half on temp copies -- NO GPU, NO MODEL, NO
      # 27B LOAD -- because the arm's real-run half is the operator's job and must
      # never be driven from here (the standing rule: the user runs the 27B
      # gates). That is also why this matrix is cheap enough to be an ordinary
      # step rather than the excluded POST-STACK class.
      timeout -k 30 900 bash "$SCRIPT_DIR/mutate_tp2_selfcheck.sh"; return $? ;;
    mutate_rowpair_equiv.sh)
      # AUDIT #21 direction for the test_rowpair_equiv.sh REWRITE (2026-10-07):
      # that rewrite added the base-order kill-switch restoration, the reference-
      # stability guard, and INT2-only comparison. A rewritten checker is a
      # rewritten artifact, so the matrix mutates the ARTIFACT -- a bit-exact
      # codebook-index bug planted on VERBMP and on VERBSEL -- and requires the
      # gate to REFUSE each. Baseline (pristine) must PASS first, or every state is
      # void. Each state copies src/ include/ examples/ scripts/ + CMakeLists
      # (NOT the whole tree -- the test/ model weights are ~2.2 GB and a whole-tree
      # copy OOM-killed this matrix) and configures its own build dir, so the
      # tracked tree is never touched.
      timeout -k 60 1800 bash "$SCRIPT_DIR/mutate_rowpair_equiv.sh"; return $? ;;
    mutate_checker_matrix_coverage.sh)
      # Each state BOTH poisons the tree with an unmated checker AND removes the
      # requirement that would catch it -- weakening alone is unobservable while
      # the real tree has full coverage (AUDIT #21).
      timeout -k 30 1800 bash "$SCRIPT_DIR/mutate_checker_matrix_coverage.sh"; return $? ;;
    mutate_gate_pop.sh)
      # Removes ONE case (its echo AND its own counter) from check_gate_pop.sh.
      # patch_literal is called with THREE arguments -- file, exact-old,
      # exact-new -- and old/new each span a literal newline. Passing one quoted
      # string per LINE was the original bug: five states, all "PATCH VERIFY
      # FAIL", found with bash -x rather than by guessing.
      timeout -k 30 1800 bash "$SCRIPT_DIR/mutate_gate_pop.sh"; return $? ;;
    mutate_gate_enum.sh)
      # Patches SANDBOX COPIES under the build tree, never scripts/, so a
      # mutated helper cannot leak into a later glob of that directory.
      timeout -k 30 1800 bash "$SCRIPT_DIR/mutate_gate_enum.sh"; return $? ;;
    mutate_l0_cite.sh)
      # Mutates a COPY of the pinned upstream tree under build/, never tracked
      # source, so this matrix is safe to run alongside the rest of the stack.
      timeout -k 30 3600 bash "$SCRIPT_DIR/mutate_l0_cite.sh"; return $? ;;
    mutate_fold_matrix.sh)
      timeout -k 30 7200 bash "$SCRIPT_DIR/mutate_fold_matrix.sh"; return $? ;;
    mutate_gate_inventory.sh)
      timeout -k 30 7200 bash "$SCRIPT_DIR/mutate_gate_inventory.sh"; return $? ;;
    mutate_device_precond.sh)
      timeout -k 30 7200 bash "$SCRIPT_DIR/mutate_device_precond.sh"; return $? ;;
    mutate_run_all_checks.sh)
      timeout -k 30 3600 bash "$SCRIPT_DIR/mutate_run_all_checks.sh"; return $? ;;
    *)
      # Seam: an external step directory. It exists because this script is the
      # ENTRY POINT for a ~45 minute stack, so a mutation matrix that invoked it
      # directly would need three hours -- and an assumption the harness cannot
      # redirect is an assumption the harness cannot test (same rationale as
      # INTFER_SYCL_LS and INTFER_BUILD_DIR). The self-check and its matrix run
      # here. Deliberately the `*)` arm: the first version placed this AFTER the
      # `esac` with no `*)`, so a MATCHING arm ran its command and then fell
      # straight through to "unknown step" and returned 97 -- every real step was
      # broken, and it went unnoticed because the only tests in that commit drove
      # synthetic steps, which take the fallback path and work.
      if [ -n "${INTFER_RUN_ALL_EXTRA_STEPS_DIR:-}" ] && [ -x "$INTFER_RUN_ALL_EXTRA_STEPS_DIR/$1" ]; then
        "$INTFER_RUN_ALL_EXTRA_STEPS_DIR/$1"
        return $?
      fi
      echo "internal: unknown step '$1'" >&2
      return 97
      ;;
  esac
}

step() { # $1 = label
  local label="$1" log rc verdict
  log="$BUILD_DIR/run_all_$(printf '%s' "$label" | tr -c 'a-zA-Z0-9' '_').log"
  printf '  %-34s ' "$label"
  dispatch "$label" >"$log" 2>&1
  rc=$?
  verdict="$(classify_rc "$rc" "$label")"
  case "$verdict" in
    ok) printf 'ok\n' ;;
    stale)
      # NOT counted as a failure: nothing is wrong with the code, the EVIDENCE
      # describes an older tree. The remedy is a re-run of the engine gates, and
      # saying "real regression" here would tell the reader to go and edit working
      # code. Counted separately so the run still ends non-zero (the evidence is
      # genuinely absent) -- the same rc=2 rule UNPROVEN follows.
      printf 'STALE rc=%s (evidence older than HEAD, NOT a regression -- re-run scripts/run_ten_gates.sh; %s)\n' "$rc" "$log"
      UNPROVEN=$((UNPROVEN + 1)) ;;
    unproven)
      printf 'UNPROVEN rc=%s (stale evidence, not a regression -- %s)\n' "$rc" "$log"
      UNPROVEN=$((UNPROVEN + 1)) ;;
    fail)
      printf 'FAIL rc=%s (see %s)\n' "$rc" "$log"
      FAILED=$((FAILED + 1)) ;;
  esac
  # ONE accumulation for BOTH evidence verdicts, in one place. The first version
  # duplicated `UNPROVEN_LABELS=...` across the two branches, and the very next
  # matrix run refused to score: its anchor pre-flight found the line twice and
  # said "want 1" -- correct, because an anchor that matches two sites cannot say
  # which one a patch rewrites. Fixing the ANCHOR would have been the wrong move
  # (the standing rule is re-point, never loosen); the duplication was the defect,
  # so the duplication is what went.
  case "$verdict" in
    stale|unproven) UNPROVEN_LABELS="$UNPROVEN_LABELS $label" ;;
  esac
  RC_SUMMARY="$RC_SUMMARY$(printf '%-34s %s\n' "$label" "rc=$rc")\n"
  return 0
}

# --- CLASSIFIER SELF-CHECK (behavioural, gating) ---------------------------
# Every other checker in this stack opens with a self-check that its own
# classification can fail, because a harness that would reject everything still
# reads as a passing proof (the audit-#17 always-red lesson). run_all_checks.sh
# had none, so a state that made it always-red would have looked healthy.
#
# Three synthetic steps, because the three verdicts are the three things this
# script exists to keep apart: a regression (rc 1) must NOT be reported as
# missing evidence (rc 2). Each writes a sentinel, so "the step never ran" is
# distinguishable from "the step ran and was classified".
selfcheck() {
  local d bad=0 g rc verdict want sent label code rest step_name
  d="$(mktemp -d "$BUILD_DIR/.selfcheck.XXXXXX")" || return 1
  # Fields are split ONCE into three named variables. The first version used
  # "${spec#*:}" on "sc_ok:0:ok", which yields "0:ok" and wrote `exit 0:ok` --
  # an invalid exit argument, so ALL THREE steps returned bash's exit-error
  # status 2 and the self-check reported every case wrong. The self-check caught
  # a bug in itself, which is the only reason to trust it.
  # The engine-cert case is the CONVERSE of the map above and both halves are
  # required: rc=1 must be `stale` for THAT step and must stay `fail` for every
  # other one. A per-step map that swallowed 1 globally would satisfy the first
  # case and silently disarm the whole stack.
  for spec in "sc_ok:0:ok::" "sc_unproven:2:unproven::" "sc_fail:1:fail::" \
              "sc_stale:1:stale:check_engine_cert.sh" \
              "sc_stale_converse:1:fail::"; do
    label="${spec%%:*}"
    rest="${spec#*:}"
    code="${rest%%:*}"; rest="${rest#*:}"
    want="${rest%%:*}"; rest="${rest#*:}"
    step_name="${rest%%:*}"
    g="$d/$label"
    sent="$d/$label.ran"
    printf '#!/usr/bin/env bash\n: > %q\nexit %s\n' "$sent" "$code" >"$g"
    chmod +x "$g"
    rm -f "$sent"
    "$g" >/dev/null 2>&1
    rc=$?
    verdict="$(classify_rc "$rc" "$step_name")"
    if [ ! -f "$sent" ]; then
      echo "  SELFCHECK: $label never ran (no sentinel) -- the check is vacuous" >&2
      bad=1; continue
    fi
    if [ "$verdict" != "$want" ]; then
      echo "  SELFCHECK: $label exited $rc and classified '$verdict', want '$want'" >&2
      bad=1
    fi
  done
  rm -rf "$d"
  return $bad
}
echo "== 0/3 classifier self-check (must be able to fail) =="
if ! selfcheck; then
  echo >&2
  echo "RUN ALL CHECKS: FAIL -- the classifier self-check did not hold, so every" >&2
  echo "  verdict below is meaningless. A checker that cannot fail proves nothing." >&2
  exit 1
fi
echo "  ok    sc_ok->ok, sc_unproven->unproven, sc_fail->fail, sc_stale->stale"

# --- 0b dispatch coverage, derived by ENUMERATION ----------------------------
# dispatch() and STEPS are hand-written lists. So a new check_*.sh or
# mutate_*.sh that nobody remembers to add to them is not "a step that fails" --
# it is a step that NEVER RUNS, and this entry point would still print PASS and
# exit 0. That is the audits #7/#8 shape ("closed" was an overclaim) one level
# up, and the whole reason the stamp list above is found with find(1).
#
# The arm list is extracted from dispatch()'s own case block, not retyped, so the
# two cannot drift. If the extraction yields nothing, every file reads as
# undispatched and this FAILS LOUDLY -- a coverage check that silently passes
# because it parsed nothing is worse than no check.
echo "== 0b dispatch coverage (every check_*/mutate_* must have a dispatch arm) =="
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
enum_reset

# *** THE QUERY'S OWN RESULT CLASS IS CHECKED, NOT JUST ITS OUTPUT. ***
# grep returns 0 match / 1 no-match / >=2 THE GREP FAILED. The pipeline below
# previously discarded that: a malformed pattern or a read error yielded an
# EMPTY $ARMS, and the empty check turned it into "could not extract" -- while
# the per-script loop had ALREADY reported every script as undispatched on the
# strength of a query that never ran. Both answers were wrong for the same
# reason. This bit this session twice, in my own tooling: a pattern containing
# an unclosed ')' made grep exit 2 and the "no match" was read as real.
grep_rc=0
ARMS="$(sed -n '/^dispatch() {/,/^}$/p' "$0" \
        | grep -oE '^[[:space:]]+[^[:space:]]+\.sh\)' \
        | tr -d '[:blank:])' | LC_ALL=C sort -u)" || grep_rc=$?
if [ "$grep_rc" -ge 2 ]; then
  echo "  FAIL  the arm-list extraction QUERY FAILED (rc=$grep_rc) -- the grep did" >&2
  echo "        not run, so coverage is UNKNOWN. This is not a coverage verdict." >&2
  exit 1
fi
# Anti-vacuity (AUDIT #20): the extraction must have HARVESTED something before
# any verdict about coverage is reported. Zero rows means the query ran and
# found nothing -- which is a different fact from "every script has an arm".
enum_require_nonempty "dispatch-arms" "$(printf '%s\n' "$ARMS" | grep -c . || true)" \
  "dispatch() arm patterns" || exit 1

UNDISPATCHED=""
QUERY_FAILED=""
N_DISPATCHABLE=0
for f in "$SCRIPT_DIR"/check_*.sh "$SCRIPT_DIR"/mutate_*.sh; do
  [ -f "$f" ] || continue
  b="$(basename "$f")"
  N_DISPATCHABLE=$((N_DISPATCHABLE + 1))
  grc=0
  printf '%s\n' "$ARMS" | grep -qxF -- "$b" >/dev/null 2>&1 || grc=$?
  if [ "$grc" -ge 2 ]; then
    # The lookup failed; saying "no dispatch arm" here would be a verdict
    # about a query that never ran.
    QUERY_FAILED="$QUERY_FAILED $b"
  elif [ "$grc" -eq 1 ]; then
    UNDISPATCHED="$UNDISPATCHED $b"
  fi
done
if [ -n "$QUERY_FAILED" ]; then
  echo "  FAIL  the per-script arm lookup FAILED for:$QUERY_FAILED" >&2
  echo "        grep did not run for those, so coverage is UNKNOWN, not absent." >&2
  exit 1
fi
if [ -z "$ARMS" ]; then
  echo "  FAIL  could not extract dispatch()'s arm list -- coverage is UNCHECKED." >&2
  exit 1
fi
if [ -n "$UNDISPATCHED" ]; then
  echo "  FAIL  no dispatch() arm for:$UNDISPATCHED" >&2
  echo "        They exist in scripts/ as checkers/matrices, but this entry point" >&2
  echo "        would never run them and would still report PASS. Add a dispatch()" >&2
  echo "        arm (and a STEPS entry)." >&2
  FAILED=$((FAILED + 1))
elif [ "$N_DISPATCHABLE" -eq 0 ]; then
  # *** MEASURED, and it is AUDIT #20 one level up. *** The sandbox run behind state
  # N-P held only run_all_checks.sh in its scripts/, so this glob matched NOTHING
  # and the section printed "0 check_*/mutate_* script(s), all dispatched" -- a
  # green verdict over an empty census. The arm-list extraction above is guarded by
  # enum_require_nonempty; the PER-SCRIPT harvest was not, so the guard covered the
  # QUERY and not the population the query is about.
  echo "  FAIL  no check_*/mutate_* files were found in $SCRIPT_DIR -- coverage is" >&2
  echo "         UNKNOWN, not complete. 'All dispatched' over an empty set is not" >&2
  echo "         a result." >&2
  FAILED=$((FAILED + 1))
else
  echo "  ok    $N_DISPATCHABLE check_*/mutate_* script(s), all dispatched"
fi

# The step list, overridable so the self-check and its mutation matrix can run
# this script in seconds instead of 45 minutes.
if [ -n "${INTFER_RUN_ALL_STEPS:-}" ]; then
  STEPS="$INTFER_RUN_ALL_STEPS"
else
  STEPS="gate_inventory.sh check_device_precond.sh check_e2_ctx.sh check_gate_enum.sh check_gate_pop.sh check_gpu_recover.sh check_engine_cert.sh check_checker_matrix_coverage.sh check_build_dir_consumers.sh check_verdict_scope.sh check_dri_probe.sh check_trap_release.sh check_guard_identity.sh check_decode1_copies.sh check_decode1_shipped_anchor.sh check_flag_gate_coverage.sh check_hw_constants.sh check_dpas_shape_audit.sh check_env_not_box_state.sh check_destructive_git.sh check_dispatch_steps_consistency.sh check_pending_operator.sh check_nullglob_sites.sh check_act_cache_invalidation.sh check_scripts_live.sh"
STEPS="$STEPS run_ten_gates.sh"   # SELF-CHECK ONLY -- see the dispatch arm
  # A REAL gate, but it needs a GPU and takes ~30 s, so it sits in the
  # same INTFER_QUICK-off band as check_l0_cite.sh. Quick mode is meant to be
  # checkers-only, and a band that silently skipped a gate would be the
  # audit #18 shape (an arm that never runs certifies nothing).
  [ "${INTFER_QUICK:-0}" = "1" ] || \
    STEPS="$STEPS test_roofline_probe.sh test_roofline_bisect.sh test_roofline_bisect1.sh test_roofline_bisect3.sh test_roofline_bisect4.sh test_rowpair_equiv.sh"
  [ "${INTFER_QUICK:-0}" = "1" ] || \
    STEPS="$STEPS check_l0_cite.sh"
  [ "${INTFER_QUICK:-0}" = "1" ] || \
    STEPS="$STEPS mutate_fold_matrix.sh mutate_gate_inventory.sh mutate_device_precond.sh mutate_run_all_checks.sh mutate_l0_cite.sh mutate_hw_constants.sh mutate_roofline_bisect.sh mutate_roofline_bisect1.sh mutate_roofline_bisect3.sh mutate_roofline_bisect4.sh mutate_roofline_probe.sh mutate_build_dir_consumers.sh mutate_gate_enum.sh mutate_gpu_recover.sh mutate_e2_ctx.sh mutate_gate_pop.sh mutate_engine_cert.sh mutate_checker_matrix_coverage.sh mutate_verdict_scope.sh mutate_stack_cert.sh mutate_dri_probe.sh mutate_trap_release.sh mutate_guard_identity.sh mutate_tp2_selfcheck.sh mutate_decode1_shipped_anchor.sh mutate_decode1_copies.sh mutate_flag_gate_coverage.sh mutate_env_not_box_state.sh mutate_destructive_git.sh mutate_dispatch_steps_consistency.sh mutate_pending_operator.sh mutate_nullglob_sites.sh mutate_act_cache_invalidation.sh mutate_scripts_live.sh mutate_rowpair_equiv.sh"
fi

# --- 0c every check_*/mutate_* is SCHEDULED, not merely dispatchable ---------
# *** MEASURED DEFECT, 2026-10-02, and 0b could not see it. *** A new checker and
# a new matrix were added with their dispatch() arms (so 0b reported
# "31 script(s), all dispatched") but WITHOUT a STEPS entry -- and STEPS is the
# list the run actually iterates. The matrix therefore never ran, and step 3/3
# reported its stamp STALE: a symptom that reads like "the evidence is older than
# the tree" with "re-run the named matrix" as the printed remedy, when the real
# fact is that the step was NEVER SCHEDULED and re-running it changes nothing.
# Same class as N-J one level up -- "a step that never runs is not a step that
# fails" -- and the fix is the same shape: derive the population, compare against
# the REAL list, fail loudly.
#
# SKIPPED, LOUDLY, when INTFER_RUN_ALL_STEPS is set: the override exists so the
# self-check and its matrix can drive synthetic steps, and requiring synthetic
# coverage of the real scripts would make the seam unusable. It says SKIP rather
# than passing silently, because a quiet skip reads as a verdict.
#
# ALSO skipped, LOUDLY, under INTFER_QUICK=1. That mode exists to run CHECKERS only
# and deliberately leaves every matrix out of STEPS, so 0c saw the whole matrix
# population as "never scheduled" and FAILED the run. That made INTFER_QUICK=1
# STRUCTURALLY UNABLE TO PASS -- not "it failed once" but "no code state can make
# it pass", which is the worst kind of gate result, because it trains everyone to
# ignore the line. Found while re-certifying after a revert, i.e. exactly when a
# fast check-only pass is most wanted.
#
# SKIP and not PASS: under QUICK the schedule genuinely is not being asserted,
# and saying SKIP keeps "an assumption no harness can redirect is one no harness
# can test" intact. QUICK removes steps, so membership cannot be checked.
if [ -n "${INTFER_RUN_ALL_STEPS:-}" ] || [ "${INTFER_QUICK:-0}" = "1" ]; then
  if [ "${INTFER_QUICK:-0}" = "1" ]; then
    echo "== 0c STEPS-membership check SKIPPED (INTFER_QUICK=1: this run runs" >&2
    echo "      CHECKERS only and leaves every matrix out of STEPS by design, so" >&2
    echo "      scheduling is NOT being asserted. Not a pass, and not a failure.)" >&2
  else
    echo "== 0c STEPS-membership check SKIPPED (INTFER_RUN_ALL_STEPS is set: this run"
    echo "      drives synthetic steps, so real coverage is NOT being asserted) =="
  fi
else
  echo "== 0c every check_*/mutate_* is SCHEDULED (an arm alone never runs) =="
  UNSCHEDULED=""
  POST_STACK=""
  bad_post=""
  N_SCHEDULABLE=0
  for f in "$SCRIPT_DIR"/check_*.sh "$SCRIPT_DIR"/mutate_*.sh; do
    [ -f "$f" ] || continue
    b="$(basename "$f")"
    N_SCHEDULABLE=$((N_SCHEDULABLE + 1))
    # A checker that declares itself POST-STACK in its header is not required in
    # STEPS: it grades the certificate the stack WRITES, so it must run AFTER the
    # stack. Measured: wired in as a step it could only ever report STALE, so a PASS
    # stack turned into FAIL. The declaration is read FROM THE FILE (no
    # hand-maintained list) and printed, so the exemption is never quiet; requiring
    # it in the first 40 lines stops a passing mention in the body from exempting
    # anything. It is tested BEFORE the `case` because a case branch cannot hold a
    # compound command -- the first attempt put the `if` inside one and bash blamed
    # a token three lines later.
    if head -40 "$f" | grep -q '^# POST-STACK:'; then
      # *** AND IT MUST NOT ALSO BE SCHEDULED. *** The exemption says "this grades
      # the certificate the stack WRITES, so it runs after". A file that is both
      # exempt and scheduled does the opposite: it grades the cert it is supposed to
      # certify, BEFORE that cert is written, and can only ever report STALE --
      # which is exactly how check_stack_cert.sh was first wired, and the stack went
      # from PASS to FAIL. So the exemption is honest only while the two conditions
      # are mutually exclusive, and that is asserted here rather than trusted.
      case " $STEPS " in
        *" $b "*)
          echo "  FAIL  $b declares POST-STACK *and* is in STEPS." >&2
          echo "         As a step it grades the certificate the stack has not" >&2
          echo "         written yet, so it can only report STALE. Remove it from" >&2
          echo "         STEPS -- do NOT remove the declaration." >&2
          bad_post="$bad_post $b" ;;
      esac
      POST_STACK="$POST_STACK $b"
      continue
    fi
    case " $STEPS " in
      *" $b "*) ;;
      # run_ten_gates.sh is a COMPOSITE step, not a check_ or mutate_ file, so it
      # is never in this population; nothing to exempt here.
      *) UNSCHEDULED="$UNSCHEDULED $b" ;;
    esac
  done
  if [ -n "$UNSCHEDULED" ]; then
    echo "  FAIL  never scheduled, so this run will NOT execute:$UNSCHEDULED" >&2
    echo "         They have a dispatch() arm (0b is happy) and a file, but the run" >&2
    echo "         iterates STEPS. Their stamps will read STALE at step 3/3, which" >&2
    echo "         looks like missing evidence and is not." >&2
    FAILED=$((FAILED + 1))
  elif [ "$N_SCHEDULABLE" -eq 0 ]; then
    # The same empty-census trap as 0b, in this section: "every script is
    # scheduled" over ZERO scripts is a tautology, not a result.
    echo "  FAIL  no check_*/mutate_* files were found in $SCRIPT_DIR -- 'every" >&2
    echo "         one is scheduled' over an empty set proves nothing." >&2
    FAILED=$((FAILED + 1))
  else
    echo "  ok    every check_*/mutate_* is in STEPS as well as dispatch()"
  fi
  [ -z "$bad_post" ] || FAILED=$((FAILED + 1))
  if [ -n "$POST_STACK" ]; then
    for b in $POST_STACK; do
      echo "  ok    POST-STACK, exempt from STEPS (declared in its own header):$b"
      echo "        it grades the certificate run_all_checks.sh WRITES, so it runs"
      echo "        AFTER the stack -- as a step it could only ever report STALE."
    done
  fi
  # *** 0c DOES NOT DETECT A DUPLICATE, AND THAT IS A RECORDED LIMITATION, NOT AN
  # OMISSION. *** It answers "is it scheduled", not "is it scheduled twice".
  # Measured on this very session: a checker was added to BOTH STEPS lines, so
  # every full stack ran it twice -- idempotent, but double the wall time and a
  # duplicated step line -- and 0c stayed green, which is correct for the question
  # it was built to answer (a matrix that NEVER RAN). Deleting the duplicate was
  # the right fix; making this section reject duplicates would be scope creep on
  # the section that exists because a step was silently unscheduled. So a
  # duplicate is reported as a NOTE that names the file and the count, without
  # failing.
  # *** CONVERSION TRIGGER: promote the NOTE to a FAIL the moment any step becomes
  # NON-IDEMPOTENT -- one that stamps, writes, takes the lock, or mutates source.
  # Then running it twice stops being cosmetic and becomes a wrong verdict.
  DUPED=""
  for f in "$SCRIPT_DIR"/check_*.sh "$SCRIPT_DIR"/mutate_*.sh; do
    [ -f "$f" ] || continue
    b="$(basename "$f")"
    n="$(printf '%s' "$STEPS" | tr ' ' '\n' | grep -cxF -- "$b" || true)"
    [ "${n:-0}" -gt 1 ] && DUPED="$DUPED $b(x${n})"
  done
  if [ -n "$DUPED" ]; then
    echo "  NOTE  scheduled MORE THAN ONCE -- harmless only while every step is"
    echo "        idempotent (conversion trigger in the source):$DUPED"
  fi
  # A fast, honest seam for testing 0c itself: run the coverage sections and
  # stop. Without it, proving 0c can fail costs a 45-minute stack.
  if [ "${INTFER_RUN_ALL_COVERAGE_ONLY:-0}" = "1" ]; then
    echo
    if [ "$FAILED" -eq 0 ]; then
      echo "COVERAGE ONLY: OK (sections 0, 0b, 0c). No step was executed."
      exit 0
    fi
    echo "COVERAGE ONLY: FAIL (a coverage section drifted; no step was executed)" >&2
    exit 1
  fi
fi
echo "== 1/2 checkers =="
for label in $STEPS; do step "$label"; done

if [ "${INTFER_QUICK:-0}" != "1" ]; then
  echo "== 2/2 mutation matrices (slow; each edits and restores tracked source) =="

  # --- re-validate anything that was unproven -------------------------------
  # ORDERING, and the first run got this wrong. The fold matrix's stamp is read
  # by its own checkers, and the checkers run BEFORE the matrices -- so on a tree
  # whose stamps were stale it correctly reported rc=2, the matrices then
  # re-stamped everything, and the run still exited 2 claiming evidence was
  # missing when it was present. A verdict computed from a snapshot the rest of
  # the run invalidated.
  #
  # The fix RE-RUNS the steps that came back rc=2, rather than hard-coding which
  # checkers depend on which stamp. That keeps it derived: a future checker that
  # reads a matrix stamp is re-validated automatically, and a hand-written
  # dependency table would quietly omit it. Reported, never silent.
  if [ -n "$UNPROVEN_LABELS" ]; then
    echo "== 2b re-validating unproven step(s) after the matrices =="
    for label in $UNPROVEN_LABELS; do
      printf '  %-34s ' "$label (re-run)"
      log="$BUILD_DIR/run_all_$(printf '%s' "$label" | tr -c 'a-zA-Z0-9' '_')_rerun.log"
      dispatch "$label" >"$log" 2>&1
      rc=$?
      case "$(classify_rc "$rc" "$label")" in
        ok)
          printf 'ok (evidence now current)\n'
          UNPROVEN=$((UNPROVEN - 1)) ;;
        unproven)
          printf 'still rc=%s (%s)\n' "$rc" "$log" ;;
        stale)
          # The one rc that must NOT be escalated here. check_engine_cert.sh is
          # stale because an ENGINE input moved, and steps 1-2 mutate only
          # scripts/ -- so it will still be stale after the matrices, and the
          # remedy is a human re-running the 45-minute engine gates. Calling that
          # "REGRESSED ... is now a real failure" would be false on every run
          # where an engine commit landed, and would move UNPROVEN -> FAILED,
          # turning "the evidence is old" into "the code is broken".
          printf 'still STALE rc=%s (evidence older than HEAD -- re-run scripts/run_ten_gates.sh; %s)\n' "$rc" "$log" ;;
        fail)
          # A re-run that turns STALE EVIDENCE into a REAL FAILURE is a
          # regression, not missing evidence. The first version only branched on
          # success/failure and left FAILED at 0 either way, so a step that
          # answered rc=2 on the first pass and rc=1 on the re-run produced the
          # line "still rc=1" and then the final verdict "RUN ALL CHECKS: ... NOT
          # a regression: ... Nothing is wrong with the code" with exit 2 -- the
          # script calling a checker failure healthy, which is the one thing this
          # script exists to prevent. Proven before the fix (a synthetic step
          # exiting 2 then 1). The three verdicts come from the SAME classifier
          # the main loop uses, so this cannot disagree with it.
          printf 'REGRESSED rc=%s (%s) -- was unproven, is now a real failure\n' "$rc" "$log"
          UNPROVEN=$((UNPROVEN - 1))
          FAILED=$((FAILED + 1)) ;;
      esac
    done
  fi
fi

# --- 2c did HEAD move under us? -----------------------------------------------
# MEASURED, not hypothetical. Three times in one session a SECOND writer (another
# agent session on the same clone) committed while this ~45 minute stack was
# running, so HEAD changed between the start-of-run snapshot and the stamp
# assertion. Consequences, and they are not all benign:
#   * the matrices stamped the LATER commit, so step 3 compared them against the
#     EARLIER $HEAD_SHA and reported 5 STALE -- a correct-looking verdict about a
#     tree that no longer existed;
#   * the operator is told "re-run the named matrix", which is the wrong remedy:
#     re-running cannot help, because the stamps already name the newer commit and
#     the mismatch was an artifact of the snapshot, not of missing evidence.
# A stamp check against a moved HEAD is exactly the "citation vs guess" line: it
# looks like a verdict about coverage and is really a verdict about a race.
#
# So HEAD is re-read HERE, immediately before the comparison, and a move is its
# own LOUD condition with its own remedy -- distinct from stale stamps, and NOT
# folded into STALE, because the two call for opposite actions.
HEAD_NOW="$(git rev-parse HEAD)"
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

# --- 3. the stamp assertion, over an ENUMERATED set -------------------------
# find, not a table. A hand-written list is a table that quietly stops covering a
# matrix when one is added. STAMP_DIR is overridable so the mutation matrix can
# point this at a synthetic stamp population and observe the enumeration.
#
# The comparison uses $HEAD_NOW, not the start-of-run $HEAD_SHA. Measured, and it
# is the bug this very line used to have: the first version read
#   [ "$HEAD_MOVED" -eq 1 ] || HEAD_SHA="$HEAD_NOW"
# which assigns ONLY when HEAD did NOT move -- the exact inverse of the intent,
# and a no-op in the healthy case (where the two are equal anyway). So on a real
# mid-run move, step 3 compared against the commit the run STARTED on and printed
# "STALE ... (want <dead sha>)" for stamps that were current for the tree on disk.
# Observed live: a concurrent commit landed, 2c correctly reported the move, and
# then all five stamps were reported STALE naming a commit that no longer
# existed -- the misleading-verdict shape, reintroduced one line below the fix
# for it. `&&` is the correct polarity: adopt the current HEAD exactly when it
# moved.
STAMP_DIR="${INTFER_RUN_ALL_STAMP_DIR:-$BUILD_DIR}"
echo "== 3/3 stamp assertion (enumerated from $STAMP_DIR) =="
STALE=0
STAMP_COUNT=0
[ "$HEAD_MOVED" -eq 1 ] && HEAD_SHA="$HEAD_NOW"
while IFS= read -r stamp; do
  [ -n "$stamp" ] || continue
  STAMP_COUNT=$((STAMP_COUNT + 1))
  sha="$(sed -n 's/^sha=//p' "$stamp" | head -1)"
  extra="$(sed -n 's/^\(caught\|states\)=//p' "$stamp" | head -1)"
  rel="${stamp#$STAMP_DIR/}"
  if [ "$sha" = "$HEAD_SHA" ]; then
    printf '  ok    %-46s %s %s\n' "$rel" "${sha:0:7}" "$extra"
  else
    printf '  STALE %-46s %s (want %s)\n' "$rel" "${sha:0:7}" "${HEAD_SHA:0:7}"
    STALE=$((STALE + 1))
  fi
done <<EOF
$(find "$STAMP_DIR" -name '*last_pass.stamp' -type f 2>/dev/null | sort)
EOF
if [ "$STAMP_COUNT" -eq 0 ]; then
  echo "  FAIL  no stamps found under $STAMP_DIR -- nothing was proven." >&2
  STALE=$((STALE + 1))
fi

# --- 4. did this run leave the tree as it found it? --------------------------
# The entry point already refuses to START on a dirty tree. This is the other
# half of that invariant, and it is the one that covers every matrix at once.
#
# Every mutation matrix edits tracked source and restores it, but each one only
# verifies the files IT declared as targets. A matrix that accidentally dirtied
# something else -- a wrong `$SRC`, a target list that lost an entry, an applier
# that writes outside its scope -- would leave no trace in its own report. And
# untracked residue matters too: a scratch file left in scripts/, in src/, or at
# the repository root would be picked up by whatever globs it next (several gates
# enumerate scripts/*.sh and build targets from the CMake graph).
#
# The untracked half is a DIFF against the snapshot taken before step 1, over the
# WHOLE work tree -- not a tally of one directory. See the snapshot's comment for
# the measurement that forced this.
#
# Placed HERE, once, rather than copied into four matrices: "nine hand-rolled
# copies of the block is precisely how the family reopened" (audit #7). The driver
# is the thing that ran them all, so the driver is what checks they cleaned up.
echo "== 4/4 tree hygiene (this run must leave the tree as it found it) =="
if git diff --quiet && git diff --cached --quiet; then
  echo "  ok    tracked tree clean"
else
  echo "  FAIL  the tracked tree is DIRTY after this run." >&2
  echo "         A step mutated something it did not restore." >&2
  git status --porcelain --untracked-files=no >&2
  FAILED=$((FAILED + 1))
fi
RESIDUE="$(LC_ALL=C comm -13 \
             <(LC_ALL=C sort "$UNTRACKED_BEFORE") \
             <(git status --porcelain --untracked-files=all | LC_ALL=C sort) \
           | grep -v '^ M' || true)"
if [ -n "$RESIDUE" ]; then
  echo "  FAIL  untracked residue left behind by this run -- a later check that" >&2
  echo "         globs those paths (several enumerate scripts/*.sh or the CMake" >&2
  echo "         build graph) would run against it:" >&2
  printf '%s\n' "$RESIDUE" | sed 's/^/         /' >&2
  FAILED=$((FAILED + 1))
else
  echo "  ok    no new untracked files anywhere in the work tree"
fi

# --- verdict ----------------------------------------------------------------
# *** THE VERDICT IS WRITTEN TO DISK, WITH ITS OWN EXIT CODE. *** Measured gap,
# 2026-10-02: after a full PASS run the only artifact was a log file whose final
# line said PASS, and the operator's next probe ("grep -c 'RUN ALL CHECKS:
# PASS'") printed a COUNT, which is not an exit code -- so the script's own `$?`
# existed nowhere. A printed verdict is the script talking about itself; this is
# the AUDIT #22 traceable-artifact rule applied to the harness that certifies
# everything else ("a number in prose is a citation"), and it is what lets a later
# session say "the stack passed at c57f2bf" by reading a file.
#
# *** DELIBERATELY NOT NAMED *last_pass.stamp. *** That suffix is the stamp
# vocabulary the stamp assertion enumerates, and a stack certification is not a
# matrix pass: it would join the population it is supposed to be asserting about
# and be graded by the same rules (one home per stamp dir, known-matrix
# ownership). It is a CERTIFICATION, a different artifact, in its own directory.
CERT_DIR="$BUILD_DIR/stack_cert"
CERT="$CERT_DIR/cert"
mkdir -p "$CERT_DIR" 2>/dev/null || true
write_cert() { # $1 rc, $2 one-word verdict
  {
    echo "stamp=stack_cert"
    echo "sha=$HEAD_SHA"
    echo "short=$HEAD_SHORT"
    echo "rc=$1"
    echo "verdict=$2"
    echo "failed=$FAILED"
    echo "stale=$STALE"
    echo "unproven=$UNPROVEN"
    echo "stamps=$STAMP_COUNT"
    # *** THE THREE DERIVED COUNTS TRAVEL TOGETHER, BECAUSE TRANSCRIBING THEM IS
    # HOW A HAND-COUNTED TOTAL GOES STALE. *** Measured confusion this session: a
    # summary said "14 checkers, 18 matrices" while 0b derived 31 (=14+17) and
    # step 3/3 counted 18 STAMP FILES -- a third, different quantity, because
    # device_precond_matrix writes TWO stamp files into ONE directory. Nothing was
    # wrong except that three numbers with three meanings were quoted in prose.
    # The certificate is the artifact of record, so the census belongs here.
    echo "scripts=$N_DISPATCHABLE"
    echo "checks=$(ls "$SCRIPT_DIR"/check_*.sh 2>/dev/null | wc -l | tr -d ' ')"
    echo "matrices=$(ls "$SCRIPT_DIR"/mutate_*.sh 2>/dev/null | wc -l | tr -d ' ')"
    echo "stamp_dirs=$(find "$STAMP_DIR" -name '*last_pass.stamp' -type f 2>/dev/null \
      | sed 's|/[^/]*$||' | sort -u | wc -l | tr -d ' ')"
    echo "quick=${INTFER_QUICK:-0}"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$CERT.tmp" 2>/dev/null && mv "$CERT.tmp" "$CERT"
}

echo
if [ "$FAILED" -ne 0 ]; then
  write_cert 1 FAIL
  echo "RUN ALL CHECKS: FAIL ($FAILED step(s) FAILED) -- a real regression." >&2
  [ "$UNPROVEN" -ne 0 ] && echo "  plus $UNPROVEN step(s) unproven (stale evidence)." >&2
  echo "  certified in $CERT (rc=1)" >&2
  echo "  A REAL REGRESSION: see the FAIL lines above and the per-step logs in" >&2
  echo "  $BUILD_DIR/run_all_*.log. Fix, re-run, then verify with" >&2
  echo "  scripts/check_stack_cert.sh." >&2
  exit 1
fi
if [ "$STALE" -ne 0 ] || [ "$UNPROVEN" -ne 0 ]; then
  write_cert 2 UNPROVEN
  echo "RUN ALL CHECKS: $STAMP_COUNT stamp(s) checked, $STALE STALE, $UNPROVEN step(s) unproven, for $HEAD_SHORT." >&2
  echo "  NOT a regression: the evidence does not cover this commit. Nothing is" >&2
  echo "  wrong with the code; the proof is just older than the tree." >&2
  echo "  Re-run the named matrix (or all of them) so its stamp names $HEAD_SHORT." >&2
  echo "  certified in $CERT (rc=2)" >&2
  echo "  STALE EVIDENCE, not a regression: scripts/check_stack_cert.sh gives the" >&2
  echo "  same verdict this run gave; re-run the stack so the stamps name HEAD." >&2
  exit 2
fi
echo "RUN ALL CHECKS: PASS ($STAMP_COUNT stamp(s), all at $HEAD_SHORT)"
echo "  every checker and matrix passed, and every enumerated stamp names HEAD."
if [ "${INTFER_QUICK:-0}" = "1" ]; then
  echo "  NOTE: INTFER_QUICK=1 -- no matrix was RE-RUN, so only pre-existing stamps"
  echo "  were asserted. That is weaker than a full run: re-run without the flag."
fi
write_cert 0 PASS
echo "  certified in $CERT (rc=0)"
# *** PRINT THE POINTER, so the next session is told BY THE TOOL. *** A
# certification nobody re-derives is a transcript with a filename: for three
# sessions running, each one hand-ran the same six probes before it could say the
# stack passes. scripts/check_stack_cert.sh does that now, and it is POST-STACK BY
# DESIGN (it grades the certificate this run writes), so it must be run after --
# the one step a reader cannot infer from this output unless it is stated here.
echo
echo "  VERIFY IT NOW:  scripts/check_stack_cert.sh"
echo "    rc=0 -> this certificate is evidence for THIS commit"
echo "    rc=1 -> re-run the stack (the evidence is older than the tree)"
echo "    rc=2 -> unproven, no certificate"
echo "  Do not hand-check it: the checker re-derives the sha, the rc and the census."
exit 0
