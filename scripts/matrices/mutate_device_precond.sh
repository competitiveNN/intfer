#!/usr/bin/env bash
# Reproducible non-vacuity proof for scripts/checks/check_device_precond.sh.
#
# WHY THIS EXISTS. Audit #17 committed that checker and the two device-failure
# fixtures, but the first proof that the checker could actually FAIL was a few
# hand-run mutations in a terminal transcript. That is the identical defect the
# fixtures were committed to end, one level up: "a mutation that only exists in a
# transcript cannot be re-run by the next session, so the next session re-derives
# (or worse, re-asserts) it." The house already answers this twice
# (mutate_fold_matrix.sh, mutate_gate_inventory.sh); this is the third.
#
# Each state breaks ONE mechanism and the checker must then exit non-zero FOR THE
# EXPECTED REASON. Requiring the reason -- not merely some non-zero exit -- is the
# difference between a real catch and a contaminated one, and it is not
# theoretical: an early hand-run of M-C in a parallel batch inherited M-B's
# chmod and "passed" by aborting on M-B's pre-flight. A checker that exits
# non-zero for an unrelated reason is not evidence that the mutated mechanism is
# load-bearing.
#
#   M-B  clear a fixture's EXECUTE BIT
#        PATH lookup SKIPS a non-executable file, so a 644 stub hands every probe
#        back to the real tool and BOTH failure directions become no-ops that
#        still report green. The most dangerous shape: it does not fail, it stops
#        testing. Caught by the fixture pre-flight that stats the mode.
#   M-A  strip a real gate's device guard entirely
#        "Needs a device and forgot to say so" (audit #15): the gate then exits 0
#        with no usable device, silently claiming coverage it never provided.
#   M-C  weaken gate_require_device to gate_require_render_node
#        A HALF guard: still correct for the node-unreachable direction, blind to
#        the 0-device (libumf) direction this box actually hits. The gate then runs
#        and exits 0. A single-direction matrix would score this as a PASS, which
#        is why both directions are needed.
#   M-D  narrow ONLY this checker's GPU-token regex, so its derived gate list
#        shrinks while gate_inventory.sh still lists the same gate.
#        This exists because M-A did NOT test the drift comparison: M-A tripped
#        the per-file "carries NO guard at all" message, so the
#        `own_devices != inv_devices` branch was never the assertion that fired
#        -- a claimed load-bearing check that was structurally blind. Here the two
#        derivations must DISAGREE with neither gate broken, and the harness must
#        report "classification drift" specifically.
#   M-E  rot the INTFER_SYCL_LS seam: make gate_gpu_count report "0 devices,
#        counted fine" when sycl-ls is absent instead of "unmeasurable".
#        Both cases then exit 2 with the same diagnostic, so the zero-device
#        direction degrades into "sycl-ls is missing" while still reporting
#        green. The harness's seam discrimination exists for exactly this and is
#        the only thing that can see it.
#   M-F  make self-check case (c) -- the "harness is not simply always-red" case
#        -- reject the CORRECT cause, by rewording only the synthetic gate's
#        message. The always-red detection must fire, or a harness that
#        rejects everything would look like a passing non-vacuity proof.
#   M-G  gate_inventory.sh returns NO TABLE (an early exit of its own)
#   M-H  gate_inventory.sh's table arrives but classifies NO device gate
#        The cross-check compares two INDEPENDENT derivations that must agree, so
#        "the other side produced nothing" is not "the other side disagrees" -- and
#        reporting it as drift sends the next reader to reconcile two lists when
#        there is nothing to reconcile. These two guards shipped INERT: they called
#        an undefined `continue_gate`, which is not a syntax error and not a crash.
#        `bash -n` stayed green, bash printed "command not found", control fell
#        through, and the log carried the refusal AND the misleading
#        "classification drift" line. So these states assert the ABSENCE of that
#        line as well as the presence of the refusal -- an exit code cannot tell
#        "refused correctly" from "refused and then did the forbidden thing".
#        They are ARTIFACT states (audit #21): each stub is a legitimately
#        reachable gate_inventory output, so the checker must reject all of them.
#
# WHY THE HEALTHY COLUMN IS DISABLED HERE. Every state sets
# INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1. M-A/M-C break the GUARDS, so on a
# healthy box the gate would run to completion (minutes) and its rc would change
# for reasons unrelated to the mutation. A stated purpose, not a silent
# default-off; the checker runs the column by default.
#
# ANTI-PARALLELISM. An earlier hand-run of M-B and M-C in a PARALLEL tool batch
# shared this working tree, so M-C inherited M-B's chmod and scored a false
# catch. These states touch the SAME files, so they must never run concurrently;
# the lock below makes that structural, and each state re-asserts a clean
# pre-state so a leftover from a killed run is caught rather than inherited.
#
# This script edits tracked source and restores it from a SNAPSHOT taken at startup,
# so the dirty-tree refusal is OPT-IN (INTFER_MATRIX_ALLOW_DIRTY=1): when set, an
# uncommitted target is snapshotted alongside the committed baseline and restored
# byte-for-byte, instead of aborting. Default (unset) keeps the strict guard so a
# careless invocation cannot destroy uncommitted work. Caller's working set is
# named as INPUT: UNCOMMITTED with blob hashes in the stamp (same convention as
# mutate_gate_pop.sh/mutate_gpu_recover.sh).
#
# Env: INTFER_BUILD_DIR  build tree (default $ROOT/build) for logs and the stamp.
#      INTFER_MATRIX_ALLOW_DIRTY  1 = run over an uncommitted working set (opt-in).
# Exit: 0 all states caught for the right reason; 1 otherwise.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/device_precond_matrix"
STAMP="$LOG_DIR/mutate_last_pass.stamp"
CHECKER="scripts/checks/check_device_precond.sh"
LOCK="$LOG_DIR/.mutate.lock"
export INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1
mkdir -p "$LOG_DIR" || exit 1

# Sourced at TOP LEVEL, never inside a conditional: a guard sourced only on an
# error path is a guard that is usually absent, and its call site then dies with
# "command not found" on the NORMAL path (recorded in AGENTS.md, and it is the
# same shape as the defect this matrix now covers).
# shellcheck source=scripts/lib/_gate_pred.sh
. "$SCRIPT_DIR/../lib/_gate_pred.sh"

# --- lock --------------------------------------------------------------------
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_device_precond.sh is running (or died holding $LOCK)." >&2
  echo "  These states edit the SAME tracked files, so two concurrent runs would" >&2
  echo "  cross-contaminate each other's verdicts -- a real, already-observed" >&2
  echo "  failure. If no run is active, remove the lock and re-run." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

TGT_A="scripts/gates/test_dpas_fold.sh"
TGT_C="scripts/gates/test_http_batch.sh"
TGT_B="scripts/fixtures/no-render-node/python3"
TGT_H="scripts/checks/check_device_precond.sh"
TGT_P="scripts/lib/_gate_precond.sh"
TGT_G="scripts/gate_inventory.sh"
DIRTY="$TGT_A $TGT_C $TGT_B $TGT_H $TGT_P"

# shellcheck source=scripts/lib/_gate_snapshot.sh
. "$SCRIPT_DIR/../lib/_gate_snapshot.sh"
# shellcheck source=scripts/lib/_gate_state.sh
. "$SCRIPT_DIR/../lib/_gate_state.sh"
GATE_SNAP_ALLOW_DIRTY="${INTFER_MATRIX_ALLOW_DIRTY:-0}" gate_snap_dirty_guard "$DIRTY" || exit 1
GATE_SNAP_DIR_PLACEHOLDER=0
gate_snap_init "$DIRTY $TGT_G" >/dev/null
export SNAPDIR="$GATE_SNAP_DIR"

CHECKER_BLOB="$(git hash-object "$CHECKER" 2>/dev/null || sha256sum "$CHECKER" | cut -d' ' -f1)"
if ! git diff --quiet -- "$CHECKER" 2>/dev/null; then
  echo "  INPUT: UNCOMMITTED file=$CHECKER blob=$CHECKER_BLOB" >&2
  CHECKER_DIRTY=yes
else
  CHECKER_DIRTY=no
fi

restore() {
  gate_snap_restore_all "$DIRTY $TGT_G"
}

trap 'restore; rm -f "scripts/gates/test_zz_unmeasured_dummy.sh" "$LOG_DIR"/test_zz_unmeasured_dummy.*.log 2>/dev/null; rm -rf "$GATE_SNAP_DIR" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'restore; rm -f "scripts/gates/test_zz_unmeasured_dummy.sh" "$LOG_DIR"/test_zz_unmeasured_dummy.*.log 2>/dev/null; rm -rf "$GATE_SNAP_DIR" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'restore; rm -f "scripts/gates/test_zz_unmeasured_dummy.sh" "$LOG_DIR"/test_zz_unmeasured_dummy.*.log 2>/dev/null; rm -rf "$GATE_SNAP_DIR" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

# --- ANCHOR PRE-FLIGHT ------------------------------------------------------
# A refactor can move or delete a mutation's target; the patch then matches
# nothing, the file is unchanged, the checker passes, and the state reads AS
# EXPECTED for the wrong reason. Each anchor must occur EXACTLY ONCE, up front.
# Patterns live in a QUOTED ARRAY with a parallel file array -- audit #16 found
# the earlier `pat|desc` + IFS='|' table silently TRUNCATED at the first `|`,
# which these patterns themselves contain (ERE alternation, `||`).
#
# $1 = the exact text each state patches, which must therefore be unique in its
# file. The M-B anchor is a mode, not text, so it anchors on a structural line
# that proves the file is still the node-stub: 'python3' is NOT unique there
# (3x: two comments + the exec) and the pre-flight refused it, correctly -- an
# anchor matching several sites proves nothing about which is being mutated.
check_anchor() { # $1 = file, $2 = literal text, $3 = description
  local n
  # `grep -c` prints "0" AND exits 1 on no match, so `|| echo 0` appended a
  # second "0" and a miss reported "found 0\n0 times". Fails closed either way,
  # but a harness's diagnostic is part of what it certifies.
  n="$(grep -cF -- "$2" "$1" 2>/dev/null || true)"
  if [ "$n" != "1" ]; then
    echo "FAIL: anchor for $3 found $n times in $1, want 1." >&2
    echo "  Target: $2" >&2
    echo "  Re-point the anchor (do NOT loosen it) if the code moved." >&2
    exit 1
  fi
}
A_GUARD='gate_require_device || exit 2'
A_EXEC='exec /usr/bin/python3'
A_TOKENS='(--gpu|sycl-ls|level_zero|xpu-smi|dpas_fold_probe|onemkl_probe)'
A_NOTAVAIL='echo "SKIP: $sycl_ls not available; cannot count GPUs" >&2'
# The gate_env_diagnose call inside that branch. SINGLE-quoted for the same reason
# as A_NOTAVAIL: it contains $sycl_ls, and every double-quoted interpolation of
# it tried so far silently mangled the dollar. Declared here (before the
# check_anchor pre-flight) and consumed at the M-E state below.
A_DIAG='    gate_env_diagnose "'"'"'$sycl_ls'"'"' is not executable on PATH."'
A_CASEC='echo "SKIPPED: render node opens but Level Zero enumerates 0 device(s)."'
check_anchor "$TGT_A" "$A_GUARD" "M-A: the whole-gate device guard to strip"
check_anchor "$TGT_C" "$A_GUARD" "M-C: the guard line to weaken to render_node-only"
check_anchor "$TGT_B" "$A_EXEC"  "M-B: the fixture whose execute bit is cleared"
check_anchor "$TGT_H" "$A_TOKENS" "M-D: the GPU-token regex this checker derives its list from"
check_anchor "$TGT_P" "$A_NOTAVAIL" "M-E: the unmeasurable branch of the count helper"
check_anchor "$TGT_H" "$A_CASEC" "M-F: self-check case (c)'s synthetic correct-cause gate"
# M-G / M-H REPLACE gate_inventory.sh wholesale with a stub, so there is no
# literal anchor for patch_literal to drift against. What CAN go stale is the
# COUPLING: the checker's refusal guards look for the string "RENDER-NODE" in
# gate_inventory's output, so a rename of that column header would make the guards
# fire forever on a healthy tree -- a checker that cries wolf. So the pre-flight
# asserts the header the checker greps for is still emitted, exactly once.
[ -f "$TGT_G" ] || { echo "FAIL: $TGT_G is gone; M-G/M-H cannot stub it." >&2; exit 1; }
check_anchor "$TGT_G" '1-GPU-ARM RENDER-NODE' "M-G/M-H: the column header the cross-check greps for"
echo "anchor pre-flight: ok (7/7 targets located exactly once)"

CAUGHT=0
TOTAL=0
STATES_LOG="$LOG_DIR/mutate_states.txt"
: >"$STATES_LOG"

# Patch with an EXACT literal replacement, never sed. The first version used
# `sed -i "s|$A_TOKENS|(--gpu)|"` and the anchor itself CONTAINS '|'
# (ERE alternation), so sed died with "unknown option to `s'" and changed
# nothing -- which then scored as "the mechanism is NOT load-bearing", the exact
# false conclusion audit #16 warns about (a `pat|desc` table and an IFS='|'
# parser silently truncating at a '|' inside a pattern; here the delimiter).
# A quoted python literal has no delimiter to collide with.
patch_literal() { # $1 = file, $2 = exact old text, $3 = exact new text
  python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
if n != 1:
    sys.exit("patch_literal: anchor found %d times in %s, want 1" % (n, path))
open(path, "w").write(src.replace(old, new))
PY
}

# AND prove the patch landed. Compare hash to the SNAPSHOT baseline via shared helper.
verify_patched() { # $1 = file, $2 = state id
  if gate_snap_verify_patched "$1"; then return 0; fi
  echo "  ABORT $2  the patch did NOT change $1." >&2
  exit 1
}

assert_clean() { # $1 = file, $2 = state id
  if gate_snap_assert_clean "$1"; then return 0; fi
  echo "  ABORT $2  pre-state DIRTY: $1 already modified." >&2
  return 1
}

# $5 = OPTIONAL pattern that must be ABSENT from the log. Some defects are not
# "the checker stayed quiet" but "the checker said the RIGHT thing AND the
# MISLEADING thing as well", and an rc alone cannot tell those apart -- which is
# exactly the defect M-G/M-H exist for: the cross-check guards printed their
# refusal and then ran the comparison anyway, so the log carried both the refusal
# and the "classification drift" line they exist to suppress. Asserted through
# _gate_pred.sh so the POLARITY is named at the call site, per the recorded rule.
run_state() { # $1 = id, $2 = description, $3 = file to restore (- = none), $4 = expected reason, $5 = must-be-absent ERE, $6 = mode (fail|keep-green, default fail)
  local id="$1" desc="$2" restore="$3" want="$4" absent="${5:-}" mode="${6:-fail}" log rc reason verdict keep_ok
  TOTAL=$((TOTAL + 1))
  mkdir -p "$LOG_DIR" || exit 1
  log="$LOG_DIR/mutate_${id}.log"
  timeout 1800 bash "$CHECKER" >"$log" 2>&1
  rc=$?
  reason="$(grep -m1 -E 'FAIL' "$log" | cut -c1-170)"
  verdict=MISSED
  if [ "$mode" = "keep-green" ]; then
    # Converse: the checker must STAY GREEN (rc=0) and REPORT the wanted token.
    # A state that makes the checker red, or silences the report, is MISSED.
    if [ "$rc" -ne 0 ]; then
      reason="checker exited $rc, want 0 (baseline broken)"
    elif ! grep -q "UNMEASURED" "$log"; then
      reason="no UNMEASURED reported for the planted gate"
    elif ! grep -q "test_zz_unmeasured_dummy" "$log"; then
      reason="planted gate not reported as UNMEASURED"
    else
      verdict=ok
      CAUGHT=$((CAUGHT + 1))
      reason="UNMEASURED reported, baseline stays green"
    fi
  else
    if [ "$rc" -eq 0 ]; then
      reason="<checker still exited 0>"
    elif [ -z "$reason" ]; then
      reason="<checker failed but named no assertion>"
    elif ! printf '%s' "$reason" | grep -qF -- "$want"; then
      reason="WRONG REASON: caught, but not by '$want' -> $reason"
    elif [ -n "$absent" ] && ! pred_expect_absent "$id" "$(cat "$log")" "$absent"; then
      reason="MISLEADING LINE STILL EMITTED: caught by '$want' AND also printed '$absent'"
    else
      verdict=ok
      CAUGHT=$((CAUGHT + 1))
    fi
  fi
  if [ "$verdict" = ok ]; then
    printf '  ok    %-4s caught for the right reason\n' "$id"
    printf '         via: %s\n' "$(printf '%s' "$reason" | sed 's/^ *//')"
  else
    printf '  MISSED %-4s %s\n' "$id" "$reason"
    case "$reason" in
      "<checker still exited 0>"*)
        echo "         -> the mechanism this state breaks is NOT load-bearing." | sed 's/^ *//' ;;
      "WRONG REASON"*)
        echo "         -> it DID fail, but on an unrelated assertion, so this state" | sed 's/^ *//'
        echo "            is not yet evidence. Fix the patch or the expectation." | sed 's/^ *//' ;;
      "MISLEADING LINE"*)
        echo "         -> it DID refuse, but also emitted the line the refusal is" | sed 's/^ *//'
        echo "            meant to suppress. A guard that reports and then does the" | sed 's/^ *//'
        echo "            forbidden thing anyway is worse than no guard." | sed 's/^ *//' ;;
      *)
        echo "         -> unattributed catch: a non-zero exit with no failing" | sed 's/^ *//'
        echo "            assertion named is not evidence of anything." | sed 's/^ *//' ;;
    esac
  fi
  gate_state_record "$STATES_LOG" "$id" "$verdict" "$want" "${absent:-none}"
  # Restore from snapshot via shared helper (skip when no file was mutated).
  if [ "$restore" = "-" ]; then
    printf '         no restore needed (no tracked file mutated)\n'
    return 0
  fi
  if gate_snap_restore "$restore"; then
    printf '         restored: %s clean\n' "$restore"
  else
    printf '  MISSED %-4s RESTORE FAILED for %s -- manual repair needed\n' "$id" "$restore"
    CAUGHT=$((CAUGHT - 1))
  fi
  return 0
}

echo "== MUTATION MATRIX: check_device_precond.sh =="

# --- BASELINE, and it is the CONVERSE that the firing states need ------------
# There was no baseline in this matrix, so nothing asserted the checker is GREEN
# on an unmodified tree -- and the two states below assert that two guards FIRE.
# A guard that fires on a healthy tree satisfies both of them, so the quiet
# direction is the half that gets forgotten (and it is the half that was actually
# broken: the refusal guards fired AND the comparison ran anyway). Asserted
# through pred_expect_absent so the polarity is named, not a bare grep -q.
echo "-- baseline (unmutated tree must be GREEN and SILENT) --"
mkdir -p "$LOG_DIR" || exit 1
BASE_LOG="$LOG_DIR/mutate_baseline.log"
timeout 1800 bash "$CHECKER" >"$BASE_LOG" 2>&1
BASE_RC=$?
if [ "$BASE_RC" -ne 0 ]; then
  echo "FAIL: the checker does not pass on a clean tree (rc=$BASE_RC), so every" >&2
  echo "  state below would be scored against a broken baseline." >&2
  grep -m3 -E 'FAIL' "$BASE_LOG" | sed 's/^/  /' >&2
  exit 1
fi
pred_reset
if pred_expect_absent baseline "$(cat "$BASE_LOG")" \
     'NO TABLE to cross-check|classifies NO gate as needing|cross-check NOT PERFORMED|command not found'; then
  echo "  ok    baseline: rc=0, both cross-check guards SILENT"
else
  echo "FAIL: a cross-check guard fires on a HEALTHY tree." >&2
  echo "  Every state below would then be satisfied by an always-red guard." >&2
  exit 1
fi
# And the other half of the converse, stated positively: the cross-check must
# actually AGREE on the real tree, or "silent" would also be satisfied by a
# cross-check that never runs at all.
pred_reset
if pred_expect_present baseline "$(cat "$BASE_LOG")" 'both derivations agree: [0-9]+ device gate'; then
  echo "  ok    baseline: the cross-check really does run and agree"
else
  echo "FAIL: the cross-check did not report agreement on a healthy tree." >&2
  exit 1
fi

# M-B first: cheapest, and it mutates a MODE, which a concurrent run silently
# inherits. The clean-state assertion must run BEFORE the chmod, since the chmod
# IS the mutation -- asserting inside run_state would always abort.
assert_clean "$TGT_B" M-B || exit 1
chmod -x "$TGT_B"
run_state "M-B" "fixture execute bit cleared" "$TGT_B" "not executable"
assert_clean "$TGT_B" M-B || exit 1

# M-A
assert_clean "$TGT_A" M-A || exit 1
sed -i '/^gate_require_device || exit 2$/d' "$TGT_A"
verify_patched "$TGT_A" M-A
run_state "M-A" "device guard stripped from a real gate" "$TGT_A" "carries NO guard at all"

# M-C
assert_clean "$TGT_C" M-C || exit 1
sed -i 's/^gate_require_device || exit 2$/gate_require_render_node || exit 2/' "$TGT_C"
verify_patched "$TGT_C" M-C
run_state "M-C" "guard weakened to render_node-only (half a guard)" "$TGT_C" "want 2 (SKIP)"

# M-D: narrow ONLY this checker's token regex, so its list shrinks and the two
# derivations disagree with every gate left intact. Replaces the alternation
# with just --gpu, which drops test_dpas_fold.sh (it names dpas_fold_probe and
# carries no --gpu) while gate_inventory.sh still lists it.
assert_clean "$TGT_H" M-D || exit 1
patch_literal "$TGT_H" "$A_TOKENS" '(--gpu)'
verify_patched "$TGT_H" M-D
run_state "M-D" "checker's own GPU-token regex narrowed (list drifts)" "$TGT_H" "classification drift"

# M-E: rot the seam the way a real simplification would -- gate_gpu_count stops
# ANNOUNCING that it could not count and just reports 0 devices, so the
# "unmeasurable" and "0 devices" paths become indistinguishable.
#
# The first version kept the `echo "SKIP: ... not available"` line, and the state
# then read as NOT load-bearing -- but the fault was the state, not the checker:
# probe_diag takes the FIRST 'SKIPPED|SKIP:' line, and that surviving echo
# precedes gate_require_device's message, so the two diagnostics still differed.
# A rotted seam that stays self-describing is not rotted. Both lines go.
assert_clean "$TGT_P" M-E || exit 1
# *** THE ANCHOR HAD TO BE RE-POINTED, AND THAT IS THE POINT OF THIS STATE. ***
# It used to be the two-line literal <SKIP echo> + <return 1>. Adding
# gate_env_diagnose between those two lines -- which is exactly what the
# ENVIRONMENT-vs-BOX-STATE fix did -- left the two-line adjacency GONE, so
# patch_literal found 0 and ABORTED. That abort is the tooling working: the fault
# was in the STATE, and the instruction it prints is "fix the patch, not the
# checker". Widening the checker to match stale code would have inverted that.
# So the anchor is re-pointed at the CURRENT three-line shape -- echo,
# diagnostic, return -- and the state still rots the seam exactly as intended:
# all three go, and gate_gpu_count again reports 0 devices as if it had counted.
# *** PUT THE DIAGNOSTIC LINE IN ITS OWN SINGLE-QUOTED VARIABLE. *** Three
# attempts at inlining it failed, and the third is the instructive one: inside a
# DOUBLE-quoted argument the obvious `\$sycl_ls` escape did not survive --
# bash carried a literal backslash through, so the anchor read
# `gate_env_diagnose \$sycl_ls' is not executable on $'PATH.` and matched
# nothing. The symptom is IDENTICAL to a stale anchor ("0 occurrences"), so it is
# indistinguishable from the thing it is imitating, which is what made it
# expensive. A_NOTAVAIL survives this because it is SINGLE-quoted at its
# definition; the fix is to give the new line the same treatment rather than to
# out-quote the interpolation.
check_anchor "$TGT_P" "$A_DIAG" "M-E: the environment diagnostic line"
patch_literal "$TGT_P" "$A_NOTAVAIL
$A_DIAG
    return 1" "printf '0\\n'
    return 0"
verify_patched "$TGT_P" M-E
run_state "M-E" "INTFER_SYCL_LS seam rotted (unmeasurable looks like 0 devices)" "$TGT_P" "indistinguishable"

# M-F: reword ONLY self-check case (c)'s synthetic gate, so the always-red case
# actually rejects the correct cause.
assert_clean "$TGT_H" M-F || exit 1
patch_literal "$TGT_H" "$A_CASEC" 'echo "SKIPPED: render node opens but Level Zero enumerates 7 device(s)."'
verify_patched "$TGT_H" M-F
run_state "M-F" "self-check case (c) made always-red" "$TGT_H" "CORRECT cause was REJECTED"

# M-G / M-H: the cross-check's REFUSAL guards, and the reason they are
# ARTIFACT states (audit #21) rather than checker-weakening states. The checker is
# correct by construction once it refuses properly, so weakening IT is
# unobservable -- every stub below is a legitimately reachable gate_inventory
# output, so the cross-check must reject all of them.
#
# The defect these two were written for is SPECIFIC and is not visible in an exit
# code: both guards recorded their refusal and then fell through to the
# comparison anyway, so the log carried the refusal AND "classification drift"
# with an EMPTY gate_inventory side -- sending the next reader to reconcile two
# lists when there was nothing to reconcile. Hence the must-be-absent assertion
# (`$5`), which is why run_state grew that parameter.
assert_clean "$TGT_G" M-G || exit 1
printf '#!/bin/sh\nexit 1\n' >"$TGT_G"          # an early exit: NO table at all
chmod +x "$TGT_G"
verify_patched "$TGT_G" M-G
# The must-be-absent pattern is ANCHORED on the verdict line (`FAIL  <spaces>
# classification drift:`), and that precision was bought by reading the log rather
# than by reasoning about it. A bare `classification drift` matched the REFUSAL's
# own sentence -- "That is not classification drift, so do not debug the gates" --
# so M-G scored MISSED against a checker that had behaved exactly right. The
# anchored form still matches the genuine drift verdict (M-D's log) and still does
# not match the refusal. Same rule as everywhere else here: write the predicate
# from what is ACTUALLY PRINTED, not from what you expect.
A_DRIFT_VERDICT='^ *FAIL +classification drift:'
run_state "M-G" "gate_inventory returns NO table (early exit)" "$TGT_G" \
  "NO TABLE to cross-check" "$A_DRIFT_VERDICT"
assert_clean "$TGT_G" M-G || exit 1

assert_clean "$TGT_G" M-H || exit 1
cat >"$TGT_G" <<'EOS'
#!/bin/sh
# A table that ARRIVES but classifies no gate as needing a device: the header the
# checker greps for is present, every row is `not-gpu`. The audit-#20 shape -- an
# empty harvest is not a disagreement.
printf '%-26s %-9s %-22s %-11s %-12s %s\n' GATE BUILD-GUARD PRECONDITION GRANULARITY 1-GPU-ARM RENDER-NODE
printf '%-26s %-9s %-22s %-11s %-12s %s\n' test_cpu_parity.sh none none whole-gate not-gpu
echo "GATE INVENTORY: OK"
EOS
chmod +x "$TGT_G"
verify_patched "$TGT_G" M-H
run_state "M-H" "gate_inventory's table classifies NO device gate" "$TGT_G" \
  "classifies NO gate as needing" "$A_DRIFT_VERDICT"

# M-I: a gate with an earlier declared precondition -- the UNMEASURED direction.
# The harness must classify it UNMEASURED (never MEASURED), not green.
M_I_GATE="scripts/gates/test_zz_unmeasured_dummy.sh"
cat > "$M_I_GATE" <<'GATE'
#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
. "$SCRIPT_DIR/../lib/_gate_precond.sh"
. "$SCRIPT_DIR/../lib/_gate_build.sh"
gate_require_build intfer_cli || exit 1
if [ ! -f /nonexistent/intfer_unmeasured_fixture.so ]; then
  echo "SKIP: fixture precondition: intfer_unmeasured_fixture.so not installed" >&2
  exit 2
fi
gate_require_device || exit 2
echo "would run --gpu --tp 1 arm"
GATE
chmod +x "$M_I_GATE"
mkdir -p "$LOG_DIR" || exit 1
run_state "M-I" "planted gate reads UNMEASURED, never MEASURED/PASS" "-" "UNMEASURED" "" keep-green
rm -f "$M_I_GATE" "$LOG_DIR"/test_zz_unmeasured_dummy.*.log

# final integrity: every mutated target must equal its startup snapshot (bytes + mode)
if gate_snap_integrity "$DIRTY $TGT_G"; then :; else exit 1; fi

echo
if [ "$CAUGHT" -eq "$TOTAL" ]; then
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$(awk '{printf "%s%s", (n++?",":""), $1}' "$STATES_LOG")"
    echo "caught=$CAUGHT/$TOTAL"
    echo "each_caught_for_its_own_reason=yes"
    echo "baseline=green_and_both_crosscheck_guards_silent"
    echo "healthy_column=skipped_by_design(INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1)"
  echo "checker_blob=$CHECKER_BLOB"
  echo "checker_dirty=$CHECKER_DIRTY"
  gate_snap_attest "$STAMP.attestation" >/dev/null 2>&1 || true
  targets_dirty="${GATE_SNAP_ALLOW_DIRTY:-0}"
  for f in $DIRTY $TGT_G; do
    printf 'target_blob_%s=%s\n' "$(printf '%s' "$f" | tr / _)" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
  echo "snapshot_dir=$GATE_SNAP_DIR"
  } >"$STAMP"
  cat "$STAMP.attestation" 2>/dev/null >>"$STAMP" || true
  echo "DEVICE PRECOND MUTATION: PASS ($CAUGHT/$TOTAL caught, each for its OWN reason)"
  echo "  check_device_precond.sh is non-vacuous: every mechanism this matrix names"
  echo "  can be observed to fail, including the HALF-GUARD case a single-direction"
  echo "  matrix would have scored as a pass, the DRIFT comparison M-A never"
  echo "  exercised, the ROTTED seam, the always-red self-check, and the pair of"
  echo "  cross-check guards that must refuse WITHOUT also printing the misleading"
  echo "  'classification drift' line. Read the live counts from $STAMP."
  echo "  stamp: $STAMP"
  exit 0
fi
echo "DEVICE PRECOND MUTATION: FAIL ($CAUGHT/$TOTAL caught for the right reason)" >&2
echo "  Distinguish the three failure kinds the log prints before editing the" >&2
echo "  checker: not-load-bearing, caught-for-an-unrelated-reason, or" >&2
echo "  unattributed. Only the first means the checker is wrong." >&2
exit 1
