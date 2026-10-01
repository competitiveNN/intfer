#!/usr/bin/env bash
# Reproducible non-vacuity proof for scripts/check_device_precond.sh.
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
# FAIL CLOSED on a dirty tree: this script edits tracked source and restores it
# with `git checkout --`, which would destroy uncommitted work.
#
# Env: INTFER_BUILD_DIR  build tree (default $ROOT/build) for logs and the stamp.
# Exit: 0 all states caught for the right reason; 1 otherwise.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/device_precond_matrix"
STAMP="$LOG_DIR/mutate_last_pass.stamp"
CHECKER="scripts/check_device_precond.sh"
LOCK="$LOG_DIR/.mutate.lock"
export INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1
mkdir -p "$LOG_DIR" || exit 1

# --- lock --------------------------------------------------------------------
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_device_precond.sh is running (or died holding $LOCK)." >&2
  echo "  These states edit the SAME tracked files, so two concurrent runs would" >&2
  echo "  cross-contaminate each other's verdicts -- a real, already-observed" >&2
  echo "  failure. If no run is active, remove the lock and re-run." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

TGT_A="scripts/test_dpas_fold.sh"
TGT_C="scripts/test_http_batch.sh"
TGT_B="scripts/fixtures/no-render-node/python3"
TGT_H="scripts/check_device_precond.sh"
TGT_P="scripts/_gate_precond.sh"
DIRTY="$TGT_A $TGT_C $TGT_B $TGT_H $TGT_P"
if ! git diff --quiet -- $DIRTY || ! git diff --cached --quiet -- $DIRTY; then
  echo "FAIL: uncommitted changes in: $DIRTY" >&2
  echo "  This script edits them and restores with 'git checkout --', which would" >&2
  echo "  DESTROY that work. Commit or stash first, then re-run." >&2
  exit 1
fi

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
A_TOKENS='(--gpu|sycl-ls|level_zero|xpu-smi|dpas_fold_probe)'
A_NOTAVAIL='echo "SKIP: $sycl_ls not available; cannot count GPUs" >&2'
A_CASEC='echo "SKIPPED: render node opens but Level Zero enumerates 0 device(s)."'
check_anchor "$TGT_A" "$A_GUARD" "M-A: the whole-gate device guard to strip"
check_anchor "$TGT_C" "$A_GUARD" "M-C: the guard line to weaken to render_node-only"
check_anchor "$TGT_B" "$A_EXEC"  "M-B: the fixture whose execute bit is cleared"
check_anchor "$TGT_H" "$A_TOKENS" "M-D: the GPU-token regex this checker derives its list from"
check_anchor "$TGT_P" "$A_NOTAVAIL" "M-E: the unmeasurable branch of the count helper"
check_anchor "$TGT_H" "$A_CASEC" "M-F: self-check case (c)'s synthetic correct-cause gate"
echo "anchor pre-flight: ok (6/6 targets located exactly once)"

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

# AND prove the patch landed. An unverified patch is worse than no state: if it
# silently does nothing, the checker passes, and the log reports "the mechanism is
# NOT load-bearing" -- a false conclusion about the CHECKER, when the real fault
# is in the state. BOTH M-D and M-E failed this way on the first run (a sed
# delimiter collision, and a rot that was too weak to reproduce the defect).
# So every patch is followed by a changed-file assertion.
verify_patched() { # $1 = file, $2 = state id
  if git diff --quiet -- "$1"; then
    echo "  ABORT $2  the patch did NOT change $1." >&2
    echo "         The state would score as 'not load-bearing' -- a conclusion" >&2
    echo "         about the CHECKER that is actually a fault in the state." >&2
    echo "         Fix the patch, not the checker. (A common cause: a sed" >&2
    echo "         delimiter that occurs inside the pattern.)" >&2
    exit 1
  fi
}

assert_clean() { # $1 = file, $2 = state id -- must run BEFORE the mutation
  if ! git diff --quiet -- "$1"; then
    echo "  ABORT $2  pre-state DIRTY: $1 already modified (leftover from a killed" >&2
    echo "         run, or real uncommitted work). Refusing to score a state whose" >&2
    echo "         starting point is unknown -- that is how a state fakes a PASS." >&2
    echo "         Inspect, then 'git checkout -- $1' if the change was ours." >&2
    return 1
  fi
  return 0
}

run_state() { # $1 = id, $2 = description, $3 = file to restore, $4 = expected reason
  local id="$1" desc="$2" restore="$3" want="$4" log rc reason verdict
  TOTAL=$((TOTAL + 1))
  log="$LOG_DIR/mutate_${id}.log"
  timeout 1800 bash "$CHECKER" >"$log" 2>&1
  rc=$?
  reason="$(grep -m1 -E 'FAIL' "$log" | cut -c1-170)"
  verdict=MISSED
  if [ "$rc" -eq 0 ]; then
    reason="<checker still exited 0>"
  elif [ -z "$reason" ]; then
    reason="<checker failed but named no assertion>"
  elif ! printf '%s' "$reason" | grep -qF -- "$want"; then
    reason="WRONG REASON: caught, but not by '$want' -> $reason"
  else
    verdict=ok
    CAUGHT=$((CAUGHT + 1))
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
      *)
        echo "         -> unattributed catch: a non-zero exit with no failing" | sed 's/^ *//'
        echo "            assertion named is not evidence of anything." | sed 's/^ *//' ;;
    esac
  fi
  printf '%-4s verdict=%-6s want=%s\n' "$id" "$verdict" "$want" >>"$STATES_LOG"
  # Restoring text is not restoring the experiment: prove the tree is clean
  # afterwards, or the next state inherits this mutation. `git checkout --`
  # restores the recorded mode too (verified: 644 -> 755), so M-B is covered.
  git checkout -- "$restore" 2>/dev/null
  if git diff --quiet -- "$restore"; then
    printf '         restored: %s clean\n' "$restore"
  else
    printf '  MISSED %-4s RESTORE FAILED for %s -- manual repair needed\n' "$id" "$restore"
    CAUGHT=$((CAUGHT - 1))
  fi
  return 0
}

echo "== MUTATION MATRIX: check_device_precond.sh =="

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
run_state "M-C" "guard weakened to render_node-only (half a guard)" "$TGT_C" "exited rc=0, want 2"

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
patch_literal "$TGT_P" "$A_NOTAVAIL
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

echo
if [ "$CAUGHT" -eq "$TOTAL" ]; then
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=M-B,M-A,M-C,M-D,M-E,M-F"
    echo "caught=$CAUGHT/$TOTAL"
    echo "each_caught_for_its_own_reason=yes"
    echo "healthy_column=skipped_by_design(INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1)"
  } >"$STAMP"
  echo "DEVICE PRECOND MUTATION: PASS ($CAUGHT/$TOTAL caught, each for its OWN reason)"
  echo "  check_device_precond.sh is non-vacuous: all six load-bearing mechanisms"
  echo "  can be observed to fail, including the HALF-GUARD case a single-direction"
  echo "  matrix would have scored as a pass, the DRIFT comparison M-A never"
  echo "  exercised, the ROTTED seam, and the always-red self-check."
  echo "  stamp: $STAMP"
  exit 0
fi
echo "DEVICE PRECOND MUTATION: FAIL ($CAUGHT/$TOTAL caught for the right reason)" >&2
echo "  Distinguish the three failure kinds the log prints before editing the" >&2
echo "  checker: not-load-bearing, caught-for-an-unrelated-reason, or" >&2
echo "  unattributed. Only the first means the checker is wrong." >&2
exit 1
