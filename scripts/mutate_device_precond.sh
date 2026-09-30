#!/usr/bin/env bash
# Reproducible non-vacuity proof for scripts/check_device_precond.sh.
#
# WHY THIS EXISTS. Audit #17 committed that checker and the two device-failure
# fixtures, but the first proof that the checker could actually FAIL was three
# hand-run mutations in a terminal transcript. That is the identical defect the
# fixtures were committed to end, one level up: "a mutation that only exists in a
# transcript cannot be re-run by the next session, so the next session
# re-derives (or worse, re-asserts) it." The house already answers this twice
# (mutate_fold_matrix.sh, mutate_gate_inventory.sh); this is the third.
#
# Each state breaks ONE mechanism in the checker or in a real gate, and the
# checker must then exit non-zero. A state that still passes is scored MISSED
# and fails the run.
#
#   M-A  strip a gate's device guard entirely
#        The "needs a device and forgot to say so" case (audit #15): the gate
#        then exits 0 on a box with no usable device, silently claiming coverage
#        it never provided. Must be caught by the classification cross-check.
#   M-B  clear a fixture's execute bit
#        PATH lookup SKIPS a non-executable file, so a 644 stub hands every probe
#        back to the real tool and BOTH failure directions become no-ops that
#        still report green. The most dangerous shape of this bug: it does not
#        fail, it stops testing. Must be caught by the fixture pre-flight.
#   M-C  weaken gate_require_device to gate_require_render_node
#        A HALF guard: still correct for the node-unreachable direction, blind
#        to the 0-device (libumf) direction -- which is the one this box actually
#        hits. The gate then runs and exits 0 when libumf is missing. This is the
#        audit-#16 defect in its purest form, and the reason the harness checks
#        BOTH directions rather than one: a single-direction matrix would score
#        this state as PASS.
#
# ANTI-PARALLELISM, and it is not hypothetical. An earlier hand-run of M-B and
# M-C in a PARALLEL tool batch shared this working tree, so M-C inherited M-B's
# chmod and "passed" by aborting on M-B's pre-flight -- a cross-contaminated
# result that would have been reported as a clean catch. Two of these states
# touch the SAME files, so they must never run concurrently; the lock below
# makes that structural instead of a convention, and each state re-asserts the
# clean pre-state so a leftover from a killed run is caught, not inherited.
#
# FAIL CLOSED on a dirty tree: this script edits tracked source and restores it
# with `git checkout --`, which would destroy uncommitted work.
#
# Env: INTFER_BUILD_DIR  build tree (default $ROOT/build) for logs and the stamp.
# Exit: 0 all states caught; 1 any MISSED or a broken precondition.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/device_precond_matrix"
STAMP="$LOG_DIR/mutate_last_pass.stamp"
CHECKER="scripts/check_device_precond.sh"
LOCK="$LOG_DIR/.mutate.lock"
mkdir -p "$LOG_DIR" || exit 1

# --- lock (see ANTI-PARALLELISM) -------------------------------------------
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_device_precond.sh is running (or died holding $LOCK)." >&2
  echo "  These states edit the SAME tracked gates, so two concurrent runs would" >&2
  echo "  cross-contaminate each other's verdicts -- a real, already-observed" >&2
  echo "  failure. If no run is active, remove the lock and re-run." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# --- fail closed on a dirty tree -------------------------------------------
TGT_A="scripts/test_dpas_fold.sh"
TGT_C="scripts/test_http_batch.sh"
TGT_B="scripts/fixtures/no-render-node/python3"
if ! git diff --quiet -- "$TGT_A" "$TGT_C" "$TGT_B" \
   || ! git diff --cached --quiet -- "$TGT_A" "$TGT_C" "$TGT_B"; then
  echo "FAIL: $TGT_A / $TGT_C / $TGT_B have uncommitted changes." >&2
  echo "  This script edits them and restores with 'git checkout --', which would" >&2
  echo "  DESTROY that work. Commit or stash first, then re-run." >&2
  exit 1
fi

# --- ANCHOR PRE-FLIGHT ------------------------------------------------------
# A refactor can move or delete a mutation's target; the sed then matches
# nothing, the file is unchanged, the checker passes, and the state reads AS
# EXPECTED for the wrong reason. Each target must occur EXACTLY ONCE, up front.
# Patterns live in a QUOTED ARRAY with an explicit per-row file -- audit #16
# found the earlier `pat|desc` + IFS='|' table silently TRUNCATED at the first
# `|`, which these patterns themselves contain (ERE alternation, `||`).
ANCHOR_FILE=("$TGT_A" "$TGT_B" "$TGT_C")
ANCHOR_DESC=(
  "M-A: the whole-gate device guard to strip"
  "M-B: the fixture whose execute bit is cleared"
  "M-C: the guard line to weaken to render_node-only"
)
ANCHOR_PAT=(
  'gate_require_device || exit 2'
  # M-B mutates a MODE, so its anchor must be a structural line that proves
  # this is still the node-stub and not something else. 'python3' is NOT unique
  # here (it occurs 3x: two comments + the exec), and the pre-flight caught
  # exactly that -- an anchor that matches several sites proves nothing about
  # which one is being mutated.
  'exec /usr/bin/python3'
  'gate_require_device || exit 2'
)
for i in 0 1 2; do
  n="$(grep -cF -- "${ANCHOR_PAT[$i]}" "${ANCHOR_FILE[$i]}" 2>/dev/null || echo 0)"
  if [ "$n" != "1" ]; then
    echo "FAIL: anchor for ${ANCHOR_DESC[$i]} found $n times in ${ANCHOR_FILE[$i]}, want 1." >&2
    echo "  Target: ${ANCHOR_PAT[$i]}" >&2
    echo "  Re-point the anchor (do NOT loosen it) if the guard moved." >&2
    exit 1
  fi
done
echo "anchor pre-flight: ok (3/3 targets located exactly once)"

CAUGHT=0
TOTAL=0
UNATTRIBUTED=0
STATES_LOG="$LOG_DIR/mutate_states.txt"
: >"$STATES_LOG"

assert_clean() { # $1 = file, $2 = state id  -- must run BEFORE the mutation
  if ! git diff --quiet -- "$1"; then
    echo "  ABORT $2  pre-state DIRTY: $1 already modified (leftover from a killed" >&2
    echo "         run, or real uncommitted work). Refusing to score a state whose" >&2
    echo "         starting point is unknown -- that is how a state fakes a PASS." >&2
    echo "         Inspect, then 'git checkout -- $1' if the change was ours." >&2
    return 1
  fi
  return 0
}

run_state() { # $1 = id, $2 = description, $3 = file to restore (content AND mode)
  local id="$1" desc="$2" restore="$3" log rc
  TOTAL=$((TOTAL + 1))
  log="$LOG_DIR/mutate_${id}.log"
  timeout 600 bash "$CHECKER" >"$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    CAUGHT=$((CAUGHT + 1))
    printf '  ok    %-4s caught (checker rc=%s)  %s\n' "$id" "$rc" "$desc"
    # Record WHICH assertion fired: a catch for an unrelated reason (e.g. a
    # pre-flight abort) is not evidence this state is load-bearing. This is the
    # distinction that caught the contaminated M-C of the first hand-run.
    #
    # The pattern must match FAIL ANYWHERE on the line, not at line start: the
    # matrix prints a per-cell prefix ("  test_x.sh  zero-device  ") BEFORE the
    # FAIL, so a '^ *FAIL' anchor silently dropped M-C's reason and left a real
    # catch looking unattributed. Measured, not assumed.
    reason="$(grep -m1 -E 'FAIL' "$log" | cut -c1-150)"
    if [ -n "$reason" ]; then
      printf '         via: %s\n' "$(printf '%s' "$reason" | sed 's/^ *//')"
    else
      printf '         via: <NONE FOUND> -- the catch is UNATTRIBUTED. Treat as\n'
      printf '               unverified: a checker that fails without naming a reason\n'
      printf '               may be failing for an unrelated reason.\n'
      UNATTRIBUTED=$((UNATTRIBUTED + 1))
    fi
  else
    printf '  MISSED %-4s checker still PASSED  %s\n' "$id" "$desc"
    printf '         -> the mechanism this state breaks is NOT load-bearing.\n'
  fi
  printf '%-4s caught=%s  %s\n' "$id" "$([ "$rc" -ne 0 ] && echo yes || echo NO)" "$desc" >>"$STATES_LOG"
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
# inherits (that is exactly how the first hand-run of M-C went wrong). The
# clean-state assertion must run BEFORE the chmod, since the chmod IS the
# mutation -- asserting inside run_state would always abort.
assert_clean "$TGT_B" M-B || exit 1
chmod -x "$TGT_B"
run_state "M-B" "fixture execute bit cleared (PATH would bypass the stub)" "$TGT_B"

# M-A
assert_clean "$TGT_A" M-A || exit 1
sed -i '/^gate_require_device || exit 2$/d' "$TGT_A"
run_state "M-A" "device guard stripped from a real gate" "$TGT_A"

# M-C
assert_clean "$TGT_C" M-C || exit 1
sed -i 's/^gate_require_device || exit 2$/gate_require_render_node || exit 2/' "$TGT_C"
run_state "M-C" "guard weakened to render_node-only (half a guard)" "$TGT_C"

echo
if [ "$CAUGHT" -eq "$TOTAL" ] && [ "$UNATTRIBUTED" -eq 0 ]; then
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=M-B,M-A,M-C"
    echo "caught=$CAUGHT/$TOTAL"
    echo "unattributed=$UNATTRIBUTED"
  } >"$STAMP"
  echo "DEVICE PRECOND MUTATION: PASS ($CAUGHT/$TOTAL caught, $UNATTRIBUTED unattributed)"
  echo "  check_device_precond.sh is non-vacuous: every one of its load-bearing"
  echo "  mechanisms can be observed to fail, including the HALF-GUARD case that a"
  echo "  single-direction matrix would have scored as a pass."
  echo "  stamp: $STAMP"
  exit 0
fi
echo "DEVICE PRECOND MUTATION: FAIL ($CAUGHT/$TOTAL caught, $UNATTRIBUTED unattributed)" >&2
echo "  A state the checker does not fail on is evidence nothing: either the" >&2
echo "  mechanism is decorative, or the state failed to REPRODUCE its defect." >&2
echo "  Distinguish those two before editing the checker." >&2
if [ "$UNATTRIBUTED" -ne 0 ]; then
  echo "  $UNATTRIBUTED catch(es) named no failing assertion, so they are not yet" >&2
  echo "  evidence: a checker can exit non-zero for an unrelated reason." >&2
fi
exit 1
