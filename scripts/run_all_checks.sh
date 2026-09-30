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
# "citation vs guess" line -- see scripts/fold_report_counts.sh.)
#
# Env:
#   INTFER_BUILD_DIR   build tree (default $ROOT/build) for logs and stamps
#   INTFER_SKIP_HEALTHY=1  forward to the device-precond checker to skip its
#                     healthy column (saves ~6.5 min; the stamps then record the
#                     column as NOT RUN, which the assertion below still sees as
#                     a valid stamp for that sha)
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

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another run_all_checks.sh is running (or died holding $LOCK)." >&2
  echo "  The mutation matrices edit tracked source, so a concurrent run would" >&2
  echo "  restore another run's mutations over its own. If no run is active," >&2
  echo "  remove $LOCK and re-run." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

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
[ "${INTFER_QUICK:-0}" = "1" ] && echo "run_all_checks: INTFER_QUICK=1 -- checkers only, NO mutation matrices"

# --- run a step, record its rc ---------------------------------------------
# rc 2 from a step means UNPROVEN, not broken. fold_report_counts.sh exits 2 when
# its stamp predates HEAD: "Nothing is wrong with the code; the EVIDENCE is about
# an older tree." Reporting that as FAIL would be the exact doctrine violation this
# repo audits for -- conflating missing evidence with a regression, so a red that
# means "re-run the matrix" is indistinguishable from a red that means "fix the
# code", and people learn to ignore both.
RC_SUMMARY=""
FAILED=0
UNPROVEN=0
UNPROVEN_LABELS=""

# One place that knows what each step's command is, so the post-matrix re-run
# dispatches through the SAME definition instead of a second hand-maintained list.
# A duplicated table is exactly how a step silently stops being re-run.
dispatch() { # $1 = label -> runs it
  case "$1" in
    gate_inventory.sh)
      timeout 1800 bash "$SCRIPT_DIR/gate_inventory.sh" ;;
    fold_report_counts.sh)
      timeout 1800 bash "$SCRIPT_DIR/fold_report_counts.sh" ;;
    check_device_precond.sh)
      if [ "${INTFER_SKIP_HEALTHY:-0}" = "1" ]; then
        INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1 timeout 1800 bash "$SCRIPT_DIR/check_device_precond.sh"
      else
        timeout 3600 bash "$SCRIPT_DIR/check_device_precond.sh"
      fi ;;
    mutate_fold_matrix.sh)
      timeout 7200 bash "$SCRIPT_DIR/mutate_fold_matrix.sh" ;;
    mutate_gate_inventory.sh)
      timeout 7200 bash "$SCRIPT_DIR/mutate_gate_inventory.sh" ;;
    mutate_device_precond.sh)
      timeout 7200 bash "$SCRIPT_DIR/mutate_device_precond.sh" ;;
    *) echo "internal: unknown step '$1'" >&2; return 97 ;;
  esac
}

step() { # $1 = label
  local label="$1" log rc
  log="$BUILD_DIR/run_all_$(printf '%s' "$label" | tr -c 'a-zA-Z0-9' '_').log"
  printf '  %-34s ' "$label"
  dispatch "$label" >"$log" 2>&1
  rc=$?
  case "$rc" in
    0) printf 'ok\n' ;;
    2) printf 'UNPROVEN rc=2 (stale evidence, not a regression -- %s)\n' "$log"
       UNPROVEN=$((UNPROVEN + 1))
       UNPROVEN_LABELS="$UNPROVEN_LABELS $label" ;;
    *) printf 'FAIL rc=%s (see %s)\n' "$rc" "$log"
       FAILED=$((FAILED + 1)) ;;
  esac
  RC_SUMMARY="$RC_SUMMARY$(printf '%-34s %s\n' "$label" "rc=$rc")\n"
  return 0
}

echo "== 1/2 checkers =="
step "gate_inventory.sh"
step "fold_report_counts.sh"
step "check_device_precond.sh"

if [ "${INTFER_QUICK:-0}" != "1" ]; then
  echo "== 2/2 mutation matrices (slow; each edits and restores tracked source) =="
  step "mutate_fold_matrix.sh"
  step "mutate_gate_inventory.sh"
  step "mutate_device_precond.sh"

  # --- re-validate anything that was unproven -------------------------------
  # ORDERING, and the first run got this wrong. fold_report_counts.sh reads the
  # fold matrix's stamp, and the checkers run BEFORE the matrices -- so on a tree
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
      if dispatch "$label" >"$log" 2>&1; then
        printf 'ok (evidence now current)\n'
        UNPROVEN=$((UNPROVEN - 1))
      else
        rc=$?
        printf 'still rc=%s (%s)\n' "$rc" "$log"
      fi
    done
  fi
fi

# --- 3. the stamp assertion, over an ENUMERATED set -------------------------
# find, not a table. A hand-written list is a table that quietly stops covering a
# matrix when one is added.
echo "== 3/3 stamp assertion (enumerated) =="
STALE=0
STAMP_COUNT=0
while IFS= read -r stamp; do
  [ -n "$stamp" ] || continue
  STAMP_COUNT=$((STAMP_COUNT + 1))
  sha="$(sed -n 's/^sha=//p' "$stamp" | head -1)"
  extra="$(sed -n 's/^\(caught\|states\)=//p' "$stamp" | head -1)"
  rel="${stamp#$BUILD_DIR/}"
  if [ "$sha" = "$HEAD_SHA" ]; then
    printf '  ok    %-46s %s %s\n' "$rel" "${sha:0:7}" "$extra"
  else
    printf '  STALE %-46s %s (want %s)\n' "$rel" "${sha:0:7}" "${HEAD_SHORT}"
    STALE=$((STALE + 1))
  fi
done <<EOF
$(find "$BUILD_DIR" -name '*last_pass.stamp' -type f 2>/dev/null | sort)
EOF
if [ "$STAMP_COUNT" -eq 0 ]; then
  echo "  FAIL  no stamps found under $BUILD_DIR -- nothing was proven." >&2
  STALE=$((STALE + 1))
fi

# --- verdict ----------------------------------------------------------------
echo
if [ "$FAILED" -ne 0 ]; then
  echo "RUN ALL CHECKS: FAIL ($FAILED step(s) FAILED) -- a real regression." >&2
  [ "$UNPROVEN" -ne 0 ] && echo "  plus $UNPROVEN step(s) unproven (stale evidence)." >&2
  exit 1
fi
if [ "$STALE" -ne 0 ] || [ "$UNPROVEN" -ne 0 ]; then
  echo "RUN ALL CHECKS: $STAMP_COUNT stamp(s) checked, $STALE STALE, $UNPROVEN step(s) unproven, for $HEAD_SHORT." >&2
  echo "  NOT a regression: the evidence does not cover this commit. Nothing is" >&2
  echo "  wrong with the code; the proof is just older than the tree." >&2
  echo "  Re-run the named matrix (or all of them) so its stamp names $HEAD_SHORT." >&2
  exit 2
fi
echo "RUN ALL CHECKS: PASS ($STAMP_COUNT stamp(s), all at $HEAD_SHORT)"
echo "  every checker and matrix passed, and every enumerated stamp names HEAD."
if [ "${INTFER_QUICK:-0}" = "1" ]; then
  echo "  NOTE: INTFER_QUICK=1 -- no matrix was RE-RUN, so only pre-existing stamps"
  echo "  were asserted. That is weaker than a full run: re-run without the flag."
fi
exit 0
