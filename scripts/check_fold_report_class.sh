#!/usr/bin/env bash
# Behavioural self-check for the CLASSIFICATION in scripts/fold_report_counts.sh.
#
# WHY. That script's whole job is to keep two classes apart: rc=1 means "a real
# regression, go fix the code" and rc=2 means "the evidence is about an older
# tree, re-run the matrix". Those print OPPOSITE remedies, so a misclassification
# sends the reader the wrong way. It WAS misclassified, and measured: the stale
# branch was reachable only when summary.txt was clean, so a damaged summary left
# by a previous failed run exited 1 -- "regression" -- when the truth was "stale
# evidence". A prose note about that would be exactly the defect this repo audits;
# so the four directions are asserted here, in a sandbox, every run.
#
# AUDIT #21: the cases mutate the INPUT STATE (summary contents and stamp sha),
# never the classifier's leniency.
#
# Usage: scripts/check_fold_report_class.sh
# Exit: 0 = all four directions correct, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/intfer_foldclass.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
STAMP_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/fold_matrix"
HEAD_SHA="$(git -C "$ROOT" rev-parse HEAD)"
CLEAN_SHA="0000000000000000000000000000000000000000"
bad=0

echo "== FOLD REPORT CLASSIFICATION CHECK =="

if [ ! -d "$STAMP_DIR" ]; then
  echo "SKIP: $STAMP_DIR absent (no matrix has run); nothing to classify."
  echo "       Run scripts/mutate_fold_matrix.sh first."
  exit 2
fi
SB_SUM="$(mktemp)"; SB_STAMP="$(mktemp)"
cp "$STAMP_DIR/summary.txt" "$SB_SUM"; cp "$STAMP_DIR/last_pass.stamp" "$SB_STAMP"

restore() { cp "$SB_SUM" "$STAMP_DIR/summary.txt"; cp "$SB_STAMP" "$STAMP_DIR/last_pass.stamp"; }
cleanup() { rm -f "$SB_SUM" "$SB_STAMP"; rm -rf "$WORK"; }
trap 'cleanup' EXIT

damaged() {
  printf 'S0_clean AS EXPECTED\nS1_m1 UNEXPECTED\nS2_m2 UNEXPECTED\nS3_m3 UNEXPECTED\nS4_r UNEXPECTED\nS5_m5 UNEXPECTED\nS6_idx UNEXPECTED\n' \
    > "$STAMP_DIR/summary.txt"
}
set_sha() { sed "s/^sha=.*/sha=$1/" "$SB_STAMP" > "$STAMP_DIR/last_pass.stamp"; }
run_rc() { "$SCRIPT_DIR/fold_report_counts.sh" >/dev/null 2>&1; echo $?; }

expect() { # $1 label, $2 want, $3 got
  if [ "$2" = "$3" ]; then
    echo "  ok    $1 -> rc=$3"
  else
    echo "  FAIL  $1 -> rc=$3, want $2"
    bad=$((bad+1))
  fi
}

echo "-- the four classification directions --"
# The stamp is set EXPLICITLY rather than inherited from the ambient one. At
# phase 1/2 of run_all_checks.sh this script runs BEFORE mutate_fold_matrix, so
# the on-disk stamp is still the previous run's and predates HEAD -- the case
# then read rc=2 and failed for a reason that had nothing to do with the
# classifier. A case must control its own fixture.
restore; set_sha "$HEAD_SHA"; expect "clean summary + current stamp" 0 "$(run_rc)"
restore; damaged; set_sha "$CLEAN_SHA"
expect "damaged summary + STALE stamp (was rc=1)"     2 "$(run_rc)"
restore; damaged; set_sha "$HEAD_SHA"
expect "damaged summary + current stamp = REAL regression" 1 "$(run_rc)"
restore; cp "$SB_SUM" /dev/null 2>/dev/null; rm -f "$STAMP_DIR/summary.txt"
set_sha "$CLEAN_SHA"
expect "missing summary + STALE stamp"                2 "$(run_rc)"
restore; rm -f "$STAMP_DIR/last_pass.stamp"
expect "no stamp at all"                              2 "$(run_rc)"

restore
echo
if [ "$bad" -eq 0 ]; then
  echo "FOLD REPORT CLASSIFICATION: PASS (rc=1 only for a real regression on current evidence)"
  echo "RESULT: fold_report_counts keeps regression and stale-evidence apart in both directions"
  exit 0
fi
echo "FOLD REPORT CLASSIFICATION: FAIL ($bad direction(s))"
echo "RESULT: fold_report_counts MISCLASSIFIES evidence -- the printed remedy may be wrong"
exit 1