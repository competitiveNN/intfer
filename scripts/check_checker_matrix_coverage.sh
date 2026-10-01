#!/usr/bin/env bash
# Derive the "every checker has a dispatched mutation matrix" claim FROM SOURCE
# and fail if it has drifted.
#
# WHY THIS EXISTS. That claim has only ever been written in prose (session
# summaries, AGENTS.md). A hand-counted total presented as a live claim is
# exactly what gate_inventory.sh's hand-counted-total rule exists to catch, and
# the property itself is load-bearing: three vacuous-greens would have shipped
# green because no matrix made them fail (AUDIT #20). So it is derived, not
# remembered.
#
# The population is enumerated from the filesystem, the same way run_all_checks.sh
# section 0b does -- `check_*.sh` and `mutate_*.sh` -- never from a list.
#
# NOTE ON "ORPHAN" MATRICES: there is deliberately no rule that a matrix must
# have a same-named checker. mutate_fold_matrix.sh has no check_fold_matrix.sh
# because fold_matrix's checker IS the matrix; mutate_gate_inventory.sh and
# mutate_run_all_checks.sh check OTHER files. An earlier version of this script
# flagged all three, i.e. three false alarms on a healthy tree -- and a check that
# cries wolf is the mirror defect: it teaches people to ignore it.
#
# Usage: scripts/check_checker_matrix_coverage.sh
# Exit: 0 = every checker has a matrix and every matrix is dispatched, 1 = drift.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

echo "== CHECKER/MATRIX COVERAGE (derived from scripts/) =="

CHECKERS="$(ls scripts/check_*.sh 2>/dev/null | xargs -r -n1 basename | sort)"
MATRICES="$(ls scripts/mutate_*.sh 2>/dev/null | xargs -r -n1 basename | sort)"
n_chk="$(printf '%s\n' "$CHECKERS" | grep -c . || true)"
n_mat="$(printf '%s\n' "$MATRICES" | grep -c . || true)"

# The enumeration must actually harvest something. With none found, every check
# below would pass over an empty population -- the sites_found=0 vacuous-green
# class, which is why this guard exists rather than a bare loop.
if [ "$n_chk" -eq 0 ]; then
  echo "  FAIL  no check_*.sh found -- enumeration is broken, so this would be vacuous"
  exit 1
fi
echo "-- population: $n_chk checkers, $n_mat matrices (enumerated, not remembered)"

bad=0
printf '%-36s %-34s %s\n' "checker" "matrix" "dispatch arm"
for c in $CHECKERS; do
  stem="${c%.sh}"; stem="${stem#check_}"
  m="mutate_${stem}.sh"
  have="MISSING"
  printf '%s\n' "$MATRICES" | grep -qx "$m" && have="$m"
  # grep rc: 0 match, 1 no-match, >=2 the QUERY FAILED. A failed query must not
  # become arm="NO" -- that is a verdict about code the query never examined,
  # and it is the class that bit this session twice in my own tooling.
  arm="NO"; arm_rc=0
  grep -qxF -- "    $m)" scripts/run_all_checks.sh >/dev/null 2>&1 || arm_rc=$?
  case "$arm_rc" in
    0) arm="yes" ;;
    1) arm="NO" ;;
    *) arm="QUERY-FAILED" ;;
  esac
  # REQUIREMENT 1: a checker with no matrix is drift -- nothing can falsify it.
  [ "$have" = "MISSING" ] && bad=$((bad+1))
  # REQUIREMENT 2: a matrix with no dispatch arm NEVER RUNS (audit #18), so the
  # suite would still print PASS. Also drift.
  [ "$arm" = "NO" ] && bad=$((bad+1))
  [ "$arm" = "QUERY-FAILED" ] && bad=$((bad+1))
  printf '%-36s %-34s %s\n' "$c" "$have" "$arm"
done

echo
if [ "$bad" -eq 0 ]; then
  echo "CHECKER/MATRIX COVERAGE: PASS ($n_chk/$n_chk checkers have a dispatched matrix)"
  echo "RESULT: coverage derived from scripts/ -- $n_chk checkers, $n_mat matrices, 0 drift"
  exit 0
fi
echo "CHECKER/MATRIX COVERAGE: FAIL ($bad drift(s))"
echo "RESULT: coverage claim does NOT match what is on disk"
exit 1