#!/usr/bin/env bash
# Print the fold mutation-matrix report table CELLS, derived from the per-state
# logs, instead of leaving them as hand-transcribed prose.
#
# WHY. The report (reports/2026-09-30-dpas48-fold.md) carries a 7-row table whose
# numeric cells drifted twice, both times silently:
#   * S3's contract cell read "42 viol." -- that is the probe's AGGREGATE
#     (contract + index + readback). The contract column is 40.
#   * AGENTS.md separately carried "S4=2", a figure from BEFORE the index check
#     existed at all (now 720 index / 722 aggregate).
# Three quantities, similar labels, no stated basis. This script prints all
# three, named, so the report's cell can be traced to a command.
#
# It also prints the 7/7 state count and the per-state exit codes, which is what
# "5/5 as characterized" (a third stale form, found in three separate files)
# should have been traceable to.
#
# READ-ONLY. It runs no mutation, builds nothing, and needs no GPU -- it only
# reads build/fold_matrix/*.log. Safe to run at any time; it is not the matrix
# (that is scripts/mutate_fold_matrix.sh, which edits tracked files).
#
# Usage: scripts/fold_report_counts.sh
# Env:   INTFER_BUILD_DIR  (default <root>/build)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/fold_matrix"

if [ ! -d "$LOG_DIR" ]; then
  echo "fold-counts: NO LOGS at $LOG_DIR" >&2
  echo "  Run scripts/mutate_fold_matrix.sh once to produce them. Nothing inferred." >&2
  exit 2
fi

STATES="S0_clean S1_M1_default_only S2_M2_noval_only S3_M1_and_M2 S4_reader_index S5_writer_gate S6_index_s0"

printf '%-22s %-4s %-10s %-10s %-12s %s\n' STATE rc contract index readback "probe-agg"
printf '%s\n' "----------------------------------------------------------------------------------------"

missing=0; found=0
for s in $STATES; do
  f="$LOG_DIR/$s.log"
  if [ ! -f "$f" ]; then
    printf '%-22s %s\n' "$s" "MISSING LOG"
    missing=$((missing + 1))
    continue
  fi
  found=$((found + 1))
  # Contract: violated iff the summary line carries the "-- NO" suffix.
  if grep -qE '^ *contract: [0-9]+ \(nOp,stride\) pairs.* -- NO' "$f"; then c="violated"; else c="clean"; fi
  # Index: wrong iff a "halves WRONG" line appears.
  if grep -qE '^ *index: +[0-9]+ .*WRONG' "$f"; then i="wrong"; else i="correct"; fi
  # Readback: count the FAILing regimes out of the 3.
  rb="$(grep -cE '^ *readback .*FAIL' "$f")"
  # rc: the probe's own last-line verdict, not this script's opinion.
  if grep -q '^\[dpas-fold-probe\] PASS' "$f"; then rc=0; else rc=1; fi
  # Probe aggregate: the single combined "N violations" figure on the last line.
  agg="$(grep -oE '[0-9]+ violations' "$f" | tail -1 | cut -d' ' -f1)"
  printf '%-22s %-4s %-10s %-10s %-12s %s\n' "$s" "$rc" "$c" "$i" "${rb}/3 regimes" "${agg:-n/a}"
done

echo
printf 'states_present=%s states_expected=%s\n' "$found" "$(echo $STATES | wc -w)"
printf 'matrix_verdict: '
if [ "$found" -eq 0 ]; then
  echo "CANNOT DETERMINE (no logs)" >&2
  exit 2
fi
if [ "$found" -ne "$(echo $STATES | wc -w)" ]; then
  echo "PARTIAL ($missing state log(s) missing -- the 7/7 claim is NOT supported by these logs)" >&2
  exit 1
fi
# The verdict is derived from summary.txt's per-state rows, which DO carry
# "AS EXPECTED"/"UNEXPECTED". (The script's own "MATRIX: PASS" banner goes to
# stdout, not into summary.txt -- checking for that string here reported NOT
# VERIFIED on a run that passed 7/7, i.e. a check that cries wolf on success.)
# So: 7 expected rows, 0 unexpected, 7 logs present.
if [ -f "$LOG_DIR/summary.txt" ]; then
  n_ok="$(grep -c 'AS EXPECTED' "$LOG_DIR/summary.txt")"
  n_bad="$(grep -c 'UNEXPECTED' "$LOG_DIR/summary.txt")"
  if [ "$n_ok" = "$found" ] && [ "$n_bad" = "0" ]; then
    echo "7/7 as characterized (summary.txt: $n_ok AS EXPECTED, $n_bad UNEXPECTED)"
  else
    echo "NOT VERIFIED (summary.txt: $n_ok AS EXPECTED, $n_bad UNEXPECTED)" >&2
    exit 1
  fi
else
  echo "NOT VERIFIED (no $LOG_DIR/summary.txt)" >&2
  exit 1
fi

cat <<'EOF'

COUNTING BASIS -- the whole point of this script:
  contract / index  = the two CHECKS' own verdicts (violated/correct, wrong/correct).
  readback          = how many of the probe's 3 device readback regimes FAILED.
  probe-agg         = the probe's final combined "N violations" counter, which
                      SUMS contract + index + readback. It is a DIFFERENT
                      quantity from either check column and must never be
                      quoted in one of them.
Report cells should be filled from the check columns, and the "5/5" or "7/7"
state count from the verdict line above -- not from memory.
EOF
