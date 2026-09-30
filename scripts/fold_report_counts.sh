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
  # Index: wrong iff a "halves WRONG" line appears appears.
  if grep -qE '^ *index: +[0-9]+ .*WRONG' "$f"; then i="wrong"; else i="correct"; fi
  # Readback: count the FAILing regimes out of the 3.
  rb="$(grep -cE '^ *readback .*FAIL' "$f")"
  # rc: the probe's own last-line verdict, not this script's opinion.
  if grep -q '^\[dpas-fold-probe\] PASS' "$f"; then rc=0; else rc=1; fi
  # Probe aggregate: the single combined "N violations" figure on the last line.
  agg="$(grep -oE '[0-9]+ violations' "$f" | tail -1 | cut -d' ' -f1)"
  printf '%-22s %-4s %-10s %-10s %-12s %s\n' "$s" "$rc" "$c" "$i" "${rb}/3 regimes" "${agg:-n/a}"
done

# --- PROVENANCE (audit #14) -------------------------------------------------
# These logs are evidence about a TREE, and nothing in a bare .log records which
# one. That is family 3 (stale artifact) wearing a new hat: the matrix was
# stamped at a806ef3, and 4b31d25 -- a comment-only change to the very header
# five of the six mutations target -- landed afterwards. The logs were still
# valid (only comments changed, and --anchors-only re-proves the anchors still
# resolve), but nothing in this output said so; it printed "7/7 as
# characterized" as if the logs were current. A reader could not tell.
#
# So compare the stamp's sha with HEAD and print the delta, and refuse the
# unqualified verdict when they differ. Re-running the matrix is what clears it.
STAMP="$LOG_DIR/last_pass.stamp"
STALE=0
s_sha="<no stamp>"
if [ -f "$STAMP" ]; then
  s_sha="$(sed -nE 's/^sha=([0-9a-f]+)$/\1/p' "$STAMP" | head -1)"
  s_date="$(sed -nE 's/^date=(.*)$/\1/p' "$STAMP" | head -1)"
  h_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo
  echo "provenance: matrix stamped $s_sha on $s_date; HEAD is $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
  if [ "$s_sha" = "$h_sha" ]; then
    echo "            logs match HEAD -- the table above is evidence for THIS code."
  else
    n_commits="$(git -C "$ROOT" rev-list --count "$s_sha..$h_sha" 2>/dev/null || echo '?')"
    touched="$(git -C "$ROOT" diff --name-only "$s_sha..$h_sha" -- \
      include/intfer/gemm/quant_gemm.hpp src/gemm/quant_gemm.cpp 2>/dev/null | tr '\n' ' ')"
    STALE=1
    echo "            STALE by $n_commits commit(s). Cells are evidence for $s_sha, NOT for HEAD."
    if [ -n "$touched" ]; then
      echo "            MUTATED FILES CHANGED SINCE: $touched"
      echo "            Re-run scripts/mutate_fold_matrix.sh before quoting these numbers."
    else
      echo "            The mutated files are unchanged since the stamp, so the states still"
      echo "            describe the same code -- but that is an argument, not the proof a"
      echo "            re-run gives. Confirm with mutate_fold_matrix.sh --anchors-only."
    fi
  fi
else
  STALE=1
  echo
  echo "provenance: NO STAMP at $STAMP -- these logs have no recorded tree."
fi

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
    if [ "$STALE" = "1" ]; then
      echo "7/7 as characterized AS OF $s_sha -- STALE w.r.t. HEAD, do not quote as current" >&2
      # rc=2, not 0 and not 1: the states DID behave as characterized at the
      # stamped tree, so this is not a regression (1), and the cells must not be
      # lifted into a report as current (0). Same rc-2-means-refuse-to-certify
      # convention the box-state skips use.
      echo "  Nothing is wrong with the code; the EVIDENCE is about an older tree." >&2
      exit 2
    else
      echo "7/7 as characterized (summary.txt: $n_ok AS EXPECTED, $n_bad UNEXPECTED)"
    fi
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

PROVENANCE: these numbers are evidence for ONE tree (see the provenance block).
The verdict line says so when the stamp no longer matches HEAD. A stale stamp
does not make the numbers wrong, it makes them not-about-this-code -- which is
the difference between a citation and a guess.
EOF
