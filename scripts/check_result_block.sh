#!/usr/bin/env bash
# Assert the RESULT-filling benchmark block in the reports is still RUNNABLE and
# still refuses to fabricate wall-time.
#
# WHY. The block (reports/2026-09-30-pending-bench-backlog.md, "ONE COMMAND FILLS
# EVERY `RESULT:` LINE ABOVE") was written by hand and verified by hand once.
# Nothing re-verified it afterwards, and the properties that matter are exactly
# the ones a later edit silently breaks:
#   * a flag renamed in source, leaving the block setting an env var nobody reads
#     -- the run then "succeeds" while measuring the BASE arm three times;
#   * a numeric tok/s pasted in, which is the retracted-wall-time class;
#   * a cross-reference or a RESULT line orphaned so part of the batch is
#     unreachable from the block.
# Those are cheap to check and impossible to notice by reading.
#
# Uses the repo's existing guards rather than new copies:
#   _gate_enum.sh  enum_require_nonempty / enum_require_min  (AUDIT #20)
#   _gate_pred.sh  pred_expect_absent / pred_expect_present  (polarity named)
#
# Usage: scripts/check_result_block.sh
# Exit: 0 = all properties hold, 1 = a property drifted, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
enum_reset

BACKLOG="reports/2026-09-30-pending-bench-backlog.md"
FOLD="reports/2026-09-30-dpas48-fold.md"
SRC_DIRS="src examples/intfer_cli scripts"

echo "== RESULT-BLOCK CHECK =="

extract_block() { # the fenced bash block under the ONE COMMAND heading
  awk '/^## ONE COMMAND FILLS/{f=1} f&&/^```bash$/{p=1;next} p&&/^```$/{exit} p' "$BACKLOG"
}
BLOCK="$(extract_block)"

# Anti-vacuity: an empty harvest cannot certify anything (AUDIT #20).
enum_require_nonempty "result-block" "$(printf '%s\n' "$BLOCK" | grep -c . || true)" \
  "lines in the RESULT-filling block" || exit 1
if [ -z "$BLOCK" ]; then echo "  FAIL  the block did not extract"; exit 1; fi

bad=0

# ---- (a) every flag the block sets must exist in source --------------------
# Only EXECUTED settings: `export FOO=` / `FOO=` prefixes, not every token that
# merely looks like one. A comment or a prose mention is not a setting.
flags="$(printf '%s\n' "$BLOCK" \
  | grep -oE '(^|[[:space:]])(export[[:space:]]+)?INTFER_[A-Z0-9_]+=' \
  | grep -oE 'INTFER_[A-Z0-9_]+' | LC_ALL=C sort -u)"
enum_require_nonempty "block-flags" "$(printf '%s\n' "$flags" | grep -c . || true)" \
  "INTFER_* flags set by the block" || bad=$((bad+1))
printf '%-4s %-28s %s\n' "" "flag" "found in source"
for f in $flags; do
  if grep -rqF -- "$f" $SRC_DIRS 2>/dev/null; then
    printf '  %-4s %-28s %s\n' "ok" "$f" "yes"
  else
    printf '  %-4s %-28s %s\n' "FAIL" "$f" "NOT FOUND -- the block would set a var nobody reads"
    bad=$((bad+1))
  fi
done

# ---- (b) no numeric tok/s inside the block --------------------------------
# The block is instructions; a measured tok/s in it is a claim with no run
# behind it. Polarity named at the call site.
toks="$(printf '%s\n' "$BLOCK" | grep -cE '[0-9]+(\.[0-9]+)?[[:space:]]*tok/s' || true)"
if pred_expect_absent "no-tok-s-in-block" "$toks" '^[1-9]'; then
  :
else
  echo "  note  $toks line(s) in the block match a tok/s pattern"
fi
if printf '%s\n' "$BLOCK" | grep -qE '[0-9]+(\.[0-9]+)?[[:space:]]*tok/s'; then
  echo "  FAIL  a numeric tok/s appears inside the block -- that is a result, not"
  echo "        an instruction, and it would be an unmeasured claim."
  bad=$((bad+1))
else
  echo "  ok    no numeric tok/s inside the block"
fi

# ---- (c) every blank RESULT line is anchored to the block ------------------
n_b="$(grep -cF 'RESULT (27B TP=2' "$BACKLOG" 2>/dev/null || true)"
n_f="$(grep -cF 'RESULT (27B TP=2' "$FOLD" 2>/dev/null || true)"
n_res=$(( n_b + n_f ))
enum_require_min "result-lines" "$n_res" 6 || bad=$((bad+1))
echo "  ok    $n_res RESULT lines across both reports (expect 6)"

# ---- (d) the fold report points at the block -------------------------------
if pred_expect_present "fold-crossref" "$(cat "$FOLD")" 'ONE COMMAND FILLS'; then
  :
else
  echo "  FAIL  $FOLD does not point at the block, so its blank RESULT lines are"
  echo "        not reachable from it."
  bad=$((bad+1))
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "RESULT-BLOCK CHECK: PASS ($n_res RESULT lines reachable, flags resolve, no tok/s)"
  echo "RESULT: the benchmark block still names only real flags and claims no tok/s"
  exit 0
fi
echo "RESULT-BLOCK CHECK: FAIL ($bad property/properties drifted)"
echo "RESULT: the RESULT-filling block has drifted from what the source supports"
exit 1