#!/usr/bin/env bash
# Mutation matrix for scripts/check_pipefail_grep_q.sh.
#
# WHAT IS MUTATED, AND WHY IT IS THE CORPUS AND NOT THE CHECKER (audit #21).
# The rule and the subject are both shell scripts, and the checker is correct by
# construction on a clean tree. So a state that WEAKENS THE CHECKER is a no-op
# with extra steps. The observable difference has to come from damaging an
# ARTIFACT: here, a script in the corpus that pipes into a quiet grep under
# pipefail, or one that defeats the rule by another route. The corpus is handed
# over through INTFER_SCRIPTS_DIR, so the real scripts/ is never touched.
#
# States:
#   P-A  a compliant corpus                                    -> must PASS
#   P-B  a corpus script pipes into `grep -q` under pipefail   -> must REJECT
#   P-C  ... with the bundled spelling `grep -qE`              -> must REJECT
#        (catches a rule that only knows the bare `-q`)
#   P-D  ... with the long spelling `grep --quiet`             -> must REJECT
#   P-E  an EMPTY CORPUS (no pipefail anywhere)                -> must REFUSE
#   P-F  THE CONVERSE: non-quiet grep on a pipe under pipefail -> must PASS
#   P-G  THE CONVERSE: file-backed `grep -q` under pipefail    -> must PASS
#   P-H  pipefail with no grep at all                          -> must PASS
#
# Exit: 0 = every state behaved as specified, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK="$SCRIPT_DIR/check_pipefail_grep_q.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/pipefail_grep_q_mutate"
LOG="$BUILD_DIR/pipefail_grep_q_mutate"
SB="$(mktemp -d "${TMPDIR:-/dev/shm}/pipefail_mut.XXXXXX")"
mkdir -p "$STAMP" "$LOG"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

. "$SCRIPT_DIR/_gate_pred.sh"
. "$SCRIPT_DIR/_gate_enum.sh"

# A corpus that looks like the real one: pipefail scripts that use the SANCTIONED
# forms (file-backed grep -q, non-quiet grep on a pipe, no grep at all).
mk_corpus() {
  rm -rf "$SB"; mkdir -p "$SB"
  cat >"$SB/zz_filebacked.sh" <<'EOF'
set -o pipefail
code="$(printf 'needle\nhaystack\n')"
grep -q needle <<<"$code" && echo found
EOF
  cat >"$SB/zz_nonquiet.sh" <<'EOF'
set -o pipefail
printf 'a\nb\n' | grep a
EOF
  cat >"$SB/zz_nogrep.sh" <<'EOF'
set -o pipefail
echo hello
EOF
}
run_check() { INTFER_SCRIPTS_DIR="$SB" bash "$CHECK" 2>&1; }

echo "== PIPEFAIL GREP-Q MUTATION MATRIX (corpus is handed over via INTFER_SCRIPTS_DIR) =="

echo "-- baseline: a compliant corpus must PASS --"
mk_corpus
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL -- the checker does not pass on a compliant corpus:"
  printf '%s\n' "$b" | tail -8; exit 1
fi
if ! grep -q 'zz_filebacked.sh' <<<"$b"; then
  echo "BASELINE FAIL -- the corpus was not even examined (an empty harvest)" >&2
  exit 1
fi
echo "  ok    baseline passes and the corpus was examined"

caught=0; total=0
SUMMARY="$LOG/summary.txt"; : >"$SUMMARY"

run_reject() {
  local id="$1" why="$2"
  total=$((total + 1))
  mk_corpus
  if ! MUTATE_FN; then
    echo "  $id MISSED -- the mutation did not apply (STATE fault, not a mechanism)"
    echo "$id MISSED (mutation fn)" >>"$SUMMARY"; mk_corpus; return 1
  fi
  local o r
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" >"$LOG/$id.log"
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still exited 0 with the defect present ($why)"
    echo "$id MISSED (rc=0)" >>"$SUMMARY"; mk_corpus; return 1
  fi
  if ! pred_expect_present "$id rejection" "$o" 'PIPEFAIL GREP-Q: FAIL'; then
    echo "$id MISSED (rejection not reported)" >>"$SUMMARY"; mk_corpus; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-present: $why"
  echo "$id caught" >>"$SUMMARY"; caught=$((caught + 1)); return 0
}

mk_bad() { printf '%s\n' "$1" >"$SB/zz_bad.sh"; }

P_B='set -o pipefail
printf "%s" x | grep -q needle'
P_C='set -o pipefail
printf "%s" x | grep -qE needle'
P_D='set -o pipefail
printf "%s" x | grep --quiet needle'

MUTATE_FN() { mk_bad "$P_B"; }
run_reject "P-B" "pipe into bare grep -q under pipefail" MUTATE_FN

MUTATE_FN() { mk_bad "$P_C"; }
run_reject "P-C" "pipe into grep -qE (bundled flag) under pipefail" MUTATE_FN

MUTATE_FN() { mk_bad "$P_D"; }
run_reject "P-D" "pipe into grep --quiet (long form) under pipefail" MUTATE_FN

# P-E: an EMPTY CENSUS must be refused, not passed. "No script pipes into a
# quiet grep" and "no script sets pipefail at all" are different claims.
MUTATE_FN() { rm -rf "$SB"; mkdir -p "$SB"; printf 'echo hello\n' >"$SB/zz_nothing.sh"; }
MUTATE_FN || { echo "  P-E MISSED -- mutation fn failed"; exit 1; }
total=$((total + 1))
o="$(run_check)"; r=$?
printf '%s\n' "$o" >"$LOG/P-E.log"
if [ "$r" -ne 0 ] && grep -q 'never harvested' <<<"$o"; then
  echo "  P-E caught -- rc=$r, a corpus with no pipefail script is REFUSED, not passed"
  echo "P-E caught" >>"$SUMMARY"; caught=$((caught + 1))
else
  echo "  P-E MISSED -- rc=$r; a corpus with no pipefail script was not refused"
  echo "P-E MISSED" >>"$SUMMARY"
fi
mk_corpus

# P-F/P-G/P-H: THE CONVERSES. A non-quiet grep on a pipe, a file-backed grep -q
# and a pipefail script with no grep must all stay clean. Without these states a
# rule that flagged EVERY grep under pipefail would pass P-B..P-E, and the matrix
# would be proving that a check cannot fail -- the whole defect class this repo
# keeps auditing for.
total=$((total + 1))
o="$(run_check)"; r=$?
printf '%s\n' "$o" >"$LOG/P-F.log"
if [ "$r" -eq 0 ] && grep -q 'zz_nonquiet.sh' <<<"$o" \
   && grep -q 'zz_filebacked.sh' <<<"$o" && grep -q 'zz_nogrep.sh' <<<"$o" \
   && ! grep -q '^  FAIL' <<<"$o"; then
  echo "  P-F caught -- rc=0, non-quiet/file-backed/no-grep all ACCEPTED (converse holds)"
  echo "P-F caught" >>"$SUMMARY"; caught=$((caught + 1))
else
  echo "  P-F MISSED -- rc=$r; a sanctioned grep form was not accepted cleanly"
  echo "P-F MISSED" >>"$SUMMARY"
fi

echo
echo "== summary =="
sed 's/^/  /' "$SUMMARY"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "PIPEFAIL GREP-Q MUTATE: PASS ($caught/$total states behaved as specified)"
  { echo "stamp=pipefail_grep_q_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"
    echo "converse=P-F non-quiet + file-backed + no-grep all accepted under pipefail"; } >"$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "PIPEFAIL GREP-Q MUTATE: FAIL ($caught/$total)"
exit 1
