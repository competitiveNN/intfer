#!/usr/bin/env bash
# Run the CORRECTNESS GATES (scripts/test_*.sh) and certify the engine against the
# current tree, leaving an on-disk artifact that names HEAD.
#
# WHY THIS EXISTS. `run_all_checks.sh` certifies the TOOLING. It does not run the
# engine's correctness gates, and a session once reported "10 PASS / 0 PARTIAL /
# 0 SKIPPED" backed only by a DIFF ARGUMENT -- "the files that changed since the
# gates ran are not engine inputs". That argument is sound and it is still not
# evidence: it is the same shape AUDIT #22 stopped accepting when a tok/s was
# published with no artifact behind it. A number at the end of a session needs a
# file on disk that says which tree produced it. This is that file.
#
# *** WHY THE ARTIFACT IS NOT NAMED last_pass.stamp ***
# Deliberate, and the reason is worth keeping. `run_all_checks.sh` enumerates
# `find build -name '*last_pass.stamp'` and fails on any directory it does not
# recognise; scripts/_gate_pop.sh derives that recognised set from `mutate_*.sh`
# alone. A stamp from a suite that is not a mutation matrix would therefore be
# reported as a LEAK, and the only ways out would be to loosen a green guard or to
# reclassify this as a matrix. Neither is right: the tooling stack and the engine
# certification are DIFFERENT evidence classes and conflating them is what made
# the original gap hard to see. So this writes gates.log + HEAD beside the
# results, outside the stamp population, and says so at every use site.
#
# Usage: scripts/run_ten_gates.sh
# Exit:  0 = every gate rc=0; 1 = a real failure; 2 = setup.
#
# Cost: this runs the engine. It takes tens of minutes and needs both GPUs. It is
# NOT a step of run_all_checks.sh for exactly that reason -- it is the slow
# half of certification, run separately and re-run when the engine or a gate
# changes, not on every tooling edit.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
enum_reset

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
OUT="$BUILD_DIR/ten_gates"
LOG="$OUT/gates.log"
mkdir -p "$OUT" || { echo "FAIL: cannot create $OUT" >&2; exit 2; }

# The population is DERIVED, never listed: the gates are exactly the scripts named
# test_*.sh, which is also the definition gate_inventory.sh classifies. A
# hand-written list of ten names would be a claim that rots the moment a gate is
# added or renamed -- and the tally printed below is the thing people quote.
GATES=""
for g in "$SCRIPT_DIR"/test_*.sh; do [ -f "$g" ] && GATES="$GATES $g"; done
n_gates="$(printf '%s\n' $GATES | grep -c .)"
enum_require_min "gate-population" "$n_gates" 5 || {
  echo "RESULT: only $n_gates gate(s) enumerated -- census too small to certify" >&2
  exit 2; }

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "== TEN CORRECTNESS GATES =="
echo "HEAD=$HEAD_SHA  gates=$n_gates  out=$OUT"
echo "log: $LOG"

{
  echo "=== ten correctness gates $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "HEAD=$HEAD_SHA"
  echo "gate population (derived from scripts/test_*.sh): $n_gates"
} > "$LOG"

pass=0; partial=0; skip=0; fail=0; failed=""
for g in $GATES; do
  name="$(basename "$g" .sh)"; name="${name#test_}"
  printf '== %s ==\n' "$name" >> "$LOG"
  timeout "${INTFER_GATE_TIMEOUT:-2400}" bash "$g" >> "$LOG" 2>&1; rc=$?
  printf 'GATE_RESULT %s rc=%s\n' "$name" "$rc" >> "$LOG"
  case "$rc" in
    0)  pass=$((pass+1));    echo "  ok    $name" ;;
    2)  partial=$((partial+1)); echo "  PART  $name (rc=2: a coverage precondition was unmet)" ;;
    124) skip=$((skip+1));  failed="$failed $name(TIMEOUT)"
         echo "  SKIP  $name (TIMEOUT -- not scored as a pass)" ;;
    *)  fail=$((fail+1));   failed="$failed $name"
         echo "  FAIL  $name (rc=$rc)" ;;
  esac
done

# Tally by EXIT CODE, the only non-double-countable measure (the AUDIT #13
# counting trap: counting a gate's green inner sections double-counts it).
printf 'TALLY pass=%s partial=%s skip=%s fail=%s\n' "$pass" "$partial" "$skip" "$fail" >> "$LOG"
{
  echo "HEAD=$HEAD_SHA"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gates=$n_gates"
  echo "pass=$pass partial=$partial skip=$skip fail=$fail"
  echo "failed=${failed:-none}"
  echo "log=$LOG"
} > "$OUT/HEAD"

echo
echo "TALLY pass=$pass partial=$partial skip=$skip fail=$fail"
echo "artifact: $OUT/HEAD   (full output: $LOG)"
if [ "$fail" -ne 0 ]; then
  echo "RESULT: TEN GATES FAIL at $HEAD_SHA -- $failed"
  exit 1
fi
echo "RESULT: $n_gates correctness gate(s) certified at $HEAD_SHA ($pass PASS, $partial PARTIAL, $skip SKIPPED)"
exit 0