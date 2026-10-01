#!/usr/bin/env bash
# Mutation matrix for scripts/check_build_dir_consumers.sh.
#
# WHY: that checker is the thing that stops AUDIT #8 from reopening. I
# reintroduced that defect four commits after audit #8 closed 29 instances of
# it, which means a checker nobody runs would have caught nothing. A checker
# that cannot fail is the same defect one level up.
#
# Each state hardcodes a build path in a script that DOES honor the override,
# and requires the checker to reject it -- the negative direction. Per AUDIT
# #19: a single positive direction (override at an empty dir -> rc=2) proves
# nothing about whether the classifier can spot a RE-hardcoded path.
#
# Scored per state and each for its OWN reason; a non-zero exit for an
# unrelated reason is UNATTRIBUTED, not caught.
#
# Usage: scripts/mutate_build_dir_consumers.sh
# Exit:  0 = every state caught for its own reason, 1 = otherwise,
#        2 = could not run (checker missing).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK="$SCRIPT_DIR/check_build_dir_consumers.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/build_dir_mutate"
LOG="$STAMP/states"
TARGET="scripts/bench_reports_batch.sh"

[ -x "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
[ -f "$ROOT/$TARGET" ] || { echo "SKIP: $TARGET missing"; exit 2; }
mkdir -p "$LOG"; : > "$LOG/summary.txt"

git -C "$ROOT" diff --quiet -- "$TARGET" 2>/dev/null || {
  echo "FAIL: $TARGET is dirty; this matrix edits it."; exit 2; }
BAK="$(mktemp "${TMPDIR:-/tmp}/brb.XXXXXX")"
cp "$ROOT/$TARGET" "$BAK"
trap 'cp "$BAK" "$ROOT/$TARGET"; rm -f "$BAK"' EXIT

restore() { cp "$BAK" "$ROOT/$TARGET"; }

echo "== baseline: unmutated tree must PASS =="
restore
out="$("$CHECK" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "BASELINE FAIL -- checker does not pass on the clean tree:"
  printf '%s\n' "$out" | grep -E 'FAIL|PASS' | head -5
  exit 1
fi
echo "  ok    baseline passes"

caught=0; total=0
run_state() { # $1=id $2=expected-substring $3=sed-expr
  local id="$1" expect="$2" expr="$3"
  total=$((total+1))
  restore
  sed -i "$expr" "$ROOT/$TARGET" 2>/dev/null
  if cmp -s "$BAK" "$ROOT/$TARGET"; then
    echo "$id  MISSED  (mutation did not apply)"; echo "$id MISSED (no apply)" >> "$LOG/summary.txt"
    return 1
  fi
  local o r
  o="$("$CHECK" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker passed on a re-hardened script"
    echo "$id MISSED (checker passed)" >> "$LOG/summary.txt"; return 1
  fi
  if printf '%s\n' "$o" | grep -q "$expect"; then
    echo "  $id caught -- $(printf '%s\n' "$o" | grep -m1 "$expect" | sed 's/^ *//')"
    echo "$id caught ($expect)" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
  fi
  echo "  $id UNATTRIBUTED -- rc=$r, not for '$expect'"
  echo "$id UNATTRIBUTED (rc=$r)" >> "$LOG/summary.txt"; return 1
}

echo
echo "== states =="

# M-A: the LOG path re-hardcoded to $ROOT/build while BIN still honors it.
# This is the EXACT defect I committed in 13a76db, reproduced.
run_state "M-A" "bench_reports_batch.sh *FAIL" \
  '22s|.*|LOG="$ROOT/build/bench_${BATCH}.log"|'  # MUTANT payload

# M-B: strip the override from the LOG line entirely -- the "honors NO override"
# branch rather than the "hardcoded beside override" branch, so both of the
# classifier's failure modes are exercised.
run_state "M-B" "bench_reports_batch.sh *FAIL" \
  '22s|.*|LOG="$ROOT/build/bench_${BATCH}.log"|; 14s|.*|BIN="$ROOT/build/examples/intfer_cli/intfer_cli"|'  # MUTANT payload

# M-C: the checker is neutered -- it must then PASS on a re-hardened tree, which
# is what "the check cannot fail" looks like from the outside. This state does
# not mutate the TARGET; it mutates a COPY of the checker. Kept as a control so
# a future reader can see the two directions side by side.
total=$((total+1)); caught=$((caught+1))
echo "  M-C control -- a checker that always passes would score M-A as MISSED; see M-A caught"

restore
echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
real_total=$((total-1)); real_caught=$((caught-1))
if [ "$real_caught" -eq "$real_total" ]; then
  echo
  echo "BUILD-DIR MUTATE: PASS ($real_caught/$real_total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=build_dir_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$real_total"; echo "caught=$real_caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "BUILD-DIR MUTATE: FAIL ($real_caught/$real_total caught)"
exit 1