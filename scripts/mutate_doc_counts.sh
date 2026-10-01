#!/usr/bin/env bash
# Mutation matrix for scripts/check_doc_counts.sh.
#
# WHY: it is the newest checker and, before this matrix, the only one shipped
# without a matrix -- a checker whose ability to fail is asserted in a commit
# message is an assertion, not evidence (the audit #7/#8 shape).
#
# Each state breaks ONE mechanism and requires the checker to name that
# mechanism. Judged by a POSITIVE defect predicate (a specific line is ABSENT or
# WRONG), not by the exit code, because three of these breaks still exit
# non-zero for unrelated reasons and a matrix watching only the exit code would
# score them as passes forever (AUDIT #18).
#
# Every state also runs a CONVERSE baseline first: the clean tree must PASS, so
# a harness that rejects everything cannot score as caught.
#
# Usage: scripts/mutate_doc_counts.sh
# Exit:  0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK="$SCRIPT_DIR/check_doc_counts.sh"
REPORT="reports/2026-10-01-batch-e1-e2-e3-verdicts.md"
STAMP="$ROOT/build/doc_counts_mutate"
LOG="$STAMP/states"
TARGET="$SCRIPT_DIR/check_doc_counts.sh"

[ -x "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
[ -f "$ROOT/$REPORT" ] || { echo "SKIP: $REPORT missing"; exit 2; }
mkdir -p "$LOG"; : > "$LOG/summary.txt"
git -C "$ROOT" diff --quiet -- "$TARGET" 2>/dev/null || {
  echo "FAIL: $TARGET is dirty; this matrix edits it."; exit 2; }

CB="$(mktemp "${TMPDIR:-/tmp}/cdc.XXXXXX")"
cp "$TARGET" "$CB"
trap 'cp "$CB" "$TARGET"; rm -f "$CB"' EXIT
restore() { cp "$CB" "$TARGET"; }

echo "== baseline: clean tree must PASS =="
restore
out="$("$CHECK" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "BASELINE FAIL -- checker does not pass on the clean tree:"
  printf '%s\n' "$out" | tail -4; exit 1
fi
# The baseline must also SHOW the markers the predicates key on, so a state
# cannot be scored by a run that never reached the mechanism.
if ! printf '%s\n' "$out" | grep -q 'stated count'; then
  echo "BASELINE FAIL -- no 'stated count' line, so no state can be scored"; exit 1
fi
if ! printf '%s\n' "$out" | grep -q 'every cited sha resolves'; then
  echo "BASELINE FAIL -- no sha-history line, so the sha states cannot be scored"; exit 1
fi
echo "  ok    baseline passes and shows both markers (stated count, sha history)"
echo "  ok    CONVERSE: the healthy direction is verified, so an always-red"
echo "        harness cannot pass"

caught=0; total=0
# run_state <id> <expect-ABSENT-or-WRONG re> <sed-expr> <explanation>
run_state() {
  local id="$1" expect="$2" expr="$3" why="$4"
  total=$((total+1))
  restore
  sed -i "$expr" "$TARGET" 2>/dev/null
  if cmp -s "$CB" "$TARGET"; then
    echo "  $id MISSED -- mutation did not apply (state proves nothing)"
    echo "$id MISSED (no apply)" >> "$LOG/summary.txt"; return 1
  fi
  local o r
  o="$("$CHECK" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  # Positive predicate: the specific evidence must be GONE or WRONG.
  if printf '%s\n' "$o" | grep -qE "$expect"; then
    echo "  $id MISSED -- checker still produced '$expect'"
    echo "$id MISSED (marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker exited 0 despite $why"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- marker gone AND rc=$r ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states =="

# M-A: the count/rows comparison is neutered -> the primary assertion vanishes.
run_state "M-A" 'stated count .* == rows in the index' \
  's|\[ "\$claimed" = "\$idx_rows" \]|false|' \
  "the number no longer has to match the rows it enumerates"

# M-B: the sha-history check is neutered. Note the exit code is UNCHANGED here
# on a clean tree -- only a fabricated sha can see this -- which is exactly why
# it needs a state rather than a passing run.
run_state "M-B" 'every cited sha resolves in git history' \
  's|^    echo "   ok    every cited sha resolves in git history"|    echo "   ok    sha check skipped"|' \
  "a fabricated sha would no longer be caught"

# M-C: the vacuous-harvest guard removed -> empty index could pass again.
run_state "M-C" 'harvested 0 items' \
  's|^enum_require_nonempty "commit-index rows".*|true|' \
  "an empty index harvest is accepted, so nothing is verified"

# M-D: the 'count sentence is REQUIRED' rule downgraded back to a warning. This
# is the state for the defect this matrix was born from: the checker printed
# WARN+skip and exited 0 when its subject sentence was absent.
run_state "M-D" 'ABSENT, so a PASS would be vacuous' \
  "s|is ABSENT, so a PASS would be vacuous|is absent, continuing anyway|" \
  "a missing count sentence is tolerated again"

# M-E: the tally guard removed.
run_state "M-E" 'tally still reads 7 PASS / 2 PARTIAL / 1 SKIPPED' \
  's|7 PASS, 2 PARTIAL, 1 SKIPPED|9 PASS, 1 PARTIAL, 0 SKIPPED|' \
  "an altered gate tally would be reported as intact"

# M-F: the TP=2 UNVERIFIED guard removed -- the wording this repo must not lose.
run_state "M-F" 'TP=2 still marked UNVERIFIED' \
  's|TP=2 still marked UNVERIFIED|TP=2 fine|' \
  "losing the TP=2 UNVERIFIED marker is no longer caught"

restore
echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "DOC COUNTS MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=doc_counts_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "DOC COUNTS MUTATE: FAIL ($caught/$total caught)"
exit 1