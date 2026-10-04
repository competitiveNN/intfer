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
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"

CHECK="$SCRIPT_DIR/check_doc_counts.sh"
REPORT="reports/archive/2026-10-01-batch-e1-e2-e3-verdicts.md"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/doc_counts_mutate"
LOG="$STAMP/states"
TARGET="$SCRIPT_DIR/check_doc_counts.sh"

[ -x "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
[ -f "$ROOT/$REPORT" ] || { echo "SKIP: $REPORT missing"; exit 2; }
mkdir -p "$LOG"; : > "$LOG/summary.txt"
git -C "$ROOT" diff --quiet -- "$TARGET" 2>/dev/null || {
  echo "FAIL: $TARGET is dirty; this matrix edits it."; exit 2; }

CB="$(mktemp "${TMPDIR:-/tmp}/cdc.XXXXXX")"
cp "$TARGET" "$CB"
trap 'cp "$CB" "$TARGET"; rm -rf "$WORK" "$CB"' EXIT
trap 'cp "$CB" "$TARGET"; rm -rf "$WORK" "$CB"; exit 130' INT
trap 'cp "$CB" "$TARGET"; rm -rf "$WORK" "$CB"; exit 143' TERM
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
  # POLARITY NAMED AT THE CALL SITE (AUDIT #20 pattern): a bare grep records
  # nothing about which side the marker belongs on.
  if ! pred_expect_absent "$id verdict" "$o" "$expect"; then
    echo "  $id MISSED -- polarity or marker survived: '$expect'"
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

# The four states below POISON THE REPORT and NEUTER THE CHECK TOGETHER.
#
# My first version mutated only the CHECKER's printed message and required
# rc!=0 -- and 4 of 6 states scored MISSED, correctly. Removing a message
# changes nothing about behaviour: on a CLEAN tree a checker with a silenced
# message SHOULD still exit 0. So those states proved only that the message
# could be edited.
#
# The fix is to give the checker a real defect to miss. Each state writes the
# defect into the REPORT (the poison) and disables the check that would have
# caught it. If the checker still reports ok/rc=0 with the defect PRESENT, the
# mechanism is provably not load-bearing -- which is the only way to test a
# check whose healthy path is indistinguishable from a broken one by exit code
# alone. Compare M-A/M-E, which mutate the COMPARISON rather than a message and
# so are visible on a clean tree.
REPORT_PATH="$ROOT/$REPORT"; export REPORT_PATH
POISON_BAK="$(mktemp "${TMPDIR:-/tmp}/rep.XXXXXX")"
cp "$ROOT/$REPORT" "$POISON_BAK"
poison_restore() { cp "$POISON_BAK" "$ROOT/$REPORT"; }
# Named poisons, each a small python script written to $WORK. NOT inline shell:
# three earlier versions used sed expressions, and every one of them failed for
# a DIFFERENT quoting reason -- a table row's literal `|` colliding with the
# s|...| delimiter, ``**N**`` needing BRE escaping, and backticks in the anchor
# being evaluated by the shell. Two states silently no-op'd, which is the AUDIT
# #17 anchor lesson: an anchor that does not resolve proves nothing and must be
# verified, not assumed.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/intfer_dcm.XXXXXX")"
export REPORT_PATH="$ROOT/$REPORT"

cat > "$WORK/poison_fabricate_sha.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "| `" + "f047c42" + "` |"
poison = "| `" + "deadbeef1" + "` | fabricated |\n"
if anchor not in s:
    sys.exit(3)          # anchor vanished -> refuse, do not silently no-op
open(p, "w").write(s.replace(anchor, poison + anchor, 1))
PYEOF

cat > "$WORK/poison_break_count_sentence.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.find("agree at **")
if i < 0:
    sys.exit(3)
j = s.index("\n", i)
open(p, "w").write(s[:i] + "agree at some number" + s[j:])
PYEOF

cat > "$WORK/poison_empty_index.py" <<'PYEOF'
import sys, re
p = sys.argv[1]
s = open(p).read()
# drop every commit-index row, leaving the index empty
out = re.sub(r"^\| `[0-9a-f]{7,40}` \|.*$", "| (removed) |", s, flags=re.M)
if out == s:
    sys.exit(3)
open(p, "w").write(out)
PYEOF

cat > "$WORK/poison_drop_unverified.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
if "remains UNVERIFIED" not in s:
    sys.exit(3)
open(p, "w").write(s.replace("remains UNVERIFIED", "is fine", 1))
PYEOF

# run_poisoned <id> <poison-command> <check-sed|empty> <absent-re> <why>
run_poisoned() {
  local id="$1" psed="$2" csed="$3" expect="$4" why="$5"
  total=$((total+1))
  poison_restore; restore
  # The poison is a SHELL COMMAND, not a sed expression. Two states silently
  # no-op'd as sed because BRE-escaping markdown (`` **N** ``, a literal `|`
  # inside a table row) is its own source of bugs -- the state reported MISSED
  # for the wrong reason twice before this. python3 does a literal replace.
  python3 "$psed" "$ROOT/$REPORT" || { echo "  $id MISSED -- poison script failed"
    echo "$id MISSED (poison error)" >> "$LOG/summary.txt"; poison_restore; return 1; }
  # "-" means POISON-ONLY: nothing in the checker is touched on purpose. The
  # still-applied guard below is skipped for those states, because there is no
  # checker edit to verify.
  if [ "$csed" != "-" ]; then
    sed -i "$csed" "$TARGET"
    if cmp -s "$CB" "$TARGET"; then
      echo "  $id MISSED -- check mutation did not apply"
      echo "$id MISSED (no apply)" >> "$LOG/summary.txt"; poison_restore; return 1
    fi
  fi
  if cmp -s "$POISON_BAK" "$ROOT/$REPORT"; then
    echo "  $id MISSED -- POISON did not apply (state proves nothing)"
    echo "$id MISSED (poison no-op)" >> "$LOG/summary.txt"; poison_restore; return 1
  fi
  local o r
  o="$("$CHECK" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  poison_restore
  # A MISS is: the checker exited 0 (or printed ok) while the defect is present.
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker exited 0 although $why"
    echo "$id MISSED (blind to the defect)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_absent "$id poisoned-verdict ($why)" "$o" "$expect"; then
    echo "  $id MISSED -- polarity or marker survived: '$expect' although $why"
    echo "$id MISSED (marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r and the ok-marker is gone, though $why"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

# M-B: sha-history check neutered + a fabricated sha added to the index.
#      Only this pairing can see the defect: with no fabricated sha the check
#      and the report both look healthy.
# The anchor is a line that actually exists in the REPORT (my first two
# attempts used strings that live in the CHECKER, so the poison never applied
# and the state scored MISSED for the wrong reason -- a state that mutates
# nothing proves nothing, which is why run_poisoned verifies the poison landed).
# M-B: POISON ONLY. The sha-history check is load-bearing iff a fabricated sha
#      makes the checker FAIL. No checker mutation needed -- and a checker that
#      PASSES with a fabricated sha present is provably blind to it.
run_poisoned "M-B" "$WORK/poison_fabricate_sha.py" \
  '-' \
  'every cited sha resolves' \
  "a fabricated sha is in the index"
#      that turns "no rows" from silent into a failure.
run_poisoned "M-C" "$WORK/poison_empty_index.py" \
  's|^enum_require_nonempty "commit-index rows".*|true|' \
  'harvested 0 items' \
  "the index has no rows at all"
#      pattern matches nothing. This is the exact defect the matrix was born
#      from: WARN + skip + exit 0 with the subject absent.
# Anchor on a whole line that EXISTS in the report (the count sentence wraps
# across two lines, so a bare `agree at **N**` pattern is not a line anchor).
run_poisoned "M-D" "$WORK/poison_break_count_sentence.py" \
  's|is ABSENT, so a PASS would be vacuous|is absent, continuing anyway|' \
  'ABSENT, so a PASS would be vacuous' \
  "the count sentence is absent"
run_poisoned "M-F" "$WORK/poison_drop_unverified.py" \
  's|echo "   ok    TP=2 still marked UNVERIFIED"|echo "   ok    TP=2 fine"|' \
  'TP=2 still marked UNVERIFIED' \
  "the TP=2 UNVERIFIED wording is gone"
rm -f "$POISON_BAK"

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