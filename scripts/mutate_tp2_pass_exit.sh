#!/usr/bin/env bash
# Mutation matrix for scripts/check_tp2_pass_exit.sh.
#
# WHY A MATRIX FOR A GATE'S VERDICT CONVERSION. check_tp2_pass_exit.sh reads
# scripts/test_tp2_27b.sh's SOURCE and asserts the gate converts its PASS
# variable into an exit code. The defect it pins was shipping in the working
# tree until this session: the script set PASS=1 at :715 and PASS=0 on every
# failure, then fell off the end at `rm -f "$LOG"` with exit 0, and
# run_ten_gates.sh classifies by exit code (rc 0 -> GPASS, :79) -- so the gate
# that certifies the 27B --tp 2 target reported PASS with EVERY arm failing.
#
# WHY SOURCE-BASED, NOT RUN-BASED. The gate needs both B580s and an 18 GiB
# model load, so the real arm cannot be driven from here (standing rule 2).
# And a synthetic-log version would build its fixture from the FIXED script
# and pass by construction -- the same mistake the first decode1 self-check
# made. So the matrix mutates the SCRIPT and grades the CHECKER against it,
# on temp copies: the tracked file is never modified.
#
# STATES. The polarity is the point and differs per state:
#   PE-1  the PASS -> exit conversion DELETED (the shipped defect)   [REFUSE]
#   PE-2  the conversion's FAIL branch changed to exit 0             [REFUSE]
#   PE-3  the conversion moved ABOVE the trailing rm -f              [REFUSE]
#   PE-4  PASS=1 deleted, conversion left in place (tests nothing)   [REFUSE]
#   PE-5  healthy baseline, unmutated                                 [BASELINE]
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="$SCRIPT_DIR/test_tp2_27b.sh"
CHECK="$SCRIPT_DIR/check_tp2_pass_exit.sh"
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/tp2_pass_exit_mutate"
STAMP="$LOG_DIR/last_pass.stamp"
BUILD="$LOG_DIR"

[ -f "$TARGET" ] || { echo "MUTATE: no target: $TARGET" >&2; exit 1; }
[ -f "$CHECK" ]  || { echo "MUTATE: no checker: $CHECK" >&2; exit 1; }
mkdir -p "$BUILD"
rm -f "$STAMP"

SB="$(mktemp -d)"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM
TARGET_DIGEST_BEFORE="$(cat "$TARGET" | cksum)"

total=0; caught=0; bad=0; baselines=0
declare -a RESULTS

. "$SCRIPT_DIR/_gate_patch.sh"

# --- pre-flight: every anchor must occur exactly once, BEFORE any state runs ---
preflight() {
  # `|` is the label/pattern separator, NOT a space: several of these patterns
  # contain no space at all (`^[[:space:]]*PASS=1[[:space:]]*$`), and splitting
  # on the first space would hand `pat` an empty string and report "0
  # occurrences" for every anchor -- a pre-flight that always fails. MEASURED
  # on the first run of this matrix.
  local anchors=(
    'PASS_VAR|^[[:space:]]*PASS=1[[:space:]]*$'
    'CONVERSION|if \[ "\$PASS" = 1 \]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
    'TRAILINGRM|rm -f "\$LOG"  # \*_OUT/\*_ERR are handled by the EXIT trap \(kept on FAIL\)'
  )
  local a c label pat
  for a in "${anchors[@]}"; do
    label="${a%%|*}"; pat="${a#*|}"
    c="$(grep -cE -- "$pat" "$TARGET" 2>/dev/null || true)"
    c="${c:-0}"
    if [ "$c" -ne 1 ]; then
      echo "ANCHOR STALE in test_tp2_27b.sh: $label matched $c occurrence(s), want 1." >&2
      return 1
    fi
  done
  echo "anchor pre-flight: ok (${#anchors[@]}/${#anchors[@]} targets located exactly once)"
}

anchor() { printf '%s' "$2" >"$SB/anchor_$1.txt"; }

# --- baseline: the UNMUTATED gate must PASS the checker, and say so ------------
# Without this the matrix would report "4/4 caught" over a checker whose
# baseline itself broke -- e.g. if the gate is later rewritten to a form the
# checker no longer recognises. Every state below is scored against this.
baseline() {
  local out rc
  out="$(bash "$CHECK" "$TARGET" 2>&1)"; rc=$?
  baselines=$((baselines+1))
  if [ "$rc" -ne 0 ]; then
    echo "BASELINE FAIL: unmutated gate exited rc=$rc -- every state below" >&2
    echo "  would be scored against a broken baseline." >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  case "$out" in
    *"TP2-PASS-EXIT PIN: PASS"*) ;;
    *) echo "BASELINE FAIL: the healthy gate did not print its PASS verdict" >&2; exit 1 ;;
  esac
  echo "  ok baseline rc=0, PASS verdict present, no FAIL line"
}

# run_state <id> <expected-regex> <mutation-fn>
# A state is CAUGHT when the mutated gate's checker exits NON-ZERO **and**
# prints the expected FAIL predicate marker. rc alone is not enough: a state
# can be refused for an unrelated reason, which is how a contaminated
# hand-run once scored a false catch (audit #12's self-contradicting report).
# The mutant runs on a TEMP COPY, so the tracked file is never left mutated.
run_state() {
  local id="$1" expect="$2" fn="$3" m out rc
  m="$SB/$id.sh"
  cp "$TARGET" "$m"
  "$fn" "$m" || { record "$id" MISSED "(mutation fn failed)"; return; }
  if ! bash -n "$m" 2>/dev/null; then
    record "$id" MISSED "(mutation broke syntax -- not evidence)"; return
  fi
  out="$(bash "$CHECK" "$m" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$BUILD/$id.log"
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE -- "$expect"; then
    caught=$((caught+1)); record "$id" CAUGHT ""
  elif [ "$rc" -eq 0 ]; then
    record "$id" MISSED "(checker still PASS -- unobservable)"
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    record "$id" MISSED "(caught for the WRONG reason; wanted /$expect/)"
  else
    record "$id" MISSED "(rc=$rc, marker absent)"
  fi
}

record() { total=$((total+1)); RESULTS+=("$1 $2 $3"); }

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  PE-1  PASS -> exit conversion deleted (shipped defect)         [REFUSE]"
  echo "  PE-2  conversion FAIL branch exits 0 (green verdict gate)       [REFUSE]"
  echo "  PE-3  conversion moved ABOVE the trailing rm -f                 [REFUSE]"
  echo "  PE-4  PASS=1 deleted (conversion tests a variable nobody sets) [REFUSE]"
  echo "  PE-5  split-form conversion with a green FAIL branch            [REFUSE]"
  echo "  PE-6  bare PASS echo preceded by exit 0                        [REFUSE]"
  echo "  PE-7  exit 0 as the last statement                             [REFUSE]"
  exit 0
fi

echo "== anchor pre-flight =="
anchor PASS_VAR   '^[[:space:]]*PASS=1[[:space:]]*$'
anchor CONVERSION 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
anchor TRAILINGRM 'rm -f "$LOG"  # *_OUT/*_ERR are handled by the EXIT trap (kept on FAIL)'
preflight || exit 1

echo "== baseline =="
baseline
echo "== states =="

# *** PE-1 IS THE SHIPPED DEFECT, AND IT IS THE INTERESTING ONE. ***
# It deletes everything from the trailing `rm -f "$LOG"` to the end of the
# file, leaving the rm as the last statement. That is byte-for-byte what was
# in the working tree this session: the script set PASS=1 at :715, set PASS=0
# on every failure, and fell off the end at the rm -- so run_ten_gates.sh's
# rc 0 -> GPASS classification counted the gate as a pass with every arm
# failing. The clone at intfer-int8-dpas-work/scripts/test_tp2_27b.sh:748
# carries the canonical form that line once had; this state proves the
# checker would have caught its absence.
m_pe1() { # $1 = mutated copy
  python3 - "$1" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
rm = 'rm -f "$LOG"  # *_OUT/*_ERR are handled by the EXIT trap (kept on FAIL)'
i = s.index(rm)
# Delete everything from the rm line to the end. The last statement is now
# the rm itself, so the shell returns rm's rc (0) and the gate reports PASS.
open(p, 'w').write(s[:i + len(rm)].rstrip('\n') + '\n')
PY
}

# PE-2: a GREEN VERDICT GATE. The conversion exists but its FAIL branch exits
# 0, so the gate still reports PASS with every arm failing. This is the
# other half of the defect: not "the conversion was lost" but "the conversion
# cannot fail" -- and it is caught by predicate (b), not (a), which is why
# both exist.
m_pe2() {
  python3 - "$1" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
new = 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; exit 0; else echo "TP2-27B TEST: FAIL" >&2; exit 0; fi #PE2 green verdict'
assert s.count(old) == 1, s.count(old)
open(p, 'w').write(s.replace(old, new))
PY
}

# PE-3: predicate (d)'s escape hatch. The conversion is moved ABOVE the
# trailing `rm -f`, so predicates (a)(b)(c) all still hold and only (d) --
# the last-statement check -- can see it. This is the shape of a "cleanup"
# that relocates the exit to the top of the file without noticing the rm
# after it: the shell falls off the end and returns 0 from rm.
m_pe3() {
  python3 - "$1" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
conv = 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
rm   = 'rm -f "$LOG"  # *_OUT/*_ERR are handled by the EXIT trap (kept on FAIL)'
assert s.count(conv) == 1
ci = s.index(conv)
ri = s.index(rm)
assert ri < ci
new = (s[:ri]
       + conv + '\n'
       + rm + '   #PE3 escape hatch: shell falls off the end after this rm\n'
       + s[ci + len(conv):])
open(p, 'w').write(new)
PY
}

# PE-4: the conversion tests a variable nobody writes. PASS=1 is deleted, so
# the conversion is decorative -- it cannot distinguish a passing gate from a
# failing one. Predicate (c) is the only one that can see it.
m_pe4() {
  python3 - "$1" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'PASS=1\n'
new = 'PASS=0   #PE4 conversion now tests a variable nobody sets\n'
assert s.count(old) == 1, s.count(old)
open(p, 'w').write(s.replace(old, new, 1))
PY
}

run_state PE-1 'FAIL \(a\)' m_pe1
run_state PE-2 'FAIL \(b\)' m_pe2
run_state PE-3 'FAIL \(d\)' m_pe3
run_state PE-4 'FAIL \(c\)' m_pe4

# *** PE-5/6/7: THE WIDENED ACCEPTANCE IS NOT A SILENT WIDENING. *** The sweep
# forced the checker to accept three forms it previously refused -- split
# conversions (`if [ "$PASS" != 1 ]; then` ... `exit 1; fi`), `exit "$RC"`
# relays, and bare `echo "... TEST: PASS"` preceded by a FAIL exit. Accepting
# them was necessary (they are correct gates) and widening a check to make it
# green is the defect this repo has a memory tag for, so each widening is
# pinned by a state that plants the DEFECTIVE version of the same form:
#   PE-5  split conversion whose FAIL branch exits 0  -> (b) must still fire
#   PE-6  bare PASS echo preceded by `exit 0`      -> (d) must still fire
#   PE-7  `exit 0` as the last statement            -> (d) must still fire
# A widening that is not pinned by its own negative direction is unproven.

# PE-5: the split form (which (a) now accepts) with a GREEN FAIL branch. The
# checker accepts the split form on (a) and (b) separately, so the question is
# whether (b) still sees the green branch when it is not on the same line as
# the test -- which is exactly the form device_argmax and gpu_optin use.
m_pe5() {
  python3 - "$1" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
new = ('if [ "$PASS" != 1 ]; then\n'
       '  echo "TP2-27B TEST: FAIL" >&2\n'
       '  exit 0   #PE5 green FAIL branch in the split form\n'
       'fi\n'
       'echo "TP2-27B TEST: PASS"')
assert s.count(old) == 1, s.count(old)
open(p, 'w').write(s.replace(old, new))
PY
}

# PE-6: a bare PASS echo (shape (iii)) whose preceding line exits 0. The
# checker accepts the bare echo only if a non-zero exit precedes it, so the
# defect is to swap that exit for 0 -- the shell then returns the echo's rc.
# ISOLATES (d): the conversion is APPENDED AFTER, not replaced, so (a)(b)(c)
# all still hold (the conversion line is untouched) and only the tail is
# defective. (Deleting the conversion, as a first draft did, trips (a) first
# and proves nothing about (d); replacing the trailing rm, as a second draft
# did, left the conversion as the last statement and proved even less.)
m_pe6() {
  python3 - "$1" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
conv = 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
assert s.count(conv) == 1, s.count(conv)
new = (conv + '\n'
       'exit 0   #PE6 preceding exit is green\n'
       'echo "TP2-27B TEST: PASS"')
open(p, 'w').write(s.replace(conv, new))
PY
}

# PE-7: shape (i) accepts any `exit`, including 0. The defect is to make the
# last statement `exit 0`, which the shell returns as PASS. (b)'s second grep
# keys on the $PASS conversion, so it does not fire here -- only (d) can.
# ISOLATES (d) for the same reason as PE-6: the conversion stays, the defect
# is appended after it.
m_pe7() {
  python3 - "$1" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
conv = 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
assert s.count(conv) == 1, s.count(conv)
new = conv + '\nexit 0   #PE7 last statement exits 0 -> shell returns PASS'
open(p, 'w').write(s.replace(conv, new))
PY
}

run_state PE-5 'FAIL \(b\)' m_pe5
run_state PE-6 'FAIL \(d\)' m_pe6
run_state PE-7 'FAIL \(d\)' m_pe7

# --- residue: the tracked file must be byte-identical to its pre-run state ---
if [ "$(cat "$TARGET" | cksum)" != "$TARGET_DIGEST_BEFORE" ]; then
  echo "RESIDUE: $TARGET was modified by the matrix" >&2
  bad=$((bad+1))
else
  echo "residue: ok (target byte-identical to its pre-run state)"
fi

# --- verdict. Printed LAST, after the table, never before it -----------------
{
  echo
  for r in "${RESULTS[@]}"; do echo "  $r"; done
  echo
  echo "states=$total caught=$caught baselines=$baselines"
  # One baseline for the whole run: there is exactly one unmutated gate.
  # caught == total, baselines >= 1, bad == 0 are the three conditions.
  if [ "$caught" -ne "$total" ] || [ "$baselines" -lt 1 ] || [ "$bad" -ne 0 ]; then
    echo "RESULT: FAIL ($caught/$total caught, $baselines baseline(s), $bad other problem(s))"
  else
    echo "RESULT: PASS ($caught/$total states caught, each for its own reason)"
  fi
} | tee "$BUILD/summary.txt"

# Same three conditions as the printed verdict, so the exit code can never
# disagree with the line above it.
if [ "$caught" -ne "$total" ] || [ "$baselines" -lt 1 ] || [ "$bad" -ne 0 ]; then
  exit 1
fi
{ echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "states=$total"
  echo "caught=$caught"
  echo "baselines=$baselines"
} >"$STAMP.tmp" && mv "$STAMP.tmp" "$STAMP"
exit 0
