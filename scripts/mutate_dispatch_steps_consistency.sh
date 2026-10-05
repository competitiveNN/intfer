#!/usr/bin/env bash
# Mutation matrix for scripts/check_dispatch_steps_consistency.sh.
#
# What the checker claims: the dispatch case labels and the STEPS lists AGREE --
# every scheduled script has an arm, and every arm is scheduled or self-declared
# POST-STACK.
#
# DS-1 is the state this whole checker exists for, and it is the EXACT shape of the
# bug that shipped in this session: an arm's script list was corrupted into a
# single filename containing a space
#     bash "$SCRIPT_DIR/a.sh b.sh"
# which fails rc=127 at runtime. Reproduced here as a dispatch-arm mutation.
#
# DS-2 removes a STEPS entry. Section 0c already asserts every check_*/mutate_* is
# SCHEDULED, so this state is the always-red twin working in reverse: 0c would
# catch it, and this checker must too, because the two look at DIFFERENT sides of
# the same property and 0c cannot see a step whose script cannot run.
#
# DS-3 removes an arm while leaving it scheduled -- the audit #18 shape (an arm
# that can never be reached). Also caught by 0b, so again this checker is the
# independent second reader, not the only one.
#
# DS-4 strips the POST-STACK self-declaration from check_stack_cert.sh. The
# exemption must be DERIVED from the script's own header; if it were carried in
# this checker as a list, deleting the header would be invisible here.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/dispatch_steps_matrix"
CHECK="scripts/check_dispatch_steps_consistency.sh"
T="scripts/run_all_checks.sh"
CERT="scripts/check_stack_cert.sh"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

TOUCHED="$T $CERT"
PRISTINE="$SB/pristine"
mkdir -p "$PRISTINE"
for f in $TOUCHED; do cp "$f" "$PRISTINE/$(basename "$f")"; done
restore_all() { local f; for f in $TOUCHED; do cp "$PRISTINE/$(basename "$f")" "$f"; done; }

patch1() { python3 - "$1" "$2" "$3" <<'PYQ'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read(); c = s.count(o)
if c != 1:
    print("anchor found %d times (want 1) in %s" % (c, p), file=sys.stderr)
    sys.exit(1)
open(p, 'w').write(s.replace(o, n, 1))
PYQ
}
patchN() { python3 - "$1" "$2" "$3" <<'PYQ'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) < 1:
    print("anchor absent in %s" % p, file=sys.stderr); sys.exit(1)
open(p, 'w').write(s.replace(o, n))
PYQ
}

baseline() {
  bash "$CHECK" >"$SB/base.log" 2>&1; local rc=$?
  [ "$rc" -eq 0 ] || { echo "  FAIL baseline rc=$rc" >&2; tail -8 "$SB/base.log" >&2; return 1; }
  grep -q '^RESULT: ' "$SB/base.log" || { echo "  FAIL baseline has no RESULT:" >&2; return 1; }
  echo "  ok baseline rc=0, PASS verdict present"
}

total=0; caught=0; RESULTS=()
run_state() {
  local id="$1" expect="$2" out rc
  total=$((total + 1))
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  restore_all
  if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then RESULTS+=("$id MISSED '(wrong reason)'")
  else caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''"); fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  DS-1  an arm's script list corrupted into a spaced filename [must REJECT]"
  echo "  DS-2  a STEPS entry whose script has no arm                 [must REJECT]"
  echo "  DS-3  an arm removed while left scheduled                  [must REJECT]"
  echo "  DS-4  the POST-STACK self-declaration stripped              [must REJECT]"
  echo "  DS-5  a step token moved OUTSIDE the STEPS quotes          [must REJECT]"
  exit 0
fi

echo "== baseline =="
baseline || exit 1
echo "== states =="

# DS-1 THE SHIPPED BUG. A dispatch arm whose script path contains a space: it
# cannot execute, and `bash -n` passes it because it is valid syntax.
ds1() { patch1 "$T" 'bash "$SCRIPT_DIR/check_env_not_box_state.sh"; return $? ;;' \
            'bash "$SCRIPT_DIR/check_env_not_box_state.sh check_destructive_git.sh"; return $? ;;'; }

# DS-2 schedule a script that has no dispatch arm at all.
ds2() { patch1 "$T" 'STEPS="$STEPS run_ten_gates.sh"' \
            'STEPS="$STEPS run_ten_gates.sh check_no_such_arm.sh"'; }

# DS-3 remove an arm, leaving the script scheduled.
ds3() { patchN "$T" 'check_dri_probe.sh)' ': removed-by-DS3)'; }

# DS-4 remove the self-declared POST-STACK header, so the exemption this checker
# grants can no longer be derived.
ds4() { patchN "$CERT" '# POST-STACK:' '# (header removed by DS-4)'; }

ds1; run_state DS-1 'has NO dispatch arm|FAIL'
ds2; run_state DS-2 'has NO dispatch arm|FAIL'
ds3; run_state DS-3 'NOT in STEPS|FAIL'
ds4; run_state DS-4 'NOT in STEPS|FAIL'

# DS-5 a .sh token moved past the closing quote of a STEPS line. Inside the
# quotes everything is fine, which is why a checker that harvests only inside
# them is structurally blind to this -- the defect IS the token being on the
# wrong side of a quote. It becomes a COMMAND: the stack dies with
# "mutate_pending_operator.sh: command not found".
ds5() { patchN "$T" 'check_pending_operator.sh"' 'check_pending_operator.sh" mutate_pending_operator.sh'; }
ds5; run_state DS-5 'stray step token|FAIL'

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

leftover=""
for f in $TOUCHED; do cmp -s "$f" "$PRISTINE/$(basename "$f")" || leftover="$leftover $f"; done
if [ -n "$leftover" ]; then
  echo "  FAIL residue: differ from pre-matrix state:$leftover" >&2
  exit 1
fi
echo "  ok residue: all $(printf '%s\n' $TOUCHED | grep -c .) touched files byte-identical to their pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "DISPATCH/STEPS MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  DS-1: the spaced-filename arm (the bug that shipped) is observable"
  echo "  DS-2: a scheduled script with no arm is observable"
  echo "  DS-3: an arm removed while scheduled is observable"
  echo "  DS-4: the POST-STACK exemption is derived from the script, not a list here"
  echo "  DS-5: a step token outside the quotes is a COMMAND, and is refused"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DISPATCH/STEPS MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
