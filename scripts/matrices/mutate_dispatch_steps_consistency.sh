#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_dispatch_steps_consistency.sh.
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
#
# DS-6 is the state that motivated section 4 of the checker: a step whose SCRIPT
# FILE DOES NOT EXIST, planted so that it is armed, scheduled and named exactly
# -- every textual section passes and only the filesystem one can refuse. It is
# the audit-#18 family's last member ("a step that cannot run"), reached by
# deletion rather than by corruption, and a matrix that stops at DS-5 cannot see
# it because no state planted here ever removed a file.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/dispatch_steps_matrix"
CHECK="scripts/checks/check_dispatch_steps_consistency.sh"
T="scripts/run_all_checks.sh"
CERT="scripts/checks/check_stack_cert.sh"
# *** SANDBOX LOCK, TAKEN OUTSIDE THE TREE IT WIPES. *** $SB below is a FIXED path
# and is rm -rf'd at start, so two concurrent invocations share one sandbox: the
# second empties the first's temp root mid-state and the first then grades a state
# against a corpus the other just restored (MEASURED 2026-10-21). The lock lives
# under $BUILD, OUTSIDE $SB, and is taken BEFORE the wipe. Derived from source by
# check_matrix_sandbox_locks.sh.
LOCK="$BUILD/.mutate_dispatch_steps_consistency.lock"
mkdir -p "$BUILD" 2>/dev/null || true
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another .mutate_dispatch_steps_consistency.lock is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM


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
  echo "  DS-6  a scheduled step whose script file does NOT exist    [must REJECT]"
  exit 0
fi

echo "== baseline =="
baseline || exit 1
echo "== states =="

# DS-1 THE SHIPPED BUG. A dispatch arm whose script path contains a space: it
# cannot execute, and `bash -n` passes it because it is valid syntax.
# The anchor is DERIVED, for the reason DS-5's and DS-6's are: it used to spell
# the arm body as `$SCRIPT_DIR/../checks/check_env_not_box_state.sh`, and the 2026-10-20
# hierarchy move rewrote that segment to `$SCRIPT_DIR/checks/...` -- so the
# literal anchor found 0 occurrences, patch1 refused, and the state MISSED while
# the checker was (correctly) still passing. Read the arm's real body off the
# tree instead of re-pointing the literal.
ds1() { python3 - "$T" <<'PYD'
import re, sys
p = sys.argv[1]
lines = open(p).read().split("\n")
old = new = None
for k, line in enumerate(lines):
    m = re.search(r'bash "\$SCRIPT_DIR/[^"]*check_env_not_box_state\.sh"; return \$\? ;;', line)
    if m:
        old = m.group(0)
        new = old.replace('"; return', ' check_destructive_git.sh"; return', 1)
        break
if old is None:
    sys.exit("no check_env_not_box_state.sh arm body found")
s = open(p).read()
if s.count(old) != 1:
    sys.exit("anchor found %d times (want 1)" % s.count(old))
open(p, "w").write(s.replace(old, new, 1))
PYD
}

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
# DS-5's anchor is DERIVED, for the same reason N-V's is: it used to name the
# checkers list's tail literally, and adding check_nullglob_sites moved that tail,
# so the state MISSED while the checker was, correctly, still agreeing. A state
# anchored to a moving list is the recurring failure in this session, and the fix
# is always the same -- derive the anchor, do not re-point it.
ds5() { python3 - "$T" <<'PYD'
import re, sys
p = sys.argv[1]
lines = open(p).read().split("\n")
idx = None
for k, line in enumerate(lines):
    m = re.match(r'\s*STEPS="([^"]*)"\s*$', line)
    if m and m.group(1).split() and all(t.endswith(".sh") for t in m.group(1).split()):
        idx = k
        break
if idx is None:
    sys.exit("no checkers STEPS list found")
body = re.match(r'(\s*STEPS=")([^"]*)("\s*)$', lines[idx])
toks = body.group(2).split()
lines[idx] = body.group(1) + " ".join(toks) + '" zz_stray_step_token.sh' + body.group(3)
open(p, "w").write("\n".join(lines))
PYD
}
ds5; run_state DS-5 'stray step token|FAIL'

# DS-6 a step that is SCHEDULED, has an ARM, and has a body naming the label
# exactly -- but whose SCRIPT FILE DOES NOT EXIST. Sections 1-3 all compare
# strings to strings and cannot see it: memberships agree in both directions and
# the body is faithful, so the only predicate in the checker that can fire is
# the filesystem one. This is the state that motivated section 4, and it is
# planted so that every other section PASSES -- a state caught by the wrong
# predicate proves the wrong thing. The anchors are DERIVED (first matching
# STEPS list, first matching arm line) for the reason DS-5's are: a state
# anchored to a literal in a moving list MISSES while the checker is correctly
# still agreeing.
ds6() { python3 - "$T" <<'PYD'
import re, sys
p = sys.argv[1]
lines = open(p).read().split("\n")
name = "zz_no_such_file_dispatched.sh"
idx = None
for k, line in enumerate(lines):
    m = re.match(r'\s*STEPS="([^"]*)"\s*$', line)
    if m and m.group(1).split() and all(t.endswith(".sh") for t in m.group(1).split()):
        idx = k
        break
if idx is None:
    sys.exit("no checkers STEPS list found")
body = re.match(r'(\s*STEPS=")([^"]*)("\s*)$', lines[idx])
lines[idx] = body.group(1) + body.group(2) + " " + name + body.group(3)
arm_idx = None
for k, line in enumerate(lines):
    if re.match(r'\s*[A-Za-z0-9_]+\.sh\)\s*$', line):
        arm_idx = k
        break
if arm_idx is None:
    sys.exit("no dispatch arm found")
lines[arm_idx:arm_idx] = ['    %s)' % name,
                          '      bash "$SCRIPT_DIR/%s"; return $? ;;' % name]
open(p, "w").write("\n".join(lines))
PYD
}
ds6; run_state DS-6 'SCHEDULED but scripts/[^ ]* does not exist'

# ISOLATION, and it is not optional: the generic harness is satisfied by ONE
# matching line, so DS-6 would be recorded CAUGHT if sections 1, 2 or 3 fired
# instead of section 4 -- which would certify the wrong predicate and leave the
# filesystem check exactly as unproven as before. The planted step is armed,
# scheduled and named faithfully BY CONSTRUCTION, so the other sections' FAIL
# strings must be absent from DS-6's log. A defect that would break this is any
# future change that makes the STEPS-entry or arm-body comparison reject a name
# it has not been asked about.
if grep -q 'SCHEDULED but scripts/' "$SB/DS-6.log" \
   && ! grep -q 'has NO dispatch arm' "$SB/DS-6.log" \
   && ! grep -q 'NOT in STEPS' "$SB/DS-6.log" \
   && ! grep -q 'stray step token' "$SB/DS-6.log" \
   && ! grep -q 'a filename with a space' "$SB/DS-6.log"; then
  echo "  ok    DS-6 isolated to section 4 (no other section fired on it)"
else
  echo "  FAIL  DS-6 was NOT isolated to section 4 -- the state caught by the" >&2
  echo "        wrong predicate proves the wrong thing; graders must not pass it" >&2
  cat "$SB/DS-6.log" >&2
  exit 1
fi

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
  echo "  DS-6: a step scheduled with no script on disk is refused (section 4)"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DISPATCH/STEPS MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
