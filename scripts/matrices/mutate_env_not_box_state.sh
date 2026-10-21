#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_env_not_box_state.sh.
#
# The artifact is scripts/lib/_gate_precond.sh -- the shared device preconditions.
# These states mutate THAT, not the checker, because the claim being certified is
# a property of the helper's runtime behaviour and messages, and weakening the
# checker cannot make an unprepared shell start printing a banner.
#
# EN-1 is the state that matters most and the one this whole family exists for:
# delete the diagnostic call from a single skip path, and the "cannot measure"
# line goes back to reading like a statement about the hardware -- exactly the
# text that caused a misattributed hardware-fault claim in this session.
#
# EN-2 is the ALWAYS-RED TWIN and is the one most easily forgotten: delete the
# banner TEXT but keep every call site. A checker that only ever checks "is the
# function called" would pass EN-2, because the calls are all still there. The
# check must be the OTHER way round too -- the banner must actually print, which
# is what its live probe exists to establish.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/env_box_state_matrix"
CHECK="scripts/checks/check_env_not_box_state.sh"
ART="scripts/lib/_gate_precond.sh"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"
PRISTINE="$SB/pristine.artifact"
cp "$ART" "$PRISTINE"

patch_art() { python3 - "$ART" "$1" "$2" <<'PY'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) < 1:
    print("anchor not found: %r" % (o,), file=sys.stderr); sys.exit(1)
open(p, 'w').write(s.replace(o, n))
PY
}

baseline() {
  bash "$CHECK" >"$SB/base.log" 2>&1; local rc=$?
  [ "$rc" -eq 0 ] || { echo "  FAIL baseline rc=$rc" >&2; tail -6 "$SB/base.log" >&2; return 1; }
  grep -q '^RESULT: ' "$SB/base.log" || { echo "  FAIL baseline has no RESULT:" >&2; return 1; }
  echo "  ok baseline rc=0, PASS verdict present"
}

total=0; caught=0; RESULTS=()
run_state() {
  local id="$1" expect="$2" want="$3" out rc
  total=$((total + 1))
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  cp "$PRISTINE" "$ART"
  if [ "$want" = 1 ]; then
    if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
    elif ! printf '%s' "$out" | grep -qE -- "$expect"; then RESULTS+=("$id MISSED '(wrong reason)'")
    else caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''"); fi
  else
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE)'"); fi
  fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  EN-1  one skip path stops calling the diagnostic    [must REJECT]"
  echo "  EN-2  all call sites kept, banner text deleted     [must REJECT]"
  echo "  EN-3  the banner asserts BOX STATE again           [must REJECT]"
  echo "  EN-4  the remedy line removed                      [must REJECT]"
  exit 0
fi

echo "== baseline =="
baseline || exit 1
echo "== states =="

# EN-1: one skip path stops naming the environment. ALL call sites are removed
# from gate_gpu_count's path, which is the one this session misread.
en1() { python3 - "$ART" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
# drop every diagnostic call that follows the sycl-ls SKIP line
out, dropped = [], 0
i = 0
lines = s.split('\n')
while i < len(lines):
    out.append(lines[i])
    if 'SKIP: $sycl_ls not available' in lines[i]:
        j = i + 1
        while j < len(lines) and lines[j].strip() != '' :
            out.append(lines[j]); j += 1
        # the diagnostic line is the one right before `return 1`
        for k in range(len(out) - 1, -1, -1):
            if 'gate_env_diagnose' in out[k]:
                out[k] = '    :  #EN1 diagnostic removed'
                break
        i = j
        continue
    i += 1
open(p, 'w').write('\n'.join(out))
PY
}

# EN-2: THE ALWAYS-RED TWIN. Keep every call site, destroy the banner. A
# call-site-count check alone would sail through this.
en2() { patch_art 'ENVIRONMENT, NOT BOX STATE' 'SOME PROBLEM' ; }

# EN-3: the banner starts asserting hardware state again.
en3() { patch_art 'ENVIRONMENT, NOT BOX STATE' 'BOX STATE, NOT A CODE FAILURE' ; }

# EN-4: the remedy is deleted, leaving "not box state" with nothing to do.
en4() { patch_art 'source /opt/intel/oneapi/setvars.sh     (then re-run)' 'see the docs' ; }

en1; run_state EN-1 'diagnostic call site|Fail|FAIL' 1
en2; run_state EN-2 'ENVIRONMENT banner is really printed|FAIL' 1
en3; run_state EN-3 'asserts .BOX STATE|FAIL' 1
en4; run_state EN-4 'no concrete remedy|FAIL' 1

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

if ! cmp -s "$ART" "$PRISTINE"; then
  echo "  FAIL residue: $ART differs from its pre-matrix state" >&2
  exit 1
fi
echo "  ok residue: $ART byte-identical to its pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "ENV/BOX-STATE MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  EN-1: a skip path that stops naming the environment is observable"
  echo "  EN-2: banner text deleted while every call site survives is observable"
  echo "  EN-3: re-asserting BOX STATE in the banner is observable"
  echo "  EN-4: deleting the remedy is observable"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "ENV/BOX-STATE MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total states caught)" >&2
exit 1
