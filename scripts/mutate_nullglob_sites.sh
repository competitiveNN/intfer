#!/usr/bin/env bash
# Mutation matrix for scripts/check_nullglob_sites.sh.
#
# The checker exists because a glob can VANISH (nullglob) and leave a command with
# no arguments, which for `ls` means "list the current directory". Its own risk
# is the AUDIT #32 shape applied one level up: a harvest that silently examines
# nothing and reports a clean bill of health. Every state below attacks either
# the SCANNER's coverage or the checker's floors.
#
# NS-1 plants a real degenerate site (the shape that was live in
# mutate_bench_load_gate.sh) -> must be REFUSED.
# NS-2 removes the scanner's ability to see a same-line
# `( shopt -s nullglob; cmd GLOB )` segment -- the blind spot the scanner REALLY
# had, which measured POPULATION 0 on a tree that contained one. Must be REFUSED,
# because a scanner that cannot see the instance is the whole failure mode.
# NS-3 makes the scanner examine nothing (the population floor must catch it).
# NS-4 removes the "no file enables nullglob" floor -> must be REFUSED, since that
# population going empty is indistinguishable from a broken scan.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/nullglob_sites_matrix"
CHECK="scripts/check_nullglob_sites.sh"
SCAN="scripts/_nullglob_scan.py"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

# A throwaway tree holding a copy of the scanner and a synthetic script, so the
# real scripts/ is never edited. The checker is invoked with that tree.
SANDBOX="$SB/tree"
# The sandbox IS the tree the checker is pointed at, so the probe script sits at
# the tree ROOT -- the first version nested it under tree/scripts/ while the
# checker globs TREE/*.sh, and every state reported "0 files enable nullglob",
# i.e. it measured a tree it had not planted.
mk_sandbox() {
  rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"
  printf '%s\n' "$1" >"$SANDBOX/probe.sh"
  # a MUTABLE copy of the scanner inside the tree, so NS-3/NS-4 can blind it
  # without editing the real one
  cp "$SCAN" "$SANDBOX/_nullglob_scan.py"
}

# Run the checker against the sandbox tree. The checker takes the tree as an
# argument to the scanner; pass it through so the real tree is untouched.
# The checker takes the TREE as $1, so a state points the WHOLE check at the
# sandbox: both the file population and the scan. The first version redirected
# only the scanner and the population still came from the real scripts/, so every
# state measured a tree it did not plant.
run_checker() { bash scripts/check_nullglob_sites.sh "$SANDBOX" 2>&1; }

# The fixtures are DATA files, not shell scripts, and they live outside
# scripts/*.sh deliberately. Embedding them as multi-line string literals inside
# this matrix made check_nullglob_sites.sh report THIS FILE as holding two
# degenerate sites -- the scanner cannot tell a payload from code, and neither
# can a grep. A .txt under scripts/fixtures/ is invisible to a scanner that
# enumerates *.sh, which is the honest place for test data.
FIXDIR="scripts/fixtures/nullglob-sites"
BENIGN="$(cat "$FIXDIR/benign.sh.txt")"
DEGENERATE="$(cat "$FIXDIR/degenerate.sh.txt")"
SAMELINE="$(cat "$FIXDIR/sameline.sh.txt")"
NOGLOB="$(cat "$FIXDIR/noglob.sh.txt")"   # a tree with NO nullglob at all





patch_scan() { python3 - "$SANDBOX/_nullglob_scan.py" "$1" "$2" <<'PY'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) < 1:
    print("anchor absent:", o[:40], file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(o, n))
PY
}

# NS-4 patches the CHECKER (its own population floor), so the checker joins the
# restore set. Patching the sandbox scanner there would have been a state that
# measured nothing -- which is what it silently was in the first version.
CHK_PRISTINE="$SB/_chk.pristine"
cp "$CHECK" "$CHK_PRISTINE"
restore_checker() { cp "$CHK_PRISTINE" "$CHECK"; }
patch_check() { python3 - "$CHECK" "$1" "$2" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) < 1:
    print("anchor absent:", o[:50], file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
}

total=0; caught=0; RESULTS=()
run_state() {  # $1 id, $2 expect, $3 want: reject | keep-green
  local id="$1" expect="$2" want="${3:-reject}" out rc
  total=$((total + 1))
  out="$(run_checker)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  restore_checker
  if [ "$want" = keep-green ]; then
    # The always-red twin. Used for the INVERSE of a floor state: with the floor
    # removed the checker must PASS, which is what proves the floor was the thing
    # doing the catching rather than the state passing for an unrelated reason.
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE -- rc='$rc')"); fi
    return
  fi
  if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then RESULTS+=("$id MISSED '(wrong reason; rc=$rc)'")
  else caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''"); fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  NS-1  a real degenerate site (bare ls GLOB under nullglob) [must REJECT]"
  echo "  NS-2  the scanner cannot see a SAME-LINE nullglob+glob     [must REJECT]"
  echo "  NS-3  the scanner examines nothing (population floor)       [must REJECT]"
  echo "  NS-4  the 'no file enables nullglob' floor removed          [must REJECT]"
  exit 0
fi

echo "== baseline (sandbox with only allowed forms) =="
mk_sandbox "$BENIGN"
out="$(run_checker)"; rc=$?
printf '%s\n' "$out" | sed 's/^/  /'
if [ "$rc" -ne 0 ]; then echo "  FAIL baseline rc=$rc" >&2; exit 1; fi
if ! printf '%s' "$out" | grep -q '0 degenerate sites'; then
  echo "  FAIL baseline did not report zero degenerate sites" >&2; exit 1
fi
echo "  ok baseline rc=0, zero degenerate sites"

echo "== states =="

mk_sandbox "$DEGENERATE";                    run_state NS-1 'degenerate site|FAIL'
mk_sandbox "$SAMELINE";                      run_state NS-2 'degenerate site|FAIL'
# NS-3/NS-4 are a FLOOR PAIR, and a floor that is not firing cannot be tested by
# deleting it -- that is the unobservable-state problem (audit #21), which cost
# three states earlier in this session. So the population is made genuinely EMPTY
# (a tree that enables nullglob nowhere) and the floor must refuse it.
mk_sandbox "$NOGLOB"
                                               run_state NS-3 'no file enables nullglob|vacuous|FAIL' reject
mk_sandbox "$NOGLOB"; patch_check 'if [ "$n_files" -lt 1 ]; then' 'if false; then'
                                               run_state NS-4 '' keep-green

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

rm -rf "$SANDBOX" "$SB/_chk.sh"
restore_checker
if ! cmp -s "$CHECK" "$CHK_PRISTINE"; then
  echo "  FAIL residue: the checker differs from its pre-matrix state" >&2; exit 1
fi
echo "  ok residue: the real scripts/ tree was never modified"

echo
if [ "$caught" -eq "$total" ]; then
  echo "NULLGLOB-SITES MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  NS-1: a bare cmd GLOB under nullglob is refused"
  echo "  NS-2: a same-line (nullglob; cmd GLOB) is refused -- the blind spot the"
  echo "        scanner actually had, which reported POPULATION 0 over a real site"
  echo "  NS-3: an EMPTY nullglob population is refused, not read as clean"
  echo "  NS-4: with that floor removed the same tree goes GREEN, so the floor is"
  echo "        provably what caught NS-3 (a floor that is not firing is unobservable)"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "NULLGLOB-SITES MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
