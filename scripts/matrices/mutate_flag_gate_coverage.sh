#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_flag_gate_coverage.sh.
#
# WHAT THE CHECKER CLAIMS, and therefore what must be falsifiable: that the
# flag residue is CLASSIFIED -- opt-in vs kill-switch vs diagnostic vs
# operational -- and that the opt-in class is the only real certification gap.
#
# So the states attack the two things that could make that claim hollow:
#
#   * the HARVEST. If the queries stop finding flags, "0 ungated" becomes a
#     statement about nothing. FG-1 breaks the source-side query and FG-2 the
#     gate-side one; both must be REFUSED. This is the audit #3 vacuous-green
#     arriving through a refactor rather than a deletion, and it is the most
#     likely way this checker rots. FG-1 is what FOUND the real hole here: the
#     first version floored n_all but not n_src, so breaking the census regex
#     emptied the residue while n_all stayed at 62 and the checker reported
#     "0 ungated" and PASSED. FG-3 is its inverse and exists so FG-1 cannot be
#     counted as caught for some reason other than the floor.
#
#   * the CLASSIFIER. If a kill switch starts being filed as an opt-in gap, or
#     the reverse, the printed residue is wrong in exactly the way that sends
#     someone to write a stream-identity arm for a flag whose DEFAULT is what the
#     gates already test. FG-3 must therefore be REFUSED for the same reason a
#     drift is.
#
# FG-4 IS THE ALWAYS-RED TWIN and is the one most easily forgotten: naming an
# additional opt-in flag in the CLASSIFIER as if it were a diagnostic makes the
# residue SHORTER without changing anything real. A checker that only ever
# reports "more" would pass FG-1..FG-3 and go red here, i.e. it would be a
# checker nobody could act on.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/flag_gate_coverage_matrix"
# *** WIPE LOCK, OUTSIDE THE TREE IT WIPES (check_matrix_sandbox_locks.sh). ***
# The sandbox below is a FIXED path wiped on every invocation, so two concurrent
# runs share one tree: the second's wipe empties the first's mid-state and the
# first then grades a corpus the other has just rebuilt (MEASURED 2026-10-21 on
# mutate_durable_reads.sh). Taken BEFORE the first wipe and released on
# EXIT/INT/TERM, so a run that dies mid-state does not brick the next.
LOCK="$BUILD/.mutate_flag_gate_coverage.lock"
mkdir -p "$BUILD" 2>/dev/null || true
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_flag_gate_coverage.sh is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

CHECK="scripts/checks/check_flag_gate_coverage.sh"
CLS="scripts/checks/check_flag_gate_coverage.sh"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"
PRISTINE="$SB/pristine.checker"
cp "$CLS" "$PRISTINE"

baseline() {
  bash "$CHECK" >"$SB/base.log" 2>&1; local rc=$?
  [ "$rc" -eq 0 ] || { echo "  FAIL baseline rc=$rc" >&2; tail -6 "$SB/base.log" >&2; return 1; }
  grep -q '^RESULT: ' "$SB/base.log" || { echo "  FAIL baseline has no RESULT:" >&2; return 1; }
  grep -q '  FAIL ' "$SB/base.log" && { echo "  FAIL baseline printed a FAIL line" >&2; return 1; }
  echo "  ok baseline rc=0, PASS verdict present, no FAIL line"
}

total=0; caught=0; RESULTS=()
run_state() {
  local id="$1" expect="$2" want="$3" out rc
  total=$((total + 1))
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  cp "$PRISTINE" "$CLS"
  if [ "$want" = 1 ]; then
    if [ "$rc" -eq 0 ]; then
      RESULTS+=("$id MISSED '(still PASS -- unobservable)'")
    elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
      RESULTS+=("$id MISSED '(rejected, wrong reason)'")
    else
      caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    fi
  else
    if [ "$rc" -eq 0 ]; then
      caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else
      RESULTS+=("$id MISSED '(FALSE POSITIVE -- always-red)'")
    fi
  fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  FG-1  source-side harvest broken (anti-vacuity floor)   [must REJECT]"
  echo "  FG-2  gate-side harvest broken                          [must REJECT]"
  echo "  FG-3  census broken AND floor deleted  -> must STAY GREEN [inverse]"
  echo "  FG-4  a real opt-in flag reclassified as diagnostic     [must REJECT]"
  exit 0
fi

echo "== baseline =="
baseline || exit 1
echo "== states =="

# All four states use ONE patcher that takes the anchor as an argv value. The
# first three implementations embedded the anchors inside a python heredoc and
# every one of them died on quoting -- "SyntaxError: invalid decimal literal" --
# which surfaced as MISSED for all four states. An anchor passed as data cannot
# be re-quoted by the heredoc that consumes it.
patch_checker() {  # $1 old, $2 new
  python3 - "$CLS" "$1" "$2" <<'PY'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) != 1:
    print("anchor count %d (want 1): %r" % (s.count(o), o), file=sys.stderr)
    sys.exit(1)
open(p, 'w').write(s.replace(o, n, 1))
PY
}

# FG-1: the source-side census regex stops matching. This is what a reformat of
# the getenv calls would really look like, and "0 ungated" would then be a
# statement about nothing.
fg1() { patch_checker '[a-z_]*\("INTFER_[A-Z0-9_]+"'"'"' -- src examples include' 'Zzz[_-]never[_-]matches'"'"' -- src examples include'; }

# FG-2: same for the gate-side query.
fg2() { patch_checker "INTFER_[A-Z0-9_]+' -- 'scripts/gates/test_" "Zzz_never' -- 'scripts/gates/test_"; }

# FG-3: delete the anti-vacuity floors outright -- the classic "weakening alone
# is unobservable while the real tree has full coverage".
# FG-3 IS THE INVERSE OF FG-1, and it is the state that keeps FG-1 honest.
# If the harvest is broken AND BOTH floors are gone, the checker must go GREEN --
# which proves the floors were the thing doing the catching in FG-1 and FG-4,
# rather than either being caught for some unrelated reason. Scoring this "must
# REJECT" was the first draft and it was wrong: deleting a guard that is not
# firing is unobservable on a healthy tree (audit #21), so it cannot be scored
# that way. NOTE both floors must go: the opt-in floor fires on the empty
# residue independently, so deleting only the census floor still goes red -- which
# is FG-3 catching itself, and is why the first version of this state failed as
# a FALSE POSITIVE rather than as a missing observation.
# ALL THREE floors must go, not two. A third harvest (the '=0 disables' class)
# gained its own floor, and with only the first two removed that third fired on
# the emptied residue -- so FG-3 went red for a reason unrelated to what it tests.
# That is the always-red twin working in REVERSE: the false positive arrived from
# ADDING a guard rather than removing one.
fg3() { fg1
        patch_checker 'if [ "$n_src" -lt 10 ]; then'  'if false; then  #FG3 census floor gone'
        patch_checker 'if [ "$n_opt" -lt 1 ]; then'   'if false; then  #FG3 opt-in floor gone'
        patch_checker 'if [ "$n_dis" -lt 1 ]; then'   'if false; then  #FG3 disabler floor gone'; }

# FG-4: reclassify a genuine opt-in flag as a diagnostic, which SHORTENS the
# opt-in residue. A checker that can only report "more" fails here.
# FG-4 must empty the WHOLE class, not shave one flag off it: the opt-in floor
# is a floor of 1, so moving 11 -> 10 does not trip it, and a floor of 11 would
# be a hand-counted total -- which goes stale and is exactly what this repo
# forbids (audit #4). Emptying the class is observable AND keeps the total
# derived.
# The replacement must keep the braces BALANCED: an earlier version commented
# out the grep and swallowed the function's closing brace into the comment, so
# the checker died with a syntax error and the state was scored "rejected, wrong
# reason" -- a syntax error is not evidence a mechanism is load-bearing. Closing
# is_diag immediately and pushing the original text into a never-called dummy
# leaves valid shell.
fg4() { patch_checker 'is_diag() { grep -qE' 'is_diag() { true; }; is_diag_orig() { grep -qE'; }

fg1; run_state FG-1 'census query harvested only|could not be trusted' 1
fg2; run_state FG-2 'gate-side query is broken|could not be trusted' 1
fg3; run_state FG-3 '' 0
fg4; run_state FG-4 'class is EMPTY|residue is EMPTY|could not be trusted' 1

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

if ! cmp -s "$CLS" "$PRISTINE"; then
  echo "  FAIL residue: $CLS differs from its pre-matrix state" >&2
  exit 1
fi
echo "  ok residue: $CLS byte-identical to its pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "FLAG-COVERAGE MUTATION MATRIX: PASS ($caught/$total states, each for its own reason)"
  echo "  FG-1: a broken source-side harvest is REFUSED, not read as 'fully covered'"
  echo "  FG-2: a broken gate-side harvest is REFUSED"
  echo "  FG-3: with the harvest broken and the floor gone it goes GREEN, so the"
  echo "        floor is provably what catches FG-1 (an unobservable guard would too)"
  echo "  FG-4: a classifier that classifies the whole gap away is observable"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "FLAG-COVERAGE MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total states caught)" >&2
exit 1
