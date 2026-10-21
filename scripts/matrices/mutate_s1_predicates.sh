#!/usr/bin/env bash
# mutate_s1_predicates.sh -- the eight-state negative direction for
# check_s1_predicates.sh. A predicate that can only pass is not a predicate.
#
# WHAT WAS ALREADY WRONG WHEN THIS WAS BUILT, AND IS FIXED HERE. Three of these
# eight states caught defects in the CHECKER rather than in the probe, and the
# standing fix is asserted in this file rather than left as an intention: every
# mutation asserts it LANDED before it is graded.
#
#   * DEAD CONJUNCTION -- a shell function returns only its LAST statement's
#     status, so `p_a; p_b` silently discarded every guard clause before the last
#     and reported green on a source missing seven of them. Each predicate is
#     therefore evaluated on its OWN line and appended to a failure list, and the
#     verdict is taken from the list's length.
#   * INVERTED POLARITY -- a state that DELETED `getenv` made the source MORE
#     compliant than the healthy original, so "the checker refused it" proved
#     nothing. ND-envvar here PLANTS an env reference into a copy that has none.
#   * UNLANDED MUTATION -- a planted `\n` stayed a literal two-character escape,
#     gluing the claim onto the line it was meant to be absent from, so the
#     mutation changed nothing and was graded as a catch. Every state below
#     asserts its own landing with `cmp`.
#
# THE STATES, each named for the predicate it must trip:
#   ND-distinct   delete the same-buffer guard            -> p_distinct
#   ND-hosttype   delete the ZE_MEMORY_TYPE_HOST guard    -> p_host_type
#   ND-token      delete the per-round token comparison   -> p_token
#   ND-passes2    delete the --passes >= 2 refusal        -> p_passes2
#   ND-guard      move that refusal ABOVE the CLI parse   -> p_guard_after_parse
#   ND-report     delete the `RESULT: PASS` relay         -> p_report
#   ND-envvar     plant an INTFER_ env reference          -> p_no_env
#   ND-verdict    plant a printed verdict                 -> p_no_verdict
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIVE="$ROOT/examples/uncached_doorbell_probe/main.cpp"
CHECKER="$ROOT/scripts/checks/check_s1_predicates.sh"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/s1mut.XXXXXX")"
trap 'rm -rf "$SANDBOX"' EXIT
# INT/TERM must ALSO reach the cleanup: an EXIT trap alone never runs on a signal
# (bash fires EXIT only at a reached end or an explicit exit), so a SIGTERM killed
# the script outright and leaked $SANDBOX -- AUDIT #29. The exit routes the signal
# back through the EXIT handler; "clean up and resume" leaves the process alive,
# which is strictly worse (check_trap_release.sh).
trap 'rm -rf "$SANDBOX"; exit 130' INT
trap 'rm -rf "$SANDBOX"; exit 143' TERM

die() { echo "S1-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }

# ---- ANCHOR PREFLIGHT. Every mutation targets an anchor; if an anchor is not
#      present in the live source, the state below is vacuous by construction and
#      must not be graded.
for a in 'base_buf == (void*)cand_buf' \
         'cand_type != (int)ZE_MEMORY_TYPE_HOST' \
         'buf[i] != (std::uint8_t)(token' \
         'if (passes < 2)' \
         'for (int i = 1; i < argc' \
         'RESULT: PASS'; do
  n="$(grep -c -F -- "$a" "$LIVE")"
  if [ "$n" -lt 1 ]; then die "anchor absent from the live source: $a"; fi
done
if grep -qE 'getenv|setenv|INTFER_' "$LIVE"; then
  die "the live source already carries an env reference; ND-envvar would be vacuous"
fi
echo "S1-MUT: anchor preflight ok (6 anchors, env-free baseline confirmed)"

# ---- BASELINE. The unmutated source must PASS the checker, or every "caught"
#      below is indistinguishable from a checker that refuses everything.
bash "$CHECKER" "$LIVE" >"$SANDBOX/baseline.txt" 2>&1
if [ $? -ne 0 ]; then
  cat "$SANDBOX/baseline.txt"
  die "the healthy source FAILS the checker -- the negative direction is meaningless"
fi
grep -q 'S1-PRED: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
echo "S1-MUT: baseline PASS on the unmutated source"

BEFORE="$(sha256sum "$LIVE" | cut -d' ' -f1)"

# ---- plant: copy, mutate, assert LANDED --------------------------------
# $1 = state name, $2 = target file, $3 = description
plant_and_grade() {
  local state="$1" f="$2" desc="$3" rc
  cp -f "$LIVE" "$f"
  "$4" "$f" || die "$state: the mutator command failed"
  # THE STANDING FIX: assert the mutation LANDED before grading it. An unlanded
  # mutation changes nothing, so a "catch" from it certifies the checker never
  # saw the defect -- the exact vacuous green this file exists to prevent.
  if cmp -s "$f" "$LIVE"; then
    die "$state: the mutation did NOT LAND -- refusing to grade a no-op as a catch"
  fi
  bash "$CHECKER" "$f" >"$SANDBOX/$state.out" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "S1-MUT: $state NOT CAUGHT -- the checker passed a defective source"
    sed 's/^/    /' "$SANDBOX/$state.out"
    return 1
  fi
  local why
  why="$(sed -n 's/^S1-PRED: FAIL -- //p' "$SANDBOX/$state.out" | head -1)"
  echo "S1-MUT: $state CAUGHT ($why)"
  return 0
}

# ---- the eight mutators. Each is a one-purpose function so the LANDED
#      assertion above can name which one failed.
m_distinct() { sed -i '/base_buf == (void\*)cand_buf/d' "$1"; }
m_hosttype() { sed -i '/cand_type != (int)ZE_MEMORY_TYPE_HOST/d' "$1"; }
m_token()    { sed -i '/buf\[i\] != (std::uint8_t)(token/d' "$1"; }
m_passes2()  { sed -i '/if (passes < 2)/d' "$1"; }
m_report()   { sed -i '/std::printf("RESULT: PASS/d' "$1"; }
m_envvar()   { sed -i '/^int main(int argc, char\*\* argv) {$/a\  if (getenv("INTFER_S1")) bytes = 1;' "$1"; }
m_verdict()  { sed -i '/no verdict emitted: this is a precondition measurement/i\  std::printf("cand is faster and the flag is adopted\\n");' "$1"; }
# ND-guard MOVES the refusal above the parse. Deleting the original and planting
# the same text one statement earlier is what makes p_guard_after_parse the
# predicate that fires: the guard still exists, so p_passes2 is satisfied, and it
# now precedes the loop that reads --passes.
m_guard() {
  sed -i '/if (passes < 2)/d' "$1"
  sed -i '/^int main(int argc, char\*\* argv) {$/a\  if (passes < 2) return 1;  // MOVED ABOVE THE PARSE' "$1"
}

STATES_REAL=0
STATES_CAUGHT=0
for st in ND-distinct ND-hosttype ND-token ND-passes2 ND-guard ND-report ND-envvar ND-verdict; do
  fn="m_$(echo "$st" | sed 's/^ND-//;s/-/_/g')"
  if ! declare -f "$fn" >/dev/null; then die "$st: no mutator function $fn"; fi
  STATES_REAL=$((STATES_REAL + 1))
  if plant_and_grade "$st" "$SANDBOX/$st.cpp" "$(echo "$st" | tr a-z A-Z)" "$fn"; then
    STATES_CAUGHT=$((STATES_CAUGHT + 1))
  fi
done

if [ "$STATES_REAL" -ne 8 ]; then
  die "expected 8 states, graded $STATES_REAL -- a matrix that never ran cannot report a tally"
fi

# ---- RESIDUE. The tracked file must be byte-identical to its pre-run state.
AFTER="$(sha256sum "$LIVE" | cut -d' ' -f1)"
if [ "$AFTER" != "$BEFORE" ]; then
  die "the live source CHANGED during the matrix (sha $BEFORE -> $AFTER)"
fi
if [ -n "$(cd "$ROOT" && git status --porcelain -- examples/uncached_doorbell_probe 2>/dev/null | grep -v '^??')" ]; then
  die "git reports a tracked-file change under examples/uncached_doorbell_probe"
fi
echo "S1-MUT: residue byte-identical (sha256 ${AFTER:0:16}...)"

echo "S1-MUT: matrix $STATES_CAUGHT/$STATES_REAL states caught"
if [ "$STATES_CAUGHT" -eq "$STATES_REAL" ]; then
  echo "S1-PRED MATRIX: PASS ($STATES_CAUGHT/$STATES_REAL, LANDED asserted per state, residue byte-identical)"
  echo "RESULT: PASS"
  exit 0
fi
echo "S1-PRED MATRIX: FAIL"
echo "RESULT: FAIL"
exit 1
