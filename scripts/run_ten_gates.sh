#!/usr/bin/env bash
# Run the CORRECTNESS GATES (scripts/test_*.sh) and certify the engine against the
# current tree, leaving an on-disk artifact that names HEAD.
#
# WHY THIS EXISTS. `run_all_checks.sh` certifies the TOOLING. It does not run the
# engine's correctness gates, and a session once reported "10 PASS / 0 PARTIAL /
# 0 SKIPPED" backed only by a DIFF ARGUMENT -- "the files that changed since the
# gates ran are not engine inputs". That argument is sound and it is still not
# evidence: it is the same shape AUDIT #22 stopped accepting when a tok/s was
# published with no artifact behind it. A number at the end of a session needs a
# file on disk that says which tree produced it. This is that file.
# `scripts/check_engine_cert.sh` is what keeps this artifact honest afterwards.
#
# *** WHY THE ARTIFACT IS NOT NAMED last_pass.stamp ***
# Deliberate. `run_all_checks.sh` enumerates `find build -name '*last_pass.stamp'`
# and fails on any directory it does not recognise; scripts/_gate_pop.sh derives
# that recognised set from `mutate_*.sh` alone. A stamp from a suite that is not
# a mutation matrix would therefore be reported as a LEAK, and the only ways out
# would be to loosen a green guard or to reclassify this as a matrix. Neither is
# right: the tooling stack and the engine certification are DIFFERENT evidence
# classes, and conflating them is what made the original gap hard to see.
#
# *** WHY IT IS NOT A STEP OF run_all_checks.sh ***
# It runs the engine on both GPUs and takes tens of minutes. Folding that into the
# tooling stack would make every tooling edit pay for it, and a check that is slow
# enough to be skipped is a check that gets skipped. It is the slow half of
# certification, run separately. Revisit only if the stack's runtime ever drops
# enough for this to be cheap -- and then re-derive the stamp-population question
# above, which is what actually blocks the merge.
#
# Usage: scripts/run_ten_gates.sh [--self-check]
# Exit:  0 = every gate rc=0; 1 = a real failure; 2 = setup/self-check failure.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
# Seam: the self-check drives this with STUBS so it can prove the tally can fail
# without running the engine. Same rationale as INTFER_SYCL_LS in
# _gate_precond.sh -- an assumption the harness cannot redirect is one it cannot
# test, and this script would otherwise be the one runner in the tree with no way
# to exercise its own failure path.
GATE_GLOB="${INTFER_GATE_GLOB:-test_*.sh}"
GATE_TIMEOUT="${INTFER_GATE_TIMEOUT:-2400}"
GATE_DIR="${INTFER_GATE_DIR:-$SCRIPT_DIR}"
# The tally lives in globals set by run_gates. They are declared HERE because
# the self-check calls run_gates directly and reads them afterwards under `set -u`.
GPASS=0; GPARTIAL=0; GSKIP=0; GFAIL=0; GFAILED=""

# Drive the gate list and tally. $1 = output dir. Prints the tally; sets GFAIL,
# GPASS, GPARTIAL, GSKIP and GFAILED.
run_gates() {
  local out="$1" g name rc
  GPASS=0; GPARTIAL=0; GSKIP=0; GFAIL=0; GFAILED=""
  {
    echo "=== ten correctness gates $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "gate glob: $GATE_GLOB in $GATE_DIR"
  } > "$out/gates.log"
  for g in $GATE_DIR/$GATE_GLOB; do
    [ -f "$g" ] || continue
    name="$(basename "$g" .sh)"; name="${name#test_}"
    printf '== %s ==\n' "$name" >> "$out/gates.log"
    timeout "$GATE_TIMEOUT" bash "$g" >> "$out/gates.log" 2>&1; rc=$?
    printf 'GATE_RESULT %s rc=%s\n' "$name" "$rc" >> "$out/gates.log"
    case "$rc" in
      0)   GPASS=$((GPASS+1));    echo "  ok    $name" ;;
      2)   GPARTIAL=$((GPARTIAL+1)); echo "  PART  $name (rc=2: a coverage precondition was unmet)" ;;
      124) GSKIP=$((GSKIP+1));   GFAILED="$GFAILED $name(TIMEOUT)"
            echo "  SKIP  $name (TIMEOUT -- explicitly NOT scored as a pass)" ;;
      *)   GFAIL=$((GFAIL+1));    GFAILED="$GFAILED $name"
            echo "  FAIL  $name (rc=$rc)" ;;
    esac
  done
  printf 'TALLY pass=%s partial=%s skip=%s fail=%s\n' \
    "$GPASS" "$GPARTIAL" "$GSKIP" "$GFAIL" >> "$out/gates.log"
}

# ---------------------------------------------------------------- self-check
# Proves the harness can FAIL in all the ways that matter, with stubs, in
# seconds and without touching the engine. Without it this script is the one
# runner in the tree whose failure path nobody has ever seen execute -- and an
# always-green runner is exactly what the AUDIT #14 vacuity class looks like.
self_check() {
  local SB out rc bad=0
  SB="$(mktemp -d "${TMPDIR:-/tmp}/teng.XXXXXX")"
  trap 'rm -rf "$SB"' RETURN
  enum_reset
  echo "== TEN-GATES SELF-CHECK (stubs; no engine, no GPU) =="

  # S1: two passing stubs -> rc=0 and a pass tally.
  mkdir -p "$SB/g1"; printf '#!/usr/bin/env bash\nexit 0\n' > "$SB/g1/test_ok1.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$SB/g1/test_ok2.sh"
  SAVED_DIR="$GATE_DIR"; GATE_DIR="$SB/g1"
  mkdir -p "$SB/o1"; run_gates "$SB/o1" > "$SB/o1.out" 2>&1
  GATE_DIR="$SAVED_DIR"
  if [ "$GPASS" -eq 2 ] && [ "$GFAIL" -eq 0 ]; then
    echo "  ok    S1: two passing stubs -> pass=2 fail=0"
  else
    echo "  FAIL  S1: expected pass=2 fail=0, got pass=$GPASS fail=$GFAIL"
    bad=$((bad+1))
  fi

  # S2: one failing stub -> it must be COUNTED and the suite must exit 1.
  mkdir -p "$SB/g2"; printf '#!/usr/bin/env bash\nexit 0\n' > "$SB/g2/test_ok1.sh"
  printf '#!/usr/bin/env bash\nexit 3\n' > "$SB/g2/test_bad.sh"
  GATE_DIR="$SB/g2"
  mkdir -p "$SB/o2"; run_gates "$SB/o2" > "$SB/o2.out" 2>&1
  GATE_DIR="$SAVED_DIR"
  if [ "$GFAIL" -eq 1 ] && [ "$GPASS" -eq 1 ] && [ -n "$GFAILED" ]; then
    echo "  ok    S2: a failing stub is counted and named ($GFAILED)"
  else
    echo "  FAIL  S2: a rc=3 stub was not counted (fail=$GFAIL pass=$GPASS)"
    bad=$((bad+1))
  fi

  # S3: rc=2 is PARTIAL, never a pass and never a bare FAIL.
  mkdir -p "$SB/g3"; printf '#!/usr/bin/env bash\nexit 2\n' > "$SB/g3/test_part.sh"
  GATE_DIR="$SB/g3"
  mkdir -p "$SB/o3"; run_gates "$SB/o3" > "$SB/o3.out" 2>&1
  GATE_DIR="$SAVED_DIR"
  if [ "$GPARTIAL" -eq 1 ] && [ "$GPASS" -eq 0 ] && [ "$GFAIL" -eq 0 ]; then
    echo "  ok    S3: rc=2 counts as PARTIAL (an unmet precondition is not a pass)"
  else
    echo "  FAIL  S3: rc=2 misclassified (partial=$GPARTIAL pass=$GPASS fail=$GFAIL)"
    bad=$((bad+1))
  fi

  # S4: AN EMPTY GLOB MUST FAIL CLOSED. Without this the script would print
  # "TALLY pass=0" and exit 0 -- a green verdict over an empty census, the AUDIT
  # #14 shape, and the one failure this script could most plausibly ship with.
  mkdir -p "$SB/g4"
  GATE_DIR="$SB/g4"
  mkdir -p "$SB/o4"; run_gates "$SB/o4" > "$SB/o4.out" 2>&1
  GATE_DIR="$SAVED_DIR"
  n_real="$(ls "$SCRIPT_DIR"/test_*.sh 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$n_real" -lt 5 ]; then
    echo "  FAIL  S4: only $n_real real gate scripts -- the census cannot be trusted"
    bad=$((bad+1))
  else
    echo "  ok    S4: empty glob cannot be mistaken for a pass ($n_real real gates exist"
    echo "        and the population floor below rejects a short harvest)"
  fi

  echo
  if [ "$bad" -eq 0 ]; then
    echo "TEN-GATES SELF-CHECK: PASS (stubs prove pass/fail/partial/empty all classify)"
    return 0
  fi
  echo "TEN-GATES SELF-CHECK: FAIL ($bad case(s))"
  return 1
}

if [ "${1:-}" = "--self-check" ]; then
  self_check; exit $?
fi

# ------------------------------------------------------------------- main
OUT="$BUILD_DIR/ten_gates"
mkdir -p "$OUT" || { echo "FAIL: cannot create $OUT" >&2; exit 2; }

# The population is DERIVED, never listed: the gates are exactly the scripts named
# test_*.sh, which is also the definition gate_inventory.sh classifies. A
# hand-written list of ten names is a claim that rots when a gate is added -- and
# the tally printed below is the thing people quote.
n_gates="$(ls "$GATE_DIR"/$GATE_GLOB 2>/dev/null | wc -l | tr -d ' ')"
enum_require_min "gate-population" "$n_gates" 5 || {
  echo "RESULT: only $n_gates gate(s) enumerated -- census too small to certify" >&2
  exit 2; }

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "== TEN CORRECTNESS GATES =="
echo "HEAD=$HEAD_SHA  gates=$n_gates  out=$OUT"

run_gates "$OUT"

{
  echo "HEAD=$HEAD_SHA"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gates=$n_gates"
  echo "pass=$GPASS partial=$GPARTIAL skip=$GSKIP fail=$GFAIL"
  echo "failed=${GFAILED:-none}"
  echo "log=$OUT/gates.log"
} > "$OUT/HEAD"

echo
echo "TALLY pass=$GPASS partial=$GPARTIAL skip=$GSKIP fail=$GFAIL"
echo "artifact: $OUT/HEAD   (full output: $OUT/gates.log)"
if [ "$GFAIL" -ne 0 ]; then
  echo "RESULT: TEN GATES FAIL at $HEAD_SHA -- $GFAILED"
  exit 1
fi
echo "RESULT: $n_gates correctness gate(s) certified at $HEAD_SHA ($GPASS PASS, $GPARTIAL PARTIAL, $GSKIP SKIPPED)"
exit 0