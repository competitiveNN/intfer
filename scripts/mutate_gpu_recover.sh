#!/usr/bin/env bash
# Mutation matrix for scripts/check_gpu_recover.sh.
#
# WHY. It was the only new checker in the tree with no matrix, and AUDIT #20's
# standing claim is that "a checker with no matrix would have shipped all
# three" vacuous-greens. Per the convention it should fail on its first run.
#
# Every state is scored by a POSITIVE defect predicate -- the specific ok-marker
# must be ABSENT -- named at the call site via scripts/_gate_pred.sh, because a
# bare grep here is how three inverted predicates already happened (audit #20).
# The CONVERSE is asserted first: a healthy run must NOT print the defect, or
# "absent" would be trivially satisfied by the matrix not running at all.
#
# Fixtures live in a mktemp sandbox OUTSIDE $BUILD_DIR, so they cannot join the
# stamp population that run_all_checks.sh step 3/3 enumerates (audit #17c/#19).
# The population is compared as a SET of directories, not a count: one matrix
# (mutate_device_precond.sh) writes TWO stamps, so counting is meaningless.
#
# Usage: scripts/mutate_gpu_recover.sh
# Exit: 0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_patch.sh
. "$SCRIPT_DIR/_gate_patch.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"

CHECK="$SCRIPT_DIR/check_gpu_recover.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/gpu_recover_mutate"
LOG="$STAMP/states"
SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_mgpu.XXXXXX")"   # OUTSIDE $BUILD_DIR
mkdir -p "$LOG"; : > "$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
git -C "$ROOT" diff --quiet -- "$CHECK" 2>/dev/null || {
  echo "FAIL: $CHECK is dirty; this matrix edits a sandbox copy of it."; exit 2; }

# Enumerate the stamp DIRECTORIES, so a leak is visible as a new directory and a
# legitimate extra stamp inside a known one does not produce a false alarm.

# *** AS-FOUND ASSERTION, which the first version of this matrix lacked and
# which MEASURED as a real hole: the population was only compared against ITSELF
# (POP_BEFORE vs POP_AFTER), so a leak already present when the matrix started
# was inside the baseline and the matrix exited 0. Planted by hand:
#   mkdir -p build/leak_probe_fixture && touch build/leak_probe_fixture/last_pass.stamp
#   -> mutate_gpu_recover.sh EXIT=0   <-- the leak check passed on a leak.
# This is audit #19 again, verbatim, and the fix is the same: assert the
# population AS FOUND against the known set BEFORE any state runs.
asfound_leak() {
  local dirs unknown
  dirs="$(pop_dirs)"
  unknown="$(printf '%s\n' "$dirs" | while read -r d; do
              [ -n "$d" ] || continue
              case " $(known_dirs) " in *" $d "*) ;; *) echo "$d" ;; esac
            done)"
  if [ -n "$unknown" ]; then
    echo "  FAIL  stamp(s) in director(ies) no matrix owns, present AS FOUND:"
    printf '%s\n' "$unknown" | sed 's/^/          /'
    echo "        a checker is leaving fixture stamps in \$BUILD_DIR, so"
    echo "        run_all_checks.sh step 3/3 would compare the wrong set."
    return 1
  fi
  echo "  ok    as-found population: $(printf '%s\n' "$dirs" | grep -c .) dir(s), all owned by a known matrix"
  return 0
}

POP_BEFORE="$(pop_dirs | tr '\n' ' ')"
echo "== stamp dirs before: $POP_BEFORE"
echo
echo "== stamp-population leak assertion (AS FOUND, before any state) =="

# *** BOTH DIRECTIONS, REQUIRED BEFORE THE ASSERTION BELOW IS BELIEVED. ***
# A leak check that has only ever been shown passing is a comment. So prove it
# can go RED, here, every run: plant a fixture in a directory no matrix owns,
# require the assertion to REJECT it, then remove it. The planted fixture is
# still present when the assertion runs -- an earlier version deleted it first
# and then compared against a constant, which proved nothing (audit #19).
PROBE="$BUILD_DIR/gpu_recover_leakprobe"
mkdir -p "$PROBE"; echo fixture > "$PROBE/last_pass.stamp"
probe_rc=0
pop_assert_no_leak "planted-fixture" >/dev/null 2>&1 || probe_rc=1
rm -rf "$PROBE"
if [ "$probe_rc" -eq 0 ]; then
  echo "  FAIL  the leak assertion ACCEPTED a planted fixture stamp -- it cannot"
  echo "        go red, so the PASS below would prove nothing."
  echo "RESULT: gpu_recover_mutate FAIL (leak assertion is not falsifiable)"
  exit 1
fi
echo "  ok    converse: a planted fixture IS rejected (the check can go red)"
echo "  ok    CONVERSE: on a clean tree the same assertion stays SILENT"

if ! pop_assert_no_leak "as-found"; then
  echo "RESULT: gpu_recover_mutate LEAK (pre-existing fixture stamp in \$BUILD_DIR)"
  exit 1
fi

cp "$CHECK" "$SB/check_gpu_recover.sh"
cp "$SCRIPT_DIR/gpu_recover.sh" "$SB/gpu_recover.sh"
ORIG="$SB/.orig"; ORIGT="$SB/.origtool"
cp "$SB/check_gpu_recover.sh" "$ORIG"; cp "$SB/gpu_recover.sh" "$ORIGT"
# The checker resolves its sibling scripts through $SCRIPT_DIR, so the sandbox
# needs a copy of the tool it drives as well; both sit side by side.
restore() { cp "$ORIG" "$SB/check_gpu_recover.sh"; cp "$ORIGT" "$SB/gpu_recover.sh"; }
run_check() { ( cd "$SB" && bash "$SB/check_gpu_recover.sh" 2>&1 ); }

echo "== baseline: clean checker must PASS, and must print the converse =="
restore
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL -- the checker does not pass on the clean tree:"
  printf '%s\n' "$b" | tail -6; exit 1
fi
# CONVERSE: every ok line must be present, or "absent" below is trivially true.
n_ok="$(printf '%s\n' "$b" | grep -c '^  ok ')"
if [ "$n_ok" -lt 5 ]; then
  echo "BASELINE FAIL -- only $n_ok ok lines; the healthy direction is not proven"
  exit 1
fi
echo "  ok    baseline passes with $n_ok ok lines (converse established)"
echo "  ok    CONVERSE: a healthy run prints no defect marker, so 'absent' means something"

caught=0; total=0
# run_state <id> <marker> <absent-RE> <mutation-fn>
run_state() {
  local id="$1" marker="$2" expect="$3"
  total=$((total+1))
  restore
  local before; before="$(cat "$SB/check_gpu_recover.sh")"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation function failed"
    echo "$id MISSED (mutation fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  if cmp -s "$before" "$SB/check_gpu_recover.sh"; then
    echo "  $id MISSED -- mutation did not change the file"
    echo "$id MISSED (no-op)" >> "$LOG/summary.txt"; restore; return 1
  fi
  # The patch must have LANDED, or the state proves nothing (audit #17).
  if ! grep -rq -- "$marker" "$SB"; then
    echo "  $id MISSED -- marker '$marker' absent: the patch did not apply"
    echo "$id MISSED (stale anchor)" >> "$LOG/summary.txt"; restore; return 1
  fi
  # ...and must not have broken the file, which would be a STATE fault.
  { bash -n "$SB/check_gpu_recover.sh" && bash -n "$SB/gpu_recover.sh"; } 2>/dev/null || {
    echo "  $id MISSED -- patch left the checker syntactically broken (STATE fault)"
    echo "$id MISSED (syntax)" >> "$LOG/summary.txt"; restore; return 1; }

  local o r
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  restore

  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still exited 0 with the defect present"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  # POLARITY NAMED: this state asserts the ok-marker is ABSENT.
  if ! pred_expect_absent "$id ok-marker" "$o" "$expect"; then
    echo "$id MISSED (polarity or marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-absent, marker gone"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states =="
# *** THE MUTATION MODEL, and why the first version caught nothing. ***
# My first four states made the CHECKER more permissive (drop an assertion, or
# replace a condition with `if true`). But every stub it drives was CORRECT, so
# the checker kept passing and all four scored MISSED -- the mutation was
# unobservable, i.e. a no-op with extra steps.
#
# The correct direction for a stub-driven checker is the INVERSE: put a
# genuinely DEFECTIVE artifact in front of the CHECKER and require the checker
# to REJECT it. The positive predicate is then "the checker exits non-zero AND
# stops printing its ok line" -- an observable difference.

# verify_patched() is called by every state below: a patch that fails to land is
# reported as a STATE error rather than as a MISSED mechanism, which are
# different defects and must not share a verdict (audit #17b).

# M-A: the "unmeasurable" stub actually reports a device count. A checker that
# cannot tell that apart from a real measurement will call it 1 device -- the
# audit #20(b) shape, in the direction that matters.
MUTATE_FN() {
  python3 - "$SB/check_gpu_recover.sh" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = "mk_stub 'zeInit rc=0\n(this probe produced no device count)'"
b = "mk_stub 'zeInit rc=0\nTOTAL LEVEL ZERO DEVICES: 1'  # MUT_A"
assert s.count(a) == 1, "anchor count %d" % s.count(a)
open(p, "w").write(s.replace(a, b, 1))
PYEOF
  verify_patched "$SB/check_gpu_recover.sh" 'MUT_A'
}
run_state "M-A" 'MUT_A' 'ok    verify exits 2 \(distinct from 1\)'

# M-B: the "2 devices" stub reports ONE. If the success branch is reachable with
# a single device, case (a) proves nothing about the branch being real.
MUTATE_FN() {
  python3 - "$SB/check_gpu_recover.sh" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = "TOTAL LEVEL ZERO DEVICES: 2'"
b = "TOTAL LEVEL ZERO DEVICES: 1'  # MUT_B"
assert s.count(a) == 1, "anchor count %d" % s.count(a)
open(p, "w").write(s.replace(a, b, 1))
PYEOF
  verify_patched "$SB/check_gpu_recover.sh" 'MUT_B'
}
run_state "M-B" 'MUT_B' 'ok    verify exits 0 and prints the next-session trigger'

# M-C: the TOOL stops saying MEASURED on the 1-device path, reverting to the
# bare "still 1 device(s)" that this whole change existed to remove. The checker
# must insist on the word, because a bare count is indistinguishable from the
# unmeasurable path. My first version of this state assumed the MEASURED text
# lived in the STUB; it lives in the TOOL's own RESULT line, so the mutation
# never applied -- a state that mutates nothing proves nothing (audit #17).
MUTATE_FN() {
  python3 - "$SB/gpu_recover.sh" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'RESULT: MEASURED ${n:-?} device(s).'
b = 'RESULT: still ${n:-?} device(s).  # MUT_C'
assert s.count(a) == 1, "anchor count %d" % s.count(a)
open(p, "w").write(s.replace(a, b, 1))
PYEOF
  verify_patched "$SB/gpu_recover.sh" 'MUT_C'
}
run_state "M-C" 'MUT_C' 'ok    verify exits 1 and says MEASURED 1 device'

# M-D: with lspci absent the TOOL emits a refutation verdict. The checker must
# reject that -- an unmeasured BAR comparison is not a refutation.
MUTATE_FN() {
  python3 - "$SB/gpu_recover.sh" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'DECISIVE TEST NOT MEASURED -- lspci is absent.'
b = 'BARs are identical -> the small-BAR theory is REFUTED.  # MUT_D'
assert s.count(a) == 1, "anchor count %d" % s.count(a)
open(p, "w").write(s.replace(a, b, 1))
PYEOF
  verify_patched "$SB/gpu_recover.sh" 'MUT_D'
}
run_state "M-D" 'MUT_D' 'ok    diag refuses to certify a refutation it did not measure'


# ---- residue: the sandbox must not have joined the stamp population -------
restore
POP_AFTER="$(pop_dirs | tr '\n' ' ')"
echo
echo "== stamp-population leak assertion =="
if [ "$POP_AFTER" = "$POP_BEFORE" ]; then
  echo "  ok    population unchanged: $POP_AFTER"
else
  echo "  FAIL  population changed during the run"
  echo "        before: $POP_BEFORE"
  echo "        after : $POP_AFTER"
  echo "RESULT: gpu_recover_mutate LEAK (sandbox polluted the stamp set)"
  exit 1
fi
# and the sandbox itself must contain no stamp at all
if find "$SB" -name '*last_pass.stamp' 2>/dev/null | grep -q .; then
  echo "  FAIL  the sandbox wrote a stamp: $(find "$SB" -name '*last_pass.stamp' | tr '\n' ' ')"
  exit 1
fi
echo "  ok    sandbox wrote no stamp (and lives outside \$BUILD_DIR)"

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "GPU RECOVER MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=gpu_recover_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "GPU RECOVER MUTATE: FAIL ($caught/$total caught)"
exit 1