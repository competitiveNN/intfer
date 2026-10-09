#!/usr/bin/env bash
# Mutation matrix for scripts/test_roofline_bisect4.sh.
#
# WHY. That gate is not a checker of source code -- it is a checker of a BINARY's
# exit semantics, and the semantics it exists to enforce were themselves the
# defect: its first version ran the probe under `timeout 1800`, the same budget
# its runner allows, and returned **rc=124 with EMPTY OUTPUT**. That is worse than
# no gate:
#
#   * rc=124 is NOT in the runner's vocabulary. check_device_precond.sh's
#     expect_healthy reads healthy rc as 0, or 2 WITH a skip diagnostic; anything
#     else is "a gate must not go red on a healthy box" -> a REGRESSION. So an
#     unrunnable probe masqueraded as a code failure.
#   * empty output meant even the `need` assertions had nothing to match, so there
#     was no diagnostic to act on;
#   * and it consumed the runner's whole budget on a step that cannot finish.
#
# A self-validating gate is not evidence. These states break the gate's OWN
# safeguards and require the breakage to be REJECTED, because the failure mode of a
# guard that was never itself tested is precisely the one it exists to catch.
#
# States:
#   RB4-1  the probe budget is raised to the runner's own  [must REFUSE]
#   RB4-2  a `need` for an arm is deleted                  [must REFUSE]
#   RB4-3  a `need` is renamed to a wrong arm              [must REFUSE]
#   RB4-4  pristine                                        [must hold]
#
# NON-DESTRUCTIVE: every state runs a COPY of the gate under build/, against a
# STUB probe, so no real binary is launched and the tracked tree is never edited.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/roofline_bisect4_matrix"
GATE="scripts/test_roofline_bisect4.sh"

rm -rf "$SB"; mkdir -p "$SB/bin" || exit 2
GATE_PRISTINE="$SB/gate.pristine"
cp "$GATE" "$GATE_PRISTINE" || exit 2

# A STUB probe: sleeps past the gate's own budget, which is exactly the hang the
# real binary shows. Launching the real one here would take minutes per state and
# would put a real GPU job in a mutation matrix.
cat >"$SB/bin/roofline_probe_bisect4" <<'STUB'
#!/usr/bin/env bash
# Stands in for a probe that hangs. Prints nothing, like the real one did.
sleep 3600
STUB
chmod +x "$SB/bin/roofline_probe_bisect4"

# A SECOND stub: the vacuous-but-green run this box actually produces. It returns
# rc=0, prints RESULT: PASS, and prints every arm at 0.000 ms -- the exact
# output the old gate certified with PASS. RB4-5 needs this one, because a
# hanging probe exits UNPROVEN before any `need` is reached and so cannot test
# the arm list at all.
cat >"$SB/bin/roofline_probe_bisect4_vacuous" <<'STUB'
#!/usr/bin/env bash
cat <<EOF
[bisect-bisect4] device 0: Intel(R) Arc(TM) B580 Graphics  (max_compute_units=160)
[bisect-bisect4] ceiling    0.000 ms      inf GB/s     inf% of peak
[bisect-bisect4] twin        0.000 ms  ratio to full = -nan (linear would be 0.25)
[bisect-bisect4] -- MLP_GATE(INT2) (N_local=8704 K=5120 g16) --
[bisect-bisect4]   MLP_GATE(INT2)/VERBATIM       0.000 ms      inf GB/s   1.00x vs VERBATIM
[bisect-bisect4]   MLP_GATE(INT2)/VERB+bitcast  0.000 ms      inf GB/s   1.11x vs VERBATIM
[bisect-bisect4]   MLP_GATE(INT2)/VERB+bitcast+4 0.000 ms     inf GB/s   1.11x vs VERBATIM
[bisect-bisect4]   MLP_GATE(INT2)/VERB+bitcast+4+vsel 0.000 ms inf GB/s 1.11x vs VERBATIM
[bisect-bisect4]   MLP_GATE(INT2)/CLONE         0.000 ms      inf GB/s   1.12x vs VERBATIM
== WHAT THIS RUN DOES AND DOES NOT ESTABLISH ==
RESULT: PASS
EOF
exit 0
STUB
chmod +x "$SB/bin/roofline_probe_bisect4_vacuous"

restore_gate() { cp "$GATE_PRISTINE" "$SB/gate_under_test.sh"; }
# The gate resolves SCRIPT_DIR from its own location, so it must live beside a
# scripts/ dir containing with_oneapi.sh and _gate_build.sh for the stub path.
setup_tree() {
  rm -rf "$SB/root"; mkdir -p "$SB/root/scripts" \
    "$SB/root/build/examples/roofline_probe_bisect4"
  cp "$GATE_PRISTINE" "$SB/root/scripts/test_roofline_bisect4.sh"
  cp "$SCRIPT_DIR/with_oneapi.sh" "$SCRIPT_DIR/_gate_build.sh" \
     "$SB/root/scripts/" 2>/dev/null || true
  # The gate resolves BIN as $BUILD_DIR/examples/... with BUILD_DIR defaulting to
  # $ROOT/build, and `[ -x "$BIN" ]` runs BEFORE the no-build seam -- so the stub has
  # to sit exactly where the real binary would. The first version put it at the tree
  # root and every state died on "binary missing", which reads as a gate failure.
  cp "$SB/bin/roofline_probe_bisect4" \
     "$SB/root/build/examples/roofline_probe_bisect4/roofline_probe_bisect4"
}
# Same as setup_tree, but the probe under test is the VACUOUS stub (rc=0,
# RESULT: PASS, all arms 0.000 ms) rather than the hanging one.
setup_vacuous_tree() {
  setup_tree
  cp "$SB/bin/roofline_probe_bisect4_vacuous" \
     "$SB/root/build/examples/roofline_probe_bisect4/roofline_probe_bisect4"
}
patch_gate() { python3 - "$SB/root/scripts/test_roofline_bisect4.sh" "$1" "$2" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
c = s.count(o)
if c != 1:
    print("ANCHOR %s: %d matches (want 1): %r"
          % ("ABSENT" if c == 0 else "AMBIGUOUS", c, o[:60]), file=sys.stderr)
    sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
}
run_gate() { # $1 = OPTIONAL extra env assignment
  local t="$SB/out.$$"
  # INTFER_GATE_NO_BUILD keeps the gate from demanding the real binary; a short
  # INTFER_BISECT_TIMEOUT keeps each state to seconds rather than minutes. NOTE
  # the env var name is BISECT_TIMEOUT, not BISECT4_TIMEOUT -- the gate reads
  # `INTFER_BISECT_TIMEOUT` and a 4-suffixed name is silently ignored, which is
  # how the first version of this matrix hung for the full 120 s per state and
  # timed out the matrix itself.
  if [ -n "${1:-}" ]; then
    ( cd "$SB/root" && env INTFER_GATE_NO_BUILD=1 INTFER_BISECT_TIMEOUT=3 "$1" \
        bash scripts/test_roofline_bisect4.sh ) >"$t" 2>&1
  else
    ( cd "$SB/root" && env INTFER_GATE_NO_BUILD=1 INTFER_BISECT_TIMEOUT=3 \
        bash scripts/test_roofline_bisect4.sh ) >"$t" 2>&1
  fi
  local rc=$?
  GATE_OUT="$(cat "$t")"; rm -f "$t"
  return $rc
}

total=0; caught=0; RESULTS=()
record() { case "$2" in ok) caught=$((caught+1)); RESULTS+=("$1 CAUGHT ''");;
                       *)  RESULTS+=("$1 $2");; esac; }
verdict() { # id want-expect-re out rc
  local id="$1" expect="$2" out="$3" rc="$4"
  if [ "$rc" -eq 0 ]; then
    record "$id" "MISSED '(the gate returned 0 -- a hang now looks like a PASS)'"
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    record "$id" "MISSED '(non-zero, but no expected diagnostic: $expect)'"
  else record "$id" ok; fi
}
state() { # id expect-re old new
  total=$((total+1)); setup_tree
  if patch_gate "$3" "$4"; then
    run_gate; verdict "$1" "$2" "$GATE_OUT" $?
  else
    RESULTS+=("$1 ANCHOR-DRIFT (state not run)"); fi
  restore_gate
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  RB4-1  probe budget raised to runner's own     [must REFUSE]"
  echo "  RB4-2  a \`need\` for an arm deleted             [recorded; see note]"
  echo "  RB4-3  a \`need\` renamed to wrong arm          [recorded; see note]"
  echo "  RB4-4  pristine                                 [must hold]"
  echo "  RB4-5  a vacuous-but-green probe stub            [must REFUSE]"
  exit 0
fi

echo "== baseline (pristine gate vs a STUB probe that hangs) =="
setup_tree; restore_gate
cp "$GATE_PRISTINE" "$SB/root/scripts/test_roofline_bisect4.sh"
run_gate; rc=$?
printf '%s\n' "$GATE_OUT" | tail -6 | sed 's/^/  /'
if [ "$rc" -ne 2 ]; then
  echo "  FAIL baseline: the pristine gate returned rc=$rc against a hanging probe," >&2
  echo "        and must return 2 (UNPROVEN). Every state below would be scored" >&2
  echo "        against a gate that does not hold its own contract." >&2
  exit 1
fi
printf '%s' "$GATE_OUT" | grep -q 'UNPROVEN' || {
  echo "  FAIL baseline: rc=2 but no UNPROVEN diagnostic" >&2; exit 1; }
total=$((total+1)); caught=$((caught+1)); RESULTS+=("RB4-4 CAUGHT ''")
echo "  ok baseline rc=2 with an UNPROVEN diagnostic -- the gate holds its contract"

echo
echo "== states =="

# RB4-1: the self-bound is the whole point of the fix. Raise the probe budget to the
# runner's own (below the timeout used in this matrix) and the gate must REFUSE to
# start, because then a hang costs the runner everything and surfaces as rc=124 -> a
# phantom regression. Lowering the BUDGET to 1 while the harness runs at timeout=3
# changes nothing, so the first version of this state was a no-op that "passed" for
# the wrong reason: the gate simply took the normal timeout path. The budget must go
# BELOW the timeout in use.
state RB4-1 'not under the runner' \
  'RUNNER_BUDGET=1800' 'RUNNER_BUDGET=1'

# RB4-2/RB4-3: mutate a `need` assertion. On its own that makes the gate WEAKER,
# not broken — but there is a subtlety: while the probe hangs, the gate exits
# UNPROVEN *before* any `need()` is reached, so these two states observe the
# honest finding that the gate is honest because of its TIMEOUT verdict, not
# because it checked anything. They are recorded with that reason, like RB3-5/6,
# rather than dressed up as "the arm list is load-bearing". A real arm-load-bearing
# check would need a stub probe that RETURNS a defined rc — which conflicts with
# the stub hanging (the failure the gate exists to report).
state RB4-2 'UNPROVEN' \
  "need 'VERB\\+bitcast\\+4'      \"the 4-byte code-load arm\"" \
  "true"

state RB4-3 'UNPROVEN' \
  "need 'VERB\\+bitcast\\+4'      \"the 4-byte code-load arm\"" \
  "need 'NOCLONE'                 \"the 4-byte code-load arm\""

# RB4-5: the vacuity floor. The whole reason this matrix exists beyond RB4-1 is a
# probe that RETURNS rc=0 and prints RESULT: PASS while measuring nothing. The
# old gate certified that with PASS, because `need` (string existence) and
# `RESULT: PASS` both match vacuous output. So this state replaces the hanging
# stub with one that mimics the real box's vacuous run: ceiling 0.000 ms, every
# arm 0.000 ms, RESULT: PASS, exit 0. The gate must REFUSE it -- if it does not,
# the default-ON promotion of INTFER_DECODE1_VERBHALF is resting on a green
# light that a probe can earn without measuring.
total=$((total+1))
setup_vacuous_tree
cp "$GATE_PRISTINE" "$SB/root/scripts/test_roofline_bisect4.sh"
run_gate
verdict RB4-5 'ZERO VERBATIM median' "$GATE_OUT" $?
restore_gate

echo
echo "== results ($caught / $total states pass) =="
for r in "${RESULTS[@]}"; do echo "  $r"; done
if [ "$caught" -ne "$total" ]; then
  echo "FAIL: $total states, $caught passed" >&2
  exit 1
fi
echo
echo "ROOFLINE-BISECT4 MUTATION MATRIX: PASS ($caught/$total states)"
echo "  RB4-1: the self-bound FIRES. A probe budget at or above the runner's is"
echo "        REFUSED, so a hang can never again consume the runner's whole step or"
echo "        surface as rc=124, which the runner reads as a REGRESSION."
echo "  RB4-2/RB4-3: recorded, not dressed up. While the probe hangs, the gate"
echo "        exits UNPROVEN *before* any \`need\`() is reached \u2014 so these two states"
echo "        prove the gate is honest because of its TIMEOUT verdict and"
echo "        diagnostics, not because it checked the arm list. The arm list is"
echo "        load-bearing only in a run where the probe actually runs; RB4-1"
echo "        guarantees that run can never be left hanging unreported."
echo "  RB4-4: the pristine gate holds its contract (rc=2 + UNPROVEN) against a"
echo "        hanging probe."
echo "RESULT: PASS ($caught/$total states caught)"
exit 0
