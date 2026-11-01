#!/usr/bin/env bash
# Mutation matrix for scripts/gates/test_roofline_bisect1.sh.
#
# WHY. That gate is not a checker of source code -- it is a checker of a BINARY's
# exit semantics, and the semantics it exists to enforce were themselves the defect:
# its first version ran the probe under `timeout 1800`, the same budget its runner
# allows, and returned **rc=124 with EMPTY OUTPUT**. That is worse than no gate:
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
#   RB1-1  the probe budget is raised to the runner's own budget  [must REFUSE]
#   RB1-2  the timeout verdict is changed 124 -> 0 (silent pass)   [EXCLUDED]
#   RB1-3  the SKIP diagnostic is removed, rc=2 kept             [EXCLUDED]
#   RB1-4  the self-bound check is removed entirely               [DROPPED - see below]
#   RB1-5  a `need` assertion for an arm is deleted               [must be caught by
#                                                                the mutation of RB1-6]
#   RB1-6  a `need` is renamed to the wrong arm                   [must REFUSE]
#   RB1-7  pristine                                             [must hold]
#
# NON-DESTRUCTIVE: every state runs a COPY of the gate under build/, against a
# STUB probe, so no real binary is launched and the tracked tree is never edited.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/roofline_bisect1_matrix"
# *** WIPE LOCK, OUTSIDE THE TREE IT WIPES (check_matrix_sandbox_locks.sh). ***
# The sandbox below is a FIXED path wiped on every invocation, so two concurrent
# runs share one tree: the second's wipe empties the first's mid-state and the
# first then grades a corpus the other has just rebuilt (MEASURED 2026-10-21 on
# mutate_durable_reads.sh). Taken BEFORE the first wipe and released on
# EXIT/INT/TERM, so a run that dies mid-state does not brick the next.
LOCK="$BUILD/.mutate_roofline_bisect1.lock"
mkdir -p "$BUILD" 2>/dev/null || true
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_roofline_bisect1.sh is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

GATE="scripts/gates/test_roofline_bisect1.sh"

rm -rf "$SB"; mkdir -p "$SB/bin" || exit 2
GATE_PRISTINE="$SB/gate.pristine"
cp "$GATE" "$GATE_PRISTINE" || exit 2

# A STUB probe: sleeps past the gate's own budget, which is exactly the hang the
# real binary shows. Launching the real one here would take minutes per state and
# would put a real GPU job in a mutation matrix.
cat >"$SB/bin/roofline_probe_bisect1" <<'STUB'
#!/usr/bin/env bash
# Stands in for a probe that hangs. Prints nothing, like the real one did.
sleep 3600
STUB
chmod +x "$SB/bin/roofline_probe_bisect1"

# A THIRD stub: a REAL sub-millisecond run. The box now measures for real
# (device timestamps live again 2026-10-12), with VERBATIM medians of
# 0.249-0.622 ms, and the gate's floor was refusing every one of them. RB1-9
# needs this stub: it is the state that proves the floor is a FLOOR and not an
# always-red predicate. The numbers are the real probe's own output, captured
# from build/run_all_test_roofline_bisect1_sh.log (the probe section, first
# `[bisect-...]` line to `RESULT: PASS`).
cat >"$SB/bin/roofline_probe_bisect1_real" <<'STUB'
#!/usr/bin/env bash
cat <<EOF
[bisect-bisect1] device 0: Intel(R) Arc(TM) B580 Graphics  (max_compute_units=160)
[bisect-bisect1] ceiling    0.211 ms    248.3 GB/s    54.5% of peak CITATION(techpowerup arc-b580.c4244 / ARK 241598: 456.0 GB/s, 20 Xe cores)
[bisect-bisect1] twin        0.050 ms  ratio to full = 0.24 (linear would be 0.25)
[bisect-bisect1] -- MLP_GATE(INT2) (N_local=8704 K=5120 g16) -- CITATION(bisect-%s: every clone shares this buffer with every other clone, interleaved per rep)
[bisect-bisect1]   MLP_GATE(INT2)/table        0.385 ms     28.9 GB/s   1.00x vs VERBATIM
[bisect-bisect1]   MLP_GATE(INT2)/VERBATIM     0.340 ms     32.8 GB/s   1.13x vs VERBATIM
[bisect-bisect1] -- MLP_DOWN(INT2) (N_local=2560 K=17408 g16) -- CITATION(bisect-%s: every clone shares this buffer with every other clone, interleaved per rep)
[bisect-bisect1]   MLP_DOWN(INT2)/table        0.622 ms     17.9 GB/s   1.00x vs VERBATIM
[bisect-bisect1]   MLP_DOWN(INT2)/VERBATIM     0.538 ms     20.7 GB/s   1.16x vs VERBATIM

== WHAT THIS RUN DOES AND DOES NOT ESTABLISH ==
  ESTABLISHED by examples/roofline_probe (THE OTHER binary):
    the shipped INT2 kernel runs at 30-41 GB/s on MLP_GATE/MLP_DOWN
    (device event timestamps, medians over 12 interleaved reps).
  ESTABLISHED BY THIS RUN: whatever the columns below say, all measured
    against the VERBATIM control on the SAME buffers, interleaved per rep;
    ratios are within-run vs VERBATIM (index 0).
  NOT ESTABLISHED: a mechanism, unless one of the single-change arms
    closes most of the table-vs-VERBATIM gap. SHIPPED is anchored from
    roofline_probe (cross-binary).

RESULT: PASS
EOF
exit 0
STUB
chmod +x "$SB/bin/roofline_probe_bisect1_real"
# rc=0, prints RESULT: PASS, and prints every arm at 0.000 ms -- the exact
# output the old gate certified with PASS. RB1-8 needs this one, because a hanging
# probe exits UNPROVEN before any `need` is reached and so cannot test the arm
# list or the vacuity floor.
cat >"$SB/bin/roofline_probe_bisect1_vacuous" <<'STUB'
#!/usr/bin/env bash
cat <<EOF
[bisect-bisect1] device 0: Intel(R) Arc(TM) B580 Graphics  (max_compute_units=160)
[bisect-bisect1] ceiling    0.000 ms      inf GB/s     inf% of peak
[bisect-bisect1] twin        0.000 ms  ratio to full = -nan (linear would be 0.25)
[bisect-bisect1] -- MLP_GATE(INT2) (N_local=8704 K=5120 g16) --
[bisect-bisect1]   MLP_GATE(INT2)/table       0.000 ms      inf GB/s   0.88x vs VERBATIM
[bisect-bisect1]   MLP_GATE(INT2)/VERBATIM    0.000 ms      inf GB/s   1.00x vs VERBATIM
== WHAT THIS RUN DOES AND DOES NOT ESTABLISH ==
RESULT: PASS
EOF
exit 0
STUB
chmod +x "$SB/bin/roofline_probe_bisect1_vacuous"

restore_gate() { cp "$GATE_PRISTINE" "$SB/gate_under_test.sh"; }
# The gate resolves SCRIPT_DIR from its own location, so it must live beside a
# scripts/ dir containing with_oneapi.sh and _gate_build.sh for the stub path.
setup_tree() {
  rm -rf "$SB/root"; mkdir -p "$SB/root/scripts" \
    "$SB/root/build/examples/roofline_probe_bisect1"
  cp "$GATE_PRISTINE" "$SB/root/scripts/gates/test_roofline_bisect1.sh"
  cp "\"$SCRIPT_DIR/../with_oneapi.sh" "$SCRIPT_DIR/../lib/_gate_build.sh" \
     "$SB/root/scripts/" 2>/dev/null || true
  # The gate resolves BIN as $BUILD_DIR/examples/... with BUILD_DIR defaulting to
  # $ROOT/build, and `[ -x "$BIN" ]` runs BEFORE the no-build seam -- so the stub
  # has to sit exactly where the real binary would. The first version put it at the
  # tree root and every state died on "binary missing", which reads as a gate
  # failure.
  cp "$SB/bin/roofline_probe_bisect1" \
     "$SB/root/build/examples/roofline_probe_bisect1/roofline_probe_bisect1"
}
# Same as setup_tree, but the probe under test is the VACUOUS stub (rc=0,
# RESULT: PASS, all arms 0.000 ms) rather than the hanging one.
setup_vacuous_tree() {
  setup_tree
  cp "$SB/bin/roofline_probe_bisect1_vacuous" \
     "$SB/root/build/examples/roofline_probe_bisect1/roofline_probe_bisect1"
}
# Same again, but the probe under test is a REAL sub-millisecond run (the
# numbers the box actually measures). The gate must ACCEPT this one -- a floor
# that cannot be satisfied by a real measurement is an always-red twin, not a
# guard. See RB1-9.
setup_real_tree() {
  setup_tree
  cp "$SB/bin/roofline_probe_bisect1_real" \
     "$SB/root/build/examples/roofline_probe_bisect1/roofline_probe_bisect1"
}
patch_gate() { python3 - "$SB/root/scripts/gates/test_roofline_bisect1.sh" "$1" "$2" <<'PYX'
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
run_gate() { # $1 = OPTIONAL extra env assignment, e.g. INTFER_BISECT1_TIMEOUT=1800
  local t="$SB/out.$$"
  # `env VAR=v cmd` with an EMPTY argument fails outright ("env: '': No such file
  # or directory", rc=127) -- which the first version of this matrix did, and which
  # looked like "the gate does not hold its contract" rather than "the harness
  # passed an empty argument". Only pass the extra assignment when there is one.
  # INTFER_GATE_NO_BUILD keeps the gate from demanding the real binary; a short
  # INTFER_BISECT1_TIMEOUT keeps each state to seconds rather than minutes.
  if [ -n "${1:-}" ]; then
    ( cd "$SB/root" && env INTFER_GATE_NO_BUILD=1 INTFER_BISECT1_TIMEOUT=3 "$1" \
        bash scripts/gates/test_roofline_bisect1.sh ) >"$t" 2>&1
  else
    ( cd "$SB/root" && env INTFER_GATE_NO_BUILD=1 INTFER_BISECT1_TIMEOUT=3 \
        bash scripts/gates/test_roofline_bisect1.sh ) >"$t" 2>&1
  fi
  local rc=$?
  GATE_OUT="$(cat "$t")"; rm -f "$t"
  return $rc
}

total=0; caught=0; RESULTS=()
record() { case "$2" in ok) caught=$((caught+1)); RESULTS+=("$1 CAUGHT ''");;
                      *)  RESULTS+=("$1 $2");; esac; }
verdict() { # id want-re expect-re out rc
  local id="$1" expect="$2" out="$3" rc="$4"
  # The states here assert a NEGATIVE: the gate must NOT look like a clean pass.
  # "Looks like a pass" = rc 0, or rc 2 with no diagnostic.
  if [ "$rc" -eq 0 ]; then
    record "$id" "MISSED '(the gate returned 0 -- a hang now looks like a PASS)'"
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    record "$id" "MISSED '(non-zero, but no expected diagnostic: $expect)'"
  else record "$id" ok; fi
}
pass_verdict() { # id out rc -- the POSITIVE direction: the gate must ACCEPT
  local id="$1" out="$2" rc="$3"
  if [ "$rc" -eq 0 ]; then
    record "$id" ok
  else
    record "$id" "MISSED '(the gate refused a REAL sub-millisecond run: rc=$rc)'"
  fi
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
  echo "  RB1-1  probe budget raised to the runner's own      [must REFUSE]"
  echo "  RB1-2  timeout verdict 124 -> 0 (silent pass)     [EXCLUDED - see below]"
  echo "  RB1-3  SKIP diagnostic removed, rc=2 kept         [EXCLUDED - see below]"
  echo "  RB1-4  the self-bound check removed              [DROPPED - 1800 s per run]"
  echo "  RB1-5  a \`need\` for an arm deleted                [must REFUSE]"
  echo "  RB1-6  a \`need\` renamed to the wrong arm         [must REFUSE]"
  echo "  RB1-7  pristine                                   [must hold]"
  echo "  RB1-8  a vacuous-but-green probe stub            [must REFUSE]"
  exit 0
fi

echo "== baseline (pristine gate vs a STUB probe that hangs) =="
setup_tree; restore_gate
cp "$GATE_PRISTINE" "$SB/root/scripts/gates/test_roofline_bisect1.sh"
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
total=$((total+1)); caught=$((caught+1)); RESULTS+=("RB1-7 CAUGHT ''")
echo "  ok baseline rc=2 with an UNPROVEN diagnostic -- the gate holds its contract"

echo
echo "== states =="

# RB1-1: the self-bound is the whole point of the fix. Raise the probe budget to the
# runner's own and the gate must REFUSE to start, because then a hang costs the
# runner everything and surfaces as rc=124 -> a phantom regression.
# The bound fires when PROBE_TIMEOUT >= RUNNER_BUDGET. Lowering the BUDGET to 1
# while the harness runs at timeout=3 changes nothing, so the first version of this
# state was a no-op that "passed" for the wrong reason: the gate simply took the
# normal timeout path. The budget must go BELOW the timeout in use.
state RB1-1 'not under the runner' \
  'RUNNER_BUDGET=1800' 'RUNNER_BUDGET=1'

# RB1-2 and RB1-3 are EXCLUDED WITH A REASON, not counted. Both soften the DIAGNOSTIC
# TEXT (turn "UNPROVEN: ..." into a cheerful line; drop the SKIP block but keep
# rc=2). With the probe still hanging, neither changes the VERDICT -- it stays rc=2,
# which is exactly what a correct gate returns -- so a state asserting either
# direction is asserting something the harness cannot distinguish. That is the
# unobservable-state trap (AGENTS.md: a state that cannot fail is excluded with its
# reason, not counted), and forcing it would mean adding a seam to the gate purely
# to host the test.
#
# WHAT WOULD MAKE THEM OBSERVABLE, for whoever has the budget: run the mutation
# against a probe STUB THAT RETURNS A DEFINED rc rather than one that hangs. Then
# "rc=2 with no diagnostic" and "rc=0" become distinguishable, and these two states
# become real. The stub here deliberately hangs, because hanging is the failure the
# gate exists to report, so the two requirements conflict and the diagnostic-text
# states lose.
echo "  --    RB1-2, RB1-3 EXCLUDED WITH REASON: diagnostic-text softening is"
echo "        unobservable while the probe hangs (rc=2 either way)."

# RB1-4 (a second route to RB1-1's property) is DROPPED WITH A REASON. It removed the
# bound AND raised the probe timeout to the runner's own 1800 s, which is the exact
# configuration that produced the original rc=124 -- so observing it means WAITING
# 1800 s per matrix run, for a property RB1-1 already proves fires. A state that costs
# half an hour to tick a box the next state ticks in three seconds is not coverage.
# The guarantee that matters -- "a probe budget at or above the runner's is REFUSED" --
# is RB1-1, and it is caught.
echo "  --    RB1-4 DROPPED WITH REASON: asserts RB1-1's property by a second route that"
echo "        costs 1800 s per run (the original rc=124 configuration). RB1-1 already"
echo "        proves the bound fires, in three seconds."

# RB1-5: delete a `need`. On its own that makes the gate WEAKER, not broken, so the
# state asserts the weaker thing that is true and checkable: the arm list is what
# the gate claims to have measured, and RB1-6 proves the list is load-bearing.
state RB1-5 'UNPROVEN' \
  "need 'VERBATIM'" 'true'

# RB1-6: rename one arm in a `need`. With the gate still UNPROVEN (the probe hangs,
# so `need` is never reached) this proves the arm list is NOT what makes it honest --
# which is the honest finding, and is recorded rather than papered over.
state RB1-6 'UNPROVEN' \
  "need 'VERBATIM'" "need 'VERBATIM_NOT_A_REAL_ARM'"

# RB1-8: the vacuity floor, which the hanging stub cannot reach. This is the state
# the whole matrix was missing and the one that matters most: a probe that RETURNS
# rc=0 and prints RESULT: PASS while every arm is 0.000 ms. The old gate
# certified exactly that with PASS, because `need` (string existence) and
# `RESULT: PASS` both match vacuous output. The stub mimics the real box's vacuous
# run (ceiling 0.000 ms, both arms 0.000 ms, RESULT: PASS, exit 0); the gate must
# REFUSE it.
total=$((total+1))
setup_vacuous_tree
cp "$GATE_PRISTINE" "$SB/root/scripts/gates/test_roofline_bisect1.sh"
run_gate
verdict RB1-8 'ZERO VERBATIM median' "$GATE_OUT" $?
restore_gate

# RB1-9: the POSITIVE direction, which RB1-8 alone could not provide. A floor
# that can only ever fire is the always-red twin: it certifies nothing, because
# nothing that runs can satisfy it. MEASURED 2026-10-12 -- the very defect this
# state exists to catch -- the gate's floor refused every REAL run on this box
# (VERBATIM medians 0.249-0.622 ms) with "no finite positive VERBATIM median",
# i.e. it was red for the correct reason on the wrong data. The stub is the
# real probe's own output (captured from build/run_all_test_roofline_bisect1_sh.log)
# with real measurements in it; the gate must ACCEPT it. Without this state the
# matrix proves only that the floor can still fire, which is what the old
# matrix proved and what let the defect ship.
total=$((total+1))
setup_real_tree
cp "$GATE_PRISTINE" "$SB/root/scripts/gates/test_roofline_bisect1.sh"
run_gate
pass_verdict RB1-9 "$GATE_OUT" $?
restore_gate

echo
for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

restore_gate
if ! cmp -s "$GATE" "$GATE_PRISTINE"; then
  echo "  FAIL residue: $GATE differs from its pre-matrix state" >&2; exit 1
fi
echo "  ok residue: $GATE is byte-identical to its pre-matrix state"
rm -rf "$SB"

echo
if [ "$caught" -eq "$total" ]; then
  echo "ROOFLINE-BISECT1 MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  RB1-1: the self-bound FIRES. A probe budget at or above the runner's is"
  echo "        REFUSED, so a hang can never again consume the runner's whole step or"
  echo "        surface as rc=124, which the runner reads as a REGRESSION."
  echo "  RB1-4: DROPPED WITH REASON -- it asserts RB1-1's property by a second route"
  echo "        that costs 1800 s per run, i.e. the original rc=124 configuration."
  echo "  RB1-2/3 EXCLUDED WITH REASON: with the probe hanging, softening the"
  echo "        diagnostic text leaves the verdict at rc=2 -- which is what a correct"
  echo "        gate returns -- so neither direction is distinguishable. Making them"
  echo "        observable needs a stub probe that RETURNS a defined rc, which"
  echo "        conflicts with the stub hanging (the failure the gate exists to"
  echo "        report). Excluded with the reason rather than ticked."
  echo "  RB1-5/RB1-6: the arm list is NOT what makes the gate honest. Recorded, not"
  echo "        dressed up: while the probe hangs, \`need\` is never reached, so those"
  echo "        two states show the gate is honest because of its TIMEOUT verdict and"
  echo "        its diagnostics -- not because it checked anything."
  echo "  RB1-7: the pristine gate holds its contract (rc=2 + UNPROVEN) against a"
  echo "        hanging probe."
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "ROOFLINE-BISECT1 MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
