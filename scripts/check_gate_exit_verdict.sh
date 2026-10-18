#!/usr/bin/env bash
# *** WHAT THIS IS. *** A POPULATION SWEEP over every gate in scripts/test_*.sh.
# It derives the population from the filesystem (never a hand list) and asserts
# the harvest count, then runs the four predicates from check_tp2_pass_exit.sh
# against each. "0 findings" and "looked at nothing" are different claims: a
# sweep that harvested zero gates would be reporting on an empty corpus.
#
# *** WHAT IT DOES NOT DO. *** It does not run the gates (that needs the model
# and both B580s) and it does not synthesise a log. A synthetic-log version
# would build its fixture from the fixed script and pass by construction.
#
# *** WHY A SEPARATE SCRIPT. *** check_tp2_pass_exit.sh pins ONE gate. This
# sweep exists because the defect it catches is not unique to test_tp2_27b.sh:
# any gate whose verdict never reaches its exit status reports PASS with every
# arm failing, and the five gates that end in an unconditional
# `echo "... TEST: PASS"` are safe ONLY if every FAIL branch exits non-zero
# before that line -- precisely the property test_tp2_27b.sh lacked, and which
# this sweep asserts rather than infers.
set -u
# INTFER_SCRIPTS_DIR override: the mutation harness runs a TEMP COPY of this
# sweep, so BASH_SOURCE points at the temp dir and SCRIPT_DIR would resolve
# CHECK (check_tp2_pass_exit.sh) into a directory that does not contain it --
# the sweep then failed its own pre-flight with "check_tp2_pass_exit.sh
# missing" on every mutant. The override lets the harness pin the real scripts
# dir, the same convention check_dri_probe.sh already uses for the same reason.
SCRIPT_DIR="${INTFER_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

CHECK="$SCRIPT_DIR/check_tp2_pass_exit.sh"
# `-f`, not `-x`: the tree invokes every checker through `bash "$SCRIPT_DIR/x.sh"`
# (run_all_checks.sh:310), so the executable bit is not part of the contract and
# `-x` would refuse a checker that is otherwise correct.
[ -f "$CHECK" ] || { echo "FAIL: $CHECK missing" >&2; exit 2; }

GATES="$(ls scripts/test_*.sh 2>/dev/null)"
[ -n "$GATES" ] || { echo "FAIL: no scripts/test_*.sh found" >&2; exit 2; }
N="$(printf '%s\n' "$GATES" | wc -l)"

echo "== GATE EXIT-VERDICT SWEEP ($N gates, population derived from scripts/test_*.sh) =="

# *** TWO VERDICT FAMILIES, NOT ONE. *** The sweep found what the single-gate
# checker could not: only 2 of 18 gates write a `PASS` variable and convert it
# to an exit code (test_tp2_27b.sh, test_cpu_parity.sh, test_oracle_equiv.sh,
# test_device_argmax.sh, test_gpu_optin.sh -- the latter two use `!= 1`).
# The other 16 relay the probe's OWN rc through `exit "$RC"`. Predicate (a) is
# written for the PASS-variable family; calling those 16 "unsafe" would be
# weakening the check to make it green. What distinguishes the families is
# structural, not cosmetic: a PASS-variable gate's verdict is a variable the
# gate itself writes and converts, so a missing conversion is a real defect;
# a probe-relay gate's verdict is the probe's rc, so the question is whether
# the gate's last statement is that relay.
#
# *** WHAT IS ACTUALLY AT RISK, IN BOTH FAMILIES. *** The property this sweep
# certifies is: the gate's exit status is determined by code that runs BEFORE
# any unconditional `echo "... TEST: PASS"`. In the PASS-variable family that
# is the conversion; in the probe-relay family it is the `exit "$RC"` (or an
# `exit 1` on the FAIL branch immediately preceding the echo). A gate that
# prints PASS and then falls off the end is unsafe in BOTH families, and the
# three probe-relay gates that do exactly that (test_gguf_roundtrip.sh,
# test_http_batch.sh, test_serve_load.sh) are safe ONLY because their FAIL
# branch exits 1 on the line immediately before the echo -- verified by hand,
# not by the checker, because the checker asserts the conversion, not the
# adjacency. That adjacency is the real invariant and it is asserted here.
#
# *** WHAT THIS SWEEP DOES NOT DO. *** It does not run the gates. It does not
# synthesise a log. And it does not re-derive the predicate set -- that lives in
# check_tp2_pass_exit.sh, which now CLASSIFIES each gate by family (PASS=1
# written -> PASS-variable predicates a/b/c/d; otherwise -> probe-relay
# predicate d2) and applies the set that governs. The sweep's job is to run
# that checker over the whole population and assert it harvested something.
#
# *** WHY THE CLASSIFICATION LIVES IN THE CHECKER, NOT THE SWEEP. *** Putting
# d2 in the sweep and letting the checker fail probe-relay gates on (a) would
# have made the sweep's verdict depend on an ordering detail: a probe-relay
# gate that failed (a) would never reach d2, so d2 would be dead code and the
# three bare-PASS-echo gates would be "safe by hand" forever. Moving the
# family test to the TOP of the checker makes d2 reachable for exactly the
# gates it governs, and the PE-4 mutation (delete PASS=1) can no longer flip
# a gate's family to hide the defect -- it would flip it the other way and
# make a PASS-variable gate fail d2, which is a caught mutation, not a
# silence.
harvested=0
passvar=0
relay=0
unsafe=0
for g in $GATES; do
  if grep -Eq '^[[:space:]]*PASS=1[[:space:]]*$' "$g"; then
    family=PASS-variable; passvar=$((passvar+1))
  else
    family=probe-relay; relay=$((relay+1))
  fi
  out="$(bash "$CHECK" "$g" 2>&1)"; rc=$?
  harvested=$((harvested+1))
  if [ "$rc" -ne 0 ]; then
    unsafe=$((unsafe+1))
    printf '%-32s UNSAFE  %-12s rc=%s\n' "$g" "$family" "$rc"
    printf '%s\n' "$out" | sed 's/^/    /'
  else
    printf '%-32s ok      %-12s\n' "$g" "$family"
  fi
done

# safe is DERIVED, never counted: the loop above increments `unsafe` for every
# gate that fails (a) or (d2), and every other harvested gate is safe by
# elimination. Counting `safe` separately would let a gate that neither
# passed nor failed silently inflate the total -- the same defect the sweep
# exists to catch, one layer up.
safe=$((harvested - unsafe))
echo "harvested=$harvested safe=$safe unsafe=$unsafe (pass-variable=$passvar probe-relay=$relay)"

if [ "$harvested" -ne "$N" ]; then
  echo "FAIL: harvested $harvested of $N gates -- the sweep did not look at everything" >&2
  exit 1
fi
if [ "$harvested" -eq 0 ]; then
  echo "FAIL: harvested 0 gates -- the sweep looked at nothing, so 0 findings" >&2
  echo "      is indistinguishable from 0 unsafe. Re-derive the population." >&2
  exit 1
fi
if [ "$unsafe" -ne 0 ]; then
  echo "GATE-EXIT-VERDICT SWEEP: FAIL ($unsafe/$harvested gates unsafe)" >&2
  exit 1
fi
echo "GATE-EXIT-VERDICT SWEEP: PASS ($safe/$harvested gates safe, verdict reaches exit status)"
echo "RESULT: PASS"
exit 0
