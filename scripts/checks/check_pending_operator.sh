#!/usr/bin/env bash
# Separate "the OPERATOR owes a run" from "an unproven step nobody has explained".
#
# WHY. The stack currently ends rc=2 with a line that lumps two very different
# things together: "1 step(s) unproven". That step is check_engine_cert.sh, whose
# remediation is printed in plain sight -- "Re-run scripts/run_ten_gates.sh" --
# and which stays rc=1 until the operator runs the ten gates, because that run
# loads the 27B model and the standing rule forbids an agent session from driving
# it. So the rc=2 is not a defect and not a surprise, yet it is presented exactly
# like a stale stamp that a rerun would clear, and those two DO differ: one clears
# in 40 minutes on this machine, the other cannot be cleared from here at all.
#
# A verdict nobody can act on trains people to ignore the verdict. So the
# operator-deferred steps get their own DERIVED class, with the exact command.
#
# BOTH SIGNALS ARE DERIVED, NEVER LISTED:
#   * the remediation is read out of the failing step's own log, which is where the
#     tooling already prints it -- so this checker cannot disagree with the advice
#     the operator was given;
#   * "needs the operator" means the remediation names a step that binds the 27B
#     model. The 27B-bound gates are found by searching scripts/gates/test_*.sh for the
#     model name, so adding a second one needs no edit here.
# A hardcoded list of operator-only steps would rot exactly the way the
# hand-written flag census in AGENTS.md did.
#
# ANTI-VACUITY. Both sets are printed with their sizes. An empty log directory is
# reported as "nothing to classify" rather than as "nothing pending", because those
# are different claims and only one of them is reassuring.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
echo "== OPERATOR-DEFERRED STEPS (derived, not listed) =="

# --- the 27B-bound gates, derived from the gate scripts themselves ---------
mapfile -t BIG_GATES < <(grep -lE 'Qwen3\.8-27B|27B-int2' scripts/gates/test_*.sh 2>/dev/null | xargs -r -n1 basename)
n_big="${#BIG_GATES[@]}"
echo "-- gates that bind the 27B model (operator-only): $n_big"
[ "$n_big" -gt 0 ] && printf '     %s\n' "${BIG_GATES[@]}"

# *** FLOOR ON THE DERIVED 27B SET, added because PO-2 MISSED against it. *** PO-2
# breaks the derivation (no gate script names the model) and nothing happened: the
# classifier reports the count, prints "0", and carries on -- because the
# classification itself keys off each step's REMEDIATION TEXT, not off this set,
# so the set was decoration. A derived set that can come out EMPTY without
# failing is a decorative set, and a decorative set is worse than none: it reads
# as "there are no operator-only gates", which is a claim about the world.
#
# This repo has a TP=2 27B gate (test_tp2_27b.sh) and a standing rule that keeps
# an agent session from driving it, so an empty set means the derivation broke.
if [ "$n_big" -lt 1 ]; then
  echo "  FAIL  no gate script names the 27B model, so the operator-only set is" >&2
  echo "        EMPTY -- a broken derivation, not an absence. test_tp2_27b.sh" >&2
  echo "        exists and binds it." >&2
  exit 1
fi

# --- the per-step logs a stack run leaves behind ---------------------------
# *** NULLGLOB + BARE ls IS A TRAP, AND IT BIT THIS FILE. *** With nullglob on, a
# glob matching nothing expands to NOTHING -- so `ls -1 "$BUILD"/run_all_*.log`
# became `ls -1` with no argument, which lists the CURRENT DIRECTORY. An empty
# sandbox therefore reported "14 per-step logs found", the empty case was never
# reached, and the PO-1 anti-vacuity state passed against a checker reading a
# completely unrelated listing. The array form below cannot do that: with
# nullglob the array is simply EMPTY, which is the answer we want.
shopt -s nullglob
LOGS=("$BUILD"/run_all_*.log)
shopt -u nullglob
n_logs="${#LOGS[@]}"
echo "-- per-step logs found: $n_logs"

if [ "$n_logs" -eq 0 ]; then
  echo "  nothing to classify: no $BUILD/run_all_*.log from a stack run."
  echo "  NOT the same claim as 'nothing is pending the operator' -- run"
  echo "  scripts/run_all_checks.sh first, or the directory moved."
  echo "RESULT: INCONCLUSIVE (no stack logs to classify)"
  exit 2
fi

# --- classify: a step is operator-deferred when its own remediation names a
#     27B-bound step. Read from the log, so it cannot disagree with the advice. ---
pending=(); other=(); declare -A seen_step=()
# The UNIT IS THE STEP, NOT THE LOG. A step that fails and is then re-run leaves
# TWO logs (`run_all_X_sh.log` and `run_all_X_sh_rerun.log`) and the stack prints
# the step once -- so counting logs reported "2 steps need the 27B run" for what is
# ONE step. A count that does not mean what its label says is the class this repo
# keeps auditing, and it overstates the operator's work, which is the one number in
# this repo that must be exactly right.
step_of() {  # $1 = log basename -> the step name it belongs to
  local b="${1%.log}"
  b="${b%_rerun}"
  printf '%s\n' "$b"
}

pending=(); other=(); declare -A seen_step=()
for f in "${LOGS[@]}"; do
  b="$(basename "$f")"
  st="$(step_of "$b")"
  # Detect an unproven step from ITS OWN log, not from the stack's classification
  # line. The first version grepped for 'STALE rc=' and found nothing in any of
  # the 63 logs: that line is printed by run_all_checks.sh to STDERR, while a
  # step's own log carries its own `RESULT:` verdict. So the classifier reported
  # "0 pending, 0 clearable" on a tree where one step IS pending -- a vacuous
  # zero read as a clean bill of health. Each step announces itself; read that.
  if grep -qE '^RESULT:.*(STALE|UNPROVEN|FAIL)' "$f" 2>/dev/null; then
    # One entry per STEP: the re-run of a step is the same obligation.
    [ -n "${seen_step[$st]:-}" ] && continue
    seen_step[$st]=1
    # *** THE PREDICATE MUST MATCH THE ADVICE, NOT ANY MENTION OF THE SCRIPT. ***
    # MEASURED 2026-10-16, the original `run_ten_gates\.sh|run_ten_gates\b` read
    # the checker's own OUTPUT as if it were remediation advice and put two
    # TEXT-ONLY steps into the operator-deferred set:
    #   run_all_check_scripts_live_sh.log         line 4   [gate glob parsed from run_ten_gates.sh]
    #                                              line 5   -- gate class: test_*.sh (parsed from run_ten_gates.sh
    #                                              line 107 run_ten_gates.sh   NAMED by check_engine_cert.sh
    #   run_all_check_build_dir_consumers_sh.log  line 74  run_ten_gates.sh   ok    routes through a documented override
    # Neither needs a GPU, a model or the 27B run -- both were re-run here and
    # PASSED, yet the checker was telling the operator to spend "tens of minutes"
    # and both GPUs on a step that only greps source. The advice this classifier
    # actually reads is always the IMPERATIVE form -- check_engine_cert.sh:56
    # "run scripts/run_ten_gates.sh", :245 "Re-run scripts/run_ten_gates.sh",
    # run_all_checks.sh:779/1148 "re-run scripts/run_ten_gates.sh" -- so the
    # predicate now requires a verb directly in front of the PATH form. A subject
    # mention is a noun ("run_ten_gates.sh  NAMED by ..."), never a command, and
    # cannot match. PO-4 pins it: a log whose only mentions are subjects must be
    # reported CLEARABLE, not operator-deferred.
    if grep -qiE '(re-?run|run)[[:space:]]+scripts/run_ten_gates\.sh' "$f" 2>/dev/null; then
      pending+=("$st")
    else
      other+=("$st")
    fi
  fi
done

n_pending="${#pending[@]}"; n_other="${#other[@]}"
echo
echo "-- classification (unit = STEP, not log; a re-run is the same debt) --"
echo "   PENDING-OPERATOR step(s) (remediation needs the 27B run): $n_pending"
[ "$n_pending" -gt 0 ] && printf '     %s\n' "${pending[@]}"
echo "   CLEARABLE-HERE step(s) (re-run on this machine): $n_other"
[ "$n_other" -gt 0 ] && printf '     %s\n' "${other[@]}"

echo
echo "-- the exact commands owed --"
if [ "$n_pending" -gt 0 ]; then
  echo "     source /opt/intel/oneapi/setvars.sh"
  echo "     scripts/run_ten_gates.sh     # certifies the engine; writes build/ten_gates/HEAD"
  echo "     scripts/checks/check_engine_cert.sh # must stop reporting STALE"
fi
if [ "$n_other" -gt 0 ]; then
  echo "     (the CLEARABLE set needs no operator action -- re-run the stack here)"
fi

echo
echo "PENDING-OPERATOR: $n_pending step(s) need the 27B gate run; $n_other are clearable on this box."
echo "  (counted in STEPS: a step that fails and is re-run leaves two logs but one debt)"
echo "  This is why the stack ends rc=2 rather than rc=1: nothing is wrong with the"
echo "  code, and no amount of work in an agent session will clear the pending set."
echo "RESULT: $n_pending operator-deferred, $n_other locally clearable"
exit 0
