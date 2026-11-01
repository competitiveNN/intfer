#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_pending_operator.sh.
#
# What the checker claims: the stack's unproven steps split into
# OPERATOR-DEFERRED (a remediation that needs the 27B gate run, which an agent
# session must not drive) and CLEARABLE HERE -- and, the property that actually
# bites, an ABSENCE OF LOGS is INCONCLUSIVE rather than "nothing pending".
#
# PO-1 empties the log directory. This is the anti-vacuity state, and it is the
# one that matters: the first version of the classifier grepped for the stack's
# classification line ("STALE rc="), which run_all_checks.sh prints to STDERR, so
# it found nothing in any of the 63 logs and printed "0 pending, 0 clearable" on
# a tree where one step IS pending. A vacuous zero that reads as a clean bill of
# health. PO-1 reproduces exactly that and must be REFUSED, not passed as "no
# pending work".
#
# PO-2 destroys the 27B-model marker, so the "gates that bind the 27B model" set
# derives empty. The classification then cannot tell pending from clearable, and
# the checker must say so rather than quietly reporting zero of each -- a derived
# set that comes out empty is a broken query, not an absence.
#
# PO-3 is the always-red twin in the OTHER direction: a step whose RESULT line
# says STALE but whose remediation does NOT mention the gate run must be reported
# as CLEARABLE. If the classifier called everything pending it would pass PO-1..2
# for the wrong reason and be useless.
#
# PO-4 is the DEFECT THAT WAS MEASURED on the real tree, 2026-10-16. The classifier's
# predicate was `run_ten_gates\.sh|run_ten_gates\b`, which matches the checker's own
# OUTPUT as well as its remediation advice. Two TEXT-ONLY steps were therefore filed
# under "remediation needs the 27B run" -- the operator was told to spend tens of
# minutes and both GPUs on steps that only grep source, and both were re-run here
# and PASSED. The logs that produced it (kept verbatim in the state, because the
# matrix must grade the shape that actually occurred, not a tidier one):
#   run_all_check_scripts_live_sh.log   [gate glob parsed from run_ten_gates.sh]
#                                       -- gate class: test_*.sh (parsed from run_ten_gates.sh,
#                                       run_ten_gates.sh    NAMED by check_engine_cert.sh
#   run_all_check_build_dir_consumers_sh.log   run_ten_gates.sh  ok  routes through ...
# Every one of those is a SUBJECT the checker reports on, not a command it issues.
# The advice the classifier must read is always the imperative form -- a verb
# directly in front of the PATH (`run scripts/run_ten_gates.sh`), which is what
# check_engine_cert.sh:56/:245 and run_all_checks.sh:779/1148 print. PO-4 plants
# exactly the subject-only shape and requires CLEARABLE.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/pending_operator_matrix"
# *** WIPE LOCK, OUTSIDE THE TREE IT WIPES (check_matrix_sandbox_locks.sh). ***
# The sandbox below is a FIXED path wiped on every invocation, so two concurrent
# runs share one tree: the second's wipe empties the first's mid-state and the
# first then grades a corpus the other has just rebuilt (MEASURED 2026-10-21 on
# mutate_durable_reads.sh). Taken BEFORE the first wipe and released on
# EXIT/INT/TERM, so a run that dies mid-state does not brick the next.
LOCK="$BUILD/.mutate_pending_operator.lock"
mkdir -p "$BUILD" 2>/dev/null || true
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_pending_operator.sh is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

CHECK="scripts/checks/check_pending_operator.sh"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

# The checker reads $BUILD/run_all_*.log. The states therefore operate on a SANDBOX
# log directory via INTFER_BUILD_DIR, never on the real one -- the logs are
# evidence and deleting them to test a classifier would destroy the very thing it
# classifies.
SANDBOX="$SB/sandbox"
mkdir -p "$SANDBOX"

# A healthy pair of step logs: one pending (remediation names the gate run), one
# clearable (stale, but re-runnable here).
mklogs() { # $1 = dir
  mkdir -p "$1"
  cat >"$1/run_all_check_engine_cert_sh.log" <<'EOF'
RESULT: engine certification STALE -- engine input changed since it was measured
        Re-run scripts/run_ten_gates.sh to re-certify.
EOF
  cat >"$1/run_all_check_doc_counts_sh.log" <<'EOF'
RESULT: doc counts STALE -- re-run scripts/matrices/mutate_doc_counts.sh on this machine
EOF
  cat >"$1/run_all_ok_step.log" <<'EOF'
RESULT: check ok
EOF
}

restore() { rm -rf "$SANDBOX"; mklogs "$SANDBOX"; }

# The two classification blocks, extracted from a SAVED checker log. PO-4/PO-4b
# assert on WHICH block a step lands in, not on a bare count: "pending=0" would
# also be satisfied by a classifier that dropped the step out of BOTH classes,
# which is the vacuous direction -- a missing answer is not a correct one.
pend_block() { sed -n '/PENDING-OPERATOR step(s)/,/CLEARABLE-HERE/p' "$1"; }
clear_block() { sed -n '/CLEARABLE-HERE step(s)/,/^[[:space:]]*$/p' "$1"; }

total=0; caught=0; RESULTS=()
# $1 id, $2 expect-regex, $3 want: reject | keep-green, $4 checker to run (defaults
# to the real one). $4 exists because PO-2 mutates a COPY of the checker; the
# first version of this harness always ran the real one, so PO-2's mutation was
# written to the sandbox and then discarded -- and the state scored MISSED for a
# reason that had nothing to do with the checker.
run_state() {
  local id="$1" expect="$2" want="${3:-reject}" chk="${4:-$CHECK}" out rc
  total=$((total + 1))
  out="$(INTFER_BUILD_DIR="$SANDBOX" bash "$chk" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  restore
  if [ "$want" = "keep-green" ]; then
    # The always-red twin: a correct classifier must STAY green here.
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE -- rc=$rc, want 0)'"); fi
    return
  fi
  if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'"); return; fi
  if printf '%s' "$out" | grep -qE -- "$expect"; then
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  else
    RESULTS+=("$id MISSED '(rejected, wrong reason; rc=$rc)'")
  fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  PO-1  no step logs at all -> must be INCONCLUSIVE, not 'nothing pending'"
  echo "  PO-2  the 27B-model marker destroyed -> derived set empty     [must REJECT]"
  echo "  PO-3  a stale step with NO gate remediation is CLEARABLE HERE   [must ACCEPT]"
  echo "  PO-4  a subject-only run_ten_gates mention is CLEARABLE        [must ACCEPT]"
  echo "  PO-4b the ORIGINAL loose predicate misfiles it as pending     [must CATCH]"
  exit 0
fi

echo "== baseline (sandbox with a healthy log pair) =="
restore
INTFER_BUILD_DIR="$SANDBOX" bash "$CHECK" >"$SB/base.log" 2>&1; rc=$?
sed -n '/classification of steps/,/CLEARABLE HERE/p' "$SB/base.log" | sed 's/^/  /'
if [ "$rc" -ne 0 ]; then
  echo "  FAIL baseline rc=$rc" >&2; tail -6 "$SB/base.log" >&2; exit 1
fi
# Match on the COUNT, not on the label's prose. An earlier version asserted the
# whole sentence, so renaming the label (adding "step(s)", to make the unit
# explicit) silently broke the baseline -- a checker whose expectation encodes one
# exact wording is a checker that fails for a reason unrelated to its subject.
if ! grep -qE 'PENDING-OPERATOR step\(s\).*: 1$' "$SB/base.log"; then
  echo "  FAIL baseline did not classify exactly 1 pending step" >&2
  echo "        got: $(grep -E 'PENDING-OPERATOR|CLEARABLE' "$SB/base.log" | tr '\n' ' ')" >&2
  echo "        (the whole matrix measures the split, so a baseline that does not" >&2
  echo "         split proves nothing)" >&2
  exit 1
fi
echo "  ok baseline rc=0, exactly 1 pending and 1 clearable"
echo "== states =="

# PO-1 empty the corpus.
po1() { rm -f "$SANDBOX"/run_all_*.log; }

# PO-2 break the derivation that finds the 27B-bound gates, in the SANDBOX copy of
# the checker so the real one is untouched.
# Match the model FILENAME, which appears literally in the checker -- no regex
# escaping to get wrong. The first version used the dotted `Qwen3.\.8-27B`
# pattern through two layers of quoting and matched nothing, so the state scored
# MISSED for a reason that had nothing to do with the floor it was testing.
po2() { sed "s/27B-int2/ZZZ_NO_SUCH_MODEL/g" "$CHECK" >"$SANDBOX/chk.sh"
        chmod +x "$SANDBOX/chk.sh"; }

# PO-3 drop the gate-run remediation from the pending step: it must become
# CLEARABLE, so this state must stay GREEN. If the classifier called everything
# pending it would pass PO-1/2 for the wrong reason.
po3() { cat >"$SANDBOX/run_all_check_engine_cert_sh.log" <<'EOF'
RESULT: engine certification STALE -- re-run the stack on this machine
EOF
}

# PO-4 plant a TEXT-ONLY step whose only mentions of run_ten_gates.sh are the
# SUBJECTS it classifies -- the exact shape measured in the two real logs above.
# It must be reported CLEARABLE, not operator-deferred.
po4() { cat >"$SANDBOX/run_all_check_scripts_live_sh.log" <<'EOF'
== SCRIPTS LIVE (population: scripts/*.{sh,py,c}) ==
  ok    [gate glob parsed from run_ten_gates.sh] harvested 9 item(s)
-- gate class: test_*.sh (parsed from run_ten_gates.sh, not restated)
  ok    run_ten_gates.sh                   NAMED by scripts/checks/check_engine_cert.sh (+34 more)
RESULT: FAIL (2) -- a script nothing runs is not tooling
EOF
}

po1; run_state PO-1 'INCONCLUSIVE|no .*logs' reject
po2; run_state PO-2 'no gate script names the 27B model|broken derivation' reject "$SANDBOX/chk.sh"
po3; run_state PO-3 '' keep-green

# PO-4 is decided on the CLASSIFICATION, not on the rc: check_pending_operator.sh
# ends in `exit 0` on every path, so an rc-based state would be satisfied by a
# classifier that files EVERY failing step in the operator's queue -- the
# always-red twin in disguise. Two directions, and the second is what stops the
# first from being vacuous:
#   (a) the CORRECT predicate must leave the planted step in CLEARABLE;
#   (b) a copy carrying the ORIGINAL loose predicate must move the SAME log into
#       PENDING -- proving the state can still see the defect it is named after.
# NB the sandbox is the STANDARD pair plus the planted log, and `restore` re-seeds
# `engine_cert` (whose advice legitimately matches) -- so the expected split is
# 1 pending / 2 clearable. The first draft of this state asserted 0/1, forgot
# that re-seed, and would have MISSED a correct checker. A state whose
# expectation is wrong about its own fixture is worse than no state: it fails the
# code that is right and teaches the next session to distrust a green matrix.
po4
total=$((total + 1))
out="$(INTFER_BUILD_DIR="$SANDBOX" bash "$CHECK" 2>&1)"; rc=$?
printf '%s\n' "$out" >"$SB/PO-4.log"
restore
if [ "$rc" -eq 0 ] \
   && grep -qE 'PENDING-OPERATOR step\(s\)[^:]*: 1$' "$SB/PO-4.log" \
   && ! pend_block "$SB/PO-4.log" | grep -qE '^[[:space:]]+run_all_check_scripts_live_sh$' \
   && clear_block "$SB/PO-4.log" | grep -qE '^[[:space:]]+run_all_check_scripts_live_sh$'; then
  caught=$((caught + 1)); RESULTS+=("PO-4 CAUGHT ''")
else
  RESULTS+=("PO-4 MISSED '(a subject mention was filed as operator-deferred)'")
fi

# (b) the negative direction. The copy CANNOT live in $SANDBOX: the checker derives
# ROOT from its own directory, and $SANDBOX sits two levels below the build tree,
# so `cd $ROOT` leaves the real scripts/gates/test_*.sh unfindable and the 27B-gate
# derivation (the PO-2 floor) refuses BEFORE the classifier runs -- MEASURED on the
# first draft, which printed "broken derivation" and rc=1. mktemp directly under
# the REPO ROOT keeps the parent correct; the directory is removed immediately so
# the residue check still sees a clean tree.
LOOSE_DIR="$(mktemp -d "$ROOT/.po4loose.XXXXXX")" || LOOSE_DIR=""
if [ -n "$LOOSE_DIR" ]; then
  sed 's/grep -qiE .(re-?run|run)\[\[:space:\]\]+scripts\/run_ten_gates\\.sh./grep -qE "run_ten_gates\\.sh|run_ten_gates\\b"/g' \
      "$CHECK" >"$LOOSE_DIR/chk_loose.sh" 2>/dev/null
  chmod +x "$LOOSE_DIR/chk_loose.sh" 2>/dev/null
fi
if [ -n "$LOOSE_DIR" ] && grep -qF 'run_ten_gates\.sh|run_ten_gates\b' "$LOOSE_DIR/chk_loose.sh" 2>/dev/null; then
  po4
  total=$((total + 1))
  lout="$(INTFER_BUILD_DIR="$SANDBOX" bash "$LOOSE_DIR/chk_loose.sh" 2>&1)"; lrc=$?
  printf '%s\n' "$lout" >"$SB/PO-4b.log"
  restore
  if [ "$lrc" -eq 0 ] \
     && grep -qE 'PENDING-OPERATOR step\(s\)[^:]*: 2$' "$SB/PO-4b.log" \
     && pend_block "$SB/PO-4b.log" | grep -qE '^[[:space:]]+run_all_check_scripts_live_sh$'; then
    caught=$((caught + 1)); RESULTS+=("PO-4b CAUGHT ''")
  else
    RESULTS+=("PO-4b MISSED '(the loose predicate no longer misfiles -- the state proves nothing)'")
  fi
else
  total=$((total + 1))
  RESULTS+=("PO-4b MISSED '(could not build a loose-predicate copy to prove the direction)'")
fi
[ -n "${LOOSE_DIR:-}" ] && rm -rf "$LOOSE_DIR"

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

rm -rf "$SANDBOX"
echo "  ok residue: the real $BUILD/run_all_*.log untouched (states ran in a sandbox)"

echo
if [ "$caught" -eq "$total" ]; then
  echo "PENDING-OPERATOR MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  PO-1: an empty log set is INCONCLUSIVE, never 'nothing pending' -- the"
  echo "        vacuous zero that the first version of the classifier actually printed"
  echo "  PO-2: a broken 27B derivation is visible rather than reported as zero"
  echo "  PO-3: a stale step with no gate remediation is CLEARABLE (not always-pending)"
  echo "  PO-4 / PO-4b: the predicate names the ADVICE, not any mention -- a subject-only"
  echo "        run_ten_gates.sh line is CLEARABLE, and PO-4b proves the original loose"
  echo "        predicate still misfiles that very log as operator-deferred."
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "PENDING-OPERATOR MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
