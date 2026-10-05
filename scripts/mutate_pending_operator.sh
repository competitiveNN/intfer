#!/usr/bin/env bash
# Mutation matrix for scripts/check_pending_operator.sh.
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
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/pending_operator_matrix"
CHECK="scripts/check_pending_operator.sh"

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
RESULT: doc counts STALE -- re-run scripts/mutate_doc_counts.sh on this machine
EOF
  cat >"$1/run_all_ok_step.log" <<'EOF'
RESULT: check ok
EOF
}

restore() { rm -rf "$SANDBOX"; mklogs "$SANDBOX"; }

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
  exit 0
fi

echo "== baseline (sandbox with a healthy log pair) =="
restore
INTFER_BUILD_DIR="$SANDBOX" bash "$CHECK" >"$SB/base.log" 2>&1; rc=$?
sed -n '/classification of steps/,/CLEARABLE HERE/p' "$SB/base.log" | sed 's/^/  /'
if [ "$rc" -ne 0 ]; then
  echo "  FAIL baseline rc=$rc" >&2; tail -6 "$SB/base.log" >&2; exit 1
fi
if ! grep -q 'PENDING-OPERATOR (remediation needs the 27B run): 1' "$SB/base.log"; then
  echo "  FAIL baseline did not classify exactly 1 pending step" >&2
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

po1; run_state PO-1 'INCONCLUSIVE|no .*logs' reject
po2; run_state PO-2 'no gate script names the 27B model|broken derivation' reject "$SANDBOX/chk.sh"
po3; run_state PO-3 '' keep-green

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
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "PENDING-OPERATOR MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
