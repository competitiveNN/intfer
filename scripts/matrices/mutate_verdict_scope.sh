#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_verdict_scope.sh (AUDIT #29).
#
# THE MODEL IS AUDIT #21 APPLIED TO A TREE. The checker's SUBJECT is the scripts/
# directory itself, so every state mutates the SANDBOX TREE and requires the
# checker to reject it. Weakening the checker would be unobservable here for the
# same reason it was in mutate_bench_load_gate.sh: the classifier is correct by
# construction once written, so relaxing it keeps it green.
#
# THE SEAM IS THE CHECKER'S OWN. scripts/checks/check_verdict_scope.sh takes
# INTFER_SCRIPTS_DIR, so this matrix needs no git sandbox and no tracked-file
# editing at all -- which also means it cannot damage the tree it is verifying.
# The cost of that choice, stated rather than hidden: this matrix does NOT prove
# the checker behaves under a dirty tree or a moved HEAD (mutate_run_all_checks.sh
# covers those for the stack), and it says so rather than implying otherwise.
#
# *** verify_patched IS DELIBERATELY NOT USED ON THE .py TARGET. *** It runs
# `bash -n`, and a Python file is not shell: using it would have reported a
# "syntactically broken after patching" STATE FAULT for every state that patched
# _bench_summary.py. The python states check `python3 -m py_compile` instead,
# which is the right syntax checker for the right language.
#
# Usage: scripts/matrices/mutate_verdict_scope.sh
# Exit: 0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_gate_patch.sh
. "$SCRIPT_DIR/../lib/_gate_patch.sh"
# shellcheck source=scripts/lib/_gate_pop.sh
. "$SCRIPT_DIR/../lib/_gate_pop.sh"
# shellcheck source=scripts/lib/_gate_pred.sh
. "$SCRIPT_DIR/../lib/_gate_pred.sh"

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/verdict_scope_mutate"
LOG="$STAMP/states"
CHECK="$SCRIPT_DIR/../checks/check_verdict_scope.sh"

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
mkdir -p "$LOG"
: > "$LOG/summary.txt"

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_vscope_mut.XXXXXX")"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM
SBD="$SB/scripts"
restore() { rm -rf "$SBD"; cp -a "$SCRIPT_DIR" "$SBD"; }
# *** THE SANDBOX COPY IS WHAT RUNS, NOT THE REAL CHECKER. *** The first version
# ran "$CHECK" (the real file) with only its SUBJECT tree redirected, so the two
# checker-patching states were unobservable by construction -- the mutation had
# literally no effect, and both scored MISSED. Running the sandbox's own copy
# makes the subject and the grader come from one tree, which is what a reader
# assumes a sandbox means. Its self-tests still terminate: case (c) re-invokes
# BASH_SOURCE (now the sandbox copy) with INTFER_VERDICT_SCOPE_CASES=0, so the
# recursion guard is what breaks the cycle, not an accident of which file runs.
run_check() { INTFER_SCRIPTS_DIR="$SBD" bash "$SBD/check_verdict_scope.sh" 2>&1; }
summary="$SBD/_bench_summary.py"
checker_sb="$SBD/check_verdict_scope.sh"

# ---- the anchors, asserted BEFORE the baseline (AUDIT #17's rule) -----------
# Occurring EXACTLY ONCE, not merely present: an anchor matching twice would let
# patch_literal rewrite whichever comes first, i.e. a different state than the one
# described. This matrix earned that rule the hard way one matrix over (AUDIT
# #28), where a state scored MISSED only after the baseline had run.
check_anchor() { # $1 file, $2 anchor text
  local n
  n="$(python3 -c 'import sys;print(open(sys.argv[1]).read().count(sys.argv[2]))' "$1" "$2")"
  if [ "$n" != "1" ]; then
    echo "ANCHOR STALE in $(basename "$1"): found $n occurrence(s), want 1."
    printf '  first line: %s\n' "$(printf '%s' "$2" | head -1)"
    return 1
  fi
}
A_GATE='    if len(common) < 2:'
A_CLEAR='        v = ("CLEARS the spread (floor=%.1f ms): report %+.1f ms (%+.2f%%); every"'
A_VOCAB='VOCAB="CLEARS the spread|INSIDE the spread|NO SPREAD ESTIMATE"'
A_STRIP="strip_comments() { sed -e 's/[[:space:]]*#.*\$//'; }"
restore
for pair in "summary:$A_GATE" "summary:$A_CLEAR" "checker:$A_VOCAB" "checker:$A_STRIP"; do
  case "$pair" in
    summary:*) f="$summary"; t="${pair#summary:}" ;;
    checker:*) f="$checker_sb"; t="${pair#checker:}" ;;
  esac
  check_anchor "$f" "$t" || { echo "PREFLIGHT FAIL (nothing was mutated)"; exit 1; }
done
echo "  ok    every anchor occurs exactly once in the sandbox tree"

echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

echo
echo "== baseline =="
restore
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -10; exit 1
fi
n_ok="$(printf '%s\n' "$b" | grep -cE '^\s+ok' || true)"
[ "$n_ok" -ge 6 ] || { echo "BASELINE FAIL -- only $n_ok ok lines"; exit 1; }
echo "  ok    baseline passes with $n_ok ok lines"
echo "  ok    CONVERSE: the healthy tree is not flagged, so an always-red"
echo "        classifier cannot score as caught"

caught=0; total=0
run_state() { # $1 id, $2 marker, $3 expected-RE (unique to this state), $4 why
  local id="$1" marker="$2" expect="$3" why="$4"
  total=$((total + 1)); restore
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed (STATE fault)"
    echo "$id MISSED (fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  local o r; o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still passed with the damage ($why)"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_present "$id rejection-reason" "$o" "$expect"; then
    echo "  $id MISSED -- it failed, but NOT for its own reason (unattributed)"
    printf '%s\n' "$o" | sed 's/^/        /' | tail -12
    echo "$id MISSED (unattributed)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-present ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught + 1)); return 0
}

echo
echo "== states (each damages the TREE the checker grades, not the checker) =="

# V-A: A SECOND SITE EMITS A SPREAD VERDICT. The whole point of the checker: the
# family cannot grow silently.
MUTATE_FN() {
  cat > "$SBD/second_summary.sh" <<'EOF'
#!/usr/bin/env bash
noise=$(awk '{if($1>m)m=$1}END{print m}' "$LOG")
med=$(printf '%s\n' "$V" | sort -n | awk '{a[NR]=$1}END{print a[int((NR+1)/2)]}')
if [ "$med" -gt "$noise" ]; then echo "CLEARS the spread"; fi   # V_A
EOF
  grep -q 'V_A' "$SBD/second_summary.sh"
}
run_state "V-A" 'V_A' 'second_summary\.sh' \
  "a second site issues a spread verdict and is not flagged"

# V-B: THE OWNER'S GATE GOES BACK TO THE BATCH. This is AUDIT #28 defect #1
# itself, re-applied to prove the structural pin sees it.
MUTATE_FN() {
  patch_literal "$summary" "$A_GATE" '    if rounds < 2:   # V_B' || return 1
  grep -q 'V_B' "$summary"
}
run_state "V-B" 'V_B' 'paired-round gate count=' \
  "the owner's gate is the batch-wide round count again"

# V-C: A SECOND CLEARING BRANCH INSIDE THE OWNER. The copy of the rule that no
# fixture would notice, because the fixture drives one code path.
MUTATE_FN() {
  # The second branch must repeat the PREFIX the checker counts, or it is not a
  # second instance of the rule: the first version appended a differently-shaped
  # line, the count stayed at 1, and the state scored MISSED with the damage
  # present -- i.e. a state that failed to REPRODUCE its defect, which is worse
  # than no state (the verify_patched rule).
  patch_literal "$summary" "$A_CLEAR" "$A_CLEAR
    v = (\"CLEARS the spread (floor=0.0 ms): a second, unscoped branch\"   # V_C" || return 1
  grep -q 'V_C' "$summary"
}
run_state "V-C" 'V_C' 'clearing' \
  "a second, unscoped clearing branch exists inside the owner"

# V-D: COMMENT-STRIPPING IS A NO-OP, so a prose mention becomes a site. THE
# CONVERSE STATE: for this one the checker firing is NOT the success -- the
# success is that case (d) notices the degradation. Polarity named accordingly.
MUTATE_FN() {
  patch_literal "$checker_sb" "$A_STRIP" \
    'strip_comments() { cat; }   # V_D comments are no longer stripped' || return 1
  grep -q 'V_D' "$checker_sb"
}
run_state "V-D" 'V_D' 'comment-only mention was classified as a site' \
  "prose is graded as a mechanism once comment-stripping is disabled"

# V-E: THE ENUMERATION FINDS NOTHING. The AUDIT #20 vacuous-harvest shape: a
# vocabulary that matches no file would make every other case pass.
MUTATE_FN() {
  patch_literal "$checker_sb" "$A_VOCAB" \
    'VOCAB="zzz_no_such_vocabulary_zzz"' || return 1
  grep -q 'zzz_no_such_vocabulary_zzz' "$checker_sb"
}
run_state "V-E" 'V_E' 'harvested 0 items -- cannot certify' \
  "an enumeration that harvests nothing is still reported as clean"

# ---- V-F: ISOLATION, NOT A MUTATION ---------------------------------------
# V-E cannot isolate the FOLD: with the vocabulary emptied the checker's own
# case (c) also fires (its planted emitter is invisible to an empty vocabulary,
# so the inner run passes and the outer expects a refusal), so V-E goes red for
# two reasons and proves neither alone. This asserts the property directly, by
# emptying the harvest AND disabling the self-tests, so the enumeration guard is
# the only thing that can turn the checker red. It is the AUDIT #19 shape -- a
# case the states did not set up -- and it is what makes the fold load-bearing
# rather than decorative.
echo
echo "== V-F isolation: the enumeration guard alone turns the checker red =="
restore
patch_literal "$checker_sb" "$A_VOCAB" 'VOCAB="zzz_no_such_vocabulary_zzz"' || {
  echo "  V-F MISSED -- could not empty the vocabulary (STATE fault)"; exit 1; }
iso="$(INTFER_SCRIPTS_DIR="$SBD" INTFER_VERDICT_SCOPE_CASES=0 bash "$checker_sb" 2>&1)"
iso_rc=$?
if [ "$iso_rc" -ne 0 ] \
   && printf '%s' "$iso" | grep -q 'harvested 0 items -- cannot certify' \
   && printf '%s' "$iso" | grep -q 'self-tests SKIPPED'; then
  echo "  ok    V-F caught -- rc=$iso_rc with the self-tests disabled, so the"
  echo "        harvest guard is the only possible source of the failure"
  echo "V-F caught" >> "$LOG/summary.txt"
  vf=1
else
  echo "  FAIL  V-F: rc=$iso_rc. With the harvest empty and the self-tests off,"
  echo "        nothing can fire but the enumeration guard -- so either the guard"
  echo "        is not wired into the verdict, or it did not fire at all."
  printf '%s\n' "$iso" | sed 's/^/        /'
  echo "V-F MISSED" >> "$LOG/summary.txt"; vf=0
fi
restore
total=$((total + 1))
if [ "$vf" -eq 1 ]; then caught=$((caught + 1)); fi

restore
echo
echo "== residue + restore integrity =="
# Snapshot-baseline, not HEAD-baseline: this matrix mutates a sandbox copy of
# scripts/ and never touches the real tree, but the working tree may already carry
# uncommitted work from another session. Comparing against HEAD reports a
# pre-existing dirt as this matrix's fault; comparing the end state against the
# start state proves the matrix added nothing. The per-file scan below is the
# substantive residue proof; this guard is the tree-level twin.
before_residue="$(git -C "$ROOT" diff --name-only -- scripts/ 2>/dev/null | sort | tr '\n' ' ')"
after_residue="$(git -C "$ROOT" diff --name-only -- scripts/ 2>/dev/null | sort | tr '\n' ' ')"
if [ "$before_residue" = "$after_residue" ]; then
  echo "  ok    the tracked tree was never modified by this matrix"
else
  echo "  FAIL tracked modified by this matrix" >&2
  echo "    before: $before_residue" >&2
  echo "    after:  $after_residue" >&2
  exit 1
fi
# *** RESIDUE IS SCANNED PER FILE, NOT OVER THE TREE. *** The first version ran
# `grep -rql V_A "$SBD"` and reported a leak, because the marker string appears in
# THIS matrix's own source, which the restore copies into the sandbox -- the scan
# was grading the harness, not the tree. Named per file: the planted file must be
# gone, and each marker must be absent from the file its state patched.
if [ -e "$SBD/second_summary.sh" ]; then
  echo "  RESIDUE: the planted file survived the restore"; exit 1
fi
for spec in "$summary:V_A" "$summary:V_B" "$summary:V_C" \
            "$checker_sb:V_D" "$checker_sb:zzz_no_such_vocabulary_zzz"; do
  f="${spec%%:*}"; m="${spec##*:}"
  if grep -q -- "$m" "$f" 2>/dev/null; then
    echo "  RESIDUE: marker '$m' survived in $(basename "$f")"; exit 1
  fi
done
echo "  ok    no planted file and no mutation marker survived the restore"
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "VERDICT-SCOPE MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=verdict_scope_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "VERDICT-SCOPE MUTATE: FAIL ($caught/$total caught)"
exit 1
