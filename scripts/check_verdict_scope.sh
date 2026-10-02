#!/usr/bin/env bash
# Assert that every spread-bounded VERDICT in the tree is issued by ONE owner, and
# that the owner's gate is computed over the data it judges.
#
# WHY THIS EXISTS (AUDIT #28, defect #1). The bench summariser computed a delta
# over the rounds two arms hold IN COMMON but gated the verdict on the BATCH-wide
# round count (the max over arms). An arm with one surviving round was therefore
# scored against a spread of 0.0 and printed
#   d=+98.0 (rounds 2) -> CLEARS the spread: report +98.0 ms (+0.60%)
# which is exactly the false claim AUDIT #26 had already retracted for a
# different experiment, arriving through a different door.
#
# *** A SINGLE FIXED INSTANCE IS NOT A CLOSED FAMILY. *** The repo's rule is
# "enumerate the family, never the files the last fix happened to touch"
# (AUDIT #7/#8/#10), and this class is the purest form of it: the defect is a
# SHAPE -- a verdict whose gate is derived from a population wider than the data
# being judged -- so it can reappear in any new summariser, and nothing in the
# tree would notice. Enumerated by hand this session: the only script that
# computes a spread/delta and then judges on it is scripts/_bench_summary.py
# (_gate_loadlog.sh's MAXIMUM is over one arm's own load samples, which is the
# intended population, not a scope slip; fold_report_counts.sh judges stamps, not
# data; bench_quiet_ab.sh and bench_chat.sh print no verdict at all). **A
# hand-enumerated "there is only one" is a citation, not a guard** -- the next
# summariser would arrive unflagged.
#
# SO THIS CHECKER MAKES THE ENUMERATION FAIL-LOUD INSTEAD:
#   1. The population is DERIVED: every NON-VERIFIER file in scripts/ that emits a
#      spread-bounded verdict (the vocabulary below) must be the OWNER or a
#      DECLARED site, so the family cannot grow without a decision.
#   2. The owner's gate must be over the PAIRED subset -- structurally pinned
#      (the round-scoped condition, plus its position relative to the verdict).
#   3. enum_require_nonempty, so "0 emitters found" is rc=1 and never a PASS.
#
# *** SCOPE, STATED BECAUSE IT IS A LIMITATION AND NOT A COINCIDENCE. *** Files
# named check_* / mutate_* are excluded as VERIFIERS: they assert on the
# vocabulary (grep patterns, fixture text) and never emit a verdict a reader could
# quote. The name prefix is load-bearing here for the same reason it is in
# gate_inventory.sh, where `ls scripts/test_*.sh` DEFINES the gate population
# (AUDIT #17, "naming is load-bearing"). The cost is stated rather than hidden: a
# check_* script that genuinely re-derived the rule would be excluded too. That
# gap is covered elsewhere -- the owner's behaviour is fixture-proven in
# check_bench_load_gate.sh case (7a) -- and a checker claiming completeness it
# does not have would be worse than one that names its scope.
#
# *** NON-VACUITY IS A PLANTED INSTANCE, NOT A CLAIM. *** (AUDIT #21: mutate the
# artifact and require rejection; AUDIT #20: a harvest that matches nothing is not
# a clean bill of health.) Case (b) plants a second emitter in a sandbox tree and
# requires rejection; case (c) plants a COMMENT that merely mentions the
# vocabulary and requires it to be IGNORED, which is the converse that stops this
# from being an always-red checker (a check that cries wolf is the mirror defect,
# AUDIT #15). Comments are stripped for membership, for the reason recorded at the
# top of check_bench_load_gate.sh.
#
# Usage: scripts/check_verdict_scope.sh
# Env:   INTFER_SCRIPTS_DIR  point the derivation at another scripts/ tree
#                           (the seam that lets the cases below plant an
#                           instance; same rationale as INTFER_BUILD_DIR -- an
#                           assumption the harness cannot redirect is one it
#                           cannot test)
#         INTFER_VERDICT_SCOPE_CASES=0  skip the self-tests (c) and (d).
#                           *** REQUIRED BY THE SELF-TESTS THEMSELVES. *** Case
#                           (c) runs THIS script against a planted tree, so
#                           without the guard it runs itself, which runs itself:
#                           the first version recursed until the call timed out.
#                           The inner run grades the POPULATION ONLY, and says so
#                           in its own output so an outer run can never claim the
#                           cases ran when they did not (the `ddea91d` defect --
#                           an OK line must not claim a check ran when it skipped).
# Exit:  0 = one owner, gate scoped to the judged data, 1 = drift, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SD="${INTFER_SCRIPTS_DIR:-$ROOT/scripts}"
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
enum_reset

# The verdict vocabulary this class speaks. Deliberately specific: "median" or
# "spread" alone would match prose and unrelated arithmetic, and a classifier that
# cries wolf is worse than none.
VOCAB="CLEARS the spread|INSIDE the spread|NO SPREAD ESTIMATE"
OWNER="_bench_summary.py"
# Non-owner emitters that are legitimate, as "file why". EMPTY ON PURPOSE: an
# empty list against a harvested population of one is the proof the derivation
# can see the owner at all, rather than a coincidence of the classifier.
DECLARED=""

[ -d "$SD" ] || { echo "SKIP: no scripts dir: $SD"; exit 2; }
OWN="$SD/$OWNER"
[ -f "$OWN" ] || { echo "FAIL: owner missing: $OWN"; exit 1; }

strip_comments() { sed -e 's/[[:space:]]*#.*$//'; }

# site_class <file> -> not-a-site | verifier | owner | declared | undeclared
# ONE classification, used by the population sweep AND by the self-tests, so the
# self-tests exercise the code that runs in production rather than a second copy
# of it (the AUDIT #3/#12 shape, appearing inside a checker).
site_class() {
  local f="$1" b
  b="$(basename "$f")"
  # Comments are not mechanisms; strip them before deciding membership.
  strip_comments < "$f" | grep -qE "$VOCAB" || { echo not-a-site; return 0; }
  case "$b" in
    check_*|mutate_*) echo verifier; return 0 ;;
  esac
  [ "$b" = "$OWNER" ] && { echo owner; return 0; }
  printf '%s' "$DECLARED" | grep -q "^$b " && { echo declared; return 0; }
  echo undeclared
  return 0
}

bad=0
echo "== VERDICT-SCOPE CHECK (AUDIT #28 defect #1, closed as a family) =="
echo "   scripts dir: $SD"

# ---- (a) the population is DERIVED, and the owner is in it -----------------
echo "-- (a) derived population of spread-bounded verdicts --"
emitters=""
for f in "$SD"/*.sh "$SD"/*.py; do
  [ -f "$f" ] || continue
  b="$(basename "$f")"
  case "$(site_class "$f")" in
    not-a-site) continue ;;
    verifier) echo "  ok    $b is a VERIFIER (asserts the vocabulary, emits none)"
              continue ;;
    owner)    echo "  ok    $b is the OWNER" ;;
    declared) echo "  ok    $b is a DECLARED non-owner site" ;;
    undeclared)
      printf '  %-4s %-34s emits a spread-bounded verdict and is neither the\n' "FAIL" "$b"
      echo "        owner nor a declared site. Either it re-derives the rule"
      echo "        (AUDIT #3/#12 shape) or it is legitimate -- in which case add it"
      echo "        to DECLARED with a reason. Do NOT silence it by renaming."
      bad=$((bad + 1)) ;;
  esac
  emitters="$emitters $b"
done
# *** AND THE HARVEST MUST HAVE FOUND SOMETHING. *** Otherwise this checker is a
# tautology that passes on a tree where the search is broken -- AUDIT #20.
# enum_require_nonempty, NOT enum_require_min: the first takes the reason as its
# THIRD argument. Passing four arguments (a minimum AND a prose reason) is the
# "argument accepted and never forwarded" defect, and it cost this state its
# marker: V-E went red for a different reason and scored UNATTRIBUTED, because
# the prose I thought I was printing was never printed anywhere.
enum_require_nonempty "verdict-emitters" "$(echo $emitters | wc -w)" \
  "no spread-bounded verdict found anywhere: the search is broken, not clean"

# ---- (b) the owner's gate is over the PAIRED subset, structurally ----------
echo "-- (b) the owner's gate is computed over the data it judges --"
n_pair="$(python3 -c 'import sys;print(open(sys.argv[1]).read().count(sys.argv[2]))' \
          "$OWN" 'if len(common) < 2:')"
n_batch="$(python3 -c 'import sys;print(open(sys.argv[1]).read().count(sys.argv[2]))' \
           "$OWN" 'if rounds < 2:')"
if [ "$n_pair" -eq 1 ] && [ "$n_batch" -eq 1 ]; then
  echo "  ok    exactly one paired-round gate and one single-round header gate"
else
  echo "  FAIL  paired-round gate count=$n_pair (want 1), batch-wide gate count="
  echo "        $n_batch (want 1). The paired-round gate is what stops a one-round"
  echo "        delta clearing a noise floor of 0.0 (AUDIT #28 defect #1)."
  echo "        Re-point it; do not delete it."
  bad=$((bad + 1))
fi
# *** ONE OF EACH VERDICT BRANCH, NOT A CLAIM ABOUT THEIR ORDER. *** The first
# version asserted that "NO SPREAD ESTIMATE" appears BEFORE "CLEARS the spread",
# and that was FALSE-RED, measured on the first run: the summary PRINTS an
# explanatory header line containing the phrase "CLEARS the spread", so the first
# mention of each string is never the verdict I meant. The property worth pinning
# is the one a COPY would break -- exactly ONE refusal branch and ONE clearing
# branch -- so there is no second, unscoped site inside the owner itself. The
# behavioural proof (a one-round pair never CLEARS) lives in
# check_bench_load_gate.sh case (7a) and is deliberately not duplicated here.
n_refuse="$(python3 -c 'import sys;print(open(sys.argv[1]).read().count(sys.argv[2]))' \
            "$OWN" 'v = ("NO SPREAD ESTIMATE:')"
n_clear="$(python3 -c 'import sys;print(open(sys.argv[1]).read().count(sys.argv[2]))' \
           "$OWN" 'v = ("CLEARS the spread (floor=')"
if [ "$n_refuse" -eq 1 ] && [ "$n_clear" -eq 1 ]; then
  echo "  ok    exactly one refusal branch and one clearing branch in the owner"
else
  echo "  FAIL  owner has $n_refuse refusal branch(es) and $n_clear clearing"
  echo "        branch(es); want 1 each. A second clearing branch is a second,"
  echo "        possibly unscoped, site for the rule -- exactly AUDIT #28 #1."
  bad=$((bad + 1))
fi

# ---- (c) NON-VACUITY: a planted SECOND EMITTER must be rejected ------------
if [ "${INTFER_VERDICT_SCOPE_CASES:-1}" != "1" ]; then
  echo "-- (c)/(d) self-tests SKIPPED (INTFER_VERDICT_SCOPE_CASES=0) --"
  echo "        grading the population only; this run certifies NO non-vacuity."
else
echo "-- (c) non-vacuity: a planted second emitter is REJECTED --"
SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_vscope.XXXXXX")"
trap 'rm -rf "$SB"' EXIT
mkdir -p "$SB/scripts"
cp "$SD"/*.sh "$SD"/*.py "$SB/scripts/" 2>/dev/null
# A second summariser, correct in every way except that nobody decided about it.
cat > "$SB/scripts/planted_summary.sh" <<'EOF'
noise=$(awk '{if($1>m)m=$1}END{print m}' "$LOG")
med=$(printf '%s\n' "$VALUES" | sort -n | awk '{a[NR]=$1}END{print a[int((NR+1)/2)]}')
if [ "$med" -gt "$noise" ]; then echo "CLEARS the spread"; fi
EOF
out="$(INTFER_SCRIPTS_DIR="$SB/scripts" INTFER_VERDICT_SCOPE_CASES=0 \
       bash "$SCRIPT_DIR/check_verdict_scope.sh" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'planted_summary.sh'; then
  echo "  ok    rejected, and it named planted_summary.sh (the family cannot grow silently)"
else
  echo "  FAIL  a planted second emitter passed (rc=$rc). That is a tautology:"
  echo "        the check would stay green while a second site re-derives the rule."
  printf '%s\n' "$out" | sed 's/^/        /'
  bad=$((bad + 1))
fi
rm -f "$SB/scripts/planted_summary.sh"

# ---- (d) THE CONVERSE: a COMMENT mentioning the vocabulary is NOT a site ----
echo "-- (d) converse: a comment-only mention is IGNORED --"
cat > "$SB/scripts/comment_only.sh" <<'EOF'
#!/usr/bin/env bash
# The old summary printed "CLEARS the spread" from a heredoc; that is why this
# file mentions the phrase in a comment and must NOT be classified as a site.
echo "no verdict here"
EOF
out="$(INTFER_SCRIPTS_DIR="$SB/scripts" INTFER_VERDICT_SCOPE_CASES=0 \
       bash "$SCRIPT_DIR/check_verdict_scope.sh" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && ! printf '%s' "$out" | grep -q 'comment_only'; then
  echo "  ok    ignored: a prose mention is not a mechanism"
else
  echo "  FAIL  a comment-only mention was classified as a site (rc=$rc). This"
  echo "        checker would then fire on prose, and would be ignored for that."
  bad=$((bad + 1))
fi
rm -f "$SB/scripts/comment_only.sh"
fi   # end INTFER_VERDICT_SCOPE_CASES

echo
# *** THE ENUMERATION GUARD'S VERDICT MUST REACH THIS ONE. *** _gate_enum.sh sets
# _enum_bad in the CALLER's shell (which is why it must not be invoked inside a
# command substitution -- the AUDIT #20 lesson), and the first version never
# folded it in: the guard PRINTED "FAIL [verdict-emitters] harvested 0 items" and
# the checker still exited 0. A guard whose verdict nobody reads is a comment
# with a shebang. The isolation proof is state V-F in mutate_verdict_scope.sh,
# which empties the harvest with the self-tests disabled so nothing else can fire.
bad=$((bad + ${_enum_bad:-0}))
if [ "$bad" -eq 0 ]; then
  echo "VERDICT-SCOPE CHECK: PASS (1 owner, gate scoped to the judged data,"
  echo "  derived population, and a planted second emitter is refused)"
  echo "RESULT: no spread-bounded verdict is issued outside scripts/_bench_summary.py"
  echo "RESULT: and its own gate is over the rounds the two arms hold in common"
  exit 0
fi
echo "VERDICT-SCOPE CHECK: FAIL ($bad problem(s))"
echo "RESULT: a verdict is issued by an undeclared site, or its gate is not the"
echo "RESULT: judged subset. AUDIT #28 defect #1 is a shape, not an instance."
exit 1
