#!/usr/bin/env bash
# Mutation matrix for scripts/check_engine_cert.sh (AUDIT #24).
#
# *** THE REAL ARTIFACT IS NEVER TOUCHED. *** The checker honours
# INTFER_BUILD_DIR, so a sandbox build dir is all this needs. An earlier draft
# planted into build/ten_gates/HEAD and restored from a backup; that works, but an
# abnormal exit can leave a FABRICATED CERTIFICATION on disk for the next session
# to read -- the worst possible residue for this particular artifact.
#
# The mutant lives in scripts/ (not /tmp) so SCRIPT_DIR and ROOT still resolve to
# the REAL tree: the staleness verdict needs `git diff`, which a copy under /tmp
# cannot do. The leading underscore keeps it invisible to every check_*/mutate_*
# glob, so the copy cannot pollute the population the stack enumerates.
#
# Per AUDIT #21, checker-weakening is normally void because the artifact is
# correct -- but E-D weakens the checker while the planted artifact is
# DELIBERATELY STALE, and that is exactly the case where weakening becomes
# observable.
#
# Usage: scripts/mutate_engine_cert.sh
# Exit: 0 = every state behaved as characterized, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/engine_cert_mutate"
LOG="$STAMP/states"
ART="$BUILD_DIR/ten_gates/HEAD"
CHECK="$SCRIPT_DIR/check_engine_cert.sh"

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
if [ ! -f "$ART" ]; then
  echo "SKIP: no $ART -- run scripts/run_ten_gates.sh first. The baseline needs a"
  echo "      healthy artifact to be meaningful, so there is nothing to preserve."
  exit 2
fi
git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null || {
  echo "FAIL: scripts/ is dirty; this matrix edits a sandbox copy."; exit 2; }

mkdir -p "$LOG"; : > "$LOG/summary.txt"
SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_ec.XXXXXX")"
mkdir -p "$SB/ten_gates"
MUT="$SCRIPT_DIR/_ec_mutant.sh"
ORIG_C="$SB/.orig_check"; cp "$CHECK" "$ORIG_C"
cp "$CHECK" "$MUT"
trap 'rm -f "$MUT"; rm -rf "$SB"' EXIT

run_check() { INTFER_BUILD_DIR="$SB" bash "$MUT" 2>&1; }
plant() { printf 'HEAD=%s\npass=10 partial=0 skip=0 fail=0\n' "$1" > "$SB/ten_gates/HEAD"; }
restore_check() { cp "$ORIG_C" "$MUT"; }

OLD="$(git -C "$ROOT" rev-list --max-parents=0 HEAD | head -1)"
HEAD_SHA="$(git -C "$ROOT" rev-parse HEAD)"
HEAD_PARENT="$(git -C "$ROOT" rev-parse HEAD~1)"

echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

echo
echo "== baseline: a healthy artifact must be accepted =="
plant "$HEAD_SHA"
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -5; exit 1
fi
echo "  ok    baseline passes: $(printf '%s\n' "$b" | sed -n 's/^RESULT: //p')"
echo "  ok    CONVERSE: a healthy certification is accepted, so an always-red"
echo "        checker cannot score as caught"

caught=0; total=0
check_state() { # $1 id $2 expected-rc $3 expected-RE $4 why
  local id="$1" erc="$2" ere="$3" why="$4"
  total=$((total+1)); restore_check
  local o r; o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  if [ "$r" -ne "$erc" ]; then
    echo "  $id MISSED -- rc=$r, want $erc ($why)"
    echo "$id MISSED (rc=$r want $erc)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_present "$id property" "$o" "$ere"; then
    echo "  $id MISSED -- rc right but marker absent: $ere"
    echo "$id MISSED (marker absent)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-present ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states =="

# E-A: a TOOLING-only commit must NOT invalidate a 45-minute engine run. This is
# the state that stops the check from being so aggressive that people disable it.
plant "$HEAD_PARENT"
check_state "E-A" 0 'carries to HEAD' "a tooling-only commit must not invalidate the engine run"

# E-B: an ENGINE input moving must invalidate it. The load-bearing state.
plant "$OLD"
check_state "E-B" 1 'certification STALE' "an engine input changed since the certification"

# E-C: no artifact is UNPROVEN, never a silent pass and never a false FAIL.
rm -f "$SB/ten_gates/HEAD"
check_state "E-C" 2 'UNPROVEN' "no artifact at all is unproven, not a regression"

# E-E: an artifact naming a commit THIS REPOSITORY DOES NOT CONTAIN is UNPROVEN,
# not "carries". Added because the audit found the checker answering rc=0 "carries"
# for a 40-zero sha, printing "0 file(s) changed ... (List: )" -- it certified an
# engine it could not compare. `git diff` exits 128 with empty stdout and the old
# `|| true` turned that into an empty harvest, i.e. a clean tree.
#
# *** THIS IS AN ARTIFACT-DEFECT STATE, WHICH IS THE FORM AUDIT #21 REQUIRES. ***
# The checker here is CORRECT by construction -- every other state plants a real
# revision -- so weakening the checker would have been unobservable (0/4 is exactly
# what AUDIT #21's wasted pass looked like). The defect had to be put in the
# ARTIFACT, and the checker had to reject it.
#
# The converse is already covered by E-A: a real revision that legitimately carries
# is still accepted, so E-E cannot be satisfied by an always-refusing checker.
plant "0000000000000000000000000000000000000000"
check_state "E-E" 2 'UNPROVEN' "a cert naming a commit this repo does not contain is unmeasurable, not a certification"

# E-F: THE CONVERSE -- the new guards must not refuse a certification that is
# genuinely fine. Two distinct shas with an IDENTICAL tree (an empty commit on top
# of HEAD) mean the certified source IS this source, so carrying is CORRECT.
#
# *** THIS STATE IS WHY E-E MEANS ANYTHING, AND IT IS ALSO THE RECORD OF A GUARD
# THAT WAS WRITTEN AND THEN REMOVED. *** The fix's first version added
# enum_require_nonempty on the changed-file count -- the AUDIT #20 rule, applied
# where it does not apply. Measured: `git diff --name-only` between same-tree commits
# emits a single "\n" and no filenames, so a legitimately clean comparison harvests
# ZERO files and the guard refused valid evidence. A converse is what catches that;
# had E-E been the only state, the over-assertion would have shipped green as a
# "fix". The quiet case is the one that gets forgotten (the named rule), and here it
# is the one that had to be invented before the firing case could be trusted.
total=$((total+1))
restore_check
SAME_TREE_SHA="$(git -C "$ROOT" commit-tree "$(git -C "$ROOT" rev-parse HEAD^{tree})" \
  -p "$(git -C "$ROOT" rev-parse HEAD)" -m "engine-cert: empty commit, same tree as HEAD" 2>/dev/null || true)"
if [ -z "$SAME_TREE_SHA" ]; then
  echo "  E-F MISSED -- could not mint a same-tree commit (STATE fault)"
  echo "E-F MISSED (mint)" >> "$LOG/summary.txt"
else
  plant "$SAME_TREE_SHA"
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/E-F.log"
  if [ "$r" -eq 0 ] && pred_expect_present "E-F property" "$o" 'carries to HEAD'; then
    echo "  E-F caught -- a same-tree certification still CARRIES, so the new guards"
    echo "        reject an unresolvable sha without refusing valid evidence"
    echo "E-F caught" >> "$LOG/summary.txt"; caught=$((caught+1))
  else
    echo "  E-F MISSED -- rc=$r, want 0 'carries to HEAD'. The guards over-refuse: an"
    echo "        empty harvest between two identical trees is a legitimate carry."
    printf '%s\n' "$o" | tail -4 | sed 's/^/        /'
    echo "E-F MISSED (rc=$r)" >> "$LOG/summary.txt"
  fi
fi

# E-D: weaken the engine-input filter while the artifact is STALE.
#
# *** THE REPLACEMENT MUST MATCH NOTHING, AND THE FIRST ATTEMPT DID NOT. *** It
# weakened the filter to docs/|reports/|\.md$ -- but the artifact is planted at the
# ROOT commit, and this repo really does have doc and report changes since then, so
# the weakened filter still found "touched" files and still returned STALE. The
# state then reported "the filter is not load-bearing", which was a confident false
# conclusion about a mechanism that HAD been broken. A weakening must be provably
# total, not merely weaker.
#
# `total` was also not incremented on this branch, so the tally said 3/3 PASS while
# the summary listed E-D MISSED -- the self-contradicting-report class this repo
# has now fixed three times, reintroduced by a state that bypasses the shared
# check_state() helper. Every state now increments `total`, by construction here.
total=$((total+1))
restore_check; plant "$OLD"
if ! pred_expect_present "E-D patch" "$(cat "$MUT")" 'examples/'; then
  echo "  E-D MISSED -- the input-filter anchor is gone (STATE fault)"
  echo "E-D MISSED (anchor)" >> "$LOG/summary.txt"
else
  MUT="$MUT" python3 -c '
import os, sys
p = os.environ["MUT"]
s = open(p).read()
old = "grep -E '"'"'^(src/|examples/|include/|scripts/test_|scripts/_gate_precond\\.sh|scripts/_gate_build\\.sh|scripts/setvars\\.sh)'"'"'"
new = "grep -E '"'"'^(ZZZ_NEVER_MATCHES_THIS)'"'"'"
assert s.count(old) == 1, "anchor count %d" % s.count(old)
open(p, "w").write(s.replace(old, new))
' || { echo "  E-D MISSED -- patch did not apply (STATE fault)"; echo "E-D MISSED (patch)" >> "$LOG/summary.txt"; }
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/E-D.log"
  if [ "$r" -ne 0 ]; then
    echo "  E-D MISSED -- the weakened filter STILL rejected it, so the filter is not"
    echo "        what makes E-B catchable and the state proves nothing"
    echo "E-D MISSED (still caught)" >> "$LOG/summary.txt"
  else
    echo "  E-D caught -- with the input filter weakened the stale artifact is NO"
    echo "        longer detected (checker passed), so the filter is load-bearing"
    echo "E-D caught" >> "$LOG/summary.txt"; caught=$((caught+1))
  fi
fi

restore_check
git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null \
  && echo "  ok    the tracked checker was never modified" || { echo "  FAIL tracked modified"; exit 1; }
[ -f "$MUT" ] || echo "  ok    the sandbox mutant is removed on exit (trap)"
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "ENGINE CERT MUTATE: PASS ($caught/$total states behaved as characterized)"
  mkdir -p "$STAMP"
  { echo "stamp=engine_cert_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "ENGINE CERT MUTATE: FAIL ($caught/$total)"
exit 1