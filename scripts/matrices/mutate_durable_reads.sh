#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_durable_reads.sh -- the checker that pins the
# DURABLE-READ population of the mutation matrices (the seven matchers whose assertions
# depend on the literal text of a durable record, each carrying SUBJECT_READS).
#
# WHY THIS MATRIX RUNS IN A TEMP ROOT RATHER THAN IN THE REPO. The checker derives its
# population from $ROOT/scripts, so a mutation must reach a COPY of scripts/ that is not
# the live tree. Copying the corpus into $SB/root and running the checker with that root
# as $1 means no state ever touches a tracked file, which makes the "live tree
# untouched" property provable by a residue check on the LIVE paths rather than by a
# clean-tree precondition. This is the opposite trade from mutate_destructive_git.sh,
# which needs a clean git root BECAUSE it edits tracked files in place; here nothing is
# edited in place, so requiring a clean tree would be a precondition the design has
# already made unnecessary. Both are recorded rather than harmonised away.
#
# WHAT IS CHECKED, and what each state attacks:
#   DR-1 UNMARKED READER       delete SUBJECT_READS from a reader -> must REFUSE
#   DR-2 DECORATIVE MARKER     add SUBJECT_READS to a non-reader   -> must REFUSE
#   DR-3 STALE SURFACE         name a non-corpus surface           -> must REFUSE
#   DR-4 STRIP IS LOAD-BEARING break the comment stripper, so a comment-only citation
#                             becomes a "reader"                   -> must REFUSE
#   DR-5 EMPTY CORPUS          the corpus fails to resolve         -> must REFUSE
#   DR-6 ZERO READERS          the read predicate is neutered      -> must REFUSE
#   DR-7 VARIABLE READ         a reader reads through a VARIABLE    -> must REFUSE
#   DR-8 GIT READ              a reader reads through GIT           -> must REFUSE
#
# DR-7 and DR-8 pin the two mechanisms that were ADDED to the checker's derivation on
# 2026-10-20, and each pins ONE of them so neither is decorative: DR-7 reads a durable
# record through a variable (`cat "$DOC"` with `DOC="AGENTS.md"`) and is INVISIBLE
# without the resolver, and DR-8 reads one through git (`git diff --quiet -- AGENTS.md`)
# and is INVISIBLE without git in the operator set. MEASURED: adding the git operators
# ALONE catches ZERO, because the member name is behind the variable -- so the two
# mechanisms are needed TOGETHER, and these two states are what makes that claim
# falsifiable rather than a sentence in a comment.
# DR-4's expected predicate is MEASURED, not predicted: breaking the stripper makes
# mutate_dpas_shape_audit.sh count as a reader, which is the exact comment-only-citation
# shape the checker exists to exclude.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/durable_reads_matrix"
TROOT="$SB/root"
# The state count this matrix asserts it runs. A DROPPED state must be a FAIL.
EXPECTED_STATES=8
# THIS MATRIX IS ITSELF A DURABLE READER -- it copies HISTORY.md,
# AGENTS.md, docs/*.md and docs/engines/*.txt into its temp root -- so it carries the
# marker like the five it certifies. Found by the checker's own first run against this
# file, not asserted: a matrix that certifies the durable-read population while being
# an undeclared member of it would be certifying a population it had misdescribed.
SUBJECT_READS="HISTORY.md AGENTS.md docs/ docs/engines/"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$TROOT/scripts" "$TROOT/docs" "$TROOT/docs/engines"

# The corpus the checker needs, copied from the LIVE tree so the matrix certifies the
# shape of the real records rather than of a synthetic stand-in.
cp -rf "$ROOT"/scripts/. "$TROOT/scripts/" || exit 2
# Each copy names its member on the SAME line as the read verb. A `for _d in A B C; do
# cp ...` form lists the members BEFORE the verb and behind a `;`, so a per-line reader
# test cannot see the dependency -- MEASURED: the matrix's own first run detected only
# `docs/` and reported the three primaries as "names X but its code never reads it".
# Writing the three copies out is both clearer and honest about what it depends on.
cp "$ROOT/HISTORY.md" "$TROOT/" || exit 2
cp "$ROOT/AGENTS.md" "$TROOT/" || exit 2
for _d in "$ROOT"/docs/*.md; do [ -f "$_d" ] && cp "$_d" "$TROOT/docs/"; done
for _e in "$ROOT"/docs/engines/*.txt; do [ -f "$_e" ] && cp "$_e" "$TROOT/docs/engines/"; done

# pristine copies of everything a state may touch, so restore is exact and residue is
# checkable over the DERIVED set rather than over the two files the first draft looked
# at.
TOUCHED="$(cd "$TROOT" && ls $(find scripts -maxdepth 2 -not -path "*/archive/*" -name '*.sh' -type f | LC_ALL=C sort) HISTORY.md AGENTS.md docs/*.md docs/engines/*.txt 2>/dev/null)"
PRISTINE="$SB/pristine"
mkdir -p "$PRISTINE"
for f in $TOUCHED; do cp "$TROOT/$f" "$PRISTINE/$(echo "$f" | tr / _)"; done
restore_all() { local f; for f in $TOUCHED; do cp "$PRISTINE/$(echo "$f" | tr / _)" "$TROOT/$f"; done; }

# THE LIVE-RESIDUE SNAPSHOT, taken BEFORE any state runs and compared AFTER them.
#
# THE PREVIOUS VERSION OF THIS CHECK WAS VACUOUS, and in two distinct ways, both of
# which are the class this repo refuses. It printed "ok live residue: no live file was
# written" from an UNCONDITIONAL ECHO -- no comparison was made at all, so the claim
# was true by construction of the echo rather than by measurement of the tree. The
# variant that copied the live file over its own pristine before `cmp` was
# SELF-FULFILLING for the same reason: the comparison could not fail, because the thing
# being compared had just been made equal to the thing it was compared against.
#
# A pre-matrix SHA SNAPSHOT is the only form that CAN fail: the snapshot is taken once,
# before any mutation exists, and compared against a fresh reading afterwards. And an
# EMPTY snapshot is refused outright, so "nothing to compare" cannot read as "nothing
# changed".
LIVE_PRE="$SB/live_pre.sha"
( cd "$ROOT" && sha256sum $(find scripts -maxdepth 2 -not -path "*/archive/*" -name '*.sh' -type f | LC_ALL=C sort) HISTORY.md AGENTS.md docs/*.md docs/engines/*.txt ) >"$LIVE_PRE" 2>/dev/null || exit 2
LIVE_N="$(grep -c . "$LIVE_PRE" 2>/dev/null || true)"
if [ "${LIVE_N:-0}" -lt 1 ]; then
  echo "  FAIL the live-residue snapshot is EMPTY -- a residue check over nothing must" >&2
  echo "        never read as a clean tree" >&2
  exit 1
fi

patch1() { python3 - "$1" "$2" "$3" <<'PYQ'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) != 1:
    print("anchor found %d times (want 1) in %s" % (s.count(o), p), file=sys.stderr); sys.exit(1)
open(p, 'w').write(s.replace(o, n, 1))
PYQ
}

# plant_synth FILE CONTENT -- write a SYNTHETIC POPULATION MEMBER into the temp root,
# so a state can grade a DERIVATION change that no live matcher exercises. It is
# removed once the state is graded, and the residue check refuses a survivor: a
# synthetic member left behind would silently enlarge the population every later state
# is graded against.
SYNTH=""
plant_synth() { printf '%s\n' "$2" >"$TROOT/scripts/$1"; SYNTH="$SYNTH $1"; }

total=0; caught=0; RESULTS=()

# run_state id expect state_fn -- THE REASON IS HARVESTED and the plant must LAND.
# A patch that fails its anchor leaves the temp tree unmutated, so grading it would
# measure a healthy tree; run_state invokes the state function first and gates the
# grading on its return code. Same mechanism mutate_destructive_git.sh records.
run_state() {
  local id="$1" expect="$2" fn="$3" out rc reason
  total=$((total + 1))
  if ! "$fn"; then
    RESULTS+=("$id MISSED '(the mutation did not LAND -- anchor drift, aborting the state)'")
    restore_all
    return
  fi
  out="$(bash "$TROOT/scripts/checks/check_durable_reads.sh" "$TROOT" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  for _s in $SYNTH; do rm -f "$TROOT/scripts/$_s"; done
  SYNTH=""
  restore_all
  reason="$(printf '%s\n' "$out" | grep -E -- "$expect" | head -1 | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
  if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS -- the checker passed a defective corpus)")
  elif [ -z "$reason" ]; then RESULTS+=("$id MISSED '(no harvested reason: refused by something OTHER than: $expect)")
  else caught=$((caught + 1)); RESULTS+=("$id CAUGHT (${reason})"); fi
}

# DR-1: the marker line itself is the anchor. xmx_attrib's marker is a single literal
# assignment, which makes it the clean single-occurrence anchor.
dr1() { python3 - "$TROOT/scripts/matrices/mutate_xmx_attrib.sh" <<'PYQ'
import sys, re
p = sys.argv[1]; s = open(p).read()
lines = s.splitlines(True)
kept = [l for l in lines if not re.match(r'^[ \t]*SUBJECT_READS=', l)]
assert len(kept) == len(lines) - 1, "marker line not found exactly once"
open(p, 'w').write(''.join(kept))
PYQ
}

# DR-2: a marker on a matcher that reads nothing is the vacuous-green shape in reverse.
# hw_constants.sh is the measured candidate -- it mentions AGENTS.md only in a comment
# and writes a SANDBOX copy, never reading a durable record.
dr2() { patch1 "$TROOT/scripts/matrices/mutate_hw_constants.sh" 'CHECK="scripts/checks/check_hw_constants.sh"' 'SUBJECT_READS="AGENTS.md"
CHECK="scripts/checks/check_hw_constants.sh"'; }

# DR-3: the marker names a surface that is not in the corpus: a lie about the dependency.
dr3() { patch1 "$TROOT/scripts/matrices/mutate_xmx_attrib.sh" 'SUBJECT_READS="docs/engines/"' 'SUBJECT_READS="engines_sources.txt"'; }

# DR-4: break the comment stripper, so prose citations become readers.
dr4() { patch1 "$TROOT/scripts/checks/check_durable_reads.sh" \
  'strip_comments() { sed -e '"'"'s/[[:space:]]#.*$//'"'"' -e '"'"'s/^[[:space:]]*#.*$//'"'"' "$1"; }' \
  'strip_comments() { cat "$1"; }'; }

# DR-5: the corpus fails to resolve, so the population claim has no subject.
dr5() { python3 - "$TROOT/scripts/checks/check_durable_reads.sh" <<'PYQ'
import sys
p = sys.argv[1]; s = open(p).read()
old = 'CORPUS="$ROOT/HISTORY.md $ROOT/AGENTS.md"'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, 'CORPUS=""')
# EACH glob re-populates CORPUS, so the corpus STILL resolved and the refusal came
# from the wrong predicate (the names check). MEASURED, not predicted: the first DR-5
# landed the primary-line patch alone and the checker reported 10 name mismatches with
# the corpus predicate silent, because docs/*.md alone satisfied ">= 3 members". The
# docs/engines/*.txt glob added to the corpus on 2026-10-20 re-opens the same hole
# (it carries eight files on its own), so BOTH globs are neutralised or the state
# would grade a corpus that resolves via the survey-history records.
old2 = 'for _d in "$ROOT"/docs/*.md; do'
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, 'for _d in "$ROOT"/NO_SUCH_DOCS/*.md; do')
old3 = 'for _e in "$ROOT"/docs/engines/*.txt; do'
assert s.count(old3) == 1, s.count(old3)
s = s.replace(old3, 'for _e in "$ROOT"/NO_SUCH_ENGINES/*.txt; do')
open(p, 'w').write(s)
PYQ
}

# DR-6: the read predicate is neutered, so the harvest finds nothing. Must be refused by
# the anti-vacuity floor, never reported as "0 readers, nothing to check".
#
# THIRD SELF-COLLISION in this matrix, and the reason the payload is a REGEX and not a
# bare token. MEASURED, not predicted: the first payload was the bare token
# ZZZ_NEVER_MATCHES_ZZZ. That token is written LITERALLY into this file -- the payload
# line itself contains it -- and this matrix is itself INSIDE the harvest population,
# because it copies the corpus and is therefore a durable reader. So the neutered test
# matched this file's own payload line, the harvest found ONE reader instead of zero,
# the anti-vacuity floor ("harvested ZERO durable-surface readers") never fired, and the
# state was graded MISSED by the ORPHAN-MARKER predicate instead -- a refusal for the
# wrong reason, which is the same defect class as the DR-1 anchor and the checker's own
# SUBJECT_READS anchoring: a harness whose own payload is indistinguishable from the
# thing it hunts.
#
# The fix is a pattern that CANNOT self-match. A regex anchored at both ends matches only
# a line whose ENTIRE content is the payload; no line of this file is exactly that,
# because the payload line carries the pattern as a SUBSTRING and not as its whole
# content. A bare token cannot have that property: any token written into this file is,
# by construction, present in it.
dr6() { python3 - "$TROOT/scripts/checks/check_durable_reads.sh" <<'PYQ'
import sys
p = sys.argv[1]; s = open(p).read()
old = 'grep -qE "(sed|grep|awk|cat|cp|sha256sum|head|tail|python3?|git (diff|show|cat-file|hash-object))[^;|]*$member|open\\(.*$member"'
assert s.count(old) == 1, s.count(old)
open(p, 'w').write(s.replace(old, 'grep -qE "^ZZZ$"'))
PYQ
}

# DR-7: the resolver. A synthetic matcher reads a durable record THROUGH A VARIABLE,
# which is the shape that was invisible before the resolver was added: the member name
# is assigned at run time and never appears next to a read operator.
dr7() { plant_synth mutate_synth_var_read.sh '#!/usr/bin/env bash
# SYNTHETIC POPULATION MEMBER -- planted by mutate_durable_reads.sh (DR-7).
# Reads a durable record THROUGH A VARIABLE: invisible without the resolver.
DOC="AGENTS.md"
cat "$DOC" >/dev/null
'; }

# DR-8: the git operator. A synthetic matcher reads a durable record through GIT, which
# is invisible while git is absent from the operator set.
dr8() { plant_synth mutate_synth_git_read.sh '#!/usr/bin/env bash
# SYNTHETIC POPULATION MEMBER -- planted by mutate_durable_reads.sh (DR-8).
# Reads a durable record through GIT: invisible while git is not a read operator.
git diff --quiet -- AGENTS.md
'; }

# baseline: the unmutated temp root must PASS, or the states are grading a checker that
# is already red and every refusal is unfalsifiable.
out="$(bash "$TROOT/scripts/checks/check_durable_reads.sh" "$TROOT" 2>&1)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "  FAIL baseline rc=$brc" >&2; printf '%s\n' "$out" | tail -8 >&2
  echo "RESULT: FAIL (baseline is red, so no state is falsifiable)" >&2; exit 1
fi
printf '%s\n' "$out" >"$SB/baseline.log"
echo "  ok baseline rc=0, PASS verdict present"

run_state DR-1 'mutate_xmx_attrib\.sh +READS the durable surface but carries no SUBJECT_READS' dr1
run_state DR-2 'mutate_hw_constants\.sh +carries SUBJECT_READS but reads NO durable surface' dr2
run_state DR-3 'SUBJECT_READS names "engines_sources.txt", which is not a corpus member' dr3
run_state DR-4 'mutate_dpas_shape_audit\.sh +READS the durable surface but carries no SUBJECT_READS' dr4
run_state DR-5 'the durable corpus did not resolve' dr5
run_state DR-6 'harvested ZERO durable-surface readers' dr6
run_state DR-7 'mutate_synth_var_read\.sh +READS the durable surface but carries no SUBJECT_READS' dr7
run_state DR-8 'mutate_synth_git_read\.sh +READS the durable surface but carries no SUBJECT_READS' dr8

# THE STATE COUNT IS ASSERTED, not merely printed. A matrix that reports "PASS (5/5)"
# after a state is deleted is the vacuous-green shape: the printed ratio is always
# consistent with whatever ran, so the ratio can never disagree with itself. The
# expected count is a CONSTANT here and the verdict is gated on it, so a dropped or
# renamed state is a FAIL rather than a smaller PASS.
if [ "$total" -ne "$EXPECTED_STATES" ]; then
  echo "  FAIL this matrix declares $EXPECTED_STATES states but ran $total -- a dropped" >&2
  echo "        state must read as a FAIL, not as a smaller ratio" >&2
  exit 1
fi

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

# residue over the TEMP root
leftover=""
for f in $TOUCHED; do
  cmp -s "$TROOT/$f" "$PRISTINE/$(echo "$f" | tr / _)" || leftover="$leftover $f"
done
if [ -n "$leftover" ]; then
  echo "  FAIL residue: these differ from their pre-matrix state:$leftover" >&2
  exit 1
fi
echo "  ok residue: all $(printf '%s\n' $TOUCHED | grep -c .) temp-root files byte-identical to their pre-matrix state"
# A synthetic population member left behind would silently enlarge the population every
# LATER state is graded against, so a survivor is a residue failure in its own right.
if ls "$TROOT"/scripts/matrices/mutate_synth_*.sh >/dev/null 2>&1; then
  echo "  FAIL residue: a synthetic population member survived the matrix" >&2
  ls -1 "$TROOT"/scripts/matrices/mutate_synth_*.sh >&2
  exit 1
fi

# residue over the LIVE tree -- the matrix must not have touched a tracked file at all.
# Compared against the PRE-MATRIX SHA SNAPSHOT taken above, which is a reading of the
# live tree from before any state existed. This comparison CAN fail.
LIVE_POST="$SB/live_post.sha"
( cd "$ROOT" && sha256sum $(find scripts -maxdepth 2 -not -path "*/archive/*" -name '*.sh' -type f | LC_ALL=C sort) HISTORY.md AGENTS.md docs/*.md docs/engines/*.txt ) >"$LIVE_POST" 2>/dev/null || exit 2
if ! cmp -s "$LIVE_PRE" "$LIVE_POST"; then
  echo "  FAIL live residue: the LIVE tree changed while the matrix ran" >&2
  diff "$LIVE_PRE" "$LIVE_POST" 2>/dev/null | head -20 >&2
  exit 1
fi
echo "  ok live residue: all $LIVE_N live file(s) byte-identical to their pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "DURABLE-READ MATRIX: PASS ($caught/$total states)"
  echo "  DR-1: a matcher that reads a durable record without SUBJECT_READS is observable"
  echo "  DR-2: a decorative marker on a non-reader is observable"
  echo "  DR-3: a marker naming a non-corpus surface is observable"
  echo "  DR-4: the comment stripper is load-bearing -- without it a comment-only"
  echo "        citation is counted as a reader and the checker refuses"
  echo "  DR-5: an unresolvable corpus is refused"
  echo "  DR-6: an emptied harvest is refused, never read as 'nothing to check'"
echo "  DR-7: a read THROUGH A VARIABLE is visible -- the resolver is load-bearing"
echo "  DR-8: a read through GIT is visible -- git is a read operator"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DURABLE-READ MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
