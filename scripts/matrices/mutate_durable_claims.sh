#!/usr/bin/env bash
# mutate_durable_claims.sh -- the negative direction for check_durable_claims.sh.
# A predicate that can only pass is not a predicate.
#
# WHY THE SUBJECT IS A TEMP ROOT AND NOT THE LIVE RECORDS. The checker's whole
# subject is the durable prose (HISTORY.md, AGENTS.md, every docs/*.md record and
# every docs/engines/*.txt survey-history record) plus the sha of every TRACKED file,
# because p_pair_before
# resolves a pair's BEFORE half against the tracked population rather than against
# scripts/ alone. Mutating the live records in place would rewrite the very history
# this checker exists to protect, so the matrix stamps a COMPLETE fake root -- every
# record on the surface plus every script -- and grades the checker against that.
# The live files are only ever READ (they are copied in), and the residue assertion
# at the end proves they came out byte-identical.
#
# THE TEMP ROOT IS NOT A GIT REPO, so the checker's `git -C "$ROOT" ls-files` finds
# nothing there and p_pair_before reports an empty tracked population. That is the
# checker failing closed on a population it could not resolve, and the matrix's
# baseline tolerates it because the baseline is graded on the PROSE predicates; what
# the baseline must show is a PASS, never a crash. Recorded so the interplay is
# visible rather than rediscovered as a mystery red.
#
# THE THREE STANDING RULES from docs/audit-log-tooling.md failure mode #36, enforced
# by construction rather than intended:
#   * every mutation asserts it LANDED, with cmp, before it is graded -- a no-op
#     mutation is a FAIL, never a catch;
#   * every state names the predicate it must trip, and the refusal reason is
#     HARVESTED from the checker's output, so a state caught for the wrong reason is
#     visible rather than counted;
#   * the baseline runs first: an unmutated temp root must PASS the checker, or every
#     "catch" below is indistinguishable from a checker that refuses everything.
#
# THE STATES, each named for the predicate it must trip:
#   DC-comment-only  HISTORY.md becomes comment-only          -> corpus refusal
#   DC-no-pairs      the sha pair is erased from every record  -> p_pair_harvest
#   DC-fake-before   the pair's BEFORE sha is forged          -> p_pair_before
#   DC-no-disp       the pair's disposition marker is removed -> p_pair_regen
#   DC-phantom       REGEN names a script that does not exist -> p_pair_hook
#   DC-no-hook       REGEN names a script with no equivalence hook -> p_pair_hook
#   DC-engines-only  the pair survives only in the live-state record -> p_convention
#   DC-unmarked-cite one /var/tmp citation loses its marker   -> p_tmp_regen
#   DC-no-cites      every /var/tmp citation is erased         -> p_tmp_regen vacuity
#   DC-unmarked-tally one matrix tally loses its marker        -> p_tally_regen
#   DC-docs-unmarked the dispositions are stripped from the newly-scoped records
#                    (AGENTS.md + docs/*.md) ONLY               -> p_tally_regen
#   DC-no-tallies    every matrix tally is erased              -> p_tally_regen vacuity
#
# DC-docs-unmarked is the state that certifies THE WIDENING ITSELF. Every other tally
# state corrupts a record the old two-file matrix already covered, so the pre-widening
# checker would catch all of them; this one can only be caught by a harvest that
# actually opens AGENTS.md and docs/*.md. It is deliberately one-sided -- the old
# records keep their markers -- so the state fails if the corpus silently collapses
# back to two files.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$ROOT/scripts/checks/check_durable_claims.sh"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/dcmut.XXXXXX")"
TR="$SANDBOX/root"

die() { echo "DC-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }
note_ok() { echo "DC-MUT: $1 CAUGHT (${2:-no refusal reason harvested})"; }

[ -f "$ROOT/HISTORY.md" ] || die "live record missing ($ROOT/HISTORY.md)"
[ -f "$ROOT/AGENTS.md" ] || die "live record missing ($ROOT/AGENTS.md) -- the durable surface this matrix certifies includes it"
[ -s "$ROOT/docs/engines/SOURCES.txt" ] || die "the engine-source index is missing ($ROOT/docs/engines/SOURCES.txt)"
[ -f "$CHECKER" ] || die "checker missing ($CHECKER)"

# ---- the pair subjects, derived from the live records ------------------------
# Defined AFTER surface(), because pairs_in_live_records iterates over it and bash
# resolves a function name at CALL time: calling it before the definition would make
# bash look for a FILE named `surface` and die with a confusing path error. The
# helpers are derived, not hand-listed, so the fake root grants the checker exactly
# the population the PROSE names and nothing else.
pairs_in_live_records() {
  local f
  for f in $(surface); do
    tr '\n' ' ' < "$f" | tr -s ' ' | grep -oE '[0-9a-f]{64} -> [0-9a-f]{64}'
  done | sort -u
}
# _pair_subject_candidates: the live paths a pair's BEFORE could name. All TRACKED
# files, symlinks excluded, because following the model fixtures would hash tens of
# GB to answer a question about a path. Cost is bounded and MEASURED: hashing the
# full live population took ~7 ms, and it runs once per matrix, not per state.
_pair_subject_candidates() {
  local s
  while IFS= read -r -d '' s; do
    [ -L "$ROOT/$s" ] && continue
    [ -f "$ROOT/$s" ] || continue
    printf '%s\n' "$s"
  done < <(git -C "$ROOT" ls-files -z 2>/dev/null)
}

# ---- stamp a complete fake root ---------------------------------------------
# Every script is copied because p_pair_hook resolves the REGEN target inside it,
# and every tracked file is INDEXED because p_pair_before resolves a pair's BEFORE
# half against the tracked population. A partial copy would make the checker fail
# for the wrong reason -- which is what the widening itself did on its first run:
# the temp root has no .git, so `git -C "$TR" ls-files` returned nothing, the
# checker reported an empty tracked population and REFUSED the baseline, and every
# catch below would have been graded against a checker that fails for its own
# scaffolding. Hence the `git init` below: it makes the temp root's population what
# the checker's predicate actually asks about (a set of indexed paths) without
# requiring the whole tree to be copied.
mkdir -p "$TR/scripts" "$TR/docs" "$TR/docs/engines"
cp -rf "$ROOT"/scripts/. "$TR/scripts/" 2>/dev/null || true
cp -f "$ROOT/HISTORY.md" "$ROOT/AGENTS.md" "$TR/" || true
cp -f "$ROOT"/docs/*.md "$TR/docs/" 2>/dev/null || true
cp -f "$ROOT"/docs/engines/*.txt "$TR/docs/engines/" 2>/dev/null || true
[ -s "$TR/scripts/matrices/mutate_knob_sweep_contract.sh" ] \
  || die "could not stamp the temp root (the equivalence-hook subject is missing)"
[ -s "$TR/HISTORY.md" ] || die "could not stamp the temp root (HISTORY.md missing)"
[ -s "$TR/AGENTS.md" ] || die "could not stamp the temp root (AGENTS.md missing)"
[ "$(ls -1 "$TR"/docs/*.md 2>/dev/null | wc -l)" -ge 1 ] \
  || die "could not stamp the temp root (no docs/*.md; a surface with no docs cannot
     certify the harvest that looks at them)"
# Index the temp root so the tracked-file harvest resolves inside it. If git is
# unavailable the index stays empty and the checker refuses -- failing closed.
# The index is built AFTER the pair subjects are stamped below, and the stamping is
# below surface() because it iterates over surface(); the ORDER is: stamp prose here,
# define surface(), resolve+stamp the subjects, then index.
( cd "$TR" && git init -q . ) >/dev/null 2>&1 || true
# The temp root must carry the WHOLE surface, and this asserts it rather than
# assuming it: a partial stamp makes the checker fail for a missing file, which reads
surface() {
  printf '%s\n' "$TR/HISTORY.md" "$TR/AGENTS.md"
  for _d in "$TR"/docs/*.md; do [ -f "$_d" ] && printf '%s\n' "$_d"; done
  for _e in "$TR"/docs/engines/*.txt; do [ -f "$_e" ] && printf '%s\n' "$_e"; done
}
surface_count() { surface | wc -l; }

# as a refusal for the WRONG reason and would be graded as a catch it is not.
# ---- resolve and stamp the pair subjects -------------------------------------
# Every pair's BEFORE half names a file in the LIVE tree by its sha. p_pair_before
# resolves that sha against the population it is handed, so the temp root must carry
# the NAMED FILES, not merely a scripts/ directory -- otherwise a pair legitimately
# recorded against src/engine/engine.cpp (which mutate_xmx_attrib.sh's residue
# assertion is about) is refused by a population that never had it, and the baseline
# below goes red while every state is graded against a checker failing for its own
# scaffolding. DERIVED from the live records, not hand-listed: a new pair's subject
# is stamped the day it is recorded. Placed AFTER surface() because
# pairs_in_live_records iterates it, and bash resolves the name at call time.
declare -a SUBJ_FILES=()
while IFS= read -r _p; do
  [ -n "$_p" ] || continue
  _want="${_p%% -> *}"
  # No word-splitting risk: every tracked path that can carry a pair subject is
  # whitespace-free; the one doc with spaces is a .html the pairs never name, and
  # the guard is the sha compare, not the path name.
  for _f in $(_pair_subject_candidates); do
    if [ "$(sha256sum "$ROOT/$_f" 2>/dev/null | cut -d' ' -f1)" = "$_want" ]; then
      SUBJ_FILES+=("$_f")
      break
    fi
  done
done <<< "$(pairs_in_live_records)"
if [ "${#SUBJ_FILES[@]}" -lt 1 ]; then
  die "no pair subject could be resolved in the live tree; p_pair_before cannot be
      granted a subject to resolve, so the baseline would fail for its own
      scaffolding rather than for a defect"
fi
for _f in $(printf '%s\n' "${SUBJ_FILES[@]}" | sort -u); do
  mkdir -p "$TR/$(dirname "$_f")"
  cp -f "$ROOT/$_f" "$TR/$_f"
done
( cd "$TR" && git add -A . ) >/dev/null 2>&1 || true
TRACKED_N="$(cd "$TR" && git ls-files 2>/dev/null | wc -l)"
if [ "$TRACKED_N" -lt 1 ]; then
  die "the temp root indexed $TRACKED_N file(s); p_pair_before would refuse an empty
      tracked population, so every catch below would be graded against a checker
      failing for its own scaffolding"
fi
SURF_N="$(surface_count)"
LIVE_N=2
for _d in "$ROOT"/docs/*.md; do [ -f "$_d" ] && LIVE_N=$((LIVE_N + 1)); done
for _e in "$ROOT"/docs/engines/*.txt; do [ -f "$_e" ] && LIVE_N=$((LIVE_N + 1)); done
[ "$SURF_N" -eq "$LIVE_N" ] \
  || die "the temp root carries $SURF_N durable record(s) but the live surface has $LIVE_N; a partial copy is a refusal for the wrong reason"
echo "DC-MUT: temp root stamped with $SURF_N durable record(s) + $(ls -1 "$TR"/scripts/*.sh 2>/dev/null | wc -l) script(s)"

restamp() {
  cp -f "$ROOT/HISTORY.md" "$ROOT/AGENTS.md" "$TR/" || true
  cp -f "$ROOT"/docs/*.md "$TR/docs/" 2>/dev/null || true
  cp -f "$ROOT"/docs/engines/*.txt "$TR/docs/engines/" 2>/dev/null || true
}

# the whole surface, for the states that must corrupt every record carrying a marker
# and for the LANDED/residue assertions that must not silently shrink to two files.

# ---- BASELINE. The unmutated temp root must PASS, or every catch below is
#      indistinguishable from a checker that refuses everything.
bash "$CHECKER" "$TR" >"$SANDBOX/baseline.txt" 2>&1
if [ $? -ne 0 ]; then
  cat "$SANDBOX/baseline.txt"
  die "the healthy temp root FAILS the checker"
fi
grep -q 'DC-CONTRACT: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
echo "DC-MUT: baseline PASS on the unmutated temp root"
HARVEST_TALLIES="$(grep -oE 'tallies=[0-9]+' "$SANDBOX/baseline.txt" | tail -1 | cut -d= -f2)"
HARVEST_CITES="$(grep -oE 'tmp_citations=[0-9]+' "$SANDBOX/baseline.txt" | tail -1 | cut -d= -f2)"
echo "DC-MUT: baseline harvest: /var/tmp citations=$HARVEST_CITES tallies=$HARVEST_TALLIES"

RESID_FILES="HISTORY.md AGENTS.md"
while IFS= read -r _d; do
  [ -n "$_d" ] || continue
  RESID_FILES="$RESID_FILES docs/${_d##*/}"
done < <(cd "$ROOT" && ls -1 docs/*.md 2>/dev/null)
while IFS= read -r _e; do
  [ -n "$_e" ] || continue
  RESID_FILES="$RESID_FILES docs/engines/${_e##*/}"
done < <(cd "$ROOT" && ls -1 docs/engines/*.txt 2>/dev/null)
# SUBJECT_READS is the uniform spelling of the same set RESID_FILES carries, so
# the re-run-after-the-edits rule is derivable with one grep over every matrix.
SUBJECT_READS="$RESID_FILES"
BEFORE="$(cd "$ROOT" && sha256sum $RESID_FILES | sha256sum | cut -d' ' -f1)"
echo "DC-MUT: residue covers $(printf '%s\n' $RESID_FILES | wc -l) file(s) on the durable surface"

STATES_REAL=0
STATES_CAUGHT=0
EXPECT=16

run_state() { # $1 = state name, $2 = expected predicate (or "corpus" for a hard die)
  local st="$1" want="$2"
  restamp
  local h="$TR/HISTORY.md" a="$TR/AGENTS.md"
  local -a allf=("$h" "$a")
  local _d
  for _d in "$TR"/docs/*.md; do [ -f "$_d" ] && allf+=("$_d"); done
  # THE SAME OMISSION THE mkdir FIX JUST REPAIRED, IN ITS OTHER HALF. allf is
  # the population every mutation below edits; surface() and the checker's own
  # CORPUS both enumerate docs/engines/*.txt, but this list did not, so every
  # state that erases a harvest key (DC-no-pairs, DC-no-cites, DC-no-tallies)
  # silently no-op'd on 6 of the 11 /var/tmp citations and 2 of the 4 engines
  # records: DC-no-cites harvested 6 citations post-erasure and PASSed a
  # defective corpus. DERIVED from the same glob, so it cannot drift again.
  local _e
  for _e in "$TR"/docs/engines/*.txt; do [ -f "$_e" ] && allf+=("$_e"); done
  local -a docs_list=()
  for _d in "${allf[@]}"; do
    case "$_d" in "$TR"/docs/*) docs_list+=("$_d") ;; esac
  done
  [ "${#docs_list[@]}" -ge 1 ] || die "the temp root carries no docs/*.md; DC-docs-unmarked would have nothing to strip"
  case "$st" in
    # ---- corpus: a records file that strips to nothing must be refused ---------
    DC-comment-only)
      printf '# only a comment\n# and another\n\n' > "$h" ;;
    # ---- the pair census: erasing every pair must not read as PASS -------------
    # Erase the shas themselves, not a "PAIR-ERASED" stub: the census harvest is
    # `grep -oE '[0-9a-f]{64} -> [0-9a-f]{64}'`, so a stub still leaves the corpus
    # non-empty and the state is caught by a LATER predicate instead of by the
    # harvest-vacuity guard it exists to trip. Per-sha, for the same wrap reason as
    # DC-engines-only.
    DC-no-pairs)
      python3 - "${allf[@]}" <<'PY'
import sys, re
# DYNAMIC anchor, NOT hardcoded shas. This matrix previously named the 2026-10-18
# pair (4313556.../b1869bfb...) literally; that pair is no longer a full string on
# the surface (it survives only abridged in prose), so the three pair states below
# silently no-op'd and a matrix state that never lands cannot report a tally. Harvest
# whatever pair the surface carries TODAY and erase it: wrap-independent because the
# pair regex tolerates the HISTORY.md line wrap, and the stub is non-hex so the
# census grep finds zero pairs (p_pair_harvest trips, not a later predicate).
pat = re.compile(r'[0-9a-f]{64} -> [0-9a-f]{64}')
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    out = pat.sub('ERASED -> ERASED', s)
    if out != s:
        open(p, 'w').write(out)
        hit += 1
assert hit >= 1, 'no pair found anywhere on the surface to erase'
PY
      ;;
    # ---- a fabricated BEFORE sha is visible without running anything ----------
    DC-fake-before)
      python3 - "$h" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
m = re.search(r'[0-9a-f]{64}( -> )', s)
assert m, 'no pair BEFORE sha found to forge'
out = s[:m.start()] + '0' * 64 + s[m.start() + 64:]
assert out != s, 'the BEFORE-sha forge did not land'
open(p, 'w').write(out)
PY
      ;;
    # ---- the pair must carry a disposition ------------------------------------
    DC-no-disp)
      python3 - "${allf[@]}" <<'PY'
import re, sys
for p in sys.argv[1:]:
    s = open(p).read()
    # remove EVERY REGEN/EPHEMERAL(...) marker: no disposition anywhere on the
    # surface, or a marker surviving in a doc keeps the pair predicate satisfied
    open(p, 'w').write(re.sub(r'(REGEN|EPHEMERAL)\([^)]*\)', '', s))
PY
      ;;
    # ---- a REGEN naming a command that was never shipped ----------------------
    # BOTH records, for the same reason as DC-no-hook: the checker accepts a
    # disposition from EITHER record, so a one-file corruption is not a defect from
    # the predicate's point of view and would be graded as an unlanded state.
    DC-phantom)
      python3 - "${allf[@]}" <<'PY'
import re, sys
PAIR = re.compile(r'[0-9a-f]{64} -> [0-9a-f]{64}')
hit = 0
for p in sys.argv[1:]:
    lines = open(p).read().split('\n')
    changed = False
    for i, ln in enumerate(lines):
        # TARGET THE PAIR'S DISPOSITION, paren-count-agnostic. The old anchor named
        # a double-paren spelling (`...sh))`) that matches only outer prose
        # parentheses `(REGEN(bash scripts/matrices/mutate_knob_sweep_contract.sh))`, NOT the
        # pair line's single-paren disposition -- so it corrupted 4 prose mentions
        # and left the real disposition intact, grading 'wrong reason' or nothing.
        if PAIR.search(ln) and 'scripts/matrices/mutate_knob_sweep_contract.sh' in ln:
            lines[i] = ln.replace('scripts/matrices/mutate_knob_sweep_contract.sh',
                                  'scripts/zz_never_shipped.sh')
            changed = True
    if changed:
        open(p, 'w').write('\n'.join(lines))
        hit += 1
assert hit >= 1, 'the pair disposition marker was not found on any surface record'
PY
      ;;
    # ---- a REGEN naming a real script that carries no equivalence hook --------
    # BOTH records, because the checker accepts a disposition from EITHER one: it
    # stops at the first marker it finds, so a one-file mutation is a no-op against
    # the predicate and would be graded as an unlanded state. (That the checker
    # settles for "either record carries it" is deliberate -- the two records
    # corroborate -- so a state for it must corrupt both.)
    DC-no-hook)
      python3 - "${allf[@]}" <<'PY'
import re, sys
PAIR = re.compile(r'[0-9a-f]{64} -> [0-9a-f]{64}')
HOOK_TARGET = 'scripts/checks/check_s1_predicates.sh'
hit = 0
for p in sys.argv[1:]:
    lines = open(p).read().split('\n')
    changed = False
    for i, ln in enumerate(lines):
        # Point the PAIR's disposition at a real script that carries NO residue
        # equivalence hook -- paren-count-agnostic (replaces the script name), so it
        # targets the single-paren disposition on the pair line, not prose.
        if PAIR.search(ln) and 'scripts/matrices/mutate_knob_sweep_contract.sh' in ln:
            lines[i] = ln.replace('scripts/matrices/mutate_knob_sweep_contract.sh', HOOK_TARGET)
            changed = True
    if changed:
        open(p, 'w').write('\n'.join(lines))
        hit += 1
assert hit >= 1, 'the pair disposition marker was not found on any surface record'
PY
      ;;
    # ---- live state without chronology: exactly how an invented figure looks ---
    # Per-sha replacement, NOT a pair regex: HISTORY.md wraps its pair across two
    # indented lines, so re.sub('A -> B') cannot match the raw file and the state
    # would silently no-op -- which the cmp guard then reports as NOT LANDED rather
    # than as a catch. Erasing each half separately is wrap-independent.
    DC-engines-only)
      python3 - "$h" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
pat = re.compile(r'[0-9a-f]{64} -> [0-9a-f]{64}')
out, n = pat.subn('A-ERASED-FROM-CHRONOLOGY -> B-ERASED-FROM-CHRONOLOGY', s)
assert out != s, 'the pair shas to erase were not found in the chronology'
open(p, 'w').write(out)
PY
      ;;
    # ---- one citation loses its marker ----------------------------------------
    DC-unmarked-cite)
      python3 - "$h" <<'PY'
import re, sys
p = sys.argv[1]
lines = open(p).read().split('\n')
# LINE-TARGETED, and that is load-bearing: the defect shape is "one /var/tmp
# citation loses its marker", so the mutation must strip the marker from the line
# that CITES the artifact. A plain count=1 re.sub over the whole file removed the
# first occurrence of the marker anywhere -- which lives on a matrix-TALLY line
# -- and the checker then refused on p_tally_regen instead of p_tmp_regen. The
# state was still CAUGHT, but for the wrong predicate, which is the shape the
# mutator reports rather than counts. Anchoring on the citation line keeps the
# state aimed at the predicate it exists to trip.
for i, ln in enumerate(lines):
    if '/var/tmp/' in ln and 'REGEN(' in ln:
        new = re.sub(r'REGEN\([^)]*\)', '', ln, count=1)
        assert new != ln, 'the marked /var/tmp line carried a REGEN that did not strip'
        lines[i] = new
        open(p, 'w').write('\n'.join(lines))
        print('DC-unmarked-cite: stripped the disposition from line ' + str(i + 1))
        sys.exit(0)
sys.stderr.write('REFUSE: no /var/tmp citation line carrying a REGEN was found\n')
sys.exit(1)
PY
      ;;
    # ---- the vacuity half: an empty harvest must FAIL, not pass ---------------
    DC-no-cites)
      python3 - "${allf[@]}" <<'PY'
import re, sys
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    out = re.sub(r'/var/tmp/[A-Za-z0-9._/-]+', 'ARTIFACT-ERASED', s)
    if out != s:
        open(p, 'w').write(out)
        hit += 1
assert hit >= 1, 'no /var/tmp citation found anywhere on the surface to erase'
PY
      ;;
    DC-unmarked-tally)
      python3 - "$h" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
# Same re-anchor as DC-unmarked-cite: the old pattern carried the wrapping
# parentheses of the marker spelling used in the destroyed lines, and the live
# surface now spells it unwrapped.
out = re.sub(r'REGEN\(bash scripts/matrices/mutate_pending_operator\.sh\)', '', s, count=1)
assert out != s, 'the tally REGEN marker to remove was not found'
open(p, 'w').write(out)
PY
      ;;
    DC-docs-unmarked)
      python3 - "$a" "${docs_list[@]}" <<'PY'
import re, sys
# Strip the dispositions from the NEWLY-SCOPED RECORDS ONLY -- AGENTS.md and
# docs/*.md -- deliberately leaving HISTORY.md and the docs/engines/*.txt records marked.
# A harvest scoped to the old two files sees a fully-disposed record and PASSES,
# so this state can only be caught by a checker that opens the widened surface.
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    out = re.sub(r'(REGEN|EPHEMERAL)\([^)]*\)', '', s)
    if out != s:
        open(p, 'w').write(out)
        hit += 1
assert hit >= 1, 'no disposition found in the newly-scoped records to strip'
PY
      ;;
    DC-no-tallies)
      python3 - "${allf[@]}" <<'PY'
import re, sys
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    # 'states' is OPTIONAL: the checker's harvest matches the bare "N/N CAUGHT" form
    # too, so erasing only the longer wording would leave the short-form tallies intact
    # and the state would pass the checker -- a vacuity direction that is not vacuous.
    out = re.sub(r'[0-9]+/[0-9]+ (states )?[Cc][Aa][Uu][Gg][Hh][Tt]', 'TALLY-ERASED', s)
    if out != s:
        open(p, 'w').write(out)
        hit += 1
assert hit >= 1, 'no matrix tally found anywhere on the surface to erase'
PY
      ;;
    # ---- a REGEN that resolves nowhere -----------------------------------------
    # p_tally_regen and p_runresult_regen require a marker to be PRESENT; a marker
    # naming a script that was never shipped satisfies them while certifying nothing.
    # p_pair_hook closes exactly this hole for the PAIR class, so the state is aimed
    # at a class p_pair_hook does not reach (a TALLY) and can only be caught by the
    # predicate that resolves the target itself. MEASURED GAP: before p_target_exists
    # this record PASSED the checker.
    DC-phantom-target)
      python3 - "${allf[@]}" <<'PY'
import sys
# Same re-anchor as the two "unmarked" states: the live surface spells the
# marker without the wrapping parentheses the destroyed lines used.
OLD = 'REGEN(bash scripts/matrices/mutate_pending_operator.sh)'
NEW = 'REGEN(bash scripts/zz_never_shipped.sh)'
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    if OLD in s:
        open(p, 'w').write(s.replace(OLD, NEW))
        hit += 1
assert hit >= 1, 'the tally REGEN marker to make phantom was not found anywhere on the surface'
PY
      ;;
    # ---- the run-result class ("PASS N/N ...") loses its dispositions ----------
    # Strips the run-result dispositions ONLY, by exact marker string, leaving every
    # tally, citation and pair marker in place -- so it is the analogue of
    # DC-docs-unmarked for the class this round opened: a checker that never harvested
    # run-result tallies sees a fully-disposed record and PASSES. The five marker
    # strings occur at exactly six sites, all of them run-result sites (checked by
    # enumeration, not assumed); each appears only at run-result lines, so stripping
    # every occurrence cannot trip a different class's predicate by accident.
    DC-runres-unmarked)
      python3 - "${allf[@]}" <<'PY'
import sys
MARKERS = ['REGEN(bash scripts/matrices/mutate_roofline_probe.sh)',
           'REGEN(bash scripts/gates/test_rowpair_equiv.sh)',
           'REGEN(bash scripts/checks/check_decode1_copies.sh)',
           'REGEN(bash scripts/checks/check_gate_exit_verdict.sh)',
           'REGEN(bash scripts/checks/check_destructive_git.sh)']
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    out = s
    for m in MARKERS:
        out = out.replace(m, '')
    if out != s:
        open(p, 'w').write(out)
        hit += 1
assert hit >= 1, 'no run-result disposition found anywhere on the surface to strip'
PY
      ;;
    # ---- the un-harvested ratio class GROWS ------------------------------------
    # The class is not harvested, so no predicate fails on its CONTENT -- what fails is
    # the RECONCILIATION: the checker derives the class every run and compares it to the
    # number in its own header. Planting one new ratio occurrence is therefore the state
    # that can only be caught by a checker that opens the class at all. It is aimed at a
    # docs record, so the LANDED guard's whole-surface cmp is what proves it landed.
    DC-ratio-grown)
      python3 - "$TR/docs/next_plan.md" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
assert 'trend parity' not in s, 'the plant text is already present (rerun residue)'
open(p, 'w').write(s + "\nthe sweep reported 7/7 trend parity on the planted line\n")
PY
      ;;
    # ---- the un-harvested ratio class EMPTIES ----------------------------------
    # The vacuity half, and it erases ONLY the occurrences the census classifies:
    # a line carrying a "states caught" tally or a run-result is left alone, so the
    # failure is the census's own zero-harvest guard rather than a tally predicate.
    # The classification is re-implemented here deliberately -- a mutator must PLANT a
    # defect, and the LANDED check plus the per-predicate grading are what catch a
    # plant that drifted from the checker.
    DC-ratio-erased)
      python3 - "${allf[@]}" <<'PY'
import re, sys
TALLY = re.compile(r'[0-9]+/[0-9]+ (states )?caught', re.I)
RUNRES = re.compile(r'PASS[ ,(]+\(?[0-9]+/[0-9]+', re.I)
RATIO = re.compile(r'([0-9]+/[0-9]+)([ \t]+[A-Za-z]+)')
hit = 0
for p in sys.argv[1:]:
    lines = open(p).read().split('\n')
    out = []
    changed = False
    for ln in lines:
        if RATIO.search(ln) and not TALLY.search(ln) and not RUNRES.search(ln):
            new = RATIO.sub(r'RATIO-ERASED\2', ln)
            if new != ln:
                changed = True
                ln = new
        out.append(ln)
    if changed:
        open(p, 'w').write('\n'.join(out))
        hit += 1
assert hit >= 1, 'no ratio-class occurrence found anywhere on the surface to erase'
PY
      ;;
    *) die "$st: unknown state" ;;
  esac

  # THE STANDING FIX: cmp against the live SURFACE proves the mutation LANDED. It is
  # looped over every record, not over the old two: a state that corrupts a file the
  # LANDED guard does not look at is graded as a no-op and reported NOT CAUGHT -- the
  # exact shape this widening closes, since states now edit AGENTS.md and docs/*.md.
  local changed=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    cmp -s "$f" "$ROOT/${f#$TR/}" || changed=1
  done < <(surface)
  if [ "$changed" -eq 0 ]; then
    echo "DC-MUT: $st NOT CAUGHT -- the mutation did not LAND (a no-op must be a FAIL, never a catch)"
    return
  fi

  STATES_REAL=$((STATES_REAL + 1))
  bash "$CHECKER" "$TR" >"$SANDBOX/$st.out" 2>&1
  if [ $? -eq 0 ]; then
    echo "DC-MUT: $st NOT CAUGHT -- the checker passed a defective record"
    sed 's/^/    /' "$SANDBOX/$st.out" | head -12
  else
    if [ "$want" = "corpus" ]; then
      # a corpus refusal is a hard die before the predicate table; accept it, but
      # only if the output names this checker
      if grep -q 'DC-CONTRACT: FAIL' "$SANDBOX/$st.out"; then
        note_ok "$st" "corpus refusal (hard die, no predicate table)"
        STATES_CAUGHT=$((STATES_CAUGHT + 1))
      else
        echo "DC-MUT: $st NOT CAUGHT -- it died, but not with this checker's refusal"
      fi
    else
      why="$(grep -oE "p_$want|p_[a-z_]*" "$SANDBOX/$st.out" | head -1)"
      # the harvested predicate must be the one this state exists to trip
      if printf '%s\n' "$(grep 'FAIL -- refused by:' "$SANDBOX/$st.out")" | grep -q "p_$want"; then
        note_ok "$st" "p_$want"
        STATES_CAUGHT=$((STATES_CAUGHT + 1))
      else
        echo "DC-MUT: $st CAUGHT FOR THE WRONG REASON -- wanted p_$want, got: ${why:-none}"
        echo "DC-MUT:   (two predicates firing at once is visible rather than counted)"
        grep 'FAIL -- refused by:' "$SANDBOX/$st.out" | sed 's/^/    /'
      fi
    fi
  fi
}

for st in DC-comment-only DC-no-pairs DC-fake-before DC-no-disp DC-phantom \
          DC-no-hook DC-engines-only DC-unmarked-cite DC-no-cites \
          DC-unmarked-tally DC-docs-unmarked DC-no-tallies \
          DC-runres-unmarked DC-phantom-target DC-ratio-grown DC-ratio-erased; do
  case "$st" in
    DC-comment-only) run_state "$st" corpus ;;
    DC-no-pairs)     run_state "$st" pair_harvest ;;
    DC-fake-before)  run_state "$st" pair_before ;;
    DC-no-disp)      run_state "$st" pair_regen ;;
    DC-phantom|DC-no-hook) run_state "$st" pair_hook ;;
    DC-engines-only) run_state "$st" convention ;;
    DC-unmarked-cite|DC-no-cites) run_state "$st" tmp_regen ;;
    DC-unmarked-tally|DC-docs-unmarked|DC-no-tallies) run_state "$st" tally_regen ;;
    DC-runres-unmarked) run_state "$st" runresult_regen ;;
    DC-phantom-target)  run_state "$st" target_exists ;;
    DC-ratio-grown|DC-ratio-erased) run_state "$st" ratio_census ;;
    *) die "no expectation registered for $st" ;;
  esac
done

if [ "$STATES_REAL" -ne "$EXPECT" ]; then
  die "expected $EXPECT states, graded $STATES_REAL -- a matrix that never ran cannot report a tally"
fi

# ---- RESIDUE. The live records must be byte-identical to their pre-run state.
AFTER="$(cd "$ROOT" && sha256sum $RESID_FILES | sha256sum | cut -d' ' -f1)"
if [ "$AFTER" != "$BEFORE" ]; then
  die "the live durable surface CHANGED during the matrix (sha $BEFORE -> $AFTER)"
fi
# they are TRACKED (committed by the operator during this session's work), so a
# working-tree modification must also be visible to git
# UNTRACKED IS EXPECTED, NOT A MUTATION. docs/engines/*.txt (12 record(s)) are
# the survey histories this tree has never committed, so `git status --porcelain`
# reports them as `??` and this supplementary traversal read every one of them as
# "an unexpected change" -- a false alarm the sha256 above already settles, since
# byte-identical is the residue contract and git's staging view is not. Only a
# TRACKED durable record leaving the ` M`/`M `/staged-modified spellings, or a
# rename/deletion (R/D), is a real residue violation.
if [ -n "$(cd "$ROOT" && git status --porcelain -- $RESID_FILES 2>/dev/null | grep -v '^ M' | grep -v '^M ' | grep -v '^MM' | grep -v '^??')" ]; then
  die "git reports an unexpected change under the durable records"
fi
echo "DC-MUT: residue byte-identical (live surface untouched: $(printf '%s\n' $RESID_FILES | wc -l) file(s))"

echo "DC-MUT: matrix $STATES_CAUGHT/$STATES_REAL states caught"
if [ "$STATES_CAUGHT" -eq "$STATES_REAL" ]; then
  echo "DC-CONTRACT MATRIX: PASS ($STATES_CAUGHT/$STATES_REAL, LANDED asserted per state, each for its own predicate)"
  echo "RESULT: PASS"
  exit 0
fi
echo "DC-CONTRACT MATRIX: FAIL"
echo "RESULT: FAIL"
exit 1
