#!/usr/bin/env bash
# mutate_durable_claims.sh -- the negative direction for check_durable_claims.sh.
# A predicate that can only pass is not a predicate.
#
# WHY THE SUBJECT IS A TEMP ROOT AND NOT THE LIVE RECORDS. The checker's whole
# subject is the durable prose (HISTORY.md, engines_sources.txt, AGENTS.md and every
# docs/*.md record) plus the sha of every script in scripts/. Mutating the live
# records in place would rewrite the very history this checker exists to protect, so
# the matrix stamps a COMPLETE fake root -- every record on the surface plus every
# script -- and grades the checker against that.
# The live files are only ever READ (they are copied in), and the residue assertion
# at the end proves they came out byte-identical.
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
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check_durable_claims.sh"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/dcmut.XXXXXX")"
TR="$SANDBOX/root"

die() { echo "DC-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }
note_ok() { echo "DC-MUT: $1 CAUGHT (${2:-no refusal reason harvested})"; }

[ -f "$ROOT/HISTORY.md" ] || die "live record missing ($ROOT/HISTORY.md)"
[ -f "$ROOT/engines_sources.txt" ] || die "live record missing ($ROOT/engines_sources.txt)"
[ -f "$ROOT/AGENTS.md" ] || die "live record missing ($ROOT/AGENTS.md) -- the durable surface this matrix certifies includes it"
[ -f "$CHECKER" ] || die "checker missing ($CHECKER)"

# ---- stamp a complete fake root ---------------------------------------------
# Every script is copied because p_pair_before hashes the whole of scripts/*.sh and
# p_pair_hook resolves the REGEN target inside it. A partial copy would make the
# checker fail for the wrong reason.
mkdir -p "$TR/scripts" "$TR/docs"
cp -f "$ROOT"/scripts/*.sh "$TR/scripts/" 2>/dev/null || true
cp -f "$ROOT/HISTORY.md" "$ROOT/engines_sources.txt" "$ROOT/AGENTS.md" "$TR/"
cp -f "$ROOT"/docs/*.md "$TR/docs/" 2>/dev/null || true
[ -s "$TR/scripts/mutate_knob_sweep_contract.sh" ] \
  || die "could not stamp the temp root (the equivalence-hook subject is missing)"
[ -s "$TR/HISTORY.md" ] || die "could not stamp the temp root (HISTORY.md missing)"
[ -s "$TR/AGENTS.md" ] || die "could not stamp the temp root (AGENTS.md missing)"
[ "$(ls -1 "$TR"/docs/*.md 2>/dev/null | wc -l)" -ge 1 ] \
  || die "could not stamp the temp root (no docs/*.md; a surface with no docs cannot
     certify the harvest that looks at them)"
# The temp root must carry the WHOLE surface, and this asserts it rather than
# assuming it: a partial stamp makes the checker fail for a missing file, which reads
surface() {
  printf '%s\n' "$TR/HISTORY.md" "$TR/engines_sources.txt" "$TR/AGENTS.md"
  for _d in "$TR"/docs/*.md; do [ -f "$_d" ] && printf '%s\n' "$_d"; done
}
surface_count() { surface | wc -l; }

# as a refusal for the WRONG reason and would be graded as a catch it is not.
SURF_N="$(surface_count)"
LIVE_N=3
for _d in "$ROOT"/docs/*.md; do [ -f "$_d" ] && LIVE_N=$((LIVE_N + 1)); done
[ "$SURF_N" -eq "$LIVE_N" ] \
  || die "the temp root carries $SURF_N durable record(s) but the live surface has $LIVE_N; a partial copy is a refusal for the wrong reason"
echo "DC-MUT: temp root stamped with $SURF_N durable record(s) + $(ls -1 "$TR"/scripts/*.sh 2>/dev/null | wc -l) script(s)"

restamp() {
  cp -f "$ROOT/HISTORY.md" "$ROOT/engines_sources.txt" "$ROOT/AGENTS.md" "$TR/"
  cp -f "$ROOT"/docs/*.md "$TR/docs/" 2>/dev/null || true
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

RESID_FILES="HISTORY.md engines_sources.txt AGENTS.md"
while IFS= read -r _d; do
  [ -n "$_d" ] || continue
  RESID_FILES="$RESID_FILES docs/${_d##*/}"
done < <(cd "$ROOT" && ls -1 docs/*.md 2>/dev/null)
BEFORE="$(cd "$ROOT" && sha256sum $RESID_FILES | sha256sum | cut -d' ' -f1)"
echo "DC-MUT: residue covers $(printf '%s\n' $RESID_FILES | wc -l) file(s) on the durable surface"

STATES_REAL=0
STATES_CAUGHT=0
EXPECT=16

run_state() { # $1 = state name, $2 = expected predicate (or "corpus" for a hard die)
  local st="$1" want="$2"
  restamp
  local h="$TR/HISTORY.md" e="$TR/engines_sources.txt" a="$TR/AGENTS.md"
  local -a allf=("$h" "$e" "$a")
  local _d
  for _d in "$TR"/docs/*.md; do [ -f "$_d" ] && allf+=("$_d"); done
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
import sys
SHAS = ['4313556ca71c4dfc74161ded2531a00d42c7f72a8800ce06261415177a280ae8',
        'b1869bfbb8677171ec8ef64eb328e483de765eec821507e10c53cf1ab3b4057d']
# accumulate, do NOT assert per file: the surface now includes records that carry no
# pair, and a per-file assert would die on the first such record AFTER having already
# rewritten the earlier ones -- a state that "lands" by accident of argv order.
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    out = s
    for x in SHAS:
        out = out.replace(x, 'ERASED')
    if out != s:
        open(p, 'w').write(out)
        hit += 1
assert hit >= 1, 'no pair sha found anywhere on the surface to erase'
PY
      ;;
    # ---- a fabricated BEFORE sha is visible without running anything ----------
    DC-fake-before)
      python3 - "$h" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'4313556ca71c4dfc74161ded2531a00d42c7f72a8800ce06261415177a280ae8',
           '0' * 64, s, count=1)
open(p, 'w').write(s)
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
import sys
OLD = 'REGEN(bash scripts/mutate_knob_sweep_contract.sh))'
NEW = 'REGEN(bash scripts/zz_never_shipped.sh))'
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    if OLD in s:
        open(p, 'w').write(s.replace(OLD, NEW))
        hit += 1
assert hit >= 1, 'the REGEN marker to corrupt was not found anywhere on the surface'
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
import sys
OLD = 'REGEN(bash scripts/mutate_knob_sweep_contract.sh))'
NEW = 'REGEN(bash scripts/check_s1_predicates.sh))'
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    if OLD in s:
        open(p, 'w').write(s.replace(OLD, NEW))
        hit += 1
assert hit >= 1, 'the REGEN marker to corrupt was not found anywhere on the surface'
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
A = '4313556ca71c4dfc74161ded2531a00d42c7f72a8800ce06261415177a280ae8'
B = 'b1869bfbb8677171ec8ef64eb328e483de765eec821507e10c53cf1ab3b4057d'
out = s.replace(A, 'A-ERASED-FROM-CHRONOLOGY')
out = out.replace(B, 'B-ERASED-FROM-CHRONOLOGY')
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
out = re.sub(r'REGEN\(bash scripts/mutate_pending_operator\.sh\)', '', s, count=1)
assert out != s, 'the tally REGEN marker to remove was not found'
open(p, 'w').write(out)
PY
      ;;
    DC-docs-unmarked)
      python3 - "$a" "${docs_list[@]}" <<'PY'
import re, sys
# Strip the dispositions from the NEWLY-SCOPED RECORDS ONLY -- AGENTS.md and
# docs/*.md -- deliberately leaving HISTORY.md and engines_sources.txt marked.
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
OLD = 'REGEN(bash scripts/mutate_pending_operator.sh)'
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
MARKERS = ['REGEN(bash scripts/mutate_roofline_probe.sh)',
           'REGEN(bash scripts/test_rowpair_equiv.sh)',
           'REGEN(bash scripts/check_decode1_copies.sh)',
           'REGEN(bash scripts/check_gate_exit_verdict.sh)',
           'REGEN(bash scripts/check_destructive_git.sh)']
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
if [ -n "$(cd "$ROOT" && git status --porcelain -- $RESID_FILES 2>/dev/null | grep -v '^ M' | grep -v '^M ')" ]; then
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
