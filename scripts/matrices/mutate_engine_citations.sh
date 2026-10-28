#!/usr/bin/env bash
# mutate_engine_citations.sh -- negative direction for
# scripts/checks/check_engine_citations.sh.
#
# THE SUBJECT, and why it needed a matrix and not just a green run. The citation
# rule (a citation is location-bearing AND path-qualified) was enforced ONLY by a
# scratch tool at /tmp/citecheck3.py: no gate could catch a regression in it, and
# it depended on ephemeral /tmp control bytes. Promotion into the tree makes the
# rule enforceable; a matrix makes the enforcement believable. A green checker
# run tells you the live corpus is clean, NOT that the checker would refuse a
# dirty corpus -- and "refuses nothing" and "refuses the dirty corpus" are
# indistinguishable from the outside except by planting.
#
# THE FIXTURE IS IN-REPO AND PORTABLE, and the note about it is honest. The judge
# asked for a fixture derived from the preserved pre-fix bytes; those bytes were
# /tmp-resident and /tmp was cleaned, so they are GONE. The fixture is a
# RECONSTRUCTION that carries the historical defect SHAPES (bare tokens with their
# recorded line numbers) written so the checker's own detector CANNOT self-match
# on them -- a fixture that documents the defect in the shape the detector looks
# for trips its own detector, which was MEASURED on this fixture before the
# spelling was changed. It is hosted on a fixture-local mini-clone, so the matrix
# runs on a box with no /var/home/fra/dev/intfer-upstream/ at all.
#
# BOTH POLARITIES ARE GRADED. A matrix whose only states plant defects cannot
# distinguish a working CHECKER from one that refuses EVERYTHING. So EC-ok plants
# a VALID citation and requires the checker to ACCEPT it.
#
# THE STATE ONLY THE CONTROL CAN CATCH (EC-control). This is the reason the
# checker's positive control exists at all. The plant makes the resolver accept a
# bare basename citation, so:
#     p_qualified_rule PASSES (0 bare found -- the detector is blind)
#     the verdict would read GREEN
#     ONLY p_control refuses, because the synthetic control is graded through the
# SAME resolve() the verdict comes from.
# Without that control the checker would certify a corpus of pure bare citations.
#
# WHY THE REFUSAL IS HARVESTED AS A SET. Several predicates legitimately co-fire,
# so a state asserts its EXPECTED predicate is AMONG the failing set rather than
# that it is the first line printed. A state that demanded "the first FAIL" would
# be decided by print order and could pass on the wrong predicate.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$SCRIPT_DIR/../checks/check_engine_citations.sh"
FIXTURE="$ROOT/scripts/fixtures/engine-citations"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/ecmut.XXXXXX")"
trap 'rm -rf "$SANDBOX"; exit 130' INT
trap 'rm -rf "$SANDBOX"; exit 143' TERM
die() { echo "EC-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }

[ -f "$CHECKER" ] || die "checker missing ($CHECKER)"
[ -d "$FIXTURE" ] || die "the fixture is missing ($FIXTURE); a matrix without a corpus grades nothing"
[ -f "$FIXTURE/arcaine.txt" ] || die "the fixture's arcaine.txt is missing"
[ -f "$FIXTURE/jitllm.txt" ] || die "the fixture's jitllm.txt is missing"
[ -d "$FIXTURE/clone" ] || die "the fixture's mini-clone is missing; without it every upstream/ token is NOT_FOUND and no state is attributable"

# NO SUBJECT_READS marker, and the reason is recorded rather than left as a
# silence. check_durable_reads.sh requires that marker on a matcher whose ASSERTIONS
# read a durable surface (HISTORY.md, AGENTS.md, docs/*.md, docs/engines/*.txt), and
# refuses a marker on a matcher that reads none, because "a marker no assertion
# depends on is decorative". This matrix's subject is the IN-REPO fixture at
# scripts/fixtures/engine-citations/ -- deliberately not a durable record -- so it
# carries no marker at all. The marker its first draft DID carry was exactly that
# decorative shape: it was inherited from mutate_engines_sources_paths.sh, whose
# subject is the live corpus, and no assertion in this matrix depended on it. The
# durable surface this pair is about (docs/engines/*.txt) is the CHECKER's subject,
# not the matrix's; the matrix only ever sees a private copy of the fixture.

# ---- instantiate the fixture into a private corpus ----------------------------
# One shared instantiation per state would let one mutation reach another. The
# clone path is substituted per state dir, so each state has its own.
instantiate() {
  local sd="$1"
  mkdir -p "$sd/corpus"
  cp "$FIXTURE/arcaine.txt" "$FIXTURE/jitllm.txt" "$sd/corpus/"
  cp -r "$FIXTURE/clone" "$sd/clone"
  sed -i "s|@@CLONE@@|$sd/clone|g" "$sd/corpus"/*.txt
}

# ---- ANCHOR PREFLIGHT ---------------------------------------------------------
# Fixed-string counts, for the reason mutate_engines_sources_paths.sh records: an
# anchor carrying regex metacharacters is a parse error under -E, and `grep -cE`
# then reports "0" for a query that never ran -- which reads as an absent anchor.
for a in 'Path style: intfer/<repo-relative path>' \
         'Clone: @@CLONE@@' \
         'upstream/src/engine/engine.cpp:1' \
         'intfer/AGENTS.md:1' ; do
  n=0
  n=$((n + $(grep -cF -- "$a" "$FIXTURE/arcaine.txt" 2>/dev/null || true) ))
  n=$((n + $(grep -cF -- "$a" "$FIXTURE/jitllm.txt" 2>/dev/null || true) ))
  if [ "$n" -lt 1 ]; then die "anchor absent from the fixture: $a -- a state targeting it is NO test, not a weak one"; fi
done
echo "EC-MUT: anchor preflight ok"

# ---- BASELINE. The unmutated fixture must PASS, or every "caught" below is
#      indistinguishable from a checker that refuses everything.
BASE="$SANDBOX/base"
instantiate "$BASE"
ENG_CITE_CORPUS="$BASE/corpus" ENG_CITE_ROOT="$ROOT" bash "$CHECKER" >"$SANDBOX/baseline.txt" 2>&1
if [ $? -ne 0 ]; then
  sed 's/^/    /' "$SANDBOX/baseline.txt"
  die "the healthy fixture FAILS the checker -- the negative direction is meaningless"
fi
grep -q 'ENG-CITE: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
echo "EC-MUT: baseline PASS on the unmutated fixture"

# ---- RESIDUE, before anything is touched --------------------------------------
BEFORE_FIX="$(cd "$FIXTURE" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
BEFORE_CHECK="$(sha256sum "$CHECKER" | cut -d' ' -f1)"
BEFORE_STATUS="$(cd "$ROOT" && git status --porcelain -- scripts/fixtures scripts/checks scripts/matrices docs/engines 2>/dev/null | LC_ALL=C sort)"

# ---- the grader ---------------------------------------------------------------
# $1 = state, $2 = accept | <a predicate name>, $3 = mutator function.
grade() {
  local state="$1" expect="$2" mut="$3"
  local sd="$SANDBOX/$state"
  instantiate "$sd"
  local pre
  pre="$( (cd "$sd" && find . -type f | LC_ALL=C sort | xargs cat | sha256sum) | cut -d' ' -f1 )"
  "$mut" "$sd" || die "$state: the mutator command failed"
  local post
  post="$( (cd "$sd" && find . -type f | LC_ALL=C sort | xargs cat | sha256sum) | cut -d' ' -f1 )"
  # THE STANDING FIX: assert the mutation LANDED before grading it. An unlanded
  # mutation changes nothing, so a "catch" from it certifies that the checker
  # never saw the defect -- the exact vacuous green this matrix exists to prevent.
  [ "$pre" = "$post" ] && die "$state: the mutation did NOT LAND -- refusing to grade a no-op as a catch"

  local out rc failing
  out="$(ENG_CITE_CORPUS="$sd/corpus" ENG_CITE_ROOT="$ROOT" bash "$CHECKER" 2>&1)"; rc=$?
  failing="$(printf '%s\n' "$out" | sed -n 's/^ENG-CITE: FAIL \([a-z_]*\).*/\1/p' | tr '\n' ' ')"

  if [ "$expect" = "accept" ]; then
    if [ "$rc" -ne 0 ]; then
      echo "EC-MUT: $state NOT ACCEPTED -- the checker refused a VALID corpus (failing: $failing)" >&2
      printf '%s\n' "$out" | sed 's/^/    /' >&2
      return 1
    fi
    echo "EC-MUT: $state ACCEPTED (a valid path-qualified citation is not refused)"
    return 0
  fi
  if [ "$rc" -eq 0 ]; then
    echo "EC-MUT: $state NOT CAUGHT -- the checker passed a defective corpus" >&2
    printf '%s\n' "$out" | sed 's/^/    /' >&2
    return 1
  fi
  case " $failing " in
    *" $expect "*)
      echo "EC-MUT: $state CAUGHT (failing: $failing; expected: $expect)"; return 0 ;;
  esac
  # CAUGHT, but not by the predicate this state exists to exercise. A state that
  # tripped an unrelated refusal is a weaker claim than "caught" suggests, and
  # recording it as a catch is how a matrix ends up certifying a rule it never
  # probed.
  echo "EC-MUT: $state CAUGHT FOR THE WRONG REASON (failing: $failing; expected: $expect)" >&2
  printf '%s\n' "$out" | sed 's/^/    /' >&2
  return 1
}

# A helper: replace the first occurrence of a token in the corpus files, and
# PROVE it landed.  The replacement is done in PYTHON, not sed: the first draft
# used `sed -i "s|$from|$to|"`, and the style-statement anchor CONTAINS a `|`,
# which made the sed script malformed -- the delimiter appeared inside the
# pattern, the command failed, and the state was graded as if it had planted
# something.  A mutator that silently no-ops is the failure mode that turns a
# state into a grade of nothing.
plant() {
  local file="$1" from="$2" to="$3" n
  n=$(grep -cF -- "$from" "$file" 2>/dev/null || true)
  [ "$n" -ge 1 ] || die "plant: the anchor '$from' is not in $(basename "$file")"
  python3 - "$file" "$from" "$to" <<'PY'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(path, encoding='utf-8').read()
if frm not in txt:
    sys.stderr.write("plant: the anchor vanished before the replacement\n")
    sys.exit(3)
open(path, 'w', encoding='utf-8').write(txt.replace(frm, to, 1))
PY
  grep -qF -- "$to" "$file" || die "plant: the replacement '$to' did not land in $(basename "$file")"
}

# ---- the mutators, one defect shape each --------------------------------------
# EC-1 the historical arcaine shape: the bare basename `engine.cpp` (the recorded
# pre-fix token), planted on the citation that the re-cut qualified.
m_ec1() {
  plant "$1/corpus/arcaine.txt" 'upstream/src/engine/engine.cpp:1' 'engine.cpp:2124'
}
# EC-2 the historical jitllm shape: the bare basename `README.md` (the recorded
# pre-fix token), which is the one the OLD resolver resolved ROOT-FIRST and
# declared GREEN against intfer's own README.md.
m_ec2() {
  plant "$1/corpus/jitllm.txt" 'upstream/README.md:6' 'README.md:6'
}
m_ec3() {  # a cited line past the end of the file it names
  plant "$1/corpus/arcaine.txt" 'upstream/src/engine/engine.cpp:1' 'upstream/src/engine/engine.cpp:999999'
}
m_ec4() {  # a path-qualified citation that resolves nowhere
  plant "$1/corpus/arcaine.txt" 'upstream/src/engine/engine.cpp:1' 'upstream/no/such/file.cpp:1'
}
m_ec5() {  # a plain path that exists in BOTH trees -- the AMBIGUOUS_TREE case
  # The basename alone would NOT reach this class: the path rule refuses a bare
  # basename BEFORE any resolution is attempted, so `README.md:1` fires
  # BARE_CITATION, not AMBIGUOUS_TREE.  The ambiguous token must still be
  # path-qualified -- so it is a DIRECTORY path that both trees carry.
  plant "$1/corpus/arcaine.txt" 'upstream/include/intfer/engine.hpp:2' 'include/intfer/engine.hpp:2'
}
m_ec6() {  # delete the clone declaration, so every upstream/ token is orphaned
  sed -i '/^# Clone: /d' "$1/corpus/arcaine.txt"
  grep -q 'upstream/src/engine/engine.cpp:1' "$1/corpus/arcaine.txt" \
    || die "EC-6: the upstream citation to orphan is no longer in the corpus"
}
m_ec7() {  # the absent sentinel on a file that EXISTS
  plant "$1/corpus/arcaine.txt" 'upstream/src/engine/engine.cpp:1' 'absent/README.md:1'
}
m_ec8() {  # an absolute-path citation -- the house-style violation
  plant "$1/corpus/arcaine.txt" 'upstream/src/engine/engine.cpp:1' "$ROOT/README.md:1"
}
m_ec9() {  # an absolute-path MENTION (no location)
  # Planted over a genuine MENTION (a token with no location).  Planting it
  # over a citation's PREFIX would leave the location in place and make it an
  # absolute CITATION instead -- which the previous draft did, and which the
  # p_house_style predicate cannot tell apart from the mention case.
  plant "$1/corpus/jitllm.txt" 'intfer/scripts/knob_sweep.sh' "$ROOT/README.md"
}
m_ec10() {  # the `# Path style:` header statement deleted
  sed -i '/^# Path style:/,/^# clone named by the header line above/d' "$1/corpus/arcaine.txt"
  grep -q 'Path style:' "$1/corpus/arcaine.txt" && die "EC-10: the style statement is still present"
  return 0
}
m_ec11() {  # a malformed location: a colon followed by something that is not a number
  plant "$1/corpus/arcaine.txt" 'upstream/src/engine/engine.cpp:1' 'upstream/src/engine/engine.cpp:abc'
}
m_ec12() {  # a path-qualified mention that names a file that is not there
  # Over a genuine MENTION.  Planted over a CITATION's prefix the first draft
  # kept the trailing `:1`, so the planted token was a CITATION and the state
  # fired p_resolve (NOT_FOUND) instead of p_mentions (UNRESOLVED_MENTION) --
  # caught for the wrong reason, and caught by a rule this state does not test.
  plant "$1/corpus/jitllm.txt" 'intfer/scripts/knob_sweep.sh' 'upstream/no/such/notes.md'
}
m_ec13() {  # an EMPTY corpus: looked at nothing must not read as PASS
  rm -f "$1/corpus"/*.txt
  [ -z "$(ls -A "$1/corpus")" ] || die "EC-13: the corpus is not empty"
}
# EC-ok the ACCEPTANCE polarity: a valid, resolving, range-checked citation.
m_ecok() {
  plant "$1/corpus/arcaine.txt" 'upstream/src/engine/engine.cpp:1' 'intfer/README.md:1'
}
# EC-control the state ONLY the positive control can catch: make the resolver
# accept a bare basename citation. p_qualified_rule goes blind; only p_control
# refuses. Planted on the LIBRARY, which is what the rule lives in -- and the
# residue check then has to notice that the library was touched, which is exactly
# what it is there for.
m_eccontrol() {
  python3 - "$1" <<'PY'
import sys
sd = sys.argv[1]
p = sd + "/lib.py"
txt = open(p).read()
old = "if kind == 'plain' and '/' not in tok and is_citation:"
new = "if False:  # EC-MUT PLANTED DEFECT: the path rule is disabled"
if txt.count(old) != 1:
    sys.stderr.write("EC-MUT: the planted anchor is absent or ambiguous\n")
    sys.exit(3)
open(p, "w").write(txt.replace(old, new))
PY
  [ -f "$1/lib.py" ] || die "EC-control: the mutator did not produce the library copy"
}
# EC-style-noverb: the style statement exists but does NOT NAME the qualifiers,
# which is a decorative marker rather than a rule.
m_ecstyverb() {
  plant "$1/corpus/arcaine.txt" '# Path style: intfer/<repo-relative path>' '# Path style: (the rule)'
}

# The control state needs the library copied into the sandbox and the checker
# aimed at it. It cannot run against the real one without editing a tracked file,
# which is the destructive probe the standing rule exists for.
grade_eccontrol() {
  local sd="$SANDBOX/EC-control"
  instantiate "$sd"
  cp "$ROOT/scripts/lib/_engine_citations_scan.py" "$sd/lib.py"
  # ROOT stays the REAL repo: the corpus's intfer/ tokens must keep resolving
  # against the live tree, and only the RESOLVER is damaged.  That is what the
  # ENG_CITE_LIB seam exists for -- the library is a separate seam from the repo
  # precisely so a matrix can damage one without the other.
  local pre
  pre="$(sha256sum "$sd/lib.py" | cut -d' ' -f1)"
  m_eccontrol "$sd"
  local post
  post="$(sha256sum "$sd/lib.py" | cut -d' ' -f1)"
  [ "$pre" = "$post" ] && die "EC-control: the mutation did NOT LAND"
  local out rc failing
  out="$(ENG_CITE_CORPUS="$sd/corpus" ENG_CITE_ROOT="$ROOT" ENG_CITE_LIB="$sd/lib.py" bash "$CHECKER" 2>&1)"; rc=$?
  failing="$(printf '%s\n' "$out" | sed -n 's/^ENG-CITE: FAIL \([a-z_]*\).*/\1/p' | tr '\n' ' ')"
  if [ "$rc" -eq 0 ]; then
    echo "EC-MUT: EC-control NOT CAUGHT -- the checker certified a blind resolver" >&2
    printf '%s\n' "$out" | sed 's/^/    /' >&2
    return 1
  fi
  case " $failing " in
    *" p_control "*)
      echo "EC-MUT: EC-control CAUGHT (failing: $failing; expected: p_control)"; return 0 ;;
  esac
  echo "EC-MUT: EC-control CAUGHT FOR THE WRONG REASON (failing: $failing; expected: p_control)" >&2
  printf '%s\n' "$out" | sed 's/^/    /' >&2
  return 1
}

# ---- RUN ----------------------------------------------------------------------
caught=0
states=0
run_state() {
  states=$((states + 1))
  if grade "$@"; then caught=$((caught + 1)); return 0; fi
  return 1
}

rc_all=0
run_state EC-1          p_qualified_rule  m_ec1      || rc_all=1
run_state EC-2          p_qualified_rule  m_ec2      || rc_all=1
run_state EC-3          p_range           m_ec3      || rc_all=1
run_state EC-4          p_resolve         m_ec4      || rc_all=1
run_state EC-5          p_resolve         m_ec5      || rc_all=1
run_state EC-6          p_resolve         m_ec6      || rc_all=1
run_state EC-7          p_sentinel        m_ec7      || rc_all=1
run_state EC-8          p_house_style     m_ec8      || rc_all=1
run_state EC-9          p_house_style     m_ec9      || rc_all=1
run_state EC-10         p_style_header    m_ec10     || rc_all=1
run_state EC-11         p_malformed       m_ec11     || rc_all=1
run_state EC-12         p_mentions        m_ec12     || rc_all=1
run_state EC-13         p_pop             m_ec13     || rc_all=1
run_state EC-ok         accept            m_ecok     || rc_all=1
run_state EC-style-noverb p_style_header  m_ecstyverb || rc_all=1
states=$((states + 1))
if grade_eccontrol; then caught=$((caught + 1)); else rc_all=1; fi

# ---- the residue must be byte-identical ---------------------------------------
AFTER_FIX="$(cd "$FIXTURE" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
AFTER_CHECK="$(sha256sum "$CHECKER" | cut -d' ' -f1)"
AFTER_STATUS="$(cd "$ROOT" && git status --porcelain -- scripts/fixtures scripts/checks scripts/matrices docs/engines 2>/dev/null | LC_ALL=C sort)"
[ "$BEFORE_FIX" = "$AFTER_FIX" ] || die "the FIXTURE was modified by this matrix (that is the AUDIT #21 shape: the harness damage the corpus, not the checker)"
[ "$BEFORE_CHECK" = "$AFTER_CHECK" ] || die "the CHECKER was modified by this matrix"
[ "$BEFORE_STATUS" = "$AFTER_STATUS" ] || die "the git status set moved during the matrix"

echo "EC-MUT: $caught/$states states caught; residue byte-identical (fixture, checker, git status)"
if [ "$rc_all" -ne 0 ]; then
  echo "EC-MUT: FAIL ($((states - caught)) state(s) not caught, or caught for the wrong reason)"
  echo "RESULT: FAIL"
  exit 1
fi
echo "EC-MUT: PASS"
echo "RESULT: PASS -- the citation rule refuses every planted defect shape, each for its own named predicate"
exit 0
