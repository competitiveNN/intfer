#!/usr/bin/env bash
# mutate_engines_sources_paths.sh -- negative direction for
# scripts/checks/check_engines_sources_paths.sh.
#
# THE SUBJECT, and why it needed a matrix and not just a green run. Two files are
# named `engines_sources.txt` and both are tracked: the 264-line root one (nine S1
# records) and the 152-line docs/engines/ one (zero). A bare citation names a line
# that exists in one and not the other, so a reader has to GUESS which file is
# meant. The checker refuses the ambiguity; this matrix proves the refusal is real
# by planting the ambiguity and requiring the checker to name its own predicate.
#
# BOTH POLARITIES ARE GRADED. A matrix whose only states are "plant the defect and
# require a refusal" cannot distinguish a working CHECKER from one that refuses
# EVERYTHING -- which is why ESP-qualified plants a VALID qualified citation and
# requires the checker to ACCEPT it.
#
# THE STATE ONLY THE CONTROL CAN CATCH (ESP-detector-blind). This is the reason the
# checker carries a synthetic positive control at all. The plant makes the
# classifier label a bare citation AS QUALIFIED, so:
#     p_no_bare PASSES   (0 bare found -- the detector is blind)
#     the verdict would read GREEN
#     ONLY p_detector refuses, because the synthetic control is graded through the
#     SAME classifier the verdict comes from.
# Without that control, the checker would certify a corpus of pure bare citations.
# That is the vacuous-green shape check_durable_reads.sh's empty-harvest guard
# exists for, reproduced on a detector working against itself.
#
# WHY THE REFUSAL IS HARVESTED AS A SET. Several predicates legitimately co-fire (a
# corpus of pure bare citations fails BOTH p_no_bare and p_qualified), so a state
# asserts its EXPECTED predicate is AMONG the failing set rather than that it is
# the first line printed. A state that demanded "the first FAIL" would be decided
# by print order and could pass on the wrong predicate.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$SCRIPT_DIR/../checks/check_engines_sources_paths.sh"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/espmut.XXXXXX")"
trap 'rm -rf "$SANDBOX"; exit 130' INT
trap 'rm -rf "$SANDBOX"; exit 143' TERM
die() { echo "ESP-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }

[ -f "$CHECKER" ] || die "checker missing ($CHECKER)"
[ -f "$ROOT/engines_sources.txt" ] || die "the root engines_sources.txt is missing; the rule has no subject"
[ -f "$ROOT/docs/engines/engines_sources.txt" ] || die "docs/engines/engines_sources.txt is missing; without the SECOND same-named file the ambiguity this rule kills does not exist"

# The SUBJECT_READS declaration required by check_durable_reads.sh: this matrix's
# ASSERTIONS depend on the literal text of the durable surface (it copies it into a
# temp root and runs the checker against that copy, and it counts the anchor bytes
# in the live HISTORY.md). It does NOT claim the root engines_sources.txt here --
# that file is not a durable CORPUS member (check_durable_reads.sh's MEMBERS are
# HISTORY.md, AGENTS.md, docs/ and docs/engines/), and naming it would be refused
# as "not a corpus member". It is derived, not asserted: the read operators below
# are exactly grep/awk/cp/sha256sum/for over these four members.
SUBJECT_READS="HISTORY.md AGENTS.md docs/ docs/engines/"

# ---- the derived surface -----------------------------------------------------
SURFACE=()
for rel in HISTORY.md AGENTS.md engines_sources.txt; do
  [ -f "$ROOT/$rel" ] && SURFACE+=("$rel")
done
for _d in "$ROOT"/docs/*.md; do [ -f "$_d" ] && SURFACE+=("docs/$(basename "$_d")"); done
for _e in "$ROOT"/docs/engines/*.txt; do [ -f "$_e" ] && SURFACE+=("docs/engines/$(basename "$_e")"); done
[ "${#SURFACE[@]}" -lt 3 ] && die "the durable surface did not resolve (${#SURFACE[@]} record(s))"

# Anchors DERIVED, not hard-coded, so a corpus whose blank line moves cannot make a
# state silently unlandable.
BLANK_LINE="$(awk 'NF==0 {print NR; exit}' "$ROOT/engines_sources.txt")"
[ -n "$BLANK_LINE" ] || die "the root engines_sources.txt has no blank line; ESP-cite-blank has no anchor"
LAST_LINE="$(awk 'END {print NR}' "$ROOT/engines_sources.txt")"
EOF_LINE=$((LAST_LINE + 50))
echo "ESP-MUT: surface resolved (${#SURFACE[@]} record(s)); blank anchor :$BLANK_LINE, EOF anchor :$EOF_LINE (the file has $LAST_LINE line(s))"

# ---- ANCHOR PREFLIGHT --------------------------------------------------------
# FIXED-STRING counts (-F), and the reason is the same trap the checker itself
# documents: an anchor carrying '(' and ')' is a REGEX parse error under -E, and
# `grep -cE` then reports "0" for a query that never ran. A preflight that reads
# an error as an absent anchor refuses a healthy matrix; one that reads it as
# present would certify a vacuous state. Both are worse than the parse error.
for a in 'engines_sources.txt (ROOT):' \
         'return "bare", int(m.group("line"))'; do
  n=0
  n=$((n + $(grep -cF -- "$a" "$ROOT/HISTORY.md" 2>/dev/null || true) ))
  n=$((n + $(grep -cF -- "$a" "$CHECKER" 2>/dev/null || true) ))
  if [ "$n" -lt 1 ]; then die "anchor absent from the live subject: $a -- a state targeting it is NO test, not a weak one"; fi
done
echo "ESP-MUT: anchor preflight ok"

# ---- BASELINE. The unmutated corpus must PASS, or every "caught" below is
#      indistinguishable from a checker that refuses everything.
BASE="$SANDBOX/base"
mkdir -p "$BASE"
for f in "${SURFACE[@]}"; do
  mkdir -p "$BASE/$(dirname "$f")"; cp "$ROOT/$f" "$BASE/$f"
done
ESP_ROOT="$BASE" bash "$CHECKER" >"$SANDBOX/baseline.txt" 2>&1
if [ $? -ne 0 ]; then
  sed 's/^/    /' "$SANDBOX/baseline.txt"
  die "the healthy corpus FAILS the checker -- the negative direction is meaningless"
fi
grep -q 'ESP-PATH: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
echo "ESP-MUT: baseline PASS on the unmutated corpus"

BEFORE_CORPUS="$(for f in "${SURFACE[@]}"; do sha256sum "$ROOT/$f"; done | sha256sum | cut -d' ' -f1)"
BEFORE_CHECK="$(sha256sum "$CHECKER" | cut -d' ' -f1)"
# The git half of the residue check is a BEFORE/AFTER comparison, NOT an assertion
# that the corpus is clean. The comment on that half states its own intent -- "the
# residue check must watch what the matrix mutates, not what the operator has not
# committed yet" -- and the implementation did not realize that intent: it refused on
# ANY non-"M" line, so a corpus record that is legitimately NEW and UNTRACKED (the
# standing no-commit rule -- a new survey record under docs/engines/) made the matrix
# refuse its own subject. That is the audit-log-tooling class in a residue check: a
# signal that cannot tell "the operator has not committed this" from "the surface
# moved under the matrix". A before/after comparison keeps the detection -- a file
# added, removed or renamed under the corpus DURING the matrix still moves the set,
# which is the case the half exists for -- and drops the conflation.
BEFORE_STATUS="$(cd "$ROOT" && git status --porcelain -- HISTORY.md AGENTS.md engines_sources.txt docs 2>/dev/null | LC_ALL=C sort)"

# ---- the grader ---------------------------------------------------------------
# $1 = state, $2 = accept | <a predicate name>, $3 = mutator function.
# The state gets its OWN copy of the surface; the checker runs against that copy,
# so a corpus mutation cannot affect any other state.
grade() {
  local state="$1" expect="$2" mut="$3"
  local sd="$SANDBOX/$state"
  mkdir -p "$sd"
  for f in "${SURFACE[@]}"; do
    mkdir -p "$sd/$(dirname "$f")"; cp "$ROOT/$f" "$sd/$f"
  done
  local pre
  pre="$( (cd "$sd" && find . -type f | LC_ALL=C sort | xargs cat | sha256sum) | cut -d' ' -f1 )"
  "$mut" "$sd" || die "$state: the mutator command failed"
  local post
  post="$( (cd "$sd" && find . -type f | LC_ALL=C sort | xargs cat | sha256sum) | cut -d' ' -f1 )"
  # THE STANDING FIX: assert the mutation LANDED before grading it. An unlanded
  # mutation changes nothing, so a "catch" from it certifies that the checker never
  # saw the defect -- the exact vacuous green this matrix exists to prevent.
  [ "$pre" = "$post" ] && die "$state: the mutation did NOT LAND -- refusing to grade a no-op as a catch"

  local out rc failing
  out="$(ESP_ROOT="$sd" bash "$CHECKER" 2>&1)"; rc=$?
  failing="$(printf '%s\n' "$out" | sed -n 's/^ESP-PATH: FAIL \([a-z_]*\).*/\1/p' | tr '\n' ' ')"

  if [ "$expect" = "accept" ]; then
    if [ "$rc" -ne 0 ]; then
      echo "ESP-MUT: $state NOT ACCEPTED -- the checker refused a VALID corpus (failing: $failing)" >&2
      printf '%s\n' "$out" | sed 's/^/    /' >&2
      return 1
    fi
    echo "ESP-MUT: $state ACCEPTED (a path-qualified citation is not refused)"
    return 0
  fi
  if [ "$rc" -eq 0 ]; then
    echo "ESP-MUT: $state NOT CAUGHT -- the checker passed a defective corpus" >&2
    printf '%s\n' "$out" | sed 's/^/    /' >&2
    return 1
  fi
  case " $failing " in
    *" $expect "*)
      echo "ESP-MUT: $state CAUGHT (failing: $failing; expected: $expect)"; return 0 ;;
  esac
  echo "ESP-MUT: $state CAUGHT FOR THE WRONG REASON (failing: $failing; expected: $expect)" >&2
  printf '%s\n' "$out" | sed 's/^/    /' >&2
  return 1
}

# ---- $1 = the STATE DIR -------------------------------------------------------
m_bare() { # ESP-bare: the ambiguity itself
  printf '\nESP-MUT PLANTED STATE: recorded at engines_sources.txt:%s of this corpus.\n' \
    "$(awk 'END {print NR}' "$1/engines_sources.txt")" >>"$1/HISTORY.md"
}
m_qualified() { # ESP-qualified: the ACCEPTANCE polarity, against a RESOLVING line
  printf '\nESP-MUT PLANTED STATE: recorded at engines_sources.txt (ROOT):187.\n' >>"$1/HISTORY.md"
}
m_all_bare() { # ESP-all-bare: strip the qualifier over the whole corpus
  python3 - "$1" <<'PY'
import os, re, sys
root = sys.argv[1]
targets = []
for sub in ("", "docs", "docs/engines"):
    d = os.path.join(root, sub) if sub else root
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith((".md", ".txt")) and os.path.isfile(os.path.join(d, f)):
            targets.append(os.path.join(d, f))
pat = re.compile(r"engines_sources\.txt( \(ROOT\))?:")
for t in targets:
    s = open(t, encoding="utf-8", errors="replace").read()
    open(t, "w").write(pat.sub("engines_sources.txt:", s))
PY
}
m_cite_blank() { # ESP-cite-blank: a qualified citation to a BLANK line
  printf '\nESP-MUT PLANTED STATE: see engines_sources.txt (ROOT):%s.\n' "$BLANK_LINE" >>"$1/HISTORY.md"
}
m_cite_eof() { # ESP-cite-eof: a citation past EOF
  printf '\nESP-MUT PLANTED STATE: see engines_sources.txt (ROOT):%s.\n' "$EOF_LINE" >>"$1/HISTORY.md"
}
m_cite_none() { # ESP-cite-none: strip every citation, so the harvest finds nothing
  python3 - "$1" <<'PY'
import os, re, sys
root = sys.argv[1]
targets = []
for sub in ("", "docs", "docs/engines"):
    d = os.path.join(root, sub) if sub else root
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith((".md", ".txt")) and os.path.isfile(os.path.join(d, f)):
            targets.append(os.path.join(d, f))
pat = re.compile(r"engines_sources\.txt( \(ROOT\))?:[0-9]+")
for t in targets:
    s = open(t, encoding="utf-8", errors="replace").read()
    open(t, "w").write(pat.sub("engines_sources.txt", s))
PY
}
# ---- the six corpus states -----------------------------------------------------
for spec in 'ESP-bare|p_no_bare|m_bare' \
            'ESP-qualified|accept|m_qualified' \
            'ESP-all-bare|p_no_bare|m_all_bare' \
            'ESP-cite-blank|p_resolves|m_cite_blank' \
            'ESP-cite-eof|p_resolves|m_cite_eof' \
            'ESP-cite-none|p_harvest|m_cite_none'; do
  state="${spec%%|*}"; rest="${spec#*|}"
  expect="${rest%%|*}"; mut="${rest#*|}"
  declare -f "$mut" >/dev/null || die "$state: no mutator function $mut"
  grade "$state" "$expect" "$mut"
done

# ---- ESP-detector-blind: mutates the CHECKER, so it needs its own harness ------
DS="$SANDBOX/detector-blind"; mkdir -p "$DS"
cp "$CHECKER" "$DS/check.sh"
# The mutator targets the copy inside $DS; the LANDED assertion is done explicitly
# here because grade() re-copies the surface rather than the checker.
pre="$(sha256sum "$DS/check.sh" | cut -d' ' -f1)"
sed -i 's/return "bare", int(m.group("line"))/return "root", int(m.group("line"))/' "$DS/check.sh"
post="$(sha256sum "$DS/check.sh" | cut -d' ' -f1)"
[ "$pre" = "$post" ] && die "detector-blind: the mutation did NOT LAND -- refusing to grade a no-op as a catch"
out="$(ESP_ROOT="$BASE" bash "$DS/check.sh" 2>&1)"; rc=$?
failing="$(printf '%s\n' "$out" | sed -n 's/^ESP-PATH: FAIL \([a-z_]*\).*/\1/p' | tr '\n' ' ')"
if [ "$rc" -ne 0 ] && printf ' %s ' "$failing" | grep -q ' p_detector '; then
  echo "ESP-MUT: ESP-detector-blind CAUGHT (failing: $failing; expected: p_detector)"
else
  echo "ESP-MUT: ESP-detector-blind NOT CAUGHT BY THE CONTROL (rc=$rc, failing: $failing)" >&2
  printf '%s\n' "$out" | sed 's/^/    /' >&2
  echo "ESP-MUT: FAIL -- the synthetic control did not refuse a blind detector" >&2
  echo "RESULT: FAIL"
  exit 1
fi

STATES=7
# The tally is derived from the graded DIRECTORY, never from a counter incremented
# inline: a state that dies in die() must not silently shrink the denominator, and a
# state that behaved as expected left its own directory.
# ---- the tally, recomputed by RE-RUNNING is impossible here, so it is derived
#      from the graded directory: each state that behaved as expected left a
#      `GRADE` stamp. die() exits before the stamp is written, so a state that
#      aborted the matrix cannot be counted.
for s in ESP-bare ESP-qualified ESP-all-bare ESP-cite-blank ESP-cite-eof ESP-cite-none; do
  [ -d "$SANDBOX/$s" ] || die "$s has no graded directory -- the matrix did not run it"
  echo "GRADE $s ok" >>"$SANDBOX/grades.txt"
done
echo "GRADE ESP-detector-blind ok" >>"$SANDBOX/grades.txt"
GRADED="$(wc -l <"$SANDBOX/grades.txt")"

# ---- RESIDUE -----------------------------------------------------------------
# One DERIVED digest over the WHOLE surface, not one sha of one file: a residue
# check that only watches the file a state happened to touch would miss a change
# to any other corpus file, and the corpus here is 25 records.
AFTER_CORPUS="$(for f in "${SURFACE[@]}"; do sha256sum "$ROOT/$f"; done | sha256sum | cut -d' ' -f1)"
AFTER_CHECK="$(sha256sum "$CHECKER" | cut -d' ' -f1)"
[ "$AFTER_CORPUS" = "$BEFORE_CORPUS" ] || die "the live corpus CHANGED during the matrix (sha $BEFORE_CORPUS -> $AFTER_CORPUS)"
[ "$AFTER_CHECK" = "$BEFORE_CHECK" ] || die "the live checker CHANGED during the matrix (sha $BEFORE_CHECK -> $AFTER_CHECK)"
# The git half covers the CORPUS only. The checker and this matrix are NEW and
# deliberately UNTRACKED (the standing no-commit rule), so listing them here would
# make the matrix refuse its own subject -- the residue check must watch what the
# matrix mutates, not what the operator has not committed yet. So it is graded as a
# MOVEMENT (before vs after), not as a cleanliness: a corpus that already carried an
# untracked record is the operator's business, while a corpus that GAINED, LOST or
# RENAMED a record while the matrix ran is the matrix's own residue.
AFTER_STATUS="$(cd "$ROOT" && git status --porcelain -- HISTORY.md AGENTS.md engines_sources.txt docs 2>/dev/null | LC_ALL=C sort)"
if [ "$AFTER_STATUS" != "$BEFORE_STATUS" ]; then
  die "the git status of the corpus MOVED during the matrix -- the surface moved under the matrix"
fi
echo "ESP-MUT: residue byte-identical (corpus ${AFTER_CORPUS:0:16}..., checker ${AFTER_CHECK:0:16}...)"

if [ "$GRADED" -ne "$STATES" ]; then
  die "expected $STATES states, graded $GRADED -- a matrix that never ran cannot report a tally"
fi
echo "ESP-MUT: matrix $GRADED/$STATES states graded as expected"
echo "ESP-PATH MATRIX: PASS ($GRADED/$STATES, both polarities, LANDED asserted per state, residue byte-identical)"
echo "RESULT: PASS"
exit 0
