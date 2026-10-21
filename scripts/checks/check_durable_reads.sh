#!/usr/bin/env bash
# SOURCE-level certification of the DURABLE-READ population of the mutation matrices.
#
# THE QUESTION, and why it needed a checker rather than a comment. Six matrices
# (mutate_durable_claims.sh, mutate_durable_reads.sh, mutate_history_pointers.sh,
# mutate_knob_sweep_contract.sh, mutate_tally_propagation.sh, mutate_xmx_attrib.sh)
# make ASSERTIONS whose satisfaction depends on the literal text of a durable record
# -- HISTORY.md, AGENTS.md, docs/*.md or a docs/engines/*.txt history file. Each carries a uniform
# `SUBJECT_READS=` marker naming the surfaces its assertions read. Nothing pinned that
# population: a seventh matrix could start reading durable text and never be marked,
# and the dependency would stay invisible until that matrix silently MISSed on an
# anchor drift.
#
# WHAT THE DERIVATION COVERS, stated precisely because the RESULT line must not claim
# more than it has. A matrix is a READER when its EXECUTABLE code applies a read
# operator to a corpus member. Three things make that judgement, and each was added
# because MEASUREMENT showed the narrower derivation was WRONG, not merely narrow:
#   1. COMMENTS ARE STRIPPED, so a prose citation is not a read.
#   2. THE MATCHER'S OWN VARIABLE ASSIGNMENTS ARE RESOLVED, so a read through a
#      variable (`$DIRTY` -> ... AGENTS.md ...) is visible. MEASURED: without this,
#      mutate_destructive_git.sh was INVISIBLE -- its dirty-tree guard runs
#      `git diff --quiet -- $DIRTY` and DG-3 patches the rule IN AGENTS.md, but the
#      member name is assembled at run time and never appears next to an operator.
#      The population was SIX while the true count was SEVEN: a reader could delete
#      its declaration and no gate would notice, which is the gap this file closes.
#   3. GIT IS A READ OPERATOR, because `git diff`/`git show`/`git cat-file`/
#      `git hash-object` read a file's content exactly as `cat` does. MEASURED:
#      adding the git operators ALONE catches ZERO -- the name is behind the
#      variable -- so (2) and (3) are needed TOGETHER and neither is decorative.
#
# THIS IS NOT A NAME-MENTION SWEEP, and the reason is measured, not cautionary. A
# comment-stripped grep for the durable names returns TEN matchers; five of them name
# a durable record only inside a `echo`/`die`/comment string (mutate_roofline_probe.sh,
# mutate_run_all_checks.sh, mutate_hw_constants.sh's writer, mutate_destru...). Those
# are prose citations, not reads. The POPULATION is therefore derived from
# comment-stripped EXECUTABLE code only, and the stripper is what makes the two
# distinguishable -- the same lesson as check_destructive_git.sh's comment strip before
# its harvest.
#
# WHAT IS CHECKED:
#   1. n_readers      every matcher whose EXECUTABLE code reads a corpus member
#   2. p_marked       every such matcher carries SUBJECT_READS naming >= 1 member
#   3. p_names_real   every surface the marker names is a corpus member it really reads
#                     (a marker naming a stale or absent file is a lie, not a hedge)
#   4. p_not_decorative every SUBJECT_READS in the tree belongs to a real reader
#                     (an unread marker is the vacuous-green shape)
#   5. p_corpus       the corpus resolved non-empty, so 0 readers is refusable
#
# ANTI-VACUITY: a corpus that resolves to nothing and a population of zero readers both
# FAIL. "Nothing to check" must never read as "checked, nothing found".
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

echo "== DURABLE-READ COVERAGE OF THE MUTATION MATRICES =="

# THE CORPUS, derived the same way check_durable_claims.sh derives its own, so the two
# checkers cannot disagree about what the durable surface IS.
CORPUS="$ROOT/HISTORY.md $ROOT/AGENTS.md"
for _d in "$ROOT"/docs/*.md; do
  [ -f "$_d" ] && CORPUS="$CORPUS $_d"
done
for _e in "$ROOT"/docs/engines/*.txt; do
  [ -f "$_e" ] && CORPUS="$CORPUS $_e"
done

MATCHERS="$(ls "$ROOT"/scripts/matrices/mutate_*.sh 2>/dev/null)"
[ -n "$MATCHERS" ] || { echo "  FAIL no mutation matrix found under $ROOT/scripts" >&2; exit 1; }

# strip comments so a name inside prose cannot masquerade as a read. `#` to EOL is
# enough: these are shell scripts.
strip_comments() { sed -e 's/[[:space:]]#.*$//' -e 's/^[[:space:]]*#.*$//' "$1"; }

# THE RESOLVER: bounded, STATIC expansion of the matcher's OWN variable assignments.
# It never executes the matcher -- it expands only `VAR="literal"` assignments and
# `for VAR in ...` lists, for a fixed number of passes.
#
# IT IS FED THE ALREADY-STRIPPED CODE, deliberately. If the resolver stripped comments
# too, then breaking the shell stripper would be masked by the resolver and the
# matrix state that pins the stripper (DR-4) would stop testing it. The stripper
# stays the only comment authority, and the resolver only widens the READ test.
#
# The program avoids single quotes entirely (\x27 for a quote) so it can live inside a
# shell single-quoted string without escaping, and it reads its input on stdin.
resolve_code() { python3 -c '
import re,sys
code=sys.stdin.read()
vals={}
for m in re.finditer(r"^[ \t]*([A-Za-z_][A-Za-z_0-9]*)=\"([^\"\n]*)\"[ \t]*$", code, re.M): vals[m.group(1)]=m.group(2)
for m in re.finditer(r"^[ \t]*([A-Za-z_][A-Za-z_0-9]*)=\x27([^\x27\n]*)\x27[ \t]*$", code, re.M): vals.setdefault(m.group(1), m.group(2))
for m in re.finditer(r"^[ \t]*for ([A-Za-z_][A-Za-z_0-9]*) in ([^\n]*?);?[ \t]*do", code, re.M): vals.setdefault(m.group(1), m.group(2))
out=code
for _ in range(8):
    before=out
    for k,v in vals.items(): out=out.replace("${"+k+"}", v).replace("$"+k, v)
    if out==before: break
sys.stdout.write(out)
'; }

# THE MEMBER UNIVERSE, and it is ONE list shared by the reader test and the marker
# test, because two lists would be two hand-maintained inventories -- the thing this
# repo keeps deleting. The two primary records are named by basename; the docs/ tree
# and the docs/engines/ tree are ONE token each, because the corpus derivation globs
# them and a glob is not a name.
MEMBERS="HISTORY.md AGENTS.md docs/ docs/engines/"

corpus_has() {
  local member="$1" c
  for c in $CORPUS; do
    [ "$(basename "$c")" = "$member" ] && return 0
    # the docs glob is a corpus member under its own name
    [ "$member" = "docs/" ] && case "$c" in */docs/*.md) return 0;; esac
    # the engines-history glob is a corpus member under its own name too
    [ "$member" = "docs/engines/" ] && case "$c" in */docs/engines/*.txt) return 0;; esac
  done
  return 1
}

n_readers=0; n_unmarked=0; readers=""
bad_marked=0; bad_names=0; bad_decorative=0
marked_total=0

for f in $MATCHERS; do
  code="$(strip_comments "$f")"
  # the READ test runs on the RESOLVED code; the MARKER test below runs on the
  # unresolved one, because the marker is a statement about what the file declares.
  resolved="$(printf '%s\n' "$code" | resolve_code)"
  # A READ is executable contact with a corpus MEMBER: a shell read (sed/grep/cat/awk
  # on the path), a python open() of it, or a cp/sha256sum of it. Matching the bare
  # basename in stripped code is what excludes the prose citations.
  #
  # MEASURED FALSE POSITIVE, recorded rather than silently excluded: the first version
  # of this predicate counted `open(os.path.join(root, "AGENTS.md"), "w")` as a read.
  # That line is a WRITER into the SANDBOX root mutate_hw_constants.sh builds, whose
  # own header states "the tracked tree is never edited". A read must therefore be
  # judged PER LINE so a write-mode open is rejected on its own line; a whole-strip
  # grep returns a file's verdict from ANY single matching line, which is exactly how
  # the writer was counted. That is the same shape as the name-mention sweep this
  # checker was written to replace, caught in its own first run.
  # MEASURED COST, and the reason this is not a per-LINE loop. The first version
  # ran the whole test inside `while read` over every stripped line, spawning two
  # processes per line per matcher: 40 SECONDS for one run of this checker, which
  # made its own matrix time out (7 runs x 40 s). The per-LINE requirement is real
  # -- a write-mode open must be rejected on its own line -- and `grep -v` gives
  # exactly that semantics in ONE pass, so the filter is applied to the block and
  # the member test then runs over the surviving block. Same semantics, ~4
  # processes per matcher instead of ~1200.
  code_ro="$(printf '%s\n' "$resolved" | grep -vE 'open\(.*"[wa]"')"
  reads=""
  for member in $MEMBERS; do
    if printf '%s\n' "$code_ro" | grep -qE "(sed|grep|awk|cat|cp|sha256sum|head|tail|python3?|git (diff|show|cat-file|hash-object)|for)[^;|]*$member|open\(.*$member"; then
      case " $reads " in *" $member "*) ;; *) reads="$reads $member";; esac
    fi
  done

  # A MARKER IS AN ASSIGNMENT, so it is ANCHORED to the start of a statement.
  # MEASURED, and found by this checker's own matrix: the unanchored form
  # `grep -oE 'SUBJECT_READS=.*'` counted the MUTATION PAYLOAD of
  # scripts/matrices/mutate_durable_reads.sh -- which must WRITE the marker string into
  # another file in order to plant a decorative marker -- as if that mutator
  # carried a marker itself. The matrix's own baseline went red for it, which is
  # the same class as mutate_destructive_git.sh's DG-2 decoy: a harness whose own
  # payload is indistinguishable from the thing it hunts. Anchoring to
  # `^[[:space:]]*SUBJECT_READS=` says what a marker IS, and leaves the payload
  # mid-line where it belongs.
  marker="$(printf '%s\n' "$code" | grep -oE '^[[:space:]]*SUBJECT_READS=.*' | head -1)"

  if [ -n "$reads" ]; then
    n_readers=$((n_readers + 1))
    readers="$readers $(basename "$f")"
    if [ -z "$marker" ]; then
      n_unmarked=$((n_unmarked + 1))
      printf '  FAIL  %-38s READS the durable surface but carries no SUBJECT_READS\n' \
        "$(basename "$f")" >&2
      echo "        reads:$reads -- the dependency is unpinned" >&2
    else
      printf '  ok    %-38s SUBJECT_READS=%s (reads:%s)\n' "$(basename "$f")" \
        "$(printf '%s' "$marker" | sed 's/SUBJECT_READS=//; s/"//g')" "$reads"
      # every named surface must be one it really reads. The marker may name a VARIABLE
      # that holds the surface list (SUBJECT_READS=$RESID_FILES with
      # RESID_FILES="HISTORY.md AGENTS.md docs/..." two hundred lines above),
      # so the value is EXPANDED through that matcher's own assignment before it is
      # compared. Comparing the raw "$RESID_FILES" token against the corpus names is a
      # false negative that would refuse two correct matchers -- the always-red twin of
      # the vacuous green, and the same coupling that made the unexpanded writer a
      # false positive above.
      _mkraw="$(printf '%s' "$marker" | sed 's/SUBJECT_READS=//; s/"//g')"
      _mkexp="$_mkraw"
      for _v in $(printf '%s\n' "$_mkraw" | grep -oE '\$[A-Za-z_][A-Za-z_0-9]*' | sort -u); do
        _val="$(printf '%s\n' "$code" | grep -oE "^${_v#\$}=\"[^\"]*\"" | head -1 | sed "s/^${_v#\$}=\"//; s/\"$//")"
        _mkexp="${_mkexp//$_v/ $_val}"
      done
      for named in $_mkexp; do
        [ "$named" = "any" ] && continue
        if ! corpus_has "$named"; then
          bad_names=$((bad_names + 1))
          printf '  FAIL  %-38s SUBJECT_READS names "%s", which is not a corpus member\n' \
            "$(basename "$f")" "$named" >&2
        elif ! printf '%s\n' "$reads" | grep -qF "$named"; then
          bad_names=$((bad_names + 1))
          printf '  FAIL  %-38s SUBJECT_READS names "%s" but its code never reads it\n' \
            "$(basename "$f")" "$named" >&2
        fi
      done
    fi
  elif [ -n "$marker" ]; then
    bad_decorative=$((bad_decorative + 1))
    printf '  FAIL  %-38s carries SUBJECT_READS but reads NO durable surface\n' \
      "$(basename "$f")" >&2
    echo "        a marker no assertion depends on is decorative" >&2
  fi
done

# count markers separately so a marker on a NON-reader is caught even though the loop
# above only reports it when the reader branch is not taken
while IFS= read -r f; do
  code="$(strip_comments "$f")"
  printf '%s\n' "$code" | grep -qE '^[[:space:]]*SUBJECT_READS=' && marked_total=$((marked_total + 1))
done <<< "$MATCHERS"

echo "-- durable-surface readers: $n_readers, marked: $((n_readers - n_unmarked)), unmarked: $n_unmarked, markers in tree: $marked_total"

bad=0
if [ -z "$CORPUS" ] || [ "$(printf '%s\n' $CORPUS | grep -c .)" -lt 3 ]; then
  echo "  FAIL  the durable corpus did not resolve (need >= 3 members)" >&2
  bad=$((bad + 1))
fi
if [ "$n_readers" -lt 1 ]; then
  echo "  FAIL  harvested ZERO durable-surface readers; a checker that examined nothing" >&2
  echo "        must not read as PASS" >&2
  bad=$((bad + 1))
fi
if [ "$n_unmarked" -gt 0 ]; then
  echo "  FAIL  $n_unmarked matcher(s) read the durable surface with no SUBJECT_READS marker" >&2
  bad=$((bad + 1))
fi
if [ "$bad_marked" -gt 0 ] || [ "$bad_names" -gt 0 ]; then
  echo "  FAIL  $bad_names SUBJECT_READS surface name(s) do not match what the code reads" >&2
  bad=$((bad + 1))
fi
if [ "$bad_decorative" -gt 0 ]; then
  echo "  FAIL  $bad_decorative decorative marker(s) on a matcher that reads nothing" >&2
  bad=$((bad + 1))
fi
# every marker must belong to a reader: markers == readers means no orphan markers
if [ "$marked_total" -ne "$n_readers" ]; then
  echo "  FAIL  $marked_total SUBJECT_READS marker(s) against $n_readers reader(s) -- an" >&2
  echo "        orphan marker or an unmarked reader; both are refused" >&2
  bad=$((bad + 1))
fi

echo
if [ "$bad" = 0 ]; then
  echo "DURABLE-READ COVERAGE: PASS ($n_readers reader(s), all marked, no orphan markers;"
  echo "  population derived from comment-stripped executable code, not a name grep)"
  echo "RESULT: every matrix that reads a durable record carries SUBJECT_READS naming it"
echo "        (population: comment-stripped executable code, the matcher's own variable"
echo "        assignments resolved, and git counted as a read operator -- so a read"
echo "        through a variable or through git IS visible; the reader set is derived,"
echo "        not asserted)"
  exit 0
fi
echo "DURABLE-READ COVERAGE: FAIL ($bad problem(s))" >&2
echo "RESULT: FAIL -- the durable-read population is unpinned" >&2
exit 1
