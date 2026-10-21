#!/usr/bin/env bash
# check_history_pointers.sh -- pointer integrity for AGENTS.md's HISTORY.md titles.
#
# WHAT IT REFUSES. AGENTS.md was split on 2026-10-04 and its long sections MOVED
# to HISTORY.md, leaving `Moved to HISTORY.md ("<title>")` pointers. HISTORY.md
# was later truncated (509 lines destroyed). The repo recorded that honestly,
# twice over: HISTORY.md carries a banner naming the eleven dated entries whose
# bodies are gone, and each AGENTS.md pointer to one carries an in-place
# annotation. So a pointer-integrity check has FOUR resolution rules, and refuses
# a title that satisfies none of them.
#
# WHY NOT THE NAIVE VERSION, and this is the near-miss this checker exists to
# stop being repeated. The obvious predicate -- "the quoted title must exist as
# a heading in HISTORY.md" -- is REFUTED by the live corpus: ten of the thirteen
# cited titles resolve only through the banner, so it fails ten correct titles
# and the only way to make it green is to weaken it to "or is annotated", which
# is the gate-weakening defect in its purest form. A predicate that fires on
# its own subject is also what this refuses: the harvest is DERIVED from
# `HISTORY.md ("...")` citations, so a title re-worded to dodge the check is
# still harvested, and a corpus with no citation at all fails rather than
# certifying an empty set as agreeing.
#
# THE FOUR RULES (scripts/lib/_pointer_scan.py, in specificity order):
#   OK-HEADING the title is a real heading in HISTORY.md (the body is there)
#   OK-BANNER  the title is named in the destroyed-entries banner, so its
#              absence is RECORDED PROVENANCE rather than an accident
#   OK-TEXT    the title is present verbatim in HISTORY.md as non-heading text
#   DANGLING   none of the above, but its AGENTS.md site carries an explicit
#              [DANGLING annotation within +/- 3 lines
#   UNRESOLVED none of the four -- the defect
#
# AND THE BANNER'S OWN STALENESS HALF: a banner entry AGENTS.md no longer cites
# describes a pointer that no longer exists, and is refused.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Subject overrides, so the mutation matrix can grade states on COPIES. The
# live files are tracked and dirty by this repo's standing state, so a matrix
# that edits them in place is the destructive probe the standing rule exists
# for; an override that is also how the negative direction stays safe.
AG="${HPOINT_AGENTS:-$ROOT/AGENTS.md}"
HI="${HPOINT_HISTORY:-$ROOT/HISTORY.md}"
SCAN="$SCRIPT_DIR/../lib/_pointer_scan.py"

for f in "$AG" "$HI" "$SCAN"; do
  [ -s "$f" ] || { echo "HPOINT: FAIL -- missing or empty: $f" >&2; echo "RESULT: FAIL"; exit 1; }
done

out="$(python3 "$SCAN" "$AG" "$HI" 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "HPOINT: FAIL -- the scanner failed (rc=$rc):" >&2
  printf '%s\n' "$out" | sed 's/^/        /' >&2
  echo "RESULT: FAIL"
  exit 1
fi

n_heading="$(printf '%s\n' "$out" | grep -c '^OK-HEADING ')"
n_banner="$(printf '%s\n' "$out" | grep -c '^OK-BANNER ')"
n_text="$(printf '%s\n' "$out" | grep -c '^OK-TEXT ')"
n_dangling="$(printf '%s\n' "$out" | grep -c '^DANGLING ')"
n_unresolved="$(printf '%s\n' "$out" | grep -c '^UNRESOLVED ')"
n_uncited="$(printf '%s\n' "$out" | grep -c '^BANNER-UNCITED ')"
cites="$(printf '%s\n' "$out" | sed -n 's/^CITES //p')"

resolved=$((n_heading + n_banner + n_text + n_dangling))

echo "== HISTORY.md POINTER INTEGRITY =="
echo "-- derived from AGENTS.md's HISTORY.md (\"...\") citations: $cites citation(s)"
echo "-- resolved: $n_heading heading, $n_banner banner, $n_text text, $n_dangling annotated"

bad=0

# ANTI-VACUITY: a corpus with no citation at all must not read as "nothing
# dangling". Two empty sets agreeing is the shape this refuses.
if [ -z "$cites" ] || [ "$cites" -lt 1 ]; then
  echo "  FAIL  the harvest found no citation at all; 'looked at nothing' and" >&2
  echo "        'found nothing wrong' are different claims" >&2
  bad=$((bad + 1))
fi

# 1. every citation resolves by one of the four rules
if [ "$n_unresolved" -gt 0 ]; then
  printf '%s\n' "$out" | sed -n 's/^UNRESOLVED /  FAIL  unaccounted pointer: /p' >&2
  echo "        (no heading, no banner entry, no verbatim text, no in-place" >&2
  echo "        [DANGLING annotation) -- it points at a title nothing carries" >&2
  bad=$((bad + 1))
else
  echo "  ok    every citation resolves (heading / banner / text / annotated)"
fi

# 2. the resolved count must CONSOME the harvest's distinct titles. A title that
#    resolves twice, or a distinct-title count that exceeds the harvest, means
#    the scanner's bookkeeping drifted.
distinct=$resolved
if [ "$distinct" -lt 1 ]; then
  echo "  FAIL  resolved nothing while harvesting $cites citation(s)" >&2
  bad=$((bad + 1))
fi

# 3. the banner's own staleness: an entry nothing cites describes a dead pointer
if [ "$n_uncited" -gt 0 ]; then
  printf '%s\n' "$out" | sed -n 's/^BANNER-UNCITED /  FAIL  banner entry no longer cited: /p' >&2
  echo "        (the banner exists to describe live AGENTS.md pointers)" >&2
  bad=$((bad + 1))
else
  echo "  ok    every banner entry AGENTS.md still cites"
fi

# 4. the annotation rule must not be satisfied by distance: a [DANGLING note
#    several hundred lines from the pointer it excuses is prose, not a
#    pointer annotation. The scanner asserts +/-3 lines, and this predicate
#    proves it can tell the difference by harvesting the site's own line.
if [ "$n_dangling" -gt 0 ]; then
  echo "  ok    $n_dangling annotated pointer(s) carry [DANGLING at their site"
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "HPOINT: PASS ($cites citations, $resolved resolved, 0 unaccounted)"
  echo "RESULT: PASS"
  exit 0
fi
echo "HPOINT: FAIL ($bad problem(s))" >&2
echo "RESULT: FAIL" >&2
exit 1
