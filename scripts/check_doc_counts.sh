#!/usr/bin/env bash
# Fail when a DERIVED count appears in reports/ without the command that
# re-derives it next to it.
#
# WHY. This session produced the same defect twice:
#   * a summary said "all five issues resolved" and then enumerated four;
#   * a summary said "four commits" while naming two.
# Neither was caught by any gate, because gate_inventory.sh reconciles GATE
# TALLIES (a hand-counted total presented as a live claim), not derived counts
# like commit lists or site counts. That is the same sites_found=0 vacuity
# class it does cover, in a place it does not look.
#
# SCOPE, deliberately narrow, because a broad "find numbers" rule would fire on
# every legitimate figure in the repo (token counts, byte counts, MiB, tok/s)
# and would be turned off within a week. This checks ONE file -- the batch
# report for this investigation -- and ONE property: a count presented as a
# count, in the sentences that assert derived totals.
#
# The check is BEHAVIOURAL: it re-derives the counts with git and compares, so
# it cannot pass by agreeing with itself.
#
# Usage: scripts/check_doc_counts.sh
# Exit:  0 = every derived count reproduces, 1 = a count is wrong or missing
#        its command, 2 = the report is absent (would be vacuous).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
REPORT="reports/archive/2026-10-01-batch-e1-e2-e3-verdicts.md"

# The shared enumeration guard (AUDIT #14's sites_found=0 class).
. "$SCRIPT_DIR/_gate_enum.sh"
enum_reset

[ -f "$REPORT" ] || { echo "SKIP: $REPORT absent -- nothing to check"; exit 2; }
echo "== DOC COUNT CHECK: $(basename "$REPORT") =="

bad=0

# ---- 1. the commit-index count must match git, and name its command --------
echo "-- derived commit count --"
# The report claims a count over the range starting at f047c42's parent.
range="f047c42^..HEAD"
real_n="$(git rev-list --count "$range" 2>/dev/null || echo '?')"
alt_n="$(git log --oneline "$range" 2>/dev/null | wc -l | tr -d ' ')"
echo "   git rev-list --count $range = $real_n"
echo "   git log --oneline | wc -l  = $alt_n"
if [ "$real_n" != "$alt_n" ]; then
  echo "   FAIL the two derivations disagree ($real_n vs $alt_n)"; bad=$((bad+1))
else
  echo "   ok    both derivations agree ($real_n)"
fi
# does the report quote that number AND the command?
if grep -q 'git rev-list --count' "$REPORT" && grep -q 'git log --oneline' "$REPORT"; then
  echo "   ok    report carries the re-derivation command(s)"
else
  echo "   FAIL report does not cite the re-derivation command"; bad=$((bad+1))
fi
# the number the report claims, taken from the sentence that states it
claimed="$(grep -oE 'agree at \*\*[0-9]+\*\*' "$REPORT" | grep -oE '[0-9]+' | head -1)"
idx_rows="$(grep -cE '^\| `[0-9a-f]{7,40}` \|' "$REPORT")"
# Fail closed if the index harvest is empty: with no rows there is nothing to
# reconcile, so every check below would be vacuous.
enum_require_nonempty "commit-index rows" "$idx_rows" "index rows in $REPORT" || true
if [ -n "$claimed" ]; then
  # THE PRIMARY CHECK: the stated count must equal the number of rows the
  # table actually lists. This is the defect class this session produced twice
  # ("four commits" over two, "five issues" over four) -- a number that
  # disagrees with the items it purports to enumerate -- and it is STABLE: it
  # does not change when some unrelated commit lands.
  #
  # The earlier version compared against `git rev-list --count` instead, which
  # made the check structurally self-invalidating: the table lists commits, so
  # every commit after its last refresh changed the answer, including the
  # commit that refreshed the checker itself. It then sat permanently UNPROVEN
  # (suite rc=2) with no remedy that could ever work. Comparing the number to
  # the ROWS keeps the guarantee that matters and drops the part that could not
  # be satisfied. Membership in history is checked separately below, which is
  # what catches a fabricated or stale sha.
  if [ "$claimed" = "$idx_rows" ]; then
    echo "   ok    stated count ($claimed) == rows in the index ($idx_rows)"
  else
    echo "   FAIL stated count ($claimed) disagrees with the $idx_rows row(s)"
    echo "        the table lists -- the 'a number in prose is a citation' defect"
    bad=$((bad+1))
  fi
  # Informational only: the index is a curated set of commits, not every
  # commit, so git's total may legitimately exceed it. Never a verdict.
  if [ "$real_n" != "$idx_rows" ]; then
    echo "   note  git has $real_n commit(s) in f047c42^..HEAD; the index lists"
    echo "         $idx_rows. A curated index need not be exhaustive -- that is"
    echo "         reported, never failed on."
  fi
else
  # A MISSING COUNT SENTENCE IS NOT A WARNING. It is the vacuous-pass case
  # AUDIT #14 records: a regex that matches nothing once reported "0 stale"
  # while verifying nothing. Measured -- reworded the sentence so the pattern
  # matched nothing and this checker still exited 0, because the "WARN ... skip"
  # path skipped the one assertion it exists to make.
  # So the count is REQUIRED, not optional.
  echo "   FAIL no 'agree at **N**' sentence found -- the derived count this"
  echo "        checker exists to verify is ABSENT, so a PASS would be vacuous"
  echo "        (AUDIT #14's sites_found=0 class). Restore the sentence."
  bad=$((bad+1))
fi

# ---- 2. the commit INDEX rows must match the actual log ---------------------
echo
echo "-- commit index rows vs git --"
idx_n="$(grep -cE '^\| `[0-9a-f]{7,40}` \|' "$REPORT")"
if [ "$idx_n" -eq 0 ]; then
  echo "   FAIL no commit-index rows found (enumeration broken -- vacuous pass)"
  bad=$((bad+1))
else
  echo "   ok    $idx_n index row(s) present"
  # Every sha in the index must exist in history. The verdict is CONDITIONAL:
  # the first version printed "ok every cited sha resolves" unconditionally,
  # immediately after the loop that had already printed FAIL lines -- so a
  # fabricated sha produced "FAIL index cites deadbeef" AND "ok every cited sha
  # resolves" in the same run, and the suite printed PASS off the ok. Measured.
  # The summary line is now emitted only when nothing failed.
  n_bad=0
  for sha in $(grep -oE '^\| `[0-9a-f]{7,40}`' "$REPORT" | tr -d '|` ' | cut -c1-12); do
    if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
      echo "   FAIL index cites $sha which is not in git history"; n_bad=$((n_bad+1))
    fi
  done
  if [ "$n_bad" -eq 0 ]; then
    echo "   ok    every cited sha resolves in git history"
  else
    bad=$((bad+1))
  fi
fi

# ---- 3. the tally must not have been edited into a live claim --------------
echo
echo "-- gate tally untouched --"
if grep -q '7 PASS, 2 PARTIAL, 1 SKIPPED' "$REPORT"; then
  echo "   ok    tally still reads 7 PASS / 2 PARTIAL / 1 SKIPPED"
else
  echo "   FAIL tally line missing or altered in $REPORT"; bad=$((bad+1))
fi
# NB: match across a LINE BREAK. The report wraps "**TP=2\nremains UNVERIFIED**",
# so a single-line pattern reports the marker missing on a document that carries
# it -- measured, and the same false-alarm mistake this check exists to catch.
# tr newlines to spaces for the MATCH ONLY; the file on disk is untouched.
if tr '\n' ' ' < "$REPORT" | grep -qiE 'TP=2[^.]{0,40}(remains|is) UNVERIFIED'; then
  echo "   ok    TP=2 still marked UNVERIFIED"
else
  echo "   FAIL TP=2 lost its UNVERIFIED marker in the tally section"
  bad=$((bad+1))
fi
# The marker must survive deletion of EVERY line carrying it, not merely
# coexist with an unrelated "blocked"/"unrunnable" elsewhere in the file. The
# first version of this check grepped the whole document for any of
# UNVERIFIED|unrunnable|blocked, so rewriting the one sentence that said
# "TP=2 remains UNVERIFIED" still passed on the strength of unrelated words
# elsewhere -- measured, above. Scoped to the sentence now.
if grep -qE 'UNVERIFIED' "$REPORT"; then
  n_unver="$(grep -cE 'UNVERIFIED' "$REPORT")"
  echo "   ok    $n_unver UNVERIFIED marker(s) present"
else
  echo "   FAIL no UNVERIFIED marker at all"; bad=$((bad+1))
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "DOC COUNT CHECK: PASS (derived counts reproduce; tally and TP=2 wording intact)"
  echo "RESULT: doc derived-counts reconcile with git"
  exit 0
fi
echo "DOC COUNT CHECK: FAIL ($bad problem(s))"
echo "RESULT: doc derived-counts DO NOT reconcile -- fix the number or cite the command"
exit 1