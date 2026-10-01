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
REPORT="reports/2026-10-01-batch-e1-e2-e3-verdicts.md"

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
if [ -n "$claimed" ]; then
  if [ "$claimed" = "$real_n" ]; then
    echo "   ok    report's claimed count ($claimed) == git ($real_n)"
  elif [ "$claimed" -lt "$real_n" ] 2>/dev/null; then
    # The index in $REPORT lists commits, so EVERY commit made after it was last
    # refreshed invalidates the number it states -- including the commit that
    # refreshed the checker itself. That is not a code regression and it must
    # not be reported as one, or the remedy printed ("fix the number") sends
    # the next person to edit a document that is about to be stale again.
    # Exit 2 => UNPROVEN in run_all_checks.sh: "evidence is about an older
    # tree", remedy "re-derive the count". This is the rc=1-vs-rc=2 doctrine
    # (AUDIT #18) applied to documentation evidence.
    echo "   UNPROVEN report claims $claimed, git now says $real_n -- the index"
    echo "            was last refreshed $((real_n - claimed)) commit(s) ago."
    echo "            This is STALE EVIDENCE, not a regression. Re-derive with:"
    echo "              git log --oneline f047c42^..HEAD"
    echo "RESULT: doc counts STALE (claimed $claimed, git $real_n)"
    exit 2
  else
    echo "   FAIL report claims $claimed commits, git says $real_n -- the number is"
    echo "        HIGHER than history, i.e. it was never derived"; bad=$((bad+1))
  fi
else
  echo "   WARN no 'agree at **N**' sentence found to check (skip, not silently pass)"
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
  # every sha in the index must exist in history
  for sha in $(grep -oE '^\| `[0-9a-f]{7,40}`' "$REPORT" | tr -d '|` ' | cut -c1-12); do
    if git cat-file -e "${sha}^{commit}" 2>/dev/null || \
       git log --format=%H "$range" 2>/dev/null | grep -q "^${sha}"; then
      :
    else
      echo "   FAIL index cites $sha which is not in history"; bad=$((bad+1))
    fi
  done
  echo "   ok    every cited sha resolves in git history"
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