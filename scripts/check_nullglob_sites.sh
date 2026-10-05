#!/usr/bin/env bash
# No file that ENABLES nullglob may pass a glob to a command with a meaningful
# zero-argument form.
#
# WHY (AUDIT #32). With `shopt -s nullglob` on, a glob that matches nothing expands
# to NOTHING -- not to the literal pattern. So
#     mapfile -t LOGS < <(ls -1 "$BUILD"/run_all_*.log 2>/dev/null)
# becomes bare `ls -1`, which LISTS THE CURRENT DIRECTORY. The 2>/dev/null does
# NOT protect: there is no error to suppress, there is simply a different
# command. An empty corpus therefore reported "14 logs found", the
# INCONCLUSIVE branch was unreachable, and the anti-vacuity state passed against a
# checker reading an unrelated directory listing.
#
# That is worse than the ordinary vacuous harvest, which at least yields 0 and
# trips a floor. Here the harvest returns a HEALTHY-LOOKING non-zero count, so
# every "did I find anything?" guard is satisfied by the wrong directory.
#
# THE POPULATION IS DERIVED, NOT REMEMBERED. Five files enable nullglob today and
# the sixth will be some future edit; the audit #7 rule (enumerate, never
# remember) is the whole point. Finding only the site you remember is exactly how
# the original instance survived a session in which it was diagnosed in detail.
#
# WHAT IS ALLOWED, and why:
#   arr=(glob) / mapfile -t arr < <(compgen -G glob)
#       the glob becomes the ARRAY, so no-match means empty and that is the answer.
#   a glob on a command that has NO meaningful zero-argument form
#       (grep, sed, test -e, awk) cannot degenerate this way.
# Everything else is a site.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

echo "== NULLGLOB SITE ENUMERATION (AUDIT #32) =="
[ -n "${1:-}" ] && echo "-- tree: $1"

# --- the population: files that turn nullglob on ---------------------------
# TREE is overridable so the mutation matrix can point the WHOLE check at a
# sandbox tree. Redirecting only the scanner is not enough: the file population
# comes from a second enumeration, and a state that exercises the scanner while
# the population still comes from the real scripts/ measures nothing.
TREE="${1:-scripts}"
[ -d "$TREE" ] || { echo "FAIL: $TREE is not a directory" >&2; exit 2; }

mapfile -t FILES < <(grep -l 'shopt -s nullglob' "$TREE"/*.sh 2>/dev/null | sort)
n_files="${#FILES[@]}"
echo "-- files enabling nullglob: $n_files"
[ "$n_files" -gt 0 ] && printf '     %s\n' "${FILES[@]##*/}"

# ANTI-VACUITY. If this scan stops finding the five files it knows about, the
# population has gone empty and every "0 sites" below would be a statement about
# nothing. The floor is the count, and it is deliberately >0.
if [ "$n_files" -lt 1 ]; then
  echo "  FAIL  no file enables nullglob -- either the scan is broken or they" >&2
  echo "        were all removed. Either way this checker is now vacuous." >&2
  exit 1
fi

# --- the sites, parsed -----------------------------------------------------
# A SEPARATE .py file, not a heredoc. AUDIT #32 is about a glob collapsing
# because of how it was WRITTEN, and a heredoc is exactly the quoting context
# that has gone wrong repeatedly here -- mangled backslashes, a prematurely
# terminated delimiter. The scanner for that class should not depend on it.
# Prefer a scanner INSIDE the tree when there is one: the mutation matrix puts a
# mutable copy in its sandbox so a state can blind the scanner without editing the
# real one. Falling back to the sibling copy keeps a normal tree working.
if [ -f "$TREE/_nullglob_scan.py" ]; then SC="$TREE/_nullglob_scan.py"
else SC="$(dirname "${BASH_SOURCE[0]}")/_nullglob_scan.py"; fi
scan_out="$(python3 "$SC" "$TREE" 2>&1)"
if ! printf '%s' "$scan_out" | grep -q '^POPULATION '; then
  echo "  FAIL  the scanner produced no POPULATION line:" >&2
  printf '%s\n' "$scan_out" | sed 's/^/        /' >&2
  exit 1
fi

population="$(printf '%s\n' "$scan_out" | sed -n 's/^POPULATION //p')"
mapfile -t SITES < <(printf '%s\n' "$scan_out" | sed -n 's/^SITE //p')

if ! printf '%s' "$scan_out" | grep -q '^POPULATION '; then
  echo "  FAIL  the scanner produced no POPULATION line:" >&2
  printf '%s\n' "$scan_out" | sed 's/^/        /' >&2
  exit 1
fi

n_sites="${#SITES[@]}"
echo "-- glob-bearing command invocations inside nullglob regions: $population"
echo "-- degenerate sites (a bare cmd GLOB that becomes a cwd listing): $n_sites"

bad=0
if [ "$n_sites" -eq 0 ]; then
  echo "  ok    no degenerate site: every glob in a nullglob region is consumed"
  echo "        by an array, by compgen, or is piped (a no-match yields an empty"
  echo "        stream rather than a listing of the current directory)"
else
  for s in "${SITES[@]}"; do
    echo "  FAIL  $s" >&2
    echo "        with nullglob on and no match, the glob vanishes and the" >&2
    echo "        command runs with NO arguments -- `ls` lists the CWD (AUDIT" >&2
    echo "        #32). Use arr=(glob), or mapfile -t arr < <(compgen -G glob)." >&2
    bad=$((bad + 1))
  done
fi

echo
if [ "$bad" = 0 ]; then
  echo "NULLGLOB SITES: PASS ($n_files file(s) enable it, 0 degenerate sites)"
  echo "RESULT: no glob can collapse into a listing of the current directory"
  exit 0
fi
echo "NULLGLOB SITES: FAIL ($bad degenerate site(s))" >&2
echo "RESULT: FAIL ($bad) -- a collapsed glob will read the wrong directory" >&2
exit 1
