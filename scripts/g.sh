#!/usr/bin/env bash
# Fixed-string grep, default-on. For AD-HOC terminal use, not for scripts.
#
# WHY. grep returns 0 match / 1 no-match / >=2 THE GREP FAILED (malformed
# pattern, unreadable file). A pattern containing an unclosed ')' is a REGEX
# ERROR, and its empty result reads exactly like "no match". That mistake was
# made three times in one session -- twice against scripts/, once in a report
# grep -- and twice the "failure" was mistaken for a finding.
#
# scripts/lib/_gate_enum.sh (grep_classify) now guards this INSIDE the verification
# tooling. This covers the other half: greps typed by hand.
#
#   g.sh PATTERN FILE...      fixed string (like grep -F); no FILE = stdin
#   g.sh -r PATTERN DIR       recursive, quiet
#   g.sh -E PATTERN FILE...   opt into regex, with a warning
# Exit: 0 match, 1 no match, 2 the QUERY FAILED -- never reported as "no match".
set -u
if [ "${1:-}" = "-E" ]; then
  echo "g.sh: regex requested. Unescaped ( ) [ ] . * will ERROR, and a grep error" >&2
  echo "      is rc>=2, NOT a no-match. Prefer a fixed string." >&2
  shift; exec grep -E "$@"
fi
recursive=0
[ "${1:-}" = "-r" ] && { recursive=1; shift; }
if [ "$#" -lt 1 ]; then echo "usage: g.sh [-r] PATTERN [FILE...]" >&2; exit 2; fi
pat="$1"; shift
if [ "$recursive" = 1 ]; then
  [ "$#" -lt 1 ] && { echo "g.sh -r needs PATTERN and DIR" >&2; exit 2; }
  exec grep -rqlF -- "$pat" "$@"
fi
if [ "$#" -eq 0 ]; then grep -qF -- "$pat"; exit $?; fi   # stdin
for f in "$@"; do
  if [ ! -r "$f" ]; then echo "g.sh: cannot read $f" >&2; exit 2; fi
done
last=1
for f in "$@"; do
  grep -qF -- "$pat" "$f" 2>/dev/null; rc=$?
  case $rc in
    0) last=0 ;;
    1) : ;;
    *) echo "g.sh: QUERY FAILED on $f (grep rc=$rc) -- NOT a no-match" >&2; exit 2 ;;
  esac
done
exit "$last"
