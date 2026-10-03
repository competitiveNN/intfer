#!/usr/bin/env bash
# SHARED: assert that a harvest actually harvested something, before any verdict
# derived from it is allowed to be reported as PASS.
#
# WHY THIS EXISTS. AUDIT #14's headline defect: a hand-counted-total checker
# whose classification regex matched nothing once reported "0 stale" and
# "nothing to fix" while verifying nothing. It looked healthy. The same shape
# bit this session twice in check_build_dir_consumers.sh:
#   * `grep -rhoE 'add_executable\(\s*[A-Za-z0-9_]+' | awk '{print $2}'`
#     yielded ZERO targets -- awk does not split on '(' -- and the checker
#     exited 2 looking like a working check that simply found no consumers.
#   * check_doc_counts.sh printed "WARN ... skip, not silently pass" when its
#     count sentence was absent, and exited 0.
#
# A checker that cannot harvest its subject cannot certify its subject. This
# helper is the one place that fact is enforced, so the next enumerator does not
# have to rediscover it.
#
# Usage:  enum_require_nonempty <label> <count> <what-should-have-been-found>
#         enum_require_min <label> <count> <minimum>
# Return: 0 when satisfied, 1 when NOT (caller must fail closed, never WARN).
#
# Sourced, not executed: see scripts/_gate_precond.sh for the sibling of this.
_enum_bad=0

enum_require_nonempty() { # $1 label, $2 count, $3 expected-subject
  if [ "${2:-0}" -gt 0 ] 2>/dev/null; then
    printf '  ok    [%s] harvested %s item(s)\n' "$1" "$2"
    return 0
  fi
  printf '  FAIL  [%s] harvested 0 items -- cannot certify "%s"\n' "$1" "$3"
  printf '        A checker that cannot harvest its subject verifies NOTHING\n'
  printf '        (AUDIT #14: sites_found=0 reported "nothing to fix" while\n'
  printf '        checking nothing). Fix the enumeration, do NOT soften this.\n'
  _enum_bad=1
  return 1
}

enum_require_min() { # $1 label, $2 count, $3 minimum
  if [ "${2:-0}" -ge "${3:-1}" ] 2>/dev/null; then
    printf '  ok    [%s] %s >= required minimum %s\n' "$1" "$2" "$3"
    return 0
  fi
  printf '  FAIL  [%s] found %s, need at least %s\n' "$1" "$2" "$3"
  _enum_bad=1
  return 1
}

enum_bad() { echo "$_enum_bad"; }
enum_reset() { _enum_bad=0; }

# ---- grep result-classification (AUDIT #21-adjacent tooling class) -----------
# grep returns 0 = match, 1 = NO MATCH, and >=2 = THE GREP ITSELF FAILED (a
# malformed pattern, a missing file, a bad option). Collapsing >=2 into "no
# match" is a silent lie: the query never ran, and the caller reports a verdict
# about code it never examined. This bit this session twice -- a pattern
# containing an unclosed `)` made grep exit 2, and that was read as a legitimate
# empty result rather than a broken query.
#
# grep_classify <fixed-string> <file>     -> 0 match, 1 no match, 2 query failed
# Callers MUST distinguish 2 from 1; use grep_classify_ok for the boolean case
# where an error should fail closed.
grep_classify() { # $1 = fixed string, $2 = file (BRE, so no regex metachars)
  local rc=0
  grep -qxF -- "$1" "$2" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -ge 2 ]; then
    echo "QUERY FAILED (grep rc=$rc) for '$1' in $2 -- the query did not run" >&2
    return 2
  fi
  return "$rc"
}

grep_classify_ok() { # true only on a REAL match; an error is false
  local rc=0
  grep -qxF -- "$1" "$2" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ]
}

# True when grep's non-zero status was a genuine "no match" rather than a
# failed query. Use after `grep ... || rc=$?` to refuse a verdict built on a
# query that never ran.
grep_rc_is_nomatch() { [ "$1" -eq 1 ]; }
