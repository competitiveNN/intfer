#!/usr/bin/env bash
# Committed behavioural self-check for scripts/_gate_enum.sh, the SHARED
# vacuity guard used by check_doc_counts.sh, check_build_dir_consumers.sh and
# gate_inventory.sh.
#
# WHY A HARNESS FOR A 20-LINE HELPER. This session produced the vacuous-green
# class THREE times -- a tally regex that matched nothing and reported "0 stale"
# (gate_inventory, audit #14), a count sentence whose absence was WARNed and
# skipped at exit 0 (check_doc_counts), and a self-exclusion widened until the
# checker printed "PASS (0 consumer(s), all honor INTFER_BUILD_DIR)" at exit 0
# (check_build_dir_consumers, found by its own matrix state M-E). A helper
# standing in for that class across three callers needs its own proof, and a
# proof in a commit message is not one.
#
# FOUR cases, because the quiet direction is the one that gets forgotten:
#   (a) zero harvested        -> must REJECT
#   (b) empty/non-numeric     -> must REJECT (fails CLOSED, not open)
#   (c) below a minimum       -> must REJECT
#   (d) well-formed, satisfied-> must ACCEPT, SILENTLY
# (d) is what makes (a)-(c) mean something: without it, a helper that rejects
# everything scores 3/3 and proves nothing. This is the same converse the repo
# requires of check_device_precond.sh's self-check and mutate_run_all_checks'
# N-Kq/N-L.
#
# Behavioural: it drives the real functions, not a grep of their source (audit
# #13: a self-check that greps itself can never fail, because the pattern sits
# inside the check).
#
# Usage: scripts/check_gate_enum.sh
# Exit:  0 = all four cases behaved as specified, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_gate_enum.sh"

echo "== GATE ENUM SELF-CHECK (_gate_enum.sh) =="
bad=0
SELFCHECK_OUT="$(mktemp "${TMPDIR:-/tmp}/intfer_enum_sc.XXXXXX")"
trap 'rm -f "$SELFCHECK_OUT"' EXIT

must_reject() { # $1 label, then: func + that func's own arguments, forwarded WHOLE
  local label="$1"; shift
  enum_reset
  local out rc
  "$@" > "$SELFCHECK_OUT" 2>&1; rc=$?
  out="$(cat "$SELFCHECK_OUT")"
  if [ "$rc" -eq 0 ] || [ "$(enum_bad)" != "1" ]; then
    echo "  FAIL  $label -- helper ACCEPTED it (rc=$rc, enum_bad=$(enum_bad))"
    printf '%s\n' "$out" | sed 's/^/          /'
    bad=$((bad+1))
  else
    echo "  ok    $label -- rejected, and said why"
  fi
}
must_accept() { # $1 label, then: func + that func's own arguments, forwarded WHOLE
  local label="$1"; shift
  enum_reset
  local out rc
  "$@" > "$SELFCHECK_OUT" 2>&1; rc=$?
  out="$(cat "$SELFCHECK_OUT")"
  if [ "$rc" -ne 0 ] || [ "$(enum_bad)" != "0" ]; then
    echo "  FAIL  $label -- helper REJECTED well-formed input (rc=$rc)"
    printf '%s\n' "$out" | sed 's/^/          /'
    bad=$((bad+1))
  elif printf '%s\n' "$out" | grep -q 'FAIL'; then
    echo "  FAIL  $label -- accepted but printed a FAIL line"
    printf '%s\n' "$out" | sed 's/^/          /'
    bad=$((bad+1))
  else
    echo "  ok    $label -- accepted silently"
  fi
}

# (a) zero harvested. This is the exact case the helper exists for.
must_reject "(a) zero harvested"        enum_require_nonempty "tally-sites" 0 "doc tally sites"
# (b) empty and non-numeric must fail CLOSED. A numeric test on a non-number is
# the vacuous-pass shape in miniature: `[ "" -gt 0 ]` is FALSE, so a naive
# implementation would treat "" as "fine".
must_reject "(b1) empty count"          enum_require_nonempty "x" ""  "subject"
must_reject "(b2) non-numeric count"    enum_require_min      "x" "abc" 1
# (c) below a stated minimum.
must_reject "(c) below minimum"         enum_require_min      "rows" 1 2
# (d) THE CONVERSE. Must accept, and must NOT print FAIL -- so a helper that
# rejects everything cannot score 3/3 above and call itself proven.
must_accept "(d) well-formed nonempty" enum_require_nonempty "tally-sites" 11 "doc tally sites"
must_accept "(d) minimum satisfied"    enum_require_min      "rows" 2 2

# (e) enum_reset actually resets, or the flags would stick across cases and
# every later assertion would be decided by the first.
enum_reset; enum_require_nonempty "sticky" 0 "x" >/dev/null 2>&1
enum_reset
if [ "$(enum_bad)" != "0" ]; then
  echo "  FAIL  (e) enum_reset does not clear the flag"
  bad=$((bad+1))
else
  echo "  ok    (e) enum_reset clears the failure flag"
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "GATE ENUM SELF-CHECK: PASS (must-reject and must-accept cases all correct)"
  echo "RESULT: _gate_enum.sh rejects 0/empty/non-numeric/below-minimum and accepts the well-formed case silently"
  exit 0
fi
echo "GATE ENUM SELF-CHECK: FAIL ($bad case(s))"
echo "RESULT: _gate_enum.sh does NOT behave as its callers assume"
exit 1