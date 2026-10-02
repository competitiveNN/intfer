#!/usr/bin/env bash
# Assert the ENGINE certification on disk is still evidence for this tree.
#
# WHY (AUDIT #24). `run_all_checks.sh` certifies the TOOLING. `scripts/
# run_ten_gates.sh` certifies the ENGINE and writes build/ten_gates/HEAD naming
# the tree it measured. For a while the two were only ever reconciled BY HAND, and
# a session reported "10 PASS / 0 PARTIAL / 0 SKIPPED" justified by an argument --
# "the files that changed since the gates ran are not engine inputs". That
# argument was sound and it was STILL NOT EVIDENCE: it is AUDIT #22's shape, where
# a number's backing was a claim about the tree instead of a file on disk.
#
# This mechanises exactly that argument, so it cannot be forgotten or skipped:
#   * artifact names HEAD                      -> certified here            (0)
#   * artifact older, but NO engine input moved -> the evidence CARRIES      (0)
#   * artifact older AND an engine input moved  -> STALE, re-run the gates   (1)
#   * no artifact at all                        -> UNPROVEN, never certified (2)
#
# The middle case is the whole point. A tooling-only commit must NOT invalidate a
# 45-minute engine certification -- that would make the stack unusable and would
# train people to ignore it, which is how such a check gets "fixed" by weakening
# it. Only a change to what the gates actually READ does.
#
# Deliberately NOT folded into gate_inventory.sh: that checker derives the gate
# TALLY from preconditions, while this asks a different question -- whether the
# measurement backing that tally is still about this code.
#
# Usage: scripts/check_engine_cert.sh
# Exit: 0 = certified or carrying, 1 = stale (engine input changed), 2 = unproven.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
enum_reset

ART="$ROOT/build/ten_gates/HEAD"
echo "== ENGINE CERTIFICATION CHECK =="

# What the correctness gates actually read. Derived from the gate population
# itself: scripts/test_*.sh, plus the engine and the shared helpers those gates
# source. A hand-written list here would rot the first time a gate gains a helper.
GATE_POP="$(ls "$SCRIPT_DIR"/test_*.sh 2>/dev/null | wc -l | tr -d ' ')"
enum_require_min "gate-scripts-present" "$GATE_POP" 5 || {
  echo "  FAIL  fewer than 5 gate scripts found -- the input set is not derivable"
  echo "RESULT: cannot certify; the engine input set did not enumerate"
  exit 1; }

if [ ! -f "$ART" ]; then
  echo "  note  no $ART -- the engine has never been certified on this box"
  echo "  note  run scripts/run_ten_gates.sh (tens of minutes, needs both GPUs)"
  echo "RESULT: UNPROVEN -- no engine certification on disk (not a regression)"
  exit 2
fi

CERT_SHA="$(sed -n 's/^HEAD=//p' "$ART" | head -1)"
CERT_SHORT="$(printf '%s' "$CERT_SHA" | cut -c1-7)"
enum_require_min "cert-sha-length" "${#CERT_SHA}" 7 || {
  echo "  FAIL  $ART does not name a usable HEAD"
  echo "RESULT: engine certification artifact is malformed"
  exit 1; }
CERT_TALLY="$(sed -n 's/^pass=//p' "$ART" | head -1)"
echo "  artifact: $ART"
echo "  certified tree: $CERT_SHORT   tally: pass=$CERT_TALLY ..."

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [ "$CERT_SHA" = "$HEAD_SHA" ]; then
  echo "  ok    the certification names HEAD exactly -- no argument needed"
  echo "RESULT: engine certified at HEAD (pass=$CERT_TALLY)"
  exit 0
fi

# Older than HEAD. Does the difference touch anything a gate reads?
CHANGED="$(git diff --name-only "$CERT_SHA"..HEAD 2>/dev/null || true)"
TOUCHED="$(printf '%s\n' "$CHANGED" | grep -E '^(src/|examples/|include/|scripts/test_|scripts/_gate_precond\.sh|scripts/_gate_build\.sh|scripts/setvars\.sh)' || true)"
n_touched="$(printf '%s\n' "$TOUCHED" | grep -c . || true)"

if [ "$n_touched" -eq 0 ]; then
  n_changed="$(printf '%s\n' "$CHANGED" | grep -c . || true)"
  echo "  ok    certified at $CERT_SHORT, HEAD is ${HEAD_SHA:0:7}; $n_changed file(s) changed"
  echo "        and NONE is an engine or gate input, so the evidence CARRIES."
  echo "        (List: $CHANGED)" | tr '\n' ' '; echo
  echo "RESULT: engine certification carries to HEAD -- no engine input changed"
  exit 0
fi

echo "  FAIL  the engine certification is STALE."
echo "        Certified at $CERT_SHORT; these engine/gate inputs changed since:"
printf '%s\n' "$TOUCHED" | sed 's/^/          /'
echo "        Re-run scripts/run_ten_gates.sh to re-certify. Until then the"
echo "        'N PASS' tally is evidence for $CERT_SHORT, not for this tree."
echo "RESULT: engine certification STALE -- engine input changed since it was measured"
exit 1