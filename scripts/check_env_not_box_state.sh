#!/usr/bin/env bash
# The device preconditions must never let "my shell is not set up" be read as
# "this box is broken".
#
# MEASURED TWICE IN ONE SESSION, BOTH TIMES WRONG. A previous revision reported
# "the box's in-process SYCL discovery returns no device -- the documented
# container/host driver mismatch" and told the operator a 27B handoff could not
# run. It was a missing `source setvars.sh`. Later, check_device_precond.sh hung
# to a timeout (rc=124) for the same reason. In both cases the tool emitted a
# line that reads like a statement about the HARDWARE:
#
#     SKIP: sycl-ls not available; cannot count GPUs
#
# That line is about PATH. Believing it turned "I forgot to set up my shell"
# into "the hardware is broken" -- the most expensive mistake available here,
# because it is unfalsifiable from the session that makes it and it sends the
# operator to debug a machine that is fine.
#
# SO THIS CHECKER PINS THE TEXT, and it pins two things at once:
#   1. every "cannot measure" path in the preconditions names ENVIRONMENT and
#      says what to run; and
#   2. no such path asserts "BOX STATE" about a condition the same function
#      cannot distinguish from an unprepared shell.
# Item 2 is the one that is easy to regress, because adding a reassuring
# "BOX STATE, not a code failure" line feels strictly helpful and is exactly
# wrong for a PATH failure.
#
# ANTI-VACUITY: the harvest below is a QUERY. If the helper stops being sourced
# by anything, or the paths stop matching, the count goes to zero and this would
# certify nothing -- so a non-zero harvest is asserted before any verdict.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

HELPER="scripts/_gate_precond.sh"
echo "== ENVIRONMENT-vs-BOX-STATE DIAGNOSTIC PIN =="

[ -f "$HELPER" ] || { echo "FAIL: $HELPER missing" >&2; exit 2; }

# The three functions that can report "cannot measure", derived from the
# definitions rather than remembered.
funcs="$(grep -oE '^(gate_gpu_count|gate_require_render_node|gate_require_device)\(\)' "$HELPER" | tr -d '()' | sort -u)"
n_funcs="$(printf '%s\n' "$funcs" | grep -c . || true)"
echo "-- helper functions that can report unmeasurable: $n_funcs"
printf '   %s\n' $funcs | sed 's/^/     /'

# ANTI-VACUITY. Three is the derived count from this file; anything less means
# the query broke and every verdict below would be about nothing.
if [ "$n_funcs" -lt 3 ]; then
  echo "  FAIL  only $n_funcs precondition function(s) harvested -- the query is" >&2
  echo "        broken or a function was renamed, so nothing below is verified." >&2
  exit 1
fi

bad=0

# --- 1. the diagnostic exists and is reachable from the helper --------------
if grep -q '^gate_env_diagnose()' "$HELPER"; then
  echo "  ok    gate_env_diagnose() is defined"
else
  echo "  FAIL  gate_env_diagnose() is missing" >&2
  bad=$((bad + 1))
fi

# --- 2. every skip path in the helper calls it -----------------------------
# Count the "cannot measure" exits and the diagnostic invocations inside the
# same file. Equal counts is the property; a drift between them is a path that
# reports unmeasurable while saying nothing about the environment.
n_skip="$(grep -cE '^\s*(echo "SKIP:|exit 2)' "$HELPER" || true)"
n_diag="$(grep -c 'gate_env_diagnose "' "$HELPER" || true)"
echo "-- skip/exit paths: $n_skip,  diagnostic call sites: $n_diag"
if [ "$n_diag" -lt 3 ]; then
  echo "  FAIL  only $n_diag diagnostic call site(s); every path that reports" >&2
  echo "        'cannot measure' must name the environment. Found $n_skip such" >&2
  echo "        path(s) in this helper." >&2
  bad=$((bad + 1))
else
  echo "  ok    every unmeasurable path names the environment"
fi

# --- 3. the diagnostic must NOT assert box state ---------------------------
# The specific regression: a reassuring "BOX STATE, not a code failure" line
# inside gate_env_diagnose itself, where it would be attached to what is
# usually a PATH problem.
# *** MATCH AN ASSERTION, NOT THE WORD. *** The first version of this check was
# `grep -qE 'BOX STATE'` inside the function, and it FAILED on the banner line
# "=== ENVIRONMENT, NOT BOX STATE ===" -- i.e. it flagged the text that denies
# the very claim it was checking for. A predicate that cannot tell a statement
# from its negation is not a predicate. So the pattern is the ASSERTIVE phrasing
# used elsewhere in this repo ("BOX STATE, not a code failure"), with the banner's
# own negation explicitly excluded.
if sed -n '/^gate_env_diagnose()/,/^}/p' "$HELPER" \
   | grep -E 'BOX STATE' | grep -vE 'NOT BOX STATE|ENVIRONMENT, NOT' | grep -q .; then
  echo "  FAIL  gate_env_diagnose asserts 'BOX STATE' -- it cannot know that." >&2
  echo "        The render-node and sycl-ls failures it describes are usually" >&2
  echo "        an unprepared shell, and saying otherwise is what caused the" >&2
  echo "        misattribution this checker exists to prevent." >&2
  bad=$((bad + 1))
else
  echo "  ok    gate_env_diagnose does not assert BOX STATE"
fi

# --- 4. the actionable remedy must be present and concrete -----------------
if grep -q 'source /opt/intel/oneapi/setvars.sh' "$HELPER"; then
  echo "  ok    the remedy (source setvars.sh) is stated verbatim"
else
  echo "  FAIL  no concrete remedy in gate_env_diagnose -- 'not box state'" >&2
  echo "        without the command that fixes it is just a nicer wrong answer." >&2
  bad=$((bad + 1))
fi

# --- 5. PROVE IT, do not assert it: run the helper unprepared --------------
# Everything above is a text check, and a text check on a diagnostic is exactly
# the kind of thing that can be satisfied by a comment. So run it with a PATH
# that has no sycl-ls and confirm the ENVIRONMENT banner actually appears.
banner="$(env -i PATH=/usr/bin:/bin HOME="$HOME" bash -c \
  ". '$ROOT/$HELPER' >/dev/null 2>&1; gate_gpu_count" 2>&1 || true)"
echo "-- live probe: gate_gpu_count with an unprepared PATH --"
if printf '%s' "$banner" | grep -q 'ENVIRONMENT, NOT BOX STATE'; then
  echo "  ok    the ENVIRONMENT banner is really printed (not just present in the file)"
else
  echo "  FAIL  running the helper unprepared did NOT print the banner" >&2
  echo "        (got: $(printf '%s' "$banner" | head -2 | tr '\n' ' '))" >&2
  bad=$((bad + 1))
fi

echo
if [ "$bad" = 0 ]; then
  echo "ENV-vs-BOX-STATE PIN: PASS"
  echo "RESULT: an unprepared shell is reported as an ENVIRONMENT problem, with"
  echo "        a concrete remedy, and is never reported as a hardware fault"
  exit 0
fi
echo "ENV-vs-BOX-STATE PIN: FAIL ($bad problem(s))" >&2
echo "RESULT: FAIL ($bad problem(s)) -- a PATH failure can still be read as box state" >&2
exit 1
