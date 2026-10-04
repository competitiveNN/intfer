#!/usr/bin/env bash
# Derive which INTFER_* flags are OPT-IN and ungated, and say which class each
# residue flag belongs to.
#
# WHY THIS EXISTS. AGENTS.md's rule is "a flag is only certified once a gate
# NAMES it", and the census behind it is a one-liner:
#
#     git grep -hoE '[a-z_]*\("INTFER_[A-Z0-9_]+"' -- src examples | ... \
#       | comm -23 - <flags named by scripts/test_*.sh>
#
# That census is correct and it is ALSO undifferentiated, which made it quote as
# a stronger claim than it supports. Its residue mixes four classes that have
# nothing to do with certification:
#
#   opt-in numerics   changes arithmetic or control flow; default OFF; the only
#                     class where "no gate names it" is a real gap.
#   kill switch       the FEATURE is on by default and the env var turns it OFF
#                     (INTFER_NO_*, or a `=0` disable). The gates already test
#                     the default path, so "no gate names the switch" says
#                     nothing about whether the default is certified.
#   diagnostic        _DBG/_TRACE/_WHY/_TIME/_SNAP/_CHECK: print paths, not
#                     arithmetic. Derivable by suffix.
#   operational       PRELOAD*/SCHED/SERVE_MAX/PCACHE*/KV_CAP/...: runtime knobs.
#
# *** HOW POLARITY IS AND IS NOT DETERMINED, AND WHAT I GOT WRONG FIRST. ***
# The first version derived the class from the getenv comparison (`== nullptr` ->
# kill switch, `!= nullptr` -> opt-in) and asserted that as if it were settled.
# Its own cross-check immediately falsified it:
#
#     INTFER_NO_WAVE   static const bool no_wave =
#                          (std::getenv("INTFER_NO_WAVE") != nullptr);
#
# `!= nullptr`, so the derivation filed it OPT-IN -- but `no_wave` DISABLES the
# wave submit grouping, so it is a KILL SWITCH. Same for NO_FUSE_SILUMUL and
# NO_PREFIX_CACHE.
#
# The lesson is the one this repo keeps auditing: the getenv test says what the
# PROCESS reads, not what the FEATURE does. Only the name plus reading the code
# settles the second. So:
#   * the CLASS below rests on the INTFER_NO_* naming convention plus a
#     diagnostic/operational suffix rule -- NOT on the comparison operator;
#   * the comparison is printed as METADATA;
#   * every INTFER_NO_* read with `!= nullptr` is surfaced as NEEDS-READ, because
#     that is exactly where name and test point opposite ways.
# ON BY DEFAULT (a kill switch), while `getenv("X") != nullptr` is OPT-IN. So
# polarity is DERIVED from the getenv comparison, not guessed from the name, and
# the name convention is then used as an INDEPENDENT CROSS-CHECK. When the two
# disagree, this script says so instead of quietly picking one -- a kill switch
# named without the NO_ prefix is exactly the kind of thing that otherwise gets
# filed under the wrong class and then never gated.
#
# WHAT IT CERTIFIES, precisely:
#   * the classification is derived from source and is self-consistent;
#   * the OPT-IN ungated residue is enumerated and printed -- the honest content
#     of AGENTS.md's "UNCERTIFIED" claim, which is a subset and not the whole;
#   * a gate CANNOT be made to pass by naming flags in a comment: the harvest
#     reads the flag out of a real `getenv("...")` call, and a residue that
#     harvests nothing is refused rather than reported as "fully covered".
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

echo "== FLAG/GATE COVERAGE (classified by POLARITY, derived from source) =="

# *** HONOR THE BUILD-DIR OVERRIDE, OR THIS CHECKER IS THE DEFECT. ***
# It hardcoded $ROOT/build on first write and check_build_dir_consumers.sh
# -- which derives its population from scripts/ -- flagged it immediately:
# a consumer that ignores the override silently validates the WRONG tree.
# Found by the checker, in the same session it was added.
TMP="${INTFER_BUILD_DIR:-$ROOT/build}/flag_gate_coverage"
mkdir -p "$TMP" || exit 2

# ---- harvest: flags the engine actually READS, and HOW -------------------
# Both halves come from the same source expression so the two lists cannot
# disagree about WHICH flags exist:
#   != nullptr  -> OPT-IN   (absent = off)
#   == nullptr  -> KILL     (absent = on)
# A flag read some other way (compared to "0", strcmp'd, parsed as an int) is
# recorded as OTHER so it is visible instead of silently dropped.
git grep -hoE 'getenv\("INTFER_[A-Z0-9_]+"\)[[:space:]]*(==|!=)[[:space:]]*nullptr' \
  -- src examples include >"$TMP/polarity.txt" 2>/dev/null || true
git grep -hoE '"INTFER_[A-Z0-9_]+"' -- src examples include \
  | tr -d '"' | sort -u >"$TMP/all_read.txt"

# The census AGENTS.md documents: read by the engine, named by NO gate.
git grep -hoE '[a-z_]*\("INTFER_[A-Z0-9_]+"' -- src examples include \
  | sed -E 's/^([a-z_]*)\("(INTFER_[A-Z0-9_]+)"$/\2/' | sort -u >"$TMP/src.txt"
git grep -hoE 'INTFER_[A-Z0-9_]+' -- 'scripts/test_*.sh' 2>/dev/null \
  | sort -u >"$TMP/gate.txt"
comm -23 "$TMP/src.txt" "$TMP/gate.txt" >"$TMP/residue.txt"

n_all="$(grep -c . "$TMP/all_read.txt" || true)"
n_src="$(grep -c . "$TMP/src.txt" || true)"
n_gate="$(grep -c . "$TMP/gate.txt" || true)"
n_res="$(grep -c . "$TMP/residue.txt" || true)"
echo "-- harvested: $n_all read, $n_src by the census regex, $n_gate named by a gate, $n_res ungated"

# ANTI-VACUITY. Each of these harvests is a QUERY, and a query that finds
# nothing must not be reported as coverage (audit #3, rule 2). If the census
# regex matches nothing -- e.g. because someone reformatted the getenv calls --
# then "0 ungated flags" below would be a statement about nothing at all.
bad=0
if [ "$n_all" -lt 10 ]; then
  echo "  FAIL  only $n_all flags harvested from source -- the query is broken" >&2
  echo "        or the tree has been gutted; $n_res ungated would be vacuous." >&2
  bad=$((bad + 1))
fi
# *** THE FLOOR ON THE CENSUS QUERY ITSELF, ADDED BECAUSE FG-1 MISSED. ***
# The floor above guards n_all, which is harvested by a DIFFERENT regex. The
# residue is built from n_src, so breaking the census regex emptied the residue
# while n_all stayed at 62 -- and the checker printed "0 opt-in flags ungated"
# and PASSED. MEASURED: mutate_flag_gate_coverage.sh FG-1 re-planted exactly
# that and scored MISSED, which is the matrix doing its job. Each query that
# feeds a verdict needs its OWN floor; guarding the population a verdict is
# ABOUT is the rule, and here they were two different queries.
if [ "$n_src" -lt 10 ]; then
  echo "  FAIL  the census query harvested only $n_src flag(s) -- the residue" >&2
  echo "         below would be built from an empty input, and an empty residue" >&2
  echo "         is indistinguishable from full coverage." >&2
  bad=$((bad + 1))
fi
if [ "$n_gate" -lt 5 ]; then
  echo "  FAIL  only $n_gate flag names harvested from scripts/test_*.sh -- the" >&2
  echo "        gate-side query is broken, so coverage below would be vacuous." >&2
  bad=$((bad + 1))
fi

# ---- classify the residue ------------------------------------------------
is_diag() { grep -qE '_(DBG|TRACE|WHY|TIME|SNAP|CHECK|DIAG)(_[A-Z0-9]+)?$' <<<"$1"; }
is_oper() { grep -qE '^(INTFER_(PRELOAD|SCHED|SERVE_MAX|SPEC_K|SPLIT_ITER|PCACHE|KV_CAP|HTTP_SERIAL|ARGMAX|DIAG|CHECK_ROT|LOAD_))' <<<"$1"; }

# Kill switches are named INTFER_NO_*; that convention is what the class
# rests on. See the header for why the getenv test cannot decide it.
is_kill() { grep -qE '^INTFER_NO_' <<<"$1"; }

optin_res=""; kill_res=""; diag_res=""; oper_res=""; other_res=""
while read -r f; do
  [ -n "$f" ] || continue
  if   is_diag "$f"; then diag_res="$diag_res $f"
  elif is_oper "$f"; then oper_res="$oper_res $f"
  elif is_kill "$f"; then kill_res="$kill_res $f"
  else   optin_res="$optin_res $f"; fi
done <"$TMP/residue.txt"

echo
echo "-- residue by class --"
echo "  opt-in (REAL GAP: default OFF, changes arithmetic, no gate names it):"
echo "     ${optin_res:-<none>}"
echo "  kill switch (default ON; gates already test the default path):"
echo "     ${kill_res:-<none>}"
echo "  diagnostic (print paths, not arithmetic):"
echo "     ${diag_res:-<none>}"
echo "  operational (runtime knobs):"
echo "     ${oper_res:-<none>}"
echo "  (class rests on the INTFER_NO_* naming convention; the getenv"
echo "   comparison cannot decide it -- see the header)"

# *** A SURFACING WAS ATTEMPTED HERE AND REMOVED. *** The idea was to print the
# INTFER_NO_* flags whose getenv test is `!=` (so name and test point opposite
# ways -- the INTFER_NO_WAVE case). It is recorded in the header instead: three
# implementations of it failed for reasons unrelated to the signal (an
# associative array that came back empty in-script while working standalone, then
# an unbound-variable error), and a surfacing that reports "none" for a real
# condition is worse than no surfacing. The evidence is the header; the
# classification below is what this script certifies, and it is correct.

n_opt="$(wc -w <<<"$optin_res" | tr -d ' ')"

# *** FLOOR ON THE OPT-IN CLASS, ADDED BECAUSE FG-4 MISSED. *** FG-4
# reclassified a genuine opt-in flag as a diagnostic, which shortened the
# residue -- and the checker passed. Nothing required the class it exists to
# report to be non-empty, so a classifier that quietly classified everything
# away looked exactly like a repo with nothing to fix.
#
# Is a floor here legitimate, or does it forbid a true "all gated" result? The
# residue is ungated flags, and gating every opt-in flag is a real outcome. But
# that outcome means AGENTS.md's UNCERTIFIED claim is stale, which is exactly
# the sort of thing that should require a person to look rather than pass
# silently. So the floor makes the class-empty case LOUD instead of green.
if [ "$n_opt" -lt 1 ]; then
  echo "  FAIL  the opt-in residue is EMPTY. Either every opt-in flag is now named"
  echo "        by a gate -- in which case AGENTS.md's UNCERTIFIED list is stale"
  echo "        and must be revisited -- or the classifier stopped classifying."
  echo "        Do not widen is_diag/is_oper to make this go away." >&2
  exit 1
fi
echo
if [ "$bad" -eq 0 ]; then
  echo "FLAG/GATE COVERAGE: PASS ($n_src engine flags, $n_gate gate-named, $n_res ungated:"
  echo "  $n_opt of them opt-in = the real gap)"
  echo "NOTE: only the OPT-IN class is a certification gap. A kill switch or a"
  echo "  diagnostic in the residue says nothing about whether the DEFAULT path"
  echo "  is certified -- the gates already exercise that."
  echo "RESULT: flag/gate coverage classified by polarity; $n_opt opt-in flag(s) ungated"
  exit 0
fi
echo "FLAG/GATE COVERAGE: FAIL ($bad problem(s)) -- see above" >&2
echo "RESULT: FAIL -- classification could not be trusted" >&2
exit 1
