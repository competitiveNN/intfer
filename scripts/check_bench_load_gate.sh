#!/usr/bin/env bash
# Assert every wall-time measurement harness gates its figures on load and logs
# every arm, through the ONE shared helper.
#
# WHY (AUDIT #23, following AUDIT #22). Three harnesses had each grown their own
# quiet gate, and all three did the same wrong thing: print
#   WARN: box never quieted (load1=... >= ...); results are variance, not signal
# and then measure and publish anyway. The block in the backlog report had the
# deeper version of the same bug -- it sampled load1 at the START of a round and
# then measured for minutes -- and a tok/s was published for arms whose own logs
# read TPS=SKIPPED-CONTENDED beside LOAD1=4.67.
#
# Two properties, both of which the defect violates and neither of which is a
# judgement call:
#   1. EVERY harness that can publish a figure routes through
#      scripts/_gate_loadlog.sh, so the criterion has exactly one home. "Nine
#      hand-rolled copies of the block is precisely how the family reopened"
#      (AUDIT #7) -- and the family DID reopen, three times.
#   2. NO harness writes a bare throughput field. A tok/s may only reach a log or
#      a report through loadlog_tps_field, which returns SKIPPED-CONTENDED for an
#      arm whose MAXIMUM load crossed the threshold.
#
# The population is DERIVED from the scripts themselves, never from a list of
# names this file remembers: a non-gate script that scrapes a timing figure.
# Comment lines are stripped first, because a prose mention of "tok/s" is not a
# figure -- the AUDIT #21 rule, and without it this checker would demand a load
# gate from run_all_checks.sh over a comment.
#
# *** CRITERION DISAGREEMENT, RECORDED HERE BECAUSE IT IS DELIBERATE. ***
# This file STRIPS comments before matching; its sibling
# scripts/check_build_dir_consumers.sh does NOT. Both choices are right for their
# own question, and a previous round of this audit proved the difference the hard
# way: a usage comment in _gate_loadlog.sh that spelled out a build-tree path was
# graded as a build-path consumer and turned the stack red, because that
# classifier does not strip. Here, NOT stripping would pull run_all_checks.sh into
# the population over a prose mention of "tok/s" and demand a load gate from a
# script that publishes no figure. So: strip when the match decides MEMBERSHIP of
# a population, because a comment is not a behaviour; do not strip when the match
# is only a hint. Do not "harmonise" these two without re-deriving both.
#
# *** AND A NOTE ON WRITING THIS NOTE, WHICH COST A FULL STACK RUN. ***
# The first draft of the paragraph above QUOTED the offending path verbatim, and
# that made THIS file a build-dir consumer, so check_build_dir_consumers.sh failed
# on it -- the third instance of the same class, and the second caused by
# documentation. A checker that spells out another checker's literal patterns will
# be graded by them. So the paths are described, never quoted, and if you need the
# exact spelling, read the sibling.
#
# Usage: scripts/check_bench_load_gate.sh
# Exit: 0 = every harness gated, 1 = drift, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
enum_reset

HELPER="scripts/_gate_loadlog.sh"
echo "== BENCH LOAD-GATE CHECK =="

# Strip comment lines. A prose mention of a figure is not a figure.
strip_comments() { sed 's/^[[:space:]]*#.*$//' "$1"; }

# ---- population, derived ---------------------------------------------------
POP=""
for f in scripts/*.sh; do
  case "$f" in
    scripts/test_*|scripts/check_*|scripts/mutate_*) continue ;;
  esac
  [ "$f" = "$HELPER" ] && continue          # the helper IS the home, not a consumer
  # A "timing figure" is a RATE (tok/s, "tokens in") or a LATENCY (TTFT, TPOT,
  # E2E). The latency arm was added when bench_chat.sh was wired: it measures
  # seconds off the SSE stream and quotes none of the rate tokens, so the original
  # pattern excluded a genuine wall-time harness -- which is precisely the
  # "derive, do not remember" lesson closing on itself. State B-G proves this
  # widened arm can fail.
  if strip_comments "$f" | grep -qE 'tok/s|tokens in|TTFT|TPOT|E2E'; then
    POP="$POP $f"
  fi
done
n_pop="$(printf '%s\n' $POP | grep -c .)"
# Anti-vacuity (AUDIT #20): a population of zero would make every assertion below
# vacuously true and this checker would report PASS over an empty census.
enum_require_min "bench-harness-population" "$n_pop" 3 || exit 1
printf '  ok    population: %s harness(es) can publish a wall-time figure\n' "$n_pop"
for f in $POP; do printf '          %s\n' "$f"; done

bad=0

# ---- (1) every harness routes through the one helper ----------------------
printf '%-4s %-34s %s\n' "" "harness" "load-gated via _gate_loadlog.sh"
for f in $POP; do
  if strip_comments "$f" | grep -q '_gate_loadlog\.sh'; then
    printf '  %-4s %-34s %s\n' "ok" "$(basename "$f")" "yes"
  else
    printf '  %-4s %-34s %s\n' "FAIL" "$(basename "$f")" \
      "NO -- it can publish a figure without the shared load gate"
    bad=$((bad+1))
  fi
done

# ---- (2) every harness routes its FIGURE through the gate ------------------
# Two accepted forms of "the figure was consulted before it was emitted", both
# real, neither a loophole:
#   $(loadlog_field ...)  a value routed through the withholding primitive
#   $_LL_USABLE           an emission gated on the arm's verdict (bench_chat.sh
#                         withholds a whole block of per-turn figures at once,
#                         so there is no single field to wrap)
#
# An earlier version of this property tried to detect "a bare throughput write"
# line-by-line. It was a FALSE-POSITIVE MACHINE and is recorded here so nobody
# reinstates it: the legit shapes all trip it -- the scrape itself
# (`tps="$(grep -aoE tok/s ...)"`), a printf whose format says `TPS=%s` while the
# VALUE arrives from loadlog_tps_field on the NEXT line, and `tok/s=$TPS_FIELD`
# which is already gated. A check that cries wolf is the mirror defect.
printf '\n%-4s %-34s %s\n' "" "harness" "figure consulted before emission"
for f in $POP; do
  if strip_comments "$f" | grep -qE '\$\(loadlog_(tps_)?field|_LL_USABLE'; then
    printf '  %-4s %-34s %s\n' "ok" "$(basename "$f")" "yes"
  else
    printf '  %-4s %-34s %s\n' "FAIL" "$(basename "$f")" \
      "NO -- it emits a figure without consulting the gate that withholds"
    bad=$((bad+1))
  fi
done

# ---- (3) exactly ONE home for the criterion -------------------------------
# If a harness redefines the helper's functions it owns the criterion again, and
# property (1) would still report "yes" because it sources the file. This is the
# copy-drift family (AUDITs #3/#12: a rule fixed at one site and re-derived at
# another), caught at the definition rather than at the call.
n_redef=0
for f in $POP; do
  d="$(strip_comments "$f" | grep -cE '^[[:space:]]*(loadlog_arm|loadlog_tps_field)[[:space:]]*\(\)' || true)"
  if pred_expect_absent "no-redefinition:$(basename "$f")" "$d" '^[1-9]'; then
    printf '  %-4s %-34s %s\n' "ok" "$(basename "$f")" "does not redefine the criterion"
  else
    printf '  %-4s %-34s %s\n' "FAIL" "$(basename "$f")" \
      "REDEFINES loadlog_arm/loadlog_tps_field -- the criterion has a second home"
    n_redef=$((n_redef+1)); bad=$((bad+1))
  fi
done
[ "$n_redef" -eq 0 ] && echo "  ok    the criterion has exactly one home ($HELPER)"

# ---- (4) the warn-and-continue shape is gone ------------------------------
# All three harnesses printed the CORRECT warning and then measured anyway. A
# warning that does not change behaviour is a comment.
n_warn=0
for f in $POP; do
  if strip_comments "$f" | grep -q 'box never quieted'; then
    echo "  note  $(basename "$f"): mentions 'box never quieted'"
  fi
  if strip_comments "$f" | grep -q 'results are variance, not signal'; then
    echo "  FAIL  $(basename "$f") still prints the warn-and-continue line"
    n_warn=$((n_warn+1)); bad=$((bad+1))
  fi
done
[ "$n_warn" -eq 0 ] && echo "  ok    no harness prints the warn-and-continue line"

# ---- (5) BEHAVIOURAL: the helper rejects an arm that starts quiet and goes busy
# (1)-(3) are text checks, and text checks were how the retracted table got in.
# This RUNS the helper against a fixture loadavg that DRIFTS, because a constant
# fixture cannot distinguish "maximum over the arm" from "first sample" -- which
# is the exact defect. Both directions are required: a gate that can only say
# USABLE is the bug, and one that can only say CONTENDED is useless.
TDIR="$(mktemp -d "${TMPDIR:-/tmp}/blg.XXXXXX")"
trap 'rm -rf "$TDIR"' EXIT
if [ ! -f "$HELPER" ]; then
  echo "  FAIL  $HELPER is missing -- the shared home does not exist"
  bad=$((bad+1))
else
  (
    . "./$HELPER"
    loadlog_init "$TDIR/run" >/dev/null
    ld="$TDIR/load"; printf '0.5\n' > "$ld"; LOADAVG="$ld"
    drift() { echo "decode: 65 tokens in 23000.0 ms (2.83 tok/s)"; sleep 1
              printf '7.5\n' > "$ld"; sleep 2; }
    loadlog_arm drift_arm drift >/dev/null 2>&1
    echo "DRIFT_USABLE=$_LL_USABLE DRIFT_MAX=$_LL_MAX DRIFT_TPS=$(loadlog_tps_field '2.83 tok/s')"
    printf '0.5\n' > "$ld"
    quiet() { echo "decode: 65 tokens in 23000.0 ms (2.83 tok/s)"; sleep 1; }
    loadlog_arm quiet_arm quiet >/dev/null 2>&1
    echo "QUIET_USABLE=$_LL_USABLE QUIET_MAX=$_LL_MAX QUIET_TPS=$(loadlog_tps_field '2.83 tok/s')"
  ) > "$TDIR/probe" 2>&1
  if grep -q 'DRIFT_USABLE=0' "$TDIR/probe" \
     && grep -q 'DRIFT_TPS=SKIPPED-CONTENDED' "$TDIR/probe"; then
    echo "  ok    behavioural: an arm quiet at 0.5 that hit 7.5 is WITHHELD"
  else
    echo "  FAIL  behavioural: the helper published a figure for a drifting arm."
    echo "        This is the retracted-defect mechanism; see $(basename "$0") probe:"
    sed 's/^/          /' "$TDIR/probe"
    bad=$((bad+1))
  fi
  if grep -q 'QUIET_USABLE=1' "$TDIR/probe" \
     && grep -q 'QUIET_TPS=2.83' "$TDIR/probe"; then
    echo "  ok    behavioural: a quiet arm still publishes (the gate is not always-red)"
  else
    echo "  FAIL  a quiet arm was also withheld -- the gate cannot say USABLE, so"
    echo "        USABLE proves nothing and every real run would be discarded."
    bad=$((bad+1))
  fi
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "BENCH LOAD-GATE CHECK: PASS ($n_pop harness(es), one shared home, no bare"
  echo "  throughput field, and the maximum-over-arm criterion proven in both directions)"
  echo "RESULT: every wall-time harness logs its arms and withholds contended figures"
  exit 0
fi
echo "BENCH LOAD-GATE CHECK: FAIL ($bad property/properties drifted)"
echo "RESULT: a wall-time harness can publish a figure without a load gate or a log"
exit 1
