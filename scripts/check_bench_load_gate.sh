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
trap 'rm -rf \"$TDIR\"' EXIT INT TERM
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
  # The guard must be LOUD, not silently withholding. Proven by consuming a figure
  # with no preceding loadlog_arm: before _ll_need existed, the empty verdict
  # compared unequal to "1" and loadlog_field returned SKIPPED-CONTENDED -- the
  # SAFE direction, but a SILENT one. A harness wired wrong would withhold every
  # figure forever and never say why. State B-H removes the guard; this notices.
  (
    . "./$HELPER"
    loadlog_init "$TDIR/noguard" >/dev/null
    out="$(loadlog_field '2.83 tok/s' 2>&1)"; rc=$?
    echo "NOGUARD_RC=$rc NOGUARD_OUT=$out"
  ) > "$TDIR/noguard" 2>&1
  if grep -q 'NOGUARD_RC=1' "$TDIR/noguard" \
     && grep -q 'loadlog_arm must run' "$TDIR/noguard"; then
    echo "  ok    behavioural: a figure consumed with no arm FAILS LOUDLY (rc=1)"
  else
    echo "  FAIL  consuming a figure with no preceding loadlog_arm did not fail loudly."
    echo "        It must refuse and say why; silent withholding is how a harness"
    echo "        wired wrong goes blank forever without a word. Got:"
    sed 's/^/          /' "$TDIR/noguard"
    bad=$((bad+1))
  fi
fi

# ---- (6) the gate is consulted AFTER the arm, so _ll_need stays unreachable ---
# _ll_need (the fail-loud guard on an unset verdict) is exercised ONLY by the
# fixture above, never by a real run -- which is correct, and also worth stating
# rather than leaving implied. It is unreachable precisely because every harness
# consults the verdict after loadlog_arm has set it. Reverse that order and every
# real figure would be REFUSED: a loud, total failure. So the invariant that keeps
# the guard a wiring-error detector instead of a production outage is itself a
# checked property, per derived harness, on comment-stripped source.
#
# *** The comment strip here was added because this check reproduced the bug it
# exists to prevent. *** Its first version flagged bench_chat.sh GATE-BEFORE-ARM
# on a COMMENT that merely mentions the verdict variable -- the same class as the
# usage comment that turned the stack red twice. Caught on its first run, which is
# the only reason a fresh script gets the benefit of the doubt.
printf '\n%-4s %-34s %s\n' "" "harness" "gate consulted after the arm"
for f in $POP; do
  arm_line="$(strip_comments "$f" | grep -n 'loadlog_arm' | head -1 | cut -d: -f1)"
  gate_line="$(strip_comments "$f" \
    | grep -nE 'loadlog_usable|_LL_USABLE|loadlog_(tps_)?field' | head -1 | cut -d: -f1)"
  if [ -z "$arm_line" ]; then
    why="no loadlog_arm -- covered by property 1"
  elif [ -z "$gate_line" ]; then
    why="no gate consulted -- covered by property 2"
  elif [ "$gate_line" -gt "$arm_line" ]; then
    why="ok (arm@$arm_line, gate@$gate_line)"
  else
    why="GATE-BEFORE-ARM (arm@$arm_line, gate@$gate_line)"
  fi
  if pred_expect_absent "order:$(basename "$f")" \
       "$(printf '%s' "$why" | grep -c 'GATE-BEFORE-ARM' || true)" '^[1-9]'; then
    printf '  %-4s %-34s %s\n' "ok" "$(basename "$f")" "$why"
  else
    printf '  %-4s %-34s %s\n' "FAIL" "$(basename "$f")" "$why"
    echo "        Every figure from this harness would be REFUSED at runtime by"
    echo "        _ll_need. Reorder: run the arm, THEN consult the verdict."
    bad=$((bad+1))
  fi
done

# ---- (7) BEHAVIOURAL: an arm that produced NO data is a FAILED record ---------
# *** MEASURED DEFECT, 2026-10-02, DURING THE E2 BATCH. *** Two E2 arms died with
# `UR_RESULT_ERROR_DEVICE_LOST`, so the CLI emitted no `[GPU profile @decode]` line
# at all. The harness recorded each of them as an ordinary row with `BUCKETS=""`,
# printed a summary full of dashes, and **exited 0**. A batch whose arms all died
# therefore reported itself as a completed measurement -- the AUDIT #20
# vacuous-green class one level above the checkers: absence of data written down
# as data. The load gate could never have caught it, because the gate withholds a
# figure it HAS; it has no opinion on a figure that was never produced.
#
# This runs scripts/_bench_summary.py (extracted from the harness precisely so it
# is drivable without a GPU) against fixtures in BOTH directions. The converse is
# not optional: a guard that fires on everything would satisfy a one-case test and
# make every healthy batch look broken.
SUM="$SCRIPT_DIR/_bench_summary.py"
FIX="$TDIR"
rec() { # $1=path $2=arm $3=round $4=kernel-ms ("" = the arm produced nothing)
  if [ -z "$4" ]; then
    printf 'ARM=%s ROUND=%s TPS=SKIPPED-CONTENDED LOAD1=3.10 BUCKETS=""\n' "$2" "$3" >> "$1"
  else
    printf 'ARM=%s ROUND=%s TPS=SKIPPED-CONTENDED LOAD1=3.10 BUCKETS="@decode] kernel=%sms submit=700.0ms waitblock=900.0ms copy=10.0ms readback=300.0ms usm_write=300.0ms nk=1000 nw=900 nr=90"\n' \
      "$2" "$3" "$4" >> "$1"
  fi
}
if [ ! -f "$SUM" ]; then
  echo "  FAIL  $SUM is missing -- the summary cannot be tested, so it is unchecked"
  bad=$((bad + 1))
else
  # Round-major interleaved, both arms in ONE log -- the shape the harness writes.
  # (The first version of this fixture put each arm in its OWN file and pointed the
  # summary at one of them, so the second arm had no records at all and the
  # healthy direction failed for a reason that had nothing to do with the summary.)
  : > "$FIX/ok.log"
  rec "$FIX/ok.log" BASE 1 1800.0; rec "$FIX/ok.log" VAR 1 1700.0
  rec "$FIX/ok.log" BASE 2 1801.0; rec "$FIX/ok.log" VAR 2 1701.0
  rec "$FIX/ok.log" BASE 3 1802.0; rec "$FIX/ok.log" VAR 3 1702.0
  python3 "$SUM" "$FIX/ok.log" "BASE VAR" BASE > "$FIX/ok.out" 2>&1
  rc_ok=$?
  # The converse: a healthy 3-round pair must print normally and stay silent.
  if pred_expect_present "summary-healthy-rc0" "$rc_ok" '^0$' \
     && pred_expect_absent "summary-healthy-notamarker" \
          "$(grep -c 'NOT A RESULT' "$FIX/ok.out" || true)" '^[1-9]' \
     && pred_expect_present "summary-healthy-delta" \
          "$(grep -c 'rounds 1,2,3' "$FIX/ok.out" || true)" '^[1-9]'; then
    echo "  ok    converse: a healthy 3-round batch prints a delta and stays quiet"
  else
    echo "  FAIL  the healthy direction is wrong (rc=$rc_ok). An always-red summary"
    echo "        would make this whole section prove nothing. Got:"
    sed 's/^/          /' "$FIX/ok.out"
    bad=$((bad + 1))
  fi

  # The firing direction: ONE arm of two lost every round, written in the OLD log
  # shape (`BUCKETS=""` with no FAILED tag) because that is exactly what is
  # already on disk from the lost E2 batch.
  : > "$FIX/dead.log"
  rec "$FIX/dead.log" BASE 1 ""; rec "$FIX/dead.log" VAR 1 ""
  python3 "$SUM" "$FIX/dead.log" "BASE VAR" BASE > "$FIX/dead.out" 2>&1
  rc_dead=$?
  if pred_expect_present "summary-dead-rc3" "$rc_dead" '^3$' \
     && pred_expect_present "summary-dead-marker" \
          "$(grep -c 'THIS BATCH IS NOT A RESULT' "$FIX/dead.out" || true)" '^[1-9]' \
     && pred_expect_present "summary-dead-names-arm" \
          "$(grep -c 'VAR r1: no_bucket_data_in_record' "$FIX/dead.out" || true)" '^[1-9]'; then
    echo "  ok    behavioural: a batch whose arms produced no data is REFUSED (rc=3)"
  else
    echo "  FAIL  a batch with a dead arm summarised cleanly (rc=$rc_dead). That is"
    echo "        the E2 defect: absence of data reported as a result. Got:"
    sed 's/^/          /' "$FIX/dead.out"
    bad=$((bad + 1))
  fi

  # And a PARTIAL death: VAR loses round 2 only. The delta must be computed over
  # the rounds BOTH arms have, paired by round identity, and the batch still
  # refused. Positional pairing would silently compare VAR's round 3 against
  # BASE's round 2.
  : > "$FIX/partial.log"
  rec "$FIX/partial.log" BASE 1 1800.0; rec "$FIX/partial.log" VAR 1 1700.0
  rec "$FIX/partial.log" BASE 2 1801.0; rec "$FIX/partial.log" VAR 2 ""
  rec "$FIX/partial.log" BASE 3 1802.0; rec "$FIX/partial.log" VAR 3 1702.0
  python3 "$SUM" "$FIX/partial.log" "BASE VAR" BASE > "$FIX/partial.out" 2>&1
  rc_part=$?
  if pred_expect_present "summary-partial-rc3" "$rc_part" '^3$' \
     && pred_expect_present "summary-partial-pairs-by-round" \
          "$(grep -c 'd=-100.0 -100.0 (rounds 1,3)' "$FIX/partial.out" || true)" '^[1-9]' \
     && pred_expect_present "summary-partial-verdict-3rounds" \
          "$(grep -cE '\-> (CLEARS|INSIDE)' "$FIX/partial.out" || true)" '^[1-9]'; then
    echo "  ok    behavioural: a half-dead arm is paired by ROUND (1,3) and refused"
  else
    echo "  FAIL  partial-death handling is wrong (rc=$rc_part). Positional pairing"
    echo "        would compare VAR round 3 against BASE round 2. Got:"
    sed 's/^/          /' "$FIX/partial.out"
    bad=$((bad + 1))
  fi

  # (7a) *** A SINGLE ROUND IN COMMON MUST NOT CLEAR ANYTHING. *** AUDIT #28,
  # MEASURED. The delta verdict was gated on the batch's round count -- the MAX
  # over arms -- while the delta itself is computed over the rounds the two arms
  # hold in COMMON. So an arm with one surviving round was scored against a
  # spread of 0.0, and the table printed
  #   d=+98.0 (rounds 2) ... -> CLEARS the spread: report +98.0 ms (+0.60%);
  #   every round agrees in sign: yes
  # which is precisely the false claim AUDIT #26 retracted for E1, arriving
  # through a different door -- and the sign check was vacuous, since one round
  # always agrees with itself.
  : > "$FIX/one.log"
  rec "$FIX/one.log" BASE 1 1800.0; rec "$FIX/one.log" VAR 1 ""
  rec "$FIX/one.log" BASE 2 1801.0; rec "$FIX/one.log" VAR 2 1701.0
  rec "$FIX/one.log" BASE 3 1802.0; rec "$FIX/one.log" VAR 3 ""
  python3 "$SUM" "$FIX/one.log" "BASE VAR" BASE > "$FIX/one.out" 2>&1
  rc_one=$?
  # Polarity matters here, so it is NAMED: a `grep -q CLEARS` alone would also
  # match the explanatory header line, and this is the inverted-predicate trap
  # (AGENTS.md: an inverted predicate scores correct code as broken). The verdict
  # lines are the ones that begin `-> `.
  if pred_expect_present "summary-oneround-rc3" "$rc_one" '^3$' \
     && pred_expect_present "summary-oneround-refused" \
          "$(grep -cE '\-> NO SPREAD ESTIMATE' "$FIX/one.out" || true)" '^[1-9]' \
     && pred_expect_present "summary-oneround-clears-count-zero" \
          "$(grep -cE '\-> CLEARS' "$FIX/one.out" || true)" '^0$'; then
    echo "  ok    behavioural: ONE round in common is refused, never 'CLEARS'"
  else
    echo "  FAIL  a one-round delta was scored against a zero noise floor"
    echo "        (rc=$rc_one). That reports a cost or a saving with no spread"
    echo "        estimate at all. Got:"
    sed 's/^/          /' "$FIX/one.out"
    bad=$((bad + 1))
  fi

  # (7b) *** THE CONVERSE OF (7a): a real 3-round clearing pair MUST still be
  # reported as clearing. *** Without this, (7a) would also be satisfied by a
  # summary that never says CLEARS -- the check would be unfalsifiable in the
  # direction that matters (AUDIT #19: prove the harness with a case it did not
  # set up).
  if pred_expect_present "summary-converse-clears" \
       "$(grep -cE '\-> CLEARS the spread' "$FIX/partial.out" || true)" '^[1-9]'; then
    echo "  ok    converse: a 3-round pair with a 100 ms effect does say CLEARS"
  else
    echo "  FAIL  a healthy 3-round pair with a 100 ms effect did not CLEAR, so"
    echo "        the refusal in (7a) could be an unfalsifiable always-refuse. Got:"
    grep -E '\-> (CLEARS|INSIDE|NO SPREAD)' "$FIX/partial.out" | sed 's/^/          /'
    bad=$((bad + 1))
  fi
  # And its mirror: a 3-round pair whose effect is INSIDE the run's own spread
  # must say INSIDE, not CLEARS. The healthy fixture above is a clearing case,
  # so without this one the table could refuse every delta and still pass (7a).
  : > "$FIX/noise.log"
  rec "$FIX/noise.log" BASE 1 1800.0; rec "$FIX/noise.log" VAR 1 1800.4
  rec "$FIX/noise.log" BASE 2 1801.0; rec "$FIX/noise.log" VAR 2 1799.6
  rec "$FIX/noise.log" BASE 3 1802.0; rec "$FIX/noise.log" VAR 3 1801.4
  python3 "$SUM" "$FIX/noise.log" "BASE VAR" BASE > "$FIX/noise.out" 2>&1
  rc_noise=$?
  if pred_expect_present "summary-converse-healthy-rc0" "$rc_noise" '^0$' \
     && pred_expect_present "summary-converse-inside" \
          "$(grep -cE '\-> INSIDE the spread' "$FIX/noise.out" || true)" '^[1-9]'; then
    echo "  ok    converse: a 3-round pair inside its own spread says INSIDE (rc=0)"
  else
    echo "  FAIL  a healthy batch whose delta sits inside its own spread was not"
    echo "        reported as INSIDE (rc=$rc_noise). That is how the AUDIT #26"
    echo "        'median +689 ms' reading was manufactured. Got:"
    grep -E '\-> (CLEARS|INSIDE|NO SPREAD)' "$FIX/noise.out" | sed 's/^/          /'
    bad=$((bad + 1))
  fi

  # (7c) *** A RECORD WHOSE BUCKETS ARE ONLY PARTIALLY RECOVERABLE IS A FAILED
  # RECORD, NOT A DASH. *** AUDIT #28, MEASURED: the reader used to split the
  # whole record on whitespace and build a dict, so the first token without '='
  # raised ValueError -- a batch printed NO summary and the harness exited 0
  # (see (7d)). The same truncation that produced the crash also used to yield
  # '-' in the exact-counter columns of an otherwise "good" round. Here the write
  # is cut mid-line, which is what a killed arm leaves behind.
  : > "$FIX/trunc.log"
  rec "$FIX/trunc.log" BASE 1 1800.0
  printf 'ARM=VAR ROUND=1 TPS=SKIPPED-CONTENDED LOAD1=3.10 BUCKETS="@decode] kernel=1700.0ms submit=700.0ms nk=1000"\n' >> "$FIX/trunc.log"
  python3 "$SUM" "$FIX/trunc.log" "BASE VAR" BASE > "$FIX/trunc.out" 2>&1
  rc_tr=$?
  if pred_expect_present "summary-trunc-rc3" "$rc_tr" '^3$' \
     && pred_expect_present "summary-trunc-names-missing" \
          "$(grep -c 'incomplete_bucket_line_missing' "$FIX/trunc.out" || true)" '^[1-9]' \
     && pred_expect_present "summary-trunc-no-traceback" \
          "$(grep -c 'Traceback' "$FIX/trunc.out" || true)" '^0$'; then
    echo "  ok    behavioural: a partially-written record is NAMED, not a dash"
  else
    echo "  FAIL  a partially-written record summarised cleanly or crashed"
    echo "        (rc=$rc_tr). Either way it is not a measurement. Got:"
    sed 's/^/          /' "$FIX/trunc.out"
    bad=$((bad + 1))
  fi

  # (7e) *** A RECORD WITH A MALFORMED TOKEN IS NAMED, NOT FATAL. *** AUDIT #28,
  # MEASURED: the reader used to build a dict from every whitespace token of the
  # record, so a single token without '=' raised ValueError -- traceback, no table
  # at all, exit 1, and the harness discarded the rc (see (7d)), so the batch
  # exited 0 having printed nothing. One bad record must cost that record, not the
  # whole batch. The stray `OK` is what a future positional field in the bucket
  # line would look like.
  : > "$FIX/malformed.log"
  rec "$FIX/malformed.log" BASE 1 1800.0; rec "$FIX/malformed.log" BASE 2 1801.0
  rec "$FIX/malformed.log" BASE 3 1802.0
  rec "$FIX/malformed.log" VAR 1 1701.0
  printf 'ARM=VAR ROUND=2 TPS=SKIPPED-CONTENDED LOAD1=3.10 BUCKETS="@decode] kernel=1702.0ms submit=700.0ms waitblock=900.0ms copy=10.0ms readback=300.0ms usm_write=300.0ms nk=1000 nw=900 nr=90" OK\n' >> "$FIX/malformed.log"
  rec "$FIX/malformed.log" VAR 3 1703.0
  python3 "$SUM" "$FIX/malformed.log" "BASE VAR" BASE > "$FIX/malformed.out" 2>&1
  rc_mal=$?
  if pred_expect_present "summary-malformed-rc3" "$rc_mal" '^3$' \
     && pred_expect_present "summary-malformed-names-token" \
          "$(grep -c 'malformed_record_token' "$FIX/malformed.out" || true)" '^[1-9]' \
     && pred_expect_present "summary-malformed-no-traceback" \
          "$(grep -c 'Traceback' "$FIX/malformed.out" || true)" '^0$' \
     && pred_expect_present "summary-malformed-keeps-other-rounds" \
          "$(grep -c 'd=-99.0 -99.0 (rounds 1,3)' "$FIX/malformed.out" || true)" '^[1-9]'; then
    echo "  ok    behavioural: one malformed record costs that record, not the batch"
  else
    echo "  FAIL  a malformed record aborted the summary or was summarised cleanly"
    echo "        (rc=$rc_mal). A crash prints NO table, which is strictly worse"
    echo "        than a named refusal. Got:"
    sed 's/^/          /' "$FIX/malformed.out"
    bad=$((bad + 1))
  fi

  # (8) *** END-TO-END THROUGH THE REAL HARNESS, WITH A STUB CLI. *** Everything
  # above drives scripts/_bench_summary.py directly, and (7d) checks the
  # propagation STATICALLY. Neither proves that running the harness for real
  # refuses -- and AUDIT #27 is precisely a defect that a fixture-only test suite
  # passed, because the only way to drive the summary was to run a 27B benchmark.
  # A stub CLI closes that: it satisfies the harness's preconditions and emits a
  # controlled stream of token ids (so the IDENTITY phase passes) plus, or
  # deliberately without, a `[GPU profile @decode]` line. No engine, no GPU work,
  # no model read -- measured at 0.9 s per invocation, so the matrix can afford it
  # in every state.
  #
  # BOTH DIRECTIONS ARE REQUIRED. A one-case test here is satisfiable by a harness
  # that always refuses, which would be worse than no test.
  #
  # *** NO RENDER NODE IS rc=2, NOT rc=0. *** The harness bails at its own
  # precondition with exit 2, which is the SAME code the refusal uses -- so on a
  # GPU-less box this case is indistinguishable from a pass unless the TEXT is
  # checked, and the honest outcome is "unproven", never a green line. That is the
  # AUDIT #3c family ("a gate that cannot exercise the configuration cannot
  # protect it") and the reason the verdict block below can exit 2.
  if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
    echo "  UNPROVEN  e2e-harness-refusal: no render node, so the harness stops at"
    echo "            its precondition and this case cannot run. The rest of this"
    echo "            checker is still graded; this one is NOT certified."
    e2e_unproven=1
  else
    e2e_unproven=0
    STUBDIR="$TDIR/stub/examples/intfer_cli"
    mkdir -p "$STUBDIR"
    : > "$TDIR/model.intifer"          # the harness only checks it exists
    write_stub() { # $1 = 0 no profile line | 1 complete profile line | 2 INCOMPLETE
      # *** THE MODE IS BAKED IN AT GENERATION TIME. *** The first version wrote
      # `case "$1" in` into the stub and escaped it as \$1, so the stub read the
      # CLI's own first argument ("--gpu") at RUN time, matched no case, emitted
      # no profile line in any direction, and the checker reported BOTH the
      # incomplete and the healthy direction refusing -- an always-red half that
      # looked like two harness defects. Assigning MODE="$1" here bakes the value
      # into the file; the stub then reads its own MODE.
      local mode="$1"
      cat > "$STUBDIR/intfer_cli" <<STUB
#!/usr/bin/env bash
# STUB CLI for case (8). Emits token ids so the IDENTITY phase passes.
MODE=$mode
while [ \$# -gt 0 ]; do case "\$1" in -n) want="\$2"; shift 2 ;; *) shift ;; esac; done
i=0; while [ "\$i" -lt "\${want:-8}" ]; do i=\$((i + 1)); echo "decode step \$i: token=\$((279 + i))"; done
echo "tok/s: (99.00 tok/s)"
case "\$MODE" in
  1) echo "[GPU profile @decode] kernel=1800.0ms submit=700.0ms waitblock=900.0ms rebuild=0.0ms(0) usm_write=300.0ms readback=300.0ms copy=10.0ms nk=1000 nw=900 nr=90" ;;
  # 2: a profile line with NO exact counters. This is the direction that reaches
  # the SUMMARY-PROPAGATION branch rather than finish_batch: note_arm finds the
  # line, so the shell counts NO dead arm (ARM_FAILS=0), while the summariser --
  # which requires kernel AND nk/nw/nr -- refuses. State B-O damages exactly
  # that branch's exit code, and it is invisible to the other two directions.
  2) echo "[GPU profile @decode] kernel=1800.0ms submit=700.0ms waitblock=900.0ms" ;;
esac
exit 0
STUB
      chmod +x "$STUBDIR/intfer_cli"
    }
    run_harness() { # $1 profile? -> writes $TDIR/e2e.out, returns the harness rc
      INTFER_BUILD_DIR="$TDIR/stub" INTFER_08B="$TDIR/model.intifer" \
        timeout 300 bash "$SCRIPT_DIR/bench_reports_batch.sh" E2 1 \
        > "$TDIR/e2e.out" 2>&1
    }
    # -- firing direction: no profile line, so every arm dies --
    write_stub 0
    run_harness; e2e_dead_rc=$?
    if pred_expect_present "e2e-dead-stub-rc2" "$e2e_dead_rc" '^2$' \
       && pred_expect_present "e2e-dead-stub-says-not-a-result" \
            "$(grep -c 'THIS BATCH IS NOT A RESULT' "$TDIR/e2e.out" || true)" '^[1-9]' \
       && pred_expect_present "e2e-dead-stub-names-the-signature" \
            "$(grep -c 'no_@decode_line_in_arm_output' "$TDIR/e2e.out" || true)" '^[1-9]'; then
      echo "  ok    e2e: the REAL harness refuses a batch whose arms all died (rc=2)"
    else
      echo "  FAIL  e2e-dead-stub: the harness did not refuse a batch in which"
      echo "        every arm produced no measurement (rc=$e2e_dead_rc). This is"
      echo "        the AUDIT #27 defect, driven through the real script. Got:"
      tail -12 "$TDIR/e2e.out" | sed 's/^/          /'
      bad=$((bad + 1))
    fi
    # -- the direction that exercises the PROPAGATION branch specifically --
    write_stub 2
    run_harness; e2e_inc_rc=$?
    if pred_expect_present "e2e-incomplete-rc2" "$e2e_inc_rc" '^2$' \
       && pred_expect_present "e2e-incomplete-uses-the-propagation-branch" \
            "$(grep -c 'NOT A RESULT -- see the refusal block above' "$TDIR/e2e.out" || true)" '^[1-9]' \
       && pred_expect_present "e2e-incomplete-names-the-missing-fields" \
            "$(grep -c 'incomplete_bucket_line_missing' "$TDIR/e2e.out" || true)" '^[1-9]' \
       && pred_expect_present "e2e-incomplete-is-not-the-dead-arm-branch" \
            "$(grep -c 'arm run(s) produced NO measurement at all' "$TDIR/e2e.out" || true)" '^0$'; then
      echo "  ok    e2e: the harness refuses on the SUMMARY's verdict alone"
      echo "        (shell counted no dead arm; the branch under test is the one"
      echo "        that makes B-O observable)"
    else
      echo "  FAIL  e2e-incomplete: rc=$e2e_inc_rc. This direction exists because"
      echo "        the other two cannot reach the propagation branch at all --"
      echo "        finish_batch exits 2 first, so a defect in the summary's exit"
      echo "        code is invisible without it. Got:"
      tail -12 "$TDIR/e2e.out" | sed 's/^/          /'
      bad=$((bad + 1))
    fi
    # -- converse: a profile line, so every arm measures --
    write_stub 1
    run_harness; e2e_ok_rc=$?
    if pred_expect_present "e2e-healthy-stub-rc0" "$e2e_ok_rc" '^0$' \
       && pred_expect_present "e2e-healthy-stub-not-refused" \
            "$(grep -c 'NO wall-time claimed' "$TDIR/e2e.out" || true)" '^[1-9]' \
       && pred_expect_present "e2e-healthy-stub-quiet" \
            "$(grep -c 'THIS BATCH IS NOT A RESULT' "$TDIR/e2e.out" || true)" '^0$'; then
      echo "  ok    converse: a batch whose arms all measure is accepted (rc=0)"
    else
      echo "  FAIL  e2e-healthy-stub: rc=$e2e_ok_rc. A harness that refuses"
      echo "        everything would pass the case above, so this one is required."
      tail -12 "$TDIR/e2e.out" | sed 's/^/          /'
      bad=$((bad + 1))
    fi
  fi

  # (7d) *** THE HARNESS MUST HONOUR THE SUMMARY'S VERDICT. *** STATIC, and
  # labelled so: driving the harness end-to-end needs a GPU and a model, and a
  # comment-shaped check would be worse than none (see the note at the top about
  # checks that read prose). So the population is DERIVED -- every harness that
  # invokes _bench_summary.py -- and each must both capture its exit code and
  # refuse on it. AUDIT #28: it did neither, so a summary that raised a
  # traceback produced no table and the batch still exited 0.
  n_sum=0
  for f in $POP; do
    strip_comments "$f" | grep -q '_bench_summary\.py' || continue
    n_sum=$((n_sum + 1))
    body="$(strip_comments "$f")"
    if printf '%s' "$body" | grep -q 'SUM_RC=\$?' \
       && printf '%s' "$body" | grep -q 'SUM_RC" -ne 0'; then
      printf '  %-4s %-34s %s\n' "ok" "$(basename "$f")" \
        "propagates the summary's refusal (static)"
    else
      printf '  %-4s %-34s %s\n' "FAIL" "$(basename "$f")" \
        "runs _bench_summary.py but neither captures nor refuses on its exit code"
      bad=$((bad + 1))
    fi
  done
  enum_require_min "summary-caller-rc-propagation" "$n_sum" 1 \
    "no harness invokes _bench_summary.py, so the propagation rule is unchecked"

  # And the harness side: a harness that writes MACHINE-READABLE per-arm records
  # must refuse a batch that lost an arm. Scoped by DERIVATION, not by name: only
  # a harness that writes `ARM=` records can accumulate absence-of-data into a
  # summary, and asserting the refusal of all seven harnesses would have produced
  # six false alarms on a healthy tree -- a check that cries wolf is the mirror
  # defect (AUDIT #15), and it would have trained everyone to ignore this line.
  for f in $POP; do
    strip_comments "$f" | grep -q 'ARM=' || continue
    if strip_comments "$f" | grep -q 'finish_batch'; then
      printf '  %-4s %-34s %s\n' "ok" "$(basename "$f")" "refuses a batch that lost an arm"
    else
      printf '  %-4s %-34s %s\n' "FAIL" "$(basename "$f")" \
        "writes ARM= records but never refuses a lost arm"
      bad=$((bad + 1))
    fi
  done
fi

echo
if [ "$bad" -eq 0 ]; then
  if [ "${e2e_unproven:-0}" -eq 1 ]; then
    echo "BENCH LOAD-GATE CHECK: UNPROVEN (case (8) could not run -- no render node)"
    echo "RESULT: every other property held, but the end-to-end refusal path was"
    echo "RESULT: NOT exercised, so this checker certifies nothing about it."
    exit 2
  fi
  echo "BENCH LOAD-GATE CHECK: PASS ($n_pop harness(es), one shared home, no bare"
  echo "  throughput field, the maximum-over-arm criterion proven in both directions,"
  echo "  a batch that lost an arm refused, and the refusal proven END-TO-END"
  echo "  through the real harness with a stub CLI)"
  echo "RESULT: every wall-time harness logs its arms, withholds contended figures,"
  echo "RESULT: and refuses to certify a batch whose arms produced no data"
  exit 0
fi
echo "BENCH LOAD-GATE CHECK: FAIL ($bad property/properties drifted)"
echo "RESULT: a wall-time harness can publish a figure without a load gate or a log,"
echo "RESULT: or can summarise a batch whose arms produced no data"
exit 1
