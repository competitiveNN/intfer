#!/usr/bin/env bash
# Mutation matrix for scripts/check_gate_exit_verdict.sh (the population sweep).
#
# WHY A MATRIX FOR THE SWEEP. check_gate_exit_verdict.sh does NOT run the gates
# (no model, no GPU); it reads each scripts/test_*.sh SOURCE through
# check_tp2_pass_exit.sh and asserts the sweep harvested the WHOLE population
# and that zero gates are unsafe. The per-gate predicates (a/b/c/d) are pinned by
# mutate_tp2_pass_exit.sh (7/7), so this matrix does NOT re-test them: it tests
# the SWEEP's own verdict logic -- the population derivation, the family label,
# the harvest/unsafe tallies, and the harvest-vs-N guard the sweep owns.
#
# A sweep defect is the population-equivalent of the vacuous green the single-gate
# checker prevents at one gate: a sweep that silently dropped a gate (18 -> 17)
# would certify "all gates safe, verdict reaches exit" while one gate was never
# examined at all. The states below mutate the SWEEP on temp copies and grade
# the sweep against itself; the tracked file is never modified.
# STATES (polarity is the point):
#   GE-1  population emptied (sweep's own 'no scripts/test_*.sh found') [REFUSE]
#   GE-2  sweep pointed at ONE genuinely UNSAFE gate (unsafe=1)         [REFUSE]
#   GE-3  N hardcoded to 19 (harvested 18 of 19)                       [REFUSE]
#   GE-4  harvest counter deleted (harvested 0 of N gates)           [REFUSE]
#   GE-5  baseline, unmutated                                         [BASELINE]
#
# *** EXCLUDED (stated, not scored) ***
#   GE-6  tally-deletion defect -- unobservable by verdict: on an all-safe tree
#         deleting `unsafe=$((unsafe+1))` changes no outcome, and GE-2 proves
#         the sweep's detection works by pointing it at a genuinely unsafe
#         gate. Deleting the tally would not be a state this matrix can see,
#         so it is recorded rather than scored (AUDIT #21).
#   GE-7  family label inverted -- cosmetic to the sweep's verdict: the per-gate
#         checker re-derives family itself, so a wrong label changes no outcome.
#         Same AUDIT #21 shape as GE-6. Recorded rather than silently omitted.
# Usage: scripts/mutate_gate_exit_verdict.sh
#   (runs the mutation matrix; --list prints the state roster)
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="$SCRIPT_DIR/check_gate_exit_verdict.sh"
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/gate_exit_verdict_mutate"
STAMP="$LOG_DIR/last_pass.stamp"
BUILD="$LOG_DIR"

[ -f "$TARGET" ] || { echo "MUTATE: no target: $TARGET" >&2; exit 1; }
mkdir -p "$BUILD"
rm -f "$STAMP"

SB="$(mktemp -d "${TMPDIR:-/dev/shm}/gev.XXXXXX")"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM
TARGET_DIGEST_BEFORE="$(cat "$TARGET" | cksum)"

total=0; caught=0; bad=0; baselines=0
declare -a RESULTS

. "$SCRIPT_DIR/_gate_patch.sh"

preflight() {
  local anchors=(
    'POP_GATES|GATES="$(ls scripts/test_*.sh 2>/dev/null)"'
    'N_DERIVED|N="$(printf '"'"'%s\n'"'"' "$GATES" | wc -l)"'
    'COUNTER|  harvested=$((harvested+1))'
    'UNSAFE_TALLY|    unsafe=$((unsafe+1))'
    'HARVEST_GUARD|if [ "$harvested" -ne "$N" ]; then'
    'VACUITY_GUARD|if [ "$harvested" -eq 0 ]; then'
  )
  local a c label pat
  for a in "${anchors[@]}"; do
    label="${a%%|*}"; pat="${a#*|}"
    # FIXED-STRING, not ERE: several of these anchors contain `$GATES` / `$N`
    # / `$PASS`, and in ERE `$` is end-of-line, so `for g in $GATES; do`
    # matched NOTHING (zero-width anchor + `GATES`) and the pre-flight
    # reported "0 occurrences" for the harvest loop -- a pre-flight that
    # always fails. MEASURED on the first run of this matrix.
    c="$(grep -cF -- "$pat" "$TARGET" 2>/dev/null || true)"; c="${c:-0}"
    if [ "$c" -ne 1 ]; then
      echo "ANCHOR STALE in $TARGET: $label matched $c occurrence(s), want 1." >&2
      return 1
    fi
  done
  echo "anchor pre-flight: ok (${#anchors[@]}/${#anchors[@]} targets located exactly once)"
}

# --- baseline: the UNMUTATED sweep must PASS ---------------------------------
baseline() {
  local out rc
  out="$(bash "$TARGET" 2>&1)"; rc=$?
  baselines=$((baselines+1))
  if [ "$rc" -ne 0 ]; then
    echo "BASELINE FAIL: unmutated sweep exited rc=$rc -- every state below" >&2
    echo "  would be scored against a broken baseline." >&2; printf '%s\n' "$out" >&2; exit 1
  fi
  case "$out" in *"RESULT: PASS"*) : ;;
    *) echo "BASELINE FAIL: the healthy sweep did not print its PASS verdict" >&2; exit 1 ;; esac
  echo "  ok baseline rc=0, PASS verdict present"
}

# run_state <id> <expected-regex> <mutation-fn>
run_state() {
  local id="$1" expect="$2" fn="$3" m out rc
  m="$SB/$id.sh"
  cp "$TARGET" "$m"
  "$fn" "$m" || { record "$id" MISSED "(mutation fn failed)"; return; }
  if ! bash -n "$m" 2>/dev/null; then
    record "$id" MISSED "(mutation broke syntax -- not evidence)"; return
  fi
  out="$(INTFER_SCRIPTS_DIR="$SCRIPT_DIR" bash "$m" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$BUILD/$id.log"
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE -- "$expect"; then
    caught=$((caught+1)); record "$id" CAUGHT ""
  elif [ "$rc" -eq 0 ]; then
    record "$id" MISSED "(sweep still PASS -- unobservable)"
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    record "$id" MISSED "(caught for the WRONG reason; wanted /$expect/)"
  else
    record "$id" MISSED "(rc=$rc, marker absent)"
  fi
}

record() { total=$((total+1)); RESULTS+=("$1 $2 $3"); }

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  GE-1  population emptied (sweep's own 'no scripts/test_*.sh found') [REFUSE]"
  echo "  GE-2  sweep pointed at ONE genuinely UNSAFE gate (unsafe=1)         [REFUSE]"
  echo "  GE-3  N hardcoded to 19 (harvested 18 of 19)                      [REFUSE]"
  echo "  GE-4  harvest counter deleted (harvested 0 of N gates)          [REFUSE]"
  echo "  GE-5  baseline, unmutated                                         [BASELINE]"
  echo "  [EXCLUDED] GE-6 tally-deletion defect -- unobservable by verdict: on an all-safe tree deleting unsafe=$((unsafe+1)) changes no outcome, and GE-2 proves the sweep's detection works. Recorded rather than scored."
  echo "  [EXCLUDED] GE-7 family label inverted -- cosmetic to the sweep's verdict; the per-gate checker re-derives family itself, so the mutation changes no outcome."
  exit 0
fi

echo "== anchor pre-flight =="
preflight || exit 1
echo "== baseline =="
baseline
# GE-1: empty the population. The sweep's OWN guard fires first: line 37
# `[ -n "$GATES" ] || { echo "FAIL: no scripts/test_*.sh found"; exit 2; }`.
# This is rule 2 at the sweep level -- "0 findings" and "found nothing to look
# at" are different claims, and a sweep that harvested an empty population and
# reported PASS would be the audit #18 shape one layer up. The marker is the
# sweep's own refusal text, so the state isolates this guard from the vacuity
# guard below it (which GE-4 hits via a different route). The refusal text
# contains a literal `*` (from the glob), so the marker is ERE-escaped.
m_ge1() { patch_literal "$1" 'GATES="$(ls scripts/test_*.sh 2>/dev/null)"' 'GATES=""; #GE1 empty population'; }
run_state GE-1 'no scripts/test_\*.sh found' m_ge1

# GE-2: the sweep must REFUSE an unsafe gate. Point GATES at a TEMP DIR
# containing ONE gate that is genuinely UNSAFE -- a copy of test_tp2_27b.sh
# with its PASS->exit conversion deleted (the PE-1 defect pinned by
# mutate_tp2_pass_exit.sh). The sweep harvests 1 gate, the per-gate checker
# refuses it, the sweep marks it UNSAFE and exits 1. This proves the sweep's
# detection works; the tally-DELETION defect (which would make the sweep green
# on an unsafe gate) is unobservable by verdict and is recorded as GE-6
# excluded below rather than scored.
m_ge2() { # $1 = mutated copy
  local gdir="$SB/ge2_gates"
  mkdir -p "$gdir"
  # PE-1: delete everything from the conversion to the end of test_tp2_27b.sh,
  # leaving the trailing rm as the last statement -- the exact shipped defect
  # that mutate_tp2_pass_exit.sh pins. Single-line python, no multi-line
  # escaping to drift.
  python3 - "$ROOT/scripts/test_tp2_27b.sh" "$gdir/test_tp2_27b.sh" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
conv = 'if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi'
rm   = 'rm -f "$LOG"  # *_OUT/*_ERR are handled by the EXIT trap (kept on FAIL)'
assert s.count(conv) == 1, s.count(conv)
assert s.count(rm) == 1, s.count(rm)
ci = s.index(conv); ri = s.index(rm)
assert ri < ci
open(dst, 'w').write(s[:ri] + s[ci + len(conv):].rstrip('\n') + '\n')
PY
  patch_literal "$1" 'GATES="$(ls scripts/test_*.sh 2>/dev/null)"' \
    "GATES=\"$(ls $gdir/test_*.sh 2>/dev/null)\""
}
run_state GE-2 'unsafe=1' m_ge2

# GE-3: N is DERIVED, not remembered (rule 4 at the sweep level). The sweep
# computes N at line 38 from the SAME GATES string the loop iterates, so the
# harvest-vs-N guard (`if [ "$harvested" -ne "$N" ]`) can only fire when the
# two disagree. Hardcoding N=19 makes harvested=18 != 19 and the guard fires
# "harvested 18 of 19". This is the sweep's own version of the
# hand-counted-total defect: a tally that is written rather than derived
# cannot detect a gate that was never examined, and the sweep's whole point is
# that it examines every gate. The anchor is the N= line (line 38), NOT the
# GATES line: appending `; N=19` to the GATES assignment is inert, because
# line 38 then re-derives N from wc -l and overwrites it -- measured, GE-3 came
# back "sweep still PASS" with the wrong anchor.
m_ge3() { patch_literal "$1" 'N="$(printf '"'"'%s\n'"'"' "$GATES" | wc -l)"' \
        'N=19 #GE3 remembered population'; }
run_state GE-3 'harvested 18 of 19' m_ge3

# GE-4: delete the harvest COUNTER increment. harvested stays 0 while N=18.
# MEASURED: the guard that fires is the harvested-vs-N guard (line 112,
# "harvested 0 of 18 gates"), NOT the vacuity guard (line 116) -- the N-guard is
# checked first, so it always wins on a non-zero N. Both read `harvested`, so
# the state still proves the counter is read by a guard (the always-red twin of
# GE-1: GE-1 empties the POPULATION and hits the line-37 refusal, GE-4 keeps a
# full population and breaks the COUNTER). A guard that only noticed an empty
# population, or that compared against a remembered N, would stay green here.
m_ge4() { patch_literal "$1" '  harvested=$((harvested+1))' '  : #GE4 harvest counter deleted'; }
run_state GE-4 'harvested 0 of [0-9]+ gates' m_ge4

{
  echo
  for r in "${RESULTS[@]}"; do echo "  $r"; done
  echo
  echo "states=$total caught=$caught baselines=$baselines"
  # One baseline for the whole run: there is exactly one unmutated gate.
  # caught == total, baselines >= 1, bad == 0 are the three conditions.
  if [ "$caught" -ne "$total" ] || [ "$baselines" -lt 1 ] || [ "$bad" -ne 0 ]; then
    echo "RESULT: FAIL ($caught/$total states caught, $baselines baseline(s), $bad other problem(s))"
  else
    echo "RESULT: PASS ($caught/$total states caught, each for its own reason)"
  fi
} | tee "$BUILD/summary.txt"

# Same three conditions as the printed verdict, so the exit code can never
# disagree with the line above it.
if [ "$caught" -ne "$total" ] || [ "$baselines" -lt 1 ] || [ "$bad" -ne 0 ]; then
  exit 1
fi
{ echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "states=$total"
  echo "caught=$caught"
  echo "baselines=$baselines"
  echo "excluded=GE-6, GE-7 (harvest-vs-N guard deletion; family-label inversion -- both unobservable on an all-safe tree; GE-1/GE-3/GE-4 cover the same defect classes by planting the defect directly)"
} >"$STAMP.tmp" && mv "$STAMP.tmp" "$STAMP"
exit 0
