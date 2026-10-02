#!/usr/bin/env bash
# Benchmark ONE batch of report experiments (E1/E2/E3) and emit a keep/kill table.
#
# Protocol (deliberately strict, follows scripts/bench_quiet_ab.sh):
#   1. device preconditions, quiet gate (load1 < INTFER_QUIET_LOAD)
#   2. STREAM IDENTITY FIRST, on every arm, before any timing is believed
#   3. interleaved rounds (round-major, arm-minor), every round reported
#   4. wall time AND the [GPU profile] buckets side by side -- but SEPARATED BY
#      WHETHER THEY MAY BE BELIEVED: `nk`/`nw`/`nr` are exact counters and
#      `kernel` is device-side event time, while `submit`/`waitblock`/`copy`/
#      `readback`/`usm_write` are HOST wall clock and track the box. This line
#      used to call the whole set "load-insensitive", which is false, and the
#      summary it heads repeated it until AUDIT #26.
#
# Usage: scripts/bench_reports_batch.sh <E1|E2|E3|ARGMAX> [ROUNDS]
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"
TP="${INTFER_TP:-1}"
BATCH="${1:-}"
ROUNDS="${2:-3}"
# Per-batch: variable names unset with `env -u` on EVERY arm, before that arm's own
# vars are applied. Needed for presence-only flags, where there is no value that
# means OFF -- `INTFER_DEVICE_ARGMAX=0` still ENABLES it (main.cpp getenv site), so
# the only way to express OFF is for the variable to be ABSENT. Without this, a
# caller who exported the flag in their shell gets a BASE arm that is silently ON,
# and the A/B measures flag-vs-flag and reads as a wash.
UNSET_VARS=""
QLOAD="${INTFER_QUIET_LOAD:-2.0}"
# QWAIT / INTFER_QUIET_WAIT are GONE, not defaulted: the blocking wait they
# configured was removed (see the load-notification block below), and leaving a
# dead "QWAIT=0" behind would be a variable that reads as a knob and does nothing.
# scripts/bench_when_quiet.sh is now the single place that waits. Old command lines
# that still export INTFER_QUIET_WAIT are harmless -- an unread env var is not an
# error.
LOG="${INTFER_BENCH_LOG:-${INTFER_BUILD_DIR:-$ROOT/build}/bench_${BATCH}.log}"

# ARGMAX is a real batch here (the case below accepts it); the usage string said
# E1|E2|E3 only, so a user reading it would not know the lever's own batch name.
[ -n "$BATCH" ] || { echo "usage: $0 <E1|E2|E3|ARGMAX> [ROUNDS]"; exit 1; }
[ -x "$BIN" ] || { echo "FAIL: cli missing: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL"; exit 1; }
python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null \
  || { echo "SKIP: no render-node access from this shell."; exit 2; }
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"

NDEV="$(sycl-ls 2>/dev/null | grep -c '\[level_zero:gpu\]')"
echo "== batch $BATCH  model=$(basename "$MODEL")  tp=$TP  devices=$NDEV =="

# ---- arm table: name -> env assignments. env -u is used at run time so no
# arm can inherit another arm's flags.
case "$BATCH" in
  E1) ARMS="BASE DPAS48 PREQ"
      arm_env_BASE=""; arm_env_DPAS48="INTFER_DPAS48=1"; arm_env_PREQ="INTFER_DPAS48=1 INTFER_DPAS_PREQ=1"
      # invariant: PREQ must equal DPAS48 (both quantize); BASE is EXPECTED to differ
      REF_ARM="DPAS48"; ;;
  E2) ARMS="BASE KV KVQ8"
      arm_env_BASE=""; arm_env_KV="INTFER_INT8_KV=1"
      arm_env_KVQ8="INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1"
      REF_ARM="BASE" ;;   # all three must be byte-identical
  E3) ARMS="BASE DPAS48 FOLD FOLDPREQ"
      arm_env_BASE=""; arm_env_DPAS48="INTFER_DPAS48=1"
      arm_env_FOLD="INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1"
      arm_env_FOLDPREQ="INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1 INTFER_DPAS_PREQ=1"
      REF_ARM="DPAS48" ;;  # FOLD/FOLDPREQ must equal DPAS48
  # Device-argmax. NOT an E-batch: it is a separate lever with its own report
  # (reports/archive/2026-10-01-device-argmax.md), so it is named after the flag rather
  # than given an E-number -- E4 is already taken by the DPAS 2D numerics contract,
  # which has no flag at all and is therefore not benchable.
  #
  # *** TP>1 DECLINES THIS FLAG OUTRIGHT (main.cpp:1913) ***, so a --tp 2 A/B is a
  # silent NO-OP that would read as a wash. Refuse it here instead of quoting one:
  # a measurement of a declined path is worse than no measurement, because it looks
  # like evidence. (This is also why the 27B number is unobtainable for this lever:
  # 27B TP=1 OOMs on the lm_head tile, so the ONLY configuration where the flag does
  # anything is 0.8B TP=1. That is a fact about the lever, not a scoping choice.)
  ARGMAX) ARMS="BASE ARGMAX"
      arm_env_BASE=""; arm_env_ARGMAX="INTFER_DEVICE_ARGMAX=1"
      REF_ARM="BASE"    # the flag must be argmax-EXACT: identical streams
      # Presence-only flag: OFF means ABSENT, and `=0` still enables it. Unset on
      # every arm so no inherited value from the caller's shell can make BASE
      # silently ON -- which would compare the flag against itself and read as a
      # wash. Lives INSIDE this case branch; as its own `case` line it terminated
      # the branch early and left a stray `;;` (bash -n caught it; the load-gate
      # checker cannot, being a text checker -- see the note below).
      UNSET_VARS="INTFER_DEVICE_ARGMAX" ;;
  *) echo "FAIL: unknown batch '$BATCH'"; exit 1 ;;
esac

PROMPT="${INTFER_PROMPT:-The capital of France is}"
# shellcheck source=scripts/_gate_loadlog.sh
. "$SCRIPT_DIR/_gate_loadlog.sh"
# Per-arm logs + load traces live beside $LOG, not in the build root, so a figure
# can always be traced back to the arm that produced it (AUDIT #22 root cause 2).
loadlog_init "$(dirname "$LOG")/bench_logs/${BATCH}_$(date -u +%Y%m%dT%H%M%SZ)" >/dev/null

# ------------------------------------------------- non-vacuity, per batch
# *** PROVE THE LEVER MOVES BEFORE TRUSTING ANY A/B OF IT. *** The device-argmax
# report requires this by hand ("verify the OFF arm actually is OFF before trusting
# any A/B"), because that flag is presence-only: if the OFF arm were accidentally ON,
# both arms would be identical, the identity gate below would PASS (two equal
# streams), and the run would report a clean wash -- certifying that the lever does
# nothing when in fact nothing was toggled. That is the AUDIT #21 unobservable-state
# class applied to an A/B rather than to a checker.
#
# The probe is per-arm and asymmetric: OFF must emit 0 counters, ON must emit >=1.
# A probe that only asserted "OFF == 0" would be satisfied by a dead code path.
#
# It lives HERE, after PROMPT/TOK/UNSET_VARS exist. The first version sat directly
# after the device preconditions and read `$PROMPT` before it was assigned; under
# `set -u` that aborted BOTH probes, and the ON arm then reported "the lever is dead
# on this build" -- a confident false conclusion about a lever that was never run.
# The fail-closed direction was right and the diagnosis was wrong; same lesson as
# AUDIT #21's E-D ("a confident false conclusion about a mechanism that HAD been
# broken") and as this session's E-F, where a state was written before its
# precondition was measured.
if [ "$BATCH" = "ARGMAX" ]; then
  echo; echo "== NON-VACUITY: the flag must actually toggle =="
  if [ "$TP" != "1" ]; then
    echo "RESULT: TP=$TP DECLINES INTFER_DEVICE_ARGMAX outright (main.cpp:1913), so"
    echo "RESULT: this A/B would measure a no-op path and read as a wash. Re-run at"
    echo "RESULT: TP=1. (For 27B this is unobtainable: TP=1 OOMs on the lm_head tile.)"
    exit 1
  fi
  probe() { # $1 = "on"|"off" -> prints the [argmax] counter count
    local u="" v on=""
    for v in $UNSET_VARS; do u="$u -u $v"; done
    [ "$1" = on ] && on="INTFER_DEVICE_ARGMAX=1"
    # shellcheck disable=SC2086
    env $u $on INTFER_ARGMAX_DBG=1 \
      timeout 300 "$BIN" --gpu --tp "$TP" -m "$MODEL" -p "$PROMPT" -n 4 \
      --no-stream $TOK 2>&1 | grep -ac 'argmax' || true
  }
  n_off="$(probe off)"; n_on="$(probe on)"
  echo "  OFF arm: ${n_off:-0} [argmax] line(s)   (expect 0)"
  echo "  ON  arm: ${n_on:-0} [argmax] line(s)   (expect >= 1)"
  if [ "${n_off:-0}" -ne 0 ]; then
    echo "  FAIL  the OFF arm is NOT off -- a counter fired with the flag absent."
    echo "        Reporting this A/B would compare the flag against itself."
    echo "RESULT: OFF arm is not OFF -- the A/B would be vacuous"
    exit 1
  fi
  if [ "${n_on:-0}" -lt 1 ]; then
    echo "  FAIL  the ON arm emitted no counter -- the lever did not move, so a"
    echo "        wash would mean nothing rather than 'no gain'."
    echo "RESULT: ON arm did not engage -- the lever is dead on this build"
    exit 1
  fi
  echo "  ok    the flag demonstrably toggles; an A/B of it can mean something"
fi

TMPOUT="$(mktemp "${TMPDIR:-/tmp}/brb.XXXXXX")"
trap 'rm -f "$TMPOUT"' EXIT

# *** run_arm MUST NOT BE CALLED INSIDE $( ). *** loadlog_arm records its verdict in
# the globals _LL_USABLE / _LL_MAX / _LL_LOG, and a command substitution is a
# SUBSHELL -- those assignments die there and the parent reads them as UNSET, which
# under `set -u` aborts on `$_LL_MAX`. The repo already records this exact class for
# `_gate_enum.sh`'s `_enum_bad` ("a helper that records its verdict in a global
# silently loses it in $(...)", fix = capture to a file so the call runs in the
# current shell); it is the same bug again, one helper over.
#
# *** THIS BROKE BOTH PATHS OF THIS HARNESS SINCE AUDIT #23. *** All three call
# sites used `out="$(run_arm ...)"`. E1/E2/E3 were last measured at 05024c9 (10-01),
# BEFORE the loadlog plumbing landed at 281fd8f (10-02), so the recorded verdicts are
# valid for their tree -- but the harness has been unable to COMPLETE a timing or
# counters run since, and nothing caught it because the run dies at the first arm
# and the identity gate correctly refuses to compare arms that produced no ids. A
# harness that fails closed on its own bug still blocks the user, so "it failed
# loudly" was not good enough.
ARM_OUT=""
run_arm_capture() { # $1=arm $2=ntok [$3=log label] -> sets ARM_OUT; globals SURVIVE
  run_arm "$1" "$2" ${3:+"$3"} > "$TMPOUT"
  ARM_OUT="$(cat "$TMPOUT")"
}

run_arm() {  # $1=arm $2=ntok [$3=log label]  -> stdout = raw CLI output
  # `u` and `U` need an explicit empty value: `local u` leaves them UNSET, and the
  # very next line reads `$u`. Under `set -u` that aborts the arm, and because the
  # identity gate refuses to compare arms that produced no ids, the run stopped with
  # "identity check is VACUOUS" rather than reporting a wash -- fail-closed, but for
  # a reason that had nothing to do with the lever. `v` is local so the loop cannot
  # clobber a caller's variable.
  local arm="$1" ntok="$2" e rc u="" U="" v
  e="$(eval echo "\$arm_env_$arm")"
  # Unquoted on purpose: a word list, and empty must contribute NOTHING (a bare
  # `env -u` with no operand is an error, which would break the other three batches).
  for v in $UNSET_VARS; do u="$u -u $v"; done; U="$u"
  if [ "$#" -ge 3 ]; then
    # Logged + continuously sampled. The raw output is still echoed to stdout so
    # every existing caller keeps working, and the CLI's exit status is preserved
    # rather than being swallowed by the cat.
    loadlog_arm "$3" env $U $e timeout 900 "$BIN" --gpu --tp "$TP" -m "$MODEL" \
      -p "$PROMPT" -n "$ntok" --no-stream $TOK || rc=$?
    cat "$_LL_LOG"
    return "${rc:-0}"
  fi
  # shellcheck disable=SC2086
  env $U $e timeout 900 "$BIN" --gpu --tp "$TP" -m "$MODEL" \
    -p "$PROMPT" -n "$ntok" --no-stream $TOK 2>&1
}

load1() { awk '{print $1}' /proc/loadavg; }

# ------------------------------------------------- a FAILED ARM IS NOT A DATA ROW
# *** MEASURED, 2026-10-02, DURING THE E2 BATCH. *** The E2 arms died with
# `UR_RESULT_ERROR_DEVICE_LOST` / `FATAL: signal 6`, so the CLI printed **no
# `[GPU profile @decode]` line at all** -- and the harness recorded each of those
# arms as a perfectly ordinary row with `BUCKETS=""`, then printed a summary and
# exited **0**. That is the AUDIT #20 vacuous-green class one level above the
# checkers: a harvest that returned nothing was written down as if it had
# returned something, and the batch reported "done" for a batch where **nothing
# was measured**. The identity gate did not catch it either, because the
# identity phase runs 8 tokens and *did* succeed -- the device was lost later.
#
# So a missing bucket line is now a FIRST-CLASS outcome:
#   * the record carries `FAILED=<signature>` (spaces folded to `_`, because the
#     log format is whitespace-separated and a value containing one would make the
#     summariser's `kv.split("=", 1)` raise);
#   * it is counted and echoed loudly at the moment it happens;
#   * the batch ends rc=2 and says the batch is NOT A RESULT.
#
# *** rc=2, NOT rc=1, AND NOT rc=0. *** rc=0 would certify coverage nobody got;
# rc=1 would be indistinguishable from a code regression. And the harness
# deliberately does **not** diagnose the signature: `UR_RESULT_ERROR_DEVICE_LOST`
# is box state, but asserting that from a log line is the AUDIT #9/#11 mistake
# (a plausible mechanism recorded as a verified one). It prints the signature
# verbatim and points at the runbook instead.
ARM_FAILS=0
FAILED_ARMS=""
FAILED_SIGS=""
ARM_LINE=""
ARM_SIG=""
note_arm() { # $1=arm $2=round-label $3=raw arm output -> sets ARM_LINE / ARM_SIG
  ARM_LINE="$(printf '%s\n' "$3" | grep -aoE '@decode\] .*' | head -1)"
  ARM_SIG=""
  [ -n "$ARM_LINE" ] && return 0
  ARM_SIG="$(printf '%s\n' "$3" \
    | grep -aoE 'UR_RESULT_ERROR_[A-Z_]+|terminate called|signal [0-9]+|out of memory|std::[a-z_]+_error' \
    | head -2 | tr ' \n' '__')"
  [ -n "$ARM_SIG" ] || ARM_SIG="no_@decode_line_in_arm_output"
  ARM_FAILS=$((ARM_FAILS + 1))
  case " $FAILED_ARMS " in *" $1 "*) ;; *) FAILED_ARMS="$FAILED_ARMS $1" ;; esac
  FAILED_SIGS="$FAILED_SIGS
  $1/$2: $ARM_SIG"
}

finish_batch() { # refuse to certify a batch that lost an arm
  [ "$ARM_FAILS" -gt 0 ] || return 0
  echo
  echo "RESULT: $ARM_FAILS arm run(s) produced NO measurement at all, so this batch is"
  echo "RESULT: NOT A RESULT and its summary above is not evidence. Arms:$FAILED_ARMS"
  printf '%s\n' "$FAILED_SIGS" | sed 's/^/RESULT:   /'
  echo "RESULT: Do NOT diagnose from that signature. A driver error such as"
  echo "RESULT: UR_RESULT_ERROR_DEVICE_LOST is BOX STATE, not a code regression:"
  echo "RESULT:   scripts/gpu_recover.sh diag          (read-only, no root)"
  echo "RESULT:   sudo scripts/gpu_recover.sh rebind   (the remedy that worked before)"
  echo "RESULT:   scripts/gpu_recover.sh verify        (0 = 2 devices, 1 = still 1)"
  exit 2
}

# ---------------------------------------------------------------- 2. identity
echo; echo "== STREAM IDENTITY (before any timing) =="
declare -A IDS
for arm in $ARMS; do
  run_arm_capture "$arm" 8; out="$ARM_OUT"
  ids="$(printf '%s\n' "$out" | grep -ao 'decode step [0-9]*: token=[0-9]*' | grep -o '[0-9]*$' | tr '\n' ' ')"
  IDS[$arm]="$ids"
  printf '  %-9s %s\n' "$arm" "${ids:-<no ids>}"
done
REF="${IDS[$REF_ARM]}"
IDENT=0
# ANTI-VACUITY: an arm that produced NO ids must never compare equal. An empty
# stream used to satisfy `"" == ""` and print "OK", i.e. the check passed
# precisely when the run had failed (e.g. KV capacity abort). Every arm must
# produce a non-empty id list before any comparison is meaningful.
EMPTY=""
for arm in $ARMS; do
  if [ -z "${IDS[$arm]}" ]; then
    echo "  FAIL  $arm produced NO token ids (run failed - see output below)"
    EMPTY="$EMPTY $arm"
  fi
done
if [ -n "$EMPTY" ]; then
  echo
  echo "RESULT: identity check is VACUOUS for:$EMPTY - the run produced nothing."
  echo "RESULT: (typical cause: prompt exceeds KV capacity; raise INTFER_KV_CAP)"
  exit 1
fi
for arm in $ARMS; do
  [ "$arm" = "$REF_ARM" ] && continue
  if [ "${IDS[$arm]}" = "$REF" ]; then
    echo "  OK    $arm == $REF_ARM"
  elif [ "$arm" = "BASE" ] && [ "$REF_ARM" != "BASE" ]; then
    # Derived, not batch-named: whenever the reference arm is an activating
    # (quantizing) arm, the unquantized BASE is EXPECTED to differ from it.
    # Hard-coding this to one batch name is how E3 wrongly FAILED on this line.
    echo "  OK    BASE differs from $REF_ARM (EXPECTED: $REF_ARM quantizes acts)"
  else
    echo "  FAIL  $arm != $REF_ARM  <-- stream divergence, timings below are meaningless"
    IDENT=1
  fi
done
[ "$IDENT" -eq 0 ] || { echo; echo "RESULT: STREAM DIVERGENCE - stop, do not report timings."; exit 1; }

# ---------------------------------------------------------------- quiet gate
if [ "${INTFER_BENCH_TIMING:-1}" != "1" ]; then
  echo
  echo "== COUNTERS-ONLY MODE (INTFER_BENCH_TIMING=0) =="
  echo "Wall-time tok/s is DELIBERATELY NOT produced: the box is contended, so it"
  echo "would be variance, not signal. Reported below are load-insensitive counters"
  echo "(device-side kernel ms, nk/nw/nr) plus the stream-identity verdict above."
  echo "load1 at start = $(load1)"
  CNTLOG="$LOG.counters"
  : > "$CNTLOG"
  for arm in $ARMS; do
    run_arm_capture "$arm" 64 "cnt_${arm}"; out="$ARM_OUT"
    note_arm "$arm" "counters" "$out"; line="$ARM_LINE"
    if [ -n "$ARM_SIG" ]; then
      printf 'ARM=%s TPS=NOT-A-RESULT LOAD1=%s BUCKETS="" FAILED=%s\n' \
        "$arm" "$_LL_MAX" "$ARM_SIG" >> "$CNTLOG"
      echo "  !! $arm produced NO measurement: $ARM_SIG"
    else
      printf 'ARM=%s TPS=%s LOAD1=%s BUCKETS="%s"\n' \
        "$arm" "$(loadlog_tps_field '')" "$_LL_MAX" "$line" >> "$CNTLOG"
      echo "  $arm  $line"
    fi
  done
  LOG="$CNTLOG"
else
# ---- load: NOTIFY ONLY, NEVER WAIT (the blocking wait was removed 2026-10-02) --
# WHY IT WENT. This loop blocked for up to QWAIT (default 3600s) and then measured
# ANYWAY, so on a box held busy by a long-running neighbour it cost an hour and
# bought nothing. It was also a hand-rolled copy in three harnesses -- the AUDIT #7
# shape ("nine hand-rolled copies of the block is precisely how the family
# reopened"). There is now ONE home for "run this when the box is quiet":
# scripts/bench_when_quiet.sh.
#
# *** THE PROTECTION IS A DIFFERENT MECHANISM AND IS UNTOUCHED. *** Whether a figure
# is quotable is decided per-arm by loadlog_arm, on the MAXIMUM load over the whole
# arm -- not here, and not by this print. Removing the wait does not make a contended
# measurement quotable; it makes it visible sooner. Deleting the withholding would
# have been the AUDIT #22 defect re-opened, so it was NOT done.
L="$(load1)"
if awk -v l="$L" -v q="$QLOAD" 'BEGIN{exit !(l<q)}'; then
  echo "load1=$L at start: under $QLOAD"
else
  echo "load1=$L at start is >= $QLOAD. NOT waiting -- running anyway."
  echo "  Each arm whose MAXIMUM load crosses $QLOAD has its figure withheld as"
  echo "  SKIPPED-CONTENDED, so a busy box cannot produce a quotable number."
  echo "  For a quiet-window run use scripts/bench_when_quiet.sh."
fi

# ---------------------------------------------------------------- 3. timing
fi
if [ "${INTFER_BENCH_TIMING:-1}" = "1" ]; then
: > "$LOG"
echo; echo "== TIMING: decode -n64, interleaved, $ROUNDS rounds =="
for r in $(seq 1 "$ROUNDS"); do
  for arm in $ARMS; do
    # AUDIT #22/#23: the arm is logged and its load sampled CONTINUOUSLY, and the
    # TPS field is SKIPPED-CONTENDED unless the MAXIMUM load over the whole arm
    # stayed under the threshold. This loop used to scrape tok/s unconditionally
    # and record it, so a figure measured on a busy box landed in $LOG as a
    # machine-readable record that a human then pasted into a report.
    run_arm_capture "$arm" 64 "t${r}_${arm}"; out="$ARM_OUT"
    tps="$(printf '%s\n' "$out" | grep -aoE '\([0-9.]+ tok/s\)' | head -1 | tr -d '() tok/s')"
    note_arm "$arm" "r$r" "$out"; line="$ARM_LINE"
    if [ -n "$ARM_SIG" ]; then
      printf 'ARM=%s ROUND=%s TPS=NOT-A-RESULT LOAD1=%s BUCKETS="" FAILED=%s\n' \
        "$arm" "$r" "$_LL_MAX" "$ARM_SIG" >> "$LOG"
      echo "  !! r$r $arm produced NO measurement: $ARM_SIG"
      continue
    fi
    printf 'ARM=%s ROUND=%s TPS=%s LOAD1=%s BUCKETS="%s"\n' \
      "$arm" "$r" "$(loadlog_tps_field "${tps:-NA}")" "$_LL_MAX" "$line" >> "$LOG"
    echo "  r$r $arm  max_load1=$_LL_MAX  $(loadlog_tps_field "${tps:-NA} tok/s")"
    echo "     $line"
  done
done

# ---------------------------------------------------------------- 4. summary
# *** THE HEADER USED TO CALL THE WHOLE ROW "LOAD-INSENSITIVE". IT IS NOT. ***
# Only `kernel` is device-side event time. submit / waitblock / copy / readback /
# usm_write / rebuild are HOST wall-clock accumulators, and this repo's own reports
# already say so (`copy`, `usm_write` and `readback` are host-side and are NOT
# load-insensitive). Printing them in the same voice as `kernel` is how a contended
# run yields a "PREQ saves 3.9 s of submit" reading that is pure box load: the
# 2026-10-02 E1 run shows exactly that (submit 7308.5 -> 3403.6 ms at load1 4-17),
# and it is an artifact. A mislabelled table invites the wrong verdict; it does not
# merely mislabel one.
#
# *** AND A MEDIAN HIDES THE SPREAD, WHICH IS LARGER THAN THE EFFECTS BEING TESTED. ***
# The old table printed one median per bucket. On 27B, kernel ranges 16940-18816 ms
# across three rounds of the SAME arm and flags -- a 10.5% spread -- so a median
# delta of +689 ms reads as a measured cost when the per-round deltas are -58,
# +705 and +1545. Every bucket is now printed per round with its spread, and the
# delta table below states explicitly whether a median delta clears that spread.
#
# *** AND THE NOISE FLOOR ITSELF IS A PROPERTY OF THE RUN, NOT OF THE ARM. ***
# Measured the same day: the BASE arm's round-to-round kernel spread was 1875.5 ms
# in the E1 run and 64.5 ms in the E3 run (identical model, shape and round count),
# so the same arm's noise differs by ~29x between runs. The delta verdict is
# therefore conditional on the spreads shown beside it, and the table says so. Do
# not cache a spread, and do not quote "CLEARS the spread" without it.
echo; echo "== SUMMARY (wall tok/s per round, then median) =="
echo "   kernel + nk/nw/nr : DEVICE-side or exact -- may be believed."
echo "   submit/waitblock/copy/readback/usm_write : HOST wall clock, NOT"
echo "   load-insensitive. On a contended box they track the load and cannot"
echo "   decide a verdict. Shown as median(spread), for completeness."
# The summary lives in scripts/_bench_summary.py, NOT in a heredoc, so it can be
# driven against fixtures without a GPU -- see that file's docstring. Extracting
# it is what made the failed-arm defect below testable at all.
#
# *** ITS EXIT CODE IS PROPAGATED (AUDIT #28). *** It used to be discarded, which
# left TWO independent rules for "this batch is not a result": ARM_FAILS in the
# shell (an arm output with no @decode] line) and the summariser's own diagnosis
# (a record with no usable bucket data). They agreed on the measured failure, but
# a malformed record made the summariser raise -- printing NO summary at all --
# and the harness exited 0 having printed nothing. A rule enforced at two sites
# where only one site's verdict is checked is the AUDIT #3 shape; now there is
# one rule and the harness honours its verdict, whatever the cause.
python3 "$SCRIPT_DIR/_bench_summary.py" "$LOG" "$ARMS" "$REF_ARM"
SUM_RC=$?
if [ "$SUM_RC" -ne 0 ] && [ "$ARM_FAILS" -eq 0 ]; then
  # The shell counted no dead arm but the summary refused: do not explain which
  # is right. Report BOTH verdicts and refuse.
  echo
  echo "RESULT: the summary refused this batch (exit $SUM_RC) although no arm run"
  echo "RESULT: was flagged as dead by the harness, so the two disagree about what"
  echo "RESULT: was measured. The batch is NOT A RESULT either way. Re-read the"
  echo "RESULT: log above: a record with unusable bucket data is named in the"
  echo "RESULT: refusal block the summary printed."
fi
# Refuse to certify a batch that lost an arm -- checked LAST so the summary above
# is still printed (it is diagnostic) but the exit code cannot read as success.
finish_batch
# And refuse when the SUMMARY refused and the shell did not: same rc=2, for the
# same reasons (rc=1 is reserved for a code regression, rc=0 would certify
# coverage nobody got). This branch is load-bearing, not defensive -- without it
# the script falls through to its last `echo`, which exits 0.
if [ "$SUM_RC" -ne 0 ]; then
  echo
  echo "RESULT: NOT A RESULT -- see the refusal block above."
  exit 2
fi
echo "RESULT ($BATCH, $(basename "$MODEL") tp=$TP, devices=$NDEV) - NO wall-time claimed:"
echo "RESULT:"
fi
# Same refusal on the counters-only path, which has no summary block of its own.
[ "${INTFER_BENCH_TIMING:-1}" = "1" ] || finish_batch
