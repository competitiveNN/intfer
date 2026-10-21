# shellcheck shell=bash
# Shared load-sampling + per-arm logging for the wall-time measurement harnesses.
#
# WHY THIS EXISTS (AUDIT #23, following AUDIT #22).
#   Two defects let a tok/s be published for arms whose own logs read
#   TPS=SKIPPED-CONTENDED beside LOAD1=4.67:
#     1. the quiet gate sampled load1 ONCE (at the start of a round) and then
#        measured for minutes -- and this box has been observed going from
#        load1 1.94 to 10.69 in under two minutes, so the pass certified
#        nothing about the measurement;
#     2. no arm wrote a log, so a figure could only be read off a terminal and
#        nothing on disk could contradict it.
#   Both are one-liners here so there is exactly ONE place to get them right.
#   "Nine hand-rolled copies of the block is precisely how the family reopened"
#   (AUDIT #7) -- three harnesses had already grown their own warn-and-continue
#   quiet gate, all three of which PRINT the right warning and then emit the
#   figure anyway.
#
# USAGE
#     . "$(dirname "$0")/_gate_loadlog.sh"
#     loadlog_init "$LOGDIR/run1"
#     loadlog_arm "r${r}_${arm}" env $e "$BIN" ...      # runs, logs, samples
#     tps="$(loadlog_tps_field "$scraped")"             # '' -> SKIPPED-CONTENDED
#     log="$_LL_LOG"; mx="$_LL_MAX"; usable="$_LL_USABLE"
#
# This helper owns NO build path. The CALLER chooses the directory and must
# honour INTFER_BUILD_DIR itself -- that is what check_build_dir_consumers.sh
# enforces. An earlier version of this comment showed `loadlog_init
# build/bench_logs/run1`, which read as though the helper defaulted to the build
# tree; the consumer classifier matched the comment (it does not strip comments)
# and failed a script that consumes no artifact at all. The example is a variable
# now because that is what the code actually does.
#
# SEAMS (both exist so a test can drive them; see _gate_precond.sh's INTFER_SYCL_LS
# for the same rationale -- an assumption the harness cannot redirect is an
# assumption the harness cannot test):
#   $LOADAVG               loadavg source        (default /proc/loadavg)
#   $INTFER_QUIET_LOAD     threshold             (default 2)
#   $INTFER_LOAD_SAMPLE    sampling period, sec  (default 1)
#   $LOADLOG_DIR           where logs go         (set by loadlog_init)

[ -n "${_GATE_LOADLOG:-}" ] && return 0
_GATE_LOADLOG=1

: "${INTFER_QUIET_LOAD:=2}"
: "${LOADAVG:=${INTFER_LOADAVG:-/proc/loadavg}}"
: "${INTFER_LOAD_SAMPLE:=1}"

loadlog_init() { # $1 = directory for per-arm logs and load traces
  LOADLOG_DIR="$1"
  mkdir -p "$LOADLOG_DIR" || { echo "loadlog: cannot create $LOADLOG_DIR" >&2; return 1; }
  echo "loadlog: per-arm logs in $LOADLOG_DIR (threshold load1 < $INTFER_QUIET_LOAD)"
}

loadlog_load1() { awk '{print $1}' "$LOADAVG" 2>/dev/null; }

# The maximum sampled load over the arm. This is the WHOLE POINT: a start-of-arm
# sample cannot certify a measurement that runs for minutes.
loadlog_max() { # $1 = trace file
  [ -s "$1" ] || return 1
  sort -g "$1" | tail -1
}

loadlog_arm() { # $1 = label; the rest is the command (env comes from the caller)
  local label="$1"; shift
  local log="$LOADLOG_DIR/$label.log" tr="$LOADLOG_DIR/$label.load"
  [ -d "$LOADLOG_DIR" ] || { echo "loadlog: call loadlog_init first" >&2; return 1; }
  : > "$tr"
  ( while :; do awk '{print $1}' "$LOADAVG" >> "$tr" 2>/dev/null; sleep "$INTFER_LOAD_SAMPLE"; done ) &
  local mon=$!
  "$@" > "$log" 2>&1
  local rc=$?
  kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null

  _LL_LOG="$log"; _LL_MAX="$(loadlog_max "$tr" 2>/dev/null || true)"; _LL_USABLE=1
  if [ -z "$_LL_MAX" ] || ! awk -v m="$_LL_MAX" -v q="$INTFER_QUIET_LOAD" \
        'BEGIN{exit !(m+0 < q+0)}'; then
    _LL_USABLE=0
  fi
  if [ "$_LL_USABLE" = 1 ]; then
    echo "loadlog: $label USABLE max_load1=$_LL_MAX log=$log"
  else
    echo "loadlog: $label CONTENDED max_load1=${_LL_MAX:-NA} >= $INTFER_QUIET_LOAD" >&2
    echo "loadlog:   NO FIGURE FROM THIS ARM IS USABLE (raw output kept: $log)" >&2
  fi
  return "$rc"
}

# Fail CLOSED and LOUDLY when a figure is consumed before loadlog_arm has run.
# Without this the empty verdict compared unequal to "1" and loadlog_field
# returned SKIPPED-CONTENDED -- the SAFE direction, but a silent one: a harness
# wired wrong would withhold every figure forever and never say why. With it, the
# wiring error is the message. This is the AUDIT #20 shape applied to a value: a
# guard that cannot fail is not a guard, and neither is an empty verdict.
_ll_need() { # $1 = variable name
  if [ -z "${!1-}" ]; then
    echo "loadlog: $1 is unset -- loadlog_arm must run (and succeed) before any" >&2
    echo "loadlog:   figure is emitted. Refusing to guess: withholding a figure" >&2
    echo "loadlog:   without saying why is how a harness goes quietly blank." >&2
    return 1
  fi
}

# The arm's verdict, or a loud failure. Harnesses may use this to gate a block of
# figures (bench_chat.sh does), and it is the form that is safe under `set -u`.
loadlog_usable() { _ll_need _LL_USABLE || return 1; [ "$_LL_USABLE" = 1 ]; }

# THE primitive: a value a caller is about to publish, or the withheld token.
# Every figure this repo quotes goes through here, whatever its unit -- a tok/s, a
# TTFT in seconds, an aggregate rate. One primitive means one definition of
# "withheld", so two harnesses cannot invent two spellings of it.
loadlog_field() { # $1 = the value (may be empty)
  _ll_need _LL_USABLE || return 1
  if [ "$_LL_USABLE" = 1 ]; then
    printf '%s' "$1"
  else
    printf 'SKIPPED-CONTENDED'
  fi
}

# The tok/s spelling, kept as a named alias because every caller reads better with
# it and the counters-only mode's field is literally called TPS.
loadlog_tps_field() { loadlog_field "$1"; }
