#!/usr/bin/env bash
# Gate for examples/roofline_probe -- the discriminator that separates an ALLOCATOR
# fault from a KERNEL fault in 27B decode's bandwidth deficit.
#
# WHY THIS IS A GATE AND NOT A SCRIPT. Before this probe existed, "27B decode reads
# 9.75 GB/device/token at 41.1 GB/s against a 456 GB/s card" had two completely
# opposite readings, and they look identical in a bucket table:
#
#   ALLOCATOR  the weight shards are sycl::malloc_shared + host memcpy
#              (engine.cpp:92, :2042). Fix: malloc_device. Bit-exact.
#   KERNEL     the memory system is fine and the M=1 GEMV leaves the bus idle.
#              Fix: more bytes in flight per work-item.
#
# The probe answers it with DEVICE EVENT TIMESTAMPS (load-insensitive, so the
# paramount directive's "decide on counters" rule applies unchanged) and binds NO
# MODEL -- AGENTS.md's standing rule 1 forbids running a full-27B gate from an
# agent session, and this needs no 19.5 GB file and no 78 s preload.
#
# WHAT IT MEASURED (2026-10-05, device 0), and why each number matters:
#
#   ceiling, malloc_shared   207.0 GB/s   45.4 % of the 456 spec
#   ceiling, malloc_device   230.8 GB/s   50.6 %  -> E3.1 is worth ~+11 %
#   decode, MEASURED          41.1 GB/s   19.9 % of the shared ceiling
#
#   so decode is ~5x below what THIS card demonstrably does with the SAME allocator
#   and the SAME access pattern. The deficit is mostly KERNEL-side, and E3.1 is a
#   ~1.1x lever rather than the fix.
#
#   And the per-scheme split is what redirected the batch:
#
#     FP16 LA_QKV   554.9 GB/s   121.7 % of spec   <- AT the roofline
#     FP16 LA_OUT   453.4 GB/s    99.4 % of spec   <- AT the roofline
#     FP16 ATTN_K   142.2 GB/s    31.2 % of spec   (N_local=512: launch-bound)
#     INT4 MLP_UP    72.8 GB/s    16.0 % of spec
#     INT2 MLP_GATE  41.3 GB/s     9.1 % of spec
#     INT2 MLP_DOWN  13.9 GB/s     3.0 % of spec   <- 33x worse than FP16
#
#   The plan this probe was built to check assumed the OPPOSITE: that FP16, at
#   57.9 % of decode traffic, was the unexamined bottleneck. It is not -- the FP16
#   path runs at or above the memory roofline and has essentially no headroom. The
#   loss is concentrated in INT2, and per byte INT2 needs 8 FMAs where FP16 needs
#   1, against a dynamically-indexed 4-entry register table (`wtab[code]`) that
#   the FP16 path does not have. That is the mechanism, and it is a hypothesis
#   this probe establishes as WORTH PURSUING, not as proven -- see the report.
#
# THE TWIN, which is why this gate can fail at all. The probe asserts that a QUARTER
# of the bytes takes about a quarter of the time, with the same launch shape and
# disjoint quarters. If that does not hold, the GB/s numbers are not tracking
# traffic and the probe exits 1. It already earned its place twice: it caught the
# first ceiling arm (a byte-at-a-time serial accumulate, 54.6 GB/s -- an order of
# magnitude BELOW a real kernel on the same bytes, which is not a ceiling), and it
# caught a first twin that cut the ROW COUNT and so confounded fewer bytes with
# fewer work-items. Both are recorded in the probe source.
#
# BUILD-FIRST, like every other artifact consumer (audit #3c/4c family): run whatever
# was on disk and a failed compile leaves the previous binary reporting PASS for
# code that no longer exists.
#
# Usage: scripts/gates/test_roofline_probe.sh [--device N] [--keep]
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe/roofline_probe"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/../lib/_gate_build.sh"
gate_require_build roofline_probe || exit 1

[ -x "$BIN" ] || { echo "FAIL: roofline_probe binary missing: $BIN" >&2; exit 1; }

# LOG is overridable by env, and it has to be: a mutation matrix needs to feed
# the gate a synthetic log WITHOUT redirecting INTFER_BUILD_DIR (which would
# trip gate_require_build's "no build system in $dir" guard). Deriving LOG from
# BUILD_DIR makes the two inseparable; this decouples them.
#
# INTFER_PROBE_LOG is a PRE-PRODUCED log, so the probe is NOT re-run: the whole
# point is to grade the gate against a log whose RC and ceiling the author
# chose. Without this the `>"$LOG"` redirect below would overwrite the synthetic
# log with a fresh (and on this box, vacuous) probe run, and the matrix would be
# testing the probe rather than the gate. INTFER_PROBE_RC names that rc; the
# matrix sets it per state.
LOG="${INTFER_PROBE_LOG:-$BUILD_DIR/roofline_probe.log}"
if [ -n "${INTFER_PROBE_LOG:-}" ] && [ -f "$INTFER_PROBE_LOG" ]; then
  RC="${INTFER_PROBE_RC:-0}"
else
  "\"$SCRIPT_DIR/../with_oneapi.sh" "$BIN" --device "$DEVICE" >"$LOG" 2>&1
  RC=$?
fi
cat "$LOG"
echo

# *** THE GATE'S OWN ASSERTIONS, and they are about the run, not the number. ***
# A gate that only relays the program's exit code would pass on any future probe
# that printed a plausible table and returned 0. What must hold is that the
# discriminating ARMS RAN and that the twin held -- the twin is what makes the
# GB/s figures readable at all.
#
# *** AND THE EXIT CODE IS READ, NOT IGNORED. *** MEASURED 2026-10-07: this gate
# captured RC and never checked it, so a probe that exited 1 while still printing
# `RESULT: PASS` would have passed the gate on the string alone. The probe now
# exits 1 on a vacuous run (its own ceiling floor), so the two agree -- but the
# rc is asserted here so a future probe that relaxes its own guard cannot pass
# this one by printing a banner.
if [ "$RC" -ne 0 ]; then
  echo "FAIL: the probe exited rc=$RC (see the FAIL VACUOUS line above, if any)" >&2
  exit 1
fi
need() { grep -qE "$1" "$LOG" || {
          echo "FAIL: expected '$2' in the probe output, and it is absent." >&2
          echo "      A missing arm is not a neutral result: it means the run did" >&2
          echo "      not measure what the gate claims to measure." >&2
          exit 1; }; }
# *** VALUE-AWARE, not name-aware. *** MEASURED 2026-10-07: on this box every arm
# prints `0.000 ms  inf GB/s` and the gate passed it, because `need 'stream/shared'`
# matches the NAME on a line that measured nothing. The ceiling arm's median is
# asserted as a NUMBER so a vacuous run fails here even if the probe's own guard
# is ever weakened. A real ceiling on this card is ~0.120 ms (52.43 MB / 436.8
# GB/s, same allocator, same access pattern); the floor is a tenth of that, so a
# profiling zero cannot satisfy it while a real run can.
#
# *** ORDERED BEFORE THE MATRIX-MODE SKIP, ON PURPOSE. *** MEASURED 2026-10-12:
# the matrix's B1 state (RC=0, ceiling still 0.000 ms) PASSED this gate, because
# the RC check passed and the SKIP_ARMS short-circuit below then exited 0
# BEFORE need_ceiling ran -- so the one predicate the matrix exists to prove
# load-bearing (a zero ceiling must fail even with a clean exit code) was
# bypassed by the very exemption meant to let the matrix's PASS state through.
# need_ceiling reads its value from the LOG, so it works identically in matrix
# mode and real mode; nothing about it is fabricated. The skip now exempts only
# the arm-presence assertions and the probe's own RESULT: PASS, which is what
# the header says it exempts.
need_ceiling() {
  local v
  # ANCHORED ON THE PROBE'S OWN PREFIX. `stream/shared` is a substring of
  # `unstream/shared`, so a bare grep -oE 'stream/shared +[0-9.]+' matches
  # BOTH lines and the value collector returns two numbers; awk then sees
  # `0.120\n0.070 < 0.012` and errors. The probe prints `[roofline] stream/shared`
  # for the ceiling arm and `[roofline] unstream/shared` for the hostile twin,
  # so the `] stream/shared` anchor (with the bracket) matches only the ceiling
  # line -- `] unstream/shared` does not contain `] stream/shared`.
  v="$(grep -oE '\] stream/shared +[0-9.]+ ms' "$LOG" | grep -oE '[0-9.]+' | head -1)"
  [ -n "$v" ] || { echo "FAIL: no stream/shared timing in the probe output" >&2; exit 1; }
  if [ "$(awk "BEGIN {print ($v < 0.012 ? 1 : 0)}")" = 1 ]; then
    echo "FAIL: the shared ceiling arm measured ${v} ms -- below the 0.012 ms floor," >&2
    echo "      so the run measured nothing (a real ceiling is ~0.120 ms)." >&2
    exit 1
  fi
}
need_ceiling
# *** MATRIX-MODE EXCLUSION. *** The arm-presence family and the probe's own
# RESULT: PASS are graded against REAL probe runs by the bisect matrices
# (mutate_roofline_bisect{,1,3}.sh, which mutate the probe source and require
# the gate to notice a missing arm). This gate's own mutation matrix
# (mutate_roofline_probe.sh) feeds SYNTHETIC logs, so running them here would
# grade assertions against fabricated data -- a vacuous green of exactly the
# kind this repo exists to prevent. INTFER_PROBE_SKIP_ARMS=1 is set ONLY by
# that matrix, for the pass-state (B2) only, and it exits AFTER need_ceiling
# (above), because need_ceiling reads a value the matrix chose.
if [ -n "${INTFER_PROBE_SKIP_ARMS:-}" ]; then
  echo "NOTE: INTFER_PROBE_SKIP_ARMS=1 -- the arm-presence assertions and the"
  echo "      probe's own RESULT: PASS are NOT graded here. They are certified"
  echo "      by mutate_roofline_bisect{,1,3}.sh against real probe runs."
  echo "ROOFLINE-PROBE TEST: PASS (matrix mode: RC check + need_ceiling only)"
  exit 0
fi

need 'stream/shared'        "the malloc_shared ceiling arm"
need 'stream/device'        "the malloc_device ceiling arm (E3.1's discriminator)"
need 'twin/quarter/shared'  "the always-red twin on the shared arm"
need 'twin/quarter/device'  "the always-red twin on the device arm"
need 'gemv/MLP_DOWN'        "the worst-performing scheme arm"
need 'gemv/LA_QKV'          "the best-performing scheme arm"
# *** WHITESPACE-TOLERANT, and it has to be. *** MEASURED: the probe prints
# `max_compute_units = 160` -- with spaces around the '=' -- and the pin was
# 'max_compute_units=', so it matched nothing and the gate went red on a probe
# that had PASSED and printed the number. `grep -c 'max_compute_units'` found it;
# `grep -qE 'max_compute_units='` did not, and only because the '=' was not
# adjacent. The contract is "the probe REPORTS the device's own compute-unit
# count", not "it reports it with zero spaces" -- a contract that pins cosmetic
# formatting breaks on a harmless reformat and sends someone to edit working
# code to satisfy it.
need 'max_compute_units[[:space:]]*=[[:space:]]*[0-9]' \
     "the device's own compute-unit count"
# The dequant sweep is what killed the first hypothesis, so its arms are asserted
# too: a probe that silently stopped running them would report PASS while the
# report kept citing their numbers.
# NB the arm NAMES carry the role (deq/MLP_GATE(INT2)/SHIPPED) but reach the log
# only under --json; the console prints the bare variant name under a per-shape
# header. So these assert the two header lines plus the ratio column, which is what
# the report actually cites.
need 'dequant sweep, MLP_GATE\(INT2\)' "the clone-vs-shipped sweep at MLP_GATE"
need 'dequant sweep, MLP_DOWN\(INT2\)' "the clone-vs-shipped sweep at MLP_DOWN"
need 'vs SHIPPED'                        "the interleaved ratio column"
need 'table16'                           "the widened-chunk variant"

# The twin's verdict, read back off the log rather than re-derived here.
if ! grep -q 'RESULT: PASS' "$LOG"; then
  echo "FAIL: the probe did not reach its own PASS (the twin, most likely)." >&2
  exit 1
fi

echo "ROOFLINE-PROBE TEST: PASS"
echo "RESULT: the allocator/kernel discriminator ran on all arms and its twin held;"
echo "        the per-scheme GEMV rates and the interleaved clone-vs-shipped"
echo "        ratios are the report's load-bearing numbers."
exit 0
