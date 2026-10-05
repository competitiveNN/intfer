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
# Usage: scripts/test_roofline_probe.sh [--device N] [--keep]
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/roofline_probe/roofline_probe"
DEVICE=0
[ "${1:-}" = "--device" ] && DEVICE="${2:-0}"

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build roofline_probe || exit 1

[ -x "$BIN" ] || { echo "FAIL: roofline_probe binary missing: $BIN" >&2; exit 1; }

LOG="$BUILD_DIR/roofline_probe.log"
"$SCRIPT_DIR/with_oneapi.sh" "$BIN" --device "$DEVICE" >"$LOG" 2>&1
RC=$?
cat "$LOG"
echo

# *** THE GATE'S OWN ASSERTIONS, and they are about the run, not the number. ***
# A gate that only relays the program's exit code would pass on any future probe
# that printed a plausible table and returned 0. What must hold is that the
# discriminating ARMS RAN and that the twin held -- the twin is what makes the
# GB/s figures readable at all.
need() { grep -qE "$1" "$LOG" || {
          echo "FAIL: expected '$2' in the probe output, and it is absent." >&2
          echo "      A missing arm is not a neutral result: it means the run did" >&2
          echo "      not measure what the gate claims to measure." >&2
          exit 1; }; }

need 'stream/shared'        "the malloc_shared ceiling arm"
need 'stream/device'        "the malloc_device ceiling arm (E3.1's discriminator)"
need 'twin/quarter/shared'  "the always-red twin on the shared arm"
need 'twin/quarter/device'  "the always-red twin on the device arm"
need 'gemv/MLP_DOWN'        "the worst-performing scheme arm"
need 'gemv/LA_QKV'          "the best-performing scheme arm"
need 'max_compute_units='   "the device's own compute-unit count"
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
