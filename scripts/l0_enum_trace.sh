#!/usr/bin/env bash
# Build + run libze_intel_gpu's OWN symbols to find WHICH card it drops and why.
#
# WHY THIS EXISTS. Reading compute-runtime source gives the FIVE silent skip
# points in the per-card loop (see AGENTS.md). A "silent" skip is precisely the
# kind that is unobservable from outside -- so instead of guessing which one
# fired, we make the driver TELL US, by running its real discovery code and
# turning on its own debug output.
#
#   * NEOReadDebugKeys=1        -> the driver re-reads sysfs for debug keys
#   * PrintDebugMessages=1      -> prints the skip reasons to stderr, e.g.
#                                  "FATAL: Cannot query device ID and revision"
#                                  "FATAL: Unknown device"
#                                  "WARNING: Resizable BAR not detected"
#
# The loader itself does no discovery (it only dlopens libze_intel_gpu.so.1 and
# returns a static device list), so this is the only level where the per-card
# decisions are observable.
#
# This does NOT modify the tracked tree and does NOT need root. If the release
# driver was built with debug messages compiled out, this reports that honestly
# rather than pretending to have found anything.
#
# Usage: scripts/l0_enum_trace.sh
# Env:   INTFER_L0_DRIVER_LIB=DIR   test a different libze_intel_gpu.so.1
# Exit: 0 = BOTH devices, or a skip diagnostic was printed (information obtained)
#        1 = could not run, 2 = no driver found / did not run
#        3 = RAN but printed no skip reason => INCONCLUSIVE (not success)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/intfer_l0trace.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT INT TERM

# ---- locate the driver we are going to interrogate -------------------------
DRV_DIR="${INTFER_L0_DRIVER_LIB:-/usr/lib64}"
DRV="$(ls "$DRV_DIR"/libze_intel_gpu.so.1 2>/dev/null | head -1)"
if [ -z "$DRV" ]; then
  DRV="$(ldconfig -p 2>/dev/null | awk '/libze_intel_gpu\.so\.1/ {print $NF; exit}')"
fi
[ -n "$DRV" ] && [ -r "$DRV" ] || { echo "SKIP: libze_intel_gpu.so.1 not found"; exit 2; }
echo "driver under test: $DRV"

# ---- build the raw L0 probe (same source as scripts/l0_enumerate.c) --------
CBIN="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/bin 2>/dev/null | sort -V | tail -1)"
CLIB="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/lib 2>/dev/null | sort -V | tail -1)"
[ -n "$CBIN" ] || { echo "SKIP: no oneAPI compiler"; exit 1; }
INC=/usr/include/level_zero
[ -f "$INC/ze_api.h" ] || INC="$(dirname "$(find / -name ze_api.h 2>/dev/null | head -1)")"
rm -f "$WORK/probe"
PATH="$CBIN:$PATH" LD_LIBRARY_PATH="$CLIB:${LD_LIBRARY_PATH:-}" \
  icpx -x c -o "$WORK/probe" "$SCRIPT_DIR/l0_enumerate.c" -I"$INC" -lze_loader 2>&1 \
  | grep -vi deprecated | head -5
[ -x "$WORK/probe" ] || { echo "FAIL: probe did not build"; exit 1; }

run() { # $1 = label
  echo
  echo "--- $1 ---"
  env NEOReadDebugKeys=1 PrintDebugMessages=1 \
      LD_LIBRARY_PATH="$(dirname "$DRV"):/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}" \
      timeout 180 "$WORK/probe" 2>&1
}

run "with NEOReadDebugKeys=1 PrintDebugMessages=1"

echo
echo "=== interpretation ==="
# stderr is CAPTURED, not discarded: the skip reasons go to stderr, so
# 2>/dev/null would throw away the only thing this script exists to see.
trace="$(env NEOReadDebugKeys=1 PrintDebugMessages=1 \
       LD_LIBRARY_PATH="$(dirname "$DRV"):/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}" \
       timeout 180 "$WORK/probe" 2>&1)"
n="$(printf '%s\n' "$trace" | sed -n 's/^TOTAL LEVEL ZERO DEVICES: //p')"
# The skip reasons we are hunting for, per compute-runtime's own strings.
diag="$(printf '%s\n' "$trace" | grep -iE 'FATAL|Resizable BAR|Unknown device|Cannot query|Caps reader|Topology|EUCount|Query mem' || true)"

if [ -z "$n" ]; then
  echo "NO INFORMATION: driver produced no TOTAL line -- it did not run."
  echo "RESULT: l0_enum_trace INCONCLUSIVE (driver did not run)"
  exit 2
fi

if [ -n "$diag" ]; then
  echo "SKIP DIAGNOSTIC(S) FOUND -- the driver named the failure:"
  printf '%s\n' "$diag" | sed 's/^/  /'
fi

if [ "$n" -ge 2 ]; then
  echo "BOTH devices enumerate ($n) -- the drop is NOT in this driver's discovery path."
  echo "RESULT: l0_enum_trace BOTH_DEVICES ($n)"
  exit 0
fi

# Fewer than 2 devices. Distinguish "we learned WHICH skip fired" from
# "we learned nothing", because those demand different next steps and the
# difference is invisible if both exit 0.
if [ -n "$diag" ]; then
  echo "still $n device(s), AND a skip diagnostic was printed above -- act on it."
  echo "RESULT: l0_enum_trace DIAGNOSTIC ($n device, reason printed)"
  exit 0
fi

echo "NO INFORMATION: still $n device(s), and the driver printed NO skip diagnostic."
echo "  PrintDebugMessages is evidently compiled out of this release build, so"
echo "  skips (c)/(d) are silent BY CONSTRUCTION. This run therefore says"
echo "  NOTHING about which skip fired. The remaining suspects (small BAR on"
echo "  card1; BMG deferred-engine-init) must be distinguished HOST-side by"
echo "  comparing Resizable BAR / Region 2 size for 0000:04:00.0 vs 0000:86:00.0,"
echo "  or with a DEBUG build of compute-runtime."
echo "  Do not read 'no output' as 'no problem found'."
echo "RESULT: l0_enum_trace INCONCLUSIVE ($n device, no skip reason printed)"
exit 3
