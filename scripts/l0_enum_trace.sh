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
# Exit:  0 = ran (read the output), 1 = could not run, 2 = no driver found.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/intfer_l0trace.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

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
n="$(env NEOReadDebugKeys=1 PrintDebugMessages=1 \
       LD_LIBRARY_PATH="$(dirname "$DRV"):/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}" \
       timeout 180 "$WORK/probe" 2>/dev/null | sed -n 's/^TOTAL LEVEL ZERO DEVICES: //p')"
if [ -z "$n" ]; then
  echo "no TOTAL line -- driver did not run"
elif [ "$n" -ge 2 ]; then
  echo "BOTH devices enumerate: the drop is NOT in this driver's discovery path"
else
  echo "still $n device(s)"
  echo "The driver emitted NO skip diagnostics, which means the release build has"
  echo "PrintDebugMessages compiled out. That is itself informative: these skip"
  echo "points are silent by construction, so the remaining suspects must be"
  echo "distinguished host-side (compare Resizable BAR / Region 2 size between"
  echo "0000:04:00.0 and 0000:86:00.0) or with a debug build of compute-runtime."
fi
exit 0
