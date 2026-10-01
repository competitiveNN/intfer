#!/usr/bin/env bash
# Enumerate Level Zero devices at the DRIVER API, bypassing SYCL and UR.
#
# WHY: `sycl-ls` answering "1 GPU" is ambiguous -- it cannot distinguish "the
# runtime filtered it", "UR failed to load the driver", and "the L0 driver itself
# does not return it". This calls zeInit -> zeDriverGet -> zeDeviceGet on
# libze_loader directly, so the answer is attributable to the driver.
#
# It also prints the driver UUID. Group 4 of an Intel L0 UUID encodes PCI
# bus:device:function, which is what identifies WHICH physical GPU survived
# enumeration when there is more than one card in the machine.
#
# Usage: scripts/l0_enumerate.sh
# Env:   INTFER_L0_DRIVER_LIB=DIR   prepend DIR to LD_LIBRARY_PATH, to test a
#                                   different libze_intel_gpu.so.1 (e.g. the host's
#                                   /run/host/usr/lib64). The driverVersion field
#                                   in the output makes the swap VERIFIABLE --
#                                   do not trust a result where it did not change.
# Exit:  0 if >=1 device found, 2 if the loader found no driver at all.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${INTFER_L0_OUT:-${TMPDIR:-/tmp}/intfer_l0_enumerate.$$}"
trap 'rm -f "$OUT" "$OUT.bin"' EXIT

# Locate the compiler the same way scripts/build.sh does (avoids sourcing
# setvars.sh, which calls exit and would kill this script).
CBIN="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/bin 2>/dev/null | sort -V | tail -1)"
CLIB="$(ls -dv /opt/intel/oneapi/compiler/[0-9]*/lib 2>/dev/null | sort -V | tail -1)"
[ -n "$CBIN" ] || { echo "SKIP: no oneAPI compiler found."; exit 2; }
INC=/usr/include/level_zero
[ -f "$INC/ze_api.h" ] || INC="$(dirname "$(find / -name ze_api.h 2>/dev/null | head -1)")"

# Build into a fresh path every time, and VERIFY the binary appeared. A stale
# binary that still prints results is the family-3 defect (stale artifact) this
# repo keeps auditing -- a probe that silently re-runs yesterday's build is
# worse than no probe.
rm -f "$OUT" "$OUT.bin"
PATH="$CBIN:$PATH" LD_LIBRARY_PATH="$CLIB:${LD_LIBRARY_PATH:-}" \
  icpx -x c -o "$OUT.bin" "$SCRIPT_DIR/l0_enumerate.c" -I"$INC" -lze_loader 2>&1 \
  | grep -vi deprecated | head -5
if [ ! -x "$OUT.bin" ]; then
  echo "FAIL: probe did not build; refusing to report (a stale binary would lie)."
  exit 1
fi

export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
if [ -n "${INTFER_L0_DRIVER_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$INTFER_L0_DRIVER_LIB:$LD_LIBRARY_PATH"
  echo "forced driver lib from: $INTFER_L0_DRIVER_LIB"
fi

timeout 120 "$OUT.bin"
rc=$?
echo
case $rc in
  0) echo "L0 ENUMERATE: >=1 device (see TOTAL above)" ;;
  2) echo "L0 ENUMERATE: loader found NO driver" ;;
  *) echo "L0 ENUMERATE: unexpected rc=$rc" ;;
esac
exit $rc
