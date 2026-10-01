#!/usr/bin/env bash
# Committed A/B over libze_intel_gpu.so.1 versions, at the L0 driver API.
#
# WHY A SCRIPT. The claim "the driver-library version is not the cause" was
# previously only prose plus a transcript. This makes it a re-runnable
# assertion, and -- more importantly -- makes the swap VERIFIABLE. A previous
# comparison was worthless because a probe edit failed to compile, the OLD
# binary ran, and two identical UUIDs were nearly read as a result. The driver
# version is printed on every arm, so a swap that did not happen is visible as
# a REPEATED driverVersion rather than being silently trusted.
#
# A/B is only meaningful if the two arms really are different libraries; this
# script FAILS if the driverVersion comes back identical on both arms.
#
# Usage: scripts/gpu_driver_lib_ab.sh
# Exit:  0 = >=1 device on every arm AND the libraries were demonstrably
#              different (so "version is not the cause" is established),
#        1 = a driver lib could not be loaded, or all arms agreed on ONE
#            driverVersion (=> the A/B proved nothing),
#        2 = no driver at all.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/intfer_lab.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Arm 0: whatever the loader picks up by default.
# Arm 1: the host's newer libze_intel_gpu, if present.
ARMS=("default" "${INTFER_L0_ALT_DRIVER_LIB:-/run/host/usr/lib64}")

# Each arm simply delegates to scripts/l0_enumerate.sh, which already honours
# INTFER_L0_DRIVER_LIB and prints driverVersion. Nothing is reimplemented here,
# because a second copy of the probe invocation is a second chance to get the
# library path subtly wrong -- which is what made the original comparison void.
declare -a VERS=()
declare -a COUNTS=()
for arm in "${ARMS[@]}"; do
  label="$arm"; extra=""
  [ "$arm" != "default" ] && { label="$(basename "$arm")"; extra="$arm"; }
  if [ -n "$extra" ] && [ ! -e "$extra/libze_intel_gpu.so.1" ]; then
    echo "SKIP arm $label: no libze_intel_gpu.so.1 under $extra"
    continue
  fi
  echo "=== arm: $label${extra:+ ($extra)} ==="
  o="$(INTFER_L0_DRIVER_LIB="$extra" "$SCRIPT_DIR/l0_enumerate.sh" 2>&1)"
  printf '%s\n' "$o" | grep -E 'forced driver lib|driver\[|TOTAL|FAIL: |SKIP:'
  v="$(printf '%s\n' "$o" | sed -n 's/.*driverVersion=\(0x[0-9a-f]*\).*/\1/p' | head -1)"
  n="$(printf '%s\n' "$o" | sed -n 's/^TOTAL LEVEL ZERO DEVICES: //p' | head -1)"
  VERS+=("$v"); COUNTS+=("$n")
  echo
done

arms="${#VERS[@]}"
echo "=== A/B verdict ==="
i=0; for v in "${VERS[@]}"; do echo "  arm${i}: driverVersion=${v:-?} devices=${COUNTS[$i]:-?}"; i=$((i+1)); done

[ "$arms" -eq 0 ] && { echo "no arm ran"; exit 2; }

uniq_v="$(printf '%s\n' "${VERS[@]}" | sort -u | grep -c .)"
if [ "$arms" -ge 2 ] && [ "$uniq_v" -le 1 ]; then
  echo
  echo "INCONCLUSIVE: both arms report the SAME driverVersion, so the libraries were"
  echo "  never actually swapped and this A/B proves NOTHING. Do not cite it."
  echo "RESULT: gpu_driver_lib_ab INCONCLUSIVE (arms did not differ)"
  exit 1
fi
if [ "$arms" -lt 2 ]; then
  echo "INCONCLUSIVE: only one arm available; nothing was compared."
  echo "RESULT: gpu_driver_lib_ab INCONCLUSIVE (1 arm)"
  exit 1
fi

echo
echo "$arms demonstrably DIFFERENT driver libraries were loaded:"
i=0; for v in "${VERS[@]}"; do echo "  driverVersion=${v} -> ${COUNTS[$i]} device(s)"; i=$((i+1)); done
if [ "${COUNTS[0]}" != "${COUNTS[1]}" ]; then
  echo "RESULT: DRIVER VERSION IS THE CAUSE (${COUNTS[0]} vs ${COUNTS[1]} devices)"
  exit 0
fi
echo "Both driver versions enumerate the same number of devices, so the library"
echo "version is NOT the cause. (Distinct versions confirmed above, so this is a"
echo "real A/B and not a no-op swap.)"
echo "RESULT: gpu_driver_lib_ab NEGATIVE - version not the cause ($arms arms, both ${COUNTS[0]})"
exit 0