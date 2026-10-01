#!/usr/bin/env bash
# Diagnose and recover the missing Arc B580 in Level Zero enumeration.
#
# Run this ON THE HOST (or in a --privileged container) as root. It is the
# script that would have closed the investigation; two of its steps need
# privileges the dev container does not have:
#     * reading PCI config space beyond the legacy 64-byte header
#       (the container is limited to 4096 bytes and reads only 64 of them,
#        and ReBAR is a PCIe extended capability at offset 0x100+), and
#     * writing to /sys/bus/pci/... to rebind a device (EACCES here).
#
# USAGE
#   scripts/gpu_recover.sh diag      # read-only. Safe. Run this FIRST.
#   scripts/gpu_recover.sh rebind    # unbind+rebind the missing card. Root.
#   scripts/gpu_recover.sh verify    # re-run the driver-API enumeration.
#
# WHAT IS ALREADY RULED OUT (do not re-try these; see AGENTS.md AUDIT #9/#11/#20)
#   * driver-library version -- 1.15.38646 and 1.17.39758 both return 1 device
#   * cgroup device policy, NUMA, power state, another process holding GPU 1
#   * ONEAPI_DEVICE_SELECTOR / SYCL_DEVICE_FILTER -- every form still yields 1
#   * the `tile0/gt*/freq0/{min,max}_freq` ECANCELED reads -- refuted: the driver
#     reads those files and DISCARDS the return value, so they cannot gate
#     enumeration
#
# THE LEADING CAUSE, and the one decisive test for it. compute-runtime's
# discovery (drm_neo.cpp) drops a card when its VRAM BAR is smaller than its
# VRAM:
#     } else if (getMemoryInfo()->isSmallBarDetected()) {
#         if (!ioctlHelper->isSmallBarConfigAllowed()) { return -1; }   // drop
# and IoctlHelperXe::isSmallBarConfigAllowed() returns FALSE (i915's returns
# true), so this bites on xe only. Upstream compute-runtime issue #905: "with
# only 256MB BAR, the compute-runtime silently rejected the DG2 device when
# using the xe driver (while i915 worked fine with small BAR)". Our two cards
# are on DIFFERENT PCI roots (04:00.0 on pci0000:00, 86:00.0 on pci0000:80),
# which is exactly where Resizable BAR support commonly differs.
# Citations are machine-checked by scripts/check_l0_cite.sh.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MISSING_SLOT="${INTFER_MISSING_SLOT:-0000:86:00.0}"
PRESENT_SLOT="${INTFER_PRESENT_SLOT:-0000:04:00.0}"

is_root() { [ "$(id -u)" = "0" ]; }
need_root() { is_root || { echo "ERROR: '$1' needs root (writes to /sys/bus/pci)."; exit 3; }; }

# ---------------------------------------------------------------- diag
cmd_diag() {
  echo "=== GPU RECOVER: DIAG (read-only) ==="

  echo
  echo "-- 1. kernel: is the card still bound to a driver? --"
  for s in "$PRESENT_SLOT" "$MISSING_SLOT"; do
    p="/sys/bus/pci/devices/$s"
    [ -d "$p" ] || { echo "  $s  ABSENT from /sys/bus/pci/devices"; continue; }
    printf '  %s  driver=%s  runtime=%s  reset=%s\n' "$s" \
      "$(basename "$(readlink -f "$p/driver" 2>/dev/null)" 2>/dev/null)" \
      "$(cat "$p/power/runtime_status" 2>/dev/null)" \
      "$([ -e "$p/reset" ] && echo yes || echo no)"
  done

  echo
  echo "-- 2. render nodes (both must exist AND open) --"
  for n in /dev/dri/renderD128 /dev/dri/renderD129; do
    if [ -e "$n" ]; then
      if python3 -c "import os,sys;fd=os.open('$n',os.O_RDWR);os.close(fd)" 2>/dev/null; then
        echo "  $n  present, opens O_RDWR"
      else
        echo "  $n  present, BUT open FAILED (permission?)"
      fi
    else
      echo "  $n  MISSING"
    fi
  done

  echo
  echo "-- 3. THE DECISIVE TEST: VRAM BAR size vs VRAM size per slot --"
  echo "     (needs lspci, or root to read config space past the 64-byte header)"
  if command -v lspci >/dev/null 2>&1; then
    for s in "$PRESENT_SLOT" "$MISSING_SLOT"; do
      bus="${s#0000:}"; bus="${bus%%:*}"; dev="${bus#*:}"; dev="${dev%%.*}"
      echo "  --- $s ---"
      lspci -s "$bus:$dev.0" 2>/dev/null | sed 's/^/    /'
      # Resizable BAR capability and the Region 2 size
      lspci -vv -s "$bus:$dev.0" 2>/dev/null \
        | grep -iE 'Region 2|Resizable BAR|BAR2|physical' | sed 's/^/    /'
    done
    echo
    echo "  INTERPRET: if 86:00.0's Region 2 is SMALLER than 04:00.0's (or the"
    echo "  Resizable BAR line is missing on one), the small-BAR drop is the"
    echo "  cause. NO in-software fix exists for that -- it is a BIOS/slot"
    echo "  setting (enable 'Above 4G Decoding' and 'Resizable BAR' / 'Re-Size BAR')."
    echo "  If BARs are IDENTICAL, the small-BAR theory is refuted: go to"
    echo "  'rebind' and read the kernel log."
  else
    echo "  lspci not installed. Install it (dnf install pciutils) or read the"
    echo "  config space as root with setpci. Without this the BAR sizes -- the"
    echo "  single decisive measurement -- cannot be compared."
  fi

  echo
  echo "-- 4. what the compute runtime currently sees --"
  if [ -r /opt/intel/oneapi/setvars.sh ]; then
    set +u; . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
  fi
  export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
  "$ROOT/scripts/l0_enumerate.sh" 2>&1 | grep -E 'driver\[|TOTAL' | sed 's/^/  /'
  echo "  Group 4 of the UUID encodes PCI bus:dev:function. One device with"
  echo "  group 4 == 0400 means card0 survives and 86:00.0 is absent."

  echo
  echo "-- 5. kernel log (the one thing the container could NOT read) --"
  if dmesg >/dev/null 2>&1; then
    dmesg 2>/dev/null | grep -iE 'xe |86:00|04:00|resizable|rebar|guc|reset' | tail -25 | sed 's/^/  /'
  elif journalctl -k >/dev/null 2>&1; then
    journalctl -k --no-pager 2>/dev/null | grep -iE 'xe |86:00|resizable|rebar|guc' | tail -25 | sed 's/^/  /'
  else
    echo "  (no access to the kernel log here -- needs root)"
  fi
  echo
  echo "Next: if the BARs are equal, run  scripts/gpu_recover.sh rebind"
}

# ---------------------------------------------------------------- rebind
cmd_rebind() {
  need_root rebind
  echo "=== GPU RECOVER: REBIND $MISSING_SLOT ==="
  p="/sys/bus/pci/devices/$MISSING_SLOT"
  [ -d "$p" ] || { echo "ERROR: $MISSING_SLOT not present."; exit 1; }

  drv="$(basename "$(readlink -f "$p/driver" 2>/dev/null)" 2>/dev/null)"
  echo "-- current driver: ${drv:-<none>}"
  if [ -z "$drv" ]; then
    echo "   the device is NOT bound to any driver; bind it directly"
    echo "0000:${MISSING_SLOT#0000:}" > "/sys/bus/pci/drivers/$DRV/bind" 2>/dev/null \
      || echo "   (bind failed; is the driver module loaded?)"
  else
    echo "-- stopping any client on the card first"
    xpu-smi 2>/dev/null | sed -n '/Processes/,$p' | sed 's/^/   /'
    echo "   (if a process is listed, stop it: a busy device will not unbind cleanly)"

    echo "-- recording the BAR state BEFORE (for comparison after)"
    before="$(cat "$p/resource" 2>/dev/null | tr '\n' ' ')"
    echo "   resource: $before"

    echo "-- unbinding $MISSING_SLOT from $drv"
    if ! echo "$MISSING_SLOT" > "/sys/bus/pci/drivers/$drv/unbind"; then
      echo "   unbind FAILED. If the device is in use, that is the cause:"
      echo "   find the holder with  fuser -v /dev/dri/renderD129"
      exit 1
    fi
    sleep 2
    echo "-- rebinding"
    if ! echo "$MISSING_SLOT" > "/sys/bus/pci/drivers/$drv/bind"; then
      echo "   bind FAILED; trying a PCI rescan instead"
      echo 1 > /sys/bus/pci/rescan 2>/dev/null
      sleep 2
    fi
    after="$(cat "/sys/bus/pci/devices/$MISSING_SLOT/resource" 2>/dev/null | tr '\n' ' ')"
    echo "-- BAR state AFTER : $after"
    [ "$before" = "$after" ] && echo "   (unchanged -- note this in any bug report)"
  fi
  sleep 3
  echo
  cmd_verify
  echo
  echo "If the device still does not appear, the BAR comparison in 'diag'"
  echo "matters more than this rebind: a card whose VRAM BAR is smaller than"
  echo "its VRAM is dropped by design on xe and no rebind will change that."
}

# ---------------------------------------------------------------- verify
cmd_verify() {
  echo "=== GPU RECOVER: VERIFY (driver API) ==="
  [ -r /opt/intel/oneapi/setvars.sh ] && { set +u; . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u; }
  export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
  out="$("$ROOT/scripts/l0_enumerate.sh" 2>&1)"
  printf '%s\n' "$out" | grep -E 'driver\[|uuid|TOTAL' | sed 's/^/  /'
  n="$(printf '%s\n' "$out" | sed -n 's/^TOTAL LEVEL ZERO DEVICES: //p')"
  echo
  if [ "${n:-0}" -ge 2 ] 2>/dev/null; then
    echo "RESULT: 2 devices. The enumeration fault is GONE."
    echo "  Follow the next-session trigger in AGENTS.md:"
    echo "    1. sycl-ls | grep -c '\\[level_zero:gpu\\]'  -> must be 2"
    echo "    2. re-run scripts/run_all_checks.sh end-to-end (stamps must name HEAD)"
    echo "    3. re-run scripts/test_tp2_27b.sh  (the only gate whose sole arm"
    echo "       is --tp 2 on 27B)"
    echo "    4. then fill the RESULT: lines in reports/ and re-decide E1/E2/E3"
    exit 0
  fi
  echo "RESULT: still ${n:-?} device(s). The rebind did not restore enumeration."
  echo "  Read the BAR comparison from 'diag' -- if Region 2 on $MISSING_SLOT is"
  echo "  smaller than on $PRESENT_SLOT, this is a BIOS/slot setting, not a"
  echo "  software bug: enable Above 4G Decoding + Resizable BAR, or move the"
  echo "  card to a slot whose root complex reports full-size BAR."
  exit 1
}

case "${1:-diag}" in
  diag)   cmd_diag ;;
  rebind) cmd_rebind ;;
  verify) cmd_verify ;;
  *) echo "usage: $0 {diag|rebind|verify}"; echo "  INTFER_MISSING_SLOT / INTFER_PRESENT_SLOT override the slots"; exit 1 ;;
esac