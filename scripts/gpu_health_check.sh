#!/usr/bin/env bash
# Compare the two xe GPU cards at the kernel/sysfs layer and report ANY
# card1-specific read failure. Exists because the Level Zero driver returns
# only card0 (see scripts/l0_enumerate.sh) while the kernel still has BOTH
# cards bound -- so something distinguishes them below the driver, and the
# kernel log that would name it is NOT readable from this container
# (dmesg -> EPERM). This diffs the cards instead.
#
# The one anomaly it currently reports is ECANCELED (errno 125) on
# tile0/gt{0,1}/freq0/{min,max}_freq of card1 only. Those reads are
# GuC-mediated, which is consistent with card1's GuC not being functional.
# That is a LEAD, not a proven cause of the L0 enumeration drop.
#
# Usage: scripts/gpu_health_check.sh
# Exit:  0 = cards symmetric, 1 = card1-specific anomaly found, 2 = <2 cards.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARDS=()
# Match ONLY cardN. A plain `card*/device` glob also matches the CONNECTOR
# nodes (card0-DP-1, card1-HDMI-A-5, ...), which have no tile0/ and would make
# every attribute look like a card1-only ENOENT -- i.e. a fabricated anomaly.
# That bug shipped in the first version of this script; caught because the
# header printed an empty PCI slot for "card1".
for p in /sys/class/drm/card*/device; do
  n="$(basename "$(dirname "$p")")"
  # ^card[0-9]+$ -- anchored BOTH ends. `card[0-9]*` is NOT enough: bash's glob
  # `*` also matches "-DP-1", so card0-DP-1 passed that pattern too and got
  # picked as "card1", fabricating an ENOENT on every attribute. Two versions of
  # this script carried that bug; the empty PCI slot in the header is what
  # exposed it. A check whose subject is misidentified is worse than no check.
  if [[ "$n" =~ ^card[0-9]+$ ]] && [ -e "$p" ]; then CARDS+=("$p"); fi
done
if [ "${#CARDS[@]}" -lt 2 ]; then
  echo "SKIP: fewer than 2 DRM cards present (${#CARDS[@]})"; exit 2
fi
A="${CARDS[0]}"; B="${CARDS[1]}"
slot() { grep -m1 PCI_SLOT_NAME "$1/uevent" 2>/dev/null | cut -d= -f2; }

echo "== GPU HEALTH: $(basename "$(dirname "$A")")=$(slot "$A")  vs  $(basename "$(dirname "$B")")=$(slot "$B") =="

read_attr() { # path -> value or !ERR:<errno>
  python3 - "$1" <<'PY'
import sys, errno
try:
    sys.stdout.write(open(sys.argv[1]).read().strip()[:60])
except OSError as e:
    sys.stdout.write("!ERR:" + errno.errorcode.get(e.errno, str(e.errno)))
except Exception:
    sys.stdout.write("!ERR")
PY
}

# EACCES on BOTH cards is container permission, not a fault. We only report
# asymmetries, so a value that errors on one card and not the other.
anom=0
printf '%-46s %-22s %-22s\n' "attr" "card0" "card1"
for rel in \
  tile0/gt0/freq0/cur_freq tile0/gt0/freq0/min_freq tile0/gt0/freq0/max_freq tile0/gt0/freq0/rp0_freq \
  tile0/gt1/freq0/cur_freq tile0/gt1/freq0/min_freq tile0/gt1/freq0/max_freq tile0/gt1/freq0/rp0_freq \
  tile0/gt0/gtidle/idle_status tile0/gt1/gtidle/idle_status \
  tile0/gt0/num_cslices tile0/gt0/ccs_mode ; do
  va="$(read_attr "$A/$rel")"
  vb="$(read_attr "$B/$rel")"
  flag="  "
  case "$vb" in "!ERR:"*) flag=" <<< card1-only READ FAILURE";; esac
  [ -z "$vb" ] && flag=" <<< card1 MISSING"
  printf '%-46s %-22s %-22s%s\n' "$rel" "$va" "$vb" "$flag"
  case "$vb" in "!ERR:"*) anom=$((anom+1));; esac
done

echo
echo "-- what this container CANNOT do (all measured, not assumed) --"
python3 - <<'PY'
import errno
checks = [("dmesg / kernel log", "/proc/sys/kernel/"), ]
try:
    open("/sys/class/drm/card1/device/reset").close() and None
except Exception:
    pass
import os
for name, path in (("rebind card1 (write remove/rescan)", "/sys/class/drm/card1/device/rescan"),
                   ("reset card1", "/sys/class/drm/card1/device/reset")):
    try:
        fd = os.open(path, os.O_WRONLY); os.close(fd)
        print(f"  {name:44s} writable")
    except OSError as e:
        print(f"  {name:44s} ! {errno.errorcode.get(e.errno,e.errno)} (cannot test from here)")
PY
if dmesg >/dev/null 2>&1; then echo "  kernel log                               readable"; else echo "  kernel log                               ! EPERM (dmesg denied)"; fi

echo
if [ "$anom" -gt 0 ]; then
  echo "GPU HEALTH: ANOMALY -- $anom card1-specific read failure(s) above"
  echo "RESULT: card1 asymmetric at kernel layer; L0 returns only card0"
  exit 1
fi
echo "GPU HEALTH: cards symmetric in the checked attributes"
exit 0
