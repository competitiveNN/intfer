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
echo "   (HEAD $(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo unknown))"

# ---- ANTI-VACUITY SELF-CHECK, run FIRST and gating everything below ----------
# The first version of this script globbed `card*/device`, which also matches
# CONNECTOR nodes (card0-DP-1, card1-HDMI-A-5). Those have no tile0/, so every
# attribute read as a card1-only ENOENT and the script reported 12 FABRICATED
# anomalies. The "fix" `card[0-9]*` was also wrong: bash's glob `*` matches
# "-DP-1" as well. A check whose SUBJECT is misidentified is worse than no
# check, so the subject selection is now proven before any reading.
selftest() {
  local bad=0 tmp
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/intfer_selftest.XXXXXX")"
  mkdir -p "$tmp/card0/device/tile0" "$tmp/card0-DP-1/device" "$tmp/card1/device/tile0"
  # a connector must be REJECTED, a real card ACCEPTED
  for n in card0 card1; do
    if ! ( cd "$tmp" && for p in card*/device; do
             b="$(basename "$(dirname "$p")")"
             [[ "$b" =~ ^card[0-9]+$ ]] && echo "$b"
           done ) | grep -qx "$n"; then
      echo "  SELFTEST FAIL: $n not selected by ^card[0-9]+\$"; bad=1
    fi
  done
  if ( cd "$tmp" && for p in card*/device; do
         b="$(basename "$(dirname "$p")")"
         [[ "$b" =~ ^card[0-9]+$ ]] && echo "$b"
       done ) | grep -qx 'card0-DP-1'; then
    echo "  SELFTEST FAIL: connector card0-DP-1 was selected"; bad=1
  fi
  rm -rf "$tmp"
  [ "$bad" -eq 0 ] && echo "  ok    subject selection rejects connectors, accepts cardN"
  return $bad
}
if ! selftest; then
  echo "SELF-CHECK FAILED -- subject selection is broken, so every reading below"
  echo "  would be about the wrong device. Refusing to report."
  exit 1
fi

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
# *** THE SECOND CARD'S PATH IS PASSED IN, NOT HARDCODED AS `card1`. *** Measured
# 2026-10-02: after a device-lost event the second B580 came back as
# /dev/class/drm/card2 (226:2) where it had been card1 (226:1), so the literal
# below pointed at a node that does not exist and this section reported
# "cannot test from here" about a path that was never the right one to test --
# a FALSE diagnostic, worse than printing nothing. The main body of this script
# already derives $A/$B by glob; this embedded block was the one place that had
# not been converted, which is the AUDIT #3/#12 shape again: the rule held at one
# site and was re-derived (wrongly) at the next.
GPU_HEALTH_CARD_B="$B" python3 - <<'PY'
import errno, os
b = os.environ["GPU_HEALTH_CARD_B"]
name_b = os.path.basename(os.path.dirname(b))
for name, path in ((f"rebind {name_b} (write rescan)", b + "/rescan"),
                   (f"reset {name_b}", b + "/reset")):
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
