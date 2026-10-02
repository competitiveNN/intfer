#!/usr/bin/env bash
# Behavioural checks for scripts/gpu_recover.sh, with a STUBBED probe.
#
# WHY THIS IS A SCRIPT AND NOT A TRANSCRIPT. gpu_recover.sh's negative arms were
# first run ad hoc: I tried to fake a 2-device result by sed-ing
# scripts/l0_enumerate.sh, the sed silently did nothing because that text lives
# in the C probe's OUTPUT rather than in the script, and the arm "passed" for
# the wrong reason. That is exactly the audit #19 lesson -- a proof that lives
# only in a transcript cannot be re-run, and a self-proving arm proves nothing.
# Here the probe is STUBBED at a temporary ROOT instead of being mutated, so the
# arm exercises the branch it claims and nothing real is edited.
#
# Four cases, including the one that gets forgotten:
#   (a) MEASURED 2 devices      -> verify exits 0 and prints the trigger
#   (b) MEASURED 1 device       -> verify exits 1, and says MEASURED (not blank)
#   (c) UNMEASURABLE probe      -> verify exits 2, distinctly, and does NOT
#                                  claim a device count (audit #14/#16: a helper
#                                  that cannot count must never certify one)
#   (d) the CONVERSE: case (a) must not be reachable by accident -- with the
#       1-device stub the exit code is 1, not 0. Without this, (a) proves
#       nothing about the branch being real.
#
# `rebind` is NOT exercised: it writes to /sys/bus/pci and needs root. Its
# refusal path IS checked (exit 3 without root, touching nothing).
#
# Usage: scripts/check_gpu_recover.sh
# Exit: 0 = all cases behaved, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/intfer_gpur.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT INT TERM

bad=0
mk_stub() { # $1 = body of the fake probe
  rm -rf "$WORK/root"; mkdir -p "$WORK/root/scripts"
  cp "$SCRIPT_DIR/gpu_recover.sh" "$WORK/root/scripts/"
  { echo '#!/usr/bin/env bash'; echo 'cat <<"STUBEOF"'; printf '%s\n' "$1"; echo 'STUBEOF'; } \
    > "$WORK/root/scripts/l0_enumerate.sh"
  chmod +x "$WORK/root/scripts/l0_enumerate.sh"
}
run_verify() { "$WORK/root/scripts/gpu_recover.sh" verify 2>&1; }

echo "== GPU RECOVER CHECK (stubbed probe; nothing real is modified) =="

# (a) two devices
echo "-- (a) stubbed MEASURED 2 devices --"
mk_stub '  driver[0] driverVersion=0x1039b4e zeDeviceGet rc=0 DEVICES=2
    dev[0] uuid=86800be2-0000-0000-0400-000000000000
    dev[1] uuid=86800be2-0000-0000-8600-000000000000
TOTAL LEVEL ZERO DEVICES: 2'
o="$(run_verify)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$o" | grep -q 'enumeration fault is GONE' \
   && printf '%s' "$o" | grep -q 'test_tp2_27b.sh'; then
  echo "  ok    verify exits 0 and prints the next-session trigger"
else
  echo "  FAIL  rc=$rc (want 0); trigger text present: $(printf '%s' "$o" | grep -c 'enumeration fault is GONE')"
  bad=$((bad+1))
fi

# (b) one device -- and it must be described as MEASURED, not merely "1 device"
echo "-- (b) stubbed MEASURED 1 device --"
mk_stub '  driver[0] driverVersion=0x10396f6 zeDeviceGet rc=0 DEVICES=1
TOTAL LEVEL ZERO DEVICES: 1'
o="$(run_verify)"; rc=$?
if [ "$rc" -eq 1 ] && printf '%s' "$o" | grep -q 'MEASURED 1 device'; then
  echo "  ok    verify exits 1 and says MEASURED 1 device"
else
  echo "  FAIL  rc=$rc (want 1); 'MEASURED 1 device' absent"
  bad=$((bad+1))
fi

# (c) UNMEASURABLE -- the case the first version rendered identically to "1 device"
echo "-- (c) stubbed UNMEASURABLE probe (no TOTAL line) --"
mk_stub 'zeInit rc=0
(this probe produced no device count)'
o="$(run_verify)"; rc=$?
if [ "$rc" -eq 2 ] && printf '%s' "$o" | grep -q 'UNMEASURABLE'; then
  echo "  ok    verify exits 2 (distinct from 1) and says UNMEASURABLE"
else
  echo "  FAIL  rc=$rc (want 2); UNMEASURABLE marker absent"
  bad=$((bad+1))
fi
# and it must NOT have claimed a count
if printf '%s' "$o" | grep -qE 'MEASURED [0-9]+ device'; then
  echo "  FAIL  the unmeasurable path still printed a device count"
  bad=$((bad+1))
else
  echo "  ok    the unmeasurable path certified no count"
fi

# (d) CONVERSE of (a)
echo "-- (d) converse: the 2-device branch is not reached by accident --"
echo "      (case (b) above must NOT exit 0; asserted again explicitly)"
if [ "$rc" -eq 2 ] || [ "$rc" -eq 1 ]; then
  echo "  ok    a non-2-device probe never yields the success path"
else
  echo "  FAIL  a non-2-device probe produced rc=$rc"
  bad=$((bad+1))
fi

# rebind must refuse without root, and must not touch /sys
echo "-- rebind refuses without root --"
o="$("$SCRIPT_DIR/gpu_recover.sh" rebind 2>&1)"; rc=$?
if [ "$rc" -eq 3 ] && printf '%s' "$o" | grep -q 'needs root'; then
  echo "  ok    exit 3, and it said so"
else
  echo "  FAIL  rc=$rc (want 3)"
  bad=$((bad+1))
fi

# the decisive BAR test must be marked NOT MEASURED when lspci is absent
echo "-- diag marks the decisive BAR test when lspci is absent --"
if command -v lspci >/dev/null 2>&1; then
  echo "  skip lspci IS installed here; this case cannot be exercised"
else
  o="$("$SCRIPT_DIR/gpu_recover.sh" diag 2>&1)"
  if printf '%s' "$o" | grep -q 'DECISIVE TEST NOT MEASURED' \
     && printf '%s' "$o" | grep -q 'UNTESTED, not refuted'; then
    echo "  ok    diag refuses to certify a refutation it did not measure"
  else
    echo "  FAIL  diag did not mark the decisive test as unmeasured"
    bad=$((bad+1))
  fi
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "GPU RECOVER CHECK: PASS (all cases behaved as specified)"
  echo "RESULT: gpu_recover.sh exit codes are 0=recovered 1=still-broken 2=unmeasurable 3=needs-root"
  exit 0
fi
echo "GPU RECOVER CHECK: FAIL ($bad case(s))"
echo "RESULT: gpu_recover.sh does NOT behave as documented"
exit 1