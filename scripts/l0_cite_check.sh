#!/usr/bin/env bash
# Verify the compute-runtime citations this repo relies on, against a PINNED
# upstream commit -- so the AGENTS.md / report claims are checkable, not a
# transcript somebody has to trust.
#
# WHY: the "the L0 driver silently drops card1" analysis (see
# reports/2026-10-01-batch-e1-e2-e3-verdicts.md) cites specific file:line in
# intel/compute-runtime. Those were read by a research subagent, i.e. a
# transcript. The repo's recurring defect is exactly that: evidence that cannot
# be re-derived. This script clones the pinned commit and greps the quoted
# lines, so the claim either holds or the docs are wrong.
#
# PIN: 63647bb8214cdc581d2bd5ed3822f157f7d34679 (compute-runtime master).
# If upstream MOVES a line and this fails, that is a signal to re-read the code
# and UPDATE the docs -- not to loosen the grep.
#
# Claims checked (each is load-bearing somewhere in the docs):
#   1. product_helper_drm.cpp discards getMaxGpuFrequency's return value
#      -> refutes the "ECANCELED on max_freq causes the drop" theory
#   2. drm_neo.cpp returns -1 (drops the card) on small BAR under xe
#      -> the STRONGEST candidate for the drop
#   3. IoctlHelperXe::isSmallBarConfigAllowed() == false, the i915 one == true
#      -> why claim 2 only bites on xe
#   4. discovery scans /dev/dri/by-path, NOT /sys/class/drm
#   5. DRM_IOCTL_VERSION name must be exactly "i915" or "xe"
#   6. no GuC firmware load/check exists in shared/source (the GuC theory is dead)
#
# Usage: scripts/l0_cite_check.sh [--keep]
# Exit:  0 = every claim verified at the pin, 1 = a claim does NOT hold,
#        2 = could not resolve the pin (no network, no git) -- in which case the
#        docs must be read as UNVERIFIED, and the script says so.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIN="63647bb8214cdc581d2bd5ed3822f157f7d34679"
URL="https://github.com/intel/compute-runtime.git"
WORK="${INTFER_CR_TREE:-$ROOT/build/cr}"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

if [ ! -d "$WORK/.git" ]; then
  echo "== cloning compute-runtime at $PIN (sparse, this is ~1 min once) =="
  command -v git >/dev/null || { echo "SKIP: no git"; exit 2; }
  rm -rf "$WORK"
  git clone --quiet --depth 1 --filter=blob:none --no-checkout "$URL" "$WORK" 2>/dev/null \
    || { echo "SKIP: clone failed (no network?) -- citations are UNVERIFIED"; exit 2; }
  git -C "$WORK" sparse-checkout init --cone >/dev/null 2>&1
  git -C "$WORK" sparse-checkout set \
    shared/source/os_interface/linux shared/source/os_interface/linux/xe \
    shared/source/dll/linux >/dev/null 2>&1
  git -C "$WORK" checkout --quiet "$PIN" >/dev/null 2>&1 \
    || { echo "SKIP: cannot check out pin $PIN -- citations are UNVERIFIED"; exit 2; }
fi

got="$(git -C "$WORK" rev-parse HEAD 2>/dev/null)"
if [ "$got" != "$PIN" ]; then
  echo "FAIL: tree is at ${got:-<none>}, expected pinned $PIN"
  echo "      re-point or re-clone; do NOT weaken the checks below."
  exit 1
fi
echo "== compute-runtime pinned at $got =="
SRC="$WORK/shared/source"
bad=0
check() { # $1 = label, $2 = file, $3 = ERE, $4 = why it matters
  local n
  n="$(grep -nE "$3" "$SRC/$2" 2>/dev/null | head -1)"
  if [ -n "$n" ]; then
    printf '  OK    %-42s %s:%s\n' "$1" "$2" "${n%%:*}"
    printf '        %s\n' "$n"
  else
    printf '  FAIL  %-42s (%s)\n' "$1" "$4"
    bad=$((bad+1))
  fi
}

check "1 getMaxGpuFrequency return discarded" \
  "os_interface/linux/product_helper_drm.cpp" \
  '^[[:space:]]*drm->getMaxGpuFrequency\(' \
  "the quoted line does NOT read as quoted; the refutation of the ECANCELED theory loses its basis"

check "2 small-BAR drop: return -1" \
  "os_interface/linux/drm_neo.cpp" \
  'return -1;' \
  "strongest candidate for the drop is gone"

check "2b small-BAR guard: isSmallBarDetected" \
  "os_interface/linux/drm_neo.cpp" \
  'isSmallBarDetected\(\)' \
  "small-BAR branch not found"

check "2c small-BAR guard: isSmallBarConfigAllowed" \
  "os_interface/linux/drm_neo.cpp" \
  '!ioctlHelper->isSmallBarConfigAllowed\(\)' \
  "small-BAR branch not found"

check "3a xe rejects small BAR" \
  "os_interface/linux/xe/ioctl_helper_xe.h" \
  'isSmallBarConfigAllowed\(\) const override \{ return false; \}' \
  "the xe-vs-i915 asymmetry that makes claim 2 bite"

check "3b i915 allows small BAR" \
  "os_interface/linux/ioctl_helper.h" \
  'isSmallBarConfigAllowed\(\) const override \{ return true; \}' \
  "the i915 side of the asymmetry"

check "4 discovery scans by-path" \
  "dll/linux/options_linux.inl" \
  'pciDevicesDirectory = "/dev/dri/by-path"' \
  "we claim discovery does NOT use /sys/class/drm"

check "5 DRM version must be i915 or xe" \
  "os_interface/linux/drm_neo.cpp" \
  '"i915" == drmVersion \|\| "xe" == drmVersion' \
  "skip condition (b) is gone"

# claim 6: absence of GuC code. An ABSENCE, so grep must find nothing that
# looks like a firmware load or a GuC readiness check.
guc="$(grep -rniE 'GuC|gu_com' "$SRC" 2>/dev/null | grep -viE 'UC_TYPE_GUC_SUBMISSION|pmu|/guc_hwconfig|guc_fw' | head -5)"
if [ -z "$guc" ]; then
  printf '  OK    %-42s (no GuC load/check)\n' "6 no GuC firmware code in driver"
else
  printf '  FAIL  %-42s\n' "6 no GuC firmware code in driver"
  printf '%s\n' "$guc" | sed 's/^/        /'
  bad=$((bad+1))
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "L0 CITE CHECK: PASS (all claims verified at $PIN)"
  echo "RESULT: compute-runtime citations verified at pinned commit $PIN"
  [ "$KEEP" = "1" ] && echo "tree kept at $WORK"
  exit 0
fi
echo "L0 CITE CHECK: FAIL ($bad claim(s) do not hold at $PIN)"
echo "RESULT: at least one cited line does not read as quoted --"
echo "        RE-READ the code and UPDATE the docs. Do not loosen these greps."
exit 1