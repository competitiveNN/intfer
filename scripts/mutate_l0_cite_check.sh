#!/usr/bin/env bash
# Mutation matrix for scripts/l0_cite_check.sh.
#
# WHY: the cite-checker is load-bearing for a claim this repo now depends on
# ("the L0 driver silently drops card1, and the small-BAR branch at
# drm_neo.cpp is where"). A checker that cannot fail proves nothing, and one
# that can only pass also proves nothing. AGENTS.md's AUDIT #14 lesson: a
# claim in a commit message is not a re-runnable artifact.
#
# Each state mutates a COPY of the pinned compute-runtime tree (never the real
# one, never tracked source) and asserts the checker rejects it FOR THE RIGHT
# REASON. Scored MISSED if the checker passed, and separately if it went red
# for a different reason -- a checker that fails for an unrelated reason is
# evidence of nothing.
#
# Usage: scripts/mutate_l0_cite_check.sh
# Exit:  0 = every state caught for its own reason, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_TREE="${INTFER_CR_TREE:-$ROOT/build/cr}"
STAMP="$ROOT/build/l0_cite_mutate"
MUT="$ROOT/build/cr.mut"
LOG="$STAMP/states"

# The checker itself must work before we judge its ability to fail.
if [ ! -d "$SRC_TREE/.git" ]; then
  echo "SKIP: no pinned compute-runtime tree at $SRC_TREE"
  echo "      run scripts/l0_cite_check.sh first (it clones on demand)."
  exit 2
fi

mkdir -p "$LOG"
: > "$LOG/summary.txt"
caught=0; total=0

# run_state <id> <expect-substring-in-FAIL-line> <mutation-command...>
run_state() {
  local id="$1" expect="$2"; shift 2
  total=$((total+1))
  rm -rf "$MUT"; cp -r "$SRC_TREE" "$MUT" || { echo "$id SETUP-FAIL" >> "$LOG/summary.txt"; return 1; }
  if ! ( cd "$MUT" && "$@" ); then
    echo "$id  MISSED  (mutation did not apply)" >> "$LOG/summary.txt"
    echo "  $id MISSED -- mutation did not apply; a state that mutates nothing scores nothing"
    return 1
  fi
  local out rc
  out="$(INTFER_CR_TREE="$MUT" "$SCRIPT_DIR/l0_cite_check.sh" 2>&1)"; rc=$?
  printf '%s\n' "$out" > "$LOG/$id.log"
  if [ "$rc" -eq 0 ]; then
    echo "$id  MISSED  (checker still passed)" >> "$LOG/summary.txt"
    echo "  $id MISSED -- checker passed on a mutated tree"
    return 1
  fi
  # right reason, not merely "non-zero"
  if printf '%s\n' "$out" | grep -qE "$expect"; then
    echo "$id  caught  ($expect)" >> "$LOG/summary.txt"
    echo "  $id caught -- $(printf '%s\n' "$out" | grep -m1 -E 'FAIL' | sed 's/^ *//')"
    caught=$((caught+1)); return 0
  fi
  echo "$id  UNATTRIBUTED  (rc=$rc but no '$expect')" >> "$LOG/summary.txt"
  echo "  $id UNATTRIBUTED -- rc=$rc but not for the expected reason"
  printf '%s\n' "$out" | grep -E 'FAIL|RESULT' | sed 's/^/      /'
  return 1
}

echo "== baseline: unmutated tree must PASS =="
base="$(INTFER_CR_TREE="$SRC_TREE" "$SCRIPT_DIR/l0_cite_check.sh" 2>&1)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL -- checker does not pass on the clean pinned tree."
  printf '%s\n' "$base" | tail -5
  exit 1
fi
echo "  ok    baseline passes (so a FAIL below is caused by the mutation, not by the tree)"

echo
echo "== states =="

# 3a: xe accepts small BAR -> the xe-vs-i915 asymmetry disappears
run_state "M-A" "3a xe rejects small BAR" \
  sed -i 's|isSmallBarConfigAllowed() const override { return false; }|isSmallBarConfigAllowed() const override { return true; }|' \
  shared/source/os_interface/linux/xe/ioctl_helper_xe.h

# 3b: i915 rejects small BAR -> asymmetry gone the other way
run_state "M-B" "3b i915 allows small BAR" \
  sed -i 's|isSmallBarConfigAllowed() const override { return true; }|isSmallBarConfigAllowed() const override { return false; }|' \
  shared/source/os_interface/linux/ioctl_helper.h

# 1: the discarded return is "fixed" -> the refutation of the ECANCELED theory
#    loses its basis
run_state "M-C" "1 getMaxGpuFrequency return discarded" \
  sed -i 's|^    drm->getMaxGpuFrequency(\*outHwInfo, maxGpuFreq);|    if (drm->getMaxGpuFrequency(*outHwInfo, maxGpuFreq)) { return -1; }|' \
  shared/source/os_interface/linux/product_helper_drm.cpp

# 2c: the small-BAR guard is removed -> the strongest candidate evaporates
run_state "M-D" "2c small-BAR guard: isSmallBarConfigAllowed" \
  sed -i 's|if (!ioctlHelper->isSmallBarConfigAllowed()) {|if (false) {|' \
  shared/source/os_interface/linux/drm_neo.cpp

# 4: discovery stops scanning by-path -> our "not /sys/class/drm" claim breaks
run_state "M-E" "4 discovery scans by-path" \
  sed -i 's|const char \*pciDevicesDirectory = "/dev/dri/by-path";|const char *pciDevicesDirectory = "/sys/class/drm";|' \
  shared/source/dll/linux/options_linux.inl

# 6: plant GuC firmware code -> the "no GuC in the driver" absence claim breaks
run_state "M-F" "6 no GuC firmware code in driver" \
  bash -c 'printf "\nvoid loadGucFirmware() {}\n" >> shared/source/os_interface/linux/drm_neo.cpp'

# ABSENCE claims are the ones most likely to pass vacuously, so give the
# absence check its own state: delete the xe header entirely. 3a must then fail
# for a DIFFERENT reason than M-A (missing file), which is also worth seeing.
run_state "M-G" "3a xe rejects small BAR" \
  rm -f shared/source/os_interface/linux/xe/ioctl_helper_xe.h

rm -rf "$MUT"

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "L0 CITE MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=l0_cite_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "pin=63647bb8214cdc581d2bd5ed3822f157f7d34679"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "L0 CITE MUTATE: FAIL ($caught/$total caught)"
exit 1