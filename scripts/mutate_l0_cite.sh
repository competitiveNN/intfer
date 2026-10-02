#!/usr/bin/env bash
# Mutation matrix for scripts/check_l0_cite.sh.
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
# Usage: scripts/mutate_l0_cite.sh
# Exit:  0 = every state caught for its own reason, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
# Shared stamp-population enumeration (pop_dirs / known_dirs / pop_assert_no_leak),
# so this matrix is not a fourth copy of that derivation -- the same
# "rule fixed at one site, re-derived at another" shape as patch_literal.
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"
# Predicates with POLARITY NAMED at the call site. Sourced at top level, never
# inside a conditional: a guard sourced inside a conditional is a guard that is
# usually absent (the recorded _gate_enum.sh lesson).
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
SRC_TREE="${INTFER_CR_TREE:-$BUILD_DIR/cr}"
STAMP="$BUILD_DIR/l0_cite_mutate"
MUT="$BUILD_DIR/cr.mut"
LOG="$STAMP/states"

# The checker itself must work before we judge its ability to fail.
if [ ! -d "$SRC_TREE/.git" ]; then
  echo "SKIP: no pinned compute-runtime tree at $SRC_TREE"
  echo "      run scripts/check_l0_cite.sh first (it clones on demand)."
  exit 2
fi

mkdir -p "$LOG"
: > "$LOG/summary.txt"
caught=0; total=0

# The population this matrix requires, derived from what legitimately stamps
# rather than hardcoded: every run_all_checks.sh mutation matrix writes exactly
# one last_pass.stamp, and the census below confirms which directories those are.
# Hardcoding "6" is the hand-counted-total defect gate_inventory.sh exists to
# catch -- adding a 7th matrix would make this assertion fire on a legitimate
# tree, which is how a check gets "fixed" by weakening it.
pop_total() { find "$BUILD_DIR" -name '*last_pass.stamp' 2>/dev/null | sort; }
# BASENAME of the enclosing dir, which is what run_all_checks.sh reports and
# what a human compares. Deriving this from the mutate_*.sh NAMES was wrong in
# both directions: mutate_device_precond.sh stamps device_precond_matrix/
# mutate_last_pass.stamp (two stamps, two names) while mutate_fold_matrix.sh
# stamps fold_matrix/last_pass.stamp -- so a name-derived census said 5 for a
# correct 6 and the assertion fired on a healthy tree. A check that fails when
# nothing is wrong is the mirror defect, and it is how an assertion teaches
# people to ignore it. Census the ACTUAL directories instead and compare the
# SET, which is the thing the suite's step 3/3 enumerates.
# The set of stamp directories a matrix legitimately owns. Defined up here
# because BOTH the leak assertion and state M-H evaluate the same predicate
# against it; when it lived only next to the assertion, M-H ran first and died
# with "known_dirs: unbound variable", which set -u turns into a failed state.
# A helper referenced before it exists is the shell equivalent of a stale
# binary: the caller appears to run and proves nothing.
# DERIVED, not hand-listed. Each matrix declares where its own stamp lives
# (STAMP=... or LOG_DIR=...), so read those rather than maintaining a list.
# Hand-listing is what just broke: adding mutate_build_dir_consumers.sh made
# this assertion report a LEAK on a completely healthy tree, because the new
# matrix's build_dir_mutate/ dir was "unknown". That is the copy-drift family
# AUDIT #14/#15 closed twice, and the cure is ENUMERATION, not another entry --
# a false alarm here trains people to ignore the leak check.
#
# mutate_run_all_checks.sh stamps under a mktemp WORK dir, not the build tree,
# so it contributes nothing here; that is correct, not an omission.
# Read every path-shaped assignment to a build-tree root that a matrix makes --
# WORK=, LOG_DIR=, STAMP= -- because which of the three each uses varies
# (mutate_run_all_checks stamps via WORK, mutate_device_precond via LOG_DIR,
# the others via STAMP). Collecting only STAMP/LOG_DIR found four of five and
# then reported a LEAK on the fifth's healthy output.
# $WORK/$LOG_DIR indirections resolve to nothing on their own, so a value that
# is still a bare variable after substitution is dropped.

# pop_total is deliberately defined HERE, before any state runs: M-H calls it,
# and a first version of that state called it ~100 lines later, so bash printed
# "pop_total: command not found" and the state scored nothing. A helper
# referenced before it exists is the shell equivalent of a stale binary -- the
# mutation appears to run and proves nothing.

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
  out="$(INTFER_CR_TREE="$MUT" "$SCRIPT_DIR/check_l0_cite.sh" 2>&1)"; rc=$?
  printf '%s\n' "$out" > "$LOG/$id.log"
  if [ "$rc" -eq 0 ]; then
    echo "$id  MISSED  (checker still passed)" >> "$LOG/summary.txt"
    echo "  $id MISSED -- checker passed on a mutated tree"
    return 1
  fi
  # right reason, not merely "non-zero"
  # Polarity NAMED at the call site (inverted-predicate rule). This file did not
  # even source _gate_pred.sh, which is how it stayed invisible to the audit that
  # claimed the conversion was done.
  if pred_expect_present "$id property" "$out" "$expect"; then
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
base="$(INTFER_CR_TREE="$SRC_TREE" "$SCRIPT_DIR/check_l0_cite.sh" 2>&1)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL -- checker does not pass on the clean pinned tree."
  printf '%s\n' "$base" | tail -5
  exit 1
fi
echo "  ok    baseline passes (so a FAIL below is caused by the mutation, not by the tree)"

echo
# ---- the assertion itself, defined as a FUNCTION and called EARLY ----------
# ORDERING, third version. The assertion used to sit AFTER the mutation states
# as inline code, which meant a pre-existing leak in build/ was DELETED by M-H's
# `rm -rf` before the assertion read the population -- so planting a fixture
# before invoking the script still exited 0. Caught by running the negative
# direction BY HAND, twice: a self-consistent M-H keeps passing, because it only
# ever inspects a population it has just made clean itself.
#
# As a function it is (a) callable at the top against the tree AS FOUND, before
# anything mutates anything, and (b) callable by M-H against a deliberately
# poisoned population. One predicate, proven in both directions, one home.
leak_assert() { # $1 = label for the message
  local dirs unknown
  dirs="$(pop_dirs)"
  unknown="$(printf '%s\n' "$dirs" | while read -r d; do
              [ -n "$d" ] || continue
              case " $(known_dirs) " in *" $d "*) ;; *) echo "$d" ;; esac
            done)"
  if [ -n "$unknown" ]; then
    echo "  FAIL  [$1] stamp(s) in director(ies) no matrix owns:"
    printf '%s\n' "$unknown" | sed 's/^/          /'
    echo "        a checker is leaving fixture stamps in \$BUILD_DIR; run_all_checks"
    echo "        step 3/3 would then compare the wrong set. Full list:"
    pop_total | sed 's/^/          /'
    return 1
  fi
  echo "  ok    [$1] $(printf '%s\n' "$dirs" | grep -c .) stamp dirs, all owned by a known matrix ($(pop_total | wc -l) stamps)"
  return 0
}

echo
echo "== stamp-population leak assertion (BEFORE any mutation) =="
if ! leak_assert "as found"; then
  echo "RESULT: l0_cite_mutate LEAK (pre-existing fixture stamp in \$BUILD_DIR)"
  exit 1
fi

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

# M-H is NOT a claim mutation: it poisons the stamp population so the leak
# assertion must go RED. Without it the assertion is a comment.
#
# ORDERING, and this is the bug the first version of M-H shipped with. It
# planted the fixture, counted, then REMOVED it, and only THEN compared the
# count against $expected_pop -- which is a constant of this script, not a
# measurement of anything happening now. So with a real leak already present in
# build/ the state printed "ok -- population grew to 7" and the run exited 0
# anyway: the mutation was applied, observed, and then erased before the thing
# that could go red ran. Caught by running the negative test by HAND (planting
# the fixture before invoking the script), which is the only reason it was
# found; a self-consistent M-H would have kept passing.
#
# Fixed: the fixture stays planted across the assertion, the assertion runs
# while it is present, and its rc is required to be non-zero. M-H passes only
# when the assertion FAILS.
echo
echo "M-H (leak proof): plant a fixture stamp under build/ and require the"
echo "  leak assertion to REJECT it (the fixture is still present when it runs)"
POISON="$BUILD_DIR/l0_cite_mutate_fixture/last_pass.stamp"
mkdir -p "$(dirname "$POISON")"
echo "fixture" > "$POISON"

leak_dirs="$(pop_dirs)"
poisoned=0
if printf '%s\n' "$leak_dirs" | grep -qx 'l0_cite_mutate_fixture'; then poisoned=1; fi
if [ "$poisoned" -eq 0 ]; then
  echo "  M-H UNREACHABLE -- pop_total cannot see the planted fixture, so the"
  echo "    assertion is vacuous against exactly the defect it exists to catch."
  rm -rf "$BUILD_DIR/l0_cite_mutate_fixture"
  echo "M-H  BROKEN  (pop_total blind to fixtures)" >> "$LOG/summary.txt"
  echo "RESULT: l0_cite_mutate FAIL (M-H vacuous - pop_total blind to fixtures)"
  exit 1
fi

# Call the SAME leak_assert() against the POISONED population, while the
# fixture is still on disk. No second copy of the predicate: the first version
# inlined a duplicate here, so the state proved the DUPLICATE could fail while
# the shipped assertion stayed unproven -- the audit #3 "rule fixed at one site
# and re-derived at another" shape, in shell.
assert_rc=0
leak_assert "poisoned by M-H" || assert_rc=1

rm -rf "$BUILD_DIR/l0_cite_mutate_fixture"
if [ "$assert_rc" -ne 0 ]; then
  echo "  M-H ok -- the fixture dir WAS visible to pop_dirs and the SHIPPED"
  echo "    assertion rejected it while it was on disk. It CAN go red."
else
  echo "  M-H MISSED -- the fixture was on disk yet the assertion accepted it"
  echo "M-H  MISSED  (assertion accepted a leaked population)" >> "$LOG/summary.txt"
  echo "RESULT: l0_cite_mutate FAIL (M-H missed - leak assertion cannot fail)"
  exit 1
fi

rm -rf "$MUT"

# ---- STAMP-POPULATION LEAK ASSERTION (fail-closed, not a note) -------------
# snap the population AT THIS POINT in the run, not by re-deriving it from a
# constant. The first version compared the live count against the hardcoded
# $expected_pop=6, so with a pre-existing leak the run still exited 0: it
# printed "ok population is exactly 6" while seven stamps were on disk, because
# M-H had already deleted the fixture it planted and the count it compared was
# therefore its own doing rather than a property of the tree. Caught by running
# the negative test by HAND.
# run_all_checks.sh enumerates the stamp set with
# `find $BUILD_DIR -name '*last_pass.stamp'`, so this matrix's own fixtures --
# which necessarily live under build/ because they are mutated COPIES of the
# pinned tree -- must not add anything to that population. AUDIT #17c's leak
# class: an enumerator measuring its own test data.
#
# This is an ASSERTION with its own exit code, not a printed note. The first
# version only echoed a LEAK line and then exited 0 on the states, which means a
# leaking run still read as PASS -- a comment with a shebang, the same defect
# shape as AUDIT #7. It also FAILED CLOSED on the wrong trigger: it probed for
# stamps inside build/cr.mut AFTER `rm -rf`, so it could never observe anything
# even if a state leaked, i.e. a check that cannot fail. State M-H poisons the
# population deliberately to prove this now goes red.
# SECOND CALL, after all states have run and restored. The early call guards the
# tree as FOUND; this one guards what the matrix itself left behind. The inline
# duplicate that used to live here is GONE -- it was a second copy of the
# predicate, so the early assertion and this one could disagree, and M-H would
# have proven the copy rather than the shipped check (the audit #3 shape).
if ! leak_assert "after states"; then
  echo "RESULT: l0_cite_mutate LEAK (a state left a fixture stamp behind)"
  exit 1
fi

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