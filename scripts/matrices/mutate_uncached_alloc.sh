#!/usr/bin/env bash
# mutate_uncached_alloc.sh -- the ten-state negative direction for
# check_uncached_alloc.sh. A predicate that can only pass is not a predicate.
#
# WHY A SEPARATE FILE AND NOT mutate_s1_predicates.sh. That name is TAKEN: it is
# the negative direction for check_s1_predicates.sh, which certifies the PROBE
# (examples/uncached_doorbell_probe/main.cpp). This matrix certifies a
# different subject entirely -- the engine's uncached ALLOCATOR in
# src/engine/engine.cpp -- so reusing the name would make the coverage sweep's
# derive-by-stem rule (check_checker_matrix_coverage.sh) map two different
# subjects onto one matrix, and one of them would silently lose its negative
# direction.
#
# THE THREE STANDING FIXES, asserted here rather than intended. Each was a
# defect that made a "CAUGHT" mean less than it says:
#
#   * DEAD CONJUNCTION -- a shell function returns only its LAST statement's
#     status, so a predicate chain silently discards every guard before the
#     last. The checker therefore evaluates each predicate on its OWN line and
#     takes its verdict from a list's length.
#   * INVERTED POLARITY -- a state that DELETED the env read would make the
#     source MORE compliant, so "the checker refused it" would prove nothing.
#     Every state below PLANTS a non-compliant construct instead of deleting a
#     compliant one.
#   * UNLANDED MUTATION -- a mutation that changed nothing and was graded as a
#     catch certifies that the checker never saw the defect. Every state below
#     asserts it LANDED (byte-compare against the live subject) BEFORE it is
#     graded, and the refusal must name the predicate THAT STATE targets -- a
#     refusal for an unrelated reason would leave the predicate under test
#     unexercised, which is the "widened a check until the defect it exists to
#     find goes green" shape.
#
# THE STATES, each named for the predicate it must trip:
#   UC-presence      presence spelling -> exact-"1" spelling    p_presence
#   UC-default-on    add a disjunct to the enable test          p_default_off
#   UC-act           land an uncached allocation in .act        p_act_excluded
#   UC-alloc-outside a second zeMemAllocHost OUTSIDE the
#                    helper namespace                           p_alloc_confined
#   UC-release-api   zeMemFree -> sycl::free                    p_release_api
#   UC-release-drop  drop one release call                      p_site_pairing
#   UC-slot-null     null out the wave slot's assignment        p_both_workspaces
#   UC-provenance    plant a GLOBAL uncached pointer set        p_provenance
#   UC-announce      remove the env echo from the announcement  p_announce_echo
#   UC-drain         remove the destructor's pre-release drain  p_drain
#
# NO RESIDUE RISK. Every mutation is applied to a sandbox COPY of the subject;
# the tracked file is never edited. The residue assertion compares the tracked
# subject's sha AND its git status line against their PRE-RUN values, so it
# works on a tree that already carries uncommitted work (the "git status must be
# clean" form would fail here for a reason that has nothing to do with this
# matrix).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPP="$ROOT/src/engine/engine.cpp"
HPP="$ROOT/include/intfer/engine.hpp"
CHECKER="$ROOT/scripts/checks/check_uncached_alloc.sh"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/ucmut.XXXXXX")"
cleanup() { rm -rf "$SANDBOX"; }
trap cleanup EXIT
# INT/TERM must ALSO reach the cleanup: an EXIT trap alone never runs on a signal
# (bash fires EXIT only at a reached end or an explicit exit), so a SIGTERM
# killed the script outright and leaked $SANDBOX -- AUDIT #29. The exit routes
# the signal back through the EXIT handler; "clean up and resume" leaves the
# process alive, which is strictly worse (check_trap_release.sh).
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

die() { echo "UC-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }

command -v perl >/dev/null 2>&1 || die "perl unavailable; the multi-line mutations cannot LAND and must not be graded"
[ -f "$CPP" ] || die "subject missing ($CPP)"
[ -f "$HPP" ] || die "header missing ($HPP)"
[ -f "$CHECKER" ] || die "checker missing ($CHECKER)"

# ---- the two perl helpers: count an EXACT multi-line anchor, replace it -----
# \Q..\E so the anchors are matched literally: the engine's own text contains
# `(`, `*`, `>`, `[`, `$` and `"` that a bare regex would reinterpret.
count_of() { # $1 = file, $2 = literal
  [ -n "$2" ] || { echo 0; return 0; }
  UC_OLD="$2" perl -0777 -ne 'my @m = (/\Q$ENV{UC_OLD}\E/g); print scalar(@m), "\n"' "$1" 2>/dev/null || echo 0
}
sub_of() { # $1 = file, $2 = literal old, $3 = literal new
  UC_OLD="$2" UC_NEW="$3" perl -0777 -pi -e 's/\Q$ENV{UC_OLD}\E/$ENV{UC_NEW}/' "$1"
}

# ---- ANCHOR PREFLIGHT. Every mutation targets an anchor; an anchor that is
#      absent, or present more than once, makes its state vacuous (no target)
#      or ambiguous (a substitution that hits a different site than the one the
#      state names -- the exact shape of a mutation that changes nothing).
while IFS= read -r a; do
  [ -n "$a" ] || continue
  n="$(count_of "$CPP" "$a")"
  [ "$n" = "1" ] || die "anchor matched $n site(s) in the live engine.cpp (need exactly 1): $a"
done <<'ANCHORS'
std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr
static const bool on = (std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr);
          s.out = static_cast<std::uint16_t*>(uc);
    (void)zeMemFree(dev.ze_context, p);
      if (ws_[d].out_uncached) uncached_out::release(ws_[d].out, devices_[d]);
      if (w.out_uncached) uncached_out::release(w.out, devices_[d]);
site, std::getenv("INTFER_UNCACHED_HOST_USM"),
}  // namespace uncached_out
ANCHORS
n_ml="$(count_of "$CPP" $'    if (any_uncached)\n      for (size_t d = 0; d < devices_.size(); ++d) devices_[d].queue.wait();')"
[ "$n_ml" = "1" ] || die "the destructor's drain anchor matched $n_ml site(s) (need exactly 1)"
echo "UC-MUT: anchor preflight ok (9 anchors, each matched exactly once)"

# ---- BASELINE. The healthy source must PASS the checker, or every "caught"
#      below is indistinguishable from a checker that refuses everything.
bash "$CHECKER" "$CPP" "$HPP" >"$SANDBOX/baseline.txt" 2>&1
if [ $? -ne 0 ]; then
  cat "$SANDBOX/baseline.txt"
  die "the healthy source FAILS the checker -- the negative direction is meaningless"
fi
grep -q 'UC-ALLOC: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
echo "UC-MUT: baseline PASS on the unmutated source"

B_CPP="$(sha256sum "$CPP" | cut -d' ' -f1)"
B_HPP="$(sha256sum "$HPP" | cut -d' ' -f1)"
B_STAT="$(cd "$ROOT" && git status --porcelain -- src/engine/engine.cpp include/intfer/engine.hpp 2>/dev/null)"

# ---- plant: copy, mutate, assert LANDED, grade -----------------------------
# $1 = state, $2 = expected predicate, $3 = mutator function name
plant_and_grade() {
  local state="$1" want="$2" fn="$3" rc out why
  cp -f "$CPP" "$SANDBOX/$state.cpp"
  cp -f "$HPP" "$SANDBOX/$state.hpp"
  "$fn" "$SANDBOX/$state.cpp" "$SANDBOX/$state.hpp" || die "$state: the mutator command failed"
  # THE STANDING FIX: assert the mutation LANDED before grading it.
  if cmp -s "$SANDBOX/$state.cpp" "$CPP" && cmp -s "$SANDBOX/$state.hpp" "$HPP"; then
    die "$state: the mutation did NOT LAND -- refusing to grade a no-op as a catch"
  fi
  bash "$CHECKER" "$SANDBOX/$state.cpp" "$SANDBOX/$state.hpp" >"$SANDBOX/$state.out" 2>&1
  rc=$?
  out="$(cat "$SANDBOX/$state.out")"
  if [ "$rc" -eq 0 ]; then
    echo "UC-MUT: $state NOT CAUGHT -- the checker passed a defective source"
    printf '%s\n' "$out" | sed 's/^/    /'
    return 1
  fi
  why="$(printf '%s\n' "$out" | sed -n 's/^UC-ALLOC: FAIL -- refused by: //p' | head -1)"
  if ! printf '%s\n' "$why" | grep -qwF "$want"; then
    echo "UC-MUT: $state caught for the WRONG reason"
    echo "        expected: $want"
    echo "        refused:  $why"
    return 1
  fi
  echo "UC-MUT: $state CAUGHT ($want)"
  return 0
}

# ---- the ten mutators ------------------------------------------------------
m_presence() {
  sub_of "$1" 'std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr' 'std::getenv("INTFER_UNCACHED_HOST_USM") == "1"'
}
m_default_on() {
  sub_of "$1" 'static const bool on = (std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr);' \
             'static const bool on = (std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr) || true;'
}
m_act() {
  sub_of "$1" '          s.out = static_cast<std::uint16_t*>(uc);' \
             $'          s.act = static_cast<std::uint16_t*>(uc);\n          s.out = static_cast<std::uint16_t*>(uc);'
}
m_alloc_outside() {
  sub_of "$1" '}  // namespace uncached_out' \
             $'}  // namespace uncached_out\nstatic void* s1_duplicate_alloc(std::size_t bytes, const device::DeviceContext& dev) { ze_host_mem_alloc_desc_t d{}; d.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED; void* p = nullptr; (void)zeMemAllocHost(dev.ze_context, &d, bytes, 64, &p); return p; }'
}
m_release_api() {
  sub_of "$1" '    (void)zeMemFree(dev.ze_context, p);' '    (void)sycl::free(dev.ze_context, p);'
}
m_release_drop() {
  sub_of "$1" '      if (w.out_uncached) uncached_out::release(w.out, devices_[d]);' \
             '      if (w.out_uncached) (void)0;'
}
m_slot_null() {
  sub_of "$1" '          s.out = static_cast<std::uint16_t*>(uc);' \
             '          s.out = static_cast<std::uint16_t*>(nullptr);'
}
m_provenance() {
  sub_of "$1" '}  // namespace uncached_out' \
             $'}  // namespace uncached_out\n// A GLOBAL provenance set: which slot an uncached buffer lives in\n// is no longer recorded, so every free site must look the pointer up.\nstatic std::unordered_set<void*> g_uncached_set;'
}
m_announce() {
  sub_of "$1" 'site, std::getenv("INTFER_UNCACHED_HOST_USM"),' 'site, "unset",'
}
m_drain() {
  sub_of "$1" $'    if (any_uncached)\n      for (size_t d = 0; d < devices_.size(); ++d) devices_[d].queue.wait();' \
             $'    if (any_uncached)\n      for (size_t d = 0; d < devices_.size(); ++d) (void)0;'
}

STATES_REAL=0
STATES_CAUGHT=0
for st in UC-presence UC-default-on UC-act UC-alloc-outside UC-release-api UC-release-drop UC-slot-null UC-provenance UC-announce UC-drain; do
  fn="m_$(echo "$st" | sed 's/^UC-//;s/-/_/g')"
  declare -f "$fn" >/dev/null || die "$st: no mutator function $fn"
  # The expected predicate is stated once per state, here, so a rename of a
  # state cannot leave its expected predicate stale.
  case "$st" in
    UC-presence)      want=p_presence ;;
    UC-default-on)    want=p_default_off ;;
    UC-act)           want=p_act_excluded ;;
    UC-alloc-outside) want=p_alloc_confined ;;
    UC-release-api)   want=p_release_api ;;
    UC-release-drop)  want=p_site_pairing ;;
    UC-slot-null)     want=p_both_workspaces ;;
    UC-provenance)    want=p_provenance ;;
    UC-announce)      want=p_announce_echo ;;
    UC-drain)         want=p_drain ;;
    *) die "$st: no expected predicate mapped" ;;
  esac
  STATES_REAL=$((STATES_REAL + 1))
  if plant_and_grade "$st" "$want" "$fn"; then
    STATES_CAUGHT=$((STATES_CAUGHT + 1))
  fi
done

if [ "$STATES_REAL" -ne 10 ]; then
  die "expected 10 states, graded $STATES_REAL -- a matrix that never ran cannot report a tally"
fi

# ---- RESIDUE. The tracked subjects must be byte-identical to their pre-run
#      state AND their git status line unchanged: every mutation above ran on
#      a sandbox copy, so ANY change here is a mutation that escaped its copy.
A_CPP="$(sha256sum "$CPP" | cut -d' ' -f1)"
A_HPP="$(sha256sum "$HPP" | cut -d' ' -f1)"
[ "$A_CPP" = "$B_CPP" ] || die "engine.cpp CHANGED during the matrix (sha $B_CPP -> $A_CPP)"
[ "$A_HPP" = "$B_HPP" ] || die "engine.hpp CHANGED during the matrix (sha $B_HPP -> $A_HPP)"
A_STAT="$(cd "$ROOT" && git status --porcelain -- src/engine/engine.cpp include/intfer/engine.hpp 2>/dev/null)"
[ "$A_STAT" = "$B_STAT" ] || die "git status under the subjects CHANGED during the matrix"
echo "UC-MUT: residue byte-identical (engine ${A_CPP:0:16}..., header ${A_HPP:0:16}...)"

echo "UC-MUT: matrix $STATES_CAUGHT/$STATES_REAL states caught"
if [ "$STATES_CAUGHT" -eq "$STATES_REAL" ]; then
  echo "UC-ALLOC MATRIX: PASS ($STATES_CAUGHT/$STATES_REAL, LANDED asserted per state, residue byte-identical)"
  echo "RESULT: PASS"
  exit 0
fi
echo "UC-ALLOC MATRIX: FAIL"
echo "RESULT: FAIL"
exit 1
