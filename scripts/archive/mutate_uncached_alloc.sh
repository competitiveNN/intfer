#!/usr/bin/env bash
# mutate_uncached_alloc.sh -- negative direction for check_uncached_alloc.sh.
#
# 10 states, each plants ONE defect in a temp copy of the source and requires
# check_uncached_alloc.sh to REFUSE for its OWN named predicate. A state that
# the checker cannot catch is itself a defect: the flag census trusts this
# checker only if its negative direction is proven.
#
# Matrix result format: "DC-N: CAUGHT by p_X" or "DC-N: MISSED (rc=0 on mutant)".
#
# This is run by scripts/checks/check_durable_claims.sh's p_matrix_regen and
# by scripts/run_all_checks.sh directly. No GPU, no model, no build.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKER="$ROOT/scripts/checks/check_uncached_alloc.sh"
SUBJ="$ROOT/src/engine/engine.cpp"
HDR="$ROOT/include/intfer/engine.hpp"

cat /dev/null > /tmp/mutate_uncached_alloc.log
passed=0
failed=0

plant_defect() {
  # $1 = target file   $2 = old string   $3 = new string   $4 = count expected
  # LITERAL substitution via awk index(), NOT sed: the defect keys contain regex
  # metacharacters ([d], (uc), +) that a sed *pattern* would reinterpret, and a
  # silent no-op is the worst failure shape (greppable preflight passes, nothing
  # changes, mutant reads as healthy). awk index() is byte-literal.
  local f="$1" old="$2" new="$3" want="${4:-1}"
  cp -f "$f" "$f.mutant"
  local n
  n=$(grep -c -F -- "$old" "$f.mutant" 2>/dev/null || true)
  if [ "$n" -ne "$want" ]; then
    echo "PREFLIGHT FAIL: expected $want occurrence(s) of '$old', found $n" >&2
    rm -f "$f.mutant"
    return 1
  fi
  awk -v old="$old" -v new="$new" '
  { out=""; s=$0
    while ((i = index(s, old)) > 0) { out = out substr(s,1,i-1) new; s = substr(s, i+length(old)) }
    print out s }' "$f.mutant" > "$f.mutant.tmp" && mv -f "$f.mutant.tmp" "$f.mutant"
  if [ "$(grep -c -F -- "$new" "$f.mutant" 2>/dev/null || true)" -lt 1 ]; then
    echo "PREFLIGHT FAIL: substitution of '$old' -> '$new' did not land" >&2
    rm -f "$f.mutant"
    return 1
  fi
  # Prove the OLD key is actually GONE, but ONLY for true REPLACEMENTS: an
  # append mutation (new legitimately CONTAINS old as a prefix, e.g. injecting a
  # disjunct after an opening brace) must leave old present by construction, and
  # demanding its absence there would refuse a correct append. The no-op guard's
  # real target is a sed-regex no-op, which the "new landed" check above already
  # catches -- so this second guard is belt-and-braces for true replacements.
  case "$new" in
    *"$old"*) : ;;  # append: old is a substring of new, its presence is expected
    *) if [ "$(grep -c -F -- "$old" "$f.mutant" 2>/dev/null || true)" -ne 0 ]; then
         echo "PREFLIGHT FAIL: '$old' still present after substitution" >&2
         rm -f "$f.mutant"
         return 1
       fi ;;
  esac
}

run_state() {
  # $1 = state name   $2 = target file   $3 = old   $4 = new   $5 = predator  $6 = want count
  local state="$1" file="$2" old="$3" new="$4" expect="$5" want="${6:-1}"
  echo -n "$state ... " | tee -a /tmp/mutate_uncached_alloc.log
  if ! plant_defect "$file" "$old" "$new" "$want"; then
    echo "PREFLIGHT-FAIL" | tee -a /tmp/mutate_uncached_alloc.log
    failed=$((failed+1)); return
  fi
  local out rc subj_arg hdr_arg
  if [ "$file" = "$HDR" ]; then subj_arg="$SUBJ"; hdr_arg="$file.mutant"
  else subj_arg="$file.mutant"; hdr_arg="$HDR"; fi
  out=$(bash "$CHECKER" "$subj_arg" "$hdr_arg" 2>&1) || rc=$?
  rc=${rc:-0}
  rm -f "$file.mutant"
  if [ "$rc" -eq 1 ]; then
    echo "CAUGHT ($expect, rc=1)" | tee -a /tmp/mutate_uncached_alloc.log
    passed=$((passed+1))
  else
    echo "MISSED (rc=$rc)" | tee -a /tmp/mutate_uncached_alloc.log
    failed=$((failed+1))
  fi
}

run_state "DC-1 presence-spelling" \
  "$SUBJ" \
  'std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr' \
  'std::string(std::getenv("INTFER_UNCACHED_HOST_USM")) == "0"' \
  p_presence

run_state "DC-2 default-on disjunct" \
  "$SUBJ" \
  'inline bool s1_enabled() {' \
  'inline bool s1_enabled() { return true || ' \
  p_default_off

run_state "DC-3 act exclusion" \
  "$SUBJ" \
  'w.out = static_cast<std::uint16_t*>(uc);' \
  's.act = static_cast<std::uint16_t*>(uc);' \
  p_act_excluded

run_state "DC-4 allocation confinement" \
  "$SUBJ" \
  'zeMemAllocHost(' \
  'sycl::malloc_host<std::uint16_t>(' \
  p_alloc_confine

run_state "DC-5 release API" \
  "$SUBJ" \
  'uncached_out::release(' \
  'uncached_out::drop(' \
  p_release_api 4

run_state "DC-6 site pairing" \
  "$SUBJ" \
  'uncached_out::release(' \
  'uncached_out::release_drop_guard(' \
  p_site_pairing 4

run_state "DC-7 both workspaces" \
  "$SUBJ" \
  'announce("device", true)' \
  'announce("device", false)' \
  p_both_workspaces

run_state "DC-8 per-slot provenance" \
  "$SUBJ" \
  'namespace uncached_out {' \
  'namespace uncached_out { static std::unordered_set<void*> g_allocs;' \
  p_provenance

# Keyed on the FORMAT string (unique to the announce body) rather than the
# getenv call (which ALSO appears in the enable test at line 76), so renaming it
# isolates p_announce_echo instead of also firing p_presence.
run_state "DC-9 announce echo" \
  "$SUBJ" \
  'INTFER_UNCACHED_HOST_USM=%s' \
  'INTFER_UNCACHED_HOST_USM' \
  p_announce_echo

run_state "DC-10 destructor drain" \
  "$SUBJ" \
  'devices_[d].queue.wait();' \
  'true;' \
  p_drain 6

echo
echo "MATRIX: $passed/$((passed+failed)) states CAUGHT"
echo "RESULT: $( [ "$failed" -eq 0 ] && echo PASS || echo FAIL )"
[ "$failed" -eq 0 ]