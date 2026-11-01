#!/usr/bin/env bash
# check_q4_q5_twin_symmetry.sh
#
# Pins EVERY KV activation site in main.cpp to its predicate, AND pins the
# three Hadamard-twin couplings (q8/had, q4/q4_had, q5_1/q5_1_had) to one
# identical structural shape. The 2026-10-28 re-run found main.cpp:10999
# shipped
#
#   ly.kv_q4_had = false;            # hardcoded -- the flag was never read
#
# while the twin at :11003 shipped
#
#   ly.kv_q5_1_had = ly.kv_q5_1 && kv_q5_1_had_enabled(); # correct
#
# T1 covers the codec-ON lines (kv_int8/q8/q4/q5_1/kvarn) so a blind mutant
# that swaps the predicate for a constant (false OR true -- both are behaviourally
# identical-to-defect forms, DC-4 could not see either) is caught at the call
# site rather than at model-load time.
# T2 covers the Hadamard coupling: each had line MUST carry the codec-ON guard.
# T3 covers the hardcoded-false defect (zero matches required).
# T4 covers the hardcoded-true defect (zero matches required).
# T5 covers the getenv-before-return-false body invariant: every kv_*_enabled()
# function body MUST read std::getenv and MUST NOT contain `return false;` as the
# body. This closes the second blind-mutant vector: DC-9/DC-13/DC-14 plant
# `return false;` *inside the function body* (not at the call site, where T1
# lives), and T1 alone cannot see that because the call site still consults
# the predicate. Covers kv_int8/kv_had/kv_kvarn (not just q4/q5_1) so the
# same late-disabling form can no longer hide.
#
# EXIT: 0 = all invariants; 1 = a site drifted. Each exact_one needs EXACTLY 1.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

SRC="examples/intfer_cli/main.cpp"
[ -f "$SRC" ] || { echo "FAIL: $SRC not found" >&2; exit 2; }

exact_one() { # $1 = label, $2 = pat
  local label="$1" pat="$2"
  n="$(grep -cF -- "$pat" "$SRC")"
  case "$n" in
    1) printf '  ok    %s   (1 match)\n' "$label" ;;
    0) echo "FAIL: T/$label violated: pattern not found (>=1 expected)" >&2
       echo "      $pat" >&2
       echo "      The site drifts: it stopped consulting its predicate." >&2
       exit 1 ;;
    *) echo "FAIL: T/$label ambiguous: matched $n times (expected 1)" >&2
       echo "      $pat" >&2
       echo "      Either the anchor drifted or both homes drifted together." >&2
       exit 1 ;;
  esac
}

echo "== checking KV activation sites are predatorially coupled =="
echo "  source: examples/intfer_cli/main.cpp"
echo

echo "-- T1: codec-ON lines consult the enabled() predicate (x5) --"
exact_one "T1-kv_int8-codec-on" "ly.kv_int8 = kv_int8_enabled();"
exact_one "T1-kv_q8-codec-on"    "ly.kv_q8 = kv_q8_enabled();"
exact_one "T1-kv_q4-codec-on"    "ly.kv_q4 = kv_q4_enabled();"
exact_one "T1-kv_q5_1-codec-on"  "ly.kv_q5_1 = kv_q5_1_enabled();"
exact_one "T1-kv_kvarn-codec-on"  "ly.kv_kvarn = kv_kvarn_enabled();"

echo
echo "-- T2: Hadamard twins carry the codec-ON guard (x3) --"
exact_one "T2-kv-had"    "ly.kv_had = ly.kv_q8 && kv_had_enabled();"
exact_one "T2-kv_q4-had" "ly.kv_q4_had = ly.kv_q4 && kv_q4_had_enabled();"
exact_one "T2-kv_q5_1-had" "ly.kv_q5_1_had = ly.kv_q5_1 && kv_q5_1_had_enabled();"

echo
echo "-- T3: hardcoded-false defect appears zero times (x9) --"
for ln in "ly.kv_int8 = false;" "ly.kv_q8 = false;" "ly.kv_had = false;" \
          "ly.kv_q4 = false;" "ly.kv_q4_had = false;" \
          "ly.kv_q5_1 = false;" "ly.kv_q5_1_had = false;" \
          "ly.kv_kvarn = false;"; do
  n="$(grep -cF -- "$ln" "$SRC")"
  case "$n" in
    0) printf '  ok    T3   no hardcoded false: %s   (0 matches)\n' "$ln" ;;
    *) echo "FAIL: T3 violated: hardcoded-false defect present: $ln" >&2
       echo "      this is the exact defect form that shipped on the q4_had twin" >&2
       exit 1 ;;
  esac
done

echo
echo "-- T4: hardcoded-true defect appears zero times (x9) --"
for ln in "ly.kv_int8 = true;" "ly.kv_q8 = true;" "ly.kv_had = true;" \
          "ly.kv_q4 = true;" "ly.kv_q4_had = true;" \
          "ly.kv_q5_1 = true;" "ly.kv_q5_1_had = true;" \
          "ly.kv_kvarn = true;"; do
  n="$(grep -cF -- "$ln" "$SRC")"
  case "$n" in
    0) printf '  ok    T4   no hardcoded true: %s   (0 matches)\n' "$ln" ;;
    *) echo "FAIL: T4 violated: hardcoded-true defect present: $ln" >&2
       echo "      predicate bypassed; INTFER_KV_* may not be honoured" >&2
       exit 1 ;;
  esac
done

echo
echo "-- T5: each of the 8 enabled() bodies reads std::getenv, not return false (x8) --"
# The eight enabled functions are enumerated EXACTLY by name, not by a broad
# count: a count-based check is satisfiable by an extra unrelated getenv in
# kv_capacity() (the 9th match) or by an unrelated `return false;` elsewhere in
# the block. For each name the check extracts the function body (signature to
# its closing `}`) and asserts the body contains `std::getenv` AND does NOT
# contain `return false;`. A mutant that swaps the env read for `return false;`
# (DC-9/DC-13/DC-14..21 blind-mutant class) breaks the per-name anchor rather
# than passing.
T5_NAMES=("kv_int8_enabled" "kv_q8_enabled" "kv_had_enabled" "kv_q4_enabled" "kv_q5_1_enabled" "kv_kvarn_enabled" "kv_q4_had_enabled" "kv_q5_1_had_enabled")
for fn in "${T5_NAMES[@]}"; do
  # Extract the function body using the literal function name as a marker.
  # awk matches by literal substring (index), not regex, so `() {` survives intact.
  sig="static inline bool ${fn}(){"
  body="$(awk -v sig="$sig" 'BEGIN{p=0} index($0,sig)==1 {p=1} p{print} p && $0=="}" {exit}' "$SRC")"
  if [ -z "$body" ]; then
    echo "FAIL: T5-${fn} violated: function body not found" >&2; exit 1
  fi
  case "$body" in
    *std::getenv*) printf '  ok    T5-%s   (body reads std::getenv)\n' "$fn" ;;
    *) echo "FAIL: T5-${fn} violated: body does not read std::getenv" >&2; exit 1 ;;
  esac
  case "$body" in
    *"return false"*) echo "FAIL: T5-${fn} violated: body contains return false" >&2; (exit 1) ;;
    *) printf '  ok    T5-%s   (no `return false` in body)\n' "$fn" ;;
  esac
done

echo
echo "KV ACTIVATION: PASS (5 invariant classes across 5 codecs, 3 Hadamard twins, 8 enabled fns)"
echo "  If this check fails, one activation site drifted or a blind mutant reran."
echo "RESULT: PASS"
exit 0
