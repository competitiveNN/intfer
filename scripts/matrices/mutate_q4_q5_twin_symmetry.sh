#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_q4_q5_twin_symmetry.sh.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/twin_sym_matrix"
CHECK="scripts/checks/check_q4_q5_twin_symmetry.sh"
ART="examples/intfer_cli/main.cpp"

[ -f "$CHECK" ] || { echo "FAIL: $CHECK missing" >&2; exit 2; }
bash -n "$CHECK" || { echo "FAIL: $CHECK not syntactically valid" >&2; exit 2; }

mkdir -p "$SB" 2>/dev/null || { echo "FAIL: cannot create $SB" >&2; exit 2; }
PRISTINE="$SB/pristine.main.cpp"
cp "$ART" "$PRISTINE"

restore() { cp "$PRISTINE" "$ART"; }

total=0; caught=0; RESULTS=()
record() { case "$2" in ok) caught=$((caught + 1)); RESULTS+=("$1 CAUGHT");; *) RESULTS+=("$1 $2");; esac; }

run_state() {  # $1 = id, $2 = plant pattern (exact), $3 = replacement
  local id="$1" pat="$2" repl="$3"
  total=$((total + 1))
  python3 - "$ART" "$pat" "$repl" <<'PY'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) != 1:
    print("ANCHOR-DRIFT: %d matches for %r" % (s.count(o), o), file=sys.stderr); sys.exit(1)
open(p, 'w').write(s.replace(o, n))
PY
  if [ $? -ne 0 ]; then record "$id" "ANCHOR-DRIFT"; restore; return; fi
  bash "$CHECK" >"$SB/$id.log" 2>&1; local rc=$?
  restore
  if [ "$rc" -ne 0 ]; then record "$id" ok
  else record "$id" "MISSED (checker still PASS)"; fi
}

echo "== baseline (pristine source) =="
bash -n "$CHECK" >/dev/null 2>&1 && bash "$CHECK" >"$SB/base.log" 2>&1; rc=$?
if [ "$rc" -ne 0 ]; then echo "  FAIL baseline rc=$rc" >&2; cat "$SB/base.log" >&2; exit 1; fi
grep -q '^RESULT: ' "$SB/base.log" || { echo "  FAIL baseline has no RESULT:" >&2; exit 1; }
echo "  ok baseline rc=0, PASS verdict present"

echo
echo "== states =="
# Twin-A/B: original twin drift
run_state "TA-HARDSF" "ly.kv_q4_had = ly.kv_q4 && kv_q4_had_enabled();" "ly.kv_q4_had = false;"
run_state "TB-HARDSF" "ly.kv_q5_1_had = ly.kv_q5_1 && kv_q5_1_had_enabled();" "ly.kv_q5_1_had = false;"
# Twin-C: T2 shape drift (guard dropped)
run_state "TC-SHAPEDRIFT" "ly.kv_q5_1_had = ly.kv_q5_1 && kv_q5_1_had_enabled();" "ly.kv_q5_1_had = kv_q5_1_had_enabled();"
# Twin-D/E: T1 codec-ON stops consulting predicate (true)
run_state "TD-T1DRIFT" "ly.kv_q4 = kv_q4_enabled();" "ly.kv_q4 = true;"
run_state "TE-T1DRIFT" "ly.kv_q5_1 = kv_q5_1_enabled();" "ly.kv_q5_1 = true;"
# DC-14: q8 codec-ON hardcoded false (blind mutant)
run_state "DF-Q8-HARDSF" "ly.kv_q8 = kv_q8_enabled();" "ly.kv_q8 = false;"
# DC-15: kvarn codec-ON hardcoded false
run_state "DG-KVARN-HARDSF" "ly.kv_kvarn = kv_kvarn_enabled();" "ly.kv_kvarn = false;"
# T1 true drift for q8/kvarn
run_state "DH-Q8-HARDTRUE" "ly.kv_q8 = kv_q8_enabled();" "ly.kv_q8 = true;"
run_state "DI-KVARN-HARDTRUE" "ly.kv_kvarn = kv_kvarn_enabled();" "ly.kv_kvarn = true;"
# q8 kv_had twin
run_state "DJ-KV-HAD-HARDSF" "ly.kv_had = ly.kv_q8 && kv_had_enabled();" "ly.kv_had = false;"
run_state "DK-KV-HAD-HARDTRUE" "ly.kv_had = ly.kv_q8 && kv_had_enabled();" "ly.kv_had = true;"
# q8 T2 shape drift (guard dropped)
run_state "DL-KV-HAD-SHAPEDRIFT" "ly.kv_had = ly.kv_q8 && kv_had_enabled();" "ly.kv_had = kv_had_enabled();"
# int8 codec-ON
run_state "DM-INT8-HARDSF" "ly.kv_int8 = kv_int8_enabled();" "ly.kv_int8 = false;"
# DC-14..DC-21: T5 refusal directions -- plant `return false;` inside each
# enabled-function body (the DC-9/DC-13 blind-mutant class at the function rather
# than the call site). T5's global scan catches each. 8 states, one per enabled fn body.
run_state "DC-14" $'static inline bool kv_int8_enabled(){\n  static const bool on = (std::getenv("INTFER_INT8_KV") != nullptr);\n  return on;\n}' $'static inline bool kv_int8_enabled(){\n  return false;\n}'
run_state "DC-15" $'static inline bool kv_q8_enabled(){\n  static const bool on = (std::getenv("INTFER_KV_Q8") != nullptr);\n  return on;\n}' $'static inline bool kv_q8_enabled(){\n  return false;\n}'
run_state "DC-16" $'static inline bool kv_had_enabled(){\n  static const bool on = (std::getenv("INTFER_KV_HAD") != nullptr);\n  return on;\n}' $'static inline bool kv_had_enabled(){\n  return false;\n}'
run_state "DC-17" $'static inline bool kv_q4_enabled(){\n  static const bool on = (std::getenv("INTFER_KV_Q4") != nullptr);\n  return on;\n}' $'static inline bool kv_q4_enabled(){\n  return false;\n}'
run_state "DC-18" $'static inline bool kv_q5_1_enabled(){\n  static const bool on = (std::getenv("INTFER_KV_Q5_1") != nullptr);\n  return on;\n}' $'static inline bool kv_q5_1_enabled(){\n  return false;\n}'
run_state "DC-19" $'static inline bool kv_kvarn_enabled(){\n  static const bool on = (std::getenv("INTFER_KV_KVARN") != nullptr);\n  return on;\n}' $'static inline bool kv_kvarn_enabled(){\n  return false;\n}'
run_state "DC-20" $'static inline bool kv_q5_1_had_enabled(){\n  static const bool on = (std::getenv("INTFER_KV_Q5_1_HAD") != nullptr);\n  return on;\n}' $'static inline bool kv_q5_1_had_enabled(){\n  return false;\n}'
run_state "DC-21" $'static inline bool kv_q4_had_enabled(){\n  static const bool on = (std::getenv("INTFER_KV_Q4_HAD") != nullptr);\n  return on;\n}' $'static inline bool kv_q4_had_enabled(){\n  return false;\n}'

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

echo
restore
if ! cmp -s "$ART" "$PRISTINE"; then
  echo "  FAIL residue: $ART differs from its pre-matrix state" >&2
  exit 1
fi
echo "  ok residue: $ART byte-identical to its pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "KV TWIN SYMMETRY MUTATION MATRIX: PASS ($caught/$total states)"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "TWIN SYMMETRY MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
