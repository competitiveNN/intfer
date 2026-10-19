#!/usr/bin/env bash
# The two per-role candidate ladders must agree, and they are re-derived from
# COMPILED SOURCE, not parsed.
#
# WHY. default_candidates() lives in TWO files -- src/quant_spec.cpp (the one
# scripts/build_quant_real.sh compiles, i.e. the file that PRODUCED
# Qwen3.8-27B-int2-v2.intifer) and src/quant/quant_spec.cpp (its mirror). They
# cannot be linked together (duplicate symbols), so divergence between them is
# SILENT -- nothing in the build detects it. It has already gone wrong in both
# directions: originally the examples offered INT2/INT4/INT8 to LINEAR_ATTN_\*
# where the real file offered only FP16 (which is why the shipped 19 GB artifact
# has 56.8% of its bytes in FP16), and with the arms in place the mirror's
# ATTN_O/MLP_\* group carried an FP16 escape the real file deliberately withholds.
#
# WHY COMPILED AND NOT PARSED. The two default_candidates use opposite parameter
# orders for their mk() helper -- the real one is (sch, g, act, glob), the mirror
# (sch, g, glob, act) -- so a text comparison is not merely fragile, it is wrong
# by construction: pasting a call site between them writes an ActScheme into a
# bool. A grep/parse checker would have to re-implement both mk() spellings to
# be right, and a checker that is subtly wrong is worse than none. Instead both
# TUs are compiled against the shared header and the resulting candidate tables
# are dumped and diffed, which is ground truth by definition.
#
# WHAT IS CHECKED. For all 18 TensorRole values: the scheme ladder, the group
# sizes, AND the order (the XMX g32-first policy is ORDER: g32 is offered before
# g16 exactly so the accuracy search keeps g16 only where it earns 0.5 bpw, so a
# candidate list with the same members in a different order is a different
# allocation policy, not a formatting difference).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

# INTfer_LADDER_REAL points the REAL side at another file for the mutation matrix
# only (scripts/mutate_ladder_sync.sh): a negative direction must be run against
# a COPY of the source, because editing the tracked file in place to prove a
# checker fails is the destructive-probe shape this repo's standing rule forbids.
# It defaults to the shipped file, so a plain run is always the real tree.
REAL="${INTFER_LADDER_REAL:-src/quant_spec.cpp}"
MIRR="src/quant/quant_spec.cpp"
HDR="include/intfer/quant_spec.h"
echo "== CANDIDATE-LADDER SYNC ($REAL vs $MIRR) =="

for f in "$REAL" "$MIRR" "$HDR"; do
  [ -f "$f" ] || { echo "FAIL: $f missing" >&2; exit 2; }
done

# The two ladders cannot be linked into one binary (duplicate symbols), but they
# can each be linked into their OWN one TU probe. A temp dir keeps the tree clean
# and makes the checker's own failure mode visible: if the probe does not run,
# every verdict below would be vacuous.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/probe.cpp" <<'EOF'
#include "intfer/quant_spec.h"
#include <cstdio>
static const char* S(intfer::QuantScheme s){
  switch(s){case intfer::QuantScheme::INT2:return "INT2";
            case intfer::QuantScheme::INT4:return "INT4";
            case intfer::QuantScheme::INT8:return "INT8";
            default:return "FP16";}}
static const char* A(intfer::ActScheme a){
  return a==intfer::ActScheme::FP16?"fp16":"int8a";}
int main(){
  for(int r=0;r<=(int)intfer::TensorRole::OTHER;r++){
    auto v=intfer::default_candidates((intfer::TensorRole)r);
    printf("%02d ",r);
    for(size_t i=0;i<v.size();i++)
      printf("%s%s%d/%s",i?"|":"",S(v[i].scheme),v[i].group_size,A(v[i].act));
    printf("\n");
  }
}
EOF

build_one() { # <src> <out-binary> <label>
  local src=$1 bin=$2 label=$3 out rc path
  case "$src" in
    /*) path="$src" ;;
    *)  path="$ROOT/$src" ;;
  esac
  out="$(g++ -std=c++17 -O0 -I"$ROOT/include" "$TMP/probe.cpp" "$path" -o "$bin" 2>&1)"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAIL  $label ($src) did not compile -- the ladder cannot be read" >&2
    printf '%s\n' "$out" | sed 's/^/        /' >&2
    exit 1
  fi
}

build_one "$REAL" "$TMP/p_real" "the REAL ladder (compiled by build_quant_real.sh)"
build_one "$MIRR" "$TMP/p_mir"  "the MIRROR ladder"

"$TMP/p_real" > "$TMP/real.txt" || { echo "FAIL: real probe exited non-zero" >&2; exit 1; }
"$TMP/p_mir"  > "$TMP/mir.txt"  || { echo "FAIL: mirror probe exited non-zero" >&2; exit 1; }

n_real="$(wc -l < "$TMP/real.txt")"
n_mir="$(wc -l < "$TMP/mir.txt")"
echo "-- roles harvested: real=$n_real mirror=$n_mir (TensorRole::OTHER = 17, so 18 expected)"

# Harvest guard. "0 roles compared" is not the same claim as "the ladders agree",
# and an empty diff means exactly nothing -- this is the vacuity defect every
# checker in this repo is required to refuse.
if [ "$n_real" != 18 ] || [ "$n_mir" != 18 ]; then
  echo "  FAIL  did not harvest 18 roles from each ladder (real=$n_real," >&2
  echo "        mirror=$n_mir) -- the dump is broken, so agreement below" >&2
  echo "        would be vacuous. Did TensorRole gain a member?" >&2
  exit 1
fi
echo "  ok    both probes harvested the full 18-role TensorRole enum"

# Static pre-conditions that must hold whatever the diff says. LINEAR_ATTN_QKV,
# _Z and _OUT are 56.8% of the shipped model's bytes and they earned their arms
# precisely because FP16 was their ONLY candidate; an FP16 rung reappearing in
# either ladder re-opens that exact defect.
for f in "$TMP/real.txt" "$TMP/mir.txt"; do
  for r in 11 12 16; do  # LINEAR_ATTN_QKV, _Z, _OUT
    if grep -q "^$r .*FP16" "$f"; then
      echo "  FAIL  role $r carries an FP16 rung in $f -- that is the" >&2
      echo "        defect the ladder arms exist to remove (56.8% of the" >&2
      echo "        model's bytes left un-quantized)" >&2
      exit 1
    fi
  done
done
echo "  ok    LINEAR_ATTN_QKV/_Z/_OUT (roles 11,12,16) carry no FP16 rung in either ladder"

# The XMX group-size policy: g32 offered before g16 for the three byte-carrying
# roles. g16-first would double the per-group scale traffic inside the XMX
# dequant for no accuracy reason, and reordering it is a silent policy change.
for f in "$TMP/real.txt" "$TMP/mir.txt"; do
  for r in 11 12 16; do
    first="$(grep "^$r " "$f" | cut -d'|' -f1)"   # e.g. "11 INT232/int8a"
    grp="${first##*INT2}"                          # e.g. "32/int8a"
    case "${grp%%/*}" in
      32) ;;
      *)  echo "  FAIL  role $r does not offer INT2 g32 FIRST ($f: $first)" >&2
          echo "        -- g16 there doubles the per-group scale traffic inside the" >&2
          echo "        XMX BK=32 dequant for no accuracy reason" >&2
          exit 1 ;;
    esac
  done
done
echo "  ok    g32 precedes g16 on roles 11,12,16 (XMX BK=32-aligned scale reuse)"

# ---- the actual comparison ------------------------------------------------
if diff -u "$TMP/real.txt" "$TMP/mir.txt" > "$TMP/diff.txt"; then
  echo "  PASS  18/18 candidate ladders IDENTICAL (scheme, group, order, act scheme)"
  echo
  echo "LADDER SYNC: PASS (18/18 roles agree; compiled, not parsed)"
  echo "RESULT: the real and mirror default_candidates offer the same allocation policy"
  exit 0
fi

echo "  FAIL  the two ladders DIVERGE -- the mirror is a silent second opinion" >&2
echo "        and nothing in the build detects it (duplicate symbols)." >&2
cat "$TMP/diff.txt" | sed 's/^/        /' >&2
echo
echo "LADDER SYNC: FAIL" >&2
echo "RESULT: FAIL -- the real and mirrored candidate ladders disagree" >&2
exit 1
