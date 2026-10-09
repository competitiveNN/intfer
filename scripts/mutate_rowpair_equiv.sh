#!/usr/bin/env bash
# Mutation matrix for scripts/test_rowpair_equiv.sh's REWRITE (2026-10-07).
#
# WHY A NEW MATRIX. On 2026-10-07 the gate was materially rewritten: the base order
# restores the reference path via the two kill switches (VERBHALF/VERBMP are now
# default-ON), a reference-stability guard was added, and every comparison is
# filtered to INT2 shapes because the FP16 decode kernel is nondeterministic on
# this box. A rewritten checker is a rewritten artifact, and AGENTS.md rule 11 says
# that is provable only by mutating the ARTIFACT and requiring the (rewritten)
# checker to REJECT it.
#
# RUN: scripts/mutate_rowpair_equiv.sh
# EXIT: 0 = every state behaved; 1 = a defect escaped or baseline is wrong.
set -u
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
. "$SCRIPT_DIR/_gate_patch.sh"

GATE="scripts/test_rowpair_equiv.sh"
SRC="src/gemm/quant_gemm.cpp"
BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/rowpair_equiv_matrix"
LOG_DIR="$SB/logs"
[ -f "$GATE" ] || { echo "MUTATE: no gate: $GATE" >&2; exit 1; }
[ -f "$SRC" ]   || { echo "MUTATE: no source: $SRC" >&2; exit 2; }

rm -rf "$SB"; mkdir -p "$LOG_DIR" || exit 2
cp "$SRC" "$SB/quant_gemm.pristine"

# *** THE REAL PLANTED DEFECT. *** The SHIPPED-INTP codebook-index bug from
# INTFER_DECODE1_SPLITK, transcribed onto the VERBMP arm's chunk loop:
#     (bv >> ((j & 3) * 2)) & 3     correct: &3 wraps j=4..7 to read the low bits
#     (bv >> (j * 2))        & 3     bug: j=4..7 shifts a uint8_t by 8-14 -> 0, so
#                                    codes 4..7 silently read zero.
# It is the same class that lost the device in the original SPLITK. The arm is
# claimed BIT-EXACT BY CONSTRUCTION, so the gate's value-diff is the right
# instrument -- and the arm MUST still announce itself, so a refusal that fired
# for a NONCE-31 marker would be testing the announcement, not the arithmetic.
PLANT_OLD='const int code = (bv >> ((j & 3) * 2)) & 3;
                      acc += av * wtab[code];'
PLANT_NEW='const int code = (bv >> (j * 2)) & 3;
                      acc += av * wtab[code];'
[ "$(grep -c -F "$PLANT_OLD" "$SRC")" = 1 ] || {
  echo "MUTATE: PLANT anchor not found exactly once (tree moved?)" >&2; exit 2; }

setup_state() { # $1 = id, $2 = mutation fn, $3 = old anchor, $4 = new text
  # Copy the WHOLE tree under build/, not just the bits the gate reads. The gate
  # sources _gate_build.sh, which runs cmake --build on $BUILD_DIR and needs the
  # CMakeLists, include/, examples/ and src/ together; a partial copy silently
  # produced "no build system in $dir" and every state read as a MISS.
  local id="$1" fn="$2" old="$3" new="$4" mutroot="$SB/$id"
  rm -rf "$mutroot"; mkdir -p "$mutroot"
  # Copy ONLY what the rowpair_equiv build needs, NOT the whole tree -- the `test/`
  # directory (27B model weights, ~2.2 GB) and the build dirs must not be copied,
  # and a whole-tree copy OOM-killed this matrix on first attempt.
  mkdir -p "$mutroot/src" "$mutroot/include" "$mutroot/examples" "$mutroot/scripts"
  cp "$ROOT/CMakeLists.txt" "$mutroot/CMakeLists.txt"
  cp -r "$ROOT/src/." "$mutroot/src/"
  cp -r "$ROOT/include/." "$mutroot/include/"
  cp -r "$ROOT/examples/." "$mutroot/examples/"
  cp -r "$ROOT/scripts/." "$mutroot/scripts/"
  cp "$SB/quant_gemm.pristine" "$mutroot/src/gemm/quant_gemm.cpp"
  "$fn" "$mutroot/src/gemm/quant_gemm.cpp" "$old" "$new"
  # gate_require_build runs `cmake --build`, which needs a CONFIGURED tree, not a
  # bare one -- configure it here or every state fails with "no build system in
  # $dir" and the matrix certifies nothing. One configure per state is the cost of
  # exercising the REAL kernel end-to-end.
  cmake -S "$mutroot" -B "$mutroot/build" -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx -DINTFER_ENABLE_L0=OFF >/dev/null 2>&1 \
    || { echo "  MUTATE: configure failed for $id" >&2; return 1; }
}
run_gate() { # $1 = id
  local id="$1"
  INTFER_BUILD_DIR="$SB/$id/build" timeout -k 60 1800 bash "$SB/$id/scripts/test_rowpair_equiv.sh" \
    >"$LOG_DIR/$id.log" 2>"$LOG_DIR/$id.err"
  return $?
}

# The real planted defect: the SHIPPED-INTP codebook-index bug from
# INTFER_DECODE1_SPLITK, transcribed onto the chunk loop of a bit-exact arm.
#     (bv >> ((j & 3) * 2)) & 3     correct: &3 wraps j=4..7 to read the low bits
#     (bv >> (j * 2))        & 3     bug: j=4..7 shifts a uint8_t by 8-14 -> 0,
#                                    so codes 4..7 silently read zero.
# Same class that lost the device in the original SPLITK. The engine reads this
# idiom at four arm bodies, so planting by TEXT MATCH would be ambiguous -- the
# plant is instead PINNED to a single arm by walking forward from that arm's
# `decode1_announce("decode1-XX")` call, which exists exactly once per arm. The
# two arms below are the bit-exact CLAIMED-BY-CONSTRUCTION ones: VERBMP (the
# default-ON arm planted here for RPE-1) and VERBSEL (the opt-in arm for RPE-2).
plant_codebook_bug() { python3 - "$1" "$2" <<'PY'
import sys
path, tag = sys.argv[1], sys.argv[2]
s = open(path).read()
anchor = 'decode1_announce("%s")' % tag
i = s.find(anchor)
if i < 0:
    sys.stderr.write("plant: no announce(%r)\n" % tag); sys.exit(1)
if s.count(anchor) != 1:
    sys.stderr.write("plant: announce(%r) not unique\n" % tag); sys.exit(1)
needle = "const int code = (bv >> ((j & 3) * 2)) & 3;"
j = s.find(needle, i)
if j < 0:
    sys.stderr.write("plant: no code-idiom after %r\n" % tag); sys.exit(1)
old = needle
new = "const int code = (bv >> (j * 2)) & 3;"
s2 = s[:j] + s[j:].replace(old, new, 1)
if s2 == s:
    sys.stderr.write("plant: replace was a no-op\n"); sys.exit(1)
open(path, "w").write(s2)
PY
}
# RPE-1 plants it on VERBMP (the default-ON arm); RPE-2 on VERBSEL (opt-in).
# The anchors differ only in the tail expression, so the same defect is
# exercised on two arms the gate claims differently.

total=0; caught=0
record() { total=$((total+1)); case "$2" in ok) caught=$((caught+1)); echo "  ok   $1";; *) echo "  $1 $2";; esac; }

echo "== baseline (pristine source) =="
setup_state B0 "true" "" ""
run_gate B0; brc=$?
if [ "$brc" -eq 0 ] && grep -q 'DECODE1-EQUIV TEST: PASS' "$LOG_DIR/B0.log"; then
  record "baseline PASS" ok
else
  echo "  MISSED baseline did not pass (rc=$brc)" >&2; tail -8 "$LOG_DIR/B0.err" >&2
  record "baseline" "MISSED (rc=$brc)"; exit 1
fi

echo "== RPE-1: VERBMP codebook-index bug planted -> must REFUSE =="
total=$((total+1))
setup_state RPE1 "plant_codebook_bug" "$RPE1_OLD" "$RPE1_NEW"
run_gate RPE1; rc=$?
if [ "$rc" -ne 0 ] && grep -qE 'INTFER_DECODE1_VERBMP.*NOT bitwise identical' "$LOG_DIR/RPE1.log" \
  && grep -qa 'decode1-verbmp' "$SB/RPE1/build/rowpair_equiv.vm.err"; then
  record "RPE-1 REFUSED (INT2 diff + arm announced itself)" ok
else
  echo "  RPE-1 MISSED (rc=$rc): defect escaped the rewritten gate." >&2
  tail -8 "$LOG_DIR/RPE1.err" >&2
  record "RPE-1" "MISSED (rc=$rc)"
fi

echo "== RPE-2: the SAME defect planted on VERBSEL -> must ALSO REFUSE =="
total=$((total+1))
setup_state RPE2 "plant_codebook_bug" "$RPE2_OLD" "$RPE2_NEW"
run_gate RPE2; rc=$?
if [ "$rc" -ne 0 ] && grep -qE 'INTFER_DECODE1_VERBSEL.*NOT bitwise identical' "$LOG_DIR/RPE2.log" \
  && grep -qa 'decode1-verbsel' "$SB/RPE2/build/rowpair_equiv.vs.err"; then
  record "RPE-2 REFUSED (same defect, different bit-exact arm -- not special-cased)" ok
else
  echo "  RPE-2 MISSED (rc=$rc)" >&2; tail -8 "$LOG_DIR/RPE2.err" >&2
  record "RPE-2" "MISSED (rc=$rc)"
fi

echo "== RPE-1: VERBMP codebook-index bug planted -> must REFUSE =="
total=$((total+1))
setup_state RPE1 "plant_codebook_bug" "$RPE1_OLD" "$RPE1_NEW"
run_gate RPE1; rc=$?
if [ "$rc" -ne 0 ] && grep -qE 'INTFER_DECODE1_VERBMP.*NOT bitwise identical' "$LOG_DIR/RPE1.log" \
  && grep -qa 'decode1-verbmp' "$SB/RPE1/build/rowpair_equiv.vm.err"; then
  record "RPE-1 REFUSED (INT2 diff + arm announced itself)" ok
else
  echo "  RPE-1 MISSED (rc=$rc): defect escaped the rewritten gate." >&2
  tail -8 "$LOG_DIR/RPE1.err" >&2
  record "RPE-1" "MISSED (rc=$rc)"
fi

echo "== RPE-2: the SAME defect planted on VERBSEL -> must ALSO REFUSE =="
total=$((total+1))
setup_state RPE2 "plant_codebook_bug" "$RPE2_OLD" "$RPE2_NEW"
run_gate RPE2; rc=$?
if [ "$rc" -ne 0 ] && grep -qE 'INTFER_DECODE1_VERBSEL.*NOT bitwise identical' "$LOG_DIR/RPE2.log" \
  && grep -qa 'decode1-verbsel' "$SB/RPE2/build/rowpair_equiv.vs.err"; then
  record "RPE-2 REFUSED (same defect, different bit-exact arm -- not special-cased)" ok
else
  echo "  RPE-2 MISSED (rc=$rc)" >&2; tail -8 "$LOG_DIR/RPE2.err" >&2
  record "RPE-2" "MISSED (rc=$rc)"
fi

echo
echo "== residue: tracked tree untouched =="
if diff -q "$SRC" "$SB/quant_gemm.pristine" >/dev/null; then
  echo "  ok   $SRC byte-identical to pre-matrix"
  rm -rf "$SB"
else
  echo "  FAIL $SRC differs from pristine" >&2; exit 1
fi
echo
if [ "$caught" -eq "$total" ]; then
  echo "ROWPAIR-EQUIV MUTATION MATRIX: PASS ($caught/$total)"
  echo "  baseline: pristine tree PASS"
  echo "  RPE-1: VERBMP codebook-index bug is caught (INT2 filter + arm announced)"
  echo "  RPE-2: the SAME defect in VERBSEL is also caught -- the checker is not"
  echo "         special-cased to one arm's name."
  echo "  Exclusion: a 'strip the INT2 filter, no defect -> green' floor is NOT run:"
  echo "  this box's FP16 path is nondeterministic, so a full-payload diff refuses for"
  echo "  a box reason -- exactly why the gate withholds FP16. An FP16-stable box is"
  echo "  the remedy, and none is available; the exclusion is recorded, not hidden."
  exit 0
fi
echo "ROWPAIR-EQUIV MUTATION MATRIX: FAIL ($caught/$total)"
exit 1
