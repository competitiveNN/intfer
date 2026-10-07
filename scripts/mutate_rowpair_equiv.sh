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
[ -f "$SRC" ] || { echo "MUTATE: no source: $SRC" >&2; exit 2; }

rm -rf "$SB"; mkdir -p "$LOG_DIR" || exit 2
cp "$SRC" "$SB/quant_gemm.pristine"

# *** THE REAL PLANTED DEFECT. *** The SHIPPED-INTP codebook-index bug from
# INTFER_DECODE1_SPLITK, transcribed onto a bit-exact arm's chunk loop:
#     (bv >> ((j & 3) * 2)) & 3     correct: &3 wraps j=4..7 to read the low bits
#     (bv >> (j * 2))        & 3     bug: j=4..7 shifts a uint8_t by 8-14 -> 0, so
#                                    codes 4..7 silently read zero.
# Same class that lost the device in the original SPLITK. The arm is claimed
# BIT-EXACT BY CONSTRUCTION, so the gate's value-diff is the right instrument.
#
# ANCHORING. The correct `(bv >> ((j & 3) * 2)) & 3` idiom is the engine's general
# INT2 dequant and appears at ~8 arm bodies (base, VERBDEAD, VERBSEL, VERBVEC, the
# base tail, ...). A bare text-match anchor is therefore ambiguous (the count-1
# guard matched 10 sites and the matrix refused itself before it ran a state).
# The plant is instead PINNED to ONE arm by walking forward from that arm's unique
# `decode1_announce("decode1-XX")` call -- exactly one per arm in the tree -- and
# mutating the FIRST code idiom after it. So the same defect is exercised on two
# arms the gate claims differently, and a refusal that fired for a NONCE-31 marker
# would be testing the announcement, not the arithmetic.

# Pre-flight: both announce anchors must exist exactly once. plant_codebook_bug
# checks the same thing at apply time, but fail FAST and name the arm here so a
# renumbered tree is a clear message, not a per-state crash.
need_announce() {  # $1 = arm tag, e.g. decode1-verbmp
  local n
  n="$(grep -c -F "decode1_announce(\"$1\")" "$SRC")"
  [ "$n" = 1 ] || {
    echo "MUTATE: announce($1) found ${n:-0} times, want 1 (tree moved?)" >&2; exit 2; }
}
need_announce "decode1-verbmp"
need_announce "decode1-verbsel"

plant_codebook_bug() {  # $1 = path, $2 = arm tag
  python3 - "$1" "$2" <<'PY'
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
# The idiom can legitimately recur inside the SAME arm (chunk loop + tail); only
# the FIRST one after the announce anchors the arm. replace(...,1) from position i.
s2 = s[:j] + s[j:].replace(old, new, 1)
if s2 == s:
    sys.stderr.write("plant: replace was a no-op\n"); sys.exit(1)
open(path, "w").write(s2)
PY
}

# setup_state copies ONLY the dirs the rowpair_equiv build consumes (no test/
# model weights, ~2.2 GB, and no build/ or .git). A prior whole-tree rsync
# OOM-killed this matrix; a bare copy produced "no build system in $dir". The
# gate itself configures+c makes under $BUILD_DIR, so it lives out its own build.
setup_state() {  # $1 = id, $2 = arm tag ("", "verbmp", "verbsel")
  # `${2-}` NOT `${2:-}`: an empty-string second arg is a LEGITIMATE value
  # ("no arm to plant on" = baseline), and `${2:-}` leaves the variable UNSET,
  # which under `set -u` makes the `[ -n "$tag" ]` guard below die with
  # "unbound variable" on the very first state. The default form exists to
  # distinguish "arg absent" from "arg empty"; here the arg is always passed.
  local id="$1" tag="${2-}"
  # NOTE the split: `local id="$1" mutroot="$SB/$id"` in ONE statement evaluates
  # every RHS in the enclosing scope FIRST, so `$id` there is unset and, under
  # `set -u`, dies with "unbound variable" on the very first state. Two
  # statements bind them in order.
  local mutroot="$SB/$id"
  rm -rf "$mutroot"; mkdir -p "$mutroot"
  mkdir -p "$mutroot/src" "$mutroot/include" "$mutroot/examples" "$mutroot/scripts"
  cp "$ROOT/CMakeLists.txt" "$mutroot/CMakeLists.txt"
  cp -r "$ROOT/src/."    "$mutroot/src/"
  cp -r "$ROOT/include/." "$mutroot/include/"
  cp -r "$ROOT/examples/." "$mutroot/examples/"
  cp -r "$ROOT/scripts/." "$mutroot/scripts/"
  cp "$SB/quant_gemm.pristine" "$mutroot/src/gemm/quant_gemm.cpp"
  if [ -n "$tag" ]; then
    plant_codebook_bug "$mutroot/src/gemm/quant_gemm.cpp" "$tag" || {
      echo "  MUTATE: plant failed for $id" >&2; return 1; }
  fi
  # _gate_build.sh's gate_require_build runs `cmake --build`, which needs a
  # CONFIGURED tree, not a bare one -- a copy without this line makes every
  # state fail "no build system in $dir" and the matrix certifies nothing.
  # One configure per state is the cost of exercising the REAL kernel end-to-end.
  #
  # COMPILER PATH. `icpx` is NOT on the default PATH (it lives under
  # /opt/intel/oneapi/compiler/<ver>/bin), so a bare `-DCMAKE_CXX_COMPILER=icpx`
  # fails with "is not a full path and was not found in the PATH" -- which is
  # exactly what the first run here produced, silently swallowed by >/dev/null.
  # scripts/build.sh resolves this by prepending the toolkit bin dir; do the same.
  local _tkbin _tklib
  _tkbin="$(ls -dv "${ONEAPI_ROOT:-/opt/intel/oneapi}"/compiler/[0-9]*/bin 2>/dev/null \
            | sort -V | tail -1)"
  _tklib="$(ls -dv "${ONEAPI_ROOT:-/opt/intel/oneapi}"/compiler/[0-9]*/lib 2>/dev/null \
            | sort -V | tail -1)"
  PATH="$_tkbin:$PATH" LD_LIBRARY_PATH="$_tklib:${LD_LIBRARY_PATH:-}" \
    cmake -S "$mutroot" -B "$mutroot/build" -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx -DINTFER_ENABLE_L0=OFF >/dev/null 2>&1 \
    || { echo "  MUTATE: configure failed for $id" >&2; return 1; }
}

run_gate() {  # $1 = id
  local id="$1"
  INTFER_BUILD_DIR="$SB/$id/build" timeout -k 60 1800 bash "$SB/$id/scripts/test_rowpair_equiv.sh" \
    >"$LOG_DIR/$id.log" 2>"$LOG_DIR/$id.err"
  return $?
}

total=0; caught=0
record() { total=$((total+1)); case "$2" in ok) caught=$((caught+1)); echo "  ok   $1";; *) echo "  $1 $2";; esac; }

echo "== baseline (pristine source) =="
setup_state B0 ""
run_gate B0; brc=$?
if [ "$brc" -eq 0 ] && grep -q 'DECODE1-EQUIV TEST: PASS' "$LOG_DIR/B0.log"; then
  record "baseline PASS" ok
else
  echo "  MISSED baseline did not pass (rc=$brc)" >&2; tail -8 "$LOG_DIR/B0.err" >&2
  record "baseline" "MISSED (rc=$brc)"; exit 1
fi

echo "== RPE-1: VERBMP codebook-index bug planted -> must REFUSE =="
# `record()` owns the increment; a manual `total=$((total+1))` here double-counts
# this state (total ends at 5 with only 3 states run, so the PASS gate is
# unreachable). The baseline block never had the duplicate, which is why it
# passed while the matrix reported FAIL (3/5).
setup_state RPE1 "decode1-verbmp"
run_gate RPE1; rc=$?
# BOTH predicates are read off the SAME stream the gate writes them to. The
# "NOT bitwise identical" verdict goes to STDERR (the gate's FAIL: line), and
# the arm's own marker goes to the per-order stderr file the gate opens. Reading
# the verdict off .log (stdout) and the marker off .err is the transpose of what
# the gate emits, and it reported a REFUSAL that had already happened as a MISS.
if [ "$rc" -ne 0 ] && grep -qE 'INTFER_DECODE1_VERBMP.*NOT bitwise identical' "$LOG_DIR/RPE1.err" \
  && grep -qa 'decode1-verbmp' "$SB/RPE1/build/rowpair_equiv.vm.err"; then
  record "RPE-1 REFUSED (INT2 diff + arm announced itself)" ok
else
  # DEBUG: the state tree and the gate's own stderr are the evidence for why
  # this state missed; print them before the verdict so a failure is
  # self-explaining rather than a bare MISSED.
  echo "  RPE-1 MISSED (rc=$rc): defect escaped the rewritten gate." >&2
  echo "  --- $LOG_DIR/RPE1.err ---" >&2
  [ -s "$LOG_DIR/RPE1.err" ] && tail -8 "$LOG_DIR/RPE1.err" >&2 \
    || echo "    (empty or absent)" >&2
  echo "  --- $LOG_DIR/RPE1.log ---" >&2
  [ -s "$LOG_DIR/RPE1.log" ] && tail -8 "$LOG_DIR/RPE1.log" >&2 \
    || echo "    (empty or absent)" >&2
  echo "  --- $SB/RPE1/build/rowpair_equiv.vm.err (the seeded arm's stderr) ---" >&2
  [ -s "$SB/RPE1/build/rowpair_equiv.vm.err" ] && tail -8 "$SB/RPE1/build/rowpair_equiv.vm.err" >&2 \
    || echo "    (empty or absent -- the arm never ran)" >&2
  record "RPE-1" "MISSED (rc=$rc)"
fi

echo "== RPE-2: the SAME defect planted on VERBSEL -> must ALSO REFUSE =="
setup_state RPE2 "decode1-verbsel"
run_gate RPE2; rc=$?
if [ "$rc" -ne 0 ] && grep -qE 'INTFER_DECODE1_VERBSEL.*NOT bitwise identical' "$LOG_DIR/RPE2.err" \
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
else
  echo "  FAIL $SRC differs from pristine" >&2; exit 1
fi
echo
if [ "$caught" -eq "$total" ]; then
  echo "ROWPAIR-EQUIV MUTATION MATRIX: PASS ($caught/$total)"
  # Clean up ONLY on success: a failing matrix that deletes its own evidence
  # leaves the operator with a FAIL verdict and no log to explain it, which is
  # the "remedy cannot help" shape this repo keeps auditing. On failure the
  # state trees under $SB stay for inspection.
  rm -rf "$SB"
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
