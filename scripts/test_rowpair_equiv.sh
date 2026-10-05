#!/usr/bin/env bash
# Gate for examples/rowpair_equiv -- the MEASURED bitwise-equivalence check for
# INTFER_DECODE1_ROWPAIR.
#
# WHY A GATE AND NOT A COMMENT. Row-pairing's claim was "bit-exact BY
# CONSTRUCTION: each output keeps its own accumulator and its own ascending
# order, and the two chains are interleaved in the instruction stream, never
# summed together." That is a READING argument, and this repo has paid for
# exactly that class twice, both times on this very kernel:
#
#   * INTFER_DECODE1_SPLITK shipped `(bv >> (j * 2)) & 3` instead of
#     `(bv >> ((j & 3) * 2)) & 3`, and was localised only because the stream
#     diverged while the kill switch stayed clean -- the argument had been "the
#     order is safe".
#   * The reverted fused-decode kernel was built on a normalized statement diff
#     that said the arithmetic was identical. The diff was right, and it cost
#     231 114 ms. AGENTS.md's generalised lesson: structural argument is not
#     measurement.
#
# Row-pairing shares ONE activation load across TWO code pointers, which is
# precisely the lane where an index that assumes a single pointer goes wrong. So
# the claim is worth nothing until measured, and this measures it: bitwise, not to
# a tolerance, because the claim is exactness.
#
# TWO PROCESSES, NOT ONE. quant_gemm_forward_usm reads the flag into a
# function-local `static const bool` (the same pattern every engine flag uses; a
# per-call getenv in a decode kernel would be a hot-path syscall). A `static` is
# set once per PROCESS, so a single process cannot run both orders. The gate runs
# the binary twice and diffs the two payloads.
#
# NO MODEL: this binds none, per AGENTS.md's standing rule 1.
#
# Usage: scripts/test_rowpair_equiv.sh [--keep]
# Exit:  0 = row-pair output is BITWISE IDENTICAL to base on every shape and
#           dataset; 1 = a difference, or a case that produced no output.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/rowpair_equiv/rowpair_equiv"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build rowpair_equiv || exit 1
[ -x "$BIN" ] || { echo "FAIL: rowpair_equiv binary missing: $BIN" >&2; exit 1; }

BASE="$BUILD_DIR/rowpair_equiv.base.txt"
RP="$BUILD_DIR/rowpair_equiv.rp.txt"

# The flag is a presence-only getenv, so OFF is ABSENCE, not `=0`. See the bench
# harness's note: `INTFER_X=0` still ENABLES an X read with getenv() != nullptr,
# so the only way to express OFF is for the variable to be unset. `env -u` on
# EVERY arm, for the same reason.
env -u INTFER_DECODE1_ROWPAIR ROWPAIR_EQUIV_ORDER=base \
    "$SCRIPT_DIR/with_oneapi.sh" "$BIN" >"$BASE" 2>"$BUILD_DIR/rowpair_equiv.base.err"
B_RC=$?
env INTFER_DECODE1_ROWPAIR=1 ROWPAIR_EQUIV_ORDER=rp \
    "$SCRIPT_DIR/with_oneapi.sh" "$BIN" >"$RP" 2>"$BUILD_DIR/rowpair_equiv.rp.err"
R_RC=$?

if [ "$B_RC" -ne 0 ] || [ "$R_RC" -ne 0 ]; then
  echo "FAIL: rowpair_equiv base rc=$B_RC rowpair rc=$R_RC" >&2
  echo "--- base stderr ---" >&2; tail -15 "$BUILD_DIR/rowpair_equiv.base.err" >&2
  echo "--- rowpair stderr ---" >&2; tail -15 "$BUILD_DIR/rowpair_equiv.rp.err" >&2
  exit 1
fi

# *** THE ASSERTION IS A DIFF, AND A DIFF IS NOT ENOUGH ON ITS OWN. *** A diff of
# two empty files is a PASS, and "the probe printed nothing" and "the probe agreed"
# look identical to `cmp`. So the HARVEST is asserted first: both arms must have
# emitted a non-trivial number of per-element lines, and the two must cover the
# same case set. That is the empty-stream vacuity the repo's bench harness has its
# own checker for, in miniature.
n_base="$(grep -c '|0x' "$BASE" || true)"
n_rp="$(grep -c '|0x' "$RP" || true)"
if [ "${n_base:-0}" -lt 1000 ] || [ "${n_rp:-0}" -lt 1000 ]; then
  echo "FAIL: the harvest is too small to conclude anything" >&2
  echo "      base=$n_base rowpair=$n_rp element lines (need >= 1000 each)." >&2
  echo "      An empty payload compares EQUAL to any other empty payload, so a" >&2
  echo "      diff alone would certify a probe that measured nothing." >&2
  exit 1
fi
if [ "$n_base" -ne "$n_rp" ]; then
  echo "FAIL: the two arms emitted different case counts (base=$n_base rowpair=$n_rp)" >&2
  exit 1
fi

# Strip the order marker and compare VALUES only. The order tag differs by design,
# so diffing the raw text would always differ.
sed 's/|rowpair$//' "$RP" >"$RP.norm"
sed 's/|base$//'   "$BASE" >"$BASE.norm"

if ! cmp -s "$BASE.norm" "$RP.norm"; then
  echo "FAIL: INTFER_DECODE1_ROWPAIR is NOT bitwise identical to the base kernel." >&2
  echo "      first differing lines:" >&2
  diff "$BASE.norm" "$RP.norm" | head -20 >&2
  echo "      (The construction argument predicted exact equality. It is wrong, and" >&2
  echo "       this is the measurement that says so.)" >&2
  [ "$KEEP" -eq 1 ] || rm -f "$BASE" "$RP" "$BASE.norm" "$RP.norm"
  exit 1
fi

# Per-dataset coverage, so a green verdict says WHICH cases passed rather than only
# that they agreed. Derived from the payload, not from a remembered list.
# The header line of each case is "<shape>|<scheme>|<d>|<dataset>|base"; the
# per-element lines carry an index too, so both counters must anchor on the
# header form specifically. Counting element lines gave "40 shapes" for 8.
n_runs="$(grep -cE '^[^|]+\|[A-Z0-9]+\|[0-9]+\|[a-z-]+\|base$' "$BASE" || true)"
n_shapes="$(grep -oE '^[^|]+\|' "$BASE" | sort -u | grep -c . || true)"
n_ds="$(grep -oE '\|[a-z-]+\|base$' "$BASE" | sort -u | grep -c . || true)"
echo "ROWPAIR-EQUIV: $n_base element values bitwise identical across:"
grep -oE '^(MLP_GATE\(INT2\)|MLP_DOWN\(INT2\)|LA_QKV\(FP16\)|LA_OUT\(FP16\)|MLP_UP\(INT4\)|ATTN_K\(FP16\)|MLP_GATE-ODD\(INT2\))\|[A-Z0-9]+\|[0-9]+\|[a-z-]+' "$BASE" \
  | sort -u | awk -F'|' '{printf "  %-20s %-5s %s\n", $1, $2, $5}' | head -50
echo "  ... (the ODD-N shape must take the BASE path; it is included so a silent"
echo "   fallback to a different kernel is visible rather than assumed)"

[ "$KEEP" -eq 1 ] || rm -f "$BASE" "$RP" "$BASE.norm" "$RP.norm"
echo
echo "ROWPAIR-EQUIV TEST: PASS"
# DERIVED, never hardcoded: an earlier revision printed "5 datasets x 7 shapes"
# and went on saying 7 after a shape was added. That is the hand-written-total
# defect in its smallest form, and it was in the line that states the gate's claim.
echo "RESULT: INTFER_DECODE1_ROWPAIR is BITWISE IDENTICAL to the base M=1 kernel"
echo "        on $n_base element values over $n_runs (shape,dataset) runs:"
echo "        $n_shapes distinct shapes x $n_ds distinct datasets -- the claim is"
echo "        MEASURED, not argued from the construction."
exit 0
