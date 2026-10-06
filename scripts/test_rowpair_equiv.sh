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
# FOUR ORDERS, ONE PROCESS EACH. quant_gemm_forward_usm reads each arm's flag into a
# function-local `static const bool`, so a flag is set ONCE PER PROCESS; one binary
# cannot hold two orders. Hence four invocations and three diffs.
#
# The two extra orders are INTFER_DECODE1_VERBHALF and INTFER_DECODE1_VERBMP, and
# they are here for a REASON beyond symmetry: VERBMP's "bit-exact BY CONSTRUCTION"
# is a CLAIM, and the ROWPAIR precedent is that a bit-exactness argument is worth
# nothing until measured -- ROWPAIR's is exactly what shipped a device-losing
# double offset. This settles it without a model and without a 27B load.
run_order() {  # $1 = order tag
  local tag="$1" f="$2"
  env -u INTFER_DECODE1_ROWPAIR -u INTFER_DECODE1_VERBHALF -u INTFER_DECODE1_VERBMP \
    DECODE1_EQUIV_ORDER="$tag" \
    "$SCRIPT_DIR/with_oneapi.sh" "$BIN" >"$f" \
    2>"$BUILD_DIR/rowpair_equiv.$tag.err"
  return $?
}

B_RC=0; run_order base "$BASE" || B_RC=$?
R_RC=0; run_order rp   "$RP"   || R_RC=$?
# *** THE ARM TABLE IS DERIVED, NOT TYPED TWICE. *** Adding an order to the binary
# and forgetting it here would leave an arm silently uncertified, which is how
# INTFER_DECODE1_SUBGROUP ended up red in the first place -- so the names live in
# one place and the case arm loop reads them.
# ONE SPELLING. The binary's order table uses the short tags below; an earlier draft
# of this gate passed the long names (verbhalf, verbdead, ...) and every new order
# failed with "unknown DECODE1_EQUIV_ORDER", which is exactly the "added to one place,
# forgotten in the other" failure this table exists to prevent. The long names are the
# FLAG names, not the ORDER names; keep the two vocabularies distinct.
ARMS="rp vh vm vd vs vv"
ORDERS="base $ARMS"

rcs=""
for t in $ORDERS; do
  f="$BUILD_DIR/rowpair_equiv.$t.txt"
  case "$t" in
    base) f="$BASE" ;;
    rp)   f="$RP" ;;
    vh)   f="$VH" ;;
    vm)   f="$VM" ;;
    vd)   f="$VD" ;;
    vs)   f="$VS" ;;
    vv)   f="$VV" ;;
  esac
  r=0; run_order "$t" "$f" || r=$?
  # *** THE ARM MUST HAVE ACTUALLY FIRED, AND *ONLY* IT. *** A killed arm emits the
  # base kernel's output, so a base-vs-base diff reads as a clean pass. That guard was
  # added and immediately killed a whole turn's headline.
  #
  # It is now STRICTER: the order's OWN marker must be present AND NO OTHER arm's
  # marker may be. Presence alone was fooled too -- `[decode1-rowpair]` appeared for
  # every non-base order while `vh`/`vm` were actually the arms running, because
  # each arm's announcement was a `static const bool` latch initialised once per
  # PROCESS on the first kernel call. That latch is gone (one keyed
  # `decode1_announce(tag)` in quant_gemm.cpp, per-ARM by construction and carrying
  # a call counter), and this is the always-red twin that pins it: a wrong marker is
  # a FAILURE, and FP16/INT4 orders stay silent because their branches are guarded
  # on `bits == 2` and never print.
  if [ "$t" != base ]; then
    case "$t" in
      rp) want=decode1-rowpair ;; vh) want=decode1-verbhalf ;;
      vm) want=decode1-verbmp ;;  vd) want=decode1-verbdead ;;
      vs) want=decode1-verbsel ;; vv) want=decode1-verbvec ;;
    esac
    errf="$BUILD_DIR/rowpair_equiv.$t.err"
    if ! grep -aq "\[$want\]" "$errf" 2>/dev/null; then
      echo "FAIL: order $t did not announce ITS OWN marker [$want]." >&2
      echo "      present instead:" >&2
      grep -aoE '\[decode1-[a-z-]+\]' "$errf" 2>/dev/null | sort -u | sed 's/^/        /' >&2
      r=1
    fi
    others="$(grep -aoE '\[decode1-[a-z-]+\]' "$errf" 2>/dev/null \
              | sort -u | grep -v "^\[$want\]$" || true)"
    if [ -n "$others" ]; then
      echo "FAIL: order $t ALSO announced another arm: $others" >&2
      echo "      One arm's marker must never stand in for another's -- that is" >&2
      echo "      exactly how a wrong arm certified itself before." >&2
      r=1
    fi
  fi
done
for t in $ORDERS; do
  case "$rcs" in *"$t=0") ;; *)
    echo "FAIL: decode1 equiv order $t did not complete ($rcs)" >&2
    for u in $ORDERS; do
      echo "--- $u stderr ---" >&2
      tail -8 "$BUILD_DIR/rowpair_equiv.$u.err" 2>/dev/null >&2
    done
    exit 1 ;;
  esac
done

# *** THE ASSERTION IS A DIFF, AND A DIFF IS NOT ENOUGH ON ITS OWN. *** A diff of
# two empty files is a PASS, and "the probe printed nothing" and "the probe agreed"
# look identical to `cmp`. So the HARVEST is asserted first: both arms must have
# emitted a non-trivial number of per-element lines, and the two must cover the
# same case set. That is the empty-stream vacuity the repo's bench harness has its
# own checker for, in miniature.
VH="$BUILD_DIR/rowpair_equiv.verbhalf.txt"
VM="$BUILD_DIR/rowpair_equiv.verbmp.txt"
VD="$BUILD_DIR/rowpair_equiv.verbdead.txt"
VS="$BUILD_DIR/rowpair_equiv.verbsel.txt"
VV="$BUILD_DIR/rowpair_equiv.verbvec.txt"
n_base="$(grep -c '|0x' "$BASE" || true)"
for pair in "rowpair:$RP" "verbhalf:$VH" "verbmp:$VM" \
            "verbdead:$VD" "verbsel:$VS" "verbvec:$VV"; do
  lbl="${pair%%:*}"; f="${pair#*:}"
  n="$(grep -c '|0x' "$f" || true)"
  if [ "${n_base:-0}" -lt 1000 ] || [ "${n:-0}" -lt 1000 ]; then
    echo "FAIL: the harvest is too small to conclude anything" >&2
    echo "      base=$n_base $lbl=$n element lines (need >= 1000 each)." >&2
    echo "      An empty payload compares EQUAL to any other empty payload, so a" >&2
    echo "      diff alone would certify a probe that measured nothing." >&2
    exit 1
  fi
  if [ "$n_base" -ne "$n" ]; then
    echo "FAIL: base and $lbl emitted different case counts ($n_base vs $n)" >&2
    exit 1
  fi
done

# Strip the order marker and compare VALUES only. The order tag differs by design,
# so diffing the raw text would always differ.
sed 's/|base$//' "$BASE" >"$BASE.norm"

# One comparison function, three callers. The EXPECTATION DIFFERS PER ARM and that
# difference is the point:
#   rowpair   was claimed bit-exact  -> must be IDENTICAL, or the claim is wrong.
#   verbmp    is claimed bit-exact  -> must be IDENTICAL, same reason.
#   verbhalf  is claimed NOT exact  -> may differ; it is STREAM-IDENTITY-ONLY on the
#             real model, and the question here is whether the denormal/FTZ concern
#             is REAL on these datasets or merely theoretical. A difference is
#             information, not a failure, and is reported as such.
check_identical() {  # $1 = label, $2 = file
  local lbl="$1" f="$2"
  sed 's/|rp$//; s/|vh$//; s/|vm$//; s/|vd$//; s/|vs$//; s/|vv$//' "$f" >"$f.norm"
  if cmp -s "$BASE.norm" "$f.norm"; then
    echo "  ok    $lbl is BITWISE IDENTICAL to base ($n_base element values)"
  else
    echo "FAIL: $lbl is NOT bitwise identical to base -- and it was CLAIMED to be." >&2
    echo "      (A bit-exactness ARGUMENT is what shipped the ROWPAIR double" >&2
    echo "       offset and lost the device. This is the measurement saying the" >&2
    echo "       argument is wrong.)" >&2
    echo "      first differing lines:" >&2
    diff "$BASE.norm" "$f.norm" | head -20 >&2
    return 1
  fi
}
rc=0
# Every arm whose claim is BIT-EXACT must be IDENTICAL, or the claim is wrong and
# the gate says so. Round 2's three are all in this class by construction.
check_identical "INTFER_DECODE1_ROWPAIR " "$RP" || rc=1
check_identical "INTFER_DECODE1_VERBMP  " "$VM" || rc=1
check_identical "INTFER_DECODE1_VERBDEAD" "$VD" || rc=1
check_identical "INTFER_DECODE1_VERBSEL " "$VS" || rc=1
check_identical "INTFER_DECODE1_VERBVEC " "$VV" || rc=1
# VERBHALF: report, do not gate.
sed 's/|verbhalf$//' "$VH" >"$VH.norm"
if cmp -s "$BASE.norm" "$VH.norm"; then
  echo "  ok    INTFER_DECODE1_VERBHALF is bitwise identical to base on THESE datasets"
  echo "        -- which does NOT prove the FTZ/denormal concern away, only that"
  echo "        these synthetic scales never reach it."
else
  nd="$(diff "$BASE.norm" "$VH.norm" | grep -c '^<' || true)"
  echo "  ..   INTFER_DECODE1_VERBHALF DIFFERS from base in ~${nd} of ${n_base}"
  echo "        values. EXPECTED IN PRINCIPLE and not a gate failure: this arm is"
  echo "        claimed NOT bit-exact (denormal + default FTZ). It is reported so"
  echo "        the magnitude is on the record rather than assumed."
fi
[ "$rc" -eq 0 ] || { [ "$KEEP" -eq 1 ] || rm -f "$BASE" "$RP" "$VH" "$VM" "$BASE.norm"; exit 1; }

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
echo "DECODE1-EQUIV TEST: PASS"
echo "RESULT: on $n_base element values over $n_runs (shape,dataset) runs,"
echo "        $n_shapes distinct shapes x $n_ds distinct datasets:"
echo "          INTFER_DECODE1_ROWPAIR  BITWISE IDENTICAL  -- claim MEASURED"
echo "          INTFER_DECODE1_VERBMP   BITWISE IDENTICAL  -- claim MEASURED"
echo "          INTFER_DECODE1_VERBDEAD BITWISE IDENTICAL  -- pure code motion"
echo "          INTFER_DECODE1_VERBSEL  BITWISE IDENTICAL  -- codebook by select"
echo "          INTFER_DECODE1_VERBVEC  BITWISE IDENTICAL  -- activation ptr type"
echo "        INTFER_DECODE1_VERBHALF reported above: it is claimed NOT bit-exact"
echo "        (denormal + default FTZ), so a difference is information, not a failure."
echo "        PERFORMANCE IS STILL UNMEASURED for all three; see"
echo "        reports/2026-10-06-decode1-gemv-bisect.md."
exit 0
