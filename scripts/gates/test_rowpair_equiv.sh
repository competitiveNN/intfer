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
# the claim is worth nothing until measured, and this measures it: bitwise, not
# to a tolerance, because the claim is exactness.
#
# TWO PROCESSES, NOT ONE. quant_gemm_forward_usm reads the flag into a
# function-local `static const bool` (the same pattern every engine flag uses; a
# per-call getenv in a decode kernel would be a hot-path syscall). A `static` is
# set once per PROCESS, so a single process cannot run both orders. The gate runs
# the binary twice and diffs the two payloads.
#
# NO MODEL: this binds none, per AGENTS.md's standing rule 1.
#
# Usage: scripts/gates/test_rowpair_equiv.sh [--keep] [--self-check]
# Exit:  0 = row-pair output is BITWISE IDENTICAL to base on every shape and
#           dataset; 1 = a difference, or a case that produced no output.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/rowpair_equiv/rowpair_equiv"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

BASE="$BUILD_DIR/rowpair_equiv.base.txt"
RP="$BUILD_DIR/rowpair_equiv.rp.txt"

# THE DEFAULTS CHANGED 2026-10-07. ROWPAIR / VERBDEAD / VERBSEL / VERBVEC are
# still presence-only opt-ins (OFF is ABSENCE, not `=0`; see the bench harness's
# note that `INTFER_X=0` still ENABLES an X read with getenv() != nullptr, so the
# only way to express OFF is for the variable to be unset, hence `env -u` on every
# arm). VERBHALF and VERBMP are now DEFAULT-ON kill switches: the feature runs
# unless `INTFER_NO_DECODE1_VERBHALF=1` / `=1` is set. So `base` is no longer "the
# state with nothing set" -- with nothing set it would run the two default-ON arms
# and its payload would be the thing under test, not the reference. The BINARY
# restores the reference for the base order by setting both kill switches; the gate
# does not have to, and must not (it would be a no-op on the two opt-in arms).
# ONE ORDER PER PROCESS. quant_gemm_forward_usm reads each arm's flag into a
# function-local `static const bool`, so a flag is set ONCE PER PROCESS; one binary
# cannot hold two orders. Hence one invocation per order (base plus every
# arm) and one diff per arm.
#
# The two extra orders are INTFER_DECODE1_VERBHALF and INTFER_DECODE1_VERBMP, and
# they are here for a REASON beyond symmetry: VERBMP's "bit-exact BY CONSTRUCTION"
# is a CLAIM, and the ROWPAIR precedent is that a bit-exactness argument is worth
# nothing until measured -- ROWPAIR's is exactly what shipped a device-losing
# double offset. This settles it without a model and without a 27B load.
# VERBHALF is the same class as VERBMP but is claimed NOT bit-exact (denormal +
# default FTZ), so it is REPORTED, not gated: a difference is information.
# ---- THE MARKER PREDICATE, AS A FUNCTION, so it can be twin-tested -------------
# Rule 6: a check needs a case where the thing MUST appear AND a case where it must
# stay quiet, or "0 findings" and "found nothing" are the same claim. Both directions
# are exercised below against SYNTHETIC logs, in --self-check, with no GPU and no
# launch -- because the real logs are only obtainable on a healthy two-GPU box, and a
# check that can only be verified there is a check nobody verifies.
marker_for() {  # $1 = order tag -> the marker that order MUST print
  case "$1" in
    rp) echo decode1-rowpair ;; vh) echo decode1-verbhalf ;;
    vm) echo decode1-verbmp ;;  vd) echo decode1-verbdead ;;
    vs) echo decode1-verbsel ;; vv) echo decode1-verbvec ;;
    wv) echo decode1-wvec ;;
    *)   echo "" ;;
  esac
}
# marker_verdict <order> <errfile> -> 0 ok, 1 must-fail.
# The QUIET direction: base prints NO [decode1-*] marker at all, because no arm is
# live. If base ever prints one, an arm ran when none was selected -- which is the
# vacuous-pass shape this gate exists to refuse.
marker_verdict() {
  local t="$1" errf="$2" want others
  want="$(marker_for "$t")"
  others="$(grep -aoE '\[decode1-[a-z-]+\]' "$errf" 2>/dev/null \
            | sort -u | grep -v '^\[decode1-bisect\]$' || true)"
  if [ "$t" = base ]; then
    if [ -n "$others" ]; then
      echo "    base printed a marker but no arm was selected: $others" >&2
      return 1
    fi
    return 0
  fi
  if ! grep -aq "\[$want\]" "$errf" 2>/dev/null; then
    echo "    order $t did not announce ITS OWN marker [$want]; present instead:" >&2
    printf '%s\n' "$others" | sed 's/^/        /' >&2
    return 1
  fi
  local foreign
  foreign="$(printf '%s\n' "$others" | grep -v "^\[$want\]$" || true)"
  if [ -n "$foreign" ]; then
    echo "    order $t ALSO announced another arm: $foreign" >&2
    echo "    one arm's marker must never stand in for another's" >&2
    return 1
  fi
  return 0
}

# ---- SELF-CHECK, BEFORE any build or GPU requirement -----------------------
# The real stderr logs are only obtainable on a healthy two-GPU box, so the
# predicate itself is certified here against synthetic logs: one case where the
# marker MUST appear, one where it MUST stay quiet, three where a wrong marker
# MUST be refused. No GPU, no launch, no binary.
if [ "${1:-}" = "--self-check" ]; then
  SC="$(mktemp -d "${TMPDIR:-/dev/shm}/rpequiv.XXXXXX")"
  trap 'rm -rf "$SC"' EXIT
  trap 'rm -rf "$SC"; exit 130' INT
  trap 'rm -rf "$SC"; exit 143' TERM
  bad=0
  # FIXTURES. Each is a COMPLETE, correctly-shaped stderr; they differ in EXACTLY
  # ONE thing, because a fixture that differs in several cannot isolate its defect.
  printf 'x\n[decode1-rowpair] 2 rows/work-item ACTIVE\ny\n' >"$SC/rp.good"
  printf 'x\ny\n'                                              >"$SC/base.quiet"
  printf 'x\n[decode1-verbhalf] scale via bit_cast ACTIVE\ny\n' >"$SC/vh.good"
  printf 'x\n[decode1-rowpair] 2 rows/work-item ACTIVE\ny\n'    >"$SC/vh.wrong"
  printf 'x\n[decode1-verbhalf] a\n[decode1-rowpair] b\ny\n'    >"$SC/vh.two"
  printf 'x\ny\n'                                              >"$SC/rp.silent"
  printf 'x\n[decode1-wvec] weight bytes via vec load ACTIVE\ny\n' >"$SC/wv.good"
  printf 'x\n[decode1-verbhalf] scale via bit_cast ACTIVE\ny\n' >"$SC/wv.wrong"
  # tc <expected-rc> <label> <order> <fixture>
  tc() { local want_rc=$1 lbl=$2 ord=$3 f=$4 rc=0
         marker_verdict "$ord" "$f" >/dev/null 2>&1 || rc=1
         if [ "$rc" -eq "$want_rc" ]; then echo "  ok   $lbl"
         else echo "  SELF-CHECK FAIL: $lbl" >&2; bad=1; fi; }
  tc 0 "rp + its own marker                          -> ACCEPTED" rp    "$SC/rp.good"
  tc 0 "base + NO marker (the QUIET direction)       -> ACCEPTED" base  "$SC/base.quiet"
  tc 0 "vh + its own marker                          -> ACCEPTED" vh    "$SC/vh.good"
  tc 1 "vh carrying ROWPAIR's marker instead         -> REFUSED"  vh    "$SC/vh.wrong"
  tc 1 "vh carrying TWO markers                      -> REFUSED"  vh    "$SC/vh.two"
  tc 1 "rp silent (the MUST-appear direction)        -> REFUSED"  rp    "$SC/rp.silent"
  tc 0 "wv + its own marker                          -> ACCEPTED" wv    "$SC/wv.good"
  tc 1 "wv carrying VERBHALF's marker instead        -> REFUSED"  wv    "$SC/wv.wrong"
  if [ "$bad" -ne 0 ]; then echo "DECODE1-EQUIV SELF-CHECK: FAIL" >&2; exit 1; fi
  echo "DECODE1-EQUIV SELF-CHECK: PASS (2 must-appear, 1 must-stay-quiet,"
  echo "  4 must-be-refused -- synthetic logs, no GPU, no launch)"
  exit 0
fi

. "$SCRIPT_DIR/../lib/_gate_build.sh"
gate_require_build rowpair_equiv || exit 1
[ -x "$BIN" ] || { echo "FAIL: rowpair_equiv binary missing: $BIN" >&2; exit 1; }

run_order() {  # $1 = order tag
  local tag="$1" f="$2"
  # *** WHICH ARMS ARE UNSET HERE, AND WHY IT CHANGED. *** Until 2026-10-07 every
  # arm was default-OFF, so unsetting every arm's enable var left the reference
  # path running by default -- correct, and the base order needed no env at all.
  # VERBHALF and VERBMP are now default-ON kill switches, so unsetting their
  # enable vars is a NO-OP (they have none) and the reference still runs the
  # default path. Only the five remaining opt-in arms (ROWPAIR / VERBDEAD /
  # VERBSEL / VERBVEC / WVEC) are unset here, which is all that is needed to
  # keep the reference free of them. (The binary also unsets and kills every
  # arm in its own order table, so this is belt-and-braces against a stray
  # INTFER_DECODE1_*=1 inherited from the operator's shell.)
  env -u INTFER_DECODE1_ROWPAIR -u INTFER_DECODE1_VERBDEAD \
    -u INTFER_DECODE1_VERBSEL -u INTFER_DECODE1_VERBVEC \
    -u INTFER_DECODE1_WVEC \
    DECODE1_EQUIV_ORDER="$tag" \
    "\"$SCRIPT_DIR/../with_oneapi.sh" "$BIN" >"$f" \
    2>"$BUILD_DIR/rowpair_equiv.$tag.err"
  return $?
}

# *** THE ARM TABLE IS DERIVED, NOT TYPED TWICE. *** Adding an order to the binary
# and forgetting it here would leave an arm silently uncertified, which is how
# INTFER_DECODE1_SUBGROUP ended up red in the first place -- so the names live in
# one place and the case arm loop reads them.
# ONE SPELLING. The binary's order table uses the short tags below; an earlier draft
# of this gate passed the long names (verbhalf, verbdead, ...) and every new order
# failed with "unknown DECODE1_EQUIV_ORDER", which is exactly the "added to one place,
# forgotten in the other" failure this table exists to prevent. The long names are the
# FLAG names, not the ORDER names; keep the two vocabularies distinct.
ARMS="rp vh vm vd vs vv wv"
ORDERS="base $ARMS"
# The per-order payload paths, in ONE place, BEFORE the loop that reads them --
# bash expands an unset variable to the empty string, and `run_order t ""` would
# silently write the payload to a file with no name. Defining them after the loop
# is the same "added to one place, forgotten in the other" failure again.
VH="$BUILD_DIR/rowpair_equiv.verbhalf.txt"
VM="$BUILD_DIR/rowpair_equiv.verbmp.txt"
VD="$BUILD_DIR/rowpair_equiv.verbdead.txt"
VS="$BUILD_DIR/rowpair_equiv.verbsel.txt"
VV="$BUILD_DIR/rowpair_equiv.verbvec.txt"
WV="$BUILD_DIR/rowpair_equiv.wvec.txt"

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
    wv)   f="$WV" ;;
  esac
  r=0; run_order "$t" "$f" || r=$?
  if ! marker_verdict "$t" "$BUILD_DIR/rowpair_equiv.$t.err"; then
    echo "FAIL: order $t marker contract violated" >&2
    r=1
  fi
  rcs="$rcs $t=$r"
done
for t in $ORDERS; do
  # Word-boundary match: `*"$t=0"` is a SUBSTRING test, so `base=0` also matches
  # `rp=0` and every later order is reported "did not complete" while rcs says
  # all-zero. Anchor on the space that precedes each `tag=rc` field.
  case " $rcs " in *" $t=0 "*) ;; *)
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
n_base="$(grep -c '|0x' "$BASE" || true)"
for pair in "rowpair:$RP" "verbhalf:$VH" "verbmp:$VM" \
            "verbdead:$VD" "verbsel:$VS" "verbvec:$VV" "wvec:$WV"; do
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

# *** THE SUBJECT IS INT2, AND ONLY INT2 IS CERTIFIABLE HERE. ***
# Every arm in this gate is guarded on `bits == 2` and therefore cannot touch the
# FP16 fast path (`gemm_usm_decode1_f16`, the `bits == 16` branch). Measured on
# this box, tight-interleaved: `vm` is self-consistent (vm1==vm2==vm3) and vm1 is
# BYTE-IDENTICAL to b1 (0 diff lines, all shapes), while `base` itself flickers
# (b1==b2==b4, b1!=b3) -- the flicker lands entirely on the FP16 shapes. So:
#   * the INT2 portion of the reference is deterministic, and an arm's diff
#     against it is a real measurement;
#   * the FP16 portion is NOT, and comparing an arm against it would report a box
#     glitch as a code fault -- the exact verdict this gate exists to refuse.
# The comparisons below are therefore filtered to INT2 shapes. FP16 shapes are
# WITHHELD, named, and the reason stated, rather than quietly dropped.
BASE2="$BUILD_DIR/rowpair_equiv.base2.txt"
r=0; run_order base "$BASE2" || r=$?
if ! marker_verdict base "$BUILD_DIR/rowpair_equiv.base.err"; then
  echo "FAIL: the reference base order failed its own marker contract" >&2
  exit 1
fi
int2filt() { grep -E '\(INT2\)' "$1"; }
int2filt "$BASE"  | sed 's/|base$//; s/|rp$//; s/|vh$//; s/|vm$//; s/|vd$//; s/|vs$//; s/|vv$//; s/|rowpair$//; s/|verbhalf$//; s/|verbmp$//; s/|verbdead$//; s/|verbsel$//; s/|verbvec$//; s/|wv$//; s/|wvec$//' >"$BASE.int2.norm"
int2filt "$BASE2" | sed 's/|base$//; s/|rp$//; s/|vh$//; s/|vm$//; s/|vd$//; s/|vs$//; s/|vv$//; s/|rowpair$//; s/|verbhalf$//; s/|verbmp$//; s/|verbdead$//; s/|verbsel$//; s/|verbvec$//; s/|wv$//; s/|wvec$//' >"$BASE2.int2.norm"
n_int2="$(grep -c '|0x' "$BASE.int2.norm" || true)"
if ! cmp -s "$BASE.int2.norm" "$BASE2.int2.norm"; then
  echo "FAIL: the INT2 reference is NONDETERMINISTIC on this box -- base ran twice" >&2
  echo "      and the INT2 payloads differ, so no arm's diff can be believed here." >&2
  /usr/bin/diff "$BASE.int2.norm" "$BASE2.int2.norm" | grep -E '^[<>]' \
    | grep -oE '\|[A-Za-z0-9]+\([A-Za-z0-9]*\)\|[A-Za-z0-9]+\|[0-9]+\|[a-z-]+\|' \
    | sort | uniq -c | sort -rn | sed 's/^/        /' >&2
  exit 1
fi
if ! cmp -s "$BASE" "$BASE2"; then
  echo "  ..   reference nondeterministic on the FP16 shapes (base ran twice and" >&2
  echo "        differed); FP16 certification is WITHHELD on this box. The INT2" >&2
  echo "        portion above is stable, so the INT2 comparisons below are real." >&2
  echo "        Remedy for FP16: scripts/gpu_recover.sh rebind (sudo) on a" >&2
  echo "        contended box, then re-run this gate." >&2
else
  echo "  ok    reference stable (base ran twice, identical payloads, all shapes)"
fi

# Strip the order marker and compare VALUES only. The order tag differs by design,
# so diffing the raw text would always differ.
sed 's/|base$//' "$BASE" >"$BASE.norm"

# *** THE COMPARISON IS INT2-ONLY. *** Both operands are filtered to INT2 shapes
# before the diff, so the FP16 nondeterminism on this box (measured: base itself
# flickers across runs while `vm` is byte-identical to base on INT2) cannot turn a
# box glitch into a code fault. FP16 shapes are withheld and named above; they are
# not silently dropped. `n_int2` is the count the ok-line quotes.
check_identical() {  # $1 = label, $2 = file
  local lbl="$1" f="$2"
  # Both sides stripped of their trailing order tag: `base` carries `|base`, every
  # arm carries its own (`|rp`, `|vm`, ...), and `int2filt` above leaves the tag on
  # the base file. Stripping it from BOTH is what makes the diff a value diff.
  int2filt "$f" | sed 's/|base$//; s/|rp$//; s/|vh$//; s/|vm$//; s/|vd$//; s/|vs$//; s/|vv$//; s/|rowpair$//; s/|verbhalf$//; s/|verbmp$//; s/|verbdead$//; s/|verbsel$//; s/|verbvec$//; s/|wv$//; s/|wvec$//' >"$f.int2.norm"
  if cmp -s "$BASE.int2.norm" "$f.int2.norm"; then
    echo "  ok    $lbl is BITWISE IDENTICAL to base on INT2 ($n_int2 element values)"
  else
    echo "FAIL: $lbl is NOT bitwise identical to base on INT2 -- and it was CLAIMED to be." >&2
    echo "      (A bit-exactness ARGUMENT is what shipped the ROWPAIR double" >&2
    echo "       offset and lost the device. This is the measurement saying the" >&2
    echo "       argument is wrong.)" >&2
    echo "      first differing lines:" >&2
    /usr/bin/diff "$BASE.int2.norm" "$f.int2.norm" | head -20 >&2
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
check_identical "INTFER_DECODE1_WVEC   " "$WV" || rc=1
# VERBHALF: report, do not gate.
int2filt "$VH" | sed 's/|base$//; s/|verbhalf$//' >"$VH.int2.norm"
if cmp -s "$BASE.int2.norm" "$VH.int2.norm"; then
  echo "  ok    INTFER_DECODE1_VERBHALF is bitwise identical to base on INT2 datasets"
  echo "        -- which does NOT prove the FTZ/denormal concern away, only that"
  echo "        these synthetic scales never reach it."
else
  nd="$(/usr/bin/diff "$BASE.int2.norm" "$VH.int2.norm" | grep -c '^<' || true)"
  echo "  ..   INTFER_DECODE1_VERBHALF DIFFERS from base on INT2 in ~${nd} of ${n_int2}"
  echo "        values. EXPECTED IN PRINCIPLE and not a gate failure: this arm is"
  echo "        claimed NOT bit-exact (denormal + default FTZ). It is reported so"
  echo "        the magnitude is on the record rather than assumed."
fi
[ "$rc" -eq 0 ] || { [ "$KEEP" -eq 1 ] || rm -f "$BASE" "$RP" "$VH" "$VM" "$VD" "$VS" "$VV" "$WV" "$BASE.norm" "$RP.norm" "$VH.norm" "$VM.norm" "$VD.norm" "$VS.norm" "$VV.norm" "$WV.norm" "$BASE.int2" "$BASE.int2.norm" "$BASE2.int2" "$BASE2"; exit 1; }

# Per-dataset coverage, so a green verdict says WHICH cases passed rather than only
# that they agreed. Derived from the payload, not from a remembered list.
# The header line of each case is "<shape>|<scheme>|<d>|<dataset>|base"; the
# per-element lines carry an index too, so both counters must anchor on the
# header form specifically. Counting element lines gave "40 shapes" for 8.
n_runs="$(grep -cE '^[^|]+\|[A-Z0-9]+\|[0-9]+\|[a-z-]+\|base$' "$BASE.int2" || true)"
n_shapes="$(grep -oE '^[^|]+\|' "$BASE.int2" | sort -u | grep -c . || true)"
n_ds="$(grep -oE '\|[a-z-]+\|base$' "$BASE.int2" | sort -u | grep -c . || true)"
echo "ROWPAIR-EQUIV: $n_int2 element values bitwise identical across:"
grep -oE '^(MLP_GATE\(INT2\)|MLP_DOWN\(INT2\)|MLP_GATE-ODD\(INT2\))\|[A-Z0-9]+\|[0-9]+\|[a-z-]+' "$BASE.int2" \
  | sort -u | awk -F'|' '{printf "  %-20s %-5s %s\n", $1, $2, $5}' | head -50
echo "  ... (the ODD-N shape must take the BASE path; it is included so a silent"
echo "   fallback to a different kernel is visible rather than assumed)"
echo "  *** WITHHELD: the FP16 shapes (LA_QKV / LA_OUT / ATTN_K) are NOT certified"
echo "      here -- the FP16 decode kernel is nondeterministic on this box (base"
echo "      itself flickers across runs), so no arm's diff against them is a real"
echo "      measurement. The INT2 arms cannot reach that path anyway (guarded on"
echo "      `bits == 2`); a stable FP16 reference is the remedy."

[ "$KEEP" -eq 1 ] || rm -f "$BASE" "$RP" "$BASE.norm" "$RP.norm"
echo
echo "DECODE1-EQUIV TEST: PASS"
echo "RESULT: on $n_int2 element values over $n_runs (shape,dataset) runs,"
echo "        $n_shapes distinct shapes x $n_ds distinct datasets:"
echo "          INTFER_DECODE1_ROWPAIR  BITWISE IDENTICAL  -- claim MEASURED"
echo "          INTFER_DECODE1_VERBMP   BITWISE IDENTICAL  -- claim MEASURED"
echo "          INTFER_DECODE1_VERBDEAD BITWISE IDENTICAL  -- pure code motion"
echo "          INTFER_DECODE1_VERBSEL  BITWISE IDENTICAL  -- codebook by select"
echo "          INTFER_DECODE1_VERBVEC  BITWISE IDENTICAL  -- activation ptr type"
echo "          INTFER_DECODE1_WVEC     BITWISE IDENTICAL  -- weight load forming"
echo "        INTFER_DECODE1_VERBHALF reported above: it is claimed NOT bit-exact"
echo "        (denormal + default FTZ), so a difference is information, not a failure."
echo "        *** DEFAULTS AS OF 2026-10-07: VERBHALF and VERBMP are the DEFAULT"
echo "        decode path (kill switches INTFER_NO_DECODE1_VERBHALF=1 / =1 restore"
echo "        the old forms). ROWPAIR / VERBDEAD / VERBSEL / VERBVEC are still"
echo "        opt-in. PERFORMANCE of the two defaults is UNMEASURED: the"
echo "        clone figures (+14% / +17% MLP_GATE VERBHALF, 1.01x VERBMP)"
echo "        were RETRACTED (5bfd8dd: every bisect arm read 0.000 ms /"
echo "        inf GB/s, so the ratios were 0/0). The ENGINE arms in"
echo "        test_tp2_27b.sh are the only way to measure it on real 27B weights."
exit 0
