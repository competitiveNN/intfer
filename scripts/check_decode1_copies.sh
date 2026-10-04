#!/usr/bin/env bash
# The two M=1 decode bodies are DELIBERATE duplicates, and this pins them
# against EACH OTHER.
#
# WHY DUPLICATION IS ACCEPTED HERE AT ALL. quant_gemm.cpp holds two inline
# copies of the M=1 INT2 decode arithmetic:
#   gemm_usm_decode1_vec          the original, one work-item per output
#   gemm_usm_decode1_vec_splitk   INTFER_DECODE1_SPLITK, S partials per output
# They are NOT factored into a shared device function on purpose.
# gemm::quant_gemm_fused_decode1 was built by extracting that body into a shared
# helper; a normalized statement-by-statement diff of the two bodies said the
# arithmetic was identical -- it WAS identical -- and 27B decode still went
# 1284 ms -> 231 114 ms, bisected to the EXTRACTION and not to the fusion it was
# extracted for. One home is therefore the thing that actually broke this box.
# Two homes plus this checker is the cheaper trade.
#
# *** WHAT IS BEING CLAIMED, PRECISELY. *** This checker asserts the two copies
# AGREE. It does NOT assert either one is CORRECT. Those are different claims
# and merging them would produce a checker that cannot be satisfied:
#   * if it compared the copies to a list of remembered literals, then changing
#     BOTH copies together would fail it -- and that change is sometimes
#     correct, so the checker would go red on a legitimate edit and be disabled.
#     (MEASURED: the first version did exactly that, and its DC-6 "both copies
#     drift together" state came back FALSE POSITIVE.)
#   * correctness against the token oracle belongs to the INTFER_DECODE1_SPLITK
#     arm in scripts/test_tp2_27b.sh, which runs the model and compares the
#     stream to the FIXED oracle. That arm is the only thing that can say
#     "right"; this checker is the only thing that says "the same in both
#     places", which is the cheap question runnable in milliseconds.
# So: DRIFT is caught here, in seconds. WRONGNESS is caught by the gate, in
# minutes. Neither is asked to do the other's job.
#
# AND DRIFT ALREADY HAPPENED, MEASURED. The split-K copy shipped with
#     (bv >> (j * 2)) & 3
# where the original has
#     (bv >> ((j & 3) * 2)) & 3
# INT2 packs FOUR codes per byte, so j=4..7 must re-read bit 0 of the second
# byte. Shifting a uint8_t by 8..14 is >= its width: the integer promotion makes
# every one of those four codes 0. Found by a 27B reachability probe, because
# the stream diverged at step 0 while INTFER_NO_DECODE1_SPLITK stayed clean --
# which localised it to the arithmetic rather than to the guard, the reduction,
# or the phase arithmetic. This checker exists so the next drift is caught in
# milliseconds instead of in a 27B run.
#
# WHY NOT "DIFF THE WHOLE BODY". The two kernels are legitimately DIFFERENT:
# split-K adds phase arithmetic, a local reduction and a barrier. Diffing whole
# bodies would report those as drift on every run and train everyone to ignore
# the output. So the comparison is over the SHARED ARITHMETIC -- the statements
# that carry the codebook and the accumulation -- compared as MULTISETS between
# the two bodies. The split-specific statements are simply not in the selection,
# so they cannot produce noise.
#
# ANTI-VACUITY, both directions. A comparison that matches nothing must not
# certify agreement, so the number of compared statements is asserted non-zero
# and both bodies must extract at all (a body that fails to extract must be
# refused, never read as "no differences").
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

SRC="src/gemm/quant_gemm.cpp"
[ -f "$SRC" ] || { echo "FAIL: $SRC missing" >&2; exit 2; }

echo "== DECODE1 DUPLICATE-BODY PIN (two inline copies must AGREE) =="

extract_body() {  # $1 = parallel_for template name, derived from source
  awk -v want="$1" '
    index($0, "parallel_for<class " want ">") { inside=1; next }
    inside && /^[[:space:]]*\}\);?[[:space:]]*$/ { exit }
    inside { print }
  ' "$SRC"
}

body_base="$(extract_body gemm_usm_decode1_vec)"
body_sk="$(extract_body gemm_usm_decode1_vec_splitk)"

n_base="$(printf '%s\n' "$body_base" | grep -c . || true)"
n_sk="$(printf '%s\n' "$body_sk" | grep -c . || true)"
echo "-- extracted bodies: base=$n_base line(s), splitk=$n_sk line(s)"
if [ "$n_base" -lt 5 ] || [ "$n_sk" -lt 5 ]; then
  echo "  FAIL  a body did not extract (base=$n_base splitk=$n_sk) -- an" >&2
  echo "        extractor that matches nothing must not certify agreement." >&2
  exit 1
fi

# The SHARED ARITHMETIC selection: normalized lines that carry the codebook or
# the accumulation. Derived from what the arithmetic LOOKS LIKE, not from a list
# of remembered lines, so a refactor that moves them does not read as drift.
# Split-specific lines (phase arithmetic, reduction, barrier) contain none of
# these idioms and are excluded by construction.
# *** PIPE THE BODY IN, DO NOT PASS IT AS AN ARGUMENT. *** sed read a
# multi-line argument as a FILENAME and died with "File name too long", which
# surfaced as base=0/splitk=0 selected -- i.e. the anti-vacuity guard caught a
# real defect in the checker and would otherwise have been tuned away.
# ---------- ATOMS, NOT LINES ----------
# Two normalizations were needed before this comparison meant anything, and both
# were found by running it rather than by reasoning about it:
#
#  1. LINES. The two bodies wrap the same codebook differently -- the general
#     kernel puts `scale * -1.0f, scale * (-1.0f / 3.0f),` on one line and
#     split-K spreads them over two. Comparing line-by-line reported invented
#     statements on a tree that had NOT drifted.
#  2. WHITESPACE. `scale * -1.0f` and `scale*-1.0f` are one atom. All
#     whitespace is removed before extraction.
#
# The atom shapes are derived from what the arithmetic DOES -- extract a code
# from a byte, accumulate a term, build a codebook entry, read a group scale --
# not from a list of remembered lines (audit #7/#8), so re-wrapping does not read
# as drift while a changed operator or constant does.
atoms() {  # $1 = body
  printf '%s\n' "$1" \
  | sed -e 's://.*::' -e 's/[[:space:]]//g' \
  | grep -oE '\(bv>>[^;{}]*|acc\+=[^;{}]*|scale\*[^,;{}]*|from_half_bits\(sptr\[[a-z0-9]+\]\)' \
  | sort -u
}

atoms_base="$(atoms "$body_base")"
atoms_sk="$(atoms "$body_sk")"
n_base_a="$(printf '%s\n' "$atoms_base" | grep -c . || true)"
n_sk_a="$(printf '%s\n' "$atoms_sk" | grep -c . || true)"
echo "-- arithmetic atoms: base=$n_base_a, splitk=$n_sk_a"
echo "   split-K atoms the general body must also hold:"
printf '%s\n' "$atoms_sk" | sed 's/^/     /'

# ANTI-VACUITY, asserted. Zero atoms means the extractor stopped matching the
# source and every agreement below would be agreement with nothing (audit #3).
if [ "$n_sk_a" -lt 4 ] || [ "$n_base_a" -lt 4 ]; then
  echo "  FAIL  too few arithmetic atoms extracted (base=$n_base_a splitk=$n_sk_a)" >&2
  echo "        -- the extractor drifted from the source, or a body lost its" >&2
  echo "        arithmetic. Fix either; do not lower the floor." >&2
  exit 1
fi

# ---------- SUBSET, NOT EQUALITY ----------
# The two bodies are NOT two copies of one code: the general kernel also carries
# the INT8 path and the generic packed path, while split-K is INT2-only because
# its branch is guarded by `bits == 2`. MEASURED on a tree that has NOT drifted,
# base yields more atoms than split-K, so demanding equality fails on correct
# code. The correct relation is the specialization one: every atom the split-K
# body holds must ALSO be held by the general body. So split-K cannot INVENT
# arithmetic the general kernel lacks, and cannot change an operator or a
# constant without leaving an atom the general kernel does not contain. That is
# exactly the drift class that shipped -- (bv>>(j*2))&3 against
# (bv>>((j&3)*2))&3 -- and it is caught from EITHER direction:
#     split-K drifts -> it holds an atom the general body does not -> FAIL
#     general drifts -> it LACKS an atom split-K holds          -> FAIL
# and BOTH changing together leaves the subset intact -> correctly GREEN, since a
# joint change is for the token oracle to judge, not drift.
invented="$(comm -13 <(printf '%s\n' "$atoms_base") <(printf '%s\n' "$atoms_sk") || true)"
n_invented="$(printf '%s\n' "$invented" | grep -c . || true)"
if [ "$n_invented" != 0 ]; then
  echo "  FAIL  THE SPLIT-K COPY INVENTED ARITHMETIC the general kernel does not" >&2
  echo "        have -- $n_invented atom(s): THE COPIES DIVERGED."
  printf '%s\n' $invented | sed 's/^/        /' >&2
  echo "RESULT: the two M=1 decode copies DISAGREE -- fix the drifted copy;" >&2
  echo "        do not widen the atom shapes to silence this." >&2
  exit 1
fi

echo "  ok    all $n_sk_a split-K atom(s) also present in the general body"
echo
echo "DECODE1-COPY PIN: PASS ($n_sk_a/$n_sk_a split-K atoms present in the general body;"
echo "                   general body holds $n_base_a, split-K body holds $n_sk_a)"
echo "NOTE what this does and does not claim:"
echo "  DRIFT between the two inline copies ....... caught here, in milliseconds"
echo "  CORRECTNESS against the token oracle ........ test_tp2_27b.sh's arm"
echo "  (both copies changing together is NOT drift, and correctly stays green)"
echo "RESULT: the duplicated M=1 decode arithmetic has NOT drifted"
exit 0
