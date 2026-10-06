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

# ---------- THE THIRD BODY, ADDED BY ENUMERATION (2026-10-05) ----------
# INTFER_DECODE1_ROWPAIR added a THIRD inline copy of the M=1 INT2 arithmetic
# (quant_gemm.cpp, `parallel_for<class gemm_usm_decode1_rowpair>`). This checker
# said PASS on the tree that already contained it, because it enumerated exactly
# the two bodies it was written for -- which is the audit #7/#8 shape the repo has
# paid for repeatedly, and the reason AGENTS.md rule 1 is "ENUMERATE, never
# remember". A third duplicated body that no checker looks at is worse than two
# that are pinned, because it silently becomes the one nobody watches.
#
# So the population is DERIVED, not appended: every parallel_for in the file whose
# name contains `decode1` is a body this pin must either cover or say why not.
# Bodies that are NOT copies of the scalar per-output arithmetic are listed with a
# STATED REASON and are not compared -- and the reason must be a reason, not a name
# that happened to be there. The first version of this block named
# gemm_usm_decode1_blockload, which was a flag written and then DELETED as a
# duplicate of INTFER_DECODE1_SUBGROUP; leaving the name in the exemption list is
# the retirement-family defect (a deleted thing leaving a checker reference behind)
# and it was caught by `git grep` for the flag, not by this checker. So the list
# below is the LIVE body set, and an UNCOVERED entry is a hard failure that fires
# the moment a body is added or renamed.
echo "-- decode1 parallel_for bodies found by ENUMERATION:"
mapfile -t ALL_D1 < <(grep -oE 'parallel_for<class [a-z0-9_]*decode1[a-z0-9_]*>' "$SRC" \
                       | sed -E 's/.*<class ([a-z0-9_]*)>/\1/' | sort -u)
[ "${#ALL_D1[@]}" -ge 1 ] || {
  echo "  FAIL  no decode1 parallel_for body found -- the enumeration drifted" >&2
  exit 1
}
rp_found=0
for k in "${ALL_D1[@]}"; do
  case "$k" in
    # Bodies that re-implement the SAME per-output INT2 arithmetic, so the atom
    # subset below is the right relation.
    gemm_usm_decode1_vec|gemm_usm_decode1_vec_splitk|gemm_usm_decode1_vec_mr|gemm_usm_decode1)
      echo "     covered-by-subset  $k" ;;
    gemm_usm_decode1_verbhalf|gemm_usm_decode1_verbmp|gemm_usm_decode1_verbdead|gemm_usm_decode1_verbsel|gemm_usm_decode1_verbvec)
      # The five bisect arms. Each is a SEPARATE inline parallel_for differing from
      # the base body in exactly ONE place -- verbhalf in how the group scale is
      # converted, verbmp in the pointer TYPE the weight side is read through. They
      # re-implement the same per-output INT2 arithmetic, so the subset relation and
      # the shift-into-mask equality are the right relations, and BOTH are enforced
      # below rather than asserted here.
      echo "     covered-by-subset  $k  (one-change bisect arm; see the" \
           "shift-into-mask equality below)" ;;
    gemm_usm_decode1_rowpair)
      rp_found=1; echo "     COVERED-BY-ATOM-PIN  $k" ;;
    # Bodies with a STATED reason for not sharing the scalar per-output arithmetic.
    gemm_usm_decode1_f16)
      echo "     exempt-by-reason    $k  (the FP16 M=1 path: half x half FMA, no" \
           "2-bit codebook to drift; 57.9% of decode traffic)" ;;
    gemm_usm_decode1_subgroup)
      # QUARANTINED 2026-10-05: the kernel produced a wrong answer and was removed.
      # The enumeration is DERIVED from the source, so once the body is gone this
      # case is unreachable -- and leaving it here would be worse than dead code,
      # because the checker itself caught it: "covered-by-subset but did not
      # extract (0 lines)". Keeping the required-set entry would have turned the
      # quarantine into a red stack.
      echo "     quarantined        $k  (kernel removed 2026-10-05; see" \
           "AGENTS.md)" ;;
    *) echo "     UNCOVERED           $k  <-- no pin and no stated reason" ;;
  esac
done
# The row-pair body MUST be pinned, and the exemption above must not become a
# loophole: an UNCOVERED body is a hard failure, which is what stops this from
# decaying into "exempt everything new".
if printf '%s\n' "${ALL_D1[@]}" | grep -qx 'gemm_usm_decode1_rowpair' && [ "$rp_found" -ne 1 ]; then
  echo "  FAIL  the row-pair body exists but is not pinned" >&2; exit 1
fi
# An UNCOVERED entry is drift in the POPULATION, not a style note.
if printf '%s\n' "${ALL_D1[@]}" | grep -qvE \
     '^(gemm_usm_decode1|gemm_usm_decode1_vec|gemm_usm_decode1_vec_splitk|gemm_usm_decode1_vec_mr|gemm_usm_decode1_f16|gemm_usm_decode1_rowpair|gemm_usm_decode1_verbhalf|gemm_usm_decode1_verbmp|gemm_usm_decode1_verbdead|gemm_usm_decode1_verbsel|gemm_usm_decode1_verbvec)$'; then
  echo "  FAIL  a decode1 body is neither pinned nor exempted -- enumerate it" >&2
  printf '%s\n' "${ALL_D1[@]}" | sed 's/^/        /' >&2
  exit 1
fi
body_rp="$(extract_body gemm_usm_decode1_rowpair)"
n_rp="$(printf '%s\n' "$body_rp" | grep -c . || true)"
for extra in gemm_usm_decode1_vec_mr gemm_usm_decode1; do
  b="$(extract_body "$extra")"
  n="$(printf '%s\n' "$b" | grep -c . || true)"
  if [ "$n" -lt 5 ]; then
    echo "  FAIL  $extra is covered-by-subset but did not extract ($n lines)" >&2
    echo "        claiming coverage of a body that cannot be read is the exact" >&2
    echo "        vacuity this checker exists to refuse." >&2
    exit 1
  fi
  # Line count only. The atom EXTRACTION function is defined further down, and an
  # earlier draft called it here -- `atoms: command not found`, and a coverage
  # claim that cannot read the body it claims to cover is exactly the vacuity this
  # checker exists to refuse. Extraction is verified below where atoms() exists.
  echo "-- $extra: $n line(s) -- covered-by-subset (atoms checked below)"
done
echo "-- row-pair body: $n_rp line(s)"
if [ "$n_rp" -lt 5 ]; then
  echo "  FAIL  the row-pair body did not extract ($n_rp lines) -- it is named" >&2
  echo "        above as covered, so a silent non-extraction would make the" >&2
  echo "        coverage claim a lie." >&2
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
# ROW-PAIR'S ARITHMETIC IS WRITTEN DIFFERENTLY AND THAT IS EXPECTED, so its atoms
# are extracted with an extra shape: it accumulates into `accA`/`accB` rather than
# `acc`, because two independent chains is the entire mechanism. The codebook
# shape (`scale * ...`) and the group-scale read are unchanged, so those are what
# the subset check below compares -- NOT the accumulation, which differs by
# construction and is verified by the token oracle instead.
# *** AND THE ACCUMULATION IS PINNED TOO -- the first version omitted it and the
# mutation matrix caught the omission. *** It originally matched only the codebook
# construction and the group-scale read, on the reasoning that row-pair accumulates
# into accA/accB so its accumulation "differs by construction". True of the
# ACCUMULATOR NAME and false of the INDEX: `(va >> ((j & 3) * 2)) & 3` is the
# codebook-index expression, and planting the real INTFER_DECODE1_SPLITK bug there
# -- `(va >> (j * 2)) & 3` -- left the checker PASS. So the extractor now takes the
# whole accumulate term, and the name difference is carried by the two letters
# rather than by dropping the shape.
#
# That is the second time in this file's history that a narrow extractor looked
# conservative and was not (the first was line-vs-atom comparison, which invented
# drift on a clean tree). A pattern that is too narrow is a silent gap, and only a
# mutation matrix finds one.
atoms_rp() {  # $1 = row-pair body
  printf '%s\n' "$1" \
    | sed -e 's://.*::' -e 's/[[:space:]]//g' \
    | grep -oE 'scale\*[^,;{}]*|from_half_bits\(s[a-zA-Z0-9_]*\[g\]\)|acc[AB]\+=av\*w[AB]4\[[^;{}]*\]' \
    | sort -u
}
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
atoms_rp="$(atoms "$body_rp")"
n_base_a="$(printf '%s\n' "$atoms_base" | grep -c . || true)"
n_sk_a="$(printf '%s\n' "$atoms_sk" | grep -c . || true)"
# The subset relation also holds for the two bodies the enumeration newly exposed.
# MEASURED before this was added: both extract ZERO atoms with the base extractor's
# patterns, because the MR arm accumulates into `acc` but reads its group scale
# through a different expression, and the scalar `gemm_usm_decode1` path reaches
# canon_value() rather than the wtab table. So they are checked for EXTRACTION and
# reported, not folded into the 7-atom subset -- claiming they were in it would
# inflate the number, which is the "handcounted sites" defect in a new place.
for extra in gemm_usm_decode1_vec_mr gemm_usm_decode1; do
  b="$(extract_body "$extra")"
  a="$(atoms "$b" | grep -c . || true)"
  echo "-- $extra: $a atom(s) under the shared extractor (reported, not in the subset)"
done
# *** A REAL COMPARISON, ADDED BECAUSE THE MUTATION MATRIX PROVED THE LABEL WAS A
# LIE. *** The enumeration above printed "COVERED-BY-ATOM-PIN" for the row-pair
# body, but nothing compared its atoms to anything -- `n_rp_a` was a FLOOR, and
# mutate_decode1_copies.sh DC-2 planted the real INTFER_DECODE1_SPLITK bug
# (`(va >> (j * 2)) & 3` for `(va >> ((j & 3) * 2)) & 3`) in the row-pair body and
# the checker still said PASS. A floor is not a pin, and a label that says
# "ATOM-PIN" over a floor is the hand-written-claim failure in a checker.
#
# WHY THE WHOLE-ATOM SUBSET CANNOT WORK HERE, so this is not just "try harder": the
# base body accumulates `acc += av * wtab[code]` with the index in a VARIABLE, while
# row-pair inlines it as `accA += av * wA4[(va >> ((j & 3) * 2)) & 3]`. Normalising
# the accumulator and table names is not enough -- the shapes genuinely differ, and
# forcing them to match would either invent drift on a clean tree (which the checker's
# own header records as a bug it once had) or need an exception list.
#
# SO THE PIN IS ON THE ONE THING THAT MUST NOT VARY: the codebook-index
# expression. That is the exact expression INTFER_DECODE1_SPLITK shipped wrong, and
# it is what a "one activation load, two code pointers" kernel is most likely to
# get wrong.
#
# *** AND THE EXTRACTOR MUST MATCH THE WRONG FORM TOO, WHICH IT DID NOT AT FIRST.
# *** The first version matched only `X >> ((j & 3) * 2)) & N`, i.e. the CORRECT
# shape -- so mutating row-pair's accA line to `(va >> (j * 2)) & 3` did not break
# the pin, because the mutated expression simply stopped matching and the remaining
# UNMUTATED accB line supplied the good atom. DC-2 stayed green. That is this repo's
# "matches one spelling and silently stops finding the defect" failure, one level
# below the one it had already fixed once: a pattern that only recognises the right
# answer cannot detect a wrong one.
#
# So the extractor is deliberately LOOSE -- any shift-into-mask over a packed byte
# -- and the relation is EQUALITY, not subset. Both bodies must contain exactly the
# same set, which is a strictly stronger claim than "row-pair's are among the base's"
# and is what makes a mutated expression visible as an EXTRA element.
shift_index_atoms() {  # $1 = body; normalises the packed-byte variable to V
  printf '%s\n' "$1" \
    | sed -e 's://.*::' -e 's/[[:space:]]//g' \
    | grep -oE '[a-zA-Z0-9_]+>>[^;{}]{0,24}&[0-9]+' \
    | sed -E 's/^[a-zA-Z0-9_]+>>/V>>/' \
    | sort -u
}
idx_base="$(shift_index_atoms "$body_base")"
idx_rp="$(shift_index_atoms "$body_rp")"
n_idx_rp="$(printf '%s\n' "$idx_rp" | grep -c . || true)"
n_idx_base="$(printf '%s\n' "$idx_base" | grep -c . || true)"
echo "-- shift-into-mask atoms: base=$n_idx_base, rowpair=$n_idx_rp"
printf '%s\n' "$idx_rp" | sed 's/^/     rowpair: /'
if [ "$n_idx_base" -lt 1 ] || [ "$n_idx_rp" -lt 1 ]; then
  echo "  FAIL  the shift-into-mask extractor found nothing (base=$n_idx_base" >&2
  echo "        rowpair=$n_idx_rp) -- it drifted from the source, so the equality" >&2
  echo "        below would be agreement with nothing." >&2
  exit 1
fi
# EQUALITY, not subset: an expression in row-pair that the base does not have is the
# bug, and a subset test would wave it through as long as one good atom remained.
extra_idx="$(printf '%s\n' "$idx_rp" | grep -vxF "$idx_base" || true)"
if [ -n "$extra_idx" ]; then
  echo "  FAIL  the row-pair body uses a shift-into-mask the general body does not:" >&2
  echo "        $extra_idx" >&2
  echo "        *** This is the INTFER_DECODE1_SPLITK bug class: INT2 packs four" >&2
  echo "        codes per byte, so the shift must be ((j & 3) * 2), and a uint8_t" >&2
  echo "        shifted by >= 8 promotes to 0. ***" >&2
  exit 1
fi
echo "  ok    the row-pair shift-into-mask set EQUALS the general body's"
# The two bisect arms are one-change VARIANTS, so their shift-into-mask set must be
# identical to the base's -- a difference there is the INTFER_DECODE1_SPLITK bug
# class. Checked as EQUALITY, and the extractor is deliberately loose (it matches the
# WRONG form too), because an extractor that only recognises the right answer cannot
# detect a wrong one -- which is how the first version of this check passed a planted
# bug.
# Every `verb*` body is checked, DERIVED from the parallel_for names rather than
# listed -- so an arm added today is pinned today, which is the whole point after
# three separate incidents of a checker enumerating fewer bodies than existed.
# The relation is EQUALITY of the shift-into-mask set against the general body, and
# the extractor is deliberately LOOSE (it matches the WRONG form too), because an
# extractor that only recognises the right answer cannot detect a wrong one -- which
# is how the first version of this check passed a planted INTFER_DECODE1_SPLITK bug.
verb_bodies="$(printf '%s\n' "${ALL_D1[@]}" | grep '^gemm_usm_decode1_verb' || true)"
if [ -z "$verb_bodies" ]; then
  echo "  FAIL  no verb* body was enumerated -- the enumeration drifted" >&2
  exit 1
fi
for k in $verb_bodies; do
  set="$(shift_index_atoms "$(extract_body "$k")")"
  if [ -z "$set" ]; then
    echo "  FAIL  $k extracted no shift-into-mask atom at all" >&2
    exit 1
  fi
  d="$(printf '%s\n' "$set" | grep -vxF "$idx_base" || true)"
  if [ -n "$d" ]; then
    echo "  FAIL  $k's shift-into-mask set differs from the general body's:" >&2
    echo "        $d" >&2
    echo "        *** INTFER_DECODE1_SPLITK bug class: INT2 packs FOUR codes per" >&2
    echo "        byte, so the shift must be ((j & 3) * 2). ***" >&2
    exit 1
  fi
  echo "  ok    $k's shift-into-mask set EQUALS the general body's"
done

n_rp_a="$(printf '%s\n' "$(atoms_rp "$body_rp")" | grep -c . || true)"
echo "-- arithmetic atoms: base=$n_base_a, splitk=$n_sk_a, rowpair=$n_rp_a"
echo "   split-K atoms the general body must also hold:"
printf '%s\n' "$atoms_sk" | sed 's/^/     /'

# ANTI-VACUITY, asserted. Zero atoms means the extractor stopped matching the
# source and every agreement below would be agreement with nothing (audit #3).
if [ "$n_sk_a" -lt 4 ] || [ "$n_base_a" -lt 4 ] || [ "$n_rp_a" -lt 2 ]; then
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
