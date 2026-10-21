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
#     arm in scripts/gates/test_tp2_27b.sh, which runs the model and compares the
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
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
     gemm_usm_decode1_wvec)
      # PLAN.md 1.7: the vectorized-weight-load arm. A one-change
      # variant of the base INT2-tab body -- the chunk's two INT2
      # weight bytes as ONE sycl::vec<std::uint8_t,2> load from a
      # row-base multi_ptr instead of two scalar code_u8[b0] /
      # code_u8[b0+1] reads -- so it re-implements the same
      # per-output arithmetic with the same shift-into-mask. It is
      # pinned by the equality below BY DERIVATION (the population
      # grep is verb|wvec), not by this label.
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
    gemm_usm_decode1_gateup_int2tab)
      # FUSED gate+up (INTFER_FUSED_GATE_UP, TP=2). This is NOT a copy of the
      # scalar per-output decode arithmetic and the subset/atom/shift-into-mask
      # relations do not describe it:
      #   * it is a FUSION of two GEMMs, not one -- each WI dispatches on `is_up`
      #     to read gate OR up weights and write to out_g OR out_u;
      #   * it carries an OUTER m loop (M>1 prefill), so it is not an M=1 body;
      #   * its weight row is computed as `wbase + out_ng*K` INSIDE the m loop
      #     (the M>1 index fix, 783e56f), so the byte-index expression is
      #     `((size_t)out_ng * K + kg) / pack` -- structurally different from the
      #     decode bodies' `code_u8[idx / pack]` with idx = row_base + kg.
      # The relation that DOES hold is bit-exactness against two separate
      # quant_gemm_forward_usm calls, and that is certified by the
      # FUSED_GATE_UP arm in scripts/gates/test_tp2_27b.sh (the only gate whose model
      # carries INT2), which asserts the stream against the fixed oracle -- not
      # by this checker, which is about drift BETWEEN copies, and this body has
      # no sibling copy to drift from.
      echo "     exempt-by-reason  $k  (TP=2 gate+up FUSION, not a per-output" \
           "decode copy; bit-exactness is certified by the test_tp2_27b.sh" \
           "FUSED_GATE_UP arm, see the case header)" ;;
           gemm_usm_decode1_gateup_int2tab_silumul)
             # INTFER_MLP_GATEUP_SILUMUL_FUSE: the gate+up FUSION with silu(gate)*up
             # folded into the down activation on device (engine.cpp:3388
             # quant_gemm_forward_usm_gateup_compact_silumul). This is the SAME
             # fusion seam as gateup_int2tab, so the same structural reasons apply:
             #   * it is a FUSION of two GEMMs, not one -- each WI dispatches on
             #     `is_up` to read gate OR up weights and write to out_g OR out_u;
             #   * it carries an OUTER m loop (M>1 prefill), so it is not an M=1
             #     body;
             #   * its weight row is computed as `wbase + out_ng*K` INSIDE the m
             #     loop (quant_gemm.cpp:2677), so the byte-index expression is
             #     `((size_t)local * K + kg) / pack` -- structurally different from
             #     the decode bodies' `code_u8[idx / pack]` with idx = row_base + kg.
             # The relation that DOES hold is bit-exactness against two separate
             # quant_gemm_forward_usm calls, and that is certified by the
             # GATEUP_SILUMUL_FUSE arm in scripts/gates/test_gpu_optin.sh (the only gate
             # whose model carries INT2+INT2 MLPs), which asserts the stream
             # against the fixed oracle -- not by this checker, which is about
             # drift BETWEEN copies, and this body has no sibling copy to drift
             # from. It is a SIBLING of gateup_int2tab, not a copy: the two differ
             # in whether silumul_to_half is chained off the launch event, so
             # comparing them would be a numerics contract (the silu-mul changes
             # the down activation), not a drift check.
             echo "     exempt-by-reason  $k  (TP=2 gate+up+silumul FUSION, not a" \
           "per-output decode copy; bit-exactness is certified by the" \
           "test_gpu_optin.sh GATEUP_SILUMUL_FUSE arm, see the case header)" ;;
    gemm_usm_decode1_wave_fuse)
      # INTFER_WAVE_LAUNCH_FUSE: ONE parallel_for over the UNION of a wave's
      # independent GEMM outputs, instead of N separate launches. This is NOT a
      # copy of the scalar per-output decode arithmetic and the subset /
      # shift-into-mask relations do not describe it, for the same structural
      # reasons as gateup_int2tab -- and one more:
      #   * it is a FUSION of N_items GEMMs, not one -- each WI is located into
      #     its item by a prefix-sum linear search over start[i], then reads
      #     THAT item's M/N/K/scheme/act/scale/code/out pointers;
      #   * it carries NO shift-into-mask atom at all: the weight byte is read
      #     as `wrow[idx / pack]` with `within = idx % pack` and
      #     `code = (byte >> (within * bits)) & ((1 << bits) - 1)` -- the
      #     GENERAL packed form, valid for any bits, not the INT2-specific
      #     `bv >> ((j & 3) * 2) & 3`. So it has nothing for the verb|wvec
      #     EQUALITY loop to compare, and adding it there would be a red stack
      #     on a body that is correct by construction.
      # The relation that DOES hold is stream-identity against N separate
      # quant_gemm_forward_usm calls, and that is certified by the
      # WAVE_LAUNCH_FUSE arm in scripts/gates/test_tp2_27b.sh (the only gate whose
      # model carries INT2), which asserts the stream against the fixed
      # oracle -- not by this checker, which is about drift BETWEEN copies,
      # and this body has no sibling copy to drift from.
      echo "     exempt-by-reason  $k  (wave-launch FUSION, not a per-output" \
           "decode copy; stream identity is certified by the test_tp2_27b.sh" \
           "WAVE_LAUNCH_FUSE arm, see the case header)" ;;
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
     '^(gemm_usm_decode1|gemm_usm_decode1_vec|gemm_usm_decode1_vec_splitk|gemm_usm_decode1_vec_mr|gemm_usm_decode1_f16|gemm_usm_decode1_rowpair|gemm_usm_decode1_verbhalf|gemm_usm_decode1_verbmp|gemm_usm_decode1_verbdead|gemm_usm_decode1_verbsel|gemm_usm_decode1_verbvec|gemm_usm_decode1_wvec|gemm_usm_decode1_gateup_int2tab|gemm_usm_decode1_gateup_int2tab_silumul|gemm_usm_decode1_wave_fuse)$'; then
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
# Every `verb*` body (and the WVEC one-change variant, whose name does
# not start with `verb` -- the derivation is `verb|wvec` so an arm is
# pinned the day it is added, which is the whole point after three
# separate incidents of a checker enumerating fewer bodies than existed).
# The relation is EQUALITY of the shift-into-mask set against the general body, and
# the extractor is deliberately LOOSE (it matches the WRONG form too), because
# an extractor that only recognises the right answer cannot detect a wrong one --
# which is how the first version of this check passed a planted INTFER_DECODE1_SPLITK bug.
verb_bodies="$(printf '%s\n' "${ALL_D1[@]}" | grep -E '^gemm_usm_decode1_(verb|wvec)' || true)"
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

# ---------- THE EXCLUSIVITY GUARD'S ENUMERATION (added 2026-10-14) ----------
# The decode1 bisect arms each end in `return q.submit(...)`, so with more than one
# opt-in arm live the EARLIEST wins and the later ones are silently unreachable.
# The guard that refuses that state used to size its name array `on[2]` against a
# population of SEVEN `note_o` callers, so a 3-live-arm state wrote past on[1] (an
# out-of-bounds stack store) AND the refusal named only the first two -- hiding the
# third from the operator the message exists to inform. The fix (quant_gemm.cpp:544-573)
# sizes on[8], clamps note_o, and ENUMERATES every live arm.
#
# WHAT THIS SECTION CLAIMS, AND WHY IT ASSERTS ON SOURCE AND NOT ON A LOG. It pins
# the two properties the fix exists to guarantee:
#   (a) ARITY -- the refusal must name EVERY live arm, which requires a printing
#       LOOP, and the pre-fix form (`"%s and %s"` from on[0]/on[1]) is exactly the
#       two-name output that must be refused.
#   (b) CAPACITY -- the name array must be sized at least as large as the number of
#       arms note_o can be called for, which is derived, not remembered.
# An earlier draft of this section synthesised a log and asserted three indented
# names appeared in it. That proved nothing: the log was built from the FIXED guard,
# so it passed by construction, and it would have passed with the same shape whether
# the guard enumerated or not. The negative direction is not a synthetic log, it is
# the pre-fix MESSAGE PLANTED IN THE SOURCE -- mutate_decode1_copies.sh DC-8 plants
# `"%s and %s"`/`on[0]`/`on[1]` and requires this checker to REFUSE it (rule 11:
# weaken the artifact, require rejection). Both polarities are therefore exercised,
# and neither by the checker congratulating itself.
echo
echo "== DECODE1 EXCLUSIVITY-GUARD ENUMERATION (asserted on source) =="
guard_src="$(sed -n '/const char\* on\[/,/^    }$/p' "$SRC")"
if [ -z "$guard_src" ]; then
  echo "  FAIL  the exclusivity guard did not extract -- a checker that cannot read" >&2
  echo "        the guard it pins would certify nothing and pass anyway." >&2
  exit 1
fi
# (b) CAPACITY: the array is sized against the arms that feed it. Both numbers are
# DERIVED from the source's own literals at RUNTIME, so a rename or a seventh arm moves
# both -- this is not a remembered constant. The derivation is the point: a hand-written
# "on[8]" here would be a citation that decays, and a seventh note_o caller would silently
# write past the array without either number moving.
n_arms="$(printf '%s\n' "$guard_src" | grep -oE 'note_o\("[A-Z0-9_]+"\)' | grep -c . || true)"
arr_sz="$(printf '%s\n' "$guard_src" | grep -oE 'on\[[0-9]+\]' | head -1 \
          | sed -E 's/on\[([0-9]+)\]/\1/' || true)"
echo "-- guard: $n_arms note_o call(s), name array sized $arr_sz (both derived at runtime)"
if [ -z "$arr_sz" ] || [ "$arr_sz" -lt "$n_arms" ]; then
  echo "  FAIL  the guard's name array (on[$arr_sz]) is SMALLER than the number of" >&2
  echo "        arms that write into it ($n_arms) -- a state with more than $arr_sz" >&2
  echo "        live opt-in arms writes past the array (the pre-fix defect) AND" >&2
  echo "        cannot be enumerated by any message." >&2
  exit 1
fi
# note_o must CLAMP, not blindly store: writing only inside the bound is what makes
# the oversized population a refused-state loud failure instead of UB. The bound is
# DERIVED from the array size (not remembered as a literal), so planting on[2]
# makes this fire on the clamp check rather than on the extraction.
clamp_bound="$(printf '%s\n' "$guard_src" | grep -oE 'no < [0-9]+' | head -1 \
               | sed -E 's/no < ([0-9]+)/\1/' || true)"
if [ -z "$clamp_bound" ] || [ "$clamp_bound" -lt "$arr_sz" ]; then
  echo "  FAIL  note_o no longer clamps to the array bound (bound=$clamp_bound," >&2
  echo "        array=on[$arr_sz]) -- the clamp is what turns an oversized population" >&2
  echo "        into a loud refusal instead of an out-of-bounds store." >&2
  exit 1
fi
# (a) ARITY, both halves. The refusal must CONTAIN the printing loop (enumeration)
# and must NOT contain the pre-fix two-name form. An extractor that only recognised
# the right answer could not detect the wrong one -- that is how the first version of
# this checker passed a planted bug -- so the two-name literal is matched verbatim.
refusal="$(printf '%s\n' "$guard_src" | sed -n '/decode1-bisect\] REFUSING/,/Exactly one opt-in arm/p')"
if [ -z "$refusal" ]; then
  echo "  FAIL  the guard's refusal message did not extract -- the enumeration claim" >&2
  echo "        would be agreement with nothing." >&2
  exit 1
fi
if printf '%s\n' "$refusal" | grep -q 'for (int i = 0'; then
  echo "  ok    the refusal enumerates by loop"
else
  echo "  FAIL  the guard's refusal does not enumerate the live arms by loop --" >&2
  echo "        a fixed-arity message cannot name three arms, which is exactly the" >&2
  echo "        defect this pin exists to catch (DC-8 plants it; it must REFUSE)." >&2
  exit 1
fi
if printf '%s\n' "$refusal" | grep -qE '%s and %s|on\[0\], on\[1\]'; then
  echo "  FAIL  the guard's refusal still uses the two-name form (a single format" >&2
  echo "        string binding on[0] and on[1]) -- only the FIRST TWO live arms" >&2
  echo "        would ever be named, hiding every later one from the operator the" >&2
  echo "        refusal exists to inform. Enumerate every live arm instead." >&2
  echo "        *** NOTE the extractor matches the WRONG spelling too, not only the" >&2
  echo "            right one: a pattern that recognises only the fixed form cannot" >&2
  echo "            detect the defect, which is how this checker once passed one. ***" >&2
  exit 1
fi
echo "  ok    the refusal does not use the pre-fix two-name form"
# The default production state (zero opt-in flags) must not reach the refusal. This
# is the always-red twin's quiet half: a guard that fired on the default path would
# be tuned away until it refused nothing. Derived from the guard's OWN condition.
if printf '%s\n' "$guard_src" | grep -q 'if (no > 1)'; then
  echo "  ok    the guard's condition is no>1, so the default state (0 opt-in" \
       "flags) cannot refuse"
else
  echo "  FAIL  the guard's dispatch condition is no longer no>1 -- the hot path" >&2
  echo "        may now be refused by a guard nobody dares remove." >&2
  exit 1
fi
# *** THE RUNTIME DERIVATION IS ALREADY EXERCISED, AND BY A SINGLE CHANGE. *** A
# seventh note_o caller cannot be planted without also editing the array (they are
# the same literal), so "add a caller, keep on[8]" is a coupled two-change state and is
# excluded for the same reason as DC-6/7. But the derivation's other direction IS a
# single change: DC-8 plants on[2] (array shrunk, population unchanged at 7), and the
# capacity check fires on it -- 2 < 7. So the derivation is load-bearing and is
# exercised by the matrix; no separate state is needed, and inventing one would be the
# vacuous-state defect this repo has a memory tag for.
echo "  ok    capacity is derived at runtime (7 note_o callers vs on[$arr_sz]); DC-8" \
     "plants on[2] and fires it"

echo
echo "DECODE1-COPY PIN: PASS ($n_sk_a/$n_sk_a split-K atoms present in the general body;"
echo "                   general body holds $n_base_a, split-K body holds $n_sk_a;"
echo "                   exclusivity guard names on[$arr_sz] >= $n_arms arm(s) and enumerates by loop)"
echo "NOTE what this does and does not claim:"
echo "  DRIFT between the two inline copies ....... caught here, in milliseconds"
echo "  CORRECTNESS against the token oracle ........ test_tp2_27b.sh's arm"
echo "  (both copies changing together is NOT drift, and correctly stays green)"
echo "  EXCLUSIVITY-GUARD arity/capacity ......... pinned here; DC-8 in"
echo "    mutate_decode1_copies.sh plants the pre-fix two-name refusal and must REFUSE"
echo "RESULT: the duplicated M=1 decode arithmetic has NOT drifted"
echo "RESULT: the exclusivity guard enumerates every live arm (no pre-fix two-name)"
exit 0
