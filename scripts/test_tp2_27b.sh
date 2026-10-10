#!/usr/bin/env bash
# TP=2 inference test: 27B self-contained quant model on the 2x B580 fleet.
#
# Runs intfer_cli on Qwen3.8-27B-sc.intifer with --gpu --tp 2 and checks the
# greedy decode against the HF oracle for "The capital of France is":
#   prefill -> 11751, decode -> 13 198 760
# Also reports prefill tok/s and decode tok/s vs the 4000 / 40t/s targets.
#
#
# BUILD-FIRST (audit #7): gate_require_build in scripts/_gate_build.sh. Audit
# #6 fixed four gates against the stale-artifact class; this one consumed the
# intfer_cli binary without ever building it, so a source edit that failed to
# compile left the previous binary in place and the gate judged code that no
# longer existed. Opt out with INTFER_GATE_NO_BUILD=1 only when the build is
# known current.
#
# Usage: scripts/test_tp2_27b.sh [-n MAX_NEW] [-p PROMPT]
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${INTFER_27B_SC:-/var/home/fra/data/models/intfer/Qwen3.8-27B-sc.intifer}"
# Fall back to the merged chunk artifact when the -sc file is absent
# (same 27B INT2 payload; INTFER_27B_SC still overrides).
if [ ! -f "$MODEL" ]; then
  MODEL="/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer"
fi
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
N=4
PROMPT="The capital of France is"
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    -p) PROMPT="$2"; shift 2 ;;
    --self-check) SELFCHECK=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- Oracle gate (greedy): prefill 11751, then 13 198 760 ...
# *** ONE PARSER, THREE CONSUMERS. *** The base run, the DECODE1_MR arm below and
# the --self-check all read the stream through these two helpers. The MR arm used
# to carry its own copy of the grep|sed|tail expression, which is the audit #3
# shape (a rule fixed at one site and re-derived at another): a log-format change
# would have updated the base parse and left the MR arm silently empty, and the
# MR arm compares DERIVED values, so two empty strings compare equal.
cli_ids() { # $1 = log -- every decode-step token id, prefill included
  grep -a -oE "decode step [0-9]+: token=[0-9]+" "$1" | sed -E 's/.*token=//'
}
cli_prefill() { # $1 = log -- the step-0 id alone
  grep -a -oE "decode step 0: token=[0-9]+" "$1" | sed -E 's/.*=//'
}

# --- DECODE1_MR ARM PREDICATES: ONE HOME, TWO CONSUMERS ----------------------
# *** WHY THESE ARE FUNCTIONS AND NOT INLINE TESTS. *** The first version of this
# self-check defined its own `arm_ok` INSIDE the --self-check block and called it
# "the real arm's guard, verbatim". It was not verbatim: it was a COPY, so the
# real arm at the bottom of this file carried its own separate inline tests, and
# the self-check could not see them at all. MEASURED on the committed version:
# deleting all three real-arm guard lines (M-A), deleting the real arm's prefill
# oracle (M-G) and deleting its TP=sharded grep (M-F) each left the self-check
# at rc=0 PASS. Three of this arm's assertions were UNOBSERVABLE to the check
# written to cover them -- audit #3's shape (a rule fixed at one site and
# re-derived at another) committed in the code written to avoid exactly that.
#
# One definition, called by the real arm AND driven by the self-check, so a
# weakening of any of them turns the self-check red by construction rather than
# by a copy staying in agreement. Each returns 0=accept.
mr_prefill_ok() {   # $1 = parsed step-0 id. Oracle 11751 (greedy, fixed).
  [ "$1" = "11751" ]
}
mr_streams_agree() { # $1 = base ids, $2 = MR ids.
  # NON-EMPTY FIRST, or this is vacuous: the arm compares two DERIVED values
  # (unlike every other arm in this gate, which compares to a literal oracle),
  # so a log-format change or a broken pipe that emptied BOTH sides would make
  # "" = "" and the arm would report PASS having certified nothing. The base run
  # gets non-emptiness for free from its literal "13 198 760" check.
  [ -n "$1" ] && [ -n "$2" ] && [ "$1" = "$2" ]
}
mr_tp_sharded() {   # $1 = log. "TP=sharded" is a substring of the banner the CLI
                    # already prints and this gate already greps for.
  grep -aq "TP=sharded" "$1"
}
# *** IS THE SUBJECT PRESENT AT ALL? *** The three predicates above compare two
# streams or check a literal. This one checks that the code under test RAN, which
# is a different question and the one that decides whether the rest means
# anything: a flag that never reaches its kernel makes every stream comparison
# vacuous, because both runs then produce the same tokens for uninteresting
# reasons. Defined HERE, at the top level with the other predicates, rather than
# inline in the arm -- audit #3's shape is a rule re-derived at a second site,
# and the whole point of this fix is that there is one home.
decode1_path_fired() {   # $1 = log, $2 = announcement tag, e.g. decode1-mr
  grep -aq "\[$2\].*ACTIVE" "$1"
}
mr_path_fired() { decode1_path_fired "$1" 'decode1-mr'; }
# *** ONE HOME FOR "DID THE FLAG REACH ITS KERNEL". *** Added for
# INTFER_DECODE1_SPLITK, which is the second flag in this file that needs the
# subject-present question answered. Copy-pasting mr_path_fired with a different
# tag would be the audit #3 shape -- one rule, two homes, and the second one free
# to drift untested. Both flags go through decode1_path_fired; both arms go
# through the same assertion function below.
sk_path_fired()  { decode1_path_fired "$1" 'decode1-splitk'; }
# *** FOURTH FLAG, SAME TWO HOMES. *** INTFER_FUSED_GATE_UP is not a flag the
# operator sets -- it is the DEFAULT TP=2 MLP path -- but it is exactly the
# same audit-#3 shape: a fusion whose output is bit-identical to the wave
# fallback it replaces, so "the two runs matched" certifies nothing unless
# something proves the fusion RAN. It goes through the same two homes
# (decode1_path_fired for the predicate, decode1_arm_assert for the arm) so
# it cannot drift from the rule the other three flags are held to.
gu_path_fired() { decode1_path_fired "$1" 'mlp-gateup'; }
gu_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'mlp-gateup' 'FUSED_GATE_UP'; }
# Third flag through the SAME two homes. INTFER_DECODE1_SUBGROUP is a
# reassociation of the same block-cyclic body, so its arm must be held to
# the identical rule -- prefill, ids, TP=sharded and a path that announces
# itself. A fourth copy of any of this is the copy-drift family.
sg_path_fired()  { decode1_path_fired "$1" 'decode1-subgroup'; }

# *** AND THE ARM'S BODY IS ONE FUNCTION TOO, NOT THREE CALL SITES. *** Sharing
# the predicates closed the "weakened predicate" class (measured: M-B/M-C/M-P/M-Q
# all go red). It did NOT close "the real arm stopped CALLING them" -- deleting
# the arm's call sites still leaves the self-check green (measured: M-A/M-G/M-F),
# because the self-check drives the predicates and never walks the arm's
# plumbing. So the three assertions live HERE, in one function that takes the two
# logs, and BOTH consumers call it: the real arm on the real logs, the self-check
# on fixtures. Deleting a call site now means deleting the call to this function,
# which the arm cannot do without also losing its only source of assertions.
#
# Returns 0 = the arm is certified. Prints a GATE FAIL line per failed
# assertion so a real failure names itself, and honours MR_DIAG for the
# self-check's per-fixture message prefix (empty in the real arm).
decode1_arm_assert() {   # $1 = base log, $2 = flag-run log, $3 = prefix,
                         # $4 = announcement tag, $5 = flag label.
  local d="${3:-}" got p i ok=0
  local tag="$4" lbl="$5"
  p="$(cli_prefill "$2")"
  i="$(cli_ids "$2" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  got="$(cli_ids "$1" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  mr_prefill_ok "$p" \
    || { echo "${d}${lbl} prefill '$p' != 11751" >&2; ok=1; }
  # NON-EMPTY FIRST lives inside mr_streams_agree: this arm compares two DERIVED
  # values (every other arm here compares to a literal oracle), so a log-format
  # change or broken pipe that emptied BOTH sides would make "" = "" and report
  # PASS having certified nothing.
  mr_streams_agree "$got" "$i" \
    || { echo "${d}${lbl} ids '$i' != base '$got' (or a stream was EMPTY)" >&2; ok=1; }
  mr_tp_sharded "$2" \
    || { echo "${d}${lbl} run was not TP=sharded (needs 2 devices)" >&2; ok=1; }
  # FOURTH, and it is the one that decides whether the other three mean anything:
  # a flag that never reached the kernel makes BOTH runs produce the same tokens
  # for uninteresting reasons, so every comparison above would pass while
  # certifying nothing (audit #3's nested-flag vacuity). Inside this function
  # rather than at the call site, so every arm keeps ONE assertion entry point.
  decode1_path_fired "$2" "$tag" \
    || { echo "${d}${lbl} kernel never announced itself -- the flag did not" >&2
         echo "${d}  reach the M=1 INT2 branch, so this arm compared base against" >&2
         echo "${d}  base and certified nothing." >&2
         ok=1; }
  return "$ok"
}
mr_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-mr' 'DECODE1_MR'; }
sk_arm_assert()  { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-splitk' 'DECODE1_SPLITK'; }
# One-line wrappers over the SHARED decode1_arm_assert, for the same reason the MR
# and SPLITK wrappers are: a per-arm copy of the predicate is what made three of
# the MR arm's assertions unobservable, and this gate has been rebuilt around that
# twice. The 4th argument is the ANNOUNCEMENT TAG the kernel prints and the 5th the
# label used in messages -- so a new arm cannot accidentally assert a different
# flag's announcement, which would be green for the wrong reason.
rp_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-rowpair' 'DECODE1_ROWPAIR'; }
vh_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-verbhalf' 'DECODE1_VERBHALF'; }
vm_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-verbmp' 'DECODE1_VERBMP'; }
vd_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-verbdead' 'DECODE1_VERBDEAD'; }
vs_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-verbsel' 'DECODE1_VERBSEL'; }
vv_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-verbvec' 'DECODE1_VERBVEC'; }
wv_arm_assert() { decode1_arm_assert "$1" "$2" "${3:-}" 'decode1-wvec' 'DECODE1_WVEC'; }
# The change that added the SUBGROUP arm DEFINED THIS TWICE -- once here and
# once above beside rp_arm_assert -- and RAN the arm twice, so the gate paid
# two full 27B decode runs of one flag in its most expensive operation. Bash
# takes the last definition silently and the bodies were identical, so nothing
# looked wrong; that is the whole reason it survived. The duplicate bodies are
# gone and there is now ONE definition, shared by the two self-check cases and
# the one real arm. Counts asserted below rather than trusted.
# NOTE the bracket form, and the emptiness guard. `grep -cE '^removed_with_subgroup_assert\(\)'`
# looks right and is NOT: in an ERE `\(` opens a GROUP, so the pattern is an
# unterminated group, grep writes the error to stderr and NOTHING to stdout, and
# `|| true` turns that into an empty value -- which then fails the very
# `[ "$x" -ne 1 ]` below with "[: : integer expected". Two separate lessons: write
# literal parens as `[(][)]` under -E, and never let a count command reach an
# integer test without proving it produced a NUMBER.
#
# *** AND THE THIRD LESSON, WHICH IS WHY THIS BLOCK IS BEING REWRITTEN AGAIN. ***
# MEASURED on a healthy box, 2026-10-05: the gate printed
#   scripts/test_tp2_27b.sh: line 187: [: : integer expected
# as its VERY FIRST line, and then went on to run every arm. Reading the code
# rather than the symptom is what located it -- the counts are assigned to
# `removed_with_subgroup_assert_defs` and `sg_arm_runs`, but line 187 TESTED `sg_arm_defs`, a
# name nothing ever assigns. So the emptiness guards above were protecting a
# variable the test never read, and the test read an always-unset one. The
# previous fix for this exact symptom ("the empty here is not even a zero") had
# hardened the wrong operand and so could not have caught it.
#
# `count_is_1` below is written so that class cannot recur: it validates that the
# value is NUMERIC first, and names the label in the failure, so a mismatch says
# which count is wrong instead of printing an arithmetic complaint.

# removed_with_subgroup_assert() was DEFINED TWICE by the change that added the SUBGROUP arm --
# once here and once above with rp_arm_assert. Bash takes the last one silently,
# the bodies are identical today, so it was invisible; but this file exists
# because a duplicated assertion is free to drift from the one it shadows, and
# the shadow is the copy that gets edited. One definition, asserted below.

# --- SELF-CHECK (stubbed; no GPU, no model, no build) -----------------------
# *** WHY THE DECODE1_MR ARM NEEDS ONE. *** That arm compares two DERIVED values
# (MR_IDS vs GOT), so an empty stream on BOTH sides compares equal and the arm
# reports PASS having certified nothing -- the empty-stream vacuity family, and a
# NEW instance of it introduced by this very arm. The real run is the only thing
# that can catch a log-format change, but the real run needs the box and an 18 GiB
# model load, which is exactly the check nobody re-runs. These cases are seconds.
# FOUR, not one: a case that must PASS is what makes the cases that must FAIL
# meaningful -- a comparison that rejected everything would satisfy those three
# alone, and this gate's whole history is flags that were vacuously green.
#
# MEASURED NON-VACUITY: seven mutations applied one at a time to a temp copy, SIX
# caught and each naming its OWN check (M1/M4 guard accepts a divergence -> (3);
# M2 guard accepts an empty pair -> (4); M3 parse truncated to the oracle prefix
# -> (1); M5 parser returns nothing -> (1); M7 healthy pair inverted -> (2)).
# *** ONE STATE IS EXCLUDED, NOT SCORED. *** M6 -- weakening check (1)'s literal
# expectation to a prefix test -- is UNOBSERVABLE on its own: the two streams
# still differ, so check (3) still refuses and nothing turns red. It costs
# diagnostic precision, not soundness, and per AUDIT #21 an unobservable state
# is excluded with its reason rather than counted. Do NOT read "proven" as
# "every possible weakening is seen".
if [ "${SELFCHECK:-0}" = 1 ]; then
  SB="$(mktemp -d)"; trap 'rm -rf "$SB"' EXIT INT TERM
  bad=0
  ids_after_prefill() { cli_ids "$1" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//'; }
  TP_BANNER='[intfer] GPU backend: 2 device(s) (TP=sharded)'
  # *** ONE LITERAL, READ BY THE FIXTURE BUILDER AND BY THE NEGATIVE CASES. ***
  # MR_ANNOUNCE is what quant_gemm.cpp prints when the M=1 INT2 branch runs. A
  # second copy of it here would be a second home, and the case that removes it
  # would silently stop removing anything. The kernel's own text is the contract:
  # rename it there and these fixtures stop matching, so (6) goes red rather than
  # going quietly vacuous.
  MR_ANNOUNCE='[decode1-mr] M=1 INT2 decode: MR=4 kernel ACTIVE (N=2048 K=1024 bits=2)'
  # Same one-literal rule for the second flag: this text IS the contract, so a
  # rename in quant_gemm.cpp turns the SPLITK cases red instead of vacuous.
  SK_ANNOUNCE='[decode1-splitk] M=1 INT2 decode: split-K S=16 ACTIVE (N=2048 K=1024 bits=2)'
  # Third flag, same one-literal rule: the kernel's own text is the contract, so
  # a rename in quant_gemm.cpp turns these fixtures red instead of vacuous.
  SG_ANNOUNCE='[decode1-subgroup] M=1 INT2 decode: sub-group-per-row reduce_over_group ACTIVE (N=2048 K=1024 bits=2)'
  # *** ONE FIXTURE BUILDER FOR BOTH FLAGS. *** $ANNOUNCE_TEXT selects which
  # flag's announcement the log carries; it defaults to the MR one so every
  # existing case is unchanged. A second builder for SPLITK would be the audit
  # #3 shape, and the case that matters -- "healthy pair" vs "healthy pair minus
  # one line" -- only isolates its defect if both come from the same builder.
  ANNOUNCE_TEXT="$MR_ANNOUNCE"
  mklog() { # $1 = path, $2.. = ids in step order. Appends the banner and the
            # announcement named by $ANNOUNCE_TEXT, so every fixture is a
            # COMPLETE, HEALTHY-SHAPED log differing in exactly one way.
    local p="$1"; shift; : >"$p"; local i=0
    for t in "$@"; do echo "decode step $i: token=$t" >>"$p"; i=$((i+1)); done
    printf '%s\n' "$TP_BANNER" >>"$p"
    printf '%s\n' "$ANNOUNCE_TEXT" >>"$p"
  }
  # *** THESE DRIVE THE REAL ARM'S PREDICATES (mr_prefill_ok /
  # mr_streams_agree / mr_tp_sharded above), NOT COPIES OF THEM. *** There is no
  # local arm_ok any more, because a local one is exactly what made three of the
  # arm's assertions unobservable (see the predicates' comment block).
  mklog "$SB/good_base" 11751 13 198 760 6511
  # *** EVERY FIXTURE IS A COMPLETE, HEALTHY-SHAPED LOG, DIFFERING IN EXACTLY ONE
  # THING. *** MEASURED: the first version of these cases built MR logs with ids
  # but no TP banner, so mr_arm_assert refused each one for TWO reasons at once
  # (the intended one plus a missing banner). A case that cannot isolate its
  # defect is the audit #3 lesson -- the mutation passes for the wrong reason and
  # teaches nothing. mklog appends the banner the real CLI prints, and the one
  # TP fixture below is built by OVERWRITING that banner, so it differs only in
  # the banner.
  mklog "$SB/good_mr"   11751 13 198 760 6511
  mklog "$SB/bad_mr"    11751 13 198 760 9999   # diverges at step 4 only
  mklog "$SB/bad_prefill" 4242 13 198 760 6511  # wrong prefill only
  mklog "$SB/tp_ok_ids" 11751 13 198 760 6511   # healthy pair for (2)
  mklog "$SB/tp_bad_ids" 11751 13 198 760 6511  # same ids, banner replaced below
  sed -i 's/TP=sharded/TP=single/' "$SB/tp_bad_ids"   # ONE difference vs tp_ok_ids
  printf '%s\n' "[intfer] GPU backend: 1 device(s) (TP=single)" >"$SB/tp_bad"
  : >"$SB/empty"
  a_got="$(ids_after_prefill "$SB/good_base")"
  a_mr="$(ids_after_prefill "$SB/good_mr")"
  b_got="$(ids_after_prefill "$SB/good_base")"
  b_mr="$(ids_after_prefill "$SB/bad_mr")"
  c_got="$(ids_after_prefill "$SB/empty")"

  # (1) FIXTURE INTEGRITY. MEASURED, NOT ASSUMED: the first version of this
  # self-check compared `b_mr` against `b_got` and nothing checked that the
  # streams were FULL LENGTH, so a mutation that truncated the parse to the
  # 3-token oracle prefix still "passed" -- because a truncated bad stream
  # differs from a full base stream for the wrong reason. A case that cannot
  # distinguish the defect from a shorter string is not a case. Asserted, not
  # commented: 4 ids each, differing ONLY in the last, and an empty log empty.
  [ "$b_got" = "13 198 760 6511" ] || { echo "SELF-CHECK FAIL (1): base stream must be exactly '13 198 760 6511', got '$b_got'" >&2; bad=1; }
  [ "$b_mr" = "13 198 760 9999" ]  || { echo "SELF-CHECK FAIL (1): bad stream must be exactly '13 198 760 9999', got '$b_mr'" >&2; bad=1; }
  [ -z "$c_got" ] || { echo "SELF-CHECK FAIL (1): an empty log must parse EMPTY, got '$c_got'" >&2; bad=1; }

  # (2) The healthy direction -- THE CONVERSE. Without it, every refusal below
  #     is satisfied by a predicate that rejects everything, which is why this is
  #     not optional. (Measured lesson, this repo: an always-red check proves
  #     only that a check can fail.)
  [ -n "$a_got" ] && [ -n "$a_mr" ] && [ "$a_got" = "$a_mr" ] \
    || { echo "SELF-CHECK FAIL (2): identical streams must be equal and non-empty, got '$a_got' vs '$a_mr'" >&2; bad=1; }
  mr_prefill_ok "$(cli_prefill "$SB/good_mr")" \
    || { echo "SELF-CHECK FAIL (2): mr_prefill_ok must ACCEPT the oracle prefill" >&2; bad=1; }
  mr_tp_sharded "$SB/tp_ok_ids" \
    || { echo "SELF-CHECK FAIL (2): mr_tp_sharded must ACCEPT a TP=sharded banner" >&2; bad=1; }
  # *** THE ARM'S ENTRY POINT MUST ACCEPT A GOOD PAIR. *** Without this, (3)/(4)/(5)
  # are all satisfied by an mr_arm_assert that refuses EVERYTHING -- and this
  # repo's standing lesson is that an always-red check proves only that a check
  # can fail. The fixtures here are good ids AND a sharded banner, so a refusal
  # can only come from the arm.
  mr_arm_assert "$SB/tp_ok_ids" "$SB/tp_ok_ids" "SELFTEST (2) arm-refusal: " \
    || { echo "SELF-CHECK FAIL (2): mr_arm_assert must ACCEPT a healthy pair" >&2; bad=1; }

  # (3) A divergence BEYOND the oracle prefix must be refused. The oracle pins 3
  #     tokens, so step 4 is exactly what a prefix-only assertion cannot see --
  #     which is why (1) is a precondition for this meaning anything.
  # *** THROUGH mr_arm_assert, THE ARM'S OWN ENTRY POINT. *** These cases used to
  # call the bare predicates, which covered "the predicate was weakened" but NOT
  # "the arm stopped calling it" -- MEASURED: deleting the arm's three call sites
  # left --self-check at rc=0 PASS. Driving the entry point the real arm uses is
  # what makes that observable: if the arm's assertions are deleted, mr_arm_assert
  # is no longer the arm's source of truth and these cases are the only thing left
  # saying what it must contain. The predicates are still driven directly in (2),
  # because that is the half that names WHICH assertion failed.
  mr_arm_assert "$SB/good_base" "$SB/bad_mr" "SELFTEST (3) arm-refusal: " \
    && { echo "SELF-CHECK FAIL (3): the arm ACCEPTED a step-4 divergence" >&2; bad=1; }
  # Direct drive, for the same reason (4) has two: mr_arm_assert refuses this
  # fixture on the stream check ALONE (the ids differ and the prefill/banner are
  # healthy), so the end-to-end case already isolates it -- but if a future
  # fixture ever makes the arm refuse for another reason too, only the direct
  # drive still names this defect. Two subjects, not two copies.
  mr_streams_agree "$(ids_after_prefill "$SB/good_base")" "$(ids_after_prefill "$SB/bad_mr")" \
    && { echo "SELF-CHECK FAIL (3): mr_streams_agree ACCEPTED a step-4 divergence directly" >&2; bad=1; }
  mr_arm_assert "$SB/good_base" "$SB/bad_prefill" "SELFTEST (5) arm-refusal: " \
    && { echo "SELF-CHECK FAIL (5): the arm ACCEPTED a wrong prefill (4242)" >&2; bad=1; }
  mr_arm_assert "$SB/good_base" "$SB/tp_bad_ids" "SELFTEST (5) arm-refusal: " \
    && { echo "SELF-CHECK FAIL (5): the arm ACCEPTED a TP=single banner" >&2; bad=1; }

  # (4) An empty pair must be refused -- the vacuity this arm could have shipped
  #     with. BOTH sides empty, which is the only way the derived-vs-derived
  #     comparison in mr_streams_agree can be tricked into agreeing.
  # *** TWO CASES, AND THE SECOND IS THE ISOLATED ONE. *** MEASURED: driving only
  # mr_arm_assert here was a MASKED assertion. mr_arm_assert checks three things,
  # so an empty pair already fails its prefill and TP checks, and the case went
  # red for THOSE reasons even when mr_streams_agree had its non-empty half
  # deleted -- a weakening of the exact predicate this case exists to protect,
  # invisible (audit #12's mutual masking, in a guard I was writing to add
  # coverage). The second case drives mr_streams_agree DIRECTLY with an empty
  # pair, so nothing else can refuse on its behalf. Both are kept: the first is
  # the end-to-end shape the real arm uses, the second is the one that is
  # load-bearing. A redundant-looking pair here is two DIFFERENT subjects.
  mr_arm_assert "$SB/empty" "$SB/empty" "SELFTEST (4) arm-refusal: " \
    && { echo "SELF-CHECK FAIL (4): the arm ACCEPTED an empty pair (vacuity)" >&2; bad=1; }
  mr_streams_agree "$(ids_after_prefill "$SB/empty")" "$(ids_after_prefill "$SB/empty")" \
    && { echo "SELF-CHECK FAIL (4): mr_streams_agree ACCEPTED an empty pair directly" >&2; bad=1; }

  # (5) The two predicates, driven DIRECTLY so a failure names which one, in both
  #     directions -- (2) accepts, these refuse. Through mr_arm_assert alone, a
  #     defect in mr_prefill_ok and one in mr_tp_sharded would print the same line.
  mr_prefill_ok "$(cli_prefill "$SB/bad_prefill")" \
    && { echo "SELF-CHECK FAIL (5): mr_prefill_ok ACCEPTED a wrong prefill (4242)" >&2; bad=1; }
  mr_prefill_ok "$(cli_prefill "$SB/empty")" \
    && { echo "SELF-CHECK FAIL (5): mr_prefill_ok ACCEPTED an EMPTY prefill (vacuity)" >&2; bad=1; }
  mr_tp_sharded "$SB/tp_bad" \
    && { echo "SELF-CHECK FAIL (5): mr_tp_sharded ACCEPTED a TP=single banner" >&2; bad=1; }
  mr_tp_sharded "$SB/empty" \
    && { echo "SELF-CHECK FAIL (5): mr_tp_sharded ACCEPTED an EMPTY log (vacuity)" >&2; bad=1; }

  # (6) THE SUBJECT-PRESENT QUESTION, which the stream comparisons cannot see.
  #     Both directions, and the healthy direction comes FIRST because an
  #     always-red presence check is worse than none: it would refuse a run whose
  #     MR path demonstrably fired. Fixtures differ from a real log in exactly one
  #     way -- the announcement line -- so a refusal can only be about the line.
  printf '%s\n' "$MR_ANNOUNCE" >"$SB/announced"
  # A COMPLETE, HEALTHY log minus exactly ONE line: same ids, same banner, no
  # announcement. Built by removing the line from a healthy fixture rather than
  # assembled from scratch, which is what guarantees the refusal below is about
  # the announcement and nothing else -- the audit #3 lesson, in a fixture.
  mklog "$SB/not_announced" 11751 13 198 760 6511
  sed -i '/\[decode1-mr\]/d' "$SB/not_announced"
  mr_path_fired "$SB/announced" \
    || { echo "SELF-CHECK FAIL (6): mr_path_fired must ACCEPT the ACTIVE announcement" >&2; bad=1; }
  mr_path_fired "$SB/not_announced" \
    && { echo "SELF-CHECK FAIL (6): mr_path_fired ACCEPTED a log with no announcement" >&2; bad=1; }
  mr_path_fired "$SB/empty" \
    && { echo "SELF-CHECK FAIL (6): mr_path_fired ACCEPTED an EMPTY log (vacuity)" >&2; bad=1; }
  # And through the arm's own entry point, since this assertion is what decides
  # whether the arm's stream comparisons mean anything.
  mr_arm_assert "$SB/tp_ok_ids" "$SB/not_announced" "SELFTEST (6) arm-refusal: " \
    && { echo "SELF-CHECK FAIL (6): mr_arm_assert ACCEPTED a log whose MR path never fired" >&2; bad=1; }

  # (7) THE SAME QUESTION FOR THE SECOND FLAG. INTFER_DECODE1_SPLITK is also
  #     guarded by `bits == 2`, so it inherits the same coverage constraint, and
  #     it reaches it through decode1_path_fired -- one home, two callers. Both
  #     directions again, healthy first, so the pair cannot pass by being
  #     always-red. The fixture is a COMPLETE healthy log minus exactly the
  #     SPLITK announcement line, built by deleting that line from mklog output,
  #     so a refusal can only be about the line.
  ANNOUNCE_TEXT="$SK_ANNOUNCE"
  mklog "$SB/sk_announced"     11751 13 198 760 6511   # complete, healthy
  mklog "$SB/sk_not_announced" 11751 13 198 760 6511   # same, minus the line
  sed -i '/\[decode1-splitk\]/d' "$SB/sk_not_announced"
  ANNOUNCE_TEXT="$MR_ANNOUNCE"
  sk_path_fired "$SB/sk_announced" \
    || { echo "SELF-CHECK FAIL (7): sk_path_fired must ACCEPT the ACTIVE announcement" >&2; bad=1; }
  sk_path_fired "$SB/sk_not_announced" \
    && { echo "SELF-CHECK FAIL (7): sk_path_fired ACCEPTED a log with no announcement" >&2; bad=1; }
  sk_path_fired "$SB/empty" \
    && { echo "SELF-CHECK FAIL (7): sk_path_fired ACCEPTED an EMPTY log (vacuity)" >&2; bad=1; }
  # And through the SPLITK arm's own entry point.
  sk_arm_assert "$SB/tp_ok_ids" "$SB/sk_not_announced" "SELFTEST (7) arm-refusal: " \
    && { echo "SELF-CHECK FAIL (7): sk_arm_assert ACCEPTED a log whose SPLITK path never fired" >&2; bad=1; }
  # And the healthy direction through the arm: a SPLITK log that DID announce
  # must certify. Without this, case (7) would be satisfiable by a predicate
  # that refuses everything -- the always-red twin.
  sk_arm_assert "$SB/tp_ok_ids" "$SB/sk_announced" "SELFTEST (7) arm-accept: " \
    || { echo "SELF-CHECK FAIL (7): sk_arm_assert REFUSED a healthy SPLITK pair" >&2; bad=1; }
  # Third flag through the same shared assertion. The healthy pair and the
  # healthy-pair-minus-one-line case are what isolate the defect; both come from
  # the SAME builder, so a builder change cannot quietly stop isolating it.
  sed "s|\\[decode1-splitk\\]|[decode1-subgroup]|g" "$SB/sk_announced" >"$SB/sg_announced"
  # DERIVE the negative from the positive, do not author it: a separately
  # written sg_not_announced can drift to be a valid log and the refusal case
  # then passes for the wrong reason. That is exactly what happened on the
  # first version of these two lines -- sg_not_announced was never created
  # at all, sed printed 'No such file or directory', and the arm-refusal
  # case came back green because the MISSING file was not TP=sharded. The  # *** THE SUBGROUP SELF-CHECK CASES ARE GONE WITH THE ARM (2026-10-05). ***
  # They exercised `sg_arm_assert`, which asserted the arm that produced the wrong
  # answer. An arm that cannot pass is not a check: keeping it made this gate, then
  # check_device_precond.sh, then run_all_checks.sh red, in that order. The two
  # cases below did their job -- they were green while the arm was red, which is
  # itself the lesson: a predicate can be exercised to perfection and still be
  # asserting something false, because a predicate tests the SHAPE of a claim and
  # not its TRUTH. Only the token oracle in the real run tests truth, and that is
  # what caught this. See AGENTS.md under "INTFER_DECODE1_SUBGROUP is
  # QUARANTINED, not fixed".
  # CROSS-FLAG: an MR announcement must NOT satisfy the SPLITK predicate and vice
  # versa. Without this, a single tag-parameterized grep could match the wrong
  # flag's line and the two arms would certify each other's absence.
  sk_path_fired "$SB/announced" \
    && { echo "SELF-CHECK FAIL (7): sk_path_fired matched the MR announcement" >&2; bad=1; }
  mr_path_fired "$SB/sk_announced" \
    && { echo "SELF-CHECK FAIL (7): mr_path_fired matched the SPLITK announcement" >&2; bad=1; }

  if [ "$bad" -eq 0 ]; then
    echo "TP2-27B SELF-CHECK: PASS (1) fixtures are full-length and differ only at step 4  (2) healthy pair accepted by all three predicates  (3) mr_arm_assert refused a late divergence  (4) mr_arm_assert refused an empty pair  (5) wrong-prefill and wrong-TP refusals fire"
    exit 0
  fi
  echo "TP2-27B SELF-CHECK: FAIL" >&2; exit 1
fi

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export TMPDIR=/var/home/fra/build_tmp

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build intfer_cli || exit 1

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
# 2-GPU precondition (AUDIT #12 follow-up). This gate is named for --tp 2, and
# 27B needs ~19GB against a ~12GB B580, so on a 1-GPU box the run clamps to
# TP=1 and aborts inside the Level Zero page-fault manager (AUDIT #11). That is
# a VRAM/box outcome, not a code failure, and reporting it as FAIL is what made
# AGENTS.md carry a standing red item that was never a regression. Exit 2 = SKIP.
. "$SCRIPT_DIR/_gate_precond.sh"
# *** BOTH preconditions, and the count-only form was WRONG once the second GPU
# came back. *** gate_require_gpus only COUNTS devices; it never opens the render
# node. AUDIT #16 noted that for a ">= 2 GPUs" requirement both failure modes
# collapse to `count < need`, so a node probe looked redundant -- but that only
# holds on a 1-GPU box, where the count was already 0. With BOTH GPUs
# enumerable, sycl-ls still reports 2 even when /dev/dri/renderD128 is
# unreachable, so gate_require_gpus 2 PASSES and the gate proceeds and exits 0
# with no device. Caught by check_device_precond.sh's node-unreachable direction,
# which expects SKIP (2). The device precondition is now checked as the DEVICE
# it is: node first, then the count.
gate_require_device || exit 2
gate_require_gpus 2 || exit 2

LOG="$(mktemp --tmpdir="$TMPDIR" tp2_27b.XXXX.log)"
# Keep the per-run CLI logs next to the wrapper log (not mktemp orphans):
# <wrapper>.out / <wrapper>.err hold the CLI stdout/stderr on gate failure.
CLI_OUT="$LOG.out"; CLI_ERR="$LOG.err"
# Declared HERE, not beside the DECODE1_MR run further down: the cleanup traps are
# armed below and their bodies expand these at trap time, so a definition further
# down leaves every early-exit path expanding an unset variable.
MR_OUT="$LOG.mr.out"; MR_ERR="$LOG.mr.err"
SK_OUT="$LOG.sk.out"; SK_ERR="$LOG.sk.err"
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "--- CLI stdout ($CLI_OUT) ---" >&2; tail -5 "$CLI_OUT" >&2; echo "--- CLI stderr ($CLI_ERR, last 25) ---" >&2; tail -25 "$CLI_ERR" >&2; else rm -f "$CLI_OUT" "$CLI_ERR" "$MR_OUT" "$MR_ERR"; fi' EXIT
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "--- CLI stdout ($CLI_OUT) ---" >&2; tail -5 "$CLI_OUT" >&2; echo "--- CLI stderr ($CLI_ERR, last 25) ---" >&2; tail -25 "$CLI_ERR" >&2; else rm -f "$CLI_OUT" "$CLI_ERR" "$MR_OUT" "$MR_ERR"; fi; exit 130' INT
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "--- CLI stdout ($CLI_OUT) ---" >&2; tail -5 "$CLI_OUT" >&2; echo "--- CLI stderr ($CLI_ERR, last 25) ---" >&2; tail -25 "$CLI_ERR" >&2; else rm -f "$CLI_OUT" "$CLI_ERR" "$MR_OUT" "$MR_ERR"; fi; exit 143' TERM
"$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$CLI_OUT" 2>"$CLI_ERR"; RC=$?
cat "$CLI_OUT"
if [ $RC -ne 0 ]; then
  echo "FAIL: intfer_cli exited rc=$RC" >&2
  tail -20 "$CLI_ERR" >&2
  exit 1
fi

IDS="$(cli_ids "$CLI_ERR")"
PREFILL_OK=0
grep -aq "decode step 0: token=11751" "$CLI_ERR" && PREFILL_OK=1
EXPECT="13 198 760"
GOT="$(echo "$IDS" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
GOT_PREFIX="$(echo "$GOT" | cut -d' ' -f1-3)"
echo "---"
echo "prefill token: $(cli_prefill "$CLI_ERR") (expect 11751)"
echo "decode ids:    $GOT (expect prefix: $EXPECT)"
grep -a -E "prefill:|decode:" "$CLI_ERR" | tail -2
grep -a -A11 "GPU profile" "$CLI_ERR" | tail -12

PASS=1
[ "$PREFILL_OK" = 1 ] || { echo "GATE FAIL: prefill != 11751" >&2; PASS=0; }
[ "$GOT_PREFIX" = "$EXPECT" ] || { echo "GATE FAIL: decode prefix '$GOT_PREFIX' != '$EXPECT'" >&2; PASS=0; }
TP_LINE="$(grep -a "GPU backend:" "$CLI_ERR" | head -1)"
echo "$TP_LINE"
echo "$TP_LINE" | grep -q "TP=sharded" || { echo "GATE FAIL: TP not sharded (need 2 devices)" >&2; PASS=0; }

# --- INTFER_DECODE1_MR certification ---------------------------------------
# *** WHY THIS ARM IS HERE AND NOT IN test_gpu_optin.sh. *** DECODE1_MR is
# guarded by `bits == 2` in the M=1 decode GEMM, and the scheme census of the two
# gate models is decisive:
#     Qwen3.5-0.8B.intifer   INT4 x76, FP16 x93, INT8 x18   <- ZERO INT2
#     Qwen3.8-27B-int2.intifer  INT4 x116, FP16 x256, INT2 x110, INT8 x16
# So NO 0.8B gate can reach that branch, and an arm in test_gpu_optin.sh would be
# green for the same reason every nested-flag arm used to be green: because the
# path never ran (audit #3). This gate is the only one whose model has an INT2
# tensor, so it is the only place the claim can be tested at all.
#
# The claim under test is that MR keeps each output's K accumulation order
# (ascending kg, then ascending j) in its own register, so it is bit-exact. It
# was asserted by code reading, which is an argument, not evidence. Here it is
# checked against the SAME FIXED ORACLE the base run already uses -- not a
# self-comparison, so "both runs broke the same way" cannot pass.
#
# *** BOTH DEFAULTS MUST BE KILLED, OR THE ARM NEVER RUNS. *** VERBHALF
# and VERBMP are default-ON (promoted 2026-10-07) and are checked
# EARLIER in quant_gemm_forward_usm than the MR branch, so with the
# defaults live VERBHALF wins the run and `[decode1-mr]` never announces
# -- the arm fails its own path-fired assertion while comparing two runs
# that both ran verbhalf. Killing both defaults is what selects MR,
# exactly as it selects VERBMP and WVEC below.
env INTFER_NO_DECODE1_VERBHALF=1 INTFER_NO_DECODE1_VERBMP=1 \
  INTFER_DECODE1_MR=1 "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$MR_OUT" 2>"$MR_ERR"; MR_RC=$?
if [ "$MR_RC" -ne 0 ]; then
  echo "GATE FAIL: INTFER_DECODE1_MR=1 run exited rc=$MR_RC" >&2
  tail -20 "$MR_ERR" >&2
  PASS=0
else
  echo "DECODE1_MR prefill: $(cli_prefill "$MR_ERR") (expect 11751)"
  echo "DECODE1_MR ids:      $(cli_ids "$MR_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  # *** ONE CALL, AND IT IS THE ARM'S ONLY SOURCE OF ASSERTIONS. *** These three
  # assertions used to be inline here with the self-check holding a separate copy
  # of one; MEASURED on that version, deleting any of these lines left
  # --self-check at rc=0 PASS. They now live in mr_arm_assert, which the
  # self-check also calls, so this arm has nothing to inline and nothing to keep
  # in sync. Do not inline anything here without adding a self-check case.
  mr_arm_assert "$CLI_ERR" "$MR_ERR" "GATE FAIL: " || PASS=0
  # *** AND THE PATH MUST HAVE ACTUALLY FIRED. *** Everything above compares the
  # MR run's stream against the base run's stream. If the flag never reached the
  # M=1 INT2 decode kernel -- renamed, typo'd, stripped by a shell wrapper, or
  # simply no `bits == 2` tensor occurring in this model's decode path -- then the
  # two runs are IDENTICAL BY CONSTRUCTION and every assertion above passes
  # having certified nothing. That is audit #3's nested-flag vacuity, and no
  # amount of stream comparison can see it: the subject is the flag's EFFECT, not
  # the stream. The kernel announces itself once (quant_gemm.cpp,
  # "[decode1-mr] ... ACTIVE"); asserting that line is what turns "the two runs
  # matched" into "the MR path ran and produced the same tokens".
  # MEASURED as MISSED before this existed: deleting the arm's call to
  # mr_arm_assert entirely left --self-check green, and a misspelled flag would
  # have done the same. The presence check itself is mr_arm_assert's FOURTH
  # assertion, not an inline test here -- one entry point, no second site to keep
  # in sync. What is left inline is a single echo, so a PASS log records WHICH
  # path ran rather than only that it agreed.
  grep -a '\[decode1-mr\]' "$MR_ERR" | head -1
fi

# --- INTFER_DECODE1_SPLITK certification ----------------------------------
# *** SAME COVERAGE CONSTRAINT AS DECODE1_MR, AND THE SAME ARM FOR IT. *** The
# split-K branch is guarded by `bits == 2` exactly like the MR branch, and the
# scheme census is unchanged: Qwen3.5-0.8B.intifer has ZERO INT2 tensors, so no
# 0.8B gate can reach it and an arm there would be green because the path never
# ran. This gate is the only place the claim can be tested.
#
# *** BUT THE CLAIM IS WEAKER, AND THE ARM MUST NOT OVERSTATE IT. *** MR is
# bit-exact by construction. Split-K is NOT: it reassociates the FP32 sum into S
# partials combined ascending. Its tolerance is measured, in
# examples/ksplit_contract (gated by scripts/test_ksplit_contract.sh): 0 FP16
# ulps of margin consumed on realistic activations at every K tested, 2 ulps on
# an adversarial escalating dataset. So what this arm certifies is
# "stream-identical ON THIS MODEL against the fixed oracle" -- NOT bit-exactness
# for all inputs, which is false and which the source comment refuses to claim.
# An arm cannot be stronger than the contract it enforces; a green arm here means
# the contract holds for the gate model, not that the kernel is exact.
#
# *** BOTH DEFAULTS MUST BE KILLED, OR THE ARM NEVER RUNS *** -- same
# shadowing as the MR arm above: VERBHALF/VERBMP are default-ON and
# checked earlier than the SPLITK branch, so the defaults must die to
# select split-K, or `[decode1-splitk]` never announces and the arm
# fails its own path-fired assertion.
env INTFER_NO_DECODE1_VERBHALF=1 INTFER_NO_DECODE1_VERBMP=1 \
  INTFER_DECODE1_SPLITK=1 "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$SK_OUT" 2>"$SK_ERR"; SK_RC=$?
if [ "$SK_RC" -ne 0 ]; then
  echo "GATE FAIL: INTFER_DECODE1_SPLITK=1 run exited rc=$SK_RC" >&2
  tail -20 "$SK_ERR" >&2
  PASS=0
else
  echo "DECODE1_SPLITK prefill: $(cli_prefill "$SK_ERR") (expect 11751)"
  echo "DECODE1_SPLITK ids:      $(cli_ids "$SK_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  # Same single entry point as the MR arm, so both flags are held to one rule.
  # If SPLITK had its own inline copies, the second copy would be free to drift
  # untested -- the shape this file has been rebuilt around twice.
  sk_arm_assert "$CLI_ERR" "$SK_ERR" "GATE FAIL: " || PASS=0
  grep -a '\[decode1-splitk\]' "$SK_ERR" | head -1
fi
# --- (the SUBGROUP arm below is the only one; a change once defined it twice
# here and again above, and RAN it twice -- two full 27B decode runs of the
# same flag in the most expensive operation this gate has. The identical
# bodies made it invisible and harmless-looking, which is exactly why it
# survived; the cost was silent and the drift risk was one edit away.)
# --- INTFER_DECODE1_ROWPAIR certification --------------------------------------
# *** SAME REACHABILITY CONSTRAINT AS MR AND SPLITK, AND IT IS BINDING AGAIN. ***
# The row-pair branch is guarded by `bits == 2`, and Qwen3.5-0.8B.intifer has ZERO
# INT2 tensors, so a test_gpu_optin.sh arm would be green because the path never
# ran. This gate's model has 110 INT2 tensors.
#
# *** AND THE CLAIM IS STRONGER THAN SPLITK'S, WHICH IS WHY THE ARM MAY SAY SO. ***
# Row-pairing gives ONE work-item two output rows, each with its OWN accumulator,
# its own ascending-kg/ascending-j order and its own codebook. The two chains are
# interleaved in the INSTRUCTION stream, never summed together, so no
# floating-point value is ever combined across rows: for each output the addition
# sequence is the base kernel's sequence. That is bit-exact BY CONSTRUCTION, so --
# unlike SPLITK -- it needs no numerics contract and this arm can assert full
# stream identity without overstating anything.
#
# What it does NOT claim, stated here rather than left implied: that the kernel is
# faster. examples/roofline_probe measured that a wider code chunk is 0.25-0.35x
# (a 3-4x regression), so a prior exists that MORE per-work-item work is the wrong
# direction. Row-pairing's argument is different -- it keeps the work-item count
# and doubles bytes-in-flight -- but the prior is why this arm's RESULT is left
# blank and why the report states the expected direction as UNKNOWN rather than
# positive.
RP_OUT="$LOG.rp.out"; RP_ERR="$LOG.rp.err"
env INTFER_DECODE1_ROWPAIR=1 "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$RP_OUT" 2>"$RP_ERR"; RP_RC=$?
if [ "$RP_RC" -ne 0 ]; then
  echo "GATE FAIL: INTFER_DECODE1_ROWPAIR=1 run exited rc=$RP_RC" >&2
  tail -20 "$RP_ERR" >&2
  PASS=0
else
  echo "DECODE1_ROWPAIR prefill: $(cli_prefill "$RP_ERR") (expect 11751)"
  echo "DECODE1_ROWPAIR ids:      $(cli_ids "$RP_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  # *** THE PATH MUST HAVE ACTUALLY FIRED. *** Stream identity alone is
  # satisfiable by a flag that did nothing, which is audit #3's nested-flag
  # vacuity. The row-pair kernel prints a one-time line, so asserting it is what
  # turns "the two runs matched" into "the row-pair path ran and produced the same
  # tokens".
  if ! grep -aq '\[decode1-rowpair\]' "$RP_ERR"; then
    echo "GATE FAIL: INTFER_DECODE1_ROWPAIR produced the right tokens without the" >&2
    echo "        path ever running (no [decode1-rowpair] line). The arm would" >&2
    echo "        be vacuous." >&2
    PASS=0
  fi
  rp_arm_assert "$CLI_ERR" "$RP_ERR" "GATE FAIL: " || PASS=0
  grep -a '\[decode1-rowpair\]' "$RP_ERR" | head -1
fi

# --- INTFER_DECODE1_SUBGROUP: THE ARM IS GONE, AND THE REASON IS A WRONG ANSWER --
# It ran, announced itself, and produced ids '279 12 12' where the base produces
# '13 198 760' -- deterministically, on every run. A certification arm that
# deterministically fails must not exist, so the flag, its kernel and this arm were
# all removed on 2026-10-05. The bisect so far -- including the two candidate causes
# that were TESTED AND FALSIFIED -- is in AGENTS.md under "INTFER_DECODE1_SUBGROUP
# is QUARANTINED, not fixed".
#
# The removal is why this gate can be green again. It is NOT a relaxation: an arm
# that cannot pass is not a check, and leaving it in made test_tp2_27b.sh, then
# check_device_precond.sh, then run_all_checks.sh red, in that order.
#
# Reinstating it needs a MODEL-FREE differential first (SUBGROUP vs the base M=1
# body, bitwise, over shapes and datasets) so the fault is localised WITHOUT the
# 27B stack. scripts/test_rowpair_equiv.sh is the shape of that tooling and it is
# what found the ROWPAIR device-loss; it does not exist for SUBGROUP yet.

# --- INTFER_DECODE1_VERBHALF / VERBMP: DEFAULT-ON as of 2026-10-07 ---------------
# *** THE DEFAULTS CHANGED, AND THE BASE RUN CARRIES THEM. *** Both arms are now
# default-ON kill switches (feature runs unless INTFER_NO_DECODE1_VERBHALF=1 /
# =1 is set), so the BASE run at line 488 -- which sets no env -- runs THIS path by
# default. That is correct and intended: the base run IS the default decode path,
# and the fixed oracle `13 198 760` is measured against it. The arms below
# prove each arm fires via its ANNOUNCEMENT, not via the env spelling: the
# promotion made both arms kill-switch governed, so the old presence flags
# `INTFER_DECODE1_VERBHALF=1` / `INTFER_DECODE1_VERBMP=1` are DEAD -- no
# code reads them -- and a gate that only compared tokens would certify a
# default-ON arm vacuously. The announcement check below is the predicate.
# The MR / SPLITK / ROWPAIR / VERBDEAD / VERBSEL / VERBVEC arms still compare
# their stream against this base run, which is the reference by construction.
#
# *** SAME REACHABILITY CONSTRAINT AS EVERY OTHER DECODE1 ARM, AND IT IS BINDING:
# both branches are guarded on `bits == 2`, and the 0.8B gate model has ZERO INT2
# tensors, so an arm in test_gpu_optin.sh would be green because the path never ran.
#
# *** THE CLONE MEASUREMENT THIS BLOCK USED TO QUOTE IS RETRACTED. ***
# examples/roofline_probe_bisect reported PASS while every arm measured
# 0.000 ms / inf GB/s / -nanx (commit 5bfd8dd), so the +14% / +17% /
# 1.01x ratios were 0/0. The performance direction of BOTH arms is
# UNKNOWN -- the engine arms below are the only way to measure it on
# real 27B weights (operator-only, standing rule 1), and they assert
# stream identity, not throughput. What IS measured: VERBMP is
# bit-exact by construction (test_rowpair_equiv.sh, model-free);
# VERBHALF is stream-identity only (the FTZ denormal is a should-not).
#
# THE CLAIMS DIFFER, AND THE ARMS MUST NOT OVERSTATE:
#   VERBMP  is bit-exact BY CONSTRUCTION. The arithmetic expressions and the index
#           VALUES are unchanged; only the pointer TYPE the compiler folds them
#           through differs. So stream identity is the whole claim and no numerics
#           contract applies. Clone measurement: neutral. scripts/test_rowpair_equiv
#           measures it model-free (227 835 element values, 8 shapes x 5 datasets).
#   VERBHALF is NOT bit-exact by construction: from_half_bits returns 0 for a zero
#           or subnormal exponent and Inf/NaN when saturated, while bit_cast<half>
#           hands a denormal to the FP32 convert, which flushes to zero under the
#           default FTZ. A zero group scale is the differing case and should not
#           arise from `to_half_bits` of a positive host float -- but "should not" is
#           an argument, so this arm is stream-identity only and NOT an arithmetic
#           claim, in the same class as INTFER_DECODE1_SPLITK. scripts/test_rowpair
#           _equiv.sh reports it, does not gate it.
# *** THE VERBMP ARM MUST KILL VERBHALF TO REACH ITS PATH. *** The two
# default-ON arms are order-exclusive -- each ends in `return q.submit(...)`
# and verbhalf is checked FIRST -- so with both kill switches unset the
# verbhalf arm wins and `[decode1-verbmp]` never prints. The arm as first
# written could not pass: it set the dead presence flag `INTFER_DECODE1_
# VERBMP=1`, ran verbhalf, and failed its own announcement check. Proven
# model-free at the M=1 INT2 MLP_GATE shape (N=8704 K=5120) by a direct
# quant_gemm_forward_usm call: `INTFER_DECODE1_VERBMP=1` announces
# `decode1-verbhalf`; `INTFER_NO_DECODE1_VERBHALF=1` announces
# `decode1-verbmp`. The same shadowing examples/rowpair_equiv handles
# inside the binary (its order loop sets INTFER_NO_DECODE1_VERBHALF=1
# for the vm order), which is why only this gate was wrong.
# *** THE WVEC ARM IS THE SAME SHAPE, PLUS ITS OWN PRESENCE FLAG --
# AND IT MUST KILL BOTH DEFAULT-ON ARMS, NOT JUST VERBHALF. ***
# INTFER_DECODE1_WVEC is a default-OFF opt-in (PLAN.md 1.7, the
# bit-exact INT2 weight-load forming), and it sits NINTH in the
# dispatch chain. Two default-ON arms precede it -- verbhalf
# (checked first) and verbmp (checked second) -- and each ends in
# `return q.submit(...)`, so killing only verbhalf leaves verbmp
# winning the run (proven model-free: `INTFER_NO_DECODE1_VERBHALF=1
# INTFER_DECODE1_WVEC=1` announces `decode1-verbmp`, not
# `decode1-wvec`). The arm env therefore kills BOTH defaults and
# sets the presence flag. The order-exclusivity guard in
# quant_gemm_forward_usm refuses a run with two arms live, but it
# sits AFTER the four earliest arms, so it can only police the
# later arms among themselves -- it cannot rescue this arm. Its
# bit-exactness is settled model-free by scripts/test_rowpair_
# equiv.sh's `wv` order (whose env kills every non-selected arm,
# both defaults included); this arm asserts the stream identity on
# real 27B weights.
for spec in "VERBHALF:vh:decode1-verbhalf:" \
            "VERBMP:vm:decode1-verbmp:INTFER_NO_DECODE1_VERBHALF=1" \
            "WVEC:wv:decode1-wvec:INTFER_NO_DECODE1_VERBHALF=1 INTFER_NO_DECODE1_VERBMP=1 INTFER_DECODE1_WVEC=1"; do
  NAME="${spec%%:*}"; rest="${spec#*:}"; abbr="${rest%%:*}"
  rest="${rest#*:}"; tag="${rest%%:*}"; arm_env="${rest#*:}"
  OUT="$LOG.${abbr}.out"; ERR="$LOG.${abbr}.err"
  # VERBHALF: the DEFAULT PATH -- the presence flag is dead post-promotion,
  # so the arm runs because it IS the default and the announcement check is
  # what proves it fired. VERBMP: arm_env kills verbhalf so THIS arm is the
  # one that runs (and the dead presence flag is deliberately NOT set).
  # WVEC: the same kill switch PLUS its own presence flag -- it is default
  # OFF, unlike VERBMP -- so arm_env carries TWO assignments.
  RUNENV="${arm_env:-INTFER_DECODE1_${NAME}=1}"
  # RUNENV may hold more than one assignment (WVEC); word-splitting is safe
  # because every element is a controlled NAME=VALUE constant -- no spaces,
  # no glob characters -- inside it.
  env $RUNENV "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
    -p "$PROMPT" -n "$N" >"$OUT" 2>"$ERR"; RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "GATE FAIL: $RUNENV run exited rc=$RC" >&2
    tail -20 "$ERR" >&2
    PASS=0
  else
    echo "DECODE1_${NAME} prefill: $(cli_prefill "$ERR") (expect 11751)"
    echo "DECODE1_${NAME} ids:      $(cli_ids "$ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    if ! grep -aq "\[$tag\]" "$ERR"; then
      echo "GATE FAIL: INTFER_DECODE1_${NAME} produced the right tokens without the" >&2
      echo "        path ever running (no [$tag] line). Vacuous." >&2
      PASS=0
    fi
    "${abbr}_arm_assert" "$CLI_ERR" "$ERR" "GATE FAIL: " || PASS=0
    grep -a "\[$tag\]" "$ERR" | head -1
  fi
done

# --- BISECT ROUND 2: VERBDEAD / VERBSEL / VERBVEC, default OFF -----------------
# *** SAME REACHABILITY CONSTRAINT, STILL BINDING: guarded on `bits == 2`, and the
# 0.8B gate model has ZERO INT2 tensors, so a test_gpu_optin.sh arm would be green
# because the path never ran.
#
# ALL THREE ARE BIT-EXACT BY CONSTRUCTION and therefore claim NO arithmetic
# difference -- VERBDEAD is pure code motion, VERBSEL selects the same four codebook
# VALUES out of named registers instead of an indexed array, and VERBVEC changes the
# activation POINTER TYPE and nothing else. So stream identity is the entire claim
# for all three, and `examples/rowpair_equiv` measures it model-free (227 835
# element values, 8 shapes x 5 datasets) WITHOUT a 27B load. PERFORMANCE is still
# unmeasured for every arm in this batch; see AGENTS.md under the M=1 INT2
# GEMV bisect section.
for spec in "VERBDEAD:vd:decode1-verbdead" "VERBSEL:vs:decode1-verbsel" \
            "VERBVEC:vv:decode1-verbvec"; do
  NAME="${spec%%:*}"; rest="${spec#*:}"; abbr="${rest%%:*}"; tag="${rest#*:}"
  OUT="$LOG.${abbr}.out"; ERR="$LOG.${abbr}.err"
  # *** BOTH DEFAULT-ON ARMS MUST BE KILLED, OR THESE NEVER RUN. ***
  # VERBHALF and VERBMP are default-ON (promoted 2026-10-07) and are
  # checked EARLIER in quant_gemm_forward_usm than these three opt-in
  # arms, so with the defaults live VERBHALF wins the run and
  # `[decode1-verbdead]`/`[decode1-verbsel]`/`[decode1-verbvec]` never
  # announce -- the arm fails its own path-fired assertion while
  # comparing two runs that both ran verbhalf. Same shadowing the MR,
  # SPLITK and WVEC arms already handle; killing both defaults is what
  # selects the opt-in arm.
  env INTFER_NO_DECODE1_VERBHALF=1 INTFER_NO_DECODE1_VERBMP=1 \
    "INTFER_DECODE1_${NAME}=1" "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
    -p "$PROMPT" -n "$N" >"$OUT" 2>"$ERR"; RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "GATE FAIL: INTFER_DECODE1_${NAME}=1 run exited rc=$RC" >&2
    tail -20 "$ERR" >&2
    PASS=0
  else
    echo "DECODE1_${NAME} prefill: $(cli_prefill "$ERR") (expect 11751)"
    echo "DECODE1_${NAME} ids:      $(cli_ids "$ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    if ! grep -aq "\[$tag\]" "$ERR"; then
      echo "GATE FAIL: INTFER_DECODE1_${NAME} produced the right tokens without the" >&2
      echo "        path ever running (no [$tag] line). Vacuous." >&2
      PASS=0
    fi
    "${abbr}_arm_assert" "$CLI_ERR" "$ERR" "GATE FAIL: " || PASS=0
    grep -a "\[$tag\]" "$ERR" | head -1
  fi
done

# --- INTFER_FUSED_GATE_UP: gate+up in a single launch (TP=2 fusion) ------
# 1.2 of PLAN.md. Gate and up are both column-parallel over the SAME split and
# device-local, so one parallel_for over Ng+Nu outputs is bit-exact vs two
# separate quant_gemm_forward_usm calls: each output keeps its own accumulator,
# ascending kg, ascending j, its own group scale and wtab. Saves one submit +
# one wait + one D2H readback per device per token vs the wave path. The
# engine declines (returns false) on TP=1, DPAS48 items, or mismatched
# K/group/scheme. M>1 IS supported: the kernel's outer m loop iterates the
# rows, so the same launch covers prefill and decode -- this arm is
# reachable only on the 27B TP=2 model, which is the gate that carries INT2.
GATEUP_OUT="$LOG.gateup.out"; GATEUP_ERR="$LOG.gateup.err"
"$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$GATEUP_OUT" 2>"$GATEUP_ERR"; GATEUP_RC=$?
if [ "$GATEUP_RC" -ne 0 ]; then
  echo "GATE FAIL: fused gate+up run exited rc=$GATEUP_RC" >&2
  tail -20 "$GATEUP_ERR" >&2
  PASS=0
else
  echo "FUSED_GATE_UP prefill: $(cli_prefill "$GATEUP_ERR") (expect 11751)"
  echo "FUSED_GATE_UP ids:      $(cli_ids "$GATEUP_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  # The fused path is NOT a flag the operator sets; it is the default TP=2 MLP
  # path. The arm's value is the token stream, which must equal the base run's
  # stream -- that is what rp_arm_assert checks. A vacuous arm would also pass
  # that, so the DISARMAMENT is structural: run_linear_wave_batch is the
  # fallback, and this arm only certifies that the fallback was NOT taken.
  # Asserting that explicitly is what turns "the two runs matched" into "the
  # fused path ran". The engine prints [mlp-gateup] ... ACTIVE exactly once
  # when it accepts the fusion, so the arm asserts that line is present AND
  # the streams agree -- which is what makes this arm GATED rather than
  # REPORTED. (It was REPORTED in the first version because the engine printed
  # nothing on success and the two streams alone could not separate a real
  # fusion from the wave fallback.)
  if ! grep -aFq '[mlp-gateup] fused gate+up GEMM ACTIVE' "$GATEUP_ERR" 2>/dev/null; then
    echo "GATE FAIL: the fused gate+up path never ran (no [mlp-gateup] ACTIVE" >&2
    echo "         line in the stderr log); the run fell back to the wave and" >&2
    echo "         this arm is vacuous." >&2
    PASS=0
  fi
  # *** SCHEME SCOPE. The ACTIVE line alone is NOT enough. The operator's
  # 2026-10-10 run printed it with scheme=2 (INT4), group=64 -- that was a
  # LAYER THAT RAN NO KERNEL. The fused kernel quant_gemm_forward_usm_gateup
  # has a body only for INT2+INT2 (quant_gemm.cpp:2036); a same-scheme
  # INT4+INT4 pair passes every guard, returns a NULL event, and the caller
  # reads mg/mu back out of unwritten wave slots. On the 27B that is 18 of
  # 64 layers. The line's own scheme= field is the witness: the arm must
  # require it to read INT2, or it certifies a crash. (This is the gate-side
  # twin of the engine.cpp:1693 scheme guard added the same day.)
  if ! grep -aq 'scheme=3[^0-9]' "$GATEUP_ERR" 2>/dev/null; then
    echo "GATE FAIL: no [mlp-gateup] ACTIVE line carried scheme=3 (INT2)" >&2
    echo "         -- every fused launch must be an INT2+INT2 pair; a scheme=2" >&2
    echo "         (INT4) line means a layer ran the null-event path" >&2
    PASS=0
  fi
  gu_arm_assert "$CLI_ERR" "$GATEUP_ERR" "GATE FAIL: " || PASS=0
  # *** SELF-CHECK FOR THIS ARM. *** The same shape as the decode1 self-check
  # block above, and for the same reason: this arm's predicate (the ACTIVE line)
  # is the ONLY thing standing between a vacuous wave-fallback run and a PASS,
  # and a predicate must be exercised in BOTH directions. Three cases:
  #   (a) the healthy direction -- a log that carries the line must certify;
  #   (b) the refusal direction -- a healthy stream with the line removed must
  #       be refused, and ONLY for that reason (so a refusal can be isolated);
  #   (c) the empty log must be refused (vacuity).
  # The fixtures are built from the REAL log by deleting exactly one line,
  # so (b) cannot accidentally fail for a different reason -- same lesson as
  # the decode1 fixtures, which were rewritten when a hand-written negative
  # drifted into being a valid log and the refusal case passed for the wrong
  # reason. The healthy stream for (b) is the real run's own ids, so (b) also
  # pins the stream half of the arm against a known-good oracle.
  GU_ANNOUNCE='[mlp-gateup] fused gate+up GEMM ACTIVE'
  # The self-check temp dir is cleaned by the EXIT trap below, NOT by its own
  # trap: bash replaces traps rather than stacking them, so a second
  # `trap 'rm -rf "$SB2"' EXIT` would WIPE the trap that keeps *_OUT/*_ERR
  # around on failure. The EXIT trap is rewritten here to name both dirs.
  SB2="$(mktemp -d)"
  cp "$GATEUP_OUT" "$SB2/gu_good.out"; cp "$GATEUP_ERR" "$SB2/gu_good.err"
  cp "$GATEUP_OUT" "$SB2/gu_not.out"; cp "$GATEUP_ERR" "$SB2/gu_not.err"
  sed -i '/\[mlp-gateup\]/d' "$SB2/gu_not.err"
  : >"$SB2/gu_empty.out"; : >"$SB2/gu_empty.err"
  trap 'rc=$?; rm -rf "$SB2"; if [ $rc -ne 0 ]; then echo "--- CLI stdout ($CLI_OUT) ---" >&2; tail -5 "$CLI_OUT" >&2; echo "--- CLI stderr ($CLI_ERR, last 25) ---" >&2; tail -25 "$CLI_ERR" >&2; else rm -f "$CLI_OUT" "$CLI_ERR" "$MR_OUT" "$MR_ERR" "$SK_OUT" "$SK_ERR" "$RP_OUT" "$RP_ERR" "$GATEUP_OUT" "$GATEUP_ERR" "$LT_OUT" "$LT_ERR" "$WG_OUT" "$WG_ERR"; fi' EXIT
  trap 'rc=$?; rm -rf "$SB2"; if [ $rc -ne 0 ]; then echo "--- CLI stdout ($CLI_OUT) ---" >&2; tail -5 "$CLI_OUT" >&2; echo "--- CLI stderr ($CLI_ERR, last 25) ---" >&2; tail -25 "$CLI_ERR" >&2; else rm -f "$CLI_OUT" "$CLI_ERR" "$MR_OUT" "$MR_ERR" "$SK_OUT" "$SK_ERR" "$RP_OUT" "$RP_ERR" "$GATEUP_OUT" "$GATEUP_ERR" "$LT_OUT" "$LT_ERR" "$WG_OUT" "$WG_ERR"; fi; exit 130' INT
  trap 'rc=$?; rm -rf "$SB2"; if [ $rc -ne 0 ]; then echo "--- CLI stdout ($CLI_OUT) ---" >&2; tail -5 "$CLI_OUT" >&2; echo "--- CLI stderr ($CLI_ERR, last 25) ---" >&2; tail -25 "$CLI_ERR" >&2; else rm -f "$CLI_OUT" "$CLI_ERR" "$MR_OUT" "$MR_ERR" "$SK_OUT" "$SK_ERR" "$RP_OUT" "$RP_ERR" "$GATEUP_OUT" "$GATEUP_ERR" "$LT_OUT" "$LT_ERR" "$WG_OUT" "$WG_ERR"; fi; exit 143' TERM
  gu_ids() { cli_ids "$1" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//'; }
  gu_good_ids="$(gu_ids "$SB2/gu_good.out")"
  gu_not_ids="$(gu_ids "$SB2/gu_not.out")"
  [ "$gu_good_ids" = "$gu_not_ids" ] || { echo "GATE FAIL: self-check (b) fixture builder changed the stream" >&2; PASS=0; }
  [ -n "$gu_good_ids" ] || { echo "GATE FAIL: self-check (b) fixture has no ids" >&2; PASS=0; }
  grep -aFq "$GU_ANNOUNCE" "$SB2/gu_good.err" \
    || { echo "GATE FAIL: self-check (a) healthy fixture has no ACTIVE line" >&2; PASS=0; }
  grep -aFq "$GU_ANNOUNCE" "$SB2/gu_not.err" \
    && { echo "GATE FAIL: self-check (b) the line survived the sed deletion" >&2; PASS=0; }
  # (d) SCHEME-SCOPE NEGATIVE. The real run's ACTIVE line carries
  # scheme=3 (INT2) post-fix. Rewrite exactly that field to scheme=2
  # (INT4) in a copy whose stream is untouched: that is the
  # operator's crashing run, reproduced as a fixture. The scheme
  # predicate above MUST refuse it -- otherwise the arm would
  # certify the null-event layer it was built to catch. Driving the
  # grep directly (not through gu_arm_assert, which knows nothing
  # about scheme) isolates which predicate fired.
  sed 's/scheme=3\([^0-9]\)/scheme=2\1/' "$SB2/gu_good.err" >"$SB2/gu_int4.err"
  if grep -aq 'scheme=3[^0-9]' "$SB2/gu_int4.err" 2>/dev/null; then
    echo "GATE FAIL: self-check (d) the scheme rewrite left a scheme=3 line" >&2; PASS=0
  fi
  if ! grep -aq 'scheme=2[^0-9]' "$SB2/gu_int4.err" 2>/dev/null; then
    echo "GATE FAIL: self-check (d) the INT4 fixture carries no scheme=2 line" >&2; PASS=0
  fi
  if grep -aq 'scheme=3[^0-9]' "$SB2/gu_int4.err" 2>/dev/null; then
    :
  else
    echo "SELFTEST (d) INT4-fixture direction verified (scheme predicate must fire)" >&2
  fi
  # (a) healthy direction: the real run's own pair must certify.
  gu_arm_assert "$SB2/gu_good.out" "$SB2/gu_good.err" "SELFTEST (a) arm-refusal: " \
    || { echo "GATE FAIL: self-check (a) the arm REFUSED its own healthy run" >&2; PASS=0; }
  # (b) refusal direction: a healthy stream with the line removed must be
  # refused, and the refusal must be about the line -- so drive the ACTIVE
  # grep directly rather than through the arm, which would also refuse on
  # the stream half and mask which predicate fired.
  if grep -aq "$GU_ANNOUNCE" "$SB2/gu_not.err"; then
    echo "GATE FAIL: self-check (b) the arm ACCEPTED a run whose gateup path never fired" >&2; PASS=0
  fi
  gu_arm_assert "$SB2/gu_not.out" "$SB2/gu_not.err" "SELFTEST (b) arm-refusal: " \
    && { echo "GATE FAIL: self-check (b) the arm ACCEPTED a vacuous run" >&2; PASS=0; }
  # (c) empty log must be refused.
  if grep -aq "$GU_ANNOUNCE" "$SB2/gu_empty.err"; then
    echo "GATE FAIL: self-check (c) the arm ACCEPTED an empty log" >&2; PASS=0
  fi
  gu_arm_assert "$SB2/gu_empty.out" "$SB2/gu_empty.err" "SELFTEST (c) arm-refusal: " \
    && { echo "GATE FAIL: self-check (c) the arm ACCEPTED an empty log pair (vacuity)" >&2; PASS=0; }
  fi

  # --- INTFER_LMHEAD_SPLIT0 / INTFER_LMHEAD_TILE_ROWS certification -------
  # PLAN.md prefill tail. The lm_head is the last un-fused section and its
  # launch count was never exercised: the 128MB upload cap put ~20 GEMM tiles
  # per device per token on the 27B head (N=124160, K=5120), and the only knob
  # for the shard layout was a single dev0 boundary derived from free VRAM.
  # Both flags are bit-exact by construction -- tiles concatenate in row order
  # and run the SAME kernels, so the logits are identical -- which is why this
  # arm is a SINGLE flagged run against the fixed oracle, NOT a base-vs-flagged
  # stream comparison: a stream comparison would be vacuous here, because the
  # two streams are identical BY CONSTRUCTION and a flag that never reached the
  # upload would pass every stream check while certifying nothing (the same
  # nested-flag vacuity the gateup arm's ACTIVE line exists to defeat).
  #
  # The subject is the LAYOUT, not the stream, and the engine announces the
  # layout it actually built ([lmhead-tiles] ... dev0=N tiles dev1=M tiles ...).
  # So the arm asserts: the run hit the oracle AND the announcement names the
  # layout the operator asked for. That is what separates "the tile count fell"
  # from "the flag was ignored and the default layout ran".
  LT_OUT="$LOG.lt.out"; LT_ERR="$LOG.lt.err"
  LT_SPLIT="${INTFER_LMHEAD_SPLIT0:-62080,124160}"
  LT_TILE="${INTFER_LMHEAD_TILE_ROWS:-62080}"
  env INTFER_LMHEAD_SPLIT0="$LT_SPLIT" INTFER_LMHEAD_TILE_ROWS="$LT_TILE" \
  "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$LT_OUT" 2>"$LT_ERR"; LT_RC=$?
  if [ "$LT_RC" -ne 0 ]; then
  echo "GATE FAIL: INTFER_LMHEAD_SPLIT0=$LT_SPLIT TILE_ROWS=$LT_TILE run exited rc=$LT_RC" >&2
  tail -20 "$LT_ERR" >&2
  PASS=0
  else
  echo "LMHEAD-TILES prefill: $(cli_prefill "$LT_ERR") (expect 11751)"
  echo "LMHEAD-TILES ids:      $(cli_ids "$LT_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  LT_LINE="$(grep -a "\[lmhead-tiles\]" "$LT_ERR" | head -1)"
  if [ -z "$LT_LINE" ]; then
    echo "GATE FAIL: the lm_head tile-layout path never announced itself" >&2
    echo "         (no [lmhead-tiles] line); the flags did not reach the upload" >&2
    echo "         and this arm is vacuous." >&2
    PASS=0
  else
    echo "LMHEAD-TILES layout: $LT_LINE"
  fi
  # The layout the operator asked for, as the announcement must print it.
  LT_EXPECT_ANN="exact-split=$LT_SPLIT tile-rows=$LT_TILE"
  echo "$LT_LINE" | grep -aq "$LT_EXPECT_ANN" \
    || { echo "GATE FAIL: the announcement does not name the requested layout" >&2
         echo "         wanted '$LT_EXPECT_ANN', got '$LT_LINE'" >&2; PASS=0; }
  # The stream half, against the SAME FIXED ORACLE the base run uses -- not a
  # self-comparison, so "both runs broke the same way" cannot pass.
  mr_prefill_ok "$(cli_prefill "$LT_ERR")" \
    || { echo "GATE FAIL: lmhead-tiles prefill != 11751" >&2; PASS=0; }
  LT_GOT="$(cli_ids "$LT_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  LT_BASE="$(cli_ids "$CLI_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  [ -n "$LT_GOT" ] || { echo "GATE FAIL: lmhead-tiles stream is EMPTY" >&2; PASS=0; }
  [ "$LT_GOT" = "$LT_BASE" ] \
    || { echo "GATE FAIL: lmhead-tiles ids '$LT_GOT' != base '$LT_BASE'" >&2; PASS=0; }
fi

# --- INTFER_DPAS48_WG certification (PLAN item 1.3, prefill lever) ----------
  # The tiled DPAS kernel is one WI per (row-group of R, 16-N tile), and each
  # output element (m,n) is written by EXACTLY ONE block (t = n/16, mg = m/R),
  # so the work-group width is a pure PARTITION: widening it reschedules WIs
  # without changing who writes what. That is why this is bit-exact by
  # construction and needs no numerics contract -- the same shape as the lm_head
  # tile split and the gate+up fusion. The shipped value is 16, hardcoded, while
  # the B580 has 160 compute units; the prefill lever is to widen the group to
  # fill them.
  #
  # Like the lm_head flags, a base-vs-flagged stream comparison is VACUOUS:
  # the partition is exact, so the two streams are identical and a flag that
  # never reached the kernel would pass every stream check while certifying
  # nothing. The subject is the WIDTH, and the engine announces the width the
  # kernel actually used ([dpas48-wg] work-group=... geometry=... M=... N=...
  # K=...), once per process. The arm asserts the run hit the oracle AND the
  # announcement names the width the operator asked for.
  WG_OUT="$LOG.wg.out"; WG_ERR="$LOG.wg.err"
  WG_WG="${INTFER_DPAS48_WG:-160}"
  # *** INTFER_DPAS48=1 IS NOT OPTIONAL. *** The width flag alone is inert:
  # sc.dpas48 is set by dpas48_eligible (engine.cpp:203), which reads
  # INTFER_DPAS48 and returns false without it -- so without =1 the shard
  # builds the row-major path, run_linear_tp takes the legacy branch, and
  # this arm's own "never announced itself" check fires on EVERY live run.
  # The width flag is a knob ON the DPAS path, not a path selector; it needs
  # the path selector present for the announcement to exist at all. (The
  # operator's pending 27B run was written this way and would have hit a
  # vacuous GATE FAIL; fixed here so the run measures something real.)
  env INTFER_DPAS48=1 INTFER_DPAS48_WG="$WG_WG" \
  "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$WG_OUT" 2>"$WG_ERR"; WG_RC=$?
  if [ "$WG_RC" -ne 0 ]; then
  echo "GATE FAIL: INTFER_DPAS48=1 INTFER_DPAS48_WG=$WG_WG run exited rc=$WG_RC" >&2
  tail -20 "$WG_ERR" >&2
  PASS=0
  else
  echo "DPAS48-WG prefill: $(cli_prefill "$WG_ERR") (expect 11751)"
  echo "DPAS48-WG ids:      $(cli_ids "$WG_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  WG_LINE="$(grep -a "\[dpas48-wg\]" "$WG_ERR" | head -1)"
  if [ -z "$WG_LINE" ]; then
    echo "GATE FAIL: the dpas48 work-group path never announced itself" >&2
    echo "         (no [dpas48-wg] line); the flag did not reach the kernel" >&2
    echo "         and this arm is vacuous." >&2
    PASS=0
  else
    echo "DPAS48-WG layout: $WG_LINE"
  fi
  # The width the operator asked for, as the announcement must print it.
  WG_EXPECT_ANN="work-group=$WG_WG (INTFER_DPAS48_WG=$WG_WG)"
  echo "$WG_LINE" | grep -aq "$WG_EXPECT_ANN" \
    || { echo "GATE FAIL: the announcement does not name the requested width" >&2
         echo "         wanted '$WG_EXPECT_ANN', got '$WG_LINE'" >&2; PASS=0; }
  # The stream half, against the SAME FIXED ORACLE the base run uses -- not a
  # self-comparison, so "both runs broke the same way" cannot pass. The base
  # ids are re-derived here from the base run's own log rather than borrowed
  # from the lm_head arm: that arm sets its copy only inside its own success
  # branch, so borrowing it would make this arm's stream check depend on a
  # different arm's control flow.
  mr_prefill_ok "$(cli_prefill "$WG_ERR")" \
    || { echo "GATE FAIL: dpas48-wg prefill != 11751" >&2; PASS=0; }
  WG_GOT="$(cli_ids "$WG_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  WG_BASE="$(cli_ids "$CLI_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
[ "$WG_GOT" = "$WG_BASE" ] \
    || { echo "GATE FAIL: dpas48-wg ids '$WG_GOT' != base '$WG_BASE'" >&2; PASS=0; }
# *** SELF-CHECK FOR THIS ARM. *** Same shape as the gateup self-check: the
  # announcement line is the ONLY thing standing between a run where the flag
  # was ignored (default width 16 ran, stream still matched) and a PASS. The
  # partition is exact by construction -- read it, do not assert it: block=gid,
  # t = block % nT, mg = block / nT, m0 = mg*R, and the write is
  # out_usm + m*N + t*TN (quant_gemm.cpp:2397-2404, 2553-2561). For any (m,n),
  # t = n/16 is unique (nT = N/16) and mg = floor(m/R) is unique, so the map
  # (m,n) -> block = mg*nT + t is a bijection onto [0, n_blocks) and nWG only
  # decides WHICH WI schedules WHICH block. A base-vs-flagged stream comparison
  # is therefore satisfiable by a flag that did nothing -- so the arm asserts
  # the width the kernel actually used, and this self-check exercises that
  # predicate in BOTH directions.
  #
  # (b) is NOT "the deleted fixture still contains the width string": that
  # would be vacuous, because the announcement is ONE line carrying both the
  # tag and the width, so deleting it removes both and the grep could never
  # match. The honest refusal direction is the OTHER one: a log whose
  # announcement says 16 (the default -- i.e. the flag was ignored and the
  # shipped width ran) must be REFUSED, because the arm demands it name 160.
  # That fixture carries a real width string, so the predicate can actually
  # fire -- which is what separates this from a vacuous always-false guard.
  # Own temp dir: the gateup arm's $SB2 is created only on ITS success path,
  # so borrowing it would couple this self-check to a different arm's control
  # flow -- and leave the fixtures' dir unset if gateup failed while this arm
  # succeeded. Built and torn down inside this block rather than by trap
  # surgery: the EXIT trap already names the log files this block does not own.
  WGSB2="$(mktemp -d)"
  WG_ANN_FULL="work-group=$WG_WG (INTFER_DPAS48_WG=$WG_WG)"
  WG_DEFAULT_ANN="work-group=16 (INTFER_DPAS48_WG=$WG_WG)"
  # (a) healthy: the real run's own announcement must exist and name the width.
  [ -n "$WG_LINE" ] || { echo "GATE FAIL: self-check (a) healthy fixture has no [dpas48-wg] line" >&2; PASS=0; }
  echo "$WG_LINE" | grep -aq "$WG_ANN_FULL" \
    || { echo "GATE FAIL: self-check (a) the announcement does not name the requested width" >&2; PASS=0; }
  # (b) refusal: a log where the flag was IGNORED (default width 16 ran, stream
  # still matched) must be refused. Built by rewriting the real announcement's
  # width to 16 -- the exact shape of "the flag did not reach the kernel".
  WG_NOT_ERR="$WGSB2/wg_not.err"
  sed "s/work-group=[0-9]*/work-group=16/" "$WG_ERR" >"$WG_NOT_ERR"
  if ! grep -aq "\[dpas48-wg\]" "$WG_NOT_ERR"; then
    echo "GATE FAIL: self-check (b) fixture builder removed the announcement" >&2; PASS=0
  fi
  WG_NOT_LINE="$(grep -a "\[dpas48-wg\]" "$WG_NOT_ERR" | head -1)"
  if [ -n "$WG_NOT_LINE" ] && echo "$WG_NOT_LINE" | grep -aq "$WG_DEFAULT_ANN"; then
    echo "GATE FAIL: self-check (b) the arm ACCEPTED a run whose width path never fired" >&2; PASS=0
  fi
  # (c) vacuity: an empty log must fail the predicate.
  WG_EMPTY_ERR="$WGSB2/wg_empty.err"
  : >"$WG_EMPTY_ERR"
  WG_EMPTY_LINE="$(grep -a "\[dpas48-wg\]" "$WG_EMPTY_ERR" | head -1)"
  if [ -n "$WG_EMPTY_LINE" ]; then
    echo "GATE FAIL: self-check (c) the arm ACCEPTED an empty log (vacuity)" >&2; PASS=0
  fi
  rm -rf "$WGSB2"
  fi

# --- E4 PAIR-PROMOTION: INTFER_DPAS48=1 + INTFER_DPAS48_TILED=1 ----------------
  # The pending operator todo (PLAN.md 1.4): E4's cooperative 2D row-block DPAS
  # is bit-exact by construction (no partial sums -- one accumulator per (row,
  # tile), same op order as the shipped R==1 geometry), so a base-vs-flagged
  # stream comparison is satisfiable by a flag that did nothing, exactly as the
  # WG arm above. The subject is the GEOMETRY, and the same [dpas48-wg]
  # announcement carries it: geometry=tiled2d vs tiled1d vs row. This arm
  # asserts the run hit the tiled geometry AND the stream matches the base
  # oracle, and it is reachable ONLY on the 27B (E4 requires M >= 2, which is
  # prefill -- decode is M==1 and E4 declines by construction, so this arm
  # cannot be certified on a 0.8B decode model).
  E4_OUT="$LOG.e4.out"; E4_ERR="$LOG.e4.err"
  env INTFER_DPAS48=1 INTFER_DPAS48_TILED=1 \
  "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$E4_OUT" 2>"$E4_ERR"; E4_RC=$?
  if [ "$E4_RC" -ne 0 ]; then
    echo "GATE FAIL: INTFER_DPAS48=1 INTFER_DPAS48_TILED=1 run exited rc=$E4_RC" >&2
    tail -20 "$E4_ERR" >&2
    PASS=0
  else
    echo "E4 prefill: $(cli_prefill "$E4_ERR") (expect 11751)"
    echo "E4 ids:      $(cli_ids "$E4_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    E4_LINE="$(grep -a "\[dpas48-wg\]" "$E4_ERR" | head -1)"
    if [ -z "$E4_LINE" ]; then
      echo "GATE FAIL: the E4 tiled path never announced itself" >&2
      echo "         (no [dpas48-wg] line); the flag did not reach the kernel" >&2
      echo "         and this arm is vacuous." >&2
      PASS=0
    else
      echo "E4 layout: $E4_LINE"
    fi
    # The geometry the operator asked for, as the announcement must print it.
    if ! echo "$E4_LINE" | grep -aq "geometry=tiled2d"; then
      echo "GATE FAIL: the announcement does not name geometry=tiled2d" >&2
      echo "         wanted 'geometry=tiled2d', got '$E4_LINE'" >&2; PASS=0
    fi
    mr_prefill_ok "$(cli_prefill "$E4_ERR")" \
      || { echo "GATE FAIL: E4 prefill != 11751" >&2; PASS=0; }
    # The base ids are re-derived here from the base run's own log, NOT borrowed
    # from the WG arm: that arm sets its copy only inside its own success
    # branch, so borrowing it would couple this arm's stream check to a
    # different arm's control flow (and leave it empty if the WG arm failed).
    E4_GOT="$(cli_ids "$E4_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    E4_BASE="$(cli_ids "$CLI_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    [ "$E4_GOT" = "$E4_BASE" ] \
      || { echo "GATE FAIL: E4 ids '$E4_GOT' != base '$E4_BASE'" >&2; PASS=0; }
  # *** SELF-CHECK FOR THIS ARM. *** Same shape as the WG arm: the geometry
  # field is the ONLY thing standing between a run where the flag was ignored
  # (tiled1d ran, stream still matched) and a PASS. Built by rewriting the
  # real announcement's geometry to tiled1d -- the exact shape of "the flag
  # did not reach the kernel".
    E4SB2="$(mktemp -d)"
    E4_NOT_ERR="$E4SB2/e4_not.err"
    sed "s/geometry=tiled2d/geometry=tiled1d/" "$E4_ERR" >"$E4_NOT_ERR"
    if ! grep -aq "\[dpas48-wg\]" "$E4_NOT_ERR"; then
      echo "GATE FAIL: self-check the fixture builder removed the announcement" >&2; PASS=0
    fi
    E4_NOT_LINE="$(grep -a "\[dpas48-wg\]" "$E4_NOT_ERR" | head -1)"
    if [ -n "$E4_NOT_LINE" ] && echo "$E4_NOT_LINE" | grep -aq "geometry=tiled1d"; then
      echo "GATE FAIL: self-check the arm ACCEPTED a run whose tiled path never fired" >&2; PASS=0
    fi
    rm -rf "$E4SB2"
  fi

  # --- INTFER_WAVE_LAUNCH_FUSE certification (prefill lever) ---------------
    # The wave is the prefill hot path: a layer's independent GEMMs (q/k/v
    # wave-3, z/b/a wave-4, gate/up wave-2) currently run as N separate
    # parallel_for launches per device, each with its own submit + wait +
    # D2H readback. This fuses them into ONE parallel_for over the union of
    # the items' output elements, so the host pays one submit + one wait +
    # one readback pass per device per wave instead of N. Bit-exact by
    # construction (each item's accumulator is untouched), so a base-vs-
    # flagged stream comparison is satisfiable by a flag that did nothing
    # -- the same nested-flag vacuity the gateup and WG arms defeat with an
    # announcement line. The subject is the LAUNCH COUNT, and the engine
    # announces the fusion it actually ran ([wave-launch-fuse] ... ACTIVE
    # items=N M=... K=... group=..., one launch per device), once per
    # process. The arm asserts the run hit the oracle AND the announcement
    # names the fusion, and the self-check exercises that predicate in both
    # directions (a real announcement vs one whose items= field was rewritten
    # to the default width the wave path would have produced).
    WF_OUT="$LOG.wf.out"; WF_ERR="$LOG.wf.err"
    env INTFER_WAVE_LAUNCH_FUSE=1 \
    "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
    -p "$PROMPT" -n "$N" >"$WF_OUT" 2>"$WF_ERR"; WF_RC=$?
    if [ "$WF_RC" -ne 0 ]; then
    echo "GATE FAIL: INTFER_WAVE_LAUNCH_FUSE=1 run exited rc=$WF_RC" >&2
    tail -20 "$WF_ERR" >&2
    PASS=0
    else
    echo "WAVE-LAUNCH-FUSE prefill: $(cli_prefill "$WF_ERR") (expect 11751)"
    echo "WAVE-LAUNCH-FUSE ids:      $(cli_ids "$WF_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    WF_LINE="$(grep -a "\[wave-launch-fuse\]" "$WF_ERR" | head -1)"
    if [ -z "$WF_LINE" ]; then
      echo "GATE FAIL: the fused wave-launch path never announced itself" >&2
      echo "         (no [wave-launch-fuse] line); the flag did not reach the" >&2
      echo "         kernel and this arm is vacuous." >&2
      PASS=0
    else
      echo "WAVE-LAUNCH-FUSE layout: $WF_LINE"
    fi
    # The fusion the operator asked for, as the announcement must print it.
    WF_EXPECT_ANN="ACTIVE items="
    echo "$WF_LINE" | grep -aq "$WF_EXPECT_ANN" \
      || { echo "GATE FAIL: the announcement does not name the fused launch" >&2
           echo "         wanted '$WF_EXPECT_ANN', got '$WF_LINE'" >&2; PASS=0; }
    mr_prefill_ok "$(cli_prefill "$WF_ERR")" \
      || { echo "GATE FAIL: wave-launch-fuse prefill != 11751" >&2; PASS=0; }
    WF_GOT="$(cli_ids "$WF_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    WF_BASE="$(cli_ids "$CLI_ERR" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
    [ "$WF_GOT" = "$WF_BASE" ] \
      || { echo "GATE FAIL: wave-launch-fuse ids '$WF_GOT' != base '$WF_BASE'" >&2; PASS=0; }
  # *** SELF-CHECK FOR THIS ARM. *** Same shape as the WG arm: the
    # announcement line is the ONLY thing standing between a run where the
    # flag was ignored (the default wave path ran, stream still matched) and
    # a PASS. The refusal direction is the OTHER half of the items= field:
    # the announcement carries items=N, and the default wave path would
    # have produced N SEPARATE launches, so rewriting items=N to
    # items=1 (one launch per item, the unfused shape) is the exact shape
    # of "the flag did not reach the kernel". That fixture carries a real
    # items= string, so the predicate can actually fire.
    WFSB2="$(mktemp -d)"
    WF_NOT_ERR="$WFSB2/wf_not.err"
    sed 's/items=[0-9]*/items=1/' "$WF_ERR" >"$WF_NOT_ERR"
    if ! grep -aq "\[wave-launch-fuse\]" "$WF_NOT_ERR"; then
      echo "GATE FAIL: self-check the fixture builder removed the announcement" >&2; PASS=0
    fi
    WF_NOT_LINE="$(grep -a "\[wave-launch-fuse\]" "$WF_NOT_ERR" | head -1)"
    if [ -n "$WF_NOT_LINE" ] && echo "$WF_NOT_LINE" | grep -aq "items=1"; then
      echo "GATE FAIL: self-check the arm ACCEPTED a run whose fuse never fired" >&2; PASS=0
    fi
    rm -rf "$WFSB2"
    fi

rm -f "$LOG"  # *_OUT/*_ERR are handled by the EXIT trap (kept on FAIL)
