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
mr_arm_assert() {   # $1 = base log, $2 = MR log. $3 = diagnostic prefix.
  local d="${3:-}" got mr_p mr_i ok=0
  mr_p="$(cli_prefill "$2")"
  mr_i="$(cli_ids "$2" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  got="$(cli_ids "$1" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
  mr_prefill_ok "$mr_p" \
    || { echo "${d}DECODE1_MR prefill '$mr_p' != 11751" >&2; ok=1; }
  # NON-EMPTY FIRST lives inside mr_streams_agree: this arm compares two DERIVED
  # values (every other arm here compares to a literal oracle), so a log-format
  # change or broken pipe that emptied BOTH sides would make "" = "" and report
  # PASS having certified nothing.
  mr_streams_agree "$got" "$mr_i" \
    || { echo "${d}DECODE1_MR ids '$mr_i' != base '$got' (or a stream was EMPTY)" >&2; ok=1; }
  mr_tp_sharded "$2" \
    || { echo "${d}DECODE1_MR run was not TP=sharded (needs 2 devices)" >&2; ok=1; }
  return "$ok"
}

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
  mklog() { # $1 = path, $2.. = ids in step order. Appends the sharded TP banner,
            # so every fixture is shaped like a real MR log.
    local p="$1"; shift; : >"$p"; local i=0
    for t in "$@"; do echo "decode step $i: token=$t" >>"$p"; i=$((i+1)); done
    printf '%s\n' '[intfer] GPU backend: 2 device(s) (TP=sharded)' >>"$p"
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
env INTFER_DECODE1_MR=1 "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
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
fi
rm -f "$LOG"  # *_OUT/*_ERR are handled by the EXIT trap (kept on FAIL)
if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi
