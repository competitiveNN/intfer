#!/usr/bin/env bash
# GPU opt-in gate: KV-mirror and SSM_GPU paths are stream-identical to the
# base GPU path on the self-contained .intifer model.
#
# Rationale (2026-09-19): the CPU parity gate already asserts INT8-KV/Q8
# vs the HF oracle on --no-gpu, and AGENTS.md records SSM_GPU M=1 as -9%
# (default-off). What was unverified is the GPU combination: KV mirrors
# (INTFER_INT8_KV / INTFER_KV_Q8 / INTFER_KV_HAD) and device SSM
# (INTFER_SSM_GPU=1) under --gpu x TP=1/2. This gate certifies the baseline
# so the next quiet-box session can measure wall-time with a trusted setup.
#
# NOTE: this uses the self-contained .intifer model, whose greedy stream
# (279/6511/314/279) differs from the HF oracle by design — the 11751
# oracle lives in the legacy .intfer file (see memories/
# model_files_intfer_vs_intifer.md). This gate asserts cross-knob identity
# on the .intifer stream, not oracle equivalence.
# BUILD-FIRST (audit #6, shared since #7): gate_require_build in
# scripts/_gate_build.sh builds intfer_cli before running it. Without this step
# a source edit that fails to compile leaves the previous binary in place and
# the gate prints PASS for code that no longer exists -- the stale-artifact
# class observed live during audit #4c on the sibling gate, and then found in
# EVERY gate that consumes a build artifact (audit #7). Opt out with
# INTFER_GATE_NO_BUILD=1 only when the build is known current.
#
# Usage: scripts/test_gpu_optin.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/intfer_cli/intfer_cli"
PROMPT="The capital of France is"
# Wide prefill: ~97 prompt tokens, so the chunked prefill runs ONE ~97-row
# batch instead of the ~5-row batch the short PROMPT produces.
WIDE_PROMPT="In a distant galaxy a cartographer of nebulae catalogued every drifting island, noting their altitude, their colour, and the names the locals gave them; she travelled for eleven years before the atlas was complete, and when she finally returned home she found the coastline had itself moved, so she added a final chapter explaining that maps are only ever arguments with time, and that the honest cartographer admits in the margin where the ink has faded what she no longer dares to assert."
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build intfer_cli || exit 1

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
# 2-GPU precondition, PER ARM (AUDIT #13). The `for tp in 1 2` loop below runs
# ~14 assertions per tp. Engine::load clamps min(tp_devices, dm.size()), so on a
# 1-GPU box the tp=2 pass silently re-runs tp=1 and every assertion still HOLDS
# -- the gate reports PASS while certifying nothing about tensor parallelism.
# That is the worst of the three outcomes the precond helper exists to prevent:
# not a false FAIL, not a visible skip, but a false PASS. The tp=1 loop is fully
# runnable here, so the guard drops tp=2 only, and the verdict reports the lost
# coverage with exit 2.
# gate_require_device counts devices, so it must run AFTER setvars puts sycl-ls
# on PATH -- called earlier it sees an empty count and, per the helper's own
# rc/emptiness contract, refuses to certify rather than silently passing. It is
# the whole-gate precondition ("can this shell reach a device at all"); the TP=2
# arm count below is the per-arm one.
. "$SCRIPT_DIR/_gate_precond.sh"
gate_require_device || exit 2
# `|| N_GPUS=0`, not `|| echo 0`: see the identical note in
# test_device_argmax.sh (audit #14 -- `|| echo 0` produced a two-line "0\n0"
# whenever sycl-ls saw no device, and stderr stays visible so a failed COUNT is
# distinguishable from a counted zero).
N_GPUS="$(gate_gpu_count)" || N_GPUS=0
HAVE_TP2=0; [ "${N_GPUS:-0}" -ge 2 ] && HAVE_TP2=1
SKIPPED_TP2=0
TPS="1"
if [ "$HAVE_TP2" = 1 ]; then
  TPS="1 2"
  echo "precondition: $N_GPUS GPU(s) visible (need 2 for the tp=2 pass)" >&2
else
  echo "SKIPPING the tp=2 pass: need 2 GPUs, ${N_GPUS:-0} visible." >&2
  echo "  BOX STATE, not a code failure. The tp=1 pass still runs and still asserts." >&2
  echo "  Without this the tp=2 arms would re-run tp=1 and PASS vacuously." >&2
  SKIPPED_TP2=1
fi

PASS=1
fail() { echo "GATE FAIL: $1" >&2; PASS=0; }
ids_of() { # $1=tp $2=env... -- prints decode ids
  local tp="$1"; shift
  env "$@" "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n "${N:-4}" \
    --no-stream --ids $TOK 2>&1 >/dev/null \
    | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
    | tr '\n' ' ' | sed 's/ *$//'
}

wide_ids_of() { # $1=tp $2=env... -- prints decode ids for the ~97-token prefill
  local tp="$1"; shift
  env "$@" "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$WIDE_PROMPT" -n 8 \
    --no-stream --ids $TOK 2>&1 >/dev/null \
    | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
    | tr '\n' ' ' | sed 's/ *$//'
}

for tp in $TPS; do
  BASE="$(ids_of "$tp")"
  echo "TP=$tp base: '$BASE'"
  [ -n "$BASE" ] || fail "TP=$tp base empty (broken GPU run)"
  # INTFER_FUSE_ACT=1 skips a redundant FP32->FP16 convert plus one host upload
  # per device when consecutive linears consume the SAME activation vector (an
  # MLP's gate and up). It touches no arithmetic -- the destination already holds
  # exactly the bytes the skipped call would have written -- so it belongs in the
  # stream-identity family above, not in the DPAS48 divergence family below.
  #
  # IT IS REACHABLE HERE, which is why this gate and not a 27B one: the branch is
  # not guarded on `bits == 2`, so the 0.8B model enters it. An arm in a gate
  # whose model could not reach the branch would be green because the path never
  # ran (audit #3's nested-flag vacuity).
  #
  # The arm ALSO requires the kernel's one-time announcement, so "both runs
  # produced the same tokens because the flag never took effect" cannot pass.
  fuse_act_announced() {  # $1 = log
    grep -aq '\[fuse-act\] activation staging reuse ACTIVE' "$1"
  }
  for mode in "INTFER_INT8_KV=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1" "INTFER_SSM_GPU=1"; do
    # shellcheck disable=SC2086
    GOT="$(env $mode "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n 4 \
      --no-stream --ids $TOK 2>&1 >/dev/null \
      | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
      | tr '\n' ' ' | sed 's/ *$//')"
    echo "TP=$tp [$mode]: '$GOT'"
    [ "$GOT" = "$BASE" ] || fail "TP=$tp [$mode] '$GOT' != base '$BASE'"
  done

  # --- INTFER_WEIGHTS_DEVICE: stream identity, and reachability is structural --
  # Weight shards move from malloc_shared to malloc_device USM, filled by
  # q.memcpy instead of a host memcpy. Same bytes, same kernels, same order, so
  # this is BIT-EXACT and needs no numerics contract -- only proof that the stream
  # did not move. Measured upside by examples/roofline_probe: +11 % to +16 % on the
  # sequential-read ceiling for the identical buffer (206-231 -> 231-280 GB/s).
  #
  # WHY THIS GATE AND NOT A 27B ONE: the change is in pool_acquire, which every
  # GEMM on every scheme goes through, so the 0.8B model reaches it. An arm in a
  # gate whose model could not reach the branch would be green because the path
  # never ran (audit #3).
  #
  # REACHABILITY IS ASSERTED, not assumed, and it is asserted DIFFERENTLY from the
  # arms above. Those need a kernel announcement because they change a DEVICE
  # kernel's behaviour. This one changes an ALLOCATOR, and the device-visible
  # symptom is the opposite of an announcement: the engine must NOT print its
  # device-USM-OOM abort, and the run must complete. So the arm checks the run
  # exited 0 (which is where the abort would show up) and that the ids are
  # non-empty. A stub that ignored the flag would also pass -- which is stated
  # here rather than glossed, because the alternative (an announcement line for a
  # memory property) would add a print to the hot preload path to certify a change
  # that is bit-exact by construction.
  WD_LOG="$BUILD_DIR/gpu_optin_weights_device_tp${tp}.log"
  env INTFER_WEIGHTS_DEVICE=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$WD_LOG" 2>&1
  WD_RC=$?
  WD_IDS="$(grep -a -oE "decode step [0-9]+: token=[0-9]+" "$WD_LOG" \
           | sed -E 's/.*token=//' | tr '\n' ' ' | sed 's/ *$//')"
  echo "TP=$tp INTFER_WEIGHTS_DEVICE rc=$WD_RC: '$WD_IDS'"
  [ "$WD_RC" -eq 0 ] || fail "TP=$tp INTFER_WEIGHTS_DEVICE run exited rc=$WD_RC (see $WD_LOG)"
  [ -n "$WD_IDS" ] || fail "TP=$tp INTFER_WEIGHTS_DEVICE produced no tokens (see $WD_LOG)"
  [ "$WD_IDS" = "$BASE" ] || fail "TP=$tp INTFER_WEIGHTS_DEVICE '$WD_IDS' != base '$BASE'"
  grep -aq 'FATAL: USM alloc' "$WD_LOG" \
    && fail "TP=$tp INTFER_WEIGHTS_DEVICE hit a device-USM OOM (see $WD_LOG)"

  # --- INTFER_FUSE_ACT: reachability AND stream identity, in one arm --------
  # Stream identity alone is not enough here. The optimization is "skip work when
  # the activation repeats"; if it never repeats on this model, the flag is inert
  # and the stream trivially matches. So the arm asserts the kernel ANNOUNCED a
  # hit as well -- a vacuous-pass guard, not a second numerics claim.
  {
    # shellcheck disable=SC2086
    FUSE_OUT="$(mktemp)"; FUSE_ERR="$(mktemp)"
    env INTFER_FUSE_ACT=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n "${N:-4}" \
      --no-stream --ids $TOK >"$FUSE_OUT" 2>"$FUSE_ERR"; FUSE_RC=$?
    if [ "$FUSE_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_FUSE_ACT=1 run exited rc=$FUSE_RC"; tail -10 "$FUSE_ERR" >&2
    else
      FUSE_IDS="$(ids_of "$tp" INTFER_FUSE_ACT=1)"
      echo "TP=$tp [INTFER_FUSE_ACT=1]: '$FUSE_IDS'"
      [ -n "$FUSE_IDS" ] || fail "TP=$tp [INTFER_FUSE_ACT=1] stream EMPTY"
      [ "$FUSE_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_FUSE_ACT=1] '$FUSE_IDS' != base '$BASE' (it must be bit-exact)"
      fuse_act_announced "$FUSE_ERR" \
        || fail "TP=$tp [INTFER_FUSE_ACT=1] kernel never announced a staging HIT -- the flag never reached the repeat path, so the stream match above was vacuous"
    fi
    rm -f "$FUSE_OUT" "$FUSE_ERR"
  }

  # --- DPAS48 family --------------------------------------------------------
  # INTFER_DPAS48=1 switches the GEMM to the INT8/GA64 act-quant DPAS path,
  # which is a DIFFERENT numerics from the FP16-act base above — DPAS48 is
  # expected to diverge from BASE and is not asserted against it. What MUST be
  # bit-exact is the act-quant PRE-PASS against its own host-quant baseline:
  #
  #   DPAS48        != BASE  (intentional: different act quant)
  #   DPAS48+PREQ   == DPAS48 (the pre-pass reproduces the host GA64 quant)
  #   PREQ alone    == BASE  (PREQ is nested inside DPAS48 and inert without it)
  #
  # The middle equality is the real bit-contract gate for INTFER_DPAS_PREQ.
  # Asserting PREQ against BASE instead is a vacuous/misleading test.
  # N=24 (not the default -n 4): at n=4 the DPAS48 and base streams happen to
  # coincide, so a short run can pass while the numerics actually differ. The
  # longer window makes the DPAS-vs-BASE divergence visible, which proves the
  # pre-pass equality below is a real comparison and not a coincidence.
  D48="$(N=24 ids_of "$tp" INTFER_DPAS48=1)"
  echo "TP=$tp [INTFER_DPAS48=1 n=24]: '$D48'"
  [ -n "$D48" ] || fail "TP=$tp DPAS48 stream empty (broken run)"

  # MUST be bit-exact: the device pre-pass reproduces the host GA64 quant.
  GOT="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS_PREQ=1)"
  echo "TP=$tp [DPAS48+PREQ n=24]: '$GOT'"
  [ "$GOT" = "$D48" ] || fail "TP=$tp [DPAS48+PREQ] '$GOT' != DPAS48 '$D48'"

  # Also gated: the pre-pass must be inert when the row-cooperative geometry
  # is selected (use_row and preq_ok are mutually exclusive by construction).
  GOT="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS_ROW=1 INTFER_DPAS_PREQ=1)"
  echo "TP=$tp [DPAS48+ROW+PREQ n=24]: '$GOT'"
  [ "$GOT" = "$D48" ] || fail "TP=$tp [DPAS48+ROW+PREQ] '$GOT' != DPAS48 '$D48'"

  # SSM_GPU composes with the pre-pass. This WAS a known divergence at decode
  # step 14 and is now GATED, because the cause was fixed, not tolerated:
  # the GA64 rule q = trunc(x*inv +- 0.5) amplifies a 1-ulp difference in the
  # reciprocal `inv` into a whole +-1 code error whenever the product lands on
  # a .5 tie. `1.0f/s` is not a portable bit-source (a device division lowers
  # to rcp.approx, 1 ulp off on ~10% of legal scales; host call sites in
  # different contexts have been observed to disagree), so the two sides
  # disagreed by one code on ~2 elements per ~1950 GEMM calls. The pre-pass now
  # consumes the HOST's reciprocal verbatim (ActQuantDevParams::scale_inv_in
  # fed by dashq::act_quant_ga64_scale_inv_batch, which routes through the
  # single noinline ga64_recip site) instead of dividing. Do not "simplify"
  # that back into a device or kernel-local division. The numeric invariant
  # itself is gated independently and much faster by scripts/test_ga64_recip.sh
  # -- this stream gate only proves it survives the real model numerics.
  GOT="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS_PREQ=1 INTFER_SSM_GPU=1)"
  echo "TP=$tp [DPAS48+PREQ+SSM_GPU n=24]: '$GOT'"
  [ "$GOT" = "$D48" ] || fail "TP=$tp [DPAS48+PREQ+SSM_GPU] '$GOT' != DPAS48 '$D48'"

  # SSM_GPU alone against the DPAS48 host-act-quant baseline.
  GOT="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_SSM_GPU=1)"
  echo "TP=$tp [DPAS48+SSM_GPU n=24]: '$GOT'"
  [ "$GOT" = "$D48" ] || fail "TP=$tp [DPAS48+SSM_GPU] '$GOT' != DPAS48 '$D48'"

  # WIDE-PREFILL (M>1) ARM. Every assertion above uses the 5-token PROMPT, so
  # the pre-pass only ever runs with M=5 during prefill there; the short-prompt
  # arms prove nothing about the helper's 2*M*nGA row stride (act #4's hole).
  # This prompt tokenizes to ~97 tokens, which the default chunk_M=128 forwards
  # as ONE ~97-row batch, so DPAS48+PREQ is checked against DPAS48 with the
  # pre-pass consuming M=97 rows of shipped (scale, inverse) pairs per device.
  WD48="$(wide_ids_of "$tp" INTFER_DPAS48=1)"
  WGOT="$(wide_ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS_PREQ=1)"
  echo "TP=$tp [wide DPAS48]: '$WD48'"
  echo "TP=$tp [wide DPAS48+PREQ]: '$WGOT'"
  [ -n "$WD48" ] || fail "TP=$tp wide DPAS48 stream empty (broken run)"
  [ "$WGOT" = "$WD48" ] || fail "TP=$tp [wide DPAS48+PREQ] '$WGOT' != wide DPAS48 '$WD48'"

  # base and the PREQ-alone inertness check. Defined HERE (not at the end of
  # this block) because the FOLD arms below reuse BOTH: PREQ-alone must be
  # inert without DPAS48, and FOLD-alone likewise, and both compare to base.
  B24="$(N=24 ids_of "$tp")"
  PQ="$(N=24 ids_of "$tp" INTFER_DPAS_PREQ=1)"
  echo "TP=$tp [base n=24]: '$B24'"
  echo "TP=$tp [INTFER_DPAS_PREQ=1 alone]: '$PQ'"
  [ "$PQ" = "$B24" ] || fail "TP=$tp PREQ-alone '$PQ' != base '$B24' (PREQ must be inert without DPAS48)"

  # --- INTFER_DPAS48_FOLD arms (default-off row-fold hoist) -------------
  # The fold pre-pass hoists the per-(row,op) activation column sums out of
  # quant_gemm_dpas48's 16-N tile loop. It is a pure WORK hoist: the values are
  # exact int32 sums of int8 codes, so the contract is bit-identity with plain
  # DPAS48 — NOT identity with the FP16-act base (same as DPAS48 itself).
  # Do not "simplify" this into a device-side sum of a different order or a
  # wider type; the exactness argument is what makes it safe.
  FOLD="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1)"
  echo "TP=$tp [DPAS48+FOLD n=24]: '$FOLD'"
  [ -n "$FOLD" ] || fail "TP=$tp DPAS48+FOLD stream empty (broken run)"
  [ "$FOLD" = "$D48" ] || fail "TP=$tp [DPAS48+FOLD] '$FOLD' != DPAS48 '$D48'"

  # Composes with the pre-pass (the fold dep-chains on it).
  FOLDP="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS_PREQ=1 INTFER_DPAS48_FOLD=1)"
  echo "TP=$tp [DPAS48+PREQ+FOLD n=24]: '$FOLDP'"
  [ "$FOLDP" = "$D48" ] || fail "TP=$tp [DPAS48+PREQ+FOLD] '$FOLDP' != DPAS48 '$D48'"

  # WIDE arm, and the one that actually matters for the fold: the row-stride
  # contract needs >= 2*nOp, which is invisible at M=1 (single row, nothing can
  # be overwritten) and only bites at M>1 with a workspace stride smaller than
  # the call's 2*nOp. The ~97-token prompt is forwarded as one ~97-row batch.
  WFOLD="$(wide_ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1)"
  echo "TP=$tp [wide DPAS48+FOLD]: '$WFOLD'"
  [ "$WFOLD" = "$WD48" ] || fail "TP=$tp [wide DPAS48+FOLD] '$WFOLD' != wide DPAS48 '$WD48'"

  # FOLD without DPAS48 must be inert (nested-flag guard, same shape as PREQ).
  FOLDA="$(N=24 ids_of "$tp" INTFER_DPAS48_FOLD=1)"
  echo "TP=$tp [INTFER_DPAS48_FOLD=1 alone]: '$FOLDA'"
  [ "$FOLDA" = "$B24" ] || fail "TP=$tp [INTFER_DPAS48_FOLD=1 alone] '$FOLDA' != base '$B24'"

  # --- INTFER_DPAS48_TILED arms (E4: cooperative 2D row block) ------------
  # E4 gives each work-item 2 ROWS of ONE 16-N tile, so the B op-tile is fetched
  # once per op and reused across both rows. It is BIT-EXACT with plain DPAS48 --
  # each (row, tile) keeps one accumulator and still sums ops in the same order,
  # so no FP32 add is reordered. That is why these are hard `==` assertions and
  # not the E4 tolerance gate: the tolerance gate (backward_err / ULP / fraction)
  # lives in examples/dpas_fold_probe and covers the GEOMETRY question, while
  # this covers the ENGINE wiring, which a device-level probe cannot see.
  #
  # These arms exist because of a recorded failure mode: every flag nested inside
  # DPAS48 was once vacuously green because the inner path never ran. E4 DECLINES
  # at M<2, so the short-prompt arm below is the INERTNESS check (M=1 decode) and
  # the WIDE arm is the one where the block actually fires -- one without the
  # other proves nothing.
  TILED="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS48_TILED=1)"
  echo "TP=$tp [DPAS48+TILED n=24]: '$TILED'"
  [ -n "$TILED" ] || fail "TP=$tp DPAS48+TILED stream empty (broken run)"
  [ "$TILED" = "$D48" ] || fail "TP=$tp [DPAS48+TILED] '$TILED' != DPAS48 '$D48'"

  WTILED="$(wide_ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS48_TILED=1)"
  echo "TP=$tp [wide DPAS48+TILED]: '$WTILED'"
  [ -n "$WTILED" ] || fail "TP=$tp wide DPAS48+TILED stream empty (broken run)"
  [ "$WTILED" = "$WD48" ] || fail "TP=$tp [wide DPAS48+TILED] '$WTILED' != wide DPAS48 '$WD48'"

  # Composes with the E3 hoist and the PREQ pre-pass (E4 reads the same fold
  # buffer, indexed per row -- a mis-strided read here would corrupt row 1+).
  TILDF="$(wide_ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1 INTFER_DPAS48_TILED=1)"
  echo "TP=$tp [wide DPAS48+FOLD+TILED]: '$TILDF'"
  [ "$TILDF" = "$WD48" ] || fail "TP=$tp [wide DPAS48+FOLD+TILED] '$TILDF' != wide DPAS48 '$WD48'"

  # TILED without DPAS48 must be inert (nested-flag guard, same shape as
  # PREQ/FOLD). Without this, a bug that made E4 reachable without DPAS48 would
  # change the base FP16-act stream and this arm would catch it.
  TILDA="$(N=24 ids_of "$tp" INTFER_DPAS48_TILED=1)"
  echo "TP=$tp [INTFER_DPAS48_TILED=1 alone]: '$TILDA'"
  [ "$TILDA" = "$B24" ] || fail "TP=$tp [INTFER_DPAS48_TILED=1 alone] '$TILDA' != base '$B24'"
done

if [ "$PASS" != 1 ]; then
  echo "GPU-OPTIN TEST: FAIL" >&2
  exit 1
fi
if [ "$SKIPPED_TP2" = 1 ]; then
  # tp=1 fully green; tp=2 never ran. Exit 2 so this is never mistaken for the
  # full matrix being certified on this box.
  echo "GPU-OPTIN TEST: PARTIAL PASS (tp=1 arms all green; tp=2 pass SKIPPED -- need 2 GPUs)" >&2
  exit 2
fi
echo "GPU-OPTIN TEST: PASS"
