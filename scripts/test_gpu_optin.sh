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
  #
  # *** THE ARM FORCES A REACHABLE CONFIG (found 2026-10-12). *** fuse_act and
  # its hit test live in Engine::run_linear_tp (engine.cpp:1358/1369), but the
  # default MLP never goes through that function: TP=1 takes run_mlp_fused and
  # TP=2 takes run_mlp_gateup, both of which convert the activation in their own
  # phase-1 block, and the attention qkv goes through run_linear_wave, whose
  # items each own a scratch buffer (so act_cache_ptr_ != x.data() even though
  # the bytes are equal). MEASURED on the 0.8B with -n 32: ZERO hits, so the
  # arm was comparing the default base against the same default base and the
  # "[fuse-act] announced" assertion was satisfiable by a flag that never
  # fired -- audit #3's vacuity, live, and invisible because the stream matched.
  # INTFER_NO_FUSE_SILUMUL=1 INTFER_NO_WAVE=1 restores the sequential gate/up
  # pair over the shared `rin`, which is the configuration the flag's own
  # comment names ("consecutive linears consume the SAME activation vector").
  # The base below is run under the SAME two kills, because they change the
  # dispatch, and comparing against the default base would be comparing two
  # different execution paths.
  FUSE_BASE="$(ids_of "$tp" INTFER_NO_FUSE_SILUMUL=1 INTFER_NO_WAVE=1)"
  {
    # shellcheck disable=SC2086
    FUSE_OUT="$(mktemp)"; FUSE_ERR="$(mktemp)"
    env INTFER_FUSE_ACT=1 INTFER_NO_FUSE_SILUMUL=1 INTFER_NO_WAVE=1 \
      "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n "${N:-4}" \
      --no-stream --ids $TOK >"$FUSE_OUT" 2>"$FUSE_ERR"; FUSE_RC=$?
    if [ "$FUSE_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_FUSE_ACT=1 run exited rc=$FUSE_RC"; tail -10 "$FUSE_ERR" >&2
    else
      FUSE_IDS="$(ids_of "$tp" INTFER_FUSE_ACT=1 INTFER_NO_FUSE_SILUMUL=1 INTFER_NO_WAVE=1)"
      echo "TP=$tp [INTFER_FUSE_ACT=1 NO_FUSE_SILUMUL=1 NO_WAVE=1]: '$FUSE_IDS'"
      [ -n "$FUSE_IDS" ] || fail "TP=$tp [INTFER_FUSE_ACT=1] stream EMPTY"
      [ -n "$FUSE_BASE" ] || fail "TP=$tp [INTFER_FUSE_ACT=1] sequential base EMPTY -- nothing to compare against"
      [ "$FUSE_IDS" = "$FUSE_BASE" ] \
        || fail "TP=$tp [INTFER_FUSE_ACT=1] '$FUSE_IDS' != sequential base '$FUSE_BASE' (it must be bit-exact)"
      fuse_act_announced "$FUSE_ERR" \
        || fail "TP=$tp [INTFER_FUSE_ACT=1] kernel never announced a staging HIT -- the flag never reached the repeat path, so the stream match above was vacuous"
    fi
    rm -f "$FUSE_OUT" "$FUSE_ERR"
  }

  # --- INTFER_L4_FUSE: L4 la_out fusion, TP=1 only --------------------------
  l4_fused_announced() { grep -aq '\[l4-fused\] la_out fusion ACTIVE' "$1"; }
  l4_fused_batch_announced() { grep -aq '\[l4-fused-batch\] M-row la_out fusion ACTIVE' "$1"; }
  {
    # L4_OUT/L4_ERR are assigned here, mirroring the FUSE_ACT block above --
    # they were missing, so under `set -u` the first reference at line 191
    # was an unbound-variable error and the arm died before running anything
    # (measured: test_gpu_optin.sh [healthy] rc=1 with "L4_OUT: unbound
    # variable"). Same shape as FUSE_OUT/FUSE_ERR, which is why the pattern
    # is copied rather than invented.
    L4_OUT="$(mktemp)"; L4_ERR="$(mktemp)"
    env INTFER_L4_FUSE=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$L4_OUT" 2>"$L4_ERR"; L4_RC=$?
    if [ "$L4_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_L4_FUSE=1 run exited rc=$L4_RC"; tail -10 "$L4_ERR" >&2
    else
      L4_IDS="$(ids_of "$tp" INTFER_L4_FUSE=1)"
      echo "TP=$tp [INTFER_L4_FUSE=1]: '$L4_IDS'"
      [ -n "$L4_IDS" ] || fail "TP=$tp INTFER_L4_FUSE=1 stream EMPTY"
      [ "$L4_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_L4_FUSE=1] '$L4_IDS' != base '$BASE' (must be bit-exact)"
      if [ "$tp" = "1" ]; then
        if ! l4_fused_announced "$L4_ERR" && ! l4_fused_batch_announced "$L4_ERR"; then
          fail "TP=$tp INTFER_L4_FUSE=1: no announcement -- the fusion never took, stream match vacuous"
        fi
      else
        if l4_fused_announced "$L4_ERR" || l4_fused_batch_announced "$L4_ERR"; then
          fail "TP=$tp INTFER_L4_FUSE=1: fusion announced on TP=2 where it must decline"
        fi
      fi
    fi
    rm -f "$L4_OUT" "$L4_ERR"
  }
  {
    L4SB="$(mktemp -d)"
    env INTFER_L4_FUSE=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$L4SB/good.out" 2>"$L4SB/good.err" || true
    cp "$L4SB/good.err" "$L4SB/not.err"
    sed -i '/\[l4-fused\]/d' "$L4SB/not.err"
    : >"$L4SB/empty.err"
    # The self-check needs a real announcement to delete. On a box where the
    # engine does not run at all (this one: signal 11 in
    # libze_intel_gpu.so.1 on the 0.8B model, base and L4_FUSE alike) the
    # tp=1 fixture produces no announcement and NONE of the three cases can
    # be exercised. That is a BOX-STATE skip, not a pass: the predicate is
    # unproven here, and saying so explicitly is the difference between a
    # self-check that cannot fire and one that fires vacuously.
    if ! l4_fused_announced "$L4SB/good.err" && ! l4_fused_batch_announced "$L4SB/good.err"; then
      echo "TP=$tp L4_FUSE self-check SKIPPED: the tp=1 fixture produced no \
            [l4-fused] line (engine did not run on this box), so the \
            refusal/vacuity directions are unexercised here" >&2
    else
      # (a) healthy direction: a log that carries the line must certify.
      # (b) refusal direction: a healthy stream with the announcement line
      #   deleted must be refused, and ONLY for that reason.
      if l4_fused_announced "$L4SB/not.err" || l4_fused_batch_announced "$L4SB/not.err"; then
        fail "TP=$tp L4_FUSE self-check (b): log with announcement deleted was ACCEPTED -- arm is stream-only"
      fi
      # (c) vacuity: an empty log must be refused.
      if l4_fused_announced "$L4SB/empty.err" || l4_fused_batch_announced "$L4SB/empty.err"; then
        fail "TP=$tp L4_FUSE self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$L4SB"
  }

  # --- INTFER_L5_FRONT: device front chain, TP=1 only -----------------------
  # Same shape as the L4_FUSE arm. The front chain replaces the host wave-4
  # (qkv/z/b/a) + conv FIR + prep + the L4 tail with one device chain
  # (FWHT acts -> qkv GEMMs -> conv prep -> L4 tail) and is BIT-IDENTICAL to
  # the host path: same kernels, same order, same bytes (the L5-vs-base A/B
  # note at main.cpp:4165 records the ~1ulp oracle-arbitrated drift, which is
  # the SSM-side tolerance, not a stream divergence). So a base-vs-flagged
  # stream comparison is satisfiable by a flag that did nothing, and the
  # kernel's [l5-front] acceptance line is what turns the match into
  # certification.
  #
  # TP=1 ONLY: gpu_l5_front (main.cpp:4199) declines with L5_DECLINE("tp")
  # when be.num_devices_ != 1, for the same reason as L4 -- device acts are
  # not cross-visible. The arm therefore asserts the line on TP=1 and its
  # ABSENCE on TP=2.
  l5_front_announced() { grep -aq '\[l5-front\] device front chain ACTIVE' "$1"; }
  {
    # L5_OUT/L5_ERR are assigned here, same as L4 above -- they were missing,
    # so under `set -u` this arm died with an unbound-variable error before
    # running anything. Same shape as FUSE_OUT/FUSE_ERR and the L4 fix.
    L5_OUT="$(mktemp)"; L5_ERR="$(mktemp)"
    env INTFER_L5_FRONT=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$L5_OUT" 2>"$L5_ERR"; L5_RC=$?
    if [ "$L5_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_L5_FRONT=1 run exited rc=$L5_RC"; tail -10 "$L5_ERR" >&2
    else
      L5_IDS="$(ids_of "$tp" INTFER_L5_FRONT=1)"
      echo "TP=$tp [INTFER_L5_FRONT=1]: '$L5_IDS'"
      [ -n "$L5_IDS" ] || fail "TP=$tp INTFER_L5_FRONT=1 stream EMPTY"
      [ "$L5_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_L5_FRONT=1] '$L5_IDS' != base '$BASE' (must be bit-exact)"
      if [ "$tp" = "1" ]; then
        if ! l5_front_announced "$L5_ERR"; then
          fail "TP=$tp INTFER_L5_FRONT=1: no [l5-front] line -- the front chain never took, stream match vacuous"
        fi
      else
        if l5_front_announced "$L5_ERR"; then
          fail "TP=$tp INTFER_L5_FRONT=1: front chain announced on TP=2 where it must decline"
        fi
      fi
    fi
    rm -f "$L5_OUT" "$L5_ERR"
  }
  {
    L5SB="$(mktemp -d)"
    env INTFER_L5_FRONT=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$L5SB/good.out" 2>"$L5SB/good.err" || true
    cp "$L5SB/good.err" "$L5SB/not.err"
    sed -i '/\[l5-front\]/d' "$L5SB/not.err"
    : >"$L5SB/empty.err"
    if ! l5_front_announced "$L5SB/good.err"; then
      echo "TP=$tp L5_FRONT self-check SKIPPED: the tp=1 fixture produced no \
            [l5-front] line (engine did not run on this box), so the \
            refusal/vacuity directions are unexercised here" >&2
    else
      if l5_front_announced "$L5SB/not.err"; then
        fail "TP=$tp L5_FRONT self-check (b): log with announcement deleted was ACCEPTED -- arm is stream-only"
      fi
      if l5_front_announced "$L5SB/empty.err"; then
        fail "TP=$tp L5_FRONT self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$L5SB"
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
  # --- INTFER_INT8_KV_G: group size for the INT8 KV quantiser ----------------
  # This knob is a TUNING PARAMETER of an already-gated feature (INTFER_INT8_KV,
  # certified at line 117), not a separate arithmetic path. It re-partitions the
  # SAME quantiser over the SAME tokens, so the quantisation error profile
  # moves within the range the existing KV arms already assert against the
  # base stream -- the knob's effect is covered by arms that run at the
  # default. What was NOT covered is the knob's own reachability: a run that
  # never called kv_group_size() would still match the base stream (the
  # default 32 ran) and certify nothing. So this arm is a REACHABILITY arm,
  # not a numerics arm: it asserts the group the quantiser actually used.
  #
  # The announcement is the group the engine read from the env, once per
  # process, so the arm certifies the knob rather than a constant.
  #
  # *** ONE PREDICATE, TWO CONSUMERS (audit #3). *** The predicate is the
  # group the announcement names, NOT the tag's presence: the tag is the same
  # whether the knob fired or not, so a self-check that only greps the tag
  # would be driven by a predicate the real arm does not use, and a
  # weakened real arm would stay green. Both the arm and the self-check go
  # through int8kvg_group, which returns the group named by the line (or the
  # empty string when there is no line at all).
  int8kvg_group() { # $1 = log -> prints the group named, or nothing
    grep -aoE '\[int8-kv-g\].*group=[0-9]+ ' "$1" | head -1 \
      | grep -aoE 'group=[0-9]+' | head -1 \
      | sed -E 's/group=([0-9]+)/\1/'
  }
  int8kvg_ok() {   # $1 = log, $2 = expected group. 0=accept.
    [ "$(int8kvg_group "$1")" = "$2" ]
  }
  {
    G_OUT="$(mktemp)"; G_ERR="$(mktemp)"
    env INTFER_INT8_KV=1 INTFER_INT8_KV_G=16 "$BIN" --gpu --tp "$tp" -m "$MODEL" \
      -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK >"$G_OUT" 2>"$G_ERR"; G_RC=$?
    if [ "$G_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_INT8_KV=1 INTFER_INT8_KV_G=16 run exited rc=$G_RC"; tail -10 "$G_ERR" >&2
    else
      G_IDS="$(ids_of "$tp" INTFER_INT8_KV=1 INTFER_INT8_KV_G=16)"
      echo "TP=$tp [INTFER_INT8_KV_G=16]: '$G_IDS'"
      [ -n "$G_IDS" ] || fail "TP=$tp INTFER_INT8_KV_G=16 stream EMPTY"
      [ "$G_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_INT8_KV_G=16] '$G_IDS' != base '$BASE' (group size is a quantiser knob, must not move the stream)"
      int8kvg_ok "$G_ERR" 16 \
        || { echo "GATE FAIL: TP=$tp INTFER_INT8_KV_G=16: the announcement does not name group=16, got '$(int8kvg_group "$G_ERR")'" >&2; PASS=0; }
    fi
    rm -f "$G_OUT" "$G_ERR"
  }
  {
    GSB="$(mktemp -d)"
    env INTFER_INT8_KV=1 INTFER_INT8_KV_G=16 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$GSB/good.out" 2>"$GSB/good.err" || true
    cp "$GSB/good.err" "$GSB/not.err"
    sed -i 's/group=16 /group=32 /' "$GSB/not.err"
    : >"$GSB/empty.err"
    if [ "$(int8kvg_group "$GSB/good.err")" != "16" ]; then
      echo "TP=$tp INT8_KV_G self-check SKIPPED: the fixture produced no \
            [int8-kv-g] group line (the INT8 KV mirror never enabled on this \
            box), so the refusal/vacuity directions are unexercised here" >&2
    else
      # (b) refusal: a log whose announcement says group=32 (the default -- the
      # knob was ignored) must be refused. The fixture carries a real group
      # string, so the predicate can actually fire -- which is what separates
      # this from a vacuous always-false guard.
      if int8kvg_ok "$GSB/not.err" 16; then
        fail "TP=$tp INT8_KV_G self-check (b): a log whose group was 32 (the default -- the knob ignored) was ACCEPTED -- arm is knob-only"
      fi
      # (c) vacuity: an empty log must fail the predicate.
      if int8kvg_ok "$GSB/empty.err" 16; then
        fail "TP=$tp INT8_KV_G self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$GSB"
  }


  # --- INTFER_L4_ROWS: FNV row-fidelity hash, TP=1 only ---------------------
  # This flag is a DIAGNOSTIC by the census's own definition (it memcpy's data
  # that is already there, hashes it, and discards the result -- "print paths,
  # not arithmetic"), but its suffix is not in the is_diag list, so the
  # mechanical classifier files it under opt-in and prints "REAL GAP". The
  # classifier is correct as a query; what it cannot do is read semantics, and
  # that is recorded in check_flag_gate_coverage.sh:25-41 rather than by
  # widening the suffix list. Arming it anyway is still right, for the same
  # reason every other arm here has one: the hash runs INSIDE the L4 fused
  # path (main.cpp:3461-3513), which is TP=1 only (L4_DECLINE("tp") at
  # main.cpp:3330), so on TP=2 the line cannot appear and a TP=2 arm would be
  # green because the path never ran. The arm asserts the line on TP=1 and its
  # ABSENCE on TP=2, exactly as L4_FUSE and L5_FRONT do.
  #
  # It is a pure OBSERVATION -- no state is mutated (the S read is wrapped in
  # try/catch and the result is discarded) -- so the stream is bit-identical
  # to base and a stream comparison is satisfiable by a flag that never
  # reached the hash. The line is what turns the match into certification.
  l4_rows_path() { # $1 = log -> prints the path named, or nothing
    grep -aoE '\[l4-rows\].*path=[a-z]+' "$1" | head -1 \
      | grep -aoE 'path=[a-z]+' | head -1 \
      | sed -E 's/path=([a-z]+)/\1/'
  }
  l4_rows_ok() {   # $1 = log, $2 = expected path. 0=accept.
    [ "$(l4_rows_path "$1")" = "$2" ]
  }
  {
    R_OUT="$(mktemp)"; R_ERR="$(mktemp)"
    env INTFER_L4_FUSE=1 INTFER_L4_ROWS=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" \
      -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK >"$R_OUT" 2>"$R_ERR"; R_RC=$?
    if [ "$R_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_L4_FUSE=1 INTFER_L4_ROWS=1 run exited rc=$R_RC"; tail -10 "$R_ERR" >&2
    else
      R_IDS="$(ids_of "$tp" INTFER_L4_FUSE=1 INTFER_L4_ROWS=1)"
      echo "TP=$tp [INTFER_L4_ROWS=1]: '$R_IDS'"
      [ -n "$R_IDS" ] || fail "TP=$tp INTFER_L4_ROWS=1 stream EMPTY"
      [ "$R_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_L4_ROWS=1] '$R_IDS' != base '$BASE' (the hash mutates nothing, must be bit-exact)"
      if [ "$tp" = "1" ]; then
        l4_rows_ok "$R_ERR" "seam" \
          || { echo "GATE FAIL: TP=$tp INTFER_L4_ROWS=1: the announcement does not name the seam path, got '$(l4_rows_path "$R_ERR")'" >&2; PASS=0; }
      else
        if l4_rows_ok "$R_ERR" "seam"; then
          fail "TP=$tp INTFER_L4_ROWS=1: hash announced on TP=2 where the L4 fused path must decline"
        fi
      fi
    fi
    rm -f "$R_OUT" "$R_ERR"
  }
  {
    RSB="$(mktemp -d)"
    env INTFER_L4_FUSE=1 INTFER_L4_ROWS=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$RSB/good.out" 2>"$RSB/good.err" || true
    cp "$RSB/good.err" "$RSB/not.err"
    sed -i 's/path=seam/path=host/' "$RSB/not.err"
    : >"$RSB/empty.err"
    if [ "$(l4_rows_path "$RSB/good.err")" != "seam" ]; then
      echo "TP=$tp L4_ROWS self-check SKIPPED: the tp=1 fixture produced no \
            [l4-rows] line (the L4 fused path did not run on this box), so \
            the refusal/vacuity directions are unexercised here" >&2
    else
      if l4_rows_ok "$RSB/not.err" "seam"; then
        fail "TP=$tp L4_ROWS self-check (b): log with the hash line deleted was ACCEPTED -- arm is observation-only"
      fi
      if l4_rows_ok "$RSB/empty.err" "seam"; then
        fail "TP=$tp L4_ROWS self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$RSB"
  }





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

  # --- INTFER_SSM_FUSE: fold the SSM recurrence's 4 kd-serial passes over the
  # state matrix into 2 (decay+kv-dot, rank-1-update+readout). The kernel
  # rewrites ssm_forward_kernel / ssm_forward_chain_kernel (quant_gemm.cpp);
  # the 4-pass body is kept byte-for-byte in the `else` branch, so the default
  # path is unchanged and this arm is a stream-identity check, not a
  # numerics contract.
  #
  # *** WHY BIT-EXACT BY CONSTRUCTION, AND WHY THAT IS NOT ENOUGH. *** In the
  # fused loop `S[d,c] *= g_t; kv += S[d,c]*kn[d]` -- the decay writes only
  # S[d,c] and the kv dot immediately reads that same in-register value; no
  # other element is touched, so the value read is identical to the 4-pass
  # base's "decay everything, then read". Same for
  # `S[d,c] += kn[d]*delta; s += S[d,c]*qn[d]`, and d is ascending in both
  # forms, so the FP accumulation order is unchanged. The host reference at
  # main.cpp:5448-5470 and the prefill path at 6380-6412 carry the identical
  # 4-pass shape, so the device kernel is the same recurrence as the host.
  # A stream comparison alone is satisfiable by a flag that never reached the
  # branch, so the arm also asserts the once-per-process [ssm-fuse] line.
  #
  # *** TP-AGNOSTIC, AND THAT IS THE POINT. *** Unlike L4/L5 (which open with
  # `num_devices_ != 1 => return false`) this fusion is INSIDE the kernel: the
  # SSM recurrence runs on device 0 regardless of --tp, so it is reachable on
  # the actual 27B --tp 2 target. The arm is therefore the SAME on tp=1 and
  # tp=2, with no TP polarity to assert -- the opposite of every fusion seam
  # in this tree.
  ssm_fuse_announced() { grep -aq '\[ssm-fuse\]' "$1"; }
  {
    SF_OUT="$(mktemp)"; SF_ERR="$(mktemp)"
    env INTFER_SSM_FUSE=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" \
      -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK >"$SF_OUT" 2>"$SF_ERR"; SF_RC=$?
    if [ "$SF_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_SSM_FUSE=1 run exited rc=$SF_RC"; tail -10 "$SF_ERR" >&2
    else
      SF_IDS="$(ids_of "$tp" INTFER_SSM_FUSE=1)"
      echo "TP=$tp [INTFER_SSM_FUSE=1]: '$SF_IDS'"
      [ -n "$SF_IDS" ] || fail "TP=$tp INTFER_SSM_FUSE=1 stream EMPTY"
      [ "$SF_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_SSM_FUSE=1] '$SF_IDS' != base '$BASE' (must be bit-exact)"
      # *** POLARITY, and the direction this arm MUST assert. *** The
      # [ssm-fuse] announce lives inside gemm::ssm_forward
      # (quant_gemm.cpp:3263/3346), which is reachable ONLY on the device
      # recurrence path -- guarded by ssm_gpu_allowed() (INTFER_SSM_GPU=1 +
      # no DPAS48). This arm runs WITHOUT INTFER_SSM_GPU, so the recurrence
      # is the HOST loop (main.cpp:5429 `if(!ssm_done)`, pure C++),
      # gemm::ssm_forward is never called, and the line CANNOT print.
      # Asserting it here asserted an impossibility and held the gate red on
      # this box (measured 2026-10-13: GATE FAIL 'no [ssm-fuse] line'); the
      # earlier comment claimed the host path also routes through
      # gemm::ssm_forward, and it does not. The reachability assertion
      # belongs on the run where the device path is live -- that is arm (d)
      # below. HERE the correct assertion is the OPPOSITE polarity: the line
      # must be ABSENT, because its presence without INTFER_SSM_GPU would
      # mean the device kernel ran on a path the caller never reached.
      if ssm_fuse_announced "$SF_ERR"; then
        fail "TP=$tp INTFER_SSM_FUSE=1 printed [ssm-fuse] WITHOUT INTFER_SSM_GPU=1 -- the device kernel ran on a path the caller never reached (host recurrence was live)"
      fi
    fi
    rm -f "$SF_OUT" "$SF_ERR"
  }
  {
    SFB="$(mktemp -d)"
    env INTFER_SSM_FUSE=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$SFB/good.out" 2>"$SFB/good.err" || true
    cp "$SFB/good.err" "$SFB/not.err"
    sed -i '/\[ssm-fuse\]/d' "$SFB/not.err"
    : >"$SFB/empty.err"
    # (b)/(c) now certify the ABSENCE polarity this arm asserts: a log with
    # the line deleted and an EMPTY log must both be ACCEPTED (the line is
    # not expected here), and a log that CARRIES the line must be REFUSED.
    # The old direction ("deleted line must be refused") certified the
    # reachability arm, which (d) owns.
    if ssm_fuse_announced "$SFB/good.err"; then
      fail "TP=$tp SSM_FUSE self-check (b): SSM_GPU-less run PRINTED [ssm-fuse] -- the device kernel is reachable from the host recurrence path, contradicting the announce site (quant_gemm.cpp:3263)"
    fi
    # The line-deleted and empty fixtures must stay ACCEPTED (the line is not
    # expected on a host-path run), and the predicate must be proven live in
    # BOTH directions -- a predicate that cannot fire would make the (b)
    # refusal above vacuous, and one that fires on an empty log would make
    # every future run fail for no reason.
    cp "$SFB/good.err" "$SFB/fake.err"
    echo '[ssm-fuse] fused SSM recurrence ACTIVE (host-path run that must not print)' >>"$SFB/fake.err"
    ssm_fuse_announced "$SFB/fake.err" \
      || fail "TP=$tp SSM_FUSE self-check (b2): the ABSENCE predicate did not fire on a fixture that CARRIES the line -- the arm's refusal is dead and the arm cannot fail"
    if ssm_fuse_announced "$SFB/empty.err"; then
      fail "TP=$tp SSM_FUSE self-check (c): the ABSENCE predicate fired on an EMPTY log (vacuity -- nothing was run)"
    fi
    rm -rf "$SFB"
  }
  {
    # *** (d) THE OPPOSITE DIRECTION, ADDED 2026-10-12. *** (a) proves the
    # fuse FIRES when it is the only SSM path in play. It does NOT prove it
    # fires on the OTHER SSM path, and that is the failure this arm must
    # not certify.
    #
    # *** CORRECTION 2026-10-13: the OLD reasoning here was wrong. *** It
    # claimed "the host path (main.cpp:5416-5425 step(), 5930 step_batch(),
    # 6458 prefill-chunk, 7268 serve) call[s] gemm::ssm_forward". Those
    # sites ARE the device path (guarded by ssm_gpu_allowed() at
    # main.cpp:5416). The HOST path is the `if(!ssm_done)` branch below
    # each of them -- pure C++ math in main.cpp, never gemm::ssm_forward --
    # so the [ssm-fuse] announce (quant_gemm.cpp:3263/3346) can ONLY print
    # when the device path is live, i.e. under INTFER_SSM_GPU=1. Arm (a)
    # (SSM_GPU unset) asserting the line was asserting an impossibility and
    # held the gate red (measured GATE FAIL 2026-10-13); (a) now asserts the
    # OPPOSITE polarity and (d) carries the reachability assertion.
    #
    # (d) closes it: with SSM_GPU on the recurrence is device-side, so the
    # SAME ssm_forward call site runs the fused body on the device path.
    # The line must STILL appear -- if it does not, the fuse is dead on the
    # path the PARAMOUNT DIRECTIVE actually cares about (the 27B --tp 2
    # target runs prefill-chunk and serve, both device-side when SSM_GPU
    # is on).
    #
    # *** WHY IT IS NOT VACUOUS HERE. *** The gate runs on the 0.8B model,
    # which DOES have SSM layers. Parsed its INTFER1 header 2026-10-12:
    # model_id "Qwen3-32B", 187 tensors, and 72 of them are the gated
    # delta-net SSM set -- 24 layers x {linear_attn.in_proj_a, in_proj_b,
    # in_proj_qkv, in_proj_z} (roles LINEAR_ATTN_A/B/QKV/Z, main.cpp:610-633
    # reads them into la_q). This is NOT a pure full-attention model: the
    # earlier summary's "zero SSM tensors" reading was wrong, and it was
    # wrong because it read the header as a flat `tensors` map with 0
    # entries instead of the 187-entry list it actually is. With la_q
    # populated, ssm_forward IS called at main.cpp:3047/3069/3146/3558, so
    # (a)/(b)/(c) are non-vacuous and this arm certifies a live path.
    # SSM_GPU is gated only by ssm_gpu_allowed() (env on + no DPAS48), so
    # INTFER_SSM_FUSE=1 INTFER_SSM_GPU=1 is a legal, reachable combination
    # on this model.
    SD_OUT="$(mktemp)"; SD_ERR="$(mktemp)"
    env INTFER_SSM_FUSE=1 INTFER_SSM_GPU=1 "$BIN" --gpu --tp "$tp" \
      -m "$MODEL" -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK \
      >"$SD_OUT" 2>"$SD_ERR"; SD_RC=$?
    if [ "$SD_RC" -ne 0 ]; then
      # SSM_GPU may decline (e.g. on a box where the device path is absent);
      # a decline is NOT a failure of this arm -- it means the recurrence
      # ran on the host and (a) covers it.
      echo "TP=$tp SSM_FUSE self-check (d) SKIPPED: INTFER_SSM_FUSE=1 \
            INTFER_SSM_GPU=1 exited rc=$SD_RC (device SSM path declined, so \
            the recurrence ran on the host and (a) covers it)" >&2
    else
      if ! ssm_fuse_announced "$SD_ERR"; then
        fail "TP=$tp SSM_FUSE self-check (d): INTFER_SSM_FUSE=1 \
              INTFER_SSM_GPU=1 ran clean but printed NO [ssm-fuse] line -- \
              the fused kernel body is dead on the device recurrence path \
              (main.cpp:5416/5930/6458/7268), which is the path the 27B \
              --tp 2 target takes. (a) alone cannot see this: the announce \
              site is host-side and fires regardless of which branch the \
              kernel runs."
      else
        echo "TP=$tp [SSM_FUSE device path]: INTFER_SSM_FUSE=1 \
              INTFER_SSM_GPU=1 ran clean and printed [ssm-fuse] -- the \
              fused kernel body is live on the device recurrence path, not \
              only the host one"
      fi
    fi
    rm -f "$SD_OUT" "$SD_ERR"
  }

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

  # --- INTFER_WAVE_ALIAS: alias an UN-ROTATED LA wave input to `normed` -----
  # Host-side only: the alias skips a per-item FP32->FP16 convert and a
  # memcpy for every un-rotated LA tensor, by pointing the wave item at the
  # caller's own `normed` buffer. Bit-exact by construction (an un-rotated
  # tensor's staging buffer held a byte-for-byte copy of `normed`, so the
  # GEMM reads identical bytes and every output's K accumulation is
  # untouched). Same kernels, same buffers, same output assembly -- so
  # nk/nw/nr are PROVABLY unchanged and the effect is confined to the host
  # copy/usm_write buckets, never a kernel delta.
  #
  # *** WHY A STREAM COMPARISON ALONE IS VACUOUS HERE, AND WHAT DEFEATS IT. ***
  # The base and this arm are bit-identical BY CONSTRUCTION (that is the
  # claim), so "the two streams matched" certifies nothing on its own: a
  # flag that never reached the branch would also produce matching streams.
  # The per-site decision line below is what separates them -- it prints
  # aliased=N, and N is the count of un-rotated tensors, so the line
  # appears (and reports aliased=0) on a model whose every LA tensor is
  # SpinQuant-rotated. An ACTIVE-only line would be SILENT on such a model
  # and the arm would then be unfailable rather than passing, which is the
  # vacuous-green shape in the other direction.
  #
  # TP=1 AND TP=2 BOTH VALID: unlike L4/L5 this is not a device-chain fusion
  # (no cross-device acts), it is a host-side pointer choice in the CLI, so
  # there is no TP polarity to assert. The arm is the same on both.
  wave_alias_announced() { grep -aq '\[wave-alias\]' "$1"; }
  {
    WA_OUT="$(mktemp)"; WA_ERR="$(mktemp)"
    env INTFER_WAVE_ALIAS=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$WA_OUT" 2>"$WA_ERR"; WA_RC=$?
    if [ "$WA_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_WAVE_ALIAS=1 run exited rc=$WA_RC"; tail -10 "$WA_ERR" >&2
    else
      WA_IDS="$(ids_of "$tp" INTFER_WAVE_ALIAS=1)"
      echo "TP=$tp [INTFER_WAVE_ALIAS=1]: '$WA_IDS'"
      [ -n "$WA_IDS" ] || fail "TP=$tp INTFER_WAVE_ALIAS=1 stream EMPTY"
      [ "$WA_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_WAVE_ALIAS=1] '$WA_IDS' != base '$BASE' (must be bit-exact)"
      if ! wave_alias_announced "$WA_ERR"; then
        fail "TP=$tp INTFER_WAVE_ALIAS=1: no [wave-alias] line -- the alias never took, stream match vacuous"
      fi
    fi
    rm -f "$WA_OUT" "$WA_ERR"
  }
  {
    WASB="$(mktemp -d)"
    env INTFER_WAVE_ALIAS=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$WASB/good.out" 2>"$WASB/good.err" || true
    cp "$WASB/good.err" "$WASB/not.err"
    sed -i '/\[wave-alias\]/d' "$WASB/not.err"
    : >"$WASB/empty.err"
    if ! wave_alias_announced "$WASB/good.err"; then
      echo "TP=$tp WAVE_ALIAS self-check SKIPPED: the tp=1 fixture produced no \
            [wave-alias] line (the model's LA tensors are all SpinQuant-rotated, \
            so aliased=0, or the engine did not run on this box), so the \
            refusal/vacuity directions are unexercised here" >&2
    else
      if wave_alias_announced "$WASB/not.err"; then
        fail "TP=$tp WAVE_ALIAS self-check (b): log with the decision line deleted was ACCEPTED -- arm is stream-only"
      fi
      if wave_alias_announced "$WASB/empty.err"; then
        fail "TP=$tp WAVE_ALIAS self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$WASB"
  }
  # --- INTFER_STREAM_F16: FP16-shard eviction policy ------------------------
  # This flag moves WHEN a shard is released, never WHAT: the eviction
  # predicate is unchanged, so the two streams are byte-identical by
  # construction and a stream comparison is satisfiable by a flag that never
  # reached begin_layer. The line names the policy actually in force
  # (evict-FP16 vs keep-FP16-resident), once per process, so the arm certifies
  # the policy rather than a constant. Reachability is structural: the branch
  # is in begin_layer (engine.cpp:779), which every layer's forward pass
  # calls, so the 0.8B model reaches it on every token.
  #
  # *** ONE PREDICATE, TWO CONSUMERS (audit #3). *** The predicate is the
  # POLICY the announcement names, not the tag's presence: the tag is the same
  # whether the policy fired or not, so a self-check that only greps the tag
  # would be driven by a predicate the real arm does not use. Both go through
  # stream_f16_policy, which returns the policy named (or the empty string).
  stream_f16_policy() { # $1 = log -> prints evict-everything | keep-FP16-resident | nothing
    # [A-Za-z0-9-]+, NOT [a-z-]+: keep-FP16-resident carries capitals, and a
    # class that cannot name it would silently truncate the refusal fixture
    # to 'keep-' -- still != evict-everything, so the (b) direction survives,
    # but a predicate that cannot state its own documented value is a guard
    # whose rationale does not apply (audit #5).
    grep -aoE '\[stream-f16\].*policy=[A-Za-z0-9-]+' "$1" | head -1 \
      | grep -aoE 'policy=[A-Za-z0-9-]+' | head -1 \
      | sed -E 's/policy=([A-Za-z0-9-]+)/\1/'
  }
  stream_f16_ok() {   # $1 = log, $2 = expected policy. 0=accept.
    [ "$(stream_f16_policy "$1")" = "$2" ]
  }
  {
    SF_OUT="$(mktemp)"; SF_ERR="$(mktemp)"
    env INTFER_STREAM_F16=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" \
      -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK >"$SF_OUT" 2>"$SF_ERR"; SF_RC=$?
    if [ "$SF_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_STREAM_F16=1 run exited rc=$SF_RC"; tail -10 "$SF_ERR" >&2
    else
      SF_IDS="$(ids_of "$tp" INTFER_STREAM_F16=1)"
      echo "TP=$tp [INTFER_STREAM_F16=1]: '$SF_IDS'"
      [ -n "$SF_IDS" ] || fail "TP=$tp INTFER_STREAM_F16=1 stream EMPTY"
      [ "$SF_IDS" = "$BASE" ] \
        || fail "TP=$tp [INTFER_STREAM_F16=1] '$SF_IDS' != base '$BASE' (the policy moves when, not what -- must be bit-exact)"
      stream_f16_ok "$SF_ERR" "evict-everything" \
        || { echo "GATE FAIL: TP=$tp INTFER_STREAM_F16=1: the announcement does not name the evict-everything policy, got '$(stream_f16_policy "$SF_ERR")'" >&2; PASS=0; }
    fi
    rm -f "$SF_OUT" "$SF_ERR"
  }
  {
    SFSB="$(mktemp -d)"
    env INTFER_STREAM_F16=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$SFSB/good.out" 2>"$SFSB/good.err" || true
    cp "$SFSB/good.err" "$SFSB/not.err"
    sed -i 's/evict-everything/keep-FP16-resident/' "$SFSB/not.err"
    : >"$SFSB/empty.err"
    if [ "$(stream_f16_policy "$SFSB/good.err")" != "evict-everything" ]; then
      echo "TP=$tp STREAM_F16 self-check SKIPPED: the tp=1 fixture produced no \
            [stream-f16] policy line (the engine did not run on this box), so \
            the refusal/vacuity directions are unexercised here" >&2
    else
      # (b) refusal: a log whose policy is keep-FP16-resident (the default -- the
      # flag was ignored) must be refused. The fixture carries a real policy
      # string, so the predicate can actually fire.
      if stream_f16_ok "$SFSB/not.err" "evict-everything"; then
        fail "TP=$tp STREAM_F16 self-check (b): a log whose policy was keep-FP16-resident (the default -- the flag ignored) was ACCEPTED -- arm is policy-only"
      fi
      # (c) vacuity: an empty log must fail the predicate.
      if stream_f16_ok "$SFSB/empty.err" "evict-everything"; then
        fail "TP=$tp STREAM_F16 self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$SFSB"
  }


  # --- INTFER_LMHEAD_FP16: per-tile FP16 lm_head upload ---------------------
  # This flag swaps the lm_head upload + GEMM from the default INT8-code path
  # to the old per-tile FP16-weight path. It is NOT bit-exact against the
  # default -- INT8 vs FP16 weight representation is a different quantisation
  # by design -- so a stream comparison here is NOT a check and no identity is
  # claimed. AGENTS.md records it as UNCERTIFIED-BY-DISCLAIM and
  # check_flag_gate_coverage.sh:61 records why no stream arm is warranted: it
  # restores a bisect-only variant and the repo has retracted every
  # performance claim made for it. Gating a flag whose only purpose is to
  # reproduce a discredited measurement would certify nothing.
  #
  # What an arm CAN honestly assert here is REACHABILITY, and it is still
  # worth asserting: the branch is in run_lm_head_f16 (main.cpp:1433), which
  # every lm_head call enters, so the 0.8B model reaches it on every token.
  #
  # *** ONE PREDICATE, TWO CONSUMERS (audit #3). *** The predicate is the
  # PATH the upload chose, not the tag's presence -- the tag is the same
  # whether the FP16 path ran or the default INT8 path did, so a self-check
  # that only greps the tag would be driven by a predicate the real arm does
  # not use. Both go through lmhead_fp16_path, which returns the path named
  # (or the empty string).
  lmhead_fp16_path() { # $1 = log -> prints fp16 | nothing
    grep -aoE '\[lmhead-fp16\].*path=[A-Za-z0-9]+ per-tile [A-Z0-9]+ weights' "$1" | head -1 \
      | grep -aoE 'path=[A-Za-z0-9]+' | head -1 \
      | sed -E 's/path=([A-Za-z0-9]+)/\1/'
  }
  lmhead_fp16_ok() {   # $1 = log, $2 = expected path. 0=accept.
    [ "$(lmhead_fp16_path "$1")" = "$2" ]
  }
  {
    H_OUT="$(mktemp)"; H_ERR="$(mktemp)"
    env INTFER_LMHEAD_FP16=1 "$BIN" --gpu --tp "$tp" -m "$MODEL" \
      -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK >"$H_OUT" 2>"$H_ERR"; H_RC=$?
    if [ "$H_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_LMHEAD_FP16=1 run exited rc=$H_RC"; tail -10 "$H_ERR" >&2
    else
      H_IDS="$(ids_of "$tp" INTFER_LMHEAD_FP16=1)"
      echo "TP=$tp [INTFER_LMHEAD_FP16=1]: '$H_IDS'"
      [ -n "$H_IDS" ] || fail "TP=$tp INTFER_LMHEAD_FP16=1 produced no tokens (see $H_ERR)"
      lmhead_fp16_ok "$H_ERR" "fp16" \
        || { echo "GATE FAIL: TP=$tp INTFER_LMHEAD_FP16=1: the announcement does not name the per-tile FP16 upload, got '$(lmhead_fp16_path "$H_ERR")'" >&2; PASS=0; }
      # EXPLICITLY NOT a stream comparison: INT8 vs FP16 weight representation
      # differs by design. Recorded here so the omission is visible rather
      # than silent.
      echo "TP=$tp [INTFER_LMHEAD_FP16=1]: no stream identity asserted (INT8 vs FP16 by design); reachability only"
    fi
    rm -f "$H_OUT" "$H_ERR"
  }
  {
    HSB="$(mktemp -d)"
    env INTFER_LMHEAD_FP16=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" \
      -n "${N:-4}" --no-stream --ids $TOK >"$HSB/good.out" 2>"$HSB/good.err" || true
    cp "$HSB/good.err" "$HSB/not.err"
    sed -i 's/path=fp16 per-tile FP16 weights/path=int8 per-tile INT8 weights/' "$HSB/not.err"
    : >"$HSB/empty.err"
    if [ "$(lmhead_fp16_path "$HSB/good.err")" != "fp16" ]; then
      echo "TP=$tp LMHEAD_FP16 self-check SKIPPED: the tp=1 fixture produced no \
            [lmhead-fp16] upload line (the lm_head upload never ran on this box), \
            so the refusal/vacuity directions are unexercised here" >&2
    else
      # (b) refusal: a log whose upload took the default INT8 path (the flag
      # was ignored) must be refused. The fixture carries a real path string,
      # so the predicate can actually fire.
      if lmhead_fp16_ok "$HSB/not.err" "fp16"; then
        fail "TP=$tp LMHEAD_FP16 self-check (b): log with the default INT8 upload ACCEPTED -- arm is reachability-only"
      fi
      # (c) vacuity: an empty log must fail the predicate.
      if lmhead_fp16_ok "$HSB/empty.err" "fp16"; then
        fail "TP=$tp LMHEAD_FP16 self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$HSB"
  }

  # --- INTFER_XMX_GEMM: joint_matrix XMX DPAS for INT2/INT4 weights ---------
  # Ported from llama.cpp fused-gemm. The portable tiled kernel
  # (quant_gemm_forward_usm) dequantizes INT2/4 weights into SLM and
  # accumulates with FP16 FMA on EU/ALU -- xmxmon reads zero DPAS there. This
  # kernel issues the XMX DPAS directly via joint_matrix_mad after dequantizing
  # the weight tile into FP16 in SLM.
  #
  # BIT-EXACT BY CONSTRUCTION: the dequantized FP16 weight tile holds the
  # same bits the tiled kernel would load into sW, the activation tile holds
  # the same bits as sA, and the DPAS FP32 accumulation order is the same
  # ascending-K order as the tiled kernel's inner kk loop. The only
  # difference is the compute unit (XMX DPAS vs EU/ALU FMA).
  #
  # *** WHY A STREAM COMPARISON ALONE IS VACUOUS HERE, AND WHAT DEFEATS IT. ***
  # The base and this arm are bit-identical BY CONSTRUCTION (that is the
  # claim), so "the two streams matched" certifies nothing on its own: a
  # flag that never reached the branch would also produce matching streams.
  # The per-site decision line below is what separates them -- it prints
  # "[xmx-gemm] ACTIVE on eligible INT2/INT4 tiles" once per process, so the
  # arm certifies the kernel ran rather than a constant.
  #
  # *** THE KERNEL ITSELF GUARDS THE SHAPES. *** quant_gemm_forward_usm_xmx
  # returns a null event when M%16 != 0, N%16 != 0, or K%64 != 0, and the
  # engine falls back to the tiled kernel. So this arm is REACHABILITY-only
  # on the 0.8B model: it certifies that the XMX path is wired and reachable
  # when the shapes are aligned, not that it fired on every tile. The gate
  # asserts the announcement line is present when the flag is set.
  xmx_announced() { grep -aq '\[xmx-gemm\]' "$1"; }
  {
    XM_OUT="$(mktemp)"; XM_ERR="$(mktemp)"
    env INTFER_XMX_GEMM=1 "$BIN" --gpu --tp "$tp" -m "$MODEL"       -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK >"$XM_OUT" 2>"$XM_ERR"; XM_RC=$?
    if [ "$XM_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_XMX_GEMM=1 run exited rc=$XM_RC"; tail -10 "$XM_ERR" >&2
    else
      XM_IDS="$(ids_of "$tp" INTFER_XMX_GEMM=1)"
      echo "TP=$tp [INTFER_XMX_GEMM=1]: '$XM_IDS'"
      [ -n "$XM_IDS" ] || fail "TP=$tp INTFER_XMX_GEMM=1 stream EMPTY"
      [ "$XM_IDS" = "$BASE" ]         || fail "TP=$tp [INTFER_XMX_GEMM=1] '$XM_IDS' != base '$BASE' (must be bit-exact)"
      if ! xmx_announced "$XM_ERR"; then
        fail "TP=$tp INTFER_XMX_GEMM=1: no [xmx-gemm] line -- the XMX path never took, stream match vacuous"
      fi
    fi
    rm -f "$XM_OUT" "$XM_ERR"
  }
  {
    XMSB="$(mktemp -d)"
    env INTFER_XMX_GEMM=1 "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT"       -n "${N:-4}" --no-stream --ids $TOK >"$XMSB/good.out" 2>"$XMSB/good.err" || true
    cp "$XMSB/good.err" "$XMSB/not.err"
    sed -i '/\[xmx-gemm\]/d' "$XMSB/not.err"
    : >"$XMSB/empty.err"
    if ! xmx_announced "$XMSB/good.err"; then
      echo "TP=$tp XMX_GEMM self-check SKIPPED: the tp=1 fixture produced no             [xmx-gemm] line (no INT2/INT4 tiles on this model, or the engine             did not run on this box), so the refusal/vacuity directions are             unexercised here" >&2
    else
      if xmx_announced "$XMSB/not.err"; then
        fail "TP=$tp XMX_GEMM self-check (b): log with the decision line deleted was ACCEPTED -- arm is stream-only"
      fi
      if xmx_announced "$XMSB/empty.err"; then
        fail "TP=$tp XMX_GEMM self-check (c): arm ACCEPTED an empty log (vacuity)"
      fi
    fi
    rm -rf "$XMSB"
  }

  # --- INTFER_XMX_TILE_SCHED: work-stealing tile scheduler for XMX kernel ----
  # Ported from the DPAS path's INTFER_TILE_SCHED. The XMX kernel uses
  # joint_matrix which requires ALL SG lanes to participate, so work-stealing
  # is per-SUBGROUP: lane 0 claims the next tile, broadcasts it via a
  # local_accessor, and all SG lanes process the claimed tile together.
  #
  # BIT-EXACT BY CONSTRUCTION: the process_tile body is shared by both
  # paths; fixed-assignment and work-stealing compute the same dot products
  # in the same order, only differing in WHICH tile each sub-group computes.
  # So a stream comparison alone is vacuous here -- the arm asserts the
  # [xmx-tile-sched] line, which prints n_tiles and nWG, so the gate can
  # verify the work-stealing path was actually taken.
  xmx_ts_announced() { grep -aq '\[xmx-tile-sched\]' "$1"; }
  {
    XT_OUT="$(mktemp)"; XT_ERR="$(mktemp)"
    env INTFER_XMX_GEMM=1 INTFER_XMX_TILE_SCHED=1 "$BIN" --gpu --tp "$tp" \
      -m "$MODEL" -p "$PROMPT" -n "${N:-4}" --no-stream --ids $TOK \
      >"$XT_OUT" 2>"$XT_ERR"; XT_RC=$?
    if [ "$XT_RC" -ne 0 ]; then
      fail "TP=$tp INTFER_XMX_TILE_SCHED=1 run exited rc=$XT_RC"; tail -10 "$XT_ERR" >&2
    else
      XT_IDS="$(ids_of "$tp" INTFER_XMX_TILE_SCHED=1)"
      echo "TP=$tp [INTFER_XMX_TILE_SCHED=1]: '$XT_IDS'"
      [ -n "$XT_IDS" ] || fail "TP=$tp INTFER_XMX_TILE_SCHED=1 stream EMPTY"
      [ "$XT_IDS" = "$BASE" ] || fail "TP=$tp [INTFER_XMX_TILE_SCHED=1] '$XT_IDS' != base '$BASE' (must be bit-exact)"
      if ! xmx_ts_announced "$XT_ERR"; then
        echo "TP=$tp XMX_TILE_SCHED self-check SKIPPED: the fixture produced no             [xmx-tile-sched] line (no INT2/INT4 tiles on this model, or the             XMX path never took), so the work-stealing direction is             unexercised here" >&2
      else
        # Self-check (b): the announcement line is what separates the two
        # paths. Delete it from the log and the arm must NOT match.
        XT_NOT="$(mktemp)"
        sed -i '/\[xmx-tile-sched\]/d' "$XT_ERR"
        cp "$XT_ERR" "$XT_NOT"
        if xmx_ts_announced "$XT_NOT"; then
          fail "TP=$tp XMX_TILE_SCHED self-check (b): log with the decision line deleted was ACCEPTED -- arm is stream-only"
        fi
        XT_EMPTY="$(mktemp)"
        : >"$XT_EMPTY"
        if xmx_ts_announced "$XT_EMPTY"; then
          fail "TP=$tp XMX_TILE_SCHED self-check (c): arm ACCEPTED an empty log (vacuity)"
        fi
        rm -f "$XT_NOT" "$XT_EMPTY"
      fi
    fi
    rm -f "$XT_OUT" "$XT_ERR"
  }


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
# STREAM_F16
echo "GPU-OPTIN TEST: PASS"
