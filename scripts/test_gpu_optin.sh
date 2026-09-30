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
# Usage: scripts/test_gpu_optin.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PROMPT="The capital of France is"
TOK="--tokenizer-py $SCRIPT_DIR/bpe_tokenize.py --detok-py $SCRIPT_DIR/bpe_detokenize.py"

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
  echo "SKIP: no render-node access from this shell."
  exit 2
fi
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

PASS=1
fail() { echo "GATE FAIL: $1" >&2; PASS=0; }
ids_of() { # $1=tp $2=env... -- prints decode ids
  local tp="$1"; shift
  env "$@" "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n "${N:-4}" \
    --no-stream --ids $TOK 2>&1 >/dev/null \
    | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
    | tr '\n' ' ' | sed 's/ *$//'
}

for tp in 1 2; do
  BASE="$(ids_of "$tp")"
  echo "TP=$tp base: '$BASE'"
  [ -n "$BASE" ] || fail "TP=$tp base empty (broken GPU run)"
  for mode in "INTFER_INT8_KV=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1" "INTFER_SSM_GPU=1"; do
    # shellcheck disable=SC2086
    GOT="$(env $mode "$BIN" --gpu --tp "$tp" -m "$MODEL" -p "$PROMPT" -n 4 \
      --no-stream --ids $TOK 2>&1 >/dev/null \
      | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
      | tr '\n' ' ' | sed 's/ *$//')"
    echo "TP=$tp [$mode]: '$GOT'"
    [ "$GOT" = "$BASE" ] || fail "TP=$tp [$mode] '$GOT' != base '$BASE'"
  done

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
  # a .5 tie. The host's `1.0f/s` is codegen-dependent (measured 1 ulp off the
  # correctly rounded value in this binary, checked against exact rational
  # arithmetic) while a device `1.0f/s` lowers to rcp.approx, so the two sides
  # disagreed by one code on ~2 elements per ~1950 GEMM calls. The pre-pass now
  # consumes the HOST's reciprocal verbatim (ActQuantDevParams::inv_in via
  # dashq::act_quant_ga64_inv_batch) instead of dividing, which is bit-exact by
  # construction. Do not "simplify" that back into a device division.
  GOT="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_DPAS_PREQ=1 INTFER_SSM_GPU=1)"
  echo "TP=$tp [DPAS48+PREQ+SSM_GPU n=24]: '$GOT'"
  [ "$GOT" = "$D48" ] || fail "TP=$tp [DPAS48+PREQ+SSM_GPU] '$GOT' != DPAS48 '$D48'"

  # SSM_GPU alone against the DPAS48 host-act-quant baseline.
  GOT="$(N=24 ids_of "$tp" INTFER_DPAS48=1 INTFER_SSM_GPU=1)"
  echo "TP=$tp [DPAS48+SSM_GPU n=24]: '$GOT'"
  [ "$GOT" = "$D48" ] || fail "TP=$tp [DPAS48+SSM_GPU] '$GOT' != DPAS48 '$D48'"

  # PREQ without DPAS48 must be inert (nested-flag guard).
  PQ="$(N=24 ids_of "$tp" INTFER_DPAS_PREQ=1)"
  B24="$(N=24 ids_of "$tp")"
  echo "TP=$tp [base n=24]: '$B24'"
  echo "TP=$tp [INTFER_DPAS_PREQ=1 alone]: '$PQ'"
  [ "$PQ" = "$B24" ] || fail "TP=$tp PREQ-alone '$PQ' != base '$B24' (PREQ must be inert without DPAS48)"
done

if [ "$PASS" = 1 ]; then echo "GPU-OPTIN TEST: PASS"; else echo "GPU-OPTIN TEST: FAIL" >&2; exit 1; fi
