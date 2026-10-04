#!/usr/bin/env bash
# K-split numerics contract gate (CPU-only, no model, no GPU, ~1s).
#
# 27B M=1 decode is EU-parallelism-bound (AGENTS.md), and the one remaining
# lever is splitting K across work-items -- which raises parallelism AND
# coalesces the weight loads, but reassociates the FP32 sum. AGENTS.md records
# that this "needs a tolerance-based numerics contract agreed BEFORE any code
# is written". This gate is that contract, and it is a GATE rather than a note
# so that the agreement is re-derivable instead of being prose somebody can
# reinterpret.
#
# What it certifies, all measured:
#   * On realistic activations (random / act-outliers / full cancellation) a
#     K-split does not move the FP16 store AT ALL. The FP16 activation bound
#     (|a| <= 65504) caps the dot product, which puts FP32 reassociation error
#     ~2 orders of magnitude below one FP16 ulp.
#   * split{S} and tree{S} are DIFFERENT tolerances (identical partials,
#     different combination order), so a kernel states which one it uses.
#   * It is NOT unconditional: an escalating-magnitude dataset drives the
#     consumed margin past 1 FP16 ulp, so the honest claim is "stream-identical
#     on the gate models", never "bit-exact for all inputs".
#
# This gate deliberately does NOT test the engine. It is a NUMERICS gate, in the
# same class as test_ga64_recip.sh: it certifies a property of an arithmetic
# reordering that a future kernel would perform, before that kernel exists.
#
# BUILD-FIRST, like every other artifact consumer (audit #3c / #4c family): run
# whatever was on disk and a failed compile leaves the previous binary
# reporting PASS for code that no longer exists. Opt out with
# INTFER_GATE_NO_BUILD=1 only when the build is known current.
#
# Usage: scripts/test_ksplit_contract.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/ksplit_contract/ksplit_contract"

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build ksplit_contract || exit 1

[ -x "$BIN" ] || { echo "FAIL: ksplit_contract binary missing: $BIN" >&2; exit 1; }

# --self-check drives the harness's own assertions (S=1 reproduces seq, the
# adversarial case is not reassociated away by the compiler, fp16_ulp fires and
# stays quiet, and it is not the truncating version). Run it FIRST and gate on
# it: the full run prints PASS whether or not the harness can see anything, so
# a self-check that is not required is a self-check that is not run.
"$BIN" --self-check >/dev/null 2>&1
if [ "$?" != "0" ]; then
  echo "FAIL: ksplit_contract --self-check did not pass; the harness cannot" >&2
  echo "      certify the contract, so the full run below proves nothing." >&2
  "$BIN" --self-check 2>&1 | sed -n '/SELF-CHECK/,$p' >&2
  exit 1
fi

"$BIN"
RC=$?
if [ "$RC" = "0" ]; then
  echo "KSPLIT-CONTRACT TEST: PASS"
else
  echo "KSPLIT-CONTRACT TEST: FAIL (rc=$RC)" >&2
fi
exit "$RC"
