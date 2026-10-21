#!/usr/bin/env bash
# GA64 reciprocal-identity gate (CPU-only, no model, no GPU).
#
# INTFER_DPAS_PREQ's device act-quant pre-pass does not quantize on the device:
# it consumes (scale, inverse) pairs produced on the host by
# dashq::act_quant_ga64_scale_inv_batch() and only applies
#   v = clamp(x*inv, +-127); q = (int8)(v >= 0 ? v+0.5f : v-0.5f).
# That is sound only if the shipped `inv` is BIT-IDENTICAL to the one
# act_quant_ga64_batch() (the stream oracle) uses internally: GA64's
# q = trunc(x*inv +- 0.5) turns a 1-ulp reciprocal difference into a +-1 code
# error at exact .5 ties, which changes the greedy stream. `1.0f/s` is not
# guaranteed to yield the same bits at two call sites (device lowers it to
# rcp.approx; codegen context matters on the host), so dashq routes every GA64
# reciprocal through ONE noinline site (ga64_recip).
#
# This gate is the numeric backstop for that invariant: it recovers the
# reciprocal the oracle effectively used (re-quantizing with inv-1ulp / inv /
# inv+1ulp) and fails if the helper drifts by even one ulp, or if its scales or
# codes disagree, over random / tie-adversarial / edge / non-multiple-of-64
# datasets and multi-row batches. Runs in well under a second.
#
# BUILD-FIRST (added after this gate was caught validating a STALE binary):
# the script used to run whatever $BIN happened to be on disk, so a source
# edit that failed to compile left the previous binary in place and the gate
# happily printed PASS for code that no longer existed. That is the same
# failure class as audit #3c ("gates cannot protect a configuration they
# cannot exercise"), and it is how the row-coverage hole below survived three
# commits. This target is 2 translation units and builds in seconds.
# Opt out with INTFER_GATE_NO_BUILD=1 only when the build is known current.
#
# Usage: scripts/gates/test_ga64_recip.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/ga64_recip_test/ga64_recip_test"

. "$SCRIPT_DIR/../lib/_gate_build.sh"
gate_require_build ga64_recip_test || exit 1

[ -x "$BIN" ] || { echo "FAIL: ga64_recip_test binary missing: $BIN" >&2; exit 1; }
"$BIN"
RC=$?
if [ "$RC" = "0" ]; then
  echo "GA64-RECIP TEST: PASS"
else
  echo "GA64-RECIP TEST: FAIL (rc=$RC)" >&2
fi
exit "$RC"
