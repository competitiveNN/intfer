#!/usr/bin/env bash
# DPAS row-fold identity gate (needs a GPU; no model required).
#
# INTFER_DPAS48_FOLD=1 hoists the per-(row,op) activation column sums out of
# quant_gemm_dpas48's 16-N tile loop. That is a pure WORK hoist, and the safety
# argument is entirely numeric: the hoisted values are EXACT int32 sums of int8
# codes (|sum| <= 32*128 = 4096), and integer addition is exact and
# associative, so moving the sum cannot change a bit. There is no
# rounding-order exposure, unlike the FP32 folds elsewhere in that file.
#
# But "exact sums" is a claim about VALUES, and no stream-identity gate can
# check it: DPAS48 is default-OFF and its activation numerics differ from the
# FP16-act base by design, so there is no cheap end-to-end invariant to compare.
# This gate therefore compares the values directly -- every (row,op) S0/Sfull
# the kernel produces against a host reference summing in the same element
# order -- over M=1/3/4/8, tight and row-strided workspaces, and big-K, plus
# four NEGATIVE cases that must DECLINE and write nothing.
#
# The M>1 cases are not optional decoration. The layout bug this gate was
# written for is INVISIBLE at M=1 (a single row cannot be overwritten): each row
# owns 2*nOp ints, [S0][Sfull], so a row stride below 2*nOp lets row m's Sfull
# block overwrite row m+1's S0 block. That fires only when the grow-only
# workspace stride drops under the call's 2*nOp, i.e. when a smaller-K GEMM
# follows a larger one. The first run failed on `M=8 big-K` with S0 always
# correct and Sfull garbage -- exactly that signature.
#
# Decline detection uses a sentinel prerequisite event (the kernel returns `dep`
# unchanged when it declines) because comparing a returned event against a bare
# `sycl::event{}` is not reliable across implementations.
#
# BUILD-FIRST: same stale-artifact class as the other nine gates. A failed build
# leaves the previous probe binary in place and runnable, so an unguarded gate
# would report PASS for code that no longer exists -- observed live twice in
# this session. Target is 1 TU plus intfer_gemm.
# Opt out with INTFER_GATE_NO_BUILD=1 only when the build is known current.
#
# Usage: scripts/test_dpas_fold.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/examples/dpas_fold_probe/dpas_fold_probe"

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build dpas_fold_probe || exit 1

[ -x "$BIN" ] || { echo "FAIL: dpas_fold_probe binary missing: $BIN" >&2; exit 1; }

# setvars.sh puts umf on LD_LIBRARY_PATH; without it the runtime enumerates zero
# devices. Do not "fix" device_manager.cpp if you see that warning.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

# The probe's 3 device readback regimes are the whole point of this gate, so a
# box it cannot reach must SKIP (exit 2) rather than fail red -- a red line here
# would be indistinguishable from a real regression. Found by RE-DERIVING the
# gate list instead of remembering it: this gate said "needs a GPU" in a COMMENT
# and carried no guard at all, so it was the one GPU gate that would have gone
# red on a 0-GPU box. See gate_inventory.sh's NEEDS-GPU column.
. "$SCRIPT_DIR/_gate_precond.sh"
gate_require_device || exit 2

"$BIN"
RC=$?
if [ "$RC" = "0" ]; then
  echo "DPAS-FOLD TEST: PASS"
else
  echo "DPAS-FOLD TEST: FAIL (rc=$RC) -- INTFER_DPAS48_FOLD must stay OFF" >&2
fi
exit "$RC"
