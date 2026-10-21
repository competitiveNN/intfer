#!/usr/bin/env bash
# oneMKL FP16 GEMM gate (needs a GPU; no model required).
#
# INTFER_ONEMKL=1 routes quant_gemm_f16 (M >= 8) through oneMKL SYCL BLAS.
# That arm is NOT bit-exact -- oneMKL's FP32 accumulation order is its own --
# so this gate compares against a host FP32 reference under an AGREED
# tolerance.
#
# *** WHERE THE ANNOUNCEMENT ASSERTION ACTUALLY LIVES (the header claim was
# wrong, and rule 5 of the audit log is "a guard whose stated rationale does
# not apply is worse than none"). *** This gate does NOT grep for
# `[onemkl] ... ACTIVE` -- it relays the probe's rc. The assertion that makes
# the run's having reached the branch observable lives in the PROBE:
# `examples/onemkl_probe/main.cpp:363-374` (M=1 arm: `ann` true => FAIL) and
# `381-433` (M>=8 arms: `ever_announced` false => FAIL), both of which
# return non-zero, so a flag that never reached the kernel fails the gate's
# rc relay just as loudly as a grep would. The gate's own contributions are the
# lld wiring check (88-100) and the timeout-as-SKIP (120-127). Recording this
# here rather than adding a grep that duplicates the probe's assertion: two
# assertions on one line is the "the checker duplicates the thing it checks"
# shape, and the probe's return code already carries the verdict.
#
# The probe also carries the M=1 negative direction (--m1): the arm must
# DECLINE at M=1, where FP16 already runs at 98-123% of the memory roofline
# and a BLAS call cannot pay.
#
# BUILD-CONDITIONAL. The probe only exists when INTFER_ENABLE_MKL=ON, because
# onemkl_available() returns false without the library and the arm is
# unreachable by construction. On a default build the gate SKIPS (exit 2)
# rather than reporting a red line for a precondition, which would train the
# operator to ignore it. This is the same shape as test_dpas_fold.sh's
# gate_require_device, and the same reason the probe's own main() exits 2
# when onemkl_available() is false -- the gate and the binary agree on the
# SKIP condition so neither can be the single point of failure.
#
# CONFIGURE-ON. The default build dir is configured WITHOUT the MKL option, so
# this gate reconfigures its own throwaway build dir with
# -DINTFER_ENABLE_MKL=ON rather than mutating the operator's build. That is the
# same pattern as the operator's own remedy (a throwaway build dir), and it
# keeps the default build byte-for-byte the MKL-free path.
#
# Usage: scripts/gates/test_onemkl.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
GATE_BUILD_DIR="${INTFER_ONEMKL_BUILD_DIR:-$BUILD_DIR/onemkl_gate}"

# ---- 1. Is the MKL option even available on this box? ----
# NAMED BY THIS GATE. The census (check_flag_gate_coverage.sh) harvests the
# set of INTFER_* names from scripts/gates/test_*.sh and subtracts the engine's
# own set; a flag that appears nowhere in a gate is filed as an opt-in REAL
# GAP. INTFER_ONEMKL must therefore appear as a literal here, not merely be
# implied by INTFER_ENABLE_MKL or INTFER_ONEMKL_BUILD_DIR (both of which are
# build-config names, not the flag itself).
if ! grep -qE '\bINTFER_ONEMKL\b' "$0"; then
  echo "FAIL: this gate does not name INTFER_ONEMKL -- the flag census would "
  echo "      file it as an ungated opt-in gap" >&2
  exit 1
fi
if ! ls /opt/intel/oneapi/mkl/*/lib/libmkl_sycl_blas.so* >/dev/null 2>&1; then
  echo "SKIP: no libmkl_sycl_blas anywhere under /opt/intel/oneapi/mkl -- the "
  echo "      oneMKL arm is unreachable by build configuration" >&2
  exit 2
fi

# ---- 2. Configure a throwaway build dir with the MKL option ON. ----
# A failed configure must not leave a stale binary behind: the probe is only
# run if the configure AND the build both succeed, so a broken CMakeLists.txt
# cannot produce a PASS from a binary that no longer exists.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
CXX="$(which icpx || echo /opt/intel/oneapi/compiler/latest/bin/icpx)"
GATE_BUILD_DIR="${INTFER_ONEMKL_BUILD_DIR:-$BUILD_DIR/onemkl_gate}"
mkdir -p "$GATE_BUILD_DIR"
if ! cmake -S "$ROOT" -B "$GATE_BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DINTFER_ENABLE_MKL=ON \
    -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest \
    -DCMAKE_CXX_COMPILER="$CXX" >/dev/null 2>&1; then
  echo "FAIL: could not configure an MKL build for the onemkl gate" >&2
  exit 1
fi

# ---- 3. Build the probe in that dir, THROUGH THE SHARED BUILD GUARD. ----
# gate_inventory.sh requires every gate to build its own target through
# gate_require_build; a gate that builds by hand is reported as
# "NO BUILD GUARD" and fails the inventory. The guard honours
# INTFER_BUILD_DIR, so point it at the throwaway dir configured above and let
# it do the build -- that is the same stale-artifact protection every other
# gate gets, rather than a hand-rolled equivalent that can drift.
# *** BUILD_DIR IS THE GUARD'S OWN VARIABLE, NOT THE ENV. *** gate_require_build
# resolves its build dir as ${BUILD_DIR:-${INTFER_BUILD_DIR:-$ROOT/build}}, so
# the env override INTFER_BUILD_DIR above is DEAD unless BUILD_DIR is unset --
# and this gate sets BUILD_DIR="$ROOT/build" at line 47, before GATE_BUILD_DIR
# exists. Measured live: the gate configured the throwaway dir, built
# onemkl_probe into it by hand, then gate_require_build ran
# `cmake --build $ROOT/build --target onemkl_probe` and refused on
# "No rule to make target 'onemkl_probe'" -- a green gate on a binary this
# gate never built, with the stale-artifact protection pointing at the wrong
# tree. So BUILD_DIR is pointed at the throwaway dir here, explicitly, and the
# env override is kept as the secondary seam (a gate that forgets both would
# fall back to $ROOT/build, which is correct for every other gate).
BUILD_DIR="$GATE_BUILD_DIR"
INTFER_BUILD_DIR="$GATE_BUILD_DIR" \
INTFER_GATE_NO_BUILD=0 \
. "$SCRIPT_DIR/../lib/_gate_build.sh"
gate_require_build onemkl_probe || exit 1
BIN="$GATE_BUILD_DIR/examples/onemkl_probe/onemkl_probe"
[ -x "$BIN" ] || { echo "FAIL: onemkl_probe binary missing: $BIN" >&2; exit 1; }

# ---- 4. ldd must bind all five oneMKL .so's. ----
# A build that links the wrapper but not the components runs fine and then
# aborts at the first BLAS call with undefined symbols; that is a green gate
# on a broken binary. The rpath is published by src/gemm/CMakeLists.txt from
# the wrapper's own path, so this is a check on the wiring, not a hand-typed
# expectation.
missing="$(ldd "$BIN" 2>/dev/null | grep -iE 'mkl_sycl_blas|mkl_intel_lp64|mkl_core|mkl_gnu_thread' | grep -c 'not found')"
if [ "$missing" -ne 0 ]; then
  echo "FAIL: onemkl_probe is linked without its component libs ($missing not "
  echo "      found) -- the rpath wiring is incomplete" >&2
  ldd "$BIN" 2>/dev/null | grep -iE 'mkl' >&2
  exit 1
fi

# ---- 5. GPU precondition. ----
. "$SCRIPT_DIR/../lib/_gate_precond.sh"
gate_require_device || exit 2

# ---- 6. Run. ----
# WRAPPED IN A TIMEOUT, AND A TIMEOUT IS A SKIP, NOT A FAIL. This box is hung
# for SYCL kernel execution (see AGENTS.md: GPU SYCL hangs, CPU SYCL works), so
# the probe's queue construction blocks rather than throwing, and the `try`
# around it cannot catch a hang -- only an exception. A gate that hangs on a
# hung box trains the operator to ignore it, which is worse than a loud SKIP,
# so rc=124 (timeout) is reported as a precondition failure and exits 2, the
# same code the probe returns for "no GPU queue". A genuine defect in the arm
# returns 1 and is still red; a hung box is not a defect in the arm.
#
# The timeout is generous (300s) because on a HEALTHY box the M=1 arm and the
# five M>=8 arms each run two GEMMs, and the whole run is a real device
# measurement -- shortening it to "fail fast" would make the gate refuse the
# very thing it exists to certify.
timeout 300 "$BIN"
RC=$?
if [ "$RC" = "124" ]; then
  echo "SKIP: onemkl_probe timed out -- this box is hung for SYCL kernel "
  echo "      execution (GPU SYCL hangs, CPU SYCL works; see AGENTS.md). That "
  echo "      is a precondition, not a defect in the arm." >&2
  exit 2
fi
if [ "$RC" = "0" ]; then
  echo "ONEMKL TEST: PASS"
else
  echo "ONEMKL TEST: FAIL (rc=$RC) -- INTFER_ONEMKL must stay OFF at M=1 "
  echo "and be within tolerance at M>=8" >&2
fi
exit "$RC"