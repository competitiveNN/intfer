#!/usr/bin/env bash
# Regression tests covering mature and emerging matrix types.
# Uses the snapshot helper and verifies that each matrix passes its own checks.
# Exit 0 if all tests pass, non-zero otherwise.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

run_matrix() {
  local script="$1" desc="$2"
  echo "=== Running $desc ==="
  if bash "$script"; then
    echo "✓ $desc passed"
    return 0
  fi
  echo "✗ $desc failed"
  return 1
}

status=0

# Mature mutation matrices. The build-tree matrices tolerate the dirty dev tree
# through INTFER_MATRIX_ALLOW_DIRTY; device_precond already snapshots/restores its
# inputs byte-for-byte.
INTFER_MATRIX_ALLOW_DIRTY=1 run_matrix scripts/matrices/mutate_fold_matrix.sh "Fold matrix (fresh)" || status=1
INTFER_MATRIX_ALLOW_DIRTY=1 run_matrix scripts/matrices/mutate_gate_inventory.sh "Device inventory matrix (fresh)" || status=1
INTFER_MATRIX_ALLOW_DIRTY=1 run_matrix scripts/matrices/mutate_device_precond.sh "Device precond matrix (fresh)" || status=1

# Stack-cert runs a sandbox cert and guards against leaks into the outer tree.
# On a clean checkout it proves the leak check too; on a dirty dev tree only that
# option is skipped so the four states are still graded.
INTFER_MATRIX_ALLOW_DIRTY=1 run_matrix scripts/matrices/mutate_stack_cert.sh "Stack cert matrix (fresh)" || status=1

# Emerging-matrix-type snapshot helper sample.
run_matrix scripts/matrices/mutate_roofline.sh "Roofline matrix (emerging type)" || status=1

# Attest the regression run itself, even though it has no single mutation target.
. "$ROOT/scripts/lib/_gate_snapshot.sh"
STAMP="$ROOT/build/regression_matrix/last_pass.stamp"
mkdir -p "$(dirname "$STAMP")"
gate_snap_attest "$STAMP.attestation" >/dev/null 2>&1 || true
{
  echo "stamp=regression_matrix"
  echo "result=$([ "$status" -eq 0 ] && echo PASS || echo FAIL)"
  cat "$STAMP.attestation" 2>/dev/null
} > "$STAMP"

echo "=== Regression matrix tests: $([ "$status" -eq 0 ] && echo PASS || echo FAIL) ==="
exit "$status"
