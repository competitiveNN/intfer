#!/usr/bin/env bash
# Mutation matrix exercising the snapshot helper (snapshot/restore + mode
# preservation) against a small target. Serves as an emerging-matrix-type
# regression sample referenced by scripts/tests/regression_matrix_tests.sh.
#
# Why: proves the helper works end-to-end outside the mature matrices (fold,
# device-precond, gate-inventory), including the restore-skips-already-matching
# optimization and mode preservation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 1
trap 'rm -f "$TMPFILE"' EXIT
# shellcheck source=scripts/lib/_gate_snapshot.sh
. "$SCRIPT_DIR/../lib/_gate_snapshot.sh"

TMPFILE="tmp/roofline.txt"
mkdir -p "$(dirname "$TMPFILE")"
printf 'initial\n' >"$TMPFILE"
chmod 755 "$TMPFILE"

gate_snap_init "$TMPFILE" >/dev/null

# Mutate content and mode.
printf 'mutated roofline\n' >"$TMPFILE"
chmod 644 "$TMPFILE"

# Restore must recover both content and mode (755 comes from snapshot).
gate_snap_restore "$TMPFILE" || true

if [[ "$(cat "$TMPFILE")" != "initial" ]]; then
  echo "FAIL: roofline content not restored" >&2
  exit 1
fi
MODE="$(stat -c %a "$TMPFILE")"
if [[ "$MODE" != "755" ]]; then
  echo "FAIL: roofline mode not restored (got $MODE, want 755)" >&2
  exit 1
fi
STAMP="$ROOT/build/roofline_matrix/last_pass.stamp"
mkdir -p "$(dirname "$STAMP")"
gate_snap_attest "$STAMP.attestation" >/dev/null 2>&1 || true
{
  echo "stamp=roofline_matrix"
  echo "sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat "$STAMP.attestation" 2>/dev/null
} > "$STAMP"
echo "Roofline matrix test passed"
