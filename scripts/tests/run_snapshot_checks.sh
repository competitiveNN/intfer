#!/usr/bin/env bash
# CI entry point for the snapshot/restore helper's own invariants and the
# no-direct-git-checkout lint rule. Exit 0 only if all pass.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash scripts/tests/test_gate_snapshot_edge_cases.sh
bash scripts/tests/bench_gate_snapshot.sh
bash scripts/checks/check_no_direct_git_checkout.sh

echo "SNAPSHOT CHECKS: PASS"
