#!/usr/bin/env bash
# Run a fast matrix twice and assert identical verdict outputs (idempotence).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Use dri_probe (fast) as the idempotence probe
LOGA="$(mktemp)"
LOGB="$(mktemp)"
if [ "${1:-}" = "allow-dirty" ]; then
  INTFER_MATRIX_ALLOW_DIRTY=1 bash scripts/matrices/mutate_dri_probe.sh >"$LOGA" 2>&1 || true
  INTFER_MATRIX_ALLOW_DIRTY=1 bash scripts/matrices/mutate_dri_probe.sh >"$LOGB" 2>&1 || true
else
  bash scripts/matrices/mutate_dri_probe.sh >"$LOGA" 2>&1 || true
  bash scripts/matrices/mutate_dri_probe.sh >"$LOGB" 2>&1 || true
fi
# Compare verdict lines only
grep -E "DRI PROBE MUTATE|caught|MISSED|PASS|FAIL" "$LOGA" | sort >"${LOGA}.sum"
grep -E "DRI PROBE MUTATE|caught|MISSED|PASS|FAIL" "$LOGB" | sort >"${LOGB}.sum"
if diff -q "${LOGA}.sum" "${LOGB}.sum" >/dev/null 2>&1; then
  echo "MATRIX IDEMPOTENCE: PASS"
  rm -f "$LOGA" "$LOGB" "${LOGA}.sum" "${LOGB}.sum"
  exit 0
fi
echo "MATRIX IDEMPOTENCE: FAIL"
diff "${LOGA}.sum" "${LOGB}.sum" | head -20
rm -f "$LOGA" "$LOGB" "${LOGA}.sum" "${LOGB}.sum"
exit 1
