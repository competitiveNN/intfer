#!/usr/bin/env bash
# GGUF interop gate: .intfer -> .gguf conversion of the real 0.8B model.
#
# Exercises serialize_gguf/deserialize_gguf on real quantized weights (the
# quant_pipeline self-test only covers synthetic data): converts the 0.8B
# .intfer, reads the .gguf back, and requires bit-identical dequantization
# (same payload, different container).
#
#
# BUILD-FIRST (audit #7): gate_require_build in scripts/lib/_gate_build.sh, for the
# intfer_gguf target (not intfer_cli) -- see test_gpu_optin.sh for the class of
# bug. Opt out with INTFER_GATE_NO_BUILD=1 only when the build is known current.
#
# Usage: scripts/gates/test_gguf_roundtrip.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intfer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_gguf/intfer_gguf"

. "$SCRIPT_DIR/../lib/_gate_build.sh"
gate_require_build intfer_gguf || exit 1

[ -x "$BIN" ] || { echo "FAIL: intfer_gguf binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}"
GGUF="$(mktemp --tmpdir="$TMPDIR" model.XXXX.gguf)"
trap 'rm -f "$GGUF"' EXIT
trap 'rm -f "$GGUF"; exit 130' INT
trap 'rm -f "$GGUF"; exit 143' TERM

OUT="$("$BIN" "$MODEL" "$GGUF" 2>&1)"; RC=$?
echo "$OUT"
[ "$RC" = "0" ] || { echo "GATE FAIL: intfer_gguf exited rc=$RC" >&2; exit 1; }
grep -aq "(PASS)" <<<"$OUT" || { echo "GATE FAIL: no PASS line" >&2; exit 1; }
echo "GGUF-ROUNDTRIP TEST: PASS"
