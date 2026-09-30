#!/usr/bin/env bash
# TP=2 inference test: 27B self-contained quant model on the 2x B580 fleet.
#
# Runs intfer_cli on Qwen3.8-27B-sc.intifer with --gpu --tp 2 and checks the
# greedy decode against the HF oracle for "The capital of France is":
#   prefill -> 11751, decode -> 13 198 760
# Also reports prefill tok/s and decode tok/s vs the 4000 / 40t/s targets.
#
#
# BUILD-FIRST (audit #7): gate_require_build in scripts/_gate_build.sh. Audit
# #6 fixed four gates against the stale-artifact class; this one consumed the
# intfer_cli binary without ever building it, so a source edit that failed to
# compile left the previous binary in place and the gate judged code that no
# longer existed. Opt out with INTFER_GATE_NO_BUILD=1 only when the build is
# known current.
#
# Usage: scripts/test_tp2_27b.sh [-n MAX_NEW] [-p PROMPT]
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${INTFER_27B_SC:-/var/home/fra/data/models/intfer/Qwen3.8-27B-sc.intifer}"
# Fall back to the merged chunk artifact when the -sc file is absent
# (same 27B INT2 payload; INTFER_27B_SC still overrides).
if [ ! -f "$MODEL" ]; then
  MODEL="/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer"
fi
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
N=4
PROMPT="The capital of France is"
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    -p) PROMPT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export TMPDIR=/var/home/fra/build_tmp

. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build intfer_cli || exit 1

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

LOG="$(mktemp --tmpdir="$TMPDIR" tp2_27b.XXXX.log)"
# Keep the per-run CLI logs next to the wrapper log (not mktemp orphans):
# <wrapper>.out / <wrapper>.err hold the CLI stdout/stderr on gate failure.
CLI_OUT="$LOG.out"; CLI_ERR="$LOG.err"
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "--- CLI stdout ($CLI_OUT) ---" >&2; tail -5 "$CLI_OUT" >&2; echo "--- CLI stderr ($CLI_ERR, last 25) ---" >&2; tail -25 "$CLI_ERR" >&2; else rm -f "$CLI_OUT" "$CLI_ERR"; fi' EXIT
"$BIN" -m "$MODEL" --gpu --tp 2 --ids \
  -p "$PROMPT" -n "$N" >"$CLI_OUT" 2>"$CLI_ERR"; RC=$?
cat "$CLI_OUT"
if [ $RC -ne 0 ]; then
  echo "FAIL: intfer_cli exited rc=$RC" >&2
  tail -20 "$CLI_ERR" >&2
  exit 1
fi

# --- Oracle gate (greedy): prefill 11751, then 13 198 760 ...
IDS="$(grep -a -oE "decode step [0-9]+: token=[0-9]+" "$CLI_ERR" | sed -E 's/.*token=//')"
PREFILL_OK=0
grep -aq "decode step 0: token=11751" "$CLI_ERR" && PREFILL_OK=1
EXPECT="13 198 760"
GOT="$(echo "$IDS" | tail -n +2 | tr '\n' ' ' | sed 's/ *$//')"
GOT_PREFIX="$(echo "$GOT" | cut -d' ' -f1-3)"
echo "---"
echo "prefill token: $(grep -a -oE 'decode step 0: token=[0-9]+' "$CLI_ERR" | sed -E 's/.*=//') (expect 11751)"
echo "decode ids:    $GOT (expect prefix: $EXPECT)"
grep -a -E "prefill:|decode:" "$CLI_ERR" | tail -2
grep -a -A11 "GPU profile" "$CLI_ERR" | tail -12

PASS=1
[ "$PREFILL_OK" = 1 ] || { echo "GATE FAIL: prefill != 11751" >&2; PASS=0; }
[ "$GOT_PREFIX" = "$EXPECT" ] || { echo "GATE FAIL: decode prefix '$GOT_PREFIX' != '$EXPECT'" >&2; PASS=0; }
TP_LINE="$(grep -a "GPU backend:" "$CLI_ERR" | head -1)"
echo "$TP_LINE"
echo "$TP_LINE" | grep -q "TP=sharded" || { echo "GATE FAIL: TP not sharded (need 2 devices)" >&2; PASS=0; }
rm -f "$LOG"  # CLI_OUT/CLI_ERR are handled by the EXIT trap (kept on FAIL)
if [ "$PASS" = 1 ]; then echo "TP2-27B TEST: PASS"; else echo "TP2-27B TEST: FAIL" >&2; exit 1; fi
