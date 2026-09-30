#!/usr/bin/env bash
# XMX-vs-EU attribution for intfer GPU runs: joins xmxmon's hardware
# counters with our internal [GPU profile] buckets in one command.
#
# Usage:
#   scripts/xmx_profile.sh [--tp 1] [--device 0] [--period-ms 100] \
#     [--xmxmon /path/to/xmxmon] -- <intfer_cli args...>
# Example (decode):
#   scripts/xmx_profile.sh -- -m test/models/Qwen3.5-0.8B.intifer \
#     -p "The capital of France is" -n 11 --no-stream \
#     --tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py
# Example (prefill-heavy, where XMX should light up if used):
#   INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 INTFER_BATCH_GEMM=1 \
#     scripts/xmx_profile.sh -- <same model args> -n 5
#
# How it works: xmxmon opens a Level Zero metric streamer on the device
# (device-wide, backend-agnostic) while intfer_cli runs in the foreground,
# then xmx-summary.py prints the XMX verdict + utilization. Our internal
# [GPU profile @decode/@prefill] lines print kernel/submit/waitblock buckets.
#
# Requirements (see https://github.com/meatposes/xmxmon):
#   g++ -O2 -o xmxmon xmxmon.cpp -lze_loader
#   sudo sysctl -w dev.xe.observation_paranoid=0   # Xe driver; persists via /etc/sysctl.d/
# If the streamer cannot open (paranoid sysctl, missing perms, another OA
# session), the script prints the remedy and still runs the workload so the
# internal buckets are captured — XMX stays UNKNOWN, never silently zero.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"

TP=1; DEVICE=0; PERIOD=100
XMXMON="${XMXMON:-}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tp) TP="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --period-ms) PERIOD="$2"; shift 2;;
    --xmxmon) XMXMON="$2"; shift 2;;
    --) shift; ARGS=("$@"); break;;
    *) echo "unknown flag: $1 (use -- to separate intfer args)" >&2; exit 1;;
  esac
done
[ -x "$BIN" ] || { echo "FAIL: cli missing: $BIN" >&2; exit 1; }
[ ${#ARGS[@]} -gt 0 ] || { echo "FAIL: no intfer args after --" >&2; exit 1; }
if [ -z "$XMXMON" ]; then
  for c in /tmp/xmxmon "$ROOT/../xmxmon/xmxmon" ./xmxmon; do
    [ -x "$c" ] && { XMXMON="$c"; break; }
  done
fi
[ -n "$XMXMON" ] && [ -x "$XMXMON" ] || {
  echo "SKIP-XMX: no xmxmon binary (build: g++ -O2 -o xmxmon xmxmon.cpp -lze_loader;"
  echo "  repo: https://github.com/meatposes/xmxmon). Set XMXMON=/path/to/xmxmon."
  echo "Running workload without XMX capture:"
  echo "== internal buckets only =="
  set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
  exec "$BIN" --gpu --tp "$TP" "${ARGS[@]}"
}

OUT="$(mktemp --tmpdir="${TMPDIR:-/tmp}" xmxcap.XXXX.csv)"
XMXLOG="$(mktemp --tmpdir="${TMPDIR:-/tmp}" xmxmon.XXXX.log)"
"$XMXMON" --device "$DEVICE" --group VectorEngineProfile \
  --period-ms "$PERIOD" --out "$OUT" >"$XMXLOG" 2>&1 &
XMPID=$!
sleep 1
if ! kill -0 "$XMPID" 2>/dev/null; then
  echo "SKIP-XMX: streamer failed to start:"
  tail -3 "$XMXLOG"
  echo "  remedy: sudo sysctl -w dev.xe.observation_paranoid=0"
  echo "  (or stop other OA/metric-streamer sessions on device $DEVICE)."
  echo "== internal buckets only =="
  rm -f "$XMXLOG"
  set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
  "$BIN" --gpu --tp "$TP" "${ARGS[@]}" 2>&1 | grep -aE "tokens in|GPU profile @|decode step|topology:"
  rm -f "$OUT"; exit 0
fi
echo "== xmxmon capturing on device $DEVICE (pid $XMPID) =="
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
"$BIN" --gpu --tp "$TP" "${ARGS[@]}" 2>&1 | grep -aE "tokens in|GPU profile @|decode step|topology:"
kill -INT "$XMPID" 2>/dev/null; wait "$XMPID" 2>/dev/null
echo "== XMX verdict =="
SUMMARY_PY="$(dirname "$XMXMON")/xmx-summary.py"
if [ -x "$SUMMARY_PY" ]; then
  "$SUMMARY_PY" "$OUT" 2>&1 | head -25
else
  python3 - "$OUT" <<'EOF' | head -25
import csv, sys
xmx = ["XVE_INST_EXECUTED_XMX_INT2","XVE_INST_EXECUTED_XMX_INT4",
       "XVE_INST_EXECUTED_XMX_INT8","XVE_INST_EXECUTED_XMX_FP16",
       "XVE_INST_EXECUTED_XMX_BF16"]
tot = {k: 0 for k in xmx}; n = 0
try:
    with open(sys.argv[1]) as f:
        for row in csv.DictReader(f):
            n += 1
            for k in xmx:
                try: tot[k] += float(row.get(k, 0) or 0)
                except ValueError: pass
except FileNotFoundError:
    print("no capture rows (streamer died?)"); sys.exit(0)
print(f"{n} reports")
for k in xmx:
    print(f"{k} total {tot[k]:.3g}" + ("  <-- ACTIVE" if tot[k] > 0 else ""))
print("VERDICT: " + ("XMX IS being used" if any(v > 0 for v in tot.values())
      else "XMX IDLE (all zero) — workload runs on EU/ALU pipes, not the matrix engines"))
EOF
fi
rm -f "$OUT" "$XMXLOG"
