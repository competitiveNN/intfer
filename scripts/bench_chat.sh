#!/usr/bin/env bash
# Chat-realistic benchmark: browser-like multi-turn requests against --serve-http.
#
# Reports per request: TTFT (first SSE chunk), TPOT (steady-state s/token
# excluding TTFT), E2E wall, token count — plus box loadavg and GPU clocks
# alongside, so runs are comparable. CLI tok/s numbers are decode-only on
# quiet boxes; this measures what a chat user actually sees: full history
# re-prefilled every turn + contention.
#
# Usage: scripts/bench_chat.sh [--port P] [--turns N] [--max-tokens N]
#   Starts a server on P (default 18779) unless INTFER_HTTP_URL is set.
# Env: INTFER_08B (model), INTFER_BUILD_DIR.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PORT=18779; TURNS=3; MT=30
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --max-tokens) MT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
mkdir -p /var/home/fra/build_tmp

URL="${INTFER_HTTP_URL:-http://127.0.0.1:$PORT}"
STARTED=0
if ! curl -s -m 3 "$URL/health" >/dev/null 2>&1; then
  LOG="/var/home/fra/build_tmp/bench_chat.log"
  setsid nohup "$BIN" -m "$MODEL" --gpu --tp 1 --serve-http \
    --port "$PORT" --host 127.0.0.1 >"$LOG" 2>&1 < /dev/null &
  SRV_PID=$!
  STARTED=1
  for _ in $(seq 1 60); do
    sleep 5
    curl -s -m 3 "$URL/health" >/dev/null 2>&1 && break
  done
  curl -s -m 3 "$URL/health" >/dev/null 2>&1 || { echo "FAIL: server never came up (see $LOG)" >&2; exit 1; }
fi
cleanup() { [ "$STARTED" = 1 ] && kill "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

echo "box loadavg: $(cat /proc/loadavg)"
xpu-smi discovery 2>/dev/null | grep -E "Device Name" | head -2 || true
export INTFER_BENCH_URL="$URL" INTFER_BENCH_TURNS="$TURNS" INTFER_BENCH_MT="$MT"
# Optional parity file: when INTFER_BENCH_PARITY_OUT is set, per-turn full
# texts are appended (one line per turn) for cache-ON vs cache-OFF diffing.
# When INTFER_BENCH_PARITY_EXPECT is set, each turn's text must equal the
# corresponding line or the script fails (multi-turn parity assertion).
python3 - <<'EOF'
import json, os, urllib.request, time, sys
URL=os.environ["INTFER_BENCH_URL"]; TURNS=int(os.environ["INTFER_BENCH_TURNS"]); MT=int(os.environ["INTFER_BENCH_MT"])
par_out=os.environ.get("INTFER_BENCH_PARITY_OUT","")
par_exp=os.environ.get("INTFER_BENCH_PARITY_EXPECT","")
exp_lines=open(par_exp).read().splitlines() if par_exp else []
outh=open(par_out,"a") if par_out else None
history=[{"role":"system","content":"You are a helpful assistant."}]
prompts=["What is the capital of France?","And what is its population?","What river runs through it?","Tell me more."][:TURNS]
fail=0
for turn,p in enumerate(prompts):
    history.append({"role":"user","content":p})
    body=json.dumps({"model":"x","messages":history,"max_tokens":MT,"temperature":0,"stream":True}).encode()
    req=urllib.request.Request(URL+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); first=None; n=0; full=""
    r=urllib.request.urlopen(req,timeout=600)
    while True:
        line=r.readline().decode(errors="replace")
        if not line: break
        if line.startswith("data: ") and "[DONE]" not in line:
            c=json.loads(line[6:])["choices"][0].get("delta",{}).get("content","")
            if c:
                if first is None: first=time.time()-t0
                n+=1; full+=c
    wall=time.time()-t0
    ttft=first or wall
    tpot=(wall-ttft)/max(n-1,1) if n else 0
    print(f"turn{turn}: TTFT={ttft:.1f}s TPOT={tpot:.2f}s/tok E2E={wall:.1f}s toks={n} hist_msgs={len(history)} text={full[:60]!r}")
    if outh: outh.write(full.replace("\n","\\n")+"\n")
    if exp_lines:
        exp=exp_lines[turn].replace("\\n","\n") if turn < len(exp_lines) else None
        if full != exp:
            print(f"PARITY FAIL turn{turn}: got {full[:80]!r} want {exp[:80] if exp else None!r}")
            fail=1
        else:
            print(f"turn{turn}: parity OK")
    history.append({"role":"assistant","content":full})
if outh: outh.close()
sys.exit(fail)
EOF
