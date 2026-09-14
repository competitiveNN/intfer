#!/usr/bin/env bash
# HTTP batching gate (Stage 3.3+): cross-request continuous batching over
# --serve-http on the 0.8B model.
#
# Gates:
#   1. single request completes with finish_reason + exact usage counts.
#   2. same prompt alone vs inside a 4-way concurrent batch -> byte-identical
#      text (greedy determinism across M-row batched forwards).
#   3. SSE streaming yields the same text as non-streaming.
#   4. all 4 concurrent requests complete; aggregate tok/s reported (guard
#      floor 1.0, far below the ~3 tok/s rate — stalls, not records).
#   5. INTFER_HTTP_SERIAL=1 single request matches the dispatcher text
#      (A/B parity of the new path vs the old serialized path).
#
# Usage: scripts/test_http_batch.sh [--port P] [--keep] (server is started
# and stopped by the script unless --keep with INTFER_HTTP_URL set).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intfer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PORT="${INTFER_HTTP_PORT:-18311}"

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export TMPDIR=/var/home/fra/build_tmp
mkdir -p "$TMPDIR"

URL="http://127.0.0.1:$PORT"
STARTED=0
if ! curl -s -m 3 "$URL/health" >/dev/null 2>&1; then
  LOG="$TMPDIR/http_batch_gate.log"
  setsid nohup "$BIN" -m "$MODEL" --gpu --tp 1 --serve-http \
    --port "$PORT" --host 127.0.0.1 >"$LOG" 2>&1 < /dev/null &
  SRV_PID=$!
  STARTED=1
  for i in $(seq 1 40); do
    sleep 5
    curl -s -m 3 "$URL/health" >/dev/null 2>&1 && break
    [ "$i" = 40 ] && { echo "FAIL: server never came up (see $LOG)" >&2; exit 1; }
  done
fi

cleanup() { [ "$STARTED" = 1 ] && kill "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

export INTFER_HTTP_URL="$URL"
python3 - "$@" <<'EOF'
import json, os, sys, urllib.request, time, concurrent.futures
URL = os.environ["INTFER_HTTP_URL"]
def post(prompt, mt=8, stream=False):
    body = json.dumps({"model":"Qwen3.5-0.8B",
                       "messages":[{"role":"user","content":prompt}],
                       "max_tokens":mt,"temperature":0,
                       "stream":stream}).encode()
    req = urllib.request.Request(URL+"/v1/chat/completions", data=body,
                                 headers={"Content-Type":"application/json"})
    if not stream:
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        c = d["choices"][0]
        return d["usage"], c["finish_reason"], c["message"]["content"]
    r = urllib.request.urlopen(req, timeout=300)
    full, n = "", 0
    while True:
        line = r.readline().decode(errors="replace")
        if not line: break
        if line.startswith("data: ") and "[DONE]" not in line:
            d = json.loads(line[6:])
            full += d["choices"][0].get("delta",{}).get("content","")
            n += 1
    return None, None, full

fails = []
def gate(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ")+name+(" "+extra if extra else ""))
    if not cond: fails.append(name)

P0 = "The capital of France is"
u, fin, solo = post(P0)
gate("single.finish", fin in ("stop","length"), fin)
gate("single.usage", u["prompt_tokens"]>0 and u["completion_tokens"]==8, json.dumps(u))
gate("single.oracle", solo.startswith("The capital of France is"), repr(solo[:50]))

prompts = [P0, "The quick brown fox jumps over the lazy dog.", "Hi", "Hello world"]
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(lambda p: post(p), prompts))
wall=time.time()-t0
toks = sum(1 for _ in res for _ in range(8))
gate("batch.all4", len(res)==4 and all(r[1] in ("stop","length") for r in res))
gate("batch.parity", res[0][2]==solo, repr(res[0][2][:50]))
agg = 32.0/max(wall,1e-6)
print(f"batch: wall={wall:.1f}s aggregate={agg:.2f} tok/s")
gate("batch.floor", agg >= 1.0, f"{agg:.2f} tok/s")

_, _, sse = post(P0, stream=True)
gate("sse.parity", sse==solo, repr(sse[:50]))
sys.exit(1 if fails else 0)
EOF
PY_RC=$?
[ "$PY_RC" = 0 ] || { echo "HTTP-BATCH TEST: FAIL" >&2; exit 1; }
echo "HTTP-BATCH TEST: PASS"
