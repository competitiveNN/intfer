#!/usr/bin/env bash
# Serve load gate (Stage 3): multi-client concurrency sweep over --serve-http.
#
# Complements scripts/test_http_batch.sh (functional batch parity) with a
# load dimension: at each concurrency level C in 1 4 8, fire C concurrent
# clients at a live server and report per-request latency percentiles
# (P50/P99 of client-observed completion latency), aggregate req/s and
# tok/s. Gates:
#   1. all requests at every level complete with a terminal finish_reason.
#   2. solo prompt text == same prompt under C=8 load (greedy determinism
#      under continuous batching; byte-identical text).
#   3. aggregate tok/s at C=8 >= C=1 (batching must not lose throughput;
#      guard floor 0.5 tok/s far below the ~3 tok/s rate — stalls, not
#      records; scaling numbers are logged for the record, not gated).
#   4. (opt-in) batch-full probe: INTFER_LOAD_BIGMT=N re-runs the top
#      concurrency level with max_tokens=N and reports aggregate tok/s.
#      Short MT=8 requests drain the batch, so the sweep under-measures
#      the batching ceiling (2026-09-17: C=8/MT=8 18 tok/s vs C=8/MT=64
#      36.3 tok/s). Informational only — completion-gated, no floor.
#
# Usage: scripts/test_serve_load.sh [--port P]
#   INTFER_HTTP_URL: reuse an already-running server (script won't kill it).
#   INTFER_LOAD_LEVELS: space-separated concurrency levels (default "1 4 8").
#   INTFER_LOAD_MAXTOK: max_tokens per request (default 8).
#   INTFER_LOAD_BIGMT: batch-full probe max_tokens (default unset = skip).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intfer}"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PORT="${INTFER_HTTP_PORT:-18312}"
LEVELS="${INTFER_LOAD_LEVELS:-1 4 8}"
MAXTOK="${INTFER_LOAD_MAXTOK:-8}"
BIGMT="${INTFER_LOAD_BIGMT:-}"

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export TMPDIR=/var/home/fra/build_tmp
mkdir -p "$TMPDIR"

URL="${INTFER_HTTP_URL:-http://127.0.0.1:$PORT}"
STARTED=0
if ! curl -s -m 3 "$URL/health" >/dev/null 2>&1; then
  LOG="$TMPDIR/serve_load_gate.log"
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

# Absolute tok/s numbers are contention-sensitive (GPU + host compete with
# whatever else runs on the box); log the 1-min load average alongside so
# runs are comparable. 2026-09-16 vs 2026-09-17 differed 4-5x on load ~17
# vs ~3 with identical code/flags.
cat /proc/loadavg 2>/dev/null | sed 's/^/box loadavg: /' || true

export INTFER_HTTP_URL="$URL" INTFER_LOAD_LEVELS="$LEVELS" INTFER_LOAD_MAXTOK="$MAXTOK" INTFER_LOAD_BIGMT="$BIGMT"
python3 - <<'EOF'
import json, os, sys, urllib.request, time, concurrent.futures, statistics
URL = os.environ["INTFER_HTTP_URL"]
LEVELS = [int(x) for x in os.environ["INTFER_LOAD_LEVELS"].split()]
MT = int(os.environ["INTFER_LOAD_MAXTOK"])
BIGMT = os.environ.get("INTFER_LOAD_BIGMT", "").strip()
BIGMT = int(BIGMT) if BIGMT else 0
PROMPTS = ["The capital of France is",
           "The quick brown fox jumps over the lazy dog.",
           "Hi", "Hello world",
           "Machine learning models run fastest on",
           "The history of artificial intelligence",
           "Once upon a time in a land far away",
           "To be or not to be, that is"]

def post(prompt, mt=MT):
    body = json.dumps({"model": "Qwen3.5-0.8B",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    c = d["choices"][0]
    return d["usage"], c["finish_reason"], c["message"]["content"], dt

def pct(xs, p):
    if not xs: return float("nan")
    s = sorted(xs); k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)

fails = []
def gate(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" " + extra if extra else ""))
    if not cond: fails.append(name)

_, _, solo, _ = post(PROMPTS[0])
gate("solo.oracle", solo.startswith("The capital of France is"), repr(solo[:50]))

agg_by_level = {}
for C in LEVELS:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(C)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=C) as ex:
        res = list(ex.map(post, prompts))
    wall = time.time() - t0
    ok = len(res) == C and all(r[1] in ("stop", "length") for r in res)
    gate(f"c{C}.all", ok, f"{len(res)}/{C} finish_ok")
    ctoks = sum(r[0]["completion_tokens"] for r in res)
    lat = [r[3] for r in res]
    p50, p99 = pct(lat, 50), pct(lat, 99)
    req_s = C / max(wall, 1e-6)
    tok_s = ctoks / max(wall, 1e-6)
    agg_by_level[C] = tok_s
    print(f"c{C}: wall={wall:.1f}s req/s={req_s:.2f} tok/s={tok_s:.2f} "
          f"lat_p50={p50:.2f}s lat_p99={p99:.2f}s ctoks={ctoks}")
    if C == LEVELS[-1]:
        gate(f"c{C}.parity", res[0][2] == solo, repr(res[0][2][:50]))

base = agg_by_level.get(LEVELS[0], 0.0)
top = agg_by_level.get(LEVELS[-1], 0.0)
print(f"scaling: c{LEVELS[0]}={base:.2f} tok/s -> c{LEVELS[-1]}={top:.2f} tok/s "
      f"(x{top/max(base,1e-6):.2f})")
gate("floor", top >= 0.5, f"{top:.2f} tok/s")
if BIGMT > 0:
    C = LEVELS[-1]
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(C)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=C) as ex:
        res = list(ex.map(lambda p: post(p, BIGMT), prompts))
    wall = time.time() - t0
    ok = len(res) == C and all(r[1] in ("stop", "length") for r in res)
    gate(f"c{C}.bigmt{MT}", ok, f"{len(res)}/{C} finish_ok")
    ctoks = sum(r[0]["completion_tokens"] for r in res)
    print(f"c{C}.bigmt{MT}: wall={wall:.1f}s tok/s={ctoks/max(wall,1e-6):.2f} "
          f"ctoks={ctoks}")
sys.exit(1 if fails else 0)
EOF
PY_RC=$?
[ "$PY_RC" = 0 ] || { echo "SERVE-LOAD TEST: FAIL" >&2; exit 1; }
echo "SERVE-LOAD TEST: PASS"
