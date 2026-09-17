#!/usr/bin/env bash
# Serve the quantized 27B model over OpenAI-compatible HTTP.
#
# Loads Qwen3.8-27B INT2 on the 2x B580 fleet (--gpu --tp 2; TP=1 OOMs:
# the lm_head INT8 tile alloc needs ~19GB vs ~12GB per B580).
# Model resolution mirrors scripts/test_tp2_27b.sh: INTFER_27B_SC wins,
# then the -sc artifact, then the merged -int2 chunks artifact.
#
# Usage: scripts/serve_27b.sh [--port P] [--host ADDR] [--bg]
#   --bg: detach into background (setsid+nohup), print PID + log path,
#         and wait for /health instead of exec'ing the server.
#   Without --bg the server runs in the foreground (Ctrl-C stops it).
# Env overrides: INTFER_27B_SC, INTFER_HTTP_PORT (default 18313),
#   INTFER_HTTP_HOST (default 127.0.0.1), INTFER_BUILD_DIR.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${INTFER_BUILD_DIR:-$ROOT/build}/examples/intfer_cli/intfer_cli"
PORT="${INTFER_HTTP_PORT:-18313}"
HOST="${INTFER_HTTP_HOST:-127.0.0.1}"
BG=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --bg) BG=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

MODEL="${INTFER_27B_SC:-/var/home/fra/data/models/intfer/Qwen3.8-27B-sc.intifer}"
if [ ! -f "$MODEL" ]; then
  MODEL="/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer"
fi

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: 27B model missing: $MODEL" >&2; exit 1; }

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export TMPDIR=/var/home/fra/build_tmp
mkdir -p "$TMPDIR"

echo "model: $MODEL"
echo "listen: http://$HOST:$PORT"

if [ "$BG" = 1 ]; then
  LOG="$TMPDIR/serve_27b_${PORT}.log"
  if curl -s -m 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "server already up at http://$HOST:$PORT"
    exit 0
  fi
  setsid nohup "$BIN" -m "$MODEL" --gpu --tp 2 --serve-http \
    --port "$PORT" --host "$HOST" >"$LOG" 2>&1 < /dev/null &
  echo "pid: $! log: $LOG"
  for i in $(seq 1 60); do
    sleep 5
    curl -s -m 3 "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
    [ "$i" = 60 ] && { echo "FAIL: server never came up (see $LOG)" >&2; exit 1; }
  done
  echo "up: http://$HOST:$PORT/health"
else
  exec "$BIN" -m "$MODEL" --gpu --tp 2 --serve-http \
    --port "$PORT" --host "$HOST"
fi
