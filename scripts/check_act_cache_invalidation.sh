#!/usr/bin/env bash
# INTFER_FUSE_ACT re-derives the writer set of ws_[d].act and proves every free of
# it clears the cache.
#
# WHY A CHECKER AND NOT A COMMENT. The cache skips a host convert plus one host
# upload per device when the same activation vector arrives twice. That is
# bit-exact ONLY while the bytes it remembers are the bytes the GEMM would have
# read. That argument lives in engine.hpp as prose -- and prose is exactly what
# this repo keeps finding falsified, including twice in one session by an argument
# that had been reviewed line for line.
#
# So it is re-derived from source on every stack run:
#   1. every free of the CACHED buffer must be followed by a cache clear;
#   2. the cache must be recorded only AFTER the upload loop, never before;
#   3. the flag must carry a kill switch, or it cannot be bisected.
#
# Attribution is FUNCTION-SCOPED and lives in scripts/_act_cache_scan.py, because
# the local alias `w` names DIFFERENT buffers in different functions -- ws_ in
# ensure_workspace, ws48_ in ensure_workspace48. A line-local grep cannot tell
# them apart, and getting it wrong failed here in BOTH directions: counting only
# `ws_[d].act` MISSED the ensure_workspace invalidation, and counting `w.act` too
# failed a healthy tree over a buffer the cache never touches.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

SRC="src/engine/engine.cpp"
SCAN="scripts/_act_cache_scan.py"
echo "== ACT-CACHE INVALIDATION COVERAGE (INTFER_FUSE_ACT) =="
[ -f "$SRC" ] || { echo "FAIL: $SRC missing" >&2; exit 2; }

bad=0

if grep -q 'INTFER_FUSE_ACT' "$SRC" && grep -q 'INTFER_NO_FUSE_ACT' "$SRC"; then
  echo "  ok    INTFER_FUSE_ACT present with a kill switch (INTFER_NO_FUSE_ACT)"
else
  echo "  FAIL  INTFER_FUSE_ACT has no kill switch -- it must be bisectable" >&2
  bad=$((bad + 1))
fi

out="$(python3 "$SCAN" "$SRC" 2>&1)" || {
  echo "  FAIL  the scanner failed to run:" >&2; printf '%s\n' "$out" | sed 's/^/        /' >&2; exit 1; }
printf '%s\n' "$out" | grep -q '^CACHED-FREE ' || {
  echo "  FAIL  the scanner found no free of the cached buffer at all -- the" >&2
  echo "        harvest is broken, so every verdict below would be vacuous." >&2; exit 1; }
n_free="$(printf '%s\n' "$out" | grep -c '^CACHED-FREE ' || true)"
n_clear="$(printf '%s\n' "$out" | grep -c '^CACHED-FREE .* 1$' || true)"
n_other="$(printf '%s\n' "$out" | grep -c '^FREE ' || true)"
echo "-- frees of the CACHED buffer: $n_free (clearing: $n_clear); unrelated frees: $n_other"
if [ "$n_free" != "$n_clear" ]; then
  echo "  FAIL  a free of ws_[d].act does NOT clear the cache -- a cache that" >&2
  echo "        outlives its buffer is a use-after-free, and no stream-identity" >&2
  echo "        test finds one until it is intermittent." >&2
  printf '%s\n' "$out" | grep '^CACHED-FREE' | sed 's/^/        /' >&2
  bad=$((bad + 1))
else
  echo "  ok    every free of the cached buffer clears the cache"
fi

set_line="$(grep -n 'act_cache_ptr_ = x.data();' "$SRC" | head -1 | cut -d: -f1)"
up_line="$(grep -n 'std::memcpy(act_ptr, act_staging_.data()' "$SRC" | head -1 | cut -d: -f1)"
if [ -z "$set_line" ] || [ -z "$up_line" ]; then
  echo "  FAIL  could not locate the cache record or the upload" >&2
  bad=$((bad + 1))
elif [ "$up_line" -lt "$set_line" ]; then
  echo "  ok    the cache is recorded AFTER the upload (upload $up_line, record $set_line)"
else
  echo "  FAIL  the cache is recorded BEFORE the upload (upload $up_line, record" >&2
  echo "        $set_line) -- a later call could skip an upload that never happened" >&2
  bad=$((bad + 1))
fi

for other in 'memcpy(wv_[0].act' 'memcpy(wv_[1].act'; do
  if grep -qF "$other" "$SRC"; then
    echo "  ok    $other present and distinct from ws_[d].act"
  else
    echo "  note  $other not found in $SRC -- re-read the invalidation argument" >&2
  fi
done

echo
if [ "$bad" = 0 ]; then
  echo "ACT-CACHE INVALIDATION: PASS ($n_free cached-buffer free(s), all cleared)"
  echo "RESULT: the activation cache cannot serve bytes that were freed, or never written"
  exit 0
fi
echo "ACT-CACHE INVALIDATION: FAIL ($bad problem(s))" >&2
echo "RESULT: FAIL -- the cache can serve bytes that were freed or never written" >&2
exit 1
