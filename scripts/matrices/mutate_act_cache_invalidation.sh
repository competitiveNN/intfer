#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_act_cache_invalidation.sh.
#
# The checker exists to make INTFER_FUSE_ACT's safety argument RE-DERIVABLE, so
# its states must be able to break that argument in the ways it could plausibly
# be broken -- not merely to make the checker print something.
#
# AC-1 REMOVES an invalidation. This is the dangerous one: a cache that outlives a
# free is a use-after-free, and no amount of stream-identity testing finds it
# until it is intermittent.
# AC-2 MOVES the cache record BEFORE the upload loop, so a later call can skip an
# upload that never happened and read zero pages.
# AC-3 REMOVES the kill switch, making the experiment un-bisectable.
# AC-4 INVERTS the upload-vs-set comparison in the checker, so it demands the
# opposite of the property it claims -- the bug this checker itself shipped in
# its first version, kept here so it cannot come back.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/act_cache_matrix"
# *** WIPE LOCK, OUTSIDE THE TREE IT WIPES (check_matrix_sandbox_locks.sh). ***
# The sandbox below is a FIXED path wiped on every invocation, so two concurrent
# runs share one tree: the second's wipe empties the first's mid-state and the
# first then grades a corpus the other has just rebuilt (MEASURED 2026-10-21 on
# mutate_durable_reads.sh). Taken BEFORE the first wipe and released on
# EXIT/INT/TERM, so a run that dies mid-state does not brick the next.
LOCK="$BUILD/.mutate_act_cache_invalidation.lock"
mkdir -p "$BUILD" 2>/dev/null || true
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_act_cache_invalidation.sh is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

CHECK="scripts/checks/check_act_cache_invalidation.sh"
SRC="src/engine/engine.cpp"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"
PRISTINE_SRC="$SB/pristine.src"; PRISTINE_CHK="$SB/pristine.chk"
cp "$SRC" "$PRISTINE_SRC"; cp "$CHECK" "$PRISTINE_CHK"
restore() { cp "$PRISTINE_SRC" "$SRC"; cp "$PRISTINE_CHK" "$CHECK"; }

patch() { python3 - "$1" "$2" "$3" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
c = s.count(o)
if c < 1:
    print("anchor absent: %r" % (o[:50],), file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(o, n, c))
PYX
}

total=0; caught=0; RESULTS=()
run_state() { # $1 id, $2 expect, $3 want: reject | keep-green
  local id="$1" expect="$2" want="${3:-reject}" out rc
  total=$((total + 1))
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  restore
  if [ "$want" = keep-green ]; then
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE -- rc='$rc')"); fi
    return
  fi
  if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then RESULTS+=("$id MISSED '(wrong reason; rc='$rc')")
  else caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''"); fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  AC-1  an invalidation removed (cache outlives a free)   [must REJECT]"
  echo "  AC-2  cache recorded BEFORE the upload                [must REJECT]"
  echo "  AC-3  the kill switch removed                         [must REJECT]"
  echo "  AC-4  the check's own comparison inverted            [must REJECT]"
  exit 0
fi

echo "== baseline =="
if bash "$CHECK" >"$SB/base.log" 2>&1; then echo "  ok baseline rc=0"; else
  echo "  FAIL baseline" >&2; tail -5 "$SB/base.log" >&2; exit 1; fi
echo "== states =="

# AC-1 drop the clear next to the free
ac1() { patch "$SRC" '    act_cache_ptr_ = nullptr; act_cache_M_ = 0; act_cache_K_ = 0;
    if (w.act)' '    if (w.act)'; }
# AC-2 record the cache BEFORE the upload loop, so a later call can skip an
# upload that never happened and read zero pages. Inserted immediately before the
# device loop, which is above the memcpy the checker compares against.
ac2() { patch "$SRC" '  // Issue all device work in parallel and collect kernel events.' '  if (fuse_act) { act_cache_ptr_ = x.data(); act_cache_M_ = M; act_cache_K_ = (uint32_t)t.K; }
  // Issue all device work in parallel and collect kernel events.'; }
# AC-3 remove the kill switch
ac3() { patch "$SRC" '(std::getenv("INTFER_NO_FUSE_ACT") == nullptr);' 'true;'; }
# AC-4 invert the checker's own comparison
ac4() { patch "$CHECK" '[ "$up_line" -lt "$set_line" ]' '[ "$up_line" -ge "$set_line" ]'; }

ac1; run_state AC-1 'does NOT clear the cache|FAIL' reject
ac2; run_state AC-2 'recorded BEFORE the upload|FAIL' reject
ac3; run_state AC-3 'no kill switch|FAIL' reject
ac4; run_state AC-4 'recorded BEFORE the upload|FAIL' reject

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done
if ! cmp -s "$SRC" "$PRISTINE_SRC" || ! cmp -s "$CHECK" "$PRISTINE_CHK"; then
  echo "  FAIL residue: an artifact differs from its pre-matrix state" >&2; exit 1
fi
echo "  ok residue: both artifacts byte-identical to their pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "ACT-CACHE MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  AC-1: removing an invalidation (a use-after-free waiting to happen) is caught"
  echo "  AC-2: recording the cache before the upload is caught"
  echo "  AC-3: removing the kill switch is caught"
  echo "  AC-4: inverting the checker's own comparison is caught -- the bug it shipped with"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "ACT-CACHE MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
