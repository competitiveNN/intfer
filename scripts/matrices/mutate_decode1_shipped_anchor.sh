#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_decode1_shipped_anchor.sh.
#
# WHAT IS MUTATED, AND WHY IT IS THE CORPUS AND NOT THE CHECKER (audit #21).
# check_decode1_shipped_anchor.sh pins five invariants of the bisect-3 SHIPPED
# arm (examples/roofline_probe_bisect3/main.cpp) to the engine's INT2-tab path
# in src/gemm/quant_gemm.cpp. The checker is correct by construction on a clean
# tree, so weakening it is an unobservable no-op. The observable signal comes
# from a drifted ARTIFACT: a version of the anchor whose invariant no longer
# matches the engine, which only a real drift would let the checker catch.
#
# The anchor under test is copied into an mktemp sandbox OUTSIDE $BUILD_DIR via
# `git archive HEAD`, so the real tree -- committed or not -- is never mutated.
# Restore is a file copy of the pre-patch bytes, not a `git checkout`, because
# the sandbox is a plain tree (not a git work area).
#
# States:
#   baseline  the committed anchor tracks the engine            -> must PASS
#   A         anchor drifts: I1 byte-read becomes an off-by-one
#             (`v0 = c.code[b0]` -> `v0 = c.code[b0 + 1]`)      -> must REJECT
#   B         converse: anchor restored to commit baseline      -> must PASS
#
# Exit: 0 = every state behaved as specified, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECK="$SCRIPT_DIR/../checks/check_decode1_shipped_anchor.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/decode1_shipped_anchor_mutate"
LOG="$STAMP/states"
SB="$(mktemp -d "${TMPDIR:-/dev/shm}/dsa2mut.XXXXXX")"
mkdir -p "$STAMP" "$LOG"
: >"$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

# shellcheck source=scripts/lib/_gate_patch.sh
. "$SCRIPT_DIR/../lib/_gate_patch.sh"
# shellcheck source=scripts/lib/_gate_pred.sh
. "$SCRIPT_DIR/../lib/_gate_pred.sh"
# shellcheck source=scripts/lib/_gate_pop.sh
. "$SCRIPT_DIR/../lib/_gate_pop.sh"

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
bash -n "$CHECK" || { echo "FAIL: $CHECK is not syntactically valid"; exit 2; }

ANCHOR_REL="examples/roofline_probe_bisect3/main.cpp"
ENGINE_REL="src/gemm/quant_gemm.cpp"
echo "== decode1_shipped_anchor mutation matrix =="
echo "  anchor : $ANCHOR_REL (bisect-3 SHIPPED arm)"
echo "  engine : $ENGINE_REL (INT2 table-hoist path)"

# --- archive the committed tree so the real checkout is never mutated --------
setup() {
  # rm -rf then mkdir -p, NOT just rm: mktemp -d created the dir, so a bare
  # rm leaves it gone and `tar -x -C "$SB"` fails with "Cannot open: No such
  # file or directory" -- which the matrix then reports as "could not archive
  # HEAD into sandbox", i.e. a matrix fault read as a checker fault.
  rm -rf "$SB"; mkdir -p "$SB"
  git -C "$ROOT" archive --format=tar HEAD | tar -x -f - -C "$SB" || {
    echo "FAIL: could not archive HEAD into sandbox"; exit 2; }
  [ -f "$SB/$ANCHOR_REL" ] || { echo "FAIL: anchor missing in sandbox"; exit 2; }
  [ -f "$SB/$ENGINE_REL" ]  || { echo "FAIL: engine missing in sandbox"; exit 2; }
  cp "$SB/$ANCHOR_REL" "$SB/.anchor_orig"
  ANCHOR="$SB/$ANCHOR_REL"
  ORIG="$SB/.anchor_orig"
}
restore() { cp "$ORIG" "$ANCHOR"; }
run_check() { bash "$SB/scripts/checks/check_decode1_shipped_anchor.sh" 2>&1; }

# --- verify a patch landed AND the file still parses -------------------------
# (audit #17b: a broken patch that leaves syntax errors must be a state fault,
# not a verdict about the checker; but this anchor is C++, so we only verify the
# marker landed and the file is non-empty.)
verify_patch() {
  if ! grep -q -- "$1" "$ANCHOR"; then
    echo "PATCH VERIFY FAIL: marker '$1' not present in $ANCHOR" >&2
    echo "  This is a STATE fault (fragment left dangling), not a result about" >&2
    echo "  the checker. Fix the anchor so it is unique in the file." >&2
    return 1
  fi
  return 0
}

check_state() { # $1=id  $2=want_rc  $3=expect_re  $4=why
  local id="$1" want="$2" expect="$3" why="$4" o r
  total=$((total + 1))
  o="$(run_check)" ; r=$?
  printf '%s\n' "$o" >"$LOG/$id.log"
  printf '  .. %s; checker rc=%s\n' "$id" "$r"
  if [ "$r" -eq "$want" ] && pred_expect_present "$id expected-RE" "$o" "$expect"; then
    echo "  $id caught -- checker-rc=$r, verdict-polarity=expect-present ($why)"
    echo "$id caught (rc=$r)" >>"$LOG/summary.txt"; caught=$((caught + 1))
  else
    echo "  $id MISSED -- wanted rc=$want and a match for /$expect/ ($why)"
    echo "        got rc=$r; checker output:"
    printf '%s\n' "$o" | tail -4 | sed 's/^/          '
    echo "$id MISSED (rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1))
  fi
}

# --- population snapshot for the leak assertion ------------------------------
POP_BEFORE="$(pop_dirs | tr '\n' ' ')" 2>/dev/null || POP_BEFORE=""

setup
restore
total=0; caught=0; bad=0

echo
echo "-- baseline: the committed anchor tracks the engine -> must PASS --"
b="$(run_check)"; brc=$?
printf '%s\n' "$b" >"$LOG/baseline.log"
if [ "$brc" -ne 0 ]; then
  echo "  FAIL baseline -- the checker must pass on the committed tree,"
  echo "       so a red baseline would invalidate every state below."
  printf '%s\n' "$b" | tail -6 | sed 's/^/    /'
  total=$((total + 1)); bad=$((bad + 1))
else
  pred_expect_present "baseline PASS" "$b" 'DECODE1 SHIPPED ANCHOR: PASS'
  pred_expect_present "baseline is non-vacuous" "$b" 'invariants hold in both' \
    || { echo "  FAIL baseline verdict looks vacuous (no invariant checked)" >&2; bad=$((bad + 1)); }
  echo "  ok    baseline rc=0, the anchor is pinned to the engine"
  total=$((total + 1)); caught=$((caught + 1))
fi

echo
echo "-- states --"

# A: drift I1 in the anchor -- the byte-read index becomes an off-by-one, so the
#    checker's anchor-side I1 pattern finds zero matches and must REJECT.
restore
patch_literal "$ANCHOR" 'v0 = c.code[b0]' 'v0 = c.code[b0 + 1]' \
  && verify_patch 'b0 + 1' || { echo "  FAIL A: patch did not land"; exit 2; }
check_state "A" 1 'I1-chunk-byte-reads \(anchor\) violated' "drifted anchor must be rejected"

# B (converse): restore the anchor; it must once again match the engine.
restore
check_state "B" 0 'DECODE1 SHIPPED ANCHOR: PASS' "restored anchor is accepted (converse)"

echo
echo "-- leak assertion (this matrix owns no stamp dir) --"
if [ "$POP_BEFORE" != "  " ] 2>/dev/null; then
  POP_AFTER="$(pop_dirs | tr '\n' ' ')" 2>/dev/null || POP_AFTER=""
  if [ "$POP_AFTER" = "$POP_BEFORE" ]; then
    echo "  ok    stamp dir population unchanged (no unowned stamps leaked)"
  else
    echo "  FAIL  stamp dir population changed: before=[$POP_BEFORE] after=[$POP_AFTER]"
    echo "RESULT: decode1_shipped_anchor_mutate LEAK"; bad=$((bad + 1))
  fi
else
  echo "  ok    decode1_shipped_anchor does not write stamps into an enumerated dir"
fi

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"

if [ "$caught" -eq "$total" ] && [ "$bad" -eq 0 ]; then
  echo
  echo "DECODE1 SHIPPED ANCHOR MUTATION MATRIX: PASS ($caught/$total states caught, each for its own reason)"
  { echo "stamp=decode1_shipped_anchor_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "DECODE1 SHIPPED ANCHOR MUTATION MATRIX: FAIL ($bad of $total states misbehaved)"
echo "RESULT: a deliberate anchor drift was not caught by the checker it mirrors"
exit 1
