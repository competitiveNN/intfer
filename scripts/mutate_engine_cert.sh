#!/usr/bin/env bash
# Mutation matrix for scripts/check_engine_cert.sh (AUDIT #24).
#
# THE ARTIFACT IS THE SUBJECT, AND IT LIVES IN THE BUILD TREE. The ten-gate
# certification (build/ten_gates/HEAD) is UNTRACKED by design, so states here
# plant it directly and restore from a backup taken at setup. The trap restores
# even on an abnormal exit, because leaving a planted artifact behind would make
# the NEXT session's stack report a certification it never measured.
#
# Per AUDIT #21, checker-weakening is normally void because the artifact is
# correct -- but state E-D weakens the checker while the planted artifact is
# DELIBERATELY STALE, and that is exactly the case where weakening becomes
# observable. That is why E-D is legitimate here and would not be elsewhere.
#
# Usage: scripts/mutate_engine_cert.sh
# Exit: 0 = every state behaved as characterized, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/engine_cert_mutate"
LOG="$STAMP/states"
ART="$BUILD_DIR/ten_gates/HEAD"
CHECK="$SCRIPT_DIR/check_engine_cert.sh"

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
if [ ! -f "$ART" ]; then
  echo "SKIP: no $ART -- run scripts/run_ten_gates.sh first (this matrix plants and"
  echo "      restores the REAL artifact, so there must be one to preserve."
  exit 2
fi
git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null || {
  echo "FAIL: scripts/ is dirty; this matrix edits a sandbox copy."; exit 2; }

mkdir -p "$LOG"; : > "$LOG/summary.txt"
BAK="$(mktemp "${TMPDIR:-/tmp}/engine_cert.XXXXXX")"
cp "$ART" "$BAK"
restore() { cp "$BAK" "$ART"; }
trap 'cp "$BAK" "$ART"; rm -f "$BAK"' EXIT

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_ec.XXXXXX")"
TGT="$SB/check_engine_cert.sh"
cp "$CHECK" "$TGT"
ORIG_C="$SB/.orig_check"; cp "$CHECK" "$ORIG_C"
restore_check() { cp "$ORIG_C" "$TGT"; }
run_check() { bash "$TGT" 2>&1; }
plant() { mkdir -p "$(dirname "$ART")"; printf 'HEAD=%s\npass=10 partial=0 skip=0 fail=0\n' "$1" > "$ART"; }

OLD="$(git -C "$ROOT" rev-list --max-parents=0 HEAD | head -1)"
HEAD_SHA="$(git -C "$ROOT" rev-parse HEAD)"
HEAD_PARENT="$(git -C "$ROOT" rev-parse HEAD~1)"

echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

echo
echo "== baseline: the real artifact must be accepted =="
restore
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -5; exit 1
fi
echo "  ok    baseline passes: $(printf '%s\n' "$b" | sed -n 's/^RESULT: //p')"
echo "  ok    CONVERSE: a healthy certification is accepted, so an always-red"
echo "        checker cannot score as caught"

caught=0; total=0
check_state() { # $1 id $2 expected-rc $3 expected-RE $4 why
  local id="$1" erc="$2" ere="$3" why="$4"
  total=$((total+1)); restore; restore_check
  local o r; o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  if [ "$r" -ne "$erc" ]; then
    echo "  $id MISSED -- rc=$r, want $erc ($why)"
    echo "$id MISSED (rc=$r want $erc)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_present "$id property" "$o" "$ere"; then
    echo "  $id MISSED -- rc right but marker absent: $ere"
    echo "$id MISSED (marker absent)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-present ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states =="

# E-A: a TOOLING-only commit must NOT invalidate a 45-minute engine run. This is
# the state that stops the check from being so aggressive that people disable it.
plant "$HEAD_PARENT"
check_state "E-A" 0 'carries to HEAD' "a tooling-only commit must not invalidate the engine run"

# E-B: an ENGINE input moving must invalidate it. The load-bearing state.
plant "$OLD"
check_state "E-B" 1 'certification STALE' "an engine input changed since the certification"

# E-C: no artifact is UNPROVEN, never a silent pass and never a false FAIL.
rm -f "$ART"
check_state "E-C" 2 'UNPROVEN' "no artifact at all is unproven, not a regression"

# E-D: weaken the engine-input filter while the artifact is STALE. Legitimate
# here and only here, because the planted artifact is deliberately defective.
restore; restore_check; plant "$OLD"
if ! pred_expect_present "E-D patch" "$(cat "$TGT")" '^(src/|examples/|include/|scripts/test_)'; then
  echo "  E-D MISSED -- the input filter anchor is gone (STATE fault)"
  echo "E-D MISSED (anchor)" >> "$LOG/summary.txt"
else
  python3 - "$TGT" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "grep -E '^(src/|examples/|include/|scripts/test_|scripts/_gate_precond\\.sh|scripts/_gate_build\\.sh|scripts/setvars\\.sh)'"
new = "grep -E '^(docs/|reports/|\\.md$)'"
assert s.count(old) == 1, f"anchor count {s.count(old)}"
open(p, 'w').write(s.replace(old, new))
PY
  # Now the stale artifact names a commit, but NO engine input is detected, so the
  # checker must WRONGLY pass. A passing result here is the MISSED we want.
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/E-D.log"
  if [ "$r" -ne 0 ]; then
    echo "  E-D MISSED -- the weakened filter still rejected it (the filter is not"
    echo "        what makes E-B catchable, so the state proves nothing)"
    echo "E-D MISSED (still caught)" >> "$LOG/summary.txt"
  else
    echo "  E-D caught -- with the input filter weakened the stale artifact is NO"
    echo "        longer detected (checker passed), so the filter is load-bearing"
    echo "E-D caught" >> "$LOG/summary.txt"; caught=$((caught+1))
  fi
fi

restore; restore_check
git -C "$ROOT" diff --quiet -- scripts/ 2>/dev/null \
  && echo "  ok    the tracked checker was never modified" || { echo "  FAIL tracked modified"; exit 1; }
[ -f "$ART" ] && echo "  ok    the real certification artifact was restored"
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "ENGINE CERT MUTATE: PASS ($caught/$total states behaved as characterized)"
  mkdir -p "$STAMP"
  { echo "stamp=engine_cert_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "ENGINE CERT MUTATE: FAIL ($caught/$total)"
exit 1