#!/usr/bin/env bash
# Mutation matrix for scripts/test_roofline_probe.sh's VACUITY FLOOR (2026-10-07).
#
# WHAT IT ASSERTS. The gate now reads the probe's exit code AND asserts a
# numeric ceiling floor (`need_ceiling`). This matrix proves both are
# load-bearing by breaking each and requiring the gate to REFUSE.
#
# States:
#   B0 baseline: the real vacuous log (RC=1, ceiling=0.000 ms) -> must FAIL
#   B1 patch RC: same log but RC=0 -> need_ceiling must still FAIL
#   B2 patch ceiling: RC=0, ceiling=0.120 ms -> must PASS
#
# RUN: scripts/mutate_roofline_probe.sh
# EXIT: 0 = every state behaved; 1 = a defect escaped.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
. "$SCRIPT_DIR/_gate_patch.sh"

GATE="scripts/test_roofline_probe.sh"
VAC_LOG="build/roofline_probe.log"
[ -f "$GATE" ] || { echo "MUTATE: no gate: $GATE" >&2; exit 1; }
[ -f "$VAC_LOG" ] || { echo "MUTATE: no vacuous log at $VAC_LOG (run the probe once)" >&2; exit 1; }

TMPD="$(mktemp -d "${TMPDIR:-/dev/shm}/rp_mat.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

patch_ceiling() { # $1 = file, $2 = new ms value
  python3 - "$1" "$2" <<'PY'
import sys
p, ms = sys.argv[1], sys.argv[2]
s = open(p).read()
old = "0.000 ms"
new = "%s ms" % ms
n = s.count(old)
if n == 0:
    sys.stderr.write("patch_ceiling: no '0.000 ms' found\n"); sys.exit(1)
open(p, "w").write(s.replace(old, new, n))
PY
}

# NOTE: the probe's exit code is NOT in the log. The gate reads it from the
# INTFER_PROBE_RC env var (test_roofline_probe.sh:91), so there is nothing to
# patch in the file -- the matrix sets RC per state directly. A `patch_rc`
# here would be patching a value the gate never reads, i.e. a no-op that
# reports success. It was silently absent (command not found, and `set -e` is
# not on, so the matrix continued) and is deliberately NOT added.

# B0: baseline -- the real vacuous log (RC=1, ceiling=0.000 ms) -> must FAIL
cp "$VAC_LOG" "$TMPD/b0.log"
echo "== B0: real vacuous log (RC=1, ceiling=0.000 ms) =="
# INTFER_PROBE_LOG feeds the synthetic log; INTFER_BUILD_DIR stays on the real
# build tree so gate_require_build's "no build system in $dir" guard passes.
INTFER_PROBE_LOG="$TMPD/b0.log" INTFER_PROBE_RC=1 \
  bash "$SCRIPT_DIR/test_roofline_probe.sh" \
  >"$TMPD/b0.out" 2>"$TMPD/b0.err"
  rc0=$?
  # The probe exits 1 on a vacuous run, so the gate's RC check must fire.
  # The gate prints its FAIL lines to STDERR (convention), so grep both
  # streams: a matrix that greps stdout alone reads a green gate as red and
  # vice versa -- the B0 "MISSED" was exactly that, for a line that was
  # present in b0.err and absent from b0.out.
  if [ "$rc0" -ne 0 ] && grep -q 'FAIL: the probe exited rc=1' "$TMPD/b0.out" "$TMPD/b0.err"; then
  echo "  ok   B0 REFUSED (vacuous log -> RC=1 caught by the gate)"
  else
  echo "  B0 MISSED (rc=$rc0): the vacuous log did not fail the gate." >&2
  tail -5 "$TMPD/b0.out" >&2
  tail -5 "$TMPD/b0.err" >&2
  exit 1
  fi

  # B1: RC=0, ceiling stays 0.000 -> need_ceiling must still FAIL
  # (RC is set via INTFER_PROBE_RC, not patched in the file -- see the note above.)
  cp "$VAC_LOG" "$TMPD/b1.log"
  # ceiling stays 0.000 ms -- no patch needed, the log already carries it.
  echo "== B1: RC=0, ceiling=0.000 ms (need_ceiling must fire) =="
  INTFER_PROBE_LOG="$TMPD/b1.log" INTFER_PROBE_RC=0 \
  bash "$SCRIPT_DIR/test_roofline_probe.sh" \
  >"$TMPD/b1.out" 2>"$TMPD/b1.err"
  rc1=$?
  if [ "$rc1" -ne 0 ] && grep -q 'below the 0.012 ms floor' "$TMPD/b1.out" "$TMPD/b1.err"; then
  echo "  ok   B1 REFUSED (need_ceiling caught the zero ceiling even with RC=0)"
  else
  echo "  B1 MISSED (rc=$rc1): need_ceiling did not fire." >&2
  tail -5 "$TMPD/b1.out" >&2
  tail -5 "$TMPD/b1.err" >&2
  exit 1
  fi

  # B2: RC=0, ceiling to 0.120 ms -> must PASS
  cp "$VAC_LOG" "$TMPD/b2.log"
  patch_ceiling "$TMPD/b2.log" "0.120"
  echo "== B2: RC=0, ceiling=0.120 ms (real run shape) =="
  INTFER_PROBE_LOG="$TMPD/b2.log" INTFER_PROBE_RC=0 INTFER_PROBE_SKIP_ARMS=1 \
  bash "$SCRIPT_DIR/test_roofline_probe.sh" \
  >"$TMPD/b2.out" 2>"$TMPD/b2.err"
  rc2=$?
  if [ "$rc2" -eq 0 ] && grep -q 'ROOFLINE-PROBE TEST: PASS' "$TMPD/b2.out"; then
  echo "  ok   B2 PASSED (real ceiling passes both the RC check and need_ceiling)"
  else
  echo "  B2 MISSED (rc=$rc2): a real ceiling should pass the gate." >&2
  tail -8 "$TMPD/b2.out" >&2
  tail -8 "$TMPD/b2.err" >&2
  exit 1
  fi

echo
echo "ROOFLINE-PROBE MUTATION MATRIX: PASS (3/3)"
echo "  B0: vacuous log (RC=1) -> gate refuses via the RC check"
echo "  B1: RC=0 but ceiling=0.000 -> gate refuses via need_ceiling"
echo "  B2: RC=0 and ceiling=0.120 -> gate passes"
echo
echo "EXCLUSION: B2 is a synthetic log, not a real probe run. The probe itself"
echo "is currently broken on this box (returns 0.000 ms for every arm, caught by"
echo "its own vacuity guard), so a real B2 cannot be obtained without fixing the"
echo "profiling path. The matrix certifies the gate's two predicates"
echo "independently; the probe's vacuity guard is the probe's own job and is"
echo "tested separately by running it."
rm -rf "$TMPD"
exit 0
