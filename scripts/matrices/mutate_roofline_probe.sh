#!/usr/bin/env bash
# Mutation matrix for scripts/gates/test_roofline_probe.sh's VACUITY FLOOR (2026-10-07).
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
# RUN: scripts/matrices/mutate_roofline_probe.sh
# EXIT: 0 = every state behaved; 1 = a defect escaped.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 1
. "$SCRIPT_DIR/../lib/_gate_patch.sh"

GATE="scripts/gates/test_roofline_probe.sh"
VAC_LOG="build/roofline_probe.log"
[ -f "$GATE" ] || { echo "MUTATE: no gate: $GATE" >&2; exit 1; }
[ -f "$VAC_LOG" ] || { echo "MUTATE: no vacuous log at $VAC_LOG (run the probe once)" >&2; exit 1; }

# *** THE FIXTURE IS DERIVED, NOT READ. *** MEASURED 2026-10-12: this matrix
# used to feed the REAL probe log to B0 and B1 and rely on it being vacuous
# (ceiling 0.000 ms). The probe now measures for real on this box -- the
# ceiling arm reads 0.144 ms -- so B0's premise ("RC=1, ceiling=0.000") and
# B1's premise ("RC=0, ceiling=0.000") were both false: the log carried a real
# ceiling, need_ceiling passed, and B1 reported MISSED. A matrix that reads a
# fixture file is reading a snapshot of the box, and a snapshot is exactly the
# thing that goes stale without notice. So the vacuous fixture is BUILT from
# the real log by zeroing the ceiling value: same arms, same layout, same
# RESULT: PASS, same exit code -- only the number changes. That is what the
# state asserts, and it is what the gate's need_ceiling predicate reads.
patch_ceiling() { # $1 = file, $2 = new ms value
  python3 - "$1" "$2" <<'PY'
import sys, re
p, ms = sys.argv[1], sys.argv[2]
s = open(p).read()
# The probe prints the ceiling arm as `[roofline] stream/shared <ms> ms ...`.
# Anchor on the bracket + prefix so `unstream/shared` is not matched (the
# hostile twin line).
s2, n = re.subn(r'(\] stream/shared +)[0-9.]+( ms)',
                lambda m: m.group(1) + ms + m.group(2), s, count=1)
if n != 1:
    sys.stderr.write("patch_ceiling: no 'stream/shared <ms> ms' found\n")
    sys.exit(1)
open(p, "w").write(s2)
PY
}
zero_ceiling() { patch_ceiling "$1" "0.000"; }
TMPD="$(mktemp -d "${TMPDIR:-/dev/shm}/rp_mat.XXXXXX")"
# *** CLEANUP TRAP, ALL THREE SIGNALS. *** AUDIT #31: an EXIT-only trap does
# not run on SIGTERM, so `timeout`-killing this matrix would leave the temp
# tree behind -- and a verifier whose cleanup IS the restore would leave
# TRACKED SOURCE MUTATED. The INT/TERM bodies clean up AND terminate, so the
# handler fires on every path (bash defers it across a foreground command,
# which is why the matrix's per-state runs are short and synchronous).
cleanup() { rm -rf "$TMPD"; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# NOTE: the probe's exit code is NOT in the log. The gate reads it from the
# INTFER_PROBE_RC env var (test_roofline_probe.sh:91), so there is nothing to
# patch in the file -- the matrix sets RC per state directly. A `patch_rc`
# here would be patching a value the gate never reads, i.e. a no-op that
# reports success. It was silently absent (command not found, and `set -e` is
# not on, so the matrix continued) and is deliberately NOT added.

# B0: baseline -- the derived vacuous log (ceiling zeroed from the real run,
# RC=1) -> must FAIL
cp "$VAC_LOG" "$TMPD/b0.log"
zero_ceiling "$TMPD/b0.log"
echo "== B0: derived vacuous log (RC=1, ceiling=0.000 ms) =="
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
  zero_ceiling "$TMPD/b1.log"
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

# B3: the POSITIVE direction, which B0/B1 alone could not provide. A floor that
# can only ever fire is the always-red twin: it certifies nothing, because
# nothing that runs can satisfy it. MEASURED 2026-10-12 -- the very defect
# this state exists to catch -- the probe now measures for real on this box
# (ceiling arm 0.144 ms, 362.9 GB/s, 79.6% of the 456 GB/s spec), so the
# matrix's B0 and B1 premises were both false and B1 reported MISSED while
# the gate was in fact correct. B3 is the real log, untouched: the gate must
# ACCEPT it. Without this state the matrix proves only that the gate can still
# fire, which is what it proved before and what let the defect ship.
cp "$VAC_LOG" "$TMPD/b3.log"
# ceiling is the real value the probe measured -- no patch.
echo "== B3: REAL log (RC=0, ceiling=real) -- the gate must PASS =="
INTFER_PROBE_LOG="$TMPD/b3.log" INTFER_PROBE_RC=0 \
  bash "$SCRIPT_DIR/test_roofline_probe.sh" \
  >"$TMPD/b3.out" 2>"$TMPD/b3.err"
rc3=$?
if [ "$rc3" -eq 0 ] && grep -q 'ROOFLINE-PROBE TEST: PASS' "$TMPD/b3.out"; then
echo "  ok   B3 PASSED (the real ceiling passes both the RC check and need_ceiling)"
else
echo "  B3 MISSED (rc=$rc3): the gate refused a REAL measurement." >&2
tail -8 "$TMPD/b3.out" >&2
tail -8 "$TMPD/b3.err" >&2
exit 1
fi

echo
echo "ROOFLINE-PROBE MUTATION MATRIX: PASS (4/4)"
echo "  B0: derived vacuous log (RC=1) -> gate refuses via the RC check"
echo "  B1: RC=0 but ceiling=0.000 -> gate refuses via need_ceiling"
echo "  B2: RC=0 and ceiling=0.120 -> gate passes"
echo "  B3: the REAL log untouched -> gate passes (the positive direction)"
echo
echo "NOTE: B0/B1/B2 are synthetic logs derived from the real one by zeroing or"
echo "setting the ceiling value. On this box the probe measures for real (the"
echo "ceiling arm reads 0.144 ms / 362.9 GB/s, 79.6% of the 456 GB/s spec -- MEASURED"
echo "2026-10-12), so a real B3 is available and B2's 0.120 ms is below it; the"
echo "2026-10-08 block in AGENTS.md that said the probe returns 0.000 ms for every"
echo "arm is no longer what this box produces. The matrix certifies the gate's"
echo "two predicates against derived fixtures and its positive direction against"
echo "the real run; the probe's own vacuity guard is the probe's job."
rm -rf "$TMPD"
exit 0
