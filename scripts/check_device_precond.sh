#!/usr/bin/env bash
# Re-runnable proof that every GPU gate SKIPS (exit 2) when it cannot get a
# device, in BOTH ways a device can be unavailable.
#
# WHY THIS SCRIPT EXISTS (audit #17). Audit #16 committed two stubs --
# scripts/fixtures/no-device/sycl-ls and scripts/fixtures/no-render-node/python3
# -- and AGENTS.md recorded the measured result ("node open + 1 GPU -> rc=0",
# "rc=2x5 per direction"). But NOTHING in the tree ever RAN them:
#
#     $ grep -rn 'fixtures' scripts/*.sh      # -> no matches
#
# The stubs were committed; the wiring that drives them was not. So the proof
# still lived in a transcript, with a decorative fixture attached to it -- the
# exact defect the fixture was added to fix, one level up. Anyone (including the
# next session) had to re-derive "prepend the dir to PATH" and "set
# INTFER_SYCL_LS" by reading the stub's header comments before the claim could
# be re-checked at all. A stub with no driver is a comment with a shebang.
#
# So this is the driver, in the same shape as scripts/mutate_fold_matrix.sh and
# scripts/mutate_gate_inventory.sh: it stamps the sha it was proven against and
# writes a per-direction log, so the next session runs ONE command instead of
# re-deriving the experiment. The stubs moved from "documentation" to "input".
#
# NAMING, and it is load-bearing. This file must NOT be called test_*.sh.
# `ls scripts/test_*.sh` IS the definition of the gate population that
# gate_inventory.sh classifies, and this checker's own code names the stub path
# (NODEV_STUB=".../fixtures/no-device/sycl-ls"), so as a test_* it would be
# classified a GPU gate with no device guard -- a false MISSING row, against a
# checker that by design needs no device. The other meta-checkers
# (gate_inventory.sh, fold_report_counts.sh, mutate_*.sh) are check_/mutate_/
# plain names for the same reason. Renaming it to test_* is a real regression.
#
# WHAT IT ASSERTS, and why each part is non-vacuous:
#
#   1. SEAM DISCRIMINATION. The zero-device direction hinges on
#      INTFER_SYCL_LS pointing at the stub. If that seam rots (renamed var,
#      stub deleted), gate_gpu_count returns "could not count" -- which
#      gate_require_device ALSO turns into exit 2. Same rc, different cause, and
#      the whole direction silently degrades into "sycl-ls is missing" while
#      still reporting green. So the harness requires the two causes to produce
#      DIFFERENT diagnostics, and fails if they stop being distinguishable. This
#      is the part that keeps direction 2 from being a tautology.
#
#   2. PER-GATE EXIT CODE, per direction. Every device-requiring gate must exit
#      2. Exit 0 is a silent skip wearing a pass; exit 1 is indistinguishable
#      from a real regression, which is what trained everyone to ignore red.
#
#   3. PER-GATE DIAGNOSTIC. rc=2 for the WRONG reason is still a defect: it
#      reports box state that is not the box state, and it is how a guard
#      quietly degrades into a different (weaker) mechanism. Measured, not
#      assumed: this is what showed test_tp2_27b.sh exits 2 in the
#      node-unreachable direction via its COUNT guard ("needs 2 GPU(s), 1
#      visible"), never mentioning the render node -- correct for a 2-GPU
#      requirement, where both failure modes collapse to "count < 2" and a node
#      probe would be redundant. The harness encodes that as an accepted
#      alternative rather than pretending a node guard is required.
#
#   4. CLASSIFICATION CROSS-CHECK. The device-gate list is derived here, and
#      cross-checked against the RENDER-NODE column gate_inventory.sh reports.
#      Two independent derivations that must AGREE. Deliberately not a third
#      hand-maintained table: the memory note records that the family reopened
#      precisely because nine copies of one block drifted apart silently.
#
#   5. BEHAVIOURAL SELF-CHECK. Assertions 2 and 3 are only meaningful if the
#      harness can FAIL, so expect_skip is run against two synthetic gates: one
#      that runs to completion (must be rejected -- otherwise every gate "passes"
#      because nothing is ever checked) and one that skips for the wrong cause
#      (must be rejected). Behavioural, on real executable scripts, never a
#      textual grep of this file's own source -- a self-check that greps itself
#      is vacuous by construction.
#
# SCOPE / HONEST LIMITS (do not read the pass as more than it is):
#   * THREE directions are measured: the two failure directions (skip paths,
#     asserted strictly) and a HEALTHY column (liveness only -- see the block
#     above). The healthy column asserts rc is 0 or 2, that an rc=2 is
#     EXPLAINED, and that nothing crashes, times out, or goes red. It does NOT
#     assert that a gate's correctness assertions hold; that is what running the
#     gate is for, and its per-gate results remain AGENTS.md's job.
#   * test_tp2_27b.sh needs 2 GPUs, so on a 1-GPU box its healthy rc=2 is
#     legitimate. Its two failure directions ARE covered, but only via the count
#     guard (see 3).
#   * A gate may legitimately be absent from this matrix if gate_inventory says
#     it does not need a device. That is the cross-check's job, not a table's.
#
# Env:
#   INTFER_BUILD_DIR   build tree (default $ROOT/build) -- where logs live
#   INTFER_GATE_NO_BUILD=1  recommended: this tests PRECONDITIONS, so paying a
#                     ninja no-op per gate per direction only adds minutes.
#                     Not forced, because a stale binary is a different defect.
#
# Exit: 0 all assertions held; 1 any assertion failed (including the self-check).
set -u

# The gates inherit this from our environment (we invoke them with plain
# `bash "$gate"`), so setting it here covers every run below. Defaulted to 1
# because this harness tests PRECONDITIONS: a per-gate ninja no-op would add
# minutes and prove nothing about the skip paths. Overridable (set 0) because
# silently skipping the build guard would be its own kind of unmeasured claim.
export INTFER_GATE_NO_BUILD="${INTFER_GATE_NO_BUILD:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/device_precond_matrix"
STAMP="$LOG_DIR/last_pass.stamp"
NODEV_STUB="$SCRIPT_DIR/fixtures/no-device/sycl-ls"
NONODE_DIR="$SCRIPT_DIR/fixtures/no-render-node"
mkdir -p "$LOG_DIR" || exit 1

FAILURES=0
# While the SELF-CHECK is running, expect_skip is EXPECTED to reject its synthetic
# gates, and it reports a rejection through bad(). Without this, the two intended
# rejections each add a phantom failure and the run ends FAIL(2) with every real
# assertion green -- a harness whose self-check makes it permanently red, which is
# the always-red twin of the vacuous check. Only the self-check's OWN explicit
# verdicts (the "SELFCHECK: ..." lines) are allowed to count.
SUPPRESS=0
note() { printf '%s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; [ "$SUPPRESS" = 1 ] || FAILURES=$((FAILURES + 1)); }
ok()   { printf '  ok    %s\n' "$*"; }

# --- fixture pre-flight -----------------------------------------------------
# Fail closed if an input moved: these stubs are the experiment, and a harness
# that runs against a missing fixture measures nothing while reporting green.
for f in "$NODEV_STUB" "$NONODE_DIR/python3"; do
  if [ ! -f "$f" ]; then
    echo "FAIL: fixture missing: $f" >&2
    echo "  The device-failure directions cannot be measured without it. Restore" >&2
    echo "  it (scripts/fixtures/, committed) rather than re-deriving a /tmp stub:" >&2
    echo "  a stub that only exists in a transcript is the defect this file fixes." >&2
    exit 1
  fi
  if [ ! -x "$f" ]; then
    # PATH shadowing REQUIRES the execute bit: bash skips a non-executable file
    # while resolving PATH, so a 644 stub silently hands every probe back to the
    # real tool and the direction becomes a no-op that still "passes".
    echo "FAIL: fixture not executable: $f (mode $(stat -c %a "$f" 2>/dev/null || echo '?'))" >&2
    echo "  A non-executable stub is BYPASSED by PATH lookup, so the direction" >&2
    echo "  would test nothing while reporting green. Fix: chmod +x (git records" >&2
    echo "  mode 100755; check 'git ls-files -s' if the fix does not survive)." >&2
    exit 1
  fi
done

# --- assertion 3+2: one gate, one direction, one expected cause --------------
# $1 = gate path (executable script)   $2 = direction tag (none|node-unreachable|zero-device)
# $3 = diagnostic substring that MUST appear   $4 = alternative accepted substring
#        (empty = no alternative accepted; a count-guarded 2-GPU gate legitimately
#         reports the count cause in BOTH directions -- see header note 3)
# rc of the gate is run INHERITED so the caller sees the real exit code.
expect_skip() {
  local gate="$1" dir="$2" want="$3" alt="${4:-}" out rc
  out="$LOG_DIR/$(basename "$gate" .sh).$dir.log"
  case "$dir" in
    node-unreachable) PATH="$NONODE_DIR:$PATH" timeout 300 bash "$gate" >"$out" 2>&1 ;;
    zero-device)      INTFER_SYCL_LS="$NODEV_STUB" timeout 300 bash "$gate" >"$out" 2>&1 ;;
    *)                echo "internal: bad direction '$dir'" >&2; return 99 ;;
  esac
  rc=$?
  # Classify WHICH cause fired, for the report line. A count-guarded 2-GPU gate
  # legitimately reports the count cause in BOTH directions, so this is data,
  # not a verdict -- the verdict is in the want/alt test below.
  if grep -qF 'renderD128' "$out"; then LAST_CAUSE=node
  elif grep -qF 'enumerates 0 device' "$out"; then LAST_CAUSE=count0
  elif grep -qF 'GPU(s),' "$out"; then LAST_CAUSE=count
  else LAST_CAUSE=other; fi
  if [ "$rc" != "2" ]; then
    bad "$(basename "$gate") [$dir] exited rc=$rc, want 2 (SKIP)"
    sed -n '1,6p' "$out" | sed 's/^/          | /'
    return 1
  fi
  # want, or a non-empty accepted alternative, must be present. Both absent is
  # a wrong-cause skip, which is how a guard degrades to a weaker mechanism
  # while still reading green.
  if ! grep -qF "$want" "$out" && { [ -z "$alt" ] || ! grep -qF "$alt" "$out"; }; then
    bad "$(basename "$gate") [$dir] skipped for the WRONG cause (rc=2, but said neither '$want' nor '${alt:-<none>}')"
    sed -n '1,6p' "$out" | sed 's/^/          | /'
    return 1
  fi
  return 0
}

# --- assertion 5: behavioural self-check ------------------------------------
# Runs the assertion above against synthetic gates. If expect_skip cannot fail,
# the whole matrix proves nothing, so this runs FIRST and gates the run.
selfcheck() {
  local d bad_=0 g
  d="$(mktemp -d "$LOG_DIR/.selfcheck.XXXXXX")"
  SUPPRESS=1   # rejections below are the point, not failures of this harness

  # (a) a gate that RUNS TO COMPLETION under a failure direction. expect_skip
  #     MUST reject it: this is what proves the matrix is not vacuously green.
  g="$d/runs_anyway.sh"
  cat >"$g" <<'G'
#!/usr/bin/env bash
echo "SKIPPED: cannot open /dev/dri/renderD128 from this shell."   # plausible lie
exit 0                                                            # ...but a PASS
G
  chmod +x "$g"
  if expect_skip "$g" zero-device "enumerates 0 device" >/dev/null 2>&1; then
    SUPPRESS=0; bad "SELFCHECK: a gate exiting 0 under zero-device was ACCEPTED"
    bad_=1; SUPPRESS=1
  else
    ok "SELFCHECK rejects a gate that passes under a failure direction"
  fi

  # (b) a gate that skips for the WRONG cause. If rc=2 alone were the whole
  #     test, a guard that silently degraded to a weaker mechanism would read
  #     green while reporting box state that is not the box state.
  g="$d/wrong_cause.sh"
  cat >"$g" <<'G'
#!/usr/bin/env bash
echo "SKIPPED: cannot open /dev/dri/renderD128 from this shell."
exit 2
G
  chmod +x "$g"
  if expect_skip "$g" zero-device "enumerates 0 device" >/dev/null 2>&1; then
    SUPPRESS=0; bad "SELFCHECK: rc=2 for the WRONG cause was ACCEPTED"
    bad_=1; SUPPRESS=1
  else
    ok "SELFCHECK rejects rc=2 for the wrong cause"
  fi

  # (c) the converse, so (a)+(b) cannot pass merely because expect_skip rejects
  #     EVERYTHING. The right cause must be accepted, or the harness is
  #     always-red and proves nothing in the other direction.
  g="$d/right_cause.sh"
  cat >"$g" <<'G'
#!/usr/bin/env bash
echo "SKIPPED: render node opens but Level Zero enumerates 0 device(s)."
exit 2
G
  chmod +x "$g"
  if expect_skip "$g" zero-device "enumerates 0 device" >/dev/null 2>&1; then
    ok "SELFCHECK accepts the correct cause (harness is not simply always-red)"
  else
    SUPPRESS=0; bad "SELFCHECK: the CORRECT cause was REJECTED -- always-red harness"
    bad_=1; SUPPRESS=1
  fi

  # Synthetic logs are named after the synthetic gates; drop them so a later
  # reader cannot mistake them for real per-gate evidence.
  rm -f "$LOG_DIR"/runs_anyway.*.log "$LOG_DIR"/wrong_cause.*.log \
        "$LOG_DIR"/right_cause.*.log
  SUPPRESS=0
  rm -rf "$d"
  return $bad_
}

note "== SELF-CHECK (harness must be able to fail) =="
if ! selfcheck; then
  echo >&2
  echo "DEVICE PRECOND: FAIL -- the self-check did not hold, so the matrix below" >&2
  echo "  proves nothing (a check that cannot fail is the vacuous-gate trap)." >&2
  exit 1
fi

# --- assertion 1: the zero-device seam must be DISTINGUISHABLE --------------
# If a rotted seam and a genuine 0-device box produce the SAME diagnostic, the
# whole zero-device direction becomes a tautology that can never go red.
note "== SEAM DISCRIMINATION (a rotted seam must not look like a 0-device box) =="
probe_diag() { # $1 = value for INTFER_SYCL_LS ; prints the skip diagnostic
  local stub="$1" g
  g="$(mktemp "$LOG_DIR/.seam.XXXXXX")"
  cat >"$g" <<'G'
#!/usr/bin/env bash
set -u
SCRIPT_DIR="X"; ROOT="X"
. "X/_gate_precond.sh"
gate_require_device || exit 2
exit 0
G
  chmod +x "$g"
  sed -i "s|X|$SCRIPT_DIR|g" "$g"
  INTFER_SYCL_LS="$stub" bash "$g" 2>&1 | grep -m1 -E 'SKIPPED|SKIP:'
  rm -f "$g"
}
DIAG_STUB="$(probe_diag "$NODEV_STUB")"
DIAG_ROTTEN="$(probe_diag "$LOG_DIR/definitely-not-a-real-sycl-ls")"
if [ -z "$DIAG_STUB" ]; then
  bad "seam: the 0-device stub produced NO diagnostic -- the direction measures nothing"
elif [ "$DIAG_STUB" = "$DIAG_ROTTEN" ]; then
  bad "seam: a ROTTED seam and a genuine 0-device box are indistinguishable"
  bad "  both said: $DIAG_STUB"
  bad "  -> direction 2 is a tautology. Fix INTFER_SYCL_LS or the stub first."
else
  ok "seam: 0-device and unmeasurable are distinguishable"
  note "        0-device   : $(printf '%s' "$DIAG_STUB" | cut -c1-66)"
  note "        unmeasurable: $(printf '%s' "$DIAG_ROTTEN" | cut -c1-66)"
fi

# --- derive the device-gate list, then CROSS-CHECK it (assertion 4) ----------
# Derived here from the gates' own code, independently of gate_inventory.sh, and
# then required to agree with the RENDER-NODE column gate_inventory reports. Two
# derivations that must match: if a new GPU gate lands and only one side learns
# about it, this fails loudly instead of the matrix quietly covering less.
note "== CLASSIFICATION CROSS-CHECK (this list vs gate_inventory.sh) =="
# *** PINNED TO A DETERMINISTIC DEVICE COUNT, NOT THE REAL BOX. *** Two derivations
# that must agree have to be compared under conditions that do not move, or the
# comparison is a flaky gate. This invoked gate_inventory.sh LIVE, and on a box that
# momentarily reported ZERO devices gate_inventory took its BOX branch -- a different
# verdict, a different table, exit 2 by design since audit #29 -- and the cross-check
# reported "classification drift" with nothing actually drifting.
# MEASURED, both halves: the step FAILED inside a saturated stack run and PASSED when
# re-run directly seconds later, with no code change in between. That is the shape of
# a box-state flake, and the fix is not "re-run it until it goes green" -- it is to
# stop the box from being an input at all.
# The stub is a committed fixture (scripts/fixtures/two-device/sycl-ls) rather than a
# generated one, so the reason is re-readable next to the thing it explains. The count
# is 2, this project's design target; the RENDER-NODE column being cross-checked is
# count-independent, so what matters is that it is PINNED rather than OBSERVED.
TWO_DEV_STUB="$SCRIPT_DIR/fixtures/two-device/sycl-ls"
[ -x "$TWO_DEV_STUB" ] || { echo "FAIL: missing $TWO_DEV_STUB -- the cross-check cannot be" >&2
                           echo "      made deterministic, and a live gate_inventory makes it flaky." >&2
                           exit 1; }
INV_TABLE="$(INTFER_SYCL_LS="$TWO_DEV_STUB" bash "$SCRIPT_DIR/gate_inventory.sh" 2>/dev/null)"
inv_devices="$(printf '%s\n' "$INV_TABLE" | awk '$NF=="via-count" || $NF=="render-node" {print $1}')"
# *** A SHORT OR EMPTY TABLE IS NOT "THE OTHER SIDE DISAGREES". *** The observed
# failure was gate_inventory's side MISSING ONE GATE, which the cross-check reported
# as classification drift. Pinning the invocation removes the load-sensitivity, but
# the honest residual risk is that gate_inventory exits EARLY -- before printing its
# table at all, for any of its own reasons -- and its empty result is then read as
# "the other side disagrees". So a table that did not arrive is refused in its own
# terms, and says so, instead of being compared.
case "$INV_TABLE" in
  *"RENDER-NODE"*) : ;;
  *) bad "gate_inventory.sh returned NO TABLE to cross-check (its own preconditions or"
     bad "  an early exit). That is not classification drift, so do not debug the gates:"
     bad "  there is nothing to compare against. Re-run gate_inventory.sh on its own."
     continue_gate ;;
esac
if [ -z "$inv_devices" ]; then
  bad "gate_inventory.sh's table arrived but classifies NO gate as needing a device,"
  bad "  while this harness classifies at least one. One of the two derivations is"
  bad "  empty, and an empty harvest is not a disagreement (audit #20)."
  continue_gate
fi
own_devices=""
for f in scripts/test_*.sh; do
  [ -f "$f" ] || continue
  # GPU if it mentions a GPU tool or launches --gpu; needs a guard if it also
  # carries one. Comment-stripped so a rationale comment cannot classify a gate
  # as guarded (the audit-#14 defect: the ONLY gate_require_gpus mention in
  # test_device_argmax.sh was inside a comment).
  code="$(sed -e 's/[[:space:]]#.*$//' "$f")"
  printf '%s\n' "$code" | grep -qE -- '(--gpu|sycl-ls|level_zero|xpu-smi|dpas_fold_probe)' || continue
  printf '%s\n' "$code" | grep -qE 'gate_require_device|gate_require_render_node|gate_require_gpus|gate_gpu_count' || {
    echo "  FAIL  $(basename "$f") needs a device but carries NO guard at all." >&2
    echo "        It will exit 0 on a 1-GPU box and go RED when a GPU gate loses" >&2
    echo "        its device. Add gate_require_device (both failure modes) or" >&2
    echo "        gate_require_gpus N (count only) BEFORE the first real work." >&2
    FAILURES=$((FAILURES + 1)); continue
  }
  own_devices="$own_devices $(basename "$f")"
done
own_devices="$(printf '%s\n' $own_devices | sort | tr '\n' ' ')"
inv_devices="$(printf '%s\n' $inv_devices | sort | tr '\n' ' ')"
if [ "$own_devices" != "$inv_devices" ]; then
  bad "classification drift: this harness and gate_inventory.sh disagree"
  bad "  this harness: $own_devices"
  bad "  gate_inventory: $inv_devices"
  bad "  Reconcile instead of adding a third list (the copy-drift family)."
else
  ok "both derivations agree: $(printf '%s' "$own_devices" | wc -w) device gate(s)"
fi

# --- the matrix (assertions 2 + 3) -----------------------------------------
# Per gate, the accepted cause(s). A count-guarded 2-GPU gate reports the COUNT
# cause in both directions and that is CORRECT (both failure modes collapse to
# count < 2), so it is recorded rather than force-fitted to a node message.
cause_node="cannot open /dev/dri/renderD128"
cause_zero="enumerates 0 device"

NDEV_OWN="$(printf '%s\n' $own_devices | grep -c . || true)"
note ""
note "== MATRIX: $NDEV_OWN device gate(s) x 2 failure directions =="
for name in $own_devices; do
  gate="scripts/$name"
  if grep -qE 'gate_require_device|gate_require_render_node' "$gate"; then
    # Covers BOTH failure modes: each direction has exactly one correct cause.
    alt_node=""; alt_zero=""
  else
    # Count-only guard. For a >=2 GPU requirement both failure modes collapse to
    # "count < need", so the count cause is CORRECT in both directions and a
    # node message must not be demanded. Anchored on the count wording ("N
    # visible") so a reworded message cannot silently satisfy it.
    alt_node="GPU(s),"; alt_zero="GPU(s), 0 visible"
  fi
  printf '  %-24s node-unreachable  ' "$name"
  if expect_skip "$gate" node-unreachable "$cause_node" "$alt_node"; then
    printf 'rc=2 via %s\n' "$LAST_CAUSE"
  fi
  printf '  %-24s zero-device       ' "$name"
  if expect_skip "$gate" zero-device "$cause_zero" "$alt_zero"; then
    printf 'rc=2 via %s\n' "$LAST_CAUSE"
  fi
done

# --- the HEALTHY direction (the third column) -------------------------------
# Why this exists. The two failure directions prove a gate SKIPS when it cannot
# get a device. They say NOTHING about what a gate does on a working box, and
# that result lived in AGENTS.md prose -- which is a proof in a transcript, the
# exact defect the fixtures were committed to end (audit #17), one level up. So
# the healthy column is measured here and recorded in the stamp below.
#
# What it does and does NOT assert, because over-asserting here is how a
# correctness gate becomes a flaky gate:
#   * rc 0 -> the gate ran and passed.
#   * rc 2 -> LEGITIMATE on a 1-GPU box: test_tp2_27b needs 2 devices, and the
#     per-arm gates (gpu_optin, device_argmax) report PARTIAL when their TP=2
#     arms skip. So this harness must NOT demand rc=0 -- measured on this box:
#     3 gates return 0 and 3 return 2, all healthy.
#   * rc 2 must be EXPLAINED. An rc=2 whose log carries no skip diagnostic is a
#     silent skip and is scored as a failure, not a pass.
#   * rc 1, or death by signal (>=128: 134 SIGABRT, 139 SIGSEGV), or a timeout
#     (124) -> FAIL. A gate that goes RED on a healthy box is the one thing this
#     column exists to catch, and a timeout is explicitly NOT scored as a pass.
# This column therefore cannot tell you a gate's CORRECTNESS -- only that it is
# neither red nor silently skipping on a box that is working.
#
# Env: INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1  skip this column. Default OFF, and
#   the mutation matrix sets it: states M-A/M-C break the GUARDS, so under a
#   healthy box the gate would run to completion (minutes) and its rc would
#   change for reasons unrelated to the mutation. A stated purpose, not a
#   silent default-off.
HEALTHY_RCS=""
expect_healthy() { # $1 = gate path
  local gate="$1" out rc
  out="$LOG_DIR/$(basename "$gate" .sh).healthy.log"
  timeout 900 bash "$gate" >"$out" 2>&1
  rc=$?
  HEALTHY_RCS="$HEALTHY_RCS $(basename "$gate" .sh)=$rc"
  case "$rc" in
    0) return 0 ;;
    2)
      if grep -qE 'SKIPPED|SKIP:|PARTIAL' "$out"; then return 0; fi
      bad "$(basename "$gate") [healthy] exited 2 with NO skip diagnostic -- a silent skip"
      return 1
      ;;
    124)
      bad "$(basename "$gate") [healthy] TIMED OUT (900s) -- not a pass; likely box load"
      return 1
      ;;
    *)
      bad "$(basename "$gate") [healthy] exited rc=$rc; a gate must not go red on a healthy box"
      [ "$rc" -ge 128 ] 2>/dev/null && bad "  (death by signal $(($rc - 128)))"
      sed -n '1,4p' "$out" | sed 's/^/          | /'
      return 1
      ;;
  esac
}

if [ "${INTFER_DEVICE_PRECOND_SKIP_HEALTHY:-0}" = "1" ]; then
  note ""
  note "== HEALTHY COLUMN: SKIPPED (INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1) =="
  note "   The healthy results in the stamp are therefore STALE, not re-measured."
else
  note ""
  note "== HEALTHY COLUMN: $NDEV_OWN device gate(s) on this box (rc 0 or 2; a 2 must be explained) =="
  for name in $own_devices; do
    printf '  %-24s healthy            ' "$name"
    if expect_healthy "scripts/$name"; then
      # Strip up to the LAST '=' in the accumulator to get THIS gate's rc.
      # (The first version used '##* =' with a space before the '=', which can
      # never match "name=rc", so it printed the whole accumulator.)
      printf 'rc=%s (ok)\n' "${HEALTHY_RCS##*=}"
    fi
  done
fi

# --- verdict ----------------------------------------------------------------
echo
if [ "$FAILURES" -ne 0 ]; then
  echo "DEVICE PRECOND: FAIL ($FAILURES failed assertion(s))" >&2
  echo "  Either a GPU gate does not SKIP LOUDLY when it cannot get a device, or" >&2
  echo "  one went red / silently skipped on a healthy box. Per-run logs:" >&2
  echo "  $LOG_DIR" >&2
  exit 1
fi
echo "DEVICE PRECOND: PASS"
echo "  every device gate exits 2 (SKIP) in both failure directions, naming the"
echo "  real cause; and the 0-device direction is distinguishable from a rotted"
echo "  INTFER_SYCL_LS seam, so it can still go red."
# The healthy claim must follow the healthy RUN. An OK line that says "no gate
# went red" when the column was skipped is the same defect as ddea91d ("the OK
# line must not claim the tally check ran when it skipped"), committed into this
# harness by this very change.
if [ "${INTFER_DEVICE_PRECOND_SKIP_HEALTHY:-0}" = "1" ]; then
  echo "  NOT CHECKED: the healthy column (INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1), so"
  echo "  this pass says NOTHING about gate liveness on a working box, and the"
  echo "  'healthy=' line in the stamp below is STALE, not re-measured."
else
  echo "  on this healthy box no gate went red, timed out, or silently skipped."
fi
echo "  This harness says NOTHING about whether a gate's CORRECTNESS assertions"
echo "  hold -- it checks the skip paths, plus liveness when the healthy column ran."

{
  echo "sha=$(git rev-parse HEAD)"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gates=$(printf '%s' "$own_devices" | tr ' ' ',')"
  echo "directions=node-unreachable,zero-device,healthy"
  echo "healthy=$(printf '%s' "$HEALTHY_RCS" | tr -s ' ' | sed 's/^ //;s/ /,/g')"
  echo "assertions=selfcheck,seam-discrimination,classification-crosscheck,per-gate-rc,per-gate-cause,healthy-liveness"
} >"$STAMP"
echo "  stamp: $STAMP"
exit 0
