#!/usr/bin/env bash
# Assert the RESULT-filling benchmark block in the reports is still RUNNABLE and
# still refuses to fabricate wall-time.
#
# WHY. The block (reports/2026-09-30-pending-bench-backlog.md, "ONE COMMAND FILLS
# EVERY `RESULT:` LINE ABOVE") was written by hand and verified by hand once.
# Nothing re-verified it afterwards, and the properties that matter are exactly
# the ones a later edit silently breaks:
#   * a flag renamed in source, leaving the block setting an env var nobody reads
#     -- the run then "succeeds" while measuring the BASE arm three times;
#   * a numeric tok/s pasted in, which is the retracted-wall-time class;
#   * a cross-reference or a RESULT line orphaned so part of the batch is
#     unreachable from the block.
# Those are cheap to check and impossible to notice by reading.
#
# Uses the repo's existing guards rather than new copies:
#   _gate_enum.sh  enum_require_nonempty / enum_require_min  (AUDIT #20)
#   _gate_pred.sh  pred_expect_absent / pred_expect_present  (polarity named)
#
# Usage: scripts/check_result_block.sh
# Exit: 0 = all properties hold, 1 = a property drifted, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
enum_reset

BACKLOG="reports/2026-09-30-pending-bench-backlog.md"
FOLD="reports/2026-09-30-dpas48-fold.md"
SRC_DIRS="src examples/intfer_cli scripts"

echo "== RESULT-BLOCK CHECK =="

extract_block() { # the fenced bash block under the ONE COMMAND heading
  awk '/^## ONE COMMAND FILLS/{f=1} f&&/^```bash$/{p=1;next} p&&/^```$/{exit} p' "$BACKLOG"
}
BLOCK="$(extract_block)"

# Anti-vacuity: an empty harvest cannot certify anything (AUDIT #20).
enum_require_nonempty "result-block" "$(printf '%s\n' "$BLOCK" | grep -c . || true)" \
  "lines in the RESULT-filling block" || exit 1
if [ -z "$BLOCK" ]; then echo "  FAIL  the block did not extract"; exit 1; fi

bad=0

# ---- (a) every flag the block sets must exist in source --------------------
# Only EXECUTED settings: `export FOO=` / `FOO=` prefixes, not every token that
# merely looks like one. A comment or a prose mention is not a setting.
flags="$(printf '%s\n' "$BLOCK" \
  | grep -oE '(^|[[:space:]])(export[[:space:]]+)?INTFER_[A-Z0-9_]+=' \
  | grep -oE 'INTFER_[A-Z0-9_]+' | LC_ALL=C sort -u)"
enum_require_nonempty "block-flags" "$(printf '%s\n' "$flags" | grep -c . || true)" \
  "INTFER_* flags set by the block" || bad=$((bad+1))
printf '%-4s %-28s %s\n' "" "flag" "found in source"
for f in $flags; do
  if grep -rqF -- "$f" $SRC_DIRS 2>/dev/null; then
    printf '  %-4s %-28s %s\n' "ok" "$f" "yes"
  else
    printf '  %-4s %-28s %s\n' "FAIL" "$f" "NOT FOUND -- the block would set a var nobody reads"
    bad=$((bad+1))
  fi
done

# ---- (b) no numeric tok/s inside the block --------------------------------
# The block is instructions; a measured tok/s in it is a claim with no run
# behind it. Polarity named at the call site.
toks="$(printf '%s\n' "$BLOCK" | grep -cE '[0-9]+(\.[0-9]+)?[[:space:]]*tok/s' || true)"
if pred_expect_absent "no-tok-s-in-block" "$toks" '^[1-9]'; then
  :
else
  echo "  note  $toks line(s) in the block match a tok/s pattern"
fi
if printf '%s\n' "$BLOCK" | grep -qE '[0-9]+(\.[0-9]+)?[[:space:]]*tok/s'; then
  echo "  FAIL  a numeric tok/s appears inside the block -- that is a result, not"
  echo "        an instruction, and it would be an unmeasured claim."
  bad=$((bad+1))
else
  echo "  ok    no numeric tok/s inside the block"
fi

# ---- (c) every blank RESULT line is anchored to the block ------------------
n_b="$(grep -cF 'RESULT (27B TP=2' "$BACKLOG" 2>/dev/null || true)"
n_f="$(grep -cF 'RESULT (27B TP=2' "$FOLD" 2>/dev/null || true)"
n_res=$(( n_b + n_f ))
enum_require_min "result-lines" "$n_res" 6 || bad=$((bad+1))
echo "  ok    $n_res RESULT lines across both reports (expect 6)"

# ---- (d) the fold report points at the block -------------------------------
if pred_expect_present "fold-crossref" "$(cat "$FOLD")" 'ONE COMMAND FILLS'; then
  :
else
  echo "  FAIL  $FOLD does not point at the block, so its blank RESULT lines are"
  echo "        not reachable from it."
  bad=$((bad+1))
fi

# ---- (e) every RESULT stanza is still BLANK --------------------------------
# The protocol's core invariant is "a partially filled RESULT is worse than a
# blank one" -- and NOTHING here could check it, because check (b) only looks
# INSIDE the fenced block. A tok/s table written into the report BODY, ninety
# lines from the fence, therefore left this checker green. That is precisely the
# shape of the retracted-wall-time defect found in AUDIT #22, so the property is
# now checked where the numbers actually go: the body of every RESULT stanza in
# both reports.
for f in "$BACKLOG" "$FOLD"; do
  bodies="$(awk '/^RESULT \(27B TP=2/{p=1;next} p&&/^[[:space:]]*$/{p=0;next} p' "$f")"
  d="$(printf '%s\n' "$bodies" | grep -cE '[0-9]+(\.[0-9]+)?[[:space:]]*tok/s' || true)"
  if pred_expect_absent "result-stanza-blank:$(basename "$f")" "$d" '^[1-9]'; then
    echo "  ok    $(basename "$f"): no tok/s figure inside any RESULT stanza"
  else
    echo "  FAIL  $(basename "$f"): $d RESULT-stanza line(s) carry a tok/s figure."
    echo "        A partially filled RESULT is worse than a blank one."
    bad=$((bad+1))
  fi
done

# ---- (f) every arm is routed through arm_run -------------------------------
# Without this the block's arms are unlogged and ungated, which is the state
# AUDIT #22 found: a figure could be read off a terminal with nothing on disk
# to contradict it.
if pred_expect_present "arm-run-defined" "$BLOCK" '^arm_run\(\)'; then
  echo "  ok    block defines arm_run"
else
  echo "  FAIL  block does not define arm_run -- arms are unlogged and ungated"
  bad=$((bad+1))
fi
n_bare="$(printf '%s\n' "$BLOCK" | grep -cE '^[[:space:]]*(\$BIN|env INTFER_).*\|' || true)"
if pred_expect_absent "no-bare-piped-arm" "$n_bare" '^[1-9]'; then
  echo "  ok    no arm pipes \$BIN straight into grep (all route via arm_run)"
else
  echo "  FAIL  $n_bare arm(s) still pipe \$BIN into a pipe: unlogged, ungated"
  bad=$((bad+1))
fi
n_arms="$(printf '%s\n' "$BLOCK" | grep -cE '^[[:space:]]*arm_run ' || true)"
enum_require_min "arm-run-callsites" "$n_arms" 4 || bad=$((bad+1))
echo "  ok    $n_arms arm_run call sites (E1, E3, prefill, E2)"

# ---- (g) arm_run persists the arm's own output and load trace --------------
# Family 3 (stale/absent artifact) in its purest form: a measurement with no
# artifact is a claim with nothing behind it.
if pred_expect_present "per-arm-log" "$BLOCK" '> "\$log" 2>&1'; then
  echo "  ok    arm_run tees each arm's raw output to a per-arm log"
else
  echo "  FAIL  arm_run does not persist the arm's output"
  bad=$((bad+1))
fi
if pred_expect_present "per-arm-loadtrace" "$BLOCK" '>> "\$tr"'; then
  echo "  ok    arm_run records a per-arm load trace"
else
  echo "  FAIL  arm_run does not record a load trace"
  bad=$((bad+1))
fi

# ---- (h) the load criterion is the MAX over the arm, not the start ----------
if pred_expect_present "load-max-criterion" "$BLOCK" 'sort -g "\$tr" \| tail -1'; then
  echo "  ok    the verdict uses the maximum sampled load over the arm"
else
  echo "  FAIL  no max-over-arm load criterion (a start-of-round check certified"
  echo "        nothing: this box went load1 1.94 -> 10.69 mid-measurement)"
  bad=$((bad+1))
fi
if pred_expect_present "contended-not-quotable" "$BLOCK" 'CONTENDED -- NOT quotable'; then
  echo "  ok    a contended arm's figure is tagged, never emitted bare"
else
  echo "  FAIL  a contended arm can still be copied off the terminal as a figure"
  bad=$((bad+1))
fi

# ---- (i) BEHAVIOURAL: arm_run actually rejects a contended arm -------------
# (f)-(h) are TEXT checks: they prove the words are present, which is exactly how
# a retracted number got into a report. This RUNS the function the block defines,
# against a fixture loadavg, in BOTH directions -- so a gate that can only ever
# say USABLE, the bug that let LOAD1=4.67 be recorded as a figure, cannot pass.
# The fixture is why this is not a flaky gate: it does not depend on whether the
# box happens to be busy right now (which is why arm_run reads $LOADAVG).
TDIR="$(mktemp -d "${TMPDIR:-/tmp}/rbtest.XXXXXX")"
trap 'rm -rf "$TDIR"' EXIT
FN="$(printf '%s\n' "$BLOCK" | awk '/^arm_run\(\) \{/{p=1} p{print} p&&/^\}$/{exit}')"
if ! enum_require_nonempty "arm-run-extracted" "$(printf '%s\n' "$FN" | grep -c . || true)" \
     "lines of arm_run extracted from the block"; then
  echo "  FAIL  arm_run is not extractable from the block, so it cannot be tested"
  bad=$((bad+1))
else
  run_arm() { # $1 = initial fixture load1, $2 = QLOAD, $3 = tag, $4 = load to raise to ("" = hold)
    local ld="$TDIR/load_$3"
    printf '%s\n' "$1" > "$ld"
    ( LOADAVG="$ld"; QLOAD="$2"; LOGDIR="$TDIR/out_$3"; mkdir -p "$LOGDIR"
      eval "$FN"
      # A real arm is minutes long and the box drifts under it. The fixture makes
      # that drift REPRODUCIBLE: the arm starts quiet, and $4 (if given) is the
      # load it climbs to one second in. A gate that reads only the FIRST sample
      # therefore reports USABLE here -- which is precisely how LOAD1=1.94 became
      # a figure on an arm that finished at load1 10.69.
      #
      # RAISE must be a VARIABLE, not "$4": arm_run invokes the command as
      # "$@" after shifting its own label away, so inside fake_measure "$4" is
      # fake_measure's fourth argument -- always unset. An earlier draft read
      # "${4:-}" there, the fixture never rose, and the case passed VACUOUSLY:
      # it could not have detected the defect it was written for.
      RAISE="${4:-}"
      fake_measure() {
        echo '[intfer] decode: 65 tokens in 23000.0 ms (2.83 tok/s)'
        sleep 1
        [ -n "$RAISE" ] && printf '%s\n' "$RAISE" > "$ld"
        sleep 2
      }
      arm_run "arm_$3" fake_measure
    ) 2>&1
  }
  # C1: quiet at the start, CONTENDED by the end -> must be rejected.
  out_c="$(run_arm 0.5 2 contended 7.5)"
  # C2: quiet throughout -> must be accepted (the gate is not always-red).
  out_u="$(run_arm 0.5 2 usable '')"
  # C3: contended throughout -> rejected, and the figure is tagged.
  out_c3="$(run_arm 7.5 2 contended3 '')"

  if pred_expect_present "arm-gate-rejects" "$out_c" 'CONTENDED'; then
    echo "  ok    behavioural: quiet at start, load1 7.5 by the end -> arm REJECTED"
  else
    echo "  FAIL  behavioural: an arm that started quiet and finished at load1 7.5 was"
    echo "        NOT rejected. This is the exact defect that recorded a figure"
    echo "        beside a log reading LOAD1=4.67."
    bad=$((bad+1))
  fi
  if pred_expect_present "arm-gate-accepts" "$out_u" 'USABLE'; then
    echo "  ok    behavioural: quiet throughout -> arm ACCEPTED (not always-red)"
  else
    echo "  FAIL  a genuinely quiet arm was also rejected -- the gate cannot say"
    echo "        USABLE, so USABLE proves nothing and every real run is discarded."
    bad=$((bad+1))
  fi
  n_bare_c="$(printf '%s\n' "$out_c3" | grep -cE '^\[intfer\] decode' || true)"
  if pred_expect_present "contended-verdict" "$out_c3" 'CONTENDED' \
     && pred_expect_absent "contended-line-tagged" "$n_bare_c" '^[1-9]'; then
    echo "  ok    behavioural: a rejected arm's tok/s line is never emitted bare"
  else
    echo "  FAIL  a rejected arm printed its tok/s line untagged -- copyable as a figure"
    bad=$((bad+1))
  fi
  if pred_expect_present "arm-gate-logs" "$out_u" '\.log'; then
    echo "  ok    behavioural: the accepted arm wrote a per-arm log"
  else
    echo "  FAIL  the accepted arm wrote no log"
    bad=$((bad+1))
  fi
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "RESULT-BLOCK CHECK: PASS ($n_res RESULT lines reachable and blank, flags resolve,"
  echo "  no tok/s in the block or any RESULT stanza, all arms logged + load-gated)"
  echo "RESULT: the benchmark block still names only real flags, logs every arm, and claims no tok/s"
  exit 0
fi
echo "RESULT-BLOCK CHECK: FAIL ($bad property/properties drifted)"
echo "RESULT: the RESULT-filling block has drifted from what the source supports"
exit 1