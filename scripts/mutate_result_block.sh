#!/usr/bin/env bash
# Mutation matrix for scripts/check_result_block.sh.
#
# *** THE MODEL IS AUDIT #21, APPLIED TO THE ARTIFACT. *** The reports are
# CORRECT, so weakening the CHECKER is unobservable -- it would keep passing.
# Each state therefore damages the REPORTS, in a sandbox, and requires the
# checker to reject the damage. The reverse (mutate the checker, expect a
# failure) is the trap that cost this repo three matrices.
#
# The checker resolves paths from ITS OWN location, so the sandbox must mirror
# scripts/ -- a bare copy makes ROOT resolve elsewhere and enumerates nothing.
# That missing-sibling class has cost four detours here; it is handled once, at
# the top, rather than per state.
#
# Usage: scripts/mutate_result_block.sh
# Exit: 0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_patch.sh
. "$SCRIPT_DIR/_gate_patch.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"

BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/result_block_mutate"
LOG="$STAMP/states"
CHECK="$SCRIPT_DIR/check_result_block.sh"

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
git -C "$ROOT" diff --quiet -- scripts/check_result_block.sh 2>/dev/null || {
  echo "FAIL: the checker is dirty; this matrix edits a sandbox copy."; exit 2; }

SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_rbm.XXXXXX")"
TGT="$SB/scripts/check_result_block.sh"
mkdir -p "$SB/scripts" "$SB/reports" "$LOG"
: > "$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT INT TERM
for f in "$SCRIPT_DIR"/*.sh; do ln -sf "$f" "$SB/scripts/$(basename "$f")"; done
# The checker resolves the flag lookup against ROOT/src and ROOT/examples, so
# those trees must be reachable from the sandbox too. Without them EVERY flag
# reports "NOT FOUND" and the baseline fails for an artefact reason -- the
# missing-sibling class again, fourth occurrence.
ln -sfn "$ROOT/src" "$SB/src"
ln -sfn "$ROOT/examples" "$SB/examples"
rm -f "$TGT"; cp "$CHECK" "$TGT"
for r in 2026-09-30-pending-bench-backlog.md 2026-09-30-dpas48-fold.md; do
  cp "$ROOT/reports/$r" "$SB/reports/$r"
done
ORIG_B="$SB/.orig_b"; ORIG_F="$SB/.orig_f"; ORIG_C="$SB/.orig_c"
cp "$SB/reports/2026-09-30-pending-bench-backlog.md" "$ORIG_B"
cp "$SB/reports/2026-09-30-dpas48-fold.md" "$ORIG_F"
cp "$TGT" "$ORIG_C"
# *** RESTORE THE DIRECTORIES, NOT A LIST OF FILES. ***
# This had a 3-file restore list (two reports + the checker) while carrying NINE
# states, five of which were added later. That is the exact shape that leaked
# state twice in mutate_bench_load_gate.sh -- and the fix there was to add the
# missing file to the list, which guaranteed the next round. A list is a claim
# about which files a state's damage lands in; a snapshot needs no such knowledge.
# mutate_bench_load_gate.sh leaked B-E into B-H and then B-F into B-I for exactly
# this reason, both times surfacing as "anchor found 0 times" on an anchor that
# was correct in the real tree.
PRISTINE="$SB/.pristine"
mkdir -p "$PRISTINE"
cp -a "$SB/scripts" "$PRISTINE/scripts"
cp -a "$SB/reports" "$PRISTINE/reports"
restore() { rm -rf "$SB/scripts" "$SB/reports"
            cp -a "$PRISTINE/scripts" "$SB/scripts"
            cp -a "$PRISTINE/reports" "$SB/reports"
            TGT="$SB/scripts/check_result_block.sh"; }
run_check() { ( cd "$SB" && bash "$TGT" 2>&1 ); }

echo "== stamp-population leak assertion (AS FOUND) =="
pop_assert_no_leak "before" || exit 1

echo
echo "== baseline =="
restore
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL:"; printf '%s\n' "$b" | tail -6; exit 1
fi
n_ok="$(printf '%s\n' "$b" | grep -c '^  ok' || true)"
[ "$n_ok" -ge 10 ] || { echo "BASELINE FAIL -- only $n_ok ok lines"; exit 1; }
echo "  ok    baseline passes with $n_ok ok lines"
echo "  ok    CONVERSE: the healthy reports are verified, so an always-red"
echo "        checker cannot score as caught"

caught=0; total=0
run_state() { # $1 id, $2 marker-in-checker, $3 expected-RE, $4 why
  local id="$1" marker="$2" expect="$3" why="$4"
  total=$((total+1)); restore
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation fn failed"; echo "$id MISSED (fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  bash -n "$TGT" 2>/dev/null || {
    echo "  $id MISSED -- checker left broken (STATE fault)"; restore; return 1; }
  local o r; o="$(run_check)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still passed with the damage ($why)"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  # PRESENCE. These states damage the ARTIFACT and prove the checker REJECTS it,
  # so the evidence is the checker PRINTING the failure. An earlier draft used
  # pred_expect_absent here, which scores a working checker as MISSED -- the
  # seventh instance of that polarity error, and the same one three times in the
  # last four matrices.
  if ! pred_expect_present "$id property" "$o" "$expect"; then
    echo "$id MISSED (expected the rejection marker to be PRESENT)" \
      >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-present ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states (each damages the ARTIFACT, not the checker) =="

# R-A: the block sets a flag that exists nowhere -- the silent failure where a
# rename leaves the benchmark measuring the BASE arm three times.
MUTATE_FN() {
  patch_literal "$SB/reports/2026-09-30-pending-bench-backlog.md" \
    'export INTFER_KV_CAP=8192     # default 1024 aborts the long prompt' \
    'export INTFER_KV_CAP=8192; export INTFER_FFRAGE_FLAG=1  # R_A planted'
  grep -q 'FFRAGE_FLAG' "$SB/reports/2026-09-30-pending-bench-backlog.md"
}
run_state "R-A" 'FFRAGE_FLAG' 'NOT FOUND' "a planted flag that exists in no source"

# R-B: a numeric tok/s pasted into the block -- the retracted-wall-time class,
# and the one property a reader is most likely to trust at a glance.
MUTATE_FN() {
  patch_literal "$SB/reports/2026-09-30-pending-bench-backlog.md" \
    'require_quiet "E2 long-ctx"' \
    'require_quiet "E2 long-ctx"   # previously measured 3.42 tok/s  # R_B'
  grep -q 'R_B' "$SB/reports/2026-09-30-pending-bench-backlog.md"
}
run_state "R-B" 'R_B' 'a numeric tok/s appears inside the block' "a fabricated tok/s inside the block"

# R-C: the fold report's cross-reference removed, orphaning its two blank lines.
MUTATE_FN() {
  patch_literal "$SB/reports/2026-09-30-dpas48-fold.md" \
    '> `reports/2026-09-30-pending-bench-backlog.md`, section
> "ONE COMMAND FILLS EVERY `RESULT:` LINE ABOVE". Interleaved, 3 rounds, each
> round gated on `load1 < 2` and the load printed above it.' \
    '> (cross-reference removed)  # R_C'
  grep -q 'R_C' "$SB/reports/2026-09-30-dpas48-fold.md"
}
run_state "R-C" 'R_C' 'does not point at the block' "the fold report's pointer to the block deleted"

# R-D: a RESULT line removed, so part of the batch is no longer reachable.
MUTATE_FN() {
  patch_literal "$SB/reports/2026-09-30-dpas48-fold.md" \
    'RESULT (27B TP=2 prefill, chunk361):' \
    'FILLED (27B TP=2 prefill, chunk361): -- removed  # R_D'
  grep -q 'R_D' "$SB/reports/2026-09-30-dpas48-fold.md"
}
run_state "R-D" 'R_D' 'RESULT lines across both reports' "a RESULT line deleted (5 left, min is 6)"

# ---------------------------------------------------------------------------
# AUDIT #22 states. These damage the properties added when the retracted
# wall-time table was found, and each has a marker NO other state can produce,
# so "caught" cannot be scored against a different state's mechanism.
# ---------------------------------------------------------------------------

# R-E: THE SHARPEST ONE. The criterion silently reverts to the FIRST sample
# while the line that computes the max is left in place, so the TEXT check (h)
# stays green and only the BEHAVIOURAL check (i) can see it. This is the AUDIT #12
# S4 shape -- no single check covers the class -- and it is the state that proves
# the behavioural test earns its keep. It is also the real defect: a round that
# passes at load1 1.94 and measures at load1 10.69.
MUTATE_FN() {
  patch_literal "$SB/reports/2026-09-30-pending-bench-backlog.md" \
    '  local mx; mx="$(sort -g "$tr" | tail -1)"' \
    '  local mx; mx="$(sort -g "$tr" | tail -1)"
  mx="$(head -1 "$tr")"   # R_E: first sample wins -- the max line is still there'
  grep -q 'R_E' "$SB/reports/2026-09-30-pending-bench-backlog.md"
}
run_state "R-E" 'R_E' 'arm-gate-rejects' \
  "verdict reverts to the FIRST load sample; only the behavioural check can see it"

# R-F: the gate can no longer say CONTENDED at all. The always-red converse, and
# the direction that matters: a gate that always says USABLE is how a load1-4.67
# arm gets recorded as a figure. Distinct from R-E because C3 (contended
# throughout) also flips, so `contended-verdict` is unique to this state.
#
# The anchor holds BOTH quote characters, so it is built with a QUOTED HEREDOC.
# Hand-escaping it inside single quotes is the recorded MUTATION-ANCHOR WRITING
# RULE violation and it bites immediately: it produced a syntax error in THIS
# file, one level out from the state it was patching.
MUTATE_FN() {
  local a b
  read -r -d '' a <<'EOF' || true
  if awk -v m="$mx" 'BEGIN{exit !(m < '"$QLOAD"')}'; then
EOF
  read -r -d '' b <<'EOF' || true
  if true; then   # R_F: never rejects
EOF
  patch_literal "$SB/reports/2026-09-30-pending-bench-backlog.md" "$a" "$b"
  grep -q 'R_F' "$SB/reports/2026-09-30-pending-bench-backlog.md"
}
run_state "R-F" 'R_F' 'contended-verdict' \
  "the gate can never reject an arm (always-red in the USABLE direction)"

# R-G: the arm stops writing its own log -- family 3 (no artifact) returning.
MUTATE_FN() {
  patch_literal "$SB/reports/2026-09-30-pending-bench-backlog.md" \
    '  "$@" > "$log" 2>&1' \
    '  "$@" > /dev/null 2>&1   # R_G: no per-arm log'
  grep -q 'R_G' "$SB/reports/2026-09-30-pending-bench-backlog.md"
}
run_state "R-G" 'R_G' 'per-arm-log' "the arm no longer persists its own output"

# R-H: *** THE DEFECT THIS AUDIT FOUND, REPRODUCED. *** A tok/s pasted into a
# RESULT stanza in the report BODY. The pre-existing check (b) only looks inside
# the fenced block, so this passed the old checker green -- which is exactly how
# the retracted table got written. If this state is not caught, the fix is
# theatre.
MUTATE_FN() {
  patch_literal "$SB/reports/2026-09-30-pending-bench-backlog.md" \
    'RESULT (27B TP=2 decode, 3 rounds, load1 at each round):' \
    'RESULT (27B TP=2 decode, 3 rounds, load1 at each round):
  run A 2.78 tok/s, run B 2.81 tok/s, run C 2.79 tok/s   # R_H'
  grep -q 'R_H' "$SB/reports/2026-09-30-pending-bench-backlog.md"
}
run_state "R-H" 'R_H' 'carry a tok/s figure' \
  "a tok/s pasted into a RESULT stanza -- the retracted-wall-time defect itself"

# R-I: an arm goes back to piping \$BIN straight into grep: unlogged, ungated.
MUTATE_FN() {
  local a b
  read -r -d '' a <<'EOF' || true
  arm_run "E3_${arm}" \
    $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 64 --no-stream $TOK
EOF
  read -r -d '' b <<'EOF' || true
  $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 64 --no-stream $TOK 2>&1 \
    | grep -aE 'decode: '   # R_I: back to an unlogged, ungated arm
EOF
  patch_literal "$SB/reports/2026-09-30-pending-bench-backlog.md" "$a" "$b"
  grep -q 'R_I' "$SB/reports/2026-09-30-pending-bench-backlog.md"
}
run_state "R-I" 'R_I' 'unlogged and ungated' "an arm reverted to a bare pipe"

restore
echo
echo "== residue + restore integrity =="
# The residue scan must look at the SANDBOX as a state leaves it, and the last
# state's damage is still present at that point -- so scan the REPORT copies
# only after RESTORING them, which is what proves restore() is complete. An
# earlier version scanned a fully restored tree's siblings and the pristine
# backups, and flagged the state's own marker in both.
restore
git -C "$ROOT" diff --quiet -- scripts/check_result_block.sh 2>/dev/null \
  && echo "  ok    the tracked checker was never modified" || { echo "  FAIL tracked modified"; exit 1; }
pop_assert_no_leak "after" || exit 1

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "RESULT-BLOCK MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=result_block_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "RESULT-BLOCK MUTATE: FAIL ($caught/$total caught)"
exit 1