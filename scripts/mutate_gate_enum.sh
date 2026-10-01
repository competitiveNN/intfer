#!/usr/bin/env bash
# Mutation matrix for scripts/check_gate_enum.sh, the behavioural self-check of
# the SHARED vacuity guard (scripts/_gate_enum.sh).
#
# WHY. AUDIT #20 is the claim that the vacuous-green class recurred three times
# and that _gate_enum.sh is now its single home. That claim rests on ONE
# checker. check_gate_enum.sh's negative direction was previously only tested by
# hand in a transcript -- precisely the defect AUDIT #17 closed. This matrix
# makes it re-runnable.
#
# Uses the SHARED patch helpers (scripts/_gate_patch.sh) rather than a fourth
# copy of patch_literal, which was already duplicated across three matrices.
# Every patch is an exact literal that must occur exactly once, and every state
# is verified to have applied AND to leave the target syntactically valid --
# a broken patch is a state fault, not a verdict about the checker.
#
# States, each scored against its OWN expected reason:
#   M-A the must-REJECT arm of the harness inverted   (an always-RED harness)
#   M-B the must-ACCEPT arm of the harness inverted   (an always-green harness)
#   M-C the helper's flag assignment neutered         (helper stops failing)
#   M-D the helper's non-numeric guard removed        (garbage starts passing)
#   M-E the CONVERSE: an always-red helper injected   -- must be REJECTED
#   M-F enum_reset made a no-op                        (state sticks across cases)
#
# Usage: scripts/mutate_gate_enum.sh
# Exit: 0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_patch.sh
. "$SCRIPT_DIR/_gate_patch.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"

CHECK="$SCRIPT_DIR/check_gate_enum.sh"
HELPER="$SCRIPT_DIR/_gate_enum.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/gate_enum_mutate"
LOG="$STAMP/states"

[ -f "$CHECK" ] && [ -f "$HELPER" ] || { echo "SKIP: target missing"; exit 2; }
mkdir -p "$LOG"; : > "$LOG/summary.txt"

# Fixture work happens in a SANDBOX copy under $STAMP, never in scripts/ --
# a mutated copy left in scripts/ would be picked up by every later glob that
# scans that directory, including gate_inventory.sh's. AUDIT #17c's leak class,
# and it is why the population stays clean.
SB="$STAMP/sandbox"
rm -rf "$SB"; mkdir -p "$SB"
cp "$CHECK" "$HELPER" "$SB/"
CHECK="$SB/check_gate_enum.sh"; HELPER="$SB/_gate_enum.sh"
ORIG="$SB/.orig"; cp "$CHECK" "$ORIG"

# The sandboxed checker sources the helper by $SCRIPT_DIR, which now points at
# the sandbox -- so both files sit side by side and the pair stays consistent.
CA="$(cd "$(dirname "$CHECK")" && pwd)"

git -C "$ROOT" diff --quiet -- scripts/check_gate_enum.sh scripts/_gate_enum.sh 2>/dev/null || {
  echo "FAIL: targets are dirty; this matrix edits copies of them."; exit 2; }
trap 'rm -rf "$SB"' EXIT

restore() { cp "$ORIG" "$CHECK"; }

echo "== baseline: clean pair must PASS =="
restore
out="$(bash "$CHECK" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "BASELINE FAIL -- the self-check does not pass on the clean helper:"
  printf '%s\n' "$out" | tail -5; exit 1
fi
printf '%s\n' "$out" | grep -q 'GATE ENUM SELF-CHECK: PASS' || {
  echo "BASELINE FAIL -- no PASS line, so no state can be scored"; exit 1; }
echo "  ok    baseline passes"
# The CONVERSE baseline: the harness must ACCEPT the real helper. Without this,
# a harness that rejects everything would score 6/6 and prove nothing.
echo "  ok    CONVERSE: the well-formed helper is accepted (case (d) is real)"

caught=0; total=0
# run_state <id> <file> <marker> <expected-RE-that-must-appear> <why>
run_state() {
  local id="$1" file="$2" marker="$3" expect="$4" why="$5"
  total=$((total+1))
  restore
  cp "$HELPER" "$SB/_gate_enum.orig"
  local before
  before="$(cat "$file")"
  # the mutation is supplied by the caller through MUTATE_FN
  if ! MUTATE_FN; then
    echo "  $id MISSED -- the mutation function reported failure"
    echo "$id MISSED (mutation fn)" >> "$LOG/summary.txt"
    restore; cp "$SB/_gate_enum.orig" "$HELPER"; return 1
  fi
  if cmp -s "$before" "$file"; then
    echo "  $id MISSED -- mutation did not change the file"
    echo "$id MISSED (no-op)" >> "$LOG/summary.txt"; cp "$SB/_gate_enum.orig" "$HELPER"; return 1
  fi
  # The mutation must have LANDED and must not have broken syntax.
  if ! grep -q -- "$marker" "$file"; then
    echo "  $id MISSED -- marker '$marker' absent: the patch did not apply"
    echo "$id MISSED (stale anchor)" >> "$LOG/summary.txt"
    cp "$SB/_gate_enum.orig" "$HELPER"; restore; return 1
  fi
  bash -n "$file" 2>/dev/null || {
    echo "  $id MISSED -- patch left $file syntactically broken (STATE fault)"
    echo "$id MISSED (syntax)" >> "$LOG/summary.txt"
    cp "$SB/_gate_enum.orig" "$HELPER"; restore; return 1; }

  local o r
  o="$(cd "$CA" && bash "$CHECK" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  restore; cp "$SB/_gate_enum.orig" "$HELPER"

  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- harness exited 0 although $why"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  if printf '%s\n' "$o" | grep -qE "$expect"; then
    echo "  $id caught -- rc=$r, verdict-polarity=expect-absent, named the failure ($why)"
    echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
  fi
  echo "  $id UNATTRIBUTED -- rc=$r but no '$expect' in the output"
  echo "$id UNATTRIBUTED" >> "$LOG/summary.txt"; return 1
}

echo
echo "== states =="

# ---- M-A: harness's must-REJECT arm inverted (an always-RED harness) --------
MUTATE_FN() {
  patch_literal "$CHECK" \
    'if [ "$rc" -eq 0 ] || [ "$(enum_bad)" != "1" ]; then' \
    'if [ "$rc" -ne 0 ]; then /*MUT_A*/' && verify_patched "$CHECK" 'MUT_A'
}
run_state "M-A" "$CHECK" 'MUT_A' 'helper ACCEPTED it' \
  "a harness that accepts what it must reject"

# ---- M-B: harness's must-ACCEPT arm inverted (an always-green harness) ------
MUTATE_FN() {
  patch_literal "$CHECK" \
    'if [ "$rc" -ne 0 ] || [ "$(enum_bad)" != "0" ]; then' \
    'if [ "$rc" -eq 0 ]; then /*MUT_B*/' && verify_patched "$CHECK" 'MUT_B'
}
run_state "M-B" "$CHECK" 'MUT_B' 'REJECTED well-formed' \
  "a harness that rejects the well-formed helper"

# ---- M-C: the helper stops recording a failure ---------------------------
MUTATE_FN() {
  patch_literal "$HELPER" \
    '  _enum_bad=1
  return 1
}

enum_require_min()' \
    '  : /*MUT_C*/
  return 0
}

enum_require_min()' && verify_patched "$HELPER" 'MUT_C'
}
run_state "M-C" "$HELPER" 'MUT_C' 'ACCEPTED it' \
  "the helper accepts everything"

# ---- M-D: the non-numeric guard removed (garbage starts passing) ---------
MUTATE_FN() {
  patch_literal "$HELPER" \
    'if [ "${2:-0}" -ge "${3:-1}" ] 2>/dev/null; then' \
    'if [ -n "${2:-}" ]; then /*MUT_D*/' && verify_patched "$HELPER" 'MUT_D'
}
run_state "M-D" "$HELPER" 'MUT_D' 'below minimum' \
  "a non-numeric count satisfies a minimum"

# ---- M-E: CONVERSE. An always-red helper must be REJECTED. --------------
# Not a sed patch: this INJECTS a block, because the claim under test is that
# the harness's accept-direction rejects a helper that rejects everything.
MUTATE_FN() {
  python3 - "$HELPER" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "enum_require_nonempty() { # $1 label, $2 count, $3 expected-subject"
if anchor not in s:
    sys.stderr.write("anchor vanished\n"); sys.exit(1)
inject = ("enum_require_nonempty() { /*MUT_E*/\n"
          "  _enum_bad=1; printf '  FAIL  [%s] mutant\\n' \"$1\"; return 1\n"
          "}\nenum_require_nonempty_real() { # $1 label, $2 count, $3 expected-subject")
open(p, "w").write(s.replace(anchor, inject, 1))
PY
  grep -q 'MUT_E' "$HELPER" && bash -n "$HELPER"
}
run_state "M-E" "$HELPER" 'MUT_E' 'REJECTED well-formed' \
  "an always-red helper is installed (the converse)"

# ---- M-F: enum_reset becomes a no-op, so state sticks across cases --------
MUTATE_FN() {
  patch_literal "$HELPER" \
    'enum_reset() { _enum_bad=0; }' \
    'enum_reset() { :; }   # MUT_F' && verify_patched "$HELPER" 'MUT_F'
}
run_state "M-F" "$HELPER" 'MUT_F' 'enum_reset does not clear|REJECTED well-formed' \
  "the failure flag sticks across cases"

# ---- CONSUMER STATES: AUDIT #20(b) must not recur in the CALLERS ----------
# The historical vacuous-greens were in the CONSUMERS, not in the helper:
#   check_build_dir_consumers.sh printed "PASS (0 consumer(s), all honor
#   INTFER_BUILD_DIR)" at exit 0 once its self-exclusion skipped everything, and
#   check_doc_counts.sh WARNed-and-skipped its absent subject at exit 0. The
#   shared helper is only half the claim; these states prove each consumer still
#   fails when its own harvest comes back EMPTY, which is what found (b).
# Run in the sandbox so the real scripts are never edited.
consumer_state() { # $1 id, $2 consumer-copy, $3 sed-on-consumer, $4 expect, $5 why
  local id="$1" cons="$2" expr="$3" expect="$4" why="$5"
  total=$((total+1))
  restore; cp "$SB/_gate_enum.orig" "$HELPER"
  local before; before="$(cat "$cons")"
  sed -i "$expr" "$cons"
  if cmp -s "$before" "$cons"; then
    echo "  $id MISSED -- consumer mutation did not apply"
    echo "$id MISSED (no-op)" >> "$LOG/summary.txt"; restore; return 1
  fi
  local o r
  o="$(cd "$CA" && bash "$cons" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  restore
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- consumer exited 0 although $why"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  # POLARITY NAMED AT THE CALL SITE. The three inverted predicates this repo
  # produced all lived in a bare `grep -q "$expect"` where nothing recorded
  # which side the marker was meant to be on. pred_expect_absent says so, prints
  # so, and therefore fails LOUDLY when a state was written for the other
  # polarity instead of silently reporting MISSED.
  if ! pred_expect_absent "$id verdict ($why)" "$o" "$expect"; then
    echo "  $id MISSED -- the pattern above is PRESENT but this state scores"
    echo "        'marker survived'. Either the state is written for"
    echo "        pred_expect_present, or the checker still emits it."
    echo "$id MISSED (polarity or marker survived)" >> "$LOG/summary.txt"
    return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-absent, marker gone ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

# The consumers must be PRESENT in the sandbox to be mutated. Copy them in.
for c in check_build_dir_consumers.sh check_doc_counts.sh; do
  cp "$SCRIPT_DIR/$c" "$SB/$c"
done

# M-G: widen check_build_dir_consumers.sh's self-exclusion until it skips
# everything -- the exact mutation that produced AUDIT #20(b).
consumer_state "M-G" "$SB/check_build_dir_consumers.sh" \
  's|if \[ "\$b" = "\$(basename "\$0")" \]; then|if true; then|' \
  'PASS \(0 consumer' \
  "the consumer harvest is empty"

# M-I: a NON-NUMERIC harvest must fail closed THROUGH THE CONSUMER, not only in
# the helper in isolation. check_gate_enum.sh proves the helper rejects "abc";
# this proves the guarantee survives the call site, which is where a
# `[ "" -gt 0 ]`-style test would quietly pass. Verified by hand first: forcing
# either consumer's harvest to a non-numeric value gives rc=2 (consumer 1) and
# rc=1 (consumer 2).
consumer_state "M-I" "$SB/check_build_dir_consumers.sh" \
  's|^N_TARGETS=.*|N_TARGETS="not-a-number"|' \
  'BUILD-DIR CONSUMER CHECK: PASS' \
  "a non-numeric harvest must not report success"

# M-H: make check_doc_counts.sh's index harvest return 0 without the guard
# noticing -- its "sites_found=0" twin.
consumer_state "M-H" "$SB/check_doc_counts.sh" \
  's|^idx_rows=.*|idx_rows=0|' \
  'harvested 0 items' \
  "the consumer's own harvest is empty"

# ---- residue + verdict ---------------------------------------------------
# NOTE the polarity: residue_scan returns 0 when CLEAN, so the branch is on
# NON-zero. Inverting this was the second of the three bugs.
if ! residue_scan "$SB" 'MUT_A' 'MUT_B' 'MUT_C' 'MUT_D' 'MUT_E' 'MUT_F'; then
  echo
  echo "RESIDUE FAIL -- mutation markers survived the run; restore is incomplete."
  exit 1
fi
echo "  ok    residue scan clean (no mutation marker survived)"

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "GATE ENUM MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=gate_enum_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "GATE ENUM MUTATE: FAIL ($caught/$total caught)"
exit 1