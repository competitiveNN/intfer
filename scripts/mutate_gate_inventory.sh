#!/usr/bin/env bash
# Reproducible non-vacuity proof for the GATE INVENTORY checker
# (scripts/gate_inventory.sh) -- the checker AUDIT #13 added and AUDIT #14
# found to be itself defective.
#
# WHY THIS IS A SCRIPT AND NOT A COMMIT MESSAGE. The audit-#14 evidence was first
# produced as ad-hoc inline shell: a mutation matrix printed to a console and
# narrated in a commit subject. That is the same defect the fold matrix was
# caught in -- "Prose is not evidence: the tree inherited only a PROSE TABLE" --
# one level up. A mutation that only exists in a transcript cannot be re-run by
# the next session, so the next session re-derives (or worse, re-asserts) it. So:
# the matrix lives in the tree, writes per-state logs, and stamps the sha it was
# proven against. Same shape as scripts/mutate_fold_matrix.sh, deliberately.
#
# WHAT IT ASSERTS. Each mutation breaks ONE mechanism in the checker, and the
# checker must then REFUSE TO PASS -- non-zero exit, with a recognisable
# diagnostic. A mutation that still exits 0 is scored MISSED and fails the run. A
# mutation that breaks the checker's SYNTAX is ALSO scored MISSED, because a
# syntax error is not evidence that the mechanism is load-bearing.
#
#   M1 un-anchor the line-start require test
#      The line-start test separates a real whole-gate skip statement from an
#      unplaceable mention. Un-anchoring must be observable, and it is ONLY
#      observable through the BROKEN self-check case (an audit-#14 addition: a
#      `gate_require_gpus` wrapped in a helper function). With comment-stripping
#      in place this is an EQUIVALENT mutation on the real gates -- measured --
#      so without that case it would silently pass and the anchoring would be
#      untested decoration. This is the load-bearing row: it is the proof that
#      the anchoring is tested rather than merely present.
#   M2 drop comment stripping
#      The gates' own rationale comments quote the tokens being classified, so
#      reading the raw file labels test_device_argmax.sh's PRECONDITION column
#      from prose instead of code. Must be caught.
#   M3 reintroduce the stdin-consuming grep chain
#      `grep -q` exits on its verdict and consumes the stream, so a second
#      `grep -q` in the elif sees nothing and every gate reads "none". Not
#      hypothetical: the classifier's own first draft had exactly this bug and
#      its self-check caught it on the first run.
#   M4 invert a self-check expectation
#      A self-check whose expectation is wrong must not pass. Proves the
#      expectations are load-bearing rather than decorative.
#   M5 disable the per-arm NO-SKIP-REPORT check
#      A per-arm guard that never records what it skipped is a silent skip
#      wearing a per-arm hat. Must be caught.
#   M7 disable the NEEDS-GPU-without-a-guard check
#      GRANULARITY conflates "needs no device" with "needs a device and forgot to
#      say so". Both classify as "none" and both exit 0 on a healthy box, so the
#      gap is invisible until the GPUs go away -- and then the second kind goes
#      RED. Found by RE-DERIVING the gate list instead of remembering it:
#      test_dpas_fold.sh (which said "needs a GPU" in a COMMENT and carried no
#      guard), test_http_batch.sh and test_serve_load.sh (both launch the server
#      `--gpu --tp 1`). All three PASS on a 1-GPU box, which is exactly why the
#      gap survived. They now carry a render-node guard, and this state is what
#      stops the class reopening.
#   M9 strip the render-node guard from a GPU gate (target = a GATE, not the
#      checker)
#      Audit #15 added gate_require_device to test_dpas_fold /
#      test_http_batch / test_serve_load. Their end-to-end proof was run once,
#      by hand, and the result lived in a transcript -- which is the defect this
#      whole matrix exists to stop. Here it is a state: remove the guard, and the
#      RENDER-NODE column MUST report MISSING and the checker MUST exit 1. Proves
#      both that the guard is load-bearing and that the column is live.
#   M8 make the doc-tally parser comma-only again
#      The parser only understood ", N PARTIAL". Live prose also uses the slash
#      form ("7 PASS / 2 PARTIAL / 1 SKIPPED"), which the site-finder DETECTED
#      and the parser then mis-read as 7/0/0 -- so a CORRECT line was reported
#      STALE. A site the checker finds but cannot parse is worse than not finding
#      it: it cries wolf on the line it was written to protect. Observed live
#      against this file's own new audit-#15 paragraph.
#   M6 let a 0-GPU box exit 0 again
#      The defect audit #14 actually shipped: gate_inventory did not source
#      setvars, so sycl-ls was absent, the doc-tally skipped, and the script
#      still exited 0. Simulated with a stub sycl-ls reporting no device, so it
#      needs no special hardware. Must exit 2, never 0.
#
# M1-M5 are CAUGHT by the checker's own self-check, which exits 1 before printing
# the table. M6 is scored on the checker's exit code alone, because a checker
# cannot detect its own soft-skip from the inside.
#
# SAFETY. This script EDITS ONE TRACKED FILE (scripts/gate_inventory.sh) and
# restores it with `git checkout --`, so it REFUSES TO START on a dirty tree and
# traps EXIT to restore. It re-verifies at the end that the file is byte-identical
# to HEAD. It touches no engine, quant or schedule code.
#
# Usage:  scripts/mutate_gate_inventory.sh
# Env:    INTFER_BUILD_DIR  (default <root>/build) -- log dir
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

TGT=scripts/gate_inventory.sh
# Second class of target: a GATE rather than the checker. Audit #15's end-to-end
# proof (seed an unguarded GPU gate, watch the RENDER-NODE column catch it) lived
# only in a transcript -- the exact defect #13/#14 exist to prevent, one level up
# from the prose-table problem. M9 below makes it a re-runnable state.
TGT2=scripts/test_dpas_fold.sh
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/gate_inventory_matrix"
SUMMARY="$LOG_DIR/summary.txt"
STAMP="$LOG_DIR/last_pass.stamp"

# --- fail closed on a dirty tree: we are about to `git checkout --` it --------
if ! git diff --quiet -- "$TGT" "$TGT2" || ! git diff --cached --quiet -- "$TGT" "$TGT2"; then
  echo "FAIL: $TGT or $TGT2 has uncommitted changes." >&2
  echo "  This script edits it and restores with 'git checkout --', which would" >&2
  echo "  DESTROY that work. Commit or stash first, then re-run." >&2
  exit 1
fi

# --- ANCHOR PRE-FLIGHT -------------------------------------------------------
# A mutation harness carries artifact assumptions about the code it mutates, and
# a refactor invalidates them silently: the replacement then matches nothing, the
# file is unchanged, the checker passes, and the state reads AS EXPECTED for the
# wrong reason. Every target is asserted to occur EXACTLY ONCE up front, so a
# moved/duplicated/vanished anchor is a hard stop before any state runs. (Same
# rationale as mutate_fold_matrix.sh's preflight, which exists because the index
# refactor deleted a mutation's target text outright.)
# Patterns live in a QUOTED ARRAY, not a `pat|desc` table parsed with IFS='|'.
#
# AUDIT #16. The table used `|` as the field separator while the patterns
# themselves contain `|` -- ERE alternation ('renderD|gate_require_render_node',
# 'SKIPPED_(TP2|SUB)|PARTIAL') and `||`. Every such row was silently TRUNCATED at
# the first `|`, and because a truncated prefix is still a substring, the
# pre-flight went on reporting "found exactly once" while checking strictly LESS
# than it claimed. Four of nine anchors were affected. That is precisely the
# failure the pre-flight exists to prevent -- a loosened anchor -- arriving
# through the pre-flight's own parser.
#
# A quoted array element can contain any character, so there is nothing left to
# collide with. $1 = file to check (a parameter so the self-check can run it
# against a seeded temp file and prove it CAN fail).
ANCHOR_PATTERNS=(
  '  if grep -qE '"'"'^[[:space:]]*gate_require_gpus[[:space:]]+[0-9]'"'"' "$1"; then'
  'strip_comments() { sed -e '"'"'s/[[:space:]]#.*$//'"'"''
  'elif grep -q '"'"'gate_gpu_count'"'"' "$1"; then'
  '  [ "$p" = "per-arm" ] ||'
  'grep -qE '"'"'SKIPPED_(TP2|SUB)|PARTIAL'"'"' ||'
  'unplaceable require classified'
  'PARSE_SK="$(printf '"'"'%s'"'"' "$txt" | sed -nE '"'"'s/.*[,/] ?([0-9]+) SKIPPED.*/\1/p'"'"')"'
  'doc-tally: SKIPPED'
  'if grep -qE '"'"'renderD|gate_require_render_node|gate_require_device'"'"' "$1"; then echo "render-node"'
  'gate_require_device || exit 2'
)
# Which file each anchor lives in. M9 mutates a GATE, not the checker, so its
# anchor is in $TGT2 -- a single-file pre-flight silently checked 0 of it.
ANCHOR_FILE=(TGT TGT TGT TGT TGT TGT TGT TGT TGT TGT2)
ANCHOR_DESC=(
  "M1/M7 target: line-start gate_require_gpus test"
  "M2 target: comment stripper"
  "M3 target: the elif that M3 starves"
  "M4 target: the per-arm self-check expectation"
  "M5 target: NO-SKIP-REPORT check"
  "M7b target: BROKEN self-check case (makes M1 observable)"
  "M8 target: the slash-tolerant tally parser"
  "M6b target: the 0-GPU loud-skip branch"
  "M7 target: the NEEDS-GPU verdict line"
  "M9 target: the device guard in a GPU gate (lives in $TGT2)"
)
anchors_ok() { # $1 = file to check
  local f="${1:-$TGT}" bad=0 n i tgt
  for i in "${!ANCHOR_PATTERNS[@]}"; do
    # The optional $1 overrides the FILE for every anchor, so the self-check can
    # point the whole table at a seeded copy. Without it, an anchor belonging to
    # $TGT2 would be checked against $TGT and found zero times.
    if [ -n "${f}" ] && [ "$f" != "$TGT" ]; then tgt="$f"; else tgt="${ANCHOR_FILE[$i]}"; tgt="${!tgt}"; fi
    n="$(grep -cF -- "${ANCHOR_PATTERNS[$i]}" "$tgt" 2>/dev/null || true)"
    if [ "$n" != "1" ]; then
      echo "ANCHOR STALE: ${ANCHOR_DESC[$i]} -- found $n occurrence(s), want 1" >&2
      bad=1
    fi
  done
  if [ "$bad" != "0" ]; then
    cat >&2 <<'EOF'

  The matrix is testing code that no longer exists in the shape it expects.
  Re-point the affected mutation at the new location rather than loosening the
  check -- a loosened anchor is how a mutation silently stops mutating.
EOF
    return 1
  fi
  echo "anchors: all ${#ANCHOR_PATTERNS[@]} mutation targets found exactly once (across $TGT and $TGT2)"
  return 0
}

# --- SELF-CHECK: the conclusion this script prints must be renderable ---------
# BEHAVIOURAL, NOT TEXTUAL. An earlier self-check in this repo grepped its own
# source for the strings it prints, which made it VACUOUS: the pattern lives
# inside the check, so it can never fail. This one renders the summary and
# compares it to a literal, and proves the rc-classifier is not a constant
# function. Both can actually fail.
emit_summary() {
  echo "  M1: the line-start anchoring is OBSERVABLE (un-anchoring flips the BROKEN case)"
  echo "  M2: classification reads CODE, not the gates' own prose comments"
  echo "  M3: the classifier is not stdin-starved by its own grep chain"
  echo "  M4: the self-check expectations are load-bearing, not decorative"
  echo "  M5: a per-arm guard that never reports its skip is rejected"
  echo "  M6: a 0-GPU box is refused (exit 2), never a silent exit 0"
  echo "  M7: a GPU gate with NO guard is reported, not filed as 'needs no device'"
  echo "  M8: a slash-form doc tally is PARSED, not mis-read as 7/0/0"
  echo "  M9: a GPU gate with its device guard REMOVED is caught (target = a gate)"
  echo "  => all nine mechanisms can fail the checker; none is untested."
}
EXPECTED_SUMMARY="  M1: the line-start anchoring is OBSERVABLE (un-anchoring flips the BROKEN case)
  M2: classification reads CODE, not the gates' own prose comments
  M3: the classifier is not stdin-starved by its own grep chain
  M4: the self-check expectations are load-bearing, not decorative
  M5: a per-arm guard that never reports its skip is rejected
  M6: a 0-GPU box is refused (exit 2), never a silent exit 0
  M7: a GPU gate with NO guard is reported, not filed as 'needs no device'
  M8: a slash-form doc tally is PARSED, not mis-read as 7/0/0
  M9: a GPU gate with its device guard REMOVED is caught (target = a gate)
  => all nine mechanisms can fail the checker; none is untested."

# rc class -> 0 = MISSED (checker still passed), 1 = CAUGHT by the self-check,
# 2 = CAUGHT by the loud-skip path, anything else = CAUGHT with an odd rc.
rc_class() {
  case "$1" in
    0) echo MISSED ;;
    1) echo CAUGHT-selfcheck ;;
    2) echo CAUGHT-skip ;;
    *) echo "CAUGHT-rc$1" ;;
  esac
}
selfcheck() {
  local bad=0 c in want
  if [ "$(emit_summary)" != "$EXPECTED_SUMMARY" ]; then
    echo "SELFCHECK: the PASS summary no longer renders what it promises" >&2
    bad=1
  fi
  # The rc classifier must distinguish MISSED from every CAUGHT class, or a
  # mutation exiting 7 would score the same as one exiting 0.
  for c in "0:MISSED" "1:CAUGHT-selfcheck" "2:CAUGHT-skip" "7:CAUGHT-rc7"; do
    in="${c%%:*}"; want="${c#*:}"
    [ "$(rc_class "$in")" = "$want" ] || {
      echo "SELFCHECK: rc_class($in) != $want" >&2; bad=1; }
  done
  # ...and must not be a constant function.
  [ "$(rc_class 0)" != "$(rc_class 2)" ] || {
    echo "SELFCHECK: rc_class cannot tell a pass from a skip" >&2; bad=1; }
  # THE ANCHOR PRE-FLIGHT MUST BE ABLE TO FAIL. It shipped with a `|` field
  # separator that truncated four of its own nine patterns, and reported "found
  # exactly once" anyway -- a pre-flight that cannot fail is worse than none,
  # because it manufactures confidence. Seed a duplicated anchor into a copy of
  # the target and require the pre-flight to reject it.
  local tf
  tf="$(mktemp "${TMPDIR:-/tmp}/gi_anchor_selfcheck.XXXXXX")"
  cp "$TGT" "$tf"
  printf '%s\n' "${ANCHOR_PATTERNS[0]}" >>"$tf"
  if anchors_ok "$tf" >/dev/null 2>&1; then
    echo "SELFCHECK: the anchor pre-flight accepted a DUPLICATED anchor" >&2; bad=1
  fi
  rm -f "$tf"
  # ...and must still accept the real file.
  anchors_ok "$TGT" >/dev/null 2>&1 || {
    echo "SELFCHECK: the anchor pre-flight rejects the unmutated target" >&2; bad=1; }
  return $bad
}

selfcheck || { echo "FAIL: this script's own invariants are broken" >&2; exit 1; }
anchors_ok || exit 1

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR" || exit 1
: >"$SUMMARY"

restore() {
  git checkout -- "$TGT" "$TGT2" 2>/dev/null
  if ! git diff --quiet -- "$TGT" "$TGT2"; then
    echo "RESTORING: $TGT/$TGT2 were not clean; discarding mutation residue" >&2
    git checkout -- "$TGT" "$TGT2"
  fi
}
trap restore EXIT INT TERM

# Replace `old` with `new` in $1, from the patch file $2. Hard-fails unless every
# `old` occurs exactly once AT THE TIME IT IS APPLIED -- a zero-match or
# multi-match is a stale anchor, not a mutation.
#
# MULTIPLE HUNKS. Hunks are separated by a line containing only "@@@" and applied
# in order, each requiring its own target to be unique at the time it is applied.
#
# M3 is the reason this exists, twice over. The stdin-starvation bug needs BOTH
# greps to read a SHARED stream, so it takes two coordinated edits. The first
# M3 attempt instead patched only the elif to `- <"$1"`, which reopens the file
# for that one grep, starves nothing, and changed no verdict -- the matrix scored
# it MISSED, correctly. A mutation that does not reproduce the defect is worse
# than no mutation, because it launders a red state green. The faithful version
# opens the file once on a shared fd 3 and points both greps at it, so the first
# grep's early exit leaves the elif reading from the offset onward: the
# per-arm sample, which has no `gate_require_gpus` at all, is consumed whole by
# the first grep and the elif then sees an empty stream. Call sites are left
# untouched, so the only thing under test is the classifier.
patch_once() {
  python3 - "$1" "$2" <<'PY'
import sys
tgt, pf = sys.argv[1], sys.argv[2]
s = open(tgt).read()
lines = open(pf).read().split("\n")
# The heredoc body arrives with a trailing newline; drop exactly one so a
# single-hunk patch round-trips byte-identically.
if lines and lines[-1] == "":
    lines.pop()

# Split into hunks on lines that are exactly "@@@".
hunks, cur = [], []
for ln in lines:
    if ln == "@@@":
        hunks.append(cur); cur = []
    else:
        cur.append(ln)
hunks.append(cur)
if not any("|||" in h for h in hunks):
    sys.stderr.write("patch has no '|||' separator line\n")
    sys.exit(1)

applied = 0
for h in hunks:
    if "|||" not in h:
        if [x for x in h if x.strip()]:
            sys.stderr.write("hunk %d has no '|||' separator\n" % applied)
            sys.exit(1)
        continue
    i = h.index("|||")
    old, new = "\n".join(h[:i]), "\n".join(h[i + 1:])
    n = s.count(old)
    if n != 1:
        sys.stderr.write("hunk %d: target found %d times, want 1: %r\n" % (applied, n, old[:70]))
        sys.exit(1)
    s = s.replace(old, new, 1)
    applied += 1
if applied == 0:
    sys.stderr.write("patch contained no hunks\n")
    sys.exit(1)
open(tgt, "w").write(s)
print("applied %d hunk(s)" % applied, file=sys.stderr)
PY
}

# Apply one mutation, run the checker, score the exit code.
#   $1 = state label, $2 = description, $3 = expected CAUGHT class
#   the PATCH arrives on STDIN as a QUOTED heredoc, split by a line that is
#   exactly "|||":  <old text> ||| <new text>
#
# Why a heredoc and not a "$3" argument: the first draft passed the replacement
# as a shell string, and three of the six mutations landed as SYNTAX ERRORS
# rather than behaviour changes -- a `|||` delimiter trailing inside the string,
# and unbalanced quotes. The harness correctly scored those MISSED (a syntax
# error is not evidence a mechanism is load-bearing) rather than quietly
# counting them green, which is what kept that run honest. A quoted heredoc
# interpolates nothing, so a mutation is exactly the bytes between the markers.
FAILED=0; NC=0
run_state() {
  local label="$1" desc="$2" expect="$3"
  local log="$LOG_DIR/$label.log" rc class verdict="" pf
  pf="$(mktemp "${TMPDIR:-/tmp}/gi_patch.XXXXXX")" || return 1
  cat >"$pf"
  restore
  if ! patch_once "$TGT" "$pf"; then
    echo "  $label: PATCH ERROR (target not found exactly once; see above)" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1)); rm -f "$pf"; return
  fi
  # The mutation must be SYNTACTICALLY VALID, or the checker would fail for the
  # wrong reason and every state would look "caught" for free.
  if ! bash -n "$TGT" 2>"$log"; then
    { echo "  $label: $desc -- MUTATION BROKE SYNTAX"
      echo "    (scored MISSED: a syntax error is not evidence the mechanism is load-bearing)"; } | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1)); rm -f "$pf"; restore; return
  fi
  # Prove the mutation actually changed the file, so a green state cannot come
  # from a no-op replacement the pre-flight did not catch.
  if git diff --quiet -- "$TGT"; then
    echo "  $label: $desc -- MUTATION DID NOT CHANGE THE FILE" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1)); rm -f "$pf"; restore; return
  fi

  bash "$TGT" >"$log" 2>&1
  rc=$?
  class="$(rc_class "$rc")"
  case "$class" in
    MISSED) verdict="STILL EXITED 0" ;;
    *) [ "$class" = "$expect" ] || verdict="caught as $class, expected $expect" ;;
  esac
  NC=$((NC + 1))
  if [ -z "$verdict" ]; then
    printf '  %-4s rc=%-3s %-16s %-50s AS EXPECTED\n' "$label" "$rc" "$class" "$desc" | tee -a "$SUMMARY"
  else
    printf '  %-4s rc=%-3s %-16s %-50s UNEXPECTED: %s\n' "$label" "$rc" "$class" "$desc" "$verdict" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1))
  fi
  grep -m1 -E 'SELFCHECK|ANCHOR STALE|doc-tally: SKIPPED|PROBLEM\(S\)' "$log" \
    | sed 's/^/       > /' | tee -a "$SUMMARY" || true
  rm -f "$pf"
}

# Same discipline as run_state, but the target is a GATE and the thing that must
# notice is the CHECKER. Proves the render-node guard is LOAD-BEARING: remove it
# from a GPU gate and the RENDER-NODE column must report MISSING and the checker
# must exit 1. This is audit #15's end-to-end proof, as a committed state.
run_gate_state() {
  local label="$1" desc="$2" expect="$3"
  local log="$LOG_DIR/$label.log" rc class verdict="" pf
  pf="$(mktemp "${TMPDIR:-/tmp}/gi_patch2.XXXXXX")" || return 1
  cat >"$pf"
  restore
  if ! patch_once "$TGT2" "$pf"; then
    echo "  $label: PATCH ERROR (target not found exactly once; see above)" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1)); rm -f "$pf"; return
  fi
  if ! bash -n "$TGT2" 2>"$log"; then
    echo "  $label: $desc -- MUTATION BROKE SYNTAX (scored MISSED)" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1)); rm -f "$pf"; restore; return
  fi
  if git diff --quiet -- "$TGT2"; then
    echo "  $label: $desc -- MUTATION DID NOT CHANGE THE FILE" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1)); rm -f "$pf"; restore; return
  fi
  bash "$TGT" >"$log" 2>&1
  rc=$?
  class="$(rc_class "$rc")"
  case "$class" in
    MISSED) verdict="checker STILL EXITED 0 on an unguarded GPU gate" ;;
    *) [ "$class" = "$expect" ] || verdict="caught as $class, expected $expect" ;;
  esac
  NC=$((NC + 1))
  if [ -z "$verdict" ]; then
    printf '  %-4s rc=%-3s %-16s %-50s AS EXPECTED\n' "$label" "$rc" "$class" "$desc" | tee -a "$SUMMARY"
  else
    printf '  %-4s rc=%-3s %-16s %-50s UNEXPECTED: %s\n' "$label" "$rc" "$class" "$desc" "$verdict" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1))
  fi
  grep -m1 -E 'NEEDS A GPU BUT HAS NO GUARD|MISSING' "$log" | sed 's/^/       > /' | tee -a "$SUMMARY" || true
  rm -f "$pf"
}

echo "=== gate_inventory checker mutation matrix (non-vacuity proof) ==="
echo "tree must be clean; logs -> $LOG_DIR"

# S0: baseline. The checker must PASS on the unmutated tree, or the matrix would
# be scoring states against a checker that is already broken.
# *** S0 ACCEPTS rc=2, AND *REQUIRES* IT TO BE EXPLAINED. *** Measured 2026-10-03:
# the box dropped to ONE enumerated GPU, and the checker correctly answered
# UNPROVEN (exit 2) because the doc tally describes a two-GPU configuration this
# box cannot run. The baseline demanded rc=0 and therefore reported "the checker
# does not pass on a clean tree" -- i.e. it called a CORRECT verdict a broken
# checker, which is the always-red twin of the vacuous green.
#
# So the healthy baseline is "rc 0, or rc 2 WITH the UNPROVEN diagnostic". A bare
# rc=2 with no explanation is still a failure, exactly as the device-precond healthy
# column requires ("a 2 must be explained"). Over-asserting here is how a
# correctness gate becomes a flaky gate.
echo "--- S0 baseline (rc=0, or rc=2 WITH its UNPROVEN diagnostic) ---"
bash "$TGT" >"$LOG_DIR/S0_clean.log" 2>&1
s0_rc=$?
if [ "$s0_rc" -eq 0 ]; then
  echo "  S0   rc=0  AS EXPECTED (clean tree, 2-GPU box)" | tee -a "$SUMMARY"
elif [ "$s0_rc" -eq 2 ] && grep -q 'GATE INVENTORY: UNPROVEN' "$LOG_DIR/S0_clean.log"; then
  echo "  S0   rc=2  AS EXPECTED (box cannot run the documented configuration;" | tee -a "$SUMMARY"
  echo "        the UNPROVEN diagnostic is present, so this is a BOX verdict)" | tee -a "$SUMMARY"
else
  echo "  S0   FAILED -- rc=$s0_rc on a clean tree. rc=0 is healthy and rc=2 WITH" | tee -a "$SUMMARY"
  echo "    the UNPROVEN diagnostic is healthy; anything else means the checker" | tee -a "$SUMMARY"
  echo "    is broken, and every state below would be scored against it." | tee -a "$SUMMARY"
  FAILED=$((FAILED + 1))
fi

run_state M1 "un-anchor the line-start require test" "CAUGHT-selfcheck" <<'PATCH'
  if grep -qE '^[[:space:]]*gate_require_gpus[[:space:]]+[0-9]' "$1"; then
|||
  if grep -q 'gate_require_gpus' "$1"; then
PATCH

run_state M2 "drop comment stripping (read prose again)" "CAUGHT-selfcheck" <<'PATCH'
strip_comments() { sed -e 's/[[:space:]]#.*$//' -e 's/^[[:space:]]*#.*$//' "$1"; }
|||
strip_comments() { cat "$1"; }
PATCH

run_state M3 "reintroduce the stdin-consuming grep chain" "CAUGHT-selfcheck" <<'PATCH'
granularity_of() { # $1 = comment-stripped code file
  if grep -qE '^[[:space:]]*gate_require_gpus[[:space:]]+[0-9]' "$1"; then
|||
granularity_of() { # $1 = comment-stripped code file
  exec 3<"$1"
  if grep -qE '^[[:space:]]*gate_require_gpus[[:space:]]+[0-9]' <&3; then
@@@
  elif grep -q 'gate_gpu_count' "$1"; then
|||
  elif grep -q 'gate_gpu_count' <&3; then
PATCH

run_state M4 "invert the per-arm self-check expectation" "CAUGHT-selfcheck" <<'PATCH'
  [ "$p" = "per-arm" ] || { echo "SELFCHECK: per-arm gate misread as '$p'" >&2; bad=1; }
|||
  [ "$p" = "WHOLE-gate" ] || { echo "SELFCHECK: per-arm gate misread as '$p'" >&2; bad=1; }
PATCH

run_state M5 "disable the per-arm NO-SKIP-REPORT check" "CAUGHT-selfcheck" <<'PATCH'
    printf '%s\n' "$code" | grep -qE 'SKIPPED_(TP2|SUB)|PARTIAL' || {
|||
    printf '%s\n' "$code" | grep -qE 'ZZZ_NEVER_MATCHES' || {
PATCH

# NOTE the old side names gate_require_device too. When audit #16 taught
# gpu_guard_state() to recognise the combined helper, this patch went stale --
# and the pre-flight is what said so, because its anchor is now the full line
# rather than a prefix truncated at the first `|`. Re-pointing a mutation at moved
# code is the sanctioned response; loosening its anchor is not.
run_state M7 "disable the NEEDS-GPU-without-a-guard verdict" "CAUGHT-selfcheck" <<'PATCH'
  if grep -qE 'renderD|gate_require_render_node|gate_require_device' "$1"; then echo "render-node"; else echo "MISSING"; fi
|||
  if grep -qE 'renderD|gate_require_render_node|gate_require_device' "$1"; then echo "render-node"; else echo "not-gpu"; fi
PATCH

# The anchor was REPOINTED: the parser moved from an inline block inside the
# tally loop into parse_tally_counts() so the checker's own self-check could
# exercise it on a fixture. The anchor pre-flight caught the stale one by
# refusing to run -- which is exactly what it exists for, and is why the state
# was re-pointed rather than the pre-flight loosened.
run_state M8 "make the tally parser comma-only again" "CAUGHT-selfcheck" <<'PATCH'
  PARSE_SK="$(printf '%s' "$txt" | sed -nE 's/.*[,/] ?([0-9]+) SKIPPED.*/\1/p')"
|||
  PARSE_SK="$(printf '%s' "$txt" | sed -nE 's/.*, ([0-9]+) SKIPPED.*/\1/p')"
PATCH

run_gate_state M9 "strip the render-node guard from a GPU gate" "CAUGHT-selfcheck" <<'PATCH'
gate_require_device || exit 2
|||
: guard removed on purpose
PATCH

# M6 is scored differently: it needs a STUB sycl-ls reporting no device, because
# the defect is that a box the checker cannot see still exited 0. There is no
# source to edit, which is why it gets its own runner.
echo "--- M6 0-GPU box must be refused (never exit 0) ---"
restore
STUB="$(mktemp -d "${TMPDIR:-/tmp}/gi_stub.XXXXXX")"
printf '#!/bin/sh\nexit 0\n' >"$STUB/sycl-ls"
chmod +x "$STUB/sycl-ls"
PATH="$STUB:$PATH" bash "$TGT" >"$LOG_DIR/M6_zero_gpu.log" 2>&1
m6rc=$?
rm -rf "$STUB"
m6class="$(rc_class "$m6rc")"
if [ "$m6class" = "CAUGHT-skip" ]; then
  printf '  %-4s rc=%-3s %-16s %-50s AS EXPECTED\n' M6 "$m6rc" "$m6class" \
    "0-GPU box refused with exit 2" | tee -a "$SUMMARY"
  NC=$((NC + 1))
else
  printf '  %-4s rc=%-3s %-16s %-50s UNEXPECTED: soft-skip regression\n' M6 "$m6rc" "$m6class" \
    "0-GPU box refused with exit 2" | tee -a "$SUMMARY"
  echo "    a box the checker cannot see must not exit 0" | tee -a "$SUMMARY"
  FAILED=$((FAILED + 1))
fi
grep -m1 'doc-tally: SKIPPED' "$LOG_DIR/M6_zero_gpu.log" | sed 's/^/       > /' | tee -a "$SUMMARY" || true

# --- restore and prove the tree survived -----------------------------------
restore
trap - EXIT INT TERM

echo
if git diff --quiet -- "$TGT" "$TGT2"; then
  echo "restore: $TGT and $TGT2 are byte-identical to HEAD"
else
  echo "FAIL: restore left the tree dirty" >&2; FAILED=$((FAILED + 1))
fi
if grep -qE 'ZZZ_NEVER_MATCHES' "$TGT" "$TGT2"; then
  echo "FAIL: mutation residue left in $TGT/$TGT2" >&2; FAILED=$((FAILED + 1))
else
  echo "residue: none (no ZZZ_NEVER_MATCHES marker in $TGT/$TGT2)"
fi

echo
if [ "$FAILED" = "0" ]; then
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=S0,M1,M2,M3,M4,M5,M7,M8,M9,M6"
    echo "caught=$NC/$NC"
  } >"$STAMP" 2>/dev/null
  echo "GATE-INVENTORY MUTATION MATRIX: PASS ($NC/$NC mutations refused the checker)"
  emit_summary
  echo "  stamped $(git rev-parse --short HEAD 2>/dev/null) -> $STAMP"
  exit 0
fi
echo "GATE-INVENTORY MUTATION MATRIX: FAIL ($FAILED state(s) not as characterized)" >&2
echo "  Full per-state output: $SUMMARY" >&2
exit 1
