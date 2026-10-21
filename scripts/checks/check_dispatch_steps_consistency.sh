#!/usr/bin/env bash
# Every STEPS entry must be DISPATCHABLE, and every dispatch arm must be either
# SCHEDULED or self-declared POST-STACK.
#
# MEASURED, AND THIS IS THE DEFECT CLASS IT EXISTS FOR. Adding two scripts to this
# file's STEPS lists was done with a short `s.replace(anchor, new, 1)`, and that
# same substring ALSO occurs in the dispatch case labels -- so the substitution
# landed on a dispatch arm instead of a STEPS list. Two failures, both silent:
#
#     timeout ... bash "$SCRIPT_DIR/check_env_not_box_state.sh check_destructive_git.sh"
#
#   * a FILENAME WITH A SPACE -> rc=127 "No such file or directory", and
#   * neither script reached STEPS at all, so section 0c reported neither as
#     scheduled while the stack still ran them.
#
# `bash -n` passes that: it is syntactically valid and semantically wrong. The
# only thing that caught it was the runtime rc, which is the worst place to find
# out -- a checker that exists is one rc cheaper than a stack run.
#
# WHY THIS IS NOT ALREADY COVERED. Section 0b asserts every check_*/mutate_* HAS a
# dispatch arm, and 0c asserts every one is SCHEDULED. Neither asserts the two
# sets AGREE, which is a third property: a STEPS token naming a script with no arm
# cannot run, and an arm absent from STEPS never does. 0b and 0c can both pass
# while the two lists have drifted apart, because each only looks at one side.
#
# THE POST-STACK EXEMPTION IS DERIVED, NEVER LISTED. check_stack_cert.sh grades the
# certificate this very run produces, so it must run AFTER the stack and is
# legitimately not a step. It declares that itself with a `# POST-STACK:` header,
# and this checker reads that header rather than carrying its own list -- a second
# list here would be exactly the hand-maintained inventory this repo keeps deleting.
# A script that is BOTH POST-STACK and in STEPS is refused, because those are
# contradictory claims.
#
# AND ONE FILESYSTEM FACT THE TEXTUAL SECTIONS CANNOT SEE. Sections 1-3 compare
# the two lists to each other and to each arm's body, but every comparison is
# string-vs-string: nothing asks whether the script is THERE. A STEPS entry whose
# file was deleted keeps all of its memberships and a body that names the label
# exactly -- it is scheduled in every sense above and it dies rc=127 at runtime.
# Section 4 closes that, and DS-5 in the matrix is its mutation state.
#
# ANTI-VACUITY: both sets are asserted non-empty, and a parse that yields nothing
# must fail rather than certify two empty sets as agreeing.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

T="scripts/run_all_checks.sh"
echo "== DISPATCH / STEPS CONSISTENCY =="
[ -f "$T" ] || { echo "FAIL: $T missing" >&2; exit 2; }

SCAN="scripts/lib/_dispatch_scan.py"

# Everything is derived by ONE scanner, in its own file. Two reasons, both
# measured in this session:
#   * a regex embedded in a bash heredoc had its backslashes mangled and died,
#     which surfaced as the scan finding NOTHING -- two empty sets agreeing, the
#     vacuous-green shape reached through quoting;
#   * the arm-body comparison needs the WHOLE path segment, because the corrupted
#     path's FIRST .sh still equals the label, so a reader that stops there sees
#     agreement. That was the second version of this check and DS-1 passed it.
scan="$(python3 "$SCAN" "$T" 2>&1)" || {
  echo "  FAIL  the scanner failed to run:" >&2; printf '%s\n' "$scan" | sed 's/^/        /' >&2
  exit 1
}
mapfile -t LABELS  < <(printf '%s\n' "$scan" | sed -n 's/^LABEL //p')
mapfile -t STEPS_L < <(printf '%s\n' "$scan" | sed -n 's/^STEP //p')
mapfile -t MISMATCH < <(printf '%s\n' "$scan" | sed -n 's/^ARM-MISMATCH //p')
mapfile -t STRAY < <(printf '%s\n' "$scan" | sed -n 's/^STRAY-STEP-TOKEN //p')
mapfile -t BODY_L < <(printf '%s\n' "$scan" | sed -n 's/^BODY //p')

# Where each label's script actually LIVES. Since the 2026-10-20 hierarchy move
# the label is still a basename (that is what the STEPS lists carry and what the
# case arms match) but the path it executes carries a subdirectory, so a
# filesystem test written as `scripts/$name` fails on every one of them. The
# location is DERIVED from the arm's own body rather than from a hard-coded
# subdirectory list -- a second list of where things live is exactly the
# hand-maintained inventory this repo keeps deleting.
declare -A WHERE
for b in "${BODY_L[@]}"; do
  nm="${b%% *}"; pth="${b#* }"
  [ -n "$nm" ] && [ -n "$pth" ] && WHERE["$nm"]="$pth"
done

n_labels="${#LABELS[@]}"
n_steps="${#STEPS_L[@]}"
echo "-- derived: $n_labels dispatch arm(s), $n_steps STEPS entry(ies)"

bad=0
# ANTI-VACUITY: two empty sets agree trivially. Both populations are floors.
if [ "$n_labels" -lt 10 ] || [ "$n_steps" -lt 10 ]; then
  echo "  FAIL  parse looks wrong ($n_labels arms / $n_steps entries) -- the two" >&2
  echo "        near-empty sets would otherwise 'agree' and certify nothing." >&2
  exit 1
fi

# --- 1. every STEPS entry must be dispatchable -----------------------------
# A step whose script has no arm can never run, and the run loop reports it as a
# pass rather than a skip. Section 0c cannot see this: it asks whether every
# check_*/mutate_* is SCHEDULED, which is the other direction.
for s in "${STEPS_L[@]}"; do
  printf '%s\n' "${LABELS[@]}" | grep -qxF "$s" && continue
  echo "  FAIL  $s is in STEPS but has NO dispatch arm -- it can never run" >&2
  bad=$((bad + 1))
done

# --- 2. every dispatch arm must be SCHEDULED or self-declared POST-STACK ----
scheduled="$(printf '%s\n' "${STEPS_L[@]}")"
for a in "${LABELS[@]}"; do
  printf '%s\n' "$scheduled" | grep -qxF "$a" && continue
  # Self-declared exemption, read from the script's own header. Carrying a list
  # here would be a second hand-maintained inventory, which is exactly what this
  # repo keeps deleting from its own docs.
  # Derived location (see WHERE above); falls back to scripts/ for a label
  # whose arm carries no path, which is the only case a literal could name.
  _a="${WHERE[$a]:-$a}"
  if [ -f "scripts/$_a" ] && head -40 "scripts/$_a" | grep -q '^# POST-STACK:'; then
    echo "  ok    $a  dispatch-only (declares POST-STACK in its own header)"
    continue
  fi
  echo "  FAIL  $a has a dispatch arm but is NOT in STEPS and does not declare" >&2
  echo "        POST-STACK -- audit #18: an arm that never runs certifies nothing" >&2
  bad=$((bad + 1))
done

# --- 2b. no .sh token may sit OUTSIDE a STEPS line's quotes -----------------
# Inside the quotes everything is fine; the corruption is that ONE token ended up
# on the wrong side of a quote, where it stops being a step and becomes a COMMAND.
# A checker that harvests only inside the quotes is structurally blind to it.
if [ "${#STRAY[@]}" -gt 0 ]; then
  for t in "${STRAY[@]}"; do
    echo "  FAIL  stray step token: $t -- outside the quotes it is a COMMAND" >&2
  done
  echo "        (observed twice: the stack died with 'command not found')" >&2
  bad=$((bad + 1))
else
  echo "  ok    no .sh token escapes a STEPS line's quotes"
fi

# --- 3. each arm must EXECUTE the script its label names --------------------
# This is the bug that shipped: an arm whose path became
#     bash "$SCRIPT_DIR/a.sh b.sh"
# valid shell, passes `bash -n`, and fails rc=127 at runtime. Deriving arms from
# the CASE LABELS cannot see it -- the labels were untouched.
if [ "${#MISMATCH[@]}" -gt 0 ]; then
  for m in "${MISMATCH[@]}"; do
    echo "  FAIL  dispatch arm $m -- a filename with a space cannot execute" >&2
  done
  echo "        (rc=127 at runtime; 'bash -n' passes it)" >&2
  bad=$((bad + 1))
else
  echo "  ok    every arm executes exactly the script its label names"
fi

# --- 4. every SCHEDULED script must exist on disk ----------------------------
# Sections 1-3 prove the LABELS, the ARM BODIES and the ARM/STEPS membership
# agree, and every one of them is textual: none touches the filesystem. A STEPS
# entry whose script was deleted -- or retired into scripts/archive/ without
# also being descheduled -- keeps its arm, keeps its membership in both
# directions, and keeps a body that names the label exactly. It is "scheduled"
# in every sense sections 1-3 can see and it dies rc=127 when its step runs.
for s in "${STEPS_L[@]}"; do
  _s="${WHERE[$s]:-$s}"
  [ -f "scripts/$_s" ] && continue
  echo "  FAIL  $s is SCHEDULED but scripts/$_s does not exist -- the run" >&2
  echo "        loop will invoke a missing file (rc=127), and a step that" >&2
  echo "        cannot run is reported as a pass rather than a skip" >&2
  bad=$((bad + 1))
done
[ "$bad" = 0 ] && echo "  ok    all $n_steps scheduled script file(s) exist on disk"

echo
if [ "$bad" = 0 ]; then
  echo "DISPATCH/STEPS CONSISTENCY: PASS ($n_labels arm(s), $n_steps scheduled)"
  echo "  every scheduled script can run, and every arm is scheduled or"
  echo "  self-declared POST-STACK (exemption derived from the script's header)"
  echo "RESULT: the dispatch labels and the STEPS lists agree"
  exit 0
fi
echo "DISPATCH/STEPS CONSISTENCY: FAIL ($bad problem(s))" >&2
echo "RESULT: FAIL ($bad) -- a step that cannot run, or an arm that never does" >&2
exit 1
