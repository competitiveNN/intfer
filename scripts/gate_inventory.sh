#!/usr/bin/env bash
# Report the guard state of every correctness gate, from SOURCE.
#
# WHY THIS IS A SCRIPT AND NOT A ONE-LINE GREP. An ad-hoc inventory was used
# during audit #13 that classified a gate's precondition by grepping for
# `gate_require_gpus|gate_gpu_count` and printing the SAME "per-arm" label for
# every hit. That mislabelled test_tp2_27b.sh, which has no 1-GPU arm at all and
# correctly skips whole-gate. The output looked like evidence and was wrong in
# the one column a reader would act on. A report that cannot distinguish the two
# cases is worse than no report.
#
# So this script reports the actual MECHANISM from each gate's source, and then
# checks the invariant that the mechanism must satisfy:
#
#     a gate that skips WHOLE on a missing resource must have NO arm that would
#     still run without it.
#
# Violating that is exactly the audit #13 over-skip: a top-level
# `gate_require_gpus 2` on test_device_argmax.sh threw away its TP=1 coverage,
# including the serve fence lock. This check fails if that shape ever returns.
#
# It is deliberately NOT a correctness gate and makes no assertion about the
# engine -- it is an inventory of the gates themselves. Exit 1 only if the
# invariant above is violated.
#
# Usage: scripts/gate_inventory.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

GATES=$(ls scripts/test_*.sh 2>/dev/null | sort)
[ -n "$GATES" ] || { echo "FAIL: no scripts/test_*.sh found" >&2; exit 1; }

# Is there an arm that runs on a single GPU? Two shapes count:
#   * a literal `--tp 1` / `--tp "$tp"` with tp pinned to 1
#   * a loop or helper whose tp argument is only ever 1
# We look for the explicit form plus a loop var, and report UNKNOWN rather than
# guessing when neither is present.
has_1gpu_arm() {
  local f="$1"
  grep -qE -- '--tp 1([^0-9]|$)' "$f" && return 0
  # `ids_of 1`, `count_of 1`, `spec_steps`-style direct 1-arg calls
  grep -qE '\b(ids_of|count_of|ids_of_tp) 1([^0-9]|$)' "$f" && return 0
  # `for tp in $TPS` where TPS defaults to "1" -> the tp=1 pass is real
  grep -qE '^TPS="1"$' "$f" && return 0
  return 1
}

printf '%-26s %-9s %-22s %-11s %s\n' GATE BUILD-GUARD PRECONDITION GRANULARITY 1-GPU-ARM
printf '%s\n' "----------------------------------------------------------------------------------------"

VIOLATIONS=0
for f in $GATES; do
  name="$(basename "$f")"

  # Build guard: report the ACTUAL cmake target, not just "yes".
  target="$(sed -nE 's/^[[:space:]]*gate_require_build[[:space:]]+([A-Za-z0-9_]+).*/\1/p' "$f" | head -1)"
  if [ -n "$target" ]; then build="-> $target"; else build="NONE"; fi

  # Precondition mechanism, read from source.
  if grep -q 'gate_require_gpus' "$f"; then
    precond="gate_require_gpus"
  elif grep -q 'gate_gpu_count' "$f"; then
    precond="gate_gpu_count"
  else
    precond="none"
  fi

  # Granularity: WHOLE-gate skip vs PER-ARM skip. This is the column that was
  # wrong before, so it is derived, not pattern-matched.
  if [ "$precond" = "none" ]; then
    gran="-"; arm="-"
  elif grep -qE 'HAVE_TP2|SKIPPED_TP2' "$f"; then
    gran="per-arm"; arm="$(has_1gpu_arm "$f" && echo yes || echo "no?!")"
  else
    # gate_require_gpus with no per-arm machinery => skips the whole script.
    gran="WHOLE-gate"; arm="$(has_1gpu_arm "$f" && echo yes || echo no)"
  fi

  if [ "$build" = "NONE" ]; then
    mark=" <-- NO BUILD GUARD"
    VIOLATIONS=$((VIOLATIONS + 1))
  else mark=""; fi

  # THE INVARIANT: a whole-gate skip must not strand arms that would still run.
  if [ "$gran" = "WHOLE-gate" ] && [ "$arm" = "yes" ]; then
    mark="$mark <-- OVER-SKIP: 1-GPU arms stranded (audit #13)"
    VIOLATIONS=$((VIOLATIONS + 1))
  fi

  printf '%-26s %-9s %-22s %-11s %s%s\n' "$name" \
    "$([ "$build" = NONE ] && echo "-" || echo yes)" "$precond" "$gran" "$arm" "$mark"
done

echo
if [ "$VIOLATIONS" = "0" ]; then
  echo "GATE INVENTORY: OK (every gate builds its own target; no whole-gate skip strands 1-GPU arms)"
  exit 0
fi
echo "GATE INVENTORY: $VIOLATIONS PROBLEM(S) -- see the <-- markers above" >&2
exit 1
