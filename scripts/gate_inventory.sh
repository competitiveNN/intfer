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

N_PASS=0; N_PARTIAL=0; N_SKIP=0

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

  # Accumulate the EXPECTED exit class for this gate, from source. This is what
  # makes the doc-tally reconciliation below possible without running anything.
  case "$gran" in
    "-")         N_PASS=$((N_PASS + 1)) ;;
    per-arm)     N_PARTIAL=$((N_PARTIAL + 1)) ;;
    WHOLE-gate)  N_SKIP=$((N_SKIP + 1)) ;;
  esac

  # THE INVARIANT: a whole-gate skip must not strand arms that would still run.
  if [ "$gran" = "WHOLE-gate" ] && [ "$arm" = "yes" ]; then
    mark="$mark <-- OVER-SKIP: 1-GPU arms stranded (audit #13)"
    VIOLATIONS=$((VIOLATIONS + 1))
  fi

  printf '%-26s %-9s %-22s %-11s %s%s\n' "$name" \
    "$([ "$build" = NONE ] && echo "-" || echo yes)" "$precond" "$gran" "$arm" "$mark"
done

# --- DOC TALLY RECONCILIATION -----------------------------------------------
# Close the class, not the instance. During audit #13 this file's own docs
# claimed "8 PASS, 2 PARTIAL, 1 SKIPPED" for a 10-gate suite -- a sum of 11,
# produced by listing a PARTIAL gate's green inner sections alongside the gates
# that really exited 0. A reviewer caught it. Nothing mechanical would have.
#
# The expected tally is DERIVED from the source classification above, so this
# needs no gate run:
#   no precondition          -> exits 0            -> PASS
#   per-arm precondition     -> exits 2 on 1 GPU   -> PARTIAL (its 1-GPU arms run)
#   whole-gate precondition  -> exits 2 on 1 GPU   -> SKIPPED (no 1-GPU arm)
# With 2+ GPUs visible every gate would exit 0, so a doc still claiming
# PARTIAL/SKIPPED is then stale in the other direction and this catches that too.
echo
. "$SCRIPT_DIR/_gate_precond.sh" 2>/dev/null
NDEV="$(gate_gpu_count 2>/dev/null || echo 0)"
DOCS="AGENTS.md $(ls reports/*.md 2>/dev/null)"

if [ "${NDEV:-0}" -eq 0 ]; then
  echo "doc-tally: SKIPPED (0 level_zero GPUs visible -- cannot derive the expected rc classes)."
  echo "  Loud, not silent: a precondition that cannot see the box verifies nothing."
  echo "  (This is the libumf LD_LIBRARY_PATH gotcha, not a code problem.)"
else
  TALLY_RAN=1
  if [ "$NDEV" -ge 2 ]; then EXP_PASS="$((N_PASS + N_PARTIAL + N_SKIP))"; EXP_PARTIAL=0; EXP_SKIP=0
  else EXP_PASS="$N_PASS"; EXP_PARTIAL="$N_PARTIAL"; EXP_SKIP="$N_SKIP"; fi
  echo "doc-tally: expected from source + $NDEV GPU(s) visible = ${EXP_PASS} PASS, ${EXP_PARTIAL} PARTIAL, ${EXP_SKIP} SKIPPED"
  echo "           (PASS=$N_PASS no-precond, PARTIAL=$N_PARTIAL per-arm, SKIPPED=$N_SKIP whole-gate)"
  # NOT ^-anchored. The tally appears mid-sentence in prose ("The correct
  # present-tense sentence is: *7 PASS, ..."), so an anchored pattern matches
  # NOTHING and this check reports "0 stale" while verifying zero sites -- the
  # vacuous-gate trap, hit here for real. sites_found is asserted non-zero
  # below so that can never pass again.
  #
  # Two false-positive classes had to be handled, both found by running it:
  #   (1) HISTORICAL/QUOTED text that legitimately carries an old tally. Excluded
  #       by two explicit markers ("audit-#", "COUNTING TRAP"). Excluded lines
  #       are PRINTED and counted, so the exclusion cannot silently swallow the
  #       corpus -- and the count is asserted below.
  #   (2) A tally WRAPPED across two markdown lines ("7 PASS, 2 PARTIAL," then
  #       "1 SKIPPED**"), which parses as SKIPPED=0. The next line is consulted
  #       when PARTIAL is present but SKIPPED is not.
  bad_tally=0; sites_found=0; excluded=0
  for d in $DOCS; do
    [ -f "$d" ] || continue
    # ALL lines, so the wrap look-up below can reach a continuation line that
    # does not itself match the tally regex (the "1 SKIPPED**" half never does).
    mapfile -t ALL < "$d"
    while IFS= read -r hit; do
      [ -n "$hit" ] || continue
      ln="${hit%%:*}"; txt="${hit#*:}"
      p="$(printf '%s' "$txt"  | sed -nE 's/.*[^0-9]([0-9]+) PASS.*/\1/p')"
      pa="$(printf '%s' "$txt" | sed -nE 's/.*, ([0-9]+) PARTIAL.*/\1/p')"
      sk="$(printf '%s' "$txt" | sed -nE 's/.*, ([0-9]+) SKIPPED.*/\1/p')"
      [ -n "$p" ] || continue
      # (2) wrapped tally: pull SKIPPED off the FOLLOWING FILE LINE if absent.
      if [ -n "$pa" ] && [ -z "$sk" ] && [ "$ln" -lt "${#ALL[@]}" ]; then
        sk="$(printf '%s' "${ALL[$ln]}" | sed -nE 's/^[^0-9]*([0-9]+) SKIPPED.*/\1/p')"
      fi
      pa="${pa:-0}"; sk="${sk:-0}"
      # (1) explicit, narrow exclusion for quoted/history text.
      if printf '%s' "$txt" | grep -qE 'audit-#|COUNTING TRAP'; then
        excluded=$((excluded + 1))
        printf '           excl  %s:%s  %s\n' "$d" "$ln" "$(printf '%s' "$txt" | cut -c1-66)"
        continue
      fi
      sites_found=$((sites_found + 1))
      if [ "$p" = "$EXP_PASS" ] && [ "$pa" = "$EXP_PARTIAL" ] && [ "$sk" = "$EXP_SKIP" ]; then
        printf '           ok    %s:%s  %s\n' "$d" "$ln" "$(printf '%s' "$txt" | cut -c1-66)"
      else
        printf '           STALE %s:%s  says %s/%s/%s, source implies %s/%s/%s\n' \
          "$d" "$ln" "$p" "$pa" "$sk" "$EXP_PASS" "$EXP_PARTIAL" "$EXP_SKIP"
        bad_tally=$((bad_tally + 1))
      fi
    done < <(grep -nE '[0-9]+ PASS, [0-9]+ PARTIAL|[0-9]+ PASS[^*]*$' "$d")
  done
  echo "           sites_found=$sites_found excluded_history=$excluded stale_tally_sites=$bad_tally"
  VIOLATIONS=$((VIOLATIONS + bad_tally))
  if [ "$sites_found" -eq 0 ]; then
    echo "           <-- VACUOUS: matched 0 tally sites, so nothing was verified." >&2
    VIOLATIONS=$((VIOLATIONS + 1))
  fi
fi

echo
if [ "$VIOLATIONS" = "0" ]; then
  # Do NOT claim the doc tallies reconciled when that check declined to run. An
  # OK line reading "reconcile" after a SKIP is the same overclaim this script
  # exists to catch, in the script's own voice.
  if [ "${TALLY_RAN:-0}" = "1" ]; then
    echo "GATE INVENTORY: OK (every gate builds its own target; no whole-gate skip strands 1-GPU arms; doc tallies reconcile)"
  else
    echo "GATE INVENTORY: OK on guards/granularity; DOC TALLY CHECK DID NOT RUN (see the SKIPPED note above)"
  fi
  exit 0
fi
echo "GATE INVENTORY: $VIOLATIONS PROBLEM(S) -- see the <-- markers and STALE lines above" >&2
exit 1
