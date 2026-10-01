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

# Best-effort oneAPI env, so this checker can actually SEE the box. Its sibling
# gates all `source setvars.sh` before counting; this script did not, and
# without it sycl-ls is absent from PATH, the count is 0, and the doc-tally
# check below skips -- silently, while the script still exited 0 (audit #14).
# A checker that cannot see the box must not report success.
if ! command -v sycl-ls >/dev/null 2>&1; then
  if [ -r /opt/intel/oneapi/setvars.sh ]; then
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
  fi
fi

# The shared vacuity guard: a checker that cannot harvest its subject cannot
# certify it. Sourced, not executed (sibling of scripts/_gate_precond.sh).
# Placed AFTER the oneAPI-env block -- a first version landed INSIDE it, so it
# was sourced only when sycl-ls happened to be missing from PATH, and the call
# site 400 lines later failed with "command not found" on the normal path.
# A guard sourced conditionally is a guard that is usually absent.
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
enum_reset

GATES=$(ls scripts/test_*.sh 2>/dev/null | sort)
[ -n "$GATES" ] || { echo "FAIL: no scripts/test_*.sh found" >&2; exit 1; }

# --- CODE vs COMMENT (audit #14) ---------------------------------------------
# Both classification columns used to grep the RAW file. A gate's own explanatory
# comments quote the very tokens being searched for, so the greps were reading
# prose. Two live consequences, both measured:
#   * test_device_argmax.sh's ONLY `gate_require_gpus` occurrence is inside the
#     comment explaining the audit-#13 defect it must never re-introduce -- so
#     the PRECONDITION column called it `gate_require_gpus` while it actually
#     uses gate_gpu_count, and reported test_gpu_optin.sh (identical mechanism)
#     under a DIFFERENT label.
#   * a whole-gate skip was detected by the ABSENCE of per-arm marker text, so
#     re-adding `gate_require_gpus 2 || exit 2` at top level WHILE KEEPING the
#     per-arm machinery still classified as "per-arm" and this script exited 0.
#     Verified: the checker's headline invariant had a proven false negative.
#
# So every classification now runs against a comment-stripped copy. `#` inside
# strings/expansions is vanishingly rare in these gates, and an over-strip can
# only ever REMOVE matches, which for these two columns fails toward "no
# precondition detected" (a report) or toward flagging, never toward silence.
strip_comments() { sed -e 's/[[:space:]]#.*$//' -e 's/^[[:space:]]*#.*$//' "$1"; }

# WHOLE-GATE skip = a `gate_require_gpus` STATEMENT at the start of a line.
# A per-arm guard calls gate_gpu_count and branches; it does not call
# gate_require_gpus at all.
#
# Takes a FILE, not stdin. The first version piped the stream into `grep -q` and
# then a second `grep -q` in the elif -- but `grep -q` exits as soon as it has
# its verdict, consuming the whole stream, so the elif always saw an empty input
# and every gate read as "none". The classifier self-check below caught that on
# its first run, which is the argument for having it.
#
# BROKEN is a third answer, not a fallback: a `gate_require_gpus` that is NOT a
# line-start statement (wrapped in a helper function, or a command substitution)
# is a whole-gate skip this classifier cannot reason about. Calling that
# "per-arm" would be exactly the false negative that let audit #13's defect
# through, so it refuses to classify -- fail toward flagging.
#
# The anchoring and the comment-stripping are MUTUALLY COVERING, which is why the
# anchoring needed its own case below to stop being untested decoration: measured,
# un-anchoring the line-start test changes no verdict on the real gates once
# comments are stripped, so a mutation that removes the anchor is EQUIVALENT and
# would otherwise be unobservable.
granularity_of() { # $1 = comment-stripped code file
  if grep -qE '^[[:space:]]*gate_require_gpus[[:space:]]+[0-9]' "$1"; then
    echo "WHOLE-gate"
  elif grep -q 'gate_require_gpus' "$1"; then
    echo "BROKEN"
  elif grep -q 'gate_gpu_count' "$1"; then
    echo "per-arm"
  else
    echo "none"
  fi
}

# Is there an arm that runs on a single GPU? Looked for on the stripped stream.
# Reports `yes` / `no` / `UNKNOWN` -- never guesses, and `UNKNOWN` on a
# per-arm gate is itself a violation (an unclassified 1-GPU arm is exactly the
# audit-#13 shape).
has_1gpu_arm() { # $1 = comment-stripped stream file
  local f="$1"
  grep -qE -- '--tp 1([^0-9]|$)' "$f" && { echo yes; return 0; }
  grep -qE '(ids_of|count_of|spec_steps|ids_of_tp) 1([^0-9]|$)' "$f" && { echo yes; return 0; }
  grep -qE '^[[:space:]]*TPS="1"$' "$f" && { echo yes; return 0; }
  echo no
}

# Does this gate NEED a GPU, and does it carry a guard that will SKIP rather
# than fail red if the box has none? Echoes: not-gpu | render-node | MISSING.
#
# Split out as a function of a file so the classifier self-check can exercise it
# on a seeded temp gate. Inlined in the loop it was UNREACHABLE in practice: on
# the real tree every GPU gate is guarded, so the MISSING branch never fires and a
# mutation disabling it is an EQUIVALENT mutation -- untestable, i.e. decoration.
# That is the M1 lesson applied to M7, and it is the whole reason this is a
# function and not four lines in the loop.
#
# The NEEDS-GPU test is RE-DERIVED from the code, not read from a remembered
# list, so a newly added GPU gate is classified without anyone editing a table.
# That re-derivation is what found the gap: test_dpas_fold.sh (which said "needs a
# GPU" in a COMMENT and carried no guard), test_http_batch.sh and
# test_serve_load.sh (both launch the server `--gpu --tp 1`). All three PASS on a
# 1-GPU box, which is exactly why the gap survived AUDIT #14.
gpu_guard_state() { # $1 = comment-stripped code file
  grep -qE -- '(--gpu|sycl-ls|level_zero|xpu-smi|dpas_fold_probe)' "$1" || { echo "not-gpu"; return 0; }
  # A GPU gate with no COUNT-based precondition still needs SOME guard. The
  # render-node check is the strict, adapter-independent one: a missing libumf
  # makes sycl-ls report zero devices on a perfectly healthy box, so a count
  # would refuse coverage that would actually run.
  # Matches the shared helper by name as well as the raw probe, so a gate that
  # adopts gate_require_device (which covers BOTH failure modes) is not reported
  # as unguarded.
  if grep -qE 'renderD|gate_require_render_node|gate_require_device' "$1"; then echo "render-node"; else echo "MISSING"; fi
}

# --- CLASSIFIER SELF-CHECK (behavioural, on a seeded temp gate) --------------
# The invariant above had a proven false negative, so the classifier is now
# itself gated: a synthetic gate reproducing the audit-#13 shape (top-level
# require PLUS per-arm machinery -- the combination that fooled the old check)
# must classify WHOLE-gate, and a per-arm gate must not. If this ever cannot
# fail, it is the vacuous-check trap again and it will say so.
selfcheck_classifier() {
  local bad=0 tmp w p r
  tmp="$(mktemp "${TMPDIR:-/tmp}/gate_inv_selfcheck.XXXXXX")"
  # audit-#13 defect shape, verbatim in structure: whole-gate require retained
  # while the per-arm machinery is still present.
  cat >"$tmp" <<'GATE'
. "$SCRIPT_DIR/_gate_precond.sh"
gate_require_gpus 2 || exit 2
N_GPUS="$(gate_gpu_count)" || N_GPUS=0
HAVE_TP2=0; [ "$N_GPUS" -ge 2 ] && HAVE_TP2=1
GATE
  strip_comments "$tmp" >"$tmp.c"
  w="$(granularity_of "$tmp.c")"
  [ "$w" = "WHOLE-gate" ] || { echo "SELFCHECK: audit-#13 shape misread as '$w'" >&2; bad=1; }
  # a real per-arm gate must NOT be read as whole-gate
  cat >"$tmp" <<'GATE'
. "$SCRIPT_DIR/_gate_precond.sh"
N_GPUS="$(gate_gpu_count)" || N_GPUS=0
HAVE_TP2=0; [ "$N_GPUS" -ge 2 ] && HAVE_TP2=1
GATE
  strip_comments "$tmp" >"$tmp.c"
  p="$(granularity_of "$tmp.c")"
  [ "$p" = "per-arm" ] || { echo "SELFCHECK: per-arm gate misread as '$p'" >&2; bad=1; }
  # the real whole-gate gate must be seen as whole-gate
  cat >"$tmp" <<'GATE'
. "$SCRIPT_DIR/_gate_precond.sh"
gate_require_gpus 2 || exit 2
GATE
  strip_comments "$tmp" >"$tmp.c"
  r="$(granularity_of "$tmp.c")"
  [ "$r" = "WHOLE-gate" ] || { echo "SELFCHECK: real whole-gate gate misread as '$r'" >&2; bad=1; }
  # a PROSE mention of the token must not create a precondition (the bug)
  cat >"$tmp" <<'GATE'
# an earlier revision put a `gate_require_gpus 2 || exit 2` here, which was wrong
. "$SCRIPT_DIR/_gate_precond.sh"
N_GPUS="$(gate_gpu_count)" || N_GPUS=0
GATE
  strip_comments "$tmp" >"$tmp.c"
  r="$(granularity_of "$tmp.c")"
  [ "$r" = "per-arm" ] || { echo "SELFCHECK: comment text read as code ('$r')" >&2; bad=1; }
  # A gate_require_gpus this classifier cannot place MUST NOT be silently called
  # per-arm. This case is what makes the line-start anchoring observable: drop
  # the anchor and it flips to WHOLE-gate, i.e. an equivalent mutation on the
  # real gates becomes a caught one here.
  cat >"$tmp" <<'GATE'
. "$SCRIPT_DIR/_gate_precond.sh"
need_tp2() { gate_require_gpus 2 || exit 2; }
N_GPUS="$(gate_gpu_count)" || N_GPUS=0
GATE
  strip_comments "$tmp" >"$tmp.c"
  r="$(granularity_of "$tmp.c")"
  [ "$r" = "BROKEN" ] || { echo "SELFCHECK: unplaceable require classified '$r', want BROKEN" >&2; bad=1; }
  # A GPU gate with NO guard must be reported MISSING. This case is what makes
  # the NEEDS-GPU check observable: on the real tree every GPU gate is guarded,
  # so without it a mutation disabling the MISSING branch is EQUIVALENT and
  # would silently pass.
  cat >"$tmp" <<'GATE'
. "$SCRIPT_DIR/_gate_build.sh"
gate_require_build intfer_cli || exit 1
setsid nohup "$BIN" -m "$MODEL" --gpu --tp 1 --serve-http &
GATE
  strip_comments "$tmp" >"$tmp.c"
  r="$(gpu_guard_state "$tmp.c")"
  [ "$r" = "MISSING" ] || { echo "SELFCHECK: unguarded GPU gate read '$r', want MISSING" >&2; bad=1; }
  # ...and the same gate WITH the guard must read render-node, so the case is not
  # satisfied by a function that always says MISSING.
  printf '. "$SCRIPT_DIR/_gate_precond.sh"\ngate_require_render_node || exit 2\n' >>"$tmp"
  strip_comments "$tmp" >"$tmp.c"
  r="$(gpu_guard_state "$tmp.c")"
  [ "$r" = "render-node" ] || { echo "SELFCHECK: guarded GPU gate read '$r', want render-node" >&2; bad=1; }
  # ...and a CPU gate must not be dragged into the device column at all.
  printf 'echo hello\n' >"$tmp"
  strip_comments "$tmp" >"$tmp.c"
  r="$(gpu_guard_state "$tmp.c")"
  [ "$r" = "not-gpu" ] || { echo "SELFCHECK: CPU gate read '$r', want not-gpu" >&2; bad=1; }
  rm -f "$tmp" "$tmp.c"
  return $bad
}
selfcheck_classifier || {
  echo "GATE INVENTORY: FAIL -- the classifier self-check failed, so the table below" >&2
  echo "  proves nothing. Fix the classifier before trusting any row of it." >&2
  exit 1
}

N_PASS=0; N_PARTIAL=0; N_SKIP=0

printf '%-26s %-9s %-22s %-11s %-12s %s\n' GATE BUILD-GUARD PRECONDITION GRANULARITY 1-GPU-ARM RENDER-NODE
printf '%s\n' "-----------------------------------------------------------------------------------------------------"

VIOLATIONS=0
for f in $GATES; do
  name="$(basename "$f")"
  code="$(strip_comments "$f")"
  codefile="$(mktemp "${TMPDIR:-/tmp}/gate_inv_code.XXXXXX")"
  printf '%s\n' "$code" >"$codefile"

  # Build guard: report the ACTUAL cmake target, not just "yes".
  target="$(printf '%s\n' "$code" | sed -nE 's/^[[:space:]]*gate_require_build[[:space:]]+([A-Za-z0-9_]+).*/\1/p' | head -1)"
  if [ -n "$target" ]; then build="yes"; else build="NONE"; fi

  # Precondition mechanism, from CODE.
  gran="$(granularity_of "$codefile")"
  gran_raw="$gran"   # kept: the loop rewrites "none" to "-" for display, and the
                    # NEEDS-GPU test below must see the CLASS, not the display form
  if [ "$gran" = "WHOLE-gate" ]; then
    precond="gate_require_gpus"
  elif [ "$gran" = "BROKEN" ]; then
    precond="gate_require_gpus(?)"
  elif [ "$gran" = "per-arm" ]; then
    precond="gate_gpu_count"
    # A per-arm gate that never records what it skipped is a silent skip wearing
    # a per-arm hat: it would exit 0 on a 1-GPU box while claiming the shape.
    printf '%s\n' "$code" | grep -qE 'SKIPPED_(TP2|SUB)|PARTIAL' || {
      precond="$precond(NO-SKIP-REPORT)"; gran="BROKEN"; }
  else
    precond="none"
  fi

  if [ "$gran" = "none" ]; then
    gran="-"; arm="-"
  else
    arm="$(has_1gpu_arm "$codefile")"
  fi

  if [ "$build" = "NONE" ]; then
    mark=" <-- NO BUILD GUARD"
    VIOLATIONS=$((VIOLATIONS + 1))
  else mark=""; fi

  # --- NEEDS-GPU vs HAS-A-GUARD (the class AUDIT #15 closed) ------------------
  # GRANULARITY conflates two different things: "this gate needs no device" and
  # "this gate needs a device and forgot to say so". Both classify as "none",
  # and on a healthy 1-GPU box both exit 0 -- so the distinction is invisible
  # until the second GPU goes away, at which point the second kind goes RED and
  # is indistinguishable from a regression.
  #
  # Re-derived here, not from a remembered list, so a newly added GPU gate is
  # classified without anyone editing a table. That re-derivation is what found
  # them: test_dpas_fold.sh (which said "needs a GPU" in a COMMENT and carried
  # no guard), test_http_batch.sh and test_serve_load.sh (both launch the server
  # `--gpu --tp 1`). All three passed here, which is exactly why the gap was
  # invisible until the list was rebuilt from the gates themselves.
  # RENDER-NODE column, one of four tokens and never a bare "-":
  #   via-count    a COUNT precondition (gate_gpu_count/gate_require_gpus) already
  #                covers the device check, so the render-node probe is redundant.
  #                Naming it beats "-", which a reader cannot tell from "unchecked".
  #   render-node  no count precondition, but a render-node probe is present.
  #   not-gpu      the gate needs no device.
  #   MISSING      the gate needs a device and has NO guard -> VIOLATION.
  if [ "$gran_raw" = "none" ]; then
    rnode="$(gpu_guard_state "$codefile")"
    if [ "$rnode" = "MISSING" ]; then
      mark="$mark <-- NEEDS A GPU BUT HAS NO GUARD (goes red on a 0-GPU box)"
      VIOLATIONS=$((VIOLATIONS + 1))
    fi
  else
    rnode="via-count"
  fi

  rm -f "$codefile"

  # Accumulate the EXPECTED exit class for this gate, from source. This is what
  # makes the doc-tally reconciliation below possible without running anything.
  case "$gran" in
    "-")         N_PASS=$((N_PASS + 1)) ;;
    per-arm)     N_PARTIAL=$((N_PARTIAL + 1)) ;;
    WHOLE-gate)  N_SKIP=$((N_SKIP + 1)) ;;
    BROKEN)      # Two distinct unclassifiable shapes; both are violations and
                 # both are counted as a violation, never quietly as a pass.
                 N_PASS=$((N_PASS + 1)); VIOLATIONS=$((VIOLATIONS + 1))
                 case "$precond" in
                   *"(?)"*) mark="$mark <-- gate_require_gpus not a line-start statement: cannot classify" ;;
                   *)       mark="$mark <-- per-arm guard that never reports the skip" ;;
                 esac;;
  esac

  # THE INVARIANT: a whole-gate skip must not strand arms that would still run.
  if [ "$gran" = "WHOLE-gate" ] && [ "$arm" = "yes" ]; then
    mark="$mark <-- OVER-SKIP: 1-GPU arms stranded (audit #13)"
    VIOLATIONS=$((VIOLATIONS + 1))
  fi

  printf '%-26s %-9s %-22s %-11s %-12s %s%s\n' "$name" \
    "$([ "$build" = NONE ] && echo "-" || echo yes)" "$precond" "$gran" "$arm" "$rnode" "$mark"
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
. "$SCRIPT_DIR/_gate_precond.sh"
# `|| NDEV=0`, and stderr is NOT redirected: the helper's "cannot count" note is
# the only thing distinguishing "box has no GPU" from "we cannot see the box"
# (audit #14). A 0 here must not be silently indistinguishable from a failure to
# measure.
NDEV="$(gate_gpu_count)" || NDEV=0
DOCS="AGENTS.md $(ls reports/*.md 2>/dev/null)"

if [ "${NDEV:-0}" -lt 1 ]; then
  echo "doc-tally: SKIPPED (${NDEV} level_zero GPU(s) visible -- cannot derive the expected rc classes)."
  echo "  Loud, not silent: a precondition that cannot see the box verifies nothing."
  echo "  (This is the libumf LD_LIBRARY_PATH gotcha, or setvars not sourced --"
  echo "   not a code problem. Source /opt/intel/oneapi/setvars.sh and re-run.)"
  echo "  EXIT 2 (SKIP), never 0: an OK line here would claim a reconciliation"
  echo "  that was never performed."
  exit 2
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
      # Separator is EITHER ", " or " / ": the slash form appears in live prose
      # ("7 PASS / 2 PARTIAL / 1 SKIPPED") and was detected as a site but parsed
      # with a comma-only pattern, so a CORRECT line was reported STALE. A site
      # the checker finds but cannot parse is worse than not finding it.
      p="$(printf '%s' "$txt"  | sed -nE 's/.*[^0-9]([0-9]+) PASS.*/\1/p')"
      pa="$(printf '%s' "$txt" | sed -nE 's/.*[,/] ?([0-9]+) PARTIAL.*/\1/p')"
      sk="$(printf '%s' "$txt" | sed -nE 's/.*[,/] ?([0-9]+) SKIPPED.*/\1/p')"
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
  # Routed through the SHARED vacuity guard (scripts/_gate_enum.sh) rather than
  # a second hand-rolled copy of the same rule -- the audit #7/#8 "rule fixed at
  # one site, re-derived at another" shape. Verified to behave identically here:
  # rc=1 with sites_found=0, and silent on the normal path.
  enum_reset
  if enum_require_nonempty "tally-sites" "$sites_found" "doc tally sites"; then
    :
  else
    VIOLATIONS=$((VIOLATIONS + 1))
  fi

  # --- HAND-COUNTED TOTALS must be citations, not live claims ---------------
  # The tally above is DERIVED (expected rc per gate from this file's own
  # classification), so it can be reconciled against the source. A hand-counted
  # total -- "45/45 scripts", "12 baselines", "29 artifact consumers" -- has no
  # such source: it is a snapshot of one moment. Those are legitimate HISTORICAL
  # records and must NOT be rewritten to match today (rewriting a true record is
  # the same laundering failure in the other direction). The defect is different
  # and narrower: presenting one as a claim about the CURRENT tree.
  #
  # That is not hypothetical. A session summary said "43/43 scripts" in one place
  # and "45/45" in another in the same breath, and an AGENTS.md sentence still
  # claimed "11 baselines" after N-Kq/N-L made it 12 -- which then went stale the
  # moment it was corrected. Cost one commit (e482073) chasing a number nobody
  # needed. A number in prose is a CITATION.
  #
  # Three exclusions, all measured BEFORE writing the check, so the rule is known
  # to converge rather than to fire on everything:
  #   (a) paragraph-scoped history, using this checker's OWN markers from the
  #       tally loop above (audit-#, COUNTING TRAP) -- so the two sections cannot
  #       disagree about what counts as history;
  #   (b) an explicit citation marker in the same paragraph (HISTORICAL,
  #       superseded, "read from", ...);
  #   (c) a QUOTED claim, i.e. the doc is refuting the number rather than making
  #       it. Without (c) the one live hit was '"4 gates is enough"' -- a claim
  #       being argued against, and flagging it would be a false positive.
  # Measured on the corpus as it stood: 20 sites, 3 cited, 14 paragraph-history,
  # 1 quoted-and-refuted, 0 live. Sites are counted and asserted NON-ZERO, so a
  # regex that matches nothing cannot pass as "0 stale".
# The counting is done in python, not shell. The bash version word-split the
  # matched phrase ("3 gates" -> "3", "gates"), so it reported 25 live sites on a
  # corpus that measures 0 -- every one a false positive. Prose matching is
  # exactly the case where a shell word loop is the wrong tool.
  hc_out="$(python3 - "$ROOT" <<'PYEOF'
import os, re, sys
ROOT = sys.argv[1]
docs = [os.path.join(ROOT, "AGENTS.md")]
rd = os.path.join(ROOT, "reports")
if os.path.isdir(rd):
    docs += [os.path.join(rd, f) for f in sorted(os.listdir(rd)) if f.endswith(".md")]

# The phrase is ONE token: number, optional /denominator, then the noun.
PAT = re.compile(r'\b\d+(?:/\d+)?\s+(?:scripts\b|baselines?\b|artifact consumers\b|gates\b)')
HIST = re.compile(r'audit-#|COUNTING TRAP', re.I)
CITED = re.compile(r'HISTORICAL|CITATION|at the time|superseded|re-derive|read .* from', re.I)

tot = live = hist = cited = quoted = 0
for path in docs:
    if not os.path.isfile(path):
        continue
    rel = os.path.relpath(path, ROOT)
    lines = open(path, errors="replace").read().splitlines()
    i = 0
    while i < len(lines):
        if not PAT.search(lines[i]):
            i += 1
            continue
        j = i
        while j > 0 and (lines[j-1].startswith("  ") or lines[j-1].startswith("- ")
                         or not lines[j-1].strip()):
            j -= 1
        k = i
        while k + 1 < len(lines) and (lines[k+1].startswith("  ")
                                      or lines[k+1].startswith("- ")):
            k += 1
        para = "\n".join(lines[j:k+1])
        for m in PAT.finditer(para):
            tot += 1
            if HIST.search(para):
                hist += 1
                continue
            if CITED.search(para):
                cited += 1
                continue
            # A QUOTED claim is one the doc REFUTES, not makes. Without this the
            # single live hit was '"4 gates is enough"' -- argued against in
            # reports/2026-09-30-dpas48-fold.md -- so flagging it is a false
            # positive. Quoted spans are excluded explicitly rather than by
            # loosening the pattern.
            q = re.compile(r'["\u201c\u201d][^"\u201c\u201d]*' + re.escape(m.group(0)))
            if q.search(para):
                quoted += 1
                continue
            live += 1
            print("           LIVE %s:%s  hand-counted total '%s' is not marked a citation"
                  % (rel, i + 1, m.group(0)))
        i = k + 1
print("           handcounted_sites=%d history=%d cited=%d quoted=%d live=%d"
      % (tot, hist, cited, quoted, live))
print("HC_LIVE=%d" % live)
print("HC_TOTAL=%d" % tot)
PYEOF
)"
  printf '%s\n' "$hc_out" | grep -v '^HC_' || true
  hc_live="$(printf '%s'   "$hc_out" | sed -n 's/^HC_LIVE=//p')"
  hc_total="$(printf '%s'  "$hc_out" | sed -n 's/^HC_TOTAL=//p')"
  # Both verdicts must be PRESENT: a python failure yields empty stdout, and
  # `[ "" -eq 0 ]` is a bash error whose result is FALSE -- so without this the
  # check would silently pass having verified nothing. Same class as
  # gate_require_device's silent no-op, one function over.
  if [ -z "$hc_live" ] || [ -z "$hc_total" ]; then
    echo "           <-- hand-counted check produced NO verdict (python failed?)" >&2
    VIOLATIONS=$((VIOLATIONS + 1))
  elif [ "$hc_total" -eq 0 ]; then
    echo "           <-- VACUOUS: matched 0 hand-counted-total sites, so this check" >&2
    echo "               verified nothing and would report 0 live by default." >&2
    VIOLATIONS=$((VIOLATIONS + 1))
  else
    VIOLATIONS=$((VIOLATIONS + hc_live))
  fi
fi

echo
if [ "$VIOLATIONS" = "0" ]; then
  # The OK line may claim a reconciliation ONLY because reaching this line
  # requires the tally to have run: the 0-GPU path exits 2 above, so there is no
  # longer a branch in which a skip can print "doc tallies reconcile". Kept as
  # an explicit assertion anyway, so a future edit that reintroduces a soft
  # skip cannot also reintroduce the overclaim.
  if [ "${TALLY_RAN:-0}" != "1" ]; then
    echo "GATE INVENTORY: BUG -- reached the OK line with the tally unrun" >&2
    exit 1
  fi
  echo "GATE INVENTORY: OK (every gate builds its own target; no whole-gate skip strands 1-GPU arms; doc tallies reconcile)"
  echo "  NOTE what 'reconcile' means: the doc NUMBERS agree with what the source"
  echo "  classification predicts for this box's GPU count. It is not a re-run of"
  echo "  the gates, so it cannot detect a gate that is green-numbered but red."
  exit 0
fi
echo "GATE INVENTORY: $VIOLATIONS PROBLEM(S) -- see the <-- markers and STALE lines above" >&2
exit 1
