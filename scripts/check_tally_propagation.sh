#!/usr/bin/env bash
# check_tally_propagation.sh -- SOURCE-level certification that a prose-recorded matrix
# or checker tally is re-derived by the SCRIPT THE CLAIM IS ABOUT.
#
# WHY THIS EXISTS, and it is a different question from check_durable_claims.sh. That
# checker asks "does this tally carry a REGEN or an EPHEMERAL at all" -- and
# p_target_exists asks that the named script EXISTS. Neither asks whether the marker
# names the RIGHT script. Measured: a line reading
#     `| mutate_fold_matrix.sh | PASS -- 7/7 states caught ... REGEN(bash scripts/mutate_tp2_pass_exit.sh)`
# passes BOTH of them -- the marker is present and the script exists -- while the
# disposition re-derives a different claim entirely. That is the propagation defect:
# the re-measurement was not propagated to the site that records it, so a future session
# re-runs one matrix and believes it has re-derived another's tally.
#
# THE RULE, and there is no third state. For every harvested tally site that NAMES a
# script, the disposition in its window must be either
#   EPHEMERAL(<why>)              -- the record says outright that it cannot be re-run, or
#   REGEN(<cmd naming THAT script>) -- the command re-runs the very harness whose tally
#                                   the line records ("a self-re-running equivalence half":
#                                   the mutator prints its own tally, so re-running it IS
#                                   the re-derivation of the figure on this line).
# A NAMED site with no disposition, or with a REGEN naming a different script, is REFUSED.
# A site that names no script is COUNTED and skipped -- the rule cannot be applied to it,
# and pretending otherwise would be the vacuity this file's own guard refuses.
#
# THE RULE'S TWO MEASURED LIMITS, recorded rather than implied, because a checker that
# overstates itself is the defect it exists to catch:
#   (1) THE SUBJECT IS THE FIRST SCRIPT NAMED ON THE LINE. On a line that names several
#       scripts that is an approximation, and a same-line-only rule built on it produced a
#       FALSE POSITIVE on a healthy tree: engines_sources.txt:35 names
#       scripts/check_s1_predicates.sh first (the entry REGISTERS the checker) while the
#       figure on that line is the MATRIX's "8/8 states CAUGHT", whose correct marker IS
#       REGEN(bash scripts/mutate_s1_predicates.sh) on the same line. Crying wolf there
#       would teach a reader to ignore this sweep, so the rule is window-scoped instead.
#   (2) THE DISPOSITION IS READ FROM A +/-1 LINE WINDOW, the same window
#       check_durable_claims.sh uses, so a marker wrapped onto its own line still counts.
#       Same-line-only was measured at 24/25 named sites (the one disagreement being (1));
#       the window is 25/25. The residual is that a neighbour's correct marker can satisfy
#       a line whose own marker names the wrong script -- narrowed by TP-wrong-target, not
#       closed, and stated here so nobody reads more into a PASS than the rule supports.
#
# NO EXECUTION OF ANY SUBJECT. This sweeps text only: no GPU, no model, no build, and no
# mutator is run.
#
# THE HARVEST IS PRINTED. `sites=`, `named=`, `self=`, `ephemeral=` and `unnamed=` are
# emitted every run so "0 findings" is distinguishable from "looked at nothing": the
# vacuity guard refuses a run that named no script at all, which is the only way a
# green verdict here could certify nothing.
set -u
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
die() { echo "TALLY-PROP: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }
note() { printf 'TALLY-PROP: %-14s %s\n' "$1" "$2"; }
fails=()
haverefuse() { printf '%s\n' "${fails[@]:-}" | grep -qx "$1"; }

HIST="$ROOT/HISTORY.md"
[ -f "$HIST" ] || die "durable record missing ($HIST)"

# THE SAME SURFACE check_durable_claims.sh certifies, derived the same way and from the
# same glob, so a record added to docs/ is covered on the day it lands in BOTH. This is a
# deliberate duplicate rather than a source() of the other checker: sourcing it would
# EXECUTE it, and a sweep whose verdict depends on a second checker's exit status reports
# that checker's failure as its own refusal.
CORPUS=("$HIST" "$ROOT/engines_sources.txt" "$ROOT/AGENTS.md")
for _d in "$ROOT"/docs/*.md; do
  [ -f "$_d" ] && CORPUS+=("$_d")
done
[ "${#CORPUS[@]}" -ge 3 ] || die "the durable surface did not resolve (${#CORPUS[@]} record(s))"

# A tally site is the same two shapes check_durable_claims.sh harvests, so the two
# checkers look at the SAME lines and a line cannot be harvested by one and missed by the
# other. Kept byte-identical on purpose; a future widening has to move both.
SITE_RE='[0-9]+/[0-9]+ (states )?caught|PASS[ ,(]+\(?[0-9]+/[0-9]+'
NAME_RE='(mutate|check)_[a-z0-9_]+\.sh'
DISP_RE='REGEN\([^)]*\)|EPHEMERAL\([^)]*\)'

sites=0; named=0; self_n=0; eph_n=0; unnamed=0
for f in "${CORPUS[@]}"; do
  while IFS= read -r ln; do
    [ -n "$ln" ] || continue
    sites=$((sites + 1))
    body="$(sed -n "${ln}p" "$f" 2>/dev/null)"
    subj="$(printf '%s\n' "$body" | grep -oE "$NAME_RE" | head -1)"
    if [ -z "$subj" ]; then
      unnamed=$((unnamed + 1))
      continue
    fi
    named=$((named + 1))
    win="$(sed -n "$((ln - 1)),$((ln + 1))p" "$f" 2>/dev/null)"
    disp="$(printf '%s\n' "$win" | grep -oE "$DISP_RE" | head -1)"
    if [ -z "$disp" ]; then
      note p_tally_propagation "FAIL -- $f:$ln records the tally of $subj with NO"
      note "" "        disposition at all:"
      note "" "        $(printf '%s' "$body" | cut -c1-100)"
      fails+=(p_tally_propagation)
      continue
    fi
    case "$disp" in
      EPHEMERAL*) eph_n=$((eph_n + 1)); continue ;;
    esac
    tgt="$(printf '%s\n' "$disp" | grep -oE "$NAME_RE" | head -1)"
    if [ "$tgt" = "$subj" ]; then
      self_n=$((self_n + 1))
    else
      note p_tally_propagation "FAIL -- $f:$ln records the tally of $subj but its"
      note "" "        disposition re-derives \"${tgt:-<no script named>}\":"
      note "" "        $disp"
      note "" "        (the re-measurement was not propagated to the site that"
      note "" "        records it, so re-running the marker proves nothing here)"
      fails+=(p_tally_propagation)
    fi
  done <<< "$(grep -niE -o "$SITE_RE" "$f" 2>/dev/null | cut -d: -f1 | sort -un)"
done

note p_tally_propagation "harvest: sites=$sites named=$named self=$self_n ephemeral=$eph_n unnamed=$unnamed"

if [ "$sites" -lt 1 ]; then
  note p_tally_propagation "FAIL -- harvested ZERO tally sites; a sweep that examined"
  note "" "        nothing must not read as PASS"
  fails+=(p_tally_propagation)
elif [ "$named" -lt 1 ]; then
  note p_tally_propagation "FAIL -- no tally site names a script, so the rule could not"
  note "" "        be applied to a single line; a green verdict here would certify nothing"
  fails+=(p_tally_propagation)
elif ! haverefuse p_tally_propagation; then
  note p_tally_propagation "ok ($named named site(s) propagated or declared EPHEMERAL;"
  note "" "        $unnamed un-named site(s) counted and skipped by the rule's own limit)"
fi

if [ "${#fails[@]}" -eq 0 ]; then
  echo "TALLY-PROP: PASS (every named tally site is re-derived by the script it names, or declared EPHEMERAL)"
  echo "RESULT: PASS"
  exit 0
fi
echo "TALLY-PROP: FAIL -- refused by: ${fails[*]}"
echo "RESULT: FAIL"
exit 1
