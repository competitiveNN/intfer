#!/usr/bin/env bash
# Derive the BUILD-ARTIFACT CONSUMER population from the SOURCE OF TRUTH and
# fail when any consumer ignores INTFER_BUILD_DIR.
#
# WHY THIS EXISTS. AUDIT #8 closed 29 artifact consumers that hardcoded
# "$ROOT/build/..." instead of honoring the documented INTFER_BUILD_DIR
# override. Four commits later I reintroduced exactly that defect in
# check_l0_cite.sh / mutate_l0_cite.sh (six hardcoded sites). AUDIT #10 already
# recorded why the original sweep was incomplete: grepping for three artifact
# NAMES only rediscovers what the previous fix already knew. So this derives the
# population from (a) the CMake add_executable list and (b) every script that
# references a build tree, then classifies each one.
#
# A consumer that ignores the override silently BENCHMARKS OR VALIDATES THE
# WRONG TREE -- which is the precise failure this check exists to make
# impossible to reintroduce silently.
#
# Classification is per-script and evidence-based: a script is a consumer if it
# mentions a build tree at all. It PASSES when it routes every such reference
# through the override; it FAILS when it hardcodes $ROOT/build (or a literal
# build/ path) in a way the override cannot reach.
#
# Usage: scripts/checks/check_build_dir_consumers.sh
# Exit:  0 = every consumer honors the override, 1 = a consumer ignores it,
#        2 = nothing to check (would be a vacuous pass).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

# The shared enumeration guard (AUDIT #14's sites_found=0 class).
. "$SCRIPT_DIR/../lib/_gate_enum.sh"
enum_reset

echo "== BUILD-DIR CONSUMER CHECK =="

# ---- population, part 1: from the CMake target list ------------------------
# awk with a literal space field does NOT split on '(' -- 'add_executable(foo'
# is ONE field, so '{print $2}' silently yields an empty line for every match
# and this enumeration reported 0 targets, i.e. it would have exited 2 and
# looked like a working checker that simply found nothing. Split on '('
# explicitly. Same trap as reading a list off a remembered name.
TARGETS="$(grep -rhoE 'add_executable\([A-Za-z0-9_]+' --include=CMakeLists.txt . 2>/dev/null \
           | awk -F'(' '{print $2}' | LC_ALL=C sort -u)"
N_TARGETS="$(printf '%s\n' "$TARGETS" | grep -c .)"
echo "-- population part 1: $N_TARGETS CMake add_executable target(s)"
# Shared guard: an EMPTY harvest cannot certify "no consumers ignore the
# override". This branch previously carried its own hand-rolled copy of that
# rule; one home now, so the next enumerator cannot reintroduce the defect
# (the awk-splits-on-'(' bug that made this exact check harvest 0 targets and
# look healthy while verifying nothing).
enum_require_min "cmake-targets" "$N_TARGETS" 1 || {
  echo "RESULT: build-dir consumers ENUMERATION BROKEN (0 CMake targets)"
  exit 2
}

# ---- population, part 2: every script that references a build tree -----------
# *** CRITERION DISAGREEMENT WITH check_bench_load_gate.sh, DELIBERATE. ***
# This classifier does NOT strip comments before matching; its sibling
# scripts/checks/check_bench_load_gate.sh DOES. Both are right for their own question.
#
# This exact difference turned the stack red once (AUDIT #23): a usage comment in
# scripts/lib/_gate_loadlog.sh that spelled out a build-tree path was graded as a
# build-path consumer, and this checker plus its matrix both failed behind it. The
# fix applied then was to correct the COMMENT (the helper consumes no artifact at
# all -- its caller passes the directory), NOT to teach this classifier to strip.
# It happened a THIRD time immediately after, to the note written at the sibling's
# call site, so the paths here are described rather than quoted: a checker that
# spells out this classifier's literals gets graded by them.
#
# The rule that reconciles them: strip comments when the match decides MEMBERSHIP
# of a population, because a comment is not a behaviour. Here the match is only a
# hint -- the real test is the per-script override check below -- so a comment
# costs one extra NOTE row, not a wrong verdict. Over there the match IS the
# population, so an unstripped comment would add a script that publishes no
# figure. Both sites now classify comment-only matches EXPLICITLY (a NOTE row
# here, a stripped row there) rather than letting them fall through to a verdict
# about something else.
#
# *** CONVERSION TRIGGER -- do not harmonise these without one of these. ***
# This is the same deliberate-duplication pattern already used for
# patch_literal's three copies, and it carries the same conditions. Convert when
# ANY is true, ONE checker at a time, each with its own re-run and re-stamp:
#   * one side is edited and the edit lands on only one of them -- the copies have
#     then genuinely diverged and the shared helper is a bug source;
#   * a THIRD checker needs the same classification (two is a coincidence, three
#     is the hand-maintenance cost becoming real);
#   * one of the two is next being modified anyway, so the review cost is already
#     being paid.
# Until then the divergence is intentional and both sites say why.
CONSUMERS="$(grep -rlE '(INTFER_BUILD_DIR|\$ROOT/build|\$\{INTFER_BUILD_DIR|\$\{ROOT\}/build|build/)' \
             $(find scripts -maxdepth 2 -not -path '*/archive/*' -name '*.sh' -type f) 2>/dev/null | LC_ALL=C sort)"
CONSUMERS="$(grep -rlE '(INTFER_BUILD_DIR|\$ROOT/build|\$\{INTFER_BUILD_DIR|\$\{ROOT\}/build|build/)' \
             $(find scripts -maxdepth 2 -not -path '*/archive/*' -name '*.sh' -type f) 2>/dev/null | LC_ALL=C sort)"
N_CONSUMERS="$(printf '%s\n' "$CONSUMERS" | grep -c .)"
echo "-- population part 2: $N_CONSUMERS script(s) reference a build tree"
enum_require_min "consumer-scripts" "$N_CONSUMERS" 1 || {
  echo "RESULT: build-dir consumers ENUMERATION BROKEN (0 consumer scripts)"
  exit 2
}

# ---- classify each consumer -------------------------------------------------
bad=0; checked=0; notes=0
printf '\n%-34s %-8s %s\n' "script" "verdict" "evidence"
printf '%-34s %-8s %s\n' "----------------------------------" "--------" "------------------------------"
for f in $CONSUMERS; do
  b="$(basename "$f")"
  # Skip THIS script, loudly. It necessarily contains the pattern it searches
  # for, so it would otherwise always FAIL on its own regex. Excluded by NAME
  # with the reason stated, not by pattern -- an anonymous exclusion is how a
  # check stops checking. Counted so it cannot quietly grow the skip set.
  if [ "$b" = "$(basename "$0")" ]; then
    printf '%-34s %-8s %s\n' "$b" "skip" "self: contains the pattern it searches for"
    continue
  fi
  # Does it resolve a build dir through SOME documented override? There are
  # three; audit #10 fixed the build_quant_real tree to honor its OWN
  # (INTFER_QUANT_REAL_BIN) rather than INTFER_BUILD_DIR. Honoring only one
  # flagged 4 already-correct scripts -- a false alarm teaches people to ignore
  # the check, which is how such a check gets "fixed" by weakening it.
  # A COMMENT that names an override does not HONOUR it. Counting comment lines
  # here produced a vacuous PASS: a script whose only mention of
  # INTFER_BUILD_DIR sat in prose was graded "ok -- routes through a documented
  # override", which is the worse direction, because the misleading FAIL I was
  # fixing became a silently wrong ok. Found by planting the negative direction by
  # hand rather than by reasoning about it (the AUDIT #19 rule).
  # `grep -vc` prints 0 AND exits 1 when nothing matches, so `|| true` and NOT
  # `|| echo 0` -- appending 0 would make the count the two-line string "0\n0"
  # (AUDIT #14).
  honors="$(grep -nE 'INTFER_BUILD_DIR|INTFER_QUANT_REAL_BIN' "$f" 2>/dev/null \
            | grep -vcE ':[[:space:]]*#' || true)"
  # Root-relative build paths the override cannot reach. A line that READS an
  # override is the correct default-form and is excluded. Comment/PROSE lines
  # are not paths the script resolves (this very script quotes $ROOT/build in
  # the message telling other scripts how to fix themselves), so they are
  # excluded too -- otherwise the checker fails on its own help text.
  hard="$(grep -nE '(\$ROOT/build|\$\{ROOT\}/build)' "$f" \
          | grep -vE 'INTFER_BUILD_DIR|INTFER_QUANT_REAL_BIN' \
          | grep -vE ':[[:space:]]*#' | grep -v 'MUTANT payload' | wc -l)"
  # Population match vs CODE match. A script whose ONLY match is a COMMENT is not a
  # consumer: it resolves no path at run time. Without this distinction it fell
  # through to "references a build tree, honors NO override", which is what
  # happened TWICE this session, both times on documentation that spelled out a
  # build path -- once in a usage comment, once in the note recording this very
  # criterion split. The verdict was true-ish and completely misleading, and the
  # obvious "fix" (add a fake INTFER_BUILD_DIR reference) would have satisfied it
  # while fixing nothing.
  cmt="$(grep -nE '(\$ROOT/build|\$\{ROOT\}/build|build/|INTFER_BUILD_DIR)' "$f" 2>/dev/null \
         | grep -cE ':[[:space:]]*#' || true)"
  if [ "$honors" -eq 0 ] && [ "$hard" -eq 0 ] && [ "$cmt" -gt 0 ]; then
    notes=$((notes+1))
    printf '%-34s %-8s %s\n' "$b" "NOTE" \
      "matched on a COMMENT line only ($cmt) -- not a real consumer"
    continue
  fi
  checked=$((checked+1))
  if [ "$honors" -gt 0 ] && [ "$hard" -eq 0 ]; then
    printf '%-34s %-8s %s\n' "$b" "ok" "routes through a documented override"
  elif [ "$honors" -gt 0 ]; then
    printf '%-34s %-8s %s\n' "$b" "FAIL" "$hard hardcoded site(s) beside $honors override ref(s)"
    grep -nE '(\$ROOT/build|\$\{ROOT\}/build)' "$f" | grep -vE 'INTFER_BUILD_DIR|INTFER_QUANT_REAL_BIN' | grep -vE ':[[:space:]]*#' | grep -v 'MUTANT payload' | head -3 | sed 's/^/        /'
    bad=$((bad+1))
  else
    printf '%-34s %-8s %s\n' "$b" "FAIL" "references a build tree, honors NO override"
    grep -nE '(\$ROOT/build|\$\{ROOT\}/build)' "$f" | head -3 | sed 's/^/        /'
    bad=$((bad+1))
  fi
done

echo
# The classifier loop must actually have INSPECTED something. This branch used
# to report "PASS (0 consumer(s), all honor INTFER_BUILD_DIR)" -- a green
# verdict over an empty census -- when the self-exclusion was widened until it
# skipped everything. That is AUDIT #14's shape exactly, in a checker written to
# prevent it, so it gets the shared guard rather than a hand-rolled copy.
enum_require_nonempty "inspected-scripts" "$checked" "scripts classified" || {
  echo "RESULT: build-dir consumers VACUOUS PASS ($checked scripts inspected)"
  exit 1
}
echo "-- derived from CMake: $N_TARGETS target(s); scripts inspected: $checked"
if [ "$bad" -eq 0 ]; then
  echo
  echo "BUILD-DIR CONSUMER CHECK: PASS ($checked consumer(s), all honor INTFER_BUILD_DIR)"
  echo "RESULT: every artifact consumer honors the documented build-dir override"
  exit 0
fi
echo
echo "BUILD-DIR CONSUMER CHECK: FAIL ($bad/$checked ignore INTFER_BUILD_DIR)"
echo "        An artifact consumer that ignores the override silently benchmarks or"
echo "        validates the WRONG tree (AUDIT #8). Derive paths as"
echo "        BUILD_DIR=\"\${INTFER_BUILD_DIR:-\$ROOT/build}\" and use that everywhere."
echo "RESULT: build-dir consumer FAIL ($bad ignoring the override)"
exit 1