#!/usr/bin/env bash
# Assert the ENGINE certification on disk is still evidence for this tree.
#
# WHY (AUDIT #24). `run_all_checks.sh` certifies the TOOLING. `scripts/
# run_ten_gates.sh` certifies the ENGINE and writes build/ten_gates/HEAD naming
# the tree it measured. For a while the two were only ever reconciled BY HAND, and
# a session reported "10 PASS / 0 PARTIAL / 0 SKIPPED" justified by an argument --
# "the files that changed since the gates ran are not engine inputs". That
# argument was sound and it was STILL NOT EVIDENCE: it is AUDIT #22's shape, where
# a number's backing was a claim about the tree instead of a file on disk.
#
# This mechanises exactly that argument, so it cannot be forgotten or skipped:
#   * artifact names HEAD                      -> certified here            (0)
#   * artifact older, but NO engine input moved -> the evidence CARRIES      (0)
#   * artifact older AND an engine input moved  -> STALE, re-run the gates   (1)
#   * no artifact at all                        -> UNPROVEN, never certified (2)
#
# The middle case is the whole point. A tooling-only commit must NOT invalidate a
# 45-minute engine certification -- that would make the stack unusable and would
# train people to ignore it, which is how such a check gets "fixed" by weakening
# it. Only a change to what the gates actually READ does.
#
# Deliberately NOT folded into gate_inventory.sh: that checker derives the gate
# TALLY from preconditions, while this asks a different question -- whether the
# measurement backing that tally is still about this code.
#
# Usage: scripts/checks/check_engine_cert.sh
# Exit: 0 = certified or carrying, 1 = stale (engine input changed), 2 = unproven.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2
# shellcheck source=scripts/lib/_gate_enum.sh
. "$SCRIPT_DIR/../lib/_gate_enum.sh"
enum_reset

# The artifact lives under the BUILD tree, and INTFER_BUILD_DIR is honoured for it
# like everywhere else in this repo. That is what makes the checker redirectable
# into a test sandbox: without the seam the only way to exercise it is to mutate
# the REAL certification artifact, which would leave a planted file behind on any
# abnormal exit and make the next session's stack report a run it never measured.
ART="${INTFER_BUILD_DIR:-$ROOT/build}/ten_gates/HEAD"
echo "== ENGINE CERTIFICATION CHECK =="

# What the correctness gates actually read. Derived from the gate population
# itself: scripts/gates/test_*.sh, plus the engine and the shared helpers those gates
# source. A hand-written list here would rot the first time a gate gains a helper.
GATE_POP="$(ls "$SCRIPT_DIR"/../gates/test_*.sh 2>/dev/null | wc -l | tr -d ' ')"
enum_require_min "gate-scripts-present" "$GATE_POP" 5 || {
  echo "  FAIL  fewer than 5 gate scripts found -- the input set is not derivable"
  echo "RESULT: cannot certify; the engine input set did not enumerate"
  exit 1; }

if [ ! -f "$ART" ]; then
  echo "  note  no $ART -- the engine has never been certified on this box"
  echo "  note  run scripts/run_ten_gates.sh (tens of minutes, needs both GPUs)"
  echo "RESULT: UNPROVEN -- no engine certification on disk (not a regression)"
  exit 2
fi

# The git side must work from the real tree, so refuse early if it does not --
# otherwise a non-repo checkout silently produces an EMPTY changed-file list,
# which would make a STALE artifact look like it "carries".
if ! git rev-parse HEAD >/dev/null 2>&1; then
  echo "  FAIL  not a git work tree -- the staleness comparison cannot run"
  echo "RESULT: cannot compare the certification to a tree"
  exit 1
fi

CERT_SHA="$(sed -n 's/^HEAD=//p' "$ART" | head -1)"
CERT_SHORT="$(printf '%s' "$CERT_SHA" | cut -c1-7)"
enum_require_min "cert-sha-length" "${#CERT_SHA}" 7 || {
  echo "  FAIL  $ART does not name a usable HEAD"
  echo "RESULT: engine certification artifact is malformed"
  exit 2; }
# *** THE SHA MUST RESOLVE, NOT MERELY BE THE RIGHT SHAPE. *** This guard did not
# exist and the checker CERTIFIED on an unresolvable one. `enum_require_min` above
# only asks "is it at least 7 characters long", which a 40-zero sha satisfies; then
#   CHANGED="$(git diff --name-only "$CERT_SHA"..HEAD 2>/dev/null || true)"
# exits 128 ("fatal: Invalid revision range"), writes nothing to stdout, and `|| true`
# threw the failure away -- so n_touched became 0 and the verdict was "carries", rc=0,
# printed as "0 file(s) changed ... (List: )". Measured on the unpatched tree.
#
# That is the AUDIT #14/#20 vacuous-green class, and it is the SAME hole the guard
# 20 lines above was written to close: "a non-repo checkout silently produces an
# EMPTY changed-file list, which would make a STALE artifact look like it carries."
# That reasoning was applied to `git rev-parse HEAD` and not to the revision it is
# actually about. A rule fixed at one site and merely assumed at the adjacent one --
# AUDITs #3, #12 and #19's shape, in the checker added by AUDIT #24 itself.
#
# Reachable: INTFER_BUILD_DIR is a DOCUMENTED override (README), so pointing it at a
# tree certified by a different clone, a shallow checkout, a rebased history or a
# hand-edited artifact all land here -- and the answer must not be "certified".
#
# rc=2 = UNPROVEN, not 1. Same distinction scripts/gpu_recover.sh already draws
# (2 = UNMEASURABLE, explicitly NOT "1 device"): an artifact that cannot be located in
# this repository is neither a stale certification (1) nor a certification (0). It is
# the absence of evidence, and run_all_checks.sh already routes 2 into its UNPROVEN
# bucket whose remedy is "re-run the runner".
if ! git cat-file -e "${CERT_SHA}^{commit}" 2>/dev/null; then
  echo "  FAIL  the artifact names ${CERT_SHORT:-<none>}, which does not RESOLVE to a"
  echo "        commit in THIS repository. A certification of a tree this clone does"
  echo "        not contain cannot be compared to HEAD, and must not be carried."
  echo "        (Usually a stale INTFER_BUILD_DIR pointing at another clone's tree, a"
  echo "        shallow/rebased history, or a hand-edited artifact.)"
  echo "RESULT: UNPROVEN -- the certification artifact does not describe this repository"
  exit 2
fi
CERT_TALLY="$(sed -n 's/^pass=//p' "$ART" | head -1)"
echo "  artifact: $ART"
echo "  certified tree: $CERT_SHORT   tally: pass=$CERT_TALLY ..."

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [ "$CERT_SHA" = "$HEAD_SHA" ]; then
  echo "  ok    the certification names HEAD exactly -- no argument needed"
  echo "RESULT: engine certified at HEAD (pass=$CERT_TALLY)"
  exit 0
fi

# ---------------------------------------------------------------------------
# content_changed_paths < <(git diff --numstat A..B)
# Prints the subset of changed paths whose CONTENT changed. A mode-only change
# (chmod) reports `0<TAB>0` and is dropped; anything else is kept.
#
# The binary case is why this is NOT written as "non-zero": git prints `-` for
# both columns of a binary file, so a "non-zero" test would read `-` as zero and
# silently classify a changed binary as UNTOUCHED. The test is therefore
# "NOT (0 and 0)". Getting this backwards is the classic vacuous-green shape --
# a guard that reports nothing to fix while the thing it guards moved.
# ---------------------------------------------------------------------------
content_changed_paths() {
  awk -F'\t' 'NF>=3 && !($1=="0" && $2=="0") {print $3}'
}

# *** BEHAVIOURAL SELF-CHECK, FIRST AND GATING. *** This weakens a guard (it now
# ignores mode-only changes), and AUDIT #21's rule is that a weakening must be
# PROVABLY total, not merely weaker -- E-D in this matrix is the precedent of a
# "weaker" filter that was not weak enough to observe. The matrix cannot test
# this one: it plants shas in the REAL repository's history, and manufacturing a
# mode-only commit there would mean committing to the user's clone to test a
# checker. So the classification itself is driven directly, in three cases, and
# all three must pass before the verdict below is allowed to use it.
_sc_bad=0
_sc_case() { # $1 label, $2 expected-RE, $3 numstat text, $4 expected path or empty
  local got
  got="$(content_changed_paths <<<"$3")"
  if [ -n "$4" ]; then
    printf '%s' "$got" | grep -qxF -- "$4" || {
      echo "  FAIL  self-check: $1 -- expected '$4', got [$(printf '%s' "$got" | tr '\n' ' ')]" >&2
      _sc_bad=$((_sc_bad + 1)); return; }
  else
    [ -z "$got" ] || {
      echo "  FAIL  self-check: $1 -- expected NO path, got [$(printf '%s' "$got" | tr '\n' ' ')]" >&2
      _sc_bad=$((_sc_bad + 1)); return; }
  fi
  echo "  ok    self-check: $1"
}
_sc_case "a mode-only change is NOT content"        ''  "$(printf -- '0\t0\tscripts/lib/_gate_precond.sh')" ''
_sc_case "a real edit IS content (the converse)"    'scripts/lib/_gate_precond.sh' "$(printf -- '3\t0\tscripts/lib/_gate_precond.sh')" 'scripts/lib/_gate_precond.sh'
_sc_case "a deletion IS content"                    'src/gemm/quant_gemm.cpp'  "$(printf -- '0\t7\tsrc/gemm/quant_gemm.cpp')" 'src/gemm/quant_gemm.cpp'
# The case that would be silently WRONG with a "non-zero" test: git prints `-`
# for a binary's columns, and a changed binary is certainly a content change.
# NOTE the `printf --`: a format string beginning with `-` is parsed as an OPTION
# by bash's printf and silently yields nothing. Found by this very self-check --
# the classifier was right and the FIXTURE was lying, which is the same shape as
# the recorded "evidence about the tool, not the logic" traps (grep -E with an
# unescaped `(`, `ls` hiding /dev/dri entries, `ps` reporting nothing for a live
# job). The self-check earned its keep on its first run.
_sc_case "a BINARY change is content, not mode-only" 'src/model.bin'          "$(printf -- '-\t-\tsrc/model.bin')" 'src/model.bin'
if [ "$_sc_bad" -ne 0 ]; then
  echo "RESULT: the mode-vs-content classifier is not trustworthy; refusing to grade"
  exit 1
fi

# Older than HEAD. Does the difference touch anything a gate reads?
#
# The rc is checked EXPLICITLY rather than swallowed with `|| true`, because git's
# failure mode here is SILENT: `git diff <bad>..HEAD` exits 128, writes NOTHING to
# stdout, and `|| true` discarded that -- an unknown difference then read as a clean
# tree. That is the unresolvable-sha defect above, one line down.
#
# *** AND THE OBVIOUS anti-vacuity GUARD IS DELIBERATELY NOT HERE. *** The first
# version of this fix added enum_require_nonempty on the changed-file count, which
# looks like the AUDIT #20 rule and is WRONG here. Measured: for two commits with the
# same tree, `git diff --name-only` emits a single "\n" and nothing else, so a
# legitimately clean comparison harvests zero files -- and two distinct shas with an
# IDENTICAL tree is a case where carrying is CORRECT (the certified source IS this
# source). The guard would have refused valid evidence: over-assertion is how a
# correctness gate becomes a flaky gate, and a false refusal trains people to
# disable it. The rc check is the precise guard, because it distinguishes "the query
# did not run" from "the query ran and found nothing" -- which a count cannot.
if ! CHANGED="$(git diff --name-only "$CERT_SHA..HEAD" 2>/dev/null)"; then
  echo "  FAIL  git diff $CERT_SHORT..HEAD failed; the changed-file set is unknown."
  echo "        An UNKNOWN difference is not a clean tree -- refusing to certify."
  echo "RESULT: UNPROVEN -- the certification could not be compared to HEAD"
  exit 2
fi
# *** A MODE CHANGE IS NOT A CONTENT CHANGE, AND THE DIFFERENCE IS FREE. ***
# Measured 2026-10-03: giving every scripts/*.sh the execute bit (closing the
# exec-bit class) put scripts/lib/_gate_build.sh and scripts/lib/_gate_precond.sh into
# `git diff --name-only`, so this checker declared a 10/10 ENGINE certification
# STALE and demanded a 45-minute re-run -- for a change that cannot alter a
# single byte of what those gates execute. This is the failure AGENTS.md names
# explicitly: "a check that fires on every commit is a check people learn to
# ignore". Left alone it would fire on every chmod, forever, and each firing
# costs 45 minutes to discharge.
#
# `git diff --numstat` distinguishes them for free: a mode-only change reports
# `0<TAB>0`, a content change reports non-zero. BINARY files report `-` for both
# columns, so the test is "NOT (0 and 0)" rather than "non-zero" -- otherwise a
# changed binary would be silently read as untouched. The rc of the numstat call
# is checked explicitly, for the same silent-failure reason as the diff above.
if ! CONTENT="$(git diff --numstat "$CERT_SHA..HEAD" 2>/dev/null)"; then
  echo "  FAIL  git diff --numstat $CERT_SHORT..HEAD failed; whether any change is a"
  echo "        CONTENT change is unknown. Refusing to certify."
  echo "RESULT: UNPROVEN -- the certification could not be compared to HEAD"
  exit 2
fi
CONTENT_PATHS="$(content_changed_paths <<<"$CONTENT")"
CHANGED_CONTENT="$(printf '%s\n' "$CHANGED" | while IFS= read -r p; do
  [ -n "$p" ] || continue
  if printf '%s\n' "$CONTENT_PATHS" | grep -qxF -- "$p"; then printf '%s\n' "$p"; fi
done)"
TOUCHED="$(printf '%s\n' "$CHANGED_CONTENT" \
  | grep -E '^(src/|examples/|include/|scripts/gates/test_|scripts/lib/_gate_precond\.sh|scripts/lib/_gate_build\.sh|scripts/setvars\.sh)' || true)"
n_touched="$(printf '%s\n' "$TOUCHED" | grep -c . || true)"
n_mode_only=$(( $(printf '%s\n' "$CHANGED" | grep -c . || true) - $(printf '%s\n' "$CHANGED_CONTENT" | grep -c . || true) ))
[ "$n_mode_only" -gt 0 ] && \
  echo "  ok    $n_mode_only path(s) changed MODE only (e.g. chmod); content is" \
       "byte-identical, so no gate input moved"

if [ "$n_touched" -eq 0 ]; then
  n_changed="$(printf '%s\n' "$CHANGED" | grep -c . || true)"
  echo "  ok    certified at $CERT_SHORT, HEAD is ${HEAD_SHA:0:7}; $n_changed file(s) changed"
  echo "        and NONE is an engine or gate input, so the evidence CARRIES."
  echo "        (List: $CHANGED)" | tr '\n' ' '; echo
  echo "RESULT: engine certification carries to HEAD -- no engine input changed"
  exit 0
fi

echo "  FAIL  the engine certification is STALE."
echo "        Certified at $CERT_SHORT; these engine/gate inputs changed since:"
printf '%s\n' "$TOUCHED" | sed 's/^/          /'
echo "        Re-run scripts/run_ten_gates.sh to re-certify. Until then the"
echo "        'N PASS' tally is evidence for $CERT_SHORT, not for this tree."
echo "RESULT: engine certification STALE -- engine input changed since it was measured"
exit 1