#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_scripts_live.sh.
#
# WHAT IS MUTATED, AND WHY IT IS THE RULE AND NOT A CORPUS. The artifact this
# checker judges is a CLASSIFICATION of a population, and on a clean tree the
# classification is correct by construction -- so a state that weakens the rule
# is unobservable for the same reason audit #21 records for mutate_fold_matrix.sh:
# there is nothing wrong in the tree to reveal it. The observable has to come from
# mutating the rule itself, in a SANDBOX COPY, and requiring a synthetic tree that
# the UNMUTATED checker classifies one way to be classified the other way by the
# mutant. The tracked scripts/ is never written to: the copy lives under $SB and
# is discarded, which also means the matrix is safe on a dirty tree and needs no
# git restore.
#
# The trees are built HERE, not by the checker's own plants, because the mutant
# runs with --scan-only. If the checker built its own tree the mutant would be
# scored on a fixture it also authors, and a rule that cannot classify anything
# would author a tree it happens to pass -- which is the vacuity class with an
# extra step.
#
# States (every one must be REJECTED, i.e. the mutant must DISAGREE with the
# baseline on the same tree -- a mutant that agrees is a MISS, and that comparison
# is the point: it is what makes each state a claim about this rule rather than
# about a tree):
#   SL-A  the docs/ exclusion dropped                     -> a CITATION becomes a
#          call site, so every retired script survives. This is the exact defect
#          that hid all sixteen scripts during the 2026-10-05 cleanup. `reports/`
#          was retired 2026-10-06 with the batch-and-test loop; the tree now only
#          holds `docs/` as a prose record, so the exclusion is a single site.
#   SL-B  the scripts/archive/ exclusion dropped         -> the retired set becomes
#          immortal, because each archived script keeps its siblings alive.
#   SL-C  the GLOB class disabled                       -> every gate reads dead,
#          because nothing NAMES a gate: `ls scripts/gates/test_*.sh` IS the gate
#          population. This is the class a NAMED-only rule loses.
#   SL-D  the verdict condition inverted                 -> a tree with an
#          unreachable script reports PASS.
#   SL-E  a dead script no longer COUNTS                 -> the same failure one
#          site earlier, which is the pair that proves the classification does the
#          catching rather than the verdict formatting coinciding with it.
#
# The direction that gets forgotten is the baseline: a fully wired tree must still
# PASS, or every state above is scored against a checker that fires on everything
# -- the always-red twin, which is how a check like this gets switched off rather
# than fixed.
#
# Exit: 0 = every state behaved as specified, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECK="$SCRIPT_DIR/../checks/check_scripts_live.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/scripts_live_mutate"
LOG="$BUILD_DIR/scripts_live_mutate"
SB="$(mktemp -d "${TMPDIR:-/dev/shm}/slivemut.XXXXXX")"
mkdir -p "$STAMP" "$LOG"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

# shellcheck source=scripts/lib/_gate_pred.sh
. "$SCRIPT_DIR/../lib/_gate_pred.sh"
# shellcheck source=scripts/lib/_gate_patch.sh
. "$SCRIPT_DIR/../lib/_gate_patch.sh"

# ---- the sandbox checker: a copy plus the helpers it sources --------------
# THE COPY KEEPS THE REAL LAYOUT, and the layout is load-bearing (MEASURED
# 2026-10-30): after the 2026-10-21 restructure the checker sources its helpers as
# `$SCRIPT_DIR/../lib/...`, so a copy dropped directly in $SB looked for
# /dev/shm/lib/_gate_pred.sh, died on `enum_require_nonempty: command not found`,
# and every state reported PATCH VERIFY FAIL for a reason that has nothing to do
# with the mutation. Staging the copy at $SB/stage/scripts/checks/ with the helpers
# at $SB/stage/scripts/lib/ makes SCRIPT_DIR resolve exactly as it does in the tree.
STAGE="$SB/stage/scripts"
CHECKCOPY="$STAGE/checks/check.sh"
restore_checker() { mkdir -p "$STAGE/checks" "$STAGE/lib"; cp "$CHECK" "$CHECKCOPY"; }
CHECK_SUM="$(cksum <"$CHECK")"
restore_checker
cp "$SCRIPT_DIR/../lib/_gate_pred.sh" "$STAGE/lib/_gate_pred.sh"
cp "$SCRIPT_DIR/../lib/_gate_enum.sh" "$STAGE/lib/_gate_enum.sh"

# ---- the trees -------------------------------------------------------------
# $1 = none | wired | prose | archive. Every tree names wired.sh from a src/ file
# so the ONLY unreachable script is the orphan under test.
mk_tree() {
  rm -rf "$SB/tree"; mkdir -p "$SB/tree/scripts/gates" "$SB/tree/docs" "$SB/tree/reports" "$SB/tree/src"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$SB/tree/scripts/gates/test_gate_a.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$SB/tree/scripts/gates/test_gate_b.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$SB/tree/scripts/wired.sh"
  printf '#!/usr/bin/env bash\nsource scripts/wired.sh\nexit 0\n' >"$SB/tree/scripts/caller.sh"
  printf '#!/usr/bin/env bash\nGATE_GLOB="${INTFER_GATE_GLOB:-test_*.sh}"\n' \
    >"$SB/tree/scripts/run_ten_gates.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$SB/tree/scripts/zz_orphan.sh"
  printf 'a finding naming scripts/wired.sh\n' >"$SB/tree/docs/history.md"
  printf '/* drives scripts/run_ten_gates.sh, which drives scripts/caller.sh */\nint main(void){return 0;}\n' \
    >"$SB/tree/src/entry.c"
  case "${1:-none}" in
    wired)   printf 'bash scripts/zz_orphan.sh\n' >>"$SB/tree/scripts/caller.sh" ;;
    prose)   printf 'a finding naming scripts/zz_orphan.sh\n' >>"$SB/tree/docs/history.md" ;;
    archive) mkdir -p "$SB/tree/scripts/archive"
             printf '#!/usr/bin/env bash\nbash scripts/zz_orphan.sh\n' \
               >"$SB/tree/scripts/archive/old.sh" ;;
    none)    : ;;
    *)       echo "mk_tree: unknown case '${1}'" >&2; exit 2 ;;
  esac
}
run_scan() { INTFER_SCRIPTS_DIR="$SB/tree/scripts" INTFER_SL_ROOT="$SB/tree" \
               bash "$CHECKCOPY" --scan-only 2>&1; }

: >"$LOG/summary.txt"
total=0; caught=0; bad=0

echo "== SCRIPTS-LIVE MUTATION MATRIX (checker copied to \$SB; tracked tree untouched) =="

# ---- baseline: a fully wired tree MUST pass -------------------------------
mk_tree wired
b="$(run_scan)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "  FAIL baseline -- the unmutated checker does not PASS a fully wired tree, so" >&2
  echo "        every state below would be scored against a checker that fires on" >&2
  echo "        everything. That is the always-red twin, and it is not a verdict." >&2
  printf '%s\n' "$b" | tail -8 | sed 's/^/          /' >&2
  exit 1
fi
if pred_expect_present "baseline classified a non-empty population" "$b" '[0-9]+ script\(s\): [0-9]+ NAMED, [0-9]+ GLOB, 0 unreachable'; then
  echo "  ok    baseline rc=0 and classified a non-empty population"
else
  echo "  FAIL baseline harvested nothing reachable -- a green verdict is vacuous" >&2
  printf '%s\n' "$b" | tail -8 | sed 's/^/          /' >&2
  exit 1
fi

# ---- one mutation, one tree, one expected disagreement -------------------
# The BASELINE is re-measured here, from a pristine copy, on the SAME tree the
# mutant sees. Measuring it once at the top and reusing it was the first version's
# bug, and it is worth naming because it fails in a way that looks like the
# mutations being unobservable: three states reported "the mutant AGREES with the
# baseline" when in fact the baseline slot was holding the MUTANT's rc, taken
# after patching. Both sides must come from the same tree with different checkers.
# $1 id, $2 tree, $3 marker, $4 exact-old, $5 exact-new, $6 want-mutant-rc,
# $7 RE that must be PRESENT in the mutant, $8 RE that must be ABSENT, $9 why
state() {
  local id="$1" tree="$2" marker="$3" old="$4" new="$5" want="$6" \
        want_re="$7" not_re="$8" why="$9" o r brc br
  total=$((total + 1))
  mk_tree "$tree"
  restore_checker                       # pristine -> baseline
  br="$(run_scan)"; brc=$?
  if ! patch_literal "$CHECKCOPY" "$old" "$new"; then
    echo "  $id MISSED -- patch_literal refused (anchor drifted or is ambiguous)"
    echo "$id MISSED (patch refused)" >>"$LOG/summary.txt"; bad=$((bad + 1)); return
  fi
  if ! verify_patched "$CHECKCOPY" "$marker"; then
    echo "  $id MISSED -- the mutation did not apply"
    echo "$id MISSED (verify failed)" >>"$LOG/summary.txt"; bad=$((bad + 1)); return
  fi
  o="$(run_scan)"; r=$?
  printf '%s\n' "$o" >"$LOG/$id.log"
  if [ "$brc" = "$r" ]; then
    echo "  $id MISSED -- the mutant AGREES with the baseline (both rc=$r), so the" >&2
    echo "        mutation is unobservable and this state proves nothing ($why)" >&2
    echo "$id MISSED (agreed with baseline rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1)); return
  fi
  if [ "$r" -eq "$want" ] \
     && pred_expect_present "$id expected-RE" "$o" "$want_re" \
     && pred_expect_absent  "$id expected-absent-RE" "$o" "$not_re"; then
    echo "  $id caught -- baseline rc=$brc, mutant rc=$r (verdict-polarity: the mutant must"
    echo "       DISAGREE with the baseline on the same tree; $why)"
    echo "$id caught (baseline=$brc mutant=$r)" >>"$LOG/summary.txt"; caught=$((caught + 1))
  else
    echo "  $id MISSED -- wanted mutant rc=$want with /$want_re/ present and /$not_re/ absent;" >&2
    echo "        got rc=$r (baseline rc=$brc). Mutant output tail:" >&2
    printf '%s\n' "$o" | tail -4 | sed 's/^/          /' >&2
    echo "$id MISSED (rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1))
  fi
}

# WHERE THE MARKER CAN AND CANNOT GO -- two attempts, both instructive.
#
# 1. A `#` comment appended to the anchor line breaks the file. That line ends in
#    a line-continuation backslash, so the comment is spliced into the MIDDLE of a
#    pipeline: the previous line's `\` swallows it, the logical line ends at the
#    comment, and the next line's leading `|` starts a new command. `bash -n`
#    fails inside verify_patched, which reports it as "the mutation did not
#    apply" -- a STATE fault wearing the message of a different fault.
# 2. A comment on its OWN line, spliced in as part of the replacement, breaks it
#    for the same reason.
#
# What works: the marker rides INSIDE the quoted ERE, as an alternation branch.
# `grep -vE '^\.git/|MSL_MUT_A'` is one pattern with two alternatives, so on a tree
# with no path containing the marker the ONLY behavioural change is the one the
# state claims -- the exclusion the state is about is gone -- and the file stays
# valid. The marker is not a comment here; it is an inert branch, and the state
# says so rather than pretending otherwise.
#
# What does NOT work, and cost a round: appending the marker as a bare WORD after
# the quoted pattern (`grep -vE '^\.git/' MSL_MUT_A`). grep reads every argument
# after the first as a FILENAME, so it printed `grep: MSL_MUT_A: No such file or
# directory` once per scan line, every NAMED hit became an error, and the mutant
# failed for a reason that had nothing to do with the rule under test -- three
# states' worth of confusion from one word. `grep -e` would have added it as a
# pattern; an alternation branch inside the existing quote is the same thing with
# one fewer token.
state "SL-A" prose "MSL_MUT_A" \
$'    | grep -vE \'^(build|\.git|docs|reports|node_modules)/\' \\' \
$'    | grep -vE \'^(build|\.git|node_modules)/|MSL_MUT_A\' \\' \
  0 'SCRIPTS LIVE: PASS' 'DEAD  zz_orphan' \
  "a docs/ citation becomes a call site, so every retired script survives"

state "SL-B" archive "MSL_MUT_B" \
$'    | grep -vE \'^scripts/archive/\' \\' \
$'    | grep -vE \'^scripts/ZZZ_no_such_dir/|MSL_MUT_B\' \\' \
  0 'SCRIPTS LIVE: PASS' 'DEAD  zz_orphan' \
  "scripts/archive/ stops being excluded, so the retired set keeps itself alive"

# The `case "$f" in` opener is NOT a usable anchor: the independent glob
# cross-check below has one too, and patch_literal correctly refuses an anchor
# that occurs twice (drifted and ambiguous must both fail loudly -- audit #17b).
# The 4-space-indented pattern line is unique; disabling it makes the case match
# nothing, which removes the class without touching the loop around it.
state "SL-C" wired "MSL_MUT_C" \
'    $GATE_GLOB)' \
'    $GATE_GLOB.MSL_MUT_C)' \
  1 'DEAD  test_gate_a' 'SCRIPTS LIVE: PASS' \
  "no gate is NAMED by anything, so disabling the glob kills the whole gate family"

state "SL-D" prose "MSL_MUT_D" \
'  if [ "$n_dead" -eq 0 ]; then' \
'  if [ "$n_dead" -ne 0 ]; then   # MSL_MUT_D: verdict inverted' \
  0 'SCRIPTS LIVE: PASS' 'SCRIPTS LIVE: FAIL' \
  "the verdict condition is inverted, so a tree with a dead script reports PASS"

# SL-E spans two lines. The newline goes INSIDE one quoted argument -- passing one
# argument per line is the five-argument bug _gate_patch.sh documents, and it
# replaces line 1 with line 2, applies nothing, and reports success.
state "SL-E" prose "MSL_MUT_E" \
'  n_dead=$((n_dead + 1))
  DEAD="$DEAD $f"' \
'  :   # MSL_MUT_E: an unreachable script is never counted
  DEAD="$DEAD $f"' \
  0 'SCRIPTS LIVE: PASS' 'SCRIPTS LIVE: FAIL' \
  "the classification no longer counts one -- same failure as SL-D, one site earlier"

echo
echo "== residue + restore integrity =="
# Restored BEFORE the scan, not after: the sandbox still holds the last mutant, and
# scanning a directory that is supposed to contain a restored copy would report
# the mutation as escaped when nothing escaped at all. The first version did
# exactly that on SL-E and called it a sandbox leak.
restore_checker
if residue_scan "$SB" MSL_MUT_A MSL_MUT_B MSL_MUT_C MSL_MUT_D MSL_MUT_E >/dev/null 2>&1; then
  echo "  ok    no mutation marker survived the restore of the sandbox checker"
else
  echo "  FAIL a mutation marker escaped $SB -- the sandbox is not isolated"
  residue_scan "$SB" MSL_MUT_A MSL_MUT_B MSL_MUT_C MSL_MUT_D MSL_MUT_E | sed 's/^/        /'
  exit 1
fi
# The claim is that THIS MATRIX wrote nothing outside $SB, so it is measured on
# the ONE file it reads and copies. `git diff --quiet -- scripts/` was the first
# version and it is the wrong instrument twice over: it cannot see untracked
# files, so a checker that had been edited and not yet added reported clean, and
# it fires on ANY pre-existing unstaged edit in scripts/ -- including the edits
# that created this checker -- so it reported a leak when nothing escaped. A
# guard whose stated rationale does not apply is worse than none (rule 5).
if [ "$(cksum <"$CHECK")" = "$CHECK_SUM" ]; then
  echo "  ok    the checker under test is byte-identical to the one at entry"
else
  echo "  FAIL check_scripts_live.sh changed during the matrix"; exit 1
fi
if git -C "$ROOT" status --porcelain -- scripts/ 2>/dev/null | grep -q '^ M scripts/'; then
  echo "  note scripts/ carries unstaged edits that predate this matrix; the"
  echo "       cksum above is the load-bearing check, this line is informational"
fi

echo
echo "-- summary --"
sed 's/^/  /' "$LOG/summary.txt"
if [ "$bad" -eq 0 ] && [ "$caught" -eq "$total" ]; then
  echo
  echo "SCRIPTS-LIVE MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  { echo "stamp=scripts_live_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"
    echo "caught=$caught"; } >"$STAMP/last_pass.stamp"
  exit 0
fi
echo "SCRIPTS-LIVE MUTATE: FAIL ($caught/$total caught, $bad problem(s)) -- NOT stamping" >&2
exit 1