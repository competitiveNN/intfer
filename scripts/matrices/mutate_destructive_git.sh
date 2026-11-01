#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_destructive_git.sh.
#
# WHAT THE CHECKER CLAIMS, and therefore what must be falsifiable: that no script
# can destroy uncommitted work -- every script performing a destructive git op
# refuses to run on a dirty tree -- and that the prompt case (a destructive
# command typed by a human or an agent) is covered by a standing rule.
#
# The states attack both halves, and the FIRST one is the instructive one.
#
# DG-1 REMOVES A DIRTY-TREE GUARD from a script that really does restore with
#      `git checkout --`. Without it the checker would be reporting "guarded"
#      purely because the word appeared somewhere in the file.
# DG-2 STRIPS THE COMMENT HARVEST. This is the decoy the design is built around:
#      mutate_tp2_selfcheck.sh contains the literal `git checkout --` exactly once,
#      inside a comment saying it deliberately needs no guard. A harvest that did
#      not strip comments would flag that script as unguarded -- correctly-shaped
#      noise about a script that is right. DG-2 is graded want=1 (CAUGHT) because
#      the refusal IS the proof that the stripping is load-bearing: replacing it
#      with `cat` makes the checker go RED on a HEALTHY tree by flagging the decoy.
#      Had it stayed green the stripping would have been decorative.
#
#      RECORDED CORRECTION, 2026-10-20 -- THE OLD PROSE HERE WAS WRONG AND THE
#      MEASUREMENT DISPROVED IT. This paragraph, run_state's else-branch comment,
#      and the PASS summary all said DG-2 "must stay GREEN" and that the mutation
#      "makes the checker MORE permissive". Measured against a clean root
#      (bash scripts/matrices/mutate_destructive_git.sh /tmp/dg_clean):
#        DG-2 CAUGHT (FAIL mutate_tp2_selfcheck.sh 1 destructive op(s), NO dirty-tree guard)
#      The mutation makes the checker STRICTER, not more permissive -- it flags a
#      script that has no real destructive op -- and it REFUSES, so a want=0 grade
#      here would have reported FAIL on a correctly-caught state. The three claims
#      were prose written without a re-measurement behind them; the harvested reason
#      in the verdict is what exposed the contradiction. The grading was right, the
#      words were wrong, and the words are corrected rather than left beside a green
#      tick they do not describe.
# DG-3 REMOVES THE AGENTS.md RULE, which is the only thing covering the prompt
#      case. No script can enforce that, so if the rule can be deleted silently
#      the checker is decorative on the half that actually bit.
# DG-4 EMPTIES THE CORPUS (every destructive op commented out). Must be REFUSED by
#      the anti-vacuity floor, never reported as "4/4 guarded, nothing to check".
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROOT IS OVERRIDABLE, AND THIS IS THE FIX FOR A GUARD THAT COULD NEVER PASS.
# The dirty-tree guard below is CORRECT and is not being weakened: a matrix that
# snapshots tracked files someone else had already modified, then "restores" them,
# restores them to the dirty bytes and measures something other than what it thinks.
# The problem is that this repo's STANDING state is dirty -- the standing no-commit
# rule means every session leaves uncommitted work -- so an unconditional guard
# against the live tree made all 16 states permanently ungradeable, which is the
# same defect shape mutate_knob_sweep_contract.sh had (see its [SUPERSEDED] note).
# The remedy is NOT to drop the requirement but to make the SUBJECT selectable: the
# matrix runs against a clean checkout/worktree, exporting one as $1, and the guard
# then means what it says. PROVEN, not assumed: the live tree still refuses.
#   bash scripts/matrices/mutate_destructive_git.sh                                   # live tree
#   bash scripts/matrices/mutate_destructive_git.sh /tmp/dg_clean                    # clean root
ROOT="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
[ -d "$ROOT/scripts" ] || { echo "FAIL: no scripts/ under the selected root ($ROOT)" >&2; exit 2; }
cd "$ROOT" || exit 2

# BUILD comes from SCRIPT_DIR, never from the alternate root: writing sandbox
# artifacts into the root under test would dirty the very tree the guard certifies.
# Identical to the old $ROOT/build derivation when $1 is absent (the default), so
# this is not a behaviour change for existing callers.
BUILD="${INTFER_BUILD_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/build}"

# SB IS KEYED ON THE SELECTED ROOT, so two invocations cannot race on `rm -rf`.
# MEASURED, 2026-10-20 -- found because I ran the live-tree and the clean-root
# invocations in the same shell by mistake. The live run's `rm -rf "$SB"` emptied
# the clean-root run's PRISTINE_DIR mid-flight, so every subsequent `restore_all`
# copied from a snapshot that no longer existed: the matrix reported DG-4 MISSED
# and a residue failure over ~140 files, and it left AGENTS.md plus two tracked
# scripts PATCHED in the exported root. The un-suffixed SB was the defect: the
# root is SELECTABLE, so the sandbox had to be selectable with it. A residue
# reading that "differs" is the least of it -- an exported root can be a real
# checkout, and an unrestored tracked file is destroyed work, which is the exact
# class this repo's commit-before-a-destructive-probe rule exists for.
ROOT_KEY="$(printf '%s' "$ROOT" | cksum | cut -d' ' -f1)"
SB="$BUILD/destructive_git_matrix.$ROOT_KEY"
CHECK="scripts/checks/check_destructive_git.sh"
DOC="AGENTS.md"
ART="scripts/matrices/mutate_device_precond.sh"   # a real destructive script

# SUBJECT_READS names the durable corpus surfaces this matrix's assertions read.
# It is AGENTS.md and not "any": DG-3 patches the commit-before-a-destructive-probe
# rule IN that file and asserts the checker refuses without it, so the assertion's
# satisfaction depends on the literal text of the record -- the same dependency the
# other five declared matrices have.
#
# MEASURED, 2026-10-20: this matrix was INVISIBLE to check_durable_reads.sh until the
# checker's derivation was widened, because the read is `git diff --quiet -- $DIRTY`
# and $DIRTY is assembled from $DOC at RUN TIME -- the member name never appears next
# to an operator in the source. The population was six while the true count was
# seven, so this declaration could have been deleted with no gate noticing: the
# "opt-in flag no gate can reach" shape, in the coverage tooling itself.
SUBJECT_READS="AGENTS.md"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

# DIRTY-TREE GUARD. This matrix writes tracked files (AGENTS.md, several
# scripts/) and restores them from its own pristine copies. Restoring from a copy
# is safer than `git checkout --` -- it cannot reach a path this matrix did not
# snapshot -- but the guard is still required, and check_destructive_git.sh
# flagged this file for its absence the moment its guard detection was tightened
# from "any of these words" to "an actual test". The finding is correct: a matrix
# that snapshots a tree someone else had already modified is measuring something
# other than what it thinks.
#
# THE REQUIREMENT IS UNCHANGED AND THE SUBJECT IS NOW SELECTABLE. Filtered to
# files that EXIST, so a root at a different commit -- where one of these paths may
# be absent -- fails with a legible list rather than an opaque git error, and a
# missing file cannot silently drop a path out of the list the loop checked.
DIRTY=""
for _p in $CHECK $DOC scripts/matrices/mutate_device_precond.sh scripts/matrices/mutate_fold_matrix.sh \
          scripts/matrices/mutate_gate_inventory.sh scripts/run_all_checks.sh; do
  [ -f "$_p" ] && DIRTY="$DIRTY $_p"
done
[ -n "$DIRTY" ] || { echo "FAIL: no dirty-tree subject resolved under $ROOT" >&2; exit 2; }
# `git diff` exits non-zero for BOTH "changed" and "not a git repository"; the
# combined test below cannot tell them apart, so it is split deliberately and the
# not-a-repo case is named. A matrix run against a plain directory must fail
# closed, not skip its own precondition.
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "FAIL: the selected root is not a git repository ($ROOT)" >&2
  echo "  Export a clean checkout/worktree as \$1: git worktree add --detach /tmp/dg_clean HEAD" >&2
  exit 1
fi
if ! git diff --quiet -- $DIRTY || ! git diff --cached --quiet -- $DIRTY; then
  echo "FAIL: uncommitted changes in: $DIRTY" >&2
  ndirty=0; for f in $DIRTY; do git diff --quiet -- "$f" 2>/dev/null || ndirty=$((ndirty+1)); done
  for f in $DIRTY; do
    if ! git diff --quiet -- "$f" 2>/dev/null; then
      echo "  INPUT: UNCOMMITTED file=$f blob=$(git hash-object "$f")" >&2
    fi
  done
  if [ "${1:-}" = "" ]; then
    echo "  This script edits and restores them; the standing no-commit rule blocks"
    echo "  committing or stashing, so export a CLEAN root instead:" >&2
    echo "    git worktree add --detach /tmp/dg_clean HEAD && bash scripts/matrices/mutate_destructive_git.sh /tmp/dg_clean" >&2
  else
    echo "  The alternate root \$1 is itself dirty; export a clean one." >&2
  fi
  exit 1
fi
echo "  ok dirty-tree guard: $(printf '%s\n' $DIRTY | grep -c .) tracked subject(s) clean at $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
# Snapshot EVERY file any state may touch, and DERIVE the list. The first version
# kept two copies while states edited four scripts, so it reported "residue ok" on
# the two it checked and left the other two patched in the tree -- a residue check
# lying by omission. Derived rather than named, because a hardcoded list went stale
# the moment a fifth destructive script appeared (audit #7, applied to my states).
TOUCHED="$(find scripts -maxdepth 2 -not -path "*/archive/*" -name '*.sh' -type f | LC_ALL=C sort | tr '\n' ' ') $DOC"
PRISTINE_DIR="$SB/pristine"
rm -rf "$PRISTINE_DIR"; mkdir -p "$PRISTINE_DIR"
for f in $TOUCHED; do cp "$f" "$PRISTINE_DIR/$(echo "$f" | tr / _)"; done
restore_all() { local f; for f in $TOUCHED; do cp "$PRISTINE_DIR/$(echo "$f" | tr / _)" "$f"; done; }

patch1() { python3 - "$1" "$2" "$3" <<'PYQ'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read(); c = s.count(o)
if c != 1:
    print("anchor found %d times (want 1) in %s" % (c, p), file=sys.stderr); sys.exit(1)
open(p, 'w').write(s.replace(o, n, 1))
PYQ
}
patchN() { python3 - "$1" "$2" "$3" <<'PYQ'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) < 1:
    print("anchor absent in %s" % p, file=sys.stderr); sys.exit(1)
open(p, 'w').write(s.replace(o, n))
PYQ
}

baseline() {
  bash "$CHECK" >"$SB/base.log" 2>&1; local rc=$?
  [ "$rc" -eq 0 ] || { echo "  FAIL baseline rc=$rc" >&2; tail -6 "$SB/base.log" >&2; return 1; }
  grep -q '^RESULT: ' "$SB/base.log" || { echo "  FAIL baseline has no RESULT:" >&2; return 1; }
  echo "  ok baseline rc=0, PASS verdict present"
}

total=0; caught=0; RESULTS=()
# run_state id expect want state_fn
#
# THE REASON IS HARVESTED, NOT ASSERTED-EMPTY. The original form appended the literal
# string `CAUGHT ''` -- an empty reason that PRINTED as though the state had been
# graded on the checker's own words while nothing was read. Measured consequence:
# DG-3, whose first anchor had drifted to text AGENTS.md no longer carries, fell
# through to its second patch and was counted CAUGHT for the refusal the second one
# caused, so a PARTIAL mutation was graded as a clean catch. That is the exact
# convention mutate_durable_claims.sh records ("each state caught for the wrong
# reason is visible rather than counted"), and it was violated here in the printed
# verdict. The harvest is the checker's own line, so the catch now shows WHY.
#
# AND A STATE THAT DID NOT LAND IS A MISS, NOT A CATCH. A patch that fails its
# anchor leaves the tree unmutated, so running the checker after it measures the
# healthy tree -- which would be reported as "the checker refused" only if the
# checker refuses everything, and as a spurious catch otherwise. The state function
# is now invoked BEFORE the checker and its return code gates the grading.
run_state() {
  local id="$1" expect="$2" want="$3" fn="$4" out rc reason
  total=$((total + 1))
  if ! "$fn"; then
    RESULTS+=("$id MISSED '(the mutation did not LAND -- anchor drift, aborting the state)'")
    restore_all
    return
  fi
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  restore_all
  reason="$(printf '%s\n' "$out" | grep -E -- "$expect" | head -1 | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
  if [ "$want" = 1 ]; then
    if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS -- the checker passed a defective tree)")
    elif [ -z "$reason" ]; then RESULTS+=("$id MISSED '(no harvested reason: refused by something OTHER than: $expect)")
    else caught=$((caught + 1)); RESULTS+=("$id CAUGHT (${reason})"); fi
  else
    # The want=0 direction, kept GENERIC. No state uses it today -- all four are
    # want=1. It was written for DG-2, which an earlier (wrong) reading of the
    # decoy had graded as a must-stay-green state; see the DG-2 correction in the
    # header, which records that reading as disproved rather than deleting it. A
    # mutation that leaves the checker green proves SPECIFICITY -- the mutation is
    # not the defect this checker looks for -- and a refusal here means a false
    # positive on a healthy tree, so the harvested reason is what makes that
    # legible instead of merely present. Kept with its DG-2 attribution removed
    # rather than deleted, so the next state needing it is not designed from a
    # reading that was measured false.
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT (no refusal: the state stayed green by design)")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE): $reason"); fi
  fi
}

# DG-1 EVERY occurrence: mutate_device_precond.sh carries five dirty-tree queries,
# so replacing one left four and the state scored MISSED while the checker was,
# correctly, still seeing a guard. A state removes the MECHANISM, not a token.
dg1() { patchN scripts/matrices/mutate_device_precond.sh 'git diff --quiet' '# DG1 removed'; }

# DG-2 remove the comment-strip. mutate_tp2_selfcheck.sh mentions
# `git checkout --` exactly once, inside a comment saying it deliberately needs no
# guard -- so with comments included a CORRECT script gets flagged. The strip is
# the only reason it does not, which makes it load-bearing.
dg2() { patch1 "$CHECK" 'strip_comments() { sed -e' 'strip_comments() { cat "$1"; } #DG2'; }

# DG-3 the checker regex accepts EITHER "commit...before...destructive" OR
# "reset --hard...uncommitted", and rule 16 satisfies BOTH -- so deleting only
# the heading left the rule matching on its own body and the state scored MISSED.
#
# ANCHOR DRIFT, RECORDED AS THE FAILURE IT WAS. Rule 16's heading is now
# "**Commit before a destructive probe.**" (title case, "a destructive probe"), and
# the old all-lowercase "**commit before any destructive git operation**" anchor had
# gone absent in BOTH the working tree and at HEAD. `patchN` then failed its
# pre-flight and printed "anchor absent in AGENTS.md", while the SECOND patch on the
# same line still ran -- so DG-3 mutated half its target and was still counted
# CAUGHT, because run_state's verdict carried no harvested reason to see it in.
# Both halves are now ASSERTED: the anchor is re-derived from the live text, and a
# patch that does not land aborts the state as a MISS instead of grading it.
dg3() {
  local bad=""
  patchN "$DOC" '**Commit before a destructive probe.**' '[DELETED BY DG-3]' \
    || bad="$bad heading-anchor"
  patchN "$DOC" 'reset --hard' 'a destructive command' \
    || bad="$bad reset-anchor"
  [ -z "$bad" ] || { echo "  DG-3 anchor drift: $bad -- the mutation did not land, so the state" >&2
                     echo "  is a MISS (the checker would be measuring a healthy tree), never a catch" >&2
                     return 1; }
}

# DG-4 neuter the harvest REGEX -- the direct anti-vacuity test. The first attempt
# commented out every `git checkout --` across scripts/, ambiguous twice: the
# checker's pattern is WIDER (reset --hard, clean -f) AND this file's own mutation
# strings are executable occurrences of it, so the state could not tell whether it
# had emptied the corpus. Neuter the query instead -- that is what "finds nothing"
# actually looks like (the FG-1 move).
dg4() { patch1 "$CHECK" "checkout[[:space:]]+--)'" "zzz_never)'"; }

# THE STATE FUNCTION IS THE FOURTH ARGUMENT, INVOKED BY run_state ITSELF. It was
# previously invoked at the call site ($dg1; run_state ... 1), which meant the
# mutation ran BEFORE run_state existed to see its exit code -- so a patch that
# failed its anchor silently left a healthy tree and the state was graded on it.
# run_state now calls it and gates the grading on its return code, so the anchor
# pre-flight can only be read where the verdict is decided.
run_state DG-1 'NO dirty-tree guard|destroy uncommitted work|FAIL' 1 dg1
run_state DG-2 'NO dirty-tree guard|destroy uncommitted work|FAIL' 1 dg2
run_state DG-3 'does not carry the commit-before-a-destructive-probe|FAIL' 1 dg3
run_state DG-4 'harvest is|broken' 1 dg4

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

# Residue over EVERY touched file, not the two the first version checked -- a
# residue check that counts only what it happens to look at is the same shape as
# a checker that certifies only the corpus it enumerated.
leftover=""
for f in $TOUCHED; do
  cmp -s "$f" "$PRISTINE_DIR/$(echo "$f" | tr / _)" || leftover="$leftover $f"
done
if [ -n "$leftover" ]; then
  echo "  FAIL residue: these differ from their pre-matrix state:$leftover" >&2
  exit 1
fi
echo "  ok residue: all $(printf '%s\n' $TOUCHED | grep -c .) touched files byte-identical to their pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "DESTRUCTIVE-GIT MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  DG-1: removing a real script's dirty-tree guard is observable"
  echo "  DG-2: the comment harvest is load-bearing -- breaking it makes the"
  echo "        checker flag the mutate_tp2_selfcheck.sh DECOY and go RED on a"
  echo "        healthy tree, so the refusal is the proof, not an anomaly"
  echo "  DG-3: deleting the AGENTS.md standing rule is observable"
  echo "  DG-4: an emptied corpus is refused, not read as 'nothing to check'"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DESTRUCTIVE-GIT MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
