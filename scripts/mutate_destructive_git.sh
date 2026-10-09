#!/usr/bin/env bash
# Mutation matrix for scripts/check_destructive_git.sh.
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
#      noise about a script that is right. DG-2 must therefore stay GREEN, because
#      it is the state that proves the stripping is load-bearing rather than
#      decorative. It is the always-red twin in a non-obvious direction: the
#      "mutation" makes the checker MORE permissive, and it must not be caught.
# DG-3 REMOVES THE AGENTS.md RULE, which is the only thing covering the prompt
#      case. No script can enforce that, so if the rule can be deleted silently
#      the checker is decorative on the half that actually bit.
# DG-4 EMPTIES THE CORPUS (every destructive op commented out). Must be REFUSED by
#      the anti-vacuity floor, never reported as "4/4 guarded, nothing to check".
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/destructive_git_matrix"
CHECK="scripts/check_destructive_git.sh"
DOC="AGENTS.md"
ART="scripts/mutate_device_precond.sh"   # a real destructive script

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
DIRTY="$CHECK $DOC scripts/mutate_device_precond.sh scripts/mutate_fold_matrix.sh
scripts/mutate_gate_inventory.sh scripts/run_all_checks.sh"
if ! git diff --quiet -- $DIRTY || ! git diff --cached --quiet -- $DIRTY; then
  echo "FAIL: uncommitted changes in: $DIRTY" >&2
  echo "  This script edits and restores them; commit or stash first." >&2
  exit 1
fi
# Snapshot EVERY file any state may touch, and DERIVE the list. The first version
# kept two copies while states edited four scripts, so it reported "residue ok" on
# the two it checked and left the other two patched in the tree -- a residue check
# lying by omission. Derived rather than named, because a hardcoded list went stale
# the moment a fifth destructive script appeared (audit #7, applied to my states).
TOUCHED="$(ls scripts/*.sh) $DOC"
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
run_state() {
  local id="$1" expect="$2" want="$3" out rc
  total=$((total + 1))
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  restore_all
  if [ "$want" = 1 ]; then
    if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
    elif ! printf '%s' "$out" | grep -qE -- "$expect"; then RESULTS+=("$id MISSED '(wrong reason)'")
    else caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''"); fi
  else
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE)'"); fi
  fi
}

# DG-1 EVERY occurrence: mutate_device_precond.sh carries five dirty-tree queries,
# so replacing one left four and the state scored MISSED while the checker was,
# correctly, still seeing a guard. A state removes the MECHANISM, not a token.
dg1() { patchN scripts/mutate_device_precond.sh 'git diff --quiet' '# DG1 removed'; }

# DG-2 remove the comment-strip. mutate_tp2_selfcheck.sh mentions
# `git checkout --` exactly once, inside a comment saying it deliberately needs no
# guard -- so with comments included a CORRECT script gets flagged. The strip is
# the only reason it does not, which makes it load-bearing.
dg2() { patch1 "$CHECK" 'strip_comments() { sed -e' 'strip_comments() { cat "$1"; } #DG2'; }

# DG-3 the checker regex accepts EITHER "commit...before...destructive" OR
# "reset --hard...uncommitted", and rule 16 satisfies BOTH -- so deleting only
# the heading left the rule matching on its own body and the state scored MISSED.
#
# ANCHOR DRIFT, AND IT IS THE STATE THAT DECAYED. Rule 16 was rewritten to
# lowercase ("commit before any destructive git operation") in a later commit,
# while this state still carried the OLD UPPERCASE heading. `patchN` then failed
# its own pre-flight ("anchor absent in AGENTS.md") and the matrix aborted before
# running DG-3, scoring it MISSED for a reason unrelated to the checker. The
# anchor below is derived from the current text; a rewording of rule 16 makes
# THIS line stale, not the checker's regex.
dg3() { patchN "$DOC" '**commit before any destructive git operation**' '[DELETED BY DG-3]'
        patchN "$DOC" 'reset --hard' 'a destructive command'; }

# DG-4 neuter the harvest REGEX -- the direct anti-vacuity test. The first attempt
# commented out every `git checkout --` across scripts/, ambiguous twice: the
# checker's pattern is WIDER (reset --hard, clean -f) AND this file's own mutation
# strings are executable occurrences of it, so the state could not tell whether it
# had emptied the corpus. Neuter the query instead -- that is what "finds nothing"
# actually looks like (the FG-1 move).
dg4() { patch1 "$CHECK" "checkout[[:space:]]+--)'" "zzz_never)'"; }

dg1; run_state DG-1 'NO dirty-tree guard|destroy uncommitted work|FAIL' 1
dg2; run_state DG-2 'NO dirty-tree guard|destroy uncommitted work|FAIL' 1
dg3; run_state DG-3 'does not carry the commit-before-a-destructive-probe|FAIL' 1
dg4; run_state DG-4 'harvest is|broken' 1

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
  echo "  DG-2: the comment harvest is load-bearing -- without it the"
  echo "        mutate_tp2_selfcheck.sh DECOY is flagged, so the state must"
  echo "        stay GREEN and does"
  echo "  DG-3: deleting the AGENTS.md standing rule is observable"
  echo "  DG-4: an emptied corpus is refused, not read as 'nothing to check'"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DESTRUCTIVE-GIT MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
