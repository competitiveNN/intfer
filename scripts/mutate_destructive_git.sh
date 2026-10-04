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
PRISTINE_ART="$SB/pristine.art"
PRISTINE_DOC="$SB/pristine.doc"
cp "$ART" "$PRISTINE_ART"
cp "$DOC" "$PRISTINE_DOC"

patch() { python3 - "$1" "$2" "$3" <<'PY'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) < 1:
    print("anchor not found in %s: %r" % (p, o), file=sys.stderr)
    sys.exit(1)
open(p, 'w').write(s.replace(o, n))
PY
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
  cp "$PRISTINE_ART" "$ART"; cp "$PRISTINE_DOC" "$DOC"
  if [ "$want" = 1 ]; then
    if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
    elif ! printf '%s' "$out" | grep -qE -- "$expect"; then RESULTS+=("$id MISSED '(wrong reason)'")
    else caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''"); fi
  else
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE)'"); fi
  fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  DG-1  a real destructive script loses its dirty-tree guard  [must REJECT]"
  echo "  DG-2  comment-stripping removed from the harvest           [must REJECT]"
  echo "  DG-3  the AGENTS.md standing rule deleted                  [must REJECT]"
  echo "  DG-4  every destructive op commented out (empty corpus)    [must REJECT]"
  exit 0
fi

echo "== baseline =="
baseline || exit 1
echo "== states =="

# DG-1: remove the guard from a script that really does restore with checkout.
dg1() { patch "$ART" 'DIRTY="$TGT_A $TGT_C $TGT_B $TGT_H $TGT_P"' 'DIRTY=""  #DG1 guard removed'; }

# DG-2: make the harvest read comments too, so the mutate_tp2_selfcheck.sh
# DECOY is flagged. Must stay GREEN: more permissive, not less.
dg2() { patch "$CHECK" "sed -e 's/[[:space:]]#.*\$//' -e 's/^[[:space:]]*#.*\$//' \"\$1\"" 'cat "$1"'; }

# DG-3: delete the standing rule.
dg3() { patch "$DOC" '16. **COMMIT BEFORE ANY DESTRUCTIVE GIT' '16. **[DELETED BY DG-3]** old text' ; }

# DG-4: empty the corpus.
dg4() { for f in scripts/mutate_device_precond.sh scripts/mutate_fold_matrix.sh \
               scripts/mutate_gate_inventory.sh scripts/run_all_checks.sh; do
           patch "$f" 'git checkout --' '# git checkout -- (commented out)'
       done; }

dg1; run_state DG-1 'NO dirty-tree guard|destroy uncommitted work|FAIL' 1
dg2; run_state DG-2 'NO dirty-tree guard|destroy uncommitted work|FAIL' 1
dg3; run_state DG-3 'does not carry the commit-before-a-destructive-probe|FAIL' 1
dg4; run_state DG-4 'harvest is|broken' 1

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

if ! cmp -s "$ART" "$PRISTINE_ART" || ! cmp -s "$DOC" "$PRISTINE_DOC"; then
  echo "  FAIL residue: an artifact differs from its pre-matrix state" >&2
  exit 1
fi
echo "  ok residue: both artifacts byte-identical to their pre-matrix state"

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
