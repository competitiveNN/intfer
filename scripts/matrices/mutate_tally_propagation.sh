#!/usr/bin/env bash
# mutate_tally_propagation.sh -- the negative direction for check_tally_propagation.sh.
#
# WHY A MATRIX AND NOT A SPOT CHECK. The propagation rule ("a named tally site is
# re-derived by the script it names, or declared EPHEMERAL") is only worth anything if a
# violation of it is actually refused. Every state below plants a defect in a TEMP COPY of
# the durable records; the live records are never touched, and the residue half proves it.
#
# THE SUBJECT IS TEXT, SO THERE IS NO scripts/ COPY. check_tally_propagation.sh sweeps the
# durable surface and resolves no script -- unlike mutate_durable_claims.sh, which copies
# scripts/*.sh because p_pair_before hashes them. Copying them here would be a fiction: it
# would suggest the sweep reads them.
#
# LANDING IS ASSERTED, NOT ASSUMED. Every state cmp-checks that the file it edited really
# differs from the live one before it is graded, so a mutation that silently no-ops is
# reported NOT CAUGHT (a no-op is a FAIL, never a catch).
#
# NO EXECUTION OF ANY SUBJECT: text only, no GPU, no model, no build, no mutator run.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$ROOT/scripts/checks/check_tally_propagation.sh"
SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/tp_mut.XXXXXX")"
TR="$SANDBOX/root"

die() { echo "TP-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }
note_ok() { echo "TP-MUT: $1 CAUGHT (${2:-no refusal reason harvested})"; }

[ -f "$CHECKER" ] || die "checker missing ($CHECKER)"
for _f in HISTORY.md AGENTS.md; do
  [ -f "$ROOT/$_f" ] || die "durable record missing ($ROOT/$_f)"
done
[ -d "$ROOT/docs/engines" ] || die "durable record missing ($ROOT/docs/engines)"

mkdir -p "$TR/docs"
cp -f "$ROOT/HISTORY.md" "$ROOT/AGENTS.md" "$TR/" || true
cp -f "$ROOT"/docs/*.md "$TR/docs/" 2>/dev/null || true
mkdir -p "$TR/docs/engines"
cp -f "$ROOT"/docs/engines/*.txt "$TR/docs/engines/" 2>/dev/null || true
[ -s "$TR/AGENTS.md" ] || die "could not stamp the temp root (AGENTS.md missing)"
[ "$(ls -1 "$TR"/docs/*.md 2>/dev/null | wc -l)" -ge 1 ] \
  || die "could not stamp the temp root (no docs/*.md)"
[ "$(ls -1 "$TR"/docs/engines/*.txt 2>/dev/null | wc -l)" -ge 1 ] \
  || die "could not stamp the temp root (no docs/engines/*.txt)"

surface() {
  printf '%s\n' "$TR/HISTORY.md" "$TR/AGENTS.md"
  for _d in "$TR"/docs/*.md; do [ -f "$_d" ] && printf '%s\n' "$_d"; done
  for _e in "$TR"/docs/engines/*.txt; do [ -f "$_e" ] && printf '%s\n' "$_e"; done
}
restamp() {
  cp -f "$ROOT/HISTORY.md" "$ROOT/AGENTS.md" "$TR/"
  cp -f "$ROOT"/docs/*.md "$TR/docs/" 2>/dev/null || true
  cp -f "$ROOT"/docs/engines/*.txt "$TR/docs/engines/" 2>/dev/null || true
}
surface_count() { surface | wc -l; }

SURF_N="$(surface_count)"
LIVE_N=2
for _d in "$ROOT"/docs/*.md; do [ -f "$_d" ] && LIVE_N=$((LIVE_N + 1)); done
for _e in "$ROOT"/docs/engines/*.txt; do [ -f "$_e" ] && LIVE_N=$((LIVE_N + 1)); done
[ "$SURF_N" -eq "$LIVE_N" ] \
  || die "the temp root carries $SURF_N record(s) but the live surface has $LIVE_N; a partial copy is a refusal for the wrong reason"
echo "TP-MUT: temp root stamped with $SURF_N durable record(s), text-only (the sweep resolves no script)"

# ---- BASELINE. An unmutated temp root must PASS, or every catch below is
#      indistinguishable from a sweep that refuses everything.
bash "$CHECKER" "$TR" >"$SANDBOX/baseline.txt" 2>&1
[ $? -eq 0 ] || { cat "$SANDBOX/baseline.txt"; die "the healthy temp root FAILS the sweep"; }
grep -q 'TALLY-PROP: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
BASE_HARVEST="$(grep -oE 'sites=[0-9]+ named=[0-9]+' "$SANDBOX/baseline.txt" | tail -1)"
[ -n "$BASE_HARVEST" ] || die "the baseline printed no harvest line; a sweep that looked at nothing cannot certify the states below"
echo "TP-MUT: baseline PASS on the unmutated temp root ($BASE_HARVEST)"

RESID_FILES="HISTORY.md AGENTS.md"
while IFS= read -r _d; do
  [ -n "$_d" ] || continue
  RESID_FILES="$RESID_FILES docs/${_d##*/}"
done < <(cd "$ROOT" && ls -1 docs/*.md 2>/dev/null)
while IFS= read -r _e; do
  [ -n "$_e" ] || continue
  RESID_FILES="$RESID_FILES docs/engines/${_e##*/}"
done < <(cd "$ROOT" && ls -1 docs/engines/*.txt 2>/dev/null)
# SUBJECT_READS is the uniform spelling of the same set RESID_FILES carries, so
# the re-run-after-the-edits rule is derivable with one grep over every matrix.
SUBJECT_READS="$RESID_FILES"
BEFORE="$(cd "$ROOT" && sha256sum $RESID_FILES | sha256sum | cut -d' ' -f1)"

STATES_REAL=0
STATES_CAUGHT=0
EXPECT=4

run_state() { # $1 = state name, $2 = expected predicate
  local st="$1" want="$2"
  restamp
  local a="$TR/AGENTS.md" h="$TR/HISTORY.md"
  local -a allf=("$h" "$a")
  local _d _e
  for _d in "$TR"/docs/*.md; do [ -f "$_d" ] && allf+=("$_d"); done
  for _e in "$TR"/docs/engines/*.txt; do [ -f "$_e" ] && allf+=("$_e"); done
  case "$st" in
    # ---- the propagation defect itself: the marker names a DIFFERENT real script -----
    # This is the state NO other predicate can catch. check_durable_claims.sh sees a
    # disposition (p_tally_regen ok) and a script that exists (p_target_exists ok), and
    # the record is still wrong: re-running the marker re-derives another matrix's tally.
    TP-wrong-target)
      python3 - "$a" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
OLD = 'REGEN(bash scripts/matrices/mutate_fold_matrix.sh)'
NEW = 'REGEN(bash scripts/matrices/mutate_tp2_pass_exit.sh)'
assert s.count(OLD) == 1, 'the fold-matrix marker was not found exactly once'
open(p, 'w').write(s.replace(OLD, NEW))
PY
      ;;
    # ---- a named site with no disposition at all -------------------------------------
    TP-undisposed)
      python3 - "$a" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
OLD = ' REGEN(bash scripts/matrices/mutate_fold_matrix.sh)'
assert s.count(OLD) == 1, 'the fold-matrix marker was not found exactly once'
open(p, 'w').write(s.replace(OLD, ''))
PY
      ;;
    # ---- the rule cannot be applied: no site names a script ---------------------------
    # The vacuity half. The sites still harvest, but every subject name is erased, so a
    # sweep that reported PASS here would be certifying a population it could not judge.
    TP-unnamed)
      python3 - "${allf[@]}" <<'PY'
import re, sys
SITE = re.compile(r'[0-9]+/[0-9]+ (states )?caught|PASS[ ,(]+\(?[0-9]+/[0-9]+', re.I)
NAME = re.compile(r'\b(?:mutate|check)_[a-z0-9_]+\.sh\b')
hit = 0
for p in sys.argv[1:]:
    lines = open(p).read().split('\n')
    changed = False
    for i, ln in enumerate(lines):
        if SITE.search(ln) and NAME.search(ln):
            new = NAME.sub('SCRIPT-ERASED', ln)
            if new != ln:
                lines[i] = new
                changed = True
    if changed:
        open(p, 'w').write('\n'.join(lines))
        hit += 1
assert hit >= 1, 'no tally site naming a script was found anywhere on the surface'
PY
      ;;
    # ---- the harvest itself empties ---------------------------------------------------
    TP-no-sites)
      python3 - "${allf[@]}" <<'PY'
import re, sys
TALLY = re.compile(r'[0-9]+/[0-9]+ (states )?caught', re.I)
RUNRES = re.compile(r'PASS[ ,(]+\(?[0-9]+/[0-9]+', re.I)
hit = 0
for p in sys.argv[1:]:
    s = open(p).read()
    out = RUNRES.sub('RUNRES-ERASED', TALLY.sub('TALLY-ERASED', s))
    if out != s:
        open(p, 'w').write(out)
        hit += 1
assert hit >= 1, 'no tally or run-result text was found anywhere on the surface'
PY
      ;;
    *) die "$st: unknown state" ;;
  esac

  local changed=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    cmp -s "$f" "$ROOT/${f#$TR/}" || changed=1
  done < <(surface)
  if [ "$changed" -eq 0 ]; then
    echo "TP-MUT: $st NOT CAUGHT -- the mutation did not LAND (a no-op must be a FAIL, never a catch)"
    return
  fi

  STATES_REAL=$((STATES_REAL + 1))
  bash "$CHECKER" "$TR" >"$SANDBOX/$st.out" 2>&1
  if [ $? -eq 0 ]; then
    echo "TP-MUT: $st NOT CAUGHT -- the sweep passed a defective record"
    sed 's/^/    /' "$SANDBOX/$st.out" | head -12
  else
    if grep -q "FAIL -- refused by:.*p_$want" "$SANDBOX/$st.out"; then
      note_ok "$st" "p_$want"
      STATES_CAUGHT=$((STATES_CAUGHT + 1))
    else
      echo "TP-MUT: $st CAUGHT FOR THE WRONG REASON -- wanted p_$want, got:"
      grep 'FAIL -- refused by:' "$SANDBOX/$st.out" | sed 's/^/    /'
      grep 'TALLY-PROP: p_' "$SANDBOX/$st.out" | sed 's/^/    /' | head -4
    fi
  fi
}

for st in TP-wrong-target TP-undisposed TP-unnamed TP-no-sites; do
  run_state "$st" tally_propagation
done

if [ "$STATES_REAL" -ne "$EXPECT" ]; then
  die "expected $EXPECT states, graded $STATES_REAL -- a matrix that never ran cannot report a tally"
fi
if [ "$STATES_CAUGHT" -ne "$STATES_REAL" ]; then
  echo "TP-MUT: matrix $STATES_CAUGHT/$STATES_REAL states caught"
  echo "TALLY-PROP MATRIX: FAIL"
  echo "RESULT: FAIL"
  exit 1
fi

AFTER="$(cd "$ROOT" && sha256sum $RESID_FILES | sha256sum | cut -d' ' -f1)"
[ "$AFTER" = "$BEFORE" ] \
  || die "the live durable surface CHANGED during the matrix (sha $BEFORE -> $AFTER)"
echo "TP-MUT: residue byte-identical (live surface untouched: $(printf '%s\n' $RESID_FILES | wc -l) file(s))"

echo "TP-MUT: matrix $STATES_CAUGHT/$STATES_REAL states caught"
echo "TALLY-PROP MATRIX: PASS ($STATES_CAUGHT/$STATES_REAL, LANDED asserted per state, each for its own predicate)"
echo "RESULT: PASS"
exit 0
