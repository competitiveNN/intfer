#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_history_pointers.sh.
#
# WHAT THE CHECKER CLAIMS: every AGENTS.md citation of a HISTORY.md title
# resolves by one of four rules (real heading / destroyed-entries banner /
# verbatim text / in-place [DANGLING annotation), and no banner entry describes
# a pointer AGENTS.md no longer carries.
#
# WHY THE SUBJECTS ARE COPIES AND NEVER THE LIVE FILES. AGENTS.md and HISTORY.md
# are the two largest tracked documents in the repo and both are DIRTY in this
# repo's standing state. A matrix that edits them in place is the destructive
# probe the standing rule exists for -- the knob-sweep matrix records exactly
# that mistake ("the original 12-state session appended to the live subject and
# got away with it only because a cp backup had been taken first"). So every
# state mutates a sandbox COPY and the checker is pointed at it through its
# HPOINT_AGENTS / HPOINT_HISTORY overrides, and the live files' shas are
# re-asserted at the end.
#
# THE STATES:
#   HP-1  an unaccounted pointer (a title nothing carries)   -> REFUSE
#   HP-2  the SAME pointer, now annotated in place            -> PASS (control)
#   HP-3  a banner entry AGENTS.md no longer cites            -> REFUSE
#   HP-4  the [DANGLING annotation moved FAR from its pointer -> REFUSE
#   HP-5  every citation removed (the harvest empties)        -> REFUSE
#   HP-6  a banner entry's title corrupted in AGENTS.md        -> REFUSE
#
# HP-1 AND HP-2 ARE A PAIR AND THE PAIR IS THE POINT. HP-1 alone would prove the
# checker refuses a dangling title; it cannot tell a checker that refuses
# EVERYTHING from one that refuses only unaccounted titles. HP-2 plants the same
# defect plus the annotation that is supposed to excuse it, and requires a PASS,
# so the pair together proves rule 4 is doing work rather than being a rubber
# stamp -- and that the checker is not simply red on any new title.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/history_pointer_matrix"
CHECK="scripts/checks/check_history_pointers.sh"
mkdir -p "$BUILD" "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

P_AG="$SB/AGENTS.md"
P_HI="$SB/HISTORY.md"
cp "$ROOT/AGENTS.md" "$P_AG" || exit 2
cp "$ROOT/HISTORY.md" "$P_HI" || exit 2

LIVE_AG_SHA="$(sha256sum "$ROOT/AGENTS.md" | cut -d' ' -f1)"
LIVE_HI_SHA="$(sha256sum "$ROOT/HISTORY.md" | cut -d' ' -f1)"

# SUBJECT_READS names the durable corpus surfaces this matrix's assertions read.
# Declaring it is what makes the "a matrix must be re-run after the edits it
# certifies" rule machine-derivable: an undeclared read is invisible to any
# ordering check, which is how a false ordering claim survived in HISTORY.md.
SUBJECT_READS="HISTORY.md AGENTS.md"

run_state() { # $1 = id, $2 = expect regex, $3 = want(1=REFUSE, 0=PASS)
  local id="$1" expect="$2" want="$3" out rc reason
  out="$(HPOINT_AGENTS="$P_AG" HPOINT_HISTORY="$P_HI" bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  # restore both subjects to their pre-state bytes
  cp "$ROOT/AGENTS.md" "$P_AG"
  cp "$ROOT/HISTORY.md" "$P_HI"
  # THE REASON IS HARVESTED, NOT ASSERTED-EMPTY -- the same defect recorded in
  # mutate_destructive_git.sh, where `CAUGHT ''` printed as though the state had
  # been graded on the checker's own words while nothing was read. Measured here
  # before the fix: every one of the six state logs carries a specific, harvestable
  # refusal naming the planted defect ("unaccounted pointer: 2026-10-19:
  # INTFER_NO_SUCH_RECORD_ANYWHERE" for HP-1, the TYPO'd title for HP-6), so the
  # catch was real and only the PRINTED VERDICT was opaque -- and HP-6's log shows
  # a SECOND collateral refusal ("banner entry no longer cited") that a carried
  # reason makes visible to the reader instead of counted silently.
  reason="$(printf '%s\n' "$out" | grep -E -- "$expect" | head -1 | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
  if [ "$want" = 1 ]; then
    if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
    elif [ -z "$reason" ]; then RESULTS+=("$id MISSED '(no harvested reason: refused by something OTHER than: $expect)'")
    else CAUGHT=$((CAUGHT + 1)); RESULTS+=("$id CAUGHT (${reason})"); fi
  else
    if [ "$rc" -eq 0 ]; then CAUGHT=$((CAUGHT + 1)); RESULTS+=("$id CAUGHT (no refusal: the state stayed green by design)")
    else RESULTS+=("$id MISSED '(FALSE POSITIVE): $reason'"); fi
  fi
}

RESULTS=(); total=0; CAUGHT=0
begin_state() { total=$((total + 1)); }

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  HP-1  an unaccounted HISTORY.md title pointer            [must REJECT]"
  echo "  HP-2  the same pointer WITH its [DANGLING annotation      [must PASS]"
  echo "  HP-3  a banner entry AGENTS.md no longer cites            [must REJECT]"
  echo "  HP-4  the [DANGLING annotation moved far from its pointer [must REJECT]"
  echo "  HP-5  every HISTORY.md citation removed                   [must REJECT]"
  echo "  HP-6  a cited banner title corrupted in AGENTS.md         [must REJECT]"
  exit 0
fi

echo "== baseline =="
out="$(HPOINT_AGENTS="$P_AG" HPOINT_HISTORY="$P_HI" bash "$CHECK" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then printf '%s\n' "$out" >&2; echo "HPOINT-MUT: FAIL baseline rc=$rc" >&2; exit 1; fi
grep -q '^RESULT: PASS' <<<"$out" || { echo "HPOINT-MUT: FAIL baseline has no PASS verdict" >&2; exit 1; }
echo "  ok baseline rc=0, PASS verdict present"

# The synthetic title HP-1/HP-2 plant. BOTH THE UNIQUE TOKEN AND THE PRECONDITION
# BELOW ARE LOAD-BEARING, and the precondition is the durable half.
#
# MEASURED DEFECT, 2026-10-21: this session's own HISTORY.md entry recorded HP-1's
# harvested reason, which quotes the checker's own words -- and those words contain
# the planted title verbatim. The checker's rule 3 resolves a pointer by finding its
# title as text anywhere in HISTORY.md, so the durable RECORD made the planted defect
# resolvable: HP-1's mutated copy then PASSED and the matrix reported MISSED (5/6).
# Loud, but defeated -- the state could no longer test anything.
#
# A mutation matrix whose own state descriptions can defeat its states has to say so.
# The token keeps the title out of reach of any prose that quotes this matrix, and the
# precondition makes a collision abort the run with a diagnosis instead of grading a
# state that cannot fail. Both were needed: the token alone is defeated by the next
# record that quotes it, and the precondition alone turns a working matrix red.
PLANT_TITLE='2026-10-19: INTFER_NO_SUCH_RECORD_ANYWHERE_HP_UNRESOLVABLE_PLANT'
assert_plant_unresolvable() { # $1 = state id
  if grep -q "$PLANT_TITLE" "$P_HI" 2>/dev/null; then
    echo "HPOINT-MUT: FAIL $1's plant title already resolves in the checker's" >&2
    echo "  HISTORY.md corpus, so this state cannot fail: a durable record quotes" >&2
    echo "  it. Pick a new PLANT_TITLE token." >&2
    exit 1
  fi
}

# ---- HP-1: a title that exists in NO rule ----
begin_state
assert_plant_unresolvable HP-1
python3 - "$P_AG" "$PLANT_TITLE" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
o = "\n"
n = ("\nMoved to HISTORY.md (\"" + sys.argv[2] + "\"). This file\n"
     "keeps only live directives; the full decision record lives there.\n")
assert t.count(o) > 0
open(p, "w").write(t + n if t.endswith("\n") else t + n)
PY
[ $? -eq 0 ] || { echo "HPOINT-MUT: FAIL HP-1 plant did not land" >&2; exit 1; }
cmp -s "$P_AG" "$ROOT/AGENTS.md" || echo "  ok HP-1 landed (bytes differ)"
run_state HP-1 'unaccounted pointer' 1

# ---- HP-2: the SAME pointer, now excused by an in-place annotation ----
# Self-contained on purpose: run_state restores the copy before each state, so
# this cannot depend on HP-1's residue. The pair HP-1/HP-2 still holds, because
# both plant the identical pointer and differ only by the annotation.
begin_state
assert_plant_unresolvable HP-2
python3 - "$P_AG" "$PLANT_TITLE" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
block = ("\nMoved to HISTORY.md (\"" + sys.argv[2] + "\"). This file\n"
         "keeps only live directives; the full decision record lives there. [DANGLING 2026-10-19:\n"
         "no HISTORY.md section carries this title; the record above is the surviving one]\n")
open(p, "w").write(t + block if t.endswith("\n") else t + block)
PY
[ $? -eq 0 ] || { echo "HPOINT-MUT: FAIL HP-2 plant did not land" >&2; exit 1; }
run_state HP-2 '' 0

# ---- HP-3: remove every citation of a banner entry ----
begin_state
python3 - "$P_AG" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
title = "Flag gate coverage closed"
# The TITLE WRAPS ACROSS LINES in AGENTS.md ("Flag gate\ncoverage closed"), so
# a pattern anchored on the single-spaced spelling matches nothing and the
# state would fail to land -- measured once already, in this very draft.
pat = re.compile(r'HISTORY\.md\s*\("([^"]*Flag gate\s+coverage closed[^"]*)"\)', re.S)
ms = pat.findall(t)
assert ms, "no citation found for the banner entry"
before = t
t = pat.sub(lambda m: 'HISTORY.md (title withdrawn)', t)
assert t != before, "substitution did not change the corpus"
open(p, "w").write(t)
print("  removed %d citation(s) of: %s" % (len(ms), re.sub(r'\s+', ' ', ms[0])))
PY
[ $? -eq 0 ] || { echo "HPOINT-MUT: FAIL HP-3 plant did not land" >&2; exit 1; }
run_state HP-3 'banner entry no longer cited|banner entry' 1

# ---- HP-4: relocate the annotation to the far end of the file ----
# The target is NOT the first [DANGLING in the file. Twelve titles carry an
# annotation and ELEVEN of them also carry a banner entry, so they resolve
# anyway; moving one of those proves nothing. The one title whose ONLY
# resolution is the annotation is INTFER_LMHEAD_FP16 reachability, and the
# scanner's +/-3-line proximity window is what this state has to defeat -- the
# first draft of this state grabbed line 1604's annotation, missed, and the
# matrix correctly reported it.
begin_state
python3 - "$P_AG" <<'PY'
import re, sys
p = sys.argv[1]
txt = open(p).read()
lines = txt.split("\n")
cite = re.search(r'HISTORY\.md\s*\("([^"]*INTFER_LMHEAD_FP16\s+reachability[^"]*)"\)',
                 txt, re.S)
assert cite, "the LMHEAD_FP16 citation is absent from the corpus"
ln = txt.count("\n", 0, cite.start()) + 1
idx = None
for i in range(max(0, ln - 1), min(len(lines), ln + 3)):
    if "[DANGLING" in lines[i]:
        idx = i
        break
assert idx is not None, "no [DANGLING within 3 lines of that citation"
moved = lines.pop(idx)
lines.append("")
lines.append("X " + moved.strip() + "  (moved FAR from the pointer it excuses)")
open(p, "w").write("\n".join(lines))
print("  moved the annotation from line %d to line %d"
      % (idx + 1, len(lines) - 1))
PY
[ $? -eq 0 ] || { echo "HPOINT-MUT: FAIL HP-4 plant did not land" >&2; exit 1; }
run_state HP-4 'unaccounted pointer' 1

# ---- HP-5: empty the harvest (the anti-vacuity half) ----
begin_state
python3 - "$P_AG" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
n = re.sub(r'HISTORY\.md\s*\("[^"]*"\)', 'the history file', t, flags=re.S)
assert n != t, "no citation removed"
assert not re.search(r'HISTORY\.md\s*\("', n), "a citation survived"
open(p, "w").write(n)
PY
[ $? -eq 0 ] || { echo "HPOINT-MUT: FAIL HP-5 plant did not land" >&2; exit 1; }
run_state HP-5 'no citation at all|FAIL' 1

# ---- HP-6: corrupt a cited banner title so it matches nothing ----
begin_state
python3 - "$P_AG" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
pat = re.compile(r'(HISTORY\.md\s*\(")([^"]*INTFER_LA_OUT_TP2[^"]*)("\))', re.S)
m = pat.search(t)
assert m, "no INTFER_LA_OUT_TP2 citation found"
open(p, "w").write(t[:m.start(2)] + m.group(2) + " TYPO" + t[m.end(2):])
PY
[ $? -eq 0 ] || { echo "HPOINT-MUT: FAIL HP-6 plant did not land" >&2; exit 1; }
run_state HP-6 'unaccounted pointer' 1

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

# ---- residue: the LIVE files were never targets, and their shas prove it ----
ag_now="$(sha256sum "$ROOT/AGENTS.md" | cut -d' ' -f1)"
hi_now="$(sha256sum "$ROOT/HISTORY.md" | cut -d' ' -f1)"
if [ "$ag_now" = "$LIVE_AG_SHA" ] && [ "$hi_now" = "$LIVE_HI_SHA" ]; then
  echo "  ok residue: AGENTS.md and HISTORY.md byte-identical to their pre-matrix state"
else
  echo "  FAIL residue: a live file changed during the matrix" >&2
  exit 1
fi

echo
if [ "$CAUGHT" -eq "$total" ]; then
  echo "HPOINT MUTATION MATRIX: PASS ($CAUGHT/$total states)"
  echo "  HP-1: a title no rule accounts for is observable"
  echo "  HP-2: the same title WITH its annotation is correctly ACCEPTED (control)"
  echo "  HP-3: a banner entry nothing cites is observable"
  echo "  HP-4: an annotation far from its pointer does not rescue it"
  echo "  HP-5: an empty harvest is a FAIL, not a pass"
  echo "  HP-6: a corrupted title matches no rule and is observable"
  echo "RESULT: PASS ($CAUGHT/$total states)"
  exit 0
fi
echo "HPOINT MUTATION MATRIX: FAIL ($CAUGHT/$total caught)" >&2
echo "RESULT: FAIL ($CAUGHT/$total caught)" >&2
exit 1
