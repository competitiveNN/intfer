#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_hw_constants.sh.
#
# WHAT THE CHECKER CLAIMS: a document that states a hardware capability figure
# must name its source, and the citation must be in the same PARAGRAPH as the
# figure. That rule exists because "512 Xe cores / ~179 GB/s" (the A770's figures)
# sat in AGENTS.md, three code comments and two live reports, and a whole session of
# optimization was built on them. `git log -S '512 Xe'` shows it entered at
# 40219bf and was only ever propagated.
#
# WHY THIS MATRIX MUTATES THE CHECKER AND NOT A DOCUMENT. AGENTS.md rule 11:
# "If the artifact under test is correct by construction, weakening the CHECKER is
# unobservable -- mutate the ARTIFACT and require the checker to REJECT it." Here
# the artifact under test IS the checker (a shell script whose rule is one regex and
# one paragraph split). A document planted with a wrong figure would prove nothing,
# because the checker would reject a correct document's typo just as readily.
# So every state degrades the RULE and asserts the degraded rule now ACCEPTS the
# defect -- the only thing that distinguishes "the rule works" from "the fixture
# happened to trip something else".
#
# *** ONE CORPUS PER STATE, AND THAT IS THE NON-OBVIOUS PART. *** The first version
# planted all three defect classes in ONE corpus, so every state that removed a
# single class still had the other two to trip over and reported "still REJECTED" --
# six states that could not fail for a reason anyone could act on, i.e. exactly the
# unobservable-state defect (audit #21). Each state below therefore gets a corpus
# holding ONLY the defect class it attacks, so removing that class must turn the
# whole corpus green.
#
# EVERY STATE IS NON-DESTRUCTIVE. Each runs against a MUTABLE COPY of the checker
# under build/, pointed at a sandbox corpus through the INTFER_DOCS_DIR seam the
# checker already carries. The tracked tree is never edited (AGENTS.md rule 16), and
# the residue assertion at the end re-proves that rather than asserting it.
#
# States:
#   HW-1  CITATION( no longer required            [self-check must FIRE]
#   HW-2  the `xe cores` alternative removed       [self-check must FIRE]
#   HW-3  the `tflops` alternative removed         [self-check must FIRE]
#   HW-4  case-insensitivity removed              [self-check must FIRE]
#   HW-5  the bandwidth byte class deleted        [self-check must FIRE]
#   HW-1b same as HW-1 but the self-check GATE also neutered  [must ACCEPT]
#   HW-6  the corpus floor removed                [must ACCEPT an all-prose corpus]
#   HW-7  a self-check FIXTURE neutered           [self-check must FIRE]
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/hw_constants_matrix"
CHECK="scripts/checks/check_hw_constants.sh"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

# --- corpora: ONE defect class each -------------------------------------------
# Planted as DATA under build/, never as literals in this file: the spec vocabulary
# contains quotes and parentheses, and a hand-quoted version of it was wrong before
# it was ever run once already.
python3 - "$SB" <<'PYEOF'
import os, sys
sb = sys.argv[1]
XE = " Xe cores"
CITE = "CITATION(techpowerup arc-b580.c4244)"

def corpus(name, docs):
    root = os.path.join(sb, name)
    for sub in ("docs", "reports"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    with open(os.path.join(root, "AGENTS.md"), "w") as f:
        f.write("# live\n\nSub-group size is 16, register count is 255.\n")
    for rel, body in docs.items():
        with open(os.path.join(root, rel), "w") as f:
            f.write(body)

def clean_pair():
    return {"docs/good_cited.md": "Decode runs at ~179 GB/s  %s\n" % CITE,
            "docs/good_prose.md":  "The kernel is bandwidth-bound: a claim, not a figure.\n"}

# --- the three single-class defective corpora ---------------------------------
corpus("c_bw",   dict(clean_pair(), **{"docs/bad.md": "Decode runs at ~179 GB/s on the B580.\n"}))
corpus("c_cores",dict(clean_pair(), **{"docs/bad.md": "A layer GEMM spans 512%s.\n" % XE}))
corpus("c_tflops",dict(clean_pair(), **{"docs/bad.md": "FP32 throughput is 13.67 TFLOPS.\n"}))

# --- a corpus with NO figure at all: only reachable by the floor state -------
corpus("c_nofig", {"docs/prose.md": "The kernel is bandwidth-bound; that is a claim.\n"})

# --- the full three-class corpus, for the baseline and for HW-1 ---------------
corpus("c_all", dict(clean_pair(), **{
    "docs/bad_bw.md":     "Decode runs at ~179 GB/s on the B580.\n",
    "docs/bad_cores.md":  "A layer GEMM spans 512%s.\n" % XE,
    "docs/bad_tflops.md": "FP32 throughput is 13.67 TFLOPS.\n"}))
PYEOF
[ $? -eq 0 ] || { echo "FAIL: could not plant corpora" >&2; exit 1; }

# --- the checker under mutation ------------------------------------------------
CHK="$SB/check_under_test.sh"
PRISTINE="$SB/check.pristine"
cp "$CHECK" "$PRISTINE" || exit 2
cp "$CHECK" "$CHK" || exit 2
restore() { cp "$PRISTINE" "$CHK"; }

# patch_literal semantics: EXACTLY ONE match. 0 (drifted) and >1 (ambiguous) both
# fail loudly rather than silently mutating the wrong thing (AGENTS.md rule 10).
patch_check() { python3 - "$CHK" "$1" "$2" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
c = s.count(o)
if c != 1:
    print("ANCHOR %s: %d matches (want exactly 1): %r" %
          ("ABSENT" if c == 0 else "AMBIGUOUS", c, o[:70]), file=sys.stderr)
    sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
}

run_checker() { INTFER_DOCS_DIR="$1" bash "$CHK" 2>&1; }

total=0; caught=0; RESULTS=()

# *** THE POLARITY OF EVERY RULE-CLASS STATE, and it was WRONG at first. *** The
# first version asserted that degrading a rule class makes the checker ACCEPT a
# defect. It does not -- and that is the checker's own self-check doing its job: a
# deleted unit class breaks a planted fixture, the self-check notices, and the run
# refuses with "self-check did not behave". Six states were therefore reported
# "still REJECTED -- the rule was not load-bearing", which is a verdict that reads
# as "this direction is a dead end" when the truth is the opposite: the rule class
# is load-bearing AND the self-check is what proves it, in-process, on every run.
#
# So the correct polarity for a rule-class state is REFUSE-WITH-THE-SELF-CHECK
# message, and the ACCEPT direction belongs to exactly ONE state (HW-1b), where the
# self-check is ALSO neutered. That pair is the real proof: HW-1 shows the
# self-check catches the degradation, HW-1b shows the degraded rule really would
# have accepted the defect -- so HW-1's catch is attributable and not incidental.

# A rule-class state: degrade one class of the rule, REQUIRE the self-check to fire.
run_refuse() { # $1 id, $2 corpus dir
  local id="$1" corpus="$2" out rc
  total=$((total + 1))
  out="$(run_checker "$SB/$corpus")"; rc=$?
  restore
  printf '%s\n' "$out" >"$SB/$id.log"
  if [ "$rc" -eq 0 ]; then
    RESULTS+=("$id MISSED '(PASSED -- the degraded rule was NOT caught)'")
  elif ! printf '%s' "$out" | grep -q 'self-check did not behave'; then
    RESULTS+=("$id MISSED '(refused, but not by the self-check; rc='$rc')")
  else
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  fi
}

# The always-red twin: same degradation, but the self-check GATE is neutered too, so
# the run must then go GREEN on the defective corpus. Proves the self-check is the
# thing doing the catching.
run_accept() { # $1 id, $2 corpus dir, $3 expect-re in the PASS output
  local id="$1" corpus="$2" expect="$3" out rc
  total=$((total + 1))
  out="$(run_checker "$SB/$corpus")"; rc=$?
  restore
  printf '%s\n' "$out" >"$SB/$id.log"
  if [ "$rc" -ne 0 ]; then
    RESULTS+=("$id MISSED '(still REJECTED, rc='$rc') -- not a vacuous-checker demo")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    RESULTS+=("$id MISSED '(accepted, but output lacked: $expect)'")
  else
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  fi
}

# A fixture is neutered so a planted REJECT case becomes acceptable; the self-check
# must notice that its own expectation no longer holds.
run_twin() { # $1 id, $2 corpus dir
  local id="$1" corpus="$2" out rc
  total=$((total + 1))
  out="$(run_checker "$SB/$corpus")"; rc=$?
  restore
  printf '%s\n' "$out" >"$SB/$id.log"
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'self-check did not behave'; then
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  else
    RESULTS+=("$id MISSED '(the neutered fixture went unnoticed, rc='$rc') -- vacuous green)")
  fi
}

state_refuse() { # $1 id, $2 corpus, $3 old, $4 new
  if patch_check "$3" "$4"; then run_refuse "$1" "$2"
  else RESULTS+=("$1 ANCHOR-DRIFT '(state not run; the anchor moved -- update the matrix)'")
       total=$((total + 1)); fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  HW-1  CITATION( no longer required            [self-check must FIRE]"
  echo "  HW-2  the 'xe cores' alternative removed       [self-check must FIRE]"
  echo "  HW-3  the 'tflops' alternative removed         [self-check must FIRE]"
  echo "  HW-4  case-insensitivity removed              [self-check must FIRE]"
  echo "  HW-5  the bandwidth byte class deleted        [self-check must FIRE]"
  echo "  HW-1b as HW-1, self-check GATE also neutered   [must ACCEPT]"
  echo "  HW-6  the corpus floor removed                [must ACCEPT an all-prose corpus]"
  echo "  HW-7  a self-check FIXTURE neutered           [self-check must FIRE]"
  exit 0
fi

echo "== baseline (real rule vs c_all: 3 defect classes + 2 clean docs) =="
out="$(run_checker "$SB/c_all")"; rc=$?
printf '%s\n' "$out" | sed 's/^/  /'
if [ "$rc" -eq 0 ]; then
  echo "  FAIL baseline: the real checker ACCEPTED three uncited figures, so no" >&2
  echo "        state below can mean anything." >&2
  exit 1
fi
n_bad="$(printf '%s' "$out" | grep -c 'uncited figure' || true)"
if [ "${n_bad:-0}" -lt 3 ]; then
  echo "  FAIL baseline flagged ${n_bad:-0} of 3 planted defect classes (rc=$rc)" >&2
  exit 1
fi
echo "  ok baseline rc=$rc, all 3 planted defect classes refused"

echo
echo "== per-class corpora are rejected by the real rule (the states' premise) =="
for c in c_bw c_cores c_tflops; do
  out="$(run_checker "$SB/$c")"; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  FAIL premise $c: the real rule accepted a single-class defect corpus" >&2
    exit 1
  fi
  echo "  ok $c refused by the real rule (rc=$rc)"
done
out="$(run_checker "$SB/c_nofig")"; rc=$?
if [ "$rc" -eq 0 ]; then
  echo "  FAIL premise c_nofig: an all-prose corpus went green, so the floor is" >&2
  echo "        not the thing distinguishing empty from clean -- HW-6 is moot." >&2
  exit 1
fi
printf '%s' "$out" | grep -q 'harvested nothing to judge' \
  || { echo "  FAIL premise c_nofig: refused for the WRONG reason" >&2; exit 1; }
echo "  ok c_nofig refused BY THE FLOOR (harvested nothing to judge)"

echo
echo "== states =="
PASSMARK='HW CONSTANTS: PASS'
CITATION_RULE='missing = [p for p in hits if "CITATION(" not in p]'
CITATION_DEAD='missing = []  # (CITATION requirement DELETED)'
# NOT "missing = [p for p in hits]". That was the first attempt and it is a NO-OP for
# the verdict: dropping the CITATION filter makes missing EQUAL hits, which is still
# non-empty, so the rule keeps rejecting everything and the state reports "still
# REJECTED" for a reason that has nothing to do with the citation. Degrading the rule
# means making it claim nothing is ever missing.
SELFGATE='if [ "$self_bad" -eq 0 ]; then'
SELFGATE_DEAD='if true; then'

# --- rule-class states: the self-check must FIRE -------------------------------
state_refuse HW-1 c_all \
  "$CITATION_RULE" "$CITATION_DEAD"

state_refuse HW-2 c_cores \
  '    r"[0-9]+\s*(xe[\s-]*cores?)"      # 20 Xe cores / 512 Xe-cores' \
  '    r"(zzz)"                           # (xe-core alternative DELETED)'

state_refuse HW-3 c_tflops \
  '    r"|[0-9]+(\.[0-9]+)?\s*tflops",     # 13.67 TFLOPS' \
  '    r"|(zzz)",                          # (tflops alternative DELETED)'

# The corpus writes "Xe cores" with a capital X, so a case-sensitive rule whose
# pattern says "xe" cannot see it. This is "matches one spelling and silently
# stops finding the defect".
state_refuse HW-4 c_cores \
  '    re.IGNORECASE)' '    )'

state_refuse HW-5 c_bw \
  '    r"|[0-9]+(\.[0-9]+)?\s*(g|t)i?b/s"  # 456 GB/s, 35.5 GiB/s, 179 gb/s' \
  '    r"|(zzz)"                           # (bandwidth class DELETED)'

# --- HW-1b: the always-red twin, and the ONLY accept state ---------------------
# HW-1 says "the self-check catches a deleted CITATION requirement". That is only
# evidence if the degraded rule really would have accepted the defect -- otherwise
# the catch could be incidental (a different fixture failing). So neuter the
# self-check GATE as well and require the defective corpus to go GREEN.
if patch_check "$CITATION_RULE" "$CITATION_DEAD" && patch_check "$SELFGATE" "$SELFGATE_DEAD"; then
  run_accept HW-1b c_all "$PASSMARK"
else
  RESULTS+=("HW-1b ANCHOR-DRIFT '(state not run; an anchor moved)'")
fi

# --- HW-6: the corpus floor, which no fixture covers ---------------------------
if patch_check 'if [ "$sites" -eq 0 ]; then' 'if false; then'; then
  run_accept HW-6 c_nofig "$PASSMARK"
else
  RESULTS+=("HW-6 ANCHOR-DRIFT '(state not run; the floor anchor moved)'")
fi

# --- HW-7: a self-check FIXTURE neutered ---------------------------------------
# The planted "must be REJECTED" Xe-core case is written with a citation, so the
# real (correct) rule accepts it and the self-check must notice that its own
# expectation no longer holds.
if patch_check \
  '    "bad_cores.sh":        "the card has 512%s\n" % XE,' \
  '    "bad_cores.sh":        "the card has 512%s  %s\n" % (XE, CITE),'; then
  run_twin HW-7 c_all
else
  RESULTS+=("HW-7 ANCHOR-DRIFT '(state not run; the fixture line moved)'")
fi

echo
for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

# --- residue: the tracked tree was never modified ------------------------------
restore
if ! cmp -s "$CHECK" "$PRISTINE"; then
  echo "  FAIL residue: $CHECK differs from its pre-matrix state" >&2
  exit 1
fi
echo "  ok residue: $CHECK is byte-identical to its pre-matrix state"
rm -rf "$SB"

echo
if [ "$caught" -eq "$total" ]; then
  echo "HW-CONSTANTS MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  HW-1..HW-5: each class of the rule is load-bearing, and the checker's own"
  echo "        self-check is what proves it -- deleting a unit class, or the"
  echo "        CITATION requirement, or case-insensitivity, is caught in-process"
  echo "  HW-1b: with the self-check gate ALSO neutered the degraded rule really does"
  echo "        accept a corpus of three uncited figures, so HW-1's catch is"
  echo "        attributable to the self-check and not to something incidental"
  echo "  HW-6: the corpus floor is load-bearing and is NOT covered by any fixture --"
  echo "        it is what distinguishes an EMPTY harvest from a clean tree (the"
  echo "        c_nofig premise above proves the real rule refuses it BY the floor)"
  echo "  HW-7: a neutered self-check FIXTURE is caught too, so none of the above is"
  echo "        unfalsifiable"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "HW-CONSTANTS MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
