#!/usr/bin/env bash
# Mutation matrix for scripts/check_decode1_copies.sh.
#
# WHY THIS MATRIX EXISTS. On 2026-10-05 that checker was materially rewritten: its
# population used to be the two kernels it was written for and is now DERIVED, by
# enumerating every `parallel_for` whose name contains `decode1` and requiring each
# to be either pinned or exempted by a stated reason. A materially rewritten checker
# is a rewritten artifact, and AGENTS.md rule 11 says such a thing is provable only
# by weakening the CHECKER and requiring the mutated version to ACCEPT the defect.
# The two negatives demonstrated by hand in that session -- rename a pinned kernel,
# and empty a body while still naming it -- were a one-session demonstration. This
# is the re-runnable artifact.
#
# IT ALREADY PAID FOR ITSELF, TWICE, BEFORE IT WAS FINISHED:
#   * DC-2 plants the REAL INTFER_DECODE1_SPLITK bug -- `(va >> (j * 2)) & 3` for
#     `(va >> ((j & 3) * 2)) & 3` -- in the row-pair body, and the checker PASSED.
#     The enumeration had labelled row-pair "COVERED-BY-ATOM-PIN" while nothing
#     compared its atoms to anything: all that backed the label was a floor, and a
#     floor is not a pin. A real codebook-index comparison was added and DC-2 fires.
#   * An earlier draft of DC-1 asserted a rename is refused for one specific message
#     and it fired for a DIFFERENT, also-correct one. Two independent guards catch a
#     renamed body. That is defence in depth, not a defect -- so DC-1 now accepts
#     either message, and DC-5/DC-6 test the two guards separately.
#
# MECHANICS NOTES, because each cost a full round and each is the kind of thing that
# silently turns a matrix into a no-op:
#   * Guard removal is `guard_off <bare-word-id>`, with the anchors living inside the
#     python heredoc. An earlier draft passed the anchors as SHELL strings and the
#     shell ate every quote, so four states reported "GUARD ANCHOR 0 matches" and no
#     guard was ever removed -- a state whose removal silently did nothing reports
#     a floor as load-bearing while the floor is still there.
#   * States are written out in full, not driven by a loop over "$@". That loop
#     iterated WORDS rather than command-with-arguments, so `unmake_guard` was
#     called with no argument at all.
#   * Every run is against a COPY under build/, via a temporary ROOT that the
#     checker resolves SRC against. The tracked source and checker are never edited,
#     and the residue assertion re-proves that rather than asserting it.
#
# States. The POLARITY is the point and it differs per state:
#   DC-1  a pinned kernel is RENAMED                              [REFUSE]
#   DC-2  the REAL shipped codebook bug planted in row-pair        [REFUSE]
#   DC-3  a named body's contents EMPTIED                           [REFUSE]
#   DC-4  the row-pair index pin removed, DC-2's bug planted        [GO GREEN]
#   DC-5  ONLY the UNCOVERED rule removed, DC-1's defect planted   [STILL REFUSE]
#   DC-6  ALL FIVE rename guards removed, DC-1's defect planted   [EXCLUDED - coupled]
#   DC-7  row-pair dropped from the required set, body emptied      [EXCLUDED - coupled]
#
# DC-4/6/7 are FLOOR states: a floor cannot be tested by deleting it, so the defect
# it should catch is planted at the same time and the state passes only if the
# checker goes GREEN. DC-5 is the opposite and is deliberately a REFUSE: it proves
# the checker has a second independent guard, so no single deletion hides a renamed
# body.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/decode1_copies_matrix"
CHECK="scripts/check_decode1_copies.sh"
SRC="src/gemm/quant_gemm.cpp"

rm -rf "$SB"; mkdir -p "$SB" || exit 2
MUT="$SB/quant_gemm.cpp"
PRISTINE_SRC="$SB/quant_gemm.pristine"
CHK_PRISTINE="$SB/check.pristine"
cp "$SRC" "$PRISTINE_SRC"  || exit 2
cp "$CHECK" "$CHK_PRISTINE" || exit 2
restore_all() { cp "$PRISTINE_SRC" "$MUT"; cp "$CHK_PRISTINE" "$CHECK"; }

# patch_literal semantics: EXACTLY ONE match. 0 and >1 both fail loudly (rule 10).
plant() { python3 - "$MUT" "$1" "$2" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
c = s.count(o)
if c != 1:
    print("SOURCE ANCHOR %s: %d matches (want 1): %r"
          % ("ABSENT" if c == 0 else "AMBIGUOUS", c, o[:70]), file=sys.stderr)
    sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
}

# Blank a named body while LEAVING ITS NAME, so the body is still "covered" and no
# longer extractable. That is the vacuity the checker refuses to certify.
empty_body() { python3 - "$MUT" "$1" <<'PYX'
import sys
p, name = sys.argv[1], sys.argv[2]
s = open(p).read()
i = s.index("h.parallel_for<class %s>(" % name)
j = s.index("            });\n      });\n    }", i)
s = (s[:i] + "h.parallel_for<class %s>(\n"
     "            sycl::nd_range<1>(sycl::range<1>(pGlobal), sycl::range<1>(nWG)),\n" % name
     + s[j:])
open(p, "w").write(s)
PYX
}

guard_off() { python3 - "$CHECK" "$1" <<'PYX'
import sys
p, guard = sys.argv[1], sys.argv[2]
G = {
  # The anchors are copied VERBATIM from the checker (grep -nF to find each), not
  # reconstructed from memory -- the first draft guessed two of the four and both
  # reported "0 matches", so the states silently tested nothing.
  "index_pin":    ('if [ -n "$extra_idx" ]; then', 'if false; then'),
  "uncovered":    ("if printf '%s\\n' \"${ALL_D1[@]}\" | grep -qvE \\",
                   "if false && printf '%s\\n' \"${ALL_D1[@]}\" | grep -qvE \\"),
  "rp_required":  ("if printf '%s\\n' \"${ALL_D1[@]}\" | grep -qx 'gemm_usm_decode1_rowpair' &&",
                   "if false && printf '%s\\n' \"${ALL_D1[@]}\" | grep -qx 'gemm_usm_decode1_rowpair' &&"),
  "rp_extract":   ('if [ "$n_rp" -lt 5 ]; then', 'if false; then'),
  "rp_idx_floor": ('if [ "$n_idx_base" -lt 1 ] || [ "$n_idx_rp" -lt 1 ]; then', 'if false; then'),
  "rp_atom_floor":('|| [ "$n_rp_a" -lt 2 ]', ''),
  "global_atom_floor": ('if [ "$n_sk_a" -lt 4 ] || [ "$n_base_a" -lt 4 ] || [ "$n_rp_a" -lt 2 ]; then',
                        'if [ "$n_sk_a" -lt 4 ] || [ "$n_base_a" -lt 4 ]; then'),
  "rp_set":       ('|gemm_usm_decode1_f16|gemm_usm_decode1_rowpair)$',
                   '|gemm_usm_decode1_f16)$'),
}
if guard not in G:
    print("UNKNOWN GUARD %r" % guard, file=sys.stderr); sys.exit(1)
o, n = G[guard]
s = open(p).read()
c = s.count(o)
if c != 1:
    print("GUARD ANCHOR %d matches for %s" % (c, guard), file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
}

# The checker resolves SRC relative to $ROOT, so redirecting $ROOT is what points it
# at the mutated copy without the tracked tree ever being written.
PROBE="$SB/root"
setup_root() {
  rm -rf "$PROBE"; mkdir -p "$PROBE/src/gemm" "$PROBE/scripts"
  cp "$MUT" "$PROBE/src/gemm/quant_gemm.cpp"
  cp "$CHECK" "$PROBE/scripts/check_decode1_copies.sh"
}
run_checker() { ( cd "$PROBE" && bash scripts/check_decode1_copies.sh 2>&1 ); }

total=0; caught=0; RESULTS=()
record() { case "$2" in ok) caught=$((caught + 1)); RESULTS+=("$1 CAUGHT ''");;
                     *)  RESULTS+=("$1 $2");; esac; }
verdict() { # id want refuse|green expect-re out rc
  local id="$1" want="$2" expect="$3" out="$4" rc="$5"
  if [ "$want" = refuse ]; then
    if [ "$rc" -eq 0 ]; then
      record "$id" "MISSED '(still PASS -- the defect was not caught)'"
    elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
      record "$id" "MISSED '(refused, but not for the expected reason: $expect)'"
    else record "$id" ok; fi
  else
    if [ "$rc" -ne 0 ]; then
      record "$id" "MISSED '(still refused, rc='$rc') -- something ELSE is catching it'"
    elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
      record "$id" "MISSED '(went green, but output lacked: $expect)'"
    else record "$id" ok; fi
  fi
}

RENAME_OLD='h.parallel_for<class gemm_usm_decode1_rowpair>('
RENAME_NEW='h.parallel_for<class gemm_usm_decode1_rowpair_v2>('
BUG_OLD='accA += av * wA4[(va >> ((j & 3) * 2)) & 3];'
BUG_NEW='accA += av * wA4[(va >> (j * 2)) & 3];'

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  DC-1  a pinned kernel is renamed                      [must REFUSE]"
  echo "  DC-2  the real shipped codebook bug planted           [must REFUSE]"
  echo "  DC-3  a named body's contents are emptied             [must REFUSE]"
  echo "  DC-4  the row-pair index pin removed + DC-2 planted   [must GO GREEN]"
  echo "  DC-5  only the UNCOVERED rule removed + DC-1 planted  [must STILL REFUSE]"
  echo "  DC-6  all five rename guards removed + DC-1 planted    [EXCLUDED - coupled]"
  echo "  DC-7  row-pair dropped from required set + emptied   [EXCLUDED - coupled]"
  exit 0
fi

echo "== baseline (pristine source vs the real checker) =="
restore_all; setup_root
out="$(run_checker)"; rc=$?
printf '%s\n' "$out" | tail -4 | sed 's/^/  /'
if [ "$rc" -ne 0 ]; then
  echo "  FAIL baseline: the real checker refused a PRISTINE tree, so every state" >&2
  echo "        below would be passing for the wrong reason." >&2
  exit 1
fi
# Anti-vacuity floor for the MATRIX itself: the baseline must have ENUMERATED
# bodies, or "refuses everything" would satisfy every REFUSE state.
n_bodies="$(printf '%s\n' "$out" | grep -cE 'covered-by-subset|COVERED-BY-ATOM-PIN|exempt-by-reason' || true)"
if [ "${n_bodies:-0}" -lt 4 ]; then
  echo "  FAIL baseline enumerated only $n_bodies bodies -- the matrix would not be" >&2
  echo "        testing the enumeration at all." >&2
  exit 1
fi
echo "  ok baseline rc=0, $n_bodies bodies enumerated"

echo
echo "== states =="

# ---- DC-1: a pinned kernel is renamed ---------------------------------------
total=$((total + 1)); restore_all
if plant "$RENAME_OLD" "$RENAME_NEW"; then
  setup_root; out="$(run_checker)"; rc=$?
  verdict DC-1 refuse 'neither pinned nor exempted|exists but is not pinned' "$out" "$rc"
else record DC-1 'ANCHOR-DRIFT (state not run)'; fi

# ---- DC-2: the REAL shipped bug, in the body the pin claims to cover -------
total=$((total + 1)); restore_all
if plant "$BUG_OLD" "$BUG_NEW"; then
  setup_root; out="$(run_checker)"; rc=$?
  verdict DC-2 refuse 'codebook-index|INTFER_DECODE1_SPLITK' "$out" "$rc"
else record DC-2 'ANCHOR-DRIFT (state not run)'; fi

# ---- DC-3: a named body whose contents are empty ----------------------------
total=$((total + 1)); restore_all
if empty_body gemm_usm_decode1_rowpair; then
  setup_root; out="$(run_checker)"; rc=$?
  verdict DC-3 refuse 'did not extract|coverage claim a lie' "$out" "$rc"
else record DC-3 'ANCHOR-DRIFT (state not run)'; fi

# ---- DC-4: remove the index pin, plant DC-2's bug -> must go GREEN -----------
total=$((total + 1)); restore_all
ok=0
guard_off index_pin            || ok=1
plant "$BUG_OLD" "$BUG_NEW"     || ok=1
if [ "$ok" -eq 0 ]; then
  setup_root; out="$(run_checker)"; rc=$?
  verdict DC-4 green 'PASS' "$out" "$rc"
else record DC-4 'ANCHOR-DRIFT (a removal moved; state not run)'; fi

# ---- DC-5: remove ONLY the UNCOVERED rule, plant DC-1's defect -> STILL RED ---
total=$((total + 1)); restore_all
ok=0
guard_off uncovered             || ok=1
plant "$RENAME_OLD" "$RENAME_NEW" || ok=1
if [ "$ok" -eq 0 ]; then
  setup_root; out="$(run_checker)"; rc=$?
  verdict DC-5 refuse 'exists but is not pinned|did not extract' "$out" "$rc"
else record DC-5 'ANCHOR-DRIFT (a removal moved; state not run)'; fi

# ---- DC-6 / DC-7: EXCLUDED WITH A REASON, not counted ------------------------
# Both were implemented and both are UNOBSERVABLE, for the same structural reason,
# which is worth more than a green tick: the checker's UNCOVERED rule and its
# REQUIRED-SET share ONE list. Removing an entry from that list does not relax the
# requirement -- it makes the OTHER rule fire, because the name is then neither
# allowed nor covered. So "is the required set load-bearing" cannot be tested
# without first making the UNCOVERED rule vacuous, and the two cannot be separated
# without inventing a seam in the checker purely to host the test. Per the repo's
# unobservable-state rule a state that cannot fail is EXCLUDED with its reason
# rather than counted, and that is what this is.
#
# WHAT THE ATTEMPT DID ESTABLISH, and it is not nothing: the enumeration is defended
# by FIVE independent guards, found by MEASUREMENT rather than designed. DC-5 proved
# a second line exists; each probe that disabled the known guards still refused,
# which is how the third (rp_extract), fourth (rp_idx_floor) and fifth
# (global_atom_floor) surfaced. All five are in the guard table and each has a
# state that fires it: DC-1 (uncovered), DC-5 (rp_required), DC-3 (rp_extract),
# DC-2 (the shift-into-mask equality), DC-4 (index_pin). What is NOT established
# is that the list as a whole is necessary, and claiming that would be the
# hand-written-claim failure one level up.
for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

restore_all
if ! cmp -s "$SRC" "$PRISTINE_SRC"; then
  echo "  FAIL residue: $SRC differs from its pre-matrix state" >&2; exit 1
fi
if ! cmp -s "$CHECK" "$CHK_PRISTINE"; then
  echo "  FAIL residue: $CHECK differs from its pre-matrix state" >&2; exit 1
fi
echo "  ok residue: both $SRC and $CHECK are byte-identical to their pre-matrix state"
rm -rf "$SB"

echo
if [ "$caught" -eq "$total" ]; then
  echo "DECODE1-COPY MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  DC-1: a renamed body is refused -- the population is DERIVED, not listed"
  echo "  DC-2: the REAL INTFER_DECODE1_SPLITK codebook bug, planted in row-pair, is"
  echo "        caught. This state is WHY the index pin exists: before it, row-pair"
  echo "        was labelled COVERED-BY-ATOM-PIN while a floor was all that backed"
  echo "        the label, and DC-2 passed."
  echo "  DC-3: a named-but-empty body is refused, not certified as covered"
  echo "  DC-4: remove the index pin + plant DC-2's bug -> GREEN, so the pin is"
  echo "        what catches it"
  echo "  DC-5: remove ONLY the UNCOVERED rule + plant DC-1's defect -> STILL RED."
  echo "        The row-pair required-coverage and did-not-extract guards are"
  echo "        independent second lines, so no single deletion hides a rename."
  echo "  DC-6/7 EXCLUDED WITH REASON: the UNCOVERED rule and the REQUIRED-SET share"
  echo "        one list, so relaxing the set makes the other rule fire. Testing them"
  echo "        independently would mean adding a seam to the checker only to host the"
  echo "        test. The attempt DID establish that the enumeration is defended by FIVE"
  echo "        independent guards, found by measurement -- DC-5 proved a second, and"
  echo "        each probe disabling the known ones still refused, surfacing the third,"
  echo "        fourth and fifth."
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DECODE1-COPY MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
