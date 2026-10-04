#!/usr/bin/env bash
# Mutation matrix for scripts/check_decode1_copies.sh.
#
# *** THESE MUTATE THE ARTIFACT, NOT THE CHECKER. *** The artifact here is
# src/gemm/quant_gemm.cpp -- two deliberately duplicated inline bodies of the
# M=1 decode arithmetic. If the ARTIFACT is correct by construction then
# weakening the CHECKER is unobservable (audit #11), because nothing in the
# checker can tell "the invariant is still enforced" from "the invariant is
# still true". So every state below edits one of the two COPIES in the source
# and requires the checker to REJECT it.
#
# The states are the bugs this has actually seen or could plausibly see, and
# DC-1 is not hypothetical: `(bv >> (j * 2)) & 3` shipped in the split-K copy
# and was found by a 27B reachability probe, not by a checker. DC-1 re-plants
# exactly that.
#
# DC-6 IS THE ALWAYS-RED TWIN AND IS THE ONE MOST LIKELY TO BE FORGOTTEN: it
# mutates BOTH copies identically. A checker that reports drift on any change
# would pass DC-1..DC-5 and fail DC-6, i.e. it would be a checker nobody could
# act on. It must stay green, and it stays green only because the checker
# compares the copies to EACH OTHER rather than to a remembered literal.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/decode1_copies_matrix"
CHECK="scripts/check_decode1_copies.sh"
SRC="src/gemm/quant_gemm.cpp"

mkdir -p "$SB" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

# Exact-literal replacement that refuses unless there is EXACTLY ONE match.
# Drift (0) and ambiguity (2) are both hard stops: a silently-stale anchor turns
# a state into a no-op that scores MISSED and teaches the wrong lesson.
patch_literal() { # $1 file, $2 old, $3 new
  local f="$1" o="$2" n="$3" c
  c="$(grep -cF -- "$o" "$f" || true)"
  if [ "$c" != 1 ]; then
    echo "patch_literal: '$o' found $c occurrence(s) in $(basename "$f"), want 1." >&2
    return 1
  fi
  python3 - "$f" "$o" "$n" <<'PY'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert s.count(o) == 1, "anchor count changed under us"
open(p, 'w').write(s.replace(o, n, 1))
PY
}

# Replace the LAST occurrence (the split-K copy) or the FIRST (the base copy).
patch_nth() { # $1 file, $2 old, $3 new, $4 nth (0=first, -1=last)
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
p, o, n, kth = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
s = open(p).read()
idx = [i for i in range(len(s)) if s.startswith(o, i)]
if not idx:
    print("anchor not found", file=sys.stderr); sys.exit(1)
i = idx[kth]
open(p, 'w').write(s[:i] + n + s[i + len(o):])
PY
}

baseline() {
  bash "$CHECK" >"$SB/base.log" 2>&1; local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL baseline rc=$rc -- the tree must PASS before any state runs" >&2
    tail -5 "$SB/base.log" >&2; return 1
  fi
  grep -q '^RESULT: ' "$SB/base.log" || { echo "  FAIL baseline has no RESULT: line" >&2; return 1; }
  grep -q 'FAIL' "$SB/base.log" && { echo "  FAIL baseline printed a FAIL line" >&2; return 1; }
  echo "  ok baseline rc=0, PASS verdict present, no FAIL line"
}

total=0; caught=0; RESULTS=()

# run_state <id> <expected-regex> <expect-rc: 1=must-reject, 0=must-stay-green>
# *** THE SNAPSHOT IS TAKEN ONCE, BEFORE ANY STATE RUNS. *** The first version
# snapshotted INSIDE run_state, which is after the mutation function had already
# edited the file -- so "restore" wrote the mutated bytes back and every state
# inherited the previous one's damage. MEASURED: DC-1..DC-5 all scored CAUGHT
# and DC-6 then reported a FALSE POSITIVE that had nothing to do with DC-6. A
# state that cannot restore its artifact is not a state, it is a cascade.
PRISTINE="$SB/pristine.src"

run_state() {
  local id="$1" expect="$2" want="$3" out rc
  total=$((total + 1))
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  cp "$PRISTINE" "$SRC"

  if [ "$want" = 1 ]; then
    if [ "$rc" -eq 0 ]; then
      RESULTS+=("$id MISSED '(checker still PASS -- unobservable)'")
    elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
      RESULTS+=("$id MISSED '(rejected, but not with the expected reason)'")
    else
      caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    fi
  else
    # The always-red twin: must stay green.
    if [ "$rc" -eq 0 ]; then
      caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else
      RESULTS+=("$id MISSED '(FALSE POSITIVE -- checker is always-red)'")
    fi
  fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  DC-1  split-K copy: shift loses the (j & 3) mask   [must REJECT]"
  echo "  DC-2  base copy: shift loses the (j & 3) mask      [must REJECT]"
  echo "  DC-3  split-K copy: wtab constant changed          [must REJECT]"
  echo "  DC-4  split-K copy: accumulation form changed      [must REJECT]"
  echo "  DC-5  split-K body removed (extractor must fail)   [must REJECT]"
  echo "  DC-6  BOTH copies mutated identically              [must STAY GREEN]"
  exit 0
fi

echo "== baseline =="
baseline || exit 1

echo "== states =="

SHIFT='(bv >> ((j & 3) * 2)) & 3'
# The shift expression occurs THREE times in this file: the MR kernel, the
# split-K kernel, and the original decode1 kernel -- in that order. Assert it,
# because an index that silently retargets is a state that scores MISSED and
# teaches the wrong lesson (audit #10).
occ() { grep -cF -- "$SHIFT" "$SRC" || true; }

# DC-1: THE BUG THAT ACTUALLY SHIPPED, re-planted in the split-K copy (2nd).
d1() { patch_nth "$SRC" "$SHIFT" '(bv >> (j * 2)) & 3' 1; }

# DC-2: same bug, in the ORIGINAL copy (3rd). A one-directional comparison that
# only ever watched the newer body would pass this state.
d2() { patch_nth "$SRC" "$SHIFT" '(bv >> (j * 2)) & 3' -1; }

# DC-3: a codebook constant drifts in the split-K copy only.
d3() { patch_nth "$SRC" 'scale * (-1.0f / 3.0f)' 'scale * (-1.0f / 2.0f)' 1; }

# DC-4: the accumulation itself drifts in the split-K copy only.
d4() { patch_nth "$SRC" 'acc += av * wtab[code];' 'acc += wtab[code] * av;' 1; }

# DC-5: the whole split-K body is gone, so the extractor matches nothing. A
# comparison that "found no difference" here would be reporting agreement
# between one body and the empty set -- the vacuous-green in its purest form.
d5() { python3 - "$SRC" <<'PY2'
import sys
p = sys.argv[1]; s = open(p).read()
i = s.index('parallel_for<class gemm_usm_decode1_vec_splitk>')
j = s.index('});', i) + 3
open(p, 'w').write(s[:i] + '// DC-5: body removed\n' + s[j:])
PY2
}

# DC-6: THE ALWAYS-RED TWIN. Every copy drifts TOGETHER, so they still agree.
# The checker compares the copies to each other, not to a remembered literal,
# so it must stay GREEN. A checker that went red here would be one nobody could
# act on -- and it is the state most likely to be forgotten, because every other
# state in this matrix is expected to go red.
d6() { python3 - "$SRC" <<'PY2'
import sys
p = sys.argv[1]
o = '(bv >> ((j & 3) * 2)) & 3'
n = '(bv >> ((j & 1) * 2)) & 3'
s = open(p).read()
assert s.count(o) == 3, f'expected 3 occurrences, got {s.count(o)}'
open(p, 'w').write(s.replace(o, n))
PY2
}

cp "$SRC" "$PRISTINE"      # pristine snapshot, BEFORE any state mutates it
n_occ="$(occ)"
echo "-- shift-expression occurrences in $SRC: $n_occ (want 3)"
if [ "$n_occ" != 3 ]; then
  echo "  FAIL the state indices below are anchored to exactly 3 occurrences;" >&2
  echo "        got $n_occ. Fix the indices, do not relax them." >&2
  exit 1
fi

# Mutate FIRST, then measure. Each state restores the artifact before returning.
d1; run_state DC-1 'DIVERGED|cannot certify' 1
d2; run_state DC-2 'DIVERGED|cannot certify' 1
d3; run_state DC-3 'DIVERGED|cannot certify' 1
d4; run_state DC-4 'DIVERGED|cannot certify' 1
d5; run_state DC-5 'did not extract|neither copy' 1
d6; run_state DC-6 '' 0

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

# RESIDUE is measured against the PRE-MATRIX SNAPSHOT, not against HEAD.
# The matrix's contract is "restore the artifact to the state I found it in", not
# "leave the tree clean" -- and the two differ whenever the tree is legitimately
# dirty. MEASURED: the first version compared to HEAD, so a matrix run against a
# tree with an uncommitted fix reported residue FAIL even though every state had
# restored the file perfectly. A residue check that cannot tell "I left a mess"
# from "you had changes already" is a check that trains people to ignore it.
if ! cmp -s "$SRC" "$PRISTINE"; then
  echo "  FAIL residue: $SRC differs from its PRE-MATRIX state" >&2
  diff -u "$PRISTINE" "$SRC" | head -20 >&2
  exit 1
fi
echo "  ok residue: $SRC byte-identical to its PRE-MATRIX state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "DECODE1-COPY MUTATION MATRIX: PASS ($caught/$total states, each for its own reason)"
  echo "  DC-1: the shift bug that actually shipped is observable in seconds"
  echo "  DC-2: the comparison is bidirectional, not newer-copy-only"
  echo "  DC-3: a codebook constant cannot drift in one copy"
  echo "  DC-4: the accumulation cannot drift in one copy"
  echo "  DC-5: an extractor matching nothing is refused, not read as agreement"
  echo "  DC-6: all copies drifting together stays GREEN (not always-red)"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DECODE1-COPY MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total states caught)" >&2
exit 1
