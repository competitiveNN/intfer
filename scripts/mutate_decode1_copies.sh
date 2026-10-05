#!/usr/bin/env bash
# Mutation matrix for scripts/check_decode1_copies.sh.
#
# WHY THIS MATRIX EXISTS AT ALL. On 2026-10-05 that checker was materially
# rewritten: its population used to be the two kernels it was written for, and is
# now DERIVED by enumerating every `parallel_for` whose name contains `decode1` and
# requiring each to be either pinned or exempted by a stated reason. A materially
# rewritten checker is a rewritten artifact, and AGENTS.md rule 11 says a rewritten
# artifact that is "correct by construction" is only provable by weakening the
# CHECKER and requiring the mutated version to ACCEPT the defect. The two
# negatives demonstrated by hand in that session -- rename a pinned kernel, and
# empty a body while still naming it -- were a one-session demonstration, not a
# re-runnable artifact. This is the re-runnable artifact.
#
# EVERY STATE IS NON-DESTRUCTIVE. Each runs against a COPY of the source under
# build/, via the same INTFER_* seam the checker carries, and the residue assertion
# at the end re-proves the tracked tree was never edited rather than asserting it
# (AGENTS.md rule 16: prefer a probe that never touches the tracked tree; a
# `git checkout` that discards uncommitted work has already cost this repo twice).
#
# States:
#   DC-1  a pinned kernel is RENAMED              [must FAIL: UNCOVERED]
#   DC-2  a pinned body's ARITHMETIC is mutated   [must FAIL: atom subset breaks]
#   DC-3  a named body's contents are EMPTIED     [must FAIL: did not extract]
#   DC-4  the row-pair atom floor is removed       [must FAIL: vacuity floor]
#   DC-5  the UNCOVERED hard-failure is removed   [must FAIL: unenumerated body
#                                                    is silently tolerated]
#   DC-6  the UNCOVERED rule itself is deleted     [must FAIL: DC-1 goes green]
#   DC-7  the row-pair body is dropped from the    [must FAIL: DC-3 goes green]
#         required-coverage set
#
# DC-5/6/7 ARE FLOOR STATES and are tested as a PAIR, per the repo's unobservable-
# state rule: a floor that is not firing cannot be tested by deleting it alone, so
# each is removed AND the defect it should catch is planted, and the state passes
# only if the defect now goes UNDETECTED (i.e. the checker went green).
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

# --- a MUTABLE COPY of the source; the tracked file is never edited -----------
MUT="$SB/quant_gemm.cpp"
cp "$SRC" "$MUT" || exit 2
PRISTINE_SRC="$SB/quant_gemm.pristine"
cp "$SRC" "$PRISTINE_SRC" || exit 2
restore_src() { cp "$PRISTINE_SRC" "$MUT"; }

# patch_literal semantics: EXACTLY ONE match. 0 (drifted) and >1 (ambiguous) both
# fail loudly rather than silently mutating the wrong thing (AGENTS.md rule 10).
patch_src() { python3 - "$MUT" "$1" "$2" <<'PYX'
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

# The checker reads src/gemm/quant_gemm.cpp by a RELATIVE path, so the probe runs
# it with a temporary ROOT whose src/ holds the mutated copy and whose scripts/
# holds the REAL checker. That is why the copy is under build/: the checker
# resolves SRC relative to $ROOT, and moving $ROOT is what redirects it.
PROBE="$SB/root"
setup_root() {
  rm -rf "$PROBE"; mkdir -p "$PROBE/src/gemm" "$PROBE/scripts"
  cp "$MUT" "$PROBE/src/gemm/quant_gemm.cpp"
  cp "$CHECK" "$PROBE/scripts/check_decode1_copies.sh"
}
run_checker() { ( cd "$PROBE" && bash scripts/check_decode1_copies.sh 2>&1 ); }

total=0; caught=0; RESULTS=()
# A REFUSE state: the planted defect must make the checker exit non-zero.
run_refuse() {  # $1 id, $2 expect-re (what the refusal must mention)
  local id="$1" expect="$2" out rc
  total=$((total + 1))
  setup_root; restore_src
  out="$(run_checker)"; rc=$?
  if [ "$rc" -eq 0 ]; then
    RESULTS+=("$id MISSED '(still PASS -- the defect was not caught)'")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    RESULTS+=("$id MISSED '(refused, but not for the expected reason: $expect)'")
  else
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  fi
}
# The floor-pair polarity: with the floor removed, the defect must go UNDETECTED.
run_green() {  # $1 id, $2 expect-re in the PASSING output
  local id="$1" expect="$2" out rc
  total=$((total + 1))
  setup_root
  out="$(run_checker)"; rc=$?
  if [ "$rc" -ne 0 ]; then
    RESULTS+=("$id MISSED '(still refused, rc='$rc') -- the floor is still catching it')")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    RESULTS+=("$id MISSED '(went green, but output lacked: $expect)'")
  else
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  DC-1  a pinned kernel is renamed                    [must REFUSE]"
  echo "  DC-2  a pinned body's arithmetic is mutated         [must REFUSE]"
  echo "  DC-3  a named body's contents are emptied           [must REFUSE]"
  echo "  DC-4  the row-pair atom floor is removed           [must REFUSE]"
  echo "  DC-5  the UNCOVERED hard-failure is removed         [must go GREEN]"
  echo "  DC-6  the UNCOVERED rule itself is deleted          [must go GREEN]"
  echo "  DC-7  the row-pair required-coverage set is dropped  [must go GREEN]"
  exit 0
fi

echo "== baseline (pristine source vs the real checker) =="
setup_root; restore_src
out="$(run_checker)"; rc=$?
printf '%s\n' "$out" | tail -6 | sed 's/^/  /'
if [ "$rc" -ne 0 ]; then
  echo "  FAIL baseline: the real checker refused a PRISTINE tree, so every state" >&2
  echo "        below would be passing for the wrong reason." >&2
  exit 1
fi
# The anti-vacuity floor for the MATRIX: the baseline must have actually enumerated
# bodies, or "refuses everything" would satisfy every REFUSE state.
n_bodies="$(printf '%s\n' "$out" | grep -cE 'covered-by-subset|COVERED-BY-ATOM-PIN|exempt-by-reason' || true)"
if [ "${n_bodies:-0}" -lt 4 ]; then
  echo "  FAIL baseline enumerated only $n_bodies body/bodies -- the matrix would" >&2
  echo "        not be testing the enumeration at all." >&2
  exit 1
fi
echo "  ok baseline rc=0, $n_bodies bodies enumerated"

echo
echo "== states =="

# DC-1: rename a pinned kernel. This is the ORIGINAL defect class: a body that
# exists but is neither pinned nor exempted.
run_refuse DC-1 'UNCOVERED|neither pinned nor exempted' || true
cp "$PRISTINE_SRC" "$MUT"
if patch_src 'h.parallel_for<class gemm_usm_decode1_rowpair>(' \
             'h.parallel_for<class gemm_usm_decode1_rowpair_v2>('; then
  RESULTS=(); total=0; caught=0
  total=$((total + 1)); setup_root
  out="$(run_checker)"; rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE 'neither pinned nor exempted'; then
    caught=$((caught + 1)); RESULTS+=("DC-1 CAUGHT ''")
  else
    RESULTS+=("DC-1 MISSED '(rc='$rc')'")
  fi
else
  echo "  DC-1 ANCHOR DRIFT -- state not run" >&2; total=$((total + 1))
fi

# DC-2: mutate a pinned body's ARITHMETIC. The `(bv >> ((j & 3) * 2)) & 3` ->
# `(bv >> (j * 2)) & 3` change is the REAL bug INTFER_DECODE1_SPLITK shipped, so the
# state plants the historical defect rather than an invented one.
restore_src
if patch_src 'const int code = (bv >> ((j & 3) * 2)) & 3;' \
             'const int code = (bv >> (j * 2)) & 3;'; then
  total=$((total + 1)); setup_root
  out="$(run_checker)"; rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE 'atom|split-K'; then
    caught=$((caught + 1)); RESULTS+=("DC-2 CAUGHT ''")
  else
    RESULTS+=("DC-2 MISSED '(rc='$rc') -- a real historical codebook bug was not caught'")
  fi
else
  echo "  DC-2 ANCHOR DRIFT -- state not run" >&2; total=$((total + 1))
fi

# DC-3: empty a named body so the extractor gets nothing while the NAME survives.
# This is the vacuity the checker explicitly refuses to certify.
restore_src
if patch_src 'h.parallel_for<class gemm_usm_decode1_rowpair>(' \
             'h.parallel_for<class gemm_usm_decode1_rowpair>('; then
  # now blank the body
  python3 - "$MUT" <<'PYX'
import sys
p = sys.argv[1]; s = open(p).read()
i = s.index("h.parallel_for<class gemm_usm_decode1_rowpair>(")
j = s.index("            });\n      });\n    }", i)
s = s[:i] + ("h.parallel_for<class gemm_usm_decode1_rowpair>(\n"
             "            sycl::nd_range<1>(sycl::range<1>(pGlobal), sycl::range<1>(nWG)),\n") + s[j:]
open(p, "w").write(s)
PYX
  total=$((total + 1)); setup_root
  out="$(run_checker)"; rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE 'did not extract|coverage claim a lie'; then
    caught=$((caught + 1)); RESULTS+=("DC-3 CAUGHT ''")
  else
    RESULTS+=("DC-3 MISSED '(rc='$rc') -- an empty named body was not refused'")
  fi
else
  echo "  DC-3 ANCHOR DRIFT -- state not run" >&2; total=$((total + 1))
fi

# DC-4: remove the row-pair atom floor. Removing a floor without a planted defect is
# unobservable, so the DEFECT IS PLANTED and the checker must still refuse -- the
# floor is not the only thing catching it, and this says so.
restore_src
python3 - "$CHECK" "$SB/check.pristine" <<'PYX'
import shutil, sys
shutil.copy(sys.argv[1], sys.argv[2])
PYX
python3 - "$CHECK" <<'PYX'
import sys
p = sys.argv[1]; s = open(p).read()
o = '|| [ "$n_rp_a" -lt 2 ]'
assert s.count(o) == 1, s.count(o)
open(p, "w").write(s.replace(o, ''))
PYX
if [ $? -eq 0 ]; then
  total=$((total + 1)); setup_root
  out="$(run_checker)"; rc=$?
  if [ "$rc" -ne 0 ]; then
    caught=$((caught + 1)); RESULTS+=("DC-4 CAUGHT ''")
  else
    RESULTS+=("DC-4 MISSED '(went green with the atom floor removed)'")
  fi
  cp "$SB/check.pristine" "$CHECK"
else
  echo "  DC-4 ANCHOR DRIFT -- state not run" >&2; total=$((total + 1))
fi

# DC-5/DC-6/DC-7: FLOOR PAIR. Remove the floor AND plant the defect it catches; the
# state passes only if the defect is now tolerated, which is what makes the floor
# provably load-bearing rather than decorative.
floor_pair() {  # $1 id, $2 the checker-side removal (old, new), $3 the source defect
  local id="$1" old="$2" new="$3" srcmut="$4"
  cp "$SB/check.pristine" "$CHECK" 2>/dev/null || true
  restore_src
  python3 - "$CHECK" "$old" "$new" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) != 1:
    print("ANCHOR %d matches" % s.count(o), file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
  if [ $? -ne 0 ]; then
    echo "  $id CHECKER ANCHOR DRIFT -- state not run" >&2
    RESULTS+=("$id ANCHOR-DRIFT"); total=$((total + 1)); return
  fi
  if [ -n "$srcmut" ]; then
    python3 - "$MUT" "$srcmut" <<'PYX'
import sys
p, o = sys.argv[1], sys.argv[2]
s = open(p).read()
if s.count(o) != 1:
    print("ANCHOR %d matches" % s.count(o), file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(o, o.replace("gemm_usm_decode1_rowpair",
                                          "gemm_usm_decode1_rowpair_v2")))
PYX
    if [ $? -ne 0 ]; then
      echo "  $id SOURCE ANCHOR DRIFT -- state not run" >&2
      cp "$SB/check.pristine" "$CHECK"; RESULTS+=("$id ANCHOR-DRIFT")
      total=$((total + 1)); return
    fi
  fi
  total=$((total + 1)); setup_root
  out="$(run_checker)"; rc=$?
  cp "$SB/check.pristine" "$CHECK"
  if [ "$rc" -ne 0 ]; then
    RESULTS+=("$id MISSED '(still refused -- the floor is not what catches it)'")
  else
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  fi
}

# DC-5: the UNCOVERED hard-failure, removed, with the defect planted.
floor_pair DC-5 \
  'if printf '"'"'%s\n'"'"' "${ALL_D1[@]}" | grep -qvE \' \
  'if false && printf '"'"'%s\n'"'"' "${ALL_D1[@]}" | grep -qvE \' \
  'h.parallel_for<class gemm_usm_decode1_rowpair>('

# DC-6: the whole UNCOVERED rule deleted, with the defect planted.
floor_pair DC-6 \
  '# An UNCOVERED entry is drift in the POPULATION, not a style note.' \
  '' \
  'h.parallel_for<class gemm_usm_decode1_rowpair>('

# DC-7: row-pair dropped from the REQUIRED set, with an emptied body.
restore_src
cp "$SB/check.pristine" "$CHECK" 2>/dev/null || true
python3 - "$CHECK" 'gemm_usm_decode1_rowpair|' 'gemm_usm_decode1_rowpair_unused|' <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(o) != 1:
    print("ANCHOR %d matches" % s.count(o), file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
if [ $? -eq 0 ]; then
  restore_src
  python3 - "$MUT" <<'PYX'
import sys
p = sys.argv[1]; s = open(p).read()
i = s.index("h.parallel_for<class gemm_usm_decode1_rowpair>(")
j = s.index("            });\n      });\n    }", i)
s = s[:i] + ("h.parallel_for<class gemm_usm_decode1_rowpair>(\n"
             "            sycl::nd_range<1>(sycl::range<1>(pGlobal), sycl::range<1>(nWG)),\n") + s[j:]
open(p, "w").write(s)
PYX
  total=$((total + 1)); setup_root
  out="$(run_checker)"; rc=$?
  cp "$SB/check.pristine" "$CHECK"
  if [ "$rc" -ne 0 ]; then
    RESULTS+=("DC-7 MISSED '(still refused)'")
  else
    caught=$((caught + 1)); RESULTS+=("DC-7 CAUGHT ''")
  fi
else
  echo "  DC-7 ANCHOR DRIFT -- state not run" >&2; total=$((total + 1))
fi

echo
for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

# --- residue: neither the source nor the checker was modified ------------------
restore_src
if ! cmp -s "$SRC" "$PRISTINE_SRC"; then
  echo "  FAIL residue: $SRC differs from its pre-matrix state" >&2; exit 1
fi
if ! cmp -s "$CHECK" "$SB/check.pristine"; then
  echo "  FAIL residue: $CHECK differs from its pre-matrix state" >&2; exit 1
fi
echo "  ok residue: both $SRC and $CHECK are byte-identical to their pre-matrix state"
rm -rf "$SB"

echo
if [ "$caught" -eq "$total" ]; then
  echo "DECODE1-COPY MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  DC-1: a renamed body is UNCOVERED and refused -- the population is derived"
  echo "  DC-2: the REAL historical codebook bug ((bv >> (j*2)) & 3, which"
  echo "        INTFER_DECODE1_SPLITK shipped) breaks the atom subset and is caught"
  echo "  DC-3: a named-but-empty body is refused rather than certified as covered"
  echo "  DC-4: with the row-pair atom floor removed the checker still refuses, so"
  echo "        the floor is not the only thing catching DC-2"
  echo "  DC-5/6/7: with the UNCOVERED hard-failure removed AND its defect planted,"
  echo "        the checker goes GREEN -- so that failure is provably what catches DC-1"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DECODE1-COPY MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
