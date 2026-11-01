#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_dri_probe.sh.
#
# WHAT IS MUTATED, AND WHY IT IS THE CORPUS AND NOT THE CHECKER (audit #21).
# The rule and the subject are both shell scripts, and the checker is correct by
# construction on a clean tree. So a state that WEAKENS THE CHECKER is a no-op with
# extra steps -- exactly what this repo's first wasted matrix attempt looked like
# (0/4, every stub correct so the checker kept passing). The observable difference
# has to come from damaging an ARTIFACT: here, a script in the corpus that
# identifies a card by DRM minor, or one that defeats the rule by another route.
# The corpus is handed over through INTFER_SCRIPTS_DIR, so the real scripts/ is
# never touched.
#
# States:
#   D-A  a corpus script names two literal minors      -> must REJECT
#   D-B  ... through the INTFER_DRI_ROOT seam           -> must REJECT (defeated v1)
#   D-C  ... with a by-path mention nearby              -> must REJECT (defeated v2)
#   D-D  a corpus of scripts that probe NOTHING         -> must REFUSE (empty census)
#   D-E  the single-minor LIVENESS probe is made to fail -> the checker must still
#         PASS. THE CONVERSE. Without it, a rule that flags every render-node probe
#         satisfies D-A..D-D and this whole matrix proves nothing. It is the state
#         that gets forgotten, so it is a state.
#
# Exit: 0 = every state behaved as specified, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECK="$SCRIPT_DIR/../checks/check_dri_probe.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/dri_probe_mutate"
LOG="$BUILD_DIR/dri_probe_mutate"
SB="$(mktemp -d "${TMPDIR:-/dev/shm}/dri_mut.XXXXXX")"
mkdir -p "$STAMP" "$LOG"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

# Sourced shared helpers, so this matrix does not carry a second copy.
. "$SCRIPT_DIR/../lib/_gate_pred.sh"
. "$SCRIPT_DIR/../lib/_gate_enum.sh"
. "$SCRIPT_DIR/../lib/_gate_state.sh"

# A corpus that looks like the real one: one compliant by-path probe, one
# single-minor liveness probe. Both must stay clean in every state.
mk_corpus() {
  rm -rf "$SB"; mkdir -p "$SB"
  cat >"$SB/zz_resolved.sh" <<'EOF'
probe() { ls /dev/dri/by-path/pci-0000:04:00.0-render >/dev/null; }
EOF
  cat >"$SB/zz_liveness.sh" <<'EOF'
probe() { python3 -c "open('/dev/dri/renderD128','rb').close()"; }
EOF
}
run_check() { INTFER_SCRIPTS_DIR="$SB" bash "$CHECK" 2>&1; }

echo "== DRI PROBE MUTATION MATRIX (corpus is handed over via INTFER_SCRIPTS_DIR) =="

echo "-- baseline: a compliant corpus must PASS --"
mk_corpus
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "BASELINE FAIL -- the checker does not pass on a compliant corpus:"
  printf '%s\n' "$b" | tail -8; exit 1
fi
if ! printf '%s' "$b" | grep -q 'zz_resolved.sh'; then
  echo "BASELINE FAIL -- the corpus was not even examined (an empty harvest)" >&2
  exit 1
fi
echo "  ok    baseline passes and the corpus was examined"

caught=0; total=0
SUMMARY="$LOG/summary.txt"; : >"$SUMMARY"

# run_reject <id> <why> <mutation-fn>: the checker must exit NON-ZERO.
run_reject() {
  local id="$1" why="$2"
  total=$((total + 1))
  mk_corpus
  if ! MUTATE_FN; then
    echo "  $id MISSED -- the mutation did not apply (STATE fault, not a mechanism)"
    gate_state_record --raw "$SUMMARY" "$id MISSED (mutation fn)"; mk_corpus; return 1
  fi
  local o r
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" >"$LOG/$id.log"
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker still exited 0 with the defect present ($why)"
    gate_state_record --raw "$SUMMARY" "$id MISSED (rc=0)"; mk_corpus; return 1
  fi
  # POLARITY NAMED at the call site, never inferred.
  if ! pred_expect_present "$id rejection" "$o" 'DRI PROBE: FAIL'; then
    gate_state_record --raw "$SUMMARY" "$id MISSED (rejection not reported)"; mk_corpus; return 1
  fi
  echo "  $id caught -- rc=$r, verdict-polarity=expect-present: $why"
  gate_state_record --raw "$SUMMARY" "$id caught"; caught=$((caught + 1)); return 0
}

D_A='probe() { for n in /dev/dri/renderD128 /dev/dri/renderD129; do :; done; }
MUT_DRI_MAJOR=128; MUT_DRI_MINOR=129'
D_B='probe() { for n in "$DRI_ROOT/renderD128" "$DRI_ROOT/renderD129"; do :; done; }
MUT_DRI_MAJOR=128; MUT_DRI_MINOR=129'
D_C='probe() { for n in "$DRI_ROOT/renderD128" "$DRI_ROOT/renderD129"; do :; done; }
: # /dev/dri/by-path/pci-0000:86:00.0-render
MUT_DRI_MAJOR=128; MUT_DRI_MINOR=129'

mk_bad() { printf '%s\n' "$1" >"$SB/zz_bad.sh"; }

# D-A: the ORIGINAL defect -- two literal minors.
MUTATE_FN() { mk_bad "$D_A"; }
run_reject "D-A" "two literal minors identify cards by DRM minor" MUTATE_FN

# D-B: the same defect through the INTFER_DRI_ROOT seam. The first version of the
# rule matched the literal "/dev/dri/renderD<minor>" and scored this PASS.
MUTATE_FN() { mk_bad "$D_B"; }
run_reject "D-B" "minors behind the DRI_ROOT seam must not be exempt" MUTATE_FN

# D-C: minors PLUS a by-path mention nearby. The second version excused any file
# mentioning by-path ANYWHERE, which is not the claim "the probe resolves by slot".
MUTATE_FN() { mk_bad "$D_C"; }
run_reject "D-C" "a by-path mention must not excuse a hard-coded minor" MUTATE_FN

# D-D: an EMPTY CENSUS must be refused, not passed. "No script names cards by
# minor" and "no script probes render nodes at all" are different claims.
MUTATE_FN() { rm -rf "$SB"; mkdir -p "$SB"; printf 'echo hello\n' >"$SB/zz_nothing.sh"; }
# *** THIS CALL WAS MISSING IN THE FIRST DRAFT. *** MUTATE_FN was defined and never
# invoked, so D-D ran against D-C's leftover corpus and "caught" the empty-census
# guard for D-C's already-planted defect -- rc=1 was right and the REASON was
# wrong. A state that does not reproduce its own defect is worse than no state
# (audit #17b: verify_patched), and the only reason it surfaced is that D-D
# asserted the guard's MESSAGE rather than just its exit code.
MUTATE_FN || { echo "  D-D MISSED -- mutation fn failed"; exit 1; }
total=$((total + 1))
o="$(run_check)"; r=$?
printf '%s\n' "$o" >"$LOG/D-D.log"
if [ "$r" -ne 0 ] && printf '%s' "$o" | grep -q 'never harvested'; then
  echo "  D-D caught -- rc=$r, an empty corpus is REFUSED, not passed"
  gate_state_record --raw "$SUMMARY" "D-D caught"; caught=$((caught + 1))
else
  echo "  D-D MISSED -- rc=$r; a corpus with no render-node probe was not refused"
  gate_state_record --raw "$SUMMARY" "D-D MISSED"
fi
mk_corpus

# D-E: THE CONVERSE. A single-minor liveness probe must stay clean. Without this
# state a rule that flagged EVERY render-node probe would pass D-A..D-D, and the
# matrix would be proving that a check cannot fail -- which is the whole defect
# class this repo keeps auditing for.
total=$((total + 1))
o="$(run_check)"; r=$?
printf '%s\n' "$o" >"$LOG/D-E.log"
if [ "$r" -eq 0 ] && printf '%s' "$o" | grep -q 'zz_liveness.sh' \
   && ! printf '%s' "$o" | grep -q '^  FAIL'; then
  echo "  D-E caught -- rc=0, and the liveness probe was ACCEPTED (converse holds)"
  gate_state_record --raw "$SUMMARY" "D-E caught"; caught=$((caught + 1))
else
  echo "  D-E MISSED -- rc=$r; a single-minor liveness probe was not accepted cleanly"
  gate_state_record --raw "$SUMMARY" "D-E MISSED"
fi

echo
echo "== summary =="
sed 's/^/  /' "$SUMMARY"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "DRI PROBE MUTATE: PASS ($caught/$total states behaved as specified)"
  { echo "stamp=dri_probe_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"
    echo "converse=D-E single-minor liveness probe accepted"; } >"$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "DRI PROBE MUTATE: FAIL ($caught/$total)"
exit 1