#!/usr/bin/env bash
# Reproducible non-vacuity proof for the DPAS row-fold gate (test_dpas_fold.sh).
#
# WHY THIS EXISTS. The S0-S4 mutation matrix behind AUDIT #12 was originally
# produced by hand-rolled inline sed/python during a session, so the tree
# inherited only a PROSE TABLE. Prose is not evidence: an earlier draft of that
# table asserted one row (M2 "masked") that had actually been measured with
# another mutation still applied, and the row was simply wrong. This script
# makes the matrix re-runnable so the next session verifies instead of trusting.
#
# WHAT IT ASSERTS, per state -- and this is the part that matters. It does not
# just dump output; it asserts the CHARACTER of each state, because the whole
# point of the matrix is that the two checks are NOT redundant:
#
#   S0 clean                        rc=0
#   S1 default 2*nOp -> nOp          rc=1, readback MUST stay green (masked).
#                                   Only the contract check can see this one.
#   S2 fold_readable unchecked      rc=1, readback MUST fail (30/64). The
#                                   deliberately-too-small stride=1 regime is
#                                   what exposes it -- a stride the engine
#                                   would never send, which is why no
#                                   end-to-end arm ever could.
#   S3 S1 + S2 together             rc=1, readback MUST fail.
#   S4 reader index +nOp+o -> +o    rc=1, contract MUST stay green (it is
#                                   structurally blind to kernel-side indexing)
#                                   while readback MUST fail at 60/64.
#
# S1 and S4 are the load-bearing pair: S1 is invisible to the readback and S4 is
# invisible to the contract check, so neither gate alone is sufficient.
#
# SAFETY. This script EDITS TRACKED FILES (the fold header and quant_gemm.cpp)
# and restores them with `git checkout --`. That is destructive to uncommitted
# work, so it REFUSES TO START unless the tree is clean, and traps EXIT to
# restore. It also re-verifies at the end that the tree is byte-identical to
# HEAD and free of mutation residue.
#
# Usage:  scripts/mutate_fold_matrix.sh
# Env:    INTFER_GATE_NO_BUILD=1  reuse the current build (faster; same checks)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

HDR=include/intfer/gemm/quant_gemm.hpp
SRC=src/gemm/quant_gemm.cpp
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/fold_matrix"
SUMMARY="$LOG_DIR/summary.txt"
STAMP="$LOG_DIR/last_pass.stamp"
ANCHORS_ONLY=0
[ "${1:-}" = "--anchors-only" ] && ANCHORS_ONLY=1

# --- ANCHOR PRE-FLIGHT -----------------------------------------------------
# A mutation harness has its own artifact assumptions, and a refactor silently
# invalidates them. That is not hypothetical: the index refactor moved the
# reader's index expression into fold_index_sfull, which deleted the exact text
# apply_m4 used to corrupt. The matrix then failed closed with "M4 did not
# apply" -- but only because of the order in which it happened to be run; had
# the anchor merely become AMBIGUOUS rather than absent, a state could have been
# quietly mis-targeted and still "pass".
#
# So every anchor is asserted to occur EXACTLY ONCE before any state runs. One
# hit = a refactor moved the code; zero hits = the code is gone; two or more =
# the sed is too broad and would corrupt a site it was not meant to. All three
# are a hard stop with an explicit message, because in every case the matrix
# would otherwise be testing something other than what it claims.
anchor_count() { grep -cF -- "$2" "$1" 2>/dev/null || true; }

preflight_anchors() {
  local bad=0 n desc
  #  M1 -> fold_row_stride default
  desc="M1 target: fold_row_stride default '2u * nOp'"
  n=$(anchor_count "$HDR" 'return (fold_stride == 0) ? 2u * nOp : fold_stride;')
  [ "$n" = "1" ] || { echo "ANCHOR STALE: $desc found $n time(s), want 1" >&2; bad=1; }
  #  M2 -> fold_readable validation
  desc="M2 target: fold_readable validation '>= 2u * nOp'"
  n=$(anchor_count "$HDR" 'return fold_row_stride(fold_stride, nOp) >= 2u * nOp ? usm : nullptr;')
  [ "$n" = "1" ] || { echo "ANCHOR STALE: $desc found $n time(s), want 1" >&2; bad=1; }
  #  M4 -> fold_index_sfull Sfull offset
  desc="M4 target: fold_index_sfull '+(std::size_t)nOp'"
  n=$(anchor_count "$HDR" 'return m * (std::size_t)fold_stride + (std::size_t)nOp + (std::size_t)o;')
  [ "$n" = "1" ] || { echo "ANCHOR STALE: $desc found $n time(s), want 1" >&2; bad=1; }
  #  Structural invariants the matrix depends on.
  desc="call sites routed through the index helpers (want 5)"
  n=$(grep -c 'fold_index_sfull\|fold_index_s0' "$SRC" 2>/dev/null || true)
  [ "$n" = "5" ] || {
    echo "ANCHOR STALE: $desc (got $n). A read site bypasses the index helpers," >&2
    echo "  so M4 would no longer reach every reader." >&2; bad=1; }
  desc="no raw 'Fold[(size_t)m * fold_stride' index expressions left (want 0)"
  n=$(grep -c 'Fold\[(size_t)m \* fold_stride' "$SRC" 2>/dev/null || true)
  [ "$n" = "0" ] || {
    echo "ANCHOR STALE: $desc (got $n). Re-route them through" >&2
    echo "  fold_index_sfull/fold_index_s0 or M4 becomes partial." >&2; bad=1; }

  if [ "$bad" != "0" ]; then
    cat >&2 <<EOF

  The matrix is testing code that no longer exists in the shape it expects.
  Re-point the affected apply_*() at the new location rather than loosening the
  check -- a loosened anchor is how a mutation silently stops mutating.
EOF
    return 1
  fi
  echo "anchors: all M1/M2/M4 targets and structural invariants OK"
  return 0
}

# --- SELF-CHECK: the strings this script's own PASS line depends on ---------
# Edits to this file were silently lost twice in the session that produced it
# (same failure mode AGENTS.md already records for the bulk edit tool), and both
# losses were in exactly these strings: the closing summary and the residue
# scan. A gate that reports a conclusion it can no longer print is a gate nobody
# can trust, so assert them before doing any work.
#
# BEHAVIOURAL, NOT TEXTUAL. The first version of this check grepped its own
# source for the summary strings -- and was therefore VACUOUS: the pattern sits
# inside the check, so the check could never fail. It reported OK with the S1
# line deleted, which is the very failure class this session has been about,
# committed one level down. The check below compares what the script RENDERS
# against a literal, and proves the residue scan DETECTS each marker by seeding
# a temp file. Both can actually fail.
emit_summary() {
  echo "  S1: readback stays GREEN (masked)  -> only the contract check sees it"
  echo "  S4: contract stays GREEN (blind)   -> index + readback see it"
  echo "  => no single check covers the class; all three are load-bearing."
}
EXPECTED_SUMMARY="  S1: readback stays GREEN (masked)  -> only the contract check sees it
  S4: contract stays GREEN (blind)   -> index + readback see it
  => no single check covers the class; all three are load-bearing."

# Returns 0 if any mutation marker is present in the files given.
has_residue() {
  local pat f
  # -F, so these are the markers LITERALLY as the apply_*() seds write them:
  # '/*M1*/'. Escaping the stars here (as the old `git grep -e '/\*M1\*/'`
  # regex form did) is correct only for a regex; under -F the backslash is a
  # literal character and the pattern can never match anything. The selfcheck
  # caught that within one run of being introduced.
  for pat in '/*M1*/' '/*M2*/' '/*M4*/'; do
    for f in "$@"; do
      grep -qF -- "$pat" "$f" 2>/dev/null && return 0
    done
  done
  return 1
}

selfcheck() {
  local bad=0 tmp m
  if [ "$(emit_summary)" != "$EXPECTED_SUMMARY" ]; then
    echo "SELFCHECK: the PASS summary no longer renders what it promises" >&2
    bad=1
  fi
  tmp="$(mktemp "${TMPDIR:-/tmp}/fold_matrix_selfcheck.XXXXXX")"
  for m in M1 M2 M4; do
    printf 'int x; /*%s*/\n' "$m" >"$tmp"
    has_residue "$tmp" || { echo "SELFCHECK: residue scan misses a /*$m*/ marker" >&2; bad=1; }
  done
  printf 'int clean;\n' >"$tmp"
  has_residue "$tmp" && { echo "SELFCHECK: residue scan false-positives on clean input" >&2; bad=1; }
  rm -f "$tmp"
  return $bad
}

if [ "$ANCHORS_ONLY" = "1" ]; then
  # Fast path: no build, no mutation, <1s. Use to check the harness still matches
  # the tree after a refactor, e.g. before a long gate run.
  preflight_anchors || exit 1
  selfcheck || exit 1
  echo "FOLD MATRIX ANCHORS: OK"
  exit 0
fi

# --- fail closed on a dirty tree: we are about to git checkout -- tracked files
if ! git diff --quiet -- "$HDR" "$SRC" || ! git diff --cached --quiet -- "$HDR" "$SRC"; then
  echo "FAIL: $HDR or $SRC has uncommitted changes." >&2
  echo "  This script edits tracked files and restores them with 'git checkout --'," >&2
  echo "  which would DESTROY that work. Commit or stash first, then re-run." >&2
  exit 1
fi
if [ -n "$(git ls-files --others --exclude-standard -- "$HDR" "$SRC")" ]; then
  echo "FAIL: untracked files present at $HDR / $SRC" >&2
  exit 1
fi

selfcheck || { echo "FAIL: this script's own invariants are broken" >&2; exit 1; }
preflight_anchors || exit 1

restore() {
  git checkout -- "$HDR" "$SRC" 2>/dev/null
  # Never leave a mutation behind, even on Ctrl-C.
  if ! git diff --quiet -- "$HDR" "$SRC"; then
    echo "RESTORING: $HDR/$SRC were not clean; discarding mutation residue" >&2
    git checkout -- "$HDR" "$SRC"
  fi
}
trap restore EXIT INT TERM

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR" || exit 1
: >"$SUMMARY"

# --- mutation appliers -----------------------------------------------------
# M1: the reader/writer shared rule resolves the 0 default to nOp, not 2*nOp.
apply_m1() {
  sed -i 's|return (fold_stride == 0) ? 2u \* nOp : fold_stride;|return (fold_stride == 0) ? nOp : fold_stride;  /*M1*/|' "$HDR"
  grep -q '/\*M1\*/' "$HDR"
}
# M2: fold_readable stops validating the stride.
apply_m2() {
  sed -i 's|return fold_row_stride(fold_stride, nOp) >= 2u \* nOp ? usm : nullptr;|return usm;  /*M2*/|' "$HDR"
  grep -q '/\*M2\*/' "$HDR"
}
# M4: the reader's index expression drops the Sfull half. After the index
# refactor this expression has exactly ONE home -- fold_index_sfull in the
# header -- so the mutation goes there and the kernel bodies stay untouched. The
# STRIDE contract stays green (it reasons about stride arithmetic and is
# structurally blind to indexing); the cheap INDEX check turns red, as does the
# device readback. That combination is precisely what S4 demonstrates, and it
# is why the index is now gated without needing a device launch.
apply_m4() {
  sed -i 's|  return m \* (std::size_t)fold_stride + (std::size_t)nOp + (std::size_t)o;|  return m * (std::size_t)fold_stride + (std::size_t)o;  /*M4*/|' "$HDR"
  grep -q '/\*M4\*/' "$HDR" || { echo "  M4: fold_index_sfull not mutated" >&2; return 1; }
  # The S0 helper must be untouched, so an over-broad sed is caught here.
  if grep -A3 'fold_index_s0(std::size_t m' "$HDR" | grep -q 'M4'; then
    echo "  M4: fold_index_s0 was also hit -- anchor too broad" >&2; return 1
  fi
  return 0
}

# --- runner ---------------------------------------------------------------
# $1 = state label, $2 = expected rc, $3 = expected readback FAIL count class
#       ("none" | "some"), $4 = expected contract status ("clean" | "violated")
run_state() {
  local label="$1" want_rc="$2" want_rb="$3" want_ct="$4" want_ix="${5:-clean}"
  local log="$LOG_DIR/$label.log"
  bash scripts/test_dpas_fold.sh >"$log" 2>&1
  local rc=$?

  local rb ct ix verdict=""
  rb=$(grep -cE '^  readback .*FAIL' "$log")
  if grep -qE '^  contract: [0-9]+ \(nOp,stride\) pairs.* -- NO' "$log"; then
    ct=violated
  else
    ct=clean
  fi
  if grep -qE '^  index: +[0-9]+ .*WRONG' "$log"; then ix=wrong; else ix=clean; fi

  [ "$rc" = "$want_rc" ] || verdict="$verdict rc(want=$want_rc got=$rc)"
  case "$want_rb" in
    none) [ "$rb" = "0" ] || verdict="$verdict readback_expected_clean_got_${rb}_failures" ;;
    some) [ "$rb" -ge 1 ] || verdict="$verdict readback_expected_failure_got_none" ;;
  esac
  [ "$ct" = "$want_ct" ] || verdict="$verdict contract(want=$want_ct got=$ct)"
  [ "$ix" = "$want_ix" ] || verdict="$verdict index(want=$want_ix got=$ix)"

  if [ -z "$verdict" ]; then
    printf '  %-26s rc=%s  readback_fail=%-2s contract=%-8s index=%-6s  AS EXPECTED\n' \
      "$label" "$rc" "$rb" "$ct" "$ix" | tee -a "$SUMMARY"
  else
    printf '  %-26s rc=%s  readback_fail=%-2s contract=%-8s index=%-6s  UNEXPECTED:%s\n' \
      "$label" "$rc" "$rb" "$ct" "$ix" "$verdict" | tee -a "$SUMMARY"
    FAILED=$((FAILED + 1))
  fi
  grep -E '^  contract: [0-9]+ \(nOp,stride\)|^  index: |^  readback ' "$log" \
    | sed 's/^/       /' >>"$SUMMARY"
}

echo "=== DPAS fold mutation matrix (non-vacuity proof) ==="
echo "worktree must be clean; logs -> $LOG_DIR"
FAILED=0

restore
run_state S0_clean            0 none clean

restore; apply_m1 || { echo "FAIL: M1 did not apply" >&2; exit 1; }
run_state S1_M1_default_only  1 none violated

restore; apply_m2 || { echo "FAIL: M2 did not apply" >&2; exit 1; }
run_state S2_M2_noval_only    1 some violated

restore; apply_m1 && apply_m2 || { echo "FAIL: M1+M2 did not apply" >&2; exit 1; }
run_state S3_M1_and_M2        1 some violated

restore; apply_m4 || { echo "FAIL: M4 did not apply" >&2; exit 1; }
run_state S4_reader_index     1 some clean wrong

# --- restore and prove the tree survived -------------------------------
restore
trap - EXIT INT TERM

echo
if git diff --quiet -- "$HDR" "$SRC"; then
  echo "restore: $HDR and $SRC are byte-identical to HEAD"
else
  echo "FAIL: restore left the tree dirty" >&2; FAILED=$((FAILED + 1))
fi
if has_residue "$HDR" "$SRC"; then
  echo "FAIL: mutation residue left in tracked sources" >&2; FAILED=$((FAILED + 1))
else
  echo "residue: none (no M1/M2/M4 markers in tracked sources)"
fi

echo
if [ "$FAILED" = "0" ]; then
  # Stamp the tree this proof was actually established against, so a later
  # session can tell "the matrix passed" from "the matrix passed on THIS code".
  # Without it, a refactor can move the anchors and the only signal is a
  # failure buried in someone's gate log.
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=S0,S1,S2,S3,S4"
  } >"$STAMP" 2>/dev/null
  echo "FOLD MUTATION MATRIX: PASS (5/5 states behaved as characterized)"
  emit_summary
  echo "  stamped $(git rev-parse --short HEAD 2>/dev/null) -> $STAMP"
  exit 0
fi
echo "FOLD MUTATION MATRIX: FAIL ($FAILED state(s) not as characterized)" >&2
echo "  Full per-state output: $SUMMARY" >&2
exit 1
