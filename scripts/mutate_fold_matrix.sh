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
if git grep -qI -e '/\*M1\*/' -e '/\*M2\*/' -e '/\*M4\*/' -- "$HDR" "$SRC" 2>/dev/null; then
  echo "FAIL: mutation residue left in tracked sources" >&2; FAILED=$((FAILED + 1))
else
  echo "residue: none (no M1/M2/M4 markers in tracked sources)"
fi

echo
if [ "$FAILED" = "0" ]; then
  echo "FOLD MUTATION MATRIX: PASS (5/5 states behaved as characterized)"
  echo "  S1 invisible to readback / S4 invisible to contract => both checks required."
  exit 0
fi
echo "FOLD MUTATION MATRIX: FAIL ($FAILED state(s) not as characterized)" >&2
echo "  Full per-state output: $SUMMARY" >&2
exit 1
