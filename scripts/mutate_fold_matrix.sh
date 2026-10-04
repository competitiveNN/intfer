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
#   S5 writer gate removed          rc=1, contract/index/readback MUST ALL stay
#                                   green. Only the probe's two NEG stride rows
#                                   can see this (AUDIT #13).
#   S6 fold_index_s0 row offset     rc=1, ONLY the index check sees it -- no
#     dropped                        device path in the tree reaches
#                                   fold_index_s0 (u2/G16 only; the readback
#                                   runs bits=4).
#
# S1 and S4 are the load-bearing pair: S1 is invisible to the readback and S4 is
# invisible to the contract check, so neither gate alone is sufficient. S5 and
# S6 are load-bearing for a second reason: each is invisible to every check the
# others can see, so removing any ONE of the four checks leaves a real defect
# undetected.
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
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
LOG_DIR="$BUILD_DIR/fold_matrix"
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
  #  M2 -> fold_readable validation (delegates to the shared decision)
  desc="M2 target: fold_readable 'return fold_stride_ok(fold_stride, nOp) ? usm : nullptr;'"
  n=$(anchor_count "$HDR" 'return fold_stride_ok(fold_stride, nOp) ? usm : nullptr;')
  [ "$n" = "1" ] || { echo "ANCHOR STALE: $desc found $n time(s), want 1" >&2; bad=1; }
  #  M5 -> the WRITER's gate in quant_gemm_dpas48_rowfold
  desc="M5 target: writer gate '!fold_usm || !fold_stride_ok(fold_stride, nOp)'"
  n=$(anchor_count "$SRC" 'if (!fold_usm || !fold_stride_ok(fold_stride, nOp)) return dep;')
  [ "$n" = "1" ] || { echo "ANCHOR STALE: $desc found $n time(s), want 1" >&2; bad=1; }
  #  M6 -> fold_index_s0 row offset
  desc="M6 target: fold_index_s0 'm * (std::size_t)fold_stride + (std::size_t)o'"
  n=$(anchor_count "$HDR" 'return m * (std::size_t)fold_stride + (std::size_t)o;')
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
  #  The writer must route its accept/reject through the shared predicate too
  #  (AUDIT #13). A re-open-coded `fold_stride < 2*nOp` in the writer is the
  #  second copy of the rule, and M5 would then have no target at all.
  desc="no open-coded 'fold_stride < 2 * nOp' gate left in the writer (want 0)"
  n=$(grep -c 'fold_stride < 2 \* nOp' "$SRC" 2>/dev/null || true)
  [ "$n" = "0" ] || {
    echo "ANCHOR STALE: $desc (got $n). The writer must call fold_stride_ok();" >&2
    echo "  a second copy of the rule is what AUDIT #12/#13 exist to prevent." >&2
    bad=1; }

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
  echo "  S5: ONLY the writer's NEG cases see it (contract/index/readback all blind)"
  echo "  S6: ONLY the index check sees it (no device path reaches fold_index_s0)"
  echo "  => no single check covers the class; all four are load-bearing."
}
EXPECTED_SUMMARY="  S1: readback stays GREEN (masked)  -> only the contract check sees it
  S4: contract stays GREEN (blind)   -> index + readback see it
  S5: ONLY the writer's NEG cases see it (contract/index/readback all blind)
  S6: ONLY the index check sees it (no device path reaches fold_index_s0)
  => no single check covers the class; all four are load-bearing."

# Returns 0 if any mutation marker is present in the files given.
has_residue() {
  local pat f
  # -F, so these are the markers LITERALLY as the apply_*() seds write them:
  # '/*M1*/'. Escaping the stars here (as the old `git grep -e '/\*M1\*/'`
  # regex form did) is correct only for a regex; under -F the backslash is a
  # literal character and the pattern can never match anything. The selfcheck
  # caught that within one run of being introduced.
  for pat in '/*M1*/' '/*M2*/' '/*M4*/' '/*M5*/' '/*M6*/'; do
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
  for m in M1 M2 M4 M5 M6; do
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
trap 'restore' EXIT
trap 'restore; exit 130' INT
trap 'restore; exit 143' TERM

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR" || exit 1
: >"$SUMMARY"

# --- mutation appliers -----------------------------------------------------
# EVERY applier verifies its own patch landed, by re-grepping for the marker it
# injects. That is the audit-#17b discipline (a patch that silently does nothing
# is reported as "the mechanism is not load-bearing", a confident false verdict
# about code that was never exercised) -- and here it is marker-based rather than
# the cmp-based verify_patched() of the newer matrices. Both mechanisms are sound;
# this one is older. M1/M2 previously signalled failure only as a bare trailing
# `grep` exit code with no diagnostic, unlike M4/M5/M6, so a stale anchor
# produced an unexplained "M1 did not apply" from the call site.
#
# These are patch_literal() exact replacements, not `sed s|...|...|`. Two concrete
# reasons, both learned:
#   * M5's pattern CONTAINS `||`, so it only worked because the delimiter was
#     escaped to `\|`. That is the audit-#17b delimiter collision class, sitting
#     live in a passing state: a small edit to that line can silently stop the sed
#     from matching, and the state then reports "not load-bearing" -- a verdict
#     about the checker when the fault is in the state.
#   * sed replaces the FIRST match and says nothing about how many there were.
#     M6's anchor is a PREFIX of M4's, so after M4 runs, M6's pattern still matches
#     the M4-mutated line. Nothing caught that; states happen to restore in
#     between. patch_literal REQUIRES exactly one match, so an ambiguous anchor is
#     now a hard stop instead of a silent wrong-site patch.
patch_literal() { # $1 = file, $2 = exact old text, $3 = exact new text
  python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
if n != 1:
    sys.stderr.write("patch_literal: anchor found %d times in %s, want 1: %r\n"
                     % (n, path, old[:80]))
    sys.exit(1)
open(path, "w").write(src.replace(old, new))
PY
}
apply_m1() {
  patch_literal "$HDR" \
    'return (fold_stride == 0) ? 2u * nOp : fold_stride;' \
    'return (fold_stride == 0) ? nOp : fold_stride;  /*M1*/'
  grep -q '/\*M1\*/' "$HDR" || { echo "  M1: shared stride default not mutated (anchor stale?)" >&2; return 1; }
  return 0
}
# M2: fold_readable stops validating the stride.
apply_m2() {
  patch_literal "$HDR" \
    'return fold_stride_ok(fold_stride, nOp) ? usm : nullptr;' \
    'return usm;  /*M2*/'
  grep -q '/\*M2\*/' "$HDR" || { echo "  M2: fold_readable stride validation not removed" >&2; return 1; }
  return 0
}
# M4: the reader's index expression drops the Sfull half. After the index
# refactor this expression has exactly ONE home -- fold_index_sfull in the
# header -- so the mutation goes there and the kernel bodies stay untouched. The
# STRIDE contract stays green (it reasons about stride arithmetic and is
# structurally blind to indexing); the cheap INDEX check turns red, as does the
# device readback. That combination is precisely what S4 demonstrates, and it
# is why the index is now gated without needing a device launch.
apply_m4() {
  patch_literal "$HDR" \
    '  return m * (std::size_t)fold_stride + (std::size_t)nOp + (std::size_t)o;' \
    '  return m * (std::size_t)fold_stride + (std::size_t)o;  /*M4*/'
  grep -q '/\*M4\*/' "$HDR" || { echo "  M4: fold_index_sfull not mutated" >&2; return 1; }
  # The S0 helper must be untouched, so an over-broad sed is caught here.
  if grep -A3 'fold_index_s0(std::size_t m' "$HDR" | grep -q 'M4'; then
    echo "  M4: fold_index_s0 was also hit -- anchor too broad" >&2; return 1
  fi
  return 0
}
# M5: the WRITER stops rejecting a too-small stride. Added in AUDIT #13.
#
# This is the state that proves the writer was a real coverage hole rather than a
# cosmetic duplication. Nothing else sees it: the contract check only calls
# fold_readable (untouched here), the index check does not reason about strides,
# and the readback drives the READER, which still declines correctly. Only the
# probe's two NEG stride cases (stride 100 and 127 against 2*nOp=128) notice --
# so before this state existed, the writer's copy of the rule was verified by
# exactly two hand-picked probe rows and by nothing else.
apply_m5() {
  # The literal below contains `||` -- the reason this used to be an escaped
  # `sed s|...|...|` and why that was fragile.
  patch_literal "$SRC" \
    'if (!fold_usm || !fold_stride_ok(fold_stride, nOp)) return dep;' \
    'if (!fold_usm || false) return dep;  /*M5*/'
  grep -q '/\*M5\*/' "$SRC" || { echo "  M5: writer gate not mutated" >&2; return 1; }
  return 0
}
# M6: fold_index_s0 loses its row offset, so every row reads row 0's S0 half.
#
# Note what this state does NOT do: it leaves the device readback GREEN. The
# only reader path that calls fold_index_s0 is the u2/G16 split branch, and the
# probe's readback runs bits=4, i.e. the u4 path, which reads only
# fold_index_sfull. So fold_index_s0 has NO device coverage anywhere in the
# tree and the host-side index check is its ONLY gate -- which is the strongest
# argument for having added that check.
apply_m6() {
  patch_literal "$HDR" \
    '  return m * (std::size_t)fold_stride + (std::size_t)o;' \
    '  return (std::size_t)o;  /*M6*/'
  grep -q '/\*M6\*/' "$HDR" || { echo "  M6: fold_index_s0 not mutated" >&2; return 1; }
  if grep -A3 'fold_index_sfull(std::size_t m' "$HDR" | grep -q 'M6'; then
    echo "  M6: fold_index_sfull was also hit -- anchor too broad" >&2; return 1
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

restore; apply_m5 || { echo "FAIL: M5 did not apply" >&2; exit 1; }
run_state S5_writer_gate     1 none clean  clean

restore; apply_m6 || { echo "FAIL: M6 did not apply" >&2; exit 1; }
run_state S6_index_s0        1 none clean  wrong

# --- restore the ARTIFACT too, and prove the tree is GREEN, not just clean ----
# AUDIT #15. `git checkout --` restores the SOURCE and nothing else. The last
# state run is S6, so the dpas_fold_probe binary left in the build tree is the
# MUTATED one -- and any later gate that reuses the build without rebuilding
# (INTFER_GATE_NO_BUILD=1) judges mutated code. Observed live and caught by the
# gate rather than by the harness: test_dpas_fold.sh reported rc=1 with the S6
# signature (360 index tuples WRONG, m=4095 reading S0=224) while
# `git status` was clean and a one-line rebuild made it PASS. This is family 3
# (stale artifact) committed BY the harness that exists to prove the checks work.
#
# So the matrix's last act is now: restore the source, REBUILD the probe through
# the shared build helper, and run the gate with the build disabled. A green
# result is the real proof that the tree was left in the state the matrix claims
# -- and it also catches an artifact poisoned by an EARLIER run, which a
# byte-identical-restore check cannot see at all.
restore
trap - EXIT INT TERM

echo
if git diff --quiet -- "$HDR" "$SRC"; then
  echo "restore: $HDR and $SRC are byte-identical to HEAD"
else
  echo "FAIL: restore left the tree dirty" >&2; FAILED=$((FAILED + 1))
fi

echo "rebuild: dpas_fold_probe from the restored source (not the mutated one)"
. "$SCRIPT_DIR/_gate_build.sh"
if gate_require_build dpas_fold_probe; then
  if INTFER_GATE_NO_BUILD=1 bash scripts/test_dpas_fold.sh >"$LOG_DIR/post_restore_green.log" 2>&1; then
    echo "post-restore: test_dpas_fold.sh rc=0 on the LEFT-BEHIND artifact (tree is green)"
  else
    echo "FAIL: the tree is clean but the gate is RED on the left-behind artifact --" >&2
    echo "  the build tree is poisoned (by this run or an earlier one). Any gate" >&2
    echo "  reusing it without rebuilding would judge the wrong code." >&2
    FAILED=$((FAILED + 1))
  fi
else
  echo "FAIL: could not rebuild dpas_fold_probe after restore" >&2; FAILED=$((FAILED + 1))
fi
if has_residue "$HDR" "$SRC"; then
  echo "FAIL: mutation residue left in tracked sources" >&2; FAILED=$((FAILED + 1))
else
  echo "residue: none (no M1/M2/M4/M5/M6 markers in tracked sources)"
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
    echo "states=S0,S1,S2,S3,S4,S5,S6"
  } >"$STAMP" 2>/dev/null
  echo "FOLD MUTATION MATRIX: PASS (7/7 states behaved as characterized)"
  emit_summary
  echo "  stamped $(git rev-parse --short HEAD 2>/dev/null) -> $STAMP"
  exit 0
fi
echo "FOLD MUTATION MATRIX: FAIL ($FAILED state(s) not as characterized)" >&2
echo "  Full per-state output: $SUMMARY" >&2
exit 1
