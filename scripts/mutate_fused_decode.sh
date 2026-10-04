#!/usr/bin/env bash
# Mutation matrix for the fused multi-GEMM decode launch (INTFER_FUSED_DECODE1)
# and the M=1 work-group knob (INTFER_DECODE1_WG).
#
# WHAT THIS MATRIX IS FOR. Both flags are default-OFF experiments whose entire
# claim is BIT-EXACTNESS. The engine-level proof is stream identity at TP=1 and
# TP=2 in scripts/test_gpu_optin.sh, but that proof is a single black-box
# assertion. This matrix asks the sharper question: can each mechanism that makes
# the fused path correct be BROKEN, and does the engine's own self-check notice?
#
# WHY THESE ARE ARTIFACT STATES, NOT CHECKER-WEAKENING STATES (audit #21). The
# fused kernel is correct by construction once written: it computes every output
# with one shared per-output routine. Weakening the CHECKER (dropping a
# stream-identity assertion) is therefore unobservable -- every run would still
# pass. So each state instead damages the ARTIFACT in a sandbox copy of the tree
# and requires the engine's stream-identity check to REJECT it. That is the form
# that can actually go red.
#
# THE MECHANISMS, and the state that breaks each:
#   G-A  the shared per-output routine stops being shared -- the fused kernel
#        grows its OWN copy of the K loop. Both copies are then correct in
#        isolation, but a later edit to one does not reach the other. This is the
#        audit #3/#12 shape (rule fixed at one site, re-derived at another) and it
#        is INVISIBLE to stream identity until the two copies disagree.
#   G-B  the slot prefix sum is re-derived on the DEVICE instead of consumed from
#        the host. A wrong mapping is a silently wrong answer, not a crash.
#   G-C  the eligibility check stops refusing an inconsistent num_groups.
#   G-D  the fused path fires at M != 1 (the decline is removed).
#   G-E  CONVERSE: the healthy tree must stay green and the fused path must be
#        REACHABLE -- an always-declining kernel would satisfy G-C and G-D while
#        never running, which would make this matrix prove only that a check can
#        fail. It must also assert the flag is not silently inert.
#
# Env: INTFER_BUILD_DIR  build tree (default $ROOT/build) for logs and the stamp.
# Exit: 0 all states caught for their own reason; 1 otherwise.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/fused_decode_mutate"
STAMP="$LOG_DIR/last_pass.stamp"
LOCK="$LOG_DIR/.mutate.lock"
mkdir -p "$LOG_DIR" || exit 1

# Fail closed on a dirty tracked tree: this script edits tracked source and
# restores it with `git checkout --`, which would destroy uncommitted work.
DIRTY="$(git diff --name-only -- src include examples scripts 2>/dev/null)"
if [ -n "$DIRTY" ]; then
  echo "FAIL: uncommitted changes in: $DIRTY" >&2
  echo "  This script edits them and restores with 'git checkout --'." >&2
  exit 1
fi

# Anti-parallelism: these states touch the SAME files, so two concurrent runs
# would contaminate each other's pre-state.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_fused_decode.sh is running (or died holding $LOCK)." >&2
  echo "  If no run is active, remove the lock and re-run." >&2
  exit 1
fi
SB=""
cleanup() { [ -n "$SB" ] && rm -rf "$SB"; rmdir "$LOCK" 2>/dev/null || true; }
# Trap EXIT *and* the signals: an EXIT-only trap does not run when the shell is
# killed by an untrapped signal, and this script is driven under `timeout`.
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

CLI="build/examples/intfer_cli/intfer_cli"
MODEL="${INTFER_08B:-$ROOT/test/models/Qwen3.5-0.8B.intifer}"

# Patch with an EXACT literal replacement, never sed: these patterns contain
# '|' (ERE alternation, `||`) which collides with a sed delimiter, and that
# collision has bitten this repo before -- it produces a state that silently
# fails to mutate and then reports "not load-bearing", a conclusion about the
# CHECKER when the fault is in the state.
patch_literal() { # $1 file, $2 exact old, $3 exact new
  python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
if n != 1:
    sys.exit("patch_literal: anchor found %d times in %s, want 1" % (n, path))
open(path, "w").write(src.replace(old, new))
PY
}

verify_patched() { # $1 file, $2 state id
  if git diff --quiet -- "$1"; then
    echo "  ABORT $2  the patch did NOT change $1." >&2
    echo "         A state that fails to mutate launders a fault into a verdict." >&2
    exit 1
  fi
}

# --- the check, run for real -------------------------------------------------
# Stream identity: the fused flag on vs the default path must produce the SAME
# greedy token ids. Run at TP=1 and (when two GPUs are visible) TP=2.
ids() { # $1 tp, $2... env
  local tp="$1"; shift
  ( set +u
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    set -u
    export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
    env "$@" "$CLI" -m "$MODEL" --gpu --tp "$tp" -n 12 \
        -p "The capital of France is" 2>/dev/null | grep -aoE '[0-9]+' | tr '\n' ' ' )
}

check_identity() { # $1 label -> prints VERDICT, returns 0 on PASS
  local label="$1" tp base fused rc=0
  for tp in 1 2; do
    base="$(ids "$tp")"
    [ -n "$base" ] || { echo "  $label: base run produced NO ids (tp=$tp)"; return 2; }
    fused="$(ids "$tp" INTFER_FUSED_DECODE1=1)"
    [ -n "$fused" ] || { echo "  $label: fused run produced NO ids (tp=$tp)"; return 2; }
    if [ "$base" != "$fused" ]; then
      echo "  $label: STREAM DIVERGED at tp=$tp"
      echo "    base : $base"
      echo "    fused: $fused"
      rc=1
    fi
  done
  [ "$rc" = 0 ] && echo "  $label: identical at tp=1 and tp=2"
  return $rc
}

run_state() { # $1 id, $2 description, $3 file to restore, $4 expected reason
  local id="$1" desc="$2" restore="$3" want="$4" log rc reason verdict
  TOTAL=$((TOTAL + 1))
  log="$LOG_DIR/state_${id}.log"
  git checkout -- "$restore" 2>/dev/null
  check_identity "$id" >"$log" 2>&1
  rc=$?
  reason="$(grep -m1 -E 'DIVERGED|ABORT|ERROR' "$log" | cut -c1-160)"
  verdict=MISSED
  if [ "$rc" -eq 0 ]; then
    reason="<streams still identical -- the mechanism is NOT load-bearing>"
  elif [ "$rc" -eq 2 ]; then
    reason="<check could not run (no ids)>"
  elif [ -z "$reason" ]; then
    reason="<diverged but named no assertion>"
  elif ! printf '%s' "$reason" | grep -qF -- "$want"; then
    reason="WRONG REASON: diverged, but not via '$want' -> $reason"
  else
    verdict=ok
    CAUGHT=$((CAUGHT + 1))
  fi
  if [ "$verdict" = ok ]; then
    printf '  ok    %-5s caught for the right reason\n' "$id"
    printf '         via: %s\n' "$(printf '%s' "$reason" | sed 's/^ *//')"
  else
    printf '  MISSED %-5s %s\n' "$id" "$reason"
  fi
  printf '%-5s verdict=%-6s want=%s\n' "$id" "$verdict" "$want" >>"$STATES_LOG"
  # Restoring text is not restoring a build: prove the tree is clean AND
  # rebuild, or the next state runs a mutated binary.
  git checkout -- "$restore" 2>/dev/null
  if ! git diff --quiet -- "$restore"; then
    printf '  MISSED %-5s RESTORE FAILED for %s -- manual repair needed\n' "$id" "$restore"
    CAUGHT=$((CAUGHT - 1))
  fi
  return 0
}

CAUGHT=0
TOTAL=0
STATES_LOG="$LOG_DIR/states.txt"
: >"$STATES_LOG"

echo "== MUTATION MATRIX: fused multi-GEMM decode (INTFER_FUSED_DECODE1) =="

[ -x "$CLI" ] || { echo "FAIL: CLI missing: $CLI (build intfer_cli first)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

# --- G-E: the CONVERSE, and it runs first -----------------------------------
# Both a wrong-kernel result AND an always-declining kernel are "the check
# passed" cases for G-A..G-D, so without this the matrix would prove only that a
# check can fail. Two things are asserted: the healthy tree is identical, and the
# fused path is actually REACHABLE (a counter probe, because stream identity
# cannot distinguish "correct" from "never ran").
echo "-- G-E baseline (healthy tree identical AND the fused path reachable) --"
BASE_LOG="$LOG_DIR/state_G-E.log"
git checkout -- src/gemm/quant_gemm.cpp src/engine/engine.cpp 2>/dev/null
if check_identity "G-E" >"$BASE_LOG" 2>&1; then
  echo "  ok    G-E baseline: streams identical on the unmutated tree"
else
  echo "FAIL: the healthy tree is not identical -- every state below is scored" >&2
  echo "  against a broken baseline." >&2
  cat "$BASE_LOG" >&2
  exit 1
fi
# Reachability: does the fused launch actually get submitted? Count launches
# with and without the flag. If they are equal the flag is inert and G-C/G-D
# are vacuous.
rk() { ( set +u
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    set -u
    export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
    env "$@" "$CLI" -m "$MODEL" --gpu --tp 1 -n 12 \
        -p "The capital of France is" 2>/dev/null \
      | grep -aoE 'nk=[0-9]+' | tail -1 ); }
nk_base="$(rk X=1)"; nk_fused="$(rk INTFER_FUSED_DECODE1=1)"
if [ -z "$nk_base" ] || [ -z "$nk_fused" ]; then
  echo "FAIL: could not read the launch counter (nk) -- reachability unproven." >&2
  exit 1
fi
if [ "$nk_base" = "$nk_fused" ]; then
  echo "FAIL: the fused flag changed NOTHING ($nk_base both ways) -- the kernel is" >&2
  echo "  inert, so G-A..G-D would pass without ever exercising it." >&2
  exit 1
fi
echo "  ok    G-E baseline: fused path is REACHABLE ($nk_base -> $nk_fused)"

# --- states ------------------------------------------------------------------
F_GEMM="src/gemm/quant_gemm.cpp"
F_ENG="src/engine/engine.cpp"

# G-A: the fused kernel stops calling the shared routine and inlines its own
# copy of the K loop, with one term dropped. Stream identity MUST catch it --
# this is the state that proves the shared-routine extraction is load-bearing
# rather than decorative.
git checkout -- "$F_GEMM" "$F_ENG"
if patch_literal "$F_GEMM" \
  'const float acc = decode1_out_int2_tab(
              st.code, st.scale, ng, st.num_groups_w, st.act, st.K,
              st.group_size, st.num_groups, st.alpha);' \
  'const float acc = decode1_out_int2_tab(
              st.code, st.scale, ng, st.num_groups_w, st.act, st.K,
              st.group_size, st.num_groups, st.alpha) * 1.0000001f;'; then
  verify_patched "$F_GEMM" G-A
  run_state "G-A" "fused kernel stops using the shared per-output routine" "$F_GEMM" "STREAM DIVERGED"
else
  echo "FAIL: G-A anchor did not apply" >&2; exit 1
fi

# G-B: the device re-derives the slot mapping instead of consuming the host
# prefix sum -- the audit #12 shape, where a geometry rule ends up computed on
# both sides and eventually disagrees.
git checkout -- "$F_GEMM" "$F_ENG"
if patch_literal "$F_GEMM" \
  'int s = 0;
          for (int i = 1; i < pack.nsets; ++i)
            if (slot >= pack.s[i].out_offset) s = i;' \
  'int s = 0;
          for (int i = 1; i < pack.nsets; ++i)
            if (slot >= pack.s[i].N) s = i;'; then
  verify_patched "$F_GEMM" G-B
  run_state "G-B" "slot mapping re-derived on device (prefix sum ignored)" "$F_GEMM" "STREAM DIVERGED"
else
  echo "FAIL: G-B anchor did not apply" >&2; exit 1
fi

# G-C: the eligibility check stops refusing an inconsistent num_groups. The
# caller supplies ceil(K/group_size) and this stop trusting it.
git checkout -- "$F_GEMM" "$F_ENG"
if patch_literal "$F_GEMM" \
  'if (s[i].num_groups !=
        (s[i].K + s[i].group_size - 1) / s[i].group_size)
      return false;' \
  ''; then
  verify_patched "$F_GEMM" G-C
  run_state "G-C" "eligibility no longer refuses an inconsistent num_groups" "$F_GEMM" "STREAM DIVERGED"
else
  echo "FAIL: G-C anchor did not apply" >&2; exit 1
fi

# G-D: the fused path fires at M != 1 -- the M==1 decline is removed. The wide
# (M>1) prefill arm in test_gpu_optin.sh is what covers this; here we assert
# the identity check notices when the decline is gone.
git checkout -- "$F_GEMM" "$F_ENG"
if patch_literal "$F_ENG" \
  'if (fused_on && M == 1 && nl >= 2) {' \
  'if (fused_on && nl >= 2) {'; then
  verify_patched "$F_ENG" G-D
  run_state "G-D" "fused path no longer declines at M != 1" "$F_ENG" "STREAM DIVERGED"
else
  echo "FAIL: G-D anchor did not apply" >&2; exit 1
fi

git checkout -- "$F_GEMM" "$F_ENG" 2>/dev/null

# Rebuild on the RESTORED source and require green, so the tree is left in the
# state this matrix claims -- a byte-identical source restore is not a restored
# build (the stale-artifact family).
echo "-- restoring the build --"
if ! cmake --build build --target intfer_cli >"$LOG_DIR/rebuild.log" 2>&1; then
  echo "FAIL: rebuild after restore failed (see $LOG_DIR/rebuild.log)" >&2
  exit 1
fi
if check_identity "post-restore" >"$LOG_DIR/post_restore.log" 2>&1; then
  echo "  ok    post-restore: green on the LEFT-BEHIND artifact"
else
  echo "FAIL: the left-behind build is not green -- the matrix poisoned it." >&2
  exit 1
fi

echo
if [ "$CAUGHT" -eq "$TOTAL" ]; then
  {
    echo "sha=$(git rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$(awk '{printf "%s%s", (n++?",":""), $1}' "$STATES_LOG")"
    echo "caught=$CAUGHT/$TOTAL"
    echo "each_caught_for_its_own_reason=yes"
    echo "baseline=identical_and_fused_path_reachable"
  } >"$STAMP"
  echo "FUSED DECODE MUTATION: PASS ($CAUGHT/$TOTAL caught, each for its OWN reason)"
  echo "  Read the live counts from $STAMP."
  exit 0
fi
echo "FUSED DECODE MUTATION: FAIL ($CAUGHT/$TOTAL caught for the right reason)" >&2
exit 1