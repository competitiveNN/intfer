#!/usr/bin/env bash
# Mutation matrix for scripts/check_e2_ctx.sh -- the last checker in the tree
# with no matrix (enumerated from `ls scripts/check_*.sh`, not from memory:
# audit #7/#8 says a remembered list only rediscovers what the last fix knew).
#
# SCOPE, STATED PLAINLY BECAUSE IT LIMITS WHAT THIS PROVES.
# check_e2_ctx.sh has two halves and they are NOT equally testable:
#   * the SELF-CHECK (synthetic _verdict cases) -- fully testable here, because
#     its inputs are synthetic and a defect in the predicate is observable.
#   * the REAL-RUN assertions (prompt line, decode-step count, id equality
#     against the actual 0.8B model) -- NOT testable by weakening, because the
#     artifact they read is CORRECT by construction. Weakening a checker against
#     a correct artifact is the unobservable no-op described in AUDIT #21, and
#     scoring it as "caught" would be a lie. Testing those needs a deliberately
#     broken model, which is out of scope here.
# So this matrix proves the SELF-CHECK is load-bearing, and says so.
#
# Predicates use scripts/_gate_pred.sh with the polarity named at the call site
# (AUDIT #20's inverted-predicate rule). Every state routes through
# verify_patched(). Fixtures are a sandbox OUTSIDE $BUILD_DIR.
#
# Usage: scripts/mutate_e2_ctx.sh
# Exit: 0 = every state caught for its own reason, 1 = otherwise, 2 = setup.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_patch.sh
. "$SCRIPT_DIR/_gate_patch.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"

CHECK="$SCRIPT_DIR/check_e2_ctx.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/e2_ctx_mutate"
LOG="$STAMP/states"
SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_me2c.XXXXXX")"   # OUTSIDE $BUILD_DIR
mkdir -p "$LOG"; : > "$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT INT TERM

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
git -C "$ROOT" diff --quiet -- "$CHECK" 2>/dev/null || {
  echo "FAIL: $CHECK is dirty; this matrix edits a sandbox copy."; exit 2; }

POP_BEFORE="$(pop_dirs | tr '\n' ' ')"
# The checker sources scripts/_gate_precond.sh and resolves BIN from
# INTFER_BUILD_DIR, so the sandbox needs those too. Sourcing failed with
# "_gate_precond.sh: No such file or directory" on the first run, and it also
# resolved BIN under /tmp/build -- both because the sandbox had only the one
# file. Link the siblings in rather than copying (copying would drift).
mkdir -p "$SB/scripts"
for sib in _gate_precond.sh bpe_tokenize.py bpe_detokenize.py; do
  [ -e "$SCRIPT_DIR/$sib" ] && ln -sf "$SCRIPT_DIR/$sib" "$SB/scripts/$sib"
done
cp "$CHECK" "$SB/scripts/check_e2_ctx.sh"; ORIG="$SB/.orig"; cp "$SB/scripts/check_e2_ctx.sh" "$ORIG"
restore() { cp "$ORIG" "$SB/scripts/check_e2_ctx.sh"; }
# Run ONLY the self-check: the real-run half needs the GPU and a model, and the
# baseline is about the predicate, not about the model.
run_selfcheck() { # runs check_e2_ctx.sh far enough to emit the SELFTEST lines
  ( cd "$SB/scripts" && INTFER_BUILD_DIR="$BUILD_DIR" INTFER_08B="$ROOT/test/models/Qwen3.5-0.8B.intifer" timeout 300 bash "$SB/scripts/check_e2_ctx.sh" 2>&1 )
}

echo "== stamp-population leak assertion (AS FOUND) =="
unknown="$(printf '%s\n' "$(pop_dirs)" | while read -r d; do
            [ -n "$d" ] || continue
            case " $(known_dirs) " in *" $d "*) ;; *) echo "$d" ;; esac; done)"
if [ -n "$unknown" ]; then
  echo "  FAIL  pre-existing foreign stamp dir(s): $unknown"; exit 1
fi
echo "  ok    $POP_BEFORE"

echo
echo "== baseline: the self-check must PASS and print all three case lines =="
restore
b="$(run_selfcheck)"
if ! printf '%s\n' "$b" | grep -q 'SELF-CHECK FAILED'; then :; else
  echo "BASELINE FAIL -- the self-check failed on the clean tree"; exit 1
fi
n_ok="$(printf '%s\n' "$b" | grep -cE '^\s+ok\s+\(')"
if [ "$n_ok" -lt 3 ]; then
  echo "BASELINE FAIL -- only $n_ok self-check ok lines; the healthy direction is unproven"
  printf '%s\n' "$b"; exit 1
fi
echo "  ok    baseline: $n_ok self-check case lines green"
echo "  ok    CONVERSE: a healthy self-check prints no SELFTEST FAIL line, so"
echo "        'absent' below means something"

caught=0; total=0
run_state() { # $1 id, $2 marker, $3 absent-RE
  local id="$1" marker="$2" expect="$3"
  total=$((total+1)); restore
  local before; before="$(cat "$SB/scripts/check_e2_ctx.sh")"
  if ! MUTATE_FN; then
    echo "  $id MISSED -- mutation function failed"
    echo "$id MISSED (mutation fn)" >> "$LOG/summary.txt"; restore; return 1
  fi
  cmp -s "$before" "$SB/scripts/check_e2_ctx.sh" && {
    echo "  $id MISSED -- mutation did not change the file"
    echo "$id MISSED (no-op)" >> "$LOG/summary.txt"; restore; return 1; }
  if ! grep -q -- "$marker" "$SB/scripts/check_e2_ctx.sh"; then
    echo "  $id MISSED -- marker absent: patch did not apply"
    echo "$id MISSED (stale anchor)" >> "$LOG/summary.txt"; restore; return 1
  fi
  bash -n "$SB/scripts/check_e2_ctx.sh" 2>/dev/null || {
    echo "  $id MISSED -- patch left the checker syntactically broken (STATE fault)"
    echo "$id MISSED (syntax)" >> "$LOG/summary.txt"; restore; return 1; }
  local o r
  o="$(run_selfcheck)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"; restore
  if [ "$r" -eq 0 ] || ! printf '%s\n' "$o" | grep -q 'SELF-CHECK FAILED'; then
    echo "  $id MISSED -- the checker still passed with the defect present"
    echo "$id MISSED (not detected)" >> "$LOG/summary.txt"; return 1
  fi
  if ! pred_expect_absent "$id ok-marker" "$o" "$expect"; then
    echo "$id MISSED (polarity or marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- SELF-CHECK FAILED, verdict-polarity=expect-absent"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

echo
echo "== states (self-check half only; see SCOPE) =="

# M-A: the must-REJECT predicate made unconditional (always REJECT). Cases (a)
# and (b) still pass, because they WANT a rejection -- what breaks is the
# CONVERSE, which is why my first version scored 0/4: it expected (a)'s ok line
# to vanish, and (a) was never what changed.
MUTATE_FN() {
  patch_literal "$SB/scripts/check_e2_ctx.sh" \
    'if [ "$1" != "1" ] || [ "$2" != "1" ]; then echo REJECT; return; fi' \
    'if true; then echo REJECT; return; fi  # MUT_A' \
    && verify_patched "$SB/scripts/check_e2_ctx.sh" 'MUT_A'
}
run_state "M-A" 'MUT_A' 'accepts a complete run'

# M-B: the converse case inverted, so a correct run is treated as wrong. NOT
# deleted -- deleting a case is silent, and a silent removal is exactly the
# unobservable mutation AUDIT #21 describes.
MUTATE_FN() {
  patch_literal "$SB/scripts/check_e2_ctx.sh" \
    'if [ "$(_verdict 1 1)" != "ACCEPT" ]; then' \
    'if [ "$(_verdict 1 1)" = "ACCEPT" ]; then  # MUT_B' \
    && verify_patched "$SB/scripts/check_e2_ctx.sh" 'MUT_B'
}
run_state "M-B" 'MUT_B' 'accepts a complete run'

# M-C: the no-decode-steps case evaluated against a COMPLETE run, so an
# incomplete one is scored as if it were fine.
MUTATE_FN() {
  patch_literal "$SB/scripts/check_e2_ctx.sh" \
    'if [ "$(_verdict 1 0)" = "ACCEPT" ]; then' \
    'if [ "$(_verdict 1 1)" = "ACCEPT" ]; then  # MUT_C' \
    && verify_patched "$SB/scripts/check_e2_ctx.sh" 'MUT_C'
}
run_state "M-C" 'MUT_C' 'rejects a run with no decode steps'

# M-D IS DELIBERATELY ABSENT, and the reason is the point. Removing the gate
# (`if ! selfcheck; then` -> `if false`) only becomes observable if the run
# REACHES the real-run half and produces a verdict there -- which needs a GPU
# and the model. Measured on this box: the ungated run does not reach a
# verdict, so the state cannot be scored, and scoring an unobservable state as
# "caught" is the lie AUDIT #21 describes. It is excluded rather than faked.
# To add it properly, run the matrix where the real half can complete.

echo
echo "== stamp-population leak assertion (after states) =="
POP_AFTER="$(pop_dirs | tr '\n' ' ')"
if [ "$POP_AFTER" = "$POP_BEFORE" ]; then
  echo "  ok    population unchanged"
else
  echo "  FAIL  population changed: before=$POP_BEFORE after=$POP_AFTER"
  echo "RESULT: e2_ctx_mutate LEAK"; exit 1
fi

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
if [ "$caught" -eq "$total" ]; then
  echo
  echo "E2 CTX MUTATE: PASS ($caught/$total self-check states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=e2_ctx_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "E2 CTX MUTATE: FAIL ($caught/$total caught)"
exit 1