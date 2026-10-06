#!/usr/bin/env bash
# Mutation matrix for scripts/check_build_dir_consumers.sh.
#
# WHAT IS MUTATED, AND WHY IT IS THE CORPUS AND NOT THE CHECKER (audit #21). The
# checker classifies every script that references a build tree as honoring the
# INTFER_BUILD_DIR override or hardcoding a root-relative path. The checker is
# correct by construction on a clean corpus, so a state that WEAKENS the checker
# is an unobservable no-op (it would pass whether or not the rule fires). The
# observable signal therefore has to come from damaging an ARTIFACT: a planted
# consumer script whose build-dir path is hardcoded, which only a real defect in
# that script would let the checker see as red.
#
# The corpus lives in an mktemp sandbox OUTSIDE $BUILD_DIR; the checker resolves
# ROOT from its own location, so a copy of the checker run from inside the
# sandbox grades the sandbox's scripts/ and CMakeLists.txt -- the real scripts/
# is never touched.
#
# States:
#   A  a consumer hardcodes "$ROOT/build"       -> must REJECT
#   B  a consumer hardcodes "${ROOT}/build"      -> must REJECT
#   C  a consumer mentions "$ROOT/build" ONLY in a comment -> must PASS (NOTE,
#      "matched on a COMMENT line only -- not a real consumer"). The mirror of
#      the defect: prose about a path is not a runtime reference.
#   D  converse: the planted consumer honors the override -> must PASS
#
# Exit: 0 = every state behaved as specified, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK="$SCRIPT_DIR/check_build_dir_consumers.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/build_dir_consumers_mutate"
LOG="$STAMP/states"
# *** sandbox OUTSIDE $BUILD_DIR so its fixtures cannot join the stamp census
# 3/3 compares, and cannot be scored as real residue under scripts/. ***
SB="$(mktemp -d "${TMPDIR:-/dev/shm}/bdcmut.XXXXXX")"
mkdir -p "$STAMP" "$LOG"
: >"$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

# shellcheck source=scripts/_gate_patch.sh
. "$SCRIPT_DIR/_gate_patch.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
# shellcheck source=scripts/_gate_pop.sh
. "$SCRIPT_DIR/_gate_pop.sh"

[ -f "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
bash -n "$CHECK" || { echo "FAIL: $CHECK is not syntactically valid"; exit 2; }

echo "== build_dir_consumers mutation matrix =="

# Build a minimal repo the checker can grade: a CMakeLists.txt (so the
# add_executable census is non-empty) and one COMPLIANT consumer so the
# enumeration guard (AUDIT #14) has a real subject -- without it the checker
# would refuse on "0 consumer scripts", indistinguishable from a broken query.
setup() {
  rm -rf "$SB"; mkdir -p "$SB/scripts" "$SB/build"
  cp "$CHECK" "$SB/scripts/check_build_dir_consumers.sh"
  ln -sf "$SCRIPT_DIR/_gate_enum.sh" "$SB/scripts/_gate_enum.sh"
  printf '# minimal\nadd_executable(failme failme.cpp)\n' > "$SB/CMakeLists.txt"
  cat > "$SB/scripts/compliant_consumer.sh" <<'EOF'
#!/usr/bin/env bash
FOO="${INTFER_BUILD_DIR:-$ROOT/build}/model"
EOF
}

# The planted body arrives on STDIN (heredoc); this avoids the audit #19 trap of
# a staging file that mk_corpus could rm and erase its own plant.
run_check() {
  bash "$SB/scripts/check_build_dir_consumers.sh" 2>&1
}

check_state() { # $1=id  $2=want_rc  $3=expect_re  $4=why
  local id="$1" want="$2" expect="$3" why="$4" o r
  total=$((total + 1))
  o="$(run_check)" ; r=$?
  printf '%s\n' "$o" >"$LOG/$id.log"
  printf '  .. %s planted; checker rc=%s\n' "$id" "$r"
  if [ "$r" -eq "$want" ] && pred_expect_present "$id expected-RE" "$o" "$expect"; then
    echo "  $id caught -- checker-rc=$r, verdict-polarity=expect-present ($why)"
    echo "$id caught (rc=$r)" >>"$LOG/summary.txt"; caught=$((caught + 1))
  else
    echo "  $id MISSED -- wanted rc=$want and a match for /$expect/ ($why)"
    echo "        got rc=$r; checker output tail:"
    printf '%s\n' "$o" | tail -4 | sed 's/^/          '
    echo "$id MISSED (rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1))
  fi
}

# ---- population snapshot for the leak assertion ------------------------------
# The leak check exists so a fixture stamp under a matrix-owned dir cannot be
# misread as real residue in run_all_checks.sh step 3/3.
POP_BEFORE="$(pop_dirs | tr '\n' ' ')" 2>/dev/null || POP_BEFORE=""

setup
total=0; caught=0; bad=0

echo
echo "-- baseline: a compliant corpus must PASS (the converse of always-red) --"
b="$(run_check)"; brc=$?
printf '%s\n' "$b" >"$LOG/baseline.log"
if [ "$brc" -ne 0 ]; then
  echo "  FAIL baseline -- the checker does not pass on a clean corpus, so"
  echo "       every state below would be scored against a broken baseline."
  printf '%s\n' "$b" | tail -8 | sed 's/^/    /'
  printf '%s\n' "$b" | tail -3 >>"$LOG/summary.txt"
  # still count it so the tally is honest
  total=$((total + 1)); bad=$((bad + 1))
else
  pred_expect_present "baseline harvested a subject" "$b" 'BUILD-DIR CONSUMER CHECK: PASS'
  # NOTE the parens: the checker prints `consumer(s)` literally, and an ERE
  # without them matches the bare word `consumers` -- which the checker
  # prints nowhere. Escaping is load-bearing here; the first version of this
  # matrix failed its own baseline for exactly that reason and every state
  # below scored against a broken baseline.
  pred_expect_present "baseline is non-vacuous" "$b" 'consumer\(s\), all honor INTFER_BUILD_DIR' \
    || { echo "  FAIL baseline verdict looks vacuous (no subject inspected)" >&2; bad=$((bad + 1)); }
  echo "  ok    baseline rc=0, population non-empty"
  total=$((total + 1)); caught=$((caught + 1))
fi

echo
echo "-- states --"

# A: hardcoded "$ROOT/build" -- the checker must catch it as a non-honoring consumer.
cat > "$SB/scripts/mutant.sh" <<'EOF'
#!/usr/bin/env bash
FOO="$ROOT/build/model"
EOF
check_state "A" 1 'ignore INTFER_BUILD_DIR' "hardcodes \$ROOT/build with no override"

# B: hardcoded "${ROOT}/build" -- the other root-relative spelling.
cat > "$SB/scripts/mutant.sh" <<'EOF'
#!/usr/bin/env bash
FOO="${ROOT}/build/model"
EOF
check_state "B" 1 'ignore INTFER_BUILD_DIR' "hardcodes \${ROOT}/build with no override"

# C: path appears only in a comment -- prose is not a runtime reference (NOTE).
cat > "$SB/scripts/mutant.sh" <<'EOF'
#!/usr/bin/env bash
# FOO would be "$ROOT/build/model" if this were a real consumer.
FOO=bar
EOF
check_state "C" 0 'BUILD-DIR CONSUMER CHECK: PASS' "did not grade a comment-only mention as a consumer"

# D: the planted consumer honors the override -- the checker must PASS it.
cat > "$SB/scripts/mutant.sh" <<'EOF'
#!/usr/bin/env bash
FOO="${INTFER_BUILD_DIR:-$ROOT/build}/model"
EOF
check_state "D" 0 'BUILD-DIR CONSUMER CHECK: PASS' "a compliant consumer is accepted (converse)"

echo
echo "-- stamp-population leak assertion (this matrix must own every stamp it writes) --"
if [ "$POP_BEFORE" != "  " ] 2>/dev/null; then
  POP_AFTER="$(pop_dirs | tr '\n' ' ')" 2>/dev/null || POP_AFTER=""
  if [ "$POP_AFTER" = "$POP_BEFORE" ]; then
    echo "  ok    stamp dir population unchanged (no unowned stamps leaked)"
  else
    echo "  FAIL  stamp dir population changed: before=[$POP_BEFORE] after=[$POP_AFTER]"
    echo "RESULT: build_dir_consumers_mutate LEAK"; bad=$((bad + 1))
  fi
else
  echo "  ok    build_dir_consumers does not write stamps into an enumerated dir"
fi

echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"

if [ "$caught" -eq "$total" ] && [ "$bad" -eq 0 ]; then
  echo
  echo "BUILD-DIR CONSUMER MUTATION MATRIX: PASS ($caught/$total states caught, each for its own reason)"
  { echo "stamp=build_dir_consumers_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"; echo "caught=$caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "BUILD-DIR CONSUMER MUTATION MATRIX: FAIL ($bad of $total states misbehaved)"
echo "RESULT: a planted consumer was not caught by the checker it mirrors"
exit 1
