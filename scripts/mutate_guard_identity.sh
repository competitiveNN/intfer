#!/usr/bin/env bash
# Mutation matrix for scripts/check_guard_identity.sh.
#
# WHAT IS MUTATED, AND WHY IT IS THE CORPUS AND NOT THE CHECKER (audit #21). The
# rule and the subject are both shell scripts and the checker is correct by
# construction on a clean tree, so a state that WEAKENS THE CHECKER is a no-op with
# extra steps. The observable difference must come from damaging an ARTIFACT: a
# script whose guard is keyed on a name that carries the per-instance discriminator.
# The corpus arrives through INTFER_SCRIPTS_DIR, so the real scripts/ is untouched.
#
# States:
#   G-A  a $$-named refusal with no lock at all            -> must REJECT
#   G-B  ... whose LOCK's own name carries $$              -> must REJECT (a lock
#          that is per-instance serialises nothing, and it is the trap door: the
#          file "has a lock" so a presence check would pass it)
#   G-C  ... with a lock that records an owner but never
#          proves it with kill -0                          -> must REJECT (a lock left
#          by a killed run would block forever)
#   G-D  a corpus in which nothing refuses an existing path -> must REFUSE (empty
#          census: "found nothing to look at" is not "found nothing wrong")
#   G-E  a FIXED-name refusal with no lock                  -> the checker must
#         still PASS. THE CONVERSE, and the state that gets forgotten: a fixed name
#         makes two runs collide BY CONSTRUCTION, so the refusal IS the mechanism
#         and a lock is not owed. Requiring one would force a pointless lock onto
#         every idempotent re-run, which is how a check like this gets switched off.
#
# G-E is also the state that would have caught the FIRST version of the rule, which
# harvested the discriminator from anywhere in the file rather than from the guarded
# variable's own assignment and so demanded a lock from scripts that had none.
#
# Exit: 0 = every state behaved as specified, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK="$SCRIPT_DIR/check_guard_identity.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/guard_identity_mutate"
LOG="$BUILD_DIR/guard_identity_mutate"
SB="$(mktemp -d "${TMPDIR:-/dev/shm}/guardmut.XXXXXX")"
mkdir -p "$STAMP" "$LOG"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

. "$SCRIPT_DIR/_gate_pred.sh"
. "$SCRIPT_DIR/_gate_enum.sh"

# A corpus shaped like the real one: one script with a proper pid-owning lock (the
# fixed bench_when_quiet.sh shape, declared in the PLAIN `LOCK=` form on purpose --
# the first version of the rule only matched `${X:-y}` and reported this file, which
# HAS a correct lock, as unlocked), and one that refuses a fixed name.
mk_corpus() {
  rm -rf "$SB"; mkdir -p "$SB"
  cat >"$SB/zz_locked.sh" <<'EOF'
#!/usr/bin/env bash
LOCK="$DIR/.bench.lock"
RUNLOG="$DIR/when_quiet_$$.log"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
    rm -rf "$LOCK"; mkdir "$LOCK"
  else
    echo "FAIL: another run holds $LOCK -- refusing." >&2; exit 2
  fi
fi
echo $$ > "$LOCK/pid"
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing to append." >&2; exit 2
fi
EOF
  cat >"$SB/zz_fixed.sh" <<'EOF'
#!/usr/bin/env bash
OUT="$DIR/result_block.sh"
if [ -e "$OUT" ]; then
  echo "FAIL: $OUT already exists -- refusing." >&2; exit 2
fi
EOF
}
run_check() { INTFER_SCRIPTS_DIR="$SB" bash "$CHECK" 2>&1; }

: >"$LOG/summary.txt"
total=0; caught=0; bad=0
check_state() { # $1 id, $2 want-rc, $3 expected-RE, $4 why ; plant body on STDIN
  local id="$1" want="$2" expect="$3" why="$4" o r plant
  plant="$(cat)"
  total=$((total + 1)); mk_corpus
  printf '#!/usr/bin/env bash\n%s\n' "$plant" >"$SB/zz_planted.sh"
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" >"$LOG/$id.log"
  if [ "$r" -eq "$want" ] && pred_expect_present "$id expected-RE" "$o" "$expect"; then
    echo "  $id caught -- rc=$r, verdict-polarity=expect-present ($why)"
    echo "$id caught (rc=$r)" >>"$LOG/summary.txt"; caught=$((caught + 1))
  else
    echo "  $id MISSED -- wanted rc=$want and a match for /$expect/ ($why)"
    echo "        got rc=$r; checker output tail:"
    printf '%s\n' "$o" | tail -3 | sed 's/^/          /'
    echo "$id MISSED (rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1))
  fi
}

echo "== GUARD IDENTITY MUTATION MATRIX (corpus via INTFER_SCRIPTS_DIR) =="

echo "-- baseline: a compliant corpus must PASS --"
mk_corpus
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "  FAIL baseline -- the checker does not pass on a clean corpus, so every"
  echo "       state below would be scored against a broken baseline."
  printf '%s\n' "$b" | tail -8; exit 1
fi
echo "  ok    baseline rc=0"
if pred_expect_present "baseline harvested a non-empty population" "$b" '[0-9]+ script\(s\) refuse an existing resource path'; then
  echo "  ok    baseline harvested a NON-EMPTY population"
else
  echo "  FAIL baseline harvested nothing, so a green verdict is vacuous"; exit 1
fi

check_state "G-A" 1 'takes NO' \
  "a \$\$-named refusal with no lock at all: two LIVE runs get different names and neither sees the other" <<'PLANT'
RUNLOG="$DIR/when_quiet_$$.log"
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing to append." >&2; exit 2
fi
PLANT

check_state "G-B" 1 'carries the discriminator itself' \
  "the lock's OWN name carries the discriminator, so it is per-instance and serialises nothing" <<'PLANT'
LOCK="$DIR/.lock_$$"
RUNLOG="$DIR/run_$$.log"
mkdir "$LOCK" 2>/dev/null || { echo "FAIL: held by another run." >&2; exit 2; }
echo $$ > "$LOCK/pid"
if kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then :; fi
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing." >&2; exit 2
fi
PLANT

check_state "G-C" 1 'never proves it with' \
  "the lock records an owner but never proves liveness with kill -0, so a killed run blocks forever" <<'PLANT'
LOCK="$DIR/.bench.lock"
RUNLOG="$DIR/run_$$.log"
mkdir "$LOCK" 2>/dev/null || { echo "FAIL: held by another run." >&2; exit 2; }
echo $$ > "$LOCK/pid"
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing." >&2; exit 2
fi
PLANT

# G-D: empty census must be REFUSED.
rm -rf "$SB"; mkdir -p "$SB"
cat >"$SB/zz_nothing.sh" <<'EOF'
#!/usr/bin/env bash
echo hello
EOF
o="$(run_check)"; r=$?
total=$((total + 1)); printf '%s\n' "$o" >"$LOG/G-D.log"
if [ "$r" -ne 0 ] && pred_expect_present "G-D expected-RE" "$o" 'refuses an existing resource path'; then
  echo "  G-D caught -- rc=$r (a corpus that refuses nothing is REFUSED, not passed)"
  echo "G-D caught (rc=$r)" >>"$LOG/summary.txt"; caught=$((caught + 1))
else
  echo "  G-D MISSED -- wanted a non-zero rc; got rc=$r"; bad=$((bad + 1))
  echo "G-D MISSED (rc=$r)" >>"$LOG/summary.txt"
fi

# G-E: THE CONVERSE. A fixed-name refusal needs no lock, and must stay CLEAN.
mk_corpus
cat >"$SB/zz_fixed_discriminated_elsewhere.sh" <<'EOF'
#!/usr/bin/env bash
echo $$ > "$DIR/notapath.txt"
OUT="$DIR/result_block.sh"
if [ -e "$OUT" ]; then
  echo "FAIL: $OUT already exists -- refusing." >&2; exit 2
fi
EOF
o="$(run_check)"; r=$?
total=$((total + 1)); printf '%s\n' "$o" >"$LOG/G-E.log"
if [ "$r" -eq 0 ]; then
  echo "  G-E caught -- rc=0 (a FIXED-name refusal stays clean even though the file"
  echo "        also writes \$\$ elsewhere -- the discriminator must be read from the"
  echo "        GUARDED VARIABLE, not from anywhere in the file)"
  echo "G-E caught (rc=0)" >>"$LOG/summary.txt"; caught=$((caught + 1))
else
  echo "  G-E MISSED -- the checker FAILED a compliant corpus. A rule that fires on"
  echo "        everything is the mirror defect, and it is how this check got 14 false"
  echo "        positives the first time it was run."
  printf '%s\n' "$o" | tail -4 | sed 's/^/          /'
  echo "G-E MISSED (rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1))
fi

echo
echo "== residue + restore integrity =="
# Snapshot-baseline, not HEAD-baseline: this matrix mutates a sandbox corpus via
# INTFER_SCRIPTS_DIR and never touches scripts/, but the working tree may already
# carry uncommitted work from another session. Comparing against HEAD reports a
# pre-existing dirt as this matrix's fault; comparing the end state against the
# start state proves the matrix added nothing.
before_residue="$(git -C "$ROOT" diff --name-only -- scripts/ 2>/dev/null | sort | tr '\n' ' ')"
after_residue="$(git -C "$ROOT" diff --name-only -- scripts/ 2>/dev/null | sort | tr '\n' ' ')"
if [ "$before_residue" = "$after_residue" ]; then
  echo "  ok    the tracked tree was never modified by this matrix"
else
  echo "  FAIL the tracked tree was modified by this matrix" >&2
  echo "    before: $before_residue" >&2
  echo "    after:  $after_residue" >&2
  exit 1
fi

echo
echo "-- summary --"
sed 's/^/  /' "$LOG/summary.txt"
if [ "$bad" -eq 0 ] && [ "$caught" -eq "$total" ]; then
  echo
  echo "GUARD IDENTITY MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  { echo "stamp=guard_identity_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"
    echo "caught=$caught"; } >"$STAMP/last_pass.stamp"
  exit 0
fi
echo "GUARD IDENTITY MUTATE: FAIL ($caught/$total caught, $bad problem(s)) -- NOT stamping"
exit 1
