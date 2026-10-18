#!/usr/bin/env bash
# Mutation matrix for scripts/check_trap_release.sh.
#
# WHAT IS MUTATED, AND WHY IT IS THE CORPUS AND NOT THE CHECKER (audit #21). The
# rule and the subject are both shell scripts, and the checker is correct by
# construction on a clean tree. A state that WEAKENS THE CHECKER is therefore a
# no-op with extra steps -- which is exactly what this repo's first wasted matrix
# attempt looked like (0/4, every fixture correct so the checker kept passing). The
# observable difference has to come from damaging an ARTIFACT: a script whose
# cleanup trap does not release, or does not terminate. The corpus is handed over
# through INTFER_SCRIPTS_DIR, so the real scripts/ is never touched.
#
# States:
#   T-A  a corpus script traps EXIT only                 -> must REJECT
#   T-B  ... INT/TERM that clean up but do not exit      -> must REJECT (the
#         `timeout`-cannot-kill-it defect: the process survives the signal)
#   T-C  ... an INT trap present but the TERM one missing -> must REJECT (asymmetric:
#         a half-applied fix is the normal way this regresses)
#   T-D  a corpus of scripts with NO cleanup trap at all -> must REFUSE (empty
#         census: "found nothing to look at" is not "found nothing wrong")
#   T-E  the compliant shape, and a `trap -` disarm        -> the checker must still
#         PASS. THE CONVERSE. Without it a rule that flags everything satisfies
#         T-A..T-D and the whole matrix proves nothing -- and it is the state that
#         gets forgotten, so it is a state.
#
# Exit: 0 = every state behaved as specified, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK="$SCRIPT_DIR/check_trap_release.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/trap_release_mutate"
LOG="$BUILD_DIR/trap_release_mutate"
SB="$(mktemp -d "${TMPDIR:-/dev/shm}/trapmut.XXXXXX")"
mkdir -p "$STAMP" "$LOG"
# *** summary.txt IS TRUNCATED, NOT APPENDED. *** The first version only mkdir'd
# it, so a second run printed the PREVIOUS run-s MISSED lines directly above its
# own "caught" lines under a PASS banner: the in-memory counters made the
# VERDICT right while the EVIDENCE was wrong, which is the worst place for it
# (audit #23). The counters are what decide the exit code; the file is what a
# human reads, and it must describe THIS run.
: >"$LOG/summary.txt"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

. "$SCRIPT_DIR/_gate_pred.sh"
. "$SCRIPT_DIR/_gate_enum.sh"

# A corpus shaped like the real one: one compliant, one that arms cleanup then
# explicitly runs it and DISARMS. Both must stay clean in every state.
mk_corpus() {
  rm -rf "$SB"; mkdir -p "$SB"
  cat >"$SB/zz_good.sh" <<'EOF'
#!/usr/bin/env bash
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
trap 'rm -rf "$WORK"; exit 130' INT
trap 'rm -rf "$WORK"; exit 143' TERM
EOF
  cat >"$SB/zz_disarmed.sh" <<'EOF'
#!/usr/bin/env bash
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
cleanup
trap - EXIT INT TERM
EOF
}
run_check() { INTFER_SCRIPTS_DIR="$SB" bash "$CHECK" 2>&1; }

total=0; caught=0; bad=0
# The planted script body arrives on STDIN. An earlier version staged it in
# $SB/.plant and then called mk_corpus, which does `rm -rf "$SB"` -- so it deleted
# its own plant and all three states scored MISSED for a reason that had nothing to
# do with the rule. A state that fails for an unrelated cause is worse than a MISSED
# one: it looks like the checker is weak when the HARNESS is broken.
check_state() { # $1 id, $2 want-rc (1 = must reject, 0 = must pass), $3 expected-RE, $4 why
  local id="$1" want="$2" expect="$3" why="$4" o r plant
  plant="$(cat)"
  total=$((total + 1)); mk_corpus
  printf '#!/usr/bin/env bash\n%s\n' "$plant" >"$SB/zz_planted.sh"
  o="$(run_check)"; r=$?
  printf '%s\n' "$o" >"$LOG/$id.log"
  printf '  .. %s planted; checker rc=%s\n' "$id" "$r"
  if [ "$r" -eq "$want" ] && pred_expect_present "$id expected-RE" "$o" "$expect"; then
    echo "  $id caught -- rc=$r, verdict-polarity=expect-present ($why)"
    echo "$id caught (rc=$r)" >>"$LOG/summary.txt"; caught=$((caught + 1))
  else
    echo "  $id MISSED -- wanted rc=$want and a match for /$expect/ ($why)"
    echo "        got rc=$r; checker output tail:"; printf '%s\n' "$o" | tail -3 | sed 's/^/          /'
    echo "$id MISSED (rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1))
  fi
  }

echo "== TRAP RELEASE MUTATION MATRIX (corpus via INTFER_SCRIPTS_DIR) =="

echo "-- baseline: a compliant corpus must PASS --"
mk_corpus
b="$(run_check)"; brc=$?
if [ "$brc" -ne 0 ]; then
  echo "  FAIL baseline -- the checker does not pass on a clean corpus, so every"
  echo "       state below would be scored against a broken baseline."; printf '%s\n' "$b" | tail -6
  exit 1
fi
echo "  ok    baseline rc=0"
pred_expect_present "baseline shows the subject was harvested" "$b" '[0-9]+ script\(s\) install a cleanup trap' \
  && echo "  ok    baseline harvested a NON-EMPTY population" \
  || { echo "  FAIL baseline harvested nothing, so a green verdict is vacuous"; exit 1; }

# T-A: EXIT only.
check_state "T-A" 1 'cleanup trap present but NO INT trap' \
  "an EXIT-only trap never runs on SIGINT/SIGTERM, so it leaks (the original AUDIT #29 bug)" <<'PLANT'
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PLANT

# T-B: clean up but do not terminate. The `timeout`-cannot-kill-it defect.
check_state "T-B" 1 'TERM trap cleans up but does NOT terminate' \
  "the TERM trap cleans up but does not exit, so the process survives the signal and timeout waits forever" <<'PLANT'
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
trap 'rm -rf "$WORK"; exit 130' INT
trap 'rm -rf "$WORK"' TERM
PLANT

# T-C: asymmetric -- INT present, TERM missing. The normal way a hand-applied fix
# regresses: one line gets edited and the other does not.
check_state "T-C" 1 'cleanup trap present but NO TERM trap' \
  "an asymmetric fix: INT releases and terminates, TERM is absent entirely" <<'PLANT'
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
trap 'rm -rf "$WORK"; exit 130' INT
PLANT

# T-D: empty census must be REFUSED, not passed over.
rm -rf "$SB"; mkdir -p "$SB"
cat >"$SB/zz_nothing.sh" <<'EOF'
#!/usr/bin/env bash
echo hello
EOF
o="$(run_check)"; r=$?
total=$((total + 1))
printf '%s\n' "$o" >"$LOG/T-D.log"
if [ "$r" -ne 0 ] && pred_expect_present "T-D expected-RE" "$o" 'installs a cleanup trap'; then
  echo "  T-D caught -- rc=$r (a corpus with no cleanup trap is REFUSED, not passed)"
  echo "T-D caught (rc=$r)" >>"$LOG/summary.txt"; caught=$((caught + 1))
else
  echo "  T-D MISSED -- wanted a non-zero rc; got rc=$r. 'Found nothing to look at'"
  echo "        and 'found nothing wrong' are different claims."; bad=$((bad + 1))
  echo "T-D MISSED (rc=$r)" >>"$LOG/summary.txt"
fi

# T-E: THE CONVERSE. A compliant corpus -- including the `trap -` disarm -- must
# stay clean. Without this, a rule that flagged every trap would satisfy T-A..T-D.
mk_corpus
o="$(run_check)"; r=$?
total=$((total + 1))
printf '%s\n' "$o" >"$LOG/T-E.log"
if [ "$r" -eq 0 ]; then
  echo "  T-E caught -- rc=0 (the compliant shape and the \`trap -\` disarm stay CLEAN)"
  echo "T-E caught (rc=0)" >>"$LOG/summary.txt"; caught=$((caught + 1))
else
  echo "  T-E MISSED -- the checker FAILED a compliant corpus. A rule that fires on"
  echo "        everything is the mirror defect and would train everyone to ignore it."
  printf '%s\n' "$o" | tail -4 | sed 's/^/          /'
  echo "T-E MISSED (rc=$r)" >>"$LOG/summary.txt"; bad=$((bad + 1))
fi

# Residue: nothing may be left behind, and the real tree must be untouched.
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
left="$(find "$SB" -mindepth 1 | wc -l)"
echo "  ok    the sandbox is a throwaway dir under TMPDIR (nothing under the build tree)"

echo
echo "-- summary --"
sed 's/^/  /' "$LOG/summary.txt"
if [ "$bad" -eq 0 ] && [ "$caught" -eq "$total" ]; then
  echo
  echo "TRAP RELEASE MUTATE: PASS ($caught/$total states caught, each for its own reason)"
  { echo "stamp=trap_release_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$total"
    echo "caught=$caught"; } >"$STAMP/last_pass.stamp"
  exit 0
fi
echo "TRAP RELEASE MUTATE: FAIL ($caught/$total caught, $bad problem(s)) -- NOT stamping"
exit 1
