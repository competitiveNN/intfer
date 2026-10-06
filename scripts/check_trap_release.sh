#!/usr/bin/env bash
# A cleanup trap must RELEASE ON EVERY EXIT PATH *AND TERMINATE*.
#
# WHY THIS EXISTS, and it is not a style rule. Two defects, measured:
#
#   (1) "trap 'rm -rf "$X"' EXIT INT TERM" -- the AUDIT #29 remedy, applied
#       mechanically -- cleans up and then RESUMES. The process SURVIVES the
#       signal. Every dispatch arm in run_all_checks.sh wraps its target in
#       `timeout`, which sends SIGTERM and then WAITS FOR THE CHILD, so a step
#       that ignores its signal does not fail: it hangs the whole stack, with no
#       output and no verdict. That is strictly worse than the leak AUDIT #29
#       recorded, because it is silent and it blocks certification.
#   (2) An EXIT-ONLY trap does not run at all on SIGTERM, which is the ORIGINAL
#       AUDIT #29 bug. Among those 17 files were the git-restore matrices
#       (mutate_run_all_checks.sh, mutate_engine_cert.sh, mutate_stack_cert.sh,
#       ...) whose cleanup IS the restore -- so terminating one leaves TRACKED
#       SOURCE MUTATED, and the next run's dirty-tree guard then refuses.
#
# And a third that hides both: bash DEFERS a trap while a FOREGROUND command
# runs. A plain `sleep 30` in a wait loop means the handler does not run for up
# to 30 seconds. Fixing (1) alone therefore LOOKS like it failed -- the handler
# was correct and had simply not run yet. Measured on bench_when_quiet.sh: 10s
# after SIGTERM the lock was still present and the process still alive; the same
# probe after backgrounding the sleep and `wait`ing on it reports rc 143 and the
# lock gone. So the rule here has a THIRD clause, and it is the one that is easy
# to forget because nothing in the tree states it.
#
# THE RULE, derived from source and never from a list of remembered files (the
# audit #7/#8 rule -- a hand-remembered list of the 40 files I happened to touch
# is a citation, not a closure):
#
#     a script that installs a cleanup trap (a `trap <body> EXIT` whose body is
#     not `-`) must ALSO install INT and TERM traps, and each of those bodies must
#     both clean up AND terminate (`exit <n>`).
#
#     `trap - ...` is a DISARM, not a cleanup, and is never a subject.
#     `trap ... RETURN` is a different mechanism (a function-return trap) and is
#     out of scope; the cost is named below rather than hidden.
#
# SCOPE, and its cost, stated rather than implied. Matching is anchored at LINE
# START after comment-stripping, because that is what a real `trap` command looks
# like; a mutation harness that quotes trap text as a patch argument is
# correctly not a subject, and needs no exemption to pass. There is deliberately
# NO check_*/mutate_* exclusion here, unlike check_dri_probe.sh: a verifier that
# installs a real trap at line start SHOULD comply, and the correct response to a
# flag is to fix the trap, not to exempt the file. The cost is real and named --
# a verifier whose cleanup genuinely cannot terminate would have to be exempted,
# and that exemption would be a hand-maintained list.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${INTFER_SCRIPTS_DIR:-$SCRIPT_DIR}"

# classify <file>: 0 = compliant, 1 = VIOLATION, 2 = not a subject (no cleanup trap)
# REASON is set to a short human string naming WHICH clause failed, because a
# population-wide FAIL that does not say which of three rules broke is a report
# nobody can act on.
classify() {
  local f="$1" code
  code="$(sed 's/^[[:space:]]*#.*$//' "$f")"
  REASON=""

  # A cleanup trap: `trap <body> EXIT`, body not `-` (that is a disarm).
  if ! printf '%s\n' "$code" | grep -qE '^[[:space:]]*trap[[:space:]]+[^[:space:]-].*EXIT'; then
    return 2
  fi
  # RETURN traps are a different mechanism; ignore them entirely.
  local cleanup
  cleanup="$(printf '%s\n' "$code" | grep -E '^[[:space:]]*trap[[:space:]]+[^[:space:]-].*EXIT' \
              | grep -v 'RETURN' | head -1)"
  [ -n "$cleanup" ] || return 2

  local sig
  for sig in INT TERM; do
    local body
    body="$(printf '%s\n' "$code" | grep -E "^[[:space:]]*trap[[:space:]]+.*\b$sig\b" | head -1)"
    if [ -z "$body" ]; then
      REASON="cleanup trap present but NO $sig trap -- an EXIT-only trap never runs on SIG$sig, so it leaks"
      return 1
    fi
    # Clean up AND terminate. `trap - INT` is a disarm and exempt.
    case "$body" in
      *"trap -"*) continue ;;
    esac
    if ! printf '%s' "$body" | grep -qE 'exit[[:space:]]+[0-9]+'; then
      REASON="$sig trap cleans up but does NOT terminate: '$body' -- the process survives the signal, and \`timeout\` waits for the child, so the step HANGS instead of failing"
      return 1
    fi
  done
  return 0
}

# *** BEHAVIOURAL SELF-CHECK FIRST, and it GATES the run. *** A rule that cannot
# fail proves nothing; one that fires on everything is the mirror defect. All four
# shapes are EXECUTABLE files driven through this exact function -- never a grep of
# this script's own source, which was the recorded self-defeating first attempt in
# mutate_fold_matrix.sh's own self-check.
TMP="$(mktemp -d "${TMPDIR:-/dev/shm}/trap_release.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
trap 'rm -rf "$TMP"; exit 130' INT
trap 'rm -rf "$TMP"; exit 143' TERM
self_bad=0

cat >"$TMP/exit_only.sh" <<'EOF'
#!/usr/bin/env bash
trap 'rm -rf "$WORK"' EXIT
EOF
cat >"$TMP/survives.sh" <<'EOF'
#!/usr/bin/env bash
trap 'rm -rf "$WORK"' EXIT
trap 'rm -rf "$WORK"; exit 130' INT
trap 'rm -rf "$WORK"' TERM
EOF
cat >"$TMP/good.sh" <<'EOF'
#!/usr/bin/env bash
trap 'rm -rf "$WORK"' EXIT
trap 'rm -rf "$WORK"; exit 130' INT
trap 'rm -rf "$WORK"; exit 143' TERM
EOF
cat >"$TMP/disarmed.sh" <<'EOF'
#!/usr/bin/env bash
trap serve_cleanup EXIT
trap 'serve_cleanup; exit 130' INT
trap 'serve_cleanup; exit 143' TERM
serve_cleanup
trap - EXIT INT TERM
EOF

classify "$TMP/exit_only.sh";  r1=$?
classify "$TMP/survives.sh";   r2=$?
classify "$TMP/good.sh";       r3=$?
classify "$TMP/disarmed.sh";   r4=$?
[ "$r1" -eq 1 ] || { echo "SELF-CHECK FAIL: an EXIT-only trap was not REJECTED (rc=$r1)"; self_bad=$((self_bad+1)); }
[ "$r2" -eq 1 ] || { echo "SELF-CHECK FAIL: a clean-up-without-terminate trap was not REJECTED (rc=$r2)"; self_bad=$((self_bad+1)); }
[ "$r3" -eq 0 ] || { echo "SELF-CHECK FAIL: the compliant shape was not ACCEPTED (rc=$r3)"; self_bad=$((self_bad+1)); }
[ "$r4" -eq 0 ] || { echo "SELF-CHECK FAIL: a later \`trap -\` disarm broke compliance (rc=$r4)"; self_bad=$((self_bad+1)); }
# And the converse that gets forgotten: a file with NO cleanup trap is not a
# subject at all, so this must not be able to fail on the whole tree.
cat >"$TMP/none.sh" <<'EOF'
#!/usr/bin/env bash
echo hello
EOF
classify "$TMP/none.sh"; r5=$?
[ "$r5" -eq 2 ] || { echo "SELF-CHECK FAIL: a script with no cleanup trap is not exempt (rc=$r5)"; self_bad=$((self_bad+1)); }

echo "== TRAP RELEASE CHECK (a cleanup trap must clean up AND terminate) =="
if [ "$self_bad" -eq 0 ]; then
  echo "  ok    self-check: EXIT-only REJECTED; clean-up-without-terminate REJECTED;"
  echo "        compliant ACCEPTED; later \`trap -\` disarm ACCEPTED; no-trap exempt"
else
  echo
  echo "TRAP RELEASE: FAIL (self-check did not behave; the rule is not trustworthy)"
  exit 1
fi

shopt -s nullglob
SCRIPTS=("$SCRIPTS_DIR"/*.sh)
shopt -u nullglob
# *** SELF-EXCLUSION BY IDENTITY, NOT BY NAME. *** This file plants the violating
# text, so including itself is a guaranteed false positive; comparing to $0 is an
# exclusion that cannot drift onto a different file.
self_is() { [ "$1" = "$0" ]; }

viol=0; sites=0
for f in "${SCRIPTS[@]}"; do
  self_is "$f" && continue
  classify "$f"; r=$?
  [ "$r" -eq 2 ] && continue
  sites=$((sites + 1))
  if [ "$r" -eq 0 ]; then
    printf '  ok    %-34s releases on EXIT/INT/TERM and terminates on INT/TERM\n' "$(basename "$f")"
  else
    printf '  FAIL  %-34s %s\n' "$(basename "$f")" "$REASON"
    viol=$((viol + 1))
  fi
done

# *** POPULATION GUARD. *** Assert the SUBJECT still exists in the corpus -- not
# that a violation exists. A healthy tree must be ALLOWED to have none: demanding
# sites > 0 is the audit #20 shape inverted, and it fails the moment the defect is
# fixed. (check_dri_probe.sh recorded exactly that mistake.)
if [ "$sites" -eq 0 ]; then
  echo "FAIL: no script in scripts/ installs a cleanup trap, so a green verdict" >&2
  echo "      would certify a population it never harvested." >&2
  exit 1
fi

echo "  ..    $sites script(s) install a cleanup trap"
echo
if [ "$viol" -eq 0 ]; then
  echo "TRAP RELEASE: PASS ($sites cleanup trap(s); all release and terminate)"
  echo "RESULT: every cleanup trap runs on EXIT, INT and TERM, and the INT/TERM"
  echo "        handlers terminate -- so \`timeout\` can always stop a step, and a"
  echo "        terminated matrix restores instead of leaving source mutated"
  exit 0
fi
echo "TRAP RELEASE: FAIL ($viol of $sites script(s) with a cleanup trap do not release/terminate)"
echo "RESULT: a cleanup trap does not release and terminate on every path"
exit 1
