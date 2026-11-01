#!/usr/bin/env bash
# No script may restore tracked files with `git checkout -- <path>`.
#
# WHY THIS EXISTS, and it is not a style rule. `git checkout -- <paths>`
# OVERWRITES working-tree changes with HEAD — it destroys uncommitted work.
# Every mutation matrix that edits tracked source in place now restores through
# scripts/lib/_gate_snapshot.sh (snapshot + mode/mtime-preserving restore),
# precisely so uncommitted work is never destroyed. A script that reaches for
# `git checkout --` instead is a regression waiting to destroy someone's tree.
#
# Measured: every pre-migration reference to `git checkout --` across scripts/
# was a comment explaining the defect, not executable code. This check strips
# comments first (matching check_trap_release.sh's convention) so those
# historical references do not fire, and flags only actual executable usage.
#
# THE RULE, derived from source and never from a remembered list:
#
#     an executable line matching `git checkout -- <path...>` in any
#     scripts/*.sh is a violation. `git checkout <branch>` (branch switching,
#     no `--`) is not in scope. The correct replacement is snapshot restore:
#       . "$SCRIPT_DIR/../lib/_gate_snapshot.sh"
#       gate_snap_init "$TGT_A $TGT_B"
#       gate_snap_restore "$TGT_A"
#
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${INTFER_SCRIPTS_DIR:-$SCRIPT_DIR}"
SELF_ABS="$(readlink -f "$0" 2>/dev/null || echo "$0")"

# classify <file>: 1 = VIOLATION (destructive git checkout -- in executable code),
#                 2 = not a subject (no destructive restore at all)
# rc=0 is unreachable: any executable `git checkout -- <path>` is always a
# violation; there is no "compliant subject" state. REASON names the offending line.
classify() {
  local f="$1" code lineno line
  REASON=""
  # Strip full-line comments, matching check_trap_release.sh's convention.
  code="$(sed 's/^[[:space:]]*#.*$//' "$f")"
  # Match `git checkout --` followed by at least one path argument.
  # `git checkout <branch>` (no --) is not matched and not a subject.
  lineno="$(printf '%s\n' "$code" | grep -nE 'git checkout --[[:space:]]+[^[:space:]]' | head -1 | cut -d: -f1)"
  [ -z "$lineno" ] && return 2
  line="$(printf '%s\n' "$code" | sed -n "${lineno}p")"
  REASON="line $lineno uses \`git checkout --\` (destructive tracked-file restore): $line"
  return 1
}

# *** BEHAVIOURAL SELF-CHECK FIRST, and it GATES the run. ***
TMP="$(mktemp -d "${TMPDIR:-/dev/shm}/no_git_checkout.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
trap 'rm -rf "$TMP"; exit 130' INT
trap 'rm -rf "$TMP"; exit 143' TERM
self_bad=0

cat >"$TMP/violating.sh" <<'EOF'
#!/usr/bin/env bash
# Destructive restore: overwrites uncommitted work with HEAD.
git checkout -- scripts/checks/check_stack_cert.sh
EOF

cat >"$TMP/compliant.sh" <<'EOF'
#!/usr/bin/env bash
. "$SCRIPT_DIR/../lib/_gate_snapshot.sh"
gate_snap_init "$TGT_A $TGT_B"
gate_snap_restore "$TGT_A"
EOF

cat >"$TMP/comment_only.sh" <<'EOF'
#!/usr/bin/env bash
# Historical note: `git checkout -- scripts/` destroyed ~40 minutes of work.
# This matrix now uses snapshot restore instead; no executable git checkout.
echo "restoring via snapshot"
EOF

cat >"$TMP/branch_switch.sh" <<'EOF'
#!/usr/bin/env bash
# Switching branches is legitimate; only `git checkout -- <path>` is banned.
git checkout main
EOF

classify "$TMP/violating.sh";   r1=$?
classify "$TMP/compliant.sh";   r2=$?
classify "$TMP/comment_only.sh"; r3=$?
classify "$TMP/branch_switch.sh"; r4=$?
[ "$r1" -eq 1 ] || { echo "SELF-CHECK FAIL: executable git checkout -- was not REJECTED (rc=$r1)"; self_bad=$((self_bad+1)); }
[ "$r2" -eq 2 ] || { echo "SELF-CHECK FAIL: snapshot-restore script was not EXEMPTED (rc=$r2)"; self_bad=$((self_bad+1)); }
[ "$r3" -eq 2 ] || { echo "SELF-CHECK FAIL: comment-only git checkout -- was not EXEMPTED (rc=$r3)"; self_bad=$((self_bad+1)); }
[ "$r4" -eq 2 ] || { echo "SELF-CHECK FAIL: branch switch was not EXEMPTED (rc=$r4)"; self_bad=$((self_bad+1)); }

echo "== NO DIRECT GIT CHECKOUT CHECK (tracked-file restore must use snapshot) =="
if [ "$self_bad" -eq 0 ]; then
  echo "  ok    self-check: destructive restore REJECTED; snapshot restore EXEMPTED;"
  echo "        comment-only EXEMPTED; branch switch EXEMPTED"
else
  echo
  echo "NO DIRECT GIT CHECKOUT: FAIL (self-check did not behave; the rule is not trustworthy)"
  exit 1
fi

shopt -s nullglob
SCRIPTS=("$SCRIPTS_DIR"/*.sh)
shopt -u nullglob

self_is() { [ "$(readlink -f "$1" 2>/dev/null || echo "$1")" = "$SELF_ABS" ]; }

viol=0; subjects=0
for f in "${SCRIPTS[@]}"; do
  self_is "$f" && continue
  classify "$f"; r=$?
  [ "$r" -eq 2 ] && continue
  subjects=$((subjects + 1))
  if [ "$r" -eq 1 ]; then
    printf '  FAIL  %-34s %s\n' "$(basename "$f")" "$REASON"
    viol=$((viol + 1))
  fi
done

# Population guard: demand scripts exist, but do NOT demand violations.
if [ "${#SCRIPTS[@]}" -eq 0 ]; then
  echo "FAIL: no script in scripts/ to harvest, so a green verdict" >&2
  echo "      would certify a population it never examined." >&2
  exit 1
fi

echo "  ..    ${#SCRIPTS[@]} script(s) examined; $subjects use destructive git checkout --"
echo
if [ "$viol" -eq 0 ]; then
  echo "NO DIRECT GIT CHECKOUT: PASS (${#SCRIPTS[@]} scripts; none restore tracked files destructively)"
  echo "RESULT: no script reintroduces git checkout -- <path>; every matrix restores via"
  echo "        scripts/lib/_gate_snapshot.sh and preserves uncommitted work"
  exit 0
fi
echo "NO DIRECT GIT CHECKOUT: FAIL ($viol of $subjects script(s) restore tracked files destructively)"
echo "RESULT: a script reintroduces git checkout -- <path>, which destroys uncommitted work"
exit 1
