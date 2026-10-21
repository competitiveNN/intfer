#!/usr/bin/env bash
# No script may perform a DESTRUCTIVE git operation without refusing to run on a
# dirty tree -- and the standing rule must be written down where agents read it.
#
# WHY THIS EXISTS, MEASURED. In one session a throwaway probe was reverted with
#
#     git reset --hard HEAD~1
#
# run from a shell that was carrying roughly forty minutes of unrelated
# uncommitted work. The command succeeded, the work was gone, and nothing said
# so. No script did that -- the hazard was a human (or agent) typing it -- which
# is precisely why it needed an in-repo artifact rather than a memory note: this
# repo's own rule is that a certification is a re-runnable artifact, not a
# transcript, and a habit that destroyed a session's work is no more durable than
# a habit that certified one.
#
# WHAT IS CHECKED, and it is deliberately NARROW, because each half of it has a
# decoy:
#
#  1. HARVEST FROM CODE, NOT PROSE. scripts/matrices/mutate_tp2_selfcheck.sh contains the
#     literal `git checkout --` exactly ONCE -- inside a comment explaining that
#     it deliberately has no dirty-tree guard because it does not restore that
#     way. A grep over the file therefore flags a script that is correct, and the
#     obvious response -- adding a guard nobody needs -- makes the corpus worse.
#     So comments are stripped BEFORE the harvest. This is the same lesson as
#     removing the hand-written flag census from AGENTS.md: a claim derived from
#     prose decays, and the fix is to derive from the executable text.
#
#  2. EVERY SCRIPT THAT DOES ONE MUST GUARD. The property is a pairing, not a
#     count: a script that overwrites a tracked file must have refused to start if
#     that file was already modified. Unpaired = the file it restores is
#     indistinguishable from the work it destroyed.
#
# ANTI-VACUITY. The harvest is a QUERY. If it stops finding destructive scripts
# the corpus silently becomes "nothing to check", which is the audit #2/#3 shape.
# So the number found is printed, required non-zero, and the accepted/rejected
# split is required to account for every one of them.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

echo "== DESTRUCTIVE-GIT GUARD COVERAGE =="

# Strip comments before harvesting. A `#` to end-of-line is enough here: these
# are shell scripts, and none of them contain a `#` inside a string that carries a
# git operation. If one ever did, the harvest would go quiet rather than wrong,
# and the anti-vacuity floor below would notice.
strip_comments() { sed -e 's/[[:space:]]#.*$//' -e 's/^[[:space:]]*#.*$//' "$1"; }

# The destructive set, derived from the scripts rather than remembered:
#   reset --hard   discards staged AND unstaged work, including UNTRACKED-adjacent
#                  state that cannot be recovered
#   clean -f       deletes untracked files outright
#   checkout --    overwrites working-tree changes with HEAD
DESTRUCTIVE_RE='git[[:space:]]+(reset[[:space:]]+--hard|clean[[:space:]]+-[a-zA-Z]*f|checkout[[:space:]]+--)'

n_found=0; n_guarded=0; n_guards=0; unguarded=""
for f in $(find scripts -maxdepth 2 -not -path "*/archive/*" -name '*.sh' -type f); do
  [ -f "$f" ] || continue
  ops="$(strip_comments "$f" | grep -cE "$DESTRUCTIVE_RE" || true)"
  [ "$ops" -gt 0 ] || continue
  # A GUARD is not an EXECUTOR. scripts/guard.sh contains every destructive
  # literal -- it is the thing that REFUSES them, and each occurrence ends in
  # `command git "$@"`. Without this exemption the checker flags the guard for
  # doing its job, and the tempting "fix" is to weaken the guard. Self-declared,
  # for the same reason POST-STACK is: a second list in a checker is a
  # hand-maintained inventory, which is what this repo keeps deleting.
  if head -3 "$f" | grep -q '^# GUARD:'; then
    printf '  ok    %-38s %s destructive literal(s), SELF-DECLARED GUARD\n' \
      "$(basename "$f")" "$ops"
    n_guards=$((n_guards + 1))
    continue
  fi
  n_found=$((n_found + 1))
  if strip_comments "$f" \
     | grep -qE 'git diff --quiet|git status --porcelain|git_guard_clean'; then
    n_guarded=$((n_guarded + 1))
    # TIGHTENED, and the mutation matrix is what found it. The first version also
    # accepted the bare words DIRTY and "uncommitted changes", so a script whose
    # guard had been REPLACED BY `if false` still matched on the variable
    # assignment it left behind -- the checker reported "guarded" from a WORD. A
    # guard is a test, so only a test counts: a dirty-tree query or the shared
    # helper. The prose around the mechanism is not the mechanism.
    printf '  ok    %-38s %s destructive op(s), dirty-tree guard present\n' \
      "$(basename "$f")" "$ops"
  else
    printf '  FAIL  %-38s %s destructive op(s), NO dirty-tree guard\n' \
      "$(basename "$f")" "$ops" >&2
    unguarded="$unguarded $(basename "$f")"
  fi
done

echo "-- destructive scripts: $n_found (guarded $n_guarded), self-declared guards: $n_guards, unguarded: ${unguarded:-none}"

bad=0
# ANTI-VACUITY: this corpus is populated, and stays populated. A refactor that
# removed every `git checkout --` would otherwise make this checker certify
# nothing while looking identical to a healthy tree.
if [ "$n_found" -lt 1 ]; then
  echo "  FAIL  no destructive git operation found in any script -- the harvest is" >&2
  echo "        broken, so this would certify an empty corpus." >&2
  exit 1
fi
if [ -n "$unguarded" ]; then
  echo "  FAIL  these scripts can destroy uncommitted work: $unguarded" >&2
  bad=$((bad + 1))
fi

# ---- the standing RULE must be written where an agent reads it -------------
# A guard that only protects scripts leaves the case that actually bit me -- a
# destructive command typed at a prompt -- entirely uncovered. That half cannot be
# enforced by a script, so it is asserted as a standing rule in AGENTS.md.
if grep -qE 'commit .{0,40}before .{0,40}destructive|reset --hard.{0,80}uncommitted' AGENTS.md; then
  echo "  ok    AGENTS.md carries the commit-before-a-destructive-probe rule"
else
  echo "  FAIL  AGENTS.md does not carry the commit-before-a-destructive-probe" >&2
  echo "        rule. The script guard cannot cover a command typed at a prompt," >&2
  echo "        and that is the case that actually destroyed work." >&2
  bad=$((bad + 1))
fi

echo
if [ "$bad" = 0 ]; then
  echo "DESTRUCTIVE-GIT GUARD: PASS ($n_guarded/$n_found destructive scripts guarded, $n_guards self-declared guard(s) exempt;"
  echo "  the AGENTS.md rule covers the prompt case)"
  echo "RESULT: no script can destroy uncommitted work, and the rule for the"
  echo "        prompt case is in the standing rules"
  exit 0
fi
echo "DESTRUCTIVE-GIT GUARD: FAIL ($bad problem(s))" >&2
echo "RESULT: FAIL ($bad problem(s)) -- uncommitted work is destroyable" >&2
exit 1
