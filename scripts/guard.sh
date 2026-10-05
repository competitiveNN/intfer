# Sourced by an interactive or agent shell to make destructive git refuse on a
# dirty tree. NOT executable -- it must be SOURCED, because the whole point is to
# shadow the `git` FUNCTION for the rest of the session.
#
#   . scripts/guard.sh
#
# WHY A WRAPPER AND NOT A NOTE. AGENTS.md rule 16 and
# memory/destructive_probe_loses_uncommitted_work.md both say "commit before any
# destructive git". They demonstrably did not change behaviour: in the session that
# wrote the note, `git checkout -- scripts/` destroyed ~40 minutes of
# uncommitted work TWICE, and the second time was after the note existed. That is
# the argument for an artifact over a transcript -- the repo's own rule about
# certifications, applied to a habit.
#
# WHAT IT GUARDS. These are the commands whose success is indistinguishable from
# their danger when the tree is dirty:
#     git reset --hard [<ref>]      discards staged AND unstaged work
#     git checkout -- <paths...>    overwrites working-tree changes with HEAD
#     git checkout <branch>         discards local edits while switching
#     git clean -f [-d]             deletes untracked files outright
#     git stash drop                discards a stash entry permanently
#     git stash clear               discards ALL stash entries
#
# TWO DESIGN DECISIONS, both measured rather than assumed:
#
#  1. IT REFUSES, IT DOES NOT ASK. A prompt is a decision point, and this session
#     made the wrong decision at one twice. Defaulting to refusal means the safe
#     outcome does not require remembering anything. If the work really is
#     disposable, commit or `git stash push` and then run the command -- both are
#     non-destructive, so the correct action is always available.
#
#  2. THE GUARD IS ABOUT DIRTYNESS, NOT PERMISSION. A dirty tree is not
#     forbidden work -- it is unbacked work, and `git commit` or `git stash push`
#     makes it safe in one step. The refusal names that step, so the message is a
#     route forward rather than a wall.
#
# KNOWN LIMIT, STATED BECAUSE IT IS THE POINT OF DOCUMENTING LIMITS. This only
# affects shells that source it. A `git reset --hard` typed in a shell that did
# not source this file is unaffected, and so is anything run by a tool that
# bypasses the function. It is a seatbelt, not an interlock: it helps exactly when
# someone remembered to put it on, and the rule in AGENTS.md remains the backstop
# for when they did not.
#
# To turn it off for one command:  command git reset --hard HEAD~1
# To turn it off for the session:  unset -f git 2>/dev/null
#
# Env:
#   INTFER_ALLOW_DESTRUCTIVE=1   bypass the guard entirely (for a sandbox repo
#                                 the mutation matrices create, and for the
#                                 matrices' own throwaway clones)

_git_guard_porcelain() {
  # `git <cmd> --porcelain` is the scriptable form: a pager or colour would make
  # "is the tree dirty" a question about the terminal.
  git --no-pager "$@" 2>/dev/null
}

git() {
  local cmd="" dirty="" what="" needs_args=0
  for a in "$@"; do
    case "$a" in
      -*) continue ;;
      *) cmd="$a"; break ;;
    esac
  done

  case "$cmd" in
    reset)
      for a in "$@"; do [ "$a" = "--hard" ] && needs_args=1; done
      what="reset --hard discards staged and unstaged work"
      ;;
    clean)
      for a in "$@"; do case "$a" in -[a-zA-Z]*f*) needs_args=1 ;; esac; done
      what="git clean -f deletes untracked files permanently"
      ;;
    checkout)
      # `git checkout -- <paths>` and `git checkout <branch>` are both destructive
      # to local edits, and they are distinguished by whether a ref follows --.
      for a in "$@"; do
        if [ "$a" = "--" ]; then needs_args=1; break; fi
      done
      if [ "$needs_args" = 0 ]; then
        # branch switch: only the flags form is unambiguous, and a bare
        # `git checkout <thing>` is a branch switch in every real use here.
        needs_args=1
        what="git checkout <branch> discards local edits while switching"
      else
        what="git checkout -- overwrites working-tree changes with HEAD"
      fi
      ;;
    stash)
      for a in "$@"; do
        case "$a" in
          drop|clear) needs_args=1; what="git stash $a discards stashed work permanently" ;;
        esac
      done
      ;;
    *) command git "$@"; return $? ;;
  esac

  [ "$needs_args" = 1 ] || { command git "$@"; return $?; }

  if [ "${INTFER_ALLOW_DESTRUCTIVE:-0}" = "1" ]; then
    command git "$@"; return $?
  fi

  dirty="$(_git_guard_porcelain status --porcelain)"
  if [ -z "$dirty" ]; then
    command git "$@"; return $?
  fi

  local n; n="$(printf '%s\n' "$dirty" | grep -c . || true)"
  cat >&2 <<EOF

  REFUSED: $what, and this tree has $n uncommitted change(s).

  \$ git status --short

  This is the case that destroyed a session's work twice in one day, so it
  refuses rather than prompts -- a prompt is a decision point, and the wrong
  decision was made at one twice.

  Make the work safe first (both are non-destructive):

      git add -A && git commit -m "wip: before <what>"     # then re-run
      git stash push -m "<what>"                          # then re-run

  If the change really is disposable, say so explicitly:

      INTFER_ALLOW_DESTRUCTIVE=1 git <...>

  NOTE: this guard only covers shells that sourced scripts/guard.sh.

EOF
  return 1
}

cat >&2 <<'EOF'
git guard active: reset --hard / checkout / clean -f / stash drop|clear will
REFUSE on a dirty tree. Non-destructive git is unaffected. To inspect or remove:
    unset -f git
EOF
