#!/usr/bin/env bash
# A script must not run `grep -q` as the RIGHT-HAND side of a pipe when the
# script sets `set -o pipefail`. ROOT CAUSE, MEASURED on this box 2026-10-15:
# gate_inventory.sh:305 piped a ~37 KB variable into `grep -qE` and failed 16
# times in 60 identical runs with real content. `grep -q` exits at the FIRST
# match; the writer (printf) is still writing -- glibc stdio chunks writes at
# ~4-8 KB, so any variable larger than one chunk is several write() calls --
# and when the match lands in the first chunk the read end closes and the
# writer's NEXT write() gets SIGPIPE (exit 141). With pipefail that converts a
# SUCCESSFUL match into pipeline failure: the grep found what it was looking
# for and the script took the failure branch anyway. bash -x masked it by
# slowing every command; nothing in the repo had changed between the passing
# and failing runs.
#
# THE RULE, and why it is scoped the way it is:
#   * pipefail is REQUIRED for the defect. Without it the pipeline's exit
#     status is the last command (grep), so a SIGPIPE'd writer is silent and
#     the grep result is still correct. A script that never sets pipefail is
#     genuinely fine -- flagging it would be a false positive.
#   * a NON-quiet grep on a pipe is fine: grep without -q reads ALL input, so
#     there is no early exit and no SIGPIPE on the writer. This is the
#     converse the mutation matrix must hold (state P-E).
#   * a FILE-BACKED `grep -q file` is fine: the file is already closed; grep
#     reads at its own pace. This is the converse (state P-F) and the
#     sanctioned remedy (gate_inventory.sh now greps $codefile).
#   * the long spelling `--quiet` is the same flag and the same defect
#     (state P-D); so is a bundled short flag like `-qE` (state P-C).
#
# KNOWN SCOPE LIMIT, stated rather than hidden: a quiet grep in an EARLIER
# segment of a multi-segment pipeline (grep as a WRITER: `grep -q x | cat`)
# is not flagged -- the rule grades the last segment only, which is where the
# defect lived. No script in the corpus has that shape; if one appears, widen
# the rule here and re-run the matrix.
#
# SCOPE, and its cost, stated rather than implied. check_*/mutate_* are
# excluded as VERIFIERS by name prefix, the same exclusion check_dri_probe.sh
# makes: a harness that PLANTS the defect has to be able to name it, and
# mutate_gate_inventory.sh legitimately carries `printf | grep -qE` as fixture
# text inside run_state heredocs. The cost is real and named: a pipefail+grep-q
# defect hidden inside a check_*/mutate_* would be invisible here. Trigger for
# revisiting: the day a check_*/mutate_* grows a REAL pipe into a quiet grep
# rather than fixture text.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Seam, for the same reason INTFER_SCRIPTS_DIR exists in check_dri_probe.sh:
# the subject of this checker is OTHER scripts, so a mutation harness must be
# able to hand it a corpus of its own. Without it the only way to test the rule
# is to damage the real scripts/ directory.
SCRIPTS_DIR="${INTFER_SCRIPTS_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# seg_is_quiet_grep <segment>: 0 = the segment invokes grep with a quiet flag,
#                            1 = otherwise. read -ra word-splits WITHOUT glob
#                            expansion, so a pattern like `'*'` cannot widen
#                            into a filesystem match.
seg_is_quiet_grep() {
  local seg="$1" rest
  local -a toks
  [[ $seg =~ ^[[:space:]]*([^[:space:]]*/)?grep[[:space:]] ]] || return 1
  rest="${seg#*grep}"
  read -ra toks <<<"$rest"
  local w
  for w in "${toks[@]}"; do
    case "$w" in
      --quiet) return 0 ;;
      -*) case "$w" in *q*) return 0 ;; esac ;;
    esac
  done
  return 1
}

# classify <file>: 0 = clean, 1 = VIOLATION (quiet grep on a pipe under
#                pipefail), 2 = not applicable (verifier prefix, no pipefail,
#                or self-excluded by identity at the call site).
# SELF-CHECK drives this exact function on planted files, so what is proven
# able to fail is what judges the tree -- not a re-run of the whole script,
# which would recurse into its own self-check.
# CONTINUATION LINES are tracked: `foo | \` with `grep -q x` on the next line
# is the same defect on two source lines, and a line-by-line scan that ignores
# the trailing backslash misses it.
classify() {
  local f="$1" code line seg prev_continues=0 continues
  case "$(basename "$f")" in check_*|mutate_*) return 2 ;; esac
  code="$(sed 's/^[[:space:]]*#.*$//' "$f")"
  # Does the script set pipefail at all? bash regex, no pipe, no grep.
  [[ $code =~ (^|[^[:alnum:]_])set[[:space:]]+.*pipefail ]] || return 2
  while IFS= read -r line; do
    continues=0
    case "$line" in *'\') continues=1 ;; esac
    # A continuation line with no pipe of its own may still be the reading
    # end of the PREVIOUS line's pipeline.
    if [ "$prev_continues" -eq 1 ] && [[ $line != *'|'* ]]; then
      seg_is_quiet_grep "$line" && return 1
      prev_continues="$continues"
      continue
    fi
    prev_continues="$continues"
    [[ $line == *'|'* ]] || continue
    seg="${line##*|}"            # last pipeline segment = the reading end
    seg_is_quiet_grep "$seg" && return 1
  done <<< "$code"
  return 0
}

mapfile -t SCRIPTS < <(find "$SCRIPTS_DIR" -maxdepth 2 -not -path "*/archive/*" -name '*.sh' -type f | LC_ALL=C sort)

# *** SELF-EXCLUSION BY IDENTITY, NOT BY NAME. *** This file must be able to
# plant the defect text in its self-check, so including itself is a guaranteed
# false positive. Comparing to $0 is the same exclusion that cannot drift onto
# a different file.
self_is() { [ "$1" = "$0" ]; }

# *** BEHAVIOURAL SELF-CHECK FIRST, and it GATES the run. *** A rule that
# cannot fail proves nothing; one that fires on everything is the mirror
# defect. Five planted cases cover both polarities of each clause of the rule.
TMP="$(mktemp -d "${TMPDIR:-/dev/shm}/pipefail_grep_q.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
trap 'rm -rf "$TMP"; exit 130' INT
trap 'rm -rf "$TMP"; exit 143' TERM
self_bad=0

printf 'set -o pipefail\nprintf %%s x | grep -q y\n' >"$TMP/viol.sh"
printf 'set -o pipefail\ngrep -q y "$1"\n' >"$TMP/filebacked.sh"
printf 'set -o pipefail\necho x | grep y\n' >"$TMP/nonquiet.sh"
printf 'echo x | grep -q y\n' >"$TMP/nopipefail.sh"
printf 'set -o pipefail\necho hello\n' >"$TMP/nogrep.sh"

classify "$TMP/viol.sh";      r=$?
classify "$TMP/filebacked.sh"; r2=$?
classify "$TMP/nonquiet.sh";  r3=$?
classify "$TMP/nopipefail.sh"; r4=$?
classify "$TMP/nogrep.sh";    r5=$?
[ "$r"  -eq 1 ] || { echo "SELF-CHECK FAIL: pipe into grep -q under pipefail not REJECTED (rc=$r)"; self_bad=$((self_bad+1)); }
[ "$r2" -eq 0 ] || { echo "SELF-CHECK FAIL: file-backed grep -q not ACCEPTED (rc=$r2)"; self_bad=$((self_bad+1)); }
[ "$r3" -eq 0 ] || { echo "SELF-CHECK FAIL: non-quiet grep on a pipe not ACCEPTED (rc=$r3)"; self_bad=$((self_bad+1)); }
# rc=0 (clean) and rc=2 (not applicable -- the script never sets pipefail,
# so the rule has nothing to bite on) are BOTH "not flagged as a violation".
# The predicate is -ne 1, not -eq 0: a script without pipefail is classified
# not-applicable by the contract at line 71-73, and demanding exactly 0 here
# would fail on a correct classifier. MEASURED on the first run of this checker.
[ "$r4" -ne 1 ] || { echo "SELF-CHECK FAIL: grep -q on a pipe WITHOUT pipefail was FLAGGED (rc=$r4)"; self_bad=$((self_bad+1)); }
[ "$r5" -eq 0 ] || { echo "SELF-CHECK FAIL: pipefail with no grep not ACCEPTED (rc=$r5)"; self_bad=$((self_bad+1)); }
# The bundled-flag and long spellings are the same defect; prove both.
printf 'set -o pipefail\necho x | grep -qE y\n' >"$TMP/bundled.sh"
printf 'set -o pipefail\necho x | grep --quiet y\n' >"$TMP/longform.sh"
classify "$TMP/bundled.sh";  r6=$?
classify "$TMP/longform.sh"; r7=$?
[ "$r6" -eq 1 ] || { echo "SELF-CHECK FAIL: grep -qE (bundled) not REJECTED (rc=$r6)"; self_bad=$((self_bad+1)); }
[ "$r7" -eq 1 ] || { echo "SELF-CHECK FAIL: grep --quiet (long form) not REJECTED (rc=$r7)"; self_bad=$((self_bad+1)); }
# And the verifier exclusion must be real, not decorative: the same violation is
# clean when planted under a check_* name, so the prefix cannot rot unnoticed.
cp "$TMP/viol.sh" "$TMP/check_planted.sh"
classify "$TMP/check_planted.sh"; r8=$?
[ "$r8" -eq 2 ] || { echo "SELF-CHECK FAIL: check_* verifier exclusion is not in force (rc=$r8)"; self_bad=$((self_bad+1)); }

echo "== PIPEFAIL + QUIET-GREP CHECK (no grep -q as the reading end of a pipe under pipefail) =="
if [ "$self_bad" -eq 0 ]; then
  echo "  ok    self-check: pipe+grep-q REJECTED; file-backed ACCEPTED; non-quiet"
  echo "        ACCEPTED; no-pipefail ACCEPTED; no-grep ACCEPTED; -qE and --quiet"
  echo "        REJECTED; check_* verifier exclusion in force"
else
  echo
  echo "PIPEFAIL GREP-Q: FAIL (self-check did not behave; the rule is not trustworthy)"
  exit 1
fi

viol=0; sites=0; corpus=0; clean=0
for f in "${SCRIPTS[@]}"; do
  self_is "$f" && continue
  classify "$f"; r=$?
  [ "$r" -eq 2 ] && continue
  corpus=$((corpus + 1))
  b="$(basename "$f")"
  if [ "$r" -eq 0 ]; then
    printf '  ok    %-30s pipefail, no quiet grep on a pipe\n' "$b"
    clean=$((clean + 1))
  else
    printf '  FAIL  %-30s pipefail + quiet grep as the reading end of a pipe\n' "$b"
    viol=$((viol + 1)); sites=$((sites + 1))
  fi
done

# *** POPULATION GUARD. *** Assert the SUBJECT still exists in the corpus -- a
# script that sets pipefail -- not that a violation exists. A healthy tree must
# be allowed to have none: demanding sites > 0 is evidence of a violation from
# a healthy tree (the audit #20 shape, inverted), and it fails the moment the
# defect is fixed.
if [ "$corpus" -eq 0 ]; then
  echo "FAIL: no script in scripts/ sets pipefail, so a green verdict would" >&2
  echo "      certify a population it never harvested." >&2
  exit 1
fi

echo "  ..    $corpus script(s) set pipefail; $clean clean"
echo
if [ "$viol" -eq 0 ]; then
  echo "PIPEFAIL GREP-Q: PASS (no quiet grep as the reading end of a pipe)"
  echo "RESULT: every pipefail script either greps a file, greps non-quiet, or"
  echo "        does not pipe into grep at all (see the SIGPIPE root cause above)"
  exit 0
fi
echo "PIPEFAIL GREP-Q: FAIL ($viol of $corpus pipefail script(s) pipe into a quiet grep)"
echo "RESULT: grep -q exits at the first match; the writer's next write() gets"
echo "        SIGPIPE and pipefail turns the match into a failure"
exit 1
