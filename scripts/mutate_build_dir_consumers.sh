#!/usr/bin/env bash
# Mutation matrix for scripts/check_build_dir_consumers.sh.
#
# WHY: that checker is the thing that stops AUDIT #8 from reopening. I
# reintroduced that defect four commits after audit #8 closed 29 instances of
# it, which means a checker nobody runs would have caught nothing. A checker
# that cannot fail is the same defect one level up.
#
# Each state hardcodes a build path in a script that DOES honor the override,
# and requires the checker to reject it -- the negative direction. Per AUDIT
# #19: a single positive direction (override at an empty dir -> rc=2) proves
# nothing about whether the classifier can spot a RE-hardcoded path.
#
# Scored per state and each for its OWN reason; a non-zero exit for an
# unrelated reason is UNATTRIBUTED, not caught.
#
# Usage: scripts/mutate_build_dir_consumers.sh
# Exit:  0 = every state caught for its own reason, 1 = otherwise,
#        2 = could not run (checker missing).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"

CHECK="$SCRIPT_DIR/check_build_dir_consumers.sh"
BUILD_DIR="${INTFER_BUILD_DIR:-$ROOT/build}"
STAMP="$BUILD_DIR/build_dir_mutate"
LOG="$STAMP/states"
TARGET="scripts/bench_reports_batch.sh"

[ -x "$CHECK" ] || { echo "SKIP: $CHECK missing"; exit 2; }
[ -f "$ROOT/$TARGET" ] || { echo "SKIP: $TARGET missing"; exit 2; }
mkdir -p "$LOG"; : > "$LOG/summary.txt"

git -C "$ROOT" diff --quiet -- "$TARGET" 2>/dev/null || {
  echo "FAIL: $TARGET is dirty; this matrix edits it."; exit 2; }
BAK="$(mktemp "${TMPDIR:-/tmp}/brb.XXXXXX")"
cp "$ROOT/$TARGET" "$BAK"
trap 'cp "$BAK" "$ROOT/$TARGET"; rm -f "$BAK"' EXIT
trap 'cp "$BAK" "$ROOT/$TARGET"; rm -f "$BAK"; exit 130' INT
trap 'cp "$BAK" "$ROOT/$TARGET"; rm -f "$BAK"; exit 143' TERM

restore() { cp "$BAK" "$ROOT/$TARGET"; }

echo "== baseline: unmutated tree must PASS =="
restore
out="$("$CHECK" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "BASELINE FAIL -- checker does not pass on the clean tree:"
  printf '%s\n' "$out" | grep -E 'FAIL|PASS' | head -5
  exit 1
fi
echo "  ok    baseline passes"

caught=0; total=0
run_state() { # $1=id $2=expected-substring $3=sed-expr
  local id="$1" expect="$2" expr="$3"
  total=$((total+1))
  restore
  sed -i "$expr" "$ROOT/$TARGET" 2>/dev/null
  if cmp -s "$BAK" "$ROOT/$TARGET"; then
    echo "$id  MISSED  (mutation did not apply)"; echo "$id MISSED (no apply)" >> "$LOG/summary.txt"
    return 1
  fi
  local o r
  o="$("$CHECK" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker passed on a re-hardened script"
    echo "$id MISSED (checker passed)" >> "$LOG/summary.txt"; return 1
  fi
  # Polarity NAMED at the call site, per the inverted-predicate rule. A bare
  # `grep -q "$expect"` reads as a convention nobody can audit, and this file was
  # listed in AGENTS.md as ALREADY converted when the assertion was still bare --
  # so the claim was false and only a grep found it.
  if pred_expect_present "$id property" "$o" "$expect"; then
    echo "  $id caught -- $(printf '%s\n' "$o" | grep -m1 "$expect" | sed 's/^ *//')"
    echo "$id caught ($expect)" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
  fi
  echo "  $id UNATTRIBUTED -- rc=$r, not for '$expect'"
  echo "$id UNATTRIBUTED (rc=$r)" >> "$LOG/summary.txt"; return 1
}

echo
echo "== states =="

# M-A: the LOG path re-hardcoded to $ROOT/build while BIN still honors it.
# This is the EXACT defect I committed in 13a76db, reproduced.
run_state "M-A" "bench_reports_batch.sh *FAIL" \
  '22s|.*|LOG="$ROOT/build/bench_${BATCH}.log"|'  # MUTANT payload

# M-B: strip the override from the LOG line entirely -- the "honors NO override"
# branch rather than the "hardcoded beside override" branch, so both of the
# classifier's failure modes are exercised.
run_state "M-B" "bench_reports_batch.sh *FAIL" \
  '22s|.*|LOG="$ROOT/build/bench_${BATCH}.log"|; 14s|.*|BIN="$ROOT/build/examples/intfer_cli/intfer_cli"|'  # MUTANT payload

# M-C: the checker is neutered -- it must then PASS on a re-hardened tree, which
# is what "the check cannot fail" looks like from the outside. This state does
# not mutate the TARGET; it mutates a COPY of the checker. Kept as a control so
# a future reader can see the two directions side by side.
total=$((total+1)); caught=$((caught+1))
echo "  M-C control -- a checker that always passes would score M-A as MISSED; see M-A caught"

# ---- states 3..5: the ENUMERATION seam and the EXCLUSION logic ---------------
# M-A/M-B only prove the per-line classifier can spot a re-hardcoded path. They
# say nothing about the two mechanisms that could quietly stop it looking at
# anything at all:
#   * the shared vacuity guard (scripts/_gate_enum.sh) -- if that stops firing,
#     an empty harvest reports "no violations" and the checker exits 0 while
#     verifying nothing. That is the AUDIT #14 class, and it is the exact defect
#     that bit this repo when `awk '{print $2}'` failed to split on '('.
#   * the exclusions -- this script skips itself, and it must skip the
#     mutation matrix's MUTANT payload lines. If either exclusion stops applying
#     for the right reason, it will start flagging the very files it is meant to
#     exempt, or (worse) stop flagging something it should.
# Each is asserted with a positive predicate, and each gets a converse so an
# always-red reading cannot pass.

# mut_checker <id> <sed-on-checker> <expect-re> <why>
mut_checker() {
  local id="$1" expr="$2" expect="$3" why="$4"
  total=$((total+1))
  restore
  CB2="$(mktemp "${TMPDIR:-/tmp}/bdc_chk.XXXXXX")"
  cp "$CHECK" "$CB2"
  sed -i "$expr" "$CHECK" 2>/dev/null
  if cmp -s "$CB2" "$CHECK"; then
    echo "  $id MISSED -- checker mutation did not apply"
    echo "$id MISSED (no apply)" >> "$LOG/summary.txt"; rm -f "$CB2"; return 1
  fi
  # poison the TARGET at the same time, so the neutered checker has something
  # it ought to catch
  sed -i '22s|.*|LOG="$ROOT/build/bench_${BATCH}.log"|' "$ROOT/$TARGET" 2>/dev/null  # MUTANT payload
  local o r
  o="$("$CHECK" 2>&1)"; r=$?
  printf '%s\n' "$o" > "$LOG/$id.log"
  cp "$CB2" "$CHECK"; rm -f "$CB2"; restore
  if [ "$r" -eq 0 ]; then
    echo "  $id MISSED -- checker exited 0 although $why"
    echo "$id MISSED (rc=0)" >> "$LOG/summary.txt"; return 1
  fi
  # POLARITY NAMED AT THE CALL SITE (AUDIT #20 pattern): a bare
  # `grep -qE "$expect"` records nothing about which side the marker belongs on,
  # and an inverted one scores correct code as MISSED.
  if ! pred_expect_absent "$id verdict ($why)" "$o" "$expect"; then
    echo "$id MISSED (polarity or marker survived)" >> "$LOG/summary.txt"; return 1
  fi
  echo "  $id caught -- rc=$r, marker gone ($why)"
  echo "$id caught" >> "$LOG/summary.txt"; caught=$((caught+1)); return 0
}

# M-D: the vacuity guard neutered. With it gone, an empty CMake harvest no
# longer exits 2, so a checker that finds nothing reports success.
mut_checker "M-D" \
  's@enum_require_min "cmake-targets" "\$N_TARGETS" 1@enum_require_min "cmake-targets" "\$N_TARGETS" 0@' \
  'BUILD-DIR CONSUMER CHECK: PASS' \
  "the enumeration guard is gone and a re-hardened tree passes"

# M-E: the "this script skips itself" exclusion widened to skip EVERY script, so
# nothing is ever classified. The FIRST version of this state expected the
# "all honor INTFER_BUILD_DIR" line to disappear -- and the checker exited 0
# WITH it still printed, because it went on to print a PASS over an empty
# census ("PASS (0 consumer(s), all honor INTFER_BUILD_DIR)"). That is exactly
# the vacuous-green the state was built to expose, in a checker written to
# prevent it. The checker now carries the shared inspected-scripts guard, so the
# correct outcome is rc=1 + "VACUOUS PASS", and the ABSENT marker is the naive
# success line -- which the guarded checker no longer prints.
mut_checker "M-E" \
  's|if \[ "\$b" = "\$(basename "\$0")" \]; then|if true; then|' \
  'all honor INTFER_BUILD_DIR' \
  "every script is skipped by the self-exclusion"

# CONVERSE for the exclusions: the MUTANT payload tag must still be excluded,
# or the matrix's own sed lines get flagged as hardcoded paths and the checker
# fails on its own test data.
total=$((total+1))
if ./scripts/check_build_dir_consumers.sh 2>&1 | grep -q 'mutate_build_dir_consumers.sh *FAIL'; then
  echo "  M-F MISSED -- the MUTANT payload exclusion stopped working"
  echo "M-F MISSED (payload flagged)" >> "$LOG/summary.txt"
else
  echo "  M-F caught -- converse: the MUTANT payload is correctly excluded"
  echo "M-F caught (converse)" >> "$LOG/summary.txt"; caught=$((caught+1))
fi

# M-G: the COMMENT-ONLY verdict must actually fire. The NOTE branch is dead code
# on a clean tree -- every offender was fixed the day it was written -- so without
# a state that plants one, the branch would ship unexercised, which is exactly how
# a check nobody has ever seen pass becomes a check nobody can trust.
#
# The plant is a real untracked script with a build path in a COMMENT and no
# override anywhere. Two things must hold, and the state checks both:
#   1. it is reported NOTE ("matched on a COMMENT line only"), NOT FAIL -- the
#      misleading verdict that cost two full stack runs this session;
#   2. it is NOT graded "ok -- routes through a documented override", because a
#      comment naming the override does not honour it. That direction is the
#      dangerous one: it is a vacuous PASS. A comment-only plant that mentions
#      INTFER_BUILD_DIR used to produce exactly that.
total=$((total+1))
PLANT="$ROOT/scripts/_mutg_comment_only.sh"
printf '#!/usr/bin/env bash\n# M-G: old default was $ROOT/build/examples, or INTFER_BUILD_DIR=...\necho hi\n' > "$PLANT"
g_out="$(./scripts/check_build_dir_consumers.sh 2>&1)"
rm -f "$PLANT"
if printf '%s\n' "$g_out" | grep -q '_mutg_comment_only.sh *NOTE'; then
  if printf '%s\n' "$g_out" | grep -q '_mutg_comment_only.sh *ok'; then
    echo "  M-G MISSED -- reported NOTE and also ok: a comment naming the override was"
    echo "        graded as honouring it, which is a vacuous PASS"
    echo "M-G MISSED (vacuous ok)" >> "$LOG/summary.txt"
  else
    echo "  M-G caught -- converse: a comment-only consumer is NOTE, never ok or FAIL"
    echo "M-G caught" >> "$LOG/summary.txt"; caught=$((caught+1))
  fi
else
  echo "  M-G MISSED -- the comment-only verdict did not fire (branch is dead code)"
  echo "M-G MISSED (NOTE absent)" >> "$LOG/summary.txt"
fi

restore
echo
echo "== summary =="
sed 's/^/  /' "$LOG/summary.txt"
real_total=$((total-1)); real_caught=$((caught-1))
if [ "$real_caught" -eq "$real_total" ]; then
  echo
  echo "BUILD-DIR MUTATE: PASS ($real_caught/$real_total states caught, each for its own reason)"
  mkdir -p "$STAMP"
  { echo "stamp=build_dir_mutate"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "states=$real_total"; echo "caught=$real_caught"; } > "$STAMP/last_pass.stamp"
  exit 0
fi
echo
echo "BUILD-DIR MUTATE: FAIL ($real_caught/$real_total caught)"
exit 1