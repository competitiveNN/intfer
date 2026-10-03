#!/usr/bin/env bash
# POST-STACK: run this AFTER scripts/run_all_checks.sh, NEVER as one of its steps.
# It grades "the certificate for the CURRENT commit", and a step of the stack runs
# BEFORE that stack writes its certificate -- so as a step it could only ever report
# STALE, and the stack would never be able to pass. Measured: wired in as a step, it
# failed its own run with rc=1 (cert names the previous commit) and turned a PASS
# stack into FAIL. Section 0c reads this declaration and exempts the file from the
# STEPS requirement, printing the exemption so it is visible rather than silent.
#
# Assert the STACK certification on disk is still evidence for THIS tree.
#
# WHY THIS EXISTS. `scripts/run_all_checks.sh` writes build/stack_cert/cert naming
# the commit it certified, with its own rc (AUDIT #29 added the file because the
# only artifact of a PASS run was a log line, and `grep -c` returns a COUNT rather
# than an exit code). But a certificate nobody re-derives is a transcript with a
# filename: every session since hand-ran the same six probes -- cert sha vs HEAD,
# rc, stale/unproven, tree clean, lock absent, census -- before it may say "the
# stack passes". *** That is AUDIT #22's shape applied to the harness that
# certifies everything else: a claim about the tree instead of a check of it. ***
# Three sessions in a row re-derived it by hand, and one of them mis-quoted a
# mid-sequence failure as the final state.
#
# So this mechanises the re-derivation. Sibling of check_engine_cert.sh, which
# certifies the ten_gates artifact; the difference is deliberate and IS the design:
#
#   * check_engine_cert.sh has a "CARRIES" case -- a tooling-only commit must not
#     invalidate a 45-minute engine certification, or the stack becomes unusable
#     and people learn to ignore it.
#   * THIS has NO carrying case, because the stack's own rule is that every stamp
#     must name HEAD. ANY commit makes the evidence older than the tree, so the
#     only honest answers are CERTIFIED (rc=0) or RE-RUN (rc=1). Weakening this to
#     "carries" would reintroduce exactly the gap it closes.
#
#   cert absent                   -> 2 UNPROVEN (nothing claimed, nothing stale)
#   cert sha != HEAD, or rc != 0  -> 1 STALE   (re-run the stack; not a regression)
#   cert sha == HEAD and rc == 0  -> 0 CERTIFIED, plus the census cross-checks
#
# THE CENSUS CROSS-CHECKS are the part that keeps paying. The cert carries derived
# counts (scripts / checks / matrices / stamps / stamp_dirs) precisely because
# quoting them in prose goes stale, and in this session a summary said "14
# checkers, 18 matrices" while 0b derived 31 (=14+17) and step 3/3 counted 18 stamp
# FILES across 17 DIRECTORIES. Each number is re-derived from the filesystem and
# compared, so a stale or hand-edited cert cannot be read as a current one. It pays
# on its first run: adding this file made the tree 32/15 while the cert said 31/14,
# and the census caught it immediately.
#
# *** SELF-CHECK FIRST AND GATING, and it needs a recursion guard. *** The
# self-check drives THIS script against synthetic certs, so without a guard it
# drives itself, which drives itself -- the first version had no guard and the call
# TIMED OUT rather than failing, which is the worst of both. Nested runs pass
# INTFER_STACK_CERT_NO_SELFCHECK=1 and SAY SO, so a nested run's silence can never
# be read as "the self-check passed". Same seam rationale as INTFER_SYCL_LS and
# INTFER_BUILD_DIR: an assumption the harness cannot redirect is one it cannot test.
#
# Usage: scripts/check_stack_cert.sh
# Env:   INTFER_BUILD_DIR                    where the cert lives (the seam that
#                                            lets the matrix redirect it -- without
#                                            it, the only way to test this is to
#                                            mutate the REAL certification).
#         INTFER_STACK_CERT_NO_SELFCHECK=1   skip the self-tests (nested runs)
#         INTFER_STACK_CERT_ROOT             the CHECKOUT being graded (a sandbox
#                                            repo, so the tree-derived branches --
#                                            the census and the dirty-tree test --
#                                            are testable at all)
# Exit:  0 = certified for this tree, 1 = stale/re-run or a broken checker,
#        2 = unproven (no certificate at all).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# *** A SECOND SEAM, for the repository the verdict is about. *** INTFER_BUILD_DIR
# redirects the ARTIFACT, but this checker grades a TREE: `git rev-parse HEAD`,
# the census and the dirty-tree test all read the checkout, which was hard-wired to
# $ROOT. So the dirty-tree branch was untestable -- a sandbox cannot make the REAL
# tree dirty, and a checker with an untestable branch is a branch nobody can claim.
# INTFER_STACK_CERT_ROOT points it at a sandbox checkout instead. SCRIPT_DIR stays
# real, so a redirected run still executes THIS checker.
REPO="${INTFER_STACK_CERT_ROOT:-$ROOT}"
cd "$REPO" || exit 2
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
enum_reset

BD="${INTFER_BUILD_DIR:-$ROOT/build}"
CERT="$BD/stack_cert/cert"
field() { # $1 file, $2 key
  [ -f "$1" ] || return 0
  sed -n "s/^$2=//p" "$1" | head -1
}

# The census, re-derived ONCE and used by both the live verdict and the fixtures.
# Deriving it separately per use would let a fixture and the check agree by
# construction on the very number that is supposed to be independent of them.
n_scripts=$(find "$REPO/scripts" -maxdepth 1 \( -name 'check_*.sh' -o -name 'mutate_*.sh' \) -type f | wc -l | tr -d ' ')
n_checks=$(find "$REPO/scripts" -maxdepth 1 -name 'check_*.sh' -type f | wc -l | tr -d ' ')
n_matrices=$(find "$REPO/scripts" -maxdepth 1 -name 'mutate_*.sh' -type f | wc -l | tr -d ' ')
n_stamps=$(find "$BD" -name '*last_pass.stamp' -type f 2>/dev/null | wc -l | tr -d ' ')
n_dirs=$(find "$BD" -name '*last_pass.stamp' -type f 2>/dev/null | sed 's|/[^/]*$||' | sort -u | wc -l | tr -d ' ')

bad=0
echo "== STACK CERTIFICATION CHECK =="
echo "   census (re-derived): $n_scripts scripts = $n_checks checkers + $n_matrices matrices;"
echo "                        $n_stamps stamp FILES across $n_dirs directories"

# ============================ SELF-CHECK, FIRST AND GATING ===================
if [ "${INTFER_STACK_CERT_NO_SELFCHECK:-0}" = "1" ]; then
  echo "   self-tests SKIPPED (INTFER_STACK_CERT_NO_SELFCHECK=1) -- grading the"
  echo "   artifact only; THIS RUN CERTIFIES NO NON-VACUITY."
else
SELF="$BD/.stack_cert_selfcheck"
rm -rf "$SELF"; mkdir -p "$SELF/stack_cert"
# *** THE FIXTURE MUST GIVE THE NESTED RUN A CENSUS TO DERIVE. *** The census for
# stamps comes from $BD, and a nested run's $BD is $SELF -- so the first version
# baked the LIVE counts into the fixture, the nested run derived 0 stamps from the
# empty temp tree, and self-check (i) refused a cert that was perfectly good. The
# case was right about the checker and wrong about the fixture, which is the
# mirror of the always-red trap. So the fixture plants REAL stamp files and writes
# THEIR counts, and the nested run re-derives exactly those.
for i in 1 2 3; do mkdir -p "$SELF/m$i"; echo "sha=deadbeef" > "$SELF/m$i/last_pass.stamp"; done
n_self_stamps=3
n_self_dirs=3
# Only the field under test differs between fixtures; the census is copied from
# the LIVE counts, so no case can pass by agreeing with itself.
mk() { # $1 sha $2 rc $3 verdict $4 stale $5 unproven $6 failed
  { echo "stamp=stack_cert"; echo "sha=$1"; echo "short=${1:0:7}"; echo "rc=$2"
    echo "verdict=$3"; echo "failed=$6"; echo "stale=$4"; echo "unproven=$5"
    echo "stamps=$n_self_stamps"; echo "scripts=$n_scripts"; echo "checks=$n_checks"
    echo "matrices=$n_matrices"; echo "stamp_dirs=$n_self_dirs"
    echo "quick=0"; echo "date=1970-01-01T00:00:00Z"; } > "$SELF/stack_cert/cert"
}
sc_run() { INTFER_BUILD_DIR="$SELF" INTFER_STACK_CERT_NO_SELFCHECK=1 \
             bash "$SCRIPT_DIR/check_stack_cert.sh" >"$SELF/out" 2>&1; echo $?; }
sc_bad=0
HEAD_SHA_NOW="$(git rev-parse HEAD)"

# (i) POSITIVE, and required: without a case the checker ACCEPTS, every firing
# case below is satisfied by a checker that refuses everything.
mk "$HEAD_SHA_NOW" 0 PASS 0 0 0
rc_i="$(sc_run)"
if [ "$rc_i" = "0" ]; then
  echo "  ok    self-check (i): a matching rc=0 cert is ACCEPTED"
else
  echo "  FAIL  self-check (i): a matching rc=0 cert was REFUSED (rc=$rc_i)"
  sed 's/^/          /' "$SELF/out"; sc_bad=$((sc_bad + 1))
fi
# (ii) a cert naming another commit -> STALE, rc=1
mk "dead0000000000000000000000000000000000000" 0 PASS 0 0 0
rc_ii="$(sc_run)"
if [ "$rc_ii" = "1" ] && grep -q 'STALE' "$SELF/out"; then
  echo "  ok    self-check (ii): a cert naming another commit is STALE (rc=1)"
else
  echo "  FAIL  self-check (ii): a foreign sha was not reported stale (rc=$rc_ii)"
  sc_bad=$((sc_bad + 1))
fi
# (iii) right sha, rc=2 -> NOT a pass. The case that matters: the file EXISTS and
# names HEAD, and still certifies nothing.
mk "$HEAD_SHA_NOW" 2 UNPROVEN 1 0 0
rc_iii="$(sc_run)"
if [ "$rc_iii" = "1" ] && grep -q 'NOT A PASS' "$SELF/out"; then
  echo "  ok    self-check (iii): a cert naming HEAD with rc=2 is NOT a pass (rc=1)"
else
  echo "  FAIL  self-check (iii): an rc=2 cert was accepted (rc=$rc_iii)"
  sc_bad=$((sc_bad + 1))
fi
# (iv) no cert at all -> 2, never 0
rm -f "$SELF/stack_cert/cert"
rc_iv="$(sc_run)"
if [ "$rc_iv" = "2" ]; then
  echo "  ok    self-check (iv): no cert at all is UNPROVEN (rc=2), not a pass"
else
  echo "  FAIL  self-check (iv): a missing cert gave rc=$rc_iv; it must be 2"
  sc_bad=$((sc_bad + 1))
fi
# (v) a census the tree does not match -> 1. Without this the census cross-check
# is decorative and "the census is stale" is a line nothing tests.
mk "$HEAD_SHA_NOW" 0 PASS 0 0 0
sed -i "s/^checks=.*/checks=999/" "$SELF/stack_cert/cert"
rc_v="$(sc_run)"
if [ "$rc_v" = "1" ] && grep -q 'the census is stale' "$SELF/out"; then
  echo "  ok    self-check (v): a census the tree does not match is refused (rc=1)"
else
  echo "  FAIL  self-check (v): a mismatched census was accepted (rc=$rc_v)"
  sc_bad=$((sc_bad + 1))
fi
# (vi) self-contradiction (rc=0 but stale=1) -> 1: a hand-edited or truncated cert
# must not certify even when every other field agrees.
mk "$HEAD_SHA_NOW" 0 PASS 1 0 0
rc_vi="$(sc_run)"
if [ "$rc_vi" = "1" ] && grep -q 'contradicts itself' "$SELF/out"; then
  echo "  ok    self-check (vi): a cert that contradicts itself is refused (rc=1)"
else
  echo "  FAIL  self-check (vi): a self-contradicting cert was accepted (rc=$rc_vi)"
  sc_bad=$((sc_bad + 1))
fi
# (vii) a cert with NO usable sha -> 1, not a comparison that happens to pass.
# *** A STRING TEST, NOT enum_require_nonempty *** whose test is `[ -gt 0 ]`, i.e.
# NUMERIC: the first version used it here and reported EVERY cert as having no sha,
# including the real one. Mirror of the rule in _gate_enum.sh's own header.
{ echo "stamp=stack_cert"; echo "rc=0"; echo "verdict=PASS"; } > "$SELF/stack_cert/cert"
rc_vii="$(sc_run)"
if [ "$rc_vii" = "1" ] && grep -q 'no usable sha' "$SELF/out"; then
  echo "  ok    self-check (vii): a cert with no sha is refused (rc=1)"
else
  echo "  FAIL  self-check (vii): a sha-less cert was accepted (rc=$rc_vii)"
  sc_bad=$((sc_bad + 1))
fi
rm -rf "$SELF"
if [ "$sc_bad" -ne 0 ]; then
  echo
  echo "STACK CERT CHECK: FAIL ($sc_bad self-check failure(s)) -- the checker cannot"
  echo "  be trusted to grade anything, including its own verdict below."
  exit 1
fi
fi   # end self-check

# ================================ LIVE VERDICT ===============================
echo
if [ ! -f "$CERT" ]; then
  echo "  UNPROVEN  no certification at $CERT -- the stack has not certified this"
  echo "            tree (or its build dir was cleared). Run scripts/run_all_checks.sh."
  echo "RESULT: STACK CERT: UNPROVEN (no artifact)"
  exit 2
fi
echo "   cert: $CERT"
cert_sha="$(field "$CERT" sha)"
cert_rc="$(field "$CERT" rc)"
cert_verdict="$(field "$CERT" verdict)"
HEAD_SHA="$(git rev-parse HEAD)"
HEAD_SHORT="${HEAD_SHA:0:7}"
echo "   cert says: sha=$cert_sha rc=$cert_rc verdict=$cert_verdict"
echo "   HEAD is  : sha=$HEAD_SHA"

if [ -z "$(printf '%s' "$cert_sha" | tr -d '[:space:]')" ]; then
  echo "  FAIL   the certification has no usable sha= line; it cannot be compared to"
  echo "         HEAD. A cert without an identity is not evidence."
  bad=$((bad + 1))
elif [ "$cert_sha" != "$HEAD_SHA" ]; then
  echo "  STALE  the certificate names ${cert_sha:0:7}, HEAD is $HEAD_SHORT"
  echo "         The stack's own rule is that every stamp names HEAD, so ANY commit"
  echo "         makes the evidence older than the tree. There is deliberately NO"
  echo "         'carries' case here (check_engine_cert.sh has one, and must): a"
  echo "         tooling-only commit cannot invalidate an ENGINE certification, but"
  echo "         it does invalidate this one."
  echo "         NOT a regression: re-run scripts/run_all_checks.sh."
  echo "RESULT: STACK CERT: STALE (re-run the stack; nothing is wrong with the code)"
  exit 1
fi

if [ "$cert_rc" != "0" ]; then
  echo "  STALE  the certificate names HEAD but its rc=$cert_rc ($cert_verdict)"
  echo "         A cert is written on EVERY verdict path, so a matching sha alone"
  echo "         certifies nothing. Measured: that is what the previous commit's"
  echo "         first stack run produced."
  echo "RESULT: STACK CERT: NOT A PASS at HEAD (rc=$cert_rc)"
  exit 1
fi

for k in stale unproven failed; do
  v="$(field "$CERT" "$k")"
  if [ "${v:-x}" != "0" ]; then
    echo "  FAIL   cert says $k=$v while rc=0 -- the file contradicts itself."
    echo "         A hand-edited or truncated cert must not certify."
    bad=$((bad + 1))
  fi
done

for pair in "scripts:$n_scripts" "checks:$n_checks" "matrices:$n_matrices" \
            "stamps:$n_stamps" "stamp_dirs:$n_dirs"; do
  k="${pair%%:*}"; live="${pair##*:}"; claimed="$(field "$CERT" "$k")"
  if [ "$claimed" != "$live" ]; then
    printf '  FAIL   cert says %s=%s, the tree has %s -- the census is stale\n' \
      "$k" "${claimed:-<absent>}" "$live"
    bad=$((bad + 1))
  fi
done
[ "$bad" -eq 0 ] && echo "  ok    census re-derived and identical to the cert"

if git diff --quiet && git diff --cached --quiet; then
  echo "  ok    tracked tree clean (a dirty tree is not what the cert measured)"
else
  echo "  FAIL   the tracked tree is DIRTY; the cert describes a different tree"
  git status --porcelain --untracked-files=no | sed 's/^/         /'
  bad=$((bad + 1))
fi
# A live lock means a stack is running RIGHT NOW, so a cert written before it
# finished does not describe it. Reported, not fatal: the run may be one that is
# about to replace this very cert.
if [ -e "$BD/.run_all_checks.lock" ]; then
  lock_owner="$(cat "$BD/.run_all_checks.lock/pid" 2>/dev/null || echo '?')"
  if [ -n "$lock_owner" ] && kill -0 "$lock_owner" 2>/dev/null; then
    echo "  NOTE  a run_all_checks.sh is LIVE (pid $lock_owner) -- any cert written"
    echo "        before it finished does not describe the run in progress."
  else
    echo "  NOTE  a lock is present from a dead pid ($lock_owner); the next run"
    echo "        self-heals it. Harmless for this verdict."
  fi
fi

echo
if [ "$bad" -ne 0 ]; then
  echo "STACK CERT CHECK: FAIL ($bad problem(s))"
  echo "RESULT: the certificate does not describe this tree"
  exit 1
fi
echo "STACK CERT CHECK: PASS (cert $HEAD_SHORT rc=0, census identical, tree clean)"
echo "RESULT: the stack certification on disk is evidence for THIS commit"
echo "RESULT: census: $n_scripts scripts = $n_checks checkers + $n_matrices matrices;"
echo "RESULT: $n_stamps stamp FILES across $n_dirs directories"
exit 0
