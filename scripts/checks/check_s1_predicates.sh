#!/usr/bin/env bash
# check_s1_predicates.sh -- SOURCE-level structural certification of the S1
# uncached-host-USM doorbell probe (examples/uncached_doorbell_probe/main.cpp).
#
# WHY A SOURCE CHECKER AND NOT A GATE. The probe's four STRUCTURAL predicates are
# the reason its timing can be read at all, and every one of them is a claim about
# the source: that the two arms cannot be the same buffer, that the candidate is
# host USM the kernel can address, that every round's token really round-trips,
# and that a single-pass run is refused. A runtime gate would need the GPU and
# would then be certifying the number, which is the operator's decision, not
# this checker's. So this checker grades the GUARDS, at source, with no
# execution -- and the timing stays unmeasured on purpose.
#
# THE EIGHT PREDICATES. Each one refuses the shape of a defect that was actually
# planted and measured (mutate_s1_predicates.sh):
#   p_distinct          the same-buffer guard exists        (ND-distinct)
#   p_host_type         the ZE_MEMORY_TYPE_HOST guard       (ND-hosttype)
#   p_token             the per-round token comparison       (ND-token)
#   p_passes2           the --passes >= 2 refusal exists     (ND-passes2)
#   p_guard_after_parse that refusal sits AFTER the CLI parse(ND-guard)
#   p_report            `RESULT: PASS` is printed            (ND-report)
#   p_no_env            no env var drives the probe          (ND-envvar)
#   p_no_verdict        no adopt/decline verdict is printed  (ND-verdict)
#
# WHY COMMENTS ARE STRIPPED FIRST. p_no_verdict must not fire on the probe's own
# prose: the healthy source says "it prints no adopt/decline verdict" and "say
# whether ... got faster" in COMMENTS, which is the exact claim being asserted.
# MEASURED: an unstripped grep for adopt|faster returns 3 hits on the healthy
# source, so the predicate would refuse a correct file -- the same false-positive
# family check_destructive_git.sh guards against by stripping comments before its
# harvest. A predicate that fires on its own subject has no negative direction.
#
# ANTI-VACUITY. An empty or missing subject is a FAIL, never a pass: "looked at
# nothing" and "found nothing wrong" are different claims.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUBJ="${1:-$ROOT/examples/uncached_doorbell_probe/main.cpp}"

if [ ! -f "$SUBJ" ]; then
  echo "S1-PRED: FAIL -- subject missing ($SUBJ)"
  echo "RESULT: FAIL"
  exit 1
fi
if [ ! -s "$SUBJ" ]; then
  echo "S1-PRED: FAIL -- subject empty ($SUBJ); a subject with nothing in it"
  echo "               cannot hold a predicate and must not read as a pass"
  echo "RESULT: FAIL"
  exit 1
fi

CODE="$(mktemp "${TMPDIR:-/tmp}/s1pred.XXXXXX")"
trap 'rm -f "$CODE"' EXIT
# INT/TERM must ALSO reach the cleanup: an EXIT trap alone never runs on a signal
# (bash fires EXIT only at a reached end or an explicit exit), so a SIGTERM killed
# the script outright and leaked $CODE -- AUDIT #29. The exit routes the signal back
# through the EXIT handler; "clean up and resume" leaves the process alive, which
# is strictly worse (check_trap_release.sh).
trap 'rm -f "$CODE"; exit 130' INT
trap 'rm -f "$CODE"; exit 143' TERM
# Strip // line comments. The probe has no '//' inside a string literal, so this
# cannot remove a real statement.
sed -e 's://.*::' "$SUBJ" > "$CODE"

fails=()
note() { printf 'S1-PRED: %-20s %s\n' "$1" "$2"; }

# --- 1. the two arms cannot be the same buffer -------------------------------
if grep -Fq 'base_buf == (void*)cand_buf' "$CODE"; then
  note p_distinct "ok"
else
  note p_distinct "FAIL -- the same-buffer guard is absent; a probe that times"
  note "" "      one buffer twice is vacuous"
  fails+=(p_distinct)
fi

# --- 2. the candidate allocation must be host USM ---------------------------
if grep -Fq 'cand_type != (int)ZE_MEMORY_TYPE_HOST' "$CODE"; then
  note p_host_type "ok"
else
  note p_host_type "FAIL -- the ZE_MEMORY_TYPE_HOST guard is absent; the"
  note "" "      allocation could be something the SYCL kernel cannot address"
  fails+=(p_host_type)
fi

# --- 3. every round's token must round-trip --------------------------------
if grep -Fq 'buf[i] != (std::uint8_t)(token' "$CODE"; then
  note p_token "ok"
else
  note p_token "FAIL -- the per-round token comparison is absent; a stale read"
  note "" "      (which is the very effect the candidate flag changes) is silent"
  fails+=(p_token)
fi

# --- 4/5. --passes >= 2, and it must be AFTER the CLI parse ------------------
parse_ln="$(grep -n -F 'for (int i = 1; i < argc' "$CODE" | head -1 | cut -d: -f1)"
guard_ln="$(grep -n -F 'if (passes < 2)' "$CODE" | head -1 | cut -d: -f1)"
if [ -n "$guard_ln" ]; then
  note p_passes2 "ok (line $guard_ln)"
else
  note p_passes2 "FAIL -- the --passes >= 2 refusal is absent; a single-round"
  note "" "      delta has no spread floor to clear and must not be reported"
  fails+=(p_passes2)
fi
if [ -n "$guard_ln" ] && [ -n "$parse_ln" ] && [ "$guard_ln" -gt "$parse_ln" ]; then
  note p_guard_after_parse "ok (guard $guard_ln after parse $parse_ln)"
elif [ -n "$guard_ln" ] && [ -n "$parse_ln" ]; then
  note p_guard_after_parse "FAIL -- the guard at $guard_ln precedes the CLI parse"
  note "" "      at $parse_ln, so --passes 3 is refused before it is read"
  fails+=(p_guard_after_parse)
else
  note p_guard_after_parse "FAIL -- could not locate both the parse and the guard"
  fails+=(p_guard_after_parse)
fi

# --- 6. the probe must relay a verdict to the runner ------------------------
if grep -Fq 'RESULT: PASS' "$CODE"; then
  note p_report "ok"
else
  note p_report "FAIL -- no \`RESULT: PASS\` relay; the stack cannot classify it"
  fails+=(p_report)
fi

# --- 7. no env var may drive the probe -------------------------------------
env_hits="$(grep -n -E 'getenv|setenv|INTFER_' "$CODE" | wc -l)"
if [ "$env_hits" -eq 0 ]; then
  note p_no_env "ok (0 env references)"
else
  note p_no_env "FAIL -- $env_hits env reference(s); a probe-local INTFER_ var"
  note "" "      would file an uncertified flag in check_flag_gate_coverage.sh's"
  note "" "      derived census"
  fails+=(p_no_env)
fi

# --- 8. no adopt/decline verdict may be printed ----------------------------
# A printed verdict, not a mention: restrict to printf lines from the
# comment-stripped copy, or the probe's own prose refuses the healthy source.
verdict_hits="$(grep -E 'printf' "$CODE" | grep -icE 'adopt|faster|slower|reject|speedup' || true)"
if [ "$verdict_hits" -eq 0 ]; then
  note p_no_verdict "ok (0 printed verdicts)"
else
  note p_no_verdict "FAIL -- $verdict_hits printed verdict line(s); the S1"
  note "" "      decision is the operator's on nk/nw/nr, never the probe's"
  fails+=(p_no_verdict)
fi

evaluated=8
echo "S1-PRED: evaluated $evaluated predicate(s) on $(basename "$SUBJ")"
if [ "${#fails[@]}" -eq 0 ]; then
  echo "S1-PRED: PASS (all $evaluated predicates hold)"
  echo "RESULT: PASS"
  exit 0
fi
echo "S1-PRED: FAIL -- refused by: ${fails[*]}"
echo "RESULT: FAIL"
exit 1
