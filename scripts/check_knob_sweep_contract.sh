#!/usr/bin/env bash
# check_knob_sweep_contract.sh -- SOURCE-level structural certification of the
# operator-run knob sweep (scripts/knob_sweep.sh).
#
# WHY A SOURCE CHECKER AND NOT A GATE. knob_sweep.sh is OPERATOR-ONLY: it loads
# the 27B --tp 2 target, so the standing rule forbids an agent session from
# running it, and a gate built by running it would be a green tick certifying
# nothing -- the "opt-in flag no gate can reach" defect in its purest form. What
# CAN be certified without touching the box is the harness's CONTRACT: that it
# enumerates a product over the three intra-device geometry knobs, that it reads
# the load-insensitive counters off the CLI's own profile line, that a delta must
# clear a spread floor across >= 2 paired rounds, that a contended round
# publishes no figure, and that it decides on counters rather than a tok/s.
# Every one of those is a claim about the SOURCE, so this checker grades the
# guards at source, with no execution and no GPU -- and the sweep's own numbers
# stay UNMEASURED on purpose, for the operator.
#
# THE FOURTEEN PREDICATES. Each refuses the shape of a defect that was actually
# planted and measured (mutate_knob_sweep_contract.sh). The count is DERIVED, not
# remembered: the `preds=` tally below counts the `note p_` sites THIS file emits,
# and is asserted against the literal in the `if [ "$preds" -ne N ]` test and the
# `evaluated=` line, so a predicate added without bumping both cannot grade green.
# THIS LIST WAS ITSELF THE STALE ARTIFACT TWICE, and both are recorded rather than
# quietly fixed: it read "ELEVEN" above twelve entries, and it still read TWELVE
# after the THIRTEENTH (p_product_enum) was added without being named here -- the
# same drift the checker exists to catch, in the checker's own header:
#   p_knob_catalog   the 3 geometry knobs are the default catalogue (KS-catalog)
#   p_product_enum   the product ACTUALLY ENUMERATES -- EXECUTED, not grepped (KS-product)
#   p_knob_consumed  every catalogued knob HAS A CONSUMER -- EXECUTED      (KS-consumed)
#   p_knob_override  INTFER_SWEEP_KNOBS overrides that catalogue  (KS-override)
#   p_baseline_first the baseline runs before every combo          (KS-baseline)
#   p_profile_scrape counters come from [GPU profile @decode]      (KS-scrape)
#   p_fields         the nk/nw/nr integer field regexes exist      (KS-fields)
#   p_spread_floor   a delta needs usable>=2 AND max(two spreads)  (KS-floor)
#   p_load_gate      a CONTENDED round publishes NO figure         (KS-loadgate)
#   p_guard_before_publish the verdict guard precedes publication  (KS-guardafter)
#   p_no_eval        the knob spec is exported, never eval'd       (KS-eval)
#   p_no_tokps       no tok/s is printed or quoted                 (KS-tokps)
#   p_operator_only  build/device/2-GPU/render-node preconditions  (KS-precond)
#   p_table          the table is written to a durable artifact    (KS-table)
#
# WHY COMMENTS ARE STRIPPED FIRST. p_load_gate greps for `loadlog_usable` and
# `SKIPPED-CONTENDED`, and the script's OWN HEADER COMMENT names both: it was
# written to describe the guard. An unstripped grep therefore passes a file whose
# comment describes a guard the code does not contain -- which is precisely the
# defect that guard was shipped to fix (the first revision of knob_sweep.sh
# claimed "a combo's figure is withheld (SKIPPED-CONTENDED)" while its code read
# loadlog_arm's exit code, so a CONTENDED round arrived with rc=0 AND a scrapeable
# profile line and fed the median). A predicate that fires on its own subject has
# no negative direction. Full-line `#` comments are removed; no code line in the
# subject begins with `#`, and the surviving comments are trailing-only.
#
# ANTI-VACUITY. An empty or missing subject is a FAIL, never a pass: "looked at
# nothing" and "found nothing wrong" are different claims.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBJ="${1:-$ROOT/scripts/knob_sweep.sh}"

if [ ! -f "$SUBJ" ]; then
  echo "KS-CONTRACT: FAIL -- subject missing ($SUBJ)"
  echo "RESULT: FAIL"
  exit 1
fi
if [ ! -s "$SUBJ" ]; then
  echo "KS-CONTRACT: FAIL -- subject empty ($SUBJ); a subject with nothing in it"
  echo "                cannot hold a predicate and must not read as a pass"
  echo "RESULT: FAIL"
  exit 1
fi

CODE="$(mktemp "${TMPDIR:-/tmp}/kscontract.XXXXXX")"
trap 'rm -f "$CODE"' EXIT
# Strip whole-line comments only, so a guard named in prose cannot satisfy a
# predicate about the code. `sed -e 's:#.*::'` would also truncate the parameter
# expansion ${#KNOBS[@]} and is not what is meant.
sed -e '/^[[:space:]]*#/d' "$SUBJ" > "$CODE"
if [ ! -s "$CODE" ]; then
  echo "KS-CONTRACT: FAIL -- nothing but comments ($SUBJ); a strippable file is"
  echo "                a file with no contract in it and must not read as a pass"
  echo "RESULT: FAIL"
  exit 1
fi

fails=()
note() { printf 'KS-CONTRACT: %-16s %s\n' "$1" "$2"; }

# --- 1. the three intra-device geometry knobs are the default catalogue ------
cat_ok=1
for a in 'INTFER_DECODE1_WG:-,16,64,256' \
         'INTFER_XMX_WG:-,16' \
         'INTFER_LA_CONV_PREP_WG:-,64'; do
  if ! grep -Fq -- "$a" "$CODE"; then
    note p_knob_catalog "FAIL -- the catalogue is missing $a; the product would"
    note "" "        silently not cover that knob, which is the gap this harness"
    note "" "        was written to close"
    cat_ok=0
    break
  fi
done
if [ "$cat_ok" -eq 1 ]; then note p_knob_catalog "ok (DECODE1_WG / XMX_WG / LA_CONV_PREP_WG)"; else fails+=(p_knob_catalog); fi

# --- 1b. the Cartesian product ACTUALLY ENUMERATES (EXECUTED, not grepped) -----
# THE DEFECT THIS EXISTS FOR. The shipped harness's walk() read its values with
# `while IFS= read -r v` -- a NEWLINE split -- while the documented spec form is
# comma-separated (`KNOB:-,16,64,256`, per this file's own header). So a knob's
# whole comma list collapsed into ONE value: PRODUCT held a single entry whose env
# string was `INTFER_DECODE1_WG=-,16,64,256`, `export` set the knob to that
# literal, and the CLI's atoi() of it is 0 = the SHIPPED DEFAULT. The sweep then
# ran ONE arm indistinguishable from the baseline and printed a
# baseline-relative table of deltas against itself.
# Every one of the 12 predicates above that existed at the time passed it:
# they grade "the catalogue names the knobs" and "the field regexes are right",
# never "the product has more than one arm". A source grep cannot see it --
# `tr ',' '\n'` and `IFS= read` are both correct-looking in isolation.
# So this one RUNS the real parse, from the real source, and compares the arm
# count against the count derived from the spec's own comma lists. A predicate
# that cannot fail on a one-arm product is not covering the thing that mattered.
prod_run="$(mktemp "${TMPDIR:-/tmp}/ksprod.XXXXXX")"
{
  printf 'set -u\n'
  printf "DEFAULT_KNOBS='%s'\n" "$(grep -m1 '^DEFAULT_KNOBS=' "$CODE" \
      | cut -d= -f2- | sed "s/^'//; s/'\$//")"
  # the real parse block, harvested from the subject so a rewritten walk()
  # is what gets executed rather than a reconstruction of it
  printf 'KNOB_SPEC="${INTFER_SWEEP_KNOBS:-$DEFAULT_KNOBS}"\n'
  sed -n '/^declare -a KNOB_NAMES=()/,/^walk 0 ""/p' "$CODE"
  printf 'echo "NKNOBS=$NKNOBS"\necho "PRODUCT=${#PRODUCT[@]}"\n'
  printf 'for i in "${!PRODUCT[@]}"; do echo "ARM[$i]<<${PRODUCT[$i]}>>"; done\n'
} >"$prod_run"
# derive the expected arm count from the spec TEXT: one branch per listed value,
# because walk() emits one branch per value and a "-" branch adds no env pair
exp=1
OLDIFS="$IFS"; IFS='|'
# NOTE: grep the FILE ($CODE), never `printf '%s\n' "$CODE" | grep` -- that pipes
# the mktemp PATH as text, matches nothing, and leaves exp at 1, which reads as
# "the spec derives one arm" and fails a healthy product. Hit once already.
for pair in $(grep -m1 '^DEFAULT_KNOBS=' "$CODE" | cut -d= -f2-); do
  n="$(printf '%s' "${pair#*:}" | awk -F, '{print NF}')"
  exp=$((exp * n))
done
IFS="$OLDIFS"
if bash "$prod_run" >"$prod_run.out" 2>"$prod_run.err"; then
  nk="$(sed -n 's/^NKNOBS=//p' "$prod_run.out")"
  nprod="$(sed -n 's/^PRODUCT=//p' "$prod_run.out")"
  if [ "${nprod:-0}" -gt 1 ] && [ "${exp:-0}" -gt 1 ] \
     && [ "$nprod" -eq "$exp" ]; then
    note p_product_enum "ok (EXECUTED: $nk knob(s) -> $nprod arm(s), matches the"
    note "" "             derived count $exp; a one-arm product would FAIL here)"
  else
    note p_product_enum "FAIL -- the harness's own parse yields $nprod arm(s) from"
    note "" "             $nk knob(s) while the spec derives $exp. A product that"
    note "" "             does not enumerate runs one arm against itself and still"
    note "" "             prints a baseline-relative table -- the exact vacuity this"
    note "" "             checker's other 13 predicates could not see."
    fails+=(p_product_enum)
  fi
else
  note p_product_enum "FAIL -- the harness's own parse could not be executed:"
  sed 's/^/        /' "$prod_run.err" | head -3
  fails+=(p_product_enum)
fi
rm -f "$prod_run" "$prod_run.out" "$prod_run.err"

# --- 1c. every catalogued knob HAS A CONSUMER, proved by EXECUTING the harvest --
# THE DEFECT CLASS THIS EXISTS FOR. A knob the catalogue enumerates that NO source
# file reads is an INERT ARM GENERATOR: the product still enumerates (so
# p_product_enum is satisfied), all three shipped names are still present (so
# p_knob_catalog is satisfied), every arm exports a value for a flag nothing reads,
# every arm then runs the IDENTICAL shipped default -- and the sweep prints a
# confident baseline-relative table of deltas against itself. That is the same
# vacuous-but-green shape as the one-arm product, one layer further out, and
# knob_sweep.sh cannot refuse it: its spec parser validates only that a knob NAME
# is non-empty, never that anything consumes it. INTFER_SWEEP_KNOBS makes the case
# reachable without editing this harness at all, so an operator sweeping a retired
# or typo'd knob gets N arms and no warning.
# WHY THE SEARCH INCLUDES examples/ AND include/, and not src/ alone: the four
# DFlash draft knobs the HISTORY.md runbook sweeps
# (INTFER_DRAFT_SOURCE/_ASYNC/_CANCEL_STALE/_SCORE_POLICY) are all read in
# examples/intfer_cli/main.cpp, so a src/-only predicate would REDDEN the operator
# runbook this checker's sibling section documents -- a checker wrong about its own
# subject is worse than the gap it was written to close. MEASURED, not assumed:
# all six DFlash/SPEC knob names resolve in examples/, all three geometry knobs
# resolve in src/gemm/quant_gemm.cpp.
cons_spec="$(grep -m1 '^DEFAULT_KNOBS=' "$CODE" | cut -d= -f2- | sed "s/^'//; s/'\$//")"
cons_ok=1; cons_n=0; cons_bad=""
if [ -n "$cons_spec" ]; then
  OLDIFS3="$IFS"; IFS='|'
  # shellcheck disable=SC2206
  read -ra CONS_PAIRS <<<"$cons_spec"
  IFS="$OLDIFS3"
  for ck in "${CONS_PAIRS[@]}"; do
    cn="${ck%%:*}"
    [ -n "$cn" ] || continue
    cons_n=$((cons_n + 1))
    chits="$(grep -rlF --include='*.cpp' --include='*.hpp' "getenv(\"$cn\")" \
            "$ROOT/src" "$ROOT/examples" "$ROOT/include" 2>/dev/null | wc -l)"
    if [ "$chits" -lt 1 ]; then
      cons_ok=0; cons_bad="$cons_bad $cn"
    fi
  done
fi
if [ "$cons_ok" -eq 1 ] && [ "$cons_n" -ge 1 ]; then
  note p_knob_consumed "ok (EXECUTED: all $cons_n catalogued knob(s) have a getenv"
  note "" "                   reader under src/ examples/ include/; an inert arm"
  note "" "                   generator would FAIL here despite a real product)"
elif [ "$cons_n" -lt 1 ]; then
  note p_knob_consumed "FAIL -- harvested ZERO knobs from DEFAULT_KNOBS;"
  note "" "                   'every knob is consumed' is vacuously true of an"
  note "" "                   empty catalogue, which must not read as a pass"
  fails+=(p_knob_consumed)
else
  note p_knob_consumed "FAIL -- no source file reads:$cons_bad"
  note "" "                   Every arm exports a value for that flag and nothing"
  note "" "                   reads it, so all N arms run the identical shipped"
  note "" "                   default and the table reports deltas against itself"
  fails+=(p_knob_consumed)
fi

# --- 2. the catalogue is env-overridable, and the override is the spec --------
if grep -Fq 'KNOB_SPEC="${INTFER_SWEEP_KNOBS:-$DEFAULT_KNOBS}"' "$CODE" \
   && grep -Fq 'INTFER_SWEEP_KNOBS' "$CODE"; then
  note p_knob_override "ok (INTFER_SWEEP_KNOBS overrides DEFAULT_KNOBS)"
else
  note p_knob_override "FAIL -- no INTFER_SWEEP_KNOBS override of DEFAULT_KNOBS;"
  note "" "        a fixed catalogue cannot be narrowed for a targeted A/B"
  fails+=(p_knob_override)
fi

# --- 3. the baseline is PRODUCT[0] and runs before every combo ---------------
base_ln="$(grep -n -F 'BASELINE="${PRODUCT[0]}"' "$CODE" | head -1 | cut -d: -f1)"
call_ln="$(grep -n -F 'run_combo "BASELINE($BASELINE)"' "$CODE" | head -1 | cut -d: -f1)"
loop_ln="$(grep -n -F 'for combo in "${PRODUCT[@]:1}"' "$CODE" | head -1 | cut -d: -f1)"
if [ -n "$base_ln" ] && [ -n "$call_ln" ] && [ -n "$loop_ln" ] \
   && [ "$call_ln" -lt "$loop_ln" ]; then
  note p_baseline_first "ok (baseline at $call_ln, loop at $loop_ln)"
else
  note p_baseline_first "FAIL -- the baseline is not PRODUCT[0] run before the"
  note "" "        combos (base=$base_ln call=$call_ln loop=$loop_ln); every"
  note "" "        delta would then be measured against whatever ran first"
  fails+=(p_baseline_first)
fi

# --- 4. the counters come from the CLI's own load-insensitive profile line ---
if grep -Fq 'grep -a ' "$CODE" && grep -Fq '\[GPU profile @decode\]' "$CODE"; then
  note p_profile_scrape "ok ([GPU profile @decode])"
else
  note p_profile_scrape "FAIL -- no [GPU profile @decode] scrape; without the"
  note "" "        device's own event-timestamped profile line the sweep would"
  note "" "        be deciding on host wall-clock, which PARAMOUNT retired"
  fails+=(p_profile_scrape)
fi

# --- 5. the nk/nw/nr integer field regexes exist ------------------------------
field_ok=1
for a in 'nk=([0-9]+)' 'nw=([0-9]+)' 'nr=([0-9]+)'; do
  if ! grep -Fq -- "$a" "$CODE"; then
    note p_fields "FAIL -- the field regex $a is absent; that counter would be"
    note "" "        silently dropped from the reported set"
    field_ok=0
    break
  fi
done
if [ "$field_ok" -eq 1 ]; then note p_fields "ok (nk/nw/nr exact integers)"; else fails+=(p_fields); fi

# --- 6. a delta must CLEAR a spread floor across >= 2 paired rounds ----------
floor_ok=1
for a in '[ "$usable" -ge 2 ]' \
         'CLEARS SPREAD FLOOR' \
         'INSIDE spread floor' \
         '(a>b?a:b)'; do
  if ! grep -Fq -- "$a" "$CODE"; then
    note p_spread_floor "FAIL -- $a is absent from the spread-floor rule"
    floor_ok=0
    break
  fi
done
if [ "$floor_ok" -eq 1 ]; then
  note p_spread_floor "ok (usable>=2, floor=max of the two spreads, both verdicts)"
else
  note "" "        A delta that cannot fail to clear a floor is not a result, and"
  note "" "        a single-round delta has no noise floor to clear at all"
  fails+=(p_spread_floor)
fi

# --- 7. a CONTENDED round publishes NO figure --------------------------------
# THE DEFECT THIS GUARD FIXES, and the reason the comments are stripped: the
# first revision of knob_sweep.sh named SKIPPED-CONTENDED in its header comment
# and never withheld anything, because loadlog_arm returns the CLI's exit code.
gate_ok=1
grep -Fq 'if ! loadlog_usable; then' "$CODE" || gate_ok=0
# The withheld round must be RECORDED into the results, not merely mentioned or
# echoed: an echo to the terminal is silent to everyone reading sweep_table.txt,
# and a withholding the operator cannot see is how a harness goes quietly blank
# (the AUDIT #20 shape this repo records for an empty verdict).
grep -Fq 'result_lines["$label"]+="r$r SKIPPED-CONTENDED' "$CODE" || gate_ok=0
# BOTH usable filters -- the per-arm one and the baseline one -- must exclude the
# withheld round, or the baseline becomes a contaminated reference.
[ "$(grep -c -F '*SKIPPED*)' "$CODE")" -ge 2 ] || gate_ok=0
if [ "$gate_ok" -eq 1 ]; then
  note p_load_gate "ok (loadlog_usable consulted, round recorded, 2 filters exclude)"
else
  note p_load_gate "FAIL -- a contended round is not withheld. loadlog_arm returns"
  note "" "        the CLI's rc, so a CONTENDED round arrives with rc=0 AND a"
  note "" "        scrapeable profile line; without this guard the sweep reports a"
  note "" "        delta measured on a contended box -- the exact failure mode the"
  note "" "        load gate exists to prevent"
  fails+=(p_load_gate)
fi


# --- 12. the verdict guard must sit where the "no figure" guarantee needs it --
# TWO orderings, both necessary, and the first one is NOT optional:
rc_chk_ln="$(grep -n -F 'if [ $rc -ne 0 ]; then' "$CODE" | head -1 | cut -d: -f1)"
guard_ln="$(grep -n -F 'if ! loadlog_usable; then' "$CODE" | head -1 | cut -d: -f1)"
publish_ln="$(grep -n -F 'result_lines["$label"]+="r$r $prof"' "$CODE" | head -1 | cut -d: -f1)"
if [ -n "$rc_chk_ln" ] && [ -n "$guard_ln" ] && [ -n "$publish_ln" ] \
   && [ "$guard_ln" -gt "$rc_chk_ln" ] && [ "$guard_ln" -lt "$publish_ln" ]; then
  note p_guard_before_publish "ok (guard $guard_ln after rc $rc_chk_ln, before publish $publish_ln)"
else
  note p_guard_before_publish "FAIL -- the load-verdict guard is not between the rc"
  note "" "        check and the round's publication (rc=$rc_chk_ln guard=$guard_ln"
  note "" "        publish=$publish_ln)"
  fails+=(p_guard_before_publish)
fi
# WHY THE GUARD FOLLOWS THE rc CHECK AND NOT THE OTHER WAY ROUND, recorded here so
# a future session does not 'simplify' it into the wrong order. Reading the verdict
# BEFORE the rc check is not a harmless refactor:
#   * it SWALLOWS a crashed round's exit code. A contended AND crashed round would
#     be recorded SKIPPED-CONTENDED, so the operator reads 'the box was busy' and
#     re-runs later, never learning the binary died at all;
#   * loadlog_arm also returns 1 for its OWN precondition failure ("loadlog_init
#     first", _gate_loadlog.sh:68), which has nothing to do with load -- labelling
#     that SKIPPED-CONTENDED is actively wrong;
#   * the rc check costs nothing to run first: a dead round has no figure to
#     withhold, so FAILED is strictly the more informative of the two records.
# The half that IS load-bearing is the SECOND ordering -- the guard must precede the
# line that publishes the round, or a contended arm's profile line reaches the
# usable set and feeds the median and the spread floor.
# --- 8. the knob spec is exported, never eval'd ------------------------------
if grep -qE '(^|[^[:alnum:]_])eval([^[:alnum:]_]|$)' "$CODE"; then
  note p_no_eval "FAIL -- eval is used on the knob spec; the comment says the"
  note "" "        env is exported explicitly and eval would make a product entry"
  note "" "        a command-line, not a variable assignment"
  fails+=(p_no_eval)
else
  note p_no_eval "ok (exported explicitly, no eval)"
fi

# --- 9. no tok/s may be printed or quoted ------------------------------------
# Restricted to output statements from the stripped copy, or the header comment
# that says exactly this refuses the healthy source.
tps_hits="$(grep -E 'printf|echo' "$CODE" | grep -icE 'tok/s|tok per s|tps' || true)"
if [ "$tps_hits" -eq 0 ]; then
  note p_no_tokps "ok (0 printed tok/s)"
else
  note p_no_tokps "FAIL -- $tps_hits output line(s) carry a tok/s; this box has"
  note "" "        never held load1 < 2 and the sweep must decide on counters"
  fails+=(p_no_tokps)
fi

# --- 10. the operator-only preconditions are present ------------------------
pre_ok=1
for a in 'gate_require_build intfer_cli' \
         'gate_require_device' \
         'gate_require_gpus 2' \
         '/dev/dri/renderD128' \
         'exit 2'; do
  if ! grep -Fq -- "$a" "$CODE"; then
    note p_operator_only "FAIL -- the precondition $a is absent; the sweep would"
    note "" "        run on a box that cannot count its devices and report a"
    note "" "        SKIPPED arm as a measured one"
    pre_ok=0
    break
  fi
done
if [ "$pre_ok" -eq 1 ]; then
  note p_operator_only "ok (build / device / 2 GPUs / render node)"
else
  fails+=(p_operator_only)
fi

# --- 11. the table is written to a durable artifact -------------------------
if grep -Fq 'tee "$LOGDIR/sweep_table.txt"' "$CODE"; then
  note p_table "ok (sweep_table.txt)"
else
  note p_table "FAIL -- the table is not tee'd to a durable file; a table read"
  note "" "        once off the terminal cannot be re-derived, which is the"
  note "" "        'number reached a report without an artifact' failure"
  fails+=(p_table)
fi

# p_product_enum was added 2026-10-19 (the one-arm-product defect) and
# p_knob_consumed the same day (the inert-consumer product). The count is
# DERIVED from the `note p_*` sites above and asserted against the literal, so a
# predicate added without bumping anything grades green only if it is also named
# here -- and a name added here with no site is a vacuous predicate the same way.
# count the sites in THIS checker, not in the subject it grades
preds="$(grep -o 'note p_[a-z_]* ' "${BASH_SOURCE[0]}" 2>/dev/null | sort -u | wc -l)"
if [ "$preds" -ne 14 ]; then
  echo "KS-CONTRACT: FAIL -- this checker derives $preds distinct p_* predicate(s)"
  echo "                but asserts 14; the literal and the sites have drifted, so"
  echo "                neither can be trusted and the count must be reconciled"
  echo "RESULT: FAIL"
  exit 1
fi
evaluated=14
echo "KS-CONTRACT: evaluated $evaluated predicate(s) on $(basename "$SUBJ")"
if [ "${#fails[@]}" -eq 0 ]; then
  echo "KS-CONTRACT: PASS (all $evaluated predicates hold)"
  echo "RESULT: PASS"
  exit 0
fi
echo "KS-CONTRACT: FAIL -- refused by: ${fails[*]}"
echo "RESULT: FAIL"
exit 1
