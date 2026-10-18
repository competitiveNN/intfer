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
# THE ELEVEN PREDICATES. Each refuses the shape of a defect that was actually
# planted and measured (mutate_knob_sweep_contract.sh):
#   p_knob_catalog   the 3 geometry knobs are the default catalogue (KS-catalog)
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

evaluated=12
echo "KS-CONTRACT: evaluated $evaluated predicate(s) on $(basename "$SUBJ")"
if [ "${#fails[@]}" -eq 0 ]; then
  echo "KS-CONTRACT: PASS (all $evaluated predicates hold)"
  echo "RESULT: PASS"
  exit 0
fi
echo "KS-CONTRACT: FAIL -- refused by: ${fails[*]}"
echo "RESULT: FAIL"
exit 1
