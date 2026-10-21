#!/usr/bin/env bash
# check_xmx_attrib.sh -- SOURCE-level structural certification of the engine's
# XMX attribution line ([xmx-attrib]) and the counter triple behind it.
#
# WHY A SOURCE CHECKER AND NOT A GATE. The line is printed under
# INTFER_PROFILE_GPU by examples/intfer_cli/main.cpp beside nk/nw/nr, and
# nk/nw/nr are the ONLY load-insensitive decision surface this project has
# (AGENTS.md, PARAMOUNT DIRECTIVE). A runtime gate would need the 27B --tp 2
# run, because that run is what prints it -- so an agent session could never
# certify the instrument, which is the "opt-in arm no gate can reach" defect in
# its purest form. What CAN be graded at source is the CONTRACT that makes the
# line trustworthy, and each half of it is a claim about THIS source:
#   * it carries a three-way VERDICT for a question that is not a number --
#     "did the matrix engines run?" -- and the ABSENCE is NAMED rather than
#     printed as an XVE-shaped zero, because a reader would read "0 XMX" as a
#     measurement;
#   * EVERY [GPU profile] dump site prints it, so one code path cannot silently
#     lose the only instrument the XMX verdict is read from;
#   * every XMX dispatch ATTEMPT is noted exactly once, so a declined launch is
#     recorded as a decline and a launch that never happened is not recorded as
#     an attempt. Without that, the three-way verdict is wrong in exactly the
#     direction AGENTS.md's "the matrix engines are idle" claim depends on.
#
# THE SIX PREDICATES. Each refuses the shape of a defect that was actually
# planted and measured (mutate_xmx_attrib.sh):
#   p_line               the [xmx-attrib] line exists as CODE    (XA-line)
#   p_matrix_engines     it carries matrix_engines=<verdict>      (XA-engines)
#   p_triple             all three counters are named in it      (XA-triple)
#   p_named_absence      the absence is NAMED, never a zero      (XA-absence)
#   p_verdict_is_a_word  that value is a SELECTED word, not a number (XA-word)
#   p_xmxmon_independence  that verdict never consults the env  (XA-xmxmon)
#   p_dump_universality  every [GPU profile] site prints it      (XA-dump)
#   p_site_coverage      every attempt is noted exactly once     (XA-site)
#
# WHY COMMENTS ARE STRIPPED FIRST. [xmx-attrib] appears in THREE places in
# engine.cpp: ONE is the construction, TWO are full-line comments describing
# it. An unstripped grep passes a file whose PROSE describes a line the code
# does not produce -- a predicate that fires on its own subject has no negative
# direction. MEASURED, not assumed: 3 hits raw, 1 after stripping (the same
# trap check_s1_predicates.sh documents for its own verdict words).
#
# THE COUNTS ARE DERIVED, NOT HAND-WRITTEN. p_dump_universality compares the
# number of xmx_attrib_line( call sites against the number of [GPU profile
# markers it finds, and p_site_coverage derives the attempt-flag population the
# same way. No literal arm count is asserted, so a site added to either file
# cannot leave this checker asserting a stale total -- the mechanism that
# caught a REAL missing dump site on 2026-10-16, when five [GPU profile markers
# carried four xmx_attrib_line calls and the operator reading the fifth prefill
# line had no XMX verdict at all.
#
# ANTI-VACUITY. A missing or empty subject is a FAIL, never a pass: "looked at
# nothing" and "found nothing wrong" are different claims.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUBJ="${1:-$ROOT/src/engine/engine.cpp}"
DUMP="${2:-$ROOT/examples/intfer_cli/main.cpp}"
HDR="$ROOT/include/intfer/engine.hpp"

for f in "$SUBJ" "$DUMP" "$HDR"; do
  if [ ! -s "$f" ]; then
    echo "XA-ATTRIB: FAIL -- subject missing or empty ($f)"
    echo "RESULT: FAIL"
    exit 1
  fi
done

CODE="$(mktemp "${TMPDIR:-/tmp}/xa_attrib.XXXXXX")"
DCODE="$(mktemp "${TMPDIR:-/tmp}/xa_dump.XXXXXX")"
HCODE="$(mktemp "${TMPDIR:-/tmp}/xa_hdr.XXXXXX")"
trap 'rm -f "$CODE" "$DCODE" "$HCODE"' EXIT
# INT/TERM must ALSO reach the cleanup: an EXIT trap alone never runs on a signal
# (bash fires EXIT only at a reached end or an explicit exit), so a SIGTERM killed
# the script outright and leaked all three temp files -- AUDIT #29. The exit routes
# the signal back through the EXIT handler; "clean up and resume" leaves the process
# alive, which is strictly worse (check_trap_release.sh).
trap 'rm -f "$CODE" "$DCODE" "$HCODE"; exit 130' INT
trap 'rm -f "$CODE" "$DCODE" "$HCODE"; exit 143' TERM
# Delete full-line // comments only. No code line in either file begins with
# // and the surviving comments are trailing-only, so a trailing comment cannot
# be mistaken for the construction under test.
sed -e '/^[[:space:]]*\/\//d' "$SUBJ" > "$CODE"
sed -e '/^[[:space:]]*\/\//d' "$DUMP" > "$DCODE"
sed -e '/^[[:space:]]*\/\//d' "$HDR"  > "$HCODE"
if [ ! -s "$CODE" ] || [ ! -s "$DCODE" ]; then
  echo "XA-ATTRIB: FAIL -- nothing but comments; a strippable file is a file"
  echo "           with no contract in it and must not read as a pass"
  echo "RESULT: FAIL"
  exit 1
fi

fails=()
note() { printf 'XA-ATTRIB: %-19s %s\n' "$1" "$2"; }
# fixed-string count, always a number, never a bare grep exit code
cmp_cnt() { printf '%s' "$(grep -Fc -- "$1" "$2" 2>/dev/null || true)"; }

# --- 1. the [xmx-attrib] line exists as code, not as prose -------------------
if [ "$(cmp_cnt '[xmx-attrib]' "$CODE")" -eq 1 ]; then
  note p_line "ok (one construction; the two prose mentions stripped)"
else
  note p_line "FAIL -- [xmx-attrib] is not constructed exactly once in"
  note "" "        engine.cpp; the only instrument the XMX verdict is read"
  note "" "        from either does not exist or exists twice"
  fails+=(p_line)
fi

# --- 2. the line carries a verdict for a non-numeric question ----------------
# matrix_engines= is the field that answers "did the matrix engines run?" with
# a WORD. A near-miss spelling would leave the operator reading
# requests/launches/declines and drawing the verdict themselves.
if [ "$(cmp_cnt 'matrix_engines=' "$CODE")" -eq 1 ]; then
  note p_matrix_engines "ok (matrix_engines=<verdict>)"
else
  note p_matrix_engines "FAIL -- matrix_engines= is absent; the line would"
  note "" "        report three integers and no verdict, which is the shape"
  note "" "        that let 'requested, all declined' read as 'idle'"
  fails+=(p_matrix_engines)
fi

# --- 3. all three counters are named in the line ----------------------------
# The triple is what distinguishes "never requested" from "requested, every
# tile declined" from "actually ran" (include/intfer/engine.hpp). A line that
# names only two of them cannot make that distinction.
triple_ok=1
for f in 'attempts=' 'launches=' 'declined='; do
  if [ "$(cmp_cnt "$f" "$CODE")" -ne 1 ]; then
    note p_triple "FAIL -- $f is not printed exactly once; the three-way"
    note "" "        verdict cannot be derived from a partial triple"
    triple_ok=0
    break
  fi
done
if [ "$triple_ok" -eq 1 ]; then
  note p_triple "ok (attempts= launches= declined=)"
else
  fails+=(p_triple)
fi

# --- 4. the ABSENCE is named, never an XVE-shaped zero ----------------------
# Printing nothing, or a zero, for "the hardware counters were not read" would
# be the vacuous-green shape: a reader would read 0 as a measurement. The line
# must therefore carry the two distinguishable absence verdicts AND say in words
# that the hardware counters were not read.
abs_ok=1
for a in 'not-requested' 'requested-all-declined' 'NOT READ'; do
  if [ "$(cmp_cnt "$a" "$CODE")" -lt 1 ]; then
    note p_named_absence "FAIL -- the absence marker '$a' is absent; a silent"
    note "" "        or zeroed absence would read as a measurement of zero"
    abs_ok=0
    break
  fi
done
if [ "$abs_ok" -eq 1 ]; then
  note p_named_absence "ok (absence named, hardware counters marked NOT READ)"
else
  fails+=(p_named_absence)
fi

# --- 4b. the verdict is SELECTED among those literals, not computed --------
# p_named_absence proves the three WORDS EXIST. It cannot prove the emitted
# value is one of them, which is the whole question: a `matrix_engines=%u` fed
# from xmx_launch_count, or `s += std::to_string(p.xmx_launch_count)`, renders
# "matrix_engines=0" for the not-requested state -- the vacuous green the words
# were added to prevent -- while all three words still sit in the file beside
# it, so p_named_absence stays green. MEASURED as a planted defect (XA-numverdict
# in mutate_xmx_attrib.sh), not anticipated.
# SO THE VALUE IS FOLLOWED. Three assertions, all derived from the source:
#   (a) exactly one matrix_engines= append exists, and the statement right
#       after it appends a BARE IDENTIFIER -- a conversion of a counter is
#       refused by the identifier shape alone, so `%u`/`%d`/`to_string`/an int
#       cast/a counter field cannot pass;
#   (b) that identifier is declared a STRING type (`const char*`), so it cannot
#       hold a number;
#   (c) it is ASSIGNED from all three literals, i.e. it is SELECTED among them
#       rather than defaulted once and never re-pointed.
word_ok=1
n_app="$(cmp_cnt 's += " matrix_engines=";' "$CODE")"
if [ "$n_app" -ne 1 ]; then
  note p_verdict_is_a_word "FAIL -- the matrix_engines= append is constructed"
  note "" "        $n_app time(s); the value's shape cannot be checked against"
  note "" "        a population that is not exactly one"
  word_ok=0
else
  feed="$(grep -A1 's += " matrix_engines=";' "$CODE" | sed -n '2{s/.*s += //;s/;.*//;s/[[:space:]]*$//;p}')"
  case "$feed" in
    '') note p_verdict_is_a_word "FAIL -- nothing is appended after"
        note "" "        matrix_engines=, so the field is empty and a reader"
        note "" "        cannot tell 'idle' from 'not measured'"
        word_ok=0 ;;
    *[!A-Za-z0-9_]*|[0-9]*)
        note p_verdict_is_a_word "FAIL -- matrix_engines= is fed '${feed:-<empty>}',"
        note "" "        not a bare identifier. A number, a %u/%d conversion, a"
        note "" "        std::to_string of a counter or a cast is the exact"
        note "" "        vacuous green p_named_absence cannot see"
        word_ok=0 ;;
  esac
  if [ "$word_ok" -eq 1 ]; then
    if ! grep -qE "const +char *\* +$feed\b" "$CODE"; then
      note p_verdict_is_a_word "FAIL -- $feed is not declared a string type;"
      note "" "        a numeric identifier could hold a count that reads"
      note "" "        as a verdict"
      word_ok=0
    fi
    n_sel=0
    # FLATTENED first: the ternary's second literal sits on the NEXT source
    # line, so a line-oriented grep finds 2 of the 3 on the healthy file and
    # this predicate then refuses the shipped code -- the same wrapped-citation
    # trap check_durable_claims.sh documents for HISTORY.md's pairs.
    flat="$(tr '\n' ' ' < "$CODE" | tr -s ' ')"
    for lit in not-requested requested-all-declined active; do
      if printf '%s' "$flat" | grep -qE "$feed[^;]*\"$lit\""; then
        n_sel=$((n_sel + 1))
      fi
    done
    if [ "$n_sel" -ne 3 ]; then
      note p_verdict_is_a_word "FAIL -- $feed is assigned only $n_sel of the three"
      note "" "        verdict literals; a verdict that is set once and never"
      note "" "        re-pointed cannot distinguish all three states"
      word_ok=0
    fi
    # and no numeric printf spelling of the field exists anywhere
    n_fmt="$(grep -cE 'matrix_engines=%[ud]' "$CODE" || true)"
    if [ "$n_fmt" -ne 0 ]; then
      note p_verdict_is_a_word "FAIL -- $n_fmt matrix_engines=%u/%d spelling(s)"
      note "" "        exist; a numeric printf is the defect this predicate"
      note "" "        exists to refuse, wherever it is written"
      word_ok=0
    fi
    if [ "$word_ok" -eq 1 ]; then
      note p_verdict_is_a_word "ok (matrix_engines= is fed the selected '$feed')"
    fi
  fi
fi
if [ "$word_ok" -ne 1 ]; then
  fails+=(p_verdict_is_a_word)
fi

# --- 4c. the VERDICT's inputs do not depend on xmxmon -----------------------
# THE INTEGRATION GAP, and it is why this predicate exists rather than a note.
# check_flag_gate_coverage.sh harvests getenv("INTFER_*") flags by regex, so
# XMXMON -- no INTFER_ prefix -- is NOT in its population and the census says
# nothing about it (measured: 107 read / 105 by the census regex / 0 xmxmon
# sites). The claim that mattered was therefore left resting on a checker that
# never examined it: "the XMX verdict is readable on this box, where xmxmon
# cannot run (dev.xe.observation_paranoid=1, no sudo)". That is true only while
# the verdict's INPUTS never consult the environment, and nothing asserted it.
# MEASURED as a planted defect (XA-xmxmon), not anticipated.
# THE ASSERTION IS ON ASSIGNMENTS, NOT ON A REGION. The xmxmon PATH discovery
# sits between the selection and the print (it appends an optional annotation
# and never touches `verdict`), so any region wide enough to catch a
# getenv-conditioned assignment also contains a legitimate getenv -- and a
# region narrow enough to stay clean does not catch the planted defect. Two
# precise checks instead: no line that ASSIGNS the verdict consults the
# environment, and the declaration-to-last-assignment window carries no
# environment read. The first version was region-based and refused the HEALTHY
# source; the second double-caught XA-intverdict because an unresolvable region
# made this predicate fire alongside p_verdict_is_a_word. Both are recorded at
# the matrix.
indep_ok=1
if ! indep_line="$(python3 - "$CODE" <<'PY'
import re
import sys
lines = open(sys.argv[1]).read().split("\n")
# THE ASSIGNMENT LINES, not a region: a verdict that is CONDITIONED on the
# environment lands on one line carrying both getenv and `verdict =`, so the
# predicate is "no assignment of the verdict consults the environment". That
# formulation keeps working when the declaration's TYPE changes (XA-intverdict
# renames it to int), which a region-from-`const char*` could not: an
# unresolvable region made this predicate fire too, so the state proved
# nothing. MEASURED, not anticipated.
assigns = [ln for ln in lines
           if re.search(r"\bverdict\s*=[^=]", ln) or re.search(r"^\s*verdict\s*=", ln)]
if not assigns:
    sys.stderr.write("no verdict assignment found\n")
    sys.exit(2)
for ln in assigns:
    if "getenv" in ln or "XMXMON" in ln:
        sys.stdout.write(ln.strip()[:80])
        sys.exit(1)
# AND the narrow window from the declaration to the last assignment carries no
# unconditional environment read either.
start = next((i for i, ln in enumerate(lines) if re.search(r"\bverdict\b", ln)), None)
if start is not None:
    last = max(i for i, ln in enumerate(lines) if re.search(r"\bverdict\s*=", ln))
    for ln in lines[start:last + 1]:
        if "getenv" in ln or "XMXMON" in ln:
            sys.stdout.write(ln.strip()[:80])
            sys.exit(1)
sys.exit(0)
PY
)"; then
  note p_xmxmon_independence "FAIL -- the region that SELECTS the verdict"
  note "" "        consults the environment:"
  note "" "        ${indep_line:-<unresolved region>}"
  note "" "        so the verdict is readable only when xmxmon is installed,"
  note "" "        and on this box it is not -- which makes the one instrument"
  note "" "        the XMX claim rests on box-dependent"
  indep_ok=0
fi
if [ "$indep_ok" -eq 1 ]; then
  note p_xmxmon_independence "ok (the verdict's inputs never read the environment)"
fi
if [ "$indep_ok" -ne 1 ]; then
  fails+=(p_xmxmon_independence)
fi

# --- 5. EVERY [GPU profile] dump site prints the line -----------------------
# Derived, not a hand count: the population of dump sites is whatever the CLI
# has, and the calls must equal it. THIS IS THE PREDICATE THAT CAUGHT A REAL
# DEFECT -- five markers carried four calls, so the missing prefill site
# printed nk/nw/nr with no XMX verdict at all.
n_call="$(cmp_cnt 'xmx_attrib_line(' "$DCODE")"
n_site="$(cmp_cnt '[GPU profile' "$DCODE")"
if [ "$n_call" -eq "$n_site" ] && [ "$n_site" -ge 1 ]; then
  note p_dump_universality "ok ($n_call call(s) == $n_site dump site(s))"
else
  note p_dump_universality "FAIL -- $n_call xmx_attrib_line call(s) against"
  note "" "        $n_site [GPU profile] dump site(s); a site without the line"
  note "" "        prints the load-insensitive counters and NO XMX verdict,"
  note "" "        which is how an operator loses the only usable instrument"
  fails+=(p_dump_universality)
fi

# --- 6. every XMX dispatch ATTEMPT is noted, exactly once -------------------
# One helper exists so the triple cannot drift between the sites; a site that
# forgot the call would under-count attempts silently, and a site that counted
# a decline as a launch would make the three-way verdict wrong in the direction
# AGENTS.md depends on. The attempt-flag population is DERIVED from the code:
# every `bool xmx_ran*` declaration except the post-loop aggregate is one
# dispatch attempt.
n_def="$(cmp_cnt 'void xmx_note_dispatch(bool ran)' "$CODE")"
n_mentions="$(cmp_cnt 'xmx_note_dispatch(' "$CODE")"
n_flags="$(grep -cE 'bool +xmx_ran' "$CODE" || true)"
n_agg="$(grep -cE 'bool +xmx_ran_any' "$CODE" || true)"
n_sites=$(( n_mentions - n_def ))
n_attempts=$(( n_flags - n_agg ))
site_ok=1
if [ "$n_def" -ne 1 ]; then
  note p_site_coverage "FAIL -- the maintenance helper is not defined exactly"
  note "" "        once (found $n_def); more than one body means the triple"
  note "" "        can be maintained two different ways"
  site_ok=0
fi
if [ "$n_sites" -ne "$n_attempts" ] || [ "$n_attempts" -lt 1 ]; then
  note p_site_coverage "FAIL -- $n_sites noted dispatch site(s) against"
  note "" "        $n_attempts XMX attempt flag(s); an attempt nobody notes is"
  note "" "        an under-count that reads as 'the engines were idle'"
  site_ok=0
fi
# and each snapshot field is advanced exactly once, inside that helper
n_incr="$(grep -cE 'xmx_[a-z_]+_count \+= 1' "$CODE" || true)"
n_fields="$(grep -oE 'xmx_[a-z_]+_count' "$HCODE" | sort -u | wc -l)"
if [ "$n_incr" -ne "$n_fields" ] || [ "$n_fields" -lt 3 ]; then
  note p_site_coverage "FAIL -- $n_incr counter increment(s) for $n_fields"
  note "" "        snapshot field(s); a field advanced twice is double-counted"
  note "" "        and one advanced never reads as a constant zero"
  site_ok=0
fi
if [ "$site_ok" -eq 1 ]; then
  note p_site_coverage "ok ($n_sites sites == $n_attempts attempts, $n_incr/$n_fields fields advanced)"
else
  fails+=(p_site_coverage)
fi

# the count is DERIVED from this file's own note sites, so a predicate added
# without bumping the literal cannot grade green
preds="$(grep -o 'note p_[a-z_]* ' "${BASH_SOURCE[0]}" 2>/dev/null | sort -u | wc -l)"
if [ "$preds" -ne 8 ]; then
  echo "XA-ATTRIB: FAIL -- this checker derives $preds distinct p_* predicate(s)"
  echo "           but asserts 8; the literal and the sites have drifted, so"
  echo "           neither can be trusted and the count must be reconciled"
  echo "RESULT: FAIL"
  exit 1
fi
evaluated=8
echo "XA-ATTRIB: evaluated $evaluated predicate(s) on $(basename "$SUBJ") + $(basename "$DUMP")"
if [ "${#fails[@]}" -eq 0 ]; then
  echo "XA-ATTRIB: PASS (all $evaluated predicates hold)"
  echo "RESULT: PASS"
  exit 0
fi
echo "XA-ATTRIB: FAIL -- refused by: ${fails[*]}"
echo "RESULT: FAIL"
exit 1
