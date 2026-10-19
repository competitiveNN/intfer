#!/usr/bin/env bash
# mutate_knob_sweep_contract.sh -- the twelve-state negative direction for
# check_knob_sweep_contract.sh. A predicate that can only pass is not a predicate.
#
# WHY THE SUBJECT IS EXECUTABLE CODE AND NOT A LOG. knob_sweep.sh is
# operator-only: it loads the 27B, so the standing rule forbids an agent session
# from running it, and a matrix built by running it would certify nothing. Every
# state below is a SOURCE mutation on a temp copy -- which also means the matrix
# is runnable on a busy box for free. Nothing here executes knob_sweep.sh.
#
# THE THREE LESSONS THIS MUTATOR ENFORCES BY CONSTRUCTION, each one a failure
# mode already recorded in docs/audit-log-tooling.md (#36 and its siblings) and
# each one caught by inspection of the S1 matrix:
#
#   * EVERY MUTATION ASSERTS IT LANDED BEFORE IT IS GRADED. An unlanded mutation
#     changes nothing, so a "catch" from it certifies only that the checker
#     refuses something it never saw -- a green tick on a vacuously-passing state.
#     `cmp` is the assertion; a no-op is a FAIL, never a catch.
#   * EVERY STATE NAMES ITS OWN PREDICATE, and the refusal reason is harvested
#     from the checker's output, so a state caught for the WRONG reason (two
#     predicates firing at once) is visible rather than counted.
#   * THE ANCHOR PREFLIGHT RUNS FIRST. If an anchor is absent from the live
#     subject, that state is vacuous by construction and the matrix dies rather
#     than grading it.
#
# THE STATES, each named for the predicate it must trip:
#   KS-catalog   drop a knob from the default catalogue   -> p_knob_catalog
#   KS-override  remove the INTFER_SWEEP_KNOBS override  -> p_knob_override
#   KS-baseline  run the baseline AFTER the combos        -> p_baseline_first
#   KS-scrape    scrape a line that is not @decode       -> p_profile_scrape
#   KS-fields    drop the nw field regex                  -> p_fields
#   KS-floor     accept a single usable round             -> p_spread_floor
#   KS-loadgate  never read the load verdict              -> p_load_gate
#   KS-guardafter move the guard AFTER the round publishes -> p_guard_before_publish
#   KS-eval      eval the knob spec                       -> p_no_eval
#   KS-tokps     print a tok/s                            -> p_no_tokps
#   KS-precond   drop gate_require_gpus 2                 -> p_operator_only
#   KS-table     stop tee-ing the durable table           -> p_table
#   KS-product   the comma/newline parse collapse (one arm) -> p_product_enum
#   KS-consumed  a catalogued knob nothing READS           -> p_knob_consumed
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE="${1:-$ROOT/scripts/knob_sweep.sh}"
CHECKER="$SCRIPT_DIR/check_knob_sweep_contract.sh"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/ksmut.XXXXXX")"
PLANTED_FOR_CLEANUP=""
# The durable-residue block plants a copy of THIS file inside scripts/ so the
# child's SCRIPT_DIR/CHECKER resolve. It removes its own plant on every path,
# but a FAIL mid-flight would otherwise leak that plant into the repo -- and a
# leaked plant inside scripts/ is exactly the kind of stray file a residue
# check is supposed to notice. The trap is the durable backstop.
trap 'rm -rf "$SANDBOX"; [ -z "${PLANTED_FOR_CLEANUP:-}" ] || rm -f "$PLANTED_FOR_CLEANUP"' EXIT

die() { echo "KS-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }

[ -f "$LIVE" ] || die "live subject missing ($LIVE)"
[ -f "$CHECKER" ] || die "checker missing ($CHECKER)"

# ---- the mutation engine, embedded so the whole matrix is one durable file ----
cat > "$SANDBOX/mut.py" <<'PYMUT'
"""Source mutations for mutate_knob_sweep_contract.sh.

Each state asserts its anchors BEFORE substituting and asserts the file actually
CHANGED before writing: an unlanded mutation changes nothing, so a "catch" from
it would certify that the checker never saw the defect -- the exact vacuous green
docs/audit-log-tooling.md failure mode #36 records.
argv: state, file. Exit 3 = an anchor was absent (vacuous, must not be graded);
exit 4 = the mutation did not land.
"""
import sys


def sub(s, old, new, n=1, what=''):
    if s.count(old) != n:
        sys.stderr.write('anchor %s count=%d want=%d\n' % (what or old[:48], s.count(old), n))
        sys.exit(3)
    return s.replace(old, new)


def main():
    state, p = sys.argv[1], sys.argv[2]
    before = open(p).read()
    s = before

    if state == 'KS-catalog':
        # the product silently stops covering one of the three knobs
        s = sub(s, 'INTFER_XMX_WG:-,16|', '', what='XMX_WG catalogue entry')

    elif state == 'KS-override':
        # the operator cannot narrow the product for a targeted A/B
        s = sub(s, 'KNOB_SPEC="${INTFER_SWEEP_KNOBS:-$DEFAULT_KNOBS}"',
                'KNOB_SPEC="$DEFAULT_KNOBS"', what='KNOB_SPEC override')

    elif state == 'KS-baseline':
        # the reference run is no longer first: the delta is measured against
        # whichever combo happened to run first
        s = sub(s, 'run_combo "BASELINE($BASELINE)" $BASELINE\n', '',
                what='baseline run_combo')
        s = sub(s, '} | tee "$LOGDIR/sweep_table.txt"',
                'run_combo "BASELINE($BASELINE)" $BASELINE\n'
                '} | tee "$LOGDIR/sweep_table.txt"', what='tee anchor')

    elif state == 'KS-scrape':
        # the counters are read from a line that is not the device's own
        # event-timestamped profile line
        s = sub(s, "grep -a '\\[GPU profile @decode\\]'", "grep -a 'GPU profile'",
                what='profile scrape')

    elif state == 'KS-fields':
        # one of the three exact integer counters is silently dropped
        s = sub(s, "    nw)       re='nw=([0-9]+)' ;;\n", '', what='nw field regex')

    elif state == 'KS-floor':
        # a single round has no noise floor to clear, so the delta can never fail
        s = sub(s, '[ "$usable" -ge 2 ]', '[ "$usable" -ge 1 ]', what='usable>=2 gate')

    elif state == 'KS-loadgate':
        # the shipped defect: the load verdict is never read, so a CONTENDED
        # round publishes its profile line and feeds the median and the floor
        s = sub(s, '    if ! loadlog_usable; then\n'
                   '      result_lines["$label"]+="r$r SKIPPED-CONTENDED'
                   ' max_load1=${_LL_MAX:-NA}"$\'\\n\'\n'
                   '      continue\n'
                   '    fi\n', '', what='loadlog_usable guard')
        s = sub(s, '*FAILED*|*NO_PROFILE*|*SKIPPED*)', '*FAILED*|*NO_PROFILE*)',
                n=2, what='both usable filters')

    elif state == 'KS-guardafter':
        # THE DEFECT THE ORDERING PREDICATE EXISTS FOR: the guard is relocated
        # AFTER the line that publishes the round, so a CONTENDED arm's profile
        # line reaches result_lines and feeds the median and the spread floor.
        # Deliberately NOT the same as KS-loadgate -- the guard, the withheld-
        # round record and both filter exclusions all still exist, so only the
        # ordering predicate can refuse this.
        lines = s.split('\n')
        gi = next((i for i, l in enumerate(lines)
                   if l.strip() == 'if ! loadlog_usable; then'), None)
        if gi is None:
            sys.stderr.write('guard block start not found\n')
            sys.exit(3)
        gj = next((i for i in range(gi, len(lines)) if lines[i].strip() == 'fi'),
                  None)
        if gj is None:
            sys.stderr.write('guard block is unterminated\n')
            sys.exit(3)
        block = lines[gi:gj + 1]
        del lines[gi:gj + 1]
        pi = next((i for i, l in enumerate(lines) if 'r$r $prof' in l), None)
        if pi is None:
            sys.stderr.write('publication line not found\n')
            sys.exit(3)
        lines[pi + 1:pi + 1] = block
        s = '\n'.join(lines)

    elif state == 'KS-eval':
        # a product entry becomes a command line rather than a variable
        # assignment, which is what the "exported explicitly, no eval" contract
        # exists to refuse
        s = sub(s, '    for kv in "$@"; do\n      export "$kv"\n    done',
                '    for kv in "$@"; do\n      eval "export $kv"\n    done',
                what='export loop')

    elif state == 'KS-tokps':
        # the sweep decides on a load-sensitive figure
        s = sub(s, 'echo "raw logs: $LOGDIR"',
                'echo "tok/s: $tps"\necho "raw logs: $LOGDIR"', what='raw logs echo')

    elif state == 'KS-precond':
        # the sweep runs on a box that cannot count its devices and reports a
        # SKIPPED arm as a measured one
        s = sub(s, 'gate_require_gpus 2 || exit 2\n', '', what='gate_require_gpus')

    elif state == 'KS-table':
        # the table exists only on the terminal and cannot be re-derived
        s = sub(s, '} | tee "$LOGDIR/sweep_table.txt"', '} | cat', what='tee anchor')

    elif state == 'KS-product':
        # THE SHIPPED DEFECT, restored: KNOB_VALS keeps the raw comma list while
        # walk() reads with `while IFS= read -r v`, a NEWLINE split. Each knob's
        # whole list then collapses to ONE value, PRODUCT holds one arm whose env
        # string is `INTFER_DECODE1_WG=-,16,64,256`, the CLI's atoi() of that is 0
        # (the SHIPPED DEFAULT), and the sweep runs one arm indistinguishable from
        # the baseline -- a baseline-relative table of deltas against itself.
        # Deliberately the ONLY mutation no grep-based predicate can see: the
        # catalogue is intact, the override is intact, the scrape/fields/floor are
        # intact, and both the comma list and `IFS= read` look correct in
        # isolation. p_product_enum is the predicate that RUNS the parse, so it
        # alone must refuse this state.
        s = sub(s, 'KNOB_VALS["$knob"]="$(tr \',\' \'\\n\' <<<"$vals")"',
                'KNOB_VALS["$knob"]="$vals"', what='comma->newline split')

    elif state == 'KS-consumed':
        # A catalogued knob that NOTHING READS -- an inert arm generator. The
        # catalogue keeps all three shipped names (p_knob_catalog stays green) and
        # the product still enumerates: 4x2x2x3 = 24 arms, exactly the count the
        # spec text derives (p_product_enum stays green). The fourth knob exports
        # a value for a flag no source file reads, so all 24 arms run the
        # IDENTICAL shipped default and the sweep prints a baseline-relative table
        # of deltas against itself. Deliberately the only state p_knob_consumed can
        # refuse, and the reason that predicate searches examples/ and include/
        # too: a src/-only version would still catch this state but would ALSO
        # redden the DFlash runbook's knobs, which live in
        # examples/intfer_cli/main.cpp.
        s = sub(s,
                "DEFAULT_KNOBS='INTFER_DECODE1_WG:-,16,64,256|INTFER_XMX_WG:-,16|INTFER_LA_CONV_PREP_WG:-,64'",
                "DEFAULT_KNOBS='INTFER_DECODE1_WG:-,16,64,256|INTFER_XMX_WG:-,16|INTFER_LA_CONV_PREP_WG:-,64|INTFER_NOPE_WG:-,16,256'",
                what='inert catalogue knob')

    else:
        sys.stderr.write('unknown state %s\n' % state)
        sys.exit(2)

    if s == before:
        sys.stderr.write('mutation did not land\n')
        sys.exit(4)
    open(p, 'w').write(s)
    sys.stdout.write('landed\n')


main()
PYMUT
[ -s "$SANDBOX/mut.py" ] || die "could not stage the mutation engine"
# ---- the residue plant, staged so the durable residue check can re-run its
#      own negative direction without touching the live subject
cat > "$SANDBOX/resid_plant.py" <<'PYRES'
#!/usr/bin/env python3
"""Plant the residue negative direction into a COPY of the mutator.

usage: resid_plant.py --strict|--allow-missing SRC DST

The plant is one shell statement -- `echo ... >> "$LIVE"` -- inserted AFTER the
residue anchor comment and BEFORE the `AFTER=` assignment, so it changes the
subject's bytes while it is still between the BEFORE and AFTER measurements. That
position is the whole point: an insert placed after `AFTER=` would be a no-op and
the matrix would report a catch that never happened.

--strict       the anchor is required; absent anchor is a hard error (exit 3).
--allow-missing the anchor may be absent; SRC is then copied VERBATIM (exit 0 and
               the caller proves non-landing with cmp). This is the vacuity arm's
               mode.

Either mode proves it landed by comparing the written bytes to SRC; a silent
no-op is exit 4, never exit 0 -- the #36 vacuous-green shape.
"""

import sys

ANCHOR = "# ---- RESIDUE. The live subject"
# THE WORDING IS LOAD-BEARING, NOT DECORATIVE. The recorded pair's second sha is
# the subject's bytes after THIS comment was appended, so re-wording the plant
# changes the measured pair and the equivalence check refuses -- correctly. Restoring
# the historical wording re-derives the recorded measurement; rewriting the record to
# match a new wording would be moving the durable number to agree with the tool.
PLANT = 'echo "# residue negative-direction plant" >> "$LIVE"'


def main(argv):
    if len(argv) != 4 or argv[1] not in ("--strict", "--allow-missing"):
        sys.stderr.write(
            "usage: resid_plant.py --strict|--allow-missing SRC DST\n")
        return 2
    mode, src, dst = argv[1], argv[2], argv[3]

    with open(src) as fh:
        lines = fh.readlines()

    idx = None
    for i, ln in enumerate(lines):
        if ln.startswith(ANCHOR):
            idx = i
            break

    if idx is None:
        if mode == "--strict":
            sys.stderr.write("PLANT: FAIL -- anchor absent (anchored mutation "
                             "refuses rather than guessing)\n")
            return 3
        with open(dst, "w") as fh:
            fh.writelines(lines)
        sys.stdout.write("PLANT: NOT LANDED (anchor absent, verbatim copy)\n")
        return 0

    out = lines[:idx + 1] + [PLANT + "\n"] + lines[idx + 1:]
    body = "".join(out)

    if body == "".join(lines):
        sys.stderr.write("PLANT: FAIL -- the plant produced identical bytes\n")
        return 4

    with open(dst, "w") as fh:
        fh.write(body)
    sys.stdout.write("PLANT: LANDED after line %d (= residue anchor)\n" % (idx + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PYRES
[ -s "$SANDBOX/resid_plant.py" ] || die "could not stage the residue plant"

# ---- DURABLE RESIDUE EQUIVALENCE -------------------------------------------
# A sha pair recorded in prose is prose until something re-runs the harness that
# produced it and asserts the pair again. This repo's own case: the residue
# negative direction was MEASURED once at 11 states (pair 3af3c2ea... ->
# 4313556...), and the durable records then asserted a 12-state re-measurement
# whose SECOND sha appeared in no artifact anywhere. That is the "a number
# reached a report without an artifact behind it" failure this repo has retracted
# three claims for, sitting in HISTORY.md -- the one file a future session reads
# as truth. So the mutator now re-runs its OWN planted negative direction on a
# COPY and requires the pair it harvests to be the pair the records claim.
#
# FOUR THINGS THIS BLOCK ESTABLISHES AT ONCE:
#
#   * PROSE-EQUIVALENCE, against the documented recording convention rather than
#     by a blind equality. engines_sources.txt is the LIVE-STATE record, so the
#     re-measured pair must be one it lists; HISTORY.md is the CHRONOLOGY, so the
#     re-measured pair must appear in it. A pair that has been superseded may
#     legitimately live on in the chronology, which is why HISTORY is tested for
#     CONTAINMENT and engines_sources for the CURRENT set -- a naive
#     "the two files must be identical" would go red on the first honest revision.
#   * SAFETY. The plant runs against a COPY of the live subject in the sandbox,
#     never the live file. The original 12-state session appended to the live
#     subject and got away with it only because a `cp` backup had been taken
#     first; here the copy is unconditional and the live file is never a target.
#   * STALENESS. The measured BEFORE sha must equal the live subject's sha right
#     now, so a pair measured against a different revision is refused as stale
#     prose rather than accepted as an echo of itself.
#   * A VACUITY HALF. "No refusal harvested" is a FAIL, not a pass: an
#     equivalence check satisfiable by the plant not landing is the #36 vacuous
#     green. The vacuity arm below neutralises the anchor -- so the plant provably
#     does not land, asserted with cmp and not assumed -- runs that copy, and
#     requires NO refusal, which is what proves the positive arm could not have
#     been satisfied by absence.
#
# NO RECURSION. The planted copy is a copy of THIS FILE, so it contains this
# block. KS_RESIDUECHILD makes the child skip it. The positive child dies at the
# residue assertion above before it could reach this point; the sentinel is what
# covers the vacuity arm, where the plant does not land, that assertion passes,
# and the child would otherwise recurse into itself forever.
resid_verdict() { # $1 = child output file; echoes the measured "A -> B" or nothing
  # NOTE ON THE REGEX: it deliberately does NOT require a trailing period. The
  # die() spelling emits no period ("...sha A -> B)"), and a demand for one that
  # the subject never produces is a predicate that can only fail -- the mirror
  # image of #36, and it would have rejected a CORRECT measurement.
  sed -n 's/^KS-MUT: FAIL -- the live subject CHANGED during the matrix (sha \([0-9a-f]\{64\}\) -> \([0-9a-f]\{64\}\)).*$/\1 -> \2/p' "$1" | head -1
}

durable_residue_equivalence() {
  if [ -n "${KS_RESIDUECHILD:-}" ]; then
    echo "KS-MUT: durable residue equivalence SKIPPED in the planted child (sentinel)"
    return 0
  fi

  local self planted child_out subj live_sha measured me_a me_b
  local rec_pairs rec_n hist_pairs vac_mut vac_subj vac_measured
  self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  # Planted INSIDE scripts/ so SCRIPT_DIR/CHECKER resolve: a /tmp plant dies on
  # "checker missing" because SCRIPT_DIR derives from BASH_SOURCE, and never
  # reaches the assertion under test. Cleaned by the EXIT trap.
  planted="$SCRIPT_DIR/zz_ks_resid_plant.sh"
  PLANTED_FOR_CLEANUP="$planted"

  # ---- POSITIVE ARM: re-measure the pair before trusting any prose about it ---
  if ! python3 "$SANDBOX/resid_plant.py" --strict "$self" "$planted"; then
    echo "KS-MUT: durable residue equivalence FAIL -- the residue plant did not LAND"
    return 1
  fi
  if cmp -s "$planted" "$self"; then
    echo "KS-MUT: durable residue equivalence FAIL -- cmp says the plant is a no-op,"
    echo "         and a no-op mutation must be a FAIL, never a catch"
    return 1
  fi
  echo "KS-MUT: residue plant landed ($(wc -l < "$planted") lines vs $(wc -l < "$self"))"

  subj="$SANDBOX/resid_subject.sh"
  cp -f "$LIVE" "$subj"
  KS_RESIDUECHILD=1 bash "$planted" "$subj" >"$SANDBOX/resid_child.out" 2>&1
  child_out="$SANDBOX/resid_child.out"

  measured="$(resid_verdict "$child_out")"
  if [ -z "$measured" ]; then
    echo "KS-MUT: durable residue equivalence FAIL -- the planted child produced NO"
    echo "         refusal. A residue assertion that cannot fail is not a guard, and"
    echo "         an equivalence check satisfiable by absence is the #36 vacuous"
    echo "         green: refusing rather than passing."
    echo "         child output tail:"
    tail -5 "$child_out" | sed 's/^/           /'
    return 1
  fi
  me_a="${measured%% -> *}"
  me_b="${measured##* -> }"

  # The pair must describe THIS subject: its BEFORE half is the live sha. A pair
  # measured against another revision is stale prose.
  live_sha="$(sha256sum "$LIVE" | cut -d' ' -f1)"
  if [ "$me_a" != "$live_sha" ]; then
    echo "KS-MUT: durable residue equivalence FAIL -- the measured BEFORE sha is not"
    echo "         the live subject sha, so the pair measured describes some other"
    echo "         revision of the subject"
    echo "         live:  $live_sha"
    echo "         meas:  $me_a"
    return 1
  fi

  # ---- THE RECORDS, each tested against the convention that governs it -------
  rec_pairs="$(grep -oE '[0-9a-f]{64} -> [0-9a-f]{64}' "$ROOT/engines_sources.txt" 2>/dev/null | sort -u)"
  rec_n="$(printf '%s\n' "$rec_pairs" | grep -c .)"
  if [ "$rec_n" -lt 1 ]; then
    echo "KS-MUT: durable residue equivalence FAIL -- the live-state record"
    echo "         engines_sources.txt carries NO sha pair; a durable equivalence"
    echo "         check with no recorded subject must not read as PASS"
    return 1
  fi
  if ! printf '%s\n' "$rec_pairs" | grep -qxF "$measured"; then
    echo "KS-MUT: durable residue equivalence FAIL -- the re-measured pair is not the"
    echo "         one the LIVE-STATE record (engines_sources.txt) records"
    echo "         measured: $measured"
    echo "         recorded: $(printf '%s' "$rec_pairs" | tr '\n' ' ')"
    return 1
  fi
  # HISTORY.md is the chronology: it KEEPS superseded pairs on purpose, so it is
  # tested for containment. Its pair is line-wrapped and indented, hence the
  # flatten-and-squeeze -- a naive grep finds zero pairs there.
  hist_pairs="$(tr '\n' ' ' < "$ROOT/HISTORY.md" 2>/dev/null | tr -s ' ' \
                | grep -oE '[0-9a-f]{64} -> [0-9a-f]{64}' | sort -u)"
  if ! printf '%s\n' "$hist_pairs" | grep -qxF "$measured"; then
    echo "KS-MUT: durable residue equivalence FAIL -- the re-measured pair appears"
    echo "         nowhere in the chronology (HISTORY.md, compared newline-flattened"
    echo "         and whitespace-squeezed because the pair wraps across indented"
    echo "         lines)"
    echo "         measured: $measured"
    echo "         found:    $(printf '%s' "$hist_pairs" | tr '\n' ' ')"
    return 1
  fi
  echo "KS-MUT: DURABLE RESIDUE EQUIVALENCE: PASS"
  echo "         re-measured against a copy of the live subject, and the same pair"
  echo "         is recorded by engines_sources.txt ($rec_n pair(s)) and by"
  echo "         HISTORY.md ($(printf '%s\n' "$hist_pairs" | grep -c .) pair(s)):"
  echo "           $me_a"
  echo "           $me_b"

  # ---- VACUITY ARM: prove the positive arm cannot be satisfied by absence ----
  # Neutralise the anchor so the plant matches nothing, then cmp-PROVE the plant
  # did not land (asserted, not assumed), run that copy, and require NO refusal.
  vac_mut="$SANDBOX/resid_vacuity.sh"
  sed 's|^# ---- RESIDUE\. The live subject|# ---- RESIDUE-NEUTRALISED-FOR-THE-VACUITY-ARM live subject|' \
      "$self" > "$vac_mut"
  if cmp -s "$vac_mut" "$self"; then
    echo "KS-MUT: durable residue equivalence FAIL -- could not neutralise the anchor"
    echo "         for the vacuity arm; the vacuity half would be vacuous itself"
    return 1
  fi
  if ! python3 "$SANDBOX/resid_plant.py" --allow-missing "$vac_mut" "$planted" >/dev/null 2>&1; then
    echo "KS-MUT: durable residue equivalence FAIL -- the --allow-missing plant errored"
    return 1
  fi
  if ! cmp -s "$planted" "$vac_mut"; then
    echo "KS-MUT: durable residue equivalence FAIL -- the plant LANDED in the"
    echo "         neutralised copy, so the vacuity arm would be running the very"
    echo "         defect it exists to withhold"
    return 1
  fi
  vac_subj="$SANDBOX/resid_vacuity_subject.sh"
  cp -f "$LIVE" "$vac_subj"
  KS_RESIDUECHILD=1 bash "$planted" "$vac_subj" >"$SANDBOX/resid_vacuity.out" 2>&1
  vac_measured="$(resid_verdict "$SANDBOX/resid_vacuity.out")"
  if [ -n "$vac_measured" ]; then
    echo "KS-MUT: durable residue equivalence FAIL -- the vacuity child DID refuse, so"
    echo "         the positive arm does not need its plant to land; that is a"
    echo "         residue assertion satisfiable without its own defect"
    return 1
  fi
  echo "KS-MUT: vacuity arm ok -- the anchor-neutralised child (plant provably absent"
  echo "         by cmp) produced no refusal, so the positive arm above could NOT"
  echo "         have been satisfied by the plant not landing"
  return 0
}


# ---- ANCHOR PREFLIGHT. Every mutation targets an anchor; an anchor absent from
#      the live subject makes that state vacuous by construction.
for a in 'INTFER_XMX_WG:-,16|' \
         'KNOB_SPEC="${INTFER_SWEEP_KNOBS:-$DEFAULT_KNOBS}"' \
         'run_combo "BASELINE($BASELINE)" $BASELINE' \
         'grep -a '\''\[GPU profile @decode\]'\''' \
         '    nw)       re='\''nw=([0-9]+)'\'' ;;' \
         '[ "$usable" -ge 2 ]' \
         'if ! loadlog_usable; then' \
         '    result_lines["$label"]+="r$r $prof"' \
         '*FAILED*|*NO_PROFILE*|*SKIPPED*)' \
         '    for kv in "$@"; do' \
         'echo "raw logs: $LOGDIR"' \
         'gate_require_gpus 2 || exit 2' \
         '} | tee "$LOGDIR/sweep_table.txt"' \
         "DEFAULT_KNOBS='INTFER_DECODE1_WG:-,16,64,256|INTFER_XMX_WG:-,16|INTFER_LA_CONV_PREP_WG:-,64'" \
         "KNOB_VALS[\"\$knob\"]=\"\$(tr ',' '\\n' <<<\"\$vals\")\""; do
  n="$(grep -c -F -- "$a" "$LIVE")"
  if [ "$n" -lt 1 ]; then die "anchor absent from the live subject: $a"; fi
done
echo "KS-MUT: anchor preflight ok (15 anchors)"

# ---- BASELINE. The unmutated subject must PASS the checker, or every "caught"
#      below is indistinguishable from a checker that refuses everything.
bash "$CHECKER" "$LIVE" >"$SANDBOX/baseline.txt" 2>&1
if [ $? -ne 0 ]; then
  cat "$SANDBOX/baseline.txt"
  die "the healthy subject FAILS the checker"
fi
grep -q 'KS-CONTRACT: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
echo "KS-MUT: baseline PASS on the unmutated subject"

BEFORE="$(sha256sum "$LIVE" | cut -d' ' -f1)"

STATES_REAL=0
STATES_CAUGHT=0
EXPECT=14
for st in KS-catalog KS-override KS-baseline KS-scrape KS-fields KS-floor \
          KS-loadgate KS-guardafter KS-eval KS-tokps KS-precond KS-table \
          KS-product KS-consumed; do
  f="$SANDBOX/$st.sh"
  cp -f "$LIVE" "$f"
  if ! python3 "$SANDBOX/mut.py" "$st" "$f" >"$SANDBOX/$st.land" 2>&1; then
    cat "$SANDBOX/$st.land"
    die "$st: the mutation did not land (a no-op must be a FAIL, never a catch)"
  fi
  # THE STANDING FIX, asserted again here: cmp, not the engine's own word for it.
  if cmp -s "$f" "$LIVE"; then
    die "$st: cmp says the mutation did NOT LAND -- refusing to grade a no-op"
  fi
  STATES_REAL=$((STATES_REAL + 1))
  bash "$CHECKER" "$f" >"$SANDBOX/$st.out" 2>&1
  if [ $? -eq 0 ]; then
    echo "KS-MUT: $st NOT CAUGHT -- the checker passed a defective source"
    sed 's/^/    /' "$SANDBOX/$st.out"
  else
    why="$(sed -n 's/^KS-CONTRACT: FAIL -- refused by: //p' "$SANDBOX/$st.out" | head -1)"
    echo "KS-MUT: $st CAUGHT (${why:-no refusal reason harvested})"
    STATES_CAUGHT=$((STATES_CAUGHT + 1))
  fi
done

if [ "$STATES_REAL" -ne "$EXPECT" ]; then
  die "expected $EXPECT states, graded $STATES_REAL -- a matrix that never ran cannot report a tally"
fi

# ---- RESIDUE. The live subject must be byte-identical to its pre-run state.
AFTER="$(sha256sum "$LIVE" | cut -d' ' -f1)"
if [ "$AFTER" != "$BEFORE" ]; then
  die "the live subject CHANGED during the matrix (sha $BEFORE -> $AFTER)"
fi
# [SUPERSEDED 2026-10-18 by the operator's commit 530f61f, recorded not deleted]
# This comment used to read: knob_sweep.sh is UNTRACKED in this repo's standing
# state (registered in run_all_checks.sh but never committed, per the standing
# no-commit rule), so git status reports it as '??'; that is expected and filtered.
# That was true when written and is no longer true: the operator committed the
# whole harness -- checkers, mutators and knob_sweep.sh itself -- at 530f61f while
# this session was running, so the file is TRACKED at HEAD and the filter below is
# a no-op. It is kept anyway, because it masks nothing: a modified TRACKED file
# reports ' M' and never '??', so an actual working-tree edit still fires. The sha
# assertion above remains the real residue check -- a byte comparison, not a status
# reading, which is why this check did not notice the commit that made it stale.
if [ -n "$(cd "$ROOT" && git status --porcelain -- scripts/knob_sweep.sh 2>/dev/null | grep -v '^??')" ]; then
  die "git reports a TRACKED-file change under scripts/knob_sweep.sh"
fi
echo "KS-MUT: residue byte-identical (sha256 ${AFTER:0:16}...)"

# ---- DURABLE RESIDUE EQUIVALENCE. Byte-identity proves the matrix left the
#      subject alone; it says nothing about the sha pair the durable records
#      CLAIM for that same run. Those two facts are only interesting together,
#      and the records are prose until this runs.
durable_residue_equivalence
RESID_EQ_RC=$?

echo "KS-MUT: matrix $STATES_CAUGHT/$STATES_REAL states caught"
echo "KS-MUT: durable residue equivalence rc=$RESID_EQ_RC"
if [ "$STATES_CAUGHT" -eq "$STATES_REAL" ] && [ "$RESID_EQ_RC" -eq 0 ]; then
  echo "KS-CONTRACT MATRIX: PASS ($STATES_CAUGHT/$STATES_REAL, LANDED asserted per state, residue byte-identical, recorded pair re-measured)"
  echo "RESULT: PASS"
  exit 0
fi
echo "KS-CONTRACT MATRIX: FAIL"
echo "RESULT: FAIL"
exit 1
