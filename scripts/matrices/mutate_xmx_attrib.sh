#!/usr/bin/env bash
# mutate_xmx_attrib.sh -- the nine-state negative direction for
# check_xmx_attrib.sh. A predicate that can only pass is not a predicate.
#
# WHY NINE STATES AND NOT FEWER. Three of the six predicates are conjunctions over
# derived counts (p_dump_universality, p_site_coverage), and a conjunction is
# exactly where a vacuous green hides: it passes if the count it compares is
# empty on BOTH sides. Every state below therefore plants a defect that is
# observable only through its own predicate, and each asserts it LANDED before
# it is graded.
#
# TWO SUBJECTS, DELIBERATELY. The line is BUILT in src/engine/engine.cpp and
# PRINTED from examples/intfer_cli/main.cpp, so the checker takes both. Seven
# states mutate the builder and one (XA-dump) mutates the printer, because the
# defect it plants -- a dump site with no attribution line -- only exists in the
# printer. That state is not decorative: it is the shape of a defect that was
# really present on 2026-10-16, when five [GPU profile] markers carried four
# xmx_attrib_line calls.
#
# THE STATES, each named for the predicate it must trip:
#   XA-line      delete the [xmx-attrib] construction    -> p_line
#   XA-engines   spell the verdict field matrix_engine= -> p_matrix_engines
#   XA-triple    drop the declined= segment             -> p_triple
#   XA-absence   make the default verdict "active"      -> p_named_absence
#   XA-dump      delete ONE printer site's line         -> p_dump_universality
#   XA-site      stop one dispatch attempt being noted  -> p_site_coverage
#   XA-numverdict feed matrix_engines= a COUNT, not a  -> p_verdict_is_a_word
#                 selected word (the %u-style fallback)
#   XA-intverdict the verdict's type is int, not char*  -> p_verdict_is_a_word
#   XA-xmxmon    the verdict's inputs read the env        -> p_xmxmon_independence
#
# THE THREE NEWEST STATES EACH PIN A DIFFERENT HALF OF THE VERDICT CONTRACT.
# XA-numverdict and XA-intverdict both trip p_verdict_is_a_word but break
# different things (the appended VALUE's shape vs the VARIABLE's type);
# XA-xmxmon trips p_xmxmon_independence and no other predicate, which is the
# state that proves the census gap above is closed and not merely noted.
#
# THE TWO NEWEST STATES PIN ONE PREDICATE'S TWO HALVES DELIBERATELY. They both
# trip p_verdict_is_a_word, and neither substitutes for the other: XA-numverdict
# breaks the appended VALUE's shape, XA-intverdict breaks the VARIABLE's type.
# A matrix that graded only the first would certify half a predicate, which is
# the same failure as certifying a flag no gate can reach.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# THE TWO SUBJECTS ARE OVERRIDABLE, and that is what makes the durable residue
# hook possible at all: the positive arm plants a copy of THIS FILE inside
# scripts/ and runs it against COPIES of both subjects, so a mutation that
# "leaks" lands in the sandbox and never in engine.cpp or main.cpp. Without
# the override the child would edit the live sources, which is the failure the
# residue assertion exists to catch -- measured against a live file.
SRC="${1:-$ROOT/src/engine/engine.cpp}"
DUMP="${2:-$ROOT/examples/intfer_cli/main.cpp}"
CHECKER="$ROOT/scripts/checks/check_xmx_attrib.sh"

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/xamut.XXXXXX")"
# The durable-residue block plants a copy of THIS file inside scripts/ so the
# child's SCRIPT_DIR/CHECKER resolve, and it removes that plant on every path.
# A FAIL mid-flight would otherwise leak the plant into scripts/ -- the exact
# stray file the residue check exists to notice -- so the trap is the backstop.
PLANTED_FOR_CLEANUP=""
trap 'rm -rf "$SANDBOX"; [ -z "${PLANTED_FOR_CLEANUP:-}" ] || rm -f "$PLANTED_FOR_CLEANUP"' EXIT
# INT/TERM must ALSO reach the cleanup: an EXIT trap alone never runs on a signal
# (bash fires EXIT only at a reached end or an explicit exit), so a SIGTERM killed
# the script outright and leaked the SANDBOX -- and, worse, leaked any planted copy
# still sitting inside scripts/, which is exactly the stray file the residue check
# exists to notice (AUDIT #29). The exit routes the signal back through the EXIT
# handler; "clean up and resume" leaves the process alive, which is strictly worse
# (check_trap_release.sh).
trap 'rm -rf "$SANDBOX"; [ -z "${PLANTED_FOR_CLEANUP:-}" ] || rm -f "$PLANTED_FOR_CLEANUP"; exit 130' INT
trap 'rm -rf "$SANDBOX"; [ -z "${PLANTED_FOR_CLEANUP:-}" ] || rm -f "$PLANTED_FOR_CLEANUP"; exit 143' TERM

die() { echo "XA-MUT: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }

[ -s "$SRC" ] || die "builder subject missing ($SRC)"
[ -s "$DUMP" ] || die "printer subject missing ($DUMP)"
[ -x "$CHECKER" ] || [ -f "$CHECKER" ] || die "checker missing ($CHECKER)"

# ---- ANCHOR PREFLIGHT. Every mutation targets an anchor that must be present
#      in the live source exactly as expected, or the state is vacuous by
#      construction. Counts are asserted, not just presence: XA-triple and
#      XA-site target text that occurs several times.
for a in 'std::string s = "[xmx-attrib] attempts=";' \
         's += " matrix_engines=";' \
         's += " declined=";' \
         'const char* verdict = "not-requested";'; do
  n="$(grep -c -F -- "$a" "$SRC")"
  if [ "$n" -ne 1 ]; then die "anchor absent or not unique in engine.cpp ($n): $a"; fi
done
n_call="$(grep -c -F 'std::fprintf(stderr, "%s\n", xmx_attrib_line(p).c_str());' "$DUMP")"
if [ "$n_call" -lt 2 ]; then
  die "only $n_call printer call(s); XA-dump needs to delete one and still leave a subject"
fi

cat > "$SANDBOX/mut.py" <<'PYMUT'
import sys


def sub(s, old, new, n=1, what=''):
    if s.count(old) != n:
        sys.stderr.write('anchor %s count=%d want=%d\n' % (what or old[:48], s.count(old), n))
        sys.exit(3)
    return s.replace(old, new, 1)


def main():
    state, p = sys.argv[1], sys.argv[2]
    before = open(p).read()
    s = before

    if state == 'XA-line':
        # the line's own construction goes away, leaving only the prose that
        # DESCRIBES it -- the exact shape a comment-stripping checker exists for
        s = sub(s, 'std::string s = "[xmx-attrib] attempts=";',
                '// (the attribution line was removed)', what='attrib line construction')

    elif state == 'XA-engines':
        # A near-miss spelling, NOT a deletion. The three integer counters still
        # print, so the line still looks populated; the operator is simply left
        # to draw the verdict themselves from three numbers.
        s = sub(s, 's += " matrix_engines=";', 's += " matrix_engine=";',
                what='matrix_engines field')

    elif state == 'XA-triple':
        # the decline half disappears, so "requested, every tile declined" is
        # indistinguishable from a launch count that happens to be zero
        s = sub(s, 's += " declined=";\n', '', what='declined= segment')

    elif state == 'XA-absence':
        # THE INVERSION. The absence is no longer NAMED: a run that never
        # requested the matrix engines would report itself as active. This is
        # the state that proves the predicate is not just "a string exists".
        s = sub(s, 'const char* verdict = "not-requested";',
                'const char* verdict = "active";', what='default verdict')

    elif state == 'XA-dump':
        call = '    std::fprintf(stderr, "%s\\n", xmx_attrib_line(p).c_str());\n'
        # NB: the source's own \n inside the printf format is TWO characters
        # (backslash, n) in the file, and the statement ends with a real
        # newline -- hence \\n for the first and \n for the second.
        # delete exactly ONE of the printer sites, so the count mismatch is
        # observable without emptying the population (an empty population would
        # satisfy an anti-vacuity guard for the wrong reason). The population
        # is DERIVED rather than a literal, so a printer site added later
        # updates this state instead of failing a hard-coded anchor.
        npop = s.count(call)
        if npop < 2:
            sys.stderr.write('need at least 2 printer calls to delete one and '
                             'leave a non-empty population, found %d\n' % npop)
            sys.exit(3)
        s = sub(s, call, '', n=npop, what='one printer call site of %d' % npop)

    elif state == 'XA-site':
        # one XMX dispatch attempt stops being noted: the attempt counter is
        # short by one for the rest of the process, and the three-way verdict
        # silently under-reports. p_site_coverage compares noted sites against
        # the attempt-flag population, so only it can refuse this. The
        # population is DERIVED for the same reason XA-dump's is: the identical
        # `xmx_note_dispatch(xmx_ran);` arm appears once per call site, and a
        # hand-written count here would turn a new dispatch seam into a silent
        # "mutation did not land" instead of a graded state.
        call = 'xmx_note_dispatch(xmx_ran);'
        npop = s.count(call)
        if npop < 1:
            sys.stderr.write('no noted dispatch site found\n')
            sys.exit(3)
        s = sub(s, call, '', n=npop, what='one noted dispatch site of %d' % npop)

    elif state == 'XA-numverdict':
        # THE DEFECT p_verdict_is_a_word EXISTS FOR, and it is the shape the
        # predicate was written for: the verdict stops being SELECTED and
        # becomes a COUNT. `matrix_engines=` is fed the launch count, so a run
        # that never requested the matrix engines prints matrix_engines=0 and
        # a reader cannot tell "idle" from "not measured" -- the exact vacuous
        # green p_named_absence cannot see, because all three word literals
        # still sit in the file beside it and it only tests that they EXIST.
        # Only the value's SHAPE can refuse it: the statement after the append
        # is no longer a bare identifier.
        s = sub(s, 's += verdict;',
                's += std::to_string(p.xmx_launch_count);',
                what='the verdict append (numeric fallback)')

    elif state == 'XA-intverdict':
        # The SAME defect in the other spelling: the verdict stays a bare
        # identifier, so the identifier-shape half passes, but its type is an
        # `int` rather than a string. This pins the declaration half of the
        # predicate -- without it, that half is asserted but never proven, and
        # a predicate half nobody can demonstrate failing is a half that has
        # quietly stopped being tested.
        # WHY THE SPARE VARIABLE. The first draft planted `int verdict = 0;`
        # alone, which DELETES the "not-requested" literal from the code, and
        # p_named_absence then caught the state too -- so it certified the
        # shape/type half correctly but left the "only its own predicate can
        # refuse it" property unproven, which is the whole point of the state.
        # MEASURED, not anticipated: the run printed
        # `XA-intverdict CAUGHT (p_named_absence p_verdict_is_a_word)`. Moving
        # the literal onto a spare variable keeps all three words present and
        # leaves the type defect as the ONLY thing that changed about the
        # verdict the line emits.
        s = sub(s, 'const char* verdict = "not-requested";',
                'int verdict = 0; const char* verdict_spare = "not-requested";',
                what='the verdict declaration (numeric type)')

    elif state == 'XA-xmxmon':
        # THE INTEGRATION GAP, planted so the new predicate has a negative
        # direction. The verdict becomes read-only-when-xmxmon-is-installed:
        # it is downgraded to "active" whenever the env var is set. This box
        # runs dev.xe.observation_paranoid=1 with no sudo, so xmxmon is NOT
        # installed, and the three-word verdict AGENTS.md's "matrix engines
        # are idle" claim depends on would silently become unreadable -- which
        # is exactly the shape check_flag_gate_coverage.sh cannot see, because
        # its census regex is scoped to getenv("INTFER_*") and XMXMON has no
        # such prefix. Every other predicate stays green: all three words are
        # still present and still assigned to the same const char* variable,
        # so ONLY p_xmxmon_independence can refuse this state.
        anchor = '    std::string s = "[xmx-attrib] attempts=";'
        s = sub(s, anchor,
                '    if (std::getenv("XMXMON")) verdict = "active";\n' + anchor,
                what='the verdict region (xmxmon-dependent)')

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
#      own negative direction ------------------------------------------------
cat > "$SANDBOX/resid_plant.py" <<'PYRES'
"""Source mutation for mutate_xmx_attrib.sh's durable residue equivalence.

Either mode proves it landed by comparing the written bytes to SRC; a silent
no-op is an error, never a clean exit -- the #36 vacuous-green shape.
argv: mode, src, dst.  mode --strict refuses an absent anchor; --allow-missing
copies verbatim so the caller can prove, with cmp, that nothing was planted.
"""
import sys

ANCHOR = "# ---- RESIDUE. Both live subjects"
# THE WORDING IS LOAD-BEARING, NOT DECORATIVE. The recorded pair's second sha is
# the subject's bytes after THIS line was appended, so re-wording the plant
# changes the measured pair and the equivalence check refuses -- correctly.
# Restoring the historical wording re-derives the recorded measurement;
# rewriting the record to match a new wording would be moving the durable
# number to agree with the tool.
PLANT = 'echo "# residue negative-direction plant" >> "$SRC"'


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
# produced it and asserts the pair again, and this matrix is the repo's own case
# for why: it reports per-run "residue byte-identical" for BOTH subjects and
# recorded NO pair anywhere, so nothing outside the run could ever re-derive it.
# This block is the same mechanism mutate_knob_sweep_contract.sh carries,
# adapted to TWO subjects, and it establishes four things at once:
#
#   * PROSE-EQUIVALENCE against the documented recording convention.
#     docs/engines/intfer.txt is the LIVE-STATE record, so the re-measured pair must
#     be one it lists; HISTORY.md is the CHRONOLOGY and keeps superseded pairs,
#     so it is tested for CONTAINMENT. A naive "the two files must be identical"
#     would go red on the first honest revision.
#   * SAFETY. The plant runs against COPIES of both subjects in the sandbox,
#     never the live files, because SRC/DUMP are overridable (see the head of
#     this file).
#   * STALENESS. The measured BEFORE sha must equal the live BUILDER's sha right
#     now, so a pair measured against another revision is refused as stale.
#   * A VACUITY HALF. "No refusal harvested" is a FAIL, not a pass: an
#     equivalence check satisfiable by the plant not landing is the #36 vacuous
#     green.
#
# NO RECURSION. The planted copy is a copy of THIS FILE, so it contains this
# block. XA_RESIDUECHILD makes the child skip it: the positive child dies at the
# residue assertion above before reaching this point, and the sentinel covers
# the vacuity arm, where the plant does not land, that assertion passes, and the
# child would otherwise recurse into itself forever.
resid_verdict() { # $1 = child output file; echoes the measured "A -> B" or nothing
  # NOTE ON THE REGEX: it deliberately does NOT require a trailing period. The
  # die() spelling emits no period, and a demand for one the subject never
  # produces is a predicate that can only fail -- the mirror image of #36, and
  # it would have rejected a CORRECT measurement.
  sed -n 's/^XA-MUT: FAIL -- engine\.cpp CHANGED during the matrix (sha \([0-9a-f]\{64\}\) -> \([0-9a-f]\{64\}\))$/\1 -> \2/p' "$1" | head -1
}

durable_residue_equivalence() {
  if [ -n "${XA_RESIDUECHILD:-}" ]; then
    echo "XA-MUT: durable residue equivalence SKIPPED in the planted child (sentinel)"
    return 0
  fi

  local self planted child_out subj subj_d live_sha measured me_a me_b
  local rec_pairs rec_n hist_pairs vac_mut vac_subj vac_dump vac_measured
  self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  # Planted INSIDE scripts/ so SCRIPT_DIR/CHECKER resolve: a /tmp plant dies on
  # "checker missing" because SCRIPT_DIR derives from BASH_SOURCE, and never
  # reaches the assertion under test. Cleaned by the EXIT trap.
  planted="$SCRIPT_DIR/zz_xa_resid_plant.sh"
  PLANTED_FOR_CLEANUP="$planted"

  # ---- POSITIVE ARM: re-measure the pair before trusting any prose about it ---
  if ! python3 "$SANDBOX/resid_plant.py" --strict "$self" "$planted"; then
    echo "XA-MUT: durable residue equivalence FAIL -- the residue plant did not LAND"
    return 1
  fi
  if cmp -s "$planted" "$self"; then
    echo "XA-MUT: durable residue equivalence FAIL -- cmp says the plant is a no-op,"
    echo "         and a no-op mutation must be a FAIL, never a catch"
    return 1
  fi
  echo "XA-MUT: residue plant landed ($(wc -l < "$planted") lines vs $(wc -l < "$self"))"

  # BOTH subjects are copied, because the child asserts both and the PRINTER's
  # own residue assertion would otherwise fire on its own and mask the pair.
  subj="$SANDBOX/resid_subject.cpp"
  subj_d="$SANDBOX/resid_printer.cpp"
  cp -f "$SRC" "$subj"
  cp -f "$DUMP" "$subj_d"
  XA_RESIDUECHILD=1 bash "$planted" "$subj" "$subj_d" >"$SANDBOX/resid_child.out" 2>&1
  child_out="$SANDBOX/resid_child.out"

  measured="$(resid_verdict "$child_out")"
  if [ -z "$measured" ]; then
    echo "XA-MUT: durable residue equivalence FAIL -- the planted child produced NO"
    echo "         refusal. A residue assertion that cannot fail is not a guard, and"
    echo "         an equivalence check satisfiable by absence is the #36 vacuous"
    echo "         green: refusing rather than passing."
    echo "         child output tail:"
    tail -5 "$child_out" | sed 's/^/           /'
    return 1
  fi
  me_a="${measured%% -> *}"
  me_b="${measured##* -> }"

  # The pair must describe THIS subject: its BEFORE half is the BUILDER's live
  # sha. A pair measured against another revision is stale prose.
  live_sha="$(sha256sum "$SRC" | cut -d' ' -f1)"
  if [ "$me_a" != "$live_sha" ]; then
    echo "XA-MUT: durable residue equivalence FAIL -- the measured BEFORE sha is not"
    echo "         the live builder sha, so the pair measured describes some other"
    echo "         revision of the subject"
    echo "         live:  $live_sha"
    echo "         meas:  $me_a"
    return 1
  fi

  # ---- THE RECORDS, each tested against the convention that governs it -------
  # SUBJECT_READS names the durable corpus this matrix's assertions read. An
  # undeclared read is invisible to an ordering check, so the re-run-after-
  # the-edits rule stays a prose claim until the read is named here.
  SUBJECT_READS="docs/engines/"
  rec_pairs="$(grep -oE '[0-9a-f]{64} -> [0-9a-f]{64}' "$ROOT/docs/engines/intfer.txt" 2>/dev/null | sort -u)"
  rec_n="$(printf '%s\n' "$rec_pairs" | grep -c .)"
  if [ "$rec_n" -lt 1 ]; then
    echo "XA-MUT: durable residue equivalence FAIL -- the durable record"
    echo "         docs/engines/intfer.txt carries NO sha pair; a durable equivalence"
    echo "         check with no recorded subject must not read as PASS"
    return 1
  fi
  if ! printf '%s\n' "$rec_pairs" | grep -qxF "$measured"; then
    echo "XA-MUT: durable residue equivalence FAIL -- the re-measured pair is not the"
    echo "         one the durable record (docs/engines/intfer.txt) records"
    echo "         measured: $measured"
    echo "         recorded: $(printf '%s' "$rec_pairs" | tr '\n' ' ')"
    return 1
  fi
  # HISTORY.md is the chronology: it KEEPS superseded pairs on purpose, so it is
  # tested for containment. Its pairs wrap across indented lines, hence the
  # flatten-and-squeeze -- a naive grep finds zero pairs there.
  hist_pairs="$(tr '\n' ' ' < "$ROOT/HISTORY.md" 2>/dev/null | tr -s ' ' \
                | grep -oE '[0-9a-f]{64} -> [0-9a-f]{64}' | sort -u)"
  if ! printf '%s\n' "$hist_pairs" | grep -qxF "$measured"; then
    echo "XA-MUT: durable residue equivalence FAIL -- the re-measured pair appears"
    echo "         nowhere in the chronology (HISTORY.md, compared newline-flattened"
    echo "         and whitespace-squeezed because the pair wraps across indented"
    echo "         lines)"
    echo "         measured: $measured"
    echo "         found:    $(printf '%s' "$hist_pairs" | tr '\n' ' ')"
    return 1
  fi
  echo "XA-MUT: DURABLE RESIDUE EQUIVALENCE: PASS"
  echo "         re-measured against copies of the live subjects, and the same pair"
  echo "         is recorded by docs/engines/intfer.txt ($rec_n pair(s)) and by"
  echo "         HISTORY.md ($(printf '%s\n' "$hist_pairs" | grep -c .) pair(s)):"
  echo "           $me_a"
  echo "           $me_b"

  # ---- VACUITY ARM: prove the positive arm cannot be satisfied by absence ----
  # Neutralise the anchor so the plant matches nothing, then cmp-PROVE the plant
  # did not land (asserted, not assumed), run that copy, and require NO refusal.
  vac_mut="$SANDBOX/resid_vacuity.sh"
  sed 's|^# ---- RESIDUE\. Both live subjects|# ---- RESIDUE-NEUTRALISED-FOR-THE-VACUITY-ARM live subjects|' \
      "$self" >"$vac_mut"
  if cmp -s "$vac_mut" "$self"; then
    echo "XA-MUT: durable residue equivalence FAIL -- could not neutralise the anchor"
    echo "         for the vacuity arm; the vacuity half would be vacuous itself"
    return 1
  fi
  if ! python3 "$SANDBOX/resid_plant.py" --allow-missing "$vac_mut" "$planted" >/dev/null 2>&1; then
    echo "XA-MUT: durable residue equivalence FAIL -- the --allow-missing plant errored"
    return 1
  fi
  if ! cmp -s "$planted" "$vac_mut"; then
    echo "XA-MUT: durable residue equivalence FAIL -- the plant LANDED in the"
    echo "         neutralised copy, so the vacuity arm would be running the very"
    echo "         defect it exists to withhold"
    return 1
  fi
  vac_subj="$SANDBOX/resid_vacuity_subject.cpp"
  vac_dump="$SANDBOX/resid_vacuity_printer.cpp"
  cp -f "$SRC" "$vac_subj"
  cp -f "$DUMP" "$vac_dump"
  XA_RESIDUECHILD=1 bash "$planted" "$vac_subj" "$vac_dump" >"$SANDBOX/resid_vacuity.out" 2>&1
  vac_measured="$(resid_verdict "$SANDBOX/resid_vacuity.out")"
  if [ -n "$vac_measured" ]; then
    echo "XA-MUT: durable residue equivalence FAIL -- the vacuity child DID refuse, so"
    echo "         the positive arm does not need its plant to land; that is a"
    echo "         residue assertion satisfiable without its own defect"
    return 1
  fi
  echo "XA-MUT: vacuity arm ok -- the anchor-neutralised child (plant provably absent"
  echo "         by cmp) produced no refusal, so the positive arm above could NOT"
  echo "         have been satisfied by the plant not landing"
  return 0
}

# ---- BASELINE. Both subjects unmutated must PASS the checker, or every "caught"
#      below is indistinguishable from a checker that refuses everything.
bash "$CHECKER" "$SRC" "$DUMP" >"$SANDBOX/baseline.txt" 2>&1
if [ $? -ne 0 ]; then
  cat "$SANDBOX/baseline.txt"
  die "the healthy subjects FAIL the checker -- the negative direction is meaningless"
fi
grep -q 'XA-ATTRIB: PASS' "$SANDBOX/baseline.txt" || die "baseline did not report PASS"
echo "XA-MUT: baseline PASS on the unmutated subjects"

B_SRC="$(sha256sum "$SRC" | cut -d' ' -f1)"
B_DUMP="$(sha256sum "$DUMP" | cut -d' ' -f1)"

STATES_REAL=0
STATES_CAUGHT=0
# XA-dump mutates the PRINTER, so its subject order is swapped for the checker.
for st in XA-line XA-engines XA-triple XA-absence XA-dump XA-site XA-numverdict XA-intverdict XA-xmxmon; do
  if [ "$st" = "XA-dump" ]; then
    f="$SANDBOX/$st.main.cpp"; cp -f "$DUMP" "$f"; subj="$SRC"; dump="$f"
  else
    f="$SANDBOX/$st.engine.cpp"; cp -f "$SRC" "$f"; subj="$f"; dump="$DUMP"
  fi
  if ! python3 "$SANDBOX/mut.py" "$st" "$f" >"$SANDBOX/$st.land" 2>&1; then
    cat "$SANDBOX/$st.land"
    die "$st: the mutation did not land (a no-op must be a FAIL, never a catch)"
  fi
  # cmp, not the mutator's own word for it.
  if [ "$st" = "XA-dump" ]; then cmp -s "$f" "$DUMP" && die "$st: cmp says the mutation did NOT LAND"
  else cmp -s "$f" "$SRC" && die "$st: cmp says the mutation did NOT LAND"; fi
  STATES_REAL=$((STATES_REAL + 1))
  bash "$CHECKER" "$subj" "$dump" >"$SANDBOX/$st.out" 2>&1
  if [ $? -eq 0 ]; then
    echo "XA-MUT: $st NOT CAUGHT -- the checker passed a defective source"
    sed 's/^/    /' "$SANDBOX/$st.out"
    continue
  fi
  why="$(sed -n 's/^XA-ATTRIB: FAIL -- //p' "$SANDBOX/$st.out" | head -1)"
  which="$(sed -n 's/^XA-ATTRIB: FAIL -- refused by: //p' "$SANDBOX/$st.out" | head -1)"
  echo "XA-MUT: $st CAUGHT (${which:-no predicate named} -- ${why:-no reason harvested})"
  STATES_CAUGHT=$((STATES_CAUGHT + 1))
done

if [ "$STATES_REAL" -ne 9 ]; then
  die "expected 9 states, graded $STATES_REAL -- a matrix that never ran cannot report a tally"
fi

# ---- RESIDUE. Both live subjects must be byte-identical to their pre-run state.
A_SRC="$(sha256sum "$SRC" | cut -d' ' -f1)"
A_DUMP="$(sha256sum "$DUMP" | cut -d' ' -f1)"
[ "$A_SRC" = "$B_SRC" ] || die "engine.cpp CHANGED during the matrix (sha $B_SRC -> $A_SRC)"
[ "$A_DUMP" = "$B_DUMP" ] || die "main.cpp CHANGED during the matrix (sha $B_DUMP -> $A_DUMP)"
# NO git-status RESIDUE CHECK HERE, AND THE REASON IS RECORDED RATHER THAN
# OMITTED. mutate_s1_predicates.sh asserts `git status --porcelain` is clean for
# its subject; that check is a STATUS reading, and this repo's standing state is
# "9 stashes untouched, nothing committed" -- so both subjects here report ' M'
# while the matrix runs and a status check would die on every run over the
# session's own uncommitted work. The sha comparison above is the real residue
# check (a byte comparison, not a status reading), which is exactly the
# distinction mutate_knob_sweep_contract.sh records at its own [SUPERSEDED]
# note, where a status check "did not notice the commit that made it stale".
# The one thing a status check adds that sha does not -- a file modified by
# SOME OTHER process during the run -- is not a resolvable claim here.
echo "XA-MUT: residue byte-identical (engine ${A_SRC:0:16}..., printer ${A_DUMP:0:16}...)"

# ---- DURABLE RESIDUE EQUIVALENCE. Byte-identity proves the matrix left both
#      live subjects alone; it says nothing about a sha pair the durable records
#      CLAIM for that same run. Those two facts are only interesting together,
#      and the records are prose until this runs.
durable_residue_equivalence
RESID_EQ_RC=$?
echo "XA-MUT: durable residue equivalence rc=$RESID_EQ_RC"

echo "XA-MUT: matrix $STATES_CAUGHT/$STATES_REAL states caught"
# THE DURABLE HALF GATES THE VERDICT. It is not decoration: without it the
# matrix printed `durable residue equivalence rc=1` on one line and
# `RESULT: PASS` on the next, which is the same shape as the test_tp2_27b.sh
# defect this repo already recorded -- a gate that reports PASS with a failing
# arm. MEASURED while wiring this block: the run before this line existed would
# have done exactly that.
if [ "$STATES_CAUGHT" -eq "$STATES_REAL" ] && [ "$RESID_EQ_RC" -eq 0 ]; then
  echo "XA-ATTRIB MATRIX: PASS ($STATES_CAUGHT/$STATES_REAL, LANDED asserted per state, residue byte-identical, durable pair re-measured)"
  echo "RESULT: PASS"
  exit 0
fi
if [ "$STATES_CAUGHT" -eq "$STATES_REAL" ]; then
  echo "XA-ATTRIB MATRIX: FAIL"
  echo "RESULT: FAIL"
  echo "the state matrix is green but the durable residue equivalence is not"
  echo "(rc=$RESID_EQ_RC); a run whose recorded pair cannot be re-derived is not"
  echo "a PASS, and reporting one would be exactly what check_durable_claims.sh"
  echo "exists to refuse."
  exit 1
fi
echo "XA-ATTRIB MATRIX: FAIL"
echo "RESULT: FAIL"
exit 1
