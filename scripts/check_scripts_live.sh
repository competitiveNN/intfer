#!/usr/bin/env bash
# Every script in scripts/ must be REACHABLE from something live. A script nothing
# runs, nothing dispatches and no entry point names is dead weight that reads like
# tooling, and the next session cannot tell it apart from the real thing.
#
# WHY THIS EXISTS. MEASURED on 2026-10-05, not predicted. Deriving the inbound
# references of all 120 entries of scripts/ with `git grep` and classifying each
# hit found SIXTEEN with zero code references -- nine of them a superseded
# generation of one-shot 27B quantize/merge drivers, four wall-time A/B harnesses
# for experiments AGENTS.md already records as dead ends or withdrew, and three
# probes for the one-GPU driver window. Every one of them was cited by `docs/` or
# `reports/` prose, which is exactly what made them survive: a doc that names a
# file is not evidence that anything calls it.
#
# The cleanup is the easy half. The half that matters is that scripts/ does NOT
# drift back to 120 entries, so the rule has to be DERIVED and this has to be a
# step. A hand-maintained "these are fine" list would be the copy-drift family
# (audit #3/#12) with a brand-new population to rot, and would also be the one
# thing that could make this checker report green forever.
#
# WHAT COUNTS AS A REFERENCE. Two classes, both derived from the tree:
#
#   NAMED  the filename appears verbatim in a LIVE SOURCE. Live sources are every
#          text file in the tree EXCEPT `docs/` and `scripts/archive/`.
#          Both are records, not entry points: `docs/audit-log-tooling.md` is a
#          verbatim audit log and `scripts/archive/` is the retired set. Treating a
#          citation in either as a call site is the mistake that hid all sixteen.
#
#   GLOB   the filename matches a glob that a live runner EXPANDS, so it runs
#          without being named anywhere. Exactly one family does this: the
#          correctness gates, which `run_ten_gates.sh` runs by expanding
#          `GATE_GLOB`. The glob is PARSED OUT OF THE RUNNER, not restated here,
#          so renaming the runner's default retires this class automatically.
#          Without it `test_gguf_roundtrip.sh` reads as dead, which would be wrong:
#          `ls scripts/test_*.sh` is the gate population and nothing names a gate.
#
# THE LIMIT, STATED BECAUSE IT IS THE POINT. A COMMENT inside a live source counts
# as a NAMED reference, and that is a real miss: during this cleanup one live
# harness cited a retired one purely as an example of a `set -u` idiom, and that
# single comment was the ONLY thing keeping the retired harness out of the archive.
# Stripping comments out of shell reliably is not possible (a `#` inside a string
# is not a comment), and every other checker here that tried drew the line
# somewhere arbitrary. So the failure mode of this rule is a MISS -- a retired
# script that survives on a stale comment -- never a false alarm on a live one. A
# weaker check that cannot be argued with is the right trade; a rule that fires on
# live scripts gets switched off, and a switched-off check is worse than none.
#
# WHICH IS WHY THIS HEADER NAMES NO REAL SCRIPT. A checker that cites the files it
# judges is a checker that can mark them reachable by saying so: on its first run
# the gate class below was satisfied entirely by this comment, which is the rule
# certifying itself. The examples are described, not named.
#
# Read-only, no GPU, no build, no model.
# Usage: scripts/check_scripts_live.sh [--scan-only]
#   --scan-only  report the classification and exit, skipping the planted
#                self-checks. Used by scripts/mutate_scripts_live.sh to drive a
#                synthetic tree; a scan of a synthetic tree has nothing to plant.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2
SCAN_ONLY=0
[ "${1:-}" = "--scan-only" ] && SCAN_ONLY=1

# Seams, both used by scripts/mutate_scripts_live.sh to drive a synthetic tree.
# INTFER_SCRIPTS_DIR is the SAME seam check_guard_identity.sh already uses, so
# there is one name for "the corpus" in this directory.
SCRIPTS="${INTFER_SCRIPTS_DIR:-$SCRIPT_DIR}"
SL_ROOT="${INTFER_SL_ROOT:-$(cd "$SCRIPTS/.." && pwd)}"

# shellcheck source=scripts/_gate_pred.sh
. "$SCRIPT_DIR/_gate_pred.sh"
# shellcheck source=scripts/_gate_enum.sh
. "$SCRIPT_DIR/_gate_enum.sh"
pred_reset; enum_reset

echo "== SCRIPTS LIVE (population: ${SCRIPTS##*/}/*.{sh,py,c}) =="

# ---------------------------------------------------------------- population
# maxdepth 1 is load-bearing: it is what keeps scripts/archive/ out of every
# family, and it is the reason archiving is a retirement mechanism rather than a
# rename.
mapfile -t POP < <(find "$SCRIPTS" -maxdepth 1 -type f \
                     \( -name '*.sh' -o -name '*.py' -o -name '*.c' \) -printf '%f\n' | sort)
enum_require_nonempty "scripts-live population" "${#POP[@]}" "scripts/*.{sh,py,c}" || exit 2

# The exclusion of docs/ archive/ is the RULE, so the filter is asserted
# to have removed something. A filter that silently matched nothing would make
# every script reachable-by-anything and report green on a tree full of corpses
# -- the vacuous-green shape, reached through a broken query instead of a broken
# regex.
mapfile -t PROSE < <(find "$SL_ROOT" -type f \
                       \( -path "$SL_ROOT/docs/*" \
                          -o -path "$SCRIPTS/archive/*" \) \
                       \( -name '*.sh' -o -name '*.py' -o -name '*.c' -o -name '*.md' \
                          -o -name '*.txt' -o -name '*.cpp' -o -name '*.hpp' \
                          -o -name '*.h' -o -name '*.cmake' \) 2>/dev/null | sort)
enum_require_nonempty "live-source EXCLUSION removed the record files" "${#PROSE[@]}" \
  "docs/ + scripts/archive/ prose (if this is 0 the filter is broken)" || exit 2

# ------------------------------------------------------------- the gate glob
# Parsed out of the runner. Restating `test_*.sh` here would make the class a
# second source of truth that can drift from the runner it claims to describe.
RUNNER="$SCRIPTS/run_ten_gates.sh"
GATE_GLOB=""
if [ -f "$RUNNER" ]; then
  GATE_GLOB="$(sed -n 's/^GATE_GLOB="\${INTFER_GATE_GLOB:-\(.*\)}"$/\1/p' "$RUNNER" | head -1)"
fi
enum_require_nonempty "gate glob parsed from run_ten_gates.sh" \
  "$(printf '%s' "$GATE_GLOB" | wc -c | tr -d ' ')" \
  'GATE_GLOB="${INTFER_GATE_GLOB:-...}" (if this is 0, the GLOB class cannot fire)' || exit 2
echo "-- gate class: $GATE_GLOB (parsed from $(basename "$RUNNER"), not restated)"

# ---------------------------------------------------------------- the scan
# $1 = filename. Prints the live source paths that name it, one per line.
#
# THE EXCLUSION IS DONE IN THE PIPELINE, NOT WITH `grep --exclude-dir`. Passing the
# directories to grep looked correct and was not: `--exclude-dir=.git` and
# `--exclude-dir=build` were both present on the command line and neither took
# effect, so `build/bdc.log` -- a run log, and the least live thing in the tree --
# was accepted as a live source and the scan reported it as the reference that
# makes a script reachable. The directory list was also a SECOND place to edit
# when a new generated tree appears, which is the drift rule 5 warns about. One
# `grep -vE` over the output is a single source of truth, and it is the filter
# whose effect the assertion above can see.
named_by_live_source() {
  grep -rlIF -- "$1" "$SL_ROOT" 2>/dev/null \
    | sed "s|^$SL_ROOT/||" \
    | grep -vE '^(build|\.git|docs|reports|node_modules)/' \
    | grep -vE '^scripts/archive/' \
    | grep -vF "scripts/$1"
}

n_named=0; n_glob=0; n_dead=0
DEAD=""
for f in "${POP[@]}"; do
  # GLOB class FIRST, and it is the stronger claim: the file runs without being
  # named anywhere, so no reference can vouch for it. Classifying NAMED first
  # would let an incidental mention hide the fact that the GLOB class is doing the
  # work -- and on a tree where every gate happens to be mentioned in prose, that
  # ordering reports GLOB 0 on a perfectly healthy population, which reads as a
  # broken rule and is indistinguishable from one.
  # `case` on the DERIVED glob, so the test is the runner's own and not a pattern
  # restated here that can drift from it.
  # shellcheck disable=SC2254
  case "$f" in
    $GATE_GLOB)
      n_glob=$((n_glob + 1))
      printf '  ok    %-34s GLOB  (%s expands it; no gate needs to be named)\n' "$f" "$GATE_GLOB"
      continue ;;
  esac
  hits="$(named_by_live_source "$f")"
  if [ -n "$hits" ]; then
    n_named=$((n_named + 1))
    n_hits="$(printf '%s\n' "$hits" | wc -l)"
    printf '  ok    %-34s NAMED by %s (+%d more)\n' "$f" \
      "$(printf '%s\n' "$hits" | head -1)" "$((n_hits - 1))"
    continue
  fi
  n_dead=$((n_dead + 1))
  DEAD="$DEAD $f"
  printf '  DEAD  %-34s no live source names it and no glob expands it\n' "$f"
done

total=$((n_named + n_glob + n_dead))
echo "-- derived: $total script(s): $n_named NAMED, $n_glob GLOB, $n_dead unreachable"

# ANTI-VACUITY, PLUS A CROSS-CHECK THAT IS NOT A FLOOR. A scan that classified
# nothing reachable is not a pass, it is a query that failed -- and a query that
# returns "every script is named" is exactly what a broken grep looks like. Both
# classes must be non-empty and the population must be non-trivial. These are
# deliberately WEAK floors (one of each, not five): the floors exist to catch a
# scan that classified nothing at all, and the real proof that the NAMED scan
# works in both directions is the planted MUST-FIRE / MUST-STAY-QUIET pair below,
# not an arbitrary constant that a synthetic tree cannot satisfy. The first
# version floored NAMED at 5, which made every planted state fail on the floor
# instead of on the rule -- the states stopped measuring the thing they exist to
# measure, which is the vacuity class wearing a floor's clothes.
#
# On top of that the GLOB count is DERIVED a second time, by expanding the glob
# over the population, and must agree with the count the scan classified: if the
# two disagree then one of them is reading a different glob than the runner does,
# which is the drift this class exists to prevent and which no single count shows.
n_gate_glob="$(for f in "${POP[@]}"; do
                # shellcheck disable=SC2254
                case "$f" in $GATE_GLOB) echo "$f" ;; esac
              done | wc -l | tr -d ' ')"
if [ "$n_named" -lt 1 ] || [ "$n_glob" -lt 1 ] || [ "$total" -lt 5 ] \
   || [ "$n_glob" -ne "$n_gate_glob" ]; then
  echo "  FAIL  classification looks wrong ($n_named NAMED / $n_glob GLOB / $total total," >&2
  echo "        and the glob independently expands to $n_gate_glob) -- a near-empty," >&2
  echo "        all-one-class or self-inconsistent result is a BROKEN SCAN, not a" >&2
  echo "        clean tree, and reporting it as a pass is the AUDIT #14 shape." >&2
  exit 1
fi

verdict() {
  echo
  if [ "$n_dead" -eq 0 ]; then
    echo "SCRIPTS LIVE: PASS ($total script(s), $n_named NAMED, $n_glob GLOB, 0 unreachable)"
    echo "  every script in scripts/ is reachable from a live source or the gate glob"
    echo "RESULT: every scripts/ entry is reachable; scripts/archive/ is excluded by construction"
    return 0
  fi
  echo "SCRIPTS LIVE: FAIL ($n_dead unreachable script(s))" >&2
  for f in $DEAD; do
    echo "  $f -- wire it to an entry point (run_all_checks.sh STEPS/dispatch, or" >&2
    echo "     README.md if a human runs it), or git mv it into scripts/archive/" >&2
    echo "     and record why in scripts/archive/README.md." >&2
  done
  echo "RESULT: FAIL ($n_dead) -- a script nothing runs is not tooling" >&2
  return 1
}

# A synthetic tree has nothing to plant into, and planting against itself would
# recurse. The mutation matrix drives this path.
if [ "$SCAN_ONLY" = 1 ]; then
  verdict
  exit $?
fi

# ------------------------------------------------- planted self-checks
# Both polarities, in a temp tree, before any verdict is reported. A rule proved
# only in the firing direction is the always-red twin; one proved only in the
# quiet direction certifies a tree it never examined.
SB="$(mktemp -d "${TMPDIR:-/tmp}/slive.XXXXXX")"
trap 'rm -rf "$SB"' EXIT
trap 'rm -rf "$SB"; exit 130' INT
trap 'rm -rf "$SB"; exit 143' TERM

# $1 = how the orphan is referenced: none | wired | prose | archive
mk_tree() {
  rm -rf "$SB"; mkdir -p "$SB/scripts" "$SB/docs" "$SB/reports" "$SB/src"
  for g in test_gate_a test_gate_b; do
    printf '#!/usr/bin/env bash\nexit 0\n' >"$SB/scripts/$g.sh"
  done
  printf '#!/usr/bin/env bash\nexit 0\n' >"$SB/scripts/wired.sh"
  # The caller names ONE script in every case, so the NAMED class is populated in
  # every planted tree and the floors below are measuring the reachability rule
  # rather than the emptiness of the fixture.
  printf '#!/usr/bin/env bash\nsource scripts/wired.sh\nexit 0\n' >"$SB/scripts/caller.sh"
  printf '#!/usr/bin/env bash\nGATE_GLOB="${INTFER_GATE_GLOB:-test_*.sh}"\n' \
    >"$SB/scripts/run_ten_gates.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$SB/scripts/zz_orphan.sh"
  # A source file that names the two harness scripts, so the ONLY unreachable
  # script in a planted tree is the orphan under test. Without it the fixture has
  # three unreachable scripts and the MUST-STAY-QUIET state fails for a reason
  # that has nothing to do with reachability -- a state that measures the fixture.
  printf '/* runs scripts/run_ten_gates.sh, which drives scripts/caller.sh */\nint main(void){return 0;}\n' \
    >"$SB/src/entry.c"
  # A record file is ALWAYS present, including in the `none` case. The exclusion
  # floor asserts that the docs/reports filter removed something, so a tree with
  # empty docs/ and reports/ would trip it -- correctly, since that is what a
  # broken filter looks like, but useless as a test bed. Every real tree has
  # records in it, so the synthetic ones do too.
  printf 'a finding that names scripts/wired.sh\n' >"$SB/docs/history.md"
  case "${1:-none}" in
    wired)   printf 'bash scripts/zz_orphan.sh\n' >>"$SB/scripts/caller.sh" ;;
    prose)   printf 'a finding that names scripts/zz_orphan.sh\n' >>"$SB/docs/history.md" ;;
    archive) mkdir -p "$SB/scripts/archive"
             printf '#!/usr/bin/env bash\nbash scripts/zz_orphan.sh\n' \
               >"$SB/scripts/archive/old.sh" ;;
    none)    : ;;
    *)       echo "mk_tree: unknown case '${1}'" >&2; exit 2 ;;
  esac
}
scan_tree() { INTFER_SCRIPTS_DIR="$SB/scripts" INTFER_SL_ROOT="$SB" \
                bash "$0" --scan-only 2>&1; }

echo
echo "-- planted self-checks (the rule must fire AND must stay quiet) --"

mk_tree none
o="$(scan_tree)"; rc=$?
if [ "$rc" -ne 0 ] && pred_expect_present "plant MUST-FIRE" "$o" 'DEAD  zz_orphan\.sh'; then
  echo "  ok    MUST-FIRE: a script no live source names is reported DEAD (rc=$rc)"
else
  echo "  FAIL MUST-FIRE missed -- wanted a non-zero rc AND a DEAD line; got rc=$rc"
  printf '%s\n' "$o" | tail -6 | sed 's/^/          /'; exit 1
fi

# THE DIRECTION THAT GETS FORGOTTEN. Same tree, one live reference added. A rule
# with no working quiet case fires on everything, and a check that fires on
# everything is switched off rather than fixed.
mk_tree wired
o="$(scan_tree)"; rc=$?
if [ "$rc" -eq 0 ] && pred_expect_absent "plant MUST-STAY-QUIET" "$o" 'DEAD  zz_orphan\.sh'; then
  echo "  ok    MUST-STAY-QUIET: one live source naming it makes it reachable (rc=0)"
else
  echo "  FAIL MUST-STAY-QUIET missed -- wanted rc=0; got rc=$rc"
  printf '%s\n' "$o" | tail -6 | sed 's/^/          /'; exit 1
fi

# The class a NAMED-only rule loses: a gate is reached by the glob alone.
mk_tree none
o="$(scan_tree)"
if pred_expect_present "plant GLOB" "$o" 'test_gate_a\.sh +GLOB '; then
  echo "  ok    GLOB class: a script matched by the runner's own glob is reachable"
else
  echo "  FAIL GLOB class did not fire -- every gate would read as dead"; exit 1
fi

# Prose is not a call site. This is the exact mistake that hid all sixteen.
mk_tree prose
o="$(scan_tree)"; rc=$?
if [ "$rc" -ne 0 ] && pred_expect_present "plant docs/-is-not-a-source" "$o" 'DEAD  zz_orphan\.sh'; then
  echo "  ok    docs/ does NOT make a script reachable (a citation is not a call)"
else
  echo "  FAIL docs/ was accepted as a reference -- every retired script survives"
  printf '%s\n' "$o" | tail -6 | sed 's/^/          /'; exit 1
fi

# Nor does a sibling in scripts/archive/.
mk_tree archive
o="$(scan_tree)"; rc=$?
if [ "$rc" -ne 0 ] && pred_expect_present "plant archive-is-not-a-source" "$o" 'DEAD  zz_orphan\.sh'; then
  echo "  ok    scripts/archive/ does NOT keep its own dead set alive"
else
  echo "  FAIL archive counted as a live source -- the retired set is immortal"
  printf '%s\n' "$o" | tail -6 | sed 's/^/          /'; exit 1
fi

verdict
exit $?