#!/usr/bin/env bash
# A script must not identify a GPU CARD by its DRM minor. Minors are not stable:
# when the second B580 re-enumerated on 2026-10-03 it took the next free minor
# and became card2 / renderD130 (226:2), where it had been card1 / renderD129.
#
# WHAT THAT COST, MEASURED. scripts/gpu_recover.sh -- the tool AGENTS.md's runbook
# tells the operator to use to diagnose exactly this fault -- probed the LITERAL
# minors /dev/dri/renderD128 and /dev/dri/renderD129, and therefore printed
#
#     /dev/dri/renderD129  MISSING
#
# on a healthy box while never opening the real second node at all. Two failures in
# one line, both silent: a false alarm naming a device that merely got renamed, and
# the actual subject left unexamined. It now resolves through
# /dev/dri/by-path/pci-<slot>-render, which is named for the slot and survives.
#
# THE RULE, derived from source and never from a list of remembered names (the
# audit #7/#8 rule, and the reason check_build_dir_consumers.sh exists for the
# INTFER_BUILD_DIR family):
#
#     a script that probes render nodes AND names TWO OR MORE distinct
#     renderD<minor> literals is identifying cards by minor -- the defect.
#     Naming ZERO minors means it resolves by slot (or globs), which is correct.
#     Naming ONE is a liveness probe ("can we reach a GPU at all?").
#
# The verdict is deliberately UNCONDITIONAL on by-path. Two earlier versions were
# vacuous and both were caught by driving a mutation through them:
#   * v1 matched the literal "/dev/dri/renderD<minor>", so a script that hard-codes
#     a minor through the INTFER_DRI_ROOT seam ("$DRI_ROOT/renderD128") -- the very
#     seam added to make the probe testable -- scored PASS;
#   * v2 matched the minor prefix-agnostically but then excused any file that
#     mentioned "by-path" ANYWHERE, which is not the same claim as "the probe
#     resolves by slot": the mutation put the by-path line right next to a
#     hard-coded minor and passed.
# A compliant script names no minors, so it needs no exemption to pass. There is
# deliberately no escape hatch to game.
#
# SCOPE, and its cost, stated rather than implied. check_*/mutate_* are excluded as
# VERIFIERS by name prefix: a harness that PLANTS the defect has to be able to name
# it, and check_gpu_recover.sh legitimately contains renderD129 and renderD130 as
# fixture text. This is the same exclusion check_verdict_scope.sh already makes and
# for the same reason (ls scripts/test_*.sh DEFINES a population). The cost is
# real and named: a card-identity probe hidden inside a check_*/mutate_* would be
# invisible here. Trigger for revisiting: the day a check_*/mutate_* grows a REAL
# device probe rather than fixture text.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Seam, for the same reason INTFER_SYCL_LS / INTFER_BUILD_DIR / INTFER_SCRIPTS_DIR
# exist: the subject of this checker is OTHER scripts, so a mutation harness must be
# able to hand it a corpus of its own. Without it the only way to test the rule is
# to damage the real scripts/ directory.
SCRIPTS_DIR="${INTFER_SCRIPTS_DIR:-$SCRIPT_DIR}"

# classify <file>: 0 = clean, 1 = VIOLATION (identifies cards by minor)
#                2 = not applicable (does not probe render nodes at all)
# SELF-CHECK drives this exact function on planted files, so what is proven able to
# fail is what judges the tree -- not a re-run of the whole script, which would
# recurse into its own self-check (an earlier version did that and hung).
classify() {
  local f="$1" code minors n
  code="$(sed 's/^[[:space:]]*#.*$//' "$f")"
  # A verifier may PLANT the defect; excluding by PREFIX (see SCOPE above).
  case "$(basename "$f")" in check_*|mutate_*) return 2 ;; esac
  printf '%s' "$code" | grep -qE '/dev/dri|DRI_ROOT' || return 2
  # The MINOR, wherever it is anchored -- not the whole path. See v1 in the header.
  minors="$(printf '%s\n' "$code" | grep -oE 'renderD[0-9]+' | sort -u)"
  n="$(printf '%s\n' "$minors" | grep -c . || true)"
  CLASS_MINORS="$(printf '%s' "$minors" | tr '\n' ' ')"
  [ "$n" -ge 2 ] && return 1
  return 0
}

shopt -s nullglob
SCRIPTS=("$SCRIPTS_DIR"/*.sh)
shopt -u nullglob

# *** SELF-EXCLUSION BY IDENTITY, NOT BY NAME. *** This file must be able to plant
# the defect text, so including itself is a guaranteed false positive. Comparing to
# $0 is the same exclusion that cannot drift onto a different file.
self_is() { [ "$1" = "$0" ]; }

# *** BEHAVIOURAL SELF-CHECK FIRST, and it GATES the run. *** A rule that cannot
# fail proves nothing; one that fires on everything is the mirror defect.
TMP="$(mktemp -d "${TMPDIR:-/dev/shm}/dri_probe.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
trap 'rm -rf "$TMP"; exit 130' INT
trap 'rm -rf "$TMP"; exit 143' TERM
self_bad=0
dri="$(printf '/dev/dri/%s' x)"; dri="${dri%x}"   # built at runtime, see note
ra="$(printf '%s/render%s' "$dri" D128)"
rb="$(printf '%s/render%s' "$dri" D129)"
rc_node="$(printf '%s/render%s' "$dri" D130)"

printf 'probe() { for n in %s %s; do :; done; }\n' "$ra" "$rb" >"$TMP/bad.sh"
printf 'probe() { ls %s/by-path/pci-0000:86:00.0-render >/dev/null; }\n' "$dri" \
  >"$TMP/resolved.sh"
printf 'probe() { : ; } # %s\n' "$ra" >"$TMP/liveness.sh"

classify "$TMP/bad.sh";      r=$?
classify "$TMP/resolved.sh"; r2=$?
classify "$TMP/liveness.sh"; r3=$?
[ "$r"  -eq 1 ] || { echo "SELF-CHECK FAIL: 2 minors not REJECTED (rc=$r)";  self_bad=$((self_bad+1)); }
[ "$r2" -eq 0 ] || { echo "SELF-CHECK FAIL: by-path resolution not ACCEPTED (rc=$r2)"; self_bad=$((self_bad+1)); }
[ "$r3" -eq 0 ] || { echo "SELF-CHECK FAIL: single-minor liveness not ACCEPTED (rc=$r3)"; self_bad=$((self_bad+1)); }
# And the verifier exclusion must be real, not decorative: the same two minors are
# clean when planted under a check_* name, so the prefix cannot rot unnoticed.
cp "$TMP/bad.sh" "$TMP/check_planted.sh"
classify "$TMP/check_planted.sh"; r4=$?
[ "$r4" -eq 2 ] || { echo "SELF-CHECK FAIL: check_* exclusion is not in force (rc=$r4)"; self_bad=$((self_bad+1)); }

echo "== DRI RENDER-NODE PROBE CHECK (a card must not be identified by DRM minor) =="
if [ "$self_bad" -eq 0 ]; then
  echo "  ok    self-check: 2 minors REJECTED; by-path ACCEPTED; single-minor"
  echo "        liveness ACCEPTED; check_* verifier exclusion in force"
else
  echo
  echo "DRI PROBE: FAIL (self-check did not behave; the rule is not trustworthy)"
  exit 1
fi

viol=0; sites=0; corpus=0; resolved=0
for f in "${SCRIPTS[@]}"; do
  self_is "$f" && continue
  classify "$f"; r=$?
  [ "$r" -eq 2 ] && continue
  corpus=$((corpus + 1)); sites=$((sites + 1))
  b="$(basename "$f")"
  if [ "$r" -eq 0 ]; then
    printf '  ok    %-26s names no minors\n' "$b"
    grep -q 'by-path' "$f" && resolved=$((resolved + 1))
  else
    printf '  FAIL  %-26s identifies cards by minor: %s\n' "$b" "$CLASS_MINORS"
    viol=$((viol + 1))
  fi
done

# *** POPULATION GUARD. *** Assert the SUBJECT still exists in the corpus -- not
# that a violation exists. A healthy tree must be allowed to have none: the first
# version of this checker demanded sites > 0, i.e. evidence of a violation from a
# healthy tree (the audit #20 shape, inverted), and it failed the moment the
# defect was fixed.
if [ "$corpus" -eq 0 ]; then
  echo "FAIL: no script in scripts/ probes render nodes at all, so a green" >&2
  echo "      verdict would certify a population it never harvested." >&2
  exit 1
fi

echo "  ..    $corpus script(s) probe render nodes; $resolved resolve by PCI slot"
echo
if [ "$viol" -eq 0 ]; then
  echo "DRI PROBE: PASS (no script identifies a card by DRM minor)"
  echo "RESULT: card identity resolves through /dev/dri/by-path, which survives a"
  echo "        renumber (single-minor liveness probes are deliberately allowed)"
  exit 0
fi
echo "DRI PROBE: FAIL ($viol of $sites script(s) name render nodes by minor)"
echo "RESULT: a script identifies a card by DRM minor, which is not stable"
exit 1