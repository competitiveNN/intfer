#!/usr/bin/env bash
# A document that states a HARDWARE CAPABILITY figure must say where that figure
# came from. This is the enforcement half of the correction made on 2026-10-05,
# after the repo spent a session optimizing against a GPU it had never asked about.
#
# WHAT WENT WRONG, MEASURED. AGENTS.md and src/gemm/quant_gemm.cpp carried
#
#     "N ~ 2048 per rank over 512 Xe cores, ~4 threads/EU"
#     "35.5 GiB/s ... about 21% of the B580's ~179 GB/s"
#
# The Arc Pro B580 has 20 Xe cores, 160 vector engines, and 456.0 GB/s
# (CITATION(docs/Intel Arc B580 Specs _ TechPowerUp GPU Database.html, saved from
# techpowerup.com/gpu-specs/arc-b580.c4244; cross-checked against Intel ARK
# sku/241598 -- both agree)). 512 Xe cores is the A770's count (4096 shader units
# / 8). `git log -S '512 Xe'` shows the claim entered at 40219bf (2026-10-04) and
# was thereafter only propagated, never measured -- the same shape as
# docs/next_plan.md's per-role table summing to 25.5 GB against its own stated
# 19.5 GB.
#
# Consequence, stated so the correction is not mistaken for a re-tuning: the
# "EU occupancy / parallelism-bound" MODEL of decode was wrong (at 2560 work-items
# against ~2560 FP32 lanes the GEMV is at ~100% lane occupancy, not starved), so
# decode is at ~9% of the memory roofline rather than ~21%. MR and SPLITK were
# therefore aimed at a bottleneck that does not exist. THEIR MEASUREMENTS STAND;
# their explanation does not.
#
# THE RULE, and it is deliberately narrow:
#
#     A line in the documentation corpus that states a hardware capability or
#     bandwidth figure in a SPEC UNIT must carry a CITATION(...) marker.
#
# "Spec unit" = a number followed by a Xe-core / bandwidth / throughput unit:
#     Xe cores | <n> GB/s | <n> GiB/s | <n> TFLOPS
#
# This fires on a MEASURED figure as well as a SPEC one, and that is intended: a
# throughput number in prose should name the run it came from, and several here do
# (perf_base.err, a counters-only [GPU profile] line). A line with the word
# "bandwidth" and no number/unit is NOT a figure and is not flagged -- which is why
# this checker does not, and cannot, catch a stale percentage whose denominator is
# on another line. That limit is named here rather than left to be discovered.
#
# SCOPE, with its cost stated. `docs/` is EXCLUDED on purpose. A docs/ file is a
# record of what was believed when it was written; rewriting it to match
# today's hardware is the same laundering failure AGENTS.md warns about in the
# other direction (see its [SUPERSEDED ...] discipline on gate tallies). The
# correction is delivered instead as an annotation on the LIVE doc plus this
# checker, which stops the claim from being re-read as current.
#
# *** CORRECTION TO THE NOTE AT LINE 47-48, 2026-10-06: `reports/` WAS NOT
# retired and DOES hold live reports. The claim that it no longer holds live
# reports was itself a stale prose assertion about scope -- precisely the
# laundering this checker exists to stop -- and it let an uncited figure
# through (reports/2026-10-06-batch-bisect-4.md). The note is corrected here
# rather than the scope quietly widened, because widening scope until a
# defect disappears is the failure this file was written to prevent.
# `scripts/` IS excluded: the subject is documents, not tooling, and a checker
# here legitimately contains the defect text as a fixture; gating itself on its
# own fixtures would be a class of defect this repo has already paid for.
#
# Scope of the citation window, chosen deliberately: a figure is cited if a
# CITATION(...) marker appears anywhere in the same blank-line-delimited PARAGRAPH.
# That is "at the claim site" -- not on the same line, and emphatically not
# somewhere in the document, which is the vacuous version. It still rejects every
# one of the four real defects in the corpus today (verified below), because in
# each the wrong figure sits in a paragraph that never says where it came from.
classify() {
  python3 - "$1" <<'PYEOF'
import re, sys

SPEC = re.compile(
    r"[0-9]+\s*(xe[\s-]*cores?)"      # 20 Xe cores / 512 Xe-cores
    r"|[0-9]+(\.[0-9]+)?\s*(g|t)i?b/s"  # 456 GB/s, 35.5 GiB/s, 179 gb/s
    r"|[0-9]+(\.[0-9]+)?\s*tflops",     # 13.67 TFLOPS
    re.IGNORECASE)

path = sys.argv[1]
try:
    text = open(path, encoding="utf-8", errors="replace").read()
except OSError:
    print("CLASS_MISSING=<unreadable>", file=sys.stderr)
    sys.exit(2)

# Paragraph = blank-line-delimited block. Markdown tables are NOT blank-line
# delimited, so a table is one paragraph -- which is the right granularity: the
# table's caption/citation must sit with the table, as it now does in AGENTS.md.
paras = re.split(r"\n[ \t]*\n", text)
hits = []
for p in paras:
    if not SPEC.search(p):
        continue
    hits.append(p)

if not hits:
    sys.exit(2)                      # no figure anywhere -> not a site

missing = [p for p in hits if "CITATION(" not in p]
shown = []
for p in missing:
    for line in p.splitlines():
        if SPEC.search(line):
            shown.append(line.strip())
            break
print("CLASS_MISSING=" + "|".join(shown[:20]))
sys.exit(1 if missing else 0)
PYEOF
}

# *** BEHAVIOURAL SELF-CHECK FIRST, and it GATES the run. ***
TMP="$(mktemp -d "${TMPDIR:-/dev/shm}/hwconst.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
trap 'rm -rf "$TMP"; exit 130' INT
trap 'rm -rf "$TMP"; exit 143' TERM
self_bad=0

# Fixtures are written by a python heredoc rather than by bash quoting: the spec
# vocabulary contains single quotes and parentheses, and a hand-quoted version of
# this block was wrong before it was ever run. Distinct delimiter (AGENTS rule 15).
python3 - "$TMP" <<'PYEOF'
import os, sys
tmp = sys.argv[1]
XE   = " Xe cores"
CITE = "CITATION(techpowerup arc-b580.c4244)"
cases = {
    # --- must be REJECTED: a figure with no source ---
    "bad_scratch.sh":      "decode is ~179 GB/s\n",
    "bad_measured.sh":     "bandwidth: 41.1 GB/s at 9% of peak\n",
    "bad_cores.sh":        "the card has 512%s\n" % XE,
    "bad_tflops.sh":       "FP32 13.67 TFLOPS\n",
    # --- the exact claim that motivated this checker, planted verbatim ---
    "bad_known_case.sh":   "N~2048 per rank over 512%s\n" % XE,
    # --- must be ACCEPTED ---
    "good_cited.sh":       "decode is ~179 GB/s  %s\n" % CITE,
    "good_measured_cited.sh": "bandwidth: 41.1 GB/s  %s\n" % CITE,
    "good_no_figure.sh":   "this is bandwidth-bound, not a figure\n",
    "good_unrelated_unit.sh": "sub-group size is 16, registers are 255\n",
}
for name, body in cases.items():
    with open(os.path.join(tmp, name), "w") as f:
        f.write(body)
PYEOF
[ $? -eq 0 ] || { echo "HW CONSTANTS: FAIL (could not plant fixtures)"; exit 1; }

classify "$TMP/bad_scratch.sh"        >/dev/null; r1=$?
classify "$TMP/bad_measured.sh"       >/dev/null; r2=$?
classify "$TMP/bad_cores.sh"          >/dev/null; r3=$?
classify "$TMP/bad_tflops.sh"         >/dev/null; r4=$?
classify "$TMP/good_cited.sh"         >/dev/null; r5=$?
classify "$TMP/good_measured_cited.sh">/dev/null; r6=$?
classify "$TMP/good_no_figure.sh"     >/dev/null; r7=$?
classify "$TMP/good_unrelated_unit.sh">/dev/null; r8=$?
classify "$TMP/bad_known_case.sh"     >/dev/null; r9=$?

chk() { [ "$2" -eq "$3" ] && return 0; echo "  SELF-CHECK FAIL: $1 (rc=$2, want $3)"; self_bad=$((self_bad+1)); }
chk "an uncited SPEC bandwidth figure is REJECTED"      "$r1" 1
chk "an uncited MEASURED bandwidth figure is REJECTED"  "$r2" 1
chk "an uncited Xe-core count is REJECTED"              "$r3" 1
chk "an uncited TFLOPS figure is REJECTED"              "$r4" 1
chk "a CITED spec figure is ACCEPTED"                  "$r5" 0
chk "a CITED measured figure is ACCEPTED"              "$r6" 0
# r7/r8 expect 2 (NOT APPLICABLE), not 0 (clean): a file that states no figure at
# all is outside this rule's population, and conflating the two would let a corpus
# of irrelevant files pass as "checked".
chk "prose using the word bandwidth with no figure is NOT A SITE" "$r7" 2
chk "an unrelated unit on the same line is NOT A SITE"            "$r8" 2
chk "the 512 Xe cores case that motivated this is REJECTED"       "$r9" 1

echo "== HARDWARE-CONSTANT CITATION CHECK (a capability figure must be cited) =="
if [ "$self_bad" -eq 0 ]; then
  echo "  ok    self-check: 5 uncited-figure cases REJECTED (spec BW, measured"
  echo "        BW, Xe cores, TFLOPS, and the verbatim 512-core claim that"
  echo "        motivated this); 4 correct cases ACCEPTED (cited spec, cited"
  echo "        measured, prose with no figure, unrelated unit)"
else
  echo
  echo "HW CONSTANTS: FAIL (self-check did not behave; the rule is not trustworthy)"
  exit 1
fi

# --- population, derived by ENUMERATION (never a hand-written list) -----------
shopt -s nullglob
CORPUS=()
[ -f "$DOCS_DIR/AGENTS.md" ] && CORPUS+=("$DOCS_DIR/AGENTS.md")
for f in "$DOCS_DIR"/docs/*.md "$DOCS_DIR"/"$REPORTS_SUBDIR"/*.md; do CORPUS+=("$f"); done
shopt -u nullglob

if [ "${#CORPUS[@]}" -eq 0 ]; then
  echo "FAIL: documentation corpus is empty ($DOCS_DIR), so a green verdict" >&2
  echo "      would certify a population it never harvested." >&2
  exit 2
fi

viol=0; sites=0; n=0
for f in "${CORPUS[@]}"; do
  n=$((n+1))
  CLASS_MISSING="$(classify "$f")"; r=$?
  CLASS_MISSING="${CLASS_MISSING#CLASS_MISSING=}"
  [ "$r" -eq 2 ] && continue
  sites=$((sites+1))
  b="${f#$DOCS_DIR/}"
  if [ "$r" -eq 0 ]; then
    printf '  ok    %-46s every figure cited\n' "$b"
  else
    printf '  FAIL  %-46s uncited figure(s):\n' "$b"
    printf '%s\n' "$CLASS_MISSING" | tr '|' '\n' | sed '/^$/d;s/^/          /'
    viol=$((viol+1))
  fi
done

# Assert the SUBJECT still exists in the corpus -- NOT that a violation exists. A
# healthy corpus must be allowed to have none; demanding sites > 0 as evidence of
# a violation is the audit #20 shape inverted, and it fails the moment the defect
# is fixed.
if [ "$sites" -eq 0 ]; then
  echo "FAIL: none of the $n document(s) states a hardware figure, so this run" >&2
  echo "      harvested nothing to judge." >&2
  exit 1
fi

echo "  ..    $n document(s) scanned; $sites state a hardware figure"
echo
if [ "$viol" -eq 0 ]; then
  echo "HW CONSTANTS: PASS (every hardware figure in the docs is cited)"
  echo "RESULT: each capability figure names its source, so a wrong figure (the"
  echo "        512-Xe-core / 179-GB-s case) cannot be re-read as current."
  exit 0
fi
echo "HW CONSTANTS: FAIL ($viol of $sites document(s) state an uncited figure)"
echo "RESULT: add CITATION(<source>) to the flagged line, or correct the figure."
exit 1
