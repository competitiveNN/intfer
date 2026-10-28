#!/usr/bin/env bash
# check_engines_sources_paths.sh -- every citation of an engines_sources.txt line
# must NAME WHICH engines_sources.txt.
#
# WHY THIS EXISTS (measured 2026-10-21). TWO FILES carry the name
# `engines_sources.txt`, and BOTH are tracked:
#
#   engines_sources.txt              (repo root)  264 lines  md5 a2d6c65131e9828bdd528f324d80611c
#   docs/engines/engines_sources.txt              152 lines  md5 72b484ee7862ed5d8baead73365c80c7
#
# The root one carries NINE S1 records; the docs one carries ZERO. A durable
# record that says `engines_sources.txt:187` therefore names a line that exists in
# ONE file and not in the other, and a reader has to GUESS which -- the bare
# spelling resolves only by inferring "root". A refutation this same day probed
# `docs/engines/engines_sources.txt` (0 S1 hits) and declared a perfectly good
# root citation FABRICATED, because the probe named the other file. Then a
# correction replaced the VALID citation with a different set of sites and
# appended that to the durable record: a retraction that launders a guess into a
# correction, which is the failure this checker exists to make impossible.
#
# THE RULE: a citation must be path-qualified. Two accepted forms:
#   engines_sources.txt (ROOT):NNN           the 264-line root file
#   docs/engines/engines_sources.txt:NNN     the 152-line docs file
# A bare `engines_sources.txt:NNN` is REFUSED. A path-qualified citation is also
# required to RESOLVE: the line must exist in the file it names and must not be
# blank -- a citation to a blank line is the same defect in a new dress, and the
# live corpus carried one (`engines_sources.txt:189`, now blank) until it was
# retargeted to :187.
#
# THE HARVEST CANNOT SELF-MATCH, and that is what makes it safe to run over a
# corpus that DOCUMENTS this rule. The pattern requires DIGITS after the colon,
# so the rule written in prose as `engines_sources.txt:NNN` -- with a literal
# NNN, as every note about it does -- is not a citation and cannot trip its own
# detector. That is the same "a pattern that CANNOT self-match" argument
# mutate_durable_reads.sh makes for its SUBJECT_READS anchor, and it is the
# prose-corpus analogue of the comment-stripping the shell checkers do.
#
# THE POSITIVE CONTROL, and why a checker without one is vacuous-green. The
# verdict is "no bare citations". A harvest that found NOTHING would also report
# zero bare citations -- "looked at nothing" and "found nothing wrong" are
# different claims, and collapsing them is the shape this repo keeps refusing.
# So the checker proves its own instrument three ways before it may report:
#   p_detector  a synthetic bare citation IS classified bare and a synthetic
#               qualified one IS classified qualified (the detector works)
#   p_harvest   the corpus really carries >= 1 citation (the corpus is read)
#   p_qualified the corpus really carries >= 1 QUALIFIED citation (the
#               qualifier recogniser has a positive instance on real content)
# A corpus with zero qualified citations is REFUSED, not certified: with no
# positive instance, a "0 bare" verdict proves nothing about the detector.
#
# SUBJECT: the durable surface -- HISTORY.md, AGENTS.md, the root
# engines_sources.txt, docs/*.md and docs/engines/*.txt -- derived by glob, so a
# new doc is covered the day it lands.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The subject root, so the mutation matrix can grade states on COPIES. The live
# files are tracked and dirty by this repo's standing state; a matrix that edits
# them in place is the destructive probe the standing rule exists for.
SR="${ESP_ROOT:-${1:-$ROOT}}"

fails=0
note()  { printf 'ESP-PATH: ok   %-14s %s\n' "$1" "$2"; }
bad()   { printf 'ESP-PATH: FAIL %-14s %s\n' "$1" "$2" >&2; fails=$((fails + 1)); }
die()   { printf 'ESP-PATH: FAIL -- %s\n' "$*" >&2; echo "RESULT: FAIL"; exit 1; }

[ -d "$SR" ] || die "subject root missing ($SR)"

# ---- p_subject: the surface resolves, and it is not a vacuous corpus --------
SURFACE=()
for rel in HISTORY.md AGENTS.md engines_sources.txt; do
  [ -f "$SR/$rel" ] && SURFACE+=("$SR/$rel")
done
[ -d "$SR/docs" ] && for _d in "$SR"/docs/*.md; do
  [ -f "$_d" ] && SURFACE+=("$_d")
done
[ -d "$SR/docs/engines" ] && for _e in "$SR"/docs/engines/*.txt; do
  [ -f "$_e" ] && SURFACE+=("$_e")
done
[ "${#SURFACE[@]}" -lt 3 ] && die "the durable surface did not resolve (${#SURFACE[@]} record(s)); a corpus this small cannot certify anything"
for f in "${SURFACE[@]}"; do
  [ -s "$f" ] || die "$(basename "$f") is empty; a corpus with nothing in it cannot certify anything"
  [ "$(grep -vc '^\s*#' "$f")" -gt 0 ] || die "$(basename "$f") is comment-only; a strippable corpus must not be harvested"
done
note p_subject "ok (${#SURFACE[@]} record(s) on the durable surface, all non-empty and carrying prose)"

# ---- the harvest + the classifier --------------------------------------------
# The classifier is ONE function, used both for the corpus and for the synthetic
# positive control, so the control tests the code that produces the verdict.
out="$(python3 - "$SR" "${SURFACE[@]}" <<'PY'
import os, re, sys

root = sys.argv[1]
files = sys.argv[2:]

# A citation: the filename, an OPTIONAL " (ROOT)" qualifier, a colon, digits.
# The digits are what make the pattern unable to self-match on prose that
# documents the rule (which writes ":NNN", never ":<digits>").
CITE = re.compile(r"engines_sources\.txt(?P<qual> \(ROOT\))?:(?P<line>\d+)")

def classify(txt, start, m):
    """-> (kind, line) where kind is 'bare' | 'root' | 'docs' | 'path'."""
    if m.group("qual"):
        return "root", int(m.group("line"))
    pre = txt[max(0, start - 32):start]
    if pre.endswith("docs/engines/"):
        return "docs", int(m.group("line"))
    if pre.endswith("/"):
        # An absolute path already names the file. Resolve it where possible.
        tail = pre[pre.rfind("\n") + 1:]
        if tail.startswith("/"):
            return "path", int(m.group("line"))
        return "docs", int(m.group("line"))
    return "bare", int(m.group("line"))

bare, qualified = [], []
for f in files:
    txt = open(f, encoding="utf-8", errors="replace").read()
    for m in CITE.finditer(txt):
        kind, line = classify(txt, m.start(), m)
        ln = txt.count("\n", 0, m.start()) + 1
        rec = (f, ln, m.group(0), line, kind)
        if kind == "bare":
            bare.append(rec)
        else:
            qualified.append(rec)

# ---- the POSITIVE CONTROL, run through the SAME classifier ------------------
CONTROL_FAIL = []
ctl_bare = "the record cites engines_sources.txt:187 in prose"
ctl_root = "the record cites engines_sources.txt (ROOT):187 in prose"
ctl_docs = "the record cites docs/engines/engines_sources.txt:47 in prose"
for label, text, want in (("bare", ctl_bare, "bare"),
                          ("root", ctl_root, "root"),
                          ("docs", ctl_docs, "docs")):
    m = CITE.search(text)
    if not m:
        CONTROL_FAIL.append("control %s: the classifier found no citation in %r" % (label, text))
        continue
    got, _ = classify(text, m.start(), m)
    if got != want:
        CONTROL_FAIL.append("control %s: classified %r as %s, expected %s" % (label, text, got, want))

# ---- resolution: a qualified citation must name a real, non-blank line -------
unresolved = []
for f, ln, cited, line, kind in qualified:
    if kind == "root":
        target = os.path.join(root, "engines_sources.txt")
    elif kind == "docs":
        target = os.path.join(root, "docs", "engines", "engines_sources.txt")
    else:
        target = None
    if target is None:
        continue  # an absolute path resolves itself; nothing to check here
    if not os.path.isfile(target):
        unresolved.append("%s:%d cites %s but %s does not exist" % (f, ln, cited, target))
        continue
    lines = open(target, encoding="utf-8", errors="replace").read().split("\n")
    if line < 1 or line > len(lines):
        unresolved.append("%s:%d cites %s but that file has %d line(s)" % (f, ln, cited, len(lines)))
    elif not lines[line - 1].strip():
        unresolved.append("%s:%d cites %s but that line is BLANK" % (f, ln, cited))

print("BARE=%d" % len(bare))
print("QUALIFIED=%d" % len(qualified))
print("CONTROL_FAIL=%d" % len(CONTROL_FAIL))
for b in bare:
    print("BARE_SITE=%s:%d\t%s" % (b[0], b[1], b[2]))
for c in CONTROL_FAIL:
    print("CONTROL_BAD=%s" % c)
for u in unresolved:
    print("UNRESOLVED=%s" % u)
PY
)"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "ESP-PATH: FAIL -- the harvest failed (rc=$rc):" >&2
  printf '%s\n' "$out" | sed 's/^/        /' >&2
  echo "RESULT: FAIL"
  exit 1
fi

n_bare="$(printf '%s\n' "$out" | sed -n 's/^BARE=//p' | head -1)"
n_qual="$(printf '%s\n' "$out" | sed -n 's/^QUALIFIED=//p' | head -1)"
n_ctl="$(printf '%s\n' "$out" | sed -n 's/^CONTROL_FAIL=//p' | head -1)"
n_unres="$(printf '%s\n' "$out" | grep -c '^UNRESOLVED=' || true)"

# ---- p_harvest: the corpus really carries citations ------------------------
if [ "${n_bare:-0}" -eq 0 ] && [ "${n_qual:-0}" -eq 0 ]; then
  bad p_harvest "the durable surface carries NO engines_sources.txt citation; a checker that harvested nothing certifies nothing"
else
  note p_harvest "ok ($(( ${n_bare:-0} + ${n_qual:-0} )) citation(s) on the surface: ${n_bare:-0} bare, ${n_qual:-0} qualified)"
fi

# ---- p_detector: the POSITIVE CONTROL --------------------------------------
if [ "${n_ctl:-1}" -ne 0 ]; then
  bad p_detector "the classifier mis-classified its own synthetic control: $(printf '%s\n' "$out" | grep '^CONTROL_BAD=' | head -1 | sed 's/^CONTROL_BAD=//')"
else
  note p_detector "ok (a synthetic bare citation is classified bare; a synthetic (ROOT) and a synthetic docs/ citation are classified qualified)"
fi

# ---- p_no_bare: THE VERDICT ------------------------------------------------
if [ "${n_bare:-0}" -ne 0 ]; then
  bad p_no_bare "${n_bare} bare citation(s); each must name WHICH engines_sources.txt: $(printf '%s\n' "$out" | grep '^BARE_SITE=' | head -3 | tr '\n' ' ')"
else
  note p_no_bare "ok (no bare engines_sources.txt citation on the durable surface)"
fi

# ---- p_qualified: a positive instance of the qualifier ---------------------
if [ "${n_qual:-0}" -lt 1 ]; then
  bad p_qualified "the surface carries no PATH-QUALIFIED citation; with no positive instance, a '0 bare' verdict proves nothing about the detector"
else
  note p_qualified "ok (${n_qual} path-qualified citation(s) recognised on the surface)"
fi

# ---- p_resolves: a qualified citation must point at real content ------------
if [ "${n_unres:-0}" -ne 0 ]; then
  bad p_resolves "${n_unres} qualified citation(s) do not resolve to a non-blank line: $(printf '%s\n' "$out" | grep '^UNRESOLVED=' | head -2 | tr '\n' ' ')"
else
  note p_resolves "ok (every qualified citation resolves to a non-blank line in the file it names)"
fi

if [ "$fails" -ne 0 ]; then
  echo "ESP-PATH: FAIL ($fails failing predicate(s))"
  echo "RESULT: FAIL -- a durable citation names an engines_sources.txt without saying which"
  exit 1
fi
echo "ESP-PATH: PASS"
echo "RESULT: PASS -- every engines_sources.txt citation names which file it means"
exit 0
