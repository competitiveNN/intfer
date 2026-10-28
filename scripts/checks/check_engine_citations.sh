#!/usr/bin/env bash
# check_engine_citations.sh -- every citation in the engine survey corpus must be
# LOCATION-BEARING AND PATH-QUALIFIED, and must resolve to a real line in the
# file it names.
#
# WHY THIS EXISTS. The citation discipline the whole upstream-survey round was
# about was enforced ONLY by a scratch tool at /tmp/citecheck3.py: no gate could
# catch a regression in it, and it depended on ephemeral /tmp control bytes. A
# rule nobody can regress is not a rule. This checker is the promoted resolver,
# and the resolver itself lives in scripts/lib/_engine_citations_scan.py for the
# measured reason _dispatch_scan.py gives (a heredoc mangles backslashes in
# regexes, and the failure mode is the parser dying silently so the harvest
# returns NOTHING -- a corpus with no citations read as a corpus with no
# defects, which is the vacuous-green shape this repo keeps refusing).
#
# THE RULE.
#   * a CITATION is location-bearing (it carries `:<numbers>`) AND
#     path-qualified (it carries a path). A bare basename -- `engine.cpp:337` --
#     is REFUSED: it resolves only by GUESSING which file of that name is
#     meant. That is not hypothetical: the old resolver resolved a bare
#     `README.md:6` ROOT-FIRST, so it pointed at intfer's README.md -- a real
#     file, in range, GREEN -- while the quoted text was jitllm's. A citation
#     could name the WRONG FILE and satisfy every predicate.
#   * every NUMBER in a location list is range-checked. A comma list is N
#     checks, not one.
#   * resolution is TREE-AWARE and headed by the report itself:
#         intfer/<p>    the repo root
#         upstream/<p>  the clone named in the report (block-scoped)
#         <p>           repo root first, then the clone; existing in BOTH is an
#                      AMBIGUITY, not a resolution
#         /abs/path     resolves exactly, and is refused as a house-style
#                      violation: an absolute path is machine-specific
#     and each report states that rule in a `# Path style:` header, naming the
#     four qualifiers. A header statement that does not NAME the qualifiers is
#     a decorative marker, so the checker requires the qualifiers.
#   * MENTIONS are a SEPARATE class: path-qualified ones are existence-checked,
#     bare ones are CLASSED by whether the basename is ambiguous across the
#     trees that report talks about.
#   * the same-basename audit PRINTS the competing paths, because that audit is
#     the reason a path is required at all.
#
# THE ANTI-VACUITY RULES, and why each exists:
#   p_pop        the population is non-empty. "Looked at nothing" and "found
#                nothing wrong" are different claims; collapsing them is the
#                vacuous-green shape.
#   p_control    the resolver's OWN POSITIVE CONTROL passed. The control runs
#                through the same resolve()/classify_citation() the corpus uses,
#                so a control that reimplemented the rule would certify the
#                control rather than the resolver.
#   p_harvest    the corpus really carries citations AND mentions. Without a
#                positive instance, a "0 defects" verdict proves nothing.
#   p_qualified  the corpus really carries a RESOLVED, tree-qualified citation
#                -- the positive instance of the qualifier recogniser on real
#                content.
#   p_samebase   the same-basename audit RAN. An audit that printed nothing
#                because it never indexed anything is not a clean audit.
#
# SEAMS, for the mutation matrix (the same reason INTFER_SCRIPTS_DIR exists in
# check_dri_probe.sh): the matrix must be able to hand this checker a corpus of
# its OWN without damaging the tracked files.
#   ENG_CITE_ROOT     the repo root (default: this script's repo)
#   ENG_CITE_CORPUS   the report corpus (default: $ROOT/docs/engines)
#   ENG_CITE_LIB      the resolver (default: $ROOT/scripts/lib/_engine_citations_scan.py).
#                     A SEPARATE seam from ROOT, because the mutation matrix has
#                     to damage a COPY of the resolver while still resolving the
#                     corpus's intfer/ tokens against the real repo. Collapsing
#                     the two would force the matrix to run its corpus against
#                     a sandbox root it does not live in.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ENG_CITE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CORPUS="${ENG_CITE_CORPUS:-$ROOT/docs/engines}"

fails=0
note() { printf 'ENG-CITE: ok   %-14s %s\n' "$1" "$2"; }
bad()  { printf 'ENG-CITE: FAIL %-14s %s\n' "$1" "$2" >&2; fails=$((fails + 1)); }
die()  { printf 'ENG-CITE: FAIL -- %s\n' "$*" >&2; echo "RESULT: FAIL"; exit 1; }

[ -d "$ROOT" ] || die "the repo root does not resolve ($ROOT)"
[ -d "$CORPUS" ] || die "the corpus does not resolve ($CORPUS)"
LIB="${ENG_CITE_LIB:-$ROOT/scripts/lib/_engine_citations_scan.py}"
[ -f "$LIB" ] \
  || die "the resolver is missing ($LIB)"

# ---- the harvest --------------------------------------------------------------
facts="$(python3 "$LIB" --corpus "$CORPUS" --repo "$ROOT" 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "ENG-CITE: FAIL -- the harvest failed (rc=$rc):" >&2
  printf '%s\n' "$facts" | sed 's/^/        /' >&2
  echo "RESULT: FAIL"
  exit 1
fi

# Counts are taken with awk, not `grep -c`: the shell wrapper reformats
# `grep -c` output, and a reformatted count is a count that cannot be compared.
cnt() { printf '%s\n' "$facts" | awk -v k="$1" '$1==k {n++} END {print n+0}'; }
cnt_class() { printf '%s\n' "$facts" | awk -v k="$1" '$1=="DEFECT" && $2==k {n++} END {print n+0}'; }
sum_val() { printf '%s\n' "$facts" | awk -v k="$1" '$1=="SUMMARY" && $2==k {print $3; found=1} END {if(!found) print "-"}'; }

POP="$(sum_val POP)"
n_ctl="$(cnt CONTROL_BAD)"
n_cite="$(cnt CITE)"
n_mention="$(cnt MENTION)"
n_defect="$(cnt DEFECT)"
n_samebase="$(sum_val SAMEBASE)"

# ---- p_pop: the population is not empty ---------------------------------------
if [ "${POP:-0}" -lt 1 ]; then
  bad p_pop "the corpus yielded NO reports; a checker that looked at nothing certifies nothing"
else
  note p_pop "ok (${POP} report(s) in the population)"
fi

# ---- p_control: the resolver's own positive control ---------------------------
if [ "${n_ctl:-1}" -ne 0 ]; then
  bad p_control "the resolver mis-classified its own synthetic control: $(printf '%s\n' "$facts" | awk '$1=="CONTROL_BAD" {print substr($0, 12); exit}')"
else
  note p_control "ok (the synthetic bare/out-of-range/absent/absolute cases were all classified correctly)"
fi

# ---- p_harvest: the corpus really carries citations AND mentions --------------
if [ "${n_cite:-0}" -lt 1 ] || [ "${n_mention:-0}" -lt 1 ]; then
  bad p_harvest "the corpus carries ${n_cite:-0} citation(s) and ${n_mention:-0} mention(s); without a positive instance on both, a '0 defects' verdict proves nothing"
else
  note p_harvest "ok (${n_cite} citation(s) and ${n_mention} mention(s) harvested)"
fi

# ---- p_qualified: a RESOLVED, tree-qualified citation exists ------------------
n_resolved_qualified="$(printf '%s\n' "$facts" | awk '$1=="CITE" && $6=="OK" {n++} END {print n+0}')"
if [ "${n_resolved_qualified:-0}" -lt 1 ]; then
  bad p_qualified "the corpus carries no RESOLVED citation; with no positive instance, a '0 defects' verdict proves nothing about the resolver"
else
  note p_qualified "ok (${n_resolved_qualified} resolved citation(s) on real content)"
fi

# ---- the defect predicates, one per named class -------------------------------
d() { cnt_class "$1"; }

n_bare="$(d BARE_CITATION)"
if [ "${n_bare:-0}" -ne 0 ]; then
  bad p_qualified_rule "${n_bare} bare citation(s): a citation carries a location AND a path; $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && $2=="BARE_CITATION" {print $3 ":" $4 " " $5; exit}')"
else
  note p_qualified_rule "ok (no bare citation: every location-bearing citation names a path)"
fi

n_resolve="$(d NOT_FOUND)"; n_ambig="$(d AMBIGUOUS_TREE)"; n_noclone="$(d NO_CLONE)"
n_resolve=$((n_resolve + n_ambig + n_noclone))
if [ "${n_resolve:-0}" -ne 0 ]; then
  bad p_resolve "${n_resolve} citation(s) did not resolve unambiguously (NOT_FOUND $(d NOT_FOUND), AMBIGUOUS_TREE $(d AMBIGUOUS_TREE), NO_CLONE $(d NO_CLONE)): $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && ($2=="NOT_FOUND"||$2=="AMBIGUOUS_TREE"||$2=="NO_CLONE") {print $2 " " $3 ":" $4 " " $5; exit}')"
else
  note p_resolve "ok (every citation resolves to exactly one file, in the tree it names)"
fi

n_range="$(d OUT_OF_RANGE)"
if [ "${n_range:-0}" -ne 0 ]; then
  bad p_range "${n_range} citation(s) name a line outside the file: $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && $2=="OUT_OF_RANGE" {print $3 ":" $4 " " $5; exit}')"
else
  note p_range "ok (every number in every location list is in range)"
fi

n_malformed="$(d MALFORMED_LOCATION)"
if [ "${n_malformed:-0}" -ne 0 ]; then
  bad p_malformed "${n_malformed} citation(s) carry a malformed location: $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && $2=="MALFORMED_LOCATION" {print $3 ":" $4 " " $5; exit}')"
else
  note p_malformed "ok (every location list is a well-formed list of line numbers)"
fi

n_sentinel="$(d SENTINEL_MISUSE)"
if [ "${n_sentinel:-0}" -ne 0 ]; then
  bad p_sentinel "${n_sentinel} sentinel(s) name a file that EXISTS: $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && $2=="SENTINEL_MISUSE" {print $3 ":" $4 " " $5; exit}')"
else
  note p_sentinel "ok (no absent/ sentinel names a file that exists)"
fi

n_abs_c="$(d ABSOLUTE_CITATION)"; n_abs_m="$(d ABSOLUTE_MENTION)"
if [ "$((n_abs_c + n_abs_m))" -ne 0 ]; then
  bad p_house_style "$((n_abs_c + n_abs_m)) absolute path(s) (${n_abs_c} citation, ${n_abs_m} mention): an absolute path is machine-specific; $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && ($2=="ABSOLUTE_CITATION"||$2=="ABSOLUTE_MENTION") {print $2 " " $3 ":" $4; exit}')"
else
  note p_house_style "ok (no absolute path: every path is tree-qualified)"
fi

n_style="$(d STYLE_STATEMENT_MISSING)"
if [ "${n_style:-0}" -ne 0 ]; then
  bad p_style_header "${n_style} report(s) do not state the house path style in the header: $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && $2=="STYLE_STATEMENT_MISSING" {print $3; exit}')"
else
  note p_style_header "ok (every report states the path style and names the four qualifiers)"
fi

n_mention_bad="$(d UNRESOLVED_MENTION)"
if [ "${n_mention_bad:-0}" -ne 0 ]; then
  bad p_mentions "${n_mention_bad} path-qualified mention(s) name a file that is not there: $(printf '%s\n' "$facts" | awk '$1=="DEFECT" && $2=="UNRESOLVED_MENTION" {print $3 ":" $4 " " $5; exit}')"
else
  note p_mentions "ok (every path-qualified mention resolves to a real file)"
fi

# ---- p_samebase: the audit RAN, and its competing paths are printed ----------
if ! printf '%s\n' "$facts" | awk '$1=="SUMMARY" && $2=="SAMEBASE" {found=1} END {exit(found?0:1)}'; then
  bad p_samebase "the same-basename audit did not run; an audit that printed nothing because it never indexed anything is not a clean audit"
else
  note p_samebase "ok (${n_samebase} same-basename collision(s) -- printed below; this is why a path is required)"
  # ONE line per distinct basename, not one per collision: the live corpus
  # carries 1830 collisions and printing every row buries the verdict in
  # 200 KB of output, which is how a real finding gets scrolled past.
  printf '%s\n' "$facts" | awk '$1=="SAMEBASE" {
      n=split($5, a, "/"); base=a[n];
      if (!(base in seen)) { seen[base]=1; order[++k]=base; ex[base]=$6 }
      cnt[base]++
    } END {
      for (i=1; i<=k; i++) { b=order[i]
        printf "ENG-CITE: AUDIT same-base %-30s %4d collision(s), e.g. %s\n", b, cnt[b], ex[b] }
    }'
fi

# ---- the bare-mention classes, printed ----------------------------------------
printf '%s\n' "$facts" | awk '$1=="SUMMARY" && $2 ~ /^MENTION_/ {print "ENG-CITE: AUDIT " $2 " " $3}'

if [ "$fails" -ne 0 ]; then
  echo "ENG-CITE: FAIL ($fails failing predicate(s))"
  echo "RESULT: FAIL -- a citation in the engine survey corpus is not location-bearing and path-qualified"
  exit 1
fi
echo "ENG-CITE: PASS"
echo "RESULT: PASS -- every engine-survey citation is location-bearing, path-qualified and in range"
exit 0
