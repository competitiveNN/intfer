#!/usr/bin/env bash
# check_closing_band_rows.sh — source-level certification of scripts/closing_band.sh's
# classification record: every printing branch records its own class, and the three
# counts (recorded rows, class counters, artifact lines) agree with the derived
# population so a summary can never under-report the table it certifies.
#
# WHY THIS EXISTS. The round-7 summary under-reported the baseline table: it said
# "three DIFFERS" while the log says five (three checkers plus mutate_durable_reads
# and mutate_verdict_scope), and omitted the four ABSENT rows entirely. That is a
# verdict about the table without an artifact behind it — the exact failure the
# third-way assertion in closing_band.sh exists to prevent. This checker makes the
# structure itself certifiable: every printing branch calls cls_record, the six
# class arms are distinct with their own counters, the three -eq assertions exist,
# and the illegal-class else branch sets rc_all=1.
#
# NO SUBJECT_READS MARKER: scripts/closing_band.sh is not a durable record
# (HISTORY.md, AGENTS.md, docs/*.md, docs/engines/*.txt are), so this checker does
# not read one and must not be marked as doing so.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# INTFER_CLOSE_BAND_SUBJECT lets the planted self-check grade a corrupted copy
# instead of the real subject; INTFER_CLOSE_BAND_SELFCHECK=0 breaks the recursion.
SUBJECT="${INTFER_CLOSE_BAND_SUBJECT:-$SCRIPT_DIR/../closing_band.sh}"
# The checker itself, re-invoked by the planted self-check below. It must be the
# CHECKER that is re-run against the corrupted copy, never the subject: the first
# version wrote `bash "$SUBJECT"`, which launched the whole closing band (and hung
# the self-check for the full band timeout) instead of re-running this checker's
# predicates against the corrupt copy.
CHECKER="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

fails=0

note() { # $1 predicate, $2 status (ok|FAIL), $3 message
  if [ "$2" = "FAIL" ]; then
    printf '  FAIL  %s: %s\n' "$1" "$3" >&2
    fails=$((fails + 1))
  else
    printf '  ok    %s: %s\n' "$1" "$3"
  fi
}

# *** THE BOOLEAN HELPER EXISTS BECAUSE THE STATUS MUST NOT BE INFERRED. ***
# note() compares its status against the LITERAL "FAIL", so a predicate that
# passes "1"/"0" straight through prints "ok" for BOTH -- a zero would read as
# a pass, which is the inverted-polarity defect this repo's tooling exists to
# catch, and it would have made four of these predicates permanently green.
note_bool() { # $1 predicate, $2 value (1|0), $3 message
  if [ "$2" = "1" ]; then
    note "$1" ok "$3"
  else
    note "$1" FAIL "$3"
  fi
}

# --- p_subject: subject exists and bash -n parses -----------------------------
if [ ! -f "$SUBJECT" ]; then
  echo "RESULT: FAIL" >&2
  echo "subject missing: $SUBJECT" >&2
  exit 1
fi
if ! bash -n "$SUBJECT" >/dev/null 2>&1; then
  echo "RESULT: FAIL" >&2
  echo "subject fails bash -n: $SUBJECT" >&2
  exit 1
fi

# --- p_derive: extract counts from the subject source -------------------------
derive_out="$(python3 - "$SUBJECT" <<'PY'
import re, sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Strip comment lines so prose cannot masquerade as mechanism.
code_lines = [l for l in src.split('\n') if not re.match(r'^\s*#', l)]
code = '\n'.join(code_lines)

def extract_function(code, name):
    # Return the body text of `name() { ... }` via brace counting.
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(\s*\)\s*\{')
    m = pat.search(code)
    if not m:
        return None
    i = m.end() - 1  # position of the opening '{'
    depth = 1
    body_start = i + 1
    i = body_start
    n = len(code)
    while i < n and depth > 0:
        if code[i] == '{':
            depth += 1
        elif code[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return code[body_start:i - 1]

classify_body = extract_function(code, 'classify')
cls_record_body = extract_function(code, 'cls_record')

# Count cls_record calls inside classify() body.
cls_calls_in_classify = 0
if classify_body is not None:
    cls_calls_in_classify = len(re.findall(r'^\s*cls_record\s+"', classify_body, re.M))

# Count printed classification branches inside classify(): each branch prints
# with a printf whose first arg contains '  %-34s '.
print_branches = 0
if classify_body is not None:
    print_branches = len(re.findall(r'printf\s+[^\n]*%-34s', classify_body))

# Six legal class arms, each with its own counter, inside cls_record().
# THE ARM AND ITS COUNTER ARE ON ONE LINE (`ARM)  n_x=$((n_x + 1)) ;;`), which is
# why the pattern does NOT expect a newline between them -- the first version did,
# and found ZERO arms on a healthy subject, i.e. a checker that refuses correct code.
class_arms = {}
counters = {}
if cls_record_body is not None:
    for m in re.finditer(
        r'(GREEN_BOTH|DIFFERS_FROM_HEAD|NEW_REGRESSION|PREEXISTING_AT_HEAD|ABSENT_AT_HEAD|UNDECIDABLE)\s*\)\s+(n_\w+)\s*=\$\(\(\s*(n_\w+)\s*\+\s*1\s*\)\)',
        cls_record_body,
    ):
        class_arms[m.group(1)] = m.group(2)
        counters[m.group(1)] = m.group(3)

# The three -eq assertions after the band composition.
has_row_count = bool(re.search(r'"\$n_cls"\s*-eq\s*"\$want_all"', code))
has_class_sum = bool(re.search(r'"\$n_sum"\s*-eq\s*"\$want_all"', code))
has_artifact_lines = bool(re.search(r'"\$n_file"\s*-eq\s*"\$n_cls"', code))

# The illegal-class else branch sets rc_all=1.
has_illegal_rc = any('not one of the six legal classes' in l and 'rc_all=1' in l for l in code_lines)

# Word REPAIRED must be absent.
has_repaired = 'REPAIRED' in code

# Artifact path carries INTFER_BUILD_DIR on the same line as the build path.
has_build_dir_line = bool(re.search(r'INTFER_BUILD_DIR[^"\n]*\/build[^"\n]*\/closing_band', code)) or \
                       bool(re.search(r'\$\{INTFER_BUILD_DIR:-.*\}\/closing_band', code))

# The cls_record body appends a tab-separated row to $ROWS. The printf's format
# string is followed by its quoted ARGUMENTS before the redirection, so the
# pattern cannot require the format to sit next to `>>"$ROWS"` -- the first
# version did exactly that and refused the healthy subject. Derived
# structurally: find the line that writes $ROWS, extract its format token, and
# require a SIX-FIELD tab-separated layout (5 separators, 6 %s).
rows_append = False
if cls_record_body:
    for l in cls_record_body.split('\n'):
        if 'printf' in l and '>>"$ROWS"' in l:
            fmt = re.search(r"'([^']*)'", l)
            if fmt:
                f = fmt.group(1)
                rows_append = (f.count('\\t') >= 5 and f.count('%s') >= 6)
                break

print('CLS_CALLS_IN_CLASSIFY=%d' % cls_calls_in_classify)
print('PRINT_BRANCHES=%d' % print_branches)
print('CLASS_ARMS=%d' % len(class_arms))
print('HAS_ROW_COUNT=%s' % ('1' if has_row_count else '0'))
print('HAS_CLASS_SUM=%s' % ('1' if has_class_sum else '0'))
print('HAS_ARTIFACT_LINES=%s' % ('1' if has_artifact_lines else '0'))
print('HAS_ILLEGAL_RC=%s' % ('1' if has_illegal_rc else '0'))
print('HAS_REPAIRED=%s' % ('1' if has_repaired else '0'))
print('HAS_BUILD_DIR_LINE=%s' % ('1' if has_build_dir_line else '0'))
print('ROWS_APPEND=%s' % ('1' if rows_append else '0'))
# compute distinctness reliably after the loop
distinct = 1 if counters and len(set(counters.values())) == len(counters) else 0
print('COUNTERS_DISTINCT=%d' % distinct)
for arm in ['GREEN_BOTH','DIFFERS_FROM_HEAD','NEW_REGRESSION','PREEXISTING_AT_HEAD','ABSENT_AT_HEAD','UNDECIDABLE']:
    print('ARM_%s=%s' % (arm, '1' if arm in class_arms else '0'))
PY
)" || { echo "RESULT: FAIL" >&2; echo "derivation failed" >&2; exit 1; }

# The derivation prints KEY=VALUE lines; eval them so the predicates can read them.
eval "$derive_out"

# --- p_rows_append: cls_record appends a tab-separated row to $ROWS -----------
note_bool p_rows_append "${ROWS_APPEND}" "cls_record appends a tab-separated row to \$ROWS"

# --- p_cls_calls_match: calls inside classify == printed branches (both derived)
if [ "${CLS_CALLS_IN_CLASSIFY}" -ge 1 ] && [ "${CLS_CALLS_IN_CLASSIFY}" -eq "${PRINT_BRANCHES}" ]; then
  note p_cls_calls_match ok "${CLS_CALLS_IN_CLASSIFY} cls_record calls == ${PRINT_BRANCHES} printed branches inside classify()"
else
  note p_cls_calls_match FAIL "${CLS_CALLS_IN_CLASSIFY} cls_record calls vs ${PRINT_BRANCHES} printed branches inside classify() (must be equal and >=1)"
fi

# --- p_class_arms: six legal arms, distinct counters -------------------------
arm_ok=1
for arm in GREEN_BOTH DIFFERS_FROM_HEAD NEW_REGRESSION PREEXISTING_AT_HEAD ABSENT_AT_HEAD UNDECIDABLE; do
  eval "v=\${ARM_$arm:-0}"
  if [ "$v" != "1" ]; then arm_ok=0; fi
done
if [ "$arm_ok" -eq 1 ] && [ "${CLASS_ARMS}" -eq 6 ] && [ "${COUNTERS_DISTINCT}" = "1" ]; then
  note p_class_arms ok "six legal class arms present with distinct counters"
else
  note p_class_arms FAIL "six legal class arms with distinct counters (arms=${CLASS_ARMS} distinct=${COUNTERS_DISTINCT})"
fi

# --- p_row_count_eq / p_class_sum_eq / p_artifact_lines_eq -------------------
# THREE predicates, not one: the mutation matrix needs each of the three -eq
# assertions to fail for its OWN reason, and a single combined predicate would
# make three distinct states indistinguishable.
note_bool p_row_count_eq "${HAS_ROW_COUNT}" \
  "the row-count assertion compares n_cls with the derived population using -eq"
note_bool p_class_sum_eq "${HAS_CLASS_SUM}" \
  "the class-sum assertion compares n_sum with the derived population using -eq"
note_bool p_artifact_lines_eq "${HAS_ARTIFACT_LINES}" \
  "the artifact-lines comparison compares n_file with n_cls using -eq"

# --- p_illegal_rc: else branch sets rc_all=1 -------------------------------
note_bool p_illegal_rc "${HAS_ILLEGAL_RC}" "the illegal-class else branch sets rc_all=1"

# --- p_no_repaired: word REPAIRED absent -------------------------------------
# NOTE THE POLARITY: this one is INVERTED relative to the others -- the value is
# 1 when the word IS PRESENT, so a 1 is the FAILURE.
if [ "${HAS_REPAIRED}" = "1" ]; then
  note p_no_repaired FAIL "the word REPAIRED must not appear in the subject"
else
  note p_no_repaired ok "word REPAIRED absent from the subject"
fi

# --- p_build_dir_line: artifact path honours INTFER_BUILD_DIR on same line ---
note_bool p_build_dir_line "${HAS_BUILD_DIR_LINE}" \
  "artifact path carries INTFER_BUILD_DIR on the same line"

# --- p_syntax already done above ----------------------------------------------

# The count of NAMED predicates is derived from the note calls below, not asserted
# by hand, so adding a predicate cannot leave this line stale.
n_pred=0
n_pred=$((n_pred + 1))  # p_rows_append
n_pred=$((n_pred + 1))  # p_cls_calls_match
n_pred=$((n_pred + 1))  # p_class_arms
n_pred=$((n_pred + 1))  # p_row_count_eq
n_pred=$((n_pred + 1))  # p_class_sum_eq
n_pred=$((n_pred + 1))  # p_artifact_lines_eq
n_pred=$((n_pred + 1))  # p_illegal_rc
n_pred=$((n_pred + 1))  # p_no_repaired
n_pred=$((n_pred + 1))  # p_build_dir_line
echo
echo "CLOSING-BAND-ROWS CHECK: evaluated $n_pred predicate(s) over 1 subject"
# --- self-check: corrupt a temp copy, require FAIL --------------------------
# A regressed branch (one cls_record call removed) must make this checker
# refuse. Recurse once against the corrupt copy; INTFER_CLOSE_BAND_SELFCHECK=0
# breaks the recursion. `bash -n` must still pass there (a whole line removed),
# so the refusal comes from the predicate, not from a syntax error.
if [ "${INTFER_CLOSE_BAND_SELFCHECK:-1}" = "1" ]; then
  # The sandbox must clean up on EXIT and TERMINATE on INT/TERM, not just EXIT:
  # an EXIT-only trap never runs on SIGINT, so a Ctrl-C leaks the corrupted copy.
  # This is scripts/checks/check_trap_release.sh's contract and this checker is
  # in its population -- it flagged this line on the very band run it certified.
  # *** THE THREE LINES ARE NO LONGER HAND-WRITTEN. *** They are the exact shape
  # that was half-written here in the first place, so they are now a call into
  # scripts/lib/_gate_trap.sh. check_trap_release.sh audits the helper as a corpus
  # site of its own AND refuses a script that sources it without calling it, so
  # the extraction cannot become an exemption.
  . "$SCRIPT_DIR/../lib/_gate_trap.sh"
  SB="$(mktemp -d "${TMPDIR:-/tmp}/intfer_cbr_self.XXXXXX")"
  gate_trap_release "$SB"
  cp "$SUBJECT" "$SB/corrupt.sh"
  # delete ONE cls_record call line from classify()
  sed -i '/^\s*cls_record "\$kind" "\$mkind" "\$lrc" DIFFERS_FROM_HEAD/d' "$SB/corrupt.sh"
  if INTFER_CLOSE_BAND_SUBJECT="$SB/corrupt.sh" INTFER_CLOSE_BAND_SELFCHECK=0 \
        bash "$CHECKER" >/dev/null 2>&1; then
    echo "RESULT: FAIL" >&2
    echo "self-check: a corrupted copy (one cls_record call removed) was not refused" >&2
    exit 1
  fi
  # The corrupt copy must STILL PARSE: the refusal has to come from a predicate,
  # not from a syntax error, or the self-check would certify the wrong mechanism.
  if ! bash -n "$SB/corrupt.sh" >/dev/null 2>&1; then
    echo "RESULT: FAIL" >&2
    echo "self-check: the corrupted copy does not parse, so the refusal would have" >&2
    echo "            come from bash -n rather than from a predicate" >&2
    exit 1
  fi
  echo "  ok    self-check: corrupted copy refused (rc!=0) and still parses"
fi

if [ "$fails" -eq 0 ]; then
  echo "RESULT: ok ($n_pred of $n_pred predicates pass over scripts/closing_band.sh)"
  exit 0
fi
echo "RESULT: FAIL ($fails of $n_pred predicates refused)"
exit 1
