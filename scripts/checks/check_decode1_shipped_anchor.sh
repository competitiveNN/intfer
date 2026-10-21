#!/usr/bin/env bash
# check_decode1_shipped_anchor.sh
#
# Pins the bisect-3 `SHIPPED` arm (examples/roofline_probe_bisect3/main.cpp,
# arm_shipped) to the SHIPPED INT2-tab path in src/gemm/quant_gemm.cpp's
# gemm_usm_decode1_vec. The bisect-3 binary measures the SHIPPED/VERBATIM ratio
# within one TU so that "the ~4.5-5x of the 5.6-6.4x gap that survives the module
# move (1.26x) and the runtime branch (1.007x) sits between VERBATIM and SHIPPED" is
# measured rather than assumed -- but a ratio against a cloned body is only as good
# as the clone's fidelity, so this check exists to make that fidelity DERIVED, not
# remembered.
#
# WHAT IT CHECKS. Five load-bearing invariants of the engine's INT2-tab path that
# the SHIPPED arm MUST preserve (one drift = the anchor desyncs and the ratio is
# read against a phantom):
#
#   I1  the chunk reads EXACTLY TWO weight bytes per 8-code window
#       (`b0v = ...; b1v = ...b0 + 1; ... b0 += 2`), i.e. bytes_per_chunk == 2.
#   I2  the byte index is ABSOLUTE per group: `(row_base + kg) / 4` (or `pack`),
#       never a cumulative offset -- the double-offset (ROWPAIR) defect class.
#   I3  the INT2 decode is the literal 4-code-packed-per-byte shift
#       `(bv >> ((j & 3) * 2)) & 3` (or the equivalent within*n form with bits==2).
#   I4  the canonical 4-entry weight table
#       `{scale * -1.0f, scale * (-1.0f / 3.0f), scale * (1.0f / 3.0f), scale * 1.0f}`.
#   I5  the tail uses ABSOLUTE indexing `idx = row_base + kg`, `code_u8[idx / pack]`
#       (or `idx / 4`) -- the same invariant as the chunk, for the ragged tail.
#
# EXIT: 0 = all invariants hold in BOTH bodies; 1 = an invariant drifted in either
# direction (the clone and the engine must track each other). Each invariant is
# required with EXACTLY ONE match -- "0 matches" (drift) and "2 matches" (ambiguity)
# both fail loudly.
#
# NON-DESTRUCTIVE: read-only; never writes the tracked tree.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

ENGINE="src/gemm/quant_gemm.cpp"
ANCHOR="examples/roofline_probe_bisect3/main.cpp"

[ -f "$ENGINE" ] || { echo "FAIL: $ENGINE not found" >&2; exit 2; }
[ -f "$ANCHOR" ] || { echo "FAIL: $ANCHOR not found" >&2; exit 2; }

# ---- extract the engine INT2-tab body (use_int2_tab && bits == 2 branch) -------
# The branch opens at `if (use_int2_tab && bits == 2) {` (line 1574) and closes at
# the next `} else {` (line 1606). Use awk to slice it. Both bounds are distinctive
# literals in this TU.
ENGINE_BODY="$(awk '/if \(use_int2_tab && bits == 2\) \{/{f=1; next} f && /} else \{/{exit} f' "$ENGINE")"
[ -n "$ENGINE_BODY" ] || {
  echo "FAIL: engine INT2-tab body not extracted (anchor line drifted?)" >&2; exit 1; }

# sanity: the table-hoist comment is the landmark for this branch. It lives just
# before the branch (line 1560), so a file-level check is the correct one -- if
# the anchor drifts or the comment vanishes, extraction must fail. The extracted
# body must also start with the wtab definition, proving we grabbed the table-
# hoist block and not the sibling `else` path.
grep -q 'INT2 table-hoist path' "$ENGINE" || {
  echo "FAIL: engine anchor landmark missing -- extraction may have drifted" >&2; exit 1; }
printf '%s' "$ENGINE_BODY" | head -1 | grep -q 'const float wtab\[4\]' || {
  echo "FAIL: extracted body is not the table-hoist INT2 block (expected wtab def)" >&2; exit 1; }

# ---- extract the bisect3 SHIPPED arm (arm_shipped function body) ----------------
# Match the DEFINITION (opening brace on the signature line), not the forward
# declaration earlier in the TU: a forward-decl match pulls a stub and all five
# invariant checks read phantom source. The engine side has no such decl, so this
# one pattern is the whole asymmetry.
ANCHOR_BODY="$(awk '/^sycl::event arm_shipped\(.*\{$/{f=1} f{print} f && /^}/{exit}' "$ANCHOR")"
[ -n "$ANCHOR_BODY" ] || {
  echo "FAIL: arm_shipped not extracted (function name drifted?)" >&2; exit 1; }

exactly_one() { # $1 = label, $2 = body-var, $3 = pattern
  local label="$1" body="$2" pat="$3"
  local n
  # PCRE, not ERE. GNU grep ERE has two traps here that cost a full round:
  #   * `&` is a WHOLE-LINE anchor, so the INT2 shift `(j & 3)` matches nothing;
  #   * a literal-paren character class `[(]` immediately followed by a group
  #     `(...)` misbehaves when there is a prefix (e.g. `bv >> [(](j & 3)`).
  # PCRE treats `()` as literals when escaped and `&` as a literal, so the
  # patterns below read like the code they pin. Caller must regex-escape the
  # few remaining metacharacters (`[]{}.*+?^$|`).
  n="$(printf '%s' "$body" | grep -cP -- "$pat")"
  case "$n" in
    1) printf '  ok    %s\n' "$label" ;;
    0) echo "FAIL: I/$label violated: pattern not found ($pat)" >&2
       echo "      The clone and the engine must both contain this invariant exactly" >&2
       echo "      once; 0 matches means it has drifted in the source." >&2
       exit 1 ;;
    *) echo "FAIL: I/$label ambiguous: pattern matched $n times ($pat)" >&2
       echo "      An ambiguous pattern means the extractor grabbed too much; the" >&2
       echo "      invariant is not pinned down." >&2
       exit 1 ;;
  esac
}

echo "== checking the SHIPPED anchor is pinned to the engine INT2-tab path =="
echo "  engine body: src/gemm/quant_gemm.cpp :: gemm_usm_decode1_vec :: use_int2_tab"
echo "  anchor     : examples/roofline_probe_bisect3/main.cpp :: arm_shipped"

# I1: two weight byte reads per chunk + b0 += 2. The engine writes the index form
# `(row_base + (std::size_t)kg) / (std::size_t)pack`; the clone writes the inlined
# `(row_base + static_cast<std::size_t>(kg)) / 4u` / `(row_base + kg)`. The invariant
# is the chunk reads themselves (b0v/b1v -> v0/v1 + b0 += 2), which are identical
# in form. The engine's chunk uses b0v/b1v, the clone's SHIPPED arm uses v0/v1.
exactly_one "I1-chunk-byte-reads (engine)" "$ENGINE_BODY" 'b0v = code_u8\[b0\]'
exactly_one "I1-chunk-byte-reads (anchor)" "$ANCHOR_BODY" 'v0 = c\.code\[b0\]'

# I2: absolute byte index per group. Engine: `(row_base + (std::size_t)kg) /
# (std::size_t)pack`. Clone: `(row_base + static_cast<std::size_t>(kg)) / 4u`. The
# shared invariant: the divisor is 4 (pack=4 for INT2) and the dividend INCLUDES
# row_base; a cumulative (row-pair-style) offset would NOT have row_base here.
# PCRE: `\(`/`\)` are literal parens, so the source reads verbatim.
exactly_one "I2-absolute-index (engine)" "$ENGINE_BODY" '\(row_base \+ \(std::size_t\)kg\) / \(std::size_t\)pack'
exactly_one "I2-absolute-index (anchor)" "$ANCHOR_BODY" '\(row_base \+ .*kg.*\) / 4u'

# I3: the INT2 4-coded-per-byte decode. Both bodies use `(bv >> ((j & 3) * 2)) & 3`.
exactly_one "I3-code-extract (engine)"  "$ENGINE_BODY" '\(bv >> \(\(j & 3\) \* 2\)\) & 3'
exactly_one "I3-code-extract (anchor)" "$ANCHOR_BODY" '\(bv >> \(\(j & 3\) \* 2\)\) & 3'

# I4: the canonical 4-entry weight table {scale*-1, scale*(-1/3), scale*(1/3), scale*1}.
# The engine's table spans two lines (definition + first value), the clone's is one
# line; checking the wtab[4] definition and the scale*-1.0f value anchors both.
exactly_one "I4-wtable (engine)"  "$ENGINE_BODY" 'wtab\[4\] = \{'
exactly_one "I4-wtable (anchor)"  "$ANCHOR_BODY" 'wtab\[4\] = \{'
exactly_one "I4-scale-neg (engine)"  "$ENGINE_BODY" 'scale \* -1\.0f'
exactly_one "I4-scale-neg (anchor)" "$ANCHOR_BODY" 'scale \* -1\.0f'

# I5: the tail absolute index. Engine tail: `idx = row_base + kg`, `code_u8[idx / pack]`.
# Clone tail: `idx = row_base + kg`, `c.code[idx / 4u]`.
exactly_one "I5-tail-idx (engine)"  "$ENGINE_BODY" 'idx = row_base \+ kg'
exactly_one "I5-tail-idx (anchor)" "$ANCHOR_BODY" 'idx = row_base \+ static_cast<std::size_t>\(kg\)'
exactly_one "I5-tail-byte (engine)"  "$ENGINE_BODY" 'code_u8\[idx / pack\]'
exactly_one "I5-tail-byte (anchor)" "$ANCHOR_BODY" 'c\.code\[idx / 4u\]'

echo
echo "DECODE1 SHIPPED ANCHOR: PASS (5 invariants hold in both the engine and the clone)"
echo "  The bisect-3 SHIPPED arm tracks the engine's INT2-tab path. If this check"
echo "  fails, the bisect-3 ratio is measured against a phantom and must not be read."
exit 0
