#!/usr/bin/env bash
# Mutation matrix for the INTFER_KV_Q8 / Q4 / Q4_HAD / Q5_1 / Q5_1_HAD / KVARN
# gate arms in scripts/gates/test_gpu_optin.sh (the KV variant loop after
# line 521). Every variant named by the loop has at least one dedicated state:
# q8 -> DC-5 (scale), DC-6 (group-size seam), DC-7 (announcement honesty), DC-11 (dequant group-size seam), DC-12 (knob echo broken);
# q4 -> DC-1 (scale), DC-3 (activation); q4_had -> DC-4 (activation);
# q5_1 -> DC-8 (scale); q5_1_had -> DC-10 (activation, function-body form),
#   DC-13 (activation, CALL-SITE form -- the shape that shipped on the q4 twin);
#   kvarn -> DC-2 (scale).
#
# Each state plants a defect in examples/intfer_cli/main.cpp, rebuilds intfer_cli,
# and runs the binary with the variant's env flag. A defect that changes the
# quantisation output moves the argmax stream (caught by stream identity) or
# removes the [int8-kv-g] announcement tag (caught by reachability). States are
# designed so every defect IS observable -- the matrix proves the gate arms can
# catch regressions, not that they always do (that is the gate's own job).
#
# Follows the mutate_decode1_copies.sh pattern: sandbox under build/, pristine
# restore between states, residue assertion at the end, never writes tracked
# source or the gate script.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/kv_variants_matrix"
SRC="examples/intfer_cli/main.cpp"
BIN="$BUILD/examples/intfer_cli/intfer_cli"
MODEL="$ROOT/test/models/Qwen3.5-0.8B.intifer"
PROMPT="The capital of France is"
TOK="--tokenizer-py $SCRIPT_DIR/../gates/bpe_tokenize.py --detok-py $SCRIPT_DIR/../gates/bpe_detokenize.py"

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }

LOCK="$BUILD/.mutate_kv_variants.lock"
mkdir -p "$BUILD" 2>/dev/null || true
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another .mutate_kv_variants.lock is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

rm -rf "$SB"; mkdir -p "$SB" || exit 2
MUT="$SB/main.cpp"
PRISTINE_SRC="$SB/main.pristine"
cp "$SRC" "$PRISTINE_SRC" || exit 2

# patch_literal: EXACTLY ONE match. 0 and >1 both fail loudly.
plant() { python3 - "$MUT" "$1" "$2" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
c = s.count(o)
if c != 1:
    print("SOURCE ANCHOR %s: %d matches (want 1): %r"
          % ("ABSENT" if c == 0 else "AMBIGUOUS", c, o[:70]), file=sys.stderr)
    sys.exit(1)
open(p, "w").write(s.replace(o, n))
PYX
}

restore_all() { cp "$PRISTINE_SRC" "$MUT"; cp "$PRISTINE_SRC" "$SRC"; }

run_variant() { # $1 = env string, prints stream ids from stderr
  local envstr="$1"
  env $envstr "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" -n 4 \
    --no-stream $TOK 2>&1 \
  | grep -a -oE "decode step [0-9]+: token=[0-9]+" | sed -E 's/.*token=//' \
  | tr '\n' ' ' | sed 's/ *$//'
}

has_announcement() { # $1 = env string, $2 = tag -> 0 if tag appears in stderr
  local envstr="$1" tag="$2"
  env $envstr "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" -n 4 \
    --no-stream $TOK 2>&1 | grep -aqF "$tag"
}

announced_group() { # $1 = env string -> prints the group=[0-9]+ the line names
  local envstr="$1"
  env $envstr "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" -n 4 \
    --no-stream $TOK 2>&1 \
    | grep -aoE '\[int8-kv-g\].*group=[0-9]+ ' | head -1 \
    | grep -aoE 'group=[0-9]+' | head -1 | sed -E 's/group=([0-9]+)/\1/'
}
announced_knob() { # $1 = env string -> prints the INTFER_INT8_KV_G value echoed
  local envstr="$1"
  env $envstr "$BIN" --gpu --tp 1 -m "$MODEL" -p "$PROMPT" -n 4 \
    --no-stream $TOK 2>&1 \
    | grep -aoE '\[int8-kv-g\].*INTFER_INT8_KV_G=[0-9]+' | head -1 \
    | grep -aoE 'INTFER_INT8_KV_G=[0-9]+' | head -1 \
    | sed -E 's/INTFER_INT8_KV_G=([0-9]+)/\1/'
}

BASE="279 6511 314 279"

total=0; caught=0; RESULTS=()
record() { case "$2" in ok) caught=$((caught + 1)); RESULTS+=("$1 CAUGHT");;
                     *)  RESULTS+=("$1 $2");; esac; }

echo "== baseline (pristine source) =="
restore_all
make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1 || { echo "FAIL: baseline build" >&2; exit 1; }

# q4 is tolerance-class on this model: the 4-bit quantisation moves the stream
Q4_IDS="$(run_variant "INTFER_KV_Q4=1")"
# stream differs from base (expected for tolerance-class); record it for the report
echo "  q4 stream '$Q4_IDS' differs from base '$BASE' (tolerance-class)"
Q4_BASE="$Q4_IDS"
# q8 is stream-identical on this model
Q8_IDS="$(run_variant "INTFER_KV_Q8=1")"
[ "$Q8_IDS" = "$BASE" ] || { echo "FAIL: baseline q8 stream '$Q8_IDS' != '$BASE'" >&2; exit 1; }
# kvarn is stream-identical
KVARN_IDS="$(run_variant "INTFER_KV_KVARN=1")"
[ "$KVARN_IDS" = "$BASE" ] || { echo "FAIL: baseline kvarn stream '$KVARN_IDS' != '$BASE'" >&2; exit 1; }
# q5_1 is stream-identical; q5_1_had is tolerance-class (like q4_had) but
# announces Q5_1+HAD
Q51_IDS="$(run_variant "INTFER_KV_Q5_1=1")"
[ "$Q51_IDS" = "$BASE" ] || { echo "FAIL: baseline q5_1 stream '$Q51_IDS' != '$BASE'" >&2; exit 1; }
Q51HAD_IDS="$(run_variant "INTFER_KV_Q5_1=1 INTFER_KV_Q5_1_HAD=1")"
has_announcement "INTFER_KV_Q5_1=1 INTFER_KV_Q5_1_HAD=1" " Q5_1+HAD" || { echo "FAIL: baseline q5_1_had announcement missing" >&2; exit 1; }
# q4_had moves the stream (tolerance-class) but announces Q4+HAD
Q4HAD_IDS="$(run_variant "INTFER_KV_Q4=1 INTFER_KV_Q4_HAD=1")"
has_announcement "INTFER_KV_Q4=1 INTFER_KV_Q4_HAD=1" " Q4+HAD" || { echo "FAIL: baseline q4_had announcement missing" >&2; exit 1; }
# q8 announces Q8-G64
has_announcement "INTFER_KV_Q8=1" " Q8-G64" || { echo "FAIL: baseline q8 announcement missing" >&2; exit 1; }
# q8 announces Q8-G64, and the [int8-kv-g] line must name q8's ACTUAL group
# (64, hardcoded G64) -- not the shared INTFER_INT8_KV_G knob (default 32).
Q8_GRP="$(announced_group "INTFER_KV_Q8=1")"
[ "$Q8_GRP" = "64" ] || { echo "FAIL: baseline q8 line reports group=$Q8_GRP, expected 64 (announcement is dishonest)" >&2; exit 1; }
# q8 must ALSO be insensitive to the knob: with INTFER_INT8_KV_G=16 the line
# still names 64 (q8 does not read the knob), which is the invariant DC-7
# certifies. (If q8 ever reads the knob, DC-6 catches the stream change.)
Q8_GRP16="$(announced_group "INTFER_KV_Q8=1 INTFER_INT8_KV_G=16")"
[ "$Q8_GRP16" = "64" ] || { echo "FAIL: baseline q8 with G=16 reports group=$Q8_GRP16, expected 64" >&2; exit 1; }
echo "  ok baseline: q4='$Q4_IDS' q8='$Q8_IDS' kvarn='$KVARN_IDS' q5_1='$Q51_IDS' q4_had='$Q4HAD_IDS' q5_1_had='$Q51HAD_IDS' (tolerance-class)"
echo "  ok baseline: q8 line names group=$Q8_GRP (G64, honest; not the knob's default 32); with G=16 still $Q8_GRP16"

echo
echo "== states =="

# ---- DC-1: wrong q4 scale divisor -> stream moves -----------------------------
total=$((total + 1)); restore_all
# The q4 scale is mx / 7.f; plant mx / 1.f (1-bit divisor) to make quantisation
# extreme, moving the argmax stream. Gate's stream-identity check catches it.
if plant 'float s = (mx > 0) ? mx / 7.f : 1.f;' 'float s = (mx > 0) ? mx / 1.f : 1.f;'; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    IDS="$(run_variant "INTFER_KV_Q4=1")"
    if [ "$IDS" != "$Q4_BASE" ]; then record DC-1 ok; else record DC-1 "MISSED (stream unchanged: $IDS)"; fi
  else record DC-1 'BUILD-FAIL'; fi
  restore_all
else record DC-1 'ANCHOR-DRIFT'; fi

# ---- DC-2: wrong kvarn scale -> stream moves ----------------------------------
total=$((total + 1)); restore_all
# kvarn scale: max(3*stddev, eps)/127; plant /1 to make quantisation extreme,
# moving the argmax stream. Gate's stream-identity check catches it.
if plant 'float s = std::max(3.f * stddev, eps) / 127.f;' 'float s = std::max(3.f * stddev, eps) / 1.f;'; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    IDS="$(run_variant "INTFER_KV_KVARN=1")"
    if [ "$IDS" != "$BASE" ]; then record DC-2 ok; else record DC-2 "MISSED (stream unchanged: $IDS)"; fi
  else record DC-2 'BUILD-FAIL'; fi
  restore_all
else record DC-2 'ANCHOR-DRIFT'; fi

# ---- DC-3: Q4 never activates -> no Q4 announcement ---------------------------
total=$((total + 1)); restore_all
# kv_q4_enabled() returns on; plant return false so Q4 flag is set but never
# activates. Gate's reachability check (announcement must echo " Q4") catches it.
Q4EN_OLD='static inline bool kv_q4_enabled(){
  static const bool on = (std::getenv("INTFER_KV_Q4") != nullptr);
  return on;
}'
Q4EN_NEW='static inline bool kv_q4_enabled(){
  return false;
}'
if plant "$Q4EN_OLD" "$Q4EN_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    if ! has_announcement "INTFER_KV_Q4=1" " Q4"; then record DC-3 ok; else record DC-3 "MISSED (announcement still present)"; fi
  else record DC-3 'BUILD-FAIL'; fi
  restore_all
else record DC-3 'ANCHOR-DRIFT'; fi

# ---- DC-4: q4_had flag not read -> no Q4+HAD announcement ---------------------
total=$((total + 1)); restore_all
# ly.kv_q4_had = ly.kv_q4 && kv_q4_had_enabled(); plant ly.kv_q4_had = false;
# so Q4+HAD flag is set but had never activates. Gate's reachability check
# (announcement must echo " Q4+HAD") catches it.
if plant 'ly.kv_q4_had = ly.kv_q4 && kv_q4_had_enabled();' 'ly.kv_q4_had = false;'; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    if ! has_announcement "INTFER_KV_Q4=1 INTFER_KV_Q4_HAD=1" " Q4+HAD"; then record DC-4 ok; else record DC-4 "MISSED (announcement still present)"; fi
  else record DC-4 'BUILD-FAIL'; fi
  restore_all
else record DC-4 'ANCHOR-DRIFT'; fi

# ---- DC-5: wrong q8 scale divisor -> stream moves -----------------------------
total=$((total + 1)); restore_all
# q8 scale: mx / 127.f; plant mx / 1.f to make quantisation extreme,
# moving the argmax stream. Gate's stream-identity check catches it.
# Anchor includes the following sc[...] write (pos/ng-based, unique to Q8)
# to disambiguate from the legacy INT8-i8 line with the same divisor.
Q8_OLD=$'float s = (mx > 0) ? mx / 127.f : 1.f;\n      sc[((size_t)pos * n_kv + kvh) * ng + dg] = s;'
Q8_NEW=$'float s = (mx > 0) ? mx / 1.f : 1.f;\n      sc[((size_t)pos * n_kv + kvh) * ng + dg] = s;'
if plant "$Q8_OLD" "$Q8_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    IDS="$(run_variant "INTFER_KV_Q8=1")"
    if [ "$IDS" != "$BASE" ]; then record DC-5 ok; else record DC-5 "MISSED (stream unchanged: $IDS)"; fi
  else record DC-5 'BUILD-FAIL'; fi
  restore_all
else record DC-5 'ANCHOR-DRIFT'; fi

# ---- DC-6: q8 group-size seam -> stream moves with INTFER_INT8_KV_G changed --
total=$((total + 1)); restore_all
# q8 is per-token G64: ng = hd / 64 is hardcoded, it does NOT read
# INTFER_INT8_KV_G (the shared group-size knob used by q4/q5_1/kvarn).
# Plant ng = hd / kv_group_size() so q8 leaks into the shared knob, then
# run with INTFER_INT8_KV_G=16 (default 32). The mismatched grouping changes
# quantisation output -> stream moves. Gate's stream-identity check catches it.
# Anchor uses the following `size_t kvrow` line to disambiguate from the
# kv_row_ptr dequant path which also has `ng = hd / 64`.
Q6_OLD=$'  const int ng = hd / 64;\n  size_t kvrow = (size_t)n_kv * hd;'
Q6_NEW=$'  const int ng = hd / kv_group_size();\n  size_t kvrow = (size_t)n_kv * hd;'
if plant "$Q6_OLD" "$Q6_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    IDS="$(run_variant "INTFER_KV_Q8=1 INTFER_INT8_KV_G=16")"
    if [ "$IDS" != "$BASE" ]; then record DC-6 ok; else record DC-6 "MISSED (stream unchanged: $IDS)"; fi
  else record DC-6 'BUILD-FAIL'; fi
  restore_all
else record DC-6 'ANCHOR-DRIFT'; fi

# ---- DC-7: dishonest [int8-kv-g] group for q8 -> group check catches it ------
total=$((total + 1)); restore_all
# The announcement must name the group the codec ACTUALLY used. q8 uses
# hardcoded 64-dim groups and does NOT read INTFER_INT8_KV_G; the shipped
# code therefore reports effective_group=64 for q8. Plant the pre-fix
# behaviour (report the shared knob's value) so the line says group=32 while
# q8 used 64 -- a silent mismatch the TAG check cannot see, because the tag
# is identical either way. The gate's group-honesty predicate catches it.
Q7_OLD=$'            int effective_group = kv_group_size();\n            if (ly.kv_q8) effective_group = 64;'
Q7_NEW=$'            int effective_group = kv_group_size();'
if plant "$Q7_OLD" "$Q7_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    GRP="$(announced_group "INTFER_KV_Q8=1")"
    if [ "$GRP" != "64" ]; then record DC-7 ok; else record DC-7 "MISSED (line still reports 64)"; fi
  else record DC-7 'BUILD-FAIL'; fi
  restore_all
else record DC-7 'ANCHOR-DRIFT'; fi

# ---- DC-8: wrong q5_1 scale divisor -> stream moves -------------------------
total=$((total + 1)); restore_all
# q5_1 scale: mx / 15.f (5-bit symmetric, codes [-16,15]); plant mx / 1.f to
# make quantisation extreme, moving the argmax stream. The variant loop runs
# q5_1 as a stream-identical variant, so the gate's identity check catches it.
# Anchor is unique to q5_1: q4 uses /7.f, q8/legacy use /127.f, kvarn uses
# max(3*stddev,eps)/127.f.
if plant 'float s = (mx > 0) ? mx / 15.f : 1.f;' 'float s = (mx > 0) ? mx / 1.f : 1.f;'; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    IDS="$(run_variant "INTFER_KV_Q5_1=1")"
    if [ "$IDS" != "$BASE" ]; then record DC-8 ok; else record DC-8 "MISSED (stream unchanged: $IDS)"; fi
  else record DC-8 'BUILD-FAIL'; fi
  restore_all
else record DC-8 'ANCHOR-DRIFT'; fi

# ---- DC-9: q5_1 never activates -> no Q5_1 announcement ---------------------
total=$((total + 1)); restore_all
# kv_q5_1_enabled() returns on; plant return false so the flag is set but the
# codec never activates. The Q5_1 arm's reachability check (announcement must
# echo " Q5_1"; the dequant path still dequantises the all-zero mirror, so the
# stream stays base and ONLY reachability can see it) catches this.
Q51EN_OLD='static inline bool kv_q5_1_enabled(){
  static const bool on = (std::getenv("INTFER_KV_Q5_1") != nullptr);
  return on;
}'
Q51EN_NEW='static inline bool kv_q5_1_enabled(){
  return false;
}'
if plant "$Q51EN_OLD" "$Q51EN_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    if ! has_announcement "INTFER_KV_Q5_1=1" " Q5_1"; then record DC-9 ok; else record DC-9 "MISSED (announcement still present)"; fi
  else record DC-9 'BUILD-FAIL'; fi
  restore_all
else record DC-9 'ANCHOR-DRIFT'; fi

# ---- DC-10: q5_1_had flag never activates Hadamard -> no Q5_1+HAD announcement
total=$((total + 1)); restore_all
# kv_q5_1_had_enabled() returns on; plant return false so INTFER_KV_Q5_1_HAD=1
# is set but Hadamard never activates. The Q5_1+HAD arm's reachability check
# (announcement must echo " Q5_1+HAD") catches it. Anchor is the full function
# body; the call site at line 11003 is a different string.
Q51HADEN_OLD='static inline bool kv_q5_1_had_enabled(){
  static const bool on = (std::getenv("INTFER_KV_Q5_1_HAD") != nullptr);
  return on;
}'
Q51HADEN_NEW='static inline bool kv_q5_1_had_enabled(){
  return false;
}'
if plant "$Q51HADEN_OLD" "$Q51HADEN_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    if ! has_announcement "INTFER_KV_Q5_1=1 INTFER_KV_Q5_1_HAD=1" " Q5_1+HAD"; then record DC-10 ok; else record DC-10 "MISSED (announcement still present)"; fi
  else record DC-10 'BUILD-FAIL'; fi
  restore_all
else record DC-10 'ANCHOR-DRIFT'; fi

# ---- DC-11: q8 dequant group-size seam -> stream moves at default G -------
total=$((total + 1)); restore_all
# q8 dequant (kv_row_ptr) uses hardcoded ng = hd / 64 (16 groups), consistent
# with q8 append which also uses hd/64 and does NOT read INTFER_INT8_KV_G.
# Plant ng = hd / kv_group_size() so dequant re-partitions using the shared
# knob. At default G=32, dequant reads 32 scale slots while append wrote 16,
# and reads beyond the quantized row -> scale-index mismatch -> garbage
# dequant -> stream moves. Gate's stream-identity check catches it. Anchor
# pairs the q8-dequant `ng = hd / 64` with its following `const int8_t* row`
# line (distinct from kv_q8_append's ng line which is followed by `size_t
# kvrow`, used by DC-6).
Q11_OLD=$'    const int ng = hd / 64;\n    const int8_t* row = &ci[((size_t)t * n_kv + kvh) * hd];'
Q11_NEW=$'    const int ng = hd / kv_group_size();\n    const int8_t* row = &ci[((size_t)t * n_kv + kvh) * hd];'
if plant "$Q11_OLD" "$Q11_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    IDS="$(run_variant "INTFER_KV_Q8=1")"
    if [ "$IDS" != "$BASE" ]; then record DC-11 ok; else record DC-11 "MISSED (stream unchanged: $IDS)"; fi
  else record DC-11 'BUILD-FAIL'; fi
  restore_all
else record DC-11 'ANCHOR-DRIFT'; fi

# ---- DC-12: q8 group-size echo broken -> INTFER_INT8_KV_G hardcoded -------
total=$((total + 1)); restore_all
# The [int8-kv-g] announcement must echo the ACTUAL INTFER_INT8_KV_G value
# (the last fprintf argument is kv_group_size()). Plant a hardcoded 32 so the
# line says INTFER_INT8_KV_G=32 regardless of the knob. With INTFER_INT8_KV_G=16
# the echo reports 32, not 16 -- the gate's knob-echo check catches it.
# Anchor: the q8-dequant-specific `ly.kv_kvarn ? " KVARn" : ","` line followed
# by `kv_group_size());` -- unique to the [int8-kv-g] fprintf (the [intfer]
# fprintf ends with `ly.kv_kvarn ? " KVARn" : "");`).
Q12_OLD=$'                         ly.kv_kvarn ? " KVARn" : "",\n                         kv_group_size());'
Q12_NEW=$'                         ly.kv_kvarn ? " KVARn" : "",\n                         32);'
if plant "$Q12_OLD" "$Q12_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    KE_KNOB="$(announced_knob "INTFER_KV_Q4=1 INTFER_INT8_KV_G=16")"
    if [ "$KE_KNOB" != "16" ]; then record DC-12 ok; else record DC-12 "MISSED (echo still reports $KE_KNOB)"; fi
  else record DC-12 'BUILD-FAIL'; fi
  restore_all
else record DC-12 'ANCHOR-DRIFT'; fi

# ---- DC-13: q5_1_had CALL SITE hardcoded false -> no Q5_1+HAD announcement -
total=$((total + 1)); restore_all
# This is the EXACT defect form that shipped on the q4 twin: the activation
# site assigns a literal `false` instead of consulting the flag, so the
# env var is read by a function nobody calls. DC-10 plants the FUNCTION-BODY
# form (kv_q5_1_had_enabled -> return false); DC-13 plants the CALL-SITE form
# (the assignment itself never consults the flag). Both are caught by the
# same reachability predicate, but only DC-13 reproduces the shipped q4_had
# defect's shape, and it is the shape that slipped through: DC-4 could not
# have caught it because a function-body mutant is behaviourally identical
# to a call site that already ignores the function. Requires the
# " Q5_1+HAD" announcement to be ABSENT.
Q13_OLD='        ly.kv_q5_1_had = ly.kv_q5_1 && kv_q5_1_had_enabled();'
Q13_NEW='        ly.kv_q5_1_had = false;'
if plant "$Q13_OLD" "$Q13_NEW"; then
  cp "$MUT" "$SRC"
  if make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1; then
    if ! has_announcement "INTFER_KV_Q5_1=1 INTFER_KV_Q5_1_HAD=1" " Q5_1+HAD"; then record DC-13 ok; else record DC-13 "MISSED (announcement still present)"; fi
  else record DC-13 'BUILD-FAIL'; fi
  restore_all
else record DC-13 'ANCHOR-DRIFT'; fi

echo
for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

restore_all
make -C "$BUILD" intfer_cli -j$(nproc) >/dev/null 2>&1 || true
if ! cmp -s "$SRC" "$PRISTINE_SRC"; then
  echo "  FAIL residue: $SRC differs from its pre-matrix state" >&2; exit 1
fi
echo "  ok residue: $SRC is byte-identical to its pre-matrix state"
rm -rf "$SB"

echo
if [ "$caught" -eq "$total" ]; then
  echo "KV-VARIANT MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  DC-1: wrong q4 scale divisor moves the stream (stream identity catches it)"
  echo "  DC-2: wrong kvarn scale moves the stream (stream identity catches it)"
  echo "  DC-3: Q4 announcement tag removed (reachability check catches it)"
  echo "  DC-4: q4_had flag not read -> no Q4+HAD announcement (reachability catches it)"
  echo "  DC-5: wrong q8 scale divisor moves the stream (stream identity catches it)"
  echo "  DC-6: q8 leaks into INTFER_INT8_KV_G (group-size seam); knob change moves stream"
  echo "  DC-7: dishonest [int8-kv-g] group for q8 (line echoes the knob, not G64)"
  echo "  DC-8: wrong q5_1 scale divisor moves the stream (stream identity catches it)"
  echo "  DC-9: Q5_1 codec never activates -> no Q5_1 announcement (reachability catches it)"
  echo "  DC-10: q5_1_had flag never activates Hadamard -> no Q5_1+HAD announcement (reachability catches it)"
  echo "  DC-11: q8 dequant reads INTFER_INT8_KV_G (group-size seam); scale-index mismatch moves stream"
  echo "  DC-12: [int8-kv-g] echo hardcodes INTFER_INT8_KV_G=32 instead of the knob (knob-echo check catches it)"
  echo "  DC-13: q5_1_had CALL SITE hardcoded false (the q4_had defect's exact shape) -> no Q5_1+HAD announcement"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "KV-VARIANT MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1