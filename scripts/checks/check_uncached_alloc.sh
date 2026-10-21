#!/usr/bin/env bash
# check_uncached_alloc.sh -- SOURCE-level structural certification of the S1
# uncached-host-USM ALLOCATOR in the engine (src/engine/engine.cpp), the half of
# S1 that actually shipped.
#
# WHY THIS EXISTS AND WHY IT IS NOT check_s1_predicates.sh. The judge asked for a
# source-level checker so S1 is not certified solely by the operator-only 27B
# gate. check_s1_predicates.sh already exists and certifies the PROBE
# (examples/uncached_doorbell_probe/main.cpp): the probe's own four structural
# guards and its eight predicates. It says NOTHING about the engine's allocator,
# which is a different file, a different subject, and a different set of
# failure modes -- so this is a NEW checker on a NEW subject, not a second
# opinion on the probe's. The name is deliberately NOT check_s1_predicates.sh
# and the matrix is deliberately NOT mutate_s1_predicates.sh: both of those
# names are taken by the probe's pair, and reusing either would make the
# coverage sweep's derive-by-stem rule map two different subjects onto one
# matrix.
#
# WHAT IS CERTIFIED, AND WHAT IS NOT. The S1 doorbell is a REFUSED CANDIDATE
# (upstream's own clone measures that a memory-only host spin never runs the
# kernel: "THE WAIT MUST POLL THE DRIVER, NOT MEMORY"), so there is no
# doorbell to certify. This checker certifies the ALLOCATOR half and the
# release discipline that makes it safe: an uncached buffer is a HOST ALLOCATION
# that zeMemFree UNMAPS, so a buffer freed while the device is still writing it
# faults the card. It grades TEXT, at source, with no execution -- so it certifies
# the guards, and the PERFORMANCE question stays the operator's, on nk/nw/nr and
# the host buckets, never on a tok/s.
#
# THE TEN PREDICATES. Each refuses the shape of a defect that was planted and
# measured (mutate_uncached_alloc.sh):
#   p_presence        the enable test is PRESENCE-spelled   (UC-presence)
#   p_default_off     the enable test has no disjunct        (UC-default-on)
#   p_act_excluded    .act is never routed through the
#                     uncached allocator                     (UC-act)
#   p_alloc_confined  zeMemAllocHost appears once, inside
#                     namespace uncached_out                 (UC-alloc-confine)
#   p_release_api     the release API is zeMemFree, once,
#                     inside release()                        (UC-release-api)
#   p_site_pairing    every .out free site has an uncached
#                     counterpart (DERIVED equality)          (UC-release-drop)
#   p_both_workspaces both workspace kinds are covered      (UC-site-drop)
#   p_provenance      provenance is PER SLOT, not a global
#                     pointer set                             (UC-provenance)
#   p_announce_echo   the announcement echoes the env value
#                     and is self-gated                       (UC-announce)
#   p_drain           the destructor drains before it
#                     releases an uncached buffer            (UC-drain)
#
# WHY COMMENTS ARE STRIPPED FIRST. The engine's own prose names the very
# strings the predicates look for -- "zeMemAllocHost + ZE_HOST_MEM_ALLOC_FLAG_
# BIAS_UNCACHED", "`.act` NEVER carries this flag" -- so an unstripped harvest
# counts a comment as code. MEASURED on the healthy source: zeMemAllocHost
# appears in a comment as well as at its one call site, and `.act` is named in
# the direction-constraint comment. Stripping preserves line numbers (sed
# substitutes in place), which the line-order predicates need.
#
# ANTI-VACUITY. A missing or empty subject is a FAIL, never a pass: "looked at
# nothing" and "found nothing wrong" are different claims. Every count that
# feeds a predicate is also PRINTED, so a predicate that passes because a
# harvest returned zero is visible in the output rather than silent.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUBJ="${1:-$ROOT/src/engine/engine.cpp}"
HDR="${2:-$ROOT/include/intfer/engine.hpp}"

die() { echo "UC-ALLOC: FAIL -- $*"; echo "RESULT: FAIL"; exit 1; }

[ -f "$SUBJ" ] || die "subject missing ($SUBJ)"
[ -s "$SUBJ" ] || die "subject empty ($SUBJ); a subject with nothing in it cannot hold a predicate and must not read as a pass"
[ -f "$HDR" ]  || die "header missing ($HDR)"
[ -s "$HDR" ]  || die "header empty ($HDR)"

CODE="$(mktemp "${TMPDIR:-/tmp}/ucalloc.XXXXXX")"
HDRC="$(mktemp "${TMPDIR:-/tmp}/ucalloc.XXXXXX")"
cleanup() { rm -f "$CODE" "$HDRC"; }
trap cleanup EXIT
# INT/TERM must ALSO reach the cleanup: an EXIT trap alone never runs on a
# signal (bash fires EXIT only at a reached end or an explicit exit), so a
# SIGTERM killed the script outright and leaked the temp files -- AUDIT #29.
# The exit routes the signal back through the EXIT handler; "clean up and
# resume" leaves the process alive, which is strictly worse
# (check_trap_release.sh).
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# Strip // line comments IN PLACE (line numbers preserved). The engine has no
# block comments and no '//' inside a string literal, so this cannot remove a
# real statement.
sed -e 's://.*::' "$SUBJ" > "$CODE"
sed -e 's://.*::' "$HDR"  > "$HDRC"

fails=()
note() { printf 'UC-ALLOC: %-20s %s\n' "$1" "$2"; }

# count of a FIXED string in a file
nfix() { grep -c -F -- "$2" "$1" 2>/dev/null || true; }

# --- 1. the enable test must be PRESENCE-spelled ----------------------------
if grep -Fq 'std::getenv("INTFER_UNCACHED_HOST_USM") != nullptr' "$CODE"; then
  note p_presence "ok (getenv(...) != nullptr)"
else
  note p_presence "FAIL -- the enable test is not the presence spelling"
  note "" "      getenv(\"INTFER_UNCACHED_HOST_USM\") != nullptr; a different"
  note "" "      spelling makes the flag's ON/OFF contract a second guess"
  fails+=(p_presence)
fi
if grep -Fq 'getenv("INTFER_UNCACHED_HOST_USM") == "1"' "$CODE"; then
  note p_presence "FAIL -- an exact-\"1\" spelling is ALSO present; two"
  note "" "      spellings of one flag is the shape of a flag nobody can arm"
  fails+=(p_presence)
fi

# --- 2. the enable test must have no disjunct (default OFF) ------------------
s1_body="$(awk '/inline bool s1_enabled\(\)/{f=1} f{print} f&&/^}/{exit}' "$CODE")"
if [ -z "$s1_body" ]; then
  note p_default_off "FAIL -- could not locate s1_enabled(); the enable test"
  note "" "      cannot be graded, and a predicate over a missing function is"
  note "" "      a predicate that cannot fire"
  fails+=(p_default_off)
elif printf '%s\n' "$s1_body" | grep -q '||'; then
  note p_default_off "FAIL -- the enable test has a disjunct; the flag can then"
  note "" "      be on without the env being set"
  note "" "      body: $(printf '%s' "$s1_body" | tr '\n' ' ' | tr -s ' ')"
  fails+=(p_default_off)
else
  note p_default_off "ok (single conjunct, no default-on path)"
fi

# --- 3. .act must never be routed through the uncached allocator -------------
# Two directions, because either alone is satisfiable by the wrong source:
# (a) no uncached allocation may LAND in an .act slot, and (b) every uncached
# allocation must LAND somewhere, i.e. it must set an out_uncached provenance
# flag -- otherwise the allocation leaks a host mapping with no release path.
n_alloc="$(grep -c -F 'uncached_out::alloc(' "$CODE" || true)"
n_prov_true="$(grep -c -F 'out_uncached = true' "$CODE" || true)"
act_hit="$(awk '
  /uncached_out::alloc\(/ { inalloc=1; base=NR; next }
  inalloc { if (NR-base<=4 && $0 ~ /\.act[[:space:]]*=/) c++; if (NR-base>4) inalloc=0 }
  END { print c+0 }' "$CODE")"
if [ "$act_hit" -eq 0 ] && [ "$n_alloc" -eq "$n_prov_true" ] && [ "$n_alloc" -ge 1 ]; then
  note p_act_excluded "ok ($n_alloc allocation site(s), none landing in .act)"
else
  note p_act_excluded "FAIL -- act_hit=$act_hit alloc_sites=$n_alloc provenance_sets=$n_prov_true"
  note "" "      .act is HOST-WRITES / DEVICE-READS and BIAS_UNCACHED gives no"
  note "" "      DEVICE-side guarantee, so an uncached .act can hand the device"
  note "" "      a stale line; and an allocation with no provenance flag is a"
  note "" "      host mapping with no release path"
  fails+=(p_act_excluded)
fi

# --- 4. the allocation must be confined to the helper namespace -------------
# THE BOUNDS ARE TAKEN FROM THE RAW FILE, not the stripped one, and that is not
# cosmetic: the namespace's closing line is `}  // namespace uncached_out`, so
# the marker LIVES IN A COMMENT and the comment-stripping pass removes it. A
# predicate that looked for its marker in the stripped copy would find no
# closing bound at all and could not tell "the namespace is unterminated" from
# "I looked in the wrong file". Line numbers are preserved by the stripping
# pass, so the raw line numbers apply to the stripped copy unchanged.
ns_open="$(grep -n -F 'namespace uncached_out {' "$SUBJ" | head -1 | cut -d: -f1)"
ns_close="$(grep -n -F '}  // namespace uncached_out' "$SUBJ" | head -1 | cut -d: -f1)"
# The CALL is counted, not the bare identifier: the announcement's own format
# string names the API ("ACTIVE alloc=zeMemAllocHost|BIAS_UNCACHED"), so
# counting the identifier would see two sites on the HEALTHY source and the
# predicate would be wrong on a correct file.
alloc_ln="$(grep -n -F 'zeMemAllocHost(' "$CODE" | head -1 | cut -d: -f1)"
n_zealloc="$(nfix "$CODE" 'zeMemAllocHost(')"
if [ "$n_zealloc" -eq 1 ] && [ -n "$alloc_ln" ] && [ -n "$ns_open" ] && [ -n "$ns_close" ] \
   && [ "$alloc_ln" -gt "$ns_open" ] && [ "$alloc_ln" -lt "$ns_close" ]; then
  note p_alloc_confined "ok (one zeMemAllocHost, at line $alloc_ln, inside the namespace)"
else
  note p_alloc_confined "FAIL -- zeMemAllocHost count=$n_zealloc line=${alloc_ln:-none}"
  note "" "      namespace=[$ns_open..$ns_close]; the uncached allocation must"
  note "" "      exist in exactly one place, or a second one would silently"
  note "" "      create a buffer with no release path"
  fails+=(p_alloc_confined)
fi

# --- 5. the release API must be zeMemFree ------------------------------------
rel_ln="$(grep -n -F 'zeMemFree(' "$CODE" | head -1 | cut -d: -f1)"
n_zefree="$(nfix "$CODE" 'zeMemFree(')"
rel_open="$(grep -n -F 'inline void release(' "$CODE" | head -1 | cut -d: -f1)"
rel_close="$(awk 'NR>'"${rel_open:-0}"' && /^}/{print NR; exit}' "$CODE")"
if [ "$n_zefree" -eq 1 ] && [ -n "$rel_ln" ] && [ -n "$rel_open" ] && [ -n "$rel_close" ] \
   && [ "$rel_ln" -gt "$rel_open" ] && [ "$rel_ln" -lt "$rel_close" ]; then
  note p_release_api "ok (one zeMemFree, at line $rel_ln, inside release())"
else
  note p_release_api "FAIL -- zeMemFree count=$n_zefree line=${rel_ln:-none} release()=[${rel_open:-none}..${rel_close:-none}]"
  note "" "      a Level-Zero host allocation must be released with zeMemFree;"
  note "" "      sycl::free on it is not the same release path"
  fails+=(p_release_api)
fi

# --- 6. every .out free site must have an uncached counterpart (DERIVED) -----
# The count is DERIVED from the source, not hand-written: the number of
# sycl::free() calls on an .out buffer must EQUAL the number of
# uncached_out::release() calls. A dropped release shows up as a mismatch, and
# so does a stray extra one. Each release must also be provenance-guarded, so
# the equality cannot be satisfied by a release that frees an ordinary buffer.
n_outfree="$(grep -c -E 'sycl::free\(.*\.out' "$CODE" || true)"
n_relcall="$(grep -c -F 'uncached_out::release(' "$CODE" || true)"
unguarded="$(awk '
  /uncached_out::release\(/ {
    # The CURRENT line counts: three of the four release sites carry the
    # provenance guard on the same line (`if (x.out_uncached)
    # uncached_out::release(...))`), so a window that started at NR-1 would
    # report three of four releases as unguarded on the HEALTHY source.
    hit=0
    if ($0 ~ /out_uncached/) hit=1
    for (i=NR-2; i<NR; i++) if (i>0 && prev[i] ~ /out_uncached/) hit=1
    if (!hit) c++
  }
  { prev[NR]=$0 }
  END { print c+0 }' "$CODE")"
if [ "$n_outfree" -ge 1 ] && [ "$n_relcall" -eq "$n_outfree" ] && [ "$unguarded" -eq 0 ] \
   && [ "$n_alloc" -ge 2 ]; then
  note p_site_pairing "ok (alloc sites=$n_alloc, release sites=$n_relcall, .out frees=$n_outfree, all provenance-guarded)"
else
  note p_site_pairing "FAIL -- alloc_sites=$n_alloc release_sites=$n_relcall out_free_sites=$n_outfree unguarded_releases=$unguarded"
  note "" "      every place an .out buffer is freed must have an uncached"
  note "" "      counterpart, and every uncached release must be provenance-guarded"
  fails+=(p_site_pairing)
fi

# --- 7. both workspace kinds must be covered --------------------------------
# The wave slot and the per-device workspace are DIFFERENT slots with different
# lifetimes, so an allocator applied to one and not the other would leave a
# seam reading a buffer that was never uncached.
# A textual check cannot see REACHABILITY (a branch guarded by `if (false &&`
# still reads as covered), so coverage is asserted on the ASSIGNMENT SITES: the
# uncached pointer must actually LAND in a WaveSlot member and in a Workspace
# member, each chained off its own allocation. `s.` and `w.` are the two
# workspace variables, so the assignment lines are distinct and greppable.
wave_ok=0; dev_ok=0
wave_ln="$(grep -n -F 's.out = static_cast<std::uint16_t*>(uc);' "$CODE" | head -1 | cut -d: -f1)"
dev_ln="$(grep -n -F 'w.out = static_cast<std::uint16_t*>(uc);' "$CODE" | head -1 | cut -d: -f1)"
[ -n "$wave_ln" ] && [ "$(awk -v l="$wave_ln" 'NR>=l-5 && NR<=l && /uncached_out::alloc\(/{c++} END{print c+0}' "$CODE")" -ge 1 ] && wave_ok=1
[ -n "$dev_ln" ]  && [ "$(awk -v l="$dev_ln"  'NR>=l-5 && NR<=l && /uncached_out::alloc\(/{c++} END{print c+0}' "$CODE")" -ge 1 ] && dev_ok=1
n_wave_true="$(nfix "$CODE" 'announce("wave", true)')"
n_dev_true="$(nfix "$CODE" 'announce("device", true)')"
if [ "$wave_ok" -eq 1 ] && [ "$dev_ok" -eq 1 ] && [ "$n_wave_true" -eq 1 ] && [ "$n_dev_true" -eq 1 ]; then
  note p_both_workspaces "ok (wave slot line $wave_ln, device slot line $dev_ln, both chained off an alloc)"
else
  note p_both_workspaces "FAIL -- wave_slot=${wave_ln:-none} device_slot=${dev_ln:-none} wave_announce=$n_wave_true device_announce=$n_dev_true"
  note "" "      the wave slot and the per-device workspace are different slots"
  note "" "      with different lifetimes; covering one and not the other leaves"
  note "" "      a seam reading a buffer that was never uncached"
  fails+=(p_both_workspaces)
fi

# --- 8. provenance must be PER SLOT, not a global pointer set ---------------
ws_body="$(awk '/struct Workspace \{/{f=1} f{print} f&&/^  \};/{exit}' "$HDRC")"
wv_body="$(awk '/struct WaveSlot \{/{f=1} f{print} f&&/^  \};/{exit}' "$HDRC")"
glob_hits="$(grep -c -E 'unordered_set<|unordered_map<|set<void|map<void' "$CODE" || true)"
if [ -n "$ws_body" ] && [ -n "$wv_body" ] \
   && printf '%s\n' "$ws_body" | grep -q 'out_uncached' \
   && printf '%s\n' "$wv_body" | grep -q 'out_uncached' \
   && [ "$glob_hits" -eq 0 ]; then
  note p_provenance "ok (per-slot flags on Workspace and WaveSlot, no global pointer set)"
else
  note p_provenance "FAIL -- Workspace_flag=$(printf '%s' "$ws_body" | grep -c 'out_uncached') WaveSlot_flag=$(printf '%s' "$wv_body" | grep -c 'out_uncached') global_container_hits=$glob_hits"
  note "" "      a GLOBAL provenance set is what upstream's clone uses, and it"
  note "" "      makes every free site an implicit lookup; per-slot flags keep"
  note "" "      every free site explicit and greppable"
  fails+=(p_provenance)
fi

# --- 9. the announcement must echo the env value and be self-gated ----------
ann_body="$(awk '/inline void announce\(/{f=1} f{print} f&&/^}/{exit}' "$CODE")"
ann_gate_ln="$(printf '%s\n' "$ann_body" | grep -n 's1_enabled()' | head -1 | cut -d: -f1)"
ann_fprintf_ln="$(printf '%s\n' "$ann_body" | grep -n 'fprintf' | head -1 | cut -d: -f1)"
ann_echo="$(printf '%s\n' "$ann_body" | grep -c 'getenv("INTFER_UNCACHED_HOST_USM")' || true)"
ann_fmt="$(printf '%s\n' "$ann_body" | grep -c 'INTFER_UNCACHED_HOST_USM=%s' || true)"
if [ -n "$ann_body" ] && [ "$ann_echo" -ge 1 ] && [ "$ann_fmt" -ge 1 ] \
   && [ -n "$ann_gate_ln" ] && [ -n "$ann_fprintf_ln" ] && [ "$ann_gate_ln" -lt "$ann_fprintf_ln" ]; then
  note p_announce_echo "ok (echoes the env value, self-gated at line $ann_gate_ln before the print at $ann_fprintf_ln)"
else
  note p_announce_echo "FAIL -- echo=$ann_echo fmt=%ann_fmt gate=${ann_gate_ln:-none} print=${ann_fprintf_ln:-none}"
  note "" "      the announcement must print the value the operator set (so the"
  note "" "      log says WHICH spelling was used) and must be gated on the env"
  note "" "      READ inside the announcement, not only at its call site -- the"
  note "" "      [decode1-wg] line once printed on the BASE run and broke a gate"
  fails+=(p_announce_echo)
fi

# --- 10. the destructor must drain before it releases an uncached buffer -----
dt_open="$(grep -n -F 'Engine::~Engine()' "$CODE" | head -1 | cut -d: -f1)"
dt_body="$(awk -v s="${dt_open:-0}" 'NR>s{ if (/^}/) exit; print }' "$CODE")"
dt_wait="$(printf '%s\n' "$dt_body" | grep -n 'queue.wait()' | head -1 | cut -d: -f1)"
dt_rel="$(printf '%s\n' "$dt_body" | grep -n 'uncached_out::release(' | head -1 | cut -d: -f1)"
if [ -n "$dt_body" ] && [ -n "$dt_wait" ] && [ -n "$dt_rel" ] && [ "$dt_wait" -lt "$dt_rel" ]; then
  note p_drain "ok (queue.wait() at $dt_wait precedes the first release at $dt_rel)"
else
  note p_drain "FAIL -- destructor wait=${dt_wait:-none} first_release=${dt_rel:-none}"
  note "" "      zeMemFree UNMAPS a host allocation, so an Engine torn down with"
  note "" "      work in flight would unmap a buffer the device is still writing"
  fails+=(p_drain)
fi

evaluated=10
echo "UC-ALLOC: evaluated $evaluated predicate(s) on $(basename "$SUBJ")"
if [ "${#fails[@]}" -eq 0 ]; then
  echo "UC-ALLOC: PASS (all $evaluated predicates hold)"
  echo "RESULT: PASS"
  exit 0
fi
echo "UC-ALLOC: FAIL -- refused by: ${fails[*]}"
echo "RESULT: FAIL"
exit 1
