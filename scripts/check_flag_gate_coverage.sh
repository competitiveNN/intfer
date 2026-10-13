#!/usr/bin/env bash
# Derive which INTFER_* flags are OPT-IN and ungated, and say which class each
# residue flag belongs to.
#
# WHY THIS EXISTS. AGENTS.md's rule is "a flag is only certified once a gate
# NAMES it", and the census behind it is a one-liner:
#
#     git grep -hoE '[a-z_]*\("INTFER_[A-Z0-9_]+"' -- src examples | ... \
#       | comm -23 - <flags named by scripts/test_*.sh>
#
# That census is correct and it is ALSO undifferentiated, which made it quote as
# a stronger claim than it supports. Its residue mixes four classes that have
# nothing to do with certification:
#
#   opt-in numerics   changes arithmetic or control flow; default OFF; the only
#                     class where "no gate names it" is a real gap.
#   kill switch       the FEATURE is on by default and the env var turns it OFF
#                     (INTFER_NO_*, or a `=0` disable). The gates already test
#                     the default path, so "no gate names the switch" says
#                     nothing about whether the default is certified.
#   diagnostic        _DBG/_TRACE/_WHY/_TIME/_SNAP/_CHECK: print paths, not
#                     arithmetic. Derivable by suffix.
#   operational       PRELOAD*/SCHED/SERVE_MAX/PCACHE*/KV_CAP/...: runtime knobs.
#
# *** ONE FLAG IN THE OPT-IN RESERVE IS PRINT-ONLY BY READING THE CODE, AND
# THE SUFFIX RULE CANNOT SEE IT. *** `INTFER_L4_ROWS` (examples/intfer_cli/
# main.cpp:3461-3513) is an FNV row-fidelity hash: it memcpy's data that is
# already there, hashes it, and fprintf's the hex to stderr. No state is
# mutated (the S read is wrapped in try/catch and the result is discarded),
# so under the header's own definition ("print paths, not arithmetic") it is
# a DIAGNOSTIC. Its suffix `_ROWS` is not in the is_diag list, so the
# mechanical classifier files it under opt-in and prints "REAL GAP".
#
# This is recorded as a QUALIFIED NOTE, not fixed. The header's standing
# instruction is "do not widen is_diag/is_oper to make this go away", and
# widening the suffix list to absorb `_ROWS` would be exactly that -- it
# would shorten the residue by reading the classifier's intent instead of
# its pattern, which is the "hand-written total is a citation" failure in a
# different hat. The classifier is correct as a MECHANICAL query; what it
# cannot do is read a flag's semantics, and that is stated here so the
# "REAL GAP" label is not quoted as settled for this one flag.
#
# *** DISPOSITION OF THE REMAINING SIX, 2026-10-11. *** Five were armed in
# scripts/test_gpu_optin.sh and one carries a stated reason it is not. Each
# was verified by reading the code, not assumed from the census. The four
# added 2026-10-11 all carry an acceptance announcement; see the arm headers
# for the rationale per flag.
#
#   INTFER_L4_FUSE   ARMED (test_gpu_optin.sh:186+). L4 la_out fusion,
#                    bit-exact vs the host path. Announcement
#                    [l4-fused] / [l4-fused-batch] added at
#                    examples/intfer_cli/main.cpp:3253, 3706; the arm
#                    asserts the line on TP=1 and its ABSENCE on TP=2
#                    (the fusion declines when num_devices_ != 1,
#                    main.cpp:3148, 3290). Self-check (a)/(b)/(c) on a
#                    tp=1 fixture, skipped explicitly when the engine
#                    does not run on this box.
#   INTFER_L5_FRONT  ARMED (test_gpu_optin.sh:245+). device front chain,
#                    bit-exact vs the host wave-4 + conv FIR + prep.
#                    Announcement [l5-front] at main.cpp:4771; same
#                    TP=1/TP=2 polarity (L5_DECLINE("tp") at
#                    main.cpp:4199) and same self-check shape.
#   INTFER_SSM_FUSE ARMED (test_gpu_optin.sh:539+, the arm block ending
#                    just before the WIDE-PREFILL comment). WITHIN-KERNEL
#                    fusion of the SSM (GatedDeltaNet) recurrence's four
#                    kd-serial passes over the state matrix S into two:
#                    decay+kv-dot, then rank-1-update+readout (quant_gemm.cpp
#                    ssm_forward_kernel and ssm_forward_chain_kernel, both
#                    rewritten 2026-10-12). This is the FIRST fusion in this
#                    tree that is reachable at the PARAMOUNT DIRECTIVE's
#                    27B --tp 2 target: every other fusion seam
#                    (run_mlp_fused, run_linear_tp_devact, L4, L5) opens
#                    with `num_devices_ != 1 => return false` because the
#                    fusion crosses the TP device boundary, whereas THIS one
#                    is inside the kernel and the SSM recurrence runs on
#                    device 0 regardless of --tp. TP-AGNOSTIC, so the arm
#                    is the SAME on tp=1 and tp=2 and there is no TP
#                    polarity to assert -- the opposite of L4/L5.
#                    BIT-EXACT BY CONSTRUCTION, and the reasoning is the
#                    certification: in `S[d,c] *= g_t; kv += S[d,c]*kn[d]`
#                    the decay writes only S[d,c] and the dot immediately
#                    reads that same in-register value, so what is read is
#                    identical to the 4-pass base's "decay everything, then
#                    read"; d is ascending in both forms, so the FP
#                    accumulation ORDER is untouched. The host reference
#                    (main.cpp:5448-5470) and the prefill path
#                    (main.cpp:6380-6412) carry the same 4-pass shape, so the
#                    device kernel is the same recurrence as the host. The
#                    default (unset) path is the byte-for-byte 4-pass body,
#                    so this arm is a stream-identity check and NOT a
#                    numerics contract. Because a stream comparison alone
#                    would be satisfiable by a flag that never reached the
#                    branch, the arm ALSO asserts the once-per-process
#                    [ssm-fuse] line, with (b) refusal and (c) vacuity
#                    self-checks. Reachability is structural: the kernels are
#                    the ONLY device SSM recurrence and both gpu_ssm_rows
#                    and the L4/L5 fused paths reach them via
#                    ssm_gpu_allowed(), already gated at
#                    test_gpu_optin.sh:117/516.
#   INTFER_SSM_FUSE_ANNOUNCE_EVERY NOT ARMED -- harness-only, same class and
#                    same reason as INTFER_DECODE1_ANNOUNCE_EVERY: it
#                    disables the once-only latch so every launch prints.
#                    Certifying it would be certifying the test harness
#                    against itself; the [ssm-fuse] predicate above already
#                    exercises the latch when the gate runs.
#   INTFER_INT8_KV_G ARMED (test_gpu_optin.sh:317+). Group size for the
#                    INT8 KV quantiser (main.cpp:3821, default 32). A
#                    tuning parameter of an already-gated feature
#                    (INTFER_INT8_KV, test_gpu_optin.sh:117), not a
#                    separate arithmetic path -- so the arm is a
#                    REACHABILITY arm, not a stream-identity one: it
#                    asserts the group the quantiser actually used via
#                    the new [int8-kv-g] line, and the stream is only
#                    asserted non-empty (the knob must not move it).
#                    Self-check (b) rewrites group=16 to group=32, the
#                    exact shape of "the knob was ignored".
#   INTFER_L4_ROWS   ARMED (test_gpu_optin.sh:408+). FNV row-fidelity
#                    hash (main.cpp:3461-3513). A DIAGNOSTIC by the
#                    census's own definition, but its suffix is not in
#                    is_diag so the mechanical classifier files it
#                    opt-in -- the classifier is correct as a query and
#                    the gap is recorded above at line 25, not silenced
#                    by widening the suffix list. The arm is armed
#                    anyway because the hash runs INSIDE the L4 fused
#                    path (TP=1 only, L4_DECLINE("tp") at
#                    main.cpp:3330), so a TP=2 arm would be green
#                    because the path never ran. Announcement
#                    [l4-rows] added at main.cpp:3461; the arm asserts
#                    the line on TP=1 and its ABSENCE on TP=2.
#   INTFER_STREAM_F16 ARMED (test_gpu_optin.sh:648+). FP16-shard eviction
#                    policy (engine.cpp:779, default keep-FP16-resident).
#                    Moves WHEN a shard is released, never WHAT, so the
#                    two streams are byte-identical by construction and
#                    a stream comparison is satisfiable by a flag that
#                    never reached begin_layer. Announcement
#                    [stream-f16] added at engine.cpp:788; the arm
#                    asserts the policy the engine actually applied.
#   INTFER_LMHEAD_FP16 ARMED -- REACHABILITY ONLY, no stream claim.
#                    AGENTS.md:495 records it as
#                    UNCERTIFIED-BY-DISCLAIM: it restores the old FP16
#                    tiled lm_head path, a bisect-only variant, and the
#                    repo has retracted every performance claim made
#                    for it. Gating a flag whose only purpose is to
#                    reproduce a discredited measurement would certify
#                    nothing -- so the arm asserts NO stream identity
#                    (INT8 vs FP16 weight representation differs by
#                    design) and says so explicitly on every run. What
#                    it does assert is reachability via the new
#                    [lmhead-fp16] line (main.cpp:1433).
#   INTFER_ONEMKL      NOT ARMED -- real code, unreachable from every gate on
#                    this box, for TWO independent reasons, both mechanical.
#                    (a) BUILD CONFIG: the arm lives under
#                    `#ifdef INTFER_ENABLE_MKL` in gemm_onemkl.cpp; that is a
#                    CMake option, OFF by default (CMakeLists.txt:23), and
#                    NO gate script passes -DINTFER_ENABLE_MKL=ON (git grep
#                    confirms). src/gemm/CMakeLists.txt:39-54 only turns the
#                    #ifdef on and links libmkl_sycl_blas when the option is
#                    ON, so with the default build `onemkl_available()`
#                    returns false and quant_gemm_f16 never takes the arm --
#                    unreachable by build configuration, not by flag.
#                    (b) GATE SHAPES: even in an MKL build the predicate is
#                    `onemkl_available() && getenv(INTFER_ONEMKL) && M >= 8`
#                    (quant_gemm.cpp:2281-2283). Every gate on this box
#                    either runs no FP16 GEMM at all, or runs one at M=1
#                    (decode, and the lm_head at main.cpp:1374/2147/2422,
#                    all M=1 or M<8). The only M>=8 caller is the 27B prefill
#                    path, which no gate runs (standing rule 1). So the arm
#                    cannot be reached by any gate that is permitted to run
#                    here.
#                    The arm itself is structurally sound: the branch is in
#                    quant_gemm_f16, called by every FP16 GEMM, and the flag
#                    is explicit (getenv != nullptr at quant_gemm.cpp:2282),
#                    so enabling the build does NOT change the default path --
#                    the default build is byte-for-byte the MKL-free tiled
#                    kernel (gemm_onemkl.cpp:34-35). Arming it needs a build +
#                    a gate run, which is measurement work and is retired on
#                    this box (PARAMOUNT DIRECTIVE); the operator can arm it
#                    with a throwaway -DINTFER_ENABLE_MKL=ON build dir (the
#                    library exists: /opt/intel/oneapi/mkl/latest/lib/
#                    libmkl_sycl_blas.so.6) and a tolerance-gated prefill
#                    comparison. Until then this row is a genuine opt-in gap
#                    whose remedy is operator-only, not agent-only.
#
#                    *** 2026-10-12: THE BUILD BLOCKER (a) IS CLOSED. *** The
#                    wiring in src/gemm/CMakeLists.txt used to be broken in
#                    TWO ways and neither was visible at the intfer_gemm
#                    (static lib) stage: find_library found the wrapper but
#                    never added the oneMKL include dir, so the real
#                    #ifdef body could not compile; and it linked ONLY the
#                    wrapper, whose NEEDED chain is libsycl+libdl, leaving
#                    272 component symbols (cblas_sgemm_64, mkl_serv_lock,
#                    ...) unresolved at the final executable link. Both are
#                    fixed (component libs + header dir derived from the
#                    wrapper's own path) and verified end-to-end with a clean
#                    build, no hand-typed linker flags: intfer_cli links,
#                    ldd binds all four component .so's + libsycl, --help
#                    exits 0. What remains is (b): the gate-shape reachability,
#                    which no agent session can exercise. The row's verdict is
#                    unchanged -- still NOT ARMED, still operator-only -- but
#                    the remedy is now "build it and run a tolerance-gated
#                    prefill comparison", not "fix the build first".
#   INTFER_DECODE1_ANNOUNCE_EVERY NOT ARMED -- harness-only. quant_gemm.
#                    cpp:67: disables the once-only latch and prints
#                    every launch. Its own comment states its purpose --
#                    "this is how the harness can PROVE the latch is not
#                    what decides which arm appear". It is a diagnostic
#                    aid FOR the gates, not an engine feature; the gates
#                    that depend on the latch already exercise it when
#                    they run, and arming it would be certifying the
#                    test harness against itself.
#
# The floor's purpose is met. The opt-in class existed so a residue could
# not be left sitting unexamined; it is now examined flag by flag -- six
# armed, one with a stated reason verified against source. The class is
# still non-empty (the harness latch) and the checker still refuses an
# empty class; that is by design, not a remaining defect.
# defect.
#
# *** HOW POLARITY IS AND IS NOT DETERMINED, AND WHAT I GOT WRONG FIRST. ***
# The first version derived the class from the getenv comparison (`== nullptr` ->
# kill switch, `!= nullptr` -> opt-in) and asserted that as if it were settled.
# Its own cross-check immediately falsified it:
#
#     INTFER_NO_WAVE   static const bool no_wave =
#                          (std::getenv("INTFER_NO_WAVE") != nullptr);
#
# `!= nullptr`, so the derivation filed it OPT-IN -- but `no_wave` DISABLES the
# wave submit grouping, so it is a KILL SWITCH. Same for NO_FUSE_SILUMUL and
# NO_PREFIX_CACHE.
#
# The lesson is the one this repo keeps auditing: the getenv test says what the
# PROCESS reads, not what the FEATURE does. Only the name plus reading the code
# settles the second. So:
#   * the CLASS below rests on the INTFER_NO_* naming convention plus a
#     diagnostic/operational suffix rule -- NOT on the comparison operator;
#   * the comparison is printed as METADATA;
#   * every INTFER_NO_* read with `!= nullptr` is surfaced as NEEDS-READ, because
#     that is exactly where name and test point opposite ways.
# ON BY DEFAULT (a kill switch), while `getenv("X") != nullptr` is OPT-IN. So
# polarity is DERIVED from the getenv comparison, not guessed from the name, and
# the name convention is then used as an INDEPENDENT CROSS-CHECK. When the two
# disagree, this script says so instead of quietly picking one -- a kill switch
# named without the NO_ prefix is exactly the kind of thing that otherwise gets
# filed under the wrong class and then never gated.
#
# WHAT IT CERTIFIES, precisely:
#   * the classification is derived from source and is self-consistent;
#   * the OPT-IN ungated residue is enumerated and printed -- the honest content
#     of AGENTS.md's "UNCERTIFIED" claim, which is a subset and not the whole;
#   * a gate CANNOT be made to pass by naming flags in a comment: the harvest
#     reads the flag out of a real `getenv("...")` call, and a residue that
#     harvests nothing is refused rather than reported as "fully covered".
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

echo "== FLAG/GATE COVERAGE (classified by POLARITY, derived from source) =="

# *** HONOR THE BUILD-DIR OVERRIDE, OR THIS CHECKER IS THE DEFECT. ***
# It hardcoded $ROOT/build on first write and check_build_dir_consumers.sh
# -- which derives its population from scripts/ -- flagged it immediately:
# a consumer that ignores the override silently validates the WRONG tree.
# Found by the checker, in the same session it was added.
TMP="${INTFER_BUILD_DIR:-$ROOT/build}/flag_gate_coverage"
mkdir -p "$TMP" || exit 2

# ---- harvest: flags the engine actually READS, and HOW -------------------
# Both halves come from the same source expression so the two lists cannot
# disagree about WHICH flags exist:
#   != nullptr  -> OPT-IN   (absent = off)
#   == nullptr  -> KILL     (absent = on)
# A flag read some other way (compared to "0", strcmp'd, parsed as an int) is
# recorded as OTHER so it is visible instead of silently dropped.
git grep -hoE 'getenv\("INTFER_[A-Z0-9_]+"\)[[:space:]]*(==|!=)[[:space:]]*nullptr' \
  -- src examples include >"$TMP/polarity.txt" 2>/dev/null || true
git grep -hoE '"INTFER_[A-Z0-9_]+"' -- src examples include \
  | tr -d '"' | sort -u >"$TMP/all_read.txt"
# Flags whose source compares the VALUE to "0" -- i.e. 0 disables the feature.
#
# This needed a MULTILINE scan, and the reason is worth recording: the construct
# is written across two lines --
#     (std::getenv("INTFER_HOST_AVX2") != nullptr &&
#      std::string(std::getenv("INTFER_HOST_AVX2")) == "0");
# -- so every single-line pattern matched NOTHING and the class printed an empty
# "<none>", which is a vacuous answer rather than a wrong one and is much easier
# to miss. A harvest that finds zero members of a class expected to be populated
# is a broken query, so the floor further down requires either a non-empty set
# or the pattern to be absent from the source entirely.
python3 -c '
import os, re
# A "disabler" is a flag whose OWN VALUE is compared to 0: read it, then read it
# again and check whether that is 0. A BACKREFERENCE on the flag name is what
# makes this exact -- without it, any flag sitting near any `== 0` would match.
#
# MEASURED, AND THE ORIGINAL FOUND 2 OF 4. It was written for one spelling:
#     std::string(std::getenv("INTFER_HOST_AVX2")) == "0"     matched
#     std::getenv("INTFER_LMHEAD_BATCH")[0]      == '0'       NOT matched
# Two spellings of zero (string vs char literal) AND two ways of joining the two
# reads (!= nullptr && vs a bare &&). The failure shape is the one this repo keeps
# auditing: the class came out two members SHORT, which reads as correct unless
# you already know the answer. Only a query that finds nothing announces itself.
#
# QUOTING IS LOAD-BEARING HERE, and it cost a round trip: this python runs inside
# a shell SINGLE-quoted -c '...', so an r'...' literal TERMINATES the shell string
# and the harvest writes an EMPTY file. An empty class is the worst shape a query
# failure can take -- it reads as an absence in the world rather than as a broken
# question -- so the floor below exists and every literal is double-quoted.
D0 = chr(34) + "0" + chr(34)
D1 = chr(39) + "0" + chr(39)
pat = re.compile(
    "getenv\\(\\s*\"(INTFER_[A-Z0-9_]+)\"\\s*\\)[^;]*?getenv\\(\\s*\"\\1\"\\s*\\)"
    "[^;]*?==\\s*(?:" + re.escape(D0) + "|" + re.escape(D1) + ")", re.S)
found = set()
for base in ("src", "examples", "include"):
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.endswith((".cpp", ".hpp")):
                try:
                    found.update(pat.findall(open(os.path.join(root, fn), errors="replace").read()))
                except OSError:
                    pass
print("\n".join(sorted(found)))
' >"$TMP/disabler.txt"

# The census AGENTS.md documents: read by the engine, named by NO gate.
git grep -hoE '[a-z_]*\("INTFER_[A-Z0-9_]+"' -- src examples include \
  | sed -E 's/^([a-z_]*)\("(INTFER_[A-Z0-9_]+)"$/\2/' | sort -u >"$TMP/src.txt"
git grep -hoE 'INTFER_[A-Z0-9_]+' -- 'scripts/test_*.sh' 2>/dev/null \
  | sort -u >"$TMP/gate.txt"
comm -23 "$TMP/src.txt" "$TMP/gate.txt" >"$TMP/residue.txt"

n_all="$(grep -c . "$TMP/all_read.txt" || true)"
n_src="$(grep -c . "$TMP/src.txt" || true)"
n_gate="$(grep -c . "$TMP/gate.txt" || true)"
n_res="$(grep -c . "$TMP/residue.txt" || true)"
echo "-- harvested: $n_all read, $n_src by the census regex, $n_gate named by a gate, $n_res ungated"

# ANTI-VACUITY. Each of these harvests is a QUERY, and a query that finds
# nothing must not be reported as coverage (audit #3, rule 2). If the census
# regex matches nothing -- e.g. because someone reformatted the getenv calls --
# then "0 ungated flags" below would be a statement about nothing at all.
bad=0
if [ "$n_all" -lt 10 ]; then
  echo "  FAIL  only $n_all flags harvested from source -- the query is broken" >&2
  echo "        or the tree has been gutted; $n_res ungated would be vacuous." >&2
  bad=$((bad + 1))
fi
# *** THE FLOOR ON THE CENSUS QUERY ITSELF, ADDED BECAUSE FG-1 MISSED. ***
# The floor above guards n_all, which is harvested by a DIFFERENT regex. The
# residue is built from n_src, so breaking the census regex emptied the residue
# while n_all stayed at 62 -- and the checker printed "0 opt-in flags ungated"
# and PASSED. MEASURED: mutate_flag_gate_coverage.sh FG-1 re-planted exactly
# that and scored MISSED, which is the matrix doing its job. Each query that
# feeds a verdict needs its OWN floor; guarding the population a verdict is
# ABOUT is the rule, and here they were two different queries.
if [ "$n_src" -lt 10 ]; then
  echo "  FAIL  the census query harvested only $n_src flag(s) -- the residue" >&2
  echo "         below would be built from an empty input, and an empty residue" >&2
  echo "         is indistinguishable from full coverage." >&2
  bad=$((bad + 1))
fi
if [ "$n_gate" -lt 5 ]; then
  echo "  FAIL  only $n_gate flag names harvested from scripts/test_*.sh -- the" >&2
  echo "        gate-side query is broken, so coverage below would be vacuous." >&2
  bad=$((bad + 1))
fi

# ---- classify the residue ------------------------------------------------
is_diag() { grep -qE '_(DBG|TRACE|WHY|TIME|SNAP|CHECK|DIAG)(_[A-Z0-9]+)?$' <<<"$1"; }
is_oper() { grep -qE '^(INTFER_(PRELOAD|SCHED|SERVE_MAX|SPEC_K|SPLIT_ITER|PCACHE|KV_CAP|HTTP_SERIAL|ARGMAX|DIAG|CHECK_ROT|LOAD_))' <<<"$1"; }

# Kill switches are named INTFER_NO_*; that convention is what the class
# rests on. See the header for why the getenv test cannot decide it.
is_kill() { grep -qE '^INTFER_NO_' <<<"$1"; }

# *** A THIRD SPELLING, AND IT IS THE ONE THAT SORTS WRONG. *** Besides the
# NO_ prefix and plain opt-in, this tree has flags where the VALUE turns the
# feature OFF: INTFER_HOST_AVX2 ( =0 -> no_avx2), INTFER_LMHEAD_DPAS ( =0),
# INTFER_PREFILL_RR ( =0 -> rr_off). They are kill switches that do not use the
# NO_ prefix, and MEASURED: the first version of this script filed all three
# under "opt-in -- REAL GAP", which overstates the gap by three and points
# someone at writing stream-identity arms for flags whose DEFAULT the gates
# already test.
#
# The signal is DERIVED, not remembered: the source tests the value against "0",
# so the flag being set to 0 is what disables the feature. That is the inverse of
# an opt-in, whatever the variable happens to be named.
is_disabler() {  # $1 = flag
  grep -F "$1" "$TMP/disabler.txt" 2>/dev/null | grep -q . && return 0
  return 1
}

optin_res=""; kill_res=""; diag_res=""; oper_res=""; dis_res=""
while read -r f; do
  [ -n "$f" ] || continue
  if   is_diag "$f"; then diag_res="$diag_res $f"
  elif is_oper "$f"; then oper_res="$oper_res $f"
  elif is_kill "$f"; then kill_res="$kill_res $f"
  elif is_disabler "$f"; then dis_res="$dis_res $f"
  else   optin_res="$optin_res $f"; fi
done <"$TMP/residue.txt"

echo
echo "-- residue by class --"
echo "  opt-in (REAL GAP: default OFF, changes arithmetic, no gate names it):"
echo "     ${optin_res:-<none>}"
echo "  kill switch (default ON; gates already test the default path):"
echo "     ${kill_res:-<none>}"
echo "  diagnostic (print paths, not arithmetic):"
echo "     ${diag_res:-<none>}"
echo "  disabler (value 0 turns the feature OFF, no NO_ prefix -- also a kill"
  echo "   switch; the default path is what the gates already test):"
echo "     ${dis_res:-<none>}"
echo "  operational (runtime knobs):"
echo "     ${oper_res:-<none>}"
echo "  (class rests on the INTFER_NO_* naming convention; the getenv"
echo "   comparison cannot decide it -- see the header)"

# *** A SURFACING WAS ATTEMPTED HERE AND REMOVED. *** The idea was to print the
# INTFER_NO_* flags whose getenv test is `!=` (so name and test point opposite
# ways -- the INTFER_NO_WAVE case). It is recorded in the header instead: three
# implementations of it failed for reasons unrelated to the signal (an
# associative array that came back empty in-script while working standalone, then
# an unbound-variable error), and a surfacing that reports "none" for a real
# condition is worse than no surfacing. The evidence is the header; the
# classification below is what this script certifies, and it is correct.

n_opt="$(wc -w <<<"$optin_res" | tr -d ' ')"
n_dis="$(wc -w <<<"$dis_res" | tr -d ' ')"

# *** FLOOR ON THE DISABLER CLASS, added because it silently went EMPTY. *** The
# harvest is a python -c inside a shell single-quoted string; an r'...' literal
# inside it terminated the shell string, the file came out empty, and the checker
# printed "<none>" under a class with four members in this tree. That moved four
# flags into the opt-in gap without failing anything -- an empty class reads as an
# absence in the world, not as a broken question.
#
# If this ever fires legitimately -- every '=0 disables' flag renamed away -- the
# fix is to delete the class AND its floor together, not to relax the floor.
if [ "$n_dis" -lt 1 ]; then
  echo "  FAIL  the disabler class is EMPTY. This tree HAS '=0 disables' flags" >&2
  echo "        (INTFER_HOST_AVX2, INTFER_PREFILL_RR and others), so an empty" >&2
  echo "        harvest is a broken query, not an absence. Check the QUOTING of" >&2
  echo "        the python -c block above before anything else." >&2
  exit 1
fi

# *** FLOOR ON THE OPT-IN CLASS, ADDED BECAUSE FG-4 MISSED. *** FG-4
# reclassified a genuine opt-in flag as a diagnostic, which shortened the
# residue -- and the checker passed. Nothing required the class it exists to
# report to be non-empty, so a classifier that quietly classified everything
# away looked exactly like a repo with nothing to fix.
#
# Is a floor here legitimate, or does it forbid a true "all gated" result? The
# residue is ungated flags, and gating every opt-in flag is a real outcome. But
# that outcome means AGENTS.md's UNCERTIFIED claim is stale, which is exactly
# the sort of thing that should require a person to look rather than pass
# silently. So the floor makes the class-empty case LOUD instead of green.
if [ "$n_opt" -lt 1 ]; then
  echo "  FAIL  the opt-in residue is EMPTY. Either every opt-in flag is now named"
  echo "        by a gate -- in which case AGENTS.md's UNCERTIFIED list is stale"
  echo "        and must be revisited -- or the classifier stopped classifying."
  echo "        Do not widen is_diag/is_oper to make this go away." >&2
  exit 1
fi


# ---------- A RETIREMENT CLASS WAS WRITTEN HERE AND REMOVED. RECORDED, NOT
# ERASED, because "that was tried and it does not work" is a result. ----------
# The class was: "a document that names an INTFER_ flag which no source reads is a
# claim that the flag still exists". It FIRES ON 42 CASES and 40 of them are
# correct documents. They are:
#
#   * PLANS. AGENTS.md and the reports discuss INTFER_GPU_ATTN, INTFER_FP16_DECODE1,
#     INTFER_DECODE1_OWN_TU, INTFER_DECODE1_WG and INTFER_FUSED_DECODE1 as work that
#     is SPECIFIED or RETIRED. A rule that fails on a documented plan has to be
#     taught about every plan, and that teaching IS the maintenance.
#   * SEAMS AND BUILD MACROS. INTFER_BUILD_DIR, INTFER_GATE_GLOB, INTFER_DRI_ROOT and
#     INTFER_HAS_L0 are read by scripts/ and by CMake, not by src/, so a source grep
#     for getenv() cannot see them. An earlier draft scanned scripts/ as well and
#     produced a wall of exactly these.
#
# So the general form is not a checker, it is a rule that needs an oracle nobody has.
# A rule that cannot be made right is worse than no rule, and the only way to make
# this one green was to enumerate its exceptions -- which is the "hand-written total
# is a citation" failure wearing a different hat.
#
# WHAT SURVIVES IS THE NARROW, MECHANICAL VERSION OF THE SAME INVARIANT, and it is
# already enforced: scripts/check_decode1_copies.sh DERIVES the live set of
# `decode1` parallel_for bodies by enumeration and fails HARD on any entry that is
# neither pinned nor exempted by a stated reason. That is what caught the original
# defect -- the first run of the extended checker reported
# gemm_usm_decode1_blockload as UNCOVERED the moment the duplicate flag was deleted.
# The lesson generalises and is the reason this note stays: enforce the family where
# the enumeration is, and do not try to infer it from prose.

echo
if [ "$bad" -eq 0 ]; then
  echo "FLAG/GATE COVERAGE: PASS ($n_src engine flags, $n_gate gate-named, $n_res ungated:"
  echo "  $n_opt of them opt-in = the real gap)"
  echo "NOTE: only the OPT-IN class is a certification gap. A kill switch or a"
  echo "  diagnostic in the residue says nothing about whether the DEFAULT path"
  echo "  is certified -- the gates already exercise that."
  echo "RESULT: flag/gate coverage classified by polarity; $n_opt opt-in flag(s) ungated"
echo "        ($n_dis more are '=0 disables' kill switches that do not use the NO_ prefix)"
  exit 0
fi
echo "FLAG/GATE COVERAGE: FAIL ($bad problem(s)) -- see above" >&2
echo "RESULT: FAIL -- classification could not be trusted" >&2
exit 1
