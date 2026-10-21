# PLAN — intfer optimization backlog (2x Intel Arc Pro B580, TP=2)

Merged 2026-10-08 from `PLAN.md` (original upstream-port plan),
`REMAINING_TODOS.md` (CLI-era TODOs, now deleted) and
`optimizations-survey.md` (llama.cpp SYCL / vllm-xpu-kernels /
sgl-kernel-xpu survey). The three sources were one file each; this is
the single plan file. Section 1 is the backlog, sorted biggest
performance gain first; sections 2-5 are records, kept verbatim from
their sources because a measurement that disagrees with today's tree
is still a true record of when it was written.

**Objective (standing):** 27B `--tp 2` decode 40 tok/s, prefill
4000 tok/s. Measured state (derive, never quote from prose):
`scripts/model/model_byte_roofline.sh` — 9.752 GB/device/token against
456.0 GB/s peak; `kernel 237.5 ms/tok -> 41.1 GB/s = 9.0%`.

---

## 1. Backlog, biggest performance gain first

Order is by expected gain on the measured deficit. **The 5.6-6.4x
"clone-vs-shipped" deficit is RETRACTED 2026-10-08 and does not exist**
(see HISTORY.md, "THE 6.39x ITSELF IS NOT REPRODUCIBLE"): a fresh run of
`examples/roofline_probe` returns VACUOUS, and a saved bisect4 log from
10-06 measured the same clone algorithm at 55.1 GB/s while the shipped
arm stayed at ~40-49 GB/s in both. The clone number moved 4.7x between
two saved logs of the same kernel; the shipped number did not. The
honest ratio is **1.12x**, and it is a *cosmetic* win (`bit_cast<half>`
for the group scale, `av * wtab[code]` for the tail) that
`INTFER_DECODE1_VERBHALF` already lands in the engine — not a 6x.

What remains is the reproducible finding: the shipped INT2 decode runs at
~49 GB/s on MLP_GATE against a 456 GB/s ceiling, i.e. ~10.7% of peak.
That is the real deficit, and every row below is judged against it on
load-insensitive counters (`kernel` bucket), clearing the spread floor.

### 1.1 `block_load<uint8_t,128>` on the INT2 weight stream — **CLOSED 2026-10-12, not implemented, and the "prior" it cites does not exist**

* **The prior is a misattribution, and it is the whole reason this row
  sat untouched.** PLAN.md used to read "the wider-chunk probe already
  measured `block_load<uint8_t,256>` at 0.25-0.35x". There is no
  `block_load` arm in `examples/roofline_probe/main.cpp`. The 0.25-0.35x
  is the probe's **ceiling twin ratio** — `main.cpp:710-713`, "ratio to
  full = 0.25 (linear would be 0.25)", i.e. *a quarter of the bytes
  taking a quarter of the time*, the always-red validator that proves
  the GB/s figures are tracking traffic. It says nothing about load
  forming. A fabricated prior is worse than no prior, so it is struck.
* **The probe is vacuous on this box anyway** (AGENTS.md "The
  profiling defect"): every arm returns 0.000 ms and the gate refuses
  it. So even a real `block_load` arm could not have produced a number
  here; the `kernel` bucket is unverified for the engine, and the
  verdict would rest on `nk`/`nw`/`nr` alone, which a load-forming
  change does not move.
* **It is not implementable as specified, and that is the geometry, not
  a judgement call.** The shipped INT2-tab body (`quant_gemm.cpp:1055-1062`,
  the `use_verbhalf` branch, `CC = 8`) reads **two** weight bytes per 8-code chunk — `code_u8[b0]`
  and `code_u8[b0+1]`, 16 bits over 8 codes. `block_load<uint8_t,128>`
  is 128 B across the subgroup, i.e. **64 codes** at 2 bits each. There
  is no chunk size in the shipped kernel that a 128-B gather would
  service; the 256-B variant is worse still. The row's own comment
  (`quant_gemm.cpp:700-714`) already says the arm is "SUBGROUP's body
  with the two `code_u8[b0]`/`[b0+1]` reads replaced by one
  `block_load<uint8_t,128>` per 32 codes" — 32 codes is 64 B, still not
  128, and the replacement is not written anywhere. Writing it means
  redesigning the chunk loop, which is a new kernel, not a one-change
  arm, and it would need its own bit-exactness argument (the gather
  changes which bytes land in which lane).
* **Disposition: CLOSED, not attempted.** The only honest route to a
  `block_load` number on this box is a working device-event profiling
  path (PLAN.md section 5), which is a measurement fix, not an
  optimization, and is itself blocked on the driver. The code comment
  at `quant_gemm.cpp:700-714` is the record; it is not a TODO.
* **Gate:** none, and none is warranted — a flag with no implementation
  has nothing to certify. (The row's old "arm in `test_tp2_27b.sh`"
  was aspirational and is removed.)

### 1.2 gate+up in a single launch (the one TP=2-valid fusion)

* **Why:** fusion is the standing directive; both `gate` and `up` are
  column-parallel over the same split and device-local, and each
  output's K accumulation is untouched, so it is bit-exact. The two
  existing fusion seams (`run_mlp_fused` `engine.cpp:1858`,
  `run_linear_tp_devact` `engine.cpp:2506`) are **TP=1 ONLY** and
  are dead code at the target — do not route through them.
* **Gain:** removes one submit + one wait per MLP per token. The
  fused-MLP-down precedent is +13% decode for exactly that.
* **Where:** `src/engine/engine.cpp`, the TP=2 MLP path.
* **Gate:** stream-identity at TP=2, arm in `test_tp2_27b.sh`. **GATED,**
  not REPORTED: the engine prints `[mlp-gateup] fused gate+up GEMM ACTIVE`
  once on acceptance, and the arm asserts that line is present AND the
  streams agree. The first version of this arm was REPORTED only, because
  the engine printed nothing on success and the two streams alone cannot
  separate a real fusion from the wave fallback they replace (bit-identical
  by construction). See `scripts/gates/test_tp2_27b.sh` for the arm's own
  self-check, which exercises the ACTIVE-line predicate in both directions.

**AUDIT 2026-10-08 — two defects found and fixed in the 1.2 implementation:**

1. **[SUPERSEDED 2026-10-09 by `783e56f`, record only — do not act on.**
   M>1 prefill arms would read zeroed wave slots as mg/mu. The fused kernel
   had no tiled M>1 body; it returned a NULL event for M>1. The engine
   declined explicitly (`if (M > 1) return false;`) instead of letting the
   caller memcpy out of an unwritten buffer, and the M>1 prefill arms took
   `run_linear_wave_batch`, which is M-row aware. Without the guard the
   prefill arms were a silent correctness bug, not a performance miss.
   **The decline was the wrong fix.** `783e56f` found the kernel's M>1
   indexing was itself broken (it read the wrong weight row and the wrong
   activation row), so the guard was concealing a correctness bug rather
   than preventing one. The M>1 path is fixed in the kernel and the guard
   is removed: the SAME launch now covers prefill and decode, so the
   fusion is live on the prefill hot path and not just at M=1. The
   engine.hpp and HISTORY.md prose that described the old decline were
   corrected 2026-10-10; this note keeps the decline-era record.

2. **`nk` was short by one per device on every fused token.** The gateup
   kernel is not a `quant_gemm_forward_usm` call, so it never hit
   `profile_note_launch` — the same blind spot SSM/norm kernels had. It now
   calls `profile_note_launch()` once per launch and records the device
   event timestamps, so the load-insensitive launch counter is honest.
   (This one was a genuine accounting fix and stands.)

### 1.3 SLM work-stealing tile scheduler for prefill tiles

* **Status 2026-10-11: IMPLEMENTED and GATED (commit after this line).**
  `INTFER_TILE_SCHED=1` (default OFF) turns the tiled DPAS kernel's
  fixed `block % nWG` assignment into a per-device SLM work-stealing
  claim queue: a WI that drains its block pulls the next unclaimed one
  instead of idling. The claim is `atomic_update<atomic_op::inc>` on a
  `local_accessor<uint32_t>` per block (the ESIMD primitive; `sycl::atomic_ref`
  is not permitted in a `SYCL_ESIMD_KERNEL`). It is a pure DISPATCH
  change — each tile computes the same dot products in the same order —
  so the tiled kernel output is bit-identical to the fixed-assignment
  path and no numerics contract is needed. Per-device, so TP-compatible.
  The arithmetic is ONE shared device body (`process_block`, carrying the
  `sycl_explicit_simd` attribute) called by both paths, so they cannot
  drift. Announced `[tile-sched] INTFER_TILE_SCHED=on|off n_blocks=.. nWG=..`
  once per process; gate arm in `test_tp2_27b.sh` with a 3-part self-check
  (healthy / mode rewritten to `off` / empty log).
* **Why:** vllm/sgl `TileScheduler` (one `atomicAdd` per tile-chunk)
  keeps the device busy under saturation; submit inflates ~84 us/launch
  at M=1 to ~600 us at M=128, so fewer, better-filled launches are
  the prefill lever. Per-device, so TP-compatible.
* **Does NOT apply to M=1 decode:** one output per work-item, nothing
  to steal. Do not attempt it there.
* **Where:** `quant_gemm_dpas48_tiled<R>` (`src/gemm/quant_gemm.cpp`),
  reached via `run_linear_tp_dpas48` (`src/engine/engine.cpp:1248`) —
  i.e. only when `INTFER_DPAS48=1`. On the default prefill path
  (`sc.dpas48 == false`) the kernel is `quant_gemm_forward_usm`'s
  64x64 tiled body, which this flag does NOT touch.
* **Kill criterion:** if tile occupancy is already saturated at the
  current launch shape (probe with the device's own event timestamps),
  close it. The measured verdict is OPERATOR-ONLY: this box's
  `kernel` bucket is unverified for the engine (see below), so the
  decision rests on `nk`/`nw`/`nr` and the host buckets, never on a
  tok/s.

### 1.4 oneMKL `gemm` fallback for large contiguous FP16

* **Status 2026-10-11: IMPLEMENTED, NOT GATED — and the "GATED" claim that
  used to sit here was false.** `INTFER_ONEMKL=1` (default OFF) routes
  `quant_gemm_f16` through `gemm::gemm_onemkl_fp16` when
  `onemkl_available() && M >= 8`. The library is linked at build time
  (`INTFER_ENABLE_MKL=ON`); when it is absent `gemm_onemkl.cpp` compiles
  the empty-fallback spellings of `onemkl_available()` /
  `gemm_onemkl_fp16()` so the engine links either way and the arm is
  unreachable rather than wrong. Announced
  `[onemkl] FP16 gemm via oneMKL SYCL BLAS ACTIVE (M=.. N=.. K=..)`,
  once per process. **There is no gate arm for this flag anywhere in
  `scripts/gates/test_*.sh`** (`git grep INTFER_ONEMKL -- scripts/gates/test_*.sh`
  returns nothing), so `check_flag_gate_coverage.sh` files it as the
  sole remaining opt-in REAL GAP and its own disposition block
  (`check_flag_gate_coverage.sh` header, the authoritative flag-by-flag
  record) does not list it among the armed flags. The row's old "Gate
  arm in `test_tp2_27b.sh`" was aspirational prose, not a measurement;
  it is struck here and in AGENTS.md:233.
  **[SUPERSEDED 2026-10-21 by re-measurement: every claim in the paragraph above is
  now false, and it is annotated rather than rewritten because rewriting a true
  historical record to match today is the other direction of the same laundering
  failure.]** (a) `git grep INTFER_ONEMKL -- 'scripts/gates/test_*.sh'` returns
  **8 hits in `scripts/gates/test_onemkl.sh`**, not nothing — that gate names the
  flag literally, and self-checks that it does (`if ! grep -qE '\bINTFER_ONEMKL\b' "$0"`
  -> `FAIL: this gate does not name INTFER_ONEMKL -- the flag census would file it as
  an ungated opt-in gap`), so a gate that loses the name cannot pass its own
  preflight. (b) `check_flag_gate_coverage.sh` returns rc=0 reporting
  `the opt-in residue is EMPTY: every opt-in flag is now named by a gate` and
  `FLAG/GATE COVERAGE: PASS (106 engine flags, 94 gate-named, 37 ungated: 0 of them
  opt-in = the real gap)`, so this row has not been "the sole remaining opt-in REAL
  GAP" since 2026-10-13. (c) The `AGENTS.md:233` citation names the wrong subject:
  line 233 is inside the `[xmx-attrib]` paragraph (the `matrix_engines=<verdict>`
  word-not-a-zero contract), with nothing to do with this row; the authoritative
  onemkl record in AGENTS.md is `INTFER_ONEMKL — DEFECT, FIXED` and the
  `test_onemkl.sh could never test the oneMKL arm` entry. The gate itself is
  build-conditional and SKIPs (exit 2) when `libmkl_sycl_blas` is absent, so on a
  stock box it is red-by-precondition-free rather than falsely red.
  * **Why no gate can reach it on this box, and it is two independent
    blockers, not one.** (a) BUILD CONFIG: the real oneMKL body lives
    under `#ifdef INTFER_ENABLE_MKL` in `gemm_onemkl.cpp`; that is a
    CMake option, OFF by default (`CMakeLists.txt:23`), and NO gate
    script passes `-DINTFER_ENABLE_MKL=ON` (`src/gemm/CMakeLists.txt:39-54`
    only turns the #ifdef on and links `libmkl_sycl_blas` when the
    option is ON)
    **[SUPERSEDED 2026-10-21 by measurement, not by reading: exactly one gate
    script passes it — `scripts/gates/test_onemkl.sh:78`, in its own throwaway
    build dir (`GATE_BUILD_DIR="${INTFER_ONEMKL_BUILD_DIR:-$BUILD_DIR/onemkl_gate}"`,
    configured with `-DINTFER_ENABLE_MKL=ON -DCMAKE_PREFIX_PATH=.../mkl/latest`)
    precisely so the operator's default build stays MKL-free. The accurate claim is
    "no gate passes it into the default build", and the neighbouring claim about the
    arm being unreachable by gate shape is separately closed by
    `examples/onemkl_probe`, which runs `quant_gemm_f16` directly on synthetic
    M>=8 tensors and needs no 27B load.]**
    option is ON). With the default build `onemkl_available()` returns
    false and `quant_gemm_f16` never takes the arm -- unreachable by
    build configuration, not by flag. (b) GATE SHAPES: even in an MKL
    build the predicate is
    `onemkl_available() && getenv(INTFER_ONEMKL) && M >= 8`
    (`quant_gemm.cpp:3073-3075`). Every gate on this box either runs no
    FP16 GEMM at all, or runs one at M=1 (decode, and the lm_head at
    `main.cpp:1374/2147/2422`, all M=1 or M<8). The only M>=8 caller is
    the 27B prefill path, which no gate runs (standing rule 1). So the
    arm cannot be reached by any gate that is permitted to run here.
    **[CITATION DEFECT, 2026-10-21: those three lines are lm_head buffer
    SIZING and guard lines, not FP16 GEMM call sites, so they cannot support "all
    M=1 or M<8" for any caller.]** Measured against `examples/intfer_cli/main.cpp`:
    1374 is `lm_head_*M_dev_`/`lm_head_cM_cap_` sizing, 2147 sits inside the
    `INTFER_LMHEAD_DEVARGMAX_TP2` comment block, and 2422 is `lm_head_xM_dev_`
    assignment. The actual `quant_gemm_f16` sites are **1451, 1948, 2248, 2535**,
    and the M-row ones carry a *variable* M — line 1451 passes
    `quant_gemm_f16(q, x_row, ..._w_dev_[d][t], lm_head_cM_dev_[d][t], M, Nlt, K)`
    with M the prefill chunk width, while the scalar decode sites pass M=1. So
    "every gate runs an FP16 GEMM only at M=1 or M<8" is NOT established by the
    lines cited, and the M-row premise needs a run to settle rather than a
    citation. It does not change the row's verdict — reachability is governed by
    the build config above (superseded: `test_onemkl.sh` now configures its own
    throwaway MKL build), and by `examples/onemkl_probe`, which reaches the branch
    on synthetic M>=8 tensors and so closes the gate-shape half independently of
    any engine M. Both halves are re-derived here rather than assumed.
  * **The arm itself is sound and the default path is untouched.** The
    branch is in `quant_gemm_f16`, called by every FP16 GEMM, and the
    flag is explicit (`getenv != nullptr` at `quant_gemm.cpp:3074`),
    so enabling the build does NOT change the default path -- the default
    build is byte-for-byte the MKL-free tiled kernel
    (`gemm_onemkl.cpp:115-124`, the `#else !INTFER_ENABLE_MKL` empty
    fallbacks). The blocker is the build config, exactly
    the shape `INTFER_ENABLE_L0` already solves in `scripts/build.sh:28`.
  * **Arming it is operator-only, not agent-only.** It needs a build +
    a gate run, which is measurement work and is retired on this box
    (PARAMOUNT DIRECTIVE). The library exists
    (`/opt/intel/oneapi/mkl/latest/lib/libmkl_sycl_blas.so.6`), so a
    throwaway `-DINTFER_ENABLE_MKL=ON` build dir (the
    `mutate_rowpair_equiv.sh` pattern: `cmake -S .. -B $dir -D...`) makes
    the arm real; then a tolerance-gated prefill comparison certifies
    it. Until that run exists this row is a genuine opt-in gap whose
    remedy is operator-only.
* **Why:** llama.cpp `ggml-sycl.cpp:3115` shape. **[EXTERNAL CITATION 2026-10-21: `ggml-sycl.cpp` is not in this tree (`ls ggml-sycl.cpp` -> No such file or directory), so this line cannot be resolved locally and is not a stale local reference. It names llama.cpp's `ggml/src/ggml-sycl/gemm.hpp` upstream; re-derive it against a fetched clone, never against this repo.]**
* **Why it is fourth:** FP16 is 57.9% of decode traffic but already
  runs at 98-123% of the memory roofline (LA_QKV 559.9 GB/s, LA_OUT
  477.1 GB/s) — at the wall. This cannot pay on decode; it is a
  prefill/large-M item only.
* **Contract: TOLERANCE-GATED, not bit-exact.** oneMKL's FP32
  accumulation order is an internal implementation detail of the
  `mkl::blas` kernel and is NOT the engine's row-major ascending-k
  order, so no bit-exactness claim is made and none is warranted.
  Any gate compares the token stream against base under an AGREED
  tolerance. The `[onemkl]` announcement is what makes a run's having
  reached the arm observable — without it a flag that never reached
  the kernel would pass a tolerance-gated comparison just as easily.

### 1.5 Barrier-free subgroup reduction (SUBGROUP retry)

* **Why:** the SHAPE is still open even though the first attempt was
  wrong — xor butterfly, no `local_accessor`, no `it.barrier`, which
  is the specific cause SPLITK named (+43.8% kernel).
* **Blocker:** `INTFER_DECODE1_SUBGROUP` was numerically wrong
  (ids '279 12 12' vs base '13 198 760') and the cause is NOT
  located (HISTORY.md, "QUARANTINED, not fixed"). Do not start on a
  structural argument — the reverted fused-GEMM work proved reading
  two identical bodies proves nothing.
* **Precondition:** the numerics contract (stream-identity on 27B
  TP=2) agreed before code, and a gate that reaches the branch.

### 1.6 Multi-split decode attention tiles

* **Why:** vllm/sgl paged/FMHA decode multi-split.
* **Why last:** layer GEMMs are 99% of launches. The attention tail
  is not the deficit. Only after 1.1-1.3 are settled.

### 1.7 Bit-exact weight-load forming in the M=1 INT2 decode
  (IMPLEMENTED 2026-10-10 as `INTFER_DECODE1_WVEC`, default
  OFF — the candidate below is now code, gated and model-free
  measured; only the performance verdict is still open)

* **The finding (static).** The default-ON INT2-tab branch of
  `gemm_usm_decode1_vec` loads the weight through **scalar**
  `code_u8[b0]` / `code_u8[b0+1]` byte reads, while the sibling
  INT8 path loads its weight through a vectorized
  `sycl::vec<std::uint8_t,8>::load` via a `w_mp` multi_ptr. The
  multi_ptr is constructed for the INT2 path too but is **never
  used** in it — the path reads the raw `code_u8` pointer
  instead. So the two paths use different load idioms for the
  same kind of operand, and the INT2 path (the shipped
  +5.7-6.0% default) takes the scalar one.
* **The candidate (bit-exact by construction).** Load the two
  INT2 weight bytes through the multi_ptr as a
  `sycl::vec<std::uint8_t,2>`, matching the INT8 idiom.
  Identical bytes read, identical codes extracted, identical
  accumulation order -> byte-identical output.
* **Status 2026-10-10: shipped behind the flag.**
  `INTFER_DECODE1_WVEC=1` runs `gemm_usm_decode1_wvec` — the
  base INT2-tab body with the chunk's two weight bytes read as
  one `sycl::vec<std::uint8_t,2>::load` from a row-base
  multi_ptr (the VERBMP idiom: `b0` is RELATIVE to that base
  and the tail keeps the UNOFFSET absolute `code_u8` indices, so
  the ROWPAIR double-offset class has no room to exist).
  Reaching it needs BOTH default-ON arms killed —
  `INTFER_NO_DECODE1_VERBHALF=1 INTFER_NO_DECODE1_VERBMP=1`
  (verbhalf is checked first, verbmp second, and each ends in
  `return q.submit(...)`; killing only verbhalf leaves verbmp
  winning the run, proven model-free at the M=1 INT2
  MLP_GATE shape) — plus the presence flag
  `INTFER_DECODE1_WVEC=1`. That is exactly what the
  `test_tp2_27b.sh` WVEC arm sets, and the rowpair_equiv
  `wv` order reaches it the same way (its env kills every
  non-selected arm, both defaults included). **Bit-exactness
  is MEASURED, not argued:** the `wv` order diffs it against
  the restored reference, INT2 shapes only, bitwise,
  model-free.
* **What is still open (operator-only):** the performance
  direction. A bit-exact load-forming change moves **only** the
  `kernel` bucket (the sole device-event-timestamp bucket);
  every other counter — `nk`/`nw`/`nr` and the host
  `submit`/`waitblock`/`copy`/`readback`/`usm_write` buckets —
  is provably unchanged: same launch count, same wait, same
  transfer sizes. The verdict can come **only** from `kernel_us`
  (3/3 same sign, clearing the spread floor), whose liveness on
  this box is **UNVERIFIED for the engine** (verified dead only
  for `roofline_probe`; AGENTS.md "The profiling defect"). The
  27B engine arm in `test_tp2_27b.sh` asserts the stream
  identity on real weights; the operator measures the delta.

### 1.8 Every remaining UNMEASURED item is DECLINED-WITH-REASON for an agent session, and the reason is measured

**Status: DISPOSITIONED 2026-10-21. There is no un-declined item left in this backlog.** The
shipped-but-unmeasured flags above (1.2, 1.3, 1.4, 1.7 and their siblings) each read
`UNMEASURED — operator-only A/B`, and that phrase is not a deferral: each is **declined from an
agent session for a measured reason**, carrying the exact command an operator runs. The full
runbook is HISTORY.md "the operator-only backlog, DISPOSITIONED WITH EXACT COMMANDS" — this
row is the plan-side pointer and the backlog consequence, not a second copy.

**The four reasons are measured, not recited.** (a) Loading the 27B is forbidden by the standing
rule: the shard USM is tens of GiB and a raw load is tens of minutes, which is not a box probe a
session may spend. (b) The wall-time benchmark queue is **RETIRED**: a tok/s needs `load1 < 2`
for the whole of every arm and this box has not held that window. (c) The device-event profiling
defect means `kernel` cannot be a decision bucket here, so the load-insensitive integers
`nk`/`nw`/`nr` are the decision surface and only an operator run produces them. (d)
`scripts/knob_sweep.sh` already refuses a contended round `SKIPPED-CONTENDED` and excludes it
from both usable filters, so it is the operator's tool **by construction** — a session that ran
it would either spend the load or withhold its own figures.

**The two decisions that are NOT operator-only and therefore still open here.** Promote only
what clears the spread floor on ≥2 paired rounds (a single-round delta has no floor to clear),
and never quote a tok/s that was not measured on a quiet box. Both are standing rules, not
backlog work.

**DECLINED as well, with the reason not being "not yet attempted".** `INTFER_MLA_ATTN` — no MLA
model exists on this box, so the arm cannot be run at all (the capability gap, that it realises
none of MLA's KV-cache memory win, is the verdict in `engines_sources.txt` line 50);
`INTFER_KV_KVARN` — not implemented, because it needs an agreed tolerance contract before any
code and the KV cache is correctness-gated with no tok/s claim.

## 2. Not a backlog item (measured, closed)

Every row here has a measurement in HISTORY.md or the dead-ends table
in AGENTS.md. Do not re-attempt as-is.

* `INTFER_DECODE1_MR` — 2.3x slower at M=1.
* `INTFER_DECODE1_SPLITK` — +43.8% kernel, stream-identical.
* `INTFER_DECODE1_SUBGROUP` — numerically wrong, quarantined.
* Fused multi-GEMM via extracted device function — 231 114 ms;
  keep arithmetic INLINE at each call site.
* 16-code chunks — 0.25-0.35x.
* `INTFER_DPAS48` enforced for all linears — net loss at every
  tested shape; default OFF is the verdict.
* SYCL command graphs for decode — 19.7 us vs 11.9 us plain submit.
* Native L0 immediate submit, host-private DPAS staging, staged
  readback, `q.prefetch()` over all shards — same floor / neutral /
  no-op.
* XMX at M=1 — the shapes are memory-bound; do not re-attempt.
* GQA-packing the head-group into the M/row dimension at M=1
  (vllm-xpu `decode_policy_qpacked_head` / `qgroup` 8/16) —
  **DECLINED WITH EVIDENCE, 2026-10-10** (HISTORY.md, "GQA-packing
  DECLINED with evidence"). `INTFER_DECODE1_GQAPACK` is NOT in the
  tree and will not be added. Dispatch trace: the only device
  attention kernels in the tree are `flash_attention_forward_f16`
  and `flash_attention_forward_int8_kv` (`src/attention/flash_attention.cpp`),
  and neither is reachable from the engine or the inference path —
  their only callers are `examples/flash_attention_demo/` and
  `examples/qwen_forward_gpu/` (both non-inference), the rest are
  comments. `src/engine/engine.cpp` includes the header and makes
  zero attention calls. The decode attention that runs is the
  GatedDeltaNet SSM recurrence (no KV cache, no GQA group); the
  `is_full` softmax decode is host-only scalar; the M=1 INT2 decode
  is the layer GEMM. A device paged-KV softmax-decode kernel would
  need to be written first — that is where `qgroup` packing would
  apply.

## 3. Dependencies required by the big items

| Item | Dependency | Status |
|---|---|---|
| oneDNN matmul (llama's actual XMX path) | link oneDNN + a tolerance-based numerics contract agreed before code | NOT in tree; the repo has a measured matmul-precision divergence history, so the contract comes first |
| CUTLASS `TiledMMA` / `block_2d_copy` / `xe_split_barrier` | import tens of kLOC of tiled-GEMM machinery | NOT in tree; not cherry-pickable as one kernel |
| FP8 / MXFP8 / FP4 / MXFP4 | quant + RCO + kernels + contract | no consumer: `quant_spec.cpp` tops out at INT8/FP16 and no gate model carries FP8/FP4 tensors |
| Xe2/Xe3 dispatch | none needed | single-arch target (2x B580, Xe2); the Xe3 half has no device here |

## 4. What the three repos do (reference inventory)

**llama.cpp SYCL** (`ggml/src/ggml-sycl/`) — leans on oneDNN/oneMKL
for matmul; hand-written subgroup tile kernels for the quant path.
`DnnlGemmWrapper::row_gemm` probes `dnnl::matmul` and rejects the
`"ref"` impl (`ggml-sycl.cpp:123` **[external — see the EXTERNAL CITATION note under "Why:"; upstream llama.cpp, absent from this tree]**); `oneapi::mkl::blas::column_major::gemm`
fallback (`:3115`); quantized matvec in `dmmv.cpp`/`mmq.cpp` with
`reqd_sub_group_size(WARP_SIZE)`, dequant-on-the-fly and
`permute_sub_group_by_xor` warp reduction — **no DPAS/XMX**, EU vector
ALUs only. Its XMX hand-written path is `#if 0`'d (`:177`). Bottom
line: llama.cpp's matmul throughput is oneDNN's internal XMX dispatch.

**vllm-xpu-kernels** (`csrc/xpu/`) — CUTLASS-first with per-arch trees.
`XE_DPAS_TT`/`XE_BDPAS_TT`, `MainloopIntelXeXMX16*` dispatch policies,
`onednn/onednn_matmul.cpp` (w4a4_fp4, w8a16_int4, w8a8_fp8),
`kernel/xe_tile_scheduler.hpp` (persistent + non-persistent,
work-stealing), `make_block_2d_prefetch` + `barrier_arrive/wait`
K-pipeline, `cute::xe_split_barrier`, sub-group 16, paged FMHA
decode, FP8/MXFP8 quantization.

**sgl-kernel-xpu** (`src/sycl/`) — same CUTLASS base, no oneDNN,
broader kernel surface. `XE_8x16x16_F32F16F16F32_TT` XMX MAC,
separate `xe20/` vs `xe35/` trees selected at
`torch_extension_sycl.cc:256`, SLM-backed atomic counter tile
scheduling, block-2D prefetch + split barriers, and flash-attention
v2 / MLA (dense + sparse 2-stage) / GDN / HiSparse / MoE / LoRA /
MQA-logits / HC-prefill.

**Comparison matrix** (external feature -> present where):

| Feature | llama.cpp | vllm-xpu | sgl-xpu |
|---|---|---|---|
| DPAS/XMX hand-written | detect-only, unused | CUTLASS `XE_DPAS_TT`/`BDPAS` | CUTLASS `XE_DPAS_TT`/`BDPAS` |
| XMX-backed matmul | oneDNN probe | CUTLASS `MainloopIntelXeXMX16*` | same + `XE_8x16x16_*` |
| oneDNN | gemm, attention | w4a4/w8a8/fp4/fp8 | none |
| oneMKL BLAS | column_major::gemm | via oneDNN | via oneDNN |
| Sub-group width | 16 | 16 | 16 (some 32) |
| SLM work-stealing scheduler | no | TileScheduler | SLM counter |
| K-pipeline prefetch+barrier | no | yes | yes |
| Split barriers | no | yes | yes |
| Async USM | `ext_oneapi_async_memory_alloc` | no | no |
| MoE / FP4-MXFP4 | no / no | yes / no | yes / yes |

## 5. Prerequisite measurement (blocking 1.1's verdict)

The `roofline_probe` profiling path currently returns `0.000 ms` for
every arm on this box (`build/roofline_probe.log`: `FAIL VACUOUS`).
The gate correctly refuses it (`mutate_roofline_probe.sh` PASS 3/3),
so no new INT2 number can be produced until the device-event
profiling path is fixed. This is a measurement fix, not an
optimization, and it is the reason the `INTFER_DECODE1_VERBHALF`
default-ON promotion is a correctness bet with an unknown payoff.
