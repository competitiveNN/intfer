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
`scripts/model_byte_roofline.sh` — 9.752 GB/device/token against
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

### 1.1 `block_load<uint8_t,128>` on the INT2 weight stream

* **Why it is first:** INT2 is 20.2% of decode traffic
  (1.837 GiB/device/token). The old rationale ("the entire 5.6-6.4x
  deficit sits in the shared text") is void; this is now a plain
  load-forming experiment on a ~10% utilisation kernel, and the
  wider-chunk probe already measured `block_load<uint8_t,256>` at
  0.25-0.35x — a 3-4x regression from register pressure.
* **Where:** `src/gemm/quant_gemm.cpp`, M=1 INT2 decode lambda.
  (`src/gemm/quant_gemm.cpp:457-465`.)
* **Kill criterion (pre-committed):** < 2x closes the experiment —
  that also closes llama.cpp's `block_load` direction for INT2, which
  the wider-chunk probe already measured at 0.25-0.35x.
* **Gate:** arm in `test_tp2_27b.sh` (TP=2 is the only gate whose
  27B model carries INT2), stream-identity asserted.

### 1.2 gate+up in a single launch (the one TP=2-valid fusion)

* **Why:** fusion is the standing directive; both `gate` and `up` are
  column-parallel over the same split and device-local, and each
  output's K accumulation is untouched, so it is bit-exact. The two
  existing fusion seams (`run_mlp_fused` `engine.cpp:1352`,
  `run_linear_tp_devact` `engine.cpp:1484`) are **TP=1 ONLY** and
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
  by construction). See `scripts/test_tp2_27b.sh` for the arm's own
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
  reached via `run_linear_tp_dpas48` (`src/engine/engine.cpp:1163`) —
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

* **Status 2026-10-11: IMPLEMENTED and GATED (commit after this line).**
  `INTFER_ONEMKL=1` (default OFF) routes `quant_gemm_f16` through
  `gemm::gemm_onemkl_fp16` when `onemkl_available() && M >= 8`. The
  library is linked at build time (`INTFER_ENABLE_MKL=ON`); when it is
  absent `gemm_onemkl.cpp` compiles the empty-fallback spellings of
  `onemkl_available()` / `gemm_onemkl_fp16()` so the engine links either
  way and the arm is unreachable rather than wrong. Announced
  `[onemkl] FP16 gemm via oneMKL SYCL BLAS ACTIVE (M=.. N=.. K=..)`,
  once per process. Gate arm in `test_tp2_27b.sh`.
* **Why:** llama.cpp `ggml-sycl.cpp:3115` shape.
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
`"ref"` impl (`ggml-sycl.cpp:123`); `oneapi::mkl::blas::column_major::gemm`
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
