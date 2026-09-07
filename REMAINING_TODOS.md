# intfer_cli — Remaining Performance TODOs

The two main wins are applied (single-GPU mode + M=1 decode kernel). Below are
the remaining optimization ideas to push GPU throughput past CPU. Each item
has a "why" and a concrete next step.

---

## TODO 1 — Vectorized FP16 loads in the M=1 decode kernel

**Status:** DONE in `a173efe` (went beyond the sketch: group-outer loop
hoisting the per-group scale, uchar8 INT8 loads, shift/mask INT4/INT2
extracts). Probe M=1: INT8 334→100µs, INT4 487→65µs, N=248320
5.4ms→0.44ms. 0.8B decode 2.73→3.05 tok/s, GPU stream bit-identical.

**Why:** The current `gemm_usm_decode1` kernel reads `act_usm[kg]` and
`code_u8[idx]` one element at a time (scalar loads). On Xe-core this means
one transaction per element. With `sycl::vec` or `sycl::half` aligned loads
we can issue 8-wide vector loads and halve the issue pressure on the LSU.

**Where:** `src/gemm/quant_gemm.cpp` — `quant_gemm_forward_usm` M=1 branch
(class `gemm_usm_decode1`).

**Next step:**
- Read `act_usm[kg..kg+8]` as a `sycl::vec<sycl::half, 8>` per K-step.
- For INT8 weights, read `code_u8[idx..idx+8]` as a `sycl::vec<int8_t, 8>`
  and do 8-way FMA via `sycl::vec<float, 8>` accumulator.
- Keep the dequant codebook lookup scalar (it's already cheap and the
  compiler can hoist it).
- For INT4/INT2, do 8× packed-byte loads and unpack on the fly.

**Expected gain:** 1.5–2× on the kernel itself; multiplied across the 7
linears × 24 layers of the decode loop this should pull GPU clearly past
CPU.

---

## TODO 2 — Persistent kernel across the 7 Linears × 24 layers

**Status:** MEASURED DEAD — do not pursue. Probe 2026-09-07
(`/var/home/fra/probe_graph.cpp`): executable-graph launch = 19.7µs vs
plain `q.submit` = 11.9µs on B580, i.e. graphs are *slower*. The ~250µs
"submit" cost was misattributed launch overhead; probes (`probe_gemm.cpp`)
proved it is the dequant kernel body itself (fixed by TODO 1 → a173efe).
Remaining per-submit host cost (~150-250µs in-engine) exceeds kernel time,
but graphs cannot reduce it. If pursued at all, the lever is launch-COUNT
reduction (fusion), not launch-cost reduction. Scaffold
(`INTFER_GRAPH_CAPTURE` + `build_decode_graph`) left untouched.

**Why (original, superseded):** Every `run_linear_tp` call currently goes through:
  host pre-convert FP32→FP16 → direct USM write → `q.submit()` → kernel
launch → `event.wait()` → host readback. For 7 linears × 24 layers × ~20
decode steps = thousands of kernel launches per generated token. The launch
overhead alone is non-trivial.

**Where:** `src/engine/engine.cpp` `run_linear_tp` + `examples/intfer_cli/main.cpp`
forward step.

**Next step:**
- Pre-allocate the per-step activation USM buffer once and let it stay
  resident (already done).
- Replace per-call `q.submit()` with the SYCL command-graph capture
  (`sycl::ext::oneapi::experimental::command_graph`) — record the full
  forward pass once, then replay it each step.
- Backed by the `INTFER_GRAPH_CAPTURE` machinery already in `engine.cpp`
  (see `build_decode_graph` / `DeviceUSM` struct).
- The graph is currently restricted to a single tensor; extend it to cover
  all 7 Linears per layer × 24 layers for the full decode step.

**Expected gain:** 2–3× launch-overhead reduction → 10–30% wall-time on
decode.

---

## TODO 3 — Multi-token decode (M=4 or M=8) instead of M=1

**Status:** Partially implemented. Engine now exposes `run_linear_tp(..., M)` and `decode_step_batch` with batched GEMM execution. CLI `--decode-batch` flag is wired and `step_batch()` wrapper exists, but `step()` itself remains scalar; true M>1 forward pass requires batching RMSNorm/RoPE/SSM layers.

**Why:** Decode of a single token is memory-bound. With 7 Linears per
layer × 24 layers we already do ~7×24=168 GEMMs per token. Each GEMM
launches a kernel of size O(N) work-items. By batching 4–8 tokens we can
keep the GPU compute units busy and amortize the launch overhead across
the batch.

**Where:** `examples/intfer_cli/main.cpp` `step()` and the forward loop.

**Next step:**
- Restructure `step()` to accept M hidden states and vectorize RMSNorm/RoPE/SSM across M.
- Use Engine batched `run_linear_tp` per layer for the M tokens.
- Verify the M=1 fast path still beats the tiled path for M=1 (it should
  — the tiled path was the source of the original slowdown).
- For speculative decoding: this is a natural fit for the MTP-style
  approach described in AGENTS.md Stage 2.

**Expected gain:** 2–4× when amortized with speculative verification.

---

## TODO 4 — Host hot path (updated 2026-09-07: detok + 4a + FWHT done, memcpy dead)

**Status:**
- DONE `711000d`: in-process BPE detokenizer (killed 158ms/token python
  spawn). CPU 3.05→5.19 tok/s, GPU 3.3→4.01 tok/s. Byte-identical stdout.
- DONE `73494a2`: 4a scratch arena (`StepScratch` + `run_linear_into` +
  engine `act_staging_`). Mallocs 1540→~420/token, churn 4.8→1.1MB.
  Wall-time NEUTRAL — the loop is compute-bound, not allocation-bound
  (correction to the "allocation-bound" scoping below).
- DONE (this commit): 4c FWHT online rotations. Probe (`INTFER_TIME_SSM`,
  since removed) showed SSM = 2.7ms/tok (1.3%, vectorizes fine — DEAD end)
  but SpinQuant replays = ~108ms/tok (~50% of GPU decode): the CLI ran
  dense O(n²) matvecs (1.49ms each, strided columns) for Hadamard rotations.
  Fix: `apply_rotation_fwht` + `rotation_is_hadamard` in spinquant lib,
  flag re-detected after deserialize (serializer doesn't persist it),
  wired in CLI `rot_apply`/`rot_left` + `Engine::load`. Rot cost
  860→1.3ms/run. CPU 4.21→8.03 tok/s, GPU 4.06→8.55 tok/s, both parities
  intact (`854×12`, `16 13 198 760 4087`). Handles non-pow2 dims via
  blockdiag(H_p, I) (needed for 27B H=5120).
- VALIDATED on 27B post-commit (no code change): `.re` GPU `-n 1` run gives
  prefill→11751 + decode→13, matching the HF oracle sequence
  `11751 13 198 760`. Prefill 414→331s. The blockdiag FWHT path (p=4096 +
  identity tail, all 4 rotations dim=5120 detected hadamard=1) is
  argmax-correct at 27B scale.

**Status:**
- DONE `711000d`: in-process BPE detokenizer (killed 158ms/token python
  spawn). CPU 3.05→5.19 tok/s, GPU 3.3→4.01 tok/s. Byte-identical stdout.
- DONE `73494a2`: 4a scratch arena (`StepScratch` + `run_linear_into` +
  engine `act_staging_`). Mallocs 1540→~420/token, churn 4.8→1.1MB.
  Wall-time NEUTRAL — the loop is compute-bound, not allocation-bound
  (correction to the "allocation-bound" scoping below).
- DEAD (reverted, uncommitted): memcpy H2D/D2H in `run_linear_tp`.
  1236→1379ms decode; per-memcpy enqueue (~164µs) costs more than the
  touch loop (~59µs) at 3KB/transfer. Lesson: at 128 SYCL calls/token,
  per-CALL overhead dominates; fewer calls is the only lever.
- GPU profile (0.8B, 128 GEMM/tok): submit 38ms + wait 10.6 + write 7.5 +
  readback 5.4 + kernel 9.7ms ≈ 72ms GPU-attributable; host elementwise
  ≈134ms/tok (65% of GPU decode). Submit overhead largely irreducible
  (graphs dead per TODO 2, fusion blocked by sequential deps).

**Remaining (priority order):**
- 4c (continued): host elementwise now ~45ms/tok on GPU path (was ~134ms).
  Next candidates: `name()` snprintf + map finds per linear (4b, hygiene),
  norm/copy fusion. GPU profile now: submit 36ms + wait 10 + write 5.7 +
  readback 5.2 + kernel 9.7 ≈ 67ms device-attributable vs 117ms decode —
  host (~50ms) and submit overhead (~36ms) are the remaining chunks.
- 4b: hoist `name()` snprintf + map finds — kills remaining ~420
  mallocs/token; hygiene, minimal wall-time (compute-bound).
- 4d (later): host attention tiling for long context.

**Why (original):** Only the 7 quantized Linears per layer are GPU-accelerated. The
RMSNorm, RoPE, attention scores, mamba SSM, lm_head matmul (248320×1024),
and the sampling loop all run on the host. The lm_head matmul alone is
~248M MAC per token, which is the largest single matmul in the model.

**Where:** `examples/intfer_cli/main.cpp` — `step()` function, the
`host_matmul` lambda, and the `logits` computation after the final norm.

**Next step:**
- Add a small "host-vector→device-USM" helper for the residual
  (already pattern in `run_linear_tp`).
- Add a non-quantized `gemm_f16` SYCL kernel for the lm_head (FP16×FP16
  matmul → XMX DPAS FP16 path; trivial with one `dpas` call per 8×16 tile).
- For the SSM/attention: this is more invasive but the per-token attention
  is currently `O(pos × N_HEADS × HEAD_DIM)` on host, which grows linearly
  with context length.

**Expected gain:** 1.5–2× on tokens/sec once the lm_head is off-CPU.

---

## TODO 5 — Profile with GPU timestamps to identify the real bottleneck

**Why:** We've optimized based on hypotheses (tiling waste, PCIe
contention). Without measurements we don't know if the remaining
bottleneck is:
  - Kernel compute time (improving with TODOs 1, 3)
  - Kernel launch overhead (improving with TODO 2)
  - Host-to-device activation upload (improving with persistent USM,
    already partly done)
  - lm_head matmul (improving with TODO 4)

**Where:** Anywhere; add to `examples/intfer_cli/main.cpp`.

**Next step:**
- Add `auto t0 = q.ext_oneapi_submit_barrier();` + event profiling around
  the kernel, then read the event timestamps with
  `e.get_profiling_info<sycl::info::event_profiling::command_start/end>()`.
- Make sure the queue was created with the `enable_profiling` property
  (currently `in_order` is set but profiling is not — add it in
  `device_manager.cpp:52`).
- Print per-stage timings: act upload, kernel, host readback.

**Expected gain:** Knowledge; lets us pick the right next step.

---

## TODO 6 — INT8 DPAS path (requires FP16→INT8 quant of activations)

**Why:** The current XMX DPAS path requires INT8×INT8 (see
`quant_gemm_dpas_int8` in `quant_gemm.cpp:376`). With FP16 activations we
never take that path — we go through the scalar dequant-on-the-fly kernel
which doesn't use XMX at all. The XMX systolic array gives 4× the
throughput of scalar FP16 MMA on Xe-core.

**Where:** `src/gemm/quant_gemm.cpp` `quant_gemm_forward_usm`.

**Next step:**
- Add a per-tensor INT8 activation quantizer: compute per-tensor
  `a_scale = max(|act|) / 127`, quantize activation to INT8, then call
  `quant_gemm_dpas_int8` with the existing INT8 weight (which we
  already store for INT8-scheme tensors).
- For INT4/INT2 weight tensors, dequant to INT8 first, then use DPAS.
- Symmetric scales only for now (per-tensor activation scale is fine for
  decode).

**Expected gain:** 3–4× on the GEMM kernel itself, dominating the decode
loop. This is the single biggest unlock available.

---

## Priority order

If you only have time for one, do **TODO 6** (INT8 DPAS) — the XMX
matrix engines are the entire point of the B580 for inference and we
currently don't touch them.

If that's too invasive, **TODO 1** (vectorized loads) is the lowest
hanging fruit and is a self-contained change inside the M=1 kernel.

**TODO 5** (profiling) is recommended first to validate which of the
above actually moves the needle.
