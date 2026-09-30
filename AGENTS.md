Port ninfer (https://github.com/Neroued/ninfer) to our local setup made of intel arc pro b580 (2x gpus)

Stage 1.
- choose specific model quantization analyze what makes the quantization format chosen by ninfer really fast
- Construct our own int2 quantization based on this schema:
[ FP16 Model ] 
      │
      ▼
[ 1. SpinQuant Preprocessing ]  ──► Rotate W & A to eliminate outliers
      │
      ▼
[ 2. DASH-Q Solver ]            ──► Generate ultra-low error quantized tensor variants
      │
      ▼
[ 3. RCO Bit-Budget Search ]    ──► Allocate DASH-Q variants per tensor under exact VRAM budget
      │
      ▼
[ Standard GGUF / Native SYCL ]

SpinQuant (Orthogonal Rotation Preprocessing): A spatial transformation step. By multiplying weight matrices and activations by learned/randomized orthogonal rotation matrices ($W' = W \cdot R$), activation/weight outliers are suppressed and spread evenly across dimensions.
DASH-Q (Diagonal Hessian & Iterative Least Squares): A non-heuristic PTQ solver. Instead of relying on off-diagonal Hessian error propagation (like GPTQ, which suffers from sampling noise at extremely low bits), DASH-Q uses diagonal Hessian approximations to solve independent weighted least-squares problems for weight assignments.

Stage 2. (tensor parallel since we have 2x gpus)
- Implement the engine (based on oneapi level zero and sycl)
1. Low-Level Native C++/SYCL Kernels (Direct Engine Port)

If you were to adapt ninfer's exact philosophy—bypassing heavy frameworks for lean C++ execution—you would target Intel's hardware primitives directly via oneAPI SYCL and Level Zero:

XMX MMA Intrinsics: Replace NVIDIA mma.sync / wmma PTX calls with Intel DPAS (Dot Product Accumulate Systolic) or XeTile SYCL primitives for INT8/INT4 matrix multiplications.

Kernel Fusion & Memory Layout: Port the fused FlashAttention and quantized GEMM kernels directly to SYCL. Xe matrix engines rely on specific register layouts (e.g., 2D block loads), so layout transposition must match Arc's L1/L2 cache and subgroup sizes (typically 16 or 32 threads per SIMD lane).

SYCL Command Graphs: Replace CUDA Graphs with sycl::ext::oneapi::experimental::command_graph to record token execution sequences and submission queues, keeping CPU launch overhead near zero during generation loops.

| ninfer Technique | Intel Arc Hardware / API Equivalent | Implementation Feasibility |
| :--- | :--- | :--- |
| **Low-Bit Quantization (NVFP4 / W4A4 / INT8)** | **Intel XMX (Xe Matrix Extensions)** & DPAS instructions via SYCL / oneAPI | **Fully Possible:** Intel Arc Alchemist and Battlemage GPUs feature XMX AI engines that natively support INT8, INT4, FP16, and BF16 matrix multiply operations with high throughput. |
| **CUDA Graphs** | **SYCL Graph Extensions** | **Fully Possible:** SYCL supports workload graph recording and submission, eliminating host CPU-to-GPU queue submission overhead during decode loops. |
| **Fused FlashAttention / W4A4 MMA Kernels** | **Intel XeTile / Intel Extension for PyTorch (IPEX)** custom SYCL kernels | **Possible:** Hand-tuned SYCL / Intel Xe-Assembly kernels can match low-level memory layout optimizations for fused operations. |
| **Multi-Token Prediction (MTP) / Speculative Decoding** | **SYCL / Level Zero C++ Runtime** | **Fully Possible:** MTP is an algorithmic framework (running speculative draft tokens parallel to main heads). It is vendor-agnostic and relies purely on batch memory bandwidth. |
| **Groupwise-INT8 KV Cache** | **Custom SYCL Quantization Kernels** | **Fully Possible:** Groupwise INT8 KV cache management reduces VRAM footprint and memory bandwidth usage on Intel Arc GPUs. |

Stage 3.
Implement features for maximizing throughput for concurrent requests, starting from continuous batching.

## Performance targets (2x Arc Pro B580)

> FOCUS (2026-09-24): our optimization target is the **27B model** — decode
> and prefill work, benchmarks, and perf claims are scoped to 27B unless
> explicitly stated otherwise. The 0.8B numbers below are legacy reference
> only, not the target.

- Token generation (decode): **40 tok/s** (per-stream, 27B model)
- Prefill: **4000 tok/s** (27B model)
- Current (2026-09-18, scripts/bench_gpu_ab.sh, DPAS48 default OFF,
  fused MLP default-ON, quiet box loadavg ~1.9):
  0.8B GPU decode TP=1 27.24 tok/s (-n 64), TP=2 14.14 tok/s.
  Pre-fusion quiet baseline was TP=1 ~24.4 (3x 24.37/24.41/24.45)
  / TP=2 ~14.3: quiet-box fused gain = +11.6% TP1, TP=2 unchanged
  as designed (fused path is TP=1 only). TP=1 still wins ~1.9x:
  2x submits/transfers/readbacks outweigh halved kernel
  time at tiny shapes. DPAS48 fast path defaulted OFF (was default
  ON): host GA64 act-quant ~6ms/token dominated, XMX kernel no faster
  than tiled legacy at M=1/128; greedy streams bit-identical either
  way (opt back in with INTFER_DPAS48=1). Decode +16-18%, prefill
  +5% vs DPAS-on (pre-fusion baselines). Prefill TP=1 (361-tok,
  2026-09-18 quiet box, fused default-ON):
  chunk8 54.7, chunk32 111.8, chunk128 149.2, chunk361 156.6 tok/s;
  TP=2: 42.3 / 89.0 / 122.4 / 134.0 tok/s. Chunk 361 (whole prompt,
  one pass) still best (fewer layer passes → fewer submits).
  Steady-state chunk128 TP=1 ~= 176 tok/s excluding the one-time 466ms
  rebuild (186 shards); recorded ~195 was lower-load.
  Prefill
  attribution (2026-09-18, chunk128 TP=1, 361 tok in 2800ms): rebuild
  924ms = preload_all one-time (186 shards, amortized to zero in
  serving); steady-state ~195 tok/s = kernel 475 + waitblock 504 +
  submit 303 + transfers ~214 + host ~380ms. Submit inflates
  84us/launch (M=1) to ~600us/launch (M=128): UR submit blocks when
  the device is saturated — same serialization wall as decode, only
  L0 immediate lists or cross-call fusion move it. 27B GPU decode ~2.8 tok/s pinned TP=2
  (kernel 66% of decode — TP tradeoff reverses vs 0.8B); TP=1 OOMs on 27B
  (lm_head INT8 tile alloc fails: ~19GB on one ~12GB B580).
  Per-token profile (0.8B TP=2): submit ~34%, host act/copy ~33%,
  kernel ~20%, wait ~11%; run-to-run variance is +/-10%.
- XMX verdict (2026-09-18, source-level + xmxmon runbook): **zero DPAS
  instructions execute in the default path** — decode M=1
  (gemm_usm_decode1_vec, quant_gemm.cpp:388) and prefill M>1 tiled
  (quant_gemm.cpp:526) are pure EU/ALU dot-products (dequant-on-the-fly
  + FP32 FMA); xmxmon would read all-zero XVE_INST_EXECUTED_XMX_* on
  every workload. Matrix engines 100% idle. Only DPAS-issuing kernels
  in the tree: quant_gemm_dpas48 (u4/u2/u8, opt-in INTFER_DPAS48=1,
  default OFF — host GA64 act-quant dominated) and quant_gemm_dpas_s8w
  (lm_head-only). Per xmxmon docs, mixed-precision DPAS counts under
  the WIDER operand (W2A8 -> INT8 counter, not INT2). Blocker:
  metric streamer needs `sudo sysctl -w dev.xe.observation_paranoid=0`
  (this box: paranoid=1, no sudo — zetMetricStreamerOpen 0x7ffffffe on
  both GPUs; counters verified present via --list-metrics).
  Harness: scripts/xmx_profile.sh joins xmxmon capture + internal
  [GPU profile] buckets, degrades gracefully to buckets-only when the
  streamer is blocked. Implication: prefill 4000 tok/s needs XMX (EU
  pipes are the whole prefill story today); decode 40 tok/s needs
  launch-count/transfer reduction (XMX at M=1 barely helps — shapes
  are memory-bound).
- XMX CORRECTION (2026-09-18, empirical via xmxmon after sysctl unlock):
  the "zero DPAS" verdict above is WRONG for lm_head — lm_head runs
  INT8 DPAS (quant_gemm_dpas_s8w, main.cpp:1231 use_dpas, default-ON
  whenever the repack exists, no env gate). Measured: prefill-only
  361tok capture = 180M INT8 slots; decode-only M=1 = 5.5M INT8 slots
  (~0.8M/token, consistent with vocab/16 tiles x K/32 ops). Layer GEMMs
  (99% of launches) remain EU/ALU, XMX idle there — that half stands.
  DEVICE ACT-QUANT PRE-PASS (2026-09-29 49b21d3; AUDITED + CORRECTED
  2026-09-30, INTFER_DPAS_PREQ=1, default-OFF): kills the host GA64
  act-quant loop that made DPAS a net +6% wall loss on 27B despite a -15%
  device-side kernel win (copy bucket 269us/GEMM + submit 127us). Quant
  moves to a standalone pre-pass kernel, dep-chained into the DPAS launch
  (trailing `sycl::event` PARAMETER, NOT a Dpas48Params member) so it adds
  NO host wait. One launch PER DEVICE, never one for both — each rank owns
  its own sycl::context (device_manager.cpp), so a kernel on rank 0's queue
  cannot legally write rank 1's malloc_shared buffer.
  === 2026-09-30 AUDIT: THE ORIGINAL GATE EVIDENCE WAS VACUOUS ===
  PREQ is NESTED INSIDE the DPAS48 path, and DPAS48 is default-OFF. The
  gates cited in 49b21d3 (cpu parity, http batch, gpu-optin 10/10) all ran
  WITHOUT INTFER_DPAS48=1, so run_linear_tp_dpas48 was never entered and
  the pre-pass NEVER FIRED in any of them. They prove nothing about PREQ.
  Worse, the cited invariant "streams bit-identical base vs PREQ" is
  impossible: DPAS48's own INT8/GA64 act-quant legitimately differs from
  the FP16-act base. The correct invariant is PREQ == DPAS48 (the pre-pass
  reproduces the host GA64 quant).
  === 2026-09-30 AUDIT #2: THE NEWTON "FIX" WAS BACKWARDS. DO NOT RE-ADD IT ===
  An earlier revision of this audit concluded "device 1.0f/s is not IEEE,
  add an fma Newton step". WRONG IN DIRECTION, and reverted. Truth, from a
  direct device probe (9,689,600 realistic GA64 elements incl. adversarial
  exact-.5 tie zones: 0 mismatches for the bare kernel expression vs the
  host AVX2 lane path) plus exact rational arithmetic on the bit patterns:
    * `q = trunc(x*inv +- 0.5)` is PATHOLOGICALLY ILL-CONDITIONED in `inv`:
      a 1-ulp reciprocal difference moves the product across a .5 tie and
      flips the code by a full +-1. Two implementations that do not share
      the SAME reciprocal bits disagree on ~2 elements per ~1950 GEMM
      calls. Correctness here means sharing `inv`, NOT being correctly
      rounded.
    * A DEVICE `1.0f/s` lowers to rcp.approx: 15,868/151,400 mismatches vs
      host divss over legal s.
    * The HOST `1.0f/s` in dashq.cpp is ALSO 1 ulp off the correctly
      rounded value IN THIS BINARY (exact integer rational comparison of the
      two candidate bit patterns). Same expression, same -O3 flags, no
      fast-math anywhere: codegen-context dependent, never assume IEEE.
  => "Correcting" the device moved it AWAY from the reference and CREATED
  the decode-step-14 divergence this same commit then documented as an
  unfixable "KNOWN, UNRESOLVED tie rule". The Newton step is GONE.
  === THE FIX: ship the host's (scale, inverse) to the device ===
  dashq::act_quant_ga64_scale_inv_batch() computes per 64-slice
  (s, inv) = (max(1e-6,max|x|)/127, 1.0f/s) ON THE HOST; the pre-pass
  consumes both verbatim via ActQuantDevParams::scale_inv_in (2*M*nGA FP32
  interleaved, per-device `ws48_[d].si` landing zone, tiny H2D).
  Bit-exact BY CONSTRUCTION. Bonus: the device no longer redoes the O(64)
  max-abs reduction, removing the only way the scale it WRITES could
  disagree with the inverse it used. Cost: the O(M*K) max-abs reduction is
  back (the pre-pass was built to delete the full host quant); it is a
  reduction plus M*K/64 divides instead of a full host quant, and stays off
  the GEMM critical path (dep-chained, no added wait).
  === OTHER FIXES ===
  (a) PARTIAL-DECLINE RACE (nondeterminism): if device d's pre-pass
  launched but device d+1 declined, the already-queued d kernel still
  writes ws48_[d].act while the host fallback memcpy's the same buffer.
  Now every submitted preq_event is drained before falling through.
  (b) `if (xd != x.data())` compared a USM pointer against a host
  std::vector pointer — never legitimately equal, always copied. Now an
  unconditional memcpy with a comment.
  (c) profile_note_launch() ran BEFORE the decline check, so a declined
  call inflated nk. Moved after.
  (d) Indentation regression in quant_gemm_dpas4_row fixed.
  === FIXED (was "KNOWN, UNRESOLVED"): PREQ x SSM_GPU tie divergence ===
  Real and deterministic (step 14, TP=1 and TP=2, ~2 mismatched elements per
  ~1950 GEMM calls), but NOT a tie-rule bug: it was the reciprocal (AUDIT #2
  above) — the device was correctly rounded and the HOST was the 1-ulp-off
  one. The earlier "host gives 48 where trunc(v+0.5) gives 47" evidence was
  an artifact of a reference using `_mm_cvtss_si32` (CVTSS2SI = round to
  NEAREST) instead of `_mm_cvttss_si32` (CVTTSS2SI = truncate); dashq
  correctly uses `_mm256_cvttps_epi32`. Writing a host reference for this
  rule with the rounding intrinsic makes the real, correct device path look
  wrong on ~53% of elements. PREQ now composes with SSM_GPU and the gate
  ASSERTS it. Correctness-gated: gpu-optin PASS (base arms + PREQ==DPAS48 +
  ROW+PREQ + PREQ+SSM_GPU + SSM_GPU + PREQ-alone inert, n=24, TP1/TP2), cpu
  parity PASS, http batch PASS. Path fires: decode nk 1116->1488
  (+372 = 4tok x 93), nw unchanged 744.
  NO wall-time claim (never benchmarked here) — needs a quiet-box 27B
  INT4/G64 A/B before any perf claim.
  Orthogonally: device-act-quant INTO the fused DPAS kernel stays a dead
  end (replication), but the standalone pre-pass is a different shape and
  is now open. SYCL-ENVS GOTCHA: a plain non-sourced shell cannot
  enumerate ANY GPU ("No devices matching filter 'Arc'") because
  libumf.so.1 (at /opt/intel/oneapi/umf/1.1/lib) is off LD_LIBRARY_PATH,
  so every UR adapter fails to load. Export
  LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:... before any --gpu run;
  verify with sycl-ls. xpu-smi still shows the GPUs fine — do not debug
  device_manager.cpp for this warning. (memory: sycl_env_libumf.md)
  DPAS48 re-exam at M=361 (interleaved A/B, contended box, 2/2 rounds):
  OFF wins prefill 147.8/146.3 vs 132.7/131.0 (~11%) AND decode
  6.70/6.87 vs 5.13/5.06 (~30%); kernel 318->496ms (one-WI-per-16N-tile
  replicates O(K) fold work nT x), rebuild 500->964ms (DPAS repack),
  copy 12.6->48.4ms (host act-quant). Streams bit-identical. Default-OFF
  stands at ALL shapes (M=1/128/361).
  DPAS48 M=1 device-side breakdown (2026-09-19, -n8 TP1 decode buckets —
  kernel_us is device event time, LOAD-IMMUNE, box load ~7): legacy
  kernel 138.0 / submit 136.4 / waitblock 154.9 / copy 0.1 / readback 65.6
  nk=1680 vs DPAS48 kernel 203.5 (+48% DEVICE-SIDE — the M=1 DPAS kernel
  itself is slower, not just its host quant) / submit 235.6 (+73% at LOWER
  nk=1488: UR submit blocks when the device is saturated by the longer
  kernels) / copy 91.8 (host GA64) / readback 100.2 (nr 1056 vs 576, ~2x
  readbacks). DPAS48 loses on EVERY bucket at M=1. Consequence: a
  standalone device-side GA64 act-quant kernel (row-parallel pre-pass —
  feasible, no replication, HalfAway rounding verifiable like the RNE
  convert) canNOT save DPAS48 alone (would fix only the 92ms copy of a
  path already +65ms kernel / +100ms submit underwater). XMX prefill
  needs FIRST a rewritten 2D-tiled DPAS kernel (proper subgroup DPAS, no
  one-WI-per-16N-tile K-fold replication); device act-quant is the
  complement AFTER, not the lever. Neither is a session-scale build —
  scoped, not started. Do NOT retry device-quant-into-DPAS (dead end
  stands) and do NOT build standalone device-quant expecting a DPAS48
  win without the kernel rewrite.
- Prefill defaults flipped (2026-09-18, a52715f): chunked M-row prefill
  is now default-ON (chunk_M 8->128, batch_gemm default-ON; opt-outs
  INTFER_PREFILL_CHUNK=0 / INTFER_BATCH_GEMM=0 reproduce old path
  bit-identically). Defaults previously ran the token-by-token step loop
  (20-25 tok/s on 361tok); new defaults 148.2 tok/s (6x, contended box).
  chunk128 ~= 95% of chunk361 gain (quiet-box 149.2 vs 156.6).
  Decode -n64 unchanged (26.90->26.14, run variance). 27B/GPU joint-batch
  wide-chunk guard synced to default-ON semantics (would have disarmed).
  Gates: cpu parity + http batch PASS, GPU streams identical.
  Serve re-baseline on new defaults (2026-09-18, contended load ~9):
  sweep PASS (C=1 2.46 / C=4 3.59 / C=8 2.27 tok/s MT=8; parity held);
  MT=64 probe 7.66 tok/s over 338 ctoks — same-run pair with MT=8 C=8
  2.27 (short requests prefill-dominated, batch drains).
- 40 tok/s recipe (2026-09-18, per-token arithmetic @25.9 tok/s =
  38.6ms: kernel 9.6 / submit 11.7 / waitblock 9.5 / transfers 10.4ms,
  194 launches): device-act seams kill transfers (->35 tok/s), batched
  L0 multi-launch halves submit (->31), both = 22.2ms = 45 tok/s.
  Engine-invasive; needs quiet box (load<2) + gates. Not attempted
  under load. (memory: path_to_target_40_4000.md)
- OpenVINO evaluated, NOT adopted (2026-09-18 spike): OV 2026.3 sees
  both B580s, but FP16 matmul M=361 2.6ms (~our EU kernel, 1.0 TFLOPS
  vs 100+ peak — no XMX unlock) and M=1 1.2ms vs our ~50us (20x worse,
  framework overhead). Native SYCL stays; XMX win must come from our
  own DPAS kernels with device-resident act-quant.
- Wave shared-input convert-once (2026-09-18, f12f76b): run_linear_wave
  Phase 1 converts each distinct (pointer, bytes) input once, reuses
  staging bytes for duplicates (gate/up share rin; M-row batch gate/up
  share c_rin). Bit-identical, slots stay per-item, call-local scope.
  Dormant for distinct inputs (k/v, z/b/a) and fused path. Measured
  load-immune copy counts (NO_FUSE -n 6): 1116->972 converts (-13%);
  fused TP=1 unchanged (276=276). Streams identical TP1/NO_FUSE/TP2.
  Gates: cpu parity + http batch PASS. Attribution CLOSED (2026-09-19,
  INTFER_WAVE_DBG probe, since reverted): the "6 full-attn skip" note was
  wrong — layers 3,7,11,15,19,23 fire the gate/up dedup-hit like every
  other layer (168 hits = 24 prefill + 144 decode = 24 layers x 7 passes).
  Dedup fires only on 2-item shared-input gate/up waves; never on z/b/a
  (3 distinct rotated inputs) or k/v (2 distinct). -n6 wall +2.8% (run
  variance — convert is not the wall, submit is).
- Dead ends (measured, do not retry as-is): device-side act-quant
  (bit-exact probe passed but fused kernel slower: one-WI-per-16N-tile
  layout replicates O(K) quant work nT x), host-private staging of DPAS
  writes (neutral), staged readback (neutral), SYCL graphs on B580
  (19.7us vs 11.9us plain submit), LA-branch run-parallel OMP in
  step_batch (2026-09-16, parity-clean incl. segfault fix for TLS reads
  inside OMP workers, but zero gain: CPU C=8 batch WIP 10.5s vs HEAD
  10.8s @20T, 16.18s vs 16.34s @1T; GPU serve-load C=8 3.86 vs 3.89
  tok/s. LA host recurrence is a negligible share — GEMMs dominate.
  Revisit only if profiling shows recurrence >10% of a step. NOTE
  2026-09-18: the 8.16/15.33/17.98 (C=1/4/8) + 36.3 (MT=64 probe)
  numbers claimed 2026-09-17 are WITHDRAWN as unreproducible. Four
  independent C=1/MT=8 sweeps (incl. quiet-box start-load 0.69 and
  1.57) all land 3.27-3.53 tok/s; only docs commits separate the
  3.89 (09-16) and 8.16 (09-17) readings, i.e. same code. Bisect
  (single warm req, quiet box): fused ON 1.78s vs OFF 1.77s (neutral),
  joint 1.78s vs split 2.01s (joint helps, active), dispatcher 1.78s
  vs serial 2.07s (dispatcher active), chunk361 1.68s (chunking minor).
  The MT=64 batch-full probe (9.02 tok/s over 338 ctoks) proves the
  path batches — MT=8 is prefill-dominated, not contention. C=8
  scaling inversions under rising box load are a sweep-harness
  artifact (load 2.5→10+ mid-run). Quoting rule retired (C=1>=7 never
  fires): quote the MT=8 sweep AND the MT=64 probe from the same run.
  Dead-end verdict on OMP stands),
  device-side SSM recurrence M=1
  (2026-09-16, INTFER_SSM_GPU=1, parity-clean incl. step_batch +
  spec-rollback resync: -9% decode, 7.4->6.7 tok/s pinned 3x3
  interleaved. Per-LA-layer attribution: prep ~4us, H2D ~15-130us,
  kernel-submit ~25us, D2H+wait ~150-550us vs replaced host loop
  ~0.1-0.2ms — the blocking wait is structural: host norm needs
  core on-host same-layer. Revisit only with fused device-side
  SSM+norm feeding la_out GEMM from device buffers, or M>>1 where
  the single wait/layer amortizes; M=8 estimate still loses),
  native L0 immediate submit (2026-09-18, examples/l0_submit_probe spike,
  since removed: native immediate append+host-sync 27.6us vs UR
  submit+event.wait 25.2us per launch on a trivial kernel, correctness
  PASS — same floor, not a UR-adapter artifact. Revisit only with a batched
  multi-launch list amortizing one sync over many kernels),
  INTFER_NO_PROF queue without enable_profiling (2026-09-18: 19.21 vs
  18.75 tok/s TP1 decode under box contention — within run variance;
  per-event L0 timestamping is not the submit tax).
- PLAN.md triage (2026-09-19, grounded vs tree): preload_all/USM-pool /
  fused-MLP already DONE (preload log: 186 shards pinned, 534MiB dev0,
  no evict; pool_acquire routed, pool holds 0 bufs post-preload as
  designed — grow-only workspaces). DPAS-enforce + SYCL-graphs + L0-P2P
  are measured dead-ends above (do not retry as-is). INT8-KV
  (INTFER_INT8_KV / INTFER_KV_Q8) + SSM-chain (ssm_forward_chain via
  INTFER_SSM_GPU) exist as opt-in paths, certified stream-identical on
  GPU by scripts/test_gpu_optin.sh (10/10: base/KV/Q8/Q8+HAD/SSM_GPU x
  TP1/TP2, .intifer stream 279/6511/314/279) — wall-time comparison
  still needs quiet box (load<2).
- First-touch migration (2026-09-21, measured on 27B TP=2 serve):
  preload_all builds 496 shards in ~49s and pins them (zero rebuilds on
  first request), but shard USM is malloc_shared so pages stay host-resident
  until first kernel touch: xpu-smi device-local at startup dev0 791MB /
  dev1 51MB, jumping to 11.27GB / 10.94GB only after the first request.
  Same-prompt back-to-back (contended box, load ~6): req1 9.28s vs req2/req3
  2.75s/2.75s — first request pays ~6.5s of migration stalls, steady state
  reached after exactly one request (streams identical). q.prefetch() over
  all shards was built, measured (18.7GB "migrated" in 59ms), and REVERTED:
  device counters unchanged, first-request jump identical — prefetch is a
  no-op hint for shared USM on this Level Zero driver (do not retry as-is;
  note in engine.cpp preload_all). Mitigation: one warmup request after
  /health goes live (scripts/serve_27b.sh does this, INTFER_NO_WARMUP=1
  skips). A startup touch-kernel over all 496 shards would only move the
  cost, not eliminate it — unscoped.
- Wave-submit + vectorized host math (2026-09-18, wave 7d7ca72, AVX2
  5285b47, extended a78e99f, quantified on quiet box loadavg<2): z/b/a +
  gate/up + full-attn k/v GEMMs convert+submit back-to-back, one wait
  per wave (scalar + M-row paths; INTFER_NO_WAVE=1 restores sequential).
  a78e99f adds: scalar full-attn k/v wave-2 (q stays sequential — its
  split/norm/rope consumers precede k/v input prep), M-row wave-batch
  for full-attn k/v + gate/up at all three batch sites. 8bce85c
  (2026-09-19) joins q into the k/v wave-3 and qkv into the z/b/a
  wave-4 in step() decode: both GEMMs' consumers (q split/norm/rope,
  qkv conv) run after, so the joins are pure regrouping, bit-identical
  by the wave contract. CORRECTION: the joins do NOT reduce kernel-wait
  count (nw 828 = 828 — wait_count increments per kernel event, and
  grouping changes no event count). What they halve is host round-trip
  phases: one wait-loop + one readback pass per wave instead of per
  GEMM (decode nr 828->432). M-row joins (2026-09-19, this session):
  the same q/qkv joins applied to the M-row paths — prefill-chunk
  full-attn q+k/v wave-3 and qkv+z/b/a wave-4, plus step_batch (serve)
  full-attn wave-3 and LA wave-4 (4 sites; q/qkv inputs staged into
  c_q_in/c_qkv_in/b_q_in/b_qkv_in copies since c_x is reused for k/v/z/b/a
  staging). INTFER_NO_WAVE=1 still restores sequential at all sites.
  Verified load-immune: prefill nr 138->72 (25-tok prompt), decode
  streams identical TP1/TP2/NOWAVE (279 6511 314 279 ...); cpu parity +
  http batch + gpu-optin PASS. Wall-time A/B deferred (box load >2).
  Each saved phase removes a blocking
  event-wait loop + host readback-loop setup (~24/layer-pass pairs per
  token). Measured load-immune (TP1 fused -n 6),
  streams identical TP1/TP2/NOWAVE; cpu parity + http batch + gpu-optin
  PASS. A/B CLOSED (2026-09-19, quiet box load~1.6, -n64 TP1 interleaved
  2 rounds + 3 JOIN-only): JOIN 25.97/25.99, PARENT 26.09/25.97 = NEUTRAL
  (-0.2%, run variance). Buckets: kernel/submit/waitblock/usm_write
  identical; readback +8-12ms (bigger wave-3/4 readback loops cost what
  the saved loop setups save); nr 4608 vs 6144 confirms grouping.
  Keep (fewer phases, same streams) but quote NO tok/s gain — submit
  (~745ms) + kernel (~645ms) dominate unchanged. Precision:
  wave does NOT reduce kernel-launch count (nk identical 1674/10tok
  both ways — engine still issues one kernel per item); it groups
  convert+submit back-to-back for queue depth and one wait per wave.
  AVX2/FMA rmsnorm/l2norm (INTFER_HOST_AVX2=0 restores scalar). Both oracle-certified: CPU
  parity + HTTP batch PASS, scalar-vs-AVX2 streams identical on CPU
  AND GPU paths. Wave: +0.6-0.7% decode, 3/3 interleaved rounds
  (TP1 24.61->24.78, 25.03->25.17, 24.94->25.09) — real but small:
  wave saves waits, submit (~830ms/65tok) is the wall. AVX2 norms:
  NEUTRAL (+0.7/-1.3/+0.4%, mean ~0) — host rmsnorm/l2norm is not on
  the decode critical path (the ~33% host share is GEMM act
  convert+memcpy, not norms). Keep both (correct, default-on, free),
  but quote no big win; remaining gap needs submit-side work.
- L4 design verdict (2026-09-18, investigated; MLP half IMPLEMENTED):
  PART A (done, fused MLP): the device-act GEMM seam now EXISTS —
  run_mlp_fused feeds the down GEMM straight from a device buffer
  (fused silu-mul output), skipping host convert + ws memcpy.
  PART B (not done): fused device-side SSM+norm feeding la_out GEMM.
  Data flow per LA layer today (18/24 layers on 0.8B): ssm_rows kernel
  -> D2H core (2048 floats) + blocking wait -> host ssm-norm+silu(z)
  -> H2D normed_core via run_linear_into -> la_out GEMM (engine.cpp
  converts FP32->FP16 host-side, memcpy to ws act, submit, wait, D2H).
  Fusion would kill per-LA-layer: 1 D2H + 1 wait + 1 H2D + 1 FP32->FP16
  host convert (la_out act 1024 FP32 = 4KB). BUT the old -9% stands and
  the reason is now precise: ssm_rows itself already pays H2D inputs
  (~15-130us/layer) + kernel-submit (~25us) + D2H+wait (~150-550us);
  fused norm+FP16-convert on device saves only the core round-trip and
  the host norm (~0.05ms), while the la_out engine seam
  (run_linear_tp: host FP32 in -> FP16 convert -> ws memcpy -> submit)
  has NO device-pointer entry — feeding it from a device buffer needs
  a new Engine method (device-act GEMM: skip convert+ws-write, submit
  with USM dep on the fused event). Net estimate: saves ~0.2-0.3ms of
  ~1.7ms/LA-layer (submit+waitblock dominate) over 18 layers =
  ~4-5ms/token of ~41ms = ~+10% IF the new seam costs zero extra
  submits. Verdict: viable but engine-invasive (new device-act GEMM
  entry + fused ssm_forward_norm_convert kernel + oracle gate on the
  fused FP16-convert rounding); do it only together with batched
  multi-launch L0 lists (same seam work amortizes both). Standalone
  INTFER_SSM_GPU=1 stays default-off. NOTE: the SAME device-act seam
  pattern was just proven by the fused MLP below (+13% decode) — the
  L4 SSM fusion is now a second application of a working technique,
  not a new invention; the remaining L4-specific risk is only the
  fused norm+silu(z) numerics.
  Data flow per LA layer today (18/24 layers on 0.8B): ssm_rows kernel
  -> D2H core (2048 floats) + blocking wait -> host ssm-norm+silu(z)
  -> H2D normed_core via run_linear_into -> la_out GEMM (engine.cpp
  converts FP32->FP16 host-side, memcpy to ws act, submit, wait, D2H).
  Fusion would kill per-LA-layer: 1 D2H + 1 wait + 1 H2D + 1 FP32->FP16
  host convert (la_out act 1024 FP32 = 4KB). BUT the old -9% stands and
  the reason is now precise: ssm_rows itself already pays H2D inputs
  (~15-130us/layer) + kernel-submit (~25us) + D2H+wait (~150-550us);
  fused norm+FP16-convert on device saves only the core round-trip and
  the host norm (~0.05ms), while the la_out engine seam
  (run_linear_tp: host FP32 in -> FP16 convert -> ws memcpy -> submit)
  has NO device-pointer entry — feeding it from a device buffer needs
  a new Engine method (device-act GEMM: skip convert+ws-write, submit
  with USM dep on the fused event). Net estimate: saves ~0.2-0.3ms of
  ~1.7ms/LA-layer (submit+waitblock dominate) over 18 layers =
  ~4-5ms/token of ~41ms = ~+10% IF the new seam costs zero extra
  submits. Verdict: viable but engine-invasive (new device-act GEMM
  entry + fused ssm_forward_norm_convert kernel + oracle gate on the
  fused FP16-convert rounding); do it only together with batched
  multi-launch L0 lists (same seam work amortizes both). Standalone
  INTFER_SSM_GPU=1 stays default-off.
- Fused MLP down-projection (2026-09-18, IMPLEMENTED, TP=1 only):
  Engine::run_mlp_fused chains gate/up wave-submit on-device through a
  fused silu-mul + RNE FP16-convert kernel (gemm::silumul_to_half) straight
  into the down GEMM: one host wait + one D2H per MLP instead of wave-wait
  + 2 D2H (gate/up) + host silu-mul + convert + H2D + down-wait.
  TP>1 / DPAS48 / trace (NaN/H) / bad shapes return false and the caller
  keeps the host path (mg/mu consumers are trace-gated). Kill switch:
  INTFER_NO_FUSE_SILUMUL=1 restores wave + host silu-mul (bisection).
  Bit-contract: FP32 op order matches the host loop; sycl::exp may differ
  ~1ulp on rare elements; dev_float_to_half is RNE bit-identical to
  dashq::float_to_half (verified exhaustively over e in [-10,0], 2M
  random + boundary mantissas, zero mismatches — the AGENTS.md "e == -10
  edge" note was wrong: dashq mutates mant with the hidden bit BEFORE
  the round test, so host and device test the same bit).
  (box contended loadavg 5-11, interleaved A/B 2 rounds, TP=1 decode
  -n 32): 28.06/27.81 vs 24.77/24.48 = +13.3%/+13.6% (~+3.2 tok/s);
  prefill chunk361: 158.3 vs 154.2 = +2.7%. GPU streams bit-identical
  ON vs OFF (279 6511 314 279 3046 13 198 760) and vs TP2; all gates
  PASS with fused default-ON (cpu parity, http batch, serve-load).
  Quiet-box confirmed 2026-09-18 (loadavg ~1.9, fused default-ON,
  -n 64): TP=1 decode 27.24 tok/s (+11.6% vs 24.4 pre-fusion quiet
  baseline — matches the +13.3/+13.6% contended A/B within run
  variance), TP=2 14.14 (unchanged, fused is TP=1 only), prefill
  chunk361 156.6 (+2.8% vs 152.2, matches contended +2.7%). New
  single-stream ceiling: ~27 tok/s TP1. The +13% relative gain is the
  durable claim (waits/round-trips removed, submit wall untouched).
  fused device-side SSM+norm feeding la_out GEMM from device buffers.
  Data flow per LA layer today (18/24 layers on 0.8B): ssm_rows kernel
  -> D2H core (2048 floats) + blocking wait -> host ssm-norm+silu(z)
  -> H2D normed_core via run_linear_into -> la_out GEMM (engine.cpp
  converts FP32->FP16 host-side, memcpy to ws act, submit, wait, D2H).
  Fusion would kill per-LA-layer: 1 D2H + 1 wait + 1 H2D + 1 FP32->FP16
  host convert (la_out act 1024 FP32 = 4KB). BUT the old -9% stands and
  the reason is now precise: ssm_rows itself already pays H2D inputs
  (~15-130us/layer) + kernel-submit (~25us) + D2H+wait (~150-550us);
  fused norm+FP16-convert on device saves only the core round-trip and
  the host norm (~0.05ms), while the la_out engine seam
  (run_linear_tp: host FP32 in -> FP16 convert -> ws memcpy -> submit)
  has NO device-pointer entry — feeding it from a device buffer needs
  a new Engine method (device-act GEMM: skip convert+ws-write, submit
  with USM dep on the fused event). Net estimate: saves ~0.2-0.3ms of
  ~1.7ms/LA-layer (submit+waitblock dominate) over 18 layers =
  ~4-5ms/token of ~41ms = ~+10% IF the new seam costs zero extra
  submits. Verdict: viable but engine-invasive (new device-act GEMM
  entry + fused ssm_forward_norm_convert kernel + oracle gate on the
  fused FP16-convert rounding); do it only together with batched
  multi-launch L0 lists (same seam work amortizes both). Standalone
  INTFER_SSM_GPU=1 stays default-off.
- lm_head audit (2026-09-19): step_batch/serve scoring is ALREADY M-row
  batched — run_lm_head_batch (main.cpp:1174, M>1) does one launch per
  device tile instead of M scalar calls; INTFER_LMHEAD_BATCH=0 restores
  per-row for bisection. Nothing to implement; todo item cancelled.
  Verified on HEAD: cpu parity + http batch + gpu-optin (10/10) PASS,
  WAVE vs NOWAVE GPU streams identical TP1/TP2 (279 6511 314 279).
- L4-lite IMPLEMENTED, default-off (2026-09-19, INTFER_L4_FUSE=1 opt-in):
  second application of the fused-MLP device-act pattern, scoped to TP=1
  scalar decode (the 40 tok/s target). Per LA layer it chains solo-run
  ssm_forward -> fused ssm_norm_silu_to_half kernel (rms-norm + silu(z) +
  RNE FP16 convert, same loop order as host) -> la_out GEMM via the new
  Engine::run_linear_tp_devact seam (device FP16 act, no host convert/H2D,
  dep-chained, single wait + single D2H). New files touched:
  quant_gemm.hpp/cpp (SsmNormParams + kernel), engine.hpp/cpp
  (run_linear_tp_devact: TP=1-only, FP16-act-only, no-mutation-on-false,
  ws_[0].out alias guard), main.cpp (gpu_ssm_resolve extracted from
  gpu_ssm_rows as a shared helper — pure move; GpuBackend L4 staging
  buffers + ti_of; gpu_la_out_fused; step() hook with trace_h forcing the
  host path; INTFER_L4_DBG=1 debug counter). Saves vs ssm+host path: 1
  D2H (core) + 1 H2D (normed) + host norm/convert + 1 wait per LA layer;
  costs 1 norm submit + 2 small H2D (z, norm-w). Correctness-gated NOW
  (load-insensitive): L4-vs-base streams IDENTICAL TP1 n=4/16/32 (279 6511
  314 ...), TP2 flag-vs-noflag identical (L4 silent no-op TP>1), SSM_GPU x
  L4 4-way identical, cpu parity + http batch + gpu-optin (10/10) PASS on
  the final binary; [l4] counter proves the path fires (72 calls/run).
  Wall-time A/B NOT run (box load 8.78, rising — quiet rule load1<2):
  run scripts/bench_quiet_ab.sh on a quiet box before any perf claim.
  L0 verdict stands: raw command lists measured same-floor (27.6 vs
  25.2us), so no L0 scaffolding was built — the multi-submit-single-wait
  chain IS the achievable amortization on in-order UR queues, and the
  devact seam is the prep a future batched launcher needs.
- L4 serve M==1 fast path IMPLEMENTED, default-off (2026-09-19, same
  INTFER_L4_FUSE=1 flag): single-row step_batch serve batches route each
  LA layer through the certified gpu_la_out_fused (conv loop advances
  conv_state first, matching scalar order; success skips the ssm_rows +
  host-norm + M-row la_out sections for that layer; any false falls
  through to the standard path with eligibility cleared for the layer).
  M>1 deliberately NOT fused: run_linear_batch(la_out) is already one
  M-row GEMM (single submit+wait), so a thin norm offload would only trade
  host convert for M norm submits — wash or loss; the M>1 win needs
  device-resident cores (same precondition as the L4 verdict above).
  Gates (all PASS, load-insensitive): serve C=1 L4-vs-base byte-identical
  ([l4] fires on the serve path, BASE counter 0), TP2 flag/no-flag
  identical (silent no-op), SSM_GPU x L4 4-way CLI identical, SSM_GPU+L4
  serve stream identical to base (shared SsmDev residency consistent when
  mixing),   cpu parity + http batch (with L4 ON: single/batch/SSE/serial
  all PASS, oracle Paris) + gpu-optin 10/10. Wall-time A/B NOT run (box
  load 5.58 — quiet rule load1<2): run scripts/bench_quiet_l4_ab.sh on a
  quiet box before any perf claim (new script; covers CLI decode TP1 -n64
  + chunk361 no-op guard).
- Profiler blind spot FIXED (2026-09-19, profile_note_launch): SSM/norm
  kernels submit directly through quant_gemm free functions, so their
  launches never reached the [GPU profile] nk counter (L4-vs-base decode
  showed nk=1680 both ways while submit dropped 91->64ms — invisible
  work). ssm_forward/chain + ssm_norm_silu_to_half now count one launch
  each via intfer::gemm::profile_note_launch (defined in engine.cpp,
  zero-cost when profiling is off; intfer_gemm gets INTFER_PROFILE_GPU in
  src/CMakeLists.txt). Re-run TP1 -n8: prefill nk=210 both (L4 no-op,
  exact), decode BASE nk=1680 vs L4 nk=1968 = +288 = 18 LA layers x 8
  tokens x (1 SSM + 1 norm), streams identical. The +288 proves the whole
  decode fires the fused chain; the -n64 quiet A/B will show whether the
  submit/usm_write savings (-30%/-12% in buckets, contention noise aside)
  convert to tok/s.
- nk accounting matrix VALIDATED (2026-09-19, TP1/TP2 x BASE/SSM/L4/BOTH,
  -n8 decode, counters only — load 4-5, no wall-time read): all 8 streams
  identical (6511 314 279 3046 13 198 760 6511). TP1: BASE 1680, SSM 1824
  (+144 = 18 LA x 8 tok x 1 SSM), L4 1968 (+288 = x2, SSM+norm), BOTH 1968
  (L4 subsumes the SSM launch — no double-SSM, shared SsmDev residency
  confirmed in counters). TP2: BASE 2976 (= 2 x 1488? — per-device GEMM
  launches counted per shard while SSM stays single-queue; do NOT read
  TP2 nk as 2x work), SSM 3120 (+144, same single-queue SSM delta),
  L4/BOTH 2976/3120 (L4 silent no-op TP>1 — zero fused launches, the
  TP=1-only guard proven in the counter, not just in streams). The
  counters now decompose every launch source on the decode path.
- L4 M-row serve fusion IMPLEMENTED, default-off (2026-09-19, fcbade9,
  same INTFER_L4_FUSE=1 flag, TP1 only): supersedes the "M>1 deliberately
  NOT fused" note above — step_batch LA layers with M>1 now route through
  gpu_la_out_fused_batch (one cross-key ssm_forward over M distinct
  (req,layer) states via a gathered pointer table + one M-row
  ssm_norm_silu_to_half_m kernel + one M-row devact la_out GEMM) instead
  of ssm_rows + host norm + M-row la_out. Shared-state rows (same-request
  prefill runs in joint batches) and TP>1/DPAS48/bad shapes decline
  inside the callee to the standard path. The M>1 precondition that made
  the old verdict ("needs device-resident cores") is met by staging a
  contiguous M-row core block + pointer table in GpuBackend L4 buffers
  (grow-only, freed in dtor). Cost per LA layer: 1 SSM submit + 1 norm
  submit + seam GEMM submits (vs ssm_rows solos + host norm/convert +
  H2D + la_out-wait); host work unchanged (M x O(kd) q/k prep + 2 small
  H2D). Gates (all PASS, load-insensitive): 3-way concurrent serve batch
  L4-vs-base byte-identical (M=2/M=3 fused-batch fired in [l4] counters),
  scalar streams identical TP1/TP2 (279 6511 314 279 3046 13 198 760),
  cpu parity + http batch (4.12 tok/s) + gpu-optin 10/10 on the final
  binary. Debug: INTFER_L4_DBG=1 prints per-path counters ([l4] fused
  la_out / fused batch la_out M=%u). Wall-time A/B NOT run (box load
  ~6 — quiet rule load1<2): scripts/bench_quiet_l4_ab.sh running in
  background, gated on quiet.
- L4 M>1 measured on M=3 sched batch (2026-09-19, 3/3 BASE/L4 pairs,
  near-quiet load 2-3.8, 27 toks/3 reqs, -n8): submit −24% systematic
  (110.8/112.6/112.0 -> 84.4/84.7/85.2ms, variance <2ms — real), nk
  1536->1824 (+288, fused chain fires), usm_write −5ms, BUT waitblock
  +21ms (597->618ms) eats nearly all of it: net wall −1.5% (946-948 ->
  929-935ms, consistent sign 3/3). Mechanism: at M=3 the device is
  saturated (kernel 597ms dominates the 940ms wall), so fewer host gaps
  just deepen the device queue — submit savings only convert to tok/s
  where the device has idle gaps (scalar M=1, kernel ~26% of decode).
  Implication for 40 tok/s: the M>1 L4 batch is correctness-certified
  infrastructure (streams identical, fewer host round-trips), NOT a
  wall-time lever under saturation. UPDATE 2026-09-19 (bench_quiet_l4_ab.sh
  3 rounds DONE, box never quiet — load 1.91/2.24/2.55/2.78/3.04/3.17):
  scalar A/B is a VARIANCE WASH, no promotion. -n64 TP1 decode L4 vs BASE
  = 17.19/19.52, 19.63/16.28, 15.32/26.45 (round-3 BASE 26.45 at load 3.17
  beats round-1 L4 17.19 at load 1.91 — the box, not the code, decides).
  Buckets: submit L4 wins 2/3 (716/955, 665/1186, 845/726) but
  kernel/waitblock flip randomly with load; nk=15744 vs 13440 every round
  (+2304 = 18 LA x 64 x 2, chain fires). Prefill guard clean (152-155 both,
  L4 no-op confirmed). L4 STAYS default-OFF opt-in (certified infra, no
  tok/s claim either way). Do NOT re-run hoping for a better number —
  re-run ONLY on a sustained-quiet box (load1<2). Both A/Bs point the same
  way: submit-side savings convert only where the device has idle gaps.
- L5 LA-front chain (2026-09-19 hooked 6c31136, stream-identical; 2026-09-20
  op-cut 7a6cee3): extends the device-act pattern FORWARD from la_out to the
  whole LA input path. Per LA layer scalar path today = host wave-4 (4 GEMM
  round-trips) + host conv + host prep + L4 tail. L5 replaces with: H2D
  normed (1x) -> fwht_rot_to_half kernel (4x FWHT rot + RNE convert)
  -> 4x devact GEMM (existing seam) -> la_conv_prep kernel (conv FIR+silu +
  qk-prep pack into sd->in) -> certified L4 tail (ssm_forward + norm +
  devact la_out). conv_state goes device-resident in SsmDev (dirty-tracked,
  host resyncs). Bit-contracts: FWHT/conv-FIR/packs exact host order;
  silu/exp/log1p ~1ulp (oracle arbitrates). Kill switch: INTFER_L5_FRONT=1
  (default OFF); any false keeps the L4/host path. Stream-gated TP1 -n 6:
  base == L5 == L5+HOSTPREP == L5-TP2-decline (279 6511 314 279 3046 13).
  SUBMIT-WALL VERDICT (2026-09-20, structural, load-immune): L5-front scalar
  decode is a DEAD END as built — do not chase wall-time on it. Buckets
  (TP1 -n 6): base nk=1260/nw=828/kernel=63.7ms vs L5 nk=2124/nw=828/
  kernel=61.3ms, waitblock 318 vs 62ms. Same device work, same wait COUNT
  (the single-wait seam works — devact + dep chains, no extra waits), but
  +144 launches/token = +8/LA layer (fwht, 4x half_to_float, prep, tail
  ssm+norm the host path never launches) and ~150 extra UR submits/token
  serialize the in-order queue: the device starves between host submits and
  the final wait per layer drains idle time, inflating waitblock 5x. 7a6cee3
  cut the resend fat (per-layer cw/dtb/alog/kinds const cache l5c_ +
  dropped the provably-redundant conv-writeback wait, which returned
  immediately) — nk/nw confirm the remainder is kernel-launch count, not
  waits or H2Ds. Fusing 4x h2f into 1 kernel would save only ~3 launches/
  layer (~4ms) for real risk — not worth it. L5 stays default-OFF; the
  kernels + stream-identity proof are retained for a future fused
  fwht->GEMM->prep kernel or M-row/serve amortization, where launch count
  amortizes. FIX 2026-09-21: L4 divergence was a scoping bug from the 6c31136
  L5-hook restructure, NOT a numeric issue — the final la_out dispatch guard
  was renamed if(!l4_done)->if(!l5_done) when the outer scope was added, but
  l4_done (init =l5_done) is the flag recording L4 success. L4-on/L5-off:
  fused path filled o_la on-device, then the host GEMM re-ran over stale
  normed_core and clobbered it (step-1 divergence). Fixed by hoisting l4_done
  to l5_done scope and guarding if(!l5_done && !l4_done) (conjunction needed:
  L5 success skips the block where l4_done=l5_done is assigned). Root cause
  excluded the RNE-converter changes (shared converters verified neutral —
  all 8 combos identical). Stream-gated TP1 -n6: base==L4==L5==HOSTPREP==
  SSM==SSMxL4==TP2-base==TP2-L4 (279 6511 314 279 3046 13); cpu parity +
  gpu-optin + http batch PASS.
- Path to target: single-stream is ~27 tok/s TP1 (fused MLP, was ~24.4
  pre-fusion, was ~26 ceiling pre-DPAS-off):
  ~169 launches/token (~80us submit each: submit ~30%, waitblock +
  transfers most of the rest). 40 tok/s must come from aggregate
  batching (--serve scheduler already M-row batches; cross-request HTTP
  batching is the highest-value lever, C=8/MT=64 probe 9.02 tok/s
  aggregate on quiet box 2026-09-18 (338 ctoks, proves batching; the
  old 36.3 figure is withdrawn with the 8.16 line above). Gate
  re-verified PASS on HEAD (wave+AVX2+fused serving-certified) — the
  C=1 3.27-3.53 tok/s across four independent sweeps is structural
  (MT=8 prefill-dominated), not contention; quoting rule retired.
  Structural options: batched
  multi-launch L0 lists amortizing one sync over many kernels (single
  immediate append+sync measured no faster than UR submit+wait, so the
  win must come from fewer syncs, not a faster submit),
  device-side SSM/attention (host O(pos) attention dominates long ctx).
- Prefix-cache TTFT verdict (2026-09-26, ec86808, 27B TP=2 serve,
  bench_chat.sh --turns 3 --max-tokens 30, parity in
  /var/home/fra/build_tmp/chat_quiet_on.txt; logs serve_27b_18411
  (ON) + serve_27b_18412 (OFF)): single-entry last-completed-request
  snapshot (grid K=16, lowest-4+highest-12, INTFER_NO_PREFIX_CACHE=1
  kill switch, INTFER_LOAD_ORACLE override for non-0.8B serve-load
  oracle). REPRODUCED 6 consecutive sessions. Quiet-box A/B (BOTH
  sides load ~1.1, 27B TP=2, logs serve_27b_18413 ON +
  serve_27b_18412 OFF, parity chat_quiet_on2.txt == chat_quiet_on.txt
  bit-identical): ON t0 TTFT 3.6s/E2E 8.0s, t1 3.9s/23.6s pchit=16
  (prompt=54), t2 4.8s/28.1s pchit=48 (prompt=100). OFF (zero pchit,
  all 3 turns parity OK, rc=0): t0 3.7s/8.1s, t1 4.8s/28.5s, t2
  8.0s/42.9s. Delta t1 -0.9s (-19%), t2 -3.2s (-40%); all 6
  sessions agree within ~0.1-0.2s. The earlier loaded ON run (load
  ~9.2: t0 3.7/8.1, t1 3.9/23.4, t2 4.8/28.0) matches the quiet ON
  to ~0.1s — load inflates absolutes far less than feared on the
  cache-hit path. Quiet-box debt PAID (both sides); numbers above
  are quotable. use=16/48 are
  K=16 grid multiples, use=48 on prompt=100 proves mid-stream snaps
  survive retention (full-stream-only would decline). Sr_old-4 chat
  divergence absorbed by largest-grid-pos<=hit. q.fill=use ->
  P=min(chunk_M,Sr-fill) structurally skips `use` prefill rows.
  Divergence-hunt CLOSED same session: claimed HIT/MISS divergence
  does NOT reproduce on clean tree (install() copies bit-exact,
  10/10 + 12/12 identical outputs; earlier "alternation" was a probe
  history bug). TRACE-PROOF 2026-09-26 (INTFER_PCACHE_DBG, CPU serve,
  identical 17-tok prompts): REQ1 miss records pos=16 + snapshot
  state_len=24 grid_keys=16,24; REQ2 match hit=17 (capped 16) ->
  install use=16 pchit=16 -> snapshot grid_keys=24; installed
  ssm[0]/ssm[-1]/conv[0]/conv[-1] == recorded values on all 18 LA
  layers (BIT-EXACT 18/18), completions byte-identical
  ("The capital of France is **Paris**."). Log: build_tmp/pc_trace3.log. Memory: ~159MB/snap x 16 grid-max ~= 2.5GB host max
  (+KV mirrors), CONSTANT single-entry bound (code-bound: harvest
  caps grid at 16 incl. completion-top; cap unstressed at <=100tok
  prompts; no multi-entry LRU). Serve-load cache-ON (27B, load ~8,
  MT=8 sweep + MT=64 probe): c1/c4/c8 completion + c8 parity + floor
  + bigmt all PASS (c1 1.84 -> c8 1.59 tok/s MT=8 prefill-dominated
  as expected; MT=64 probe 2.45 tok/s over 346 ctoks); solo.oracle
  gate now has INTFER_LOAD_ORACLE override (249d3f8) for non-0.8B
  models — c8.parity byte-identical stays the real determinism gate.
  Cache fires under concurrency (pchit=16 at admit with pq=8).


# guidelines

- remember to commit any meaningful change made such that in case of regressions everything can be bisected.
- remember to not litter /tmp with large tmp files, RAM is precious
