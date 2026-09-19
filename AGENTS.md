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

- Token generation (decode): **40 tok/s** (per-stream, 0.8B model)
- Prefill: **4000 tok/s**
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
  DPAS48 re-exam at M=361 (interleaved A/B, contended box, 2/2 rounds):
  OFF wins prefill 147.8/146.3 vs 132.7/131.0 (~11%) AND decode
  6.70/6.87 vs 5.13/5.06 (~30%); kernel 318->496ms (one-WI-per-16N-tile
  replicates O(K) fold work nT x), rebuild 500->964ms (DPAS repack),
  copy 12.6->48.4ms (host act-quant). Streams bit-identical. Default-OFF
  stands at ALL shapes (M=1/128/361).
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
  by the wave contract. Saves 1 host wait/layer (24/token). Measured
  load-immune (TP1 fused -n 6): decode nr 828->432 readback groups,
  streams identical TP1/TP2/NOWAVE; cpu parity + http batch + gpu-optin
  PASS. Wall-time A/B deferred (load rose past 2 during gates). Precision:
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


# guidelines

- remember to commit any meaningful change made such that in case of regressions everything can be bisected.
- remember to not litter /tmp with large tmp files, RAM is precious
