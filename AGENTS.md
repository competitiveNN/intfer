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
- Current (2026-09-18, scripts/bench_gpu_ab.sh, DPAS48 default OFF):
  0.8B GPU decode ~13.9 tok/s TP=2, ~24.0 tok/s TP=1 (TP=1 still
  wins ~1.7x: 2x submits/transfers/readbacks outweigh halved kernel
  time at tiny shapes). DPAS48 fast path defaulted OFF (was default
  ON): host GA64 act-quant ~6ms/token dominated, XMX kernel no faster
  than tiled legacy at M=1/128; greedy streams bit-identical either
  way (opt back in with INTFER_DPAS48=1). Decode +16-18%, prefill
  +5% vs DPAS-on. Prefill (361-tok): TP=1 ~147 tok/s (chunk 361),
  TP=2 ~121 tok/s (chunk 361); chunk 361 (whole prompt, one pass)
  still best (fewer layer passes → fewer submits).
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
  2026-09-17: the 2026-09-16 serve-load numbers (C=8 3.86-3.89 tok/s)
  were measured under box contention; clean re-run gives C=1 8.16,
  C=4 15.33, C=8 17.98 tok/s (x2.20), and a batch-full C=8/MT=64
  probe hits 36.3 tok/s aggregate — batching works, short MT=8
  requests drain the batch. Dead-end verdict on OMP stands),
  device-side SSM recurrence M=1
  (2026-09-16, INTFER_SSM_GPU=1, parity-clean incl. step_batch +
  spec-rollback resync: -9% decode, 7.4->6.7 tok/s pinned 3x3
  interleaved. Per-LA-layer attribution: prep ~4us, H2D ~15-130us,
  kernel-submit ~25us, D2H+wait ~150-550us vs replaced host loop
  ~0.1-0.2ms — the blocking wait is structural: host norm needs
  core on-host same-layer. Revisit only with fused device-side
  SSM+norm feeding la_out GEMM from device buffers, or M>>1 where
  the single wait/layer amortizes; M=8 estimate still loses).
- Path to target: single-stream ceiling is ~26 tok/s (TP1, DPAS-off);
  ~169 launches/token (~80us submit each: submit ~30%, waitblock +
  transfers most of the rest). 40 tok/s must come from aggregate
  batching (--serve scheduler already M-row batches; cross-request HTTP
  batching is the highest-value lever). Structural options: native L0
  immediate command lists (bypass UR submit overhead, multi-day),
  device-side SSM/attention (host O(pos) attention dominates long ctx).


# guidelines

- remember to commit any meaningful change made such that in case of regressions everything can be bisected.
- remember to not litter /tmp with large tmp files, RAM is precious
