# Ninfer Quantization Format & Speed Analysis — Engineering Brief (t1)

> Source: `github.com/Neroued/ninfer` (master), inspected headers/kernels/recipe + README + `docs/performance.md`.
> Goal: identify the exact quantization format(s), the speed-critical mechanisms, and map them to Intel Arc B580 (XMX/DPAS).
> This brief feeds t0 (target model + INT2 spec), t3 (SpinQuant), t4 (DASH-Q), t5 (RCO), t6 (pipeline).

---

## 1. What ninfer actually ships

Ninfer is a from-scratch C++/CUDA engine for **one GPU (RTX 5090, sm_120a / Blackwell)** and a handful of registered Qwen checkpoints. It exposes **two weight profiles** per model:

| Profile | Bits | Kernel class | Notes |
|---|---|---|---|
| `nvfp4` | **4-bit (W4A4)** | `nvfp4_*` (tensor-core `mma_nvfp4_e4m3`) | The speed leader. 5.67× C8/C1 decode gain on Qwen3.6-27B. |
| `groupwise-int` | 4/5/6/8-bit groupwise | `q4_/q5_/q6_/w8_` (IMMA row-split GEMM) | Higher-accuracy fallback; still fast but < NVFP4. |

Plus **FP8 E4M3 (row-scaled)** linears used for some tensors, and **INT8 group-64 / FP8 E4M3 KV cache**.

**Important reality check:** ninfer does **not** invent a PTQ solver. For the `nvfp4` profile it **re-packs an already-quantized source** (`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` and `unsloth/Qwen3.8-27B-NVFP4`). `recipe_nvfp4.py` only fuses/reshapes/packs those pre-quantized `weight_packed` + `weight_scale` tensors and validates them. The quantization *quality* comes from PrismaSCOUT/QuaRot-style NVFP4 methods, not from ninfer's own solver. This is exactly why Stage 1 of our plan (SpinQuant + DASH-Q + RCO) exists — to *produce* the equivalent low-bit weights ourselves for Arc.

---

## 2. The NVFP4 format in detail (the part that is "really fast")

This is the format ninfer consumes/emit and the layout its fastest kernels require.

### 2.1 Logical per-tensor fields (`recipe_nvfp4.py` → `_expected_source_names`)
For each source matrix `[N, K]` (K is the contraction dim, padded to %64; N padded to %128):
- `weight_packed` : `[N, K//2]` **U8** — two E2M1 (NVFP4) values per byte. 4 bits/weight.
- `weight_scale`  : `[N, K//16]` **F8_E4M3** — **one FP8 E4M3 scale per 16 weights** (micro-scaling block of 16).
- `weight_global_scale` : `[1]` **F32** — per-tensor global weight scale.
- `input_global_scale`  : `[1]` **F32** — per-tensor global activation scale.

### 2.2 Physical `.ninfer` layout (`nvfp4_format.cpp :: validate_nvfp4_weight`)
Layout tag: `BlockScaleK16M128x4`, `scale_dtype = FP8_E4M3`, `group_size = 16`, `group = 16`, `ndim = 2`.
- **Code plane**: `elements / 2` bytes (4-bit codes), aligned.
- **Scale plane**: `elements / 16` bytes (1 FP8 scale / 16 weights), offset 256-byte aligned.
- **Tail**: `sizeof(float)` = a `weight_scale_divisor` / `input_scale_divisor` pair (the two global FP32 scales folded in).
- Effective cost: 4 code bits + (8 scale bits / 16 weights) = **4.5 bits/weight avg** for the microscaling overhead.

### 2.3 Decode / scaling math (`nvfp4_codec.cuh :: quantize_nvfp4_k16`)
Per 16-element group of BF16 activation `x`:
```
scale_unencoded = (input_scale_divisor * max_abs(x)) / 6.0      # 6.0 = E2M1 max
scale_fp8       = cvt_float_to_fp8(scale_unencoded, E4M3)
value           = x * input_scale_divisor / decode(scale_fp8)   # dequant-normalize
code            = pack_e2m1x16(value)                            # 4-bit per element
```
Two-level scaling: **global FP32 divisor** (whole-tensor dynamic range) × **per-16 FP8 E4M3 block scale** (local magnitude). The global divisor is what lets the hot MMA loop avoid any per-token scaling — it becomes a single `accumulators * alpha` multiply at the epilogue (`alpha = weight_scale_divisor * input_scale_divisor` essentially).

Weights use the same two-level scheme with `weight_global_scale` + per-16 E4M3 scale.

---

## 3. Why it is fast — ranked mechanisms (RTX 5090 evidence)

Measured (README/`docs/performance.md`), Qwen3.6-27B, INT8-g64 KV + CUDA Graphs + MTP3:

| Mechanism | Evidence / magnitude |
|---|---|
| **W4A4 tensor-core MMA (`mma_nvfp4_e4m3`, 5th-gen Blackwell TC)** | Biggest lever. 4×4-bit ⇒ ~2× FP8, ~8× BF16 math throughput + 8× weight BW. nvfp4 C8 decode **1,146.9 tok/s** vs groupwise-int 535 (2.1× at same accuracy class). |
| **Micro-scaling per-16 E4M3 scale** | Keeps 4-bit usable (4.5 eff bits) so the aggressive W4A4 path is actually accurate enough to ship. |
| **Two global FP32 divisors folded into `alpha`** | Removes per-token scaling from the MMA loop; one epilogue multiply only. |
| **TMA + cp_async + ldmatrix + swizzled shared layout** (`nvfp4_w4a4_tma.cuh`, `nvfp4_w4a4_mma.cuh`) | Async bulk load of codes **and** scales into shared mem, `ldmatrix` to registers; hides global-memory latency; `Stages≤4` ping-pong double-buffering. |
| **Fused kernels** (`linear_add`, `linear_swiglu`, `gdn_*`, `attn_input_proj`) | GEMM fused with bias/add/SiLU/attention; intermediates stay in registers/shared; cuts global round-trips & launch count. |
| **CUDA Graphs on decode** | Fixed-shape decode captured as a graph ⇒ near-zero host launch overhead per step. |
| **MTP3 speculative decoding** (`--spec mtp --draft-tokens 3 --lm-head-draft`) | 46–71% acceptance ⇒ amortizes per-step cost. Primary concurrency multiplier: nvfp4 C8/C1 = **5.67×**, groupwise-int = 2.88×. |
| **Specialized decode/small-T GEMV schedules** (`nvfp4_config.h`) | Per-(N,K,token) occupancy/register crossover tuned on 5090 (e.g. `Nvfp4LinearDecodeProductionSchedule`, small-T schedules T=2..32). |
| **Mixed precision** | NVFP4 linears; FP8 E4M3 for some; INT8-g64 / FP8-E4M3 KV; BF16 for a few sensitive tensors (early attn in_proj, gdn out, linear_attn in_proj a/b). |
| **KV cache compression** (2.2) | INT8 group-64 (per-256-dim, 4 groups of 64) and FP8 E4M3 row-scaled ⇒ 2–4× KV bandwidth, lets context scale to 240k tokens. |

### 2.4 groupwise-int profile (`quantize.py`, `q4_rowsplit_gemm_*` )
- Grouped **symmetric** quantization, scale **rounded to FP16** before coding (host oracle rounds through FP64→FP32→FP16 for exactness; reciprocal precomputed).
- Layout `row-split-k128-v1`: codes `[N, groups, group_size]` packed + per-group FP16 scale; supports 4/5/6/8-bit.
- GEMM via **IMMA row-split** tensor cores (`q4_rowsplit_gemm_mma.cu`). Slower than NVFP4 but a robust higher-accuracy fallback.

### 2.5 KV cache codecs
- `int8_g64_codec.cuh`: signed **INT8, per-token group-64** over head-dim 256 (4 groups). Scale = FP16-RNE(absmax/127), codes use its reciprocal. (Includes `hadamard_d256` — a Hadamard rotation of K/V, i.e. outlier smoothing, same family as SpinQuant/RQUANT.)
- `fp8_e4m3_row_codec.cuh`: **FP8 E4M3, row (whole-256-dim) scaled**, scale FP16 clamped to E4M3 range.

---

## 4. Mapping to Intel Arc B580 (XMX / DPAS / SYCL)

| ninfer technique (NVIDIA) | Intel Arc B580 equivalent | Feasibility for our INT2 target |
|---|---|---|
| NVFP4 W4A4 TC `mma_nvfp4_e4m3` | **XMX DPAS** (`dpas`/`dpasw`) on INT4/INT8; BF16/FP16. Arc Battlemage XMX has no native FP4, so 4-bit and below must be **unpacked** to INT4 lanes. | **INT2** = 4 values/packed byte. Run as INT4 DPAS with 2-bit values zero-extended into 4-bit lanes (uses half the INT4 MACs) **or** custom unpack + INT4 DPAS exploiting 2-bit fit. This is the core kernel work in Stage 2. |
| Per-16 FP8 E4M3 micro-scale | **Manual block dequant**: load per-group FP8/FP16 scale, multiply in register/systolic. XMX has no native microscaling, so fold scale into the epilogue or pre-scale activations. | Mirror ninfer's two-level scheme: global FP32 divisor → single `alpha` multiply; per-group FP8 scale applied at dequant. |
| Dual global FP32 divisor → `alpha` | Same: fold into one accumulator multiply (Fused? via `sycl::mul`/`dpas` accumulate). | Direct port. |
| TMA / cp_async | **SYCL async copies** (`ext::oneapi::experimental::async` / `memcpy` with `sycl::queue`), **2D block loads / XeTile** for subgroup-aligned staging. | Direct; layout must match XMX block shape (subgroup 16/32, 2D blocks). |
| `ldmatrix` | **Subgroup `joint_matrix` load** / explicit `intel::sub_group` block loads + shuffles. | Direct. |
| Fused kernels | **Hand-tuned SYCL fused kernels** (GEMM+bias+SiLU/attention). | Direct. |
| CUDA Graphs | **`sycl::ext::oneapi::experimental::command_graph`** — record decode loop, submit once. | Direct. |
| MTP3 speculative | Vendor-agnostic algorithmic layer (draft heads + verify). | Direct port in Stage 3. |
| INT8-g64 / FP8-E4M3 KV | **Custom SYCL quant/dequant kernels** (arc supports INT8/FP8 math). | Direct; include the Hadamard K/V rotation (outlier smoothing) — relevant to our SpinQuant t3. |

### Key transferable principles for our INT2 design
1. **Two-level scaling is non-negotiable for ultra-low bits**: global FP32 divisor (cheap, one multiply) + fine per-group scale (per-16 for ninfer; we can choose per-16 or per-32 for INT2). This is what keeps 4-bit (and our target INT2) accurate.
2. **Fold global scale out of the inner loop** into a single post-MMA multiply (`alpha`).
3. **Async bulk staging of codes AND scales** is required to feed XMX/DPAS without stalling — layout/swizzle must match the matrix-engine block shape.
4. **Fuse GEMM with elementwise/attention** to keep data on-chip.
5. **Capture the decode graph** (SYCL command graph) to kill host overhead.
   — DISPROVEN 2026-09-13: executable-graph launch 19.7µs vs plain submit
   11.9µs on B580; scaffold removed, lever is fusion (launch-count), not
   graphs. See HISTORY.md (command-graphs dead end).
6. **Speculative decoding (MTP)** is the dominant *throughput* multiplier at concurrency and is hardware-agnostic.
7. **Mixed precision**: keep a BF16/FP16 escape hatch for outlier-sensitive tensors (early attention, gating) — ninfer does exactly this.

---

## 5. Caveats / open questions for downstream tasks
- ninfer delegates PTQ quality to an external NVFP4 recipe; **we must build our own** (SpinQuant t3 + DASH-Q t4 + RCO t5) to hit INT2 with acceptable error. The format *container* (two-level scale + packed codes + global divisor) is the part worth cloning; the *quantization solver* is the part we must write.
- Arc XMX has **no native FP4/NVFP4**; INT2/INT4 requires explicit unpacking and INT4-DPAS. Throughput per MAC is lower than Blackwell's dedicated NVFP4 TC — so our gains come more from BW reduction + fusion + MTP than from a native 4-bit TC.
- Our Stage 1 `int2` is *below* what ninfer ships; expect to fall back to INT4/INT8 for some tensors via RCO (t5) to meet the accuracy budget — exactly ninfer's mixed-precision philosophy.
