# Intel Arc GPU Optimizations Survey (llama.cpp SYCL, vllm-xpu-kernels, sgl-kernel-xpu)

Date: 2026-10-07, reframed 2026-10-08 against this C++ SYCL tree (2x B580, TP=2 target)
Source: cloned HEAD of all three repos, surveyed source tree

---

## 1. llama.cpp SYCL backend (`ggml/src/ggml-sycl/`)

**Approach: lean on oneDNN/oneMKL for matmul; hand-written subgroup tile kernels for quant path.**

| Optimization | Where | Notes |
|---|---|---|
| XMX/systolic detection | `common.cpp:58` `gpu_has_xmx()` | Probes `sycl::aspect::ext_intel_matrix` |
| XMX hand-written path | commented out (`ggml-sycl.cpp:177`) | `SYCL_USE_XMX` is `#if 0`'d with note "not used for XMX really" |
| oneDNN matmul | `ggml-sycl.cpp:123` → `DnnlGemmWrapper::row_gemm` | Probes `dnnl::matmul` primitive desc; rejects `"ref"` impl. Called for large contiguous matmuls (`ggml-sycl.cpp:3008`, `:3061`) |
| oneMKL BLAS `gemm` | `ggml-sycl.cpp:3115` | `oneapi::mkl::blas::column_major::gemm` fallback |
| Quantized matvec tile kernels | `dmmv.cpp`, `mmq.cpp` | Custom `nd_range` + `reqd_sub_group_size(WARP_SIZE)` kernels; dequant-on-the-fly; `permute_sub_group_by_xor` warp reduction. **No DPAS/XMX** — runs on EU vector ALUs only |
| BF16 | `common.hpp:33` | `sycl::ext::oneapi::bfloat16` |
| Sub-group width = 16 | `common.hpp:226` | Preferred on Intel GPUs |
| SLM | `fattn-tile.cpp`, `fattn-sparse.cpp` | `local_accessor` staging for flash-attention tiles |

**Bottom line:** llama.cpp does **not** hand-write DPAS/XMX. It relies on **oneDNN's internal XMX dispatch** for matmul throughput.

---

## 2. vllm-xpu-kernels (`csrc/xpu/`)

**Approach: CUTLASS-first with architecture-specific (Xe2/Xe3) kernel trees; oneDNN for some quant GEMMs.**

| Optimization | Where |
|---|---|
| CUTLASS DPAS matmul | `XE_DPAS_TT<8, float, ...>` in `grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:104`, `attn/xe_2/paged_decode.hpp:506`, `attn/xe_3/...` |
| CUTLASS BDPAS (Xe3) | `XE_BDPAS_TT` via Xe3 grouped-gemm & attention paths |
| XMX-backed dispatch policies | `MainloopIntelXeXMX16*` (FP8 scaling, block-scaled, fused epilogue with `IntelXeXMX16`) across fp8/mxfp8 paths |
| oneDNN GEMM | `onednn/onednn_matmul.cpp` — w4a4_fp4, w8a16_int4, w8a8_fp8, w4a16_int4 via `dnnl_matmul_*` |
| oneMKL BLAS | via oneDNN wrappers |
| Split-K / TileScheduler | `kernel/xe_tile_scheduler.hpp` — persistent + non-persistent, work-stealing |
| SLM work-stealing | TileScheduler with one `atomicAdd` per tile-chunk |
| Block-2D async prefetch + barrier K-pipeline | `make_block_2d_prefetch` + `prefetch(...)` + `barrier_arrive/wait` in Xe2 kernels |
| Split barriers | `cute::xe_split_barrier` |
| Sub-group SG = 16 | `xe20/w4a16/gemm_xe2.hpp` + all Xe2 kernels |
| Flash decoding (FMHA) | Paged KV, multi-split, per-head-group tiles in `attn/xe_2/`, `attn/xe_3/` |
| FP8/MXFP8/FP4/MXFP4 quantized activation | `quantization/`, `moe/*` |

---

## 3. sgl-kernel-xpu (`src/sycl/`)

**Approach: Similar CUTLASS base as vllm, but no oneDNN; broader kernel surface (MLA sparse, GDN, HiSparse, etc.).**

| Optimization | Where |
|---|---|
| CUTLASS DPAS (Xe2) | `xe20/w4a16/gemm_xe2.hpp` (mainloop: `dpas` referenced explicitly), `xe20/bf16/moe_mainloop.hpp`, `gdn_attn/gemm.hpp` |
| CUTLASS BDPAS (Xe3) | `XE_BDPAS_TT<8, float, ...>` in `xe35/blockwise_moe_mxfp4.cpp:69`, `xe35/fp8_scaled_mm.cpp:114`, `xe35/dsv3_fused_a_gemm.cpp:141` |
| XMX 8×16×16 f16 MAC | `XE_8x16x16_F32F16F16F32_TT` in `xe35/fp8_scaled_mm.cpp:114` |
| Xe2/Xe3 dispatch | Separate `xe20/` vs `xe35/` trees; `torch_extension_sycl.cc:256` selects arch |
| SLM work-stealing tile scheduler | `xe20/w4a16/gemm_xe2.hpp` — SLM-backed atomic counter for tile scheduling |
| Block-2D async prefetch + barrier K-pipeline | `make_block_2d_prefetch` + `prefetch` + `barrier_arrive/wait` in `xe20/w4a16/gemm_xe2.hpp:135-166`, `gdn_attn/gemm.hpp` |
| Split barriers | `cute::xe_split_barrier` in `xe20/w4a16/gemm_xe2.hpp:36` |
| Sub-group shuffle | `sub_group_size = 16` in sampler kernels; `reqd_sub_group_size(32)` in `multimodal_rope.cpp` |
| Flash-attention-v2, MLA (dense + sparse 2-stage), GDN/GatedDeltaNet, HiSparse, MoE, LoRA, MQA logits, HC prefill | `src/sycl/kernels/` tree |
| No oneDNN | Verified: zero `dnnl`/`oneDNN` refs in `src/sycl/` |

---

## 4. Comparison matrix

| Feature | llama.cpp SYCL | vllm-xpu-kernels | sgl-kernel-xpu |
|---|---|---|---|
| DPAS/XMX hand-written | detect-only, unused | CUTLASS `XE_DPAS_TT`/`XE_BDPAS_TT` | CUTLASS `XE_DPAS_TT`/`XE_BDPAS_TT` |
| XMX-backed matmul | oneDNN probe → `DnnlGemmWrapper` | CUTLASS `MainloopIntelXeXMX16*` | CUTLASS `MainloopIntelXeXMX16*` + `XE_8x16x16_*` |
| oneDNN | ✓ (gemm, attention) | ✓ w4a4/w8a8/fp4/fp8 | ✗ (not used) |
| oneMKL BLAS | ✓ column_major::gemm | via oneDNN wrapper | via oneDNN wrapper |
| Sub-group width | 16 | 16 SG | 16 SG (some kernels use 32) |
| SLM work-stealing scheduler | ✗ | ✓ TileScheduler | ✓ SLM counter |
| K-pipeline prefetch+barrier | ✗ | ✓ `make_block_2d_prefetch`+`barrier` | ✓ same |
| Split barriers (`xe_split_barrier`) | ✗ | ✓ | ✓ |
| Async USM / SYCL graphs | ✓ `ext_oneapi_async_memory_alloc` | ✗ | ✗ |
| MoE | ✗ | ✓ | ✓ |
| Flash-attention | sparse fattn | FMHA paged decode/prefill | FMHA v2 + MLA sparse |
| GDN / GatedDeltaNet | ✓ (`gated_delta_net.cpp`) | ✓ | ✓ |
| FP4 / MXFP4 | ✗ | ✗ | ✓ (`mxfp4_blockwise_moe`, etc.) |
| Xe2/Xe3 dispatch | ✗ | ✓ | ✓ |

---

## 5. What this means for intfer (C++ SYCL, 2x B580, TP=2 target)

This tree is C++ SYCL, so unlike the Julia framing this section replaces,
almost everything surveyed is the SAME language -- the question is never
"is there a binding" but "does it have a consumer at TP=2 and does it survive
the measured dead ends in AGENTS.md".

### 5.1 Already in the tree (same technique, own implementation)

| External technique | intfer equivalent | Where |
|---|---|---|
| Hand-written XMX/DPAS GEMM (vllm/sgl `XE_DPAS_TT`) | ESIMD `xmx` DPAS kernels (`dpas48`, `dpas48_2d`/`tiled` E4, `dpas4_row`, `dpas_s8w` lm_head, `dpas_int8`) | `src/gemm/quant_gemm.cpp:1971-2597`, `include/intfer/gemm/quant_gemm.hpp:376-512` |
| Quantized matvec, dequant-on-the-fly, sg=16 (llama `dmmv`/`mmq`) | M=1 INT2/INT4 decode GEMV (`gemm_usm_decode1_vec` family + VERBHALF/VERBMP/VERBSEL/VERBVEC arms) | `src/gemm/quant_gemm.cpp` (`INTFER_DECODE1_*`) |
| Sub-group xor warp reduction (llama `permute_sub_group_by_xor`) | Tried as `INTFER_DECODE1_SUBGROUP` (`reduce_over_group` xor butterfly, no `local_accessor`/barrier) -- QUARANTINED, numerically wrong, cause unlocated | AGENTS.md "QUARANTINED, not fixed" |
| SLM staging for tiled GEMM | Prefill tiled kernels stage `sA`/`sW` in `local_accessor<uint16_t>` (`kBM x kBK`) | `src/gemm/quant_gemm.cpp:1737-1911` |
| Flash attention, FP16 + quantized KV | `flash_attention_forward_f16`, `flash_attention_forward_int8_kv` (groupwise INT8 KV) | `src/attention/flash_attention.cpp`, `include/intfer/attention/flash_attention.hpp` |
| Sub-group width 16 on Intel | `kThr=16`, queried `SUB_GROUP_SIZE==16` inside a kernel | `src/gemm/quant_gemm.cpp:64`, `examples/roofline_probe` |
| BF16/FP16 bit reinterpret | `from_half_bits` scalar twiddle vs `bit_cast<half>` (VERBHALF arm) | `src/gemm/quant_gemm.cpp` (`INTFER_DECODE1_VERBHALF`) |
| Quantized weight family | INT2 G=16/32, INT4 G=64, INT8 G=32, FP16 escape via SpinQuant -> DASH-Q -> RCO | `src/quant/{spinquant,dashq,rco,quant_spec}.cpp` |
| Fused producer->consumer (epilogue fusion spirit) | `run_mlp_fused`, `la_out_fused` via `run_linear_tp_devact` -- **TP=1 ONLY**, dead code at TP=2 | `src/engine/engine.cpp:1352,1484` |

### 5.2 Directly portable (same language, no new dependency)

* **`block_load<uint8_t,128>` for the INT2 weight stream** (llama `ggml-sycl/esimd.hpp`
  `mac_pair`, copied in-tree as the comment at `quant_gemm.cpp:396`). The one genuinely
  new idea from the deleted draft (`quant_gemm.cpp:457-465`): 8 B/lane at sg=16, one
  block load per 32 codes. Named next experiment with a pre-committed kill criterion
  (< 2x closes it, because it also closes llama's `block_load` direction for INT2).
* **SLM work-stealing tile scheduler** (vllm/sgl `TileScheduler`, one `atomicAdd` per
  tile-chunk). Per-device, so TP-compatible; applies to the prefill tiled kernels where
  the device is saturated, NOT to M=1 decode (one output per work-item, nothing to steal).
* **Split-barrier / barrier-free reduction shape.** SYCL here has no `xe_split_barrier`
  without CUTLASS, but the SUBGROUP direction is exactly this shape (xor butterfly, no
  `local_accessor`, no `it.barrier`) -- quarantined, not refuted as a shape. Any retry
  inherits its numerics contract (stream-identity on 27B TP=2) and its cause-unlocated
  status.
* **Multi-split decode attention tiles** (vllm/sgl paged/FMHA decode). Applicable to the
  attention tail only -- layer GEMMs are 99% of launches, so this is low priority at target.
* **oneMKL `gemm` fallback for large contiguous FP16** (llama `ggml-sycl.cpp:3115` shape).
  Zero oneMKL linkage in the tree today (verified: no `dnnl`/`onemkl`/`mkl::` ref in
  `src`/`include`/`examples`), so this is a link-only change, testable on the host side.
  Needs the same stream-identity contract every GEMM change needs.

### 5.3 Portable only with a new dependency or a new quant tier (major work)

* **oneDNN matmul** (llama's ACTUAL XMX path -- `DnnlGemmWrapper`, ref-impl rejection).
  New dependency, new dispatch layer, and a precision contract: this repo has a measured
  matmul-precision divergence history, so a tolerance-based numerics gate comes BEFORE code.
* **CUTLASS `TiledMMA` / `block_2d_copy` / `xe_split_barrier` / fused epilogues**
  (the vllm/sgl substrate under `XE_DPAS_TT`/`XE_BDPAS_TT`). No CUTLASS in tree; porting
  means importing tens of kLOC of tiled-GEMM machinery, not cherry-picking one kernel.
* **FP8 / MXFP8 / FP4 / MXFP4 activation+weight paths** (`quantization/`, `moe/*`, Xe3
  `XE_BDPAS_TT`). No consumer: `quant_spec.cpp` tops out at INT8/FP16 and no gate model
  carries FP8/FP4 tensors. Needs quant + RCO + kernels + a numerics contract, in that order.
* **MoE / MLA / GDN / HiSparse / LoRA / MQA-logits / HC-prefill** (sgl `kernels/` tree).
  No model in any gate uses them; out of scope for the 27B target.
* **Xe2/Xe3 dispatch trees.** Single-arch target (2x B580, Xe2); the Xe3 half has no device
  here. Skip.
* **Async USM alloc** (llama `ext_oneapi_async_memory_alloc`). Alloc-path, NOT launch-path:
  distinct from the retired SYCL-command-graph direction (19.7 us vs 11.9 us, dead).

### 5.4 Surveyed but already measured dead HERE -- do not retry as-is

| Technique (external name) | Local verdict |
|---|---|
| Command-graph capture of decode step (`engine.hpp:12-28` still describes it) | 19.7 us vs 11.9 us plain submit. Dead |
| Enforced XMX/DPAS for all linears | `INTFER_DPAS48` net loss at M=1/128/361. Default OFF is the verdict |
| More outputs per work-item at M=1 | `INTFER_DECODE1_MR`, 2.3x slower. Closed |
| Split-K with `local_accessor` + barrier | `INTFER_DECODE1_SPLITK`, +43.8% kernel. Closed; the barrier is the named cause |
| Fused multi-GEMM via extracted device function | 231 114 ms catastrophe on identical arithmetic. Keep bodies INLINE |
| `block_load`-wider chunks (16 codes/chunk) | 0.25-0.35x. Retired with the llama `block_load` direction for INT2 |
| Native L0 immediate / host-private staging / staged readback / `q.prefetch()` | Same floor / neutral / no-op. Dead |

---

## 6. Conclusion

**The entire performance gap is the GEMM tier, and it is THIS tree's GEMM tier.**
At 27B `--tp 2` decode M=1 the box runs at 9.0% of the 456 GB/s roof
(`scripts/model_byte_roofline.sh`); the FP16 path is saturated (98-123%) and the
INT2 path is the deficit (SHIPPED anchor 39.9 GB/s GATE / 13.9 GB/s DOWN). Nothing
from these three repos fixes that for free:

* llama.cpp's real XMX path is oneDNN -- a new dependency plus a precision contract;
  its hand-written kernels are EU-only, exactly where this tree already is.
* vllm/sgl's XMX path is CUTLASS -- tens of kLOC of tiled-GEMM machinery, not one kernel.
* The in-tree ESIMD DPAS kernels already cover what the tree can reach without either.

So the ordering this survey implies: `block_load<uint8_t,128>` first (same language,
named experiment, kill criterion already written at `quant_gemm.cpp:465`), tile
scheduling and the oneMKL fallback next, oneDNN/CUTLASS/FP8 only with a numerics
contract agreed first. Anything in section 5.4 is not an idea, it is a measurement.

---

## References (file:line within each cloned repo)

| Repo | Key files |
|---|---|
| llama.cpp SYCL | `ggml/src/ggml-sycl/common.cpp:58`, `common.hpp:99,226`, `ggml-sycl.cpp:123,177,3008,3061`, `dmmv.cpp`, `mmq.cpp` |
| vllm-xpu-kernels | `csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:104`, `csrc/xpu/attn/xe_2/paged_decode.hpp:506`, `csrc/xpu/attn/xe_3/paged_decode.hpp:508`, `csrc/xpu/onednn/onednn_matmul.cpp` |
| sgl-kernel-xpu | `src/sycl/xe35/fp8_scaled_mm.cpp:69,114-118`, `src/sycl/xe35/blockwise_moe_mxfp4.cpp:69,75-76`, `src/sycl/kernels/moe/xe20/w4a16/gemm_xe2.hpp`, `src/sycl/kernels/moe/xe20/w8a16/moe_mainloop.hpp:155-166`, `src/sycl/kernels/gdn_attn/gemm.hpp` |
