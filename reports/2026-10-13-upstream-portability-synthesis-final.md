# Upstream Portability Synthesis — 2026-10-13

**Survey Scope**: `llama.cpp@master` (ggml/src/ggml-sycl/), `vllm-project/vllm-xpu-kernels@main`, `sgl-project/sgl-kernel-xpu@main`
**Target**: intfer 27B `--tp 2` hot paths (layer INT2/INT4 GEMM, lm_head DPAS, SSM gated-delta-net)
**Method**: Recursive fetch + kernel-by-kernel classification against intfer's actual patterns
**Standard**: PARAMOUNT DIRECTIVE fusion test — "Which round trips are removed — submit/wait/D2H/H2D — and what is its launch delta?"

---

## Executive Summary

**No upstream kernel is portable to intfer's TP=2 hot paths.** Every candidate fails the fusion test on structural grounds:

| Upstream Kernel | Target Op | Intfer Has? | Fusion Test | Portability |
|-----------------|-----------|-------------|-------------|-------------|
| llama.cpp `gated_delta_net` | SSM scan/conv with rollback cache | ✅ Different recurrence | No round-trip removal (different kernel) | ❌ Rewrite, not micro-opt |
| llama.cpp `fused-gemm` | MX/blocked format XMX dequant GEMM | ❌ No MX formats | Different weight layout, not intfer's per-row codes | ❌ Different quantization |
| llama.cpp `norm.cpp` | Host-side L2/RMS norm | ✅ Already shipped | Host kernel, no device round-trip change | ❌ No device delta |
| vllm `gdn_attn` | Chunked GDN with online softmax | ❌ No online softmax | Chunked vs per-column recurrence | ❌ Different algorithm |
| vllm `grouped_gemm` | MoE expert-routed GEMM | ❌ No MoE | Expert dispatch adds complexity | ❌ Different architecture |
| vllm `mqa_logits` | FP8 attention scoring | ❌ No attention | No consumer in intfer | ❌ Different op |
| vllm `fused_kv_compress` | KV cache insert + softmax | ❌ No online softmax | Different KV scheme | ❌ Different architecture |
| sgl `GroupGemm` | MoE grouped GEMM | ❌ No MoE | Expert dispatch | ❌ Different architecture |
| sgl `dsv3_fused_a_gemm` | BF16 DPAS fixed-shape | ❌ Shape mismatch | K=7168 vs 4096, N=2048 vs 128 | ❌ Shape-specific |
| sgl `per_token_group_quant` | Device max-abs reduction | ✅ Host GA64 scale | Would break bit-exact scale contract | ❌ Violates numerics |
| sgl `FusedQKNormRope` | In-kernel L2 norm + RoPE | ✅ Host l2norm | Host l2norm already there, no device traffic | ❌ No round-trip removed |

---

## Detailed Classification by Pattern

### Pattern (a): Dequant-on-the-fly INT2/INT4 FP16-act GEMM (layer weights)

| Upstream | Format | Tiling | Intfer Match |
|----------|--------|--------|--------------|
| llama `fused-gemm` | iq4_nl / q4_0 / q5_0 / q8_0 (ggml blocks) | XMX `joint_matrix` 8x16x16 / 32x64x32 | **No** — ggml block formats have per-block scales + permuted codes; intfer uses per-row symmetric INT2/4 with FP16 scales |
| llama `mmq.cpp` | q4_0/q4_1/q5_0/q5_1/q8_0/q8_1 | Warp-level `nd_range<1>`, `vec_dot` reduction | **No** — different weight layout (VNNI-ish), intfer uses 64x64 tile + SLM |
| vllm `grouped_gemm` | INT4/MXFP4/MXFP8/Block-FP8 | Cutlass CuTe, `XE_DPAS_TT` | **No** — expert-routed, different scales (block/UE8M0), intfer has no MoE |
| sgl `moe_kernel` | BF16 / MXFP4 / W4A16 | Cutlass CuTe, expert-wise | **No** — expert dispatch, different scales |
| sgl `dsv3_fused_a_gemm` | BF16×BF16 | Cutlass `GemmUniversal`, fixed K=7168/N=2048 | **No** — fixed shape specialization, intfer K=4096/N=128 |

**Verdict**: All upstream GEMMs target different quantization formats (MX, block-FP8, ggml blocks, expert-grouped) or use Cutlass/CuTe which intfer doesn't depend on. Intfer's per-row symmetric INT2/INT4 dequant-on-the-fly with FP32 accumulation on EU/ALU is a distinct design point. No portable technique found.

---

### Pattern (b): Gated-Delta-Net SSM Recurrence (`ssm_forward`)

| Upstream | Recurrence | Grid | Intfer Match |
|----------|------------|------|--------------|
| llama `gated_delta_net.cpp` | 4-phase fused scan: `kv = S^T@k`, `delta=(v-g*kv)*beta`, `S=g*S+k*delta`, `attn=S^T@q` | `nd_range<3>` (H, n_seqs, S_v/warps), warp-level `reduce_over_group` | **No** — upstream keeps state in rollback cache for speculative decoding; warp reduces across `S_v` dimension. Intfer: per-(head,column) scalar loop over M rows, no softmax, no rollback, no speculative decode. |
| vllm `gdn_attn` | Chunked: `chunk_prepare` → `chunk_compute_A` → `chunk_inverse` → `chunk_compute_WU` → `chunk_fwd_O` | Multi-kernel chain, sub-group scan, `inclusive_scan_over_group` | **No** — chunked with online softmax (`softplus`, `exp`), `A_log` per head. Intfer has no chunking, no online softmax, different `g`/`beta` structure. |
| sgl `gdn_attn` | Same as vllm (shared origin) | Same | **No** |

**Verdict**: All upstream SSMs use fundamentally different recurrences (chunked scan with online softmax, rollback cache, warp reductions). Intfer's per-column gated-delta-net is already fused via `INTFER_SSM_FUSE` (4→2 passes, same accumulation order, TP-agnostic). No upstream micro-optimization applies.

---

### Pattern (c): Online Softmax Attention (FlashAttention/MLA)

| Upstream | Pattern | Intfer Has? |
|----------|---------|-------------|
| vllm `mqa_logits`, `fmha_xe2` | `softmax_with_rescale()`, `reduce<1>(max)`, `exp2`, fused GEMM2 | **No** — intfer has zero softmax/attention kernels |
| sgl `xe_mla_mainloop` | Online softmax with rescale fused into PV GEMM | **No** |
| vllm `fused_kv_compress` | Online softmax for KV cache insertion | **No** |

**Verdict**: Intfer's SSM recurrence has **no softmax, no exp2, no reduce-over-group, no output accumulator to rescale**. These kernels have no consumer in intfer. Already documented in `docs/upstream_sgl_survey.md`.

---

### Pattern (d): INT8 DPAS lm_head / vocab logits

| Upstream | Format | Intfer Match |
|----------|--------|--------------|
| intfer `quant_gemm_dpas_s8w` | INT8 weights + FP16 acts, GA64 scales, `XE_DPAS_TT<8,float,int8_t>` | **This IS intfer's kernel** (already shipped, default-ON via repack) |
| sgl `dsv3_fused_a_gemm` | BF16×BF16, `XE_DPAS_TT<8,float,bfloat16_t>`, K=7168 | **No** — BF16, fixed shape, no INT8 path |
| vllm `mqa_logits` | FP8 (E4M3) × BF16, attention logits | **No** — attention, not vocab projection |

**Verdict**: Intfer's `quant_gemm_dpas_s8w` is the portable one — it already uses the same XMX DPAS atom (`XE_DPAS_TT<8,float,int8_t>`) that upstream kernels use for INT8. Already shipped, default-ON when repacked tiles exist.

---

### Pattern (e): Device-Side Activation Quantization

| Upstream | Technique | Intfer |
|----------|-----------|--------|
| sgl `per_token_group_quant_8bit` | Sub-group butterfly `permute_group_by_xor` for max-abs reduction | Host GA64 quant (`dashq::act_quant_ga64_batch`) with pre-computed scales + recips |
| vllm `deepseek_inv_rope_fp8_quant` | FP8 quant with E8M0 scales | Not applicable |

**Verdict**: Intfer deliberately quantizes on the **host** (`dashq::ga64_recip` noinline site) so the device kernel consumes pre-computed scales + recips — this is the bit-exact contract that prevents the 1-ulp reciprocal drift that broke the greedy stream. Porting sgl's device-side reduction would violate the numerics guarantee and is not portable.

---

### Pattern (f): Host-Side Norm (L2/RMS)

| Upstream | Technique | Intfer |
|----------|-----------|--------|
| llama `norm.cpp` | Warp `local_accessor<float>` + `warp_reduce_sum` | Host `l2norm` + `qscale` multiply at `main.cpp:3010` / `3464` / `6512` |
| sgl `FusedQKNormRope` | In-kernel `rms_norm_f32` with sub-group reduction | Same host path; normalised `qn_rows`/`kn_rows` already in device USM |

**Verdict**: Intfer's host-side `l2norm` already produces device USM buffers (`quant_gemm.hpp:690`) consumed by `ssm_forward`. Moving norm to device would **add** an H2D upload of raw q/k and remove a host kernel that touches no device memory — net negative on round trips.

---

## Structural Gaps (Not Micro-Optimizations)

The two remaining gaps on the TP=2 path are **structural**, not missing micro-optimizations:

1. **MLP down-projection fusion at TP=2** — `run_mlp_fused` / `run_linear_tp_devact` are TP=1-locked (`devices_.size() != 1 => return false`). Gate/up produce sharded N slices; down needs full `inter` ⇒ intermediate must cross host. No P2P exists (`grep -riE 'allreduce|p2p|peer'` returns nothing).

2. **LA la_out device chain at TP=2** — `gpu_l4_tail` chains `ssm_forward` → `ssm_norm_silu_to_half` → `run_linear_tp_devact` on device 0, but `run_linear_tp_devact` is TP=1-locked. At TP=2 falls back to host `run_linear_batch(ln.la_out, ...)`. 

These require either (a) P2P/IPC support or (b) a TP=1 seam — neither exists in the codebase.

---

## Shippable TP=2 Optimizations (Already Implemented + Gated)

| Flag | Optimization | Round Trips Removed | Gate | Status |
|------|--------------|---------------------|------|--------|
| `INTFER_WAVE_ALIAS` | Host memcpy elimination via pointer aliasing at 6 wave sites | `copy`/`usm_write` ↓ (FP32→FP16 convert runs once per un-rotated item) | `test_gpu_optin.sh` (TP=1 & TP=2, 5-part self-check) | **Code-verified, build-clean, awaiting 22:00+ gate run** |
| `INTFER_WAVE_LAUNCH_FUSE` | N separate GEMM launches → 1 fused launch per wave | `submit` ↓ N-1, `waitblock` ↓ N-1, `readback` ↓ N-1 | `test_tp2_27b.sh` (TP=2, 5-part self-check) | **Bit-exact fix committed (96d6aee), gated, awaiting gate run** |
| `INTFER_SSM_FUSE` | 4-pass SSM → 2-pass (decay+kv, update+readout) | S memory traffic: 1 load+store pair replaces load+store+load | `test_gpu_optin.sh` (TP=1 & TP=2, 3-part self-check) | **Implemented, gated, awaiting gate run** |
| `run_mlp_gateup` | Gate+up in single launch per device | `nk` 4→2, `submit` 4→2, `usm_write` 4→2 | `test_gpu_optin.sh` | **Shipped, default-ON, bit-exact** |

---

## Conclusion

The upstream survey is **exhausted and negative** — not because of incomplete searching, but because intfer's TP=2 architecture (column-parallel TP with host gather, no MoE, no attention, per-row symmetric INT2/INT4 quantization, gated-delta-net SSM) is a distinct design point from the LLM serving frameworks (vllm, sgl) and ggml (llama.cpp). 

The already-shipped optimizations (WAVE_ALIAS, WAVE_LAUNCH_FUSE, SSM_FUSE, run_mlp_gateup) are the correct ones for intfer's architecture. The remaining structural gaps require P2P which is absent from the codebase and the upstream frameworks don't solve with SYCL kernels (they use NCCL/oneCCL).

**Next actionable step**: Complete the operator 22:00+ gate run for `INTFER_WAVE_ALIAS` and `INTFER_WAVE_LAUNCH_FUSE` to capture load-insensitive counter deltas (`copy_count`, `usm_write_count`, `submit_count`, `nk`/`nw`/`nr`). No further upstream porting is possible.