# sgl-kernel-xpu upstream survey (2026-10-13)

**Repo**: `sgl-project/sgl-kernel-xpu`, default branch `main`. Recursive tree = 696 entries.
All files fetched and read against intfer's actual codebase. **No optimisations ported** — all candidates fail the PARAMOUNT DIRECTIVE's fusion test (which round trips removed? what is the launch delta?).

## Summary of candidates and verdicts

| File | What it does | Why it fails the fusion test / is not applicable |
|------|--------------|--------------------------------------------------|
| `src/sycl/kernels/mla/xe20/collective/xe_mla_mainloop.hpp` (637 lines) | Cutlass-tiled MLA mainloop: `TiledMMAQK`/`TiledMMAPV`, `reduce<1>(FragS, plus{})` cross-SG softmax reduction, `native::exp2`, online-softmax with `softmax_with_rescale()` | intfer has **no softmax attention** at all — its decoder is gated-delta-net SSM (decay → kv-dot → rank-1 update → readout). The MLA rescale-fused-with-GEMM2 pattern (`cute::gemm(mma_pv, tArP, tArV, tArA)`) targets an output accumulator that does not exist in intfer's SSM. Also Cutlass-tiled; intfer is hand-written `nd_range<2>` + `local_accessor`. |
| `src/sycl/kernels/mla/xe20/collective/xe_mla_epilogue.hpp` (447 lines) | Cutlass MLA epilogue: `reduce_A` cross-SG, `write_lse`, softmax divide. | Same — no softmax consumer in intfer. |
| `src/sycl/kernels/mla/xe20/kernel/mla_tile_scheduler.hpp` (178 lines) | `XeMlaIndividualTileScheduler` + `XeMlaReduceSplitKScheduler`: grid = (V tiles, Q tiles, batch*heads), `FastDivmod`, `BlockIdxZ/Y/X`, `num_kv_splits`. | Pure Cutlass dispatch plumbing; no arithmetic. |
| `src/sycl/kernels/flash_attention_v2/collective/fmha_fusion.hpp` (94 lines) | Variable-length attention helpers (`VariableLength`, `apply_variable_length`). | Helper types, not a kernel. |
| `src/sycl/kernels/flash_attention_v2/collective/fmha_relative_bias.hpp` (70 lines) | Sheared bias band computation (`rel_bias_band_cols`, `rel_bias_padded_cols`, `rel_bias_col_origin`). | Helper for flash attention relative bias; intfer has no flash attention. |
| `src/sycl/kernels/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp` (1561 lines) | Flash Attention v2 Xe2 forward kernel (Cutlass). | Flash attention kernel; intfer has no softmax attention. |
| `src/sycl/FusedQKNormRope.cpp` (1937 lines) | JIT elementwise kernels: `fused_q_norm_rope`, `fused_k_norm_rope_flashmla`, `fused_inplace_qknorm_rope`. Uses `local_accessor<float,1> stage[head_dim]`, `reduce_over_group(item.get_group(), sum_of_squares, plus{})`, `rsqrt`. Sub-group-per-row, SLM staging, grid-stride persistent. | intfer's Q/K normaliser is **host-side**: `l2norm` + `qscale` at `main.cpp:3010-3011` (M-row twin at 3464-3467, prefill-chunk at 5662-5664). The device SSM kernel reads the already-normalised `qn_rows`/`kn_rows` (device USM, `quant_gemm.hpp:690-691`; `la_conv_prep` at `quant_gemm.cpp:3698-3709` computes them on device). Porting the in-kernel l2norm would ADD an H2D upload of raw q/k (same byte count as the packed upload) and REMOVE the host `l2norm`+`qscale` kernel which touches NO device memory. Net: no round trip removed, launch count goes UP. The `reduce_over_group` is not portable (splits v-dim across sub-group lanes; intfer has one WI per (head, column) with scalar accumulators). **Fails the fusion test on both counts.** |
| `src/sycl/RMSNorm.cpp` (1051 lines) | Fused RMSNorm variants: `reduce_over_group(item.get_group(), sum_value, plus{})`, `rsqrt`, fused-add residual. | intfer's LA norm is `ssm_norm_silu_to_half` (`quant_gemm.cpp:3484-3520`), already device-side, already fused with silu(z) and FP16 convert. The "fused add_rmsnorm" would target a residual path intfer's LA layers don't have (residual is at MLP/down, host-side). |
| `src/sycl/xe35/dsv3_fused_a_gemm.cpp` (379 lines) | Cutlass `GemmUniversal`, fixed shape: K=7168, N∈{2048,2112}, M∈[1,16], BF16×BF16, `XE_DPAS_TT<8,float,bfloat16_t>` MMA atom, `MainloopXeL1Staged<2>`, epilogue selector (FP32→IntelXeGeneric+XE_STORE_2D, BF16/FP16→IntelXeXMX16+16-bit store). | DeepSeek-V3 specific shapes; intfer's 27B has K=4096 (layer), N=128 (lm_head tile), M=361 (prefill). No shape match. Cutlass not in intfer build (`grep -ri cutlass` returns nothing). The epilogue selector is a dtype-dispatch pattern; intfer's `quant_gemm_f16` already writes FP16 via tiled kernel, ONEMKL arm is the only M≥8 path. |
| `src/sycl/kernels/moe/xe20/w4a16/gemm_xe2.hpp` (494 lines) | W4A16 MoE GEMM: register-level scale folding (`scale_ptr[thr_N * channel_num]` hoisted OUT of k-tile loop as pointer array, line 192), padded-N skip (`kSkipPaddedM`, line 276). | **Scale-row hoist already shipped in intfer**: every `gemm_usm_decode1_*` arm computes `sptr = scale_usm + ng * num_groups_w` once outside the group loop (`quant_gemm.cpp:1041,1141,1213,1297,1399,1780`). Padded-N skip is MoE-specific (expert boundary); intfer has no expert grid. |
| `src/sycl/gdn_attn/gated_delta_rule.hpp` + `chunk_gated_delta_rule.cpp` | Gated delta-net (chunked scan). Already fully surveyed in `upstream_sgl_gdn_chunk_20261012.md` (memory). 2-pass fold = intfer's `INTFER_SSM_FUSE`; in-kernel l2norm fails fusion test; subgroup reduction not portable; iteration shape differs (sgl iterates tokens inside kernel with rollback cache; intfer iterates M rows with state passed in). | **Already falsified in round 3** (commit `20454ae`). |

## The fusion test applied consistently

The PARAMOUNT DIRECTIVE's test (AGENTS.md):  
**"Which round trips does it remove — submit / wait / D2H / H2D — and what is its launch delta? No round-trip removal and no launch reduction is unlikely to pay on this box."**

Every candidate above either:
- Removes **zero** round trips and **adds** launches (FusedQKNormRope, RMSNorm),
- Targets an op (softmax attention) that **does not exist** in intfer's model (MLA, FMHA),
- Is Cutlass-tiled and requires pulling in the entire Cutlass dependency (MLA, FMHA, dsv3),
- Or is a technique **already shipped** in intfer (scale-row hoist in gemm_xe2).

## Concrete porting targets (from the survey's negative result)

The survey rules out upstream ports, but surfaces the **internal fusion surface that IS reachable on the 27B --tp 2 hot path**:

1. **Prefill MLP gate+up+down device chain (TP=2)** — `run_mlp_gateup` already fuses gate+up (one launch, INT2+INT2). The down-projection (`run_linear_tp`) still requires the **full inter** (K = full inter) while gate/up only produce this device's N slice. The silu(gate)*up intermediate must cross the host. **Not reachable at TP=2** (no P2P). This is the structural limit documented in AGENTS.md "BOTH FUSION SEAMS ARE TP=1 ONLY".

2. **SSM norm+silu+la_out GEMM device chain (TP=2)** — `gpu_l4_tail` chains `ssm_forward` → `ssm_norm_silu_to_half` → `run_linear_tp_devact` (la_out GEMM) on device 0, removing the core D2H + host norm + H2D + la_out wait. **But `run_linear_tp_devact` is TP=1-locked** (`engine.cpp:1954: devices_.size() != 1 => return false`). On --tp 2 the LA layers fall back to host la_out. This is a known gap.

3. **Prefill M-row wave launch fusion** — `INTFER_WAVE_LAUNCH_FUSE` already ships this for any wave (one launch per device instead of N). The prefill M-row wave (q/k/v/z/b/a/la_out) already benefits. **Shipped.**

**Conclusion**: The survey's value is confirming that the **already-shipped fusions are the correct ones for intfer's actual architecture**, and the remaining gaps are structural (TP=1 locking, no P2P, model op mismatch), not missing micro-optimisations. The four round survey (llama.cpp × 2, sgl-kernel-xpu × 3) has exhausted the accessible upstream surface.