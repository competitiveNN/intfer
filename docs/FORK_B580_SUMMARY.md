# Torchit1/llama.cpp `arc-b580` Fork - Optimizations for Intel Arc B580

**Fork:** https://github.com/Torchit1/llama.cpp (default branch `arc-b580`, plus `sycl-fa-q4-xmx`, `no-builtins`, `master`)

This is a **PrismML fork** of llama.cpp targeting **Bonsai 2 27B ternary (PTQ1_0)** on **Intel Arc B580 (Xe2 / "Battlemage", BMG-G21, 12 GB)** at **128K context** via the **SYCL (Intel oneAPI) backend**. It starts from PrismML's `bonsai-combo` branch and tracks mainline llama.cpp.

**Headline results** (B580 12 GB, Linux, oneAPI 2025.3, Level Zero 26.35, q4_0 KV cache, MTP head + 4 drafts, n-gram drafts, thinking off, greedy/temperature 0):

| Workload | SYCL (this fork) | Vulkan (this fork) |
|---|---|---|
| Fresh code answer @128K | 90 t/s | 53 t/s |
| Rename symbol in pasted code | 365 t/s | 194 t/s |
| Small edit to pasted code | 258 t/s | 135 t/s |
| Plain generation, no speculation | 42 t/s | 32 t/s |
| 48K code in context: new / edit | 63 / 34 t/s | - |
| ~115K in context: new / edit | 44 / 27 t/s | - |

Authoritative doc: `docs/bonsai-arc-b580.md` in the `arc-b580` branch.

---

## Table of Contents

1. [Software Stack and Requirements](#1-software-stack-and-requirements)
2. [Ternary Weights on XMX](#2-ternary-weights-on-xmx-matrix-units)
3. [Decode Attention on XMX from q4_0 KV Cache](#3-decode-attention-on-xmx-from-q4_0-kv-cache)
4. [Long-Prompt Attention](#4-long-prompt-attention)
5. [Gated Delta-Net Kernels](#5-gated-delta-net-kernels)
6. [Kernel Fusions (v0.8)](#6-kernel-fusions-v08)
7. [Speculative Decoding Overhaul (v0.10)](#7-speculative-decoding-overhaul-v010)
8. [v0.10.1 Incremental (Oct 10, 2026)](#8-v0101-incremental-oct-10-2026)
9. [Memory Management for 128K on 12 GB](#9-memory-management-for-128k-on-12-gb)
10. [Complete Switch Reference](#10-complete-switch-reference)
11. [Vulkan Backend](#11-vulkan-backend)
12. [Other Models: Bonsai 8B and Gemma 4 12B](#12-other-models)
13. [Comparative Benchmarks](#13-comparative-benchmarks)
14. [Quality Verification](#14-quality-verification)
15. [Build and Run](#15-build-and-run)
16. [Caveats and Gotchas](#16-caveats-and-gotchas)
17. [Credits](#17-credits)
18. [2×B580 / Tensor-Parallel Analysis](#2xb580--tensor-parallel-analysis)

---

## 1. Software Stack and Requirements

| Component | Version |
|---|---|
| Intel oneAPI Base Toolkit | 2025.3+ (icx/icpx) |
| Level Zero GPU driver | 26.35 |
| Vulkan driver | Mesa ANV 26.2 |
| Meson / cmake | cmake 3.x |

Vendored dependency: `libxsmm/TernSYCL` (BSD 3-Clause) at `ggml/src/ggml-sycl/ternsycl` provides the int8 x int2 DPAS kernels.

The XMX path requires an **Xe2 or newer GPU** (Arc B-series, Lunar Lake, Panther Lake). On other Intel GPUs it disables itself with a warning.

---

## 2. Ternary Weights on XMX Matrix Units

**Switch:** `GGML_SYCL_PTQ1_T2=all` (or `ffn`). On by default in the release scripts. Off = upstream path.

### What it does
- At model load, **every PTQ1_0 weight tensor is repacked in place** from the base-3 ternary packing to a **2-bit layout**.
- Matrices then run through **int8 x int2 DPAS kernels** from TernSYCL, hitting the XMX (Xe Matrix Extensions) units instead of the ALUs.
- The stock PTQ1_0 base-3 packing must be decoded on the ALUs first, which made the multi-token **verify** step of speculative decoding compute-bound. The 2-bit layout removes that decode.
- Activations are quantized to **int8 with round-to-nearest** instead of TernSYCL's truncation. This alone cut KL divergence 5x.
- Costs **~31% more weight memory** for the repacked layout.

### Derived sub-optimizations
- `GGML_SYCL_PTQ1_T2_GEMM_OFF` - disables the XMX GEMM path only.
- `GGML_SYCL_PTQ1_MULTI` / `GGML_SYCL_PTQ1_MULTI_NCOLS` / `GGML_SYCL_PTQ1_MULTI_NCOLS_DEC_OFF` - the "multi-column" TernSYCL kernels for wider batches.
- Multi-column layout: multiple weight columns interleaved so one launch covers several mat-vecs.
- `GGML_SYCL_PTQ1_GLU1` - fused gate/up projection variant.
- `GGML_SYCL_PTQ1_PAIRS` - pairing of mat-vecs sharing an input into one launch.

### Prompt path (oneDNN int8 GEMM)
For prompt batches of **1024+ tokens**, the 2-bit weights are converted to int8 and multiplied with **oneDNN's int8 GEMM**, which beats TernSYCL's GEMM at that size. A 48K-token code prompt reads at **~800 t/s**.

**Switch:** `GGML_SYCL_T2_W8A8_MIN=1024` (0 = off). Measured ~10% faster long prompts.

---

## 3. Decode Attention on XMX from q4_0 KV Cache

**Switches:** `GGML_SYCL_FA_DEC_DPAS=1` (opt-in, Xe2-only), `GGML_SYCL_FA_DEC_GRF=256` (v0.10.1, default), `GGML_SYCL_FA_DEC_OFF` (disable).

Developed on the `sycl-fa-q4-xmx` branch; the three commits there are:
- `f72a7ce4` "sycl: q4_0 flash attention for decode on Xe2 XMX (s8 x s4 DPAS)"
- `b322be71` "tests: FLASH_ATTN_EXT q4_0 K/V cases for decode and small batches"
- `5e408ae1` "sycl: fattn-dpas prefetches the next phase's K/V rows into L1"

### Mechanics
- **Q.K** uses an **int8 x int4 DPAS builtin that IGC exposes on Xe2**: `intel_sub_group_i8_i4_matrix_mad_k32`. This is *not* in the public extension list; the fork calls it directly. The raw q4_0 cache bytes are the B operand, so no dequantization pass is needed.
- **P.V** runs on **f16 DPAS** (the probabilities are softmaxed floats anyway).
- **One kernel serves 1-32 query tokens**, head sizes 128 and 256, up to 32 query heads per KV head (GQA-aware). So MTP and n-gram verify batches never convert the whole cache to f16.
- Without it, a q4_0 cache kernel serves only 1-4 tokens.
- **Generation at 48K context: +14-16%.** It also removed an out-of-VRAM crash near 128K.

### Prefetch
`GGML_SYCL_FA_DEC_PF` prefetches the **next attention tile's K/V rows into L1**. With MTP: **+4% at 48K, +7.5% at 115K** context.

### 256-register file (v0.10.1)
`GGML_SYCL_FA_DEC_GRF=256` (default) runs the DPAS decode kernel with a **256-register file for 3+ query tokens**. The 3+ token launch is one wave of 2 work-groups x 16 subgroups per Xe core = 32 threads, exactly what the 256-register mode allows, so it costs **no occupancy** and removes the spills. 1-2 tokens keep the default register file (their wave puts 4 work-groups on a core).

Per-layer us at 128K (real cache layout), 5 / 8 / 12 / 16 tokens: `1073 / 1434 / 2920 / 2939` -> `1004 / 1240 / 2465 / 2445`. 1 token unchanged. Kernel 6-17% faster at 128K. Whole model at 48K +2.0%, 115K +4.4%, output md5 identical.

### Transposed 2D block loads (opt-in, off by default)
`GGML_SYCL_FA_DEC_B2D=1` uses `xe2dp::ld2d_t32_8x16` (`lsc_load_block2d d32.8x16tn`): one message gives each lane 8 dwords of its own key row, replacing the per-lane dword gathers in Q.K (5 messages per 16 keys instead of 40) and P.V (1 instead of 5). Same bytes, same math, 89/89 FLASH_ATTN_EXT cases pass. Engages only on **64-byte aligned K/V surfaces** (logged once).

**Not enabled by default**: on its own it is *slower* at 128K due to register spills (5 tokens 1069 -> 1220 us, 16 tokens 2945 -> 3610 us). It becomes useful once combined with the 256-register file.

---

## 4. Long-Prompt Attention

### Chunked-attention softmax fix
The chunked attention path gave **each query row to one work-item**. Now a **work-group per row** (parallel reduction across the row). **2.5x faster generation at ~115K context.**

### oneDNN cap for the prompt path
`GGML_SYCL_FA_ONEDNN_MAX_KV=98304`. The fast prompt-attention path converts the whole KV cache to f16 (~4 KB/token of context). At 128K with ~0.5 GB spare VRAM, a prompt of ~119K tokens ran out of memory and killed the server. Above the cap, prompt attention uses the **chunked path** instead: slower (~330 t/s prompt reading at 120K) but safe. Shorter prompts unaffected.

---

## 5. Gated Delta-Net Kernels

Bonsai uses a hybrid attention + gated delta-net (linear attention) architecture with 48 delta-net layers.

- **Blocked kernel** (`GGML_SYCL_GDN_BLOCKED_OFF` to disable) for the delta-net recurrence.
- **Fused state writes** - single kernel writes the recurrent state.
- Single-token decode fusions:
  - same-input mat-vecs in one launch
  - narrow concat
  - fused gate/up
- **Delta-net conv chain** (`GGML_SYCL_CONV_CHAIN`): concat, conv-state write, conv, SiLU, L2 norm as **one kernel**.
- **Delta-net gate block** (`GGML_SYCL_DN_GATES`): its gate block runs as one kernel.
- **K/V-only MTP prompt pass** also benefits here (see section 7).

---

## 6. Kernel Fusions (v0.8)

| Fusion | Effect |
|---|---|
| Hadamard rotate + quantize activation once, reused by every mat-vec reading it (`GGML_SYCL_PTQ1_T2_QCACHE`) | Removes redundant activation prep |
| Residual add + RMS norm folded into the rotation (`GGML_SYCL_PTQ1_T2_NORMQ`, `GGML_SYCL_NORM_ROT_FAR`) | Fewer kernels |
| FFN SwiGLU folded into the rotation (`GGML_SYCL_GLU_ROT`) | Fewer kernels |
| Delta-net pre-norm folded into the rotation | Fewer kernels |
| Delta-net conv chain as a single kernel (`GGML_SYCL_CONV_CHAIN`) | Removes launch overhead |
| Delta-net gate block as a single kernel (`GGML_SYCL_DN_GATES`) | Removes launch overhead |
| Q/K/V mat-vecs in one launch (`GGML_SYCL_T2_MULTI`) | Fewer launches |
| gate/up mat-vecs in one launch (`GGML_SYCL_PTQ1_PAIRS`) | Fewer launches |
| Small verify batches (<=32 rows) use a smaller TernSYCL tile (`GGML_SYCL_PTQ1_NCOLS_DEC_OFF` off) | Better occupancy |
| Decode attention prefetches the next tile's rows (`GGML_SYCL_FA_DEC_PF`) | Hides latency |

### Impact
- Plain generation (no speculation): short prompt ~41 -> 46 t/s, 48K context ~36 -> 40, 115K ~31.5 -> 35.
- With MTP, the attention prefetch alone is +4% at 48K and +7.5% at 115K.
- Covered by `test-backend-ops -o ARCLAB_*`; KL divergence unchanged within noise.

---

## 7. Speculative Decoding Overhaul (v0.10)

All on by default. Greedy outputs byte-identical to v0.8.

### 7.1 Tuned verify tiles (`GGML_SYCL_PTQ1_T2_TUNED`)
A **sweep of 9,140 launch configurations** of the TernSYCL small-batch kernel at the model's actual shapes found faster **bit-identical** tiles for the 5-token MTP verify:

| Kernel | New time |
|---|---|
| FFN down | 0.78x |
| FFN gate+up | 0.90x |
| Output head | 0.88x |
| Attention q+k+v | 0.92x |

**B580 only by default** (tuned for its 20 Xe cores). `=1` forces on other GPUs. CITATION(techpowerup arc-b580.c4244 -- the B580 is 20 Xe cores / 160 vector engines; locally saved as `docs/Intel Arc B580 Specs _ TechPowerUp GPU Database.html`, cross-checked against Intel ARK sku/241598)

### 7.2 Delta-net rollback by replay (`LLAMA_GDN_REPLAY`)
Instead of keeping a full recurrent state per verified token so it can roll back rejected drafts, the **48 delta-net layers keep one state plus each token's small update terms**, and re-apply the accepted ones on the next step.
- Fewer state writes per verify.
- **432 MiB less VRAM at 128K.**
- Independent implementation of the technique described by [llamADALL](https://github.com/straightdot/llamADALL) (no code used).

### 7.3 One-graph MTP drafts (`LLAMA_MTP_CHAIN`, `LLAMA_MTP_CHAIN_GPU_EMBD`, `LLAMA_MTP_CHAIN_TIGHT`)
All 4 drafts of a step run in **one graph**, each step taking the previous step's **in-graph argmax**, with a **device copy of the draft vocabulary's embedding rows**. The draft ids are read back once.
- Needs a GGUF with a **trimmed draft head** (see below); `--spec-draft-n-max 4`.
- Independent implementation of the technique described by llamADALL.
- Sub-switches (bitmask, default 31 from v0.10.1):
  - **bit 8**: output-row index rides at the end of the position input (one upload, sliced on the device); V index input dropped from the graph when V is stored with the K indices. Saves 1-2 split inputs (each a sync + blocking copy) per chain decode.
  - **bit 16**: **all draft ids in one tensor, read back with one copy**. Each small device->host read on this SYCL/Level Zero stack costs **~0.2 ms even for 4 bytes** (measured: reading only step 0's id cut the draft call from 6.07 to 5.16 ms). The ids are concatenated into one I32 [n] tensor as the graph's only output.

### 7.4 K/V-only MTP prompt pass (`LLAMA_MTP_KV_ONLY`)
Over a prompt, the MTP head computes **only the K/V rows it keeps** - no attention, FFN or output head for those rows. Long prompts read **~5% faster** with MTP on. Independent implementation of a technique described by [GRIMOIRE](https://github.com/doopeworld/GRIMOIRE).

### 7.5 Trimmed draft head
An MTP GGUF can carry `blk.<n>.nextn.draft_head` (top-K most frequent tokens), which speeds up the draft step **~5% on the B580**. `LLAMA_MTP_DRAFT_HEAD_OFF=1` ignores it.

Tool to add one:
```sh
python3 scripts/bonsai/add_draft_head.py Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf \
        bonsai-dh32k.gguf scripts/bonsai/freq_public.json 32768
```
Token frequencies come from public text only (llama.cpp and Python stdlib source, FineWeb-Edu, Alpaca, and the model's own answers to public prompts).

### 7.6 Adaptive n-gram draft length (`LLAMA_ARG_SPEC_NGRAM_ADAPT`)
n-gram draft length **starts at 64, doubles on a full accept, halves on a rejection**.

### 7.7 SYCL sync trimming (`LLAMA_SYNC_TRIM`)
Skips redundant context syncs and keeps the KV rotation matrices on the device. Default 3.

### 7.8 Level Zero adapter default
The SYCL backend **defaults to the legacy Level Zero adapter**, which is slightly faster here. `SYCL_UR_USE_LEVEL_ZERO_V2=1` keeps the V2 adapter.

### v0.10 impact on B580 (vs v0.8)
- MTP on mtp-lean: 71.9 -> 78.3 t/s (+9%)
- With trimmed 32K-token draft head: 73.8 -> 80.4 (+9%)
- Live MTP + n-gram at 128K context: short 69.9 -> 75.9 (+8.6%), 48K 73.7 -> 78.7 (+6.8%), 115K 78.8 -> 83.5 (+6.0%)

---

## 8. v0.10.1 Incremental (Oct 10, 2026)

All on by default. Greedy outputs byte-identical to v0.10.

1. **One read-back for the draft chain** - `LLAMA_MTP_CHAIN_TIGHT` bit 16. See section 7.3.
2. **256-register decode attention** - `GGML_SYCL_FA_DEC_GRF=256`. See section 3.
3. **Three general fixes from PrismML's llama.cpp fork** (authorship preserved):
   - **#311 `kv-cache: limit seq_rm to the used cell range`** - `seq_rm` walked all cells of the cache on every call; cells outside `[used_min, used_max_p1)` are empty, so skip them. Showed up in the host profile with a large context and speculative decoding (seq_rm after rejected drafts).
   - **#312 `server: reuse the buffers of evicted prompt checkpoints`** - `create_checkpoint()` evicted old checkpoints then emplaced a new one whose data vectors were resized from empty, so every checkpoint (~150 MiB for a hybrid model) was zero-filled and page-faulted again. Now the last evicted checkpoint's buffers are handed to the next one. Saves ~18 ms per turn in a growing multi-turn conversation (645 -> 627 ms). Decode speed and outputs unchanged.
   - **#313 `common: take the top-k straight from the logits when top-k starts the chain`** - `common_sampler_sample()` built a token array for the whole vocabulary (set_logits) and let the top-k sampler partial-sort it. When every sampler before top-k does nothing with the current settings (penalties, dry and top-n-sigma off, no logit bias, no mirostat, no backend sampling), no grammar applied first and the reasoning budget not forcing, select the k largest logits directly. The selection repeats the heap steps of libstdc++ `std::partial_sort`, so entries and the order of equal logits are identical. Checked against `std::partial_sort` on 120,000 random arrays (ties, NaN, inf) and on server outputs. **+1.9%** with MTP n-max 2 (greedy benchmark), +1.85% with the Hermes sampling settings.

### v0.10.1 impact on B580 (vs v0.10)
| Config | v0.10 | v0.10.1 | Delta |
|---|---|---|---|
| MTP with trimmed 32K draft head | 80.4 | 84.7 | +5.4% |
| MTP on mtp-lean | 78.3 | 80.5 | +2.8% |
| Live MTP + n-gram @128K, short | 75.9 | 78.6 | +3.6% |
| Live MTP + n-gram @48K | 78.7 | 82.4 | +4.7% |
| Live MTP + n-gram @115K | 83.5 | 88.9 | +6.5% |

Against v0.8 that is **+12 to +15%**.

Also in v0.10.1: two Windows build fixes (the sync trace's caller lookup using cxxabi/dlfcn/execinfo is Linux-only; the Windows SDK defines `small` as a macro so the trace's counters were renamed).

---

## 9. Memory Management for 128K on 12 GB

| Technique | Saving |
|---|---|
| MTP draft context takes its own KV cache type (`-ctkd/-ctvd q4_0`) | Reduces draft-cache footprint |
| Smaller physical batch for the draft context (`LLAMA_ARG_SPEC_DRAFT_UBATCH=512`) | Smaller compute buffers |
| Main prompt batch at `-ub 1024` (instead of 2048) | Fits alongside draft context |
| Delta-net replay rollback (section 7.2) | **432 MiB at 128K** |
| `GGML_SYCL_PTQ1_T2=ffn` (FFN weights only on XMX) | ~380 MiB weight memory freed |
| 2-bit weight repack costs **+31%** on the weight side | (trade-off, see section 2) |

`GGML_SYCL_PTQ1_T2=ffn` puts only the feed-forward weights on XMX: about 380 MiB less weight memory, enough for `-ub 2048` at 128K, and most of the speed (fresh 80, rename 314, edit 213).

---

## 10. Complete Switch Reference

All optimizations are on by default **except the XMX path**, which needs `GGML_SYCL_PTQ1_T2`.

### XMX weight path
| Switch | Effect |
|---|---|
| `GGML_SYCL_PTQ1_T2=all` / `=ffn` / unset | Enable all / FFN-only / disable XMX ternary |
| `GGML_SYCL_PTQ1_T2_GEMM_OFF` | Disable XMX GEMM |
| `GGML_SYCL_PTQ1_MULTI=0` | Disable multi-column kernels |
| `GGML_SYCL_PTQ1_MULTI_NCOLS=0` | Disable multi-column col count |
| `GGML_SYCL_PTQ1_NCOLS_DEC_OFF` | Disable small-batch decode tile |
| `GGML_SYCL_PTQ1_GLU1=0` | Disable fused gate/up |
| `GGML_SYCL_PTQ1_PAIRS=0` | Disable paired mat-vecs |
| `GGML_SYCL_PTQ1_T2_QCACHE=0` | Disable activation quantize cache |
| `GGML_SYCL_PTQ1_T2_FWHTQ=0` | Disable fast Walsh-Hadamard quantize |
| `GGML_SYCL_PTQ1_T2_NORMQ=0` | Disable norm folded into rotation |
| `GGML_SYCL_PTQ1_T2_TUNED=0` / `=1` | B580 verify tiles; =1 forces on other GPUs |
| `GGML_SYCL_T2_W8A8_MIN=N` | Prompt batches of N+ tokens via oneDNN int8 (0 = off, default 1024) |

### Attention
| Switch | Effect |
|---|---|
| `GGML_SYCL_FA_DEC_DPAS=1` | XMX decode attention from q4_0 cache (Xe2, opt-in) |
| `GGML_SYCL_FA_DEC_OFF` | Disable decode attention path |
| `GGML_SYCL_FA_DEC_GRF=256` / `=0` | 256-register file for 3+ query tokens |
| `GGML_SYCL_FA_DEC_B2D=1` | Transposed 2D block loads (opt-in) |
| `GGML_SYCL_FA_DEC_PF=0` | Disable next-tile prefetch |
| `GGML_SYCL_FA_ONEDNN_MAX_KV=98304` | Chunked-attention cap for prompt path |

### Delta-net
| Switch | Effect |
|---|---|
| `GGML_SYCL_GDN_BLOCKED_OFF` | Disable blocked kernel |
| `GGML_SYCL_CONV_CHAIN=0` | Disable conv-chain fusion |
| `GGML_SYCL_DN_GATES=0` | Disable gate-block fusion |
| `LLAMA_GDN_REPLAY=0` | Disable delta-net replay rollback |

### Fusion
| Switch | Effect |
|---|---|
| `GGML_SYCL_GLU_FUSE_OFF` | Disable GLU fusions |
| `GGML_SYCL_GLU_ROT=0` | Disable FFN SwiGLU in rotation |
| `GGML_SYCL_NORM_ROT_FAR=0` | Disable norm-in-rotation |
| `GGML_SYCL_T2_MULTI=0` | Disable q/k/v merged launches |

### MTP / speculation
| Switch | Effect |
|---|---|
| `LLAMA_MTP_KV_ONLY=0` | Disable K/V-only MTP prompt pass |
| `LLAMA_MTP_CHAIN=0` / `=1` | One-graph draft chain (auto for trimmed head, =1 forces) |
| `LLAMA_MTP_CHAIN_GPU_EMBD=0` | Device copy of draft-vocab embeddings |
| `LLAMA_MTP_CHAIN_TIGHT=31` | Bitmask draft graph layout (bit 8, 16) |
| `LLAMA_MTP_DRAFT_HEAD_OFF=1` | Ignore trimmed draft head in GGUF |
| `LLAMA_ARG_SPEC_NGRAM_ADAPT=0` | Adaptive n-gram draft length |
| `LLAMA_ARG_SPEC_DRAFT_UBATCH=512` | Draft context physical batch |
| `LLAMA_MTP_DRAFT_TIMING=1` | Diagnostics: draft timing |

### Host / sync
| Switch | Effect |
|---|---|
| `LLAMA_SYNC_TRIM=3` | Skip redundant syncs, keep KV rot mats on device |
| `GGML_SYCL_TOPK_OLD=1` | Old top-k path |
| `GGML_SYCL_OP_TIMING=1` | Diagnostics: per-op timing |
| `GGML_SYCL_HOST_TIMING=1` | Diagnostics: host timing |
| `GGML_SYCL_SYNC_TRACE=1` | Diagnostics: sync trace |
| `SYCL_UR_USE_LEVEL_ZERO_V2=1` | Use the V2 Level Zero adapter |

### Vulkan
| Switch | Effect |
|---|---|
| `GGML_VK_PTQ1_MC_OFF=1` | Disable multi-column PTQ1_0 mat-vec |

---

## 11. Vulkan Backend

The Vulkan backend received most of the same kernel work but runs on any Vulkan driver (tested on Mesa ANV 26.2). **SYCL is clearly faster on this card** - roughly 1.7x on fresh code, 1.9x on rename, 1.3x on plain generation.

Build:
```sh
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j --target llama-server llama-bench llama-cli
```
At 128K this left about 1.2 GB of VRAM free on the B580. Vulkan uses `--spec-draft-n-max 3` (not 4) and `-ub 2048`.

---

## 12. Other Models

### Ternary Bonsai 8B (`Ternary-Bonsai-8B-PQ2_0.gguf`, 2 GB)
Also runs the XMX path. Qwen3-based, so **no MTP head** - use n-gram drafts, or the 1.7B model (`Ternary-Bonsai-1.7B-PQ2_0.gguf`) as a draft model.

B580, 32K context, temperature 0:

| Drafts | new code | rename | edit |
|---|---|---|---|
| none | 100 t/s | 75 | 80 |
| n-gram | 102 | 506 | 658 |
| 1.7B drafter + n-gram | 110 | 471 | 608 |

XMX path: **134 t/s** plain generation (118 without), **765 vs 408 t/s** on 8-token batches, identical perplexity.

### Gemma 4 12B QAT q4_0
The same XMX work covers Gemma's shapes: int8 x int4 decode attention handles its global (head 512) and sliding (head 256) layers, and a q4_0 small-batch GEMM on XMX streams the weights once for verify batches of 5-256 tokens.

Extra switches: `GGML_SYCL_Q4_0_DPAS=1` (q4_0 verify batches on XMX), `GGML_SYCL_MMVQ_CHUNK_MAX=128` (anything up to 128 tokens as chunked mat-vecs, not dequantize-everything).

Needs a one-time conversion of Google's assistant draft model to GGUF.

| Gemma 4 12B QAT q4_0, B580, 128K | new code | rename | edit | new/edit @48K |
|---|---|---|---|---|
| n-gram drafts only | 40 t/s | 164 | 64 | - |
| + assistant (3 drafts) | 92 | 186 | 136 | 47 / 44 |
| + XMX attention and q4_0 GEMM | **102** | **275** | **168** | **65 / 57** |

Gemma writes slightly better code (HumanEval+ 152 vs 141 of 164) and is faster on new code; Bonsai is faster on edits (~255 vs 168 t/s) and renames (~370 vs 275).

### Tool calling (BFCL v3, first 100 of each category)

| model | simple | multiple | parallel | no call needed |
|---|---|---|---|---|
| Ternary Bonsai 2 27B | 98 | 95 | 92 | 79 |
| Ternary Bonsai 8B | 96 | 92 | 83 | 92 |
| Gemma 4 12B (QAT q4_0) | 96 | 92 | 84 | 83 |

---

## 13. Comparative Benchmarks

### Against PrismML's own fork (basic SYCL support for PTQ1_0/PQ2_0, 24 Sep)
Same B580, same public mtp-lean model, q4_0 KV cache, flash attention, 32K context, MTP depth 3, no n-gram drafts, identical server flags:

| Metric | PrismML fork | This fork | Speedup |
|---|---|---|---|
| Prompt reading (`llama-bench pp512`) | 207 t/s | 895 t/s | **4.3x** |
| Plain generation (`tg128`) | 20.6 t/s | 42.3 t/s | **2.1x** |
| Generation with 32K in context | 8.1 t/s | 35.8 t/s | **4.4x** |
| Chat, fresh code, MTP drafts | 37.6 t/s | 88.4 t/s | **2.35x** |
| Chat, fresh code, no drafts | 20.3 t/s | 40.7 t/s | **2.0x** |

Draft acceptance was the same on both (166/210 vs 165/213), so the gap is the kernels: the XMX units for the weights and for attention.

### Against the first working SYCL port of this model (32K context)
| Metric | First port | This branch |
|---|---|---|
| Fresh code | 55.8 | 85.5 t/s |
| Rename | 217.5 | 368.8 t/s |
| Edit | 143.0 | 255.1 t/s |
| Plain generation | 31.4 | 40.7 t/s |

Quality: **KL divergence against the reference logits 0.00022** (99.2% same top token); the plain PTQ1_0 kernels score 0.0003. Greedy outputs on the test prompts are byte-identical to the plain PTQ1_0 kernels.

### Real-world chat UI
- ~1000-token new answer at ~65 t/s
- whole edited file at ~205 t/s (temperature 0, 128K context)
- With sampling (temperature 0.6): ~75-80 t/s new code
- New prose drafts worse than code

### Long-context recall (needle in a haystack)
A passphrase hidden in *Pride and Prejudice* at 10%, 50% and 90% depth: **exact at 32K, 64K and 120K, 9 of 9.**

### As a coding agent (Claude Code via the Anthropic Messages API)
Asked for a Snake game with unit-tested logic in Node, it wrote the game and a test suite, ran the tests, fixed the one failure and re-ran to a clean pass in **~5.5 minutes**.

Things that helped:
- A `CLAUDE.md` in the project with a few rules (build only what was asked, simplest code that works, always run tests, report tool failures honestly, list commands run and their results at the end).
- Patience on the first turn: Claude Code's system prompt and tool list are ~20-27K tokens, ~30 s to read. Later turns reuse the cache.
- One user at a time (`-np 1`).

Note that Gemma 4 12B is faster but skipped steps and reported tests as passed without running them, so Bonsai is the recommended agent model on this card.

### Thinking on/off
Thinking is off in the default command because it is faster for coding. For maths and logic puzzles turn it on with `--chat-template-kwargs '{"enable_thinking":true}'`: in one tester's run a timer puzzle that the model brute-forced for ~6,000 tokens with thinking off was solved first try in ~1,000 tokens with it on (42 vs 55 t/s per token, but about 4x sooner overall).

---

## 14. Quality Verification

| Check | Result |
|---|---|
| KL divergence vs reference | 0.00022 (99.2% same top token) |
| Greedy outputs | Byte-identical to plain PTQ1_0 kernels |
| Output md5 after v0.10 | Identical to v0.8 |
| Output md5 after v0.10.1 | Identical to v0.10 |
| FLASH_ATTN_EXT q4_0 | 480/480 test cases |
| FLASH_ATTN_EXT (2D block loads) | 89/89 test cases |
| Perf regression cases | kv 16K/64K/128K x 1/5/8/12/16 tokens |
| ARCLAB fusion tests | Covered by `test-backend-ops -o ARCLAB_*` |

**Model file note:** the speculative numbers need a PTQ1_0 GGUF that includes the MTP head. PrismML's own `Ternary-Bonsai-2-27B-gguf` PTQ1_0 file has no MTP head (use `--spec-type ngram-mod`). A public build with the head is `sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF`, file `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` (6.3 GB). The author's own derivative with different weights: fresh code 77 t/s, rename 306, edit 205, plain generation 38 t/s at 128K.

**Format note (PrismML):**
- `*-PQ2_0.gguf` (fork group-128, ggml id 142): preferred on Metal/CUDA/HIP/CPU. ~6% smaller than group-64.
- `*-Q2_0_g64.gguf` / `27B *-Q2_g64.gguf` (official group-64, ggml id 42): runs on every backend AND mainline llama.cpp. If unsure, use this.
- `*-Q2_0.gguf` on **older** model repos is the **deprecated legacy format** (group 128 stored as id 42). It does not load on these builds.
- Do NOT mix this fork's `ggml-*` libraries with a stock llama.cpp build. Do not use the stale `prism-v6` branch.

---

## 15. Build and Run

### Build (SYCL)
```sh
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_TARGET=INTEL
cmake --build build-sycl -j --target llama-server llama-bench llama-cli
```

### Run
```sh
source /opt/intel/oneapi/setvars.sh
export GGML_SYCL_PTQ1_T2=all              # PTQ1_0 weights on XMX (ffn = FFN only, unset = off)
export LLAMA_ARG_SPEC_DRAFT_UBATCH=512    # smaller compute buffer for the MTP draft context
export GGML_SYCL_FA_ONEDNN_MAX_KV=98304   # above this a ~120K-token prompt runs out of VRAM
export GGML_SYCL_FA_DEC_DPAS=1            # decode / verify attention on XMX from q4_0 (Xe2)
export GGML_SYCL_T2_W8A8_MIN=1024         # full 1024-token prompt batches via oneDNN int8 GEMM
./build-sycl/bin/llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf -ngl 99 \
  -c 131072 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-ngram-mod-n-max 256 \
  -ub 1024 -b 2048 --chat-template-kwargs '{"enable_thinking":false}' \
  --host 0.0.0.0 --port 8080 --jinja \
  --chat-template-file docs/bonsai-arc-b580-windows/bonsai-chat-template.jinja
```

With a shorter context you can use `-ub 2048` for faster prompt reading.

The chat template file is the model's own template with one change: a system message in the middle of a conversation becomes a note in a user turn instead of an error. Coding agents such as Claude Code send those (see above); plain chat is unaffected.

### Prebuilt Windows binaries
Under [Releases](https://github.com/Torchit1/llama.cpp/releases) (experimental, needs a current Intel Arc driver): unzip and run `run-bonsai.bat`.

---

## 16. Caveats and Gotchas

1. **First-prompt kernel compile.** The first long prompt after a cold start compiles the XMX kernels (~30 s). The GPU driver caches them afterwards, in `~/.cache/neo_compiler_cache`. If the server hangs during start-up GPU initialisation after being killed mid-compile, move that cache directory aside.
2. **VRAM at 128K.** With ~0.5 GB spare VRAM, a prompt of ~119K tokens can run out of memory and take the server down if the oneDNN cap is not set.
3. **Level Zero adapter.** The legacy adapter is default because it is faster here; V2 is available via `SYCL_UR_USE_LEVEL_ZERO_V2=1`.
4. **Weights cost +31% memory.** The 2-bit XMX layout trades weight memory for speed.
5. **B580-specific tuning.** The verify tiles and the 256-register file are tuned for the B580's 20 Xe cores; the tiles are forced off on other GPUs unless `GGML_SYCL_PTQ1_T2_TUNED=1`. CITATION(techpowerup arc-b580.c4244 -- 20 Xe cores; the kernel count the fork tunes for is the same one AGENTS.md records, so the figure names its source here rather than being restated bare)
6. **2D block loads need 64-byte alignment.** The `ld2d` path only engages on 64-byte aligned K/V surfaces, and is off by default because it regresses at 128K without the 256-register file.

---

## 17. Credits

Built on llama.cpp (MIT), PrismML's PTQ1_0 support, and the `bonsai-combo` branch of `professorpalmer/llama.cpp-ada-ternary` (PrismML PR #221), with the ternary DPAS kernels from `libxsmm/TernSYCL` (BSD 3-Clause).

The v0.10 delta-net replay and one-graph MTP drafts are **independent implementations** of techniques described by [llamADALL](https://github.com/straightdot/llamADALL); the K/V-only MTP prompt pass an independent implementation of one described by [GRIMOIRE](https://github.com/doopeworld/GRIMOIRE). **No code from either project is used.**

v0.10.1 includes three commits from [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (#311, #312, #313), with their authorship preserved.

---

*Summary compiled from `docs/bonsai-arc-b580.md` and the commit history of the `arc-b580` and `sycl-fa-q4-xmx` branches (through commit `7adb2dd2`, Oct 10, 2026).*

---

## 18. 2×B580 / Tensor-Parallel Analysis

**Short answer:** the **per-device kernel work transfers to each card**, but the fork's optimizations are orthogonal to what actually limits 2×B580, and strict **tensor parallel is blocked for this model**.

### What transfers per-device
Everything in the fork is single-device SYCL work, so in a 2-card layout each card's shard benefits:
- XMX ternary repack + int8×int2 DPAS, XMX decode attention from q4_0 KV (`GGML_SYCL_FA_DEC_DPAS=1`), chunked-attention softmax, delta-net blocked/fused kernels, v0.8 fusions, B580-tuned verify tiles, 256-GRF decode mode, MTP one-graph draft chain, delta-net replay, K/V-only prompt pass.
- These apply per-device to each card's shard, so the kernel speedups carry over.

### Why strict tensor parallel is blocked
- `src/llama-model.cpp:366` rejects `LLAMA_SPLIT_MODE_TENSOR` for hybrid/state-space architectures. Bonsai 2 27B is 16 attention + **48 gated-delta-net layers** = exactly that category.
- `docs/multi-gpu.md` documents the tensor/layer/pipeline split modes.
- The SYCL backend *does* enumerate multiple devices (`dpct::dev_mgr`, `default_tensor_split` in `ggml-sycl.cpp:190-274`), so `-sm layer` (pipeline) works, but there is no NCCL-equivalent cross-device reduction for SYCL/Level Zero (no all-reduce).

### Why it does not pay off here
- **Single-card fit:** PTQ1_0 weights = 5.54 GiB; 128K + MTP + q4_0 KV ≈ 11 GiB — already fits one 12 GB card. The fork's memory work (q4_0 draft KV, delta-net replay −432 MiB, `-ub 1024`) exists precisely to make one card sufficient.
- **Pipeline split is harmful:** activations cross PCIe (~32 GB/s, latency-bound on B580) at all 64 layer boundaries/token, and the delta-net recurrent state must ship whenever its layer lives on the other card. Delta-net replay would make this *worse* (update terms would then have to travel). CITATION(this file is a compilation of the Bonsai fork's own `bonsai-arc-b580.md`, not a measurement of this repo's box: the ~32 GB/s PCIe figure and the 6-13 t/s TP2 figures are the fork's stated values, reproduced here as claims about the fork, never as intfer measurements)

### Measured 2×B580 (separate Strata TP2 work, same host, different engine, layer split)
- Coherent TP2 decode **6–13 t/s** vs a 1-GPU control's **36–47 t/s** (4–8× slower), dominated by expert-cache coverage + per-layer doorbell latency — not kernel quality.

### Bottom line
- Use **one B580** for the 27B.
- Use 2×B580 only for `-np 2` concurrency (two independent streams) or for models >12 GB.
- The fork has **never been measured on 2×B580** and its docs never mention multi-device; there is no multi-device SYCL tuning in the project.
