# INT2 Quantization Spec & Tensor IR (t0)

> Stage 1 — target model selection + final mixed-INT2 quantization scheme and the
> tensor-IR contract consumed by **t3** (SpinQuant), **t4** (DASH-Q), **t5** (RCO), **t6** (pipeline/output).
> Derived from `docs/ninfer-quantization-brief.md` (t1). All downstream tasks must conform to the
> Tensor IR in §6 and the header `include/intfer/quant_spec.h`.

---

## 1. Hardware assumptions & VRAM budget

| Item | Value | Notes |
|---|---|---|
| GPU | Intel Arc **Pro B580** ×2 (Battlemage XMX / Xe2) | No native FP4; XMX DPAS supports INT8/INT4; FP16/BF16 MMA |
| VRAM per GPU | **24 GB** (assumption: Pro 24 GB variant) | If consumer 12 GB, see fallback in §7; budget is a parameter `VRAM_PER_GPU_GB` |
| Total VRAM | 48 GB | Tensor-parallel: weights sharded 50/50, one shard/GPU |
| Usable weight budget / GPU | ~18 GB | after KV cache + activations + runtime reserve |
| Effective total weight budget | **~36 GB** | Σ of both shards; this is the hard constraint **RCO (t5)** optimizes against |

INT2 benefit on Arc is **weight-bandwidth + cache residency** (8× vs FP16 in memory) with
dequant-on-the-fly into XMX FP16/INT8 MMA. There is **no native FP4 TC**, so MAC-density
gains come from the INT4/INT8 DPAS path (W2A8 variant), not from a dedicated 2-bit unit — this
is the honest analog of ninfer's Blackwell NVFP4 TC and is called out explicitly so Stage 2 (t7)
benchmarks it correctly.

---

## 2. Target model

The spec is **model-agnostic** (applies to any HF transformer with `nn.Linear` attentions/MLPs),
but pins concrete targets:

| Role | Model | Params | FP16 size | INT2-mixed est. | Fit on 2×B580 |
|---|---|---|---|---|---|
| **Primary (dense)** | **Qwen3-32B-Instruct** | 32 B | 64 GB | ~12 GB | ✅ huge headroom |
| Dev / bring-up (single GPU) | **Qwen3-8B-Instruct** | 8 B | 16 GB | ~3 GB | ✅ 1 GPU |
| MoE alt (TP demo) | **Qwen3-30B-A3B-Instruct** | 30 B (3 B active) | 60 GB | ~11 GB | ✅ experts shard per-GPU |

Rationale: ninfer's registered checkpoints are Qwen-class, so matching the Qwen3 family lets us
reuse its tokenizer/architecture and compare against ninfer's NVFP4 Qwen results. Qwen3-32B is the
largest dense model that fits 2×B580 at INT2 with room for long-context KV (handled in Stage 3).
Qwen3-8B is the validation vehicle for the whole Stage 1+2 bring-up on a single GPU.

---

## 3. INT2 weight quantization scheme (canonical)

Clones ninfer NVFP4's **two-level microscaling** (the part worth copying per t1 §4) but at 2 bits.

### 3.1 Codes
- **2-bit signed, symmetric, zero-point-free.** Per group, DASH-Q (t4) learns a 4-entry codebook
  `C = {−c₃, −c₁, +c₁, +c₃}` (symmetric, non-uniform allowed). The integer "code" is the index
  0..3; magnitude/asymmetry come from the learned codebook + group scale.
- Stored packed: **4 codes per byte** (`uint8_t`, low 2 bits = first element).
- Logical tensor `[N, K]` where **K is the contraction dimension** (input dim), padded to %64;
  N padded to %128 (matches ninfer `BlockScaleK16M128x4` alignment so Stage 2 block loads are clean).

### 3.2 Grouping & scales (microscaling)
- **Group size G = 16 (canonical)**, G = 32 allowed as a budget variant (set by RCO).
- **Per-group scale `s` = group_maxabs / codebook_maxabs**, stored as **FP16** (canonical).
  - *Variant:* FP8-E4M3 storage (saves 0.5 bit/weight) for the extreme-budget RCO path; converted
    to FP16 at load (cheap scalar op, no XMX FP8 TC on Arc).
- **Per-tensor FP32 global divisor `α`** (one `float` per tensor), folded into a single epilogue
  multiply like ninfer — keeps the MMA inner loop scale-free. `dequant(x) = codebook[code] * s * α`.

### 3.3 Effective bits / weight (the scale overhead math)
| Scheme | code | scale | eff bits/w |
|---|---|---|---|
| **INT2-G16 / FP16 scale** | 2.0 | 1.0 (16b/16w) | **3.0** |
| INT2-G32 / FP16 scale | 2.0 | 0.5 (16b/32w) | 2.5 |
| INT2-G16 / FP8 scale | 2.0 | 0.5 (8b/16w) | 2.5 |
| INT4-G64 / FP16 | 4.0 | 0.25 | 4.25 |
| INT8-G32 / FP16 | 8.0 | 0.5 | 8.5 |
| FP16 | 16.0 | 0 | 16.0 |

Canonical INT2 = **3.0 eff bits/w** (maximizes accuracy; matches NVFP4's 4.5 strategy of
"spend bits on scales to keep ultra-low bits usable"). RCO may drop to G32/FP8 to hit the budget.

---

## 4. Activation handling (W2A*x*)

| Path | Weights | Activations | Stage-2 kernel | When |
|---|---|---|---|---|
| **Canonical W2A8** | INT2 (OTF dequant) | **INT8 per-token G64** (like ninfer `int8_g64`) | INT2×INT8 **DPAS** (`s2×s8`), scale folded to epilogue | Production decode (bandwidth + MAC density) |
| **Fallback W2A16-OTF** | INT2 (unpack→FP16 in registers) | FP16/BF16 | XMX FP16 MMA | Kernel bring-up; sensitive tensors |
| Sensitive escape | FP16 (no quant) | FP16 | XMX FP16 MMA | See §5 |

Activation INT8 scale is **per-token, per-group-64, FP16** (computed at runtime, *not* stored in the
weight artifact). This mirrors ninfer's `int8_g64_codec` and reuses the Hadamard K/V outlier
smoothing (SpinQuant family, t3) for the KV path in Stage 3.

---

## 5. Bit-allocation policy (mixed precision — input to RCO/t5)

Per-tensor role, the **allowed candidate set** DASH-Q (t4) emits and RCO (t5) selects from
under the §1 VRAM budget. Sensitive roles are *constrained* (cannot drop below a floor):

| Tensor role | Allowed candidates (t4) | Floor (min bits) | Rationale |
|---|---|---|---|
| `embed` (token embedding) | INT4, INT8, FP16 | INT4 | vocabulary lookup, sensitive |
| `lm_head` (unembed) | INT4, INT8, FP16 | INT4 | output logits, accuracy-critical |
| `attn_q/k/v` (in_proj) | INT2-G16, INT2-G32, INT4, INT8, FP16 | INT2* | *first layer forced ≥ INT4 (ninfer-style early-outlier escape)* |
| `attn_o` (out_proj) | INT2-G16, INT2-G32, INT4, INT8 | INT2 | |
| `mlp_gate/up/down` | INT2-G16, INT2-G32, INT4, INT8 | INT2 | bulk of params → INT2 target |
| MoE expert `gate/up/down` | same as MLP | INT2 | per-expert, sharded in TP |
| `norm` / `rope` / softmax | **FP16 (never quantized)** | FP16 | elementwise, cheap, must stay precise |

RCO maximizes a global accuracy proxy `Σ_tensor w_role·(−est_error)` subject to
`Σ bits ≤ VRAM budget`, where `est_error` is the DASH-Q relative-L2 reconstruction error (§6.4).
This is exactly ninfer's mixed-precision philosophy extended to INT2 (t1 §5 caveat #3).

---

## 6. Tensor IR — metadata contract for t3→t4→t5→t6

A model is represented as a **`ModelIR`** = ordered list of **`TensorIR`** entries plus three packed
byte buffers (`code_buf`, `scale_buf`, `global_buf`). This is the single interchange object:

- **t3 (SpinQuant):** reads FP16 `TensorIR` (role + shape + `rotation_id`), writes *rotated* FP16
  tensors back into the IR (updates `rotation_id`, keeps `scheme=FP16`); does not quantize.
- **t4 (DASH-Q):** for each rotatable tensor, emits **one or more candidate quantizations**; each
  candidate is a `TensorIR` with a `QuantConfig`, a `code`/`scale`/`global` buffer slice, and an
  `est_error`. For RCO, t4 may attach the candidate list to the parent tensor (see `candidates`).
- **t5 (RCO):** consumes the candidate set, picks exactly **one** `QuantConfig` per tensor that
  fits the budget, sets it as the canonical `TensorIR`.
- **t6 (pipeline):** serializes the final `ModelIR` (canonical configs only) to GGUF or native
  SYCL loader (§6.5).

### 6.1 Enumerations
```
QuantScheme : FP16 | INT8 | INT4 | INT2
TensorRole  : EMBED | LM_HEAD | ATTN_Q | ATTN_K | ATTN_V | ATTN_O
            | MLP_GATE | MLP_UP | MLP_DOWN | NORM | ROPE | OTHER
ActScheme   : FP16 | BF16 | INT8_G64
ScaleDtype  : FP16 | FP8_E4M3
```

### 6.2 `QuantConfig`
| field | type | meaning |
|---|---|---|
| `scheme` | QuantScheme | weight scheme |
| `group_size` | uint16 | K-group for scales (0 = per-tensor) |
| `scale_dtype` | ScaleDtype | FP16 canonical / FP8 variant |
| `has_global_divisor` | bool | emit per-tensor FP32 `α` |
| `act` | ActScheme | activation handling for this tensor |
| `rotation_id` | uint8 | SpinQuant R applied (0 = none; t3 sets this) |

### 6.3 `TensorIR`
| field | type | meaning |
|---|---|---|
| `name` | string | HF key, e.g. `model.layers.0.self_attn.q_proj.weight` |
| `role` | TensorRole | drives §5 policy |
| `N`, `K` | uint32 | logical `[N, K]`, K = contraction dim (padded %64 / %128) |
| `qcfg` | QuantConfig | chosen config (t5 finalizes) |
| `code_offset`, `code_bytes` | uint64 | slice in `code_buf` |
| `scale_offset`, `scale_bytes` | uint64 | slice in `scale_buf` |
| `global_offset`, `global_bytes` | uint64 | slice in `global_buf` (0 if none) |
| `est_error` | float | DASH-Q relative-L2 reconstruction error (set by t4) |
| `candidates` | vector<TensorIR> | DASH-Q variant list for RCO (t4 fills, t5 consumes) |

### 6.4 Conformance oracle (error metric)
- **Primary:** relative Frobenius/L2 reconstruction error of the (rotated) weight matrix after
  dequant vs FP16 reference: `‖W_fp16 − dequant(Q)‖_F / ‖W_fp16‖_F`.
- **Secondary (end-to-end):** perplexity delta on a calibration hold-out vs FP16 baseline.
- t4 reports `est_error` per candidate; t5 uses it as the accuracy proxy; t7/t10 validate the
  SYCL dequant path against the same reference.

### 6.5 Serialized layout (t6)
- **Native SYCL loader:** `ModelIR` binary = header (magic `INTFER1`) + JSON metadata (all
  `TensorIR` fields, buffers referenced by offset) + raw `code_buf`/`scale_buf`/`global_buf`.
- **GGUF (interop):** map each `TensorIR` to a GGUF tensor with a `quant_type` tag
  (`Q2_K`-like custom `INT2_G16`/`INT2_G32`) carrying the per-group FP16 scale + FP32 global divisor
  as auxiliary tensors. Tensor names preserved 1:1.

The concrete C++ definition lives in **`include/intfer/quant_spec.h`** (authoritative; t3–t6 include it).

---

## 7. Open parameters (configurable, not hardcoded)

| Param | Default | Notes |
|---|---|---|
| `VRAM_PER_GPU_GB` | 24 | set 12 for consumer B580; recomputes §1 budget |
| `INT2_GROUP_SIZE` | 16 | 32 = budget variant |
| `SCALE_DTYPE` | FP16 | FP8_E4M3 = storage variant |
| `ACT_SCHEME` | INT8_G64 | FP16 = bring-up fallback |
| target model id | Qwen3-32B | overridable per run |

---

## 8. Summary for downstream tasks
- **t3**: consume `TensorIR{role,shape,rotation_id=0}` FP16 → produce rotated FP16, set `rotation_id`.
- **t4**: consume rotated FP16 → emit candidate `TensorIR`s (INT2-G16/32, INT4-G64, INT8-G32, FP16)
  with `est_error` into `candidates`.
- **t5**: select one candidate/tensor under `Σbits ≤ ~36 GB` weight budget via §5 policy + §6.4 error.
- **t6**: serialize finalized `ModelIR` to native/GGUF with correct offsets + scales + global divisors.
