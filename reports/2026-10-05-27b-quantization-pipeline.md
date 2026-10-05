# The 27B quantization pipeline, and what precision each layer actually got

**Date:** 2026-10-05
**Subject:** `Qwen3.8-27B-int2.intifer` (19,923,282,182 bytes), the model
`scripts/test_tp2_27b.sh` runs at `--tp 2`.
**Status:** documentation, not a benchmark batch. There is no `RESULT:` line
because nothing here was measured on a quiet box and nothing here needs one —
every number is either derived from the artifact by a script in `scripts/`, or
cited to the run log that produced it.

---

## 0. How to read this: what is derived and what is cited

The one question this report exists to answer — *at which precision is each layer
quantized* — **cannot be answered from the source tree**, because the answer is
not in the source. It is the output of SpinQuant → DASH-Q → RCO under one
specific budget, for one specific checkpoint. Every statement of the form "the
MLP is INT2" is a citation of a run.

It is also not derivable by reading the role enum, because **the ladder a role
offers is not the scheme it receives.** Two different files in this repo define
`default_candidates()` and they disagree (§7.1).

So two derivation scripts were written, and every figure below comes from one of
them or is explicitly marked `CITATION`:

```
scripts/quant_precision_map.sh       # the map, the census, the byte accounting
scripts/rco_chunk_feasibility.sh     # why the run was chunked, and at what budget
```

Both read the model's own metadata header. They cannot drift from the model they
describe. Where a document and these scripts disagree, the document is wrong.

---

## 1. The artifact

| | |
|---|---|
| path | `/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer` |
| `model_id` in the header | `Qwen3-32B` |
| layers | 64 |
| tensors | 498 (496 with a layer index, + `embed_tokens` + `lm_head`) |
| architecture | hybrid: 16 full-attention layers (every 4th, `L % 4 == 3`) + 48 linear-attention (gated-delta / SSM) layers |
| parameters | 26,893,352,960 (26.893 G) |
| payload | 19,503,668,424 B = **18.164 GiB** |
| rotations | 4 matrices, 5120×5120 FP32 each = 400.00 MiB |
| whole file | 19,923,282,182 B = **18.555 GiB** |
| **stored bits/weight** | **5.802** |
| compression vs FP16 | **2.758×** payload-only, **2.700×** whole-file, **2.671×** layers-only |
| per device at TP=2 | **9.752 GB per token** |

`CITATION(scripts/quant_precision_map.sh, scripts/model_byte_roofline.sh)` for
every figure in this table.

### 1.1 File format (`.intifer`, magic `INTFER1`)

Laid out by `serialize_intfer` (`src/quant/output.cpp:396`):

```
  8 B   magic "INTFER1\0"
  4 B   version (1)
  4 B   n_tensors (498)
  8 B   code_len      19,503,513,600
  8 B   scale_len           153,856
  8 B   global_len                968
  4 B   n_rotations (4)
  2 B   reserved
  8 B   json_len             183,304   <- per-tensor metadata, see below
  …     code   payload   (all tensors' packed codes, concatenated)
  …     scale  payload   (all FP16 group scales, concatenated)
  …     global payload   (per-tensor FP32 alpha, concatenated)
  …     4 x 5120x5120 FP32 rotation matrices
```

The JSON header carries, per tensor: `name`, `role`, `N`, `K`, and a `qcfg` of
`{scheme, group_size, scale_dtype, has_global_divisor, act, rotation_id}`, plus
`code_offset/bytes`, `scale_offset/bytes`, `global_offset/bytes`, and
`est_error`. **That `est_error` is the only quality number the artifact keeps**
(§6).

---

## 2. The pipeline, stage by stage

`run_pipeline` (`src/quant/pipeline.cpp:29`) is four stages. For the 27B they ran
with these settings, all hard-coded in `examples/quant_model_real/main.cpp:288`:

```cpp
cfg.sq.type            = RotationType::Hadamard;   // not learned
cfg.sq.per_layer       = false;                    // 4 shared rotations, not 64
cfg.learn_codebook     = false;                    // canonical codebook
cfg.use_calibration    = false;                    // UNIFORM diagonal Hessian
cfg.skip_int8_above    = 16'000'000;               // no INT8 candidate above 16M params
cfg.rco.greedy_refine  = true;                     // coordinate-descent polish
cfg.rco.budget_bytes   = (uint64_t)(fp16_bytes * budget_frac);
```

### 2.1 SpinQuant — orthogonal rotation

`spinquant_preprocess` (`src/spinquant.cpp:443`) rotates the weights so that
outlier channels are spread before quantization. The rotation is applied to the
**weights** at four sites, and the engine replays the matching transform on the
**activations** online (`fwht_rot_to_half`, `src/gemm/quant_gemm.cpp:2499`), so
no extra storage is needed for it.

| id | site | roles rotated | side |
|---|---|---|---|
| 1 | `QKV` | `ATTN_Q/K/V`, `LINEAR_ATTN_QKV/Z/B/A` | Right (rotates K) |
| 2 | `O` | `ATTN_O`, `LINEAR_ATTN_OUT` | Left (rotates N) |
| 3 | `FFN_OUT` | `MLP_DOWN` | Left |
| 4 | `FFN_IN` | `MLP_GATE`, `MLP_UP` | Right |
| 0 | none | `EMBED`, `LM_HEAD` | — |

**Every rotation has `dim = 5120`, and 5120 is not a power of two.**
`make_hadamard` (`src/spinquant.cpp:99`) therefore emits
`blockdiag(Hadamard_4096, I_1024)` and warns on every one of the four sites. The
consequence is precise and worth stating: **the rotation mixes only 4096 of each
tensor's 5120 channels; the last 1024 pass through untouched.** So the outlier
suppression that justifies low-bit quantization covers 80 % of the channel space.
`CITATION(each chunks_int2/*.log, 4 identical WARN lines per chunk, src/spinquant.cpp:121)`

The learned rotation (`learn_rotation`, weighted Jacobi outlier suppression,
`src/spinquant.cpp:132`) exists and is **not used** — `type = Hadamard`.

### 2.2 DASH-Q — candidate generation

For every tensor, `default_candidates(role)` proposes a ladder, and
`dashq::quantize` (`src/quant/dashq.cpp:586`) materializes each rung:

| role | ladder offered by the **compiled** `src/quant_spec.cpp` |
|---|---|
| `EMBED`, `LM_HEAD` | INT4 g64, INT8 g32, FP16 |
| `ATTN_Q/K/V` | INT2 g16, INT2 g32, INT4 g64, INT8 g32, FP16 |
| `ATTN_O`, `MLP_GATE/UP/DOWN` | INT2 g16, INT2 g32, INT4 g64, INT8 g32 |
| `LINEAR_ATTN_*`, `NORM`, `ROPE`, default | **FP16 only** |

Per group the solver alternates assignment and scale for 4 passes against the
canonical symmetric codebook (`optimize_group`, `src/quant/dashq.cpp:432`); the
cost is `Σ H_ii (w − dequant)²` and `est_error = sqrt(mse / norm)`.

**Two things did not run:**

* `use_calibration = false` ⇒ `DiagHessian::uniform(K)`. **The diagonal Hessian
  was inert for the 27B.** DASH-Q's entire Hessian-weighted objective collapsed
  to plain L2, because a uniform Hessian contributes a constant factor to both
  `mse` and `norm`. The "diagonal Hessian + ILS, no off-diagonal error
  propagation" description of this pipeline is true of the *code* and false of
  *this artifact*.
* `learn_codebook = false` ⇒ canonical codebook, no per-tensor codebook search.

**What "group size" actually means here — read this before reasoning about the
numerics.** `src/quant/dashq.cpp:638-640` sizes the scale array as
`n_groups * 2` bytes: **one FP16 scale per K-group for the whole tensor,
broadcast across all N output rows.** So a "group" is a block of K columns
applied to every output row — an input-channel-block scale, which is what lets
the GEMM factor the scale out of the inner K loop. It is *not* per-row group
quantization, and the scale overhead is `16 / (G · N)` bits per weight, not
`16 / G`. Measured consequence: **all scale storage in the model is 150.2 KiB,
0.0008 % of the payload.**

`skip_int8_above = 16M` removes the INT8 rung from every tensor above 16 M
parameters, which is every MLP, `ATTN_Q`, `EMBED` and `LM_HEAD`. It survives on
exactly **two** roles, `ATTN_K` and `ATTN_V` (5.24 M params each) — which is why
`ATTN_V` is the model's only INT8 tensor.

### 2.3 RCO — bit-budget selection

`rco::run_rco` (`src/quant/rco.cpp`) is a multiple-choice knapsack solved exactly
enough to be boring:

1. **Floor filter** (`floor_scheme`, `src/quant/rco.cpp:38`): `EMBED`/`LM_HEAD` ≥
   INT4; `ATTN_Q/K/V` and `LINEAR_ATTN_QKV` ≥ INT4 in layer 0 else ≥ INT2;
   everything else ≥ INT2. A rung below the floor is not offered to the solver.
2. **Lagrangian bisection** for the smallest λ whose total cost fits the budget,
   per-tensor `argmin(value + λ·cost)` where `cost` is *real stored bytes*
   (`src/quant/rco.cpp:98`) and `value = role_weight · est_error`.
3. **Coordinate-descent polish** (`local_search`, `src/quant/rco.cpp:196`): move
   one tensor at a time to the best still-feasible option until no improvement.
4. If even the cheapest legal allocation exceeds the budget, `lagrange_feasible`
   returns that cheapest allocation and `run_pipeline` **throws**
   (`src/quant/pipeline.cpp:209`). There is no partial result.

Floors are what make the run's budget fraction load-bearing rather than
cosmetic: the *floor itself* costs 30.25 % of a two-linear-attention-layer
chunk's FP16 footprint, so no budget below that can succeed (§4.3).

### 2.4 Merge

The 32 chunks were concatenated by `examples/quant_merge`, which dropped 31
duplicate copies of `lm_head` and kept one `embed_tokens`
`CITATION(/var/home/fra/data/models/intfer/merge_int2.log: "tensors: 498 (dropped
31 duplicate names)", "FINAL verify: magic=INTFER1 hdr[8]=1 n_tensors=498")`.

---

## 3. **At which precision each layer is quantized** — the complete answer

Read from the artifact. `2g16` = INT2 group 16, `2g32` = INT2 group 32,
`4g64` = INT4 group 64, `8g32` = INT8 group 32.

### 3.1 By role

| role | tensors | shape (N×K) | scheme | GiB | % bytes | % params | bits/w | act |
|---|---|---|---|---|---|---|---|---|
| `LINEAR_ATTN_QKV` | 48 | 10240×5120 | **FP16** | 4.688 | 25.8 % | 9.4 % | 16.000 | FP16 |
| `LINEAR_ATTN_Z` | 48 | 6144×5120 | **FP16** | 2.812 | 15.5 % | 5.6 % | 16.000 | FP16 |
| `LINEAR_ATTN_OUT` | 48 | 5120×6144 | **FP16** | 2.812 | 15.5 % | 5.6 % | 16.000 | FP16 |
| `MLP_UP` | 64 | 17408×5120 | **mixed** 4g64×33 / 2g16×29 / 2g32×2 | 2.013 | 11.1 % | 21.2 % | 3.031 | INT8_G64 |
| `MLP_DOWN` | 64 | 5120×17408 | **mixed** 2g16×34 / 4g64×30 | 1.951 | 10.7 % | 21.2 % | 2.938 | INT8_G64 |
| `MLP_GATE` | 64 | 17408×5120 | **mixed** 2g16×33 / 4g64×19 / 2g32×12 | 1.722 | 9.5 % | 21.2 % | 2.594 | INT8_G64 |
| `LM_HEAD` | 1 | 248320×5120 | **INT4 g64** | 0.592 | 3.3 % | 4.7 % | 4.000 | INT8_G64 |
| `EMBED` | 1 | 248320×5120 | **INT4 g64** | 0.592 | 3.3 % | 4.7 % | 4.000 | INT8_G64 |
| `ATTN_Q` | 16 | 12288×5120 | **INT4 g64** | 0.469 | 2.6 % | 3.7 % | 4.000 | INT8_G64 |
| `ATTN_O` | 16 | 5120×6144 | **INT4 g64** | 0.234 | 1.3 % | 1.9 % | 4.000 | INT8_G64 |
| `ATTN_K` | 16 | 1024×5120 | **FP16** | 0.156 | 0.9 % | 0.3 % | 16.000 | FP16 |
| `ATTN_V` | 16 | 1024×5120 | **INT8 g32** | 0.078 | 0.4 % | 0.3 % | 8.000 | INT8_G64 |
| `LINEAR_ATTN_A` | 48 | 48×5120 | **FP16** | 0.022 | 0.1 % | 0.04 % | 16.000 | FP16 |
| `LINEAR_ATTN_B` | 48 | 48×5120 | **FP16** | 0.022 | 0.1 % | 0.04 % | 16.000 | FP16 |

Three families, and the split is not the one the role enum suggests:

* **uniformly FP16:** the five `LINEAR_ATTN_*` roles plus `ATTN_K` — 256
  tensors, 10.513 GiB, **57.9 % of bytes but only 21.0 % of parameters**
* **uniformly quantized:** `ATTN_Q`/`ATTN_O`/`EMBED`/`LM_HEAD` at INT4 g64,
  `ATTN_V` at INT8 g32
* **mixed tensor-to-tensor:** `MLP_GATE`, `MLP_UP`, `MLP_DOWN` only. RCO decides
  these **per tensor, not per role**, and they are the only place in the model
  where the bit allocation is genuinely searched.

### 3.2 Scheme mix, both ways — the number that matters for decode

| scheme | GiB | % of bytes | G params | % of params |
|---|---|---|---|---|
| **FP16** | 10.513 | **57.9 %** | 5.644 | 21.0 % |
| INT4 g64 | 5.291 | 29.1 % | 11.361 | 42.2 % |
| INT2 g16 | 1.992 | 11.0 % | 8.556 | 31.8 % |
| INT2 g32 | 0.291 | 1.6 % | 1.248 | 4.6 % |
| INT8 g32 | 0.078 | 0.4 % | 0.084 | 0.3 % |

**The model is majority-FP16 by bytes.** Any plan that reasons about "the INT2
model" is reasoning about 11 % of what decode actually streams.

### 3.3 The complete 64-layer map

`-` = tensor absent in that layer (the hybrid architecture). `type` is `full`
for the 16 full-attention layers, `lin` for the 48 linear-attention ones.

```
  L  type     Q     K     V     O   QKV     Z     A     B     o  gate    up  down
  0   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  4g64  2g16
  1   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  4g64  2g16
  2   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  4g64  4g64
  3  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
  4   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
  5   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g32  2g16
  6   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  2g16
  7  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
  8   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
  9   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 10   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 11  full  4g64  FP16  8g32  4g64     -     -     -     -     -  2g32  4g64  4g64
 12   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  2g16  2g16
 13   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 14   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 15  full  4g64  FP16  8g32  4g64     -     -     -     -     -  2g16  4g64  4g64
 16   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  2g16  2g16
 17   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  2g16  2g16
 18   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  2g16
 19  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
 20   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 21   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  2g16  2g16
 22   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  4g64  4g64
 23  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
 24   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 25   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 26   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  4g64  4g64
 27  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
 28   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 29   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 30   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 31  full  4g64  FP16  8g32  4g64     -     -     -     -     -  2g32  4g64  4g64
 32   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 33   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  2g16  2g16
 34   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  4g64  4g64
 35  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
 36   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 37   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 38   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  4g64  4g64
 39  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
 40   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 41   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 42   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 43  full  4g64  FP16  8g32  4g64     -     -     -     -     -  2g32  4g64  4g64
 44   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  2g16  2g16
 45   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g32  2g16  2g16
 46   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 47  full  4g64  FP16  8g32  4g64     -     -     -     -     -  2g32  4g64  4g64
 48   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 49   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 50   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  4g64  2g16
 51  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  4g64  4g64
 52   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 53   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 54   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 55  full  4g64  FP16  8g32  4g64     -     -     -     -     -  2g16  4g64  4g64
 56   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 57   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 58   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 59  full  4g64  FP16  8g32  4g64     -     -     -     -     -  2g16  4g64  4g64
 60   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g32  2g16
 61   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  2g16  2g16  2g16
 62   lin     -     -     -     -  FP16  FP16  FP16  FP16  FP16  4g64  4g64  4g64
 63  full  4g64  FP16  8g32  4g64     -     -     -     -     -  4g64  2g16  4g64
```

**`MLP_GATE`, `MLP_UP` and `MLP_DOWN` are the only tensors in the model whose
precision was searched rather than fixed.** Every other cell in this table is
either a role-level constant (`ATTN_*`, `LINEAR_ATTN_*`, `EMBED`, `LM_HEAD`) or a
consequence of the chunk the layer landed in (§4.3).

---

## 4. Why the map looks the way it does

### 4.1 RCO never saw the whole model

**The 27B was quantized as 32 chunks of 2 layers each**, not in one pass, and
each chunk got its own budget: `budget = (that chunk's FP16 bytes) ×
budget_frac` (`examples/quant_model_real/main.cpp:293`). The chunks were then
concatenated.

So the allocator's problem was never "distribute 36 GiB across 64 layers". It was
"distribute ~1.1 GiB across 2 layers plus a full copy of `lm_head`", 32 times,
independently, with no information passing between chunks. `RcoOptions::budget_bytes`
defaults to 36 GiB and is **not** what was used.

### 4.2 `lm_head` eats 54 % of every chunk's budget

`lm_head` is not a layer tensor, so it is not filtered by the chunk's layer range
— every chunk loads and pays for it (`--no-embed` suppresses only
`embed_tokens`, and only chunk 0 carries that).

| | |
|---|---|
| `lm_head` stored (INT4 g64, 248320×5120) | **606.3 MiB** |
| budget of a LIN+SELF chunk | 1121.1 MiB |
| ⇒ share of the chunk budget | **54.1 %** |
| budget left for the chunk's 15 layer tensors | 514.8 MiB |
| those 15 tensors' FP16 footprint | 1440.8 MiB |
| ⇒ effective layer budget | **35.7 % of FP16**, not the 29 % the fraction names |

The same 606.3 MiB is re-derived 31 times and then 30 of those copies are thrown
away by the merge. It is 54 % of every chunk's decision and 3.3 % of the final
artifact.

### 4.3 Two budget fractions, and the split is forced

`scripts/archive/run_27b_int2_chunks.sh` uses **0.29** for chunk 0 and every
chunk holding one linear-attention + one full-attention layer, and **0.305** for
chunks holding two linear-attention layers. Its header justifies the split only
as "LIN+LIN chunks need 0.305", citing four probe logs.

That is derivable from the code's own floors and cost function, and
`scripts/rco_chunk_feasibility.sh` does it:

| chunk class | layers | frac | FP16 denominator | floor cost | headroom | verdict |
|---|---|---|---|---|---|---|
| chunk 0 (LIN+LIN, carries `embed`) | 0,1 | 0.290 | 8736.9 MiB | 1781.9 MiB | **+48.6 MiB** | FEASIBLE |
| LIN+SELF | 2,3 | 0.290 | 3865.9 MiB | 979.7 MiB | **+141.4 MiB** | FEASIBLE |
| LIN+LIN | 4,5 | 0.290 | 3886.9 MiB | 1175.6 MiB | −48.4 MiB | INFEASIBLE |
| LIN+LIN | 4,5 | 0.302 | 3886.9 MiB | 1175.6 MiB | −1.8 MiB | INFEASIBLE |
| LIN+LIN | 4,5 | 0.305 | 3886.9 MiB | 1175.6 MiB | **+9.9 MiB** | FEASIBLE |

The LIN+LIN floor is 30.25 % of FP16 — ten FP16 linear-attention tensors, a
606.3 MiB `lm_head` at its INT4 floor, and six MLPs pinned at INT2 g32. The
feasibility edge is **0.30246**, which is why the archived probes exist at 0.30,
0.302 and 0.305 and why 0.302 failed by 1.8 MiB.

Three things this reproduces that the archived comment does not state:

* **the derived floor, 1175.6 MiB, equals the `selected 1175.6 MiB` in the
  shipped chunk logs to the byte** — so the LIN+LIN chunks were solved with the
  budget as their only constraint;
* **one bit-width upgrade costs 21.25 MiB.** Against 9.9 MiB of headroom, *no MLP
  tensor in a LIN+LIN chunk can leave INT2.* Against chunk 0's 48.6 MiB,
  *at most 2 of 6 can.* Both are falsifiable predictions, and
  `scripts/quant_precision_map.sh` tests the first:

```
MLP scheme by CHUNK CLASS
  class               layers    2g16    2g32    4g64    8g32
  chunk0(LIN+LIN)          2       4       0       2       0   (of 6 MLP tensors)
  LIN+LIN                 30      81       9       0       0   (of 90 MLP tensors)
  LIN+SELF                32      11       5      80       0   (of 96 MLP tensors)
  prediction: zero INT4/INT8 MLPs in a LIN+LIN chunk. Observed violations: none
```

**Zero of 90 MLP tensors in the 30 LIN+LIN layers is INT4; 80 of 96 in the
LIN+SELF layers are. Chunk 0 upgraded exactly 2 of 6, which is the most its
48.6 MiB of headroom could buy.** Of the model's 192 MLP tensors, 110 are INT2,
and 90 of those 110 are in chunks where INT4 was arithmetically unaffordable.

The residual `2g16` vs `2g32` scatter is not structural — it is the Lagrangian
trading a few hundred bytes of scale storage against a small error gain, and it
is genuinely per-tensor.

### 4.4 Within a chunk, the allocation *is* error-driven

Where RCO has headroom it allocates by error, and the result is not noise. From
`chunks_int2/2_4.log`, layers 2 and 3 share a chunk and have **identical MLP
shapes**:

```
MLP_GATE   L2  est_error 0.3478  -> 2g16
MLP_GATE   L3  est_error 0.1882  -> 4g64
MLP_UP     L2  est_error 0.1814  -> 4g64
MLP_DOWN   L2  est_error 0.1840  -> 4g64
```

Same shape, same ladder, same λ — and the higher-error tensor got fewer bits.
That is the search working. It is worth separating from §4.3, where the
allocation had no freedom at all.

### 4.5 A display bug that makes the chunk logs misread

`examples/quant_model_real/main.cpp:210` prints the budget fraction with
`%.2f`, and **on this box `printf("%.2f", 0.305)` prints `0.30`** — verified by
compiling and running it against glibc, and reproduced by Python's `'%.2f' %
0.305`. (`%.20g` of the same value prints `0.30500000000000000001`, so the loss
is in the two-decimal presentation, not in the stored value.) Every LIN+LIN
chunk log therefore says `budget_frac: 0.30` while the value used was 0.305,
which is provable from the same log: `1185.5 MiB / 3886.9 MiB = 0.30500`. Anyone
reading the logs to find out what ran will get this wrong.

---

## 5. What the precision costs

### 5.1 Bytes

| | GiB | share |
|---|---|---|
| FP16 (5 linear-attn roles + `ATTN_K`) | 10.513 | 57.9 % |
| INT4 g64 | 5.291 | 29.1 % |
| INT2 (g16 + g32) | 2.283 | 12.6 % |
| INT8 g32 | 0.078 | 0.4 % |
| scale payload | 0.00015 | 0.0008 % |
| global-divisor payload | 0.0000009 | 0.000005 % |
| **payload** | **18.164** | |
| 4 rotation matrices (FP32, 5120²) | 0.391 | 2.1 % of file |
| **file** | **18.555** | |

### 5.2 Per device at `--tp 2`

TP slices N, so each device reads `code_bytes/2` of every tensor once per token:

| | GB/device/token |
|---|---|
| FP16 | 5.256 (57.9 %) |
| INT4 g64 | 1.950 (21.5 %) |
| INT2 | 1.837 (20.2 %) |
| INT8 g32 | 0.039 (0.4 %) |
| **total** | **9.752** |

### 5.3 Compression: the log's headline is not the artifact's

Each chunk log prints `compression 3.45x` / `3.31x`. Those are
`chunk_FP16 / chunk_selected`, and the chunk's FP16 denominator includes a
`lm_head` that belongs to no layer. The model's actual figures:

* payload only: **2.758×**
* whole file (incl. the 400 MiB of rotations): **2.700×**
* layer tensors only: **2.671×**

### 5.4 A projection, not a measurement

If the `LINEAR_ATTN_QKV/Z/OUT` triple were quantized at the INT4 g64 rate the
rest of the model gets, its 5.537 G parameters would occupy 2.739 GiB instead of
10.312 GiB, and the payload would fall from 18.164 GiB to **≈10.6 GiB** — a
further 1.7×. This is arithmetic on the shapes in §3.1 and the stored-bytes rule
in `scripts/rco_chunk_feasibility.sh`. **It is not a measurement, no such model
exists, and the error it would introduce is unknown** (§7.1).

---

## 6. Quality, and the population the quality number describes

`est_error` is stored per tensor in the artifact: the Hessian-weighted relative
L2 reconstruction error, `sqrt(Σ H_ii (w−dq)² / Σ H_ii w²)`. Because the Hessian
was uniform (§2.2) this is plain relative L2.

Params-weighted over the model:

| population | est_error | share of params |
|---|---|---|
| all 498 tensors | **0.2131** | 100 % |
| the 242 quantized tensors | **0.2696** | 79.0 % |
| the 256 FP16 tensors | 0.0000 | 21.0 % |

**The FP16 tensors pull the all-tensors mean down by 1.27×.** This matters
because the run's own PASS/REVIEW thresholds are applied to the all-tensors
number (`examples/quant_model_real/main.cpp:398-399`: mean weight recon < 0.25,
linear-output error < 0.40). Every chunk passed, with whole-chunk means of
0.2141–0.2252 (LIN+SELF) and 0.2431–0.2443 (LIN+LIN), and max linear-output
errors of 0.3525–0.3689 `CITATION(chunks_int2/*.log)`. But a threshold written
against a mean that includes 21 % of parameters at exactly zero error is not a
statement about the quantized weights, and the quantized population's 0.2696 is
above the same 0.25 line.

Per role (`est_error`, params-weighted within the role):

| role | est_error |
|---|---|
| `LINEAR_ATTN_*`, `ATTN_K` | 0.0000 (exact — they are FP16) |
| `ATTN_V` (INT8) | 0.0121 |
| `ATTN_O` (INT4) | 0.1923 |
| `ATTN_Q` (INT4) | 0.2221 |
| `MLP_UP` (mixed) | 0.2616 |
| `MLP_DOWN` (mixed) | 0.2694 |
| `EMBED` (INT4) | 0.2631 |
| `LM_HEAD` (INT4) | 0.2434 |
| `MLP_GATE` (mixed) | 0.3042 |

**What was never measured:** no autoregressive generation, no perplexity, no
task metric. `examples/quant_model_real/main.cpp:400` prints
`(operator-level fidelity; full autoregressive generation not yet run)`, and that
caveat is the whole scope of the claim. The gates in `scripts/run_ten_gates.sh`
check that the *engine reproduces this artifact* bit-for-bit on the token stream;
they do not check that the artifact is a good model.

---

## 7. Findings, ranked by how much they change what to do next

### 7.1 The linear-attention stack is FP16 by omission, not by budget — and two different ladders exist in the repo

`src/quant_spec.cpp` — the file `scripts/build_quant_real.sh` compiles, and the
one `src/CMakeLists.txt` puts in `intfer_quant_core` — **has no `case` for
`LINEAR_ATTN_QKV/Z/B/A/OUT`.** They fall through to `default:` and receive FP16
as their only candidate. RCO therefore never had a choice to make for them.

`src/quant/quant_spec.cpp` **does** extend the ladder to those roles, and is what
`six examples/*` targets compile (`dashq_demo`, `quant_pipeline`, `rco_demo`,
`quant_merge`, `intfer_fidelity`, `intfer_gguf`). The header comment at
`include/intfer/quant_spec.h:20-25` states the intent explicitly: *"so DASH-Q can
target them with the same INT2/INT4/INT8 candidate ladder as self_attn/MLP."*

The two cannot be linked together — duplicate symbols — so each target silently
picks one, and the divergence produces no build error and no warning. The
consequence is measurable: **every demo prints a scheme distribution the real
quantizer cannot produce**, and the real artifact carries **10.513 GiB of FP16
(57.9 % of bytes, 256 of 498 tensors) that the demos show as quantized.**

Note this cuts against the header comment's own framing, which describes
linear-attn as "the bulk of Qwen3.5's parameters" — at 27B they are 21.0 % of
parameters but 57.9 % of bytes, because FP16 does not compress.

Fixing it is not a one-line change: the artifact would have to be re-quantized,
and every engine kernel reachable from a `LINEAR_ATTN_*` role would move from the
FP16 path to the dequant-on-the-fly path.

### 7.2 The allocator never saw the model

§4.1–4.3. Thirty-two independent 2-layer problems instead of one 64-layer
problem, with `lm_head` consuming 54 % of each budget and thrown away 30 times.
The visible symptom is §4.3's prediction confirmed: half the MLP weights are
INT2 for budget reasons that have nothing to do with their error.

A whole-model RCO run was not possible as configured — the harness caps itself at
`0.80 × MemAvailable` because `run_pipeline` holds every candidate of every
tensor in RAM simultaneously (`src/quant/pipeline.cpp` merges all thread-local
buffers into one contiguous `code_buf`), and the FP16 working set for 64 layers
does not fit. `CITATION(chunks_int2/*.log: "[max-mem] available=41294 MiB
cap=33035 MiB (80.0%) per_layer_footprint=725 MiB")`. The chunking is a memory
workaround, and the bit allocation inherited it.

### 7.3 DASH-Q's Hessian was inert

§2.2. `use_calibration = false` ⇒ uniform diagonal Hessian ⇒ the objective is
plain L2. Nothing in the pipeline collects calibration activations
(`CalibrationData` exists and `DiagHessian::from_activations` is implemented,
but no runner feeds it: `main.cpp:292` says so in a comment). Every `est_error`
in the artifact, and therefore every RCO decision, is activation-agnostic.

### 7.4 The rotation covers 80 % of each tensor

§2.1. `dim = 5120` is not a power of two, so all four rotations are
`blockdiag(Hadamard_4096, I_1024)`. Channels 4096–5119 are never mixed. On a
model where the whole justification for INT2 MLP weights is outlier suppression,
20 % of the channel space went unsuppressed — and this is not visible anywhere in
the artifact, which records `dim: 5120` and nothing about the block structure.

### 7.5 The global divisor is stored and always 1.0

`has_global_divisor` is true on all 242 quantized tensors, and
`src/quant/dashq.cpp:681-685` writes `α = 1.0f` unconditionally. Total cost:
968 bytes. The dequant path multiplies by it (`quant_gemm.cpp:317`). It is a
vestigial field, not a feature that was tuned.

### 7.6 `effective_bits()` overstates the bits/weight by 1.077×

`effective_bits` (`src/quant_spec.cpp:19`) charges scale overhead as
`scale_bits / group_size`, which assumes a scale per (row, group). The layout is
one scale per K-group for the whole tensor, so the true overhead is
`scale_bits / (group_size · N)`. Over the model: **6.2503 bits/weight reported vs
5.8018 actually stored.**

RCO's *cost* is `TensorIR::weight_bytes()`, i.e. real bytes
(`src/quant/rco.cpp:98`), so **the allocation is unaffected.** Only
`RcoReport::mean_eff_bits` (`src/quant/rco.cpp:280`) is wrong, and nothing
currently reads it.

### 7.7 The artifact cannot explain its own choices

`apply_solution` clears the candidate list after selection
(`src/quant/rco.cpp:329`: `t.candidates.clear()`), and the serializer writes only
the winning rung. So the per-candidate `est_error` values that drove every
decision are gone.

One decision is visibly unexplained by the artifact: **`ATTN_K` is FP16 in all
16 full-attention layers**, the only role in the quantized family where RCO chose
the most expensive rung, while its twin `ATTN_V` (same 1024×5120 shape, same
rotation site, same ladder, also under the 16 M INT8 cutoff) went to INT8 g32.
Resolving it needs the discarded candidate table.

### 7.8 Non-uniformity is mostly chunk bookkeeping, not search

§4.3's cross-tab: the `MLP` scheme is almost perfectly predicted by chunk class.
Of the 192 MLP tensors, **110 are INT2**, and **90 of those 110 sit in LIN+LIN
chunks where the budget could not afford a single bit-width upgrade**; of the
82 INT4 tensors, 80 are in LIN+SELF chunks and 2 in chunk 0. Read the map in
§3.3 with that in mind: it looks like a per-layer sensitivity finding, and for
the 30 linear-attention layers it is not one.

---

## 8. What this means for the optimization work

The performance narrative in `AGENTS.md` is an INT2 narrative, and the largest
measured deficit in the engine is *"the M=1 INT2 GEMV is 5.6–6.4× slower than a
clone of its own algorithm."* Two derived facts change how much that is worth:

1. **INT2 is 20.2 % of per-device decode traffic.** The other 57.9 % is FP16
   linear-attention, and `AGENTS.md` already records the FP16 M=1 path running at
   **98–123 % of the memory roofline**. So the 6× kernel deficit applies to a
   fifth of the bytes, and the majority of decode traffic is already at the wall.
2. **A 1.7× smaller model is on the table** (§5.4) — not because the kernel got
   faster, but because `LINEAR_ATTN_QKV/Z/OUT` are currently stored at 16 bits
   with no search having been performed on them. Whether that is a net win is a
   quality question (§7.1), not a performance one, and it cannot be answered from
   an agent session because it requires a re-quantize and a fidelity run.

Neither of these is an optimization to hand to the operator. They are scoping
facts: they say the remaining headroom in decode is smaller than "INT2 kernel is
6× slow" suggests, and that the largest single lever is a *model* change rather
than a kernel change.

---

## 9. Reproducing every number in this report

```bash
# the map, the census, the byte accounting, the chunk-class cross-tab
scripts/quant_precision_map.sh                       # full, with the 64-row map
scripts/quant_precision_map.sh --no-map              # census only

# per-device bytes and the roofline denominator
scripts/model_byte_roofline.sh                       # TP=2 by default

# the feasibility edge, and the proof that LIN+LIN chunks had no freedom
scripts/rco_chunk_feasibility.sh <model> 2,3 0.29
scripts/rco_chunk_feasibility.sh <model> 4,5 0.29 0.302 0.305

# the run that produced it (archived; these are records, not entry points)
cat /var/home/fra/data/models/intfer/chunks_int2_run.log
cat /var/home/fra/data/models/intfer/chunks_int2_run2.log
cat /var/home/fra/data/models/intfer/chunks_int2/2_4.log
cat /var/home/fra/data/models/intfer/merge_int2.log
ls   /var/home/fra/dev/intfer/test/quant_int2/probe_4_6_b*.log
```

The two `scripts/` drivers for the 27B quantization live in
`scripts/archive/` (`run_27b_int2_chunks.sh`, `quant_27b_chunks_redo.sh`,
`quant_27b_re.sh`). They are archived because nothing calls them and they have
been superseded; they are read here as the record of how the artifact was made,
which is the only thing they are still good for.
