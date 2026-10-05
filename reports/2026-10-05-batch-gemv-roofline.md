# Batch: decode roofline — the plan was aimed wrong, and the box says so

> ## ==> READ THIS FIRST. TWO OF THE FIVE EXPERIMENTS ARE RETIRED BY THEIR OWN
> ==> MEASUREMENTS, AND ONE NEW FINDING IS BIGGER THAN ALL FIVE. <==
>
> **E1 (`INTFER_FP16_DECODE1`) is RETIRED.** The plan said the FP16 M=1 path —
> 57.9 % of decode's traffic — was the unexamined bottleneck and that its inner
> loop was the thing to fix. It runs at **98–123 % of the memory roofline**. It has
> no headroom. Optimising it would have been work on a kernel already at the wall.
>
> **E2b — the wide-load / `block_load` direction — is RETIRED for the INT2 path.**
> (Named `INTFER_DECODE1_BLOCKLOAD` in the plan, but it was **never shipped as a
> flag**: a draft was written, found to duplicate the already-present
> `INTFER_DECODE1_SUBGROUP`, and deleted before it was ever gated. See
> `reports/2026-10-05-batch-upstream-simd-gemv.md` §2. Do not go looking for the
> flag; what follows retires the *direction*.) Widening the
> code chunk from 8 to 16 — more bytes in flight per iteration — measured
> **0.25–0.35×, a 3–4× REGRESSION**. Register pressure dominates. This is the same
> shape of result as sgl-kernel-xpu's own "do not prefetch scales" (~3 % loss).
>
> **The finding that is bigger than the batch: the shipped INT2 kernel is
> 5.6–6.4× slower than a faithful clone of its own algorithm**, on the same buffers,
> with the same instrument, interleaved so contention cannot explain it. That is
> measured and reproducible. Its **mechanism is not located**, and the batch
> refuses to guess — see §4.

**Date:** 2026-10-05 · **Box:** 2x Arc Pro B580 (both enumerate) ·
**Commits:** `e8fbd4d` `cb51c21` `2d6299d` `d52e272`

---

## 0. What the operator has to run

**Almost nothing.** Every number this batch needed was measured in-session with
**device event timestamps** and **no model loaded**, so it is load-insensitive and
the paramount directive's "decide on the counters" rule applies unchanged. There is
no tok/s anywhere below that I did not measure, and there is no wall-time arm in
this batch.

The only thing worth your box is **one confirmation**, in §6.

```
scripts/test_roofline_probe.sh      # ~30 s, re-derives everything in §2
bash   scripts/test_gpu_optin.sh    # certifies E1's replacement (E3.1)
```

---

## 1. The derivation everything else is measured against

`scripts/model_byte_roofline.sh` reads `code_bytes` straight out of the
`.intifer` metadata, so it cannot drift from the model:

```
packed weights   : 19.50 GB total (18.16 GiB)
per device/token at TP=2 : 9.752 GB

role                  n   N_loc      K  scheme   GB/dev/tok   share
LINEAR_ATTN_QKV      48    5120   5120    FP16        2.517     25.8%
LINEAR_ATTN_Z        48    3072   5120    FP16        1.510     15.5%
LINEAR_ATTN_OUT      48    2560   6144    FP16        1.510     15.5%
MLP_UP               64    8704   5120    INT4        1.081     11.1%
MLP_DOWN             64    2560  17408    INT2        1.047     10.7%
MLP_GATE             64    8704   5120    INT2        0.925      9.5%
LM_HEAD               1  124160   5120    INT4        0.318      3.3%
EMBED                 1  124160   5120    INT4        0.318      3.3%
ATTN_Q               16    6144   5120    INT4        0.252      2.6%
ATTN_O               16    2560   6144    INT4        0.126      1.3%
ATTN_K               16     512   5120    FP16        0.084      0.9%
ATTN_V               16     512   5120    INT8        0.042      0.4%
LINEAR_ATTN_A/B      48      24   5120    FP16        0.024      0.2%

scheme mix, per device per token:
  FP16     5.256 GiB   57.9%
  INT4     1.950 GiB   21.5%
  INT2     1.837 GiB   20.2%
  INT8     0.039 GiB    0.4%
```

**A correction that has to be here.** The roofline this batch was planned against
said "21 % of ~179 GB/s". The B580's peak is **456 GB/s**
(CITATION(TechPowerUp `arc-b580.c4244`, saved at
`docs/Intel Arc B580 Specs _ TechPowerUp GPU Database.html`; Intel ARK sku/241598
agrees) and it has **20 Xe cores / 160 vector engines**, not the 512 that
`AGENTS.md`, three code comments and two live reports carried.
`git log -S '512 Xe'` shows the figure entered at `40219bf` (2026-10-04) and was
only ever propagated; 512 Xe cores is the **A770's** count. So the real figure is
**~9 % of peak, not ~21 %**, and the whole "EU occupancy / parallelism-bound"
model it supported is void. `scripts/check_hw_constants.sh` now refuses an uncited
capability figure anywhere in `AGENTS.md` / `docs/` / `reports/`.
The driver agrees independently: `max_compute_units=160` = 20 cores x 8 engines.

---

## 2. The discriminator, and what it decided

`examples/roofline_probe` — **no model bound** (AGENTS.md standing rule 1),
**device event timestamps** (same instrument as the engine's own `kernel` bucket).
Its question: is decode's bandwidth deficit the **allocator** or the **kernel**?

### Ceiling

| arm | GB/s | % of 456 spec |
|---|---|---|
| sequential read, `malloc_shared` (**what decode uses**) | 206–264 | 45–58 % |
| sequential read, `malloc_device` (E3.1) | 231–288 | 51–63 % |
| decode, **measured** (`perf_base.err`) | **41.1** | 9.0 % |

So decode runs at **15.6–19.9 % of the ceiling this card demonstrates with the same
allocator and the same access pattern**. A ~5× deficit, and it is **kernel-side**.

### Per scheme — the measurement that retired E1

```
gemv/MLP_GATE(INT2)    0.274 ms     40.6 GB/s    8.9% of peak  (N_local=8704 K=5120 bits=2)
gemv/MLP_UP(INT4)      0.308 ms     72.4 GB/s   15.9% of peak  (N_local=8704 K=5120 bits=4)
gemv/MLP_DOWN(INT2)    0.807 ms     13.8 GB/s    3.0% of peak  (N_local=2560 K=17408 bits=2)
-- CITATION(roofline_probe, device 0, device event timestamps: medians of 12
-- reps. COUNTERS, not wall time; see the contention note in section 3.)
gemv/LA_QKV(FP16)      0.094 ms    559.9 GB/s  122.8% of peak  (N_local=5120 K=5120 bits=16)
gemv/LA_OUT(FP16)      0.066 ms    477.1 GB/s  104.6% of peak  (N_local=2560 K=6144 bits=16)
gemv/ATTN_K(FP16)      0.037 ms    142.2 GB/s   31.2% of peak  (N_local=512  K=5120 bits=16)
```

**FP16 — 57.9 % of decode's traffic — is AT or ABOVE the roofline.** INT2 is
**13.8–40.6 GB/s, 14–40× slower on the same bytes**
CITATION(roofline_probe, device 0). The plan's central premise
was inverted, and measuring it cost one probe run.

### The always-red twin, and the three bugs it caught

The probe asserts that **a quarter of the bytes takes about a quarter of the
time** (same launch shape, disjoint quarters). Measured 0.21–0.24 against an
expected 0.25. It is not decoration; it earned its place three times before any
number in this report was trusted:

1. **v1 ceiling** was `for (i) acc += row[i]` — byte-at-a-time into one register,
   a 26.2M-long serial dependency chain. It measured **54.6 GB/s while the FP16
   GEMV on the same bytes reached 554** CITATION(roofline_probe, device 0). A
   "ceiling" an order of magnitude below a real kernel is not a ceiling. Now word-at-a-time with four accumulators.
2. **v1 twin** compared sequential against 64-strided and demanded sequential win.
   It fired, correctly, and the cause was (1) — a pattern comparison conflates
   "worse pattern" with "measures nothing".
3. **v2 twin** cut the ROW COUNT to a quarter, conflating fewer bytes with fewer
   work-items: 0.71× instead of 0.25×, a correct measurement of the wrong thing.

---

## 3. The two shipped changes

### E3.1 — `INTFER_WEIGHTS_DEVICE` (default OFF, kill `INTFER_NO_WEIGHTS_DEVICE`)

Weight shards move from shared USM + host `memcpy` to **device USM + `q.memcpy`**.
Same bytes, same kernels, same order ⇒ **bit-exact**, so the gate is
stream-identity and **no numerics contract**.

| | value |
|---|---|
| **what I measured, load-insensitive** | ceiling 206–264 → 231–288 GB/s CITATION(roofline_probe, device 0), i.e. **+11 % to +16 %** |
| `nk`/`nw`/`nr` | unchanged — the launch set is identical |
| direction to expect | `kernel` flat-to-down by ~10 % of the GEMM share; `usm_write` up by the shard fill (one-time, at preload) |
| **win / neutral / loss** | **win** if `kernel` drops; **neutral** if it does not (the probe already says the allocator is not the deficit); **loss** = a device-USM OOM at preload |
| correctness gates passed | `scripts/test_gpu_optin.sh` **PASS**, TP=1 **and** TP=2, both `rc=0` with streams identical to base (`279 6511 314 279`) |
| `RESULT:` | |

**The measured negative that goes with it.** The same probe puts decode at
15.6–19.9 % of the *shared* ceiling. Shared USM is **not** the pathology this
could have been — the shared ceiling is 45–58 % of spec, not 10 %. You asked for
"whatever is necessary to saturate bandwidth"; on the allocator side this is what
it was, it is worth having, and **it is not the fix**.

### E3.2 — the dequant sweep, which retired two hypotheses (no code shipped)

Four clones at the two INT2 shapes, **all variants interleaved with the shipped
kernel inside the same rep loop**. Interleaving was forced by measurement, not
taste: run separately, the shipped MLP_GATE arm came out at 17.2, 39.3 and 41.3
GB/s CITATION(roofline_probe, three consecutive runs of the same binary) on three
consecutive runs of the same binary on this contended box.

| variant | MLP_GATE GB/s | vs table | MLP_DOWN GB/s | vs table |
|---|---|---|---|---|
| `table` (verbatim copy of the shipped loop) | 260.9 | 1.00× | 77.6 | 1.00× |
| `select` (same expressions, **no array**) | 264.1 | **1.01×** | 79.0 | **1.02×** |
| `table16` (16 codes/chunk) | 64.7 | **0.25×** | 22.4 | **0.29×** |
| `select16` | 71.5 | 0.27× | 26.8 | 0.35× |
| **SHIPPED** | 40.8 | 0.16× | 13.8 | 0.18× |

* **Hypothesis 1 is DEAD, measured.** `select` is 1.01–1.02× `table`. The
  dynamically-indexed 4-entry register array costs nothing. Do not retry the
  bit-identical arithmetic dequant. (It *is* bit-identical — verified
  exhaustively over all 4 codes × 64 scales — so if it ever becomes useful it will
  not need a contract.)
* **Hypothesis 2 is DEAD, measured.** 16 codes/chunk is a **3–4× regression**.
  E2b's block-load direction is retired for INT2: more bytes in flight per
  iteration does not help, and register pressure dominates.

---

## 4. The open finding: a 5.6–6.4× clone-vs-shipped gap, mechanism unknown

`SHIPPED` is `quant_gemm_forward_usm`'s own `gemm_usm_decode1_vec`. `table` is a
faithful copy of the same algorithm — same group loop, same 8-code chunk, same
byte indexing, same table, same ragged tail — compiled in the probe's own
translation unit. Same shape, same buffers, same allocator, same instrument,
interleaved.

**The shipped kernel is 5.6–6.4× slower than a clone of its own arithmetic.**
That is measured and reproduced across every run today. So the shipped lambda is
losing to something **outside the algorithm**, and the arithmetic is not it.

### 4a. E-NEXT-1 ran, and it killed two of the three candidates

Rather than do the refactor and interpret whatever came out, the probe grew a
ladder of **single-change steps from the shipped text**, which answers the question
without touching the engine:

| arm | MLP_GATE | vs SHIPPED |
|---|---|---|
| `table` (tidy rewrite of the algorithm) | 193–244 GB/s | 6.3× |
| `VERBATIM` (shipped source, small TU) | 38.8 GB/s | **1.26×** |
| `VERB+TPL` (shipped source, flag as a template param) | 39.1 GB/s | 1.28× |
| `SHIPPED` (shipped source, 2378-line TU) | 30.7–40.8 | 1.00× |
| | | CITATION(roofline_probe) |

CITATION(roofline_probe, device 0, device event timestamps, interleaved, medians
over 12 reps. The GB/s ranges are across runs because this box is contended — the
**ratios** are the claim, never a single run's absolute.)

* **The MODULE theory is dead, by its own pre-committed criterion.** The plan said
  "≥2× or the theory is dead". Moving the shipped source to a small TU buys
  **1.26×**. So `INTFER_DECODE1_OWN_TU` would be a real 1.26× and is **not worth
  the refactor on its own**.
* **The RUNTIME-BRANCH theory is dead, measured.** `use_int2_tab` is a host-read
  bool captured into the kernel, guarding a two-copy inner loop *inside* the
  per-group loop — which runs **320 times per work-item** at MLP_GATE. "Loop
  unswitching is failing and costing registers" was the natural suspect. Making it
  a **template parameter**, so `if constexpr` deletes the branch *and* one loop body
  at compile time, measures **1.007×**.
* **So ~4.5–5× of the 5.6–6.4× sits in the shared text of those two functions.**

### 4b. The bisect is half-finished, and the blocker is part of the finding

The next two steps were single-change arms off `VERBATIM`:

* **`VERB+HALF`** — `from_half_bits` (the engine's branchy scalar FP16→FP32
  twiddle, 3 branches per call, once per group ⇒ 320× per work-item) replaced by
  `sycl::bit_cast<sycl::half>`.
* **`VERB+MP`** — the weight side read through a `multi_ptr` rather than a raw
  `code_u8[b0]`.

**Neither was measured.** Adding them pushed the probe's translation unit past the
**level-zero `UR_RESULT_ERROR_OUT_OF_RESOURCES` (40) limit on submit** — nine
near-identical kernel bodies in one JIT module will cross it. Deleting the
already-settled arms brought it back under, but not far enough to fit both new
ones.

**That limit is the actionable part**, and it is a real constraint for whoever
continues: **one probe TU holds roughly six to eight INT2 kernel bodies.** Split
the ladder across two probe binaries, or run a smaller ladder — but do not
rediscover it by adding one arm.

**The mechanism is NOT located, and this report does not guess.** The candidate is
register pressure: the shipped lambda captures ~15 scalars, two `multi_ptr`
objects and a 4-float dynamically-indexed array, inside a **2378-line translation
unit** that also holds the FP16, scalar, tiled and four DPAS kernels (two of which
contain inline vISA `dpas` asm). A launch-properties arm
(`sub_group_size<16>` + `grf_size_automatic`, both numerics-neutral, both pinned
by llama.cpp's `fattn-common.hpp` and vllm-xpu-kernels as *correctness*
constraints) was attempted to test the spill hypothesis and **could not be
built**: four spellings of the SYCL properties API, three wrong in ways whose
errors point nowhere near the cause, and the correct CTAD form still fails to
resolve against a bare lambda in SYCL 2026.1. The four failed spellings are
recorded in the probe source. An attempt to get IGC spill stats
(`-Xsycl-target-backend -fsycl-report-stats`) emitted no stats on the `spir64`
JIT target.

Rather than ship an untested fix for a suspected cause, §4a runs the named next
experiment as a measurement ladder. It answered: **the module theory is dead
(1.26× against a ≥2× criterion) and the runtime-branch theory is dead (1.007×).**
The bisect continues at `VERB+HALF` / `VERB+MP` and is blocked on the module size
limit in §4b, not on anything undecided.

---

## 5. Not started, with the work already done

### E4 — `INTFER_GPU_ATTN` (specified in the plan, not built)

The plan's finding stands and is the largest untouched piece: **decode attention
is a scalar host FP32 loop with `expf`, in four copies** (`main.cpp:4798` in
`step()`, plus `5390`, `5746`, `6402` in the prefill/batch paths), linear in
context, serial with all GPU work. `engine.hpp:9` claims the CLI uses
`flash_attention_forward_f16`; it does not — only `examples/qwen_forward_gpu` (a
demo) does, and that kernel is 1/64-utilised at decode (`seq_q=1`, `kBr=64`, so
`active = tid < 1`).

What the research settled, so the next session does not re-derive it: the whole
design is **plain SYCL and fully copyable** — PackGQA (fold the GQA group into the
M dim), split-KV with sgl's three-step `reduce_over_group(max)` →
`group_broadcast` → `exp2` → accumulate **unnormalised** partials → one divide at
the end, the four softmax folds, a host compact work-list with 2x core
oversubscription, and an SLM budget gate because sgl records that exceeding
128 KiB on BMG **hangs the device instead of failing cleanly**. Numerics: the host
reference reduces ascending-`d` then ascending-`t` with a separate softmax pass,
which is reproducible on device; `expf` → `native::exp2(x·log2e)` is the ~1 ulp
change the silu-exp precedent covers. **Scope `step()` only.**

### E5 — sycl-tla `Xe_Reorder` probe (not built)

Verdict from the research is **do not adopt**: no TMA, no mbarrier, no async
mainloop, no sub-byte mainloop (`Xe_Dequantize` is fp16/bf16-only, no group
scale), `N` fixed at 16 and depth hardcoded `.8.` for every type ⇒ **M=1 is
outside the library**. What it does have is one primitive we lack:
`Xe_Reorder<UU,int4_t,half_t>` (`include/cute/arch/reorder_xe.hpp:1070-1111` at
v0.9.3 = `7b56b936b87f799f80dde60c2fd8d11c1128cebb`), ~3 cycles/output register,
not gated to xe35. **Vendor that one function into a probe and A/B it** — do not
FetchContent. Pre-committed kill criterion: **the reorder arm must drop INT4
unpack `kernel` by >2×; neutral or negative closes sycl-tla permanently.** It
would apply to INT4 (21.5 % of traffic). Footgun to record if anyone ever adopts
it: its CMake does `add_compile_definitions(SYCL_INTEL_TARGET)` bare (⇒ `1`) while
its sources branch on the *value*, so `SYCL_INTEL_TARGET=20` must be defined
explicitly or xe35 paths are silently selected.

---

## 6. The one thing worth your box

Nothing above needs your machine. If you have a quiet moment, this is the only
measurement whose absence is a real gap — and it is a gap in **this report**, not
in the tree:

```
# The clone-vs-shipped ratio measured on THIS tree, on YOUR card, at 27B TP=2.
# Everything above came from the 0.8B-shaped probe with synthetic weights.
scripts/bench_reports_batch.sh <batch> 2
```

The probe uses synthetic weights at the exact 27B shapes. Real quantized weights
have different byte distributions (99.6 % nonzero synthetic vs the real
distribution) which could in principle change the dequant's branch behaviour. The
interleaved ratio is the claim that carries the batch; **it has not been confirmed
against the real 27B model.** That is the honest limit of what was measured here.

```
RESULT:
```

---

## 7. What the tooling now refuses

| guard | what it stops |
|---|---|
| `scripts/check_hw_constants.sh` + `mutate_hw_constants.sh` (8/8 states) | a hardware capability figure in a doc without a `CITATION(...)` **in the same paragraph** — the exact shape of the 512-Xe-core error |
| `scripts/model_byte_roofline.sh` | a hand-written per-role byte table, the kind that summed to 25.5 GB against its own stated 19.5 GB |
| `scripts/test_roofline_probe.sh` (new gate; population 11 → 12) | a roofline number that is not tracking traffic — its own twin fails the probe rather than the run |
| `scripts/check_flag_gate_coverage.sh` | opt-in ungated went 8 → 7; `INTFER_WEIGHTS_DEVICE` is now named by a gate at TP=1 and TP=2 |
