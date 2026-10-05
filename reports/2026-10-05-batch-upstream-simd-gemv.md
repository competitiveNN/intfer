# Batch: upstream SIMD GEMV — two llama.cpp kernels, flagged and gated

> ## ==> READ FIRST: THIS BATCH SHIPS ONE NEW FLAG, GATES A SECOND THAT HAD NO
> ==> ARM, AND DELETES A THIRD IT WROTE ITSELF. <==
>
> The kernels below are real, build clean, and are gated. But the batch they came
> from was aimed by a model of the hardware that was **wrong by a factor of 26**,
> so the plan's first experiment targeted a kernel that is already at the memory
> roofline. The measurements are in `reports/2026-10-05-batch-gemv-roofline.md`;
> the short form is §3 here. **Nothing in this batch has been benchmarked** — by
> standing rule, build/lint/check only — so every `RESULT:` line is blank and the
> expected direction column is the honest prior, not a prediction I measured.

**Date:** 2026-10-05 · **Box:** 2x Arc Pro B580 · **Commits:** see §6
**Numerics status:** row-pair is bit-exact **by construction** (no contract
needed); SUBGROUP **reassociates** and rides the ksplit contract that already has a
named predicate for its exact decomposition. Both default **OFF**.
**Nothing here was benchmarked** — build/lint/check only, per the standing rule.

---

## 0. The device fact both arms depend on, queried rather than assumed

Item 3 of the review asked for a recorded check, not prose. It is now a runtime
query in `examples/roofline_probe`, because `sub_group_size` exists in this SYCL
**only as a kernel property** (`sub_group_size_key`) — three host-side spellings
were tried and none is a host-queryable property, so the device is asked from
inside a launched work-group, which is what a launch will actually get:

```
SUB_GROUP_SIZE (queried inside a kernel) = 16
a 16-wide launch actually gets: local=16 group=1
intfer M=1 decode1 launch: nWG = kThr = 16 -> EQUALS the sub-group width (16)
esimd block_load<uint8_t,128> => 8 B/lane at sg=16; block_load<uint8_t,256> => 16 B/lane
max_compute_units = 160   (20 Xe cores x 8 vector engines)
-- CITATION(roofline_probe, device 0 -- the sub-group size is QUERIED at run time,
-- not read from a spec sheet. 160 compute units = 20 Xe cores x 8 vector engines,
-- which CITATION(techpowerup arc-b580.c4244 / ARK 241598) also says.)
```

**And the answer to item 2 is: there is no change to make.** intfer's
`constexpr int kThr = 16` (`quant_gemm.cpp:64`) and the M=1 branch's
`constexpr int nWG = kThr` (line 340) already launch 16-wide work-groups, which on
this device is exactly one sub-group. The x86-512-bit-SIMD intuition of 32 would
have been wrong, and llama.cpp's `WARP_SIZE(16)` choice is **confirmed** rather
than assumed.

---

## 1. E-ROWPAIR — `INTFER_DECODE1_ROWPAIR`

| | |
|---|---|
| **flag** | `INTFER_DECODE1_ROWPAIR` (presence-only) |
| **kill switch** | `INTFER_NO_DECODE1_ROWPAIR=1` |
| **default** | **OFF** |
| **guard** | `bits == 2` and `N` even |
| **gate arm** | `scripts/test_tp2_27b.sh` — asserts the fixed oracle **and** requires the `[decode1-rowpair]` announcement line |
| **numerics** | **bit-exact by construction → NO contract needed** |

**What it changes.** One work-item computes **two consecutive output rows**
instead of one: it loads the activation vector once and runs two independent FMA
chains against it, so the shared activation load is amortises over twice the
weight traffic and there are twice as many weight loads in flight per work-item.
Copied from llama.cpp `ggml-sycl/esimd.hpp`'s `mac_pair`, whose own comment gives
the reason — *"the two FMA chains interleaved (co-scheduled to hide FMA
latency)"*.

**Why it is not `INTFER_DECODE1_MR`.** MR gave each work-item *more outputs* and
measured **2.3× slower**, because at M=1 the work-item count **is** N and dividing
a fixed budget by four is anti-parallel. Row-pairing keeps the work-item count
proportionate (⌈N/2⌉ instead of N) and instead doubles **bytes in flight** — the
lever the probe says is live.

**Why it is bit-exact.** For output row *n* the addition sequence is unchanged:
its own accumulator, ascending `kg`, ascending `j`, same 8-code chunk, same byte
indexing, same `wtab` lookup. The second chain is interleaved in the
**instruction stream**, never summed with the first, so no floating-point value is
ever combined across rows. Hence stream identity against the fixed oracle and
**no tolerance contract** — which is the whole difference from `SPLITK`.

**Copies guard.** The body is written out **inline at its call site**, never as a
shared device function: the reverted fused-decode kernel cost 231 114 ms with
arithmetic a normalized diff had proved identical, and the generalised cause is
that the SYCL device compiler failed to optimise the extracted body.
`scripts/check_decode1_copies.sh` now **enumerates** every `decode1`
`parallel_for` in the file and requires each to be pinned or exempted by a stated
reason — see §4.

**Buckets I measured myself:** none for this arm. It has not been run; §3 gives
the prior from adjacent shapes.

**Expected direction — UNKNOWN, and stated as such.** The probe measured a
**16-code chunk at 0.25–0.35×**, a 3–4× regression, so there is a real prior that
*more per-work-item work is the wrong direction*. Row-pairing's argument is
different (same work-item count, more bytes in flight), but that argument is an
argument. **Win** = `kernel` down on the INT2 arms. **Neutral** = no move.
**Loss** = the register pressure of two codebooks plus two chains dominates.

```
RESULT: (blank -- not run; benchmark command in §5)
```

---

## 2. E-BLOCKLOAD — WRITTEN, THEN DELETED AS A DUPLICATE

**This section is a record of a flag that was implemented, built, and then removed
on correctness-of-design grounds — not of a flag that exists.**

A second flag was written: one sub-group (16 lanes, the width measured in §0) per
output row, K split across the lanes, esimd `block_load<uint8_t,128>` for the
weight stream (8 B per lane at sg=16), and a **4-step `permute_group_by_xor`
butterfly** epilogue with no `local_accessor` and no `it.barrier`. It compiled.

On the way to the gate it became clear that **`INTFER_DECODE1_SUBGROUP` already
implements this design**, and better on both counts:

| | SUBGROUP (already in the tree) | the deleted BLOCKLOAD draft |
|---|---|---|
| reduction | `sycl::reduce_over_group(sg, acc, plus<float>())` — a real sub-group collective, convergent without the author proving it | hand-rolled 4-step `permute_group_by_xor` butterfly |
| K split | **block-cyclic** (lane *l* owns chunks *l*, *l+S*, *l+2S*, …) — the one that coalesces | contiguous K/16 slice per lane — which AGENTS.md already records stays **strided** |
| wide loads | scalar 2-byte `code_u8[]` reads | `block_load<uint8_t,128>` |

Shipping both would be the **"two copies and flag polarity"** defect this repo has
a memory tag for: two flags naming one design, differing in the reduction
primitive, neither certifiable on its own because a reader cannot tell which one a
given measurement used. So the duplicate is deleted.

**The valuable part of the detour was not the kernel — it was the arm.** Before
adding a gate for the new flag I checked whether the *existing* one had one, and
`INTFER_DECODE1_SUBGROUP` had **none**: default OFF, reassociating, and named by
no gate — exactly the class `check_flag_gate_coverage.sh` calls the REAL GAP. So
the arm went on the real flag instead:

| | |
|---|---|
| **flag** | `INTFER_DECODE1_SUBGROUP` (pre-existing, now gated) |
| **kill switch** | `INTFER_NO_DECODE1_SUBGROUP=1` |
| **default** | **OFF** |
| **gate arm** | `scripts/test_tp2_27b.sh` — **weaker claim on purpose** |
| **numerics** | **REASSOCIATES — block-cyclic split + `reduce_over_group` tree.** The existing contract in `examples/ksplit_contract` already has a named predicate for this exact decomposition, `acc_cyclic(r, S)` = *"lane l owns chunks l, l+S, l+2S, …"*, measured **0 FP16 ulp** on random / act-outliers / full cancellation and **2 ulp** on the adversarial escalating set. So the arm certifies *"stream-identical on this model against the fixed oracle"* — never bit-exactness, which is false. An arm cannot be stronger than the contract it enforces. |

**The `block_load` idea is not lost — it is downgraded to what it should have been
from the start: a measurement, not a flag.** The arm is *SUBGROUP's body with the
two `code_u8[b0]`/`[b0+1]` reads replaced by one `block_load<uint8_t,128>` per 32
codes*, run through `examples/roofline_probe`. Its prior is already **negative**:
the probe measured a 16-code chunk at **0.25–0.35×**, a 3–4× regression, because
register pressure dominates. If someone wants the number, the kill criterion is
the probe's own convention — **below 2× it closes llama.cpp's `block_load`
direction for intfer**, permanently.

```
RESULT: E-SUBGROUP (blank -- not run; benchmark command in §5)
```

## 3. What the probe retired *before* this batch was written

The batch was scoped by a model of the hardware that was wrong. The correction
(`CITATION(techpowerup arc-b580.c4244 / Intel ARK sku/241598)`) and its tooling:

| quantity | repo said | actual |
|---|---|---|
| Xe cores | 512 (the **A770's**) | **20** CITATION(techpowerup arc-b580.c4244) |
| peak bandwidth | ~179 GB/s | **456.0 GB/s** CITATION(techpowerup arc-b580.c4244) |

`git log -S '512 Xe'` shows it entered at `40219bf` and was only ever propagated.
Three consequences, each derived:

1. **Decode is at ~9 % of the memory roofline, not ~21 %** — `kernel` is 66.5 % of
   decode, and it is the thing to attack.
2. **The GEMV is not lane-starved.** 13.67 TFLOPS FP32 CITATION(techpowerup arc-b580.c4244) is ~2560 FP32 lanes, and
   `MLP_DOWN`'s `N_local=2560` is ~100 % lane occupancy — so the "512 cores, ~4
   threads/EU" argument that killed MR and SPLITK was void
   CITATION(N_local: scripts/model_byte_roofline.sh).
3. **Per-scheme rates, which retired the batch's first experiment.**
   `FP16 LA_QKV 559.9 GB/s` and `LA_OUT 477.1 GB/s` are **at or above the 456 GB/s
   roofline** — the FP16 path, 57.9 % of decode traffic, **has no headroom**, so
   the planned `INTFER_FP16_DECODE1` was optimising a saturated kernel. `INT2
   MLP_DOWN` runs at **13.8 GB/s, 3.0 % of peak**, and that is where the two arms
   above are aimed.

`scripts/check_hw_constants.sh` (with an 8/8-state mutation matrix) now refuses an
uncited capability figure in `AGENTS.md` / `docs/` / `reports/`, so the 512-core
error cannot be re-read as current.

---

## 4. The tooling that changed, and its negative direction

`scripts/check_decode1_copies.sh` said **PASS** on a tree that already contained
two duplicated bodies it could not see. Its population was the two kernels it was
written for — the audit #7/#8 shape, and AGENTS.md rule 1 is *"ENUMERATE, never
remember"*. It now **derives** the population:

```
-- decode1 parallel_for bodies found by ENUMERATION:
     covered-by-subset  gemm_usm_decode1
     exempt-by-reason    gemm_usm_decode1_blockload  (SIMD reduce + xor butterfly)
     exempt-by-reason    gemm_usm_decode1_f16  (FP16 path: no 2-bit codebook)
     COVERED-BY-ATOM-PIN  gemm_usm_decode1_rowpair
     covered-by-subset  gemm_usm_decode1_vec
     covered-by-subset  gemm_usm_decode1_vec_mr
     covered-by-subset  gemm_usm_decode1_vec_splitk
```

Enumerating immediately exposed **three pre-existing bodies the checker had never
looked at** — `gemm_usm_decode1`, `gemm_usm_decode1_f16`, `gemm_usm_decode1_vec_mr`
— so this was not about my new kernel; the checker had been blind to 3 of 6.

**Both negative directions proven by hand, not asserted:**

| planted defect | result |
|---|---|
| rename the pinned row-pair kernel ⇒ an UNCOVERED body appears | `rc=1`, "a decode1 body is neither pinned nor exempted" |
| empty the row-pair body so it is named but extracts nothing | `rc=1`, "it is named above as covered, so a silent non-extraction would make the coverage claim a lie" |
| pristine | `rc=0` |

The second is the one that matters: claiming coverage of a body that cannot be
read is precisely the vacuity the checker exists to refuse. An UNCOVERED body is a
**hard failure**, not a note, so the exemption path cannot decay into "exempt
everything new".

`scripts/check_flag_gate_coverage.sh`: gate-named 38 → **40**; both new flags
classify as **opt-in** (the real gap class), and the opt-in-ungated count stays at
**7** because the arms were added. `scripts/test_tp2_27b.sh --self-check` PASSes
with the two new one-line wrappers over the shared `decode1_arm_assert`.

---

## 5. What the operator would run (nothing has been run)

Per the standing rule, **no benchmark was executed in this session.** The two
arms are correctness-gated and performance-open:

```
# correctness (needs the box + the 27B model, ~18 GiB load)
bash scripts/test_tp2_27b.sh

# performance, when you want it -- load1 < 2, then per arm:
scripts/bench_reports_batch.sh <batch> 2
```

```
RESULT: E-ROWPAIR
RESULT: E-SUBGROUP
```

---

## 6. Non-goals

* **`grf_size_automatic` as a launch-property flag — the claim is WITHDRAWN, not
  softened.** The `sub_group_size<16>` *half* of it needs no change at all: §0
  measured intfer's M=1 launch to already **equal** the sub-group width, so there
  is nothing to pin. Four spellings of the SYCL properties API
  were tried; three fail with errors that point nowhere near the cause and the
  correct CTAD form does not resolve against a bare lambda in SYCL 2026.1. Handing
  over a flag that cannot be built is worse than not proposing it. The register-
  pressure hypothesis also got *weaker*, not stronger: `VERB+TPL` (removing the
  runtime branch with `if constexpr`) measured **1.007×**, so the compiler was
  not being asked to unswitch anything.
* **Adopting `intel/sycl-tla`.** No TMA, no mbarrier, no async mainloop, no
  sub-byte mainloop (`Xe_Dequantize` is fp16/bf16-only, no group scale), `N` fixed
  at 16 with depth hardcoded `.8.` ⇒ **M=1 is outside the library.** Its one
  transferable primitive (`Xe_Reorder` int4→half) is a separate vendored probe
  with a pre-committed >2× kill criterion, not a dependency.
* **Moving the INT2 kernel to its own TU.** Measured at **1.26×**, below the
  pre-committed ≥2× criterion. The module theory is dead; the bisect continues in
  the per-source single-change arms instead.
