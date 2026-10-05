# Batch: the M=1 INT2 GEMV bisect — two candidates falsified, two arms landed but UNMEASURED

> ## ==> READ THE STATUS COLUMN BEFORE THE NUMBERS. <==
>
> The largest measured deficit in this engine is that the shipped INT2 decode kernel
> runs **5.6–6.4× slower than a faithful clone of its own algorithm**. Two candidate
> causes have been **measured and falsified**. Two more single-change arms were
> **written, wired and gated — but not measured**, because one probe translation unit
> crosses the level-zero resource limit. **~4.5–5× of the gap is still unexplained and
> nothing here claims otherwise.**

**Date:** 2026-10-06 · **Box:** 2x Arc Pro B580 · **Commits:** see §6
**Measurement rule obeyed throughout:** no model bound, device event timestamps, medians
over interleaved reps, **ratios not absolutes**, no tok/s, `run_ten_gates.sh` NOT run.

---

## 1. The starting measurement

`examples/roofline_probe` (gated by `scripts/test_roofline_probe.sh`) runs a faithful
clone of the shipped algorithm — same group loop, same 8-code chunk, same byte
indexing, same 4-entry table, same ragged tail — in its own TU, on the same buffers,
every variant **interleaved** with the shipped kernel inside one rep loop:

| shape | clone | shipped | ratio |
|---|---|---|---|
| MLP_GATE (INT2) | 260.9 GB/s | 40.8 GB/s | **6.39×** |
| MLP_DOWN (INT2) | 77.6 GB/s | 13.8 GB/s | **5.62×** |
CITATION(roofline_probe, device 0, interleaved)

CITATION(roofline_probe, device 0, device event timestamps, interleaved, medians over 12
reps. Ratios are the claim; the absolutes move with box load.)

**Interleaving is forced by measurement, not taste.** Run separately, the shipped arm
read **17.2, 39.3 and 41.3 GB/s** on three consecutive runs of the same binary on this
contended box CITATION(roofline_probe, three consecutive runs of the same binary).
A number measured minutes from its comparator is not a comparison.

---

## 2. FALSIFIED: the module theory (1.26× against a ≥2× criterion)

| arm | MLP_GATE | vs shipped |
|---|---|---|
| `table` (tidy rewrite) | 193–244 GB/s | 6.3× |
| `VERBATIM` (**shipped source**, small TU) | 38.8 GB/s | **1.26×** |
| `VERB+TPL` (shipped source, `use_int2_tab` a template param) | 39.1 GB/s | 1.28× |
| `SHIPPED` (shipped source, 2378-line TU) | 30.7–40.8 | 1.00× |
| | CITATION(roofline_probe, device 0) | |

The GB/s ranges are ACROSS runs because this box is contended -- the RATIOS are
the claim, never a single run's absolute.

* **Module theory: DEAD, by its own pre-committed criterion.** The plan said "≥2× or the
  theory is dead". Moving the shipped source to a small TU buys **1.26×**.
* **Runtime-branch theory: DEAD, measured.** `use_int2_tab` is a host-read bool guarding
  a two-copy inner loop *inside* the per-group loop, which runs **320× per work-item** at
  MLP_GATE. Making it a **template parameter** — so `if constexpr` deletes the branch
  *and* one loop body — measures **1.007×**.

So ~4.5–5× sits in the **shared text** of those two functions.

---

## 3. FALSIFIED ON MY OWN HYPOTHESIS, before I asserted it

Two candidate causes for the remaining gap, both tested on the CPU against the kernel's
own arithmetic **before** being asserted:

| candidate | verdict | how |
|---|---|---|
| `step = row_lanes*kVSg = 128` while a group is only `Gw = 16` wide, so the chunk loop overshoots and **half of every group is dropped** | **FALSE** | Reproducing the phase formula `k = kg0 + ((off − (kg0 % step) + step) % step)` over the 8-group period gave **128 of 128 k-values covered and all 16 lanes contributing**: group *g* is served by lanes *2g* and *2g+1*, cycling with period 8 |
| a lane offset in `code_u8[b0]` — the same shape as the `INTFER_DECODE1_ROWPAIR` double-offset that **lost the device** | **FALSE** | it indexes the **unoffset** `code_u8` with **absolute** byte offsets in both the chunk loop and the tail, which is correct. ROWPAIR's bug was the opposite shape |

Recorded because the first was the most plausible story I had and I was about to
"fix" a working decomposition. **A cheap CPU reproduction of the formula is worth more
than an argument about it.**

---

## 4. LANDED BUT UNMEASURED: `examples/roofline_probe_bisect`

### E-BISECT — `INTFER_*`: **none. This is a measurement, not a flag.**

The two blocked single-change arms, each differing from `VERBATIM` in exactly ONE
place:

| arm | the one change | why it is a candidate |
|---|---|---|
| `VERB+HALF` | `from_half_bits` (the engine's branchy **scalar** FP16→FP32 twiddle, 3 branches per call) → `sycl::bit_cast<sycl::half>` | called once per GROUP ⇒ **320× per work-item** at MLP_GATE |
| `VERB+MP` | weight side read through a `multi_ptr` instead of a raw `code_u8[b0]` | a multi_ptr lets the compiler keep the base in a register and fold the addressing |

Arms in the binary: `table` (control, shared with the other probe), `VERBATIM`
(shipped source), `VERB+HALF`, `VERB+MP`, `SHIPPED` — all interleaved per rep.

**Why a second binary rather than a smaller ladder:** one probe TU crosses level-zero
`UR_RESULT_ERROR_OUT_OF_RESOURCES` (40) at roughly **six to eight INT2 kernel bodies**.
The ladder was split, with the control and the shipped arm shared, so each binary can
report a **within-binary ratio** and cross-binary absolutes are never compared.

### Status: BUILT, WIRED, GATED — **AND NOT MEASURED**

`scripts/test_roofline_bisect.sh` exists, has a `dispatch()` arm and a `STEPS` entry,
and asserts that the twin and **all five arms** ran. It has **never reached a green
run**, for a reason I have not isolated:

* every arm **hangs**, including `table` — and testing each arm **alone** showed all
  five hang, so the fault is **not** in the shipped call;
* the sibling `roofline_probe` completes in **6.3 s** on the same box at the same
  moment, so it is **not** a GPU-wide outage;
* my repeated timed-out runs wedged the driver's
  `kworker/*+xe_page_fault_work_queue` at 61 % CPU, which is a documented box
  condition (`scripts/gpu_recover.sh`; the remedy is `rebind`, which needs sudo).

**The wrong reading is recorded in the source**, because a hang invites it: I first
read the hang as "the shipped kernel is broken here too", on no evidence. And the
first draft genuinely *did* have a buffer bug — it passed the output buffer as the
activation buffer — which I fixed, and which was still not the cause.

**I am not claiming a result from this binary.** The two arms' ratios are unknown.

---

## 5. What a flag would need, if an arm ever wins

Pre-committed so the next session does not re-derive it, and so a win is not shipped
without its certification:

| requirement | why |
|---|---|
| default **OFF**, kill switch `INTFER_NO_*` | AGENTS.md's opt-in convention |
| body **INLINE at its call site** | the reverted `quant_gemm_fused_decode1` cost **231 114 ms** with arithmetic a normalized diff proved identical; a shared device function is fatal here |
| `dispatch()` arm **and** `STEPS` entry in `run_all_checks.sh` | an arm alone never runs (audit #18) |
| mutation matrix that mutates the **artifact** | rule 11: a checker correct by construction is unprovable by weakening itself |
| stream-identity arm in `scripts/test_tp2_27b.sh` | the only gate whose 27B model has INT2 tensors (the 0.8B model has **zero**) |
| numerics contract | `VERB+HALF` changes the scale conversion and `VERB+MP` changes addressing, so **neither is bit-exact by construction** — unlike `INTFER_DECODE1_ROWPAIR`, which is measured bitwise |

The last row is the difference that matters: row-pair's claim was *measured*
(227 835 element values bitwise identical), these two would have to be *measured the
same way*, and no arithmetic argument substitutes for it.

---

## 6. Also landed this batch

* `INTFER_DECODE1_ROWPAIR` is **green and measured**: 27B TP=2 ids match base, and
  `scripts/test_rowpair_equiv.sh` measures **227 835 element values BITWISE IDENTICAL**
  to the base M=1 kernel over 8 shapes × 5 datasets in two process invocations.
  It was not green until this session — it lost the device on both INT2 shapes while
  passing FP16/INT4, because the branch is guarded on `bits == 2`.
* `INTFER_DECODE1_SUBGROUP` is **quarantined** (wrong output, cause not located; two
  candidates falsified first) — `reports/2026-10-05-decode1-subgroup.md`.
* `scripts/mutate_decode1_copies.sh`: **5/5 states caught**, plus two **excluded with
  reason** (the UNCOVERED rule and the required set share one list, so they cannot be
  tested independently). It earned its place by catching the enumeration labelling
  row-pair `COVERED-BY-ATOM-PIN` while only a **floor** backed that label.

## 7. Operator commands

```
# the two arms this batch could not measure (no model, ~1 min, counters only)
bash scripts/test_roofline_bisect.sh

# the UNPROVEN stack step. ONLY THE OPERATOR: AGENTS.md standing rule 1 forbids an
# agent session from run_ten_gates.sh (tens of minutes of model loads). The stack is
# at failed=0 stale=0 unproven=1 and this is the one item that closes it.
bash scripts/run_ten_gates.sh

# if the bisect binary hangs again, the box's page-fault worker is wedged
bash scripts/gpu_recover.sh diag && sudo bash scripts/gpu_recover.sh rebind
```

```
RESULT: E-BISECT/VERB+HALF   (blank -- not measured; see section 4)
RESULT: E-BISECT/VERB+MP     (blank -- not measured; see section 4)
```