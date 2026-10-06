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

### Status: LANDED AS ENGINE FLAGS, CORRECTNESS-GATED — **AND NOT MEASURED**

The probe could not run, so the two arms were landed **in the engine** where the
operator can measure them on real weights in one pass:

| | E-VERBHALF | E-VERBMP |
|---|---|---|
| **flag** | `INTFER_DECODE1_VERBHALF` (presence-only) | `INTFER_DECODE1_VERBMP` (presence-only) |
| **kill switch** | `INTFER_NO_DECODE1_VERBHALF=1` | `INTFER_NO_DECODE1_VERBMP=1` |
| **default** | **OFF** | **OFF** |
| **guard** | `bits == 2 && use_int2_tab` | same |
| **the one change** | group scale via `sycl::bit_cast<sycl::half>` instead of `from_half_bits` | weight side through a row-base `multi_ptr` |
| **body** | **INLINE at the call site**, a separate `parallel_for` | same |
| **numerics** | **NOT bit-exact by construction** — `from_half_bits` returns 0 for a zero/subnormal exponent and Inf/NaN when saturated, while `bit_cast<half>` hands a denormal to the FP32 convert which flushes to zero under default FTZ. Stream-identity only, same class as `INTFER_DECODE1_SPLITK` | **bit-exact — NOW MEASURED, NOT ARGUED** (see below) |
| **gate arm** | `scripts/test_tp2_27b.sh`, TP=2 | same |
| **CORRECTNESS measured** | `test_rowpair_equiv.sh`: bitwise identical to base on these datasets — which does **not** prove the FTZ concern away, only that these synthetic scales never reach it | `test_rowpair_equiv.sh`: **BITWISE IDENTICAL to base on 227 835 element values**, 8 shapes × 5 datasets |
| **buckets I measured (perf)** | **none** — not run | **none** — not run |
| **expected direction** | **UNKNOWN**, not positive | **UNKNOWN**, not positive |

### `VERBMP`'s bit-exactness is now MEASURED, not argued

Its source claim was *"bit-exact by construction — a claim, not a verified
property"*, because the `ROWPAIR` precedent is that a bit-exactness **argument** is
worth nothing until measured: `ROWPAIR`'s is precisely what shipped a
device-losing double offset. `scripts/test_rowpair_equiv.sh` was extended from two
orders to four (one process each — the flags are read into a function-local `static
const bool`, so an order is a process, not a loop) and now reports:

```
ok  INTFER_DECODE1_VERBMP  is BITWISE IDENTICAL to base (227835 element values)
ok  INTFER_DECODE1_VERBHALF is bitwise identical to base on THESE datasets
    -- which does NOT prove the FTZ/denormal concern away, only that these
    synthetic scales never reach it.
```

So `VERBMP`'s correctness claim is settled **without a model and without a 27B
load**. `VERBHALF` being identical here is information, not exoneration: the
differing case is a zero or subnormal group scale, and these synthetic scales
(`to_half_bits` of ~1.0–1.5) never produce one. **Performance is still unmeasured
for all three.**

**Why inline and not a shared device function:** the reverted
`quant_gemm_fused_decode1` cost 231 114 ms with arithmetic a normalized diff proved
identical, and the generalised cause is that the SYCL device compiler failed to
optimise an extracted body. Each arm is a separate `parallel_for` with its own copy
of the loop.

**Why `VERBMP`'s indices are written the way they are:** it is the *shape of the bug
that already shipped*. `INTFER_DECODE1_ROWPAIR` used row-**offset** `multi_ptr`s
indexed with the base kernel's **absolute** byte offsets — a double offset that read
past the shard and **lost the device** (`UR_RESULT_ERROR_DEVICE_LOST` on both INT2
shapes, while passing FP16 and INT4 because the branch is guarded on `bits == 2`). So
`VERBMP`'s chunk loop uses **row-relative** `b0` and its tail deliberately uses the
**unoffset** `code_u8` with absolute indices, exactly as the base kernel does.

**How they are pinned:** `scripts/check_decode1_copies.sh` now enumerates both new
bodies (`covered-by-subset`) *and* asserts their **shift-into-mask set EQUALS the
general body's** — an equality, not a subset, computed with a deliberately loose
extractor that matches the **wrong** form too. That looseness is load-bearing: the
first version of this check matched only the correct `((j & 3) * 2)`, so planting the
real `INTFER_DECODE1_SPLITK` bug simply stopped matching and the checker passed.

### Status of the PROBE: BUILT, WIRED, GATED — **AND NOT MEASURED**

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

**The gate's OBSERVED return status, measured rather than predicted:**

```
$ bash scripts/test_roofline_bisect.sh          # ~2 min
GATE_RC=2   SKIP: ... did not finish in 120s at HEAD (rc=124)
            UNPROVEN: bisect probe cannot run at HEAD (not a regression)
```

**The FIRST version of this gate was much worse and that is worth recording.**
`timeout 300 bash scripts/test_roofline_bisect.sh` returned **rc=124 with EMPTY
OUTPUT**, because its internal timeout was the same 1800 s the runner allows. Three
defects in one: rc=124 is not in the runner's vocabulary (`expect_healthy` reads 0, or
2 *with* a skip diagnostic; anything else is "a gate must not go red on a healthy
box" → a regression), so an unrunnable probe **masqueraded as a code failure**; the
empty output meant not even the `need` assertions had anything to match; and it
consumed the runner's entire budget on a step that cannot finish. It now self-bounds
to 120 s, refuses to start if `INTFER_BISECT_TIMEOUT` is at or above the runner's
1800 s, and exits **2 = UNPROVEN** with a `SKIP:` block naming the isolation and the
consequence.

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

# *** THE STACK IS RED AND THE CAUSE IS NOT ESTABLISHED. DO THIS FIRST. ***
# Current, MEASURED -- the "failed=0 stale=0 unproven=1" that an earlier revision of
# this report carried was measured BEFORE the bisect gate existed and was quoted
# afterwards as if current, which is a citation presented as a claim:
#     INTFER_QUICK=1 ./scripts/run_all_checks.sh
#     rc=1  verdict=FAIL  failed=2  stale=24  unproven=2  quick=1
#       check_device_precond.sh  FAIL rc=1  (4 assertions)
#       check_e2_ctx.sh          FAIL rc=1  (2 violations)
# With NO flags set the base 27B TP=2 run emits
#   "The capital of France isrowned_x财神"   prefill token 42624, expected 11751
# CITATION(measured 2026-10-06:
# build/device_precond_matrix/test_tp2_27b.healthy.log line 2, saved with the rest to
# build/boxstate/diag_2026-10-06.txt)
#
# THE BOX ATTRIBUTION IS PLAUSIBLE AND NOT CONFIRMED, and the evidence cuts BOTH
# ways, so both sides are recorded rather than the convenient one:
#   FOR:    every failure is a model-running gate and the BASE run with no flags is
#           the wrong one; kworker/*+xe_page_fault_work_queue was observed at 61% CPU
#           on ONE sample.
#   AGAINST: both render nodes present and zeDeviceGet reports 2 devices
#           CITATION(scripts/gpu_recover.sh diag ->
#           build/boxstate/diag_2026-10-06.txt); the page-fault worker is NOT
#           spinning at the next sample, so 61% is point-in-time and does not
#           reproduce; and examples/roofline_probe PASSES its always-red twin (0.23 vs
#           an expected 0.25) at a 452 GB/s ceiling
#           CITATION(build/boxstate/probe_now.txt), so GPU compute is behaving.
#           STRONGEST: the ENGINE HAS NOT CHANGED since the tp2 gate passed --
#           `git log a6c2c73..HEAD -- src/ include/ examples/intfer_cli/` is EMPTY,
#           a6c2c73 is the SUBGROUP quarantine, and `bash scripts/test_tp2_27b.sh`
#           returned PASS rc=0 immediately after it. Everything since is docs,
#           scripts/, and one example binary no gate path invokes.
# That points away from this session's code but does NOT prove it: a gate that passes
# quiet and fails loaded is exactly the failure mode in question.
sudo bash scripts/gpu_recover.sh rebind && bash scripts/gpu_recover.sh verify
bash scripts/check_device_precond.sh
# RE-MEASURED on ad71739 (after cf63f1a landed the two arms), QUICK mode:
#   rc=1 verdict=FAIL failed=3 stale=24 unproven=2 quick=1
#     check_device_precond.sh  FAIL   <- still, no flags, base emits garbage
#     check_engine_cert.sh     FAIL   <- STALE: cf63f1a changed engine code, so the
#                                         certification must be re-run. Expected, and
#                                         the remedy is operator-only (run_ten_gates.sh
#                                         is forbidden to an agent session).
#     check_e2_ctx.sh          ok     <- was FAIL at a0e3399 and RECOVERED WITH NO
#                                         rebind. Evidence for the load/box reading
#                                         and against "my changes broke it": the
#                                         engine is FURTHER from the last known-good
#                                         commit now, and e2_ctx recovered anyway.
# AND IN THE SAME WINDOW, on this same box, examples/rowpair_equiv measured 227835
# element values BITWISE IDENTICAL -- so the GPU produces correct INT2 GEMV results
# right now, which is contrary evidence against the box attribution for e2_ctx.
# *** IF THOSE STILL FAIL AFTER A CLEAN REBIND, THIS IS A CODE INVESTIGATION AND THE
# *** BOX ATTRIBUTION IS RETRACTED. Recorded now, before the evidence exists, so it
# *** cannot be quietly dropped once a convenient run comes back green. ***

# the engine gate, after the box is healthy. ONLY THE OPERATOR: AGENTS.md standing
# rule 1 forbids an agent session from run_ten_gates.sh (tens of minutes of loads).
bash scripts/run_ten_gates.sh
```

```
# E-VERBHALF / E-VERBMP, 27B --tp 2, load1 < 2. Counters only; never a tok/s claim.
# Expected direction is UNKNOWN for both (see the table), so a red kernel bucket is
# as informative as a green one.
INTFER_DECODE1_VERBHALF=1 bash scripts/run_ten_gates.sh
INTFER_DECODE1_VERBMP=1   bash scripts/run_ten_gates.sh
```

```
RESULT: E-VERBHALF  kernel=__ ms/token (baseline __)  nk/nw/nr = __/__/__  [win|neutral|loss]
RESULT: E-VERBMP    kernel=__ ms/token (baseline __)  nk/nw/nr = __/__/__  [win|neutral|loss]
```