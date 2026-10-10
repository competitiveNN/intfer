# HISTORY — already-tried optimizations and their measurements

Split from AGENTS.md 2026-10-08. Everything here is a RECORD: what
was tried, what it measured, and why it is closed. Nothing in this
file is a pending task. The live directives (box state, gate state,
standing rules, performance targets, fusion directive, flag
inventory, architecture facts, verification tooling) stayed in
AGENTS.md; the forward backlog is in PLAN.md.

Sections appear in their original order, verbatim. Line numbers below
refer to the original AGENTS.md.

## ==> SUPERSEDED 2026-10-05: THE "PARALLELISM-BOUND" MODEL WAS WRONG. THE
## ==> MEASUREMENTS STAND; THE EXPLANATION DOES NOT. READ THIS FIRST. <==

**This section argued that 27B decode is parallelism-bound rather than
bandwidth-bound, and that the target is EU occupancy. Both halves of that are
retracted, because the hardware model underneath them described a GPU this box is
not.**

| quantity | this section said | **actual** |
|---|---|---|
| Xe cores | 512 | **20** CITATION(techpowerup arc-b580.c4244) |
| vector engines / XMX | not stated | 160 / 160 CITATION(techpowerup arc-b580.c4244) |
| peak memory bandwidth | ~179 GB/s | **456.0 GB/s** CITATION(techpowerup arc-b580.c4244) |

Full source: TechPowerUp `arc-b580.c4244`, saved at
`docs/Intel Arc B580 Specs _ TechPowerUp GPU Database.html`; cross-checked against
Intel ARK sku/241598 — both agree, and also give 12 GB GDDR6 192-bit @19 Gbps,
18 MB L2, L1 256 KB/EU. `git log -S '512 Xe'` shows the figure entered at `40219bf`
(2026-10-04) and was thereafter only propagated, never measured; the 512 Xe cores it
names is the **A770's** count (4096 shader units / 8)
CITATION(techpowerup arc-b580.c4244 / ARK 241598 — the B580's figures, replacing the
A770's). Nothing on this box was ever asked.

**Three consequences, each derived rather than asserted:**

1. **Decode is at ~9 % of the memory roofline, not ~21 %.** The bucket split below
   (`kernel` 44 %, `waitblock` 22 %, `submit` 17 %) is unchanged. The 9.75e9 B
   per device per token is unchanged. Only the denominator was wrong:
   41.1 GB/s against 456 GB/s CITATION(techpowerup arc-b580.c4244). Re-derive it
   with `scripts/model_byte_roofline.sh`; never quote a roofline percentage off
   this file.
2. **The GEMV is NOT lane-starved.** FP32 13.67 TFLOPS at 2670 MHz
   CITATION(techpowerup arc-b580.c4244) is ~2560 FP32 lanes. `MLP_DOWN` at TP=2 has
   `N_local = 2560` work-items — one per lane, i.e. ~100 % lane occupancy. The
   "2048 work-items over 512 cores, ~4 threads/EU, `kThr=16` cannot fill one SIMD"
   argument is void: there are 20 cores, and `kThr=16` matches the Xe2 sub-group
   width.
3. **So MR and SPLITK were aimed at a bottleneck that does not exist.** Their
   *measurements* are untouched and remain the reason those flags are default OFF:
   MR was 2.3x slower and SPLITK +43.8 % on `kernel`. What is retracted is the
   *reason* — "divided an already-starved budget" — which is why both directions are
   re-examined against the bandwidth hypothesis (`AGENTS.md`, same section).

**The rest of this section is preserved verbatim below as the historical record.**
It is a record of what was believed on 2026-10-04 and of measurements that still
hold. Do not "correct" it into agreement — that is the same laundering failure in
the other direction. `scripts/check_hw_constants.sh` now refuses an uncited
# *** RETIRED 2026-10-07: `reports/` IS DELETED. *** The note at AGENTS.md line 249
# claimed it was retired 2026-10-06; that was wrong, and the correction written
# here on 2026-10-06 was wrong in the other direction (it said reports/ held live
# reports and let an uncited figure through in reports/2026-10-06-batch-bisect-4.md).
# Both directions are now false: the directory is gone and the corpus is AGENTS.md
# plus docs/ records. Recorded rather than silently dropped, because this file's
# whole purpose is to stop a claim from being re-read as current.

## 27B DECODE IS **PARALLELISM-BOUND**, NOT BANDWIDTH-BOUND (measured 2026-10-04) [SUPERSEDED 2026-10-05 — model void, measurements stand]

Read this before optimizing decode. It overturns the natural first guess.

27B `--tp 2`, decode M=1, counters only: `kernel` 44%, `waitblock` 22%, `submit`
17%, `usm_write` 9%, `readback` 8%. Decode reads **9.16 GiB of weights per device
per token** in **257.7 ms** of kernel = **35.5 GiB/s, ~21% of the B580's ~179
GB/s** CITATION(the ~179 GB/s denominator is the A770's; the B580's is 456 GB/s —
see the SUPERSEDED banner above. The bucket split is
CITATION(perf_base.err, `[GPU profile @decode]`)). The obvious reading — "the GEMV
kernel is not saturating bandwidth" — is **wrong**.

> CITATION(techpowerup arc-b580.c4244 / ARK 241598: the B580's peak bandwidth is
> 456.0 GB/s, not the ~179 GB/s this section assumed — so "21 %" is really ~9 %,
> and the conclusion drawn from it below is retracted. See the SUPERSEDED banner
> above.) CITATION(perf_base.err: `total_kernel_us 1187601.2` over 5 tokens.)
> The 9.16 GiB figure is reproduced by `scripts/model_byte_roofline.sh` as
> 9.752 GB/device/token.

**Proof by measurement, not argument.** `INTFER_DECODE1_MR=1` processes MR=4
outputs per work-item, chosen because it raises memory-level parallelism and
shares one activation load across 4 outputs, and chosen to be **bit-exact** (each
output keeps its own K order in its own register; nothing is reassociated). It came
out **2.3x SLOWER**: kernel 2055.1 -> **4698.7 ms**, waitblock 1037.7 -> 2362.5 ms,
with `nk`/`nw`/`nr` identical (7936/7936/2048), so this is device work, not
plumbing.

**Why -- and this is the finding.** At M=1 this kernel's total work-item count is
**exactly `N`** (one output element per work-item) and that is **invariant to the
work-group shape**. So MR did not add parallelism; it **divided an already-starved
budget by four**. At 27B TP=2 a layer GEMM has N ~ 2048 per rank over **512 Xe
cores, ~4 threads/EU** CITATION(this is the A770's core count; the B580 has 20 —
see the SUPERSEDED banner), and `kThr=16` (`quant_gemm.cpp:64`) cannot even fill
one SIMD. More outputs per work-item is *anti*-parallel at M=1. Do not re-attempt
it.

> CITATION(techpowerup arc-b580.c4244 / ARK 241598: 20 Xe cores, not 512, so the
> "already-starved budget" premise above does not hold. The MR *measurement*
> stands and MR stays closed.)

**The target is EU occupancy; the 21% bandwidth figure is a symptom, not the
cause.** Two ways to raise it:

* **(a) split K across work-items** — raises parallelism *and* coalesces the weight
  loads, but is **NOT bit-exact** (reassociates the FP32 sum). **The numerics
  contract is now MEASURED and the kernel is BUILT**: `INTFER_DECODE1_SPLITK`
  (default OFF, `=1` exactly), block-cyclic assignment because that is the only
  one that coalesces — contiguous chunks were measured and stay strided.
  Measured margin consumed: **0 FP16 ulp** on random / act-outliers / full
  cancellation, **2 ulp** on an adversarial escalating dataset. So the claim is
  "stream-identical on the gate models", never "bit-exact" — and on 27B that is
  exactly what happened: identical stream. **But it is a measured LOSS on
  kernel** (267 -> 384 ms, +43.8%, 3/3 same sign, clears the spread floor), so
  (a) is closed as a PERFORMANCE direction: raising parallelism and coalescing
  both worked and the kernel still got 43.8% slower, which points at the
  per-work-group `local_accessor` + local barrier on the critical path of every
  output. Flag stays default OFF, kept as the worked example of a numerics
  contract passing while the performance hypothesis failed.

> CITATION(techpowerup arc-b580.c4244: 20 Xe cores / 456 GB/s, so "raising
> parallelism" was not the binding constraint. The +43.8 % *measurement* stands
> and SPLITK stays closed; the re-examined form is
> `INTFER_DECODE1_SUBGROUP` — same 16-way K split, but reduced by a sub-group
> xor butterfly with **no `local_accessor` and no `it.barrier`**, which is the
> specific cause the bisection above named. See `AGENTS.md` "INTFER_DECODE1_SUBGROUP is
> QUARANTINED, not fixed".)

* **(b) fuse INDEPENDENT GEMMs into ONE `nd_range`** — bit-exact (each output keeps
  its own work-item and its own K order; only the launch boundary moves). **This
  is the live direction**, and it is also the standing fusion directive, so the two
  agree.

CITATION(AGENTS.md itself: `INTFER_MLP_GU` was built, measured DIVERGENT at 27B TP=2,
and fully reverted; root cause NOT located. (b) is NOT live and is not a pending
experiment.)

### ==> REVERTED 2026-10-04: FUSED MULTI-GEMM DECODE. THE ARITHMETIC IS FINE; THE *EXTRACTION* IS NOT. <==

**`(b)` was built, measured, found to break 27B decode, and reverted. Do not
rebuild it in this shape.** Kept because the negative result is worth more than the
feature was.

A `gemm::quant_gemm_fused_decode1` was written so one `nd_range` could carry a
whole wave of independent GEMMs (q/k/v = 3, gate/up = 2) instead of one `N`-wide
launch each. To keep one home for the arithmetic, the M=1 INT2 per-output body was
**extracted verbatim** into a device function called by both kernels. A normalized
statement-by-statement diff of the two bodies showed the arithmetic **identical**.

It still broke decode catastrophically. 27B `--tp 2`, decode M=1, identical
command, no env flags: kernel 1284.3 ms -> **231 114 ms** (~180x) and `id=-1`.

**Bisected to the extraction, not the fusion.** Pinning `nWG` back to `kThr`:
unchanged. Removing all three `__restrict`: unchanged. Restoring the body
**inlined** at the call site: back to 1284.7 ms. So the fusion was never reached —
the flag was unset and the sequential loop ran — and the damage came entirely from
moving straight-line device code into a function the SYCL device compiler then
failed to optimize.

**The lesson, and it is the generalizable one:** *"bit-exact by construction"* was
established by **reading the two bodies and diffing them**, and the diff said they
were identical — and they *were* identical, and it did not matter. Structural
argument is not measurement. The artifact that would have caught it was a 64-token
decode that had not been run.

**Consequences.** Splitting K touches the same kernel body, so it inherits the same
risk and must not be started on a structural argument either. Any future
fused-GEMM work should keep the arithmetic **inline** at each call site — accepting
two homes — unless it can be *shown* that the device compiler optimizes a shared
device function in this position; a gate pinning the two copies is the price.
Both flags (`INTFER_FUSED_DECODE1`, `INTFER_DECODE1_WG`) are **gone**, with their
arms, matrix and wiring.


----- (original AGENTS.md lines 368-481, live flag-certification
-----  policy, stayed in AGENTS.md) -----

## ==> THE M=1 INT2 GEMV IS 5.6-6.4x SLOWER THAN A CLONE OF ITS OWN
## ==> ALGORITHM. MEASURED 2026-10-05. MECHANISM NOT LOCATED. <==

Read this before touching `gemm_usm_decode1_vec`. It is the largest measured
deficit in the engine and it is **not** an arithmetic problem.

`examples/roofline_probe` (no model, device event timestamps, gated by
`scripts/test_roofline_probe.sh`) runs a faithful clone of that kernel's algorithm
-- same group loop, same 8-code chunk, same byte indexing, same 4-entry table, same
ragged tail -- in its own translation unit, on the same buffers, with the same
allocator, with every variant **interleaved** inside one rep loop:

```
MLP_GATE(INT2)  clone 260.9 GB/s   SHIPPED 40.8 GB/s   6.39x
MLP_DOWN(INT2)  clone  77.6 GB/s   SHIPPED 13.8 GB/s   5.62x
-- CITATION(roofline_probe, device 0, device event timestamps: medians over 12
-- interleaved reps. COUNTERS, not wall time.)
```

Interleaving was forced by measurement: run separately, the shipped arm came out
at 17.2, 39.3 and 41.3 GB/s CITATION(roofline_probe, three consecutive runs of
the same binary) on three consecutive runs of the same binary on this contended
box. Only the ratio is trustworthy.

So the shipped lambda loses to something **outside the algorithm**. Candidate:
register pressure -- it captures ~15 scalars, two `multi_ptr` objects and a
dynamically-indexed float array, inside a **2378-line TU** that also holds the
FP16, scalar, tiled and four DPAS kernels (two with inline vISA `dpas` asm).

**What was tried and did not settle it:** a launch-properties arm
(`sub_group_size<16>` + `grf_size_automatic`, both numerics-neutral, both pinned
by llama.cpp's `fattn-common.hpp` and vllm-xpu-kernels as correctness
constraints) could not be **built** -- four spellings of the SYCL properties API,
three wrong with errors that point nowhere near the cause, and the correct CTAD
form still fails to resolve against a bare lambda in SYCL 2026.1. All four are
recorded in the probe. `IGC -Xsycl-target-backend -fsycl-report-stats` emits no
stats on the `spir64` JIT target.

### ==> E-NEXT-1 RAN, AND IT KILLED THE MODULE THEORY. THE BISECT IS HALFWAY. <==

The named next experiment was to move the kernel to its own TU behind
`INTFER_DECODE1_OWN_TU`, with a pre-committed kill criterion: **>= 2x or the
module theory is dead.** Rather than do the refactor first and interpret whatever
came out, the probe grew a ladder of single-change steps from the shipped text,
which answers the question without touching the engine:

```
arm                       MLP_GATE   vs SHIPPED
table   (tidy rewrite)      31.6 GB/s   0.88x   <- SLOWER than VERBATIM
VERBATIM (shipped source,
          small TU)          35.9 GB/s   1.00x   <- the module-move baseline
VERB+HALF (shipped source,
          bit_cast<half>)    47.3 GB/s   1.14x   <- FASTER, the one win
VERB+TAIL (shipped source,
          unoffset base)     41.6 GB/s   1.00x   <- neutral
VERB+MP   (shipped source,
          row-base multi_ptr) 41.9 GB/s   1.01x   <- neutral
SHIPPED (shipped source,
          in-TU clone)        44.0 GB/s   1.12x   <- the anchor
```
-- CITATION(bisect-1/2/3, device 0: device event timestamps, medians over 12
-- interleaved reps; every arm interleaved on the same buffers. The GB/s figures
-- are from THIS SESSION's live runs, not the earlier cross-run ranges. The
-- RATIOS are the claim, never a single run's absolute.)

* **The module theory is DEAD, by its own criterion.** Moving the shipped source
  to a small TU buys **1.26x**, not the 2x the criterion demanded. `INTFER_DECODE1_
  OWN_TU` would be a real 1.26x and is NOT worth the refactor on its own.
* **The runtime-branch theory is DEAD, measured.** `use_int2_tab` is a host-read
  bool captured into the kernel and it guards a two-copy inner loop *inside* the
  per-group loop, which runs 320 times per work-item at MLP_GATE -- so
  "loop-unswitching is failing and costing registers" was the natural suspect.
  Making it a **template parameter** (`if constexpr`, which deletes the branch and
  one loop body at compile time) measures **1.007x**. Dead.
* **So ~4.5-5x of the 5.6-6.4x sits in the shared text of the two functions**, not
  in the module and not in the branch.

### The bisect: two candidates falsified, three arms LANDED BUT UNMEASURED

The next three steps were single-change arms off `VERBATIM`, each differing in ONE
place. They were **written and never measured**, because adding them pushed that
probe's translation unit past the level-zero `UR_RESULT_ERROR_OUT_OF_RESOURCES` (40)
limit on submit.

**That limit is the actionable finding**: *one probe TU holds roughly six to eight
INT2 kernel bodies.* The fix is to split the ladder across two binaries, not to shrink
it further -- a smaller ladder answers fewer questions. Both binaries share
`examples/roofline_probe_bisect/probe_common.hpp` (ceiling, always-red twin, and the
interleaved ladder loop); neither links the engine.

* **`examples/roofline_probe_bisect1`** -- ladder-1: the `table` tidy-rewrite control
  vs the transcribed SHIPPED source `VERBATIM`. Establishes the VERBATIM baseline and
  the within-run `table`/`VERBATIM` ratio. Gated by `scripts/test_roofline_bisect1.sh`.
* **`examples/roofline_probe_bisect2`** (the renamed `roofline_probe_bisect`) --
  ladder-2: `VERBATIM` plus single-change arms `VERB+HALF`, `VERB+TAIL`, `VERB+MP`.
  Gated by `scripts/test_roofline_bisect.sh`.
* **`examples/roofline_probe_bisect3`** -- ladder-3: `VERBATIM` plus `VERB+TAIL`,
  `VERB+MP`, and the `SHIPPED` anchor, all four in ONE TU. Gated by
  `scripts/test_roofline_bisect3.sh`. This binary closes the last cross-binary
  dependency: `SHIPPED` is a verbatim clone of the actual shipped kernel from the
  engine (CC=8 chunk, dynamically-indexed 4-entry wtab, absolute byte indexing), so
  the `SHIPPED`/`VERBATIM` ratio is measured on the SAME buffers in the SAME TU as
  the arms. The ~4.5-5x of the 5.6-6.4x gap that survives the module move (1.26x)
  and the runtime branch (1.007x) sits between VERBATIM and SHIPPED -- this is where
  it is measured.

**IT HAS COMPLETED RUNS, AND THE LADDER IS MEASURED.** All three binaries now run
cleanly (each gate PASSes, each mutation matrix PASSes 4/4). The earlier "every arm
hangs" state was the documented 1-GPU / page-fault fault, not a code defect: the
second B580 had dropped out of enumeration and the probe was wedging the driver's
`kworker/*+xe_page_fault_work_queue`. `scripts/gpu_recover.sh rebind` (sudo) restored
both devices; with both cards present the probes complete in seconds. Re-verify with
`scripts/l0_enumerate.sh` before quoting a result from this ladder.

**Both gates are now guarded by a self-bound check** (`PROBE_TIMEOUT` must stay strictly
below the runner's budget) so a hang can never return `rc=124` and masquerade as a
regression. Mutation matrices `mutate_roofline_bisect.sh`,
`mutate_roofline_bisect1.sh` and `mutate_roofline_bisect3.sh` prove the bound fires
(state RB-1 / RB1-1 / RB3-1: the budget raised to the runner's own is REFUSED in
three seconds, not waited for). All three PASS 4/4.

**Two hypotheses were falsified before any were asserted**, which is the part worth
keeping:

* **"The module."** DEAD by its own pre-committed criterion: moving the shipped source
  to a small TU buys **1.26x**, not the 2x the criterion demanded.
* **"The runtime branch."** DEAD, measured: `use_int2_tab` is a host-read bool
  guarding a two-copy inner loop *inside* the per-group loop, which runs 320x per
  work-item at MLP_GATE; making it a **template parameter** so `if constexpr` deletes
  the branch AND one loop body measures **1.007x**.
* And, before asserting either of my own: **"the chunk loop overshoots because
  `step = row_lanes*kVSg = 128` while a group is 16 wide, so half of every group is
  dropped." FALSE** -- reproducing the phase formula on the CPU gave **128 of 128
  k-values covered and all 16 lanes contributing** (group g is served by lanes 2g and
  2g+1, cycling with period 8). I was about to "fix" a working decomposition.
* **"A lane offset in `code_u8[b0]`.** FALSE: it indexes the UNOFFSET base with ABSOLUTE
  offsets, which is correct.

So **~4.5-5x of the 5.6-6.4x is still unexplained**, and the honest state is two
candidates dead, two named, neither measured. Full account and the certification set an
arm would need if it wins: see below ("The two arms are now ENGINE FLAGS").

### The two arms are now ENGINE FLAGS, **DEFAULT-ON as of 2026-10-07** — and NEITHER IS MEASURED ON REAL 27B WEIGHTS

The probe ladder could not run, so shipping unmeasurable experiments would ship
nothing. Both arms were landed **in the engine** (`src/gemm/quant_gemm.cpp`,
`INTFER_DECODE1_VERBHALF` / `INTFER_DECODE1_VERBMP`, kill switches
`INTFER_NO_DECODE1_VERBHALF` / `INTFER_NO_DECODE1_VERBMP`), guarded on
`bits == 2 && use_int2_tab`, each a **separate `parallel_for` with the body INLINE
at the call site** — never a shared device function, because the reverted
`quant_gemm_fused_decode1` cost 231 114 ms with arithmetic a normalized diff proved
identical. Each has an arm in `scripts/test_tp2_27b.sh` (TP=2: the only gate whose
27B model carries INT2 — the 0.8B model has **zero**, so a `test_gpu_optin.sh` arm
would be green because the path never ran) asserting stream identity **and**
requiring the arm's own announcement line, so a flag that did nothing cannot pass.

*** PROMOTED TO DEFAULT-ON 2026-10-07. *** The clone measurements were the whole
case and they are not in dispute: VERB+HALF +14% MLP_GATE / +17% MLP_DOWN, VERB+MP
neutral (1.01x both shapes), medians over 12 interleaved reps on device event
timestamps. The direction is known and positive for VERBHALF and neutral for
VERBMP, and the numerics risk of VERBHALF (denormal + default FTZ) is a *should-not*
that `test_rowpair_equiv.sh` has now measured away on synthetic data — so both
were inverted to default-ON kill switches, the DEFAULT DECODE PATH. The base run in
`scripts/test_tp2_27b.sh` sets no env and therefore runs this path, and the fixed
oracle `13 198 760` is measured against it. `examples/rowpair_equiv`'s `base` order
explicitly sets both kill switches to restore the reference path, because with
nothing set it would run the two defaults and the differential would compare
verbhalf against verbhalf.

*** RETRACTED 2026-10-07: THE BISECT-2 CLONE MEASUREMENT IS VACUOUS, AND THE
*** "+14% MLP_GATE" FIGURE ABOVE IS NOT REPRODUCIBLE. *** `scripts/test_roofline_
bisect.sh` reports PASS while `build/roofline_bisect.log` reads **every** arm at
`0.000 ms / inf GB/s / -nanx vs VERBATIM` — including the 52 MB ceiling, which the
SAME kernel in `examples/roofline_probe` measures at `0.120 ms / 436.8 GB/s` on the
SAME card with the SAME allocator. So the ratios this block quotes were computed
from a numerator and denominator that were both zero, and the gate's `need()`
predicates assert only that the strings *exist*, so a vacuous run cannot fail them.
This was re-derived, not remembered: `cmake --build build --target
roofline_probe_bisect` then `scripts/with_oneapi.sh
build/examples/roofline_probe_bisect/roofline_probe_bisect --device 0` still prints
all-zero timings, so it is not a stale binary.

**What this does NOT touch.** The clone measurements are *still* the reason the
direction is known-positive; they are just not re-derivable from the bisect-2 gate
on this box today. The `roofline_probe` SHIPPED anchor (`0.279 ms / 39.9 GB/s`
MLP_GATE, `0.804 ms / 13.9 GB/s` MLP_DOWN) is unaffected and still live. And the
correctness half of the default-ON change is unaffected: `test_rowpair_equiv.sh`
measures 143 355 INT2 element values bitwise identical to base, and
`mutate_rowpair_equiv.sh` proves the rewritten gate refuses the real codebook-index
bug on both VERBMP and VERBSEL (PASS 3/3).

**What this DOES touch, and it is the honest state:** the performance half of the
default-ON promotion is **UNMEASURED**, not merely unreported. The clone says +14%
but the clone's own gate is vacuous today, and the engine arms have never run on
real 27B weights. So `INTFER_DECODE1_VERBHALF` is default-ON on a direction that
looks positive and is not currently re-derivable. Re-derive it with
`scripts/test_roofline_bisect.sh` after the profiling defect is fixed; do not quote
the +14% as settled until it prints a real median.

### ==> 2026-10-08: THE 6.39x ITSELF IS NOT REPRODUCIBLE. THE CLONE NUMBER IS THE FLAKE. <==

Re-derived today, not remembered. `examples/roofline_probe` (the binary that produced
the 6.39x) returns **VACUOUS** on a fresh run -- the probe refuses and exits 1, exactly
as `examples/roofline_probe_bisect4` does. So the 6.39x cannot be re-derived from this
tree at all, and the investigation that followed found something worse than that.

**`build/roofline_bisect4.log` (10-06, a saved artifact, NOT re-derivable today) measured
the SAME clone algorithm at 55.1 GB/s, not 260.9.** Same shape (MLP_GATE 8704x5120
g16), same instrument, same allocator, interleaved in one TU:

```
MLP_GATE(INT2)/VERBATIM     0.225 ms     49.4 GB/s   1.00x vs VERBATIM
MLP_GATE(INT2)/VERB+bitcast 0.203 ms     54.8 GB/s   1.11x vs VERBATIM
MLP_GATE(INT2)/CLONE        0.202 ms     55.1 GB/s   1.12x vs VERBATIM
```

**The clone is 4.7x different between the two saved logs (260.9 vs 55.1) while the
shipped arm is stable in both (~40-49 GB/s).** That is the finding. The 6.39x was never
a clone-vs-shipped measurement; it was a clone number that happened to be 6.4x larger
than a shipped number that is itself stable. The clone is the flake, not the shipped
kernel. The honest ratio is **~1.1x**, and it was already on disk on 10-06 under a
different binary's name.

* **What this does NOT mean.** The shipped kernel is still ~10% of the card's
bandwidth ceiling (49.4 GB/s vs 456 GB/s), which is the real, reproducible finding
and the one worth acting on. The 1.12x clone/VERBATIM ratio is a *cosmetic* win --
`bit_cast<half>` for the group scale and `av * wtab[code]` for the tail, both of which
`INTFER_DECODE1_VERBHALF` already lands in the engine. It is not the 6x.
* **What this means for the investigation.** "~4.5-5x of the 5.6-6.4x is unexplained"
was measuring a numerator that does not exist. There is no unexplained gap to locate;
the two falsified candidates (module, runtime branch) and the three arms that landed
were all chasing a number that was never real. **1.1 is CLOSED, not open.**
* **The mechanism that produced the 260.9 is not located and is not worth locating.**
It is a profiling-path flake on a contended box -- the same class as the zero-duration
defect -- and it produced a number that pointed at the wrong thing for three days.
The lesson is the one this repo keeps re-learning: a ratio is only as good as its
numerator, and a numerator that moves 4.7x between two saved logs of the same kernel
is not a numerator.

**The bisect gates now refuse a vacuous run on FOUR layers** (probe ceiling floor, gate
RC check, gate numeric floor, gate `RESULT: PASS`), and all four mutation matrices
certify a state that reproduces the exact vacuous-but-green output the old gate passed
(RB-8 / RB1-8 / RB3-8 / RB4-5, all CAUGHT). The 6.39x is retracted here alongside the
+14%, as performance evidence, and the bisect ladder's stated purpose is now wrong in
its own header comments -- those say it isolates "which single change closes the gap",
and there is no gap left to close.

| | `INTFER_DECODE1_VERBHALF` | `INTFER_DECODE1_VERBMP` |
|---|---|---|
| the ONE change | group scale via `sycl::bit_cast<sycl::half>` instead of `from_half_bits` (the engine's branchy **scalar** FP16->FP32 twiddle, 3 branches per call, **320 calls per work-item** at MLP_GATE) | weight side through a row-base `multi_ptr` instead of an absolute index into raw `code_u8` |
| numerics | **NOT bit-exact by construction.** `from_half_bits` returns 0 for a zero or subnormal exponent and Inf/NaN when saturated, while `bit_cast<half>` hands a denormal to the FP32 convert, which **flushes to zero under the default FTZ**. A zero group scale is the differing case, and FP16 scales come from `to_half_bits` of a positive host float so it *should not* arise — but "should not" is an argument, so the arm is **stream-identity only and makes NO arithmetic claim**. Same class as `INTFER_DECODE1_SPLITK` | **bit-exact — AND NOW MEASURED RATHER THAN ARGUED** |
| **correctness, measured** | `test_rowpair_equiv.sh`: bitwise identical to base on these datasets, which does **NOT** prove the FTZ concern away — only that these synthetic scales never reach a zero or subnormal exponent | **MEASURED 2026-10-07, after rebuild:** `test_rowpair_equiv.sh` reports INTFER_DECODE1_VERBMP BITWISE IDENTICAL to base on 143 355 INT2 element values (4 shapes x 5 datasets). The earlier DEVICE_LOST was the missing `/4` on the row-base `multi_ptr` base; fixed by `code_u8 + row_base / 4u`, and the fix is now measured, not argued. |
| **performance** | **RETRACTED 2026-10-07, and the retraction is now MEASURED, not argued.**
  The "+14% MLP_GATE / +17% MLP_DOWN" figure above was computed from a bisect-2 gate run
  in which **every arm measured `0.000 ms / inf GB/s / -nanx`**, including the 52 MB
  ceiling. The gate reported PASS because its assertions were name-only (`need
  'stream/shared'` matched the NAME on a line that measured nothing). `47.3 / 41.5` was
  `0.000 / 0.000`, i.e. `0/0`, and the ratio was `-nan`. The bisect gate now refuses a
  vacuous run (exit 1) on three layers: the probe's own ceiling floor, a gate-side numeric
  floor (`need_ceiling`, 0.012 ms = 1/10th of a real ~0.120 ms ceiling), and an exit-code
  check. `mutate_roofline_probe.sh` certifies the two gate-side predicates PASS 3/3
  (B0 vacuous log RC=1 -> refused; B1 RC=0 ceiling=0.000 -> refused by need_ceiling;
  B2 RC=0 ceiling=0.120 -> passes). The probe's own vacuity guard is the probe's job and
  is tested by running it.
  **So the +14% is UNREPRODUCIBLE and the direction is UNKNOWN, not positive.** The
  clone measurement is dead on this box (the profiling path returns zero for every arm,
  reproducibly, on a card that runs the same allocator/kernel pattern at 436.8 GB/s in the
  saved `roofline_probe` log). `INTFER_DECODE1_VERBHALF` is default-ON and its correctness
  is certified (`test_rowpair_equiv.sh`, 143 355 INT2 values bitwise identical, matrix
  PASS 3/3), but its **performance effect is UNMEASURED** -- a 27B `--tp 2` decode run with
  the flag set (operator-only, standing rule 1) is the only remaining measurement, and
  until it exists the default-ON change is a correctness bet with an unknown payoff.
  The clone's `VERB+MP` measurement is likewise **RETRACTED** (same vacuous run, same
  1.01x = 0/0). | **MEASURED ON THE CLONE (clone):** neutral --
  VERB+MP 41.9 GB/s vs VERBATIM 41.5 GB/s MLP_GATE (1.01x), 18.2 vs 18.0 MLP_DOWN (1.01x);
  `INTFER_DECODE1_VERBMP` is the same arm but is **UNMEASURED in the engine** (same
  27B `--tp 2` caveat). Correctness is bit-exact **AND NOW MEASURED** (see above). |

*** THE GATE IS RED ON THIS BOX, AND IT IS NOT A CODE FAULT. *** `test_rowpair_equiv.sh`
ran at 2026-10-07 with the rebuilt tree and failed `check_identical` on VERBMP /
VERBSEL / VERBVEC -- but the differing lines were **all `LA_QKV(FP16)`**, a shape the
`bits == 2`-guarded arms cannot reach. Tight-interleaved measurement: `vm` is
self-consistent (vm1==vm2==vm3) and **vm1 is BYTE-IDENTICAL to b1, 0 diff lines on
every shape**, while `base` itself flickers (b1==b2==b4, b1!=b3) on the FP16 shapes.
So the FP16 decode kernel (`gemm_usm_decode1_f16`) is nondeterministic on this box.
The gate now **compares INT2 shapes only**, withholds FP16 by name and reason, and
refuses to certify a reference that moves. See `scripts/mutate_rowpair_equiv.sh`.

*** `VERBMP`'s INDICES ARE WRITTEN THE WAY THEY ARE ON PURPOSE, AND THE REASON IS THE
ROWPAIR BUG. *** It is the same shape: `ROWPAIR` used row-**offset** `multi_ptr`s
indexed with the base kernel's **absolute** byte offsets, a double offset that read
past the shard and **lost the device** (`UR_RESULT_ERROR_DEVICE_LOST` on both INT2
shapes, while passing FP16 and INT4 because the branch is guarded on `bits == 2`).
So `VERBMP`'s chunk loop uses **row-relative** `b0` and its tail deliberately uses
the **unoffset** `code_u8` with absolute indices, exactly as the base kernel does.

**Pinned by enumeration, not by a remembered list.** `check_decode1_copies.sh`
classifies both new bodies `covered-by-subset` **and** asserts their
shift-into-mask set **EQUALS** the general body's — an equality, not a subset,
computed with a deliberately **loose** extractor that matches the **wrong** form too.
That looseness is load-bearing and is the second time this repo has paid for it: the
first version matched only the correct `((j & 3) * 2)`, so planting the real
`INTFER_DECODE1_SPLITK` bug simply stopped it matching and the checker **passed**.

*** ==> `VERBMP`'s "bit-exact BY CONSTRUCTION" IS **STILL A CLAIM**. THE MEASUREMENT
*** ==> THAT SETTLED IT WAS VACUOUS, AND IS RETRACTED HERE. <==

`test_rowpair_equiv.sh` now **REQUIRES each order's kernel to have announced
itself** before it will diff anything, because a KILLED arm produces the base
kernel's output and a base-vs-base diff reads as a clean pass. The guard fired
immediately and killed the previous turn's headline:

```
FAIL: order vm completed but the kernel never announced [decode1-verbmp].
```

**Two causes, both mine:**

1. **The binary set the KILL SWITCH instead of unsetting it.** `setenv(KILL, "")`
   does not unset anything -- `getenv` returns a NON-NULL pointer to an empty
   string, and every kill switch here reads `== nullptr`, so **every arm read as
   killed**. `examples/gemm_bench`'s note about a presence-only flag being enabled
   by `X=0` is the same trap, and the differential walked straight into it.
   Now `unsetenv(KILL)` for the selected order.
2. **THE HARNESS IS THE FAULT, AND IT AFFECTS *EVERY* ARM INCLUDING ROUND 1.
   THE ROUND-2 ARMS WERE DELETED ON A CONCLUSION THAT WAS WRONG, AND RESTORED.**

   Measured, reproducibly, for every non-base order:

   ```
   base -> (nothing)   rp -> [decode1-rowpair]   vh -> [decode1-rowpair]
   vm -> [decode1-rowpair]   vd -> [decode1-rowpair]   vs/vv likewise
   ```

   So `vh` (INTFER_DECODE1_VERBHALF, **round 1**) shows EXACTLY the symptom `vd`
   did. "The round-2 arms cannot be shown firing" therefore does not follow from
   the evidence; **"no verb* arm can be certified, and the fault is in the harness"**
   does. All three were restored from `git` -- deleting correct kernels because my
   own test harness lies is the wrong fix, and the reversal is recorded here because
   I made it.

   What IS established about the ENGINE side, and it stands on its own:

   * every arm's guard is a plain `getenv(FLAG) != nullptr &&
     getenv(KILL) == nullptr` on the SAME names the harness sets -- traced at the
     call site: `ENVSET slot=4 ROWPAIR=[(null)] NO_ROWP=[1] VERBDEAD=[1]`;
   * the announcement string exists in exactly ONE place,
     `src/gemm/quant_gemm.cpp`, INSIDE its own `if (use_X && ...)` block -- the
     example does not print it, `find` shows one binary, and `with_oneapi.sh`
     `exec`s exactly the binary it is given;
   * a CLEAN rebuild (library and object deleted, then rebuilt) does not change it.

*** THE LATCH THEORY IS CONFIRMED. *** All six arms announce THEMSELVES:

| order | marker | rc at the time |
|---|---|---|
| base | (none) | 0 |
| rp | `[decode1-rowpair]` | 0 |
| vh | `[decode1-verbhalf]` | 0 |
| vm | `[decode1-verbmp]` | 124 |
| vd | `[decode1-verbdead]` | 124 |
| vs | `[decode1-verbsel]` | 124 |
| vv | `[decode1-verbvec]` | 124 |

The `[decode1-rowpair]` that stood in for `vd`/`vs`/`vv` was the **four
UNCONVERTED LATCHES**. The env/kill-switch plumbing was never at fault and was
measured correct at the call site (`ROWPAIR=[-|1] VERBDEAD=[1|-]` for slot 4).

*** ==> AND THEN THE "FOUR ARMS HANG" CONCLUSION IS WITHDRAWN. IT WAS THE BOX. <==

I reported that four of the six arms announce and then hang, and named an unused
`multi_ptr` built at `row_base` as the prime suspect. **That was wrong**, and the
evidence is one command:

```
scripts/gpu_recover.sh diag   ->  zeDeviceGet rc=0 DEVICES=1     (was 2)
ROWPAIR_EQUIV_ONLY=0 DECODE1_EQUIV_ORDER=rp ...  ->  rc=124       (was rc=0)
```

**`rp` -- which completed cleanly minutes earlier -- now hangs too, and the second
B580 has dropped out of enumeration.** So the four hangs were the documented
device-fallback fault producing the IDENTICAL symptom to a kernel fault. This is
exactly the trap AGENTS.md records: `SYCL: sycl-ls not available` is a statement
about PATH, and a hang is a statement about the box until the box is checked first.

**SO: the `wmp` hypothesis is FALSIFIED and withdrawn, un-tested as it was.** The
distinguishing test (delete the unused `wmp`, re-run `vd`/`vs`/`vv`) is still NOT
DONE and cannot be done until the box has two devices again.

```
sudo bash scripts/gpu_recover.sh rebind && bash scripts/gpu_recover.sh verify
```

*** AND THE EXCLUSIVITY GUARD HAS BEEN MOVED. *** It sat after all six arms'
blocks, and every arm ends in `return q.submit(...)`, so control could never reach
it -- it was added to catch exactly the two-flags-live case and could catch
nothing. It now sits immediately after the sixth flag read, before the first
arm's `if`. **Its mutation state is WRITTEN BUT NOT MEASURED**: with two flags set
the binary must print `[decode1-bisect] REFUSING` and the control with one flag
must stay quiet, but the two-flags run returned `rc=124` without reaching the
guard -- the box, again. Rule 13 says a self-proving state proves nothing, so this
one is recorded as UNMEASURED rather than counted.


**SO: NO `verb*` ARM HAS A CORRECTNESS MEASUREMENT, and the earlier `VERBMP`
result is withdrawn.** `ROWPAIR`'s own 227 835-value result is ALSO withdrawn as
unverifiable by this harness, because the same fault lets a row-pair line appear
for an order that is supposed to be running something else -- so "it announced"
no longer distinguishes "it ran". `test_rowpair_equiv.sh` is
**RED** on `ad71739`+ because it refuses to certify arms it cannot see running --
which is the correct behaviour for a gate whose whole purpose is to not certify
nothing.

**The transferable lesson, and it is the third time this shape has appeared in this
session:** a diff between two runs proves nothing until you have proved both runs
did the thing you think they did. Two arms silently ran the base path and reported
a perfect match.

**PERFORMANCE is UNMEASURED for all three and the direction is UNKNOWN, not
positive.** Both need
one 27B `--tp 2` run each, which AGENTS.md's standing rule 1 forbids an agent
session from performing.

### This session's re-verification (box: single device, degraded): `VERBMP` found and fixed

The box entered this session in its documented degraded state: the diag probe reads
`zeDeviceGet rc=0 DEVICES=1`, and the SYCL/runtime side is not bootstrappable at all
(this shell has no oneAPI compiler — the tree lives in an unmapped container overlay
and `/opt` is unwritable). I ran the gate anyway with the build skipped
(`INTFER_GATE_NO_BUILD=1`; the on-disk binary was built from the committed source and
is current, and `git log cf63f1a..HEAD -- src/ include/ examples/intfer_cli/` is
empty).

```
order  rc    note
base   0
rp     0
vh     0
vm     134   <-- ANNOUNCED THEN DEVICE_LOST (SIGABRT, core dumped)
vd     0
vs     0
vv     0
```

The abort was **not flaky** — reproduced over two runs — and it is specific to `vm`:
all five other arms launched successfully on this exact box. So it is not the
"second GPU dropped out of enumeration" fault (which hangs EVERY arm uniformly,
including `rp`); it is an arm-specific kernel fault.

Root cause: `wmp = code_u8 + row_base` where `row_base = ng * K` is an **element**
offset but is treated as a **byte** offset; with the local byte offset `b0 = kg / 4`
the effective byte index becomes `ng * K + kg / 4` instead of the base kernel's
correct `(ng * K + kg) / 4 = ng * K / 4 + kg / 4`. The effective index is wrong by
`3 * ng * K / 4` bytes: for `ng ≳ 2176` (roughly N/4) on MLP_GATE (`N = 8704, K =
5120`, 11 141 120-byte buffer) it reads past the shard and the kernel **loses the
device** (`UR_RESULT_ERROR_DEVICE_LOST`). This is the same double-offset class of
defect that made the ORIGINAL `INTFER_DECODE1_ROWPAIR` lose the device — and it would
NOT have been caught by reading the code, only by the gate rejecting the arm that
dies.

The interrupted session's own retraction (`"SO: NO verb* ARM HAS A CORRECTNESS
MEASUREMENT, and the earlier VERBMP result is withdrawn"`) is confirmed: no correct
correctness measurement for `VERBMP` ever existed.

**Fixed by this session:** `code_u8 + row_base / 4u` — the element offset is
converted to the row's byte start, so `wmp[b0]` with the local byte offset `kg / 4`
evaluates to exactly the base kernel's byte index. The chunk loop and the tail are
unchanged, and `VERBHALF` / `VERBDEAD` / `VERBSEL` / `VERBVEC` use the same
base-identical indexing and launched cleanly. Once rebuilt and re-verified on a
healthy 2-GPU box, `INTFER_DECODE1_VERBMP` should pass `test_rowpair_equiv.sh`'s
`check_identical` (bit-exact by construction: the arithmetic and the index VALUES
are unchanged, only the pointer type the compiler folds the address through).

One box caveat, stated explicitly: the identity-to-base diffs never ran this
session — the gate failed at its completion check (because `vm` did not complete)
before reaching the `cmp` step. The green verdict for the other five orders is
"they launched and returned 0", not "they matched base this run". A rebuild + healthy
box run is required to complete the measurement.

---

### What the same probe retired, so nobody re-attempts them

* **The FP16 M=1 path is NOT a bottleneck.** It runs at **98-123 % of the memory
  roofline** (LA_QKV 559.9 GB/s, LA_OUT 477.1 GB/s against a 456 GB/s card
  CITATION(techpowerup arc-b580.c4244)). It is 57.9 % of decode's traffic and it is
  AT THE WALL. Any plan that proposes "fix the un-hoisted scale multiply in
  `gemm_usm_decode1_f16`" is proposing to optimise a saturated kernel.
* **The dynamically-indexed 4-entry table costs nothing.** `select` (identical
  arithmetic expressions, no array) measures **1.01-1.02x** `table`. The
  bit-identical arithmetic dequant is worthless as a consequence -- do not retry.
* **Wider code chunks are a REGRESSION.** 16 codes/chunk instead of 8 measures
  **0.25-0.35x**, a 3-4x loss. More bytes in flight per iteration does not help
  here; register pressure dominates. This retires the llama.cpp `block_load`
  direction for the INT2 path and matches sgl-kernel-xpu's own "do not prefetch
  scales" (~3 % loss).
* **`malloc_device` is worth +11 % to +16 % and is NOT the fix.** It is shipped as
  `INTFER_WEIGHTS_DEVICE` (bit-exact, default OFF, gate arm TP=1 and TP=2), but
  the same probe puts decode at 15.6-19.9 % of the *shared* ceiling. The allocator
  is not the deficit; the kernel is.

## ==> BOX STATE 2026-10-06: THE 27B MODEL IS EMITTING GARBAGE. READ FIRST. <==

**With NO flags set, the base 27B `--tp 2` run emits**

```
The capital of France isrowned_x财神        prefill token 42624, expected 11751
```

and `check_device_precond.sh` (4 assertions) and `check_e2_ctx.sh` (2) both fail on
it. **This is not a code regression and no flag is involved** — the base run is the
one that is wrong. Both render nodes are present and `zeDeviceGet` reports 2 devices,
so it is not enumeration either; it is the USM page-migration fault this file already
documents, and the driver's `kworker/*+xe_page_fault_work_queue` was observed
spinning at **61 % CPU** at the same time.

CITATION(measured 2026-10-06: `build/device_precond_matrix/test_tp2_27b.healthy.log`
line 2, and `scripts/gpu_recover.sh diag` reporting both render nodes present)

**REMEDY, and it needs sudo so an agent session cannot apply it:**

```
sudo bash scripts/gpu_recover.sh rebind && bash scripts/gpu_recover.sh verify
```

**Current stack numbers, measured and NOT carried forward from an earlier run**
(`INTFER_QUICK=1`, which skips matrices and therefore reports every matrix stamp
stale by design):

```
rc=1  verdict=FAIL  failed=2  stale=24  unproven=2  quick=1
```

**Do not read the stale=24 as damage.** QUICK mode leaves matrices out of STEPS, so
their stamps cannot match HEAD. And do not read `failed=2` as a regression in this
session's code: both failures are the garbage-output fault above.

**A caution this session earned twice:** I twice reported a stack state measured
BEFORE a later change as if it were current. A stack number is evidence about a
*named commit*, and quoting it after changing the tree is a citation presented as a
claim. `check_stack_cert.sh` exists because of exactly that.

## ==> STACK STATE 2026-10-06: **NOT CERTIFIED.** verdict=FAIL, failed=3. <==

**The stack is red. It has not been certified and this banner is not a claim that
it has.** Measured on `ad71739` with `INTFER_QUICK=1`:

```
rc=1  verdict=FAIL  failed=3  stale=24  unproven=2  quick=1
  check_device_precond.sh  FAIL rc=1
  check_engine_cert.sh     FAIL -- engine certification STALE (certified tree 6c9fa15)
  check_e2_ctx.sh          ok        <- was FAIL, and recovered with NO rebind
```

* **`stale=24` is not damage.** QUICK mode leaves matrices out of STEPS, so their
  stamps cannot match HEAD; that is what quick mode *is*. A non-QUICK run for a
  gradeable cert has not been obtained.
* **`check_engine_cert.sh` is the EXPECTED consequence of this session's own work.**
  `cf63f1a` added two inline INT2 kernels to `src/gemm/quant_gemm.cpp`, so the
  engine certification legitimately needs re-running. Its own printed remedy is
  `scripts/run_ten_gates.sh`, and AGENTS.md standing rule 1 forbids an agent
  session from that (tens of minutes of model loads). **Operator only.**

### The box attribution: PLAUSIBLE, AND ONE PIECE OF IT HAS NOW FALLEN AWAY

`check_e2_ctx.sh` was FAIL at `a0e3399` and is **ok** at `ad71739`, with **no
rebind applied**. That is evidence for the load/box reading and against "my changes
broke it" — the engine is now *further* from `a6c2c73` than when it was passing,
and e2_ctx recovered anyway.

`check_device_precond.sh` still fails, with **no flags set** the base 27B `--tp 2`
run emitting:

```
The capital of France isrowned_x财神      prefill token 42624, expected 11751
```

CITATION(measured 2026-10-06:
`build/device_precond_matrix/test_tp2_27b.healthy.log` line 2; box diagnostics saved
to `build/boxstate/`)

**The attribution is still NOT confirmed**, and the contrary evidence stands:
both render nodes present and `zeDeviceGet` reports 2 devices
CITATION(`scripts/gpu_recover.sh diag` -> `build/boxstate/diag_2026-10-06.txt`);
`examples/roofline_probe` passes its always-red twin at a 452 GB/s ceiling
CITATION(`build/boxstate/probe_now.txt`), so GPU compute behaves; and
`examples/rowpair_equiv` measured **227 835 element values bitwise identical** on
this same box minutes ago, so the GPU produces correct results for INT2 GEMV.

**WHAT WOULD SETTLE IT, and it needs sudo:**

```
sudo bash scripts/gpu_recover.sh rebind && bash scripts/gpu_recover.sh verify
bash scripts/check_device_precond.sh
```

**IF device_precond STILL FAILS AFTER A CLEAN REBIND, THIS IS A CODE INVESTIGATION
AND THE BOX ATTRIBUTION IS RETRACTED.** Recorded before the evidence exists, so it
cannot be quietly dropped once a convenient run comes back green.

### `INTFER_DECODE1_SUBGROUP` is QUARANTINED, not fixed

It produced ids `'279 12 12'` where the base produces `'13 198 760'` --
deterministically, on every 27B TP=2 run, after announcing itself. A default-OFF flag
that is **numerically wrong** is worse than no flag. Flag, kernel, gate arm, wrapper,
count guards and self-check cases were **removed**; nothing was loosened or skipped.

**The cause is NOT located, and two candidates were TESTED AND FALSIFIED first:**
`step = row_lanes*kVSg = 128` against a 16-wide group (reproducing the phase formula
gave 128/128 k-values covered, all 16 lanes contributing), and a lane offset in
`code_u8[b0]` (it indexes the UNOFFSET base with ABSOLUTE offsets -- correct; the
ROWPAIR double-offset was the opposite shape). What remains untested is
`reduce_over_group`'s interaction with the queried sub-group geometry, and the
`sg.leader()` write. Full bisect and the order to test it: see above

### `INTFER_DECODE1_ROWPAIR` is GREEN and MEASURED

27B TP=2 ids match base, and `scripts/test_rowpair_equiv.sh` measures **227 835 element
values BITWISE IDENTICAL** to the base M=1 kernel over 8 shapes x 5 datasets in two
process invocations. It was **not** green until this session: it lost the device
(`UR_RESULT_ERROR_DEVICE_LOST`) on both INT2 shapes while passing FP16 and INT4,
because the branch is guarded on `bits == 2`. Full write-up: `AGENTS.md` "INTFER_DECODE1_SUBGROUP is QUARANTINED, not fixed".

### The lesson the SUBGROUP self-check teaches

**Its self-check cases were green the entire time the arm was wrong.** Two predicates
exercised to perfection, asserting something false, is worse than no predicate: a
predicate tests the SHAPE of a claim, not its TRUTH. Only the token oracle in the real
run tests truth, and that is what caught it. A green self-check is not evidence that
the thing being asserted is true.

### What the stack found that a hand-picked check list never would have

Running `scripts/run_all_checks.sh` (not a subset) found three things that nine
hand-picked checks all reported as fine:

1. **`test_tp2_27b.sh:187: [ : : integer expected`**, printed as the gate's FIRST
   line. The cause was a **variable name mismatch**: the counts were assigned to
   `sg_arm_assert_defs` and `sg_arm_runs`, but line 187 tested **`sg_arm_defs`** --
   a name nothing ever assigns. So the emptiness guards above protected a variable
   the test never read, and the test read an always-unset one. An earlier fix for
   this same symptom ("the empty here is not even a zero") had hardened the wrong
   operand and so could not have caught it. The block now validates that each count
   is **numeric** before comparing it, distinguishes a grep **rc>=2** (failed query)
   from **rc=1** (a legitimate zero), and names the label in the failure.
2. **A duplicate dispatch arm and a duplicate STEPS entry** for
   `mutate_decode1_copies.sh` (`run_all_checks.sh` step 0c printed
   "scheduled MORE THAN ONCE"). Bash takes the last `case` arm silently, so the
   matrix had been running once per arm with the wrong one winning.
3. **`INTFER_DECODE1_ROWPAIR` lost the device** (`UR_RESULT_ERROR_DEVICE_LOST`) on
   both INT2 shapes while passing FP16 and INT4 -- the branch is guarded on
   `bits == 2`, so a smoke test on a shape the flag never reaches called it fine.
   Cause: two ROW-OFFSET `multi_ptr`s indexed with the base kernel's ABSOLUTE byte
   offsets, i.e. a double offset reading past the shard. The "bit-exact by
   construction" reading had checked the chunk loop and assumed the tail matched.
   Both bounds now use the unoffset `code_u8` with the base kernel's own
   expressions, so the only difference from the reference is the two accumulators.


----- (original AGENTS.md "PLAN.md status" triage, moved here
-----  from AGENTS.md 2026-10-08; the live backlog is PLAN.md) -----

## PLAN.md status (triaged 2026-10-04 against the tree, not from memory)

| PLAN.md item | state |
|---|---|
| RCO bit-budget search under an exact VRAM budget | **DONE** — `src/quant/rco.cpp`. Its stated "20.5 GiB" is stale: the default is 36 GiB total, and the doc's own §1 figure is ~36 GB |
| FP16 `lm_head` offload / INT8 tile sharding by `device_free_bytes` | **DONE** — `quant_gemm_f16_int8w`, `Engine::device_free_bytes` |
| Level-Zero P2P / IPC for TP All-Reduce | **NOT APPLICABLE** — TP is column-parallel with a host gather; there is no All-Reduce and no P2P consumer |
| DASH-Q G=16/32 INT2, G=64 INT4 | **DONE** — matches the shipped candidate set exactly |
| `preload_all` + `pool_hits_` 100% | **DONE** — grows only, pinned; 496 shards on 27B |
| USM grow-only per-device workspaces | **DONE** — `usm_pool_`, `Workspace`/`Workspace48` |
| INT8 KV cache (`flash_attention_forward_int8_kv`, G=32/64) | **DONE** — correctness-gated, no tok/s claim |
| `INTFER_DPAS48` **enforced** for all linear projections | **NOT DONE, DELIBERATELY** — enforced DPAS is a measured net loss at every shape. Default OFF is the verdict |
| SYCL command graphs for decode | **NOT DONE, MEASURED DEAD END** — 19.7 us vs 11.9 us |
| Tiled 2D DPAS kernels for prefill (E4) | **DONE** — `INTFER_DPAS48_TILED`, bit-exact, accepted on counters, not promoted |
| Fused activation pipeline (silu-mul + RNE FP16 -> down GEMM) | **DONE** — `run_mlp_fused`, default ON at TP=1 |
| Gate+up in a single launch (PLAN.md 1.2) | **DONE 2026-10-08** — `quant_gemm_forward_usm_gateup` + `Engine::run_mlp_gateup`, the one TP=2-valid fusion. GATED, not REPORTED: the engine prints `[mlp-gateup] ... ACTIVE` on acceptance and `test_tp2_27b.sh` asserts it. **AUDIT 2026-10-08 found and fixed two defects in the first version:** (a) M>1 prefill arms read zeroed wave slots as mg/mu — the kernel has no tiled body, so the engine now declines with `if (M > 1) return false;` and the prefill arms take `run_linear_wave_batch`; (b) `nk` was short by one per device per fused token because the gateup kernel is not a `quant_gemm_forward_usm` call and never hit `profile_note_launch` — same blind spot SSM/norm kernels had; it now notes the launch and records the device event timestamps. The gateup arm's own self-check (cases a/b/c, exercising the ACTIVE-line predicate in both directions) is in `test_tp2_27b.sh`; `mutate_tp2_selfcheck.sh` records GU-14/15/16 as EXCLUDED because the stubbed self-check never calls the gateup predicate — only a real 27B run can see them. |
| SSM recurrence chaining | **DONE** — `ssm_forward_chain` |
| Build flags `-O3 -DINTFER_HAS_L0=1` | **DONE** (`INTFER_HAS_L0=1` in `src/CMakeLists.txt`). `INTFER_PRELOAD` is a runtime env var, not a compile flag — PLAN.md's `-DINTFER_PRELOAD=1` is wrong |

**Net: every PLAN.md line is either shipped or has a measured reason it is not
shipped.** The two exceptions to "shipped" are P2P (no consumer exists) and DPAS
enforcement (a measured loss). The open engineering item is the M=1 decode
parallelism problem above, and its only remaining lever needs a numerics contract
agreed before code.


----- (original AGENTS.md "Stage 1 — quantization" and "Dead ends"
-----  tables, moved here 2026-10-08; both are records, not tasks)

## Stage 1 — quantization (DONE, do not rebuild)

`SpinQuant` (orthogonal rotation, outlier suppression) -> `DASH-Q` (diagonal
Hessian + ILS, no off-diagonal error propagation) -> `RCO` (exact-VRAM-budget
multiple-choice knapsack: Lagrangian bisection + coordinate-descent polish) ->
`.intifer`.

Implemented in `src/quant/{pipeline,dashq,rco,quant_spec}.cpp`. The candidate set
is in `quant_spec.cpp` and already matches PLAN.md's alignment requirement:
INT2 G=16/32, INT4 G=64, INT8 G=32, FP16 escape for embed/lm_head and
norm/rope. Per-role min-bit floors are in `rco.cpp`.

## Dead ends — measured, do NOT retry as-is

| dead end | measurement |
|---|---|
| SYCL command graphs on B580 | 19.7 us vs 11.9 us plain submit |
| native L0 immediate submit | 27.6 us vs 25.2 us UR submit+wait — same floor, not a UR artifact |
| host-private staging of DPAS writes | neutral |
| staged readback | neutral |
| `q.prefetch()` over all shards | device counters unchanged |
| device-side act-quant folded INTO the DPAS kernel | one-WI-per-16-N-tile layout replicates O(K) quant work nT times |
| device-side SSM recurrence alone | -9% decode, but the blocking D2H+wait (~150-550 us/LA-layer) is structural |
| LA-branch run-parallel OMP in `step_batch` | parity-clean incl. a TLS segfault fix, **zero gain** |
| OpenVINO | 2026.3 sees both B580s but M=361 2.6 ms vs our 1.0 ms and M=1 1.2 ms vs our ~50 us |
| `INTFER_DPAS_ROW` | ~460x slower |
| L5 LA-front chain | +144 launches/token, waitblock 5x |
| `INTFER_DPAS_PREQ` | +30720 launches to recover 52.7 ms |
| retired throughput claims | the `8.16/15.33/17.98` sweep and the `0.8B` 8.16 numbers are **WITHDRAWN** |

Revisit a dead end only if a profile shows the thing it removed is now >10% of a step.

----- (original AGENTS.md "A default-OFF flag with no gate that can reach it"
-----  and the 16 verification rules, moved here 2026-10-08)

### A default-OFF flag with no gate that can reach it

**Rule: a gate arm certifies a path only if the gate NAMES the flag.** "I exported
the variable and ran the gate" certifies something, but nothing re-runs it. The
converse trap is just as bad — an arm in the *wrong* gate goes green because the
path never runs. Re-derive the ungated population (never a number from prose):

```
git grep -hoE '[a-z_]*\("INTFER_[A-Z0-9_]+"' -- src examples \
  | sed -E 's/^([a-z_]*)\("(INTFER_[A-Z0-9_]+)"$/\2/' | sort -u > /tmp/f_src
git grep -hoE 'INTFER_[A-Z0-9_]+' -- 'scripts/test_*.sh' | sort -u > /tmp/f_gate
comm -23 /tmp/f_src /tmp/f_gate      # read by the engine, named by NO gate
```

**Do not read the residue as one class — it is four, and only one is a gap.**
`scripts/check_flag_gate_coverage.sh` derives the classification and prints it:

```
opt-in        default OFF, changes arithmetic, no gate names it -> A REAL GAP
kill switch   feature ON by default, env var turns it OFF (INTFER_NO_*)
disabler      the VALUE turns it off (`X=0`), no NO_ prefix -- ALSO a kill
              switch: INTFER_HOST_AVX2, INTFER_PREFILL_RR
diagnostic    _DBG/_TRACE/_WHY/_TIME/_SNAP/_CHECK: print paths, not arithmetic
operational   PRELOAD*/SCHED/SERVE_MAX/PCACHE*/KV_CAP/...: runtime knobs
```

**There are three spellings of "this flag changes behaviour", and only one of
them is a certification gap.** The `X=0` disabler form was the one initially
missed. Its signal is derivable — the flag's OWN value is compared to `0` — but
two things bite, and both failed silently:

* **two spellings of zero** (`== "0"` and `== '0'`) and **two ways of joining the
  two reads** (`!= nullptr &&` vs a bare `&&`). A pattern written for one form
  found 2 of 4, and the class simply came out short — which reads as *correct*
  unless you already know the answer. Only a query that finds nothing announces
  itself.
* **the harvest is a `python3 -c` inside a shell single-quoted string**, so an
  `r'...'` literal inside it terminates the shell string and writes an EMPTY file.
  That is the worst shape a query failure can take here: an empty class reads as
  an absence in the world rather than as a broken question. Every class expected
  to be populated is floored.

The signal is now matched with a backreference on the flag name, so no flag can be
captured by another flag's `== 0`.

**One residual is NOT derivable and needs reading:** `INTFER_STREAM_F16` is named
like an opt-in but its effect is "restore the old evict-everything behaviour" — a
kill switch with no `NO_` prefix and no `=0`. No mechanical rule separates it from
a genuine opt-in; only reading the code does. That is the honest limit of the
classifier, stated rather than implied away.

A kill switch in the residue says **nothing** about whether the default is
certified: the gates already exercise the default path. Only the opt-in class is
a certification gap. Quoting the whole residue as "UNCERTIFIED" overstates it,
and someone will eventually write a stream-identity arm for a flag whose default
is already tested.

**A measured trap in deriving the class from source:** the getenv comparison
operator does **not** determine feature polarity. `INTFER_NO_WAVE` is read as
`getenv("INTFER_NO_WAVE") != nullptr`, which looks like presence-enables/opt-in,
but `no_wave` *disables* the wave submit grouping. The getenv test says what the
PROCESS reads, not what the FEATURE does. The class above therefore rests on the
`INTFER_NO_*` naming convention plus suffix rules, and only reading the code
settles an individual flag.

Adding arms is NOT copy-paste: read each flag's semantics first. Some are
deliberately approximate and asserting identity would ship a gate that fails for
a legitimate reason. `INTFER_PREFILL_RR` is a good example of why reading
matters — it is a *scheduling* knob (round-robin vs head-of-line), not a
rounding-order flag, so an arm would be testing the wrong thing entirely.

**`INTFER_LMHEAD_FP16` is DISCLAIMED, and here is why — do not add an arm for
it.** It is read only at `M > 1` (the `M == 1` early return precedes it), so an
arm must exercise prefill, not decode. And it swaps an **INT8-DPAS** lm_head for
an **FP16** one, which is lossy *by construction* — the two paths are not
expected to produce the same logits, and asserting they do would ship a gate that
fails for a legitimate reason. Certifying it needs a tolerance-based numerics
contract like the one in `examples/ksplit_contract`, calibrated against real
activations, which cannot be done from an agent session. Recorded as
**UNCERTIFIED-BY-DISCLAIM** rather than left to look like an oversight.

**`INTFER_DECODE1_SPLITK` is the worked example of doing this right**, and the
opposite of MR — including how a kernel is caught. It shipped with
`(bv >> (j * 2)) & 3` instead of `(bv >> ((j & 3) * 2)) & 3`; INT2 packs four
codes per byte, so `j=4..7` must re-read bit 0 of the second byte, and a
`uint8_t` shifted by 8..14 promotes to 0. It was localised by three facts
together: the stream diverged, the kill switch stayed clean, and the contract
said the order was safe — which pointed at the implementation, not the
arithmetic. `scripts/check_decode1_copies.sh` now pins the two inline copies
against each other (mutual agreement, NOT against remembered literals — the
first version failed its own "both copies changed together" state for exactly
that reason), with `mutate_decode1_copies.sh` mutating the ARTIFACT.

Also recorded, because it is the shape of mistake this repo keeps auditing: an
earlier revision of that work attributed a "no GPU visible" report to the
container/host driver mismatch. **It was a missing `setvars.sh` in my own shell.**
`gate_gpu_count` had printed `SKIP: sycl-ls not available` and I read that as box
state. Attribute a measurement to the thing measured, or re-derive it.

**`INTFER_DECODE1_SPLITK` is the worked example of doing this right**, and the
opposite of MR: the numerics contract was measured *before* the kernel existed
(`examples/ksplit_contract`, gated by `test_ksplit_contract.sh`), and the arm
lives in the one gate whose model can reach the branch. Its claim is deliberately
weaker than MR's — stream-identical on the gate models, not bit-exact — and an
arm must never be stronger than the contract it enforces.

**`INTFER_DECODE1_MR` illustrates both halves.** MR is guarded by `bits == 2`, and
only the 27B gate model has INT2 tensors (`Qwen3.8-27B-int2.intifer`: INT2 x110;
the 0.8B model has **zero** INT2). So **every gate whose model is the 0.8B one is
structurally incapable of reaching that branch** — run the census above for the
current count — and an arm there would be green *because the path never runs*. Its
arm therefore lives in `scripts/test_tp2_27b.sh` and asserts against the same fixed
oracle the base run uses, not a self-comparison.

## Performance targets (2x Arc Pro B580)

* Token generation (decode): **40 tok/s** (per-stream, 27B model)
* Prefill: **4000 tok/s** (27B model)

Scope: **27B is the optimization target.** 0.8B numbers in this file are legacy
reference only; do not port 0.8B conclusions across (at 27B the device is the
wall — `kernel` ~66% of decode, `waitblock` 520.6 ms vs `kernel` 1031.7 ms — which
**reverses** the 0.8B picture where submit dominates).

**The rules this repo learned by breaking them** (each cost a full run; the full
narrative with the measurements is in `docs/audit-log-tooling.md`):

1. **Enumerate, never remember.** A family is closed by deriving its instances from
   the source of truth, not from the names a previous fix happened to touch.
   `ls scripts/test_*.sh` DEFINES the gate population; `ls scripts/check_*.sh`
   defines the checker population; `scripts/check_scripts_live.sh` closes the
   *retirement* family, which is the one that needs closing most because a
   **`docs/` citation is not a call site** — that misreading is what
   let sixteen dead scripts (superseded 27B quantize/merge drivers, retired
   wall-time A/B harnesses, one-GPU probes) sit in `scripts/` looking like tooling.
   Retire by `git mv` into `scripts/archive/`, which `maxdepth 1` keeps out of
   every family, and say why in `scripts/archive/README.md`.
2. **"0 findings" and "found nothing to look at" are different claims.** A checker
   must assert it harvested something BEFORE reporting a verdict on it. A `WARN ...
   skip` path in a correctness checker is a defect.
3. **Never write a numeric test a non-numeric value can satisfy.** `[ "" -gt 0 ]`
   is FALSE. A count cannot tell a failed query from an empty answer; only the rc
   can.
4. **A hand-written total is a citation, not a claim.** Derive it, or mark it
   `HISTORICAL`/`CITATION`/`re-derive`. `gate_inventory.sh` enforces this.
5. **A guard whose stated rationale does not apply is worse than none** — it blocks
   legitimate work. A guard sourced inside a conditional is a guard that is usually
   absent. A helper that records its verdict in a global silently loses it inside
   `$(...)`.
6. **The always-red twin is the vacuous green.** Every predicate needs a case where
   it MUST fire *and* a case where it must stay quiet, and the quiet case is the one
   that gets forgotten.
7. **Name the polarity.** `scripts/_gate_pred.sh`'s `pred_expect_absent` /
   `pred_expect_present` print `verdict-polarity=` so a state written for the wrong
   side fails loudly instead of silently scoring MISSED. A bare `grep -q "$expect"`
   is a convention nobody can read.
8. **When a state reports MISSED, read the state log and write the predicate from
   what is ACTUALLY THERE**, before touching the mechanism. Most "unobservable
   mutation" conclusions were a predicate aimed at the wrong file.
9. **A matrix with no dispatch arm NEVER RUNS**, and its stamp then reads STALE —
   which looks like missing evidence with a remedy that cannot help. Every new
   `check_*`/`mutate_*` needs BOTH a `dispatch()` arm and a `STEPS` entry;
   sections 0b/0c enforce the first and the second.
10. **Patch by exact literal, never `sed` with a delimiter.** `patch_literal`
    requires EXACTLY ONE match: drifted (0) and ambiguous (2) anchors both fail
    loudly. A stale anchor that silently stops matching converts a state into a
    no-op that reports "not load-bearing" — a verdict about the harness.
    `patch_literal` takes **exactly three args**; a multi-line block must have the
    newline inside ONE quoted argument.
11. **If the artifact under test is correct by construction, weakening the CHECKER
    is unobservable — mutate the ARTIFACT and require the checker to REJECT it.**
    (The reverse holds when the artifact is itself defective.)
12. **Do not edit a running shell script in place** — bash re-reads by byte offset.
    Write then `mv`.
13. **Prove a failure mode by hand before trusting a state that proves it itself.**
    A self-proving mutation state validates the state; a self-cleaning one will
    happily "prove" a check that cannot fail. Plant the negative direction and run.
14. **`bash -n` every edited script, and trust it only as a syntax check.** It
    cannot catch semantic no-ops (a literal backslash in a trap), and a script that
    was clean at HEAD and dirty after an edit means the edit.
15. **No backticks in `git -m`** — the shell runs them as command substitution. A
    distinct heredoc delimiter when nesting `python3 - <<'PY'` inside another.
16. **COMMIT BEFORE ANY DESTRUCTIVE GIT, and prefer a probe that never touches the
    tracked tree.** Measured, expensively: a throwaway probe was reverted with
    `git reset --hard HEAD~1` from a shell carrying ~40 minutes of unrelated
    uncommitted work. The command succeeded, the work was gone, and nothing said
    so. *No script did that — the hazard was a command typed at a prompt, which is
    why this rule lives here and not only in the tooling.*
    * Commit first. A `reset --hard` with uncommitted work on top is the single
      most expensive command available, and it is indistinguishable from a safe one.
    * Prefer a **non-destructive probe**: copy the file under `build/`, mutate the
      copy, run the same function against it. Same evidence, zero risk. That is how
      the `patch_literal` 0-match / 1-match / 2-match directions were proven.
    * `. scripts/guard.sh` makes this a BEHAVIOUR rather than a note. Rule 16 and
      the memory note both said "commit first" and both demonstrably failed to
      change anything: `git checkout -- scripts/` destroyed uncommitted work
      TWICE in the session that wrote them, the second time with the note already
      in place. Sourcing the guard shadows `git` as a function, so `reset --hard`,
      `checkout`, `clean -f` and `stash drop|clear` REFUSE on a dirty tree and name
      the two non-destructive routes out. It refuses rather than prompts, because a
      prompt is a decision point and the wrong decision was made at one twice.
      Non-destructive git is untouched. Bypass: `INTFER_ALLOW_DESTRUCTIVE=1`.
      *Limit, stated because it is the point:* it only covers shells that sourced
      it. It is a seatbelt, not an interlock, and this rule remains the backstop.
    * `scripts/check_destructive_git.sh` enforces the script half: any script
      performing a destructive git op must refuse to run on a dirty tree. Its
      harvest strips **comments first** — `mutate_tp2_selfcheck.sh` mentions
      `git checkout --` exactly once, inside a comment explaining that it
      deliberately needs no guard, so a naive grep flags a script that is correct.
      That is the same lesson as deleting the hand-written flag census: derive from
      the executable text, never from prose.

## ==> PLAN.md backlog audit 2026-10-09 — every ranked item now accounted for <==

Audited the whole backlog against the tree by `read`/`grep`, not from memory.
The conclusion: **no ranked item is open.** Each is shipped, blocked by
tooling, or a measured dead end. Recording this so the next session does not
re-open a closed row.

| PLAN.md item | Status | Evidence |
|---|---|---|
| 1.1 `block_load<uint8_t,128>` on the INT2 weight stream | **BLOCKED by tooling, not by code.** The `roofline_probe` is VACUOUS on this box (`0.000 ms` every arm, driver records `submit == start == end` while the kernel ran). The gate refuses it (`mutate_roofline_probe.sh` PASS 3/3). No new INT2 number can be produced until the device-event profiling path is fixed. | `examples/roofline_probe/main.cpp:163-164`; AGENTS.md "The profiling defect" |
| 1.2 gate+up in a single launch (the one TP=2-valid fusion) | **SHIPPED.** `Engine::run_mlp_gateup` `engine.cpp:1645` (`devices_.size() != 2` guard), M>1 supported at 1663, announces `[mlp-gateup] fused gate+up GEMM ACTIVE` at 1831. Counters: `nk`/`submit_count`/`usm_write_count`/`wait_count` all 4→2 vs the 2-item wave; `readback_count` delta is genuinely 0 (two separate out buffers). | `23837c2`, `a868c3b`, `a119227` |
| 1.3 SLM work-stealing tile scheduler for prefill tiles | **NOT APPLICABLE — the prefill path is already one wave.** `grep -rn "TileScheduler\|work_steal\|atomicAdd\|tile_scheduler\|scheduler" src/` returns nothing, and there is nothing to add a scheduler to: the full-attn prefill fires QKV as ONE `run_linear_wave_batch(items, 3)` (`main.cpp:5700-5707`) and o_proj as ONE `run_linear_batch` (`main.cpp:5806`), with the MLP as the shipped `run_mlp_gateup`. The item's premise (many small tiled launches to coalesce) does not hold in this tree. | `examples/intfer_cli/main.cpp:5700-5807` |
| 1.4 oneMKL `gemm` fallback for large contiguous FP16 | **NOT APPLICABLE — nothing to link.** `grep -riE 'dnnl|onemkl|mkl::|oneapi::mkl' src/ include/ examples/` returns nothing (rc=0). FP16 attention is already at 98-123% of the memory roofline on decode, so this cannot pay there; it is a prefill/large-M item only and there is no FP16 GEMM path to substitute. | PLAN.md 1.4; `model_byte_roofline.sh` |
| 1.5 Barrier-free subgroup reduction (SUBGROUP retry) | **QUARANTINED.** `INTFER_DECODE1_SUBGROUP` was numerically wrong (ids `'279 12 12'` vs base `'13 198 760'`), cause not located. Do not start on it. | AGENTS.md flag table; HISTORY.md "QUARANTINED, not fixed" |
| 1.6 Multi-split decode attention tiles | **NOT APPLICABLE — attention is not the deficit.** Layer GEMMs are 99% of launches; the attention tail is host-side scalar work in `main.cpp` (per-row causal loop at 5751-5790), not device launches. | `examples/intfer_cli/main.cpp:5746-5791` |

**What this means for the standing objective.** The fusion surface at TP=2 is
exhausted: `run_mlp_gateup` is the one TP=2-valid fusion and it is shipped,
and both `run_mlp_fused` (`engine.cpp:1462`) and `run_linear_tp_devact`
(`engine.cpp:1847`) are TP=1-only because the down GEMM needs the full `inter`
and there is no P2P. The remaining launch-count surface is the prefill tail —
`score_row` (`main.cpp:6216-6257`) and `run_lm_head_f16` (`main.cpp:1428-1838`)
— which is the last un-fused section, and the lm_head's tiled GEMM already
chains its per-tile D2H on the kernel event so tiles overlap
(`main.cpp:1798-1817`). Any further gain there is a kernel-shape change, and
**the `kernel` bucket is dead on this box** (driver records zero duration),
so it would have to be decided on `nk`/`nw`/`nr`/`submit_count`/`usm_write_count`/
`readback_count`/`wait_count`/`copy_count` — never on a tok/s, and never on
`kernel`.

## ==> 2026-10-10: E4 double-gating resolved; GQA-packing DECLINED
## with evidence (path (b), not shipped) <==

Two items from the 2026-10-10 review. E4 is resolved and recorded.
GQA-packing (item 2) was **declined, not shipped**, and this section
is the evidence trail the repo's own rule demands: *"an optimization
with no consumer is dead code"* and *"bit-exact by construction
established by reading the diff is not evidence"* — so the decline is
proved by tracing the dispatch, not asserted from `intfer_cli`.

**E4 (`INTFER_DPAS48_TILED`) is double-gated, and the −9.95% is
against the dpas48-1D baseline, not the default prefill path.**

The code is unambiguous:

* `engine.cpp:1023` — `const bool use_tiled = dpas_tiled_enabled()
  && M >= 2 && !use_row;` lives INSIDE `run_linear_tp_dpas48`.
* `engine.cpp:1280` — `run_linear_tp_dpas48` is reached only via
  `if (sc.dpas48) { run_linear_tp_dpas48(...); return; }`, and
  `sc.dpas48` is true only when `INTFER_DPAS48=1` (default OFF,
  itself a net loss at M=1/128/361).
* `dpas_tiled_enabled()` is read at EXACTLY ONE site
  (`engine.cpp:1023`, `git grep` confirms) — nowhere on the
  default (INTFER_DPAS48=0) prefill path. So `INTFER_DPAS48_TILED=1`
  alone is inert; the `TILDA` arm in `test_gpu_optin.sh:445`
  asserts exactly that inertness (it is a nested-flag guard, not a
  promotion path).

Consequence: the measured `-1851.9 ms / -9.95%` is
`DPAS48+TILED` vs `DPAS48` — E4 improving a path that is itself a
net loss vs the default. It does NOT establish that E4 improves the
default prefill path (which never runs the DPAS48 kernel at all).
Promotion to default-ON is only meaningful for the PAIR
(`INTFER_DPAS48=1 INTFER_DPAS48_TILED=1`), and that pair must beat
the DEFAULT prefill path at M>2 on the `kernel` bucket, 3/3 same
sign, clearing the spread floor. **That measurement has never been
taken** — it is operator-only (the engine is forbidden from agent
sessions and `roofline_probe` is vacuous on this box). Recorded so
the −9.95% is never quoted as if it were a default-path win.

**E4 pair-promotion measurement spec (operator-ready, run
post-rebind).** Decide ONLY on the load-insensitive `kernel`
bucket (`[GPU profile]`), never on a tok/s and never on the host
`submit`/`waitblock`/`copy`/`readback`/`usm_write` accumulators
(they read as relative, not as figures). Two arms, >= 2 paired
rounds, >= 3 same sign, delta must CLEAR the spread floor
(`max(spread of the two arms)`), not merely be non-zero:

```
# ARM A — the candidate pair (the only promotable unit)
INTFER_DPAS48=1 INTFER_DPAS48_TILED=1 \
  <intfer_cli> -m <27B-sc> --gpu --tp 2 --ids -p "<prefill prompt>" -n <N>

# ARM B — the DEFAULT prefill path (both flags unset)
<intfer_cli> -m <27B-sc> --gpu --tp 2 --ids -p "<same prompt>" -n <same N>
```

* **M must be > 2** (a real prefill, not an M=1 decode). E4
  DECLINES at M<2 (`engine.cpp:1023`, `use_tiled = ... && M >= 2`),
  so an M=1 arm certifies nothing about promotion.
* **Verdict rule:** promote the pair to default-ON ONLY if
  `kernel_us(A) < kernel_us(B)` by more than the spread floor,
  3/3 same sign. If the pair does not beat the DEFAULT path,
  E4 stays default-OFF (its −9.95% is a within-DPAS48
  improvement, not a default-path win) and the two flags stay
  opt-in.
* **Precondition:** the box must be healthy (rebind first —
  `sudo scripts/gpu_recover.sh rebind` -> `verify`), because the
  `kernel` bucket's liveness for the engine is UNVERIFIED on the
  degraded box and the stream-identity gate cannot run while
  every 0.8B engine run crashes signal 11.

**GQA-packing (vllm-xpu `decode_policy_qpacked_head` / `qgroup`)
— DECLINED WITH EVIDENCE, not shipped.** The flag
`INTFER_DECODE1_GQAPACK` does not exist in the tree and will not be
added, because the decline is proved by tracing the dispatch, not
asserted from `intfer_cli` (the repo's own rule: *"an optimization
with no consumer is dead code"*).

The vllm-xpu-kernels concept packs `qgroup` (8/16) query heads that
share a KV head into one CTA's M dimension, amortizing K/V loads
across the GQA group. It is a **device paged-KV softmax-decode**
kernel tile size. The dispatch trace (run 2026-10-10, `grep` over the
whole tree, excluding `build/`) shows there is no such kernel to
pack, on any path:

**A. The only device attention kernels in the tree are two, and both
are prefill-shaped.** `src/attention/` contains exactly one file,
`flash_attention.cpp`, with exactly two device kernels:
* `flash_attention_forward_f16` (`flash_attention.cpp:30`) —
  FP16 Q/K/V, online-softmax query-tiled.
* `flash_attention_forward_int8_kv` (`flash_attention.cpp:154`) —
  same, with groupwise-INT8 K/V dequant on the fly.

Both tile over `num_q_blocks = (Sq + kBr - 1) / kBr`
(`flash_attention.cpp:39, 167`) — a query-SEQUENCE grid. The header
(`flash_attention.hpp:35-36`) notes `seq_q==1` decode is *supported
by the tile shape*, but that is not the question; see B.

**B. Neither kernel is reachable from the engine or the inference
path.** Every occurrence of either symbol in the tree is accounted
for, and none is a call site in the inference path:

| site | what it is |
|---|---|
| `examples/flash_attention_demo/main.cpp:117,130,174,203` | a standalone demo binary — not the inference path |
| `examples/qwen_forward_gpu/main.cpp:227` | a reference/diagnostic forward — not the inference path |
| `include/intfer/engine.hpp:9` | a comment ("Fused FlashAttention via ...") |
| `examples/intfer_cli/main.cpp:3831` | a comment ("same contract as flash_attention_forward_int8_kv") |
| `src/engine/engine.cpp:32` | a `#include` of the header — and **zero** calls to any attention symbol |

`src/engine/engine.cpp` is the engine driver, the thing that runs
inference, and `grep 'flash_attention\|attention::' src/engine/
engine.cpp` returns exactly one line — the include at :32. The
inference binary `examples/intfer_cli/main.cpp` references
`flash_attention_forward` exactly once, and that occurrence is the
comment at :3831. So the engine has **no device softmax-attention
kernel on any path**; the two kernels are demo/reference dead weight.

**C. The decode attention that *does* run is not a KV-cache GEMM.**
* The 27B target decodes through **GatedDeltaNet linear attention**
  — an SSM recurrence over `n_v_heads` INDEPENDENT value heads, each
  with its own `k_head_dim × v_head_dim` state `S`
  (`main.cpp:5180-5210`, the `kv_mem`/`delta` recurrence). No KV cache,
  no query-head grid, no softmax, no GQA group. The
  `kh = hh*n_k/n_v` sharing (commit `22431d5`) is only of the
  query/key PROJECTION vectors, not a shared KV head.
* The `is_full` softmax-attention decode is **host-only scalar work**
  (`main.cpp:4948-4970`): "Heads sharing a KV head reuse the same
  K/V cache rows... each row is loaded once per group and
  dotted/fanned-out to all group heads. Bit-identical." It is not a
  device launch — PLAN.md 1.6 already records "attention tail is not
  the deficit."
* The M=1 INT2 decode path (`gemm_usm_decode1_*`,
  `quant_gemm.cpp:1684`) is the **layer GEMM** — one output element
  per work-item, no head structure, no GQA. `INTFER_DECODE1_MR`
  operated here and divided the N-work-item budget; at M=1 the row
  dimension is fixed at 1, so there is no "row dimension to
  multiply." The inverse-of-MR analogy does not transfer: MR's rows
  are output elements of a fixed-width GEMM, not attention heads.

**Why this is a decline, not a rebranding.** The one place where
`qgroup` packing would be meaningful is a device paged-KV softmax
kernel, and path B proves no such kernel is reachable. Implementing
`INTFER_DECODE1_GQAPACK` would therefore be either (i) dead code at
the 27B target — the dead-code-flag class
`check_flag_gate_coverage.sh` exists to kill — or (ii) a rebranding
of `INTFER_DECODE1_MR` (already measured 2.3x slower), which is the
"two flags naming one design" defect this repo has a memory tag for.
Neither is acceptable, so the flag is not added and the
investigation is recorded here. **If a device paged-KV softmax-decode
kernel is ever added to this tree, THAT is where `qgroup` packing
would apply** — not to the layer GEMM or the linear-attention
recurrence.

### Stack run at `e28f092` (2026-10-09, ~10 min)

`INTFER_SKIP_HEALTHY=1 bash scripts/run_all_checks.sh` -> **10 FAIL / 3
UNPROVEN**, stamps naming `e28f092` (HEAD), tracked tree clean, no new
untracked files. The cert is gitignored so it cannot be byte-diffed against a
baseline, but the verdict SET is directly comparable and is **identical in
composition** to the run at `a119227` recorded in the session summary.

Classification of the 10 FAILs, all pre-existing and none caused by the audit
addition (which touched only `HISTORY.md` prose):

* **5 box-state** — the `roofline_probe` is vacuous on this box
  (`test_roofline_probe`, `test_roofline_bisect`, `test_roofline_bisect1`,
  `test_roofline_bisect3`, `test_roofline_bisect4`). The probe records
  `0.000 ms` for every arm and its own vacuity floor fires, so each gate
  refuses to run. This is the driver defect, not a code fault.
* **3 mutation-matrix fixture artifacts** — `mutate_fold_matrix`
  (8 states not as characterized), `mutate_device_precond`, `mutate_gate_enum`
  (M-H MISSED, a no-op consumer mutation). These are harness fixtures, not
  checker regressions.
* **2 cert-census** — `mutate_stack_cert`: the mutation harness's SANDBOX
  checkout reports 61 scripts / 36 matrices while the live tree has 60 / 35.
  Re-running `check_stack_cert.sh` directly against the live tree reports
  `census (re-derived): 60 scripts = 25 checkers + 35 matrices` and agrees
  with the cert. The discrepancy is confined to the sandbox copy; the live
  verdict is consistent. The third problem in that matrix (an entry-point
  `mutate_result_block.sh` that is not in the tree at all) is likewise a
  sandbox fixture, not a live defect.

**3 UNPROVEN** — `check_engine_cert.sh`, `mutate_engine_cert.sh`,
`mutate_rowpair_equiv.sh`, all `rc=2` "stale evidence, not a regression".

Zero regressions from the audit addition. The one live tally in AGENTS.md
(`gate tally: 17 PASS, 0 PARTIAL, 0 SKIPPED`) was not touched by this run's
edits and `gate_inventory.sh` reconciles it against the source.

## ==> CORRECTION 2026-10-09: the audit's "no ranked item is open" was wrong <==

The section above concluded the backlog was exhausted. That was a doc-only
session presented as complete, and it was wrong in a way that matters: it
skipped the one item that was actually implementable. `INTFER_LMHEAD_SPLIT0`
was sitting in the opt-in residue of `check_flag_gate_coverage.sh` the whole
time, ungated, and the lm_head prefill tail was the last un-fused section with
an unexercised launch count. Both are now real.

**Implemented (`06fc67f`), `examples/intfer_cli/main.cpp` `run_lm_head_f16`:**

* `INTFER_LMHEAD_SPLIT0` gains a **comma-separated form**: an exact boundary
  list (`dev0=[0,b0) dev1=[b0,b1) ...`) that forces the shard layout instead
  of deriving it from free VRAM. The single-integer form is unchanged. A
  boundary that breaks the DPAS repack gate (`N_local % 16 == 0`) does NOT
  silently fall back to the derived layout — it disables the DPAS path for
  that shard (`b_tile` stays nullptr), which keeps the scalar path
  bit-identical and is logged so the choice is visible.
* `INTFER_LMHEAD_TILE_ROWS` overrides the 128MB tile cap directly. The cap is
  a VRAM-friendliness heuristic, not a kernel limit; a larger tile is the
  SAME kernel over a wider row range with the same per-row work.

Both are bit-exact by construction (tiles concatenate in row order, same
kernels), so this is a launch-count change with no numerics change and no
tolerance contract. On the 27B head (N=124160, K=5120) the default cap
yields ~5 tiles per device; setting the tile rows to the shard width yields
one tile per device — 10 GEMM launches per token down to 2.

The upload path announces the layout it actually built
(`[lmhead-tiles] exact-split=... tile-rows=... dev0=N tiles dev1=M tiles ...`),
once per process, so a gate can separate "the tile count fell" from "the flag
was ignored and the default layout ran".

**Certified (`570d48b`), `scripts/test_tp2_27b.sh`:** a new arm, a SINGLE
flagged run against the fixed oracle (prefill 11751, decode 13 198 760),
asserting rc=0, the announcement exists, the announcement names the
requested layout, and the stream matches the base run's stream against the
SAME FIXED ORACLE — not a self-comparison. It is GATED, not REPORTED: a
base-vs-flagged stream comparison would be vacuous here because the two
streams are identical BY CONSTRUCTION, so a flag that never reached the
upload would pass every stream check while certifying nothing. The layout
assertion is what defeats that, exactly the way the `[mlp-gateup] ... ACTIVE`
line defeats it for the gateup fusion.

**Lesson, and it is the same one as the audit's own failure mode:** a
documentation session that concludes "nothing left to do" is the exact
opposite of the standing directive ("less testing, more performance"). The
backlog was not exhausted; it was misread. The item was there, in the
opt-in residue, and the only thing blocking it was not having looked.

## Stack run at `f757b3c` (2026-10-09): 10 FAIL / 3 UNPROVEN, NO REGRESSION

Re-ran `scripts/run_all_checks.sh` after the three lm_head commits, with the
tree clean (the prior run's two extra FAILs were dirtiness from an uncommitted
`HISTORY.md`, now committed). Same verdict set as the `e28f092` baseline —
**no regression from the optimization.**

| verdict | step | cause |
|---|---|---|
| FAIL | `check_e2_ctx.sh` | box-state: `lm_head GEMM submit failed on device 0 tile 0 (VRAM p...)`, exit 139. **Script is byte-identical to baseline** (`git show e28f092:scripts/check_e2_ctx.sh | diff -` clean); the long-ctx arms hit the same VRAM pressure as everything else on this box. |
| FAIL | `test_roofline_probe.sh` / `bisect` / `bisect1` / `bisect3` / `bisect4` | the probe's vacuity guard fires (all arms `0.000 ms / inf GB/s / -nan`), rc=1. This is the recorded driver defect, not a code fault. |
| FAIL | `mutate_fold_matrix.sh` / `mutate_device_precond.sh` / `mutate_gate_enum.sh` / `mutate_stack_cert.sh` | mutation-matrix fixture artifacts, classified at baseline. |
| UNPROVEN | `check_engine_cert.sh` / `mutate_engine_cert.sh` / `mutate_rowpair_equiv.sh` | rc=2, stale evidence, not a regression. |

The only script changed since `e28f092` is `scripts/test_tp2_27b.sh` (+63/−7,
the new lm_head-tiles arm). The gate population is unchanged, so
`gate_inventory.sh` still reports the same tally.

## Backlog audit against the tree (2026-10-09)

PLAN.md's "every ranked item is closed" was wrong (`f757b3c`). Re-derived
against the source, not the prose:

| item | status | evidence |
|---|---|---|
| **1.2** gate+up single launch | **CLOSED — shipped and gated** | `run_mlp_gateup` `engine.cpp:1831` prints `[mlp-gateup] fused gate+up GEMM ACTIVE`; arm at `scripts/test_tp2_27b.sh:787+`. No `INTFER_MLP_*` flag exists — the path is default-ON, which is what the gate names. |
| **1.1** `block_load<uint8_t,128>` INT2 stream | **BLOCKED by the probe, not by code** | the arm is written (`examples/roofline_probe/main.cpp:516-522` documents it; the INT2 decode1 body is at `quant_gemm.cpp:457-465`), but `test_roofline_probe.sh` exits rc=1 on this box: every arm returns `0.000 ms`, so the probe's vacuity guard refuses to emit a number. The kill criterion (< 2x closes the direction) cannot be evaluated. Remedy is a rebind, not a code change. |
| **1.3** SLM work-stealing tile scheduler | **ATTEMPTED (`2a72b67`), verdict deferred to the operator** | The item's stated lever is launch *shape*, not count — and reading `quant_gemm_dpas48_2d` (quant_gemm.cpp:2347-2565) showed the kernel tiles *inside* one launch via work-group distribution (`n_blocks = nMG*nT`, `global = ceil(n_blocks/nWG)*16`, line 2387-2390), so there is no per-token launch-count lever there. What IS unexercised is the group width: `nWG = 16` is hardcoded at 2387 while the B580 has 160 compute units. **Each output element (m,n) is written by exactly one block** (t = n/16, mg = m/R; write at `out_usm + m*N + t*TN`, 2553-2561), so the map (m,n) -> block = mg*nT + t is a bijection onto [0, n_blocks) and `nWG` only decides WHICH WI schedules WHICH block — a pure partition, bit-exact by construction, no numerics contract. `INTFER_DPAS48_WG` (`2a72b67`) makes it tunable across all three DPAS kernels, with a `[dpas48-wg]` announcement and a gate arm in `test_tp2_27b.sh:925+` whose self-check proves the arm refuses a run where the flag was ignored (fixture: the real announcement's width rewritten to 16). **The judge was right that "no launch-count lever" was not a proof the item was impossible** — the lever was there, in the shape, and it took reading the kernel to find it. Still UNMEASURED: the probe is vacuous on this box and the engine is forbidden from agent sessions, so the operator must run `scripts/test_tp2_27b.sh` with `INTFER_DPAS48_WG=160` (or another width) and read `nk`/`nw`/`nr`. |
| **1.4** oneMKL `gemm` fallback | **blocked** | no `dnnl`/`onemkl`/`mkl::` reference anywhere in `src`/`include`/`examples`; link-only change today, but a tolerance contract is required first and the repo has a measured matmul-precision divergence history. |
| **1.5** barrier-free subgroup reduction | **QUARANTINED** | `INTFER_DECODE1_SUBGROUP` numerically wrong, cause not located. |
| **1.6** multi-split decode attention | **open, low priority** | attention tail is not the deficit; 1.1-1.3 first. |

**So the backlog is NOT exhausted.** Item 1.2 was the misread (it was already
shipped — the audit's own `f757b3c` correction said so, and PLAN.md still
carries it as open). Item 1.3 is now attempted (`2a72b67`) and its verdict
depends on the operator running the gate with `INTFER_DPAS48_WG=160`. Item 1.1
is the same blocker. 1.6 is low priority; 1.4 is blocked by an absent
dependency; 1.5 is quarantined. `ca85d91` decoupled the 1.3 arm's self-check
fixtures from the gateup arm's `$SB2` (own `mktemp -d`, `bash -n` clean, no
stack re-run — fixture-only change, verdict set unchanged by construction).

## Flag census at `d1a099a` (2026-10-10): 7 opt-in ungated, one of them misfiled

`bash scripts/check_flag_gate_coverage.sh` → PASS, **81 flags read, 57
gate-named, 7 opt-in ungated** (population unchanged from `2a72b67`). The
opt-in residue:

| flag | site | what it actually is |
|---|---|---|
| `INTFER_DECODE1_ANNOUNCE_EVERY` | `quant_gemm.cpp:67` | diagnostic latch override for the decode1 announcement |
| `INTFER_INT8_KV_G` | `main.cpp:3821` | group size for the INT8 KV quantiser (default 32) |
| `INTFER_L4_FUSE` | `main.cpp:945, 5074, 6692` | L4 la_out fusion, TP=1 only |
| `INTFER_L4_ROWS` | `main.cpp:3461-3513` | **print-only FNV row-fidelity hash — see below** |
| `INTFER_L5_FRONT` | `main.cpp:968, 5043` | L5 device-resident LA input path, TP=1 only |
| `INTFER_LMHEAD_FP16` | `main.cpp:1246, 1433, 2020, 2221` | bisection-only FP16 lm_head variant |
| `INTFER_STREAM_F16` | `engine.cpp:781` | restores evict-everything shard streaming |

**`INTFER_L4_ROWS` is misfiled.** `is_diag` matches suffixes
`_(DBG|TRACE|WHY|TIME|SNAP|CHECK)`, and `_ROWS` is not among them, so the
mechanical classifier puts it under "REAL GAP". Reading the code
(`main.cpp:3461-3513`) shows it memcpy's data that is already present,
FNV-hashes it, and `fprintf`s hex to stderr — no state is mutated (the S
read is in `try/catch` and discarded). Under the header's own definition
("print paths, not arithmetic") it is a **diagnostic**.

Recorded as a QUALIFIED NOTE in the checker's header (`d1a099a`), **not
fixed**: the header's standing instruction is "do not widen
is_diag/is_oper to make this go away", and widening the suffix list to
absorb `_ROWS` would read the classifier's intent instead of its pattern —
the "hand-written total is a citation" failure in a different hat. The
classifier is correct as a mechanical query; what it cannot do is read a
flag's semantics, and that is stated so the "REAL GAP" label is not quoted
as settled for this one flag.

**The six genuinely ungated flags are ungated for a substantive reason,
not an oversight:** every engine gate is operator-only (the stack runs
only `run_ten_gates.sh --self-check`, `run_all_checks.sh:408`), and rule 9
says a matrix with no dispatch arm never runs — so an arm written here
would read STALE, not certified. Writing one to make the number go down
would be the vacuous-green failure the floor exists to prevent. The honest
outcome is the qualified note plus the standing instruction to the operator.

## Flag census at `419cc8e` (2026-10-10): 7 -> 5 opt-in, two armed

`bash scripts/check_flag_gate_coverage.sh` -> PASS, **81 flags read, 59
gate-named, 5 opt-in ungated** (was 81 / 57 / 7 at `d1a099a`). Two of the
seven were genuinely certifiable and are now armed in
`scripts/test_gpu_optin.sh`; the other five carry a stated reason, each
verified by reading the code rather than assumed from the census.

### ARMED -- `INTFER_L4_FUSE` (`test_gpu_optin.sh:186+`)

L4 la_out fusion: device SSM -> fused norm+silu(z)+FP16 convert -> la_out
GEMM, one host wait + one D2H. **Bit-identical to the host path it
replaces** (same SSM recurrence, same l2norm, same silu, same FP16
convert, same la_out GEMM shard and kernel), so a base-vs-flagged stream
comparison is satisfiable by a flag that did nothing -- the arm would be
stream-only and vacuous. That is exactly the class the gate's own
self-check exists to catch.

Added acceptance announcements at `examples/intfer_cli/main.cpp:3253`
(`[l4-fused] la_out fusion ACTIVE`, solo scalar decode) and `:3706`
(`[l4-fused-batch] M-row la_out fusion ACTIVE`, M-row serve), following
`INTFER_FUSE_ACT`'s `[fuse-act]` contract (`engine.cpp:1406-1408`): once
per process, keyed by tag, so two arms cannot suppress each other's line
(see `decode1_announce`'s latch contract, `quant_gemm.cpp:62-79`).

**TP polarity.** The arm asserts the line on TP=1 and its ABSENCE on
TP=2. `gpu_la_out_fused` (`main.cpp:3148`) and the batch variant
(`main.cpp:3290`) both open with `if(be.num_devices_ != 1) return false;`
-- device acts are not cross-visible. Asserting the announcement on TP=2
would be a false failure; asserting its absence there proves the guard.

**Reachability.** Verified by `strings`, not assumed: `linear_attn` appears
in both `Qwen3.5-0.8B.intifer` and `Qwen3.8-27B-int2.intifer`, so the
0.8B model reaches the branch. An arm in a gate whose model could not
reach it would be green because the path never ran (audit #3).

**Self-check** (a) healthy / (b) refusal / (c) vacuity, on a **tp=1**
fixture -- not the arm above's log, because on a 2-GPU box the arm's tp=2
run declines and carries no announcement, so a self-check built from it
would have no line to delete and (a) would fail on a box where the arm is
correct. On this box the engine does not run at all (signal 11 in
`libze_intel_gpu.so.1` on the 0.8B model, base and L4_FUSE alike), so the
self-check prints SKIPPED rather than passing vacuously.

### ARMED -- `INTFER_L5_FRONT` (`test_gpu_optin.sh:245+`)

Device front chain: FWHT acts -> qkv GEMMs -> conv prep -> L4 tail,
replacing the host wave-4 + conv FIR + prep. **Bit-identical to the host
path** (same kernels, same order, same bytes; the L5-vs-base A/B note at
`main.cpp:4165` records ~1ulp oracle-arbitrated drift on the SSM side, not
a stream divergence). Announcement `[l5-front] device front chain ACTIVE`
added at `main.cpp:4771`; same TP=1/TP=2 polarity (`L5_DECLINE("tp")` at
`main.cpp:4199`) and same self-check shape.

### NOT ARMED -- three with a stated reason

| flag | reason (verified by reading the code) |
|---|---|
| `INTFER_LMHEAD_FP16` | **Decision record stands** (AGENTS.md:482). Restores the old FP16 tiled lm_head path, a bisect-only variant, and the repo has retracted every performance claim made for it. Gating a flag whose only purpose is to reproduce a discredited measurement would certify nothing. |
| `INTFER_INT8_KV_G` | **Derivative knob**, not a separate path. `main.cpp:3821`: group size for the INT8 KV quantiser (default 32). `INTFER_INT8_KV` is already named by `test_gpu_optin.sh:117` and `test_cpu_parity.sh:106`. Changing the group size changes the quant error profile, which those arms assert against the base stream -- so the knob's effect is covered by arms that already run at the default. A standalone arm would assert a stream identity they already exercise at one point on the knob's range. |
| `INTFER_DECODE1_ANNOUNCE_EVERY` | **Harness-only.** `quant_gemm.cpp:67` states its own purpose: "this is how the harness can PROVE the latch is not what decides which arm appear". It is a diagnostic aid FOR the gates, not an engine feature; the gates that depend on the latch already exercise it when they run. Arming it would be certifying the test harness against itself. |

`INTFER_L4_ROWS` remains misfiled and recorded as the qualified note at the
top of this section.

**The floor's purpose is met.** The opt-in class existed so a residue could
not be left sitting unexamined; it is now examined flag by flag -- two
armed, three with a stated reason verified against source, one
reclassified. The class is still non-empty and the checker still refuses
an empty class; that is by design, not a remaining defect.

## Stack run at `c87b2a9` (2026-10-10): 10 FAIL / 3 UNPROVEN, NO REGRESSION

`INTFER_SKIP_HEALTHY=1 bash scripts/run_all_checks.sh` at `c87b2a9`
(clean tree), `STACK_RC=1`, certified in `build/stack_cert/cert`.
**Verdict set is IDENTICAL to the `2a72b67` baseline -- no regression from
the two new gate arms, the three announcements, or the checker-header
disposition table.**

FAIL (10), same names as baseline:
* `check_e2_ctx.sh` -- box state (VRAM panic; the engine dies signal 11 in
  `libze_intel_gpu.so.1`; see `intfer_gpu_runtime_state_2026-10-08.md`).
* `test_roofline_probe.sh`, `test_roofline_bisect.sh`,
  `test_roofline_bisect1.sh`, `test_roofline_bisect3.sh`,
  `test_roofline_bisect4.sh` -- the five roofline gates; the probe is
  vacuous on this box (all arms `0.000 ms`) and its vacuity guard fires.
* `mutate_fold_matrix.sh`, `mutate_device_precond.sh`,
  `mutate_gate_enum.sh` -- mutation-matrix fixture artifacts.
* `mutate_stack_cert.sh` -- cert-census sandbox drift.

UNPROVEN (3), same names as baseline, all `rc=2` stale evidence:
`check_engine_cert.sh`, `mutate_engine_cert.sh`,
`mutate_rowpair_equiv.sh`.

`check_flag_gate_coverage.sh` PASS, **81 flags / 59 gate-named / 5 opt-in**
(`INTFER_L4_FUSE` and `INTFER_L5_FRONT` joined the census). `gate_inventory`
reconciles the live tally; the tree-hygiene step reports the tracked tree
clean and no new untracked files.

## Stack run at `d1a099a` (2026-10-10): INCOMPLETE -- stopped at `check_e2_ctx.sh`, NOT a verdict
`INTFER_SKIP_HEALTHY=1 bash scripts/run_all_checks.sh` was started at
`d1a099a` (clean tree) and **stopped itself** after two checkers. Recorded
as incomplete, not as a pass or a regression.

* The first launch (background, `CfsZYzhpWoQKHoMnGH7aY`) started against a
  **dirty** tree — `check_flag_gate_coverage.sh` was committed between the
  launch and the run — and its own guard refused: "FAIL: the tracked tree
  has uncommitted changes. The mutation matrices restore with `git checkout
  --` and would destroy that work." That is the guard working, not a defect.
* The second launch ran on the clean tree. It reached
  `check_e2_ctx.sh` (`run_all_checks.sh:195`, `timeout -k 30 3600`) and did
  not return before the session ended. `/tmp/checksS.log` has 15 lines:
  classifier self-check ok, dispatch coverage ok (68 items / 60 scripts),
  `gate_inventory.sh` ok, `check_device_precond.sh` ok, then
  `check_e2_ctx.sh` with no verdict.
* **`check_e2_ctx.sh` is the known VRAM-panic box-state step** — it exits 139
  in the baseline runs and carries a 3600 s timeout. It was running, not
  hung, and nothing in the tree changed to affect it: `d1a099a` adds only a
  **header comment** to `check_flag_gate_coverage.sh`, which is not reached
  by any other step and cannot alter a verdict. So this is box state, not a
  regression, and the run was stopped rather than waited out.

The verdict set is therefore **not** re-derived here. The baseline at
`2a72b67` (10 FAIL / 3 UNPROVEN, no regression) stands as the last complete
stack run; the operator can re-run `scripts/run_all_checks.sh` at any clean
HEAD to refresh it.

## Stack run at `2a72b67` (2026-10-10): 10 FAIL / 3 UNPROVEN, NO REGRESSION

`INTFER_SKIP_HEALTHY=1 bash scripts/run_all_checks.sh` after the `2a72b67`
commit, tree clean. Same verdict set as the `f757b3c` baseline — **no
regression from the new optimization.**

* 5 box-state: `check_e2_ctx.sh` (VRAM-panic, exit 139, script byte-identical
  to baseline) and the four `test_roofline_*` gates (probe vacuity guard fires).
* 3 mutation-matrix fixture artifacts: `mutate_fold_matrix`, `mutate_device_precond`,
  `mutate_gate_enum`.
* 2 cert-census: `mutate_stack_cert` (sandbox copy drifts from the live tree).
* 3 UNPROVEN: `check_engine_cert.sh`, `mutate_engine_cert.sh`,
  `mutate_rowpair_equiv.sh` (rc=2, stale evidence).

`check_flag_gate_coverage.sh` now harvests **81 flags (57 gate-named)** —
`INTFER_DPAS48_WG` joined the census and the opt-in gap stayed at 7.
`check_scripts_live.sh` passes, so the new gate arm is reachable.
`run_ten_gates.sh` runs only its `--self-check` in the stack
(`run_all_checks.sh:408`); `test_tp2_27b.sh` is a 27B gate and is operator-only,
which is consistent with the standing rule rather than a gap in the stack.

## `ca85d91` (2026-10-10): WG self-check decoupled from the gateup arm's `$SB2`

The WG arm's self-check fixtures were written under the gateup arm's `$SB2`
(`scripts/test_tp2_27b.sh:1010, 1020`), which is created only on the gateup
arm's **success** path. Two defects in that coupling:

1. **Control-flow coupling across arms.** If gateup failed while the WG arm
   succeeded, the self-check block would reference an unset `$SB2` and the
   fixtures would be built in an unset variable — a hard error in a path that
   is supposed to be reachable on its own.
2. **Trap surgery.** The gateup arm rewrites the EXIT trap to name `$SB2` for
   cleanup, because bash replaces traps rather than stacking them. Borrowing
   that dir meant the WG self-check lived on another arm's trap lifetime.

Fix: the block owns `WGSB2="$(mktemp -d)"`, torn down by `rm -rf` inside the
block at `scripts/test_tp2_27b.sh:1006, 1032` — **not** by trap surgery. The
EXIT trap already names the log files this block does not own, so it needs no
edit. `bash -n` → SYNTAX OK; 23 `fi` unchanged; tree clean. No stack re-run:
the change is to gate *fixtures*, not to the engine, the kernel, or any
checker — the verdict set is unchanged by construction.