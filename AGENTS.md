Port ninfer (https://github.com/Neroued/ninfer) to our local setup: 2x intel arc pro b580.

<!--
  CLEANED 2026-10-04 (was 4684 lines). What moved where:
    docs/audit-log-tooling.md   AUDIT #7..#31, VERBATIM. Tooling/gate-hygiene
                                history. Split there because #7 is where the
                                subject changes from the ENGINE to the CHECKERS.
    (inline, kept)              AUDIT #1..#6 -- numerics/quantization findings
                                about code that is still in the tree.
  Nothing was reworded or "corrected". A historical record that disagrees with
  today's tree is still a true record of when it was written; moving it intact is
  the only way to shrink this file without laundering it. See
  docs/audit-log-tooling.md for the ~40 tooling failure modes (vacuous gates,
  inverted predicates, stale artifacts, ...) that the current tooling exists to
  prevent. READ IT before adding a checker or a mutation matrix.
-->

## READ FIRST — box state, gate state, and what is live

### Box: both B580s enumerate (verified 2026-10-04 at the driver API)

`scripts/l0_enumerate.sh` -> `zeDeviceGet rc=0 DEVICES=2`, UUIDs `...-0400-...`
(= 04:00.0) and `...-8600-...` (= 86:00.0); `sycl-ls` prints 2 `[level_zero:gpu]`
lines.

**This has dropped and come back four times with nothing in this repo changing.**
It is HOST-SIDE driver enumeration state, not a code fault. Three consequences:

* The 1-GPU investigation is a record of ONE of those windows. It is true history
  and it stays, but **do not read it as the current box state and do not "fix"
  anything in it.**
* Any tally that depends on visible device count (PARTIAL / SKIPPED) is only
  meaningful together with the count it was measured at. **Re-derive it with
  `scripts/gate_inventory.sh`; never read one off this file.**
* **IF THE SECOND GPU VANISHES:** `scripts/gpu_recover.sh diag` (read-only) ->
  `sudo scripts/gpu_recover.sh rebind` -> `verify`. A plain rebind is what fixed
  it last time; nothing about the hardware, driver-library version, cgroup, NUMA,
  power or BAR sizes had to change.
  * `verify`: **0** = 2 devices, **1** = MEASURED 1 device, **2** = UNMEASURABLE.
    A box that cannot count its devices is not a box with one device.
  * **NEVER lead with the BAR/small-BAR theory.** It was refuted when a rebind
    fixed the fault with no BAR change. Investigate BARs only if a rebind stops
    working.
  * **Derive a card from `/dev/dri/by-path/pci-<slot>-render`, never from
    `/dev/dri/renderD<n>`.** The DRM minor has flipped `card1`->`card2` /
    `renderD129`->`renderD130`->`renderD129` across sessions. `renderD128` is the
    stable liveness probe; two or more distinct minors in one script is the defect
    `scripts/check_dri_probe.sh` refuses.

### Gates: all PASS, nothing PARTIAL, nothing SKIPPED

**Do not read a gate count off this file.** `scripts/gate_inventory.sh` derives
the population from `scripts/test_*.sh` and reconciles any tally it finds
against that derivation, so adding a gate makes every hand-written count stale
*by design* — that is the mechanism working, not a defect to silence. Run it:

```
scripts/gate_inventory.sh        # population, preconditions, and doc tally
```

Every gate's precondition is met with both B580s enumerating, so nothing is
PARTIAL and nothing is SKIPPED. `test_tp2_27b` runs for real (**TP2-27B TEST:
PASS**, `GPU backend: 2 device(s) (TP=sharded)`), and `test_gpu_optin` /
`test_device_argmax` execute their `--tp 2` arms instead of skipping them.
TP=2 coverage is live.

The line below is the ONE live tally in this file, kept so `gate_inventory.sh`
has a non-historical site to reconcile against -- its own `sites_found=0`
vacuity guard refuses to certify a corpus with no live site, which is the
correct behaviour and not something to work around. Rewrite it when the gate
population changes; the checker fails if you do not:

```
gate tally: 11 PASS, 0 PARTIAL, 0 SKIPPED
```

* Historical lines recording the 1-GPU-era tally are **records of that era**. Do
  not "correct" them — rewriting a true historical record to match today's box is
  the same laundering failure in the other direction. They carry an explicit
  `[SUPERSEDED ...]` annotation instead, which records the supersession rather
  than erasing the claim.
* Present tense: *"all gates PASS; both B580s enumerate and TP=2 coverage is
  live."*

`gate_inventory.sh` also fails the build if any tally it finds in `AGENTS.md`,
`docs/*.md` or `reports/*.md` disagrees with what the source classification
predicts for the box's GPU count — so "the docs said 8 instead of 7" is a
refusal, not something a reviewer has to notice.

### Two standing rules from the operator

1. **NEVER run a full-27B gate from an agent session.** Model load is minutes of
   VRAM residency work and the operator measures. Allowed: `--self-check`, the
   mutation matrices, and any checker that binds no model. Say which half of a
   claim is proven and which is pending the box.
2. **Do not commit to this clone while `run_all_checks.sh` is running.** A second
   session committing mid-stack makes stamps name a different commit than the
   snapshot, and the stack then reports STALE — a verdict whose printed remedy
   ("re-run the named matrix") cannot possibly help. It is loud now (step 2c
   reports HEAD MOVED as its own condition), but the fix is to not do it.

## ==> PARAMOUNT DIRECTIVE (2026-10-04): OPTIMIZE FIRST. WALL-TIME BENCHMARKING IS RETIRED. <==

**The single objective is to make 27B (`--tp 2`) prefill and inference as fast as
possible.** Everything else is subordinate. This supersedes any instruction below
that says to go measure a tok/s.

### What is retired, and it is a real retirement

The **wall-time benchmark queue is closed.** The blank `RESULT:` lines in
`reports/2026-09-30-pending-bench-backlog.md` are **ABANDONED, not pending** — do
not fill them, do not re-run arms to fill them, do not move that report to
`reports/archive/`. Measured reasons, not stylistic:

* a tok/s needs `load1 < 2` for the WHOLE of every arm; this box has not held that
  window (load1 runs 3-18) and the cause is **not ours** (desktop + sibling agent
  sessions; the standing rule is not to kill them);
* this project has already **retracted three separate wall-time claims** because a
  number reached a report without an artifact behind it;
* the queue produced **zero decisions** while consuming the sessions it was meant
  to inform.

### What is explicitly NOT retired

**The load-insensitive counters stay, and they are how every optimization gets
decided.** `kernel` / `submit` / `waitblock` / `copy` / `usm_write` / `readback`
and the exact integers `nk` / `nw` / `nr` are read from the **device's own event
timestamps**, so they are immune to `load1`. This is the mechanism, not a hedge:

* **four** optimizations were killed *by counters* and would have survived a tok/s:
  L5 LA-front, DPAS48 at M=1, host-private DPAS staging, SYCL command graphs;
* **two** were adopted *by counters*: fused MLP down-proj (+13% decode, one host
  wait and one D2H removed) and E3 `INTFER_DPAS48_FOLD`;
* **E4 `INTFER_DPAS48_TILED`** was accepted on a counter verdict alone (-1851.9 ms
  prefill kernel, -9.95%, 3/3 same sign, clears the spread, zero added launches)
  with **no tok/s at all** — which is the proof the instrument is sufficient.

**Decide on counters, report counters, never wait for a quiet box.** `kernel` is
the only load-insensitive bucket; `submit` / `waitblock` / `copy` / `readback` /
`usm_write` are host wall-clock accumulators, reported as `median(spread)` and
read as *relative*, never as a figure.

**A delta must CLEAR the spread floor, not merely be non-zero.** The floor is
`max(spread of the two arms compared)` and it needs >= 2 paired rounds — a
single-round delta has no noise floor to clear and must not be reported as a
result. `scripts/_bench_summary.py` enforces this and refuses a batch in which any
arm died (`FAILED=` records, empty `BUCKETS`, incomplete `nk`/`nw`/`nr`).

### The correctness gates are unchanged and still binding

Retiring benchmarking does **not** retire `scripts/run_ten_gates.sh` or the ten
gates. Every optimization ships behind a flag and must be **stream-identical** to
its base at TP=1 and TP=2, plus a mutation matrix before its checker is believed.
A speedup that changes a token is not a speedup. This is what makes the counters
trustworthy: if the streams are identical, a kernel delta is attributable to the
kernel.

## 27B DECODE IS **PARALLELISM-BOUND**, NOT BANDWIDTH-BOUND (measured 2026-10-04)

Read this before optimizing decode. It overturns the natural first guess.

27B `--tp 2`, decode M=1, counters only: `kernel` 44%, `waitblock` 22%, `submit`
17%, `usm_write` 9%, `readback` 8%. Decode reads **9.16 GiB of weights per device
per token** in **257.7 ms** of kernel = **35.5 GiB/s, ~21% of the B580's ~179
GB/s**. The obvious reading — "the GEMV kernel is not saturating bandwidth" — is
**wrong**.

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
cores, ~4 threads/EU**, and `kThr=16` (`quant_gemm.cpp:64`) cannot even fill one
SIMD. More outputs per work-item is *anti*-parallel at M=1. Do not re-attempt it.

**The target is EU occupancy; the 21% bandwidth figure is a symptom, not the
cause.** Two ways to raise it:

* **(a) split K across work-items** — raises parallelism *and* coalesces the weight
  loads, but is **NOT bit-exact** (reassociates the FP32 sum). **The numerics
  contract is now MEASURED and the kernel is BUILT**: `INTFER_DECODE1_SPLITK`
  (default OFF, `=1` exactly), block-cyclic assignment because that is the only
  one that coalesces — contiguous chunks were measured and stay strided.
  Measured margin consumed: **0 FP16 ulp** on random / act-outliers / full
  cancellation, **2 ulp** on an adversarial escalating dataset. So the claim is
  "stream-identical on the gate models", never "bit-exact". Gated in
  `test_tp2_27b.sh`. **Unbenchmarked — the verdict is pending the box.**
* **(b) fuse INDEPENDENT GEMMs into ONE `nd_range`** — bit-exact (each output keeps
  its own work-item and its own K order; only the launch boundary moves). **This
  is the live direction**, and it is also the standing fusion directive, so the two
  agree.

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

The residue is dominated by diagnostics (`_DBG`/`_TRACE`/`_WHY`/`_TIME`, derivable
by suffix) and operational knobs (`PRELOAD*`/`SCHED`/`SERVE_MAX`), which are
legitimately not gate subjects. The interesting middle is opt-in and kill-switch
flags that change arithmetic or control flow that no gate sets — `INTFER_L4_FUSE`,
`INTFER_L5_FRONT`, `INTFER_HOST_AVX2`, `INTFER_NO_WAVE`, `INTFER_LMHEAD_FP16`,
`INTFER_PREFILL_RR`, `INTFER_STREAM_F16`. Their recorded certifications came from
a commit message, i.e. a transcript. **Honest state: UNCERTIFIED, not "probably
fine".** Adding arms is NOT copy-paste: read each flag's semantics first, because
some are *deliberately approximate* (`INTFER_PREFILL_RR` rounds in a different
order by design) and asserting identity for those ships a gate that fails for a
legitimate reason.

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

Report results in **load-insensitive buckets** (`kernel`/`submit`/`waitblock`/
`copy`/`usm_write`/`readback`, `nk`/`nw`/`nr`) and never quote a tok/s that was not
measured on a quiet box.

### ALWAYS AIM FOR KERNEL FUSION to reduce CPU-GPU communication overhead

**Standing directive.** Fusion is the DEFAULT direction for a GPU optimization
here, not a fallback. Every host<->device round trip costs a submit, a wait and a
transfer, and on Arc B580 those dominate far more often than the arithmetic does.

Three questions to ask of **any** new optimization before believing it will pay:

1. **Which round trips does it remove** — submit / wait / D2H / H2D — **and what is
   its launch delta?** No round-trip removal and no launch reduction is unlikely
   to pay on this box.
2. **Prefer fusing the CONSUMER of a device buffer with the PRODUCER**, so the
   intermediate never reaches the host. `run_mlp_fused` and `la_out_fused` are the
   two shipped examples; `run_linear_tp_devact` is the seam they use.
3. **A fusion that changes FP32 accumulation ORDER is NOT bit-exact** and needs a
   tolerance-based numerics contract agreed **before** any code. Prefer fusions
   whose intermediate is an **exact integer fold** (the FOLD hoist is bit-exact
   because integer addition is associative) or a **bit-exact converter** (the RNE
   FP16 convert was verified exhaustively).

**Why, from this repo's own measurements — do not re-derive these:**

* **More device work in fewer host phases wins; more kernels lose.** Fused MLP
  down-projection was **+13% decode** by cutting one host wait + one D2H per MLP.
  By contrast the L5 LA-front chain **added** kernels and ~8 submits per LA layer
  and is a recorded dead end despite identical device work.
* **A fusion that ADDS launches can lose even while removing waits** — at M=3 under
  device saturation the L4 M-row save nets ~-1.5%, because submit savings land on
  an already-saturated queue.
* **Prefill and decode want different fusions.** Prefill needs XMX on the
  arithmetic; decode needs launch-count and transfer reduction.

### XMX verdict — the matrix engines are idle on the hot path

**Layer GEMMs (99% of launches) are pure EU/ALU dot-products** (dequant-on-the-fly
+ FP32 FMA): decode M=1 (`gemm_usm_decode1_vec`) and prefill M>1 tiled. xmxmon
reads all-zero `XVE_INST_EXECUTED_XMX_*` there. **Correction to the original
"zero DPAS" claim: `lm_head` DOES run INT8 DPAS** (`quant_gemm_dpas_s8w`,
default-ON whenever the repack exists, no env gate) — measured 180M INT8 slots on
a 361-token prefill, 5.5M on an M=1 decode.

Implication: prefill 4000 tok/s needs XMX (EU pipes are the whole prefill story);
decode 40 tok/s needs launch-count/transfer reduction — XMX at M=1 barely helps
because the shapes are memory-bound. **Do not re-attempt XMX for M=1.**

DPAS-issuing kernels in the tree: `quant_gemm_dpas48` (`INTFER_DPAS48=1`,
default **OFF** — a net loss at every tested shape), `quant_gemm_dpas_s8w`
(lm_head only), and E4's `quant_gemm_dpas48_tiled` (`INTFER_DPAS48_TILED`,
default **OFF**). Per xmxmon docs, mixed-precision DPAS counts under the WIDER
operand (W2A8 -> INT8 counter). Harness `scripts/xmx_profile.sh` joins xmxmon
capture with internal buckets and degrades gracefully when the streamer is blocked
(this box: `dev.xe.observation_paranoid=1`, no sudo).

## Flag inventory — DERIVED, do not hand-maintain

Read the defaults off the code, not off this table; this table is a map, not a
source. Two spellings exist and they are not interchangeable:

* `getenv("X") != nullptr` -> **presence** enables (`X=1`, `X=0`, `X=anything` all
  enable it). These flags need `X=` unset to disable.
* `env[0]=='1' && env[1]=='\0'` -> **exactly `X=1`** enables.
* `INTFER_NO_*` / `INTFER_LMHEAD_BATCH` are **kill switches**: the feature is ON by
  default and `=1` turns it OFF.

Re-derive the whole population with the `git grep` recipe above.

### Perf-relevant, default OFF (opt-in experiments)

| flag | what it changes | verdict |
|---|---|---|
| `INTFER_DPAS48` | W4A8/W2A8 DPAS for layer GEMMs | net loss at M=1/128/361; host GA64 act-quant dominated. OFF |
| `INTFER_DPAS48_FOLD` | hoists exact int32 activation column sums out of the 16-N tile loop | -1.8% kernel at 27B, replicated at 0.8B. **KEEP (E3)**, not promoted |
| `INTFER_DPAS48_TILED` | E4 cooperative-2D-block DPAS, bit-exact | **-1851.9 ms prefill kernel, -9.95%, 3/3 same sign. KEEP (E4)**, not promoted; declines at M=1 by design |
| `INTFER_DPAS_PREQ` | device-side GA64 act-quant pre-pass | recovers 52.7 ms of 2559.2 ms host act-quant but pays +30720 launches. **DO NOT PROMOTE** |
| `INTFER_DPAS_ROW` | one WI owns many tiles | ~460x slower. Historical dead end, mutually exclusive with E4 |
| `INTFER_DECODE1_MR` | 4 outputs per WI at M=1 INT2 | **2.3x slower.** Do not use |
| `INTFER_DECODE1_SPLITK` | split-K block-cyclic M=1 INT2 decode: `N*S` work-items + coalesced weight loads. Exactly `=1`; kill switch `INTFER_NO_DECODE1_SPLITK` | **NOT YET BENCHMARKED.** Numerics contracted (0 FP16 ulp on realistic data) and gated in `test_tp2_27b.sh`, but no 27B run yet — see `reports/2026-10-04-ksplit-decode1.md` |
| `INTFER_L4_FUSE` | fused ssm+norm+FP16 -> la_out GEMM (TP=1) | stream-identical, but M>1 nets ~-1.5% and scalar A/B was a variance wash |
| `INTFER_L5_FRONT` | device-resident whole LA input path | **dead end as built**: +144 launches/token, waitblock 5x. More kernels lose |
| `INTFER_SSM_GPU` | device SSM recurrence | declines under `INTFER_DPAS48=1` (its ~1ulp drift flips a GA64 code at a tie) |
| `INTFER_INT8_KV`, `INTFER_KV_Q8`, `INTFER_KV_HAD` | groupwise-INT8 KV cache | correctness-gated, no tok/s claim |

### Default ON, with a kill switch

`INTFER_NO_FUSE_SILUMUL` (fused MLP, TP=1 only — the shipped +13%), `INTFER_NO_WAVE`
(wave submit grouping), `INTFER_LMHEAD_BATCH` (M-row lm_head), `INTFER_NO_WAVE`,
`INTFER_STREAM_F16` (restores evict-everything), `INTFER_PREFILL_CHUNK` +
`INTFER_BATCH_GEMM` (chunked M-row prefill, `INTFER_PREFILL_RR` restores the old
rounding order), `INTFER_HOST_AVX2`, `INTFER_DPAS_PREQ`-independent bits.

### Operational (not gate subjects)

`INTFER_PRELOAD` (`=0` restores layer streaming), `INTFER_SCHED`, `INTFER_SERVE_MAX`,
`INTFER_SPEC_K`, `INTFER_SPLIT_ITER`, `INTFER_PCACHE_*` / `INTFER_NO_PREFIX_CACHE`,
`INTFER_KV_CAP`, `INTFER_HTTP_SERIAL`, `INTFER_ARGMAX_*`, and the `_DBG`/`_TRACE`/
`_WHY`/`_TIME` diagnostics.

## Architecture facts worth knowing before you edit

* **TP is COLUMN-parallel with a HOST gather — there is no All-Reduce.** `shard_weight`
  slices the output dim N; `run_linear_tp` waits all ranks, then each rank's D2H is
  `half_to_float_batch`-ed into `out.data() + m*t.N + start`. `grep -riE
  'allreduce|p2p|peer'` returns **nothing**. So P2P/IPC has no consumer to optimise;
  a "TP All-Reduce" optimization has nothing to attach to.
* **First-touch migration is real and one-shot.** `preload_all` builds and pins all
  shards (496 on 27B, ~49 s), but shard USM is `malloc_shared`, so pages stay
  host-resident until first kernel touch: device-local goes 791 MB -> 11.27 GB only
  after the first request. Same prompt back-to-back measured 9.28 s then 2.75 s,
  2.75 s. `q.prefetch()` was built and **reverted — a no-op hint for shared USM on
  this driver**; do not retry. Mitigation is one warmup request after `/health`
  (`scripts/serve_27b.sh`, `INTFER_NO_WARMUP=1` skips).
* **Submit inflates under device saturation**: ~84 us/launch at M=1 to ~600 us at
  M=128. Fewer host phases only convert to throughput where the device has idle gaps.
* **SSM/norm kernels submit through `quant_gemm` free functions**, so their launches
  bypass the `[GPU profile]` `nk` counter unless they call
  `gemm::profile_note_launch`. This was a real blind spot; verify `nk` moves when you
  add a kernel.
* **RCO budget is `36 GiB` by default** (`include/intfer/rco.hpp`), a *total*
  across both cards, and it is a soft objective: if no feasible assignment exists
  RCO returns the least-cost one rather than failing. The 20.5 GiB figure in PLAN.md
  is stale (see below).

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

## Verification — the tooling, and the rules that keep it honest

**Entry points:**

| command | what it does |
|---|---|
| `scripts/run_all_checks.sh` | the whole tooling stack; writes `build/stack_cert/cert`. Prefix `INTFER_QUICK=1` to run checkers only, skipping the matrices |
| `scripts/run_ten_gates.sh` | the 10 engine gates; writes `build/ten_gates/{gates.log,HEAD}` |
| `scripts/check_stack_cert.sh` | grades the stack certificate (POST-STACK, not a step) |
| `scripts/check_engine_cert.sh` | does the engine certification still carry to HEAD? |
| `scripts/gate_inventory.sh` | derives the gate population, preconditions, render-node guards, and reconciles the doc tallies |

**The rules this repo learned by breaking them** (each cost a full run; the full
narrative with the measurements is in `docs/audit-log-tooling.md`):

1. **Enumerate, never remember.** A family is closed by deriving its instances from
   the source of truth, not from the names a previous fix happened to touch.
   `ls scripts/test_*.sh` DEFINES the gate population; `ls scripts/check_*.sh`
   defines the checker population.
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

## Flag certification status

Every optimization ships behind a flag, and **a flag is only certified once a gate
NAMES it.** Currently uncategorised-but-live, and honest about it:

* `INTFER_L4_FUSE`, `INTFER_L5_FRONT`, `INTFER_HOST_AVX2`, `INTFER_NO_WAVE`,
  `INTFER_LMHEAD_FP16`, `INTFER_PREFILL_RR`, `INTFER_STREAM_F16` — recorded
  certifications are **transcripts in commit messages**, not re-runnable artifacts.
  State: **UNCERTIFIED.** Closing them is scoped work, not copy-paste: read each
  flag's semantics, decide whether identity is claimed or disclaimed, and pick the
  gate whose model can actually reach the branch.

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
| SSM recurrence chaining | **DONE** — `ssm_forward_chain` |
| Build flags `-O3 -DINTFER_HAS_L0=1` | **DONE** (`INTFER_HAS_L0=1` in `src/CMakeLists.txt`). `INTFER_PRELOAD` is a runtime env var, not a compile flag — PLAN.md's `-DINTFER_PRELOAD=1` is wrong |

**Net: every PLAN.md line is either shipped or has a measured reason it is not
shipped.** The two exceptions to "shipped" are P2P (no consumer exists) and DPAS
enforcement (a measured loss). The open engineering item is the M=1 decode
parallelism problem above, and its only remaining lever needs a numerics contract
agreed before code.

# guidelines

- remember to commit any meaningful change made such that in case of regressions everything can be bisected.
- remember to not litter /tmp with large tmp files, RAM is precious
- try to parallelize work as much as possible, spawn tasks (subagents) in different worktrees that implement a specific optimization experiment, then merge all the experiments in one single batch. Every optimization experiment is gated behind a flag in the cli. The user the runs the benchmarks and reports back the results.
- MANDATORY: every batch of optimization experiments handed to the user for testing MUST be written up as a report file in `reports/` (create the directory if missing; filename `reports/<YYYY-MM-DD>-<batch-slug>.md`, e.g. `reports/2026-09-30-batch-dpas-preq.md`). The user benchmarks from these reports, so a batch delivered without one is an unfinished batch. Each report MUST contain, per experiment: (1) the flag + kill switch and default state, (2) what it changes and why it should help, (3) the exact benchmark command to run (and the quiet-box rule load1 < 2), (4) the load-INSENSITIVE buckets I measured myself (kernel/submit/waitblock/copy/usm_write/readback, nk/nw/nr counters) with the caveat that they are counters, not wall time, (5) the expected direction and what counts as a win/neutral/loss, (6) which correctness gates already passed. NEVER put a tok/s or wall-time number I did not measure in a report — leave a blank `RESULT:` line for the user to fill from a quiet-box run. Update the report (not just AGENTS.md) when a follow-up session re-scopes or kills one of the experiments, so the user's list of pending tests stays accurate.
