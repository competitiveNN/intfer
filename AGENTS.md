Port ninfer (https://github.com/Neroued/ninfer) to our local setup: 2x intel arc pro b580.

**Read `docs/audit-log-tooling.md` before adding a checker or a mutation
matrix** — it carries the ~40 tooling failure modes (vacuous gates,
inverted predicates, stale artifacts, ...) that the current tooling
exists to prevent. The 2026-10-04 cleanup that split the original
4684-line file is recorded in `HISTORY.md`.

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
* **NEVER report "no GPU visible" from an agent session without sourcing
  `setvars.sh` first.** MEASURED TWICE IN ONE SESSION, WRONG BOTH TIMES: a missing
  `source /opt/intel/oneapi/setvars.sh` was reported as "the documented
  container/host driver mismatch" and the operator was told a 27B handoff could
  not run; the second time the same omission hung `check_device_precond.sh` to a
  timeout. `SYCL: sycl-ls not available` is a statement about **PATH**, not about
  the machine. The preconditions now print an `ENVIRONMENT, NOT BOX STATE` banner
  with the remedy, pinned by `scripts/check_env_not_box_state.sh` (whose last
  assertion is a live probe, so the banner is proved to print).
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
  * **IF A REBIND STOPS WORKING, the deeper investigation kit is in
    `scripts/archive/`**, one rung below the remedy above: `gpu_health_check.sh`
    (diffs the two cards at the kernel/sysfs layer), `l0_enum_trace.sh` (runs the
    driver's own discovery code with its debug keys on, to see WHICH of its five
    silent skips fires) and `gpu_driver_lib_ab.sh` (proves a driver-library swap
    actually happened). They were archived as retired, not deleted, precisely
    because this fault has recurred; `scripts/archive/README.md` says so.

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
population changes; the checker fails if you do not.

```
gate tally: 17 PASS, 0 PARTIAL, 0 SKIPPED
```

* Historical lines recording the 1-GPU-era tally are **records of that era**.
  Do not "correct" them — rewriting a true historical record to match today's
  box is the same laundering failure in the other direction. They carry an
  explicit `[SUPERSEDED ...]` annotation instead.
* Present tense: *"all gates PASS; both B580s enumerate and TP=2 coverage is
  live."*
* The historical `+N on ...` annotations of how the count grew are recorded in
  `HISTORY.md` ("Gates: all PASS, nothing PARTIAL, nothing SKIPPED"), not here.

`gate_inventory.sh` also fails the build if any tally it finds in `AGENTS.md`,
`docs/*.md` disagrees with what the source classification predicts for the
box's GPU count — so "the docs said 8 instead of 7" is a refusal, not
something a reviewer has to notice.

## ==> PARAMOUNT DIRECTIVE (2026-10-04): OPTIMIZE FIRST. WALL-TIME BENCHMARKING IS RETIRED. <==

**The single objective is to make 27B (`--tp 2`) prefill and inference as fast as
possible.** Everything else is subordinate. This supersedes any instruction below
that says to go measure a tok/s.

### What is retired, and it is a real retirement

The **wall-time benchmark queue is closed.** The blank `RESULT:` lines in the
retired wall-time backlog are **ABANDONED, not pending** — do not fill them, do
not re-run arms to fill them, do not move them to archive. Measured reasons,

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
| `INTFER_DECODE1_SPLITK` | split-K block-cyclic M=1 INT2 decode: `N*S` work-items + coalesced weight loads. Exactly `=1`; kill switch `INTFER_NO_DECODE1_SPLITK` | **MEASURED LOSS.** Stream-identical to base on 27B (so the numerics contract held), but kernel 267 -> 384 ms (+43.8%, 3/3 same sign, clears the 6 ms spread floor; `nk/nw/nr` identical). Both halves of the hypothesis delivered and it still lost. **Do not promote.** |
| `INTFER_DECODE1_SUBGROUP` | llama.cpp mmvq shape: one SUB-GROUP per output row, lanes split K, reduced by `sycl::reduce_over_group` -- no `local_accessor`, no barrier. **REMOVED 2026-10-05**: numerically wrong (ids `'279 12 12'` vs base `'13 198 760'`) and the cause is not located. See `AGENTS.md` "INTFER_DECODE1_SUBGROUP is QUARANTINED, not fixed". |
| `INTFER_L4_FUSE` | fused ssm+norm+FP16 -> la_out GEMM (TP=1) | stream-identical, but M>1 nets ~-1.5% and scalar A/B was a variance wash |
| `INTFER_L5_FRONT` | device-resident whole LA input path | **dead end as built**: +144 launches/token, waitblock 5x. More kernels lose |
| `INTFER_SSM_GPU` | device SSM recurrence | declines under `INTFER_DPAS48=1` (its ~1ulp drift flips a GA64 code at a tie) |
| `INTFER_INT8_KV`, `INTFER_KV_Q8`, `INTFER_KV_HAD` | groupwise-INT8 KV cache | correctness-gated, no tok/s claim |
| `INTFER_DECODE1_VERBDEAD` | per-group locals removed, pure code motion | bit-exact, default OFF |
| `INTFER_DECODE1_VERBSEL` | the 4-float codebook by `select` instead of an indexed array | bit-exact, default OFF |
| `INTFER_DECODE1_VERBVEC` | activations through a plain pointer instead of a `multi_ptr` | bit-exact, default OFF |

### Default ON, with a kill switch

`INTFER_NO_FUSE_SILUMUL` (fused MLP, TP=1 only — the shipped +13%), `INTFER_NO_WAVE`
(wave submit grouping), `INTFER_LMHEAD_BATCH` (M-row lm_head), `INTFER_NO_WAVE`,
`INTFER_STREAM_F16` (restores evict-everything), `INTFER_PREFILL_CHUNK` +
`INTFER_BATCH_GEMM` (chunked M-row prefill, `INTFER_PREFILL_RR` restores the old
rounding order), `INTFER_HOST_AVX2`, `INTFER_DPAS_PREQ`-independent bits,
`INTFER_NO_DECODE1_VERBHALF` and `INTFER_NO_DECODE1_VERBMP` -- the two M=1 INT2
decode arms promoted to the DEFAULT DECODE PATH 2026-10-07. Their clone
measurements are **RETRACTED as performance evidence**: the bisect-2 gate that
produced them (VERBHALF +14% MLP_GATE / +17% MLP_DOWN, VERBMP neutral at 1.01x
both shapes) reported PASS while every arm measured `0.000 ms / inf GB/s / -nanx`
on this box, so those ratios were 0/0. The gate now refuses a vacuous run on
four layers (probe ceiling floor, gate-side exit-code check, gate-side numeric
floor, `RESULT: PASS`); the four bisect mutation matrices each certify a stub
that reproduces the exact vacuous-but-green output (RB-8 / RB1-8 / RB3-8 /
RB4-5, all CAUGHT).
VERBMP is bit-exact by construction; VERBHALF is stream-identity only (denormal
+ default FTZ is a *should-not*, now measured away on synthetic data by
`test_rowpair_equiv.sh`, 143 355 INT2 values bitwise identical, matrix PASS
3/3). **The performance direction is UNKNOWN, not positive** -- the engine
arms are still UNMEASURED on real 27B weights (operator-only, standing rule 1),
and until that run exists the default-ON change is a correctness bet with an
unknown payoff. Do not quote the +14% as settled.

**2026-10-08: the 5.6-6.4x "clone-vs-shipped" deficit is RETRACTED ENTIRELY, and
it is the clone number that was the flake, not the shipped kernel.** A fresh run of
`examples/roofline_probe` (the binary that produced the 6.39x) returns VACUOUS, and a
saved `build/roofline_bisect4.log` from 10-06 measured the SAME clone algorithm at
**55.1 GB/s** while the shipped arm stayed at ~40-49 GB/s in both saved logs. The clone
moved 4.7x between two saved logs of the same kernel; the shipped number did not. So
the 6.39x was never a clone-vs-shipped ratio, it was a flaky numerator divided by a
stable one, and the honest ratio is **1.12x** -- a *cosmetic* win (`bit_cast<half>`
for the group scale, `av * wtab[code]` for the tail) that VERBHALF already lands in the
engine. The two falsified candidates (module size, runtime branch) and the
"~4.5-5x unexplained" line were all chasing a number that was never real. **1.1 is
CLOSED, not open.** What remains is the reproducible finding: the shipped INT2 decode
runs at ~49 GB/s on MLP_GATE against a 456 GB/s ceiling, ~10.7% of peak. Full account:
HISTORY.md "THE 6.39x ITSELF IS NOT REPRODUCIBLE".

### The profiling defect: LOCATED 2026-10-08, it is the driver, not the code

`examples/roofline_probe` (and the bisect binaries that share its allocator and
instrument) returns `0.000 ms` for every arm on this box. Bisected to the
timestamp, not to the kernel:

```
[roofline] RAW submit=184192938721938 start=184192938721938 end=184192938721938
        start-submit=0 end-start=0 ns
[roofline] RAW rep 1 sink=16655032
```

`submit == start == end` for every rep, monotonic across reps (~60 ms apart, real
device up time), and the sink reads 16655032 after `q.wait()` -- so the kernel
**executed** and wrote 52 MB of non-constant data, and the driver records **no
duration** for these commands. Three wait orderings were tried (read-then-wait,
`e.wait()`-then-read, `q.wait()`-then-read) and all returned the same 0 ns, so it
is not an ordering artifact; `enable_profiling()` is accepted (timestamps are
recorded) but the durations are not.

The probe's vacuity guard is the correct behaviour and stays: it refuses rather
than substituting a host timer, which is load-sensitive and would have produced a
"ceiling" that is really a CPU number. `--raw` is a diagnostic only (it exits after
the ceiling arm and never writes a reportable table).

**Caveat, recorded rather than dropped:** the engine's own counter
(`engine.cpp:1206`, same `get_profiling_info` spelling, same `enable_profiling`
queue) is reported as real in a log that is **not in this repo** (`perf_base.err`,
archived). AGENTS.md's standing rule is to re-derive, not to remember, and that
number cannot be re-derived from this tree -- so "the engine's instrument ticks
on this box" is **UNVERIFIED HERE**, and this probe's zero is NOT evidence that
the engine's is also zero. The two could differ on a launch shape the probe never
exercises. The remedy is the same as for any box-state question: a rebind, not a
code change (`scripts/gpu_recover.sh`).

### Operational (not gate subjects)

`INTFER_PRELOAD` (`=0` restores layer streaming), `INTFER_SCHED`, `INTFER_SERVE_MAX`,
`INTFER_SPEC_K`, `INTFER_SPLIT_ITER`, `INTFER_PCACHE_*` / `INTFER_NO_PREFIX_CACHE`,
`INTFER_KV_CAP`, `INTFER_HTTP_SERIAL`, `INTFER_ARGMAX_*`, and the `_DBG`/`_TRACE`/
`_WHY`/`_TIME` diagnostics.

## Architecture facts worth knowing before you edit

* **BOTH FUSION SEAMS ARE TP=1 ONLY, so there is no local fusion at the target.**
  `Engine::run_mlp_fused` (`engine.cpp:1352`) and `Engine::run_linear_tp_devact`
  (`engine.cpp:1484`) both open with `if (devices_.size() != 1) return false;`.
  **Read this before designing an E5.** Routing a fusion through
  `run_linear_tp_devact` would make it dead code at 27B `--tp 2` — the exact
  "default-OFF flag no gate can reach" class the flag census exists to kill.
  The reason is structural, not an omission: TP slices N, so `gate`/`up` produce
  only device *d*'s slice of `inter`, while `down` has K = the **full** `inter`
  and therefore needs the other device's slice ⇒ the intermediate must cross the
  host, and there is no P2P. The one TP=2-valid fusion is **gate+up in a single
  launch** (both column-parallel over the same split, both device-local, and
  bit-exact because each output's K accumulation is untouched). Spec and the
  numerics argument: `AGENTS.md` "The TP=2 fusion surface".
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
## Verification — the tooling, and the rules that keep it honest

**Entry points:**

| command | what it does |
|---|---|
| `scripts/run_all_checks.sh` | the whole tooling stack; writes `build/stack_cert/cert`. Prefix `INTFER_QUICK=1` to run checkers only, skipping the matrices |
| `scripts/run_ten_gates.sh` | the 10 engine gates; writes `build/ten_gates/{gates.log,HEAD}` |
| `scripts/check_stack_cert.sh` | grades the stack certificate (POST-STACK, not a step) |
| `scripts/check_engine_cert.sh` | does the engine certification still carry to HEAD? |
| `scripts/gate_inventory.sh` | derives the gate population, preconditions, render-node guards, and reconciles the doc tallies |
| `scripts/check_scripts_live.sh` | every `scripts/` entry is reachable: NAMED by a live source, or matched by the gate runner's own `GATE_GLOB` parsed out of `run_ten_gates.sh`. `docs/` and `scripts/archive/` are records, not call sites |
**The rules this repo learned by breaking them** (each cost a full run; the
full narrative with the measurements is in `HISTORY.md`, "Verification —
the tooling, and the rules that keep it honest"):

1. **Enumerate, never remember.** `ls scripts/test_*.sh` DEFINES the gate
   population; `ls scripts/check_*.sh` defines the checker population.
   Retire by `git mv` into `scripts/archive/`, which `maxdepth 1` keeps out
   of every family.
2. **"0 findings" and "found nothing to look at" are different claims.** A
   checker must assert it harvested something BEFORE reporting a verdict.
3. **Never write a numeric test a non-numeric value can satisfy.** A count
   cannot tell a failed query from an empty answer; only the rc can.
4. **A hand-written total is a citation, not a claim.** Derive it, or mark it
   `HISTORICAL`/`CITATION`/`re-derive`. `gate_inventory.sh` enforces this.
5. **A guard whose stated rationale does not apply is worse than none.**
6. **The always-red twin is the vacuous green.** Every predicate needs a
   case where it MUST fire *and* a case where it must stay quiet.
7. **Name the polarity.** `scripts/_gate_pred.sh`'s `pred_expect_absent` /
   `pred_expect_present` print `verdict-polarity=`.
8. **When a state reports MISSED, read the state log and write the predicate
   from what is ACTUALLY THERE.**
9. **A matrix with no dispatch arm NEVER RUNS**, and its stamp then reads
   STALE. Every new `check_*`/`mutate_*` needs BOTH a `dispatch()` arm and a
   `STEPS` entry.
10. **Patch by exact literal, never `sed` with a delimiter.** `patch_literal`
    requires EXACTLY ONE match; drifted (0) and ambiguous (2) anchors both fail
    loudly.
11. **If the artifact under test is correct by construction, weakening the
    CHECKER is unobservable — mutate the ARTIFACT and require the checker to
    REJECT it.**
12. **Do not edit a running shell script in place** — bash re-reads by byte
    offset. Write then `mv`.
13. **Prove a failure mode by hand before trusting a state that proves it
    itself.** Plant the negative direction and run.
14. **`bash -n` every edited script, and trust it only as a syntax check.**
15. **No backticks in `git -m`** — the shell runs them as command
    substitution.
16. **commit before any destructive git operation** -- prefer a probe that
    never touches the tracked tree. `. scripts/guard.sh` shadows `git` as a
    function, so `reset --hard` / `checkout` / `clean -f` / `stash
    drop|clear` REFUSE on a dirty tree with uncommitted work and name the
    two non-destructive routes out. Bypass: `INTFER_ALLOW_DESTRUCTIVE=1`.


## Flag certification status

Every optimization ships behind a flag, and **a flag is only certified once a gate
NAMES it.**

**Do not read a flag list off this file — derive it.** This section used to carry
a hand-written list of seven "UNCERTIFIED" flags, and measured against the derived
census it was wrong in BOTH directions: it named three that are kill switches
(`INTFER_HOST_AVX2`, `INTFER_PREFILL_RR`, `INTFER_NO_WAVE` — the gates already
test their defaults) and omitted five that are genuine opt-in gaps. A hand-kept
list is a citation that decays into a guess:

```
scripts/check_flag_gate_coverage.sh     # classifies and prints every class
```

**`INTFER_LMHEAD_FP16` is UNCERTIFIED-BY-DISCLAIM** — see the census section above
for why no arm is warranted. That is a *decision record*, not a census entry, which
is precisely why it lives in prose while the census does not.

Closing the remaining opt-in gaps is scoped work, not copy-paste: read each flag's
semantics, decide whether identity is *claimed or disclaimed*, and pick the gate
whose model can actually reach the branch.

# *** SPLIT 2026-10-08 ***
# * PLAN.md    -- the forward backlog, biggest gain first.
# * HISTORY.md -- every already-tried optimization and its measurement.
# This file keeps the live directives only.
