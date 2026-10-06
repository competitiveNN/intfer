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
population changes; the checker fails if you do not:

```
gate tally: 14 PASS, 0 PARTIAL, 0 SKIPPED
  (+1 on 2026-10-05: `test_roofline_probe`, the model-free
  allocator-vs-kernel discriminator for decode's bandwidth deficit.
  Device event timestamps, no model bound, ~30 s.
  +1 on 2026-10-06: `test_roofline_bisect`, the second half of the M=1
  INT2 GEMV bisect, in its own translation unit because one probe TU
  crosses the level-zero OUT_OF_RESOURCES limit. No model bound, ~1 min.
  +1: `test_rowpair_equiv`, the MEASURED bitwise-equivalence check for
  `INTFER_DECODE1_ROWPAIR` -- 227 835 element values over 8 shapes x 5
  datasets, two process invocations, no model bound.)
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
   re-examined against the bandwidth hypothesis in
   `reports/2026-10-05-batch-gemv-roofline.md` rather than left closed.

**The rest of this section is preserved verbatim below as the historical record.**
It is a record of what was believed on 2026-10-04 and of measurements that still
hold. Do not "correct" it into agreement — that is the same laundering failure in
the other direction. `scripts/check_hw_constants.sh` now refuses an uncited
capability figure anywhere in `AGENTS.md` / `docs/` / `reports/`.

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
> specific cause the bisection above named. See
> `reports/2026-10-05-batch-gemv-roofline.md`.)

* **(b) fuse INDEPENDENT GEMMs into ONE `nd_range`** — bit-exact (each output keeps
  its own work-item and its own K order; only the launch boundary moves). **This
  is the live direction**, and it is also the standing fusion directive, so the two
  agree.

> CITATION(reports/2026-10-05-tp2-fusion-surface.md §3b: `INTFER_MLP_GU` was built,
> measured DIVERGENT at 27B TP=2, and fully reverted; root cause NOT located. (b)
> is NOT live and is not a pending experiment.)

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
table   (tidy rewrite)      193-244 GB/s   6.3x   <- 6.3x
VERBATIM (shipped source,
          small TU)           38.8 GB/s   1.26x   <- the module move alone
VERB+TPL (shipped source,
          if constexpr)       39.1 GB/s   1.28x   <- 1.007x VERBATIM
SHIPPED (shipped source,
          2378-line TU)       30.7-40.8   1.00x
```
-- CITATION(roofline_probe, device 0, device event timestamps, interleaved, medians
-- over 12 reps. The GB/s ranges are ACROSS RUNS because this box is contended --
-- the RATIOS are the claim, never a single run's absolute.)

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

### The bisect: two candidates falsified, two arms LANDED BUT UNMEASURED

The next two steps were single-change arms off `VERBATIM`, each differing in ONE
place. They were **written and never measured**, because adding them pushed that
probe's translation unit past the level-zero `UR_RESULT_ERROR_OUT_OF_RESOURCES` (40)
limit on submit.

**That limit is the actionable finding**: *one probe TU holds roughly six to eight
INT2 kernel bodies.* The fix is to split the ladder across two binaries, not to shrink
it further -- a smaller ladder answers fewer questions. `examples/roofline_probe_bisect`
is that second binary (arms: `table` control, `VERBATIM`, `VERB+HALF`, `VERB+MP`,
`SHIPPED`, all interleaved), gated by `scripts/test_roofline_bisect.sh`.

**IT HAS NEVER COMPLETED A RUN, AND NO RESULT IS CLAIMED FROM IT.** Every arm hangs
including `table`; each arm tested ALONE also hangs, so the fault is **not** in the
shipped call; the sibling `roofline_probe` completes in 6.3 s on the same box at the
same moment, so it is not a GPU-wide outage; and the timed-out runs wedged the
driver's `kworker/*+xe_page_fault_work_queue` (a documented box condition --
`scripts/gpu_recover.sh`, remedy `rebind`, needs sudo).

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
arm would need if it wins: `reports/2026-10-06-decode1-gemv-bisect.md`.

### The two arms are now ENGINE FLAGS, default OFF — and NEITHER IS MEASURED

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

| | `INTFER_DECODE1_VERBHALF` | `INTFER_DECODE1_VERBMP` |
|---|---|---|
| the ONE change | group scale via `sycl::bit_cast<sycl::half>` instead of `from_half_bits` (the engine's branchy **scalar** FP16->FP32 twiddle, 3 branches per call, **320 calls per work-item** at MLP_GATE) | weight side through a row-base `multi_ptr` instead of an absolute index into raw `code_u8` |
| numerics | **NOT bit-exact by construction.** `from_half_bits` returns 0 for a zero or subnormal exponent and Inf/NaN when saturated, while `bit_cast<half>` hands a denormal to the FP32 convert, which **flushes to zero under the default FTZ**. A zero group scale is the differing case, and FP16 scales come from `to_half_bits` of a positive host float so it *should not* arise — but "should not" is an argument, so the arm is **stream-identity only and makes NO arithmetic claim**. Same class as `INTFER_DECODE1_SPLITK` | **bit-exact — AND NOW MEASURED RATHER THAN ARGUED** |
| **correctness, measured** | `test_rowpair_equiv.sh`: bitwise identical to base on these datasets, which does **NOT** prove the FTZ concern away — only that these synthetic scales never reach a zero or subnormal exponent | `test_rowpair_equiv.sh`: **BITWISE IDENTICAL to base on 227 835 element values**, 8 shapes x 5 datasets |
| **performance** | **UNMEASURED** | **UNMEASURED** |

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

*** `VERBMP`'s CORRECTNESS claim is settled; its PERFORMANCE is not. *** The source
said "bit-exact BY CONSTRUCTION — A CLAIM, NOT A VERIFIED PROPERTY", on the
`ROWPAIR` precedent that a bit-exactness *argument* is worth nothing until measured.
`test_rowpair_equiv.sh` was extended from two orders to FOUR (one PROCESS each --
the flags are read into a function-local `static const bool`, so an order is a
process, not a loop) and now reports `INTFER_DECODE1_VERBMP is BITWISE IDENTICAL to
base (227835 element values)`. **No model, no 27B load.** `VERBHALF` is identical on
these datasets too, which is information and NOT exoneration: the differing case is a
zero or subnormal group scale, and synthetic `to_half_bits(~1.0..1.5)` scales never
produce one.

**PERFORMANCE is UNMEASURED for all three and the direction is UNKNOWN, not
positive.** Both need
one 27B `--tp 2` run each, which AGENTS.md's standing rule 1 forbids an agent
session from performing. Blank `RESULT:` lines and the exact commands are in
`reports/2026-10-06-decode1-gemv-bisect.md`.

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

## ==> STACK STATE 2026-10-06: **NOT CERTIFIED. verdict=FAIL, failed=2.** <==

**The stack is red and the reason has NOT been established. Do not read any earlier
number here as current.** An earlier revision of this banner carried
`failed=0 stale=0 unproven=1`; that was measured BEFORE the bisect gate was added and
quoting it afterwards was a citation presented as a claim. Current, measured, on
`a0e3399`:

```
INTFER_QUICK=1 ./scripts/run_all_checks.sh
rc=1  verdict=FAIL  failed=2  stale=24  unproven=2  quick=1
  check_device_precond.sh  FAIL rc=1   (4 assertions)
  check_e2_ctx.sh          FAIL rc=1   (2 violations)
```

* **`stale=24` is not damage.** QUICK mode leaves matrices out of STEPS, so their
  stamps cannot match HEAD; that is what quick mode is. A non-QUICK run is needed for
  a gradeable cert, and one has not been obtained.
* **`failed=2` is NOT fixed.** The remedy needs sudo and has not been applied.

### The symptom, and it is the BASE run that is wrong

With **NO flags set**, the base 27B `--tp 2` run emits:

```
The capital of France isrowned_x财神      prefill token 42624, expected 11751
```

CITATION(measured 2026-10-06:
`build/device_precond_matrix/test_tp2_27b.healthy.log` line 2, saved with the rest to
`build/boxstate/diag_2026-10-06.txt`)

### The box attribution: PLAUSIBLE, AND **NOT CONFIRMED**

Evidence **for** a box fault: the failure is in gates that run the model; the base
run with no flags is the wrong one; `kworker/*+xe_page_fault_work_queue` was observed
spinning at **61 % CPU** on one sample.

Evidence **against** it, or at least complicating it -- and this is why the
attribution is NOT claimed:

* both render nodes present and `zeDeviceGet` reports **2 devices**
  CITATION(`scripts/gpu_recover.sh diag`, saved to `build/boxstate/diag_2026-10-06.txt`)
* the page-fault worker is **not** spinning at the next sample, so the 61 % figure is
  point-in-time and does not reproduce;
* `examples/roofline_probe` on the same box **passes its always-red twin** (ratio
  0.23 against an expected 0.25) and reads a **452 GB/s** streaming ceiling
  CITATION(`build/boxstate/probe_now.txt`, 99.2 % of the 456 GB/s spec -- the
  206-264 GB/s of earlier runs was a CONTENDED box, not a slow one), so GPU compute
  is behaving;
* **the engine has not changed since the tp2 gate passed.**
  CITATION(`git log a6c2c73..HEAD -- src/ include/ examples/intfer_cli/` is EMPTY.
  `a6c2c73` is the SUBGROUP quarantine, and `bash scripts/test_tp2_27b.sh` returned
  **PASS rc=0** immediately after it. Everything since is docs, `scripts/`, and one
  new example binary that no gate path invokes.)

That last item is the strongest discriminator available without root, and it points
AWAY from this session's code. **But it is not proof**, because a gate that passes on
a quiet box and fails on a loaded one is exactly the failure mode in question.

**WHAT WOULD SETTLE IT, and it needs sudo, so it is the operator's:**

```
sudo bash scripts/gpu_recover.sh rebind && bash scripts/gpu_recover.sh verify
bash scripts/check_device_precond.sh && bash scripts/check_e2_ctx.sh
```

**IF THOSE STILL FAIL AFTER A CLEAN REBIND, THIS IS A CODE INVESTIGATION AND THE BOX
ATTRIBUTION IS RETRACTED.** That retraction is written down here now, before the
evidence exists, so it cannot be quietly dropped later.

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
`sg.leader()` write. Full bisect and the order to test it:
`reports/2026-10-05-decode1-subgroup.md`.

### `INTFER_DECODE1_ROWPAIR` is GREEN and MEASURED

27B TP=2 ids match base, and `scripts/test_rowpair_equiv.sh` measures **227 835 element
values BITWISE IDENTICAL** to the base M=1 kernel over 8 shapes x 5 datasets in two
process invocations. It was **not** green until this session: it lost the device
(`UR_RESULT_ERROR_DEVICE_LOST`) on both INT2 shapes while passing FP16 and INT4,
because the branch is guarded on `bits == 2`. Full write-up:
`reports/2026-10-05-batch-upstream-simd-gemv.md`.

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
| `INTFER_DECODE1_SPLITK` | split-K block-cyclic M=1 INT2 decode: `N*S` work-items + coalesced weight loads. Exactly `=1`; kill switch `INTFER_NO_DECODE1_SPLITK` | **MEASURED LOSS.** Stream-identical to base on 27B (so the numerics contract held), but kernel 267 -> 384 ms (+43.8%, 3/3 same sign, clears the 6 ms spread floor; `nk/nw/nr` identical). Both halves of the hypothesis delivered and it still lost. **Do not promote.** `reports/2026-10-04-ksplit-decode1.md` |
| `INTFER_DECODE1_SUBGROUP` | llama.cpp mmvq shape: one SUB-GROUP per output row, lanes split K, reduced by `sycl::reduce_over_group` -- no `local_accessor`, no barrier. **PORTED 2026-10-05, PERF UNMEASURED.** Numerics are the already-measured cyclic contract (0 ulp realistic, 2 ulp adversarial); arm in `test_tp2_27b.sh`. Predicted NEUTRAL: removing a barrier from a kernel at ~9% of roofline is not a step change, and the weight REPACK that would make it large is deliberately NOT ported. `reports/2026-10-05-decode1-subgroup.md` |
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
  numerics argument: `reports/2026-10-05-tp2-fusion-surface.md`.
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
| `scripts/check_scripts_live.sh` | every `scripts/` entry is reachable: NAMED by a live source, or matched by the gate runner's own `GATE_GLOB` parsed out of `run_ten_gates.sh`. `docs/`, `reports/` and `scripts/archive/` are records, not call sites |

**The rules this repo learned by breaking them** (each cost a full run; the full
narrative with the measurements is in `docs/audit-log-tooling.md`):

1. **Enumerate, never remember.** A family is closed by deriving its instances from
   the source of truth, not from the names a previous fix happened to touch.
   `ls scripts/test_*.sh` DEFINES the gate population; `ls scripts/check_*.sh`
   defines the checker population; `scripts/check_scripts_live.sh` closes the
   *retirement* family, which is the one that needs closing most because a
   **`docs/` or `reports/` citation is not a call site** — that misreading is what
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
