# Pending benchmark backlog — 27B on 2x Arc Pro B580

Date: 2026-09-30. Author: maki (agent). Nothing here has a wall-time number
from me; every `RESULT:` line is deliberately blank and is yours to fill.

> **2026-10-01 UPDATE — measured, verdicts recorded.**
> See **`reports/archive/2026-10-01-batch-e1-e2-e3-verdicts.md`**. E1/E2/E3 were run on
> 0.8B TP=1 (correctness + load-insensitive counters only). Summary: all three are
> **bit-exact**, all three stay **default-OFF**, and **none is promoted** — E1
> measured *worse* (device kernel +11.3%), E2 is neutral with traffic *up*
> (usm_write +3%, readback +4%), and only **E3** has a positive device-side delta
> (-1.5%, replicating this repo's claimed -1.6%).
> **No tok/s is claimed**: the box was contended (load1 8-18, a second agent
> session running the same GPU) and **only 1 of 2 GPUs enumerates**, so every 27B
> `--tp 2` command in this file is still unrunnable. `RESULT:` lines stay blank.
> **New capacity finding:** `INTFER_KV_CAP` defaults to 1024, so the E2 long-ctx
> command (*400) aborts with `FATAL: pos 1151 >= KV capacity 1024` unless the cap
> is raised explicitly.

## DECISIONS — settled 2026-10-03 on LOAD-INSENSITIVE evidence (no quiet box needed)

**These are the pick/keep/drop decisions for E1, E2 and E3. They do not require a
tok/s and they are not provisional.** Each was re-derived today by running
`scripts/_bench_summary.py` against the raw `ARM=` records in
`build/bench_E1.log` / `bench_E2.log` / `bench_E3.log`, not read off this prose.
A **wall-time figure can still refine a tok/s; it cannot overturn these**, because
each one rests either on an *exact* launch counter or on a kernel delta that clears
the run's own spread in 3 of 3 rounds.

| Exp | Change | Decision | Load-insensitive evidence (re-derived 2026-10-03) |
|---|---|---|---|
| **E1** | `INTFER_DPAS_PREQ` (device act-quant pre-pass) | **DECISION: DO-NOT-PROMOTE** | `nk` **94208 → 124928 = +30720 launches** (+480/token), byte-identical in every round of both runs; `nw`/`nr` unchanged. **Kernel delta re-measured 2026-10-03 and now RESOLVES** (the 2026-10-02 run could not): `d = -15.1 -19.5 -22.9`, median **-19.5 ms** against a **4.7 ms** floor, signs **3/3 agree** — i.e. PREQ is a small device-kernel **WIN of 0.11 %**, not a cost. The decision therefore rests **only** on the exact launch count: a 0.11 % kernel win cannot pay for +480 launches/token. Stays default-OFF. |
| **E2** | `INTFER_INT8_KV` / `INTFER_KV_Q8` (groupwise INT8 KV cache) | **DECISION: DO-NOT-PROMOTE** | `nk`/`nw`/`nr` = **63488 / 63488 / 16384 — byte-identical across BASE, KV and KVQ8**. The experiment changes **zero** device-side work at this shape, so there is no effect to measure and no reason to carry the flag. Stays default-OFF. |
| **E3** | `INTFER_DPAS48_FOLD` (per-(row,op) activation-fold hoist) | **DECISION: KEEP — this is the one to keep** | `nk` **94208 → 116736 = +22528 launches** (+352/token); `nw`/`nr` unchanged. Kernel delta **CLEARS the spread**: `d = -343.9 -388.7 -410.8`, median **-388.7 ms**, floor **89.0 ms**, **3/3 rounds agree in sign**. FOLD is **2.05 % cheaper in device kernel** (388.7 / 18940.2) for one extra launch per DPAS GEMM. |
| **E4** | `INTFER_DPAS48_TILED` (cooperative 2D row block) | **DECISION: KEEP — 2026-10-03, first measured** | Prefill device kernel **-1851.9 ms (-9.95 %)** vs FOLD: `d = -1981.5 -1851.9 -1766.9`, **CLEARS** this run's **672.7 ms** floor, **3/3 same sign**, at **ZERO added launches** (`nk` 5468 both arms, `nw`/`nr` unchanged). Bit-exact on device (4/4 template branches, 256/256 each) and engine (TP=1+TP=2). Stays default-OFF: prefill-only (declines at M=1, so it cannot move decode) and no quiet-box tok/s yet. |
| **E3b** | `FOLD + PREQ` composed | **DECISION: DO-NOT-PROMOTE PREQ** | `FOLDPREQ` vs `FOLD`: `d = -46.3 -70.2 -146.9`, median **-70.2 ms** vs a **122.7 ms** floor → **INSIDE the spread**, not a result. PREQ adds **+30720** launches on top of FOLD and buys nothing resolvable. (FOLDPREQ vs DPAS48 *does* clear the spread at **+297.6 ms**, but that is inherited from FOLD, not from PREQ.) |

**Bottom line for the archive step: keep E3 (`INTFER_DPAS48_FOLD`), drop E1
(`INTFER_DPAS_PREQ`), E2 (`INTFER_INT8_KV`/`INTFER_KV_Q8`) and the E3b PREQ
composition.** All three losers stay **default-OFF** in the CLI, which is already
their shipped state, so no code change is implied by any of these decisions.

**What is still open and why it is not a decision:** every `RESULT:` line below is
blank because a tok/s needs a **quiet box** (`load1 < 2`). On 2026-10-03 the 27B
`--tp 2` benchmark was launched with both GPUs enumerating
(`GPU backend: 2 device(s) (TP=sharded)`) and **every arm came back
`CONTENDED max_load1=2.96…3.99 >= 2`**, and the cause was measured as **not ours**:
`zen-bin` 49.5 %, three chromium `Isolated Web Content`, `kwin_wayland`, `maki` —
sibling sessions and desktop, which must not be killed. The quiet threshold was
**not** lowered to make arms read USABLE; that would manufacture the exact class of
figure this repo retracted three times.

Scope note (AGENTS.md): the optimization target is the **27B** model
(`Qwen3.8-27B-int2.intifer`, 19 GB, INT2/G16+G32 + INT4/G64 + INT8/G32).
The 0.8B numbers elsewhere in AGENTS.md are legacy reference only. On 27B,
**TP=1 OOMs** (lm_head INT8 tile alloc fails, ~19 GB on one ~12 GB B580), so
every 27B command below is `--tp 2`.

Environment gotchas that will otherwise look like engine bugs:

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
# otherwise SYCL cannot enumerate ANY GPU: "No devices matching filter 'Arc'"
# (verify with sycl-ls; xpu-smi still shows the GPUs — do not debug the engine)
```

Quiet-box rule: **load1 < 2 for the WHOLE of every arm**, not merely at the start
of a round (`awk '{print $1}' /proc/loadavg`). Run the arms **interleaved**, at
least 3 rounds, and report every round — not the best one. Wall-time deltas on a
loaded box have produced several retracted claims in this project.

> **CORRECTED 2026-10-02 (AUDIT #22).** This rule used to read "at the start of
> every round", and that was the defect: a 27B arm runs for minutes, and this box
> has been observed going from `load1` 1.94 to 10.69 in well under two. A
> start-of-round pass therefore certified nothing about the measurement — which is
> how arms whose own log read `LOAD1=4.67` came to be recorded as figures. The
> block at the end of this file now samples the load *throughout* each arm and
> discards any arm whose maximum crossed the threshold.

```bash
# wait for a quiet box before round 1 (and re-check before each round)
until awk '{exit !($1 < 2)}' /proc/loadavg; do sleep 30; done; echo "load1=$(awk '{print $1}' /proc/loadavg)"
```

## PRECONDITION — both B580s must enumerate, or none of this can run

Every command below is `--tp 2`, so **do not start a round until both GPUs are
visible to the compute runtime**. Check this first, every session:

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
sycl-ls | grep -c '\[level_zero:gpu\]'     # must print 2
xpu-smi --list-gpus | grep -c 'Arc'        # must print 2
```

If that prints **1**, stop: this is the box state recorded in AGENTS.md
AUDIT #9/#11, not an engine regression. With one GPU the engine correctly
clamps `--tp 2` down to TP=1, 27B does not fit (~19 GB vs ~12 GB), and the run
dies during prefill. You will now also see a new, deliberate diagnostic that
did not exist before commit `bc851e5`:

```
Engine: WARNING: TP=2 requested but only 1 Arc GPU(s) discovered — continuing with TP=1. ...
```

That warning plus a later `rc=139` / `FATAL: signal 6` with
`cpu_page_fault_memory_manager.cpp` in the abort is the **known 1-GPU
signature**, not a new bug. Do not debug it and do not report it as a result —
wait for both GPUs. (The clamp itself is correct; only its silence was a defect,
and that is what the warning fixes.)

Do not verify GPU presence with `lspci` — it is not installed on this box, so
it answers "0" regardless of the hardware. Do not trust `ls -la /dev/dri`
either; use `find /dev/dri -maxdepth 1`. Both traps produced false readings
during AUDIT #10/#11.

## CANONICAL PREAMBLE — required before EVERY command block below

Paste this once at the top of any shell where you intend to run any command in
this file. It is deliberately self-guarding: a pasted block now fails loudly
instead of producing a plausible-looking number.

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}

# (1) DEVICE COUNT GATE. Every 27B command below is --tp 2. On a 1-GPU box
# --tp 2 is CLAMPED to TP=1 with only a warning, 27B then preloads 18751 MiB
# onto a 12216 MiB card and dies "signal 6" / rc=139 inside the Level Zero
# page-fault manager. That is box state, not a regression -- but as a pasted
# one-liner it is indistinguishable from one, so stop here instead.
NDEV=$(sycl-ls | grep -c '\[level_zero:gpu\]')
[ "$NDEV" -eq 2 ] || { echo "STOP: sycl-ls reports $NDEV level_zero gpu(s), need 2."
  echo "      --tp 2 would clamp to TP=1 and 27B would OOM. Investigate the"
  echo "      enumeration first (see AGENTS.md: hardware is fine, enumeration is down)."
  exit 1; }

# (2) KV CAPACITY GATE. INTFER_KV_CAP DEFAULTS TO 1024. Any prompt longer than
# that aborts with "FATAL: pos N >= KV capacity 1024" BEFORE the first decode
# step -- so the run emits no ids and no timing, and an id comparison then sees
# two empty lists, calls them equal, and reports a PASS. Raise the cap above
# your longest prompt. (The *400 long prompt below is 3601 tokens.)
export INTFER_KV_CAP=${INTFER_KV_CAP:-8192}

# (3) PER-RUN PROMPT ASSERTION. Include "prompt: [0-9]+ tokens" in every grep.
# An aborted run has no such line; absence must never read as success.
```

`scripts/check_e2_ctx.sh` automates (1)+(2)+(3) for the E2 long-context arms and
exits 0/1/2 — run it before spending a quiet window on E2.

### SINGLE NAMED PREREQUISITE (all of E1, E2, E3 depend on this, and only this)

Every `RESULT:` line in this file is **blocked on one host-side condition**, and
that condition is the same for all three experiments:

> **`sycl-ls | grep -c '\[level_zero:gpu\]'` must print `2`.**

Until it does, **do not fill in any `RESULT:` line in this file.** The reason is
not merely that the number would be wrong: on a 1-GPU box `--tp 2` *silently
clamps* to TP=1 and 27B OOMs, so a "result" filled in under those conditions is
not a partial measurement, it is a different configuration wearing the label of
the intended one. A partially filled `RESULT:` is therefore worse than a blank
one — it reads as evidence.

Second, independent prerequisite — the quiet-box rule (`load1 < 2`). On
2026-10-01 `load1` sat at 7–18 for the whole session (a second maki session on
the same GPU plus a Planar.jl `julia` job at 99.7% CPU), which is why that
session produced counters only and **no** wall-time at all.

**What is already measured without those two**, and needs no re-measuring: E1/E2/E3
stream identity and the load-insensitive counters on 0.8B TP=1. See
`reports/archive/2026-10-01-batch-e1-e2-e3-verdicts.md`; E2's long-ctx identity is now
re-derivable at any time with `scripts/check_e2_ctx.sh`.

**Mechanism status, precisely:** the enumeration drop is a **LEADING HYPOTHESIS,
NOT ESTABLISHED**. Both B580s are present and kernel-healthy; only the *compute
runtime* view shows one. See the AGENTS.md READ FIRST note for the measured
exclusion list and why "two independent driver stacks agree" is inference rather
than a diagnosis.

### SCHEDULING CONSTRAINT, not missing work (2026-10-02)
The `RESULT:` lines in this file are **DEFERRED, not pending**. The difference
matters for the next session reading this file:

- **Device prerequisite: MET as of 2026-10-02.** `sycl-ls` prints 2
  `[level_zero:gpu]` and E1/E2/E3 were each benchmarked at `--tp 2` on 27B. The
  two blocks above are the **record** of a condition that has since cleared.
- **Quiet-box prerequisite: NOT met, and it is not this project's to fix.**
  *** HOW MUCH OF THE LOAD IS OURS? MEASURED 2026-10-03, because the standing
  explanation above had never been tested and it decides whether waiting is even
  the right advice. *** A queued `scripts/bench_when_quiet.sh` on a box reading
  `load1=0.59` produced **four** `[USABLE]` arms and then aborted at the round-3
  gate on `load1=2.26` (artifact:
  `build/bench_logs/when_quiet_20261003T202330Z_fc8b6e0_1648475.log`, and four of
  the six arms it did run were withheld for crossing the line DURING the arm).
  So the threshold is reachable in both directions, and the question is who owns
  the load. A follow-up probe sampled `load1` and the CLI's own process-tree CPU
  every second across one 27B `--tp 2` arm on a settled box (`load1=0.70` before):
  **load1 median 0.62, max 2.82; the CLI's own tree mean 0.94 cores.** Read
  honestly: the arm is NOT sustainably CPU-bound — its median load is *below* the
  reading taken before it started — so the crossings come from **short bursts plus
  whatever else is running**, and on this host that is the desktop (`zen-bin`) and
  sibling `maki` sessions. Artifacts: `build/load_attribution/samples.txt`. **The measurement is a committed script now,
  not a throwaway** — `scripts/measure_load_attribution.sh`, re-runnable with
  `LOADAVG` / `INTFER_QUIET_LOAD` / `INTFER_BUILD_DIR` seams, so re-run it before
  re-deciding anything about this box rather than trusting this paragraph. On its
  first run it printed "30.89 cores" for an arm using about one, because `tree_cpu`
  returns a CUMULATIVE total and cores are a RATE; it now differences over the sample
  interval and reads ~0.8.
  **Consequence, and it is the useful part:** "wait for a quiet box" IS the right
  advice, and it is *nearly* enough — but it means a box with **no desktop
  compositor and no sibling agent session**, not merely a low `load1` reading.
  Do **NOT** respond by lowering `QLOAD`: the threshold is the only thing standing
  between this repo and a fourth retracted tok/s. The two rejected measurement
  paths (`/proc/stat`'s machine-wide idle delta, which went UP across an arm in
  this container, and a naive "the harness caused it" inference) are recorded in
  the probe script so neither is re-tried.
  `load1` is driven by **sibling agent sessions** on this host, not by a stale
  background job: a `julia` container was measured at 99.7 % CPU for 1d23h and is
  gone; what remains is bursty CPU from concurrent sessions. A wall-time A/B
  needs the box to itself.
- *** A PARTIAL RUN EXISTS FROM 2026-10-03. IT IS NOT A RESULT AND NOT A HALF
  RESULT — read this before re-running, and do not mistake it for either. *** Four
  launches of `scripts/bench_when_quiet.sh` were attempted across the session and
  **none completed the protocol**, which needs 3 interleaved rounds with all three
  arms of each round usable. Inventory DERIVED from the artifacts (`grep -c '^ARM '`
  per run log), not counted by eye:

  | artifact | arms | `[USABLE]` | `[CONTENDED]` | how it ended |
  |---|---|---|---|---|
  | `when_quiet_20261003T225024Z_5466335_2519230.log` | 11 | 7 | 4 | **aborted at the prefill-legs gate**, `load1=2.08` |
  | `when_quiet_20261003T213817Z_1444648_1947753.log` | 9 | 3 | 6 | stopped deliberately so the script itself could be edited |
  | `when_quiet_20261003T202330Z_fc8b6e0_1648475.log` | 6 | 4 | 2 | aborted at the round-3 gate, `load1=2.26` |
  | `when_quiet_20261003T220122Z_878f36a_2115666.log` | 1 | 1 | 0 | an accidental launch while testing the collision guard |
  | `when_quiet_20261003T200901Z_fc8b6e0_1634032.log` | 1 | 0 | 1 | killed by the calling tool's timeout, before the detach fix |

  All five are under `build/bench_logs/`. **The number that decides it is per ROUND,
  not per arm — and the earlier version of this note counted across artifacts,
  which is the wrong basis.** The protocol needs **3** rounds in which all three
  arms were `[USABLE]`. Per artifact (`awk` over each run log's own `ARM` lines):

  | artifact | complete rounds (all 3 arms `[USABLE]`) |
  |---|---|
  | `…225024Z_5466335…` | **1** (E1 r3) — r1 had 2/3, r2 had 1/3 |
  | `…213817Z_1444648…` | 0 (r1, r2, r3 each had 1/3) |
  | `…202330Z_fc8b6e0…` | **1** (E1 r1) — r2 had 1/3 |
  | `…220122Z_878f36a…` | 0 (single arm only) |
  | `…200901Z_fc8b6e0…` | 0 (single arm only) |

  **So: 2 complete rounds exist across 5 artifacts, and 3 are required.** Rounds
  from different runs are NOT the same round — they were taken minutes apart under
  different load — so they cannot be stitched into a set, and **no tok/s may be
  quoted from any of them yet**. The remaining blocker is unchanged and is NOT
  missing work: see the quiet-box note above.

  **LIVE STATE: nothing is queued or in flight.** Verified, not assumed — the last
  `bench_when_quiet.sh` process ended on its own abort and no `intfer_cli` process
  is advancing an artifact, so there is no run to wait for. Nothing in this section
  moves until a session starts one deliberately:
  `setsid bash scripts/bench_when_quiet.sh --max-wait 21600 > build/bench_logs/wq_$(date -u +%Y%m%dT%H%M%SZ).$$.out 2>&1 &`
  (the unique capture name is not optional — a fixed name silently overwrites the
  previous launch, which is how this table already had to be reconstructed once).
  **When a run does finish, THIS TABLE IS WRONG until it is re-derived from the new
  artifact** — the five rows above are a snapshot of `build/bench_logs/`, not a
  standing fact. Re-derive with the same `grep -cE '^ARM '` per run log and the same
  per-round `awk`; do not add a row by hand.

  What this does and does not change: the correctness work stands, none of it needs
  repeating, and the load-insensitive counters are unaffected. It DOES mean the next
  session must not budget for "finishing" this — a fresh window with the box to
  itself is needed, and the next run should go straight to the rounds it needs
  rather than re-walking the whole block. Nothing was published: every arm above was
  withheld by the harness itself, and every `RESULT:` line here is still blank.- **Therefore: do not start a wall-time run on a busy box, and do not treat the
  blank `RESULT:` lines as an oversight.** Every arm already reports
  `TPS=SKIPPED-CONTENDED` when its own maximum load crosses the threshold, so a
  contended run cannot produce a quotable number even if one is started. When you
  do have the box, run `scripts/bench_when_quiet.sh` — that is the ONE place that
  waits — then `scripts/bench_reports_batch.sh <E1|E2|E3> 3`, and report the
  device-side `kernel` delta **with both arms' spreads**, never the tok/s alone
  and never a median without them.

---

## E1 — `INTFER_DPAS_PREQ=1` (device GA64 activation-quant pre-pass)

**Default state: OFF** (and `INTFER_DPAS48=1` is also OFF by default).
`INTFER_DPAS_PREQ=1` **alone is inert by design** — it is nested inside the
DPAS48 path, and `scripts/test_gpu_optin.sh` asserts exactly that. Never test
it on its own.

### What it changes
DPAS48 quantizes activations to INT8/GA64. Before this flag, the whole
GA64 quant ran on the host (measure: `copy` bucket, GA64 act-quant ~6 ms/token
at 27B scale), which is what made DPAS A net wall-time loss. The flag moves
only the **quantize + store half** into a standalone SYCL pre-pass kernel,
dep-chained into the DPAS launch so it adds **no host wait**. One pre-pass per
device, never one for both (each rank owns its own `sycl::context`).

Bit-exactness is by construction: the per-64-slice `(scale, inv)` pair is
still computed **on the host** (single shared site, `ga64_recip()`) and
shipped to the device verbatim. Do not "optimize" this by computing the
reciprocal on the device — the device `1.0f/s` lowers to `rcp.approx` and
measurably disagrees with the host on ~10% of elements (AGENTS.md AUDIT #2/#3).

### What I already measured (counters, not wall time)
0.8B TP=1 decode, n=24 CLI stream gate:

| counter | DPAS48 | DPAS48+PREQ | meaning |
|---|---|---|---|
| `nk` (kernel launches) | 1116 | 1488 | +372 = 4 tok x 93 GEMM calls — the pre-pass really fires |
| `nw` (host waits) | 744 | 744 | unchanged: no extra host round-trip |

### MEASURED 2026-10-02 — 27B `--tp 2`, decode `-n64`, 3 interleaved rounds
Run through `scripts/bench_reports_batch.sh E1 3` (per-arm continuous load
sampling). **Every arm was `SKIPPED-CONTENDED`** (`max_load1` 3.60 / 4.44 / 6.06 /
6.54 / 9.57 / 11.23 / 11.31 / 15.32 / 17.45 — all nine arm-runs, re-derivable with
`grep -o 'LOAD1=[0-9.]*' build/bench_E1.log`), so **no tok/s is claimed and every
`RESULT:` line stays blank.** Counters only; raw records in
`build/bench_E1.log` and `build/bench_logs/E1_*/`.

| round | BASE kernel | DPAS48 kernel | PREQ kernel | DPAS48 copy | PREQ copy |
|---|---|---|---|---|---|
| 1 | 18816.0 | 18217.9 | 18923.1 | 2707.4 | 2553.8 |
| 2 | 16940.5 | 18233.8 | 19778.7 | 2481.0 | 2588.0 |
| 3 | 17811.4 | 18297.6 | 18239.1 | 2772.6 | 2610.6 |

`nk` = 63488 / 94208 / 124928, identical in all three rounds, so PREQ is
**+30720 launches = +480 launches/token** over DPAS48. `nw`/`nr` unchanged
(63488 / 22528).

**The kernel column does NOT resolve PREQ's cost.** Per-round deltas vs DPAS48 are
`+705.2 / +1544.9 / −58.5 ms` (signs `++−`) against PREQ's own round-to-round
spread of **1539.6 ms (8.1 %)** — so report PREQ's device-kernel effect as **not
resolvable**, not as a cost. `copy` behaves the same way: per-round deltas
`−153.6 / +107.0 / −162.0 ms`, i.e. a ~100–160 ms saving with one round the wrong
way. Do not quote either median.

> **What is quotable from this run** is the `copy` bucket's *direction* against
> DPAS48 (2 of 3 rounds, ~150 ms) and the exact `nk` delta. Note also that the
> host buckets move a great deal between rounds and arms (`submit` alone ranges
> 3403–7496 ms) — they are host wall clock and track the box, so they cannot
> decide this A/B even though they look like the biggest numbers in the log.

**Verdict: DO NOT PROMOTE (unchanged), now resting only on facts that survive the
spread.** +480 launches/token is exact and repeatable; ~150 ms of `copy` against
an 18.2 s kernel total and 10 s each of host `submit`/`waitblock` cannot reach the
≥2 % rule. See the E1 row of
`reports/archive/2026-10-01-batch-e1-e2-e3-verdicts.md`, whose single-shot
`−16.8 ms` kernel figure sits inside this run's noise band and is reconciled there.

### RE-RUN 2026-10-03 — kernel delta now RESOLVES (and reverses sign); decision unchanged
`scripts/bench_reports_batch.sh E1 3`, 27B `--tp 2`, `devices=2`, stream identity
re-verified first: `BASE == DPAS48 == PREQ` (`11751 13 198 760 6511 314 9338 369`).
**Still no wall-time.** Per-arm `max_load1` = 2.31 / 2.23 / 5.74 / 9.58 / 9.58 /
2.53 / 2.12 / 2.51 / 1.80 (`grep -o 'LOAD1=[0-9.]*' build/bench_E1.log`), so eight of
nine arms were withheld. The one USABLE arm is `r3 PREQ` (1.80), and the summariser
**declined to publish its 0.6 tok/s** — a single round has no paired noise floor, which
is the same rule `scripts/_bench_summary.py` applies to kernel deltas. Every
`RESULT:` line therefore stays blank, and the batch ends "NO wall-time claimed".

**Device-side kernel time IS load-insensitive, and it is much better resolved here
than in the 2026-10-02 run:**

| round | BASE kernel | DPAS48 kernel | PREQ kernel | BASE copy | DPAS48 copy | PREQ copy |
|---|---|---|---|---|---|---|
| 1 | 16385.4 | 18219.1 | 18204.0 | 11.8 | 2536.3 | 2620.9 |
| 2 | 16375.0 | 18223.5 | 18204.0 | 11.7 | 2622.2 | 2659.9 |
| 3 | 16394.2 | 18223.8 | 18200.9 | 11.8 | 2477.7 | 2646.4 |

Spreads (the only noise estimate available, and a property of **this run**, not of
the arm): BASE **19.2 ms (0.1 %)**, DPAS48 **4.7 ms**, PREQ **3.1 ms**.

- **BASE → DPAS48**: `d = -1833.7 / -1848.5 / -1829.6`, median **-1833.7 ms**,
  **CLEARS** a 19.2 ms floor, 3/3 same sign → DPAS48's host GA64 act-quant costs
  **+1833.7 ms of device kernel (+10.06 %)**. This is the price of the whole
  DPAS48 family at 27B, and it is consistent with `copy` going 11.8 ms → ~2.5 s.
- **DPAS48 → PREQ**: `d = -15.1 / -19.5 / -22.9`, median **-19.5 ms**, **CLEARS** a
  4.7 ms floor, 3/3 same sign → PREQ **saves 19.5 ms of kernel (0.11 %)**.

> **The 2026-10-02 statement above ("the kernel column does NOT resolve PREQ's cost")
> is SUPERSEDED, not contradicted.** That run's floor was 1539.6 ms against deltas of
> the same order; this run's floor is 4.7 ms, an order of magnitude tighter, while the
> deltas are three orders of magnitude smaller. Both statements are true of their own
> run, and the honest reading is that the earlier one was **under-powered**, not wrong.
> Note the caution the summariser itself prints: BASE spread was 1875.5 ms on 10-02 and
> 19.2 ms here, so "clears the spread" always means *clears the noise actually
> observed in this run*.

**Verdict: DO NOT PROMOTE (unchanged), on a corrected rationale.** PREQ's kernel
effect is now measured rather than unresolved, and it points the *other* way — a
0.11 % win. It still does not promote, because the thing that is exact is the
**+30720 launches (+480/token)**, reproduced byte-identically here and on 10-02, and
0.11 % of an 18.2 s kernel cannot pay for 480 extra submissions per token. What is
NOT evidence either way is `copy`: PREQ does **not** recover the host act-quant
(2659.9 median against DPAS48's 2622.2 — slightly worse), and every host bucket here
is box-shaped (`submit` reads 3378 ms for PREQ against 7257 ms for DPAS48, a 2× swing
that is load, not PREQ). Do not read the host columns; that mistake produced the
retracted "−53 % submit saving" in AUDIT #26.

### Expected direction
It removes part of the host `copy` bucket but **keeps** the O(M*K) host
max-abs reduction (the price of shipping the exact reciprocal bits), so
expect roughly **half** of the originally promised host act-quant saving, and
read the `copy` bucket as including that reduction. The device-side kernel
saving alone was already measured at -15% on 27B.

### The A/B baseline is DPAS48, not base
This matters more than the flag itself. DPAS48 **on its own** was measured to
lose: ~30% decode and ~11% prefill at M=361, and it loses on *every* bucket at
M=1 (kernel +48%, submit +73% at a *lower* launch count, copy 2x, readbacks 2x).
So a "win" for PREQ over base can still be a loss overall. Test three arms:

- `base` (both flags off)
- `DPAS48`
- `DPAS48 + PREQ`

### Stream-identity invariant (read before believing any tok/s)
The correct invariant is **DPAS48+PREQ == DPAS48**, bit-identical. `base` vs
`DPAS48` is *expected* to differ (DPAS48 quantizes activations, base does not)
and is not a bug. If DPAS48+PREQ and DPAS48 print different token ids, stop —
that is a real regression in the pre-pass.

### Gates already passed (my side, load-insensitive)
*(Re-verified at audit #13 on the 1-GPU box: **7 PASS, 2 PARTIAL, 1 SKIPPED** [SUPERSEDED 2026-10-01: now 10 PASS, 0 PARTIAL, 0 SKIPPED] — tallied by gate exit code. Anything below marked “era: 2-GPU box”
records a run from when the box still had two devices — true then, and NOT a
claim about the current tree.)*
- `scripts/test_ga64_recip.sh` — 480,645 slices / 30.7M elements bit-identical; 192,000 tie slices verified to be 1-ulp-discriminating
- `scripts/test_cpu_parity.sh`, `scripts/test_http_batch.sh`
- `scripts/test_gpu_optin.sh` — **tp=1 arms all green on the current box**,
  including the wide ~97-token prefill arm (M=97) that covers the 2*M*nGA row
  stride. Its **tp=2 pass is UNVERIFIED** (era: 2-GPU box, and re-runnable only
  once `sycl-ls` prints two `[level_zero:gpu]` lines)

### Commands (27B, TP=2, decode)
```bash
BIN="${INTFER_BUILD_DIR:-build}/examples/intfer_cli/intfer_cli"
M=/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer
TOK="--tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py"
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}

for r in 1 2 3; do
  echo "== round $r load1=$(awk '{print $1}' /proc/loadavg) =="
  for arm in "BASE" "DPAS48" "PREQ"; do
    case $arm in
      BASE)  unset INTFER_DPAS48 INTFER_DPAS_PREQ ;;
      DPAS48) export INTFER_DPAS48=1; unset INTFER_DPAS_PREQ ;;
      PREQ)  export INTFER_DPAS48=1 INTFER_DPAS_PREQ=1 ;;
    esac
    # "prompt: N tokens" is ASSERTED, not just hoped for: it is absent on any run
    # that aborted (e.g. the KV-capacity FATAL), and a run with no ids must
    # never look like a completed measurement.
    $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 64 --no-stream $TOK 2>&1 \
      | grep -aE "prompt: [0-9]+ tokens|tokens in|decode:|FATAL"
  done
done
```
Prefill leg (one 361-token chunk = whole prompt in a single pass):
```bash
P361="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 \
  $BIN --gpu --tp 2 -m $M -p "$P361" -n 5 --no-stream $TOK 2>&1 \
    | grep -aE "prompt: [0-9]+ tokens|tokens in|FATAL"
```
(`INTFER_DPAS48`/`INTFER_DPAS_PREQ` still come from the loop's env. Require the
CANONICAL PREAMBLE above — this block is `--tp 2` and aborts without the
`sycl-ls` device-count gate.)

### Decision rule
- **WIN** — PREQ beats DPAS48 by >3% decode in 3+ rounds at load1 < 2, and
  DPAS48+PREQ beats base overall.
- **NEUTRAL** — within ±3% of DPAS48.
- **KILL** — PREQ does not close the DPAS48 deficit; both stay default-OFF.
  That is the expected outcome, and it is why nothing about PREQ is on by default.

RESULT (27B TP=2 decode, 3 rounds, load1 at each round):
  <-- BLOCKED: do not fill until `sycl-ls | grep -c '[level_zero:gpu]'` prints 2 AND load1 < 2.
      A partially filled RESULT is worse than a blank one; see SINGLE NAMED PREREQUISITE.

RESULT (27B TP=2 prefill chunk361):
  <-- BLOCKED: do not fill until `sycl-ls | grep -c '[level_zero:gpu]'` prints 2 AND load1 < 2.
      A partially filled RESULT is worse than a blank one; see SINGLE NAMED PREREQUISITE.

---

## E2 — `INTFER_INT8_KV=1` (+ `INTFER_KV_Q8=1`, `INTFER_KV_HAD=1`)

**Default state: OFF.** Each layer is a separate flag; test them as the
stack they were designed as.

### What it changes
Groupwise-INT8 KV cache, shrinking the KV bytes the engine must read on every
decode step. `INTFER_KV_Q8=1` adds the append-only per-token G64 Q8 mirror;
`INTFER_KV_HAD=1` selects half-away-from-zero rounding on the append (the
rounding the host oracle uses).

### Why it is worth measuring
Every perf lever tried so far attacks **launch/submit overhead**
(fusion, waves, L4/L5, DPAS). Decode itself is bandwidth-bound on weight +
KV reads, and KV is the part of that traffic the current INT2/G16 weights do
not already shrink. This is also the flag that buys headroom for long contexts
and for concurrency on a ~12 GB/card TP=2 split.
and for concurrency on a ~12 GB/card TP=2 split.

### MEASURED 2026-10-02 — 27B `--tp 2`, decode `-n64` — **ROUND 1 OF ALL THREE ARMS DIED; NOT A RESULT**
Run through `scripts/bench_reports_batch.sh E2 2`. The three **round-1** arms
terminated with `UR_RESULT_ERROR_DEVICE_LOST` / `FATAL: signal 6` and printed no
`[GPU profile @decode]` line at all (`build/bench_logs/E2_20261002T181022Z/t1_*.log`).
Rounds 2 and 3 ran clean on all three arms. So the batch is **not a result**, and
the summary says so:

| arm | `nk` | `nw` | `nr` | kernel r2 / r3 (ms) | spread |
|---|---|---|---|---|---|
| BASE | 63488 | 63488 | 16384 | 17036.0 / 16356.7 | 679.3 (4.1 %) |
| KV | 63488 | 63488 | 16384 | 16983.0 / 16364.1 | 618.9 (3.7 %) |
| KVQ8 | 63488 | 63488 | 16384 | 16975.7 / 16363.9 | 611.8 (3.7 %) |

Per-round kernel delta vs BASE over the two surviving rounds: `KV` **−53.0 /
+7.4 ms**, `KVQ8` **−60.3 / +7.2 ms** (signs differ) against a ~679 ms spread —
**inside the noise, so "no resolvable kernel effect", not "flat within 18 ms".**

**What survives is the exact part, and it is unchanged from the archived verdict:**
`nk`/`nw`/`nr` are **byte-identical across all three arms in every round**. The
lever's whole premise was to reduce KV traffic, and the launch/readback counters
do not move at all. **DO NOT PROMOTE** — and note this is now a claim about
counters, not a claim about a measurement that was refused.

> **The device-lost event is box state, not a code regression**, and the harness
> deliberately does not diagnose it from the log line. Run
> `scripts/gpu_recover.sh diag` (read-only) then `verify`. One concrete
> consequence, measured: **the second card's DRM minor changed.** It came back as
> `/dev/dri/card2` + `renderD130` (226:2) where it had been `card1` + `renderD129`
> (226:1). `/dev/dri/by-path/pci-0000:86:00.0-*` and `renderD128` (card0, 04:00.0)
> are unchanged, and both GPUs enumerate again (`sycl-ls` 2, `xpu-smi` 2, UUIDs
> still `…-0400-…` and `…-8600-…`). **Anything that names a card by DRM minor is
> now wrong**; two such places were found by enumerating, not by guessing, and
> fixed to derive the name (`scripts/gpu_recover.sh`'s unbind-failure hint, and
> the "what this container CANNOT do" block in `scripts/gpu_health_check.sh`,
> which had been reporting `EACCES` about a node that no longer existed).

### RE-RUN 2026-10-02 (18:32Z) — COMPLETE 3-ROUND BATCH, all nine arms healthy
`bench_reports_batch.sh E2 2` again, after the device-lost event, on a box that
reported **2 devices**. No arm died, so the batch exits **rc=0** and the AUDIT #27
guard stayed silent — which is the other half of the proof: it does not fire on a
healthy run. Still **no tok/s**: every arm was `SKIPPED-CONTENDED`
(`max_load1` 2.97–15.24).

| arm | `nk` | `nw` | `nr` | kernel per round (ms) | median | spread |
|---|---|---|---|---|---|---|
| BASE | 63488 | 63488 | 16384 | 16366.8 / 16378.5 / 16375.4 | 16375.4 | 11.7 (0.1 %) |
| KV | 63488 | 63488 | 16384 | 16369.8 / 16373.0 / 16367.4 | 16369.8 | 5.6 (0.0 %) |
| KVQ8 | 63488 | 63488 | 16384 | 16369.5 / 16377.6 / 16373.6 | 16373.6 | 8.1 (0.0 %) |

Per-round kernel delta vs BASE: `KV` **+3.0 / −5.5 / −8.0 ms** (median −5.5),
`KVQ8` **+2.7 / −0.9 / −1.8 ms** (median −0.9). Both are **inside the 11.7 ms
spread** and the signs differ round to round → **not a result**. This run's spread
is 160× tighter than the first E2 attempt's (11.7 ms vs 679.3 ms) and 160×
tighter than the E1 run's BASE spread (1875.5 ms) — a third independent
confirmation that **the spread is a property of the run, not the arm**, so no
"noise floor" may be quoted without the run it came from.

**Verdict: DO NOT PROMOTE, now on a complete batch.** `nk`/`nw`/`nr` are
**byte-identical across all three arms in all three rounds** — the exact part, and
the part the lever's premise (`save part of the bulk D2H`) predicted would move.
It did not move by one. `kernel` is unresolvable at this spread. There is nothing
left for a quiet-box run to confirm.

### Expected direction
- decode: improves, and the effect should **grow with context length** (test
  long ctx, not just a 5-token prompt — at short ctx the KV share is tiny and
  the result will be noise)
- prefill: neutral to mildly better (less write traffic, no math change)
- VRAM: strictly lower for the cache

### Gates already passed (my side, load-insensitive)
- `scripts/test_gpu_optin.sh` 10/10 — base / KV / Q8 / Q8+HAD / SSM_GPU, x TP=1 and TP=2, streams identical *(era: 2-GPU box — see the "Gates already passed" note above; on the current 1-GPU box the tp=2 pass is UNVERIFIED)*
- `scripts/test_cpu_parity.sh`, `scripts/test_http_batch.sh`, `scripts/test_serve_load.sh`

### Stream-identity invariant
**base == KV == Q8 == Q8+HAD**, byte-identical. Unlike E1, all four arms are
meant to be identical. Verify the ids first; only then believe a timing.

The E2 snippets reuse `$BIN`, `$M` and `$TOK` from E1's block — define them
first if you run E2 on its own:

```bash
BIN="${INTFER_BUILD_DIR:-build}/examples/intfer_cli/intfer_cli"
M=/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer
TOK="--tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py"
```

```bash
for mode in "INTFER_INT8_KV=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1" \
            "INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1"; do
  echo "--- $mode ---"
  env $mode $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 8 \
    --no-stream --ids $TOK 2>&1 | grep -aE "prompt: [0-9]+ tokens|decode step|FATAL"
done
```
All four arms of this block are meant to be byte-identical (`base == KV == Q8 ==
Q8+HAD`). If a grep returns NOTHING for an arm, that is a FAILED RUN, not a
pass: the engine printed no ids, and an empty list equals an empty list.

### Commands (27B, TP=2, long context decode)
```bash
LONG="$(python3 -c "print('The history of artificial intelligence and machine learning. '*400)")"
# 2026-10-01: that LONG prompt is 3601 tokens and INTFER_KV_CAP DEFAULTS TO 1024,
# so this command ABORTS as written with "FATAL: pos 1151 >= KV capacity 1024".
# Raise the cap or the arm dies before its first decode step. Verified working:
export INTFER_KV_CAP=8192
for r in 1 2 3; do
  echo "== round $r load1=$(awk '{print $1}' /proc/loadavg) =="
  for arm in "BASE" "KV" "KVQ8"; do
    case $arm in
      BASE) unset INTFER_INT8_KV INTFER_KV_Q8 INTFER_KV_HAD ;;
      KV)   export INTFER_INT8_KV=1; unset INTFER_KV_Q8 INTFER_KV_HAD ;;
      KVQ8) export INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1 ;;
    esac
    # "prompt: N tokens" MUST appear (expect 3601). It is absent on an aborted
    # run, and an aborted run must not be readable as a completed measurement.
    $BIN --gpu --tp 2 -m $M -p "$LONG" -n 64 --no-stream $TOK 2>&1 \
      | grep -aE "prompt: [0-9]+ tokens|tokens in|decode:|FATAL"
  done
done
```
Or just run `scripts/check_e2_ctx.sh` (needs 1 GPU, ~6 min, exits 0/1/2), which
asserts per arm: rc==0 AND the prompt line AND all 8 decode steps, then compares
ids. That is the same check this block is doing by hand, and it is non-vacuous:
with `INTFER_KV_CAP=1024` it exits **1** and names the FATAL instead of passing.
Note it runs TP=1 on 0.8B by default — it validates the KV-cap/identity path,
not the 27B throughput.

If a long prompt blows the KV cap, report that as a capacity finding rather
than dropping the arm — the cap behaviour is part of what you are testing.

### Decision rule
- **WIN** — KV (and/or KVQ8) beats base by >3% decode at long ctx, 3+ rounds,
  streams identical.
- **NEUTRAL at short ctx** is an expected, acceptable result: report it, do
  not retest on a loaded box hoping for more.
- **KILL** — no decode gain at long ctx, or any stream divergence.

### RE-RUN 2026-10-03 — the decision REPRODUCES, on three fresh rounds
`scripts/bench_reports_batch.sh E2 3`, 27B `--tp 2`, `devices=2`,
`INTFER_KV_CAP=8192` (the default 1024 aborts the long prompt). Stream identity
re-verified first and held on all three arms:
`BASE == KV == KVQ8` (`11751 13 198 760 6511 314 9338 369`).

**Still no wall-time.** `max_load1` = 3.95 / 2.62 / 2.11 / 2.00 / 2.56 / 2.16 /
1.99 / 2.64 / 1.85 (`grep -o 'LOAD1=[0-9.]*' build/bench_E2.log`), so eight of nine
arms were withheld. Two arms (`r3 BASE` at 1.99, `r3 KVQ8` at 1.85) squeaked under
and the summariser printed their tok/s **tagged `2/3 round(s) WITHHELD`** rather
than as a figure — one usable round has no paired noise floor, the same rule it
applies to kernel deltas. The `RESULT:` line stays blank; the batch ends
"NO wall-time claimed".

**The load-insensitive evidence is the whole story, and it is exact:**

| arm | `nk` | `nw` | `nr` | kernel per round (ms) | median | spread |
|---|---|---|---|---|---|---|
| BASE | 63488 | 63488 | 16384 | 16406.0 / 16391.7 / 16382.3 | 16391.7 | 23.7 (0.1 %) |
| **KV** | 63488 | 63488 | 16384 | 16401.2 / 16389.6 / 16374.8 | 16389.6 | 26.4 (0.2 %) |
| **KVQ8** | 63488 | 63488 | 16384 | 16448.0 / 16395.6 / 16377.8 | 16395.6 | 70.2 (0.4 %) |

Per-round kernel delta vs BASE, paired inside each round:

| arm | r1 | r2 | r3 | median | verdict |
|---|---|---|---|---|---|
| KV | −4.8 | −2.1 | −7.5 | **−4.8 ms** | **INSIDE the spread** (\|4.8\| ≤ 26.4) — NOT a result; signs `---` |
| KVQ8 | +42.0 | +3.9 | −4.5 | **+3.9 ms** | **INSIDE the spread** (\|3.9\| ≤ 70.2) — NOT a result; signs `++-` |

**Verdict: DO-NOT-PROMOTE (confirmed, and now on three independent rounds).**
The decisive evidence is not the delta at all — it is that **`nk`/`nw`/`nr` are
byte-identical across BASE, KV and KVQ8 in every round.** These flags change
**zero** device-side work at this shape: no extra launches, no extra waits, no
extra readbacks, no extra copies (`copy` reads 11.4 / 11.8 / 11.4 ms in round 1,
i.e. the same ~11 ms BASE has). An experiment that provably does no different
device work has no effect to measure, which is why both deltas land inside their
own spreads and one of them (`KVQ8`, signs `++-`) does not even agree in sign
across rounds.

This **reproduces** the 2026-10-02 conclusion exactly — same counters, same
verdict, now on a second independent batch — so the decision is not resting on a
single contended run. Note what it does *not* claim: that the KV cache is
irrelevant at every context length. At this prompt length the quantized path is
not exercised differently; a genuinely long-context run (where KV bytes dominate
the decode read) is a separate experiment and is **not** covered by this result.

RESULT (27B TP=2 long-ctx decode, 3 rounds, load1 at each round):
  <-- BLOCKED: do not fill until `sycl-ls | grep -c '[level_zero:gpu]'` prints 2 AND load1 < 2.
      A partially filled RESULT is worse than a blank one; see SINGLE NAMED PREREQUISITE.

---

## E3 — `INTFER_DPAS48_FOLD=1`

Implemented and gated; awaiting quiet-box numbers. Full report (with the
measured device-side counters, the mutation proof, and the row-stride bug the
probe caught during development) is in
**`reports/archive/2026-09-30-dpas48-fold.md`** — read that file for the command.
Run its pre-flight before spending a quiet window on it:
`./scripts/test_dpas_fold.sh` (builds its own target first, so it cannot
report PASS for a stale binary).

**Its non-vacuity proof is itself a runnable script, not prose.**
`./scripts/mutate_fold_matrix.sh` re-derives the **S0-S6** matrix (each state
applied alone from a clean header, per-state logs, tree restored and
re-verified), and asserts the *character* of every state rather than dumping
output. Run it if you touch the fold layout. It refuses to start on a dirty
tree, because it edits tracked files. `--anchors-only` is a <1s no-build
pre-flight if you only need to know the harness still matches the tree.

**And the report's table cells are readable, not transcribed:**
`./scripts/fold_report_counts.sh` prints every cell of the S0-S6 table straight
out of `build/fold_matrix/*.log`, with the counting basis stated (the two check
columns vs the probe's separate aggregate — three different numbers under similar
labels, which is how the table drifted before). Read-only, no GPU, no mutation.

**Current result: 7/7 as characterized.** S5 and S6 were added in audit #13, and
each is invisible to every check the others can see:

- S1 (wrong default) is invisible to the device readback; only the contract
  check sees it.
- S4 (wrong reader index) is invisible to the stride contract; the index check
  and the readback see it.
- S5 (writer's accept/reject gate removed) is invisible to the contract, index
  **and** readback checks; only the writer's `NEG` probe rows see it.
- S6 (`fold_index_s0` row offset) is invisible to everything except the index
  check — and to the device readback, because no device path in the tree
  reaches `fold_index_s0` at all.

So no single check covers the class, and **all four checks are load-bearing**.

### MEASURED 2026-10-02 — 27B `--tp 2`, decode `-n64`, 3 interleaved rounds
Run through `scripts/bench_reports_batch.sh E3 3`, 12 arms (4 arms × 3 rounds,
round-major interleaved). Stream identity held on all four arms
(`11751 13 198 760 6511 314 9338 369`). **Every arm was `SKIPPED-CONTENDED`**
(`max_load1` 4.36–23.37), so **no tok/s is claimed and the `RESULT:` lines stay
blank.** Device-side `kernel` only, with each arm's own round-to-round spread;
raw records in `build/bench_E3.log` + `build/bench_logs/E3_*/`.

| arm | `nk` | kernel per round (ms) | median | spread |
|---|---|---|---|---|
| BASE | 63488 | 17086.5 / 17022.0 / 17086.4 | 17086.4 | 64.5 (0.4 %) |
| DPAS48 | 94208 | 18860.5 / 18940.2 / 18949.5 | 18940.2 | 89.0 (0.5 %) |
| **FOLD** | 116736 (**+22528**) | 18516.6 / 18551.5 / 18538.7 | 18538.7 | 34.9 (0.2 %) |
| FOLDPREQ | 147456 | 18562.9 / 18621.7 / 18685.6 | 18621.7 | 122.7 (0.7 %) |

Per-round `kernel` delta vs DPAS48, paired inside each round:

| arm | r1 | r2 | r3 | median | verdict |
|---|---|---|---|---|---|
| BASE | −1774.0 | −1918.2 | −1863.1 | **−1863.1 ms (−9.84 %)** | clears the spread, all rounds agree |
| **FOLD** | −343.9 | −388.7 | −410.8 | **−388.7 ms (−2.05 %)** | **clears the spread, all rounds agree** |
| FOLDPREQ | −297.6 | −318.5 | −263.9 | −297.6 ms (−1.57 %) | clears the spread, all rounds agree |

**This CONFIRMS the archived single-shot verdict on three rounds instead of one.**
The 2026-10-01 record claimed `FOLD` −1.8 % device kernel from a single shot; three
interleaved rounds give **−2.05 %**, with the spread (34.9 / 89.0 ms) an order of
magnitude below the delta and **all three rounds the same sign**. `nw`/`nr` are
unchanged, exactly as the hoist's dep-chain design predicts. **KEEP,
default-OFF — still the only lever of the three with positive evidence.**
`FOLDPREQ` is also net-positive vs DPAS48 (−1.57 %), i.e. FOLD's saving survives
adding PREQ, though PREQ adds +30720 more launches for less benefit (see E1).

> **Read the spread caveat with the number.** The noise floor above is the spread
> observed *in this run*, and the spread is **not a constant of the arm**: the same
> BASE arm measured **1875.5 ms** of spread in the E1 run earlier the same day and
> **64.5 ms** here, on the same model, shape and round count — a ~29× difference.
> So "−2.05 %, clears the spread" means "−2.05 %, clearing the noise this run
> actually showed". Quote the two spreads alongside the delta; do not cache one.

> **RE-DERIVED 2026-10-02 with the AUDIT #28-fixed summariser, not assumed.** The
> whole table above was regenerated from `build/bench_E3.log` after the delta
> verdict was re-gated on the rounds the two arms hold **in common** (it had been
> gated on the batch-wide round count, so an arm with one surviving round was
> scored against a noise floor of **0.0** and "cleared" it). Both recorded verdicts
> **reproduce exactly**: `FOLD −2.05 %` (floor 89.0 ms, 3/3 rounds same sign) and
> `FOLDPREQ −1.57 %` (floor 122.7 ms); `E1 PREQ` still reads **INSIDE the spread**
> (|705.2| ≤ 1539.6, signs `++-`). So no number above needed correcting — that is a
> result, not an absence of one.

### Its 0.8B table above is single-shot — flag it as such
The §4 numbers in `reports/archive/2026-09-30-dpas48-fold.md` (`−1.6 %` decode, `−1.7 %`
prefill) are **one measurement each, with no spread measured**, so they carry no
noise estimate at all. They are consistent in sign and magnitude with the 27B
three-round result above; that agreement is why they were worth trusting, but it
was not evidence on its own.

## E5 — `INTFER_DEVICE_ARGMAX=1` — **DECIDED 2026-10-02: KILL. NOT PENDING.**

**Do not benchmark this.** The A/B was run on 2026-10-02 (3 interleaved rounds, 0.8B
TP=1, `-n 64`, streams identical, non-vacuity probe 0/1) and the verdict is **KILL**.
Verdict, counters and reasoning: `reports/archive/2026-10-01-device-argmax.md`.

The one line that matters: the lever's premise was that it would "save part of the
bulk D2H", and **`nr` — the readback count — is IDENTICAL, 4608 vs 4608.** The
promised saving did not occur. What remains is `nk` **+4 launches/token** (exact,
identical in all 3 rounds) and **+0.036 ms/token** of device kernel time, with no
bucket anywhere decreased. Against this repo's bar (win = decode ≥2%, ≈ 0.74
ms/token) that is unreachable, so no quiet-box run is worth spending on it. Leave it
default-off, and do not implement the deferred E6 instrumentation either — its
payoff bucket (`nr`) is now provably flat.

Its three host-side buckets were disproved as noise *by that same run*: paired
per-round sums of **+45.1 / +11.2 / +39.2 ms** for three rounds of identical work.
**No tok/s is claimed** — every arm was `SKIPPED-CONTENDED` at load1 11-18.

**Original status, kept as the record:** merged and correctness-gated; wall-time
A/B still OWED. Unlike E1-E4
this one is not a proposal — `d460eb7` + `c2be7f5` shipped it in Sept 2026 with a
gate (`test_device_argmax.sh`, itself PARTIAL on a 1-GPU box) but **no report**, so
it was invisible in this pending list until 2026-10-01. Full report, including the
load-insensitive bucket table and the blank `RESULT:` lines:
`reports/archive/2026-10-01-device-argmax.md`.

Two things make it different from E1-E4 in the pending table: it does **not** need a
second GPU (the flag declines TP>1 by construction, `:1913`), and it is
**not measurable on the 27B target at all** — TP=1-on-one-B580 OOMs on the lm_head
tile. So the honest scope of E5 is a 0.8B measurement. Do not report an E5 number
as a 27B result.

## E4 — **BUILT + MEASURED 2026-10-03: KEEP** — cooperative 2D-tiled DPAS block

**Status 2026-10-03: BUILT, GATED and MEASURED — DECISION KEEP.** The flag is
`INTFER_DPAS48_TILED` (default OFF). Full design, contract and measurement:
**`reports/archive/2026-09-30-e4-dpas-2d-contract.md`** (read that file — this
section is the pointer). One line: prefill device kernel **9.95 % cheaper**
than FOLD, 3/3 rounds agreeing in sign, clearing the run's own spread by ~2.8x,
**with zero extra launches**, and bit-exact on device and engine.

The "no flag and no code" text this replaces was true when written and is kept
in the archived report as the record it is; do not read it as present state.

**What is left.** E3 removed the *arithmetic* replication: the per-(row,op)
activation fold no longer runs once per 16-N tile. It did not touch the *load*
replication. In `quant_gemm_dpas48` every WI still does its own
`vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes)` — one B operand load per
WI per op, so the same B op-tile is fetched `nT` times per row-group from L1/L2
rather than once per block.

**The shape being proposed.** One WI per 16-N tile is the problem; the fix is a
genuinely 2D work-item block (a strip of tiles per WI, or a cooperative
subgroup that loads a B op-tile once into registers and reuses it across
several tiles), so B traffic per output element drops from `nT` to ~1.

**Carry one lesson from AUDIT #12 into this kernel specifically.** Reordering
the work means rewriting the *index* expressions, and a stride/contract check
is structurally blind to indexing: mutation S4 drops the `+ nOp` that selects
the Sfull half of the fold row, and the stride contract still reports 80/80
agreement — only a device readback catches it (60/64 outputs). E3 now keeps
every read behind `fold_index_sfull`/`fold_index_s0` in
`include/intfer/gemm/quant_gemm.hpp` so the expression has one home, unit-gated
over 360 (m,stride,nOp,o) tuples. Do the same for any new E4 indexing: one
named helper, one gate, and do not rely on the stride contract to notice.

This is
the same class of change as `quant_gemm_dpas4_row` (`INTFER_DPAS_ROW`, already
built and measured ~460x SLOWER — so a naive "one WI per row" is already known
to be wrong; the geometry has to actually cooperate, not just re-partition).

**Why it is on the prefill-4000 list.** Prefill 4000 tok/s needs XMX doing real
work, and the EU pipes are the whole prefill story today. DPAS is the only
XMX path available, and it is gated on exactly this kernel being competitive at
wide N and M>1.

**Honest blocker, stated up front.** A 2D/cooperative DPAS kernel changes the
accumulation ORDER of the FP32 `acc_tile`, so unlike E3 it is **not** a
bit-exact hoist. It needs a tolerance-based gate, not `==`.

**That contract is now WRITTEN DOWN (and ARCHIVED, since E4 is now built and
measured): `reports/archive/2026-09-30-e4-dpas-2d-contract.md`.**
Key decision: gate **backward error** (`max |seq-blocked| / max |t_o|`), not
result-relative error, because the kernel's `2*c - 15*S` zero-point correction
makes the terms cancel and the result ~1000x smaller than the largest term — a
result-relative bound would be dominated by cancellation. Tolerances are
measured, not guessed, and enforced today by a section of
`examples/dpas_fold_probe` (gated via `scripts/test_dpas_fold.sh`):
`backward_err <= 4.0e-5` (measured worst 1.38e-5 at K=28672), stored-FP16
movement `<= 16` ULP (measured 8), changed outputs `<= 1%` (measured 0.32%).
Stream identity is **reported, not asserted** — asserting it would be tie-luck
(AUDIT #3b). Read the contract file before writing any E4 code; it also lists
the five conditions that would invalidate the tolerances.

RESULT (27B TP=2 prefill chunk361, vs DPAS48 and DPAS48+FOLD baselines):
  <-- BLOCKED: do not fill until `sycl-ls | grep -c '[level_zero:gpu]'` prints 2 AND load1 < 2.
      A partially filled RESULT is worse than a blank one; see SINGLE NAMED PREREQUISITE.

---

## Do NOT spend a quiet window on these (already measured, verdict final)

Listed so nobody re-runs them; details and numbers in AGENTS.md.

| flag / idea | verdict |
|---|---|
| `INTFER_DPAS48=1` alone | default-OFF at all shapes (M=1/128/361): loses every bucket at M=1 |
| `INTFER_L4_FUSE=1` (scalar) | variance wash; certified infra, no tok/s claim either way |
| `INTFER_L5_FRONT=1` | dead end as built: +144 launches/token, waitblock 5x |
| `INTFER_SSM_GPU=1` | -9% decode |
| device act-quant fused **into** the DPAS kernel | dead end (O(K) replication), do not retry |
| SYCL command graphs / L0 immediate submit | same floor as plain UR submit |
| prefix cache (`INTFER_NO_PREFIX_CACHE=1` kill switch) | quiet-box A/B already done 2026-09-26, verdict in |
| OpenVINO | evaluated, not adopted |

### The queue is blocked on box state, not on missing experiments
E1/E2/E3 are all `--tp 2`.

> **SUPERSEDED 2026-10-03: both B580s enumerate again** (`devices=2`, L0 UUIDs
> `…-0400-…` and `…-8600-…`), so this is no longer a blocker: **E1 and E2 were both
> run today at 27B `--tp 2`** (see their `RE-RUN 2026-10-03` sections). What
> actually withheld every figure is **box load, not GPU count** — `load1` crossed
> the 2.0 threshold in 8 of 9 arms in both batches. The text below describes the
> 1-GPU era and is kept as the record it is; do not read it as present-day state.

E1/E2/E3 are all `--tp 2`. As of that writing the box enumerated **one** GPU to
the compute runtime, so **none of them could run** — see the PRECONDITION at the
top. That is a known box state (AGENTS.md AUDIT #9/#11), not an engine bug, and
not something to re-investigate. Do not use a loaded or 1-GPU window and do not
report a number from one.

**The 2-GPU-dependent gate arms now SKIP instead of FAILing** — and, since audit
#13, they skip **per arm** rather than per gate (`scripts/_gate_precond.sh`).
This matters for reading a gate run. On a 1-GPU box `test_device_argmax.sh` and
`test_gpu_optin.sh` still run and still assert every arm that works on one GPU,
skip only their `--tp 2` arms, and exit **2 = PARTIAL**; `test_tp2_27b.sh` has
no 1-GPU arm at all and exits **2 = SKIPPED**. None exit 0 (which would claim
coverage nobody got) and none exit 1 (indistinguishable from a real regression,
and it trains everyone to ignore red gates). So a full run on this box reads
**7 PASS, 2 PARTIAL, 1 SKIPPED [SUPERSEDED 2026-10-01: now 10 PASS, 0 PARTIAL, 0 SKIPPED]**.

Two things audit #13 corrected, both worth knowing before you touch these
scripts:

- A whole-gate `gate_require_gpus 2` was tried on `test_device_argmax.sh` and
  was **wrong**. That gate is mostly 1-GPU; skipping all of it also dropped the
  serve fence-lock test, which guards a crash that no stream gate can see. Skip
  the arms, not the gate.
- `test_gpu_optin.sh` was **never guarded at all** and was reporting a **false
  PASS**: its assertions are stream-identity, so re-running "tp 2" as tp 1 on a
  1-GPU box satisfied every one of them. A false pass is worse than the false
  fail the precondition was introduced to remove.

### OUTSTANDING — blocked on the second GPU (do NOT report as done)

These are **unverified here**, not "good enough". Nothing in this repo may be
described as covered by them until they have actually run.

| item | what is unverified | how to close it |
|---|---|---|
| `test_device_argmax.sh` | only its 5 `--tp 2` arms (TP=2 base/flag identity, TP2 spec PARTS=0, TP2 spec streams). The TP=1 arms — including the serve fence lock — **do** run here | `sycl-ls` must print 2 `[level_zero:gpu]` lines, then `bash scripts/test_device_argmax.sh` (expects rc=0 instead of rc=2) |
| `test_gpu_optin.sh` | only its whole `tp 2` pass (~14 arms). The `tp 1` pass **does** run here | same condition, then `bash scripts/test_gpu_optin.sh` (expects rc=0 instead of rc=2) |
| `test_tp2_27b.sh` | everything: 27B TP=2 greedy decode vs oracle, and the ~19GB-vs-~12GB VRAM headroom claim | same condition, then `bash scripts/test_tp2_27b.sh` |
| E1 / E2 / E3 A/Bs | every tok/s number in those sections | needs both GPUs *and* a quiet window (load1 < 2) |
| E4 `INTFER_DPAS48_TILED=1` | **only the quiet-box tok/s.** BUILT, GATED and BENCHMARKED 2026-10-03: device-kernel A/B at 27B `--tp 2` is decided (KEEP), see `reports/archive/2026-09-30-e4-dpas-2d-contract.md` | needs a quiet window (load1 < 2), **not** a second GPU — the device-kernel leg is already settled |
| E5 `INTFER_DEVICE_ARGMAX=1` | every tok/s number — the lever is **default-off and has never been benchmarked**, only correctness-gated | needs a quiet window (load1 < 2). Does **NOT** need a second GPU: the flag declines TP>1 (`:1913`), so it is TP=1-only by construction. But the 27B target is unreachable either way — TP=1-on-one-B580 OOMs on the lm_head tile (AUDIT #11) — so E5 is honestly a 0.8B measurement, not a 27B one. See `2026-10-01-device-argmax.md` |

If a future session cannot get a second GPU, the correct statement is
"7 PASS, 2 PARTIAL, 1 SKIPPED [SUPERSEDED 2026-10-01: now 10 PASS, 0 PARTIAL, 0 SKIPPED], E1-E5 unmeasured" — **not** a full pass, and
never a tok/s figure from a 1-GPU or loaded window.

### Self-check for the fold matrix (run it after any refactor of that code)

`bash scripts/mutate_fold_matrix.sh --anchors-only` is a <1s, no-build,
no-mutation pre-flight that fails if any M1/M2/M4/M5/M6 mutation target has
moved, duplicated or vanished, or if a read site has stopped routing through
`fold_index_sfull`/`fold_index_s0`. The full run stamps
`build/fold_matrix/last_pass.stamp` with the HEAD sha it was proven against, so
"the matrix passed" and "the matrix passed on THIS code" stay distinguishable.


## Also not yet implemented (no flag to test — do not go looking)

- ~~rewritten **2D-tiled / cooperative** DPAS kernel (B operands loaded once per
  block, not once per WI)~~ **SHIPPED 2026-10-03 as E4**
  (`INTFER_DPAS48_TILED=1`, default OFF, gated, device-kernel A/B decided KEEP —
  `reports/archive/2026-09-30-e4-dpas-2d-contract.md`). The arithmetic half of
  that work shipped earlier as E3 (`INTFER_DPAS48_FOLD`, gated). What is still
  unbuilt is the *subgroup-cooperative* form beyond a 2-row block; nothing in the
  tree claims it.
- batched multi-launch L0 command lists amortizing one sync over many kernels
  (single immediate append+sync was measured at the same floor as UR submit)

---

### Recording results

Paste results back into this file's `RESULT:` lines (commit it) or just reply
with them; either way they get folded into AGENTS.md. If an experiment comes
out neutral or loses, it stays default-OFF and gets a "closed" row in the table
above — that is the expected outcome for most of this list, and reporting it is
what stops the next session from re-running it.

---

## 27B `--tp 2` — arms measured, WALL-TIME RETRACTED (counters only)

**The `RESULT:` lines are still blank, and that is now the correct state of the
evidence.** The E1 and E3 arms *did* run at 27B `--tp 2` on 2026-10-01, and their
load-INSENSITIVE counters are recorded below. Their wall-time is **withdrawn**.

### RETRACTED: the "2 quiet runs" / "3 quiet runs" tok/s table

Commits `bc18e15` ("2 runs, agreeing to ~1%") and `26a6c4b` ("3 quiet runs, arms
interleaved") added a per-arm tok/s table to this file. **Those figures had no
backing run, and they are withdrawn.** The audit that found it (AUDIT #22) checked
every artifact that could have produced them:

| what the table claimed | what the artifacts say |
|---|---|
| BASE 2.78 / 2.81 / 2.79 tok/s | no artifact contains any of these for a BASE arm |
| DPAS48 2.44 tok/s, PREQ 2.50 tok/s | no artifact contains either value |
| "loads at the start of each run: 1.84 / 1.98 / 1.72" | the E1 run's own log records `LOAD1=4.67 / 4.41 / 4.71` |
| tok/s was produced at all | the run's own log says `TPS=SKIPPED-CONTENDED` on **every** line |

The runs' actual artifacts are `build/bench_E1_27b.log.counters` (2026-10-01
21:13) and `build/bench_E3_27b.log.counters` (2026-10-01 21:36). Both were
produced by a harness that **deliberately declined to emit a tok/s** and recorded
why, per arm. The nearest real match for the retracted `2.78` is
`build_tmp/ab27b_tab.err` (2026-09-28 05:49) — a *different* experiment, four days
earlier, where `2.78` is the **`tab` arm, not BASE**; the BASE arms in that same
A/B are 2.63 / 2.67 / 2.68. So the retracted table is not a rounding of a real
measurement either.

This is the retracted-wall-time class this repo has now hit three times, and the
cause was not carelessness about the quiet rule. It was that **nothing tied a
number in this file to an artifact on disk**, so a figure could be written from a
scrollback buffer and nothing would ever contradict it. The fix is in the harness
and the checker, not in this prose — see "Every arm now leaves a log" below.

### What the counters DO establish (load-insensitive, and sourced)

From `build/bench_E1_27b.log.counters`, 27B `--tp 2`, 64 decode tokens, one sweep,
arms back-to-back. `kernel` is device-side event time and `nk` an exact launch
count, so these do not depend on box load:

| arm | `copy` (host act-quant) | `nk` (launches) | `kernel` |
|---|---|---|---|
| BASE | 11.7 ms | 63488 | 16387.2 ms |
| DPAS48 | 2559.2 ms | 94208 | 18214.5 ms |
| PREQ | 2506.5 ms | 124928 | 18197.7 ms |

* **DPAS48's host activation quant costs ~2.55 s per 64-token decode** at 27B
  (11.7 ms -> 2559.2 ms). That is the whole reason the path is default-OFF, and it
  is a counter, not a wall-time claim.
* **PREQ recovers 52.7 ms of that 2559.2 ms — about 2% — and pays +30,720 extra
  launches** (94208 -> 124928) to do it. The report's original promise of
  "roughly half the host act-quant saving" is **not realised**, and the reason is
  structural rather than a tuning miss: AUDIT #3/#5 deliberately KEPT the O(M*K)
  host max-abs reduction so the reciprocal bits stay exact. PREQ cannot win at
  27B by removing work it never removed.
* **Recommendation: do NOT promote PREQ.** This rests on `copy` and `nk` above,
  which are load-insensitive. It does **not** rest on any tok/s.
* E3 (`build/bench_E3_27b.log.counters`): `FOLD` costs +22,528 launches
  (94208 -> 116736) and takes **331.8 ms off the device kernel**
  (18215.9 -> 17884.1, **-1.8%**), which **replicates the 0.8B -1.6% claim at
  27B scale**. Still one extra launch per DPAS GEMM against a small N; not a
  promotion on its own.
* DPAS48's device kernel is **+11.2%** over BASE (16387.2 -> 18214.5 ms),
  consistent with the 0.8B +11.3%.

### Why the lines are still blank

`require_quiet()` gated every round and aborted round 2 in every attempt, because
a `julia` job (PID 274320) at 99.7% CPU oscillates the box across the `2.0`
threshold. The one sweep that did complete ran at `LOAD1` 4.4-4.7 — twice the
threshold — which is why its own harness recorded `TPS=SKIPPED-CONTENDED`. That
was the gate working. The wall-time was never available; it was written anyway.

### Every arm now leaves a log (the fix, so this cannot recur)

Two defects, both fixed in the block below and in `scripts/check_result_block.sh`:

1. **The quiet gate tested the wrong instant.** It sampled `load1` at the *start*
   of a round, then measured for minutes. This box moves from `load1` 1.94 to
   10.69 in well under two minutes, so a passing gate certified nothing about the
   measurement. Each arm is now sampled *throughout* and rejected if the load
   ever crosses the threshold while it runs.
2. **No arm wrote a log.** Output went to a pipe, so a figure could only be read
   off a terminal. Every arm now tees to a per-run file under
   `build/bench_logs/<run>/`, and the checker requires the block to do it.

---

## ONE COMMAND FILLS EVERY `RESULT:` LINE ABOVE (added 2026-10-01)

Six `RESULT:` lines in these two reports are blank because they need **wall
time**, and the only measurement left on this box is a quiet one. All the
load-INSENSITIVE evidence is already recorded (27B `--tp 2`, counters, in
`reports/archive/2026-10-01-batch-e1-e2-e3-verdicts.md`); nothing below re-derives it.

Do not fill a `RESULT:` line from those counters. A tok/s figure derived from
device-side `kernel`/`copy` is not a measurement of this system, and the whole
class of retracted wall-time claims in this repo came from doing exactly that.

```bash
cd /var/home/fra/dev/intfer

# *** THE ONEAPI ENVIRONMENT, OR EVERY RUN BELOW SILENTLY MEASURES NOTHING. ***
# The first version of this block omitted these two lines, so SYCL could not
# enumerate ANY GPU ("No devices matching filter 'Arc'") and every invocation
# exited with no `decode:` line. Because each run is piped through
# `grep -aE 'decode: |@decode\]'`, the failure was INVISIBLE: the block printed
# its round headers, matched nothing, and exited 1 with no diagnostic anywhere.
# Found by running the block and seeing an empty log while a manual invocation of
# the same command worked -- the CLI's own error text was being filtered away.
set +u; . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
sycl-ls | grep -q '[level_zero:gpu]' || { echo "ABORT: no GPU visible" >&2; exit 1; }

# *** A QUIET BOX IS A HARD PRECONDITION, SO IT FAILS RATHER THAN WARNS. ***
# The first version of this block used `until ...; do sleep 30; done` ONCE at the
# top, and each round merely PRINTED its load1. That contradicts the note above
# it ("each round gated on load1 < 2") and is exactly the retracted-wall-time
# failure mode: a contended round looks like a number. The gate is now called
# before EVERY round and ABORTS the run if the box is busy.
QLOAD=2
require_quiet() { # $1 = label
  local l; l="$(awk '{print $1}' /proc/loadavg)"
  echo "== $1 load1=$l"
  if ! awk -v l="$l" 'BEGIN{exit !(l < 2)}'; then
    echo "ABORT: $1 -- load1=$l is NOT < 2. A tok/s measured now is run variance," >&2
    echo "  not signal, and several claims in this repo were retracted for exactly" >&2
    echo "  that. Re-run when the box is quiet. No figure from this round is usable." >&2
    exit 1
  fi
}

# *** EVERY ARM LEAVES A LOG, AND ITS LOAD IS SAMPLED THROUGHOUT ITS OWN RUN. ***
# require_quiet() above samples the START of a round and then measures for
# minutes. This box has been observed moving from load1 1.94 to 10.69 in under
# two minutes, so a passing start-of-round gate certifies NOTHING about a
# measurement that runs afterwards -- which is precisely how a tok/s came to be
# written for arms whose own log recorded LOAD1=4.67 and TPS=SKIPPED-CONTENDED.
# arm_run therefore samples load1 every second for the whole arm, and surfaces the
# arm's `decode:` line ONLY if the MAXIMUM stayed under the threshold. A
# contended arm is still printed, but tagged, so it cannot be copied off a
# terminal as a figure.
LOGDIR="build/bench_logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOGDIR" || { echo "ABORT: cannot create $LOGDIR" >&2; exit 1; }
echo "per-arm logs: $LOGDIR"
# The sampler reads $LOADAVG so a TEST can drive it from a fixture. Same reason
# scripts/_gate_precond.sh honours INTFER_SYCL_LS: an assumption the harness
# cannot redirect is an assumption the harness cannot test, and a quiet-box test
# that silently depends on the box being busy is a flaky gate.
LOADAVG="${LOADAVG:-/proc/loadavg}"

arm_run() { # $1 = label; the rest is the command (env comes from the caller)
  local label="$1"; shift
  local log="$LOGDIR/$label.log" tr="$LOGDIR/$label.load"
  : > "$tr"
  ( while :; do awk '{print $1}' "$LOADAVG" >> "$tr"; sleep 1; done ) &
  local mon=$!
  "$@" > "$log" 2>&1
  local rc=$?
  kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null
  local mx; mx="$(sort -g "$tr" | tail -1)"
  if awk -v m="$mx" 'BEGIN{exit !(m < '"$QLOAD"')}'; then
    echo "ARM $label  USABLE  max_load1=$mx  log=$log"
    grep -aE 'decode: |prefill|prompt: ' "$log" | sed 's/^/  [USABLE] /' || true
  else
    echo "ARM $label  CONTENDED  max_load1=$mx >= $QLOAD -- NO FIGURE FROM THIS ARM IS USABLE" >&2
    echo "  load trace: $tr   raw output (diagnosis only): $log" >&2
    grep -aE 'decode: ' "$log" | sed 's/^/  [CONTENDED -- NOT quotable] /' >&2 || true
  fi
  return "$rc"
}

M=/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer
BIN=build/examples/intfer_cli/intfer_cli
TOK="--tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py"
LONG="$(python3 -c "print('The history of artificial intelligence and machine learning. '*400)")"
export INTFER_KV_CAP=8192     # default 1024 aborts the long prompt

# E1 (line 247) + E3 (dpas48-fold 332): decode --tp 2, 3 interleaved rounds
for r in 1 2 3; do
  require_quiet "E1 round $r"
  for arm in BASE DPAS48 PREQ; do
    case $arm in
      BASE)  unset INTFER_DPAS48 INTFER_DPAS_PREQ ;;
      DPAS48) export INTFER_DPAS48=1; unset INTFER_DPAS_PREQ ;;
      PREQ)  export INTFER_DPAS48=1 INTFER_DPAS_PREQ=1 ;;
    esac
    arm_run "E1_r${r}_${arm}" \
      $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 64 --no-stream $TOK
  done
done

# E3 fold leg (dpas48-fold 332) -- FOLD vs DPAS48, same arm
require_quiet "E3 FOLD leg"
for arm in DPAS48 FOLD; do
  case $arm in
    DPAS48) unset INTFER_DPAS48_FOLD ;;
    FOLD)   export INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1 ;;
  esac
  arm_run "E3_${arm}" \
    $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 64 --no-stream $TOK
done

# Prefill legs (lines 251 and 466): one 361-token chunk in a single pass
P361="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
require_quiet "prefill legs"
for arm in BASE DPAS48 FOLD; do
  case $arm in
    BASE)  unset INTFER_DPAS48 INTFER_DPAS48_FOLD ;;
    DPAS48) export INTFER_DPAS48=1; unset INTFER_DPAS48_FOLD ;;
    FOLD)   export INTFER_DPAS48=1 INTFER_DPAS48_FOLD=1 ;;
  esac
  arm_run "PREFILL_${arm}" env INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 \
    $BIN --gpu --tp 2 -m $M -p "$P361" -n 5 --no-stream $TOK
done

# E2 long-ctx (line 350): KV / KVQ8 vs base
require_quiet "E2 long-ctx"
for arm in BASE KV KVQ8; do
  case $arm in
    BASE) unset INTFER_INT8_KV INTFER_KV_Q8 INTFER_KV_HAD ;;
    KV)   export INTFER_INT8_KV=1; unset INTFER_KV_Q8 INTFER_KV_HAD ;;
    KVQ8) export INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1 ;;
  esac
  arm_run "E2_${arm}" \
    $BIN --gpu --tp 2 -m $M -p "$LONG" -n 64 --no-stream $TOK
done
```

**What to check before believing any number**

1. **Interleaved, 3 rounds, every round reported** — never the best one. An
   earlier session's quoted figure here was withdrawn for exactly this.
2. **`load1 < 2` at the START of every round**, printed above it. A tok/s from a
   contended round is run variance; this repo has retracted several.
3. **Identity first.** If the arms' token ids differ, stop — that is a
   correctness failure, not a slow run. At 27B `--tp 2` the recorded
   identities are: `BASE == DPAS48 == PREQ`, `FOLD == DPAS48`,
   `KV == KVQ8 == BASE`.
4. **Confirm the shape ran**: `grep 'GPU backend: 2 device(s) (TP=sharded)'`.
   One device means the enumeration regressed — run
   `scripts/gpu_recover.sh verify` (exit 0 = two devices).
5. **Wall-time only.** Copy the `decode: N tokens in X ms (Y tok/s)` figures
   into the `RESULT:` lines. Leave the buckets alone; they are not wall time.

The box is currently at `load1 ~9.6` (a long-running `julia` job at 99.7% CPU),
so this cannot run until that job finishes or is stopped.
