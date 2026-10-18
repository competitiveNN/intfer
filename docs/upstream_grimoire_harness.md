# GRIMOIRE harness survey (2026-10-15) — the two named patterns

**Source:** `github.com/doopeworld/GRIMOIRE`, default branch `main`, tree
`74c7aa988952595327e88ed8bf3f97a4dd2fed9f`, fetched via the GitHub API tree
(not cloned). The two files that carry the patterns the judge named:

* `autotune_b70_grouped.cpp` (45 lines of logic, 4.4 KB) — the sweep + cold autotune.
* `run_grouped_cold_autotune_b70.sh` (1.1 KB) — the two-stage build that makes
  the runtime knobs reachable.
* Supporting: `src/xe2_grouped_bridge.cpp` (the kernel whose `policy`/`mode`
  are runtime `int` arguments of `grimoire_xe2_grouped_mxfp4_autotune`,
  declared `extern "C"` at autotune_b70_grouped.cpp:26-29).

**`docs/upstream_sgl_survey.md:152` said "GRIMOIRE is not on disk" — that line
is now partly superseded: the project *tree* is fetchable via the API even
though no checkout exists. The line's purpose (stop future sessions from
`find /`) still holds.**

## What the project is (and is not)

GRIMOIRE is a vLLM-derived serving stack for Qwen3.5-27B class models in
**MXFP4 grouped-MoE** form on Xe2 (B70, `bmg-g31-a0`). Its licence is
**GPL-3.0-or-later** — so nothing here may be copied, only cited as a pattern,
which is the same posture as `docs/FORK_B580_SUMMARY.md:514`
("**No code from either project is used**").

Its hot kernels target an **E=256-expert grid** (`E` is a compile-time `int`
in `run_shape`, `rows[e]=M/E`). intfer's 27B is **dense with no expert grid**,
so the swept shape family (M=32768 routed tokens, `32x64`..`128x256` tile
policies, `1xEU`..`8xEU` EU modes) has **no analogue** in intfer's layer
GEMMs. That alone rules out a direct port.

## Pattern 1 — the runtime fusion-mask sweep

`policy` and `mode` are **runtime `int` arguments of the launch**, not
compile-time constants:

```
extern "C" int grimoire_xe2_grouped_mxfp4_autotune(
 sycl::queue*,const void*,const unsigned char*,const unsigned char*,void*,
 int,int,const int*,int,int*,int,int);
 /*                                              N  K  rows E atomic  ^mode ^policy */
```

`main()` then sweeps the product space — **9 policies × 5 modes = 45
candidates** — over *both* production shape families (gate_up
`32768×1024×2048`, down `32768×2048×512`), sums them, picks the winner, and
prints a CSV:

```
policy,groups,gate_up_ms,down_ms,total_ms,max_abs,status
64x128,1xEU,4.317,2.084,6.401,0.00531,PASS
...
WINNER policy=64x128 groups=1xEU total=6.401 ms candidates=45
```

`run_grouped_cold_autotune_b70.sh` is the enabler: it compiles the bridge with
`-DGRIMOIRE_ENABLE_AUTOTUNE` into a `.so`, then links the tidy main against it.
A tuner that needs the knobs at runtime has to be built as a separate binary
from a separately-compiled bridge — the pattern is inseparable from that build
split.

## Pattern 2 — cold autotune, and what the "hash-verified" part actually is

The determinism half is **not a hash**; it is a reference-comparison gate, and
the reference is captured from a *named* baseline configuration:

```c
(void)run_shape(q,5,0,32768,1024,2048,&ref_gu,ignored,cold,cold_bytes);  // policy 5 = "64x128", mode 0 = "1xEU"
(void)run_shape(q,5,0,32768,2048,  512,&ref_dn,ignored,cold,cold_bytes);
...
auto&e=all[p][m]; e={gu,dn,std::max(e1,e2), std::max(e1,e2)<=0.01};
...
std::printf("...,%s\n", ..., e.ok?"PASS":"REJECT_NUMERIC");
if(e.ok&&gu+dn<winner){winner=gu+dn;wp=p;wm=m;}
...
if(wp<0){std::puts("NO VALID CONFIGURATION");return 2;}
```

Six mechanisms, each distinct and each portable on its own:

| # | mechanism | where |
|---|---|---|
| 1 | **L2 cold-eviction before every timed rep** — `constexpr size_t cold_bytes = 768<<20;` then `q.fill(cold, i+1, cold_bytes).wait();` before `launch()`. The comment states the reason: "Grimoire streams a different expert block every layer. Evict the prior block so the tuner measures that production condition, not L2 reuse." | autotune_b70_grouped.cpp:52-56 |
| 2 | **min-of-3, never mean** — `best=std::min(best, elapsed)` over 3 reps. | :44-49 |
| 3 | **A reference captured once, from a NAMED baseline**, per shape family (`reference->empty()` guard), not from the first candidate measured. | :50-58 |
| 4 | **`REJECT_NUMERIC`** — a candidate that is faster but wrong is *rejected*, not reported as a win; the winner loop is `if(e.ok && ...)`. | :74 |
| 5 | **Deterministic constant synthetic** — `q.fill(x,bf16(.125f))`, `q.memset(w,0x22)`, `q.memset(s,127)`. Weights are constants, so `maxerr` measures *arithmetic*, never input variation. | :40-41 |
| 6 | **Hard refusal on an empty candidate set** — `NO VALID CONFIGURATION`, exit 2. Never picks a default. | :79-80 |

`768 MiB` is sized for a B70/bmg-g31 with 256 GRF per thread
(`-cl-intel-256-GRF-per-thread` in the runner's COMMON array). It is **not**
portable as a number and must not be quoted as a B580 figure.

## Portability verdict — the *decision* does not port; three *mechanisms* do

**The reason a port is refused is the decision instrument, and it is the
central one.** GRIMOIRE picks its winner on `std::chrono::steady_clock` host
wall-clock with L2 eviction. intfer's PARAMOUNT DIRECTIVE retired exactly that
instrument: *"Decide on counters, report counters, never wait for a quiet
box"* and *"never quote a tok/s that was not measured on a quiet box"*, with
`nk`/`nw`/`nr` plus the host buckets as the only load-insensitive surface.
The min-of-3 / mean-of-3 discipline is a *noise-reduction* refinement of a
load-sensitive instrument; reducing its variance does not make it
load-*insensitive*. min-of-3 is precisely the number that moved 4.7x between
two saved logs of the same kernel (AGENTS.md, "THE 6.39x ITSELF IS NOT
REPRODUCIBLE").

**Two further structural refusals, each sufficient on its own:**

1. **No shape family.** The swept space is a grouped-MoE GEMM over E=256
   experts; intfer's 27B is dense. There is no intfer kernel to hang a
   `policy`/`mode` pair on.
2. **GPL-3.0-or-later.** Pattern citation only; the repo's standing posture
   is that no upstream code enters the tree.

## What DOES transfer, and is already the repo's standing direction

| GRIMOIRE mechanism | intfer status |
|---|---|
| 1. cold-eviction before each rep | **Not portable as a number** (B70 256-GRF sizing), and not portable as a *discipline* until the profiling defect is fixed — a rebind, not a code change (`scripts/gpu_recover.sh`). Recorded so nobody ports the `768<<20` literal. |
| 2. min-of-3 | **Refused** — a refinement of the retired instrument. The repo's load-insensitive floor is `max(spread of the two arms compared)` with >= 2 paired rounds, enforced by `scripts/_bench_summary.py`. |
| 3. reference from a NAMED baseline | **Already shipped** — the gates' refusal directions are built from a named healthy source, and every arm asserts its own announcement. |
| 4. `REJECT_NUMERIC` / `NO VALID CONFIGURATION` | **Already shipped in substance** — every non-bit-exact arm is tolerance-gated and every gate refuses a vacuous run. Worth copying *verbatim as a shape* only if a future tolerance-class sweep has more than one candidate. |
| 5. deterministic constant synthetic | **Already shipped** — model-free probes (`rowpair_equiv`, `xmx_probe`, `gateup_equiv`) exist precisely so a numerics gate measures arithmetic and not weights. |
| 6. the **runtime knob product-space table** | **The one real gap, and it is harness-only.** intfer already has every *knob* — `INTFER_DECODE1_WG`, `INTFER_XMX_WG`, `INTFER_DPAS48_WG`, `INTFER_TILE_SCHED`, `INTFER_DPAS48_TILED`, `INTFER_LMHEAD_TILE_ROWS`, `INTFER_XMX_GEMM` — but there is **no harness that enumerates a product over one of them and prints a table**, so every knob is measured one env value at a time by hand. **CORRECTION 2026-10-15, same day as this survey:** the XMX_WG × WAVE_LAUNCH_FUSE interaction this row names was gated the same afternoon (`test_tp2_27b.sh` combined-flags arm, AGENTS.md "XMX × WAVE_LAUNCH_FUSE precedence, now observable"), so that specific pair is no longer a gap — it is observable in one run. What remains unenumerated is the *table*: the remaining pairs (DECODE1_WG × SPLITK, DPAS48_WG × TILE_SCHED, XMX_WG × DPAS48_TILED, …) still have no product harness, and even the closed pair is certified by a *single* combination rather than a sweep, so a knob whose effect is conditional on a second knob's value would still be invisible. **[SUPERSEDED 2026-10-18 — the gap this row names is CLOSED: `scripts/knob_sweep.sh` IS the product-space table harness. It enumerates a product over the three intra-device launch-geometry knobs (`INTFER_DECODE1_WG`, `INTFER_XMX_WG`, `INTFER_LA_CONV_PREP_WG`, overridable via `INTFER_SWEEP_KNOBS`), runs the 27B `--tp 2` target per combo, parses `[GPU profile @decode]` for the load-insensitive counters, and prints a baseline-relative table under the spread-floor rule with >= 2 paired rounds — which is the discipline row 2 already demanded, so the table is not a min-of-3 regression. UNMEASURED: the operator has not run it, and an agent session may not, because it loads the 27B. Registered in `scripts/run_all_checks.sh`'s non-QUICK band so it is reachable without being in the way; the sentence above is kept as the record of what was true when written, and AGENTS.md carries the same annotation.]** |

**Verdict: the two named patterns belong to a project with an expert grid, a
different weight format, a different GPU and a retired decision instrument. No
kernel ports. The single transferable item is the sweep-table harness over
intfer's existing runtime knob surface, and it is a reporting surface, not an
optimisation.** The survey is now closed across every named source including
GRIMOIRE.

## ASSESSED 2026-10-15 — the sweep-table harness is NOT ported

The row-6 gap was carried into this session as a pending port. It is assessed
here and **refused**, on three grounds recorded so the next session does not
re-derive it as "not yet attempted":

1. **It would duplicate a certification each knob already holds.** Every knob
   in that list ships with an announce line + a 3-part self-check in a gate arm
   (`test_tp2_27b.sh` / `test_gpu_optin.sh`), which is the repo's own pattern for
   certifying a runtime knob. A product table would be a *second* certification
   surface over the same facts — the "checker duplicates the thing it checks"
   shape the audit log warns about, applied to a population that is already
   covered.

2. **The one pair with a real arithmetic interaction is operator-only.**
   DECODE1_WG × SPLITK is not merely "observable by running both": SPLITK
   derives its partial count `S` from `nWG` (`quant_gemm.cpp:1680`), so the two
   flags compose arithmetically. Certifying that product means running the
   engine on the 0.8B fixture at several env combinations, which is the
   operator-only class — the same class every UNMEASURED flag in this tree
   sits in, and the PARAMOUNT DIRECTIVE says those are decided on `nk`/`nw`/`nr`
   from operator runs, not certified by an agent harness.

3. **It would produce no decision.** A product table over knobs that are all
   UNMEASURED on the 27B `--tp 2` target is a table of *reachability*
   assertions, and reachability is already what the individual arms assert.
   Nothing downstream reads it: the decision surface is `nk`/`nw`/`nr`, and no
   flag's verdict changes with an extra column.

**What the gap is worth knowing, and it is narrower than the row says.** The
XMX_WG × WAVE_LAUNCH_FUSE pair *was* gated the same afternoon this survey was
written (combined-flags arm in `test_tp2_27b.sh`, asserting the precedence line
is present), so that specific interaction is now observable in one run and the
row's example is stale. What remains genuinely unenumerated is the *table*: the
remaining pairs still have no product harness, and even the closed pair is
certified by a single combination rather than a sweep, so a knob whose effect is
conditional on a second knob's value would still be invisible. That is recorded
as a **known limitation of the certification surface**, not as an open porting
task — the repo's standing answer to "two knobs might interact" is the
per-flag announce line plus the refusal-direction self-check, and that answer
is already in force for all seven knobs.
