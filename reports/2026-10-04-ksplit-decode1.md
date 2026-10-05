# K-split M=1 decode (`INTFER_DECODE1_SPLITK`) — operator bench batch

**Date:** 2026-10-04
**Status:** kernel BUILT, **stream-identical to base on 27B**, and
**MEASURED SLOWER** — see §6. Default OFF, gated.
§8a carries the counter verdict I measured; **§8b is blank and yours** — it is
the 64-token oracle run, and until it runs `check_engine_cert.sh` correctly
reports the engine certification as STALE.

---

## 1. The flag

| | |
|---|---|
| Flag | `INTFER_DECODE1_SPLITK` |
| **Default** | **OFF** |
| Enables | **exactly `=1`** (`env[0]=='1' && env[1]=='\0'`) — matches the `INTFER_DPAS*` opt-in convention |
| Kill switch | `INTFER_NO_DECODE1_SPLITK=1` (any value) forces it off |
| Guarded by | `bits == 2` — the M=1 INT2 decode branch only |
| Split count `S` | derived at runtime: `S = nWG` (16) capped by `K/8`, reduced until `S \| nWG` |
| Certifying gate | `scripts/test_tp2_27b.sh` (the only gate whose model has an INT2 tensor) |

`INTFER_DPAS48`, `INTFER_DPAS48_TILED` etc. use the *same* exactly-`=1`
spelling. Note the contrast with `INTFER_DECODE1_MR`, which is
**presence**-enables: `X=0` turns MR *on*. Do not assume one spelling for both.

## 2. What it changes, and why it should help

The M=1 decode kernel launches **exactly `N` work-items, one per output
element**, and that count is invariant to the work-group shape. At 27B `--tp 2`
a layer GEMM is N≈2048 per rank across 512 Xe cores (~4 threads/EU), so the
launch is **parallelism-bound, not bandwidth-bound** — measured 9.16 GiB of
weights per device per token in 257.7 ms of kernel = 35.5 GiB/s, ~21% of the
B580's ~179 GB/s CITATION(SUPERSEDED 2026-10-05: the B580 has **20** Xe cores, not
512, and 456 GB/s of bandwidth, not ~179 — so the premise "parallelism-bound" does
not hold and the kernel is at ~9 % of the memory roofline. The SPLITK *measurement*
(+43.8 % on `kernel`) stands and the flag stays closed; see
`reports/2026-10-05-batch-gemv-roofline.md` for the re-examined form, which keeps
the 16-way K split but reduces it with a sub-group xor butterfly and no
`local_accessor`/barrier.
CITATION(9.16 GiB: `scripts/model_byte_roofline.sh` -> 9.752 GB/device/token.
CITATION(257.7 ms: counters-only `[GPU profile @decode]`.)).

The sibling experiment `INTFER_DECODE1_MR` attacked this by giving each
work-item *more* outputs and came out **2.3× slower**: dividing an
already-starved work-item budget by four is anti-parallel at M=1. This kernel
goes the other way — `N*S` work-items — so parallelism **rises**.

The K axis is split **block-cyclically**, which is the only assignment that
actually coalesces:

| assignment | byte each lane reads at step `j` (K=2048, S=16, INT2) | verdict |
|---|---|---|
| contiguous chunks | `32L + 2j` — lanes 32 B apart | **strided**, same gather as the un-split kernel |
| **block-cyclic** | `2L + 2j` — lanes 2 B apart | **coalesced** |

So it targets both halves of the stated goal (parallelism *and* coalescing).
Contiguous-chunk split-K was measured and rejected: it would have delivered
parallelism while leaving the stride-K gather intact.

Cost, stated up front: one `sycl::local_accessor<float,1>` of `nWG` floats per
work-group and a **local barrier**. That is new per-launch state the un-split
kernel does not have. `S` is chosen so all `S` work-items for one output land in
one work-group, which is what makes the local reduction legal.

## 3. Numerics — read this before judging the result

**This kernel is NOT bit-exact, by contract.** It reassociates the FP32 sum into
`S` partials combined ascending. The tolerance is **measured**, not guessed, by
`examples/ksplit_contract` (host model, ~1 s, gated by
`scripts/test_ksplit_contract.sh`):

| dataset | margin consumed, FP16 ulps after the store |
|---|---|
| random | **0** |
| act-outliers | **0** |
| full cancellation | **0** |
| adversarial escalating-magnitude | **2** |

Why: activations are stored FP16 (`|a| ≤ 65504`), which caps the dot product, so
FP32 reassociation error lands ~2 orders of magnitude below one FP16 ulp.

**The honest claim is "stream-identical on the gate models", NOT "bit-exact for
all inputs."** The second is false and the source comment refuses to claim it.
If the arm in `test_tp2_27b.sh` goes green, that means the contract holds for
that model — not that the kernel is exact.

## 4. What is already proven, and what is pending the box

**Proven here (no GPU, no 27B):**
- kernel compiles clean;
- the arithmetic order (`cyclic{S}`) is the one the contract measured, at
  0 ulp margin on realistic data;
- `scripts/test_ksplit_contract.sh` rc=0, with all five of its guards
  hand-proven non-vacuous (disarm the **data**, not the guard, and each gives
  rc=1);
- `test_tp2_27b.sh --self-check` rc=0 and `mutate_tp2_selfcheck.sh` **12/12**
  states caught, including two new ones that pin the cross-flag tag
  discrimination. Removing the cross-flag self-check cases makes **SK-12 go
  MISSED**, which is the proof those cases are load-bearing rather than
  decorative.

**Pending the box:** nothing blocking. Reachability and the stream verdict are
in §6. What remains is the full-gate run (64 tokens, fixed oracle) which stays
with you by the standing rule.

## 5. Exact commands

Box precondition: both B580s enumerating (`scripts/l0_enumerate.sh` →
`DEVICES=2`). Quiet-box rule `load1 < 2` **for the whole of every arm**.

```sh
cd /var/home/fra/dev/intfer
source /opt/intel/oneapi/setvars.sh
M=/run/host/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer
BIN=./build/examples/intfer_cli/intfer_cli
```

**A. The gate (correctness; this is the one that matters first):**

```sh
./scripts/test_tp2_27b.sh
```

Expect `TP2-27B TEST: PASS`, and that the log contains a
`[decode1-splitk] ... ACTIVE` line. **If that line is missing, the arm compared
base against base and certified nothing** — the gate fails on it, but read the
log to confirm which path ran.

**B. The counter A/B (performance):** 3 paired rounds, alternating arms.

```sh
for i in 1 2 3; do
  env INTFER_DECODE1_SPLITK=0 $BIN -m "$M" --gpu --tp 2 -n 64 -p "hello" \
      2>build/sk_base_$i.err 1>/dev/null
  env INTFER_DECODE1_SPLITK=1 $BIN -m "$M" --gpu --tp 2 -n 64 -p "hello" \
      2>build/sk_arm_$i.err  1>/dev/null
done
grep -h 'GPU profile' build/sk_*.err
```

Read the **load-insensitive buckets only**: `kernel` / `submit` / `waitblock` /
`copy` / `usm_write` / `readback`, and the integers `nk` / `nw` / `nr`.
`kernel` is the only load-insensitive bucket; the others are host wall-clock
accumulators — report them as `median(spread)` and read them as **relative**,
never as a figure.

## 6. Load-insensitive counters measured by me

**The "box cannot see a GPU" claim in the previous revision of this file was
WRONG, and it was my own missing environment.** I had not sourced
`setvars.sh`; `gate_gpu_count` correctly printed `SKIP: sycl-ls not available`
and I read that as box state. With setvars sourced the box is healthy:
`sycl-ls` lists 2 `[level_zero:gpu]`, `gate_gpu_count` -> 2, and the CLI prints
`GPU backend: 2 device(s) (TP=sharded)`. Recorded rather than quietly deleted,
because "I attributed an environment mistake to the hardware" is the exact
substitution the no-unattributed-measurement rule exists to prevent.

**Measured, 27B `--tp 2`, decode, `-n 8`, prompt "hello", 3 paired rounds,
alternating arms, after warm-up.** Read `kernel` only; the rest are host
accumulators and are shown for completeness, not as figures.

| arm | kernel (ms), per round | median | spread |
|---|---|---|---|
| `INTFER_DECODE1_SPLITK=0` | 265, 268, 267 | **267** | 3 |
| `INTFER_DECODE1_SPLITK=1` | 384, 380, 386 | **384** | 6 |

`nk`/`nw`/`nr` = **992 / 992 / 256 in both arms**, identical — so the delta is
attributable to the kernel and not to a changed launch/plumbing shape.

**Delta: +117 ms kernel, +43.8 %, 3/3 same sign, and it CLEARS the spread floor**
(floor = max(3, 6) = 6 ms; the delta is ~19x that). By the decision rule this is
a **LOSS**, not a win. **Do not promote.** The flag stays default OFF.

| arm | submit | waitblock | usm_write | readback | copy |
|---|---|---|---|---|---|
| base | ~325 | ~619 | ~173 | ~178 | ~0.7 |
| split-K | ~330 | ~829 | ~173 | ~180 | ~0.7 |

### Verdict, and the one thing this cost to learn

**The numerics contract held; the performance hypothesis did not.** Stream
identity was exact:

```
base    11 585 1044 264 5286 303 279 11759
split-K 11 585 1044 264 5286 303 279 11759
```

But reaching that took a **real bug in the kernel** — `(bv >> (j * 2)) & 3`
where the original has `(bv >> ((j & 3) * 2)) & 3`. INT2 packs four codes per
byte, so `j=4..7` must re-read bit 0 of the second byte; shifting a `uint8_t` by
8..14 is past its width, so the integer promotion made every one of those codes
0. It was localised by three facts together: the stream diverged, the
`INTFER_NO_DECODE1_SPLITK` kill switch stayed clean, and the contract said the
order was safe — which pointed at the implementation rather than the arithmetic.
`scripts/check_decode1_copies.sh` now pins the two inline copies so the next one
is caught in milliseconds.

**So why slower?** The hypothesis was parallelism and coalescing, and both were
delivered — `N*S` work-items, coalesced loads — and it still lost 117 ms. The
new per-work-group `local_accessor` plus the local barrier is the obvious
suspect against an already latency-bound kernel: the reduction is on the
critical path of every output. That is a hypothesis, not a measurement, and it
is stated as such. Per §9 the arithmetic stays INLINE in this kernel, which
means this cost cannot be amortized away by factoring.

**Recommended: leave the flag OFF and keep the kernel as the worked example.**
It is the first thing in this repo where the *numerics* contract passed and the
*performance* hypothesis still failed, which is worth more than the flag.

## 7. Expected direction, and what counts as a win

The hypothesis is `kernel` **down** at 27B TP=2 M=1, with `nk` **up** (more
work-items ⇒ more launches counted through the same GEMM path). `nk`/`nw`/`nr`
should be identical between arms apart from the added work-items — if they are
not, something other than the kernel changed and the comparison is void.

- **Win:** `kernel` strictly down on **≥ 2 paired rounds with the same sign**,
  and the delta **clears** the spread floor
  (`max(spread of the two arms compared)`, needs ≥ 2 paired rounds).
- **Neutral:** within spread. Record it as neutral; do not call it a win.
- **Loss:** `kernel` up, or any `id=-1` / stream divergence.

**Spread floor:** a single-round delta has no noise floor to clear and must not
be reported as a result. `scripts/_bench_summary.py` enforces this and refuses
a batch in which any arm died.

**If it loses, that is a real result, not a failed experiment** — record it in
the table above and in this file. The likely cause, if so, is the local barrier
plus `local_accessor` cost against a kernel that is already latency-bound; in
that case do **not** re-attempt per-work-item output tiling (measured, 2.3×
slower) and do **not** retry contiguous chunks (measured strided).

## 8. RESULT — two separate claims, and only one is mine

These are deliberately NOT one block. The protocol's rule is "a partially filled
RESULT is worse than a blank one", and the reason is that three wall-time claims
in this project were retracted for having no artifact behind them. The counter
verdict below is mine, measured, paired, and spread-checked. The full-gate
confirmation is yours and is still blank. Merging them would make a filled block
look like a completed one.

### 8a. COUNTER VERDICT — measured here, load-insensitive buckets only

```
RESULT (counters, measured 2026-10-04, 27B --tp 2 decode, -n 8, prompt "hello",
        3 paired alternating rounds, warmed):
        kernel   base 265/268/267 ms  median 267  spread 3
        kernel   split-K 384/380/386  median 384  spread 6
        delta    +117 ms  +43.8 %  3/3 same sign
        spread floor = max(3,6) = 6 ms; delta clears it by ~19x
        nk/nw/nr 992/992/256 IDENTICAL in both arms
        stream   IDENTICAL to base (11 585 1044 264 5286 303 279 11759)
        VERDICT  LOSS -- do not promote; the flag stays default OFF.
        NOTE     `kernel` is the only load-insensitive bucket; no tok/s or
                 wall-time figure is claimed anywhere in this file.
```

### 8b. FULL-GATE CONFIRMATION — blank, yours

This is the run that re-certifies the engine, and it cannot be done from an agent
session (standing rule). It is currently **STALE**: `src/gemm/quant_gemm.cpp`
changed, so `scripts/check_engine_cert.sh` reports that the ten-gate
certification no longer carries to HEAD. Your run clears it.

```sh
cd /var/home/fra/dev/intfer
source /opt/intel/oneapi/setvars.sh
scripts/run_ten_gates.sh      # 64-token oracle; the SPLITK arm runs here
scripts/check_engine_cert.sh  # must stop reporting STALE
```

```
RESULT (full gate, operator):
```


## 9. Two things deliberately NOT done

- **The arithmetic is not factored into a shared device function.** The previous
  fused-decode experiment extracted this same per-output body into a helper; a
  normalized statement-by-statement diff said the arithmetic was identical, and
  27B decode still went 1284 ms → 231 114 ms, bisected to the extraction and not
  to the fusion it was extracted for. Two homes for this arithmetic is the price
  of not repeating that. If you are tempted to factor it, that measurement is
  the evidence against.
- **No 0.8B gate arm.** The 0.8B model has zero INT2 tensors, so an arm there
  would be green *because the path never runs*. Re-derive the census before
  adding one:

```sh
git grep -c 'INTFER_DECODE1_SPLITK' -- 'scripts/test_*.sh'
```
