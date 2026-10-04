# K-split M=1 decode (`INTFER_DECODE1_SPLITK`) — operator bench batch

**Date:** 2026-10-04
**Status:** kernel BUILT, compiles, default OFF, gated. **Never run on this
box from an agent session** (standing rule + the box's SYCL discovery currently
finds no device, see "Box state" below). The numbers you need are in "RESULT".

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
B580's ~179 GB/s.

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

**Pending the box (yours to run):**
- that the arm reaches the branch and the stream is identical;
- any performance verdict at all.

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

**None.** No 27B run was executed from this session: the standing rule forbids
it, and at the time of writing the box's in-process SYCL discovery returns no
device (L0 enumerates 2 via `scripts/l0_enumerate.sh`, but
`sycl::device::get_devices()` finds none — the documented container/host driver
version mismatch, not a code fault). So there is deliberately no baseline row
here rather than a number I could not attribute.

| arm | kernel | submit | waitblock | copy | usm_write | readback | nk/nw/nr |
|---|---|---|---|---|---|---|---|
| `INTFER_DECODE1_SPLITK=0` | — | — | — | — | — | — | — |
| `INTFER_DECODE1_SPLITK=1` | — | — | — | — | — | — | — |

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

## 8. RESULT:

```
RESULT: (operator fills this in from a quiet-box run -- counters, verdict)
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
