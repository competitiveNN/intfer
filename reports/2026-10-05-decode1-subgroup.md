# `INTFER_DECODE1_SUBGROUP`: quarantined — wrong output, cause not located

> ## ==> THIS IS A POST-MORTEM OF A QUARANTINE, NOT A RESULT. <==
>
> The flag, its kernel and its certification arm were **removed from the tree** on
> 2026-10-05. The kernel produced a **numerically wrong answer** on every run, and a
> default-OFF flag that is wrong is worse than no flag: it is a kernel that looks
> certified because it has a gate arm, and it makes three gates red in sequence.
> **Nothing here is claimed fixed, because nothing here is understood.**

**Box:** 2x Arc Pro B580 · **Gate:** `scripts/test_tp2_27b.sh`, 27B `--tp 2`
**Status:** removed; `run_all_checks.sh` green again subject to §5

---

## 1. What it was, and why it existed

One **sub-group** (16 lanes, the width `examples/roofline_probe` queries from
inside a kernel on this device) cooperates on one output row: lane *L* takes the
block-cyclic K slice, and the 16 partials are combined by
`sycl::reduce_over_group(sg, acc, sycl::plus<float>())`, with
`sg.leader()` writing the result.

It existed because `INTFER_DECODE1_SPLITK` does the same 16-way split, measured
**+43.8 %** on `kernel`, and its bisection named the per-work-group
`local_accessor` plus `it.barrier` as sitting on the critical path of every output.
SUBGROUP keeps the split and removes exactly those two things. That diagnosis was
good; the implementation was not.

## 2. The symptom

```
[decode1-subgroup] M=1 INT2 decode: sub-group-per-row reduce_over_group ACTIVE (N=8704 K=5120 bits=2)
DECODE1_SUBGROUP ids:  '279 12 12'     <-- wrong
base                   : '13 198 760'
```

Deterministic, on every run, after the kernel **announces itself** — so the branch
is reached and the fault is inside the kernel, not a flag that failed to take effect.
`INTFER_DECODE1_SPLITK` and `INTFER_DECODE1_ROWPAIR` were green in the same runs.

## 3. The bisect: two candidates tested and FALSIFIED, one not reached

SPLITK is correct and was intended to differ only in the reduction, so **the K split is
shared and sound, and the reduction is the suspect.** Two specific mechanisms were
reproduced on the CPU against the kernel's own arithmetic before anything was
asserted:

| candidate | verdict | how |
|---|---|---|
| `step = row_lanes*kVSg = 128` while a group is only `Gw = 16` wide, so the chunk loop overshoots and **half of every group is dropped** | **FALSE** | Reproducing the phase formula `k = kg0 + ((off - (kg0 % step) + step) % step)` over the 8-group period (128/16) gave **128 of 128 k-values covered and all 16 lanes contributing**: group *g* is served by lanes *2g* and *2g+1*, and groups cycle with period 8. Complete coverage. |
| a lane offset in `code_u8[b0]` — the same shape as the `INTFER_DECODE1_ROWPAIR` double-offset that **lost the device** | **FALSE** | SUBGROUP indexes the **unoffset** `code_u8` with **absolute** byte offsets in both the chunk loop and the tail, which is correct. (ROWPAIR's bug was the opposite: row-offset `multi_ptr`s indexed with absolute offsets.) |

**So the cause is NOT located, and it is in one of two places that have not been
tested:**

1. **The `reduce_over_group` collective's interaction with this launch's sub-group
   geometry.** The kernel launches 16-wide work-groups and *queries*
   `sg.get_local_linear_range()`, deriving `sgs_wg = nWG / sgw` and
   `row = group * sgs_wg + sgid`. If `sgw` is not what the phase arithmetic assumed
   — or if the collective spans a different scope than `sg` — the tree sums the wrong
   set. **Not tested**: the kernel never asserts that `sgw` equals the width the
   phase formula used.
2. **The `sg.leader()` write.** Only the leader writes, and the leader is lane 0 of
   the sub-group. If more than one sub-group maps to a row (which the `sgs_wg`
   arithmetic exists to handle), a non-leader sub-group's result could be lost or
   the wrong lane could win. **Not tested.**

### The tooling that does not exist yet

The honest first step for whoever picks this up is a **model-free differential**:
`INTFER_DECODE1_SUBGROUP` vs the base M=1 body, **bitwise**, over 8 shapes × 5
datasets, in two process invocations. `scripts/test_rowpair_equiv.sh` is exactly
that shape, and it is what caught ROWPAIR's device-loss without the 27B stack. Its
absence here is why this cost a 27B gate run to find.

Concretely, and in this order:

1. Assert `sgw == 16` **and** that `row_lanes * kVSg` divides the cycle the phase
   formula assumes; fail loudly if not, rather than proceeding on an assumption.
2. Write every lane's partial to a distinct USM slot and compare **per lane** against
   the base body's per-lane slice. That says immediately whether the split is wrong
   or only the sum.
3. Only then look at `reduce_over_group` itself — replace it with an explicit 16-lane
   shuffle or shared-register reduction and see whether the answer changes.

## 4. What removing it cost

Nothing measurable. AGENTS.md records the arm as ported and **performance
unmeasured**; it was never promoted, and no correctness claim depended on it. The
*design* survives — llama.cpp's `mmvq.cpp` runs exactly this shape in production,
and `INTFER_DECODE1_SPLITK`'s bisection still stands.

## 5. Why the tree is green again, and what that does and does not mean

`test_tp2_27b.sh` is in `check_device_precond.sh`'s **healthy** column, so one
deterministically-failing arm made the gate red, then `DEVICE PRECOND: FAIL`, then
`run_all_checks.sh` red — in that order. Removing the arm removes that cascade.

**This is not a relaxation.** An assertion that cannot pass is not a check; it is a
gate that has learned to expect the wrong answer. What would be a relaxation is
loosening the stream-identity predicate, dropping the arm's reachability assertion, or
skipping the arm and reporting it as a pass — none of which was done.

The lesson worth keeping is uncomfortable and is now recorded in AGENTS.md: **the
SUBGROUP self-check cases were green the whole time the arm was wrong.** A predicate
tests the *shape* of a claim, not its truth. Only the token oracle tests truth, and
that is what caught this. Two predicates exercised to perfection, asserting
something false, is a worse failure than no predicate.

```
RESULT: (intentionally blank -- this report records a removal, not a benchmark)
```