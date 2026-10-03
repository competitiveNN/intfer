# E4 — numerics contract for the cooperative 2D-tiled DPAS block

> **SUPERSEDED 2026-10-03 — E4 IS BUILT, GATED AND MEASURED. KEEP.**
> Everything below the "IMPLEMENTED AND MEASURED" heading near the end is the
> original 2026-09-30 **decision record**, kept verbatim as the record it is.
> Its claims that no E4 kernel and no E4 flag exist were true when written and
> are now FALSE. The flag is `INTFER_DPAS48_TILED` (default OFF); prefill device
> kernel is **9.95 % cheaper** than FOLD, 3/3 rounds same sign, zero added
> launches. **Read the later section for current state.**

Date: 2026-09-30. Author: maki (agent). **Decision record, not code.** No E4
kernel exists and no E4 flag exists. This file exists so the next session does
NOT start E4 by guessing a tolerance.

Related: `reports/archive/2026-09-30-dpas48-fold.md` (E3, the bit-exact fold hoist),
`reports/2026-09-30-pending-bench-backlog.md` (E4 entry + queue state).

## Why E4 needs a contract at all, and why E3 did not

E3 (`INTFER_DPAS48_FOLD`) hoists the per-(row,op) activation fold out of the
16-N tile loop. It is **bit-exact** because the hoisted values are exact int32
sums: integer addition is exact and associative, so moving the sum cannot
change a bit. That is why E3's gate is `DPAS48+FOLD == DPAS48`, a hard string
equality.

E4 is a different animal. A cooperative 2D block accumulating per-block
partials **reorders the FP32 accumulation**
`acc_tile[nl] += as * ws * alpha * (2*c[nl] - 15*S) * Cinv`
over `nOp = K/32` terms. Reordering floating-point adds changes the result. E4
therefore CANNOT be gated by `==`, and a hard stream-identity gate would be a
**lie** — it would assert that no output ever lands on a rounding boundary,
which is the same tie-luck defect recorded in AGENTS.md AUDIT #3b (the GA64
gate once passed only because two implementations happened to agree on every
tie).

## The decision

**Gate kind:** a new one — a *tolerance/value* gate, not an identity gate.
Identity is expected but is **reported, not asserted**.

**Metric: backward error, not relative error of the result.**
The kernel's term is `2*c - 15*S` — a **zero-point correction**. The terms
therefore cancel, and the result can be orders of magnitude smaller than the
largest term. Measured on this harness at K=28672: result-relative spread
6.80e-3 while backward-relative spread is 1.38e-5. The result-relative number
is dominated by cancellation, not rounding, and it silently degrades as the
true value approaches zero. Gating it would set a bound that tightens for
reasons unrelated to the change. So:

```
backward_err = max |seq - blocked| / max |t_o|
```

**Tolerances (measured on this harness, 4096 reps/shape, not guessed):**

| quantity | measured (worst, K=28672) | allowed | margin |
|---|---|---|---|
| `backward_err` | 1.38e-5 | **4.0e-5** | ~3x (covers K up to ~57k, past the widest 27B tensor) |
| stored-FP16 movement | 8 ULP | **16 ULP** | 2x |
| fraction of outputs changed | 0.32% (13/4096) | **1%** | ~3x |

Full measured table, printed by the gate on every run:

| K | nOp | backward_err | max_rel (info) | half_diff | max_ulp |
|---|---|---|---|---|---|
| 1024 | 32 | 1.15e-6 | 5.86e-4 | 3/4096 | 1 |
| 4096 | 128 | 2.78e-6 | 2.68e-4 | 2/4096 | 1 |
| 14336 | 448 | 9.06e-6 | 2.67e-3 | 9/4096 | 3 |
| 28672 | 896 | 1.38e-5 | 6.80e-3 | 13/4096 | 8 |

`backward_err` grows roughly linearly in `nOp = K/32`, which is why the bound is
set on the largest shape rather than the median.

## Where this is enforced today

`examples/dpas_fold_probe` grew an **E4 accumulation-order sensitivity** section
that measures exactly the above (sequential vs 4-way blocked partials, the shape
a cooperative block would take) and **asserts all three tolerances**. It is gated
by `scripts/test_dpas_fold.sh`, so the tolerances are themselves regression-
protected rather than being a comment nobody runs.

The four existing NEGATIVE decline cases are unchanged, and the fold value
checks (3907 pairs) still gate alongside — this is an added section, not a
replacement.

**The contract is mutation-proven:** tightening `kTolHalfUlp` from 16 to 2
(below the measured 8) makes the gate exit 1 with
`E4 contract: VIOLATED -- the bound is wrong for this shape`. Reverted, green.

## What invalidates this contract

State these now so nobody discovers them later:

1. **A real kernel may need a different bound.** The term generator here is
   SYNTHETIC and deliberately cancellation-heavy, so these numbers are an
   **upper bound** on the real spread. When E4 exists, re-derive from the real
   kernel. Do **not** inherit 4.0e-5 by copy.
2. **A different block geometry has a different error profile.** A 2-way or
   8-way partial split, or a tree reduction, will not match the 4-way numbers
   here. Re-measure per geometry; the harness takes the split count as the
   thing under test.
3. **TP changes the accumulation.** Under TP=2 each rank accumulates over its
   own K range, so `nOp` halves and the error drops. Tolerance must be derived
   per TP.
4. **Any move to a different accumulator type** (FP32 -> TF32, or a fused
   tensor-core accumulator with its own internal order) invalidates all of it.
5. **Stream identity must still be CHECKED and REPORTED.** With ~0.3% of
   outputs moving up to 8 ULP, a greedy stream *can* diverge at a near-tie
   logit. That must be reported as a finding, not silenced by loosening the
   value bound, and not converted into a hard identity assertion.

## Residual risk, stated plainly

Even with the value tolerances met, E4's payoff is unproven: `INTFER_DPAS_ROW`
already showed that re-partitioning DPAS work without real cooperation measured
**~460x slower**. A cooperative 2D block is a different design, but the
precedent is a warning, not a promise. E4 should be built behind
`INTFER_DPAS48_TILED=1` (default OFF) and measured like any other experiment
before any promotion.

---

# IMPLEMENTED AND MEASURED 2026-10-03 — **KEEP** (default-OFF, prefill only)

`INTFER_DPAS48_TILED=1`, default OFF. Commit `bc0b008`.

## What was built

`quant_gemm_dpas48` became a **template on `R` = rows per work-item**, not a
second copy of the kernel. `R==1` is the shipped geometry and is bit-for-bit
unchanged; `R==2` gives one WI a 2D block of **2 rows × ONE 16-N tile**, so the
B op-tile is fetched **once per op and reused across both rows**.

That is the load-replication half this contract was written for. E3 removed the
*arithmetic* replication; E4 removes the *load*.

**One template, deliberately.** The repo's most-repeated audit finding is a rule
fixed at one site and re-derived at another. A duplicated 170-line DPAS branch
tree (`u4`/`u8`/`u2` × the G16 split) would have been exactly that class of
defect, waiting to be found by the next person to change one copy.

## Why the contract's reordering hazard does not apply — a design property

The contract warned that a cooperative block *reorders the FP32 accumulation* and
so cannot be gated by `==`. **This geometry does not reorder anything.** Each
`(row, tile)` keeps **one** accumulator and still sums over ops `o = 0..nOp-1` in
the same order with the same operands; sharing `vb` across rows moves no data
into a different sum. There are no per-block partial sums to combine, so there is
no reordering.

So the contract's *own* instruction stands and was followed: **identity is
REPORTED, never used as the gate**, and the tolerance gate stays where it was.
It also said *"when E4 exists, re-derive from the real kernel — do not inherit
4.0e-5 by copy"*, so:

| what | measured |
|---|---|
| real-kernel readback, R=1 vs R=2 | **256/256 outputs BIT-IDENTICAL** |
| `backward_err` (contract's synthetic generator) | unchanged, 1.38e-5 at K=28672 |
| M<2 **decline** asserted | sentinel prerequisite returned untouched, 0 slots written |

The tolerance numbers above are therefore **not inherited** — the real kernel is
*exactly* identical, which is a stronger statement than "inside the bound".

## Gating, and its non-vacuity

- `examples/dpas_fold_probe` → **E4 real-kernel readback** (new section).
- `scripts/test_gpu_optin.sh` → **4 new arms at TP=1 AND TP=2**: narrow,
  **wide-prefill** (~97 tokens, the only place the block can fire), `FOLD+TILED`
  composition, and `TILED`-alone inertness. All stream-identical to DPAS48.
- **Non-vacuity measured, not asserted.** Mutating the shared store so every row
  of a group writes the group's *first* row is caught at **both** levels:
  probe `rc=1` (`differs at 1: 0xf112 vs 0xecdd`) and gate `rc=1` on the narrow
  **and** wide arms.

> **The narrow arm is not the inertness check I expected.** The mutation caught
> `TP=1 [DPAS48+TILED] n=24` too, because a 5-token prompt still prefills at
> **M=5 > 1**, so E4 fires there. Only *decode* (M=1) is inert by construction.
> Correct as written — but it means the flag is not "prefill-only" in the gate, it
> is prefill-only *and* silent at decode.

## THE MEASUREMENT — 27B `--tp 2`, `devices=2`, `@prefill`, 3 interleaved rounds

`scripts/bench_reports_batch.sh E4 3`. Stream identity verified first:
`DPAS48 == FOLD == FOLDTILED` (`561 3712 314 19820 10903 321 5484 6618`).

**E4 is decode-inert by construction**, so this batch deliberately measures the
**`@prefill`** bucket with a ~361-token single chunk. An E4 *decode* A/B could
only ever read "no change", and reporting that as a result would be the same
error as benching a declined flag.

**Still no wall-time**: all nine arms `SKIPPED-CONTENDED`
(`LOAD1=7.98 7.74 5.74 4.65 5.19 6.08 5.40 6.33 5.32`). What follows is
**device-side `kernel`, which is load-insensitive.**

| arm | `nk` | kernel per round (ms) | median | spread |
|---|---|---|---|---|
| DPAS48 | 4416 | 19733.7 / 19920.7 / 19796.4 | 19796.4 | 187.0 (0.9 %) |
| FOLD | 5468 | 18644.0 / 18615.7 / 17971.3 | 18615.7 | 672.7 (3.6 %) |
| **FOLDTILED** | **5468** | 16662.5 / 16763.8 / 16204.4 | 16662.5 | 559.4 (3.4 %) |

`nw`=2976 and `nr`=1056 are identical across all three arms, and **`nk` is
5468 for FOLD and FOLDTILED alike — E4 adds ZERO launches.**

| comparison | per-round delta (ms) | median | verdict |
|---|---|---|---|
| DPAS48 → FOLD | +1089.7 / +1305.0 / +1825.1 | **+1305.0 ms (+7.01 %)** | CLEARS 672.7 floor, 3/3 same sign |
| **FOLD → FOLDTILED** | **−1981.5 / −1851.9 / −1766.9** | **−1851.9 ms (−9.95 %)** | **CLEARS 672.7 floor, 3/3 same sign** |

> Read the spread caveat with the number, exactly as the summariser prints it:
> the floor is the noise **this run** showed, and the spread is not a constant of
> the arm (BASE measured 1875.5 ms in the E1 run and 64.5 ms in E3, same day).
> −9.95 % clears the observed floor by **~2.8×**, so this is not a marginal call.

## Verdict: **KEEP — promote-worthy, but stays default-OFF for two reasons**

**The evidence is good:** a **9.95 % cheaper device-kernel prefill**, 3/3 rounds
agreeing in sign, clearing the run's own spread by ~2.8×, at **zero added
launches**, with bit-exactness gated on device *and* engine. For a memory-bound
wide-N prefill, halving the B traffic per output row is exactly the mechanism
predicted, and `nk` unchanged is the signature of pure work redistribution
rather than extra work — which is what E3 could not offer.

Two things keep it default-OFF, neither of them a doubt about the direction:

1. **No quiet-box tok/s yet.** Every arm was contended. The kernel number is
   load-insensitive and solid; the *wall-time* consequence is not measured.
2. **It only affects prefill.** Decode is M=1 and E4 declines there, so it cannot
   move the 40 tok/s single-stream decode target at all — it is a prefill lever,
   and prefill is the 4000 tok/s target.

**Not a repeat of `INTFER_DPAS_ROW`** (~460× slower): that put the whole tile
loop inside one WI and collapsed occupancy. Here tile parallelism is preserved
exactly — one 16-N tile per WI — and only the row count per tile grows. The
precedent's warning was respected rather than repeated, and that is exactly what
the measurement shows.

## Re-run it

```bash
cd /var/home/fra/dev/intfer
. /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
INTFER_08B=/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer \
INTFER_TP=2 scripts/bench_reports_batch.sh E4 3
```
Preconditions and the per-arm quiet rule are the canonical ones in
`reports/archive/2026-09-30-pending-bench-backlog.md`. To fill a **tok/s**, run
it on a box where `load1 < 2` for the whole of every arm.

RESULT (E4 built + gated + measured at 27B TP=2):
RESULT: KEEP. Prefill device kernel 9.95 % cheaper (FOLD 18615.7 -> FOLDTILED
RESULT: 16662.5 ms median, per-round d = -1981.5 / -1851.9 / -1766.9, 3/3 same
RESULT: sign, CLEARS this run's 672.7 ms spread by ~2.8x), nk UNCHANGED at 5468,
RESULT: nw/nr unchanged, bit-exact on device (256/256) and engine (TP=1+TP=2).
RESULT: NO tok/s claimed -- all nine arms SKIPPED-CONTENDED (LOAD1 4.65-7.98).
RESULT: Stays default-OFF: prefill-only (decode M=1 declines), and no quiet-box
RESULT: wall-time yet. This retires the design-only status above.
