# E4 — numerics contract for the cooperative 2D-tiled DPAS block

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

RESULT (E4 built + gated + measured at 27B TP=2; blank until then):
