# Batch 2: `gemm_usm_decode1_vec` bisect round 2 — three more arms, and one retraction

> ## ==> READ THIS FIRST: THIS BATCH RETRACTS A RESULT FROM THE PREVIOUS ONE. <==
>
> The last batch reported `INTFER_DECODE1_VERBMP is BITWISE IDENTICAL to base on
> 227 835 element values`. **That was vacuous and is withdrawn.**
> `scripts/test_rowpair_equiv.sh` now **requires each order's kernel to have
> announced itself** before it will diff anything, and the guard fired immediately:
> `FAIL: order vm completed but the kernel never announced [decode1-verbmp]`.
> A KILLED arm emits the base kernel's output, so a base-vs-base diff reads as a
> clean pass. Two arms were silently running the base path and reporting a perfect
> match. **Only `INTFER_DECODE1_ROWPAIR` has a real measurement.**

**Date:** 2026-10-06 · **No benchmark or tok/s anywhere** (standing rule 1).

---

## 1. The three new arms — `INTFER_DECODE1_VERBDEAD` / `VERBSEL` / `VERBVEC`

All default **OFF**, kill switch `INTFER_NO_DECODE1_*=1`, guard `bits == 2 &&
use_int2_tab && num_groups > 0`, each a **separate `parallel_for` with the body
INLINE at the call site** (a shared device function is fatal here — the reverted
`quant_gemm_fused_decode1` cost 231 114 ms), each with a one-shot announcement line
and a `scripts/test_tp2_27b.sh` arm at TP=2 (the only gate whose 27B model carries
INT2 — the 0.8B model has **zero**, so a `test_gpu_optin.sh` arm would be green
because the path never runs).

| arm | the ONE change | bit-exact? |
|---|---|---|
| `VERBDEAD` | `bytes_per_chunk` / `code_mask` hoisted out of the per-group loop, where they are dead on this path | **yes — pure code motion** |
| `VERBSEL` | `wtab[code]` (4-float array, runtime index) becomes a select chain over the same four values in named registers | **yes — same expressions** |
| `VERBVEC` | activations read through a raw `const std::uint16_t*` instead of a `multi_ptr` | **yes — addressing only** |

### Two candidates I did NOT write, and why

* **"Hoist the group scale out of the per-group loop."** *Impossible as stated.* The
  scale is **per group** — `from_half_bits(sptr[g])` with `num_groups = K/Gw = 320`
  at MLP_GATE. Hoisting it would reuse one group's scale for all 320 and **change
  the arithmetic**. What *can* be hoisted is the code around it, which is
  `VERBDEAD`.
* **"Avoid the per-group `multi_ptr` re-derivation."** The base kernel already
  constructs `act_mp` and `wmp` **once, outside** the group loop. There is nothing
  per-group to avoid.

### `VERBSEL` re-tests a hypothesis that was falsified in the WRONG BODY

`examples/roofline_probe` compared `table` against `select` and got 1.01× — but
**both were in the probe's own tidy clone**, where the compiler had already
normalised the array. That does not falsify the array being free **in the shipped
body**, which is 5.6–6.4× slower than that clone and may not normalise the same
way. This arm therefore measures the same transformation where it might matter.

## 2. Pinned by enumeration, and the pin is PROVEN to fire

`check_decode1_copies.sh` classifies **every** `verb*` body `covered-by-subset` and
asserts its shift-into-mask set **EQUALS** the general body's — derived from the
enumeration, not from a list, so an arm added today is pinned today.

Proven by hand: planting the real `INTFER_DECODE1_SPLITK` bug
(`(bv >> (j * 2)) & 3`) into `verbsel` gives

```
FAIL  gemm_usm_decode1_verbsel's shift-into-mask set differs from the general body's
```

`rc=1`, naming the arm. The extractor is deliberately **loose** (it matches the
wrong form too), because the strict version silently let that bug through once.

## 3. The vacuity fix — the substantive change in this batch

`examples/rowpair_equiv` now:
* dispatches from a **single derived order table** (`base rp vh vm vd vs vv`), so
  adding an arm cannot leave the dispatch, the env plumbing or the payload tag out
  of sync — the exact "added to one place, forgotten in the other" failure;
* uses `unsetenv(KILL)` for the selected order, **not** `setenv(KILL, "")` — an
  empty string is not absence, and that was bug 1 above;
* **requires the announcement**, which is the only evidence the branch was taken.

## 4. Status, honestly

| arm | correctness | performance |
|---|---|---|
| `INTFER_DECODE1_ROWPAIR` | **MEASURED** — 227 835 values bitwise identical | unmeasured |
| `INTFER_DECODE1_VERBHALF` | not measured (does not fire) | unmeasured |
| `INTFER_DECODE1_VERBMP` | **RETRACTED** — earlier result was vacuous | unmeasured |
| `INTFER_DECODE1_VERBDEAD` | not measured (does not fire) | unmeasured |
| `INTFER_DECODE1_VERBSEL` | not measured (does not fire) | unmeasured |
| `INTFER_DECODE1_VERBVEC` | not measured (does not fire) | unmeasured |

**OPEN BUG, cause not located:** the `verb*` arms do not fire under the
differential. The guard's four conditions were instrumented and are all TRUE
(`use_verbhalf=1 bits=2 tab=1 ng=320`) and the arm body immediately follows in the
same block, yet no announcement reaches stderr. Recorded unresolved rather than
guessed at. `scripts/test_rowpair_equiv.sh` is **RED** because it refuses to
certify arms it cannot see running — correct behaviour for this gate.

## 5. Operator commands

```
# correctness, model-free, no 27B load -- currently RED by design
bash scripts/test_rowpair_equiv.sh

# the engine gate, once the box is healthy (operator-only; standing rule 1)
bash scripts/run_ten_gates.sh
bash scripts/test_tp2_27b.sh

# performance, per arm, load1 < 2. Counters only; never a tok/s claim.
INTFER_DECODE1_VERBDEAD=1 bash scripts/run_ten_gates.sh
INTFER_DECODE1_VERBSEL=1  bash scripts/run_ten_gates.sh
INTFER_DECODE1_VERBVEC=1  bash scripts/run_ten_gates.sh
```

Expected direction is **UNKNOWN for every arm** — none has been measured, and the
probe that was meant to measure them (`test_roofline_bisect.sh`) reports
`rc=2 UNPROVEN` because its binary does not complete a run.

```
RESULT: VERBDEAD  kernel=__ ms/token (baseline __)  nk/nw/nr __/__/__  [win|neutral|loss]
RESULT: VERBSEL   kernel=__ ms/token (baseline __)  nk/nw/nr __/__/__  [win|neutral|loss]
RESULT: VERBVEC   kernel=__ ms/token (baseline __)  nk/nw/nr __/__/__  [win|neutral|loss]
```
