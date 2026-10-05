# The TP=2 fusion surface — a premise correction, and the one candidate that is real

**Date:** 2026-10-05
**Status:** the fusion seam was derived from source. `INTFER_MLP_GU` was
implemented, measured DIVERGENT, and **fully reverted** (§3b) — it is **not** a
pending experiment and there is nothing here to benchmark. The one optimization
this session DID ship is `INTFER_FUSE_ACT` (§3c), which is in the tree behind a
default-OFF flag with a live gate arm.
**Answer to "do the rest of PLAN.md":** the remaining fusion work is not blocked on
engineering — it is blocked on a seam that cannot execute at the target
configuration, and the one candidate that IS valid at TP=2 was built and did not
meet its bit-exact claim.

---

## 1. The finding

The task was: implement E5 by routing it through `run_linear_tp_devact`, so a D2H
plus a host wait disappears per fused group.

**That route cannot work at 27B `--tp 2`.** Both fusion entry points refuse
anything but TP=1, on their first line:

| seam | file:line | guard |
|---|---|---|
| `Engine::run_mlp_fused` | `src/engine/engine.cpp:1352` | `if (devices_.size() != 1) return false;` |
| `Engine::run_linear_tp_devact` | `src/engine/engine.cpp:1484` | `if (devices_.size() != 1) return false;` |

The header says so explicitly (`include/intfer/engine.hpp`): *"Returns false
(caller keeps host path) on TP>1 (**device acts are not cross-visible**)"*.

So an E5 routed through that seam would be **dead code at the only configuration
the paramount directive cares about** — the 27B `--tp 2` target. That is the exact
class this repo spent the last session eliminating: a default-OFF flag that no
gate can reach, which `scripts/check_flag_gate_coverage.sh` now enumerates and
would list as a permanent, unclosable certification gap.

Shipping it would have been worse than not shipping it: it would look like
progress, it would consume a mutation matrix and a gate arm, and the arm could
never go green for a reason anyone could act on.

## 2. Why TP=2 has no local MLP fusion — derived, not assumed

TP is **column-parallel over N** (`shard_weight` slices the output dim; each
device holds rows `[start_d, end_d)` of N). From the shapes the CLI assigns:

| tensor | N | K |
|---|---|---|
| `mlp.gate_proj` | `inter` | `H` |
| `mlp.up_proj` | `inter` | `H` |
| `mlp.down_proj` | `H` | `inter` |

Therefore per device `d`:

* `gate`/`up` produce **only device d's slice of `inter`** — local, fine.
* `down` has **K = the full `inter`**. Device d computed only *its* slice, so
  `down` on device d **requires the other device's slice**.

The full intermediate must therefore pass through the host. There is no TP=2-local
MLP fusion available, and closing that would need cross-device movement — which
this repo has none of (`grep -riE 'p2p|ipc|peer'` over `src/ include/` returns
nothing; recorded in AGENTS.md's architecture facts).

This also explains *why* `run_mlp_fused` is TP=1-only by construction and not by
omission: its `shards_[..].ranges[0].first != 0` checks are the guard expressing
exactly this.

## 3. The one TP=2-valid fusion candidate

**Fuse `gate` and `up` into a single launch per device.**

* **Why it is valid at TP=2:** both are column-parallel over the same `inter`
  split, both take the same replicated `H` activation, and both outputs are
  device-local slices. Nothing crosses a device boundary.
* **Why it is bit-exact:** each output element's K-accumulation is *unchanged* —
  same activation row, same weight row, same ascending-K order. Only the launch
  boundary and the work-item indexing move. No FP32 reassociation, so **no
  tolerance contract is required** — this is the "exact" branch of the fusion
  directive, not the "changes accumulation order" branch.
* **What it removes:** one kernel launch, one host wait, and (if the engine can
  share the staging buffer) one activation upload per MLP. Per the fusion
  directive's own measurements, *more device work in fewer host phases wins* —
  and this is strictly fewer phases for identical arithmetic.
* **Per the reverted-`(b)` lesson:** the per-output body stays **inline at its
  call site**. Do not extract a shared device function. That extraction is what
  cost 231 114 ms; a normalized diff said the arithmetic was identical and it was,
  and it did not matter.

### Numerics contract

Already discharged by construction, and this is the reason it is the right first
candidate: **bit-exact, therefore zero tolerance.** The only thing to assert is
stream identity against the fixed oracle — which is an existing kind of gate arm,
not a new numerics exercise.

## 3a. The candidate, specified (nothing here is implemented)

So the next session has everything except the kernel:

| | |
|---|---|
| flag | `INTFER_MLP_GU` (gate+up), exactly `=1`; default **OFF** — **SPECIFIED, NOT IMPLEMENTED** (§3b records the attempt and its outcome) |
| kill switch | `INTFER_NO_MLP_GU=1` forces it off |
| corrected slice math | the fused launch must be handed **`nl`, the device's own column slice**, never the tensor's `N`; and its output workspace must be reserved for **`2 * nl`**, not `nl`. Both were SIGSEGVs (§3b items 1 and 2) and both are now carried here as design constraints rather than as things to rediscover |
| arm lives in | `scripts/test_tp2_27b.sh` — TP=2, so it is reachable (0.8B has zero INT2 and cannot reach a `bits==2` branch) |
| assertion | stream identical to the same fixed oracle the base run uses — **not** a self-comparison |
| numerics | **bit-exact, zero tolerance.** Each output element's K accumulation is untouched; only the launch boundary and work-item indexing move |
| inline? | **yes, arithmetic stays inline at its call site.** Do not extract a shared device function — that is the 231 114 ms lesson |
| mutation matrix | write it BEFORE the kernel, and assert the HARVEST (per NS-3/NS-4), not the list |
| expected direction | `kernel` down or flat; `submit`/`waitblock` down by one launch+wait per MLP; `nk` changes by the *added* work-items, so `nk` is a DIAGNOSTIC here, not an invariant |
| risk | adds a launch-sized work unit per MLP; the L4 M-row case shows an added launch can lose even while removing waits. If it loses, that is a result — record it, do not retry per-work-item tiling |

## 3b. ATTEMPTED, MEASURED DIVERGENT, REVERTED

`INTFER_MLP_GU` was implemented (fused `quant_gemm_forward_usm_pair`, engine seam
`run_linear_tp_pair`, CLI wave wiring, gate arm in `test_tp2_27b.sh`, four
mutation states) and then **reverted**, because its own gate arm falsified the
bit-exact claim that was the entire argument for it.

Measured on 27B `--tp 2`, the arm's stream diverged from the base run at the first
token:

```
base            13 198 760
INTFER_MLP_GU   14354 9616 43047
```

**Three real bugs, in order, and the tooling caught the first two — not me:**

1. **SIGSEGV (rc=139).** The output workspace was sized for ONE N slice while
   operand B was written at `ws_[0].out + M*nl`, past the end of the allocation.
   Found by `check_device_precond.sh`, which runs the gate as part of its
   healthy-box check.
2. **SIGSEGV again, after fixing that.** The kernel was handed the tensor's full
   `N` instead of the device's own column slice `nl`, so at TP=2 each device read
   its neighbour's weights. **Under TP=1 `nl == N`, so a one-GPU smoke test would
   never have caught it** — which is the strongest argument yet for TP=2 coverage
   of any kernel that touches sharding.
3. **Wrong results, no crash — ROOT CAUSE NOT LOCATED. THIS IS AN OPEN
   QUESTION, NOT A CLOSED ITEM.** With the sharding fixed the kernel ran and
   announced itself (`[mlp-gate-up] ... KERNEL ACTIVE, 2 x 8704 work-items
   (N=8704 K=5120 bits=4)`), and the stream still diverged.

   **What is known:** the per-output arithmetic was transcribed from
   `gemm_usm_decode1_vec` and compared against it statement by statement with no
   difference found. The kernel reported the same shape the sequential path
   reports. The two crashes were real bugs and are fixed.

   **What is NOT known, and must not be assumed:** which of the remaining three
   components is wrong — the kernel's operand selection, the engine seam's
   read-back, or the CLI's gate/up name-to-tensor mapping. Reverting on
   "reviewed line for line" is the structural-argument failure this repo audits;
   it happened here and the evidence is why the kernel is out.

   **The next attempt must BISECT, not re-derive.** The three components are
   separable and the bisection is one line: run the fused pair AND then run the
   same two items through `run_linear_wave`, and compare the two outputs host-side
   in the same process. Agreement localises nothing; disagreement localises
   immediately to kernel-vs-wiring. Do that before touching the kernel again.

   **Untested hypotheses, in the order they are worth trying** (none is a
   finding):
   * the CLI maps gate/up by *substring* on the tensor name, and a model whose
     names differ would swap the two operands — which would diverge at the FIRST
     token, exactly as observed;
   * the engine seam reads both operands out of one workspace and a later read
     could see bytes the first `half_to_float_batch` already consumed;
   * the kernel's `per` padding (`nBlocks * nWG`) makes operand B's row index
     start at `per`, not at `nl`, so a slice whose length is not a multiple of
     the work-group size would index operand B's scales with the wrong stride.

## 3c. `INTFER_FUSE_ACT` — the optimization this session DID ship

Gate+up turned out to be the wrong lever; a smaller one in the same code path is
right, and it is in the tree.

| | |
|---|---|
| flag | `INTFER_FUSE_ACT`, exactly `=1`; **default OFF**; kill switch `INTFER_NO_FUSE_ACT` |
| what it does | when consecutive linears consume the SAME activation vector (an MLP's gate and up), it skips the second FP32→FP16 convert and the per-device host upload |
| why it is bit-exact | the destination already holds exactly the bytes the skipped call would write — **no arithmetic is touched at all**, so this is the "exact" branch of the fusion directive, not a tolerance question |
| what it costs | nothing: no launch added or removed, no kernel changed |
| gate arm | `scripts/test_gpu_optin.sh` — **reachable there**, because the branch is not guarded on `bits == 2`, so the 0.8B model enters it. The arm asserts stream identity against the same base run AND requires the kernel's one-time `[fuse-act] ... ACTIVE` announcement, so an inert flag cannot pass by matching trivially |
| guard | `scripts/check_act_cache_invalidation.sh` re-derives, from source, that every free of the cached buffer clears the cache — because the cache is bit-exact only while its bytes are the bytes the GEMM reads |
| matrix | `mutate_act_cache_invalidation.sh` 4/4 |

**The audit the guard exists for.** `ws_[d].act` is written by exactly one place,
but freed by two that matter — `Engine::ensure_workspace` and `Engine::~Engine`
— and a cache that outlives its buffer is a use-after-free that no stream-identity
test finds until it is intermittent. The harvest is **function-scoped** because
the local alias `w` names `ws_` in `ensure_workspace` and `ws48_` in
`ensure_workspace48`: a line-local grep cannot tell them apart, and got it wrong
in both directions here (counting only `ws_[d].act` MISSED an invalidation;
counting `w.act` too failed a healthy tree over a buffer the cache never touches).

## 3d. Quarantine of the two crash patterns — no live occurrence

Both patterns from §3b were audited against the existing tree:

* **full `N` handed to a kernel that should get `nl`.** No live occurrence.
  `Engine::run_linear_tp` passes `N_local` (`engine.cpp:1272`), the DPAS
  dispatch passes `N_local` (`engine.cpp:1706`), and
  `Engine::run_linear_tp_devact` (`engine.cpp:1520`) refuses unless
  `Nl == t.N` — it is TP-1-only by construction, so `nl == N` there. Verified:
  `grep -n 'quant_gemm_forward_usm(' src/engine/engine.cpp` and read each site.
* **an output workspace sized for one N-slice while two operands share it.** No
  live occurrence. `Engine::run_mlp_fused` gives gate and up **separate** wave
  slots (`wv_[0].out`, `wv_[1].out`) and reserves `ws_[0].out` for the down
  projection alone (`engine.cpp:1415/1418/1439`); the wave path gives every item
  its own slot.

The first pattern is the more interesting result: it is **invisible at TP=1**,
where `nl == N`, so a one-GPU smoke test can never find it. That is the concrete
argument for keeping TP=2 coverage over any kernel that touches sharding.

## 4. Why the kernel is NOT in the tree

Because its gate arm rejects it. The claim was *bit-exact*, which is the one
property that makes this fusion cheap enough to want; a kernel that needs a
tolerance instead is a different, worse experiment that would need its own
contract before any code. Shipping it "flagged and pending" would leave the repo
holding an uncertified kernel whose one justification is falsified.

What is kept: nothing from the attempt. The seam analysis, the TP=1-only
constraint, and the bit-exact argument above stand, and the diagnosis is recorded
here so the next attempt starts from two crashes already fixed and one
divergence already measured.

**The next attempt's first job is to bisect bug 3**, not to re-derive the design:
the engine seam and the CLI wiring are separable, and a single fused-pair call with
the wave path also executed would say immediately which of the two is wrong.


This file contains **no `INTFER_*` flag**, because shipping one would put the
repo back in the state the last session was built to leave:

* an unmeasurable optimization behind a default-OFF flag;
* no arm can reach it from an agent session (27B gate is operator-only by standing
  rule), so it would be an uncertified flag from birth;
* and `src/gemm/quant_gemm.cpp` has **already produced one 180x decode regression**
  (`quant_gemm_fused_decode1`, 1284 ms → 231 114 ms) from kernel surgery in
  exactly this file. A second unmeasured kernel edit into that file, by a session
  that cannot benchmark it, is the same bet at worse odds.

The next kernel change should land with the gate arm **already written**, so the
artifact that would catch a regression exists before the change rather than after.

## 5. What is left for the operator — one action

```
RESULT (operator):
  (1) INTFER_FUSE_ACT counters -- §3c. Run scripts/test_gpu_optin.sh first (the
      correctness arm is already wired and needs no benchmark); then the paired
      3-round A/B for the copy/usm_write buckets, which is the only unmeasured
      claim here. EXPECTED: `copy` and `usm_write` DOWN, `kernel` flat, `nk`
      unchanged. Fill from a run you drive; this session ran no benchmark.
  (2) scripts/run_ten_gates.sh
      then scripts/check_engine_cert.sh
      -- the single operator-deferred step, derived by
      scripts/check_pending_operator.sh.
```

Neither §3b nor §3c quotes a tok/s. §3b's evidence is a stream comparison, which
is a correctness measurement, and §3c's counters are unmeasured here by rule.

Nothing here needs a benchmark. Nothing here quotes a tok/s. The `RESULT:` line
above is a command, not a measurement — there is deliberately no number to fill.

## 6. Re-derive the claim yourself

```sh
grep -n 'devices_.size() != 1' src/engine/engine.cpp
grep -n 'TP>1' include/intfer/engine.hpp
grep -rn 'run_mlp_fused\|run_linear_tp_devact' src/engine/engine.cpp | head
```

If a future change lifts the TP>1 guard, the seam becomes usable and this whole
report is superseded — check the lines, do not trust the prose.
