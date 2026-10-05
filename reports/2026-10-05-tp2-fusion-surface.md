# The TP=2 fusion surface — a premise correction, and the one candidate that is real

**Date:** 2026-10-05
**Status:** the fusion seam was derived from source; `INTFER_MLP_GU` was
**implemented, measured DIVERGENT, and REVERTED** (§3b). Nothing from the attempt
is in the tree.
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
| flag | `INTFER_MLP_GU` (gate+up), exactly `=1`; default **OFF** |
| kill switch | `INTFER_NO_MLP_GU=1` forces it off |
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
3. **Wrong results, no crash.** With the sharding fixed the kernel ran and
   announced itself (`[mlp-gate-up] ... KERNEL ACTIVE, 2 x 8704 work-items
   (N=8704 K=5120 bits=4)`), and the stream still diverged. The per-output
   arithmetic was transcribed from `gemm_usm_decode1_vec` and reviewed against it
   line for line, with no difference found — so the defect is in the engine seam
   or the CLI wave wiring, not in the kernel body, and **it was not found within
   this session's budget.**

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
RESULT (operator): scripts/run_ten_gates.sh
        then scripts/check_engine_cert.sh
        (the single operator-deferred step; derived by
         scripts/check_pending_operator.sh)
```

Nothing in §3b needs a benchmark to believe: the arm's own stream comparison is
the evidence, and it is a correctness measurement, not a timing one.

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
