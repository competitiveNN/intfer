# INTFER_WAVE_ALIAS — counter prediction and gate report (2026-10-13)

**Status: code-verified, build-clean, gate-armed, model-loaded run DEFERRED.**
Operator-window (22:00+) required for the gate run. No numbers below are claimed as measured.

## What this is

`INTFER_WAVE_ALIAS` (`examples/intfer_cli/main.cpp:4898-4935`) aliases an **un-rotated**
LA wave input to the caller's own normed buffer instead of `memcpy`-ing it into a
per-item scratch buffer. PRESENCE-enabling (`getenv(...) != nullptr`): unset disables,
`=0`/`=1`/`=anything` all enable it. Default OFF.

6 call sites: `step()` decoder wave-4, `prefill_layer_outer` Phase C M-row wave-4,
`step_batch` M-row mirror wave-4, and the three attention wave-3s (q/k/v).

## Bit-exactness argument (the certification)

The base and this arm are bit-identical **by construction**: an un-rotated tensor's
staging buffer held a byte-for-byte copy of `normed`, so the GEMM reads identical
bytes either way and every output's K accumulation is untouched. `rot_apply` fully
overwrites its target, so a rotated tensor still gets its own buffer. This is a
HOST-SIDE saving — same kernels, same buffers, same output assembly — so the numerics
contract is trivially held and a stream comparison alone is vacuous. What defeats the
vacuous green is the per-site decision line `[wave-alias] ... aliased=N` (main.cpp:4933),
which greps the *decision*, not a constant.

## Counter prediction (load-insensitive buckets)

| bucket | base | WAVE_ALIAS | note |
|---|---|---|---|
| `nk` / `nw` / `nr` | identical | identical | bit-exact by construction |
| `kernel_count` | identical | identical | same kernels |
| `submit_count` | identical | identical | WAVE_LAUNCH_FUSE is the launch reducer, this is the host-staging reducer |
| `wait_count` | identical | identical | |
| `readback_count` | identical | identical | |
| `copy_count` | > | **↓** | one H2D copy per un-rotated wave item removed, 6 sites × M rows × H |
| `usm_write_count` | > | **↓** | |
| `total_copy_us` | > | **↓** | |
| `total_usm_write_us` | > | **↓** | |

Per the engine's shared-input dedup (pointer identity + equal M*K), the FP32→FP16
convert runs **once** instead of once per item. `nk` is the only load-insensitive
kernel bucket and it is unchanged; the win is confined to the host `copy` / `usm_write`
accumulators, which are host wall-clock sums reported as `median(spread)`.

## The gate (already written, TP-2 covered)

`scripts/gates/test_gpu_optin.sh:785-845` (also re-runs TP=1 and the fully-rotated negative):

- (a) TP=2: stream identity vs base **AND** `[wave-alias]` line present.
- (b) TP=1: same, so the claim is not TP-polar.
- (c) TP=1 on a fully-rotated model: `[wave-alias]` still present with `aliased=0`
  — an `ACTIVE`-only line would be silent there and the arm would be unfailable
  rather than passing (vacuity in the other direction).
- (d) TP=2 with the flag unset: `[wave-alias]` absent.
- (e) empty model run: absent.

## Why it is UNMEASURED

Loading the 27B weights is forbidden in agent sessions (standing rule 2). The 22:00+
gate run derives `copy_count` / `usm_write_count` / `total_copy_us` /
`total_usm_write_us` under `INTFER_PROFILE_GPU` for base and `INTFER_WAVE_ALIAS=1`
and compares with >= 2 paired rounds, clearing `max(spread of the two arms)`.

## Decide on

`copy_count` / `usm_write_count` (load-insensitive integers). If those fall and
`nk` is unchanged, promote; if the copy counters do not fall, decline — the mechanic
is real, but the round trip reduction must actually materialize.
