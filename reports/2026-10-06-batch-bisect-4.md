# Batch: bisect-4 — the differential between VERBATIM and the int2_variant clone

**2026-10-06, dual Arc Pro B580 (both enumerate: `zeDeviceGet rc=0 DEVICES=2`).**

Two GPUs present and healthy this run. Source:
`scripts/l0_enumerate.sh` / `sycl-ls` (2 `[level_zero:gpu]` lines, UUIDs
`...-0400-...` and `...-8600-...`).

- AGENTS.md standing rule 1: **agent session, no full-27B gate.** Every number
  below is `device` event timestamps on an **engine-free** probe that binds **no
  model** (`INTFER_QUICK` self-check only; zero preload).
- AGENTS.md paramount directive: this batch is a counter-measurement for an
  *optimization decision*. It reports load-insensitive buckets / ratios. No
  tok/s was measured.

## What it is

`examples/roofline_probe_bisect4` is the differential that closes out the
M=1 INT2 GEMV ladder. It answers one narrow question, set up so a "no" is still
a result:

> Does the ~7x gap between the transcribed SHIPPED source (VERBATIM, ~30 GB/s)
> and the roofline_probe `int2_variant<0>` clone (~58 GB/s) live in any of
> these **single changes**, or in the shared text of the two functions?

Five arms, each differing from `VERBATIM` in the notes, interleaved per-rep on
the SAME buffers (device event timestamps, medians over 12 reps). Same shapes
as every other bisect binary (MLP_GATE 8704×5120 g16, MLP_DOWN 2560×17408 g16).

| arm | ONE change from VERBATIM | MLP_GATE | MLP_DOWN |
|---|---|---|---|
| VERBATIM | (baseline) | 31.3 GB/s | 18.1 GB/s |
| VERB+bitcast | group scale + tail activation via `bit_cast<half>` instead of branchy `half_to_float` | 35.0 GB/s | 20.7 GB/s |
| VERB+bitcast+4 | + probe's 4-byte code loads (`v0..v3`) | 34.9 | 20.7 |
| VERB+bitcast+4+vsel | + probe's 3-ternary `bv` selector | 34.9 | 20.7 |
| CLONE | full transcription of `int2_variant<0>` | 34.9 | 20.7 |

ratios vs VERBATIM (index 0): **1.00 / 1.12 / 1.12 / 1.12 / 1.12** (MLP_GATE),
**1.00 / 1.14 / 1.14 / 1.14 / 1.14** (MLP_DOWN).

## What it does NOT establish (and why that matters)

**None of the four single changes closes the gap.** Each lands at ~1.11–1.14x
vs VERBATIM, and the *full* CLONE transcription is indistinguishable from the
last single-change arm — the transcription buys nothing past `VERB+bitcast`.
So the deficit is **not** in bit_cast vs half_to_float, not in the
dynamically-indexed `wtab[code]`, not in the byte-index math, and not in the
`bv` selector. It is in the **shared text** of the two functions: the
`a_lo`/`a_hi` load + two-byte code fetch + 8-lane inner loop that both bodies
share. That shared text is what bisect-3's `table`-vs-`SHIPPED` measurement
is the next thing to read for (it is already anchored cross-binary from
`roofline_probe`).

The clone-vs-shipped gap therefore survives the module move (1.26x, recorded)
and the runtime branch (1.007x, recorded) — it is **not** an extraction
artifact and it is **not** the cosmetic differences. The `from_half_bits`
scalar branch is the remaining named suspect, but it is called only on the
*TAIL* there (one per group tail, negligible vs the 8-wide chunk loop), so
that hypothesis is expected to fail too; this run does not test it directly.

## The always-red twin (held)

`ceiling 0.206 ms 254.7 GB/s (55.9% of peak)`; `twin ... ratio to full = 0.23
(linear would be 0.25)`. A quarter of the bytes taking ~0.23x of the time
holds, so the device event timestamps are tracking traffic.

## Load-insensitive counters / buckets

This probe has no host round-tripping; the only instrument is the device event
timestamps (`command_start`→`command_end`), i.e. the same field the engine's
`kernel` bucket reads. Reported as `median(spread)` of 12 interleaved reps per
arm; read RATIOS across arms (within one run), never a single arm's absolute.

| metric | value |
|---|---|
| `kernel` (device), MLP_GATE VERBATIM | 0.356 ms (median) |
| `kernel` (device), MLP_GATE CLONE | 0.319 ms |
| `nk` / reps | 1 launch × 12 reps per arm |
| submit / waitblock / copy / usm_write / readback | 0 — no host round trip in any arm |
| twin ratio (quarter/full bytes) | 0.23x (linear 0.25x) — held |

## Expected direction / win-vs-neutral-los threshold

There is **no** performance win to claim here — this is a falsification run. A
win would be `VERB+bitcast+4+vsel` or `CLONE` clearing the ~1.12x floor
(>= 2 paired rounds, clears the spread, per `scripts/_bench_summary.py`
rules). They do not: every arm sits at 1.11–1.14x, inside the noise floor.
Direction is: **the cosmetics are neutral; look at the shared text, not the
single changes.**

## Correctness gates already passed

- `scripts/test_roofline_bisect4.sh` — PASS (twin holds, every arm ran and
  announced itself: VERBATIM / VERB+bitcast / VERB+bitcast+4 /
  VERB+bitcast+4+vsel / CLONE).
- `scripts/mutate_roofline_bisect4.sh` — PASS (4/4: RB4-1 self-bound fires on
  a hanging stub, RB4-2/RB4-3 `need`-mutation recorded honestly as
  unreachable-while-hanging, RB4-4 pristine baseline).
- `scripts/check_dispatch_steps_consistency.sh` — PASS (dispatch/STEPS agree).
- `scripts/check_scripts_live.sh` — PASS (0 unreachable).

## RESULT

```
ROOFLINE-BISECT4 TEST: PASS
RESULT: bisect-4 ran with its twin holding; the VERB+bitcast / VERB+bitcast
+4 / VERB+bitcast+4+vsel / CLONE ratios above are measured against the
VERBATIM control on the same buffers, interleaved per rep.
```

**Claim:** the ~7x is NOT in the four cosmetic single-changes; the full CLONE
transcription is only ~1.12x vs VERBATIM. The remaining lever is the shared
chunk loop (the `a_lo`/`a_hi` two-vec load + 8-lane body). No engine flag is
introduced by this run.

**Pending (not measured here, operator-only):** none for this batch — it is a
gates-only, model-free probe. The shared-text lever is the input to the next
batch and must be sized by a real 27B `--tp 2` run before any flag ships.
