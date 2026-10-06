# Batch 2026-10-06 — bisect split + VERB+TAIL / VERB+MP arms

**Context.** The M=1 INT2 GEMV ladder was split across two binaries because a
single translation unit holding more than roughly six to eight INT2 kernel bodies
trips Level-Zero resource limits (`UR_RESULT_ERROR_OUT_OF_RESOURCES`, 40) on
submit, which wedges the driver's `kworker/*+xe_page_fault_work_queue` and takes
the whole box down until a rebind.

- `examples/roofline_probe_bisect1` — ladder-1: the `table` control (tidy rewrite)
  vs `VERBATIM` (the SHIPPED source text, transcribed).
- `examples/roofline_probe_bisect2` (the renamed `roofline_probe_bisect`) — ladder-2:
  `VERBATIM` plus single-change arms `VERB+HALF`, `VERB+TAIL`, `VERB+MP`.
- `examples/roofline_probe_bisect3` — ladder-3: `VERBATIM` plus `VERB+TAIL`,
  `VERB+MP`, and the `SHIPPED` anchor, all four in ONE TU.

All three binaries share `examples/roofline_probe_bisect/probe_common.hpp`: ceiling
computation, the always-red twin, and the interleaved ladder loop. Neither binary
links the engine.

**Gates.** `scripts/test_roofline_bisect1.sh` (ladder-1), `scripts/test_roofline_bisect.sh`
(ladder-2), `scripts/test_roofline_bisect3.sh` (ladder-3). Mutation matrices:
`scripts/mutate_roofline_bisect1.sh`, `scripts/mutate_roofline_bisect.sh`,
`scripts/mutate_roofline_bisect3.sh`. All three gates are guarded by a self-bound check
(`PROBE_TIMEOUT` must stay strictly below the runner's budget) so a hang can never
return `rc=124` and masquerade as a regression.

The `roofline_probe` (original) still carries the already-measured results
(see AGENTS.md, "the probe ladder could not run, so shipping unmeasurable
experiments" section): `table` 193-244 GB/s vs SHIPPED 30.7-41.3 GB/s on MLP_GATE
-- CITATION(roofline_probe, device 0: device event timestamps, medians over 12
interleaved reps; the shipped arm read 17.2 / 39.3 / 41.3 GB/s on three
consecutive runs of the same binary on this contended box, so the absolute GB/s
is not a result and the ratio is the claim). The bisect binaries give the
VERBATIM baseline and the within-run, interleaved ratios that the original probe
could not state.

---

## Experiment 1 — ladder-1: `table` vs `VERBATIM`

(1) **Flag / kill switch / default:** engine-free probe, no flag. Two arms in one
binary: `table` (index 0, baseline for ratios) and `VERBATIM` (index 1, the
transcribed SHIPPED source). No kill switch.

(2) **What it changes and why it should help.** `VERBATIM` is the exact SHIPPED
kernel text moved verbatim into a small TU. If the gap between `table` and
`VERBATIM` is large, the mechanism sits in the shared source text (not the module
and not the branch — both of those were already falsified in `roofline_probe`). If
`VERBATIM` ≈ `table`, the mechanism is outside this transcription (allocator,
launch shape, or something the clone got wrong).

(3) **Benchmark command:**
```
scripts/test_roofline_bisect1.sh [--device N]
```
Quiet-box rule: load1 < 2 is not required — ratios are within-run (interleaved)
and therefore load-insensitive. Device event timestamps; no model bound; gates no
model; ~12 interleaved reps per shape.

(4) **Load-insensitive buckets:** ceiling GB/s (456.0 GB/s roofline, 20 Xe
cores) -- CITATION(docs/Intel Arc B580 Specs _ TechPowerUp GPU Database.html;
ARK sku/241598, cross-checked: the B580's 456.0 GB/s peak bandwidth and 20 Xe
cores, replacing the previously cited A770's ~179 GB/s / 512 Xe cores), twin
ratio (linear = 0.25), per-arm ms and GB/s, and the ratio column `vs VERBATIM`
for both shapes (MLP_GATE 8704x5120 G=16, MLP_DOWN 2560x17408 G=16). `nk`/`nw`/`nr`
not applicable — this is a clone, not the engine.

(5) **Expected direction / win-neutral-loss.**
- Win: `SHIPPED`/`VERBATIM` ratio < 1.0x and closes most of the gap to `table` — the
  mechanism lives in the shared text, and bisect-2's arms point at it.
- Neutral: `SHIPPED`/`VERBATIM` >> 1.0x — the mechanism lives outside the text
  (module/launcher/properties), and the bisect path is spent.
- Loss: an arm that hangs → the TU is too big → split again (the self-bound gate
  catches this and returns rc=2, not a phantom regression).

(6) **Correctness gates already passed:** none required — this is a measurement,
not engine code. Twin validation must hold (exit 1 if not).
- RESULT: (fill from `build/roofline_bisect3.log` — the `VERB+TAIL`, `VERB+MP`, and
  `SHIPPED` GB/s columns and `vs VERBATIM` ratios, per shape)

---

## Experiment 3 — ladder-2 arm: `VERB+HALF`

(1) **Flag / kill switch / default:** engine-free probe arm in `roofline_probe_bisect`.
Corresponds to the shipped `INTFER_DECODE1_VERBHALF` (default OFF,
`INTFER_NO_DECODE1_VERBHALF` kill switch) in the engine.

(2) **What it changes and why it should help.** ONE change from `VERBATIM`: the group
scale is read through `sycl::bit_cast<sycl::half>` + FP32 convert instead of the
kernel's `half_to_float` (a branchy scalar FP16->FP32 convert with ~3 branches per
call). The group scale is called once per GROUP (320 calls at MLP_GATE), so this is
the weaker of the two converts as a candidate — but it is the change actually landed
in the engine, so its cost matters independently of the bisect outcome.

(3) **Benchmark command:**
```
scripts/test_roofline_bisect.sh [--device N]
```
Then, if this arm is positive, verify in the engine with `INTFER_DECODE1_VERBHALF=1`
(the kill switch is name-presence-only, so leave `INTFER_NO_DECODE1_VERBHALF` unset)
and read back the `kernel` bucket (device event timestamps, NOT wall time). Quiet box
load1 < 2 is not needed for the probe ratios; the engine counter run should be done
on a quiet box per the standing rule against relying on tok/s.

(4) **Load-insensitive buckets:** `vs VERBATIM` ratio (interleaved), ceiling, twin;
and in the engine: `kernel` ms, `submit`/`waitblock`/`copy`/`usm_write`/`readback`
`median(spread)`, `nk`/`nw`/`nr`.

(5) **Expected direction / win-neutral-loss.**
- Win: `VERB+HALF`/`VERBATIM` < 1.0x in the probe, and the engine `kernel` bucket
  drops while `nk`/`nw`/`nr` are identical.
- Neutral: ratio ≈ 1.0x → the branchy convert is not the deficit.
- Loss: ratio > 1.0x → regress.
- Note: numerics are NOT bit-exact by construction (FP32 convert vs `half_to_float`
  on zero/subnormal exponents); correctness is stream-identity only, gated by
  `test_rowpair_equiv.sh` (stream-identical to base on the synthetic datasets).

(6) **Correctness gates already passed:** `test_rowpair_equiv.sh` is the right arm
(the 27B model carries INT2; the 0.8B model carries zero, so a smaller model cannot
reach this branch).
- RESULT: (fill — probe `vs VERBATIM` ratio for `MLP_GATE` and `MLP_DOWN`; engine
  `kernel` delta with `INTFER_DECODE1_VERBHALF=1`)

---

## Experiment 4 — ladder-2 arm: `VERB+TAIL`

(1) **Flag / kill switch / default:** engine-free probe arm in `roofline_probe_bisect`.
**There is no corresponding engine flag:** the tail-convert hypothesis was never
landed in the engine. (`INTFER_DECODE1_VERBDEAD` is a DIFFERENT, unrelated
experiment — pure dead-code motion hoisting per-group locals out of the group loop —
and must not be used to benchmark this arm.)

(2) **What it changes and why it should help.** ONE change from `VERBATIM`: the TAIL
activation dequant goes through `sycl::bit_cast<sycl::half>` + FP32 convert instead
of `half_to_float`. The tail convert is called 320x per work-item at MLP_GATE (once
per GROUP for the group scale), so this is the stronger candidate for closing the
~5x gap. If `VERBATIM` is still far behind `table` after the group-scale change,
the tail convert is the next suspect.

(3) **Benchmark command:**
```
scripts/test_roofline_bisect.sh [--device N]
```
There is no engine flag to verify this arm against; the probe is the experiment.
If a downstream engineer wants to port the tail convert to the engine, the arm
would be implemented as a new flag guarded exactly like `INTFER_DECODE1_VERBHALF`
and gated by `test_rowpair_equiv.sh` stream-identity before any `kernel` delta is
trusted.

(4) **Load-insensitive buckets:** `vs VERBATIM` ratio, ceiling, twin. No engine
counter run is possible for this arm (no engine flag exists); the probe's own
`vs VERBATIM` ratio is the measurement.

(5) **Expected direction / win-neutral-loss.**
- Win: `VERB+TAIL`/`VERBATIM` < 1.0x in the probe, and the `VERB+HALF` arm is
  neutral or worse (so the tail convert is the real deficit).
- Neutral: ratio ≈ 1.0x → the tail convert is not the deficit either.
- Loss: ratio > 1.0x → regress.
- NUMERICS: NOT bit-exact by construction (same zero/subnormal exponent concern as
  `VERB+HALF`). The probe is a clone, so stream-identity to the engine is not
  asserted here; a ported engine arm would need `test_rowpair_equiv.sh` before any
  `kernel` delta is trusted.

(6) **Correctness gates already passed:** none required for this arm — it is a
probe-only measurement. If ported to the engine, `test_rowpair_equiv.sh` (27B
INT2 only) is the gate.
- RESULT: (fill — `VERB+TAIL` GB/s and `vs VERBATIM` ratio on MLP_GATE and MLP_DOWN)

---

## Experiment 5 — ladder-2 arm: `VERB+MP`

(1) **Flag / kill switch / default:** engine-free probe arm in `roofline_probe_bisect`.
Corresponds to the shipped `INTFER_DECODE1_VERBMP` (default OFF,
`INTFER_NO_DECODE1_VERBMP` kill switch) in the engine.

(2) **What it changes and why it should help.** ONE change from `VERBATIM`: the weight
side is traversed through a row-base `sycl::multi_ptr<const std::uint8_t>` instead of
an absolute index into the raw `code_u8` buffer. Chunk loop uses a row-RELATIVE byte
offset `b0`; the tail deliberately uses the unoffset `code_u8` with absolute indices
(exactly as the base kernel). This is the same shape as the `INTFER_DECODE1_ROWPAIR`
double-offset defect class, so it carries a correctness risk even though the indices
are written to match the base kernel's values.

(3) **Benchmark command:**
```
scripts/test_roofline_bisect.sh [--device N]
```
Engine verification: `INTFER_DECODE1_VERBMP=1 INTFER_NO_DECODE1_VERBMP` left unset,
then `test_rowpair_equiv.sh` stream-identity (mandatory before trusting any `kernel`
delta — the device-loss failure mode is silent until the arm dies).

(4) **Load-insensitive buckets:** `vs VERBATIM` ratio, ceiling, twin; engine
`kernel` ms, `submit`/`waitblock`/`copy`/`usm_write`/`readback`, `nk`/`nw`/`nr`.

(5) **Expected direction / win-neutral-loss.**
- Win: `VERB+MP`/`VERBATIM` ≈ 1.0x in the probe (index reformulation, not a
  throughput play) AND `test_rowpair_equiv.sh` passes stream-identical in the engine.
- Neutral: ratio ≈ 1.0x but engine correctness fails → revert the engine flag; the
  probe cannot save it.
- Loss: ratio > 1.0x → regress; ratio < 1.0x → suspicious, recheck the indexing.
- NUMERICS: bit-exact by construction ONLY IF the indices truly match the base
  kernel — a claim that must be MEASURED (the double-offset class has cost this
  repo twice), not read off the code.

(6) **Correctness gates already passed:** `test_rowpair_equiv.sh` (27B INT2). The
`VERBMP` result in AGENTS.md is **withdrawn** (`"SO: NO verb* ARM HAS A CORRECTNESS
MEASUREMENT"`); a clean rebuild + healthy 2-GPU box is required to complete it.
- RESULT: (fill — probe ratio; `test_rowpair_equiv.sh` outcome and engine `kernel`
  delta with `INTFER_DECODE1_VERBMP=1`)

---

## Experiment 6 — `SHIPPED` anchor (baseline sanity check)

(1) **Flag / kill switch / default:** engine-free reference in `roofline_probe`, not a
bisect arm. No flag; `INTFER_DPAS48_TILED`/`INTFER_DECODE1_VERBMP` etc. are the
engine side.

(2) **What it establishes.** `SHIPPED` = the real INT2 kernel from the engine, run on
the same buffers with the same allocator and measured with device event timestamps.
Ladder-3 (`test_roofline_bisect3.sh`) now measures `SHIPPED`/`VERBATIM` in-TU and is
the authoritative value for the ~4.5-5x of the 5.6-6.4x gap that survives the module
move (1.26x) and the runtime branch (1.007x). `roofline_probe` remains as an INDEPENDENT
baseline check: the cross-binary `SHIPPED`/`table` ratio must confirm the known
5.6-6.4x deficit (AGENTS.md: SHIPPED 30.7-41.3 GB/s vs `table` 193-244 GB/s on
MLP_GATE) -- CITATION(roofline_probe, device 0: device event timestamps, medians over
12 interleaved reps; the shipped arm read 17.2 / 39.3 / 41.3 GB/s on three
consecutive runs of the same binary on this contended box, so the absolute GB/s is
not a result and the ratio is the claim). If ladder-3's in-TU `SHIPPED` ratio and this
cross-binary check diverge, the register-pressure difference between the two module
contexts is the cause — and that divergence itself is a finding about the bisect's
measurement fidelity.

(3) **Benchmark command:**
```
scripts/test_roofline_probe.sh [--device N]
```

(4) **Load-insensitive buckets:** `nk`/`nw`/`nr`, per-arm ms and GB/s, ceiling,
twin. Ratios are within-run interleaved. -- CITATION(roofline_probe, device 0:
device event timestamps, medians over 12 interleaved reps; the shipped arm read
17.2 / 39.3 / 41.3 GB/s on three consecutive runs of the same binary on this
contended box, so the absolute GB/s is not a result and the ratio is the claim)

(5) **Expected direction / win-neutral-loss.**
- Win: `SHIPPED`/`table` ratio confirms the known 5.6-6.4x deficit (matches the
  AGENTS.md numbers 30.7-41.3 GB/s for SHIPPED vs `table` on MLP_GATE)
  -- CITATION(roofline_probe, device 0: device event timestamps, medians over 12
  interleaved reps; the shipped arm read 17.2 / 39.3 / 41.3 GB/s on three
  consecutive runs of the same binary on this contended box, so the absolute GB/s
  is not a result and the ratio is the claim). If SHIPPED now measures close to
  `table`, something in this session's refactor regressed the baseline and the
  bisect numbers would be meaningless.
- Loss: ratio < 4x → the baseline is broken; bisect results are void.

(6) **Correctness gates already passed:** none required — this is the established
baseline, gated by `test_roofline_probe.sh` with its own twin.
- RESULT: (fill — SHIPPED GB/s and SHIPPED/table ratio on MLP_GATE and MLP_DOWN; must
  reproduce the 5.6-6.4x deficit)

---

## Operator notes

- **Do not read absolute GB/s off this report as a verdict.** Box load moves the
  absolutes; only within-run interleaved ratios are load-insensitive. Quote ratios,
  not absolute GB/s, in any handoff.
- **Engine counter runs use device event timestamps, not wall time.** The
  `kernel` bucket is the only load-insensitive bucket; `submit`/`waitblock`/
  `copy`/`usm_write`/`readback` are host wall-clock accumulators reported as
  `median(spread)` and read RELATIVELY (direction, not figure).
- **Load1 < 2 is not required for the bisect probes** (interleaved, same process).
  It is still the rule for any 27B `--tp 2` engine counter run (standing rule 1:
  an agent session should not perform a full 27B gate anyway).
- **Numerics contracts differ per arm.** `VERB+MP` is claimed bit-exact (but must be
  measured); `VERB+HALF` and `VERB+TAIL` are stream-identity only. Gate each with
  the corresponding `test_rowpair_equiv.sh` arm before believing a `kernel` delta.
- **Box caveat.** If a run hangs, the box may have lost a render node; run
  `scripts/gpu_recover.sh diag` and, if needed, `sudo scripts/gpu_recover.sh rebind`
  before re-running. A hang is a box symptom until the box is checked first.
- **Report the exact command used and the log file location** (`build/roofline_bisect1.log`,
  `build/roofline_bisect.log`, `build/roofline_probe_*.log`) with every result so it
  can be re-derived.
