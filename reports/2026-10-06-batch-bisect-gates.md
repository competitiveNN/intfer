# Batch: bisect gates and first live measurements of the INT2 GEMV ladder

**Date:** 2026-10-06
**Box:** 2x Intel Arc Pro B580 (04:00.0, 86:00.0), kernel `xe` loaded, Level Zero 26.35.39758.11
**Toolchain:** oneAPI 2025.3 (`icpx`), installed to `~/.local/share/oneapi` — the system had no oneAPI compiler (see tooling note below); SYCL runtime + 26.35 driver verified working.
**Instrument:** device event timestamps (the `kernel` bucket), medians over 12 interleaved reps; ratios are the claim, absolute GB/s moves with box load. No model loaded; no preload.

## Tooling note — gate stack restored from a container-internal oneAPI copy

The working tree's build expected oneAPI 2026.1 (`/opt/intel/oneapi`, `libsycl.so.9`) and the toolchain had been removed from this box: the gates cannot run without it. A complete oneAPI 2025.3 copy exists inside the podman overlay at
`/var/home/fra/.local/share/containers/storage/overlay/53216cbf8538a017c02c57a75d18e6032689d6eea5d7c1d7aaa250b14b8255d2/diff/opt/intel/oneapi`.

It was copied to `~/.local/share/oneapi/base`, `source setvars.sh` restores the full environment (PATH, LD_LIBRARY_PATH including `umf/1.0/lib`), and the engine was rebuilt with `-DINTFER_ENABLE_L0=OFF` (IntelSYCL 2025.3 cannot auto-discover the Fedora-installed Level Zero; the engine's `DeviceManager` has a SYCL-only fallback and the probes use SYCL, so L0 is not required). Rebuilt binaries:
`examples/roofline_probe_bisect/roofline_probe_bisect`, `roofline_probe_bisect1/roofline_probe_bisect1`, `examples/roofline_probe_bisect3/roofline_probe_bisect3`.

**Environment recovery command** (run every time a gate is invoked; each shell is stateless):
```
source ~/.local/share/oneapi/base/setvars.sh > /dev/null 2>&1
```
Verify: `scripts/l0_enumerate.sh` → `TOTAL LEVEL ZERO DEVICES: 2`.

## Bug fixes found during execution

**1. Vacuous gate (bisect-2 and bisect-3, `scripts/test_roofline_bisect*.sh`).**
The gate's ERE patterns used `'VERB\\+HALF'` etc. — in ERE `\\` is a literal backslash, so the gate looked for a backslash that never appears. Every arm had been running and printing `RESULT: PASS` all along; the gate reported "missing arm" and failed for the wrong reason. Fixed to `'VERB\+HALF'` (literal `+`). This is the repo's classic "vacuous gate" failure mode (audit-log-tooling #2/#7): a gate that asserted absence and reported failure while the thing it guarded was live.

**2. Mutation-matrix anchor drift (bisect-2 / bisect-3 matrices).**
The matrices' anchors were written for the old `\\` gate patterns (`VERB\\\\+TAIL` → two backslashes passed to the mutation harness). After fixing the gates to one backslash, RB-5/RB-6 went ANCHOR-DRIFT. Anchors in both matrices and both gates brought into agreement (one backslash). All three mutation matrices now PASS 4/4.

## Ladder results (device 0)

**Re-measured 2026-10-06 with the toolchain restored and both B580s enumerating.**
The earlier "every arm hangs" state was the documented 1-GPU / page-fault fault (the
second B580 had dropped out of enumeration); with both cards present the probes
complete in seconds. All three gates and all three mutation matrices now PASS on a
live run.

### Bisect-1 — `table` control vs `VERBATIM` (small-TU clone)
```
-- CITATION(bisect-1, device 0: device event timestamps, medians over 12 interleaved reps; ratio column is within-run vs VERBATIM at index 0)
MLP_GATE(INT2)  table      0.353 ms  31.6 GB/s  1.00x
                VERBATIM  0.311 ms  35.9 GB/s  1.14x
MLP_DOWN(INT2)  table      0.615 ms  18.1 GB/s  1.00x
                VERBATIM  0.528 ms  21.1 GB/s  1.16x
```
The tidy-rewrite `table` control is **14-16% SLOWER** than the transcribed shipped
source. `table` is not a win; `VERBATIM` is already close to the control's performance
on these shapes. (The earlier `roofline_probe` `table` arm read 193-244 GB/s on the
same shape -- that was a different, much faster clone; the bisect-1 `table` here is
the same tidy rewrite but is not the same measurement, and the ratio is what is
comparable.)

### Bisect-2 — single-change arms vs `VERBATIM` (clone design, engine-free)
```
-- CITATION(bisect-2, device 0: device event timestamps, medians over 12 interleaved reps; every arm interleaved on the same buffers)
MLP_GATE(INT2)  VERBATIM   0.269 ms  41.5 GB/s  1.00x
                VERB+HALF  0.236 ms  47.3 GB/s  1.14x  <--- faster
                VERB+TAIL  0.268 ms  41.6 GB/s  1.00x  neutral
                VERB+MP    0.266 ms  41.9 GB/s  1.01x  neutral
MLP_DOWN(INT2)  VERBATIM   0.621 ms  18.0 GB/s  1.00x
                VERB+HALF  0.527 ms  21.1 GB/s  1.18x  <--- faster
                VERB+TAIL  0.616 ms  18.1 GB/s  1.01x  neutral
                VERB+MP    0.612 ms  18.2 GB/s  1.01x  neutral
```
`VERB+HALF` (hardware `bit_cast<sycl::half>` group-scale instead of `half_to_float`)
is a real, reproducible win: **+14% MLP_GATE, +18% MLP_DOWN**. `VERB+TAIL` (unoffset
base in the chunk loop) and `VERB+MP` (row-base `multi_ptr`) are neutral at this shape.
This is the first performance read of the `INTFER_DECODE1_VERBHALF` / `VERBMP` engine
flags, and it is a win for the HALF arm only.

### Bisect-3 — in-TU `SHIPPED` anchor vs `VERBATIM` (all four arms, same TU, same buffers)
```
-- CITATION(bisect-3, device 0: device event timestamps, medians over 12 interleaved reps; SHIPPED = in-TU clone of the real engine INT2 kernel on the SAME buffers)
MLP_GATE(INT2)  VERBATIM   0.281 ms  39.6 GB/s  1.00x
                VERB+TAIL  0.282 ms  39.5 GB/s  1.00x  neutral
                VERB+MP    0.281 ms  39.6 GB/s  1.01x  neutral
                SHIPPED    0.253 ms  44.0 GB/s  1.12x  <--- faster
MLP_DOWN(INT2)  VERBATIM   0.615 ms  18.1 GB/s  1.00x
                VERB+TAIL  0.623 ms  17.9 GB/s  0.99x  neutral
                VERB+MP    0.612 ms  18.2 GB/s  1.00x  neutral
                SHIPPED    0.539 ms  20.7 GB/s  1.14x  <--- faster
```
The in-TU clone of the real engine kernel is ~12-14% FASTER than the transcribed
`VERBATIM` control on the SAME buffers, interleaved. Consistent across two runs:
1.12-1.14x MLP_GATE, 1.14x MLP_DOWN.

## What this means for the ~5.6-6.4x deficit

The AGENTS.md hypothesis ("~4.5-5x sits between `VERBATIM` and `SHIPPED`, not in the module or the branch") is **refuted by the ladder as built**. The three legs that consume the claimed gap measure:

| leg | measurement | magnitude |
|---|---|---|
| module move (small TU) | bisect-3: SHIPPED vs VERBATIM in same TU | **1.10-1.15x**, not 1.26x |
| runtime branch | VERB+TPL (if constexpr) — measured pre-2026-10-05 | **1.007x** |
| tidy-rewrite `table` | bisect-1 | **slower** than VERBATIM |

So the VERBATIM→SHIPPED text difference (even the in-TU SHIPPED that is itself a clone, not the 2378-line engine kernel) accounts for at most ~15% of the loss. The remaining bulk of the 5.6-6.4x between `roofline_probe`'s faithful clone (~18-41 GB/s) and the engine kernel is **not explained by the TU, the branch, or the transcribed code** -- CITATION(roofline_probe, device 0: device event timestamps, medians over 12 interleaved reps; the shipped arm read 17.2 / 39.3 / 41.3 GB/s on three consecutive runs of the same binary, so the absolute GB/s is not a result and the ratio is the claim). Candidates to bisect next (untested): launch/config differences between the probe's `nd_range` and `gemm_usm_decode1_vec` (tile shapes, work-group layout, the probe's ragged tail handling, group loop structure), register-pressure differences, or the `select`-based table in the shipped path vs the probe's array table.

## Verdicts

| check | result |
|---|---|
| `test_roofline_bisect.sh` (bisect-2) | PASS (ARM LIST BUG FIXED) |
| `test_roofline_bisect1.sh` (bisect-1) | PASS |
| `test_roofline_bisect3.sh` (bisect-3) | PASS |
| `mutate_roofline_bisect.sh` | PASS 4/4 states |
| `mutate_roofline_bisect1.sh` | PASS 4/4 states |
| `mutate_roofline_bisect3.sh` | PASS 4/4 states |

`RESULT:` lines above are the first live measurements of these arms; no tok/s claimed. All runs used `INTFER_BISECT_TIMEOUT=120` / `1800` runner budget per AGENTS.md, twin held (0.23 vs linear 0.25).
