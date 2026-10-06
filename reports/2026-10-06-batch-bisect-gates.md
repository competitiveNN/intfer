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

### Bisect-1 — `table` control vs `VERBATIM` (small-TU clone)
```
MLP_GATE(INT2)  table      0.451 ms  24.7 GB/s  1.00x
                  VERBATIM  0.396 ms  28.1 GB/s  1.14x
MLP_DOWN(INT2)  table      1.164 ms   9.6 GB/s  1.00x
                  VERBATIM  1.000 ms  11.1 GB/s  1.16x
```
The tidy-rewrite `table` control is ~14-16% SLOWER than the transcribed shipped source. The "table" arm is not a win; the shipped source is already close to this control's performance on these shapes.

### Bisect-2 — single-change arms vs `VERBATIM` (clone design, engine-free)
```
MLP_GATE(INT2)  VERBATIM   0.432 ms  25.8 GB/s  1.00x
                VERB+HALF  0.379 ms  29.4 GB/s  1.14x  <--- faster
                VERB+TAIL  0.433 ms  25.7 GB/s  1.00x  neutral
                VERB+MP    0.431 ms  25.9 GB/s  1.00x  neutral
MLP_DOWN(INT2)  VERBATIM   0.614 ms  18.1 GB/s  1.00x
                VERB+HALF  0.526 ms  21.2 GB/s  1.17x  <--- faster
                VERB+TAIL  0.614 ms  18.1 GB/s  1.00x  neutral
                VERB+MP    0.611 ms  18.2 GB/s  1.01x  neutral
```
`VERB+HALF` (hardware `bit_cast<sycl::half>` group-scale instead of `half_to_float`) is a real, reproducible win: **+14% MLP_GATE, +17% MLP_DOWN**. `VERB+TAIL` (unoffset base in the chunk loop) and `VERB+MP` (row-base `multi_ptr`) are neutral at this shape — interesting given the shipped `INTFER_DECODE1_VERBHALF`/`VERBMP` engine flags were accepted on correctness, not performance; this is their first performance read and it's a win for the HALF arm only.

### Bisect-3 — in-TU `SHIPPED` anchor vs `VERBATIM` (all four arms, same TU, same buffers)
```
MLP_GATE(INT2)  VERBATIM   0.404 ms  27.6 GB/s  1.00x
                SHIPPED    0.368 ms  30.3 GB/s  1.10x  <--- faster
MLP_DOWN(INT2)  VERBATIM   0.615 ms  18.1 GB/s  1.00x
                SHIPPED    0.542 ms  20.5 GB/s  1.13x  <--- faster
```
The in-TU clone of the real engine kernel is ~10-15% FASTER than the transcribed `VERBATIM` control on the SAME buffers, interleaved. Run twice: 1.10-1.15x MLP_GATE, 1.13-1.15x MLP_DOWN. Consistent.

## What this means for the ~5.6-6.4x deficit

The AGENTS.md hypothesis ("~4.5-5x sits between `VERBATIM` and `SHIPPED`, not in the module or the branch") is **refuted by the ladder as built**. The three legs that consume the claimed gap measure:

| leg | measurement | magnitude |
|---|---|---|
| module move (small TU) | bisect-3: SHIPPED vs VERBATIM in same TU | **1.10-1.15x**, not 1.26x |
| runtime branch | VERB+TPL (if constexpr) — measured pre-2026-10-05 | **1.007x** |
| tidy-rewrite `table` | bisect-1 | **slower** than VERBATIM |

So the VERBATIM→SHIPPED text difference (even the in-TU SHIPPED that is itself a clone, not the 2378-line engine kernel) accounts for at most ~15% of the loss. The remaining bulk of the 5.6-6.4x between `roofline_probe`'s faithful clone (~18-41 GB/s) and the engine kernel is **not explained by the TU, the branch, or the transcribed code**. Candidates to bisect next (untested): launch/config differences between the probe's `nd_range` and `gemm_usm_decode1_vec` (tile shapes, work-group layout, the probe's ragged tail handling, group loop structure), register-pressure differences, or the `select`-based table in the shipped path vs the probe's array table.

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
