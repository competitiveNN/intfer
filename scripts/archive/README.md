# scripts/archive — retired, NOT deleted

Scripts here are **not** part of any derived population. Every live family is
defined by a glob one level up (`scripts/test_*.sh` = the correctness gates,
`scripts/check_*.sh` = the checkers, `scripts/mutate_*.sh` = the matrices), so
moving a file into this directory removes it from that family without deleting
history. `git log --follow` still reaches every one of them.

**Why archive and not `git rm`.** Two things are lost by deleting: the *finding*
that killed a script is usually prose in `docs/audit-log-tooling.md` or in a
`reports/` verdict, and prose that names a file which no longer exists becomes a
citation nobody can check. Archiving keeps the pair verifiable — the doc still
points at a real file, and the file still carries its own reasoning.

**Do not run anything from here.** They are here to be read, not executed. If one
is genuinely needed again, move it back and wire it to a live entry point in the
same commit — a script that nothing references is the defect
`scripts/check_scripts_live.sh` now refuses.

## How this set was derived (do not re-derive it by hand)

Not "these look old". The population is `scripts/*.{sh,py,c}`; for each file the
inbound references were harvested with `git grep` over tracked files, and each
hit was classified **code** (a `scripts/`/src/ entry point) or **prose** (a
`docs/`, `reports/`, `AGENTS.md` or comment mention). Every file below had **zero
code references** — only prose, or one comment citing it as an example. That is
the same "derive, never remember" rule the gate and checker populations obey.

The residual is deliberately **not** reported as "unnecessary": a script can be
reachable only by a human typing its name, which is a real use. What is
retired here is scripts whose *subject* is gone, and each row says which.

| script | subject that ended | replaced by |
|---|---|---|
| `merge_27b.sh` | 27B chunk merge, first layout (`test/quant_27b`, 32 chunks) | `merge_27b_batched.sh` |
| `merge_27b_all.sh` | ditto, streaming variant | `merge_27b_batched.sh` |
| `merge_27b_final.sh` | ditto, "re-merge layers 0-15 only" | `merge_27b_batched.sh` |
| `merge_batched.sh` | ditto, *generate-then-merge* | `merge_27b_batched.sh` |
| `merge_27b_batched.sh` | the chunked build of `Qwen3.8-27B.intifer` | **finished** — no successor needed |
| `run_27b_chunks.sh` | per-chunk quantization driver (`run_27b_chunks.sh` first generation) | `run_27b_int2_chunks.sh` |
| `quant_27b_re.sh`, `quant_27b_chunks_redo.sh` | redo/retry variants of the same | `run_27b_int2_chunks.sh` |
| `run_27b_int2_chunks.sh` | the INT2 quantization run that produced the shipped model | **finished** |
| `bench_gpu_ab.sh` | the 0.8B decode/prefill tok/s sweep — WITHDRAWN | counters (`kernel`/`submit`/`waitblock`) |
| `bench_quiet_l4_ab.sh` | `INTFER_L4_FUSE` A/B, recorded a variance wash | dead-end table, AGENTS.md |
| `tab_micro_ab.sh` | A/B of the tile-threshold path once it was promoted default-ON | nothing; the flag is default |
| `gpu_driver_lib_ab.sh` | "is the driver-library version the cause", one window | `gpu_recover.sh` |
| `gpu_health_check.sh` | why card1 fails to enumerate, one window | `gpu_recover.sh diag` |
| `l0_enum_trace.sh` | which of the driver's 5 silent skips fires, one window | `gpu_recover.sh diag` |
| `stream_ids.sh` | hand-run token-id diff | `scripts/test_cpu_parity.sh`, `test_tp2_27b.sh` |
| `measure_load_attribution.sh` | "how much of `load1` during a bench arm is the benchmark itself" — the diagnostic for a *question that is gone*. The **wall-time benchmark queue is closed** (AGENTS.md, PARAMOUNT DIRECTIVE): every optimization is now decided on the load-insensitive counters (`nk`/`nw`/`nr` + the host buckets), so the quiet-box premise this script existed to interrogate no longer governs any decision. Retired 2026-10-15 by `check_scripts_live.sh`, which found it with zero code references. | counters (`nk`/`nw`/`nr`); no successor needed |

## The three findings worth keeping, which lived only in these headers

Deleting these files would have deleted the only copy. All three are reproduced
here; the originals are the evidence.

**1. Per-class budget frac for the 27B INT2 re-quant** (`run_27b_int2_chunks.sh`).
Uniform frac 0.29 is not feasible — chunk 0 is EMBED+LM_HEAD and is the binding
constraint, and the linear-attention chunks need more:

| chunk class | frac | note |
|---|---|---|
| 0 (`EMBED`+`LM_HEAD`, 18 tensors) | **0.29** | 0.28 infeasible, 0.285 `REVIEW` rc=2 |
| `LIN`+`SELF` (16 tensors, `start%4==2`) | 0.29 | |
| `LIN`+`LIN` (17 tensors, `start%4==0`, `start>0`) | **0.305** | 0.29/0.30 infeasible |

Serial `-P1`, ~2.5 min/chunk. Chunks went to `/data` scratch because home was
full. Architecture: `layer%4==3` is self-attn, every other layer linear-attn.

**2. RAM ceiling for a 2-layer chunk** (`quant_27b_re.sh`). Peak RSS is
**OMP-dependent and the OMP=4 setting OOMs**: OMP=2 → ~16 GB/chunk, OMP=4 →
~23 GB/chunk. It only fits while free RAM ≥ ~24 GB. This is why the chunking is
2 layers and not 4.

**3. A driver-library A/B is worthless unless the swap is visible**
(`gpu_driver_lib_ab.sh`). A previous comparison proved nothing because a probe
edit failed to compile, the *old* binary ran, and two identical UUIDs were nearly
read as a result. The script printed `driverVersion` on every arm precisely so
that a swap which did not happen shows up as a **REPEATED** version instead of
being silently trusted. Exit codes: `0` = ≥2 arms with DISTINCT driverVersions and
an equal device count ⇒ version not the cause; `1` = both arms the same version
⇒ the A/B proved nothing (vacuous).

## The one capability these scripts used to own

Re-merging the shipped model from its 32 chunks. The chunks are still on disk at
`/var/home/fra/data/models/intfer/chunks_int2/` (32 × `.intifer`) and the merged
output is `Qwen3.8-27B-int2.intifer` (19 GB), which `test_tp2_27b.sh` loads. The
merge *tool* is not in this directory — it is `examples/quant_merge/`, still built
by CMake — so the capability survives without a script:

```
build/examples/quant_merge/quant_merge OUT.intifer /var/home/fra/data/models/intfer/chunks_int2/*.intifer
```

Pass the chunks in layer order (`sort -V`); `merge_27b_batched.sh` is the archived
driver if the exact invocation is needed.

## 0.8B quant-sweep logs (`quant_int2_08b/`)

Eight `08b_*.log` files moved out of `test/quant_int2/` (2026-10-16) when the
0.8B was demoted from a built-in target to a gate-only fixture. They are
unreferenced historical bit/beta sweep output (`b0.30`…`b0.60`, `ne_b30`) from
the 0.8B INT2 quantiser tuning; no script reads them, and `git grep` found no
consumer. Kept because they are the record of how the 0.8B fixture's quant was
chosen — the fixture itself is still the subject of 7 gates, so its provenance
is worth keeping one directory away rather than deleted.

## The one-GPU-era investigation kit is here, and the fault recurs

`gpu_health_check.sh` + `l0_enum_trace.sh` + `gpu_driver_lib_ab.sh` are a
deliberate ladder built during a window where the second B580 stopped
enumerating. **That box state has dropped and come back four times with nothing
in this repo changing**, so the ladder is not dead — it is one rung below the
documented first rung. AGENTS.md's live remedy is
`scripts/gpu_recover.sh diag` → `sudo scripts/gpu_recover.sh rebind` → `verify`;
if a rebind stops working, read these three from here rather than rebuilding them.
Do not lead with the BAR/small-BAR theory — it was refuted when a rebind fixed
the fault with no BAR change.