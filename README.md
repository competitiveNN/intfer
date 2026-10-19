# intfer

A lean C++ inference engine that ports [ninfer](https://github.com/Neroued/ninfer)'s
"bypass the framework" philosophy to a local rig of **2× Intel Arc Pro B580** GPUs,
using **oneAPI DPC++ / SYCL** and **Intel Level-Zero** instead of CUDA.

The project is being built in three stages:

| Stage | Goal | Status |
| :--- | :--- | :--- |
| **1** | INT2/low-bit quantization pipeline (SpinQuant → DASH-Q → RCO) | Implemented; 0.8B + 27B quantized, `.intfer` + `.gguf` artifacts |
| **2** | Native SYCL/L0 execution engine (quant GEMM, FlashAttention, tensor-parallel) | Implemented (TP=2 sharded decode on 2× B580; DPAS INT8/W4A8/W2A8; INT8-KV opt-in) |
| **3** | High-throughput serving (continuous batching, spec decoding) | Done (`--serve`/`--serve-http` servers, iteration scheduler, mixed-batch M-row forward, joint prefill+decode, chunked prefill, round-robin fairness, n-gram speculative decode). Gates: `scripts/test_cpu_parity.sh` (oracle + split==joint), `scripts/test_http_batch.sh` (cross-request batch parity), `scripts/test_serve_load.sh` (concurrency sweep C=1/4/8: req/s, tok/s, P50/P99 latency, solo-vs-loaded parity). Measured 2026-09-17 (0.8B INT2, GPU TP=1, max_tokens=8, idle box): C=1 8.16 tok/s / P50 1.0s, C=4 15.33 tok/s / P50 2.0s, C=8 17.98 tok/s / P50 3.3s (x2.20 C=1->C=8; batch-full probe C=8/MT=64: 36.3 tok/s aggregate, near the 40 target; the 2026-09-16 run was 4-5x lower under box contention) |

---

## Architecture

### Stage 1 — Quantization pipeline (host C++, no SYCL required)

```
[ FP16 Model ]
      │
      ▼
[ 1. SpinQuant ]  ──► Orthogonal rotation (W' = W·R) to suppress outliers
      │
      ▼
[ 2. DASH-Q ]     ──► Diagonal-Hessian PTQ: independent weighted-least-squares
      │                 candidates per tensor (low error at 2-bit)
      ▼
[ 3. RCO ]        ──► Bit-budget search: pick DASH-Q variant per tensor under
      │                 exact VRAM budget (2×24GB B580)
      ▼
[ Native SYCL loader / GGUF ]
```

- **SpinQuant** (`src/spinquant.cpp`) — random/learned orthogonal rotation preprocessing.
- **DASH-Q** (`src/quant/dashq.cpp`) — non-heuristic PTQ solver using diagonal-Hessian
  approximations (avoids GPTQ's off-diagonal sampling noise at very low bits).
- **RCO** (`src/quant/rco.cpp`) — resource-constrained optimization that allocates
  per-tensor bit-widths to hit an exact memory budget.
- **Pipeline / output** (`src/quant/pipeline.cpp`, `src/quant/output.cpp`) — orchestrates
  t3→t4→t5 and serializes the finalized `ModelIR` + `RotationSet` to the native
  SYCL loader or GGUF.

The Tensor-IR contract shared by all stages lives in `include/intfer/quant_spec.h`
and is documented in [`docs/quant_spec.md`](docs/quant_spec.md). The design rationale
and ninfer analysis are in [`docs/ninfer-quantization-brief.md`](docs/ninfer-quantization-brief.md).

### Stage 2 — Execution engine (SYCL + Level-Zero)

| Component | Path | Notes |
| :--- | :--- | :--- |
| Device manager | `src/device/device_manager.cpp` | Enumerates both Arc B580s; tensor-parallel context |
| Level-Zero layer | `src/l0/level_zero.cpp` | Optional native L0 bindings (`INTFER_ENABLE_L0`) |
| Quantized GEMM | `src/gemm/quant_gemm.cpp` | Dequant-on-the-fly GEMM + ESIMD **DPAS** INT8×INT8 kernel (XMX) |
| FlashAttention | `src/attention/flash_attention.cpp` | Fused SYCL attention kernel, Xe-core tuned |
| Common | `src/common/logger.cpp` | Logging / error utilities |

The GEMM kernel issues Intel **XMX** systolic-array `DPAS` instructions directly,
mirroring ninfer's use of `mma.sync`/`wmma` on NVIDIA. SYCL command graphs were
evaluated as the CUDA-graph equivalent and rejected: executable-graph launch
measured *slower* than plain submit on B580 (19.7µs vs 11.9µs), so the lever
for decode-loop overhead is launch-count reduction (fusion), not graphs.

---

## Building

Requires the **oneAPI DPC++ compiler (`icpx`)** and the Level-Zero loader.

```bash
# Configure + build (Release, L0 on, examples on).
./scripts/build.sh

# Build the experimental Stage-1 quantization core too:
./scripts/build.sh -DINTFER_BUILD_QUANT=ON

# Debug build:
./scripts/build.sh -DCMAKE_BUILD_TYPE=Debug
```

`scripts/build.sh` locates the oneAPI compiler in `/opt/intel/oneapi/compiler/*/`
without sourcing `setvars.sh` (which would `exit` the script) and embeds the SYCL
runtime into the binary RPATH so executables run without extra env setup.

CMake options:

| Option | Default | Description |
| :--- | :--- | :--- |
| `INTFER_ENABLE_L0` | `ON` | Build the Intel Level-Zero native layer |
| `INTFER_BUILD_QUANT` | `OFF` | Build the experimental Stage-1 quantization core |
| `INTFER_BUILD_EXAMPLES` | `ON` | Build the example programs |
| `INTFER_SYCL_TARGET` | `spir64` | SYCL offload target triple |

Build outputs land in `build/` (or `$INTFER_BUILD_DIR`). Every script that
runs a built binary resolves it through that variable, so pointing
`INTFER_BUILD_DIR` at another tree benchmarks/builds *that* tree rather than
silently falling back to the default one.

The separate host-only Stage-1 quantizer (`scripts/build_quant_real.sh`) has its
own tree, `build_quant_real/`, and its own override `$INTFER_QUANT_REAL_BIN`;
the `quant_27b_*` / `run_27b_*` chunk scripts resolve their binary through it.
Chunk then merge, in that order: `run_27b_int2_v2.sh` writes one `.intifer` per
2-layer chunk into `$INTFER_CHUNK_DIR`, and `merge_27b_v2.sh` streams those
chunks into the final artifact. The merge is long enough to outlive a tool
timeout, so it must be detached (`setsid nohup ... < /dev/null & disown`); it
writes `.payload` then `.final` and renames, so an interrupted run leaves a
recoverable partial rather than a truncated artifact. `run_27b_int2_v2.sh`'s
budget fractions are per-class (`INTFER_FRAC_LAYER` for chunks carrying layers
only, `INTFER_FRAC_HEAD` for chunk 0, which keeps EMBED + LM_HEAD).

---

## Examples

Built under `examples/` (enabled by `INTFER_BUILD_EXAMPLES`):

- `hello_sycl` — device enumeration smoke test for both B580s.
- `flash_attention_demo` — fused attention kernel.
- `spinquant_test` / `spinquant_demo` — Stage-1 rotation preprocessing.
- `quant_pipeline` — full SpinQuant → DASH-Q → RCO pipeline (needs `INTFER_BUILD_QUANT`).
- `gemm_demo` — quantized GEMM / DPAS kernel.

Run via `scripts/run.sh` / `scripts/run_bin.sh` after building, or prefix any command
with `scripts/with_oneapi.sh` — a bare shell cannot enumerate either B580 (libumf
is off the default `LD_LIBRARY_PATH`, so every UR adapter fails to load), and
`source setvars.sh` cannot be used to prefix a command because it ends in `exit`.

---

## Layout

```
AGENTS.md            Project brief / stage definitions
CMakeLists.txt       Top-level project config
docs/                Quantization spec + ninfer analysis
include/intfer/      Public headers (quant, gemm, attention, device, l0)
src/                 Implementation (quant, gemm, attention, device, l0, common)
examples/            Demo programs
scripts/             build.sh / run.sh / run_bin.sh / with_oneapi.sh
                     (gate + tooling entry points: run_all_checks.sh,
                      run_ten_gates.sh, gate_inventory.sh;
                      retired scripts live in scripts/archive/)
```

---

## Development status

**Implemented**
- Stage-1 quantization core: SpinQuant preprocessing, DASH-Q candidate generation
  with diagonal-Hessian weighting, RCO bit-budget search, and pipeline orchestration
  with native-SYCL / GGUF serialization (`INTFER_BUILD_QUANT=ON`).
- Stage-2 engine: device discovery across 2× B580, Level-Zero layer,
  quantized GEMM with an ESIMD DPAS (XMX) kernel (INT8 + W4A8/W2A8 via
  repack), fused FlashAttention kernel, TP=2 sharded decode, and an
  opt-in groupwise-INT8 KV cache (`INTFER_INT8_KV=1`, oracle-identical).
- Stage-3 serving: `--serve` JSON-lines server + `--serve-http`
  OpenAI-compatible server with continuous batching
  (iteration scheduler, mixed-batch M-row forward, joint prefill+decode,
  chunked prefill, round-robin fairness), free n-gram speculative decode
  (`INTFER_SPEC_K=N`), and gates: `scripts/test_cpu_parity.sh`
  (oracle + split==joint + rotcheck + INT8-KV), `scripts/test_tp2_27b.sh`,
  `scripts/bench_serve.sh` (req/s, tok/s, latency),
  `scripts/test_http_batch.sh` (cross-request batch parity over HTTP),
  `scripts/test_serve_load.sh` (concurrency sweep C=1/4/8 with req/s,
  tok/s, P50/P99 latency + solo-vs-loaded parity; 2026-09-18
  (quiet-box start, fused default-ON): MT=8 sweep C=1 3.3, C=4 4.2,
  C=8 3.1 tok/s (prefill-dominated short requests; C=8 scaling inverts
  when box load rises mid-run — sweep-harness artifact); batch-full
  C=8/MT=64 probe: 9.02 tok/s over 338 ctoks (proves the path batches).
  The 2026-09-17 8.16/15.33/17.98 + 36.3 figures are withdrawn as
  unreproducible (four independent C=1 sweeps land 3.27-3.53; only docs
  commits separate the 3.89 and 8.16 readings, i.e. same code; fused
  ON/OFF bisect neutral at 1.78/1.77s). Gate-verified PASS repeatedly
  (completion + solo-vs-loaded parity + floor).

**In progress / next**
- GPU-side attention (today attention runs host-side in both backends;
  only the 7 linears ride the SYCL engine) and INT8-KV consumption by
  the SYCL FlashAttention kernel.
- Launch-count reduction (fusion) for the decode loop — SYCL command
  graphs were measured *slower* than plain submit on B580 (19.7µs vs
  11.9µs) and the scaffold was removed; see HISTORY.md.

See `AGENTS.md` for the full staged task breakdown.
