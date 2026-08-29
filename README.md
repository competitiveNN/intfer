# intfer

A lean C++ inference engine that ports [ninfer](https://github.com/Neroued/ninfer)'s
"bypass the framework" philosophy to a local rig of **2× Intel Arc Pro B580** GPUs,
using **oneAPI DPC++ / SYCL** and **Intel Level-Zero** instead of CUDA.

The project is being built in three stages:

| Stage | Goal | Status |
| :--- | :--- | :--- |
| **1** | INT2/low-bit quantization pipeline (SpinQuant → DASH-Q → RCO) | Core implemented (experimental) |
| **2** | Native SYCL/L0 execution engine (quant GEMM, FlashAttention, tensor-parallel) | Engine scaffolding + kernels implemented |
| **3** | High-throughput serving (continuous batching, spec decoding) | Planned |

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
mirroring ninfer's use of `mma.sync`/`wmma` on NVIDIA. SYCL command graphs
(`sycl::ext::oneapi::experimental::command_graph`) are the planned replacement for
CUDA graphs to keep decode-loop launch overhead near zero.

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

Build outputs land in `build/` (or `$INTFER_BUILD_DIR`).

---

## Examples

Built under `examples/` (enabled by `INTFER_BUILD_EXAMPLES`):

- `hello_sycl` — device enumeration smoke test for both B580s.
- `flash_attention_demo` — fused attention kernel.
- `spinquant_test` / `spinquant_demo` — Stage-1 rotation preprocessing.
- `quant_pipeline` — full SpinQuant → DASH-Q → RCO pipeline (needs `INTFER_BUILD_QUANT`).
- `gemm_demo` — quantized GEMM / DPAS kernel.

Run via `scripts/run.sh` / `scripts/run_bin.sh` after building.

---

## Layout

```
AGENTS.md            Project brief / stage definitions
CMakeLists.txt       Top-level project config
docs/                Quantization spec + ninfer analysis
include/intfer/      Public headers (quant, gemm, attention, device, l0)
src/                 Implementation (quant, gemm, attention, device, l0, common)
examples/            Demo programs
scripts/             build.sh / run.sh / run_bin.sh
```

---

## Development status

**Implemented**
- Stage-1 quantization core: SpinQuant preprocessing, DASH-Q candidate generation
  with diagonal-Hessian weighting, RCO bit-budget search, and pipeline orchestration
  with native-SYCL / GGUF serialization (`INTFER_BUILD_QUANT=ON`).
- Stage-2 engine skeleton: device discovery across 2× B580, Level-Zero layer,
  quantized GEMM with an ESIMD DPAS (XMX) kernel, and a fused FlashAttention kernel.

**In progress / next**
- Wire the quantized GEMM + FlashAttention kernels into an end-to-end decode loop.
- Tensor-parallel weight sharding across the two GPUs (Stage 2 mandate).
- SYCL command-graph capture for the generation loop (CUDA-graph equivalent).
- Groupwise-INT8 KV-cache management.

**Planned (Stage 3)**
- Continuous batching for concurrent requests.
- Multi-token prediction / speculative decoding.
- Throughput-focused serving path.

See `AGENTS.md` for the full staged task breakdown.
