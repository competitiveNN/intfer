To maximize both token decode and prefill performance on a dual Intel Arc B580 setup (2×12GB VRAM), optimizations must target hardware-native XMX DPAS execution, PCIe multi-GPU interconnect overhead, and SYCL graph execution.

---

### 1. VRAM Budget & Multi-GPU Topology (2×12GB)

To eliminate runtime layer-streaming overhead during generation, all model weights and workspaces must fit permanently in GPU VRAM (`preload_all()`).

* **RCO Bit-Budget Target**: Set RCO budget to **20.5 GiB total (~10.25 GiB per GPU)**. This guarantees full weight residency while reserving ~1.75 GiB per GPU for workspace buffers, activation state, and the KV cache.
* **FP16 `lm_head` Offload/Sharding**: Ensure `lm_head` is sharded as asymmetric per-row INT8 (`quant_gemm_f16_int8w`) or distributed unevenly based on device free memory (`device_free_bytes`).
* **Level-Zero P2P Direct Copy**: Use native Level-Zero IPC handles (`ze_driver`, `ze_device`) to establish direct peer-to-peer (P2P) transfers over PCIe for Tensor-Parallel (TP=2) All-Reduce operations, avoiding CPU-mediated staging.

---

### 2. Token Decode Phase (Latency & Bandwidth Bound)

| Optimization Vector | Implementation Strategy |
| --- | --- |
| **DPAS Hardware Fast-Path** | Enforce `quant_gemm_dpas48` (W4A8/W2A8 sub-byte DPAS) for all linear projections over `quant_gemm_forward_usm` (FP16 dequant fallback). |
| **SYCL Command Graphs** | Capture the full multi-layer decode forward pass into a `sycl::ext::oneapi::experimental::graph` to eliminate per-token CPU-side submission latency. |
| **INT8 KV Cache** | Use `flash_attention_forward_int8_kv` with $G=32/64$ groupwise symmetric scales. This cuts KV memory bandwidth by 50% during decode. |
| **USM Memory Pooling** | Pre-allocate grow-only per-device workspaces (`Workspace` / `Workspace48`) and route all device memory acquisitions through `usm_pool_` to ensure zero Level-Zero allocations in steady state. |

---

### 3. Token Prefill Phase (XMX Compute Bound)

* **Tiled 2D DPAS Kernels**: For $M \ge 32$, launch 2D tiled DPAS GEMM kernels ($64\times 64\times 64$ tile sizes) with Xe-core Sub-Group size 16/32 to maximize XMX systolic tensor engine utilization.
* **Fused Activation Pipeline**: Chain Gate/Up GEMMs directly into `silumul_to_half` on-device kernels before launching Down GEMM. This reduces memory traffic by avoiding host round-trips or intermediate FP32 activation dumps.
* **SSM Recurrence Chaining**: For hybrid linear-attention blocks (e.g., Qwen3.5), dispatch prefill chunks using `ssm_forward_chain` to perform multi-row sequential state updates in shared local memory (SLM) without host synchronization.

---

### 4. Codebase Tuning Execution Checklist

1. **Build Config**: Compile with `-O3 -DINTFER_HAS_L0=1 -DINTFER_PRELOAD=1`.
2. **Quantization Candidates**: Run DASH-Q with $G=16/32$ for INT2 and $G=64$ for INT4 to match Xe2 DPAS vector alignments ($K \% 32 == 0$).
3. **Execution**: Invoke `Engine::preload_all()` at startup to lock weights into USM, and verify via logs that `pool_hits_` is 100% during execution.
