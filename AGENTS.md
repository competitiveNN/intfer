Port ninfer (https://github.com/Neroued/ninfer) to our local setup made of intel arc pro b580 (2x gpus)

Stage 1.
- choose specific model quantization analyze what makes the quantization format chosen by ninfer really fast
- Construct our own int2 quantization based on this schema:
[ FP16 Model ] 
      │
      ▼
[ 1. SpinQuant Preprocessing ]  ──► Rotate W & A to eliminate outliers
      │
      ▼
[ 2. DASH-Q Solver ]            ──► Generate ultra-low error quantized tensor variants
      │
      ▼
[ 3. RCO Bit-Budget Search ]    ──► Allocate DASH-Q variants per tensor under exact VRAM budget
      │
      ▼
[ Standard GGUF / Native SYCL ]

SpinQuant (Orthogonal Rotation Preprocessing): A spatial transformation step. By multiplying weight matrices and activations by learned/randomized orthogonal rotation matrices ($W' = W \cdot R$), activation/weight outliers are suppressed and spread evenly across dimensions.
DASH-Q (Diagonal Hessian & Iterative Least Squares): A non-heuristic PTQ solver. Instead of relying on off-diagonal Hessian error propagation (like GPTQ, which suffers from sampling noise at extremely low bits), DASH-Q uses diagonal Hessian approximations to solve independent weighted least-squares problems for weight assignments.

Stage 2. (tensor parallel since we have 2x gpus)
- Implement the engine (based on oneapi level zero and sycl)
1. Low-Level Native C++/SYCL Kernels (Direct Engine Port)

If you were to adapt ninfer's exact philosophy—bypassing heavy frameworks for lean C++ execution—you would target Intel's hardware primitives directly via oneAPI SYCL and Level Zero:

XMX MMA Intrinsics: Replace NVIDIA mma.sync / wmma PTX calls with Intel DPAS (Dot Product Accumulate Systolic) or XeTile SYCL primitives for INT8/INT4 matrix multiplications.

Kernel Fusion & Memory Layout: Port the fused FlashAttention and quantized GEMM kernels directly to SYCL. Xe matrix engines rely on specific register layouts (e.g., 2D block loads), so layout transposition must match Arc's L1/L2 cache and subgroup sizes (typically 16 or 32 threads per SIMD lane).

SYCL Command Graphs: Replace CUDA Graphs with sycl::ext::oneapi::experimental::command_graph to record token execution sequences and submission queues, keeping CPU launch overhead near zero during generation loops.

| ninfer Technique | Intel Arc Hardware / API Equivalent | Implementation Feasibility |
| :--- | :--- | :--- |
| **Low-Bit Quantization (NVFP4 / W4A4 / INT8)** | **Intel XMX (Xe Matrix Extensions)** & DPAS instructions via SYCL / oneAPI | **Fully Possible:** Intel Arc Alchemist and Battlemage GPUs feature XMX AI engines that natively support INT8, INT4, FP16, and BF16 matrix multiply operations with high throughput. |
| **CUDA Graphs** | **SYCL Graph Extensions** | **Fully Possible:** SYCL supports workload graph recording and submission, eliminating host CPU-to-GPU queue submission overhead during decode loops. |
| **Fused FlashAttention / W4A4 MMA Kernels** | **Intel XeTile / Intel Extension for PyTorch (IPEX)** custom SYCL kernels | **Possible:** Hand-tuned SYCL / Intel Xe-Assembly kernels can match low-level memory layout optimizations for fused operations. |
| **Multi-Token Prediction (MTP) / Speculative Decoding** | **SYCL / Level Zero C++ Runtime** | **Fully Possible:** MTP is an algorithmic framework (running speculative draft tokens parallel to main heads). It is vendor-agnostic and relies purely on batch memory bandwidth. |
| **Groupwise-INT8 KV Cache** | **Custom SYCL Quantization Kernels** | **Fully Possible:** Groupwise INT8 KV cache management reduces VRAM footprint and memory bandwidth usage on Intel Arc GPUs. |

Stage 3.
Implement features for maximizing throughput for concurrent requests, starting from continuous batching.


# guidelines

- remember to commit any meaningful change made such that in case of regressions everything can be bisected.
- remember to not litter /tmp with large tmp files, RAM is precious
