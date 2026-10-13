# Upstream portability synthesis -- llama.cpp ggml-sycl + sgl-kernel-xpu (round 5, closing)

**Date**: 2026-10-13. Scope: the files the previous sessions fetched but never
read against intfer's code (`esimd.hpp`, `mmq.hpp`, PR #25025), plus the
consolidated verdict across all five survey rounds.

## New files read this round

### `ggml/src/ggml-sycl/esimd.hpp` -- NOT PORTABLE (weight-format mismatch; technique already shipped)

ESIMD K-quant dequantize-matvec building blocks for the GGML super-block
formats Q2_K / Q3_K / Q4_K / Q5_K / Q6_K. One skeleton per format: per
super-block, load a 256-float activation slice, load one weight block,
dequantize into 8 chunks of 32, MAC each chunk, reduce, lane-0 epilogue.
The reusable technique is `esimd_reorder_q_traits<T>::mac_pair` -- one
work-group owns a (row0,row1) ROW PAIR and runs two FMA chains interleaved
"to hide FMA latency", with the layout/dequant written once per quant type.

* Weight format: intfer stores its own groupwise INT2/INT4/INT8/FP16
  (`WeightScheme` + `canon_value` codebooks, `src/gemm/quant_gemm.cpp:139`),
  not GGML K-quants. There is no Q4_K/Q6_K block to reorder.
* The technique: intfer's decode1 `ROWPAIR` arm (default OFF,
  `quant_gemm.cpp:602+`) already ships exactly this shape -- one WI owns two
  consecutive output rows, two independent FMA chains, the shared activation
  load amortised over twice the weight traffic. Its comment cites the same
  llama.cpp provenance. So the transferable idea is present, gated, and
  measured (a variance wash per `HISTORY.md`).

### `ggml/src/ggml-sycl/mmq.hpp` -- NOT APPLICABLE (declaration header only)

Declares `ggml_sycl_op_mul_mat_q` -- the GGML quantized matrix-matrix
dispatch entry point. No kernel body, no technique. The bodies live in
`mmq.cpp`/`mmq.py` and target GGML block formats (see above).

### PR #25025 (oneMKL flash attention with GQA batching) -- NOT APPLICABLE (no consumer)

Softmax-attention BLAS batching. intfer's decoder is a gated-delta-net SSM
(`ssm_forward` / `ssm_forward_chain`, `quant_gemm.cpp:3247/3334`); it runs
no softmax attention anywhere, so there is no call site for a fused
attention GEMM. Same verdict class as the MLA/FMHA files in rounds 1-4.

## Consolidated verdict, all rounds

| Round | Source | Files | Verdict |
|-------|--------|-------|---------|
| 1-2 | llama.cpp | `gated_delta_net.cpp`, `gemm.hpp`, `fused-gemm.cpp`, `norm.cpp` | different recurrence; oneDNN wrapper; MX weight formats; host-side l2norm intfer already has |
| 3 | sgl-kernel-xpu | `chunk_gated_delta_rule.cpp/.hpp` | 2-pass fold == shipped `INTFER_SSM_FUSE`; in-kernel l2norm fails the fusion test |
| 4 | sgl-kernel-xpu | six judge-named files (MLA, FMHA, RMSNorm, FusedQKNormRope, dsv3 fused-A, tile scheduler) | Cutlass-tiled ops intfer does not run; JIT elementwise kernels that would ADD an H2D and remove a host kernel that touches no device memory |
| 5 | llama.cpp | `esimd.hpp`, `mmq.hpp`, PR #25025 | GGML K-quant formats (technique already shipped as ROWPAIR); declaration-only header; softmax attention with no consumer |

**The upstream surface is exhausted.** Every candidate fails the PARAMOUNT
DIRECTIVE's fusion test (which round trips does it remove, what is the
launch delta), targets an operator intfer does not run, needs a weight
format intfer does not store, or is a technique intfer already ships. This
is a negative result recorded so the next session does not re-derive it,
not a gap.

## What the negative result points at instead

The remaining optimization surface is INTERNAL, and this session's audit of
it produced one material correction: `INTFER_WAVE_LAUNCH_FUSE`'s
"bit-exact by construction" claim does not survive contact with the base
kernel's actual arithmetic. See
`reports/2026-10-13-gate-and-kernel-audit.md` for the mechanism and the
promotion gate that follows from it.
