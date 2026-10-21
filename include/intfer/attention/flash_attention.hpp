#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

// Fused FlashAttention SYCL kernels for Intel Arc (Xe-core / XMX).
//
// *** REACHABILITY (traced 2026-10-10, HISTORY.md "GQA-packing
// DECLINED with evidence"): these two kernels are DEMO/REFERENCE-
// ONLY. They are compiled into the intfer_attention static library
// (src/CMakeLists.txt:30) and linked into intfer_gemm, but they are
// NOT reachable from the inference path. Their only call sites are
// examples/flash_attention_demo/ and examples/qwen_forward_gpu/ (a
// reference/diagnostic forward). src/engine/engine.cpp:32 is a bare
// #include with ZERO calls to either symbol, and examples/intfer_cli/
// main.cpp references flash_attention_forward exactly once -- in a
// comment (:3831). The engine that runs inference therefore has NO
// device softmax-attention kernel on any path: the 27B decodes
// through the GatedDeltaNet SSM recurrence (host fallback,
// main.cpp:5180-5210) and the is_full softmax decode is host-only
// scalar (main.cpp:5580-5622). Do not mistake these kernels for a
// reachable decode path -- a device paged-KV softmax-decode kernel
// does not exist in this tree, so GQA head-group packing (vllm-xpu
// decode_policy_qpacked_head) has nothing to attach to.
//
// Two forward-pass variants are provided:
//   1. flash_attention_forward_f16  - FP16 Q/K/V with online-softmax tiling
//                                     (classic FlashAttention-2).
//   2. flash_attention_forward_int8_kv - FP16 Q but K/V stored as groupwise
//                                     INT8 (per-token-group symmetric scales),
//                                     dequantized on the fly inside the kernel.
//                                     This mirrors the dequant-on-the-fly path
//                                     of the quantized GEMM kernels and backs
//                                     the groupwise-INT8 KV cache layout.
//
// Both are row-tiled over the query sequence (Br query rows per work-group),
// iterate over key blocks (Bc) with a running max/sum, and keep the
// per-query accumulator in registers. Tile sizes are chosen for Arc L1/L2 and
// a 16/32 subgroup size (Br=Bc=64 -> 32 KB SLM for K+V at head_dim<=128).

namespace intfer::attention {

struct FlashAttentionParams {
  int batch = 0;
  int heads = 0;
  int seq_q = 0;
  int seq_k = 0;
  int dim = 0;        // head dimension
  float scale = 0.0f; // softmax scale; default 1/sqrt(dim)
  bool causal = true;
  int q_offset = 0;   // absolute position of query[0]; causal masks tk > qi+q_offset.
                       // For decode (seq_q==1) this is the current token position.
};

// (1) FP16 fused causal/non-causal self-attention forward.
// q,k,v,out are row-major [batch, heads, seq, dim] of sycl::half bits.
void flash_attention_forward_f16(sycl::queue& q,
                                 sycl::buffer<std::uint16_t>& q_buf,
                                 sycl::buffer<std::uint16_t>& k_buf,
                                 sycl::buffer<std::uint16_t>& v_buf,
                                 sycl::buffer<std::uint16_t>& out_buf,
                                 const FlashAttentionParams& p);

// (2) Groupwise-INT8 KV-cache fused attention forward.
//   k,v  : INT8 row-major [batch, heads, seq_k, dim]
//   k_scale, v_scale : FP32 row-major [batch, heads, num_groups],
//                      where num_groups = ceil(seq_k / group_size) and a token
//                      t uses scale index t / group_size. Dequantized as
//                      val = int8 * scale. Q remains FP16.
void flash_attention_forward_int8_kv(sycl::queue& q,
                                     sycl::buffer<std::uint16_t>& q_buf,
                                     sycl::buffer<std::int8_t>& k_buf,
                                     sycl::buffer<std::int8_t>& v_buf,
                                     sycl::buffer<float>& k_scale_buf,
                                     sycl::buffer<float>& v_scale_buf,
                                     sycl::buffer<std::uint16_t>& out_buf,
                                     const FlashAttentionParams& p,
                                     int group_size);

// Host FP32 reference for numerical validation (same layout as above, FP32).
void flash_attention_reference_f32(const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  std::vector<float>& out,
                                  const FlashAttentionParams& p);

}  // namespace intfer::attention
