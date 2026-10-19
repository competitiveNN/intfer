// src/quant_spec.cpp
// Definitions for the Stage-1 Tensor-IR contract (docs/quant_spec.md).
//
// THIS FILE AND src/quant/quant_spec.cpp BOTH DEFINE default_candidates(), AND
// THEY DISAGREE. Read this before editing either.
//
//   this file      compiled by scripts/build_quant_real.sh and by
//                  src/CMakeLists.txt's intfer_quant_core -> THIS IS THE ONE THAT
//                  PRODUCED Qwen3.8-27B-int2.intifer. LINEAR_ATTN_* now carries
//                  its own ladder arms below; the MEASURED block at the bottom is
//                  the record of the Artifact it produced when it had none, and
//                  is deliberately kept as history rather than deleted.
//   quant/ this    compiled by six examples/* targets (dashq_demo, quant_pipeline,
//     copy         rco_demo, quant_merge, intfer_fidelity, intfer_gguf). MIRRORS
//                  those arms as of 2026-10-18.
//
// They cannot be linked together (duplicate symbols), so each target picks one
// and any divergence between them is silent. The divergence ran in BOTH
// directions and both were wrong: originally the examples offered INT2/INT4/INT8
// to LINEAR_ATTN_* where this file offered only FP16; with the arms in place the
// examples must now DROP the FP16 escape this file deliberately withholds, drop
// the INT2 offered to the two tiny roles, and pick up the CONV arm.
//
// Editing either: port the DECISION, never the call-site text. The two mk()
// helpers take their 3rd/4th parameters in the OPPOSITE order -- (sch, g, act,
// glob) here versus (sch, g, glob, act) there -- so pasting a push_back between
// the files writes an ActScheme into a bool without a compiler error.
//
// MEASURED, not inferred. scripts/quant_precision_map.sh reads the shipped
// artifact's own metadata header:
//   LINEAR_ATTN_QKV 48 tensors  4.688 GiB  25.8% of bytes   all FP16
//   LINEAR_ATTN_Z   48 tensors  2.812 GiB  15.5% of bytes   all FP16
//   LINEAR_ATTN_OUT 48 tensors  2.812 GiB  15.5% of bytes   all FP16
// i.e. 56.8% of the model's bytes are FP16 because of the `default:` arm below,
// not because the budget search chose to spend them. 21.0% of parameters, 57.9%
// of bytes, 256 of 498 tensors.
//
// *** FIXED 2026-10-18: LINEAR_ATTN_* now has its own ladder arm (below). ***
// The engine side needed NO change, and that was verified against the source
// rather than assumed: Engine::shard_weight (src/engine/engine.cpp:290-349) is
// ROLE-AGNOSTIC -- it reads t.qcfg.scheme / t.qcfg.group_size / .has_global_
// divisor and branches only on the SCHEME, carrying an explicit fp16_passthrough
// arm (:340) alongside the dequant packing path. Role appears nowhere in it, so
// a LINEAR_ATTN_* tensor that arrives INT2 is sharded and GEMM-dispatched by
// exactly the same code an MLP_GATE INT2 tensor uses, including the
// run_linear_wave path these roles reach (engine.cpp:2800).
//
// What is NOT done, and must not be reported as done: the re-quantization
// itself. It cannot run on this box today because the source checkpoint
// /var/home/fra/data/models/safetensors/Qwen3.8-27B/ is GONE (the disk was full
// and it was evicted -- scripts/archive/run_27b_int2_chunks.sh records
// "Chunks go to /data scratch (home disk is full)"). Every remaining Qwen3.8-27B
// checkpoint on disk is already quantized (NVFP4 21.5 GB, GLA-g2-INT4, AWQ-INT4,
// Escha-W2 9.7 GB), so a re-quant from any of them is a double quantization.
// The ladder fix is the enabling change; the run is the operator's.
//
// The two ladders still differ in spelling (this file uses explicit case blocks
// with per-block `return v`, the quant/ copy uses fall-through push_back and a
// shared `mk` default). They cannot be linked together (duplicate symbols), so
// each target picks one and any new divergence between them is silent. Adding a
// rung HERE therefore has to be mirrored there or the six examples/* targets will
// report a scheme distribution the real quantizer cannot produce -- the exact
// defect this header exists to prevent.
#include "intfer/quant_spec.h"

#include <cmath>

namespace intfer {

float TensorIR::eff_bits_per_weight() const {
  return effective_bits(qcfg.scheme, qcfg.group_size, qcfg.scale_dtype);
}

uint64_t ModelIR::total_weight_bytes() const {
  uint64_t s = 0;
  for (const auto& t : tensors) s += t.weight_bytes();
  return s;
}

float effective_bits(QuantScheme scheme, uint16_t group_size, ScaleDtype sd) {
  // code bits + scale overhead (docs/quant_spec.md §3.3).
  float code_bits = 0.0f;
  switch (scheme) {
    case QuantScheme::FP16: code_bits = 16.0f; break;
    case QuantScheme::INT8: code_bits = 8.0f;  break;
    case QuantScheme::INT4: code_bits = 4.0f;  break;
    case QuantScheme::INT2: code_bits = 2.0f;  break;
  }
  if (scheme == QuantScheme::FP16) return 16.0f;  // scales are never stored for FP16

  // scale bits per weight = scale_bits / group_size.
  float scale_bits = (sd == ScaleDtype::FP8_E4M3) ? 8.0f : 16.0f;
  float g = static_cast<float>(group_size > 0 ? group_size : 1);
  return code_bits + scale_bits / g;
}

std::vector<QuantConfig> default_candidates(TensorRole role) {
  // Per docs/quant_spec.md §5 bit-allocation policy.
  auto mk = [](QuantScheme s, uint16_t g, ActScheme act,
               bool global = true, ScaleDtype sd = ScaleDtype::FP16) {
    QuantConfig c;
    c.scheme = s;
    c.group_size = g;
    c.scale_dtype = sd;
    c.has_global_divisor = global;
    c.act = act;
    return c;
  };
  const ActScheme a8 = ActScheme::INT8_G64;
  switch (role) {
    case TensorRole::EMBED:
    case TensorRole::LM_HEAD: {
      // Floor >= INT4 (sensitive). Offer INT4/INT8/FP16.
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
    case TensorRole::ATTN_Q:
    case TensorRole::ATTN_K:
    case TensorRole::ATTN_V: {
      // first layer forced >= INT4 by RCO; candidates include INT2/INT4/INT8/FP16.
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT2, 16, a8));
      v.push_back(mk(QuantScheme::INT2, 32, a8));
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
    case TensorRole::ATTN_O:
    case TensorRole::MLP_GATE:
    case TensorRole::MLP_UP:
    case TensorRole::MLP_DOWN: {
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT2, 16, a8));
      v.push_back(mk(QuantScheme::INT2, 32, a8));
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      return v;
    }
    // LINEAR_ATTN_* (the 48 gated-delta-net layers of the 27B). These seven
    // roles used to fall through to the `default:` arm below and received FP16
    // as their ONLY candidate, so RCO never had a choice to make for them:
    // 10.3 of the artifact's 18.2 GiB was FP16 by ROLE OMISSION, not by budget
    // (measured: LINEAR_ATTN_QKV 4.688 GiB, _Z 2.812 GiB, _OUT 2.812 GiB,
    // 240 of 498 tensors = 56.8% of bytes at 20.6% of params; _A and _B add
    // another 96 tensors at 0.2% of bytes, so the WHOLE-MODEL FP16 census is
    // 256 tensors / 5.644 G params = 21.0% of params / 57.9% of bytes -- the
    // other 16 are ATTN_K, which is tiny-N and FP16 by budget, not by omission).
    // That omission is also what made the chunked 27B re-quant report "RCO produced an
    // infeasible allocation": the cheapest allocation for a chunk still had to
    // pay those FP16 tensors in full, so it could not fit its per-chunk
    // budget and the run failed on every LIN+LIN chunk.
    //
    // Ladder is chosen for the three tensors that carry the bytes; the two
    // tiny ones are held at an INT4 floor.
    //
    // XMX/TP=2 SHAPE CONTRACT (why these exact group sizes). The XMX joint_
    // matrix path (quant_gemm.cpp:2216) is TM=TN=TK=16, SG=16, BK=32, and its
    // caller guard is M%16==0, N%16==0, K%32==0. Under --tp 2 the engine splits
    // the OUTPUT dim N (column-parallel, engine.cpp run_linear_tp), so each
    // device holds N/2 rows and the guard must hold on the SLICE, not just on
    // the tensor:
    //   QKV  N=10240 -> 5120/device   Z N=6144 -> 3072   OUT N=5120 -> 2560
    // all %16==0, and K (never split for a column-parallel row) is
    // 5120 / 6144, both %32==0. So every one of these tensors is XMX-eligible
    // at TP=1 AND TP=2 with no repack.
    //
    // Group sizes are BK-aligned on purpose: the XMX kernel dequantizes one
    // BK=32 SLM tile per joint_matrix step and reads ONE scale per group, so
    // g32 = exactly one group per tile (the scale is loaded once and reused
    // for both TK=16 steps) while g16 doubles the scale traffic inside the
    // dequant and g64 spans two tiles. g32 is therefore the Xmx-optimal rung
    // of the ladder AND 0.5 bit/weight cheaper than g16, which is why it is
    // offered first: RCO's cost is real bytes (rco.cpp), so the accuracy
    // search keeps g16 only where its error actually earns the extra 0.5 bpw.
    case TensorRole::LINEAR_ATTN_QKV:
    case TensorRole::LINEAR_ATTN_Z:
    case TensorRole::LINEAR_ATTN_OUT: {
      // No FP16 rung: offering one is what let the budget search hand the
      // whole tensor back un-quantized, and these three roles are 56.8% of the
      // shipped model's bytes. They are the shrink.
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT2, 32, a8));  // BK-aligned: 1 group/tile
      v.push_back(mk(QuantScheme::INT2, 16, a8));  // accurate rung, +0.5 bpw
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      return v;
    }
    case TensorRole::LINEAR_ATTN_A:
    case TensorRole::LINEAR_ATTN_B: {
      // N = n_v_heads ~48 and unrotated scalars: too small for INT2 to pay
      // (48 codes per row, scale row = 5120/32 = 160 FP16 = 320 B against a
      // 96-byte INT2 payload), and precision-critical gate inputs. INT4 floor
      // only -- the same policy src/quant/quant_spec.cpp documents.
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::INT4, 64, a8));
      v.push_back(mk(QuantScheme::INT8, 32, a8));
      return v;
    }
    case TensorRole::LINEAR_ATTN_CONV: {
      // conv1d (conv_k, conv_dim) is not a Linear; stays elementwise-exact.
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
    case TensorRole::NORM:
    case TensorRole::ROPE:
    case TensorRole::OTHER:
    default: {
      std::vector<QuantConfig> v;
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
  }
}

} // namespace intfer
