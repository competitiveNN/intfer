// src/quant/pipeline.cpp
// t6 — INT2 quantization pipeline: SpinQuant (t3) -> DASH-Q candidates (t4) ->
// RCO bit-budget selection (t5). Pure host C++ (no SYCL). Consumes a ModelIR
// carrying FP16 weights and produces the finalized ModelIR (+ RotationSet) that
// output.hpp serializes to the native SYCL loader / GGUF formats.
//
// Parallelism: DASH-Q per-tensor candidate generation is embarrassingly
// parallel (each tensor's optimization is independent). The pipeline below
// runs that step in parallel using OpenMP; groups within a single quantize()
// are also parallelized (see dashq.cpp). RCO is a single-threaded budget
// search and stays serial.
#include "intfer/pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "intfer/quant/dashq.hpp"
#include "intfer/quant_spec.h"
#include "intfer/spinquant.hpp"

namespace intfer {

PipelineResult run_pipeline(const ModelIR& input, const PipelineConfig& cfg) {
  PipelineResult res;

  // ---- t3: SpinQuant orthogonal rotation preprocessing ----
  SpinQuantResult sqr = spinquant_preprocess(input, cfg.sq, cfg.use_calibration ? &cfg.calib : nullptr);
  res.rotations = sqr.rotations;

  // take ownership of the rotated FP16 model
  ModelIR& m = res.model;
  m = std::move(sqr.model);
  m.model_id = cfg.model_id;
  m.n_layers = cfg.n_layers;

  // unquantized footprint for compression reporting
  for (const auto& t : input.tensors)
    res.fp16_baseline_bytes += (uint64_t)t.N * t.K * 2;

  // ---- t4: DASH-Q candidate quantization (per tensor/role) ----
  // Each tensor is independent (own weight + Hessian + candidate list).
  // Parallelize across tensors: each thread accumulates its own local
  // code/scale/global buffers and its own per-tensor candidates, then
  // we merge into m.code_buf / m.scale_buf / m.global_buf
  // in a single serial pass, fixing offsets along the way.
  const size_t nT = m.tensors.size();
#ifdef _OPENMP
  const int n_threads = omp_get_max_threads();
#else
  const int n_threads = 1;
#endif
  // Thread-local payload buffers (one per thread).
  std::vector<std::vector<uint8_t>>  thr_code(n_threads);
  std::vector<std::vector<uint8_t>>  thr_scale(n_threads);
  std::vector<std::vector<uint8_t>>  thr_global(n_threads);
  // Per-tensor bookkeeping: which thread produced it, and the base offset
  // (in the thread-local buffer) where the FIRST byte of that tensor's
  // candidates live. Each tensor gets a contiguous slice of the chosen
  // thread's buffer.
  std::vector<int>     cand_thread(nT, -1);
  std::vector<uint64_t> cand_base_code(nT, 0);
  std::vector<uint64_t> cand_base_scale(nT, 0);
  std::vector<uint64_t> cand_base_global(nT, 0);

#ifdef _OPENMP
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto& lcode  = thr_code[tid];
    auto& lscale = thr_scale[tid];
    auto& lglobal= thr_global[tid];

    #pragma omp for schedule(dynamic)
    for (intptr_t i = 0; i < (intptr_t)nT; ++i) {
#else
  {
    const int tid = 0;
    auto& lcode  = thr_code[0];
    auto& lscale = thr_scale[0];
    auto& lglobal= thr_global[0];
    for (intptr_t i = 0; i < (intptr_t)nT; ++i) {
#endif
      TensorIR& t = m.tensors[i];
      std::vector<float> W = load_tensor_fp16(m, t);
      if (W.empty()) continue; // tensor has no FP16 payload -> RCO treats as fixed

      // Diagonal Hessian: from calibration activations if provided, else uniform.
      dashq::DiagHessian H = dashq::DiagHessian::uniform(t.K);
      if (cfg.use_calibration) {
        for (const auto& kv : cfg.calib.activations) {
          if (kv.first == t.name && !kv.second.empty()) {
            const uint32_t nTok = (uint32_t)(kv.second.size() / (size_t)t.K);
            H = dashq::DiagHessian::from_activations(kv.second.data(), nTok, t.K);
            break;
          }
        }
      }

      // Record this tensor's base offset in the thread-local buffer BEFORE
      // appending any candidate payload. After the parallel region the
      // merge pass uses (thread, base) to relocate the bytes.
      cand_thread[i] = tid;
      cand_base_code[i]   = (uint64_t)lcode.size();
      cand_base_scale[i]  = (uint64_t)lscale.size();
      cand_base_global[i] = (uint64_t)lglobal.size();

      t.candidates.clear();
      const std::vector<QuantConfig> cfs = default_candidates(t.role);
      for (const auto& base : cfs) {
        if (cfg.skip_int8_above > 0 && base.scheme == QuantScheme::INT8 &&
            (uint64_t)t.N * t.K > cfg.skip_int8_above)
          continue;
        QuantConfig c = base;
        c.rotation_id = t.qcfg.rotation_id; // carry SpinQuant id if t3 ran first
        const dashq::QuantResult r = dashq::quantize(W.data(), t.N, t.K, c, H,
                                              cfg.learn_codebook);

        TensorIR cand;
        cand.name = t.name;
        cand.role = t.role;
        cand.N = t.N;
        cand.K = t.K;
        cand.qcfg = c;
        cand.est_error = r.est_error;
        // Per-candidate offsets are relative to the thread-local buffer
        // (NOT to cand_base_*); the merge pass remaps them to m.buf.
        cand.code_offset  = lcode.size();
        cand.code_bytes   = r.codes.size();
        cand.scale_offset = lscale.size();
        cand.scale_bytes  = r.scales.size();
        cand.global_offset= lglobal.size();
        cand.global_bytes = r.globals.size();
        lcode.insert(lcode.end(), r.codes.begin(), r.codes.end());
        lscale.insert(lscale.end(), r.scales.begin(), r.scales.end());
        lglobal.insert(lglobal.end(), r.globals.begin(), r.globals.end());
        t.candidates.push_back(cand);
      }
#ifndef _OPENMP
    }
    }
#else
    }
  }
#endif

  // ---- Serial merge: copy bytes from each thread-local buffer into the
  // shared m.{code,scale,global}_buf in tensor order, and fix every
  // candidate's offset to be the absolute index in the shared buffer.
  // The (cand_thread[i], cand_base_*[i]) tuple marks the START of tensor i's
  // payload in its thread's local buffer. Per-candidate offsets within that
  // span are shifted by (shared_buf.size() - cand_base_*) at the time of
  // the tensor's first append.
  m.code_buf.clear();
  m.scale_buf.clear();
  m.global_buf.clear();
  for (size_t i = 0; i < nT; ++i) {
    if (cand_thread[i] < 0) continue; // no candidates (empty W)
    const auto& lc = thr_code[cand_thread[i]];
    const auto& ls = thr_scale[cand_thread[i]];
    const auto& lg = thr_global[cand_thread[i]];
    const size_t base_c = m.code_buf.size();
    const size_t base_s = m.scale_buf.size();
    const size_t base_g = m.global_buf.size();
    // Copy this tensor's slice (from cand_base_* to end of its data).
    // We need to know the END of the tensor's slice; that is the max
    // end-offset across its candidates' payload slices. Compute it.
    uint64_t end_c = cand_base_code[i];
    uint64_t end_s = cand_base_scale[i];
    uint64_t end_g = cand_base_global[i];
    for (const auto& cand : m.tensors[i].candidates) {
      end_c = std::max(end_c, (uint64_t)cand.code_offset  + cand.code_bytes);
      end_s = std::max(end_s, (uint64_t)cand.scale_offset + cand.scale_bytes);
      end_g = std::max(end_g, (uint64_t)cand.global_offset+ cand.global_bytes);
    }
    if (end_c > cand_base_code[i])
      m.code_buf.insert(m.code_buf.end(),
                            lc.begin() + cand_base_code[i], lc.begin() + end_c);
    if (end_s > cand_base_scale[i])
      m.scale_buf.insert(m.scale_buf.end(),
                             ls.begin() + cand_base_scale[i], ls.begin() + end_s);
    if (end_g > cand_base_global[i])
      m.global_buf.insert(m.global_buf.end(),
                              lg.begin() + cand_base_global[i], lg.begin() + end_g);
    // Remap per-candidate offsets into the shared buffer:
    //   new_offset = base_X + (old_offset - cand_base_X[i])
    const int64_t d_c = (int64_t)base_c - (int64_t)cand_base_code[i];
    const int64_t d_s = (int64_t)base_s - (int64_t)cand_base_scale[i];
    const int64_t d_g = (int64_t)base_g - (int64_t)cand_base_global[i];
    for (auto& cand : m.tensors[i].candidates) {
      cand.code_offset   = (uint64_t)((int64_t)cand.code_offset   + d_c);
      cand.scale_offset  = (uint64_t)((int64_t)cand.scale_offset  + d_s);
      cand.global_offset = (uint64_t)((int64_t)cand.global_offset + d_g);
    }
  }
  // Free thread-local scratch (they held 10s-100s of MB for big models).
  for (auto& v : thr_code)   { std::vector<uint8_t>().swap(v); }
  for (auto& v : thr_scale)  { std::vector<uint8_t>().swap(v); }
  for (auto& v : thr_global) { std::vector<uint8_t>().swap(v); }

  // ---- t5: RCO bit-budget search (mutates m to single canonical config) ----
  res.rco_report = rco::run_rco(m, cfg.rco);
  if (!res.rco_report.feasible && res.rco_report.total_bytes > cfg.rco.budget_bytes) {
    throw std::runtime_error("RCO produced an infeasible allocation (exceeds VRAM budget)");
  }
  return res;
}

} // namespace intfer
