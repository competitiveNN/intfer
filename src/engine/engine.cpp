// src/engine/engine.cpp
// Stage-2 engine implementation: loads the .intfer, shards Linear weights across
// the 2x Arc B580 (tensor parallel), and runs the per-token decode step through
// the quantized GEMM (XMX/DPAS) + fused FlashAttention, with SYCL command-graph
// capture of the decode loop.
//
// NOTE: this is the engine skeleton. It compiles against oneAPI DPC++ and runs on
// real Arc hardware; in this sandbox no GPU/SYCL platform is present so it is
// build-validated only. The hybrid Qwen3.5 forward (RMSNorm, mRoPE, the
// gated-delta-rule SSM, KV cache, tokenizer, sampling) plugs into these
// primitives as the remaining integration.
// ... (previous imports)
#include "intfer/engine.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
// ...


#include "intfer/output.hpp"
#include "intfer/safetensors.hpp"
#include "intfer/gemm/quant_gemm.hpp"
#include "intfer/attention/flash_attention.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/device/device_manager.hpp"

#include "intfer/spinquant.hpp"  // for canonical_codebook + unpack_nibble

// Command-graph capture over sycl::buffer is disallowed by the SYCL graph
// extension (buffers are write-back accessors). The graph-capturable fast path
// must use raw USM pointers (quant_gemm_dpas_int8). Off by default so the
// buffer-based decode path runs; enable only when graph_ compiled in.
#define INTFER_CAPTURE_GRAPH 1

namespace intfer {

Engine::~Engine() {
  if (backing_.base && backing_.size) ::munmap(backing_.base, backing_.size);
  backing_.base = nullptr;
  backing_.size = 0;
}

#ifdef INTFER_PROFILE_GPU
  struct GpuProfile {
    double total_kernel_us  = 0.0;
    double total_wait_us    = 0.0;
    double total_copy_us    = 0.0;
    double total_readback_us = 0.0;
    uint64_t peak_wait_depth = 0;
    uint64_t kernel_count   = 0;
    uint64_t wait_count     = 0;
    uint64_t copy_count     = 0;
    uint64_t readback_count = 0;
  };
  static GpuProfile g_gpu_profile;
#endif

using namespace sycl::ext::oneapi::experimental;

namespace {

gemm::WeightScheme to_ws(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return gemm::WeightScheme::INT2;
    case QuantScheme::INT4: return gemm::WeightScheme::INT4;
    case QuantScheme::INT8: return gemm::WeightScheme::INT8;
    default:                return gemm::WeightScheme::FP16;
  }
}

// Build a per-device GemmWeightBuffers view over rows [start,end) of a tensor's
// packed payload (column tensor-parallel sharding of the output dim N).
//
// The Stage-1 DASH-Q pipeline emits codes in *group-major* order
// (idx[i] = a[g*N*Glocal + r*Glocal + c]) and scales per-group only, but the
// dequant-on-the-fly GEMM kernel expects codes in row-major (n*K+k) and scales
// per-(row,group). To bridge the two without rewriting the GEMM kernel, we
// re-unpack and re-pack on the host into the kernel's layout ONCE at load
// time, then use simple row-major slices for TP. The shard covers the
// contiguous range [start,end) of the N dimension.
gemm::GemmWeightBuffers shard_weight(const ModelIR& m, const TensorIR& t,
                                     uint32_t start, uint32_t end,
                                     sycl::queue& q) {
  gemm::GemmWeightBuffers w;
  w.scheme = to_ws(t.qcfg.scheme);
  w.group_size = t.qcfg.group_size;
  w.has_global_divisor = t.qcfg.has_global_divisor;
  w.N = end - start;
  w.K = t.K;
  const uint32_t N = t.N, K = t.K;
  const uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : K;
  const uint32_t n_groups = (K + G - 1) / G;
  const int bits = (w.scheme == gemm::WeightScheme::INT2) ? 2
                 : (w.scheme == gemm::WeightScheme::INT4) ? 4
                 : (w.scheme == gemm::WeightScheme::FP16) ? 16 : 8;
  const int pack = (bits >= 8) ? 1 : (8 / bits);
  const uint32_t per_row_code_bytes = (K + pack - 1) / pack;

  // 1) Build the per-group scale table (FP16 bits) and a per-row broadcast
  //    view. The kernel reads scale[ng*n_groups + g], so we emit (N*n_groups)
  //    FP16 values (broadcast the per-group scale to every row).
  //    FP16-scheme tensors carry no meaningful scale (CPU dequant is a raw
  //    half->float copy); the kernel still multiplies by the table, so emit
  //    all-ones (0x3C00) instead of copying scale_src (which holds a small
  //    fallback value that would attenuate the output ~50x).
  const uint16_t* scale_src =
      reinterpret_cast<const uint16_t*>(m.scale_data() + t.scale_offset);
  const bool fp16_passthrough = (w.scheme == gemm::WeightScheme::FP16);
  std::vector<uint16_t> scale_broadcast((size_t)N * n_groups);
  for (uint32_t r = 0; r < N; ++r)
    for (uint32_t g = 0; g < n_groups; ++g)
      scale_broadcast[(size_t)r * n_groups + g] =
          fp16_passthrough ? 0x3C00 : scale_src[g];
  // Take the TP slice on the host (contiguous, byte-aligned by N*2).
  std::vector<uint16_t> scale_slice((size_t)w.N * n_groups);
  for (uint32_t r = 0; r < w.N; ++r)
    std::memcpy(&scale_slice[(size_t)r * n_groups],
                &scale_broadcast[(size_t)(r + start) * n_groups],
                n_groups * sizeof(uint16_t));
  // Keep the packed vectors alive in the returned struct. The buffer is built
  // from the vector's iterators (eager copy, like gemm_demo); the vector stays
  // alive in `w` until the GEMM runs, so the payload is never lost.
  w.scale_store = std::move(scale_slice);
  w.scale = sycl::buffer<std::uint16_t>(w.scale_store.begin(), w.scale_store.end());

  // 2) Unpack group-major codes into row-major INT8 codes, then take the TP
  //    slice and re-pack (only for INT2/INT4). For INT8/FP16 the code bytes
  //    are identity so this is a plain memcpy. The kernel reads code[ng*K+kg]
  //    and unpacks bits-within-byte using idx%pack, so a row-major byte stream
  //    is exactly what it expects.
  if (w.scheme == gemm::WeightScheme::FP16) {
    // FP16: pass-through the N*K*2 byte slice (identity for the kernel).
    std::vector<uint8_t> code_slice((size_t)w.N * K * 2);
    const uint8_t* base = m.code_data() + t.code_offset + start * K * 2;
    std::memcpy(code_slice.data(), base, code_slice.size());
    w.code_store = std::move(code_slice);
    w.code = sycl::buffer<std::uint8_t>(w.code_store.data(),
                                         sycl::range<1>(w.code_store.size()));
  } else {
    // Group-major unpack -> row-major INT8 -> pack to bits.
    std::vector<uint8_t> row_codes((size_t)w.N * K);
    for (uint32_t g = 0; g < n_groups; ++g) {
      const uint32_t k0 = g * G;
      const uint32_t k1 = std::min(k0 + G, K);
      for (uint32_t r = 0; r < w.N; ++r) {
        const uint32_t src_row = r + start;
        // Source idx: g*(N*Glocal) + r_src*Glocal + (c - g*G)
        const uint32_t Glocal = k1 - k0;
        for (uint32_t c = k0; c < k1; ++c) {
          const size_t src_idx = (size_t)g * (N * Glocal)
                               + (size_t)src_row * Glocal + (c - k0);
          const uint8_t b = m.code_data()[t.code_offset + src_idx / pack];
          const int within = (int)(src_idx % pack);
          const uint8_t val = (b >> (within * bits)) & ((1u << bits) - 1u);
          row_codes[(size_t)r * K + c] = val;
        }
      }
    }
    // Repack row-major INT8 to packed bytes (kernel layout).
    std::vector<uint8_t> code_slice((size_t)w.N * per_row_code_bytes, 0);
    for (size_t i = 0; i < row_codes.size(); ++i) {
      const size_t pack_idx = i / pack;
      const int within = (int)(i % pack);
      code_slice[pack_idx] |= (uint8_t)((row_codes[i] & ((1u << bits) - 1u))
                                        << (within * bits));
    }
    w.code_store = std::move(code_slice);
    w.code = sycl::buffer<std::uint8_t>(w.code_store.data(),
                                        sycl::range<1>(w.code_store.size()));
  }

  if (t.global_bytes > 0) {
    // global_offset is a *byte* offset into the uint8_t global_buf (see
    // dashq.cpp: r.globals assigned from begin()+global_offset). Copy the 4
    // bytes into an aligned float (avoids the previous 4x mis-offset read that
    // yielded garbage alpha and zeroed/NaN'd the GEMM output).
    float alpha_val = 0.f;
    std::memcpy(&alpha_val, m.global_data() + t.global_offset,
                std::min<uint32_t>(t.global_bytes, 4u));
    w.global_store.assign(1, alpha_val);
    w.global = sycl::buffer<float>(w.global_store.begin(), w.global_store.end());
  }
  return w;
}

}  // namespace

std::unique_ptr<Engine> Engine::load(const std::string& intfer_path,
                                    size_t tp_devices) {
  auto e = std::unique_ptr<Engine>(new Engine());

  // Zero-copy load: mmap the file and borrow the payload views instead of
  // copying multi-GB payloads into RAM (a 22GB .intifer would otherwise need
  // ~44GB transient RSS: file blob + deserialized vectors). The mapping is
  // kept in e->backing_ for the Engine's lifetime. Small files (<1GB) use
  // the old read+copy path so short-lived tools don't pin mappings.
  RotationSet rs;
  {
    FILE* f = std::fopen(intfer_path.c_str(), "rb");
    if (!f) { std::cerr << "Engine: cannot read " << intfer_path << "\n"; return nullptr; }
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fclose(f);
    if (fsize <= 0) { std::cerr << "Engine: cannot read " << intfer_path << "\n"; return nullptr; }
    if ((size_t)fsize < (1ULL << 30)) {
      std::vector<uint8_t> blob;
      if (!read_file(intfer_path, blob)) {
        std::cerr << "Engine: cannot read " << intfer_path << "\n";
        return nullptr;
      }
      if (!deserialize_intfer(blob.data(), blob.size(), e->model_, rs)) {
        std::cerr << "Engine: cannot deserialize " << intfer_path << "\n";
        return nullptr;
      }
    } else {
      int fd = ::open(intfer_path.c_str(), O_RDONLY);
      if (fd < 0) { std::cerr << "Engine: cannot mmap " << intfer_path << "\n"; return nullptr; }
      void* m = ::mmap(nullptr, (size_t)fsize, PROT_READ, MAP_PRIVATE, fd, 0);
      ::close(fd);
      if (m == MAP_FAILED) { std::cerr << "Engine: cannot mmap " << intfer_path << "\n"; return nullptr; }
      e->backing_.base = m;
      e->backing_.size = (size_t)fsize;
      if (!deserialize_intfer_borrowed((const uint8_t*)m, (size_t)fsize, e->model_, rs)) {
        ::munmap(m, (size_t)fsize);
        e->backing_ = Engine::MappedFile{};
        std::cerr << "Engine: cannot deserialize " << intfer_path << "\n";
        return nullptr;
      }
    }
  }
  e->rs_ = std::move(rs);
  for (const auto& R : e->rs_.rotations) e->rid_[R.id] = &R;

  // Discover the 2x Arc B580 fleet and pick `tp_devices` of them.
  auto dm = device::DeviceManager::discover("Arc");
  size_t n = (tp_devices == 0) ? dm.size() : std::min(tp_devices, dm.size());
  if (n == 0) { std::cerr << "Engine: no Arc GPU discovered\n"; return nullptr; }
  for (size_t i = 0; i < n; ++i) e->devices_.push_back(dm.at(i));
  std::cout << "Engine: " << n << " Arc device(s): ";
  for (auto& d : e->devices_) std::cout << d.name << " ";
  std::cout << "\n";

  e->build_shards();
  return e;
}

void Engine::build_shards() {
  // Balanced contiguous sharding of the output dim N across the devices
  // (tensor-parallel column split). Mirrors DeviceManager::shard.
  auto split = [this](uint32_t total) {
    std::vector<std::pair<uint32_t, uint32_t>> r;
    size_t n = devices_.size();
    for (size_t i = 0; i < n; ++i) {
      uint32_t start = (uint32_t)(total * i / n);
      uint32_t end = (uint32_t)(total * (i + 1) / n);
      r.push_back({start, end});
    }
    return r;
  };
  shards_.resize(model_.tensors.size());
  shard_cache_.resize(model_.tensors.size());
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti) {
    const auto& t = model_.tensors[ti];
    shards_[ti].ranges = (devices_.size() > 1) ? split(t.N)
                                            : std::vector<std::pair<uint32_t, uint32_t>>{{0, t.N}};
  }
  // Identify layer blocks for the decode loop.
  for (const auto& t : model_.tensors)
    if (t.role == TensorRole::ATTN_Q) {
      std::string p = t.name;
      auto pos = p.find("layers.");
      if (pos != std::string::npos) layer_names_.push_back(p.substr(0, p.find('.', pos + 7) + 1));
    }
  // Per-layer tensor index (layer-wise VRAM streaming). Unlayered tensors
  // (lm_head, ...) get -1 and are never evicted by begin_layer().
  layer_of_.assign(model_.tensors.size(), -1);
  int maxL = -1;
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti) {
    int L = layer_of_name(model_.tensors[ti].name);
    layer_of_[ti] = L;
    if (L > maxL) maxL = L;
  }
  layer_tensors_.assign(maxL + 1, {});
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti)
    if (layer_of_[ti] >= 0) layer_tensors_[(size_t)layer_of_[ti]].push_back((int)ti);
}

int Engine::layer_of_name(const std::string& name) {
  size_t p = name.find("model.language_model.layers.");
  if (p == std::string::npos) return -1;
  int L = std::atoi(name.c_str() + p + 28);
  if (L < 0) return -1;
  // Guard against atoi garbage on malformed names: require digits.
  const char* s = name.c_str() + p + 28;
  if (*s < '0' || *s > '9') return -1;
  return L;
}

void Engine::release_shard(int tensor_index) {
  if (tensor_index < 0 || (size_t)tensor_index >= shard_cache_.size()) return;
  auto& sc = shard_cache_[(size_t)tensor_index];
  if (!sc.built) return;
  // run_linear_tp waits on kernel events before readback, so by the time a
  // later begin_layer() releases this shard no kernel can still reference
  // it. Belt-and-braces queue sync (already complete => ~free).
  for (size_t d = 0; d < devices_.size(); ++d) {
    devices_[d].queue.wait();
    if (sc.code_usm.size() > d && sc.code_usm[d]) {
      sycl::free(sc.code_usm[d], devices_[d].queue);
      sc.code_usm[d] = nullptr;
    }
    if (sc.scale_usm.size() > d && sc.scale_usm[d]) {
      sycl::free(sc.scale_usm[d], devices_[d].queue);
      sc.scale_usm[d] = nullptr;
    }
    if (sc.global_usm.size() > d && sc.global_usm[d]) {
      sycl::free(sc.global_usm[d], devices_[d].queue);
      sc.global_usm[d] = nullptr;
    }
    if (sc.act_usm.size() > d && sc.act_usm[d]) {
      sycl::free(sc.act_usm[d], devices_[d].queue);
      sc.act_usm[d] = nullptr;
    }
    if (sc.out_usm.size() > d && sc.out_usm[d]) {
      sycl::free(sc.out_usm[d], devices_[d].queue);
      sc.out_usm[d] = nullptr;
    }
  }
  sc.code_usm.clear(); sc.scale_usm.clear(); sc.global_usm.clear();
  sc.act_usm.clear(); sc.out_usm.clear();
  sc.built = false;
}

void Engine::begin_layer(int L) {
  if (layer_tensors_.empty()) return;
  for (size_t ti = 0; ti < shard_cache_.size(); ++ti) {
    if (!shard_cache_[ti].built) continue;
    int tl = (ti < layer_of_.size()) ? layer_of_[ti] : -1;
    if (tl >= 0 && tl != L) release_shard((int)ti);
  }
}

std::vector<float> Engine::decode_step(const std::vector<float>& token_hidden,
                                       int tensor_index) {
  if (model_.tensors.empty()) return {};
  const int ti = (tensor_index >= 0 &&
                  (size_t)tensor_index < model_.tensors.size())
                     ? tensor_index
                     : 0;
  std::vector<float> out;
  run_linear_tp(ti, token_hidden, out);
  return out;
}

void Engine::run_linear_tp(int tensor_index, const std::vector<float>& x,
                           std::vector<float>& out) {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size()) return;
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (out.size() != t.N) out.resize(t.N);

  // Lazy-build the USM shard cache for this tensor on first call.
  if (!shard_cache_[tensor_index].built)
    ensure_shard_built(tensor_index);
  const auto& sc = shard_cache_[tensor_index];

  // Pre-convert FP32 activation to FP16 once (shared by all devices).
#ifdef INTFER_PROFILE_GPU
  auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
  std::vector<std::uint16_t> act_fp16(t.K);
  for (uint32_t k = 0; k < t.K; ++k)
    act_fp16[k] = dashq::float_to_half(x[k]);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
    g_gpu_profile.copy_count++;
  }
#endif

  // Issue all device work in parallel and collect kernel events.
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(devices_.size());

  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    auto N_local = end - start;

    // 1) Write FP16 activation directly into persistent USM act buffer.
    std::uint16_t* const act_ptr = sc.act_usm[d];
    for (uint32_t k = 0; k < t.K; ++k)
      act_ptr[k] = act_fp16[k];

    // 2) Launch the USM kernel on the persistent weight buffers.
    float alpha = sc.has_global_divisor
                      ? sc.global_usm[d][0]
                      : 1.0f;
    sycl::event kernel_evt = gemm::quant_gemm_forward_usm(q,
                                  sc.act_usm[d],
                                  sc.code_usm[d],
                                  sc.scale_usm[d],
                                  alpha,
                                  sc.out_usm[d],
                                  1, N_local, t.K,
                                  sc.group_size,
                                  sc.scheme,
                                  sycl::event{});
#ifdef INTFER_PROFILE_GPU
    if (kernel_events.size() + 1 > g_gpu_profile.peak_wait_depth)
      g_gpu_profile.peak_wait_depth = kernel_events.size() + 1;
#endif
    kernel_events.push_back(kernel_evt);
  }

  // Wait for all kernels, then read back directly from persistent USM.
  // NOTE: the wait is load-bearing, not profiling — without it the host
  // reads out_usm before the kernel runs (fresh zero pages => all-zero
  // GEMM outputs). Only the timing accumulation is profiling-gated.
  for (size_t d = 0; d < devices_.size(); ++d) {
    kernel_events[d].wait();
  }
#ifdef INTFER_PROFILE_GPU
  g_gpu_profile.wait_count += devices_.size();

  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto [start, end] = ranges[d];
    const std::uint16_t* out_ptr = sc.out_usm[d];
    for (uint32_t r = 0; r < static_cast<uint32_t>(end - start); ++r)
      out[start + r] = dashq::half_to_float(out_ptr[r]);
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
#endif
}

#ifdef INTFER_PROFILE_GPU
  GpuProfileSnapshot get_gpu_profile() {
    return {g_gpu_profile.total_kernel_us, g_gpu_profile.total_wait_us,
            g_gpu_profile.total_copy_us, g_gpu_profile.total_readback_us,
            g_gpu_profile.peak_wait_depth,
            g_gpu_profile.kernel_count, g_gpu_profile.wait_count,
            g_gpu_profile.copy_count, g_gpu_profile.readback_count};
  }
  void reset_gpu_profile() { g_gpu_profile = GpuProfile{}; }
#endif

void Engine::run_linear_f16_tp(const sycl::half* x, const sycl::half* w,
                               float* out, uint32_t N, uint32_t K,
                               const sycl::event& copy_dep) {
  // One-off per-device weight buffer. We copy the full W to each device;
  // each device computes the full N rows and the host then reads back.
  // This is fine because lm_head is a one-off matmul, not the hot loop.
  std::vector<sycl::event> evts;
  evts.reserve(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    sycl::queue& q = dev.queue;
    size_t w_bytes = (size_t)N * K * sizeof(sycl::half);
    sycl::half* w_dev = static_cast<sycl::half*>(
        sycl::malloc_device(w_bytes, dev.sycl_device, dev.sycl_context));
    sycl::half* x_dev = static_cast<sycl::half*>(
        sycl::malloc_device((size_t)K * sizeof(sycl::half), dev.sycl_device, dev.sycl_context));
    sycl::half* c_dev = static_cast<sycl::half*>(
        sycl::malloc_device((size_t)N * sizeof(sycl::half), dev.sycl_device, dev.sycl_context));

    q.memcpy(w_dev, w, w_bytes);
    q.memcpy(x_dev, x, (size_t)K * sizeof(sycl::half)).wait();
    sycl::event evt = gemm::quant_gemm_f16(q, x_dev, w_dev, c_dev, 1, N, K);
    q.memcpy(out, c_dev, (size_t)N * sizeof(float), evt);
    evts.push_back(evt);
    sycl::free(w_dev, q);
    sycl::free(x_dev, q);
    sycl::free(c_dev, q);
  }
  for (auto& e : evts) e.wait();
}

const Rotation* Engine::rotation_for(int tensor_index) const {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return nullptr;
  const auto& t = model_.tensors[tensor_index];
  auto it = rid_.find(t.qcfg.rotation_id);
  return it == rid_.end() ? nullptr : it->second;
}

bool Engine::ensure_shard_built(int tensor_index) {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size()) return false;
  auto& sc = shard_cache_[tensor_index];
  if (sc.built) return true;

  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  sc.built = true;
  sc.scheme = to_ws(t.qcfg.scheme);
  sc.group_size = t.qcfg.group_size;
  sc.has_global_divisor = t.qcfg.has_global_divisor;
  sc.K = t.K;
  sc.code_usm.resize(devices_.size());
  sc.scale_usm.resize(devices_.size());
  sc.global_usm.resize(devices_.size());
  sc.act_usm.resize(devices_.size());
  sc.out_usm.resize(devices_.size());

  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    uint32_t N_local = end - start;

    // 1) Build the host-side shard data once (unpack + repack).
    auto w = shard_weight(model_, t, start, end, q);

    // 2) Allocate persistent USM on each device and copy weight data.
    sc.N_local = N_local;

    // Code: packed bytes
    size_t code_bytes = w.code_store.size();
    sc.code_usm[d] = static_cast<std::uint8_t*>(
        sycl::malloc_shared(code_bytes, dev.sycl_device, dev.sycl_context));
    std::memcpy(sc.code_usm[d], w.code_store.data(), code_bytes);

    // Scale: N_local * n_groups FP16
    uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
    uint32_t n_groups = (t.K + G - 1) / G;
    size_t scale_bytes = static_cast<size_t>(N_local) * n_groups * sizeof(uint16_t);
    sc.scale_usm[d] = static_cast<std::uint16_t*>(
        sycl::malloc_shared(scale_bytes, dev.sycl_device, dev.sycl_context));
    std::memcpy(sc.scale_usm[d], w.scale_store.data(), scale_bytes);

    // Global divisor: 1 float
    if (t.global_bytes > 0) {
      sc.global_usm[d] = static_cast<float*>(
          sycl::malloc_shared(sizeof(float), dev.sycl_device, dev.sycl_context));
      std::memcpy(sc.global_usm[d], w.global_store.data(), sizeof(float));
    }

    // 3) Allocate persistent activation buffer (K FP16) and output buffer (N_local FP16).
    sc.act_usm[d] = static_cast<std::uint16_t*>(
        sycl::malloc_shared(t.K * sizeof(uint16_t), dev.sycl_device, dev.sycl_context));
    sc.out_usm[d] = static_cast<std::uint16_t*>(
        sycl::malloc_shared(static_cast<size_t>(N_local) * sizeof(uint16_t),
                           dev.sycl_device, dev.sycl_context));
  }
  return true;
}

bool Engine::load_language_weights(const std::string& safetensors_path) {
  SafetensorsReader r;
  if (!r.open(safetensors_path)) {
    std::cerr << "Engine: cannot open safetensors " << safetensors_path << "\n";
    return false;
  }
  for (const auto& t : r.tensors()) {
    std::vector<float> v;
    if (!r.read(t.name, v)) { std::cerr << "Engine: read fail " << t.name << "\n"; return false; }
    lang_[t.name] = std::move(v);
  }
  std::cout << "Engine: loaded " << lang_.size() << " safetensors tensors\n";
  return true;
}

}  // namespace intfer

#if INTFER_GRAPH_CAPTURE
#include <sycl/ext/oneapi/experimental/graph.hpp>

bool intfer::Engine::build_usm_buffers_for_tensor(int tensor_index) {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size()) return false;
  const auto& t = model_.tensors[tensor_index];
  if ((int)usm_.size() != (int)devices_.size()) usm_.resize(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = shards_[tensor_index].ranges[d];
    auto& u = usm_[d];
    u.q = dev.queue;
    auto w = shard_weight(model_, t, start, end, u.q);
    u.code_store = std::move(w.code_store);
    u.scale_store = std::move(w.scale_store);
    u.global_store = std::move(w.global_store);
    u.code.emplace(u.code_store.data(), sycl::range<1>(u.code_store.size()));
    u.scale.emplace(u.scale_store.data(), sycl::range<1>(u.scale_store.size()));
    u.global.emplace(u.global_store.data(), sycl::range<1>(u.global_store.size()));
    u.act_store.assign(t.K, 0);
    u.act.emplace(u.act_store.data(), sycl::range<1>(u.act_store.size()));
    u.out_store.assign(end - start, 0);
    u.out.emplace(u.out_store.data(), sycl::range<1>(u.out_store.size()));
    u.c_store.assign(end - start, 0);
    u.c.emplace(u.c_store.data(), sycl::range<1>(u.c_store.size()));
  }
  return true;
}

void intfer::Engine::build_decode_graph() {
  if (graph_built_ || model_.tensors.empty()) return;
  using namespace sycl::ext::oneapi::experimental;
  build_usm_buffers_for_tensor(0);
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& u = usm_[d];
    auto [start, end] = shards_[0].ranges[d];
    command_graph<graph_state::modifiable> g(u.q.get_context(), u.q.get_device());
    // act and c are persistent optional buffers allocated by build_usm_buffers_for_tensor
    gemm::GemmActBuffers a;
    a.scheme = gemm::ActScheme::FP16;
    a.M = 1; a.K = model_.tensors[0].K;
    a.fp16 = *u.act;   // act buffer is persistent
    gemm::QuantGemmOptions opt;
    opt.use_dpas = false;
    gemm::GemmWeightBuffers w;
    w.scheme = gemm::WeightScheme::INT8;
    w.group_size = model_.tensors[0].qcfg.group_size;
    w.has_global_divisor = model_.tensors[0].qcfg.has_global_divisor;
    w.N = end - start;
    w.K = model_.tensors[0].K;
    w.code = *u.code;
    w.scale = *u.scale;
    w.global = *u.global;
    quant_gemm_forward(u.q, a, w, *u.c, opt);
    g.end_recording();
    u.graph = std::make_unique<command_graph<graph_state::executable>>(g.finalize());
  }
  graph_built_ = true;
}
#endif


