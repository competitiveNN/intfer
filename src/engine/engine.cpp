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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sycl/sycl.hpp>
#include <chrono>
#include <vector>
// ...


#include "intfer/output.hpp"
#include "intfer/safetensors.hpp"
#include "intfer/gemm/quant_gemm.hpp"
#include "intfer/attention/flash_attention.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/device/device_manager.hpp"

#include "intfer/spinquant.hpp"  // for canonical_codebook + unpack_nibble

namespace intfer {

Engine::~Engine() {
  if (backing_.base && backing_.size) ::munmap(backing_.base, backing_.size);
  backing_.base = nullptr;
  backing_.size = 0;
  // Drain the USM pool (body runs before member destruction, so queues live).
  for (size_t d = 0; d < usm_pool_.size() && d < devices_.size(); ++d)
    for (auto& [bytes, vec] : usm_pool_[d]) {
      (void)bytes;
      for (void* p : vec) sycl::free(p, devices_[d].queue);
      vec.clear();
    }
  // Free the shared act/out workspaces (2 live allocs per device max).
  for (size_t d = 0; d < ws_.size() && d < devices_.size(); ++d) {
    if (ws_[d].act) sycl::free(ws_[d].act, devices_[d].queue);
    if (ws_[d].out) sycl::free(ws_[d].out, devices_[d].queue);
    ws_[d].act = nullptr; ws_[d].out = nullptr;
  }
  // Free the DPAS act workspaces (3 live allocs per device max).
  for (size_t d = 0; d < ws48_.size() && d < devices_.size(); ++d) {
    if (ws48_[d].act) sycl::free(ws48_[d].act, devices_[d].queue);
    if (ws48_[d].as) sycl::free(ws48_[d].as, devices_[d].queue);
    if (ws48_[d].f32) sycl::free(ws48_[d].f32, devices_[d].queue);
    if (ws48_[d].si) sycl::free(ws48_[d].si, devices_[d].queue);
    if (ws48_[d].fold) sycl::free(ws48_[d].fold, devices_[d].queue);
    ws48_[d].act = nullptr; ws48_[d].as = nullptr; ws48_[d].f32 = nullptr;
    ws48_[d].si = nullptr; ws48_[d].fold = nullptr;
  }
  // Free the wave workspaces (2*W live allocs per device max).
  for (size_t i = 0; i < wv_.size(); ++i) {
    const size_t dd = wv_W_ > 0 ? i / wv_W_ : 0;
    if (dd < devices_.size()) {
      if (wv_[i].act) sycl::free(wv_[i].act, devices_[dd].queue);
      if (wv_[i].out) sycl::free(wv_[i].out, devices_[dd].queue);
      wv_[i].act = nullptr; wv_[i].out = nullptr;
    }
  }
  wv_.clear(); wv_W_ = 0; wv_M_ = wv_K_ = wv_Nl_ = 0;
}

void* Engine::pool_acquire(size_t device, size_t bytes) {
  if (device >= devices_.size() || bytes == 0) return nullptr;
  if (device >= usm_pool_.size()) usm_pool_.resize(device + 1);
  pool_acquires_++;
  auto& vec = usm_pool_[device][bytes];
  if (!vec.empty()) {
    void* p = vec.back();
    vec.pop_back();
    pool_hits_++;
    return p;
  }
  auto& dev = devices_[device];
  void* p = sycl::malloc_shared(bytes, dev.sycl_device, dev.sycl_context);
  if (!p) {
    // Device OOM under preload-all: fail loudly with the exact shortfall
    // instead of a downstream nullptr-memcpy SIGSEGV. Steady-state VRAM is
    // ~10.9GB weights/dev (27B-sc TP=2) vs the 12216MiB B580 budget.
    std::fprintf(stderr,
                 "[intfer] FATAL: USM alloc of %zu bytes failed on device %zu "
                 "(VRAM budget exceeded?)\n",
                 bytes, device);
    std::fflush(stderr);
    std::abort();
  }
  return p;
}

void Engine::pool_release(size_t device, void* ptr, size_t bytes) {
  if (!ptr || bytes == 0) return;
  if (device >= usm_pool_.size()) usm_pool_.resize(device + 1);
  usm_pool_[device][bytes].push_back(ptr);
}

#ifdef INTFER_PROFILE_GPU
  struct GpuProfile {
    double total_kernel_us  = 0.0;
    double total_wait_us    = 0.0;
    double total_copy_us    = 0.0;
    double total_readback_us = 0.0;
    double total_usm_write_us = 0.0;  // host USM activation write loop
    double total_submit_us = 0.0;     // q.submit (launch) host overhead
    double total_waitblock_us = 0.0;  // event.wait() blocking host time
    double total_rebuild_us = 0.0;    // ensure_shard_built (host repack + malloc_shared + memcpy)
    uint64_t rebuild_count = 0;
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

// TODO 6 (W4A8/W2A8): DPAS eligibility for a shard. Requirements:
//   * low-bit scheme with supported (bits, weight-group) combo:
//     INT4/G64, INT8/G32, INT2/G32, INT2/G16 (G16 = 2 DPAS/op split)
//   * N_local % 16 == 0, K % 32 == 0 (all 27B/0.8B tensors post-TP-split)
//   * per-tensor global divisor only via host alpha (already the case)
// Activation quant (INT8/GA64) and repack happen at shard build; dispatch
// is then a flag check in run_linear_tp. Default OFF (2026-09-18: the DPAS
// path is a net loss on 0.8B at every tested shape — host GA64 act-quant
// ~6ms/token dominates, XMX kernel no faster than the tiled legacy kernel
// at M=1/128; greedy streams bit-identical either way). INTFER_DPAS48=1
// opts back in (bisection / future shapes where XMX wins).
bool dpas48_eligible(QuantScheme scheme, uint32_t group_size,
                     uint32_t N_local, uint32_t K) {
  const char* env = std::getenv("INTFER_DPAS48");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  if (!enabled) return false;
  int bits = (scheme == QuantScheme::INT2)    ? 2
             : (scheme == QuantScheme::INT4)  ? 4
             : (scheme == QuantScheme::INT8)  ? 8
                                              : 16;
  if (bits == 16) return false;
  const uint32_t G = (group_size == 0) ? K : group_size;
  const bool combo_ok =
      (bits == 4 && G == 64) || (bits == 8 && G == 32) ||
      (bits == 2 && (G == 32 || G == 16));
  if (!combo_ok) return false;
  return (N_local % 16 == 0) && (K % 32 == 0);
}

// Row-cooperative DPAS experiment opt-in (default OFF). Host-side read;
// getenv/static are illegal in device code, so the flag is read once.
bool dpas_row_enabled() {
  const char* env = std::getenv("INTFER_DPAS_ROW");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled;
}

// E4: cooperative 2D row-block DPAS (default OFF). Same getenv/static rule.
//
// Deliberately MUTUALLY EXCLUSIVE with INTFER_DPAS_ROW: that experiment already
// owns the "one WI owns many tiles" geometry and measured ~460x slower by
// collapsing occupancy. Letting both flags be set would leave it undecidable
// which geometry ran, which is the "unattributable experiment" failure the
// report warns about. Declining the combination is loud in the sense that
// dpas_row wins and E4 does not fire.
bool dpas_tiled_enabled() {
  const char* env = std::getenv("INTFER_DPAS48_TILED");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled && !dpas_row_enabled();
}

// Device act-quant pre-pass opt-in (default OFF). Host-side read, same
// getenv/static rule as dpas_row_enabled(). The pre-pass is bit-exact (same
// GA64 HalfAway contract as dashq::act_quant_ga64_batch) — it moves the
// quant from host to device, it does not change the arithmetic.
bool dpas_preq_enabled() {
  const char* env = std::getenv("INTFER_DPAS_PREQ");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled;
}

// DPAS row-fold hoist opt-in (default OFF). Same getenv/static rule as the
// flags above. Pure work hoist: the values moved out of the tile loop are
// exact int32 sums of the int8 activation codes, so this cannot change a
// result — it only stops every 16-N tile WI from recomputing them.
bool dpas_fold_enabled() {
  const char* env = std::getenv("INTFER_DPAS48_FOLD");
  static const bool enabled = (env && env[0] == '1' && env[1] == '\0');
  return enabled;
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
  for (auto& R : e->rs_.rotations){
    // The serializer does not persist is_hadamard: re-detect empirically so
    // online replay (via rotation_for) can use the O(n log n) FWHT path.
    R.is_hadamard = rotation_is_hadamard(R);
    e->rid_[R.id] = &R;
  }

  // Discover the 2x Arc B580 fleet and pick `tp_devices` of them.
  auto dm = device::DeviceManager::discover("Arc");
  size_t n = (tp_devices == 0) ? dm.size() : std::min(tp_devices, dm.size());
  if (n == 0) { std::cerr << "Engine: no Arc GPU discovered\n"; return nullptr; }
  // AUDIT #9/#11: the min() clamp above is correct but SILENT. When the caller
  // asked for more ranks than the box actually has, the run continues at the
  // smaller TP and, if the model does not fit that many devices, dies later in
  // an opaque place — for 27B on one B580 the failure surfaces as a Level Zero
  // CPU-page-fault-memory-manager abort (signal 6) during prefill, which reads
  // like a driver bug and buries the real cause. Loud, NON-fatal: a legitimate
  // deliberate TP=1 run must not be blocked by this, so warn and continue.
  if (tp_devices != 0 && dm.size() < tp_devices) {
    std::cerr << "Engine: WARNING: TP=" << tp_devices << " requested but only "
              << dm.size() << " Arc GPU(s) discovered — continuing with TP=" << n
              << ". A model that needs more VRAM than one device holds (27B "
                 "needs ~19GB; a B580 has ~12GB) will fail below.\n"
              << std::flush;
  }
  for (size_t i = 0; i < n; ++i) e->devices_.push_back(dm.at(i));
  std::cerr << "Engine: " << n << " Arc device(s): ";
  for (auto& d : e->devices_) std::cerr << d.name << " ";
  std::cerr << "\n";

  e->build_shards();
  return e;
}

#ifdef INTFER_HAS_L0
#include <level_zero/zes_api.h>
#endif

// Free VRAM bytes on device i via sysman (see header). Sysman device order
// is NOT SYCL order (measured reversed on 2x B580), so match by native
// UUID: sycl::get_native<L0> handle -> zeDeviceGetProperties uuid vs
// zesDeviceGetProperties core.uuid (same UUID space; never call the core
// getter on a sysman handle — it segfaults). Returns 0 on any failure;
// the caller falls back to an even split.
uint64_t Engine::device_free_bytes(size_t i) {
  if (i >= devices_.size()) return 0;
#ifdef INTFER_HAS_L0
  if (!devices_[i].has_l0()) return 0;
  ze_device_properties_t want{};
  want.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (zeDeviceGetProperties(devices_[i].ze_device, &want) != ZE_RESULT_SUCCESS)
    return 0;
  if (zesInit(0) != ZE_RESULT_SUCCESS) return 0;
  uint64_t free_total = 0;
  bool matched = false;
  uint32_t ndrv = 0;
  if (zesDriverGet(&ndrv, nullptr) != ZE_RESULT_SUCCESS) return 0;
  std::vector<zes_driver_handle_t> drvs(ndrv);
  if (zesDriverGet(&ndrv, drvs.data()) != ZE_RESULT_SUCCESS) return 0;
  for (auto drv : drvs) {
    uint32_t ndev = 0;
    if (zesDeviceGet(drv, &ndev, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<zes_device_handle_t> sdevs(ndev);
    if (zesDeviceGet(drv, &ndev, sdevs.data()) != ZE_RESULT_SUCCESS) continue;
    for (auto sd : sdevs) {
      zes_device_properties_t sp{};
      sp.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      if (zesDeviceGetProperties(sd, &sp) != ZE_RESULT_SUCCESS) continue;
      bool eq = true;
      for (int b = 0; b < 16; ++b)
        if (sp.core.uuid.id[b] != want.uuid.id[b]) { eq = false; break; }
      if (!eq) continue;
      matched = true;
      uint32_t nmem = 0;
      if (zesDeviceEnumMemoryModules(sd, &nmem, nullptr) != ZE_RESULT_SUCCESS)
        return 0;
      std::vector<zes_mem_handle_t> mems(nmem);
      if (zesDeviceEnumMemoryModules(sd, &nmem, mems.data()) != ZE_RESULT_SUCCESS)
        return 0;
      for (auto mh : mems) {
        zes_mem_state_t st{};
        st.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
        if (zesMemoryGetState(mh, &st) == ZE_RESULT_SUCCESS) free_total += st.free;
      }
    }
  }
  return matched ? free_total : 0;
#else
  return 0;
#endif
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
  // Auto-disable layer-wise streaming when the full model fits comfortably
  // in VRAM. Streaming evicts + rebuilds every layer on each advance, so a
  // small model (0.8B: ~0.2GB) pays a full malloc_shared + host repack per
  // tensor per token for no benefit. begin_layer() early-returns on an
  // empty layer_tensors_, so clearing it disables eviction entirely.
  // Threshold 4GB: far above small models, far below 27B (~23GB).
  {
    uint64_t total_w = 0;
    for (const auto& t : model_.tensors) total_w += t.weight_bytes();
    // Per-device footprint ≈ weights + act/out working buffers (small).
    if (total_w < 4ULL * 1024 * 1024 * 1024) {
      layer_tensors_.clear();
      std::cerr << "Engine: streaming disabled (model "
                << total_w / (1024 * 1024) << " MB fits in VRAM)\n";
    }
  }
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
  if (pinned_) return;  // preload_all(): everything stays resident.
  if (tensor_index < 0 || (size_t)tensor_index >= shard_cache_.size()) return;
  auto& sc = shard_cache_[(size_t)tensor_index];
  if (!sc.built) return;
  // Park buffers in the shape-keyed pool instead of freeing (Phase 3): the
  // same shapes recur every layer, so steady-state rebuild is acquire+memcpy
  // with no L0 alloc or page re-pinning. Sizes recomputed exactly as in
  // ensure_shard_built (host-cache sizes when present, else model geometry).
  const auto& t = model_.tensors[tensor_index];
  // run_linear_tp waits on kernel events before readback, so by the time a
  // later begin_layer() releases this shard no kernel can still reference
  // it. Belt-and-braces queue sync (already complete => ~free).
  for (size_t d = 0; d < devices_.size(); ++d) {
    devices_[d].queue.wait();
    size_t code_bytes = sc.has_host_cache && d < sc.code_host.size()
        ? sc.code_host[d].size() : 0;
    size_t scale_bytes = sc.has_host_cache && d < sc.scale_host.size()
        ? sc.scale_host[d].size() * sizeof(uint16_t) : 0;
    // TODO 6: DPAS shards hold repacked tiles in USM (dpas_bytes), not the
    // row-major code_host bytes — pool/release by the USM size.
    if (sc.dpas48 && d < sc.dpas_bytes.size() && sc.dpas_bytes[d] > 0)
      code_bytes = sc.dpas_bytes[d];
    if (code_bytes == 0 || scale_bytes == 0) {
      // No host cache (e.g. FP16 first-build): derive from model geometry.
      // FP16 code = full N_local*K halves; INT* packed = via group math.
      // These shards are resident in practice; this is a safe fallback.
      auto& rng = shards_[tensor_index].ranges;
      uint32_t Nl = (d < rng.size()) ? rng[d].second - rng[d].first : 0;
      uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
      uint32_t ng = (t.K + G - 1) / G;
      const int bits = (sc.scheme == gemm::WeightScheme::INT2) ? 2
                     : (sc.scheme == gemm::WeightScheme::INT4) ? 4
                     : (sc.scheme == gemm::WeightScheme::FP16) ? 16 : 8;
      const int pack = (bits >= 8) ? 1 : (8 / bits);
      code_bytes = (sc.scheme == gemm::WeightScheme::FP16)
          ? (size_t)Nl * t.K * 2
          : (size_t)Nl * ((t.K + pack - 1) / pack);
      scale_bytes = (size_t)Nl * ng * sizeof(uint16_t);
    }
    if (sc.code_usm.size() > d && sc.code_usm[d]) {
      pool_release(d, sc.code_usm[d], code_bytes);
      sc.code_usm[d] = nullptr;
    }
    if (sc.scale_usm.size() > d && sc.scale_usm[d]) {
      pool_release(d, sc.scale_usm[d], scale_bytes);
      sc.scale_usm[d] = nullptr;
    }
    // NOTE: alpha is host-side only (global_host); no device global buffer.
    // NOTE: no per-shard act/out — the shared workspaces (ws_) serve all
    // shards, so release only the weight buffers (code/scale).
  }
  sc.code_usm.clear(); sc.scale_usm.clear();
  sc.built = false;
}

// Wave workspaces for run_linear_wave (see engine.hpp): per-device slot
// pairs, one slot per live wave item. Grow-only; queue-wait before
// realloc (same aliasing contract as ensure_workspace: slots never live
// across calls, every wave event is waited before readback).
void Engine::ensure_wave_workspace(uint32_t M, uint32_t W, uint32_t K,
                                   uint32_t Nl) {
  if (M == 0) M = 1;
  if (W == 0) W = 1;
  if (wv_.size() == devices_.size() * (size_t)wv_W_ && wv_W_ > 0 &&
      M <= wv_M_ && W <= wv_W_ && K <= wv_K_ && Nl <= wv_Nl_)
    return;
  for (size_t d = 0; d < devices_.size(); ++d) devices_[d].queue.wait();
  const uint32_t nM = std::max(M, wv_M_);
  const uint32_t nW = std::max(W, wv_W_);
  const uint32_t nK = std::max(K, wv_K_);
  const uint32_t nNl = std::max(Nl, wv_Nl_);
  std::vector<WaveSlot> nwv(devices_.size() * (size_t)nW);
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    for (uint32_t i = 0; i < nW; ++i) {
      WaveSlot s;
      s.act = static_cast<std::uint16_t*>(sycl::malloc_shared(
          (size_t)nM * nK * sizeof(uint16_t), dev.sycl_device, dev.sycl_context));
      s.out = static_cast<std::uint16_t*>(sycl::malloc_shared(
          (size_t)nM * nNl * sizeof(uint16_t), dev.sycl_device, dev.sycl_context));
      if (!s.act || !s.out) {
        std::fprintf(stderr,
                     "[intfer] FATAL: wave workspace alloc failed on device %zu "
                     "(M=%u W=%u K=%u Nl=%u)\n", d, nM, nW, nK, nNl);
        std::fflush(stderr);
        std::abort();
      }
      nwv[d * nW + i] = s;
    }
  }
  // Free old slots after the new ones are live (queues already waited).
  for (size_t i = 0; i < wv_.size(); ++i) {
    const size_t dd = wv_W_ > 0 ? i / wv_W_ : 0;
    if (dd < devices_.size()) {
      if (wv_[i].act) sycl::free(wv_[i].act, devices_[dd].queue);
      if (wv_[i].out) sycl::free(wv_[i].out, devices_[dd].queue);
    }
  }
  wv_ = std::move(nwv);
  wv_W_ = nW; wv_M_ = nM; wv_K_ = nK; wv_Nl_ = nNl;
}

// Shared per-device act/out workspaces (replaces per-shard act/out USM).
// One buffer pair per device serves all shards: a shard's act/out are
// written+consumed within one run_linear_tp call and never live across calls
// (single-threaded CLI), and every kernel event is waited before readback,
// so aliasing across shards is safe. Grow-only, direct malloc_shared (NOT
// the shape pool — pool entries stay live and would pin the old size
// resident). K/Nl vary per tensor, so the workspace covers the max seen.
void Engine::ensure_workspace(uint32_t M, uint32_t K, uint32_t Nl_max) {
  if (M == 0) M = 1;
  if (ws_.size() != devices_.size()) ws_.resize(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& w = ws_[d];
    if (w.act && w.out && M <= w.M && K <= w.K && Nl_max <= w.Nl) continue;
    devices_[d].queue.wait();
    uint32_t nM = std::max(M, w.M);
    uint32_t nK = std::max(K, w.K);
    uint32_t nNl = std::max(Nl_max, w.Nl);
    if (w.act) { sycl::free(w.act, devices_[d].queue); w.act = nullptr; }
    if (w.out) { sycl::free(w.out, devices_[d].queue); w.out = nullptr; }
    auto& dev = devices_[d];
    w.act = static_cast<std::uint16_t*>(sycl::malloc_shared(
        (size_t)nM * nK * sizeof(uint16_t), dev.sycl_device, dev.sycl_context));
    w.out = static_cast<std::uint16_t*>(sycl::malloc_shared(
        (size_t)nM * nNl * sizeof(uint16_t), dev.sycl_device, dev.sycl_context));
    if (!w.act || !w.out) {
      std::fprintf(stderr,
                   "[intfer] FATAL: workspace alloc failed on device %zu "
                   "(M=%u K=%u Nl=%u)\n", d, nM, nK, nNl);
      std::fflush(stderr);
      std::abort();
    }
    w.M = nM; w.K = nK; w.Nl = nNl;
  }
}

// TODO 6 (W4A8/W2A8): grow-only per-device INT8-act + act-scale workspaces.
// GA=64 fixed (matches model qcfg act INT8_G64); nGA covers max K seen.
void Engine::ensure_workspace48(uint32_t M, uint32_t K) {
  constexpr uint32_t GA = 64;
  if (M == 0) M = 1;
  if (ws48_.size() != devices_.size()) ws48_.resize(devices_.size());
  const uint32_t nGA = (K + GA - 1) / GA;
  const uint32_t nOp = K / 32;  // DPAS K-op = 32
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& w = ws48_[d];
    if (w.act && w.as && w.f32 && w.si && w.fold && M <= w.M && K <= w.K &&
        nGA <= w.nGA && nOp <= w.nOp)
      continue;
    devices_[d].queue.wait();
    uint32_t nM = std::max(M, w.M);
    uint32_t nK = std::max(K, w.K);
    uint32_t nnGA = std::max(nGA, w.nGA);
    uint32_t nnOp = std::max(nOp, w.nOp);
    if (w.act) { sycl::free(w.act, devices_[d].queue); w.act = nullptr; }
    if (w.as) { sycl::free(w.as, devices_[d].queue); w.as = nullptr; }
    if (w.f32) { sycl::free(w.f32, devices_[d].queue); w.f32 = nullptr; }
    if (w.si) { sycl::free(w.si, devices_[d].queue); w.si = nullptr; }
    if (w.fold) { sycl::free(w.fold, devices_[d].queue); w.fold = nullptr; }
    auto& dev = devices_[d];
    w.act = static_cast<std::int8_t*>(sycl::malloc_shared(
        (size_t)nM * nK, dev.sycl_device, dev.sycl_context));
    w.as = static_cast<float*>(sycl::malloc_shared(
        (size_t)nM * nnGA * sizeof(float), dev.sycl_device, dev.sycl_context));
    w.f32 = static_cast<float*>(sycl::malloc_shared(
        (size_t)nM * nK * sizeof(float), dev.sycl_device, dev.sycl_context));
    w.si = static_cast<float*>(sycl::malloc_shared(
        2 * (size_t)nM * nnGA * sizeof(float), dev.sycl_device, dev.sycl_context));
    // 2 * M * nOp INT32 (S0 + Sfull per op). Guard the multiply: nnOp is
    // derived from nK, and the count is allocated 64-bit.
    w.fold = static_cast<std::int32_t*>(sycl::malloc_shared(
        (size_t)2 * nM * nnOp * sizeof(std::int32_t), dev.sycl_device,
        dev.sycl_context));
    if (!w.act || !w.as || !w.f32 || !w.si || !w.fold) {
      std::fprintf(stderr,
                   "[intfer] FATAL: workspace48 alloc failed on device %zu "
                   "(M=%u K=%u nGA=%u nOp=%u)\n", d, nM, nK, nnGA, nnOp);
      std::fflush(stderr);
      std::abort();
    }
    w.M = nM; w.K = nK; w.nGA = nnGA; w.nOp = nnOp;
  }
}

void Engine::begin_layer(int L) {
  if (pinned_) return;  // preload_all(): steady-state zero evict+rebuild.
  if (layer_tensors_.empty()) return;
  // Phase 1: FP16-scheme shards are memcpy-identity slices (no unpack work)
  // and small enough to stay resident: layered FP16 is ~13.5GB total, i.e.
  // ~6.7GB per device under TP=2, well under the 12GB B580 budget
  // (plus ~2.5GB cached lm_head and the INT4 streaming working set).
  // Skipping their eviction kills ~75% of per-token evict+rebuild traffic.
  // INTFER_STREAM_F16=1 restores the old evict-everything behavior.
  static const bool stream_f16 = (std::getenv("INTFER_STREAM_F16") != nullptr);
  for (size_t ti = 0; ti < shard_cache_.size(); ++ti) {
    if (!shard_cache_[ti].built) continue;
    if (!stream_f16 && shard_cache_[ti].scheme == gemm::WeightScheme::FP16)
      continue;
    int tl = (ti < layer_of_.size()) ? layer_of_[ti] : -1;
    if (tl >= 0 && tl != L) release_shard((int)ti);
  }
}

// Full-VRAM residency: build every runnable Linear shard once and pin it.
// Only GEMM-runnable roles are built (ATTN Q/K/V/O, MLP gate/up/down, linear
// attention QKV/Z/B/A/OUT). Skipped: lm_head-named tensors (the CLI serves
// lm_head through its own TP-sharded run_lm_head_f16 upload, never through
// engine shards — building the slot would waste 1.27GB/dev), norms/vectors
// (K=1), conv1d and OTHER roles (host-side; ~1.27GB/dev of dead shards).
// Anything missed lazy-builds on first run_linear_tp (correct by
// construction: pinning blocks eviction, never construction).
// Afterwards the INT4 host repack cache (~9.4GB host RAM on 27B-sc) is
// dropped — with eviction disabled no rebuild can ever need it. Bit-identical:
// same bytes as lazy builds.
void Engine::preload_all() {
  if (pinned_) return;
  // No rebuilds will ever happen: skip the host repack cache entirely
  // (saves ~9.4GB host RAM; without this the OOM killer takes the preload).
  skip_host_cache_ = true;
  auto is_runnable = [](TensorRole r) {
    switch (r) {
      case TensorRole::ATTN_Q: case TensorRole::ATTN_K:
      case TensorRole::ATTN_V: case TensorRole::ATTN_O:
      case TensorRole::MLP_GATE: case TensorRole::MLP_UP:
      case TensorRole::MLP_DOWN:
      case TensorRole::LINEAR_ATTN_QKV: case TensorRole::LINEAR_ATTN_Z:
      case TensorRole::LINEAR_ATTN_B: case TensorRole::LINEAR_ATTN_A:
      case TensorRole::LINEAR_ATTN_OUT:
        return true;
      default:
        return false;
    }
  };
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t n_built = 0, n_skip = 0;
  uint64_t dev_bytes = 0;
  // Bisection knob (DEVICE_LOST triage): cap the number of preloaded shards.
  // Anything not preloaded lazy-builds on first use (correct by construction).
  size_t preload_max = (size_t)-1;
  if (const char* pm = std::getenv("INTFER_PRELOAD_MAX")) {
    long v = std::atol(pm); if (v > 0) preload_max = (size_t)v;
  }
  for (size_t ti = 0; ti < model_.tensors.size(); ++ti) {
    const auto& t = model_.tensors[ti];
    if (t.name.find("lm_head") != std::string::npos) { n_skip++; continue; }
    if (!is_runnable(t.role)) { n_skip++; continue; }
    if (t.N == 0 || t.K == 0) { n_skip++; continue; }
    if (shard_cache_[ti].built) continue;
    if (n_built >= preload_max) continue;
    ensure_shard_built((int)ti);
    n_built++;
    if ((n_built % 100) == 0) {
      std::cerr << "Engine: preload " << n_built << " shards...\n";
      std::cerr.flush();
    }
    if (const char* pt = std::getenv("INTFER_PRELOAD_TRACE")) {
      // Per-shard trace: cumulative geometry bytes per device + tensor name.
      // Correlates engine accounting with xpu-smi to find device-asymmetric
      // growth (dev0-only jumps). Enable with INTFER_PRELOAD_TRACE=1.
      static uint64_t trace_acc[8] = {0,0,0,0,0,0,0,0};
      auto& rng = shards_[ti].ranges;
      for (size_t d = 0; d < devices_.size() && d < 8; ++d) {
        uint32_t Nl = (d < rng.size()) ? rng[d].second - rng[d].first : 0;
        uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
        uint32_t ng = (t.K + G - 1) / G;
        const int tbits = (t.qcfg.scheme == QuantScheme::INT2) ? 2
                        : (t.qcfg.scheme == QuantScheme::INT4) ? 4
                        : (t.qcfg.scheme == QuantScheme::FP16) ? 16 : 8;
        const int tpack = (tbits >= 8) ? 1 : (8 / tbits);
        trace_acc[d] += (tbits == 16)
            ? (uint64_t)Nl * t.K * 2 + (uint64_t)Nl * ng * 2
            : (uint64_t)Nl * ((t.K + tpack - 1) / tpack)
              + (uint64_t)Nl * ng * 2;
      }
      auto now = std::chrono::high_resolution_clock::now();
      double el = std::chrono::duration<double>(now - t0).count();
      std::cerr << "Engine: preload #" << n_built << " t=" << (long)el
                << "s dev0=" << trace_acc[0] / (1024*1024) << "MiB dev1="
                << (devices_.size() > 1 ? trace_acc[1] / (1024*1024) : 0)
                << "MiB " << t.name << "\n";
      std::cerr.flush();
    }
  }
  // Pin first: any concurrent begin_layer() (none expected — single-threaded
  // CLI, but belt-and-braces) must not evict what we just built.
  // (skip_host_cache_ stays on: pinned shards are never rebuilt, and any
  // lazy-built straggler is single-use — caching it would be dead RAM.)
  pinned_ = true;
  // NOTE (2026-09-21, measured on 27B TP=2): shard USM is malloc_shared and
  // pages stay host-resident until first kernel touch — xpu-smi shows
  // dev0 854MB/dev1 115MB at startup, jumping to ~11GB only after the first
  // request. A q.prefetch(code)+q.prefetch(scale) pass over all shards here
  // was tried and REVERTED: 18.7GB "migrated" in 59ms per the log, but
  // xpu-smi was unchanged and the first-request jump identical — prefetch
  // is a no-op hint for shared allocations on this Level Zero driver.
  // First-request migration stalls stand; do NOT re-add prefetch as-is.
  // (A real touch kernel over every shard at startup might migrate pages,
  // but that is 496 extra launches + waits in the startup path — unscoped.)
  for (size_t d = 0; d < devices_.size(); ++d) {
    devices_[d].queue.wait();
    // Tally resident weight bytes per device (code+scale+global).
    uint64_t per_dev = 0;
    for (size_t ti = 0; ti < shard_cache_.size(); ++ti) {
      const auto& sc = shard_cache_[ti];
      if (!sc.built || d >= sc.code_usm.size() || !sc.code_usm[d]) continue;
      const auto& t = model_.tensors[ti];
      auto& rng = shards_[ti].ranges;
      uint32_t Nl = (d < rng.size()) ? rng[d].second - rng[d].first : 0;
      uint32_t G = t.qcfg.group_size > 0 ? t.qcfg.group_size : t.K;
      uint32_t ng = (t.K + G - 1) / G;
      const int bits = (sc.scheme == gemm::WeightScheme::INT2) ? 2
                     : (sc.scheme == gemm::WeightScheme::INT4) ? 4
                     : (sc.scheme == gemm::WeightScheme::FP16) ? 16 : 8;
      const int pack = (bits >= 8) ? 1 : (8 / bits);
      per_dev += (sc.scheme == gemm::WeightScheme::FP16)
          ? (uint64_t)Nl * t.K * 2
          : (uint64_t)Nl * ((t.K + pack - 1) / pack);
      per_dev += (uint64_t)Nl * ng * sizeof(uint16_t);
      // Alpha is host-side (no device global buffer).
    }
    if (d == 0) dev_bytes = per_dev;
    std::cerr << "Engine: preload device " << d << " resident weights: "
              << per_dev / (1024 * 1024) << " MiB\n";
    std::cerr.flush();
  }
  // Allocator audit: pool contents (should be empty post-preload) + total
  // acquire count, to separate geometry bytes from runtime overhead.
  {
    size_t pool_n = 0; uint64_t pool_bytes = 0;
    for (size_t d = 0; d < usm_pool_.size(); ++d)
      for (auto& [bytes, vec] : usm_pool_[d]) {
        pool_n += vec.size(); pool_bytes += bytes * vec.size();
      }
    std::cerr << "Engine: preload audit: pool holds " << pool_n
              << " bufs (" << pool_bytes / (1024 * 1024)
              << " MiB), pool_acquire calls=" << pool_acquires_
              << " pool_hits=" << pool_hits_ << "\n";
    std::cerr.flush();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cerr << "Engine: preload_all built " << n_built << " shards (skipped "
            << n_skip << ", host cache bypassed), dev0 weights "
            << dev_bytes / (1024 * 1024)
            << " MiB in " << (long)ms << " ms; shards pinned (no evict)\n";
  std::cerr.flush();
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
 
std::vector<std::vector<float>> Engine::decode_step_batch(
    const std::vector<std::vector<float>>& batch_hidden,
    int tensor_index) {
  std::vector<std::vector<float>> out;
  if (batch_hidden.empty()) return out;
  const uint32_t M = static_cast<uint32_t>(batch_hidden.size());
  const int ti = (tensor_index >= 0 &&
                  (size_t)tensor_index < model_.tensors.size())
                     ? tensor_index
                     : 0;
  const auto& t = model_.tensors[ti];
  std::vector<float> x;
  x.reserve((size_t)M * t.K);
  for (const auto& row : batch_hidden) x.insert(x.end(), row.begin(), row.end());
  std::vector<float> y;
  run_linear_tp(ti, x, y, M);
  out.reserve(M);
  for (uint32_t i = 0; i < M; ++i)
    out.emplace_back(y.begin() + i * t.N, y.begin() + (i + 1) * t.N);
  return out;
}
 
void Engine::run_linear_tp(int tensor_index, const std::vector<float>& x,
                           std::vector<float>& out) {
  run_linear_tp(tensor_index, x, out, 1);
}

// TODO 6 (W4A8/W2A8): DPAS dispatch. Fused host act-quant (FP32 -> INT8,
// per-row GA=64 slices, symmetric) writes the shared ws48_ workspaces;
// the kernel then runs fully from device-resident data. Readback reuses
// ws_[d].out (M*N_local FP16). Same event-wait contract as run_linear_tp.
void Engine::run_linear_tp_dpas48(int tensor_index,
                                  const std::vector<float>& x,
                                  std::vector<float>& out, uint32_t M) {
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (M == 0) M = 1;
  const uint32_t out_elems = M * t.N;
  if (out.size() != out_elems) out.resize(out_elems);
  const auto& sc = shard_cache_[tensor_index];
  constexpr uint32_t GA = 64;
  const uint32_t K = t.K;
  const uint32_t nGA = (K + GA - 1) / GA;
  ensure_workspace48(M, K);

  const int bits = (sc.scheme == gemm::WeightScheme::INT2)    ? 2
                   : (sc.scheme == gemm::WeightScheme::INT4)  ? 4
                                                              : 8;
  const uint32_t Gw = (sc.group_size == 0) ? K : sc.group_size;
  // EXPERIMENTAL row-cooperative geometry (default-off INTFER_DPAS_ROW=1;
  // measured ~460x slower — experiment base only). Eligibility is the KERNEL'S
  // OWN predicate (dpas4_row_eligible), not a second copy of it: this copy used
  // to live here and to agree only by coincidence, and it is the copy that makes
  // the kernel's decline path unreachable. That guarantee cannot rest on the
  // event-identity test below -- measured 2026-10-03, an event a callee returns
  // as a bare sycl::event{} does NOT compare equal to one built here, so the
  // `== sycl::event{}` half of the fallback cannot be relied on (dpas_fold_probe
  // asserts both directions).
  const bool use_row = dpas_row_enabled() && gemm::dpas4_row_eligible(bits, Gw);
  // Device act-quant pre-pass (INTFER_DPAS_PREQ=1, default OFF): per device
  // one FP32 H2D + one GA64 quant kernel, dep-chained into that rank's DPAS
  // launch. Kills the host quant loop and adds no host wait. Per device
  // (NOT one launch for both): each rank owns its sycl::context, so a kernel
  // cannot legally write the other rank's malloc_shared buffer.
  bool preq_ok = dpas_preq_enabled() && !use_row;
  for (size_t d = 0; d < devices_.size() && preq_ok; ++d)
    preq_ok = ws48_[d].f32 && ws48_[d].act && ws48_[d].as && ws48_[d].si;
  // EXPERIMENTAL DPAS row-fold hoist (default-off INTFER_DPAS48_FOLD=1).
  // Composes with the pre-pass above (the fold dep-chains on it) and with the
  // row-cooperative geometry (which has its own fold handling).
  const bool use_fold = dpas_fold_enabled();
// E4: cooperative 2D row block. Requires M >= 2, which quant_gemm_dpas48_2d
  // ALSO enforces and DECLINES on -- the flag is checked here only so the
  // experiment is inert (and attributable) rather than attempted and declined on
  // every decode call.
  const bool use_tiled = dpas_tiled_enabled() && M >= 2 && !use_row;

  // 1) Activation quant into ws48_ (per row, per GA-slice max-abs scale).
  //    Device pre-pass first; on ANY decline fall through to the host loop,
  //    which rewrites every device's rows — never partial work.
  std::vector<sycl::event> preq_events;
#ifdef INTFER_PROFILE_GPU
  auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
  if (preq_ok) {
    preq_events.resize(devices_.size());
    // (scale, inverse) pairs on the HOST, verbatim. The device must neither
    // divide nor re-reduce: q = trunc(x*inv +- 0.5) turns a 1-ulp reciprocal
    // difference into a whole +-1 code error at .5 ties, and `1.0f/s` is not a
    // portable bit-source (device lowers it to rcp.approx). Shipping the host's
    // bits -- all produced by dashq's single noinline ga64_recip site, gated
    // by scripts/test_ga64_recip.sh -- makes the pre-pass bit-exact by
    // construction. O(M*K) work + a tiny H2D; this replaces the O(M*K) host
    // quant loop it was introduced to delete (but not its max-abs reduction),
    // and stays off the GEMM's critical path (dep-chained, no added wait).
    if (si48_staging_.size() != 2 * (size_t)M * nGA)
      si48_staging_.resize(2 * (size_t)M * nGA);
    // nGA is re-derived inside the helper from K (identical expression to the
    // kernel's), so the staging size above and the kernel's slice count cannot
    // drift apart.
    dashq::act_quant_ga64_scale_inv_batch(x.data(), M, K,
                                          si48_staging_.data());
    // Stop submitting once any device declines: the remaining devices are
    // about to be overwritten by the host fallback anyway.
    for (size_t d = 0; d < devices_.size() && preq_ok; ++d) {
      sycl::queue& q = devices_[d].queue;
      float* xd = ws48_[d].f32;
      // Unconditional H2D. `x` is a host std::vector; `xd` is device USM,
      // so an address comparison can never legitimately match (and must not
      // be used to skip the copy — the source is host memory either way).
      q.memcpy(xd, x.data(), (size_t)M * K * sizeof(float));
      gemm::ActQuantDevParams ap;
      q.memcpy(ws48_[d].si, si48_staging_.data(),
               2 * (size_t)M * nGA * sizeof(float));
      ap.x_f32 = xd;
      ap.scale_inv_in = ws48_[d].si;
      ap.q_out = ws48_[d].act;
      ap.s_out = ws48_[d].as;
      ap.M = M;
      ap.K = K;
      ap.Kstride = ws48_[d].K;
      ap.nGAstride = ws48_[d].nGA;
      preq_events[d] = gemm::act_quant_ga64_dev(q, ap);
      if (preq_events[d] == sycl::event{}) preq_ok = false;
    }
    // PARTIAL-DECLINE GUARD: a pre-pass kernel already submitted for device
    // d' < d is still queued and WILL write ws48_[d'].act asynchronously. If
    // we now fall through to the host fallback, its std::memcpy into the same
    // buffer races that in-flight kernel -> nondeterministic act codes. Drain
    // every already-submitted pre-pass event first (the host path then owns
    // the buffer exclusively and overwrites all of it).
    if (!preq_ok) {
      for (auto& e : preq_events)
        if (e != sycl::event{}) e.wait_and_throw();
      preq_events.clear();
    }
  } else {
    // Host fallback: vectorized GA=64 quant into HOST staging (bit-exact vs
    // the old scalar loop, incl. the trunk's original lround half-away-from-
    // zero); then one memcpy per device row into the shared workspace. Scalar
    // stores straight into malloc_shared are ~100x slower (copy bucket
    // 96ms/7tok); memcpy emits wide vectorized stores. Staging is grow-only
    // (zero steady-state allocs); rows are tight here, workspace rows strided
    // — copy per row.
    //
    // This branch ALSO owns the (scale, inverse) pairs the pre-pass path
    // shipped from the host: both derive s and inv from the same shared
    // ga64_slice_max/ga64_recip sites, so a decline costs time, not codes.
    for (uint32_t m = 0; m < M; ++m) {
      const float* xr = x.data() + (size_t)m * K;
      if (act48_staging_.size() != (size_t)M * K)
        act48_staging_.resize((size_t)M * K);
      if (as48_staging_.size() != (size_t)M * nGA)
        as48_staging_.resize((size_t)M * nGA);
      std::int8_t* qrow = act48_staging_.data() + (size_t)m * K;
      float* srow = as48_staging_.data() + (size_t)m * nGA;
      // Trunk's original loop used std::lround (half away from zero).
      dashq::act_quant_ga64_batch(xr, qrow, srow, K,
                                  dashq::ActQuantRound::HalfAway);
      for (size_t d = 0; d < devices_.size(); ++d) {
        std::memcpy(ws48_[d].act + (size_t)m * ws48_[d].K, qrow, K);
        std::memcpy(ws48_[d].as + (size_t)m * ws48_[d].nGA, srow,
                    (size_t)nGA * sizeof(float));
      }
    }
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
    g_gpu_profile.copy_count++;
  }
#endif

  // 2) Launch per device.
  float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(devices_.size());
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    uint32_t N_local = end - start;
    gemm::Dpas48Params p;
    p.code_dpas = sc.code_usm[d];
    p.scale_usm = sc.scale_usm[d];
    p.act_int8 = ws48_[d].act;
    p.act_scale = ws48_[d].as;
    p.alpha = alpha;
    p.out_usm = ws_[d].out;
    p.M = M;
    p.N = N_local;
    p.K = K;
    p.group_size = sc.group_size;
    p.act_group = GA;
    p.bits = bits;
    p.act_stride = ws48_[d].K;
    p.as_stride = ws48_[d].nGA;
    sycl::event dep = preq_ok ? preq_events[d] : sycl::event{};
    // INTFER_DPAS48_FOLD (default OFF): hoist the per-(row,op) activation
    // column sums out of the 16-N tile loop. quant_gemm_dpas48 runs one WI per
    // (row, tile) and otherwise recomputes the identical O(K) integer fold in
    // every tile WI — the documented reason the wide-N DPAS kernel loses to the
    // legacy tiled path. Bit-identical: the values are exact int32 sums, and
    // the DPAS math, scales and accumulation order are untouched.
    // Declines (leaves p.fold_usm null => inline fold) whenever the workspace
    // or the op geometry is not usable, so the fallback is the old path.
    if (use_fold && ws48_[d].fold && K >= 32 && (K % 32) == 0 &&
        // 2*nOp per row, and the workspace must be wide enough for the
        // call's ops (not just its own max): a shorter stride would make
        // row m's Sfull block overwrite row m+1's S0 block.
        ws48_[d].nOp >= 2 * (K / 32)) {
      const uint32_t fold_stride = 2 * ws48_[d].nOp;
      sycl::event fold_evt = gemm::quant_gemm_dpas48_rowfold(
          q, ws48_[d].act, ws48_[d].fold, M, K, ws48_[d].K, fold_stride, dep);
      p.fold_usm = ws48_[d].fold;
      p.fold_stride = fold_stride;
      dep = fold_evt;
    }
#ifdef INTFER_PROFILE_GPU
    auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
    sycl::event kernel_evt = use_row ? gemm::quant_gemm_dpas4_row(q, p, dep)
                          : use_tiled ? gemm::quant_gemm_dpas48_2d(q, p, dep)
                                      : gemm::quant_gemm_dpas48(q, p, dep);
    // Decline detection. BOTH DPAS48 callees now decline the SAME way -- they return
    // `dep` UNCHANGED (quant_gemm.cpp) -- so `kernel_evt == dep` is the reliable
    // test. The `sycl::event{}` half is kept as defence in depth and NOT relied
    // on: measured on this toolchain, an event produced by `return sycl::event{}`
    // does not compare equal to a bare one built here, so it cannot be the
    // mechanism (dpas_fold_probe asserts that measurement in both directions).
    // What actually keeps a decline unreachable is the eligibility predicate the
    // caller checks BEFORE launching (gemm::dpas4_row_eligible, use_tiled's
    // M >= 2) -- one rule, one home, consulted before the launch rather than
    // inferred from a return value.
    const bool declined =
        (kernel_evt == sycl::event{}) || (kernel_evt == dep);
    if (declined) kernel_evt = gemm::quant_gemm_dpas48(q, p, dep);
    kernel_events.push_back(kernel_evt);
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
    }
    if (kernel_events.size() > g_gpu_profile.peak_wait_depth)
      g_gpu_profile.peak_wait_depth = kernel_events.size();
#endif
  }
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  for (auto& e : kernel_events) e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += devices_.size();
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif

  // 3) Read back FP16 -> FP32 (same as run_linear_tp tail).
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto [start, end] = ranges[d];
    const std::uint16_t* out_ptr = ws_[d].out;
    const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
    for (uint32_t m = 0; m < M; ++m)
      dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                 out.data() + (size_t)m * t.N + start,
                                 N_local_rb);
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
#endif
}

void Engine::run_linear_tp(int tensor_index, const std::vector<float>& x,
                           std::vector<float>& out, uint32_t M) {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size()) return;
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (M == 0) M = 1;
  const uint32_t out_elems = M * t.N;
  if (out.size() != out_elems) out.resize(out_elems);
 
  // Lazy-build the USM shard cache for this tensor on first call.
  if (!shard_cache_[tensor_index].built)
    ensure_shard_built(tensor_index);
  // Shared workspaces cover this call's (M, K, max-Nl) shape (grow-only;
  // the M>1 batched path writes M*K / M*N_local elements, so M=1 sizing
  // would OOB-write). Nl_max is over TP ranks (column sharding of N).
  {
    uint32_t Nl_max = 0;
    for (size_t d = 0; d < ranges.size(); ++d)
      Nl_max = std::max(Nl_max, ranges[d].second - ranges[d].first);
    ensure_workspace(M, t.K, Nl_max);
  }
  const auto& sc = shard_cache_[tensor_index];

  // TODO 6 (W4A8/W2A8) DPAS fast path: fused FP32->INT8 activation quant
  // (per-row, GA=64 slices) + native sub-byte DPAS kernel. Weights stay
  // bit-exact (affine fold); only the act quant is new error vs the
  // FP16-act path. INTFER_DPAS48=0 or ineligible shapes use the legacy path.
  if (sc.dpas48) {
    run_linear_tp_dpas48(tensor_index, x, out, M);
    return;
  }
 
  // Pre-convert FP32 activation to FP16 once (shared by all devices).
  // Staging buffer is reused across calls (resize only on shape growth).
#ifdef INTFER_PROFILE_GPU
  auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
  if (act_staging_.size() != (size_t)M * t.K) act_staging_.resize((size_t)M * t.K);
  std::vector<std::uint16_t>& act_fp16 = act_staging_;
  dashq::float_to_half_batch(x.data(), act_fp16.data(), (size_t)M * t.K);
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
 
    // 1) Write FP16 activation into the shared per-device workspace
    // (one wide memcpy, not scalar half-stores into malloc_shared).
    std::uint16_t* const act_ptr = ws_[d].act;
#ifdef INTFER_PROFILE_GPU
    auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
    std::memcpy(act_ptr, act_fp16.data(), (size_t)M * t.K * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
    }
#endif

    // 2) Launch the USM kernel on the persistent weight buffers.
    // Alpha comes from the host copy (kernel takes it by value; no device
    // alloc for 4 bytes).
    float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
    auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
    sycl::event kernel_evt = gemm::quant_gemm_forward_usm(q,
                                  ws_[d].act,
                                  sc.code_usm[d],
                                  sc.scale_usm[d],
                                  alpha,
                                  ws_[d].out,
                                  M, N_local, t.K,
                                  sc.group_size,
                                  sc.scheme,
                                  sycl::event{});
#ifdef INTFER_PROFILE_GPU
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
    }
#endif
#ifdef INTFER_PROFILE_GPU
    if (kernel_events.size() + 1 > g_gpu_profile.peak_wait_depth)
      g_gpu_profile.peak_wait_depth = kernel_events.size() + 1;
#endif
    kernel_events.push_back(kernel_evt);
  }
 
  // Wait for all kernels, then read back from the shared workspace.
  // NOTE: the wait is load-bearing, not profiling — without it the host
  // reads ws out before the kernel runs (fresh zero pages => all-zero
  // GEMM outputs). Only the timing accumulation is profiling-gated.
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  for (size_t d = 0; d < devices_.size(); ++d) {
    kernel_events[d].wait();
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += devices_.size();
  // Device-side kernel time from event profiling timestamps (the queue is
  // created with enable_profiling). This separates actual XMX/compute time
  // from host-side launch + wait overhead.
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }

  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto [start, end] = ranges[d];
    const std::uint16_t* out_ptr = ws_[d].out;
    const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
    for (uint32_t m = 0; m < M; ++m)
      dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                 out.data() + (size_t)m * t.N + start,
                                 N_local_rb);
  }
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
#endif
}

// Fused MLP down-projection (TP=1 only): gate/up convert + submit
// back-to-back, chain on-device through the fused silu-mul + FP16 convert
// straight into the down GEMM, then ONE host wait + ONE D2H. Saves vs the
// wave + host path: 1 wait, 2 D2H readbacks (gate/up), the host silu-mul
// + convert, and 1 H2D (down act). Bit-contract: the fused kernel is the
// same FP32 op order as the host loop; sycl::exp may differ from host
// expf by ~1ulp on rare elements, so the oracle token stream is the
// arbiter (same standing as the AVX2 host-math change). The FP16
// down-convert is RNE bit-identical to dashq::float_to_half by
// construction (dev_float_to_half); the CPU parity gate verifies it, the
// oracle gate otherwise. False (caller keeps host path) on TP>1, DPAS48
// items, bad shapes, or any missing index — never partial work.
bool Engine::run_mlp_fused(int gate_ti, int up_ti, int down_ti,
                           const std::vector<float>& rin,
                           std::vector<float>& md, uint32_t M) {
  if (devices_.size() != 1) return false;
  if (M == 0) M = 1;
  for (int ti : {gate_ti, up_ti, down_ti}) {
    if (ti < 0 || (size_t)ti >= model_.tensors.size()) return false;
    if (!shard_cache_[(size_t)ti].built && !ensure_shard_built(ti))
      return false;
    if (shard_cache_[(size_t)ti].dpas48) return false;
  }
  const auto& tg = model_.tensors[(size_t)gate_ti];
  const auto& tu = model_.tensors[(size_t)up_ti];
  const auto& td = model_.tensors[(size_t)down_ti];
  const auto& scg = shard_cache_[(size_t)gate_ti];
  const auto& scu = shard_cache_[(size_t)up_ti];
  const auto& scd = shard_cache_[(size_t)down_ti];
  // Geometry: gate/up share rin (K) and inter (N); down maps inter -> H.
  // Range-split sanity: TP=1 so each shard covers the full N.
  const uint32_t inter = tg.N;
  if (tu.N != inter || tu.K != tg.K) return false;
  if (td.K != inter) return false;
  if (rin.size() != (size_t)M * tg.K) return false;
  if (shards_[gate_ti].ranges[0].first != 0 ||
      shards_[gate_ti].ranges[0].second != inter) return false;
  if (shards_[up_ti].ranges[0].first != 0 ||
      shards_[up_ti].ranges[0].second != inter) return false;
  if (shards_[down_ti].ranges[0].first != 0 ||
      shards_[down_ti].ranges[0].second != td.N) return false;
  if (md.size() != (size_t)M * td.N) md.resize((size_t)M * td.N);

  sycl::queue& q = devices_[0].queue;
  // Phase 1: convert rin once per item into the wave slots (same convert as
  // run_linear_wave phase 1; two slots, K_max = tg.K).
  ensure_wave_workspace(M, 2, tg.K, std::max(inter, td.N));
#ifdef INTFER_PROFILE_GPU
  auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
  const size_t need = (size_t)M * tg.K;
  if (act_staging_.size() != need) act_staging_.resize(need);
  dashq::float_to_half_batch(rin.data(), act_staging_.data(), need);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
    g_gpu_profile.copy_count++;
  }
  auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
  std::memcpy(wv_[0].act, act_staging_.data(), need * sizeof(std::uint16_t));
  std::memcpy(wv_[1].act, act_staging_.data(), need * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
  }
#endif

  // Phase 2: submit gate + up back-to-back (in-order queue serializes;
  // explicit deps keep out-of-order queues correct too).
  const float ag = scg.has_global_divisor ? scg.global_host : 1.0f;
  const float au = scu.has_global_divisor ? scu.global_host : 1.0f;
  const float ad = scd.has_global_divisor ? scd.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
  auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
  sycl::event e_gate = gemm::quant_gemm_forward_usm(q, wv_[0].act,
      scg.code_usm[0], scg.scale_usm[0], ag, wv_[0].out,
      M, inter, tg.K, scg.group_size, scg.scheme, sycl::event{});
  sycl::event e_up = gemm::quant_gemm_forward_usm(q, wv_[1].act,
      scu.code_usm[0], scu.scale_usm[0], au, wv_[1].out,
      M, inter, tu.K, scu.group_size, scu.scheme, e_gate);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
  }
#endif

  // Phase 3: fused silu-mul + convert into the shared down-act workspace.
  ensure_workspace(M, td.K, td.N);
  gemm::SiluMulParams sp;
  sp.mg_fp16 = wv_[0].out;
  sp.mu_fp16 = wv_[1].out;
  sp.act_fp16 = ws_[0].act;
  sp.n = M * inter;
  sycl::event e_fused = gemm::silumul_to_half(q, sp, e_up);

  // Phase 4: down GEMM straight from the device act buffer, then the
  // single host wait + D2H of this whole chain.
  sycl::event e_down = gemm::quant_gemm_forward_usm(q, ws_[0].act,
      scd.code_usm[0], scd.scale_usm[0], ad, ws_[0].out,
      M, td.N, td.K, scd.group_size, scd.scheme, e_fused);
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  e_down.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += 1;
  for (sycl::event* e : {&e_gate, &e_up, &e_fused, &e_down}) {
    try {
      auto t0 = e->get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e->get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  const std::uint16_t* out_ptr = ws_[0].out;
  for (uint32_t m = 0; m < M; ++m)
    dashq::half_to_float_batch(out_ptr + (size_t)m * td.N,
                               md.data() + (size_t)m * td.N, td.N);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
#endif
  return true;
}

// Device-act GEMM seam (see engine.hpp): same FP16-act kernel as
// run_linear_tp but the M*K activation bits come from a caller-owned device
// buffer. Skips host FP32->FP16 convert + H2D; the dep chains after the
// producer kernel so a device-side chain pays one host wait total.
// INTFER_MLP_GU -- gate+up in one launch, TP=2-VALID, bit-exact. See
// engine.hpp for the contract and quant_gemm.cpp's quant_gemm_forward_usm_pair
// for the kernel and the "why this is NOT routed through run_mlp_fused" note.
bool Engine::run_linear_tp_pair(int ti_a, int ti_b,
                               const std::vector<float>& x,
                               std::vector<float>& out_a,
                               std::vector<float>& out_b) {
  // exactly `=1` enables, matching the INTFER_DPAS* opt-in convention
  const char* gu_env = std::getenv("INTFER_MLP_GU");
  static const bool on = (gu_env && gu_env[0] == '1' && gu_env[1] == '\0') &&
                         (std::getenv("INTFER_NO_MLP_GU") == nullptr);
  if (!on) return false;
  if (x.empty()) return false;
  if (ti_a < 0 || (size_t)ti_a >= model_.tensors.size()) return false;
  if (ti_b < 0 || (size_t)ti_b >= model_.tensors.size()) return false;
  const auto& ta = model_.tensors[(size_t)ti_a];
  const auto& tb = model_.tensors[(size_t)ti_b];
  // geometry must match or the single launch is not the same two GEMMs
  if (ta.N != tb.N || ta.K != tb.K) return false;
  if (ta.qcfg.group_size != tb.qcfg.group_size) return false;
  if (ta.qcfg.scheme != tb.qcfg.scheme) return false;
  const uint32_t M = 1, N = ta.N, K = ta.K;
  if (N == 0 || K == 0) return false;
  const auto sa_ti = (size_t)ti_a, sb_ti = (size_t)ti_b;
  if (!shard_cache_[sa_ti].built && !ensure_shard_built((int)sa_ti)) return false;
  if (!shard_cache_[sb_ti].built && !ensure_shard_built((int)sb_ti)) return false;
  const auto& sa = shard_cache_[sa_ti];
  const auto& sb = shard_cache_[sb_ti];
  // DPAS48 shards use a different kernel entirely; FP16 shards are memcpy.
  if (sa.dpas48 || sb.dpas48) return false;
  if (to_ws(ta.qcfg.scheme) != gemm::WeightScheme::INT2 &&
      to_ws(ta.qcfg.scheme) != gemm::WeightScheme::INT4) return false;
  if (sa.code_usm.empty() || sb.code_usm.empty()) return false;
  if (sa.scale_usm.empty() || sb.scale_usm.empty()) return false;
  if (sa.code_usm[0] == nullptr || sb.code_usm[0] == nullptr) return false;
  if (sa.scale_usm[0] == nullptr || sb.scale_usm[0] == nullptr) return false;
  if (sa.has_global_divisor != sb.has_global_divisor) return false;
  if (out_a.size() != (size_t)M * N || out_b.size() != (size_t)M * N) return false;

  // Column slices must be identical in extent, or the two operands are not the
  // same launch's worth of work.
  const auto& ra = shards_[sa_ti].ranges;
  const auto& rb = shards_[sb_ti].ranges;
  if (ra.size() != rb.size()) return false;
  for (size_t d = 0; d < ra.size(); ++d)
    if (ra[d].first != rb[d].first || ra[d].second != rb[d].second) return false;

  ensure_workspace(M, K, ra[0].second - ra[0].first);

  const std::uint16_t* act = ws_[0].act;
  dashq::float_to_half_batch(x.data(), (std::uint16_t*)act, K);

  const float aa = sa.has_global_divisor ? sa.global_host : 1.0f;
  const float ab = sb.has_global_divisor ? sb.global_host : 1.0f;
  sycl::queue& q = devices_[0].queue;
  sycl::event e = gemm::quant_gemm_forward_usm_pair(
      q, (std::uint16_t*)act,
      sa.code_usm[0], sa.scale_usm[0], aa, (std::uint16_t*)ws_[0].out,
      sb.code_usm[0], sb.scale_usm[0], ab,
      (std::uint16_t*)(ws_[0].out + (size_t)M * (ra[0].second - ra[0].first)),
      N, K, ta.qcfg.group_size, to_ws(ta.qcfg.scheme), sycl::event{});
  e.wait();
  // Both operands share ws_[0].out (sized for one N slice), so the second
  // output must be read out BEFORE the first is converted in place -- or rather,
  // both are read here, from distinct regions, before any host reuse.
  const std::size_t nl = (size_t)(ra[0].second - ra[0].first);
  const std::uint16_t* oa = ws_[0].out;
  const std::uint16_t* ob = oa + (size_t)M * nl;
  dashq::half_to_float_batch(oa, out_a.data(), (size_t)M * nl);
  dashq::half_to_float_batch(ob, out_b.data(), (size_t)M * nl);
  return true;
}

bool Engine::run_linear_tp_devact(int tensor_index,
                                  const std::uint16_t* dev_act,
                                  std::vector<float>& out, uint32_t M,
                                  const sycl::event& dep) {
  if (devices_.size() != 1) return false;
  if (!dev_act || M == 0) return false;
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return false;
  const auto& t = model_.tensors[(size_t)tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (ranges.size() != 1) return false;
  if (ranges[0].first != 0) return false;
  if (t.N == 0 || t.K == 0) return false;
  // No mutation on failure: the caller reuses its output buffer on the
  // host fallback path, so a shape mismatch must not resize it.
  if (out.size() != (size_t)M * t.N) return false;
  if (!shard_cache_[(size_t)tensor_index].built &&
      !ensure_shard_built(tensor_index))
    return false;
  const auto& sc = shard_cache_[(size_t)tensor_index];
  if (sc.dpas48) return false;  // FP16-act path only; host path covers DPAS
  if (sc.code_usm.empty() || !sc.code_usm[0]) return false;
  if (sc.scale_usm.empty() || !sc.scale_usm[0]) return false;
  {
    uint32_t Nl = ranges[0].second - ranges[0].first;
    if (Nl != t.N) return false;  // TP=1 full-row sanity
    ensure_workspace(M, t.K, Nl);
  }
  // Output staging reuses ws_[0].out: reject a dev_act that aliases it
  // (a caller chaining through the shared workspace would corrupt its own
  // input; dedicated buffers — the L4 pattern — never hit this). Checked
  // after ensure_workspace so the pointer is final.
  if (ws_[0].out && dev_act >= ws_[0].out &&
      dev_act < ws_[0].out + (size_t)ws_[0].M * ws_[0].Nl)
    return false;
  sycl::queue& q = devices_[0].queue;
  float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
  auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
  sycl::event e = gemm::quant_gemm_forward_usm(q,
      const_cast<std::uint16_t*>(dev_act),
      sc.code_usm[0], sc.scale_usm[0], alpha, ws_[0].out,
      M, t.N, t.K, sc.group_size, sc.scheme, dep);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
  }
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += 1;
  try {
    auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
    g_gpu_profile.kernel_count++;
  } catch (...) {}
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  const std::uint16_t* out_ptr = ws_[0].out;
  for (uint32_t m = 0; m < M; ++m)
    dashq::half_to_float_batch(out_ptr + (size_t)m * t.N,
                               out.data() + (size_t)m * t.N, t.N);
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_readback_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.readback_count++;
  }
#endif
  return true;
}

// Wave variant: one independent GEMM per item (see engine.hpp). Phases:
// (1) per item: lazy shard build + FP32->FP16 convert + per-device act
// memcpy into that item's wave slot; (2) per device: submit all items'
// kernels back-to-back; (3) wait once for the whole wave; (4) per item:
// read back. Bit-identical to sequential run_linear_tp calls: same shard
// bytes, same convert, same kernel args, same output assembly — only the
// submit/wait grouping changes, so the device runs the wave back-to-back
// and the host pays one round-trip per wave instead of per GEMM.
// DPAS48-flagged items fall back to sequential run_linear_tp (rare:
// DPAS is default-off; the wave's shared-slot layout targets the FP16-act
// path). Items with tensor_index < 0 are skipped.
void Engine::run_linear_wave(const LinearWaveItem* items, size_t n,
                             uint32_t M) {
  if (!items || n == 0) return;
  if (M == 0) M = 1;
  // Compact out skipped items (stable order = call order).
  const LinearWaveItem* live[16];
  size_t nl = 0;
  for (size_t i = 0; i < n && nl < 16; ++i)
    if (items[i].tensor_index >= 0 &&
        (size_t)items[i].tensor_index < model_.tensors.size() &&
        items[i].x && items[i].out)
      live[nl++] = &items[i];
  if (nl == 0) return;
  // DPAS48 fallback: keep the wave contract (same outputs) via the
  // sequential path. Bit-identical by construction.
  for (size_t i = 0; i < nl; ++i) {
    if (!shard_cache_[(size_t)live[i]->tensor_index].built)
      ensure_shard_built(live[i]->tensor_index);
    if (shard_cache_[(size_t)live[i]->tensor_index].dpas48) {
      for (size_t j = 0; j < nl; ++j)
        run_linear_tp(live[j]->tensor_index, *live[j]->x, *live[j]->out, M);
      return;
    }
  }
  // Phase 0: the wave workspace must cover every item's (M, K, Nl) shape
  // plus one slot per live item. Nl_max is over TP ranks AND items; K is
  // the max over items (K varies per tensor: H vs inter vs conv_dim...).
  uint32_t Nl_max = 0, K_max = 0;
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
    K_max = std::max(K_max, t.K);
    const auto& ranges = shards_[live[i]->tensor_index].ranges;
    for (size_t d = 0; d < ranges.size(); ++d)
      Nl_max = std::max(Nl_max, ranges[d].second - ranges[d].first);
  }
  ensure_wave_workspace(M, (uint32_t)nl, K_max, Nl_max);

  // Phase 1: per item — convert + write each device's wave slot.
  // Shared-input fast path: wave items often share one input vector
  // (gate/up share rin; M-row batch gate/up share X). The FP32->FP16
  // convert is a pure function of immutable wave inputs, so convert each
  // DISTINCT (pointer, bytes) input once and reuse the bytes for
  // duplicates — bit-identical (same input bytes by construction), one
  // convert + one host-side staging copy saved per duplicate. Slots stay
  // per-item (kernels read their own slot); only the convert dedupes.
  // Scope is this call only (local map — no cross-call staleness).
  std::vector<std::uint16_t> shared_staging;
  const float* shared_ptr = nullptr;
  size_t shared_need = 0;
  bool shared_valid = false;
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
    const std::vector<float>& x = *live[i]->x;
    std::vector<float>& out = *live[i]->out;
    const uint32_t out_elems = M * t.N;
    if (out.size() != out_elems) out.resize(out_elems);
    const size_t need = (size_t)M * t.K;
    const bool dup = shared_valid && x.data() == shared_ptr && need == shared_need;
    if (!dup) {
      // Is this input reused by a later item? Only then is staging worth saving.
      bool reused = false;
      for (size_t j = i + 1; j < nl; ++j) {
        const auto& tj = model_.tensors[(size_t)live[j]->tensor_index];
        if (live[j]->x->data() == x.data() && (size_t)M * tj.K == need) { reused = true; break; }
      }
#ifdef INTFER_PROFILE_GPU
      auto t_copy0 = std::chrono::high_resolution_clock::now();
#endif
      if (act_staging_.size() != need) act_staging_.resize(need);
      dashq::float_to_half_batch(x.data(), act_staging_.data(), need);
      if (reused) {
        if (shared_staging.size() != need) shared_staging.resize(need);
        std::memcpy(shared_staging.data(), act_staging_.data(), need * sizeof(std::uint16_t));
        shared_ptr = x.data();
        shared_need = need;
        shared_valid = true;
      } else {
        shared_valid = false;
      }
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_copy_us += std::chrono::duration<double, std::micro>(t1 - t_copy0).count();
        g_gpu_profile.copy_count++;
      }
      auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
      for (size_t d = 0; d < devices_.size(); ++d)
        std::memcpy(wv_[d * wv_W_ + i].act, act_staging_.data(),
                    (size_t)M * t.K * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
      }
#endif
    } else {
      // Duplicate: slots get the saved staging bytes (no convert).
#ifdef INTFER_PROFILE_GPU
      auto t_w0 = std::chrono::high_resolution_clock::now();
#endif
      for (size_t d = 0; d < devices_.size(); ++d)
        std::memcpy(wv_[d * wv_W_ + i].act, shared_staging.data(),
                    (size_t)M * t.K * sizeof(std::uint16_t));
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_usm_write_us += std::chrono::duration<double, std::micro>(t1 - t_w0).count();
      }
#endif
    }
  }

  // Phase 2: per device — submit all items' kernels back-to-back.
  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(devices_.size() * nl);
  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    sycl::queue& q = dev.queue;
    for (size_t i = 0; i < nl; ++i) {
      const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
      const auto& sc = shard_cache_[(size_t)live[i]->tensor_index];
      const auto& ranges = shards_[live[i]->tensor_index].ranges;
      auto [start, end] = ranges[d];
      auto N_local = end - start;
      float alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
#ifdef INTFER_PROFILE_GPU
      auto t_s0 = std::chrono::high_resolution_clock::now();
#endif
      sycl::event kernel_evt = gemm::quant_gemm_forward_usm(q,
                                    wv_[d * wv_W_ + i].act,
                                    sc.code_usm[d],
                                    sc.scale_usm[d],
                                    alpha,
                                    wv_[d * wv_W_ + i].out,
                                    M, N_local, t.K,
                                    sc.group_size,
                                    sc.scheme,
                                    sycl::event{});
#ifdef INTFER_PROFILE_GPU
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_gpu_profile.total_submit_us += std::chrono::duration<double, std::micro>(t1 - t_s0).count();
      }
      if (kernel_events.size() + 1 > g_gpu_profile.peak_wait_depth)
        g_gpu_profile.peak_wait_depth = kernel_events.size() + 1;
#endif
      kernel_events.push_back(kernel_evt);
    }
  }

  // Phase 3: wait once for the whole wave (load-bearing: host must not
  // read wave outputs before the kernels run).
#ifdef INTFER_PROFILE_GPU
  auto t_wb0 = std::chrono::high_resolution_clock::now();
#endif
  for (auto& e : kernel_events) e.wait();
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_waitblock_us += std::chrono::duration<double, std::micro>(t1 - t_wb0).count();
  }
  g_gpu_profile.wait_count += kernel_events.size();
  for (auto& e : kernel_events) {
    try {
      auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      g_gpu_profile.total_kernel_us += (double)(t1 - t0) / 1000.0;
      g_gpu_profile.kernel_count++;
    } catch (...) {}
  }
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif

  // Phase 4: per item — read back from that item's wave slot.
  for (size_t i = 0; i < nl; ++i) {
    const auto& t = model_.tensors[(size_t)live[i]->tensor_index];
    std::vector<float>& out = *live[i]->out;
    const auto& ranges = shards_[live[i]->tensor_index].ranges;
    for (size_t d = 0; d < devices_.size(); ++d) {
      auto [start, end] = ranges[d];
      const std::uint16_t* out_ptr = wv_[d * wv_W_ + i].out;
      const uint32_t N_local_rb = static_cast<uint32_t>(end - start);
      for (uint32_t m = 0; m < M; ++m)
        dashq::half_to_float_batch(out_ptr + (size_t)m * N_local_rb,
                                   out.data() + (size_t)m * t.N + start,
                                   N_local_rb);
    }
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
            g_gpu_profile.total_usm_write_us, g_gpu_profile.total_submit_us,
            g_gpu_profile.total_waitblock_us,
            g_gpu_profile.total_rebuild_us, g_gpu_profile.rebuild_count,
            g_gpu_profile.peak_wait_depth,
            g_gpu_profile.kernel_count, g_gpu_profile.wait_count,
            g_gpu_profile.copy_count, g_gpu_profile.readback_count};
  }
  void reset_gpu_profile() { g_gpu_profile = GpuProfile{}; }
  namespace gemm {
  void profile_note_launch() { g_gpu_profile.kernel_count++; }
  }  // namespace gemm
#endif

void Engine::run_linear_f16_tp(const sycl::half* x, const sycl::half* w,
                               float* out, uint32_t N, uint32_t K,
                               const sycl::event& copy_dep) {
  // One-off per-device weight buffer. We copy the full W to each device;
  // each device computes the full N rows; the host reads back halves from
  // device 0 and converts to float on the host (c_dev holds N halves, not
  // N floats — a direct N*sizeof(float) device read is a 2x OOB read).
  // Device frees are deferred until after all events complete (sycl::free
  // acts on the host immediately; freeing in-loop races the kernel).
  struct DevBufs { sycl::half *w_dev, *x_dev, *c_dev; sycl::queue* q; };
  std::vector<DevBufs> bufs;
  bufs.reserve(devices_.size());
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
    bufs.push_back({w_dev, x_dev, c_dev, &q});

    q.memcpy(w_dev, w, w_bytes);
    q.memcpy(x_dev, x, (size_t)K * sizeof(sycl::half)).wait();
    sycl::event evt = gemm::quant_gemm_f16(q, x_dev, w_dev, c_dev, 1, N, K);
    evts.push_back(evt);
  }
  for (auto& e : evts) e.wait();
  // Read back N halves from device 0 and convert on the host.
  std::vector<sycl::half> tmp(N);
  bufs[0].q->memcpy(tmp.data(), bufs[0].c_dev, (size_t)N * sizeof(sycl::half)).wait();
  for (uint32_t i = 0; i < N; ++i) out[i] = static_cast<float>(tmp[i]);
  for (auto& b : bufs) {
    sycl::free(b.w_dev, *b.q);
    sycl::free(b.x_dev, *b.q);
    sycl::free(b.c_dev, *b.q);
  }
}

const Rotation* Engine::rotation_for(int tensor_index) const {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return nullptr;
  const auto& t = model_.tensors[tensor_index];
  auto it = rid_.find(t.qcfg.rotation_id);
  return it == rid_.end() ? nullptr : it->second;
}

Engine::ShardView Engine::shard_view(int tensor_index) {
  ShardView v;
  if (devices_.size() != 1) return v;  // TP=1 full-row only
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size())
    return v;
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  if (ranges.size() != 1 || ranges[0].first != 0) return v;
  if (ranges[0].second - ranges[0].first != t.N) return v;  // full-row sanity
  if (t.N == 0 || t.K == 0) return v;
  if (!shard_cache_[(size_t)tensor_index].built &&
      !ensure_shard_built(tensor_index))
    return v;
  const auto& sc = shard_cache_[(size_t)tensor_index];
  if (sc.code_usm.empty() || !sc.code_usm[0]) return v;
  if (sc.scale_usm.empty() || !sc.scale_usm[0]) return v;
  v.ok = true;
  v.code = sc.code_usm[0];
  v.scale = sc.scale_usm[0];
  v.alpha = sc.has_global_divisor ? sc.global_host : 1.0f;
  v.N = t.N;
  v.K = t.K;
  v.group_size = sc.group_size;
  v.scheme = sc.scheme;
  v.dpas48 = sc.dpas48;
  return v;
}

bool Engine::ensure_shard_built(int tensor_index) {
  if (tensor_index < 0 || (size_t)tensor_index >= model_.tensors.size()) return false;
  auto& sc = shard_cache_[tensor_index];
  if (sc.built) return true;

#ifdef INTFER_PROFILE_GPU
  auto t_rb0 = std::chrono::high_resolution_clock::now();
#endif
  const auto& t = model_.tensors[tensor_index];
  const auto& ranges = shards_[tensor_index].ranges;
  sc.built = true;
  sc.scheme = to_ws(t.qcfg.scheme);
  sc.group_size = t.qcfg.group_size;
  sc.has_global_divisor = t.qcfg.has_global_divisor;
  sc.K = t.K;
  sc.code_usm.resize(devices_.size());
  sc.scale_usm.resize(devices_.size());
  // Host-cache gate (see below): FP16 is memcpy-identity + resident, so only
  // cache the schemes whose shard_weight() does scalar unpack work (INT4/8/2).
  // skip_host_cache_ (preload_all): pinned shards are never rebuilt, so the
  // cache would be dead host RAM (~9.4GB on 27B-sc) — skip populating it.
  const bool cacheable =
      (sc.scheme != gemm::WeightScheme::FP16) && !skip_host_cache_;

  for (size_t d = 0; d < devices_.size(); ++d) {
    auto& dev = devices_[d];
    auto [start, end] = ranges[d];
    sycl::queue& q = dev.queue;
    uint32_t N_local = end - start;

    // 1) Host-side shard data: repack-once cache for non-FP16 schemes.
    //    shard_weight() is a scalar unpack+repack, immutable per (tensor,
    //    rank) — run it only on the first build, then memcpy from the cache
    //    (~8.8GB for the 27B INT4 set). FP16 is memcpy-identity and stays
    //    resident since Phase 1, so caching it would only waste host RAM
    //    (it OOMs: +13.5GB). The cacheable gate is computed before the loop.
    const std::uint8_t* code_ptr;
    size_t code_bytes;
    const std::uint16_t* scale_ptr;
    size_t scale_bytes;
    const float* global_ptr = nullptr;
    gemm::GemmWeightBuffers w;
    if (sc.has_host_cache && cacheable) {
      code_ptr = sc.code_host[d].data();
      code_bytes = sc.code_host[d].size();
      scale_ptr = sc.scale_host[d].data();
      scale_bytes = sc.scale_host[d].size() * sizeof(uint16_t);
      if (t.global_bytes > 0) global_ptr = &sc.global_host;
    } else {
      w = shard_weight(model_, t, start, end, q);
      code_ptr = w.code_store.data();
      code_bytes = w.code_store.size();
      scale_ptr = w.scale_store.data();
      scale_bytes = w.scale_store.size() * sizeof(uint16_t);
      if (t.global_bytes > 0) {
        float alpha_val = 0.f;
        std::memcpy(&alpha_val, model_.global_data() + t.global_offset,
                    std::min<uint32_t>(t.global_bytes, 4u));
        sc.global_host = alpha_val;
        global_ptr = &sc.global_host;
      }
    }

    // 2) Allocate persistent USM on each device and copy weight data.
    //    On first build, retain the shard_weight() output in the host cache.
    //    TODO 6 (W4A8/W2A8): DPAS-eligible shards repack row-major codes to
    //    vertical-B tiles ONCE here (repack_dpas412); the kernel then reads
    //    code_usm[d] directly. dpas_bytes[d] records the USM size so
    //    release_shard pools by exact size (code_host keeps the row-major
    //    bytes for rebuilds — repack re-runs from them, still no scalar
    //    unpack repetition of the group-major payload).
    sc.N_local = N_local;
    if (!sc.has_host_cache && cacheable) {
      if (sc.code_host.size() != devices_.size()) {
        sc.code_host.resize(devices_.size());
        sc.scale_host.resize(devices_.size());
      }
      sc.code_host[d] = w.code_store;
      sc.scale_host[d] = w.scale_store;
    }
    if (d == 0) {
      sc.dpas48 = dpas48_eligible(t.qcfg.scheme, t.qcfg.group_size,
                                  N_local, t.K) &&
                  (ranges.size() == devices_.size());
      // All ranks share N_local only for even splits; re-check per rank
      // below (N_local varies per d on asymmetric lm_head-style ranges —
      // run_linear shards are even, but verify anyway).
      for (size_t dd = 0; dd < devices_.size(); ++dd) {
        uint32_t Nl_dd = ranges[dd].second - ranges[dd].first;
        if (!dpas48_eligible(t.qcfg.scheme, t.qcfg.group_size, Nl_dd, t.K)) {
          sc.dpas48 = false;
          break;
        }
      }
      if (sc.dpas48) sc.dpas_bytes.assign(devices_.size(), 0);
    }

    // Code: packed bytes (pool: same shapes recur every layer).
    if (sc.dpas48) {
      const int bits = (sc.scheme == gemm::WeightScheme::INT2)    ? 2
                       : (sc.scheme == gemm::WeightScheme::INT4)  ? 4
                                                                 : 8;
      std::vector<std::uint8_t> dpas_tiles;
      if (!gemm::repack_dpas412(code_ptr, N_local, t.K, bits, dpas_tiles)) {
        // Shape refused (should not happen after dpas48_eligible, but
        // safe): fall back to the row-major path for this shard.
        sc.dpas48 = false;
        sc.code_usm[d] =
            static_cast<std::uint8_t*>(pool_acquire(d, code_bytes));
        std::memcpy(sc.code_usm[d], code_ptr, code_bytes);
      } else {
        sc.dpas_bytes[d] = dpas_tiles.size();
        sc.code_usm[d] = static_cast<std::uint8_t*>(
            pool_acquire(d, dpas_tiles.size()));
        std::memcpy(sc.code_usm[d], dpas_tiles.data(), dpas_tiles.size());
      }
    } else {
      sc.code_usm[d] =
          static_cast<std::uint8_t*>(pool_acquire(d, code_bytes));
      std::memcpy(sc.code_usm[d], code_ptr, code_bytes);
    }

    // Scale: N_local * n_groups FP16
    sc.scale_usm[d] = static_cast<std::uint16_t*>(pool_acquire(d, scale_bytes));
    std::memcpy(sc.scale_usm[d], scale_ptr, scale_bytes);
    // NOTE: no global_usm alloc — alpha stays host-side (global_host) and is
    // passed to the kernel by value (see run_linear_tp).

    // NOTE: no per-shard act/out — run_linear_tp writes into the shared
    // per-device workspaces (ensure_workspace, grow-only).
  }
  if (cacheable) sc.has_host_cache = true;
#ifdef INTFER_PROFILE_GPU
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    g_gpu_profile.total_rebuild_us += std::chrono::duration<double, std::micro>(t1 - t_rb0).count();
    g_gpu_profile.rebuild_count++;
  }
#endif
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
  std::cerr << "Engine: loaded " << lang_.size() << " safetensors tensors\n";
  return true;
}

}  // namespace intfer


