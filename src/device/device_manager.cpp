#include "intfer/device/device_context.hpp"

#include "intfer/common/logger.hpp"

#ifdef INTFER_HAS_L0
#include "intfer/l0/level_zero.hpp"
#endif

#include <sycl/sycl.hpp>

#include <algorithm>
#include <string>

namespace intfer::device {

namespace {

void sycl_exception_handler(sycl::exception_list exceptions) {
  for (const auto& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (const std::exception& ex) {
      INTFER_LOG_ERROR("SYCL async exception: %s", ex.what());
    }
  }
}

}  // namespace

DeviceManager DeviceManager::discover(const std::string& name_filter) {
  DeviceManager mgr;

#ifdef INTFER_HAS_L0
  auto l0_gpus = l0::LevelZero::enumerate_gpus();
#endif

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int idx = 0;
  for (const auto& dev : devices) {
    // Only the native Level-Zero backend is used: it matches the target
    // hardware path (OpenCL exposes the same physical GPUs under a different
    // backend and would double-count them).
    if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    const std::string name = dev.get_info<sycl::info::device::name>();
    if (name.find(name_filter) == std::string::npos) continue;

    DeviceContext ctx;
    ctx.index                = idx++;
    ctx.name                 = name;
    ctx.sycl_device          = dev;
    ctx.sycl_context         = sycl::context(dev);
    ctx.queue                = sycl::queue(ctx.sycl_context, dev, sycl_exception_handler, {sycl::property::queue::in_order()});
    ctx.global_memory_bytes  = dev.get_info<sycl::info::device::global_mem_size>();
    ctx.max_compute_units    = dev.get_info<sycl::info::device::max_compute_units>();

#ifdef INTFER_HAS_L0
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    for (const auto& g : l0_gpus) {
      if (g.device == ze_dev) {
        ctx.ze_driver  = g.driver;
        ctx.ze_device  = g.device;
        ctx.ze_context = g.context;
        break;
      }
    }
#endif

    mgr.devices_.push_back(std::move(ctx));
  }

  if (mgr.devices_.empty()) {
    INTFER_LOG_WARN("No devices matching filter '%s' found", name_filter.c_str());
  }
  return mgr;
}

std::vector<std::pair<size_t, size_t>> DeviceManager::shard(size_t total) const {
  std::vector<std::pair<size_t, size_t>> ranges;
  const size_t n = devices_.size();
  if (n == 0) return ranges;

  const size_t base = total / n;
  const size_t rem  = total % n;
  size_t start = 0;
  for (size_t i = 0; i < n; ++i) {
    const size_t count = base + (i < rem ? 1 : 0);
    ranges.emplace_back(start, start + count);
    start += count;
  }
  return ranges;
}

void DeviceManager::print_summary() const {
  INTFER_LOG_INFO("=== intfer device topology (%zu device(s)) ===", devices_.size());
  for (const auto& d : devices_) {
    const double gb = static_cast<double>(d.global_memory_bytes) /
                     (1024.0 * 1024.0 * 1024.0);
    INTFER_LOG_INFO("  [%d] %s  CU=%u  VRAM=%.2f GiB  L0=%s",
                    d.index, d.name.c_str(), d.max_compute_units, gb,
                    d.has_l0() ? "yes" : "no");
  }
}

}  // namespace intfer::device
