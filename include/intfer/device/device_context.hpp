#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifdef INTFER_HAS_L0
#include <level_zero/ze_api.h>
#endif

#include <sycl/sycl.hpp>

namespace intfer::device {

// Per-GPU execution context. Wraps both the SYCL objects used for kernels and
// the native Level-Zero handles required for low-level control (command lists,
// explicit memory management, tensor-parallel sharding across the 2x B580).
struct DeviceContext {
  int index = -1;
  std::string name;

  sycl::device  sycl_device;
  sycl::context sycl_context;
  sycl::queue   queue;

#ifdef INTFER_HAS_L0
  ze_driver_handle_t  ze_driver  = nullptr;
  ze_device_handle_t  ze_device  = nullptr;
  ze_context_handle_t ze_context = nullptr;
  bool has_l0() const { return ze_device != nullptr; }
#else
  bool has_l0() const { return false; }
#endif

  uint64_t global_memory_bytes = 0;
  uint32_t max_compute_units   = 0;
};

// Discovers the local GPU fleet and owns one DeviceContext per device.
// Designed for tensor-parallel inference across the 2x Intel Arc B580.
class DeviceManager {
 public:
  // Discover GPUs whose device name contains `name_filter` (default "Arc").
  static DeviceManager discover(const std::string& name_filter = "Arc");

  size_t size() const { return devices_.size(); }

  const DeviceContext& at(size_t i) const { return devices_.at(i); }
  DeviceContext&       at(size_t i)       { return devices_.at(i); }
  const DeviceContext& operator[](size_t i) const { return devices_[i]; }
  DeviceContext&       operator[](size_t i)       { return devices_[i]; }

  // Split a dimension `total` into contiguous, balanced shards (one per device).
  // Returns [start, end) ranges suitable for column/row tensor sharding.
  std::vector<std::pair<size_t, size_t>> shard(size_t total) const;

  void print_summary() const;

 private:
  std::vector<DeviceContext> devices_;
};

}  // namespace intfer::device
