#pragma once

#include <level_zero/ze_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace intfer::l0 {

// A discovered Intel GPU exposed through the Level-Zero loader.
struct L0Gpu {
  ze_driver_handle_t   driver   = nullptr;
  ze_device_handle_t   device   = nullptr;
  ze_context_handle_t  context  = nullptr;
  std::string          name;
  uint64_t             total_memory = 0;  // bytes
};

// Thin RAII-free wrapper around the Level-Zero driver/device enumeration.
// Used to obtain native handles for the SYCL device manager and for any
// downstream low-level work (command lists, tensor-parallel sharding).
class LevelZero {
 public:
  // zeInit(ZE_INIT_FLAG_GPU_ONLY); idempotent.
  static bool initialize();

  // Enumerate GPU devices (one ze_context created per device).
  static std::vector<L0Gpu> enumerate_gpus();

  // Intentionally a no-op: contexts are kept alive for the process lifetime.
  static void finalize();

 private:
  static bool initialized_;
};

}  // namespace intfer::l0
