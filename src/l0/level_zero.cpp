#include "intfer/l0/level_zero.hpp"

#include "intfer/common/logger.hpp"

#include <level_zero/ze_api.h>

#include <vector>

namespace intfer::l0 {

bool LevelZero::initialized_ = false;

bool LevelZero::initialize() {
  if (initialized_) return true;
  ze_result_t r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (r != ZE_RESULT_SUCCESS) {
    INTFER_LOG_WARN("zeInit failed (0x%x)", static_cast<unsigned>(r));
    return false;
  }
  initialized_ = true;
  return true;
}

std::vector<L0Gpu> LevelZero::enumerate_gpus() {
  std::vector<L0Gpu> gpus;
  if (!initialize()) return gpus;

  uint32_t ndrv = 0;
  zeDriverGet(&ndrv, nullptr);
  std::vector<ze_driver_handle_t> drivers(ndrv);
  zeDriverGet(&ndrv, drivers.data());

  for (ze_driver_handle_t drv : drivers) {
    uint32_t ndev = 0;
    zeDeviceGet(drv, &ndev, nullptr);
    std::vector<ze_device_handle_t> devs(ndev);
    zeDeviceGet(drv, &ndev, devs.data());

    for (ze_device_handle_t dev : devs) {
      ze_device_properties_t props{};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      zeDeviceGetProperties(dev, &props);
      if (props.type != ZE_DEVICE_TYPE_GPU) continue;

      ze_context_desc_t cd{};
      cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
      ze_context_handle_t ctx = nullptr;
      if (zeContextCreate(drv, &cd, &ctx) != ZE_RESULT_SUCCESS) continue;

      L0Gpu g;
      g.driver  = drv;
      g.device  = dev;
      g.context = ctx;
      g.name    = props.name;

      uint32_t nm = 0;
      zeDeviceGetMemoryProperties(dev, &nm, nullptr);
      std::vector<ze_device_memory_properties_t> mp(nm);
      for (auto& m : mp) m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
      zeDeviceGetMemoryProperties(dev, &nm, mp.data());
      for (const auto& m : mp) g.total_memory += m.totalSize;

      gpus.push_back(std::move(g));
    }
  }
  return gpus;
}

void LevelZero::finalize() {
  // Contexts are intentionally kept alive for the process lifetime.
  initialized_ = false;
}

}  // namespace intfer::l0
