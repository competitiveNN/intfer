#pragma once
#include <cstdint>

namespace intfer::gemm::tile_scheduler {

// SLM work-stealing tile scheduler for the prefill tiled DPAS kernels.
constexpr int kMaxTiles = 4096;

}  // namespace intfer::gemm::tile_scheduler