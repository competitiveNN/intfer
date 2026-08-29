# Core logging / error utilities (header-only logic, one translation unit).
add_library(intfer_common STATIC common/logger.cpp)
target_include_directories(intfer_common PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(intfer_common PUBLIC IntelSYCL::SYCL_CXX)
add_sycl_to_target(TARGET intfer_common SOURCES common/logger.cpp)

# Optional Intel Level-Zero native layer.
if(INTFER_ENABLE_L0)
  add_library(intfer_l0 STATIC l0/level_zero.cpp)
  target_include_directories(intfer_l0 PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(intfer_l0 PUBLIC IntelSYCL::SYCL_CXX ${ZE_LOADER})
  target_compile_definitions(intfer_l0 PUBLIC INTFER_HAS_L0=1)
  add_sycl_to_target(TARGET intfer_l0 SOURCES l0/level_zero.cpp)
endif()

# Device discovery / tensor-parallel context (SYCL + optional L0).
add_library(intfer_device STATIC device/device_manager.cpp)
target_include_directories(intfer_device PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(intfer_device PUBLIC intfer_common)
if(INTFER_ENABLE_L0)
  target_link_libraries(intfer_device PUBLIC intfer_l0)
  target_compile_definitions(intfer_device PUBLIC INTFER_HAS_L0=1)
endif()
add_sycl_to_target(TARGET intfer_device SOURCES device/device_manager.cpp)

# ---------------------------------------------------------------------------
# Stage-2 fused FlashAttention kernels (SYCL, Battlemage/Xe-core tuned).
# Always built: this is the attention engine, independent of Stage-1 quant.
# ---------------------------------------------------------------------------
add_library(intfer_attention STATIC attention/flash_attention.cpp)
target_include_directories(intfer_attention PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(intfer_attention PUBLIC intfer_common)
add_sycl_to_target(TARGET intfer_attention SOURCES attention/flash_attention.cpp)

# ---------------------------------------------------------------------------
# Stage-2 low-level SYCL quantized GEMM kernels (XMX / DPAS).
# Always built: the dequant-on-the-fly GEMM plus an ESIMD DPAS INT8xINT8 kernel
# that issues the Intel XMX systolic-array instruction directly.
# ---------------------------------------------------------------------------
add_library(intfer_gemm STATIC gemm/quant_gemm.cpp)
target_include_directories(intfer_gemm PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(intfer_gemm PUBLIC intfer_common)
add_sycl_to_target(TARGET intfer_gemm SOURCES gemm/quant_gemm.cpp)

# ---------------------------------------------------------------------------
# Stage-1 quantization core (host-only, no SYCL needed for preprocessing):
#   - quant_spec.cpp : Tensor-IR contract helpers (docs/quant_spec.md)
#   - spinquant.cpp  : t3 SpinQuant orthogonal rotation preprocessing
#   - dashq.cpp/rco.cpp : t4/t5 DASH-Q + RCO solver (under src/quant/)
# Built only when INTFER_BUILD_QUANT=ON (experimental; not required for t2/t8).
# ---------------------------------------------------------------------------
if(INTFER_BUILD_QUANT)
  add_library(intfer_quant STATIC quant_spec.cpp spinquant.cpp
             quant/dashq.cpp quant/rco.cpp
             quant/pipeline.cpp quant/output.cpp)
  target_include_directories(intfer_quant PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(intfer_quant PUBLIC intfer_common)
endif()
