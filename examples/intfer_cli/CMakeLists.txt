add_executable(intfer_cli main.cpp)
target_link_libraries(intfer_cli PRIVATE intfer_engine intfer_quant_core intfer_common)
if(INTFER_ENABLE_L0)
  target_link_libraries(intfer_cli PRIVATE intfer_l0)
endif()
# Define INTFER_HAS_GPU_BACKEND so the CLI builds the SYCL Engine dispatch.
# The Engine itself is always SYCL-compiled; if no GPU is present at runtime
# Engine::load returns null and the CLI falls back to the CPU backend.
target_compile_definitions(intfer_cli PRIVATE INTFER_HAS_GPU_BACKEND=1)
if(INTFER_PROFILE_GPU)
  target_compile_definitions(intfer_cli PRIVATE INTFER_PROFILE_GPU)
endif()
add_sycl_to_target(TARGET intfer_cli SOURCES main.cpp)
