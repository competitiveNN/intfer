add_executable(slm_probe main.cpp)
target_link_libraries(slm_probe PRIVATE intfer_device intfer_gemm intfer_common)
add_sycl_to_target(TARGET slm_probe SOURCES main.cpp)