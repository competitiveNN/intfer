add_executable(dpas_fold_probe main.cpp)
target_link_libraries(dpas_fold_probe PRIVATE intfer_gemm intfer_common)
add_sycl_to_target(TARGET dpas_fold_probe SOURCES main.cpp)
