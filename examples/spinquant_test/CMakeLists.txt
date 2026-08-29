add_executable(spinquant_test main.cpp)
target_link_libraries(spinquant_test PRIVATE intfer_quant intfer_common)
add_sycl_to_target(TARGET spinquant_test SOURCES main.cpp)
