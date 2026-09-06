if(INTFER_BUILD_QUANT)
# SpinQuant (t3) self-test — host only, no GPU required.
add_executable(intfer_spinquant_demo main.cpp)
target_include_directories(intfer_spinquant_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(intfer_spinquant_demo PRIVATE intfer_quant intfer_common)
endif()
