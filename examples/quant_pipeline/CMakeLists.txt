# examples/quant_pipeline/CMakeLists.txt
# Standalone, SYCL-free build of the full Stage-1 INT2 pipeline + GGUF / native
# SYCL output (t6). Deliberately avoids the oneAPI/DPC++ toolchain so the whole
# SpinQuant -> DASH-Q -> RCO -> serialize path is buildable/testable on any host
# with a C++20 compiler. Configure standalone:
#   cmake -S examples/quant_pipeline -B build_pipeline && cmake --build build_pipeline
cmake_minimum_required(VERSION 3.16)
project(quant_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(quant_pipeline
  main.cpp
  ${REPO_ROOT}/src/quant/pipeline.cpp
  ${REPO_ROOT}/src/quant/output.cpp
  ${REPO_ROOT}/src/quant/rco.cpp
  ${REPO_ROOT}/src/quant/dashq.cpp
  ${REPO_ROOT}/src/quant/quant_spec.cpp
  ${REPO_ROOT}/src/spinquant.cpp
  ${REPO_ROOT}/src/common/logger.cpp)
target_include_directories(quant_pipeline PRIVATE ${REPO_ROOT}/include)
