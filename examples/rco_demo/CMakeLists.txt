# examples/rco_demo/CMakeLists.txt
# Standalone, SYCL-free build of the RCO bit-budget search (t5). Deliberately
# avoids the oneAPI/DPC++ toolchain so t5 is independently buildable/testable on
# any host with a C++20 compiler. Configure standalone:
#   cmake -S examples/rco_demo -B build_rco && cmake --build build_rco
cmake_minimum_required(VERSION 3.16)
project(rco_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(rco_demo
  main.cpp
  ${REPO_ROOT}/src/quant/rco.cpp
  ${REPO_ROOT}/src/quant/dashq.cpp
  ${REPO_ROOT}/src/quant/quant_spec.cpp)
target_include_directories(rco_demo PRIVATE ${REPO_ROOT}/include)
