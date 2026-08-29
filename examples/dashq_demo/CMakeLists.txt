# examples/dashq_demo/CMakeLists.txt
# Standalone, SYCL-free build of the DASH-Q PTQ solver demo (t4). This target
# deliberately avoids the oneAPI/DPC++ toolchain so t4 is independently
# buildable/testable on any host with a C++20 compiler. Configure from the repo
# root (cmake -S . -B build) or standalone (cmake -S examples/dashq_demo -B build_dashq).
cmake_minimum_required(VERSION 3.16)
project(dashq_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Resolve the repo root relative to THIS CMakeLists file so it works no matter
# which directory cmake was invoked from.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(dashq_demo
  main.cpp
  ${REPO_ROOT}/src/quant/dashq.cpp
  ${REPO_ROOT}/src/quant/quant_spec.cpp)
target_include_directories(dashq_demo PRIVATE ${REPO_ROOT}/include)
