cmake_minimum_required(VERSION 3.16)
project(intfer_fidelity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(intfer_fidelity
  main.cpp
  ${REPO_ROOT}/src/quant/output.cpp
  ${REPO_ROOT}/src/quant/dashq.cpp
  ${REPO_ROOT}/src/quant/quant_spec.cpp
  ${REPO_ROOT}/src/quant/safetensors.cpp
  ${REPO_ROOT}/src/spinquant.cpp
  ${REPO_ROOT}/src/common/logger.cpp)
target_include_directories(intfer_fidelity PRIVATE ${REPO_ROOT}/include)
