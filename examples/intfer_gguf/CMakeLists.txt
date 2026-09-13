cmake_minimum_required(VERSION 3.16)
project(intfer_gguf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(intfer_gguf
  main.cpp
  ${REPO_ROOT}/src/quant/output.cpp
  ${REPO_ROOT}/src/quant/dashq.cpp
  ${REPO_ROOT}/src/quant/quant_spec.cpp
  ${REPO_ROOT}/src/spinquant.cpp
  ${REPO_ROOT}/src/common/logger.cpp)
target_include_directories(intfer_gguf PRIVATE ${REPO_ROOT}/include)
