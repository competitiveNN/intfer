cmake_minimum_required(VERSION 3.15)
project(npu_probe CXX)

add_executable(npu_probe main.cpp)
set_target_properties(npu_probe PROPERTIES CXX_STANDARD 17)

# INTFER_ENABLE_NPU gates build-time support for the host OpenVINO
# libraries (libopenvino_c.so + libopenvino_intel_npu_plugin.so). They live
# under /run/host/usr/lib64 on this box, so the build stays self-contained:
# the link is purely optional and the binary keeps working when they are absent
# (it then reports the NPU as absent rather than failing to link).
option(INTFER_ENABLE_NPU "Enable NPU (OpenVINO) probe support" ON)

if(INTFER_ENABLE_NPU)
  target_compile_definitions(npu_probe PRIVATE INTFER_ENABLE_NPU=1)
  target_link_libraries(npu_probe PRIVATE dl)
endif()