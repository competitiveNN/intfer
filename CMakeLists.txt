cmake_minimum_required(VERSION 3.20)

project(intfer LANGUAGES CXX)

# ---------------------------------------------------------------------------
# oneAPI DPC++ / SYCL toolchain (IntelSYCL package ships with the compiler)
# ---------------------------------------------------------------------------
find_package(IntelSYCL REQUIRED)

if(NOT CMAKE_CXX_COMPILER MATCHES "icpx|dpcpp|icx|clang\\+\\+")
  message(WARNING
    "intfer expects the oneAPI DPC++ compiler (icpx). Current compiler is "
    "${CMAKE_CXX_COMPILER}. Use scripts/build.sh to select it.")
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(INTFER_ENABLE_L0      "Enable the Intel Level-Zero native layer" ON)
option(INTFER_BUILD_EXAMPLES "Build example programs"                   ON)
option(INTFER_BUILD_QUANT    "Build Stage-1 quantization core (experimental)" OFF)
option(INTFER_PROFILE_GPU    "Enable GPU timing instrumentation (experimental)" ON)

set(INTFER_SYCL_TARGET "spir64" CACHE STRING "SYCL offload target triple")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release (-O3) when no build type is given: icx/icpx default to
# -O0, which makes the CPU reference path ~10x slower than it should be.
# NOTE: keep the cache entry untouched when the user passes -DCMAKE_BUILD_TYPE
# explicitly; also restore the built-in -O3 -DNDEBUG defaults, which some
# toolchain files (IntelSYCL) clear from the cache.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
foreach(_bt RELEASE DEBUG MINSIZEREL RELWITHDEBINFO)
  if("${CMAKE_CXX_FLAGS_${_bt}}" STREQUAL "")
    set(_def "")
    if(_bt STREQUAL "RELEASE")
      set(_def "-O3 -DNDEBUG")
    elseif(_bt STREQUAL "DEBUG")
      set(_def "-g")
    elseif(_bt STREQUAL "MINSIZEREL")
      set(_def "-Os -DNDEBUG")
    else()
      set(_def "-O2 -g -DNDEBUG")
    endif()
    set(CMAKE_CXX_FLAGS_${_bt} "${_def}" CACHE STRING "C++ flags for ${_bt}" FORCE)
  endif()
endforeach()

# Embed the oneAPI runtime lib dir into the binary RPATH so the executable runs
# without manually exporting LD_LIBRARY_PATH / sourcing setvars.sh.
get_filename_component(_icpx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_icpx_root "${_icpx_bin}/.." ABSOLUTE)
set(_sycl_lib_dir "${_icpx_root}/lib")
if(EXISTS "${_sycl_lib_dir}/libsycl.so")
  set(CMAKE_BUILD_RPATH "${_sycl_lib_dir}")
  add_link_options("-Wl,-rpath,${_sycl_lib_dir}")
endif()

if(INTFER_ENABLE_L0)
  find_library(ZE_LOADER NAMES ze_loader)
  if(NOT ZE_LOADER)
    message(WARNING "libze_loader not found; disabling Level-Zero layer.")
    set(INTFER_ENABLE_L0 OFF)
  endif()
endif()

add_subdirectory(src)
if(INTFER_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

