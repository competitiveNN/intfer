# PLAN.md 1.3 (SLM work-stealing tile scheduler) and 1.4 (oneMKL SYCL BLAS
# fallback). Both are wired here, next to the target that consumes them, so
# they are part of the build rather than committed-but-uncompiled scaffolding.
#
# This subdir is processed AFTER intfer_gemm is defined (see src/CMakeLists.txt),
# so both targets below link BACK into intfer_gemm. That is deliberate: the
# symbols they export (tile_scheduler's namespace, gemm_onemkl_fp16 /
# onemkl_available) are called from quant_gemm.cpp, which is in intfer_gemm.
# A static library only pulls in the object files that resolve its undefined
# symbols, so without this link edge the onemkl fallback spellings would be
# real code that the linker never reaches -- an arm that compiles but cannot
# run, which is the exact "default-OFF flag no gate can reach" class the flag
# census exists to kill.

# ---------------------------------------------------------------------------
# 1.3 SLM work-stealing tile scheduler (header-only dispatch helper).
# It is an INTERFACE target (no sources, no output): its only job is to make
# the include path explicit and to be NAMED by a live build target so the
# header is compiled-by-reference rather than committed-and-orphaned. The
# runtime flag is INTFER_TILE_SCHED; the header is always available so the
# tiled DPAS kernel can #include it without a build-time branch.
add_library(intfer_tile_scheduler INTERFACE)
target_include_directories(intfer_tile_scheduler INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(intfer_tile_scheduler INTERFACE intfer_gemm)

# ---------------------------------------------------------------------------
# 1.4 oneMKL SYCL BLAS fallback for large contiguous FP16.
# gemm_onemkl.cpp carries BOTH the real oneMKL body (under
# #ifdef INTFER_ENABLE_MKL) and the empty-fallback spellings of
# onemkl_available() / gemm_onemkl_fp16() (under #else), so the engine links
# either way and the arm is unreachable rather than wrong when the library
# is absent. It is ALWAYS compiled; the #ifdef only decides which body.
# INTFER_ENABLE_MKL is OFF by default: the arm is a tolerance-gated opt-in
# (PLAN.md 1.4) and the default build must be byte-for-byte the MKL-free path.
# When the library is absent find_library fails and the option is forced
# OFF, so the MKL header is not pulled in and the fallback spellings compile.
# ---------------------------------------------------------------------------
if(INTFER_ENABLE_MKL)
  find_library(MKL_SYCL_BLAS NAMES mkl_sycl_blas)
  if(NOT MKL_SYCL_BLAS)
    message(WARNING
      "INTFER_ENABLE_MKL=ON but libmkl_sycl_blas was not found; "
      "disabling the oneMKL arm and building the empty fallback.")
    set(INTFER_ENABLE_MKL OFF)
  endif()
endif()

add_library(intfer_onemkl STATIC ${CMAKE_CURRENT_SOURCE_DIR}/gemm_onemkl.cpp)
target_include_directories(intfer_onemkl PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(INTFER_ENABLE_MKL)
  target_compile_definitions(intfer_onemkl PUBLIC INTFER_ENABLE_MKL=1)
  target_link_libraries(intfer_onemkl PUBLIC ${MKL_SYCL_BLAS})
endif()
add_sycl_to_target(TARGET intfer_onemkl SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gemm_onemkl.cpp)

# intfer_gemm CONSUMES intfer_onemkl: quant_gemm.cpp calls
# onemkl_available() / gemm_onemkl_fp16(), so the fallback (or real) object
# must be pulled into the engine's GEMM TU. Linking it here, from the
# consumer side, is what makes the symbols real and reachable -- without it
# the arm compiles but the linker never pulls it in, which is the exact
# "default-OFF flag no gate can reach" class the flag census exists to kill.
# Circular with intfer_onemkl's own include path? No: intfer_onemkl only needs
# the include dir (for quant_gemm.hpp), not a link edge back, so there is no
# cycle.
target_link_libraries(intfer_gemm PUBLIC intfer_onemkl)