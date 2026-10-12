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

  # *** DISCOVER THE MKL COMPONENT LIBS FROM THE WRAPPER'S OWN DIRECTORY. ***
  # libmkl_sycl_blas.so is a SYCL wrapper: it calls into the BLAS, core and
  # threading components and its own NEEDED chain is only libsycl + libdl, so
  # linking the wrapper ALONE leaves those component symbols unresolved. Measured
  # 2026-10-12 on this box: intfer_gemm (a static lib) built clean with the
  # wrapper linked in isolation -- a .a needs no undefined symbols -- and the
  # failure only surfaced at the final intfer_cli link, 272 undefined references
  # (cblas_sgemm_64, mkl_serv_lock, ...), none of them the
  # oneapi::mkl::blas::column_major::gemm that the wrapper DEFINES. The working
  # probe linked -lmkl_gnu_thread -lmkl_intel_lp64 -lmkl_core by hand; derive
  # the same set here from the wrapper's own path so no install prefix and no
  # per-box -DCMAKE_EXE_LINKER_FLAGS has to be typed.
  get_filename_component(_MKL_LIB_DIR "${MKL_SYCL_BLAS}" REALPATH)
  get_filename_component(_MKL_LIB_DIR "${_MKL_LIB_DIR}" DIRECTORY)
  foreach(_MKL_COMP IN ITEMS sycl_blas intel_lp64 core gnu_thread)
    find_library(_MKL_${_MKL_COMP}_LIB NAMES mkl_${_MKL_COMP}
      PATHS "${_MKL_LIB_DIR}" NO_DEFAULT_PATH)
    if(_MKL_${_MKL_COMP}_LIB)
      list(APPEND _MKL_LINK_LIBS "${_MKL_${_MKL_COMP}_LIB}")
    else()
      message(WARNING
        "oneMKL component libmkl_${_MKL_COMP}.so not found in "
        "${_MKL_LIB_DIR}; the INTFER_ENABLE_MKL link will be incomplete.")
    endif()
  endforeach()
  target_link_libraries(intfer_onemkl PUBLIC ${_MKL_LINK_LIBS})

  # *** DERIVE THE MKL HEADER DIR FROM THE LIBRARY, DO NOT GUESS IT. ***
  # gemm_onemkl.cpp's real body #includes <oneapi/mkl/blas.hpp>, which lives
  # under the oneMKL INSTALL PREFIX (e.g. .../mkl/latest/include), and that
  # prefix is NOT discoverable from find_library: the .so is found via
  # CMAKE_PREFIX_PATH but the include dir is not propagated by that call.
  # Probed 2026-10-12 on this box: a -DINTFER_ENABLE_MKL=ON configure with
  # -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest found libmkl_sycl_blas
  # and turned the #ifdef on, and then the TU failed to compile because
  # <oneapi/mkl/blas.hpp> was not on any include path -- the arm was wired,
  # the option was ON, and the code still could not build. Deriving the
  # prefix from the library's own path (dirname of the .so -> ../include)
  # makes the build work wherever the toolkit lays the .so down, with no
  # per-box -DCMAKE_CXX_FLAGS hand-edit.
  get_filename_component(_MKL_PREFIX "${_MKL_LIB_DIR}" DIRECTORY)
  if(EXISTS "${_MKL_PREFIX}/include/oneapi/mkl/blas.hpp")
    target_include_directories(intfer_onemkl PUBLIC "${_MKL_PREFIX}/include")
  else()
    message(WARNING
      "libmkl_sycl_blas found at ${MKL_SYCL_BLAS} but no oneMKL headers "
      "under ${_MKL_PREFIX}/include -- the INTFER_ENABLE_MKL body will not "
      "compile. Point the oneMKL install at this box's prefix.")
  endif()
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