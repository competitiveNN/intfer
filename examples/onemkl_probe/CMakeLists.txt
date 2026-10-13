# Model-free gate for the INTFER_ONEMKL arm of quant_gemm_f16 (PLAN.md 1.4).
#
# This probe is the FIRST gate that can actually reach the oneMKL branch, and
# it exists because the arm was wired and the option was ON while no test
# named it -- the exact "default-OFF flag no gate can reach" class the flag
# census exists to kill. check_flag_gate_coverage.sh files INTFER_ONEMKL as
# a REAL GAP precisely because no test_*.sh names it.
#
# It is gated on INTFER_ENABLE_MKL at build time: when the oneMKL library is
# not linked, onemkl_available() returns false and the probe SKIPS (exit 2),
# which is the honest shape for a precondition rather than a green line.
# Building it unconditionally would leave a gate that reports SKIP on every
# default build, which is noise, not coverage.
if(INTFER_ENABLE_MKL)
  add_executable(onemkl_probe main.cpp)
  target_link_libraries(onemkl_probe PRIVATE intfer_gemm intfer_quant_core intfer_common)
  add_sycl_to_target(TARGET onemkl_probe SOURCES main.cpp)
  # The oneMKL component libs live under the oneMKL install prefix, which is
  # NOT the compiler's lib dir, so the build's default rpath does not reach
  # them and `ldd` reports libmkl_sycl_blas.so.6 => not found. MKL_LIB_DIR is
  # published by src/gemm/CMakeLists.txt from the wrapper's own path; add it
  # here so the probe binds all five .so's with no LD_LIBRARY_PATH hand-edit.
  if(MKL_LIB_DIR)
    target_link_options(onemkl_probe PRIVATE "-Wl,-rpath,${MKL_LIB_DIR}")
  endif()
else()
  message(STATUS "onemkl_probe not built: INTFER_ENABLE_MKL is OFF")
endif()