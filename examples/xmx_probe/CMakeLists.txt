# Model-free gate for the INTFER_XMX_GEMM arm of
# quant_gemm_forward_usm_xmx. Built UNCONDITIONALLY (unlike
# examples/onemkl_probe, which is gated on INTFER_ENABLE_MKL): the XMX
# kernel is always compiled into intfer_gemm, so the arm is always
# reachable and the probe can always run.
#
# This probe exists because the arm was wired and the option was ON while
# no test_*.sh named it -- the exact "default-OFF flag no gate can reach"
# class the flag census exists to kill. Worse, the gate's own
# test_gpu_optin.sh INTFER_XMX_GEMM arm was VACUOUS: the engine wires the
# XMX path ONLY into Engine::run_linear_tp, which the 27B prefill hot
# path never calls (it uses run_linear_wave / run_mlp_gateup, both of
# which call quant_gemm_forward_usm directly), and the kernel's own gate
# (M >= 8, M%16==0, N%16==0, K%64==0) rejects every real model's prompt
# length, so the fixture produced no [xmx-gemm] line and the arm certified
# nothing. This probe calls the kernel DIRECTLY on synthetic tile-aligned
# INT2 tensors, so reachability is structural rather than dependent on a
# model's prompt length.
add_executable(xmx_probe main.cpp)
target_link_libraries(xmx_probe PRIVATE intfer_gemm intfer_common)
add_sycl_to_target(TARGET xmx_probe SOURCES main.cpp)
