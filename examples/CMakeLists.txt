# intfer Stage-2 build-system smoke test (enumerates both Arc B580 GPUs and
# runs a trivial SYCL kernel on each). Stage-1 quantization demos are built
# via their own targets when INTFER_BUILD_QUANT=ON.
add_subdirectory(quant_merge)
add_subdirectory(intfer_fidelity)
add_subdirectory(hello_sycl)
add_subdirectory(flash_attention_demo)
add_subdirectory(spinquant_test)
add_subdirectory(spinquant_demo)
add_subdirectory(quant_pipeline)
add_subdirectory(gemm_demo)
add_subdirectory(engine_demo)
add_subdirectory(gemm_bench)
add_subdirectory(qwen_forward_gpu)
add_subdirectory(qwen_forward_intfer)
add_subdirectory(intfer_cli)
