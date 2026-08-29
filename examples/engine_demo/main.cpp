// examples/engine_demo/main.cpp
// Smoke test for the Stage-2 SYCL engine: load the .intfer, capture the decode
// command graph, and run one decode step. On a machine with Arc GPUs this runs
// the quantized GEMM (DPAS) + FlashAttention tensor-parallel; without a GPU it
// reports that no device was discovered and exits cleanly.
#include "intfer/engine.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s <model.intfer>\n", argv[0]); return 1; }
  auto eng = intfer::Engine::load(argv[1]);
  if (!eng) { printf("engine: load failed (no Arc GPU / bad file)\n"); return 1; }
  printf("engine: %zu devices, %zu layers\n", eng->num_devices(), eng->num_layers());
  eng->capture_decode_graph();

  uint32_t K = eng->model().tensors.empty() ? 1024u : eng->model().tensors[0].K;
  std::vector<float> tok(K, 0.01f);
  auto out = eng->decode_step(tok);
  double n = 0;
  for (float v : out) n += (double)v * v;
  printf("decode_step -> %zu values, |out| = %.4f\n", out.size(), std::sqrt(n));
  return 0;
}
