// examples/intfer_dump/main.cpp
// Read a native-SYCL (.intfer) artifact, dequantize every tensor, and (optionally)
// UNDO the SpinQuant rotation so the weights land back in the original activation
// space. This lets a standard model (e.g. transformers) consume our quantized
// weights and tests DASH-Q quantization QUALITY in isolation from the SpinQuant
// online-rotation machinery (which the Stage-2 engine applies at inference).
//
// Output format (matches scripts/test_quantized_model.py):
//   magic "DQMP" | uint32 count | per tensor:
//     uint32 name_len | name | uint32 N | uint32 K | N*K float32 LE.
//
// Build: g++ -std=c++20 -O2 -I include src/common/logger.cpp src/quant_spec.cpp \
//        src/spinquant.cpp src/quant/dashq.cpp src/quant/output.cpp \
//        src/quant/safetensors.cpp examples/intfer_dump/main.cpp -o intfer_dump
#include "intfer/output.hpp"
#include "intfer/spinquant.hpp"
#include "intfer/quant/dashq.hpp"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace intfer;

// A Hadamard rotation is orthonormal and symmetric (R = R^T = R^-1), so the
// inverse rotation is the SAME operation as the forward one. We therefore reuse
// spinquant::rotate_weight, which already has the fast Walsh-Hadamard path.
int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: %s <model.intfer> <out.dump> [--unrotate]\n", argv[0]); return 1; }
  std::string in = argv[1], out = argv[2];
  bool unrotate = (argc > 3 && std::string(argv[3]) == "--unrotate");

  std::vector<uint8_t> blob;
  if (!read_file(in, blob)) { printf("FAIL read %s\n", in.c_str()); return 1; }
  ModelIR model; RotationSet rs;
  if (!deserialize_intfer(blob.data(), blob.size(), model, rs)) {
    printf("FAIL deserialize %s\n", in.c_str()); return 1;
  }
  // Rotations produced by our pipeline are all Hadamard (dim is a power of two);
  // the serializer doesn't persist is_hadamard, so flag it to enable the fast
  // Walsh-Hadamard apply path in rotate_weight.
  for (auto& r : rs.rotations) r.is_hadamard = true;
  printf("loaded %s: %zu tensors, %zu rotations\n", in.c_str(), model.tensors.size(), rs.size());

  FILE* f = std::fopen(out.c_str(), "wb");
  if (!f) { printf("FAIL open %s\n", out.c_str()); return 1; }
  uint32_t cnt = (uint32_t)model.tensors.size();
  fwrite("DQMP", 1, 4, f); fwrite(&cnt, 4, 1, f);
  std::vector<float> w, weff;
  uint32_t unrotated = 0;
  for (const auto& t : model.tensors) {
    w.clear(); dashq::dequant_tensor(model, t, w);
    if (unrotate) {
      RotationSite site = site_for_role(t.role);
      const Rotation* R = nullptr;
      for (size_t i = 0; i < rs.size(); ++i)
        if (rs.rotations[i].site == site) { R = &rs.rotations[i]; break; }
      if (R && R->dim > 0) {
        weff = w;
        rotate_weight(weff, t.N, t.K, *R, side_for_role(t.role));
        w = weff; ++unrotated;
      }
    }
    uint32_t nl = (uint32_t)t.name.size();
    fwrite(&nl, 4, 1, f); fwrite(t.name.data(), 1, nl, f);
    fwrite(&t.N, 4, 1, f); fwrite(&t.K, 4, 1, f);
    fwrite(w.data(), 4, w.size(), f);
  }
  std::fclose(f);
  printf("wrote %s: %u tensors (%u un-rotated)\n", out.c_str(), cnt, unrotated);
  return 0;
}
