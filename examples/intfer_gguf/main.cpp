// examples/intfer_gguf/main.cpp
// .intfer -> .gguf converter + round-trip fidelity check.
//
// Exercises the GGUF interop serializer (serialize_gguf/deserialize_gguf)
// on a REAL quantized model instead of synthetic pipeline data:
//   1. load <model.intfer> (borrowed mmap: 27B payloads stay in page cache)
//   2. serialize to GGUF, write <model.gguf>
//   3. read back the .gguf, dequantize every tensor from both the .intfer
//      and the .gguf copy, report max rel-L2 (must be ~0: same payload,
//      different container).
//
// Usage: intfer_gguf <model.intfer> <model.gguf>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"

using namespace intfer;

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return -1.0;
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  if (den == 0) return (num == 0) ? 0.0 : 1.0;
  return std::sqrt(num / den);
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: %s <model.intfer> <model.gguf>\n", argv[0]); return 1; }
  std::string in = argv[1], out = argv[2];

  std::vector<uint8_t> blob;
  if (!read_file(in, blob)) { printf("FAIL read %s\n", in.c_str()); return 1; }
  ModelIR model; RotationSet rs;
  if (!deserialize_intfer_borrowed(blob.data(), blob.size(), model, rs)) {
    printf("FAIL deserialize %s\n", in.c_str()); return 1;
  }
  printf("loaded %s: %zu tensors, %zu rotations\n",
         in.c_str(), model.tensors.size(), rs.size());

  std::vector<uint8_t> gguf = serialize_gguf(model, rs);
  if (gguf.empty()) { printf("FAIL serialize_gguf\n"); return 1; }
  if (!write_file(out, gguf)) { printf("FAIL write %s\n", out.c_str()); return 1; }
  printf("wrote %s (%.1f MiB)\n", out.c_str(), (double)gguf.size() / (1024 * 1024));

  std::vector<uint8_t> disk;
  if (!read_file(out, disk)) { printf("FAIL re-read %s\n", out.c_str()); return 1; }
  ModelIR mg; RotationSet rg;
  if (!deserialize_gguf(disk.data(), disk.size(), mg, rg)) {
    printf("FAIL deserialize_gguf %s\n", out.c_str()); return 1;
  }
  printf("gguf round-trip: %zu tensors, %zu rotations\n", mg.tensors.size(), rg.size());
  if (mg.tensors.size() != model.tensors.size()) {
    printf("FAIL tensor count %zu != %zu\n", mg.tensors.size(), model.tensors.size());
    return 1;
  }

  double worst = 0;
  std::string worst_name;
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    std::vector<float> a, b;
    dashq::dequant_tensor(model, model.tensors[i], a);
    dashq::dequant_tensor(mg, mg.tensors[i], b);
    double e = rel_l2(a, b);
    if (e > worst) { worst = e; worst_name = model.tensors[i].name; }
  }
  printf("max dequant rel-L2 (.intfer vs .gguf): %.6f (%s) %s\n",
         worst, worst_name.c_str(), worst < 1e-6 ? "(PASS)" : "(FAIL)");
  return worst < 1e-6 ? 0 : 1;
}
