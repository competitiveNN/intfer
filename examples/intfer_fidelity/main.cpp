// examples/intfer_fidelity/main.cpp
// Stage-1 quantization fidelity check: dequantize every tensor in the .intfer
// artifact (the exact payload the SYCL GEMM kernel dequantizes on the fly) and
// compare to the original FP16/FP32 safetensors weight. Reports per-role cosine
// so we can see how much error the DASH-Q / RCO pipeline introduces before the
// weights ever reach the XMX DPAS path.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "intfer/safetensors.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/spinquant.hpp"

static bool read_file(const std::string& p, std::vector<uint8_t>& out) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  out.resize(n); size_t r = fread(out.data(), 1, n, f); fclose(f);
  return r == (size_t)n;
}

int main(int argc, char** argv) {
  const char* st_path = (argc > 1) ? argv[1] : "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
  const char* intfer_path = (argc > 2) ? argv[2] : "test/models/Qwen3.5-0.8B.intfer";

  // Original weights: stream one tensor at a time (the full 27B model as
  // fp32 does not fit in RAM alongside the .intifer payload).
  intfer::SafetensorsReader r;
  if (!r.open(st_path)) { printf("FAIL open safetensors\n"); return 1; }
  printf("safetensors index open: %s\n", st_path);

  // .intifer model (borrowed mmap views: a 22GB 27B payload stays in page
  // cache instead of being copied into RAM).
  intfer::ModelIR m;
  intfer::RotationSet rs;
  {
    int fdi = ::open(intfer_path, O_RDONLY);
    if (fdi < 0) { printf("FAIL open intfer\n"); return 1; }
    struct stat sti;
    if (::fstat(fdi, &sti) != 0 || sti.st_size <= 0) { ::close(fdi); printf("FAIL stat intfer\n"); return 1; }
    void* mbase = ::mmap(nullptr, (size_t)sti.st_size, PROT_READ, MAP_PRIVATE, fdi, 0);
    ::close(fdi);
    if (mbase == MAP_FAILED) { printf("FAIL mmap intfer\n"); return 1; }
    if (!intfer::deserialize_intfer_borrowed((const uint8_t*)mbase, (size_t)sti.st_size, m, rs)) {
      printf("FAIL deserialize intfer\n"); return 1;
    }
  }
  printf("intfer tensors=%zu  (rotation sets=%zu)\n", m.tensors.size(), rs.size());

  const char* role_name[] = {"EMBED","LM_HEAD","?","ATTN_Q","ATTN_K","ATTN_V","ATTN_O",
                             "MLP_GATE","MLP_UP","MLP_DOWN","NORM","ROPE","OTHER"};

  int worst_role = -1; float worst_cos = 2.f;
  std::map<int, float> role_min;   // role -> min cosine
  std::map<int, int>   role_cnt;
  for (const auto& t : m.tensors) {
    static int cnt = 0;
    if (cnt < 12) { printf("  name=%-55s role=%d rid=%d scheme=%d gs=%d\n", t.name.c_str(), (int)t.role, (int)t.qcfg.rotation_id, (int)t.qcfg.scheme, (int)t.qcfg.group_size); cnt++; }
    std::vector<float> w;
    intfer::dashq::dequant_tensor(m, t, w);
    // The payload stores the SpinQuant-rotated weight (W' = W·Rᵀ on the
    // right site, W' = Rᵀ·W on the left site). Un-rotate with the stored
    // rotation before comparing against the unrotated safetensors weight;
    // otherwise the cosine is meaningless (~0.2 for random orthogonal R).
    // Blocked fp32 matmul (unrotation of a 17408x5120 weight is ~0.5 GFLOP).
    if (t.qcfg.rotation_id != 0) {
      const intfer::Rotation* R = rs.find(t.qcfg.rotation_id);
      if (R && R->mat.size() == (size_t)R->dim * R->dim && !w.empty()) {
        uint32_t N = t.N, K = t.K;
        intfer::RotationSide side = intfer::side_for_role(t.role);
        std::vector<float> u(w.size());
        const float* Rm = R->mat.data();
        const float* Wm = w.data();
        float* Um = u.data();
        const uint32_t B = 64;
        if (side == intfer::RotationSide::Right && R->dim == K) {
          // U = W'·R
          for (uint32_t r0 = 0; r0 < N; r0 += B)
            for (uint32_t j0 = 0; j0 < K; j0 += B)
              for (uint32_t i0 = 0; i0 < K; i0 += B) {
                uint32_t r1 = std::min(r0 + B, N), j1 = std::min(j0 + B, K),
                         i1 = std::min(i0 + B, K);
                for (uint32_t r = r0; r < r1; ++r)
                  for (uint32_t i = i0; i < i1; ++i) {
                    float a = Wm[(size_t)r * K + i];
                    for (uint32_t j = j0; j < j1; ++j)
                      Um[(size_t)r * K + j] += a * Rm[(size_t)i * K + j];
                  }
              }
          w.swap(u);
        } else if (side == intfer::RotationSide::Left && R->dim == N) {
          // U = R·W'
          for (uint32_t i0 = 0; i0 < N; i0 += B)
            for (uint32_t c0 = 0; c0 < K; c0 += B)
              for (uint32_t j0 = 0; j0 < N; j0 += B) {
                uint32_t i1 = std::min(i0 + B, N), c1 = std::min(c0 + B, K),
                         j1 = std::min(j0 + B, N);
                for (uint32_t i = i0; i < i1; ++i)
                  for (uint32_t j = j0; j < j1; ++j) {
                    float a = Rm[(size_t)i * N + j];
                    for (uint32_t c = c0; c < c1; ++c)
                      Um[(size_t)i * K + c] += a * Wm[(size_t)j * K + c];
                  }
              }
          w.swap(u);
        } else {
          printf("  [warn] rot dim mismatch %s (N=%u K=%u Rdim=%u)\n",
                 t.name.c_str(), N, K, R->dim);
        }
      }
    }
    // Stream the matching safetensors weight (try with and without the
    // language_model prefix) instead of preloading the whole model.
    std::vector<float> ref;
    if (!r.read(t.name, ref)) {
      std::string alt = t.name;
      const std::string pre = "model.";
      if (alt.compare(0, pre.size(), pre) == 0)
        alt = "model.language_model." + alt.substr(pre.size());
      if (!r.read(alt, ref)) {
        printf("  [skip] no safetensors key for %s\n", t.name.c_str());
        continue;
      }
    }
    size_t n = std::min(w.size(), ref.size());
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) { dot += w[i]*ref[i]; na += w[i]*w[i]; nb += ref[i]*ref[i]; }
    float c = (float)(dot / (std::sqrt(na)*std::sqrt(nb) + 1e-12));
    // Scale check: cosine is scale-invariant, so also report the optimal
    // least-squares scale s = <w,ref>/<w,w> (want ~1.0) and both norms.
    // A bad global alpha / merge offset shows up here, not in cos.
    double s = dot / (na + 1e-12);
    static int scnt = 0;
    if (scnt < 40) {
      printf("  [scale] %-52s s=%.4f |w|=%.4g |ref|=%.4g cos=%.6f\n",
             t.name.c_str(), s, std::sqrt(na), std::sqrt(nb), c);
      scnt++;
    }
    int role = (int)t.role;
    if (!role_min.count(role) || c < role_min[role]) role_min[role] = c;
    role_cnt[role]++;
    if (c < worst_cos) { worst_cos = c; worst_role = role; }
  }
  printf("\nPer-role min cosine (dequant .intfer vs original safetensors):\n");
  for (auto& kv : role_min)
    printf("  %-9s (n=%d): min cos = %.6f\n", role_name[kv.first], role_cnt[kv.first], kv.second);
  printf("\nWORST overall: role=%s cos=%.6f\n", role_name[worst_role], worst_cos);
  return 0;
}
