// slm_probe: measure the device's max local memory (SLM) per work-group and
// verify a large local_accessor actually works at runtime. No model, no
// weights -- a pure device query. Used to size the XMX K-split + SG_ROWS
// weight tile, which is [KSPLIT * SG_ROWS * BK * BN] FP16.
//
// Build: add to examples/CMakeLists.txt (or standalone icpx).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  sycl::queue q;
  auto dev = q.get_device();
  const size_t local_bytes = dev.get_info<sycl::info::device::local_mem_size>();
  const uint32_t n_sub = dev.get_info<sycl::info::device::sub_group_sizes>().empty()
                             ? 0
                             : dev.get_info<sycl::info::device::sub_group_sizes>().front();
  std::fprintf(stderr,
               "[slm] device=%s local_mem_size=%zu B (%.2f KB) "
               "preferred_sub_group_size=%u\n",
               dev.get_info<sycl::info::device::name>().c_str(),
               local_bytes, local_bytes / 1024.0, n_sub);

  // Candidate XMX K-split + SG_ROWS SLM sizes (FP16 weight tile only):
  //   KSPLIT * SG_ROWS * BK * BN * 2 bytes  (+ FP32 partial C: *4 bytes)
  struct { const char* name; size_t fp16; size_t fp32; } cands[] = {
    { "KSPLIT4_SGROWS16_BK64_BN16", 4u*16u*64u*16u*2u, 4u*16u*16u*4u },
    { "KSPLIT4_SGROWS16_BK32_BN16", 4u*16u*32u*16u*2u, 4u*16u*16u*4u },
    { "KSPLIT2_SGROWS16_BK64_BN16", 2u*16u*64u*16u*2u, 2u*16u*16u*4u },
    { "KSPLIT2_SGROWS16_BK32_BN16", 2u*16u*32u*16u*2u, 2u*16u*16u*4u },
    { "KSPLIT1_SGROWS16_BK64_BN16", 1u*16u*64u*16u*2u, 1u*16u*16u*4u },
  };
  for (auto& c : cands) {
    const size_t total = c.fp16 + c.fp32;
    std::fprintf(stderr,
                 "[slm] %-26s fp16=%zu B (%.1f KB) fp32=%zu B total=%zu B (%.1f KB) "
                 "fits=%d\n",
                 c.name, c.fp16, c.fp16/1024.0, c.fp32, total, total/1024.0,
                 (int)(total <= local_bytes));
  }

  // Runtime check: actually allocate the largest candidate and touch it.
  // nd_range form: the simple parallel_for(range) form rejects local_accessor
  // (it is a single-task form), so drive a real work-group with 16 threads.
  const size_t probe = 4u * 16u * 64u * 16u * 2u;  // 128 KB
  if (probe > local_bytes) {
    std::fprintf(stderr, "[slm] SKIP runtime probe: %zu B > device local_mem_size\n", probe);
    return 0;
  }
  try {
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<std::uint8_t, 1> slm(sycl::range<1>(probe), h);
      sycl::nd_range<1> rng(sycl::range<1>(16), sycl::range<1>(16));
      h.parallel_for(rng, [=](sycl::nd_item<1> it) {
        slm[it.get_local_id()[0]] = 0;
        slm[probe - 1 - it.get_local_id()[0]] = 1;
      });
    });
    q.wait();
    std::fprintf(stderr, "[slm] RUNTIME OK: %zu B local_accessor allocated and touched\n", probe);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[slm] RUNTIME FAIL: %zu B local_accessor threw: %s\n", probe, e.what());
    return 1;
  }
  return 0;
}