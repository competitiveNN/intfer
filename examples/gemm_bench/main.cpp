// Temporary micro-bench: isolate per-GEMM host overhead on the 2x B580 fleet.
// Scaffolding for the batched-prefill investigation; not for commit.
#include "intfer/engine.hpp"
#include "intfer/gemm/quant_gemm.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: %s <model.intifer> [tensor_substr] [reps]\n", argv[0]);
    return 1;
  }
  std::string substr = (argc > 2) ? argv[2] : "layers.0.mlp.gate_proj";
  int reps = (argc > 3) ? std::atoi(argv[3]) : 20;
  auto eng = intfer::Engine::load(argv[1], 2);
  if (!eng) {
    printf("engine: load failed\n");
    return 1;
  }
  printf("engine: %zu devices\n", eng->num_devices());
  // Tensor-name matching: CLI names carry the ".weight" suffix and the
  // "model.language_model." prefix; accept short substrings by also trying
  // with the suffix appended.
  auto find_tensor = [&](const std::string& s) -> int {
    for (size_t i = 0; i < eng->model().tensors.size(); ++i)
      if (eng->model().tensors[i].name.find(s) != std::string::npos)
        return (int)i;
    return -1;
  };
  int ti = find_tensor(substr);
  if (ti < 0) ti = find_tensor(substr + ".weight");
  if (ti < 0) {
    printf("tensor '%s' not found; layer-0 self_attn candidates:\n",
           substr.c_str());
    for (size_t i = 0; i < eng->model().tensors.size(); ++i) {
      const auto& nm = eng->model().tensors[i].name;
      if (nm.find("self_attn") != std::string::npos &&
          nm.find("layers.0.") != std::string::npos)
        printf("  %s\n", nm.c_str());
    }
    return 1;
  }
  const auto& t = eng->model().tensors[(size_t)ti];
  printf("tensor %d: %s N=%u K=%u scheme=%d group=%u\n", ti, t.name.c_str(),
         t.N, t.K, (int)t.qcfg.scheme, t.qcfg.group_size);

  std::vector<float> x(t.K);
  for (uint32_t k = 0; k < t.K; ++k)
    x[k] = 0.02f * (float)((int)(k % 13) - 6);

  // Warmup (builds the shard).
  std::vector<float> y;
  for (int i = 0; i < 3; ++i) eng->run_linear_tp(ti, x, y);

#ifdef INTFER_PROFILE_GPU
  // First-call vs steady-state split: one timed call, then a batch.
  {
    intfer::reset_gpu_profile();
    double s0 = now_ms();
    eng->run_linear_tp(ti, x, y);
    double s1 = now_ms();
    auto p = intfer::get_gpu_profile();
    printf("first call: wall=%.3f ms kernel=%.0f submit=%.0f waitblock=%.0f "
           "usm_write=%.0f readback=%.0f copy=%.0f\n",
           s1 - s0, p.kernel_us, p.submit_us, p.waitblock_us, p.usm_write_us,
           p.readback_us, p.copy_us);
    intfer::reset_gpu_profile();
    s0 = now_ms();
    for (int i = 0; i < reps; ++i) eng->run_linear_tp(ti, x, y);
    s1 = now_ms();
    p = intfer::get_gpu_profile();
    printf("steady/call: wall=%.3f ms kernel=%.0f submit=%.0f waitblock=%.0f "
           "usm_write=%.0f readback=%.0f copy=%.0f\n",
           (s1 - s0) / reps, p.kernel_us / reps, p.submit_us / reps,
           p.waitblock_us / reps, p.usm_write_us / reps, p.readback_us / reps,
           p.copy_us / reps);
    // Repeat on device 1's queue state: interleave a device-0-only direct
    // kernel between run_linear_tp calls to test cross-queue interference.
    intfer::reset_gpu_profile();
    s0 = now_ms();
    for (int i = 0; i < reps; ++i) eng->run_linear_tp(ti, x, y);
    s1 = now_ms();
    p = intfer::get_gpu_profile();
    printf("steady2/call: wall=%.3f ms kernel=%.0f submit=%.0f waitblock=%.0f\n",
           (s1 - s0) / reps, p.kernel_us / reps, p.submit_us / reps,
           p.waitblock_us / reps);
  }
#endif

  double t0 = now_ms();
  for (int i = 0; i < reps; ++i) eng->run_linear_tp(ti, x, y);
  double t1 = now_ms();
  printf("M=1 run_linear_tp post-profile: %.3f ms/call (mean over %d)\n",
         (t1 - t0) / reps, reps);
  printf("y[0..3] = %.6f %.6f %.6f %.6f\n", y[0], y[1], y[2], y[3]);

#ifdef INTFER_PROFILE_GPU
  {
    auto p = intfer::get_gpu_profile();
    int n = reps + 3;
    printf("profile/call: kernel=%.1fus copy=%.1f readback=%.1f usm_write=%.1f "
           "submit=%.1f waitblock=%.1f\n",
           p.kernel_us / n, p.copy_us / n, p.readback_us / n,
           p.usm_write_us / n, p.submit_us / n, p.waitblock_us / n);
  }
#endif

  // Empty-kernel submit floor on device 0 (warmed up; submit vs wait split).
  {
    sycl::queue& q = eng->device_queue(0);
    int* sink_dev =
        sycl::malloc_device<int>(1, q.get_device(), q.get_context());
    auto empty_run = [&](bool slm, bool do_wait, int n) {
      double s0 = now_ms();
      for (int i = 0; i < n; ++i) {
        sycl::event e = q.submit([&](sycl::handler& h) {
          if (slm) {
            sycl::local_accessor<std::uint16_t, 1> sA(sycl::range<1>(64 * 64),
                                                      h);
            sycl::local_accessor<std::uint16_t, 1> sW(sycl::range<1>(64 * 64),
                                                      h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                [=](sycl::nd_item<1> it) {
                  (void)sA[(size_t)it.get_local_id(0)];
                  (void)sW[(size_t)it.get_local_id(0)];
                  sink_dev[0] = (int)it.get_global_id(0);
                });
          } else {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                [=](sycl::nd_item<1> it) {
                  sink_dev[0] = (int)it.get_global_id(0);
                });
          }
        });
        if (do_wait) e.wait();
      }
      if (do_wait) q.wait();
      double s1 = now_ms();
      return (s1 - s0) / n;
    };
    empty_run(false, true, 20);  // warmup
    printf("empty kernel submit+wait: %.3f ms/call\n",
           empty_run(false, true, 100));
    printf("empty kernel submit, no wait: %.3f ms/call\n",
           empty_run(false, false, 100));
    printf("SLM kernel submit+wait: %.3f ms/call\n",
           empty_run(true, true, 100));
    sycl::free(sink_dev, q);
  }

  // Direct-kernel test: replicate the real INT4 M=1 GEMM submit with fake
  // weights, malloc_device AND malloc_shared variants. Splits submit cost
  // from device time; shared-vs-device isolates migration/residency cost.
  auto direct_case = [&](const char* tag, bool shared) {
    sycl::queue& q = eng->device_queue(0);
    const uint32_t Nl = t.N / 2, K = t.K, Gw = 64;
    const uint32_t ng = (K + Gw - 1) / Gw;
    std::uint16_t* act;
    std::uint8_t* code;
    std::uint16_t* scale;
    std::uint16_t* out;
    if (shared) {
      act = sycl::malloc_shared<std::uint16_t>((size_t)K, q.get_device(),
                                               q.get_context());
      code = sycl::malloc_shared<std::uint8_t>((size_t)Nl * K / 2,
                                               q.get_device(), q.get_context());
      scale = sycl::malloc_shared<std::uint16_t>((size_t)Nl * ng,
                                                 q.get_device(), q.get_context());
      out = sycl::malloc_shared<std::uint16_t>((size_t)Nl, q.get_device(),
                                               q.get_context());
    } else {
      act = sycl::malloc_device<std::uint16_t>((size_t)K, q.get_device(),
                                               q.get_context());
      code = sycl::malloc_device<std::uint8_t>((size_t)Nl * K / 2,
                                               q.get_device(), q.get_context());
      scale = sycl::malloc_device<std::uint16_t>((size_t)Nl * ng,
                                                 q.get_device(), q.get_context());
      out = sycl::malloc_device<std::uint16_t>((size_t)Nl, q.get_device(),
                                               q.get_context());
    }
    std::vector<std::uint16_t> z16((size_t)Nl * ng, 0x3C00);
    std::vector<std::uint8_t> z8((size_t)Nl * K / 2, 0x11);
    std::vector<std::uint16_t> za(K, 0x3C00);
    if (shared) {
      memcpy(act, za.data(), (size_t)K * 2);
      memcpy(code, z8.data(), (size_t)Nl * K / 2);
      memcpy(scale, z16.data(), (size_t)Nl * ng * 2);
    } else {
      q.memcpy(act, za.data(), (size_t)K * 2).wait();
      q.memcpy(code, z8.data(), (size_t)Nl * K / 2).wait();
      q.memcpy(scale, z16.data(), (size_t)Nl * ng * 2).wait();
    }
    auto direct_run = [&](bool do_wait, int n) {
      double s0 = now_ms();
      for (int i = 0; i < n; ++i) {
        sycl::event e = intfer::gemm::quant_gemm_forward_usm(
            q, act, code, scale, 1.0f, out, 1, Nl, K, Gw,
            intfer::gemm::WeightScheme::INT4, sycl::event{});
        if (do_wait) e.wait();
      }
      if (do_wait) q.wait();
      double s1 = now_ms();
      return (s1 - s0) / n;
    };
    direct_run(true, 5);
    printf("direct %s submit+wait: %.3f ms/call; submit,no-wait: %.3f ms/call\n",
           tag, direct_run(true, 20), direct_run(false, 20));
    sycl::free(act, q);
    sycl::free(code, q);
    sycl::free(scale, q);
    sycl::free(out, q);
  };
  direct_case("device", false);
  direct_case("shared", true);

  // M-safety gate: M=4 batched run must match 4x M=1 rows (exercises the
  // grow_act_buffers path that used to OOB-write past M=1 act/out buffers).
  {
    const uint32_t M = 4;
    std::vector<float> xb((size_t)M * t.K);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t k = 0; k < t.K; ++k)
        xb[(size_t)m * t.K + k] = 0.02f * (float)((int)((k + m * 7) % 13) - 6);
    std::vector<float> yb;
    eng->run_linear_tp(ti, xb, yb, M);
    double maxdiff = 0;
    for (uint32_t m = 0; m < M; ++m) {
      std::vector<float> xr(t.K);
      for (uint32_t k = 0; k < t.K; ++k) xr[k] = xb[(size_t)m * t.K + k];
      std::vector<float> yr;
      eng->run_linear_tp(ti, xr, yr);
      for (uint32_t n = 0; n < t.N; ++n) {
        double d = fabs((double)yb[(size_t)m * t.N + n] - (double)yr[n]);
        if (d > maxdiff) maxdiff = d;
      }
    }
    printf("M=4 vs 4xM=1 max abs diff: %.6f %s\n", maxdiff,
           maxdiff < 1e-3 ? "(OK)" : "(MISMATCH)");
    // Second M=4 call must not regrow (steady-state path).
    eng->run_linear_tp(ti, xb, yb, M);
    printf("M=4 second call ok, y[0]=%.6f\n", yb[0]);
  }
  // like a real gate_proj shard (act 10KB, code 22MB, scale 1.3MB, out 17KB).
  {
    sycl::queue& q = eng->device_queue(0);
    size_t act_n = (size_t)t.K;
    size_t code_n = (size_t)(t.N / 2) * t.K / 2;  // ~INT4 packed half-shard
    size_t scale_n = (size_t)(t.N / 2) * (t.K / 64);
    size_t out_n = (size_t)(t.N / 2);
    auto* act =
        sycl::malloc_shared<std::uint16_t>(act_n, q.get_device(), q.get_context());
    auto* code =
        sycl::malloc_shared<std::uint8_t>(code_n, q.get_device(), q.get_context());
    auto* scale = sycl::malloc_shared<std::uint16_t>(scale_n, q.get_device(),
                                                     q.get_context());
    auto* out =
        sycl::malloc_shared<std::uint16_t>(out_n, q.get_device(), q.get_context());
    for (size_t i = 0; i < act_n; ++i) act[i] = 0x3C00;
    for (size_t i = 0; i < code_n; i += 4096) code[i] = (uint8_t)i;
    for (size_t i = 0; i < scale_n; i += 4096) scale[i] = 0x3C00;
    auto copy_run = [&](int n) {
      double s0 = now_ms();
      for (int i = 0; i < n; ++i) {
        sycl::event e = q.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
              [=](sycl::nd_item<1> it) {
                size_t gid = (size_t)it.get_global_id(0);
                std::uint16_t v = act[gid % act_n] + scale[gid % scale_n];
                v = (std::uint16_t)(v + code[(gid * 173) % code_n]);
                out[gid % out_n] = v;
              });
        });
        e.wait();
      }
      double s1 = now_ms();
      return (s1 - s0) / n;
    };
    copy_run(5);  // warmup (fault in pages)
    printf("shared-USM 4-buffer kernel submit+wait: %.3f ms/call\n",
           copy_run(50));
    sycl::free(act, q);
    sycl::free(code, q);
    sycl::free(scale, q);
    sycl::free(out, q);
  }
  return 0;
}
