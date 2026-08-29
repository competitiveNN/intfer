// hello_sycl — validates the intfer toolchain on the local GPU fleet.
//
// Discovers every Intel Arc GPU (the 2x B580), runs a trivial SYCL kernel
// (C = A + B) on each device concurrently, and reports PASS/FAIL per device.
// This is the smoke test for the Stage-2 SYCL/Level-Zero build system.

#include "intfer/common/logger.hpp"
#include "intfer/device/device_context.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <vector>

namespace {

constexpr std::size_t kN = 1u << 20;  // 1M elements

bool run_on_device(intfer::device::DeviceContext& ctx) {
  std::vector<float> a(kN, 1.0f * (ctx.index + 1));
  std::vector<float> b(kN, 2.0f);
  std::vector<float> c(kN, 0.0f);

  {
    sycl::buffer<float> ba(a), bb(b), bc(c);
    ctx.queue.submit([&](sycl::handler& h) {
      auto A = ba.get_access<sycl::access::mode::read>(h);
      auto B = bb.get_access<sycl::access::mode::read>(h);
      auto C = bc.get_access<sycl::access::mode::write>(h);
      h.parallel_for<class hello_sycl_kernel>(sycl::range<1>(kN),
          [=](sycl::id<1> i) { C[i] = A[i] + B[i]; });
    });
    ctx.queue.wait();
  }

  for (std::size_t i = 0; i < kN; ++i) {
    if (c[i] != (a[i] + b[i])) return false;
  }
  return true;
}

}  // namespace

int main() {
  auto mgr = intfer::device::DeviceManager::discover("Arc");
  if (mgr.size() == 0) {
    INTFER_LOG_ERROR("No Intel Arc GPUs discovered.");
    return 1;
  }

  INTFER_LOG_INFO("Discovered %zu GPU(s).", mgr.size());
  mgr.print_summary();

  // Simple tensor-parallel sharding sanity check.
  auto ranges = mgr.shard(1000);
  for (std::size_t i = 0; i < mgr.size(); ++i) {
    INTFER_LOG_INFO("  shard[%zu] = [%zu, %zu)", i, ranges[i].first, ranges[i].second);
  }

  std::size_t failures = 0;
  for (std::size_t d = 0; d < mgr.size(); ++d) {
    const bool ok = run_on_device(mgr[d]);
    INTFER_LOG_INFO("GPU[%zu] '%s' kernel C=A+B: %s", d, mgr[d].name.c_str(),
                    ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
  }

  if (failures) {
    INTFER_LOG_ERROR("%zu device(s) failed the kernel test.", failures);
    return 1;
  }

  INTFER_LOG_INFO("All %zu GPU(s) executed the SYCL kernel successfully.", mgr.size());
  return 0;
}
