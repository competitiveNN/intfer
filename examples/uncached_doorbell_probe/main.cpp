// uncached_doorbell_probe: measure the S1 PRECONDITION for intfer's next
// doorbell optimization -- whether a ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED host
// allocation measurably reduces a host<->device round trip on THIS box.
//
// WHY THIS IS A PROBE AND NOT A GATE. The technique being preconditioned is
// Strata's host_malloc_polled (sycl/include/strata/sycl_queue.hpp:63-79): it
// replaces sycl::malloc_host with a zeMemAllocHost carrying
// ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED. A gate would need a full model run to
// say whether intfer's TP=2 doorbells got faster; a gate cannot load the 27B
// from an agent session, so what is certifiable HERE is the precondition --
// that the flag exists on this driver and that a small host buffer allocated
// with it really does round-trip differently from the shipped one. The
// performance verdict is the operator's, from an A/B on nk/nw/nr and the host
// buckets, never on a tok/s and never from this probe.
//
// THE ARMS ARE THE SHIPPED BASELINE vs THE CANDIDATE. The baseline arm is
// sycl::malloc_host, which is what intfer's doorbell buffers already use --
// the same call Strata falls back to when the flag is unavailable. The
// candidate arm is the L0 allocation with the flag. A third possibility
// (zeMemAllocHost with flags=0) is deliberately NOT an arm: it would measure
// "cached L0 alloc vs uncached L0 alloc", which is not the decision, because
// the decision is "does replacing the shipped allocation pay".
//
// WHAT THE PROBE CERTIFIES ITSELF (structural, independent of timing):
//   (a) both allocations succeeded and returned DISTINCT pointers -- a probe
//       that measures the same buffer twice is vacuous, and that is the exact
//       failure mode this repo's audit log records for vacuous gates;
//   (b) zeMemGetAllocProperties reports the candidate as ZE_MEMORY_TYPE_HOST
//       -- the allocation must be host USM the SYCL kernel can address, not
//       silently something else;
//   (c) EVERY round's token round-trips. The kernel writes
//       buf[i] = token + i (mod 256), which differs at EVERY index from the
//       previous round, so a byte the host reads that is not what the kernel
//       wrote proves the measurement is measuring nothing. That check is also
//       the mechanism the candidate flag is supposed to change, so it doubles
//       as a stale-read detector;
//   (d) at least 2 passes. A single delta has no noise floor to clear, and
//       quoting it is the defect this repo has refused numbers for.
//
// WHAT THE PROBE DOES NOT DO: it prints no adopt/decline verdict. The number
// is host wall-clock, which is a load-sensitive instrument on this box
// (load1 runs 3-18 with the desktop and sibling agent sessions), so it can
// only ever be read as a relative precondition, never as an achievement
// figure. The operator decides.
//
// Usage: uncached_doorbell_probe [--bytes N] [--rounds N] [--passes N]
// Build: add to examples/CMakeLists.txt (or standalone icpx).

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr const char* kTag = "uncached-db";
constexpr std::uint64_t kDefaultBytes = 256;   // a doorbell, not a tensor
constexpr int kDefaultRounds = 2000;
constexpr int kDefaultPasses = 3;
constexpr int kDefaultWarmup = 200;

// Select a device exactly as the engine does (src/device/device_manager.cpp:43
// filters backends). sycl::gpu_selector_v is NOT safe here: on this box it
// spans backends and returns 4 GPU devices (2 Level-Zero + 2 OpenCL), and the
// SYCL runtime then throws "The program was built for 2 devices". The engine
// never saw it because it filters before constructing any queue.
sycl::device pick_l0_gpu() {
  auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
  sycl::device chosen;
  for (const auto& d : all) {
    if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    chosen = d;
    break;
  }
  return chosen;
}

// One device-write -> host-read round trip. `token` is the round number; the
// kernel writes buf[i] = token + i (mod 256), which differs at EVERY index
// from any other round, so the host verification below can detect a stale or
// mis-routed read. Returns false the moment the kernel throws OR the token did
// not round-trip: either way the round measured nothing and must not be
// counted, because a vacuous 0 ns here would otherwise be reported as speed.
bool one_round(sycl::queue& q, std::uint8_t* buf, std::size_t bytes,
               std::uint32_t token) {
  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<1>(bytes),
                     [=](sycl::id<1> i) {
                       buf[i] =
                           (std::uint8_t)(token + (std::uint32_t)i[0]);
                     });
    });
    q.wait();
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "[%s] FAIL: round %u threw (%s)\n", kTag, token,
                 e.what());
    return false;
  }
  if (!buf) return false;
  for (std::size_t i = 0; i < bytes; ++i) {
    if (buf[i] != (std::uint8_t)(token + (std::uint32_t)i)) {
      std::fprintf(stderr,
                   "[%s] FAIL: round %u byte %zu read %u, expected %u -- the "
                   "round trip did not deliver the data\n",
                   kTag, token, i, (unsigned)buf[i],
                   (unsigned)(std::uint8_t)(token + (std::uint32_t)i));
      return false;
    }
  }
  return true;
}

struct ArmStats {
  const char* name = "";
  double med_ns = 0.0;
  double p10_ns = 0.0;
  double p90_ns = 0.0;
  double pass_spread = 0.0;   // (max-min)/median across passes
  std::vector<double> pass_medians;
  bool ok = false;
  bool alloc_ok = false;
  bool host_type = false;
};

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Percentile of a sorted copy. The latencies are pooled across passes, so the
// reported p10/p90 describe the whole run rather than one pass.
double pct_of(const std::vector<double>& src, double q) {
  if (src.empty()) return 0.0;
  std::vector<double> v(src);
  std::sort(v.begin(), v.end());
  const double idx = q * (double)(v.size() - 1);
  const std::size_t lo = (std::size_t)idx;
  const std::size_t hi = (lo + 1 < v.size()) ? lo + 1 : lo;
  const double frac = idx - (double)lo;
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

}  // namespace

int main(int argc, char** argv) {
  std::uint64_t bytes = kDefaultBytes;
  int rounds = kDefaultRounds;
  int passes = kDefaultPasses;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strcmp(a, "--bytes") == 0 && i + 1 < argc) {
      bytes = (std::uint64_t)std::strtoull(argv[++i], nullptr, 10);
    } else if (std::strcmp(a, "--rounds") == 0 && i + 1 < argc) {
      rounds = (int)std::strtol(argv[++i], nullptr, 10);
    } else if (std::strcmp(a, "--passes") == 0 && i + 1 < argc) {
      passes = (int)std::strtol(argv[++i], nullptr, 10);
    } else if (std::strcmp(a, "--help") == 0) {
      std::fprintf(stdout,
                   "usage: uncached_doorbell_probe [--bytes N] "
                   "[--rounds N] [--passes N]\n");
      return 0;
    } else {
      std::fprintf(stderr, "[%s] FAIL: unknown argument '%s'\n", kTag, a);
      return 1;
    }
  }
  if (bytes == 0 || bytes > (1u << 20)) {
    std::fprintf(stderr, "[%s] FAIL: --bytes must be in 1..1MiB (got %llu)\n",
                 kTag, (unsigned long long)bytes);
    return 1;
  }
  if (rounds < 1 || passes < 1) {
    std::fprintf(stderr,
                 "[%s] FAIL: --rounds and --passes must be >= 1 (got %d, %d)\n",
                 kTag, rounds, passes);
    return 1;
  }
  // A single pass has no noise floor to clear. Refusing here rather than
  // printing a ratio is the same rule this repo's bench summary enforces, and
  // the reason is the same: a one-sample delta is indistinguishable from noise.
  if (passes < 2) {
    std::fprintf(stderr,
                 "[%s] REFUSED: --passes must be >= 2 for the spread floor to "
                 "exist; a single-round delta may not be reported as a result "
                 "(AGENTS.md)\n",
                 kTag);
    return 1;
  }

  std::fprintf(stderr, "[%s] device selection\n", kTag);
  sycl::device dev = pick_l0_gpu();
  if (!dev.is_gpu()) {
    std::fprintf(stderr, "[%s] SKIP: no Level-Zero GPU device\n", kTag);
    return 2;
  }
  std::fprintf(stderr, "[%s] device=%s\n", kTag,
               dev.get_info<sycl::info::device::name>().c_str());

  // The queue is constructed inside try: its constructor selects the device and
  // is the call that throws on a box whose SYCL kernel path is hung. That is a
  // PRECONDITION, not a defect in this probe, so SKIP rather than abort -- a
  // probe that core-dumps on a degraded box trains the operator to ignore it.
  sycl::queue* q = nullptr;
  try {
    q = new sycl::queue{dev};
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "[%s] SKIP: no GPU queue (%s)\n", kTag, e.what());
    return 2;
  }

  // The L0 context must be the SAME context the SYCL device runs in. Pulling
  // it from the queue (Strata's spelling, sycl_queue.hpp:67) guarantees that,
  // rather than guessing a context the runtime was never told about.
  ze_context_handle_t ctx = nullptr;
  try {
    ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_context());
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "[%s] SKIP: cannot get the L0 context (%s)\n", kTag,
                 e.what());
    delete q;
    return 2;
  }

  // ---- ARM 1: the SHIPPED baseline -- sycl::malloc_host. ----
  std::uint8_t* base_buf = sycl::malloc_host<std::uint8_t>(bytes, *q);
  if (!base_buf) {
    std::fprintf(stderr, "[%s] FAIL: sycl::malloc_host returned null\n", kTag);
    delete q;
    return 1;
  }

  // ---- ARM 2: the S1 CANDIDATE -- zeMemAllocHost + BIAS_UNCACHED. ----
  ze_host_mem_alloc_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  desc.pNext = nullptr;
  desc.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;
  std::uint8_t* cand_buf = nullptr;
  const ze_result_t zr = zeMemAllocHost(ctx, &desc, bytes, 64,
                                        (void**)&cand_buf);
  if (zr != ZE_RESULT_SUCCESS || !cand_buf) {
    // The driver does not take the flag. That is the precondition failing, and
    // it is the answer S1 needs -- reported as a SKIP with the L0 result code
    // so the operator can see WHICH refusal, not a fabricated number.
    std::fprintf(stderr,
                 "[%s] SKIP: zeMemAllocHost(BIAS_UNCACHED) refused "
                 "(ze_result=0x%x) -- S1 precondition NOT MET on this driver\n",
                 kTag, (unsigned)zr);
    sycl::free(base_buf, *q);
    delete q;
    return 2;
  }

  // ---- STRUCTURAL ASSERTIONS, before any timing is deemed meaningful. ----
  bool structural_ok = true;
  if ((void*)base_buf == (void*)cand_buf) {
    std::fprintf(stderr,
                 "[%s] FAIL: both arms returned the same buffer -- the "
                 "comparison is vacuous\n",
                 kTag);
    structural_ok = false;
  }

  // NB the 4th argument is the DEVICE handle, not the type -- the memory type
  // comes back in pMemAllocProperties.type. Reading the type out of the wrong
  // field is a silent mis-read, so the values are captured immediately after
  // each call rather than shared in one desc.
  ze_memory_allocation_properties_t ap{};
  ze_device_handle_t base_dev = nullptr;
  (void)zeMemGetAllocProperties(ctx, (const void*)base_buf, &ap, &base_dev);
  const int base_type = (int)ap.type;
  ze_memory_allocation_properties_t cp{};
  ze_device_handle_t cand_dev = nullptr;
  (void)zeMemGetAllocProperties(ctx, (const void*)cand_buf, &cp, &cand_dev);
  const int cand_type = (int)cp.type;
  (void)base_dev;
  (void)cand_dev;
  if (cand_type != (int)ZE_MEMORY_TYPE_HOST) {
    std::fprintf(stderr,
                 "[%s] FAIL: the candidate allocation reports type=%d, expected "
                 "ZE_MEMORY_TYPE_HOST (%d) -- it is not host USM the SYCL "
                 "kernel can address\n",
                 kTag, cand_type, (int)ZE_MEMORY_TYPE_HOST);
    structural_ok = false;
  }
  std::fprintf(stderr,
               "[%s] alloc props: base type=%d cand type=%d distinct=%d\n",
               kTag, base_type, cand_type, (int)(base_buf != cand_buf));

  const std::size_t nbytes = (std::size_t)bytes;
  std::fprintf(stderr,
               "[%s] arms: base=sycl::malloc_host cand=zeMemAllocHost|"
               "BIAS_UNCACHED bytes=%zu rounds=%d passes=%d warmup=%d\n",
               kTag, nbytes, rounds, passes, kDefaultWarmup);

  ArmStats base{"base_malloc_host"};
  ArmStats cand{"cand_bias_uncached"};
  base.alloc_ok = true;
  cand.alloc_ok = true;
  base.host_type = (base_type == (int)ZE_MEMORY_TYPE_HOST);
  cand.host_type = (cand_type == (int)ZE_MEMORY_TYPE_HOST);

  std::vector<double> base_pool;
  std::vector<double> cand_pool;

  // ---- MEASUREMENT: each pass runs BOTH arms, so a pass is a paired round and
  //      the spread across passes is the noise floor the delta must clear. ----
  for (int p = 0; p < passes && structural_ok; ++p) {
    // Warmup is inside the pass so the first measured round of every pass is
    // not the one that pays first-touch migration or shader-load cost.
    for (int w = 0; w < kDefaultWarmup; ++w) {
      if (!one_round(*q, base_buf, nbytes, (std::uint32_t)w)) {
        structural_ok = false;
        break;
      }
    }
    for (int w = 0; w < kDefaultWarmup; ++w) {
      if (!one_round(*q, cand_buf, nbytes, (std::uint32_t)w)) {
        structural_ok = false;
        break;
      }
    }

    std::vector<double> lat;
    lat.reserve((std::size_t)rounds);
    for (int r = 0; r < rounds && structural_ok; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      const bool good = one_round(*q, base_buf, nbytes, (std::uint32_t)r);
      const auto t1 = std::chrono::steady_clock::now();
      if (!good) {
        structural_ok = false;
        break;
      }
      lat.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                         t1 - t0).count());
    }
    if (structural_ok && !lat.empty()) {
      const double m = median_of(lat);
      base.pass_medians.push_back(m);
      base_pool.insert(base_pool.end(), lat.begin(), lat.end());
    }

    std::vector<double> clat;
    clat.reserve((std::size_t)rounds);
    for (int r = 0; r < rounds && structural_ok; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      const bool good = one_round(*q, cand_buf, nbytes, (std::uint32_t)r);
      const auto t1 = std::chrono::steady_clock::now();
      if (!good) {
        structural_ok = false;
        break;
      }
      clat.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                         t1 - t0).count());
    }
    if (structural_ok && !clat.empty()) {
      const double m = median_of(clat);
      cand.pass_medians.push_back(m);
      cand_pool.insert(cand_pool.end(), clat.begin(), clat.end());
    }
  }

  if (!structural_ok) {
    std::fprintf(stderr,
                 "[%s] FAIL: a structural assertion failed -- no timing from "
                 "this run may be quoted\n",
                 kTag);
    sycl::free(base_buf, *q);
    (void)zeMemFree(ctx, cand_buf);
    delete q;
    return 1;
  }

  auto summarise = [](ArmStats& s, const std::vector<double>& pool) {
    s.ok = !pool.empty();
    s.med_ns = median_of(pool);
    s.p10_ns = pct_of(pool, 0.10);
    s.p90_ns = pct_of(pool, 0.90);
    if (!s.pass_medians.empty()) {
      const double lo = *std::min_element(s.pass_medians.begin(),
                                          s.pass_medians.end());
      const double hi = *std::max_element(s.pass_medians.begin(),
                                          s.pass_medians.end());
      s.pass_spread = (s.med_ns > 0.0) ? (hi - lo) / s.med_ns : 0.0;
    }
  };
  summarise(base, base_pool);
  summarise(cand, cand_pool);

  if (!base.ok || !cand.ok || base.med_ns <= 0.0 || cand.med_ns <= 0.0) {
    std::fprintf(stderr,
                 "[%s] FAIL: an arm produced no usable sample (base ok=%d "
                 "med=%.0f, cand ok=%d med=%.0f) -- vacuity, not speed\n",
                 kTag, (int)base.ok, base.med_ns, (int)cand.ok, cand.med_ns);
    sycl::free(base_buf, *q);
    (void)zeMemFree(ctx, cand_buf);
    delete q;
    return 1;
  }

  // ---- REPORT. Relative only, with the spread floor stated. ----
  std::printf("bytes=%zu rounds=%d passes=%d\n", nbytes, rounds, passes);
  std::printf("pass medians base:");
  for (double m : base.pass_medians) std::printf(" %.0f", m);
  std::printf(" ns\n");
  std::printf("pass medians cand:");
  for (double m : cand.pass_medians) std::printf(" %.0f", m);
  std::printf(" ns\n");
  std::printf("base(malloc_host)        median=%.0f ns p10=%.0f p90=%.0f "
              "pass_spread=%.4f\n",
              base.med_ns, base.p10_ns, base.p90_ns, base.pass_spread);
  std::printf("cand(BIAS_UNCACHED)      median=%.0f ns p10=%.0f p90=%.0f "
              "pass_spread=%.4f\n",
              cand.med_ns, cand.p10_ns, cand.p90_ns, cand.pass_spread);

  const double floor_spread =
      (base.pass_spread > cand.pass_spread) ? base.pass_spread
                                            : cand.pass_spread;
  const double delta_ns = cand.med_ns - base.med_ns;
  const double rel = (base.med_ns > 0.0) ? delta_ns / base.med_ns : 0.0;
  const bool clears = std::abs(delta_ns) > floor_spread * base.med_ns;
  std::printf("spread_floor=%.4f delta=%+.0f ns (%+.2f%%) clears=%.0f\n",
              floor_spread, delta_ns, rel * 100.0, (double)clears);

  // The probe prints NO adopt/decline verdict. It is a load-sensitive host
  // wall-clock instrument on a box whose load1 does not stay low, so the number
  // above is a relative precondition measurement for the operator, never a
  // performance claim -- and this file says so rather than letting a future
  // reader treat a printed number as an achievement.
  std::printf("[%s] no verdict emitted: this is a precondition measurement; "
              "the S1 decision is the operator's, on nk/nw/nr and the host "
              "buckets, never on a tok/s\n", kTag);
  std::printf("ALLOC UNCACHED DOORBELL PROBE: PASS\n");
  std::printf("RESULT: PASS\n");

  sycl::free(base_buf, *q);
  (void)zeMemFree(ctx, cand_buf);
  delete q;
  return 0;
}
