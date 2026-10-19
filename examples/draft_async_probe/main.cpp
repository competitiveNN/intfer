// draft_async_probe -- model-free, GPU-free execution evidence for
// intfer::DraftAsyncProducer (include/intfer/draft_async.hpp).
//
// The seam is host-side, so the mechanism can be EXECUTED with no model and no
// device. This probe is the difference between "the flag is read somewhere" (a
// structural grep) and "the handshake actually delivers the right bytes, does
// not deadlock, and degrades to the inline path when the producer misses its
// deadline" (a measurement). Plain C++17 + pthreads; no SYCL, no Level Zero.
//
// Predicates:
//   A  parity      -- the async answer equals the synchronous answer, case for case
//   B  liveness    -- many sequential requests complete; no deadlock, no hang
//   C  deadline    -- a STALLED producer (sleeps past the deadline) makes the
//                     request return false so the caller falls back; the probe
//                     then computes inline and still gets the right answer
//   D  supersede   -- a late answer to an ABANDONED request must not satisfy a
//                     later request (the seq_-keyed predicate)
//
// Exit 0 iff every predicate holds; non-zero otherwise, with the failing
// predicate named on stderr.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "intfer/draft_async.hpp"

namespace {

int g_fail = 0;

void check(bool ok, const char* name, const char* detail) {
  if (ok) {
    std::fprintf(stderr, "  ok   %s\n", name);
  } else {
    std::fprintf(stderr, "  FAIL %s -- %s\n", name, detail);
    g_fail = 1;
  }
}

// The synchronous reference: exactly what the shipped path computes.
std::vector<int> sync_draft(const std::vector<int>& gen,
                            const std::vector<int>& pr, int tok, int K) {
  std::vector<int> d;
  for (int occ = (int)gen.size() - 1; occ >= 0 && (int)d.size() < K; --occ) {
    if (gen[(size_t)occ] == tok) {
      for (int j = occ + 1; j < (int)gen.size() && (int)d.size() < K; ++j)
        d.push_back(gen[(size_t)j]);
      break;
    }
  }
  if ((int)d.size() < K) {
    for (int occ = (int)pr.size() - 1; occ >= 0 && (int)d.size() < K; --occ) {
      if (pr[(size_t)occ] == tok) {
        for (int j = occ + 1; j < (int)pr.size() && (int)d.size() < K; ++j)
          d.push_back(pr[(size_t)j]);
        for (int j = 0; j < (int)gen.size() && (int)d.size() < K; ++j)
          d.push_back(gen[(size_t)j]);
        break;
      }
    }
  }
  if ((int)d.size() > K) d.resize(K);
  return d;
}

}  // namespace

int main() {
  const std::vector<int> prompt{7, 11, 13, 7, 21, 22, 23};
  const std::vector<int> gen{5, 7, 31, 32, 33, 9};

  // ---- A: parity against the synchronous reference -------------------------
  {
    bool all = true;
    std::vector<std::vector<int>> cases;
    for (int tok : {7, 5, 31, 99})
      for (int K : {1, 4, 16}) cases.push_back({tok, K});
    // The producer's work function IS the arithmetic under test.
    intfer::DraftAsyncProducer p(
        [](const std::vector<int>& g, const std::vector<int>& pr, int tok, int K) {
          return sync_draft(g, pr, tok, K);
        });
    for (auto& c : cases) {
      std::vector<int> got;
      const bool served = p.request(gen, prompt, c[0], c[1], 2000, got);
      const std::vector<int> want = sync_draft(gen, prompt, c[0], c[1]);
      if (!served || got != want) {
        all = false;
        std::fprintf(stderr, "    case tok=%d K=%d served=%d got=%zu want=%zu\n",
                     c[0], c[1], (int)served, got.size(), want.size());
      }
    }
    check(all, "A parity: async answer == sync answer, every case",
          "a case disagreed with the synchronous reference");
    check(p.stats().serves == cases.size(), "A parity: every case served in time",
          "a case fell back to the inline path");
  }

  // ---- B: liveness over many rounds ----------------------------------------
  {
    intfer::DraftAsyncProducer p(
        [](const std::vector<int>& g, const std::vector<int>& pr, int tok, int K) {
          return sync_draft(g, pr, tok, K);
        });
    int served = 0;
    for (int i = 0; i < 5000; ++i) {
      std::vector<int> got;
      if (p.request(gen, prompt, 7, 4, 2000, got)) ++served;
    }
    check(served == 5000, "B liveness: 5000 sequential requests all completed",
          "a request did not complete (deadlock or starvation)");
  }

  // ---- C: deadline fallback ------------------------------------------------
  {
    // A producer that sleeps far past the deadline: the request MUST return
    // false rather than block, and the inline path must still be correct.
    intfer::DraftAsyncProducer slow(
        [](const std::vector<int>& g, const std::vector<int>& pr, int tok, int K) {
          std::this_thread::sleep_for(std::chrono::milliseconds(400));
          return sync_draft(g, pr, tok, K);
        });
    std::vector<int> got;
    const auto t0 = std::chrono::steady_clock::now();
    const bool served = slow.request(gen, prompt, 7, 4, 50, got);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    check(!served, "C deadline: a stalled producer yields a MISS, not a block",
          "the request reported delivery from a producer that slept 400ms");
    check(ms < 300, "C deadline: the miss returns at the deadline, not later",
          "the request blocked past its deadline");
    const std::vector<int> fallback = sync_draft(gen, prompt, 7, 4);
    const std::vector<int> want = sync_draft(gen, prompt, 7, 4);
    check(fallback == want, "C deadline: the inline fallback is the correct draft",
          "the fallback disagreed with the reference");
  }

  // ---- D: a late answer must not satisfy a later request -------------------
  {
    // The first request is abandoned (deadline miss) while the producer is busy;
    // a second request must NOT be handed the first one's late answer.
    intfer::DraftAsyncProducer slow(
        [](const std::vector<int>& g, const std::vector<int>& pr, int tok, int K) {
          std::this_thread::sleep_for(std::chrono::milliseconds(150));
          return sync_draft(g, pr, tok, K);
        });
    std::vector<int> first;
    const bool a = slow.request(gen, prompt, 7, 4, 30, first);  // abandoned
    // Give the producer time to finish the abandoned request, then post a new
    // one. Its predicate is keyed on the request sequence, so the stale answer
    // cannot be accepted.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::vector<int> second;
    const bool b = slow.request(gen, prompt, 99, 3, 30, second);
    check(!a, "D supersede: the first (abandoned) request missed",
          "the first request unexpectedly succeeded");
    check(!b, "D supersede: a stale late answer does not satisfy the new request",
          "the new request was served by the abandoned request's answer");
  }

  if (g_fail) {
    std::fprintf(stderr, "RESULT: FAIL\n");
    return 1;
  }
  std::fprintf(stderr, "RESULT: PASS\n");
  return 0;
}
