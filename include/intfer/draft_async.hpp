// DRAFT-ASYNC: the asynchronous producer/consumer split for the draft seam.
//
// docs/DFLASH_ON_NPU.txt sec.1 item 3. `INTFER_DRAFT_SOURCE=pinned` shipped the
// draft RING -- a two-slot pinned staging buffer with back-pressure, a publish
// handshake and staleness cancellation -- but the producer stayed SYNCHRONOUS:
// the consumer computed the n-gram draft itself and immediately read it back,
// so the ring had no second thread to serve. This header is that second thread.
//
// WHY A HEADER, AND WHY NO SYCL: the seam is host-side (the producer computes
// token ids; only the bounded ids array crosses into the ring), so the whole
// mechanism is plain C++ and can be EXECUTED without a model and without a GPU
// -- see examples/draft_async_probe. A seam that can only be exercised by
// loading the 27B is a seam nothing can certify; this one is proven by a
// host-only probe and then used, unchanged, by the CLI.
//
// NUMERICS: the producer calls the SAME produce function the synchronous path
// calls, on a snapshot of the request-time inputs, so the drafts are identical
// by construction -- this is a scheduling change, not an arithmetic one. There
// is no numerics contract to agree because no value changes.
//
// LIVENESS, BY CONSTRUCTION: the consumer waits with a DEADLINE. A producer
// that is slow, stalled, or dead cannot hang the decode loop and cannot change
// the result -- on a miss the caller falls back to computing inline, which is
// the shipped path. A background thread that can wedge a decode loop would be
// worse than no thread at all, so the fallback is the contract, not a safety
// net bolted on afterwards.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace intfer {

// The producer's work function. It takes the REQUEST-TIME SNAPSHOT of the
// producer's inputs (so the background thread never reads state the main
// thread is mutating), the token, and the draft width.
using DraftProduceFn = std::function<std::vector<int>(
    const std::vector<int>&, const std::vector<int>&, int, int)>;

class DraftAsyncProducer {
 public:
  struct Stats {
    std::uint64_t serves = 0;   // requests the producer answered in time
    std::uint64_t misses = 0;   // requests that fell back to the inline path
    std::uint64_t errors = 0;   // produce() threw; the caller fell back
  };

  explicit DraftAsyncProducer(DraftProduceFn fn) : fn_(std::move(fn)) {
    thread_ = std::thread([this] { run(); });
  }

  ~DraftAsyncProducer() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

  DraftAsyncProducer(const DraftAsyncProducer&) = delete;
  DraftAsyncProducer& operator=(const DraftAsyncProducer&) = delete;

  // Post one request and wait at most `deadline_ms` for the producer's answer.
  // Returns true with `out` filled iff the answer arrived in time; false on a
  // miss or a producer error (the caller then computes inline). Never blocks
  // past the deadline.
  //
  // `gen_snap` / `prompt_snap` are the caller's snapshots; they are copied into
  // the request under the lock so the background thread reads a private copy.
  bool request(const std::vector<int>& gen_snap,
               const std::vector<int>& prompt_snap, int tok, int width,
               int deadline_ms, std::vector<int>& out) {
    std::unique_lock<std::mutex> lk(m_);
    if (stop_) return false;
    gen_snap_ = gen_snap;
    prompt_snap_ = prompt_snap;
    tok_ = tok;
    width_ = width;
    ready_ = false;
    have_req_ = true;
    const std::uint64_t want = ++seq_;
    cv_.notify_all();
    // The predicate keys on `seq_` as well as `ready_`: a late answer to an
    // ABANDONED request must never satisfy a later one.
    const bool delivered =
        cv_.wait_for(lk, std::chrono::milliseconds(deadline_ms),
                     [this, want] { return ready_ && seq_ == want; }) &&
        ready_ && seq_ == want;
    if (delivered) {
      out = out_;
      ++stats_.serves;
      return true;
    }
    // Abandon: the producer is still busy, or errored. Either way this request
    // is no longer ours to wait on, and its answer must not be publishable.
    have_req_ = false;
    ++stats_.misses;
    return false;
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lk(m_);
    return stats_;
  }

 private:
  void run() {
    for (;;) {
      std::vector<int> gen, prompt;
      int tok = 0, width = 0;
      std::uint64_t seq = 0;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return stop_ || have_req_; });
        if (stop_) return;
        gen = gen_snap_;
        prompt = prompt_snap_;
        tok = tok_;
        width = width_;
        seq = seq_;
        have_req_ = false;  // take it: a second post cannot steal this one
      }
      // Compute OUTSIDE the lock -- this is the whole point of the thread.
      std::vector<int> d;
      bool ok = true;
      try {
        d = fn_(gen, prompt, tok, width);
      } catch (...) {
        ok = false;
      }
      {
        std::unique_lock<std::mutex> lk(m_);
        if (stop_) return;
        if (ok) {
          // Publish only if no newer request has superseded ours; a superseded
          // answer is dropped here rather than handed to the wrong caller.
          if (seq_ == seq) {
            out_ = std::move(d);
            ready_ = true;
          }
        } else {
          ++stats_.errors;
        }
        cv_.notify_all();
      }
    }
  }

  DraftProduceFn fn_;
  std::thread thread_;
  mutable std::mutex m_;
  std::condition_variable cv_;
  bool stop_ = false;
  bool have_req_ = false;
  bool ready_ = false;
  std::vector<int> gen_snap_, prompt_snap_, out_;
  int tok_ = 0, width_ = 0;
  std::uint64_t seq_ = 0;
  Stats stats_;
};

}  // namespace intfer
