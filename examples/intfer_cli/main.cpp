// examples/intfer_cli/main.cpp
// intfer CLI — end-to-end text generation for Qwen3.5 (text-only) with
// runtime-dispatched quantized Linear backend:
//
//   * CpuBackend : dequant-on-the-fly host matmul (verified CPU reference,
//                  matches the transformers oracle).
//   * GpuBackend : SYCL tensor-parallel via intfer::Engine on the 2x B580
//                  fleet (dequant-on-the-fly GEMM kernels; INT8 path lowers
//                  to Intel XMX DPAS). Per-layer RMSNorm / mRoPE / attention
//                  compute / SSM / lm_head / sampling stay on the host
//                  (same as examples/qwen_forward_gpu).
//
// CLI surface:
//   -m <model.intifer>   (required)
//   -s <safetensors>     (auto-detected from common locations)
//   -p "prompt"  /  -P prompt_file
//   -n N  max new tokens
//   -t T  -k K  -T P  -S SEED  sampling controls
//   --gpu / --no-gpu    default: auto (use GPU if 2+ Arc devices discoverable)
//   --no-stream / --ids
//   --tokenizer / --tokenizer-py / --detok-py
//
// The full forward (24 layers, hybrid SSM/attn) is identical across backends
// except for the 7 quantized Linears per layer (q,k,v,o + gate,up,down) which
// go through `backend.run_linear(name, x)` and `backend.rot_for(name)`.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <immintrin.h>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
static void segv_handler(int sig){
  void* bt[32];
  int n = backtrace(bt, 32);
  char msg[256];
  int m = snprintf(msg, sizeof(msg), "\n[intfer] FATAL: signal %d, backtrace:\n", sig);
  (void)write(STDERR_FILENO, msg, m);
  backtrace_symbols_fd(bt, n, STDERR_FILENO);
  // Also dump raw addresses for addr2line.
  for(int i=0;i<n;i++){
    m = snprintf(msg, sizeof(msg), "  #%d %p\n", i, bt[i]);
    (void)write(STDERR_FILENO, msg, m);
  }
  _exit(139);
}
#include <chrono>
#include <algorithm>
#include <random>
#include <functional>
#include <iostream>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <ctime>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <fcntl.h>
#include <errno.h>

#include "intfer/safetensors.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/spinquant.hpp"
#include "http.hpp"
#include "openai.hpp"

#ifdef INTFER_HAS_GPU_BACKEND
#include "intfer/engine.hpp"
#endif

using namespace intfer;

// ============================================================================
// Lightweight config.json reader (Qwen3.5 family)
// ============================================================================
struct ModelConfig {
  int H        = 1024;
  int layers   = 24;
  int inter    = 3584;
  int n_heads  = 8;
  int n_kv     = 2;
  int head_dim = 256;
  int n_k_heads = 16;
  int n_v_heads = 16;
  int k_head_dim = 128;
  int v_head_dim = 128;
  int conv_k   = 4;
  double rope_theta = 1e7;
  int rotary_dim = 64;
  float rms_eps  = 1e-6f;
  int vocab      = 248320;
  int full_attention_interval = 4;
  bool attn_output_gate = true;
  bool tied_embeddings = true;
  // Linear attention conv dim (computed from head counts).
  int conv_dim = 0;  // filled in by build()
  int sdim     = 0;  // filled in by build()
  std::string model_id = "Qwen3.5-0.8B";

  void build() {
    conv_dim = n_k_heads * k_head_dim * 2 + n_v_heads * v_head_dim;
    sdim     = n_v_heads * k_head_dim * v_head_dim;
  }
};

// Tiny JSON helper: extract a numeric value by dotted key from a JSON string.
// Supports "key" or "nested.key". Returns `def` if not found.
static int json_get_int(const std::string& json, const std::string& key, int def = -1) {
  // Navigate nested keys.
  std::string remaining = key;
  std::string cur = json;
  while (true) {
    size_t dot = remaining.find('.');
    std::string field;
    if (dot == std::string::npos) { field = remaining; remaining.clear(); }
    else { field = remaining.substr(0, dot); remaining = remaining.substr(dot + 1); }
    // Find "\"field\":".
    std::string pat = "\"" + field + "\"";
    size_t p = cur.find(pat);
    if (p == std::string::npos) return def;
    size_t colon = cur.find(':', p + pat.size());
    if (colon == std::string::npos) return def;
    // skip whitespace
    size_t q = colon + 1;
    while (q < cur.size() && (cur[q] == ' ' || cur[q] == '\t' || cur[q] == '\n' || cur[q] == '\r')) ++q;
    if (q >= cur.size()) return def;
    if (cur[q] == '"') {
      // string value — not a number.
      return def;
    }
    if (cur[q] == '{' || cur[q] == '[') {
      // nested object/array — continue navigating.
      continue;
    }
    // parse number
    size_t end = q;
    while (end < cur.size() && (std::isdigit((unsigned char)cur[end]) || cur[end] == '-' || cur[end] == '.' || cur[end] == 'e' || cur[end] == 'E' || cur[end] == '+')) ++end;
    std::string num = cur.substr(q, end - q);
    if (num.empty()) return def;
    try {
      if (remaining.empty()) return std::stoi(num);
      cur = cur.substr(q); // continue
      continue;
    } catch (...) { return def; }
  }
}

static double json_get_double(const std::string& json, const std::string& key, double def = -1.0) {
  // find the key, skip to its value
  std::string pat = "\"" + key + "\"";
  size_t p = json.find(pat);
  if (p == std::string::npos) return def;
  size_t colon = json.find(':', p + pat.size());
  if (colon == std::string::npos) return def;
  size_t q = colon + 1;
  while (q < json.size() && (json[q] == ' ' || json[q] == '\t' || json[q] == '\n' || json[q] == '\r')) ++q;
  if (q >= json.size()) return def;
  if (json[q] == '"') return def;
  size_t end = q;
  while (end < json.size() && (std::isdigit((unsigned char)json[end]) || json[end] == '-' || json[end] == '.' || json[end] == 'e' || json[end] == 'E' || json[end] == '+')) ++end;
  if (end == q) return def;
  try { return std::stod(json.substr(q, end - q)); } catch (...) { return def; }
}

static bool json_get_bool(const std::string& json, const std::string& key, bool def = false) {
  std::string pat = "\"" + key + "\"";
  size_t p = json.find(pat);
  if (p == std::string::npos) return def;
  size_t colon = json.find(':', p + pat.size());
  if (colon == std::string::npos) return def;
  size_t q = colon + 1;
  while (q < json.size() && (json[q] == ' ' || json[q] == '\t' || json[q] == '\n' || json[q] == '\r')) ++q;
  if (q + 4 <= json.size() && json.compare(q, 4, "true") == 0) return true;
  if (q + 5 <= json.size() && json.compare(q, 5, "false") == 0) return false;
  return def;
}

// Read model dimensions from <dir>/config.json. Returns defaults on failure.
static ModelConfig load_model_config(const std::string& dir) {
  ModelConfig cfg;
  std::string config_path = dir + "/config.json";
  FILE* f = fopen(config_path.c_str(), "rb");
  if (!f) return cfg;
  char buf[65536];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  if (n == 0) return cfg;
  buf[n] = '\0';
  std::string json(buf, n);

  cfg.model_id = dir;
  size_t sp = dir.rfind('/');
  if (sp != std::string::npos) cfg.model_id = dir.substr(sp + 1);

  // text_config.* fields.
  // Find "text_config" block and search within it.
  size_t tc = json.find("\"text_config\"");
  std::string sub;
  if (tc != std::string::npos) {
    size_t ob = json.find('{', tc);
    if (ob != std::string::npos) {
      int depth = 0; size_t i = ob;
      for (; i < json.size(); ++i) {
        if (json[i] == '{') ++depth;
        else if (json[i] == '}') { --depth; if (depth == 0) break; }
      }
      sub = json.substr(ob, i - ob + 1);
    }
  }
  const std::string& lookup = sub.empty() ? json : sub;

  cfg.H            = json_get_int(lookup, "hidden_size", cfg.H);
  cfg.layers       = json_get_int(lookup, "num_hidden_layers", cfg.layers);
  cfg.inter        = json_get_int(lookup, "intermediate_size", cfg.inter);
  cfg.n_heads      = json_get_int(lookup, "num_attention_heads", cfg.n_heads);
  cfg.n_kv         = json_get_int(lookup, "num_key_value_heads", cfg.n_kv);
  cfg.head_dim     = json_get_int(lookup, "head_dim", cfg.head_dim);
  cfg.rms_eps      = json_get_double(lookup, "rms_norm_eps", cfg.rms_eps);
  cfg.vocab        = json_get_int(lookup, "vocab_size", cfg.vocab);
  cfg.rope_theta   = json_get_double(lookup, "rope_theta", cfg.rope_theta);
  cfg.full_attention_interval = json_get_int(lookup, "full_attention_interval", cfg.full_attention_interval);
  cfg.attn_output_gate  = json_get_bool(lookup, "attn_output_gate", cfg.attn_output_gate);
  cfg.n_k_heads   = json_get_int(lookup, "linear_num_key_heads", cfg.n_k_heads);
  cfg.k_head_dim  = json_get_int(lookup, "linear_key_head_dim", cfg.k_head_dim);
  cfg.n_v_heads   = json_get_int(lookup, "linear_num_value_heads", cfg.n_v_heads);
  cfg.v_head_dim  = json_get_int(lookup, "linear_value_head_dim", cfg.v_head_dim);
  cfg.conv_k      = json_get_int(lookup, "linear_conv_kernel_dim", cfg.conv_k);
  cfg.tied_embeddings = json_get_bool(lookup, "tie_word_embeddings", cfg.tied_embeddings);

  // rope_parameters
  size_t rp = lookup.find("\"rope_parameters\"");
  if (rp != std::string::npos) {
    size_t ob = lookup.find('{', rp);
    if (ob != std::string::npos) {
      int depth = 0; size_t i = ob;
      for (; i < lookup.size(); ++i) {
        if (lookup[i] == '{') ++depth;
        else if (lookup[i] == '}') { --depth; if (depth == 0) break; }
      }
      std::string rp_obj = lookup.substr(ob, i - ob + 1);
      cfg.rope_theta = json_get_double(rp_obj, "rope_theta", cfg.rope_theta);
      (void)json_get_int(rp_obj, "mrope_interleaved", 0);
      double prf = json_get_double(rp_obj, "partial_rotary_factor", -1.0);
      if(prf > 0.0 && prf <= 1.0) cfg.rotary_dim = (int)(cfg.head_dim * prf);
    }
  }

  cfg.build();
  return cfg;
}

// ============================================================================
// Model config (Qwen3.5-0.8B, text-only) — populated from config.json
// ============================================================================
static ModelConfig cfg;

static std::vector<float> inv_freq;
// Global rotation-id lookup, populated once after deserialize_intfer and
// used by CpuBackend::dequant_from_ir() for lazy dequant of large models.
static std::map<uint8_t, const Rotation*> rid_map_global;

static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }

// ============================================================================
// Shared host primitives
// ============================================================================
// Vectorized host math for the per-token elementwise hot loop (rmsnorm,
// l2norm, silu-mul). 8-wide AVX2/FMA with scalar tails via function-target
// (same pattern as dashq.cpp/gemm_rows_fma); runtime-gated on
// intfer::dashq::avx2_available(). These change FP rounding vs the scalar
// loop (vector reduction order + FMA contraction), so they are
// ORACLE-gated, not bit-gated: greedy token stream must match
// (scripts/test_oracle_equiv.sh). INTFER_HOST_AVX2=0 restores scalar.
__attribute__((target("avx2,fma")))
static float vsumsq_avx2(const float* x, int n) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    acc = _mm256_fmadd_ps(v, v, acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float out = _mm_cvtss_f32(s);
  for (; i < n; i++) out += x[i] * x[i];
  return out;
}
__attribute__((target("avx2,fma")))
static void vscale_add1_avx2(const float* x, const float* w, float* out,
                             float v, int n) {
  __m256 vv = _mm256_set1_ps(v);
  __m256 one = _mm256_set1_ps(1.0f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 wv = _mm256_loadu_ps(w + i);
    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, vv),
                                            _mm256_add_ps(one, wv)));
  }
  for (; i < n; i++) out[i] = (x[i]*v)*(1.0f+w[i]);
}
static void rmsnorm(const float* x, const float* w, float* out, int dim){
  static const bool avx2_ok = intfer::dashq::avx2_available();
  static const bool no_avx2 = (std::getenv("INTFER_HOST_AVX2") != nullptr &&
                               std::string(std::getenv("INTFER_HOST_AVX2")) == "0");
  float v;
  if (avx2_ok && !no_avx2 && dim >= 32) {
    v = vsumsq_avx2(x, dim);
    v = 1.0f/sqrtf(v/dim + cfg.rms_eps);
    vscale_add1_avx2(x, w, out, v, dim);
    return;
  }
  v=0; for(int i=0;i<dim;i++) v+=x[i]*x[i];
  v = 1.0f/sqrtf(v/dim + cfg.rms_eps);
  for(int i=0;i<dim;i++) out[i] = (x[i]*v)*(1.0f+w[i]);
}
static void l2norm(const float* x, float* out, int dim){
  static const bool avx2_ok = intfer::dashq::avx2_available();
  static const bool no_avx2 = (std::getenv("INTFER_HOST_AVX2") != nullptr &&
                               std::string(std::getenv("INTFER_HOST_AVX2")) == "0");
  float s;
  if (avx2_ok && !no_avx2 && dim >= 32) {
    s = vsumsq_avx2(x, dim);
  } else {
    s=0; for(int i=0;i<dim;i++) s+=x[i]*x[i];
  }
  float inv=1.0f/sqrtf(s+1e-6f);
  for(int i=0;i<dim;i++) out[i]=x[i]*inv;
}
static void rope(float* h, int pos){
  for(int d=0; d<cfg.rotary_dim/2; d++){
    float f = inv_freq[d]*pos;
    float c=cosf(f), s=sinf(f);
    float a=h[d], b=h[d+cfg.rotary_dim/2];
    h[d]               = a*c - b*s;
    h[d+cfg.rotary_dim/2]  = a*s + b*c;
  }
}
// CPU GEMM time accumulator (INTFER_TIME_GEMM=1): microseconds spent inside
// the three CpuBackend GEMM entry points, summed thread-safely. Printed next
// to the prefill/decode tok/s lines to split GEMM math vs everything else.
static long long g_cpu_gemm_us = 0;
struct CpuGemmTimer {
  std::chrono::high_resolution_clock::time_point t0;
  bool on;
  CpuGemmTimer(): t0(std::chrono::high_resolution_clock::now()),
    on(std::getenv("INTFER_TIME_GEMM") != nullptr) {}
  ~CpuGemmTimer(){
    if(!on) return;
    auto t1 = std::chrono::high_resolution_clock::now();
    long long us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    __sync_fetch_and_add(&g_cpu_gemm_us, us);
  }
};
// AVX2+FMA dot product for the CPU GEMM hot loop. 8-wide FMA over k with a
// single vector accumulator, scalar tail. Deterministic (fixed row order,
// fixed reduction order) but NOT bit-identical to the scalar loop (FMA
// contraction + vector reduction reorder FP rounding). Oracle-gated: greedy
// token stream must match (see scripts/test_oracle_equiv.sh); the bit-exact
// parity gate does not apply while INTFER_CPU_FMA_DOT=1.
// Function-target keeps AVX2/FMA codegen local (same pattern as dashq.cpp);
// runtime-gated on intfer::dashq::avx2_available() — FMA is baseline on all
// AVX2 CPUs that matter here (checked via /proc/cpuinfo: fma present).
__attribute__((target("avx2,fma")))
static float dot_fma_avx2(const float* a, const float* b, size_t n){
  __m256 acc = _mm256_setzero_ps();
  size_t i = 0;
  for(; i + 8 <= n; i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float out = _mm_cvtss_f32(s);
  for(; i < n; i++) out += a[i] * b[i];
  return out;
}
// Full row-loop in the target region: the OpenMP parallel-for lives INSIDE
// the avx2/fma function so the FMA dot inlines per row (a non-inline
// dot_fma_avx2 call costs ~25ns/row x 1.3M rows/GEMM — slower than scalar).
// `single_x` selects shared-x (run_linear/run_linear_into) vs M-row X
// (run_linear_batch: xr = Xp + (r/N)*Kloc) layout.
__attribute__((target("avx2,fma")))
static void gemm_rows_fma(const float* W, const float* Xp, float* Yp,
                          size_t rows, int N, size_t Kloc, int strideK,
                          bool single_x, long omp_floor){
#pragma omp parallel for schedule(static) if(rows >= (size_t)omp_floor)
  for(size_t r = 0; r < rows; r++){
    size_t m = single_x ? 0 : r / (size_t)N;
    int i = single_x ? (int)r : (int)(r % (size_t)N);
    Yp[r] = dot_fma_avx2(W + (size_t)i * strideK, Xp + m * Kloc, Kloc);
  }
}
// SpinQuant online rotation helpers. Hadamard rotations (the only kind the
// quantizer emits) take the O(n log n) FWHT path; anything else keeps the
// dense O(n^2) matvec. Both compute the same map up to summation order.
static void rot_apply(const float* x, const Rotation& R, float* out){
  if(R.is_hadamard){ apply_rotation_fwht(R, x, out); return; }
  int dim = (int)R.dim;
  for(int j=0;j<dim;j++){ float s=0; for(int i=0;i<dim;i++) s+=x[i]*R.mat[i*dim+j]; out[j]=s; }
}
static void rot_left(const float* y, const Rotation& R, float* out){
  if(R.is_hadamard){ apply_rotation_fwht(R, y, out); return; }
  int dim = (int)R.dim;
  for(int j=0;j<dim;j++){ float s=0; for(int i=0;i<dim;i++) s+=R.mat[j*dim+i]*y[i]; out[j]=s; }
}

// ============================================================================
// Linear backend interface
// ============================================================================
// The forward kernel calls `backend.run_linear(tensor_name, x)` for each of
// the 7 quantized Linears per layer (q/k/v/o + gate/up/down). The CPU
// implementation dequantizes the .intifer tensor to FP32 and does a host
// matmul. The GPU implementation submits a SYCL kernel to intfer::Engine,
// which shards the GEMM tensor-parallel across the 2x B580 fleet.
struct LinearBackend {
  virtual ~LinearBackend() = default;
  virtual const char* name() const = 0;
  // Run y = W x where W is the named quantized tensor. x has length t.K,
  // y has length t.N (both from the .intifer TensorIR for `tensor_name`).
  virtual std::vector<float> run_linear(const std::string& tensor_name,
                                        const std::vector<float>& x) = 0;
  // Allocation-free variant: writes y into `out` (resized only when the
  // size differs, so a reused scratch buffer performs zero mallocs).
  // Default impl wraps run_linear; backends override for zero-copy.
  virtual void run_linear_into(const std::string& tensor_name,
                               const std::vector<float>& x,
                               std::vector<float>& out){
    out = run_linear(tensor_name, x);
  }
  // Batched variant: X holds M concatenated rows of length K; Y holds M
  // rows of length N. Default loops over run_linear_into (correct for any
  // backend); GPU overrides with a single M-row launch. Bit-identical to M
  // sequential calls (each row is an independent GEMM row).
  virtual void run_linear_batch(const std::string& tensor_name,
                                const std::vector<float>& X,
                                std::vector<float>& Y, uint32_t M){
    const std::vector<float>* xp = &X;
    std::vector<float> row;
    size_t K = (M > 0) ? X.size() / M : 0;
    Y.clear();
    for(uint32_t m = 0; m < M; m++){
      row.assign(xp->begin() + (size_t)m * K, xp->begin() + (size_t)(m + 1) * K);
      std::vector<float> y1;
      run_linear_into(tensor_name, row, y1);
      Y.insert(Y.end(), y1.begin(), y1.end());
    }
  }
  // Wave variant: one independent GEMM per item (distinct tensors, distinct
  // inputs/outputs, no data dependencies between items). Default loops over
  // run_linear_into in order (bit-identical to sequential calls: each item
  // is an independent GEMM with its own output buffer). Backends whose
  // submit path can keep the device busy across launches (e.g. GPU: convert
  // all inputs, submit all kernels, wait once, read back all) override this
  // to amortize per-launch dispatch latency. Items MUST be independent:
  // item i's input/output buffers must not alias any other item's output.
  struct LinearWaveItem {
    std::string name;
    const std::vector<float>* x = nullptr;
    std::vector<float>* out = nullptr;
  };
  virtual void run_linear_wave(const LinearWaveItem* items, size_t n) {
    for (size_t i = 0; i < n; ++i)
      run_linear_into(items[i].name, *items[i].x, *items[i].out);
  }
  // M-row wave variant: items share one M-row input staging layout —
  // item i reads its (M x K_i) rows from a caller-owned staging vector.
  // Same independence contract as the scalar wave; the engine converts +
  // submits every item back-to-back and waits once. Bit-identical to
  // sequential run_linear_batch calls. INTFER_NO_WAVE=1 restores the loop.
  struct LinearWaveBatchItem {
    std::string name;
    const std::vector<float>* X = nullptr;  // M*K row-major
    std::vector<float>* Y = nullptr;        // M*N row-major
  };
  virtual void run_linear_wave_batch(const LinearWaveBatchItem* items,
                                     size_t n, uint32_t M) {
    for (size_t i = 0; i < n; ++i)
      run_linear_batch(items[i].name, *items[i].X, *items[i].Y, M);
  }
  // Fused MLP down-projection: gate/up GEMMs + silu-mul + down GEMM in one
  // backend call. Default returns false (caller runs its inline host
  // path: wave + silu-mul + down, mg/mu filled, trace ckpts live); GPU
  // overrides with the device-chained version when legal and returns
  // true (md filled, mg/mu NOT filled — their only consumers are the
  // silu-mul folded into the chain and trace-gated ckpts, which force
  // the host path).
  virtual bool run_mlp_fused(const std::string& gate, const std::string& up,
                             const std::string& down,
                             const std::vector<float>& rin,
                             std::vector<float>& mg, std::vector<float>& mu,
                             std::vector<float>& md, uint32_t M) {
    (void)gate; (void)up; (void)down; (void)rin;
    (void)mg; (void)mu; (void)md; (void)M;
    return false;
  }
  // SpinQuant rotation for this tensor (right-site for q/k/v/gate/up;
  // left-site for o/down; nullptr if no rotation).
  virtual const Rotation* rot_for(const std::string& tensor_name) = 0;
  // True if this backend uses SYCL queues. Lets the forward kernel skip
  // host-side reductions if the GEMM has already done an all-reduce.
  virtual bool is_distributed() const { return false; }
};

// ----------------------------------------------------------------------------
// CpuBackend — verified host dequant + matmul
// ----------------------------------------------------------------------------
struct QLinear { std::vector<float> w; const Rotation* R = nullptr; };
struct CpuBackend : public LinearBackend {
  // Lazily populated by layer (L) and slot name (q/k/v/o/gate/up/down).
  // For large models (e.g. 27B), we dequantize the relevant quantized tensor
  // on first use from the ModelIR (ir_) instead of pre-populating these maps,
  // which would otherwise need ~4 bytes * total_param_count of FP32 RAM.
  std::map<int, std::map<std::string, QLinear>> attn_q;
  std::map<int, std::map<std::string, QLinear>> mlp_q;
  std::map<int, std::map<std::string, QLinear>> la_q;
  // Source-of-truth ModelIR for lazy dequant (set externally).
  const ModelIR* ir_ = nullptr;
  // O(1) name -> IR tensor index, built once via build_ir_index().
  // Replaces the O(#tensors) linear scan in dequant_from_ir()/rot_for().
  std::unordered_map<std::string, size_t> ir_index;
  void build_ir_index(){
    ir_index.clear();
    if(!ir_) return;
    for(size_t i = 0; i < ir_->tensors.size(); i++)
      ir_index[ir_->tensors[i].name] = i;
  }
  // Safetensors reader for lazy fallback when .intifer lacks a tensor.
  // Set externally before run_linear() is called.
  const SafetensorsReader* sr_fallback = nullptr;
  const std::set<std::string>* host_names_fallback = nullptr;
  // For per-tensor name -> dequantized weight (built on demand for run_linear)
  // We build this on first call to run_linear for a given name; subsequent
  // calls reuse the cached FP32 weight.
  std::map<std::string, std::vector<float>> cache_w;
  std::map<std::string, const Rotation*> cache_R;
  std::map<std::string, int> cache_N, cache_K;

  const char* name() const override { return "cpu"; }
  bool is_distributed() const override { return false; }

  // Lazy-dequant from ModelIR by full HF name. Returns true and fills out/rot
  // if found; returns false if the tensor is not in the IR.
  bool dequant_from_ir(const std::string& name, std::vector<float>& out,
                       const Rotation*& rot){
    if(!ir_) return false;
    auto it = ir_index.find(name);
    if(it == ir_index.end()) return false;
    const auto& t = ir_->tensors[it->second];
    intfer::dashq::dequant_tensor(*ir_, t, out);
    auto rit = rid_map_global.find(t.qcfg.rotation_id);
    rot = (rit != rid_map_global.end()) ? rit->second : nullptr;
    return !out.empty();
  }

  // Build a name->(W, R, N, K) cache entry for `name` from the per-layer
  // attn_q / mlp_q maps. The full HF tensor name is something like
  // "model.language_model.layers.3.self_attn.q_proj.weight"; we split on
  // the layer index to look up the right slot. N (output dim) depends on
  // the slot; K (contraction dim) is cfg.H for q/k/v/o/gate/up and cfg.inter
  // for mlp.down_proj.
  bool ensure_cached(const std::string& name){
    if(cache_w.count(name)) return true;
    size_t p = name.find("model.language_model.layers.");
    if(p == std::string::npos) return false;
    int L = std::atoi(name.c_str() + p + 28);
    std::string suf = name.substr(p + 28);
    while(!suf.empty() && suf[0] >= '0' && suf[0] <= '9') suf = suf.substr(1);
    QLinear ql;
    int N=0, K=0;
    if(suf.find("self_attn.q_proj") != std::string::npos){        ql = attn_q[L]["q"];   N=2*cfg.n_heads*cfg.head_dim; K=cfg.H; }
    else if(suf.find("self_attn.k_proj") != std::string::npos){   ql = attn_q[L]["k"];   N=cfg.n_kv*cfg.head_dim; K=cfg.H; }
    else if(suf.find("self_attn.v_proj") != std::string::npos){   ql = attn_q[L]["v"];   N=cfg.n_kv*cfg.head_dim; K=cfg.H; }
    else if(suf.find("self_attn.o_proj") != std::string::npos){   ql = attn_q[L]["o"];   N=cfg.H;    K=cfg.n_heads*cfg.head_dim; }
    else if(suf.find("mlp.gate_proj") != std::string::npos){      ql = mlp_q[L]["gate"]; N=cfg.inter; K=cfg.H; }
    else if(suf.find("mlp.up_proj") != std::string::npos){        ql = mlp_q[L]["up"];   N=cfg.inter; K=cfg.H; }
    else if(suf.find("mlp.down_proj") != std::string::npos){      ql = mlp_q[L]["down"]; N=cfg.H;     K=cfg.inter; }
    else if(suf.find("linear_attn.in_proj_qkv") != std::string::npos){
      auto itL = la_q.find(L);
      if(itL != la_q.end()){ auto itS = itL->second.find("qkv"); if(itS != itL->second.end()) ql = itS->second; }
      N=cfg.conv_dim; K=cfg.H;
    }
    else if(suf.find("linear_attn.in_proj_z") != std::string::npos){
      auto itL = la_q.find(L);
      if(itL != la_q.end()){ auto itS = itL->second.find("z"); if(itS != itL->second.end()) ql = itS->second; }
      N=cfg.n_v_heads*cfg.v_head_dim; K=cfg.H;
    }
    else if(suf.find("linear_attn.in_proj_b") != std::string::npos){
      auto itL = la_q.find(L);
      if(itL != la_q.end()){ auto itS = itL->second.find("b"); if(itS != itL->second.end()) ql = itS->second; }
      N=cfg.n_v_heads; K=cfg.H;
    }
    else if(suf.find("linear_attn.in_proj_a") != std::string::npos){
      auto itL = la_q.find(L);
      if(itL != la_q.end()){ auto itS = itL->second.find("a"); if(itS != itL->second.end()) ql = itS->second; }
      N=cfg.n_v_heads; K=cfg.H;
    }
    else if(suf.find("linear_attn.out_proj") != std::string::npos){
      auto itL = la_q.find(L);
      if(itL != la_q.end()){ auto itS = itL->second.find("out"); if(itS != itL->second.end()) ql = itS->second; }
      N=cfg.H; K=cfg.n_v_heads*cfg.v_head_dim;
    }
    else return false;
    // Lazy dequant from ModelIR if the per-layer map doesn't have FP32 data
    // (large-model path). The pre-populated attn_q/mlp_q maps are only used
    // for small models (0.8B) where eager dequant is cheap.
    if(ql.w.empty()){
      const Rotation* R = nullptr;
      std::vector<float> v;
      if(dequant_from_ir(name, v, R)){
        ql.w = std::move(v);
        ql.R = R;
      }
    }
    // Lazy fallback to safetensors for layers missing from .intifer
    if(ql.w.empty() && sr_fallback && host_names_fallback){
      std::vector<float> v;
      if(sr_fallback->read(name, v)){
        ql.w = std::move(v);
        ql.R = nullptr;
      }
    }
    if(ql.w.empty()) return false;
    // Trust the dequantized shape; the hardcoded N/K table above can go stale
    // when weight layouts change (e.g. fused q+gate projections).
    {
      size_t wn = ql.w.size();
      if(K > 0 && wn % (size_t)K == 0 && (int)(wn / (size_t)K) != N){
        std::fprintf(stderr, "[intfer] shape fix %s: N %d->%zu (K=%d)\n",
                     name.c_str(), N, wn / (size_t)K, K);
        N = (int)(wn / (size_t)K);
      }
    }
    cache_w[name] = ql.w;
    cache_R[name] = ql.R;
    cache_N[name] = N;
    cache_K[name] = K;
    return true;
  }

  std::vector<float> run_linear(const std::string& name,
                                const std::vector<float>& x) override {
    CpuGemmTimer tmgm;
    // Fast path: cached entry (both eager models and post-warmup JIT).
    // NOTE: std::map::count + operator[] does TWO tree lookups per GEMM;
    // use a single find() — this loop runs ~7×layers times per token.
    auto it = cache_w.find(name);
    if(it == cache_w.end()){
      if(!ensure_cached(name)){ std::fprintf(stderr, "cpu: unknown tensor %s\n", name.c_str()); return {}; }
      it = cache_w.find(name);
    }
    int N = cache_N[name];
    int K = cache_K[name];
    const float* W = it->second.data();
    size_t xn = x.size();
    if(xn < (size_t)K) K = (int)xn; // tolerate over-wide x (attn+gate probe)
    std::vector<float> y(N, 0.f);
    const float* xp = x.data();
    // Same OpenMP row-split as run_linear_into (bit-identical scalar rows).
    // INTFER_CPU_FMA_DOT=1 selects the AVX2+FMA dot (8x math throughput;
    // oracle-gated, not bit-identical — see dot_fma_avx2).
    static const bool fma_dot = (std::getenv("INTFER_CPU_FMA_DOT") != nullptr)
        && intfer::dashq::avx2_available();
    if(fma_dot){
      gemm_rows_fma(W, xp, y.data(), (size_t)N, N, (size_t)K, cache_K[name],
                    true, 2048);
    } else {
#pragma omp parallel for schedule(static) if(N >= 2048)
      for(int i=0;i<N;i++){ const float* wr=W+(size_t)i*cache_K[name]; float s=0;
        for(int k=0;k<K;k++) s+=wr[k]*xp[k]; y[i]=s; }
    }
    if(!keep_cached){
      // JIT path (27B): free the weight right after its GEMM so peak RAM
      // stays at ~one tensor instead of the full ~109 GB FP32 model.
      cache_w.erase(it);
      cache_R.erase(name);
    }
    return y;
  }
  // CPU override of the batched GEMM: one parallel-for over M*N output rows
  // (row (m,i) is fully independent). Forks ONCE per linear (not per row:
  // the default base impl would fork M times inside run_linear_into), and
  // the N>=2048 serial gate must not apply — M*N rows amortize the fork
  // (e.g. M=8 x N=1024 q/o: 8192 rows). Bit-identical: same scalar row
  // loop as run_linear_into, rows concatenated in order.
  void run_linear_batch(const std::string& tensor_name,
                        const std::vector<float>& X,
                        std::vector<float>& Y, uint32_t M) override {
    CpuGemmTimer tmgm;
    size_t K = (M > 0) ? X.size() / M : 0;
    auto it = cache_w.find(tensor_name);
    if(it == cache_w.end()){
      if(!ensure_cached(tensor_name)){ Y.clear(); return; }
      it = cache_w.find(tensor_name);
    }
    int N = cache_N[tensor_name];
    int Kk = cache_K[tensor_name];
    if((size_t)Kk < K) K = (size_t)Kk;
    if(K == 0 || N == 0 || M == 0){ Y.clear(); return; }
    const float* W = it->second.data();
    const float* Xp = X.data();
    const int strideK = cache_K[tensor_name];
    const size_t Kloc = K;
    Y.assign((size_t)M * N, 0.f);
    float* Yp = Y.data();
    static const bool fma_dot_b = (std::getenv("INTFER_CPU_FMA_DOT") != nullptr)
        && intfer::dashq::avx2_available();
    if(fma_dot_b){
      gemm_rows_fma(W, Xp, Yp, (size_t)M * (size_t)N, N, Kloc, strideK,
                    false, 2048);
    } else {
#pragma omp parallel for schedule(static) if((size_t)M * N >= 2048)
      for(size_t r = 0; r < (size_t)M * N; r++){
        size_t m = r / (size_t)N;
        int i = (int)(r % (size_t)N);
        const float* wr = W + (size_t)i * strideK;
        const float* xr = Xp + m * Kloc;
        float s = 0;
        for(size_t k = 0; k < Kloc; k++) s += wr[k] * xr[k];
        Yp[r] = s;
      }
    }
    if(!keep_cached){
      cache_w.erase(it);
      cache_R.erase(tensor_name);
    }
  }
  void run_linear_into(const std::string& name,
                       const std::vector<float>& x,
                       std::vector<float>& out) override {
    CpuGemmTimer tmgm;
    // Same math as run_linear, but writes into the caller's buffer.
    // out keeps its capacity across tokens: steady-state zero mallocs.
    // NOTE: the JIT (!keep_cached) path is unchanged — weights are still
    // freed after each GEMM; only the OUTPUT allocation is eliminated.
    auto it = cache_w.find(name);
    if(it == cache_w.end()){
      if(!ensure_cached(name)){ std::fprintf(stderr, "cpu: unknown tensor %s\n", name.c_str()); out.clear(); return; }
      it = cache_w.find(name);
    }
    int N = cache_N[name];
    int K = cache_K[name];
    const float* W = it->second.data();
    size_t xn = x.size();
    if(xn < (size_t)K) K = (int)xn;
    if(out.size() != (size_t)N) out.assign(N, 0.f);
    float* yp = out.data();
    const float* xp = x.data();
    // Thread over independent output rows for large GEMMs (N>=2048: gate/up
    // at 3584, lm_head at 248K). Row loop is unchanged scalar FP32 (same
    // order, same rounding -> bit-identical); OpenMP only splits rows.
    // if() keeps small GEMMs serial (fork ~50-170us would dominate).
    // Microbench (20 threads): gate/up 1684->~320us, lm_head 104ms->15.6ms.
    // INTFER_CPU_FMA_DOT=1 selects the AVX2+FMA dot (oracle-gated).
    static const bool fma_dot_i = (std::getenv("INTFER_CPU_FMA_DOT") != nullptr)
        && intfer::dashq::avx2_available();
    if(fma_dot_i){
      gemm_rows_fma(W, xp, yp, (size_t)N, N, (size_t)K, cache_K[name],
                    true, 2048);
    } else {
#pragma omp parallel for schedule(static) if(N >= 2048)
      for(int i=0;i<N;i++){ const float* wr=W+(size_t)i*cache_K[name]; float s=0;
        for(int k=0;k<K;k++) s+=wr[k]*xp[k]; yp[i]=s; }
    }
    if(!keep_cached){
      cache_w.erase(it);
      cache_R.erase(name);
    }
  }
  // Eager models keep every dequantized weight cached (fast); JIT models
  // (27B) free each weight right after its GEMM (RAM-bounded). Set by main().
  bool keep_cached = true;
  // Returns the rotation for `name` without requiring the weight to
  // be cached. Looks up via the per-layer maps (small models) or
  // via ModelIR (large models). Does NOT add to cache.
  const Rotation* rot_for(const std::string& name) override {
    // Small-model path: check cache first.
    if(cache_R.count(name)) return cache_R[name];
    // Large-model path: O(1) rotation-id lookup via the IR index.
    if(ir_){
      auto it = ir_index.find(name);
      if(it != ir_index.end()){
        auto rit = rid_map_global.find(ir_->tensors[it->second].qcfg.rotation_id);
        if(rit != rid_map_global.end()) return rit->second;
        return nullptr;
      }
    }
    // Fall back to per-layer maps with an EXACT per-slot lookup (mirrors
    // ensure_cached's dispatch). The old first-hit scan returned whatever
    // rotation sorted first in the layer map — e.g. mlp down's R for LA
    // tensors served raw from safetensors (eager population never fills
    // la_q when the .intifer lacks LA tensors) — disagreeing with the
    // per-slot rotation ensure_cached stores after the first GEMM. Cold
    // (pre-GEMM) and warm (post-GEMM) resolutions then differed, so row 0
    // of a chunk saw a different rotation than later rows and step_batch
    // (one lookup per layer, pre-GEMM) diverged from the chunk path.
    size_t p = name.find("model.language_model.layers.");
    if(p == std::string::npos) return nullptr;
    int L = std::atoi(name.c_str()+p+28);
    std::string suf = name.substr(p+28);
    while(!suf.empty() && suf[0]>='0' && suf[0]<='9') suf=suf.substr(1);
    auto slot = [&](const std::map<int, std::map<std::string, QLinear>>& mp,
                    const char* key) -> const Rotation* {
      auto itL = mp.find(L);
      if(itL == mp.end()) return nullptr;
      auto itS = itL->second.find(key);
      if(itS == itL->second.end()) return nullptr;
      return itS->second.R;
    };
    if(suf.find("self_attn.q_proj")!=std::string::npos) return slot(attn_q, "q");
    if(suf.find("self_attn.k_proj")!=std::string::npos) return slot(attn_q, "k");
    if(suf.find("self_attn.v_proj")!=std::string::npos) return slot(attn_q, "v");
    if(suf.find("self_attn.o_proj")!=std::string::npos) return slot(attn_q, "o");
    if(suf.find("mlp.gate_proj")!=std::string::npos) return slot(mlp_q, "gate");
    if(suf.find("mlp.up_proj")!=std::string::npos) return slot(mlp_q, "up");
    if(suf.find("mlp.down_proj")!=std::string::npos) return slot(mlp_q, "down");
    if(suf.find("linear_attn.in_proj_qkv")!=std::string::npos) return slot(la_q, "qkv");
    if(suf.find("linear_attn.in_proj_z")!=std::string::npos) return slot(la_q, "z");
    if(suf.find("linear_attn.in_proj_b")!=std::string::npos) return slot(la_q, "b");
    if(suf.find("linear_attn.in_proj_a")!=std::string::npos) return slot(la_q, "a");
    if(suf.find("linear_attn.out_proj")!=std::string::npos) return slot(la_q, "out");
    return nullptr;
  }
  // Unit check for the cold==warm rotation invariant (INTFER_CHECK_ROT=1):
  // rot_for() with an empty cache (cold) must agree with rot_for() after
  // ensure_cached() warms the entry (warm) for every layer x slot the
  // forward touches. Returns true when all slots agree; prints each
  // mismatch (layer, slot, cold ptr, warm ptr) to stderr.
  // NOTE: warm-up runs real GEMM weight resolution (ensure_cached reads
  // safetensors on miss), so this takes ~seconds on first run; it mutates
  // only the weight/rotation caches, which inference would populate anyway.
  bool check_cold_warm(){
    char b[256];
    auto nm = [&](int L, const char* slot) -> std::string {
      snprintf(b, sizeof(b), "model.language_model.layers.%d.%s", L, slot);
      return std::string(b);
    };
    static const char* slots[] = {
      "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight",
      "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
      "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
      "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
      "linear_attn.out_proj.weight",
    };
    bool ok = true;
    for(int L=0; L<cfg.layers; L++){
      for(const char* s : slots){
        std::string name = nm(L, s);
        cache_R.erase(name);
        cache_w.erase(name);
        const Rotation* cold = rot_for(name);
        if(!ensure_cached(name)) continue; // tensor absent: nothing to compare
        auto it = cache_R.find(name);
        const Rotation* warm = (it != cache_R.end()) ? it->second : nullptr;
        if(cold != warm){
          std::fprintf(stderr,
            "[rotcheck] MISMATCH L=%d %s cold=%p warm=%p\n",
            L, s, (const void*)cold, (const void*)warm);
          ok = false;
        }
      }
    }
    std::fprintf(stderr, "[rotcheck] %s (%d layers x %zu slots)\n",
                 ok ? "OK" : "FAIL", cfg.layers,
                 sizeof(slots)/sizeof(slots[0]));
    return ok;
  }
};

#ifdef INTFER_HAS_GPU_BACKEND
#ifdef INTFER_HAS_GPU_BACKEND
// Forward declarations: Request/Layer are defined below (Sections: Layer,
// Request); the SSM device path needs them for keying per-(request,layer)
// resident state.
struct Request;
struct Layer;
struct GpuBackend;
// One LA-branch recurrent step's inputs. query/key/value point at the conv
// rows ([n_k*kd],[n_k*kd],[n_v*vd]); b_row/a_row are gate rows ([n_v] each);
// mirror is the request's host SSM state vector (&ly->ssm_state, complete
// at call sites — the impl only needs pointer identity for req and the
// mirror vector, so Layer/Request stay incomplete here); core_out
// ([n_v*vd]) receives the readout.
struct SsmRow {
  const Request* req = nullptr;
  std::vector<float>* mirror = nullptr;
  const float* query = nullptr;
  const float* key = nullptr;
  const float* value = nullptr;
  const float* b_row = nullptr;
  const float* a_row = nullptr;
  float* core_out = nullptr;
};
// M-row device-side SSM step (M>=1). Rows are partitioned into runs of
// consecutive rows sharing one (request, layer) state: runs of length 1
// (decode rows, distinct states) share one ssm_forward launch; longer
// runs (prefill-chunk rows of one request) use ssm_forward_chain each.
// sync_mirror: also write each touched resident state back to the host
// mirror (needed when snapshots may capture ly.ssm_state afterwards).
// Returns false when unavailable so the caller keeps the host path.
static bool gpu_ssm_rows(GpuBackend& be, const SsmRow* rows, uint32_t M, int L,
                         const float* dt_bias, const float* A_log,
                         bool sync_mirror);
// Re-upload all host SSM mirrors to the device (speculative partial-accept
// rollback replaced ly.ssm_state vectors under the device; the side map
// holds the host mirror pointers so no Request access is needed here).
static void gpu_ssm_resync_all(GpuBackend& be);
#endif

// ----------------------------------------------------------------------------
// GpuBackend — SYCL tensor-parallel via intfer::Engine
// ----------------------------------------------------------------------------
struct GpuBackend : public LinearBackend {
  std::unique_ptr<Engine> eng;
  // Cache of name -> tensor index in model().tensors
  std::map<std::string, int> ti_cache;
  int num_devices_ = 0;
  // Optional safetensors fallback for tensors not in the .intifer
  // (e.g. linear_attn tensors that weren't quantized).
  const SafetensorsReader* sr_fallback = nullptr;
  // Layer-wise VRAM streaming: the Engine holds one layer's shards + lm_head
  // (~1.1GB on 27B) instead of the full ~23GB model. Track the current layer
  // from tensor names and evict the previous layer on advance.
  int cur_layer_ = -1;

  const char* name() const override { return "gpu"; }
  bool is_distributed() const override { return num_devices_ > 1; }

  // Build the name->tensor-index cache. The Engine's model has the same
  // .tensors[] layout as the deserialized ModelIR, so we just iterate.
  void build_index(){
    if(!eng || !ti_cache.empty()) return;
    for(size_t i=0;i<eng->model().tensors.size(); ++i){
      ti_cache[eng->model().tensors[i].name] = (int)i;
    }
    num_devices_ = (int)eng->num_devices();
  }

  std::vector<float> run_linear(const std::string& name,
                                const std::vector<float>& x) override {
    if(!eng) return {};
    build_index();
    auto it = ti_cache.find(name);
    if(it == ti_cache.end()){
      // Fallback: tensor not in .intifer. Read from safetensors and matmul on CPU.
      if(sr_fallback){
        std::vector<float> W;
        if(sr_fallback->read(name, W)){
          // Determine N from name (we know K from x.size())
          size_t p = name.find("model.language_model.layers.");
          if(p != std::string::npos){
            int L = std::atoi(name.c_str()+p+28);
            std::string suf = name.substr(p+28);
            while(!suf.empty() && suf[0]>='0' && suf[0]<='9') suf=suf.substr(1);
            int N = 0, K = (int)x.size();
            if(suf.find("self_attn.q_proj")!=std::string::npos){        N=cfg.n_heads*cfg.head_dim; }
            else if(suf.find("self_attn.k_proj")!=std::string::npos){   N=cfg.n_kv*cfg.head_dim; }
            else if(suf.find("self_attn.v_proj")!=std::string::npos){   N=cfg.n_kv*cfg.head_dim; }
            else if(suf.find("self_attn.o_proj")!=std::string::npos){   N=cfg.H; }
            else if(suf.find("mlp.gate_proj")!=std::string::npos){      N=cfg.inter; }
            else if(suf.find("mlp.up_proj")!=std::string::npos){        N=cfg.inter; }
            else if(suf.find("mlp.down_proj")!=std::string::npos){      N=cfg.H; }
            else if(suf.find("linear_attn.in_proj_qkv")!=std::string::npos){ N=cfg.conv_dim; }
            else if(suf.find("linear_attn.in_proj_z")!=std::string::npos){   N=cfg.n_v_heads*cfg.v_head_dim; }
            else if(suf.find("linear_attn.in_proj_b")!=std::string::npos){   N=cfg.n_v_heads; }
            else if(suf.find("linear_attn.in_proj_a")!=std::string::npos){   N=cfg.n_v_heads; }
            else if(suf.find("linear_attn.out_proj")!=std::string::npos){   N=cfg.H; }
            if(N>0 && (int)W.size() % N == 0){
              int Kw = (int)W.size() / N;
              int actual_K = std::min(K, Kw);
              std::vector<float> y(N, 0.f);
              // Same OpenMP row-split (bit-identical scalar rows).
#pragma omp parallel for schedule(static) if(N >= 2048)
              for(int i=0;i<N;i++){ const float* wr=&W[i*actual_K]; float s=0;
                for(int k=0;k<actual_K;k++) s+=wr[k]*x[k]; y[i]=s; }
              return y;
            }
          }
        }
      }
      std::fprintf(stderr, "gpu: unknown tensor %s\n", name.c_str());
      return {};
    }
    // Layer-wise streaming: evict the previous layer's device shards when the
    // forward advances (unlayered tensors like lm_head are kept resident by
    // the Engine). Rebuild on next use is transparent via ensure_shard_built.
    int L = Engine::layer_of_name(name);
    if(L >= 0 && L != cur_layer_){ eng->begin_layer(L); cur_layer_ = L; }
    return eng->run_linear_tp_ret(it->second, x);
  }
  void run_linear_into(const std::string& name,
                       const std::vector<float>& x,
                       std::vector<float>& out) override {
    if(!eng){ out.clear(); return; }
    build_index();
    auto it = ti_cache.find(name);
    if(it == ti_cache.end()){
      // Fallback path is rare (tensors missing from .intifer); keep the
      // allocating impl there and move into out (one alloc, cold path).
      out = run_linear(name, x);
      return;
    }
    int L = Engine::layer_of_name(name);
    if(L >= 0 && L != cur_layer_){ eng->begin_layer(L); cur_layer_ = L; }
    // Engine resizes out only when the size differs: steady-state zero
    // mallocs on the host output buffer.
    eng->run_linear_tp(it->second, x, out);
  }
  void run_linear_batch(const std::string& name,
                        const std::vector<float>& X,
                        std::vector<float>& Y, uint32_t M) override {
    if(!eng){ Y.clear(); return; }
    build_index();
    auto it = ti_cache.find(name);
    if(it == ti_cache.end()){
      // Cold fallback path: loop the scalar fallback (rare).
      LinearBackend::run_linear_batch(name, X, Y, M);
      return;
    }
    int L = Engine::layer_of_name(name);
    if(L >= 0 && L != cur_layer_){ eng->begin_layer(L); cur_layer_ = L; }
    eng->run_linear_tp(it->second, X, Y, M);
  }
  // Wave of independent GEMMs (see LinearBackend::run_linear_wave): one
  // engine call converts + submits every item back-to-back and waits once.
  // Env INTFER_NO_WAVE=1 restores sequential run_linear_into (bisection).
  // Layer tracking: all wave items must belong to the same layer (callers
  // group per-layer independent GEMMs); the first valid item's layer wins.
  void run_linear_wave(const LinearWaveItem* items, size_t n) override {
    if(!eng || !items || n == 0) return;
    static const bool no_wave =
        (std::getenv("INTFER_NO_WAVE") != nullptr);
    build_index();
    if (no_wave) {
      LinearBackend::run_linear_wave(items, n);
      return;
    }
    Engine::LinearWaveItem eitems[16];
    size_t ne = 0;
    int wave_layer = -2;
    for (size_t i = 0; i < n && ne < 16; ++i) {
      auto it = ti_cache.find(items[i].name);
      if (it == ti_cache.end()) {
        // Cold fallback for this item only; other items still wave.
        LinearBackend::run_linear_wave(items + i, 1);
        continue;
      }
      int L = Engine::layer_of_name(items[i].name);
      if (wave_layer == -2) wave_layer = L;
      eitems[ne].tensor_index = it->second;
      eitems[ne].x = items[i].x;
      eitems[ne].out = items[i].out;
      ++ne;
    }
    if (ne == 0) return;
    if (wave_layer >= 0 && wave_layer != cur_layer_) {
      eng->begin_layer(wave_layer);
      cur_layer_ = wave_layer;
    }
    eng->run_linear_wave(eitems, ne, 1);
  }
  // M-row wave (see LinearBackend::run_linear_wave_batch): same grouping
  // as the scalar wave, but each item carries M rows. INTFER_NO_WAVE=1
  // restores sequential run_linear_batch.
  void run_linear_wave_batch(const LinearWaveBatchItem* items, size_t n,
                             uint32_t M) override {
    if(!eng || !items || n == 0 || M == 0) return;
    static const bool no_wave =
        (std::getenv("INTFER_NO_WAVE") != nullptr);
    build_index();
    if (no_wave) {
      LinearBackend::run_linear_wave_batch(items, n, M);
      return;
    }
    Engine::LinearWaveItem eitems[16];
    size_t ne = 0;
    int wave_layer = -2;
    for (size_t i = 0; i < n && ne < 16; ++i) {
      auto it = ti_cache.find(items[i].name);
      if (it == ti_cache.end()) {
        LinearBackend::run_linear_wave_batch(items + i, 1, M);
        continue;
      }
      int L = Engine::layer_of_name(items[i].name);
      if (wave_layer == -2) wave_layer = L;
      eitems[ne].tensor_index = it->second;
      eitems[ne].x = items[i].X;
      eitems[ne].out = items[i].Y;
      ++ne;
    }
    if (ne == 0) return;
    if (wave_layer >= 0 && wave_layer != cur_layer_) {
      eng->begin_layer(wave_layer);
      cur_layer_ = wave_layer;
    }
    eng->run_linear_wave(eitems, ne, M);
  }
  // Fused MLP down-projection (TP=1 device chain: gate/up wave-submit ->
  // device silu-mul + convert -> down GEMM, one wait + one D2H).
  // INTFER_NO_FUSE_SILUMUL=1 restores wave + host silu-mul (bisection).
  // Returns true when the fused path ran (md filled, mg/mu NOT filled);
  // false keeps the caller's host path. Trace runs (NaN/H debug) always
  // take the host path so ckpts see real mg/mu.
  bool run_mlp_fused(const std::string& gate, const std::string& up,
                     const std::string& down, const std::vector<float>& rin,
                     std::vector<float>& mg, std::vector<float>& mu,
                     std::vector<float>& md, uint32_t M) override {
    static const bool no_fuse =
        (std::getenv("INTFER_NO_FUSE_SILUMUL") != nullptr) ||
        (std::getenv("INTFER_TRACE_NAN") != nullptr) ||
        (std::getenv("INTFER_TRACE_H") != nullptr);
    if (!eng || no_fuse) return false;
    build_index();
    auto itg = ti_cache.find(gate);
    auto itu = ti_cache.find(up);
    auto itd = ti_cache.find(down);
    if (itg == ti_cache.end() || itu == ti_cache.end() ||
        itd == ti_cache.end())
      return false;
    int L = Engine::layer_of_name(gate);
    if (L >= 0 && L != cur_layer_) { eng->begin_layer(L); cur_layer_ = L; }
    if (!eng->run_mlp_fused(itg->second, itu->second, itd->second,
                            rin, md, M))
      return false;
    // Fused path ran: mg/mu host vectors intentionally NOT filled (their
    // only consumers are the silu-mul folded into the device chain and
    // trace-gated ckpts, which route to the host path above).
    (void)mg; (void)mu;
    return true;
  }

  // Cached lm_head INT8 upload + per-row-symmetric INT8 GEMM on device.
  // The full FP16 head (2.54GB on 27B: 1.27GB/dev TP=2) can NEVER fit next
  // to the preloaded engine on dev0 (11360+1270=12647 > 12216 hard), so the
  // head uploads as per-row symmetric INT8 (~0.64GB/dev) + FP16 row scales:
  //   q[n,k] = round(W[n,k]/s[n]), s[n] = max|W[n,:]|/127
  // via gemm::quant_gemm_f16_int8w (same 64x64x64 tiling as quant_gemm_f16;
  // dequant is one FMA in the weight-load path; probe-gated maxerr ~1e-3
  // vs signal ~2, argmax stable over 5 seeds — see build_tmp probe).
  // TP split is ASYMMETRIC by measured free VRAM (dev0 keeps a ~393MiB
  // display reservation): after preload_all(), query eng->device_free_bytes
  // per rank, subtract a 128MiB margin (per-token x_dev/out + L0
  // fragmentation), split rows proportional to the remainder. Falls back to
  // an even split when the sysman query is unavailable (returns 0).
  // INTFER_LMHEAD_SPLIT0=<rows> overrides the dev0 row count (bisection);
  // INTFER_LMHEAD_FP16=1 restores the old FP16 tiled path (bisection only —
  // it cannot fit 27B TP=2; kept so the 0.8B parity gate can A/B).
  // Bit-considerations: GEMM rows are independent (row-blocked over N with
  // bounds guards), so any row partition concatenates to the same logits
  // up to INT8 rounding (argmax-gated, not bit-identical vs FP32).
  std::vector<float> run_lm_head_batch(const float* FN, // final norms [M,K]
                                       const float* W,  // lm_head weight [N,K]
                                       uint32_t M, uint32_t N, uint32_t K){
    if(!eng) return {};
    if(M == 0) return {};
    if(M == 1) return run_lm_head_f16(FN, W, N, K);
    size_t ndev = eng->num_devices();
    static const bool force_fp16_b = (std::getenv("INTFER_LMHEAD_FP16") != nullptr);
    static const bool no_dpas_b = (std::getenv("INTFER_LMHEAD_DPAS") &&
                                   std::getenv("INTFER_LMHEAD_DPAS")[0] == '0');
    // Ensure the upload cache exists via the scalar path (FN row 0 is a
    // valid M=1 activation). Adds a small one-time upload on first use.
    if(lm_head_q_dev_.empty() && lm_head_w_dev_.empty()){
      std::vector<float> probe = run_lm_head_f16(FN, W, N, K);
      if(probe.empty()) return {};
    }
    if(lm_head_N_ != N || lm_head_K_ != K) return {};
    const uint32_t nGA = (K + 63) / 64;
    auto seq_fallback = [&]() -> std::vector<float> {
      std::vector<float> seq;
      seq.reserve((size_t)M * N);
      for(uint32_t m = 0; m < M; ++m){
        std::vector<float> lm = run_lm_head_f16(FN + (size_t)m * K, W, N, K);
        if((int)lm.size() != (int)N) return {};
        seq.insert(seq.end(), lm.begin(), lm.end());
      }
      return seq;
    };
    // M-row host staging: FP16 acts (scalar paths) + INT8 acts/scales (DPAS).
    std::vector<sycl::half> x_fp16((size_t)M * K);
    for(uint32_t m = 0; m < M; ++m)
      for(uint32_t i = 0; i < K; ++i)
        x_fp16[(size_t)m * K + i] = sycl::half(FN[(size_t)m * K + i]);
    std::vector<std::int8_t> aq((size_t)M * K);
    std::vector<float> as((size_t)M * nGA);
    std::vector<float> out((size_t)M * N, 0.f);
    std::vector<sycl::event> evts;
    evts.reserve(ndev * 8);
    // Persistent per-device M-row buffers (grow-only, like the scalar path's
    // M=1 x_dev/aq/as + per-tile c_dev): act inputs, act-quant workspaces,
    // and one tile-tight M*Nlt output staging per tile.
    if(lm_head_xM_dev_.size() != ndev) lm_head_xM_dev_.assign(ndev, nullptr);
    if(lm_head_xM_cap_.size() != ndev) lm_head_xM_cap_.assign(ndev, 0);
    if(lm_head_aqM_dev_.size() != ndev) lm_head_aqM_dev_.assign(ndev, nullptr);
    if(lm_head_asM_dev_.size() != ndev) lm_head_asM_dev_.assign(ndev, nullptr);
    if(lm_head_aqM_cap_.size() != ndev) lm_head_aqM_cap_.assign(ndev, 0);
    if(lm_head_cM_dev_.size() != ndev) lm_head_cM_dev_.resize(ndev);
    if(lm_head_cM_cap_.size() != ndev) lm_head_cM_cap_.resize(ndev);
    if(lm_head_tM_host_.size() != ndev) lm_head_tM_host_.resize(ndev);
    struct TileD2H { size_t d; size_t t; sycl::event evt; };
    std::vector<TileD2H> d2hs;
    for(size_t d = 0; d < ndev; ++d){
      uint32_t start = (d < lm_head_ranges_.size()) ? lm_head_ranges_[d].first : N;
      uint32_t end = (d < lm_head_ranges_.size()) ? lm_head_ranges_[d].second : N;
      uint32_t Nl = (end > start) ? end - start : 0;
      if(Nl == 0 || d >= lm_head_tiles_.size() || lm_head_tiles_[d].empty()) continue;
      sycl::queue& q = eng->device_queue((int)d);
      bool use_dpas = !force_fp16_b && !no_dpas_b &&
                      d < lm_head_aq_dev_.size() && lm_head_aq_dev_[d] &&
                      d < lm_head_as_dev_.size() && lm_head_as_dev_[d];
      if(use_dpas){
        for(size_t t = 0; t < lm_head_tiles_[d].size(); ++t){
          if(t >= lm_head_b_dev_[d].size() || !lm_head_b_dev_[d][t]){
            use_dpas = false;
            break;
          }
        }
      }
      if(use_dpas){
        if(!lm_head_aqM_dev_[d] || !lm_head_asM_dev_[d] || lm_head_aqM_cap_[d] < M){
          try {
            if(lm_head_aqM_dev_[d]){ sycl::free(lm_head_aqM_dev_[d], q); lm_head_aqM_dev_[d] = nullptr; }
            if(lm_head_asM_dev_[d]){ sycl::free(lm_head_asM_dev_[d], q); lm_head_asM_dev_[d] = nullptr; }
            lm_head_aqM_dev_[d] = static_cast<std::int8_t*>(sycl::malloc_device(
                (size_t)M * K, q.get_device(), q.get_context()));
            lm_head_asM_dev_[d] = static_cast<float*>(sycl::malloc_device(
                (size_t)M * nGA * sizeof(float), q.get_device(), q.get_context()));
            lm_head_aqM_cap_[d] = M;
          } catch (...) {
            lm_head_aqM_dev_[d] = nullptr; lm_head_asM_dev_[d] = nullptr;
            lm_head_aqM_cap_[d] = 0;
          }
        }
        if(!lm_head_aqM_dev_[d] || !lm_head_asM_dev_[d]) use_dpas = false;
      }
      if(use_dpas){
        for(uint32_t m = 0; m < M; ++m)
          dashq::act_quant_ga64_batch(FN + (size_t)m * K,
                                      aq.data() + (size_t)m * K,
                                      as.data() + (size_t)m * nGA, K,
                                      dashq::ActQuantRound::RoundNear);
        q.memcpy(lm_head_aqM_dev_[d], aq.data(), (size_t)M * K);
        q.memcpy(lm_head_asM_dev_[d], as.data(), (size_t)M * nGA * sizeof(float));
      } else {
        if(!lm_head_xM_dev_[d] || lm_head_xM_cap_[d] < M){
          try {
            if(lm_head_xM_dev_[d]){ sycl::free(lm_head_xM_dev_[d], q); lm_head_xM_dev_[d] = nullptr; }
            lm_head_xM_dev_[d] = static_cast<sycl::half*>(sycl::malloc_device(
                (size_t)M * K * sizeof(sycl::half), q.get_device(), q.get_context()));
            lm_head_xM_cap_[d] = M;
          } catch (...) {
            lm_head_xM_dev_[d] = nullptr; lm_head_xM_cap_[d] = 0;
          }
        }
        if(!lm_head_xM_dev_[d]) return seq_fallback();
        q.memcpy(lm_head_xM_dev_[d], x_fp16.data(), (size_t)M * K * sizeof(sycl::half));
      }
      // Per-tile M-row output staging (grow-only): kernels write c[mg*Nlt
      // + ng] tile-tight (Nlt stride), so each tile needs its own M*Nlt
      // device buffer + host mirror; rows are scattered into out below.
      if(lm_head_cM_dev_[d].size() != lm_head_tiles_[d].size()){
        lm_head_cM_dev_[d].assign(lm_head_tiles_[d].size(), nullptr);
        lm_head_cM_cap_[d].assign(lm_head_tiles_[d].size(), 0);
        lm_head_tM_host_[d].assign(lm_head_tiles_[d].size(), {});
      }
      const sycl::half* x_row = use_dpas ? nullptr : lm_head_xM_dev_[d];
      for(size_t t = 0; t < lm_head_tiles_[d].size(); ++t){
        uint32_t Nlt = lm_head_tiles_[d][t].second - lm_head_tiles_[d][t].first;
        if(Nlt == 0) continue;
        if(!lm_head_cM_dev_[d][t] || lm_head_cM_cap_[d][t] < (size_t)M * Nlt){
          try {
            if(lm_head_cM_dev_[d][t]){ sycl::free(lm_head_cM_dev_[d][t], q); lm_head_cM_dev_[d][t] = nullptr; }
            lm_head_cM_dev_[d][t] = static_cast<sycl::half*>(sycl::malloc_device(
                (size_t)M * Nlt * sizeof(sycl::half), q.get_device(), q.get_context()));
            lm_head_cM_cap_[d][t] = (size_t)M * Nlt;
          } catch (...) {
            lm_head_cM_dev_[d][t] = nullptr; lm_head_cM_cap_[d][t] = 0;
          }
        }
        if(!lm_head_cM_dev_[d][t]) return seq_fallback();
        if(lm_head_tM_host_[d][t].size() != (size_t)M * Nlt)
          lm_head_tM_host_[d][t].resize((size_t)M * Nlt);
        sycl::event evt;
        try {
          if(force_fp16_b){
            evt = gemm::quant_gemm_f16(q, x_row, lm_head_w_dev_[d][t],
                                       lm_head_cM_dev_[d][t], M, Nlt, K);
          } else if(use_dpas){
            gemm::Dpas48Params p;
            p.code_dpas = lm_head_b_dev_[d][t];
            p.scale_usm = reinterpret_cast<const std::uint16_t*>(
                lm_head_s_dev_[d][t]);
            p.act_int8 = lm_head_aqM_dev_[d];
            p.act_scale = lm_head_asM_dev_[d];
            p.alpha = 1.0f;
            p.out_usm = reinterpret_cast<std::uint16_t*>(lm_head_cM_dev_[d][t]);
            p.M = M;
            p.N = Nlt;
            p.K = K;
            p.group_size = K;
            p.act_group = 64;
            p.bits = 8;
            p.act_stride = K;
            p.as_stride = nGA;
            evt = gemm::quant_gemm_dpas_s8w(q, p);
          } else {
            evt = gemm::quant_gemm_f16_int8w(q, x_row, lm_head_q_dev_[d][t],
                                             lm_head_s_dev_[d][t],
                                             lm_head_cM_dev_[d][t], M, Nlt, K);
          }
        } catch (...) {
          return seq_fallback();
        }
        sycl::event cp = q.memcpy(lm_head_tM_host_[d][t].data(), lm_head_cM_dev_[d][t],
                 (size_t)M * Nlt * sizeof(sycl::half), evt);
        evts.push_back(evt);
        d2hs.push_back({d, t, cp});
      }
    }
    for(auto& e : evts) e.wait();
    for(auto& r : d2hs) r.evt.wait();
    // Scatter tile-tight rows into full-vocab order.
    for(size_t d = 0; d < ndev; ++d){
      if(d >= lm_head_ranges_.size() || d >= lm_head_tiles_.size()) continue;
      uint32_t start = lm_head_ranges_[d].first;
      for(size_t t = 0; t < lm_head_tiles_[d].size(); ++t){
        uint32_t ts = lm_head_tiles_[d][t].first;
        uint32_t Nlt = lm_head_tiles_[d][t].second - ts;
        if(Nlt == 0 || t >= lm_head_tM_host_[d].size()) continue;
        const std::vector<sycl::half>& tile = lm_head_tM_host_[d][t];
        if(tile.size() != (size_t)M * Nlt) continue;
        for(uint32_t m = 0; m < M; ++m)
          for(uint32_t i = 0; i < Nlt; ++i)
            out[(size_t)m * N + (ts + i)] = static_cast<float>(tile[(size_t)m * Nlt + i]);
      }
    }
    return out;
  }

  std::vector<float> run_lm_head_f16(const float* fn, // final norm hidden [K]
                                     const float* W,   // lm_head weight [N,K]
                                     uint32_t N, uint32_t K){
    if(!eng) return {};
    size_t ndev = eng->num_devices();
    static const bool force_fp16 = (std::getenv("INTFER_LMHEAD_FP16") != nullptr);
    // Function scope: both the upload path and the per-token GEMM path
    // (x_dev alloc / submit throws under VRAM pressure) reset through
    // fail_reset so a shortfall degrades to {} instead of SIGABRT.
    std::vector<sycl::half*> x_devs;
    auto fail_reset = [&](){
      // Alloc failure (VRAM pressure) must not leave a half-initialized
      // cache: the next call would GEMM from nullptr (DEVICE_LOST). Free
      // partials and reset so the next call retries cleanly (caller gets
      // {} -> zero logits for this token). x_devs entries are aliases of
      // the persistent lm_head_x_dev_ buffers — clear the list but do NOT
      // free them here (freed with the rest of the cache below).
      x_devs.clear();
      for(size_t d=0; d<ndev; ++d){
        for(size_t t=0; t<lm_head_q_dev_[d].size(); ++t){
          if(lm_head_q_dev_[d][t]){
            sycl::free(lm_head_q_dev_[d][t], eng->device_queue((int)d));
            lm_head_q_dev_[d][t] = nullptr;
          }
          if(t<lm_head_s_dev_[d].size() && lm_head_s_dev_[d][t]){
            sycl::free(lm_head_s_dev_[d][t], eng->device_queue((int)d));
            lm_head_s_dev_[d][t] = nullptr;
          }
          if(t<lm_head_b_dev_[d].size() && lm_head_b_dev_[d][t]){
            sycl::free(lm_head_b_dev_[d][t], eng->device_queue((int)d));
            lm_head_b_dev_[d][t] = nullptr;
          }
          if(t<lm_head_c_dev_[d].size() && lm_head_c_dev_[d][t]){
            sycl::free(lm_head_c_dev_[d][t], eng->device_queue((int)d));
            lm_head_c_dev_[d][t] = nullptr;
          }
        }
        if(d < lm_head_aq_dev_.size() && lm_head_aq_dev_[d]){
          sycl::free(lm_head_aq_dev_[d], eng->device_queue((int)d));
          lm_head_aq_dev_[d] = nullptr;
        }
        if(d < lm_head_as_dev_.size() && lm_head_as_dev_[d]){
          sycl::free(lm_head_as_dev_[d], eng->device_queue((int)d));
          lm_head_as_dev_[d] = nullptr;
        }
        for(size_t t=0; t<lm_head_w_dev_[d].size(); ++t){
          if(lm_head_w_dev_[d][t]){
            sycl::free(lm_head_w_dev_[d][t], eng->device_queue((int)d));
            lm_head_w_dev_[d][t] = nullptr;
          }
        }
        if(d < lm_head_x_dev_.size() && lm_head_x_dev_[d]){
          sycl::free(lm_head_x_dev_[d], eng->device_queue((int)d));
          lm_head_x_dev_[d] = nullptr;
        }
        if(d < lm_head_xM_dev_.size() && lm_head_xM_dev_[d]){
          sycl::free(lm_head_xM_dev_[d], eng->device_queue((int)d));
          lm_head_xM_dev_[d] = nullptr;
        }
        if(d < lm_head_aqM_dev_.size() && lm_head_aqM_dev_[d]){
          sycl::free(lm_head_aqM_dev_[d], eng->device_queue((int)d));
          lm_head_aqM_dev_[d] = nullptr;
        }
        if(d < lm_head_asM_dev_.size() && lm_head_asM_dev_[d]){
          sycl::free(lm_head_asM_dev_[d], eng->device_queue((int)d));
          lm_head_asM_dev_[d] = nullptr;
        }
        for(auto cMv : lm_head_cM_dev_[d]){
          if(cMv) sycl::free(cMv, eng->device_queue((int)d));
        }
      }
      lm_head_xM_dev_.clear(); lm_head_xM_cap_.clear();
      lm_head_aqM_dev_.clear(); lm_head_asM_dev_.clear(); lm_head_aqM_cap_.clear();
      lm_head_cM_dev_.clear(); lm_head_cM_cap_.clear(); lm_head_tM_host_.clear();
      lm_head_q_dev_.clear(); lm_head_s_dev_.clear();
      lm_head_b_dev_.clear();
      lm_head_w_dev_.clear(); lm_head_c_dev_.clear();
      lm_head_aq_dev_.clear(); lm_head_as_dev_.clear();
      lm_head_x_dev_.clear();
      lm_head_N_ = 0; lm_head_K_ = 0;
    };
    if(lm_head_q_dev_.empty() && lm_head_w_dev_.empty()){
      lm_head_N_ = N; lm_head_K_ = K;
      lm_head_ranges_.resize(ndev);
      lm_head_tiles_.resize(ndev);
      if(force_fp16){
        lm_head_w_dev_.resize(ndev);
        lm_head_c_dev_.resize(ndev);
      } else {
        lm_head_q_dev_.resize(ndev);
        lm_head_s_dev_.resize(ndev);
        lm_head_b_dev_.resize(ndev);
        lm_head_c_dev_.resize(ndev);
        lm_head_aq_dev_.resize(ndev, nullptr);
        lm_head_as_dev_.resize(ndev, nullptr);
      }
      if(ndev == 1){
        lm_head_ranges_[0] = {0, N};
      } else if(force_fp16){
        for(size_t d=0; d<ndev; ++d)
          lm_head_ranges_[d] = {(uint32_t)(N * d / ndev), (uint32_t)(N * (d + 1) / ndev)};
      } else {
        // Asymmetric split by measured free VRAM (post-preload). Per-row
        // resident bytes: K INT8 codes + 1 FP16 scale.
        const uint64_t row_bytes = (uint64_t)K + sizeof(sycl::half);
        const uint64_t margin = 128u * 1024u * 1024u;
        uint64_t cap0 = 0, cap1 = 0;
        {
          uint64_t f0 = eng->device_free_bytes(0);
          uint64_t f1 = (ndev > 1) ? eng->device_free_bytes(1) : 0;
          std::fprintf(stderr, "[intfer] lm_head: free VRAM dev0=%lluMiB dev1=%lluMiB\n",
                       (unsigned long long)(f0 / (1024*1024)),
                       (unsigned long long)(f1 / (1024*1024)));
          std::fflush(stderr);
          cap0 = (f0 > margin) ? (f0 - margin) / row_bytes : 0;
          cap1 = (f1 > margin) ? (f1 - margin) / row_bytes : 0;
        }
        uint32_t split0 = 0;
        if(const char* so = std::getenv("INTFER_LMHEAD_SPLIT0")){
          long v = std::atol(so);
          if(v > 0 && (uint64_t)v < N) split0 = (uint32_t)v;
        }
        if(!split0){
          if(cap0 + cap1 >= N && (cap0 > 0 || cap1 > 0)){
            // Rows proportional to capacity; dev0 capped at its capacity.
            split0 = (uint32_t)std::min<uint64_t>(cap0, N);
            if(cap0 + cap1 > 0)
              split0 = (uint32_t)std::min<uint64_t>(cap0, (N * cap0) / (cap0 + cap1));
          } else {
            // Sysman unavailable (both 0) or short: even split, same as before.
            split0 = N / (uint32_t)ndev;
            std::fprintf(stderr, "[intfer] lm_head: free-VRAM query unavailable "
                         "(cap0=%llu cap1=%llu rows, N=%u) — even split\n",
                         (unsigned long long)cap0, (unsigned long long)cap1, N);
          }
        }
        if(split0 == 0 || split0 >= N) split0 = N / (uint32_t)ndev;
        // DPAS s8-weight path needs N_local % 16 == 0 per device (repack
        // gate): round dev0's split down to a multiple of 16. N (vocab)
        // is itself %16==0, so dev1's shard stays %16==0 too. Rows are
        // independent — any partition concatenates to the same logits.
        split0 = (split0 / 16) * 16;
        if(split0 < 16) split0 = 16;
        if(split0 >= N) split0 = (N / 16 - 1) * 16;
        lm_head_ranges_[0] = {0, split0};
        if(ndev > 1) lm_head_ranges_[1] = {split0, N};
        for(size_t d=2; d<ndev; ++d) lm_head_ranges_[d] = {N, N};
        std::fprintf(stderr, "[intfer] lm_head INT8: dev0 rows [0,%u) dev1 rows [%u,%u)\n",
                     split0, split0, N);
        std::fflush(stderr);
      }
      // Rows per tile: cap tile weight bytes at 128MB so the upload fits
      // alongside the preloaded engine. >=1 row. (INT8 tiles are 2x smaller
      // than the FP16 ones this formula was sized for — conservative.)
      // Rounded down to %16==0 so DPAS tiles repack (same %16 logic as the
      // TP split above; tile remainders of a %16 range stay %16).
      uint32_t tile_rows = (uint32_t)std::max<size_t>(
          1, (128u * 1024u * 1024u) / ((size_t)K * sizeof(sycl::half)));
      tile_rows = (tile_rows / 16) * 16;
      if(tile_rows < 16) tile_rows = 16;
      for(size_t d=0; d<ndev; ++d){
        uint32_t start = lm_head_ranges_[d].first, end = lm_head_ranges_[d].second;
        if(end <= start) continue;
        sycl::queue& q = eng->device_queue((int)d);
        // Tile this device's row range (peak transient = 1 tile).
        for(uint32_t ts = start; ts < end; ts += tile_rows){
          uint32_t te = std::min(ts + tile_rows, end);
          uint32_t Nl = te - ts;
          lm_head_tiles_[d].push_back({ts, te});
          size_t t_idx = lm_head_tiles_[d].size() - 1;
          // NOTE: sycl::malloc_device THROWS (bad_alloc / UR_RESULT_ERROR_-
          // OUT_OF_RESOURCES) on OOM instead of returning nullptr — catch
          // and route through fail_reset so a shortfall degrades to zero
          // logits for this token, not SIGABRT.
          sycl::half* c_tile = nullptr;
          try {
            c_tile = static_cast<sycl::half*>(
                sycl::malloc_device((size_t)Nl * sizeof(sycl::half), q.get_device(), q.get_context()));
          } catch (...) { c_tile = nullptr; }
          lm_head_c_dev_[d].push_back(c_tile);
          if(!lm_head_c_dev_[d][t_idx]){
            std::fprintf(stderr, "[intfer] FATAL: lm_head tile out alloc failed "
                         "on device %zu (rows %u-%u)\n", d, ts, te);
            std::fflush(stderr);
            fail_reset();
            return {};
          }
          if(force_fp16){
            size_t w_bytes = (size_t)Nl * K * sizeof(sycl::half);
            sycl::half* w_tile = nullptr;
            try {
              w_tile = static_cast<sycl::half*>(
                  sycl::malloc_device(w_bytes, q.get_device(), q.get_context()));
            } catch (...) { w_tile = nullptr; }
            lm_head_w_dev_[d].push_back(w_tile);
            if(!lm_head_w_dev_[d][t_idx]){
              std::fprintf(stderr, "[intfer] FATAL: lm_head tile alloc failed "
                           "on device %zu (rows %u-%u, VRAM budget exceeded?)\n",
                           d, ts, te);
              std::fflush(stderr);
              fail_reset();
              return {};
            }
            // Upload once. NOTE: the per-queue wait below is load-bearing, not
            // profiling — the H2D memcpy is async; without the wait the host
            // reuses/frees the staging buffer while the copy is still in flight,
            // which surfaces as DEVICE_LOST under full-model device pressure.
            std::vector<sycl::half> w_fp16((size_t)Nl * K);
            for(uint32_t r=0;r<Nl;r++) for(uint32_t k=0;k<K;k++)
              w_fp16[(size_t)r*K+k] = sycl::half(W[(size_t)(ts+r)*K+k]);
            q.memcpy(lm_head_w_dev_[d][t_idx], w_fp16.data(), w_bytes);
            q.wait();
          } else {
            // INT8 upload: quantize rows on the host (per-row symmetric),
            // upload codes + FP16 scales. Host staging is Nl*K int8 (~half
            // the FP16 tile) + Nl scales.
            std::vector<std::int8_t> q_i8((size_t)Nl * K);
            std::vector<sycl::half> q_sc(Nl);
            for(uint32_t r=0;r<Nl;r++){
              const float* Wr = W + (size_t)(ts+r)*K;
              float mx = 0;
              for(uint32_t k=0;k<K;k++){
                float a = std::fabs(Wr[k]);
                if(a > mx) mx = a;
              }
              float s = (mx > 0) ? mx / 127.0f : 1.0f;
              q_sc[r] = sycl::half(s);
              float inv = 1.0f / s;
              std::int8_t* qr = q_i8.data() + (size_t)r*K;
              for(uint32_t k=0;k<K;k++)
                qr[k] = (std::int8_t)std::lrint(Wr[k] * inv);
            }
            std::int8_t* q_tile = nullptr;
            try {
              q_tile = static_cast<std::int8_t*>(sycl::malloc_device(
                  (size_t)Nl * K, q.get_device(), q.get_context()));
            } catch (...) { q_tile = nullptr; }
            lm_head_q_dev_[d].push_back(q_tile);
            sycl::half* s_tile = nullptr;
            try {
              s_tile = static_cast<sycl::half*>(sycl::malloc_device(
                  (size_t)Nl * sizeof(sycl::half), q.get_device(), q.get_context()));
            } catch (...) { s_tile = nullptr; }
            lm_head_s_dev_[d].push_back(s_tile);
            if(!lm_head_q_dev_[d][t_idx] || !lm_head_s_dev_[d][t_idx]){
              std::fprintf(stderr, "[intfer] FATAL: lm_head INT8 tile alloc failed "
                           "on device %zu (rows %u-%u, VRAM budget exceeded?)\n",
                           d, ts, te);
              std::fflush(stderr);
              fail_reset();
              return {};
            }
            q.memcpy(lm_head_q_dev_[d][t_idx], q_i8.data(), (size_t)Nl * K);
            q.wait();
            q.memcpy(lm_head_s_dev_[d][t_idx], q_sc.data(),
                     (size_t)Nl * sizeof(sycl::half));
            q.wait();
            // DPAS s8-weight path: repack row-major codes to vertical-B
            // tiles once (same bytes: Nl*K s8 = nT*nOp*512). Shape-gated:
            // Nl%16==0 && K%32==0, else the per-token path falls back to
            // the scalar INT8 kernel for that tile.
            std::vector<std::uint8_t> dpas_tiles;
            std::int8_t* b_tile = nullptr;
            if(gemm::repack_dpas412(
                   reinterpret_cast<const std::uint8_t*>(q_i8.data()),
                   Nl, K, 8, dpas_tiles)){
              try {
                b_tile = static_cast<std::int8_t*>(sycl::malloc_device(
                    dpas_tiles.size(), q.get_device(), q.get_context()));
              } catch (...) { b_tile = nullptr; }
              if(b_tile){
                q.memcpy(b_tile, dpas_tiles.data(), dpas_tiles.size());
                q.wait();
              }
            }
            // b_tile==nullptr marks "DPAS refused": scalar fallback.
            lm_head_b_dev_[d].push_back(b_tile);
          }
        }
        // Per-device INT8 act-quant workspaces for the DPAS s8-weight path:
        // M=1 row of INT8 acts (K bytes) + ceil(K/64) FP32 GA-slice scales.
        // Lazily (re)allocated here — K is fixed after the first upload.
        if(!force_fp16 && !lm_head_tiles_[d].empty()){
          uint32_t nGA = (K + 63) / 64;
          if(!lm_head_aq_dev_[d]){
            try {
              lm_head_aq_dev_[d] = static_cast<std::int8_t*>(sycl::malloc_device(
                  K, q.get_device(), q.get_context()));
              lm_head_as_dev_[d] = static_cast<float*>(sycl::malloc_device(
                  (size_t)nGA * sizeof(float), q.get_device(), q.get_context()));
            } catch (...) {
              if(lm_head_aq_dev_[d]){ sycl::free(lm_head_aq_dev_[d], q); lm_head_aq_dev_[d] = nullptr; }
              if(lm_head_as_dev_[d]){ sycl::free(lm_head_as_dev_[d], q); lm_head_as_dev_[d] = nullptr; }
            }
            // Workspaces are best-effort: nullptr just disables the DPAS
            // path per token (scalar fallback), like b_tile==nullptr above.
          }
        }
      }
      for(size_t d=0; d<ndev; ++d)
        eng->device_queue((int)d).wait();
    }
    if(lm_head_N_ != N || lm_head_K_ != K){
      std::fprintf(stderr, "gpu: lm_head size mismatch %u x %u vs cached %u x %u\n",
                   N, K, lm_head_N_, lm_head_K_);
      return {};
    }
    // Upload activation as FP16 and run GEMM. Read back halves and
    // convert on the host (device buffer holds N halves, not N floats).
    // x_dev buffers are persistent per device (K fixed = lm_head_K_): the
    // per-token malloc_device in the hot loop cost a UR alloc round-trip
    // each call. x_devs still tracks this call's buffers only (frees are
    // deferred until after the kernel events complete — sycl::free acts on
    // the host immediately; freeing in-loop is a use-after-free race
    // against the not-yet-run kernel).
    //
    // GEMM dispatch (INT8 path): the DPAS s8-weight kernel
    // (gemm::quant_gemm_dpas_s8w) runs when the tile repacked at upload
    // (lm_head_b_dev_) AND the act-quant workspaces exist. It needs INT8
    // acts + FP32 GA=64 scales, quantized on the host per token (the cost
    // the device-side layout was rejected for is O(K) per token — the same
    // O(K) the H2D upload already pays; the win is the XMX kernel replacing
    // the 64x64x64 scalar emulation). INTFER_LMHEAD_DPAS=0 force-disables
    // (bisection). Missing tiles/workspaces fall back to
    // quant_gemm_f16_int8w per tile.
    static const bool trace_lm = (std::getenv("INTFER_TRACE_LMHEAD") != nullptr);
    static const bool no_dpas = (std::getenv("INTFER_LMHEAD_DPAS") &&
                                 std::getenv("INTFER_LMHEAD_DPAS")[0] == '0');
    auto t_lm0 = std::chrono::high_resolution_clock::now();
    std::vector<sycl::half> x_fp16(K);
    for(uint32_t i=0;i<K;i++) x_fp16[i] = sycl::half(fn[i]);
    std::vector<float> out(N, 0.f);
    std::vector<sycl::event> evts;
    evts.reserve(eng->num_devices());
    double lm_kernel_us = 0;
    // Per-device output staging (row shard), concatenated below.
    std::vector<std::vector<sycl::half>> out_seg(eng->num_devices());
    for(size_t d=0; d<eng->num_devices(); ++d){
      uint32_t start = lm_head_ranges_[d].first, end = lm_head_ranges_[d].second;
      uint32_t Nl = end - start;
      out_seg[d].resize(Nl);
      sycl::queue& q = eng->device_queue((int)d);
      // Persistent per-device activation buffer (K fixed for this head).
      // Reused across tokens; resized only if K changes (never in steady
      // state — lm_head_N_/K_ are fixed after the first upload).
      sycl::half* x_dev = (d < lm_head_x_dev_.size()) ? lm_head_x_dev_[d] : nullptr;
      if(!x_dev){
        // x_dev alloc can throw under VRAM pressure (submit path throws
        // UR_RESULT_ERROR_OUT_OF_RESOURCES too — see below): degrade to {}
        // via fail_reset instead of SIGABRT.
        try {
          x_dev = static_cast<sycl::half*>(
              sycl::malloc_device((size_t)K * sizeof(sycl::half), q.get_device(), q.get_context()));
        } catch (...) { x_dev = nullptr; }
        if(!x_dev){ fail_reset(); return {}; }
        if(d >= lm_head_x_dev_.size()) lm_head_x_dev_.resize(d + 1, nullptr);
        lm_head_x_dev_[d] = x_dev;
      }
      x_devs.push_back(x_dev);
      // DPAS path quantizes FP32->INT8 on the host (per GA=64 slice) and
      // uploads the small INT8 row + scales; the scalar path uploads FP16.
      // Only pay for the path we will actually take on this device.
      bool use_dpas = !force_fp16 && !no_dpas && d < lm_head_aq_dev_.size() &&
                      lm_head_aq_dev_[d] && lm_head_as_dev_[d] &&
                      !lm_head_tiles_[d].empty();
      if(use_dpas){
        for(size_t t=0; t<lm_head_tiles_[d].size(); ++t){
          if(t >= lm_head_b_dev_[d].size() || !lm_head_b_dev_[d][t]){
            use_dpas = false;
            break;
          }
        }
      }
      if(use_dpas){
        uint32_t nGA = (K + 63) / 64;
        std::vector<std::int8_t> aq(K);
        std::vector<float> as(nGA);
        dashq::act_quant_ga64_batch(fn, aq.data(), as.data(), K,
                                     dashq::ActQuantRound::RoundNear);
        q.memcpy(lm_head_aq_dev_[d], aq.data(), K);
        q.memcpy(lm_head_as_dev_[d], as.data(), (size_t)nGA * sizeof(float));
      } else {
        q.memcpy(x_dev, x_fp16.data(), (size_t)K * sizeof(sycl::half));
      }
      // One GEMM per resident tile; tiles concatenate in row order.
      // The submit itself can throw when the device is full — catch per
      // tile so a shortfall degrades to {} (fail_reset frees everything
      // including x_devs allocated so far) instead of terminate.
      for(size_t t=0; t<lm_head_tiles_[d].size(); ++t){
        uint32_t ts = lm_head_tiles_[d][t].first, te = lm_head_tiles_[d][t].second;
        uint32_t Nlt = te - ts;
        sycl::event evt;
        try {
          if(force_fp16){
            evt = gemm::quant_gemm_f16(q, x_dev, lm_head_w_dev_[d][t],
                                       lm_head_c_dev_[d][t], 1, Nlt, K);
          } else if(use_dpas){
            gemm::Dpas48Params p;
            p.code_dpas = lm_head_b_dev_[d][t];
            // Scale buffer is PER-TILE (Nlt entries, uploaded from this
            // tile's q_sc): the kernel indexes it tile-locally (n = t*TN+nl
            // in [0,Nlt)), so pass the tile base with NO offset. The old
            // `+ (ts - start)` treated it as a shard base and read OOB
            // garbage scales for every tile past the first (0.8B TP=1 has
            // 4 tiles: 75% of vocab rows got garbage scales -> argmax
            // flipped 11751 -> 196618; FP16/nodpas paths were unaffected).
            p.scale_usm = reinterpret_cast<const std::uint16_t*>(
                lm_head_s_dev_[d][t]);
            p.act_int8 = lm_head_aq_dev_[d];
            p.act_scale = lm_head_as_dev_[d];
            p.alpha = 1.0f;
            p.out_usm = reinterpret_cast<std::uint16_t*>(lm_head_c_dev_[d][t]);
            p.M = 1;
            p.N = Nlt;
            p.K = K;
            p.group_size = K;  // per-row scales => one w-group per row
            p.act_group = 64;
            p.bits = 8;
            evt = gemm::quant_gemm_dpas_s8w(q, p);
          } else {
            evt = gemm::quant_gemm_f16_int8w(q, x_dev, lm_head_q_dev_[d][t],
                                             lm_head_s_dev_[d][t],
                                             lm_head_c_dev_[d][t], 1, Nlt, K);
          }
        } catch (...) {
          std::fprintf(stderr, "[intfer] FATAL: lm_head GEMM submit failed "
                       "on device %zu tile %zu (VRAM pressure?)\n", d, t);
          std::fflush(stderr);
          fail_reset();
          return {};
        }
        // The D2H chains on the kernel event; its own event is waited once
        // after the tile loop (see evts), so tiles overlap while host
        // consumption stays safe.
        sycl::event cp_evt = q.memcpy(out_seg[d].data() + (ts - start),
                 lm_head_c_dev_[d][t],
                 (size_t)Nlt * sizeof(sycl::half), evt);
        // Load-bearing: waiting only on the kernel event leaves the D2H
        // readback in flight while the host consumes out_seg (and frees x_dev).
        // Under full-model device pressure that race kills the device. The
        // D2H event wait (collected below, waited once after the tile loop)
        // makes host consumption safe — WITHOUT a per-tile q.wait(), which
        // drained the whole queue per tile and serialized the 4 tiles
        // (~80ms/token of host block on TP=1; tiles now overlap).
        try {
          auto tss = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
          auto tee = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
          lm_kernel_us += (double)(tee - tss) / 1000.0;
        } catch (...) {}
        evts.push_back(evt);
        evts.push_back(cp_evt);  // D2H completion (covers the old per-tile q.wait)
      }
    }
    for(auto& e : evts) e.wait();
    // x_dev buffers are persistent (see above): do NOT free here.
    // (Old code freed + reallocated per token.)
    x_devs.clear();
    // Concatenate the TP row shards into full-vocab order.
    for(size_t d=0; d<eng->num_devices(); ++d){
      uint32_t start = lm_head_ranges_[d].first;
      for(uint32_t i=0;i<out_seg[d].size();i++)
        out[start+i] = static_cast<float>(out_seg[d][i]);
    }
    if(trace_lm){
      auto t_lm1 = std::chrono::high_resolution_clock::now();
      double wall_us = std::chrono::duration<double, std::micro>(t_lm1 - t_lm0).count();
      std::fprintf(stderr, "[lmhead] N=%u K=%u kernel=%.0fus wall=%.0fus\n",
                   N, K, lm_kernel_us, wall_us);
      std::fflush(stderr);
    }
    return out;
  }

  // Cache state for lm_head upload (tiled per device).
  // INT8 path (default): per-tile INT8 codes + FP16 row scales + FP16 out.
  // DPAS s8w sub-path: per-tile repacked vertical-B codes (lm_head_b_dev_)
  // + persistent INT8 act / FP32 GA-scale workspaces (lm_head_aq_/as_dev_).
  // FP16 path (INTFER_LMHEAD_FP16=1, bisection only): per-tile FP16 weights.
  std::vector<std::vector<sycl::half*>> lm_head_w_dev_;  // fp16 weights (fp16 path)
  std::vector<std::vector<std::int8_t*>> lm_head_q_dev_;  // int8 codes (int8 path)
  std::vector<std::vector<sycl::half*>> lm_head_s_dev_;  // fp16 row scales (int8 path)
  std::vector<std::vector<std::int8_t*>> lm_head_b_dev_;  // repacked s8 tiles (dpas path)
  std::vector<std::int8_t*> lm_head_aq_dev_;  // persistent INT8 act rows (1/device)
  std::vector<float*> lm_head_as_dev_;  // persistent FP32 GA scales (1/device)
  std::vector<std::vector<sycl::half*>> lm_head_c_dev_;  // fp16 outputs (both)
  std::vector<sycl::half*> lm_head_x_dev_;  // persistent act buffers (1/device)
  // M-row batch state for run_lm_head_batch (Lever 1): per-device M-row act
  // buffers + act-quant workspaces (grow-only), per-tile M*Nlt-tight device
  // output staging (kernels write tile stride Nlt) + host mirrors. Freed in
  // fail_reset alongside the scalar cache; the scalar path never touches
  // these, so A/B is clean.
  std::vector<sycl::half*> lm_head_xM_dev_;
  std::vector<uint32_t> lm_head_xM_cap_;
  std::vector<std::int8_t*> lm_head_aqM_dev_;
  std::vector<float*> lm_head_asM_dev_;
  std::vector<uint32_t> lm_head_aqM_cap_;
  std::vector<std::vector<sycl::half*>> lm_head_cM_dev_;
  std::vector<std::vector<size_t>> lm_head_cM_cap_;
  std::vector<std::vector<std::vector<sycl::half>>> lm_head_tM_host_;
  std::vector<std::pair<uint32_t,uint32_t>> lm_head_ranges_;  // TP row shard per device
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> lm_head_tiles_;  // upload/GEMM tiles per device
  uint32_t lm_head_N_ = 0, lm_head_K_ = 0;
  // Device-side SSM state (INTFER_SSM_GPU=1): per-(request,layer) resident
  // S state (n_v*kd*vd FP32) on device 0 + per-key staging (packed H2D
  // inputs, core output, row-pointer arrays sized to the max M seen).
  // Lazily allocated on first use; the host mirror is uploaded (never
  // zeroed — first use may be mid-run with evolved state). Host mirrors
  // stay authoritative for snapshots/rollback: each call optionally syncs
  // device->host, and gpu_ssm_resync re-uploads after a host-side restore.
  // FIFO eviction past 256 keys is always safe: eviction writes a dirty
  // entry back before freeing, and any use re-uploads from the mirror.
  struct SsmDev {
    float* S = nullptr;       // resident [nv*kd*vd]
    float* in = nullptr;      // packed staging H2D [Mcap*(2*nk*kd+nv*vd+2*nv)]
    float* core = nullptr;    // staging out [Mcap*nv*vd]
    float* ptrs = nullptr;    // device row-pointer array [7*Mcap] (as floats bits)
    uint32_t nv = 0, nk = 0, kd = 0, vd = 0, Mcap = 0;
    bool dirty = false;       // resident newer than host mirror
    bool ok = false;
  };
  // Insertion-ordered map for FIFO eviction (std::map + order deque).
  std::map<std::pair<const Request*,int>, SsmDev> ssm_dev_;
  std::deque<std::pair<const Request*,int>> ssm_dev_order_;
  // Layer* per key for eviction writeback (host mirror destination).
  // Stored as the mirror vector pointer: the impl never needs complete
  // Layer/Request (req is pointer-identity only, mirror is a complete
  // std::vector<float>), so the gpu_ssm_rows definition can stay here,
  // ahead of struct Layer/Request.
  std::map<std::pair<const Request*,int>, std::vector<float>*> ssm_dev_mirror_;
  // M-row device-side SSM step; false keeps the host path. Defined after
  // struct Request (needs complete Layer for ly->ssm_state).
  bool ssm_rows(const SsmRow* rows, uint32_t M, int L,
                const float* dt_bias, const float* A_log, bool sync_mirror) {
    return gpu_ssm_rows(*this, rows, M, L, dt_bias, A_log, sync_mirror);
  }
  void ssm_resync_all() { gpu_ssm_resync_all(*this); }
  ~GpuBackend() {
    if(!eng || ssm_dev_.empty()) return;
    sycl::queue& q = eng->device_queue(0);
    for(auto& kv : ssm_dev_) {
      SsmDev& sd = kv.second;
      if(sd.S) sycl::free(sd.S, q);
      if(sd.in) sycl::free(sd.in, q);
      if(sd.core) sycl::free(sd.core, q);
      if(sd.ptrs) sycl::free(sd.ptrs, q);
      sd = SsmDev();
    }
    ssm_dev_.clear();
    ssm_dev_order_.clear();
    ssm_dev_mirror_.clear();
  }
  const Rotation* rot_for(const std::string& name) override {
    if(!eng) return nullptr;
    auto it = ti_cache.find(name);
    if(it == ti_cache.end()) return nullptr;
    return eng->rotation_for(it->second);
  }
};

// Closes the #ifdef INTFER_HAS_GPU_BACKEND opened at struct GpuBackend.
#endif

#ifdef INTFER_HAS_GPU_BACKEND
// M-row device-side SSM step (GpuBackend::ssm_rows impl). Defined here
// (after Layer/Request are complete) rather than at the struct because it
// touches ly->ssm_state.
static bool gpu_ssm_rows(GpuBackend& be, const SsmRow* rows, uint32_t M, int L,
                         const float* dt_bias, const float* A_log,
                         bool sync_mirror) {
  if(!be.eng || !rows || M == 0 || !dt_bias || !A_log) return false;
  const uint32_t nv = (uint32_t)cfg.n_v_heads;
  const uint32_t nk = (uint32_t)cfg.n_k_heads;
  const uint32_t kd = (uint32_t)cfg.k_head_dim;
  const uint32_t vd = (uint32_t)cfg.v_head_dim;
  if(nv == 0 || nk == 0 || kd == 0 || vd == 0) return false;
  const size_t sdim = (size_t)nv * kd * vd;
  const size_t zvw = (size_t)nv * vd;
  for(uint32_t m = 0; m < M; ++m) {
    const SsmRow& r = rows[m];
    if(!r.req || !r.mirror || !r.query || !r.key || !r.value ||
       !r.b_row || !r.a_row || !r.core_out) return false;
    if(r.mirror->size() != sdim) return false;  // keep host path
  }
  sycl::queue& q = be.eng->device_queue(0);
  auto dev = q.get_device();
  auto ctx = q.get_context();
  auto malloc_f = [&](size_t n) -> float* {
    try {
      return static_cast<float*>(sycl::malloc_device(n * sizeof(float), dev, ctx));
    } catch (...) { return nullptr; }
  };
  auto malloc_b = [&](size_t nbytes) -> float* {
    try {
      return static_cast<float*>(sycl::malloc_device(nbytes, dev, ctx));
    } catch (...) { return nullptr; }
  };
  // Partition rows into runs of consecutive rows sharing one (req, layer)
  // state. Single-row runs (distinct states) batch into one ssm_forward;
  // longer runs (one request's prefill chunk) use ssm_forward_chain each.
  // Run leaders carry the key; solo[] flags length-1 runs.
  static thread_local std::vector<uint32_t> run_start, run_len;
  run_start.clear(); run_len.clear();
  for(uint32_t m = 0; m < M;) {
    uint32_t e = m + 1;
    while(e < M && rows[e].req == rows[m].req && rows[e].mirror == rows[m].mirror) ++e;
    run_start.push_back(m); run_len.push_back(e - m);
    m = e;
  }
  const uint32_t nRuns = (uint32_t)run_start.size();
  // Resolve (alloc/upload on first use) each touched key. Staging sized to
  // the run length (chain: full run; solo: gathered below into key K0).
  struct KeyRef { std::pair<const Request*,int> key; GpuBackend::SsmDev* sd; };
  static thread_local std::vector<KeyRef> keys;
  keys.clear();
  auto get_sd = [&](const Request* req, std::vector<float>* mirror, uint32_t need,
                    bool& fresh) -> GpuBackend::SsmDev* {
    std::pair<const Request*,int> key(req, L);
    auto it = be.ssm_dev_.find(key);
    if(it == be.ssm_dev_.end()) {
      if(be.ssm_dev_.size() >= 256 && !be.ssm_dev_order_.empty()) {
        // FIFO evict: write a dirty entry back first (host mirror is the
        // re-upload source), then free.
        auto ek = be.ssm_dev_order_.front(); be.ssm_dev_order_.pop_front();
        auto eit = be.ssm_dev_.find(ek);
        if(eit != be.ssm_dev_.end()) {
          GpuBackend::SsmDev& esd = eit->second;
          auto mit = be.ssm_dev_mirror_.find(ek);
          if(esd.dirty && esd.ok && mit != be.ssm_dev_mirror_.end() &&
             mit->second && mit->second->size() == sdim) {
            try {
              q.memcpy(mit->second->data(), esd.S, sdim * sizeof(float)).wait();
            } catch (...) {}
          }
          if(esd.S) sycl::free(esd.S, q);
          if(esd.in) sycl::free(esd.in, q);
          if(esd.core) sycl::free(esd.core, q);
          if(esd.ptrs) sycl::free(esd.ptrs, q);
          be.ssm_dev_.erase(eit);
          be.ssm_dev_mirror_.erase(ek);
        }
      }
      GpuBackend::SsmDev sd;
      sd.S = malloc_f(sdim);
      if(!sd.S) return nullptr;
      // Upload the CURRENT host mirror (first use may be mid-run).
      try {
        q.memcpy(sd.S, mirror->data(), sdim * sizeof(float)).wait();
      } catch (...) { sycl::free(sd.S, q); return nullptr; }
      sd.nv = nv; sd.nk = nk; sd.kd = kd; sd.vd = vd;
      sd.ok = true; sd.dirty = false;
      be.ssm_dev_[key] = sd;
      be.ssm_dev_order_.push_back(key);
      be.ssm_dev_mirror_[key] = mirror;
      fresh = true;
      return &be.ssm_dev_[key];
    }
    GpuBackend::SsmDev& sd = it->second;
    if(!sd.ok || sd.nv != nv || sd.nk != nk || sd.kd != kd || sd.vd != vd)
      return nullptr;
    // Refresh the mirror pointer (same host vector for a key's lifetime,
    // but refresh is harmless and keeps eviction writeback correct).
    be.ssm_dev_mirror_[key] = mirror;
    // Grow staging to the run length.
    fresh = false;
    if(sd.Mcap < need) {
      float* nin = malloc_f((size_t)need * (2*(size_t)nk*kd + zvw + 2*nv));
      float* ncore = malloc_f((size_t)need * zvw);
      float* nptrs = malloc_b(7 * (size_t)need * sizeof(void*));
      if(!nin || !ncore || !nptrs) {
        if(nin) sycl::free(nin, q);
        if(ncore) sycl::free(ncore, q);
        if(nptrs) sycl::free(nptrs, q);
        return nullptr;
      }
      if(sd.in) sycl::free(sd.in, q);
      if(sd.core) sycl::free(sd.core, q);
      if(sd.ptrs) sycl::free(sd.ptrs, q);
      sd.in = nin; sd.core = ncore; sd.ptrs = nptrs;
      sd.Mcap = need;
    }
    return &sd;
  };
  // Host-side O(kd) per-head prep (identical to the host loop): L2-norm
  // q/k rows, scale q, form g logits. Packed per-row layout in staging:
  // [qn(nk*kd) | kn(nk*kd) | v(nv*vd) | g(nv) | b(nv)].
  const float qscale = 1.0f / sqrtf((float)kd);
  const size_t row_in = 2*(size_t)nk*kd + zvw + 2*nv;
  static thread_local std::vector<float> hin;
  static thread_local std::vector<void*> hptrs;  // pointer-sized slots
  static thread_local std::vector<float> hcore;
  try {
    // Resolve keys first (alloc may wait on upload).
    for(uint32_t r = 0; r < nRuns; ++r) {
      const SsmRow& lr = rows[run_start[r]];
      bool fresh = false;
      GpuBackend::SsmDev* sd = get_sd(lr.req, lr.mirror, run_len[r], fresh);
      if(!sd) return false;
      if(fresh && sd->Mcap < run_len[r]) {
        sd->in = malloc_f((size_t)run_len[r] * row_in);
        sd->core = malloc_f((size_t)run_len[r] * zvw);
        sd->ptrs = malloc_b(7 * (size_t)run_len[r] * sizeof(void*));
        if(!sd->in || !sd->core || !sd->ptrs) {
          if(sd->in) sycl::free(sd->in, q);
          if(sd->core) sycl::free(sd->core, q);
          if(sd->ptrs) sycl::free(sd->ptrs, q);
          *sd = GpuBackend::SsmDev();
          // Drop the broken key so a retry re-allocates.
          auto key = std::make_pair(lr.req, L);
          be.ssm_dev_.erase(key);
          be.ssm_dev_mirror_.erase(key);
          return false;
        }
        sd->Mcap = run_len[r];
      }
      keys.push_back({std::make_pair(lr.req, L), sd});
    }
    sycl::event dep;
    bool have_dep = false;
    static thread_local std::vector<sycl::event> deps;
    deps.clear();
    // Chained-core staging: one host buffer for ALL chained runs (per-run
    // offsets), scattered to per-row outputs after the single wait below.
    static thread_local std::vector<uint32_t> chain_off;
    chain_off.assign(nRuns, 0);
    {
      size_t total = 0;
      for(uint32_t r = 0; r < nRuns; ++r)
        if(run_len[r] > 1) { chain_off[r] = (uint32_t)total; total += run_len[r]; }
      if(hcore.size() < total * zvw) hcore.assign(total * zvw, 0.f);
    }
    for(uint32_t r = 0; r < nRuns; ++r) {
      const uint32_t rs = run_start[r], rl = run_len[r];
      GpuBackend::SsmDev* sd = keys[r].sd;
      if(hin.size() < (size_t)rl * row_in) hin.assign((size_t)rl * row_in, 0.f);
      for(uint32_t i = 0; i < rl; ++i) {
        const SsmRow& rr = rows[rs + i];
        float* dst = &hin[(size_t)i * row_in];
        float* hqn = dst;
        float* hkn = dst + (size_t)nk * kd;
        float* hv = hkn + (size_t)nk * kd;
        float* hg = hv + zvw;
        float* hb = hg + nv;
        for(uint32_t kh = 0; kh < nk; ++kh) {
          l2norm(&rr.query[(size_t)kh * kd], &hqn[(size_t)kh * kd], (int)kd);
          l2norm(&rr.key[(size_t)kh * kd], &hkn[(size_t)kh * kd], (int)kd);
          for(uint32_t d = 0; d < kd; ++d) hqn[(size_t)kh * kd + d] *= qscale;
        }
        memcpy(hv, rr.value, zvw * sizeof(float));
        memcpy(hb, rr.b_row, (size_t)nv * sizeof(float));
        for(uint32_t hh = 0; hh < nv; ++hh) {
          float sp = log1pf(expf(rr.a_row[hh] + dt_bias[hh]));
          hg[hh] = -expf(A_log[hh]) * sp;
        }
      }
      // One H2D for the packed inputs; queue is in-order so the kernel
      // below observes it without explicit depends (same for e1).
      q.memcpy(sd->in, hin.data(), (size_t)rl * row_in * sizeof(float));
      if(rl == 1) {
        // Solo run: single-row ssm_forward via the device pointer array.
        // NOTE: solo runs of DIFFERENT keys cannot share one launch (one
        // state pointer array per key) — each solo run is its own launch.
        // (Batching solos across keys would need a gathered state table;
        // decode M is small; one launch/layer/token each is fine.)
        if(hptrs.size() < 7) hptrs.assign(7, nullptr);
        float* base = sd->in;
        hptrs[0] = base;                            // qn at row offset 0
        hptrs[1] = base + (size_t)nk * kd;          // kn
        hptrs[2] = base + 2 * (size_t)nk * kd;      // v
        hptrs[3] = base + 2 * (size_t)nk * kd + zvw;  // g
        hptrs[4] = base + 2 * (size_t)nk * kd + zvw + nv;  // b
        hptrs[5] = sd->S;                           // state
        hptrs[6] = sd->core;                        // core out
        q.memcpy(sd->ptrs, hptrs.data(), 7 * sizeof(void*));
        gemm::SsmParams p;
        const float* const* dptrs = reinterpret_cast<const float* const*>(sd->ptrs);
        float* const* dptrs_m = reinterpret_cast<float* const*>(sd->ptrs);
        p.qn_rows = dptrs + 0; p.kn_rows = dptrs + 1; p.v_rows = dptrs + 2;
        p.g_rows = dptrs + 3; p.b_rows = dptrs + 4;
        p.s_rows = dptrs_m + 5; p.core_rows = dptrs_m + 6;
        p.M = 1; p.n_v = nv; p.n_k = nk; p.kd = kd; p.vd = vd;
        sycl::event ke = gemm::ssm_forward(q, p);
        sycl::event e2 = q.memcpy(rows[rs].core_out, sd->core,
                                  zvw * sizeof(float), ke);
        deps.push_back(e2);
        if(sync_mirror) {
          sycl::event e3 = q.memcpy(rows[rs].mirror->data(), sd->S,
                                    sdim * sizeof(float), ke);
          deps.push_back(e3);
        }
      } else {
        // Chained run: packed per-row inputs feed ssm_forward_chain
        // directly (row_stride = packed row width).
        gemm::SsmChainParams p;
        p.qn_rows = sd->in;
        p.kn_rows = sd->in + (size_t)nk * kd;
        p.v_rows = sd->in + 2 * (size_t)nk * kd;
        p.g_rows = sd->in + 2 * (size_t)nk * kd + zvw;
        p.b_rows = sd->in + 2 * (size_t)nk * kd + zvw + nv;
        p.state = sd->S;
        p.core_rows = sd->core;
        p.M = rl; p.n_v = nv; p.n_k = nk; p.kd = kd; p.vd = vd;
        p.row_stride = (uint32_t)row_in;
        sycl::event ke = gemm::ssm_forward_chain(q, p);
        sycl::event e2 = q.memcpy(&hcore[(size_t)chain_off[r] * zvw], sd->core,
                                  (size_t)rl * zvw * sizeof(float), ke);
        deps.push_back(e2);
        if(sync_mirror) {
          // State mirror sync piggybacks on the same dependency.
          sycl::event e3 = q.memcpy(rows[rs].mirror->data(), sd->S,
                                    sdim * sizeof(float), ke);
          deps.push_back(e3);
        }
      }
      (void)have_dep; (void)dep;
    }
    for(auto& e : deps) e.wait();
    // Scatter chained cores to per-row outputs (host memcpy, device work
    // complete after the waits above). Solos wrote direct; skip them.
    for(uint32_t r = 0; r < nRuns; ++r) {
      const uint32_t rs = run_start[r], rl = run_len[r];
      if(rl == 1) continue;
      for(uint32_t i = 0; i < rl; ++i)
        memcpy(rows[rs + i].core_out, &hcore[((size_t)chain_off[r] + i) * zvw],
               zvw * sizeof(float));
    }
    for(uint32_t r = 0; r < nRuns; ++r) {
      GpuBackend::SsmDev* sd = keys[r].sd;
      sd->dirty = !sync_mirror;
    }
  } catch (...) {
    return false;
  }
  return true;
}

// Re-upload all host SSM mirrors to the device (speculative partial-accept
// rollback replaced ly.ssm_state vectors under the device; the side map
// holds the host mirror pointers so no Request access is needed here).
// Missing/undersized mirrors are skipped (their keys re-upload lazily).
static void gpu_ssm_resync_all(GpuBackend& be) {
  if(!be.eng) return;
  const size_t sdim = (size_t)cfg.n_v_heads * cfg.k_head_dim * cfg.v_head_dim;
  if(sdim == 0) return;
  sycl::queue& q = be.eng->device_queue(0);
  try {
    for(auto& kv : be.ssm_dev_) {
      GpuBackend::SsmDev& sd = kv.second;
      if(!sd.ok) continue;
      auto mit = be.ssm_dev_mirror_.find(kv.first);
      if(mit == be.ssm_dev_mirror_.end() || !mit->second) continue;
      std::vector<float>& mirror = *mit->second;
      if(mirror.size() != sdim) continue;
      q.memcpy(sd.S, mirror.data(), sdim * sizeof(float)).wait();
      sd.dirty = false;
    }
  } catch (...) {}
}
#endif

// Per-layer precomputed tensor names + host pointers (TODO 4b). step()
// used to rebuild every "model.language_model.layers.L.slot" string via
// snprintf (~366/token) plus per-call get() map finds for SSM params.
// All loop-invariant: build once after backend selection, use by
// reference. ROTATIONS ARE DELIBERATELY NOT HOISTED: CpuBackend::rot_for
// used to be state-dependent — cold (pre-GEMM) it fell through to a
// per-layer first-hit scan while warm (post-ensure_cached) it returned
// the per-slot rotation from cache_R, and resolving up front froze wrong
// pointers (854 -> 13104 on prefill). The fallback is now an exact
// per-slot lookup mirroring ensure_cached, so cold == warm; the
// no-hoist policy stays as cheap insurance (rot_for is a few map finds;
// the snprintf+malloc was the cost).
struct LayerNames {
  std::string q, k, v, o;          // full-attn linears
  std::string qkv, z, b, a, la_out;// LA linears
  std::string gate, up, down;      // MLP linears
  const float *conv_w=nullptr, *A_log=nullptr, *dt_bias=nullptr, *ssm_norm_w=nullptr;
};
static std::vector<LayerNames> g_ln;
// layer_state: pass is_full flags (Layer is defined below; vector<bool>
// avoids the forward-declaration tangle).
static void build_layer_names(
    const std::map<std::string, std::vector<float>>& T,
    const std::vector<char>& layer_is_full){
  g_ln.assign(cfg.layers, LayerNames());
  auto hptr = [&](const std::string& n) -> const float* {
    auto it = T.find(n);
    if(it==T.end()){ std::fprintf(stderr, "missing tensor: %s\n", n.c_str()); std::exit(1); }
    return it->second.data();
  };
  for(int L=0; L<cfg.layers; L++){
    LayerNames& ln = g_ln[L];
    char b[256];
    auto mk = [&](const char* slot) -> std::string {
      snprintf(b, sizeof(b), "model.language_model.layers.%d.%s", L, slot);
      return std::string(b);
    };
    ln.q=mk("self_attn.q_proj.weight"); ln.k=mk("self_attn.k_proj.weight");
    ln.v=mk("self_attn.v_proj.weight"); ln.o=mk("self_attn.o_proj.weight");
    ln.qkv=mk("linear_attn.in_proj_qkv.weight"); ln.z=mk("linear_attn.in_proj_z.weight");
    ln.b=mk("linear_attn.in_proj_b.weight");     ln.a=mk("linear_attn.in_proj_a.weight");
    ln.la_out=mk("linear_attn.out_proj.weight");
    ln.gate=mk("mlp.gate_proj.weight"); ln.up=mk("mlp.up_proj.weight");
    ln.down=mk("mlp.down_proj.weight");
    if(!layer_is_full[(size_t)L]){
      char hb[256];
      auto hm = [&](const char* slot) -> std::string {
        snprintf(hb, sizeof(hb), "model.language_model.layers.%d.%s", L, slot);
        return std::string(hb);
      };
      ln.conv_w=hptr(hm("linear_attn.conv1d.weight"));
      ln.A_log =hptr(hm("linear_attn.A_log"));
      ln.dt_bias=hptr(hm("linear_attn.dt_bias"));
      ln.ssm_norm_w=hptr(hm("linear_attn.norm.weight"));
    }
  }
  std::fprintf(stderr, "[intfer] layer names + host ptrs precomputed for %d layers\n", cfg.layers);
}
// ============================================================================
// Per-layer state (host-resident)
// ============================================================================
struct Layer {
  bool is_full;
  const float *in_ln,*post_ln,*q_norm_w,*k_norm_w;
  const float *in_qkv,*in_z,*in_b,*in_a,*conv_w,*A_log,*dt_bias,*ssm_norm_w,*ssm_out;
  std::vector<float> kcache, vcache;
  std::vector<float> ssm_state, conv_state;
  // Groupwise-INT8 KV mirror (opt-in via INTFER_INT8_KV=1; kernel-compatible
  // layout: scales are FP32 per (kv_head, 32-token group), token t uses
  // group t/32 — same contract as flash_attention_forward_int8_kv).
  // The FP32 kcache/vcache stay authoritative when the knob is off (zero
  // overhead: one branch per row); when on, appends quantize into the
  // mirror and reads dequantize on the fly.
  std::vector<int8_t> kcache_i8, vcache_i8;
  std::vector<float> kscale, vscale;
  bool kv_int8 = false;
  // Per-token G64 Q8 KV codec (opt-in via INTFER_KV_Q8=1; switchable
  // Hadamard via INTFER_KV_HAD=1). Append-only: one FP32 scale per
  // (token, kv_head, 64-elem dim group), token t quantized once at append.
  // Supersedes the legacy token-group mirror when both knobs are set.
  std::vector<int8_t> kcache_q8, vcache_q8;
  std::vector<float> kscale_q8, vscale_q8;
  bool kv_q8 = false;
  bool kv_had = false;
};

// Groupwise-INT8 KV helpers. One FP32 scale per (kv_head, group) over
// group*head_dim elements (symmetric max/127, matching the
// flash-attention demo's quantize_int8_kv). Group size via
// INTFER_INT8_KV_G (default 32, kernel-compatible); smaller groups
// (down to 1 = per-token) trade scale memory for accuracy.
static inline int kv_group_size(){
  static const int g = []{
    const char* e = std::getenv("INTFER_INT8_KV_G");
    int v = e ? std::atoi(e) : 32;
    return (v >= 1 && v <= 1024) ? v : 32;
  }();
  return g;
}
static inline bool kv_int8_enabled(){
  static const bool on = (std::getenv("INTFER_INT8_KV") != nullptr);
  return on;
}
// KV-cache capacity: hard-coded 1024 historically (kcache/vcache assign).
// INTFER_KV_CAP raises it (clamped to [1, 262144] = 27B
// max_position_embeddings) so the 260k context actually fits; allocations
// are still grow-on-demand via resize, so short runs pay nothing extra.
static inline int kv_capacity(){
  static const int cap = []{
    const char* e = std::getenv("INTFER_KV_CAP");
    int v = e ? std::atoi(e) : 1024;
    if(v < 1) return 1024;
    if(v > 262144) return 262144;
    return v;
  }();
  return cap;
}
static inline bool kv_q8_enabled(){
  static const bool on = (std::getenv("INTFER_KV_Q8") != nullptr);
  return on;
}
static inline bool kv_had_enabled(){
  static const bool on = (std::getenv("INTFER_KV_HAD") != nullptr);
  return on;
}
// In-place unscaled Walsh-Hadamard transform (hd=256 is a power of two on
// both known models, so no blockdiag tail). Normalized H/sqrt(n) is
// symmetric orthogonal, hence self-inverse: the same fwht+scale call
// serves as both forward (pre-quant smoothing) and inverse (post-dequant).
static inline void kv_fwht(float* x, int n){
  for(int len = 2; len <= n; len <<= 1){
    int half = len >> 1;
    for(int i = 0; i < n; i += len)
      for(int j = 0; j < half; ++j){
        float u = x[i+j], v = x[i+j+half];
        x[i+j] = u + v; x[i+j+half] = u - v;
      }
  }
  const float inv = 1.0f / sqrtf((float)n);
  for(int i = 0; i < n; ++i) x[i] *= inv;
}
// Requantize the group containing `pos` for all KV heads from the FP32
// cache (call after appending token `pos`). Retroactively updates earlier
// tokens in the same group — inherent to the per-group-scale format.
// Scale layout is (kvh, group) with a FIXED stride smax = cap/G+1
// (cap = kv_capacity()): ng must not vary over time, or
// earlier groups' indices shift when a new group starts.
static void kv_i8_append(Layer& ly, int n_kv, int hd, int pos, bool is_k){
  std::vector<int8_t>& ci = is_k ? ly.kcache_i8 : ly.vcache_i8;
  std::vector<float>& sc = is_k ? ly.kscale : ly.vscale;
  const std::vector<float>& cf = is_k ? ly.kcache : ly.vcache;
  size_t kvrow = (size_t)n_kv * hd;
  size_t need_i8 = (size_t)(pos + 1) * kvrow;
  if(ci.size() < need_i8) ci.resize(need_i8, 0);
  int GS = kv_group_size();
  int smax = kv_capacity() / GS + 1;
  if((int)sc.size() < n_kv * smax) sc.resize((size_t)n_kv * smax, 1.f);
  int g = pos / GS;
  int t0 = g * GS, t1 = pos;
  for(int kvh = 0; kvh < n_kv; kvh++){
    float mx = 0;
    for(int t = t0; t <= t1; t++)
      for(int d = 0; d < hd; d++){
        float v = std::fabs(cf[((size_t)t * n_kv + kvh) * hd + d]);
        if(v > mx) mx = v;
      }
    float s = (mx > 0) ? mx / 127.f : 1.f;
    sc[(size_t)kvh * smax + g] = s;
    float inv = 1.f / s;
    for(int t = t0; t <= t1; t++)
      for(int d = 0; d < hd; d++){
        float v = cf[((size_t)t * n_kv + kvh) * hd + d] * inv;
        int q = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        if(q > 127) q = 127; else if(q < -127) q = -127;
        ci[((size_t)t * n_kv + kvh) * hd + d] = (int8_t)q;
      }
  }
}
// Per-token G64 Q8 KV codec (INTFER_KV_Q8=1). Append-only: token `pos` is
// quantized once at append from the FP32 cache and never touched again
// (no retroactive group updates, unlike the legacy mirror above).
// Scale layout is (token, kv_head, dim-group) with hd/64 groups per row
// (hd=256 -> 4 groups); codes are token-major [pos][n_kv][hd], same as
// the legacy mirror. Optional Hadamard (INTFER_KV_HAD=1): the row is
// forward-rotated (H/sqrt(n), self-inverse) before quant to spread
// outliers, and rotated back after dequant in kv_row_ptr.
static void kv_q8_append(Layer& ly, int n_kv, int hd, int pos, bool is_k){
  std::vector<int8_t>& ci = is_k ? ly.kcache_q8 : ly.vcache_q8;
  std::vector<float>& sc = is_k ? ly.kscale_q8 : ly.vscale_q8;
  const std::vector<float>& cf = is_k ? ly.kcache : ly.vcache;
  const int ng = hd / 64;
  size_t kvrow = (size_t)n_kv * hd;
  size_t need_i8 = (size_t)(pos + 1) * kvrow;
  if(ci.size() < need_i8) ci.resize(need_i8, 0);
  size_t need_sc = (size_t)(pos + 1) * n_kv * ng;
  if(sc.size() < need_sc) sc.resize(need_sc, 1.f);
  static thread_local std::vector<float> work;
  if((int)work.size() < hd) work.resize(hd);
  const bool had = ly.kv_had;
  for(int kvh = 0; kvh < n_kv; kvh++){
    const float* src = &cf[((size_t)pos * n_kv + kvh) * hd];
    float* w = work.data();
    for(int d = 0; d < hd; d++) w[d] = src[d];
    if(had) kv_fwht(w, hd);
    int8_t* dst = &ci[((size_t)pos * n_kv + kvh) * hd];
    for(int dg = 0; dg < ng; dg++){
      float mx = 0;
      for(int d = 0; d < 64; d++){
        float v = std::fabs(w[dg * 64 + d]);
        if(v > mx) mx = v;
      }
      float s = (mx > 0) ? mx / 127.f : 1.f;
      sc[((size_t)pos * n_kv + kvh) * ng + dg] = s;
      float inv = 1.f / s;
      for(int d = 0; d < 64; d++){
        float v = w[dg * 64 + d] * inv;
        int q = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        if(q > 127) q = 127; else if(q < -127) q = -127;
        dst[dg * 64 + d] = (int8_t)q;
      }
    }
  }
}
// Row pointer for (t, kvh): direct FP32 pointer when INT8 is off, else
// dequantize into a thread-local scratch (K and V have separate buffers)
// and return it. Safe: every read site consumes the row (dot/accumulate)
// before the next kv_row_ptr call, so no two returned pointers are ever
// live simultaneously.
static inline const float* kv_row_ptr(const Layer& ly, bool is_k, int t,
                                      int kvh, int n_kv, int hd){
  if(!ly.kv_int8 && !ly.kv_q8)
    return is_k ? &ly.kcache[((size_t)t * n_kv + kvh) * hd]
                : &ly.vcache[((size_t)t * n_kv + kvh) * hd];
  static thread_local std::vector<float> tmp_k, tmp_v;
  std::vector<float>& tmp = is_k ? tmp_k : tmp_v;
  if((int)tmp.size() < hd) tmp.resize(hd);
  if(ly.kv_q8){
    // Per-token G64 path (supersedes the legacy mirror when both are on).
    const std::vector<int8_t>& ci = is_k ? ly.kcache_q8 : ly.vcache_q8;
    const std::vector<float>& sc = is_k ? ly.kscale_q8 : ly.vscale_q8;
    const int ng = hd / 64;
    const int8_t* row = &ci[((size_t)t * n_kv + kvh) * hd];
    for(int dg = 0; dg < ng; dg++){
      float s = sc[((size_t)t * n_kv + kvh) * ng + dg];
      for(int d = 0; d < 64; d++) tmp[dg * 64 + d] = (float)row[dg * 64 + d] * s;
    }
    // Inverse Hadamard (self-inverse): undo the forward rotation applied
    // at append so attention sees the original activation space.
    if(ly.kv_had) kv_fwht(tmp.data(), hd);
    return tmp.data();
  }
  static thread_local std::vector<float> leg_k, leg_v;
  std::vector<float>& leg = is_k ? leg_k : leg_v;
  const std::vector<int8_t>& ci = is_k ? ly.kcache_i8 : ly.vcache_i8;
  const std::vector<float>& sc = is_k ? ly.kscale : ly.vscale;
  int GS = kv_group_size();
  int smax = kv_capacity() / GS + 1;
  float s = sc[((size_t)kvh * smax + (size_t)(t / GS))];
  if((int)leg.size() < hd) leg.resize(hd);
  const int8_t* row = &ci[((size_t)t * n_kv + kvh) * hd];
  for(int d = 0; d < hd; d++) leg[d] = (float)row[d] * s;
  return leg.data();
}

// Append dispatcher: quantize token `pos` into whichever INT8 KV mirror(s)
// are enabled. Q8 (per-token G64) supersedes the legacy token-group mirror
// on reads when both are on; both mirrors are still maintained so each
// stays self-consistent.
// Hard cap guard: pos >= kv_capacity() aborts loudly instead of growing
// the caches unbounded (or reading OOB). All append + pre-size sites call
// this before touching the cache.
static inline void kv_require_cap(int pos){
  if(pos >= kv_capacity()){
    std::fprintf(stderr, "[intfer] FATAL: pos %d >= KV capacity %d (raise INTFER_KV_CAP, max 262144)\n",
                 pos, kv_capacity());
    std::exit(1);
  }
}

static inline void kv_append(Layer& ly, int n_kv, int hd, int pos){
  kv_require_cap(pos);
  if(ly.kv_q8){ kv_q8_append(ly, n_kv, hd, pos, true); kv_q8_append(ly, n_kv, hd, pos, false); }
  if(ly.kv_int8){ kv_i8_append(ly, n_kv, hd, pos, true); kv_i8_append(ly, n_kv, hd, pos, false); }
}

// ============================================================================
// Step scratch arena (TODO 4a): reusable per-token intermediates.
// ============================================================================
// step() used to allocate ~1500 std::vectors per token (GEMM outputs,
// norm copies, attention/ahead temporaries, logits). The arena holds one
// named buffer per intermediate, sized once for the model; run_linear_into
// resizes only when the shape differs, so steady-state decode performs
// zero mallocs for all of these. RSS cost ~ (a few MB); correctness is
// unchanged (same write-then-read order, no aliasing between live values).
struct StepScratch {
  bool sized = false;
  // trunk
  std::vector<float> h, res, normed;
  // full-attention path
  std::vector<float> xq, qp, gate, qpart, xk, xv, k, v, attn, gscores, o_in, o_raw;
  // linear-attention path
  std::vector<float> xqkv, mqkv, conv_out, xz, xb, xa, z, b, a, g, core,
                     kv_mem, delta, normed_core, o_la;
  // MLP
  std::vector<float> rn2, rin, mg, mu, md, mlp_out;
  // tail
  std::vector<float> fn, logits;
  void ensure(int H, int inter, int vocab, int qrows2, int kvrows, int attnrows,
              int conv, int nvb, int nvrows, int vhd){
    auto need = [&](std::vector<float>& v, size_t n){
      if(v.size() != n) v.assign(n, 0.f);
    };
    need(h, H); need(res, H); need(normed, H);
    need(xq, H); need(qp, qrows2); need(gate, attnrows); need(qpart, attnrows);
    need(xk, H); need(xv, H); need(k, kvrows); need(v, kvrows);
    need(attn, attnrows);
    // Grouped attention scores: (max_group_heads) x (pos+1), grown
  // monotonically per run (amortized ~zero). Replaces the old shared
  // single-row `scores` buffer (removed: each head needs its row live
  // through the group's shared V pass).
    need(o_in, attnrows*2);
    need(o_raw, H);
    need(xqkv, H); need(mqkv, conv); need(conv_out, conv);
    need(xz, H); need(xb, H); need(xa, H);
    need(z, nvrows); need(b, nvb); need(a, nvb); need(g, nvb);
    need(core, nvrows); need(kv_mem, vhd); need(delta, vhd);
    need(normed_core, nvrows); need(o_la, H);
    need(rn2, H); need(rin, H); need(mg, inter); need(mu, inter);
    need(md, H); need(mlp_out, H);
    need(fn, H); need(logits, vocab);
    sized = true;
  }
  // Attention scores scratch: grows monotonically with pos (amortized ~zero).
  // NOTE: shared across rows in step_batch (all rows use items[0]'s arena
  // via rar.scratch) but NOT across row kinds within one forward: rows are
  // processed strictly in order and each row's scores are fully consumed
  // (softmax + V-weighted sum + copy to b_attn) before the next row's
  // gscores_for call. Safe today; revisit if rows are ever processed
  // out of order or concurrently.
  float* gscores_for(int gh, int pos){
    if((int)gscores.size() < gh*(pos+1)) gscores.assign((size_t)gh*(pos+1), 0.f);
    return gscores.data();
  }
};
// ============================================================================
// Per-request state (Stage 3.1): one Request owns its KV/SSM layer states +
// scratch arena. Single-request behavior is identical (main() holds one
// Request); the scheduler (3.2) holds one Request per live sequence.
// Stage 3.2 fields (prompt/generated/token/pos/finished/rng/output) are
// only used by the scheduler path; the single-request path keeps its own
// locals and is untouched. Stage 3.3 adds id (server-issued request handle)
// + done_emitted (terminal status already streamed) for the serve loop.
// ============================================================================
struct Request {
  std::vector<Layer> layers;
  StepScratch scratch;
  std::vector<int> prompt_ids;
  std::vector<int> generated;
  int token = -1;   // last produced token (= next forward input)
  int pos = 0;      // absolute position `token` sits at (next forward pos)
  int fill = 0;     // serve: prompt tokens consumed by chunked prefill
  int max_new = 0;
  bool finished = false;
  bool done_emitted = false; // server: {"done"} already emitted for id
  long id = -1;              // server-issued request handle (-1 = CLI batch)
  float temperature = 0.f;   // per-request sampling (CLI: copied from globals;
  int top_k = 0;             // server: per-line overrides or CLI defaults)
  float top_p = 0.f;
  std::mt19937 rng;
  std::string output; // buffered decoded text (multi-request mode)
};

// One live sequence's contribution to a scheduler iteration: exactly one
// forward row (its next token at its own absolute position, against its
// own KV/SSM state). A decode row feeds the request's last produced token;
// a prefill row feeds prompt_ids[pos] (token is resolved from the prompt at
// forward time, since the prefill token was never "produced").
struct BatchItem { Request* req; int token; int pos; bool prefill = false; };

// ============================================================================
// Forward step
// ============================================================================
struct StepOut { int best; std::vector<float> logits; };
static StepOut step(Request& req, int token, int pos,
                    const std::map<std::string, std::vector<float>>& T,
                    const float* embed, const float* final_norm,
                    const float* lm_head, // nullptr => tied embeddings (use embed rows)
                    LinearBackend* backend,
                    bool record, std::vector<std::vector<float>>& dump){
  StepOut out;
  // NaN tracer (INTFER_TRACE_NAN=1): check every GEMM output + hidden state
  // for NaN/Inf and report the first offender. Off by default (hot loop).
  static const bool trace_nan = (std::getenv("INTFER_TRACE_NAN") != nullptr);
  // Hidden-state probe (INTFER_TRACE_H=1): print |h| + h[:4] after each
  // layer's MLP at pos 0 so CPU vs GPU runs can be diffed per layer.
  static const bool trace_h = (std::getenv("INTFER_TRACE_H") != nullptr);
  auto ckpt_h = [&](int L, const std::vector<float>& v, const char* tag){
    if(trace_h && pos==0){
      double n2=0; for(float x : v) n2+=(double)x*x;
      std::fprintf(stderr, "[h] L=%d %s |h|=%.6f h[:4]=%.6f %.6f %.6f %.6f\n",
                   L, tag, sqrt(n2), v[0], v[1], v[2], v[3]);
      std::fflush(stderr);
    }
  };
  auto is_bad = [](const std::vector<float>& v) -> bool {
    for(float x : v) if(!std::isfinite(x)) return true;
    return false;
  };
  auto ckpt = [&](const char* what, int L, const std::vector<float>& v){
    if(trace_nan && is_bad(v)){
      size_t nb = 0; for(float x : v) if(!std::isfinite(x)) nb++;
      std::fprintf(stderr, "[nan] FIRST BAD: %s L=%d pos=%d size=%zu bad=%zu\n",
                   what, L, pos, v.size(), nb);
      std::fflush(stderr);
    }
  };
  const float* row = embed + (size_t)token*cfg.H;
  // Scratch arena: sized once per model dims; all intermediates below reuse
  // these buffers (run_linear_into resizes only on shape change).
  if(!req.scratch.sized)
    req.scratch.ensure(cfg.H, cfg.inter, cfg.vocab,
                     2*cfg.n_heads*cfg.head_dim, cfg.n_kv*cfg.head_dim,
                     cfg.n_heads*cfg.head_dim, cfg.conv_dim, cfg.n_v_heads,
                     cfg.n_v_heads*cfg.v_head_dim, cfg.v_head_dim);
  std::vector<float>& h = req.scratch.h;
  for(int i=0;i<cfg.H;i++) h[i] = row[i];

  // Per-layer names/rotations/host-ptrs are precomputed once (g_ln).
   for(int L=0; L<cfg.layers; L++){
   if((L & 15) == 0){ std::fprintf(stderr, "[intfer] forward L=%d/%d pos=%d hsize=%zu layers=%zu\n", L, cfg.layers, pos, h.size(), req.layers.size()); std::fflush(stderr); }
   if(L < 0 || (size_t)L >= req.layers.size()){ std::fprintf(stderr, "[intfer] FATAL layer OOB\n"); std::exit(1); }
   const LayerNames& ln = g_ln[(size_t)L];
   Layer& ly = req.layers[L];
   std::vector<float>& res = req.scratch.res;
   for(int i=0;i<cfg.H;i++) res[i] = h[i];
   std::vector<float>& normed = req.scratch.normed;
   rmsnorm(h.data(), ly.in_ln, normed.data(), cfg.H);
   ckpt_h(L, normed, "post_in_ln");
  if(ly.is_full){
      // ---- Attention path (q/k/v right-site; o left-site) ----
      const Rotation* Rq = backend->rot_for(ln.q);
      std::vector<float>& xq = req.scratch.xq;
      // Fusion: rot_apply fully overwrites xq, so the pre-copy is dead
      // when Rq is present (else-branch preserves the no-rotation path).
      if(Rq) rot_apply(normed.data(), *Rq, xq.data());
      else for(int i=0;i<cfg.H;i++) xq[i] = normed[i];
      const Rotation* Rk = backend->rot_for(ln.k);
      const Rotation* Rv = backend->rot_for(ln.v);
      std::vector<float>& xk = req.scratch.xk;
      std::vector<float>& xv = req.scratch.xv;
      // Fusion: rot_apply fully overwrites; pre-copy only for !R path.
      if(Rk) rot_apply(normed.data(), *Rk, xk.data());
      else for(int i=0;i<cfg.H;i++) xk[i] = normed[i];
      if(Rv) rot_apply(normed.data(), *Rv, xv.data());
      else for(int i=0;i<cfg.H;i++) xv[i] = normed[i];
      std::vector<float>& qp = req.scratch.qp;
      std::vector<float>& k = req.scratch.k;
      std::vector<float>& v = req.scratch.v;
      size_t qrows = (size_t)cfg.n_heads * cfg.head_dim;
      // Wave-3: q/k/v are independent (all derive from normed + rotations;
      // qp's split/norm/rope consumers run below, after the wave) — one
      // engine call converts + submits all three back-to-back with one
      // wait-loop + one readback pass. Bit-identical to run_linear_into +
      // 2-wave (same kernels, same buffers, only submit/wait grouping
      // changes). Halves host round-trip phases vs q-sequential + k/v wave
      // (one wait-loop + readback pass instead of two).
      {
        LinearBackend::LinearWaveItem items[3] = {
          {ln.q, &xq, &qp},
          {ln.k, &xk, &k},
          {ln.v, &xv, &v},
        };
        backend->run_linear_wave(items, 3);
      }
      ckpt("attn.q_proj", L, qp);
      // q/gate split. Both known models pack q+gate per-head interleaved:
      // rows = 2*n_heads*head_dim (0.8B: 2*8*256=4096; 27B: 2*24*256=12288;
      // verified against safetensors shapes). First head_dim rows per head
      // are q, second are gate. The !packed_gate branch is a defensive
      // fallback for q-only layouts (o_proj then takes attn concat gate).
      // Detect by comparing qp.size() against 2*n_heads*head_dim.
      bool packed_gate = (qp.size() >= 2 * qrows);
      if(!packed_gate && qp.size() < qrows){
        std::fprintf(stderr, "[intfer] FATAL L=%d q_proj short: %zu < %zu; aborting\n",
                     L, qp.size(), qrows);
        std::exit(1);
      }
      std::vector<float>& gate = req.scratch.gate;
      std::vector<float>& qpart = req.scratch.qpart;
      for(size_t i=0;i<qrows;i++){ gate[i] = 0.f; }
      if(packed_gate){
        for(int hh=0; hh<cfg.n_heads; hh++) for(int d=0; d<cfg.head_dim; d++){
          qpart[hh*cfg.head_dim+d] = qp[hh*2*cfg.head_dim+d];
          gate[hh*cfg.head_dim+d]  = qp[hh*2*cfg.head_dim+cfg.head_dim+d];
        }
      } else {
        for(size_t i=0; i<qrows && i<qp.size(); i++) qpart[i] = qp[i];
      }
      for(int hh=0; hh<cfg.n_heads; hh++){
        // Fusion: rmsnorm reduces first, then writes element-wise, so
        // in-place is bit-identical (kills tmp[] + memcpy per head).
        rmsnorm(&qpart[hh*cfg.head_dim], ly.q_norm_w, &qpart[hh*cfg.head_dim], cfg.head_dim);
        rope(&qpart[hh*cfg.head_dim], pos);
      }
      ckpt("attn.k_proj", L, k); ckpt("attn.v_proj", L, v);
      for(int hh=0; hh<cfg.n_kv; hh++){
        // Fusion: in-place rmsnorm is bit-identical (see q_norm above).
        rmsnorm(&k[hh*cfg.head_dim], ly.k_norm_w, &k[hh*cfg.head_dim], cfg.head_dim);
        rope(&k[hh*cfg.head_dim], pos);
      }
      kv_require_cap(pos);
      int base = pos*(cfg.n_kv*cfg.head_dim);
      // Defensive: k/v lie about their width when a layer is missing from the
      // IR and the fallback path mis-shapes (or the cache is short). Never
      // memcpy past the cache — clamp and zero-pad instead of segfaulting.
      {
        size_t need = (size_t)cfg.n_kv*cfg.head_dim;
        size_t have_k = k.size(), have_v = v.size();
        if(have_k < need || have_v < need)
          std::fprintf(stderr, "[intfer] WARN L=%d pos=%d k/v short: %zu/%zu < %zu\n",
                       L, pos, have_k, have_v, need);
        if((size_t)base + need > ly.kcache.size())
          ly.kcache.resize((size_t)base + need, 0.f);
        if((size_t)base + need > ly.vcache.size())
          ly.vcache.resize((size_t)base + need, 0.f);
        if(have_k >= need) memcpy(&ly.kcache[base], k.data(), need*sizeof(float));
        else { memcpy(&ly.kcache[base], k.data(), have_k*sizeof(float));
               memset(&ly.kcache[base]+have_k, 0, (need-have_k)*sizeof(float)); }
        if(have_v >= need) memcpy(&ly.vcache[base], v.data(), need*sizeof(float));
        else { memcpy(&ly.vcache[base], v.data(), have_v*sizeof(float));
               memset(&ly.vcache[base]+have_v, 0, (need-have_v)*sizeof(float)); }
        kv_append(ly, cfg.n_kv, cfg.head_dim, pos);
      }
      // Attention compute, tiled by KV group (4d). Heads sharing a KV
      // head reuse the same K/V cache rows: the old per-head loop loaded
      // each kh[t]/vh[t] row n_heads/n_kv times (4x on 0.8B). Now each
      // row is loaded once per group and dotted/fanned-out to all group
      // heads. Bit-identical: every head's dot/softmax/weighted-sum keeps
      // its original element order; only the interleaving across heads
      // changed (independent reductions).
      std::vector<float>& attn = req.scratch.attn;
      // Fusion: the bulk zero is dead — each head's block is zeroed
      // (oh[d]=0) before accumulation in the loop below.
      const int G = cfg.n_heads / cfg.n_kv; // heads per KV head (group size)
      if(G > 64){ std::fprintf(stderr, "[intfer] FATAL: kv group %d > 64\n", G); std::exit(1); }
      float* gs = req.scratch.gscores_for(G, pos);
      for(int kvh=0; kvh<cfg.n_kv; kvh++){
        // Score pass: for each t load kh once, dot against every head.
        float mx[64]; for(int j=0;j<G;j++) mx[j]=-1e30f;
        for(int t=0;t<=pos;t++){
          const float* kh=kv_row_ptr(ly,true,t,kvh,cfg.n_kv,cfg.head_dim);
          for(int j=0;j<G;j++){
            int hh=kvh*G+j;
            const float* qh=&qpart[hh*cfg.head_dim];
            float s=0; for(int d=0;d<cfg.head_dim;d++) s+=qh[d]*kh[d];
            s*=1.0f/sqrtf((float)cfg.head_dim); gs[j*(pos+1)+t]=s; if(s>mx[j]) mx[j]=s;
          }
        }
        for(int j=0;j<G;j++){
          float* row=&gs[j*(pos+1)];
          float sum=0; for(int t=0;t<=pos;t++){ row[t]=expf(row[t]-mx[j]); sum+=row[t]; }
          for(int t=0;t<=pos;t++) row[t]/=sum;
        }
        // Output pass: zero group outputs, then fan-out shared vh rows.
        for(int j=0;j<G;j++){
          float* oh=&attn[(kvh*G+j)*cfg.head_dim]; for(int d=0;d<cfg.head_dim;d++) oh[d]=0;
        }
        for(int t=0;t<=pos;t++){
          const float* vh=kv_row_ptr(ly,false,t,kvh,cfg.n_kv,cfg.head_dim);
          for(int j=0;j<G;j++){
            float st=gs[j*(pos+1)+t];
            float* oh=&attn[(kvh*G+j)*cfg.head_dim];
            for(int d=0;d<cfg.head_dim;d++) oh[d]+=st*vh[d];
          }
        }
      }
      if (cfg.attn_output_gate && packed_gate)
        for(int i=0;i<cfg.n_heads*cfg.head_dim;i++) attn[i] *= sigmoidf(gate[i]);
      // o_proj contracts attn (packed-gate layout, both known models).
      // The !packed_gate fallback feeds attn concat gate; run_linear
      // contracts min(x.size(), K) so both widths are safe.
      std::vector<float>& o_raw = req.scratch.o_raw;
      {
        size_t attn_n = (size_t)cfg.n_heads * cfg.head_dim;
        // Peek at the IR weight size without caching: run_linear derives K
        // from the stored N/K; instead probe via a zero probe is wasteful,
        // so infer from q layout: unpacked-gate models need attn+gate input.
        // o_in is scratch; the !packed_gate branch copies attn+gate into it,
        // the packed branch passes attn directly (same as old `o_in = attn`
        // copy, minus the allocation).
        const std::vector<float>* o_x = &attn;
        std::vector<float>& o_in = req.scratch.o_in;
        if(!packed_gate){
          if(o_in.size() < attn_n + attn_n) o_in.assign(attn_n + attn_n, 0.f);
          for(size_t i=0;i<attn_n;i++){ o_in[i]=attn[i]; o_in[attn_n+i]=gate[i]; }
          o_x = &o_in;
        }
        backend->run_linear_into(ln.o, *o_x, o_raw);
        ckpt("attn.o_proj", L, o_raw);
        // If the backend used the packed width (K=n_heads*hd) the extra gate
        // tail is ignored by the GEMM (x longer than K); if it used the wide
        // width the full concat is consumed. Both cases are handled because
        // run_linear contracts min(x.size(), K).
      }
      const Rotation* Ro = backend->rot_for(ln.o);
      if(Ro) rot_left(o_raw.data(), *Ro, h.data());
      else for(int i=0;i<cfg.H;i++) h[i] = o_raw[i];
    } else {
      // ---- Linear-attention / SSM path (uses .intifer quantized weights) ----
      const float* conv_w = ln.conv_w;
      const float* A_log  = ln.A_log;
      const float* dt_bias= ln.dt_bias;
      // Use the .intifer dequantized weights via the CPU backend instead of
      // loading FP32 safetensors for these matmuls. The .intifer stores the
      // same weights (FP16-quantized) and the backend dequantizes on demand.
      // SpinQuant: in_proj_qkv/z/b/a are right-site (replay x.R on the
      // activation before the GEMM), out_proj is left-site (R.y after).
      // The LA branch previously skipped both replays, feeding raw normed
      // into rotated-quantized weights (27B: L0 out 0.1781 vs HF -0.0240).
        const Rotation* Rqkv = backend->rot_for(ln.qkv);
        std::vector<float>& xqkv = req.scratch.xqkv;
        // Fusion: rot_apply fully overwrites; pre-copy only for !R path.
        if(Rqkv) rot_apply(normed.data(), *Rqkv, xqkv.data());
        else for(int i=0;i<cfg.H;i++) xqkv[i] = normed[i];
        std::vector<float>& mqkv = req.scratch.mqkv;
        std::vector<float>& conv_out = req.scratch.conv_out;
        const Rotation* Rz = backend->rot_for(ln.z);
        const Rotation* Rb = backend->rot_for(ln.b);
        const Rotation* Ra = backend->rot_for(ln.a);
        std::vector<float>& xz = req.scratch.xz;
        std::vector<float>& xb = req.scratch.xb;
        std::vector<float>& xa = req.scratch.xa;
        // Fusion: rot_apply fully overwrites; pre-copy only for !R path.
        if(Rz) rot_apply(normed.data(), *Rz, xz.data());
        else for(int i=0;i<cfg.H;i++) xz[i] = normed[i];
        if(Rb) rot_apply(normed.data(), *Rb, xb.data());
        else for(int i=0;i<cfg.H;i++) xb[i] = normed[i];
        if(Ra) rot_apply(normed.data(), *Ra, xa.data());
        else for(int i=0;i<cfg.H;i++) xa[i] = normed[i];
        std::vector<float>& z = req.scratch.z;
        std::vector<float>& b = req.scratch.b;
        std::vector<float>& a = req.scratch.a;
        // Wave-4: qkv/z/b/a are independent (all derive from normed;
        // mqkv's conv consumer runs below, after the wave) — one engine
        // call converts + submits all four back-to-back with one wait-loop +
        // one readback pass. Bit-identical to run_linear_into + 3-wave (same
        // kernels, same buffers, only submit/wait grouping changes). Halves
        // host round-trip phases vs qkv-sequential + z/b/a wave (one
        // wait-loop + readback pass instead of two).
        {
          LinearBackend::LinearWaveItem items[4] = {
            {ln.qkv, &xqkv, &mqkv},
            {ln.z, &xz, &z},
            {ln.b, &xb, &b},
            {ln.a, &xa, &a},
          };
          backend->run_linear_wave(items, 4);
        }
        ckpt("la.qkv", L, mqkv);
      for(int c=0;c<cfg.conv_dim;c++){
        float s=0;
        s+=conv_w[c*cfg.conv_k+0]*ly.conv_state[c*3+0];
        s+=conv_w[c*cfg.conv_k+1]*ly.conv_state[c*3+1];
        s+=conv_w[c*cfg.conv_k+2]*ly.conv_state[c*3+2];
        s+=conv_w[c*cfg.conv_k+3]*mqkv[c];
        conv_out[c]=silu(s);
        ly.conv_state[c*3+0]=ly.conv_state[c*3+1];
        ly.conv_state[c*3+1]=ly.conv_state[c*3+2];
        ly.conv_state[c*3+2]=mqkv[c];
      }
      float* query=conv_out.data();
      float* key=conv_out.data()+cfg.k_head_dim*cfg.n_k_heads;
      float* value=conv_out.data()+2*cfg.k_head_dim*cfg.n_k_heads;
        ckpt("la.z", L, z); ckpt("la.b", L, b); ckpt("la.a", L, a);
        std::vector<float>& g = req.scratch.g;
        std::vector<float>& core = req.scratch.core;
      // ---- Device-side SSM (INTFER_SSM_GPU=1, GPU backend only) ----
      // Replaces the host scalar recurrence below with one FP32 SYCL
      // kernel over (n_v_heads x vd) work-items against the request's
      // resident device state. Bit-identical math, same element order;
      // the oracle token stream is the arbiter. Env-off (default) keeps
      // the exact host path. No snapshots in step() => no mirror sync.
      static const bool ssm_gpu_env =
          (std::getenv("INTFER_SSM_GPU") != nullptr);
      bool ssm_done = false;
#ifdef INTFER_HAS_GPU_BACKEND
      if(ssm_gpu_env) {
        if(auto* gpu_be_ssm = dynamic_cast<GpuBackend*>(backend)) {
          if(gpu_be_ssm->eng) {
            SsmRow srow{&req, &ly.ssm_state, query, key, value,
                        b.data(), a.data(), core.data()};
            ssm_done = gpu_be_ssm->ssm_rows(&srow, 1, L, dt_bias, A_log,
                                            /*sync_mirror=*/false);
          }
        }
      }
#else
      (void)ssm_gpu_env;
#endif
      if(!ssm_done) {
      for(int hh=0; hh<cfg.n_v_heads; hh++){
        float sp=log1pf(expf(a[hh]+dt_bias[hh])); g[hh]=-expf(A_log[hh])*sp;
      }
        for(int i=0;i<cfg.n_v_heads*cfg.v_head_dim;i++) core[i] = 0.f;
      float qn[cfg.k_head_dim], kn[cfg.k_head_dim];
        // HF Qwen3NextGatedDeltaNet.fix_query_key_value_ordering: the conv
        // output holds num_k_heads query/key heads but num_v_heads value
        // heads; value head hh shares key head kh = hh*n_k/n_v (GQA-style
        // grouping, n_v % n_k == 0). Indexing q/k by hh over-reads past
        // the query/key sections whenever n_v > n_k (27B: 48 vs 16).
        for(int hh=0; hh<cfg.n_v_heads; hh++){
        int kh = (cfg.n_v_heads > cfg.n_k_heads)
                 ? (int)((long long)hh * cfg.n_k_heads / cfg.n_v_heads)
                 : hh;
        l2norm(&query[kh*cfg.k_head_dim], qn, cfg.k_head_dim);
        l2norm(&key[kh*cfg.k_head_dim],  kn, cfg.k_head_dim);
        float scale=1.0f/sqrtf((float)cfg.k_head_dim);
        for(int d=0;d<cfg.k_head_dim;d++) qn[d]*=scale;
        float g_t=expf(g[hh]); float beta=sigmoidf(b[hh]);
        float* S=&ly.ssm_state[hh*cfg.k_head_dim*cfg.v_head_dim];
        for(int i=0;i<cfg.k_head_dim*cfg.v_head_dim;i++) S[i]*=g_t;
        // kv_mem/delta are per-head scratch (reused across heads).
        std::vector<float>& kv_mem = req.scratch.kv_mem;
        std::vector<float>& delta = req.scratch.delta;
        for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]=0.f;
        for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]+=S[d*cfg.v_head_dim+c]*kn[d];
        { const float* vh=&value[hh*cfg.v_head_dim];
        for(int c=0;c<cfg.v_head_dim;c++) delta[c]=(vh[c]-kv_mem[c])*beta; }
        for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) S[d*cfg.v_head_dim+c]+=kn[d]*delta[c];
        for(int c=0;c<cfg.v_head_dim;c++){
          float s=0; for(int d=0;d<cfg.k_head_dim;d++) s+=S[d*cfg.v_head_dim+c]*qn[d];
          core[hh*cfg.v_head_dim+c]=s;
        }
      }
      } // end host-SSM fallback (device path sets ssm_done)
        const float* ssm_norm_w = ln.ssm_norm_w;
      std::vector<float>& normed_core = req.scratch.normed_core;
      for(int hh=0; hh<cfg.n_v_heads; hh++){
        const float* x=&core[hh*cfg.v_head_dim];
        const float* zg=&z[hh*cfg.v_head_dim];
        float* o=&normed_core[hh*cfg.v_head_dim];
        float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=x[i]*x[i];
        float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
        for(int i=0;i<cfg.v_head_dim;i++) o[i]=(x[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
      }
      std::vector<float>& o_la = req.scratch.o_la;
      backend->run_linear_into(ln.la_out, normed_core, o_la);
      const Rotation* Ro_la = backend->rot_for(ln.la_out);
      if(Ro_la){ for(int i=0;i<cfg.H;i++) h[i] = 0.f; rot_left(o_la.data(), *Ro_la, h.data()); }
      else for(int i=0;i<cfg.H;i++) h[i] = o_la[i];
      ckpt("la.out", L, h);
  if(trace_h && pos==0 && (L==0 || L==3)){
    // Debug-only path: keep allocating impl (cold, gated by env flag).
    std::vector<float> pmqkv = backend->run_linear(ln.qkv, xqkv);
    std::vector<float> pz = backend->run_linear(ln.z, xz);
    std::vector<float> pa = backend->run_linear(ln.a, xa);
    std::vector<float> pb = backend->run_linear(ln.b, xb);
    std::fprintf(stderr, "[dbg] L=%d pos=%d h_attn[:4]=%.4f %.4f %.4f %.4f qkv_sz=%zu qkv0=%.4f z0=%.4f a0=%.4f b0=%.4f\n",
                 L, pos, h[0], h[1], h[2], h[3], pmqkv.size(),
                 pmqkv.empty()?-999:pmqkv[0], pz.empty()?-999:pz[0],
                 pa.empty()?-999:pa[0], pb.empty()?-999:pb[0]);
    std::fflush(stderr);
  }
      }
    for(int i=0;i<cfg.H;i++) h[i] += res[i];
    ckpt("h.post_attn", L, h);
    ckpt_h(L, h, "post_attn");
    // ---- MLP (gate/up right-site; down left-site) ----
    std::vector<float>& rn2 = req.scratch.rn2;
    rmsnorm(h.data(), ly.post_ln, rn2.data(), cfg.H);
    const Rotation* Rg = backend->rot_for(ln.gate);
    std::vector<float>& rin = req.scratch.rin;
    // Fusion: rot_apply fully overwrites; pre-copy only for !R path.
    if(Rg) rot_apply(rn2.data(), *Rg, rin.data());
    else for(int i=0;i<cfg.H;i++) rin[i] = rn2[i];
    std::vector<float>& mg = req.scratch.mg;
    std::vector<float>& mu = req.scratch.mu;
    std::vector<float>& md = req.scratch.md;
    // Fused MLP: device chains gate/up wave -> silu-mul -> down (TP=1);
    // otherwise the default runs wave + host silu-mul + down (mg/mu
    // filled). Fused path leaves mg/mu unfilled — their only consumers
    // below are trace-gated (which force the host path).
    if(!backend->run_mlp_fused(ln.gate, ln.up, ln.down, rin, mg, mu, md, 1)){
      // Wave: gate/up share the rin input and are independent — one engine
      // call, one host round-trip. Bit-identical to two run_linear_into.
      {
        LinearBackend::LinearWaveItem items[2] = {
          {ln.gate, &rin, &mg},
          {ln.up, &rin, &mu},
        };
        backend->run_linear_wave(items, 2);
      }
      ckpt("mlp.gate", L, mg);
      ckpt("mlp.up", L, mu);
      for(int i=0;i<cfg.inter;i++) mg[i] = silu(mg[i])*mu[i];
      backend->run_linear_into(ln.down, mg, md);
    }
    ckpt("mlp.down", L, md);
    const Rotation* Rd = backend->rot_for(ln.down);
    if(trace_h && pos==0 && L==0){
      double n2=0; for(float x : mg) n2+=(double)x*x;
      std::fprintf(stderr, "[dbg] L=%d mlp mg|.|=%.6f mg[:4]=%.6f %.6f %.6f %.6f mu[:4]=%.6f %.6f %.6f %.6f md[:4]=%.6f %.6f %.6f %.6f Rg=%p Rd=%p\n",
                   L, sqrt(n2), mg[0], mg[1], mg[2], mg[3], mu[0], mu[1], mu[2], mu[3],
                   md[0], md[1], md[2], md[3], (const void*)Rg, (const void*)Rd);
      std::fflush(stderr);
    }
    std::vector<float>& mlp_out = req.scratch.mlp_out;
    if(Rd) rot_left(md.data(), *Rd, mlp_out.data());
    else for(int i=0;i<cfg.H;i++) mlp_out[i] = md[i];
    for(int i=0;i<cfg.H;i++) h[i] = h[i] + mlp_out[i];
    ckpt("h.post_mlp", L, h);
    ckpt_h(L, h, "post_mlp");
    if(record){
      if(pos < 0 || (size_t)pos >= dump[L].size()/cfg.H){
        std::fprintf(stderr, "[intfer] FATAL L=%d dump OOB: pos=%d cap=%zu\n",
                     L, pos, dump[L].size()/cfg.H);
        std::exit(1);
      }
      memcpy(&dump[L][(size_t)pos*cfg.H], h.data(), (size_t)cfg.H*sizeof(float));
    }
  }
  // Final norm + lm_head + greedy argmax. Untied models (27B,
  // tie_word_embeddings=false) use the separate lm_head.weight rows;
  // tied models (0.8B) reuse the embed rows.
  std::vector<float>& fn = req.scratch.fn;
  rmsnorm(h.data(), final_norm, fn.data(), cfg.H);
#ifdef QP_PROBE_SIZES
  if(pos==0 || pos==5){
    std::fprintf(stderr, "[dbg] L23? pos=%d h[:4]=%.4f %.4f %.4f %.4f fn[:4]=%.4f %.4f %.4f %.4f\n",
                 pos, h[0], h[1], h[2], h[3], fn[0], fn[1], fn[2], fn[3]);
    std::fflush(stderr);
  }
#endif
  // logits live in scratch; the GPU lm_head path writes into them directly,
  // the CPU path fills them in place. out.logits keeps one copy (1 alloc +
  // 1MB memcpy per token — negligible vs the ~1500 allocs eliminated).
  std::vector<float>& logits = req.scratch.logits;
  int best=-1; float bestv=-1e30f;
  const float* head = lm_head ? lm_head : embed;
  std::fprintf(stderr, "[intfer] logits: vocab=%d H=%d head=%p\n", cfg.vocab, cfg.H, (const void*)head);
  std::fflush(stderr);
  if(backend){
    auto gpu = dynamic_cast<GpuBackend*>(backend);
    if(gpu){
      // GPU path for both untied (lm_head) and tied (embed rows) models.
      // Tied models (0.8B) otherwise burn vocab*H host MACs per token
      // (248320x1024 = 254M — ~5x the GPU GEMM work, the decode
      // bottleneck per profiling: kernels were 10.6% of decode).
      // The embed table uploads once via the backend's (N,K)-keyed cache.
      const float* W = lm_head ? lm_head : embed;
      std::vector<float> lm = gpu->run_lm_head_f16(fn.data(), W, (uint32_t)cfg.vocab, (uint32_t)cfg.H);
      if((int)lm.size() == cfg.vocab) for(int i=0;i<cfg.vocab;i++) logits[i] = lm[i];
    }else{
      for(int v=0; v<cfg.vocab; v++){
        const float* er = head + (size_t)v*cfg.H;
        float s=0; for(int i=0;i<cfg.H;i++) s+=fn[i]*er[i];
        logits[v]=s;
        if(s>bestv){ bestv=s; best=v; }
      }
    }
  }else{
    for(int v=0; v<cfg.vocab; v++){
      const float* er = head + (size_t)v*cfg.H;
      float s=0; for(int i=0;i<cfg.H;i++) s+=fn[i]*er[i];
      logits[v]=s;
      if(s>bestv){ bestv=s; best=v; }
    }
  }
  if(lm_head==nullptr || !backend || dynamic_cast<CpuBackend*>(backend) ||
     dynamic_cast<GpuBackend*>(backend)){
    for(int v=0; v<cfg.vocab; v++){
      if(logits[v]>bestv){ bestv=logits[v]; best=v; }
    }
  }
  out.best = best;
  out.logits = logits; // one copy out of scratch (see note above)
  return out;
}

// ============================================================================
// Chunked prefill, layer-outer loop (Phase B).
// ============================================================================
// The M=1 step() drives layers-inner (all 48 layers per token), so each
// prompt token evicts+rebuilds every layer's shards. This path inverts the
// loops: for each layer, run all chunk tokens through that layer's linears
// (still one M=1 run_linear_into per token in Phase B; Phase C batches
// them via run_linear_batch). KV/conv/SSM appends happen in token order
// inside the layer visit, so numerics are bit-identical to step() — the
// per-token op sequence (rmsnorm/rope/silu/softmax order, residual adds)
// is unchanged, only the interleaving across tokens differs (independent
// reductions). The final-norm + lm_head runs for the last chunk token only,
// matching what the old loop kept (generated = last position's argmax).
//
// H is the chunk's hidden states, row-major [M,H], already embedded.
// Returns the last row's greedy id + logits; fills dump rows [pos0,pos0+M).
// Caller must have sized layers' kvcache/vcache for pos0+M (step() grows
// them on demand via resize; here we pre-size to keep appends in-bounds).
// all_out (optional): when set, every row is final-normed + lm_head-scored
// (speculative verifier needs all M logits); out still returns last row.
// row_snaps (optional): when set, per-row conv/SSM snapshots are captured
// for LA layers (speculative partial-accept rollback; full-attn layers use
// absolute-position KV caches truncated by size, needing no copies).
struct CacheSnap {
  size_t ksz = 0, vsz = 0;
  std::vector<float> conv, ssm;
};
static StepOut prefill_layer_outer(Request& req, const std::vector<int>& toks, int pos0,
                     const std::map<std::string, std::vector<float>>& T,
                     std::vector<float>& H, // [M,H] in/out hidden states
                     const float* embed, const float* final_norm,
                     const float* lm_head,
                     LinearBackend* backend,
                     std::vector<std::vector<float>>& dump,
                     std::vector<StepOut>* all_out = nullptr,
                     std::vector<std::vector<CacheSnap>>* row_snaps = nullptr){
  StepOut out;
  const int M = (int)toks.size();
  const int Hdim = cfg.H;
  for(int i=0;i<M;i++){
    const float* row = embed + (size_t)toks[i]*Hdim;
    memcpy(&H[(size_t)i*Hdim], row, (size_t)Hdim*sizeof(float));
  }
  if(!req.scratch.sized)
    req.scratch.ensure(cfg.H, cfg.inter, cfg.vocab,
                     2*cfg.n_heads*cfg.head_dim, cfg.n_kv*cfg.head_dim,
                     cfg.n_heads*cfg.head_dim, cfg.conv_dim, cfg.n_v_heads,
                     cfg.n_v_heads*cfg.v_head_dim, cfg.v_head_dim);
  // Reusable per-token row buffers (no per-token mallocs in the loop).
  std::vector<float> h(Hdim), res(Hdim), normed(Hdim);
  std::vector<float> xbuf(Hdim), obuf;
  // Chunk-batch staging (Phase C): M-row normed/rotated/GEMM buffers.
  // Dedicated vectors (not the M=1 arena) since widths scale with M.
  std::vector<float> c_normed, c_x, c_y;   // [M,H] in, [M,H] rotated, [M,N] out
  std::vector<float> c_mg, c_mu;           // [M,inter] MLP gate/up
  std::vector<float> c_qp;                 // [M,2*n_heads*hd] q_proj out
  std::vector<float> c_k, c_v;             // [M,n_kv*hd]
  // Phase C: M-wide norm/rotation/GEMM staging. When INTFER_BATCH_GEMM=1
  // (and backend is GPU) the 7 linears run as single M-row launches via
  // run_linear_batch; elementwise math stays per-row (bit-identical: each
  // row is an independent GEMM row + the same scalar sequence as step()).
  // NOTE: rot_for() must be resolved per row? No — rotations are static
  // per tensor name (SpinQuant; the fallback is an exact per-slot lookup
  // so resolution is cache-state independent), so one lookup per
  // layer is enough; rows differ only in activations.
  // Default ON (2026-09-18: M-row launches halve prefill submits,
  // bit-identical rows); INTFER_BATCH_GEMM=0 forces the per-row path.
  static const char* batch_env = std::getenv("INTFER_BATCH_GEMM");
  static const bool batch_gemm =
      (batch_env == nullptr || std::string(batch_env) != "0");
  // CPU batch override exists (single fork over M*N rows), so the M-row
  // chunk path pays off on the CPU backend too — not just distributed.
  bool use_batch = batch_gemm && backend;
  // M-row activation staging: H rows -> normed -> rotated inputs.
  // Sized lazily on first chunk (chunk_M may exceed the M=1 arena widths
  // for q/k/v/o intermediates, so these live in dedicated chunk buffers).
  auto need_rows = [&](std::vector<float>& v, size_t cols){
    size_t n = (size_t)M * cols;
    if(v.size() != n) v.assign(n, 0.f);
  };
  for(int L=0; L<cfg.layers; L++){
    if((L & 15) == 0){ std::fprintf(stderr, "[intfer] prefill-chunk L=%d/%d M=%d\n", L, cfg.layers, M); std::fflush(stderr); }
    const LayerNames& ln = g_ln[(size_t)L];
    Layer& ly = req.layers[(size_t)L];
    // Pre-size KV caches for the whole chunk (step() would resize per pos).
    kv_require_cap(pos0+M-1);
    size_t kv_need = (size_t)(pos0+M)*cfg.n_kv*cfg.head_dim;
    if(ly.kcache.size() < kv_need) ly.kcache.resize(kv_need, 0.f);
    if(ly.vcache.size() < kv_need) ly.vcache.resize(kv_need, 0.f);
    bool batch_this_layer = use_batch;
    if(!batch_this_layer){
    for(int m=0; m<M; m++){
      int pos = pos0 + m;
      memcpy(h.data(), &H[(size_t)m*Hdim], (size_t)Hdim*sizeof(float));
      memcpy(res.data(), h.data(), (size_t)Hdim*sizeof(float));
      rmsnorm(h.data(), ly.in_ln, normed.data(), Hdim);
      if(ly.is_full){
        const Rotation* Rq = backend->rot_for(ln.q);
        if(Rq) rot_apply(normed.data(), *Rq, xbuf.data());
        else memcpy(xbuf.data(), normed.data(), (size_t)Hdim*sizeof(float));
        backend->run_linear_into(ln.q, xbuf, obuf);
        size_t qrows = (size_t)cfg.n_heads * cfg.head_dim;
        bool packed_gate = (obuf.size() >= 2 * qrows);
        std::vector<float>& gate = req.scratch.gate;
        std::vector<float>& qpart = req.scratch.qpart;
        for(size_t i=0;i<qrows;i++) gate[i] = 0.f;
        if(packed_gate){
          for(int hh=0; hh<cfg.n_heads; hh++) for(int d=0; d<cfg.head_dim; d++){
            qpart[hh*cfg.head_dim+d] = obuf[hh*2*cfg.head_dim+d];
            gate[hh*cfg.head_dim+d]  = obuf[hh*2*cfg.head_dim+cfg.head_dim+d];
          }
        } else {
          for(size_t i=0; i<qrows && i<obuf.size(); i++) qpart[i] = obuf[i];
        }
        for(int hh=0; hh<cfg.n_heads; hh++){
          rmsnorm(&qpart[hh*cfg.head_dim], ly.q_norm_w, &qpart[hh*cfg.head_dim], cfg.head_dim);
          rope(&qpart[hh*cfg.head_dim], pos);
        }
        const Rotation* Rk = backend->rot_for(ln.k);
        const Rotation* Rv = backend->rot_for(ln.v);
        std::vector<float>& xk = req.scratch.xk;
        std::vector<float>& xv = req.scratch.xv;
        if(Rk) rot_apply(normed.data(), *Rk, xk.data());
        else memcpy(xk.data(), normed.data(), (size_t)Hdim*sizeof(float));
        if(Rv) rot_apply(normed.data(), *Rv, xv.data());
        else memcpy(xv.data(), normed.data(), (size_t)Hdim*sizeof(float));
        std::vector<float>& qp = req.scratch.qp;
        std::vector<float>& k = req.scratch.k;
        std::vector<float>& v = req.scratch.v;
        backend->run_linear_into(ln.k, xk, k);
        backend->run_linear_into(ln.v, xv, v);
        for(int hh=0; hh<cfg.n_kv; hh++){
          rmsnorm(&k[hh*cfg.head_dim], ly.k_norm_w, &k[hh*cfg.head_dim], cfg.head_dim);
          rope(&k[hh*cfg.head_dim], pos);
        }
        kv_require_cap(pos);
        int base = pos*(cfg.n_kv*cfg.head_dim);
        size_t need = (size_t)cfg.n_kv*cfg.head_dim;
        if(k.size() >= need) memcpy(&ly.kcache[base], k.data(), need*sizeof(float));
        if(v.size() >= need) memcpy(&ly.vcache[base], v.data(), need*sizeof(float));
        kv_append(ly, cfg.n_kv, cfg.head_dim, pos);
        std::vector<float>& attn = req.scratch.attn;
        const int G = cfg.n_heads / cfg.n_kv;
        float* gs = req.scratch.gscores_for(G, pos);
        for(int kvh=0; kvh<cfg.n_kv; kvh++){
          float mx[64]; for(int j=0;j<G;j++) mx[j]=-1e30f;
          for(int t=0;t<=pos;t++){
            const float* kh=kv_row_ptr(ly,true,t,kvh,cfg.n_kv,cfg.head_dim);
            for(int j=0;j<G;j++){
              int hh=kvh*G+j;
              const float* qh=&qpart[hh*cfg.head_dim];
              float s=0; for(int d=0;d<cfg.head_dim;d++) s+=qh[d]*kh[d];
              s*=1.0f/sqrtf((float)cfg.head_dim); gs[j*(pos+1)+t]=s; if(s>mx[j]) mx[j]=s;
            }
          }
          for(int j=0;j<G;j++){
            float* row=&gs[j*(pos+1)];
            float sum=0; for(int t=0;t<=pos;t++){ row[t]=expf(row[t]-mx[j]); sum+=row[t]; }
            for(int t=0;t<=pos;t++) row[t]/=sum;
          }
          for(int j=0;j<G;j++){
            float* oh=&attn[(kvh*G+j)*cfg.head_dim]; for(int d=0;d<cfg.head_dim;d++) oh[d]=0;
          }
          for(int t=0;t<=pos;t++){
            const float* vh=kv_row_ptr(ly,false,t,kvh,cfg.n_kv,cfg.head_dim);
            for(int j=0;j<G;j++){
              float st=gs[j*(pos+1)+t];
              float* oh=&attn[(kvh*G+j)*cfg.head_dim];
              for(int d=0;d<cfg.head_dim;d++) oh[d]+=st*vh[d];
            }
          }
        }
        if (cfg.attn_output_gate && packed_gate)
          for(int i=0;i<cfg.n_heads*cfg.head_dim;i++) attn[i] *= sigmoidf(gate[i]);
        std::vector<float>& o_raw = req.scratch.o_raw;
        {
          size_t attn_n = (size_t)cfg.n_heads * cfg.head_dim;
          const std::vector<float>* o_x = &attn;
          std::vector<float>& o_in = req.scratch.o_in;
          if(!packed_gate){
            if(o_in.size() < attn_n + attn_n) o_in.assign(attn_n + attn_n, 0.f);
            for(size_t i=0;i<attn_n;i++){ o_in[i]=attn[i]; o_in[attn_n+i]=gate[i]; }
            o_x = &o_in;
          }
          backend->run_linear_into(ln.o, *o_x, o_raw);
        }
        const Rotation* Ro = backend->rot_for(ln.o);
        if(Ro) rot_left(o_raw.data(), *Ro, h.data());
        else memcpy(h.data(), o_raw.data(), (size_t)Hdim*sizeof(float));
      } else {
        const float* conv_w = ln.conv_w;
        const float* A_log  = ln.A_log;
        const float* dt_bias= ln.dt_bias;
        const Rotation* Rqkv = backend->rot_for(ln.qkv);
        std::vector<float>& xqkv = req.scratch.xqkv;
        if(Rqkv) rot_apply(normed.data(), *Rqkv, xqkv.data());
        else memcpy(xqkv.data(), normed.data(), (size_t)Hdim*sizeof(float));
        std::vector<float>& mqkv = req.scratch.mqkv;
        backend->run_linear_into(ln.qkv, xqkv, mqkv);
        std::vector<float>& conv_out = req.scratch.conv_out;
        for(int c=0;c<cfg.conv_dim;c++){
          float s=0;
          s+=conv_w[c*cfg.conv_k+0]*ly.conv_state[c*3+0];
          s+=conv_w[c*cfg.conv_k+1]*ly.conv_state[c*3+1];
          s+=conv_w[c*cfg.conv_k+2]*ly.conv_state[c*3+2];
          s+=conv_w[c*cfg.conv_k+3]*mqkv[c];
          conv_out[c]=silu(s);
          ly.conv_state[c*3+0]=ly.conv_state[c*3+1];
          ly.conv_state[c*3+1]=ly.conv_state[c*3+2];
          ly.conv_state[c*3+2]=mqkv[c];
        }
        float* query=conv_out.data();
        float* key=conv_out.data()+cfg.k_head_dim*cfg.n_k_heads;
        float* value=conv_out.data()+2*cfg.k_head_dim*cfg.n_k_heads;
        const Rotation* Rz = backend->rot_for(ln.z);
        const Rotation* Rb = backend->rot_for(ln.b);
        const Rotation* Ra = backend->rot_for(ln.a);
        std::vector<float>& xz = req.scratch.xz;
        std::vector<float>& xb = req.scratch.xb;
        std::vector<float>& xa = req.scratch.xa;
        if(Rz) rot_apply(normed.data(), *Rz, xz.data());
        else memcpy(xz.data(), normed.data(), (size_t)Hdim*sizeof(float));
        if(Rb) rot_apply(normed.data(), *Rb, xb.data());
        else memcpy(xb.data(), normed.data(), (size_t)Hdim*sizeof(float));
        if(Ra) rot_apply(normed.data(), *Ra, xa.data());
        else memcpy(xa.data(), normed.data(), (size_t)Hdim*sizeof(float));
        std::vector<float>& z = req.scratch.z;
        std::vector<float>& b = req.scratch.b;
        std::vector<float>& a = req.scratch.a;
        // Wave: z/b/a are independent (all derive from normed) — one
        // engine call, one host round-trip. Bit-identical to three
        // run_linear_into calls.
        {
          LinearBackend::LinearWaveItem items[3] = {
            {ln.z, &xz, &z},
            {ln.b, &xb, &b},
            {ln.a, &xa, &a},
          };
          backend->run_linear_wave(items, 3);
        }
        std::vector<float>& g = req.scratch.g;
        std::vector<float>& core = req.scratch.core;
      // ---- Device-side SSM (INTFER_SSM_GPU=1; scalar prefill row) ----
      // Same contract as the step() hook; sync_mirror follows row_snaps
      // (the speculative verifier captures ly.ssm_state below when set).
      // NOTE: M=1 per row here — conv_state evolves sequentially per row,
      // so rows cannot batch; the kernel still removes the O(nv*kd*vd)
      // host recurrence per row.
      bool ssm_done = false;
#ifdef INTFER_HAS_GPU_BACKEND
      {
        static const bool ssm_gpu_env =
            (std::getenv("INTFER_SSM_GPU") != nullptr);
        if(ssm_gpu_env) {
          if(auto* gpu_be = dynamic_cast<GpuBackend*>(backend)) {
            if(gpu_be->eng) {
              SsmRow srow{&req, &ly.ssm_state, query, key, value,
                          b.data(), a.data(), core.data()};
              ssm_done = gpu_be->ssm_rows(&srow, 1, L, dt_bias, A_log,
                                          /*sync_mirror=*/row_snaps != nullptr);
            }
          }
        }
      }
#endif
      if(!ssm_done) {
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          float sp=log1pf(expf(a[hh]+dt_bias[hh])); g[hh]=-expf(A_log[hh])*sp;
        }
        for(int i=0;i<cfg.n_v_heads*cfg.v_head_dim;i++) core[i] = 0.f;
        float qn[cfg.k_head_dim], kn[cfg.k_head_dim];
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          int kh = (cfg.n_v_heads > cfg.n_k_heads)
                   ? (int)((long long)hh * cfg.n_k_heads / cfg.n_v_heads)
                   : hh;
          l2norm(&query[kh*cfg.k_head_dim], qn, cfg.k_head_dim);
          l2norm(&key[kh*cfg.k_head_dim],  kn, cfg.k_head_dim);
          float scale=1.0f/sqrtf((float)cfg.k_head_dim);
          for(int d=0;d<cfg.k_head_dim;d++) qn[d]*=scale;
          float g_t=expf(g[hh]); float beta=sigmoidf(b[hh]);
          float* S=&ly.ssm_state[hh*cfg.k_head_dim*cfg.v_head_dim];
          for(int i=0;i<cfg.k_head_dim*cfg.v_head_dim;i++) S[i]*=g_t;
          std::vector<float>& kv_mem = req.scratch.kv_mem;
          std::vector<float>& delta = req.scratch.delta;
          for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]=0.f;
          for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]+=S[d*cfg.v_head_dim+c]*kn[d];
          { const float* vh=&value[hh*cfg.v_head_dim];
          for(int c=0;c<cfg.v_head_dim;c++) delta[c]=(vh[c]-kv_mem[c])*beta; }
          for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) S[d*cfg.v_head_dim+c]+=kn[d]*delta[c];
          for(int c=0;c<cfg.v_head_dim;c++){
            float s=0; for(int d=0;d<cfg.k_head_dim;d++) s+=S[d*cfg.v_head_dim+c]*qn[d];
            core[hh*cfg.v_head_dim+c]=s;
          }
        }
      } // end device/host SSM (scalar prefill row)
        const float* ssm_norm_w = ln.ssm_norm_w;
        std::vector<float>& normed_core = req.scratch.normed_core;
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          const float* x=&core[hh*cfg.v_head_dim];
          const float* zg=&z[hh*cfg.v_head_dim];
          float* o=&normed_core[hh*cfg.v_head_dim];
          float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=x[i]*x[i];
          float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
          for(int i=0;i<cfg.v_head_dim;i++) o[i]=(x[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
        }
        // Speculative verifier (TODO3.3, scalar path): same post-row
        // conv/SSM capture as the batched path below (ring states can't be
        // truncated like the absolute KV caches). Without this, spec_rows
        // stays empty on the per-row path (INTFER_BATCH_GEMM=0) and the
        // verifier's restore indexes an empty vector (SIGSEGV).
        if(row_snaps){
          auto& rws = *row_snaps;
          if((int)rws.size() != cfg.layers) rws.resize(cfg.layers);
          if((int)rws[(size_t)L].size() != M) rws[(size_t)L].resize(M);
          CacheSnap& sn = rws[(size_t)L][m];
          sn.conv = ly.conv_state;
          sn.ssm = ly.ssm_state;
        }
        std::vector<float>& o_la = req.scratch.o_la;
        backend->run_linear_into(ln.la_out, normed_core, o_la);
        const Rotation* Ro_la = backend->rot_for(ln.la_out);
        if(Ro_la){ for(int i=0;i<Hdim;i++) h[i] = 0.f; rot_left(o_la.data(), *Ro_la, h.data()); }
        else memcpy(h.data(), o_la.data(), (size_t)Hdim*sizeof(float));
      }
      for(int i=0;i<Hdim;i++) h[i] += res[i];
      std::vector<float>& rn2 = req.scratch.rn2;
      rmsnorm(h.data(), ly.post_ln, rn2.data(), Hdim);
      const Rotation* Rg = backend->rot_for(ln.gate);
      std::vector<float>& rin = req.scratch.rin;
      if(Rg) rot_apply(rn2.data(), *Rg, rin.data());
      else memcpy(rin.data(), rn2.data(), (size_t)Hdim*sizeof(float));
      std::vector<float>& mg = req.scratch.mg;
      std::vector<float>& mu = req.scratch.mu;
      std::vector<float>& md = req.scratch.md;
      // Fused MLP (device chain on TP=1, else host path below).
      if(!backend->run_mlp_fused(ln.gate, ln.up, ln.down, rin, mg, mu, md, 1)){
      // Wave: gate/up share rin and are independent — one host round-trip.
      {
        LinearBackend::LinearWaveItem items[2] = {
          {ln.gate, &rin, &mg},
          {ln.up, &rin, &mu},
        };
        backend->run_linear_wave(items, 2);
      }
      for(int i=0;i<cfg.inter;i++) mg[i] = silu(mg[i])*mu[i];
      backend->run_linear_into(ln.down, mg, md);
      }
      const Rotation* Rd = backend->rot_for(ln.down);
      std::vector<float>& mlp_out = req.scratch.mlp_out;
      if(Rd) rot_left(md.data(), *Rd, mlp_out.data());
      else memcpy(mlp_out.data(), md.data(), (size_t)Hdim*sizeof(float));
      for(int i=0;i<Hdim;i++) h[i] = h[i] + mlp_out[i];
      memcpy(&H[(size_t)m*Hdim], h.data(), (size_t)Hdim*sizeof(float));
      // Empty outer dump = skip recording (server path never records).
      // The inner guard alone is not enough: dump[L] on an empty outer
      // vector is UB before the comparison even runs.
      if(!dump.empty() && (size_t)pos < dump[L].size()/Hdim)
        memcpy(&dump[L][(size_t)pos*Hdim], h.data(), (size_t)Hdim*sizeof(float));
    } // end token loop (scalar path; skipped for batched full-attn layers)
    } // end if(!batch_this_layer)
    if(use_batch && ly.is_full){
      // ---- Batched full-attention layer (Phase C) ----
      // Rows 0..M-1 of H hold post-prev-layer hiddens (residuals already
      // include this layer's... no: H currently holds INPUT hiddens for
      // this layer (we broke out before the residual add). Stage:
      //   1. per-row: rmsnorm(in_ln) -> c_normed rows.
      //   2. q = batchGEMM(rot(c_normed)); split q/gate per row; per-row
      //      q_norm + rope(pos0+m); k,v = batchGEMM(rot(c_normed)) per row
      //      k_norm + rope; append KV in token order.
      //   3. per-row causal attention over kcache/vcache (same op order as
      //      the scalar loop above, incl. grouped tiling + output gate).
      //   4. attn rows -> batchGEMM(o) -> rot_left -> +res -> post_ln ->
      //      batchGEMM(gate/up) -> silu* -> batchGEMM(down) -> +res -> H.
      size_t qrows = (size_t)cfg.n_heads * cfg.head_dim;
      size_t kvrows = (size_t)cfg.n_kv * cfg.head_dim;
      size_t qrows2 = 2 * qrows;
      need_rows(c_normed, Hdim);
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly.in_ln, &c_normed[(size_t)m*Hdim], Hdim);
      const Rotation* Rq = backend->rot_for(ln.q);
      const Rotation* Rk = backend->rot_for(ln.k);
      const Rotation* Rv = backend->rot_for(ln.v);
      need_rows(c_x, Hdim);
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rq) rot_apply(nr, *Rq, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      need_rows(c_qp, qrows2);
      backend->run_linear_batch(ln.q, c_x, c_qp, (uint32_t)M);
      bool packed_gate = (c_qp.size() >= (size_t)M * qrows2);
      // Per-row q/gate split + q_norm + rope, staged into c_qpart rows.
      need_rows(c_k, kvrows); need_rows(c_v, kvrows);
      static thread_local std::vector<float> c_qpart, c_gate;
      need_rows(c_qpart, qrows); need_rows(c_gate, qrows);
      // k/v rotated inputs share one staging buffer (consumed per GEMM).
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rk) rot_apply(nr, *Rk, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_kv_in = c_x; // copy of rotated-k inputs
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rv) rot_apply(nr, *Rv, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_v_in = c_x; // copy of rotated-v inputs
      // Wave-batch: k/v independent (q already ran above) — one engine
      // call, one host round-trip. Bit-identical to two run_linear_batch.
      {
        LinearBackend::LinearWaveBatchItem items[2] = {
          {ln.k, &c_kv_in, &c_k},
          {ln.v, &c_v_in, &c_v},
        };
        backend->run_linear_wave_batch(items, 2, (uint32_t)M);
      }
      // Per-row q split / norms / ropes + KV appends (token order).
      // NOTE: q/k/v GEMM outputs are FP16-quantized on device (to_half_bits
      // per element) while the M=1 path goes through the same device
      // rounding — row bit-parity holds vs M sequential calls (bench-gated:
      // max abs diff 2.4e-4 on gate_proj), and the oracle gate below is the
      // end-to-end arbiter.
      for(int m=0; m<M; m++){
        int pos = pos0 + m;
        float* qpr = &c_qpart[(size_t)m*qrows];
        float* gar = &c_gate[(size_t)m*qrows];
        const float* qpr_in = &c_qp[(size_t)m*qrows2];
        for(size_t i=0;i<qrows;i++) gar[i] = 0.f;
        if(packed_gate){
          for(int hh=0; hh<cfg.n_heads; hh++) for(int d=0; d<cfg.head_dim; d++){
            qpr[hh*cfg.head_dim+d] = qpr_in[hh*2*cfg.head_dim+d];
            gar[hh*cfg.head_dim+d] = qpr_in[hh*2*cfg.head_dim+cfg.head_dim+d];
          }
        } else {
          for(size_t i=0; i<qrows; i++) qpr[i] = qpr_in[i];
        }
        for(int hh=0; hh<cfg.n_heads; hh++){
          rmsnorm(&qpr[hh*cfg.head_dim], ly.q_norm_w, &qpr[hh*cfg.head_dim], cfg.head_dim);
          rope(&qpr[hh*cfg.head_dim], pos);
        }
        float* kr = &c_k[(size_t)m*kvrows];
        float* vr = &c_v[(size_t)m*kvrows];
        for(int hh=0; hh<cfg.n_kv; hh++){
          rmsnorm(&kr[hh*cfg.head_dim], ly.k_norm_w, &kr[hh*cfg.head_dim], cfg.head_dim);
          rope(&kr[hh*cfg.head_dim], pos);
        }
        kv_require_cap(pos);
      int base = pos*(cfg.n_kv*cfg.head_dim);
        memcpy(&ly.kcache[base], kr, kvrows*sizeof(float));
        memcpy(&ly.vcache[base], vr, kvrows*sizeof(float));
        kv_append(ly, cfg.n_kv, cfg.head_dim, pos);
      }
      // Per-row causal attention (same grouped tiling as scalar path).
      static thread_local std::vector<float> c_attn;
      need_rows(c_attn, qrows);
      {
        std::vector<float>& attn = req.scratch.attn;
        const int G = cfg.n_heads / cfg.n_kv;
        for(int m=0; m<M; m++){
          int pos = pos0 + m;
          float* qpr = &c_qpart[(size_t)m*qrows];
          float* gar = &c_gate[(size_t)m*qrows];
          float* atr = &c_attn[(size_t)m*qrows];
          // Reuse the arena attn buffer as the compute target, then copy
          // the row out (keeps the exact scalar op sequence below).
          float* gs = req.scratch.gscores_for(G, pos);
          for(int kvh=0; kvh<cfg.n_kv; kvh++){
            float mx[64]; for(int j=0;j<G;j++) mx[j]=-1e30f;
            for(int t=0;t<=pos;t++){
              const float* kh=kv_row_ptr(ly,true,t,kvh,cfg.n_kv,cfg.head_dim);
              for(int j=0;j<G;j++){
                int hh=kvh*G+j;
                const float* qh=&qpr[hh*cfg.head_dim];
                float s=0; for(int d=0;d<cfg.head_dim;d++) s+=qh[d]*kh[d];
                s*=1.0f/sqrtf((float)cfg.head_dim); gs[j*(pos+1)+t]=s; if(s>mx[j]) mx[j]=s;
              }
            }
            for(int j=0;j<G;j++){
              float* row=&gs[j*(pos+1)];
              float sum=0; for(int t=0;t<=pos;t++){ row[t]=expf(row[t]-mx[j]); sum+=row[t]; }
              for(int t=0;t<=pos;t++) row[t]/=sum;
            }
            for(int j=0;j<G;j++){
              float* oh=&attn[(kvh*G+j)*cfg.head_dim]; for(int d=0;d<cfg.head_dim;d++) oh[d]=0;
            }
            for(int t=0;t<=pos;t++){
              const float* vh=kv_row_ptr(ly,false,t,kvh,cfg.n_kv,cfg.head_dim);
              for(int j=0;j<G;j++){
                float st=gs[j*(pos+1)+t];
                float* oh=&attn[(kvh*G+j)*cfg.head_dim];
                for(int d=0;d<cfg.head_dim;d++) oh[d]+=st*vh[d];
              }
            }
          }
          if (cfg.attn_output_gate && packed_gate)
            for(size_t i=0;i<qrows;i++) attn[i] *= sigmoidf(gar[i]);
          memcpy(atr, attn.data(), qrows*sizeof(float));
        }
      }
      // o_proj batched over rows (packed-gate layout; !packed_gate models
      // need attn+gate concat per row — build it).
      {
        const std::vector<float>* o_x = &c_attn;
        static thread_local std::vector<float> c_o_in;
        if(!packed_gate){
          need_rows(c_o_in, 2*qrows);
          for(int m=0; m<M; m++){
            memcpy(&c_o_in[(size_t)m*2*qrows], &c_attn[(size_t)m*qrows], qrows*sizeof(float));
            memcpy(&c_o_in[(size_t)m*2*qrows+qrows], &c_gate[(size_t)m*qrows], qrows*sizeof(float));
          }
          o_x = &c_o_in;
        }
        need_rows(c_y, Hdim);
        backend->run_linear_batch(ln.o, *o_x, c_y, (uint32_t)M);
      }
      const Rotation* Ro = backend->rot_for(ln.o);
      // Residual + post_ln + MLP (batched gate/up, elementwise, batched
      // down) + residual, per row.
      need_rows(c_mg, cfg.inter); need_rows(c_mu, cfg.inter);
      static thread_local std::vector<float> c_rn2, c_rin, c_md;
      need_rows(c_rn2, Hdim); need_rows(c_rin, Hdim); need_rows(c_md, Hdim);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];       // layer input (res base)
        float* orw = &c_y[(size_t)m*Hdim];    // o_proj row
        // rot_left writes into a target; stage through c_rn2 row then add.
        float* dst = &c_rn2[(size_t)m*Hdim];  // reuse as temp
        if(Ro) rot_left(orw, *Ro, dst);
        else memcpy(dst, orw, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] += dst[i];
      }
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly.post_ln, &c_rn2[(size_t)m*Hdim], Hdim);
      const Rotation* Rg = backend->rot_for(ln.gate);
      for(int m=0; m<M; m++){
        const float* r2 = &c_rn2[(size_t)m*Hdim];
        float* ri = &c_rin[(size_t)m*Hdim];
        if(Rg) rot_apply(r2, *Rg, ri); else memcpy(ri, r2, (size_t)Hdim*sizeof(float));
      }
      // Wave-batch: gate/up share c_rin and are independent — one
      // engine call, one host round-trip. Bit-identical to two
      // run_linear_batch.
      // Fused MLP (device chain on TP=1, else host path below).
      if(!backend->run_mlp_fused(ln.gate, ln.up, ln.down, c_rin, c_mg, c_mu, c_md, (uint32_t)M)){
      {
        LinearBackend::LinearWaveBatchItem items[2] = {
          {ln.gate, &c_rin, &c_mg},
          {ln.up,   &c_rin, &c_mu},
        };
        backend->run_linear_wave_batch(items, 2, (uint32_t)M);
      }
      for(int m=0; m<M; m++){
        float* mgr = &c_mg[(size_t)m*cfg.inter];
        const float* mur = &c_mu[(size_t)m*cfg.inter];
        for(int i=0;i<cfg.inter;i++) mgr[i] = silu(mgr[i])*mur[i];
      }
      backend->run_linear_batch(ln.down, c_mg, c_md, (uint32_t)M);
      }
      const Rotation* Rd = backend->rot_for(ln.down);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* mdr = &c_md[(size_t)m*Hdim];
        float* dst = &c_rn2[(size_t)m*Hdim]; // reuse as temp
        if(Rd) rot_left(mdr, *Rd, dst);
        else memcpy(dst, mdr, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] += dst[i];
        int pos = pos0 + m;
        if(!dump.empty() && (size_t)pos < dump[L].size()/Hdim)
          memcpy(&dump[L][(size_t)pos*Hdim], hr, (size_t)Hdim*sizeof(float));
      }
    }
    if(use_batch && !ly.is_full){
      // ---- Batched linear-attention / SSM layer ----
      // GEMMs (qkv/z/b/a/la_out) run as single M-row launches; the conv +
      // SSM recurrences stay per-row in token order (sequential state, same
      // as KV appends in the full-attn path). Bit-identical vs the scalar
      // loop above: same per-row op sequence, only GEMM rows batched.
      const float* conv_w = ln.conv_w;
      const float* A_log  = ln.A_log;
      const float* dt_bias= ln.dt_bias;
      static thread_local std::vector<float> c_mqkv, c_z, c_b, c_a;
      static thread_local std::vector<float> c_ncore;
      need_rows(c_normed, Hdim);
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly.in_ln, &c_normed[(size_t)m*Hdim], Hdim);
      const Rotation* Rqkv = backend->rot_for(ln.qkv);
      const Rotation* Rz = backend->rot_for(ln.z);
      const Rotation* Rb = backend->rot_for(ln.b);
      const Rotation* Ra = backend->rot_for(ln.a);
      need_rows(c_x, Hdim);
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rqkv) rot_apply(nr, *Rqkv, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      need_rows(c_mqkv, cfg.conv_dim);
      backend->run_linear_batch(ln.qkv, c_x, c_mqkv, (uint32_t)M);
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rz) rot_apply(nr, *Rz, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_z_in = c_x;
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rb) rot_apply(nr, *Rb, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_b_in = c_x;
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Ra) rot_apply(nr, *Ra, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      size_t zvw = (size_t)cfg.n_v_heads * cfg.v_head_dim;
      need_rows(c_z, zvw);
      need_rows(c_b, cfg.n_v_heads);
      need_rows(c_a, cfg.n_v_heads);
      // Wave: z/b/a are independent M-row GEMMs — one engine call converts
      // + submits all three back-to-back and waits once. Bit-identical to
      // three sequential run_linear_batch calls.
      {
        LinearBackend::LinearWaveBatchItem items[3] = {
          {ln.z, &c_z_in, &c_z},
          {ln.b, &c_b_in, &c_b},
          {ln.a, &c_x, &c_a},
        };
        backend->run_linear_wave_batch(items, 3, (uint32_t)M);
      }
      need_rows(c_ncore, zvw);
      // Per-row conv + SSM recurrence in token order (stateful).
      std::vector<float> conv_out(cfg.conv_dim);
      std::vector<float> kv_mem(cfg.v_head_dim), delta(cfg.v_head_dim);
      std::vector<float> qn(cfg.k_head_dim), kn(cfg.k_head_dim);
      std::vector<float> core(zvw), gg(cfg.n_v_heads);
      // Device-side SSM (INTFER_SSM_GPU=1): the conv loop below stores one
      // row copy per chunk row; a single M-row device step then replaces
      // the host recurrence (the whole chunk shares one (req, layer)
      // state => one chained run). ssm_rows false => host loop over the
      // stored rows (identical math, same order).
      static thread_local std::vector<SsmRow> c_ssm_rows;
      static thread_local std::vector<float> c_ssm_conv;  // [M*conv_dim]
      bool c_ssm_on = false;
#ifdef INTFER_HAS_GPU_BACKEND
      {
        static const bool ssm_gpu_env =
            (std::getenv("INTFER_SSM_GPU") != nullptr);
        if(ssm_gpu_env) {
          if(auto* gpu_be0 = dynamic_cast<GpuBackend*>(backend))
            c_ssm_on = (gpu_be0->eng != nullptr);
        }
      }
#endif
      if(c_ssm_on) {
        c_ssm_rows.assign(M, SsmRow());
        c_ssm_conv.assign((size_t)M * cfg.conv_dim, 0.f);
      }
      for(int m=0; m<M; m++){
        float* mqkvr = &c_mqkv[(size_t)m*cfg.conv_dim];
        for(int c=0;c<cfg.conv_dim;c++){
          float s=0;
          s+=conv_w[c*cfg.conv_k+0]*ly.conv_state[c*3+0];
          s+=conv_w[c*cfg.conv_k+1]*ly.conv_state[c*3+1];
          s+=conv_w[c*cfg.conv_k+2]*ly.conv_state[c*3+2];
          s+=conv_w[c*cfg.conv_k+3]*mqkvr[c];
          conv_out[c]=silu(s);
          ly.conv_state[c*3+0]=ly.conv_state[c*3+1];
          ly.conv_state[c*3+1]=ly.conv_state[c*3+2];
          ly.conv_state[c*3+2]=mqkvr[c];
        }
        if(c_ssm_on) {
          memcpy(&c_ssm_conv[(size_t)m*cfg.conv_dim], conv_out.data(),
                 (size_t)cfg.conv_dim*sizeof(float));
          continue;
        }
        float* query=conv_out.data();
        float* key=conv_out.data()+cfg.k_head_dim*cfg.n_k_heads;
        float* value=conv_out.data()+2*cfg.k_head_dim*cfg.n_k_heads;
        float* zr = &c_z[(size_t)m*zvw];
        float* br = &c_b[(size_t)m*cfg.n_v_heads];
        float* ar = &c_a[(size_t)m*cfg.n_v_heads];
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          float sp=log1pf(expf(ar[hh]+dt_bias[hh])); gg[hh]=-expf(A_log[hh])*sp;
        }
        for(size_t i=0;i<zvw;i++) core[i] = 0.f;
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          int kh = (cfg.n_v_heads > cfg.n_k_heads)
                   ? (int)((long long)hh * cfg.n_k_heads / cfg.n_v_heads)
                   : hh;
          l2norm(&query[kh*cfg.k_head_dim], qn.data(), cfg.k_head_dim);
          l2norm(&key[kh*cfg.k_head_dim],  kn.data(), cfg.k_head_dim);
          float scale=1.0f/sqrtf((float)cfg.k_head_dim);
          for(int d=0;d<cfg.k_head_dim;d++) qn[d]*=scale;
          float g_t=expf(gg[hh]); float beta=sigmoidf(br[hh]);
          float* S=&ly.ssm_state[hh*cfg.k_head_dim*cfg.v_head_dim];
          for(int i=0;i<cfg.k_head_dim*cfg.v_head_dim;i++) S[i]*=g_t;
          for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]=0.f;
          for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]+=S[d*cfg.v_head_dim+c]*kn[d];
          { const float* vh=&value[hh*cfg.v_head_dim];
          for(int c=0;c<cfg.v_head_dim;c++) delta[c]=(vh[c]-kv_mem[c])*beta; }
          for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) S[d*cfg.v_head_dim+c]+=kn[d]*delta[c];
          for(int c=0;c<cfg.v_head_dim;c++){
            float s=0; for(int d=0;d<cfg.k_head_dim;d++) s+=S[d*cfg.v_head_dim+c]*qn[d];
            core[hh*cfg.v_head_dim+c]=s;
          }
        }
        const float* ssm_norm_w = ln.ssm_norm_w;
        float* ncr = &c_ncore[(size_t)m*zvw];
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          const float* xx=&core[hh*cfg.v_head_dim];
          const float* zg=&zr[hh*cfg.v_head_dim];
          float* oo=&ncr[hh*cfg.v_head_dim];
          float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=xx[i]*xx[i];
          float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
          for(int i=0;i<cfg.v_head_dim;i++) oo[i]=(xx[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
        }
        // Speculative verifier (TODO3.3): capture post-row conv/SSM state so
        // a partial accept can restore the exact accepted-boundary state
        // (ring states can't be truncated like the absolute KV caches).
        if(row_snaps){
          auto& rws = *row_snaps;
          if((int)rws.size() != cfg.layers) rws.resize(cfg.layers);
          if((int)rws[(size_t)L].size() != M) rws[(size_t)L].resize(M);
          CacheSnap& sn = rws[(size_t)L][m];
          sn.conv = ly.conv_state;
          sn.ssm = ly.ssm_state;
        }
      }
      if(c_ssm_on) {
        // Device step over the whole chunk (one chained run: same key).
        // core rows land directly in c_ncore; the norm below reads them.
        for(int m=0; m<M; m++){
          SsmRow& sr = c_ssm_rows[(size_t)m];
          const float* cq = &c_ssm_conv[(size_t)m * cfg.conv_dim];
          sr.req = &req; sr.mirror = &ly.ssm_state;
          sr.query = cq;
          sr.key = cq + cfg.k_head_dim * cfg.n_k_heads;
          sr.value = cq + 2 * cfg.k_head_dim * cfg.n_k_heads;
          sr.b_row = &c_b[(size_t)m * cfg.n_v_heads];
          sr.a_row = &c_a[(size_t)m * cfg.n_v_heads];
          sr.core_out = &c_ncore[(size_t)m * zvw];
        }
        bool ok = false;
#ifdef INTFER_HAS_GPU_BACKEND
        if(auto* gpu_be = dynamic_cast<GpuBackend*>(backend)) {
          if(gpu_be->eng)
            ok = gpu_be->ssm_rows(c_ssm_rows.data(), (uint32_t)M, L,
                                  dt_bias, A_log,
                                  /*sync_mirror=*/row_snaps != nullptr);
        }
#endif
        if(ok) {
          // Norm + snapshot tail over device cores (mirrors host loop).
          const float* ssm_norm_w = ln.ssm_norm_w;
          for(int m=0; m<M; m++){
            float* ncr = &c_ncore[(size_t)m*zvw];
            float* zr = &c_z[(size_t)m*zvw];
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              float* xx=&ncr[hh*cfg.v_head_dim];
              const float* zg=&zr[hh*cfg.v_head_dim];
              float* oo=&ncr[hh*cfg.v_head_dim];
              float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=xx[i]*xx[i];
              float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
              // In-place norm: xx and oo alias (ncr row) — matches the
              // host loop, where core[] is read into ncr[] distinctly...
              // host reads core[] and writes ncr[]; here xx IS ncr. The
              // per-element update uses only xx[i] (no cross terms), so
              // in-place is elementwise-safe.
              for(int i=0;i<cfg.v_head_dim;i++)
                oo[i]=(xx[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
            }
            if(row_snaps){
              auto& rws = *row_snaps;
              if((int)rws.size() != cfg.layers) rws.resize(cfg.layers);
              if((int)rws[(size_t)L].size() != M) rws[(size_t)L].resize(M);
              CacheSnap& sn = rws[(size_t)L][m];
              sn.conv = ly.conv_state;
              sn.ssm = ly.ssm_state;
            }
          }
        } else {
          // Device declined: host recurrence over the stored conv rows
          // (same math/order as the env-off loop above).
          const float* ssm_norm_w = ln.ssm_norm_w;
          for(int m=0; m<M; m++){
            const float* cq = &c_ssm_conv[(size_t)m * cfg.conv_dim];
            const float* query = cq;
            const float* key = cq + cfg.k_head_dim * cfg.n_k_heads;
            const float* value = cq + 2 * cfg.k_head_dim * cfg.n_k_heads;
            float* zr = &c_z[(size_t)m*zvw];
            float* br = &c_b[(size_t)m*cfg.n_v_heads];
            float* ar = &c_a[(size_t)m*cfg.n_v_heads];
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              float sp=log1pf(expf(ar[hh]+dt_bias[hh])); gg[hh]=-expf(A_log[hh])*sp;
            }
            for(size_t i=0;i<zvw;i++) core[i] = 0.f;
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              int kh = (cfg.n_v_heads > cfg.n_k_heads)
                       ? (int)((long long)hh * cfg.n_k_heads / cfg.n_v_heads)
                       : hh;
              l2norm(&query[kh*cfg.k_head_dim], qn.data(), cfg.k_head_dim);
              l2norm(&key[kh*cfg.k_head_dim],  kn.data(), cfg.k_head_dim);
              float scale=1.0f/sqrtf((float)cfg.k_head_dim);
              for(int d=0;d<cfg.k_head_dim;d++) qn[d]*=scale;
              float g_t=expf(gg[hh]); float beta=sigmoidf(br[hh]);
              float* S=&ly.ssm_state[hh*cfg.k_head_dim*cfg.v_head_dim];
              for(int i=0;i<cfg.k_head_dim*cfg.v_head_dim;i++) S[i]*=g_t;
              for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]=0.f;
              for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]+=S[d*cfg.v_head_dim+c]*kn[d];
              { const float* vh=&value[hh*cfg.v_head_dim];
              for(int c=0;c<cfg.v_head_dim;c++) delta[c]=(vh[c]-kv_mem[c])*beta; }
              for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) S[d*cfg.v_head_dim+c]+=kn[d]*delta[c];
              for(int c=0;c<cfg.v_head_dim;c++){
                float s=0; for(int d=0;d<cfg.k_head_dim;d++) s+=S[d*cfg.v_head_dim+c]*qn[d];
                core[hh*cfg.v_head_dim+c]=s;
              }
            }
            float* ncr = &c_ncore[(size_t)m*zvw];
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              const float* xx=&core[hh*cfg.v_head_dim];
              const float* zg=&zr[hh*cfg.v_head_dim];
              float* oo=&ncr[hh*cfg.v_head_dim];
              float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=xx[i]*xx[i];
              float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
              for(int i=0;i<cfg.v_head_dim;i++) oo[i]=(xx[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
            }
            if(row_snaps){
              auto& rws = *row_snaps;
              if((int)rws.size() != cfg.layers) rws.resize(cfg.layers);
              if((int)rws[(size_t)L].size() != M) rws[(size_t)L].resize(M);
              CacheSnap& sn = rws[(size_t)L][m];
              sn.conv = ly.conv_state;
              sn.ssm = ly.ssm_state;
            }
          }
        }
      }
      need_rows(c_y, Hdim);
      backend->run_linear_batch(ln.la_out, c_ncore, c_y, (uint32_t)M);
      const Rotation* Ro_la = backend->rot_for(ln.la_out);
      need_rows(c_mg, cfg.inter); need_rows(c_mu, cfg.inter);
      static thread_local std::vector<float> c_rn2_la, c_rin_la, c_md_la;
      need_rows(c_rn2_la, Hdim); need_rows(c_rin_la, Hdim); need_rows(c_md_la, Hdim);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* orw = &c_y[(size_t)m*Hdim];
        // Scalar path: h = rot_left(o_la) OVERWRITES h, then h += res.
        // Stage res through c_rin_la rows (fully rewritten later) so the
        // overwrite doesn't destroy the residual base.
        float* tmp = &c_rin_la[(size_t)m*Hdim];
        float* dst = &c_rn2_la[(size_t)m*Hdim];
        memcpy(tmp, hr, (size_t)Hdim*sizeof(float));
        if(Ro_la) rot_left(orw, *Ro_la, dst);
        else memcpy(dst, orw, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] = dst[i] + tmp[i];
      }
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly.post_ln, &c_rn2_la[(size_t)m*Hdim], Hdim);
      const Rotation* Rg_la = backend->rot_for(ln.gate);
      for(int m=0; m<M; m++){
        const float* r2 = &c_rn2_la[(size_t)m*Hdim];
        float* ri = &c_rin_la[(size_t)m*Hdim];
        if(Rg_la) rot_apply(r2, *Rg_la, ri); else memcpy(ri, r2, (size_t)Hdim*sizeof(float));
      }
      // Fused MLP (device chain on TP=1, else host path below).
      if(!backend->run_mlp_fused(ln.gate, ln.up, ln.down, c_rin_la, c_mg, c_mu, c_md_la, (uint32_t)M)){
      // Wave-batch: gate/up share c_rin_la and are independent — one
      // engine call, one host round-trip. Bit-identical to two
      // run_linear_batch.
      {
        LinearBackend::LinearWaveBatchItem items[2] = {
          {ln.gate, &c_rin_la, &c_mg},
          {ln.up,   &c_rin_la, &c_mu},
        };
        backend->run_linear_wave_batch(items, 2, (uint32_t)M);
      }
      for(int m=0; m<M; m++){
        float* mgr = &c_mg[(size_t)m*cfg.inter];
        const float* mur = &c_mu[(size_t)m*cfg.inter];
        for(int i=0;i<cfg.inter;i++) mgr[i] = silu(mgr[i])*mur[i];
      }
      backend->run_linear_batch(ln.down, c_mg, c_md_la, (uint32_t)M);
      }
      const Rotation* Rd_la = backend->rot_for(ln.down);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* mdr = &c_md_la[(size_t)m*Hdim];
        float* dst = &c_rn2_la[(size_t)m*Hdim];
        if(Rd_la) rot_left(mdr, *Rd_la, dst);
        else memcpy(dst, mdr, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] += dst[i];
        int pos = pos0 + m;
        if(!dump.empty() && (size_t)pos < dump[L].size()/Hdim)
          memcpy(&dump[L][(size_t)pos*Hdim], hr, (size_t)Hdim*sizeof(float));
      }
    }
  }
  // Final norm + lm_head. Default scores the LAST chunk row only (matches
  // the old loop, which kept generated = argmax at pos S-1). When all_out
  // is non-null (speculative verifier, TODO3.3), every row is scored and
  // all_out[m] holds row m's best+logits; out still returns the last row.
  std::vector<float>& fn = req.scratch.fn;
  std::vector<float>& logits = req.scratch.logits;
  const float* head = lm_head ? lm_head : embed;
  auto gpu = backend ? dynamic_cast<GpuBackend*>(backend) : nullptr;
  auto score_row = [&](int m, StepOut& o){
    rmsnorm(&H[(size_t)m*Hdim], final_norm, fn.data(), Hdim);
    int best=-1; float bestv=-1e30f;
    if(gpu){
      const float* W = lm_head ? lm_head : embed;
      std::vector<float> lm = gpu->run_lm_head_f16(fn.data(), W, (uint32_t)cfg.vocab, (uint32_t)cfg.H);
      if((int)lm.size() == cfg.vocab) for(int i=0;i<cfg.vocab;i++) logits[i] = lm[i];
    }else{
      for(int v=0; v<cfg.vocab; v++){
        const float* er = head + (size_t)v*cfg.H;
        float s=0; for(int i=0;i<cfg.H;i++) s+=fn[i]*er[i];
        logits[v]=s;
        if(s>bestv){ bestv=s; best=v; }
      }
    }
    if(lm_head==nullptr || !backend || dynamic_cast<CpuBackend*>(backend) || gpu){
      for(int v=0; v<cfg.vocab; v++){
        if(logits[v]>bestv){ bestv=logits[v]; best=v; }
      }
    }
    o.best = best;
    o.logits = logits;
  };
  if(all_out){
    all_out->resize(M);
    for(int m=0; m<M; m++) score_row(m, (*all_out)[m]);
    out = (*all_out)[M-1];
  } else {
    score_row(M-1, out);
  }
  (void)T;
  return out;
}

// ============================================================================
// Mixed-batch M-row forward (Stage 3.2: iteration scheduler).
// ============================================================================
// One iteration of continuous batching: each live request contributes
// exactly one row (its next token at its own absolute position, against
// its own KV/SSM state). GEMMs run as single M-row launches via
// run_linear_batch; all elementwise math + state updates stay per-row in
// row order. Structure mirrors the batched (use_batch) branches of
// prefill_layer_outer with two changes: (1) per-row positions pos[m]
// instead of contiguous pos0+m, (2) per-row layer state lys[m] instead of
// one shared req. Norm weights are read-only and identical across
// requests, so per-row `ly` matters only at state-mutation sites
// (KV append, conv/SSM recurrence). Transient compute temps reuse
// items[0]'s scratch arena (no live values span rows: each row's result
// is copied to M-staging immediately, same as the chunk path).
// Bit-parity contract: step_batch([{req,tok,pos}]) == step(req,tok,pos)
// up to the known FP16-act batch rounding (bench-gated 2.4e-4 max diff
// on gate_proj); the oracle gate is the end-to-end arbiter. Returns one
// StepOut per row (every row is final-normed + lm_head-scored).
// Row kinds: a decode row (prefill=false) carries its input token in
// `token`; a prefill row (prefill=true) resolves its input from
// req->prompt_ids[pos] at forward time. Both kinds advance the request's
// own KV/SSM state at their own absolute position, so prefill chunks and
// decodes can share one M-row forward (joint iteration mechanic).
// NOTE: no dump/all_out/row_snaps — dump is write-only debug, and the
// scheduler scores every row directly (speculative verify keeps using
// prefill_layer_outer on a single request).
static std::vector<StepOut> step_batch(
                     const std::vector<BatchItem>& items,
                     const std::map<std::string, std::vector<float>>& T,
                     std::vector<float>& H, // [M,H] scratch in/out
                     const float* embed, const float* final_norm,
                     const float* lm_head,
                     LinearBackend* backend){
  std::vector<StepOut> outs;
  const int M = (int)items.size();
  const int Hdim = cfg.H;
  if(M <= 0) return outs;
  outs.resize(M);
  for(int i=0;i<M;i++){
    int tok = items[(size_t)i].prefill
            ? items[(size_t)i].req->prompt_ids[(size_t)items[(size_t)i].pos]
            : items[(size_t)i].token;
    const float* row = embed + (size_t)tok*Hdim;
    memcpy(&H[(size_t)i*Hdim], row, (size_t)Hdim*sizeof(float));
  }
  Request& rar = *items[0].req;
  if(!rar.scratch.sized)
    rar.scratch.ensure(cfg.H, cfg.inter, cfg.vocab,
                     2*cfg.n_heads*cfg.head_dim, cfg.n_kv*cfg.head_dim,
                     cfg.n_heads*cfg.head_dim, cfg.conv_dim, cfg.n_v_heads,
                     cfg.n_v_heads*cfg.v_head_dim, cfg.v_head_dim);
  std::vector<float> c_normed, c_x, c_y;   // [M,H] in, [M,H] rotated, [M,N] out
  std::vector<float> c_mg, c_mu;           // [M,inter] MLP gate/up
  std::vector<float> c_qp;                 // [M,2*n_heads*hd] q_proj out
  std::vector<float> c_k, c_v;             // [M,n_kv*hd]
  auto need_rows = [&](std::vector<float>& v, size_t cols){
    size_t n = (size_t)M * cols;
    if(v.size() != n) v.assign(n, 0.f);
  };
  // Per-row positions + layer states (the two indirections vs the chunk path).
  std::vector<int> pos_of(M);
  for(int m=0; m<M; m++) pos_of[(size_t)m] = items[(size_t)m].pos;
  for(int L=0; L<cfg.layers; L++){
    const LayerNames& ln = g_ln[(size_t)L];
    // Representative layer for read-only weights (norms); state mutations
    // below always go through lys[m].
    Layer& ly0 = items[0].req->layers[(size_t)L];
    static thread_local std::vector<Layer*> lys;
    if((int)lys.size() != M) lys.assign(M, nullptr);
    for(int m=0; m<M; m++) lys[(size_t)m] = &items[(size_t)m].req->layers[(size_t)L];
    // Pre-size each request's KV caches for its own position.
    for(int m=0; m<M; m++){
      if(!lys[(size_t)m]->is_full) continue;
      kv_require_cap(pos_of[(size_t)m]);
      size_t kv_need = (size_t)(pos_of[(size_t)m]+1)*cfg.n_kv*cfg.head_dim;
      if(lys[(size_t)m]->kcache.size() < kv_need) lys[(size_t)m]->kcache.resize(kv_need, 0.f);
      if(lys[(size_t)m]->vcache.size() < kv_need) lys[(size_t)m]->vcache.resize(kv_need, 0.f);
    }
    if(ly0.is_full){
      // ---- Batched full-attention layer (mirrors prefill Phase C) ----
      size_t qrows = (size_t)cfg.n_heads * cfg.head_dim;
      size_t kvrows = (size_t)cfg.n_kv * cfg.head_dim;
      size_t qrows2 = 2 * qrows;
      need_rows(c_normed, Hdim);
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly0.in_ln, &c_normed[(size_t)m*Hdim], Hdim);
      const Rotation* Rq = backend->rot_for(ln.q);
      const Rotation* Rk = backend->rot_for(ln.k);
      const Rotation* Rv = backend->rot_for(ln.v);
      need_rows(c_x, Hdim);
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rq) rot_apply(nr, *Rq, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      need_rows(c_qp, qrows2);
      backend->run_linear_batch(ln.q, c_x, c_qp, (uint32_t)M);
      bool packed_gate = (c_qp.size() >= (size_t)M * qrows2);
      need_rows(c_k, kvrows); need_rows(c_v, kvrows);
      static thread_local std::vector<float> b_qpart, b_gate;
      need_rows(b_qpart, qrows); need_rows(b_gate, qrows);
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rk) rot_apply(nr, *Rk, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_kv_in = c_x;
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rv) rot_apply(nr, *Rv, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_v_in = c_x;
      backend->run_linear_batch(ln.k, c_kv_in, c_k, (uint32_t)M);
      backend->run_linear_batch(ln.v, c_v_in, c_v, (uint32_t)M);
      for(int m=0; m<M; m++){
        int pos = pos_of[(size_t)m];
        Layer& ly = *lys[(size_t)m];
        float* qpr = &b_qpart[(size_t)m*qrows];
        float* gar = &b_gate[(size_t)m*qrows];
        const float* qpr_in = &c_qp[(size_t)m*qrows2];
        for(size_t i=0;i<qrows;i++) gar[i] = 0.f;
        if(packed_gate){
          for(int hh=0; hh<cfg.n_heads; hh++) for(int d=0; d<cfg.head_dim; d++){
            qpr[hh*cfg.head_dim+d] = qpr_in[hh*2*cfg.head_dim+d];
            gar[hh*cfg.head_dim+d] = qpr_in[hh*2*cfg.head_dim+cfg.head_dim+d];
          }
        } else {
          for(size_t i=0; i<qrows; i++) qpr[i] = qpr_in[i];
        }
        for(int hh=0; hh<cfg.n_heads; hh++){
          rmsnorm(&qpr[hh*cfg.head_dim], ly.q_norm_w, &qpr[hh*cfg.head_dim], cfg.head_dim);
          rope(&qpr[hh*cfg.head_dim], pos);
        }
        float* kr = &c_k[(size_t)m*kvrows];
        float* vr = &c_v[(size_t)m*kvrows];
        for(int hh=0; hh<cfg.n_kv; hh++){
          rmsnorm(&kr[hh*cfg.head_dim], ly.k_norm_w, &kr[hh*cfg.head_dim], cfg.head_dim);
          rope(&kr[hh*cfg.head_dim], pos);
        }
        kv_require_cap(pos);
      int base = pos*(cfg.n_kv*cfg.head_dim);
        memcpy(&ly.kcache[base], kr, kvrows*sizeof(float));
        memcpy(&ly.vcache[base], vr, kvrows*sizeof(float));
        kv_append(ly, cfg.n_kv, cfg.head_dim, pos);
      }
      static thread_local std::vector<float> b_attn;
      need_rows(b_attn, qrows);
      {
        std::vector<float>& attn = rar.scratch.attn;
        const int G = cfg.n_heads / cfg.n_kv;
        for(int m=0; m<M; m++){
          int pos = pos_of[(size_t)m];
          Layer& ly = *lys[(size_t)m];
          float* qpr = &b_qpart[(size_t)m*qrows];
          float* gar = &b_gate[(size_t)m*qrows];
          float* atr = &b_attn[(size_t)m*qrows];
          float* gs = rar.scratch.gscores_for(G, pos);
          for(int kvh=0; kvh<cfg.n_kv; kvh++){
            float mx[64]; for(int j=0;j<G;j++) mx[j]=-1e30f;
            for(int t=0;t<=pos;t++){
              const float* kh=kv_row_ptr(ly,true,t,kvh,cfg.n_kv,cfg.head_dim);
              for(int j=0;j<G;j++){
                int hh=kvh*G+j;
                const float* qh=&qpr[hh*cfg.head_dim];
                float s=0; for(int d=0;d<cfg.head_dim;d++) s+=qh[d]*kh[d];
                s*=1.0f/sqrtf((float)cfg.head_dim); gs[j*(pos+1)+t]=s; if(s>mx[j]) mx[j]=s;
              }
            }
            for(int j=0;j<G;j++){
              float* row=&gs[j*(pos+1)];
              float sum=0; for(int t=0;t<=pos;t++){ row[t]=expf(row[t]-mx[j]); sum+=row[t]; }
              for(int t=0;t<=pos;t++) row[t]/=sum;
            }
            for(int j=0;j<G;j++){
              float* oh=&attn[(kvh*G+j)*cfg.head_dim]; for(int d=0;d<cfg.head_dim;d++) oh[d]=0;
            }
            for(int t=0;t<=pos;t++){
              const float* vh=kv_row_ptr(ly,false,t,kvh,cfg.n_kv,cfg.head_dim);
              for(int j=0;j<G;j++){
                float st=gs[j*(pos+1)+t];
                float* oh=&attn[(kvh*G+j)*cfg.head_dim];
                for(int d=0;d<cfg.head_dim;d++) oh[d]+=st*vh[d];
              }
            }
          }
          if (cfg.attn_output_gate && packed_gate)
            for(size_t i=0;i<qrows;i++) attn[i] *= sigmoidf(gar[i]);
          memcpy(atr, attn.data(), qrows*sizeof(float));
        }
      }
      {
        const std::vector<float>* o_x = &b_attn;
        static thread_local std::vector<float> b_o_in;
        if(!packed_gate){
          need_rows(b_o_in, 2*qrows);
          for(int m=0; m<M; m++){
            memcpy(&b_o_in[(size_t)m*2*qrows], &b_attn[(size_t)m*qrows], qrows*sizeof(float));
            memcpy(&b_o_in[(size_t)m*2*qrows+qrows], &b_gate[(size_t)m*qrows], qrows*sizeof(float));
          }
          o_x = &b_o_in;
        }
        need_rows(c_y, Hdim);
        backend->run_linear_batch(ln.o, *o_x, c_y, (uint32_t)M);
      }
      const Rotation* Ro = backend->rot_for(ln.o);
      need_rows(c_mg, cfg.inter); need_rows(c_mu, cfg.inter);
      static thread_local std::vector<float> b_rn2, b_rin, b_md;
      need_rows(b_rn2, Hdim); need_rows(b_rin, Hdim); need_rows(b_md, Hdim);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* orw = &c_y[(size_t)m*Hdim];
        float* dst = &b_rn2[(size_t)m*Hdim];
        if(Ro) rot_left(orw, *Ro, dst);
        else memcpy(dst, orw, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] += dst[i];
      }
      // NOTE (aliasing): b_rn2 doubles as o_proj residual staging (above)
      // and post_ln output (below). The loops are separate, so row m's
      // staging use completes before row m's post_ln overwrite — safe.
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly0.post_ln, &b_rn2[(size_t)m*Hdim], Hdim);
      const Rotation* Rg = backend->rot_for(ln.gate);
      for(int m=0; m<M; m++){
        const float* r2 = &b_rn2[(size_t)m*Hdim];
        float* ri = &b_rin[(size_t)m*Hdim];
        if(Rg) rot_apply(r2, *Rg, ri); else memcpy(ri, r2, (size_t)Hdim*sizeof(float));
      }
      // Fused MLP (device chain on TP=1, else host path below).
      if(!backend->run_mlp_fused(ln.gate, ln.up, ln.down, b_rin, c_mg, c_mu, b_md, (uint32_t)M)){
      // Wave-batch: gate/up share b_rin and are independent — one
      // engine call, one host round-trip. Bit-identical to two
      // run_linear_batch.
      {
        LinearBackend::LinearWaveBatchItem items[2] = {
          {ln.gate, &b_rin, &c_mg},
          {ln.up,   &b_rin, &c_mu},
        };
        backend->run_linear_wave_batch(items, 2, (uint32_t)M);
      }
      for(int m=0; m<M; m++){
        float* mgr = &c_mg[(size_t)m*cfg.inter];
        const float* mur = &c_mu[(size_t)m*cfg.inter];
        for(int i=0;i<cfg.inter;i++) mgr[i] = silu(mgr[i])*mur[i];
      }
      backend->run_linear_batch(ln.down, c_mg, b_md, (uint32_t)M);
      }
      const Rotation* Rd = backend->rot_for(ln.down);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* mdr = &b_md[(size_t)m*Hdim];
        float* dst = &b_rn2[(size_t)m*Hdim];
        if(Rd) rot_left(mdr, *Rd, dst);
        else memcpy(dst, mdr, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] += dst[i];
      }
    } else {
      // ---- Batched linear-attention / SSM layer (mirrors Phase C) ----
      const float* conv_w = ln.conv_w;
      const float* A_log  = ln.A_log;
      const float* dt_bias= ln.dt_bias;
      static thread_local std::vector<float> b_mqkv, b_z, b_b, b_a;
      static thread_local std::vector<float> b_ncore;
      need_rows(c_normed, Hdim);
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly0.in_ln, &c_normed[(size_t)m*Hdim], Hdim);
      const Rotation* Rqkv = backend->rot_for(ln.qkv);
      const Rotation* Rz = backend->rot_for(ln.z);
      const Rotation* Rb = backend->rot_for(ln.b);
      const Rotation* Ra = backend->rot_for(ln.a);
      need_rows(c_x, Hdim);
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rqkv) rot_apply(nr, *Rqkv, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      need_rows(b_mqkv, cfg.conv_dim);
      backend->run_linear_batch(ln.qkv, c_x, b_mqkv, (uint32_t)M);
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rz) rot_apply(nr, *Rz, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_z_in = c_x;
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Rb) rot_apply(nr, *Rb, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      std::vector<float> c_b_in = c_x;
      for(int m=0; m<M; m++){
        const float* nr = &c_normed[(size_t)m*Hdim];
        float* xr = &c_x[(size_t)m*Hdim];
        if(Ra) rot_apply(nr, *Ra, xr); else memcpy(xr, nr, (size_t)Hdim*sizeof(float));
      }
      size_t zvw = (size_t)cfg.n_v_heads * cfg.v_head_dim;
      need_rows(b_z, zvw);
      need_rows(b_b, cfg.n_v_heads);
      need_rows(b_a, cfg.n_v_heads);
      backend->run_linear_batch(ln.z, c_z_in, b_z, (uint32_t)M);
      backend->run_linear_batch(ln.b, c_b_in, b_b, (uint32_t)M);
      backend->run_linear_batch(ln.a, c_x, b_a, (uint32_t)M);
      need_rows(b_ncore, zvw);
      std::vector<float> conv_out(cfg.conv_dim);
      std::vector<float> kv_mem(cfg.v_head_dim), delta(cfg.v_head_dim);
      std::vector<float> qn(cfg.k_head_dim), kn(cfg.k_head_dim);
      std::vector<float> core(zvw), gg(cfg.n_v_heads);
      // Device-side SSM (INTFER_SSM_GPU=1, serve hot path): the conv loop
      // below stores one conv_out copy per row (conv_state still advances
      // on the host in row order — per-request state, order preserved);
      // a single M-row ssm_rows call then replaces the host recurrence.
      // ssm_rows partitions rows itself: decode rows (distinct (req,layer)
      // states) share one ssm_forward launch, same-request prefill runs
      // use ssm_forward_chain each. No snapshots in step_batch, so no
      // mirror sync. ssm_rows false => host loop over stored rows
      // (identical math, same order).
      static thread_local std::vector<SsmRow> b_ssm_rows;
      static thread_local std::vector<float> b_ssm_conv;  // [M*conv_dim]
      bool b_ssm_on = false;
#ifdef INTFER_HAS_GPU_BACKEND
      {
        static const bool ssm_gpu_env =
            (std::getenv("INTFER_SSM_GPU") != nullptr);
        if(ssm_gpu_env) {
          if(auto* gpu_be0 = dynamic_cast<GpuBackend*>(backend))
            b_ssm_on = (gpu_be0->eng != nullptr);
        }
      }
#endif
      if(b_ssm_on) {
        b_ssm_rows.assign(M, SsmRow());
        b_ssm_conv.assign((size_t)M * cfg.conv_dim, 0.f);
      }
      for(int m=0; m<M; m++){
        Layer& ly = *lys[(size_t)m];
        float* mqkvr = &b_mqkv[(size_t)m*cfg.conv_dim];
        for(int c=0;c<cfg.conv_dim;c++){
          float s=0;
          s+=conv_w[c*cfg.conv_k+0]*ly.conv_state[c*3+0];
          s+=conv_w[c*cfg.conv_k+1]*ly.conv_state[c*3+1];
          s+=conv_w[c*cfg.conv_k+2]*ly.conv_state[c*3+2];
          s+=conv_w[c*cfg.conv_k+3]*mqkvr[c];
          conv_out[c]=silu(s);
          ly.conv_state[c*3+0]=ly.conv_state[c*3+1];
          ly.conv_state[c*3+1]=ly.conv_state[c*3+2];
          ly.conv_state[c*3+2]=mqkvr[c];
        }
        if(b_ssm_on) {
          memcpy(&b_ssm_conv[(size_t)m*cfg.conv_dim], conv_out.data(),
                 (size_t)cfg.conv_dim*sizeof(float));
          continue;
        }
        float* query=conv_out.data();
        float* key=conv_out.data()+cfg.k_head_dim*cfg.n_k_heads;
        float* value=conv_out.data()+2*cfg.k_head_dim*cfg.n_k_heads;
        float* zr = &b_z[(size_t)m*zvw];
        float* br = &b_b[(size_t)m*cfg.n_v_heads];
        float* ar = &b_a[(size_t)m*cfg.n_v_heads];
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          float sp=log1pf(expf(ar[hh]+dt_bias[hh])); gg[hh]=-expf(A_log[hh])*sp;
        }
        for(size_t i=0;i<zvw;i++) core[i] = 0.f;
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          int kh = (cfg.n_v_heads > cfg.n_k_heads)
                   ? (int)((long long)hh * cfg.n_k_heads / cfg.n_v_heads)
                   : hh;
          l2norm(&query[kh*cfg.k_head_dim], qn.data(), cfg.k_head_dim);
          l2norm(&key[kh*cfg.k_head_dim],  kn.data(), cfg.k_head_dim);
          float scale=1.0f/sqrtf((float)cfg.k_head_dim);
          for(int d=0;d<cfg.k_head_dim;d++) qn[d]*=scale;
          float g_t=expf(gg[hh]); float beta=sigmoidf(br[hh]);
          float* S=&ly.ssm_state[hh*cfg.k_head_dim*cfg.v_head_dim];
          for(int i=0;i<cfg.k_head_dim*cfg.v_head_dim;i++) S[i]*=g_t;
          for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]=0.f;
          for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]+=S[d*cfg.v_head_dim+c]*kn[d];
          { const float* vh=&value[hh*cfg.v_head_dim];
          for(int c=0;c<cfg.v_head_dim;c++) delta[c]=(vh[c]-kv_mem[c])*beta; }
          for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) S[d*cfg.v_head_dim+c]+=kn[d]*delta[c];
          for(int c=0;c<cfg.v_head_dim;c++){
            float s=0; for(int d=0;d<cfg.k_head_dim;d++) s+=S[d*cfg.v_head_dim+c]*qn[d];
            core[hh*cfg.v_head_dim+c]=s;
          }
        }
        const float* ssm_norm_w = ln.ssm_norm_w;
        float* ncr = &b_ncore[(size_t)m*zvw];
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          const float* xx=&core[hh*cfg.v_head_dim];
          const float* zg=&zr[hh*cfg.v_head_dim];
          float* oo=&ncr[hh*cfg.v_head_dim];
          float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=xx[i]*xx[i];
          float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
          for(int i=0;i<cfg.v_head_dim;i++) oo[i]=(xx[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
        }
      }
      if(b_ssm_on) {
        // Device step over all rows (per-row keys; runs partition inside).
        // Core rows land directly in b_ncore; the norm below reads them.
        for(int m=0; m<M; m++){
          SsmRow& sr = b_ssm_rows[(size_t)m];
          const float* cq = &b_ssm_conv[(size_t)m * cfg.conv_dim];
          sr.req = items[(size_t)m].req;
          sr.mirror = &lys[(size_t)m]->ssm_state;
          sr.query = cq;
          sr.key = cq + cfg.k_head_dim * cfg.n_k_heads;
          sr.value = cq + 2 * cfg.k_head_dim * cfg.n_k_heads;
          sr.b_row = &b_b[(size_t)m * cfg.n_v_heads];
          sr.a_row = &b_a[(size_t)m * cfg.n_v_heads];
          sr.core_out = &b_ncore[(size_t)m * zvw];
        }
        bool ok = false;
#ifdef INTFER_HAS_GPU_BACKEND
        if(auto* gpu_be = dynamic_cast<GpuBackend*>(backend)) {
          if(gpu_be->eng)
            ok = gpu_be->ssm_rows(b_ssm_rows.data(), (uint32_t)M, L,
                                  dt_bias, A_log,
                                  /*sync_mirror=*/false);
        }
#endif
        if(ok) {
          // Norm tail over device cores (mirrors host loop; core_out rows
          // already sit in b_ncore, so read xx from ncr in place —
          // elementwise-safe, no cross terms).
          const float* ssm_norm_w = ln.ssm_norm_w;
          for(int m=0; m<M; m++){
            float* ncr = &b_ncore[(size_t)m*zvw];
            float* zr = &b_z[(size_t)m*zvw];
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              float* xx=&ncr[hh*cfg.v_head_dim];
              const float* zg=&zr[hh*cfg.v_head_dim];
              float* oo=&ncr[hh*cfg.v_head_dim];
              float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=xx[i]*xx[i];
              float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
              for(int i=0;i<cfg.v_head_dim;i++)
                oo[i]=(xx[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
            }
          }
        } else {
          // Device declined: host recurrence over the stored conv rows
          // (same math/order as the env-off loop above).
          const float* ssm_norm_w = ln.ssm_norm_w;
          for(int m=0; m<M; m++){
            Layer& ly = *lys[(size_t)m];
            const float* cq = &b_ssm_conv[(size_t)m * cfg.conv_dim];
            const float* query = cq;
            const float* key = cq + cfg.k_head_dim * cfg.n_k_heads;
            const float* value = cq + 2 * cfg.k_head_dim * cfg.n_k_heads;
            float* zr = &b_z[(size_t)m*zvw];
            float* br = &b_b[(size_t)m*cfg.n_v_heads];
            float* ar = &b_a[(size_t)m*cfg.n_v_heads];
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              float sp=log1pf(expf(ar[hh]+dt_bias[hh])); gg[hh]=-expf(A_log[hh])*sp;
            }
            for(size_t i=0;i<zvw;i++) core[i] = 0.f;
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              int kh = (cfg.n_v_heads > cfg.n_k_heads)
                       ? (int)((long long)hh * cfg.n_k_heads / cfg.n_v_heads)
                       : hh;
              l2norm(&query[kh*cfg.k_head_dim], qn.data(), cfg.k_head_dim);
              l2norm(&key[kh*cfg.k_head_dim],  kn.data(), cfg.k_head_dim);
              float scale=1.0f/sqrtf((float)cfg.k_head_dim);
              for(int d=0;d<cfg.k_head_dim;d++) qn[d]*=scale;
              float g_t=expf(gg[hh]); float beta=sigmoidf(br[hh]);
              float* S=&ly.ssm_state[hh*cfg.k_head_dim*cfg.v_head_dim];
              for(int i=0;i<cfg.k_head_dim*cfg.v_head_dim;i++) S[i]*=g_t;
              for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]=0.f;
              for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]+=S[d*cfg.v_head_dim+c]*kn[d];
              { const float* vh=&value[hh*cfg.v_head_dim];
              for(int c=0;c<cfg.v_head_dim;c++) delta[c]=(vh[c]-kv_mem[c])*beta; }
              for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) S[d*cfg.v_head_dim+c]+=kn[d]*delta[c];
              for(int c=0;c<cfg.v_head_dim;c++){
                float s=0; for(int d=0;d<cfg.k_head_dim;d++) s+=S[d*cfg.v_head_dim+c]*qn[d];
                core[hh*cfg.v_head_dim+c]=s;
              }
            }
            float* ncr = &b_ncore[(size_t)m*zvw];
            for(int hh=0; hh<cfg.n_v_heads; hh++){
              const float* xx=&core[hh*cfg.v_head_dim];
              const float* zg=&zr[hh*cfg.v_head_dim];
              float* oo=&ncr[hh*cfg.v_head_dim];
              float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=xx[i]*xx[i];
              float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
              for(int i=0;i<cfg.v_head_dim;i++) oo[i]=(xx[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
            }
          }
        }
      }
      need_rows(c_y, Hdim);
      backend->run_linear_batch(ln.la_out, b_ncore, c_y, (uint32_t)M);
      const Rotation* Ro_la = backend->rot_for(ln.la_out);
      need_rows(c_mg, cfg.inter); need_rows(c_mu, cfg.inter);
      static thread_local std::vector<float> b_rn2_la, b_rin_la, b_md_la;
      need_rows(b_rn2_la, Hdim); need_rows(b_rin_la, Hdim); need_rows(b_md_la, Hdim);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* orw = &c_y[(size_t)m*Hdim];
        float* dst = &b_rn2_la[(size_t)m*Hdim];
        // Residual add BEFORE the post_ln overwrite below: rot_left into
        // dst, add the incoming hr (still pre-add here), write back to hr.
        // Matches the scalar path (rot_left into h, then h += res). The
        // post_ln output reuses dst only after hr is final for this row.
        if(Ro_la) rot_left(orw, *Ro_la, dst);
        else memcpy(dst, orw, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] = dst[i] + hr[i];
        rmsnorm(hr, ly0.post_ln, dst, Hdim);
      }
      const Rotation* Rg_la = backend->rot_for(ln.gate);
      for(int m=0; m<M; m++){
        const float* r2 = &b_rn2_la[(size_t)m*Hdim];
        float* ri = &b_rin_la[(size_t)m*Hdim];
        if(Rg_la) rot_apply(r2, *Rg_la, ri); else memcpy(ri, r2, (size_t)Hdim*sizeof(float));
      }
      backend->run_linear_batch(ln.gate, b_rin_la, c_mg, (uint32_t)M);
      backend->run_linear_batch(ln.up,   b_rin_la, c_mu, (uint32_t)M);
      for(int m=0; m<M; m++){
        float* mgr = &c_mg[(size_t)m*cfg.inter];
        const float* mur = &c_mu[(size_t)m*cfg.inter];
        for(int i=0;i<cfg.inter;i++) mgr[i] = silu(mgr[i])*mur[i];
      }
      backend->run_linear_batch(ln.down, c_mg, b_md_la, (uint32_t)M);
      const Rotation* Rd_la = backend->rot_for(ln.down);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* mdr = &b_md_la[(size_t)m*Hdim];
        float* dst = &b_rn2_la[(size_t)m*Hdim];
        if(Rd_la) rot_left(mdr, *Rd_la, dst);
        else memcpy(dst, mdr, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] += dst[i];
      }
    }
  }
  // Final norm + lm_head for EVERY row (scheduler needs all logits).
  // GPU: one M-row batched lm_head (run_lm_head_batch) instead of M scalar
  // calls — one launch per device tile instead of M (Lever 1). Rows are
  // independent GEMM rows (argmax-gated vs sequential; INTFER_LMHEAD_BATCH=0
  // restores the old per-row path for bisection).
  std::vector<float>& fn = rar.scratch.fn;
  std::vector<float>& logits = rar.scratch.logits;
  const float* head = lm_head ? lm_head : embed;
  auto gpu = backend ? dynamic_cast<GpuBackend*>(backend) : nullptr;
  static const bool lm_batch_off = (std::getenv("INTFER_LMHEAD_BATCH") &&
                                    std::getenv("INTFER_LMHEAD_BATCH")[0] == '0');
  std::vector<float> lm_all;
  if(gpu && !lm_batch_off && M > 1){
    std::vector<float> fn_all((size_t)M * Hdim);
    for(int m = 0; m < M; ++m)
      rmsnorm(&H[(size_t)m*Hdim], final_norm, fn_all.data() + (size_t)m * Hdim, Hdim);
    const float* W = lm_head ? lm_head : embed;
    lm_all = gpu->run_lm_head_batch(fn_all.data(), W, (uint32_t)M,
                                    (uint32_t)cfg.vocab, (uint32_t)cfg.H);
  }
  for(int m=0; m<M; m++){
    rmsnorm(&H[(size_t)m*Hdim], final_norm, fn.data(), Hdim);
    int best=-1; float bestv=-1e30f;
    if(gpu){
      if(!lm_all.empty() && (int)lm_all.size() == M * cfg.vocab){
        for(int i=0;i<cfg.vocab;i++) logits[i] = lm_all[(size_t)m * cfg.vocab + i];
      } else {
        const float* W = lm_head ? lm_head : embed;
        std::vector<float> lm = gpu->run_lm_head_f16(fn.data(), W, (uint32_t)cfg.vocab, (uint32_t)cfg.H);
        if((int)lm.size() == cfg.vocab) for(int i=0;i<cfg.vocab;i++) logits[i] = lm[i];
      }
    }else{
      for(int v=0; v<cfg.vocab; v++){
        const float* er = head + (size_t)v*cfg.H;
        float s=0; for(int i=0;i<cfg.H;i++) s+=fn[i]*er[i];
        logits[v]=s;
        if(s>bestv){ bestv=s; best=v; }
      }
    }
    if(lm_head==nullptr || !backend || dynamic_cast<CpuBackend*>(backend) || gpu){
      for(int v=0; v<cfg.vocab; v++){
        if(logits[v]>bestv){ bestv=logits[v]; best=v; }
      }
    }
    outs[(size_t)m].best = best;
    outs[(size_t)m].logits = logits;
  }
  (void)T;
  return outs;
}

// ============================================================================
// Scheduler runtime (Stage 3.3): serve-loop building blocks.
// ============================================================================
// Forward declaration (sampling section sits below the scheduler helpers).
static int sample(const std::vector<float>& logits, int top_k, float top_p,
                  float temperature, std::mt19937& rng);
// The 3.2 CLI batch path below is factored through these so --serve shares
// the exact iteration mechanics (same prefill, same step_batch decode, same
// finish rules). Emission differs: CLI buffers per-request text; the server
// streams JSON lines per token.
//
// SchedCtx bundles the model-side inputs every iteration needs (avoids a
// 12-arg helper signature; all pointers outlive the serve loop).
struct SchedCtx {
  const std::map<std::string, std::vector<float>>* T = nullptr;
  const float* embed = nullptr;
  const float* final_norm = nullptr;
  const float* lm_head = nullptr;
  LinearBackend* backend = nullptr;
  const std::string* detok_py = nullptr;
  const std::string* tokenizer_json = nullptr;
  const std::string* tokenizer_py = nullptr; // encode_text script (server)
  bool print_ids = false;
};
// Prefill one request into its own KV/SSM state (sequential; chunked by
// default, INTFER_PREFILL_CHUNK=0 forces the step loop). Appends the first
// generated id; sets
// token/pos. dump is the request's private record buffer (may be empty to
// skip recording).
static void sched_prefill(Request& q, SchedCtx& c, std::vector<std::vector<float>>& dump){
  int Sr = (int)q.prompt_ids.size();
  static const char* chunk_env = std::getenv("INTFER_PREFILL_CHUNK");
  bool chunked = (chunk_env == nullptr || std::string(chunk_env) != "0");
  int chunk_M = 128;
  if(const char* me = std::getenv("INTFER_PREFILL_CHUNK_M")){
    int v = std::atoi(me); if(v >= 1 && v <= 1024) chunk_M = v;
  }
  std::vector<float> Hbuf;
  if(chunked && Sr > 1){
    for(int c0 = 0; c0 < Sr; c0 += chunk_M){
      int cm = std::min(chunk_M, Sr - c0);
      std::vector<int> toks(q.prompt_ids.begin() + c0, q.prompt_ids.begin() + c0 + cm);
      Hbuf.assign((size_t)cm * cfg.H, 0.f);
      StepOut o = prefill_layer_outer(q, toks, c0, *c.T, Hbuf, c.embed, c.final_norm,
                                      c.lm_head, c.backend, dump);
      if(c0 + cm == Sr) q.generated.push_back(o.best);
    }
  } else {
    for(int pos=0; pos<Sr; pos++){
      StepOut o = step(q, q.prompt_ids[pos], pos, *c.T, c.embed, c.final_norm, c.lm_head,
                       c.backend, !dump.empty(), dump);
      if(pos==Sr-1) q.generated.push_back(o.best);
    }
  }
  q.token = q.generated.back();
  q.pos = Sr; // next forward position (prefill filled [0,Sr))
}
// One decode iteration over live requests: build the batch, run step_batch,
// sample/advance each row. Returns the batch width M (0 = nothing live).
// Per-row emission (buffer vs stream) is the caller's emit hook.
static int sched_iteration(std::vector<Request*>& live, SchedCtx& c,
                           std::vector<float>& Hb,
                           const std::function<void(Request&, size_t, int)>& emit,
                           int step_i){
  std::vector<BatchItem> items; std::vector<size_t> ridx;
  for(size_t i=0; i<live.size(); i++){
    Request& q = *live[i];
    if(q.finished) continue;
    if((int)q.generated.size() >= q.max_new + 1){ q.finished = true; continue; }
    items.push_back(BatchItem{&q, q.token, q.pos});
    ridx.push_back(i);
  }
  if(items.empty()) return 0;
  int M = (int)items.size();
  Hb.assign((size_t)M * cfg.H, 0.f);
  std::vector<StepOut> outs = step_batch(items, *c.T, Hb, c.embed, c.final_norm,
                                         c.lm_head, c.backend);
  for(int m=0; m<M; m++){
    Request& q = *items[(size_t)m].req;
    bool qgreedy = (q.temperature <= 0.f && q.top_k==0 && q.top_p<=0.f);
    int next = qgreedy ? outs[(size_t)m].best
                       : sample(outs[(size_t)m].logits, q.top_k, q.top_p,
                                q.temperature, q.rng);
    q.generated.push_back(next);
    q.pos++;
    emit(q, ridx[(size_t)m], next);
    q.token = next;
    if(next == 248044 || next == 248046) q.finished = true;
    if((int)q.generated.size() >= q.max_new + 1) q.finished = true;
  }
  return M;
}
// One JOINT iteration (Stage 3.3+ improvement C): decode rows for all live
// requests plus up to `chunk_M` prefill rows for the prefill-queue head,
// in a SINGLE step_batch forward. Batch layout: [D decode rows | P prefill
// rows] (outs index back into ridx/pidx). Prefill rows append KV/SSM state
// at their absolute prompt positions; intermediate rows only advance state
// (their logits are discarded — only the chunk-final row predicts the next
// token). A filled request joins live exactly as sched_prefill would set
// it (first generated id + pos=Sr); decode rows advance/sample/emit as in
// sched_iteration. Returns decode-row count (0 = nothing decoded, -1 =
// joint batch declined: caller runs the split path for this iteration).
// Identity: same per-row op sequence as the split path (one chunk forward
// + one decode forward), only the GEMM rows are fused — so greedy ids must
// match the split path exactly (gated).
static int sched_iteration_joint(std::vector<Request*>& live,
                                 std::deque<Request*>& prefill_q,
                                 int chunk_M, SchedCtx& c,
                                 std::vector<float>& Hb,
                                 const std::function<void(Request&, size_t, int)>& emit,
                                 int step_i){
  std::vector<BatchItem> items; std::vector<size_t> ridx;
  static const bool trace_joint = (std::getenv("INTFER_TRACE_JOINT") != nullptr);
  if(trace_joint){
    std::fprintf(stderr, "[joint] ENTER live=%zu pq=%zu\n", live.size(), prefill_q.size());
    for(size_t i=0;i<live.size();i++)
      std::fprintf(stderr, "  live[%zu] id=%ld gen=%zu max=%d fin=%d\n",
                   i, live[i]->id, live[i]->generated.size(),
                   live[i]->max_new, (int)live[i]->finished);
    std::fflush(stderr);
  }
  for(size_t i=0; i<live.size(); i++){
    Request& q = *live[i];
    if(q.finished) continue;
    if((int)q.generated.size() >= q.max_new + 1){ q.finished = true; continue; }
    BatchItem b{&q, q.token, q.pos, false};
    items.push_back(b);
    ridx.push_back(i);
  }
  int D = (int)items.size();
  // Prefill rows: up to chunk_M unconsumed prompt tokens of the current
  // rotation target (prefill_q front; the caller rotates the queue so
  // consecutive iterations serve consecutive requests — round-robin
  // fairness, no head-of-line stall. Per-request state + absolute
  // positions make cross-request fill order output-identical).
  // Chunk width cap: the M-row batched-GEMM chunk path (INTFER_BATCH_GEMM=1,
  // M>1) is known-divergent on 27B/GPU (pre-existing: reproduces on the
  // pristine tree without this change), so joint batches carry at most ONE
  // prefill row under BATCH_GEMM and the caller falls back to the split
  // path for wider chunks. The CPU step_batch path is fixed for
  // same-request multi-row chunks (rot_for fallback now resolves the exact
  // per-slot rotation, so cold == warm and all rows agree with the split
  // path), hence no P cap outside BATCH_GEMM.
  Request* pq = prefill_q.empty() ? nullptr : prefill_q.front();
  // Same default-ON semantics as the chunk path (batch_gemm above):
  // INTFER_BATCH_GEMM=0 forces per-row. The wide-chunk cap below must see
  // M-row as active whenever the chunk path uses it, or the 27B/GPU
  // joint-batch guard silently disarms.
  static const char* bg_env = std::getenv("INTFER_BATCH_GEMM");
  static const bool batch_gemm_env =
      (bg_env == nullptr || std::string(bg_env) != "0");
  static const bool rr_off = (std::getenv("INTFER_PREFILL_RR") != nullptr &&
                              std::string(std::getenv("INTFER_PREFILL_RR")) == "0");
  int P = 0;
  if(pq){
    int Sr = (int)pq->prompt_ids.size();
    P = std::min(chunk_M, Sr - pq->fill);
    // Wide same-request chunks ride the joint forward except under the
    // GPU batched-GEMM path (known-divergent there, pre-existing). CPU
    // batch override is fixed (bit-identical single-fork rows), so no cap.
    bool wide_ok = !(batch_gemm_env && c.backend && c.backend->is_distributed());
    if(P > 1 && !wide_ok) return -1; // caller runs the split path (same mechanics)
    for(int k=0; k<P; k++){
      BatchItem b{pq, -1, pq->fill + k, true};
      items.push_back(b);
    }
  }
  if(items.empty()) return 0;
  int M = (int)items.size();
  Hb.assign((size_t)M * cfg.H, 0.f);
  if(trace_joint){
    std::fprintf(stderr, "[joint] M=%d D=%d P=%d rows:", M, D, M-D);
    for(auto& b : items)
      std::fprintf(stderr, " (id=%ld,%s,pos=%d)", b.req->id,
                   b.prefill ? "pre" : "dec", b.pos);
    std::fprintf(stderr, "\n"); std::fflush(stderr);
  }
  std::vector<StepOut> outs = step_batch(items, *c.T, Hb, c.embed, c.final_norm,
                                         c.lm_head, c.backend);
  // Decode rows first (row order == live order; separate loop so a filled
  // prefill request joining live below cannot disturb row indexing).
  for(int m=0; m<D; m++){
    Request& q = *items[(size_t)m].req;
    bool qgreedy = (q.temperature <= 0.f && q.top_k==0 && q.top_p<=0.f);
    int next = qgreedy ? outs[(size_t)m].best
                       : sample(outs[(size_t)m].logits, q.top_k, q.top_p,
                                q.temperature, q.rng);
    q.generated.push_back(next);
    q.pos++;
    emit(q, ridx[(size_t)m], next);
    q.token = next;
    if(next == 248044 || next == 248046) q.finished = true;
    if((int)q.generated.size() >= q.max_new + 1) q.finished = true;
  }
  // Prefill rows: state already advanced by the forward; only the
  // chunk-final row's prediction matters. A filled request joins live.
  // Unfilled queue heads rotate to the back (round-robin fairness) unless
  // disabled; a filled request leaves via pop_front above. Rotation also
  // runs on the decline path (BATCH_GEMM P>1 early return above skips it —
  // the caller rotates there instead; see split_chunk).
  if(pq){
    int Sr = (int)pq->prompt_ids.size();
    pq->fill += P;
    if(pq->fill == Sr){
      int best = outs[(size_t)(D + P - 1)].best;
      pq->generated.push_back(best);
      pq->token = best;
      pq->pos = Sr;
      if(c.print_ids)
        std::fprintf(stderr, "  [serve id=%ld prefill] id=%d\n", pq->id, pq->token);
      live.push_back(pq);
      prefill_q.pop_front();
    } else if(!rr_off && prefill_q.size() > 1){
      prefill_q.pop_front();
      prefill_q.push_back(pq);
    }
  }
  (void)step_i;
  return D;
}
// Minimal JSON-string escaper for server lines (prompts/echo only; token
// pieces go out as ids + BpDecoder text is emission-side decoded).
static std::string json_escape(const std::string& s){
  std::string o;
  for(char ch : s){
    switch(ch){
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if((unsigned char)ch < 0x20){ char b[8]; snprintf(b, sizeof(b), "\\u%04x", ch); o += b; }
        else o += ch;
    }
  }
  return o;
}
// Per-line serve options (all optional; sampling keys fall back to the CLI
// globals passed as dflt_*).
struct ServeOpts {
  int max_new = 64;
  long id = -1; bool has_id = false;
  float temperature = 0.f; int top_k = 0; float top_p = 0.f;
  uint32_t seed = 0; bool has_seed = false;
};
// Last occurrence of "k" outside the prompt-value span [vs,ve) whose closing
// quote is followed by ':' (a real key, not prose inside the prompt text).
static size_t find_json_key(const std::string& line, const char* k,
                            size_t vs, size_t ve){
  std::string q = std::string("\"") + k + "\"";
  size_t best = std::string::npos, p = 0;
  while((p = line.find(q, p)) != std::string::npos){
    bool inside = (p > vs && p < ve);
    size_t e = p + q.size();
    while(e < line.size() && (line[e]==' '||line[e]=='\t')) e++;
    if(!inside && e < line.size() && line[e]==':') best = p;
    p++;
  }
  return best;
}
static const char* json_val_at(const std::string& line, size_t kp){
  size_t c2 = line.find(':', kp);
  if(c2 == std::string::npos) return nullptr;
  return line.c_str()+c2+1;
}
// Parse {"prompt":"...","max_new":N,"id":K,"temperature":T,"top_k":K,
//        "top_p":P,"seed":S} (all optional but prompt).
// Numeric keys are matched outside the prompt string, so prompt text
// containing e.g. "max_new" or "id" cannot spoof overrides (the old search
// scanned the whole line; "id" even preferred the last occurrence).
// Returns false on malformed lines (caller emits {"error":...}).
static bool parse_serve_line(const std::string& line, std::string& prompt,
                             ServeOpts& o, int dflt_n,
                             float dflt_t, int dflt_k, float dflt_p){
  prompt.clear();
  o.max_new = dflt_n; o.id = -1; o.has_id = false;
  o.temperature = dflt_t; o.top_k = dflt_k; o.top_p = dflt_p;
  o.seed = 0; o.has_seed = false;
  size_t pp = line.find("\"prompt\"");
  if(pp == std::string::npos) return false;
  size_t colon = line.find(':', pp);
  if(colon == std::string::npos) return false;
  size_t q0 = line.find('"', colon);
  if(q0 == std::string::npos) return false;
  // Decode \" \\ \n \r \t \uXXXX escapes (prompts may carry quotes).
  std::string raw;
  size_t pe = line.size(); // prompt closing-quote position (key search floor)
  for(size_t i=q0+1; i<line.size(); i++){
    char ch = line[i];
    if(ch == '\\' && i+1 < line.size()){
      char e = line[++i];
      if(e=='"') raw += '"'; else if(e=='\\') raw += '\\';
      else if(e=='n') raw += '\n'; else if(e=='r') raw += '\r';
      else if(e=='t') raw += '\t';
      else if(e=='u' && i+4 < line.size()){
        unsigned v = 0;
        for(int k=1;k<=4;k++){ char h=line[i+k];
          v <<= 4;
          if(h>='0'&&h<='9') v += h-'0';
          else if(h>='a'&&h<='f') v += h-'a'+10;
          else if(h>='A'&&h<='F') v += h-'A'+10;
        }
        i += 4;
        if(v < 0x80) raw += (char)v;
        else if(v < 0x800){ raw += (char)(0xC0|(v>>6)); raw += (char)(0x80|(v&0x3F)); }
        else { raw += (char)(0xE0|(v>>12)); raw += (char)(0x80|((v>>6)&0x3F)); raw += (char)(0x80|(v&0x3F)); }
      }
      else raw += e;
    }
    else if(ch == '"'){ pe = i; break; }
    else raw += ch;
  }
  prompt = raw;
  size_t ve = pe; // numeric keys must sit outside the prompt string
  size_t pn = find_json_key(line, "max_new", q0, ve);
  if(pn != std::string::npos){
    if(const char* v = json_val_at(line, pn)) o.max_new = std::atoi(v);
  }
  size_t pil = find_json_key(line, "id", q0, ve);
  if(pil != std::string::npos){
    if(const char* v = json_val_at(line, pil)){ o.id = std::atol(v); o.has_id = true; }
  }
  size_t pt = find_json_key(line, "temperature", q0, ve);
  if(pt != std::string::npos){
    if(const char* v = json_val_at(line, pt)) o.temperature = (float)std::atof(v);
  }
  size_t pk = find_json_key(line, "top_k", q0, ve);
  if(pk != std::string::npos){
    if(const char* v = json_val_at(line, pk)) o.top_k = std::atoi(v);
  }
  size_t pp2 = find_json_key(line, "top_p", q0, ve);
  if(pp2 != std::string::npos){
    if(const char* v = json_val_at(line, pp2)) o.top_p = (float)std::atof(v);
  }
  size_t ps = find_json_key(line, "seed", q0, ve);
  if(ps != std::string::npos){
    if(const char* v = json_val_at(line, ps)){
      o.seed = (uint32_t)std::atol(v); o.has_seed = true;
    }
  }
  if(o.max_new < 1) o.max_new = 1;
  if(o.max_new > 4096) o.max_new = 4096;
  if(o.top_k < 0) o.top_k = 0;
  if(o.top_p < 0.f) o.top_p = 0.f; if(o.top_p > 1.f) o.top_p = 1.f;
  return true;
}

// ============================================================================
// Sampling
// ============================================================================
static int sample(const std::vector<float>& logits, int top_k, float top_p,
                  float temperature, std::mt19937& rng){
  if(temperature <= 0.f) temperature = 1e-5f;
  float mx = -1e30f;
  for(float v : logits) if(v>mx) mx=v;
  // Reused across calls (single-threaded CLI); sized to vocab once.
  static std::vector<float> probs;
  if(probs.size() != logits.size()) probs.assign(logits.size(), 0.f);
  float sum = 0.f;
  for(size_t i=0;i<logits.size();i++){
    probs[i] = expf((logits[i]-mx)/temperature);
    sum += probs[i];
  }
  for(float& p : probs) p /= sum;
  if(top_k > 0 && (int)probs.size() > top_k){
    std::vector<float> sorted = probs;
    std::nth_element(sorted.begin(), sorted.end()-top_k, sorted.end());
    float kth = sorted[sorted.size()-top_k];
    for(float& p : probs) if(p < kth) p = 0.f;
    sum = 0.f; for(float p : probs) sum += p; for(float& p : probs) p /= sum;
  }
  if(top_p > 0.f && top_p < 1.f){
    std::vector<int> idx(probs.size());
    for(size_t i=0;i<probs.size();i++) idx[i]=(int)i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return probs[a] > probs[b]; });
    float cum = 0.f; int cutoff = (int)probs.size();
    for(size_t i=0;i<idx.size();i++){
      cum += probs[idx[i]];
      if(cum >= top_p){ cutoff = (int)i+1; break; }
    }
    for(size_t i=cutoff;i<idx.size();i++) probs[idx[i]] = 0.f;
    sum = 0.f; for(float p : probs) sum += p; for(float& p : probs) p /= sum;
  }
  std::uniform_real_distribution<float> U(0.f, 1.f);
  float r = U(rng);
  float acc = 0.f;
  for(size_t i=0;i<probs.size();i++){
    acc += probs[i];
    if(r <= acc) return (int)i;
  }
  return (int)(probs.size()-1);
}

// ============================================================================
// I/O helpers
// ============================================================================
static bool read_blob(const std::string& p, std::vector<uint8_t>& out){
  FILE* f=fopen(p.c_str(),"rb"); if(!f) return false;
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); out.resize(n);
  size_t r=fread(out.data(),1,n,f); fclose(f); return r==(size_t)n;
}

// mmap-based loading: returns a pointer to the mapped file. The caller must
// call munmap on the returned pointer (cast to void*) when done.
// On failure, returns nullptr.
struct MmapBlob {
  void*  base = nullptr;
  size_t size = 0;
};
static MmapBlob mmap_blob(const std::string& p) {
  MmapBlob b;
  int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) return b;
  struct stat st; if (fstat(fd, &st) != 0) { ::close(fd); return b; }
  b.size = (size_t)st.st_size;
  if (b.size == 0) { ::close(fd); return b; }
  void* m = mmap(nullptr, b.size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) { b.size = 0; return b; }
  b.base = m;
  madvise(m, b.size, MADV_SEQUENTIAL);
  return b;
}
static void munmap_blob(MmapBlob& b) {
  if (b.base && b.size) munmap(b.base, b.size);
  b.base = nullptr; b.size = 0;
}
static bool file_exists(const std::string& p){
  struct stat st; return stat(p.c_str(), &st)==0;
}

// ============================================================================
// In-process BPE detokenizer (no python subprocess per token)
// ============================================================================
// Qwen3.5/3.8 vocabs are pure GPT-2 byte-images: every vocab string consists
// solely of the 256 byte-image codepoints (printable ASCII/latin-1 kept as-is,
// other bytes mapped to U+0100..U+0143). Verified over all 248044 entries of
// both tokenizers. Decode = id->string lookup + codepoint->byte reversal.
// Parsed once from tokenizer.json ("model"."vocab" object + "added_tokens"
// array); raw UTF-8 with only \" and \\ escapes inside the vocab region.
// Byte-identical to scripts/bpe_detokenize.py for mergeable ids; added-token
// ids decode to their content (old script emitted empty — only affects text
// rendering of special tokens, never the EOS break which compares ids).
struct BpDecoder {
  std::vector<std::string> bytes_of; // bytes_of[id] = raw bytes
  bool ok = false;

  // GPT-2 byte<->unicode maps (identical to the python helper).
  static void byte_image_maps(unsigned char img2byte[324], uint32_t byte2img[256]){
    // Printable bytes map to themselves; non-printables (in byte order)
    // map to U+0100... in sequence.
    bool printable[256] = {};
    for(int b=33;b<=126;b++) printable[b]=true;
    for(int b=161;b<=172;b++) printable[b]=true;
    for(int b=174;b<=255;b++) printable[b]=true;
    for(int b=0;b<256;b++)
      if(printable[b]){ img2byte[b]= (unsigned char)b; byte2img[b]=(uint32_t)b; }
    int n = 0;
    for(int b=0;b<256;b++)
      if(!printable[b]){ img2byte[256+n]= (unsigned char)b; byte2img[b]=256+n; n++; }
  }

  // Decode one UTF-8 codepoint at p; returns bytes consumed (0 on error).
  static int utf8_decode(const char* p, const char* end, uint32_t* cp){
    unsigned char c = (unsigned char)*p;
    if(c < 0x80){ *cp = c; return 1; }
    if((c & 0xE0) == 0xC0){
      if(p+1 >= end) return 0;
      *cp = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)p[1] & 0x3F);
      return 2;
    }
    if((c & 0xF0) == 0xE0){
      if(p+2 >= end) return 0;
      *cp = ((uint32_t)(c & 0x0F) << 12) | (((unsigned char)p[1] & 0x3F) << 6)
          | ((unsigned char)p[2] & 0x3F);
      return 3;
    }
    if((c & 0xF8) == 0xF0){
      if(p+3 >= end) return 0;
      *cp = ((uint32_t)(c & 0x07) << 18) | (((unsigned char)p[1] & 0x3F) << 12)
          | (((unsigned char)p[2] & 0x3F) << 6) | ((unsigned char)p[3] & 0x3F);
      return 4;
    }
    return 0;
  }

  // token string (JSON-unescaped, UTF-8) -> raw bytes via byte-image map.
  static std::string token_to_bytes(const std::string& tok,
                                    const unsigned char img2byte[324],
                                    const uint32_t byte2img[256]){
    (void)byte2img;
    std::string out; out.reserve(tok.size());
    const char* p = tok.data(); const char* end = p + tok.size();
    while(p < end){
      uint32_t cp = 0; int n = utf8_decode(p, end, &cp);
      if(n <= 0) break; // malformed: stop (matches old script's best-effort)
      if(cp < 256)
        out.push_back((char)img2byte[cp]);
      else if(cp >= 256 && cp < 256+68)
        out.push_back((char)img2byte[cp]);
      else {
        // Non-byte-image char (never occurs in audited vocabs): re-encode
        // as UTF-8 like the old script's ch.encode("utf-8") fallback.
        if(cp < 0x80) out.push_back((char)cp);
        else if(cp < 0x800){
          out.push_back((char)(0xC0 | (cp >> 6)));
          out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if(cp < 0x10000){
          out.push_back((char)(0xE0 | (cp >> 12)));
          out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
          out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
          out.push_back((char)(0xF0 | (cp >> 18)));
          out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
          out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
          out.push_back((char)(0x80 | (cp & 0x3F)));
        }
      }
      p += n;
    }
    return out;
  }

  bool load(const std::string& tok_json_path){
    std::vector<uint8_t> blob;
    if(!read_blob(tok_json_path, blob)) return false;
    blob.push_back(0); // null-terminate for strstr/strchr/strtol
    const char* base = (const char*)blob.data();
    size_t size = blob.size();
    // Find "model" object then its "vocab" object.
    const char* p = strstr(base, "\"vocab\"");
    if(!p) return false;
    p = strchr(p, '{');
    if(!p) return false;
    p++;
    unsigned char img2byte[324]; uint32_t byte2img[256];
    byte_image_maps(img2byte, byte2img);
    std::vector<std::pair<std::string,int>> pairs;
    pairs.reserve(250000);
    auto skip_ws = [&](){
      while(p < base+size && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) p++;
    };
    while(p < base+size){
      skip_ws();
      if(p >= base+size) break;
      if(*p == '}') break;
      if(*p != '"') return false;
      p++; // opening quote of key
      std::string key;
      bool esc = false;
      while(p < base+size){
        char c = *p++;
        if(esc){
          if(c=='"'||c=='\\'||c=='/') key.push_back(c);
          else if(c=='n') key.push_back('\n');
          else if(c=='t') key.push_back('\t');
          else if(c=='u'){
            // \uXXXX (never occurs in audited vocab regions; support BMP)
            if(p+4 > base+size) return false;
            unsigned v = 0;
            for(int i=0;i<4;i++){
              char h = *p++;
              v <<= 4;
              if(h>='0'&&h<='9') v |= (unsigned)(h-'0');
              else if(h>='a'&&h<='f') v |= (unsigned)(h-'a'+10);
              else if(h>='A'&&h<='F') v |= (unsigned)(h-'A'+10);
              else return false;
            }
            if(v < 0x80) key.push_back((char)v);
            else if(v < 0x800){
              key.push_back((char)(0xC0|(v>>6)));
              key.push_back((char)(0x80|(v&0x3F)));
            } else {
              key.push_back((char)(0xE0|(v>>12)));
              key.push_back((char)(0x80|((v>>6)&0x3F)));
              key.push_back((char)(0x80|(v&0x3F)));
            }
          }
          else { key.push_back(c); } // best-effort
          esc = false;
        } else if(c == '\\'){ esc = true; }
        else if(c == '"'){ break; }
        else key.push_back(c);
      }
      skip_ws();
      if(p >= base+size || *p != ':') return false;
      p++;
      skip_ws();
      char* e = nullptr;
      long id = strtol(p, &e, 10);
      if(e == p || id < 0) return false;
      p = e;
      pairs.emplace_back(std::move(key), (int)id);
      skip_ws();
      if(p < base+size && *p == ','){ p++; continue; }
      if(p < base+size && *p == '}') break;
    }
    size_t vmax = 0;
    for(auto& kv : pairs) if((size_t)kv.second > vmax) vmax = kv.second;
    bytes_of.assign(vmax+1, std::string());
    for(auto& kv : pairs) bytes_of[kv.second] = token_to_bytes(kv.first, img2byte, byte2img);
    // Added tokens: decode to their UTF-8 content.
    const char* ap = strstr(base, "\"added_tokens\"");
    if(ap){
      const char* arr = strchr(ap, '[');
      if(arr){
        // Bounded scan: find the array's closing ']' with string awareness,
        // then parse each top-level {...} object strictly within bounds.
        // (Unbounded strstr past the array would match "id" vocab keys.)
        const char* ae = nullptr;
        {
          bool instr = false, esc = false; int depth = 0;
          for(const char* q = arr; q < base+size; q++){
            char c = *q;
            if(instr){
              if(esc) esc = false;
              else if(c == '\\') esc = true;
              else if(c == '"') instr = false;
            } else {
              if(c == '"') instr = true;
              else if(c == '[') depth++;
              else if(c == ']'){ if(--depth == 0){ ae = q; break; } }
            }
          }
        }
        if(ae){
          auto find_lit = [](const char* lo, const char* hi,
                             const char* lit) -> const char* {
            size_t n = strlen(lit);
            if(hi < lo || (size_t)(hi - lo) < n) return nullptr;
            for(const char* q = lo; q + n <= hi; q++)
              if(memcmp(q, lit, n) == 0) return q;
            return nullptr;
          };
          for(const char* q = arr + 1; q < ae;){
            const char* ob = find_lit(q, ae, "{");
            if(!ob) break;
            // match the object's closing brace (string-aware)
            const char* oe = nullptr;
            {
              bool instr = false, esc = false; int depth = 0;
              for(const char* r = ob; r < ae; r++){
                char c = *r;
                if(instr){
                  if(esc) esc = false;
                  else if(c == '\\') esc = true;
                  else if(c == '"') instr = false;
                } else {
                  if(c == '"') instr = true;
                  else if(c == '{') depth++;
                  else if(c == '}'){ if(--depth == 0){ oe = r; break; } }
                }
              }
            }
            if(!oe) break;
            const char* ids = find_lit(ob, oe, "\"id\"");
            const char* cs = find_lit(ob, oe, "\"content\"");
            if(ids && cs){
              char* e = nullptr;
              const char* num = ids + 4;
              while(num < oe && (*num==' '||*num=='\t'||*num==':')) num++;
              long id = strtol(num, &e, 10);
              const char* q1 = find_lit(cs + 9, oe, "\"");
              if(e != num && q1){
                q1++;
                std::string content;
                bool es2 = false;
                const char* r = q1;
                while(r < oe){
                  char c = *r++;
                  if(es2){ content.push_back(c=='n'?'\n':c=='t'?'\t':c); es2=false; }
                  else if(c=='\\') es2 = true;
                  else if(c=='"') break;
                  else content.push_back(c);
                }
                if(id >= 0){
                  if((size_t)id >= bytes_of.size()) bytes_of.resize((size_t)id+1);
                  bytes_of[(size_t)id] = content;
                }
              }
            }
            q = oe + 1;
          }
        }
      }
    }
    ok = true;
    return true;
  }

  std::string decode_one(int id) const {
    if(id < 0 || (size_t)id >= bytes_of.size()) return {};
    return bytes_of[id];
  }
};
static BpDecoder g_bpdec;
static std::string quote_for_shell(const std::string& s){
  std::string out = "'";
  for(char c : s){
    if(c=='\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}
static std::vector<int> encode_text(const std::string& text,
                                     const std::string& tokenizer_json,
                                     const std::string& tokenizer_py){
  std::string escaped;
  for(char c : text){
    if(c == '\'') escaped += "'\\''";
    else escaped += c;
  }
  std::string cmd = "python3 " + quote_for_shell(tokenizer_py) +
                    " --tokenizer " + quote_for_shell(tokenizer_json) +
                    " --input '" + escaped + "'";
  FILE* fp = popen(cmd.c_str(), "r");
  if(!fp){ std::fprintf(stderr, "intfer: cannot launch tokenizer\n"); return {}; }
  std::vector<int> ids;
  int v;
  while(fread(&v, sizeof(int), 1, fp)==1) ids.push_back(v);
  pclose(fp);
  return ids;
}
static std::string detokenize_one(int id, const std::string& detok_py,
                                  const std::string& tokenizer_json){
  (void)detok_py;
  // In-process fast path: parsed vocab lookup (no python spawn).
  if(g_bpdec.ok) return g_bpdec.decode_one(id);
  char idbuf[32]; snprintf(idbuf, sizeof(idbuf), " --id %d", id);
  std::string cmd = "python3 " + quote_for_shell(detok_py) +
                    " --tokenizer " + quote_for_shell(tokenizer_json) +
                    idbuf;
  FILE* fp = popen(cmd.c_str(), "r");
  if(!fp) return {};
  std::string s; char buf[256];
  size_t n = fread(buf, 1, sizeof(buf)-1, fp);
  pclose(fp);
  if(n==0) return {};
  s.assign(buf, n);
  return s;
}

// ============================================================================
// Serve loop (Stage 3.3): stdin JSON-lines continuous batching server.
// ============================================================================
// Protocol:
//   in:  {"prompt":"...","max_new":N,"id":K,"temperature":T,"top_k":K,
//         "top_p":P,"seed":S} per line (only prompt is required; sampling
//         keys default to the CLI -t/-k/-T/-S values; numeric keys are
//         matched outside the prompt string so prompt prose cannot spoof
//         them; missing id is server-assigned).
//   out: {"id":K,"token":ID,"text":"..."} per generated token,
//        {"id":K,"done":true,"generated":N} once per request,
//        {"error":"..."} on malformed input lines (server keeps running).
// Mechanics: poll stdin with select() (zero timeout: never stall decode
// waiting for input); admit pending lines FIFO up to INTFER_SERVE_MAX live
// (default computed from per-request state bytes under an 8GB host
// budget); each iteration fills ONE prompt chunk (INTFER_PREFILL_CHUNK_M,
// default 8) for the prefill-queue head, then runs one sched_iteration
// over all live decodes. EOF drains the queue + live to completion.
// Returns 0 on clean EOF drain (munmap left to the caller).
// Admission queue (Stage 3.3+ improvement B): tokenized but not yet
// prefilled. Each scheduler iteration runs ONE prompt chunk (default 8
// tokens) for the queue head, then the normal decode iteration over live
// requests — so a long prefill never stalls live decodes, and arrivals
// overlap with ongoing generation (true continuous batching: prefill and
// decode share every iteration). Chunk-at-offset is the exact tested
// sched_prefill mechanism; per-request KV/SSM state + absolute positions
// make interleaving output-identical to admit-time prefill.
// Live cap shared by --serve and the --serve-http dispatcher: VRAM-aware
// default from per-request state bytes (KV caches + scratch arena +
// gscores, all host-side FP32; GPU weights are shared). Budget 8GB host
// for request state; INTFER_SERVE_MAX still overrides.
static int serve_live_cap(const std::function<void(Request&)>& wire){
  int serve_max = 8;
  {
    size_t per_req = 0;
    int n_full = 0, n_la = 0;
    {
      Request probe; wire(probe);
      for(auto& ly : probe.layers){
        if(ly.is_full){ n_full++; per_req += ly.kcache.size() + ly.vcache.size(); }
        else { n_la++; per_req += ly.ssm_state.size() + ly.conv_state.size(); }
      }
    }
    size_t qrows2 = 2*(size_t)cfg.n_heads*cfg.head_dim;
    size_t kvrows = (size_t)cfg.n_kv*cfg.head_dim;
    size_t attnrows = (size_t)cfg.n_heads*cfg.head_dim;
    size_t nvrows = (size_t)cfg.n_v_heads*cfg.v_head_dim;
    per_req += 3*cfg.H + cfg.H + qrows2 + 2*attnrows + 2*cfg.H + 2*kvrows
             + attnrows + 2*attnrows + cfg.H + 3*cfg.H + 2*cfg.conv_dim
             + cfg.H + 3*cfg.H + nvrows + 2*cfg.n_v_heads + cfg.n_v_heads
             + 3*nvrows + 2*cfg.v_head_dim + 2*cfg.H + 2*cfg.inter
             + 2*cfg.H + cfg.H + cfg.vocab; // StepScratch::ensure widths
    per_req *= sizeof(float);
    const size_t budget = 8ULL*1024*1024*1024;
    int computed = (int)(budget / std::max<size_t>(per_req, 1));
    if(computed < 1) computed = 1;
    if(computed > 64) computed = 64;
    serve_max = computed;
    std::fprintf(stderr, "[intfer] serve cap: %d live (%.1f MB/req state: "
                 "%d full + %d LA layers)\n",
                 serve_max, per_req/(1024.0*1024.0), n_full, n_la);
  }
  if(const char* me = std::getenv("INTFER_SERVE_MAX")){
    int v = std::atoi(me); if(v >= 1 && v <= 64) serve_max = v;
  }
  return serve_max;
}

// Round-robin rotation for the split prefill path: the served head goes to
// the back unless it filled (filled requests leave via pop_front at their
// fill site). INTFER_PREFILL_RR=0 restores head-of-line (FIFO) behavior.
static void serve_rotate_unfilled(std::deque<Request*>& prefill_q, Request* served){
  static const bool rr_off2 = (std::getenv("INTFER_PREFILL_RR") != nullptr &&
                               std::string(std::getenv("INTFER_PREFILL_RR")) == "0");
  if(rr_off2 || prefill_q.empty() || prefill_q.size() < 2) return;
  if(prefill_q.front() == served && served->fill < (int)served->prompt_ids.size()){
    prefill_q.pop_front();
    prefill_q.push_back(served);
  }
}

// One scheduler iteration over (live, prefill_q), shared verbatim by
// --serve and the --serve-http dispatcher (identical forward mechanics):
// joint forward (decode rows + head prefill chunk in one step_batch) when
// possible, else the split chunk + decode forward. INTFER_SPLIT_ITER=1
// forces the split path for A/B. Per-row token emission is the caller's
// emit hook; live compaction / {"done"} accounting stays with callers.
static void serve_iteration(std::vector<Request*>& live,
                            std::deque<Request*>& prefill_q,
                            int chunk_M, SchedCtx& c,
                            std::vector<float>& Hb, std::vector<float>& Hpre,
                            std::vector<std::vector<float>>& no_dump,
                            const std::function<void(Request&, size_t, int)>& emit,
                            int& step_i, bool split_iter){
  auto split_chunk = [&](){
    if(!prefill_q.empty()){
      Request& q = *prefill_q.front();
      int Sr = (int)q.prompt_ids.size();
      int cm = std::min(chunk_M, Sr - q.fill);
      // Single-token heads ride the joint forward (fused with decodes);
      // multi-token chunks use the chunked prefill path bit-identically
      // in both modes (output-identity gate compares per-id streams).
      if(cm == 1 && !split_iter && !live.empty()){
        int D = sched_iteration_joint(live, prefill_q, chunk_M, c, Hb,
                                      emit, step_i++);
        if(D >= 0) return; // joint carried it; accounting runs below
        // D < 0: joint declined; fall through to the chunk block.
      }
      std::vector<int> toks(q.prompt_ids.begin() + q.fill,
                            q.prompt_ids.begin() + q.fill + cm);
      Hpre.assign((size_t)cm * cfg.H, 0.f);
      StepOut o = prefill_layer_outer(q, toks, q.fill, *c.T, Hpre, c.embed,
                                      c.final_norm, c.lm_head, c.backend,
                                      no_dump);
      q.fill += cm;
      if(q.fill == Sr){
        q.generated.push_back(o.best);
        q.token = q.generated.back();
        q.pos = Sr;
        if(c.print_ids)
          std::fprintf(stderr, "  [serve id=%ld prefill] id=%d\n", q.id, q.token);
        live.push_back(&q);
        prefill_q.pop_front();
      } else {
        serve_rotate_unfilled(prefill_q, &q);
      }
    }
    if(!live.empty()) sched_iteration(live, c, Hb, emit, step_i++);
  };
  if(split_iter){
    split_chunk();
  } else if(!prefill_q.empty() || !live.empty()){
    int D = sched_iteration_joint(live, prefill_q, chunk_M, c, Hb,
                                  emit, step_i++);
    if(D < 0){
      // Joint declines (BATCH_GEMM wide-chunk guard): run the split
      // chunk for the head, then rotate so the next iteration serves the
      // next queued request (round-robin; identical rotation in the
      // joint path keeps both modes comparable).
      split_chunk(); // joint declined: same split mechanics
    }
  }
}
static int run_serve(SchedCtx& c,
                     const std::function<void(Request&)>& wire,
                     int dflt_max_new, uint32_t seed,
                     float dflt_temp, int dflt_top_k, float dflt_top_p){
  std::vector<std::unique_ptr<Request>> pool; // all admitted (owning)
  std::vector<Request*> live;                 // decoding (into pool)
  std::deque<Request*> prefill_q;             // admitted, filling (into pool)
  std::deque<std::string> pending;            // FIFO of unadmitted lines
  int chunk_M = 128;
  if(const char* me = std::getenv("INTFER_PREFILL_CHUNK_M")){
    int v = std::atoi(me); if(v >= 1 && v <= 1024) chunk_M = v;
  }
  std::vector<float> Hpre; // reused prefill-chunk staging (shared, one chunk
                           // in flight per iteration)
  int serve_max = serve_live_cap(wire);
  // Joint prefill+decode forward (improvement C): one step_batch carries
  // decode rows + the head chunk. INTFER_SPLIT_ITER=1 restores the split
  // path (one chunk forward + one decode forward) for A/B testing; the
  // joint path is output-identical by construction (gated).
  static const bool split_iter = (std::getenv("INTFER_SPLIT_ITER") != nullptr);
  long next_id = 1;
  bool eof = false;
  std::vector<std::vector<float>> no_dump; // server never records
  std::vector<float> Hb;
  int step_i = 0;
  auto emit_stream = [&](Request& q, size_t, int tok){
    std::string piece = detokenize_one(tok, *c.detok_py, *c.tokenizer_json);
    std::printf("{\"id\":%ld,\"token\":%d,\"text\":\"%s\"}\n",
                q.id, tok, json_escape(piece).c_str());
    std::fflush(stdout);
  };
  // Admit one parsed request: tokenize + wire, then queue for chunked
  // prefill (B: no forward pass here — the loop below fills one chunk per
  // iteration, so decodes keep flowing while this prompt fills).
  auto admit = [&](const std::string& prompt, const ServeOpts& o){
    auto rq = std::make_unique<Request>();
    wire(*rq);
    Request& q = *rq;
    q.id = o.id;
    if(!prompt.empty())
      q.prompt_ids = encode_text(prompt, *c.tokenizer_json, *c.tokenizer_py);
    if(q.prompt_ids.empty())
      q.prompt_ids = {760, 6511, 314, 9338, 369};
    q.max_new = o.max_new;
    q.temperature = o.temperature; q.top_k = o.top_k; q.top_p = o.top_p;
    q.rng = std::mt19937(o.has_seed ? o.seed : seed + (uint32_t)q.id);
    q.fill = 0; q.pos = 0;
    std::fprintf(stderr, "[intfer] serve admit id=%ld prompt=%zu max_new=%d live=%zu pq=%zu\n",
                 q.id, q.prompt_ids.size(), q.max_new, live.size(), prefill_q.size()+1);
    prefill_q.push_back(rq.get());
    pool.push_back(std::move(rq));
  };
  std::string bufline;
  while(!eof || !pending.empty() || !prefill_q.empty() || !live.empty()){
    // --- Poll stdin (non-blocking): drain all ready lines into pending.
    if(!eof){
      for(;;){
        fd_set rfds; FD_ZERO(&rfds); FD_SET(STDIN_FILENO, &rfds);
        struct timeval tv{0, 0};
        int r = select(STDIN_FILENO+1, &rfds, nullptr, nullptr, &tv);
        if(r <= 0 || !FD_ISSET(STDIN_FILENO, &rfds)) break;
        if(!std::getline(std::cin, bufline)){ eof = true; break; }
        if(!bufline.empty() && bufline.back() == '\r') bufline.pop_back();
        if(bufline.empty()) continue;
        pending.push_back(bufline);
      }
      if(std::cin.eof()) eof = true;
    }
    // --- Admit FIFO while under the live cap.
    while(!pending.empty() && (int)live.size() < serve_max){
      std::string line = std::move(pending.front());
      pending.pop_front();
      std::string prompt; ServeOpts o;
      if(!parse_serve_line(line, prompt, o, dflt_max_new,
                           dflt_temp, dflt_top_k, dflt_top_p)){
        std::printf("{\"error\":\"malformed request line\"}\n");
        std::fflush(stdout);
        continue;
      }
      if(!o.has_id) o.id = next_id++;
      else if(o.id >= next_id) next_id = o.id + 1;
      admit(prompt, o);
    }
    // --- One scheduler iteration: joint forward (decode rows + head
    // prefill chunk) or the split path; shared verbatim with --serve-http.
    serve_iteration(live, prefill_q, chunk_M, c, Hb, Hpre, no_dump,
                    emit_stream, step_i, split_iter);
    // --- Per-iteration accounting: {"done"} emission + live compaction.
    // Runs whenever live is non-empty after the forward(s) above (a filled
    // prefill request may have joined). EOF/idle drain when all queues
    // are empty.
    if(!live.empty()){
      // Emit {"done"} once per newly finished request; compact live.
      std::vector<Request*> still;
      for(Request* qp : live){
        if(qp->finished && !qp->done_emitted){
          qp->done_emitted = true;
          std::printf("{\"id\":%ld,\"done\":true,\"generated\":%zu}\n",
                      qp->id, qp->generated.size());
          std::fflush(stdout);
        }
        if(!qp->finished) still.push_back(qp);
      }
      live.swap(still);
    }
    if(live.empty() && prefill_q.empty()){
      if(eof && pending.empty()) break;
      if(pending.empty() && !eof){
        // Nothing to do: block for the next line (avoids a hot spin when
        // idle with the pipe open but no data).
        if(!std::getline(std::cin, bufline)){ break; }
        if(!bufline.empty() && bufline.back() == '\r') bufline.pop_back();
        if(!bufline.empty()) pending.push_back(bufline);
      }
    }
  }
  return 0;
}

// ============================================================================
// CLI
// ============================================================================
enum class GpuMode { Auto, ForceOn, ForceOff };

// ============================================================================
// OpenAI-compatible HTTP server (--serve-http).
// ============================================================================
// Same engine, new skin: this section bridges the dependency-free HTTP layer
// (http.hpp) + OpenAI request parsing (openai.hpp) to the exact tested
// --serve forward mechanics (chunked prefill + joint decode iterations over
// a live batch). Token ids match the --serve path for the same
// prompt/seed/sampling settings; only the transport differs (HTTP+SSE
// instead of stdin/stdout JSON-lines).
//
// Concurrency model: the HTTP layer is thread-per-connection, but ALL engine
// work runs on ONE dispatcher thread (the forward path is not thread-safe:
// thread_local KV scratch in kv_row_ptr, shared backend weight caches, SYCL
// queue state). Handler threads tokenize (process-spawned python, parallel)
// then enqueue jobs; the dispatcher admits them into prefill_q/live and runs
// the shared serve_iteration, so concurrent HTTP requests share M-row
// forwards (cross-request continuous batching). SSE pieces flow
// dispatcher -> per-job queue -> handler thread -> socket, so the engine
// thread never touches sockets. INTFER_HTTP_SERIAL=1 restores the old
// one-request-at-a-time http_generate path (A/B + rollback lever).
//
// Token accounting (verified against the HF oracle + test_cpu_parity.sh):
// the prefill-predicted first token is emitted as generated[0], then each
// decode iteration appends one token. So acc.ids holds the FULL output
// stream including the prefill token, and every element is emitted exactly
// once (EOS id itself is never emitted as text).
static std::mutex g_engine_mu; // serial fallback only (INTFER_HTTP_SERIAL=1)
static std::atomic<long> g_completion_seq{1};

struct HttpGenOpts {
  int max_new = 64;
  float temperature = 0.f;
  int top_k = 0;
  float top_p = 0.f;
  uint32_t seed = 0;
  bool has_seed = false;
  std::vector<std::string> stop;
};

struct HttpGenOut {
  std::vector<int> ids;  // full output stream (incl. prefill token), EOS excluded
  std::string text;      // decoded text, stop-truncated, EOS excluded
  std::string finish;    // "stop" | "length"
  int prompt_tokens = 0;
};

// One HTTP completion as seen by the dispatcher: tokenized prompt +
// sampling opts in, accumulated output stream + SSE piece queue out.
// The Request (KV/SSM state) is dispatcher-owned; the handler thread only
// touches mu/cv/pieces/done/cancelled.
struct HttpJob {
  std::vector<int> prompt_ids;
  HttpGenOpts opts;
  bool streaming = false;
  long long seq = 0;             // completion id sequence (seed derivation)
  Request req;                   // wired + filled by the dispatcher
  HttpGenOut acc;                // accumulated ids/text/finish
  size_t streamed = 0;           // acc.text prefix already queued as pieces
  bool stopped = false;          // EOS/stop-string hit (finish => "stop")
  bool prefill_emitted = false;  // generated[0] already emitted
  std::atomic<bool> cancelled{false}; // handler: SSE socket died
  // Handoff to the handler thread:
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::pair<std::string,int>> pieces; // (text delta, tok)
  bool done = false;
};

// Dispatcher: single engine thread running serve_iteration over admitted
// HTTP jobs. Handler threads submit() pre-tokenized jobs and wait; the
// dispatcher loop admits (wire + prefill_q), iterates, emits per-row
// tokens into job accumulators (+ SSE piece queues), and completes jobs.
struct HttpDispatcher {
  SchedCtx* c = nullptr;
  const std::function<void(Request&)>* wire = nullptr;
  uint32_t seed_base = 0;
  int chunk_M = 8;
  int serve_max = 8;
  bool split_iter = false;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::shared_ptr<HttpJob>> incoming; // handler -> dispatcher
  std::atomic<bool> stop{false};
  std::atomic<int> active{0}; // admitted-not-released jobs (for the 429 cap)
  std::thread thr;
  // Dispatcher-thread-only state:
  std::list<std::shared_ptr<HttpJob>> jobs;      // admitted (stable addrs)
  std::vector<Request*> live;                    // decoding (into jobs)
  std::deque<Request*> prefill_q;                // admitted, filling
  std::unordered_map<Request*, HttpJob*> r2j;
  std::vector<float> Hb, Hpre;
  std::vector<std::vector<float>> no_dump;       // server never records
  int step_i = 0;

  void start(SchedCtx& cc, const std::function<void(Request&)>& w,
             uint32_t sb){
    c = &cc; wire = &w; seed_base = sb;
    if(const char* me = std::getenv("INTFER_PREFILL_CHUNK_M")){
      int v = std::atoi(me); if(v >= 1 && v <= 1024) chunk_M = v;
    }
    serve_max = serve_live_cap(*wire);
    split_iter = (std::getenv("INTFER_SPLIT_ITER") != nullptr);
    thr = std::thread([this]{ loop(); });
  }
  void shutdown(){
    stop = true;
    cv.notify_all();
    if(thr.joinable()) thr.join();
  }
  // Handler side: enqueue a pre-tokenized job (tokens already encoded off
  // the engine thread). Returns null when the live cap is hit (caller
  // answers 429); otherwise the job is admitted on the next iteration.
  std::shared_ptr<HttpJob> submit(std::vector<int> prompt_ids,
                                  const HttpGenOpts& o, bool streaming){
    auto j = std::make_shared<HttpJob>();
    j->prompt_ids = std::move(prompt_ids);
    if(j->prompt_ids.empty())
      j->prompt_ids = {760, 6511, 314, 9338, 369}; // same default as --serve
    j->opts = o;
    j->streaming = streaming;
    j->seq = g_completion_seq.fetch_add(1);
    j->acc.prompt_tokens = (int)j->prompt_ids.size();
    {
      std::lock_guard<std::mutex> lk(mu);
      if(active.load() + (int)incoming.size() >= serve_max)
        return nullptr;
      incoming.push_back(j);
      active.fetch_add(1);
    }
    cv.notify_one();
    return j;
  }
  // Handler side: wait for completion (non-streaming). Returns acc.
  HttpGenOut await_done(const std::shared_ptr<HttpJob>& j){
    std::unique_lock<std::mutex> lk(j->mu);
    j->cv.wait(lk, [&]{ return j->done; });
    return j->acc;
  }
 private:

  // Dispatcher side: per-token emission into a job accumulator (mirrors
  // the serial http_generate emit_one: EOS never text, stop strings cut
  // the text, streaming jobs get exactly the kept delta as pieces).
  void emit_one(HttpJob& j, int tok){
    if(tok == 248044 || tok == 248046){ j.stopped = true; return; }
    std::string piece = detokenize_one(tok, *c->detok_py, *c->tokenizer_json);
    j.acc.ids.push_back(tok);
    j.acc.text += piece;
    for(auto& st : j.opts.stop){
      if(st.empty()) continue;
      size_t p = j.acc.text.find(st);
      if(p != std::string::npos){
        j.acc.text.resize(p);
        j.stopped = true;
        break;
      }
    }
    if(j.streaming){
      std::lock_guard<std::mutex> lk(j.mu);
      if(j.acc.text.size() > j.streamed)
        j.pieces.emplace_back(j.acc.text.substr(j.streamed), tok);
      j.streamed = j.acc.text.size();
      j.cv.notify_one();
    }
    if(j.stopped) j.req.finished = true;
  }
  void finish_job(const std::shared_ptr<HttpJob>& j){
    {
      std::lock_guard<std::mutex> lk(j->mu);
      // ids keep every generated token (vLLM-style usage counting); text
      // alone is stop-truncated. EOS/stop-string => "stop", else the
      // request ended on the max_new budget => "length".
      j->acc.finish = j->stopped ? "stop" : "length";
      j->done = true;
    }
    j->cv.notify_all();
  }
  void loop(){
    auto emit = [&](Request& q, size_t, int tok){
      auto it = r2j.find(&q);
      if(it != r2j.end()) emit_one(*it->second, tok);
    };
    for(;;){
      // --- Admit FIFO while under the live cap (non-blocking drain).
      {
        std::unique_lock<std::mutex> lk(mu);
        while(!incoming.empty() &&
              (int)(live.size() + prefill_q.size()) < serve_max){
          auto j = incoming.front();
          incoming.pop_front();
          lk.unlock(); // wire() touches only the job's own Request
          (*wire)(j->req);
          Request& q = j->req;
          q.prompt_ids = j->prompt_ids;
          // OpenAI max_tokens caps TOTAL completion tokens (prefill token
          // included): the scheduler stops when generated.size() >=
          // max_new+1, so subtract one here (same as serial path).
          q.max_new = std::max(0, j->opts.max_new - 1);
          q.temperature = j->opts.temperature;
          q.top_k = j->opts.top_k;
          q.top_p = j->opts.top_p;
          q.rng = std::mt19937(j->opts.has_seed ? j->opts.seed
                               : seed_base + (uint32_t)(j->seq & 0xffffff));
          q.fill = 0; q.pos = 0;
          q.id = (long)j->seq;
          std::fprintf(stderr, "[intfer] http admit seq=%lld prompt=%zu max_new=%d live=%zu pq=%zu\n",
                       j->seq, q.prompt_ids.size(), q.max_new,
                       live.size(), prefill_q.size()+1);
          prefill_q.push_back(&q);
          r2j[&q] = j.get();
          jobs.push_back(std::move(j));
          lk.lock();
        }
      }
      if(live.empty() && prefill_q.empty()){
        if(stop.load()) break;
        // Nothing to do: park until arrivals (avoids a hot spin when
        // idle with the listener open but no work).
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]{ return stop.load() || !incoming.empty(); });
        if(stop.load() && incoming.empty()) break;
        continue;
      }
      // Snapshot live membership: requests joining live below have just
      // filled — their prefill token (generated[0]) is emitted after.
      std::unordered_map<Request*, char> was_live;
      for(Request* qp : live) was_live[qp] = 1;
      serve_iteration(live, prefill_q, chunk_M, *c, Hb, Hpre, no_dump,
                      emit, step_i, split_iter);
      // Newly filled: emit the prefill token first in their stream.
      for(Request* qp : live){
        if(was_live.count(qp)) continue;
        auto it = r2j.find(qp);
        if(it == r2j.end()) continue;
        HttpJob& j = *it->second;
        if(!j.prefill_emitted){
          j.prefill_emitted = true;
          emit_one(j, qp->token);
        }
      }
      // Compact live: drop finished/cancelled, complete their jobs.
      std::vector<Request*> still;
      for(Request* qp : live){
        auto it = r2j.find(qp);
        HttpJob* j = (it == r2j.end()) ? nullptr : it->second;
        bool cancel = j && j->cancelled.load();
        if(qp->finished || cancel){
          if(j){
            // Find the owning shared_ptr to finish + release state.
            for(auto it2 = jobs.begin(); it2 != jobs.end(); ++it2){
              if(it2->get() == j){
                if(cancel && !qp->finished) qp->finished = true;
                finish_job(*it2);
                r2j.erase(qp);
                jobs.erase(it2);
                active.fetch_sub(1);
                break;
              }
            }
          }
        } else {
          still.push_back(qp);
        }
      }
      live.swap(still);
    }
  }
};

// Run one completion under the engine mutex (serial fallback for
// INTFER_HTTP_SERIAL=1, and the A/B reference for the dispatcher).
// on_token(piece, id) is invoked per emitted token (used for SSE
// streaming); when null, tokens are only accumulated into the returned
// HttpGenOut.
static HttpGenOut http_generate(
    const std::string& prompt, const HttpGenOpts& o,
    SchedCtx& c, const std::function<void(Request&)>& wire,
    uint32_t seed_base,
    const std::function<void(const std::string&, int)>& on_token = {}) {
  HttpGenOut out;
  std::lock_guard<std::mutex> lk(g_engine_mu);
  Request q;
  wire(q);
  if(!prompt.empty())
    q.prompt_ids = encode_text(prompt, *c.tokenizer_json, *c.tokenizer_py);
  if(q.prompt_ids.empty())
    q.prompt_ids = {760, 6511, 314, 9338, 369};  // same default as --serve
  out.prompt_tokens = (int)q.prompt_ids.size();
  // OpenAI max_tokens caps TOTAL completion tokens (prefill token included:
  // sched_* counts generated[0] as output). The scheduler stops decoding
  // when generated.size() >= max_new+1, so subtract one here.
  q.max_new = std::max(0, o.max_new - 1);
  q.temperature = o.temperature;
  q.top_k = o.top_k;
  q.top_p = o.top_p;
  q.rng = std::mt19937(o.has_seed ? o.seed : seed_base + (uint32_t)(g_completion_seq.load() & 0xffffff));
  q.fill = 0;
  q.pos = 0;
  std::vector<std::vector<float>> no_dump;  // server never records
  std::vector<float> Hb;
  sched_prefill(q, c, no_dump);
  // Prefill token: first element of the output stream.
  bool stopped = false;
  size_t streamed = 0;  // out.text prefix already handed to on_token
  auto emit_one = [&](int tok) {
    if(tok == 248044 || tok == 248046) { stopped = true; return; }  // EOS: end, never text
    std::string piece = detokenize_one(tok, *c.detok_py, *c.tokenizer_json);
    out.ids.push_back(tok);
    out.text += piece;
    // OpenAI stop semantics: cut at the first occurrence of any stop
    // string (stop text itself excluded); boundary-spanning stops are
    // caught because the full accumulated text is searched each time.
    for(auto& st : o.stop){
      if(st.empty()) continue;
      size_t p = out.text.find(st);
      if(p != std::string::npos){
        out.text.resize(p);
        stopped = true;
        break;
      }
    }
    // Stream exactly the kept delta: previously-streamed text is never
    // re-sent, and a stop-hit piece contributes only its kept prefix
    // (possibly empty — then nothing is sent for this token).
    if(on_token && out.text.size() > streamed){
      on_token(out.text.substr(streamed), tok);
      streamed = out.text.size();
    } else if(on_token){
      streamed = out.text.size();
    }
  };
  emit_one(q.token);
  std::vector<Request*> live;
  int step_i = 0;
  auto emit_stream = [&](Request&, size_t, int tok){ emit_one(tok); };
  while(!q.finished && !stopped){
    live.clear();
    live.push_back(&q);
    int M = sched_iteration(live, c, Hb, emit_stream, step_i++);
    if(M == 0) break;
  }
  // ids keep every generated token (vLLM-style usage counting); text alone
  // is stop-truncated. finish: EOS/stop-string => "stop", else the request
  // ended because sched_iteration hit the max_new budget => "length".
  // (EOS always funnels through emit_one above, so stopped covers it.)
  out.finish = stopped ? "stop" : "length";
  return out;
}

// --serve-http request routing. All /v1/* completions funnel into the
// dispatcher (or the serial http_generate under INTFER_HTTP_SERIAL=1);
// /health and /v1/models never touch the engine.
static int run_serve_http(SchedCtx& c,
                          const std::function<void(Request&)>& wire,
                          int dflt_max_new, uint32_t seed_base,
                          float dflt_temp, int dflt_top_k, float dflt_top_p,
                          const std::string& host, int port,
                          const std::string& model_id) {
  using namespace intfer_openai;
  intfer_http::HttpServer srv;
  static const bool http_serial = (std::getenv("INTFER_HTTP_SERIAL") != nullptr);
  HttpDispatcher disp;
  if(!http_serial) disp.start(c, wire, seed_base);
  // Tokenize off the engine thread (process-spawned python, parallel
  // across connections), then enqueue. Null => live cap hit (429).
  auto submit_text = [&](const std::string& prompt, const HttpGenOpts& o,
                         bool streaming) -> std::shared_ptr<HttpJob> {
    std::vector<int> ids;
    if(!prompt.empty())
      ids = encode_text(prompt, *c.tokenizer_json, *c.tokenizer_py);
    return disp.submit(std::move(ids), o, streaming);
  };
  // Serial fallback: identical opts handling to the dispatcher submit
  // path (same defaults, same max_tokens accounting).
  auto serial_gen = [&](const std::string& prompt, const HttpGenOpts& o,
                        const std::function<void(const std::string&, int)>& on_token = {}) {
    return http_generate(prompt, o, c, wire, seed_base, on_token);
  };
  // max_new clamp mirrors ServeOpts ([1,4096]).
  auto clamp_max = [](int n, int dflt){
    if(n <= 0) return dflt;
    if(n < 1) return 1;
    if(n > 4096) return 4096;
    return n;
  };
  srv.on_log([](const std::string& m, const std::string& p, int, long ms){
    std::fprintf(stderr, "[http] %s %s %ldms\n", m.c_str(), p.c_str(), ms);
  });
  srv.on_request([&](const intfer_http::HttpRequest& req, intfer_http::Responder& rsp){
    auto err = [&](int code, const std::string& msg, const std::string& type){
      JVal e = make_error(msg, type);
      rsp.send_json(code, json_dump(e));
    };
    std::string path = req.path;
    // CORS preflight.
    if(req.method == "OPTIONS"){
      rsp.send_response(200, "text/plain", "",
                        {{"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                         {"Access-Control-Allow-Headers", "Content-Type, Authorization"}});
      return;
    }
    if(req.method == "GET" && (path == "/health" || path == "/v1/health")){
      JVal o = JVal::obj();
      o.set("status", JVal::str("ok"));
      o.set("model", JVal::str(model_id));
      rsp.send_json(200, json_dump(o));
      return;
    }
    // GET /v1/models (+ trailing-slash + single-id forms).
    if(req.method == "GET" && (path == "/v1/models" || path == "/v1/models/")){
      JVal o = JVal::obj();
      o.set("object", JVal::str("list"));
      JVal arr = JVal::arr();
      JVal m = JVal::obj();
      m.set("id", JVal::str(model_id));
      m.set("object", JVal::str("model"));
      m.set("created", JVal::number((double)std::time(nullptr)));
      m.set("owned_by", JVal::str("intfer"));
      arr.push(m);
      o.set("data", arr);
      rsp.send_json(200, json_dump(o));
      return;
    }
    if(req.method == "GET" && path.rfind("/v1/models/", 0) == 0){
      JVal m = JVal::obj();
      m.set("id", JVal::str(model_id));
      m.set("object", JVal::str("model"));
      m.set("created", JVal::number((double)std::time(nullptr)));
      m.set("owned_by", JVal::str("intfer"));
      rsp.send_json(200, json_dump(m));
      return;
    }
    if(path == "/v1/chat/completions" || path == "/v1/completions"){
      bool is_chat = (path == "/v1/chat/completions");
      if(req.method != "POST"){ err(405, "method not allowed, use POST", "invalid_request_error"); return; }
      JVal body;
      JParser parser(req.body);
      if(!parser.parse(body) || !body.is_obj()){
        err(400, "request body is not valid JSON", "invalid_request_error");
        return;
      }
      // --- legacy /v1/completions fields ---
      std::vector<std::string> legacy_prompts;
      bool has_legacy_prompt = false, legacy_echo = false;
      if(!is_chat){
        const JVal* pr = body.find("prompt");
        if(!pr){ err(400, "prompt must be a string or array of strings", "invalid_request_error"); return; }
        has_legacy_prompt = true;
        if(pr->is_str()) legacy_prompts.push_back(pr->s);
        else if(pr->is_arr()){
          for(auto& e : pr->a){
            if(!e.is_str()){ err(400, "prompt array must contain strings", "invalid_request_error"); return; }
            legacy_prompts.push_back(e.s);
          }
        } else { err(400, "prompt must be a string or array of strings", "invalid_request_error"); return; }
        const JVal* ec = body.find("echo");
        if(ec && ec->t == JVal::T::Bool) legacy_echo = ec->b;
        const JVal* n = body.find("n");
        if(n && n->is_num() && (int)n->num != 1){
          err(400, "n > 1 is not supported (single completion only)", "invalid_request_error");
          return;
        }
        for(const char* k : {"logit_bias", "logprobs", "best_of", "functions", "suffix"}){
          const JVal* v = body.find(k);
          if(v && v->t != JVal::T::Null){
            if(v->t == JVal::T::Bool && !v->b) continue;
            if(v->t == JVal::T::Arr && v->a.empty()) continue;
            if(v->t == JVal::T::Num && v->num == 0) continue;
            err(400, std::string(k) + " is not supported by this server", "invalid_request_error");
            return;
          }
        }
        (void)has_legacy_prompt;
      }
      // --- shared sampling fields ---
      ChatRequest cr;
      std::string prompt_text;
      std::string req_model;
      if(is_chat){
        if(!parse_chat_request(body, cr)){
          err(400, cr.validation_error.empty() ? "invalid request" : cr.validation_error,
              "invalid_request_error");
          return;
        }
        req_model = cr.model.empty() ? model_id : cr.model;
        prompt_text = render_chat_prompt(cr.messages, true);
      } else {
        req_model = model_id;
        const JVal* m = body.find("model");
        if(m && m->is_str() && !m->s.empty()) req_model = m->s;
      }
      HttpGenOpts o;
      o.max_new = dflt_max_new;
      o.temperature = dflt_temp; o.top_k = dflt_top_k; o.top_p = dflt_top_p;
      const JVal* mt = body.find("max_tokens");
      if(!mt) mt = body.find("max_completion_tokens");
      if(mt && mt->is_num()) o.max_new = clamp_max((int)mt->num, dflt_max_new);
      const JVal* tp = body.find("temperature");
      if(tp && tp->is_num()) o.temperature = (float)tp->num;
      const JVal* pp = body.find("top_p");
      if(pp && pp->is_num()) o.top_p = (float)pp->num;
      const JVal* tk = body.find("top_k");
      if(tk && tk->is_num()) o.top_k = (int)tk->num;
      const JVal* sd = body.find("seed");
      if(sd && sd->is_num()){ o.seed = (uint32_t)(int)sd->num; o.has_seed = true; }
      const JVal* sp = body.find("stop");
      if(sp){
        if(sp->is_str()) o.stop.push_back(sp->s);
        else if(sp->is_arr()){
          for(auto& e : sp->a){
            if(!e.is_str()){ err(400, "stop must be a string or array of strings", "invalid_request_error"); return; }
            o.stop.push_back(e.s);
          }
        } else if(sp->t != JVal::T::Null){
          err(400, "stop must be a string or array of strings", "invalid_request_error");
          return;
        }
      } else if(is_chat){
        o.stop = cr.stop;
      }
      bool stream_out = false;
      const JVal* st = body.find("stream");
      if(st && st->t == JVal::T::Bool) stream_out = st->b;
      bool want_usage = false;
      const JVal* so = body.find("stream_options");
      if(so && so->is_obj()){
        const JVal* iu = so->find("include_usage");
        if(iu && iu->t == JVal::T::Bool) want_usage = iu->b;
      }
      long long created = (long long)std::time(nullptr);
      auto usage_obj = [](int pt, int ct){
        JVal u = JVal::obj();
        u.set("prompt_tokens", JVal::number((double)pt));
        u.set("completion_tokens", JVal::number((double)ct));
        u.set("total_tokens", JVal::number((double)(pt + ct)));
        return u;
      };
      if(!is_chat && legacy_prompts.size() > 1 && !stream_out){
        // Batch legacy form: one choice per prompt (OpenAI returns one
        // completion per prompt string when prompt is an array).
        JVal resp = JVal::obj();
        long long seq = g_completion_seq.fetch_add(1);
        char idb[64]; snprintf(idb, sizeof(idb), "cmpl-%lld", seq);
        resp.set("id", JVal::str(idb));
        resp.set("object", JVal::str("text_completion"));
        resp.set("created", JVal::number((double)created));
        resp.set("model", JVal::str(req_model));
        JVal ch = JVal::arr();
        int pt = 0, ct = 0;
        if(http_serial){
          for(size_t pi = 0; pi < legacy_prompts.size(); pi++){
            HttpGenOut g = serial_gen(legacy_prompts[pi], o);
            std::string txt = legacy_echo ? legacy_prompts[pi] + g.text : g.text;
            JVal c0 = JVal::obj();
            c0.set("text", JVal::str(txt));
            c0.set("index", JVal::number((double)pi));
            c0.set("finish_reason", JVal::str(g.finish));
            ch.push(c0);
            pt += g.prompt_tokens; ct += (int)g.ids.size();
          }
        } else {
          // Submit all prompts first so their decodes share M-row
          // forwards, then await each in order.
          std::vector<std::shared_ptr<HttpJob>> bjobs;
          bjobs.reserve(legacy_prompts.size());
          for(auto& pr : legacy_prompts){
            auto j = submit_text(pr, o, false);
            if(!j){ err(429, "server busy, all decode slots full", "server_error"); return; }
            bjobs.push_back(std::move(j));
          }
          for(size_t pi = 0; pi < bjobs.size(); pi++){
            HttpGenOut g = disp.await_done(bjobs[pi]);
            std::string txt = legacy_echo ? legacy_prompts[pi] + g.text : g.text;
            JVal c0 = JVal::obj();
            c0.set("text", JVal::str(txt));
            c0.set("index", JVal::number((double)pi));
            c0.set("finish_reason", JVal::str(g.finish));
            ch.push(c0);
            pt += g.prompt_tokens; ct += (int)g.ids.size();
          }
        }
        resp.set("choices", ch);
        resp.set("usage", usage_obj(pt, ct));
        rsp.send_json(200, json_dump(resp));
        return;
      }
      if(!is_chat && legacy_prompts.size() > 1 && stream_out){
        err(400, "streaming with a prompt array is not supported (send one prompt per request)",
            "invalid_request_error");
        return;
      }
      std::string prompt_in = is_chat ? prompt_text : legacy_prompts[0];
      if(stream_out){
        if(!rsp.begin_stream("text/event-stream")) return;
        long long seq = g_completion_seq.fetch_add(1);
        char idb[64];
        snprintf(idb, sizeof(idb), is_chat ? "chatcmpl-%lld" : "cmpl-%lld", seq);
        std::string cid = idb;
        std::string obj = is_chat ? "chat.completion.chunk" : "text_completion";
        int prompt_toks = 0, comp_toks = 0;
        std::string finish = "stop";
        bool first = true;
        HttpGenOpts oo = o;
        auto send_piece = [&](const std::string& piece, int){
          JVal chunk = JVal::obj();
          chunk.set("id", JVal::str(cid));
          chunk.set("object", JVal::str(obj));
          chunk.set("created", JVal::number((double)created));
          chunk.set("model", JVal::str(req_model));
          JVal ch = JVal::arr();
          JVal c0 = JVal::obj();
          c0.set("index", JVal::number(0));
          if(is_chat){
            JVal delta = JVal::obj();
            if(first) delta.set("role", JVal::str("assistant"));
            delta.set("content", JVal::str(piece));
            c0.set("delta", delta);
          } else {
            c0.set("text", JVal::str(piece));
          }
          c0.set("finish_reason", JVal::null());
          ch.push(c0);
          chunk.set("choices", ch);
          comp_toks++;
          if(!rsp.chunk("data: " + json_dump(chunk) + "\n\n")) return;
          first = false;
        };
        HttpGenOut g;
        if(http_serial){
          auto on_tok = [&](const std::string& piece, int id){ send_piece(piece, id); };
          g = serial_gen(prompt_in, oo, on_tok);
          prompt_toks = g.prompt_tokens;
          comp_toks = (int)g.ids.size();
          finish = g.finish;
        } else {
          auto j = submit_text(prompt_in, oo, true);
          if(!j){ err(429, "server busy, all decode slots full", "server_error"); return; }
          prompt_toks = j->acc.prompt_tokens;
          // Pump SSE pieces as the dispatcher produces them; a dead
          // socket cancels the job (frees its KV/SSM state next iter).
          bool dead = false;
          for(;;){
            std::pair<std::string,int> pc;
            bool have = false, is_done = false;
            {
              std::unique_lock<std::mutex> lk(j->mu);
              j->cv.wait(lk, [&]{ return !j->pieces.empty() || j->done; });
              if(!j->pieces.empty()){
                pc = std::move(j->pieces.front());
                j->pieces.pop_front();
                have = true;
              }
              is_done = j->done && j->pieces.empty();
            }
            if(have){
              size_t before = comp_toks;
              send_piece(pc.first, pc.second);
              if(comp_toks == before){ dead = true; break; } // chunk() failed
            }
            if(is_done) break;
          }
          if(dead){ j->cancelled = true; return; }
          {
            std::lock_guard<std::mutex> lk(j->mu);
            g = j->acc;
          }
          comp_toks = (int)g.ids.size();
          finish = g.finish;
        }
        // Terminal chunk (finish reason; usage only when requested).
        {
          JVal chunk = JVal::obj();
          chunk.set("id", JVal::str(cid));
          chunk.set("object", JVal::str(obj));
          chunk.set("created", JVal::number((double)created));
          chunk.set("model", JVal::str(req_model));
          JVal ch = JVal::arr();
          JVal c0 = JVal::obj();
          c0.set("index", JVal::number(0));
          if(is_chat){
            c0.set("delta", JVal::obj());
          } else {
            c0.set("text", JVal::str(""));
          }
          c0.set("finish_reason", JVal::str(finish));
          ch.push(c0);
          chunk.set("choices", ch);
          if(want_usage) chunk.set("usage", usage_obj(prompt_toks, comp_toks));
          rsp.chunk("data: " + json_dump(chunk) + "\n\n");
        }
        rsp.chunk("data: [DONE]\n\n");
        rsp.finish();
        return;
      }
      // Non-streaming.
      HttpGenOut g;
      if(http_serial){
        g = serial_gen(prompt_in, o);
      } else {
        auto j = submit_text(prompt_in, o, false);
        if(!j){ err(429, "server busy, all decode slots full", "server_error"); return; }
        g = disp.await_done(j);
      }
      if(is_chat){
        JVal resp = JVal::obj();
        long long seq = g_completion_seq.fetch_add(1);
        char idb[64]; snprintf(idb, sizeof(idb), "chatcmpl-%lld", seq);
        resp.set("id", JVal::str(idb));
        resp.set("object", JVal::str("chat.completion"));
        resp.set("created", JVal::number((double)created));
        resp.set("model", JVal::str(req_model));
        JVal ch = JVal::arr();
        JVal c0 = JVal::obj();
        c0.set("index", JVal::number(0));
        JVal msg = JVal::obj();
        msg.set("role", JVal::str("assistant"));
        msg.set("content", JVal::str(g.text));
        c0.set("message", msg);
        c0.set("finish_reason", JVal::str(g.finish));
        ch.push(c0);
        resp.set("choices", ch);
        resp.set("usage", usage_obj(g.prompt_tokens, (int)g.ids.size()));
        rsp.send_json(200, json_dump(resp));
      } else {
        JVal resp = JVal::obj();
        long long seq = g_completion_seq.fetch_add(1);
        char idb[64]; snprintf(idb, sizeof(idb), "cmpl-%lld", seq);
        resp.set("id", JVal::str(idb));
        resp.set("object", JVal::str("text_completion"));
        resp.set("created", JVal::number((double)created));
        resp.set("model", JVal::str(req_model));
        JVal ch = JVal::arr();
        JVal c0 = JVal::obj();
        std::string txt = legacy_echo ? legacy_prompts[0] + g.text : g.text;
        c0.set("text", JVal::str(txt));
        c0.set("index", JVal::number(0));
        c0.set("finish_reason", JVal::str(g.finish));
        ch.push(c0);
        resp.set("choices", ch);
        resp.set("usage", usage_obj(g.prompt_tokens, (int)g.ids.size()));
        rsp.send_json(200, json_dump(resp));
      }
      return;
    }
    err(404, std::string("unknown endpoint: ") + path, "invalid_request_error");
  });
  if(!srv.listen(host, port)){
    std::fprintf(stderr, "[intfer] FATAL: cannot bind %s:%d\n", host.c_str(), port);
    return 1;
  }
  std::fprintf(stderr, "[intfer] OpenAI-compatible server on http://%s:%d (model=%s)\n",
               (host.empty() ? "127.0.0.1" : host).c_str(), srv.bound_port(), model_id.c_str());
  std::fflush(stderr);
  // Graceful Ctrl-C: stop the accept loop so main() returns through munmap.
  static intfer_http::HttpServer* g_srv = nullptr;
  g_srv = &srv;
  std::signal(SIGINT, [](int){ if(g_srv) g_srv->stop(); });
  std::signal(SIGTERM, [](int){ if(g_srv) g_srv->stop(); });
  srv.run();
  g_srv = nullptr;
  if(!http_serial) disp.shutdown();
  return 0;
}

static void usage(const char* p){
  std::fprintf(stderr,
    "intfer — low-bit LLM inference (Stage-1 quantized .intifer, host or SYCL GPU)\n"
    "Usage: %s -m <model.intifer> [options]\n"
    "  -m PATH        .intifer model file (required)\n"
    "  -s PATH        safetensors file for embed/norms/SSM (optional: used only\n"
    "                 as fallback for tensors missing from the .intifer)\n"
    "  -p TEXT        prompt text (repeatable: N prompts run as one scheduled\n"
    "                 batch, Stage 3.2; default single prompt when given once)\n"
    "  -P FILE        read prompt from file\n"
    "  -n N           max new tokens to generate (default: 64)\n"
    "  -t TEMP        sampling temperature (default: 0.0 = greedy)\n"
    "  -k K           top-k sampling (default: 0 = off)\n"
    "  -T TOPP        top-p / nucleus sampling (default: 0.0 = off)\n"
    "  -S SEED        RNG seed (default: 0 = time)\n"
    "  --gpu          force SYCL/GPU backend (intfer::Engine)\n"
    "  --no-gpu       force CPU backend (dequant + host matmul)\n"
    "  --tp N         tensor-parallel GPU count (default: 2, the 2x B580 fleet)\n"
    "  --tokenizer P  path to tokenizer.json (default: Qwen3.5-0.8B)\n"
    "  --tokenizer-py P path to bpe_tokenize.py\n"
    "  --detok-py P   path to bpe_detokenize.py\n"
    "  --no-stream    do not stream tokens (print full output after generation)\n"
    "  --ids          also print token ids to stderr (debug)\n"
    "  --serve        JSON-lines server on stdin/stdout (Stage 3.3): read\n"
    "                 {\"prompt\":...} per line, stream {\"id\",\"token\"}/\n"
    "                 {\"id\",\"done\"} lines back as requests finish.\n"
    "                 Optional per-request keys: max_new, temperature, top_k,\n"
    "                 top_p, seed, id (defaults: -n/-t/-k/-T/-S, id assigned).\n"
    "                 Live cap: INTFER_SERVE_MAX or VRAM-aware default\n"
    "  --serve-http     OpenAI-compatible HTTP server (same engine as --serve):\n"
    "                 GET /health, GET /v1/models,\n"
    "                 POST /v1/chat/completions (stream + non-stream),\n"
    "                 POST /v1/completions (legacy prompt form + echo).\n"
    "                 Requires --port (binds 127.0.0.1; --host overrides).\n"
    "  --port N       HTTP port for --serve-http (required with it)\n"
    "  --host ADDR    HTTP bind address (default: 127.0.0.1)\n",
    p);
}

int main(int argc, char** argv){
  std::signal(SIGSEGV, segv_handler);
  std::signal(SIGABRT, segv_handler);
  // ---- Arg parse
  std::string intfer_path, st_path, prompt_file;
  std::vector<std::string> prompts; // repeatable -p (Stage 3.2 scheduler)
  std::string tokenizer_json =
      "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B/tokenizer.json";
  std::string tokenizer_py;
  std::string detok_py;
  {
    auto try_script = [](const std::string& p) -> bool {
      FILE* f = fopen(p.c_str(), "r"); if(!f) return false; fclose(f); return true;
    };
    if(try_script("scripts/bpe_tokenize.py"))       tokenizer_py = "scripts/bpe_tokenize.py";
    else if(try_script("../scripts/bpe_tokenize.py")) tokenizer_py = "../scripts/bpe_tokenize.py";
    else                                              tokenizer_py = "/var/home/fra/dev/intfer/scripts/bpe_tokenize.py";
    if(try_script("scripts/bpe_detokenize.py"))       detok_py = "scripts/bpe_detokenize.py";
    else if(try_script("../scripts/bpe_detokenize.py")) detok_py = "../scripts/bpe_detokenize.py";
    else                                                detok_py = "/var/home/fra/dev/intfer/scripts/bpe_detokenize.py";
  }
  int max_new = 64;
  float temperature = 0.0f;
  int top_k = 0;
  float top_p = 0.0f;
  uint32_t seed = 0;
  bool stream = true;
  bool print_ids = false;
  bool serve = false;
  bool serve_http = false;
  int http_port = 0;
  std::string http_host = "127.0.0.1";
  GpuMode gpu_mode = GpuMode::Auto;
  int tp_devices = 2;  // 2x Arc B580 fleet; Engine::load clamps to discovered.
  // TP default is resolved after .intifer load (auto-TP below) from the
  // measured 2026-09-14 verdict: small models -> TP=1 (~1.7x faster
  // decode; 2x submits/transfers/readbacks outweigh halved kernel time
  // at tiny shapes), 27B scale -> TP=2 (mandatory: one ~12GB B580 OOMs
  // on the ~19GB lm_head tile).
  bool tp_explicit = false;
  int i = 1;
  while(i < argc){
    std::string a = argv[i];
    if(a=="-m" && i+1<argc) intfer_path = argv[++i];
    else if(a=="-s" && i+1<argc) st_path = argv[++i];
    else if(a=="-p" && i+1<argc) prompts.push_back(argv[++i]);
    else if(a=="-P" && i+1<argc) prompt_file = argv[++i];
    else if(a=="-n" && i+1<argc) max_new = std::atoi(argv[++i]);
    else if(a=="-t" && i+1<argc) temperature = std::atof(argv[++i]);
    else if(a=="-k" && i+1<argc) top_k = std::atoi(argv[++i]);
    else if(a=="-T" && i+1<argc) top_p = std::atof(argv[++i]);
    else if(a=="-S" && i+1<argc) seed = (uint32_t)std::atoi(argv[++i]);
    else if(a=="--gpu") gpu_mode = GpuMode::ForceOn;
    else if(a=="--no-gpu") gpu_mode = GpuMode::ForceOff;
    else if(a=="--tp" && i+1<argc){ tp_devices = (size_t)std::max(1, std::atoi(argv[++i])); tp_explicit = true; }
    else if(a=="--tokenizer" && i+1<argc) tokenizer_json = argv[++i];
    else if(a=="--tokenizer-py" && i+1<argc) tokenizer_py = argv[++i];
    else if(a=="--detok-py" && i+1<argc) detok_py = argv[++i];
    else if(a=="--no-stream") stream = false;
    else if(a=="--ids") print_ids = true;
    else if(a=="--serve") serve = true;
    else if(a=="--serve-http") serve_http = true;
    else if(a=="--port" && i+1<argc) http_port = std::atoi(argv[++i]);
    else if(a=="--host" && i+1<argc) http_host = argv[++i];
    else if(a=="-h" || a=="--help"){ usage(argv[0]); return 0; }
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    ++i;
  }
  if(intfer_path.empty()){ usage(argv[0]); return 2; }
  if(serve && serve_http){
    std::fprintf(stderr, "intfer: --serve and --serve-http are mutually exclusive\n");
    return 2;
  }
  if(serve_http && http_port <= 0){
    std::fprintf(stderr, "intfer: --serve-http requires --port N\n");
    return 2;
  }
  if(!prompt_file.empty()){
    FILE* f = fopen(prompt_file.c_str(), "r");
    if(!f){ std::fprintf(stderr, "cannot read -P %s\n", prompt_file.c_str()); return 1; }
    char buf[8192]; size_t n = fread(buf,1,sizeof(buf)-1,f); fclose(f);
    prompts.emplace_back(buf, n);
  }
  if(seed==0) seed = (uint32_t)std::chrono::high_resolution_clock::now()
                            .time_since_epoch().count();
  std::mt19937 rng(seed);

  inv_freq.assign((size_t)cfg.rotary_dim/2, 0.f);
  for(int d=0; d<cfg.rotary_dim/2; d++)
    inv_freq[d] = 1.0f/powf((float)cfg.rope_theta, (2.0f*d)/(float)cfg.rotary_dim);

  // ---- Auto-detect safetensors (and model config)
  if(st_path.empty()){
    auto try_path = [&](const std::string& p) -> bool {
      return file_exists(p + "/model.safetensors-00001-of-00001.safetensors") ||
             file_exists(p + "/model.safetensors.index.json");
    };
    auto set_st = [&](const std::string& dir){
      if(file_exists(dir + "/model.safetensors.index.json"))
        st_path = dir + "/model.safetensors.index.json";
      else
        st_path = dir + "/model.safetensors-00001-of-00001.safetensors";
    };
    // Try model-name-derived paths first (from -m model.intifer name).
    // Strip chunk suffixes (-sc/-full/-noembed/-host) so a self-contained
    // file like Qwen3.8-27B-sc.intifer resolves to the Qwen3.8-27B dir;
    // without this the lookup misses and falls through to the hardcoded
    // 0.8B default (wrong dims -> crash).
    std::string model_name;
    {
      size_t sp = intfer_path.rfind('/');
      if(sp != std::string::npos) model_name = intfer_path.substr(sp+1);
      // strip .intifer suffix
      size_t dot = model_name.rfind(".intifer");
      if(dot != std::string::npos) model_name = model_name.substr(0, dot);
      for(const char* suf : {"-sc", "-full", "-noembed", "-host", "-int2"}){
        size_t s = model_name.rfind(suf);
        if(s != std::string::npos && s + strlen(suf) == model_name.size())
          model_name = model_name.substr(0, s);
      }
    }
    if(!model_name.empty() && try_path("/var/home/fra/data/models/safetensors/" + model_name))
      set_st("/var/home/fra/data/models/safetensors/" + model_name);
    else if(!model_name.empty() && try_path("test/models/" + model_name))
      set_st("test/models/" + model_name);
    else if(try_path("test/models/Qwen3.5-0.8B"))      set_st("test/models/Qwen3.5-0.8B");
    else if(try_path("test/models/Qwen3.8-27B"))  set_st("test/models/Qwen3.8-27B");
    else if(try_path("/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"))
      set_st("/var/home/fra/data/models/safetensors/Qwen3.5-0.8B");
    else if(try_path("/var/home/fra/data/models/safetensors/Qwen3.8-27B"))
      set_st("/var/home/fra/data/models/safetensors/Qwen3.8-27B");
    else if(try_path("/home/fra/data/models/safetensors/Qwen3.5-0.8B"))
      set_st("/home/fra/data/models/safetensors/Qwen3.5-0.8B");
    else if(try_path("/home/fra/data/models/safetensors/Qwen3.8-27B"))
      set_st("/home/fra/data/models/safetensors/Qwen3.8-27B");
    else { std::fprintf(stderr, "intfer: cannot auto-detect safetensors. "
                                "Pass -s <path>\n"); return 1; }
  }
  // ---- Auto-detect model config from config.json
  // Prefer config.json alongside the safetensors (correct model dims);
  // without -s (self-contained .intifer) fall back to the directory derived
  // from -m (strip .intifer suffix) and sibling safetensors dirs.
  {
    std::string config_dir;
    auto cfg_try = [&](const std::string& dir) -> bool {
      if(dir.empty()) return false;
      std::string p = dir + "/config.json";
      FILE* f = fopen(p.c_str(), "rb");
      if(!f) return false;
      fclose(f);
      config_dir = dir;
      return true;
    };
    std::string model_name;
    {
      size_t sp = intfer_path.rfind('/');
      if(sp != std::string::npos) model_name = intfer_path.substr(sp+1);
      size_t dot = model_name.rfind(".intifer");
      if(dot != std::string::npos) model_name = model_name.substr(0, dot);
    }
    if(!st_path.empty()){
      size_t sp = st_path.rfind('/');
      if(sp != std::string::npos) cfg_try(st_path.substr(0, sp));
    }
    if(config_dir.empty() && !model_name.empty()){
      // Self-contained mode: -m /path/to/Qwen3.5-0.8B-sc.intifer ->
      // config in sibling dirs (same basename minus any -sc/-full suffix).
      std::string base = model_name;
      for(const char* suf : {"-sc", "-full", "-noembed", "-int2"}){
        size_t s = base.rfind(suf);
        if(s != std::string::npos) base = base.substr(0, s);
      }
      size_t sp = intfer_path.rfind('/');
      std::string idir = (sp != std::string::npos) ? intfer_path.substr(0, sp) : ".";
      cfg_try(idir + "/" + base);
      if(config_dir.empty()) cfg_try("/var/home/fra/data/models/safetensors/" + base);
      if(config_dir.empty()) cfg_try("test/models/" + base);
    }
    ModelConfig loaded = load_model_config(config_dir);
    cfg = loaded;
    std::fprintf(stderr, "[intfer] config: model_id=%s H=%d layers=%d inter=%d heads=%d kv=%d conv_dim=%d sdim=%d (from %s)\n",
                 cfg.model_id.c_str(), cfg.H, cfg.layers, cfg.inter, cfg.n_heads, cfg.n_kv, cfg.conv_dim, cfg.sdim,
                 config_dir.empty() ? "<defaults>" : (config_dir + "/config.json").c_str());
  }
  std::fprintf(stderr, "[intfer] intfer: %s\n", intfer_path.c_str());
  std::fprintf(stderr, "[intfer] safetensors: %s\n", st_path.c_str());

  // ---- Load .intifer FIRST (Stage-1 quantized MLP + attn, plus FP16
  // passthrough host tensors in self-contained files). Borrowed mmap views
  // (zero-copy): the 22GB code region stays in page
  // cache instead of being copied into RAM. mblob must outlive main, so it
  // is munmap'd only at the very end.
  // NOTE: this moved ahead of the safetensors/config handling: config.json
  // now falls back to the directory derived from -m, and safetensors is an
  // optional fallback for tensors missing from the .intifer.
  ModelIR m; RotationSet rs;
  MmapBlob mblob = mmap_blob(intfer_path);
  if (!mblob.base || mblob.size == 0) {
    std::fprintf(stderr, "FAIL: cannot mmap %s\n", intfer_path.c_str()); return 1;
  }
  std::fprintf(stderr, "[intfer] mmap'd %zu bytes for %s (gpu_mode=%d)...\n",
               mblob.size, intfer_path.c_str(), (int)gpu_mode); std::fflush(stderr);
  std::fprintf(stderr, "[intfer] calling deserialize_intfer_borrowed...\n"); std::fflush(stderr);
  if(!deserialize_intfer_borrowed((const uint8_t*)mblob.base, mblob.size, m, rs)){
    std::fprintf(stderr, "FAIL: cannot deserialize .intifer\n"); munmap_blob(mblob); return 1;
  }
  std::fprintf(stderr, "[intfer] deserialize_intfer done: %zu tensors, %zu rotations (borrowed)\n",
               m.tensors.size(), rs.rotations.size()); std::fflush(stderr);
  // Default TP from the measured 2026-09-14 verdict: TP=1 wins ~1.7x on
  // small models (submit/transfer/readback dominate tiny shapes), TP=2 is
  // mandatory at 27B scale (single ~12GB B580 OOMs). Heuristic on the
  // quantized weight footprint: <= 4GB -> TP=1 (0.8B file is ~1.2GB),
  // above -> TP=2. Explicit --tp always wins. Logged so A/B stays honest.
  if(!tp_explicit){
    uint64_t wb = m.total_weight_bytes();
    size_t want = (wb <= 4ULL*1024*1024*1024) ? 1 : 2;
    if(want != tp_devices){
      std::fprintf(stderr, "[intfer] auto-TP: weight footprint %.2f GB -> TP=%zu (default %d; --tp overrides)\n",
                   (double)wb/(1024.0*1024.0*1024.0), want, 2);
      tp_devices = want;
    }
  }
  std::map<uint8_t, const Rotation*> rid_map;
  for(auto& R : rs.rotations){
    // The serializer does not persist is_hadamard: re-detect it empirically
    // so the online replay can use the O(n log n) FWHT path. Safe for any
    // model file (learned/dense rotations fail the check and keep O(n^2)).
    R.is_hadamard = rotation_is_hadamard(R);
    std::fprintf(stderr, "[intfer] rotation id=%u dim=%u hadamard=%d\n",
                 R.id, R.dim, (int)R.is_hadamard);
    rid_map[R.id] = &R;
  }
  rid_map_global = rid_map;
    // Populate CPU backend's per-layer QLinear tables from the deserialized IR.
  // (Done lazily after backend selection; see below.)

  // ---- Host tensors: IR FP16 passthrough FIRST, safetensors as fallback
  // Self-contained .intifer files carry embed/norms/SSM/lm_head as unrotated
  // FP16 (GGUF-style); legacy files rely on safetensors for those. -s is now
  // optional: without it, tensors missing from the IR are fatal (as before).
  // For the GPU backend, the large q/k/v/o/gate/up/down projection weights are
  // NOT loaded into RAM — the .intifer file already contains them as quantized
  // tensors and the Engine reads them on demand. Only the small "host" tensors
  // (embed_tokens, norms, SSM params) are pulled in. For the CPU backend we
  // keep loading everything, since the CPU path uses these for run_linear().
  SafetensorsReader sr;
  bool have_sr = false;
  if(!st_path.empty()){
    if(!sr.open(st_path)){ std::fprintf(stderr, "FAIL: open safetensors\n"); return 1; }
    have_sr = true;
  }

  // Determine which tensor names are "host-only" (small). The full-attention
  // / MLP projection weights are excluded for GPU mode to keep peak RAM sane
  // on large models like 27B (otherwise we OOM trying to FP32 the whole
  // ~52GB checkpoint).
  auto layer_tensor_name = [](int L, const char* suf) {
    char b[256];
    snprintf(b, sizeof(b), "model.language_model.layers.%d.%s", L, suf);
    return std::string(b);
  };
  std::set<std::string> host_only_names;
  {
    host_only_names.insert("model.language_model.embed_tokens.weight");
    host_only_names.insert("model.language_model.norm.weight");
    // Untied output projection (27B; quant skips LM_HEAD so it stays FP32).
    // Both namings seen in the wild: bare "lm_head.weight" (27B index)
    // and "model.language_model.lm_head.weight".
    host_only_names.insert("lm_head.weight");
    host_only_names.insert("model.language_model.lm_head.weight");
    for(int L=0; L<cfg.layers; L++){
      // Full-attention path (when present).
      host_only_names.insert(layer_tensor_name(L, "input_layernorm.weight"));
      host_only_names.insert(layer_tensor_name(L, "post_attention_layernorm.weight"));
      host_only_names.insert(layer_tensor_name(L, "self_attn.q_norm.weight"));
      host_only_names.insert(layer_tensor_name(L, "self_attn.k_norm.weight"));
      host_only_names.insert(layer_tensor_name(L, "linear_attn.conv1d.weight"));
      host_only_names.insert(layer_tensor_name(L, "linear_attn.A_log"));
      host_only_names.insert(layer_tensor_name(L, "linear_attn.dt_bias"));
      host_only_names.insert(layer_tensor_name(L, "linear_attn.norm.weight"));
    }
  }

  // Skip loading the large projection weights from safetensors when the
  // GPU backend owns them (Engine reads .intifer directly, on demand).
  // The CPU path always needs the full FP32 fallback available: eager
  // models serve attn/mlp from the IR, but LA layers missing from the IR
  // (0.8B) and every tensor of JIT models (27B) fall back to `sr` reads.
  // (27B safetensors are ~52GB BF16; reading on demand via the indexed
  // reader keeps peak RAM bounded — nothing is materialized up front
  // except host_only_names below.)
  // NOTE: linear_attn.in_proj_qkv/z/b/a/out_proj are NOT added to
  // host_only_names — they are served lazily by CpuBackend::run_linear()
  // which falls back to safetensors for layers missing from .intifer.
  bool skip_large_weights = false;
  std::unique_ptr<Engine> gpu_probe;
#ifdef INTFER_HAS_GPU_BACKEND
  if(gpu_mode != GpuMode::ForceOff){
    gpu_probe = Engine::load(intfer_path, tp_devices);
    if(gpu_probe){
      skip_large_weights = true;
    } else if(gpu_mode == GpuMode::ForceOn){
      std::fprintf(stderr, "FAIL: --gpu requested but Engine::load failed "
                           "(no Arc GPU / oneAPI runtime?)\n");
      return 1;
    } else {
      std::fprintf(stderr, "[intfer] no Arc GPU discovered; falling back to CPU\n");
    }
  }
#endif

  std::map<std::string, std::vector<float>> T;
  size_t total_loaded = 0;
  // 1) IR FP16 passthrough tensors first (self-contained files). Trust the
  // dequantized shape; the IR stores N/K explicitly (conv1d is [conv_dim,
  // conv_k] flattened, norms/A_log/dt_bias are [D,1]).
  size_t ir_host = 0;
  {
    std::unordered_map<std::string, size_t> ir_idx;
    for(size_t i=0;i<m.tensors.size();++i) ir_idx[m.tensors[i].name]=i;
    auto ir_get = [&](const std::string& n, std::vector<float>& out) -> bool {
      auto it = ir_idx.find(n);
      if(it==ir_idx.end()) return false;
      const auto& t = m.tensors[it->second];
      if(t.qcfg.scheme != QuantScheme::FP16) return false;
      intfer::dashq::dequant_tensor(m, t, out);
      return !out.empty();
    };
    std::vector<std::string> want = {
      "model.language_model.embed_tokens.weight",
      "model.language_model.norm.weight",
      "lm_head.weight",
      "model.language_model.lm_head.weight",
    };
    for(int L=0; L<cfg.layers; L++) for(const char* s : {
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "linear_attn.conv1d.weight", "linear_attn.A_log",
        "linear_attn.dt_bias", "linear_attn.norm.weight"}){
      char b[256];
      snprintf(b, sizeof(b), "model.language_model.layers.%d.%s", L, s);
      want.push_back(b);
    }
    for(const auto& n : want){
      std::vector<float> v;
      if(ir_get(n, v)){ total_loaded += v.size()*sizeof(float); T[n]=std::move(v); ++ir_host; }
    }
  }
  std::fprintf(stderr, "[intfer] IR host tensors loaded=%zu\n", ir_host);
  // 2) Safetensors fallback for host tensors missing from the IR.
  if(have_sr){
  for(const auto& t : sr.tensors()){
    if(T.count(t.name)) continue;  // IR wins over safetensors
    if(skip_large_weights && !host_only_names.count(t.name)) continue;
    // CPU path: load host tensors eagerly; projection weights are read
    // on demand through sr_fallback, so skip them here even on CPU
    // (27B projections would otherwise cost ~52GB of FP32 RAM).
    if(!skip_large_weights && !host_only_names.count(t.name)) continue;
    std::vector<float> v; if(!sr.read(t.name,v)){
      std::fprintf(stderr, "FAIL: read %s\n", t.name.c_str()); return 1;
    }
    total_loaded += v.size() * sizeof(float);
    T[t.name] = std::move(v);
  }
  std::fprintf(stderr, "[intfer] safetensors host tensors loaded=%zu (%.1f MB total) projections=%s\n",
               T.size()-ir_host, total_loaded / (1024.0*1024.0),
               skip_large_weights ? "GPU-owned" : "on-demand via reader");
  } else {
    std::fprintf(stderr, "[intfer] no safetensors (self-contained .intifer): host tensors=%zu\n", T.size());
  }
  // ---- Auto-detect linear-attention head counts from weight shapes ----
  // (Linear attention may use different head counts than full attention.)
  // in_proj_a: [N_V_HEADS, H], in_proj_z: [N_V_HEADS*V_HEAD_DIM, H]
  // Scans the IR first (self-contained files), then safetensors if present.
  {
    auto scan_shape = [&](const std::string& name, uint32_t N, uint32_t K){
      if(name.find("linear_attn.in_proj_a.weight") != std::string::npos){
        if(K == (uint32_t)cfg.H && N > 0)
          cfg.n_v_heads = (int)N;
      }
      if(name.find("linear_attn.norm.weight") != std::string::npos){
        if(N > 0) cfg.v_head_dim = (int)N;
      }
      if(name.find("linear_attn.in_proj_z.weight") != std::string::npos){
        if(K == (uint32_t)cfg.H){
          int total = (int)N;
          int kk = total / cfg.n_v_heads;
          if(total == cfg.n_v_heads * cfg.k_head_dim * 3)
            cfg.k_head_dim = kk;
        }
      }
      if(name.find("linear_attn.in_proj_qkv.weight") != std::string::npos){
        // qkv total = 2*K_HEAD_DIM*N_K_HEADS + N_V_HEADS*V_HEAD_DIM
        if(K == (uint32_t)cfg.H){
          int total = (int)N;
          int vdim = cfg.n_v_heads * cfg.v_head_dim;
          int qk = total - vdim;
          if(qk > 0 && qk % (2*cfg.k_head_dim) == 0)
            cfg.n_k_heads = qk / (2*cfg.k_head_dim);
        }
      }
    };
    for(const auto& t : m.tensors) scan_shape(t.name, t.N, t.K);
    if(have_sr) for(const auto& t : sr.tensors()){
      uint32_t N = t.shape.empty() ? 0 : (uint32_t)t.shape[0];
      uint32_t K = t.shape.size() < 2 ? 0 : (uint32_t)t.shape[1];
      scan_shape(t.name, N, K);
    }
  }
  cfg.build();
  // n_heads comes from config.json num_attention_heads (24 for 27B, 8 for
  // 0.8B). Do NOT derive it from q_proj rows: q_proj packs q and gate
  // per-head interleaved (2*nh*hd rows), so a blind qdim/(2*head_dim)
  // divide is the only correct inversion — and a qdim/hd divide would
  // misdetect 24 as 48 and break the q/gate split downstream.
  cfg.build();
  std::fprintf(stderr, "[intfer] LA heads: n_k=%d n_v=%d kdim=%d vdim=%d conv_dim=%d sdim=%d n_heads=%d\n",
               cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim, cfg.conv_dim, cfg.sdim, cfg.n_heads);
  auto get = [&](const std::string& n) -> const std::vector<float>& {
    auto it = T.find(n);
    if(it==T.end()){ std::fprintf(stderr, "missing tensor: %s (T.size=%zu)\n", n.c_str(), T.size()); std::exit(1); }
    return it->second;
  };

  // ---- Wire up layers (host state only)
  const float* embed = get("model.language_model.embed_tokens.weight").data();
  // Pre-scan the IR (then safetensors fallback) to find which layers are
  // linear-attention (have linear_attn.in_proj_qkv) vs full-attention
  // (have self_attn.q_proj).
  std::set<int> la_layers;
  {
    auto scan_la = [&](const std::string& name){
      if(name.find("linear_attn.in_proj_qkv") == std::string::npos) return;
      size_t p = name.find("layers.");
      if(p == std::string::npos) return;
      std::string num = name.substr(p+7);
      size_t dot = num.find('.');
      if(dot != std::string::npos) la_layers.insert(std::atoi(num.c_str()));
    };
    for(const auto& t : m.tensors) scan_la(t.name);
    if(have_sr) for(const auto& t : sr.tensors()) scan_la(t.name);
  }
  std::fprintf(stderr, "[intfer] la_layers detected: %zu layers (", la_layers.size());
  bool first = true;
  for(int L : la_layers){ if(!first) std::fprintf(stderr, ","); std::fprintf(stderr, "%d", L); first=false; }
  std::fprintf(stderr, ")\n");
  // Stage 3.2: wire one Request's layer states (host pointers into T +
  // zeroed KV/SSM state). Extracted from the inline loop so the scheduler
  // can wire one Request per prompt; single-request path calls it once.
  auto wire_request = [&](Request& req){
    req.layers.resize(cfg.layers);
    for(int L=0; L<cfg.layers; L++){
      char b[256];
      auto p=[&](const char* s){ snprintf(b,sizeof(b),
                "model.language_model.layers.%d.%s",L,s); return std::string(b); };
      Layer& ly = req.layers[L];
      // is_full: determined from the .intifer, not safetensors.
      // The .intifer contains linear_attn tensors for linear-attn layers
      // and self_attn tensors for full-attn layers.
      ly.is_full = (la_layers.count(L) == 0);
      if(L%8==0) std::fprintf(stderr, "[intfer] setting up layer %d (is_full=%d)...\n", L, ly.is_full);
      ly.in_ln    = get(p("input_layernorm.weight")).data();
      ly.post_ln  = get(p("post_attention_layernorm.weight")).data();
      if(ly.is_full){
        ly.q_norm_w = get(p("self_attn.q_norm.weight")).data();
        ly.k_norm_w = get(p("self_attn.k_norm.weight")).data();
        // Grow-on-demand: capacity is a hard cap (pos >= cap aborts at
        // append), not a pre-allocation. At cap=262144 a full eager assign
        // would be ~1GB/layer FP32 on the 27B (16 full layers); reserve()
        // avoids that while keeping appends amortized-O(1).
        ly.kcache.clear();
        ly.vcache.clear();
        ly.kcache.reserve((size_t)cfg.n_kv*(size_t)kv_capacity()*cfg.head_dim);
        ly.vcache.reserve((size_t)cfg.n_kv*(size_t)kv_capacity()*cfg.head_dim);
        // Groupwise-INT8 KV mirror (INTFER_INT8_KV=1): flag the layer;
        // the i8 buffers + scales grow lazily on first append.
        // NOTE: L0 is a linear-attn layer (is_full=0), so anchor the
        // one-time log on the first full-attention layer, whatever its
        // index, via the empty-i8-mirror test below in kv_i8_append...
        // (simpler: log here when this is the first full layer seen).
        ly.kv_int8 = kv_int8_enabled();
        // Per-token G64 Q8 KV (INTFER_KV_Q8=1) + switchable Hadamard
        // (INTFER_KV_HAD=1). Buffers grow lazily on first kv_q8_append.
        ly.kv_q8 = kv_q8_enabled();
        ly.kv_had = ly.kv_q8 && kv_had_enabled();
        if(ly.kv_int8 || ly.kv_q8){
          static bool logged = false;
          if(!logged){ std::fprintf(stderr, "[intfer] INT8 KV cache on (group %d, cap %d)%s%s\n",
                       kv_group_size(), kv_capacity(),
                       ly.kv_q8 ? " Q8-G64" : "", ly.kv_had ? "+HAD" : ""); logged = true; }
        }
      } else {
        ly.ssm_state.assign(cfg.sdim, 0.f);
        ly.conv_state.assign((size_t)cfg.conv_dim*(cfg.conv_k-1), 0.f);
      }
      if(L%8==0) std::fprintf(stderr, "[intfer] layer %d done\n", L); std::fflush(stderr);
    }
  };
  Request req;
  wire_request(req);
  const float* final_norm = get("model.language_model.norm.weight").data();
  std::fprintf(stderr, "[intfer] final_norm=%p, layers wired up\n", final_norm); std::fflush(stderr);
  // Untied lm_head (27B): separate output projection in safetensors
  // (quant skips LM_HEAD role, so it stays FP32 here). Tied models (0.8B)
  // have no lm_head tensor — step() falls back to embed rows.
  const float* lm_head = nullptr;
  {
    auto it = T.find("lm_head.weight");
    if(it == T.end()) it = T.find("model.language_model.lm_head.weight");
    if(it != T.end()){
      // Sanity: the table must hold vocab*H floats. An undersized table
      // means the reader hit a truncated shard — abort loudly instead of
      // reading OOB in the logits loop (was a SIGSEGV on 27B).
      size_t need = (size_t)cfg.vocab * cfg.H;
      if(it->second.size() < need){
        std::fprintf(stderr, "[intfer] FATAL: lm_head %s has %zu elems, need %zu\n",
                     it->first.c_str(), it->second.size(), need);
        return 1;
      }
      lm_head = it->second.data();
    }
  }
  std::fprintf(stderr, "[intfer] lm_head: %s (tied=%s)\n",
               lm_head ? "separate" : "tied-embed",
               cfg.tied_embeddings ? "yes" : "no");

  // ---- Backend selection
  std::unique_ptr<LinearBackend> backend;
#ifdef INTFER_HAS_GPU_BACKEND
  std::unique_ptr<GpuBackend> gpu_backend;
  if(gpu_probe){
    // GPU backend probe succeeded earlier — reuse the loaded Engine.
    gpu_backend = std::make_unique<GpuBackend>();
    gpu_backend->eng = std::move(gpu_probe);
    gpu_backend->sr_fallback = have_sr ? &sr : nullptr;
    gpu_backend->build_index();
    std::fprintf(stderr, "[intfer] GPU backend: %d device(s) (TP=%s)\n",
                 gpu_backend->num_devices_,
                 gpu_backend->is_distributed() ? "sharded" : "single");
    backend = std::move(gpu_backend);
  } else if(gpu_mode == GpuMode::ForceOn){
    std::fprintf(stderr, "FAIL: --gpu requested but Engine::load failed "
                         "(no Arc GPU / oneAPI runtime?)\n");
    return 1;
  } else {
    std::fprintf(stderr, "[intfer] no Arc GPU discovered; falling back to CPU\n");
  }
#endif
  if(!backend){
    auto cpu = std::make_unique<CpuBackend>();
    cpu->ir_ = &m;
    cpu->build_ir_index();
    // Eager vs JIT: dequantize the whole IR up front when it fits
    // comfortably in RAM (small models like 0.8B: ~1.2 GB FP32). The old
    // gate (`cfg.conv_dim == 0`) was always false because build() computes
    // conv_dim=6144 from header defaults, so even 0.8B took the slow JIT
    // path (re-dequant every weight per token, ~0.9 tok/s + log spam).
    // Large models (27B: ~109 GB FP32) stay on JIT dequant + free.
    uint64_t fp32_total = 0;
    for(const auto& t : m.tensors) fp32_total += (uint64_t)t.N * t.K * 4;
    bool eager = (fp32_total < 8ULL * 1024 * 1024 * 1024);
    cpu->keep_cached = eager;
    std::fprintf(stderr, "[intfer] CPU dequant plan: %s (IR FP32 %.2f GB)\n",
                 eager ? "eager" : "JIT", fp32_total / 1e9);
    if(eager){
      for(const auto& t : m.tensors){
        size_t p = t.name.find("model.language_model.layers.");
        if(p==std::string::npos) continue;
        QLinear ql;
        intfer::dashq::dequant_tensor(m, t, ql.w);
        ql.R = rid_map_global.count(t.qcfg.rotation_id) ? rid_map_global[t.qcfg.rotation_id] : nullptr;
        int L = std::atoi(t.name.c_str()+p+28);
        std::string suf = t.name.substr(p+28);
        while(!suf.empty() && suf[0]>='0' && suf[0]<='9') suf=suf.substr(1);
        if(suf.find("self_attn.q_proj")!=std::string::npos)      cpu->attn_q[L]["q"]=ql;
        else if(suf.find("self_attn.k_proj")!=std::string::npos) cpu->attn_q[L]["k"]=ql;
        else if(suf.find("self_attn.v_proj")!=std::string::npos) cpu->attn_q[L]["v"]=ql;
        else if(suf.find("self_attn.o_proj")!=std::string::npos) cpu->attn_q[L]["o"]=ql;
        else if(suf.find("mlp.gate_proj")!=std::string::npos)    cpu->mlp_q[L]["gate"]=ql;
        else if(suf.find("mlp.up_proj")!=std::string::npos)      cpu->mlp_q[L]["up"]=ql;
        else if(suf.find("mlp.down_proj")!=std::string::npos)    cpu->mlp_q[L]["down"]=ql;
        else if(suf.find("linear_attn.in_proj_qkv")!=std::string::npos) cpu->la_q[L]["qkv"]=ql;
        else if(suf.find("linear_attn.in_proj_z")!=std::string::npos)   cpu->la_q[L]["z"]=ql;
        else if(suf.find("linear_attn.in_proj_b")!=std::string::npos)   cpu->la_q[L]["b"]=ql;
        else if(suf.find("linear_attn.in_proj_a")!=std::string::npos)   cpu->la_q[L]["a"]=ql;
        else if(suf.find("linear_attn.out_proj")!=std::string::npos)   cpu->la_q[L]["out"]=ql;
      }
      std::fprintf(stderr, "[intfer] CPU backend eagerly populated: %zu attn, %zu mlp, %zu la\n",
                   cpu->attn_q.size(), cpu->mlp_q.size(), cpu->la_q.size());
    } else {
      std::fprintf(stderr, "[intfer] CPU backend using JIT dequant (IR FP32 %.2f GB)\n",
                   fp32_total / 1e9);
    }
    // (rid_map_global is a small global lookup populated right after deserialize)
    cpu->sr_fallback = have_sr ? &sr : nullptr;
    cpu->host_names_fallback = &host_only_names;
    backend = std::move(cpu);
  }
  std::fprintf(stderr, "[intfer] backend: %s\n", backend->name());
  // Full-VRAM residency (default-on): load the entire model into VRAM
  // before prefill/decode — zero evict+rebuild in steady state.
  // INTFER_PRELOAD=0 restores layer streaming.
  {
    static const char* pre_env = std::getenv("INTFER_PRELOAD");
    bool preload = !(pre_env && std::string(pre_env) == "0");
    auto gpu = dynamic_cast<GpuBackend*>(backend.get());
    if(preload && gpu && gpu->eng){
      std::fprintf(stderr, "[intfer] preloading all shards into VRAM...\n");
      std::fflush(stderr);
      gpu->eng->preload_all();
    } else {
      std::fprintf(stderr, "[intfer] preload skipped (%s)\n",
                   !preload ? "INTFER_PRELOAD=0" : "non-GPU backend");
      std::fflush(stderr);
    }
  }
  // 4b: precompute per-layer names + host pointers (loop-invariant).
  // NOTE: rotations stay live via backend->rot_for (exact per-slot
  // lookup, so resolution is independent of GEMM/cache state).
  {
    std::vector<char> is_full((size_t)cfg.layers);
    for(int L=0; L<cfg.layers; L++) is_full[(size_t)L] = req.layers[(size_t)L].is_full ? 1 : 0;
    build_layer_names(T, is_full);
  }
  // INTFER_CHECK_ROT=1: cold==warm rotation invariant (fail-safe). For every
  // layer and every tensor slot used by the forward, rot_for() before any
  // GEMM (cold: cache_R empty, IR-index/per-slot fallback) must return the
  // same pointer as after ensure_cached populates the cache (warm). This is
  // the unit check behind the 3eb4e34 fix — the old first-hit fallback
  // violated it for LA/o/down slots. Runs on the CPU backend; GPU rot_for
  // is stateless (per-tensor engine lookup), so requesting the check on a
  // non-CPU backend is a fatal error (rc=2), never a silent skip — a skip
  // would let regressions through looking green. Failures print layer/slot
  // and exit 2.
  if(std::getenv("INTFER_CHECK_ROT") != nullptr){
    auto* cpu = dynamic_cast<CpuBackend*>(backend.get());
    if(!cpu){
      std::fprintf(stderr, "[intfer] FATAL: INTFER_CHECK_ROT=1 requires the CPU backend, have '%s'\n",
                   backend ? backend->name() : "<none>");
      munmap_blob(mblob);
      return 2;
    }
    if(!cpu->check_cold_warm()){
      std::fprintf(stderr, "[intfer] FATAL: cold==warm rotation invariant violated\n");
      munmap_blob(mblob);
      return 2;
    }
  }
  std::fprintf(stderr, "[intfer] starting prefill...\n"); std::fflush(stderr);

  // ---- Tokenize the prompt(s)
  // In-process detokenizer: parse the vocab once (12MB JSON, ~50ms) so the
  // decode loop needs no python subprocess per token (~158ms each).
  if(g_bpdec.load(tokenizer_json))
    std::fprintf(stderr, "[intfer] BpDecoder: %zu ids ready\n", g_bpdec.bytes_of.size());
  else
    std::fprintf(stderr, "[intfer] BpDecoder: load failed, using python fallback\n");
  if(prompts.empty()) prompts.emplace_back(""); // default prompt below
  // Stage 3.3 server: stdin JSON-lines loop (shares sched_* mechanics with
  // the 3.2 CLI batch path above). Branches before CLI prompt handling.
  // --serve-http: same engine behind OpenAI-compatible HTTP endpoints.
  if(serve_http){
    SchedCtx sc{&T, embed, final_norm, lm_head, backend.get(),
                &detok_py, &tokenizer_json, &tokenizer_py, print_ids};
    int rc = run_serve_http(sc, wire_request, max_new, seed,
                            temperature, top_k, top_p,
                            http_host, http_port, cfg.model_id);
    munmap_blob(mblob);
    return rc;
  }
  if(serve){
    SchedCtx sc{&T, embed, final_norm, lm_head, backend.get(),
                &detok_py, &tokenizer_json, &tokenizer_py, print_ids};
    int rc = run_serve(sc, wire_request, max_new, seed,
                       temperature, top_k, top_p);
    munmap_blob(mblob);
    return rc;
  }
  // Stage 3.2 scheduler engages with >1 prompt or INTFER_SCHED=1 (forces
  // the iteration scheduler even for a single prompt; ids must match the
  // scalar loop). Otherwise the single-request path below is untouched.
  bool sched = (prompts.size() > 1) ||
               (std::getenv("INTFER_SCHED") != nullptr);
  std::vector<Request> reqs;
  std::vector<int> prompt_ids;
  if(!sched){
    if(!prompts[0].empty()){
      prompt_ids = encode_text(prompts[0], tokenizer_json, tokenizer_py);
    }
    if(prompt_ids.empty()){
      prompt_ids = {760, 6511, 314, 9338, 369};
      std::fprintf(stderr, "[intfer] (using default prompt)\n");
    }
    std::fprintf(stderr, "[intfer] prompt: %zu tokens ->", prompt_ids.size());
    for(int t : prompt_ids) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");
  } else {
    // One Request per prompt, wired independently (separate KV/SSM state).
    reqs.resize(prompts.size());
    for(size_t r=0; r<prompts.size(); r++){
      wire_request(reqs[r]);
      Request& q = reqs[r];
      if(!prompts[r].empty())
        q.prompt_ids = encode_text(prompts[r], tokenizer_json, tokenizer_py);
      if(q.prompt_ids.empty())
        q.prompt_ids = {760, 6511, 314, 9338, 369};
      q.max_new = max_new;
      q.temperature = temperature; q.top_k = top_k; q.top_p = top_p;
      q.rng = std::mt19937(seed + (uint32_t)r);
      std::fprintf(stderr, "[intfer] sched prompt %zu: %zu tokens ->", r, q.prompt_ids.size());
      for(int t : q.prompt_ids) std::fprintf(stderr, " %d", t);
      std::fprintf(stderr, "\n");
    }
  }

  // ---- Prefill + decode loop
  // NOTE: S grows as we decode (KV positions are absolute). Size `dump`
  // for prompt + max_new so the decode-time record memcpy never writes
  // out of bounds (fixed a SIGSEGV in __intel_avx_rep_memcpy).
  // Stage 3.2 scheduler path (multi-prompt or INTFER_SCHED=1), factored
  // through the same sched_* helpers the 3.3 server uses (identical
  // prefill/decode mechanics; emission here buffers per-request text).
  // Recording is skipped: dumps were write-only debug (never read back).
  // Speculative decode is single-request only (TODO3.3 path below).
  if(sched){
    auto t0 = std::chrono::high_resolution_clock::now();
    SchedCtx sc{&T, embed, final_norm, lm_head, backend.get(),
                &detok_py, &tokenizer_json, &tokenizer_py, print_ids};
    std::vector<std::vector<float>> no_dump; // empty => skip recording
    for(size_t r=0; r<reqs.size(); r++){
      Request& q = reqs[r];
      sched_prefill(q, sc, no_dump);
      if(print_ids) std::fprintf(stderr, "  [req %zu prefill] id=%d\n", r, q.token);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    int Stot = 0; for(auto& q : reqs) Stot += (int)q.prompt_ids.size();
    std::fprintf(stderr, "[intfer] sched prefill: %d tokens / %zu reqs in %.1f ms\n",
                 Stot, reqs.size(), prefill_ms);
#ifdef INTFER_PROFILE_GPU
    {
      auto p = get_gpu_profile();
      std::fprintf(stderr,
        "[GPU profile @prefill] kernel=%.1fms submit=%.1fms waitblock=%.1fms "
        "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms "
        "nk=%lu nw=%lu nr=%lu\n",
        p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
        p.rebuild_us/1000.0, p.rebuild_count,
        p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0,
        p.kernel_count, p.wait_count, p.readback_count);
      reset_gpu_profile();
    }
#endif
    auto t_d0 = std::chrono::high_resolution_clock::now();
    std::vector<float> Hb;
    std::vector<Request*> live;
    int step_i = 0, navail = (int)reqs.size();
    // Per-request budget (q.max_new): longest request decides; shorter ones
    // leave the batch when done (dynamic leave; join arrives in 3.3).
    auto emit_cli = [&](Request& q, size_t r, int next){
      if(print_ids) std::fprintf(stderr, "  [req %zu step %d] id=%d\n", r, step_i, next);
      q.output += detokenize_one(next, detok_py, tokenizer_json);
    };
    for(; navail > 0; step_i++){
      live.clear();
      for(size_t r=0; r<reqs.size(); r++) live.push_back(&reqs[r]);
      int M = sched_iteration(live, sc, Hb, emit_cli, step_i);
      if(M == 0) break;
      navail = 0;
      for(auto& q : reqs) if(!q.finished) navail++;
      if(step_i==0 || (step_i & 15) == 0)
        std::fprintf(stderr, "[intfer] sched decode step %d: batch=%d live=%d\n",
                     step_i, M, navail);
    }
    auto t_d1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_d1-t_d0).count();
    int ng = 0; for(auto& q : reqs) ng += (int)q.generated.size();
    std::fprintf(stderr, "\n[intfer] sched decode: %d tokens / %zu reqs in %.1f ms (%.2f tok/s)\n",
                 ng, reqs.size(), decode_ms, ng*1000.0/std::max(1.0, decode_ms));
#ifdef INTFER_PROFILE_GPU
    {
      auto p = get_gpu_profile();
      std::fprintf(stderr,
        "[GPU profile @decode] kernel=%.1fms submit=%.1fms waitblock=%.1fms "
        "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms "
        "nk=%lu nw=%lu nr=%lu\n",
        p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
        p.rebuild_us/1000.0, p.rebuild_count,
        p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0,
        p.kernel_count, p.wait_count, p.readback_count);
    }
#endif
    // Single-prompt scheduler runs keep the exact stdout contract (prompt
    // echo if streaming + generated text). Multi-request: buffered outputs
    // printed per request with headers (no interleaved garble).
    if(reqs.size() == 1){
      if(stream) std::fwrite(prompts[0].c_str(), 1, prompts[0].size(), stdout);
      std::fwrite(reqs[0].output.data(), 1, reqs[0].output.size(), stdout);
      std::fwrite("\n", 1, 1, stdout);
    } else {
      for(size_t r=0; r<reqs.size(); r++){
        std::printf("--- request %zu ---\n", r);
        std::fwrite(reqs[r].output.data(), 1, reqs[r].output.size(), stdout);
        std::fwrite("\n", 1, 1, stdout);
      }
    }
    munmap_blob(mblob);
    return 0;
  }
  int S = (int)prompt_ids.size();
  std::vector<std::vector<float>> dump(cfg.layers,
      std::vector<float>((size_t)(S+max_new)*cfg.H, 0.f));
  std::vector<int> generated;
  auto t0 = std::chrono::high_resolution_clock::now();
  // Chunked prefill (Phase B): INTFER_PREFILL_CHUNK=1 selects the
  // layer-outer path (one layer residency per chunk instead of per token).
  // Chunked is the default (2026-09-18: 7x prefill win, streams
  // bit-identical vs the step loop); INTFER_PREFILL_CHUNK=0 explicitly
  // forces the old M=1 step() path (parity oracle / bisection).
  static const char* chunk_env = std::getenv("INTFER_PREFILL_CHUNK");
  bool chunked = (chunk_env == nullptr || std::string(chunk_env) != "0");
  // Chunk size: tokens per layer-resident pass. 128 captures ~95% of the
  // whole-prompt gain (quiet-box chunk128 149.2 vs chunk361 156.6 tok/s)
  // while keeping serve TTFT granularity; INTFER_PREFILL_CHUNK_M overrides.
  int chunk_M = 128;
  if(const char* me = std::getenv("INTFER_PREFILL_CHUNK_M")){
    int v = std::atoi(me); if(v >= 1 && v <= 1024) chunk_M = v;
  }
  if(chunked && S > 1){
    std::vector<float> Hbuf;
    for(int c0 = 0; c0 < S; c0 += chunk_M){
      int cm = std::min(chunk_M, S - c0);
      std::vector<int> toks(prompt_ids.begin() + c0, prompt_ids.begin() + c0 + cm);
      Hbuf.assign((size_t)cm * cfg.H, 0.f);
      StepOut o = prefill_layer_outer(req, toks, c0, T, Hbuf, embed, final_norm,
                                      lm_head, backend.get(), dump);
      if(c0 + cm == S) generated.push_back(o.best);
    }
  } else {
    for(int pos=0; pos<S; pos++){
      StepOut o = step(req, prompt_ids[pos], pos, T, embed, final_norm, lm_head,
                       backend.get(), true, dump);
      if(pos==S-1) generated.push_back(o.best);
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double prefill_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
  std::fprintf(stderr, "[intfer] prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
               S, prefill_ms, S*1000.0/std::max(1.0, prefill_ms));
  if(std::getenv("INTFER_TIME_GEMM") != nullptr)
    std::fprintf(stderr, "[intfer] cpu-gemm: %.1f ms of %.1f ms prefill (%.0f%%)\n",
                 g_cpu_gemm_us/1000.0, prefill_ms, 100.0*g_cpu_gemm_us/1000.0/std::max(1.0,prefill_ms));
#ifdef INTFER_PROFILE_GPU
  {
    auto p = get_gpu_profile();
    std::fprintf(stderr,
      "[GPU profile @prefill] kernel=%.1fms submit=%.1fms waitblock=%.1fms "
      "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms "
      "nk=%lu nw=%lu nr=%lu\n",
      p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
      p.rebuild_us/1000.0, p.rebuild_count,
      p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0,
      p.kernel_count, p.wait_count, p.readback_count);
    reset_gpu_profile();
  }
#endif

  int token = generated.back();
  std::string output;
  if(stream) std::fwrite(prompts[0].c_str(), 1, prompts[0].size(), stdout);
  std::fprintf(stderr, "[intfer] prefill done, generated=%zu max_new=%d\n",
               generated.size(), max_new); std::fflush(stderr);
  auto t_d0 = std::chrono::high_resolution_clock::now();
  // Speculative decode (TODO3.3, INTFER_SPEC_K=N): free n-gram drafts from
  // the context + one batched verify forward. Greedy-only (sampling falls
  // back to the M=1 loop); default off (K=0). With greedy acceptance the
  // output is exactly equivalent to the M=1 loop (oracle-gated).
  int spec_K = 0;
  if(const char* se = std::getenv("INTFER_SPEC_K")){
    int v = std::atoi(se); if(v >= 1 && v <= 16) spec_K = v;
  }
  bool greedy = (temperature <= 0.f && top_k==0 && top_p<=0.f);
  bool use_spec = (spec_K > 0 && greedy);
  // N-gram drafter: last occurrence of the current token in the committed
  // history; the following tokens are free drafts (no model cost).
  auto ngram_drafts = [&](int tok, int K){
    std::vector<int> d;
    // Search generated first (most recent), then the prompt.
    for(int occ = (int)generated.size()-1; occ >= 0 && (int)d.size() < K; --occ){
      if(generated[(size_t)occ] == tok){
        for(int j = occ+1; j < (int)generated.size() && (int)d.size() < K; ++j)
          d.push_back(generated[(size_t)j]);
        if((int)d.size() >= K) break;
        // Ran off the end of generated: continue into nothing (generated
        // already includes everything after the prompt). Stop here.
        break;
      }
    }
    if((int)d.size() < K){
      for(int occ = (int)prompt_ids.size()-1; occ >= 0 && (int)d.size() < K; --occ){
        if(prompt_ids[(size_t)occ] == tok){
          for(int j = occ+1; j < (int)prompt_ids.size() && (int)d.size() < K; ++j)
            d.push_back(prompt_ids[(size_t)j]);
          // Prompt followers lead into generated[0..]; append those too.
          for(int j = 0; j < (int)generated.size() && (int)d.size() < K; ++j)
            d.push_back(generated[(size_t)j]);
          break;
        }
      }
    }
    if((int)d.size() > K) d.resize(K);
    return d;
  };
  auto emit_tok = [&](int step_i, int id){
    if(print_ids) std::fprintf(stderr, "  [%d] id=%d\n", step_i, id);
    generated.push_back(id);
    std::string piece = detokenize_one(id, detok_py, tokenizer_json);
    if(stream){ std::fwrite(piece.data(), 1, piece.size(), stdout); std::fflush(stdout); }
    output += piece;
  };
  // Per-layer pre-verify snapshots + verify per-row LA snapshots (reused
  // across rounds; vector assign reuses capacity after the first round).
  std::vector<std::vector<CacheSnap>> spec_rows;
  std::vector<float> Hv;
  std::vector<StepOut> outs;
  for(int step_i = 0; step_i < max_new;){
    std::fprintf(stderr, "[intfer] decode step %d: token=%d pos=%d\n",
                 step_i, token, S+step_i); std::fflush(stderr);
    bool spec_round = false;
    if(use_spec){
      int p = S + step_i;
      std::vector<int> drafts = ngram_drafts(token, spec_K);
      if(!drafts.empty()){
        // Verify rows: [token @ p, d_1 @ p+1, ..., d_K @ p+K].
        // Drafts were never run (free n-gram tokens), so live state is
        // exactly the pre-verify state: no rollback needed before verify.
        int K = (int)drafts.size();
        std::vector<int> toks; toks.reserve(K+1);
        toks.push_back(token);
        for(int d : drafts) toks.push_back(d);
        Hv.assign((size_t)(K+1)*cfg.H, 0.f);
        outs.clear();
        if((int)spec_rows.size() != cfg.layers) spec_rows.resize(cfg.layers);
        prefill_layer_outer(req, toks, p, T, Hv, embed, final_norm, lm_head,
                            backend.get(), dump, &outs, &spec_rows);
        // outs[i].best predicts position p+i+1. Accept d_i iff
        // outs[i-1].best == d_i.
        int a = 0;
        while(a < K && outs[(size_t)a].best == drafts[(size_t)a]) a++;
        int c = outs[(size_t)a].best; // corrected (a<K) or bonus (a==K)
        // Emission list: accepted drafts up to and including the first EOS,
        // plus c iff no EOS was emitted.
        std::vector<int> emit;
        for(int i=0; i<a; i++){
          emit.push_back(drafts[(size_t)i]);
          if(drafts[(size_t)i] == 248044 || drafts[(size_t)i] == 248046) break;
        }
        bool stopped = !emit.empty() &&
          (emit.back() == 248044 || emit.back() == 248046);
        if(!stopped) emit.push_back(c);
        // Honor -n: a verify round can emit up to K+1 tokens, overshooting
        // max_new (the loop only checks step_i < max_new at round start).
        // Truncate BEFORE the state restore so pend matches what is emitted
        // (a popped bonus/correction is simply recomputed next round; a
        // popped EOS just defers stopping to the budget cap). Keep >= 1.
        {
          int budget = max_new - step_i;
          if(budget < 1) budget = 1;
          while((int)emit.size() > budget) emit.pop_back();
          stopped = !emit.empty() &&
            (emit.back() == 248044 || emit.back() == 248046);
        }
        // Truncate state to the last emitted position (pend). Verify rows
        // are u_0=token @ p, u_{i+1}=d_i @ p+i+1; post-row-r snapshots hold
        // state through position p+r. Last emitted position:
        // accepted drafts d_0..d_{e-1} then maybe c, where e = #drafts
        // emitted. pend = p + e (+0 draft rows are u_1..; token row is u_0).
        int n_draft_emit = (int)emit.size() - (stopped ? 0 : 1);
        int pend = p + n_draft_emit; // c sits at p+a; drafts at p..p+a-1
        for(int L=0; L<cfg.layers; L++){
          Layer& ly = req.layers[(size_t)L];
          if(ly.is_full){
            size_t kwn = (size_t)(pend+1)*cfg.n_kv*cfg.head_dim;
            if(ly.kcache.size() != kwn) ly.kcache.resize(kwn);
            if(ly.vcache.size() != kwn) ly.vcache.resize(kwn);
          } else {
            // Post-row-r snapshot: r = pend - p (verify row index).
            const CacheSnap& sn = spec_rows[(size_t)L][(size_t)(pend-p)];
            ly.conv_state = sn.conv;
            ly.ssm_state = sn.ssm;
          }
        }
#ifdef INTFER_HAS_GPU_BACKEND
        // Device-side SSM (INTFER_SSM_GPU=1): the restore above replaced
        // host ssm_state vectors under the resident device states — push
        // the mirrors back to the device so the next step() sees the
        // rolled-back state. No-op when the device path was never used.
        if(std::getenv("INTFER_SSM_GPU") != nullptr) {
          if(auto* gpu_rb = dynamic_cast<GpuBackend*>(backend.get()))
            gpu_rb->ssm_resync_all();
        }
#endif
        for(size_t i=0; i<emit.size(); i++) emit_tok(step_i+(int)i, emit[i]);
        step_i += (int)emit.size();
        token = generated.back();
        spec_round = true;
        if(stopped) break;
      }
    }
    if(!spec_round){
      StepOut o = step(req, token, S+step_i, T, embed, final_norm, lm_head,
                       backend.get(), false, dump);
      int next;
      if(greedy){
        next = o.best;
      } else {
        next = sample(o.logits, top_k, top_p, temperature, rng);
      }
      emit_tok(step_i, next);
      token = next;
      step_i++;
      if(next == 248044 || next == 248046) break;
    }
  }
  auto t_d1 = std::chrono::high_resolution_clock::now();
  double decode_ms = std::chrono::duration<double, std::milli>(t_d1-t_d0).count();
  int ng = (int)generated.size();
  std::fprintf(stderr, "\n[intfer] decode: %d tokens in %.1f ms (%.2f tok/s)\n",
               ng, decode_ms, ng*1000.0/std::max(1.0, decode_ms));
#ifdef INTFER_PROFILE_GPU
  {
    auto p = get_gpu_profile();
    std::fprintf(stderr,
      "[GPU profile @decode] kernel=%.1fms submit=%.1fms waitblock=%.1fms "
      "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms "
      "nk=%lu nw=%lu nr=%lu\n",
      p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
      p.rebuild_us/1000.0, p.rebuild_count,
      p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0,
      p.kernel_count, p.wait_count, p.readback_count);
  }
#endif
#ifdef INTFER_PROFILE_GPU
  {
    auto p = get_gpu_profile();
    std::fprintf(stderr,
      "\n[GPU profile]\n"
      "  total_kernel_us   : %.1f (%lu launches)\n"
      "  total_wait_us     : %.1f (%lu waits)\n"
      "  total_copy_us     : %.1f (%lu copies)\n"
      "  total_readback_us : %.1f (%lu readbacks)\n"
      "  total_usm_write_us: %.1f\n"
      "  total_submit_us   : %.1f\n"
      "  total_waitblock_us: %.1f\n"
      "  total_rebuild_us  : %.1f (%lu rebuilds)\n"
      "  peak_wait_depth   : %lu\n"
      "  kernel %% of decode : %.1f%%\n",
      p.kernel_us, p.kernel_count, p.wait_us, p.wait_count,
      p.copy_us, p.copy_count, p.readback_us, p.readback_count,
      p.usm_write_us, p.submit_us, p.waitblock_us, p.rebuild_us,
      p.rebuild_count,
      p.peak_wait_depth,
      100.0 * (p.kernel_us / 1000.0) / std::max(1.0, decode_ms));
  }
#endif
  if(!stream) std::fwrite(output.data(), 1, output.size(), stdout);
  std::fwrite("\n", 1, 1, stdout);
  munmap_blob(mblob); // release the borrowed payload mapping at the very end
  return 0;
}
