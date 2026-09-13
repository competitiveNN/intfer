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
#include <memory>
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
static void rmsnorm(const float* x, const float* w, float* out, int dim){
  float v=0; for(int i=0;i<dim;i++) v+=x[i]*x[i];
  v = 1.0f/sqrtf(v/dim + cfg.rms_eps);
  for(int i=0;i<dim;i++) out[i] = (x[i]*v)*(1.0f+w[i]);
}
static void l2norm(const float* x, float* out, int dim){
  float s=0; for(int i=0;i<dim;i++) s+=x[i]*x[i];
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
    for(int i=0;i<N;i++){ const float* wr=W+(size_t)i*cache_K[name]; float s=0;
      for(int k=0;k<K;k++) s+=wr[k]*xp[k]; y[i]=s; }
    if(!keep_cached){
      // JIT path (27B): free the weight right after its GEMM so peak RAM
      // stays at ~one tensor instead of the full ~109 GB FP32 model.
      cache_w.erase(it);
      cache_R.erase(name);
    }
    return y;
  }
  void run_linear_into(const std::string& name,
                       const std::vector<float>& x,
                       std::vector<float>& out) override {
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
    for(int i=0;i<N;i++){ const float* wr=W+(size_t)i*cache_K[name]; float s=0;
      for(int k=0;k<K;k++) s+=wr[k]*xp[k]; yp[i]=s; }
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
    // Fall back to per-layer maps.
    size_t p = name.find("model.language_model.layers.");
    if(p == std::string::npos) return nullptr;
    int L = std::atoi(name.c_str()+p+28);
    std::string suf = name.substr(p+28);
    while(!suf.empty() && suf[0]>='0' && suf[0]<='9') suf=suf.substr(1);
    if(suf.find("self_attn.q_proj")!=std::string::npos || suf.find("self_attn.k_proj")!=std::string::npos ||
       suf.find("self_attn.v_proj")!=std::string::npos || suf.find("self_attn.o_proj")!=std::string::npos ||
       suf.find("mlp.gate_proj")!=std::string::npos || suf.find("mlp.up_proj")!=std::string::npos ||
       suf.find("mlp.down_proj")!=std::string::npos || suf.find("linear_attn.in_proj_qkv")!=std::string::npos ||
       suf.find("linear_attn.in_proj_z")!=std::string::npos || suf.find("linear_attn.in_proj_b")!=std::string::npos ||
       suf.find("linear_attn.in_proj_a")!=std::string::npos || suf.find("linear_attn.out_proj")!=std::string::npos){
      // Check all maps.
      auto check = [&](const std::map<int, std::map<std::string, QLinear>>& mp) -> const Rotation* {
        auto itL = mp.find(L);
        if(itL == mp.end()) return nullptr;
        for(auto& kv : itL->second) if(kv.second.R) return kv.second.R;
        return nullptr;
      };
      if(const Rotation* r = check(attn_q)) return r;
      if(const Rotation* r = check(mlp_q)) return r;
      if(const Rotation* r = check(la_q)) return r;
    }
    return nullptr;
  }
};

#ifdef INTFER_HAS_GPU_BACKEND
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
      // {} -> zero logits for this token).
      for(size_t dd=0; dd<x_devs.size(); ++dd)
        if(x_devs[dd]) sycl::free(x_devs[dd], eng->device_queue((int)dd));
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
          if(t<lm_head_c_dev_[d].size() && lm_head_c_dev_[d][t]){
            sycl::free(lm_head_c_dev_[d][t], eng->device_queue((int)d));
            lm_head_c_dev_[d][t] = nullptr;
          }
        }
        for(size_t t=0; t<lm_head_w_dev_[d].size(); ++t){
          if(lm_head_w_dev_[d][t]){
            sycl::free(lm_head_w_dev_[d][t], eng->device_queue((int)d));
            lm_head_w_dev_[d][t] = nullptr;
          }
        }
      }
      lm_head_q_dev_.clear(); lm_head_s_dev_.clear();
      lm_head_w_dev_.clear(); lm_head_c_dev_.clear();
      lm_head_ranges_.clear(); lm_head_tiles_.clear();
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
        lm_head_c_dev_.resize(ndev);
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
      uint32_t tile_rows = (uint32_t)std::max<size_t>(
          1, (128u * 1024u * 1024u) / ((size_t)K * sizeof(sycl::half)));
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
    // x_dev frees are deferred until after the kernel events complete
    // (sycl::free acts on the host immediately; freeing in-loop is a
    // use-after-free race against the not-yet-run kernel).
    static const bool trace_lm = (std::getenv("INTFER_TRACE_LMHEAD") != nullptr);
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
      // x_dev alloc can throw under VRAM pressure (submit path throws
      // UR_RESULT_ERROR_OUT_OF_RESOURCES too — see below): degrade to {}
      // via fail_reset instead of SIGABRT.
      sycl::half* x_dev = nullptr;
      try {
        x_dev = static_cast<sycl::half*>(
            sycl::malloc_device((size_t)K * sizeof(sycl::half), q.get_device(), q.get_context()));
      } catch (...) { x_dev = nullptr; }
      if(!x_dev){ fail_reset(); return {}; }
      x_devs.push_back(x_dev);
      q.memcpy(x_dev, x_fp16.data(), (size_t)K * sizeof(sycl::half));
      // One GEMM per resident tile; tiles concatenate in row order.
      // The submit itself can throw when the device is full — catch per
      // tile so a shortfall degrades to {} (fail_reset frees everything
      // including x_devs allocated so far) instead of terminate.
      for(size_t t=0; t<lm_head_tiles_[d].size(); ++t){
        uint32_t ts = lm_head_tiles_[d][t].first, te = lm_head_tiles_[d][t].second;
        uint32_t Nlt = te - ts;
        sycl::event evt;
        try {
          evt = force_fp16
              ? gemm::quant_gemm_f16(q, x_dev, lm_head_w_dev_[d][t],
                                     lm_head_c_dev_[d][t], 1, Nlt, K)
              : gemm::quant_gemm_f16_int8w(q, x_dev, lm_head_q_dev_[d][t],
                                           lm_head_s_dev_[d][t],
                                           lm_head_c_dev_[d][t], 1, Nlt, K);
        } catch (...) {
          std::fprintf(stderr, "[intfer] FATAL: lm_head GEMM submit failed "
                       "on device %zu tile %zu (VRAM pressure?)\n", d, t);
          std::fflush(stderr);
          fail_reset();
          return {};
        }
        q.memcpy(out_seg[d].data() + (ts - start), lm_head_c_dev_[d][t],
                 (size_t)Nlt * sizeof(sycl::half), evt);
        // Load-bearing: waiting only on the kernel event leaves the D2H
        // readback in flight while the host consumes out_seg (and frees x_dev).
        // Under full-model device pressure that race kills the device; the
        // queue wait makes host consumption safe.
        q.wait();
        try {
          auto tss = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
          auto tee = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
          lm_kernel_us += (double)(tee - tss) / 1000.0;
        } catch (...) {}
        evts.push_back(evt);
      }
    }
    for(auto& e : evts) e.wait();
    for(size_t d=0; d<eng->num_devices(); ++d)
      sycl::free(x_devs[d], eng->device_queue((int)d));
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
  // FP16 path (INTFER_LMHEAD_FP16=1, bisection only): per-tile FP16 weights.
  std::vector<std::vector<sycl::half*>> lm_head_w_dev_;  // fp16 weights (fp16 path)
  std::vector<std::vector<std::int8_t*>> lm_head_q_dev_;  // int8 codes (int8 path)
  std::vector<std::vector<sycl::half*>> lm_head_s_dev_;  // fp16 row scales (int8 path)
  std::vector<std::vector<sycl::half*>> lm_head_c_dev_;  // fp16 outputs (both)
  std::vector<std::pair<uint32_t,uint32_t>> lm_head_ranges_;  // TP row shard per device
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> lm_head_tiles_;  // upload/GEMM tiles per device
  uint32_t lm_head_N_ = 0, lm_head_K_ = 0;
  const Rotation* rot_for(const std::string& name) override {
    if(!eng) return nullptr;
    auto it = ti_cache.find(name);
    if(it == ti_cache.end()) return nullptr;
    return eng->rotation_for(it->second);
  }
};
#endif

// Per-layer precomputed tensor names + host pointers (TODO 4b). step()
// used to rebuild every "model.language_model.layers.L.slot" string via
// snprintf (~366/token) plus per-call get() map finds for SSM params.
// All loop-invariant: build once after backend selection, use by
// reference. ROTATIONS ARE DELIBERATELY NOT HOISTED: CpuBackend::rot_for
// is state-dependent — cold (pre-GEMM) it falls through to a per-layer
// first-hit scan, warm (post-ensure_cached) it returns the per-slot
// rotation from cache_R. Resolving up front froze wrong pointers and
// broke output (854 -> 13104 on prefill). rot_for itself is cheap
// (a few map finds); the snprintf+malloc was the cost.
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
};

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
// own KV/SSM state).
struct BatchItem { Request* req; int token; int pos; };

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
      std::vector<float>& qp = req.scratch.qp;
      backend->run_linear_into(ln.q, xq, qp);
      ckpt("attn.q_proj", L, qp);
      // q/gate split. Both known models pack q+gate per-head interleaved:
      // rows = 2*n_heads*head_dim (0.8B: 2*8*256=4096; 27B: 2*24*256=12288;
      // verified against safetensors shapes). First head_dim rows per head
      // are q, second are gate. The !packed_gate branch is a defensive
      // fallback for q-only layouts (o_proj then takes attn concat gate).
      // Detect by comparing qp.size() against 2*n_heads*head_dim.
      size_t qrows = (size_t)cfg.n_heads * cfg.head_dim;
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
      const Rotation* Rk = backend->rot_for(ln.k);
      const Rotation* Rv = backend->rot_for(ln.v);
      std::vector<float>& xk = req.scratch.xk;
      std::vector<float>& xv = req.scratch.xv;
      // Fusion: rot_apply fully overwrites; pre-copy only for !R path.
      if(Rk) rot_apply(normed.data(), *Rk, xk.data());
      else for(int i=0;i<cfg.H;i++) xk[i] = normed[i];
      if(Rv) rot_apply(normed.data(), *Rv, xv.data());
      else for(int i=0;i<cfg.H;i++) xv[i] = normed[i];
      std::vector<float>& k = req.scratch.k;
      std::vector<float>& v = req.scratch.v;
      backend->run_linear_into(ln.k, xk, k);
      backend->run_linear_into(ln.v, xv, v);
      ckpt("attn.k_proj", L, k); ckpt("attn.v_proj", L, v);
      for(int hh=0; hh<cfg.n_kv; hh++){
        // Fusion: in-place rmsnorm is bit-identical (see q_norm above).
        rmsnorm(&k[hh*cfg.head_dim], ly.k_norm_w, &k[hh*cfg.head_dim], cfg.head_dim);
        rope(&k[hh*cfg.head_dim], pos);
      }
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
          const float* kh=&ly.kcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
          const float* vh=&ly.vcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
        backend->run_linear_into(ln.qkv, xqkv, mqkv);
        ckpt("la.qkv", L, mqkv);
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
        backend->run_linear_into(ln.z, xz, z);
        backend->run_linear_into(ln.b, xb, b);
        backend->run_linear_into(ln.a, xa, a);
        ckpt("la.z", L, z); ckpt("la.b", L, b); ckpt("la.a", L, a);
        std::vector<float>& g = req.scratch.g;
      for(int hh=0; hh<cfg.n_v_heads; hh++){
        float sp=log1pf(expf(a[hh]+dt_bias[hh])); g[hh]=-expf(A_log[hh])*sp;
      }
        std::vector<float>& core = req.scratch.core;
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
    backend->run_linear_into(ln.gate, rin, mg);
    ckpt("mlp.gate", L, mg);
    std::vector<float>& mu = req.scratch.mu;
    backend->run_linear_into(ln.up,   rin, mu);
    ckpt("mlp.up", L, mu);
    for(int i=0;i<cfg.inter;i++) mg[i] = silu(mg[i])*mu[i];
    std::vector<float>& md = req.scratch.md;
    backend->run_linear_into(ln.down, mg, md);
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
  // per tensor name (SpinQuant, fixed after warmup), so one lookup per
  // layer is enough; rows differ only in activations.
  static const bool batch_gemm =
      (std::getenv("INTFER_BATCH_GEMM") != nullptr);
  bool use_batch = batch_gemm && backend && backend->is_distributed();
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
        std::vector<float>& k = req.scratch.k;
        std::vector<float>& v = req.scratch.v;
        backend->run_linear_into(ln.k, xk, k);
        backend->run_linear_into(ln.v, xv, v);
        for(int hh=0; hh<cfg.n_kv; hh++){
          rmsnorm(&k[hh*cfg.head_dim], ly.k_norm_w, &k[hh*cfg.head_dim], cfg.head_dim);
          rope(&k[hh*cfg.head_dim], pos);
        }
        int base = pos*(cfg.n_kv*cfg.head_dim);
        size_t need = (size_t)cfg.n_kv*cfg.head_dim;
        if(k.size() >= need) memcpy(&ly.kcache[base], k.data(), need*sizeof(float));
        if(v.size() >= need) memcpy(&ly.vcache[base], v.data(), need*sizeof(float));
        std::vector<float>& attn = req.scratch.attn;
        const int G = cfg.n_heads / cfg.n_kv;
        float* gs = req.scratch.gscores_for(G, pos);
        for(int kvh=0; kvh<cfg.n_kv; kvh++){
          float mx[64]; for(int j=0;j<G;j++) mx[j]=-1e30f;
          for(int t=0;t<=pos;t++){
            const float* kh=&ly.kcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
            const float* vh=&ly.vcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
        backend->run_linear_into(ln.z, xz, z);
        backend->run_linear_into(ln.b, xb, b);
        backend->run_linear_into(ln.a, xa, a);
        std::vector<float>& g = req.scratch.g;
        for(int hh=0; hh<cfg.n_v_heads; hh++){
          float sp=log1pf(expf(a[hh]+dt_bias[hh])); g[hh]=-expf(A_log[hh])*sp;
        }
        std::vector<float>& core = req.scratch.core;
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
        // stays empty when INTFER_BATCH_GEMM is unset and the verifier's
        // restore indexes an empty vector (SIGSEGV in vector::operator=).
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
      backend->run_linear_into(ln.gate, rin, mg);
      std::vector<float>& mu = req.scratch.mu;
      backend->run_linear_into(ln.up,   rin, mu);
      for(int i=0;i<cfg.inter;i++) mg[i] = silu(mg[i])*mu[i];
      std::vector<float>& md = req.scratch.md;
      backend->run_linear_into(ln.down, mg, md);
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
      backend->run_linear_batch(ln.k, c_kv_in, c_k, (uint32_t)M);
      backend->run_linear_batch(ln.v, c_v_in, c_v, (uint32_t)M);
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
        int base = pos*(cfg.n_kv*cfg.head_dim);
        memcpy(&ly.kcache[base], kr, kvrows*sizeof(float));
        memcpy(&ly.vcache[base], vr, kvrows*sizeof(float));
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
              const float* kh=&ly.kcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
              const float* vh=&ly.vcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
      backend->run_linear_batch(ln.gate, c_rin, c_mg, (uint32_t)M);
      backend->run_linear_batch(ln.up,   c_rin, c_mu, (uint32_t)M);
      for(int m=0; m<M; m++){
        float* mgr = &c_mg[(size_t)m*cfg.inter];
        const float* mur = &c_mu[(size_t)m*cfg.inter];
        for(int i=0;i<cfg.inter;i++) mgr[i] = silu(mgr[i])*mur[i];
      }
      backend->run_linear_batch(ln.down, c_mg, c_md, (uint32_t)M);
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
      backend->run_linear_batch(ln.z, c_z_in, c_z, (uint32_t)M);
      backend->run_linear_batch(ln.b, c_b_in, c_b, (uint32_t)M);
      backend->run_linear_batch(ln.a, c_x, c_a, (uint32_t)M);
      need_rows(c_ncore, zvw);
      // Per-row conv + SSM recurrence in token order (stateful).
      std::vector<float> conv_out(cfg.conv_dim);
      std::vector<float> kv_mem(cfg.v_head_dim), delta(cfg.v_head_dim);
      std::vector<float> qn(cfg.k_head_dim), kn(cfg.k_head_dim);
      std::vector<float> core(zvw), gg(cfg.n_v_heads);
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
      backend->run_linear_batch(ln.gate, c_rin_la, c_mg, (uint32_t)M);
      backend->run_linear_batch(ln.up,   c_rin_la, c_mu, (uint32_t)M);
      for(int m=0; m<M; m++){
        float* mgr = &c_mg[(size_t)m*cfg.inter];
        const float* mur = &c_mu[(size_t)m*cfg.inter];
        for(int i=0;i<cfg.inter;i++) mgr[i] = silu(mgr[i])*mur[i];
      }
      backend->run_linear_batch(ln.down, c_mg, c_md_la, (uint32_t)M);
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
    const float* row = embed + (size_t)items[(size_t)i].token*Hdim;
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
        int base = pos*(cfg.n_kv*cfg.head_dim);
        memcpy(&ly.kcache[base], kr, kvrows*sizeof(float));
        memcpy(&ly.vcache[base], vr, kvrows*sizeof(float));
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
              const float* kh=&ly.kcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
              const float* vh=&ly.vcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
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
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly0.post_ln, &b_rn2[(size_t)m*Hdim], Hdim);
      const Rotation* Rg = backend->rot_for(ln.gate);
      for(int m=0; m<M; m++){
        const float* r2 = &b_rn2[(size_t)m*Hdim];
        float* ri = &b_rin[(size_t)m*Hdim];
        if(Rg) rot_apply(r2, *Rg, ri); else memcpy(ri, r2, (size_t)Hdim*sizeof(float));
      }
      backend->run_linear_batch(ln.gate, b_rin, c_mg, (uint32_t)M);
      backend->run_linear_batch(ln.up,   b_rin, c_mu, (uint32_t)M);
      for(int m=0; m<M; m++){
        float* mgr = &c_mg[(size_t)m*cfg.inter];
        const float* mur = &c_mu[(size_t)m*cfg.inter];
        for(int i=0;i<cfg.inter;i++) mgr[i] = silu(mgr[i])*mur[i];
      }
      backend->run_linear_batch(ln.down, c_mg, b_md, (uint32_t)M);
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
      need_rows(c_y, Hdim);
      backend->run_linear_batch(ln.la_out, b_ncore, c_y, (uint32_t)M);
      const Rotation* Ro_la = backend->rot_for(ln.la_out);
      need_rows(c_mg, cfg.inter); need_rows(c_mu, cfg.inter);
      static thread_local std::vector<float> b_rn2_la, b_rin_la, b_md_la;
      need_rows(b_rn2_la, Hdim); need_rows(b_rin_la, Hdim); need_rows(b_md_la, Hdim);
      for(int m=0; m<M; m++){
        float* hr = &H[(size_t)m*Hdim];
        float* orw = &c_y[(size_t)m*Hdim];
        float* tmp = &b_rin_la[(size_t)m*Hdim];
        float* dst = &b_rn2_la[(size_t)m*Hdim];
        memcpy(tmp, hr, (size_t)Hdim*sizeof(float));
        if(Ro_la) rot_left(orw, *Ro_la, dst);
        else memcpy(dst, orw, (size_t)Hdim*sizeof(float));
        for(int i=0;i<Hdim;i++) hr[i] = dst[i] + tmp[i];
      }
      for(int m=0; m<M; m++)
        rmsnorm(&H[(size_t)m*Hdim], ly0.post_ln, &b_rn2_la[(size_t)m*Hdim], Hdim);
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
  std::vector<float>& fn = rar.scratch.fn;
  std::vector<float>& logits = rar.scratch.logits;
  const float* head = lm_head ? lm_head : embed;
  auto gpu = backend ? dynamic_cast<GpuBackend*>(backend) : nullptr;
  for(int m=0; m<M; m++){
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
// Prefill one request into its own KV/SSM state (sequential; chunked when
// INTFER_PREFILL_CHUNK is set). Appends the first generated id; sets
// token/pos. dump is the request's private record buffer (may be empty to
// skip recording).
static void sched_prefill(Request& q, SchedCtx& c, std::vector<std::vector<float>>& dump){
  int Sr = (int)q.prompt_ids.size();
  static const char* chunk_env = std::getenv("INTFER_PREFILL_CHUNK");
  bool chunked = (chunk_env != nullptr && std::string(chunk_env) != "0");
  int chunk_M = 8;
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
// (default 8: each Request holds full KV/SSM state, ~hundreds of MB on 27B);
// prefill each admission immediately (interleaved with live decodes), then
// run one sched_iteration over all live. EOF drains live to completion.
// Returns 0 on clean EOF drain (munmap left to the caller).
static int run_serve(SchedCtx& c,
                     const std::function<void(Request&)>& wire,
                     int dflt_max_new, uint32_t seed,
                     float dflt_temp, int dflt_top_k, float dflt_top_p){
  std::vector<std::unique_ptr<Request>> pool; // all admitted (live + done)
  std::vector<Request*> live;                 // points into pool (unfinished)
  std::deque<std::string> pending;            // FIFO of unadmitted lines
  // Live cap: VRAM-aware default from per-request state bytes (KV caches +
  // scratch arena + gscores, all host-side FP32; GPU weights are shared).
  // Budget 8GB host for request state; INTFER_SERVE_MAX still overrides.
  // Prefill input rows reuse one shared Hbuf (not per-request).
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
  // Admit one parsed request: tokenize, wire state, prefill immediately
  // (B replaces this with queued chunked prefill; until then the prefill
  // runs here, interleaved between decode iterations).
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
    sched_prefill(q, c, no_dump);
    if(c.print_ids)
      std::fprintf(stderr, "  [serve id=%ld prefill] id=%d\n", q.id, q.token);
    std::fprintf(stderr, "[intfer] serve admit id=%ld prompt=%zu max_new=%d live=%zu\n",
                 q.id, q.prompt_ids.size(), q.max_new, live.size()+1);
    live.push_back(rq.get());
    pool.push_back(std::move(rq));
  };
  std::string bufline;
  while(!eof || !pending.empty() || !live.empty()){
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
    // --- One decode iteration over all live (dynamic leave inside).
    if(!live.empty()){
      int M = sched_iteration(live, c, Hb, emit_stream, step_i++);
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
      if(M == 0 && eof && pending.empty()) break;
    } else if(eof && pending.empty()){
      break;
    } else if(live.empty() && pending.empty() && !eof){
      // Nothing to do: block for the next line (avoids a hot spin when
      // idle with the pipe open but no data).
      if(!std::getline(std::cin, bufline)){ break; }
      if(!bufline.empty() && bufline.back() == '\r') bufline.pop_back();
      if(!bufline.empty()) pending.push_back(bufline);
    }
  }
  return 0;
}

// ============================================================================
// CLI
// ============================================================================
enum class GpuMode { Auto, ForceOn, ForceOff };

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
    "                 Live cap: INTFER_SERVE_MAX or VRAM-aware default\n",
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
  GpuMode gpu_mode = GpuMode::Auto;
  size_t tp_devices = 2;  // 2x Arc B580 fleet; Engine::load clamps to discovered
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
    else if(a=="--tp" && i+1<argc) tp_devices = (size_t)std::max(1, std::atoi(argv[++i]));
    else if(a=="--tokenizer" && i+1<argc) tokenizer_json = argv[++i];
    else if(a=="--tokenizer-py" && i+1<argc) tokenizer_py = argv[++i];
    else if(a=="--detok-py" && i+1<argc) detok_py = argv[++i];
    else if(a=="--no-stream") stream = false;
    else if(a=="--ids") print_ids = true;
    else if(a=="--serve") serve = true;
    else if(a=="-h" || a=="--help"){ usage(argv[0]); return 0; }
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    ++i;
  }
  if(intfer_path.empty()){ usage(argv[0]); return 2; }
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
        ly.kcache.assign((size_t)cfg.n_kv*1024*cfg.head_dim, 0.f);
        ly.vcache.assign((size_t)cfg.n_kv*1024*cfg.head_dim, 0.f);
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
  // NOTE: rotations stay live via backend->rot_for (state-dependent,
  // must not be resolved before the first GEMM warms the cache).
  {
    std::vector<char> is_full((size_t)cfg.layers);
    for(int L=0; L<cfg.layers; L++) is_full[(size_t)L] = req.layers[(size_t)L].is_full ? 1 : 0;
    build_layer_names(T, is_full);
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
        "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms\n",
        p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
        p.rebuild_us/1000.0, p.rebuild_count,
        p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0);
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
        "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms\n",
        p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
        p.rebuild_us/1000.0, p.rebuild_count,
        p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0);
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
  // Default (unset) keeps the M=1 step() loop as the parity oracle.
  // INTFER_PREFILL_CHUNK=0 explicitly forces the old path.
  static const char* chunk_env = std::getenv("INTFER_PREFILL_CHUNK");
  bool chunked = (chunk_env != nullptr && std::string(chunk_env) != "0");
  // Chunk size: tokens per layer-resident pass. Small default keeps the
  // parity gate tight; raised via INTFER_PREFILL_CHUNK_M for long prompts.
  int chunk_M = 8;
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
#ifdef INTFER_PROFILE_GPU
  {
    auto p = get_gpu_profile();
    std::fprintf(stderr,
      "[GPU profile @prefill] kernel=%.1fms submit=%.1fms waitblock=%.1fms "
      "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms\n",
      p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
      p.rebuild_us/1000.0, p.rebuild_count,
      p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0);
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
      "rebuild=%.1fms(%lu) usm_write=%.1fms readback=%.1fms copy=%.1fms\n",
      p.kernel_us/1000.0, p.submit_us/1000.0, p.waitblock_us/1000.0,
      p.rebuild_us/1000.0, p.rebuild_count,
      p.usm_write_us/1000.0, p.readback_us/1000.0, p.copy_us/1000.0);
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
