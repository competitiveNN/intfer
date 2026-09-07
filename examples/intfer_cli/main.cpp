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
#include <memory>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/mman.h>
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
// SpinQuant online rotation helpers
static void rot_apply(const float* x, const Rotation& R, float* out){
  int dim = (int)R.dim;
  for(int j=0;j<dim;j++){ float s=0; for(int i=0;i<dim;i++) s+=x[i]*R.mat[i*dim+j]; out[j]=s; }
}
static void rot_left(const float* y, const Rotation& R, float* out){
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

  // Cached lm_head FP16 upload + one-off FP16 GEMM on device.
  // The first call uploads W[N,K] as FP16 to each device; later calls reuse
  // the cached device pointer and just launch the GEMM.
  std::vector<float> run_lm_head_f16(const float* fn, // final norm hidden [K]
                                     const float* W,   // lm_head weight [N,K]
                                     uint32_t N, uint32_t K){
    if(!eng) return {};
    if(lm_head_w_dev_.empty()){
      size_t w_bytes = (size_t)N * K * sizeof(sycl::half);
      lm_head_N_ = N; lm_head_K_ = K;
      lm_head_w_dev_.resize(eng->num_devices());
      lm_head_c_dev_.resize(eng->num_devices());
      for(size_t d=0; d<eng->num_devices(); ++d){
        sycl::queue& q = eng->device_queue((int)d);
        lm_head_w_dev_[d] = static_cast<sycl::half*>(
            sycl::malloc_device(w_bytes, q.get_device(), q.get_context()));
        lm_head_c_dev_[d] = static_cast<sycl::half*>(
            sycl::malloc_device((size_t)N * sizeof(sycl::half), q.get_device(), q.get_context()));
        // Upload once.
        std::vector<sycl::half> w_fp16((size_t)N * K);
        for(uint32_t i=0;i<(size_t)N*K;i++) w_fp16[i] = sycl::half(W[i]);
        q.memcpy(lm_head_w_dev_[d], w_fp16.data(), w_bytes);
      }
      for(size_t d=0; d<eng->num_devices(); ++d)
        eng->device_queue((int)d).wait();
    }
    if(lm_head_N_ != N || lm_head_K_ != K){
      std::fprintf(stderr, "gpu: lm_head size mismatch %u x %u vs cached %u x %u\n",
                   N, K, lm_head_N_, lm_head_K_);
      return {};
    }
    // Upload activation as FP16 and run GEMM.
    std::vector<sycl::half> x_fp16(K);
    for(uint32_t i=0;i<K;i++) x_fp16[i] = sycl::half(fn[i]);
    std::vector<float> out(N, 0.f);
    std::vector<sycl::event> evts;
    evts.reserve(eng->num_devices());
    for(size_t d=0; d<eng->num_devices(); ++d){
      sycl::queue& q = eng->device_queue((int)d);
      sycl::half* x_dev = static_cast<sycl::half*>(
          sycl::malloc_device((size_t)K * sizeof(sycl::half), q.get_device(), q.get_context()));
      q.memcpy(x_dev, x_fp16.data(), (size_t)K * sizeof(sycl::half));
      sycl::event evt = gemm::quant_gemm_f16(q, x_dev, lm_head_w_dev_[d],
                                             lm_head_c_dev_[d], 1, N, K);
      q.memcpy(out.data(), lm_head_c_dev_[d], (size_t)N * sizeof(float), evt);
      evts.push_back(evt);
      sycl::free(x_dev, q);
    }
    for(auto& e : evts) e.wait();
    return out;
  }

  // Cache state for lm_head upload.
  std::vector<sycl::half*> lm_head_w_dev_;
  std::vector<sycl::half*> lm_head_c_dev_;
  uint32_t lm_head_N_ = 0, lm_head_K_ = 0;
  const Rotation* rot_for(const std::string& name) override {
    if(!eng) return nullptr;
    auto it = ti_cache.find(name);
    if(it == ti_cache.end()) return nullptr;
    return eng->rotation_for(it->second);
  }
};
#endif

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
static std::vector<Layer> layers;

// ============================================================================
// Forward step
// ============================================================================
struct StepOut { int best; std::vector<float> logits; };
static StepOut step(int token, int pos,
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
  std::vector<float> h(row, row+cfg.H);

  // Per-layer HF-tensor name prefix
  auto name = [&](int L, const char* slot){
    char b[256]; snprintf(b, sizeof(b),
        "model.language_model.layers.%d.%s", L, slot);
    return std::string(b);
  };
  auto get = [&](const std::string& n) -> const std::vector<float>& {
    auto it = T.find(n);
    if(it==T.end()){ std::fprintf(stderr, "missing tensor: %s\n", n.c_str()); std::exit(1); }
    return it->second;
  };

   for(int L=0; L<cfg.layers; L++){
   if((L & 15) == 0){ std::fprintf(stderr, "[intfer] forward L=%d/%d pos=%d hsize=%zu layers=%zu\n", L, cfg.layers, pos, h.size(), layers.size()); std::fflush(stderr); }
   if(L < 0 || (size_t)L >= layers.size()){ std::fprintf(stderr, "[intfer] FATAL layer OOB\n"); std::exit(1); }
   Layer& ly = layers[L];
   std::vector<float> res = h;
   std::vector<float> normed(cfg.H);
   rmsnorm(h.data(), ly.in_ln, normed.data(), cfg.H);
   ckpt_h(L, normed, "post_in_ln");
  if(ly.is_full){
      // ---- Attention path (q/k/v right-site; o left-site) ----
      const Rotation* Rq = backend->rot_for(name(L, "self_attn.q_proj.weight"));
      std::vector<float> xq(normed);
      if(Rq) rot_apply(normed.data(), *Rq, xq.data());
      std::vector<float> qp = backend->run_linear(name(L, "self_attn.q_proj.weight"), xq);
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
      std::vector<float> gate(cfg.n_heads*cfg.head_dim, 0.f), qpart(cfg.n_heads*cfg.head_dim);
      if(packed_gate){
        for(int hh=0; hh<cfg.n_heads; hh++) for(int d=0; d<cfg.head_dim; d++){
          qpart[hh*cfg.head_dim+d] = qp[hh*2*cfg.head_dim+d];
          gate[hh*cfg.head_dim+d]  = qp[hh*2*cfg.head_dim+cfg.head_dim+d];
        }
      } else {
        for(size_t i=0; i<qrows && i<qp.size(); i++) qpart[i] = qp[i];
      }
      for(int hh=0; hh<cfg.n_heads; hh++){
        float tmp[cfg.head_dim]; rmsnorm(&qpart[hh*cfg.head_dim], ly.q_norm_w, tmp, cfg.head_dim);
        memcpy(&qpart[hh*cfg.head_dim], tmp, cfg.head_dim*sizeof(float));
        rope(&qpart[hh*cfg.head_dim], pos);
      }
      const Rotation* Rk = backend->rot_for(name(L, "self_attn.k_proj.weight"));
      const Rotation* Rv = backend->rot_for(name(L, "self_attn.v_proj.weight"));
      std::vector<float> xk(normed), xv(normed);
      if(Rk) rot_apply(normed.data(), *Rk, xk.data());
      if(Rv) rot_apply(normed.data(), *Rv, xv.data());
      std::vector<float> k = backend->run_linear(name(L, "self_attn.k_proj.weight"), xk);
      std::vector<float> v = backend->run_linear(name(L, "self_attn.v_proj.weight"), xv);
      ckpt("attn.k_proj", L, k); ckpt("attn.v_proj", L, v);
      for(int hh=0; hh<cfg.n_kv; hh++){
        float tmp[cfg.head_dim]; rmsnorm(&k[hh*cfg.head_dim], ly.k_norm_w, tmp, cfg.head_dim);
        memcpy(&k[hh*cfg.head_dim], tmp, cfg.head_dim*sizeof(float));
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
      // Attention compute (host, since the SSM/attn kernel is a 1-token
      // decode and the TP shard of K/V is small — see `qwen_forward_gpu`
      // for the FlashAttention integration on the 1st device).
      std::vector<float> attn(cfg.n_heads*cfg.head_dim, 0.f);
      for(int hh=0; hh<cfg.n_heads; hh++){
        int kvh=hh/(cfg.n_heads/cfg.n_kv);
        std::vector<float> scores(pos+1); float mx=-1e30f;
        for(int t=0;t<=pos;t++){
          const float* qh=&qpart[hh*cfg.head_dim];
          const float* kh=&ly.kcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
          float s=0; for(int d=0;d<cfg.head_dim;d++) s+=qh[d]*kh[d];
          s*=1.0f/sqrtf((float)cfg.head_dim); scores[t]=s; if(s>mx) mx=s;
        }
        float sum=0; for(int t=0;t<=pos;t++){ scores[t]=expf(scores[t]-mx); sum+=scores[t]; }
        for(int t=0;t<=pos;t++) scores[t]/=sum;
        float* oh=&attn[hh*cfg.head_dim]; for(int d=0;d<cfg.head_dim;d++) oh[d]=0;
        for(int t=0;t<=pos;t++){
          const float* vh=&ly.vcache[(t*cfg.n_kv+kvh)*cfg.head_dim];
          for(int d=0;d<cfg.head_dim;d++) oh[d]+=scores[t]*vh[d];
        }
      }
      if (cfg.attn_output_gate && packed_gate)
        for(int i=0;i<cfg.n_heads*cfg.head_dim;i++) attn[i] *= sigmoidf(gate[i]);
      // o_proj contracts attn (packed-gate layout, both known models).
      // The !packed_gate fallback feeds attn concat gate; run_linear
      // contracts min(x.size(), K) so both widths are safe.
      std::vector<float> o_raw;
      {
        size_t attn_n = (size_t)cfg.n_heads * cfg.head_dim;
        // Peek at the IR weight size without caching: run_linear derives K
        // from the stored N/K; instead probe via a zero probe is wasteful,
        // so infer from q layout: unpacked-gate models need attn+gate input.
        std::vector<float> o_in;
        if(!packed_gate){
          o_in.resize(attn_n + attn_n);
          for(size_t i=0;i<attn_n;i++){ o_in[i]=attn[i]; o_in[attn_n+i]=gate[i]; }
        } else {
          o_in = attn;
        }
        o_raw = backend->run_linear(name(L, "self_attn.o_proj.weight"), o_in);
        ckpt("attn.o_proj", L, o_raw);
        // If the backend used the packed width (K=n_heads*hd) the extra gate
        // tail is ignored by the GEMM (x longer than K); if it used the wide
        // width the full concat is consumed. Both cases are handled because
        // run_linear contracts min(x.size(), K).
      }
      const Rotation* Ro = backend->rot_for(name(L, "self_attn.o_proj.weight"));
      if(Ro) rot_left(o_raw.data(), *Ro, h.data());
      else h = o_raw;
    } else {
      // ---- Linear-attention / SSM path (uses .intifer quantized weights) ----
      const float* conv_w = get(name(L,"linear_attn.conv1d.weight")).data();
      const float* A_log  = get(name(L,"linear_attn.A_log")).data();
      const float* dt_bias= get(name(L,"linear_attn.dt_bias")).data();
      // Use the .intifer dequantized weights via the CPU backend instead of
      // loading FP32 safetensors for these matmuls. The .intifer stores the
      // same weights (FP16-quantized) and the backend dequantizes on demand.
      // SpinQuant: in_proj_qkv/z/b/a are right-site (replay x.R on the
      // activation before the GEMM), out_proj is left-site (R.y after).
      // The LA branch previously skipped both replays, feeding raw normed
      // into rotated-quantized weights (27B: L0 out 0.1781 vs HF -0.0240).
        const Rotation* Rqkv = backend->rot_for(name(L,"linear_attn.in_proj_qkv.weight"));
        std::vector<float> xqkv(normed);
        if(Rqkv) rot_apply(normed.data(), *Rqkv, xqkv.data());
        std::vector<float> mqkv = backend->run_linear(name(L,"linear_attn.in_proj_qkv.weight"), xqkv);
        ckpt("la.qkv", L, mqkv);
        std::vector<float> conv_out(cfg.conv_dim);
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
        const Rotation* Rz = backend->rot_for(name(L,"linear_attn.in_proj_z.weight"));
        const Rotation* Rb = backend->rot_for(name(L,"linear_attn.in_proj_b.weight"));
        const Rotation* Ra = backend->rot_for(name(L,"linear_attn.in_proj_a.weight"));
        std::vector<float> xz(normed), xb(normed), xa(normed);
        if(Rz) rot_apply(normed.data(), *Rz, xz.data());
        if(Rb) rot_apply(normed.data(), *Rb, xb.data());
        if(Ra) rot_apply(normed.data(), *Ra, xa.data());
        std::vector<float> z = backend->run_linear(name(L,"linear_attn.in_proj_z.weight"), xz);
        std::vector<float> b = backend->run_linear(name(L,"linear_attn.in_proj_b.weight"), xb);
        std::vector<float> a = backend->run_linear(name(L,"linear_attn.in_proj_a.weight"), xa);
        ckpt("la.z", L, z); ckpt("la.b", L, b); ckpt("la.a", L, a);
          std::vector<float> g(cfg.n_v_heads);
      for(int hh=0; hh<cfg.n_v_heads; hh++){
        float sp=log1pf(expf(a[hh]+dt_bias[hh])); g[hh]=-expf(A_log[hh])*sp;
      }
        std::vector<float> core(cfg.n_v_heads*cfg.v_head_dim,0.f);
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
        std::vector<float> kv_mem(cfg.v_head_dim,0.f);
        for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) kv_mem[c]+=S[d*cfg.v_head_dim+c]*kn[d];
        std::vector<float> delta(cfg.v_head_dim); const float* vh=&value[hh*cfg.v_head_dim];
        for(int c=0;c<cfg.v_head_dim;c++) delta[c]=(vh[c]-kv_mem[c])*beta;
        for(int d=0;d<cfg.k_head_dim;d++) for(int c=0;c<cfg.v_head_dim;c++) S[d*cfg.v_head_dim+c]+=kn[d]*delta[c];
        for(int c=0;c<cfg.v_head_dim;c++){
          float s=0; for(int d=0;d<cfg.k_head_dim;d++) s+=S[d*cfg.v_head_dim+c]*qn[d];
          core[hh*cfg.v_head_dim+c]=s;
        }
      }
        const float* ssm_norm_w = get(name(L,"linear_attn.norm.weight")).data();
      std::vector<float> normed_core(cfg.n_v_heads*cfg.v_head_dim);
      for(int hh=0; hh<cfg.n_v_heads; hh++){
        const float* x=&core[hh*cfg.v_head_dim];
        const float* zg=&z[hh*cfg.v_head_dim];
        float* o=&normed_core[hh*cfg.v_head_dim];
        float v=0; for(int i=0;i<cfg.v_head_dim;i++) v+=x[i]*x[i];
        float inv=1.0f/sqrtf(v/cfg.v_head_dim+cfg.rms_eps);
        for(int i=0;i<cfg.v_head_dim;i++) o[i]=(x[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
      }
      std::vector<float> o_la = backend->run_linear(name(L,"linear_attn.out_proj.weight"), normed_core);
      const Rotation* Ro_la = backend->rot_for(name(L,"linear_attn.out_proj.weight"));
      if(Ro_la){ h.assign(cfg.H, 0.f); rot_left(o_la.data(), *Ro_la, h.data()); }
      else h = std::move(o_la);
      ckpt("la.out", L, h);
  if(trace_h && pos==0 && (L==0 || L==3)){
    std::vector<float> pmqkv = backend->run_linear(name(L,"linear_attn.in_proj_qkv.weight"), xqkv);
    std::vector<float> pz = backend->run_linear(name(L,"linear_attn.in_proj_z.weight"), xz);
    std::vector<float> pa = backend->run_linear(name(L,"linear_attn.in_proj_a.weight"), xa);
    std::vector<float> pb = backend->run_linear(name(L,"linear_attn.in_proj_b.weight"), xb);
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
    std::vector<float> rn2(cfg.H);
    rmsnorm(h.data(), ly.post_ln, rn2.data(), cfg.H);
    const Rotation* Rg = backend->rot_for(name(L, "mlp.gate_proj.weight"));
    std::vector<float> rin(rn2);
    if(Rg) rot_apply(rn2.data(), *Rg, rin.data());
    std::vector<float> mg = backend->run_linear(name(L, "mlp.gate_proj.weight"), rin);
    ckpt("mlp.gate", L, mg);
    std::vector<float> mu = backend->run_linear(name(L, "mlp.up_proj.weight"),   rin);
    ckpt("mlp.up", L, mu);
    for(int i=0;i<cfg.inter;i++) mg[i] = silu(mg[i])*mu[i];
    std::vector<float> md = backend->run_linear(name(L, "mlp.down_proj.weight"), mg);
    ckpt("mlp.down", L, md);
    const Rotation* Rd = backend->rot_for(name(L, "mlp.down_proj.weight"));
    if(trace_h && pos==0 && L==0){
      double n2=0; for(float x : mg) n2+=(double)x*x;
      std::fprintf(stderr, "[dbg] L=%d mlp mg|.|=%.6f mg[:4]=%.6f %.6f %.6f %.6f mu[:4]=%.6f %.6f %.6f %.6f md[:4]=%.6f %.6f %.6f %.6f Rg=%p Rd=%p\n",
                   L, sqrt(n2), mg[0], mg[1], mg[2], mg[3], mu[0], mu[1], mu[2], mu[3],
                   md[0], md[1], md[2], md[3], (const void*)Rg, (const void*)Rd);
      std::fflush(stderr);
    }
    std::vector<float> mlp_out(cfg.H);
    if(Rd) rot_left(md.data(), *Rd, mlp_out.data());
    else mlp_out = md;
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
  std::vector<float> fn(cfg.H); rmsnorm(h.data(), final_norm, fn.data(), cfg.H);
#ifdef QP_PROBE_SIZES
  if(pos==0 || pos==5){
    std::fprintf(stderr, "[dbg] L23? pos=%d h[:4]=%.4f %.4f %.4f %.4f fn[:4]=%.4f %.4f %.4f %.4f\n",
                 pos, h[0], h[1], h[2], h[3], fn[0], fn[1], fn[2], fn[3]);
    std::fflush(stderr);
  }
#endif
  std::vector<float> logits(cfg.vocab, 0.f);
  int best=-1; float bestv=-1e30f;
  const float* head = lm_head ? lm_head : embed;
  std::fprintf(stderr, "[intfer] logits: vocab=%d H=%d head=%p\n", cfg.vocab, cfg.H, (const void*)head);
  std::fflush(stderr);
  if(lm_head && backend){
    auto gpu = dynamic_cast<GpuBackend*>(backend);
    if(gpu){
      logits = gpu->run_lm_head_f16(fn.data(), lm_head, (uint32_t)cfg.vocab, (uint32_t)cfg.H);
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
  if(lm_head==nullptr || !backend || dynamic_cast<CpuBackend*>(backend)){
    for(int v=0; v<cfg.vocab; v++){
      if(logits[v]>bestv){ bestv=logits[v]; best=v; }
    }
  }
  out.best = best;
  out.logits = std::move(logits);
  return out;
}

// ============================================================================
// Sampling
// ============================================================================
static int sample(const std::vector<float>& logits, int top_k, float top_p,
                  float temperature, std::mt19937& rng){
  if(temperature <= 0.f) temperature = 1e-5f;
  float mx = -1e30f;
  for(float v : logits) if(v>mx) mx=v;
  std::vector<float> probs(logits.size());
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
// CLI
// ============================================================================
enum class GpuMode { Auto, ForceOn, ForceOff };

static void usage(const char* p){
  std::fprintf(stderr,
    "intfer — low-bit LLM inference (Stage-1 quantized .intifer, host or SYCL GPU)\n"
    "Usage: %s -m <model.intifer> [options]\n"
    "  -m PATH        .intifer model file (required)\n"
    "  -s PATH        safetensors file for embed/norms/SSM (auto-detected)\n"
    "  -p TEXT        prompt text (default: empty)\n"
    "  -P FILE        read prompt from file\n"
    "  -n N           max new tokens to generate (default: 64)\n"
    "  -t TEMP        sampling temperature (default: 0.0 = greedy)\n"
    "  -k K           top-k sampling (default: 0 = off)\n"
    "  -T TOPP        top-p / nucleus sampling (default: 0.0 = off)\n"
    "  -S SEED        RNG seed (default: 0 = time)\n"
    "  --gpu          force SYCL/GPU backend (intfer::Engine)\n"
    "  --no-gpu       force CPU backend (dequant + host matmul)\n"
    "  --tokenizer P  path to tokenizer.json (default: Qwen3.5-0.8B)\n"
    "  --tokenizer-py P path to bpe_tokenize.py\n"
    "  --detok-py P   path to bpe_detokenize.py\n"
    "  --no-stream    do not stream tokens (print full output after generation)\n"
    "  --ids          also print token ids to stderr (debug)\n",
    p);
}

int main(int argc, char** argv){
  std::signal(SIGSEGV, segv_handler);
  std::signal(SIGABRT, segv_handler);
  // ---- Arg parse
  std::string intfer_path, st_path, prompt, prompt_file;
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
  GpuMode gpu_mode = GpuMode::Auto;
  int i = 1;
  while(i < argc){
    std::string a = argv[i];
    if(a=="-m" && i+1<argc) intfer_path = argv[++i];
    else if(a=="-s" && i+1<argc) st_path = argv[++i];
    else if(a=="-p" && i+1<argc) prompt = argv[++i];
    else if(a=="-P" && i+1<argc) prompt_file = argv[++i];
    else if(a=="-n" && i+1<argc) max_new = std::atoi(argv[++i]);
    else if(a=="-t" && i+1<argc) temperature = std::atof(argv[++i]);
    else if(a=="-k" && i+1<argc) top_k = std::atoi(argv[++i]);
    else if(a=="-T" && i+1<argc) top_p = std::atof(argv[++i]);
    else if(a=="-S" && i+1<argc) seed = (uint32_t)std::atoi(argv[++i]);
    else if(a=="--gpu") gpu_mode = GpuMode::ForceOn;
    else if(a=="--no-gpu") gpu_mode = GpuMode::ForceOff;
    else if(a=="--tokenizer" && i+1<argc) tokenizer_json = argv[++i];
    else if(a=="--tokenizer-py" && i+1<argc) tokenizer_py = argv[++i];
    else if(a=="--detok-py" && i+1<argc) detok_py = argv[++i];
    else if(a=="--no-stream") stream = false;
    else if(a=="--ids") print_ids = true;
    else if(a=="-h" || a=="--help"){ usage(argv[0]); return 0; }
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    ++i;
  }
  if(intfer_path.empty()){ usage(argv[0]); return 2; }
  if(!prompt_file.empty()){
    FILE* f = fopen(prompt_file.c_str(), "r");
    if(!f){ std::fprintf(stderr, "cannot read -P %s\n", prompt_file.c_str()); return 1; }
    char buf[8192]; size_t n = fread(buf,1,sizeof(buf)-1,f); fclose(f);
    prompt.assign(buf, n);
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
    // Try model-name-derived paths first (from -m model.intifer name)
    std::string model_name;
    {
      size_t sp = intfer_path.rfind('/');
      if(sp != std::string::npos) model_name = intfer_path.substr(sp+1);
      // strip .intifer suffix
      size_t dot = model_name.rfind(".intifer");
      if(dot != std::string::npos) model_name = model_name.substr(0, dot);
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
  {
    std::string config_dir;
    // Prefer config.json alongside the safetensors (correct model dims).
    size_t sp = st_path.rfind('/');
    if(sp != std::string::npos) config_dir = st_path.substr(0, sp);
    ModelConfig loaded = load_model_config(config_dir);
    cfg = loaded;
    std::fprintf(stderr, "[intfer] config: model_id=%s H=%d layers=%d inter=%d heads=%d kv=%d conv_dim=%d sdim=%d\n",
                 cfg.model_id.c_str(), cfg.H, cfg.layers, cfg.inter, cfg.n_heads, cfg.n_kv, cfg.conv_dim, cfg.sdim);
  }
  std::fprintf(stderr, "[intfer] intfer: %s\n", intfer_path.c_str());
  std::fprintf(stderr, "[intfer] safetensors: %s\n", st_path.c_str());

  // ---- Load .intifer (Stage-1 quantized MLP + attn)
  // Borrowed mmap views (zero-copy): the 22GB code region stays in page
  // cache instead of being copied into RAM. mblob must outlive main, so it
  // is munmap'd only at the very end.
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
  for(const auto& R : rs.rotations) rid_map[R.id] = &R;
  rid_map_global = rid_map;
    // Populate CPU backend's per-layer QLinear tables from the deserialized IR.
  // (Done lazily after backend selection; see below.)

  // ---- Load safetensors (embed, norms, SSM)
  // For the GPU backend, the large q/k/v/o/gate/up/down projection weights are
  // NOT loaded into RAM — the .intifer file already contains them as quantized
  // tensors and the Engine reads them on demand. Only the small "host" tensors
  // (embed_tokens, norms, SSM params) are pulled in. For the CPU backend we
  // keep loading everything, since the CPU path uses these for run_linear().
  SafetensorsReader sr;
  if(!sr.open(st_path)){ std::fprintf(stderr, "FAIL: open safetensors\n"); return 1; }

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
    gpu_probe = Engine::load(intfer_path, 1);
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
  for(const auto& t : sr.tensors()){
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
  std::fprintf(stderr, "[intfer] safetensors host tensors loaded=%zu (%.1f MB) projections=%s\n",
               T.size(), total_loaded / (1024.0*1024.0),
               skip_large_weights ? "GPU-owned" : "on-demand via reader");
  // ---- Auto-detect linear-attention head counts from safetensors shapes ----
  // (Linear attention may use different head counts than full attention.)
  // in_proj_a: [N_V_HEADS, H], in_proj_z: [N_V_HEADS*V_HEAD_DIM, H]
  for(const auto& t : sr.tensors()){
    if(t.name.find("linear_attn.in_proj_a.weight") != std::string::npos){
      if(t.shape.size()>=2 && t.shape[0] > 0 && t.shape[1] == cfg.H)
        cfg.n_v_heads = (int)t.shape[0];
    }
    if(t.name.find("linear_attn.norm.weight") != std::string::npos){
      if(!t.shape.empty()) cfg.v_head_dim = (int)t.shape[0];
    }
    if(t.name.find("linear_attn.in_proj_z.weight") != std::string::npos){
      if(t.shape.size()>=2 && t.shape[1] == cfg.H){
        int total = (int)t.shape[0];
        int kk = total / cfg.n_v_heads;
        if(total == cfg.n_v_heads * cfg.k_head_dim * 3)
          cfg.k_head_dim = kk;
      }
    }
    if(t.name.find("linear_attn.in_proj_qkv.weight") != std::string::npos){
      // qkv total = 2*K_HEAD_DIM*N_K_HEADS + N_V_HEADS*V_HEAD_DIM
      if(t.shape.size()>=2 && t.shape[1] == cfg.H){
        int total = (int)t.shape[0];
        int vdim = cfg.n_v_heads * cfg.v_head_dim;
        int qk = total - vdim;
        if(qk > 0 && qk % (2*cfg.k_head_dim) == 0)
          cfg.n_k_heads = qk / (2*cfg.k_head_dim);
      }
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
  // Pre-scan the safetensors to find which layers are linear-attention
  // (have linear_attn.in_proj_qkv) vs full-attention (have self_attn.q_proj).
  std::set<int> la_layers;
  for(const auto& t : sr.tensors()){
    if(t.name.find("linear_attn.in_proj_qkv") != std::string::npos){
      size_t p = t.name.find("layers.");
      if(p != std::string::npos){
        std::string num = t.name.substr(p+7);
        size_t dot = num.find('.');
        if(dot != std::string::npos) la_layers.insert(std::atoi(num.c_str()));
      }
    }
  }
  std::fprintf(stderr, "[intfer] la_layers detected: %zu layers (", la_layers.size());
  bool first = true;
  for(int L : la_layers){ if(!first) std::fprintf(stderr, ","); std::fprintf(stderr, "%d", L); first=false; }
  std::fprintf(stderr, ")\n");
  layers.resize(cfg.layers);
  for(int L=0; L<cfg.layers; L++){
    char b[256];
    auto p=[&](const char* s){ snprintf(b,sizeof(b),
              "model.language_model.layers.%d.%s",L,s); return std::string(b); };
    Layer& ly = layers[L];
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
    gpu_backend->sr_fallback = &sr;
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
    cpu->sr_fallback = &sr;
    cpu->host_names_fallback = &host_only_names;
    backend = std::move(cpu);
  }
  std::fprintf(stderr, "[intfer] backend: %s\n", backend->name());
  std::fprintf(stderr, "[intfer] starting prefill...\n"); std::fflush(stderr);

  // ---- Tokenize the prompt
  std::vector<int> prompt_ids;
  if(!prompt.empty()){
    prompt_ids = encode_text(prompt, tokenizer_json, tokenizer_py);
  }
  if(prompt_ids.empty()){
    prompt_ids = {760, 6511, 314, 9338, 369};
    std::fprintf(stderr, "[intfer] (using default prompt)\n");
  }
  std::fprintf(stderr, "[intfer] prompt: %zu tokens ->", prompt_ids.size());
  for(int t : prompt_ids) std::fprintf(stderr, " %d", t);
  std::fprintf(stderr, "\n");

  // ---- Prefill + decode loop
  // NOTE: S grows as we decode (KV positions are absolute). Size `dump`
  // for prompt + max_new so the decode-time record memcpy never writes
  // out of bounds (fixed a SIGSEGV in __intel_avx_rep_memcpy).
  int S = (int)prompt_ids.size();
  std::vector<std::vector<float>> dump(cfg.layers,
      std::vector<float>((size_t)(S+max_new)*cfg.H, 0.f));
  std::vector<int> generated;
  auto t0 = std::chrono::high_resolution_clock::now();
  for(int pos=0; pos<S; pos++){
    StepOut o = step(prompt_ids[pos], pos, T, embed, final_norm, lm_head,
                     backend.get(), true, dump);
    if(pos==S-1) generated.push_back(o.best);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double prefill_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
  std::fprintf(stderr, "[intfer] prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
               S, prefill_ms, S*1000.0/std::max(1.0, prefill_ms));

  int token = generated.back();
  std::string output;
  if(stream) std::fwrite(prompt.c_str(), 1, prompt.size(), stdout);
  std::fprintf(stderr, "[intfer] prefill done, generated=%zu max_new=%d\n",
               generated.size(), max_new); std::fflush(stderr);
  auto t_d0 = std::chrono::high_resolution_clock::now();
  for(int step_i = 0; step_i < max_new; step_i++){
    std::fprintf(stderr, "[intfer] decode step %d: token=%d pos=%d\n",
                 step_i, token, S+step_i); std::fflush(stderr);
    StepOut o = step(token, S+step_i, T, embed, final_norm, lm_head,
                     backend.get(), false, dump);
    int next;
    if(temperature <= 0.f && top_k==0 && top_p<=0.f){
      next = o.best;
    } else {
      next = sample(o.logits, top_k, top_p, temperature, rng);
    }
    if(print_ids) std::fprintf(stderr, "  [%d] id=%d\n", step_i, next);
    generated.push_back(next);
    std::string piece = detokenize_one(next, detok_py, tokenizer_json);
    if(stream){
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
    }
    output += piece;
    token = next;
    if(next == 248044 || next == 248046) break;
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
      "\n[GPU profile]\n"
      "  total_kernel_us   : %.1f (%lu launches)\n"
      "  total_wait_us     : %.1f (%lu waits)\n"
      "  total_copy_us     : %.1f (%lu copies)\n"
      "  total_readback_us : %.1f (%lu readbacks)\n"
      "  peak_wait_depth   : %lu\n"
      "  kernel %% of decode : %.1f%%\n",
      p.kernel_us, p.kernel_count, p.wait_us, p.wait_count,
      p.copy_us, p.copy_count, p.readback_us, p.readback_count,
      p.peak_wait_depth,
      100.0 * (p.kernel_us / 1000.0) / std::max(1.0, decode_ms));
  }
#endif
  if(!stream) std::fwrite(output.data(), 1, output.size(), stdout);
  std::fwrite("\n", 1, 1, stdout);
  munmap_blob(mblob); // release the borrowed payload mapping at the very end
  return 0;
}
