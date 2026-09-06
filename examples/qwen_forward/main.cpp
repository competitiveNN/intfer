// examples/qwen_forward/main.cpp
// CPU reference forward for Qwen3.5-0.8B (text-only), loaded entirely from the
// original safetensors (exact weights) so it validates the *architecture*
// (gated-delta-rule SSM, mRoPE, attention, RMSNorm, MLP) independent of the
// Stage-1 quantization. Outputs per-layer hidden states (prefill) + greedy tokens
// for comparison against scripts/oracle_qwen35.py.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include "intfer/safetensors.hpp"

using namespace intfer;

// ---- config (Qwen3.5-0.8B text_config) ----
static const int   H        = 1024;
static const int   LAYERS   = 24;
static const int   INTER    = 3584;
static const int   N_HEADS  = 8;
static const int   N_KV     = 2;
static const int   HEAD_DIM = 256;
static const int   N_K_HEADS   = 16;
static const int   N_V_HEADS   = 16;
static const int   K_HEAD_DIM  = 128;
static const int   V_HEAD_DIM  = 128;
static const int   CONV_K   = 4;
static const double ROPE_THETA = 1e7;
static const int   ROTARY_DIM  = 64;   // int(HEAD_DIM * 0.25)
static const float RMS_EPS  = 1e-6f;
static const int   VOCAB    = 248320;
static const int   CONV_DIM = N_K_HEADS*K_HEAD_DIM*2 + N_V_HEADS*V_HEAD_DIM; // 6144
static const int   SDIM     = N_V_HEADS*K_HEAD_DIM*V_HEAD_DIM;               // 16*128*128

static float inv_freq[ROTARY_DIM/2];

// ---- math helpers ----
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }

static std::vector<float> linear(const std::vector<float>& W, const std::vector<float>& x, int N, int K){
  std::vector<float> y(N, 0.f);
  for(int i=0;i<N;i++){ const float* wr=&W[i*K]; float s=0; for(int k=0;k<K;k++) s+=wr[k]*x[k]; y[i]=s; }
  return y;
}
static void rmsnorm(const float* x, const float* w, float* out, int dim){
  float v=0; for(int i=0;i<dim;i++) v+=x[i]*x[i];
  v = 1.0f/sqrtf(v/dim + RMS_EPS);
  for(int i=0;i<dim;i++) out[i] = (x[i]*v)*(1.0f+w[i]); // Qwen style (1+weight)
}
static void l2norm(const float* x, float* out, int dim){
  float s=0; for(int i=0;i<dim;i++) s+=x[i]*x[i];
  float inv=1.0f/sqrtf(s+1e-6f);
  for(int i=0;i<dim;i++) out[i]=x[i]*inv;
}
// rotate first ROTARY_DIM of a HEAD_DIM vector at position pos
static void rope(float* h, int pos){
  for(int d=0; d<ROTARY_DIM/2; d++){
    float f = inv_freq[d]*pos;
    float c=cosf(f), s=sinf(f);
    float a=h[d], b=h[d+ROTARY_DIM/2];
    h[d]          = a*c - b*s;
    h[d+ROTARY_DIM/2] = a*s + b*c;
  }
}

struct Layer {
  bool is_full;
  const float *q_proj,*k_proj,*v_proj,*o_proj,*q_norm_w,*k_norm_w;
  const float *in_qkv,*in_z,*in_b,*in_a,*conv_w,*A_log,*dt_bias,*ssm_norm_w,*ssm_out;
  const float *in_ln,*post_ln,*mlp_gate,*mlp_up,*mlp_down;
  // caches
  std::vector<float> kcache, vcache;   // full-attn
  std::vector<float> ssm_state;        // SDIM
  std::vector<float> conv_state;       // CONV_DIM*(CONV_K-1)
};

int main(int argc, char** argv){
  for(int d=0; d<ROTARY_DIM/2; d++) inv_freq[d] = (float)(1.0/pow(ROPE_THETA, 2.0*d/ROTARY_DIM));

  const char* st_path = (argc>1)? argv[1] : "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
  const char* dump_path = (argc>2)? argv[2] : "test/models/qwen35_cpp.bin";
  const char* prompt = (argc>3)? argv[3] : "The capital of France is";

  printf("reading safetensors %s ...\n", st_path);
  SafetensorsReader r;
  if(!r.open(st_path)){ printf("FAIL open\n"); return 1; }
  std::map<std::string, std::vector<float>> T;
  for(const auto& t : r.tensors()){
    std::vector<float> v;
    if(!r.read(t.name, v)){ printf("FAIL read %s\n", t.name.c_str()); return 1; }
    T[t.name] = std::move(v);
  }
  printf("loaded %zu tensors\n", T.size());
  auto get = [&](const std::string& n)->const std::vector<float>&{ return T.at(n); };

  const float* embed = get("model.language_model.embed_tokens.weight").data();

  // build layers
  std::vector<Layer> layers(LAYERS);
  for(int L=0; L<LAYERS; L++){
    char b[256]; auto p=[&](const char* s){ snprintf(b,sizeof(b), "model.language_model.layers.%d.%s", L, s); return std::string(b); };
    bool is_full = T.count(p("self_attn.q_proj.weight"))>0;
    Layer& ly = layers[L]; ly.is_full = is_full;
    ly.in_ln = get(p("input_layernorm.weight")).data();
    ly.post_ln = get(p("post_attention_layernorm.weight")).data();
    ly.mlp_gate = get(p("mlp.gate_proj.weight")).data();
    ly.mlp_up   = get(p("mlp.up_proj.weight")).data();
    ly.mlp_down = get(p("mlp.down_proj.weight")).data();
    if(is_full){
      ly.q_proj = get(p("self_attn.q_proj.weight")).data();
      ly.k_proj = get(p("self_attn.k_proj.weight")).data();
      ly.v_proj = get(p("self_attn.v_proj.weight")).data();
      ly.o_proj = get(p("self_attn.o_proj.weight")).data();
      ly.q_norm_w = get(p("self_attn.q_norm.weight")).data();
      ly.k_norm_w = get(p("self_attn.k_norm.weight")).data();
      ly.kcache.assign(N_KV*1024*HEAD_DIM, 0.f); // maxlen 1024
      ly.vcache.assign(N_KV*1024*HEAD_DIM, 0.f);
    } else {
      ly.in_qkv = get(p("linear_attn.in_proj_qkv.weight")).data();
      ly.in_z   = get(p("linear_attn.in_proj_z.weight")).data();
      ly.in_b   = get(p("linear_attn.in_proj_b.weight")).data();
      ly.in_a   = get(p("linear_attn.in_proj_a.weight")).data();
      ly.conv_w = get(p("linear_attn.conv1d.weight")).data();
      ly.A_log  = get(p("linear_attn.A_log")).data();
      ly.dt_bias= get(p("linear_attn.dt_bias")).data();
      ly.ssm_norm_w = get(p("linear_attn.norm.weight")).data();
      ly.ssm_out = get(p("linear_attn.out_proj.weight")).data();
      ly.ssm_state.assign(SDIM, 0.f);
      ly.conv_state.assign(CONV_DIM*(CONV_K-1), 0.f);
    }
  }
  const float* final_norm = get("model.language_model.norm.weight").data();
  { for(int L=0;L<LAYERS;L++) printf("layer %d is_full=%d\n", L, layers[L].is_full); }

  // tokenize prompt crudely? No — accept space-separated token ids for now.
  std::vector<int> prompt_ids;
  if(strcmp(prompt, "ids")==0){
    for(int i=4;i<argc;i++) prompt_ids.push_back(atoi(argv[i]));
  } else {
    // Hard-coded tokenization for the standard test prompt is not available here;
    // instead we read ids from a file if present.
    FILE* f = fopen("test/models/prompt_ids.txt","r");
    if(f){ int x; while(fscanf(f,"%d",&x)==1) prompt_ids.push_back(x); fclose(f); }
    else { printf("no prompt_ids.txt; using generic fallback\n"); prompt_ids={760,6511,314,9338,369}; }
  }
  int S = (int)prompt_ids.size();
  printf("prompt ids (%d):", S); for(int x:prompt_ids) printf(" %d", x); printf("\n");

  // dump buffers: per-layer hidden after each prefill position
  std::vector<std::vector<float>> dump(LAYERS, std::vector<float>(S*H, 0.f));

  std::vector<int> generated;
  int maxlen = 1024;

  auto step = [&](int token, int pos, bool record){
    const float* row = embed + (size_t)token*H;
    std::vector<float> h(row, row+H);
    for(int L=0; L<LAYERS; L++){
      Layer& ly = layers[L];
      std::vector<float> res = h;
      std::vector<float> normed(H);
      rmsnorm(h.data(), ly.in_ln, normed.data(), H);
      if(ly.is_full){
        // q_proj -> (N_HEADS, 2*HEAD_DIM) interleaved: per head, first 256 = query, last 256 = gate
        std::vector<float> qp = linear({ly.q_proj,ly.q_proj+4096*H}, normed, 4096, H);
        std::vector<float> gate(N_HEADS*HEAD_DIM);
        std::vector<float> qpart(N_HEADS*HEAD_DIM);
        for(int hh=0; hh<N_HEADS; hh++){
          for(int d=0; d<HEAD_DIM; d++){
            qpart[hh*HEAD_DIM + d] = qp[hh*2*HEAD_DIM + d];
            gate[hh*HEAD_DIM + d]  = qp[hh*2*HEAD_DIM + HEAD_DIM + d];
          }
        }
        // q_norm per head (8,256)
        for(int hh=0; hh<N_HEADS; hh++){
          float tmp[HEAD_DIM];
          rmsnorm(&qpart[hh*HEAD_DIM], ly.q_norm_w, tmp, HEAD_DIM);
          memcpy(&qpart[hh*HEAD_DIM], tmp, HEAD_DIM*sizeof(float));
          rope(&qpart[hh*HEAD_DIM], pos);
        }
        std::vector<float> k = linear({ly.k_proj,ly.k_proj+512*H}, normed, 512, H);
        std::vector<float> v = linear({ly.v_proj,ly.v_proj+512*H}, normed, 512, H);
        for(int hh=0; hh<N_KV; hh++){
          float tmp[HEAD_DIM];
          rmsnorm(&k[hh*HEAD_DIM], ly.k_norm_w, tmp, HEAD_DIM);
          memcpy(&k[hh*HEAD_DIM], tmp, HEAD_DIM*sizeof(float));
          rope(&k[hh*HEAD_DIM], pos);
        }
        // append to cache
        int base = pos*(N_KV*HEAD_DIM);
        memcpy(&ly.kcache[base], k.data(), N_KV*HEAD_DIM*sizeof(float));
        memcpy(&ly.vcache[base], v.data(), N_KV*HEAD_DIM*sizeof(float));
        // attention
        std::vector<float> attn(N_HEADS*HEAD_DIM, 0.f);
        for(int hh=0; hh<N_HEADS; hh++){
          int kvh = hh/(N_HEADS/N_KV);
          std::vector<float> scores(pos+1);
          float mx=-1e30f;
          for(int t=0;t<=pos;t++){
            const float* qh=&qpart[hh*HEAD_DIM];
            const float* kh=&ly.kcache[(t*N_KV+kvh)*HEAD_DIM];
            float s=0; for(int d=0;d<HEAD_DIM;d++) s+=qh[d]*kh[d];
            s*= 1.0f/sqrtf((float)HEAD_DIM);
            scores[t]=s; if(s>mx) mx=s;
          }
          float sum=0; for(int t=0;t<=pos;t++){ scores[t]=expf(scores[t]-mx); sum+=scores[t]; }
          for(int t=0;t<=pos;t++) scores[t]/=sum;
          float* oh=&attn[hh*HEAD_DIM];
          for(int d=0;d<HEAD_DIM;d++) oh[d]=0;
          for(int t=0;t<=pos;t++){
            const float* vh=&ly.vcache[(t*N_KV+kvh)*HEAD_DIM];
            for(int d=0;d<HEAD_DIM;d++) oh[d]+=scores[t]*vh[d];
          }
        }
        // * sigmoid(gate), then o_proj
        for(int i=0;i<N_HEADS*HEAD_DIM;i++) attn[i]*=sigmoidf(gate[i]);
        h = linear({ly.o_proj,ly.o_proj+1024*(N_HEADS*HEAD_DIM)}, attn, H, N_HEADS*HEAD_DIM);
      } else {
        // SSM gated delta rule
        std::vector<float> mqkv = linear({ly.in_qkv,ly.in_qkv+CONV_DIM*H}, normed, CONV_DIM, H);
        // depthwise causal conv (single token)
        std::vector<float> conv_out(CONV_DIM);
        for(int c=0;c<CONV_DIM;c++){
          float s=0;
          // new = [conv_state(3), mqkv(1)]
          s += ly.conv_w[c*CONV_K+0]*ly.conv_state[c*3+0];
          s += ly.conv_w[c*CONV_K+1]*ly.conv_state[c*3+1];
          s += ly.conv_w[c*CONV_K+2]*ly.conv_state[c*3+2];
          s += ly.conv_w[c*CONV_K+3]*mqkv[c];
          conv_out[c]=silu(s);
          // update conv_state = [past1,past2,cur]
          ly.conv_state[c*3+0]=ly.conv_state[c*3+1];
          ly.conv_state[c*3+1]=ly.conv_state[c*3+2];
          ly.conv_state[c*3+2]=mqkv[c];
        }
        float* query = conv_out.data();
        float* key   = conv_out.data()+K_HEAD_DIM*N_K_HEADS;        // +2048
        float* value = conv_out.data()+2*K_HEAD_DIM*N_K_HEADS;      // +4096
        std::vector<float> z = linear({ly.in_z,ly.in_z+ (N_V_HEADS*V_HEAD_DIM)*H}, normed, N_V_HEADS*V_HEAD_DIM, H);
        std::vector<float> b = linear({ly.in_b,ly.in_b+ N_V_HEADS*H}, normed, N_V_HEADS, H);
        std::vector<float> a = linear({ly.in_a,ly.in_a+ N_V_HEADS*H}, normed, N_V_HEADS, H);
        std::vector<float> g(N_V_HEADS);
        for(int hh=0; hh<N_V_HEADS; hh++){
          float sp = log1pf(expf(a[hh]+ly.dt_bias[hh])); // softplus
          g[hh] = -expf(ly.A_log[hh]) * sp;
        }
        std::vector<float> core(N_V_HEADS*V_HEAD_DIM, 0.f);
        float qn[K_HEAD_DIM], kn[K_HEAD_DIM];
        for(int hh=0; hh<N_V_HEADS; hh++){
          // GQA-style grouping (HF fix_query_key_value_ordering): value
          // head hh shares key head kh. No-op when N_V==N_K.
          int kh = (N_V_HEADS > N_K_HEADS) ? (hh * N_K_HEADS / N_V_HEADS) : hh;
          l2norm(&query[kh*K_HEAD_DIM], qn, K_HEAD_DIM);
          l2norm(&key[kh*K_HEAD_DIM],  kn, K_HEAD_DIM);
          float scale = 1.0f/sqrtf((float)K_HEAD_DIM);
          for(int d=0;d<K_HEAD_DIM;d++) qn[d]*=scale; // torch: query = query*scale
          float g_t = expf(g[hh]);
          float beta = sigmoidf(b[hh]);
          float* S = &ly.ssm_state[hh*K_HEAD_DIM*V_HEAD_DIM];
          // S = S * g_t
          for(int i=0;i<K_HEAD_DIM*V_HEAD_DIM;i++) S[i]*=g_t;
          // kv_mem = S @ k
          std::vector<float> kv_mem(V_HEAD_DIM,0.f);
          for(int d=0;d<K_HEAD_DIM;d++) for(int c=0;c<V_HEAD_DIM;c++) kv_mem[c]+=S[d*V_HEAD_DIM+c]*kn[d];
          // delta = (v - kv_mem)*beta
          std::vector<float> delta(V_HEAD_DIM);
          const float* vh=&value[hh*V_HEAD_DIM];
          for(int c=0;c<V_HEAD_DIM;c++) delta[c]=(vh[c]-kv_mem[c])*beta;
          // S += outer(k, delta)
          for(int d=0;d<K_HEAD_DIM;d++) for(int c=0;c<V_HEAD_DIM;c++) S[d*V_HEAD_DIM+c]+=kn[d]*delta[c];
          // out = S @ q
          for(int c=0;c<V_HEAD_DIM;c++){ float s=0; for(int d=0;d<K_HEAD_DIM;d++) s+=S[d*V_HEAD_DIM+c]*qn[d]; core[hh*V_HEAD_DIM+c]=s; }
        }
        // RMSNormGated: per (V_HEAD_DIM) block with shared weight, then * silu(z)
        std::vector<float> normed_core(N_V_HEADS*V_HEAD_DIM);
        for(int hh=0; hh<N_V_HEADS; hh++){
          const float* x=&core[hh*V_HEAD_DIM];
          const float* zg=&z[hh*V_HEAD_DIM];
          float* o=&normed_core[hh*V_HEAD_DIM];
          float v=0; for(int i=0;i<V_HEAD_DIM;i++) v+=x[i]*x[i];
          float inv=1.0f/sqrtf(v/V_HEAD_DIM + RMS_EPS);
          for(int i=0;i<V_HEAD_DIM;i++) o[i]=(x[i]*inv)*ly.ssm_norm_w[i]*silu(zg[i]);
        }
        h = linear({ly.ssm_out,ly.ssm_out+ H*(N_V_HEADS*V_HEAD_DIM)}, normed_core, H, N_V_HEADS*V_HEAD_DIM);
      }
      for(int i=0;i<H;i++) h[i]+=res[i]; // residual
      // MLP
      std::vector<float> rn2(H);
      rmsnorm(h.data(), ly.post_ln, rn2.data(), H);
      std::vector<float> mg = linear({ly.mlp_gate,ly.mlp_gate+INTER*H}, rn2, INTER, H);
      std::vector<float> mu = linear({ly.mlp_up,  ly.mlp_up+  INTER*H}, rn2, INTER, H);
      for(int i=0;i<INTER;i++) mg[i]=silu(mg[i])*mu[i];
      std::vector<float> mlp_out = linear({ly.mlp_down,ly.mlp_down+H*INTER}, mg, H, INTER);
      for(int i=0;i<H;i++) h[i]=h[i]+mlp_out[i];
      if(record) memcpy(&dump[L][pos*H], h.data(), H*sizeof(float));
    }
    // final norm
    std::vector<float> fn(H);
    rmsnorm(h.data(), final_norm, fn.data(), H);
    // logits = fn @ embed^T
    int best=-1; float bestv=-1e30f;
    for(int v=0; v<VOCAB; v++){
      const float* er = embed + (size_t)v*H;
      float s=0; for(int i=0;i<H;i++) s+=fn[i]*er[i];
      if(s>bestv){ bestv=s; best=v; }
    }
    return best;
  };

  // prefill
  for(int pos=0; pos<S; pos++){
    int nxt = step(prompt_ids[pos], pos, true);
    if(pos==S-1){ generated.push_back(nxt); printf("prefill last -> %d\n", nxt); }
  }
  // generate
  int token = generated.back();
  for(int step_i=1; step_i<30; step_i++){
    int nxt = step(token, S+step_i-1, false);
    generated.push_back(nxt);
    token = nxt;
    if(nxt==0) break;
  }
  printf("generated:"); for(int x:generated) printf(" %d", x); printf("\n");

  // dump binary
  FILE* f = fopen(dump_path,"wb");
  int hdr[3]={LAYERS, S, H};
  fwrite(hdr,4,3,f);
  for(int L=0;L<LAYERS;L++) fwrite(dump[L].data(), 4, S*H, f);
  // final-norm hidden at last prefill pos
  fclose(f);
  printf("wrote %s\n", dump_path);
  return 0;
}
