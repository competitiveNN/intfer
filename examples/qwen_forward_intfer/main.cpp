// examples/qwen_forward_intfer/main.cpp
// CPU reference forward for Qwen3.5-0.8B (text-only) that uses the *quantized*
// Stage-1 weights from the .intfer artifact (dequantized on the fly, exactly as
// the SYCL XMX/DPAS GEMM kernel does) for MLP + attention Linears, and the
// original safetensors for the SSM / norms / embeddings. Crucially, the .intfer
// weights are SpinQuant-rotated, so this replays the online rotation:
//   right-site (q/k/v, gate/up):  x' = x R   BEFORE the GEMM
//   left-site  (o, down):         y  = y' R  AFTER  the GEMM
// This is the same SpinQuant online replay the Stage-2 engine must perform.
// Verified against scripts/oracle_qwen35.py (tokens + per-layer cosine 1.0).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include "intfer/safetensors.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/spinquant.hpp"

using namespace intfer;

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
static const int   ROTARY_DIM  = 64;
static const float RMS_EPS  = 1e-6f;
static const int   VOCAB    = 248320;
static const int   CONV_DIM = N_K_HEADS*K_HEAD_DIM*2 + N_V_HEADS*V_HEAD_DIM;
static const int   SDIM     = N_V_HEADS*K_HEAD_DIM*V_HEAD_DIM;

static float inv_freq[ROTARY_DIM/2];

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
  for(int i=0;i<dim;i++) out[i] = (x[i]*v)*(1.0f+w[i]);
}
static void l2norm(const float* x, float* out, int dim){
  float s=0; for(int i=0;i<dim;i++) s+=x[i]*x[i];
  float inv=1.0f/sqrtf(s+1e-6f);
  for(int i=0;i<dim;i++) out[i]=x[i]*inv;
}
static void rope(float* h, int pos){
  for(int d=0; d<ROTARY_DIM/2; d++){
    float f = inv_freq[d]*pos;
    float c=cosf(f), s=sinf(f);
    float a=h[d], b=h[d+ROTARY_DIM/2];
    h[d]          = a*c - b*s;
    h[d+ROTARY_DIM/2] = a*s + b*c;
  }
}
// SpinQuant online rotation helpers. rot_apply = x' = x R = Rᵀ·x (right-multiply,
// used on the activation for RIGHT-site tensors before the GEMM).
// rot_left  = y = R·y' (left-multiply, used to undo the rotation on the OUTPUT of
// LEFT-site tensors after the GEMM: stored weight is Rᵀ·W_orig, so R·(W'·x)=W_orig·x).
static void rot_apply(const float* x, const Rotation& R, float* out){
  int dim = (int)R.dim;
  for(int j=0;j<dim;j++){ float s=0; for(int i=0;i<dim;i++) s+=x[i]*R.mat[i*dim+j]; out[j]=s; }
}
static void rot_left(const float* y, const Rotation& R, float* out){
  int dim = (int)R.dim;
  for(int j=0;j<dim;j++){ float s=0; for(int i=0;i<dim;i++) s+=R.mat[j*dim+i]*y[i]; out[j]=s; }
}

// A quantized Linear with its SpinQuant rotation replayed online.
struct QLinear { std::vector<float> w; const Rotation* R = nullptr; };

struct Layer {
  bool is_full;
  // safetensors (SSM + norms)
  const float *in_ln,*post_ln,*q_norm_w,*k_norm_w;
  const float *in_qkv,*in_z,*in_b,*in_a,*conv_w,*A_log,*dt_bias,*ssm_norm_w,*ssm_out;
  // quantized .intfer weights (attn for full layers, mlp all layers)
  QLinear q_proj,k_proj,v_proj,o_proj, mlp_gate,mlp_up,mlp_down;
  std::vector<float> kcache, vcache;
  std::vector<float> ssm_state, conv_state;
};

static std::vector<Layer> layers(LAYERS);

// ---- Quantized store built from the .intfer ----
static std::map<int, std::map<std::string, QLinear>> attn_q;  // L -> kind -> QLinear
static std::map<int, std::map<std::string, QLinear>> mlp_q;   // L -> kind -> QLinear

static bool read_blob(const std::string& p, std::vector<uint8_t>& out){
  FILE* f=fopen(p.c_str(),"rb"); if(!f) return false;
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); out.resize(n);
  size_t r=fread(out.data(),1,n,f); fclose(f); return r==(size_t)n;
}

int main(int argc, char** argv){
  const char* st_path = "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
  const char* intfer_path = "test/models/Qwen3.5-0.8B.intfer";
  for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--intfer") && i+1<argc) intfer_path=argv[++i]; }

  for(int d=0; d<ROTARY_DIM/2; d++) inv_freq[d] = 1.0f/powf((float)ROPE_THETA, (2.0f*d)/(float)ROTARY_DIM);

  // --- original safetensors (SSM, norms, embed) ---
  SafetensorsReader r;
  if(!r.open(st_path)){ printf("FAIL open safetensors\n"); return 1; }
  std::map<std::string, std::vector<float>> T;
  for(const auto& t : r.tensors()){
    std::vector<float> v; if(!r.read(t.name,v)){ printf("FAIL read %s\n",t.name.c_str()); return 1; }
    T[t.name]=std::move(v);
  }
  printf("loaded %zu safetensors tensors\n", T.size());
  auto get=[&](const std::string& n)->const std::vector<float>&{ return T.at(n); };

  // --- .intfer (quantized + SpinQuant-rotated MLP + attn) ---
  std::vector<uint8_t> blob;
  if(!read_blob(intfer_path, blob)){ printf("FAIL read intfer %s\n", intfer_path); return 1; }
  ModelIR m; RotationSet rs;
  if(!deserialize_intfer(blob.data(), blob.size(), m, rs)){ printf("FAIL deserialize intfer\n"); return 1; }
  std::map<uint8_t, const Rotation*> rid_map;
  for(const auto& R : rs.rotations) rid_map[R.id] = &R;
  printf("intfer tensors=%zu rotations=%zu\n", m.tensors.size(), rs.size());

  for(const auto& t : m.tensors){
    // Skip MTP-head tensors: they are named "mtp.layers.<L>.*" and would otherwise
    // overwrite the main-model weights in attn_q/mlp_q. We only want the main model
    // ("model.language_model.layers.<L>.*").
    size_t p = t.name.find("model.language_model.layers.");
    if(p==std::string::npos) continue;
    QLinear ql; intfer::dashq::dequant_tensor(m, t, ql.w); ql.R = rid_map.count(t.qcfg.rotation_id)? rid_map[t.qcfg.rotation_id] : nullptr;
    int L = atoi(t.name.c_str()+p+28);
    std::string suf = t.name.substr(p+28); while(!suf.empty() && suf[0]>='0' && suf[0]<='9') suf=suf.substr(1);
    if(suf.find("self_attn.q_proj")!=std::string::npos) attn_q[L]["q"]=ql;
    else if(suf.find("self_attn.k_proj")!=std::string::npos) attn_q[L]["k"]=ql;
    else if(suf.find("self_attn.v_proj")!=std::string::npos) attn_q[L]["v"]=ql;
    else if(suf.find("self_attn.o_proj")!=std::string::npos) attn_q[L]["o"]=ql;
    else if(suf.find("mlp.gate_proj")!=std::string::npos) mlp_q[L]["gate"]=ql;
    else if(suf.find("mlp.up_proj")!=std::string::npos) mlp_q[L]["up"]=ql;
    else if(suf.find("mlp.down_proj")!=std::string::npos) mlp_q[L]["down"]=ql;
  }

  const float* embed = get("model.language_model.embed_tokens.weight").data();
  std::map<int, bool> has_attn;
  for(int L=0; L<LAYERS; L++){
    char b[256]; auto p=[&](const char* s){ snprintf(b,sizeof(b),"model.language_model.layers.%d.%s",L,s); return std::string(b); };
    Layer& ly=layers[L];
    ly.is_full = T.count(p("self_attn.q_proj.weight"))>0;
    ly.in_ln = get(p("input_layernorm.weight")).data();
    ly.post_ln= get(p("post_attention_layernorm.weight")).data();
    if(ly.is_full){
      ly.q_norm_w = get(p("self_attn.q_norm.weight")).data();
      ly.k_norm_w = get(p("self_attn.k_norm.weight")).data();
      // quantized attn weights
      ly.q_proj = attn_q[L]["q"]; ly.k_proj = attn_q[L]["k"]; ly.v_proj = attn_q[L]["v"]; ly.o_proj = attn_q[L]["o"];
      ly.kcache.assign(N_KV*1024*HEAD_DIM, 0.f);
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
    ly.mlp_gate = mlp_q[L]["gate"]; ly.mlp_up = mlp_q[L]["up"]; ly.mlp_down = mlp_q[L]["down"];
  }
  const float* final_norm = get("model.language_model.norm.weight").data();

  // prompt
  std::vector<int> prompt_ids;
  FILE* f=fopen("test/models/prompt_ids.txt","r");
  if(f){ int x; while(fscanf(f,"%d",&x)==1) prompt_ids.push_back(x); fclose(f); }
  else prompt_ids={760,6511,314,9338,369};
  int S=(int)prompt_ids.size();
  printf("prompt ids (%d):", S); for(int x:prompt_ids) printf(" %d",x); printf("\n");

  std::vector<std::vector<float>> dump(LAYERS, std::vector<float>(S*H,0.f));
  std::vector<int> generated;
  int maxlen=1024;

  auto step=[&](int token, int pos, bool record){
    const float* row=embed+(size_t)token*H;
    std::vector<float> h(row, row+H);
    for(int L=0; L<LAYERS; L++){
      Layer& ly=layers[L];
      std::vector<float> res=h;
      std::vector<float> normed(H);
      rmsnorm(h.data(), ly.in_ln, normed.data(), H);
      if(ly.is_full){
        // right-site rotation (q/k/v) on the activation before GEMM
        std::vector<float> rn(normed);
        if(ly.q_proj.R) rot_apply(normed.data(), *ly.q_proj.R, rn.data());
        std::vector<float> qp = linear(ly.q_proj.w, rn, 4096, H);
        std::vector<float> gate(N_HEADS*HEAD_DIM);
        std::vector<float> qpart(N_HEADS*HEAD_DIM);
        for(int hh=0; hh<N_HEADS; hh++) for(int d=0; d<HEAD_DIM; d++){
          qpart[hh*HEAD_DIM+d] = qp[hh*2*HEAD_DIM+d];
          gate[hh*HEAD_DIM+d]  = qp[hh*2*HEAD_DIM+HEAD_DIM+d];
        }
        for(int hh=0; hh<N_HEADS; hh++){
          float tmp[HEAD_DIM]; rmsnorm(&qpart[hh*HEAD_DIM], ly.q_norm_w, tmp, HEAD_DIM);
          memcpy(&qpart[hh*HEAD_DIM], tmp, HEAD_DIM*sizeof(float));
          rope(&qpart[hh*HEAD_DIM], pos);
        }
        std::vector<float> rk(normed), rv(normed);
        if(ly.k_proj.R) rot_apply(normed.data(), *ly.k_proj.R, rk.data());
        if(ly.v_proj.R) rot_apply(normed.data(), *ly.v_proj.R, rv.data());
        std::vector<float> k = linear(ly.k_proj.w, rk, 512, H);
        std::vector<float> v = linear(ly.v_proj.w, rv, 512, H);
        for(int hh=0; hh<N_KV; hh++){
          float tmp[HEAD_DIM]; rmsnorm(&k[hh*HEAD_DIM], ly.k_norm_w, tmp, HEAD_DIM);
          memcpy(&k[hh*HEAD_DIM], tmp, HEAD_DIM*sizeof(float));
          rope(&k[hh*HEAD_DIM], pos);
        }
        int base=pos*(N_KV*HEAD_DIM);
        memcpy(&ly.kcache[base], k.data(), N_KV*HEAD_DIM*sizeof(float));
        memcpy(&ly.vcache[base], v.data(), N_KV*HEAD_DIM*sizeof(float));
        std::vector<float> attn(N_HEADS*HEAD_DIM, 0.f);
        for(int hh=0; hh<N_HEADS; hh++){
          int kvh=hh/(N_HEADS/N_KV);
          std::vector<float> scores(pos+1); float mx=-1e30f;
          for(int t=0;t<=pos;t++){
            const float* qh=&qpart[hh*HEAD_DIM]; const float* kh=&ly.kcache[(t*N_KV+kvh)*HEAD_DIM];
            float s=0; for(int d=0;d<HEAD_DIM;d++) s+=qh[d]*kh[d];
            s*=1.0f/sqrtf((float)HEAD_DIM); scores[t]=s; if(s>mx) mx=s;
          }
          float sum=0; for(int t=0;t<=pos;t++){ scores[t]=expf(scores[t]-mx); sum+=scores[t]; }
          for(int t=0;t<=pos;t++) scores[t]/=sum;
          float* oh=&attn[hh*HEAD_DIM]; for(int d=0;d<HEAD_DIM;d++) oh[d]=0;
          for(int t=0;t<=pos;t++){ const float* vh=&ly.vcache[(t*N_KV+kvh)*HEAD_DIM]; for(int d=0;d<HEAD_DIM;d++) oh[d]+=scores[t]*vh[d]; }
        }
        for(int i=0;i<N_HEADS*HEAD_DIM;i++) attn[i]*=sigmoidf(gate[i]);
        if (pos == 0 && L == 3 || pos == 5 && L == 3) {
          printf("DEBUG CPU L%d pos=%d attn*sig[:4]=%.4f %.4f %.4f %.4f\n", L, pos, attn[0], attn[1], attn[2], attn[3]);
        }
        std::vector<float> o_raw = linear(ly.o_proj.w, attn, H, N_HEADS*HEAD_DIM);
        if (pos == 0 && L == 3 || pos == 5 && L == 3) {
          printf("DEBUG CPU L%d pos=%d o_raw[:4]=%.4f %.4f %.4f %.4f\n", L, pos, o_raw[0], o_raw[1], o_raw[2], o_raw[3]);
        }
        if(ly.o_proj.R) rot_left(o_raw.data(), *ly.o_proj.R, h.data()); // left-site: R·(W'·x)
        else h=o_raw;
        if (pos == 0 && L == 3 || pos == 5 && L == 3) {
          printf("DEBUG CPU L%d pos=%d h_attn[:4]=%.4f %.4f %.4f %.4f\n", L, pos, h[0], h[1], h[2], h[3]);
        }
      } else {
        std::vector<float> mqkv = linear({ly.in_qkv, ly.in_qkv+CONV_DIM*H}, normed, CONV_DIM, H);
        std::vector<float> conv_out(CONV_DIM);
        for(int c=0;c<CONV_DIM;c++){
          float s=0;
          s+=ly.conv_w[c*CONV_K+0]*ly.conv_state[c*3+0];
          s+=ly.conv_w[c*CONV_K+1]*ly.conv_state[c*3+1];
          s+=ly.conv_w[c*CONV_K+2]*ly.conv_state[c*3+2];
          s+=ly.conv_w[c*CONV_K+3]*mqkv[c];
          conv_out[c]=silu(s);
          ly.conv_state[c*3+0]=ly.conv_state[c*3+1]; ly.conv_state[c*3+1]=ly.conv_state[c*3+2]; ly.conv_state[c*3+2]=mqkv[c];
        }
        float* query=conv_out.data();
        float* key=conv_out.data()+K_HEAD_DIM*N_K_HEADS;
        float* value=conv_out.data()+2*K_HEAD_DIM*N_K_HEADS;
        std::vector<float> z = linear({ly.in_z, ly.in_z+(N_V_HEADS*V_HEAD_DIM)*H}, normed, N_V_HEADS*V_HEAD_DIM, H);
        std::vector<float> b = linear({ly.in_b, ly.in_b+N_V_HEADS*H}, normed, N_V_HEADS, H);
        std::vector<float> a = linear({ly.in_a, ly.in_a+N_V_HEADS*H}, normed, N_V_HEADS, H);
        std::vector<float> g(N_V_HEADS);
        for(int hh=0; hh<N_V_HEADS; hh++){ float sp=log1pf(expf(a[hh]+ly.dt_bias[hh])); g[hh]=-expf(ly.A_log[hh])*sp; }
        std::vector<float> core(N_V_HEADS*V_HEAD_DIM,0.f);
        float qn[K_HEAD_DIM], kn[K_HEAD_DIM];
        for(int hh=0; hh<N_V_HEADS; hh++){
          l2norm(&query[hh*K_HEAD_DIM], qn, K_HEAD_DIM);
          l2norm(&key[hh*K_HEAD_DIM],  kn, K_HEAD_DIM);
          float scale=1.0f/sqrtf((float)K_HEAD_DIM);
          for(int d=0;d<K_HEAD_DIM;d++) qn[d]*=scale;
          float g_t=expf(g[hh]); float beta=sigmoidf(b[hh]);
          float* S=&ly.ssm_state[hh*K_HEAD_DIM*V_HEAD_DIM];
          for(int i=0;i<K_HEAD_DIM*V_HEAD_DIM;i++) S[i]*=g_t;
          std::vector<float> kv_mem(V_HEAD_DIM,0.f);
          for(int d=0;d<K_HEAD_DIM;d++) for(int c=0;c<V_HEAD_DIM;c++) kv_mem[c]+=S[d*V_HEAD_DIM+c]*kn[d];
          std::vector<float> delta(V_HEAD_DIM); const float* vh=&value[hh*V_HEAD_DIM];
          for(int c=0;c<V_HEAD_DIM;c++) delta[c]=(vh[c]-kv_mem[c])*beta;
          for(int d=0;d<K_HEAD_DIM;d++) for(int c=0;c<V_HEAD_DIM;c++) S[d*V_HEAD_DIM+c]+=kn[d]*delta[c];
          for(int c=0;c<V_HEAD_DIM;c++){ float s=0; for(int d=0;d<K_HEAD_DIM;d++) s+=S[d*V_HEAD_DIM+c]*qn[d]; core[hh*V_HEAD_DIM+c]=s; }
        }
        std::vector<float> normed_core(N_V_HEADS*V_HEAD_DIM);
        for(int hh=0; hh<N_V_HEADS; hh++){
          const float* x=&core[hh*V_HEAD_DIM]; const float* zg=&z[hh*V_HEAD_DIM]; float* o=&normed_core[hh*V_HEAD_DIM];
          float v=0; for(int i=0;i<V_HEAD_DIM;i++) v+=x[i]*x[i];
          float inv=1.0f/sqrtf(v/V_HEAD_DIM+RMS_EPS);
          for(int i=0;i<V_HEAD_DIM;i++) o[i]=(x[i]*inv)*ly.ssm_norm_w[i]*silu(zg[i]);
        }
        h = linear({ly.ssm_out, ly.ssm_out+H*(N_V_HEADS*V_HEAD_DIM)}, normed_core, H, N_V_HEADS*V_HEAD_DIM);
      }
      for(int i=0;i<H;i++) h[i]+=res[i];
      // MLP (quantized) with SpinQuant rotation replay
      std::vector<float> rn2(H);
      rmsnorm(h.data(), ly.post_ln, rn2.data(), H);
      std::vector<float> rin(rn2);
      if(ly.mlp_gate.R) rot_apply(rn2.data(), *ly.mlp_gate.R, rin.data()); // right-site FFN_IN
      std::vector<float> mg = linear(ly.mlp_gate.w, rin, INTER, H);
      std::vector<float> mu = linear(ly.mlp_up.w,   rin, INTER, H);
      for(int i=0;i<INTER;i++) mg[i]=silu(mg[i])*mu[i];
      std::vector<float> md = linear(ly.mlp_down.w, mg, H, INTER);
      std::vector<float> mlp_out(H);
      if(ly.mlp_down.R) rot_left(md.data(), *ly.mlp_down.R, mlp_out.data()); // left-site: R·(W'·x)
      else mlp_out=md;
      if (pos == 0 && L == 3 || pos == 5 && L == 3) {
        printf("DEBUG CPU L%d pos=%d mlp_out[:4]=%.4f %.4f %.4f %.4f\n", L, pos, mlp_out[0], mlp_out[1], mlp_out[2], mlp_out[3]);
      }
      for(int i=0;i<H;i++) h[i]=h[i]+mlp_out[i];
      if (pos == 0 && L == 3 || pos == 5 && L == 3) {
        printf("DEBUG CPU L%d pos=%d h[:4]=%.4f %.4f %.4f %.4f\n", L, pos, h[0], h[1], h[2], h[3]);
      }
      if(record) memcpy(&dump[L][pos*H], h.data(), H*sizeof(float));
    }
    std::vector<float> fn(H); rmsnorm(h.data(), final_norm, fn.data(), H);
    if (pos == 0 || pos == 5) {
      printf("DEBUG CPU pos=%d fn[:4]=%.4f %.4f %.4f %.4f\n", pos, fn[0], fn[1], fn[2], fn[3]);
      int top5[5] = {-1,-1,-1,-1,-1}; float topv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
      for(int v=0; v<VOCAB; v++){ const float* er=embed+(size_t)v*H; float s=0; for(int i=0;i<H;i++) s+=fn[i]*er[i]; if(s>topv[0]){for(int k=4;k>0;k--){topv[k]=topv[k-1];top5[k]=top5[k-1];}topv[0]=s;top5[0]=v;} else if(s>topv[1]){for(int k=4;k>2;k--){topv[k]=topv[k-1];top5[k]=top5[k-1];}topv[1]=s;top5[1]=v;} else if(s>topv[2]){for(int k=4;k>3;k--){topv[k]=topv[k-1];top5[k]=top5[k-1];}topv[2]=s;top5[2]=v;} }
      printf("DEBUG CPU pos=%d top5=%d(%.4f) %d(%.4f) %d(%.4f) %d(%.4f) %d(%.4f)\n", pos, top5[0],topv[0],top5[1],topv[1],top5[2],topv[2],top5[3],topv[3],top5[4],topv[4]);
    }
    int best=-1; float bestv=-1e30f;
    for(int v=0; v<VOCAB; v++){ const float* er=embed+(size_t)v*H; float s=0; for(int i=0;i<H;i++) s+=fn[i]*er[i]; if(s>bestv){bestv=s; best=v;} }
    return best;
  };

  for(int pos=0; pos<S; pos++){
    int nxt=step(prompt_ids[pos], pos, true);
    if(pos==S-1){ generated.push_back(nxt); printf("prefill last -> %d\n", nxt); }
  }
  int token=generated.back();
  for(int step_i=1; step_i<30; step_i++){
    int nxt=step(token, S+step_i-1, false);
    generated.push_back(nxt); token=nxt; if(nxt==0) break;
  }
  printf("generated:"); for(int x:generated) printf(" %d", x); printf("\n");

  FILE* g=fopen("test/models/qwen35_intfer.bin","wb");
  int32_t hdr[3]={(int32_t)LAYERS,(int32_t)S,(int32_t)H}; fwrite(hdr,4,3,g);
  for(int L=0;L<LAYERS;L++) fwrite(dump[L].data(),4,(size_t)S*H,g);
  fclose(g);
  return 0;
}
