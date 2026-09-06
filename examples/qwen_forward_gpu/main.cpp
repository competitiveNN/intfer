// examples/qwen_forward_gpu/main.cpp
// Full Qwen3.5-0.8B text forward on the 2x Intel Arc Pro B580 via the intfer
// Stage-2 engine. Quantized MLP + attention Linears run as tensor-parallel
// dequant-on-the-fly GEMMs (quant_gemm_forward / XMX-DPAS) across the fleet via
// Engine::run_linear_tp_ret; attention runs as a fused FlashAttention kernel on
// one device; SSM / RMSNorm / mRoPE / sigmoid-gate / silu / residual / final-norm
// / lm_head run on the host (non-quantized, Stage-1-free). The SpinQuant online
// rotation is replayed on the host activation before each right-site GEMM and
// after each left-site GEMM (same as the verified CPU reference).
//
// Verified against examples/qwen_forward_intfer (CPU) which matches the
// transformers oracle: greedy tokens 11751 13 198 760 6511 314 9338 369 ...
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "intfer/engine.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/safetensors.hpp"
#include "intfer/spinquant.hpp"
#include "intfer/attention/flash_attention.hpp"
#include "intfer/gemm/quant_gemm.hpp"

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
static const int   ROTARY_DIM  = 64;
static const float RMS_EPS  = 1e-6f;
static const int   VOCAB    = 248320;
static const int   CONV_DIM = N_K_HEADS*K_HEAD_DIM*2 + N_V_HEADS*V_HEAD_DIM;
static const int   SDIM     = N_V_HEADS*K_HEAD_DIM*V_HEAD_DIM;

static float inv_freq[ROTARY_DIM/2];

static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }

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
// SpinQuant online rotation: rot_apply = x' = x R (right-multiply, activation
// side for RIGHT-site tensors before the GEMM); rot_left = y = R y' (left-site
// tensors after the GEMM: stored weight is R^T W_orig so R(W'x)=W_orig x).
static void rot_apply(const float* x, const Rotation& R, float* out){
  int dim=(int)R.dim;
  for(int j=0;j<dim;j++){ float s=0; for(int i=0;i<dim;i++) s+=x[i]*R.mat[i*dim+j]; out[j]=s; }
}
static void rot_left(const float* y, const Rotation& R, float* out){
  int dim=(int)R.dim;
  for(int j=0;j<dim;j++){ float s=0; for(int i=0;i<dim;i++) s+=R.mat[j*dim+i]*y[i]; out[j]=s; }
}
static std::vector<std::uint16_t> to_fp16(const std::vector<float>& v){
  std::vector<std::uint16_t> o(v.size());
  for(size_t i=0;i<v.size();i++) o[i]=dashq::float_to_half(v[i]);
  return o;
}
static std::vector<float> linear_host(const float* W, const float* x, int N, int K){
  std::vector<float> y(N, 0.f);
  for(int i=0;i<N;i++){ const float* wr=W+(size_t)i*K; float s=0; for(int k=0;k<K;k++) s+=wr[k]*x[k]; y[i]=s; }
  return y;
}

static std::vector<int> tokenize(const std::string& text) {
  std::string cmd = "echo " + std::string("\"") + text + std::string("\"") + " | python3 /var/home/fra/dev/intfer/scripts/bpe_tokenize.py";
  FILE* fp = popen(cmd.c_str(), "rb");
  if (!fp) return {};
  std::vector<int> ids;
  int val;
  while (fread(&val, sizeof(int), 1, fp) == 1) ids.push_back(val);
  pclose(fp);
  return ids;
}

int main(int argc, char** argv){
  const char* intfer_path = (argc>1)? argv[1] : "test/models/Qwen3.5-0.8B.intfer";
  const char* st_path = (argc>2)? argv[2] :
    "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";

  for(int d=0; d<ROTARY_DIM/2; d++) inv_freq[d] = 1.0f/powf((float)ROPE_THETA, (2.0f*d)/(float)ROTARY_DIM);

  auto eng = Engine::load(intfer_path, 0);
  if(!eng){ printf("engine load failed\n"); return 1; }
  printf("engine devices=%zu layers=%zu\n", eng->num_devices(), eng->num_layers());

  // load safetensors (embed, norms, SSM) on the host
  SafetensorsReader sr;
  if(!sr.open(st_path)){ printf("open safetensors failed: %s\n", st_path); return 1; }
  std::map<std::string,std::vector<float>> T;
  for(const auto& t: sr.tensors()){ std::vector<float> v; if(!sr.read(t.name,v)){ printf("read fail %s\n",t.name.c_str()); return 1; } T[t.name]=std::move(v); }
  printf("loaded %zu safetensors tensors\n", T.size());
  auto get=[&](const std::string& n)->const std::vector<float>&{ return T.at(n); };

  const ModelIR& m = eng->model();
  auto find_ti=[&](const std::string& nm)->int{
    for(size_t i=0;i<m.tensors.size();i++) if(m.tensors[i].name==nm) return (int)i;
    return -1;
  };
  int t_q[LAYERS], t_k[LAYERS], t_v[LAYERS], t_o[LAYERS];
  int t_gate[LAYERS], t_up[LAYERS], t_down[LAYERS];
  bool is_full[LAYERS];
  for(int L=0; L<LAYERS; L++){
    auto p=[&](const char* s){ char b[256]; snprintf(b,sizeof(b),"model.language_model.layers.%d.%s",L,s); return std::string(b); };
    t_q[L]=find_ti(p("self_attn.q_proj.weight"));
    t_k[L]=find_ti(p("self_attn.k_proj.weight"));
    t_v[L]=find_ti(p("self_attn.v_proj.weight"));
    t_o[L]=find_ti(p("self_attn.o_proj.weight"));
    t_gate[L]=find_ti(p("mlp.gate_proj.weight"));
    t_up[L]=find_ti(p("mlp.up_proj.weight"));
    t_down[L]=find_ti(p("mlp.down_proj.weight"));
    is_full[L] = (t_q[L]>=0);
  }
  const float* embed = get("model.language_model.embed_tokens.weight").data();
  const float* final_norm = get("model.language_model.norm.weight").data();

  // KV caches + SSM state
  std::vector<std::vector<float>> kc(LAYERS), vc(LAYERS);
  std::vector<std::vector<float>> ssm_state(LAYERS, std::vector<float>(SDIM,0.f));
  std::vector<std::vector<float>> conv_state(LAYERS, std::vector<float>(CONV_DIM*(CONV_K-1),0.f));

  // prompt
  std::vector<int> prompt_ids;
  if (argc > 3) {
    std::string text = argv[3];
    prompt_ids = tokenize(text);
  } else {
    FILE* f=fopen("test/models/prompt_ids.txt","r");
    if(f){ int x; while(fscanf(f,"%d",&x)==1) prompt_ids.push_back(x); fclose(f); }
    else prompt_ids={760,6511,314,9338,369};
  }
  int S=(int)prompt_ids.size();
  printf("prompt (%d):", S); for(int x:prompt_ids) printf(" %d",x); printf("\n");

  std::vector<int> generated;
  int maxlen=30;

  auto step=[&](int token, int pos, bool record)->int{
    const float* row=embed+(size_t)token*H;
    std::vector<float> h(row, row+H);
    for(int L=0; L<LAYERS; L++){
      auto p=[&](const char* s){ char b[256]; snprintf(b,sizeof(b),"model.language_model.layers.%d.%s",L,s); return std::string(b); };
      std::vector<float> res=h;
      std::vector<float> normed(H);
      rmsnorm(h.data(), get(p("input_layernorm.weight")).data(), normed.data(), H);
      if(is_full[L]){
        const Rotation* Rq=eng->rotation_for(t_q[L]);
        std::vector<float> xq(normed); if(Rq) rot_apply(normed.data(),*Rq,xq.data());
        std::vector<float> qp = eng->run_linear_tp_ret(t_q[L], xq); // N=4096
        std::vector<float> qpart(N_HEADS*HEAD_DIM), gate(N_HEADS*HEAD_DIM);
        for(int hh=0; hh<N_HEADS; hh++) for(int d=0; d<HEAD_DIM; d++){
          qpart[hh*HEAD_DIM+d] = qp[hh*2*HEAD_DIM+d];
          gate[hh*HEAD_DIM+d]  = qp[hh*2*HEAD_DIM+HEAD_DIM+d];
        }
        for(int hh=0; hh<N_HEADS; hh++){
          float tmp[HEAD_DIM]; rmsnorm(&qpart[hh*HEAD_DIM], get(p("self_attn.q_norm.weight")).data(), tmp, HEAD_DIM);
          memcpy(&qpart[hh*HEAD_DIM], tmp, HEAD_DIM*sizeof(float));
          rope(&qpart[hh*HEAD_DIM], pos);
        }
        const Rotation* Rk=eng->rotation_for(t_k[L]);
        std::vector<float> xk(normed); if(Rk) rot_apply(normed.data(),*Rk,xk.data());
        const Rotation* Rv=eng->rotation_for(t_v[L]);
        std::vector<float> xv(normed); if(Rv) rot_apply(normed.data(),*Rv,xv.data());
        std::vector<float> k = eng->run_linear_tp_ret(t_k[L], xk); // N=512
        std::vector<float> v = eng->run_linear_tp_ret(t_v[L], xv); // N=512
        for(int hh=0; hh<N_KV; hh++){
          float tmp[HEAD_DIM]; rmsnorm(&k[hh*HEAD_DIM], get(p("self_attn.k_norm.weight")).data(), tmp, HEAD_DIM);
          memcpy(&k[hh*HEAD_DIM], tmp, HEAD_DIM*sizeof(float));
          rope(&k[hh*HEAD_DIM], pos);
        }
        kc[L].insert(kc[L].end(), k.begin(), k.end());
        vc[L].insert(vc[L].end(), v.begin(), v.end());
        // expanded k/v in [batch, heads, seq, HEAD_DIM] (the layout flash_attention
        // expects: off3 = ((b*H+h)*S+t)*D). KV head hh/(N_HEADS/N_KV) is shared by
        // the query heads. NOTE: head must be the OUTER dim, seq the middle.
        int seq=pos+1;
        std::vector<std::uint16_t> kfa((size_t)1*N_HEADS*seq*HEAD_DIM), vfa((size_t)1*N_HEADS*seq*HEAD_DIM);
        for(int hh=0;hh<N_HEADS;hh++){
          int kvh=hh/(N_HEADS/N_KV);
          for(int t=0;t<seq;t++) for(int d=0;d<HEAD_DIM;d++){
            kfa[((size_t)hh*seq+t)*HEAD_DIM+d] = dashq::float_to_half(kc[L][((size_t)t*N_KV+kvh)*HEAD_DIM+d]);
            vfa[((size_t)hh*seq+t)*HEAD_DIM+d] = dashq::float_to_half(vc[L][((size_t)t*N_KV+kvh)*HEAD_DIM+d]);
          }
        }
        std::vector<std::uint16_t> qfa = to_fp16(qpart); // [N_HEADS, HEAD_DIM]
        sycl::buffer<std::uint16_t> qb(qfa.begin(), qfa.end());
        sycl::buffer<std::uint16_t> kb(kfa.begin(), kfa.end());
        sycl::buffer<std::uint16_t> vb(vfa.begin(), vfa.end());
        sycl::buffer<std::uint16_t> ob(sycl::range<1>((size_t)N_HEADS*HEAD_DIM));
        attention::FlashAttentionParams pa;
        pa.batch=1; pa.heads=N_HEADS; pa.seq_q=1; pa.seq_k=seq; pa.dim=HEAD_DIM;
        pa.scale=1.0f/sqrtf((float)HEAD_DIM); pa.causal=true; pa.q_offset=pos;
        attention::flash_attention_forward_f16(eng->device_queue(0), qb, kb, vb, ob, pa);
        auto h_o = ob.get_host_access(sycl::read_only);
        std::vector<float> attn(N_HEADS*HEAD_DIM);
        for(int i=0;i<N_HEADS*HEAD_DIM;i++) attn[i]=dashq::half_to_float(h_o[i]);
        for(int i=0;i<N_HEADS*HEAD_DIM;i++) attn[i]*=sigmoidf(gate[i]);
        const Rotation* Ro=eng->rotation_for(t_o[L]);
        std::vector<float> o_raw = eng->run_linear_tp_ret(t_o[L], attn); // N=H
        std::vector<float> h_attn(H);
        if(Ro) rot_left(o_raw.data(), *Ro, h_attn.data()); else h_attn=o_raw;
        for(int i=0;i<H;i++) h[i]=res[i]+h_attn[i];
      } else {
        // === SSM / linear-attention branch (host, safetensors) ===
        std::vector<float> mqkv = linear_host(get(p("linear_attn.in_proj_qkv.weight")).data(), normed.data(), CONV_DIM, H);
        std::vector<float> conv_out(CONV_DIM);
        const float* conv_w = get(p("linear_attn.conv1d.weight")).data();
        for(int c=0;c<CONV_DIM;c++){
          float s=0;
          s+=conv_w[c*CONV_K+0]*conv_state[L][c*3+0];
          s+=conv_w[c*CONV_K+1]*conv_state[L][c*3+1];
          s+=conv_w[c*CONV_K+2]*conv_state[L][c*3+2];
          s+=conv_w[c*CONV_K+3]*mqkv[c];
          conv_out[c]=silu(s);
          conv_state[L][c*3+0]=conv_state[L][c*3+1]; conv_state[L][c*3+1]=conv_state[L][c*3+2]; conv_state[L][c*3+2]=mqkv[c];
        }
        float* query=conv_out.data();
        float* key=conv_out.data()+K_HEAD_DIM*N_K_HEADS;
        float* value=conv_out.data()+2*K_HEAD_DIM*N_K_HEADS;
        std::vector<float> z = linear_host(get(p("linear_attn.in_proj_z.weight")).data(), normed.data(), N_V_HEADS*V_HEAD_DIM, H);
        std::vector<float> b = linear_host(get(p("linear_attn.in_proj_b.weight")).data(), normed.data(), N_V_HEADS, H);
        std::vector<float> a = linear_host(get(p("linear_attn.in_proj_a.weight")).data(), normed.data(), N_V_HEADS, H);
        const float* A_log=get(p("linear_attn.A_log")).data();
        const float* dt_bias=get(p("linear_attn.dt_bias")).data();
        std::vector<float> g(N_V_HEADS);
        for(int hh=0; hh<N_V_HEADS; hh++){ float sp=log1pf(expf(a[hh]+dt_bias[hh])); g[hh]=-expf(A_log[hh])*sp; }
        std::vector<float> core(N_V_HEADS*V_HEAD_DIM,0.f);
        float qn[K_HEAD_DIM], kn[K_HEAD_DIM];
        for(int hh=0; hh<N_V_HEADS; hh++){
          // GQA-style grouping (HF fix_query_key_value_ordering): value
          // head hh shares key head kh. No-op when N_V==N_K.
          int kh = (N_V_HEADS > N_K_HEADS) ? (hh * N_K_HEADS / N_V_HEADS) : hh;
          l2norm(&query[kh*K_HEAD_DIM], qn, K_HEAD_DIM);
          l2norm(&key[kh*K_HEAD_DIM],  kn, K_HEAD_DIM);
          float scale=1.0f/sqrtf((float)K_HEAD_DIM);
          for(int d=0;d<K_HEAD_DIM;d++) qn[d]*=scale;
          float g_t=expf(g[hh]); float beta=sigmoidf(b[hh]);
          float* S=&ssm_state[L][hh*K_HEAD_DIM*V_HEAD_DIM];
          for(int i=0;i<K_HEAD_DIM*V_HEAD_DIM;i++) S[i]*=g_t;
          std::vector<float> kv_mem(V_HEAD_DIM,0.f);
          for(int d=0;d<K_HEAD_DIM;d++) for(int c=0;c<V_HEAD_DIM;c++) kv_mem[c]+=S[d*V_HEAD_DIM+c]*kn[d];
          std::vector<float> delta(V_HEAD_DIM); const float* vh=&value[hh*V_HEAD_DIM];
          for(int c=0;c<V_HEAD_DIM;c++) delta[c]=(vh[c]-kv_mem[c])*beta;
          for(int d=0;d<K_HEAD_DIM;d++) for(int c=0;c<V_HEAD_DIM;c++) S[d*V_HEAD_DIM+c]+=kn[d]*delta[c];
          for(int c=0;c<V_HEAD_DIM;c++){ float s=0; for(int d=0;d<K_HEAD_DIM;d++) s+=S[d*V_HEAD_DIM+c]*qn[d]; core[hh*V_HEAD_DIM+c]=s; }
        }
        const float* ssm_norm_w=get(p("linear_attn.norm.weight")).data();
        std::vector<float> normed_core(N_V_HEADS*V_HEAD_DIM);
        for(int hh=0; hh<N_V_HEADS; hh++){
          const float* x=&core[hh*V_HEAD_DIM]; const float* zg=&z[hh*V_HEAD_DIM]; float* o=&normed_core[hh*V_HEAD_DIM];
          float v=0; for(int i=0;i<V_HEAD_DIM;i++) v+=x[i]*x[i];
          float inv=1.0f/sqrtf(v/V_HEAD_DIM+RMS_EPS);
          for(int i=0;i<V_HEAD_DIM;i++) o[i]=(x[i]*inv)*ssm_norm_w[i]*silu(zg[i]);
        }
        const float* ssm_out=get(p("linear_attn.out_proj.weight")).data();
        h = linear_host(ssm_out, normed_core.data(), H, N_V_HEADS*V_HEAD_DIM);
        for(int i=0;i<H;i++) h[i]+=res[i];
      }
      // MLP (quantized, SpinQuant replay) — TP GEMM across the fleet
      std::vector<float> rn2(H);
      rmsnorm(h.data(), get(p("post_attention_layernorm.weight")).data(), rn2.data(), H);
      const Rotation* Rg=eng->rotation_for(t_gate[L]);
      std::vector<float> rin(rn2); if(Rg) rot_apply(rn2.data(), *Rg, rin.data()); // right-site FFN_IN
      std::vector<float> mg = eng->run_linear_tp_ret(t_gate[L], rin); // N=INTER
      std::vector<float> mu = eng->run_linear_tp_ret(t_up[L],   rin); // N=INTER
      for(int i=0;i<INTER;i++) mg[i]=silu(mg[i])*mu[i];
      std::vector<float> md = eng->run_linear_tp_ret(t_down[L], mg); // N=H
        const Rotation* Rd=eng->rotation_for(t_down[L]);
        std::vector<float> mlp_out(H);
        if(Rd) rot_left(md.data(), *Rd, mlp_out.data()); else mlp_out=md;
        for(int i=0;i<H;i++) h[i]=h[i]+mlp_out[i];
    }
    // final norm + lm_head (tied embeddings) + greedy argmax
    std::vector<float> fn(H); rmsnorm(h.data(), final_norm, fn.data(), H);
    int best=-1; float bestv=-1e30f;
    for(int v=0; v<VOCAB; v++){ const float* er=embed+(size_t)v*H; float s=0; for(int i=0;i<H;i++) s+=fn[i]*er[i]; if(s>bestv){bestv=s; best=v;} }
    if(record) generated.push_back(best);
    return best;
  };

  for(int pos=0; pos<S; pos++){
    int nxt=step(prompt_ids[pos], pos, true);
    if(pos==S-1) printf("prefill last -> %d\n", nxt);
  }
  int token=generated.back();
  for(int si=1; si<maxlen; si++){
    int nxt=step(token, S+si-1, false);
    generated.push_back(nxt); token=nxt; if(nxt==0) break;
  }
  printf("generated:"); for(int x:generated) printf(" %d", x); printf("\n");
  return 0;
}
