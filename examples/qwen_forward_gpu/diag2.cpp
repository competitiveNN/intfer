// examples/qwen_forward_gpu/diag2.cpp
// Replicates Engine::shard_weight + quant_gemm_forward for one tensor and prints
// the actual buffers handed to the kernel, to localize the 0-output bug.
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include "intfer/engine.hpp"
#include "intfer/output.hpp"
#include "intfer/quant/dashq.hpp"
#include "intfer/safetensors.hpp"
#include "intfer/gemm/quant_gemm.hpp"
using namespace intfer;
using namespace intfer::gemm;

static double cosine(const std::vector<float>& a, const std::vector<float>& b){
  double d=0,na=0,nb=0; size_t n=std::min(a.size(),b.size());
  for(size_t i=0;i<n;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  return d/(sqrt(na)*sqrt(nb)+1e-12);
}

// copy of Engine::shard_weight (anonymous namespace in engine.cpp)
static gemm::WeightScheme to_ws(QuantScheme s){
  switch(s){case QuantScheme::INT2:return gemm::WeightScheme::INT2;case QuantScheme::INT4:return gemm::WeightScheme::INT4;
    case QuantScheme::INT8:return gemm::WeightScheme::INT8;default:return gemm::WeightScheme::FP16;}
}
static gemm::GemmWeightBuffers shard_weight(const ModelIR& m, const TensorIR& t, uint32_t start, uint32_t end, sycl::queue& q){
  gemm::GemmWeightBuffers w;
  w.scheme=to_ws(t.qcfg.scheme); w.group_size=t.qcfg.group_size; w.has_global_divisor=t.qcfg.has_global_divisor;
  w.N=end-start; w.K=t.K;
  const uint32_t N=t.N,K=t.K;
  const uint32_t G=t.qcfg.group_size>0?t.qcfg.group_size:K;
  const uint32_t n_groups=(K+G-1)/G;
  const int bits=(w.scheme==WeightScheme::INT2)?2:(w.scheme==WeightScheme::INT4)?4:(w.scheme==WeightScheme::FP16)?16:8;
  const int pack=(bits>=8)?1:(8/bits);
  const uint32_t per_row_code_bytes=(K+pack-1)/pack;
  const uint16_t* scale_src=reinterpret_cast<const uint16_t*>(m.scale_buf.data()+t.scale_offset);
  std::vector<uint16_t> scale_broadcast((size_t)N*n_groups);
  for(uint32_t r=0;r<N;r++) for(uint32_t g=0;g<n_groups;g++) scale_broadcast[(size_t)r*n_groups+g]=scale_src[g];
  std::vector<uint16_t> scale_slice((size_t)w.N*n_groups);
  for(uint32_t r=0;r<w.N;r++) std::memcpy(&scale_slice[(size_t)r*n_groups],&scale_broadcast[(size_t)(r+start)*n_groups],n_groups*sizeof(uint16_t));
  w.scale_store=std::move(scale_slice);
  w.scale=sycl::buffer<uint16_t>(w.scale_store.begin(),w.scale_store.end());
  if(w.scheme==WeightScheme::FP16){
    std::vector<uint8_t> code_slice((size_t)w.N*K*2);
    const uint8_t* base=m.code_buf.data()+t.code_offset+start*K*2;
    std::memcpy(code_slice.data(),base,code_slice.size());
    w.code_store=std::move(code_slice);
    w.code=sycl::buffer<uint8_t>(w.code_store.data(),sycl::range<1>(w.code_store.size()));
  } else {
    std::vector<uint8_t> row_codes((size_t)w.N*K);
    for(uint32_t g=0;g<n_groups;g++){
      uint32_t k0=g*G, k1=std::min(k0+G,K); uint32_t Glocal=k1-k0;
      for(uint32_t r=0;r<w.N;r++){
        uint32_t src_row=r+start;
        for(uint32_t c=k0;c<k1;c++){
          size_t src_idx=(size_t)g*(N*Glocal)+(size_t)src_row*Glocal+(c-k0);
          const uint8_t b=m.code_buf[t.code_offset+src_idx/pack];
          int within=(int)(src_idx%pack);
          uint8_t val=(b>>(within*bits))&((1u<<bits)-1u);
          row_codes[(size_t)r*K+c]=val;
        }
      }
    }
    std::vector<uint8_t> code_slice((size_t)w.N*per_row_code_bytes,0);
    for(size_t i=0;i<row_codes.size();i++){
      size_t pidx=i/pack; int within=(int)(i%pack);
      code_slice[pidx]|=(uint8_t)((row_codes[i]&((1u<<bits)-1u))<<(within*bits));
    }
    w.code_store=std::move(code_slice);
    w.code=sycl::buffer<uint8_t>(w.code_store.begin(),w.code_store.end());
  }
  if(t.global_bytes>0){ float av=0; std::memcpy(&av,m.global_buf.data()+t.global_offset,std::min<uint32_t>(t.global_bytes,4u)); w.global_store.assign(1,av); w.global=sycl::buffer<float>(w.global_store.begin(),w.global_store.end()); }
  return w;
}
static void dump_buf(gemm::GemmWeightBuffers& w){
  auto wc=w.code.get_host_access(sycl::read_only);
  auto ws=w.scale.get_host_access(sycl::read_only);
  printf("  BUF code.range=%zu scale.range=%zu\n",(size_t)w.code.get_range()[0],(size_t)w.scale.get_range()[0]);
  for(int i=0;i<8;i++) printf("  BUF code[%d]=%u scale[%d]=%.4f\n",i,wc[i],i,dashq::half_to_float(ws[i]));
}

int main(int argc,char**argv){
  const char* ip=(argc>1)?argv[1]:"test/models/Qwen3.5-0.8B.intfer";
  int ti=(argc>2)?atoi(argv[2]):30;
  auto eng=Engine::load(ip,0); if(!eng){printf("load fail\n");return 1;}
  const ModelIR& m=eng->model();
  const auto& t=m.tensors[ti];
  printf("tensor %d name=%s N=%u K=%u scheme=%d grp=%u gdiv=%d gbytes=%u goff=%u\n",ti,t.name.c_str(),t.N,t.K,(int)t.qcfg.scheme,t.qcfg.group_size,t.qcfg.has_global_divisor,t.global_bytes,t.global_offset);
  sycl::queue q=eng->device_queue(0);
  auto w=shard_weight(m,t,0,t.N,q);
  printf("w.N=%u w.K=%u w.scheme=%d w.grp=%u w.gdiv=%d  code_store=%zu scale_store=%zu\n",w.N,w.K,(int)w.scheme,w.group_size,w.has_global_divisor,w.code_store.size(),w.scale_store.size());
  if(t.global_bytes>0){ float av=0; std::memcpy(&av,m.global_buf.data()+t.global_offset,std::min<uint32_t>(t.global_bytes,4u)); printf("  global divisor (alpha) = %.6f\n",av); }
  dump_buf(w);
  { auto wc3=w.code.get_host_access(sycl::read_only); auto ws3=w.scale.get_host_access(sycl::read_only);
    const int Gw=(int)t.qcfg.group_size; const int ng_w=(int)((t.K+Gw-1)/Gw); float c0=0;
    for(uint32_t k=0;k<t.K;k++){int code=(int)wc3[k];int g=(int)(k/Gw);float s=dashq::half_to_float(ws3[(size_t)0*ng_w+g]);c0+=0.1f*s*((2.0f*(float)code-255.0f)/255.0f);}
    printf("PRE-kernel CPU-row0-from-buffer=%.4f  wc3[0..3]=%d %d %d %d\n", c0, wc3[0],wc3[1],wc3[2],wc3[3]); }
  for(int i=0;i<8;i++) printf("  code[%d]=%u scale[%d]=%.4f\n",i,w.code_store[i],i,dashq::half_to_float(w.scale_store[i]));
  // synthetic FP16 activation all 0.1
  std::vector<uint16_t> act_fp16(t.K); for(uint32_t k=0;k<t.K;k++) act_fp16[k]=dashq::float_to_half(0.1f);
  gemm::GemmActBuffers a; a.scheme=gemm::ActScheme::FP16; a.M=1; a.K=t.K; a.fp16=sycl::buffer<uint16_t>(act_fp16.begin(),act_fp16.end());
  sycl::buffer<uint16_t> c(sycl::range<1>((size_t)1*t.N));
  gemm::QuantGemmOptions opt; opt.use_dpas=false;
  quant_gemm_forward(q,a,w,c,opt);
  auto hc=c.get_host_access(sycl::read_only);
  std::vector<float> out(t.N); for(uint32_t n=0;n<t.N;n++) out[n]=dashq::half_to_float(hc[n]);
  // CPU dot of row 0 using the ACTUAL w.code read back from the buffer
  auto wc2=w.code.get_host_access(sycl::read_only);
  auto ws2=w.scale.get_host_access(sycl::read_only);
  const int Gw=(int)t.qcfg.group_size; const int ng_w=(int)((t.K+Gw-1)/Gw);
  float cpu0=0;
  for(uint32_t k=0;k<t.K;k++){
    int code=(int)wc2[(size_t)0*t.K+k];
    int g=(int)(k/Gw);
    float s=dashq::half_to_float(ws2[(size_t)0*ng_w+g]);
    float wv=s*((2.0f*(float)code-255.0f)/255.0f);
    cpu0+=0.1f*wv;
  }
  printf("CPU-row0-from-buffer=%.4f  GPU-out[0]=%.4f\n", cpu0, out[0]);
  double na=0; for(float v:out) na+=v*v; printf("|out|=%.4f  out[0..3]=%.4f %.4f %.4f %.4f\n",sqrt(na),out[0],out[1],out[2],out[3]);
  // CPU ref via dequant_tensor
  std::vector<float> W; dashq::dequant_tensor(m,t,W);
  std::vector<float> ref(t.N,0.f); for(uint32_t n=0;n<t.N;n++) for(uint32_t k=0;k<t.K;k++) ref[n]+=0.1f*W[(size_t)n*t.K+k];
  printf("cos(out,ref)=%.6f  |ref|=%.4f\n",cosine(out,ref),sqrt([&]{double s=0;for(float v:ref)s+=v*v;return s;}()));
  return 0;
}
