#include "intfer/attention/flash_attention.hpp"

#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace intfer::attention {
namespace {

// Tile sizes tuned for Intel Arc (Xe-core). Br query rows / work-group;
// Bc key rows / block. At head_dim<=128 this uses 2 * Bc*dim*2 bytes = 32 KB
// of SLM (well within a Xe-core's local memory budget).
constexpr int kBr = 64;   // query rows per work-group
constexpr int kBc = 64;   // key rows per key block
constexpr int kMaxDim = 256;  // hard cap on head dimension

inline float lf(sycl::half h) { return static_cast<float>(h); }
inline sycl::half hf(float f) { return sycl::half(f); }

// Offset into a [batch, heads, seq, dim] row-major tensor.
inline int off3(int b, int h, int t, int H, int S, int D) {
  return ((b * H + h) * S + t) * D;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) FP16 fused FlashAttention-2 forward (online softmax, query-tiled).
// ---------------------------------------------------------------------------
void flash_attention_forward_f16(sycl::queue& q,
                                 sycl::buffer<std::uint16_t>& q_buf,
                                 sycl::buffer<std::uint16_t>& k_buf,
                                 sycl::buffer<std::uint16_t>& v_buf,
                                 sycl::buffer<std::uint16_t>& out_buf,
                                 const FlashAttentionParams& p) {
  const int B = p.batch, H = p.heads, Sq = p.seq_q, Sk = p.seq_k, D = p.dim;
  const float scale = p.scale > 0.0f ? p.scale : (1.0f / std::sqrt(static_cast<float>(D)));
  const bool causal = p.causal;
  const int num_q_blocks = (Sq + kBr - 1) / kBr;
  const int num_k_blocks = (Sk + kBc - 1) / kBc;

  q.submit([&](sycl::handler& h) {
    auto qa = q_buf.get_access<sycl::access::mode::read>(h);
    auto ka = k_buf.get_access<sycl::access::mode::read>(h);
    auto va = v_buf.get_access<sycl::access::mode::read>(h);
    auto oa = out_buf.get_access<sycl::access::mode::write>(h);
    sycl::local_accessor<std::uint16_t, 1> sK(sycl::range<1>(kBc * D), h);
    sycl::local_accessor<std::uint16_t, 1> sV(sycl::range<1>(kBc * D), h);

    sycl::nd_range<1> nd(sycl::range<1>(B * H * num_q_blocks * kBr),
                         sycl::range<1>(kBr));

    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int tid = it.get_local_id(0);        // 0..kBr-1 (query row in block)
      const int block = it.get_global_id(0) / kBr;
      const int qb = block % num_q_blocks;
      const int bh = block / num_q_blocks;
      const int h_idx = bh % H;
      const int b_idx = bh / H;
      const int qi = qb * kBr + tid;             // global query token index (within Q tensor)
      const int qi_abs = qi + p.q_offset;        // absolute position in the KV cache
      const bool active = qi < Sq;

      const int base = off3(b_idx, h_idx, 0, H, Sq, D);
      float Q[kMaxDim];
      if (active)
        for (int d = 0; d < D; ++d)
          Q[d] = lf(sycl::bit_cast<sycl::half>(qa[base + qi * D + d]));

      float O[kMaxDim] = {0.0f};
      float m = -1e30f;   // running max of softmax
      float l = 0.0f;     // running sum of exp

      for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_start = kb * kBc;

        // Cooperative load K block into SLM (all threads participate).
        if (tid < kBc) {
          const int tk = k_start + tid;
          for (int d = 0; d < D; ++d) {
            const std::uint16_t val =
                (tk < Sk) ? ka[off3(b_idx, h_idx, tk, H, Sk, D) + d] : 0;
            sK[tid * D + d] = val;
          }
        }
        it.barrier();

        // Score this query row against the loaded K block.
        float pj[kBc];
        float m_block = -1e30f;
        if (active) {
          for (int j = 0; j < kBc; ++j) {
            const int tk = k_start + j;
            if (tk >= Sk || (causal && tk > qi_abs)) {
              pj[j] = -1e30f;
              continue;
            }
            float s = 0.0f;
            for (int d = 0; d < D; ++d)
              s += Q[d] * lf(sycl::bit_cast<sycl::half>(sK[j * D + d]));
            pj[j] = s * scale;
            if (pj[j] > m_block) m_block = pj[j];
          }
        }

        // Rescale factor for the running state into the new max.
        const float m_new = sycl::fmax(m, m_block);
        const float alpha = sycl::exp(m - m_new);
        const float beta = sycl::exp(m_block - m_new);

        // Cooperative load V block into SLM.
        if (tid < kBc) {
          const int tk = k_start + tid;
          for (int d = 0; d < D; ++d) {
            const std::uint16_t val =
                (tk < Sk) ? va[off3(b_idx, h_idx, tk, H, Sk, D) + d] : 0;
            sV[tid * D + d] = val;
          }
        }
        it.barrier();

        if (active) {
          for (int d = 0; d < D; ++d) O[d] *= alpha;
          float l_block = 0.0f;
          for (int j = 0; j < kBc; ++j) {
            const int tk = k_start + j;
            if (tk >= Sk || (causal && tk > qi_abs)) continue;
            const float w = sycl::exp(pj[j] - m_block);
            l_block += w;
            for (int d = 0; d < D; ++d)
              O[d] += beta * w * lf(sycl::bit_cast<sycl::half>(sV[j * D + d]));
          }
          l = alpha * l + beta * l_block;
          m = m_new;
        }
        it.barrier();
      }

      if (active) {
        const int o = off3(b_idx, h_idx, qi, H, Sq, D);
        for (int d = 0; d < D; ++d)
          oa[o + d] = sycl::bit_cast<std::uint16_t>(hf(O[d] / l));
      }
    });
  });
  q.wait();
}

// ---------------------------------------------------------------------------
// (2) Groupwise-INT8 KV-cache fused attention forward (dequant-on-the-fly).
// K/V are INT8; scales are FP32 [batch, heads, num_groups] with
// num_groups = ceil(seq_k / group_size); token t uses group t / group_size.
// ---------------------------------------------------------------------------
void flash_attention_forward_int8_kv(sycl::queue& q,
                                     sycl::buffer<std::uint16_t>& q_buf,
                                     sycl::buffer<std::int8_t>& k_buf,
                                     sycl::buffer<std::int8_t>& v_buf,
                                     sycl::buffer<float>& k_scale_buf,
                                     sycl::buffer<float>& v_scale_buf,
                                     sycl::buffer<std::uint16_t>& out_buf,
                                     const FlashAttentionParams& p,
                                     int group_size) {
  const int B = p.batch, H = p.heads, Sq = p.seq_q, Sk = p.seq_k, D = p.dim;
  const int num_groups = (Sk + group_size - 1) / group_size;
  const float scale = p.scale > 0.0f ? p.scale : (1.0f / std::sqrt(static_cast<float>(D)));
  const bool causal = p.causal;
  const int num_q_blocks = (Sq + kBr - 1) / kBr;
  const int num_k_blocks = (Sk + kBc - 1) / kBc;

  q.submit([&](sycl::handler& h) {
    auto qa = q_buf.get_access<sycl::access::mode::read>(h);
    auto ka = k_buf.get_access<sycl::access::mode::read>(h);
    auto va = v_buf.get_access<sycl::access::mode::read>(h);
    auto ksa = k_scale_buf.get_access<sycl::access::mode::read>(h);
    auto vsa = v_scale_buf.get_access<sycl::access::mode::read>(h);
    auto oa = out_buf.get_access<sycl::access::mode::write>(h);
    sycl::local_accessor<std::uint16_t, 1> sK(sycl::range<1>(kBc * D), h);
    sycl::local_accessor<std::uint16_t, 1> sV(sycl::range<1>(kBc * D), h);

    sycl::nd_range<1> nd(sycl::range<1>(B * H * num_q_blocks * kBr),
                         sycl::range<1>(kBr));

    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int tid = it.get_local_id(0);
      const int block = it.get_global_id(0) / kBr;
      const int qb = block % num_q_blocks;
      const int bh = block / num_q_blocks;
      const int h_idx = bh % H;
      const int b_idx = bh / H;
      const int qi = qb * kBr + tid;
      const int qi_abs = qi + p.q_offset;
      const bool active = qi < Sq;

      const int base = off3(b_idx, h_idx, 0, H, Sq, D);
      const int s_base = (b_idx * H + h_idx) * num_groups;
      float Q[kMaxDim];
      if (active)
        for (int d = 0; d < D; ++d)
          Q[d] = lf(sycl::bit_cast<sycl::half>(qa[base + qi * D + d]));

      float O[kMaxDim] = {0.0f};
      float m = -1e30f;
      float l = 0.0f;

      for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_start = kb * kBc;

        if (tid < kBc) {
          const int tk = k_start + tid;
          const float ks = (tk < Sk) ? ksa[s_base + tk / group_size] : 0.0f;
          const float vs = (tk < Sk) ? vsa[s_base + tk / group_size] : 0.0f;
          for (int d = 0; d < D; ++d) {
            const std::int8_t kv =
                (tk < Sk) ? ka[off3(b_idx, h_idx, tk, H, Sk, D) + d] : 0;
            const std::int8_t vv =
                (tk < Sk) ? va[off3(b_idx, h_idx, tk, H, Sk, D) + d] : 0;
            sK[tid * D + d] = sycl::bit_cast<std::uint16_t>(hf(ks * kv));
            sV[tid * D + d] = sycl::bit_cast<std::uint16_t>(hf(vs * vv));
          }
        }
        it.barrier();

        float pj[kBc];
        float m_block = -1e30f;
        if (active) {
          for (int j = 0; j < kBc; ++j) {
            const int tk = k_start + j;
            if (tk >= Sk || (causal && tk > qi_abs)) {
              pj[j] = -1e30f;
              continue;
            }
            float s = 0.0f;
            for (int d = 0; d < D; ++d)
              s += Q[d] * lf(sycl::bit_cast<sycl::half>(sK[j * D + d]));
            pj[j] = s * scale;
            if (pj[j] > m_block) m_block = pj[j];
          }
        }

        const float m_new = sycl::fmax(m, m_block);
        const float alpha = sycl::exp(m - m_new);
        const float beta = sycl::exp(m_block - m_new);
        it.barrier();

        if (active) {
          for (int d = 0; d < D; ++d) O[d] *= alpha;
          float l_block = 0.0f;
          for (int j = 0; j < kBc; ++j) {
            const int tk = k_start + j;
            if (tk >= Sk || (causal && tk > qi_abs)) continue;
            const float w = sycl::exp(pj[j] - m_block);
            l_block += w;
            for (int d = 0; d < D; ++d)
              O[d] += beta * w * lf(sycl::bit_cast<sycl::half>(sV[j * D + d]));
          }
          l = alpha * l + beta * l_block;
          m = m_new;
        }
        it.barrier();
      }

      if (active) {
        const int o = off3(b_idx, h_idx, qi, H, Sq, D);
        for (int d = 0; d < D; ++d)
          oa[o + d] = sycl::bit_cast<std::uint16_t>(hf(O[d] / l));
      }
    });
  });
  q.wait();
}

// ---------------------------------------------------------------------------
// Host FP32 reference for numerical validation.
// ---------------------------------------------------------------------------
void flash_attention_reference_f32(const std::vector<float>& q,
                                   const std::vector<float>& k,
                                   const std::vector<float>& v,
                                   std::vector<float>& out,
                                   const FlashAttentionParams& p) {
  const int B = p.batch, H = p.heads, Sq = p.seq_q, Sk = p.seq_k, D = p.dim;
  const float scale =
      p.scale > 0.0f ? p.scale : (1.0f / std::sqrt(static_cast<float>(D)));
  out.assign(B * H * Sq * D, 0.0f);

  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int qi = 0; qi < Sq; ++qi) {
        const int qo = off3(b, h, qi, H, Sq, D);
        float mx = -1e30f;
        std::vector<float> s(Sk);
        for (int tk = 0; tk < Sk; ++tk) {
          if (p.causal && tk > qi) {
            s[tk] = -1e30f;
            continue;
          }
          float acc = 0.0f;
          const int ko = off3(b, h, tk, H, Sk, D);
          for (int d = 0; d < D; ++d) acc += q[qo + d] * k[ko + d];
          acc *= scale;
          s[tk] = acc;
          if (acc > mx) mx = acc;
        }
        float sum = 0.0f;
        for (int tk = 0; tk < Sk; ++tk) {
          if (s[tk] <= -1e29f) continue;
          s[tk] = std::exp(s[tk] - mx);
          sum += s[tk];
        }
        const float inv = 1.0f / sum;
        for (int tk = 0; tk < Sk; ++tk) {
          if (s[tk] <= 0.0f) continue;
          const float w = s[tk] * inv;
          const int ko = off3(b, h, tk, H, Sk, D);
          for (int d = 0; d < D; ++d) out[qo + d] += w * v[ko + d];
        }
      }
}

}  // namespace intfer::attention
