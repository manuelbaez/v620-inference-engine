#ifndef QW_KERNEL_QSA_ATTN_CUH
#define QW_KERNEL_QSA_ATTN_CUH

#ifndef __AMDGCN__
#error "qsa_attn.cuh is a device header: it must be compiled for an AMDGCN target (HIP), never from host code"
#endif

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// ============================================================================
// QSA (full-attention) single-token decode kernel for Qwen3.8-Flash-Next,
// targeting gfx1030 (RDNA2): fp16 storage/compute, fp32 accumulation,
// warpSize 32. No bf16/fp8/WMMA/MFMA on this part.
//
// Shape (per QSA layer, per decode token):
//   query:  q[n_qh][head_dim]          (n_qh = 24)
//   keys:   K[n_tok][n_kv][head_dim]   (n_kv = 2, GQA 12:1)
//   values: V[n_tok][n_kv][head_dim]
//   out:    out[n_qh][head_dim]
//
// GQA mapping is DERIVED from the head counts, not hardcoded:
//   head_group  = (n_qh / n_kv) = 12
//   kv_head(qh) = qh / head_group
// so qh 0..11 -> KV head 0 and qh 12..23 -> KV head 1 for the default
// (24 q heads, 2 kv heads). Any GQA config with n_qh % n_kv == 0 works.
//
// The KV is stored exactly as include/qw/kvcache.h lays it out for one
// layer of one token (layer-major within the token block; the harness
// builds a flat buffer matching this layout):
//
//   token t, kv head k:
//     K: kbuf[t][k][d]  at byte offset t * (2 * n_kv * head_dim * 2) + k * head_dim * 2
//     V: vbuf[t][k][d]  at K offset + n_kv * head_dim * 2 (V follows K in the slot)
//
//   bytes_per_token_per_layer = 2 (K,V) * n_kv * head_dim * 2 (fp16)
//
// EXPECTED BYTES READ PER TOKEN PER LAYER (this kernel):
//   K + V:  2 * n_kv * head_dim * 2 = 2 * 2 * 256 * 2 = 2048 bytes.
//   That is exactly the figure in include/qw/kvcache.h
//   (bytes_per_token_per_layer = 2048) and, across the 12 QSA layers,
//   12 * 2048 = 24576 bytes per token — the same total the header asserts
//   and that qw_model_estimate() (config.h: kv_per_token) reports. The two
//   sources AGREE: 2048 B/token/layer is the correct per-layer number, and
//   24576 B/token is the correct per-model total. The 2048 B figure IS the
//   whole per-token KV footprint of one layer (nothing else in the slot);
//   the kernel reads it once per layer.
// ============================================================================

namespace qsa {

constexpr int kD = 256;            // head_dim (fixed for this model)
constexpr int kWarp = 32;          // warpSize on gfx1030
constexpr int kNPerLane = 8;       // fp16x4 per lane (4 * 8 = 32 = D / warp)
constexpr int kChunk = 64;         // tokens per split-K chunk (256 B each)

// Partial result per (query head, split) for the combine step:
// running max m, running sum-of-exponentials l, weighted sum acc.
struct alignas(16) Partial {
  float m;
  float l;
  float acc[kD];
};

// ----------------------------------------------------------------------------
// fp16x4 packed load (64-bit). Requires 8-byte alignment of p.
// ----------------------------------------------------------------------------
static __device__ __forceinline__ uint2 qw_load4_f16(const uint16_t *p) {
  uint2 v;
  const uint64_t *q = reinterpret_cast<const uint64_t *>(p);
  v.x = (uint32_t)(*q & 0xffffffffu);
  v.y = (uint32_t)(*q >> 32);
  return v;
}

static __device__ __forceinline__ void qw_f16x4_to_f32(uint2 v, float out[4]) {
  const __half *hx = reinterpret_cast<const __half *>(&v.x);
  const __half *hy = reinterpret_cast<const __half *>(&v.y);
  out[0] = __half2float(hx[0]);
  out[1] = __half2float(hx[1]);
  out[2] = __half2float(hy[0]);
  out[3] = __half2float(hy[1]);
}

// Dot product of two head_dim=256 fp16 vectors loaded as fp16x4 per lane.
// Each lane holds 8 fp16 (4 from K slot + 4 from V slot across 2 packs);
// we accumulate in fp32.
// Each lane owns 8 consecutive fp16 (dims lane*8 .. lane*8+7), loaded as two
// consecutive fp16x4 packs at q+lane*8 and q+lane*8+4. 32 lanes * 8 = 256.
static __device__ __forceinline__ float qw_qk_dot256(const __half *q, const __half *k, int lane) {
  float acc = 0.f;
  #pragma unroll
  for (int half = 0; half < 2; half++) {
    int base = lane * 8 + half * 4;
    uint2 qv = qw_load4_f16(reinterpret_cast<const uint16_t *>(q + base));
    uint2 kv = qw_load4_f16(reinterpret_cast<const uint16_t *>(k + base));
    float qa[4], ka[4];
    qw_f16x4_to_f32(qv, qa);
    qw_f16x4_to_f32(kv, ka);
    #pragma unroll
    for (int i = 0; i < 4; i++)
      acc += qa[i] * ka[i];
  }
  return acc;
}

// ----------------------------------------------------------------------------
// Core: one workgroup per query head, optionally split over KV length.
//
// ONLINE / STREAMING SOFTMAX (flash-attention style, no second pass):
//   We cannot materialize all L scores before normalizing (L can be 262144),
//   so we maintain a running max m and running sum l as we stream tokens.
//
//   Initialize: m = -inf, l = 0, acc[d] = 0.
//
//   For each token t (score s_t = dot(q, k_t) / sqrt(D)):
//     m_new = max(m, s_t)
//     // rescale the existing accumulation by exp(m - m_new) to keep it
//     // expressed relative to the NEW max (prevents underflow/overflow).
//     alpha = expf(m - m_new)
//     l     = l * alpha + expf(s_t - m_new)
//     acc[d] = acc[d] * alpha + expf(s_t - m_new) * v_t[d]
//     m     = m_new
//
//   Because every term is always relative to the current global max, all
//   exponents are <= 0 (exp <= 1): no overflow, and the sum l stays bounded
//   by L. The final output is out[d] = acc[d] / l.
//
//   Numerical stability: subtracting the running max inside exp() is the
//   standard trick; the rescale-by-alpha step keeps the partial sums
//   equivalent to a single-pass softmax over all tokens seen so far.
// ----------------------------------------------------------------------------
template <int SPLIT>
__global__ void qsa_attn_kernel(
    const __half * __restrict__ q,       // [n_qh][head_dim]
    const __half * __restrict__ k,       // [n_tok][n_kv][head_dim]
    const __half * __restrict__ v,       // [n_tok][n_kv][head_dim]
    float * __restrict__ out,            // [n_qh][head_dim]
    Partial * __restrict__ partials,     // [n_qh][n_split] (when SPLIT)
    int n_tok, int n_qh, int n_kv, int head_dim, int head_group)
{
  const int qh = blockIdx.x;
  if (qh >= n_qh) return;
  const int n_split = SPLIT ? (n_tok + kChunk - 1) / kChunk : 1;
  const int split = SPLIT ? blockIdx.y : 0;
  if (SPLIT && split >= n_split) return;

  const int kv = qh / head_group;   // derived GQA mapping
  const int t0 = split * kChunk;
  const int t1 = min(t0 + kChunk, n_tok);
  const int lane = threadIdx.x;     // 0..31 (one warp per workgroup)

  // q for this head: base offset
  const __half *qbase = q + (size_t)qh * head_dim;

  float m = -INFINITY;
  float l = 0.f;
  float acc[kD];
  #pragma unroll
  for (int d = 0; d < kD; d++) acc[d] = 0.f;

  const float inv_sqrt_d = rsqrtf((float)head_dim);

  for (int t = t0; t < t1; t++) {
    // K row: [n_tok][n_kv][head_dim]
    const __half *krow = k + ((size_t)t * n_kv + kv) * head_dim;
    const __half *vrow = v + ((size_t)t * n_kv + kv) * head_dim;

    // --- QK dot (256 fp16) across the warp, reduce ---
    float s = qw_qk_dot256(qbase, krow, lane) * inv_sqrt_d;
    // warp reduce
    #pragma unroll
    for (int off = kWarp / 2; off > 0; off /= 2)
      s += __shfl_down_sync(0xffffffffull, s, off);
    s = __shfl_sync(0xffffffffull, s, 0);  // broadcast to all lanes

    // --- online softmax update ---
    float m_new = fmaxf(m, s);
    float alpha = (m == -INFINITY) ? 0.f : expf(m - m_new);
    float e = expf(s - m_new);
    l = l * alpha + e;
    m = m_new;

    // --- acc update: acc *= alpha; acc += e * v_row ---
    // each lane handles 8 consecutive fp16 of v (dims lane*8 .. lane*8+7)
    #pragma unroll
    for (int half = 0; half < 2; half++) {
      int base = lane * 8 + half * 4;
      uint2 vv = qw_load4_f16(reinterpret_cast<const uint16_t *>(vrow + base));
      float vf[4];
      qw_f16x4_to_f32(vv, vf);
      #pragma unroll
      for (int i = 0; i < 4; i++)
        acc[base + i] = acc[base + i] * alpha + e * vf[i];
    }
  }

  // --- write result ---
  if (SPLIT) {
    // write partial triple
    Partial *p = &partials[qh * n_split + split];
    p->m = m;
    p->l = l;
    // each lane writes its 8 floats
    #pragma unroll
    for (int half = 0; half < 2; half++) {
      int base = lane * 8 + half * 4;
      #pragma unroll
      for (int i = 0; i < 4; i++)
        p->acc[base + i] = acc[base + i];
    }
  } else {
    // single pass: normalize in-register
    float inv_l = (l > 0.f) ? 1.f / l : 0.f;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
      int base = lane * 8 + half * 4;
      #pragma unroll
      for (int i = 0; i < 4; i++)
        out[qh * head_dim + base + i] = acc[base + i] * inv_l;
    }
  }
}

// ----------------------------------------------------------------------------
// Combine kernel: fold n_split partial (m, l, acc) triples per query head
// into the final output using the same online-softmax recurrence.
//
//   For partial i with (m_i, l_i, acc_i):
//     M_new = max(M, m_i)
//     alpha = exp(M - M_new)            // rescale running state
//     l     = l * alpha + l_i * exp(m_i - M_new)
//     acc[d] = acc[d] * alpha + acc_i[d] * exp(m_i - M_new)
//     M     = M_new
//
//   This is exactly the pairwise merge of two online-softmax states, so the
//   result equals a single-pass softmax over all tokens.
// ----------------------------------------------------------------------------
__global__ void qsa_attn_combine(
    const Partial * __restrict__ partials,  // [n_qh][n_split]
    float * __restrict__ out,               // [n_qh][head_dim]
    int n_qh, int n_split, int head_dim)
{
  const int qh = blockIdx.x;
  if (qh >= n_qh) return;
  const int lane = threadIdx.x;

  float M = -INFINITY;
  float l = 0.f;
  float acc[kD];
  #pragma unroll
  for (int d = 0; d < kD; d++) acc[d] = 0.f;

  for (int s = 0; s < n_split; s++) {
    const Partial *p = &partials[qh * n_split + s];
    float mi = p->m;
    float li = p->l;
    float M_new = fmaxf(M, mi);
    float alpha = (M == -INFINITY) ? 0.f : expf(M - M_new);
    float scale = (mi == -INFINITY) ? 0.f : expf(mi - M_new);
    l = l * alpha + li * scale;
    M = M_new;
    // each lane merges its 8 floats
    #pragma unroll
    for (int half = 0; half < 2; half++) {
      int base = lane * 8 + half * 4;
      #pragma unroll
      for (int i = 0; i < 4; i++)
        acc[base + i] = acc[base + i] * alpha + p->acc[base + i] * scale;
    }
  }

  float inv_l = (l > 0.f) ? 1.f / l : 0.f;
  #pragma unroll
  for (int half = 0; half < 2; half++) {
    int base = lane * 8 + half * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
      out[qh * head_dim + base + i] = acc[base + i] * inv_l;
  }
}

// ----------------------------------------------------------------------------
// Launch wrapper: runs the attention kernel and, when split-K is active,
// follows with the combine kernel. Returns hipSuccess or the first error.
// split: 0 = single pass over the whole sequence; 1 = split-K over kChunk
// token pieces plus combine.
// ----------------------------------------------------------------------------
inline hipError_t qsa_attn_launch(
    const __half *q, const __half *k, const __half *v, float *out,
    int n_tok, int n_qh, int n_kv, int head_dim, int split)
{
  const int head_group = n_qh / n_kv;   // derived GQA group size
  const int n_split = (n_tok + kChunk - 1) / kChunk;
  Partial *d_part = nullptr;

  if (split && n_split > 1) {
    size_t b = (size_t)n_qh * n_split * sizeof(Partial);
    if (hipMalloc((void **)&d_part, b) != hipSuccess)
      return hipGetLastError();
  }

  dim3 grid(n_qh, split ? n_split : 1);
  if (split && n_split > 1) {
    qsa_attn_kernel<1><<<grid, kWarp>>>(q, k, v, out, d_part,
                                        n_tok, n_qh, n_kv, head_dim, head_group);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) { (void)hipFree(d_part); return e; }
    qsa_attn_combine<<<dim3(n_qh, 1, 1), kWarp>>>(d_part, out, n_qh, n_split, head_dim);
    hipError_t ec = hipGetLastError();
    (void)hipFree(d_part);
    return ec;
  }
  qsa_attn_kernel<0><<<grid, kWarp>>>(q, k, v, out, nullptr,
                                      n_tok, n_qh, n_kv, head_dim, head_group);
  return hipGetLastError();
}

} // namespace qsa

#endif
