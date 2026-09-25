#include "kernels/model_ops.hpp"

#include "core/config.hpp"
#include "kernels/common.cuh"
#include "kernels/model_common.cuh"

namespace qw::gpu {

using namespace cfg;

// ------------------------------------------------------------------ HC
__global__ __launch_bounds__(256) void hc_pre_kernel(float *X, const float *pend, const float *inj,
                                                     const float *w1, __half *y, float *send) {
    __shared__ float red[8];
    for (int s = 0; s < HC; ++s) {
        const float wgt = pend ? 2.f * sigmoid(inj[s] / HC) : 0.f;
        float ss = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            const int i = s * SH + j;
            float x = X[i];
            if (pend) {
                x += pend[j] * wgt;
                X[i] = x;
            }
            y[i] = __float2half(x * w1[i]);
            ss += x * x;
        }
        ss = block_sum<8>(ss, red);
        if (threadIdx.x == 0) send[s] = ss;
    }
}

void hc_pre(float *X, const float *pend, const float *inj, const float *w1, uint16_t *y,
            float *send, hipStream_t s) {
    hc_pre_kernel<<<1, 256, 0, s>>>(X, pend, inj, w1, reinterpret_cast<__half *>(y), send);
}

// grid SH/8 blocks, one wave per residual column j (4 rows of `up`).
__global__ __launch_bounds__(256) void hc_up_mix_kernel(const float *red, const uint4 *up,
                                                        const __half *y, __half *block_in,
                                                        float *inj_out) {
    __shared__ __align__(16) __half u[HC_RANK];
    __shared__ float r[HC];
    const int t = threadIdx.x;
    if (t < HC) r[t] = rsqrtf(red[t] / H + EPS);
    __syncthreads();
    for (int o = t; o < HC_RANK; o += blockDim.x) {
        float d = 0.f;
#pragma unroll
        for (int s = 0; s < HC; ++s) d += r[s] * red[HC + s * HC_DOWN + o];
        u[o] = __float2half(silu(d / HC));
    }
    if (inj_out && blockIdx.x == 0 && t < HC) {
        float v = 0.f;
        for (int s = 0; s < HC; ++s) v += r[s] * red[HC + s * HC_DOWN + HC_RANK + t];
        inj_out[t] = v;
    }
    __syncthreads();
    const int lane = t % WAVE, j = blockIdx.x * 8 + t / WAVE;
    const uint4 *uv = reinterpret_cast<const uint4 *>(u);
    constexpr int CH = HC_RANK / 8;  // 40 uint4
    float out = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        const uint4 *row = up + (size_t(j) * HC + s) * CH;
        float acc = dot8(row[lane], uv[lane], 0.f);
        if (lane + WAVE < CH) acc = dot8(row[lane + WAVE], uv[lane + WAVE], acc);
        const float g = wave_sum(acc);
        out += sigmoid(g) * __half2float(y[s * SH + j]) * r[s];
    }
    if (lane == 0) block_in[j] = __float2half(out / HC);
}

void hc_up_mix(const float *red, const uint16_t *up, const uint16_t *y, uint16_t *block_in,
               float *inj_out, hipStream_t s) {
    hc_up_mix_kernel<<<SH / 8, 256, 0, s>>>(red, reinterpret_cast<const uint4 *>(up),
                                            reinterpret_cast<const __half *>(y),
                                            reinterpret_cast<__half *>(block_in), inj_out);
}

__global__ void hc_combine_kernel(float *X, const float *pend, const float *inj) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HC * SH) return;
    X[i] += pend[i % SH] * 2.f * sigmoid(inj[i / SH] / HC);
}

void hc_combine(float *X, const float *pend, const float *inj, hipStream_t s) {
    hc_combine_kernel<<<HC * SH / 256, 256, 0, s>>>(X, pend, inj);
}

__global__ void embed_kernel(float *X, const float *emb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HC * SH) X[i] = emb[i % SH];
}

void embed_streams(float *X, const float *emb, hipStream_t s) {
    embed_kernel<<<HC * SH / 256, 256, 0, s>>>(X, emb);
}

// ------------------------------------------------------------------ QSA
__global__ __launch_bounds__(256) void qsa_prep_kernel(QsaStep p) {
    __shared__ float buf[HEAD_DIM], red[8];
    const int t = threadIdx.x;
    const int64_t pos = *p.pos;
    const float *kp = p.proj + QSA_LOCAL_HEADS * 2 * HEAD_DIM;
    const float *vp = kp + HEAD_DIM;
    const float *ip = vp + HEAD_DIM;
    for (int h = 0; h < QSA_LOCAL_HEADS; ++h) {
        const float *src = p.proj + h * 2 * HEAD_DIM;
        norm_shared(src, p.qn, buf, HEAD_DIM, red);
        rope_shared(buf, pos);
        p.q[h * HEAD_DIM + t] = __half_as_ushort(__float2half(buf[t]));
        p.gate[h * HEAD_DIM + t] = src[HEAD_DIM + t];
        __syncthreads();
    }
    norm_shared(kp, p.kn, buf, HEAD_DIM, red);
    rope_shared(buf, pos);
    p.K[pos * HEAD_DIM + t] = __half_as_ushort(__float2half(buf[t]));
    p.V[pos * HEAD_DIM + t] = __half_as_ushort(__float2half(vp[t]));
    __syncthreads();
    for (int h = 0; h < IDX_HEADS; ++h) {
        norm_shared(ip + h * IDX_DIM, p.iqn, buf, IDX_DIM, red);
        rope_shared(buf, pos);
        if (t < IDX_DIM) p.iq[h * IDX_DIM + t] = buf[t];
        __syncthreads();
    }
    const float *rk = ip + IDX_HEADS * IDX_DIM;
    if (t < IDX_DIM) p.raw_k[pos * IDX_DIM + t] = rk[t];
    if ((pos + 1) % IDX_RATIO == 0) {
        __shared__ float mean[IDX_DIM];
        if (t < IDX_DIM) {
            float m = rk[t];
            for (int r = 1; r < IDX_RATIO; ++r) m += p.raw_k[(pos - r) * IDX_DIM + t];
            mean[t] = m / IDX_RATIO;
        }
        __syncthreads();
        norm_shared(mean, p.ikn, buf, IDX_DIM, red);
        rope_shared(buf, pos + 1 - IDX_RATIO);
        if (t < IDX_DIM) p.ck[(pos / IDX_RATIO) * IDX_DIM + t] = __half_as_ushort(__float2half(buf[t]));
    }
}

void qsa_prep(const QsaStep &p, hipStream_t s) { qsa_prep_kernel<<<1, 256, 0, s>>>(p); }

// 16 lanes per compressed key: lane l covers dims 8l..8l+7. Fixed grid,
// strided over the groups, so the launch shape does not depend on pos.
constexpr int SCORE_BLOCKS = 256;
__global__ __launch_bounds__(256) void qsa_scores_kernel(const float *iq, const uint4 *ck,
                                                         float *scores, const int64_t *posp) {
    const int nb = int((*posp + 1) / IDX_RATIO);
    if (nb <= IDX_TOPK_BLOCKS) return;
    const int lane16 = threadIdx.x % 16;
    for (int c0 = blockIdx.x * 16; c0 < nb; c0 += SCORE_BLOCKS * 16) {
    const int c = c0 + threadIdx.x / 16;
    const bool ok = c < nb;
    uint4 kv = ok ? ck[size_t(c) * (IDX_DIM / 8) + lane16] : make_uint4(0, 0, 0, 0);
    const __half *kh = reinterpret_cast<const __half *>(&kv);
    float total = 0.f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float *qh = iq + h * IDX_DIM + lane16 * 8;
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) d += qh[i] * __half2float(kh[i]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) d += __shfl_xor(d, o, 16);
        total += fmaxf(d, 0.f);
    }
    if (ok && lane16 == 0) scores[c] = total * rsqrtf(float(IDX_DIM));
    }
}

void qsa_index_scores(const float *iq, const uint16_t *ck, float *scores, const int64_t *pos,
                      hipStream_t s) {
    qsa_scores_kernel<<<SCORE_BLOCKS, 256, 0, s>>>(iq, reinterpret_cast<const uint4 *>(ck), scores, pos);
}

__global__ __launch_bounds__(1024) void qsa_select_kernel(const float *scores, const int64_t *posp,
                                                          int32_t *list, int32_t *count) {
    select_body(scores, *posp, list, count);
}

void qsa_select(const float *scores, const int64_t *pos, int32_t *list, int32_t *count,
                hipStream_t s) {
    qsa_select_kernel<<<1, 1024, 0, s>>>(scores, pos, list, count);
}

__global__ __launch_bounds__(256) void qsa_attend_kernel(const uint4 *q, const uint4 *K, const uint4 *V,
                                                         const int32_t *list, const int32_t *count,
                                                         float *partial) {
    attend_body(q, K, V, list, *count, partial + size_t(blockIdx.x) * QSA_LOCAL_HEADS * ATT_REC);
}

__global__ __launch_bounds__(256) void qsa_combine_kernel(const float *partial, const float *gate, __half *out) {
    combine_body(partial, gate, out);
}

void qsa_attend(const uint16_t *q, const float *gate, const uint16_t *K, const uint16_t *V,
                const int32_t *list, const int32_t *count, float *partial, uint16_t *out,
                hipStream_t s) {
    qsa_attend_kernel<<<ATT_BLOCKS, 256, 0, s>>>(reinterpret_cast<const uint4 *>(q),
                                                 reinterpret_cast<const uint4 *>(K),
                                                 reinterpret_cast<const uint4 *>(V), list, count,
                                                 partial);
    qsa_combine_kernel<<<1, HEAD_DIM, 0, s>>>(partial, gate, reinterpret_cast<__half *>(out));
}

// ------------------------------------------------------------------ MoE
__global__ __launch_bounds__(512) void moe_route_kernel(const float *logits, int first, int n_local,
                                                        int32_t *ids, float *weights, int32_t *count) {
    __shared__ int top[TOP_K];
    __shared__ float topp[TOP_K];
    route_body(logits, top, topp);
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int k = 0; k < TOP_K; ++k) s += topp[k];
        int c = 0;
        for (int k = 0; k < TOP_K; ++k) {
            const int e = top[k];
            if (e >= first && e < first + n_local) {
                ids[c] = e - first;
                weights[c] = topp[k] / s;
                ++c;
            }
        }
        *count = c;
    }
}

void moe_route(const float *logits, int first, int n_local, int32_t *ids, float *weights,
               int32_t *count, hipStream_t s) {
    moe_route_kernel<<<1, N_EXPERTS, 0, s>>>(logits, first, n_local, ids, weights, count);
}

__global__ void swiglu_kernel(const float *v, int n, const float *scale, __half *h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float sc = scale ? sigmoid(*scale) : 1.f;
    h[i] = __float2half(silu(v[i]) * v[n + i] * sc);
}

void swiglu(const float *v, int n, const float *scale, uint16_t *h, hipStream_t s) {
    swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(v, n, scale, reinterpret_cast<__half *>(h));
}

__global__ void to_f16_kernel(const float *x, __half *y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

void to_f16(const float *x, uint16_t *y, int n, hipStream_t s) {
    to_f16_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, reinterpret_cast<__half *>(y), n);
}

// ------------------------------------------------------------------ PLE
__global__ __launch_bounds__(256) void ple_stats_kernel(const float *kv, const float *X, const float *nk,
                                                        const float *nq, float *send) {
    __shared__ float red[8];
    const float *key = kv, *val = kv + HC * SH;
    for (int s = 0; s < HC; ++s) {
        float sk = 0.f, sx = 0.f, d = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            const int i = s * SH + j;
            const float k = key[i], x = X[i];
            sk += k * k;
            sx += x * x;
            d += k * nk[i] * x * nq[i];
        }
        sk = block_sum<8>(sk, red);
        sx = block_sum<8>(sx, red);
        d = block_sum<8>(d, red);
        if (threadIdx.x == 0) {
            send[s] = sk;
            send[4 + s] = sx;
            send[8 + s] = d;
        }
    }
    float sv = 0.f;
    for (int j = threadIdx.x; j < SH; j += blockDim.x) sv += val[j] * val[j];
    sv = block_sum<8>(sv, red);
    if (threadIdx.x == 0) {
        send[12] = sv;
        send[13] = send[14] = send[15] = 0.f;
    }
}

void ple_stats(const float *kv, const float *X, const float *nk, const float *nq, float *send,
               hipStream_t s) {
    ple_stats_kernel<<<1, 256, 0, s>>>(kv, X, nk, nq, send);
}

constexpr int PLE_RING = 10;

__global__ void ple_apply_kernel(const float *kv, const float *red, const float *nc,
                                 const float *conv_w, float *ring, const int64_t *posp, float *X) {
    const int64_t pos = *posp;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HC * SH) return;
    const int s = i / SH, j = i % SH;
    const float rk = rsqrtf(red[s] / H + EPS), rq = rsqrtf(red[4 + s] / H + EPS);
    const float gs = red[8 + s] * rk * rq * rsqrtf(float(H));
    const float mag = sqrtf(fmaxf(fabsf(gs), 1e-6f));
    const float gate = sigmoid(gs > 0.f ? mag : (gs < 0.f ? -mag : 0.f));
    const float rc = rsqrtf(gate * gate * red[12] / H + EPS);
    const float gv = gate * kv[HC * SH + j];
    const int slot = int(pos % PLE_RING);
    ring[slot * HC * SH + i] = gv * rc * nc[i];
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < PLE_CONV; ++k) {
        const int64_t p = pos - (PLE_CONV - 1 - k) * PLE_DILATION;
        acc += conv_w[i * PLE_CONV + k] * ring[int(((p % PLE_RING) + PLE_RING) % PLE_RING) * HC * SH + i];
    }
    X[i] += gv + silu(acc);
}

void ple_apply(const float *kv, const float *red, const float *nc, const float *conv_w,
               float *ring, const int64_t *pos, float *X, hipStream_t s) {
    ple_apply_kernel<<<HC * SH / 256, 256, 0, s>>>(kv, red, nc, conv_w, ring, pos, X);
}

}  // namespace qw::gpu
