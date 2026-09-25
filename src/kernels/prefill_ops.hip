#include "kernels/prefill_ops.hpp"

#include "core/common.hpp"
#include "kernels/model_common.cuh"
#include "kernels/model_ops.hpp"

namespace qw::gpu {

namespace {
constexpr int GDN_VL = GDN_V_HEADS / 4;                             // 12
constexpr int GDN_KL = GDN_QK_HEADS / 4;                            // 4
constexpr int GDN_QKV_L = 2 * GDN_KL * GDN_DIM + GDN_VL * GDN_DIM;  // 2560
constexpr int GDN_PROJ_L = GDN_QKV_L + GDN_VL * GDN_DIM + 2 * GDN_VL;  // 4120
constexpr int QSA_PROJ_L = QSA_LOCAL_HEADS * 2 * HEAD_DIM + 2 * HEAD_DIM + (IDX_HEADS + 1) * IDX_DIM;
constexpr int XW = HC * SH;  // 2560 local residual columns
}  // namespace

void gemm(rocblas_handle h, const uint16_t *W, int N, int K, const uint16_t *X, int ldx, int T, void *Y,
          int ldy, bool f16_out) {
    const float alpha = 1.f, beta = 0.f;
    const auto ct = f16_out ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
    rocblas_status st = rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, T, K, &alpha, W,
                                        rocblas_datatype_f16_r, K, X, rocblas_datatype_f16_r, ldx, &beta, Y, ct, ldy,
                                        Y, ct, ldy, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (st != rocblas_status_success) fail("rocblas_gemm_ex failed: " + std::to_string(int(st)));
}

// ------------------------------------------------------------------ HC
__global__ __launch_bounds__(256) void hc_pre_T_kernel(float *X, const float *pend, const float *inj, float *ssq) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    float *x = X + size_t(t) * XW;
    for (int s = 0; s < HC; ++s) {
        const float wgt = pend ? 2.f * sigmoid(inj[t * HC + s] / HC) : 0.f;
        float ss = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            float v = x[s * SH + j];
            if (pend) {
                v += pend[size_t(t) * SH + j] * wgt;
                x[s * SH + j] = v;
            }
            ss += v * v;
        }
        ss = block_sum<8>(ss, red);
        if (threadIdx.x == 0) ssq[t * HC + s] = ss;
    }
}

void hc_pre_T(float *X, const float *pend, const float *inj, int T, float *ssq, hipStream_t s) {
    hc_pre_T_kernel<<<T, 256, 0, s>>>(X, pend, inj, ssq);
}

__global__ void hc_norm_T_kernel(const float *X, const float *ssq, const float *w1, int T, __half *xn) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * XW) return;
    const int t = int(i / XW), c = int(i % XW), s = c / SH;
    xn[i] = __float2half(X[i] * rsqrtf(ssq[t * HC + s] / H + EPS) * w1[c]);
}

void hc_norm_T(const float *X, const float *ssq, const float *w1, int T, uint16_t *xn, hipStream_t s) {
    hc_norm_T_kernel<<<(size_t(T) * XW + 255) / 256, 256, 0, s>>>(X, ssq, w1, T, reinterpret_cast<__half *>(xn));
}

__global__ void hc_mid_T_kernel(const float *d, int T, __half *u, float *inj) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D_PAD) return;
    const int t = i / D_PAD, o = i % D_PAD;
    const float v = d[i];
    if (o < HC_RANK) u[t * HC_RANK + o] = __float2half(silu(v / HC));
    else if (o < HC_RANK + HC && inj) inj[t * HC + o - HC_RANK] = v;
}

void hc_mid_T(const float *d, int T, uint16_t *u, float *inj, hipStream_t s) {
    hc_mid_T_kernel<<<(T * D_PAD + 255) / 256, 256, 0, s>>>(d, T, reinterpret_cast<__half *>(u), inj);
}

__global__ void hc_mix_T_kernel(const __half *g, const __half *xn, int T, __half *bin) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * SH) return;
    const int t = i / SH, j = i % SH;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s)
        acc += sigmoid(__half2float(g[size_t(t) * XW + j * HC + s])) * __half2float(xn[size_t(t) * XW + s * SH + j]);
    bin[i] = __float2half(acc / HC);
}

void hc_mix_T(const uint16_t *g, const uint16_t *xn, int T, uint16_t *bin, hipStream_t s) {
    hc_mix_T_kernel<<<(T * SH + 255) / 256, 256, 0, s>>>(reinterpret_cast<const __half *>(g),
                                                         reinterpret_cast<const __half *>(xn), T,
                                                         reinterpret_cast<__half *>(bin));
}

__global__ void hc_combine_T_kernel(float *X, const float *pend, const float *inj, int T) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * XW) return;
    const int t = int(i / XW), c = int(i % XW);
    X[i] += pend[size_t(t) * SH + c % SH] * 2.f * sigmoid(inj[t * HC + c / SH] / HC);
}

void hc_combine_T(float *X, const float *pend, const float *inj, int T, hipStream_t s) {
    hc_combine_T_kernel<<<(size_t(T) * XW + 255) / 256, 256, 0, s>>>(X, pend, inj, T);
}

__global__ void embed_T_kernel(float *X, const float *emb, int T) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * XW) return;
    const int t = int(i / XW), c = int(i % XW);
    X[i] = emb[size_t(t) * SH + c % SH];
}

void embed_T(float *X, const float *emb, int T, hipStream_t s) {
    embed_T_kernel<<<(size_t(T) * XW + 255) / 256, 256, 0, s>>>(X, emb, T);
}

// ------------------------------------------------------------------ GDN
// One block per token. Channel layout (local): q 0..511 | k 512..1023 | v 1024..2559.
__global__ __launch_bounds__(256) void gdn_conv_T_kernel(const __half *P, const float *ring, const float *conv_w,
                                                         const float *A_log, const float *dt_bias, int64_t start,
                                                         float *qkv, float *gb) {
    __shared__ float vals[GDN_QKV_L];
    const int t = blockIdx.x;
    const int64_t pos = start + t;
    for (int ch = threadIdx.x; ch < GDN_QKV_L; ch += blockDim.x) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < GDN_CONV; ++k) {
            const int64_t q = pos - (GDN_CONV - 1 - k);
            float x;
            if (q >= start) x = __half2float(P[size_t(q - start) * GDN_PROJ_L + ch]);
            else if (q >= 0) x = ring[int(q % GDN_RING) * GDN_QKV_L + ch];
            else x = 0.f;
            acc += conv_w[ch * GDN_CONV + k] * x;
        }
        vals[ch] = silu(acc);
    }
    __syncthreads();
    // l2-normalize the 4 q heads and 4 k heads: wave w handles head w
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    {
        float ss = 0.f;
        for (int i = lane; i < GDN_DIM; i += WAVE) ss += vals[w * GDN_DIM + i] * vals[w * GDN_DIM + i];
        ss = wave_sum(ss);
        const float r = rsqrtf(ss + EPS) * (w < GDN_KL ? rsqrtf(float(GDN_DIM)) : 1.f);
        for (int i = lane; i < GDN_DIM; i += WAVE) vals[w * GDN_DIM + i] *= r;
    }
    __syncthreads();
    float *o = qkv + size_t(t) * GDN_QKV_L;
    for (int ch = threadIdx.x; ch < GDN_QKV_L; ch += blockDim.x) o[ch] = vals[ch];
    if (threadIdx.x < GDN_VL) {
        const int h = threadIdx.x;
        const __half *row = P + size_t(t) * GDN_PROJ_L;
        const float b = __half2float(row[GDN_QKV_L + GDN_VL * GDN_DIM + h]);
        const float a = __half2float(row[GDN_QKV_L + GDN_VL * GDN_DIM + GDN_VL + h]) + dt_bias[h];
        const float sp = a > 20.f ? a : log1pf(__expf(a));
        gb[(t * GDN_VL + h) * 2] = __expf(-__expf(A_log[h]) * sp);
        gb[(t * GDN_VL + h) * 2 + 1] = sigmoid(b);
    }
}

void gdn_conv_T(const uint16_t *P, const float *ring, const float *conv_w, const float *A_log,
                const float *dt_bias, int64_t start, int T, float *qkv, float *gb, hipStream_t s) {
    gdn_conv_T_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(P), ring, conv_w, A_log, dt_bias, start,
                                        qkv, gb);
}

__global__ void gdn_ring_update_kernel(const __half *P, float *ring, int64_t start, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= GDN_QKV_L) return;
    for (int64_t q = std::max<int64_t>(start, start + T - (GDN_CONV - 1)); q < start + T; ++q)
        ring[int(q % GDN_RING) * GDN_QKV_L + ch] = __half2float(P[size_t(q - start) * GDN_PROJ_L + ch]);
}

void gdn_ring_update_T(const uint16_t *P, float *ring, int64_t start, int T, hipStream_t s) {
    gdn_ring_update_kernel<<<GDN_QKV_L / 256, 256, 0, s>>>(reinterpret_cast<const __half *>(P), ring, start, T);
}

// grid 48 = 12 heads x 4 slices of 32 v-columns. Thread (rg, c): column
// j = slice*32 + c, k-rows rg*16 .. +16. Columns of the delta-rule state are
// independent, so slices need no communication.
__global__ __launch_bounds__(256) void gdn_scan_T_kernel(const float *qkv, const float *gb, float *S, int T,
                                                         float *o_raw) {
    __shared__ float red[8][32], red2[8][32];
    const int h = blockIdx.x / 4, slice = blockIdx.x % 4;
    const int kh = h / (GDN_V_HEADS / GDN_QK_HEADS);
    const int c = threadIdx.x % 32, rg = threadIdx.x / 32;
    const int j = slice * 32 + c;
    float *Sh = S + size_t(h) * GDN_DIM * GDN_DIM;
    float s[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = Sh[(rg * 16 + i) * GDN_DIM + j];
    for (int t = 0; t < T; ++t) {
        const float *row = qkv + size_t(t) * GDN_QKV_L;
        const float *kk = row + GDN_KL * GDN_DIM + kh * GDN_DIM + rg * 16;
        const float *qq = row + kh * GDN_DIM + rg * 16;
        const float decay = gb[(t * GDN_VL + h) * 2], beta = gb[(t * GDN_VL + h) * 2 + 1];
        float pu = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            s[i] *= decay;
            pu += s[i] * kk[i];
        }
        red[rg][c] = pu;
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int g = 0; g < 8; ++g) tot += red[g][c];
        const float u = (row[2 * GDN_KL * GDN_DIM + h * GDN_DIM + j] - tot) * beta;
        __syncthreads();
        float po = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            s[i] += kk[i] * u;
            po += s[i] * qq[i];
        }
        red2[rg][c] = po;
        __syncthreads();
        if (rg == 0) {
            float o = 0.f;
#pragma unroll
            for (int g = 0; g < 8; ++g) o += red2[g][c];
            o_raw[size_t(t) * GDN_VL * GDN_DIM + h * GDN_DIM + j] = o;
        }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) Sh[(rg * 16 + i) * GDN_DIM + j] = s[i];
}

void gdn_scan_T(const float *qkv, const float *gb, float *S, int T, float *o_raw, hipStream_t s) {
    gdn_scan_T_kernel<<<GDN_VL * 4, 256, 0, s>>>(qkv, gb, S, T, o_raw);
}

// grid (T, 12), 128 threads
__global__ __launch_bounds__(128) void gdn_norm_T_kernel(const float *o_raw, const __half *P, const float *w,
                                                         __half *out) {
    __shared__ float red[4];
    const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    const size_t i = size_t(t) * GDN_VL * GDN_DIM + h * GDN_DIM + j;
    const float o = o_raw[i];
    const float ss = block_sum<4>(o * o, red);
    const float z = __half2float(P[size_t(t) * GDN_PROJ_L + GDN_QKV_L + h * GDN_DIM + j]);
    out[i] = __float2half(o * rsqrtf(ss / GDN_DIM + EPS) * w[j] * sigmoid(z));
}

void gdn_norm_T(const float *o_raw, const uint16_t *P, const float *norm_w, int T, uint16_t *out,
                hipStream_t s) {
    gdn_norm_T_kernel<<<dim3(T, GDN_VL), 128, 0, s>>>(o_raw, reinterpret_cast<const __half *>(P), norm_w,
                                                      reinterpret_cast<__half *>(out));
}

// ------------------------------------------------------------------ QSA
// One block per token.
__global__ __launch_bounds__(256) void qsa_prep_T_kernel(const __half *P, const float *qn, const float *kn,
                                                         const float *iqn, int64_t start, __half *q16,
                                                         __half *gate, __half *iq16, __half *K, __half *V,
                                                         float *raw_k) {
    __shared__ float src[HEAD_DIM], buf[HEAD_DIM], red[8];
    const int t = blockIdx.x, th = threadIdx.x;
    const int64_t pos = start + t;
    const __half *row = P + size_t(t) * QSA_PROJ_L;
    for (int h = 0; h < QSA_LOCAL_HEADS; ++h) {
        __syncthreads();
        src[th] = __half2float(row[h * 2 * HEAD_DIM + th]);
        gate[size_t(t) * QSA_LOCAL_HEADS * HEAD_DIM + h * HEAD_DIM + th] = row[h * 2 * HEAD_DIM + HEAD_DIM + th];
        __syncthreads();
        norm_shared(src, qn, buf, HEAD_DIM, red);
        rope_shared(buf, pos);
        q16[size_t(t) * QSA_LOCAL_HEADS * HEAD_DIM + h * HEAD_DIM + th] = __float2half(buf[th]);
    }
    const __half *kp = row + QSA_LOCAL_HEADS * 2 * HEAD_DIM, *vp = kp + HEAD_DIM, *ip = vp + HEAD_DIM;
    __syncthreads();
    src[th] = __half2float(kp[th]);
    __syncthreads();
    norm_shared(src, kn, buf, HEAD_DIM, red);
    rope_shared(buf, pos);
    K[pos * HEAD_DIM + th] = __float2half(buf[th]);
    V[pos * HEAD_DIM + th] = vp[th];
    for (int h = 0; h < IDX_HEADS; ++h) {
        __syncthreads();
        if (th < IDX_DIM) src[th] = __half2float(ip[h * IDX_DIM + th]);
        __syncthreads();
        norm_shared(src, iqn, buf, IDX_DIM, red);
        rope_shared(buf, pos);
        if (th < IDX_DIM) iq16[(size_t(t) * IDX_HEADS + h) * IDX_DIM + th] = __float2half(buf[th]);
    }
    if (th < IDX_DIM) raw_k[pos * IDX_DIM + th] = __half2float(ip[IDX_HEADS * IDX_DIM + th]);
}

void qsa_prep_T(const uint16_t *P, const float *qn, const float *kn, const float *iqn, int64_t start,
                int T, uint16_t *q16, uint16_t *gate, uint16_t *iq16, uint16_t *K, uint16_t *V,
                float *raw_k, hipStream_t s) {
    qsa_prep_T_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(P), qn, kn, iqn, start,
                                        reinterpret_cast<__half *>(q16), reinterpret_cast<__half *>(gate),
                                        reinterpret_cast<__half *>(iq16), reinterpret_cast<__half *>(K),
                                        reinterpret_cast<__half *>(V), raw_k);
}

// One 128-thread block per group completed in the chunk.
__global__ __launch_bounds__(128) void qsa_compress_T_kernel(const float *raw_k, const float *ikn, int64_t c0,
                                                             __half *ck) {
    __shared__ float mean[IDX_DIM], buf[IDX_DIM], red[4];
    const int64_t c = c0 + blockIdx.x;
    const int th = threadIdx.x;
    float m = 0.f;
    for (int r = 0; r < IDX_RATIO; ++r) m += raw_k[(c * IDX_RATIO + r) * IDX_DIM + th];
    mean[th] = m / IDX_RATIO;
    __syncthreads();
    float ss = block_sum<4>(mean[th] * mean[th], red);
    buf[th] = mean[th] * rsqrtf(ss / IDX_DIM + EPS) * ikn[th];
    rope_shared(buf, c * IDX_RATIO);
    ck[c * IDX_DIM + th] = __float2half(buf[th]);
}

void qsa_compress_T(const float *raw_k, const float *ikn, int64_t start, int T, uint16_t *ck,
                    hipStream_t s) {
    const int64_t c_first = start / IDX_RATIO;           // first group whose last token is >= start
    const int64_t c_last = (start + T - IDX_RATIO) / IDX_RATIO;  // last group completed by the chunk
    if (start + T < IDX_RATIO || c_last < c_first) return;
    qsa_compress_T_kernel<<<int(c_last - c_first + 1), 128, 0, s>>>(raw_k, ikn, c_first,
                                                                  reinterpret_cast<__half *>(ck));
}

__global__ void qsa_score_reduce_kernel(const __half *sc16, int ldsc, int64_t q0, float *scores) {
    const int q = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = int((q0 + q + 1) / IDX_RATIO);
    if (c >= nb) return;
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) s += fmaxf(__half2float(sc16[(size_t(q) * IDX_HEADS + h) * ldsc + c]), 0.f);
    scores[size_t(q) * ldsc + c] = s * rsqrtf(float(IDX_DIM));
}

void qsa_score_reduce_T(const uint16_t *sc16, int ldsc, int64_t q0, int Q, float *scores, hipStream_t s) {
    const int nb_max = int((q0 + Q) / IDX_RATIO);
    qsa_score_reduce_kernel<<<dim3((nb_max + 255) / 256, Q), 256, 0, s>>>(reinterpret_cast<const __half *>(sc16),
                                                                         ldsc, q0, scores);
}

__global__ __launch_bounds__(1024) void qsa_select_T_kernel(const float *scores, int ldsc, int64_t q0,
                                                            int32_t *lists, int32_t *counts) {
    const int q = blockIdx.x;
    select_body(scores + size_t(q) * ldsc, q0 + q, lists + size_t(q) * LIST_W, counts + q);
}

void qsa_select_T(const float *scores, int ldsc, int64_t q0, int Q, int32_t *lists, int32_t *counts,
                  hipStream_t s) {
    qsa_select_T_kernel<<<Q, 1024, 0, s>>>(scores, ldsc, q0, lists, counts);
}

__global__ __launch_bounds__(256) void qsa_attend_T_kernel(const uint4 *q16, const uint4 *K, const uint4 *V,
                                                           const int32_t *lists, const int32_t *counts,
                                                           float *partial) {
    const int q = blockIdx.y;
    attend_body(q16 + size_t(q) * QSA_LOCAL_HEADS * (HEAD_DIM / 8), K, V, lists + size_t(q) * LIST_W, counts[q],
                partial + (size_t(q) * ATT_BLOCKS + blockIdx.x) * QSA_LOCAL_HEADS * ATT_REC);
}

__global__ __launch_bounds__(256) void qsa_combine_T_kernel(const float *partial, const __half *gate, __half *out) {
    const int q = blockIdx.x;
    combine_body(partial + size_t(q) * ATT_BLOCKS * QSA_LOCAL_HEADS * ATT_REC,
                 gate + size_t(q) * QSA_LOCAL_HEADS * HEAD_DIM, out + size_t(q) * QSA_LOCAL_HEADS * HEAD_DIM);
}

void qsa_attend_T(const uint16_t *q16, const uint16_t *gate, const uint16_t *K, const uint16_t *V,
                  const int32_t *lists, const int32_t *counts, int Q, float *partial, uint16_t *out,
                  hipStream_t s) {
    qsa_attend_T_kernel<<<dim3(ATT_BLOCKS, Q), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(q16), reinterpret_cast<const uint4 *>(K), reinterpret_cast<const uint4 *>(V),
        lists, counts, partial);
    qsa_combine_T_kernel<<<Q, HEAD_DIM, 0, s>>>(partial, reinterpret_cast<const __half *>(gate),
                                                reinterpret_cast<__half *>(out));
}

// ------------------------------------------------------------------ MoE
constexpr int EXP_L = N_EXPERTS / 4;  // 128

__global__ __launch_bounds__(512) void moe_route_T_kernel(const float *logits, int first, int32_t *tok_e,
                                                          float *tok_w, int32_t *counts) {
    __shared__ int top[TOP_K];
    __shared__ float topp[TOP_K];
    const int t = blockIdx.x;
    route_body(logits + size_t(t) * N_EXPERTS, top, topp);
    if (threadIdx.x < TOP_K) {
        float sum = 0.f;
        for (int k = 0; k < TOP_K; ++k) sum += topp[k];
        const int k = threadIdx.x, e = top[k];
        const bool local = e >= first && e < first + EXP_L;
        tok_e[t * TOP_K + k] = local ? e - first : -1;
        tok_w[t * TOP_K + k] = topp[k] / sum;
        if (local) atomicAdd(&counts[e - first], 1);
    }
}

void moe_route_T(const float *logits, int T, int first, int32_t *tok_e, float *tok_w, int32_t *counts,
                 hipStream_t s) {
    moe_route_T_kernel<<<T, N_EXPERTS, 0, s>>>(logits, first, tok_e, tok_w, counts);
}

__global__ void moe_offsets_kernel(const int32_t *counts, int32_t *offsets, int32_t *fill) {
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < EXP_L; ++e) {
            offsets[e] = acc;
            acc += counts[e];
            fill[e] = 0;
        }
        offsets[EXP_L] = acc;
    }
}

__global__ void moe_scatter_kernel(const int32_t *tok_e, const int32_t *offsets, int32_t *fill, int T,
                                   int32_t *tok_slot, int32_t *pair_tok) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * TOP_K) return;
    const int e = tok_e[i];
    if (e < 0) {
        tok_slot[i] = -1;
        return;
    }
    const int slot = offsets[e] + atomicAdd(&fill[e], 1);
    tok_slot[i] = slot;
    pair_tok[slot] = i / TOP_K;
}

void moe_scatter_T(const int32_t *tok_e, const int32_t *counts, int T, int32_t *offsets, int32_t *fill,
                   int32_t *tok_slot, int32_t *pair_tok, hipStream_t s) {
    moe_offsets_kernel<<<1, 64, 0, s>>>(counts, offsets, fill);
    moe_scatter_kernel<<<(T * TOP_K + 255) / 256, 256, 0, s>>>(tok_e, offsets, fill, T, tok_slot, pair_tok);
}

// Grouped int4 GEMM for the routed experts. Work is a device-built list of
// (expert, 32-token tile); each block computes a 32-token x BN-row output tile
// with the int4 weights dequantized into LDS once per 64-wide K step, and a
// register micro-tile per thread (4 tokens x 2 rows [x gate/up]).
// LDS: A tiles use a 72-half row stride (16-byte aligned, reads broadcast);
// B tiles a 66-half stride (33 words) so the 32 lanes of a wave, which read 32
// consecutive rows, hit 32 different banks.
constexpr int MT = 32, BK = 64, A_LD = BK + 8, B_LD = BK + 2;

__global__ void moe_tiles_kernel(const int32_t *counts, int32_t *tile_e, int32_t *tile_m0, int32_t *ntiles) {
    if (threadIdx.x != 0) return;
    int n = 0;
    for (int e = 0; e < EXP_L; ++e)
        for (int m0 = 0; m0 < counts[e]; m0 += MT) {
            tile_e[n] = e;
            tile_m0[n] = m0;
            ++n;
        }
    *ntiles = n;
}

// Dequantize one uint4 of packed int4 (32 consecutive k) * scale into LDS
// halves dst[0..32) (4-byte stores: the row stride is not 16-byte aligned).
__device__ __forceinline__ void dequant32(const uint4 &w, float scale, __half *dst) {
    const __half2 sc = __float2half2_rn(scale);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    __half2 *d2 = reinterpret_cast<__half2 *>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int sh = 0; sh < 4; ++sh) {
            const uint32_t v = nib2_to_h2(ws[i] >> (4 * sh));
            d2[i * 4 + sh] = __hmul2(*reinterpret_cast<const __half2 *>(&v), sc);
        }
}

// gate/up: grid (max tiles, FFN / 64). 256 threads: ty = tid/32 owns tokens
// ty*4..+3, tx = tid%32 owns rows tx and tx+32 (of gate and of up).
__global__ __launch_bounds__(256) void moe_gate_up_T_kernel(const uint4 *gw, const __half *gs, const uint4 *uw,
                                                            const __half *us, const int32_t *counts,
                                                            const int32_t *offsets, const int32_t *tile_e,
                                                            const int32_t *tile_m0, const int32_t *ntiles,
                                                            const int32_t *pair_tok, const uint4 *bin, __half *h) {
    constexpr int CH = H / 32;  // packed uint4 per weight row
    __shared__ __align__(16) __half As[MT][A_LD];
    __shared__ __half Bg[64][B_LD], Bu[64][B_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 64;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    // A loader: token tid/8, 8-half chunk tid%8
    const int la_tok = tid / 8, la_c = tid % 8;
    const uint4 *a_src = la_tok < mt ? bin + size_t(pair_tok[base + la_tok]) * (H / 8) : nullptr;
    // B loader: threads 0..127 gate, 128..255 up; row (tid%128)/2, half (tid%2) of the 64 k
    const bool lb_up = tid >= 128;
    const int lb_row = (tid % 128) / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * FFN + row0 + lb_row;
    const uint4 *b_src = (lb_up ? uw : gw) + wrow * CH;
    const __half *b_sc = (lb_up ? us : gs) + wrow * (H / QGROUP);
    __half *b_dst = (lb_up ? &Bu[lb_row][0] : &Bg[lb_row][0]) + lb_part * 32;

    float ag[4][2] = {}, au[4][2] = {};
    for (int k0 = 0; k0 < H; k0 += BK) {
        __syncthreads();
        *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 8]) = a_src ? a_src[k0 / 8 + la_c] : make_uint4(0, 0, 0, 0);
        dequant32(b_src[k0 / 32 + lb_part], __half2float(b_sc[k0 / QGROUP]), b_dst);
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; kk += 2) {
            uint32_t a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
            const uint32_t g0 = *reinterpret_cast<const uint32_t *>(&Bg[tx][kk]);
            const uint32_t g1 = *reinterpret_cast<const uint32_t *>(&Bg[tx + 32][kk]);
            const uint32_t u0 = *reinterpret_cast<const uint32_t *>(&Bu[tx][kk]);
            const uint32_t u1 = *reinterpret_cast<const uint32_t *>(&Bu[tx + 32][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ag[i][0] = dot2(a[i], g0, ag[i][0]);
                ag[i][1] = dot2(a[i], g1, ag[i][1]);
                au[i][0] = dot2(a[i], u0, au[i][0]);
                au[i][1] = dot2(a[i], u1, au[i][1]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            h[size_t(base + tok) * FFN + row0 + tx + 32 * j] = __float2half(silu(ag[i][j]) * au[i][j]);
    }
}

// down: grid (max tiles, H / 128). tx owns rows tx, +32, +64, +96.
__global__ __launch_bounds__(256) void moe_down_T_kernel(const uint4 *dw, const __half *ds, const int32_t *counts,
                                                         const int32_t *offsets, const int32_t *tile_e,
                                                         const int32_t *tile_m0, const int32_t *ntiles,
                                                         const uint4 *h, __half *yp) {
    constexpr int CH = FFN / 32;  // 20
    __shared__ __align__(16) __half As[MT][A_LD];
    __shared__ __half Bs[128][B_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 128;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    const int la_tok = tid / 8, la_c = tid % 8;
    const uint4 *a_src = la_tok < mt ? h + size_t(base + la_tok) * (FFN / 8) : nullptr;
    const int lb_row = tid / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * H + row0 + lb_row;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < FFN; k0 += BK) {
        __syncthreads();
        *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 8]) = a_src ? a_src[k0 / 8 + la_c] : make_uint4(0, 0, 0, 0);
        dequant32(dw[wrow * CH + k0 / 32 + lb_part], __half2float(ds[wrow * (FFN / QGROUP) + k0 / QGROUP]),
                  &Bs[lb_row][lb_part * 32]);
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; kk += 2) {
            uint32_t a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const uint32_t *>(&Bs[tx + 32 * j][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = dot2(a[i], b[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) yp[size_t(base + tok) * H + row0 + tx + 32 * j] = __float2half(acc[i][j]);
    }
}

// ---------------------------------------------------------------- W4A8
// Same tiling as above, on v_dot4_i32_i8 (1.7x the fp16 dot rate on gfx1030):
// activations quantized per row to int8 (x = q * scale), int4 weights widened
// to int8 exactly (nibble - 8). int32 accumulation per 64-wide K step, then the
// weight group scale (constant within a step, groups are 128 wide) in fp32.
// LDS rows: A 80 bytes (16-byte aligned), B 68 bytes (17 words: conflict-free
// across the 32 consecutive rows a wave reads).
constexpr int A8_LD = BK + 16, B8_LD = BK + 4;

// Per-(row, 128-group) int8 quantization: scale[row][g] = max|x| / 127 over
// the group. One wave per group (4 values per lane).
__global__ __launch_bounds__(256) void quant_rows_kernel(const __half *x, int n, int8_t *q, float *scale) {
    const int row = blockIdx.x, ng = n / QGROUP;
    const int lane = threadIdx.x % WAVE;
    for (int g = threadIdx.x / WAVE; g < ng; g += blockDim.x / WAVE) {
        const __half *xg = x + size_t(row) * n + g * QGROUP;
        float v[QGROUP / WAVE], mx = 0.f;
#pragma unroll
        for (int i = 0; i < QGROUP / WAVE; ++i) {
            v[i] = __half2float(xg[lane + i * WAVE]);
            mx = fmaxf(mx, fabsf(v[i]));
        }
        mx = wave_max(mx);
        const float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc;
#pragma unroll
        for (int i = 0; i < QGROUP / WAVE; ++i)
            q[size_t(row) * n + g * QGROUP + lane + i * WAVE] = int8_t(__float2int_rn(v[i] * inv));
        if (lane == 0) scale[size_t(row) * ng + g] = sc;
    }
}

// 32 int4 weights (one packed uint4, engine nibble layout) -> 32 int8 into dst.
__device__ __forceinline__ void widen32(const uint4 &w, int8_t *dst) {
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    uint32_t *d = reinterpret_cast<uint32_t *>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // word i holds k = 8i..8i+7: k(2m) at bits 4m, k(2m+1) at bits 16+4m
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            lo |= (((ws[i] >> (4 * m)) & 0xfu) - 8u & 0xffu) << (16 * m);
            lo |= (((ws[i] >> (16 + 4 * m)) & 0xfu) - 8u & 0xffu) << (16 * m + 8);
            hi |= (((ws[i] >> (4 * (m + 2))) & 0xfu) - 8u & 0xffu) << (16 * m);
            hi |= (((ws[i] >> (16 + 4 * (m + 2))) & 0xfu) - 8u & 0xffu) << (16 * m + 8);
        }
        d[2 * i] = lo;
        d[2 * i + 1] = hi;
    }
}

__device__ __forceinline__ int sdot4(uint32_t a, uint32_t b, int c) {
    return __builtin_amdgcn_sdot4(int(a), int(b), c, false);
}

__global__ __launch_bounds__(256) void moe_gate_up_i8_kernel(const uint4 *gw, const __half *gs, const uint4 *uw,
                                                             const __half *us, const int32_t *counts,
                                                             const int32_t *offsets, const int32_t *tile_e,
                                                             const int32_t *tile_m0, const int32_t *ntiles,
                                                             const int32_t *pair_tok, const int8_t *xq,
                                                             const float *xs, __half *h) {
    constexpr int CH = H / 32;
    __shared__ __align__(16) int8_t As[MT][A8_LD];
    __shared__ __align__(4) int8_t Bg[64][B8_LD], Bu[64][B8_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 64;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    // A loader (threads 0..127): token tid/4, 16-byte chunk tid%4
    const int la_tok = tid / 4, la_c = tid % 4;
    const bool la = tid < 128 && la_tok < mt;
    const uint4 *a_src = la ? reinterpret_cast<const uint4 *>(xq + size_t(pair_tok[base + la_tok]) * H) : nullptr;
    const bool lb_up = tid >= 128;
    const int lb_row = (tid % 128) / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * FFN + row0 + lb_row;
    const uint4 *b_src = (lb_up ? uw : gw) + wrow * CH;
    int8_t *b_dst = (lb_up ? &Bu[lb_row][0] : &Bg[lb_row][0]) + lb_part * 32;
    const __half *gsr0 = gs + (size_t(e) * FFN + row0 + tx) * (H / QGROUP);
    const __half *usr0 = us + (size_t(e) * FFN + row0 + tx) * (H / QGROUP);
    const size_t rs32 = size_t(32) * (H / QGROUP);  // scale stride for row + 32
    const float *xsr[4];  // per-group activation scales of this thread's 4 tokens
#pragma unroll
    for (int i = 0; i < 4; ++i)
        xsr[i] = ty * 4 + i < mt ? xs + size_t(pair_tok[base + ty * 4 + i]) * (H / QGROUP) : nullptr;

    float fg[4][2] = {}, fu[4][2] = {};
    for (int k0 = 0; k0 < H; k0 += BK) {
        __syncthreads();
        if (tid < 128)
            *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 16]) = la ? a_src[k0 / 16 + la_c] : make_uint4(0, 0, 0, 0);
        widen32(b_src[k0 / 32 + lb_part], b_dst);
        __syncthreads();
        int ig[4][2] = {}, iu[4][2] = {};
#pragma unroll 4
        for (int kk = 0; kk < BK; kk += 4) {
            uint32_t a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
            const uint32_t g0 = *reinterpret_cast<const uint32_t *>(&Bg[tx][kk]);
            const uint32_t g1 = *reinterpret_cast<const uint32_t *>(&Bg[tx + 32][kk]);
            const uint32_t u0 = *reinterpret_cast<const uint32_t *>(&Bu[tx][kk]);
            const uint32_t u1 = *reinterpret_cast<const uint32_t *>(&Bu[tx + 32][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ig[i][0] = sdot4(a[i], g0, ig[i][0]);
                ig[i][1] = sdot4(a[i], g1, ig[i][1]);
                iu[i][0] = sdot4(a[i], u0, iu[i][0]);
                iu[i][1] = sdot4(a[i], u1, iu[i][1]);
            }
        }
        const int grp = k0 / QGROUP;
        const float sg0 = __half2float(gsr0[grp]), sg1 = __half2float(gsr0[rs32 + grp]);
        const float su0 = __half2float(usr0[grp]), su1 = __half2float(usr0[rs32 + grp]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float xg = xsr[i] ? xsr[i][grp] : 0.f;
            fg[i][0] += float(ig[i][0]) * sg0 * xg;
            fg[i][1] += float(ig[i][1]) * sg1 * xg;
            fu[i][0] += float(iu[i][0]) * su0 * xg;
            fu[i][1] += float(iu[i][1]) * su1 * xg;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            h[size_t(base + tok) * FFN + row0 + tx + 32 * j] = __float2half(silu(fg[i][j]) * fu[i][j]);
    }
}

__global__ __launch_bounds__(256) void moe_down_i8_kernel(const uint4 *dw, const __half *ds, const int32_t *counts,
                                                          const int32_t *offsets, const int32_t *tile_e,
                                                          const int32_t *tile_m0, const int32_t *ntiles,
                                                          const int8_t *hq, const float *hs, __half *yp) {
    constexpr int CH = FFN / 32;
    __shared__ __align__(16) int8_t As[MT][A8_LD];
    __shared__ __align__(4) int8_t Bs[128][B8_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 128;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    const int la_tok = tid / 4, la_c = tid % 4;
    const bool la = tid < 128 && la_tok < mt;
    const uint4 *a_src = la ? reinterpret_cast<const uint4 *>(hq + size_t(base + la_tok) * FFN) : nullptr;
    const int lb_row = tid / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * H + row0 + lb_row;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < FFN; k0 += BK) {
        __syncthreads();
        if (tid < 128)
            *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 16]) = la ? a_src[k0 / 16 + la_c] : make_uint4(0, 0, 0, 0);
        widen32(dw[wrow * CH + k0 / 32 + lb_part], &Bs[lb_row][lb_part * 32]);
        __syncthreads();
        int ia[4][4] = {};
#pragma unroll 4
        for (int kk = 0; kk < BK; kk += 4) {
            uint32_t a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const uint32_t *>(&Bs[tx + 32 * j][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) ia[i][j] = sdot4(a[i], b[j], ia[i][j]);
        }
        const int grp = k0 / QGROUP;
        float hg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) hg[i] = ty * 4 + i < mt ? hs[size_t(base + ty * 4 + i) * (FFN / QGROUP) + grp] : 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float sc = __half2float(ds[(size_t(e) * H + row0 + tx + 32 * j) * (FFN / QGROUP) + grp]);
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i][j] += float(ia[i][j]) * sc * hg[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) yp[size_t(base + tok) * H + row0 + tx + 32 * j] = __float2half(acc[i][j]);
    }
}

void moe_experts_T(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                   const uint32_t *dw, const uint16_t *ds, const int32_t *counts, const int32_t *offsets,
                   const int32_t *pair_tok, const uint16_t *bin, int T, int32_t *tiles, uint16_t *h, uint16_t *yp,
                   hipStream_t s, int8_t *q8, float *q8s) {
    const int max_tiles = T * TOP_K / MT + EXP_L;
    int32_t *tile_e = tiles, *tile_m0 = tiles + max_tiles, *ntiles = tiles + 2 * max_tiles;
    moe_tiles_kernel<<<1, 64, 0, s>>>(counts, tile_e, tile_m0, ntiles);
    if (q8) {  // W4A8. q8: >= T*10*640 bytes, q8s: >= T*10*5 floats (per-128 group scales)
        quant_rows_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(bin), H, q8, q8s);
        moe_gate_up_i8_kernel<<<dim3(max_tiles, FFN / 64), 256, 0, s>>>(
            reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs),
            reinterpret_cast<const uint4 *>(uw), reinterpret_cast<const __half *>(us), counts, offsets, tile_e,
            tile_m0, ntiles, pair_tok, q8, q8s, reinterpret_cast<__half *>(h));
        const int n_pairs = T * TOP_K;  // upper bound; rows past the real pair count are never read
        quant_rows_kernel<<<n_pairs, 256, 0, s>>>(reinterpret_cast<const __half *>(h), FFN, q8, q8s);
        moe_down_i8_kernel<<<dim3(max_tiles, H / 128), 256, 0, s>>>(
            reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), counts, offsets, tile_e,
            tile_m0, ntiles, q8, q8s, reinterpret_cast<__half *>(yp));
        return;
    }
    moe_gate_up_T_kernel<<<dim3(max_tiles, FFN / 64), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs),
        reinterpret_cast<const uint4 *>(uw), reinterpret_cast<const __half *>(us), counts, offsets, tile_e, tile_m0,
        ntiles, pair_tok, reinterpret_cast<const uint4 *>(bin), reinterpret_cast<__half *>(h));
    moe_down_T_kernel<<<dim3(max_tiles, H / 128), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), counts, offsets, tile_e, tile_m0,
        ntiles, reinterpret_cast<const uint4 *>(h), reinterpret_cast<__half *>(yp));
}

// Small batches (decode): one wave per (pair, output row), like the
// single-token decode kernels. Weight reads are GEMV-shaped (no tile padding);
// tokens sharing an expert re-read its rows, mostly from L2.
__device__ __forceinline__ int pair_expert(const int32_t *offsets, int p) {
    int lo = 0, hi = EXP_L;  // largest e with offsets[e] <= p (skips empty experts)
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] <= p) lo = mid;
        else hi = mid;
    }
    return lo;
}

__global__ __launch_bounds__(256) void moe_gate_up_P_kernel(const uint4 *__restrict__ gw, const __half *__restrict__ gs,
                                                            const uint4 *__restrict__ uw, const __half *__restrict__ us,
                                                            const int32_t *__restrict__ offsets,
                                                            const int32_t *__restrict__ pair_tok,
                                                            const uint4 *__restrict__ bin, __half *__restrict__ h) {
    constexpr int CH = H / 32;
    const int p = blockIdx.y;
    if (p >= offsets[EXP_L]) return;
    const int lane = threadIdx.x % WAVE, row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int e = pair_expert(offsets, p);
    const size_t rbase = size_t(e) * FFN + row;
    const uint4 *g = gw + rbase * CH, *u = uw + rbase * CH;
    const __half *gsc = gs + rbase * (H / QGROUP), *usc = us + rbase * (H / QGROUP);
    const uint4 *x = bin + size_t(pair_tok[p]) * (H / 8);
    float ag = 0.f, au = 0.f;
    for (int c = lane; c < CH; c += WAVE) {
        const uint4 g0 = g[c], u0 = u[c];
        const uint4 *xc = x + c * 4;
        const int grp = c / (QGROUP / 32);
        ag += __half2float(gsc[grp]) * dot_int4x32(g0, xc, 0.f);
        au += __half2float(usc[grp]) * dot_int4x32(u0, xc, 0.f);
    }
    ag = wave_sum(ag);
    au = wave_sum(au);
    if (lane == 0) h[size_t(p) * FFN + row] = __float2half(silu(ag) * au);
}

__global__ __launch_bounds__(256) void moe_down_P_kernel(const uint4 *__restrict__ dw, const __half *__restrict__ ds,
                                                         const int32_t *__restrict__ offsets,
                                                         const uint4 *__restrict__ h, __half *__restrict__ yp) {
    constexpr int CH = FFN / 32;  // 20 uint4 per row
    const int p = blockIdx.y;
    if (p >= offsets[EXP_L]) return;
    const int lane = threadIdx.x % WAVE, row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int e = pair_expert(offsets, p);
    const size_t rbase = size_t(e) * H + row;
    float acc = 0.f;
    if (lane < CH)
        acc = __half2float(ds[rbase * (FFN / QGROUP) + lane / (QGROUP / 32)]) *
              dot_int4x32(dw[rbase * CH + lane], h + size_t(p) * (FFN / 8) + lane * 4, 0.f);
    acc = wave_sum(acc);
    if (lane == 0) yp[size_t(p) * H + row] = __float2half(acc);
}

void moe_experts_P(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                   const uint32_t *dw, const uint16_t *ds, const int32_t *offsets, const int32_t *pair_tok,
                   const uint16_t *bin, int T, uint16_t *h, uint16_t *yp, hipStream_t s) {
    moe_gate_up_P_kernel<<<dim3(FFN / 8, T * TOP_K), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs),
        reinterpret_cast<const uint4 *>(uw), reinterpret_cast<const __half *>(us), offsets, pair_tok,
        reinterpret_cast<const uint4 *>(bin), reinterpret_cast<__half *>(h));
    moe_down_P_kernel<<<dim3(H / 8, T * TOP_K), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), offsets,
        reinterpret_cast<const uint4 *>(h), reinterpret_cast<__half *>(yp));
}

__global__ void moe_combine_T_kernel(const __half *yp, const int32_t *tok_slot, const float *tok_w,
                                     const __half *shared, __half *out) {
    const int t = blockIdx.x;
    for (int n = threadIdx.x; n < H; n += blockDim.x) {
        float acc = __half2float(shared[size_t(t) * H + n]);
        for (int k = 0; k < TOP_K; ++k) {
            const int slot = tok_slot[t * TOP_K + k];
            if (slot >= 0) acc += tok_w[t * TOP_K + k] * __half2float(yp[size_t(slot) * H + n]);
        }
        out[size_t(t) * H + n] = __float2half(acc);
    }
}

void moe_combine_T(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w,
                   const uint16_t *shared, int T, uint16_t *out, hipStream_t s) {
    moe_combine_T_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(yp), tok_slot, tok_w,
                                           reinterpret_cast<const __half *>(shared), reinterpret_cast<__half *>(out));
}

__global__ void swiglu_T_kernel(const float *v, int ldv, int n, int T, __half *h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    const int t = i / n, j = i % n;
    const float *row = v + size_t(t) * ldv;
    h[i] = __float2half(silu(row[j]) * row[n + j] * sigmoid(row[2 * n]));
}

void swiglu_T(const float *v, int ldv, int n, int T, uint16_t *h, hipStream_t s) {
    swiglu_T_kernel<<<(T * n + 255) / 256, 256, 0, s>>>(v, ldv, n, T, reinterpret_cast<__half *>(h));
}

// ------------------------------------------------------------------ PLE
__global__ __launch_bounds__(256) void ple_stats_T_kernel(const __half *kv, const float *X, const float *nk,
                                                          const float *nq, float *send) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    const __half *key = kv + size_t(t) * (XW + SH), *val = key + XW;
    const float *x = X + size_t(t) * XW;
    float *o = send + t * 16;
    for (int s = 0; s < HC; ++s) {
        float sk = 0.f, sx = 0.f, d = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            const int i = s * SH + j;
            const float k = __half2float(key[i]), xv = x[i];
            sk += k * k;
            sx += xv * xv;
            d += k * nk[i] * xv * nq[i];
        }
        sk = block_sum<8>(sk, red);
        sx = block_sum<8>(sx, red);
        d = block_sum<8>(d, red);
        if (threadIdx.x == 0) {
            o[s] = sk;
            o[4 + s] = sx;
            o[8 + s] = d;
        }
    }
    float sv = 0.f;
    for (int j = threadIdx.x; j < SH; j += blockDim.x) {
        const float v = __half2float(val[j]);
        sv += v * v;
    }
    sv = block_sum<8>(sv, red);
    if (threadIdx.x == 0) {
        o[12] = sv;
        o[13] = o[14] = o[15] = 0.f;
    }
}

void ple_stats_T(const uint16_t *kv, const float *X, const float *nk, const float *nq, int T, float *send,
                 hipStream_t s) {
    ple_stats_T_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(kv), X, nk, nq, send);
}

__device__ __forceinline__ float ple_gate(const float *red, int s) {
    const float rk = rsqrtf(red[s] / H + EPS), rq = rsqrtf(red[4 + s] / H + EPS);
    const float gs = red[8 + s] * rk * rq * rsqrtf(float(H));
    const float mag = sqrtf(fmaxf(fabsf(gs), 1e-6f));
    return sigmoid(gs > 0.f ? mag : (gs < 0.f ? -mag : 0.f));
}

__global__ void ple_nc_T_kernel(const __half *kv, const float *red, const float *nc, int T, float *ncbuf) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * XW) return;
    const int t = int(i / XW), c = int(i % XW), s = c / SH, j = c % SH;
    const float *r = red + t * 16;
    const float gate = ple_gate(r, s);
    const float rc = rsqrtf(gate * gate * r[12] / H + EPS);
    ncbuf[i] = gate * __half2float(kv[size_t(t) * (XW + SH) + XW + j]) * rc * nc[c];
}

__global__ void ple_conv_T_kernel(const __half *kv, const float *red, const float *conv_w, const float *ring,
                                  const float *ncbuf, int64_t start, int T, float *X) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * XW) return;
    const int t = int(i / XW), c = int(i % XW), s = c / SH, j = c % SH;
    const int64_t pos = start + t;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < PLE_CONV; ++k) {
        const int64_t q = pos - (PLE_CONV - 1 - k) * PLE_DILATION;
        float v;
        if (q >= start) v = ncbuf[size_t(q - start) * XW + c];
        else if (q >= 0) v = ring[int(q % PLE_RING) * XW + c];
        else v = 0.f;
        acc += conv_w[c * PLE_CONV + k] * v;
    }
    const float gv = ple_gate(red + t * 16, s) * __half2float(kv[size_t(t) * (XW + SH) + XW + j]);
    X[i] += gv + silu(acc);
}

__global__ void ple_ring_T_kernel(const float *ncbuf, float *ring, int64_t start, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= XW) return;
    const int hist = (PLE_CONV - 1) * PLE_DILATION;
    for (int64_t q = std::max<int64_t>(start, start + T - hist); q < start + T; ++q)
        ring[int(q % PLE_RING) * XW + c] = ncbuf[size_t(q - start) * XW + c];
}

void ple_apply_T(const uint16_t *kv, const float *red, const float *nc, const float *conv_w, float *ring,
                 int64_t start, int T, float *ncbuf, float *X, hipStream_t s) {
    const auto *k = reinterpret_cast<const __half *>(kv);
    const unsigned g = unsigned((size_t(T) * XW + 255) / 256);
    ple_nc_T_kernel<<<g, 256, 0, s>>>(k, red, nc, T, ncbuf);
    ple_conv_T_kernel<<<g, 256, 0, s>>>(k, red, conv_w, ring, ncbuf, start, T, X);
    ple_ring_T_kernel<<<XW / 256, 256, 0, s>>>(ncbuf, ring, start, T);
}

__global__ void f16_to_f32_kernel(const __half *x, float *y, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

void f16_to_f32_T(const uint16_t *x, float *y, size_t n, hipStream_t s) {
    f16_to_f32_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(reinterpret_cast<const __half *>(x), y, n);
}

}  // namespace qw::gpu
