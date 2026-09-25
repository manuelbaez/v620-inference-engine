#include "kernels/qsa.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

// ---- prefill
// One block per token.
__global__ __launch_bounds__(256) void qsa_prep_T_kernel(const __half *P, const float *qn, const float *kn,
                                                         const float *iqn, int64_t start, __half *q16, __half *gate,
                                                         __half *iq16, __half *K, __half *V, float *raw_k) {
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

void qsa_prep_T(const uint16_t *P, const float *qn, const float *kn, const float *iqn, int64_t start, int T,
                uint16_t *q16, uint16_t *gate, uint16_t *iq16, uint16_t *K, uint16_t *V, float *raw_k, hipStream_t s) {
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

void qsa_compress_T(const float *raw_k, const float *ikn, int64_t start, int T, uint16_t *ck, hipStream_t s) {
    const int64_t c_first = start / IDX_RATIO;                   // first group whose last token is >= start
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
    qsa_score_reduce_kernel<<<dim3((nb_max + 255) / 256, Q), 256, 0, s>>>(reinterpret_cast<const __half *>(sc16), ldsc,
                                                                          q0, scores);
}

__global__ __launch_bounds__(1024) void qsa_select_T_kernel(const float *scores, int ldsc, int64_t q0, int32_t *lists,
                                                            int32_t *counts) {
    const int q = blockIdx.x;
    select_body(scores + size_t(q) * ldsc, q0 + q, lists + size_t(q) * LIST_W, counts + q);
}

void qsa_select_T(const float *scores, int ldsc, int64_t q0, int Q, int32_t *lists, int32_t *counts, hipStream_t s) {
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

void qsa_attend_T(const uint16_t *q16, const uint16_t *gate, const uint16_t *K, const uint16_t *V, const int32_t *lists,
                  const int32_t *counts, int Q, float *partial, uint16_t *out, hipStream_t s) {
    qsa_attend_T_kernel<<<dim3(ATT_BLOCKS, Q), 256, 0, s>>>(reinterpret_cast<const uint4 *>(q16),
                                                            reinterpret_cast<const uint4 *>(K),
                                                            reinterpret_cast<const uint4 *>(V), lists, counts, partial);
    qsa_combine_T_kernel<<<Q, HEAD_DIM, 0, s>>>(partial, reinterpret_cast<const __half *>(gate),
                                                reinterpret_cast<__half *>(out));
}

// ---- batched decode rows
__global__ __launch_bounds__(256) void qsa_prep_B_kernel(const float *proj_all, const float *qn, const float *kn,
                                                         const float *iqn, const SlotPtrs *tab, int qi, Rows rows,
                                                         __half *q16, float *gate, float *iq) {
    __shared__ float buf[HEAD_DIM], red[8];
    const int r = blockIdx.x, t = threadIdx.x;
    const SlotPtrs &sp = tab[rows.slot[r]];
    const int64_t pos = rows.pos[r];
    const float *proj = proj_all + size_t(r) * QSA_PROJ_L;
    const float *kp = proj + QSA_LOCAL_HEADS * 2 * HEAD_DIM, *vp = kp + HEAD_DIM, *ip = vp + HEAD_DIM;
    for (int h = 0; h < QSA_LOCAL_HEADS; ++h) {
        const float *src = proj + h * 2 * HEAD_DIM;
        norm_shared(src, qn, buf, HEAD_DIM, red);
        rope_shared(buf, pos);
        q16[(size_t(r) * QSA_LOCAL_HEADS + h) * HEAD_DIM + t] = __float2half(buf[t]);
        gate[(size_t(r) * QSA_LOCAL_HEADS + h) * HEAD_DIM + t] = src[HEAD_DIM + t];
        __syncthreads();
    }
    norm_shared(kp, kn, buf, HEAD_DIM, red);
    rope_shared(buf, pos);
    sp.K[qi][pos * HEAD_DIM + t] = __float2half(buf[t]);
    sp.V[qi][pos * HEAD_DIM + t] = __float2half(vp[t]);
    __syncthreads();
    for (int h = 0; h < IDX_HEADS; ++h) {
        norm_shared(ip + h * IDX_DIM, iqn, buf, IDX_DIM, red);
        rope_shared(buf, pos);
        if (t < IDX_DIM) iq[(size_t(r) * IDX_HEADS + h) * IDX_DIM + t] = buf[t];
        __syncthreads();
    }
    if (t < IDX_DIM) sp.raw_k[qi][pos * IDX_DIM + t] = ip[IDX_HEADS * IDX_DIM + t];
}

void qsa_prep_B(const float *proj, const float *qn, const float *kn, const float *iqn, const SlotPtrs *tab, int qi,
                Rows rows, uint16_t *q16, float *gate, float *iq, hipStream_t s) {
    qsa_prep_B_kernel<<<rows.M, 256, 0, s>>>(proj, qn, kn, iqn, tab, qi, rows, reinterpret_cast<__half *>(q16), gate,
                                             iq);
}

__global__ __launch_bounds__(128) void qsa_compress_B_kernel(const float *ikn, const SlotPtrs *tab, int qi, Rows rows) {
    __shared__ float mean[IDX_DIM], buf[IDX_DIM], red[4];
    const int r = blockIdx.x, t = threadIdx.x;
    const int64_t pos = rows.pos[r];
    if ((pos + 1) % IDX_RATIO != 0) return;
    const SlotPtrs &sp = tab[rows.slot[r]];
    float m = 0.f;
    for (int k = 0; k < IDX_RATIO; ++k) m += sp.raw_k[qi][(pos - k) * IDX_DIM + t];
    mean[t] = m / IDX_RATIO;
    __syncthreads();
    const float ss = block_sum<4>(mean[t] * mean[t], red);
    buf[t] = mean[t] * rsqrtf(ss / IDX_DIM + EPS) * ikn[t];
    rope_shared(buf, pos + 1 - IDX_RATIO);
    sp.ck[qi][(pos / IDX_RATIO) * IDX_DIM + t] = __float2half(buf[t]);
}

void qsa_compress_B(const float *ikn, const SlotPtrs *tab, int qi, Rows rows, hipStream_t s) {
    qsa_compress_B_kernel<<<rows.M, 128, 0, s>>>(ikn, tab, qi, rows);
}

constexpr int SCORE_BLOCKS_B = 128;
__global__ __launch_bounds__(256) void qsa_scores_B_kernel(const float *iq_all, const SlotPtrs *tab, int qi, Rows rows,
                                                           float *scores_all, int ld) {
    const int r = blockIdx.y;
    const int nb = int((rows.pos[r] + 1) / IDX_RATIO);
    if (nb <= IDX_TOPK_BLOCKS) return;
    const uint4 *ck = reinterpret_cast<const uint4 *>(tab[rows.slot[r]].ck[qi]);
    const float *iq = iq_all + size_t(r) * IDX_HEADS * IDX_DIM;
    float *scores = scores_all + size_t(r) * ld;
    const int lane16 = threadIdx.x % 16;
    for (int c0 = blockIdx.x * 16; c0 < nb; c0 += SCORE_BLOCKS_B * 16) {
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

__global__ __launch_bounds__(1024) void qsa_select_B_kernel(const float *scores, int ld, Rows rows, int32_t *lists,
                                                            int32_t *counts) {
    const int r = blockIdx.x;
    select_body(scores + size_t(r) * ld, rows.pos[r], lists + size_t(r) * LIST_W, counts + r);
}

__global__ __launch_bounds__(256) void qsa_attend_B_kernel(const uint4 *q16, const SlotPtrs *tab, int qi, Rows rows,
                                                           const int32_t *lists, const int32_t *counts,
                                                           float *partial) {
    const int r = blockIdx.y;
    const SlotPtrs &sp = tab[rows.slot[r]];
    attend_body(q16 + size_t(r) * QSA_LOCAL_HEADS * (HEAD_DIM / 8), reinterpret_cast<const uint4 *>(sp.K[qi]),
                reinterpret_cast<const uint4 *>(sp.V[qi]), lists + size_t(r) * LIST_W, counts[r],
                partial + size_t(r) * ATT_PART + size_t(blockIdx.x) * QSA_LOCAL_HEADS * ATT_REC);
}

__global__ __launch_bounds__(256) void qsa_combine_B_kernel(const float *partial, const float *gate, __half *out) {
    const int r = blockIdx.x;
    combine_body(partial + size_t(r) * ATT_PART, gate + size_t(r) * QSA_LOCAL_HEADS * HEAD_DIM,
                 out + size_t(r) * QSA_LOCAL_HEADS * HEAD_DIM);
}

void qsa_attend_B(const uint16_t *q16, const float *gate, const float *iq, const SlotPtrs *tab, int qi, Rows rows,
                  float *scores, int ld, int32_t *lists, int32_t *counts, float *partial, uint16_t *out,
                  hipStream_t s) {
    qsa_scores_B_kernel<<<dim3(SCORE_BLOCKS_B, rows.M), 256, 0, s>>>(iq, tab, qi, rows, scores, ld);
    qsa_select_B_kernel<<<rows.M, 1024, 0, s>>>(scores, ld, rows, lists, counts);
    qsa_attend_B_kernel<<<dim3(ATT_BLOCKS, rows.M), 256, 0, s>>>(reinterpret_cast<const uint4 *>(q16), tab, qi, rows,
                                                                 lists, counts, partial);
    qsa_combine_B_kernel<<<rows.M, HEAD_DIM, 0, s>>>(partial, gate, reinterpret_cast<__half *>(out));
}

}  // namespace qw::gpu
