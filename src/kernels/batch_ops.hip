#include "kernels/batch_ops.hpp"

#include "kernels/model_common.cuh"
#include "kernels/model_ops.hpp"
#include "kernels/prefill_ops.hpp"

namespace qw::gpu {

namespace {
constexpr int GDN_VL = GDN_V_HEADS / 4;                             // 12
constexpr int GDN_KL = GDN_QK_HEADS / 4;                            // 4
constexpr int GDN_QKV_L = 2 * GDN_KL * GDN_DIM + GDN_VL * GDN_DIM;  // 2560
constexpr int GDN_PROJ_L = GDN_QKV_L + GDN_VL * GDN_DIM + 2 * GDN_VL;  // 4120
constexpr int QSA_PROJ_L = QSA_LOCAL_HEADS * 2 * HEAD_DIM + 2 * HEAD_DIM + (IDX_HEADS + 1) * IDX_DIM;  // 4224
constexpr int XW = HC * SH;
constexpr int ATT_PART = ATT_BLOCKS * QSA_LOCAL_HEADS * ATT_REC;  // per row
constexpr size_t GDN_STATE = size_t(GDN_VL) * GDN_DIM * GDN_DIM;  // per layer, per slot
}  // namespace

// ------------------------------------------------------------------ GEMV
template <int M>
__global__ __launch_bounds__(256) void gemv_rows_kernel(const uint4 *__restrict__ W, const uint4 *__restrict__ x,
                                                        int xs16, float *__restrict__ y, int ys, int N, int K,
                                                        int rpg, int accumulate) {
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + threadIdx.x / WAVE;
    if (row >= N) return;
    const int chunks = K / 8;
    const uint4 *w = W + size_t(row) * chunks;
    if (rpg) x += size_t(row / rpg) * chunks;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    int c = lane;
    for (; c + WAVE < chunks; c += 2 * WAVE) {
        const uint4 w0 = w[c], w1 = w[c + WAVE];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] = dot8(w0, x[size_t(m) * xs16 + c], acc[m]);
            acc[m] = dot8(w1, x[size_t(m) * xs16 + c + WAVE], acc[m]);
        }
    }
    if (c < chunks) {
        const uint4 w0 = w[c];
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = dot8(w0, x[size_t(m) * xs16 + c], acc[m]);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float v = wave_sum(acc[m]);
        if (lane == 0) {
            float *o = y + size_t(m) * ys + row;
            *o = accumulate ? *o + v : v;
        }
    }
}

void gemv_rows(const uint16_t *W, const uint16_t *x, int xs, float *y, int ys, int N, int K, int M, int rpg,
               bool accumulate, hipStream_t s) {
    auto w = reinterpret_cast<const uint4 *>(W);
    auto xv = reinterpret_cast<const uint4 *>(x);
    const dim3 grid((N + 7) / 8), block(256);
    switch (M) {
#define CASE(m) \
    case m: gemv_rows_kernel<m><<<grid, block, 0, s>>>(w, xv, xs / 8, y, ys, N, K, rpg, accumulate); break;
        CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
        CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16)
#undef CASE
        default: break;
    }
}

// ------------------------------------------------------------------ HC
__global__ __launch_bounds__(256) void hc_pre_B_kernel(float *X, const float *pend, const float *inj, const float *w1,
                                                       __half *y, float *send) {
    __shared__ float red[8];
    const int r = blockIdx.x;
    float *x = X + size_t(r) * XW;
    __half *yr = y + size_t(r) * XW;
    for (int s = 0; s < HC; ++s) {
        const float wgt = pend ? 2.f * sigmoid(inj[r * HC + s] / HC) : 0.f;
        float ss = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            const int i = s * SH + j;
            float v = x[i];
            if (pend) {
                v += pend[size_t(r) * SH + j] * wgt;
                x[i] = v;
            }
            yr[i] = __float2half(v * w1[i]);
            ss += v * v;
        }
        ss = block_sum<8>(ss, red);
        if (threadIdx.x == 0) send[size_t(r) * HC_SEND + s] = ss;
    }
}

void hc_pre_B(float *X, const float *pend, const float *inj, const float *w1, uint16_t *y, float *send, int M,
              hipStream_t s) {
    hc_pre_B_kernel<<<M, 256, 0, s>>>(X, pend, inj, w1, reinterpret_cast<__half *>(y), send);
}

__global__ __launch_bounds__(256) void hc_up_mix_B_kernel(const float *red_all, const uint4 *up, const __half *y_all,
                                                          __half *bin_all, float *inj_all) {
    __shared__ __align__(16) __half u[HC_RANK];
    __shared__ float rs[HC];
    const int row = blockIdx.y, t = threadIdx.x;
    const float *red = red_all + size_t(row) * HC_SEND;
    const __half *y = y_all + size_t(row) * XW;
    if (t < HC) rs[t] = rsqrtf(red[t] / H + EPS);
    __syncthreads();
    for (int o = t; o < HC_RANK; o += blockDim.x) {
        float d = 0.f;
#pragma unroll
        for (int s = 0; s < HC; ++s) d += rs[s] * red[HC + s * HC_DOWN + o];
        u[o] = __float2half(silu(d / HC));
    }
    if (inj_all && blockIdx.x == 0 && t < HC) {
        float v = 0.f;
        for (int s = 0; s < HC; ++s) v += rs[s] * red[HC + s * HC_DOWN + HC_RANK + t];
        inj_all[row * HC + t] = v;
    }
    __syncthreads();
    const int lane = t % WAVE, j = blockIdx.x * 8 + t / WAVE;
    const uint4 *uv = reinterpret_cast<const uint4 *>(u);
    constexpr int CH = HC_RANK / 8;
    float out = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        const uint4 *rowp = up + (size_t(j) * HC + s) * CH;
        float acc = dot8(rowp[lane], uv[lane], 0.f);
        if (lane + WAVE < CH) acc = dot8(rowp[lane + WAVE], uv[lane + WAVE], acc);
        const float g = wave_sum(acc);
        out += sigmoid(g) * __half2float(y[s * SH + j]) * rs[s];
    }
    if (lane == 0) bin_all[size_t(row) * SH + j] = __float2half(out / HC);
}

void hc_up_mix_B(const float *red, const uint16_t *up, const uint16_t *y, uint16_t *block_in, float *inj_out, int M,
                 hipStream_t s) {
    hc_up_mix_B_kernel<<<dim3(SH / 8, M), 256, 0, s>>>(red, reinterpret_cast<const uint4 *>(up),
                                                       reinterpret_cast<const __half *>(y),
                                                       reinterpret_cast<__half *>(block_in), inj_out);
}

// ------------------------------------------------------------------ QSA
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
                                                           const int32_t *lists, const int32_t *counts, float *partial) {
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

// ------------------------------------------------------------------ GDN
__global__ __launch_bounds__(256) void gdn_conv_B_kernel(const float *proj_all, const float *conv_w, const float *A_log,
                                                         const float *dt_bias, const SlotPtrs *tab, int gi, Rows rows,
                                                         float *qkv, float *gb) {
    __shared__ float vals[GDN_QKV_L];
    const int r = blockIdx.x;
    const int64_t pos = rows.pos[r], first_pos = rows.pos[rows.run_first[r]];
    const float *ring = tab[rows.slot[r]].ring + size_t(gi) * GDN_RING * GDN_QKV_L;
    for (int ch = threadIdx.x; ch < GDN_QKV_L; ch += blockDim.x) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < GDN_CONV; ++k) {
            const int64_t q = pos - (GDN_CONV - 1 - k);
            float x;
            if (q >= first_pos) x = proj_all[size_t(r - (pos - q)) * GDN_PROJ_L + ch];
            else if (q >= 0) x = ring[int(q % GDN_RING) * GDN_QKV_L + ch];
            else x = 0.f;
            acc += conv_w[ch * GDN_CONV + k] * x;
        }
        vals[ch] = silu(acc);
    }
    __syncthreads();
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    {
        float ss = 0.f;
        for (int i = lane; i < GDN_DIM; i += WAVE) ss += vals[w * GDN_DIM + i] * vals[w * GDN_DIM + i];
        ss = wave_sum(ss);
        const float rn = rsqrtf(ss + EPS) * (w < GDN_KL ? rsqrtf(float(GDN_DIM)) : 1.f);
        for (int i = lane; i < GDN_DIM; i += WAVE) vals[w * GDN_DIM + i] *= rn;
    }
    __syncthreads();
    float *o = qkv + size_t(r) * GDN_QKV_L;
    for (int ch = threadIdx.x; ch < GDN_QKV_L; ch += blockDim.x) o[ch] = vals[ch];
    if (threadIdx.x < GDN_VL) {
        const int h = threadIdx.x;
        const float *row = proj_all + size_t(r) * GDN_PROJ_L;
        const float b = row[GDN_QKV_L + GDN_VL * GDN_DIM + h];
        const float a = row[GDN_QKV_L + GDN_VL * GDN_DIM + GDN_VL + h] + dt_bias[h];
        const float sp = a > 20.f ? a : log1pf(__expf(a));
        gb[(r * GDN_VL + h) * 2] = __expf(-__expf(A_log[h]) * sp);
        gb[(r * GDN_VL + h) * 2 + 1] = sigmoid(b);
    }
}

void gdn_conv_B(const float *proj, const float *conv_w, const float *A_log, const float *dt_bias,
                const SlotPtrs *tab, int gi, Rows rows, float *qkv, float *gb, hipStream_t s) {
    gdn_conv_B_kernel<<<rows.M, 256, 0, s>>>(proj, conv_w, A_log, dt_bias, tab, gi, rows, qkv, gb);
}

__global__ void gdn_ring_B_kernel(const float *proj, const SlotPtrs *tab, int gi, Rows rows) {
    const int r = blockIdx.y, ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= GDN_QKV_L) return;
    float *ring = tab[rows.slot[r]].ring + size_t(gi) * GDN_RING * GDN_QKV_L;
    ring[int(rows.pos[r] % GDN_RING) * GDN_QKV_L + ch] = proj[size_t(r) * GDN_PROJ_L + ch];
}

void gdn_ring_B(const float *proj, const SlotPtrs *tab, int gi, Rows rows, hipStream_t s) {
    gdn_ring_B_kernel<<<dim3(GDN_QKV_L / 256, rows.M), 256, 0, s>>>(proj, tab, gi, rows);
}

// grid (runs, 12 heads x 4 column slices); see gdn_scan_T_kernel.
__global__ __launch_bounds__(256) void gdn_scan_B_kernel(const float *qkv, const float *gb, const SlotPtrs *tab, int gi,
                                                         const int32_t *run_start, const int32_t *run_len,
                                                         const int32_t *run_slot, float *o_raw, float *save) {
    __shared__ float red[8][32], red2[8][32];
    const int run = blockIdx.x;
    const int n = run_len[run];
    if (n == 0) return;
    const int r0 = run_start[run];
    const int h = blockIdx.y / 4, slice = blockIdx.y % 4;
    const int kh = h / (GDN_V_HEADS / GDN_QK_HEADS);
    const int c = threadIdx.x % 32, rg = threadIdx.x / 32;
    const int j = slice * 32 + c;
    float *Sh = tab[run_slot[run]].S + size_t(gi) * GDN_STATE + size_t(h) * GDN_DIM * GDN_DIM;
    float s[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = Sh[(rg * 16 + i) * GDN_DIM + j];
    for (int t = r0; t < r0 + n; ++t) {
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
        if (save) {
            float *sv = save + size_t(t) * GDN_STATE + size_t(h) * GDN_DIM * GDN_DIM;
#pragma unroll
            for (int i = 0; i < 16; ++i) sv[(rg * 16 + i) * GDN_DIM + j] = s[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) Sh[(rg * 16 + i) * GDN_DIM + j] = s[i];
}

void gdn_scan_B(const float *qkv, const float *gb, const SlotPtrs *tab, int gi, const int32_t *run_start,
                const int32_t *run_len, const int32_t *run_slot, int M, float *o_raw, float *save, hipStream_t s) {
    gdn_scan_B_kernel<<<dim3(M, GDN_VL * 4), 256, 0, s>>>(qkv, gb, tab, gi, run_start, run_len, run_slot, o_raw, save);
}

__global__ __launch_bounds__(128) void gdn_norm_B_kernel(const float *o_raw, const float *proj, const float *w,
                                                         __half *out) {
    __shared__ float red[4];
    const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    const size_t i = size_t(r) * GDN_VL * GDN_DIM + h * GDN_DIM + j;
    const float o = o_raw[i];
    const float ss = block_sum<4>(o * o, red);
    const float z = proj[size_t(r) * GDN_PROJ_L + GDN_QKV_L + h * GDN_DIM + j];
    out[i] = __float2half(o * rsqrtf(ss / GDN_DIM + EPS) * w[j] * sigmoid(z));
}

void gdn_norm_B(const float *o_raw, const float *proj, const float *norm_w, int M, uint16_t *out, hipStream_t s) {
    gdn_norm_B_kernel<<<dim3(M, GDN_VL), 128, 0, s>>>(o_raw, proj, norm_w, reinterpret_cast<__half *>(out));
}

// ------------------------------------------------------------------ PLE
__global__ __launch_bounds__(256) void ple_stats_B_kernel(const float *kv_all, const float *X_all, const float *nk,
                                                          const float *nq, float *send_all) {
    __shared__ float red[8];
    const int r = blockIdx.x;
    const float *key = kv_all + size_t(r) * (XW + SH), *val = key + XW;
    const float *x = X_all + size_t(r) * XW;
    float *o = send_all + size_t(r) * 16;
    for (int s = 0; s < HC; ++s) {
        float sk = 0.f, sx = 0.f, d = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            const int i = s * SH + j;
            const float k = key[i], xv = x[i];
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
    for (int j = threadIdx.x; j < SH; j += blockDim.x) sv += val[j] * val[j];
    sv = block_sum<8>(sv, red);
    if (threadIdx.x == 0) {
        o[12] = sv;
        o[13] = o[14] = o[15] = 0.f;
    }
}

void ple_stats_B(const float *kv, const float *X, const float *nk, const float *nq, int M, float *send, hipStream_t s) {
    ple_stats_B_kernel<<<M, 256, 0, s>>>(kv, X, nk, nq, send);
}

__device__ __forceinline__ float ple_gate_B(const float *red, int s) {
    const float rk = rsqrtf(red[s] / H + EPS), rq = rsqrtf(red[4 + s] / H + EPS);
    const float gs = red[8 + s] * rk * rq * rsqrtf(float(H));
    const float mag = sqrtf(fmaxf(fabsf(gs), 1e-6f));
    return sigmoid(gs > 0.f ? mag : (gs < 0.f ? -mag : 0.f));
}

__global__ void ple_nc_B_kernel(const float *kv, const float *red, const float *nc, float *ncbuf) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= XW) return;
    const float *rr = red + size_t(r) * 16;
    const float gate = ple_gate_B(rr, i / SH);
    const float rc = rsqrtf(gate * gate * rr[12] / H + EPS);
    ncbuf[size_t(r) * XW + i] = gate * kv[size_t(r) * (XW + SH) + XW + i % SH] * rc * nc[i];
}

__global__ void ple_conv_B_kernel(const float *kv, const float *red, const float *conv_w, const SlotPtrs *tab,
                                  Rows rows, const float *ncbuf, float *X) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= XW) return;
    const int64_t pos = rows.pos[r], first_pos = rows.pos[rows.run_first[r]];
    const float *ring = tab[rows.slot[r]].ple;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < PLE_CONV; ++k) {
        const int64_t q = pos - (PLE_CONV - 1 - k) * PLE_DILATION;
        float v;
        if (q >= first_pos) v = ncbuf[size_t(r - (pos - q)) * XW + i];
        else if (q >= 0) v = ring[int(q % PLE_RING) * XW + i];
        else v = 0.f;
        acc += conv_w[i * PLE_CONV + k] * v;
    }
    const float gv = ple_gate_B(red + size_t(r) * 16, i / SH) * kv[size_t(r) * (XW + SH) + XW + i % SH];
    X[size_t(r) * XW + i] += gv + silu(acc);
}

__global__ void ple_ring_B_kernel(const float *ncbuf, const SlotPtrs *tab, Rows rows) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= XW) return;
    tab[rows.slot[r]].ple[int(rows.pos[r] % PLE_RING) * XW + i] = ncbuf[size_t(r) * XW + i];
}

void ple_apply_B(const float *kv, const float *red, const float *nc, const float *conv_w, const SlotPtrs *tab,
                 Rows rows, float *ncbuf, float *X, hipStream_t s) {
    const dim3 grid(XW / 256, rows.M);
    ple_nc_B_kernel<<<grid, 256, 0, s>>>(kv, red, nc, ncbuf);
    ple_conv_B_kernel<<<grid, 256, 0, s>>>(kv, red, conv_w, tab, rows, ncbuf, X);
    ple_ring_B_kernel<<<grid, 256, 0, s>>>(ncbuf, tab, rows);
}

// ------------------------------------------------------------------ MoE
__global__ void moe_combine_B_kernel(const __half *yp, const int32_t *tok_slot, const float *tok_w,
                                     const float *shared, float *out) {
    const int r = blockIdx.x;
    for (int n = threadIdx.x; n < H; n += blockDim.x) {
        float acc = shared[size_t(r) * H + n];
        for (int k = 0; k < TOP_K; ++k) {
            const int slot = tok_slot[r * TOP_K + k];
            if (slot >= 0) acc += tok_w[r * TOP_K + k] * __half2float(yp[size_t(slot) * H + n]);
        }
        out[size_t(r) * H + n] = acc;
    }
}

void moe_combine_B(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w, const float *shared, int M,
                   float *out, hipStream_t s) {
    moe_combine_B_kernel<<<M, 256, 0, s>>>(reinterpret_cast<const __half *>(yp), tok_slot, tok_w, shared, out);
}

__global__ __launch_bounds__(1024) void row_lse_kernel(const float *__restrict__ logits, int n,
                                                       float *__restrict__ out) {
    __shared__ float red[32];
    const float *x = logits + size_t(blockIdx.x) * n;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, x[i]);
    for (int o = WAVE / 2; o > 0; o /= 2) mx = fmaxf(mx, __shfl_xor(mx, o));
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    if (lane == 0) red[w] = mx;
    __syncthreads();
    mx = red[0];
    for (int i = 1; i < 32; ++i) mx = fmaxf(mx, red[i]);
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) z += __expf(x[i] - mx);
    z = block_sum<32>(z, red);
    if (threadIdx.x == 0) {
        out[2 * blockIdx.x] = mx;
        out[2 * blockIdx.x + 1] = z;
    }
}

void row_lse(const float *logits, int n, int M, float *out, hipStream_t s) {
    row_lse_kernel<<<M, 1024, 0, s>>>(logits, n, out);
}

// ------------------------------------------------------------------ MTP
__global__ __launch_bounds__(256) void mtp_ssq_kernel(const float *X, const float *emb, float *send) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    float a = 0.f, b = 0.f;
    for (int i = threadIdx.x; i < XW; i += blockDim.x) a += X[size_t(m) * XW + i] * X[size_t(m) * XW + i];
    for (int i = threadIdx.x; i < SH; i += blockDim.x) b += emb[size_t(m) * SH + i] * emb[size_t(m) * SH + i];
    a = block_sum<8>(a, red);
    b = block_sum<8>(b, red);
    if (threadIdx.x == 0) {
        send[4 * m] = a;
        send[4 * m + 1] = b;
        send[4 * m + 2] = 0.f;
        send[4 * m + 3] = 0.f;
    }
}

void mtp_ssq(const float *X, const float *emb, int rows, float *send, hipStream_t s) {
    mtp_ssq_kernel<<<rows, 256, 0, s>>>(X, emb, send);
}

__global__ __launch_bounds__(256) void mtp_norm_kernel(const float *X, const float *emb, const float *red,
                                                       const float *nh, const float *ne, int rows, __half *xn) {
    const int m = blockIdx.x;
    const float rh = rsqrtf(red[4 * m] / HCH + EPS), re = rsqrtf(red[4 * m + 1] / H + EPS);
    for (int i = threadIdx.x; i < XW; i += blockDim.x) {
        const int st = i / SH, j = i % SH;
        xn[(size_t(st) * rows + m) * SH + j] = __float2half(X[size_t(m) * XW + i] * rh * nh[i]);
    }
    for (int j = threadIdx.x; j < SH; j += blockDim.x)
        xn[(size_t(HC) * rows + m) * SH + j] = __float2half(emb[size_t(m) * SH + j] * re * ne[j]);
}

void mtp_norm(const float *X, const float *emb, const float *red, const float *nh, const float *ne, int rows,
              uint16_t *xn, hipStream_t s) {
    mtp_norm_kernel<<<rows, 256, 0, s>>>(X, emb, red, nh, ne, rows, reinterpret_cast<__half *>(xn));
}

__global__ __launch_bounds__(256) void mtp_add_kernel(const float *out, int rows, float *X) {
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < XW; i += blockDim.x) {
        const int st = i / SH, j = i % SH;
        X[size_t(m) * XW + i] = out[(size_t(st) * rows + m) * SH + j] + out[(size_t(HC) * rows + m) * SH + j];
    }
}

void mtp_add(const float *out, int rows, float *X, hipStream_t s) { mtp_add_kernel<<<rows, 256, 0, s>>>(out, rows, X); }

__global__ __launch_bounds__(256) void gather_rows_kernel(const float *const *src, int n, float *dst) {
    const float *p = src[blockIdx.x];
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[size_t(blockIdx.x) * n + i] = p[i];
}

void gather_rows(const float *const *src, int rows, int n, float *dst, hipStream_t s) {
    gather_rows_kernel<<<rows, 256, 0, s>>>(src, n, dst);
}

__global__ __launch_bounds__(256) void save_last_rows_kernel(const float *X, const SlotPtrs *tab,
                                                             const int32_t *run_start, const int32_t *run_len,
                                                             const int32_t *run_slot) {
    const int run = blockIdx.x;
    if (run_len[run] == 0) return;
    const float *src = X + size_t(run_start[run] + run_len[run] - 1) * XW;
    float *dst = tab[run_slot[run]].mtp_pend;
    if (!dst) return;
    for (int i = threadIdx.x; i < XW; i += blockDim.x) dst[i] = src[i];
}

void save_last_rows(const float *X, const SlotPtrs *tab, const int32_t *run_start, const int32_t *run_len,
                    const int32_t *run_slot, int M, hipStream_t s) {
    save_last_rows_kernel<<<M, 256, 0, s>>>(X, tab, run_start, run_len, run_slot);
}

__global__ __launch_bounds__(1024) void row_argmax_kernel(const float *logits, int n, float *out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float *x = logits + size_t(blockIdx.x) * n;
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) {  // strided: i increases, so the first max of this thread wins
            v = x[i];
            idx = i;
        }
    for (int o = WAVE / 2; o > 0; o /= 2) {
        const float v2 = __shfl_xor(v, o);
        const int i2 = __shfl_xor(idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) {
            v = v2;
            idx = i2;
        }
    }
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    if (lane == 0) {
        bv[w] = v;
        bi[w] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < int(blockDim.x) / WAVE; ++k)
            if (bv[k] > v || (bv[k] == v && bi[k] < idx)) {
                v = bv[k];
                idx = bi[k];
            }
        out[2 * blockIdx.x] = v;
        out[2 * blockIdx.x + 1] = __int_as_float(idx);
    }
}

void row_argmax(const float *logits, int n, int M, float *out, hipStream_t s) {
    row_argmax_kernel<<<M, 1024, 0, s>>>(logits, n, out);
}

__global__ __launch_bounds__(256) void mtp_shift_kernel(const float *pend, const float *X, int o, float *Xm) {
    const int t = blockIdx.x;
    const float *src = t < o ? pend : X + size_t(t - o) * XW;
    for (int i = threadIdx.x; i < XW; i += blockDim.x) Xm[size_t(t) * XW + i] = src[i];
}

void mtp_shift(const float *pend, const float *X, int o, int rows, float *Xm, hipStream_t s) {
    if (rows > 0) mtp_shift_kernel<<<rows, 256, 0, s>>>(pend, X, o, Xm);
}

}  // namespace qw::gpu
