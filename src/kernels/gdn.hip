#include "kernels/gdn.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

// ---- prefill
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
            if (q >= start)
                x = __half2float(P[size_t(q - start) * GDN_PROJ_L + ch]);
            else if (q >= 0)
                x = ring[int(q % GDN_RING) * GDN_QKV_L + ch];
            else
                x = 0.f;
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

void gdn_conv_T(const uint16_t *P, const float *ring, const float *conv_w, const float *A_log, const float *dt_bias,
                int64_t start, int T, float *qkv, float *gb, hipStream_t s) {
    gdn_conv_T_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(P), ring, conv_w, A_log, dt_bias, start, qkv,
                                        gb);
}

__global__ void gdn_ring_update_kernel(const __half *P, float *ring, int64_t start, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= GDN_QKV_L) return;
    for (int64_t q = std::max<int64_t>(start, start + T - (GDN_CONV - 1)); q < start + T; ++q)
        ring[int(q % GDN_RING) * GDN_QKV_L + ch] = __half2float(P[size_t(q - start) * GDN_PROJ_L + ch]);
}

// The whole ring as it stands at position c (start < c <= start + T).
__global__ void gdn_ring_capture_kernel(const __half *P, const float *ring, int64_t start, int64_t c, float *dst) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= GDN_QKV_L) return;
    for (int64_t q = c - GDN_RING; q < c; ++q) {
        const int cell = int((q + GDN_RING) % GDN_RING);
        float x = 0.f;
        if (q >= start)
            x = __half2float(P[size_t(q - start) * GDN_PROJ_L + ch]);
        else if (q >= 0)
            x = ring[cell * GDN_QKV_L + ch];
        dst[cell * GDN_QKV_L + ch] = x;
    }
}

void gdn_ring_update_T(const uint16_t *P, float *ring, int64_t start, int T, hipStream_t s, const Captures &cap) {
    for (int i = 0; i < cap.n; ++i)
        if (cap.pos[i] > start && cap.pos[i] <= start + T)
            gdn_ring_capture_kernel<<<GDN_QKV_L / 256, 256, 0, s>>>(reinterpret_cast<const __half *>(P), ring, start,
                                                                     cap.pos[i], cap.dst[i]);
    gdn_ring_update_kernel<<<GDN_QKV_L / 256, 256, 0, s>>>(reinterpret_cast<const __half *>(P), ring, start, T);
}

// grid 48 = 12 heads x 4 slices of 32 v-columns. Thread (rg, c): column
// j = slice*32 + c, k-rows rg*16 .. +16. Columns of the delta-rule state are
// independent, so slices need no communication.
__global__ __launch_bounds__(256) void gdn_scan_T_kernel(const float *qkv, const float *gb, float *S, int T,
                                                         float *o_raw, int64_t start, Captures cap) {
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
        for (int i = 0; i < cap.n; ++i)
            if (cap.pos[i] == start + t + 1) {
                float *D = cap.dst[i] + size_t(h) * GDN_DIM * GDN_DIM;
#pragma unroll
                for (int k = 0; k < 16; ++k) D[(rg * 16 + k) * GDN_DIM + j] = s[k];
            }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) Sh[(rg * 16 + i) * GDN_DIM + j] = s[i];
}

void gdn_scan_T(const float *qkv, const float *gb, float *S, int T, float *o_raw, hipStream_t s, int64_t start,
                const Captures &cap) {
    gdn_scan_T_kernel<<<GDN_VL * 4, 256, 0, s>>>(qkv, gb, S, T, o_raw, start, cap);
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

void gdn_norm_T(const float *o_raw, const uint16_t *P, const float *norm_w, int T, uint16_t *out, hipStream_t s) {
    gdn_norm_T_kernel<<<dim3(T, GDN_VL), 128, 0, s>>>(o_raw, reinterpret_cast<const __half *>(P), norm_w,
                                                      reinterpret_cast<__half *>(out));
}

// ---- batched decode rows
// One block of GDN_DIM threads per (row, 128-channel group): the causal conv
// over the ring and the batch's earlier rows, silu, then the L2 norm of the q
// and k groups (the first 2 * GDN_KL). Group 0 also computes decay / beta.
__global__ __launch_bounds__(GDN_DIM) void gdn_conv_B_kernel(const float *proj_all, const float *conv_w,
                                                             const float *A_log, const float *dt_bias,
                                                             const SlotPtrs *tab, int gi, Rows rows, float *qkv,
                                                             float *gb) {
    __shared__ float red[GDN_DIM / WAVE];
    const int r = blockIdx.x, grp = blockIdx.y, j = threadIdx.x;
    const int ch = grp * GDN_DIM + j;
    const int64_t pos = rows.pos[r], first_pos = rows.pos[rows.run_first[r]];
    const float *ring = tab[rows.slot[r]].ring + size_t(gi) * GDN_RING * GDN_QKV_L;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < GDN_CONV; ++k) {
        const int64_t q = pos - (GDN_CONV - 1 - k);
        float x;
        if (q >= first_pos) {
            QW_DCHECK(r - (pos - q) >= 0 && r - (pos - q) <= r);
            x = proj_all[size_t(r - (pos - q)) * GDN_PROJ_L + ch];
        } else if (q >= 0)
            x = ring[int(q % GDN_RING) * GDN_QKV_L + ch];
        else
            x = 0.f;
        acc += conv_w[ch * GDN_CONV + k] * x;
    }
    float v = silu(acc);
    if (grp < 2 * GDN_KL) {
        const float ss = block_sum<GDN_DIM / WAVE>(v * v, red);
        v *= rsqrtf(ss + EPS) * (grp < GDN_KL ? rsqrtf(float(GDN_DIM)) : 1.f);
    }
    qkv[size_t(r) * GDN_QKV_L + ch] = v;
    if (grp == 0 && j < GDN_VL) {
        const int h = j;
        const float *row = proj_all + size_t(r) * GDN_PROJ_L;
        const float b = row[GDN_QKV_L + GDN_VL * GDN_DIM + h];
        const float a = row[GDN_QKV_L + GDN_VL * GDN_DIM + GDN_VL + h] + dt_bias[h];
        const float sp = a > 20.f ? a : log1pf(__expf(a));
        gb[(r * GDN_VL + h) * 2] = __expf(-__expf(A_log[h]) * sp);
        gb[(r * GDN_VL + h) * 2 + 1] = sigmoid(b);
    }
}

void gdn_conv_B(const float *proj, const float *conv_w, const float *A_log, const float *dt_bias, const SlotPtrs *tab,
                int gi, Rows rows, float *qkv, float *gb, hipStream_t s) {
    gdn_conv_B_kernel<<<dim3(rows.M, GDN_QKV_L / GDN_DIM), GDN_DIM, 0, s>>>(proj, conv_w, A_log, dt_bias, tab, gi, rows,
                                                                            qkv, gb);
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
    QW_DCHECK(n > 0 && r0 >= 0 && r0 + n <= int(gridDim.x) && run_slot[run] >= 0 && run_slot[run] < 8);
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

}  // namespace qw::gpu
