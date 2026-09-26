#include "kernels/mtp.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

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

void mtp_add(const float *out, int rows, float *X, hipStream_t s) {
    mtp_add_kernel<<<rows, 256, 0, s>>>(out, rows, X);
}

__global__ __launch_bounds__(256) void gather_rows_kernel(const float *const *src, int n, float *dst) {
    const float *p = src[blockIdx.x];
    QW_DCHECK(p != nullptr);
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

__global__ __launch_bounds__(256) void save_rows_hist_kernel(const float *X, const SlotPtrs *tab, Rows rows) {
    const int r = blockIdx.x;
    float *dst = tab[rows.slot[r]].mtp_hist;
    if (!dst) return;
    dst += size_t(rows.pos[r] % MTP_HIST) * XW;
    const float *src = X + size_t(r) * XW;
    for (int i = threadIdx.x; i < XW; i += blockDim.x) dst[i] = src[i];
}

void save_rows_hist(const float *X, const SlotPtrs *tab, Rows rows, hipStream_t s) {
    save_rows_hist_kernel<<<rows.M, 256, 0, s>>>(X, tab, rows);
}

__global__ __launch_bounds__(256) void mtp_shift_kernel(const float *pend, const float *X, int o, float *Xm) {
    const int t = blockIdx.x;
    const float *src = t < o ? pend : X + size_t(t - o) * XW;
    for (int i = threadIdx.x; i < XW; i += blockDim.x) Xm[size_t(t) * XW + i] = src[i];
}

void mtp_shift(const float *pend, const float *X, int o, int rows, float *Xm, hipStream_t s) {
    if (rows > 0) mtp_shift_kernel<<<rows, 256, 0, s>>>(pend, X, o, Xm);
}

namespace {
__global__ __launch_bounds__(256) void draft_pick_kernel(const float *all, int stride, const int32_t *last,
                                                          int32_t *tokens, const uint16_t *embed, int rank, float *emb) {
    const int q = blockIdx.x;
    __shared__ int32_t tok;
    if (threadIdx.x == 0) {
        const int row = last[q];
        float best = -INFINITY;
        int32_t t = 0;
        for (int r = 0; r < RANKS; ++r) {
            const float v = all[size_t(r) * stride + size_t(row) * 2];
            const int32_t idx = __float_as_int(all[size_t(r) * stride + size_t(row) * 2 + 1]);
            if (v > best) {  // ranks in order: on a tie the lower rank's (lower) id stays
                best = v;
                t = r * VOCAB_L + idx;
            }
        }
        tok = t;
        tokens[q] = t;
    }
    __syncthreads();
    if (!emb) return;
    const uint16_t *e = embed + size_t(tok) * H + size_t(rank) * SH;
    for (int j = threadIdx.x; j < SH; j += blockDim.x) emb[size_t(q) * SH + j] = __uint_as_float(uint32_t(e[j]) << 16);
}
}  // namespace

void draft_pick(const float *all, int stride, const int32_t *last, int nreq, int rank, int32_t *tokens,
                const uint16_t *embed, float *emb, hipStream_t s) {
    draft_pick_kernel<<<nreq, 256, 0, s>>>(all, stride, last, tokens, embed, rank, emb);
}

}  // namespace qw::gpu
