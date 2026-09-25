#include "kernels/hc.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

// ---- prefill (two-phase: ssq all-reduce, then the full down projection)
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
    if (o < HC_RANK)
        u[t * HC_RANK + o] = __float2half(silu(v / HC));
    else if (o < HC_RANK + HC && inj)
        inj[t * HC + o - HC_RANK] = v;
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
    hc_mix_T_kernel<<<(T * SH + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<const __half *>(g), reinterpret_cast<const __half *>(xn), T, reinterpret_cast<__half *>(bin));
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

// ---- batched decode rows (one phase)
// One block per (row, stream).
__global__ __launch_bounds__(256) void hc_pre_B_kernel(float *X, const float *pend, const float *inj, const float *w1,
                                                       __half *y, float *send) {
    __shared__ float red[8];
    const int r = blockIdx.x, st = blockIdx.y;
    float *x = X + size_t(r) * XW + st * SH;
    __half *yr = y + size_t(r) * XW + st * SH;
    const float *w = w1 + st * SH;
    const float wgt = pend ? 2.f * sigmoid(inj[r * HC + st] / HC) : 0.f;
    float ss = 0.f;
    for (int j = threadIdx.x; j < SH; j += blockDim.x) {
        float v = x[j];
        if (pend) {
            v += pend[size_t(r) * SH + j] * wgt;
            x[j] = v;
        }
        yr[j] = __float2half(v * w[j]);
        ss += v * v;
    }
    ss = block_sum<8>(ss, red);
    if (threadIdx.x == 0) send[size_t(r) * HC_SEND + st] = ss;
}

void hc_pre_B(float *X, const float *pend, const float *inj, const float *w1, uint16_t *y, float *send, int M,
              hipStream_t s) {
    hc_pre_B_kernel<<<dim3(M, HC), 256, 0, s>>>(X, pend, inj, w1, reinterpret_cast<__half *>(y), send);
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

}  // namespace qw::gpu
