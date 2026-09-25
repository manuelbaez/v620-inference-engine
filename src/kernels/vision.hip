#include "kernels/vision.hpp"

#include <hip/hip_fp16.h>

#include <cmath>

#include "kernels/common.cuh"

namespace qw::gpu {

namespace {

__device__ inline float h2f(uint16_t v) {
    return __half2float(__ushort_as_half(v));
}
__device__ inline uint16_t f2h(float v) {
    return __half_as_ushort(__float2half(v));
}

template <int N>
__device__ inline float block_max(float v, float *red) {
    v = wave_max(v);
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = red[0];
#pragma unroll
    for (int i = 1; i < N; ++i) t = fmaxf(t, red[i]);
    return t;
}

__global__ __launch_bounds__(256) void layernorm_kernel(const float *x, const float *w, const float *b, int D,
                                                        uint16_t *y) {
    __shared__ float red[8];
    const float *row = x + size_t(blockIdx.x) * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += row[i];
    const float mean = block_sum<8>(s, red) / float(D);
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) v += (row[i] - mean) * (row[i] - mean);
    const float rstd = rsqrtf(block_sum<8>(v, red) / float(D) + 1e-6f);
    uint16_t *out = y + size_t(blockIdx.x) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = f2h((row[i] - mean) * rstd * w[i] + b[i]);
}

__global__ void residual_add_kernel(float *x, const uint16_t *y, int ldy, const float *bias, int N, int D) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(N) * D) return;
    const int n = int(i / D), d = int(i % D);
    x[i] += h2f(y[size_t(n) * ldy + d]) + bias[d];
}

__global__ void bias_gelu_kernel(uint16_t *y, const float *bias, int N, int D, bool tanh_approx) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(N) * D) return;
    const float v = h2f(y[i]) + bias[i % D];
    float g;
    if (tanh_approx)
        g = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    else
        g = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
    y[i] = f2h(g);
}

__global__ void bias_kernel(uint16_t *y, const float *bias, int N, int D) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < size_t(N) * D) y[i] = f2h(h2f(y[i]) + bias[i % D]);
}

__global__ void embed_kernel(float *x, const uint16_t *y, const float *bias, const float *table, const int32_t *idx,
                             const float *wts, int N, int D) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(N) * D) return;
    const int n = int(i / D), d = int(i % D);
    float p = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) p += wts[n * 4 + k] * table[size_t(idx[n * 4 + k]) * D + d];
    x[i] = h2f(y[i]) + bias[d] + p;
}

// one block per patch, 64 threads: thread t < 36 rotates the pair (t, t + 36) of q and of k for every head
__global__ __launch_bounds__(64) void rope2d_kernel(uint16_t *qkv, const int32_t *pos_hw, int heads) {
    constexpr int HD = 72, HALF = 36, NF = 18;
    const int n = blockIdx.x, t = threadIdx.x;
    if (t >= HALF) return;
    const int f = t % NF;
    const float inv = powf(10000.f, -2.f * float(f) / float(HALF));
    const float pos = float(t < NF ? pos_hw[n * 2] : pos_hw[n * 2 + 1]);
    float sn, cs;
    sincosf(pos * inv, &sn, &cs);
    uint16_t *row = qkv + size_t(n) * 3 * heads * HD;
    for (int part = 0; part < 2; ++part)  // q, k
        for (int h = 0; h < heads; ++h) {
            uint16_t *v = row + (size_t(part) * heads + h) * HD;
            const float a = h2f(v[t]), b = h2f(v[t + HALF]);
            v[t] = f2h(a * cs - b * sn);
            v[t + HALF] = f2h(b * cs + a * sn);
        }
}

__global__ __launch_bounds__(256) void softmax_kernel(uint16_t *s, int n, int ld) {
    __shared__ float red[8];
    uint16_t *row = s + size_t(blockIdx.x) * ld;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, h2f(row[i]));
    mx = block_max<8>(mx, red);
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) z += __expf(h2f(row[i]) - mx);
    const float inv = 1.f / block_sum<8>(z, red);
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] = f2h(__expf(h2f(row[i]) - mx) * inv);
}

__global__ void bias_out_kernel(const uint16_t *y, const float *bias, int N, int D, float *out) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < size_t(N) * D) out[i] = h2f(y[i]) + bias[i % D];
}

unsigned blocks(size_t n) {
    return unsigned((n + 255) / 256);
}

}  // namespace

void vit_layernorm(const float *x, const float *w, const float *b, int N, int D, uint16_t *y, hipStream_t s) {
    layernorm_kernel<<<N, 256, 0, s>>>(x, w, b, D, y);
}
void vit_residual_add(float *x, const uint16_t *y, int ldy, const float *bias, int N, int D, hipStream_t s) {
    residual_add_kernel<<<blocks(size_t(N) * D), 256, 0, s>>>(x, y, ldy, bias, N, D);
}
void vit_bias_gelu(uint16_t *y, const float *bias, int N, int D, bool tanh_approx, hipStream_t s) {
    bias_gelu_kernel<<<blocks(size_t(N) * D), 256, 0, s>>>(y, bias, N, D, tanh_approx);
}
void vit_bias(uint16_t *y, const float *bias, int N, int D, hipStream_t s) {
    bias_kernel<<<blocks(size_t(N) * D), 256, 0, s>>>(y, bias, N, D);
}
void vit_embed(float *x, const uint16_t *y, const float *bias, const float *table, const int32_t *idx,
               const float *wts, int N, int D, hipStream_t s) {
    embed_kernel<<<blocks(size_t(N) * D), 256, 0, s>>>(x, y, bias, table, idx, wts, N, D);
}
void vit_rope2d(uint16_t *qkv, const int32_t *pos_hw, int N, int heads, hipStream_t s) {
    rope2d_kernel<<<N, 64, 0, s>>>(qkv, pos_hw, heads);
}
void vit_softmax(uint16_t *s_rows, int rows, int n, int ld, hipStream_t s) {
    softmax_kernel<<<rows, 256, 0, s>>>(s_rows, n, ld);
}
void vit_bias_out(const uint16_t *y, const float *bias, int N, int D, float *out, hipStream_t s) {
    bias_out_kernel<<<blocks(size_t(N) * D), 256, 0, s>>>(y, bias, N, D, out);
}

}  // namespace qw::gpu
