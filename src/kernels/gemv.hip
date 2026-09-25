#include "kernels/gemv.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

template <int M>
__global__ __launch_bounds__(256) void gemv_rows_kernel(const uint4 *__restrict__ W, const uint4 *__restrict__ x,
                                                        int xs16, float *__restrict__ y, int ys, int N, int K, int rpg,
                                                        int accumulate) {
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

// Four int8 weights (one 32-bit word) -> two half2 words, exactly: each byte,
// offset to unsigned, becomes the low byte of 1024.0 in fp16 (0x6400 | u),
// then 1152 (1024 + 128) is subtracted.
__device__ __forceinline__ void i8x4_to_h2x2(uint32_t w, uint32_t &lo, uint32_t &hi) {
    const uint32_t u = w ^ 0x80808080u;
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t a = (u & 0x000000FFu) | ((u & 0x0000FF00u) << 8) | 0x64006400u;
    const uint32_t b = ((u >> 16) & 0x000000FFu) | ((u >> 8) & 0x00FF0000u) | 0x64006400u;
    const h2 off{(_Float16)1152.0f, (_Float16)1152.0f};
    lo = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, a) - off);
    hi = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, b) - off);
}

// 16 int8 weights against 16 fp16 activations (two uint4).
__device__ __forceinline__ float dot16_i8(const uint4 &w, const uint4 &x0, const uint4 &x1, float acc) {
    uint32_t h[8];
    i8x4_to_h2x2(w.x, h[0], h[1]);
    i8x4_to_h2x2(w.y, h[2], h[3]);
    i8x4_to_h2x2(w.z, h[4], h[5]);
    i8x4_to_h2x2(w.w, h[6], h[7]);
    acc = dot2(h[0], x0.x, acc);
    acc = dot2(h[1], x0.y, acc);
    acc = dot2(h[2], x0.z, acc);
    acc = dot2(h[3], x0.w, acc);
    acc = dot2(h[4], x1.x, acc);
    acc = dot2(h[5], x1.y, acc);
    acc = dot2(h[6], x1.z, acc);
    return dot2(h[7], x1.w, acc);
}

template <int M>
__global__ __launch_bounds__(256) void gemv_rows_i8_kernel(const uint4 *__restrict__ W, const float *__restrict__ scale,
                                                           const uint4 *__restrict__ x, int xs16, float *__restrict__ y,
                                                           int ys, int N, int K, int rpg, int accumulate) {
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + threadIdx.x / WAVE;
    if (row >= N) return;
    const int chunks = K / 16;  // 16 weights per uint4
    const uint4 *w = W + size_t(row) * chunks;
    if (rpg) x += size_t(row / rpg) * (K / 8);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const float *sc = scale + size_t(row) * (K / I8_GROUP);
    constexpr int CPG = I8_GROUP / 16;  // chunks per scale group
    int c = lane;
    for (; c + WAVE < chunks; c += 2 * WAVE) {
        const uint4 w0 = w[c], w1 = w[c + WAVE];
        const float s0 = sc[c / CPG], s1 = sc[(c + WAVE) / CPG];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 *xm = x + size_t(m) * xs16;
            acc[m] += s0 * dot16_i8(w0, xm[2 * c], xm[2 * c + 1], 0.f);
            acc[m] += s1 * dot16_i8(w1, xm[2 * (c + WAVE)], xm[2 * (c + WAVE) + 1], 0.f);
        }
    }
    if (c < chunks) {
        const uint4 w0 = w[c];
        const float s0 = sc[c / CPG];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 *xm = x + size_t(m) * xs16;
            acc[m] += s0 * dot16_i8(w0, xm[2 * c], xm[2 * c + 1], 0.f);
        }
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

void gemv_rows_i8(const int8_t *W8, const float *scale, const uint16_t *x, int xs, float *y, int ys, int N, int K, int M,
                  int rpg, bool accumulate, hipStream_t s) {
    auto w = reinterpret_cast<const uint4 *>(W8);
    auto xv = reinterpret_cast<const uint4 *>(x);
    const dim3 grid((N + 7) / 8), block(256);
    switch (M) {
#define CASE(m)                                                                                             \
    case m:                                                                                                 \
        gemv_rows_i8_kernel<m><<<grid, block, 0, s>>>(w, scale, xv, xs / 8, y, ys, N, K, rpg, accumulate); \
        break;
        CASE(1)
        CASE(2)
        CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15)
            CASE(16)
#undef CASE
                default : break;
    }
}

void gemv_rows(const uint16_t *W, const uint16_t *x, int xs, float *y, int ys, int N, int K, int M, int rpg,
               bool accumulate, hipStream_t s) {
    auto w = reinterpret_cast<const uint4 *>(W);
    auto xv = reinterpret_cast<const uint4 *>(x);
    const dim3 grid((N + 7) / 8), block(256);
    switch (M) {
#define CASE(m)                                                                                  \
    case m:                                                                                      \
        gemv_rows_kernel<m><<<grid, block, 0, s>>>(w, xv, xs / 8, y, ys, N, K, rpg, accumulate); \
        break;
        CASE(1)
        CASE(2)
        CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15)
            CASE(16)
#undef CASE
                default : break;
    }
}

}  // namespace qw::gpu
