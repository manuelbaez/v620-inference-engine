#include "kernels/gemv.hpp"

#include <string>

#include "core/common.hpp"
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

// Four E4M3 bytes -> two h2 (bytes 0-1, 2-3), each value x 2^-8, exactly: the sign to bit 15, exponent and mantissa
// shifted 7 to fp16's (E4M3's exponent bias is 7, fp16's 15; E4M3 subnormals land on fp16 subnormals).
__device__ __forceinline__ void e4m3x4_to_h2x2(uint32_t w, uint32_t &lo, uint32_t &hi) {
    lo = ((w & 0x7Fu) << 7) | ((w & 0x80u) << 8) | ((w & 0x7F00u) << 15) | ((w & 0x8000u) << 16);
    hi = ((w >> 9) & 0x3F80u) | ((w >> 8) & 0x8000u) | ((w >> 1) & 0x3F800000u) | (w & 0x80000000u);
}

// 16 int8 (or E4M3) weights against 16 fp16 activations (two uint4).
template <bool F8 = false>
__device__ __forceinline__ float dot16_i8(const uint4 &w, const uint4 &x0, const uint4 &x1, float acc) {
    uint32_t h[8];
    if constexpr (F8) {
        e4m3x4_to_h2x2(w.x, h[0], h[1]);
        e4m3x4_to_h2x2(w.y, h[2], h[3]);
        e4m3x4_to_h2x2(w.z, h[4], h[5]);
        e4m3x4_to_h2x2(w.w, h[6], h[7]);
    } else {
        i8x4_to_h2x2(w.x, h[0], h[1]);
        i8x4_to_h2x2(w.y, h[2], h[3]);
        i8x4_to_h2x2(w.z, h[4], h[5]);
        i8x4_to_h2x2(w.w, h[6], h[7]);
    }
    acc = dot2(h[0], x0.x, acc);
    acc = dot2(h[1], x0.y, acc);
    acc = dot2(h[2], x0.z, acc);
    acc = dot2(h[3], x0.w, acc);
    acc = dot2(h[4], x1.x, acc);
    acc = dot2(h[5], x1.y, acc);
    acc = dot2(h[6], x1.z, acc);
    return dot2(h[7], x1.w, acc);
}

template <int M, bool F8>
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
            acc[m] += s0 * dot16_i8<F8>(w0, xm[2 * c], xm[2 * c + 1], 0.f);
            acc[m] += s1 * dot16_i8<F8>(w1, xm[2 * (c + WAVE)], xm[2 * (c + WAVE) + 1], 0.f);
        }
    }
    if (c < chunks) {
        const uint4 w0 = w[c];
        const float s0 = sc[c / CPG];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 *xm = x + size_t(m) * xs16;
            acc[m] += s0 * dot16_i8<F8>(w0, xm[2 * c], xm[2 * c + 1], 0.f);
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
                  int rpg, bool accumulate, hipStream_t s, bool e4m3) {
    auto w = reinterpret_cast<const uint4 *>(W8);
    auto xv = reinterpret_cast<const uint4 *>(x);
    const dim3 grid((N + 7) / 8), block(256);
    switch (M) {
#define CASE(m)                                                                                                       \
    case m:                                                                                                           \
        if (e4m3)                                                                                                     \
            gemv_rows_i8_kernel<m, true><<<grid, block, 0, s>>>(w, scale, xv, xs / 8, y, ys, N, K, rpg, accumulate);  \
        else                                                                                                          \
            gemv_rows_i8_kernel<m, false><<<grid, block, 0, s>>>(w, scale, xv, xs / 8, y, ys, N, K, rpg, accumulate); \
        break;
        CASE(1)
        CASE(2)
        CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15)
            CASE(16)
            CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28)
            CASE(29) CASE(30) CASE(31) CASE(32)
#undef CASE
                default : fail("gemv_rows: " + std::to_string(M) + " rows (1..32)");
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
            CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28)
            CASE(29) CASE(30) CASE(31) CASE(32)
#undef CASE
                default : fail("gemv_rows: " + std::to_string(M) + " rows (1..32)");
    }
}

__global__ __launch_bounds__(256) void e4m3_to_f16_kernel(const uint4 *__restrict__ W, const float *__restrict__ scale,
                                                          int64_t K, int64_t n16, uint4 *__restrict__ out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // 16 weights
    if (i >= n16) return;
    const float sc = scale[i * 16 / I8_GROUP];  // the rows' groups are consecutive: [N][K / I8_GROUP]
    const uint4 w = W[i];
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    uint32_t o[8];
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo, hi;
        e4m3x4_to_h2x2(ws[j], lo, hi);
        const uint32_t hs[2] = {lo, hi};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const h2 v = __builtin_bit_cast(h2, hs[k]);
            const float a = fminf(fmaxf(float(v.x) * sc, -65504.f), 65504.f), b = fminf(fmaxf(float(v.y) * sc, -65504.f), 65504.f);
            o[2 * j + k] = __builtin_bit_cast(uint32_t, h2{(_Float16)a, (_Float16)b});
        }
    }
    out[2 * i] = make_uint4(o[0], o[1], o[2], o[3]);
    out[2 * i + 1] = make_uint4(o[4], o[5], o[6], o[7]);
    (void)K;
}

void e4m3_to_f16(const int8_t *W8, const float *scale, int64_t N, int64_t K, uint16_t *out, hipStream_t s) {
    const int64_t n16 = N * K / 16;
    e4m3_to_f16_kernel<<<dim3(unsigned((n16 + 255) / 256)), 256, 0, s>>>(reinterpret_cast<const uint4 *>(W8), scale, K,
                                                                          n16, reinterpret_cast<uint4 *>(out));
}

}  // namespace qw::gpu
