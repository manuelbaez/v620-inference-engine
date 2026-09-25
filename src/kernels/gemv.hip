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
        CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14)
            CASE(15) CASE(16)
#undef CASE
                default : break;
    }
}

}  // namespace qw::gpu
