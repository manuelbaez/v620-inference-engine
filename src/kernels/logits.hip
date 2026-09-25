#include "kernels/logits.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

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

}  // namespace qw::gpu
