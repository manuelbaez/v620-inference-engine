#include "kernels/blend.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

// Same layout and parallelism as gdn_scan_T_kernel (grid 12 heads x 4 column slices).
__global__ __launch_bounds__(256) void gdn_transfer_T_kernel(const float *qkv, const float *gb, float *M, int T) {
    __shared__ float red[8][32];
    const int h = blockIdx.x / 4, slice = blockIdx.x % 4;
    const int kh = h / (GDN_V_HEADS / GDN_QK_HEADS);
    const int c = threadIdx.x % 32, rg = threadIdx.x / 32;
    const int j = slice * 32 + c;
    float *Mh = M + size_t(h) * GDN_DIM * GDN_DIM;
    float s[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = Mh[(rg * 16 + i) * GDN_DIM + j];
    for (int t = 0; t < T; ++t) {
        const float *kk = qkv + size_t(t) * GDN_QKV_L + GDN_KL * GDN_DIM + kh * GDN_DIM + rg * 16;
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
        const float u = -tot * beta;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) s[i] += kk[i] * u;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) Mh[(rg * 16 + i) * GDN_DIM + j] = s[i];
}

void gdn_transfer_T(const float *qkv, const float *gb, float *M, int T, hipStream_t s) {
    gdn_transfer_T_kernel<<<GDN_VL * 4, 256, 0, s>>>(qkv, gb, M, T);
}

__global__ void gdn_transfer_init_kernel(float *M, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const size_t e = i % (GDN_DIM * GDN_DIM);
    M[i] = e / GDN_DIM == e % GDN_DIM ? 1.f : 0.f;
}

void gdn_transfer_init(float *M, int layers, hipStream_t s) {
    const size_t n = size_t(layers) * GDN_STATE;
    gdn_transfer_init_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(M, n);
}

// grid (layers * 12 heads, 128 rows), 128 threads (columns): tmp = S_out + M (S - S_in)
__global__ __launch_bounds__(GDN_DIM) void gdn_compose_kernel(const float *S, const float *M, const float *S_in,
                                                              const float *S_out, float *tmp) {
    __shared__ float mrow[GDN_DIM];
    const size_t hb = size_t(blockIdx.x) * GDN_DIM * GDN_DIM;
    const int i = blockIdx.y, j = threadIdx.x;
    mrow[j] = M[hb + size_t(i) * GDN_DIM + j];
    __syncthreads();
    float acc = 0.f;
    for (int k = 0; k < GDN_DIM; ++k) acc += mrow[k] * (S[hb + size_t(k) * GDN_DIM + j] - S_in[hb + size_t(k) * GDN_DIM + j]);
    tmp[hb + size_t(i) * GDN_DIM + j] = S_out[hb + size_t(i) * GDN_DIM + j] + acc;
}

void gdn_compose(float *S, const float *M, const float *S_in, const float *S_out, float *tmp, int layers,
                 hipStream_t s) {
    gdn_compose_kernel<<<dim3(unsigned(layers * GDN_VL), GDN_DIM), GDN_DIM, 0, s>>>(S, M, S_in, S_out, tmp);
    (void)hipMemcpyAsync(S, tmp, size_t(layers) * GDN_STATE * 4, hipMemcpyDeviceToDevice, s);
}

__global__ __launch_bounds__(HEAD_DIM) void rope_shift_kernel(__half *K, int64_t delta) {
    __shared__ float buf[HEAD_DIM];
    __half *row = K + size_t(blockIdx.x) * HEAD_DIM;
    buf[threadIdx.x] = __half2float(row[threadIdx.x]);
    rope_shared(buf, delta);
    if (threadIdx.x < ROPE_DIM) row[threadIdx.x] = __float2half(buf[threadIdx.x]);
}

void rope_shift(uint16_t *K, int64_t n, int64_t delta, hipStream_t s) {
    if (n > 0) rope_shift_kernel<<<unsigned(n), HEAD_DIM, 0, s>>>(reinterpret_cast<__half *>(K), delta);
}

}  // namespace qw::gpu
