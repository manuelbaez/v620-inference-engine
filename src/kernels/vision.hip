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

// One thread per query row, ATT_Q rows per block, one head per blockIdx.y. Keys
// and values stream through LDS in tiles of ATT_TILE: keys as half2 pairs along
// the head dim (scores: 36 dot2 per key), values transposed as half2 pairs of
// consecutive keys (the output update is 36 dot2 per key too). All lanes of a
// wave read the same LDS words (broadcast), 16 bytes at a time. Online softmax
// per group of ATT_GROUP keys, fp32 accumulation.
constexpr int ATT_HD = 72;
constexpr int KW = ATT_HD / 8;  // uint4 per key row (8 halves each)
template <int ATT_Q, int ATT_TILE, int ATT_GROUP>
__global__ __launch_bounds__(ATT_Q) void attention_kernel(const uint16_t *qkv, int L, int heads, uint16_t *out) {
    constexpr int VW = ATT_TILE / 2 / 4;  // uint4 per value column (4 key pairs each)
    __shared__ uint4 k_s[ATT_TILE][KW];  // [key][8 dims]
    __shared__ uint4 vt_s[ATT_HD][VW];   // [dim][4 key pairs]
    const int h = blockIdx.y, q = blockIdx.x * ATT_Q + threadIdx.x;
    const size_t ld = size_t(3) * heads * ATT_HD;
    const float scale = rsqrtf(float(ATT_HD));
    uint32_t qv[ATT_HD / 2];  // pre-scaled query, half2 pairs
    if (q < L) {
        const uint32_t *qp = reinterpret_cast<const uint32_t *>(qkv + size_t(q) * ld + size_t(h) * ATT_HD);
#pragma unroll
        for (int i = 0; i < ATT_HD / 2; ++i) {
            const float2 f = __half22float2(__builtin_bit_cast(__half2, qp[i]));
            qv[i] = __builtin_bit_cast(uint32_t, __floats2half2_rn(f.x * scale, f.y * scale));
        }
    }
    float o[ATT_HD];
#pragma unroll
    for (int d = 0; d < ATT_HD; ++d) o[d] = 0.f;
    float m = -INFINITY, l = 0.f;
    const __half *kbase = reinterpret_cast<const __half *>(qkv) + size_t(heads + h) * ATT_HD;
    const __half *vbase = reinterpret_cast<const __half *>(qkv) + size_t(2 * heads + h) * ATT_HD;
    uint32_t *vt_w = reinterpret_cast<uint32_t *>(vt_s);
    for (int k0 = 0; k0 < L; k0 += ATT_TILE) {
        __syncthreads();
        for (int i = threadIdx.x; i < ATT_TILE * KW; i += ATT_Q) {  // keys: rows of 8 halves
            const int kr = i / KW, c = i % KW, key = k0 + kr;
            k_s[kr][c] = key < L ? reinterpret_cast<const uint4 *>(kbase + size_t(key) * ld)[c] : make_uint4(0, 0, 0, 0);
        }
        for (int i = threadIdx.x; i < ATT_HD * ATT_TILE / 2; i += ATT_Q) {  // values: pairs of keys per dim
            const int d = i % ATT_HD, kp = i / ATT_HD, key = k0 + 2 * kp;
            const __half a = key < L ? vbase[size_t(key) * ld + d] : __float2half(0.f);
            const __half b = key + 1 < L ? vbase[size_t(key + 1) * ld + d] : __float2half(0.f);
            vt_w[d * VW * 4 + kp] = __builtin_bit_cast(uint32_t, __halves2half2(a, b));
        }
        __syncthreads();
        if (q >= L) continue;
        const int n_keys = min(ATT_TILE, L - k0);
        for (int g = 0; g < n_keys; g += ATT_GROUP) {
            float sc[ATT_GROUP];
            float gmax = -INFINITY;
#pragma unroll
            for (int j = 0; j < ATT_GROUP; ++j) {
                float acc = 0.f;
#pragma unroll
                for (int c = 0; c < KW; ++c) {
                    const uint4 kv = k_s[g + j][c];
                    acc = dot2(qv[4 * c], kv.x, acc);
                    acc = dot2(qv[4 * c + 1], kv.y, acc);
                    acc = dot2(qv[4 * c + 2], kv.z, acc);
                    acc = dot2(qv[4 * c + 3], kv.w, acc);
                }
                sc[j] = g + j < n_keys ? acc : -INFINITY;
                gmax = fmaxf(gmax, sc[j]);
            }
            const float m_new = fmaxf(m, gmax);
            const float alpha = __expf(m - m_new);
            uint32_t pp[ATT_GROUP / 2];  // probabilities as half2 pairs of consecutive keys
            float psum = 0.f;
#pragma unroll
            for (int j = 0; j < ATT_GROUP; j += 2) {
                const float p0 = __expf(sc[j] - m_new), p1 = __expf(sc[j + 1] - m_new);
                psum += p0 + p1;
                pp[j / 2] = __builtin_bit_cast(uint32_t, __floats2half2_rn(p0, p1));
            }
            l = l * alpha + psum;
            const int gp = g / 8;  // first uint4 of this group's key pairs
#pragma unroll
            for (int d = 0; d < ATT_HD; ++d) {
                float acc = o[d] * alpha;
#pragma unroll
                for (int u = 0; u < ATT_GROUP / 8; ++u) {
                    const uint4 v = vt_s[d][gp + u];
                    acc = dot2(pp[4 * u], v.x, acc);
                    acc = dot2(pp[4 * u + 1], v.y, acc);
                    acc = dot2(pp[4 * u + 2], v.z, acc);
                    acc = dot2(pp[4 * u + 3], v.w, acc);
                }
                o[d] = acc;
            }
            m = m_new;
        }
    }
    if (q >= L) return;
    const float inv = 1.f / l;
    uint32_t *op = reinterpret_cast<uint32_t *>(out + size_t(q) * heads * ATT_HD + size_t(h) * ATT_HD);
#pragma unroll
    for (int i = 0; i < ATT_HD / 2; ++i)
        op[i] = __builtin_bit_cast(uint32_t, __floats2half2_rn(o[2 * i] * inv, o[2 * i + 1] * inv));
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
template <int Q, int TILE, int GROUP>
void launch_attention(const uint16_t *qkv, int L, int heads, uint16_t *out, hipStream_t s) {
    attention_kernel<Q, TILE, GROUP><<<dim3(unsigned((L + Q - 1) / Q), unsigned(heads)), Q, 0, s>>>(qkv, L, heads, out);
}

// 256 queries per block pays from ~6k patches (a 1080p image: 1.34 -> 1.23 s
// for the whole tower); below that the grid gets too small (720p: 0.34 ->
// 0.37 s). Keys per LDS tile and per softmax group were swept too (32-128,
// 8-32): within noise of 64 / 16.
void vit_attention(const uint16_t *qkv, int L, int heads, uint16_t *out, hipStream_t s) {
    if (L >= 6000)
        launch_attention<256, 64, 16>(qkv, L, heads, out, s);
    else
        launch_attention<128, 64, 16>(qkv, L, heads, out, s);
}

void vit_softmax(uint16_t *s_rows, int rows, int n, int ld, hipStream_t s) {
    softmax_kernel<<<rows, 256, 0, s>>>(s_rows, n, ld);
}
void vit_bias_out(const uint16_t *y, const float *bias, int N, int D, float *out, hipStream_t s) {
    bias_out_kernel<<<blocks(size_t(N) * D), 256, 0, s>>>(y, bias, N, D, out);
}

}  // namespace qw::gpu
