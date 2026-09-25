#include "kernels/decode.hpp"

#include "core/config.hpp"
#include "kernels/common.cuh"

namespace qw::gpu {

using namespace cfg;

// ---------------------------------------------------------------- fp16 GEMV
// One wave per output row. Each lane streams 16-byte chunks of the row with
// two chunks in flight, dots them against every token's x, and the wave
// reduces with shuffles.
template <int M>
__global__ __launch_bounds__(256) void gemv_f16_kernel(const uint4 *__restrict__ W,
                                                       const uint4 *__restrict__ x,
                                                       float *__restrict__ y, int N, int K,
                                                       int accumulate, int rows_per_group) {
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + threadIdx.x / WAVE;
    if (row >= N) return;
    const int chunks = K / 8;  // uint4 per row
    const uint4 *w = W + size_t(row) * chunks;
    if (rows_per_group) x += size_t(row / rows_per_group) * chunks;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    int c = lane;
    for (; c + WAVE < chunks; c += 2 * WAVE) {
        uint4 w0 = w[c], w1 = w[c + WAVE];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] = dot8(w0, x[m * chunks + c], acc[m]);
            acc[m] = dot8(w1, x[m * chunks + c + WAVE], acc[m]);
        }
    }
    if (c < chunks) {
        uint4 w0 = w[c];
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = dot8(w0, x[m * chunks + c], acc[m]);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = wave_sum(acc[m]);
        if (lane == 0) {
            float *o = y + size_t(m) * N + row;
            *o = accumulate ? *o + v : v;
        }
    }
}

void gemv_f16_grouped(const uint16_t *W, const uint16_t *x, float *y, int N, int K,
                      int rows_per_group, hipStream_t s) {
    gemv_f16_kernel<1><<<(N + 7) / 8, 256, 0, s>>>(reinterpret_cast<const uint4 *>(W),
                                                    reinterpret_cast<const uint4 *>(x), y, N, K, 0,
                                                    rows_per_group);
}

void gemv_f16(const uint16_t *W, const uint16_t *x, float *y, int N, int K, int M, hipStream_t s,
              bool accumulate) {
    dim3 block(256), grid((N + 7) / 8);
    auto w = reinterpret_cast<const uint4 *>(W);
    auto xv = reinterpret_cast<const uint4 *>(x);
    switch (M) {
#define CASE(m) case m: gemv_f16_kernel<m><<<grid, block, 0, s>>>(w, xv, y, N, K, accumulate, 0); break;
        CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
        default: break;
    }
}

// ------------------------------------------------------------- int4 experts
void pack_int4_words(const int32_t *ct, uint32_t *out, size_t n_words) {
    for (size_t i = 0; i < n_words; ++i) {
        uint32_t w = uint32_t(ct[i]), r = 0;
        for (int k = 0; k < 8; ++k) {
            uint32_t nib = (w >> (4 * k)) & 0xf;
            int bit = (k & 1) ? 16 + 4 * (k >> 1) : 4 * (k >> 1);
            r |= nib << bit;
        }
        out[i] = r;
    }
}

// Grid: (FFN / 8, top_k). One wave per (expert slot, row); gate and up
// streams in the same wave.
__global__ __launch_bounds__(256) void moe_gate_up_kernel(
    const uint4 *__restrict__ gw, const __half *__restrict__ gs, const uint4 *__restrict__ uw,
    const __half *__restrict__ us, const int32_t *__restrict__ ids, const int32_t *__restrict__ count,
    const uint4 *__restrict__ x, __half *__restrict__ h) {
    constexpr int CH = H / 32;  // 80 uint4 per row
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int slot = blockIdx.y;
    if (count && slot >= *count) return;
    const int e = ids[slot];
    const size_t rbase = (size_t(e) * FFN + row);
    const uint4 *g = gw + rbase * CH, *u = uw + rbase * CH;
    const __half *gsc = gs + rbase * (H / QGROUP), *usc = us + rbase * (H / QGROUP);
    float ag = 0.f, au = 0.f;
    for (int c = lane; c < CH; c += WAVE) {
        uint4 g0 = g[c], u0 = u[c];
        const uint4 *xc = x + c * 4;
        const int grp = c / (QGROUP / 32);
        ag += __half2float(gsc[grp]) * dot_int4x32(g0, xc, 0.f);
        au += __half2float(usc[grp]) * dot_int4x32(u0, xc, 0.f);
    }
    ag = wave_sum(ag);
    au = wave_sum(au);
    if (lane == 0) h[slot * FFN + row] = __float2half(silu(ag) * au);
}

void moe_gate_up(const uint32_t *gate_w, const uint16_t *gate_s, const uint32_t *up_w,
                 const uint16_t *up_s, const int32_t *ids, const int32_t *count, const uint16_t *x,
                 uint16_t *h, int top_k, hipStream_t s) {
    dim3 grid(FFN / 8, top_k), block(256);
    moe_gate_up_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const uint4 *>(gate_w), reinterpret_cast<const __half *>(gate_s),
        reinterpret_cast<const uint4 *>(up_w), reinterpret_cast<const __half *>(up_s), ids, count,
        reinterpret_cast<const uint4 *>(x), reinterpret_cast<__half *>(h));
}

// One wave per output row; the wave's lanes stride over (expert slot, chunk)
// pairs so all TOP_K experts' rows are read in one pass and combined in
// registers.
__global__ __launch_bounds__(256) void moe_down_kernel(
    const uint4 *__restrict__ dw, const __half *__restrict__ ds, const int32_t *__restrict__ ids,
    const int32_t *__restrict__ count, const float *__restrict__ weight, const uint4 *__restrict__ h,
    float *__restrict__ y, int top_k, int accumulate) {
    constexpr int CH = FFN / 32;  // 20 uint4 per row
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int n = count ? *count : top_k;
    float acc = 0.f;
    for (int j = lane; j < n * CH; j += WAVE) {
        const int slot = j / CH, c = j % CH;
        const size_t rbase = size_t(ids[slot]) * H + row;
        uint4 w0 = dw[rbase * CH + c];
        float sc = __half2float(ds[rbase * (FFN / QGROUP) + c / (QGROUP / 32)]);
        acc += weight[slot] * sc * dot_int4x32(w0, h + slot * (FFN / 8) + c * 4, 0.f);
    }
    acc = wave_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

void moe_down(const uint32_t *down_w, const uint16_t *down_s, const int32_t *ids,
              const int32_t *count, const float *weight, const uint16_t *h, float *y, int top_k,
              bool accumulate, hipStream_t s) {
    moe_down_kernel<<<H / 8, 256, 0, s>>>(reinterpret_cast<const uint4 *>(down_w),
                                          reinterpret_cast<const __half *>(down_s), ids, count,
                                          weight, reinterpret_cast<const uint4 *>(h), y, top_k,
                                          accumulate);
}

// ---------------------------------------------------------------- GDN step
// One block (256 threads) per v-head. Threads 0..127 own channel j of q, k and
// v for the conv; for the state update thread t owns column j = t % 128 and
// k-rows [64*(t/128), +64).
__global__ __launch_bounds__(256) void gdn_decode_kernel(GdnStep p) {
    const int h = blockIdx.x;
    const int kh = h / (GDN_V_HEADS / GDN_QK_HEADS);
    const int key = p.n_v / (GDN_V_HEADS / GDN_QK_HEADS) * GDN_DIM;  // local key dim
    const int qkv_dim = 2 * key + p.n_v * GDN_DIM;
    const int t = threadIdx.x;
    __shared__ float q[GDN_DIM], k[GDN_DIM], v[GDN_DIM], red[8], part[2][GDN_DIM];

    const int slot = int(*p.pos % 4);
    auto conv = [&](int ch) {
        // new row goes to `slot`; taps are pos-3, pos-2, pos-1, pos
        float x = p.qkv[ch];
        float acc = p.conv_w[ch * 4 + 3] * x;
#pragma unroll
        for (int d = 1; d < 4; ++d) acc += p.conv_w[ch * 4 + 3 - d] * p.conv_ring[((slot - d + 4) % 4) * qkv_dim + ch];
        return silu(acc);
    };
    if (t < GDN_DIM) {
        const int qc = kh * GDN_DIM + t, kc = key + kh * GDN_DIM + t, vc = 2 * key + h * GDN_DIM + t;
        q[t] = conv(qc);
        k[t] = conv(kc);
        v[t] = conv(vc);
    }
    __syncthreads();
    // Write this step's row into the ring. The q/k channels are shared by 3
    // v-heads: the first of them writes. Nobody reads slot `slot` this step.
    if (t < GDN_DIM) {
        const int qc = kh * GDN_DIM + t, kc = key + kh * GDN_DIM + t, vc = 2 * key + h * GDN_DIM + t;
        float *ring = p.conv_ring + slot * qkv_dim;
        if (h % (GDN_V_HEADS / GDN_QK_HEADS) == 0) {
            ring[qc] = p.qkv[qc];
            ring[kc] = p.qkv[kc];
        }
        ring[vc] = p.qkv[vc];
    }
    float sq = t < GDN_DIM ? q[t] * q[t] : 0.f, sk = t < GDN_DIM ? k[t] * k[t] : 0.f;
    const float nq = block_sum<8>(sq, red);
    const float nk = block_sum<8>(sk, red);
    __syncthreads();
    if (t < GDN_DIM) {
        q[t] = q[t] * rsqrtf(nq + EPS) * rsqrtf(float(GDN_DIM));
        k[t] = k[t] * rsqrtf(nk + EPS);
    }
    __syncthreads();

    const float av = p.a[h] + p.dt_bias[h];
    const float sp = av > 20.f ? av : log1pf(__expf(av));
    const float decay = __expf(-__expf(p.A_log[h]) * sp);
    const float beta = sigmoid(p.b[h]);

    const int j = t % GDN_DIM, half = t / GDN_DIM;
    float *S = p.S + size_t(h) * GDN_DIM * GDN_DIM;
    float s[64];
    float pu = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        const int ki = half * 64 + i;
        s[i] = S[ki * GDN_DIM + j] * decay;
        pu += s[i] * k[ki];
    }
    part[half][j] = pu;
    __syncthreads();
    const float u = (v[j] - part[0][j] - part[1][j]) * beta;
    __syncthreads();
    float po = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        const int ki = half * 64 + i;
        s[i] += k[ki] * u;
        S[ki * GDN_DIM + j] = s[i];
        po += s[i] * q[ki];
    }
    part[half][j] = po;
    __syncthreads();
    const float o = part[0][j] + part[1][j];
    const float ss = block_sum<8>(half == 0 ? o * o : 0.f, red);
    if (half == 0) {
        const float r = rsqrtf(ss / GDN_DIM + EPS);
        p.out[h * GDN_DIM + j] = __half_as_ushort(
            __float2half(o * r * p.norm_w[j] * sigmoid(p.z[h * GDN_DIM + j])));
    }
}

void gdn_decode(const GdnStep &p, hipStream_t s) {
    gdn_decode_kernel<<<p.n_v, 256, 0, s>>>(p);
}

}  // namespace qw::gpu
