#include "kernels/moe_tiling.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

// Same tiling as above, on v_dot4_i32_i8 (1.7x the fp16 dot rate on gfx1030):
// activations quantized per row to int8 (x = q * scale), int4 weights widened
// to int8 exactly (nibble - 8). int32 accumulation per 64-wide K step, then the
// weight group scale (constant within a step, groups are 128 wide) in fp32.
// LDS rows: A 80 bytes (16-byte aligned), B 68 bytes (17 words: conflict-free
// across the 32 consecutive rows a wave reads).
constexpr int A8_LD = BK + 16, B8_LD = BK + 4;

// Per-(row, 128-group) int8 quantization: scale[row][g] = max|x| / 127 over
// the group. One wave per group (4 values per lane).
__global__ __launch_bounds__(256) void quant_rows_kernel(const __half *x, int n, int8_t *q, float *scale) {
    const int row = blockIdx.x, ng = n / QGROUP;
    const int lane = threadIdx.x % WAVE;
    for (int g = threadIdx.x / WAVE; g < ng; g += blockDim.x / WAVE) {
        const __half *xg = x + size_t(row) * n + g * QGROUP;
        float v[QGROUP / WAVE], mx = 0.f;
#pragma unroll
        for (int i = 0; i < QGROUP / WAVE; ++i) {
            v[i] = __half2float(xg[lane + i * WAVE]);
            mx = fmaxf(mx, fabsf(v[i]));
        }
        mx = wave_max(mx);
        const float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc;
#pragma unroll
        for (int i = 0; i < QGROUP / WAVE; ++i)
            q[size_t(row) * n + g * QGROUP + lane + i * WAVE] = int8_t(__float2int_rn(v[i] * inv));
        if (lane == 0) scale[size_t(row) * ng + g] = sc;
    }
}

// 32 int4 weights (one packed uint4, engine nibble layout) -> 32 int8 into dst.
__device__ __forceinline__ void widen32(const uint4 &w, int8_t *dst) {
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    uint32_t *d = reinterpret_cast<uint32_t *>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // word i holds k = 8i..8i+7: k(2m) at bits 4m, k(2m+1) at bits 16+4m
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            lo |= (((ws[i] >> (4 * m)) & 0xfu) - 8u & 0xffu) << (16 * m);
            lo |= (((ws[i] >> (16 + 4 * m)) & 0xfu) - 8u & 0xffu) << (16 * m + 8);
            hi |= (((ws[i] >> (4 * (m + 2))) & 0xfu) - 8u & 0xffu) << (16 * m);
            hi |= (((ws[i] >> (16 + 4 * (m + 2))) & 0xfu) - 8u & 0xffu) << (16 * m + 8);
        }
        d[2 * i] = lo;
        d[2 * i + 1] = hi;
    }
}

__device__ __forceinline__ int sdot4(uint32_t a, uint32_t b, int c) {
    return __builtin_amdgcn_sdot4(int(a), int(b), c, false);
}

__global__ __launch_bounds__(256) void moe_gate_up_i8_kernel(const uint4 *gw, const __half *gs, const uint4 *uw,
                                                             const __half *us, const int32_t *counts,
                                                             const int32_t *offsets, const int32_t *tile_e,
                                                             const int32_t *tile_m0, const int32_t *ntiles,
                                                             const int32_t *pair_tok, const int8_t *xq, const float *xs,
                                                             __half *h) {
    constexpr int CH = H / 32;
    __shared__ __align__(16) int8_t As[MT][A8_LD];
    __shared__ __align__(4) int8_t Bg[64][B8_LD], Bu[64][B8_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 64;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    // A loader (threads 0..127): token tid/4, 16-byte chunk tid%4
    const int la_tok = tid / 4, la_c = tid % 4;
    const bool la = tid < 128 && la_tok < mt;
    const uint4 *a_src = la ? reinterpret_cast<const uint4 *>(xq + size_t(pair_tok[base + la_tok]) * H) : nullptr;
    const bool lb_up = tid >= 128;
    const int lb_row = (tid % 128) / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * FFN + row0 + lb_row;
    const uint4 *b_src = (lb_up ? uw : gw) + wrow * CH;
    int8_t *b_dst = (lb_up ? &Bu[lb_row][0] : &Bg[lb_row][0]) + lb_part * 32;
    const __half *gsr0 = gs + (size_t(e) * FFN + row0 + tx) * (H / QGROUP);
    const __half *usr0 = us + (size_t(e) * FFN + row0 + tx) * (H / QGROUP);
    const size_t rs32 = size_t(32) * (H / QGROUP);  // scale stride for row + 32
    const float *xsr[4];                            // per-group activation scales of this thread's 4 tokens
#pragma unroll
    for (int i = 0; i < 4; ++i)
        xsr[i] = ty * 4 + i < mt ? xs + size_t(pair_tok[base + ty * 4 + i]) * (H / QGROUP) : nullptr;

    float fg[4][2] = {}, fu[4][2] = {};
    for (int k0 = 0; k0 < H; k0 += BK) {
        __syncthreads();
        if (tid < 128)
            *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 16]) = la ? a_src[k0 / 16 + la_c] : make_uint4(0, 0, 0, 0);
        widen32(b_src[k0 / 32 + lb_part], b_dst);
        __syncthreads();
        int ig[4][2] = {}, iu[4][2] = {};
#pragma unroll 4
        for (int kk = 0; kk < BK; kk += 4) {
            uint32_t a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
            const uint32_t g0 = *reinterpret_cast<const uint32_t *>(&Bg[tx][kk]);
            const uint32_t g1 = *reinterpret_cast<const uint32_t *>(&Bg[tx + 32][kk]);
            const uint32_t u0 = *reinterpret_cast<const uint32_t *>(&Bu[tx][kk]);
            const uint32_t u1 = *reinterpret_cast<const uint32_t *>(&Bu[tx + 32][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ig[i][0] = sdot4(a[i], g0, ig[i][0]);
                ig[i][1] = sdot4(a[i], g1, ig[i][1]);
                iu[i][0] = sdot4(a[i], u0, iu[i][0]);
                iu[i][1] = sdot4(a[i], u1, iu[i][1]);
            }
        }
        const int grp = k0 / QGROUP;
        const float sg0 = __half2float(gsr0[grp]), sg1 = __half2float(gsr0[rs32 + grp]);
        const float su0 = __half2float(usr0[grp]), su1 = __half2float(usr0[rs32 + grp]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float xg = xsr[i] ? xsr[i][grp] : 0.f;
            fg[i][0] += float(ig[i][0]) * sg0 * xg;
            fg[i][1] += float(ig[i][1]) * sg1 * xg;
            fu[i][0] += float(iu[i][0]) * su0 * xg;
            fu[i][1] += float(iu[i][1]) * su1 * xg;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            h[size_t(base + tok) * FFN + row0 + tx + 32 * j] = __float2half(silu(fg[i][j]) * fu[i][j]);
    }
}

__global__ __launch_bounds__(256) void moe_down_i8_kernel(const uint4 *dw, const __half *ds, const int32_t *counts,
                                                          const int32_t *offsets, const int32_t *tile_e,
                                                          const int32_t *tile_m0, const int32_t *ntiles,
                                                          const int8_t *hq, const float *hs, __half *yp) {
    constexpr int CH = FFN / 32;
    __shared__ __align__(16) int8_t As[MT][A8_LD];
    __shared__ __align__(4) int8_t Bs[128][B8_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 128;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    const int la_tok = tid / 4, la_c = tid % 4;
    const bool la = tid < 128 && la_tok < mt;
    const uint4 *a_src = la ? reinterpret_cast<const uint4 *>(hq + size_t(base + la_tok) * FFN) : nullptr;
    const int lb_row = tid / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * H + row0 + lb_row;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < FFN; k0 += BK) {
        __syncthreads();
        if (tid < 128)
            *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 16]) = la ? a_src[k0 / 16 + la_c] : make_uint4(0, 0, 0, 0);
        widen32(dw[wrow * CH + k0 / 32 + lb_part], &Bs[lb_row][lb_part * 32]);
        __syncthreads();
        int ia[4][4] = {};
#pragma unroll 4
        for (int kk = 0; kk < BK; kk += 4) {
            uint32_t a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const uint32_t *>(&Bs[tx + 32 * j][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) ia[i][j] = sdot4(a[i], b[j], ia[i][j]);
        }
        const int grp = k0 / QGROUP;
        float hg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            hg[i] = ty * 4 + i < mt ? hs[size_t(base + ty * 4 + i) * (FFN / QGROUP) + grp] : 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float sc = __half2float(ds[(size_t(e) * H + row0 + tx + 32 * j) * (FFN / QGROUP) + grp]);
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i][j] += float(ia[i][j]) * sc * hg[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) yp[size_t(base + tok) * H + row0 + tx + 32 * j] = __float2half(acc[i][j]);
    }
}

void moe_experts_w4a8(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                      const uint32_t *dw, const uint16_t *ds, const int32_t *counts, const int32_t *offsets,
                      const int32_t *pair_tok, const uint16_t *bin, int T, const int32_t *tile_e,
                      const int32_t *tile_m0, const int32_t *ntiles, int max_tiles, uint16_t *h, uint16_t *yp,
                      int8_t *q8, float *q8s, hipStream_t s) {
    quant_rows_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(bin), H, q8, q8s);
    moe_gate_up_i8_kernel<<<dim3(max_tiles, FFN / 64), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs), reinterpret_cast<const uint4 *>(uw),
        reinterpret_cast<const __half *>(us), counts, offsets, tile_e, tile_m0, ntiles, pair_tok, q8, q8s,
        reinterpret_cast<__half *>(h));
    const int n_pairs = T * TOP_K;  // upper bound; rows past the real pair count are never read
    quant_rows_kernel<<<n_pairs, 256, 0, s>>>(reinterpret_cast<const __half *>(h), FFN, q8, q8s);
    moe_down_i8_kernel<<<dim3(max_tiles, H / 128), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), counts, offsets, tile_e, tile_m0,
        ntiles, q8, q8s, reinterpret_cast<__half *>(yp));
}

}  // namespace qw::gpu
