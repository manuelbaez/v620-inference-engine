#include "kernels/moe.hpp"

#include "kernels/model_common.cuh"
#include "kernels/moe_tiling.hpp"

namespace qw::gpu {

// Grouped int4 GEMM for the routed experts. Work is a device-built list of
// (expert, 32-token tile); each block computes a 32-token x BN-row output tile
// with the int4 weights dequantized into LDS once per 64-wide K step, and a
// register micro-tile per thread (4 tokens x 2 rows [x gate/up]).
// LDS: A tiles use a 72-half row stride (16-byte aligned, reads broadcast);
// B tiles a 66-half stride (33 words) so the 32 lanes of a wave, which read 32
// consecutive rows, hit 32 different banks.
constexpr int A_LD = BK + 8, B_LD = BK + 2;

__global__ void moe_tiles_kernel(const int32_t *counts, int32_t *tile_e, int32_t *tile_m0, int32_t *ntiles) {
    if (threadIdx.x != 0) return;
    int n = 0;
    for (int e = 0; e < EXP_L; ++e)
        for (int m0 = 0; m0 < counts[e]; m0 += MT) {
            tile_e[n] = e;
            tile_m0[n] = m0;
            ++n;
        }
    *ntiles = n;
}

// Dequantize one uint4 of packed int4 (32 consecutive k) * scale into LDS
// halves dst[0..32) (4-byte stores: the row stride is not 16-byte aligned).
__device__ __forceinline__ void dequant32(const uint4 &w, float scale, __half *dst) {
    const __half2 sc = __float2half2_rn(scale);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    __half2 *d2 = reinterpret_cast<__half2 *>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int sh = 0; sh < 4; ++sh) {
            const uint32_t v = nib2_to_h2(ws[i] >> (4 * sh));
            d2[i * 4 + sh] = __hmul2(*reinterpret_cast<const __half2 *>(&v), sc);
        }
}

// gate/up: grid (max tiles, FFN / 64). 256 threads: ty = tid/32 owns tokens
// ty*4..+3, tx = tid%32 owns rows tx and tx+32 (of gate and of up).
__global__ __launch_bounds__(256) void moe_gate_up_T_kernel(const uint4 *gw, const __half *gs, const uint4 *uw,
                                                            const __half *us, const int32_t *counts,
                                                            const int32_t *offsets, const int32_t *tile_e,
                                                            const int32_t *tile_m0, const int32_t *ntiles,
                                                            const int32_t *pair_tok, const uint4 *bin, __half *h) {
    constexpr int CH = H / 32;  // packed uint4 per weight row
    __shared__ __align__(16) __half As[MT][A_LD];
    __shared__ __half Bg[64][B_LD], Bu[64][B_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 64;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    // A loader: token tid/8, 8-half chunk tid%8
    const int la_tok = tid / 8, la_c = tid % 8;
    const uint4 *a_src = la_tok < mt ? bin + size_t(pair_tok[base + la_tok]) * (H / 8) : nullptr;
    // B loader: threads 0..127 gate, 128..255 up; row (tid%128)/2, half (tid%2) of the 64 k
    const bool lb_up = tid >= 128;
    const int lb_row = (tid % 128) / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * FFN + row0 + lb_row;
    const uint4 *b_src = (lb_up ? uw : gw) + wrow * CH;
    const __half *b_sc = (lb_up ? us : gs) + wrow * (H / QGROUP);
    __half *b_dst = (lb_up ? &Bu[lb_row][0] : &Bg[lb_row][0]) + lb_part * 32;

    float ag[4][2] = {}, au[4][2] = {};
    for (int k0 = 0; k0 < H; k0 += BK) {
        __syncthreads();
        *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 8]) = a_src ? a_src[k0 / 8 + la_c] : make_uint4(0, 0, 0, 0);
        dequant32(b_src[k0 / 32 + lb_part], __half2float(b_sc[k0 / QGROUP]), b_dst);
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; kk += 2) {
            uint32_t a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
            const uint32_t g0 = *reinterpret_cast<const uint32_t *>(&Bg[tx][kk]);
            const uint32_t g1 = *reinterpret_cast<const uint32_t *>(&Bg[tx + 32][kk]);
            const uint32_t u0 = *reinterpret_cast<const uint32_t *>(&Bu[tx][kk]);
            const uint32_t u1 = *reinterpret_cast<const uint32_t *>(&Bu[tx + 32][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ag[i][0] = dot2(a[i], g0, ag[i][0]);
                ag[i][1] = dot2(a[i], g1, ag[i][1]);
                au[i][0] = dot2(a[i], u0, au[i][0]);
                au[i][1] = dot2(a[i], u1, au[i][1]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            h[size_t(base + tok) * FFN + row0 + tx + 32 * j] = __float2half(silu(ag[i][j]) * au[i][j]);
    }
}

// down: grid (max tiles, H / 128). tx owns rows tx, +32, +64, +96.
__global__ __launch_bounds__(256) void moe_down_T_kernel(const uint4 *dw, const __half *ds, const int32_t *counts,
                                                         const int32_t *offsets, const int32_t *tile_e,
                                                         const int32_t *tile_m0, const int32_t *ntiles, const uint4 *h,
                                                         __half *yp) {
    constexpr int CH = FFN / 32;  // 20
    __shared__ __align__(16) __half As[MT][A_LD];
    __shared__ __half Bs[128][B_LD];
    const int tile = blockIdx.x;
    if (tile >= *ntiles) return;
    const int e = tile_e[tile], m0 = tile_m0[tile];
    const int mt = min(MT, counts[e] - m0), base = offsets[e] + m0;
    const int row0 = blockIdx.y * 128;
    const int tid = threadIdx.x, ty = tid / 32, tx = tid % 32;
    const int la_tok = tid / 8, la_c = tid % 8;
    const uint4 *a_src = la_tok < mt ? h + size_t(base + la_tok) * (FFN / 8) : nullptr;
    const int lb_row = tid / 2, lb_part = tid % 2;
    const size_t wrow = size_t(e) * H + row0 + lb_row;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < FFN; k0 += BK) {
        __syncthreads();
        *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 8]) = a_src ? a_src[k0 / 8 + la_c] : make_uint4(0, 0, 0, 0);
        dequant32(dw[wrow * CH + k0 / 32 + lb_part], __half2float(ds[wrow * (FFN / QGROUP) + k0 / QGROUP]),
                  &Bs[lb_row][lb_part * 32]);
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; kk += 2) {
            uint32_t a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const uint32_t *>(&As[ty * 4 + i][kk]);
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const uint32_t *>(&Bs[tx + 32 * j][kk]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = dot2(a[i], b[j], acc[i][j]);
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

void moe_experts_T(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us, const uint32_t *dw,
                   const uint16_t *ds, const int32_t *counts, const int32_t *offsets, const int32_t *pair_tok,
                   const uint16_t *bin, int T, int32_t *tiles, uint16_t *h, uint16_t *yp, hipStream_t s, int8_t *q8,
                   float *q8s) {
    const int max_tiles = T * TOP_K / MT + EXP_L;
    int32_t *tile_e = tiles, *tile_m0 = tiles + max_tiles, *ntiles = tiles + 2 * max_tiles;
    moe_tiles_kernel<<<1, 64, 0, s>>>(counts, tile_e, tile_m0, ntiles);
    if (q8) {  // W4A8. q8: >= T*10*640 bytes, q8s: >= T*10*5 floats (per-128 group scales)
        moe_experts_w4a8(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, bin, T, tile_e, tile_m0, ntiles, max_tiles,
                         h, yp, q8, q8s, s);
        return;
    }
    moe_gate_up_T_kernel<<<dim3(max_tiles, FFN / 64), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs), reinterpret_cast<const uint4 *>(uw),
        reinterpret_cast<const __half *>(us), counts, offsets, tile_e, tile_m0, ntiles, pair_tok,
        reinterpret_cast<const uint4 *>(bin), reinterpret_cast<__half *>(h));
    moe_down_T_kernel<<<dim3(max_tiles, H / 128), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), counts, offsets, tile_e, tile_m0,
        ntiles, reinterpret_cast<const uint4 *>(h), reinterpret_cast<__half *>(yp));
}

// Small batches (decode): one wave per (pair, output row), like the
// single-token decode kernels. Weight reads are GEMV-shaped (no tile padding);

// tokens sharing an expert re-read its rows, mostly from L2.
__device__ __forceinline__ int pair_expert(const int32_t *offsets, int p) {
    int lo = 0, hi = EXP_L;  // largest e with offsets[e] <= p (skips empty experts)
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] <= p)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__global__ __launch_bounds__(256) void moe_gate_up_P_kernel(const uint4 *__restrict__ gw, const __half *__restrict__ gs,
                                                            const uint4 *__restrict__ uw, const __half *__restrict__ us,
                                                            const int32_t *__restrict__ offsets,
                                                            const int32_t *__restrict__ pair_tok,
                                                            const uint4 *__restrict__ bin, __half *__restrict__ h) {
    constexpr int CH = H / 32;
    const int p = blockIdx.y;
    if (p >= offsets[EXP_L]) return;
    const int lane = threadIdx.x % WAVE, row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int e = pair_expert(offsets, p);
    QW_DCHECK(offsets[EXP_L] <= 32 * TOP_K && e >= 0 && e < EXP_L && offsets[e] <= p && p < offsets[e + 1]);
    const size_t rbase = size_t(e) * FFN + row;
    const uint4 *g = gw + rbase * CH, *u = uw + rbase * CH;
    const __half *gsc = gs + rbase * (H / QGROUP), *usc = us + rbase * (H / QGROUP);
    QW_DCHECK(pair_tok[p] >= 0 && pair_tok[p] < 32);
    const uint4 *x = bin + size_t(pair_tok[p]) * (H / 8);
    float ag = 0.f, au = 0.f;
    for (int c = lane; c < CH; c += WAVE) {
        const uint4 g0 = g[c], u0 = u[c];
        const uint4 *xc = x + c * 4;
        const int grp = c / (QGROUP / 32);
        ag += __half2float(gsc[grp]) * dot_int4x32(g0, xc, 0.f);
        au += __half2float(usc[grp]) * dot_int4x32(u0, xc, 0.f);
    }
    ag = wave_sum(ag);
    au = wave_sum(au);
    if (lane == 0) h[size_t(p) * FFN + row] = __float2half(silu(ag) * au);
}

__global__ __launch_bounds__(256) void moe_down_P_kernel(const uint4 *__restrict__ dw, const __half *__restrict__ ds,
                                                         const int32_t *__restrict__ offsets,
                                                         const uint4 *__restrict__ h, __half *__restrict__ yp) {
    constexpr int CH = FFN / 32;  // 20 uint4 per row
    const int p = blockIdx.y;
    if (p >= offsets[EXP_L]) return;
    const int lane = threadIdx.x % WAVE, row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int e = pair_expert(offsets, p);
    QW_DCHECK(offsets[EXP_L] <= 32 * TOP_K && e >= 0 && e < EXP_L && offsets[e] <= p && p < offsets[e + 1]);
    const size_t rbase = size_t(e) * H + row;
    float acc = 0.f;
    if (lane < CH)
        acc = __half2float(ds[rbase * (FFN / QGROUP) + lane / (QGROUP / 32)]) *
              dot_int4x32(dw[rbase * CH + lane], h + size_t(p) * (FFN / 8) + lane * 4, 0.f);
    acc = wave_sum(acc);
    if (lane == 0) yp[size_t(p) * H + row] = __float2half(acc);
}

void moe_experts_P(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us, const uint32_t *dw,
                   const uint16_t *ds, const int32_t *offsets, const int32_t *pair_tok, const uint16_t *bin, int T,
                   uint16_t *h, uint16_t *yp, hipStream_t s) {
    moe_gate_up_P_kernel<<<dim3(FFN / 8, T * TOP_K), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs), reinterpret_cast<const uint4 *>(uw),
        reinterpret_cast<const __half *>(us), offsets, pair_tok, reinterpret_cast<const uint4 *>(bin),
        reinterpret_cast<__half *>(h));
    moe_down_P_kernel<<<dim3(H / 8, T * TOP_K), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), offsets,
        reinterpret_cast<const uint4 *>(h), reinterpret_cast<__half *>(yp));
}

}  // namespace qw::gpu
