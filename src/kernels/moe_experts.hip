#include "kernels/moe.hpp"

#include "core/common.hpp"
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

// A group's scale and the fp16 value its nibbles are offset by (nib2_to_h2). G = QGROUP: the word is the fp16 scale,
// the nibbles are value + 8. G = QGROUP_Z: scale and zero point share the word (core/config.hpp).
struct GroupScale {
    _Float16 scale, off;
};
template <int G>
__device__ __forceinline__ GroupScale group_scale(const __half *word) {
    const uint16_t b = *reinterpret_cast<const uint16_t *>(word);
    if (G == QGROUP) return {__builtin_bit_cast(_Float16, b), (_Float16)1032.0f};
    // 0x6400 | n is 1024 + n in fp16
    return {__builtin_bit_cast(_Float16, uint16_t(b & QZ_SCALE_MASK)),
            __builtin_bit_cast(_Float16, uint16_t(0x6400 | ((b >> 12) & 8) | (b & 7)))};
}

// Dequantize one uint4 of packed int4 (32 consecutive k) * scale into LDS
// halves dst[0..32) (4-byte stores: the row stride is not 16-byte aligned).
__device__ __forceinline__ void dequant32(const uint4 &w, const GroupScale gs, __half *dst) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const h2 sc = {gs.scale, gs.scale};
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    uint32_t *d2 = reinterpret_cast<uint32_t *>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int sh = 0; sh < 4; ++sh) {
            const uint32_t v = nib2_to_h2(ws[i] >> (4 * sh), gs.off);
            d2[i * 4 + sh] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, v) * sc);
        }
}

// NVFP4 (QGROUP_F4). E2M1 pairs in the low nibbles of each 16-bit half (bits 0-3, 16-19, as nib2_to_h2) -> fp16 of
// value x 2^-14, exactly: the sign to bit 15, exponent and mantissa bit to bits 9-11 (exponent 0 reads as subnormal).
__device__ __forceinline__ uint32_t fp4_2_to_h2(uint32_t w) {
    return ((w & 0x00070007u) << 9) | ((w & 0x00080008u) << 12);
}
// An E4M3 scale byte -> fp16 of value x 2^7 (exact: 3 mantissa bits, 448 x 128 < 65504). Times an E2M1 value x 2^-14
// that gives value x scale x 2^-7, exact in fp16 too (at most 5 significant bits, >= 2^-17); the 2^7 is in the FP32
// scale per matrix (Rank weights `eglob`). Without a branch: the byte's bits moved into fp16's read as value x 2^-8,
// exactly (E4M3 subnormals land on fp16 subnormals; gemv.hip e4m3x4_to_h2x2), then x 2^15, exact.
__device__ __forceinline__ _Float16 e4m3_x128(uint32_t b) {
    const _Float16 v = __builtin_bit_cast(_Float16, uint16_t(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)));
    return v * (_Float16)32768.0f;
}
__device__ __forceinline__ void dequant32_f4(const uint4 &w, uint32_t word, __half *dst) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const _Float16 s0 = e4m3_x128(word & 0xff), s1 = e4m3_x128((word >> 8) & 0xff);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    uint32_t *d2 = reinterpret_cast<uint32_t *>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const h2 sc = i < 2 ? h2{s0, s0} : h2{s1, s1};  // words 0-1: inputs 0-15, words 2-3: 16-31
#pragma unroll
        for (int sh = 0; sh < 4; ++sh)
            d2[i * 4 + sh] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, fp4_2_to_h2(ws[i] >> (4 * sh))) * sc);
    }
}
constexpr bool is_f4(int G) { return G == QGROUP_F4 || G == QGROUP_F4P; }

// QGROUP_F4P: the two E4M3 codes of row `row`'s 32-input group k32 (as a QGROUP_F4 word) from the packed scales.
__device__ __forceinline__ uint32_t f4_code(const F4Packed &p, uint32_t row, uint32_t idx, uint32_t nib, uint32_t base) {
    if (nib != 15) return base + nib;
    const uint32_t key = ((row << 8) | idx) + 1, mask = (1u << (32 - p.shift)) - 1;
    uint32_t s = (key * 0x9E3779B1u) >> p.shift;
    while (p.key[s] != key) s = (s + 1) & mask;
    return p.val[s];
}
__device__ __forceinline__ uint32_t f4_word(const F4Packed &p, uint32_t row, uint32_t k32, uint32_t K32) {
    const uint32_t b = p.base[row], o = p.off[size_t(row) * K32 + k32];
    return f4_code(p, row, 2 * k32, o & 15, b) | (f4_code(p, row, 2 * k32 + 1, o >> 4, b) << 8);
}
// A QGROUP_F4 / QGROUP_F4P scale word of row `row`, 32-input group k32 (K32 groups per row; word: the stored ones).
template <int G>
__device__ __forceinline__ uint32_t f4_scale_word(const __half *word, const F4Packed &p, uint32_t row, uint32_t k32,
                                                  uint32_t K32) {
    if constexpr (G == QGROUP_F4P) return f4_word(p, row, k32, K32);
    return *reinterpret_cast<const uint16_t *>(word);
}

// One uint4 of weights (32 inputs) into LDS halves, with the group's scale word.
template <int G>
__device__ __forceinline__ void dequant32_g(const uint4 &w, const __half *word, __half *dst, const F4Packed &p = {},
                                            uint32_t row = 0, uint32_t k32 = 0, uint32_t K32 = 0) {
    if constexpr (is_f4(G))
        dequant32_f4(w, f4_scale_word<G>(word, p, row, k32, K32), dst);
    else
        dequant32(w, group_scale<G>(word), dst);
}
// ... and against 32 fp16 activations (the P kernels), scale included.
template <int G>
__device__ __forceinline__ float dot32_g(const uint4 &w, const uint4 *x, const __half *word, const F4Packed &p = {},
                                         uint32_t row = 0, uint32_t k32 = 0, uint32_t K32 = 0) {
    if constexpr (is_f4(G)) {
        const uint32_t b = f4_scale_word<G>(word, p, row, k32, K32), ws[4] = {w.x, w.y, w.z, w.w};
        float a[2] = {0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 xv = x[i];
            float &acc = a[i / 2];
            acc = dot2(fp4_2_to_h2(ws[i]), xv.x, acc);
            acc = dot2(fp4_2_to_h2(ws[i] >> 4), xv.y, acc);
            acc = dot2(fp4_2_to_h2(ws[i] >> 8), xv.z, acc);
            acc = dot2(fp4_2_to_h2(ws[i] >> 12), xv.w, acc);
        }
        return a[0] * float(e4m3_x128(b & 0xff)) + a[1] * float(e4m3_x128(b >> 8));
    } else {
        const GroupScale gs = group_scale<G>(word);
        return float(gs.scale) * dot_int4x32(w, x, 0.f, gs.off);
    }
}

// gate/up: grid (max tiles, FFN / 64). 256 threads: ty = tid/32 owns tokens
// ty*4..+3, tx = tid%32 owns rows tx and tx+32 (of gate and of up).
template <int G>
__global__ __launch_bounds__(256) void moe_gate_up_T_kernel(const uint4 *gw, const __half *gs, const uint4 *uw,
                                                            const __half *us, const int32_t *counts,
                                                            const int32_t *offsets, const int32_t *tile_e,
                                                            const int32_t *tile_m0, const int32_t *ntiles,
                                                            const int32_t *pair_tok, const uint4 *bin, __half *h,
                                                            const float *glob, F4Packed pg, F4Packed pu) {
    constexpr int CH = H / 32;  // packed uint4 per weight row
    constexpr int S = scale_span(G);
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
    const __half *b_sc = G == QGROUP_F4P ? nullptr : (lb_up ? us : gs) + wrow * (H / S);
    const F4Packed &b_pk = lb_up ? pu : pg;
    __half *b_dst = (lb_up ? &Bu[lb_row][0] : &Bg[lb_row][0]) + lb_part * 32;

    float ag[4][2] = {}, au[4][2] = {};
    for (int k0 = 0; k0 < H; k0 += BK) {
        __syncthreads();
        *reinterpret_cast<uint4 *>(&As[la_tok][la_c * 8]) = a_src ? a_src[k0 / 8 + la_c] : make_uint4(0, 0, 0, 0);
        dequant32_g<G>(b_src[k0 / 32 + lb_part], b_sc + (k0 + 32 * lb_part) / S, b_dst, b_pk, uint32_t(wrow),
                       uint32_t((k0 + 32 * lb_part) / 32), H / 32);
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
    const float fg = is_f4(G) ? glob[e] : 1.f, fu = is_f4(G) ? glob[EXP_L + e] : 1.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            h[size_t(base + tok) * FFN + row0 + tx + 32 * j] = __float2half(silu(ag[i][j] * fg) * (au[i][j] * fu));
    }
}

// down: grid (max tiles, H / 128). tx owns rows tx, +32, +64, +96.
template <int G>
__global__ __launch_bounds__(256) void moe_down_T_kernel(const uint4 *dw, const __half *ds, const int32_t *counts,
                                                         const int32_t *offsets, const int32_t *tile_e,
                                                         const int32_t *tile_m0, const int32_t *ntiles, const uint4 *h,
                                                         __half *yp, const float *glob, F4Packed pd) {
    constexpr int CH = FFN / 32;  // 20
    constexpr int S = scale_span(G);
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
        dequant32_g<G>(dw[wrow * CH + k0 / 32 + lb_part],
                       G == QGROUP_F4P ? nullptr : ds + wrow * (FFN / S) + (k0 + 32 * lb_part) / S,
                       &Bs[lb_row][lb_part * 32], pd, uint32_t(wrow), uint32_t((k0 + 32 * lb_part) / 32), FFN / 32);
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
    const float fd = is_f4(G) ? glob[2 * EXP_L + e] : 1.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tok = ty * 4 + i;
        if (tok >= mt) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) yp[size_t(base + tok) * H + row0 + tx + 32 * j] = __float2half(acc[i][j] * fd);
    }
}

template <int G>
static void experts_T(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                      const uint32_t *dw, const uint16_t *ds, const int32_t *counts, const int32_t *offsets,
                      const int32_t *pair_tok, const uint16_t *bin, int max_tiles, const int32_t *tile_e,
                      const int32_t *tile_m0, const int32_t *ntiles, uint16_t *h, uint16_t *yp, hipStream_t s,
                      const float *glob, const F4Packed *pk) {
    const F4Packed none{};
    moe_gate_up_T_kernel<G><<<dim3(max_tiles, FFN / 64), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs), reinterpret_cast<const uint4 *>(uw),
        reinterpret_cast<const __half *>(us), counts, offsets, tile_e, tile_m0, ntiles, pair_tok,
        reinterpret_cast<const uint4 *>(bin), reinterpret_cast<__half *>(h), glob, pk ? pk[0] : none, pk ? pk[1] : none);
    moe_down_T_kernel<G><<<dim3(max_tiles, H / 128), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), counts, offsets, tile_e, tile_m0,
        ntiles, reinterpret_cast<const uint4 *>(h), reinterpret_cast<__half *>(yp), glob, pk ? pk[2] : none);
}

void moe_experts_T(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us, const uint32_t *dw,
                   const uint16_t *ds, const int32_t *counts, const int32_t *offsets, const int32_t *pair_tok,
                   const uint16_t *bin, int T, int32_t *tiles, uint16_t *h, uint16_t *yp, hipStream_t s, int8_t *q8,
                   float *q8s, int group, const float *glob, const F4Packed *packed) {
    QW_CHECK(group == QGROUP || ((group == QGROUP_Z || is_f4(group)) && !q8), "moe_experts_T: unsupported expert format");
    QW_CHECK(!is_f4(group) || glob, "moe_experts_T: NVFP4 experts need their scales per matrix");
    QW_CHECK(group != QGROUP_F4P || packed, "moe_experts_T: packed NVFP4 scales missing");
    const int max_tiles = T * TOP_K / MT + EXP_L;
    int32_t *tile_e = tiles, *tile_m0 = tiles + max_tiles, *ntiles = tiles + 2 * max_tiles;
    moe_tiles_kernel<<<1, 64, 0, s>>>(counts, tile_e, tile_m0, ntiles);
    if (q8) {  // W4A8. q8: >= T*10*640 bytes, q8s: >= T*10*5 floats (per-128 group scales)
        moe_experts_w4a8(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, bin, T, tile_e, tile_m0, ntiles, max_tiles,
                         h, yp, q8, q8s, s);
        return;
    }
    if (group == QGROUP)
        experts_T<QGROUP>(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, bin, max_tiles, tile_e, tile_m0, ntiles, h,
                          yp, s, glob, packed);
    else if (group == QGROUP_Z)
        experts_T<QGROUP_Z>(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, bin, max_tiles, tile_e, tile_m0, ntiles,
                            h, yp, s, glob, packed);
    else if (group == QGROUP_F4)
        experts_T<QGROUP_F4>(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, bin, max_tiles, tile_e, tile_m0, ntiles,
                             h, yp, s, glob, packed);
    else
        experts_T<QGROUP_F4P>(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, bin, max_tiles, tile_e, tile_m0, ntiles,
                              h, yp, s, glob, packed);
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

template <int G>
__global__ __launch_bounds__(256) void moe_gate_up_P_kernel(const uint4 *__restrict__ gw, const __half *__restrict__ gs,
                                                            const uint4 *__restrict__ uw, const __half *__restrict__ us,
                                                            const int32_t *__restrict__ offsets,
                                                            const int32_t *__restrict__ pair_tok,
                                                            const uint4 *__restrict__ bin, __half *__restrict__ h,
                                                            const float *__restrict__ glob, F4Packed pg, F4Packed pu) {
    constexpr int CH = H / 32;
    constexpr int S = scale_span(G);
    const int p = blockIdx.y;
    if (p >= offsets[EXP_L]) return;
    const int lane = threadIdx.x % WAVE, row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int e = pair_expert(offsets, p);
    QW_DCHECK(offsets[EXP_L] <= 32 * TOP_K && e >= 0 && e < EXP_L && offsets[e] <= p && p < offsets[e + 1]);
    const size_t rbase = size_t(e) * FFN + row;
    const uint4 *g = gw + rbase * CH, *u = uw + rbase * CH;
    const __half *gsc = G == QGROUP_F4P ? nullptr : gs + rbase * (H / S),
                 *usc = G == QGROUP_F4P ? nullptr : us + rbase * (H / S);
    QW_DCHECK(pair_tok[p] >= 0 && pair_tok[p] < 32);
    const uint4 *x = bin + size_t(pair_tok[p]) * (H / 8);
    float ag = 0.f, au = 0.f;
    for (int c = lane; c < CH; c += WAVE) {
        const uint4 g0 = g[c], u0 = u[c];
        const uint4 *xc = x + c * 4;
        const int grp = c / (S / 32);
        ag += dot32_g<G>(g0, xc, gsc + grp, pg, uint32_t(rbase), uint32_t(grp), H / 32);
        au += dot32_g<G>(u0, xc, usc + grp, pu, uint32_t(rbase), uint32_t(grp), H / 32);
    }
    ag = wave_sum(ag);
    au = wave_sum(au);
    if (is_f4(G)) {
        ag *= glob[e];
        au *= glob[EXP_L + e];
    }
    if (lane == 0) h[size_t(p) * FFN + row] = __float2half(silu(ag) * au);
}

template <int G>
__global__ __launch_bounds__(256) void moe_down_P_kernel(const uint4 *__restrict__ dw, const __half *__restrict__ ds,
                                                         const int32_t *__restrict__ offsets,
                                                         const uint4 *__restrict__ h, __half *__restrict__ yp,
                                                         const float *__restrict__ glob, F4Packed pd) {
    constexpr int CH = FFN / 32;  // 20 uint4 per row
    constexpr int S = scale_span(G);
    const int p = blockIdx.y;
    if (p >= offsets[EXP_L]) return;
    const int lane = threadIdx.x % WAVE, row = blockIdx.x * 8 + threadIdx.x / WAVE;
    const int e = pair_expert(offsets, p);
    QW_DCHECK(offsets[EXP_L] <= 32 * TOP_K && e >= 0 && e < EXP_L && offsets[e] <= p && p < offsets[e + 1]);
    const size_t rbase = size_t(e) * H + row;
    float acc = 0.f;
    if (lane < CH)
        acc = dot32_g<G>(dw[rbase * CH + lane], h + size_t(p) * (FFN / 8) + lane * 4,
                         G == QGROUP_F4P ? nullptr : ds + rbase * (FFN / S) + lane / (S / 32), pd, uint32_t(rbase),
                         uint32_t(lane), FFN / 32);
    acc = wave_sum(acc);
    if (is_f4(G)) acc *= glob[2 * EXP_L + e];
    if (lane == 0) yp[size_t(p) * H + row] = __float2half(acc);
}

template <int G>
static void experts_P(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                      const uint32_t *dw, const uint16_t *ds, const int32_t *offsets, const int32_t *pair_tok,
                      const uint16_t *bin, int T, uint16_t *h, uint16_t *yp, hipStream_t s, const float *glob,
                      const F4Packed *pk) {
    const F4Packed none{};
    moe_gate_up_P_kernel<G><<<dim3(FFN / 8, T * TOP_K), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(gw), reinterpret_cast<const __half *>(gs), reinterpret_cast<const uint4 *>(uw),
        reinterpret_cast<const __half *>(us), offsets, pair_tok, reinterpret_cast<const uint4 *>(bin),
        reinterpret_cast<__half *>(h), glob, pk ? pk[0] : none, pk ? pk[1] : none);
    moe_down_P_kernel<G><<<dim3(H / 8, T * TOP_K), 256, 0, s>>>(
        reinterpret_cast<const uint4 *>(dw), reinterpret_cast<const __half *>(ds), offsets,
        reinterpret_cast<const uint4 *>(h), reinterpret_cast<__half *>(yp), glob, pk ? pk[2] : none);
}

void moe_experts_P(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us, const uint32_t *dw,
                   const uint16_t *ds, const int32_t *offsets, const int32_t *pair_tok, const uint16_t *bin, int T,
                   uint16_t *h, uint16_t *yp, hipStream_t s, int group, const float *glob, const F4Packed *packed) {
    QW_CHECK(group == QGROUP || group == QGROUP_Z || (is_f4(group) && glob), "moe_experts_P: unsupported expert format");
    QW_CHECK(group != QGROUP_F4P || packed, "moe_experts_P: packed NVFP4 scales missing");
    if (group == QGROUP)
        experts_P<QGROUP>(gw, gs, uw, us, dw, ds, offsets, pair_tok, bin, T, h, yp, s, glob, packed);
    else if (group == QGROUP_Z)
        experts_P<QGROUP_Z>(gw, gs, uw, us, dw, ds, offsets, pair_tok, bin, T, h, yp, s, glob, packed);
    else if (group == QGROUP_F4)
        experts_P<QGROUP_F4>(gw, gs, uw, us, dw, ds, offsets, pair_tok, bin, T, h, yp, s, glob, packed);
    else
        experts_P<QGROUP_F4P>(gw, gs, uw, us, dw, ds, offsets, pair_tok, bin, T, h, yp, s, glob, packed);
}

}  // namespace qw::gpu
