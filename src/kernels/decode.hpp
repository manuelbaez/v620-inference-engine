// Decode-regime kernels (M <= 8 tokens). All pointers are device pointers.
// Layouts are the engine's own, produced at load time (see pack_* helpers).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace qw::gpu {

// y[m][n] = sum_k W[n][k] * x[m][k].  W fp16 [N][K] row-major, x fp16 [M][K],
// y fp32 [M][N]. K % 8 == 0, 1 <= M <= 8. With accumulate, y += instead of =.
void gemv_f16(const uint16_t *W, const uint16_t *x, float *y, int N, int K, int M, hipStream_t s,
              bool accumulate = false);

// Grouped variant (M = 1): row n uses x + (n / rows_per_group) * K, so one
// launch serves a block-diagonal matrix (the per-stream HC projections).
void gemv_f16_grouped(const uint16_t *W, const uint16_t *x, float *y, int N, int K,
                      int rows_per_group, hipStream_t s);

// int4 expert weights in the engine layout: per (expert, row), K/32 uint4
// chunks; each 32-bit word holds 8 consecutive k values with k0,k2,k4,k6 in
// bits 0,4,8,12 and k1,k3,k5,k7 in bits 16,20,24,28, stored as q+8. Scales are
// fp16 [expert][row][K/128].
// Converts one compressed-tensors row-major matrix ([rows][K/8] int32, element
// 8c+i at bits 4i) to that layout. n_words = rows * K / 8.
void pack_int4_words(const int32_t *ct, uint32_t *out, size_t n_words);

// Routed-expert gate/up for one token: h[j][r] = silu(gate_e(x)[r]) * up_e(x)[r]
// for the experts e = ids[j], j < *count (or top_k when count is null; under
// expert parallelism the count of local experts is only known on the device).
// x fp16 [2560], h fp16 [top_k][640].
void moe_gate_up(const uint32_t *gate_w, const uint16_t *gate_s, const uint32_t *up_w,
                 const uint16_t *up_s, const int32_t *ids, const int32_t *count, const uint16_t *x,
                 uint16_t *h, int top_k, hipStream_t s);

// Routed-expert down projection with the weighted combine fused in:
// y[n] (+)= sum_j weight[j] * down_{ids[j]}(h[j])[n]. y fp32 [2560].
void moe_down(const uint32_t *down_w, const uint16_t *down_s, const int32_t *ids,
              const int32_t *count, const float *weight, const uint16_t *h, float *y, int top_k,
              bool accumulate, hipStream_t s);

// One GDN decode step for one token over `n_v` value heads (48 for the whole
// layer, 12 for one TP rank). All buffers use the local layout: with
// nk = n_v / 3 key heads, qkv is [q nk*128 | k nk*128 | v n_v*128].
struct GdnStep {
    const float *qkv;         // [qkv_dim] pre-conv projection output
    float *conv_ring;         // [4][qkv_dim] ring of pre-conv rows, slot = pos % 4
    const float *conv_w;      // [qkv_dim][4]
    const float *z;           // [n_v*128]
    const float *a, *b;       // [n_v]
    const float *A_log, *dt_bias, *norm_w;  // [n_v], [n_v], [128]
    float *S;                 // [n_v][128 k][128 v] fp32 recurrent state
    uint16_t *out;            // [n_v*128] fp16 gated-norm output
    const int64_t *pos;       // device: position of this token
    int n_v;                  // value heads (multiple of 3)
};
void gdn_decode(const GdnStep &p, hipStream_t s);

}  // namespace qw::gpu
