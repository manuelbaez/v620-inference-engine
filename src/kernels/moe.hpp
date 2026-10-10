// Mixture of experts, expert-parallel: each rank owns 128 of the 512 routed
// experts (int4 g128) and a quarter of the shared expert's columns.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "core/config.hpp"

namespace qw::gpu {

// ---- routing (moe_route.hip)
// Per token: top-10 of softmax(logits), renormalized, local ones kept.
// tok_e/tok_w [T][10] (local expert or -1), counts[128] (atomic, pre-zeroed).
// use: if not null, the (token, expert) pairs that fell to this rank are added to *use (a running count of the
// rank's routed-expert work: Engine::moe_use).
void moe_route_T(const float *logits, int T, int first, int32_t *tok_e, float *tok_w, int32_t *counts, hipStream_t s,
                 unsigned long long *use = nullptr);
// offsets = exclusive scan(counts); slot of each (token, j) pair; pair_tok[slot] = t.
void moe_scatter_T(const int32_t *tok_e, const int32_t *counts, int T, int32_t *offsets, int32_t *fill,
                   int32_t *tok_slot, int32_t *pair_tok, hipStream_t s);

// Decode rows (M <= 32): routing, counts, offsets [129] and pair slots in one
// kernel (the per-pair outputs of moe_route_T + moe_scatter_T, no pre-zeroing).
void moe_route_B(const float *logits, int M, int first, int32_t *counts, int32_t *offsets, float *tok_w,
                 int32_t *tok_slot, int32_t *pair_tok, hipStream_t s,
                 unsigned long long *use = nullptr);

// ---- routed experts (moe_experts.hip, moe_w4a8.hip)
// For every (token, expert) pair: h[slot] = silu(gate x) * up x,
// yp[slot] = down h (fp16). tiles: scratch of 2*(T*10/32 + 128) + 1 ints.
// With q8 (int8 scratch >= T*10*640 bytes) and q8s (>= T*10*5 floats), runs
// W4A8 on v_dot4_i32_i8 with per-128-group activation scales (lossy; see DESIGN).
// group: QGROUP (an fp16 scale per 128 inputs) or QGROUP_Z (a scale and zero point word per 32 inputs,
// core/config.hpp; not with q8).
void moe_experts_T(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us, const uint32_t *dw,
                   const uint16_t *ds, const int32_t *counts, const int32_t *offsets, const int32_t *pair_tok,
                   const uint16_t *bin, int T, int32_t *tiles, uint16_t *h, uint16_t *yp, hipStream_t s,
                   int8_t *q8 = nullptr, float *q8s = nullptr, int group = cfg::QGROUP);
// Same outputs as moe_experts_T (fp16 only) with GEMV-shaped kernels, one
// wave per (pair, row): faster for small T (decode batches).
void moe_experts_P(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us, const uint32_t *dw,
                   const uint16_t *ds, const int32_t *offsets, const int32_t *pair_tok, const uint16_t *bin, int T,
                   uint16_t *h, uint16_t *yp, hipStream_t s, int group = cfg::QGROUP);

// ---- shared expert and combine (moe_route.hip)
// h[t][i] = fp16(silu(v[t][i]) * v[t][n+i] * sigmoid(v[t][2n])), v rows of ldv
void swiglu_T(const float *v, int ldv, int n, int T, uint16_t *h, hipStream_t s);
// out[t][n] = fp16( sum_j tok_w[t][j] * yp[tok_slot[t][j]][n] + shared[t][n] )
void moe_combine_T(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w, const uint16_t *shared, int T,
                   uint16_t *out, hipStream_t s);
// Decode rows: out f32 [M][2560] = shared + sum_j w * yp[slot_j]
void moe_combine_B(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w, const float *shared, int M,
                   float *out, hipStream_t s);

}  // namespace qw::gpu
