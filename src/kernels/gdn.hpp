// Gated delta-net (linear attention) layers, local shard: 4 k-heads, 12 v-heads.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/types.hpp"

namespace qw::gpu {

// ---- prefill, one sequence
// P fp16 [T][4120] = qkv 2560 | z 1536 | b 12 | a 12.
// Writes post-conv, normalized q,k and v (fp32 [T][2560]) and per-token
// decay/beta [T][12][2]; reads the previous pre-conv rows from the ring
// (indexed by position).
void gdn_conv_T(const uint16_t *P, const float *ring, const float *conv_w, const float *A_log, const float *dt_bias,
                int64_t start, int T, float *qkv, float *gb, hipStream_t s);
// Copies the chunk's last pre-conv rows into the ring. Captures in (start,
// start + T] first get the ring as it stands at their position.
void gdn_ring_update_T(const uint16_t *P, float *ring, int64_t start, int T, hipStream_t s,
                       const Captures &cap = {});
// Sequential delta rule over the chunk, parallel over (head, 32-column slice):
// o_raw fp32 [T][1536], state S [12][128][128] updated in place. Captures in
// (start, start + T] also get the state at their position.
void gdn_scan_T(const float *qkv, const float *gb, float *S, int T, float *o_raw, hipStream_t s, int64_t start = 0,
                const Captures &cap = {});
// out[t][h*128+j] = fp16(rmsnorm(o_raw[t][h]) * w[j] * sigmoid(z[t][h*128+j]))
void gdn_norm_T(const float *o_raw, const uint16_t *P, const float *norm_w, int T, uint16_t *out, hipStream_t s);

// ---- batched decode rows (layer index gi). proj f32 [M][4120].
// qkv f32 [M][2560] (post-conv, normalized), gb [M][12][2]
void gdn_conv_B(const float *proj, const float *conv_w, const float *A_log, const float *dt_bias, const SlotPtrs *tab,
                int gi, Rows rows, float *qkv, float *gb, hipStream_t s);
// ring[slot][pos % GDN_RING] = pre-conv row (after gdn_conv_B)
void gdn_ring_B(const float *proj, const SlotPtrs *tab, int gi, Rows rows, hipStream_t s);
// Delta rule per run of rows (run_start/run_len/run_slot, n_runs <= M). With
// save != null, the state after each row is also written to save[row].
void gdn_scan_B(const float *qkv, const float *gb, const SlotPtrs *tab, int gi, const int32_t *run_start,
                const int32_t *run_len, const int32_t *run_slot, int M, float *o_raw, float *save, hipStream_t s);
void gdn_norm_B(const float *o_raw, const float *proj, const float *norm_w, int M, uint16_t *out, hipStream_t s);

}  // namespace qw::gpu
