// Per-layer n-gram embeddings (PLE, layer 1): the gathered n-gram rows are
// projected to keys/values, gated by the residual, and passed through a
// dilated causal conv whose tail is per-sequence state.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/types.hpp"

namespace qw::gpu {

// ---- prefill: kv fp16 [T][3200]; send [T][16]
void ple_stats_T(const uint16_t *kv, const float *X, const float *nk, const float *nq, int T, float *send,
                 hipStream_t s);
// X += gv + silu(dilated_conv(norm_conv(gv))); ring by position.
void ple_apply_T(const uint16_t *kv, const float *red, const float *nc, const float *conv_w, float *ring, int64_t start,
                 int T, float *ncbuf, float *X, hipStream_t s);

// ---- batched decode rows: kv f32 [M][3200]
void ple_stats_B(const float *kv, const float *X, const float *nk, const float *nq, int M, float *send, hipStream_t s);
void ple_apply_B(const float *kv, const float *red, const float *nc, const float *conv_w, const SlotPtrs *tab,
                 Rows rows, float *ncbuf, float *X, hipStream_t s);

}  // namespace qw::gpu
