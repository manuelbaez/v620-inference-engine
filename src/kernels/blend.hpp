// Kernels of the CacheBlend experiment (non-prefix reuse of a chunk's cached
// state, docs/DESIGN.md): the GDN state transfer of a chunk, its composition
// onto another prefix's state, and the re-rotation of cached keys.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/types.hpp"

namespace qw::gpu {

// The delta rule is affine in the incoming state: S_out = M S_in + U, with
// M = prod_t decay_t (I - beta_t k_t k_t^T) (128 x 128 per v-head). Advances
// M [12][128][128] over the chunk's tokens (the scan with v = 0); start M at
// the identity.
void gdn_transfer_T(const float *qkv, const float *gb, float *M, int T, hipStream_t s);
// M = identity for n layers of [12][128][128].
void gdn_transfer_init(float *M, int layers, hipStream_t s);
// S = S_out + M (S - S_in) for n layers of [12][128][128] (tmp: same size as S).
void gdn_compose(float *S, const float *M, const float *S_in, const float *S_out, float *tmp, int layers,
                 hipStream_t s);
// Rotates the RoPE part of cached keys K fp16 [n][256] by `delta` positions.
void rope_shift(uint16_t *K, int64_t n, int64_t delta, hipStream_t s);

}  // namespace qw::gpu
