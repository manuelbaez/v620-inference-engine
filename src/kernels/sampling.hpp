// Sampling on the GPU, per vocab shard: penalties from per-slot token counts,
// then per row the pieces the host needs to draw exactly as the CPU sampler
// does (session/sampling.cpp): the shard's max and normalizer, its top
// SAMPLE_CAND candidates in order, and a Gumbel-max draw over the whole shard
// (the untruncated distribution).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/sampling_types.hpp"

namespace qw::gpu {

// L = raw with the penalties of each row's slot: counts [2][n] per slot
// (prompt + generated, generated), plus the generated tokens of the row's run
// up to itself. raw, L: [M][n]; tok0: the shard's first token id.
void sample_penalize(const float *raw, float *L, const SampleRow *rows, int32_t *const *counts, int n, int tok0, int M,
                     hipStream_t s);
// Fills out[m] for each row from L and raw.
void sample_select(const float *L, const float *raw, const SampleRow *rows, int n, int tok0, int M, SampleOut *out,
                   hipStream_t s);
// counts[kind][t - tok0] += 1 for the tokens t of this shard; kind: all tokens
// count in [0], those with gen[i] != 0 also in [1]. (Rebuilding: memset first.)
void penalty_add(int32_t *counts, const int32_t *tokens, const uint8_t *gen, int n_tokens, int n, int tok0,
                 hipStream_t s);

}  // namespace qw::gpu
