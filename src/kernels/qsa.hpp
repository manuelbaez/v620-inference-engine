// Sparse attention layers (QSA), local shard: 6 q heads, 1 kv head; the
// indexer (compressed keys, top-512 groups beyond 2048 tokens) is replicated.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/types.hpp"

namespace qw::gpu {

// ---- prefill, one sequence
// P fp16 [T][4224]. Writes q16 [T][6][256], gate fp16 [T][1536], iq16
// [T][4][128] (normed, roped, fp16 for the score GEMM), K/V/raw_k caches.
// rope3: optional per-token (t, h, w) rotary positions [T][3] (multimodal
// RoPE); null: token positions.
void qsa_prep_T(const uint16_t *P, const float *qn, const float *kn, const float *iqn, int64_t start, int T,
                uint16_t *q16, uint16_t *gate, uint16_t *iq16, uint16_t *K, uint16_t *V, float *raw_k, hipStream_t s,
                const int32_t *rope3 = nullptr);
// Compressed keys for every group completed inside the chunk (a group's key is
// rotated by its first token's position; rope3 as in qsa_prep_T, for groups
// starting inside the chunk).
void qsa_compress_T(const float *raw_k, const float *ikn, int64_t start, int T, uint16_t *ck, hipStream_t s,
                    const int32_t *rope3 = nullptr, int64_t rope_start = 0);
// From per-head raw scores sc16 [Q][4][ldsc] (fp16, = iq_h . ck[c]) to
// scores [Q][ldsc] = sum_h relu / sqrt(128); queries are positions q0..q0+Q-1.
void qsa_score_reduce_T(const uint16_t *sc16, int ldsc, int64_t q0, int Q, float *scores, hipStream_t s);
// Token lists for queries q0..q0+Q-1: dense 0..pos when nb <= 512, else the
// radix-selected top-512 groups plus the tail. lists [Q][LIST_W], counts [Q].
void qsa_select_T(const float *scores, int ldsc, int64_t q0, int Q, int32_t *lists, int32_t *counts, hipStream_t s);
// Attention for Q queries (q16/gate rows), partial scratch [Q][33][6][258].
void qsa_attend_T(const uint16_t *q16, const uint16_t *gate, const uint16_t *K, const uint16_t *V, const int32_t *lists,
                  const int32_t *counts, int Q, float *partial, uint16_t *out, hipStream_t s);

// ---- batched decode rows (layer index qi into SlotPtrs)
// proj f32 [M][4224]; writes q16 [M][1536], gate f32 [M][1536], iq f32 [M][512],
// and the slot's K/V/raw_k at each row's position.
void qsa_prep_B(const float *proj, const float *qn, const float *kn, const float *iqn, const SlotPtrs *tab, int qi,
                Rows rows, uint16_t *q16, float *gate, float *iq, hipStream_t s);
// Compressed key for rows that complete a group (after qsa_prep_B).
void qsa_compress_B(const float *ikn, const SlotPtrs *tab, int qi, Rows rows, hipStream_t s);
// scores [M][ld]; lists [M][LIST_W]; partial [M][33][6][258]; out fp16 [M][1536]
void qsa_attend_B(const uint16_t *q16, const float *gate, const float *iq, const SlotPtrs *tab, int qi, Rows rows,
                  float *scores, int ld, int32_t *lists, int32_t *counts, float *partial, uint16_t *out, hipStream_t s);

}  // namespace qw::gpu
