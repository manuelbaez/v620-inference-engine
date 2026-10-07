// Sparse attention layers (QSA), local shard: 6 q heads, 1 kv head; the
// indexer (compressed keys, top-512 groups beyond 2048 tokens) is replicated.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/types.hpp"

namespace qw::gpu {

// ---- prefill, one sequence
// P fp16 [T][4224]. Writes q16 [T][6][256], gate fp16 [T][1536], iq16
// [T][4][128] (normed, roped, fp16 for the score GEMM), K/V/raw_k caches (kv: VRAM and spilled parts).
// rope3: optional per-token (t, h, w) rotary positions [T][3] (multimodal
// RoPE); null: token positions.
void qsa_prep_T(const uint16_t *P, const float *qn, const float *kn, const float *iqn, int64_t start, int T,
                uint16_t *q16, uint16_t *gate, uint16_t *iq16, const KvSplit &kv, hipStream_t s,
                const int32_t *rope3 = nullptr);
// Compressed keys for every group completed inside the chunk (a group's key is
// rotated by its first token's position; rope3 as in qsa_prep_T, for groups
// starting inside the chunk).
void qsa_compress_T(const KvSplit &kv, const float *ikn, int64_t start, int T, uint16_t *ck, hipStream_t s,
                    const int32_t *rope3 = nullptr, int64_t rope_start = 0);
// From per-head raw scores sc16 [Q][4][ldsc] (fp16, = iq_h . ck[c]) to
// scores [Q][ldsc] = sum_h relu / sqrt(128); queries are positions q0..q0+Q-1.
// With c0 and n: the columns are the groups [c0, c0 + n) (a rank's shard).
void qsa_score_reduce_T(const uint16_t *sc16, int ldsc, int64_t q0, int Q, float *scores, hipStream_t s, int c0 = 0,
                        int n = -1);
// Token lists for queries q0..q0+Q-1: dense 0..pos when nb <= 512, else the
// radix-selected top-512 groups plus the tail. lists [Q][LIST_W], counts [Q].
void qsa_select_T(const float *scores, int ldsc, int64_t q0, int Q, int32_t *lists, int32_t *counts, hipStream_t s);
// Sharded selection: this rank's candidates cand [Q][512] among the groups [c0, c0 + n) (scores [Q][ldsc], column i
// = group c0 + i), and the token lists from every rank's candidates cand_all [Q][RANKS][512].
void qsa_select_local_T(const float *scores, int ldsc, int c0, int n, int64_t q0, int Q, IdxCand *cand, hipStream_t s);
void qsa_select_merge_T(const IdxCand *cand_all, int64_t q0, int Q, int32_t *lists, int32_t *counts, hipStream_t s);
// Attention for Q queries (q16/gate rows), partial scratch [Q][33][6][258].
void qsa_attend_T(const uint16_t *q16, const uint16_t *gate, const KvSplit &kv, const int32_t *lists,
                  const int32_t *counts, int Q, float *partial, uint16_t *out, hipStream_t s);

// ---- batched decode rows (layer index qi into SlotPtrs)
// proj f32 [M][4224]; writes q16 [M][1536], gate f32 [M][1536], iq f32 [M][512],
// and the slot's K/V/raw_k at each row's position.
// cache: non-null to clear the SpillCache entries of the spilled positions the rows rewrite.
void qsa_prep_B(const float *proj, const float *qn, const float *kn, const float *iqn, const SlotPtrs *tab, int qi,
                Rows rows, uint16_t *q16, float *gate, float *iq, hipStream_t s, const SpillCache *cache = nullptr);
// Compressed key for rows that complete a group (after qsa_prep_B).
void qsa_compress_B(const float *ikn, const SlotPtrs *tab, int qi, Rows rows, hipStream_t s);
// scores [M][ld]; lists [M][LIST_W]; partial [M][33][6][258]; out fp16 [M][1536]
// cache: non-null to read spilled rows through the SpillCache (and list the misses).
void qsa_attend_B(const uint16_t *q16, const float *gate, const float *iq, const SlotPtrs *tab, int qi, Rows rows,
                  float *scores, int ld, int32_t *lists, int32_t *counts, float *partial, uint16_t *out, hipStream_t s,
                  const SpillCache *cache = nullptr);
// End of a decode step: copies the rows the step's attention listed as misses into the SpillCache, and empties the list.
void spill_cache_fill(const SlotPtrs *tab, const SpillCache &cache, hipStream_t s);

// Experiment (QW_SPILL_LOCALITY): for one-row steps of a spilled slot, how recently each spilled group the row selects
// was selected before (by the slot's earlier positions), i.e. what a cache keeping the groups of the last K steps would
// hit. last [slots][QSA_LAYERS_MAX][groups] (position that last selected a group, initialised very negative); stats
// [QSA_LAYERS_MAX][LOC_STATS]: 0 groups selected, 1 of them spilled, 2.. spilled ones last selected within LOC_K[i] steps.
// Rows of runs (2+ rows of a slot) count instead, at LOC_RUN: spilled groups summed over the rows, and the distinct ones
// (run_mark: the first row's position that last marked a group).
constexpr int LOC_NK = 6;
constexpr int LOC_RUN = 2 + LOC_NK;
constexpr int LOC_STATS = LOC_RUN + 2;
void qsa_locality_B(const SlotPtrs *tab, int qi, Rows rows, const int32_t *lists, const int32_t *counts, int32_t *last,
                    int32_t *run_mark, int groups, unsigned long long *stats, hipStream_t s);

}  // namespace qw::gpu
