// Row parameters and per-shard results of GPU sampling (kernels/sampling.hpp),
// shared with host code.
#pragma once

#include <cstdint>

namespace qw::gpu {

constexpr int SAMPLE_CAND = 256;  // candidates per row and rank

// Per-row parameters (device table, one per decode row).
struct SampleRow {
    float inv_t;       // 1 / temperature; 0: greedy (no normalizer or Gumbel draw)
    float presence;    // on generated tokens
    float frequency;   // on generated tokens, per occurrence
    float repetition;  // on prompt + generated tokens (1: off)
    uint64_t key;      // Gumbel noise key
    int32_t slot;      // penalty counts of this slot
    int32_t first;     // first row of the row's run (its request's rows)
    int32_t token;     // the row's input token: generated, it counts for this row and later ones of the run
    int32_t pad;
};

// Per row and rank.
struct SampleOut {
    float lmax;     // max penalized logit of the shard
    float sumexp;   // sum over the shard of exp((L - lmax) * inv_t) (inv_t > 0)
    float g_score;  // Gumbel draw: max of L * inv_t + g over the shard (inv_t > 0)
    int32_t g_idx;  // its token (global id)
    float g_raw;    // its raw logit
    int32_t n;      // candidates
    float cand_l[SAMPLE_CAND];    // penalized logits, descending (ties: lower id first)
    float cand_raw[SAMPLE_CAND];  // raw logits
    int32_t cand_idx[SAMPLE_CAND];  // global token ids
};

}  // namespace qw::gpu
