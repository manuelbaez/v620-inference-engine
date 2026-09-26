// Token sampling from a row of logits (OpenAI / vLLM sampling parameters).
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "kernels/sampling_types.hpp"

namespace qw {

struct SamplingParams {
    float temperature = 1.f;  // 0 = greedy
    float top_p = 1.f;
    int top_k = 0;  // 0 = off
    float min_p = 0.f;
    float presence_penalty = 0.f;  // on generated tokens (OpenAI semantics)
    float frequency_penalty = 0.f;
    float repetition_penalty = 1.f;  // on prompt + generated tokens (HF semantics)
    uint64_t seed = 0;               // 0 = nondeterministic
};

// log(sum(exp(raw))) over the vocabulary
float log_sum_exp(const float *raw);

// Samples a token from raw logits [VOCAB]. The sequence so far is
// hist[0, hist_len) (prompt, then generated tokens from prompt_end on), for
// the penalties; a nonzero seed makes the draw a function of (seed, position).
// logprob (optional) receives the token's log-probability under the raw
// distribution; lse_known is the row's log-sum-exp if already known (NaN:
// computed here).
int32_t sample_token(const float *raw, const SamplingParams &p, const std::vector<int32_t> &hist, size_t hist_len,
                     int64_t prompt_end, std::mt19937_64 &rng, float *logprob, float lse_known = NAN);

// sample_token() from the GPU sampling results of one row (out[r] per vocab
// shard r, kernels/sampling.hpp): the same draw from the same distribution,
// from the shards' top candidates and normalizers (untruncated: their
// Gumbel-max draws). Returns -1 when the candidates do not settle it (a
// nucleus or min_p set wider than them): sample the row's full logits then.
// lse: the row's log-sum-exp of the raw logits (for logprob).
int32_t sample_candidates(const gpu::SampleOut *out, int shards, const SamplingParams &p, size_t n_hist,
                          std::mt19937_64 &rng, float *logprob, float lse);

// The k most likely tokens and their log-probabilities.
void top_logprobs(const float *raw, int k, std::vector<int32_t> &ids, std::vector<float> &lps, float lse_known = NAN);

}  // namespace qw
