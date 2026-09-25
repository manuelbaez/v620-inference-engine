// Token sampling from a row of logits (OpenAI / vLLM sampling parameters).
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

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

// The k most likely tokens and their log-probabilities.
void top_logprobs(const float *raw, int k, std::vector<int32_t> &ids, std::vector<float> &lps, float lse_known = NAN);

}  // namespace qw
