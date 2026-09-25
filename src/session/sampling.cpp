#include "session/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "core/config.hpp"

namespace qw {

float log_sum_exp(const float *raw) {
    const int V = cfg::VOCAB;
    const float mx = *std::max_element(raw, raw + V);
    double z = 0;
    for (int i = 0; i < V; ++i) z += std::exp(double(raw[i] - mx));
    return mx + float(std::log(z));
}

int32_t sample_token(const float *raw, const SamplingParams &p, const std::vector<int32_t> &hist, size_t hist_len,
                     int64_t prompt_end, std::mt19937_64 &rng, float *logprob, float lse_known) {
    const size_t n_hist = std::min(hist_len, hist.size());
    const int V = cfg::VOCAB;
    // the log-sum-exp is only needed for the reported logprob
    const float lse = !logprob ? 0.f : std::isnan(lse_known) ? log_sum_exp(raw) : lse_known;
    auto finish = [&](int32_t tok) {
        if (logprob) *logprob = raw[tok] - lse;
        return tok;
    };
    const bool penalties = p.presence_penalty != 0.f || p.frequency_penalty != 0.f || p.repetition_penalty != 1.f;
    if (p.temperature <= 0.f && !penalties) return finish(int32_t(std::max_element(raw, raw + V) - raw));

    std::vector<float> l;  // penalized logits (only when penalties apply)
    if (penalties) {
        l.assign(raw, raw + V);
        std::unordered_map<int32_t, int> counts;
        for (size_t i = size_t(prompt_end); i < n_hist; ++i) ++counts[hist[i]];
        if (p.repetition_penalty != 1.f) {
            std::vector<bool> seen(size_t(V), false);
            for (size_t i = 0; i < n_hist; ++i) seen[size_t(hist[i])] = true;
            for (int i = 0; i < V; ++i)
                if (seen[size_t(i)])
                    l[size_t(i)] =
                        l[size_t(i)] > 0 ? l[size_t(i)] / p.repetition_penalty : l[size_t(i)] * p.repetition_penalty;
        }
        for (auto &[t, c] : counts) l[size_t(t)] -= p.presence_penalty + p.frequency_penalty * float(c);
    }
    const float *L = penalties ? l.data() : raw;
    if (p.temperature <= 0.f) return finish(int32_t(std::max_element(L, L + V) - L));

    // Weights w_i = exp((L_i - max) / T); tokens more than 30 nats below the max weigh 0.
    const float lmax = *std::max_element(L, L + V), inv_t = 1.f / p.temperature;
    auto weight = [&](int i) {
        const float z = (L[i] - lmax) * inv_t;
        return z < -30.f ? 0.0 : double(std::exp(z));
    };
    std::mt19937_64 seeded(p.seed + n_hist);
    std::mt19937_64 &g = p.seed ? seeded : rng;

    // No truncation: one pass over the vocabulary, no sort.
    if (p.top_p >= 1.f && p.top_k <= 0 && p.min_p <= 0.f) {
        double z = 0;
        for (int i = 0; i < V; ++i) z += weight(i);
        double r = std::uniform_real_distribution<double>(0, z)(g);
        for (int i = 0; i < V; ++i) {
            r -= weight(i);
            if (r <= 0) return finish(i);
        }
        return finish(int32_t(std::max_element(L, L + V) - L));
    }

    // min_p alone: tokens with weight >= min_p (the max weighs 1), one pass, no sort.
    if (p.top_p >= 1.f && p.top_k <= 0) {
        double z = 0;
        for (int i = 0; i < V; ++i)
            if (const double wi = weight(i); wi >= p.min_p) z += wi;
        double r = std::uniform_real_distribution<double>(0, z)(g);
        for (int i = 0; i < V; ++i)
            if (const double wi = weight(i); wi >= p.min_p && (r -= wi) <= 0) return finish(i);
        return finish(int32_t(std::max_element(L, L + V) - L));
    }

    // Truncation: one pass keeps the tokens within 30 nats of the max (the
    // others weigh 0) with their weights; then the top K of those by logit
    // (K = top_k, or 1024 for top_p / min_p alone, widened to all candidates if
    // the nucleus does not fit).
    thread_local std::vector<int32_t> idx;
    thread_local std::vector<float> w;
    idx.clear();
    w.resize(size_t(V));
    double z_all = 0;  // normalizer of the untruncated distribution
    for (int i = 0; i < V; ++i) {
        const float z = (L[i] - lmax) * inv_t;
        if (z < -30.f) continue;
        const float e = std::exp(z);
        w[size_t(i)] = e;
        z_all += e;
        idx.push_back(i);
    }
    auto by_logit = [&](int32_t a, int32_t b) { return L[a] > L[b] || (L[a] == L[b] && a < b); };
    const size_t n_cand = idx.size();
    size_t K = std::min(n_cand, p.top_k > 0 ? size_t(p.top_k) : size_t(1024));
    std::vector<double> prob;
    size_t keep = 0;
    for (;;) {
        if (K < n_cand) std::nth_element(idx.begin(), idx.begin() + ptrdiff_t(K), idx.end(), by_logit);
        std::sort(idx.begin(), idx.begin() + ptrdiff_t(K), by_logit);
        prob.resize(K);
        double z = 0;
        for (size_t i = 0; i < K; ++i) z += prob[i] = w[size_t(idx[i])];
        const double norm = p.top_k > 0 ? z : z_all;
        for (auto &x : prob) x /= norm;
        keep = K;
        if (p.top_p < 1.f) {
            double acc = 0;
            keep = 0;
            for (size_t i = 0; i < K && keep == 0; ++i)
                if ((acc += prob[i]) >= p.top_p) keep = i + 1;
            if (keep == 0) {
                if (p.top_k <= 0 && K < n_cand) {  // nucleus wider than K: widen
                    K = n_cand;
                    continue;
                }
                keep = K;
            }
        }
        break;
    }
    if (p.min_p > 0.f) {
        const double thr = p.min_p * prob[0];
        size_t k = 0;
        while (k < keep && prob[k] >= thr) ++k;
        keep = std::max<size_t>(k, 1);
    }
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += prob[i];
    double r = std::uniform_real_distribution<double>(0, tot)(g);
    for (size_t i = 0; i < keep; ++i) {
        r -= prob[i];
        if (r <= 0) return finish(idx[i]);
    }
    return finish(idx[keep - 1]);
}

void top_logprobs(const float *raw, int k, std::vector<int32_t> &ids, std::vector<float> &lps, float lse_known) {
    const int V = cfg::VOCAB;
    const float lse = std::isnan(lse_known) ? log_sum_exp(raw) : lse_known;
    std::vector<int32_t> idx(static_cast<size_t>(V));
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, V);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) { return raw[a] > raw[b]; });
    ids.assign(idx.begin(), idx.begin() + k);
    lps.resize(size_t(k));
    for (int i = 0; i < k; ++i) lps[size_t(i)] = raw[ids[size_t(i)]] - lse;
}

}  // namespace qw
