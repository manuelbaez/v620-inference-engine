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

    std::vector<float> l(raw, raw + V);
    if (penalties) {
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
    if (p.temperature <= 0.f) return finish(int32_t(std::max_element(l.begin(), l.end()) - l.begin()));

    // candidates: top_k if set, else everything within 30 nats (x temperature) of the max
    std::vector<int32_t> cand;
    const float lmax = *std::max_element(l.begin(), l.end());
    const float cut = lmax - 30.f * std::max(p.temperature, 1.f);
    for (int i = 0; i < V; ++i)
        if (l[size_t(i)] >= cut) cand.push_back(i);
    std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return l[size_t(a)] > l[size_t(b)]; });
    if (p.top_k > 0 && int(cand.size()) > p.top_k) cand.resize(size_t(p.top_k));
    std::vector<double> prob(cand.size());
    double sum = 0;
    for (size_t i = 0; i < cand.size(); ++i)
        sum += prob[i] = std::exp(double(l[size_t(cand[i])] - lmax) / p.temperature);
    for (auto &x : prob) x /= sum;
    size_t keep = cand.size();
    if (p.top_p < 1.f) {
        double acc = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            acc += prob[i];
            if (acc >= p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    if (p.min_p > 0.f) {
        const double thr = p.min_p * prob[0];
        size_t k = 0;
        while (k < keep && prob[k] >= thr) ++k;
        keep = std::max<size_t>(k, 1);
    }
    std::mt19937_64 seeded(p.seed + n_hist);
    std::mt19937_64 &g = p.seed ? seeded : rng;
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += prob[i];
    double r = std::uniform_real_distribution<double>(0, tot)(g);
    for (size_t i = 0; i < keep; ++i) {
        r -= prob[i];
        if (r <= 0) return finish(cand[i]);
    }
    return finish(cand[keep - 1]);
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
