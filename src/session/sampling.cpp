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
        for (size_t i = size_t(prompt_end); i < n_hist; ++i) ++counts[cfg::real_token(hist[i])];
        if (p.repetition_penalty != 1.f) {
            std::vector<bool> seen(size_t(V), false);
            for (size_t i = 0; i < n_hist; ++i) seen[size_t(cfg::real_token(hist[i]))] = true;  // vision tokens: their pad
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
    std::mt19937_64 seeded(p.seed + n_hist);
    std::mt19937_64 &g = p.seed ? seeded : rng;

    // No truncation, or min_p alone: one pass over the vocabulary gathers the
    // tokens that can be drawn (weight >= min_p, and within 30 nats of the
    // max) with their weights, each computed once; the draw walks that list.
    if (p.top_p >= 1.f && p.top_k <= 0) {
        thread_local std::vector<int32_t> cand;
        thread_local std::vector<double> cw;
        cand.clear();
        cw.clear();
        const float zmin = std::max(-30.f, p.min_p > 0.f ? std::log(p.min_p) : -30.f);
        double z = 0;
        for (int i = 0; i < V; ++i) {
            const float zi = (L[i] - lmax) * inv_t;
            if (zi < zmin) continue;
            const double wi = double(std::exp(zi));
            cand.push_back(i);
            cw.push_back(wi);
            z += wi;
        }
        double r = std::uniform_real_distribution<double>(0, z)(g);
        for (size_t c = 0; c < cand.size(); ++c)
            if ((r -= cw[c]) <= 0) return finish(cand[c]);
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

int32_t sample_candidates(const gpu::SampleOut *out, int shards, const SamplingParams &p, size_t n_hist,
                          std::mt19937_64 &rng, float *logprob, float lse) {
    constexpr int C = gpu::SAMPLE_CAND;
    auto finish = [&](int32_t tok, float raw) {
        if (logprob) *logprob = raw - lse;
        return tok;
    };
    auto before = [](float la, int32_t ia, float lb, int32_t ib) { return la > lb || (la == lb && ia < ib); };
    if (p.temperature <= 0.f) {  // the best first candidate
        int best = -1;
        for (int r = 0; r < shards; ++r)
            if (out[r].n > 0 &&
                (best < 0 || before(out[r].cand_l[0], out[r].cand_idx[0], out[best].cand_l[0], out[best].cand_idx[0])))
                best = r;
        return finish(out[best].cand_idx[0], out[best].cand_raw[0]);
    }
    const float inv_t = 1.f / p.temperature;
    float lmax = -INFINITY;
    for (int r = 0; r < shards; ++r) lmax = std::max(lmax, out[r].lmax);
    std::mt19937_64 seeded(p.seed + n_hist);
    std::mt19937_64 &g = p.seed ? seeded : rng;
    struct Cand {
        float l, raw;
        int32_t idx;
    };
    // z of the last listed candidate of a full list: tokens it left out lie at or below it
    auto cut_z = [&](const gpu::SampleOut &o) { return o.n < C ? -INFINITY : (o.cand_l[C - 1] - lmax) * inv_t; };

    if (p.top_p >= 1.f && p.top_k <= 0) {
        const float zmin = std::max(-30.f, p.min_p > 0.f ? std::log(p.min_p) : -30.f);
        if (zmin <= -30.f) {  // untruncated: the best of the shards' Gumbel-max draws
            int best = -1;
            for (int r = 0; r < shards; ++r)
                if (out[r].g_idx >= 0 && (best < 0 || out[r].g_score > out[best].g_score ||
                                          (out[r].g_score == out[best].g_score && out[r].g_idx < out[best].g_idx)))
                    best = r;
            return finish(out[best].g_idx, out[best].g_raw);
        }
        // min_p: every token with weight >= min_p must be a candidate
        std::vector<Cand> kept;
        for (int r = 0; r < shards; ++r) {
            if (cut_z(out[r]) >= zmin) return -1;
            for (int i = 0; i < out[r].n; ++i)
                if ((out[r].cand_l[i] - lmax) * inv_t >= zmin)
                    kept.push_back({out[r].cand_l[i], out[r].cand_raw[i], out[r].cand_idx[i]});
        }
        std::sort(kept.begin(), kept.end(), [](const Cand &a, const Cand &b) { return a.idx < b.idx; });  // vocab order
        std::vector<double> w(kept.size());
        double z = 0;
        for (size_t i = 0; i < kept.size(); ++i) z += w[i] = double(std::exp((kept[i].l - lmax) * inv_t));
        double r = std::uniform_real_distribution<double>(0, z)(g);
        for (size_t i = 0; i < kept.size(); ++i)
            if ((r -= w[i]) <= 0) return finish(kept[i].idx, kept[i].raw);
        size_t best = 0;
        for (size_t i = 1; i < kept.size(); ++i)
            if (before(kept[i].l, kept[i].idx, kept[best].l, kept[best].idx)) best = i;
        return finish(kept[best].idx, kept[best].raw);
    }

    // Truncation: the shards' candidates within 30 nats of the max, in order.
    // They are the global order down to the highest cut of a full list; when
    // every cut lies below the 30-nat floor they hold every token that counts.
    std::vector<Cand> merged;
    float cut_l = -INFINITY;
    bool all_in = true;
    double z_all = 0;  // normalizer of the untruncated distribution
    for (int r = 0; r < shards; ++r) {
        const gpu::SampleOut &o = out[r];
        if (o.n == C) {
            cut_l = std::max(cut_l, o.cand_l[C - 1]);
            all_in = all_in && cut_z(o) < -30.f;
        }
        z_all += double(o.sumexp) * std::exp(double((o.lmax - lmax) * inv_t));
        for (int i = 0; i < o.n; ++i)
            if ((o.cand_l[i] - lmax) * inv_t >= -30.f) merged.push_back({o.cand_l[i], o.cand_raw[i], o.cand_idx[i]});
    }
    std::sort(merged.begin(), merged.end(), [&](const Cand &a, const Cand &b) { return before(a.l, a.idx, b.l, b.idx); });
    size_t valid = merged.size();
    if (!all_in) {
        valid = 0;
        while (valid < merged.size() && merged[valid].l > cut_l) ++valid;
    }
    size_t K = valid;
    if (p.top_k > 0) {
        K = size_t(p.top_k);
        if (all_in) K = std::min(K, merged.size());
        if (K > valid) return -1;
    }
    std::vector<double> prob(K);
    double zk = 0;
    for (size_t i = 0; i < K; ++i) zk += prob[i] = double(std::exp((merged[i].l - lmax) * inv_t));
    const double norm = p.top_k > 0 ? zk : z_all;
    for (auto &x : prob) x /= norm;
    size_t keep = K;
    if (p.top_p < 1.f) {
        double acc = 0;
        keep = 0;
        for (size_t i = 0; i < K && keep == 0; ++i)
            if ((acc += prob[i]) >= p.top_p) keep = i + 1;
        if (keep == 0) {
            if (p.top_k <= 0 && !all_in) return -1;  // the nucleus reaches past the candidates
            keep = K;
        }
    }
    if (keep == 0) return -1;
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
        if (r <= 0) return finish(merged[i].idx, merged[i].raw);
    }
    return finish(merged[keep - 1].idx, merged[keep - 1].raw);
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
