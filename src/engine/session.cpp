#include "engine/session.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "core/common.hpp"
#include "core/config.hpp"

namespace qw {

Session::Session(Engine &e)
    : e_(e), slots_(size_t(e.num_slots())), snaps_(Engine::SNAPSHOTS), rng_(std::random_device{}()) {}

void Session::drop_snapshots_after(int slot, int64_t n) {
    for (auto &s : snaps_)
        if (s.valid && s.slot == slot && int64_t(s.tokens.size()) > n) s.valid = false;
}

void Session::save_snapshot(int slot) {
    const auto &hist = slots_[size_t(slot)].hist;
    const int64_t n = int64_t(hist.size());
    int idx = -1;
    for (int i = 0; i < int(snaps_.size()); ++i)
        if (snaps_[size_t(i)].valid && snaps_[size_t(i)].slot == slot && int64_t(snaps_[size_t(i)].tokens.size()) == n)
            idx = i;  // refresh in place
    if (idx < 0)
        for (int i = 0; i < int(snaps_.size()); ++i)
            if (!snaps_[size_t(i)].valid) { idx = i; break; }
    if (idx < 0) {  // evict least recently used
        idx = 0;
        for (int i = 1; i < int(snaps_.size()); ++i)
            if (snaps_[size_t(i)].used < snaps_[size_t(idx)].used) idx = i;
    }
    e_.snapshot_save(idx, slot);
    Snap &s = snaps_[size_t(idx)];
    s.valid = true;
    s.slot = slot;
    s.tokens = hist;
    s.logits = e_.logits();
    s.used = ++clock_;
}

// Tokens of `prompt` a slot could reuse (in place or from one of its snapshots).
size_t Session::reusable(int slot, const std::vector<int32_t> &prompt) const {
    const auto &hist = slots_[size_t(slot)].hist;
    size_t common = 0;
    while (common < hist.size() && common < prompt.size() && hist[common] == prompt[common]) ++common;
    size_t best = common == hist.size() ? common : 0;
    for (const Snap &s : snaps_)
        if (s.valid && s.slot == slot && s.tokens.size() <= common && s.tokens.size() > best) best = s.tokens.size();
    return best;
}

int Session::acquire(const std::vector<int32_t> &prompt, int64_t max_new) {
    const int64_t need = int64_t(prompt.size()) + std::max<int64_t>(max_new, 1);
    int best = -1;
    size_t best_reuse = 0;
    for (int i = 0; i < num_slots(); ++i) {
        const SlotInfo &si = slots_[size_t(i)];
        if (si.busy || e_.slot_capacity(i) < need) continue;
        const size_t r = reusable(i, prompt);
        if (best < 0 || r > best_reuse ||
            (r == best_reuse && (si.used < slots_[size_t(best)].used ||
                                 (si.used == slots_[size_t(best)].used && e_.slot_capacity(i) < e_.slot_capacity(best))))) {
            best = i;
            best_reuse = r;
        }
    }
    if (best >= 0) {
        slots_[size_t(best)].busy = true;
        slots_[size_t(best)].used = ++clock_;
    }
    return best;
}

void Session::release(int slot) { slots_[size_t(slot)].busy = false; }

int64_t Session::set_prompt(int slot, const std::vector<int32_t> &prompt) {
    QW_CHECK(!prompt.empty(), "empty prompt");
    QW_CHECK(int64_t(prompt.size()) <= e_.slot_capacity(slot), "prompt longer than the slot's KV capacity");
    auto &hist = slots_[size_t(slot)].hist;
    size_t common = 0;
    while (common < hist.size() && common < prompt.size() && hist[common] == prompt[common]) ++common;

    int64_t reused = 0;
    if (common == hist.size() && common > 0) {
        reused = int64_t(common);  // continue in place
        if (common == prompt.size()) {
            // Nothing new. The engine's prompt logits belong to whatever was
            // prefilled last, so only a snapshot at exactly this point can
            // provide them; otherwise recompute the last token.
            bool found = false;
            for (auto &s : snaps_)
                if (s.valid && s.slot == slot && s.tokens.size() == common) {
                    e_.set_logits(s.logits);
                    s.used = ++clock_;
                    found = true;
                }
            if (!found) {
                const int64_t n = int64_t(common) - 1;
                int best = -1;
                for (int i = 0; i < int(snaps_.size()); ++i) {
                    const Snap &s = snaps_[size_t(i)];
                    if (s.valid && s.slot == slot && int64_t(s.tokens.size()) <= n &&
                        (best < 0 || s.tokens.size() > snaps_[size_t(best)].tokens.size()))
                        best = i;
                }
                if (best >= 0) {
                    const Snap &s = snaps_[size_t(best)];
                    e_.snapshot_restore(best, slot, int64_t(s.tokens.size()), s.tokens);
                    hist = s.tokens;
                } else {
                    e_.slot_reset(slot);
                    hist.clear();
                }
                drop_snapshots_after(slot, int64_t(hist.size()));
                reused = int64_t(hist.size());
            }
        }
    } else {
        int best = -1;
        for (int i = 0; i < int(snaps_.size()); ++i) {
            const Snap &s = snaps_[size_t(i)];
            if (!s.valid || s.slot != slot || s.tokens.size() > common || s.tokens.size() > prompt.size()) continue;
            if (best < 0 || s.tokens.size() > snaps_[size_t(best)].tokens.size()) best = i;
        }
        if (best >= 0) {
            Snap &s = snaps_[size_t(best)];
            const int64_t n = int64_t(s.tokens.size());
            e_.snapshot_restore(best, slot, n, s.tokens);
            s.used = ++clock_;
            hist = s.tokens;
            drop_snapshots_after(slot, n);
            reused = n;
            if (n == int64_t(prompt.size())) e_.set_logits(s.logits);
        } else {
            e_.slot_reset(slot);
            hist.clear();
            drop_snapshots_after(slot, 0);
        }
    }

    if (prompt.size() > hist.size()) {
        std::vector<int32_t> rest(prompt.begin() + ptrdiff_t(hist.size()), prompt.end());
        drop_snapshots_after(slot, int64_t(hist.size()));
        e_.prefill(slot, rest, nullptr, [&](int64_t pos) {
            hist.insert(hist.end(), prompt.begin() + ptrdiff_t(hist.size()), prompt.begin() + ptrdiff_t(pos));
            save_snapshot(slot);
        });
    }
    slots_[size_t(slot)].prompt_end = int64_t(hist.size());
    single_ = false;
    return reused;
}

void Session::decode(const std::vector<Engine::Row> &rows) {
    for (const auto &r : rows) drop_snapshots_after(r.slot, int64_t(slots_[size_t(r.slot)].hist.size()));
    e_.decode(rows);
    row_slot_.clear();
    for (const auto &r : rows) {
        slots_[size_t(r.slot)].hist.push_back(r.token);
        row_slot_.push_back(r.slot);
    }
    single_ = rows.size() == 1;
}

int32_t Session::sample_prompt(int slot, const SamplingParams &p, float *logprob) {
    return sample_logits(e_.logits().data(), slot, p, logprob);
}

int32_t Session::sample_row(int row, const SamplingParams &p, float *logprob) {
    QW_CHECK(row >= 0 && row < int(row_slot_.size()), "sample_row: bad row");
    // Rows are sampled after their slot's tokens were appended, so the slot's
    // history is one ahead of this row when later rows of the same run exist;
    // penalties only need approximate history, which this is.
    return sample_logits(e_.logits_rows().data() + size_t(row) * cfg::VOCAB, row_slot_[size_t(row)], p, logprob,
                         e_.logits_rows_lse()[size_t(row)]);
}

int32_t Session::sample(const SamplingParams &p, float *logprob) {
    if (single_) return sample_row(0, p, logprob);
    return sample_prompt(0, p, logprob);
}

static float log_sum_exp(const float *raw) {
    const int V = cfg::VOCAB;
    const float mx = *std::max_element(raw, raw + V);
    double z = 0;
    for (int i = 0; i < V; ++i) z += std::exp(double(raw[i] - mx));
    return mx + float(std::log(z));
}

int32_t Session::sample_logits(const float *raw, int slot, const SamplingParams &p, float *logprob, float lse_known) {
    const int V = cfg::VOCAB;
    const auto &hist = slots_[size_t(slot)].hist;
    const int64_t prompt_end = slots_[size_t(slot)].prompt_end;
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
        for (size_t i = size_t(prompt_end); i < hist.size(); ++i) ++counts[hist[i]];
        if (p.repetition_penalty != 1.f) {
            std::vector<bool> seen(size_t(V), false);
            for (int32_t t : hist) seen[size_t(t)] = true;
            for (int i = 0; i < V; ++i)
                if (seen[size_t(i)])
                    l[size_t(i)] = l[size_t(i)] > 0 ? l[size_t(i)] / p.repetition_penalty : l[size_t(i)] * p.repetition_penalty;
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
    for (size_t i = 0; i < cand.size(); ++i) sum += prob[i] = std::exp(double(l[size_t(cand[i])] - lmax) / p.temperature);
    for (auto &x : prob) x /= sum;
    size_t keep = cand.size();
    if (p.top_p < 1.f) {
        double acc = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            acc += prob[i];
            if (acc >= p.top_p) { keep = i + 1; break; }
        }
    }
    if (p.min_p > 0.f) {
        const double thr = p.min_p * prob[0];
        size_t k = 0;
        while (k < keep && prob[k] >= thr) ++k;
        keep = std::max<size_t>(k, 1);
    }
    std::mt19937_64 seeded(p.seed + hist.size());
    std::mt19937_64 &g = p.seed ? seeded : rng_;
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += prob[i];
    double r = std::uniform_real_distribution<double>(0, tot)(g);
    for (size_t i = 0; i < keep; ++i) {
        r -= prob[i];
        if (r <= 0) return finish(cand[i]);
    }
    return finish(cand[keep - 1]);
}

static void top_k_of(const float *raw, int k, std::vector<int32_t> &ids, std::vector<float> &lps,
                     float lse_known = NAN) {
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

void Session::top_logprobs_row(int row, int k, std::vector<int32_t> &ids, std::vector<float> &lps) const {
    top_k_of(e_.logits_rows().data() + size_t(row) * cfg::VOCAB, k, ids, lps, e_.logits_rows_lse()[size_t(row)]);
}

void Session::top_logprobs_prompt(int k, std::vector<int32_t> &ids, std::vector<float> &lps) const {
    top_k_of(e_.logits().data(), k, ids, lps);
}

}  // namespace qw
