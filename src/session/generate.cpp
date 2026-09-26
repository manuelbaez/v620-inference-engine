// Generation steps with MTP speculative decoding (Session::generate).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/common.hpp"
#include "core/config.hpp"
#include "core/shard.hpp"
#include "session/session.hpp"

namespace qw {

namespace {
// Step cost with K drafts relative to a plain step: base (one more
// verification row and the MTP pass) + beta per further draft. Measured on
// this box: 15.0 ms plain, 18.8 / 22.6 / 26.4 ms with 1 / 2 / 3 drafts.
float env_or(const char *name, float def) {
    const char *v = std::getenv(name);
    return v ? float(std::atof(v)) : def;
}
uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
const float SPEC_BASE = env_or("QW_SPEC_BASE", 1.25f), SPEC_BETA = env_or("QW_SPEC_COST", 0.25f);

// Expected tokens per step cost with the best K in [1, k_max]; sets *best_k.
float best_rate(float a, int k_max, int *best_k) {
    float best = 0.f, expect = 1.f, p = 1.f;
    *best_k = 1;
    for (int K = 1; K <= k_max; ++K) {
        p *= a;
        expect += p;
        const float rate = expect / (SPEC_BASE + SPEC_BETA * float(K - 1));
        if (rate > best) {
            best = rate;
            *best_k = K;
        }
    }
    return best;
}
}  // namespace

// Drafts for a request's next step: K in [1, k_max] maximizing expected
// tokens per step cost, with draft j accepted with probability a^j (a: the
// slot's running acceptance).
int Session::draft_count(int slot, int k_max) const {
    if (k_max <= 1) return k_max;
    int K = 1;
    best_rate(slots_[size_t(slot)].accept, k_max, &K);
    return K;
}

// Plain decoding (no drafts, no MTP pass) while drafting costs more than it
// saves: expected tokens per step cost under 1 with any K. The MTP layer then
// falls behind; after a stretch of plain steps (doubling while drafting keeps
// not paying, up to Engine::MTP_HISTORY) its missing rows are caught up and
// drafting is tried again from a neutral acceptance.
bool Session::plain_step(int slot, int k_max) {
    SlotInfo &si = slots_[size_t(slot)];
    if (k_max <= 0) return false;
    if (si.plain_left > 0 && si.mtp_lag + 1 < Engine::MTP_HISTORY) {
        --si.plain_left;
        return true;
    }
    if (si.mtp_lag > 0) {  // resume: the MTP rows of the tokens decoded plainly, but the newest
        const int64_t len = int64_t(si.hist.size());
        const int64_t p0 = len - si.mtp_lag;
        std::vector<int32_t> next(si.hist.begin() + ptrdiff_t(p0 + 1), si.hist.end());
        e_.mtp_catch_up(slot, p0, next);
        si.mtp_lag = 0;
        si.plain_left = 0;
        si.accept = 0.5f;
        return false;
    }
    int K = 1;
    if (best_rate(si.accept, k_max, &K) >= 1.f) {
        if (si.accept > 0.5f) si.plain_len = 32;  // drafting pays again: short stretches next time
        return false;
    }
    si.plain_left = si.plain_len - 1;
    si.plain_len = std::min(si.plain_len * 2, Engine::MTP_HISTORY - 16);
    return true;
}

void Session::set_stop_tokens(int slot, std::vector<int32_t> ids) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "bad slot");
    slots_[size_t(slot)].stop = std::move(ids);
}

namespace {
// QW_TRACE: phase times of generate(), printed every 100 steps
struct GenTrace {
    bool on = std::getenv("QW_TRACE") != nullptr;
    double t[5] = {}, sample = 0, accept = 0;  // within phase 3
    uint64_t fallbacks = 0, rows = 0;         // GPU-sampled rows, and those needing their full logits
    int steps = 0;
    std::chrono::steady_clock::time_point last;
    void start() { last = std::chrono::steady_clock::now(); }
    void lap(int i) {
        const auto now = std::chrono::steady_clock::now();
        t[i] += std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
    }
    void step() {
        if (!on || ++steps % 100) return;
        std::printf("generate: first drafts %.2f, rows %.2f, decode %.2f, sample+accept %.2f (sample %.2f, accept "
                    "%.2f), drafts %.2f ms/step; sampling fallbacks %llu of %llu rows\n",
                    t[0] / steps, t[1] / steps, t[2] / steps, t[3] / steps, sample / steps, accept / steps, t[4] / steps,
                    (unsigned long long)fallbacks, (unsigned long long)rows);
        std::fflush(stdout);
    }
};
GenTrace g_trace;
}  // namespace

const std::vector<Session::StepOut> &Session::generate(const std::vector<StepReq> &reqs, int k) {
    g_trace.start();
    out_.assign(reqs.size(), {});
    if (reqs.empty()) return out_;
    const int n = int(reqs.size());
    QW_CHECK(n <= Engine::MAX_BATCH_ROWS, "generate: too many requests for one batch");
    // Verification batches are capped at 8 rows: batches of 11-16 rows (3-4 requests x 1 + 3 drafts)
    // intermittently produce wrong tokens or GPU faults on this box, cause not yet found
    // (docs/DESIGN.md, "Open issue"). QW_SPEC_MAX_ROWS=16 lifts the cap.
    static const int max_rows = std::getenv("QW_SPEC_MAX_ROWS") ? std::atoi(std::getenv("QW_SPEC_MAX_ROWS")) : 8;
    const int k_max = e_.has_mtp() ? std::max(0, std::min({k, Engine::MAX_BATCH_ROWS / n - 1, max_rows / n - 1})) : 0;
    // free positions of a slot; drafting k tokens needs k + 2
    auto room = [&](int slot) { return e_.slot_capacity(slot) - e_.slot_len(slot); };

    // requests decoding plainly this step (drafting does not pay for them now)
    std::vector<char> plain(reqs.size(), 0);
    for (size_t i = 0; i < reqs.size(); ++i) plain[i] = plain_step(reqs[i].slot, k_max);
    // first drafts of requests that have none for their pending token
    std::vector<Engine::DraftReq> dreqs;
    std::vector<size_t> which;
    for (size_t i = 0; i < reqs.size(); ++i) {
        const SlotInfo &si = slots_[size_t(reqs[i].slot)];
        if (k_max > 0 && !plain[i] && si.drafts_for != reqs[i].pending && room(reqs[i].slot) > k_max + 1) {
            dreqs.push_back({reqs[i].slot, {reqs[i].pending}});
            which.push_back(i);
        }
    }
    if (!dreqs.empty()) {
        int kd = 1;
        for (const auto &dq : dreqs) kd = std::max(kd, draft_count(dq.slot, k_max));
        const auto &d = e_.draft(dreqs, kd);
        for (size_t j = 0; j < which.size(); ++j) {
            SlotInfo &si = slots_[size_t(reqs[which[j]].slot)];
            si.drafts = d[j];
            si.drafts_for = reqs[which[j]].pending;
        }
    }

    g_trace.lap(0);
    // verification batch: pending token + drafts per request
    std::vector<Engine::Row> rows;
    std::vector<int> n_drafts(reqs.size(), 0);
    for (size_t i = 0; i < reqs.size(); ++i) {
        const StepReq &rq = reqs[i];
        const SlotInfo &si = slots_[size_t(rq.slot)];
        QW_CHECK(rq.budget >= 1 && room(rq.slot) >= 1, "generate: request has no room left");
        if (k_max > 0 && !plain[i] && si.drafts_for == rq.pending)
            n_drafts[i] = int(std::min<int64_t>(
                {draft_count(rq.slot, k_max), int64_t(si.drafts.size()), rq.budget - 1, room(rq.slot) - 1}));
        out_[i].first_row = int(rows.size());
        rows.push_back({rq.slot, rq.pending});
        for (int j = 0; j < n_drafts[i]; ++j) rows.push_back({rq.slot, si.drafts[size_t(j)]});
    }
    // Each row draws from its own generator, seeded in order from the
    // session's: exact sampling. On the GPUs (QW_GPU_SAMPLING, default on) the
    // rows' Gumbel keys come from the same seeds, and only the sampling
    // results come back (the full logits too when a request wants top logprobs).
    static const bool gpu_sampling = !std::getenv("QW_GPU_SAMPLING") || std::atoi(std::getenv("QW_GPU_SAMPLING")) != 0;
    std::vector<uint64_t> row_seed(rows.size());
    std::vector<std::pair<int, int>> row_of;  // (request, j)
    for (size_t i = 0; i < reqs.size(); ++i)
        for (int j = 0; j <= n_drafts[i]; ++j) row_of.push_back({int(i), j});
    for (auto &sd : row_seed) sd = rng_();
    Engine::SampleSpec spec;
    if (gpu_sampling) {
        for (size_t i = 0; i < reqs.size(); ++i) {
            const SamplingParams &p = reqs[i].params;
            const SlotInfo &si = slots_[size_t(reqs[i].slot)];
            spec.logits = spec.logits || reqs[i].want_logits;
            if (p.presence_penalty != 0.f || p.frequency_penalty != 0.f || p.repetition_penalty != 1.f)
                e_.penalty_sync(reqs[i].slot, si.epoch, si.prompt_end, si.hist);
        }
        for (size_t r = 0; r < rows.size(); ++r) {
            const auto [i, j] = row_of[r];
            const SamplingParams &p = reqs[size_t(i)].params;
            const size_t n_hist = slots_[size_t(reqs[size_t(i)].slot)].hist.size() + size_t(j) + 1;
            gpu::SampleRow sr{};
            sr.inv_t = p.temperature > 0.f ? 1.f / p.temperature : 0.f;
            sr.presence = p.presence_penalty;
            sr.frequency = p.frequency_penalty;
            sr.repetition = p.repetition_penalty;
            sr.key = mix64(p.seed ? p.seed + n_hist : row_seed[r]);
            spec.rows.push_back(sr);
        }
    }
    g_trace.lap(1);
    decode(rows, gpu_sampling ? &spec : nullptr);
    g_trace.lap(2);

    // Sample every row in parallel (each row only reads its own logits; the
    // acceptance below then walks them in order).
    const auto ts = std::chrono::steady_clock::now();
    std::vector<int32_t> row_tok(rows.size(), -1);
    std::vector<float> row_lp(rows.size());
    sample_pool_.parallel_for(int64_t(rows.size()), 1, [&](int64_t a, int64_t b) {
        for (int64_t r = a; r < b; ++r) {
            const auto [i, j] = row_of[size_t(r)];
            const StepReq &rq = reqs[size_t(i)];
            const SlotInfo &si = slots_[size_t(rq.slot)];
            const size_t base = si.hist.size() - size_t(n_drafts[size_t(i)] + 1);
            std::mt19937_64 g(row_seed[size_t(r)]);
            if (gpu_sampling)
                row_tok[size_t(r)] = sample_candidates(e_.sample_out().data() + size_t(r) * RANKS, RANKS, rq.params,
                                                       base + size_t(j) + 1, g, &row_lp[size_t(r)],
                                                       e_.logits_rows_lse()[size_t(r)]);
            else
                row_tok[size_t(r)] = sample_token(e_.logits_rows().data() + size_t(r) * cfg::VOCAB, rq.params, si.hist,
                                                  base + size_t(j) + 1, si.prompt_end, g, &row_lp[size_t(r)],
                                                  e_.logits_rows_lse()[size_t(r)]);
        }
    });
    // rows the GPU candidates did not settle: from their full logits
    std::vector<float> full;
    for (size_t r = 0; r < rows.size(); ++r) {
        if (row_tok[r] >= 0) continue;
        const auto [i, j] = row_of[r];
        const StepReq &rq = reqs[size_t(i)];
        const SlotInfo &si = slots_[size_t(rq.slot)];
        const size_t base = si.hist.size() - size_t(n_drafts[size_t(i)] + 1);
        const float *l = e_.logits_rows_valid() ? e_.logits_rows().data() + r * cfg::VOCAB : nullptr;
        if (!l) {
            full.resize(cfg::VOCAB);
            e_.row_logits(int(r), full.data());
            l = full.data();
        }
        std::mt19937_64 g(row_seed[r]);
        row_tok[r] = sample_token(l, rq.params, si.hist, base + size_t(j) + 1, si.prompt_end, g, &row_lp[r],
                                  e_.logits_rows_lse()[r]);
        ++fallbacks_;
        ++g_trace.fallbacks;
    }
    g_trace.rows += rows.size();
    if (g_trace.on)
        g_trace.sample += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();

    // keep each request's rows while they confirm its drafts
    std::vector<Engine::DraftReq> next;
    which.clear();
    for (size_t i = 0; i < reqs.size(); ++i) {
        const StepReq &rq = reqs[i];
        SlotInfo &si = slots_[size_t(rq.slot)];
        StepOut &o = out_[i];
        const size_t base = si.hist.size() - size_t(n_drafts[i] + 1);  // history before this step
        for (int j = 0; j <= n_drafts[i]; ++j) {
            const int row = o.first_row + j;
            const int32_t tok = row_tok[size_t(row)];
            o.tokens.push_back(tok);
            o.logprobs.push_back(row_lp[size_t(row)]);
            if (std::find(si.stop.begin(), si.stop.end(), tok) != si.stop.end()) {
                o.stopped = true;
                break;
            }
            if (j == n_drafts[i] || tok != si.drafts[size_t(j)]) break;
        }
        const int keep = int(o.tokens.size());  // rows kept: their inputs are committed
        if (g_trace.on)
            std::printf("gen: slot %d len %lld drafts %d keep %d%s\n", rq.slot,
                        (long long)(e_.slot_len(rq.slot) - n_drafts[i] - 1), n_drafts[i], keep,
                        o.stopped ? " stop" : "");
        // acceptance: drafts 0..keep-2 confirmed, draft keep-1 rejected (unless all were or a stop cut it)
        constexpr float rate = 0.1f;
        const int confirmed = keep - 1;
        for (int j = 0; j < confirmed; ++j) si.accept += rate * (1.f - si.accept);
        if (!o.stopped && confirmed < n_drafts[i]) si.accept += rate * (0.f - si.accept);
        const auto ta = std::chrono::steady_clock::now();
        e_.accept(rq.slot, keep);
        if (g_trace.on)
            g_trace.accept += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
        si.hist.resize(base + size_t(keep));
        si.drafts_for = -1;
        if (plain[i]) {  // no MTP pass: this token's MTP row is missing
            ++si.mtp_lag;
            continue;
        }
        // row r's next token is o.tokens[r]: the MTP rows of the kept tokens
        if (!o.stopped && keep < rq.budget && k_max > 0 && room(rq.slot) > k_max + 1) {
            next.push_back({rq.slot, o.tokens});
            which.push_back(i);
        }
    }
    g_trace.lap(3);
    if (!next.empty()) {
        int kd = 1;
        for (const auto &nq : next) kd = std::max(kd, draft_count(nq.slot, k_max));
        const auto &d = e_.draft(next, kd);
        for (size_t j = 0; j < which.size(); ++j) {
            SlotInfo &si = slots_[size_t(reqs[which[j]].slot)];
            si.drafts = d[j];
            si.drafts_for = out_[which[j]].tokens.back();
        }
    }
    g_trace.lap(4);
    g_trace.step();
    return out_;
}

}  // namespace qw
