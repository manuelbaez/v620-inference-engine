// Generation steps with MTP speculative decoding (Session::generate).
#include <algorithm>

#include "core/common.hpp"
#include "core/config.hpp"
#include "session/session.hpp"

namespace qw {

void Session::set_stop_tokens(int slot, std::vector<int32_t> ids) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "bad slot");
    slots_[size_t(slot)].stop = std::move(ids);
}

const std::vector<Session::StepOut> &Session::generate(const std::vector<StepReq> &reqs, int k) {
    out_.assign(reqs.size(), {});
    if (reqs.empty()) return out_;
    const int n = int(reqs.size());
    QW_CHECK(n <= Engine::MAX_BATCH_ROWS, "generate: too many requests for one batch");
    const int k_max = e_.has_mtp() ? std::max(0, std::min(k, Engine::MAX_BATCH_ROWS / n - 1)) : 0;
    // free positions of a slot; drafting k tokens needs k + 2
    auto room = [&](int slot) { return e_.slot_capacity(slot) - e_.slot_len(slot); };

    // first drafts of requests that have none for their pending token
    std::vector<Engine::DraftReq> dreqs;
    std::vector<size_t> which;
    for (size_t i = 0; i < reqs.size(); ++i) {
        const SlotInfo &si = slots_[size_t(reqs[i].slot)];
        if (k_max > 0 && si.drafts_for != reqs[i].pending && room(reqs[i].slot) > k_max + 1) {
            dreqs.push_back({reqs[i].slot, {reqs[i].pending}});
            which.push_back(i);
        }
    }
    if (!dreqs.empty()) {
        const auto &d = e_.draft(dreqs, k_max);
        for (size_t j = 0; j < which.size(); ++j) {
            SlotInfo &si = slots_[size_t(reqs[which[j]].slot)];
            si.drafts = d[j];
            si.drafts_for = reqs[which[j]].pending;
        }
    }

    // verification batch: pending token + drafts per request
    std::vector<Engine::Row> rows;
    std::vector<int> n_drafts(reqs.size(), 0);
    for (size_t i = 0; i < reqs.size(); ++i) {
        const StepReq &rq = reqs[i];
        const SlotInfo &si = slots_[size_t(rq.slot)];
        QW_CHECK(rq.budget >= 1 && room(rq.slot) >= 1, "generate: request has no room left");
        if (k_max > 0 && si.drafts_for == rq.pending)
            n_drafts[i] = int(std::min<int64_t>({k_max, int64_t(si.drafts.size()), rq.budget - 1, room(rq.slot) - 1}));
        out_[i].first_row = int(rows.size());
        rows.push_back({rq.slot, rq.pending});
        for (int j = 0; j < n_drafts[i]; ++j) rows.push_back({rq.slot, si.drafts[size_t(j)]});
    }
    decode(rows);

    // sample rows while they confirm the drafts; keep that prefix
    std::vector<Engine::DraftReq> next;
    which.clear();
    for (size_t i = 0; i < reqs.size(); ++i) {
        const StepReq &rq = reqs[i];
        SlotInfo &si = slots_[size_t(rq.slot)];
        StepOut &o = out_[i];
        const size_t base = si.hist.size() - size_t(n_drafts[i] + 1);  // history before this step
        for (int j = 0; j <= n_drafts[i]; ++j) {
            const int row = o.first_row + j;
            float lp = 0.f;
            const int32_t tok = sample_logits(e_.logits_rows().data() + size_t(row) * cfg::VOCAB, rq.slot, rq.params,
                                              &lp, e_.logits_rows_lse()[size_t(row)], base + size_t(j) + 1);
            o.tokens.push_back(tok);
            o.logprobs.push_back(lp);
            if (std::find(si.stop.begin(), si.stop.end(), tok) != si.stop.end()) {
                o.stopped = true;
                break;
            }
            if (j == n_drafts[i] || tok != si.drafts[size_t(j)]) break;
        }
        const int keep = int(o.tokens.size());  // rows kept: their inputs are committed
        e_.accept(rq.slot, keep);
        si.hist.resize(base + size_t(keep));
        si.drafts_for = -1;
        // row r's next token is o.tokens[r]: the MTP rows of the kept tokens
        if (!o.stopped && keep < rq.budget && k_max > 0 && room(rq.slot) > k_max + 1) {
            next.push_back({rq.slot, o.tokens});
            which.push_back(i);
        }
    }
    if (!next.empty()) {
        const auto &d = e_.draft(next, k_max);
        for (size_t j = 0; j < which.size(); ++j) {
            SlotInfo &si = slots_[size_t(reqs[which[j]].slot)];
            si.drafts = d[j];
            si.drafts_for = out_[which[j]].tokens.back();
        }
    }
    return out_;
}

}  // namespace qw
