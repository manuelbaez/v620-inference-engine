// Speculative decoding with the MTP head: accepting a prefix of verified
// rows (rolling back recurrent state) and drafting tokens.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

namespace qw {

void Engine::accept(int slot, int n) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "accept: bad slot");
    SlotHost &sh = slots_[size_t(slot)];
    QW_CHECK(sh.run_first >= 0 && n >= 1 && n <= sh.run_len, "accept: the slot has no such rows in the last decode");
    const int drop = sh.run_len - n;
    if (drop == 0) return;
    QW_CHECK(mtp_, "accept: needs the MTP engine (per-row states)");
    const int row = sh.run_first + n - 1;
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        auto &sl = rk.slots[size_t(slot)];
        CK(hipMemcpy2DAsync(sl.S, GDN_STATE * 4, rk.bt.gdn_save + size_t(row) * GDN_STATE,
                            size_t(gpu::MAX_ROWS) * GDN_STATE * 4, GDN_STATE * 4, N_GDN, hipMemcpyDeviceToDevice,
                            rk.s));
        CK(hipMemcpyAsync(sl.pend, rk.bt.X + size_t(row) * XW, XW * 4, hipMemcpyDeviceToDevice, rk.s));
    }
    for (auto &rkp : ranks_) CK(hipStreamSynchronize(rkp->s));
    sh.len -= drop;
    sh.hist.resize(sh.hist.size() - size_t(std::min<int>(drop, int(sh.hist.size()))));
    sh.run_len = n;
}

void Engine::mtp_catch_up(int slot, int64_t p0, const std::vector<int32_t> &next) {
    if (!mtp_ || next.empty()) return;
    const SlotHost &sh = slots_[size_t(slot)];
    const int64_t n = int64_t(next.size());
    QW_CHECK(p0 >= 0 && p0 + n <= sh.len && sh.len - p0 <= MTP_HISTORY, "mtp_catch_up: positions out of range");
    for (int64_t c0 = 0; c0 < n; c0 += gpu::MAX_ROWS) {
        const int m = int(std::min<int64_t>(gpu::MAX_ROWS, n - c0));
        std::vector<Row> rows;
        std::vector<int64_t> pos;
        djob_.src.clear();
        djob_.src_hist.clear();
        for (int i = 0; i < m; ++i) {
            rows.push_back({slot, next[size_t(c0 + i)]});
            pos.push_back(p0 + c0 + i);
            djob_.src.push_back(-1 - slot);
            djob_.src_hist.push_back(p0 + c0 + i);
        }
        stage_rows(rows, pos);
        job_ = Job::Mtp;
        dispatch();
    }
    djob_.src_hist.clear();
    for (auto &s : slots_) s.run_first = -1;  // the batch hidden now holds MTP rows
}

const std::vector<std::vector<int32_t>> &Engine::draft(const std::vector<DraftReq> &reqs, int k) {
    drafts_.assign(reqs.size(), {});
    if (!mtp_ || k <= 0 || reqs.empty()) return drafts_;
    constexpr int MR = gpu::MAX_ROWS;
    // per-rank argmax parts -> token of row i
    auto argmax_row = [&](int i) {
        float best = -INFINITY;
        int32_t tok = 0;
        for (int r = 0; r < RANKS; ++r) {
            const float v = damax_parts_[(size_t(r) * MR + i) * 2];
            int32_t idx;
            std::memcpy(&idx, &damax_parts_[(size_t(r) * MR + i) * 2 + 1], 4);
            if (v > best) {
                best = v;
                tok = int32_t(r * VOCAB_L + idx);
            }
        }
        return tok;
    };
    // step 1: the slots' not yet MTP-processed tokens
    std::vector<Row> rows;
    std::vector<int64_t> pos;
    djob_.src.clear();
    std::vector<int> last(reqs.size());
    for (size_t q = 0; q < reqs.size(); ++q) {
        const DraftReq &rq = reqs[q];
        QW_CHECK(rq.slot >= 0 && rq.slot < num_slots(), "draft: bad slot");
        const SlotHost &sh = slots_[size_t(rq.slot)];
        const int n = int(rq.next.size());
        QW_CHECK(n >= 1 && n <= sh.len, "draft: bad row count");
        // hidden of those tokens: the last decode's rows, or the slot's store for the newest one
        const bool from_batch = n > 1 && sh.run_first >= 0 && sh.run_len >= n;
        for (int i = 0; i < n; ++i) {
            rows.push_back({rq.slot, rq.next[size_t(i)]});
            pos.push_back(sh.len - n + i);
            djob_.src.push_back(i == n - 1 || !from_batch ? -1 - rq.slot : sh.run_first + sh.run_len - n + i);
        }
        last[q] = int(rows.size()) - 1;
    }
    for (int step = 0; step < k; ++step) {
        if (step > 0) {  // one row per request: the previous step's hidden and draft
            rows.clear();
            pos.clear();
            djob_.src.clear();
            for (size_t q = 0; q < reqs.size(); ++q) {
                const SlotHost &sh = slots_[size_t(reqs[q].slot)];
                rows.push_back({reqs[q].slot, drafts_[q].back()});
                pos.push_back(sh.len + step - 1);
                djob_.src.push_back(last[q]);
                last[q] = int(q);
            }
        }
        stage_rows(rows, pos);
        job_ = Job::Mtp;
        dispatch();
        for (size_t q = 0; q < reqs.size(); ++q) drafts_[q].push_back(argmax_row(last[q]));
    }
    // the batch hidden now holds MTP rows
    for (auto &sh : slots_) sh.run_first = -1;
    return drafts_;
}

}  // namespace qw
