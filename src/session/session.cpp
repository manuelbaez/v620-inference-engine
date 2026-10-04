#include "session/session.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "core/common.hpp"
#include "core/config.hpp"

namespace qw {

Session::Session(Engine &e)
    : e_(e), slots_(size_t(e.num_slots())), snaps_(Engine::SNAPSHOTS), rng_(std::random_device{}()) {
    // block store: QW_HOST_CACHE_GB (default 128, 0 = off); its disk tier:
    // QW_DISK_CACHE_DIR (off when unset), QW_DISK_CACHE_GB (default 200)
    const char *gb = std::getenv("QW_HOST_CACHE_GB");
    const size_t budget = size_t((gb ? std::atof(gb) : 128.0) * 1e9);
    if (budget > 0) {
        const char *dir = std::getenv("QW_DISK_CACHE_DIR");
        const char *dgb = std::getenv("QW_DISK_CACHE_GB");
        store_ = std::make_unique<BlockStore>(e_, budget, dir ? dir : "", size_t((dgb ? std::atof(dgb) : 200.0) * 1e9));
        store_->wait_reserve(30);  // the first request after a start (llama-swap sends it the moment /health is up)
    }
    if (const char *g = std::getenv("QW_SNAP_MIN_GAP")) min_gap_ = std::max<int64_t>(1, std::atoll(g));
    if (const char *a = std::getenv("QW_RESERVE_AHEAD")) reserve_ahead_ = std::max(0, std::atoi(a));
    const char *vgb = std::getenv("QW_VISION_CACHE_GB");
    vision_budget_ = size_t((vgb ? std::atof(vgb) : 2.0) * 1e9);
}

void Session::drop_snapshots_after(int slot, int64_t n) {
    for (auto &s : snaps_)
        if (s.valid && s.slot == slot && int64_t(s.tokens.size()) > n) s.valid = false;
}

int Session::save_snapshot(int slot, const std::vector<float> *logits) {
    const auto &hist = slots_[size_t(slot)].hist;
    const int64_t n = int64_t(hist.size());
    auto reserved = [&](int i) { return std::find(reserved_.begin(), reserved_.end(), i) != reserved_.end(); };
    int idx = -1;
    for (int i = 0; i < int(snaps_.size()); ++i)
        if (snaps_[size_t(i)].valid && snaps_[size_t(i)].slot == slot && int64_t(snaps_[size_t(i)].tokens.size()) == n)
            idx = i;  // refresh in place
    if (idx < 0)
        for (int i = 0; i < int(snaps_.size()); ++i)
            if (!snaps_[size_t(i)].valid && !reserved(i)) {
                idx = i;
                break;
            }
    if (idx < 0)  // evict least recently used
        for (int i = 0; i < int(snaps_.size()); ++i)
            if (!reserved(i) && (idx < 0 || snaps_[size_t(i)].used < snaps_[size_t(idx)].used)) idx = i;
    e_.snapshot_save(idx, slot);
    Snap &s = snaps_[size_t(idx)];
    s.valid = true;
    s.slot = slot;
    s.tokens = hist;
    s.logits = logits ? *logits : e_.logits();
    s.used = ++clock_;
    return idx;
}

void Session::set_prompt_logits(int slot, const std::vector<float> &l) {
    e_.set_logits(l);
    slots_[size_t(slot)].logits = l;
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
    // What the request will hold soon: the VRAM part of a slot should cover it (beyond it the slot spills).
    const int64_t soon = int64_t(prompt.size()) + std::min<int64_t>(std::max<int64_t>(max_new, 1), 2048);
    // Slot a is a better home than b for these tokens: its VRAM part covers them and b's does not; both cover them:
    // the smaller one (the big slot stays free for a long prompt); neither does: the larger one.
    const auto vram_better = [&](int a, int b) {
        const int64_t va = e_.slot_vram_tokens(a), vb = e_.slot_vram_tokens(b);
        if ((va >= soon) != (vb >= soon)) return va >= soon;
        return va >= soon ? va < vb : va > vb;
    };
    int best = -1;
    size_t best_reuse = 0;
    for (int i = 0; i < num_slots(); ++i) {
        const SlotInfo &si = slots_[size_t(i)];
        if (si.busy || e_.slot_capacity(i) < need) continue;
        const size_t r = reusable(i, prompt);
        const bool same_vram = best >= 0 && e_.slot_vram_tokens(i) == e_.slot_vram_tokens(best);
        if (best < 0 || r > best_reuse ||
            (r == best_reuse &&
             (same_vram ? (si.used < slots_[size_t(best)].used ||
                           (si.used == slots_[size_t(best)].used && e_.slot_capacity(i) < e_.slot_capacity(best)))
                        : vram_better(i, best)))) {
            best = i;
            best_reuse = r;
        }
    }
    // A conversation about to run past its slot's VRAM part moves to a free slot that holds it in VRAM (or, when none
    // does, to one with a larger VRAM part), when the block store can restore it there: the snapshot at the end of its
    // last prompt is there, so the move costs importing those blocks and prefilling what came after (measured: 0.17 s
    // of restore, once). Off by default (the conversation stays and spills, which costs 1-3% of decode);
    // QW_SLOT_MOVE=1 turns it on. QW_SLOT_MOVE_SLACK: the tokens the move may have to prefill again (default 4096).
    static const bool move_on = std::getenv("QW_SLOT_MOVE") && std::atoi(std::getenv("QW_SLOT_MOVE")) != 0;
    static const int64_t move_slack = std::getenv("QW_SLOT_MOVE_SLACK") ? std::atoll(std::getenv("QW_SLOT_MOVE_SLACK")) : 4096;
    if (move_on && store_ && best >= 0 && best_reuse > 0 && e_.slot_vram_tokens(best) < soon) {
        int to = -1;
        for (int i = 0; i < num_slots(); ++i) {
            if (i == best || slots_[size_t(i)].busy || e_.slot_capacity(i) < need) continue;
            if (e_.slot_vram_tokens(i) <= e_.slot_vram_tokens(best)) continue;
            if (to < 0 || vram_better(i, to)) to = i;
        }
        if (to >= 0) {
            const BlockStore::Hit hit = store_->lookup(prompt, 0);
            if (hit.n + move_slack >= int64_t(best_reuse)) {
                log("slot move: a prompt of %zu tokens leaves slot %d (%lld tokens in VRAM, %zu reusable there) for slot %d "
                    "(%lld in VRAM): %lld tokens come from the block store",
                    prompt.size(), best, (long long)e_.slot_vram_tokens(best), best_reuse, to,
                    (long long)e_.slot_vram_tokens(to), (long long)hit.n);
                best = to;
            }
        }
    }
    if (best >= 0) {
        slots_[size_t(best)].busy = true;
        slots_[size_t(best)].used = ++clock_;
    }
    return best;
}

void Session::save_to_store(int slot, int snap, const std::vector<int32_t> &tokens, const std::vector<float> *logits) {
    Timing &t = slots_[size_t(slot)].timing;
    Stopwatch save_time(&t.save_s);
    const double waited = PinnedPool::thread_wait_s();
    store_->save(slot, snap, tokens, logits);
    t.pin_wait_s += PinnedPool::thread_wait_s() - waited;
}

void Session::release(int slot) {
    SlotInfo &si = slots_[size_t(slot)];
    si.busy = false;
    si.stop.clear();
    si.drafts_for = -1;
    si.pending.clear();  // an unfinished prefill: the slot keeps what went in
    si.pending_media.clear();
}

int64_t Session::set_prompt(int slot, const std::vector<int32_t> &prompt) {
    const int64_t reused = begin_keys(slot, prompt);
    prefill_some(slot, INT64_MAX);
    return reused;
}

bool Session::prefill_some(int slot, int64_t max_tokens) {
    SlotInfo &si = slots_[size_t(slot)];
    if (si.pending.empty()) return true;
    const int64_t from = int64_t(si.hist.size()), n = int64_t(si.pending.size());
    media_ = si.pending_media.empty() ? nullptr : &si.pending_media;
    Stopwatch prefill_time(&si.timing.prefill_s);
    try {
        prefill_range(slot, si.pending, max_tokens >= n - from ? n : from + std::max<int64_t>(1, max_tokens));
    } catch (...) {
        media_ = nullptr;
        throw;
    }
    media_ = nullptr;
    if (int64_t(si.hist.size()) < n) return false;
    si.logits = e_.logits();
    si.prompt_end = n;
    si.pending.clear();
    si.pending_media.clear();
    single_ = false;
    return true;
}

// Restores what the slot, its snapshots or the block store hold of `prompt`
// and leaves the rest pending for prefill_some. Returns the reused tokens.
int64_t Session::begin_keys(int slot, const std::vector<int32_t> &prompt) {
    QW_CHECK(!prompt.empty(), "empty prompt");
    QW_CHECK(int64_t(prompt.size()) <= e_.slot_capacity(slot), "prompt longer than the slot's KV capacity");
    SlotInfo &si = slots_[size_t(slot)];
    si.timing = Timing{};
    Stopwatch restore_time(&si.timing.restore_s);
    si.drafts_for = -1;
    si.accept = 0.8f;
    ++si.epoch;
    auto &hist = si.hist;
    size_t common = 0;
    while (common < hist.size() && common < prompt.size() && hist[common] == prompt[common]) ++common;
    if (si.mtp_lag > 0 && common == hist.size() && prompt.size() > hist.size()) {
        // continuing after plain decoding: the MTP rows it skipped (the newest one runs in prefill)
        const int64_t p0 = int64_t(hist.size()) - si.mtp_lag;
        e_.mtp_catch_up(slot, p0, std::vector<int32_t>(hist.begin() + ptrdiff_t(p0 + 1), hist.end()));
    }
    si.mtp_lag = 0;
    si.plain_left = 0;
    si.plain_len = 32;

    // the block store, when it holds more of the prompt than this slot can
    si.pending.clear();
    if (restore_from_store(slot, prompt, reusable(slot, prompt), common) && common == prompt.size()) {
        si.prompt_end = int64_t(common);
        single_ = false;
        count_reuse(prompt, int64_t(common));
        return int64_t(common);
    }

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
                    set_prompt_logits(slot, s.logits);
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
            if (n == int64_t(prompt.size())) set_prompt_logits(slot, s.logits);
        } else {
            e_.slot_reset(slot);
            hist.clear();
            drop_snapshots_after(slot, 0);
        }
    }

    if (prompt.size() > hist.size()) {
        si.pending = prompt;  // prefilled by prefill_some
    } else {
        si.prompt_end = int64_t(hist.size());
        single_ = false;
    }
    count_reuse(prompt, reused);
    return reused;
}

void Session::decode(const std::vector<Engine::Row> &rows, const Engine::SampleSpec *spec) {
    for (const auto &r : rows) drop_snapshots_after(r.slot, int64_t(slots_[size_t(r.slot)].hist.size()));
    e_.decode(rows, spec);
    row_slot_.clear();
    for (const auto &r : rows) {
        slots_[size_t(r.slot)].hist.push_back(r.token);
        row_slot_.push_back(r.slot);
    }
    single_ = rows.size() == 1;
}

int32_t Session::sample_prompt(int slot, const SamplingParams &p, float *logprob) {
    const auto &l = slots_[size_t(slot)].logits;
    return sample_logits(l.empty() ? e_.logits().data() : l.data(), slot, p, logprob);
}

int32_t Session::sample_row(int row, const SamplingParams &p, float *logprob) {
    QW_CHECK(row >= 0 && row < int(row_slot_.size()), "sample_row: bad row");
    QW_CHECK(e_.logits_rows_valid(), "sample_row: the step kept no logits");
    // the slot's history already holds the rows after this one of its run
    const int slot = row_slot_[size_t(row)];
    size_t later = 0;
    for (size_t j = size_t(row) + 1; j < row_slot_.size() && row_slot_[j] == slot; ++j) ++later;
    return sample_logits(e_.logits_rows().data() + size_t(row) * cfg::VOCAB, slot, p, logprob,
                         e_.logits_rows_lse()[size_t(row)], slots_[size_t(slot)].hist.size() - later);
}

int32_t Session::sample(const SamplingParams &p, float *logprob) {
    if (single_) return sample_row(0, p, logprob);
    return sample_prompt(0, p, logprob);
}

int32_t Session::sample_logits(const float *raw, int slot, const SamplingParams &p, float *logprob, float lse_known,
                               size_t hist_len) {
    const SlotInfo &si = slots_[size_t(slot)];
    return sample_token(raw, p, si.hist, hist_len, si.prompt_end, rng_, logprob, lse_known);
}

void Session::top_logprobs_row(int row, int k, std::vector<int32_t> &ids, std::vector<float> &lps) const {
    QW_CHECK(e_.logits_rows_valid(), "top_logprobs_row: the step kept no logits (StepReq::want_logits)");
    top_logprobs(e_.logits_rows().data() + size_t(row) * cfg::VOCAB, k, ids, lps, e_.logits_rows_lse()[size_t(row)]);
}

void Session::top_logprobs_prompt(int k, std::vector<int32_t> &ids, std::vector<float> &lps) const {
    top_logprobs(e_.logits().data(), k, ids, lps);
}

void Session::top_logprobs_prompt(int slot, int k, std::vector<int32_t> &ids, std::vector<float> &lps) const {
    const auto &l = slots_[size_t(slot)].logits;
    top_logprobs(l.empty() ? e_.logits().data() : l.data(), k, ids, lps);
}

}  // namespace qw
