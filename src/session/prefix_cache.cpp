// Session's use of the block store (the host tier of the prefix cache):
// restoring a prompt's stored prefix into a slot, and snapshotting and saving
// prefixes while prefilling.
#include <algorithm>

#include "core/common.hpp"
#include "session/chunker.hpp"
#include "session/session.hpp"

namespace qw {

// Restores the store's deepest prefix of `prompt` into `slot` if it beats
// slot_reuse; common: tokens of the prompt the slot holds (updated).
bool Session::restore_from_store(int slot, const std::vector<int32_t> &prompt, size_t slot_reuse, size_t &common) {
    if (!store_) return false;
    const BlockStore::Hit hit = store_->lookup(prompt, int64_t(slot_reuse));
    if (hit.n == 0) return false;
    auto &hist = slots_[size_t(slot)].hist;
    std::vector<float> logits;
    drop_snapshots_after(slot, 0);
    if (!store_->restore(hit, prompt, slot, int64_t(common), &logits)) {
        e_.slot_reset(slot);
        hist.clear();
        common = 0;
        return false;
    }
    hist.assign(prompt.begin(), prompt.begin() + ptrdiff_t(hit.n));
    common = hist.size();
    if (common == prompt.size()) e_.set_logits(logits);
    return true;
}

std::vector<Engine::Capture> Session::plan_captures(const std::vector<int32_t> &prompt, int64_t from) {
    std::vector<Engine::Capture> caps;
    if (!store_ || boundary_ < 0) return caps;
    const int64_t chunk = e_.prefill_chunk(), n = int64_t(prompt.size());
    // chunk k of this prefill covers positions (from + k * chunk, from + (k + 1) * chunk]
    std::vector<std::vector<int64_t>> per_chunk(size_t((n - from + chunk - 1) / chunk));
    for (int64_t p = from + 1, last = from; p < n; ++p) {
        if (prompt[size_t(p)] != boundary_ || p - last < min_gap_) continue;
        last = p;
        if ((p - from) % chunk) per_chunk[size_t((p - from - 1) / chunk)].push_back(p);  // chunk ends are saved anyway
    }
    size_t most = 0;
    for (auto &c : per_chunk) {
        if (c.size() > size_t(Engine::MAX_CAPTURES))  // keep the first (e.g. the system prompt's end) and the last ones
            c.erase(c.begin() + 1, c.end() - (Engine::MAX_CAPTURES - 1));
        most = std::max(most, c.size());
    }
    // VRAM snapshots to capture into, reused by every chunk (each chunk's are saved before the next runs)
    reserved_.clear();
    while (reserved_.size() < most) {
        int idx = -1;
        for (int i = 0; i < int(snaps_.size()); ++i) {
            if (std::find(reserved_.begin(), reserved_.end(), i) != reserved_.end()) continue;
            const Snap &s = snaps_[size_t(i)];
            if (idx < 0 || (!s.valid && snaps_[size_t(idx)].valid) ||
                (s.valid == snaps_[size_t(idx)].valid && s.used < snaps_[size_t(idx)].used))
                idx = i;
        }
        snaps_[size_t(idx)].valid = false;
        reserved_.push_back(idx);
    }
    for (const auto &c : per_chunk)
        for (size_t j = 0; j < c.size(); ++j) caps.push_back({c[j], reserved_[j]});
    return caps;
}

// Prefills prompt[hist.size()..) into the slot, snapshotting at chunk ends
// and at the planned captures, and saving those to the block store.
void Session::prefill_rest(int slot, const std::vector<int32_t> &prompt) {
    auto &hist = slots_[size_t(slot)].hist;
    const int64_t from = int64_t(hist.size());
    const std::vector<int32_t> rest(prompt.begin() + ptrdiff_t(from), prompt.end());
    drop_snapshots_after(slot, from);
    const auto caps = plan_captures(prompt, from);
    if (store_) {  // pin the saves' memory while the GPUs prefill
        const size_t n = prompt.size(), chunks = size_t((int64_t(n) - from + e_.prefill_chunk() - 1) / e_.prefill_chunk());
        store_->reserve((n - size_t(from)) / BlockStore::BLOCK + 2 * (caps.size() + chunks), caps.size() + chunks);
    }
    const auto embeds = vision_embeds(from);
    size_t next = 0;
    e_.prefill(
        slot, rest, nullptr,
        [&](int64_t pos) {
            for (; next < caps.size() && caps[next].pos <= pos; ++next)
                store_->save(slot, caps[next].snap,
                             std::vector<int32_t>(prompt.begin(), prompt.begin() + ptrdiff_t(caps[next].pos)), nullptr);
            hist.insert(hist.end(), prompt.begin() + ptrdiff_t(hist.size()), prompt.begin() + ptrdiff_t(pos));
            const int idx = save_snapshot(slot);
            if (store_ && pos >= min_gap_) store_->save(slot, idx, hist, &snaps_[size_t(idx)].logits);
        },
        caps, embeds);
    reserved_.clear();
}

void Session::count_reuse(const std::vector<int32_t> &prompt, int64_t reused) {
    const uint64_t id = ++clock_;
    counters_.prompt_tokens += prompt.size();
    counters_.reused_tokens += uint64_t(reused);
    for (const Chunk &c : chunk_prompt(prompt, boundary_)) {
        auto it = seen_chunks_.find(c.hash);
        if (it != seen_chunks_.end() && it->second != id && c.start >= reused) counters_.blend_candidate_tokens += c.len;
        seen_chunks_[c.hash] = id;
    }
    if (seen_chunks_.size() > (1u << 20)) {  // keep the more recently seen half
        std::vector<uint64_t> ids;
        for (const auto &kv : seen_chunks_) ids.push_back(kv.second);
        std::nth_element(ids.begin(), ids.begin() + ptrdiff_t(ids.size() / 2), ids.end());
        const uint64_t cut = ids[ids.size() / 2];
        for (auto it = seen_chunks_.begin(); it != seen_chunks_.end();)
            it = it->second < cut ? seen_chunks_.erase(it) : std::next(it);
    }
}

void Session::persist() {
    if (!store_) return;
    for (int slot = 0; slot < num_slots(); ++slot) {  // each slot's newest snapshot
        int best = -1;
        for (int i = 0; i < int(snaps_.size()); ++i) {
            const Snap &s = snaps_[size_t(i)];
            if (s.valid && s.slot == slot && (best < 0 || s.tokens.size() > snaps_[size_t(best)].tokens.size()))
                best = i;
        }
        if (best < 0) continue;
        const Snap &s = snaps_[size_t(best)];
        if (int64_t(s.tokens.size()) < min_gap_ || store_->has_snapshot(s.tokens)) continue;
        store_->save(slot, best, s.tokens, &s.logits);
    }
    store_->flush();
    log("prefix cache: persisted the slots' conversations");
}

}  // namespace qw
