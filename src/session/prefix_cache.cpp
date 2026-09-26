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
    if (common == prompt.size()) set_prompt_logits(slot, logits);
    return true;
}

std::vector<Engine::Capture> Session::plan_captures(const std::vector<int32_t> &prompt, int64_t from, int64_t to,
                                                    int max_caps, bool append) {
    std::vector<Engine::Capture> caps;
    if (!store_ || boundary_ < 0) return caps;
    const int64_t chunk = e_.prefill_chunk(), n = to;
    // chunk k of this prefill covers positions (from + k * chunk, from + (k + 1) * chunk]
    std::vector<std::vector<int64_t>> per_chunk(size_t((n - from + chunk - 1) / chunk));
    for (int64_t p = from + 1, last = from; p < n; ++p) {
        if (prompt[size_t(p)] != boundary_ || p - last < min_gap_) continue;
        last = p;
        if ((p - from) % chunk) per_chunk[size_t((p - from - 1) / chunk)].push_back(p);  // chunk ends are saved anyway
    }
    size_t most = 0;
    for (auto &c : per_chunk) {
        if (c.size() > size_t(max_caps))  // keep the first (e.g. the system prompt's end) and the last ones
            c.erase(c.begin() + 1, c.end() - (max_caps - 1));
        most = std::max(most, c.size());
    }
    // VRAM snapshots to capture into, reused by every chunk (each chunk's are saved before the next runs)
    if (!append) reserved_.clear();
    const size_t base = reserved_.size();
    while (reserved_.size() < base + most) {
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
        for (size_t j = 0; j < c.size(); ++j) caps.push_back({c[j], reserved_[base + j]});
    return caps;
}

// Prefills prompt[hist.size(), to) into the slot, snapshotting at chunk ends
// and at the planned captures, and saving those to the block store.
void Session::prefill_range(int slot, const std::vector<int32_t> &prompt, int64_t to) {
    auto &hist = slots_[size_t(slot)].hist;
    const int64_t from = int64_t(hist.size());
    const std::vector<int32_t> rest(prompt.begin() + ptrdiff_t(from), prompt.begin() + ptrdiff_t(to));
    drop_snapshots_after(slot, from);
    const auto caps = plan_captures(prompt, from, to);
    if (store_) {  // pin the saves' memory while the GPUs prefill
        const size_t n = size_t(to), chunks = size_t((int64_t(n) - from + e_.prefill_chunk() - 1) / e_.prefill_chunk());
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
            // to the store at the prompt's end and every prefill chunk, not at every piece
            const bool keep = pos == int64_t(prompt.size()) || pos % e_.prefill_chunk() == 0;
            if (store_ && keep && pos >= min_gap_) store_->save(slot, idx, hist, &snaps_[size_t(idx)].logits);
        },
        caps, embeds);
    reserved_.clear();
}

std::vector<bool> Session::prefill_batch(const std::vector<std::pair<int, int64_t>> &reqs) {
    struct Piece {
        int slot;
        int64_t from, to;
    };
    std::vector<Piece> pieces;
    int64_t total = 0;
    bool batchable = true;
    for (const auto &[slot, max_tokens] : reqs) {
        const SlotInfo &si = slots_[size_t(slot)];
        if (si.pending.empty()) continue;
        const int64_t from = int64_t(si.hist.size()), n = int64_t(si.pending.size());
        const int64_t to = max_tokens >= n - from ? n : from + std::max<int64_t>(1, max_tokens);
        pieces.push_back({slot, from, to});
        total += to - from;
        batchable = batchable && si.pending_media.empty();
    }
    std::vector<bool> done;
    if (pieces.size() < 2 || !batchable || total > e_.prefill_chunk() ||
        pieces.size() > size_t(Engine::MAX_SEGMENTS)) {
        for (const auto &[slot, max_tokens] : reqs) done.push_back(prefill_some(slot, max_tokens));
        return done;
    }
    // snapshots to capture into: at most half the VRAM pool over all segments, the rest for their ends
    const int caps_each = std::max(1, Engine::SNAPSHOTS / 2 / int(pieces.size()));
    std::vector<Engine::Segment> segs;
    size_t blocks = 0, snaps = 0;
    reserved_.clear();
    for (const Piece &pc : pieces) {
        const auto &prompt = slots_[size_t(pc.slot)].pending;
        drop_snapshots_after(pc.slot, pc.from);
        auto caps = plan_captures(prompt, pc.from, pc.to, caps_each, true);
        blocks += size_t(pc.to - pc.from) / BlockStore::BLOCK + 2 * (caps.size() + 1);
        snaps += caps.size() + 1;
        segs.push_back({pc.slot, std::vector<int32_t>(prompt.begin() + ptrdiff_t(pc.from), prompt.begin() + ptrdiff_t(pc.to)),
                        std::move(caps)});
    }
    if (store_) store_->reserve(blocks, snaps);  // pin the saves' memory while the GPUs prefill
    const auto &logits = e_.prefill_batch(segs);
    for (size_t i = 0; i < pieces.size(); ++i) {  // what prefill_range does after a chunk, per segment
        const Piece &pc = pieces[i];
        SlotInfo &si = slots_[size_t(pc.slot)];
        const auto &prompt = si.pending;
        for (const Engine::Capture &c : segs[i].captures)
            store_->save(pc.slot, c.snap, std::vector<int32_t>(prompt.begin(), prompt.begin() + ptrdiff_t(c.pos)),
                         nullptr);
        si.hist.insert(si.hist.end(), prompt.begin() + ptrdiff_t(pc.from), prompt.begin() + ptrdiff_t(pc.to));
        const int idx = save_snapshot(pc.slot, &logits[i]);
        const bool keep = pc.to == int64_t(prompt.size()) || pc.to % e_.prefill_chunk() == 0;
        if (store_ && keep && pc.to >= min_gap_) store_->save(pc.slot, idx, si.hist, &snaps_[size_t(idx)].logits);
    }
    reserved_.clear();
    std::vector<bool> finished(pieces.size());
    for (size_t i = 0; i < pieces.size(); ++i) {
        SlotInfo &si = slots_[size_t(pieces[i].slot)];
        finished[i] = int64_t(si.hist.size()) == int64_t(si.pending.size());
        if (!finished[i]) continue;
        si.logits = logits[i];
        si.prompt_end = int64_t(si.pending.size());
        si.pending.clear();
    }
    single_ = false;
    for (const auto &[slot, max_tokens] : reqs) {
        bool d = slots_[size_t(slot)].pending.empty();
        for (size_t i = 0; i < pieces.size(); ++i)
            if (pieces[i].slot == slot) d = finished[i];
        done.push_back(d);
    }
    return done;
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
