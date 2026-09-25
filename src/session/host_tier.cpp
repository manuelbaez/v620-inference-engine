// Host-RAM tier of the prefix cache: conversations evicted from their slot are
// kept in pinned host memory (KV + recurrent state at a snapshot point, see
// Engine::export_state) and restored instead of re-prefilled.
#include <algorithm>
#include <cstdlib>

#include "core/common.hpp"
#include "session/session.hpp"

namespace qw {

namespace {
bool is_prefix(const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}
}  // namespace

// Saves the slot's longest snapshot to the tier unless the next prompt keeps
// it anyway (or it is short, or already saved).
void Session::offload_slot(int slot, const std::vector<int32_t> &next_prompt) {
    if (host_budget_ == 0) return;
    int best = -1;
    for (int i = 0; i < int(snaps_.size()); ++i) {
        const Snap &s = snaps_[size_t(i)];
        if (s.valid && s.slot == slot && (best < 0 || s.tokens.size() > snaps_[size_t(best)].tokens.size())) best = i;
    }
    if (best < 0) return;
    const Snap &s = snaps_[size_t(best)];
    if (s.tokens.size() < host_min_tokens_ || is_prefix(s.tokens, next_prompt)) return;
    for (const HostEntry &h : host_)
        if (h.tokens == s.tokens) return;
    HostEntry h;
    h.tokens = s.tokens;
    h.logits = s.logits;
    h.state = e_.export_state(slot, best, int64_t(s.tokens.size()));
    if (disk_) disk_->store(h.tokens, h.logits, h.state);  // written in the background
    log("host tier: saved %zu tokens of slot %d", h.tokens.size(), slot);
    add_host_entry(std::move(h));
}

void Session::persist() {
    if (!disk_) return;
    for (int slot = 0; slot < num_slots(); ++slot) {
        int best = -1;
        for (int i = 0; i < int(snaps_.size()); ++i) {
            const Snap &s = snaps_[size_t(i)];
            if (s.valid && s.slot == slot && (best < 0 || s.tokens.size() > snaps_[size_t(best)].tokens.size()))
                best = i;
        }
        if (best < 0) continue;
        const Snap &s = snaps_[size_t(best)];
        if (s.tokens.size() < host_min_tokens_ || disk_->has(s.tokens)) continue;
        disk_->store(s.tokens, s.logits, e_.export_state(slot, best, int64_t(s.tokens.size())));
    }
    disk_->flush();
    log("disk tier: persisted the slots' conversations");
}

// Adds an entry to the RAM tier, evicting least recently used ones to fit.
void Session::add_host_entry(HostEntry h) {
    h.bytes = Engine::host_state_bytes(*h.state) + h.logits.size() * 4;
    h.used = ++clock_;
    if (h.bytes > host_budget_) return;
    while (host_bytes_ + h.bytes > host_budget_ && !host_.empty()) {
        auto lru = std::min_element(host_.begin(), host_.end(),
                                    [](const HostEntry &a, const HostEntry &b) { return a.used < b.used; });
        host_bytes_ -= lru->bytes;
        host_.erase(lru);
    }
    host_bytes_ += h.bytes;
    host_.push_back(std::move(h));
}

// Brings the disk tier's longest prefix of `prompt` into the RAM tier if it
// beats `at_least` tokens. Returns whether it did.
bool Session::promote_from_disk(const std::vector<int32_t> &prompt, size_t at_least) {
    if (!disk_) return false;
    const size_t len = disk_->best(prompt, at_least);
    if (len == 0) return false;
    HostEntry h;
    if (!disk_->load(prompt, len, h.tokens, h.logits, h.state)) return false;
    add_host_entry(std::move(h));
    return true;
}

// The entry holding the longest prefix of `prompt`, if longer than at_least.
int Session::best_host_entry(const std::vector<int32_t> &prompt, size_t at_least) const {
    int best = -1;
    for (int i = 0; i < int(host_.size()); ++i) {
        const HostEntry &h = host_[size_t(i)];
        if (h.tokens.size() > at_least && is_prefix(h.tokens, prompt) &&
            (best < 0 || h.tokens.size() > host_[size_t(best)].tokens.size()))
            best = i;
    }
    return best;
}

}  // namespace qw
