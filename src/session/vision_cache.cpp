// Session's side of vision inputs: media tokens as content-derived ids (so the
// prefix cache tells images apart), and their embeddings for a prefill, from
// an LRU cache of vision-tower outputs or encoded on the cards on demand.
#include <algorithm>
#include <chrono>
#include <memory>

#include "core/common.hpp"
#include "core/config.hpp"
#include "session/session.hpp"

namespace qw {

namespace {

uint64_t mix(uint64_t x) {
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 29;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 32);
}

// Token j of a media item: a negative id, odd for images and even for videos
// (cfg::real_token), from the item's hash and j.
int32_t media_key(uint64_t hash, int64_t j, bool video) {
    const int32_t base = int32_t(2 + 2 * (mix(hash ^ (uint64_t(j) * 0x9E3779B97F4A7C15ull)) & 0x1FFFFFFF));
    return video ? -base : -(base + 1);
}

uint64_t slice_key(uint64_t hash, int slice) {
    return mix(hash + uint64_t(slice) * 0x9E3779B97F4A7C15ull);
}

}  // namespace

std::vector<int32_t> Session::media_keys(const std::vector<int32_t> &prompt, const std::vector<Media> &media) const {
    std::vector<int32_t> keys = prompt;
    for (const Media &m : media) {
        QW_CHECK(int(m.starts.size()) == m.t, "media: one start per temporal slice");
        const int64_t n = int64_t(m.h) * m.w / 4;
        const int32_t pad = m.video ? cfg::VIDEO_PAD : cfg::IMAGE_PAD;
        for (int s = 0; s < m.t; ++s)
            for (int64_t j = 0; j < n; ++j) {
                const int64_t p = m.starts[size_t(s)] + j;
                QW_CHECK(p >= 0 && p < int64_t(prompt.size()) && prompt[size_t(p)] == pad,
                         "media: its tokens must be its pad token in the prompt");
                keys[size_t(p)] = media_key(m.hash, s * n + j, m.video);
            }
    }
    return keys;
}

std::vector<Engine::EmbedSpan> Session::vision_embeds(int64_t from) {
    std::vector<Engine::EmbedSpan> spans;
    if (!media_) return spans;
    struct Need {
        const Media *m;
        int slice;
        uint64_t key;
    };
    std::vector<Need> need;
    for (const Media &m : *media_) {
        const int64_t n = int64_t(m.h) * m.w / 4;
        for (int s = 0; s < m.t; ++s)
            if (m.starts[size_t(s)] + n > from) need.push_back({&m, s, slice_key(m.hash, s)});
    }
    // encode the slices the cache lacks, spread over the cards
    std::vector<Engine::VisionSlice> todo;
    for (const Need &nd : need) {
        auto &e = vision_cache_[nd.key];
        if (!e.rows.empty()) continue;
        e.rows.resize(size_t(nd.m->h) * nd.m->w / 4 * cfg::H);
        vision_bytes_ += e.rows.size() * 4;
        todo.push_back({nd.m->patches + size_t(nd.slice) * nd.m->h * nd.m->w * 1536, nd.m->h, nd.m->w,
                        e.rows.data()});
    }
    if (!todo.empty()) {
        QW_CHECK(e_.has_vision(), "the prompt has images or videos but the engine has no vision tower");
        const auto t0 = std::chrono::steady_clock::now();
        e_.encode_vision(todo);
        size_t tokens = 0;
        for (const auto &s : todo) tokens += size_t(s.h) * s.w / 4;
        log("vision: %zu slices, %zu tokens encoded in %.3f s", todo.size(), tokens,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    for (const Need &nd : need) {
        auto &e = vision_cache_.at(nd.key);
        e.used = ++clock_;
        spans.push_back({nd.m->starts[size_t(nd.slice)], int64_t(nd.m->h) * nd.m->w / 4, e.rows.data()});
    }
    // LRU within the budget, never evicting what this prompt uses
    while (vision_bytes_ > vision_budget_) {
        auto lru = vision_cache_.end();
        for (auto it = vision_cache_.begin(); it != vision_cache_.end(); ++it)
            if (it->second.used < clock_ - need.size() &&
                (lru == vision_cache_.end() || it->second.used < lru->second.used))
                lru = it;
        if (lru == vision_cache_.end()) break;
        vision_bytes_ -= lru->second.rows.size() * 4;
        vision_cache_.erase(lru);
    }
    return spans;
}

int64_t Session::set_prompt(int slot, const std::vector<int32_t> &prompt, const std::vector<Media> &media) {
    if (media.empty()) return set_prompt(slot, prompt);
    media_ = &media;
    try {
        const int64_t r = set_prompt(slot, media_keys(prompt, media));
        media_ = nullptr;
        return r;
    } catch (...) {
        media_ = nullptr;
        throw;
    }
}

}  // namespace qw
