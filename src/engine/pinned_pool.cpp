#include "engine/pinned_pool.hpp"

#include <hip/hip_runtime.h>

#include "engine/device_mem.hpp"

namespace qw {

PinnedPool::PinnedPool(size_t unit_bytes, int units_per_arena, int device)
    : unit_(unit_bytes), per_arena_(units_per_arena), device_(device) {}

PinnedPool::~PinnedPool() {
    for (auto &a : arenas_) (void)hipHostFree(a->base);
}

uint8_t *PinnedPool::get() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &a : arenas_)
        if (!a->free.empty()) {
            if (int(a->free.size()) == per_arena_) --empty_;
            const int i = a->free.back();
            a->free.pop_back();
            return a->base + size_t(i) * unit_;
        }
    auto a = std::make_unique<Arena>();
    int prev = 0;
    CK(hipGetDevice(&prev));
    CK(hipSetDevice(device_));
    a->base = pinned<uint8_t>(unit_ * size_t(per_arena_));
    CK(hipSetDevice(prev));
    for (int i = per_arena_ - 1; i >= 1; --i) a->free.push_back(i);
    uint8_t *p = a->base;
    arenas_.push_back(std::move(a));
    return p;
}

void PinnedPool::put(uint8_t *p) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t k = 0; k < arenas_.size(); ++k) {
        Arena &a = *arenas_[k];
        if (p < a.base || p >= a.base + unit_ * size_t(per_arena_)) continue;
        a.free.push_back(int((p - a.base) / ptrdiff_t(unit_)));
        if (int(a.free.size()) == per_arena_ && ++empty_ > 1) {  // keep one empty arena as a spare
            (void)hipHostFree(a.base);
            arenas_.erase(arenas_.begin() + ptrdiff_t(k));
            --empty_;
        }
        return;
    }
    fail("PinnedPool::put: foreign pointer");
}

size_t PinnedPool::allocated_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return arenas_.size() * unit_ * size_t(per_arena_);
}

}  // namespace qw
