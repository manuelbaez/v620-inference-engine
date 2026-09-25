#include "engine/pinned_pool.hpp"

#include <hip/hip_runtime.h>

#include "engine/device_mem.hpp"

namespace qw {

PinnedPool::PinnedPool(size_t unit_bytes, int units_per_arena, int device, size_t low_water)
    : unit_(unit_bytes), per_arena_(units_per_arena), device_(device), low_water_(low_water) {
    prefetch_ = std::thread([this] { prefetch_loop(); });
}

PinnedPool::~PinnedPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    prefetch_.join();
    for (auto &a : arenas_) (void)hipHostFree(a->base);
}

std::unique_ptr<PinnedPool::Arena> PinnedPool::new_arena() const {
    auto a = std::make_unique<Arena>();
    CK(hipSetDevice(device_));
    a->base = pinned<uint8_t>(unit_ * size_t(per_arena_));
    for (int i = per_arena_ - 1; i >= 0; --i) a->free.push_back(i);
    return a;
}

void PinnedPool::add_arena(std::unique_ptr<Arena> a) {
    free_units_ += a->free.size();
    ++empty_;
    arenas_.push_back(std::move(a));
}

void PinnedPool::want(size_t n) {
    if (n > free_units_ && n > target_) {
        target_ = n;
        cv_.notify_all();
    }
}

void PinnedPool::reserve(size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    want(n);
}

void PinnedPool::prefetch_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return stop_ || free_units_ < target_; });
        if (stop_) return;
        busy_ = true;
        lk.unlock();
        auto a = new_arena();
        lk.lock();
        add_arena(std::move(a));
        if (free_units_ >= target_) target_ = 0;
        busy_ = false;
        ready_cv_.notify_all();
    }
}

uint8_t *PinnedPool::get() {
    std::unique_lock<std::mutex> lk(mu_);
    if (free_units_ == 0) {
        want(1);
        ready_cv_.wait(lk, [&] { return free_units_ > 0; });
    }
    for (auto &a : arenas_)
        if (!a->free.empty()) {
            if (int(a->free.size()) == per_arena_) --empty_;
            const int i = a->free.back();
            a->free.pop_back();
            --free_units_;
            want(low_water_);
            return a->base + size_t(i) * unit_;
        }
    fail("PinnedPool: free-unit count out of sync");
}

void PinnedPool::put(uint8_t *p) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t k = 0; k < arenas_.size(); ++k) {
        Arena &a = *arenas_[k];
        if (p < a.base || p >= a.base + unit_ * size_t(per_arena_)) continue;
        a.free.push_back(int((p - a.base) / ptrdiff_t(unit_)));
        ++free_units_;
        if (int(a.free.size()) == per_arena_ && ++empty_ > 1 &&
            free_units_ - size_t(per_arena_) >= low_water_) {  // keep one empty arena as a spare
            (void)hipHostFree(a.base);
            arenas_.erase(arenas_.begin() + ptrdiff_t(k));
            --empty_;
            free_units_ -= size_t(per_arena_);
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
