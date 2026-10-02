#include "engine/pinned_pool.hpp"

#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>

#include "engine/device_mem.hpp"

namespace qw {

namespace {
constexpr double SLOW_ARENA_S = 0.5;  // pinning or freeing an arena slower than this is logged
constexpr size_t HUGE_PAGE = size_t(2) << 20;

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

PoolOptions PoolOptions::from_env() {
    PoolOptions o;
    if (const char *t = std::getenv("QW_POOL_SERIAL")) o.serial = std::atoi(t) != 0;
    if (const char *a = std::getenv("QW_POOL_ARENA")) {
        const std::string v = a;
        if (v == "noncoherent") o.arena = Arena::NonCoherent;
        else if (v == "huge") o.arena = Arena::Huge;
        else if (v != "malloc") log("QW_POOL_ARENA=%s is not malloc, noncoherent or huge; using malloc", a);
    }
    return o;
}

const char *PoolOptions::name(Arena a) {
    return a == Arena::Huge ? "huge pages" : a == Arena::NonCoherent ? "non-coherent" : "hipHostMalloc";
}

PinnedPool::PinnedPool(size_t unit_bytes, int units_per_arena, int device, size_t low_water, PoolOptions opt)
    : unit_(unit_bytes), per_arena_(units_per_arena), device_(device), low_water_(low_water), opt_(opt) {
    prefetch_ = std::thread([this] { prefetch_loop(); });
}

PinnedPool::~PinnedPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    prefetch_.join();
    (void)hipSetDevice(device_);
    for (auto &a : arenas_) free_arena(*a);
    for (auto &a : retired_) free_arena(*a);
}

// An arena as a 2 MB-aligned anonymous mapping, advised for huge pages and registered with HIP. False (nothing
// left behind) when the mapping or the registration fails; the caller then falls back to hipHostMalloc.
bool PinnedPool::map_arena(Arena &a, size_t bytes) const {
    const size_t len = (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
    void *m = mmap(nullptr, len + HUGE_PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return false;
    auto *base = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(m) + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1));
    auto *end = base + len, *map_end = static_cast<uint8_t *>(m) + len + HUGE_PAGE;
    if (base > m) munmap(m, size_t(base - static_cast<uint8_t *>(m)));  // the slack around the aligned range
    if (map_end > end) munmap(end, size_t(map_end - end));
    (void)madvise(base, len, MADV_HUGEPAGE);
    if (hipHostRegister(base, len, hipHostRegisterPortable) != hipSuccess) {
        (void)hipGetLastError();
        munmap(base, len);
        return false;
    }
    a.base = base;
    a.map_bytes = len;
    return true;
}

std::unique_ptr<PinnedPool::Arena> PinnedPool::new_arena() {
    static std::mutex gate;  // shared by every pool (PoolOptions::serial)
    std::unique_lock<std::mutex> turn(gate, std::defer_lock);
    if (opt_.serial) turn.lock();
    const auto t0 = std::chrono::steady_clock::now();
    auto a = std::make_unique<Arena>();
    CK(hipSetDevice(device_));
    const size_t bytes = unit_ * size_t(per_arena_);
    if (opt_.arena == PoolOptions::Arena::Huge && map_arena(*a, bytes)) {
        // mapped and registered
    } else if (opt_.arena == PoolOptions::Arena::NonCoherent) {
        void *p = nullptr;
        CK(hipHostMalloc(&p, bytes, hipHostMallocNonCoherent));
        a->base = static_cast<uint8_t *>(p);
    } else {
        a->base = pinned<uint8_t>(bytes);
    }
    for (int i = per_arena_ - 1; i >= 0; --i) a->free.push_back(i);
    const double s = since(t0);
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.pins;
        stats_.pin_s += s;
    }
    if (s > SLOW_ARENA_S)
        log("pinned pool (device %d): pinning a %.0f MB arena took %.2f s%s", device_, double(bytes) / 1e6, s,
            a->map_bytes ? " (huge pages)" : "");
    return a;
}

void PinnedPool::free_arena(Arena &a) const {
    const auto t0 = std::chrono::steady_clock::now();
    (void)hipSetDevice(device_);
    if (a.map_bytes) {
        (void)hipHostUnregister(a.base);
        munmap(a.base, a.map_bytes);
    } else {
        (void)hipHostFree(a.base);
    }
    if (const double s = since(t0); s > SLOW_ARENA_S)
        log("pinned pool (device %d): freeing a %.0f MB arena took %.2f s", device_,
            double(unit_ * size_t(per_arena_)) / 1e6, s);
}

void PinnedPool::add_arena(std::unique_ptr<Arena> a) {
    free_units_ += a->free.size();
    ++empty_;
    arenas_.push_back(std::move(a));
}

void PinnedPool::want(size_t n) {
    if (n <= free_units_) return;
    target_ = std::max(target_, n);
    cv_.notify_all();  // even when target_ already covers n: it never blocks a wakeup the thread owes
}

void PinnedPool::reserve(size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    want(n);
}

void PinnedPool::prefetch_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return stop_ || free_units_ < target_ || !retired_.empty(); });
        if (stop_) return;
        while (!retired_.empty()) {  // arenas put() gave up: unpin them off the callers' threads
            auto a = std::move(retired_.back());
            retired_.pop_back();
            lk.unlock();
            free_arena(*a);
            lk.lock();
        }
        if (free_units_ >= target_) {  // units were put back meanwhile: nothing to pin
            target_ = 0;
            continue;
        }
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
    // Waits for an arena, or for a unit another thread puts back. A waiter that wakes to find the
    // units gone (another caller took them first) asks for an arena again: with several callers
    // at once (the scheduler and the disk loads) a single request could be left unserved forever.
    if (free_units_ == 0) {
        const auto t0 = std::chrono::steady_clock::now();
        while (free_units_ == 0) {
            want(1);
            ready_cv_.wait(lk);
        }
        ++stats_.waits;
        stats_.wait_s += since(t0);
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
        ready_cv_.notify_one();  // a caller waiting in get() can take it
        if (int(a.free.size()) == per_arena_ && ++empty_ > 1 &&
            free_units_ - size_t(per_arena_) >= low_water_) {  // keep one empty arena as a spare
            retired_.push_back(std::move(arenas_[k]));
            arenas_.erase(arenas_.begin() + ptrdiff_t(k));
            --empty_;
            free_units_ -= size_t(per_arena_);
            cv_.notify_all();
        }
        return;
    }
    fail("PinnedPool::put: foreign pointer");
}

size_t PinnedPool::allocated_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return arenas_.size() * unit_ * size_t(per_arena_);
}

PoolStats PinnedPool::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
}

}  // namespace qw
