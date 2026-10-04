#include "engine/pinned_pool.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>

#include "engine/device_mem.hpp"

namespace qw {

namespace {
constexpr double SLOW_ARENA_S = 0.5;  // pinning or freeing an arena slower than this is logged
constexpr double IDLE_S = 1.5;        // the standing reserve is topped up after this long without a get()
constexpr int STANDING_GAP_MS = 200;  // ... one arena per this: pins hold the process's memory-map lock

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

PoolOptions PoolOptions::from_env() {
    PoolOptions o;
    if (const char *t = std::getenv("QW_POOL_SERIAL")) o.serial = std::atoi(t) != 0;
    if (const char *a = std::getenv("QW_POOL_ARENA")) {
        const std::string v = a;
        if (v == "malloc") o.arena = Arena::HostMalloc;
        else if (v != "noncoherent") log("QW_POOL_ARENA=%s is not malloc or noncoherent; using noncoherent", a);
    }
    return o;
}

const char *PoolOptions::name(Arena a) {
    return a == Arena::NonCoherent ? "non-coherent" : "hipHostMalloc";
}

ArenaBank::ArenaBank(size_t arena_bytes, size_t count, const std::vector<int> &devices, PoolOptions opt)
    : bytes_(arena_bytes) {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; ++i) {
        CK(hipSetDevice(devices[i % devices.size()]));
        void *p = nullptr;
        CK(hipHostMalloc(&p, bytes_, opt.arena == PoolOptions::Arena::NonCoherent ? hipHostMallocNonCoherent : 0u));
        all_.push_back(static_cast<uint8_t *>(p));
    }
    free_ = all_;
    pin_s = since(t0);
}

ArenaBank::~ArenaBank() {
    for (uint8_t *p : all_) (void)hipHostFree(p);
}

uint8_t *ArenaBank::take() {
    std::lock_guard<std::mutex> lk(mu_);
    if (free_.empty()) return nullptr;
    uint8_t *p = free_.back();
    free_.pop_back();
    return p;
}

void ArenaBank::give(uint8_t *p) {
    std::lock_guard<std::mutex> lk(mu_);
    free_.push_back(p);
}

size_t ArenaBank::left() const {
    std::lock_guard<std::mutex> lk(mu_);
    return free_.size();
}

PinnedPool::PinnedPool(size_t unit_bytes, int units_per_arena, int device, size_t standing, PoolOptions opt,
                       ArenaBank *bank)
    : unit_(unit_bytes), per_arena_(bank ? int(bank->arena_bytes() / unit_bytes) : units_per_arena), device_(device),
      standing_(standing), opt_(opt), bank_(bank) {
    QW_CHECK(per_arena_ > 0, "PinnedPool: the bank's arenas are smaller than a unit");
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

std::unique_ptr<PinnedPool::Arena> PinnedPool::new_arena() {
    static std::mutex gate;  // shared by every pool (PoolOptions::serial)
    std::unique_lock<std::mutex> turn(gate, std::defer_lock);
    if (opt_.serial) turn.lock();
    const auto t0 = std::chrono::steady_clock::now();
    auto a = std::make_unique<Arena>();
    if (bank_)
        if (uint8_t *p = bank_->take()) {  // already pinned: nothing to wait for
            a->base = p;
            a->banked = true;
            for (int i = per_arena_ - 1; i >= 0; --i) a->free.push_back(i);
            return a;
        }
    CK(hipSetDevice(device_));
    const size_t bytes = unit_ * size_t(per_arena_);
    if (opt_.arena == PoolOptions::Arena::NonCoherent) {
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
    if (s > SLOW_ARENA_S) log("pinned pool (device %d): pinning a %.0f MB arena took %.2f s", device_, double(bytes) / 1e6, s);
    return a;
}

void PinnedPool::free_arena(Arena &a) const {
    if (a.banked) {
        bank_->give(a.base);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    (void)hipSetDevice(device_);
    (void)hipHostFree(a.base);
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
    using clk = std::chrono::steady_clock;
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        // work due now: a reserve() or a starved get() (target_), arenas to unpin, or the standing reserve after a
        // quiet period (checked again before every arena, so a get() pauses the refill)
        const auto standing_due = [&] {
            return free_units_ < standing_ && clk::now() >= last_get_ + std::chrono::duration_cast<clk::duration>(
                                                                           std::chrono::duration<double>(IDLE_S));
        };
        while (!(stop_ || free_units_ < target_ || !retired_.empty() || standing_due())) {
            if (free_units_ < standing_)
                cv_.wait_until(lk, last_get_ + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(IDLE_S)));
            else
                cv_.wait(lk);
        }
        if (stop_) return;
        while (!retired_.empty()) {  // arenas put() gave up: unpin them off the callers' threads
            auto a = std::move(retired_.back());
            retired_.pop_back();
            lk.unlock();
            free_arena(*a);
            lk.lock();
        }
        const bool standing = free_units_ >= target_ && standing_due();
        if (free_units_ >= target_ && !standing) {  // units were put back meanwhile: nothing to pin
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
        if (standing && free_units_ < standing_) {  // a gentle refill: one arena, then a pause (lock released)
            lk.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(STANDING_GAP_MS));
            lk.lock();
        }
    }
}

namespace {
thread_local double t_wait_s = 0;
}

double PinnedPool::thread_wait_s() {
    return t_wait_s;
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
        t_wait_s += since(t0);
    }
    for (auto &a : arenas_)
        if (!a->free.empty()) {
            if (int(a->free.size()) == per_arena_) --empty_;
            const int i = a->free.back();
            a->free.pop_back();
            --free_units_;
            last_get_ = std::chrono::steady_clock::now();
            if (free_units_ < standing_) cv_.notify_all();  // the refill thread times the quiet period from here
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
        if (int(a.free.size()) == per_arena_) {  // the arena is empty now
            ++empty_;
            // keep one empty arena as a spare, and the standing reserve; give the rest back
            if (empty_ > 1 && free_units_ - size_t(per_arena_) >= standing_) {
                retired_.push_back(std::move(arenas_[k]));
                arenas_.erase(arenas_.begin() + ptrdiff_t(k));
                --empty_;
                free_units_ -= size_t(per_arena_);
                cv_.notify_all();
            }
        }
        return;
    }
    fail("PinnedPool::put: foreign pointer");
}

bool PinnedPool::standing_ready() const {
    std::lock_guard<std::mutex> lk(mu_);
    return free_units_ >= standing_;
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
