// Fixed-size pinned host buffers, carved from arenas that are allocated on
// demand and freed once empty (hipHostMalloc is slow per call, and the prefix
// cache holds thousands of buffers). Pinning an arena took 0.05-0.1 s when
// measured first and 0.7-1.5 s with the four pools pinning at once (2026-10-01),
// so arenas are allocated by a background thread: ahead of a known
// need (reserve()), and whenever the free units drop below a low-water mark.
// Freeing an arena (unpinning, which also waits for the device's streams) is
// done by that thread too, so put() never stalls the caller (the scheduler
// thread returns units in bulk when the cache evicts). Arena pins and frees
// that take long (a host short of memory has to reclaim first) are logged.
// Thread-safe.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace qw {

struct PoolOptions {
    // How an arena is made. HostMalloc: hipHostMalloc with default flags (coherent, fine-grained memory on this
    // stack). NonCoherent (the default): hipHostMalloc with hipHostMallocNonCoherent, cached host memory: the usual
    // choice for staging buffers filled and read between explicit stream syncs, and half the pin cost of the default
    // flags on the bad days. (Arenas from an anonymous mapping advised for huge pages and registered with
    // hipHostRegister were slower than both on this host, where compaction fails: docs/DESIGN.md.)
    enum class Arena { HostMalloc, NonCoherent };
    Arena arena = Arena::NonCoherent;
    // One arena is pinned at a time in the whole process (a gate shared by every pool): pins hold the process's
    // memory-map lock, so overlapping ones make each other, and every thread that maps memory, wait longer.
    bool serial = false;  // QW_POOL_SERIAL=1
    // Arenas are kept once pinned: no unpinning when one empties (put() would hand it to the pool's thread, and
    // unmapping 256 MB holds the process's memory-map lock in write mode as much as pinning it does). A cache at its
    // budget evicts whole conversations, whose arenas would otherwise be unpinned and pinned again for the next
    // saves. The pinned total stays at the high-water mark the cache already reached.
    bool keep = false;  // QW_POOL_KEEP=1
    // QW_POOL_ARENA=malloc|noncoherent (default noncoherent)
    static PoolOptions from_env();
    static const char *name(Arena a);
};

struct PoolStats {
    uint64_t pins = 0;    // arenas pinned
    double pin_s = 0;     // time spent pinning them (background thread)
    uint64_t waits = 0;   // get() calls that had to wait for an arena
    double wait_s = 0;    // time those callers waited
};

class PinnedPool {
public:
    // device: made current while allocating an arena (NUMA placement). standing: free units kept ready for a burst of
    // saves: topped up one arena at a time, only after the pool has had no get() for a while (a refill that ran during
    // a long prefill would pin concurrently with its compute, which is what it is there to avoid).
    PinnedPool(size_t unit_bytes, int units_per_arena, int device, size_t standing, PoolOptions opt = {});
    ~PinnedPool();
    PinnedPool(const PinnedPool &) = delete;
    PinnedPool &operator=(const PinnedPool &) = delete;

    // Makes n free units available soon, in the background (non-blocking).
    void reserve(size_t n);
    uint8_t *get();
    void put(uint8_t *p);
    size_t unit_bytes() const { return unit_; }
    size_t allocated_bytes() const;  // arenas held
    bool standing_ready() const;     // the standing reserve is in place (or there is none)
    PoolStats stats() const;

private:
    struct Arena {
        uint8_t *base = nullptr;
        std::vector<int> free;  // unit indices
    };
    std::unique_ptr<Arena> new_arena();  // without the lock
    void free_arena(Arena &a) const;           // without the lock
    void add_arena(std::unique_ptr<Arena> a);  // with the lock held
    void want(size_t n);                       // with the lock held
    void prefetch_loop();

    size_t unit_;
    int per_arena_, device_;
    size_t standing_;
    // (starts as if the pool had been quiet for a while: the first fill of the standing reserve need not wait)
    std::chrono::steady_clock::time_point last_get_ = std::chrono::steady_clock::now() - std::chrono::seconds(60);
    PoolOptions opt_;
    mutable std::mutex mu_;
    std::condition_variable cv_, ready_cv_;
    std::vector<std::unique_ptr<Arena>> arenas_;
    std::vector<std::unique_ptr<Arena>> retired_;  // empty arenas for the thread to free
    size_t free_units_ = 0;
    int empty_ = 0;                            // arenas with every unit free
    size_t target_ = 0;  // free units the prefetch thread works toward
    bool busy_ = false, stop_ = false;
    PoolStats stats_;
    std::thread prefetch_;
};

}  // namespace qw
