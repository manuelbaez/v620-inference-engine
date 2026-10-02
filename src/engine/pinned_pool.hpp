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

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace qw {

struct PoolOptions {
    // How an arena is made. HostMalloc: hipHostMalloc with default flags (today's; on some stacks coherent,
    // fine-grained memory that the GPU reads slowly). NonCoherent: hipHostMalloc with hipHostMallocNonCoherent
    // (cached host memory). Huge: an anonymous mapping advised for transparent huge pages and registered with
    // hipHostRegister: 2 MB pages cost the kernel 4-7x less to pin (mlock of 256 MB: 0.15-0.26 s with 4 KB pages,
    // 0.03-0.05 s with huge pages, on the production host).
    enum class Arena { HostMalloc, NonCoherent, Huge };
    Arena arena = Arena::HostMalloc;
    // QW_POOL_ARENA=malloc|noncoherent|huge
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
    // device: made current while allocating an arena (NUMA placement).
    PinnedPool(size_t unit_bytes, int units_per_arena, int device, size_t low_water, PoolOptions opt = {});
    ~PinnedPool();
    PinnedPool(const PinnedPool &) = delete;
    PinnedPool &operator=(const PinnedPool &) = delete;

    // Makes n free units available soon, in the background (non-blocking).
    void reserve(size_t n);
    uint8_t *get();
    void put(uint8_t *p);
    size_t unit_bytes() const { return unit_; }
    size_t allocated_bytes() const;  // arenas held
    PoolStats stats() const;

private:
    struct Arena {
        uint8_t *base = nullptr;
        std::vector<int> free;  // unit indices
        size_t map_bytes = 0;   // hugepage arenas: the mapping (registered with HIP), else 0 (hipHostMalloc)
    };
    std::unique_ptr<Arena> new_arena();  // without the lock
    bool map_arena(Arena &a, size_t bytes) const;
    void free_arena(Arena &a) const;           // without the lock
    void add_arena(std::unique_ptr<Arena> a);  // with the lock held
    void want(size_t n);                       // with the lock held
    void prefetch_loop();

    size_t unit_;
    int per_arena_, device_;
    size_t low_water_;
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
