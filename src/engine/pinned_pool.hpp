// Fixed-size pinned host buffers, carved from arenas that are allocated on
// demand and freed once empty (hipHostMalloc is slow per call, and the prefix
// cache holds thousands of buffers). Thread-safe.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace qw {

class PinnedPool {
public:
    // device: made current while allocating an arena (NUMA placement).
    PinnedPool(size_t unit_bytes, int units_per_arena, int device);
    ~PinnedPool();
    PinnedPool(const PinnedPool &) = delete;
    PinnedPool &operator=(const PinnedPool &) = delete;

    uint8_t *get();
    void put(uint8_t *p);
    size_t unit_bytes() const { return unit_; }
    size_t allocated_bytes() const;  // arenas held

private:
    struct Arena {
        uint8_t *base = nullptr;
        std::vector<int> free;  // unit indices
    };
    size_t unit_;
    int per_arena_, device_;
    mutable std::mutex mu_;
    std::vector<std::unique_ptr<Arena>> arenas_;
    int empty_ = 0;  // arenas with every unit free (one is kept as a spare)
};

}  // namespace qw
