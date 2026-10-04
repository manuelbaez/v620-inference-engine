// KV spill: the part of a slot's KV beyond its VRAM tokens lives in pinned host memory that the kernels read and
// write directly (mapped into every device's address space; docs/DESIGN.md "KV spill").
//
// The ranks of a KV group (cfg::kv_primary: two ranks hold the same KV head) compute byte-identical K and V, so the
// host holds one copy per group, not per rank; both ranks write the same bytes and each reads what it wrote. Each group's layers alternate between the NUMA nodes nearest to its ranks' GPUs, so each rank reads
// about half its data from its own node (a preference, not a requirement: with a node short of memory the pages go
// elsewhere). The pool is allocated once at start; pinning on demand stalled the process for seconds.
#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/shard.hpp"

namespace qw {

class SpillPool {
public:
    // One QSA layer's spilled KV of one slot and KV group: tokens [0, spill_tokens) of the host part (position vt + i).
    struct Buf {
        uint16_t *K = nullptr, *V = nullptr;  // fp16 [spill][256]
    };
    // spill_tokens[slot]: tokens beyond the slot's VRAM (0: none). layers: QSA layers per slot (including MTP).
    // devices: the HIP device of each rank (for the NUMA node nearest to it).
    SpillPool(const std::vector<int> &spill_tokens, int layers, const std::array<int, cfg::RANKS> &devices);
    ~SpillPool();
    SpillPool(const SpillPool &) = delete;
    SpillPool &operator=(const SpillPool &) = delete;

    // The buffers of a slot's layer for the KV group of `rank`.
    const Buf &buf(int slot, int rank, int layer) const;
    static size_t token_bytes() { return 2 * 512; }  // K, V per token and layer
    size_t bytes() const { return bytes_; }

private:
    int layers_;
    size_t bytes_ = 0;
    std::vector<std::vector<Buf>> bufs_;  // [slot][group * layers + layer]
    std::vector<void *> allocs_;
};

// NUMA node of a HIP device's PCI function (sysfs), or -1 when unknown.
int device_numa_node(int device);

}  // namespace qw
