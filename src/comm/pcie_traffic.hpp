// PCIe traffic per card, counted by the engine (docs/DESIGN.md "PCIe traffic per card"). The V620s expose no PCIe
// byte counters (no pcie_bw on Navi 21, gpu_metrics has only the link speed and width), so the engine counts what it
// moves: host <-> GPU copies (through copy_async / copy below), the collectives' pushes to and from the peers, and the
// spilled KV's traffic with host memory by kind: decode's reads (SpillCache misses and fills, added by
// Engine::pcie_bytes), the KV rows decode and prefill write there (SpillWrite), the prefill staging's copies in and
// back (SpillStageIn / SpillStageOut, not in HostToGpu / GpuToHost), and the rows the attention of a prefill segment
// that is not staged reads from host memory (SpillPrefillRead, counted on the device from its lists, added by
// Engine::pcie_bytes). Not counted: payload-free control words and spilled rows read without the spill cache.
//
// Counting is per rank. Copies find their rank from the current HIP device. Work recorded into a HIP graph counts into
// the tally of the capture (Capture below), and the graph's launcher adds that tally on every replay.
#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "core/shard.hpp"

namespace qw::pcie {

enum Kind { HostToGpu, GpuToHost, PeerIn, PeerOut, SpillRead, SpillWrite, SpillStageIn, SpillStageOut, SpillPrefillRead,
            KINDS };
using Bytes = std::array<uint64_t, KINDS>;

// The HIP device of each rank, for the copies (before any copy is counted).
void set_devices(const std::array<int, cfg::RANKS> &devices);

// Adds bytes of `kind` to rank r, or to the tally of a capture running on this thread.
void add(int r, Kind kind, uint64_t bytes);
// Adds a graph's tally to rank r (on each launch of the graph).
void add(int r, const Bytes &tally);

// Since the start, per rank.
std::array<Bytes, cfg::RANKS> totals();

// While alive, add() on this thread counts into `tally` instead (a graph capture: its work runs on each replay).
class Capture {
public:
    explicit Capture(Bytes &tally);
    ~Capture();
    Capture(const Capture &) = delete;
    Capture &operator=(const Capture &) = delete;

private:
    Bytes *prev_;
};

// hipMemcpy / hipMemcpyAsync that count host <-> GPU bytes for the current device's rank.
hipError_t copy(void *dst, const void *src, size_t bytes, hipMemcpyKind kind);
hipError_t copy_async(void *dst, const void *src, size_t bytes, hipMemcpyKind kind, hipStream_t s);
// ... counted as `as` instead (rank r).
hipError_t copy_async_as(Kind as, int r, void *dst, const void *src, size_t bytes, hipMemcpyKind kind, hipStream_t s);

}  // namespace qw::pcie
