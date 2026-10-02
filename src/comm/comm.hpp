// Device-driven collectives over PCIe P2P for the 4 ranks (docs/DESIGN.md, "P2P").
//
// Every collective is: a push kernel on the sender that stores its payload
// straight into each peer's uncached receive slot and then bumps that slot's
// flag, and a receive kernel on the consumer that polls its own flags and
// reduces or concatenates the 4 slots in fixed rank order (so all ranks get
// bit-identical results). No host synchronization is involved, so the 4 rank
// threads just enqueue work.
//
// Payloads are matrices of `rows` rows (1 for decode, one per token for
// prefill). Large payloads are pushed by many blocks per destination; the last
// block to finish (counted with a device atomic) raises the flag.
//
// Graph-safe sequencing: the sequence number of a collective is
// *base(r) + k, where base lives in device memory and k is a static per-call
// offset. A captured graph replays with fixed k; the host bumps the base
// between replays (set_base). Consecutive collectives must use consecutive
// sequence numbers.
//
// Slots are double-buffered by sequence parity. A rank can only push sequence
// n+1 after its own wait for n finished, which needs every peer's push of n,
// which each peer only enqueues after consuming n-1: so the parity-(n+1) slots
// are free again.
#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <string>

#include "core/shard.hpp"

namespace qw {

using cfg::RANKS;

class Comm {
public:
    // Enables peer access and allocates every rank's receive buffers
    // (2 x RANKS slots of slot_bytes each, per rank). Call once from one
    // thread; devices[r] is the HIP device of rank r.
    Comm(const std::array<int, RANKS> &devices, size_t slot_bytes = 64 * 1024);
    ~Comm();
    Comm(const Comm &) = delete;
    Comm &operator=(const Comm &) = delete;

    int device(int rank) const { return dev_[size_t(rank)]; }
    size_t slot_bytes() const { return slot_; }

    // Kernel variants for the big (prefill) payloads. The 2D push gives bit-identical results to the plain kernels (the
    // sums are taken in the same order) and differs only in how many threads a copy keeps busy. Read from the
    // environment at construction; tests and benchmarks may change it between collectives (every rank must use the
    // same setting for a collective).
    struct Tuning {
        // Strided or short rows of many rows (a prefill's reduce-scatter slices and tiny-row all-reduces) pushed by all
        // threads over a flat (row, column) range: prefill +5.7% at 8k tokens, +7.6% at 16k, decode untouched
        // (docs/DESIGN.md). QW_COMM_PUSH2D=0 turns it off.
        bool push2d = true;
        static Tuning from_env();
    };
    void set_tuning(const Tuning &t) { tune_ = t; }
    const Tuning &tuning() const { return tune_; }

    // Sets rank r's sequence base (enqueued on s, from pinned memory).
    void set_base(int r, uint32_t base, hipStream_t s);
    // Adds delta to rank r's sequence base on the device (enqueued on s): for
    // steps chained without a host sync in between (set_base's staging word is
    // only rewritten after a sync).
    void bump_base(int r, uint32_t delta, hipStream_t s);

    // All of these are called by rank r's thread and enqueue on stream s,
    // with sequence number base + k. Pointers are device pointers on rank r.

    // dst[t][i] = sum over ranks of src[t][i]        (rows x n; src fp32, or fp16
    // with src_f16; the sum and dst are fp32)
    void allreduce(int r, const void *src, float *dst, int n, uint32_t k, hipStream_t s, int rows = 1,
                   bool src_f16 = false);
    // dst[t][i] = sum over ranks q of src_q[t][r*n + i]; src rows are RANKS*n
    // wide. src is fp32, or fp16 with src_f16 (the sum is fp32 either way).
    void reduce_scatter(int r, const void *src, float *dst, int n, uint32_t k, hipStream_t s, int rows = 1,
                        bool src_f16 = false);
    // dst[t][q*bytes ..] = src_q[t][..]              (bytes per rank per row, multiple of 16)
    void allgather(int r, const void *src, void *dst, size_t bytes, uint32_t k, hipStream_t s, int rows = 1);

    // Nonzero after a receive spun too long (a peer died or diverged):
    // 0x80000000 | waiting rank << 28 | missing source rank << 24 | sequence offset k.
    uint32_t error() const { return *err_; }
    static std::string describe_error(uint32_t e);

private:
    enum Op { SUM_F32 = 0, SUM_F16 = 1, CONCAT = 2 };
    void exchange(int r, const void *src, int rows, size_t row_bytes, size_t src_row_stride, size_t src_dest_offset,
                  Op op, void *dst, uint32_t k, hipStream_t s);

    std::array<int, RANKS> dev_{};
    size_t slot_ = 0;
    Tuning tune_;
    // Per rank: recv[parity][src][slot_] and flags[parity][src] (64 B apart),
    // both uncached, owned by that rank's device.
    std::array<uint8_t *, RANKS> recv_{};
    std::array<uint32_t *, RANKS> flags_{};
    std::array<uint32_t *, RANKS> counters_{};   // push completion counters [dest]
    std::array<uint32_t *, RANKS> base_{};       // device, per rank
    std::array<uint32_t *, RANKS> base_host_{};  // pinned staging
    uint32_t *err_ = nullptr;                    // pinned host memory
};

}  // namespace qw
