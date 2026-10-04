// Host copies of slot state for the block store (src/session/block_store):
// the KV of a position range and a snapshot's recurrent state, per rank,
// between VRAM and pinned host memory (a PCIe copy instead of recompute:
// ~0.1 s vs ~16 s for 32k tokens).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"
#include "kernels/qsa.hpp"

namespace qw {

namespace {

// Byte layout of n positions of one QSA layer: K, V, then the compressed keys of the n / IDX_RATIO groups. (The raw
// indexer keys a group still needs travel with the snapshot that ends inside it: Rank::Snap::raw_tail.)
struct KvLayout {
    size_t k, ck, per_layer;
    explicit KvLayout(int64_t n) {
        k = size_t(n) * HEAD_DIM * 2;
        ck = size_t(n / IDX_RATIO) * IDX_DIM * 2;
        per_layer = 2 * k + ck;
    }
};

// Positions [p0, p0 + n) of a per-position array of `row` bytes per position that is split at vt (VRAM below, mapped
// host memory from vt on: the KV spill), to and from a contiguous host buffer. The VRAM part is a stream copy; the
// spilled part waits for the rank's stream (its kernels may still be writing it) and is a plain memcpy.
void copy_split_out(uint8_t *dst, const uint8_t *dev, const uint8_t *host, int64_t vt, int64_t p0, int64_t n, size_t row,
                    hipStream_t s) {
    const int64_t nd = std::max<int64_t>(0, std::min<int64_t>(n, vt - p0));  // positions in VRAM
    if (nd) CK(hipMemcpyAsync(dst, dev + size_t(p0) * row, size_t(nd) * row, hipMemcpyDeviceToHost, s));
    if (n > nd) {
        CK(hipStreamSynchronize(s));
        std::memcpy(dst + size_t(nd) * row, host + size_t(p0 + nd - vt) * row, size_t(n - nd) * row);
    }
}
void copy_split_in(uint8_t *dev, uint8_t *host, const uint8_t *src, int64_t vt, int64_t p0, int64_t n, size_t row,
                   hipStream_t s) {
    const int64_t nd = std::max<int64_t>(0, std::min<int64_t>(n, vt - p0));
    if (nd) CK(hipMemcpyAsync(dev + size_t(p0) * row, src, size_t(nd) * row, hipMemcpyHostToDevice, s));
    if (n > nd) {
        CK(hipStreamSynchronize(s));
        std::memcpy(host + size_t(p0 + nd - vt) * row, src + size_t(nd) * row, size_t(n - nd) * row);
    }
}

}  // namespace

size_t Engine::kv_rank_bytes(int64_t n) const {
    return size_t(N_QSA + int(mtp_)) * KvLayout(n).per_layer;
}

size_t Engine::recurrent_rank_bytes() {
    return (SLOT_S + SLOT_RING + SLOT_PLE + XW + RAW_TAIL_FLOATS) * 4;
}

void Engine::export_kv(int slot, int64_t p0, int64_t n, const RankBufs &dst) {
    QW_CHECK(slot >= 0 && slot < num_slots() && p0 % IDX_RATIO == 0 && n > 0 && p0 + n <= slots_[size_t(slot)].len,
             "export_kv: bad range");
    const KvLayout lay(n);
    const hipMemcpyKind d2h = hipMemcpyDeviceToHost;
    for (size_t r = 0; r < ranks_.size(); ++r) {
        // a rank without a buffer is skipped, and so is a replica whose buffer is its group's first rank's: that
        // rank writes it (the group's KV is identical)
        const size_t first = size_t(cfg::kv_primary(int(r)));
        if (!dst[r] || (first != r && dst[r] == dst[first])) continue;
        Rank &rk = *ranks_[r];
        CK(hipSetDevice(rk.dev));
        uint8_t *p = dst[r];
        const auto &sl = rk.slots[size_t(slot)];
        for (int i = 0; i < N_QSA + int(mtp_); ++i) {
            const size_t qi = size_t(i);
            const bool sp = !sl.Kh.empty();
            const int64_t vt = sp ? sl.vt : INT64_MAX / 2;
            copy_split_out(p, reinterpret_cast<const uint8_t *>(sl.K[qi]), sp ? reinterpret_cast<const uint8_t *>(sl.Kh[qi]) : nullptr,
                           vt, p0, n, HEAD_DIM * 2, rk.s);
            copy_split_out(p + lay.k, reinterpret_cast<const uint8_t *>(sl.V[qi]), sp ? reinterpret_cast<const uint8_t *>(sl.Vh[qi]) : nullptr,
                           vt, p0, n, HEAD_DIM * 2, rk.s);
            if (lay.ck)
                CK(hipMemcpyAsync(p + 2 * lay.k, sl.ck[size_t(i)] + p0 / IDX_RATIO * IDX_DIM, lay.ck, d2h,
                                  rk.s));
            p += lay.per_layer;
        }
    }
}

void Engine::import_kv(int slot, int64_t p0, int64_t n, const RankBufs &src) {
    QW_CHECK(slot >= 0 && slot < num_slots() && p0 % IDX_RATIO == 0 && n > 0 && p0 + n <= slot_capacity(slot),
             "import_kv: bad range");
    const KvLayout lay(n);
    const hipMemcpyKind h2d = hipMemcpyHostToDevice;
    for (size_t r = 0; r < ranks_.size(); ++r) {
        Rank &rk = *ranks_[r];
        CK(hipSetDevice(rk.dev));
        const uint8_t *p = src[r];
        auto &sl = rk.slots[size_t(slot)];
        for (int i = 0; i < N_QSA + int(mtp_); ++i) {
            const size_t qi = size_t(i);
            const bool sp = !sl.Kh.empty();
            const int64_t vt = sp ? sl.vt : INT64_MAX / 2;
            copy_split_in(reinterpret_cast<uint8_t *>(sl.K[qi]), sp ? reinterpret_cast<uint8_t *>(sl.Kh[qi]) : nullptr, p, vt, p0, n,
                          HEAD_DIM * 2, rk.s);
            copy_split_in(reinterpret_cast<uint8_t *>(sl.V[qi]), sp ? reinterpret_cast<uint8_t *>(sl.Vh[qi]) : nullptr, p + lay.k, vt,
                          p0, n, HEAD_DIM * 2, rk.s);
            if (lay.ck)
                CK(hipMemcpyAsync(sl.ck[size_t(i)] + p0 / IDX_RATIO * IDX_DIM, p + 2 * lay.k, lay.ck, h2d,
                                  rk.s));
            p += lay.per_layer;
        }
    }
}

void Engine::export_recurrent(int snap, const RankBufs &dst) {
    QW_CHECK(snap >= 0 && snap < SNAPSHOTS, "export_recurrent: bad snapshot");
    for (size_t r = 0; r < ranks_.size(); ++r) {
        Rank &rk = *ranks_[r];
        CK(hipSetDevice(rk.dev));
        uint8_t *p = dst[r];
        const auto &sn = rk.snaps[size_t(snap)];
        CK(hipMemcpyAsync(p, sn.S, SLOT_S * 4, hipMemcpyDeviceToHost, rk.s));
        p += SLOT_S * 4;
        CK(hipMemcpyAsync(p, sn.ring, SLOT_RING * 4, hipMemcpyDeviceToHost, rk.s));
        p += SLOT_RING * 4;
        CK(hipMemcpyAsync(p, sn.ple, SLOT_PLE * 4, hipMemcpyDeviceToHost, rk.s));
        p += SLOT_PLE * 4;
        CK(hipMemcpyAsync(p, sn.pend, XW * 4, hipMemcpyDeviceToHost, rk.s));
        p += XW * 4;
        CK(hipMemcpyAsync(p, sn.raw_tail, RAW_TAIL_FLOATS * 4, hipMemcpyDeviceToHost, rk.s));
    }
}

void Engine::import_recurrent(int slot, const RankBufs &src, int64_t n, const std::vector<int32_t> &tail) {
    QW_CHECK(slot >= 0 && slot < num_slots() && n > 0 && n <= slot_capacity(slot), "import_recurrent: bad slot");
    for (size_t r = 0; r < ranks_.size(); ++r) {
        Rank &rk = *ranks_[r];
        CK(hipSetDevice(rk.dev));
        const uint8_t *p = src[r];
        auto &sl = rk.slots[size_t(slot)];
        CK(hipMemcpyAsync(sl.S, p, SLOT_S * 4, hipMemcpyHostToDevice, rk.s));
        p += SLOT_S * 4;
        CK(hipMemcpyAsync(sl.ring, p, SLOT_RING * 4, hipMemcpyHostToDevice, rk.s));
        p += SLOT_RING * 4;
        CK(hipMemcpyAsync(sl.ple, p, SLOT_PLE * 4, hipMemcpyHostToDevice, rk.s));
        p += SLOT_PLE * 4;
        CK(hipMemcpyAsync(sl.pend, p, XW * 4, hipMemcpyHostToDevice, rk.s));
        p += XW * 4;
        raw_tail_copy(sl, N_QSA + int(mtp_), n, reinterpret_cast<float *>(const_cast<uint8_t *>(p)), false,
                      hipMemcpyHostToDevice, rk.s);
    }
    host_copies_wait();
    SlotHost &sh = slots_[size_t(slot)];
    sh.len = n;
    sh.hist.assign(tail.end() - std::min<ptrdiff_t>(ptrdiff_t(tail.size()), 32), tail.end());
    sh.reset = false;
    sh.pend_valid = mtp_;
    sh.run_first = -1;
}

std::vector<unsigned long long> Engine::spill_locality() {
    Rank &rk = *ranks_[0];
    std::vector<unsigned long long> v(size_t(gpu::QSA_LAYERS_MAX) * gpu::LOC_STATS, 0);
    if (!rk.loc_stats) return v;
    CK(hipSetDevice(rk.dev));
    CK(hipStreamSynchronize(rk.s));
    CK(hipMemcpy(v.data(), rk.loc_stats, v.size() * 8, hipMemcpyDeviceToHost));
    CK(hipMemset(rk.loc_stats, 0, v.size() * 8));
    return v;
}

void Engine::host_copies_wait() {
    for (auto &rkp : ranks_) {
        CK(hipSetDevice(rkp->dev));
        CK(hipStreamSynchronize(rkp->s));
    }
}

int Engine::rank_device(int r) const {
    return ranks_[size_t(r)]->dev;
}

uint64_t Engine::state_layout_id() const {
    // FNV-1a over everything that shapes the per-token and per-sequence state
    // (5: blocks without raw indexer keys, snapshots with their last RAW_TAIL positions')
    const uint64_t fields[] = {5,  uint64_t(N_QSA + int(mtp_)), HEAD_DIM, IDX_DIM, IDX_RATIO, SLOT_S, SLOT_RING,
                               SLOT_PLE, XW, uint64_t(RANKS), uint64_t(VOCAB), uint64_t(gpu::RAW_TAIL)};
    uint64_t h = 1469598103934665603ull;
    for (uint64_t f : fields)
        for (int b = 0; b < 8; ++b) {
            h ^= (f >> (8 * b)) & 0xff;
            h *= 1099511628211ull;
        }
    return h;
}

}  // namespace qw
