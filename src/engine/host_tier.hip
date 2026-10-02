// Host copies of slot state for the block store (src/session/block_store):
// the KV of a position range and a snapshot's recurrent state, per rank,
// between VRAM and pinned host memory (a PCIe copy instead of recompute:
// ~0.1 s vs ~16 s for 32k tokens).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

namespace qw {

namespace {

// Byte layout of n positions of one QSA layer: K, V, raw indexer keys, then
// the compressed keys of the n / IDX_RATIO groups.
struct KvLayout {
    size_t k, raw, ck, per_layer;
    explicit KvLayout(int64_t n) {
        k = size_t(n) * HEAD_DIM * 2;
        raw = size_t(n) * IDX_DIM * 4;
        ck = size_t(n / IDX_RATIO) * IDX_DIM * 2;
        per_layer = 2 * k + raw + ck;
    }
};

}  // namespace

size_t Engine::kv_rank_bytes(int64_t n) const {
    return size_t(N_QSA + int(mtp_)) * KvLayout(n).per_layer;
}

size_t Engine::recurrent_rank_bytes() {
    return (SLOT_S + SLOT_RING + SLOT_PLE + XW) * 4;
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
            CK(hipMemcpyAsync(p, sl.K[size_t(i)] + p0 * HEAD_DIM, lay.k, d2h, rk.s));
            CK(hipMemcpyAsync(p + lay.k, sl.V[size_t(i)] + p0 * HEAD_DIM, lay.k, d2h, rk.s));
            CK(hipMemcpyAsync(p + 2 * lay.k, sl.raw_k[size_t(i)] + p0 * IDX_DIM, lay.raw, d2h, rk.s));
            if (lay.ck)
                CK(hipMemcpyAsync(p + 2 * lay.k + lay.raw, sl.ck[size_t(i)] + p0 / IDX_RATIO * IDX_DIM, lay.ck, d2h,
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
            CK(hipMemcpyAsync(sl.K[size_t(i)] + p0 * HEAD_DIM, p, lay.k, h2d, rk.s));
            CK(hipMemcpyAsync(sl.V[size_t(i)] + p0 * HEAD_DIM, p + lay.k, lay.k, h2d, rk.s));
            CK(hipMemcpyAsync(sl.raw_k[size_t(i)] + p0 * IDX_DIM, p + 2 * lay.k, lay.raw, h2d, rk.s));
            if (lay.ck)
                CK(hipMemcpyAsync(sl.ck[size_t(i)] + p0 / IDX_RATIO * IDX_DIM, p + 2 * lay.k + lay.raw, lay.ck, h2d,
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
    }
    host_copies_wait();
    SlotHost &sh = slots_[size_t(slot)];
    sh.len = n;
    sh.hist.assign(tail.end() - std::min<ptrdiff_t>(ptrdiff_t(tail.size()), 32), tail.end());
    sh.reset = false;
    sh.pend_valid = mtp_;
    sh.run_first = -1;
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
    const uint64_t fields[] = {4,  uint64_t(N_QSA + int(mtp_)), HEAD_DIM, IDX_DIM, IDX_RATIO, SLOT_S, SLOT_RING,
                               SLOT_PLE, XW, uint64_t(RANKS), uint64_t(VOCAB)};
    uint64_t h = 1469598103934665603ull;
    for (uint64_t f : fields)
        for (int b = 0; b < 8; ++b) {
            h ^= (f >> (8 * b)) & 0xff;
            h *= 1099511628211ull;
        }
    return h;
}

}  // namespace qw
