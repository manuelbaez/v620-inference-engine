// Host-RAM tier: a slot's KV and recurrent state copied to pinned host
// memory, so a conversation evicted from its slot comes back without a
// prefill (PCIe copy instead of recompute: ~0.1 s vs ~16 s for 32k tokens).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

namespace qw {

namespace {

// Bytes of one rank's state at n tokens: per QSA layer K, V (fp16 [n][256]),
// raw indexer keys (fp32 [n][128]) and compressed keys (fp16 [n/4][128]);
// then the recurrent state and the MTP input store.
struct Layout {
    size_t k, v, raw, ck, per_layer, state;
    Layout(int64_t n) {
        k = size_t(n) * HEAD_DIM * 2;
        v = k;
        raw = size_t(n) * IDX_DIM * 4;
        ck = size_t(n / IDX_RATIO) * IDX_DIM * 2;
        per_layer = k + v + raw + ck;
        state = (SLOT_S + SLOT_RING + SLOT_PLE + XW) * 4;
    }
};

}  // namespace

struct Engine::HostState {
    int64_t n = 0;
    int layers = 0;
    std::vector<uint8_t *> buf;  // per rank, pinned
    size_t bytes_per_rank = 0;
    ~HostState() {
        for (uint8_t *p : buf) (void)hipHostFree(p);
    }
};

size_t Engine::host_state_bytes(const HostState &hs) {
    return hs.bytes_per_rank * hs.buf.size();
}

std::shared_ptr<Engine::HostState> Engine::export_state(int slot, int snap, int64_t n) {
    QW_CHECK(slot >= 0 && slot < num_slots() && snap >= 0 && snap < SNAPSHOTS, "export_state: bad slot / snapshot");
    QW_CHECK(n > 0 && n <= slots_[size_t(slot)].len, "export_state: the slot does not hold n tokens");
    auto hs = std::make_shared<HostState>();
    hs->n = n;
    hs->layers = N_QSA + int(mtp_);
    const Layout lay(n);
    hs->bytes_per_rank = size_t(hs->layers) * lay.per_layer + lay.state;
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        hs->buf.push_back(pinned<uint8_t>(hs->bytes_per_rank));
        uint8_t *p = hs->buf.back();
        const auto &sl = rk.slots[size_t(slot)];
        for (int i = 0; i < hs->layers; ++i) {
            CK(hipMemcpyAsync(p, sl.K[size_t(i)], lay.k, hipMemcpyDeviceToHost, rk.s));
            CK(hipMemcpyAsync(p + lay.k, sl.V[size_t(i)], lay.v, hipMemcpyDeviceToHost, rk.s));
            CK(hipMemcpyAsync(p + lay.k + lay.v, sl.raw_k[size_t(i)], lay.raw, hipMemcpyDeviceToHost, rk.s));
            if (lay.ck)
                CK(hipMemcpyAsync(p + lay.k + lay.v + lay.raw, sl.ck[size_t(i)], lay.ck, hipMemcpyDeviceToHost, rk.s));
            p += lay.per_layer;
        }
        const auto &sn = rk.snaps[size_t(snap)];
        CK(hipMemcpyAsync(p, sn.S, SLOT_S * 4, hipMemcpyDeviceToHost, rk.s));
        p += SLOT_S * 4;
        CK(hipMemcpyAsync(p, sn.ring, SLOT_RING * 4, hipMemcpyDeviceToHost, rk.s));
        p += SLOT_RING * 4;
        CK(hipMemcpyAsync(p, sn.ple, SLOT_PLE * 4, hipMemcpyDeviceToHost, rk.s));
        p += SLOT_PLE * 4;
        CK(hipMemcpyAsync(p, sn.pend, XW * 4, hipMemcpyDeviceToHost, rk.s));
    }
    for (auto &rkp : ranks_) CK(hipStreamSynchronize(rkp->s));
    return hs;
}

void Engine::import_state(const HostState &hs, int slot, const std::vector<int32_t> &tail) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "import_state: bad slot");
    QW_CHECK(hs.n <= slot_capacity(slot), "import_state: the slot is too small");
    QW_CHECK(hs.layers == N_QSA + int(mtp_) && hs.buf.size() == ranks_.size(), "import_state: layout mismatch");
    const Layout lay(hs.n);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        Rank &rk = *ranks_[r];
        CK(hipSetDevice(rk.dev));
        const uint8_t *p = hs.buf[r];
        auto &sl = rk.slots[size_t(slot)];
        for (int i = 0; i < hs.layers; ++i) {
            CK(hipMemcpyAsync(sl.K[size_t(i)], p, lay.k, hipMemcpyHostToDevice, rk.s));
            CK(hipMemcpyAsync(sl.V[size_t(i)], p + lay.k, lay.v, hipMemcpyHostToDevice, rk.s));
            CK(hipMemcpyAsync(sl.raw_k[size_t(i)], p + lay.k + lay.v, lay.raw, hipMemcpyHostToDevice, rk.s));
            if (lay.ck)
                CK(hipMemcpyAsync(sl.ck[size_t(i)], p + lay.k + lay.v + lay.raw, lay.ck, hipMemcpyHostToDevice, rk.s));
            p += lay.per_layer;
        }
        CK(hipMemcpyAsync(sl.S, p, SLOT_S * 4, hipMemcpyHostToDevice, rk.s));
        p += SLOT_S * 4;
        CK(hipMemcpyAsync(sl.ring, p, SLOT_RING * 4, hipMemcpyHostToDevice, rk.s));
        p += SLOT_RING * 4;
        CK(hipMemcpyAsync(sl.ple, p, SLOT_PLE * 4, hipMemcpyHostToDevice, rk.s));
        p += SLOT_PLE * 4;
        CK(hipMemcpyAsync(sl.pend, p, XW * 4, hipMemcpyHostToDevice, rk.s));
    }
    for (auto &rkp : ranks_) CK(hipStreamSynchronize(rkp->s));
    SlotHost &sh = slots_[size_t(slot)];
    sh.len = hs.n;
    sh.hist.assign(tail.end() - std::min<ptrdiff_t>(ptrdiff_t(tail.size()), 32), tail.end());
    sh.reset = false;
    sh.pend_valid = mtp_;
    sh.run_first = -1;
}

}  // namespace qw
