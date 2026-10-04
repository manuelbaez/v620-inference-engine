// Recurrent-state snapshots of a slot in the VRAM pool.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

namespace qw {

void raw_tail_copy(const Rank::Slot &sl, int layers, int64_t n, float *tail, bool to_tail, hipMemcpyKind kind,
                   hipStream_t s) {
    const int64_t first = n - gpu::RAW_TAIL;  // the tail's row 0
    for (int i = 0; i < layers; ++i) {
        float *ring = sl.raw_k[size_t(i)];
        float *t = tail + size_t(i) * gpu::RAW_TAIL * IDX_DIM;
        for (int64_t p = std::max<int64_t>(0, first); p < n;) {  // at most two pieces: the ring wraps once
            const int64_t at = p % sl.raw_ring, run = std::min<int64_t>(n - p, sl.raw_ring - at);
            float *rp = ring + at * IDX_DIM, *tp = t + (p - first) * IDX_DIM;
            CK(hipMemcpyAsync(to_tail ? tp : rp, to_tail ? rp : tp, size_t(run) * IDX_DIM * 4, kind, s));
            p += run;
        }
    }
}

void Engine::snapshot_save(int snap, int slot) {
    QW_CHECK(snap >= 0 && snap < SNAPSHOTS, "bad snapshot index");
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        auto &sn = rk.snaps[size_t(snap)];
        auto &sl = rk.slots[size_t(slot)];
        CK(hipMemcpyAsync(sn.S, sl.S, SLOT_S * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sn.ring, sl.ring, SLOT_RING * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sn.ple, sl.ple, SLOT_PLE * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sn.pend, sl.pend, XW * 4, hipMemcpyDeviceToDevice, rk.s));
        raw_tail_copy(sl, N_QSA + int(mtp_), slots_[size_t(slot)].len, sn.raw_tail, true, hipMemcpyDeviceToDevice, rk.s);
    }
    for (auto &rkp : ranks_) CK(hipStreamSynchronize(rkp->s));
}

void Engine::snapshot_restore(int snap, int slot, int64_t n, const std::vector<int32_t> &tail) {
    QW_CHECK(snap >= 0 && snap < SNAPSHOTS, "bad snapshot index");
    spill_cache_bump(slot);
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        auto &sn = rk.snaps[size_t(snap)];
        auto &sl = rk.slots[size_t(slot)];
        CK(hipMemcpyAsync(sl.S, sn.S, SLOT_S * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.ring, sn.ring, SLOT_RING * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.ple, sn.ple, SLOT_PLE * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.pend, sn.pend, XW * 4, hipMemcpyDeviceToDevice, rk.s));
        raw_tail_copy(sl, N_QSA + int(mtp_), n, sn.raw_tail, false, hipMemcpyDeviceToDevice, rk.s);
    }
    for (auto &rkp : ranks_) CK(hipStreamSynchronize(rkp->s));
    SlotHost &sh = slots_[size_t(slot)];
    sh.len = n;
    sh.hist.assign(tail.end() - std::min<ptrdiff_t>(ptrdiff_t(tail.size()), 2), tail.end());
    sh.reset = false;
    sh.pend_valid = mtp_;  // snapshots are taken after prefill chunks
    sh.run_first = -1;
}

}  // namespace qw
