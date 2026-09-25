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
    }
    for (auto &rkp : ranks_) CK(hipStreamSynchronize(rkp->s));
}

void Engine::snapshot_restore(int snap, int slot, int64_t n, const std::vector<int32_t> &tail) {
    QW_CHECK(snap >= 0 && snap < SNAPSHOTS, "bad snapshot index");
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        auto &sn = rk.snaps[size_t(snap)];
        auto &sl = rk.slots[size_t(slot)];
        CK(hipMemcpyAsync(sl.S, sn.S, SLOT_S * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.ring, sn.ring, SLOT_RING * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.ple, sn.ple, SLOT_PLE * 4, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.pend, sn.pend, XW * 4, hipMemcpyDeviceToDevice, rk.s));
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
