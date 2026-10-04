// CacheBlend experiment: reusing a chunk's cached state after a different
// prefix (docs/DESIGN.md, "CacheBlend-style non-prefix reuse").
#include <hip/hip_runtime.h>

#include <algorithm>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"
#include "kernels/blend.hpp"
#include "kernels/qsa.hpp"

namespace qw {

void Engine::blend_transfer_begin() {
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        if (!rk.transfer) {
            rk.transfer = zeros<float>(SLOT_S);
            rk.blend_tmp = zeros<float>(SLOT_S);
        }
        gpu::gdn_transfer_init(rk.transfer, N_GDN, rk.s);
    }
    host_copies_wait();
    blend_transfer_ = true;
}

void Engine::blend_transfer_end() {
    blend_transfer_ = false;
}

void Engine::blend_splice(int dst, int src, int64_t src_p0, int64_t dst_p0, int64_t n) {
    QW_CHECK(n > 0 && src_p0 + n <= slots_[size_t(src)].len && dst_p0 + n <= slot_capacity(dst), "blend_splice: range");
    QW_CHECK(opt_.slot_spill[size_t(src)] == 0 && opt_.slot_spill[size_t(dst)] == 0,
             "blend_splice: not supported on slots with KV spill");
    QW_CHECK(false, "blend_splice: the slots keep only a ring of raw indexer keys now, and re-deriving a moved chunk's "
                    "compressed keys needs all of them (the CacheBlend experiment was rejected: docs/DESIGN.md)");
    const hipMemcpyKind d2d = hipMemcpyDeviceToDevice;
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        auto &s = rk.slots[size_t(src)], &d = rk.slots[size_t(dst)];
        for (int i = 0; i < N_QSA + int(mtp_); ++i) {
            const float *ikn = i == N_QSA ? rk.mtp.ikn : nullptr;
            for (int l = 0; !ikn && l < N_LAYERS; ++l)
                if (is_qsa(l) && qsa_index(l) == i) ikn = rk.L[size_t(l)].ikn;
            CK(hipMemcpyAsync(d.K[size_t(i)] + dst_p0 * HEAD_DIM, s.K[size_t(i)] + src_p0 * HEAD_DIM,
                              size_t(n) * HEAD_DIM * 2, d2d, rk.s));
            CK(hipMemcpyAsync(d.V[size_t(i)] + dst_p0 * HEAD_DIM, s.V[size_t(i)] + src_p0 * HEAD_DIM,
                              size_t(n) * HEAD_DIM * 2, d2d, rk.s));
            CK(hipMemcpyAsync(d.raw_k[size_t(i)] + dst_p0 * IDX_DIM, s.raw_k[size_t(i)] + src_p0 * IDX_DIM,
                              size_t(n) * IDX_DIM * 4, d2d, rk.s));
            gpu::rope_shift(d.K[size_t(i)] + dst_p0 * HEAD_DIM, n, dst_p0 - src_p0, rk.s);
            gpu::qsa_compress_T(d.kv(size_t(i)), ikn, dst_p0, int(n), d.ck[size_t(i)], rk.s);
        }
    }
    host_copies_wait();
}

void Engine::blend_compose(int dst, int snap_in, int snap_out, int64_t old_end, int64_t new_len,
                           const std::vector<int32_t> &tail, bool with_transfer) {
    const int64_t delta = new_len - old_end;
    auto cell = [](int64_t c, int64_t d, int ring) { return int(((c + d) % ring + ring) % ring); };
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        auto &sl = rk.slots[size_t(dst)];
        const auto &in = rk.snaps[size_t(snap_in)], &out = rk.snaps[size_t(snap_out)];
        if (with_transfer)
            gpu::gdn_compose(sl.S, rk.transfer, in.S, out.S, rk.blend_tmp, N_GDN, rk.s);
        else
            CK(hipMemcpyAsync(sl.S, out.S, SLOT_S * 4, hipMemcpyDeviceToDevice, rk.s));
        // rings are indexed by position: move each cell by the position shift
        for (int l = 0; l < N_GDN; ++l)
            for (int c = 0; c < GDN_RING; ++c)
                CK(hipMemcpyAsync(sl.ring + (size_t(l) * GDN_RING + size_t(cell(c, delta, GDN_RING))) * GDN_QKV_L,
                                  out.ring + (size_t(l) * GDN_RING + size_t(c)) * GDN_QKV_L, GDN_QKV_L * 4,
                                  hipMemcpyDeviceToDevice, rk.s));
        for (int c = 0; c < PLE_RING; ++c)
            CK(hipMemcpyAsync(sl.ple + size_t(cell(c, delta, PLE_RING)) * XW, out.ple + size_t(c) * XW, XW * 4,
                              hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.pend, out.pend, XW * 4, hipMemcpyDeviceToDevice, rk.s));
    }
    host_copies_wait();
    SlotHost &sh = slots_[size_t(dst)];
    sh.len = new_len;
    sh.hist.assign(tail.end() - std::min<ptrdiff_t>(ptrdiff_t(tail.size()), 32), tail.end());
    sh.reset = false;
    sh.pend_valid = mtp_;
    sh.run_first = -1;
}

}  // namespace qw
