// Chunked prefill of one slot: two micro-batches per chunk on two streams,
// the MTP layer over the chunk, and the host side of prefill().
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"
#include "engine/weight_convert.hpp"
#include "kernels/blend.hpp"
#include "kernels/gdn.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemv.hpp"
#include "kernels/hc.hpp"
#include "kernels/logits.hpp"
#include "kernels/moe.hpp"
#include "kernels/mtp.hpp"
#include "kernels/ple.hpp"
#include "kernels/qsa.hpp"

namespace qw {

static_assert(Engine::MAX_CAPTURES == gpu::MAX_CAPTURES);

// One prefill chunk (pjob_) on rank rk: the rows of one or more segments
// (distinct slots) concatenated. Row-wise work (projections, mixers, MoE,
// collectives) runs once over all rows; the sequence kernels (GDN conv and
// scan, QSA K/V and attention, PLE conv, MTP shift) run per segment on its
// rows. The chunk runs as two micro-batches (A = first half of the rows, B =
// second half) on two streams with separate collective channels, so one
// half's collectives overlap the other half's compute. A segment can
// straddle the halves; B then depends on A through that segment's
// sequential state, per layer: the GDN conv tail and recurrent state, the QSA
// K/V and compressed keys, and the PLE conv tail. The host enqueues layer l
// of A before layer l of B, so every event wait follows its record. Short
// chunks run as a single micro-batch.
void Engine::run_prefill_rank(Rank &rk) {
    static const bool profile = std::getenv("QW_PROFILE") != nullptr;
    static const bool no_split = std::getenv("QW_NO_SPLIT") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    const int r = rk.r;
    if (moe_stats_ && !moe_counts_[size_t(r)]) moe_counts_[size_t(r)] = pinned<int32_t>(size_t(2) * (N_LAYERS + 1) * EXP_L);
    const auto &segs = pjob_.segs;
    const int T = pjob_.T;
    for (const PrefillSeg &sg : segs)
        if (sg.reset) {
            Rank::Slot &sl = rk.slots[size_t(sg.slot)];
            CK(hipMemsetAsync(sl.S, 0, SLOT_S * 4, rk.s));
            CK(hipMemsetAsync(sl.ring, 0, SLOT_RING * 4, rk.s));
            CK(hipMemsetAsync(sl.ple, 0, SLOT_PLE * 4, rk.s));
        }
    const int64_t start0 = segs[0].start;  // (rope3 and the GDN transfer: one segment)
    int32_t *rope3 = nullptr;              // this chunk's (t, h, w) positions on the device
    if (pjob_.rope3) {
        if (!rk.rope3 || rk.rope3_cap < T) {
            if (rk.rope3) CK(hipFree(rk.rope3));
            rk.rope3 = zeros<int32_t>(size_t(opt_.prefill_chunk) * 3);
            rk.rope3_cap = opt_.prefill_chunk;
        }
        CK(hipMemcpyAsync(rk.rope3, pjob_.rope3, size_t(T) * 12, hipMemcpyHostToDevice, rk.s));
        rope3 = rk.rope3;
    }
    const bool split = T >= 256;  // two micro-batches; QW_NO_SPLIT runs them serially on one stream
    const int TA = split ? T / 2 : T;
    QW_CHECK(TA <= rk.pb[0].C && T - TA <= rk.pb[1].C, "prefill chunk larger than its buffers");

    // A segment's rows inside one micro-batch.
    struct Sub {
        int seg;
        Rank::Slot *sl;
        int64_t start;  // slot position of its first row
        int T;
        int row;    // first row in the micro-batch
        bool cont;  // continues the segment's rows of the other micro-batch
        bool last;  // holds the segment's last row
    };
    struct MB {
        PrefillBuf *b;
        hipStream_t s;
        rocblas_handle blas;
        Comm *comm;
        uint32_t *seq_base;
        uint32_t k;
        int T;
        int row0;  // offset of this micro-batch in the chunk
        const float *pend;
        std::vector<Sub> subs;
    };
    std::vector<MB> mbs;
    mbs.push_back({&rk.pb[0], rk.s, rk.blas, comm_.get(), &rk.seq_base, 0, TA, 0, nullptr, {}});
    if (split)
        mbs.push_back({&rk.pb[1], no_split ? rk.s : rk.s2, no_split ? rk.blas : rk.blas2, comm2_.get(), &rk.seq_base2,
                       0, T - TA, TA, nullptr, {}});
    for (MB &m : mbs)
        for (int i = 0; i < int(segs.size()); ++i) {
            const PrefillSeg &sg = segs[size_t(i)];
            const int a = std::max(m.row0, sg.row0), e = std::min(m.row0 + m.T, sg.row0 + sg.T);
            if (a < e)
                m.subs.push_back({i, &rk.slots[size_t(sg.slot)], sg.start + (a - sg.row0), e - a, a - m.row0,
                                  a > sg.row0, e == sg.row0 + sg.T});
        }
    // KV spill staging (EngineOptions::spill_stage): which segments read their spilled part from the staging buffer,
    // at which offset in it. A segment is staged when its chunk reaches past its slot's VRAM part and its region (the
    // spilled positions up to the chunk's end) fits after the segments staged before it; the others read the host.
    std::vector<int64_t> stage_off(segs.size(), -1);
    {
        size_t used = 0;
        for (size_t i = 0; rk.stage_tokens && i < segs.size(); ++i) {
            const Rank::Slot &sl = rk.slots[size_t(segs[i].slot)];
            const int64_t end = segs[i].start + segs[i].T;
            if (sl.Kh.empty() || end <= sl.vt) continue;
            const size_t need = size_t(end - sl.vt);
            if (used + need > rk.stage_tokens) continue;
            stage_off[i] = int64_t(used);
            used += need;
        }
    }
    const bool staging = std::any_of(stage_off.begin(), stage_off.end(), [](int64_t o) { return o >= 0; });
    // The rows of layer qi each staged segment already holds in host memory go into the buffer (rows [vt, first) with
    // `first` the first position the chunk writes in that layer), after the previous layer's write-back freed it.
    // first_of(i): that position for segment i (the MTP layer's rows start one early when the store holds a hidden).
    auto stage_in = [&](size_t qi, const std::function<int64_t(size_t)> &first_of) {
        if (!staging) return;
        CK(hipStreamWaitEvent(rk.s3, rk.stage_free, 0));
        for (size_t i = 0; i < segs.size(); ++i) {
            if (stage_off[i] < 0) continue;
            const Rank::Slot &sl = rk.slots[size_t(segs[i].slot)];
            const int64_t n = first_of(i) - sl.vt;
            if (n <= 0) continue;
            const size_t o = size_t(stage_off[i]) * HEAD_DIM;
            CK(hipMemcpyAsync(rk.stage_k + o, sl.Kh[qi], size_t(n) * HEAD_DIM * 2, hipMemcpyHostToDevice, rk.s3));
            CK(hipMemcpyAsync(rk.stage_v + o, sl.Vh[qi], size_t(n) * HEAD_DIM * 2, hipMemcpyHostToDevice, rk.s3));
        }
        CK(hipEventRecord(rk.stage_ready, rk.s3));
    };
    // ... and after both micro-batches' attention the rows the chunk wrote, [first, end) of the spilled part, go back.
    auto stage_out = [&](size_t qi, const std::function<int64_t(size_t)> &first_of,
                         const std::function<int64_t(size_t)> &end_of) {
        if (!staging) return;
        for (size_t k = 0; k < mbs.size(); ++k) CK(hipStreamWaitEvent(rk.s3, rk.stage_done[k], 0));
        for (size_t i = 0; i < segs.size(); ++i) {
            if (stage_off[i] < 0) continue;
            const Rank::Slot &sl = rk.slots[size_t(segs[i].slot)];
            const int64_t a = std::max<int64_t>(first_of(i), sl.vt) - sl.vt, e = end_of(i) - sl.vt;
            if (e <= a) continue;
            const size_t o = (size_t(stage_off[i]) + size_t(a)) * HEAD_DIM;
            CK(hipMemcpyAsync(sl.Kh[qi] + size_t(a) * HEAD_DIM, rk.stage_k + o, size_t(e - a) * HEAD_DIM * 2,
                              hipMemcpyDeviceToHost, rk.s3));
            CK(hipMemcpyAsync(sl.Vh[qi] + size_t(a) * HEAD_DIM, rk.stage_v + o, size_t(e - a) * HEAD_DIM * 2,
                              hipMemcpyDeviceToHost, rk.s3));
        }
        CK(hipEventRecord(rk.stage_free, rk.s3));
    };
    // The KV a sub's kernels use for layer qi: the staging buffer in place of the host part when its segment is staged.
    auto sub_kv = [&](const Sub &sb, size_t qi) {
        gpu::KvSplit kv = sb.sl->kv(qi);
        if (stage_off[size_t(sb.seg)] >= 0) {
            kv.Kh = rk.stage_k + size_t(stage_off[size_t(sb.seg)]) * HEAD_DIM;
            kv.Vh = rk.stage_v + size_t(stage_off[size_t(sb.seg)]) * HEAD_DIM;
        }
        return kv;
    };
    auto mb_index = [&](const MB &m) { return size_t(&m - mbs.data()); };

    // B's stream starts after A's (resets, previous chunk's state)
    if (split) {
        CK(hipEventRecord(rk.ev[0], rk.s));
        CK(hipStreamWaitEvent(rk.s2, rk.ev[0], 0));
    }

    for (auto &m : mbs) {
        PrefillBuf &b = *m.b;
        for (int t = 0; t < m.T; ++t)
            std::memcpy(b.h_emb + size_t(t) * SH, pjob_.emb + size_t(m.row0 + t) * H + r * SH, SH * 4);
        std::memcpy(b.h_ple, pjob_.ple + size_t(m.row0) * H, size_t(m.T) * H * 2);
        CK(hipMemcpyAsync(b.emb, b.h_emb, size_t(m.T) * SH * 4, hipMemcpyHostToDevice, m.s));
        CK(hipMemcpyAsync(b.ple_e, b.h_ple, size_t(m.T) * H * 2, hipMemcpyHostToDevice, m.s));
        m.comm->set_base(r, *m.seq_base, m.s);
        gpu::embed_T(b.X, b.emb, m.T, m.s);
    }

    auto gemm = [&](MB &m, const uint16_t *W, int N, int K, const uint16_t *X, int ldx, int rows, void *Y, int ldy,
                    bool f16) { gpu::gemm(m.blas, W, N, K, X, ldx, rows, Y, ldy, f16); };
    auto allreduce = [&](MB &m, const void *src, float *dst, int n, int rows, bool f16 = false) {
        m.comm->allreduce(r, src, dst, n, ++m.k, m.s, rows, f16);
    };
    auto reduce_scatter16 = [&](MB &m, const uint16_t *src, float *dst, int rows) {
        m.comm->reduce_scatter(r, src, dst, SH, ++m.k, m.s, rows, true);
    };
    auto allgather = [&](MB &m, const void *src, void *dst, size_t bytes, int rows) {
        m.comm->allgather(r, src, dst, bytes, ++m.k, m.s, rows);
    };
    // GPTQ calibration: X^T X of a matrix's input (Engine::calib_begin)
    auto cal = [&](MB &m, const std::string &name, const uint16_t *X, int ldx, int rows, int K) {
        calib_acc(rk, name, X, ldx, rows, K, m.blas);
    };
    // HC mix of X rows [0, rows) (after an optional pending combine) into b.bin.
    // cname: calibrate its down projection's inputs (one per stream) under this name.
    auto mix = [&](MB &m, const HcW &w, float *X, int rows, const float *pend, const float *pend_inj, float *inj_out,
                   const std::string &cname = std::string()) {
        PrefillBuf &b = *m.b;
        gpu::hc_pre_T(X, pend, pend_inj, rows, b.ssq, m.s);
        allreduce(m, b.ssq, b.ssq_red, HC, rows);
        gpu::hc_norm_T(X, b.ssq_red, w.w1, rows, b.xn, m.s);
        if (!cname.empty())
            for (int st = 0; st < HC; ++st)
                cal(m, cname + ".down.s" + std::to_string(st), b.xn + size_t(st) * SH, HC * SH, rows, SH);
        // fp16 partials (and a fp16 all-reduce): measured accuracy-neutral.
        gemm(m, w.down_flat, gpu::D_PAD, HC * SH, b.xn, HC * SH, rows, b.d, gpu::D_PAD, true);
        allreduce(m, b.d, b.d_red, gpu::D_PAD, rows, true);
        gpu::hc_mid_T(b.d_red, rows, b.u, inj_out, m.s);
        gemm(m, w.up, HC * SH, HC_RANK, b.u, HC_RANK, rows, b.g, HC * SH, true);
        gpu::hc_mix_T(b.g, b.xn, rows, b.bin_local, m.s);
        allgather(m, b.bin_local, b.bin, SH * 2, rows);
    };
    // Snapshot captures of a sub's segment, as the per-layer destinations of one kind of state.
    auto captures = [&](const Sub &sb, auto dst_of) {
        gpu::Captures c;
        for (const Capture &cp : segs[size_t(sb.seg)].captures) {
            c.pos[c.n] = cp.pos;
            c.dst[c.n++] = dst_of(rk.snaps[size_t(cp.snap)]);
        }
        return c;
    };
    auto record = [&](MB &m, int l, int what) {
        if (split && &m == &mbs[0]) CK(hipEventRecord(rk.ev[size_t(l * 4 + what)], m.s));
    };
    auto wait = [&](MB &m, int l, int what) {
        if (split && &m == &mbs[1]) CK(hipStreamWaitEvent(m.s, rk.ev[size_t(l * 4 + what)], 0));
    };

    // One decoder layer over rows [0, T) of X, the rows of `subs` (after the
    // PLE step, if any); l indexes the micro-batch events. Leaves the MoE
    // output pending in m.pend.
    auto layer = [&](MB &m, LayerW &w, int l, bool qsa, int idx, float *X, const std::vector<Sub> &subs, int T) {
        PrefillBuf &b = *m.b;
        hipStream_t s = m.s;
        const bool calib = l < N_LAYERS && calibrating(l);  // (l == N_LAYERS: the MTP layer)
        char lbuf[8];
        std::snprintf(lbuf, sizeof lbuf, "L%02d", l);
        const std::string ln = calib ? std::string(lbuf) : std::string();
        mix(m, w.attn_hc, X, T, m.pend, b.inj_mlp, b.inj_attn, calib ? ln + ".attn_hc" : std::string());
        if (calib) cal(m, ln + (qsa ? ".qsa_proj" : ".gdn_proj"), b.bin, H, T, H);
        if (qsa) {
            const size_t qi = size_t(idx);
            constexpr size_t QW = size_t(QH_L) * HEAD_DIM, IW = size_t(IDX_HEADS) * IDX_DIM;
            gemm(m, w.qsa_proj, QSA_PROJ_L, H, b.bin, H, T, b.P, QSA_PROJ_L, true);
            if (staging) CK(hipStreamWaitEvent(s, rk.stage_ready, 0));
            for (const Sub &sb : subs) {
                const int32_t *r3 = rope3 && l < N_LAYERS ? rope3 + size_t(sb.start - start0) * 3 : nullptr;
                gpu::qsa_prep_T(b.P + size_t(sb.row) * QSA_PROJ_L, w.qn, w.kn, w.iqn, sb.start, sb.T,
                                b.q16 + sb.row * QW, b.qgate + sb.row * QW, b.iq16 + sb.row * IW, sub_kv(sb, qi), s, r3);
            }
            wait(m, l, 2);  // A's K/V and raw keys (a group can straddle the halves)
            for (const Sub &sb : subs)
                gpu::qsa_compress_T(sb.sl->kv(qi), w.ikn, sb.start, sb.T, sb.sl->ck[qi], s,
                                    l < N_LAYERS ? rope3 : nullptr, start0);
            record(m, l, 2);
            for (const Sub &sb : subs)
                for (int q0 = 0; q0 < sb.T; q0 += b.Qb) {
                    const int Q = std::min(b.Qb, sb.T - q0);
                    const int64_t p0 = sb.start + q0;
                    const size_t row = size_t(sb.row + q0);
                    const int nb_max = int((p0 + Q) / IDX_RATIO);
                    if (nb_max > IDX_TOPK_BLOCKS) {
                        const int ld = (nb_max + 7) / 8 * 8;
                        gemm(m, sb.sl->ck[qi], nb_max, IDX_DIM, b.iq16 + row * IW, IDX_DIM, Q * IDX_HEADS, b.sc16, ld,
                             true);
                        gpu::qsa_score_reduce_T(b.sc16, ld, p0, Q, b.scores, s);
                        gpu::qsa_select_T(b.scores, ld, p0, Q, b.lists, b.lcounts, s);
                    } else {
                        gpu::qsa_select_T(b.scores, 0, p0, Q, b.lists, b.lcounts, s);
                    }
                    gpu::qsa_attend_T(b.q16 + row * QW, b.qgate + row * QW, sub_kv(sb, qi), b.lists,
                                      b.lcounts, Q, b.att_partial, b.act + row * QW, s);
                }
            if (staging) CK(hipEventRecord(rk.stage_done[mb_index(m)], s));
            if (calib) cal(m, ln + ".qsa_o", b.act, QH_L * HEAD_DIM, T, QH_L * HEAD_DIM);
            gemm(m, w.qsa_o, H, QH_L * HEAD_DIM, b.act, QH_L * HEAD_DIM, T, b.part16, H, true);
        } else {
            const size_t gi = size_t(idx);
            constexpr size_t OW = size_t(GDN_VL) * GDN_DIM, GW = size_t(GDN_VL) * 2;
            gemm(m, w.gdn_proj, GDN_PROJ_L, H, b.bin, H, T, b.P, GDN_PROJ_L, true);
            wait(m, l, 0);
            for (const Sub &sb : subs) {
                float *ring = sb.sl->ring + gi * GDN_RING * GDN_QKV_L;
                const uint16_t *P = b.P + size_t(sb.row) * GDN_PROJ_L;
                gpu::gdn_conv_T(P, ring, w.gdn_conv, w.gdn_A, w.gdn_dt, sb.start, sb.T,
                                b.qkv + size_t(sb.row) * GDN_QKV_L, b.gb + sb.row * GW, s);
                gpu::gdn_ring_update_T(P, ring, sb.start, sb.T, s, captures(sb, [&](const Rank::Snap &sn) {
                                           return sn.ring + gi * GDN_RING * GDN_QKV_L;
                                       }));
            }
            record(m, l, 0);
            wait(m, l, 1);
            for (const Sub &sb : subs) {
                gpu::gdn_scan_T(b.qkv + size_t(sb.row) * GDN_QKV_L, b.gb + sb.row * GW, sb.sl->S + gi * GDN_STATE,
                                sb.T, b.o_raw + sb.row * OW, s, sb.start,
                                captures(sb, [&](const Rank::Snap &sn) { return sn.S + gi * GDN_STATE; }));
                if (pjob_.transfer)
                    gpu::gdn_transfer_T(b.qkv + size_t(sb.row) * GDN_QKV_L, b.gb + sb.row * GW,
                                        rk.transfer + gi * GDN_STATE, sb.T, s);
            }
            record(m, l, 1);
            gpu::gdn_norm_T(b.o_raw, b.P, w.gdn_norm, T, b.act, s);
            if (calib) cal(m, ln + ".gdn_out", b.act, GDN_VL * GDN_DIM, T, GDN_VL * GDN_DIM);
            gemm(m, w.gdn_out, H, GDN_VL * GDN_DIM, b.act, GDN_VL * GDN_DIM, T, b.part16, H, true);
        }
        reduce_scatter16(m, b.part16, b.out_attn, T);

        mix(m, w.mlp_hc, X, T, b.out_attn, b.inj_attn, b.inj_mlp, calib ? ln + ".mlp_hc" : std::string());
        if (calib) cal(m, ln + ".sh_gu", b.bin, H, T, H);
        // Router logits stay fp32: fp16 output flips near-tie expert choices
        // (measured: mean |dlogprob| vs the fp32 reference 0.074 -> 0.088).
        gemm(m, w.router, N_EXPERTS, H, b.bin, H, T, b.logits, N_EXPERTS, false);
        CK(hipMemsetAsync(b.counts, 0, EXP_L * 4, s));
        gpu::moe_route_T(b.logits, T, r * EXP_L, b.tok_e, b.tok_w, b.counts, s);
        if (moe_stats_)  // this micro-batch's tokens per local expert (read after the job, see collect_moe_balance)
            CK(hipMemcpyAsync(moe_counts_[size_t(r)] + (size_t(&m != &mbs[0]) * (N_LAYERS + 1) + size_t(l)) * EXP_L,
                              b.counts, EXP_L * 4, hipMemcpyDeviceToHost, s));
        gpu::moe_scatter_T(b.tok_e, b.counts, T, b.offsets, b.fill, b.tok_slot, b.pair_tok, s);
        static const bool int8_experts = std::getenv("QW_INT8_EXPERTS") != nullptr;
        gpu::moe_experts_T(w.eg, w.egs, w.eu, w.eus, w.ed, w.eds, b.counts, b.offsets, b.pair_tok, b.bin, T, b.tiles,
                           b.moe_h, b.yp, s, int8_experts ? b.q8 : nullptr, b.q8s);
        gemm(m, w.sh_gu, 2 * SFFN_L + 1, H, b.bin, H, T, b.part16, 2 * SFFN_L + 1, true);  // fp16 out: neutral
        gpu::f16_to_f32_T(b.part16, b.shp, size_t(T) * (2 * SFFN_L + 1), s);
        gpu::swiglu_T(b.shp, 2 * SFFN_L + 1, SFFN_L, T, b.sh_h, s);
        if (calib) cal(m, ln + ".sh_down", b.sh_h, SFFN_L, T, SFFN_L);
        gemm(m, w.sh_down, H, SFFN_L, b.sh_h, SFFN_L, T, b.sh_out, H, true);
        gpu::moe_combine_T(b.yp, b.tok_slot, b.tok_w, b.sh_out, T, b.part16, s);
        reduce_scatter16(m, b.part16, b.out_moe, T);
        m.pend = b.out_moe;
    };

    for (int l = 0; l < debug_layers_; ++l) {
        LayerW &w = rk.L[size_t(l)];
        for (auto &m : mbs) {
            PrefillBuf &b = *m.b;
            hipStream_t s = m.s;
            if (l == PLE_LAYER) {
                if (m.pend) gpu::hc_combine_T(b.X, m.pend, b.inj_mlp, m.T, s);
                m.pend = nullptr;
                if (calibrating(PLE_LAYER)) cal(m, "ple_kv", b.ple_e, H, m.T, H);
                gemm(m, rk.ple_kv, PLE_KV, H, b.ple_e, H, m.T, b.kv, PLE_KV, true);
                gpu::ple_stats_T(b.kv, b.X, rk.ple_nk, rk.ple_nq, m.T, b.ple_send, s);
                allreduce(m, b.ple_send, b.ple_red, 16, m.T);
                wait(m, l, 3);
                for (const Sub &sb : m.subs)
                    gpu::ple_apply_T(b.kv + size_t(sb.row) * PLE_KV, b.ple_red + size_t(sb.row) * 16, rk.ple_nc,
                                     rk.ple_conv, sb.sl->ple, sb.start, sb.T, b.ncbuf + size_t(sb.row) * XW,
                                     b.X + size_t(sb.row) * XW, s,
                                     captures(sb, [](const Rank::Snap &sn) { return sn.ple; }));
                record(m, l, 3);
            }
            const bool qsa = is_qsa(l);
            if (qsa && &m == &mbs[0]) stage_in(size_t(qsa_index(l)), [&](size_t i) { return segs[i].start; });
            layer(m, w, l, qsa, qsa ? qsa_index(l) : gdn_index(l), b.X, m.subs, m.T);
        }
        if (is_qsa(l))
            stage_out(size_t(qsa_index(l)), [&](size_t i) { return segs[i].start; },
                      [&](size_t i) { return segs[i].start + segs[i].T; });
    }
    for (auto &m : mbs) {  // complete X: the pre-final-mixer hidden (lm_head and MTP input)
        gpu::hc_combine_T(m.b->X, m.pend, m.b->inj_mlp, m.T, m.s);
        m.pend = nullptr;
        for (const Sub &sb : m.subs)
            for (const Capture &cp : segs[size_t(sb.seg)].captures)  // a capture's MTP input: its last token's hidden
                if (cp.pos > sb.start && cp.pos <= sb.start + sb.T)
                    CK(hipMemcpyAsync(rk.snaps[size_t(cp.snap)].pend,
                                      m.b->X + size_t(sb.row + cp.pos - 1 - sb.start) * XW, XW * 4,
                                      hipMemcpyDeviceToDevice, m.s));
    }
    // MTP layer over the rows whose next token is known: row t of a segment
    // pairs X[t] with token t + 1. A segment's rows start one early, from the
    // slot's MTP input store (the previous chunk's or micro-batch's last
    // hidden), and its last hidden goes into the store.
    // the MTP layer's rows of a segment: [start - 1, end - 1) when the slot's MTP input store holds the hidden of the
    // token before the segment, else [start, end - 1)
    auto mtp_first = [&](size_t i) { return segs[i].start - (segs[i].mtp_pend ? 1 : 0); };
    auto mtp_end = [&](size_t i) { return segs[i].start + segs[i].T - 1; };
    if (mtp_) stage_in(size_t(N_QSA), mtp_first);
    if (mtp_)
        for (auto &m : mbs) {
            PrefillBuf &b = *m.b;
            hipStream_t s = m.s;
            std::vector<Sub> msubs;  // the MTP rows of each sub, packed from row 0 of Xm
            std::vector<int> first;  // row of b.emb with the next token of each one's first MTP row
            int rows = 0;
            wait(m, N_LAYERS, 3);  // A's store write and MTP K/V
            for (const Sub &sb : m.subs) {
                const int o = (sb.cont || segs[size_t(sb.seg)].mtp_pend) ? 1 : 0;
                const int n = sb.T - 1 + o;
                if (n <= 0) continue;
                gpu::mtp_shift(sb.sl->pend, b.X + size_t(sb.row) * XW, o, n, b.Xm + size_t(rows) * XW, s);
                msubs.push_back({sb.seg, sb.sl, sb.start - o, n, rows, false, false});
                first.push_back(sb.row + 1 - o);
                rows += n;
            }
            if (rows > 0) {
                // next tokens' embeddings (rank slice): a sub's MTP row j pairs with its row j + 1 - o
                const float *emb = b.emb + size_t(first[0]) * SH;
                if (msubs.size() > 1) {
                    emb = b.m_emb;
                    for (size_t j = 0; j < msubs.size(); ++j)
                        CK(hipMemcpyAsync(b.m_emb + size_t(msubs[j].row) * SH, b.emb + size_t(first[j]) * SH,
                                          size_t(msubs[j].T) * SH * 4, hipMemcpyDeviceToDevice, s));
                }
                gpu::mtp_ssq(b.Xm, emb, rows, b.m_send, s);
                allreduce(m, b.m_send, b.m_red, 4, rows);
                gpu::mtp_norm(b.Xm, emb, b.m_red, rk.mtp_nh, rk.mtp_ne, rows, b.m_xn, s);
                for (int st = 0; st < 5; ++st) {
                    gemm(m, st < HC ? rk.mtp_fc_h : rk.mtp_fc_e, H, SH, b.m_xn + size_t(st) * rows * SH, SH, rows,
                         b.part16, H, true);
                    reduce_scatter16(m, b.part16, b.m_out + size_t(st) * rows * SH, rows);
                }
                gpu::mtp_add(b.m_out, rows, b.Xm, s);
                layer(m, rk.mtp, N_LAYERS, true, N_QSA, b.Xm, msubs, rows);
                m.pend = nullptr;  // the MTP output is not needed here
            }
            for (const Sub &sb : m.subs)
                CK(hipMemcpyAsync(sb.sl->pend, b.X + size_t(sb.row + sb.T - 1) * XW, XW * 4, hipMemcpyDeviceToDevice,
                                  s));
            record(m, N_LAYERS, 3);
        }
    if (mtp_) stage_out(size_t(N_QSA), mtp_first, mtp_end);
    // final mixer + lm_head: every row in all-logits mode, else each segment's last one
    struct Out {
        MB *m;
        int rows;
        std::vector<float *> dst;  // host destination (rank slice) of each row
    };
    std::vector<Out> outs;
    for (auto &m : mbs) {
        PrefillBuf &b = *m.b;
        const bool calib_final = calibrating(N_LAYERS);
        Out o{&m, 0, {}};
        float *X = nullptr;
        if (pjob_.all_logits) {
            X = b.X;
            o.rows = m.T;
            for (int i = 0; i < m.T; ++i) o.dst.push_back(plogits_.data() + size_t(m.row0 + i) * VOCAB + size_t(r) * VOCAB_L);
        } else {
            std::vector<const Sub *> ends;
            for (const Sub &sb : m.subs)
                if (sb.last) ends.push_back(&sb);
            if (ends.empty()) continue;
            if (ends.size() == 1) {
                X = b.X + size_t(ends[0]->row + ends[0]->T - 1) * XW;
            } else {
                X = b.xg;
                for (size_t i = 0; i < ends.size(); ++i)
                    CK(hipMemcpyAsync(b.xg + i * XW, b.X + size_t(ends[i]->row + ends[i]->T - 1) * XW, XW * 4,
                                      hipMemcpyDeviceToDevice, m.s));
            }
            o.rows = int(ends.size());
            for (const Sub *sb : ends) o.dst.push_back(seg_logits_[size_t(sb->seg)].data() + size_t(r) * VOCAB_L);
        }
        mix(m, rk.final_hc, X, o.rows, nullptr, nullptr, nullptr, calib_final ? std::string("final_hc") : std::string());
        if (calib_final) cal(m, "lm_head", b.bin, H, o.rows, H);
        gemm(m, rk.lm_head, VOCAB_L, H, b.bin, H, o.rows, b.lm_logits, VOCAB_L, false);
        outs.push_back(std::move(o));
    }
    for (auto &m : mbs) {
        *m.seq_base += m.k;
        phase_[size_t(r)] = 2;
        CK(hipStreamSynchronize(m.s));
    }
    if (staging) CK(hipStreamSynchronize(rk.s3));  // the spilled rows are back in host memory before anyone reads them
    // captures: the raw indexer keys before each one (the rings still hold the whole chunk and RAW_TAIL before it)
    bool captured = false;
    for (const PrefillSeg &sg : segs)
        for (const Capture &cp : sg.captures) {
            raw_tail_copy(rk.slots[size_t(sg.slot)], N_QSA + int(mtp_), cp.pos, rk.snaps[size_t(cp.snap)].raw_tail, true,
                          hipMemcpyDeviceToDevice, rk.s);
            captured = true;
        }
    if (captured) CK(hipStreamSynchronize(rk.s));
    for (const Out &o : outs)
        for (int i = 0; i < o.rows; ++i)
            CK(hipMemcpy(o.dst[size_t(i)], o.m->b->lm_logits + size_t(i) * VOCAB_L, VOCAB_L * 4,
                         hipMemcpyDeviceToHost));
    if (profile) {
        const auto t_end = std::chrono::steady_clock::now();
        log("rank %d prefill [%lld, +%d) %zu segment(s)%s: %.1f ms", r, (long long)start0, T, segs.size(),
            split ? " split" : "", std::chrono::duration<double, std::milli>(t_end - t_start).count());
    }
}

void Engine::prefill_inputs(int slot, const std::vector<int32_t> &chunk, int row0,
                            const std::vector<EmbedSpan> &embeds) {
    const SlotHost &sh = slots_[size_t(slot)];
    prefill_inputs(sh.len, sh.hist, chunk, row0, embeds, pemb_.data(), pple_.data());
}

void Engine::prefill_inputs(int64_t len, const std::vector<int32_t> &hist, const std::vector<int32_t> &chunk, int row0,
                            const std::vector<EmbedSpan> &embeds, float *pemb, uint16_t *pple) const {
    const int T = int(chunk.size());
    for (int t = 0; t < T; ++t) {
        const int32_t tok = chunk[size_t(t)];
        QW_CHECK(tok < VOCAB, "token out of range");
        if (tok < 0) continue;  // a vision token: its row comes from `embeds`
        for (int j = 0; j < H; ++j) pemb[size_t(row0 + t) * H + j] = bf16_to_f32(embed_[size_t(tok) * H + j]);
    }
    int64_t covered = 0;
    const int64_t c_start = len;
    for (const EmbedSpan &sp : embeds) {
        const int64_t a = std::max(sp.pos, c_start), b = std::min(sp.pos + sp.n, c_start + T);
        for (int64_t p = a; p < b; ++p)
            std::memcpy(&pemb[size_t(row0 + p - c_start) * H], sp.rows + size_t(p - sp.pos) * H, H * 4);
        covered += std::max<int64_t>(0, b - a);
    }
    QW_CHECK(covered == std::count_if(chunk.begin(), chunk.end(), [](int32_t t) { return t < 0; }),
             "prefill: vision tokens without embeddings (or embeddings over text tokens)");
    auto ids = hasher_.ids_for(hist, chunk);
    std::vector<float> e(H);
    for (int t = 0; t < T; ++t) {
        ple_->gather(ids[size_t(t)], e.data());
        for (int j = 0; j < H; ++j) pple[size_t(row0 + t) * H + j] = f32_to_f16(e[size_t(j)]);
    }
}

void Engine::prefill_commit(int slot, const std::vector<int32_t> &chunk) {
    SlotHost &sh = slots_[size_t(slot)];
    const ptrdiff_t T = ptrdiff_t(chunk.size());
    sh.reset = false;
    sh.hist.insert(sh.hist.end(), chunk.end() - std::min<ptrdiff_t>(T, 32), chunk.end());
    if (sh.hist.size() > 32) sh.hist.erase(sh.hist.begin(), sh.hist.end() - 32);
    sh.len += T;
    sh.pend_valid = mtp_;
    sh.run_first = -1;
}

const std::vector<float> &Engine::prefill(int slot, const std::vector<int32_t> &tokens, std::vector<float> *all_logits,
                                          const std::function<void(int64_t)> &after_chunk,
                                          const std::vector<Capture> &captures,
                                          const std::vector<EmbedSpan> &embeds, const std::vector<int32_t> *rope3) {
    QW_CHECK(!tokens.empty(), "empty prefill");
    QW_CHECK(slot >= 0 && slot < num_slots(), "bad slot");
    SlotHost &sh = slots_[size_t(slot)];
    QW_CHECK(sh.len + int64_t(tokens.size()) <= slot_capacity(slot), "prefill exceeds the slot's capacity");
    for (const Capture &c : captures)
        QW_CHECK(c.pos > sh.len && c.pos <= sh.len + int64_t(tokens.size()) && c.snap >= 0 && c.snap < SNAPSHOTS,
                 "prefill: capture outside the new tokens");
    if (all_logits) all_logits->clear();
    const bool all = all_logits != nullptr;
    if (all) {  // lm_head output for a whole chunk
        for (auto &rkp : ranks_) {
            CK(hipSetDevice(rkp->dev));
            for (auto &b : rkp->pb) {
                CK(hipFree(b.lm_logits));
                b.lm_logits = zeros<float>(size_t(std::max(b.C, MAX_SEGMENTS)) * VOCAB_L);
            }
        }
    }
    seg_logits_.resize(1);
    // QW_PROFILE: where the wall time of a prefill goes: the host side before each chunk (embedding and PLE rows,
    // the GPUs idle), the chunk's job, and the caller's after_chunk (the prefix cache's saves, the GPUs idle again)
    static const bool profile = std::getenv("QW_PROFILE") != nullptr;
    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    double t_inputs = 0, t_job = 0, t_after = 0;
    int chunks = 0;
    const auto t_begin = clk::now();
    std::vector<float> *emb_cur = &pemb_, *emb_next = &pemb2_;
    std::vector<uint16_t> *ple_cur = &pple_, *ple_next = &pple2_;
    std::future<void> next_ready;  // the helper thread preparing the next chunk's inputs (QW_PREFILL_PIPELINE)
    const size_t chunk_size = size_t(opt_.prefill_chunk);
    for (size_t c0 = 0; c0 < tokens.size(); c0 += chunk_size) {
        const int T = int(std::min<size_t>(chunk_size, tokens.size() - c0));
        std::vector<int32_t> chunk(tokens.begin() + ptrdiff_t(c0), tokens.begin() + ptrdiff_t(c0) + T);
        const auto t0 = clk::now();
        if (next_ready.valid()) {
            next_ready.get();  // prepared during the previous chunk's job
        } else {
            emb_cur->resize(size_t(T) * H);
            ple_cur->resize(size_t(T) * H);
            prefill_inputs(sh.len, sh.hist, chunk, 0, embeds, emb_cur->data(), ple_cur->data());
        }
        const auto t1 = clk::now();
        PrefillSeg sg{slot, sh.len, T, 0, sh.reset, mtp_ && sh.pend_valid && sh.len > 0, {}};
        for (const Capture &c : captures)
            if (c.pos > sh.len && c.pos <= sh.len + T) sg.captures.push_back(c);
        QW_CHECK(int(sg.captures.size()) <= MAX_CAPTURES, "prefill: too many captures in one chunk");
        pjob_ = {{std::move(sg)}, T, all, blend_transfer_, rope3 ? rope3->data() + c0 * 3 : nullptr, emb_cur->data(),
                 ple_cur->data()};
        if (all) plogits_.assign(size_t(T) * VOCAB, 0.f);
        seg_logits_[0].resize(VOCAB);
        job_ = Job::Prefill;
        if (pipeline_ && c0 + chunk_size < tokens.size()) {
            // the next chunk's rows, while the GPUs run this one: its context is this chunk's tokens appended to the
            // slot's history (as prefill_commit will leave it), its position the slot's length plus this chunk
            const size_t Tn = std::min<size_t>(chunk_size, tokens.size() - c0 - size_t(T));
            std::vector<int32_t> next(tokens.begin() + ptrdiff_t(c0) + T, tokens.begin() + ptrdiff_t(c0) + T + ptrdiff_t(Tn));
            std::vector<int32_t> hist = sh.hist;
            hist.insert(hist.end(), chunk.end() - std::min<ptrdiff_t>(T, 32), chunk.end());
            if (hist.size() > 32) hist.erase(hist.begin(), hist.end() - 32);
            emb_next->resize(Tn * H);
            ple_next->resize(Tn * H);
            next_ready = std::async(std::launch::async, [this, len = sh.len + T, hist = std::move(hist), next = std::move(next),
                                                         &embeds, e = emb_next->data(), p = ple_next->data()] {
                prefill_inputs(len, hist, next, 0, embeds, e, p);
            });
        }
        dispatch();
        const auto t2 = clk::now();
        if (moe_stats_) collect_moe_balance();
        if (!all) std::swap(logits_, seg_logits_[0]);
        prefill_commit(slot, chunk);
        if (all) all_logits->insert(all_logits->end(), plogits_.begin(), plogits_.end());
        if (after_chunk) after_chunk(sh.len);
        std::swap(emb_cur, emb_next);
        std::swap(ple_cur, ple_next);
        t_inputs += ms(t0, t1);
        t_job += ms(t1, t2);
        t_after += ms(t2, clk::now());
        ++chunks;
    }
    if (all) std::copy(all_logits->end() - VOCAB, all_logits->end(), logits_.begin());
    if (profile && chunks)
        log("prefill %zu tokens in %.0f ms (%d chunks): host inputs %.0f ms, jobs %.0f ms, after_chunk %.0f ms",
            tokens.size(), ms(t_begin, clk::now()), chunks, t_inputs, t_job, t_after);
    if (moe_stats_) log_moe_balance();
    return logits_;
}

const std::vector<std::vector<float>> &Engine::prefill_batch(const std::vector<Segment> &segs) {
    QW_CHECK(!segs.empty() && int(segs.size()) <= MAX_SEGMENTS, "prefill_batch: 1.." + std::to_string(MAX_SEGMENTS) +
                                                                     " segments");
    int T = 0;
    for (size_t i = 0; i < segs.size(); ++i) {
        const Segment &g = segs[i];
        QW_CHECK(g.slot >= 0 && g.slot < num_slots(), "prefill_batch: bad slot");
        for (size_t j = 0; j < i; ++j) QW_CHECK(segs[j].slot != g.slot, "prefill_batch: slots must be distinct");
        QW_CHECK(!g.tokens.empty(), "prefill_batch: empty segment");
        const SlotHost &sh = slots_[size_t(g.slot)];
        QW_CHECK(sh.len + int64_t(g.tokens.size()) <= slot_capacity(g.slot), "prefill exceeds the slot's capacity");
        QW_CHECK(int(g.captures.size()) <= MAX_CAPTURES, "prefill_batch: too many captures in one segment");
        for (const Capture &c : g.captures)
            QW_CHECK(c.pos > sh.len && c.pos <= sh.len + int64_t(g.tokens.size()) && c.snap >= 0 && c.snap < SNAPSHOTS,
                     "prefill: capture outside the new tokens");
        T += int(g.tokens.size());
    }
    QW_CHECK(T <= opt_.prefill_chunk, "prefill_batch: more tokens than one prefill chunk");
    pemb_.resize(size_t(T) * H);
    pple_.resize(size_t(T) * H);
    PrefillJob job;
    job.T = T;
    job.emb = pemb_.data();
    job.ple = pple_.data();
    int row = 0;
    for (const Segment &g : segs) {
        prefill_inputs(g.slot, g.tokens, row, {});
        const SlotHost &sh = slots_[size_t(g.slot)];
        job.segs.push_back(
            {g.slot, sh.len, int(g.tokens.size()), row, sh.reset, mtp_ && sh.pend_valid && sh.len > 0, g.captures});
        row += int(g.tokens.size());
    }
    pjob_ = std::move(job);
    seg_logits_.resize(segs.size());
    for (auto &l : seg_logits_) l.resize(VOCAB);
    job_ = Job::Prefill;
    dispatch();
    for (const Segment &g : segs) prefill_commit(g.slot, g.tokens);
    return seg_logits_;
}

}  // namespace qw
