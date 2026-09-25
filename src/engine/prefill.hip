// Chunked prefill of one slot: two micro-batches per chunk on two streams,
// the MTP layer over the chunk, and the host side of prefill().
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

// One prefill chunk (pjob_) on rank rk, as two micro-batches (A = first
// half, B = second half) on two streams with separate collective channels, so
// one half's collectives overlap the other half's compute. B depends on A only
// through sequential state, per layer: the GDN conv tail and recurrent state,
// the QSA K/V and compressed keys, and the PLE conv tail. The host enqueues
// layer l of A before layer l of B, so every event wait follows its record.
// Short chunks run as a single micro-batch.
void Engine::run_prefill_rank(Rank &rk) {
    static const bool profile = std::getenv("QW_PROFILE") != nullptr;
    static const bool no_split = std::getenv("QW_NO_SPLIT") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    const int r = rk.r;
    const int64_t start = pjob_.start;
    const int T = pjob_.T;
    Rank::Slot &sl = rk.slots[size_t(pjob_.slot)];
    if (pjob_.reset) {
        CK(hipMemsetAsync(sl.S, 0, SLOT_S * 4, rk.s));
        CK(hipMemsetAsync(sl.ring, 0, SLOT_RING * 4, rk.s));
        CK(hipMemsetAsync(sl.ple, 0, SLOT_PLE * 4, rk.s));
    }
    const bool split = T >= 256;  // two micro-batches; QW_NO_SPLIT runs them serially on one stream
    const int TA = split ? T / 2 : T;
    QW_CHECK(TA <= rk.pb[0].C && T - TA <= rk.pb[1].C, "prefill chunk larger than its buffers");

    struct MB {
        PrefillBuf *b;
        hipStream_t s;
        rocblas_handle blas;
        Comm *comm;
        uint32_t *seq_base;
        uint32_t k;
        int64_t start;
        int T;
        int row0;  // offset of this micro-batch in the chunk
        const float *pend;
    };
    std::vector<MB> mbs;
    mbs.push_back({&rk.pb[0], rk.s, rk.blas, comm_.get(), &rk.seq_base, 0, start, TA, 0, nullptr});
    if (split)
        mbs.push_back({&rk.pb[1], no_split ? rk.s : rk.s2, no_split ? rk.blas : rk.blas2, comm2_.get(), &rk.seq_base2,
                       0, start + TA, T - TA, TA, nullptr});
    // B's stream starts after A's (resets, previous chunk's state)
    if (split) {
        CK(hipEventRecord(rk.ev[0], rk.s));
        CK(hipStreamWaitEvent(rk.s2, rk.ev[0], 0));
    }

    for (auto &m : mbs) {
        PrefillBuf &b = *m.b;
        for (int t = 0; t < m.T; ++t)
            std::memcpy(b.h_emb + size_t(t) * SH, pemb_.data() + size_t(m.row0 + t) * H + r * SH, SH * 4);
        std::memcpy(b.h_ple, pple_.data() + size_t(m.row0) * H, size_t(m.T) * H * 2);
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
    // HC mix of X rows [0, rows) (after an optional pending combine) into b.bin.
    auto mix = [&](MB &m, const HcW &w, float *X, int rows, const float *pend, const float *pend_inj, float *inj_out) {
        PrefillBuf &b = *m.b;
        gpu::hc_pre_T(X, pend, pend_inj, rows, b.ssq, m.s);
        allreduce(m, b.ssq, b.ssq_red, HC, rows);
        gpu::hc_norm_T(X, b.ssq_red, w.w1, rows, b.xn, m.s);
        // fp16 partials (and a fp16 all-reduce): measured accuracy-neutral.
        gemm(m, w.down_flat, gpu::D_PAD, HC * SH, b.xn, HC * SH, rows, b.d, gpu::D_PAD, true);
        allreduce(m, b.d, b.d_red, gpu::D_PAD, rows, true);
        gpu::hc_mid_T(b.d_red, rows, b.u, inj_out, m.s);
        gemm(m, w.up, HC * SH, HC_RANK, b.u, HC_RANK, rows, b.g, HC * SH, true);
        gpu::hc_mix_T(b.g, b.xn, rows, b.bin_local, m.s);
        allgather(m, b.bin_local, b.bin, SH * 2, rows);
    };
    // Snapshot captures of this chunk, as the per-layer destinations of one kind of state.
    const auto &caps = pjob_.captures;
    auto captures = [&](auto dst_of) {
        gpu::Captures c;
        for (const Capture &cp : caps) {
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

    // One decoder layer over rows [0, T) of X at positions start.. (after the
    // PLE step, if any); l indexes the micro-batch events. Leaves the MoE
    // output pending in m.pend.
    auto layer = [&](MB &m, LayerW &w, int l, bool qsa, int idx, float *X, int64_t start, int T) {
        PrefillBuf &b = *m.b;
        hipStream_t s = m.s;
        mix(m, w.attn_hc, X, T, m.pend, b.inj_mlp, b.inj_attn);
        if (qsa) {
            const int qi = idx;
            gemm(m, w.qsa_proj, QSA_PROJ_L, H, b.bin, H, T, b.P, QSA_PROJ_L, true);
            gpu::qsa_prep_T(b.P, w.qn, w.kn, w.iqn, start, T, b.q16, b.qgate, b.iq16, sl.K[size_t(qi)],
                            sl.V[size_t(qi)], sl.raw_k[size_t(qi)], s);
            wait(m, l, 2);  // A's K/V and raw keys (a group can straddle the halves)
            gpu::qsa_compress_T(sl.raw_k[size_t(qi)], w.ikn, start, T, sl.ck[size_t(qi)], s);
            record(m, l, 2);
            for (int q0 = 0; q0 < T; q0 += b.Qb) {
                const int Q = std::min(b.Qb, T - q0);
                const int64_t p0 = start + q0;
                const int nb_max = int((p0 + Q) / IDX_RATIO);
                if (nb_max > IDX_TOPK_BLOCKS) {
                    const int ld = (nb_max + 7) / 8 * 8;
                    gemm(m, sl.ck[size_t(qi)], nb_max, IDX_DIM, b.iq16 + size_t(q0) * IDX_HEADS * IDX_DIM, IDX_DIM,
                         Q * IDX_HEADS, b.sc16, ld, true);
                    gpu::qsa_score_reduce_T(b.sc16, ld, p0, Q, b.scores, s);
                    gpu::qsa_select_T(b.scores, ld, p0, Q, b.lists, b.lcounts, s);
                } else {
                    gpu::qsa_select_T(b.scores, 0, p0, Q, b.lists, b.lcounts, s);
                }
                gpu::qsa_attend_T(b.q16 + size_t(q0) * QH_L * HEAD_DIM, b.qgate + size_t(q0) * QH_L * HEAD_DIM,
                                  sl.K[size_t(qi)], sl.V[size_t(qi)], b.lists, b.lcounts, Q, b.att_partial,
                                  b.act + size_t(q0) * QH_L * HEAD_DIM, s);
            }
            gemm(m, w.qsa_o, H, QH_L * HEAD_DIM, b.act, QH_L * HEAD_DIM, T, b.part16, H, true);
        } else {
            const int gi = idx;
            float *ring = sl.ring + size_t(gi) * GDN_RING * GDN_QKV_L;
            gemm(m, w.gdn_proj, GDN_PROJ_L, H, b.bin, H, T, b.P, GDN_PROJ_L, true);
            wait(m, l, 0);
            gpu::gdn_conv_T(b.P, ring, w.gdn_conv, w.gdn_A, w.gdn_dt, start, T, b.qkv, b.gb, s);
            gpu::gdn_ring_update_T(b.P, ring, start, T, s, captures([&](const Rank::Snap &sn) {
                                       return sn.ring + size_t(gi) * GDN_RING * GDN_QKV_L;
                                   }));
            record(m, l, 0);
            wait(m, l, 1);
            gpu::gdn_scan_T(b.qkv, b.gb, sl.S + size_t(gi) * GDN_STATE, T, b.o_raw, s, start,
                            captures([&](const Rank::Snap &sn) { return sn.S + size_t(gi) * GDN_STATE; }));
            if (pjob_.transfer) gpu::gdn_transfer_T(b.qkv, b.gb, rk.transfer + size_t(gi) * GDN_STATE, T, s);
            record(m, l, 1);
            gpu::gdn_norm_T(b.o_raw, b.P, w.gdn_norm, T, b.act, s);
            gemm(m, w.gdn_out, H, GDN_VL * GDN_DIM, b.act, GDN_VL * GDN_DIM, T, b.part16, H, true);
        }
        reduce_scatter16(m, b.part16, b.out_attn, T);

        mix(m, w.mlp_hc, X, T, b.out_attn, b.inj_attn, b.inj_mlp);
        // Router logits stay fp32: fp16 output flips near-tie expert choices
        // (measured: mean |dlogprob| vs the fp32 reference 0.074 -> 0.088).
        gemm(m, w.router, N_EXPERTS, H, b.bin, H, T, b.logits, N_EXPERTS, false);
        CK(hipMemsetAsync(b.counts, 0, EXP_L * 4, s));
        gpu::moe_route_T(b.logits, T, r * EXP_L, b.tok_e, b.tok_w, b.counts, s);
        gpu::moe_scatter_T(b.tok_e, b.counts, T, b.offsets, b.fill, b.tok_slot, b.pair_tok, s);
        static const bool int8_experts = std::getenv("QW_INT8_EXPERTS") != nullptr;
        gpu::moe_experts_T(w.eg, w.egs, w.eu, w.eus, w.ed, w.eds, b.counts, b.offsets, b.pair_tok, b.bin, T, b.tiles,
                           b.moe_h, b.yp, s, int8_experts ? b.q8 : nullptr, b.q8s);
        gemm(m, w.sh_gu, 2 * SFFN_L + 1, H, b.bin, H, T, b.part16, 2 * SFFN_L + 1, true);  // fp16 out: neutral
        gpu::f16_to_f32_T(b.part16, b.shp, size_t(T) * (2 * SFFN_L + 1), s);
        gpu::swiglu_T(b.shp, 2 * SFFN_L + 1, SFFN_L, T, b.sh_h, s);
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
                gemm(m, rk.ple_kv, PLE_KV, H, b.ple_e, H, m.T, b.kv, PLE_KV, true);
                gpu::ple_stats_T(b.kv, b.X, rk.ple_nk, rk.ple_nq, m.T, b.ple_send, s);
                allreduce(m, b.ple_send, b.ple_red, 16, m.T);
                wait(m, l, 3);
                gpu::ple_apply_T(b.kv, b.ple_red, rk.ple_nc, rk.ple_conv, sl.ple, m.start, m.T, b.ncbuf, b.X, s,
                                 captures([](const Rank::Snap &sn) { return sn.ple; }));
                record(m, l, 3);
            }
            const bool qsa = is_qsa(l);
            layer(m, w, l, qsa, qsa ? qsa_index(l) : gdn_index(l), b.X, m.start, m.T);
        }
    }
    for (auto &m : mbs) {  // complete X: the pre-final-mixer hidden (lm_head and MTP input)
        gpu::hc_combine_T(m.b->X, m.pend, m.b->inj_mlp, m.T, m.s);
        m.pend = nullptr;
        for (const Capture &cp : caps)  // a capture's MTP input: the hidden of its last token
            if (cp.pos > m.start && cp.pos <= m.start + m.T)
                CK(hipMemcpyAsync(rk.snaps[size_t(cp.snap)].pend, m.b->X + size_t(cp.pos - 1 - m.start) * XW, XW * 4,
                                  hipMemcpyDeviceToDevice, m.s));
    }
    // MTP layer over the rows whose next token is known: row t of a
    // micro-batch pairs X[t] with token t + 1. The rows start one early, from
    // the slot's MTP input store (the previous chunk's or micro-batch's last
    // hidden), and the micro-batch's last hidden goes into the store.
    if (mtp_)
        for (auto &m : mbs) {
            PrefillBuf &b = *m.b;
            hipStream_t s = m.s;
            const int o = (m.row0 > 0 || pjob_.mtp_pend) ? 1 : 0;
            const int rows = m.T - 1 + o;
            wait(m, N_LAYERS, 3);  // A's store write and MTP K/V
            if (rows > 0) {
                gpu::mtp_shift(sl.pend, b.X, o, rows, b.Xm, s);
                const float *emb = b.emb + size_t(1 - o) * SH;  // next tokens' embeddings (rank slice)
                gpu::mtp_ssq(b.Xm, emb, rows, b.m_send, s);
                allreduce(m, b.m_send, b.m_red, 4, rows);
                gpu::mtp_norm(b.Xm, emb, b.m_red, rk.mtp_nh, rk.mtp_ne, rows, b.m_xn, s);
                for (int st = 0; st < 5; ++st) {
                    gemm(m, st < HC ? rk.mtp_fc_h : rk.mtp_fc_e, H, SH, b.m_xn + size_t(st) * rows * SH, SH, rows,
                         b.part16, H, true);
                    reduce_scatter16(m, b.part16, b.m_out + size_t(st) * rows * SH, rows);
                }
                gpu::mtp_add(b.m_out, rows, b.Xm, s);
                layer(m, rk.mtp, N_LAYERS, true, N_QSA, b.Xm, m.start - o, rows);
                m.pend = nullptr;  // the MTP output is not needed here
            }
            CK(hipMemcpyAsync(sl.pend, b.X + size_t(m.T - 1) * XW, XW * 4, hipMemcpyDeviceToDevice, s));
            record(m, N_LAYERS, 3);
        }
    // final mixer + lm_head: every row in all-logits mode, else the chunk's last one
    std::vector<std::pair<MB *, int>> outs;  // (micro-batch, first row)
    for (auto &m : mbs) {
        const bool last = &m == &mbs.back();
        if (!pjob_.all_logits && !last) continue;
        const int r0 = pjob_.all_logits ? 0 : m.T - 1, rows = m.T - r0;
        mix(m, rk.final_hc, m.b->X + size_t(r0) * HC * SH, rows, nullptr, nullptr, nullptr);
        gemm(m, rk.lm_head, VOCAB_L, H, m.b->bin, H, rows, m.b->lm_logits, VOCAB_L, false);
        outs.push_back({&m, r0});
    }
    for (auto &m : mbs) {
        *m.seq_base += m.k;
        phase_[size_t(r)] = 2;
        CK(hipStreamSynchronize(m.s));
    }
    for (auto &[m, r0] : outs)
        for (int i = 0; i < m->T - r0; ++i) {
            float *dst = (pjob_.all_logits ? plogits_.data() + size_t(m->row0 + r0 + i) * VOCAB : logits_.data()) +
                         size_t(r) * VOCAB_L;
            CK(hipMemcpy(dst, m->b->lm_logits + size_t(i) * VOCAB_L, VOCAB_L * 4, hipMemcpyDeviceToHost));
        }
    if (profile) {
        const auto t_end = std::chrono::steady_clock::now();
        log("rank %d prefill [%lld, +%d)%s: %.1f ms", r, (long long)start, T, split ? " split" : "",
            std::chrono::duration<double, std::milli>(t_end - t_start).count());
    }
}

const std::vector<float> &Engine::prefill(int slot, const std::vector<int32_t> &tokens, std::vector<float> *all_logits,
                                          const std::function<void(int64_t)> &after_chunk,
                                          const std::vector<Capture> &captures,
                                          const std::vector<EmbedSpan> &embeds) {
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
                b.lm_logits = zeros<float>(size_t(b.C) * VOCAB_L);
            }
        }
    }
    for (size_t c0 = 0; c0 < tokens.size(); c0 += size_t(opt_.prefill_chunk)) {
        const int T = int(std::min<size_t>(size_t(opt_.prefill_chunk), tokens.size() - c0));
        std::vector<int32_t> chunk(tokens.begin() + ptrdiff_t(c0), tokens.begin() + ptrdiff_t(c0) + T);
        pemb_.resize(size_t(T) * H);
        pple_.resize(size_t(T) * H);
        for (int t = 0; t < T; ++t) {
            const int32_t tok = chunk[size_t(t)];
            QW_CHECK(tok < VOCAB, "token out of range");
            if (tok < 0) continue;  // a vision token: its row comes from `embeds`
            for (int j = 0; j < H; ++j) pemb_[size_t(t) * H + j] = bf16_to_f32(embed_[size_t(tok) * H + j]);
        }
        int64_t covered = 0;
        const int64_t c_start = sh.len;
        for (const EmbedSpan &sp : embeds) {
            const int64_t a = std::max(sp.pos, c_start), b = std::min(sp.pos + sp.n, c_start + T);
            for (int64_t p = a; p < b; ++p)
                std::memcpy(&pemb_[size_t(p - c_start) * H], sp.rows + size_t(p - sp.pos) * H, H * 4);
            covered += std::max<int64_t>(0, b - a);
        }
        QW_CHECK(covered == std::count_if(chunk.begin(), chunk.end(), [](int32_t t) { return t < 0; }),
                 "prefill: vision tokens without embeddings (or embeddings over text tokens)");
        auto ids = hasher_.ids_for(sh.hist, chunk);
        std::vector<float> e(H);
        for (int t = 0; t < T; ++t) {
            ple_->gather(ids[size_t(t)], e.data());
            for (int j = 0; j < H; ++j) pple_[size_t(t) * H + j] = f32_to_f16(e[size_t(j)]);
        }
        pjob_ = {sh.len, T, all, slot, sh.reset, mtp_ && sh.pend_valid && sh.len > 0, {}, blend_transfer_};
        for (const Capture &c : captures)
            if (c.pos > sh.len && c.pos <= sh.len + T) pjob_.captures.push_back(c);
        QW_CHECK(int(pjob_.captures.size()) <= MAX_CAPTURES, "prefill: too many captures in one chunk");
        if (all) plogits_.assign(size_t(T) * VOCAB, 0.f);
        job_ = Job::Prefill;
        dispatch();
        sh.reset = false;
        if (all) all_logits->insert(all_logits->end(), plogits_.begin(), plogits_.end());
        sh.hist.insert(sh.hist.end(), chunk.end() - std::min<ptrdiff_t>(T, 32), chunk.end());
        if (sh.hist.size() > 32) sh.hist.erase(sh.hist.begin(), sh.hist.end() - 32);
        sh.len += T;
        sh.pend_valid = mtp_;
        sh.run_first = -1;
        if (after_chunk) after_chunk(sh.len);
    }
    if (all) std::copy(all_logits->end() - VOCAB, all_logits->end(), logits_.begin());
    return logits_;
}

}  // namespace qw
