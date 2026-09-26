// Batched decode: graph recording of a step over up to 16 rows (the model,
// or the MTP head), per-rank execution, and the host side of decode().
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
#include "kernels/gdn.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemv.hpp"
#include "kernels/hc.hpp"
#include "kernels/logits.hpp"
#include "kernels/moe.hpp"
#include "kernels/mtp.hpp"
#include "kernels/ple.hpp"
#include "kernels/qsa.hpp"
#include "kernels/sampling.hpp"

namespace qw {

static_assert(Engine::MAX_BATCH_ROWS == gpu::MAX_ROWS, "batch row limit");

// Enqueues one batched step of M rows on rank rk (graph-capturable: rows,
// positions and slots are read from device memory). Returns the number of
// collectives, whose sequence offsets are 1..n.
//   kind 0: the model over the rows (logits, per-row log-sum-exp, and each
//           run's last hidden into its slot's MTP input store);
//   kind 1: the same, also keeping every row's GDN states (for accept());
//   kind 2: the MTP head over the rows (hidden gathered into bt.m_hid,
//           next-token embeddings in bt.emb): per-row argmax of its logits.
uint32_t Engine::record_batch(Rank &rk, int M, int kind) {
    const int r = rk.r;
    hipStream_t s = rk.s;
    auto &b = rk.bt;
    uint32_t k = 0;
    auto allreduce = [&](const float *src, float *dst, int n) { comm_->allreduce(r, src, dst, n, ++k, s, M); };
    auto reduce_scatter = [&](const float *src, float *dst) { comm_->reduce_scatter(r, src, dst, SH, ++k, s, M); };
    auto allgather = [&](const void *src, void *dst, size_t bytes) { comm_->allgather(r, src, dst, bytes, ++k, s, M); };
    const gpu::Rows rows{b.row_slot, b.row_pos, b.row_first, M};
    static const int moe_pair_max = std::getenv("QW_MOE_PAIR_MAX") ? std::atoi(std::getenv("QW_MOE_PAIR_MAX")) : 16;

    auto mix = [&](const HcW &w, const float *pend, const float *pend_inj, float *inj_out) {
        gpu::hc_pre_B(b.X, pend, pend_inj, w.w1, b.y, b.hc_send, M, s);
        // rows of `down` are grouped by stream: row n reads stream n / 324 of y
        if (w.down8.w)
            gpu::gemv_rows_i8(w.down8.w, w.down8.s, b.y, HC * SH, b.hc_send + HC, HC_SEND, HC * HC_DOWN, SH, M, HC_DOWN,
                              false, s);
        else
            gpu::gemv_rows(w.down, b.y, HC * SH, b.hc_send + HC, HC_SEND, HC * HC_DOWN, SH, M, HC_DOWN, false, s);
        allreduce(b.hc_send, b.hc_red, HC_SEND);
        gpu::hc_up_mix_B(b.hc_red, w.up, b.y, b.bin_local, inj_out, M, s);
        allgather(b.bin_local, b.bin, SH * 2);
    };
    // fp16 weights, or their int8 copy when loaded (QW_INT8_DENSE=1)
    auto gemv = [&](const uint16_t *W, const Q8 &W8, int N, int K, const uint16_t *x, float *y) {
        if (W8.w)
            gpu::gemv_rows_i8(W8.w, W8.s, x, K, y, N, N, K, M, 0, false, s);
        else
            gpu::gemv_rows(W, x, K, y, N, N, K, M, 0, false, s);
    };

    // One decoder layer (after the PLE step, if any) on b.X; returns the
    // pending MoE output, combined into X by the next mix.
    auto layer = [&](LayerW &w, bool qsa, int idx, const float *pend, float *save) -> const float * {
        mix(w.attn_hc, pend, b.inj_mlp, b.inj_attn);
        if (qsa) {
            gemv(w.qsa_proj, w.qsa_proj8, QSA_PROJ_L, H, b.bin, b.proj);
            gpu::qsa_prep_B(b.proj, w.qn, w.kn, w.iqn, rk.dtab, idx, rows, b.q16, b.gate, b.iq, s);
            gpu::qsa_compress_B(w.ikn, rk.dtab, idx, rows, s);
            gpu::qsa_attend_B(b.q16, b.gate, b.iq, rk.dtab, idx, rows, b.scores, b.score_ld, b.lists, b.counts,
                              b.att_partial, b.act16, s);
            gemv(w.qsa_o, w.qsa_o8, H, QH_L * HEAD_DIM, b.act16, b.partial);
        } else {
            gemv(w.gdn_proj, w.gdn_proj8, GDN_PROJ_L, H, b.bin, b.proj);
            gpu::gdn_conv_B(b.proj, w.gdn_conv, w.gdn_A, w.gdn_dt, rk.dtab, idx, rows, b.qkv, b.gb, s);
            gpu::gdn_ring_B(b.proj, rk.dtab, idx, rows, s);
            gpu::gdn_scan_B(b.qkv, b.gb, rk.dtab, idx, b.run_start, b.run_len, b.run_slot, M, b.o_raw, save, s);
            gpu::gdn_norm_B(b.o_raw, b.proj, w.gdn_norm, M, b.act16, s);
            gemv(w.gdn_out, w.gdn_out8, H, GDN_VL * GDN_DIM, b.act16, b.partial);
        }
        reduce_scatter(b.partial, b.out_attn);

        mix(w.mlp_hc, b.out_attn, b.inj_attn, b.inj_mlp);
        // (A fork/join of the shared expert onto a second stream inside the graph
        // was measured 12% slower: cross-stream graph edges cost more than they hide.)
        gemv(w.sh_gu, w.sh_gu8, 2 * SFFN_L + 1, H, b.bin, b.shp);
        gpu::swiglu_T(b.shp, 2 * SFFN_L + 1, SFFN_L, M, b.sh_h, s);
        gemv(w.sh_down, w.sh_down8, H, SFFN_L, b.sh_h, b.sh_out);
        gemv(w.router, Q8{}, N_EXPERTS, H, b.bin, b.router);
        gpu::moe_route_B(b.router, M, r * EXP_L, b.ecounts, b.offsets, b.tok_w, b.tok_slot, b.pair_tok, s);
        if (M <= moe_pair_max)
            gpu::moe_experts_P(w.eg, w.egs, w.eu, w.eus, w.ed, w.eds, b.offsets, b.pair_tok, b.bin, M, b.moe_h, b.yp,
                               s);
        else
            gpu::moe_experts_T(w.eg, w.egs, w.eu, w.eus, w.ed, w.eds, b.ecounts, b.offsets, b.pair_tok, b.bin, M,
                               b.tiles, b.moe_h, b.yp, s);
        gpu::moe_combine_B(b.yp, b.tok_slot, b.tok_w, b.sh_out, M, b.partial, s);
        reduce_scatter(b.partial, b.out_moe);
        return b.out_moe;
    };

    if (kind == 2) {
        // X = fc_hidden(gnorm(hidden)) per stream + fc_embedding(gnorm(emb)), row-parallel fc
        gpu::mtp_ssq(b.m_hid, b.emb, M, b.m_send, s);
        comm_->allreduce(r, b.m_send, b.m_red, 4, ++k, s, M);
        gpu::mtp_norm(b.m_hid, b.emb, b.m_red, rk.mtp_nh, rk.mtp_ne, M, b.m_xn, s);
        for (int st = 0; st < 5; ++st) {
            const Q8 &f8 = st < HC ? rk.mtp_fc_h8 : rk.mtp_fc_e8;
            if (f8.w)
                gpu::gemv_rows_i8(f8.w, f8.s, b.m_xn + size_t(st) * M * SH, SH, b.m_part + size_t(st) * M * H, H, H,
                                  SH, M, 0, false, s);
            else
                gpu::gemv_rows(st < HC ? rk.mtp_fc_h : rk.mtp_fc_e, b.m_xn + size_t(st) * M * SH, SH,
                               b.m_part + size_t(st) * M * H, H, H, SH, M, 0, false, s);
        }
        comm_->reduce_scatter(r, b.m_part, b.m_out, SH, ++k, s, 5 * M);
        gpu::mtp_add(b.m_out, M, b.X, s);
        const float *pend = layer(rk.mtp, true, N_QSA, nullptr, nullptr);
        mix(rk.mtp_final, pend, b.inj_mlp, nullptr);
        gemv(rk.lm_head, rk.lm_head8, VOCAB_L, H, b.bin, b.logits);
        gpu::row_argmax(b.logits, VOCAB_L, M, b.amax, s);
        return k;
    }

    gpu::embed_T(b.X, b.emb, M, s);
    const float *pend = nullptr;
    for (int l = 0; l < debug_layers_; ++l) {
        if (l == PLE_LAYER) {
            if (pend) gpu::hc_combine_T(b.X, pend, b.inj_mlp, M, s);
            pend = nullptr;
            gemv(rk.ple_kv, rk.ple_kv8, PLE_KV, H, b.ple_e, b.kv);
            gpu::ple_stats_B(b.kv, b.X, rk.ple_nk, rk.ple_nq, M, b.ple_send, s);
            allreduce(b.ple_send, b.ple_red, 16);
            gpu::ple_apply_B(b.kv, b.ple_red, rk.ple_nc, rk.ple_conv, rk.dtab, rows, b.ncbuf, b.X, s);
        }
        const bool qsa = is_qsa(l);
        float *save = kind == 1 && !qsa ? b.gdn_save + size_t(gdn_index(l)) * gpu::MAX_ROWS * GDN_STATE : nullptr;
        pend = layer(rk.L[size_t(l)], qsa, qsa ? qsa_index(l) : gdn_index(l), pend, save);
    }
    mix(rk.final_hc, pend, b.inj_mlp, nullptr);  // also completes X: the MTP input
    if (mtp_) {
        gpu::save_last_rows(b.X, rk.dtab, b.run_start, b.run_len, b.run_slot, M, s);
        gpu::save_rows_hist(b.X, rk.dtab, rows, s);
    }
    gemv(rk.lm_head, rk.lm_head8, VOCAB_L, H, b.bin, b.logits);
    gpu::row_lse(b.logits, VOCAB_L, M, b.lse, s);
    return k;
}

// Captures (first use) and launches the graph of `kind` for M rows.
static void launch_graph(hipStream_t s, hipGraphExec_t &g, uint32_t &n_coll, const std::function<uint32_t()> &record) {
    static const bool no_graph = std::getenv("QW_NOGRAPH") != nullptr;
    if (no_graph) {
        n_coll = record();
        return;
    }
    if (!g) {
        hipGraph_t graph;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
        n_coll = record();
        CK(hipStreamEndCapture(s, &graph));
        CK(hipGraphInstantiate(&g, graph, nullptr, nullptr, 0));
        CK(hipGraphDestroy(graph));
    }
    CK(hipGraphLaunch(g, s));
}

namespace {
// QW_TRACE: host-side phases of decode steps, per row count
struct DecodeTrace {
    bool on = std::getenv("QW_TRACE") != nullptr;
    double prep[17] = {}, graph[17] = {}, copy[17] = {}, total[17] = {};
    int n[17] = {};
    static double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }
    void report(int M) {
        if (++n[M] % 50) return;
        std::printf("decode M=%d: host prep %.2f, graph %.2f (rank 0), copy %.2f (rank 0), total %.2f ms\n", M,
                    prep[M] / n[M], graph[M] / n[M], copy[M] / n[M], total[M] / n[M]);
        std::fflush(stdout);
    }
};
DecodeTrace g_dtrace;
}  // namespace

void Engine::run_decode_rank(Rank &rk) {
    const int r = rk.r, M = djob_.M;
    hipStream_t s = rk.s;
    auto &b = rk.bt;
    apply_penalty_updates(rk);
    for (int sl : djob_.reset_slots) {
        Rank::Slot &x = rk.slots[size_t(sl)];
        CK(hipMemsetAsync(x.S, 0, SLOT_S * 4, s));
        CK(hipMemsetAsync(x.ring, 0, SLOT_RING * 4, s));
        CK(hipMemsetAsync(x.ple, 0, SLOT_PLE * 4, s));
    }
    // per-step inputs through pinned staging (rewritten only after the previous step synced)
    constexpr int MR = gpu::MAX_ROWS;
    std::memcpy(b.h_i32, djob_.i32.data(), 5 * MR * 4);
    std::memcpy(b.h_pos, djob_.pos.data(), MR * 8);
    for (int i = 0; i < M; ++i) std::memcpy(b.h_emb + size_t(i) * SH, demb_.data() + size_t(i) * H + r * SH, SH * 4);
    std::memcpy(b.h_ple, dple_.data(), size_t(M) * H * 2);
    CK(hipMemcpyAsync(b.row_slot, b.h_i32, 5 * MR * 4, hipMemcpyHostToDevice, s));
    CK(hipMemcpyAsync(b.row_pos, b.h_pos, MR * 8, hipMemcpyHostToDevice, s));
    CK(hipMemcpyAsync(b.emb, b.h_emb, size_t(M) * SH * 4, hipMemcpyHostToDevice, s));
    CK(hipMemcpyAsync(b.ple_e, b.h_ple, size_t(M) * H * 2, hipMemcpyHostToDevice, s));
    comm_->set_base(r, rk.seq_base, s);
    const int kind = djob_.save ? 1 : 0;
    auto &g = rk.graphs[size_t(kind)][size_t(M)];
    auto &nc = rk.n_coll[size_t(kind)][size_t(M)];
    const auto t0 = std::chrono::steady_clock::now();
    launch_graph(s, g, nc, [&] { return record_batch(rk, M, kind); });
    rk.seq_base += nc;
    if (g_dtrace.on && r == 0) {
        CK(hipStreamSynchronize(s));
        g_dtrace.graph[M] += DecodeTrace::ms(t0, std::chrono::steady_clock::now());
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (djob_.sample) {
        std::memcpy(b.h_srows, djob_.srows.data(), size_t(M) * sizeof(gpu::SampleRow));
        CK(hipMemcpyAsync(b.srows, b.h_srows, size_t(M) * sizeof(gpu::SampleRow), hipMemcpyHostToDevice, s));
        gpu::sample_penalize(b.logits, b.Lpen, b.srows, rk.pen_tab, VOCAB_L, r * VOCAB_L, M, s);
        gpu::sample_select(b.Lpen, b.logits, b.srows, VOCAB_L, r * VOCAB_L, M, b.sout, s);
        CK(hipMemcpyAsync(b.h_sout, b.sout, size_t(M) * sizeof(gpu::SampleOut), hipMemcpyDeviceToHost, s));
    }
    if (djob_.copy_logits)
        CK(hipMemcpyAsync(b.h_logits, b.logits, size_t(M) * VOCAB_L * 4, hipMemcpyDeviceToHost, s));
    CK(hipMemcpyAsync(b.h_lse, b.lse, size_t(M) * 2 * 4, hipMemcpyDeviceToHost, s));
    phase_[size_t(rk.r)] = 2;
    CK(hipStreamSynchronize(s));
    std::memcpy(dlse_parts_.data() + size_t(r) * gpu::MAX_ROWS * 2, b.h_lse, size_t(M) * 2 * 4);
    if (djob_.sample)
        for (int i = 0; i < M; ++i) sout_[size_t(i) * RANKS + size_t(r)] = b.h_sout[i];
    if (djob_.copy_logits)
        for (int i = 0; i < M; ++i)
            std::memcpy(dlogits_.data() + size_t(i) * VOCAB + size_t(r) * VOCAB_L, b.h_logits + size_t(i) * VOCAB_L,
                        VOCAB_L * 4);
    if (g_dtrace.on && r == 0) g_dtrace.copy[M] += DecodeTrace::ms(t1, std::chrono::steady_clock::now());
}

void Engine::run_mtp_rank(Rank &rk) {
    const int r = rk.r, M = djob_.M;
    hipStream_t s = rk.s;
    auto &b = rk.bt;
    constexpr int MR = gpu::MAX_ROWS;
    std::memcpy(b.h_i32, djob_.i32.data(), 5 * MR * 4);
    std::memcpy(b.h_pos, djob_.pos.data(), MR * 8);
    for (int i = 0; i < M; ++i) {
        std::memcpy(b.h_emb + size_t(i) * SH, demb_.data() + size_t(i) * H + r * SH, SH * 4);
        const int src = djob_.src[size_t(i)];
        const int64_t hp = djob_.src_hist.empty() ? -1 : djob_.src_hist[size_t(i)];
        b.h_src[i] = src >= 0   ? b.X + size_t(src) * XW
                     : hp >= 0  ? rk.slots[size_t(-1 - src)].hist + size_t(hp % gpu::MTP_HIST) * XW
                                : rk.slots[size_t(-1 - src)].pend;
    }
    CK(hipMemcpyAsync(b.row_slot, b.h_i32, 5 * MR * 4, hipMemcpyHostToDevice, s));
    CK(hipMemcpyAsync(b.row_pos, b.h_pos, MR * 8, hipMemcpyHostToDevice, s));
    CK(hipMemcpyAsync(b.emb, b.h_emb, size_t(M) * SH * 4, hipMemcpyHostToDevice, s));
    CK(hipMemcpyAsync(b.src_ptrs, b.h_src, size_t(M) * sizeof(float *), hipMemcpyHostToDevice, s));
    gpu::gather_rows(b.src_ptrs, M, XW, b.m_hid, s);
    comm_->set_base(r, rk.seq_base, s);
    auto &g = rk.graphs[2][size_t(M)];
    auto &nc = rk.n_coll[2][size_t(M)];
    launch_graph(s, g, nc, [&] { return record_batch(rk, M, 2); });
    rk.seq_base += nc;
    CK(hipMemcpyAsync(b.h_amax, b.amax, size_t(M) * 2 * 4, hipMemcpyDeviceToHost, s));
    phase_[size_t(rk.r)] = 2;
    CK(hipStreamSynchronize(s));
    std::memcpy(damax_parts_.data() + size_t(r) * MR * 2, b.h_amax, size_t(M) * 2 * 4);
}

// A draft job: every step's MTP pass enqueued back to back. Each step's
// drafted tokens are picked on the GPUs (the ranks' argmax parts all-gathered)
// and their embeddings fetched from the mapped host table, so the next step
// needs nothing from the host; one sync at the end. The collectives' sequence
// base advances on the device between steps.
void Engine::run_draft_rank(Rank &rk) {
    const int r = rk.r;
    hipStream_t s = rk.s;
    auto &b = rk.bt;
    constexpr int MR = gpu::MAX_ROWS;
    const int steps = int(dsteps_.size()), nreq = int(dsteps_[0].last.size());
    comm_->set_base(r, rk.seq_base, s);
    for (int st = 0; st < steps; ++st) {
        const DraftStep &d = dsteps_[size_t(st)];
        const int M = d.M;
        Rank::Batch::DraftStage &h = b.h_dstage[st];
        std::memcpy(h.i32, d.i32.data(), sizeof h.i32);
        std::memcpy(h.pos, d.pos.data(), sizeof h.pos);
        std::memcpy(h.last, d.last.data(), size_t(nreq) * 4);
        for (int i = 0; i < M; ++i) {
            const int src = d.src[size_t(i)];
            h.src[i] = src >= 0 ? b.X + size_t(src) * XW : rk.slots[size_t(-1 - src)].pend;
        }
        CK(hipMemcpyAsync(b.row_slot, h.i32, sizeof h.i32, hipMemcpyHostToDevice, s));
        CK(hipMemcpyAsync(b.row_pos, h.pos, sizeof h.pos, hipMemcpyHostToDevice, s));
        CK(hipMemcpyAsync(b.src_ptrs, h.src, size_t(M) * sizeof(float *), hipMemcpyHostToDevice, s));
        CK(hipMemcpyAsync(b.d_last, h.last, size_t(nreq) * 4, hipMemcpyHostToDevice, s));
        if (st == 0) {  // later steps' embeddings come from draft_pick
            for (int i = 0; i < M; ++i)
                std::memcpy(b.h_emb + size_t(i) * SH, demb_.data() + size_t(i) * H + r * SH, SH * 4);
            CK(hipMemcpyAsync(b.emb, b.h_emb, size_t(M) * SH * 4, hipMemcpyHostToDevice, s));
        }
        gpu::gather_rows(b.src_ptrs, M, XW, b.m_hid, s);
        auto &g = rk.graphs[2][size_t(M)];
        auto &nc = rk.n_coll[2][size_t(M)];
        launch_graph(s, g, nc, [&] { return record_batch(rk, M, 2); });
        comm_->allgather(r, b.amax, b.amax_all, MR * 2 * 4, nc + 1, s);
        gpu::draft_pick(b.amax_all, MR * 2, b.d_last, nreq, r, b.dtokens + size_t(st) * MR, rk.embed_dev,
                        st + 1 < steps ? b.emb : nullptr, s);
        comm_->bump_base(r, nc + 1, s);
        rk.seq_base += nc + 1;
    }
    CK(hipMemcpyAsync(b.h_dtokens, b.dtokens, size_t(steps) * MR * 4, hipMemcpyDeviceToHost, s));
    phase_[size_t(rk.r)] = 2;
    CK(hipStreamSynchronize(s));
    if (r == 0) dtok_.assign(b.h_dtokens, b.h_dtokens + size_t(steps) * MR);
}

// Fills the row tables (slot, first row of its run, runs) and embeddings of
// a batch job; pos[i] is row i's position.
void Engine::stage_rows(const std::vector<Row> &rows, const std::vector<int64_t> &pos, bool embeddings) {
    const int M = int(rows.size());
    QW_CHECK(M >= 1 && M <= gpu::MAX_ROWS, "batch: 1..16 rows");
    constexpr int MR = gpu::MAX_ROWS;
    djob_.M = M;
    djob_.i32.assign(5 * MR, 0);
    djob_.pos.assign(MR, 0);
    if (embeddings) demb_.resize(size_t(M) * H);
    int n_runs = 0;
    for (int i = 0; i < M; ++i) {
        const Row &row = rows[size_t(i)];
        QW_CHECK(row.slot >= 0 && row.slot < num_slots(), "batch: bad slot");
        QW_CHECK(row.token >= 0 && row.token < VOCAB, "batch: token out of range");
        if (i == 0 || rows[size_t(i - 1)].slot != row.slot) {
            for (int j = 0; j < i; ++j)
                QW_CHECK(rows[size_t(j)].slot != row.slot, "batch: rows of a slot must be consecutive");
            djob_.i32[size_t(2 * MR + n_runs)] = i;         // run_start
            djob_.i32[size_t(4 * MR + n_runs)] = row.slot;  // run_slot
            ++n_runs;
        }
        ++djob_.i32[size_t(3 * MR + n_runs - 1)];                            // run_len
        djob_.i32[size_t(i)] = row.slot;                                     // row_slot
        djob_.i32[size_t(MR + i)] = djob_.i32[size_t(2 * MR + n_runs - 1)];  // row_first
        djob_.pos[size_t(i)] = pos[size_t(i)];
        QW_CHECK(pos[size_t(i)] >= 0 && pos[size_t(i)] < slot_capacity(row.slot), "batch: slot is full");
        if (embeddings)
            for (int j = 0; j < H; ++j) demb_[size_t(i) * H + j] = bf16_to_f32(embed_[size_t(row.token) * H + j]);
    }
}

const std::vector<float> &Engine::decode(const std::vector<Row> &rows) {
    return decode(rows, nullptr);
}

const std::vector<float> &Engine::decode(const std::vector<Row> &rows, const SampleSpec *spec) {
    const auto t_start = std::chrono::steady_clock::now();
    const int M = int(rows.size());
    constexpr int MR = gpu::MAX_ROWS;
    std::vector<int64_t> pos(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) {
        int first = i;
        while (first > 0 && rows[size_t(first - 1)].slot == rows[size_t(i)].slot) --first;
        QW_CHECK(rows[size_t(i)].slot >= 0 && rows[size_t(i)].slot < num_slots(), "decode: bad slot");
        pos[size_t(i)] = slots_[size_t(rows[size_t(i)].slot)].len + (i - first);
    }
    stage_rows(rows, pos);
    djob_.reset_slots.clear();
    djob_.save = false;
    dple_.resize(size_t(M) * H);
    std::vector<float> e(H);
    for (int i = 0; i < M; ++i) {
        const Row &row = rows[size_t(i)];
        SlotHost &sh = slots_[size_t(row.slot)];
        const int first = djob_.i32[size_t(MR + i)];
        if (i == first && sh.reset) {
            djob_.reset_slots.push_back(row.slot);
            sh.reset = false;
        }
        if (i > first && mtp_) djob_.save = true;
        // n-gram context: the slot's last tokens plus earlier rows of this run
        std::vector<int32_t> ctx(sh.hist.end() - std::min<ptrdiff_t>(ptrdiff_t(sh.hist.size()), 2), sh.hist.end());
        for (int j = first; j < i; ++j) ctx.push_back(rows[size_t(j)].token);
        auto ids = hasher_.ids_for(ctx, {row.token});
        ple_->gather(ids[0], e.data());
        for (int j = 0; j < H; ++j) dple_[size_t(i) * H + j] = f32_to_f16(e[size_t(j)]);
    }
    djob_.sample = spec != nullptr;
    djob_.copy_logits = !spec || spec->logits;
    if (spec) {
        QW_CHECK(int(spec->rows.size()) == M, "decode: one sampling row per decode row");
        djob_.srows = spec->rows;
        for (int i = 0; i < M; ++i) {
            gpu::SampleRow &sr = djob_.srows[size_t(i)];
            sr.slot = rows[size_t(i)].slot;
            sr.first = djob_.i32[size_t(MR + i)];
            sr.token = rows[size_t(i)].token;
        }
        sout_.resize(size_t(M) * RANKS);
    }
    if (djob_.copy_logits) dlogits_.resize(size_t(M) * VOCAB);
    dlogits_valid_ = djob_.copy_logits;
    dlogits_on_gpu_ = true;
    const auto t_disp = std::chrono::steady_clock::now();
    if (g_dtrace.on) g_dtrace.prep[M] += DecodeTrace::ms(t_start, t_disp);
    job_ = Job::Decode;
    dispatch();
    for (auto &ps : pen_) {  // applied by the job
        ps.reset = false;
        ps.tok.clear();
        ps.gen.clear();
    }
    dlse_.resize(size_t(M));
    for (int i = 0; i < M; ++i) {
        float mx = -INFINITY;
        for (int r = 0; r < RANKS; ++r) mx = std::max(mx, dlse_parts_[(size_t(r) * MR + i) * 2]);
        double z = 0;
        for (int r = 0; r < RANKS; ++r) {
            const float *pr = &dlse_parts_[(size_t(r) * MR + i) * 2];
            z += double(pr[1]) * std::exp(double(pr[0] - mx));
        }
        dlse_[size_t(i)] = mx + float(std::log(z));
    }
    for (auto &sh : slots_) sh.run_first = -1;
    for (int i = 0; i < M; ++i) {
        SlotHost &sh = slots_[size_t(rows[size_t(i)].slot)];
        if (i == 0 || rows[size_t(i - 1)].slot != rows[size_t(i)].slot) {
            sh.run_first = i;
            sh.run_len = 0;
        }
        ++sh.run_len;
        sh.len += 1;
        sh.hist.push_back(rows[size_t(i)].token);
        if (sh.hist.size() > 32) sh.hist.erase(sh.hist.begin());
        sh.pend_valid = mtp_;
    }
    if (g_dtrace.on) {
        g_dtrace.total[M] += DecodeTrace::ms(t_start, std::chrono::steady_clock::now());
        g_dtrace.report(M);
    }
    return dlogits_;
}

}  // namespace qw
