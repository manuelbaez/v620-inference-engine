// Device memory of one rank: sequence slots (recurrent state and KV), the
// snapshot pool, prefill and batched-decode scratch, streams and handles.
#include <algorithm>
#include <cstring>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/rank.hpp"

namespace qw {

namespace {

// Prefill scratch for one micro-batch of up to C tokens.
void alloc_prefill(PrefillBuf &b, int C, int max_slot_tokens, bool mtp) {
    b.C = C;
    b.nb_pad = (max_slot_tokens / IDX_RATIO + 7) / 8 * 8;
    b.X = zeros<float>(size_t(C) * HC * SH);
    b.ssq = zeros<float>(size_t(C) * HC);
    b.ssq_red = zeros<float>(size_t(C) * HC);
    b.d = zeros<float>(size_t(C) * gpu::D_PAD);
    b.d_red = zeros<float>(size_t(C) * gpu::D_PAD);
    b.out_attn = zeros<float>(size_t(C) * SH);
    b.out_moe = zeros<float>(size_t(C) * SH);
    b.inj_attn = zeros<float>(size_t(C) * HC);
    b.inj_mlp = zeros<float>(size_t(C) * HC);
    b.emb = zeros<float>(size_t(C) * SH);
    b.xn = zeros<uint16_t>(size_t(C) * HC * SH);
    b.u = zeros<uint16_t>(size_t(C) * HC_RANK);
    b.g = zeros<uint16_t>(size_t(C) * HC * SH);
    b.bin_local = zeros<uint16_t>(size_t(C) * SH);
    b.bin = zeros<uint16_t>(size_t(C) * H);
    b.P = zeros<uint16_t>(size_t(C) * std::max(GDN_PROJ_L, QSA_PROJ_L));
    b.act = zeros<uint16_t>(size_t(C) * QH_L * HEAD_DIM);
    b.part16 = zeros<uint16_t>(size_t(C) * H);
    b.ple_e = zeros<uint16_t>(size_t(C) * H);
    b.kv = zeros<uint16_t>(size_t(C) * PLE_KV);
    b.qkv = zeros<float>(size_t(C) * GDN_QKV_L);
    b.gb = zeros<float>(size_t(C) * GDN_VL * 2);
    b.o_raw = zeros<float>(size_t(C) * GDN_VL * GDN_DIM);
    b.logits = zeros<float>(size_t(C) * N_EXPERTS);
    b.tok_w = zeros<float>(size_t(C) * TOP_K);
    b.shp = zeros<float>(size_t(C) * (2 * SFFN_L + 1));
    b.ple_send = zeros<float>(size_t(C) * 16);
    b.ple_red = zeros<float>(size_t(C) * 16);
    b.ncbuf = zeros<float>(size_t(C) * HC * SH);
    b.tok_e = zeros<int32_t>(size_t(C) * TOP_K);
    b.counts = zeros<int32_t>(EXP_L);
    b.offsets = zeros<int32_t>(EXP_L + 1);
    b.fill = zeros<int32_t>(EXP_L);
    b.tok_slot = zeros<int32_t>(size_t(C) * TOP_K);
    b.pair_tok = zeros<int32_t>(size_t(C) * TOP_K);
    b.tiles = zeros<int32_t>(2 * (size_t(C) * TOP_K / 32 + EXP_L) + 1);
    b.q8 = zeros<int8_t>(size_t(C) * TOP_K * FFN);
    b.q8s = zeros<float>(size_t(C) * TOP_K * (FFN / QGROUP));
    b.moe_h = zeros<uint16_t>(size_t(C) * TOP_K * FFN);
    b.yp = zeros<uint16_t>(size_t(C) * TOP_K * H);
    b.sh_h = zeros<uint16_t>(size_t(C) * SFFN_L);
    b.sh_out = zeros<uint16_t>(size_t(C) * H);
    b.q16 = zeros<uint16_t>(size_t(C) * QH_L * HEAD_DIM);
    b.qgate = zeros<uint16_t>(size_t(C) * QH_L * HEAD_DIM);
    b.iq16 = zeros<uint16_t>(size_t(C) * IDX_HEADS * IDX_DIM);
    b.sc16 = zeros<uint16_t>(size_t(b.Qb) * IDX_HEADS * b.nb_pad);
    b.scores = zeros<float>(size_t(b.Qb) * b.nb_pad);
    b.att_partial = zeros<float>(size_t(b.Qb) * 33 * QH_L * (HEAD_DIM + 2));
    b.lists = zeros<int32_t>(size_t(b.Qb) * gpu::LIST_W);
    b.lcounts = zeros<int32_t>(b.Qb);
    b.lm_logits = zeros<float>(VOCAB_L);  // grown on demand for all-logits
    if (mtp) {
        b.Xm = zeros<float>(size_t(C) * XW);
        b.m_out = zeros<float>(size_t(C) * 5 * SH);
        b.m_send = zeros<float>(size_t(C) * 4);
        b.m_red = zeros<float>(size_t(C) * 4);
        b.m_xn = zeros<uint16_t>(size_t(C) * 5 * SH);
    }
    b.h_emb = pinned<float>(size_t(C) * SH);
    b.h_ple = pinned<uint16_t>(size_t(C) * H);
}

void alloc_slots(Rank &rk, const EngineOptions &opt, bool mtp) {
    std::vector<gpu::SlotPtrs> tab(opt.slot_tokens.size());
    for (size_t si = 0; si < opt.slot_tokens.size(); ++si) {
        const size_t cap = size_t(opt.slot_tokens[si]);
        Rank::Slot sl;
        sl.S = zeros<float>(SLOT_S);
        sl.ring = zeros<float>(SLOT_RING);
        sl.ple = zeros<float>(SLOT_PLE);
        sl.pend = zeros<float>(XW);
        if (mtp) sl.hist = zeros<float>(size_t(gpu::MTP_HIST) * XW);
        for (int i = 0; i < N_QSA + int(mtp); ++i) {
            sl.K.push_back(zeros<uint16_t>(cap * HEAD_DIM));
            sl.V.push_back(zeros<uint16_t>(cap * HEAD_DIM));
            sl.ck.push_back(zeros<uint16_t>(cap / IDX_RATIO * IDX_DIM + IDX_DIM));
            sl.raw_k.push_back(zeros<float>(cap * IDX_DIM));
        }
        gpu::SlotPtrs &t = tab[si];
        std::memset(&t, 0, sizeof t);
        for (int i = 0; i < N_QSA + int(mtp); ++i) {
            t.K[i] = sl.K[size_t(i)];
            t.V[i] = sl.V[size_t(i)];
            t.ck[i] = sl.ck[size_t(i)];
            t.raw_k[i] = sl.raw_k[size_t(i)];
        }
        t.S = sl.S;
        t.ring = sl.ring;
        t.ple = sl.ple;
        t.mtp_pend = sl.pend;
        t.mtp_hist = sl.hist;
        rk.slots.push_back(sl);
    }
    rk.dtab = upload(tab);
}

// Batched-decode scratch for MAX_ROWS rows.
void alloc_batch(Rank::Batch &b, int max_slot_tokens, bool mtp) {
    constexpr size_t M = gpu::MAX_ROWS;
    b.score_ld = (max_slot_tokens / IDX_RATIO + 7) / 8 * 8;
    b.X = zeros<float>(M * HC * SH);
    b.emb = zeros<float>(M * SH);
    b.hc_send = zeros<float>(M * HC_SEND);
    b.hc_red = zeros<float>(M * HC_SEND);
    b.proj = zeros<float>(M * std::max({GDN_PROJ_L, QSA_PROJ_L, PLE_KV}));
    b.gate = zeros<float>(M * QH_L * HEAD_DIM);
    b.iq = zeros<float>(M * IDX_HEADS * IDX_DIM);
    b.scores = zeros<float>(M * size_t(b.score_ld));
    b.att_partial = zeros<float>(M * 33 * QH_L * (HEAD_DIM + 2));
    b.partial = zeros<float>(M * H);
    b.out_attn = zeros<float>(M * SH);
    b.out_moe = zeros<float>(M * SH);
    b.inj_attn = zeros<float>(M * HC);
    b.inj_mlp = zeros<float>(M * HC);
    b.qkv = zeros<float>(M * GDN_QKV_L);
    b.gb = zeros<float>(M * GDN_VL * 2);
    b.o_raw = zeros<float>(M * GDN_VL * GDN_DIM);
    b.router = zeros<float>(M * N_EXPERTS);
    b.tok_w = zeros<float>(M * TOP_K);
    b.shp = zeros<float>(M * (2 * SFFN_L + 1));
    b.sh_out = zeros<float>(M * H);
    b.kv = b.proj;  // PLE projection reuses the projection buffer
    b.ple_send = zeros<float>(M * 16);
    b.ple_red = zeros<float>(M * 16);
    b.ncbuf = zeros<float>(M * HC * SH);
    b.logits = zeros<float>(M * VOCAB_L);
    b.lse = zeros<float>(M * 2);
    b.y = zeros<uint16_t>(M * HC * SH);
    b.bin_local = zeros<uint16_t>(M * SH);
    b.bin = zeros<uint16_t>(M * H);
    b.q16 = zeros<uint16_t>(M * QH_L * HEAD_DIM);
    b.act16 = zeros<uint16_t>(M * QH_L * HEAD_DIM);
    b.moe_h = zeros<uint16_t>(M * TOP_K * FFN);
    b.yp = zeros<uint16_t>(M * TOP_K * H);
    b.sh_h = zeros<uint16_t>(M * SFFN_L);
    b.ple_e = zeros<uint16_t>(M * H);
    b.lists = zeros<int32_t>(M * gpu::LIST_W);
    b.counts = zeros<int32_t>(M);
    b.tok_e = zeros<int32_t>(M * TOP_K);
    b.ecounts = zeros<int32_t>(EXP_L);
    b.offsets = zeros<int32_t>(EXP_L + 1);
    b.fill = zeros<int32_t>(EXP_L);
    b.tok_slot = zeros<int32_t>(M * TOP_K);
    b.pair_tok = zeros<int32_t>(M * TOP_K);
    b.tiles = zeros<int32_t>(2 * (M * TOP_K / 32 + EXP_L) + 1);
    int32_t *rows_i32 = zeros<int32_t>(5 * M);
    b.row_slot = rows_i32;
    b.row_first = rows_i32 + M;
    b.run_start = rows_i32 + 2 * M;
    b.run_len = rows_i32 + 3 * M;
    b.run_slot = rows_i32 + 4 * M;
    b.row_pos = zeros<int64_t>(M);
    b.h_i32 = pinned<int32_t>(5 * M);
    b.h_pos = pinned<int64_t>(M);
    b.h_emb = pinned<float>(M * SH);
    b.h_ple = pinned<uint16_t>(M * H);
    b.h_logits = pinned<float>(M * VOCAB_L);
    b.h_lse = pinned<float>(M * 2);
    if (mtp) {
        b.gdn_save = zeros<float>(size_t(N_GDN) * M * GDN_STATE);
        b.m_hid = zeros<float>(M * XW);
        b.m_send = zeros<float>(M * 4);
        b.m_red = zeros<float>(M * 4);
        b.m_part = zeros<float>(M * 5 * H);
        b.m_out = zeros<float>(M * 5 * SH);
        b.m_xn = zeros<uint16_t>(M * 5 * SH);
        b.amax = zeros<float>(M * 2);
        b.src_ptrs = zeros<const float *>(M);
        b.h_src = pinned<const float *>(M);
        b.h_amax = pinned<float>(M * 2);
    }
}

}  // namespace

void alloc_rank_buffers(Rank &rk, const EngineOptions &opt, int max_slot_tokens, bool mtp) {
    for (auto &sn : rk.snaps) {
        sn.S = zeros<float>(SLOT_S);
        sn.ring = zeros<float>(SLOT_RING);
        sn.ple = zeros<float>(SLOT_PLE);
        sn.pend = zeros<float>(XW);
    }
    // prefill: two micro-batches of half a chunk each, on two streams
    QW_CHECK(opt.prefill_chunk >= 512 && opt.prefill_chunk % 2 == 0, "prefill_chunk must be even and >= 512");
    alloc_prefill(rk.pb[0], opt.prefill_chunk / 2, max_slot_tokens, mtp);
    alloc_prefill(rk.pb[1], opt.prefill_chunk / 2, max_slot_tokens, mtp);
    CK(hipStreamCreateWithFlags(&rk.s2, hipStreamNonBlocking));
    for (auto &ev : rk.ev) CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    if (rocblas_create_handle(&rk.blas2) != rocblas_status_success) fail("rocblas_create_handle");
    rocblas_set_stream(rk.blas2, rk.s2);
    if (rocblas_create_handle(&rk.blas) != rocblas_status_success) fail("rocblas_create_handle");
    rocblas_set_stream(rk.blas, rk.s);
    alloc_slots(rk, opt, mtp);
    alloc_batch(rk.bt, max_slot_tokens, mtp);
    CK(hipDeviceSynchronize());
}

}  // namespace qw
