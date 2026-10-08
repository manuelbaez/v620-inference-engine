// Device memory of one rank: sequence slots (recurrent state and KV), the
// snapshot pool, prefill and batched-decode scratch, streams and handles.
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
    b.cand = zeros<gpu::IdxCand>(size_t(b.Qb) * IDX_TOPK_BLOCKS);
    b.cand_all = zeros<gpu::IdxCand>(size_t(b.Qb) * RANKS * IDX_TOPK_BLOCKS);
    b.lm_logits = zeros<float>(size_t(Engine::MAX_SEGMENTS) * VOCAB_L);  // grown on demand for all-logits
    b.xg = zeros<float>(size_t(Engine::MAX_SEGMENTS) * XW);
    if (mtp) {
        b.Xm = zeros<float>(size_t(C) * XW);
        b.m_emb = zeros<float>(size_t(C) * SH);
        b.m_out = zeros<float>(size_t(C) * 5 * SH);
        b.m_send = zeros<float>(size_t(C) * 4);
        b.m_red = zeros<float>(size_t(C) * 4);
        b.m_xn = zeros<uint16_t>(size_t(C) * 5 * SH);
    }
    b.h_emb = pinned<float>(size_t(C) * SH);
    b.h_ple = pinned<uint16_t>(size_t(C) * H);
}

void alloc_slots(Rank &rk, const EngineOptions &opt, const SpillPool *spill, bool mtp) {
    std::vector<gpu::SlotPtrs> tab(opt.slot_tokens.size());
    for (size_t si = 0; si < opt.slot_tokens.size(); ++si) {
        const size_t cap = size_t(opt.slot_tokens[si]);  // tokens in VRAM
        const size_t spilled = si < opt.slot_spill.size() ? size_t(std::max(0, opt.slot_spill[si])) : 0;
        const size_t logical = cap + spilled;
        Rank::Slot sl;
        sl.S = zeros<float>(SLOT_S);
        sl.ring = zeros<float>(SLOT_RING);
        sl.ple = zeros<float>(SLOT_PLE);
        sl.pend = zeros<float>(XW);
        if (mtp) sl.hist = zeros<float>(size_t(gpu::MTP_HIST) * XW);
        sl.pen = zeros<int32_t>(size_t(2) * VOCAB_L);
        for (int i = 0; i < N_QSA + int(mtp); ++i) {
            sl.K.push_back(zeros<uint16_t>(cap * HEAD_DIM));
            sl.V.push_back(zeros<uint16_t>(cap * HEAD_DIM));
            sl.ck.push_back(zeros<uint16_t>(logical / IDX_RATIO * IDX_DIM + IDX_DIM));
            sl.raw_k.push_back(zeros<float>(size_t(raw_ring_rows(opt)) * IDX_DIM));
        }
        sl.raw_ring = raw_ring_rows(opt);
        if (spilled) {
            sl.vt = int32_t(cap);
            for (int i = 0; i < N_QSA + int(mtp); ++i) {
                const SpillPool::Buf &b = spill->buf(int(si), rk.r, i);
                sl.Kh.push_back(b.K);
                sl.Vh.push_back(b.V);
            }
        }
        gpu::SlotPtrs &t = tab[si];
        std::memset(&t, 0, sizeof t);
        t.vt = sl.vt;
        t.raw_ring = sl.raw_ring;
        for (int i = 0; i < N_QSA + int(mtp); ++i) {
            t.K[i] = sl.K[size_t(i)];
            t.V[i] = sl.V[size_t(i)];
            t.ck[i] = sl.ck[size_t(i)];
            t.raw_k[i] = sl.raw_k[size_t(i)];
            if (spilled) {
                t.Kh[i] = sl.Kh[size_t(i)];
                t.Vh[i] = sl.Vh[size_t(i)];
            }
        }
        t.S = sl.S;
        t.ring = sl.ring;
        t.ple = sl.ple;
        t.mtp_pend = sl.pend;
        t.mtp_hist = sl.hist;
        rk.slots.push_back(sl);
    }
    rk.dtab = upload(tab);
    rk.pack_bytes = size_t(gpu::QSA_LAYERS_MAX) * (size_t(Rank::KV_PACK_TOKENS) * 2 * HEAD_DIM * 2 +
                                                   size_t(Rank::KV_PACK_TOKENS / IDX_RATIO) * IDX_DIM * 2);
    rk.pack = zeros<uint8_t>(rk.pack_bytes);
    std::vector<int32_t *> pens;
    for (const auto &sl : rk.slots) pens.push_back(sl.pen);
    rk.pen_tab = upload(pens);
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
    b.Lpen = zeros<float>(M * VOCAB_L);
    b.srows = zeros<gpu::SampleRow>(M);
    b.h_srows = pinned<gpu::SampleRow>(M);
    b.sout = zeros<gpu::SampleOut>(M);
    b.h_sout = pinned<gpu::SampleOut>(M);
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
        b.h_dstage = pinned<Rank::Batch::DraftStage>(M);
        b.d_last = zeros<int32_t>(M);
        b.dtokens = zeros<int32_t>(M * M);
        b.h_dtokens = pinned<int32_t>(M * M);
        b.amax_all = zeros<float>(size_t(RANKS) * M * 2);
    }
}

}  // namespace

size_t slot_vram_bytes(const EngineOptions &opt, bool mtp) {
    const size_t layers = size_t(N_QSA + int(mtp));
    size_t total = size_t(Engine::SNAPSHOTS) * (SLOT_S + SLOT_RING + SLOT_PLE + XW + RAW_TAIL_FLOATS) * 4;
    for (size_t si = 0; si < opt.slot_tokens.size(); ++si) {
        const size_t vram = size_t(opt.slot_tokens[si]);
        const size_t spilled = si < opt.slot_spill.size() ? size_t(std::max(0, opt.slot_spill[si])) : 0;
        total += layers * (vram * 2 * HEAD_DIM * 2                                        // K, V (fp16)
                           + size_t(raw_ring_rows(opt)) * IDX_DIM * 4                      // the raw keys' ring (fp32)
                           + ((vram + spilled) / IDX_RATIO * IDX_DIM + IDX_DIM) * 2);       // compressed keys, whole length
        total += (SLOT_S + SLOT_RING + SLOT_PLE + XW) * 4 + (mtp ? size_t(gpu::MTP_HIST) * XW * 4 : 0) +
                 size_t(2) * VOCAB_L * 4;
    }
    if (const size_t e = spill_cache_entries(opt)) total += e * (2 * HEAD_DIM * 2 + 8) + size_t(SPILL_MISS_CAP) * 8;
    return total + spill_stage_tokens(opt) * 2 * HEAD_DIM * 2;  // the prefill staging of the spill (K, V)
}

void alloc_rank_buffers(Rank &rk, const EngineOptions &opt, const SpillPool *spill, int max_slot_tokens, bool mtp) {
    {
        size_t free_b = 0, total_b = 0;
        CK(hipMemGetInfo(&free_b, &total_b));
        const size_t need = slot_vram_bytes(opt, mtp);
        QW_CHECK(need + (size_t(1) << 30) <= free_b,
                 "the slots need " + std::to_string(need >> 20) + " MiB of VRAM per card (K, V and raw keys of the resident tokens, "
                 "compressed keys of the whole capacity, recurrent state) and only " + std::to_string(free_b >> 20) +
                 " MiB are free after the weights, with ~1 GiB more needed for the prefill and decode buffers: "
                 "fewer or smaller resident slots (--slots), or a smaller --slot-max-tokens");
    }
    for (auto &sn : rk.snaps) {
        sn.S = zeros<float>(SLOT_S);
        sn.ring = zeros<float>(SLOT_RING);
        sn.ple = zeros<float>(SLOT_PLE);
        sn.pend = zeros<float>(XW);
        sn.raw_tail = zeros<float>(RAW_TAIL_FLOATS);
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
    alloc_slots(rk, opt, spill, mtp);
    if (rk.r == 0 && std::getenv("QW_SPILL_LOCALITY")) {
        int m = 0;
        for (int s : opt.slot_spill) m = std::max(m, s);
        rk.loc_groups = m / IDX_RATIO;
        const size_t n = opt.slot_tokens.size() * size_t(gpu::QSA_LAYERS_MAX) * size_t(rk.loc_groups);
        rk.loc_last = zeros<int32_t>(n);
        CK(hipMemset(rk.loc_last, 0xC0, n * 4));  // very negative: never selected
        rk.loc_run = zeros<int32_t>(n);
        CK(hipMemset(rk.loc_run, 0xC0, n * 4));
        rk.loc_stats = zeros<unsigned long long>(size_t(gpu::QSA_LAYERS_MAX) * gpu::LOC_STATS);
    }
    if (const size_t e = spill_cache_entries(opt)) {
        gpu::SpillCache &c = rk.cache;
        c.entries = uint32_t(e);
        c.tags = zeros<unsigned long long>(e);
        c.K = zeros<uint16_t>(e * HEAD_DIM);
        c.V = zeros<uint16_t>(e * HEAD_DIM);
        rk.cache_epoch = upload(std::vector<uint32_t>(opt.slot_tokens.size(), 1u));
        c.epoch = rk.cache_epoch;
        c.miss_cap = SPILL_MISS_CAP;
        c.miss = zeros<uint2>(SPILL_MISS_CAP);
        c.miss_n = zeros<uint32_t>(1);
        if (std::getenv("QW_SPILL_CACHE_STATS")) c.stats = zeros<unsigned long long>(2);
    }
    if (const size_t n = spill_stage_tokens(opt)) {
        rk.stage_tokens = n;
        rk.stage_k = zeros<uint16_t>(n * HEAD_DIM);
        rk.stage_v = zeros<uint16_t>(n * HEAD_DIM);
        CK(hipStreamCreateWithFlags(&rk.s3, hipStreamNonBlocking));
        for (hipEvent_t *e : {&rk.stage_ready, &rk.stage_free, &rk.stage_done[0], &rk.stage_done[1]})
            CK(hipEventCreateWithFlags(e, hipEventDisableTiming));
    }
    alloc_batch(rk.bt, max_slot_tokens, mtp);
    CK(hipDeviceSynchronize());
}

}  // namespace qw
