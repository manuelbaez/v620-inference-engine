// Per-GPU ("rank") state of the engine: weights, sequence slots, the
// recurrent-state snapshot pool, prefill and decode scratch, captured graphs.
// Internal to src/engine.
#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <cstdint>
#include <vector>

#include "core/safetensors.hpp"
#include "core/shard.hpp"
#include "core/threadpool.hpp"
#include "engine/engine.hpp"
#include "engine/spill.hpp"
#include "kernels/sampling_types.hpp"
#include "kernels/types.hpp"
#include "vision/vision_encoder.hpp"

namespace qw {

using namespace cfg;

// Per-sequence recurrent state sizes (floats).
constexpr size_t SLOT_S = size_t(N_GDN) * GDN_VL * GDN_DIM * GDN_DIM;  // floats
constexpr size_t SLOT_RING = size_t(N_GDN) * GDN_RING * GDN_QKV_L;
constexpr size_t SLOT_PLE = size_t(PLE_RING) * HC * SH;

// ---------------------------------------------------------------- weights
// An int8 copy of a dense matrix for the decode GEMVs (QW_INT8_DENSE=1):
// W[n][k] = w[n][k] * s[n]. Null when off; prefill keeps the fp16 original.
struct Q8 {
    int8_t *w = nullptr;
    float *s = nullptr;
};

struct HcW {
    float *w1;            // [HC][SH] 1 + hc_norm
    uint16_t *down;       // [HC][324][SH]      decode layout (per-stream groups)
    uint16_t *down_flat;  // [328][HC*SH]       prefill layout (one GEMM over all streams)
    uint16_t *up;         // [SH][HC][320]
    Q8 down8;
};

// Prefill scratch for one rank, sized for opt.prefill_chunk tokens.
struct PrefillBuf {
    int C = 0, Qb = gpu::PREFILL_QB, nb_pad = 0;
    float *X, *ssq, *ssq_red, *d, *d_red, *out_attn, *out_moe, *inj_attn, *inj_mlp, *emb;
    uint16_t *xn, *u, *g, *bin_local, *bin, *P, *act, *part16, *ple_e, *kv;
    float *qkv, *gb, *o_raw, *logits, *tok_w, *shp, *ple_send, *ple_red, *ncbuf;
    int32_t *tok_e, *counts, *offsets, *fill, *tok_slot, *pair_tok, *tiles;
    int8_t *q8;
    float *q8s;
    uint16_t *moe_h, *yp, *sh_h, *sh_out;
    uint16_t *q16, *qgate, *iq16, *sc16;
    float *scores, *att_partial;
    int32_t *lists, *lcounts;
    gpu::IdxCand *cand, *cand_all;  // [Qb][512] this rank's, [Qb][RANKS][512] every rank's (sharded selection)
    float *lm_logits;  // [C][VOCAB_L] fp32 (all-logits mode)
    // MTP pass over the chunk: its rows' hidden [C][XW], norm/fc work rows [5][C]
    float *Xm = nullptr, *m_out = nullptr, *m_send = nullptr, *m_red = nullptr;
    float *m_emb = nullptr;  // [C][SH]: the MTP rows' next-token embeddings, gathered (batched prefill)
    float *xg = nullptr;     // [MAX_SEGMENTS][XW]: segments' last rows, gathered for lm_head
    uint16_t *m_xn = nullptr;
    float *h_emb = nullptr;     // pinned [C][SH]
    uint16_t *h_ple = nullptr;  // pinned [C][H]
};

struct LayerW {
    HcW attn_hc, mlp_hc;
    // GDN
    uint16_t *gdn_proj = nullptr;  // [4120][H]
    float *gdn_conv = nullptr;     // [2560][4]
    float *gdn_A = nullptr, *gdn_dt = nullptr, *gdn_norm = nullptr;
    uint16_t *gdn_out = nullptr;  // [H][1536]
    // QSA
    uint16_t *qsa_proj = nullptr;  // [4224][H]
    float *qn, *kn, *iqn, *ikn;
    uint16_t *qsa_o = nullptr;  // [H][1536]
    // MoE
    uint16_t *router;           // [512][H]
    uint32_t *eg, *eu, *ed;     // packed int4, local experts
    uint16_t *egs, *eus, *eds;  // fp16 scales
    uint16_t *sh_gu;            // [321][H]: gate 160 | up 160 | shared_expert_gate
    uint16_t *sh_down;          // [H][160]
    Q8 qsa_proj8, qsa_o8, gdn_proj8, gdn_out8, sh_gu8, sh_down8;
};

struct Rank {
    int r = 0, dev = 0;
    hipStream_t s = nullptr;
    rocblas_handle blas = nullptr, blas2 = nullptr;
    hipStream_t s2 = nullptr;  // second prefill micro-batch
    PrefillBuf pb[2];
    uint32_t seq_base2 = 0;  // sequence base on the second comm channel
    // micro-batch B waits on A per layer: [layer][0 gdn ring | 1 gdn scan | 2 qsa kv | 3 ple ring]
    // (layer N_LAYERS is the MTP pass)
    std::array<hipEvent_t, (N_LAYERS + 1) * 4> ev{};
    std::vector<LayerW> L;
    HcW final_hc{};
    uint16_t *lm_head = nullptr;  // [VOCAB_L][H]
    const uint16_t *embed_dev = nullptr;  // the engine's pinned token embeddings, bf16 [VOCAB][H], mapped
    Q8 lm_head8, ple_kv8, mtp_fc_h8, mtp_fc_e8;
    // MTP head (QSA layer index N_QSA in the slots' KV)
    LayerW mtp{};
    HcW mtp_final{};
    uint16_t *mtp_fc_h = nullptr, *mtp_fc_e = nullptr;  // [H][SH] row-parallel column slices
    float *mtp_nh = nullptr, *mtp_ne = nullptr;         // 1 + w: [HC][SH] and [SH] slices
    // PLE
    uint16_t *ple_kv = nullptr;  // [3200][H]
    float *ple_nk, *ple_nq, *ple_nc, *ple_conv;
    // per-sequence slots, and a device table of their pointers
    struct Slot {
        float *S = nullptr, *ring = nullptr, *ple = nullptr;  // GDN state, GDN conv ring, PLE conv ring
        float *pend = nullptr;                                // [XW] MTP input store
        float *hist = nullptr;                                // [MTP_HIST][XW] decoded rows' MTP inputs
        int32_t *pen = nullptr;  // [2][VOCAB_L] token counts for the penalties: prompt + generated, generated
        std::vector<uint16_t *> K, V, ck;
        std::vector<float *> raw_k;  // rings of raw_ring rows (kernels/types.hpp)
        int32_t raw_ring = 0;
        // KV spill: positions >= vt are in pinned host memory shared with the rank's KV group (empty without spill)
        std::vector<uint16_t *> Kh, Vh;
        int32_t vt = gpu::NO_SPILL;
        gpu::KvSplit kv(size_t layer) const {
            if (Kh.empty()) return {K[layer], V[layer], raw_k[layer], nullptr, nullptr, gpu::NO_SPILL, raw_ring};
            return {K[layer], V[layer], raw_k[layer], Kh[layer], Vh[layer], vt, raw_ring};
        }
    };
    std::vector<Slot> slots;
    // KV spill staging for prefill (EngineOptions::spill_stage): K and V of up to stage_tokens spilled positions of one
    // QSA layer, indexed like a slot's host part (position vt + i at row i, plus a segment's offset); a copy stream,
    // and events: the copy in done, each micro-batch's attention done, the write-back done (the buffer is free)
    uint16_t *stage_k = nullptr, *stage_v = nullptr;
    size_t stage_tokens = 0;
    hipStream_t s3 = nullptr;
    hipEvent_t stage_ready = nullptr, stage_free = nullptr;
    std::array<hipEvent_t, 2> stage_done{};
    gpu::SlotPtrs *dtab = nullptr;
    // Routed-expert work of this rank: (token, expert) pairs it computed, [0] in prefill, [1] in decode, counted by
    // the routing kernels on the device and copied to the pinned host copy now and then (Engine::moe_use).
    unsigned long long *moe_use = nullptr, *moe_use_host = nullptr;
    unsigned moe_use_tick = 0;
    // export_kv packs a block of up to KV_PACK_TOKENS positions here in the block store's layout, then copies it once
    static constexpr int KV_PACK_TOKENS = 256;
    uint8_t *pack = nullptr;
    size_t pack_bytes = 0;
    int32_t **pen_tab = nullptr;  // device [slots]: each slot's pen
    // recurrent-state snapshots (pool, any slot)
    struct Snap {
        float *S = nullptr, *ring = nullptr, *ple = nullptr, *pend = nullptr;
        float *raw_tail = nullptr;  // [QSA_LAYERS_MAX][RAW_TAIL][IDX_DIM]: the last positions' raw indexer keys
    };
    std::array<Snap, Engine::SNAPSHOTS> snaps{};
    std::unique_ptr<VisionEncoder> vit;  // this card's copy of the vision tower (null: text only)
    // CacheBlend experiment (blend.hip), allocated on first use: a chunk's GDN
    // transfer [N_GDN][12][128][128] and a scratch copy of a slot's state
    float *transfer = nullptr, *blend_tmp = nullptr;
    int32_t *rope3 = nullptr;  // multimodal RoPE experiment: a chunk's (t, h, w) positions
    struct CalibH {
        float *H = nullptr;  // fp32 [K][K], sum over rows of x x^T
        int K = 0;
        double rows = 0;
    };
    std::map<std::string, CalibH> calib;  // GPTQ calibration (Engine::calib_begin)
    int rope3_cap = 0;
    // batched decode scratch, MAX_ROWS rows
    struct Batch {
        float *X, *emb, *hc_send, *hc_red, *proj, *gate, *iq, *scores, *att_partial, *partial, *out_attn, *out_moe;
        float *inj_attn, *inj_mlp, *qkv, *gb, *o_raw, *router, *tok_w, *shp, *sh_out, *kv, *ple_send, *ple_red;
        float *ncbuf, *logits;
        uint16_t *y, *bin_local, *bin, *q16, *act16, *moe_h, *yp, *sh_h, *ple_e;
        int32_t *lists, *counts, *tok_e, *ecounts, *offsets, *fill, *tok_slot, *pair_tok, *tiles;
        int32_t *row_slot, *row_first, *run_start, *run_len, *run_slot;
        int64_t *row_pos;
        int score_ld = 0;
        // speculative decoding: per-row GDN states [N_GDN][MAX_ROWS][GDN_STATE]; MTP work buffers
        float *gdn_save = nullptr, *m_hid = nullptr, *m_send, *m_red, *m_part, *m_out, *amax;
        uint16_t *m_xn;
        const float **src_ptrs = nullptr;  // device [MAX_ROWS]
        const float **h_src = nullptr;     // pinned
        float *h_amax = nullptr;
        // pinned staging: [row_slot | row_first | run_start | run_len | run_slot] int32, row_pos int64
        int32_t *h_i32 = nullptr;
        int64_t *h_pos = nullptr;
        float *lse = nullptr;  // [M][2] per-row {max, sumexp} of this rank's vocab shard
        float *h_emb = nullptr, *h_logits = nullptr, *h_lse = nullptr;
        uint16_t *h_ple = nullptr;
        // GPU sampling (gpu_sample.hip): penalized logits [MAX_ROWS][VOCAB_L], row table and results
        float *Lpen = nullptr;
        gpu::SampleRow *srows = nullptr, *h_srows = nullptr;
        gpu::SampleOut *sout = nullptr, *h_sout = nullptr;
        // chained draft steps (Engine::draft): per-step row tables staged from pinned memory, every
        // rank's argmax parts, and the drafted tokens [step][MAX_ROWS]
        struct DraftStage {
            int32_t i32[5 * gpu::MAX_ROWS];
            int64_t pos[gpu::MAX_ROWS];
            const float *src[gpu::MAX_ROWS];
            int32_t last[gpu::MAX_ROWS];
        };
        DraftStage *h_dstage = nullptr;  // pinned [MAX_ROWS] (one per step)
        int32_t *d_last = nullptr, *dtokens = nullptr, *h_dtokens = nullptr;
        float *amax_all = nullptr;  // [RANKS][MAX_ROWS][2]
        // penalty count updates: tokens and generated flags (device copies of the engine's staging)
        int32_t *pen_tok = nullptr;
        uint8_t *pen_gen = nullptr;
        size_t pen_cap = 0;
    } bt;
    // QW_SPILL_LOCALITY experiment (rank 0 only): last selecting position per slot, layer and spilled group; counters
    int32_t *loc_last = nullptr, *loc_run = nullptr;
    // decode-time cache of spilled rows (EngineOptions::spill_cache_mb; tags null: off) and its per-slot epochs
    gpu::SpillCache cache{};
    uint32_t *cache_epoch = nullptr;
    unsigned long long *loc_stats = nullptr;
    int loc_groups = 0;
    // captured graphs per row count: [kind][M], kind 0 decode, 1 decode + GDN state save, 2 MTP
    std::array<std::array<hipGraphExec_t, gpu::MAX_ROWS + 1>, 3> graphs{};
    std::array<std::array<uint32_t, gpu::MAX_ROWS + 1>, 3> n_coll{};
    uint32_t seq_base = 0;
};

// Loads rank rk.r's shard of every weight (weights.hip).
// cache_dir: the converted-weights cache (EngineOptions::weight_cache_dir), or "".
void load_rank_weights(Rank &rk, const SafeTensors &st, ThreadPool &pool, bool mtp, const std::string &cache_dir);
// Rows of a slot's ring of raw indexer keys: a prefill chunk, the 3 positions before it, a snapshot's tail.
inline int raw_ring_rows(const EngineOptions &opt) {
    return opt.prefill_chunk + gpu::RAW_TAIL + 64;
}
constexpr size_t RAW_TAIL_FLOATS = size_t(gpu::QSA_LAYERS_MAX) * gpu::RAW_TAIL * IDX_DIM;
// The raw indexer keys of positions [n - RAW_TAIL, n) (those >= 0) of a slot's first `layers` QSA layers, between its
// rings and a tail [QSA_LAYERS_MAX][RAW_TAIL][IDX_DIM] in position order (row RAW_TAIL - 1 is position n - 1).
void raw_tail_copy(const Rank::Slot &sl, int layers, int64_t n, float *tail, bool to_tail, hipMemcpyKind kind,
                   hipStream_t s);
// Positions the prefill staging buffer of the spill holds: the largest slot's spill (0: no spill or staging off).
inline size_t spill_cache_entries(const EngineOptions &opt) {  // 0: no cache
    bool any = false;
    for (int s : opt.slot_spill) any |= s > 0;
    if (!any || opt.spill_cache_mb <= 0) return 0;
    return (size_t(opt.spill_cache_mb) << 20) / (2 * HEAD_DIM * 2 + 8);
}
constexpr uint32_t SPILL_MISS_CAP = uint32_t(gpu::MAX_ROWS) * gpu::LIST_W * gpu::QSA_LAYERS_MAX;
inline size_t spill_stage_tokens(const EngineOptions &opt) {
    if (!opt.spill_stage) return 0;
    int m = 0;
    for (int s : opt.slot_spill) m = std::max(m, s);
    return size_t(m);
}
// Bytes of VRAM per rank that the slots (KV, compressed keys, recurrent state) and the snapshot pool take.
size_t slot_vram_bytes(const EngineOptions &opt, bool mtp);
// Allocates the rank's slots, snapshot pool, scratch, streams and rocBLAS
// handles (buffers.hip). The rank's device must be current.
void alloc_rank_buffers(Rank &rk, const EngineOptions &opt, const SpillPool *spill, int max_slot_tokens, bool mtp);

}  // namespace qw
