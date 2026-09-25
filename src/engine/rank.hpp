// Per-GPU ("rank") state of the engine: weights, sequence slots, the
// recurrent-state snapshot pool, prefill and decode scratch, captured graphs.
// Internal to src/engine.
#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <array>
#include <cstdint>
#include <vector>

#include "core/safetensors.hpp"
#include "core/shard.hpp"
#include "core/threadpool.hpp"
#include "engine/engine.hpp"
#include "kernels/types.hpp"

namespace qw {

using namespace cfg;

// Per-sequence recurrent state sizes (floats).
constexpr size_t SLOT_S = size_t(N_GDN) * GDN_VL * GDN_DIM * GDN_DIM;  // floats
constexpr size_t SLOT_RING = size_t(N_GDN) * GDN_RING * GDN_QKV_L;
constexpr size_t SLOT_PLE = size_t(PLE_RING) * HC * SH;

// ---------------------------------------------------------------- weights
struct HcW {
    float *w1;            // [HC][SH] 1 + hc_norm
    uint16_t *down;       // [HC][324][SH]      decode layout (per-stream groups)
    uint16_t *down_flat;  // [328][HC*SH]       prefill layout (one GEMM over all streams)
    uint16_t *up;         // [SH][HC][320]
};

// Prefill scratch for one rank, sized for opt.prefill_chunk tokens.
struct PrefillBuf {
    int C = 0, Qb = 256, nb_pad = 0;
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
    float *lm_logits;  // [C][VOCAB_L] fp32 (all-logits mode)
    // MTP pass over the chunk: its rows' hidden [C][XW], norm/fc work rows [5][C]
    float *Xm = nullptr, *m_out = nullptr, *m_send = nullptr, *m_red = nullptr;
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
        std::vector<uint16_t *> K, V, ck;
        std::vector<float *> raw_k;
    };
    std::vector<Slot> slots;
    gpu::SlotPtrs *dtab = nullptr;
    // recurrent-state snapshots (pool, any slot)
    struct Snap {
        float *S = nullptr, *ring = nullptr, *ple = nullptr, *pend = nullptr;
    };
    std::array<Snap, Engine::SNAPSHOTS> snaps{};
    // CacheBlend experiment (blend.hip), allocated on first use: a chunk's GDN
    // transfer [N_GDN][12][128][128] and a scratch copy of a slot's state
    float *transfer = nullptr, *blend_tmp = nullptr;
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
    } bt;
    // captured graphs per row count: [kind][M], kind 0 decode, 1 decode + GDN state save, 2 MTP
    std::array<std::array<hipGraphExec_t, gpu::MAX_ROWS + 1>, 3> graphs{};
    std::array<std::array<uint32_t, gpu::MAX_ROWS + 1>, 3> n_coll{};
    uint32_t seq_base = 0;
};

// Loads rank rk.r's shard of every weight (weights.hip).
void load_rank_weights(Rank &rk, const SafeTensors &st, ThreadPool &pool, bool mtp);
// Allocates the rank's slots, snapshot pool, scratch, streams and rocBLAS
// handles (buffers.hip). The rank's device must be current.
void alloc_rank_buffers(Rank &rk, const EngineOptions &opt, int max_slot_tokens, bool mtp);

}  // namespace qw
