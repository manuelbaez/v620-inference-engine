#include "engine/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "core/common.hpp"
#include "core/config.hpp"
#include "core/threadpool.hpp"
#include "kernels/decode.hpp"
#include "kernels/model_ops.hpp"
#include "kernels/prefill_ops.hpp"
#include "kernels/batch_ops.hpp"

namespace qw {

using namespace cfg;
using gpu::HC_DOWN;
using gpu::HC_SEND;
using gpu::SH;

namespace {

void ck(hipError_t e, const char *what) {
    if (e != hipSuccess) fail(std::string("hip: ") + what + ": " + hipGetErrorString(e));
}
#define CK(x) ck((x), #x)

constexpr int GDN_VL = GDN_V_HEADS / RANKS;          // 12 local v-heads
constexpr int GDN_KL = GDN_QK_HEADS / RANKS;         // 4 local k-heads
constexpr int GDN_QKV_L = 2 * GDN_KL * GDN_DIM + GDN_VL * GDN_DIM;  // 2560
constexpr int GDN_PROJ_L = GDN_QKV_L + GDN_VL * GDN_DIM + 2 * GDN_VL;  // 4120
constexpr int QH_L = Q_HEADS / RANKS;                 // 6
constexpr int QSA_PROJ_L = QH_L * 2 * HEAD_DIM + 2 * HEAD_DIM + (IDX_HEADS + 1) * IDX_DIM;  // 4224
constexpr int EXP_L = N_EXPERTS / RANKS;              // 128
constexpr int SFFN_L = FFN / RANKS;                   // 160 shared-expert columns
constexpr int VOCAB_L = VOCAB / RANKS;                // 62080
constexpr int PLE_KV = HC * SH + SH;                  // 3200
constexpr int XW = HC * SH;                           // local multi-stream row
constexpr size_t GDN_STATE = size_t(GDN_VL) * GDN_DIM * GDN_DIM;  // per layer

std::string lp(int layer) { return "model.language_model.layers." + std::to_string(layer) + "."; }

constexpr size_t SLOT_S = size_t(N_GDN) * GDN_VL * GDN_DIM * GDN_DIM;  // floats
constexpr size_t SLOT_RING = size_t(N_GDN) * GDN_RING * GDN_QKV_L;
constexpr size_t SLOT_PLE = size_t(PLE_RING) * HC * SH;

// bf16 -> fp16 with saturation (counted, reported once at load).
std::atomic<int64_t> g_saturated{0};
inline uint16_t bf16_to_f16(uint16_t b) {
    float f = bf16_to_f32(b);
    if (std::fabs(f) > 65504.f) {
        ++g_saturated;
        f = f > 0 ? 65504.f : -65504.f;
    }
    return f32_to_f16(f);
}

template <typename T>
T *upload(const std::vector<T> &v) {
    T *d;
    CK(hipMalloc(&d, v.size() * sizeof(T)));
    CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <typename T>
T *zeros(size_t n) {
    T *d;
    CK(hipMalloc(&d, n * sizeof(T)));
    CK(hipMemset(d, 0, n * sizeof(T)));
    return d;
}

// Appends rows [r0, r0+n) x cols [c0, c0+nc) of a bf16 matrix as fp16.
void add_rows(std::vector<uint16_t> &dst, const TensorView &t, int64_t r0, int64_t n, int64_t c0 = 0,
              int64_t nc = -1) {
    QW_CHECK(t.dtype == DType::BF16 && t.shape.size() >= 2, t.name + ": expected bf16 matrix");
    const int64_t cols = t.numel() / t.dim(0);
    if (nc < 0) nc = cols - c0;
    QW_CHECK(r0 + n <= t.dim(0) && c0 + nc <= cols, t.name + ": slice out of range");
    size_t o = dst.size();
    dst.resize(o + size_t(n * nc));
    for (int64_t r = 0; r < n; ++r) {
        const uint16_t *src = t.u16() + (r0 + r) * cols + c0;
        for (int64_t c = 0; c < nc; ++c) dst[o + size_t(r * nc + c)] = bf16_to_f16(src[c]);
    }
}

std::vector<float> f32_of(const TensorView &t) {
    QW_CHECK(t.dtype == DType::BF16, t.name + ": expected bf16");
    std::vector<float> v(size_t(t.numel()));
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16_to_f32(t.u16()[i]);
    return v;
}

// 1 + w for the rank's columns of a [HC*H] Gemma norm weight, as [HC][SH].
std::vector<float> shard_w1(const TensorView &t, int r) {
    auto w = f32_of(t);
    std::vector<float> o(size_t(HC) * SH);
    for (int s = 0; s < HC; ++s)
        for (int j = 0; j < SH; ++j) o[size_t(s * SH + j)] = 1.f + w[size_t(s * H + r * SH + j)];
    return o;
}

std::vector<float> plus_one(std::vector<float> v) {
    for (auto &x : v) x += 1.f;
    return v;
}

}  // namespace

// ---------------------------------------------------------------- rank state
struct HcW {
    float *w1;           // [HC][SH] 1 + hc_norm
    uint16_t *down;      // [HC][324][SH]      decode layout (per-stream groups)
    uint16_t *down_flat; // [328][HC*SH]       prefill layout (one GEMM over all streams)
    uint16_t *up;        // [SH][HC][320]
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
    float *lm_logits;        // [C][VOCAB_L] fp32 (all-logits mode)
    // MTP pass over the chunk: its rows' hidden [C][XW], norm/fc work rows [5][C]
    float *Xm = nullptr, *m_out = nullptr, *m_send = nullptr, *m_red = nullptr;
    uint16_t *m_xn = nullptr;
    float *h_emb = nullptr;  // pinned [C][SH]
    uint16_t *h_ple = nullptr;  // pinned [C][H]
};

struct LayerW {
    HcW attn_hc, mlp_hc;
    // GDN
    uint16_t *gdn_proj = nullptr;  // [4120][H]
    float *gdn_conv = nullptr;     // [2560][4]
    float *gdn_A = nullptr, *gdn_dt = nullptr, *gdn_norm = nullptr;
    uint16_t *gdn_out = nullptr;   // [H][1536]
    // QSA
    uint16_t *qsa_proj = nullptr;  // [4224][H]
    float *qn, *kn, *iqn, *ikn;
    uint16_t *qsa_o = nullptr;     // [H][1536]
    // MoE
    uint16_t *router;              // [512][H]
    uint32_t *eg, *eu, *ed;        // packed int4, local experts
    uint16_t *egs, *eus, *eds;     // fp16 scales
    uint16_t *sh_gu;               // [321][H]: gate 160 | up 160 | shared_expert_gate
    uint16_t *sh_down;             // [H][160]
};

struct Engine::Rank {
    int r = 0, dev = 0;
    hipStream_t s = nullptr;
    rocblas_handle blas = nullptr, blas2 = nullptr;
    hipStream_t s2 = nullptr;        // second prefill micro-batch
    PrefillBuf pb[2];
    uint32_t seq_base2 = 0;          // sequence base on the second comm channel
    // micro-batch B waits on A per layer: [layer][0 gdn ring | 1 gdn scan | 2 qsa kv | 3 ple ring]
    // (layer N_LAYERS is the MTP pass)
    std::array<hipEvent_t, (N_LAYERS + 1) * 4> ev{};
    std::vector<LayerW> L;
    HcW final_hc{};
    uint16_t *lm_head = nullptr;   // [VOCAB_L][H]
    // MTP head (QSA layer index N_QSA in the slots' KV)
    LayerW mtp{};
    HcW mtp_final{};
    uint16_t *mtp_fc_h = nullptr, *mtp_fc_e = nullptr;  // [H][SH] row-parallel column slices
    float *mtp_nh = nullptr, *mtp_ne = nullptr;         // 1 + w: [HC][SH] and [SH] slices
    // PLE
    uint16_t *ple_kv = nullptr;    // [3200][H]
    float *ple_nk, *ple_nq, *ple_nc, *ple_conv;
    // per-sequence slots, and a device table of their pointers
    struct Slot {
        float *S = nullptr, *ring = nullptr, *ple = nullptr;  // GDN state, GDN conv ring, PLE conv ring
        float *pend = nullptr;                                // [XW] MTP input store
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

Engine::Engine(const EngineOptions &opt) : opt_(opt) {
    check_config(opt_.model_dir);
    st_.add_index(opt_.model_dir);
    ple_ = std::make_unique<PleTable>(opt_.ple_dir);
    const size_t slot = std::max<size_t>(64 * 1024, size_t(opt_.prefill_chunk / 2) * gpu::D_PAD * 4 + 4096);
    comm_ = std::make_unique<Comm>(opt_.devices, slot);
    comm2_ = std::make_unique<Comm>(opt_.devices, slot);
    QW_CHECK(!opt_.slot_tokens.empty() && opt_.slot_tokens.size() <= size_t(MAX_SLOTS), "1..8 slots");
    for (int c : opt_.slot_tokens) QW_CHECK(c > 0 && c % 256 == 0, "slot sizes must be positive multiples of 256");
    slots_.resize(opt_.slot_tokens.size());
    logits_.resize(VOCAB);
    mtp_ = opt_.mtp && !std::getenv("QW_NO_MTP") && st_.has("mtp.fc_embedding.weight");

    ThreadPool pool(opt_.load_threads * RANKS);
    for (int r = 0; r < RANKS; ++r) ranks_.push_back(std::make_unique<Rank>());

    auto load_rank = [&](int r) {
        Rank &rk = *ranks_[size_t(r)];
        rk.r = r;
        rk.dev = opt_.devices[size_t(r)];
        CK(hipSetDevice(rk.dev));
        CK(hipStreamCreateWithFlags(&rk.s, hipStreamNonBlocking));

        auto hc = [&](const std::string &p, bool inject) {
            HcW w;
            w.w1 = upload(shard_w1(st_.get(p + "hc_norm.weight"), r));
            const TensorView &down = st_.get(p + "input_mix_weight_down.weight", DType::BF16, {HC_RANK, HC * H});
            std::vector<uint16_t> d;
            d.reserve(size_t(HC) * HC_DOWN * SH);
            for (int s = 0; s < HC; ++s) {
                add_rows(d, down, 0, HC_RANK, s * H + r * SH, SH);
                if (inject)
                    add_rows(d, st_.get(p + "block_inject_weight.weight", DType::BF16, {HC, HC * H}), 0, HC,
                             s * H + r * SH, SH);
                else
                    d.resize(d.size() + size_t(HC) * SH, 0);
            }
            w.down = upload(d);
            std::vector<uint16_t> df(size_t(gpu::D_PAD) * HC * SH, 0);
            for (int s2 = 0; s2 < HC; ++s2)
                for (int o = 0; o < HC_DOWN; ++o)
                    std::copy(d.begin() + (size_t(s2) * HC_DOWN + o) * SH, d.begin() + (size_t(s2) * HC_DOWN + o + 1) * SH,
                              df.begin() + size_t(o) * HC * SH + size_t(s2) * SH);
            w.down_flat = upload(df);
            const TensorView &up = st_.get(p + "input_mix_weight_up.weight", DType::BF16, {HC * H, HC_RANK});
            std::vector<uint16_t> u;
            u.reserve(size_t(SH) * HC * HC_RANK);
            for (int j = 0; j < SH; ++j)
                for (int s = 0; s < HC; ++s) add_rows(u, up, s * H + r * SH + j, 1);
            w.up = upload(u);
            return w;
        };

        // One decoder layer. The MTP layer's routed experts are bf16 (fused
        // gate_up): they are quantized here to the same int4 g128 layout (they
        // only draft tokens, which the full model verifies).
        auto load_layer = [&](LayerW &w, const std::string &p, bool qsa, bool bf16_experts) {
            w.attn_hc = hc(p + "attn_hyper_connection.", true);
            w.mlp_hc = hc(p + "mlp_hyper_connection.", true);
            if (qsa) {
                const std::string a = p + "self_attn.";
                std::vector<uint16_t> m;
                const TensorView &q = st_.get(a + "q_proj.weight", DType::BF16, {Q_HEADS * 2 * HEAD_DIM, H});
                add_rows(m, q, int64_t(r) * QH_L * 2 * HEAD_DIM, QH_L * 2 * HEAD_DIM);
                const int kvh = r / (RANKS / KV_HEADS);
                add_rows(m, st_.get(a + "k_proj.weight"), kvh * HEAD_DIM, HEAD_DIM);
                add_rows(m, st_.get(a + "v_proj.weight"), kvh * HEAD_DIM, HEAD_DIM);
                add_rows(m, st_.get(a + "indexer.index_qk_proj.weight"), 0, (IDX_HEADS + 1) * IDX_DIM);
                QW_CHECK(m.size() == size_t(QSA_PROJ_L) * H, "qsa proj size");
                w.qsa_proj = upload(m);
                w.qn = upload(plus_one(f32_of(st_.get(a + "q_norm.weight"))));
                w.kn = upload(plus_one(f32_of(st_.get(a + "k_norm.weight"))));
                w.iqn = upload(plus_one(f32_of(st_.get(a + "indexer.q_layernorm.weight"))));
                w.ikn = upload(plus_one(f32_of(st_.get(a + "indexer.k_layernorm.weight"))));
                std::vector<uint16_t> o;
                add_rows(o, st_.get(a + "o_proj.weight", DType::BF16, {H, Q_HEADS * HEAD_DIM}), 0, H,
                         int64_t(r) * QH_L * HEAD_DIM, QH_L * HEAD_DIM);
                w.qsa_o = upload(o);
            } else {
                const std::string a = p + "linear_attn.";
                const TensorView &qkv = st_.get(a + "in_proj_qkv.weight", DType::BF16, {GDN_QKV, H});
                std::vector<uint16_t> m;
                add_rows(m, qkv, int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
                add_rows(m, qkv, GDN_KEY + int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
                add_rows(m, qkv, 2 * GDN_KEY + int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
                add_rows(m, st_.get(a + "in_proj_z.weight"), int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
                add_rows(m, st_.get(a + "in_proj_b.weight"), int64_t(r) * GDN_VL, GDN_VL);
                add_rows(m, st_.get(a + "in_proj_a.weight"), int64_t(r) * GDN_VL, GDN_VL);
                QW_CHECK(m.size() == size_t(GDN_PROJ_L) * H, "gdn proj size");
                w.gdn_proj = upload(m);
                auto cw = f32_of(st_.get(a + "conv1d.weight", DType::BF16, {GDN_QKV, 1, GDN_CONV}));
                std::vector<float> lc;
                auto take = [&](int64_t c0, int64_t n) {
                    lc.insert(lc.end(), cw.begin() + c0 * GDN_CONV, cw.begin() + (c0 + n) * GDN_CONV);
                };
                take(int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
                take(GDN_KEY + int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
                take(2 * GDN_KEY + int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
                w.gdn_conv = upload(lc);
                auto A = f32_of(st_.get(a + "A_log")), dt = f32_of(st_.get(a + "dt_bias"));
                w.gdn_A = upload(std::vector<float>(A.begin() + r * GDN_VL, A.begin() + (r + 1) * GDN_VL));
                w.gdn_dt = upload(std::vector<float>(dt.begin() + r * GDN_VL, dt.begin() + (r + 1) * GDN_VL));
                w.gdn_norm = upload(f32_of(st_.get(a + "norm.weight")));
                std::vector<uint16_t> o;
                add_rows(o, st_.get(a + "out_proj.weight", DType::BF16, {H, GDN_VAL}), 0, H,
                         int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
                w.gdn_out = upload(o);
            }
            // MoE
            const std::string mp = p + "mlp.";
            std::vector<uint16_t> router;
            add_rows(router, st_.get(mp + "gate.weight", DType::BF16, {N_EXPERTS, H}), 0, N_EXPERTS);
            w.router = upload(router);
            auto pack_experts = [&](const char *proj, int rows, int K, uint32_t **wd, uint16_t **sd) {
                const size_t words = size_t(rows) * K / 8, groups = size_t(rows) * K / QGROUP;
                std::vector<uint32_t> packed(size_t(EXP_L) * words);
                std::vector<uint16_t> scales(size_t(EXP_L) * groups);
                pool.parallel_for(EXP_L, 1, [&](int64_t b, int64_t e) {
                    for (int64_t i = b; i < e; ++i) {
                        const std::string ep = mp + "experts." + std::to_string(r * EXP_L + i) + "." + proj;
                        const TensorView &pw = st_.get(ep + ".weight_packed", DType::I32, {rows, K / 8});
                        const TensorView &sw = st_.get(ep + ".weight_scale", DType::BF16, {rows, K / QGROUP});
                        gpu::pack_int4_words(pw.i32(), packed.data() + size_t(i) * words, words);
                        for (size_t g = 0; g < groups; ++g) scales[size_t(i) * groups + g] = bf16_to_f16(sw.u16()[g]);
                    }
                });
                *wd = upload(packed);
                *sd = upload(scales);
            };
            // [E][R][K] bf16 rows [row0, row0 + rows) of each local expert -> int4 g128 (symmetric, scale amax / 7)
            auto quant_experts = [&](const TensorView &t, int row0, int rows, int K, uint32_t **wd, uint16_t **sd) {
                const size_t words = size_t(rows) * K / 8, groups = size_t(rows) * K / QGROUP;
                std::vector<uint32_t> packed(size_t(EXP_L) * words);
                std::vector<uint16_t> scales(size_t(EXP_L) * groups);
                const int64_t R = t.dim(1);
                pool.parallel_for(EXP_L, 1, [&](int64_t b0, int64_t e0) {
                    std::vector<int32_t> ct(words);
                    for (int64_t i = b0; i < e0; ++i) {
                        const uint16_t *src = t.u16() + (size_t(r * EXP_L + i) * size_t(R) + size_t(row0)) * size_t(K);
                        for (size_t g = 0; g < groups; ++g) {
                            const uint16_t *x = src + g * QGROUP;
                            float amax = 0.f;
                            for (int k = 0; k < QGROUP; ++k) amax = std::max(amax, std::fabs(bf16_to_f32(x[k])));
                            const uint16_t sc16 = f32_to_f16(amax / 7.f);
                            const float sc = f16_to_f32(sc16), inv = sc > 0.f ? 1.f / sc : 0.f;
                            for (int k = 0; k < QGROUP; k += 8) {
                                uint32_t word = 0;
                                for (int j = 0; j < 8; ++j) {
                                    const long q = std::lrint(bf16_to_f32(x[k + j]) * inv);
                                    word |= uint32_t(std::clamp<long>(q, -8, 7) + 8) << (4 * j);
                                }
                                ct[(g * QGROUP + size_t(k)) / 8] = int32_t(word);
                            }
                            scales[size_t(i) * groups + g] = sc16;
                        }
                        gpu::pack_int4_words(ct.data(), packed.data() + size_t(i) * words, words);
                    }
                });
                *wd = upload(packed);
                *sd = upload(scales);
            };
            if (bf16_experts) {
                const TensorView &gu = st_.get(mp + "experts.gate_up_proj", DType::BF16, {N_EXPERTS, 2 * FFN, H});
                quant_experts(gu, 0, FFN, H, &w.eg, &w.egs);
                quant_experts(gu, FFN, FFN, H, &w.eu, &w.eus);
                quant_experts(st_.get(mp + "experts.down_proj", DType::BF16, {N_EXPERTS, H, FFN}), 0, H, FFN, &w.ed,
                              &w.eds);
            } else {
                pack_experts("gate_proj", FFN, H, &w.eg, &w.egs);
                pack_experts("up_proj", FFN, H, &w.eu, &w.eus);
                pack_experts("down_proj", H, FFN, &w.ed, &w.eds);
            }
            std::vector<uint16_t> gu;
            add_rows(gu, st_.get(mp + "shared_expert.gate_proj.weight"), int64_t(r) * SFFN_L, SFFN_L);
            add_rows(gu, st_.get(mp + "shared_expert.up_proj.weight"), int64_t(r) * SFFN_L, SFFN_L);
            add_rows(gu, st_.get(mp + "shared_expert_gate.weight"), 0, 1);
            w.sh_gu = upload(gu);
            std::vector<uint16_t> sd;
            add_rows(sd, st_.get(mp + "shared_expert.down_proj.weight", DType::BF16, {H, FFN}), 0, H,
                     int64_t(r) * SFFN_L, SFFN_L);
            w.sh_down = upload(sd);
        };
        rk.L.resize(N_LAYERS);
        for (int l = 0; l < N_LAYERS; ++l) load_layer(rk.L[size_t(l)], lp(l), is_qsa(l), false);
        if (mtp_) {
            load_layer(rk.mtp, "mtp.layers.0.", true, true);
            rk.mtp_final = hc("mtp.hyper_connection_mixer.", false);
            std::vector<uint16_t> fh, fe;
            add_rows(fh, st_.get("mtp.fc_hidden.weight", DType::BF16, {H, H}), 0, H, int64_t(r) * SH, SH);
            add_rows(fe, st_.get("mtp.fc_embedding.weight", DType::BF16, {H, H}), 0, H, int64_t(r) * SH, SH);
            rk.mtp_fc_h = upload(fh);
            rk.mtp_fc_e = upload(fe);
            rk.mtp_nh = upload(shard_w1(st_.get("mtp.pre_fc_norm_hidden.weight"), r));
            auto ne = plus_one(f32_of(st_.get("mtp.pre_fc_norm_embedding.weight")));
            rk.mtp_ne = upload(std::vector<float>(ne.begin() + r * SH, ne.begin() + (r + 1) * SH));
        }
        rk.final_hc = hc("model.language_model.hyper_connection_mixer.", false);
        std::vector<uint16_t> head;
        add_rows(head, st_.get("lm_head.weight", DType::BF16, {VOCAB, H}), int64_t(r) * VOCAB_L, VOCAB_L);
        rk.lm_head = upload(head);

        // PLE
        const std::string pp = lp(PLE_LAYER) + "ple.";
        std::vector<uint16_t> kv;
        const TensorView &kp = st_.get(pp + "key_proj.weight", DType::BF16, {HC * H, H});
        for (int s = 0; s < HC; ++s) add_rows(kv, kp, s * H + r * SH, SH);
        add_rows(kv, st_.get(pp + "value_proj.weight", DType::BF16, {H, H}), r * SH, SH);
        rk.ple_kv = upload(kv);
        rk.ple_nk = upload(shard_w1(st_.get(pp + "norm_key.weight"), r));
        rk.ple_nq = upload(shard_w1(st_.get(pp + "norm_query.weight"), r));
        rk.ple_nc = upload(shard_w1(st_.get(pp + "norm_conv.weight"), r));
        auto pc = f32_of(st_.get(pp + "conv1d.weight", DType::BF16, {HC * H, 1, PLE_CONV}));
        std::vector<float> plc(size_t(HC) * SH * PLE_CONV);
        for (int s = 0; s < HC; ++s)
            for (int j = 0; j < SH; ++j)
                for (int k = 0; k < PLE_CONV; ++k)
                    plc[size_t((s * SH + j) * PLE_CONV + k)] = pc[size_t((s * H + r * SH + j) * PLE_CONV + k)];
        rk.ple_conv = upload(plc);

        for (auto &sn : rk.snaps) {
            sn.S = zeros<float>(SLOT_S);
            sn.ring = zeros<float>(SLOT_RING);
            sn.ple = zeros<float>(SLOT_PLE);
            sn.pend = zeros<float>(XW);
        }
        {  // prefill scratch: two micro-batches of half a chunk each
            auto alloc_pb = [&](PrefillBuf &b, int C) {
                b.C = C;
                b.nb_pad = (max_slot_tokens() / IDX_RATIO + 7) / 8 * 8;
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
                if (mtp_) {
                    b.Xm = zeros<float>(size_t(C) * XW);
                    b.m_out = zeros<float>(size_t(C) * 5 * SH);
                    b.m_send = zeros<float>(size_t(C) * 4);
                    b.m_red = zeros<float>(size_t(C) * 4);
                    b.m_xn = zeros<uint16_t>(size_t(C) * 5 * SH);
                }
                CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_emb), size_t(C) * SH * 4, 0));
                CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_ple), size_t(C) * H * 2, 0));
            };
            QW_CHECK(opt_.prefill_chunk >= 512 && opt_.prefill_chunk % 2 == 0, "prefill_chunk must be even and >= 512");
            alloc_pb(rk.pb[0], opt_.prefill_chunk / 2);
            alloc_pb(rk.pb[1], opt_.prefill_chunk / 2);
            CK(hipStreamCreateWithFlags(&rk.s2, hipStreamNonBlocking));
            for (auto &ev : rk.ev)
                CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
            if (rocblas_create_handle(&rk.blas2) != rocblas_status_success) fail("rocblas_create_handle");
            rocblas_set_stream(rk.blas2, rk.s2);
            if (rocblas_create_handle(&rk.blas) != rocblas_status_success) fail("rocblas_create_handle");
            rocblas_set_stream(rk.blas, rk.s);
        }
        // sequence slots
        std::vector<gpu::SlotPtrs> tab(opt_.slot_tokens.size());
        for (size_t si = 0; si < opt_.slot_tokens.size(); ++si) {
            const size_t cap = size_t(opt_.slot_tokens[si]);
            Rank::Slot sl;
            sl.S = zeros<float>(SLOT_S);
            sl.ring = zeros<float>(SLOT_RING);
            sl.ple = zeros<float>(SLOT_PLE);
            sl.pend = zeros<float>(XW);
            for (int i = 0; i < N_QSA + int(mtp_); ++i) {
                sl.K.push_back(zeros<uint16_t>(cap * HEAD_DIM));
                sl.V.push_back(zeros<uint16_t>(cap * HEAD_DIM));
                sl.ck.push_back(zeros<uint16_t>(cap / IDX_RATIO * IDX_DIM + IDX_DIM));
                sl.raw_k.push_back(zeros<float>(cap * IDX_DIM));
            }
            gpu::SlotPtrs &t = tab[si];
            std::memset(&t, 0, sizeof t);
            for (int i = 0; i < N_QSA + int(mtp_); ++i) {
                t.K[i] = sl.K[size_t(i)];
                t.V[i] = sl.V[size_t(i)];
                t.ck[i] = sl.ck[size_t(i)];
                t.raw_k[i] = sl.raw_k[size_t(i)];
            }
            t.S = sl.S;
            t.ring = sl.ring;
            t.ple = sl.ple;
            t.mtp_pend = sl.pend;
            rk.slots.push_back(sl);
        }
        rk.dtab = upload(tab);
        {  // batched decode scratch
            auto &b = rk.bt;
            constexpr size_t M = gpu::MAX_ROWS;
            b.score_ld = (max_slot_tokens() / IDX_RATIO + 7) / 8 * 8;
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
            CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_i32), 5 * M * 4, 0));
            CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_pos), M * 8, 0));
            CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_emb), M * SH * 4, 0));
            CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_ple), M * H * 2, 0));
            CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_logits), M * VOCAB_L * 4, 0));
            CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_lse), M * 2 * 4, 0));
            if (mtp_) {
                b.gdn_save = zeros<float>(size_t(N_GDN) * M * GDN_STATE);
                b.m_hid = zeros<float>(M * XW);
                b.m_send = zeros<float>(M * 4);
                b.m_red = zeros<float>(M * 4);
                b.m_part = zeros<float>(M * 5 * H);
                b.m_out = zeros<float>(M * 5 * SH);
                b.m_xn = zeros<uint16_t>(M * 5 * SH);
                b.amax = zeros<float>(M * 2);
                b.src_ptrs = zeros<const float *>(M);
                CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_src), M * sizeof(float *), 0));
                CK(hipHostMalloc(reinterpret_cast<void **>(&b.h_amax), M * 2 * 4, 0));
            }
        }
        CK(hipDeviceSynchronize());
    };
    // Ranks load one after another (each already uses the whole pool for its
    // experts); a rank's uploads overlap the next rank's conversions poorly,
    // but loading is not the bottleneck yet.
    for (int r = 0; r < RANKS; ++r) {
        load_rank(r);
        log("rank %d loaded", r);
    }
    if (g_saturated) log("warning: %lld bf16 weights saturated to fp16 range", (long long)g_saturated.load());
    for (int r = 0; r < RANKS; ++r) threads_.emplace_back([this, r] { rank_loop(r); });
    if (opt_.warmup) {
        // rocBLAS loads Tensile kernels lazily per shape; the first real chunk
        // was ~40% slower without this. Covers a full chunk, a partial one and
        // a single decode step (which also captures the decode graphs).
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> toks(size_t(opt_.prefill_chunk) + 300, 0);
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = int32_t(i % 1000 + 100);
        prefill(0, toks);
        std::vector<Row> rows;
        for (int m = 1; m <= gpu::MAX_ROWS; ++m) {  // capture every batch size's graph
            rows.clear();
            for (int i = 0; i < m; ++i) rows.push_back({i % num_slots(), int32_t(100 + i)});
            std::stable_sort(rows.begin(), rows.end(), [](const Row &x, const Row &y) { return x.slot < y.slot; });
            decode(rows);
            if (mtp_ && m > 1) {  // one run of m rows: speculative verification (state-saving graph)
                rows.assign(size_t(m), Row{0, 100});
                decode(rows);
                accept(0, 1);
            }
        }
        if (mtp_)
            for (int m = 1; m <= gpu::MAX_ROWS; ++m) {  // MTP graphs: m rows over the slots
                std::vector<DraftReq> reqs;
                for (int i = 0; i < m; ++i) {
                    const int sl = i % num_slots();
                    if (int(reqs.size()) <= sl) reqs.push_back({sl, {}});
                    reqs[size_t(sl)].next.push_back(int32_t(100 + i));
                }
                draft(reqs, 1);
            }
        for (int sl = 0; sl < num_slots(); ++sl) slot_reset(sl);
        log("warmup done in %.1f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

Engine::~Engine() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        ++gen_;
    }
    cv_.notify_all();
    for (auto &t : threads_) t.join();
}


void Engine::rank_loop(int r) {
    Rank &rk = *ranks_[size_t(r)];
    CK(hipSetDevice(rk.dev));
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return gen_ != seen; });
            seen = gen_;
            if (stop_) return;
        }
        std::string err;
        try {
            if (job_ == Job::Prefill) run_prefill_rank(rk);
            else if (job_ == Job::Mtp) run_mtp_rank(rk);
            else run_decode_rank(rk);
        } catch (const std::exception &e) {
            err = e.what();
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!err.empty() && error_.empty()) error_ = err;
            ++done_;
        }
        done_cv_.notify_all();
    }
}

void Engine::dispatch() {
    static const bool trace = std::getenv("QW_TRACE") != nullptr;
    if (trace) {
        if (job_ == Job::Prefill)
            log("trace: prefill slot %d [%lld, +%d)%s", pjob_.slot, (long long)pjob_.start, pjob_.T,
                pjob_.mtp_pend ? " mtp_pend" : "");
        else
            log("trace: %s M=%d%s", job_ == Job::Mtp ? "mtp" : "decode", djob_.M, djob_.save ? " save" : "");
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = 0;
        ++gen_;
    }
    cv_.notify_all();
    {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return done_ == RANKS; });
    }
    if (!error_.empty()) fail(error_);
    if (comm_->error() || comm2_->error()) fail("collective timed out");
}

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
        gpu::gemv_rows(w.down, b.y, HC * SH, b.hc_send + HC, HC_SEND, HC * HC_DOWN, SH, M, HC_DOWN, false, s);
        allreduce(b.hc_send, b.hc_red, HC_SEND);
        gpu::hc_up_mix_B(b.hc_red, w.up, b.y, b.bin_local, inj_out, M, s);
        allgather(b.bin_local, b.bin, SH * 2);
    };
    auto gemv = [&](const uint16_t *W, int N, int K, const uint16_t *x, float *y, bool acc = false) {
        gpu::gemv_rows(W, x, K, y, N, N, K, M, 0, acc, s);
    };
    // One decoder layer (after the PLE step, if any) on b.X; returns the
    // pending MoE output, combined into X by the next mix.
    auto layer = [&](LayerW &w, bool qsa, int idx, const float *pend, float *save) -> const float * {
        mix(w.attn_hc, pend, b.inj_mlp, b.inj_attn);
        if (qsa) {
            gemv(w.qsa_proj, QSA_PROJ_L, H, b.bin, b.proj);
            gpu::qsa_prep_B(b.proj, w.qn, w.kn, w.iqn, rk.dtab, idx, rows, b.q16, b.gate, b.iq, s);
            gpu::qsa_compress_B(w.ikn, rk.dtab, idx, rows, s);
            gpu::qsa_attend_B(b.q16, b.gate, b.iq, rk.dtab, idx, rows, b.scores, b.score_ld, b.lists, b.counts,
                              b.att_partial, b.act16, s);
            gemv(w.qsa_o, H, QH_L * HEAD_DIM, b.act16, b.partial);
        } else {
            gemv(w.gdn_proj, GDN_PROJ_L, H, b.bin, b.proj);
            gpu::gdn_conv_B(b.proj, w.gdn_conv, w.gdn_A, w.gdn_dt, rk.dtab, idx, rows, b.qkv, b.gb, s);
            gpu::gdn_ring_B(b.proj, rk.dtab, idx, rows, s);
            gpu::gdn_scan_B(b.qkv, b.gb, rk.dtab, idx, b.run_start, b.run_len, b.run_slot, M, b.o_raw, save, s);
            gpu::gdn_norm_B(b.o_raw, b.proj, w.gdn_norm, M, b.act16, s);
            gemv(w.gdn_out, H, GDN_VL * GDN_DIM, b.act16, b.partial);
        }
        reduce_scatter(b.partial, b.out_attn);

        mix(w.mlp_hc, b.out_attn, b.inj_attn, b.inj_mlp);
        gemv(w.router, N_EXPERTS, H, b.bin, b.router);
        CK(hipMemsetAsync(b.ecounts, 0, EXP_L * 4, s));
        gpu::moe_route_T(b.router, M, r * EXP_L, b.tok_e, b.tok_w, b.ecounts, s);
        gpu::moe_scatter_T(b.tok_e, b.ecounts, M, b.offsets, b.fill, b.tok_slot, b.pair_tok, s);
        if (M <= moe_pair_max)
            gpu::moe_experts_P(w.eg, w.egs, w.eu, w.eus, w.ed, w.eds, b.offsets, b.pair_tok, b.bin, M, b.moe_h, b.yp, s);
        else
            gpu::moe_experts_T(w.eg, w.egs, w.eu, w.eus, w.ed, w.eds, b.ecounts, b.offsets, b.pair_tok, b.bin, M,
                               b.tiles, b.moe_h, b.yp, s);
        gemv(w.sh_gu, 2 * SFFN_L + 1, H, b.bin, b.shp);
        gpu::swiglu_T(b.shp, 2 * SFFN_L + 1, SFFN_L, M, b.sh_h, s);
        gemv(w.sh_down, H, SFFN_L, b.sh_h, b.sh_out);
        gpu::moe_combine_B(b.yp, b.tok_slot, b.tok_w, b.sh_out, M, b.partial, s);
        reduce_scatter(b.partial, b.out_moe);
        return b.out_moe;
    };

    if (kind == 2) {
        // X = fc_hidden(gnorm(hidden)) per stream + fc_embedding(gnorm(emb)), row-parallel fc
        gpu::mtp_ssq(b.m_hid, b.emb, M, b.m_send, s);
        comm_->allreduce(r, b.m_send, b.m_red, 4, ++k, s, M);
        gpu::mtp_norm(b.m_hid, b.emb, b.m_red, rk.mtp_nh, rk.mtp_ne, M, b.m_xn, s);
        for (int st = 0; st < 5; ++st)
            gpu::gemv_rows(st < HC ? rk.mtp_fc_h : rk.mtp_fc_e, b.m_xn + size_t(st) * M * SH, SH,
                           b.m_part + size_t(st) * M * H, H, H, SH, M, 0, false, s);
        comm_->reduce_scatter(r, b.m_part, b.m_out, SH, ++k, s, 5 * M);
        gpu::mtp_add(b.m_out, M, b.X, s);
        const float *pend = layer(rk.mtp, true, N_QSA, nullptr, nullptr);
        mix(rk.mtp_final, pend, b.inj_mlp, nullptr);
        gemv(rk.lm_head, VOCAB_L, H, b.bin, b.logits);
        gpu::row_argmax(b.logits, VOCAB_L, M, b.amax, s);
        return k;
    }

    gpu::embed_T(b.X, b.emb, M, s);
    const float *pend = nullptr;
    for (int l = 0; l < N_LAYERS; ++l) {
        if (l == PLE_LAYER) {
            if (pend) gpu::hc_combine_T(b.X, pend, b.inj_mlp, M, s);
            pend = nullptr;
            gemv(rk.ple_kv, PLE_KV, H, b.ple_e, b.kv);
            gpu::ple_stats_B(b.kv, b.X, rk.ple_nk, rk.ple_nq, M, b.ple_send, s);
            allreduce(b.ple_send, b.ple_red, 16);
            gpu::ple_apply_B(b.kv, b.ple_red, rk.ple_nc, rk.ple_conv, rk.dtab, rows, b.ncbuf, b.X, s);
        }
        const bool qsa = is_qsa(l);
        float *save = kind == 1 && !qsa ? b.gdn_save + size_t(gdn_index(l)) * gpu::MAX_ROWS * GDN_STATE : nullptr;
        pend = layer(rk.L[size_t(l)], qsa, qsa ? qsa_index(l) : gdn_index(l), pend, save);
    }
    mix(rk.final_hc, pend, b.inj_mlp, nullptr);  // also completes X: the MTP input
    if (mtp_) gpu::save_last_rows(b.X, rk.dtab, b.run_start, b.run_len, b.run_slot, M, s);
    gemv(rk.lm_head, VOCAB_L, H, b.bin, b.logits);
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

void Engine::run_decode_rank(Rank &rk) {
    const int r = rk.r, M = djob_.M;
    hipStream_t s = rk.s;
    auto &b = rk.bt;
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
    launch_graph(s, g, nc, [&] { return record_batch(rk, M, kind); });
    rk.seq_base += nc;
    CK(hipMemcpyAsync(b.h_logits, b.logits, size_t(M) * VOCAB_L * 4, hipMemcpyDeviceToHost, s));
    CK(hipMemcpyAsync(b.h_lse, b.lse, size_t(M) * 2 * 4, hipMemcpyDeviceToHost, s));
    CK(hipStreamSynchronize(s));
    std::memcpy(dlse_parts_.data() + size_t(r) * gpu::MAX_ROWS * 2, b.h_lse, size_t(M) * 2 * 4);
    for (int i = 0; i < M; ++i)
        std::memcpy(dlogits_.data() + size_t(i) * VOCAB + size_t(r) * VOCAB_L, b.h_logits + size_t(i) * VOCAB_L,
                    VOCAB_L * 4);
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
        b.h_src[i] = src >= 0 ? b.X + size_t(src) * XW : rk.slots[size_t(-1 - src)].pend;
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
    CK(hipStreamSynchronize(s));
    std::memcpy(damax_parts_.data() + size_t(r) * MR * 2, b.h_amax, size_t(M) * 2 * 4);
}

// Fills the row tables (slot, first row of its run, runs) and embeddings of
// a batch job; pos[i] is row i's position.
void Engine::stage_rows(const std::vector<Row> &rows, const std::vector<int64_t> &pos) {
    const int M = int(rows.size());
    QW_CHECK(M >= 1 && M <= gpu::MAX_ROWS, "batch: 1..16 rows");
    constexpr int MR = gpu::MAX_ROWS;
    djob_.M = M;
    djob_.i32.assign(5 * MR, 0);
    djob_.pos.assign(MR, 0);
    demb_.resize(size_t(M) * H);
    const TensorView &emb = st_.get("model.language_model.embed_tokens.weight");
    int n_runs = 0;
    for (int i = 0; i < M; ++i) {
        const Row &row = rows[size_t(i)];
        QW_CHECK(row.slot >= 0 && row.slot < num_slots(), "batch: bad slot");
        QW_CHECK(row.token >= 0 && row.token < VOCAB, "batch: token out of range");
        if (i == 0 || rows[size_t(i - 1)].slot != row.slot) {
            for (int j = 0; j < i; ++j) QW_CHECK(rows[size_t(j)].slot != row.slot, "batch: rows of a slot must be consecutive");
            djob_.i32[size_t(2 * MR + n_runs)] = i;         // run_start
            djob_.i32[size_t(4 * MR + n_runs)] = row.slot;  // run_slot
            ++n_runs;
        }
        ++djob_.i32[size_t(3 * MR + n_runs - 1)];  // run_len
        djob_.i32[size_t(i)] = row.slot;                                    // row_slot
        djob_.i32[size_t(MR + i)] = djob_.i32[size_t(2 * MR + n_runs - 1)];  // row_first
        djob_.pos[size_t(i)] = pos[size_t(i)];
        QW_CHECK(pos[size_t(i)] >= 0 && pos[size_t(i)] < slot_capacity(row.slot), "batch: slot is full");
        for (int j = 0; j < H; ++j) demb_[size_t(i) * H + j] = bf16_to_f32(emb.u16()[size_t(row.token) * H + j]);
    }
}

const std::vector<float> &Engine::decode(const std::vector<Row> &rows) {
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
    dlogits_.assign(size_t(M) * VOCAB, 0.f);
    job_ = Job::Decode;
    dispatch();
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
    return dlogits_;
}

void Engine::accept(int slot, int n) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "accept: bad slot");
    SlotHost &sh = slots_[size_t(slot)];
    QW_CHECK(sh.run_first >= 0 && n >= 1 && n <= sh.run_len, "accept: the slot has no such rows in the last decode");
    const int drop = sh.run_len - n;
    if (drop == 0) return;
    QW_CHECK(mtp_, "accept: needs the MTP engine (per-row states)");
    const int row = sh.run_first + n - 1;
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        auto &sl = rk.slots[size_t(slot)];
        CK(hipMemcpy2DAsync(sl.S, GDN_STATE * 4, rk.bt.gdn_save + size_t(row) * GDN_STATE,
                            size_t(gpu::MAX_ROWS) * GDN_STATE * 4, GDN_STATE * 4, N_GDN, hipMemcpyDeviceToDevice, rk.s));
        CK(hipMemcpyAsync(sl.pend, rk.bt.X + size_t(row) * XW, XW * 4, hipMemcpyDeviceToDevice, rk.s));
    }
    for (auto &rkp : ranks_) CK(hipStreamSynchronize(rkp->s));
    sh.len -= drop;
    sh.hist.resize(sh.hist.size() - size_t(std::min<int>(drop, int(sh.hist.size()))));
    sh.run_len = n;
}

const std::vector<std::vector<int32_t>> &Engine::draft(const std::vector<DraftReq> &reqs, int k) {
    drafts_.assign(reqs.size(), {});
    if (!mtp_ || k <= 0 || reqs.empty()) return drafts_;
    constexpr int MR = gpu::MAX_ROWS;
    // per-rank argmax parts -> token of row i
    auto argmax_row = [&](int i) {
        float best = -INFINITY;
        int32_t tok = 0;
        for (int r = 0; r < RANKS; ++r) {
            const float v = damax_parts_[(size_t(r) * MR + i) * 2];
            int32_t idx;
            std::memcpy(&idx, &damax_parts_[(size_t(r) * MR + i) * 2 + 1], 4);
            if (v > best) {
                best = v;
                tok = int32_t(r * VOCAB_L + idx);
            }
        }
        return tok;
    };
    // step 1: the slots' not yet MTP-processed tokens
    std::vector<Row> rows;
    std::vector<int64_t> pos;
    djob_.src.clear();
    std::vector<int> last(reqs.size());
    for (size_t q = 0; q < reqs.size(); ++q) {
        const DraftReq &rq = reqs[q];
        QW_CHECK(rq.slot >= 0 && rq.slot < num_slots(), "draft: bad slot");
        const SlotHost &sh = slots_[size_t(rq.slot)];
        const int n = int(rq.next.size());
        QW_CHECK(n >= 1 && n <= sh.len, "draft: bad row count");
        // hidden of those tokens: the last decode's rows, or the slot's store for the newest one
        const bool from_batch = n > 1 && sh.run_first >= 0 && sh.run_len >= n;
        for (int i = 0; i < n; ++i) {
            rows.push_back({rq.slot, rq.next[size_t(i)]});
            pos.push_back(sh.len - n + i);
            djob_.src.push_back(i == n - 1 || !from_batch ? -1 - rq.slot : sh.run_first + sh.run_len - n + i);
        }
        last[q] = int(rows.size()) - 1;
    }
    for (int step = 0; step < k; ++step) {
        if (step > 0) {  // one row per request: the previous step's hidden and draft
            rows.clear();
            pos.clear();
            djob_.src.clear();
            for (size_t q = 0; q < reqs.size(); ++q) {
                const SlotHost &sh = slots_[size_t(reqs[q].slot)];
                rows.push_back({reqs[q].slot, drafts_[q].back()});
                pos.push_back(sh.len + step - 1);
                djob_.src.push_back(last[q]);
                last[q] = int(q);
            }
        }
        stage_rows(rows, pos);
        job_ = Job::Mtp;
        dispatch();
        for (size_t q = 0; q < reqs.size(); ++q) drafts_[q].push_back(argmax_row(last[q]));
    }
    // the batch hidden now holds MTP rows
    for (auto &sh : slots_) sh.run_first = -1;
    return drafts_;
}

void Engine::slot_reset(int slot) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "bad slot");
    SlotHost &sh = slots_[size_t(slot)];
    sh.len = 0;
    sh.hist.clear();
    sh.reset = true;
    sh.pend_valid = false;
    sh.run_first = -1;
}

int64_t Engine::slot_capacity(int slot) const { return opt_.slot_tokens[size_t(slot)]; }
int64_t Engine::slot_len(int slot) const { return slots_[size_t(slot)].len; }
int Engine::max_slot_tokens() const {
    return *std::max_element(opt_.slot_tokens.begin(), opt_.slot_tokens.end());
}

// single-sequence convenience API (slot 0)
void Engine::reset() { slot_reset(0); }
int64_t Engine::position() const { return slot_len(0); }
const std::vector<float> &Engine::step(int32_t token) {
    const auto &l = decode({{0, token}});
    logits_.assign(l.begin(), l.begin() + VOCAB);
    return logits_;
}

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
        int row0;           // offset of this micro-batch in the chunk
        const float *pend;
    };
    std::vector<MB> mbs;
    mbs.push_back({&rk.pb[0], rk.s, rk.blas, comm_.get(), &rk.seq_base, 0, start, TA, 0, nullptr});
    if (split)
        mbs.push_back({&rk.pb[1], no_split ? rk.s : rk.s2, no_split ? rk.blas : rk.blas2, comm2_.get(), &rk.seq_base2, 0,
                       start + TA, T - TA, TA, nullptr});
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
            gpu::gdn_ring_update_T(b.P, ring, start, T, s);
            record(m, l, 0);
            wait(m, l, 1);
            gpu::gdn_scan_T(b.qkv, b.gb, sl.S + size_t(gi) * GDN_VL * GDN_DIM * GDN_DIM, T, b.o_raw, s);
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

    for (int l = 0; l < N_LAYERS; ++l) {
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
                gpu::ple_apply_T(b.kv, b.ple_red, rk.ple_nc, rk.ple_conv, sl.ple, m.start, m.T, b.ncbuf, b.X, s);
                record(m, l, 3);
            }
            const bool qsa = is_qsa(l);
            layer(m, w, l, qsa, qsa ? qsa_index(l) : gdn_index(l), b.X, m.start, m.T);
        }
    }
    for (auto &m : mbs) {  // complete X: the pre-final-mixer hidden (lm_head and MTP input)
        gpu::hc_combine_T(m.b->X, m.pend, m.b->inj_mlp, m.T, m.s);
        m.pend = nullptr;
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

const std::vector<float> &Engine::prefill(int slot, const std::vector<int32_t> &tokens, std::vector<float> *all_logits,
                                          const std::function<void(int64_t)> &after_chunk) {
    QW_CHECK(!tokens.empty(), "empty prefill");
    QW_CHECK(slot >= 0 && slot < num_slots(), "bad slot");
    SlotHost &sh = slots_[size_t(slot)];
    QW_CHECK(sh.len + int64_t(tokens.size()) <= slot_capacity(slot), "prefill exceeds the slot's capacity");
    const TensorView &emb = st_.get("model.language_model.embed_tokens.weight");
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
            QW_CHECK(chunk[size_t(t)] >= 0 && chunk[size_t(t)] < VOCAB, "token out of range");
            for (int j = 0; j < H; ++j)
                pemb_[size_t(t) * H + j] = bf16_to_f32(emb.u16()[size_t(chunk[size_t(t)]) * H + j]);
        }
        auto ids = hasher_.ids_for(sh.hist, chunk);
        std::vector<float> e(H);
        for (int t = 0; t < T; ++t) {
            ple_->gather(ids[size_t(t)], e.data());
            for (int j = 0; j < H; ++j) pple_[size_t(t) * H + j] = f32_to_f16(e[size_t(j)]);
        }
        pjob_ = {sh.len, T, all, slot, sh.reset, mtp_ && sh.pend_valid && sh.len > 0};
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
