// Weight loading: converts one rank's shard of the checkpoint into device
// layouts (fp16 dense matrices, repacked int4 experts, fp32 norms).
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/rank.hpp"
#include "engine/weight_convert.hpp"
#include "kernels/gemv.hpp"

namespace qw {

namespace {

// The int8 copy of an fp16 [N][K] matrix for decode (QW_INT8_DENSE=1), else none.
// QW_INT8_SKIP: comma-separated groups kept fp16 (hc, qsa, gdn, shared, lm_head, ple, mtp).
Q8 q8(const std::vector<uint16_t> &m, int64_t N, int64_t K, const char *group) {
    static const bool on = std::getenv("QW_INT8_DENSE") && std::string(std::getenv("QW_INT8_DENSE")) != "0";
    static const std::string skip = std::string(",") + (std::getenv("QW_INT8_SKIP") ? std::getenv("QW_INT8_SKIP") : "") + ",";
    if (!on || skip.find(std::string(",") + group + ",") != std::string::npos) return {};
    std::vector<int8_t> q;
    std::vector<float> s;
    quantize_rows_i8(m, N, K, q, s, gpu::I8_GROUP);
    return {upload(q), upload(s)};
}

// Hyper-connection mixer weights; inject: the block-inject rows (not in final mixers).
HcW load_hc(const SafeTensors &st, int r, const std::string &p, bool inject) {
    HcW w;
    w.w1 = upload(shard_w1(st.get(p + "hc_norm.weight"), r));
    const TensorView &down = st.get(p + "input_mix_weight_down.weight", DType::BF16, {HC_RANK, HC * H});
    std::vector<uint16_t> d;
    d.reserve(size_t(HC) * HC_DOWN * SH);
    for (int s = 0; s < HC; ++s) {
        add_rows(d, down, 0, HC_RANK, s * H + r * SH, SH);
        if (inject)
            add_rows(d, st.get(p + "block_inject_weight.weight", DType::BF16, {HC, HC * H}), 0, HC, s * H + r * SH, SH);
        else
            d.resize(d.size() + size_t(HC) * SH, 0);
    }
    w.down = upload(d);
    w.down8 = q8(d, int64_t(HC) * HC_DOWN, SH, "hc");
    std::vector<uint16_t> df(size_t(gpu::D_PAD) * HC * SH, 0);
    for (int s2 = 0; s2 < HC; ++s2)
        for (int o = 0; o < HC_DOWN; ++o)
            std::copy(d.begin() + (size_t(s2) * HC_DOWN + o) * SH, d.begin() + (size_t(s2) * HC_DOWN + o + 1) * SH,
                      df.begin() + size_t(o) * HC * SH + size_t(s2) * SH);
    w.down_flat = upload(df);
    const TensorView &up = st.get(p + "input_mix_weight_up.weight", DType::BF16, {HC * H, HC_RANK});
    std::vector<uint16_t> u;
    u.reserve(size_t(SH) * HC * HC_RANK);
    for (int j = 0; j < SH; ++j)
        for (int s = 0; s < HC; ++s) add_rows(u, up, s * H + r * SH + j, 1);
    w.up = upload(u);
    return w;
}

// One decoder layer. The MTP layer's routed experts are bf16 (fused
// gate_up): they are quantized here to the same int4 g128 layout (they only
// draft tokens, which the full model verifies).
void load_layer(LayerW &w, const SafeTensors &st, ThreadPool &pool, int r, const std::string &p, bool qsa,
                bool bf16_experts) {
    w.attn_hc = load_hc(st, r, p + "attn_hyper_connection.", true);
    w.mlp_hc = load_hc(st, r, p + "mlp_hyper_connection.", true);
    if (qsa) {
        const std::string a = p + "self_attn.";
        std::vector<uint16_t> m;
        const TensorView &q = st.get(a + "q_proj.weight", DType::BF16, {Q_HEADS * 2 * HEAD_DIM, H});
        add_rows(m, q, int64_t(r) * QH_L * 2 * HEAD_DIM, QH_L * 2 * HEAD_DIM);
        const int kvh = r / (RANKS / KV_HEADS);
        add_rows(m, st.get(a + "k_proj.weight"), kvh * HEAD_DIM, HEAD_DIM);
        add_rows(m, st.get(a + "v_proj.weight"), kvh * HEAD_DIM, HEAD_DIM);
        add_rows(m, st.get(a + "indexer.index_qk_proj.weight"), 0, (IDX_HEADS + 1) * IDX_DIM);
        QW_CHECK(m.size() == size_t(QSA_PROJ_L) * H, "qsa proj size");
        w.qsa_proj = upload(m);
        w.qsa_proj8 = q8(m, QSA_PROJ_L, H, "qsa");
        w.qn = upload(plus_one(f32_of(st.get(a + "q_norm.weight"))));
        w.kn = upload(plus_one(f32_of(st.get(a + "k_norm.weight"))));
        w.iqn = upload(plus_one(f32_of(st.get(a + "indexer.q_layernorm.weight"))));
        w.ikn = upload(plus_one(f32_of(st.get(a + "indexer.k_layernorm.weight"))));
        std::vector<uint16_t> o;
        add_rows(o, st.get(a + "o_proj.weight", DType::BF16, {H, Q_HEADS * HEAD_DIM}), 0, H,
                 int64_t(r) * QH_L * HEAD_DIM, QH_L * HEAD_DIM);
        w.qsa_o = upload(o);
        w.qsa_o8 = q8(o, H, int64_t(QH_L) * HEAD_DIM, "qsa");
    } else {
        const std::string a = p + "linear_attn.";
        const TensorView &qkv = st.get(a + "in_proj_qkv.weight", DType::BF16, {GDN_QKV, H});
        std::vector<uint16_t> m;
        add_rows(m, qkv, int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
        add_rows(m, qkv, GDN_KEY + int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
        add_rows(m, qkv, 2 * GDN_KEY + int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
        add_rows(m, st.get(a + "in_proj_z.weight"), int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
        add_rows(m, st.get(a + "in_proj_b.weight"), int64_t(r) * GDN_VL, GDN_VL);
        add_rows(m, st.get(a + "in_proj_a.weight"), int64_t(r) * GDN_VL, GDN_VL);
        QW_CHECK(m.size() == size_t(GDN_PROJ_L) * H, "gdn proj size");
        w.gdn_proj = upload(m);
        w.gdn_proj8 = q8(m, GDN_PROJ_L, H, "gdn");
        auto cw = f32_of(st.get(a + "conv1d.weight", DType::BF16, {GDN_QKV, 1, GDN_CONV}));
        std::vector<float> lc;
        auto take = [&](int64_t c0, int64_t n) {
            lc.insert(lc.end(), cw.begin() + c0 * GDN_CONV, cw.begin() + (c0 + n) * GDN_CONV);
        };
        take(int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
        take(GDN_KEY + int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
        take(2 * GDN_KEY + int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
        w.gdn_conv = upload(lc);
        auto A = f32_of(st.get(a + "A_log")), dt = f32_of(st.get(a + "dt_bias"));
        w.gdn_A = upload(std::vector<float>(A.begin() + r * GDN_VL, A.begin() + (r + 1) * GDN_VL));
        w.gdn_dt = upload(std::vector<float>(dt.begin() + r * GDN_VL, dt.begin() + (r + 1) * GDN_VL));
        w.gdn_norm = upload(f32_of(st.get(a + "norm.weight")));
        std::vector<uint16_t> o;
        add_rows(o, st.get(a + "out_proj.weight", DType::BF16, {H, GDN_VAL}), 0, H, int64_t(r) * GDN_VL * GDN_DIM,
                 GDN_VL * GDN_DIM);
        w.gdn_out = upload(o);
        w.gdn_out8 = q8(o, H, int64_t(GDN_VL) * GDN_DIM, "gdn");
    }
    // MoE
    const std::string mp = p + "mlp.";
    std::vector<uint16_t> router;
    add_rows(router, st.get(mp + "gate.weight", DType::BF16, {N_EXPERTS, H}), 0, N_EXPERTS);
    w.router = upload(router);
    auto pack_experts = [&](const char *proj, int rows, int K, uint32_t **wd, uint16_t **sd) {
        const size_t words = size_t(rows) * K / 8, groups = size_t(rows) * K / QGROUP;
        std::vector<uint32_t> packed(size_t(EXP_L) * words);
        std::vector<uint16_t> scales(size_t(EXP_L) * groups);
        pool.parallel_for(EXP_L, 1, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const std::string ep = mp + "experts." + std::to_string(r * EXP_L + i) + "." + proj;
                const TensorView &pw = st.get(ep + ".weight_packed", DType::I32, {rows, K / 8});
                const TensorView &sw = st.get(ep + ".weight_scale", DType::BF16, {rows, K / QGROUP});
                pack_int4_words(pw.i32(), packed.data() + size_t(i) * words, words);
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
        const size_t R = size_t(t.dim(1));
        pool.parallel_for(EXP_L, 1, [&](int64_t b0, int64_t e0) {
            std::vector<int32_t> ct(words);
            for (int64_t i = b0; i < e0; ++i) {
                const uint16_t *src = t.u16() + (size_t(r * EXP_L + i) * R + size_t(row0)) * size_t(K);
                quantize_int4_g128(src, groups, ct.data(), scales.data() + size_t(i) * groups);
                pack_int4_words(ct.data(), packed.data() + size_t(i) * words, words);
            }
        });
        *wd = upload(packed);
        *sd = upload(scales);
    };
    if (bf16_experts) {
        const TensorView &gu = st.get(mp + "experts.gate_up_proj", DType::BF16, {N_EXPERTS, 2 * FFN, H});
        quant_experts(gu, 0, FFN, H, &w.eg, &w.egs);
        quant_experts(gu, FFN, FFN, H, &w.eu, &w.eus);
        quant_experts(st.get(mp + "experts.down_proj", DType::BF16, {N_EXPERTS, H, FFN}), 0, H, FFN, &w.ed, &w.eds);
    } else {
        pack_experts("gate_proj", FFN, H, &w.eg, &w.egs);
        pack_experts("up_proj", FFN, H, &w.eu, &w.eus);
        pack_experts("down_proj", H, FFN, &w.ed, &w.eds);
    }
    std::vector<uint16_t> gu;
    add_rows(gu, st.get(mp + "shared_expert.gate_proj.weight"), int64_t(r) * SFFN_L, SFFN_L);
    add_rows(gu, st.get(mp + "shared_expert.up_proj.weight"), int64_t(r) * SFFN_L, SFFN_L);
    add_rows(gu, st.get(mp + "shared_expert_gate.weight"), 0, 1);
    w.sh_gu = upload(gu);
    w.sh_gu8 = q8(gu, 2 * SFFN_L + 1, H, "shared");
    std::vector<uint16_t> sd;
    add_rows(sd, st.get(mp + "shared_expert.down_proj.weight", DType::BF16, {H, FFN}), 0, H, int64_t(r) * SFFN_L,
             SFFN_L);
    w.sh_down = upload(sd);
    w.sh_down8 = q8(sd, H, SFFN_L, "shared");
}

}  // namespace

void load_rank_weights(Rank &rk, const SafeTensors &st, ThreadPool &pool, bool mtp) {
    const int r = rk.r;
    rk.L.resize(N_LAYERS);
    for (int l = 0; l < N_LAYERS; ++l) load_layer(rk.L[size_t(l)], st, pool, r, layer_prefix(l), is_qsa(l), false);
    if (mtp) {
        load_layer(rk.mtp, st, pool, r, "mtp.layers.0.", true, true);
        rk.mtp_final = load_hc(st, r, "mtp.hyper_connection_mixer.", false);
        std::vector<uint16_t> fh, fe;
        add_rows(fh, st.get("mtp.fc_hidden.weight", DType::BF16, {H, H}), 0, H, int64_t(r) * SH, SH);
        add_rows(fe, st.get("mtp.fc_embedding.weight", DType::BF16, {H, H}), 0, H, int64_t(r) * SH, SH);
        rk.mtp_fc_h = upload(fh);
        rk.mtp_fc_e = upload(fe);
        rk.mtp_fc_h8 = q8(fh, H, SH, "mtp");
        rk.mtp_fc_e8 = q8(fe, H, SH, "mtp");
        rk.mtp_nh = upload(shard_w1(st.get("mtp.pre_fc_norm_hidden.weight"), r));
        auto ne = plus_one(f32_of(st.get("mtp.pre_fc_norm_embedding.weight")));
        rk.mtp_ne = upload(std::vector<float>(ne.begin() + r * SH, ne.begin() + (r + 1) * SH));
    }
    rk.final_hc = load_hc(st, r, "model.language_model.hyper_connection_mixer.", false);
    std::vector<uint16_t> head;
    add_rows(head, st.get("lm_head.weight", DType::BF16, {VOCAB, H}), int64_t(r) * VOCAB_L, VOCAB_L);
    rk.lm_head = upload(head);
    rk.lm_head8 = q8(head, VOCAB_L, H, "lm_head");

    // PLE
    const std::string pp = layer_prefix(PLE_LAYER) + "ple.";
    std::vector<uint16_t> kv;
    const TensorView &kp = st.get(pp + "key_proj.weight", DType::BF16, {HC * H, H});
    for (int s = 0; s < HC; ++s) add_rows(kv, kp, s * H + r * SH, SH);
    add_rows(kv, st.get(pp + "value_proj.weight", DType::BF16, {H, H}), r * SH, SH);
    rk.ple_kv = upload(kv);
    rk.ple_kv8 = q8(kv, PLE_KV, H, "ple");
    rk.ple_nk = upload(shard_w1(st.get(pp + "norm_key.weight"), r));
    rk.ple_nq = upload(shard_w1(st.get(pp + "norm_query.weight"), r));
    rk.ple_nc = upload(shard_w1(st.get(pp + "norm_conv.weight"), r));
    auto pc = f32_of(st.get(pp + "conv1d.weight", DType::BF16, {HC * H, 1, PLE_CONV}));
    std::vector<float> plc(size_t(HC) * SH * PLE_CONV);
    for (int s = 0; s < HC; ++s)
        for (int j = 0; j < SH; ++j)
            for (int k = 0; k < PLE_CONV; ++k)
                plc[size_t((s * SH + j) * PLE_CONV + k)] = pc[size_t((s * H + r * SH + j) * PLE_CONV + k)];
    rk.ple_conv = upload(plc);
}

}  // namespace qw
