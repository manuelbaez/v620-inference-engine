#include "ref/ref_model.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "core/common.hpp"

namespace qw::ref {

using namespace cfg;

namespace {

inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float siluf(float x) { return x * sigmoidf(x); }
inline float softplusf(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }

inline float dot(const float *a, const float *b, int n) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void bf16_row(const uint16_t *src, float *dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}

std::vector<float> bf16_vec(const TensorView &t) {
    QW_CHECK(t.dtype == DType::BF16, t.name + " is not bf16");
    std::vector<float> v(size_t(t.numel()));
    bf16_row(t.u16(), v.data(), int(v.size()));
    return v;
}

// rms-normalize n values; gemma: multiply by (1 + w), plain: by w.
void rmsnorm(const float *x, float *y, int n, const float *w, bool gemma) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    float r = float(1.0 / std::sqrt(ss / n + EPS));
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * (w ? (gemma ? 1.f + w[i] : w[i]) : 1.f);
}

// NeoX rotation of the first ROPE_DIM dims.
void rope(float *x, int64_t pos) {
    constexpr int half = ROPE_DIM / 2;
    for (int i = 0; i < half; ++i) {
        double inv = std::pow(ROPE_THETA, -2.0 * i / ROPE_DIM);
        double a = double(pos) * inv;
        float c = float(std::cos(a)), s = float(std::sin(a));
        float x1 = x[i], x2 = x[i + half];
        x[i] = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

}  // namespace

void State::reset() {
    n_tokens = 0;
    tokens.clear();
    gdn_conv.assign(size_t(N_GDN) * (GDN_CONV - 1) * GDN_QKV, 0.f);
    gdn_S.assign(size_t(N_GDN) * GDN_V_HEADS * GDN_DIM * GDN_DIM, 0.f);
    qsa.assign(N_QSA, Qsa{});
    ple_conv.assign(size_t((PLE_CONV - 1) * PLE_DILATION) * HCH, 0.f);
}

Model::Model(const std::string &model_dir, const std::string &ple_dir, int threads)
    : pool_(threads) {
    check_config(model_dir);
    st_.add_index(model_dir);
    ple_ = std::make_unique<PleTable>(ple_dir);
}

std::string Model::lp(int layer) const {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

// y[t][o] = sum_i w[o][i] * x[t][i]
void Model::linear(const TensorView &w, const float *x, int T, int ldx, float *y, int ldy) {
    QW_CHECK(w.dtype == DType::BF16 && w.shape.size() == 2, w.name + ": expected bf16 matrix");
    const int out = int(w.dim(0)), in = int(w.dim(1));
    const uint16_t *W = w.u16();
    pool_.parallel_for(out, 16, [&](int64_t b, int64_t e) {
        std::vector<float> row(static_cast<size_t>(in));
        for (int64_t o = b; o < e; ++o) {
            bf16_row(W + o * in, row.data(), in);
            for (int t = 0; t < T; ++t) y[size_t(t) * ldy + o] = dot(row.data(), x + size_t(t) * ldx, in);
        }
    });
}

// Same for a compressed-tensors int4 g128 expert matrix.
void Model::expert_linear(const TensorView &packed, const TensorView &scale, const float *x,
                          int T, int ldx, float *y, int ldy) {
    const int out = int(packed.dim(0)), in = int(packed.dim(1)) * 8;
    QW_CHECK(scale.dim(0) == out && scale.dim(1) == in / QGROUP, packed.name + ": scale shape");
    const int32_t *P = packed.i32();
    const uint16_t *S = scale.u16();
    pool_.parallel_for(out, 16, [&](int64_t b, int64_t e) {
        std::vector<float> row(static_cast<size_t>(in));
        for (int64_t o = b; o < e; ++o) {
            const uint32_t *p = reinterpret_cast<const uint32_t *>(P + o * (in / 8));
            const uint16_t *s = S + o * (in / QGROUP);
            for (int c = 0; c < in / 8; ++c) {
                float sc = bf16_to_f32(s[c * 8 / QGROUP]);
                uint32_t word = p[c];
                for (int i = 0; i < 8; ++i)
                    row[size_t(c * 8 + i)] = float(int((word >> (4 * i)) & 0xf) - 8) * sc;
            }
            for (int t = 0; t < T; ++t) y[size_t(t) * ldy + o] = dot(row.data(), x + size_t(t) * ldx, in);
        }
    });
}

void Model::hc_mix(const std::string &prefix, bool with_inject, const float *X, int T,
                   float *block_in, float *inj) {
    std::vector<float> w = bf16_vec(st_.get(prefix + "hc_norm.weight", DType::BF16, {HCH}));
    std::vector<float> xn(size_t(T) * HCH);
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < HC; ++s)
            rmsnorm(X + size_t(t) * HCH + s * H, xn.data() + size_t(t) * HCH + s * H, H,
                    w.data() + s * H, true);
    std::vector<float> d(size_t(T) * HC_RANK);
    linear(st_.get(prefix + "input_mix_weight_down.weight", DType::BF16, {HC_RANK, HCH}),
           xn.data(), T, HCH, d.data(), HC_RANK);
    if (with_inject)
        linear(st_.get(prefix + "block_inject_weight.weight", DType::BF16, {HC, HCH}), xn.data(),
               T, HCH, inj, HC);
    for (auto &v : d) v = siluf(v / HC);
    std::vector<float> g(size_t(T) * HCH);
    linear(st_.get(prefix + "input_mix_weight_up.weight", DType::BF16, {HCH, HC_RANK}), d.data(),
           T, HC_RANK, g.data(), HCH);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < H; ++j) {
            float acc = 0.f;
            for (int s = 0; s < HC; ++s) {
                size_t i = size_t(t) * HCH + s * H + j;
                acc += sigmoidf(g[i]) * xn[i];
            }
            block_in[size_t(t) * H + j] = acc / HC;
        }
}

void Model::combine(float *X, const float *block_out, const float *inj, int T) {
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < HC; ++s) {
            float w = 2.f * sigmoidf(inj[t * HC + s] / HC);
            float *x = X + size_t(t) * HCH + s * H;
            const float *b = block_out + size_t(t) * H;
            for (int j = 0; j < H; ++j) x[j] += b[j] * w;
        }
}

void Model::ple(State &st, float *X, const std::vector<int32_t> &tokens) {
    const int T = int(tokens.size());
    const std::string p = lp(PLE_LAYER) + "ple.";
    std::vector<int32_t> hist(st.tokens.end() - std::min<ptrdiff_t>(st.tokens.size(), 2),
                              st.tokens.end());
    // Only the last 2 tokens matter: an EOS further back cannot change a 3-gram.
    auto ids = hasher_.ids_for(hist, tokens);

    std::vector<float> e(size_t(T) * H);
    for (int t = 0; t < T; ++t) ple_->gather(ids[size_t(t)], e.data() + size_t(t) * H);

    std::vector<float> key(size_t(T) * HCH), val(size_t(T) * H);
    linear(st_.get(p + "key_proj.weight", DType::BF16, {HCH, H}), e.data(), T, H, key.data(), HCH);
    linear(st_.get(p + "value_proj.weight", DType::BF16, {H, H}), e.data(), T, H, val.data(), H);
    auto nk = bf16_vec(st_.get(p + "norm_key.weight", DType::BF16, {HCH}));
    auto nq = bf16_vec(st_.get(p + "norm_query.weight", DType::BF16, {HCH}));
    auto nc = bf16_vec(st_.get(p + "norm_conv.weight", DType::BF16, {HCH}));
    auto cw = bf16_vec(st_.get(p + "conv1d.weight", DType::BF16, {HCH, 1, PLE_CONV}));

    const int hist_len = (PLE_CONV - 1) * PLE_DILATION;  // 9
    std::vector<float> seq(size_t(hist_len + T) * HCH);   // conv inputs incl. history
    std::copy(st.ple_conv.begin(), st.ple_conv.end(), seq.begin());
    std::vector<float> gv(size_t(T) * HCH), tmp(HCH), q(HCH);
    for (int t = 0; t < T; ++t) {
        float *kt = key.data() + size_t(t) * HCH;
        const float *xt = X + size_t(t) * HCH;
        for (int s = 0; s < HC; ++s) {
            rmsnorm(kt + s * H, tmp.data() + s * H, H, nk.data() + s * H, true);
            rmsnorm(xt + s * H, q.data() + s * H, H, nq.data() + s * H, true);
            float gs = dot(tmp.data() + s * H, q.data() + s * H, H) / std::sqrt(float(H));
            float mag = std::sqrt(std::max(std::fabs(gs), 1e-6f));
            float gate = sigmoidf(gs < 0 ? -mag : (gs > 0 ? mag : 0.f));
            for (int j = 0; j < H; ++j) gv[size_t(t) * HCH + s * H + j] = gate * val[size_t(t) * H + j];
        }
        float *in = seq.data() + size_t(hist_len + t) * HCH;
        for (int s = 0; s < HC; ++s)
            rmsnorm(gv.data() + size_t(t) * HCH + s * H, in + s * H, H, nc.data() + s * H, true);
    }
    for (int t = 0; t < T; ++t) {
        float *x = X + size_t(t) * HCH;
        const float *g = gv.data() + size_t(t) * HCH;
        for (int c = 0; c < HCH; ++c) {
            float acc = 0.f;
            for (int k = 0; k < PLE_CONV; ++k)
                acc += cw[size_t(c) * PLE_CONV + k] *
                       seq[size_t(t + k * PLE_DILATION) * HCH + c];  // row t+9 is the current one
            x[c] += g[c] + siluf(acc);
        }
    }
    std::copy(seq.end() - ptrdiff_t(hist_len) * HCH, seq.end(), st.ple_conv.begin());
}

void Model::gdn(State &st, int layer, const float *x, int T, float *out) {
    const std::string p = lp(layer) + "linear_attn.";
    const int gi = gdn_index(layer);
    std::vector<float> qkv(size_t(T) * GDN_QKV), z(size_t(T) * GDN_VAL), b(size_t(T) * GDN_V_HEADS),
        a(size_t(T) * GDN_V_HEADS);
    linear(st_.get(p + "in_proj_qkv.weight", DType::BF16, {GDN_QKV, H}), x, T, H, qkv.data(), GDN_QKV);
    linear(st_.get(p + "in_proj_z.weight", DType::BF16, {GDN_VAL, H}), x, T, H, z.data(), GDN_VAL);
    linear(st_.get(p + "in_proj_b.weight", DType::BF16, {GDN_V_HEADS, H}), x, T, H, b.data(), GDN_V_HEADS);
    linear(st_.get(p + "in_proj_a.weight", DType::BF16, {GDN_V_HEADS, H}), x, T, H, a.data(), GDN_V_HEADS);
    auto cw = bf16_vec(st_.get(p + "conv1d.weight", DType::BF16, {GDN_QKV, 1, GDN_CONV}));
    auto A_log = bf16_vec(st_.get(p + "A_log", DType::BF16, {GDN_V_HEADS}));
    auto dt_bias = bf16_vec(st_.get(p + "dt_bias", DType::BF16, {GDN_V_HEADS}));
    auto nw = bf16_vec(st_.get(p + "norm.weight", DType::BF16, {GDN_DIM}));

    // causal depthwise conv over [history(3) | chunk]
    float *hist = st.gdn_conv.data() + size_t(gi) * (GDN_CONV - 1) * GDN_QKV;
    std::vector<float> seq(size_t(GDN_CONV - 1 + T) * GDN_QKV);
    std::copy(hist, hist + (GDN_CONV - 1) * GDN_QKV, seq.begin());
    std::copy(qkv.begin(), qkv.end(), seq.begin() + (GDN_CONV - 1) * GDN_QKV);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < GDN_QKV; ++c) {
            float acc = 0.f;
            for (int k = 0; k < GDN_CONV; ++k)
                acc += cw[size_t(c) * GDN_CONV + k] * seq[size_t(t + k) * GDN_QKV + c];
            qkv[size_t(t) * GDN_QKV + c] = siluf(acc);
        }
    std::copy(seq.end() - (GDN_CONV - 1) * GDN_QKV, seq.end(), hist);

    float *S = st.gdn_S.data() + size_t(gi) * GDN_V_HEADS * GDN_DIM * GDN_DIM;
    std::vector<float> o(size_t(T) * GDN_VAL);
    const float qscale = 1.f / std::sqrt(float(GDN_DIM));
    for (int t = 0; t < T; ++t) {
        float *row = qkv.data() + size_t(t) * GDN_QKV;
        float *q = row, *k = row + GDN_KEY, *v = row + 2 * GDN_KEY;
        for (int h = 0; h < GDN_QK_HEADS; ++h) {
            float *qh = q + h * GDN_DIM, *kh = k + h * GDN_DIM;
            float nq = std::sqrt(dot(qh, qh, GDN_DIM) + EPS), nk = std::sqrt(dot(kh, kh, GDN_DIM) + EPS);
            for (int i = 0; i < GDN_DIM; ++i) {
                qh[i] = qh[i] / nq * qscale;
                kh[i] = kh[i] / nk;
            }
        }
        pool_.parallel_for(GDN_V_HEADS, 1, [&](int64_t hb, int64_t he) {
            std::vector<float> u(GDN_DIM);
            for (int64_t h = hb; h < he; ++h) {
                const int kh = int(h) / (GDN_V_HEADS / GDN_QK_HEADS);
                const float *kk = k + kh * GDN_DIM, *qq = q + kh * GDN_DIM, *vv = v + h * GDN_DIM;
                float g = -std::exp(A_log[size_t(h)]) * softplusf(a[size_t(t) * GDN_V_HEADS + h] + dt_bias[size_t(h)]);
                float beta = sigmoidf(b[size_t(t) * GDN_V_HEADS + h]);
                float decay = std::exp(g);
                float *Sh = S + size_t(h) * GDN_DIM * GDN_DIM;  // [k][v]
                for (int i = 0; i < GDN_DIM * GDN_DIM; ++i) Sh[i] *= decay;
                for (int j = 0; j < GDN_DIM; ++j) u[size_t(j)] = vv[j];
                for (int i = 0; i < GDN_DIM; ++i) {
                    float ki = kk[i];
                    const float *Si = Sh + i * GDN_DIM;
                    for (int j = 0; j < GDN_DIM; ++j) u[size_t(j)] -= Si[j] * ki;
                }
                for (int j = 0; j < GDN_DIM; ++j) u[size_t(j)] *= beta;
                for (int i = 0; i < GDN_DIM; ++i) {
                    float ki = kk[i];
                    float *Si = Sh + i * GDN_DIM;
                    for (int j = 0; j < GDN_DIM; ++j) Si[j] += ki * u[size_t(j)];
                }
                float *oh = o.data() + size_t(t) * GDN_VAL + h * GDN_DIM;
                for (int j = 0; j < GDN_DIM; ++j) oh[j] = 0.f;
                for (int i = 0; i < GDN_DIM; ++i) {
                    float qi = qq[i];
                    const float *Si = Sh + i * GDN_DIM;
                    for (int j = 0; j < GDN_DIM; ++j) oh[j] += Si[j] * qi;
                }
                float tmp[GDN_DIM];
                rmsnorm(oh, tmp, GDN_DIM, nw.data(), false);
                const float *zh = z.data() + size_t(t) * GDN_VAL + h * GDN_DIM;
                for (int j = 0; j < GDN_DIM; ++j) oh[j] = tmp[j] * sigmoidf(zh[j]);
            }
        });
    }
    linear(st_.get(p + "out_proj.weight", DType::BF16, {H, GDN_VAL}), o.data(), T, GDN_VAL, out, H);
}

void Model::qsa(State &st, int layer, const float *x, int T, float *out) {
    const std::string p = lp(layer) + "self_attn.";
    State::Qsa &c = st.qsa[size_t(qsa_index(layer))];
    constexpr int QG = Q_HEADS * 2 * HEAD_DIM;  // 12288
    constexpr int KV = KV_HEADS * HEAD_DIM;     // 512
    constexpr int IQ = (IDX_HEADS + 1) * IDX_DIM;
    std::vector<float> qg(size_t(T) * QG), k(size_t(T) * KV), v(size_t(T) * KV), iq(size_t(T) * IQ);
    linear(st_.get(p + "q_proj.weight", DType::BF16, {QG, H}), x, T, H, qg.data(), QG);
    linear(st_.get(p + "k_proj.weight", DType::BF16, {KV, H}), x, T, H, k.data(), KV);
    linear(st_.get(p + "v_proj.weight", DType::BF16, {KV, H}), x, T, H, v.data(), KV);
    linear(st_.get(p + "indexer.index_qk_proj.weight", DType::BF16, {IQ, H}), x, T, H, iq.data(), IQ);
    auto qn = bf16_vec(st_.get(p + "q_norm.weight", DType::BF16, {HEAD_DIM}));
    auto kn = bf16_vec(st_.get(p + "k_norm.weight", DType::BF16, {HEAD_DIM}));
    auto iqn = bf16_vec(st_.get(p + "indexer.q_layernorm.weight", DType::BF16, {IDX_DIM}));
    auto ikn = bf16_vec(st_.get(p + "indexer.k_layernorm.weight", DType::BF16, {IDX_DIM}));

    const int64_t base = st.n_tokens;
    std::vector<float> q(size_t(T) * Q_HEADS * HEAD_DIM), gate(size_t(T) * Q_HEADS * HEAD_DIM),
        iqh(size_t(T) * IDX_HEADS * IDX_DIM);
    for (int t = 0; t < T; ++t) {
        const int64_t pos = base + t;
        for (int h = 0; h < Q_HEADS; ++h) {
            const float *src = qg.data() + size_t(t) * QG + h * 2 * HEAD_DIM;
            float *qh = q.data() + (size_t(t) * Q_HEADS + h) * HEAD_DIM;
            rmsnorm(src, qh, HEAD_DIM, qn.data(), true);
            rope(qh, pos);
            std::copy(src + HEAD_DIM, src + 2 * HEAD_DIM, gate.data() + (size_t(t) * Q_HEADS + h) * HEAD_DIM);
        }
        for (int j = 0; j < KV_HEADS; ++j) {
            float *kh = k.data() + size_t(t) * KV + j * HEAD_DIM;
            float tmp[HEAD_DIM];
            rmsnorm(kh, tmp, HEAD_DIM, kn.data(), true);
            rope(tmp, pos);
            std::copy(tmp, tmp + HEAD_DIM, kh);
        }
        for (int h = 0; h < IDX_HEADS; ++h) {
            float *dst = iqh.data() + (size_t(t) * IDX_HEADS + h) * IDX_DIM;
            rmsnorm(iq.data() + size_t(t) * IQ + h * IDX_DIM, dst, IDX_DIM, iqn.data(), true);
            rope(dst, pos);
        }
        // append to caches; complete a compressed group when this token closes it
        c.k.insert(c.k.end(), k.begin() + ptrdiff_t(t) * KV, k.begin() + ptrdiff_t(t + 1) * KV);
        c.v.insert(c.v.end(), v.begin() + ptrdiff_t(t) * KV, v.begin() + ptrdiff_t(t + 1) * KV);
        const float *rk = iq.data() + size_t(t) * IQ + IDX_HEADS * IDX_DIM;
        c.raw_k.insert(c.raw_k.end(), rk, rk + IDX_DIM);
        if ((pos + 1) % IDX_RATIO == 0) {
            float mean[IDX_DIM] = {}, normed[IDX_DIM];
            for (int r = 0; r < IDX_RATIO; ++r)
                for (int i = 0; i < IDX_DIM; ++i) mean[i] += c.raw_k[size_t(pos - r) * IDX_DIM + i];
            for (int i = 0; i < IDX_DIM; ++i) mean[i] /= IDX_RATIO;
            rmsnorm(mean, normed, IDX_DIM, ikn.data(), true);
            rope(normed, pos + 1 - IDX_RATIO);
            c.ck.insert(c.ck.end(), normed, normed + IDX_DIM);
        }
    }

    // attention per query token over its selected positions
    std::vector<float> o(size_t(T) * Q_HEADS * HEAD_DIM);
    pool_.parallel_for(T, 1, [&](int64_t tb, int64_t te) {
        std::vector<int64_t> sel;
        std::vector<float> scores, logits;
        std::vector<int32_t> order;
        for (int64_t t = tb; t < te; ++t) {
            const int64_t pos = base + t;
            const int64_t nb = (pos + 1) / IDX_RATIO;
            sel.clear();
            if (nb <= IDX_TOPK_BLOCKS) {
                for (int64_t i = 0; i <= pos; ++i) sel.push_back(i);
            } else {
                scores.assign(size_t(nb), 0.f);
                const float *qi = iqh.data() + size_t(t) * IDX_HEADS * IDX_DIM;
                for (int64_t b = 0; b < nb; ++b) {
                    float s = 0.f;
                    for (int h = 0; h < IDX_HEADS; ++h)
                        s += std::max(0.f, dot(qi + h * IDX_DIM, c.ck.data() + size_t(b) * IDX_DIM, IDX_DIM));
                    scores[size_t(b)] = s / std::sqrt(float(IDX_DIM));
                }
                order.resize(size_t(nb));
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + IDX_TOPK_BLOCKS, order.end(),
                                  [&](int32_t a, int32_t b) {
                                      return scores[size_t(a)] > scores[size_t(b)] ||
                                             (scores[size_t(a)] == scores[size_t(b)] && a < b);
                                  });
                for (int i = 0; i < IDX_TOPK_BLOCKS; ++i)
                    for (int r = 0; r < IDX_RATIO; ++r) sel.push_back(int64_t(order[size_t(i)]) * IDX_RATIO + r);
                for (int64_t i = nb * IDX_RATIO; i <= pos; ++i) sel.push_back(i);
            }
            logits.resize(sel.size());
            for (int h = 0; h < Q_HEADS; ++h) {
                const int kvh = h / (Q_HEADS / KV_HEADS);
                const float *qh = q.data() + (size_t(t) * Q_HEADS + h) * HEAD_DIM;
                float mx = -INFINITY;
                for (size_t i = 0; i < sel.size(); ++i) {
                    logits[i] = dot(qh, c.k.data() + size_t(sel[i]) * KV + kvh * HEAD_DIM, HEAD_DIM) /
                                std::sqrt(float(HEAD_DIM));
                    mx = std::max(mx, logits[i]);
                }
                double sum = 0;
                for (auto &l : logits) {
                    l = std::exp(l - mx);
                    sum += l;
                }
                float *oh = o.data() + (size_t(t) * Q_HEADS + h) * HEAD_DIM;
                std::fill(oh, oh + HEAD_DIM, 0.f);
                for (size_t i = 0; i < sel.size(); ++i) {
                    float w = float(logits[i] / sum);
                    const float *vv = c.v.data() + size_t(sel[i]) * KV + kvh * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; ++d) oh[d] += w * vv[d];
                }
                const float *gh = gate.data() + (size_t(t) * Q_HEADS + h) * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) oh[d] *= sigmoidf(gh[d]);
            }
        }
    });
    linear(st_.get(p + "o_proj.weight", DType::BF16, {H, Q_HEADS * HEAD_DIM}), o.data(), T,
           Q_HEADS * HEAD_DIM, out, H);
}

void Model::moe(int layer, const float *x, int T, float *out) {
    const std::string p = lp(layer) + "mlp.";
    std::vector<float> logits(size_t(T) * N_EXPERTS);
    linear(st_.get(p + "gate.weight", DType::BF16, {N_EXPERTS, H}), x, T, H, logits.data(), N_EXPERTS);

    // route: softmax, top-k, renormalize
    std::vector<std::vector<std::pair<int, float>>> by_expert(N_EXPERTS);  // (token, weight)
    for (int t = 0; t < T; ++t) {
        float *l = logits.data() + size_t(t) * N_EXPERTS;
        float mx = *std::max_element(l, l + N_EXPERTS);
        double sum = 0;
        std::vector<float> pr(N_EXPERTS);
        for (int e = 0; e < N_EXPERTS; ++e) {
            pr[size_t(e)] = std::exp(l[e] - mx);
            sum += pr[size_t(e)];
        }
        std::vector<int> idx(N_EXPERTS);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + TOP_K, idx.end(), [&](int a, int b) {
            return pr[size_t(a)] > pr[size_t(b)] || (pr[size_t(a)] == pr[size_t(b)] && a < b);
        });
        double top = 0;
        for (int i = 0; i < TOP_K; ++i) top += pr[size_t(idx[size_t(i)])];
        for (int i = 0; i < TOP_K; ++i)
            by_expert[size_t(idx[size_t(i)])].push_back({t, float(pr[size_t(idx[size_t(i)])] / top)});
        (void)sum;
    }

    std::fill(out, out + size_t(T) * H, 0.f);
    std::vector<float> xs, g, u, y;
    for (int e = 0; e < N_EXPERTS; ++e) {
        auto &toks = by_expert[size_t(e)];
        if (toks.empty()) continue;
        const int n = int(toks.size());
        xs.resize(size_t(n) * H);
        for (int i = 0; i < n; ++i)
            std::copy(x + size_t(toks[size_t(i)].first) * H, x + size_t(toks[size_t(i)].first + 1) * H,
                      xs.begin() + ptrdiff_t(i) * H);
        const std::string ep = p + "experts." + std::to_string(e) + ".";
        g.resize(size_t(n) * FFN);
        u.resize(size_t(n) * FFN);
        y.resize(size_t(n) * H);
        expert_linear(st_.get(ep + "gate_proj.weight_packed", DType::I32, {FFN, H / 8}),
                      st_.get(ep + "gate_proj.weight_scale", DType::BF16, {FFN, H / QGROUP}), xs.data(),
                      n, H, g.data(), FFN);
        expert_linear(st_.get(ep + "up_proj.weight_packed", DType::I32, {FFN, H / 8}),
                      st_.get(ep + "up_proj.weight_scale", DType::BF16, {FFN, H / QGROUP}), xs.data(),
                      n, H, u.data(), FFN);
        for (size_t i = 0; i < g.size(); ++i) g[i] = siluf(g[i]) * u[i];
        expert_linear(st_.get(ep + "down_proj.weight_packed", DType::I32, {H, FFN / 8}),
                      st_.get(ep + "down_proj.weight_scale", DType::BF16, {H, FFN / QGROUP}), g.data(),
                      n, FFN, y.data(), H);
        for (int i = 0; i < n; ++i) {
            float w = toks[size_t(i)].second;
            float *o = out + size_t(toks[size_t(i)].first) * H;
            for (int j = 0; j < H; ++j) o[j] += w * y[size_t(i) * H + j];
        }
    }

    // shared expert
    g.resize(size_t(T) * FFN);
    u.resize(size_t(T) * FFN);
    y.resize(size_t(T) * H);
    std::vector<float> sg(static_cast<size_t>(T));
    linear(st_.get(p + "shared_expert.gate_proj.weight", DType::BF16, {FFN, H}), x, T, H, g.data(), FFN);
    linear(st_.get(p + "shared_expert.up_proj.weight", DType::BF16, {FFN, H}), x, T, H, u.data(), FFN);
    for (size_t i = 0; i < g.size(); ++i) g[i] = siluf(g[i]) * u[i];
    linear(st_.get(p + "shared_expert.down_proj.weight", DType::BF16, {H, FFN}), g.data(), T, FFN, y.data(), H);
    linear(st_.get(p + "shared_expert_gate.weight", DType::BF16, {1, H}), x, T, H, sg.data(), 1);
    for (int t = 0; t < T; ++t) {
        float w = sigmoidf(sg[size_t(t)]);
        for (int j = 0; j < H; ++j) out[size_t(t) * H + j] += w * y[size_t(t) * H + j];
    }
}

void Model::forward(State &st, const std::vector<int32_t> &tokens, std::vector<float> &logits,
                    bool all_logits) {
    const int T = int(tokens.size());
    QW_CHECK(T > 0, "empty forward");
    if (st.gdn_S.empty()) st.reset();

    // embeddings, repeated into every stream
    const TensorView &emb = st_.get("model.language_model.embed_tokens.weight", DType::BF16, {VOCAB, H});
    std::vector<float> X(size_t(T) * HCH);
    for (int t = 0; t < T; ++t) {
        QW_CHECK(tokens[size_t(t)] >= 0 && tokens[size_t(t)] < VOCAB, "token id out of range");
        float *row = X.data() + size_t(t) * HCH;
        bf16_row(emb.u16() + size_t(tokens[size_t(t)]) * H, row, H);
        for (int s = 1; s < HC; ++s) std::copy(row, row + H, row + s * H);
    }

    std::vector<float> bin(size_t(T) * H), bout(size_t(T) * H), inj(size_t(T) * HC),
        pend_out(size_t(T) * H), pend_inj(size_t(T) * HC);
    bool pending = false;
    if (dump_layers) dump_layers->clear();
    for (int L = 0; L < N_LAYERS; ++L) {
        if (pending) combine(X.data(), pend_out.data(), pend_inj.data(), T);
        if (L == PLE_LAYER) ple(st, X.data(), tokens);
        hc_mix(lp(L) + "attn_hyper_connection.", true, X.data(), T, bin.data(), inj.data());
        if (is_qsa(L)) qsa(st, L, bin.data(), T, bout.data());
        else gdn(st, L, bin.data(), T, bout.data());
        combine(X.data(), bout.data(), inj.data(), T);
        hc_mix(lp(L) + "mlp_hyper_connection.", true, X.data(), T, bin.data(), pend_inj.data());
        moe(L, bin.data(), T, pend_out.data());
        pending = true;
        if (dump_layers) {
            std::vector<float> snap = X;
            combine(snap.data(), pend_out.data(), pend_inj.data(), T);
            dump_layers->push_back(std::move(snap));
        }
    }
    combine(X.data(), pend_out.data(), pend_inj.data(), T);
    hc_mix("model.language_model.hyper_connection_mixer.", false, X.data(), T, bin.data(), nullptr);

    const TensorView &head = st_.get("lm_head.weight", DType::BF16, {VOCAB, H});
    if (all_logits) {
        logits.assign(size_t(T) * VOCAB, 0.f);
        linear(head, bin.data(), T, H, logits.data(), VOCAB);
    } else {
        logits.assign(VOCAB, 0.f);
        linear(head, bin.data() + size_t(T - 1) * H, 1, H, logits.data(), VOCAB);
    }
    st.tokens.insert(st.tokens.end(), tokens.begin(), tokens.end());
    st.n_tokens += T;
}

}  // namespace qw::ref
