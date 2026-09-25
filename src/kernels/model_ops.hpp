// Model-specific decode kernels for one TP rank (one token, M = 1).
// Shard shapes: the rank owns SH = 640 columns of every one of the 4 streams.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace qw::gpu {

constexpr int SH = 640;        // residual columns per rank per stream
constexpr int HC_DOWN = 324;   // 320 low-rank rows + 4 inject rows
constexpr int HC_SEND = 4 + 4 * HC_DOWN;  // ssq[4] + partial down[4][324] = 1300 floats

// X[s][j] += pend[j] * 2*sigmoid(inj[s]/4)   (pend may be null: no-op)
// then y[s][j] = fp16(X[s][j] * w1[s][j]) and send[s] = sum_j X[s][j]^2.
void hc_pre(float *X, const float *pend, const float *inj, const float *w1, uint16_t *y,
            float *send, hipStream_t s);

// After the all-reduce of the send buffer: rms scales, d, inj; u = silu(d/4);
// g = up @ u for the rank's rows; block_in[j] = mean_s sigmoid(g[s][j]) * y[s][j] * r_s.
// up is fp16 [SH][4][320]. inj_out (may be null) receives the 4 injection logits.
void hc_up_mix(const float *red, const uint16_t *up, const uint16_t *y, uint16_t *block_in,
               float *inj_out, hipStream_t s);

// X[s][j] += pend[j] * 2*sigmoid(inj[s]/4)
void hc_combine(float *X, const float *pend, const float *inj, hipStream_t s);

// Embedding slice into all 4 streams: X[s][j] = emb[j].
void embed_streams(float *X, const float *emb, hipStream_t s);

// ---- QSA (6 local q heads, 1 local kv head)
struct QsaStep {
    const float *proj;      // [4224]: q+gate 6x512 | k 256 | v 256 | indexer 640
    const float *qn, *kn;   // (1+w) [256]
    const float *iqn, *ikn; // (1+w) [128]
    uint16_t *q;            // out [6][256] fp16 (normed, roped)
    float *gate;            // out [6][256]
    float *iq;              // out [4][128] (normed, roped)
    uint16_t *K, *V;        // caches [T][256] fp16
    float *raw_k;           // cache [T][128]
    uint16_t *ck;           // cache [T/4][128] fp16 (normed, roped at the group's first position)
    const int64_t *pos;     // device
};
void qsa_prep(const QsaStep &p, hipStream_t s);

// With nb = (pos+1)/4 complete groups: if nb > 512,
// scores[c] = sum_h relu(iq_h . ck[c]) / sqrt(128) for c < nb; else nothing.
void qsa_index_scores(const float *iq, const uint16_t *ck, float *scores, const int64_t *pos,
                      hipStream_t s);

// The tokens the query at pos attends to: all of 0..pos when nb <= 512, else
// the top-512 groups expanded to tokens plus the tail 4*nb..pos. Writes list
// (<= 2051 entries) and *count.
void qsa_select(const float *scores, const int64_t *pos, int32_t *list, int32_t *count,
                hipStream_t s);

// Attention of the 6 local heads over the listed tokens, with the sigmoid
// gate: out fp16 [6][256]. partial: scratch, >= 33 * 6 * 258 floats.
void qsa_attend(const uint16_t *q, const float *gate, const uint16_t *K, const uint16_t *V,
                const int32_t *list, const int32_t *count, float *partial, uint16_t *out,
                hipStream_t s);

// ---- MoE
// Softmax over 512 router logits, top-10, renormalized; keeps only experts in
// [first, first + n_local) and renumbers them locally.
void moe_route(const float *logits, int first, int n_local, int32_t *ids, float *weights,
               int32_t *count, hipStream_t s);

// h[i] = fp16(silu(v[i]) * v[n + i] * (scale ? sigmoid(*scale) : 1))
void swiglu(const float *v, int n, const float *scale, uint16_t *h, hipStream_t s);

// fp32 -> fp16
void to_f16(const float *x, uint16_t *y, int n, hipStream_t s);

// ---- PLE (layer 1)
// kv: [3200] = key rows [4][640] | value rows [640]. send[16]:
// ssq_key[4], ssq_x[4], dot[4], ssq_val, 0,0,0
void ple_stats(const float *kv, const float *X, const float *nk, const float *nq, float *send,
               hipStream_t s);
// X += gv + silu(dilated_conv(norm_conv(gv))); ring [10][2560] of conv inputs.
void ple_apply(const float *kv, const float *red, const float *nc, const float *conv_w,
               float *ring, const int64_t *pos, float *X, hipStream_t s);

}  // namespace qw::gpu
