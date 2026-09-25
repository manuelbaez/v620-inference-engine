// Batched (ragged) decode kernels for one TP rank.
//
// A step processes M rows (M <= MAX_ROWS). Row r is one token of sequence
// slot[r] at position pos[r]. Rows of the same slot are consecutive and in
// position order (a plain decode step has one row per slot; speculative
// verification has 1 + K rows per slot). run_first[r] is the row index where
// r's slot run starts, so a row can find earlier tokens of its own sequence in
// the same batch.
//
// Per-sequence state lives in slots; kernels find it through a device table
// of SlotPtrs indexed by slot.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "core/config.hpp"

namespace qw::gpu {

constexpr int MAX_ROWS = 16;
constexpr int QSA_LAYERS_MAX = cfg::N_QSA + 1;  // + the MTP layer

struct SlotPtrs {
    uint16_t *K[QSA_LAYERS_MAX], *V[QSA_LAYERS_MAX], *ck[QSA_LAYERS_MAX];
    float *raw_k[QSA_LAYERS_MAX];
    float *S;     // [N_GDN][12][128][128]
    float *ring;  // [N_GDN][GDN_RING][2560]
    float *ple;   // [PLE_RING][2560]
    float *mtp_pend;  // [HC][SH] pre-final-mixer hidden of the slot's last token (MTP input)
};

struct Rows {
    const int32_t *slot;       // [M]
    const int64_t *pos;        // [M]
    const int32_t *run_first;  // [M]
    int M;
};

// y[m*ys + n] (+)= W[n] . x[m*xs + (n / rpg) * K ...]; xs, ys in elements;
// rpg = rows per group (0: no grouping). W fp16 [N][K], x fp16, y fp32.
void gemv_rows(const uint16_t *W, const uint16_t *x, int xs, float *y, int ys, int N, int K, int M, int rpg,
               bool accumulate, hipStream_t s);

// ---- HC, one-phase (decode): send[m] = ssq[4] | down partial [4][324]
void hc_pre_B(float *X, const float *pend, const float *inj, const float *w1, uint16_t *y, float *send, int M,
              hipStream_t s);
void hc_up_mix_B(const float *red, const uint16_t *up, const uint16_t *y, uint16_t *block_in, float *inj_out, int M,
                 hipStream_t s);

// ---- QSA (layer index qi into SlotPtrs)
// proj f32 [M][4224]; writes q16 [M][1536], gate f32 [M][1536], iq f32 [M][512],
// and the slot's K/V/raw_k at each row's position.
void qsa_prep_B(const float *proj, const float *qn, const float *kn, const float *iqn, const SlotPtrs *tab, int qi,
                Rows rows, uint16_t *q16, float *gate, float *iq, hipStream_t s);
// Compressed key for rows that complete a group (after qsa_prep_B).
void qsa_compress_B(const float *ikn, const SlotPtrs *tab, int qi, Rows rows, hipStream_t s);
// scores [M][ld]; lists [M][LIST_W]; partial [M][33][6][258]; out fp16 [M][1536]
void qsa_attend_B(const uint16_t *q16, const float *gate, const float *iq, const SlotPtrs *tab, int qi, Rows rows,
                  float *scores, int ld, int32_t *lists, int32_t *counts, float *partial, uint16_t *out,
                  hipStream_t s);

// ---- GDN (layer index gi). proj f32 [M][4120].
// qkv f32 [M][2560] (post-conv, normalized), gb [M][12][2]
void gdn_conv_B(const float *proj, const float *conv_w, const float *A_log, const float *dt_bias,
                const SlotPtrs *tab, int gi, Rows rows, float *qkv, float *gb, hipStream_t s);
// ring[slot][pos % GDN_RING] = pre-conv row (after gdn_conv_B)
void gdn_ring_B(const float *proj, const SlotPtrs *tab, int gi, Rows rows, hipStream_t s);
// Delta rule per run of rows (run_start/run_len/run_slot, n_runs <= M). With
// save != null, the state after each row is also written to save[row].
void gdn_scan_B(const float *qkv, const float *gb, const SlotPtrs *tab, int gi, const int32_t *run_start,
                const int32_t *run_len, const int32_t *run_slot, int M, float *o_raw, float *save, hipStream_t s);
void gdn_norm_B(const float *o_raw, const float *proj, const float *norm_w, int M, uint16_t *out, hipStream_t s);

// ---- PLE. kv f32 [M][3200]
void ple_stats_B(const float *kv, const float *X, const float *nk, const float *nq, int M, float *send, hipStream_t s);
void ple_apply_B(const float *kv, const float *red, const float *nc, const float *conv_w, const SlotPtrs *tab,
                 Rows rows, float *ncbuf, float *X, hipStream_t s);

// ---- MoE combine: out f32 [M][2560] = shared + sum_j w * yp[slot_j]
void moe_combine_B(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w, const float *shared, int M,
                   float *out, hipStream_t s);

// ---- MTP input: X_mtp[m][s] = fc_hidden(gnorm(X[m])[s]) + fc_embedding(gnorm(emb[m])),
// with the fc GEMMs row-parallel (each rank multiplies its SH columns; a
// reduce-scatter sums them). Work rows are stream-major: [5][rows] = 4
// hidden streams, then the embedding.
// send[m] = {sum X[m]^2 over the rank's HC*SH, sum emb[m]^2 over its SH, 0, 0}
// (4 floats: collectives move 16-byte rows)
void mtp_ssq(const float *X, const float *emb, int rows, float *send, hipStream_t s);
// xn[5][rows][SH] fp16 from the all-reduced sums; nh [HC][SH], ne [SH] are 1 + w
void mtp_norm(const float *X, const float *emb, const float *red, const float *nh, const float *ne, int rows,
              uint16_t *xn, hipStream_t s);
// X[m][s][j] = out[s][m][j] + out[4][m][j]; out f32 [5][rows][SH]
void mtp_add(const float *out, int rows, float *X, hipStream_t s);
// dst[m][0..n) = src[m][0..n) (src: device array of row pointers)
void gather_rows(const float *const *src, int rows, int n, float *dst, hipStream_t s);
// tab[run_slot].mtp_pend = X[run_start + run_len - 1] for each run
void save_last_rows(const float *X, const SlotPtrs *tab, const int32_t *run_start, const int32_t *run_len,
                    const int32_t *run_slot, int M, hipStream_t s);
// out[m] = {max, index as int bits} of a vocab shard row (first index on ties)
void row_argmax(const float *logits, int n, int M, float *out, hipStream_t s);
// Xm[t] = t < o ? pend : X[t - o] for t < rows (o = 0 or 1)
void mtp_shift(const float *pend, const float *X, int o, int rows, float *Xm, hipStream_t s);

// ---- per-row log-sum-exp pieces of a vocab shard: out[m] = {max, sum exp(x - max)}
void row_lse(const float *logits, int n, int M, float *out, hipStream_t s);

}  // namespace qw::gpu
