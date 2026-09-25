// Prefill (chunked, T tokens of one sequence) kernels for one TP rank.
// Row-major, token-major buffers: X[t][s*640 + j] etc. `start` is the
// position of token 0 of the chunk. See docs/DESIGN.md "Prefill".
#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdint>

namespace qw::gpu {

constexpr int D_PAD = 328;   // HC down outputs (320 + 4 inject) padded to 16 bytes

// Y[t][n] = sum_k W[n][k] * X[t][k]. W fp16 [N][K]; X fp16 rows of ldx;
// Y fp16 (f16_out) or fp32 rows of ldy. fp16 output is ~5x faster on gfx1030.
void gemm(rocblas_handle h, const uint16_t *W, int N, int K, const uint16_t *X, int ldx, int T, void *Y,
          int ldy, bool f16_out);

// ---- HC (two-phase: ssq all-reduce, then the full down projection)
// Optionally X += pend * 2 sigmoid(inj/4) and X += add, then ssq[t][s].
void hc_pre_T(float *X, const float *pend, const float *inj, int T, float *ssq, hipStream_t s);
// xn[t][s][j] = fp16(X * rsqrt(ssq/H + eps) * w1)
void hc_norm_T(const float *X, const float *ssq, const float *w1, int T, uint16_t *xn, hipStream_t s);
// u = silu(d[0:320]/4) (fp16), inj = d[320:324]
void hc_mid_T(const float *d, int T, uint16_t *u, float *inj, hipStream_t s);
// bin[t][j] = mean_s sigmoid(g[t][j*4+s]) * xn[t][s*640+j]; g fp16 [T][2560]
void hc_mix_T(const uint16_t *g, const uint16_t *xn, int T, uint16_t *bin, hipStream_t s);
// X += pend * 2 sigmoid(inj/4)
void hc_combine_T(float *X, const float *pend, const float *inj, int T, hipStream_t s);
// X[t][s][j] = emb[t][j] for all s; emb fp32 [T][640]
void embed_T(float *X, const float *emb, int T, hipStream_t s);

// ---- GDN (local: 4 k-heads, 12 v-heads)
// P fp16 [T][4120] = qkv 2560 | z 1536 | b 12 | a 12.
// Writes post-conv, normalized q,k and v (fp32 [T][2560]) and per-token
// decay/beta [T][12][2]; reads the 3 previous pre-conv rows from the decode
// ring (slot = pos % 4).
void gdn_conv_T(const uint16_t *P, const float *ring, const float *conv_w, const float *A_log,
                const float *dt_bias, int64_t start, int T, float *qkv, float *gb, hipStream_t s);
// Copies the chunk's last pre-conv rows into the ring.
void gdn_ring_update_T(const uint16_t *P, float *ring, int64_t start, int T, hipStream_t s);
// Sequential delta rule over the chunk, parallel over (head, 32-column slice):
// o_raw fp32 [T][1536], state S [12][128][128] updated in place.
void gdn_scan_T(const float *qkv, const float *gb, float *S, int T, float *o_raw, hipStream_t s);
// out[t][h*128+j] = fp16(rmsnorm(o_raw[t][h]) * w[j] * sigmoid(z[t][h*128+j]))
void gdn_norm_T(const float *o_raw, const uint16_t *P, const float *norm_w, int T, uint16_t *out,
                hipStream_t s);

// ---- QSA (local: 6 q heads, 1 kv head; indexer replicated)
// P fp16 [T][4224]. Writes q16 [T][6][256], gate fp16 [T][1536], iq16
// [T][4][128] (normed, roped, fp16 for the score GEMM), K/V/raw_k caches.
void qsa_prep_T(const uint16_t *P, const float *qn, const float *kn, const float *iqn, int64_t start,
                int T, uint16_t *q16, uint16_t *gate, uint16_t *iq16, uint16_t *K, uint16_t *V,
                float *raw_k, hipStream_t s);
// Compressed keys for every group completed inside the chunk.
void qsa_compress_T(const float *raw_k, const float *ikn, int64_t start, int T, uint16_t *ck,
                    hipStream_t s);
// From per-head raw scores sc16 [Q][4][ldsc] (fp16, = iq_h . ck[c]) to
// scores [Q][ldsc] = sum_h relu / sqrt(128); queries are positions q0..q0+Q-1.
void qsa_score_reduce_T(const uint16_t *sc16, int ldsc, int64_t q0, int Q, float *scores, hipStream_t s);
// Token lists for queries q0..q0+Q-1: dense 0..pos when nb <= 512, else the
// radix-selected top-512 groups plus the tail. lists [Q][LIST_W], counts [Q].
constexpr int LIST_W = 2052;
void qsa_select_T(const float *scores, int ldsc, int64_t q0, int Q, int32_t *lists, int32_t *counts,
                  hipStream_t s);
// Attention for Q queries (q16/gate rows), partial scratch [Q][33][6][258].
void qsa_attend_T(const uint16_t *q16, const uint16_t *gate, const uint16_t *K, const uint16_t *V,
                  const int32_t *lists, const int32_t *counts, int Q, float *partial, uint16_t *out,
                  hipStream_t s);

// ---- MoE (local experts [first, first+128))
// Per token: top-10 of softmax(logits), renormalized, local ones kept.
// tok_e/tok_w [T][10] (local expert or -1), counts[128] (atomic, pre-zeroed).
void moe_route_T(const float *logits, int T, int first, int32_t *tok_e, float *tok_w, int32_t *counts,
                 hipStream_t s);
// offsets = exclusive scan(counts); slot of each (token, j) pair; pair_tok[slot] = t.
void moe_scatter_T(const int32_t *tok_e, const int32_t *counts, int T, int32_t *offsets, int32_t *fill,
                   int32_t *tok_slot, int32_t *pair_tok, hipStream_t s);
// Routed experts for every (token, expert) pair: h[slot] = silu(gate x) * up x,
// yp[slot] = down h (fp16). tiles: scratch of 2*(T*10/32 + 128) + 1 ints.
// With q8 (int8 scratch >= T*10*640 bytes) and q8s (>= T*10*5 floats), runs
// W4A8 on v_dot4_i32_i8 with per-128-group activation scales (lossy; see DESIGN).
void moe_experts_T(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                   const uint32_t *dw, const uint16_t *ds, const int32_t *counts, const int32_t *offsets,
                   const int32_t *pair_tok, const uint16_t *bin, int T, int32_t *tiles, uint16_t *h, uint16_t *yp,
                   hipStream_t s, int8_t *q8 = nullptr, float *q8s = nullptr);
// Same outputs as moe_experts_T (fp16 only) with GEMV-shaped kernels, one
// wave per (pair, row): faster for small T (decode batches).
void moe_experts_P(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                   const uint32_t *dw, const uint16_t *ds, const int32_t *offsets, const int32_t *pair_tok,
                   const uint16_t *bin, int T, uint16_t *h, uint16_t *yp, hipStream_t s);
// out[t][n] = fp16( sum_j tok_w[t][j] * yp[tok_slot[t][j]][n] + shared[t][n] )
void moe_combine_T(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w,
                   const uint16_t *shared, int T, uint16_t *out, hipStream_t s);
// h[t][i] = fp16(silu(v[t][i]) * v[t][n+i] * sigmoid(v[t][2n])), v rows of ldv
void swiglu_T(const float *v, int ldv, int n, int T, uint16_t *h, hipStream_t s);

// ---- PLE (layer 1). kv fp16 [T][3200]; send [T][16]
void ple_stats_T(const uint16_t *kv, const float *X, const float *nk, const float *nq, int T, float *send,
                 hipStream_t s);
// X += gv + silu(dilated_conv(norm_conv(gv))); ring [10][2560] by position.
void ple_apply_T(const uint16_t *kv, const float *red, const float *nc, const float *conv_w, float *ring,
                 int64_t start, int T, float *ncbuf, float *X, hipStream_t s);

void f16_to_f32_T(const uint16_t *x, float *y, size_t n, hipStream_t s);

}  // namespace qw::gpu
