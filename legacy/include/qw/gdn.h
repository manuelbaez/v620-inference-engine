/* qw/gdn.h — GDN (gated delta net) recurrent state: layout + one-step
 * delta-rule reference. Pure C17, libc only (CPU reference, no HIP).
 *
 * GDN is the LINEAR attention used by 36 of the 48 layers of
 * Qwen3.8-Flash-Next (the repeating GDN,GDN,GDN,QSA pattern; every layer
 * except the one every 4th at 3,7,...,47). These layers keep NO kv cache;
 * each carries a recurrent state per sequence that is updated in place,
 * one step per token.
 *
 * State: one (dk x dv) matrix of fp32 per VALUE head, per layer, per
 * sequence. The descriptor below is a POD view over a single flat
 * allocation; all shapes come from the model cfg (nothing hardcoded).
 */
#ifndef QW_GDN_H
#define QW_GDN_H

#include <stddef.h>
#include <stdint.h>

#include "qw/config.h"
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- state */
/* Memory layout of the flat GDN state buffer, all fp32, row-major:
 *
 *   [layer][seq][head][i * dv + j]
 *
 *   layer in [0, n_layers)   -- every GDN layer (QSA layers hold no state)
 *   seq   in [0, n_seq)
 *   head  in [0, n_v_heads)  -- per value head
 *   i     in [0, dk),  j in [0, dv)  -- one (dk x dv) head matrix
 *
 * The fastest axis is j (column of the head matrix), matching the
 * float* handed to qw_gdn_update_block(). Per layer/seq/head block:
 * dk * dv floats = sizeof(float) * dk * dv bytes (256 KiB at the default
 * 128 x 128 shape); per layer/seq: n_v_heads blocks (12.0 MiB default);
 * whole buffer: n_gdn_layers * n_seq * n_v_heads * dk * dv * 4 bytes. */
typedef struct qw_gdn_state {
    float *data;   /* owns_data ? allocated buffer : NULL */
    size_t n_layers;  /* GDN layers in this allocation */
    size_t n_seq;     /* sequences */
    size_t n_heads;   /* value heads per layer */
    size_t dk;        /* state rows per head */
    size_t dv;        /* state cols per head */
    bool   owns_data; /* true -> qw_gdn_state_free frees data */
} qw_gdn_state;

/* Total bytes for the flat buffer: n_gdn_layers * n_seq * n_v_heads *
 * dk * dv * sizeof(float). Overflow-checked (QW_ERR_RANGE). n_seq must
 * be >= 1; n_gdn_layers < 1 is an error. */
qw_err qw_gdn_state_bytes(const qw_model_cfg *cfg, size_t n_seq,
                          size_t n_gdn_layers, uint64_t *out);

/* Allocate + zero a full buffer for n_seq sequences x n_gdn_layers.
 * n_gdn_layers must be <= the number of GDN layers in the cfg. On error
 * *out is zeroed and QW_ERR_NULL / QW_ERR_ALLOC / QW_ERR_RANGE returned. */
qw_err qw_gdn_state_alloc(const qw_model_cfg *cfg, size_t n_seq,
                          size_t n_gdn_layers, qw_gdn_state *out);

void qw_gdn_state_free(qw_gdn_state *s);

/* Zero the whole buffer (n_seq * n_gdn_layers * n_v_heads blocks). */
qw_err qw_gdn_state_zero(qw_gdn_state *s);

/* Float* to the (dk x dv) block for a GDN layer index, sequence, head.
 * `layer` is the layer INDEX (0..cfg->n_layers-1); QSA indices are
 * rejected (QW_ERR_RANGE). */
float *qw_gdn_state_at(const qw_gdn_state *s, int layer, size_t seq,
                       size_t head);

/* Copy one (dk x dv) block: qw_gdn_state_at(dst, dst_layer, dst_seq, h)
 * <- qw_gdn_state_at(src, src_layer, src_seq, h). Same-cfg src/dst. */
qw_err qw_gdn_state_copy(const qw_gdn_state *dst, int dst_layer,
                         size_t dst_seq, const qw_gdn_state *src,
                         int src_layer, size_t src_seq, size_t head);

/* ------------------------------------------------------ one-step update */
/* ONE step of the GDN gated delta rule on a single (dk x dv) state block
 * (fp32, row-major, fastest axis = dv). Readable scalar reference — this
 * is the correctness oracle for the later HIP kernel, so clarity beats
 * speed. No VLAs, no C++-isms, all inputs validated.
 *
 *   S <- S * g                    (decay: every element scaled by g)
 *   u <- v - S^T k                (delta / prediction error: u[d] = v[d]
 *                                  - sum_i S[i*dv+d]*k[i])
 *   S <- S + beta * (k outer u)   (rank-1 write-back: S[i*dv+d] +=
 *                                  beta * k[i] * u[d])
 *   o <- q^T S                    (readout: o[d] = sum_i q[i]*S[i*dv+d])
 *
 * S is read-modify-written in place; q/k/v are length dk/dv/dk; g and
 * beta are scalar pointers (decay, learning rate); out is length dv. */
qw_err qw_gdn_update_block(float *S, const float *k, const float *v,
                           const float *g, const float *beta,
                           const float *q, float *out,
                           size_t dk, size_t dv);

/* ------------------------------------------------------- chunked update */
/* T sequential steps on ONE (dk x dv) block — the prefill tiling identity.
 * (The "sliding window" name is historical: no windowing happens; it is
 * simply T delta-rule steps tiled over a prefill chunk.)
 * k_t/v_t are contiguous, T blocks of dk/dv each (token-major); g_t/beta_t
 * are T scalars each; q_t is T blocks of dk; out_t is T blocks of dv.
 * Step t uses S (running), k_t..v_t, g_t, beta_t and emits readout out_t;
 * S is updated in place across all T steps.
 *
 * This MUST equal applying qw_gdn_update_block() T times with the same
 * per-token inputs — the tiled kernel is only correct if it matches that. */
qw_err qw_gdn_update_sliding_window(float *S, const float *k, const float *v,
                                    const float *g, const float *beta,
                                    const float *q, float *out,
                                    size_t T, size_t dk, size_t dv);

#ifdef __cplusplus
}
#endif

#endif /* QW_GDN_H */
