/* src/core/gdn.c — GDN recurrent state: layout + one-step delta-rule
 * reference. C17, libc only (CPU reference; no HIP). See qw/gdn.h for the
 * layout and the formula this file implements.
 */
#include "qw/gdn.h"
#include "qw/macros.h"

#include <stdlib.h>
#include <string.h>

/* Overflow-checked multiply: *r = a*b, QW_ERR_RANGE on overflow. */
static qw_err gdn_umul(uint64_t a, uint64_t b, uint64_t *r)
{
    if (a != 0 && b > UINT64_MAX / a)
        return QW_ERR_RANGE;
    *r = a * b;
    return QW_OK;
}

/* Count GDN layers (non-QSA) in the cfg's per-layer kind array. */
static int gdn_count_gdn(const qw_model_cfg *cfg)
{
    int n = 0;
    for (int i = 0; i < cfg->n_layers; i++)
        if (cfg->layer_types[i] == QW_KIND_GDN)
            n++;
    return n;
}

/* --------------------------------------------------------------- state */
qw_err qw_gdn_state_bytes(const qw_model_cfg *cfg, size_t n_seq,
                          size_t n_gdn_layers, uint64_t *out)
{
    if (cfg == NULL || out == NULL)
        return QW_ERR_NULL;
    if (n_seq < 1 || n_gdn_layers < 1)
        return QW_ERR_RANGE;
    if (n_gdn_layers > (size_t)gdn_count_gdn(cfg))
        return QW_ERR_RANGE;

    uint64_t per_block = (uint64_t)cfg->gdn_state_dk *
                         (uint64_t)cfg->gdn_state_dv * 4ULL; /* fp32 */
    uint64_t per_layer_seq;
    if (gdn_umul(per_block, (uint64_t)cfg->gdn_n_v_heads, &per_layer_seq)
            != QW_OK)
        return QW_ERR_RANGE;
    uint64_t per_seq;
    if (gdn_umul(per_layer_seq, (uint64_t)n_gdn_layers, &per_seq) != QW_OK)
        return QW_ERR_RANGE;
    uint64_t total;
    if (gdn_umul(per_seq, (uint64_t)n_seq, &total) != QW_OK)
        return QW_ERR_RANGE;
    *out = total;
    return QW_OK;
}

qw_err qw_gdn_state_alloc(const qw_model_cfg *cfg, size_t n_seq,
                          size_t n_gdn_layers, qw_gdn_state *out)
{
    if (cfg == NULL || out == NULL)
        return QW_ERR_NULL;

    memset(out, 0, sizeof(*out));
    uint64_t bytes;
    qw_err e = qw_gdn_state_bytes(cfg, n_seq, n_gdn_layers, &bytes);
    if (e != QW_OK)
        return e;

    out->data = malloc(bytes ? bytes : 1);
    if (out->data == NULL) {
        memset(out, 0, sizeof(*out));
        return QW_ERR_ALLOC;
    }
    memset(out->data, 0, bytes);
    out->n_layers  = n_gdn_layers;
    out->n_seq     = n_seq;
    out->n_heads   = (size_t)cfg->gdn_n_v_heads;
    out->dk        = (size_t)cfg->gdn_state_dk;
    out->dv        = (size_t)cfg->gdn_state_dv;
    out->owns_data = true;
    return QW_OK;
}

void qw_gdn_state_free(qw_gdn_state *s)
{
    if (s == NULL)
        return;
    if (s->owns_data && s->data != NULL)
        free(s->data);
    memset(s, 0, sizeof(*s));
}

qw_err qw_gdn_state_zero(qw_gdn_state *s)
{
    if (s == NULL || s->data == NULL)
        return QW_ERR_NULL;
    memset(s->data, 0, s->n_layers * s->n_seq * s->n_heads *
                            s->dk * s->dv * sizeof(float));
    return QW_OK;
}

float *qw_gdn_state_at(const qw_gdn_state *s, int layer, size_t seq,
                       size_t head)
{
    if (s == NULL || s->data == NULL)
        return NULL;
    if (s->n_heads == 0 || s->dk == 0 || s->dv == 0)
        return NULL;
    if (layer < 0 || (size_t)layer >= s->n_layers)
        return NULL;
    if (seq >= s->n_seq || head >= s->n_heads)
        return NULL;
    /* [layer][seq][head] block; dk*dv floats per block, row-major. */
    size_t off = ((size_t)layer * s->n_seq + seq) * s->n_heads *
                 s->dk * s->dv + (size_t)head * s->dk * s->dv;
    return s->data + off;
}

qw_err qw_gdn_state_copy(const qw_gdn_state *dst, int dst_layer,
                         size_t dst_seq, const qw_gdn_state *src,
                         int src_layer, size_t src_seq, size_t head)
{
    if (dst == NULL || src == NULL)
        return QW_ERR_NULL;
    if (dst->dk != src->dk || dst->dv != src->dv)
        return QW_ERR_FORMAT;

    float *d = qw_gdn_state_at(dst, dst_layer, dst_seq, head);
    const float *s = qw_gdn_state_at(src, src_layer, src_seq, head);
    if (d == NULL || s == NULL)
        return QW_ERR_RANGE;

    /* Same-size blocks; overlapping dst==src handled by memmove. */
    memmove(d, s, dst->dk * dst->dv * sizeof(float));
    return QW_OK;
}

/* ------------------------------------------------------ one-step update */
qw_err qw_gdn_update_block(float *S, const float *k, const float *v,
                           const float *g, const float *beta,
                           const float *q, float *out,
                           size_t dk, size_t dv)
{
    if (S == NULL || k == NULL || v == NULL || g == NULL ||
        beta == NULL || q == NULL || out == NULL)
        return QW_ERR_NULL;
    if (dk == 0 || dv == 0)
        return QW_ERR_RANGE;

    const float decay = *g;    /* scalar decay per step */
    const float lr    = *beta; /* scalar learning rate per step */

    /* S <- S * g : decay the whole (dk x dv) state in place. */
    for (size_t i = 0; i < dk; i++)
        for (size_t j = 0; j < dv; j++)
            S[i * dv + j] *= decay;

    /* u <- v - S^T k : prediction error, one per dv column.
     * (S^T k)[d] = sum_i S[i][d] * k[i]; subtract from v[d]. */
    float *u = malloc(dv * sizeof(float));
    if (u == NULL)
        return QW_ERR_ALLOC;
    for (size_t d = 0; d < dv; d++) {
        float acc = 0.0f;
        for (size_t i = 0; i < dk; i++)
            acc += S[i * dv + d] * k[i];
        u[d] = v[d] - acc;
    }

    /* S <- S + beta * (k outer u) : rank-1 write-back.
     * (k outer u)[i][d] = k[i] * u[d]. */
    for (size_t i = 0; i < dk; i++)
        for (size_t d = 0; d < dv; d++)
            S[i * dv + d] += lr * k[i] * u[d];
    free(u);

    /* o <- q^T S : readout, one per dv column.
     * (q^T S)[d] = sum_i q[i] * S[i][d]. */
    for (size_t d = 0; d < dv; d++) {
        float acc = 0.0f;
        for (size_t i = 0; i < dk; i++)
            acc += q[i] * S[i * dv + d];
        out[d] = acc;
    }

    return QW_OK;
}

/* ------------------------------------------------------- chunked update */
qw_err qw_gdn_update_sliding_window(float *S, const float *k, const float *v,
                                    const float *g, const float *beta,
                                    const float *q, float *out,
                                    size_t T, size_t dk, size_t dv)
{
    if (S == NULL || k == NULL || v == NULL || g == NULL ||
        beta == NULL || q == NULL || out == NULL)
        return QW_ERR_NULL;
    if (dk == 0 || dv == 0)
        return QW_ERR_RANGE;

    /* T sequential one-step updates; S carries across timesteps in place.
     * k_t/v_t/q_t are token-major: token t at offset t*len. g_t/beta_t are
     * t scalars each. out_t is t*len of readouts. This is the prefill
     * tiling identity — must match T qw_gdn_update_block() calls. */
    for (size_t t = 0; t < T; t++) {
        qw_err e = qw_gdn_update_block(
            S,
            k + t * dk,
            v + t * dv,
            g + t,
            beta + t,
            q + t * dk,
            out + t * dv,
            dk, dv);
        if (e != QW_OK)
            return e;
    }
    return QW_OK;
}
