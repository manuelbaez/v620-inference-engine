/* src/core/config.c — model defaults, validation, overrides, memory model.
 *
 * All byte accounting is uint64_t with overflow-checked multiplies
 * (qw_umul). Byte sizes for each dtype: fp16 = 2 B, int8 = 1 B, int4 = 0.5 B
 * per element (packed 2 per byte), fp8 (n-gram table) = 1 B.
 */
#include "qw/config.h"
#include "qw/macros.h"

#include <stdio.h>
#include <string.h>

/* Overflow-checked multiply: *r = a*b, QW_ERR_RANGE on overflow. */
static qw_err qw_umul(uint64_t a, uint64_t b, uint64_t *r)
{
    if (a != 0 && b > UINT64_MAX / a)
        return QW_ERR_RANGE;
    *r = a * b;
    return QW_OK;
}

/* Storage bytes for n elements at dtype dt (int4 packs 2 per byte). */
static uint64_t qw_elem_bytes(uint64_t n, qw_dtype dt)
{
    switch (dt) {
    case QW_F32:    return n * 4;
    case QW_F16:
    case QW_BF16:   return n * 2;
    case QW_INT8:
    case QW_UINT8:  return n;
    case QW_INT4:   return (n + 1) / 2; /* 2 int4 per byte */
    default:        return 0;
    }
}

/* ------------------------------------------------------------------ cfg */
qw_model_cfg qw_model_qwen38_flash_next_default(void)
{
    qw_model_cfg c;
    memset(&c, 0, sizeof(c));

    /* name: Qwen3.8-Flash-Next */
    static const char *name = "Qwen/Qwen3.8-Flash-Next";
    size_t n = strlen(name);
    if (n > sizeof(c.name) - 1) n = sizeof(c.name) - 1;
    memcpy(c.name, name, n);

    c.n_layers     = 48;
    /* repeating 4-layer pattern: GDN,GDN,GDN,QSA. Full-attention at 3,7,...47
     * => 36 GDN + 12 QSA. */
    for (int i = 0; i < c.n_layers; i++)
        c.layer_types[i] = ((i % 4) == 3) ? QW_KIND_QSA : QW_KIND_GDN;

    c.hidden_size  = 2560;
    c.vocab_size   = 248320; /* padded */

    /* QSA (full attention) */
    c.qsa_n_heads    = 24;
    c.qsa_n_kv_heads = 2;    /* GQA 12:1 */
    c.qsa_head_dim   = 256;
    c.rope_rotated   = 64;   /* partial_rotary_factor 0.25 x 256 */
    c.max_position   = 262144;
    c.yarn_scale     = 4.0;
    c.yarn_orig_pos  = 1000000;

    /* GDN (linear attention, no kv cache) */
    c.gdn_n_v_heads  = 48;
    c.gdn_n_k_heads  = 16;
    c.gdn_head_dim_v = 128;
    c.gdn_head_dim_k = 128;
    c.gdn_state_dk   = c.gdn_head_dim_k; /* recurrent state width */
    c.gdn_state_dv   = c.gdn_head_dim_v;

    /* MoE: 512 routed + 1 shared, top-10 router */
    c.moe_num_experts            = 512;
    c.moe_num_shared             = 1;
    c.moe_experts_per_tok        = 10;
    c.moe_intermediate           = 640;
    c.shared_expert_intermediate = 640;

    /* n-gram PLE table: layer 2, host-resident fp8, ~47.7 GiB total
     * (20,000,000 entries, entry row = 248320 B fp8). Held in host RAM,
     * read per token by gather — never on GPU. */
    c.ngram_layer   = 2;
    c.ngram_entries = 20000000;
    /* 47.7 GiB = 47.7 * 1024^3 B (host-resident fp8 PLE table) */
    c.ngram_bytes   = (uint64_t)(47.7 * 1073741824.0); /* 51,217,485,004 B */

    c.has_mtp = true; /* ~4B-param multi-token prediction module */

    /* dtype policy: int4 everywhere (dequant in-kernel to fp16),
     * embeddings fp16, n-gram table pinned fp8-host. */
    c.dtype_policy.attention_proj = QW_INT4;
    c.dtype_policy.dense_proj     = QW_INT4;
    c.dtype_policy.moe_expert     = QW_INT4;
    c.dtype_policy.moe_shared     = QW_INT4;
    c.dtype_policy.embeddings     = QW_F16;
    c.dtype_policy.ngram_table    = QW_UINT8; /* fp8 */
    return c;
}

/* ---------------------------------------------------------------- validate */
qw_err qw_model_validate(const qw_model_cfg *cfg)
{
    if (cfg == NULL)
        return QW_ERR_NULL;
    if (cfg->n_layers < 1 || cfg->n_layers > QW_MAX_LAYERS)
        return QW_ERR_RANGE;
    if (cfg->hidden_size <= 0 || cfg->vocab_size <= 0)
        return QW_ERR_RANGE;

    /* full-attention indices must be exactly 3,7,11,...,(4n-1) */
    for (int i = 0; i < cfg->n_layers; i++) {
        bool expect_qsa = ((i % 4) == 3);
        if (cfg->layer_types[i] == QW_KIND_QSA && !expect_qsa)
            return QW_ERR_FORMAT;
        if (cfg->layer_types[i] == QW_KIND_GDN && expect_qsa)
            return QW_ERR_FORMAT;
        if (cfg->layer_types[i] != QW_KIND_GDN &&
            cfg->layer_types[i] != QW_KIND_QSA)
            return QW_ERR_FORMAT;
    }

    /* MoE: routed experts must exceed active experts */
    if (cfg->moe_num_experts <= cfg->moe_experts_per_tok)
        return QW_ERR_FORMAT;
    if (cfg->moe_num_shared < 0 || cfg->moe_intermediate <= 0 ||
        cfg->shared_expert_intermediate <= 0)
        return QW_ERR_RANGE;

    /* GDN state shape consistent with head dims */
    if (cfg->gdn_state_dk != cfg->gdn_head_dim_k ||
        cfg->gdn_state_dv != cfg->gdn_head_dim_v)
        return QW_ERR_FORMAT;
    if (cfg->gdn_n_v_heads <= 0 || cfg->gdn_n_k_heads <= 0 ||
        cfg->gdn_head_dim_v <= 0 || cfg->gdn_head_dim_k <= 0)
        return QW_ERR_RANGE;

    /* QSA head-dim consistency where applicable:
     * n_kv_heads must divide n_heads (GQA grouping). */
    if (cfg->qsa_n_heads <= 0 || cfg->qsa_n_kv_heads <= 0 ||
        cfg->qsa_head_dim <= 0)
        return QW_ERR_RANGE;
    if (cfg->qsa_n_heads % cfg->qsa_n_kv_heads != 0)
        return QW_ERR_FORMAT;

    /* n-gram table pinned fp8-host */
    if (cfg->dtype_policy.ngram_table != QW_UINT8)
        return QW_ERR_FORMAT;

    return QW_OK;
}

/* ---------------------------------------------------------------- overrides */
qw_err qw_model_apply_overrides(qw_model_cfg *cfg, int n_layers_override,
                                int n_gpu_override)
{
    if (cfg == NULL)
        return QW_ERR_NULL;
    if (n_layers_override < 1 || n_layers_override > QW_MAX_LAYERS)
        return QW_ERR_RANGE;
    if (n_gpu_override < 1 || n_gpu_override > QW_MAX_GPUS)
        return QW_ERR_RANGE;

    /* Preserve the repeating 4-layer kind pattern so full-attention stays at
     * 3,7,11,... (correct even for a 2-layer slice, where neither is full). */
    cfg->n_layers = n_layers_override;
    for (int i = 0; i < cfg->n_layers; i++)
        cfg->layer_types[i] = ((i % 4) == 3) ? QW_KIND_QSA : QW_KIND_GDN;

    /* n_gpu_override is advisory for the planner; the planner takes n_gpu
     * as a separate argument. We do not store it in the cfg (no field). */
    (void)n_gpu_override;
    return QW_OK;
}

/* --------------------------------------------------------- memory model */

/* Per-layer weight bytes. GDN and QSA differ in projection shapes; MoE
 * dominates (512 experts x 3 matrices x H x I). Returns 0 + sets *err.
 *
 * QSA layer (attention): Q/K/V/O projections.
 *   Q proj: H x (n_heads x head_dim)
 *   K proj: H x (n_kv_heads x head_dim)
 *   V proj: H x (n_kv_heads x head_dim)
 *   O proj: (n_heads x head_dim) x H
 * GDN layer (linear attention): no KV cache; in/out projections sized by
 *   k/v head counts and head dims (k-proj and v-proj + output).
 * Both layer kinds include the same MoE block (dominant term).
 */
static qw_err qw_layer_weight_bytes(const qw_model_cfg *cfg, int idx,
                                    qw_dtype_policy pol, uint64_t *out)
{
    uint64_t b = 0, t;
    int H = cfg->hidden_size;

    if (cfg->layer_types[idx] == QW_KIND_QSA) {
        /* Q proj: H x (n_heads x head_dim) */
        if (qw_umul((uint64_t)H, (uint64_t)cfg->qsa_n_heads *
                    (uint64_t)cfg->qsa_head_dim, &t) != QW_OK)
            return QW_ERR_RANGE;
        b += qw_elem_bytes(t, pol.attention_proj);
        /* K + V proj: H x (n_kv_heads x head_dim) each */
        if (qw_umul((uint64_t)H, (uint64_t)cfg->qsa_n_kv_heads *
                    (uint64_t)cfg->qsa_head_dim, &t) != QW_OK)
            return QW_ERR_RANGE;
        b += qw_elem_bytes(t, pol.attention_proj) * 2; /* K and V */
        /* O proj: (n_heads x head_dim) x H */
        if (qw_umul((uint64_t)cfg->qsa_n_heads * (uint64_t)cfg->qsa_head_dim,
                    (uint64_t)H, &t) != QW_OK)
            return QW_ERR_RANGE;
        b += qw_elem_bytes(t, pol.attention_proj);
    } else {
        /* GDN linear attention: k-proj, v-proj, output proj.
         * k proj: H x (n_k_heads x head_dim_k)
         * v proj: H x (n_v_heads x head_dim_v)
         * out proj: (n_v_heads x head_dim_v) x H */
        if (qw_umul((uint64_t)H, (uint64_t)cfg->gdn_n_k_heads *
                    (uint64_t)cfg->gdn_head_dim_k, &t) != QW_OK)
            return QW_ERR_RANGE;
        b += qw_elem_bytes(t, pol.attention_proj);
        if (qw_umul((uint64_t)H, (uint64_t)cfg->gdn_n_v_heads *
                    (uint64_t)cfg->gdn_head_dim_v, &t) != QW_OK)
            return QW_ERR_RANGE;
        b += qw_elem_bytes(t, pol.attention_proj);
        if (qw_umul((uint64_t)cfg->gdn_n_v_heads *
                    (uint64_t)cfg->gdn_head_dim_v, (uint64_t)H, &t) != QW_OK)
            return QW_ERR_RANGE;
        b += qw_elem_bytes(t, pol.attention_proj);
    }

    /* MoE block (identical across layer kinds) — the dominant term:
     * routed: num_experts x 3 matrices x H x inter
     * shared: num_shared x 3 matrices x H x shared_inter */
    if (qw_umul((uint64_t)cfg->moe_num_experts, 3ULL, &t) != QW_OK)
        return QW_ERR_RANGE;
    if (qw_umul(t, (uint64_t)H * (uint64_t)cfg->moe_intermediate, &t) != QW_OK)
        return QW_ERR_RANGE;
    b += qw_elem_bytes(t, pol.moe_expert);
    if (qw_umul((uint64_t)cfg->moe_num_shared, 3ULL, &t) != QW_OK)
        return QW_ERR_RANGE;
    if (qw_umul(t, (uint64_t)H * (uint64_t)cfg->shared_expert_intermediate,
                &t) != QW_OK)
        return QW_ERR_RANGE;
    b += qw_elem_bytes(t, pol.moe_shared);

    *out = b;
    return QW_OK;
}

qw_err qw_model_estimate(const qw_model_cfg *cfg, int n_gpu, int ctx_tokens,
                         int n_seqs, qw_mem_estimate *out)
{
    if (cfg == NULL || out == NULL)
        return QW_ERR_NULL;
    if (qw_model_validate(cfg) != QW_OK)
        return QW_ERR_FORMAT;
    if (n_gpu < 1 || n_gpu > QW_MAX_GPUS)
        return QW_ERR_RANGE;
    if (ctx_tokens < 0 || n_seqs < 0)
        return QW_ERR_RANGE;

    memset(out, 0, sizeof(*out));
    out->n_gpu      = n_gpu;
    out->ctx_tokens = (uint64_t)ctx_tokens;
    out->n_seqs     = n_seqs;

    /* ---- transformer weights at each dtype --------------------------
     * Sum per-layer weight bytes (GDN/QSA differ, MoE included per layer)
     * across all layers, plus embedding + output head (vocab x H). */
    uint64_t w_i4 = 0, w_i8 = 0, w_f16 = 0;
    for (int i = 0; i < cfg->n_layers; i++) {
        uint64_t b;
        qw_dtype_policy p4 = cfg->dtype_policy;
        p4.attention_proj = p4.dense_proj = p4.moe_expert =
            p4.moe_shared = QW_INT4;
        p4.embeddings     = QW_F16;
        p4.ngram_table    = QW_UINT8;
        if (qw_layer_weight_bytes(cfg, i, p4, &b) != QW_OK)
            return QW_ERR_RANGE;
        w_i4 += b;

        /* int8 variant */
        qw_dtype_policy p8 = p4;
        p8.attention_proj = p8.dense_proj = p8.moe_expert =
            p8.moe_shared = QW_INT8;
        if (qw_layer_weight_bytes(cfg, i, p8, &b) != QW_OK)
            return QW_ERR_RANGE;
        w_i8 += b;

        /* fp16 variant */
        qw_dtype_policy p16 = p4;
        p16.attention_proj = p16.dense_proj = p16.moe_expert =
            p16.moe_shared = p16.embeddings = QW_F16;
        if (qw_layer_weight_bytes(cfg, i, p16, &b) != QW_OK)
            return QW_ERR_RANGE;
        w_f16 += b;
    }
    /* embedding + output head: vocab x H (embeddings dtype) */
    {
        uint64_t t;
        if (qw_umul((uint64_t)cfg->vocab_size, (uint64_t)cfg->hidden_size,
                    &t) != QW_OK)
            return QW_ERR_RANGE;
        uint64_t e_i4  = qw_elem_bytes(t, QW_INT4);
        uint64_t e_i8  = qw_elem_bytes(t, QW_INT8);
        uint64_t e_f16 = qw_elem_bytes(t, QW_F16);
        w_i4  += e_i4 + e_i4;   /* input emb + output head */
        w_i8  += e_i8 + e_i8;
        w_f16 += e_f16 + e_f16;
    }
    out->total_weights_int4  = w_i4;
    out->total_weights_int8  = w_i8;
    out->total_weights_fp16  = w_f16;

    /* ---- KV cache (QSA layers only; GDN has no KV cache) ------------
     * KV bytes = n_full * 2(K,V) * n_kv_heads * head_dim * 2 B(fp16) * ctx
     * Per token:   n_full * 2 * n_kv_heads * head_dim * 2 B             */
    int n_full = 0, n_gdn = 0;
    for (int i = 0; i < cfg->n_layers; i++) {
        if (cfg->layer_types[i] == QW_KIND_QSA) n_full++;
        else n_gdn++;
    }
    {
        uint64_t per_tok = (uint64_t)n_full * 2ULL *
                           (uint64_t)cfg->qsa_n_kv_heads *
                           (uint64_t)cfg->qsa_head_dim * 2ULL;
        uint64_t kv;
        if (qw_umul(per_tok, (uint64_t)ctx_tokens, &kv) != QW_OK)
            return QW_ERR_RANGE;
        out->kv_per_token = per_tok;
        out->kv_cache_bytes = kv;
    }

    /* ---- GDN recurrent state (per sequence, fp32) -------------------
     * Each GDN layer keeps a per-value-head state matrix of shape
     * (dk x dv) per sequence. GDN state bytes =
     *   n_gdn * n_v_heads * dk * dv * sizeof(float) * n_seqs
     * = n_gdn * 48 * 128 * 128 * 4 * n_seqs. */
    {
        uint64_t per_seq = (uint64_t)n_gdn * (uint64_t)cfg->gdn_n_v_heads *
                           (uint64_t)cfg->gdn_state_dk *
                           (uint64_t)cfg->gdn_state_dv * 4ULL;
        uint64_t st;
        if (qw_umul(per_seq, (uint64_t)n_seqs, &st) != QW_OK)
            return QW_ERR_RANGE;
        out->gdn_state_bytes = st;
    }

    /* ---- host-resident n-gram PLE table (fp8) ----------------------- */
    out->ngram_table_bytes = cfg->ngram_bytes;
    out->host_ram_bytes    = cfg->ngram_bytes; /* + working set (small) */

    /* ---- per-GPU split (uneven layer runs) --------------------------
     * Use int4 weights (default policy). Split total transformer weights
     * n-ways; the exact per-GPU split comes from the planner. Here we give
     * a uniform per-GPU share for the fit check (conservative upper bound
     * per GPU = max possible one GPU holds). We approximate by even split;
     * the shard planner produces the true uneven distribution. */
    uint64_t per_gpu_w = w_i4 / (uint64_t)n_gpu;
    if (w_i4 % (uint64_t)n_gpu != 0) per_gpu_w++; /* round up */
    for (int g = 0; g < n_gpu; g++) {
        out->per_gpu_bytes[g] = per_gpu_w;
        out->per_gpu_total[g] = per_gpu_w + out->kv_cache_bytes +
                                out->gdn_state_bytes;
    }

    /* ---- 32 GiB fit with 12% safety margin --------------------------
     * 32 GiB = 32 * 1024^3. Reserve 12% for activations/graphs/workspace.
     * Usable = 32 GiB * 0.88. fits if per_gpu_total[g] <= usable for all g. */
    uint64_t giB32 = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    out->safety_margin = (giB32 * 12ULL) / 100ULL;
    uint64_t usable = giB32 - out->safety_margin;
    out->fits_32gib = 1;
    for (int g = 0; g < n_gpu; g++)
        if (out->per_gpu_total[g] > usable)
            out->fits_32gib = 0;

    return QW_OK;
}

/* ---------------------------------------------------------------- dump */
static void qw_print_gib(uint64_t bytes)
{
    /* print GiB with 2 decimals using integer math to avoid float issues */
    uint64_t gib = bytes / (1024ULL * 1024ULL * 1024ULL);
    uint64_t rem = bytes % (1024ULL * 1024ULL * 1024ULL);
    uint64_t frac = (rem * 100ULL) / (1024ULL * 1024ULL * 1024ULL);
    fprintf(stderr, "%llu.%02lu GiB", (unsigned long long)gib,
            (unsigned long)frac);
}

void qw_model_dump(const qw_model_cfg *cfg, const qw_mem_estimate *est)
{
    if (cfg == NULL) return;
    fprintf(stderr, "== model: %s ==\n", cfg->name);
    fprintf(stderr, "layers=%d hidden=%d vocab=%d\n", cfg->n_layers,
            cfg->hidden_size, cfg->vocab_size);
    fprintf(stderr, "qsa heads=%d kv=%d dim=%d rope_rot=%d maxpos=%d "
                    "yarn=%.1f/%d\n", cfg->qsa_n_heads, cfg->qsa_n_kv_heads,
            cfg->qsa_head_dim, cfg->rope_rotated, cfg->max_position,
            cfg->yarn_scale, cfg->yarn_orig_pos);
    fprintf(stderr, "gdn v_heads=%d k_heads=%d dv=%d dk=%d state=%dx%d\n",
            cfg->gdn_n_v_heads, cfg->gdn_n_k_heads, cfg->gdn_head_dim_v,
            cfg->gdn_head_dim_k, cfg->gdn_state_dk, cfg->gdn_state_dv);
    fprintf(stderr, "moe experts=%d shared=%d topk=%d inter=%d "
                    "shared_inter=%d mtp=%d\n", cfg->moe_num_experts,
            cfg->moe_num_shared, cfg->moe_experts_per_tok,
            cfg->moe_intermediate, cfg->shared_expert_intermediate,
            (int)cfg->has_mtp);
    if (est == NULL) return;
    fprintf(stderr, "-- memory estimate (ctx=%llu seqs=%d gpu=%d) --\n",
            (unsigned long long)est->ctx_tokens, est->n_seqs, est->n_gpu);
    fprintf(stderr, "weights int4: ");  qw_print_gib(est->total_weights_int4);  fprintf(stderr, "\n");
    fprintf(stderr, "weights int8: ");  qw_print_gib(est->total_weights_int8);  fprintf(stderr, "\n");
    fprintf(stderr, "weights fp16: ");  qw_print_gib(est->total_weights_fp16);  fprintf(stderr, "\n");
    fprintf(stderr, "kv/token: %llu B  kv total: ",
            (unsigned long long)est->kv_per_token);
    qw_print_gib(est->kv_cache_bytes); fprintf(stderr, "\n");
    fprintf(stderr, "gdn state: ");  qw_print_gib(est->gdn_state_bytes);  fprintf(stderr, "\n");
    fprintf(stderr, "ngram table (host, fp8): ");
    qw_print_gib(est->ngram_table_bytes); fprintf(stderr, "\n");
    fprintf(stderr, "safety margin/GPU: "); qw_print_gib(est->safety_margin);
    fprintf(stderr, "  fits_32gib=%d\n", est->fits_32gib);
    for (int g = 0; g < est->n_gpu; g++) {
        fprintf(stderr, "  gpu%d: weights=", g);
        qw_print_gib(est->per_gpu_bytes[g]);
        fprintf(stderr, " total=");
        qw_print_gib(est->per_gpu_total[g]);
        fprintf(stderr, "\n");
    }
}
