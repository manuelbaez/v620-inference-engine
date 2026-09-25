/* qw/config.h — model description, memory model, overrides. Pure C17.
 *
 * Describes Qwen/Qwen3.8-Flash-Next (the one supported model, hardcoded
 * defaults) and provides byte-exact memory estimates. All byte arithmetic
 * uses uint64_t; float appears only for tok/s reporting in qw/shard.h.
 */
#ifndef QW_CONFIG_H
#define QW_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- constants */
#define QW_MAX_LAYERS   64   /* hard cap on the per-layer type array */
#define QW_MAX_GPUS     8    /* max GPUs for estimates/plans */
#define QW_DEFAULT_CTX  4096 /* default context token count for estimates */
#define QW_DEFAULT_SEQS  8   /* default concurrent sequences for estimates */

/* Per-layer kind. The model alternates in a repeating 4-layer pattern:
 * 3x GDN (gated delta net, linear attention, no KV cache) then 1x QSA
 * (full attention). Full-attention layers sit at indices 3,7,11,...,47. */
typedef enum qw_layer_kind {
    QW_KIND_GDN = 0,
    QW_KIND_QSA = 1,
} qw_layer_kind;

/* Storage dtype per tensor class. RDNA2 (gfx1030) has no bf16/fp8/fp4
 * hardware; int4/int8 weights dequantize in-kernel to fp16 (the only
 * hardware compute dtype). The n-gram PLE table is always fp8 on host. */
typedef struct qw_dtype_policy {
    qw_dtype attention_proj;  /* QKV projections of QSA layers */
    qw_dtype dense_proj;      /* dense linear layers (norms/linear) */
    qw_dtype moe_expert;      /* routed MoE expert weights */
    qw_dtype moe_shared;      /* shared expert weights */
    qw_dtype embeddings;      /* input embedding / final output head */
    qw_dtype ngram_table;     /* pinned to QW_UINT8 (fp8); validated */
} qw_dtype_policy;

/* -------------------------------------------------------------- model cfg */
typedef struct qw_model_cfg {
    char         name[64];

    /* backbone */
    int          n_layers;
    qw_layer_kind layer_types[QW_MAX_LAYERS];

    int          hidden_size;
    int          vocab_size;      /* padded vocab, 248320 */

    /* QSA (full attention) — applies to every QW_KIND_QSA layer */
    int          qsa_n_heads;     /* 24 query heads */
    int          qsa_n_kv_heads;  /* 2 KV heads (GQA 12:1) */
    int          qsa_head_dim;    /* 256 */
    int          rope_rotated;    /* dims rotated by RoPE: head_dim * partial factor = 64 */
    int          max_position;    /* 262144 */
    double       yarn_scale;      /* 4.0 */
    int          yarn_orig_pos;   /* YaRN interpolation length: 1,000,000 */

    /* GDN (linear attention) — applies to every QW_KIND_GDN layer */
    int          gdn_n_v_heads;   /* 48 value heads */
    int          gdn_n_k_heads;   /* 16 key heads */
    int          gdn_head_dim_v;  /* 128 */
    int          gdn_head_dim_k;  /* 128 */

    /* recurrent state shape per sequence (fp32, kept per sequence):
     * one head matrix per value head of shape (dk x dv). */
    int          gdn_state_dk;    /* = gdn_head_dim_k */
    int          gdn_state_dv;    /* = gdn_head_dim_v */

    /* MoE (one per layer): num_experts routed experts + 1 shared expert,
     * router picks top num_experts_per_tok per token. */
    int          moe_num_experts;        /* 512 routed */
    int          moe_num_shared;         /* 1 shared */
    int          moe_experts_per_tok;    /* 10 active per token */
    int          moe_intermediate;       /* 640 per routed expert */
    int          shared_expert_intermediate; /* 640 */

    /* n-gram PLE table: layer 2, host-resident fp8, read per token. */
    int          ngram_layer;          /* 2 */
    uint64_t     ngram_entries;        /* 20,000,000 */
    uint64_t     ngram_bytes;          /* host-resident table size, fp8 */

    bool         has_mtp;              /* multi-token prediction module (~4B) */

    qw_dtype_policy dtype_policy;
} qw_model_cfg;

/* Fully-populated defaults for Qwen/Qwen3.8-Flash-Next.
 * dtype policy: everything int4 (fits 3x32 or 4x32 GiB VRAM), embeddings
 * fp16 (dequant cost not worth it for 248320x2560 = 0.6 GiB), n-gram table
 * always fp8-host. */
qw_model_cfg qw_model_qwen38_flash_next_default(void);

/* Consistency checks: layer array length matches n_layers; full-attention
 * indices are exactly 3,7,...,47 (n_layers/4 full layers, one every 4th
 * starting at index 3); routed experts > active experts; GDN state shape
 * consistent with head dims; vocab/hidden positive; dtype policy sane. */
qw_err qw_model_validate(const qw_model_cfg *cfg);

/* Override n_layers and n_gpus for debugging (tiny 2-layer slice, single
 * GPU). n_layers in [1, QW_MAX_LAYERS]; n_gpus in [1, 8]. Preserves the
 * repeating layer-kind pattern (full-attention at 3,7,... stays correct).
 * n_gpus is stashed in cfg->n_layers for the planner's reference only when
 * 1 < n_gpus <= 8 (planner takes n_gpu separately). */
qw_err qw_model_apply_overrides(qw_model_cfg *cfg, int n_layers_override,
                                int n_gpu_override);

/* --------------------------------------------------------- memory model */
typedef struct qw_mem_estimate {
    uint64_t total_weights_int4; /* transformer weights, all int4  */
    uint64_t total_weights_int8; /* transformer weights, all int8  */
    uint64_t total_weights_fp16; /* transformer weights, all fp16  */
    uint64_t kv_cache_bytes;     /* n_full * 2 * n_kv * head_dim * 2B * ctx */
    uint64_t kv_per_token;       /* per-token KV bytes (single token) */
    uint64_t gdn_state_bytes;    /* n_gdn * heads * dk * dv * 4B * n_seqs */
    uint64_t ngram_table_bytes;  /* host RAM, fp8 (not on GPU) */
    uint64_t host_ram_bytes;     /* ngram table + working set */
    uint64_t per_gpu_bytes[8];   /* weights split n-ways (uneven) */
    uint64_t per_gpu_total[8];   /* weights + kv + gdn state, per GPU */
    int      n_gpu;
    int      fits_32gib;         /* per-GPU total incl. 12% safety margin */
    uint64_t safety_margin;      /* 12% of 32 GiB per GPU */
    uint64_t ctx_tokens;
    int      n_seqs;
} qw_mem_estimate;

/* Compute the memory model. ctx_tokens and n_seqs are the deployment
 * assumptions the estimate is built for. n_gpu must be 1..8. */
qw_err qw_model_estimate(const qw_model_cfg *cfg, int n_gpu,
                         int ctx_tokens, int n_seqs,
                         qw_mem_estimate *out);

/* Dump cfg + estimate to stderr. */
void qw_model_dump(const qw_model_cfg *cfg, const qw_mem_estimate *est);

/* ---------------------------------------------------- decode/prefill model */
/* Forward decls so this header is self-contained for callers of the model
 * functions implemented in qw/shard.h (src/core/shard.c). */
struct qw_plan;
typedef struct qw_plan qw_plan;

/* Per-layer decode TOUCHED bytes at batch 1, context-explicit (drives
 * decode tok/s), for a layer whose stored tensors are all dtype `dtype`
 * (e.g. QW_INT4 for the default policy). Counts only the moe_experts_per_tok
 * routed experts + moe_num_shared experts (NOT all 512), the attention/GDN
 * projections, the per-token QSA KV-cache READ at ctx_tokens (fp16, grows
 * linearly with context), and the GDN recurrent-state read+write. Single
 * source of truth for decode traffic. Distinct from qw_layer_bytes, which is
 * RESIDENT bytes (all 512 experts) and drives VRAM fit / the partitioner. */
uint64_t qw_layer_decode_bytes_at(const qw_model_cfg *cfg, int layer_idx,
                                  qw_dtype dtype, int ctx_tokens);

/* Per-layer decode TOUCHED bytes at a short reference context (1024 tokens),
 * for a layer whose stored tensors are all dtype `dtype`. Same model as
 * qw_layer_decode_bytes_at() — use the _at form when a specific context
 * matters (the KV-read term grows with context). */
uint64_t qw_layer_decode_bytes(const qw_model_cfg *cfg, int layer_idx,
                               qw_dtype dtype);

/* Prefill (compute-bound) roofline for the whole model, per token.
 * FLOPs = 2 * MACs, summed over layers (active-expert MoE + attention/GDN
 * core). tok_per_sec = peak_tflops * 0.25 efficiency / FLOPs per token;
 * arithmetic_intensity = FLOPs / resident weight bytes (FLOP/B) tells
 * whether prefill is compute-bound (>> bandwidth roofline) or not. */
typedef struct qw_prefill_roofline {
    uint64_t flops_per_token;      /* 2 * MACs, all layers */
    uint64_t bytes_per_token;      /* resident weight bytes (int4 stored) */
    double   arithmetic_intensity; /* FLOP / resident byte */
    double   peak_tflops;          /* per dtype: fp16 40.55, fp32 20.28,
                                     int8 81.10 (2x fp16, v_dot4_i32_i8) */
    double   tok_per_sec;          /* at 25% efficiency */
} qw_prefill_roofline;

/* Fill the prefill roofline. dtype picks the peak rate (QW_F16, QW_F32,
 * QW_INT8; anything else maps to fp16). ctx is the prefill sequence
 * length (attention core scales with it). */
void qw_compute_prefill_roofline(const qw_model_cfg *cfg, int dtype,
                                 uint64_t ctx, qw_prefill_roofline *out);

/* Decode (batch 1) ceiling in tok/s for a plan at context length ctx_tokens:
 * TOUCHED bytes per token (active experts + projections + QSA KV read at
 * ctx + GDN state rw) summed over the plan's layers; GPUs serialize so
 * ceiling = n_gpu * 512 GB/s / total_touched. DEGRADES as ctx grows (KV
 * read traffic grows linearly and eventually dominates). */
float qw_decode_ceiling_at(const struct qw_plan *plan, const qw_model_cfg *cfg,
                           int ctx_tokens);

/* Full decode+prefill report for a plan: per-GPU resident GiB with the
 * 12%-headroom 32 GiB verdict, decode touched-bytes/token + tok/s ceiling,
 * prefill roofline (rate, arithmetic intensity, 25%-eff ceiling), and the
 * note that layer split does not speed a single stream but does raise
 * VRAM capacity and concurrency throughput. Complements the
 * single-arg qw_plan_report(plan) in qw/shard.h. */
void qw_plan_report_full(const struct qw_plan *plan, const qw_model_cfg *cfg,
                         int ctx_tokens, int dtype);

#ifdef __cplusplus
}
#endif

#endif /* QW_CONFIG_H */
