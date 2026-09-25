/* src/core/shard.c — uneven layer-split planner (exact linear partition).
 *
 * Two cost notions, kept strictly separate (they used to be conflated, which
 * made the decode model read ALL 512 experts per token instead of the 11
 * active ones):
 *
 *   RESIDENT bytes  (qw_layer_bytes): total weights STORED on a GPU. Drives
 *   VRAM fit and the partitioner's objective. Legitimately counts all 512
 *   routed experts + shared, regardless of which are active.
 *
 *   TOUCHED bytes/decode token (qw_layer_decode_bytes): weights READ per
 *   token at batch 1. Only the moe_experts_per_tok routed experts + shared
 *   expert are touched; inactive expert weights never hit memory. Plus the
 *   attention/GDN projection reads, QSA per-token KV-cache READ traffic
 *   (grows with context; eventually dominates), and GDN recurrent-state
 *   read+write traffic. Drives decode tok/s.
 *
 * Partition: contiguous runs, minimize max per-GPU RESIDENT bytes, exact via
 * binary search on budget + greedy feasibility. Decode model: single stream
 * serializes across GPUs, so per-GPU bandwidth times ADD (no speedup for one
 * stream).
 */
#include "qw/shard.h"
#include "qw/macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QW_GPU_BW_BPS 512000000000.0  /* 512 GB/s per V620 (GDDR6) */
/* Short reference context for the 3-arg decode-bytes signature. Decode
 * traffic that depends on context (QSA KV reads) is evaluated here; the
 * context-dependent ceiling uses qw_layer_decode_bytes_at() at the true
 * context. */
#define QW_DECODE_SHORT_CTX 1024
/* V620 (gfx1030) peak rates: fp16 40.55 TFLOPS, fp32 20.28 TFLOPS.
 * int8 estimated at 2x fp16: v_dot4_i32_i8 issues 4 int8 MACs per
 * lane-instruction against fp16's 2 (v_fma_f16). */
#define QW_FP16_TFLOPS 40.55
#define QW_FP32_TFLOPS 20.28
#define QW_INT8_TFLOPS 81.10
#define QW_PREFILL_EFF 0.25 /* realistic GEMM efficiency assumption */

/* Overflow-checked multiply. */
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
    case QW_INT4:   return (n + 1) / 2;
    default:        return 0;
    }
}

/* ------------------------------------------------------- layer cost model */
uint64_t qw_layer_bytes(const qw_model_cfg *cfg, int layer_idx)
{
    if (cfg == NULL || layer_idx < 0 || layer_idx >= cfg->n_layers)
        return 0;
    const qw_dtype_policy *pol = &cfg->dtype_policy;
    int H = cfg->hidden_size;
    uint64_t b = 0, t = 0;

    if (cfg->layer_types[layer_idx] == QW_KIND_QSA) {
        /* Q proj: H x (n_heads x head_dim) */
        t = (uint64_t)H * (uint64_t)cfg->qsa_n_heads *
            (uint64_t)cfg->qsa_head_dim;
        b += qw_elem_bytes(t, pol->attention_proj);
        /* K + V proj: H x (n_kv_heads x head_dim) each */
        t = (uint64_t)H * (uint64_t)cfg->qsa_n_kv_heads *
            (uint64_t)cfg->qsa_head_dim;
        b += qw_elem_bytes(t, pol->attention_proj) * 2;
        /* O proj: (n_heads x head_dim) x H */
        t = (uint64_t)cfg->qsa_n_heads * (uint64_t)cfg->qsa_head_dim *
            (uint64_t)H;
        b += qw_elem_bytes(t, pol->attention_proj);
    } else {
        /* GDN linear attention: k-proj, v-proj, output proj */
        t = (uint64_t)H * (uint64_t)cfg->gdn_n_k_heads *
            (uint64_t)cfg->gdn_head_dim_k;
        b += qw_elem_bytes(t, pol->attention_proj);
        t = (uint64_t)H * (uint64_t)cfg->gdn_n_v_heads *
            (uint64_t)cfg->gdn_head_dim_v;
        b += qw_elem_bytes(t, pol->attention_proj);
        t = (uint64_t)cfg->gdn_n_v_heads * (uint64_t)cfg->gdn_head_dim_v *
            (uint64_t)H;
        b += qw_elem_bytes(t, pol->attention_proj);
    }

    /* MoE block (dominant): RESIDENT counts ALL experts — 512 routed +
     * shared — because every expert weight lives in VRAM on some GPU. */
    if (qw_umul((uint64_t)cfg->moe_num_experts, 3ULL, &t) == QW_OK) {
        if (qw_umul(t, (uint64_t)H * (uint64_t)cfg->moe_intermediate, &t) ==
            QW_OK)
            b += qw_elem_bytes(t, pol->moe_expert);
    }
    if (qw_umul((uint64_t)cfg->moe_num_shared, 3ULL, &t) == QW_OK) {
        if (qw_umul(t, (uint64_t)H *
                    (uint64_t)cfg->shared_expert_intermediate, &t) == QW_OK)
            b += qw_elem_bytes(t, pol->moe_shared);
    }
    return b;
}

/* Attention / GDN projection READ bytes per decode token. Same projections
 * as qw_layer_bytes (every one is read once per token) — isolated so the
 * MoE term can differ: resident counts all experts, decode counts only the
 * active ones. */
static uint64_t qw_layer_proj_bytes(const qw_model_cfg *cfg, int layer_idx)
{
    const qw_dtype_policy *pol = &cfg->dtype_policy;
    int H = cfg->hidden_size;
    uint64_t b = 0, t = 0;
    if (cfg->layer_types[layer_idx] == QW_KIND_QSA) {
        t = (uint64_t)H * (uint64_t)cfg->qsa_n_heads *
            (uint64_t)cfg->qsa_head_dim;
        b += qw_elem_bytes(t, pol->attention_proj);
        t = (uint64_t)H * (uint64_t)cfg->qsa_n_kv_heads *
            (uint64_t)cfg->qsa_head_dim;
        b += qw_elem_bytes(t, pol->attention_proj) * 2;
        t = (uint64_t)cfg->qsa_n_heads * (uint64_t)cfg->qsa_head_dim *
            (uint64_t)H;
        b += qw_elem_bytes(t, pol->attention_proj);
    } else {
        t = (uint64_t)H * (uint64_t)cfg->gdn_n_k_heads *
            (uint64_t)cfg->gdn_head_dim_k;
        b += qw_elem_bytes(t, pol->attention_proj);
        t = (uint64_t)H * (uint64_t)cfg->gdn_n_v_heads *
            (uint64_t)cfg->gdn_head_dim_v;
        b += qw_elem_bytes(t, pol->attention_proj);
        t = (uint64_t)cfg->gdn_n_v_heads * (uint64_t)cfg->gdn_head_dim_v *
            (uint64_t)H;
        b += qw_elem_bytes(t, pol->attention_proj);
    }
    return b;
}

/* Per-token KV cache READ traffic of one QSA layer, in bytes:
 *   2 (K and V) * n_kv_heads * head_dim * 2 B (fp16 cache) * ctx_tokens
 * The current token attends over the whole context, so this grows linearly
 * with ctx. Over all 12 QSA layers the per-token KV read is
 * 12 * 2 * 2 * 256 * 2 = 24576 B/token per token-of-context, i.e. at
 * ctx = 4096 a single QSA layer re-reads ~2 MB of KV per token.
 *
 * Per-token GDN recurrent-state read+write traffic of one GDN layer,
 * fp32 state (n_v_heads matrices of dk x dv), read once and written once
 * per token: n_v_heads * dk * dv * 4 B * 2. Context-independent (that is
 * the point of linear attention).
 *
 * (public wrappers: qw_layer_decode_bytes_at / qw_layer_decode_bytes). */
static uint64_t qw_layer_decode_core(const qw_model_cfg *cfg, int layer_idx,
                                     int ctx_tokens);

/* Public decode model, context-explicit: TOUCHED weight/state bytes of one
 * layer per decode token at batch 1 with context length ctx_tokens. This is
 * the single source of truth for decode traffic; the 3-arg
 * qw_layer_decode_bytes() delegates here at QW_DECODE_SHORT_CTX. */
uint64_t qw_layer_decode_bytes_at(const qw_model_cfg *cfg, int layer_idx,
                                  qw_dtype dtype, int ctx_tokens)
{
    if (cfg == NULL || layer_idx < 0 || layer_idx >= cfg->n_layers)
        return 0;
    if (ctx_tokens < 0)
        ctx_tokens = 0;
    /* Force every stored tensor to the single dtype requested, so the
     * caller can compare int4 vs int8 vs fp16 decode traffic. The
     * projections and experts are the only dtype-dependent reads here
     * (KV cache is fp16, GDN state fp32 — fixed by the architecture). */
    qw_model_cfg c = *cfg;
    c.dtype_policy.attention_proj = dtype;
    c.dtype_policy.dense_proj     = dtype;
    c.dtype_policy.moe_expert     = dtype;
    c.dtype_policy.moe_shared     = dtype;
    return qw_layer_decode_core(&c, layer_idx, ctx_tokens);
}

static uint64_t qw_layer_decode_core(const qw_model_cfg *cfg, int layer_idx,
                                     int ctx_tokens)
{
    const qw_dtype_policy *pol = &cfg->dtype_policy;
    uint64_t b = qw_layer_proj_bytes(cfg, layer_idx);

    /* MoE TOUCHED bytes: only the routed experts the top-k router selected
     * (moe_experts_per_tok) plus the shared expert are read. Inactive
     * expert weights stay in VRAM (counted by qw_layer_bytes) but cost
     * zero decode traffic. */
    uint64_t n_active = (uint64_t)cfg->moe_experts_per_tok +
                        (uint64_t)cfg->moe_num_shared;
    uint64_t t = 0;
    if (qw_umul(n_active, 3ULL, &t) == QW_OK) {
        if (qw_umul(t, (uint64_t)cfg->hidden_size *
                       (uint64_t)cfg->moe_intermediate, &t) == QW_OK)
            b += qw_elem_bytes(t, pol->moe_expert);
    }
    if (qw_umul((uint64_t)cfg->moe_num_shared, 3ULL, &t) == QW_OK) {
        if (qw_umul(t, (uint64_t)cfg->hidden_size *
                    (uint64_t)cfg->shared_expert_intermediate, &t) == QW_OK)
            b += qw_elem_bytes(t, pol->moe_shared);
    }

    if (cfg->layer_types[layer_idx] == QW_KIND_QSA) {
        /* KV cache READ over the whole context (fp16 KV). */
        t = (2ULL * (uint64_t)cfg->qsa_n_kv_heads *
             (uint64_t)cfg->qsa_head_dim * 2ULL) *
            (uint64_t)ctx_tokens;
        b += t;
    } else {
        /* GDN recurrent state: read + write (fp32). */
        t = (uint64_t)cfg->gdn_n_v_heads * (uint64_t)cfg->gdn_state_dk *
            (uint64_t)cfg->gdn_state_dv * 4ULL * 2ULL;
        b += t;
    }
    return b;
}

/* Public decode model (spec signature): TOUCHED weight/state bytes of one
 * layer per decode token at batch 1, at a short reference context. Only the
 * moe_experts_per_tok routed + moe_num_shared experts are read (drives
 * decode tok/s), unlike qw_layer_bytes which counts all 512 experts for VRAM
 * fit. For a specific context use qw_layer_decode_bytes_at(). */
uint64_t qw_layer_decode_bytes(const qw_model_cfg *cfg, int layer_idx,
                                qw_dtype dtype)
{
    return qw_layer_decode_bytes_at(cfg, layer_idx, dtype,
                                    QW_DECODE_SHORT_CTX);
}

/* ------------------------------------------------------------------ free */
void qw_plan_free(qw_plan *plan)
{
    if (plan == NULL) return;
    free(plan->placement);
    plan->placement = NULL;
    plan->placement_cap = 0;
    plan->n_placement = 0;
    memset(plan, 0, sizeof(*plan));
}

/* ------------------------------------------------------------ compute */
qw_err qw_plan_compute(const qw_model_cfg *cfg, int n_gpu,
                       const uint64_t *layer_bytes, qw_plan *out)
{
    if (cfg == NULL || layer_bytes == NULL || out == NULL)
        return QW_ERR_NULL;
    if (qw_model_validate(cfg) != QW_OK)
        return QW_ERR_FORMAT;
    if (n_gpu < 1 || n_gpu > QW_MAX_GPUS)
        return QW_ERR_RANGE;
    int n_layers = cfg->n_layers;
    if (n_layers < 1)
        return QW_ERR_RANGE;
    if (n_gpu > n_layers)
        return QW_ERR_RANGE; /* each GPU needs >=1 layer */

    memset(out, 0, sizeof(*out));
    out->n_gpu = n_gpu;
    out->n_layers = n_layers;

    /* Objective = RESIDENT weight bytes (VRAM fit). The partitioner never
     * looks at decode-touched bytes: decode traffic is not a fit constraint
     * and using it here would be the exact conflation this file exists to
     * undo. */
    uint64_t total = 0;
    for (int i = 0; i < n_layers; i++)
        total += layer_bytes[i];
    out->total_bytes = total;

    /* trivial: single GPU */
    if (n_gpu == 1) {
        out->layers_per_gpu[0] = n_layers;
        out->first_layer[0] = 0;
        out->bytes_per_gpu[0] = total;
        out->max_gpu_bytes = total;
        out->gpu_time_sec[0] = (double)total / QW_GPU_BW_BPS;
        out->total_time_sec = out->gpu_time_sec[0];
        out->tok_per_sec = out->total_time_sec > 0.0 ?
                           1.0 / out->total_time_sec : 0.0;
        out->balanced = true;
        return QW_OK;
    }

    /* binary search on max per-GPU byte budget.
     * lo = ceil(total / n_gpu) (lower bound: avg), hi = total.
     * Feasibility(budget): greedily pack contiguous layers into n_gpu runs,
     * each run sum <= budget, >=1 layer per GPU. If the greedy uses <= n_gpu
     * runs with all n_layers covered and no run forced to split a layer,
     * feasible. Since runs must be contiguous and each layer atomic, the
     * greedy (start a new run when adding the next layer would exceed
     * budget) yields the MINIMUM number of runs for that budget; feasible
     * iff min_runs <= n_gpu. We also need each of the n_gpu GPUs to get a
     * run, which is possible iff min_runs <= n_gpu <= n_layers. */
    uint64_t lo = total / (uint64_t)n_gpu;
    if (total % (uint64_t)n_gpu != 0) lo++;
    uint64_t hi = total;

    /* greedy: min number of contiguous runs with each run sum <= budget */
    int feasible(uint64_t budget)
    {
        int runs = 1;
        uint64_t cur = 0;
        for (int i = 0; i < n_layers; i++) {
            if (layer_bytes[i] > budget)
                return 0; /* single layer exceeds budget */
            if (cur + layer_bytes[i] > budget) {
                runs++;
                cur = layer_bytes[i];
            } else {
                cur += layer_bytes[i];
            }
        }
        return runs <= n_gpu;
    }

    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (feasible(mid))
            hi = mid;
        else
            lo = mid + 1;
    }
    uint64_t budget = lo;

    /* now actually assign n_gpu contiguous runs minimizing max, each <= budget.
     * We split into exactly n_gpu runs: greedy fill from the left, starting a
     * new run when the current run would exceed budget, ensuring we end with
     * <= n_gpu runs. To get exactly n_gpu runs (every GPU >=1 layer) we then
     * merge the surplus by... actually the budget guarantees <= n_gpu runs;
     * to use exactly n_gpu runs we split the largest runs to fill. Simpler:
     * do a greedy that targets n_gpu runs by forcing n_gpu-1 split points.
     *
     * Exact approach: the optimal min-max value = budget (found above). Now
     * place n_gpu-1 cut points such that each segment <= budget. Greedy from
     * left: accumulate, cut before layer i when adding it would exceed budget
     * AND we still need more runs. This yields a valid partition with each
     * segment <= budget and exactly n_gpu runs (because budget is feasible
     * with <= n_gpu runs, we can always extend to exactly n_gpu by splitting
     * any segment of >=2 layers). */
    {
        int cuts[QW_MAX_GPUS]; /* start index of each run */
        int n_runs = 0;
        uint64_t cur = 0;
        cuts[n_runs++] = 0;
        for (int i = 0; i < n_layers; i++) {
            if (i > 0 && cur + layer_bytes[i] > budget) {
                cuts[n_runs++] = i;
                cur = layer_bytes[i];
            } else {
                cur += layer_bytes[i];
            }
        }
        /* n_runs <= n_gpu. If n_runs < n_gpu, we must split runs to reach
         * exactly n_gpu. Split the longest runs (by layer count) until we
         * have n_gpu runs. Each split keeps segments <= budget (a segment of
         * a run is <= the run <= budget). */
        while (n_runs < n_gpu) {
            /* find run with most layers (>=2) to split at its midpoint */
            int best = -1, best_len = 1;
            for (int r = 0; r < n_runs; r++) {
                int end = (r + 1 < n_runs) ? cuts[r + 1] : n_layers;
                int len = end - cuts[r];
                if (len > best_len) { best_len = len; best = r; }
            }
            if (best < 0)
                break; /* cannot split further (all runs len 1) */
            int mid = cuts[best] + best_len / 2;
            /* insert cut at mid */
            for (int r = n_runs; r > best; r--)
                cuts[r] = cuts[r - 1];
            cuts[best + 1] = mid;
            n_runs++;
        }
        if (n_runs != n_gpu)
            return QW_ERR_RANGE; /* safety: shouldn't happen */

        for (int g = 0; g < n_gpu; g++) {
            int s = cuts[g];
            int e = (g + 1 < n_gpu) ? cuts[g + 1] : n_layers;
            out->first_layer[g] = s;
            out->layers_per_gpu[g] = e - s;
            uint64_t sum = 0;
            for (int i = s; i < e; i++)
                sum += layer_bytes[i];
            out->bytes_per_gpu[g] = sum;
            if (sum > out->max_gpu_bytes)
                out->max_gpu_bytes = sum;
            /* NOTE: gpu_time_sec/gpu_touched is a RESIDENT-bytes figure —
             * the buggy all-experts decode cost. The true decode ceiling
             * lives in qw_decode_ceiling(); kept for ABI/back-compat. */
            out->gpu_time_sec[g] = (double)sum / QW_GPU_BW_BPS;
            out->total_time_sec += out->gpu_time_sec[g];
        }
        out->tok_per_sec = out->total_time_sec > 0.0 ?
                           1.0 / out->total_time_sec : 0.0;

        /* balance: spread within 10% of max.
         * balanced iff (mx - mn) <= 0.1 * mx  <=>  (mx - mn) * 10 <= mx / 10
         * in integer math: (mx - mn) * 100 <= mx * 10. */
        uint64_t mx = out->max_gpu_bytes;
        uint64_t mn = out->bytes_per_gpu[0];
        for (int g = 1; g < n_gpu; g++)
            if (out->bytes_per_gpu[g] < mn) mn = out->bytes_per_gpu[g];
        out->balanced = (mx == 0) || ((mx - mn) * 100ULL) <= (mx * 10ULL);
    }
    return QW_OK;
}

/* ------------------------------------------------------------ rooflines */
/* Per-layer prefill MACs: each linear projection contributes 2*MACs per
 * output element (one per weight); a projection with P weights => 2*P MACs.
 *  MoE: only the top-k routed + shared experts fire per token.
 *  QSA attention core: query over the whole context,
 *      2 * n_heads * head_dim * ctx MACs per (QK^T and softmax-V) matmul.
 *  GDN: recurrent update over the fixed state matrix:
 *      2 * n_v_heads * head_dim_v * head_dim_k MACs per token. */
/* Per-layer prefill FLOPs for one token: 2 FLOPs (1 mul + 1 add) per
 * weight of every linear layer that fires, plus the attention core.
 *  MoE: only the top-k routed + shared experts fire per token.
 *  QSA attention core: two matmuls over the whole context (QK^T and
 *      softmax-V), each n_heads * head_dim * ctx MACs.
 *  GDN: recurrent update over the fixed state matrix:
 *      n_v_heads * head_dim_v * head_dim_k MACs per token.
 * Returns FLOPs directly (already 2*MACs) so the caller does NOT multiply
 * by 2 again. */
static uint64_t qw_layer_prefill_flops(const qw_model_cfg *cfg,
                                       int layer_idx, uint64_t ctx)
{
    int H = cfg->hidden_size;
    uint64_t f = 0, t = 0;
    if (cfg->layer_types[layer_idx] == QW_KIND_QSA) {
        /* Q/K/V/O projections: only the 12 QSA full-attention layers have
         * these; the 36 GDN layers use the k/v/out projections below. */
        if (qw_umul((uint64_t)H, (uint64_t)cfg->qsa_n_heads *
                        (uint64_t)cfg->qsa_head_dim, &t) == QW_OK)
            f += 2 * t; /* Q proj: 2 FLOP/weight */
        if (qw_umul((uint64_t)H, (uint64_t)cfg->qsa_n_kv_heads *
                        (uint64_t)cfg->qsa_head_dim, &t) == QW_OK)
            f += 2 * t * 2; /* K + V proj */
        if (qw_umul((uint64_t)cfg->qsa_n_heads * (uint64_t)cfg->qsa_head_dim,
                    (uint64_t)H, &t) == QW_OK)
            f += 2 * t; /* O proj */

        /* attention core: two matmuls, each n_heads*head_dim*ctx MACs =>
         * 2 * n_heads * head_dim * ctx MACs = 4 * that in FLOPs. */
        t = (2ULL * (uint64_t)cfg->qsa_n_heads *
             (uint64_t)cfg->qsa_head_dim) * ctx;
        f += 2 * t;
    } else {
        /* GDN projections: k, v, out */
        if (qw_umul((uint64_t)H, (uint64_t)cfg->gdn_n_k_heads *
                        (uint64_t)cfg->gdn_head_dim_k, &t) == QW_OK)
            f += 2 * t;
        if (qw_umul((uint64_t)H, (uint64_t)cfg->gdn_n_v_heads *
                        (uint64_t)cfg->gdn_head_dim_v, &t) == QW_OK)
            f += 2 * t;
        if (qw_umul((uint64_t)cfg->gdn_n_v_heads *
                        (uint64_t)cfg->gdn_head_dim_v,
                    (uint64_t)H, &t) == QW_OK)
            f += 2 * t;
        /* recurrent update: n_v_heads matrices of dk x dv, 2 FLOP/MAC */
        t = (uint64_t)cfg->gdn_n_v_heads * (uint64_t)cfg->gdn_state_dk *
            (uint64_t)cfg->gdn_state_dv;
        f += 2 * t;
    }

    /* MoE: top-k routed experts fire, 3 matrices H x inter each; shared
     * expert 3 matrices H x shared_inter. 2 FLOP/weight. */
    t = (uint64_t)cfg->moe_experts_per_tok * 3ULL *
        (uint64_t)H * (uint64_t)cfg->moe_intermediate;
    f += 2 * t;
    t = (uint64_t)cfg->moe_num_shared * 3ULL * (uint64_t)H *
        (uint64_t)cfg->shared_expert_intermediate;
    f += 2 * t;
    return f;
}

/* Model prefill roofline (compute-bound ceiling), all layers, per token.
 * flops = 2 * MACs (one multiply + one add per MAC); tok/s =
 * peak_tflops * eff * 1e12 / flops. Arithmetic intensity in FLOP/B of
 * stored weights lets the caller see compute-bound vs bandwidth-bound.
 * The dtype argument picks the peak rate: fp16 40.55 TFLOPS (dequantized
 * compute dtype on gfx1030), fp32 20.28, int8 estimated 2x fp16. */
void qw_compute_prefill_roofline(const qw_model_cfg *cfg, int dtype, uint64_t ctx,
                         qw_prefill_roofline *out)
{
    memset(out, 0, sizeof(*out));
    if (cfg == NULL || out == NULL)
        return;
    uint64_t flops = 0, wbytes = 0;
    for (int i = 0; i < cfg->n_layers; i++) {
        flops += qw_layer_prefill_flops(cfg, i, ctx);
        wbytes += qw_layer_bytes(cfg, i);
    }
    out->flops_per_token = flops; /* already 2*MACs */
    out->bytes_per_token = wbytes;
    double tf = 0.0;
    if (dtype == QW_INT8)        tf = QW_INT8_TFLOPS;
    else if (dtype == QW_F32)    tf = QW_FP32_TFLOPS;
    else                         tf = QW_FP16_TFLOPS;
    out->peak_tflops = tf;
    out->arithmetic_intensity =
        out->flops_per_token > 0 ?
        (double)out->flops_per_token / (double)out->bytes_per_token : 0.0;
    /* NOTE: 1e12 must be double — as an int literal tf * 1e12 overflows
     * 32-bit int (UB, yields ~124) and the ceiling comes out 1e10x low. */
    double rate = tf * 1e12 * QW_PREFILL_EFF;
    out->tok_per_sec =
        out->flops_per_token > 0 ?
        rate / (double)out->flops_per_token : 0.0;
}

/* Decode (batch 1) ceiling at a given context: per-token TOUCHED bytes
 * (active experts + projections + KV/state reads) over all layers.
 * Single stream => GPUs serialize, so the per-GPU per-token bytes ADD;
 * ceiling = 1 / sum(bytes_g / bw) = n_gpu * bw / total_touched.
 * Context enters only through the QSA KV-read term, so the ceiling
 * DEGRADES with ctx (eventually KV traffic dominates expert weights). */
float qw_decode_ceiling_at(const qw_plan *plan, const qw_model_cfg *cfg,
                           int ctx_tokens)
{
    if (plan == NULL || cfg == NULL)
        return 0.0f;
    uint64_t total_touched = 0;
    for (int g = 0; g < plan->n_gpu; g++) {
        for (int i = plan->first_layer[g];
             i < plan->first_layer[g] + plan->layers_per_gpu[g]; i++)
            total_touched += qw_layer_decode_core(cfg, i, ctx_tokens);
    }
    return (float)(QW_GPU_BW_BPS * (double)plan->n_gpu /
                   (double)total_touched);
}

/* --------------------------------------------------------------- report */
/* Print the full decode/prefill model for one plan.
 *
 * WHY 3 vs 4 GPUs is NOT a single-stream speed question: with a layer
 * split one decode stream is serialized across GPUs — only one GPU works
 * at a time, so per-GPU times ADD and the ceiling is
 * n_gpu * 512 GB/s / total_touched, which is independent of n_gpu. More
 * GPUs buy VRAM capacity (more resident bytes fit) and CONCURRENCY:
 * at multiple streams/microbatches the pipeline overlap means aggregate
 * throughput scales with GPU count until some other resource (VRAM,
 * bandwidth aggregate, host) saturates. So the 3-vs-4 decision is
 * VRAM-fit + max concurrency, not single-stream tok/s. */
void qw_plan_report_full(const qw_plan *plan, const qw_model_cfg *cfg,
                         int ctx_tokens, int dtype)
{
    if (plan == NULL) return;
    fprintf(stderr, "== shard plan: %d gpus, %d layers ==\n", plan->n_gpu,
            plan->n_layers);
    uint64_t usable = 32ULL * 1024ULL * 1024ULL * 1024ULL * 88ULL / 100ULL;
    uint64_t total_touched = 0;
    for (int g = 0; g < plan->n_gpu; g++) {
        fprintf(stderr, "  gpu%d: layers [%d..%d) count=%d bytes=", g,
                plan->first_layer[g],
                plan->first_layer[g] + plan->layers_per_gpu[g],
                plan->layers_per_gpu[g]);
        /* GiB print */
        uint64_t b = plan->bytes_per_gpu[g];
        uint64_t gib = b / (1024ULL * 1024ULL * 1024ULL);
        uint64_t frac = ((b % (1024ULL * 1024ULL * 1024ULL)) * 100ULL) /
                        (1024ULL * 1024ULL * 1024ULL);
        fprintf(stderr, "%llu.%02lu GiB (resident, fits 12%%-headroom 32GiB: %s)\n",
                (unsigned long long)gib, (unsigned long)frac,
                b <= usable ? "yes" : "NO");
        for (int i = plan->first_layer[g];
             i < plan->first_layer[g] + plan->layers_per_gpu[g]; i++)
            if (cfg != NULL)
                total_touched += qw_layer_decode_core(cfg, i, ctx_tokens);
    }
    /* Decode ceiling from TOUCHED bytes (active experts only) */
    float dec = qw_decode_ceiling_at(plan, cfg, ctx_tokens);
    fprintf(stderr,
            "  decode: touched=%llu.%02lu MiB/token  ceiling=%.1f tok/s "
            "(ctx=%d)\n",
            (unsigned long long)(total_touched / (1024ULL * 1024ULL)),
            (unsigned long)(((total_touched % (1024ULL * 1024ULL)) * 100ULL) /
                            (1024ULL * 1024ULL)),
            (double)dec, ctx_tokens);
    /* Prefill roofline */
    if (cfg != NULL) {
        qw_prefill_roofline pr;
        qw_compute_prefill_roofline(cfg, dtype, (uint64_t)ctx_tokens, &pr);
        fprintf(stderr,
                "  prefill: %.2f TFLOPS/token  AI=%.2f FLOP/B  "
                "ceiling=%.1f tok/s (eff=%.0f%%)\n",
                (double)pr.flops_per_token / 1e12, pr.arithmetic_intensity,
                pr.tok_per_sec, QW_PREFILL_EFF * 100.0);
    }
    fprintf(stderr,
            "  layer split does NOT speed a single decode stream "
            "(times ADD); it raises VRAM capacity and throughput at "
            "concurrency > 1\n");
}

/* -------------------------------------------------------------- apply */
qw_err qw_plan_apply(const qw_plan *plan)
{
    (void)plan;
    /* TODO(phase 2): wire plan to real device tensors — allocate per-GPU
     * buffers, H2D/D2D copy weights per placement, set up ring buffers. */
    return QW_ERR_NOTIMPL;
}
