/* qw/shard.h — uneven layer-split planner. Pure C17.
 *
 * Pipeline/layer split across GPUs (llama.cpp --split-mode layer style):
 * each GPU owns a CONTIGUOUS run of ordered layers; no tensor parallelism
 * (the model has only 2 KV heads — tp>1 would replicate KV heads and waste
 * VRAM, plus no NVLink/P2P). The partition minimizes max per-GPU bytes
 * (classic linear-partition / multiprocessor-scheduling), solved exactly via
 * binary search on the per-GPU budget + greedy feasibility check.
 *
 * With a layer split, a single decode stream is serialized across GPUs:
 * one GPU is active at a time, so per-GPU bandwidth times ADD. No bandwidth
 * speedup — the headline tok/s ceiling is 1 / sum(per-gpu time).
 */
#ifndef QW_SHARD_H
#define QW_SHARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/config.h"
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QW_MAX_GPUS 8

/* Per-tensor placement entry. POD: opaque tensor id + GPU + bytes. The
 * caller owns the id-space and resolves ids to real tensors via a lookup
 * callback of its own (keeps the plan decoupled from the tensor table). */
typedef struct qw_placement {
    uint32_t tensor_id;
    int16_t  gpu;
    uint64_t bytes;
} qw_placement;

/* One contiguous-run partition of the ordered layer sequence. */
typedef struct qw_plan {
    int         n_gpu;
    int         n_layers;
    int         layers_per_gpu[QW_MAX_GPUS]; /* run length per GPU */
    int         first_layer[QW_MAX_GPUS];    /* prefix sum: start index  */
    uint64_t    bytes_per_gpu[QW_MAX_GPUS];  /* resident weight bytes   */
    uint64_t    total_bytes;
    uint64_t    max_gpu_bytes;

    /* per-GPU decode model */
    double      gpu_time_sec[QW_MAX_GPUS]; /* bytes / 512e9 B/s */
    double      total_time_sec;             /* sum: times add (1 stream) */
    double      tok_per_sec;                /* 1 / total_time_sec */
    bool        balanced;                   /* spread within 10% */
    int         n_placement;                /* filled placement entries */
    int         placement_cap;              /* capacity of placement[] */
    qw_placement *placement;                /* caller frees with free() */
} qw_plan;

void qw_plan_free(qw_plan *plan);

/* Weight bytes of one layer under the cfg's dtype policy, in BYTES.
 * GDN and QSA layers cost differently (attention vs linear-attention
 * projections); MoE dominates (512 experts x 3 matrices x H x I). */
uint64_t qw_layer_bytes(const qw_model_cfg *cfg, int layer_idx);

/* Partition n_layers ordered layers into n_gpu contiguous runs minimizing
 * max per-GPU bytes. EXACT: binary search on the per-GPU byte budget with a
 * greedy feasibility check (every run <= budget, >=1 layer per GPU, runs in
 * order). layer_bytes[i] is the resident weight bytes of layer i.
 *
 * Errors: NULL args, n_gpu < 1 or > QW_MAX_GPUS, n_layers < 1, or
 * n_gpu > n_layers (each GPU needs >=1 layer). n_gpu == 1 is trivial.
 */
qw_err qw_plan_compute(const qw_model_cfg *cfg, int n_gpu,
                       const uint64_t *layer_bytes, qw_plan *out);

/* Print per-GPU bytes, predicted tok/s, and an unbalanced-split warning
 * (>10% spread) to stderr. */
void qw_plan_report(const qw_plan *plan);

/* Placeholder: apply a plan to real device tensors (phase 2). */
qw_err qw_plan_apply(const qw_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* QW_SHARD_H */
