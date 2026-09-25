/* qw/router.h — MoE top-k expert router for Qwen3.8-Flash-Next. Pure C17.
 *
 * Each MoE layer (48 of them, 512 routed experts per layer) scores all
 * experts and keeps the top `k` (k = moe_experts_per_tok = 10) by score.
 * The 1 shared expert is always on and is NOT the router's job.
 *
 * The router runs 48 times per token, so the hot path (qw_router_topk) is
 * O(n) scan + O(k^2) fixup with no full sort of all n scores and no
 * allocation: scratch comes from the caller's arena (or the stack when
 * k <= 16 and scratch is NULL).
 *
 * Determinism is mandatory: on tied scores the LOWEST expert index wins,
 * so top-k selection is reproducible bit-for-bit across runs and
 * implementations. NaN never silently wins a comparison: non-finite input
 * is rejected up front, and the hot-path comparison is a strict > with
 * total-order tie-break (score, then lower index) so an in-band NaN would
 * still never be selected over a finite score.
 */
#ifndef QW_ROUTER_H
#define QW_ROUTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One selected expert: which expert and its normalized (softmax) weight. */
typedef struct qw_route {
    uint32_t expert;
    float    weight;
} qw_route;

/* Max k servable from the stack when scratch == NULL. */
enum { QW_ROUTER_STACK_K = 16 };

/* Select the k largest scores out of n_experts. out holds k routes SORTED
 * BY EXPERT INDEX ASCENDING (NOT by weight), each weight being
 * softmax(score) over ONLY the selected k (sums to 1 over out[0..k-1]).
 *
 * Deterministic: ties on score resolve to the lowest expert index.
 * Non-finite scores (NaN / inf) are rejected, not ranked.
 *
 * scratch: arena for working storage. May be NULL only when
 * k <= QW_ROUTER_STACK_K (stack buffer used instead). k is unbounded
 * when scratch != NULL. Returns QW_OK and fills out[0..k-1], or
 * QW_ERR_NULL (bad pointer), QW_ERR_RANGE (bad k/n_experts, or
 * scratch exhausted), QW_ERR_FORMAT (non-finite score). */
qw_err qw_router_topk(const float *scores, int n_experts, int k,
                      qw_route *out, void *scratch);

/* Batch load-balance counter. routes holds n_routes routes total
 * (e.g. 8 tokens x 10 = 80); increments counts[expert] per route.
 * counts must cover all 512 routed experts. */
qw_err qw_router_expert_load(const qw_route *routes, int n_routes,
                             uint32_t counts[512]);

/* Stable softmax in place: subtracts the running max first (so exp()
 * never overflows), then normalizes by the sum. n must be >= 1. */
qw_err qw_router_softmax_inplace(float *scores, int n);

#ifdef __cplusplus
}
#endif

#endif /* QW_ROUTER_H */
