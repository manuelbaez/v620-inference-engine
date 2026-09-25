/* src/core/router.c — MoE top-k expert router (CPU ref), see qw/router.h.
 *
 * O(n) single scan over the 512 scores plus an O(k^2) fixup on the k
 * selected; no full sort, no allocation in the hot path. Deterministic
 * tie-break: on equal scores the lower expert index wins.
 */
#include "qw/router.h"
#include "qw/macros.h"

#include <math.h>

/* Working slot: expert index + its score, kept for the fixup pass. */
typedef struct route_slot {
    uint32_t expert;
    float    score;
} route_slot;

/* Total-order "a ranks above b": higher score wins; on an exact tie the
 * LOWER expert index wins (mandatory determinism). Strict > on score keeps
 * the relation antisymmetric; an in-band NaN would make a > b false in both
 * directions and a lower index would then win — NaN never silently wins. */
static int route_above(uint32_t ea, float sa, uint32_t eb, float sb)
{
    if (sa > sb)
        return 1;
    if (sa < sb)
        return 0;
    return ea < eb;
}

qw_err qw_router_topk(const float *scores, int n_experts, int k,
                      qw_route *out, void *scratch)
{
    if (QW_UNLIKELY(scores == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n_experts <= 0 || k < 1 || k > n_experts))
        return QW_ERR_RANGE;
    if (QW_UNLIKELY(out == NULL))
        return QW_ERR_NULL;
    if (scratch == NULL && k > QW_ROUTER_STACK_K)
        return QW_ERR_RANGE;

    /* Non-finite scores are rejected up front: a NaN must never silently
     * win a comparison. (The tie-broken comparison below is a second line
     * of defense for a caller that bypassed the check.) */
    for (int i = 0; i < n_experts; i++) {
        if (QW_UNLIKELY(!isfinite(scores[i])))
            return QW_ERR_FORMAT;
    }

    /* O(k^2) fixup storage: k+1 slots (k selected + 1 guard for insertion
     * at position k while filling). Stack when k <= QW_ROUTER_STACK_K,
     * otherwise the caller's scratch. */
    route_slot *sel;
    if (k <= QW_ROUTER_STACK_K) {
        route_slot sbuf[QW_ROUTER_STACK_K + 1];
        sel = sbuf;
    } else {
        sel = (route_slot *)scratch;
    }

    /* Single O(n) scan. sel[0..ns-1] stays sorted descending by the
     * deterministic total order (score desc, then expert index asc); the
     * last slot sel[ns-1] is the current k-th best. */
    int ns = 0;
    for (int i = 0; i < n_experts; i++) {
        float s = scores[i];
        if (ns < k || route_above((uint32_t)i, s, sel[ns - 1].expert,
                                  sel[ns - 1].score)) {
            /* new entry ranks above sel[ns-1] (or the list is not full):
             * shift everything it outranks up and insert. */
            int p = ns;
            while (p > 0 && route_above((uint32_t)i, s,
                                         sel[p - 1].expert, sel[p - 1].score))
                p--;
            for (int q = ns; q > p; q--)
                sel[q] = sel[q - 1];
            sel[p].expert = (uint32_t)i;
            sel[p].score = s;
            if (ns < k)
                ns++;
        }
    }

    /* sel[0..k-1] is the top-k set in descending score order (deterministic
     * tie-break). Re-sort ascending by expert index for the output. */
    for (int i = 1; i < k; i++) {
        route_slot key = sel[i];
        int j = i - 1;
        while (j >= 0 && sel[j].expert > key.expert) {
            sel[j + 1] = sel[j];
            j--;
        }
        sel[j + 1] = key;
    }

    /* Softmax-normalize the weights over the selected k only. */
    float maxs = sel[0].score;
    for (int i = 1; i < k; i++)
        if (sel[i].score > maxs)
            maxs = sel[i].score;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sel[i].score = expf(sel[i].score - maxs);
        sum += sel[i].score;
    }
    for (int i = 0; i < k; i++) {
        out[i].expert = sel[i].expert;
        out[i].weight = sel[i].score / sum;
    }
    return QW_OK;
}

qw_err qw_router_expert_load(const qw_route *routes, int n_routes,
                             uint32_t counts[512])
{
    if (QW_UNLIKELY(routes == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(counts == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n_routes < 0))
        return QW_ERR_RANGE;

    for (int i = 0; i < n_routes; i++)
        counts[routes[i].expert]++;
    return QW_OK;
}

qw_err qw_router_softmax_inplace(float *scores, int n)
{
    if (QW_UNLIKELY(scores == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n < 1))
        return QW_ERR_RANGE;
    for (int i = 0; i < n; i++) {
        if (QW_UNLIKELY(!isfinite(scores[i])))
            return QW_ERR_FORMAT;
    }

    /* Max-subtract first so the exp() argument is <= 0 (no overflow). */
    float maxs = scores[0];
    for (int i = 1; i < n; i++)
        if (scores[i] > maxs)
            maxs = scores[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - maxs);
        sum += scores[i];
    }
    for (int i = 0; i < n; i++)
        scores[i] /= sum;
    return QW_OK;
}
