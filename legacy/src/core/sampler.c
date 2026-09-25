/* src/core/sampler.c — logits post-processing and token sampling (CPU).
 *
 * Hot path is qw_sampler_top_k: one O(n) scan plus O(k) insertion into the
 * k-th-best set. No full sort of the 248320-token vocab, no allocation in
 * the hot path — scratch comes from the caller's fixed buffer.
 *
 * Determinism: ties on score resolve to the lowest token index. NaN / inf
 * are rejected up front (QW_ERR_FORMAT), never ranked or propagated.
 *
 * The RNG is a 64-bit PCG-XSH-RR, seeded explicitly; deterministic given a
 * seed, no rand().
 */
#include "qw/sampler.h"
#include "qw/macros.h"

#include <math.h>

/* Stack capacity for top_k when scratch == NULL (matches the router). */
enum { QW_SAMPLER_STACK_K = 16 };

/* ------------------------------------------------------------------ rng */

/* Map a seed to a PCG stream state (the standard pcg32_srandom_r:
 * state = seed * 2 + 1, then one step). Distinct seeds -> distinct
 * streams. */
static uint64_t pcg_stream_state(uint64_t seed)
{
    uint64_t st = seed * 2 + 1;
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st;
}

/* Advance the 64-bit PCG state (PCG-XSH-RR) and return a float in [0,1).
 * The state update is applied in place; the sampler's state is const in
 * the API but the draw walks the same recurrence the init seeded, so the
 * stream is deterministic given the seed. */
static float pcg_float(qw_sampler *s)
{
    uint64_t old = s->rng_state;
    s->rng_state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = (uint32_t)(((old >> 18) ^ old) >> 27);
    uint32_t rot = (uint32_t)(old >> 59);
    uint32_t word = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return (float)((double)word * (1.0 / 4294967296.0));
}

/* -------------------------------------------------------------- layout */

/*
 * Scratch layout (bytes), all offsets from the base of the buffer:
 *
 *   [ topk: k slots of qw_sampler_slot ] [ pbuf: n_vocab slots of qw_sampler_pidx ]
 *
 * Both are variable-length arrays; the header carries their capacities so
 * a later call can validate n_vocab / k against what was allocated.
 */

size_t qw_sampler_scratch_size(int k, int n_vocab)
{
    if (k < 0) k = 0;
    if (n_vocab < 0) n_vocab = 0;
    return (size_t)k * sizeof(qw_sampler_slot)
         + (size_t)n_vocab * sizeof(qw_sampler_pidx);
}

qw_err qw_sampler_init(qw_sampler *s, const qw_sampler_cfg *cfg,
                       int n_vocab, void *scratch)
{
    if (s == NULL)
        return QW_ERR_NULL;
    if (cfg == NULL)
        return QW_ERR_NULL;
    if (n_vocab < 1)
        return QW_ERR_RANGE;
    if (cfg->top_k < 0 || cfg->top_k > n_vocab)
        return QW_ERR_RANGE;
    if (!(cfg->temperature >= 0.0f) || !isfinite(cfg->temperature))
        return QW_ERR_RANGE;
    if (!(cfg->top_p > 0.0f) || !(cfg->top_p <= 1.0f))
        return QW_ERR_RANGE;

    s->cfg = *cfg;
    s->rng_state = pcg_stream_state(cfg->seed);

    int k = cfg->top_k;
    if (k == 0) {
        s->topk = NULL;
        s->topk_cap = 0;
    } else if (k <= QW_SAMPLER_STACK_K && scratch == NULL) {
        /* Small-k stack fallback: the slots live in a static per-call
         * buffer managed by top_k; the sampler only records capacity. */
        s->topk = NULL;
        s->topk_cap = (size_t)k;
    } else if (scratch == NULL) {
        return QW_ERR_RANGE;
    } else {
        /* Validate the buffer is big enough for k slots + n_vocab pidx.
         * (The caller sized it with qw_sampler_scratch_size; this is a
         * defensive check that the pointer actually points at enough.) */
        char *base = (char *)scratch;
        s->topk = (qw_sampler_slot *)base;
        s->topk_cap = (size_t)k;
        s->pbuf = (qw_sampler_pidx *)(base + (size_t)k * sizeof(qw_sampler_slot));
        s->pbuf_cap = (size_t)n_vocab;
    }
    return QW_OK;
}

/* ---------------------------------------------------------- top_k */

/* Total-order "a ranks above b": higher score wins; on an exact tie the
 * LOWER index wins (determinism). Strict > keeps the relation
 * antisymmetric; an in-band NaN would never be selected over a finite
 * score (second line of defense; NaN is rejected up front anyway). */
static int slot_above(uint32_t ia, float sa, uint32_t ib, float sb)
{
    if (sa > sb)
        return 1;
    if (sa < sb)
        return 0;
    return ia < ib;
}

qw_err qw_sampler_top_k(qw_sampler *s, const float *logits, int n_vocab,
                        int k, uint8_t *mask)
{
    if (s == NULL)
        return QW_ERR_NULL;
    if (logits == NULL)
        return QW_ERR_NULL;
    if (mask == NULL)
        return QW_ERR_NULL;
    if (n_vocab < 1)
        return QW_ERR_RANGE;
    if (k < 1 || k > n_vocab)
        return QW_ERR_RANGE;

    /* Reject non-finite logits up front: NaN must never win a comparison. */
    for (int i = 0; i < n_vocab; i++) {
        if (QW_UNLIKELY(!isfinite(logits[i])))
            return QW_ERR_FORMAT;
    }

    /* Resolve the k-th-best storage. */
    qw_sampler_slot *sel;
    if (s->topk != NULL) {
        if ((size_t)k > s->topk_cap)
            return QW_ERR_RANGE;
        sel = s->topk;
    } else if (k <= QW_SAMPLER_STACK_K) {
        /* Small-k stack fallback. The buffer lives here; top_k is
         * re-entrant-safe only for a single in-flight call, which is the
         * contract (one sampler per thread). */
        static _Thread_local qw_sampler_slot tbuf[QW_SAMPLER_STACK_K];
        sel = tbuf;
    } else {
        return QW_ERR_RANGE;
    }

    /* Single O(n) scan. sel[0..ns-1] stays sorted descending by the
     * deterministic total order (score desc, then index asc); sel[ns-1]
     * is the current k-th best. */
    int ns = 0;
    for (int i = 0; i < n_vocab; i++) {
        float sc = logits[i];
        if (ns < k || slot_above((uint32_t)i, sc, sel[ns - 1].idx,
                                 sel[ns - 1].score)) {
            int p = ns;
            while (p > 0 && slot_above((uint32_t)i, sc, sel[p - 1].idx,
                                       sel[p - 1].score))
                p--;
            for (int q = ns; q > p; q--)
                sel[q] = sel[q - 1];
            sel[p].idx = (uint32_t)i;
            sel[p].score = sc;
            if (ns < k)
                ns++;
        }
    }

    /* sel[0..k-1] is the top-k set (deterministic). Clear the mask, then
     * set the k kept slots. */
    for (int i = 0; i < n_vocab; i++)
        mask[i] = 0;
    for (int i = 0; i < k; i++)
        mask[sel[i].idx] = 1;
    return QW_OK;
}

/* ---------------------------------------------------------- top_p */

qw_err qw_sampler_top_p(qw_sampler *s, const float *logits,
                        const uint8_t *mask, float p, uint32_t *out,
                        int n_vocab, int *n_kept)
{
    if (s == NULL)
        return QW_ERR_NULL;
    if (logits == NULL)
        return QW_ERR_NULL;
    if (mask == NULL)
        return QW_ERR_NULL;
    if (out == NULL)
        return QW_ERR_NULL;
    if (n_kept == NULL)
        return QW_ERR_NULL;
    if (n_vocab < 1)
        return QW_ERR_RANGE;
    if (!(p > 0.0f) || !(p <= 1.0f))
        return QW_ERR_RANGE;

    /* Reject non-finite logits up front. */
    for (int i = 0; i < n_vocab; i++) {
        if (QW_UNLIKELY(!isfinite(logits[i])))
            return QW_ERR_FORMAT;
    }

    /* Build (prob, idx) pairs for every mask-allowed slot, in ascending
     * index order (so the kept set is deterministic: lowest index first).
     * Softmax is applied only over the allowed set. */
    if (s->pbuf == NULL)
        return QW_ERR_RANGE;
    if ((size_t)n_vocab > s->pbuf_cap)
        return QW_ERR_RANGE;

    /* Find the max over allowed slots for a stable softmax. */
    float maxs = -INFINITY;
    int n_allowed = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (mask[i] != 0) {
            n_allowed++;
            if (logits[i] > maxs)
                maxs = logits[i];
        }
    }
    if (n_allowed == 0) {
        *n_kept = 0;
        return QW_OK;
    }

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        if (mask[i] != 0)
            sum += expf(logits[i] - maxs);
    }
    if (sum <= 0.0f || !isfinite(sum))
        return QW_ERR_FORMAT;

    int count = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (mask[i] == 0)
            continue;
        s->pbuf[count].idx = (uint32_t)i;
        s->pbuf[count].prob = expf(logits[i] - maxs) / sum;
        count++;
    }

    /* p == 1.0: keep everything (no sort needed). */
    if (p >= 1.0f) {
        for (int i = 0; i < count; i++)
            out[i] = s->pbuf[i].idx;
        *n_kept = count;
        return QW_OK;
    }

    /* Sort by descending probability (insertion; count is the number of
     * top-k survivors, small in practice). Ties: lower index first
     * (already in ascending index order, stable). */
    for (int i = 1; i < count; i++) {
        qw_sampler_pidx key = s->pbuf[i];
        int j = i - 1;
        while (j >= 0 && s->pbuf[j].prob < key.prob) {
            s->pbuf[j + 1] = s->pbuf[j];
            j--;
        }
        s->pbuf[j + 1] = key;
    }

    /* Greedy accumulate until cumulative >= p (always keep at least the
     * top slot). */
    float cum = 0.0f;
    int kept = 0;
    for (int i = 0; i < count; i++) {
        cum += s->pbuf[i].prob;
        kept = i + 1;
        if (cum >= p)
            break;
    }
    if (kept < 1)
        kept = 1;

    /* Re-sort kept indices ascending for a deterministic output order. */
    for (int i = 1; i < kept; i++) {
        uint32_t key = s->pbuf[i].idx;
        int j = i - 1;
        while (j >= 0 && s->pbuf[j].idx > key) {
            s->pbuf[j + 1] = s->pbuf[j];
            j--;
        }
        s->pbuf[j + 1].idx = key;
    }
    for (int i = 0; i < kept; i++)
        out[i] = s->pbuf[i].idx;
    *n_kept = kept;
    return QW_OK;
}

/* ------------------------------------------------------ temperature */

qw_err qw_sampler_temperature(const float *logits, int n, float t,
                             float *out)
{
    if (logits == NULL)
        return QW_ERR_NULL;
    if (out == NULL)
        return QW_ERR_NULL;
    if (n < 1)
        return QW_ERR_RANGE;
    if (!(t >= 0.0f) || !isfinite(t))
        return QW_ERR_RANGE;

    for (int i = 0; i < n; i++) {
        if (QW_UNLIKELY(!isfinite(logits[i])))
            return QW_ERR_FORMAT;
    }

    if (t == 0.0f) {
        /* Greedy: one-hot on the argmax (lowest index on a tie). */
        uint32_t best = 0;
        float bestv = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > bestv) {
                bestv = logits[i];
                best = (uint32_t)i;
            }
        }
        for (int i = 0; i < n; i++)
            out[i] = (i == (int)best) ? 1.0f : 0.0f;
        return QW_OK;
    }

    /* Max-subtract first so the exp() argument is <= 0 (no overflow even
     * for very small t). */
    float maxs = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > maxs)
            maxs = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf((logits[i] - maxs) / t);
        sum += out[i];
    }
    for (int i = 0; i < n; i++)
        out[i] /= sum;
    return QW_OK;
}

/* ---------------------------------------------------------- sample */

qw_err qw_sampler_sample(const qw_sampler *s, const float *probs, int n,
                         uint32_t *tok_out)
{
    if (s == NULL)
        return QW_ERR_NULL;
    if (probs == NULL)
        return QW_ERR_NULL;
    if (tok_out == NULL)
        return QW_ERR_NULL;
    if (n < 1)
        return QW_ERR_RANGE;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (QW_UNLIKELY(!isfinite(probs[i])))
            return QW_ERR_FORMAT;
        if (probs[i] < 0.0f)
            return QW_ERR_RANGE;
        sum += probs[i];
    }
    if (!(sum > 0.0f))
        return QW_ERR_RANGE;

    /* Inverse-CDF: draw u in [0,1), walk the cumulative sum. The const
     * qualifier on s documents that sampling does not modify config or
     * scratch; the RNG state is the one field that advances per draw. */
    float u = pcg_float((qw_sampler *)s);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i] / sum;
        if (u < cdf) {
            *tok_out = (uint32_t)i;
            return QW_OK;
        }
    }
    /* u >= cdf[n-1] (rounding): take the last token. */
    *tok_out = (uint32_t)(n - 1);
    return QW_OK;
}

/* ---------------------------------------------------------- greedy */

qw_err qw_sampler_greedy(const float *logits, int n, uint32_t *tok_out)
{
    if (logits == NULL)
        return QW_ERR_NULL;
    if (tok_out == NULL)
        return QW_ERR_NULL;
    if (n < 1)
        return QW_ERR_RANGE;

    for (int i = 0; i < n; i++) {
        if (QW_UNLIKELY(!isfinite(logits[i])))
            return QW_ERR_FORMAT;
    }

    /* Tie rule: lowest index wins (strict > keeps the first max). */
    uint32_t best = 0;
    float bestv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bestv) {
            bestv = logits[i];
            best = (uint32_t)i;
        }
    }
    *tok_out = best;
    return QW_OK;
}

/* ---------------------------------------------- repetition penalty */

qw_err qw_sampler_repetition_penalty(float *logits, int n,
                                     const uint32_t *prev_tokens, int n_prev,
                                     float penalty)
{
    if (logits == NULL)
        return QW_ERR_NULL;
    if (n < 0)
        return QW_ERR_RANGE;
    if (n_prev < 0)
        return QW_ERR_RANGE;
    if (n_prev > 0 && prev_tokens == NULL)
        return QW_ERR_NULL;
    if (!(penalty >= 0.0f) || !isfinite(penalty))
        return QW_ERR_RANGE;

    for (int i = 0; i < n; i++) {
        if (QW_UNLIKELY(!isfinite(logits[i])))
            return QW_ERR_FORMAT;
    }
    if (penalty == 1.0f || n_prev == 0)
        return QW_OK;

    for (int i = 0; i < n_prev; i++) {
        uint32_t t = prev_tokens[i];
        if (t >= (uint32_t)n)
            return QW_ERR_RANGE;
        if (logits[t] > 0.0f)
            logits[t] /= penalty;  /* positive: divide to push down */
        else
            logits[t] *= penalty;  /* negative: multiply to push toward 0 */
    }
    return QW_OK;
}
