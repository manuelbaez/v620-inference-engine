/* qw/sampler.h — logits post-processing and token sampling. Pure C17.
 *
 * The vocab is 248320, so the hot path (qw_sampler_top_k) must not sort the
 * whole vocabulary per token. top_k is a single O(n) scan plus an O(k)
 * insertion into a fixed-size k-th-best set: O(n log k) overall, no full
 * sort. Determinism is mandatory: ties on score resolve to the LOWEST token
 * index, so top-k selection is reproducible bit-for-bit. Non-finite logits
 * (NaN / inf) are rejected up front, never ranked or propagated.
 *
 * All working storage lives in the caller-supplied fixed scratch buffer
 * (qw_sampler_scratch_size); the sampler performs no allocation.
 */
#ifndef QW_SAMPLER_H
#define QW_SAMPLER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sampling configuration. */
typedef struct qw_sampler_cfg {
    float    temperature;      /* > 0; 0 means greedy */
    float    top_p;            /* 0 < top_p <= 1; 1.0 disables nucleus */
    int      top_k;            /* 0 or 1..n_vocab; 0 disables top-k */
    float    repetition_penalty; /* > 0; 1.0 = off */
    uint64_t seed;             /* RNG seed */
} qw_sampler_cfg;

/* One working slot for the top-k k-th-best set. */
typedef struct qw_sampler_slot {
    uint32_t idx;
    float    score;
} qw_sampler_slot;

/* One slot in the top_p index buffer. */
typedef struct qw_sampler_pidx {
    float    prob;
    uint32_t idx;
} qw_sampler_pidx;

/* Sampler state: config + reusable scratch (no per-call malloc). */
typedef struct qw_sampler {
    qw_sampler_cfg    cfg;
    qw_sampler_slot  *topk;     /* capacity = cfg.top_k */
    qw_sampler_pidx  *pbuf;     /* capacity = n_vocab */
    size_t            topk_cap;
    size_t            pbuf_cap;
    uint64_t          rng_state; /* 64-bit PCG state */
} qw_sampler;

/* Byte size of a scratch buffer for top_k = k and n_vocab = n slots. */
size_t qw_sampler_scratch_size(int k, int n_vocab);

/* Initialize the sampler with cfg; scratch must be exactly
 * qw_sampler_scratch_size(cfg.top_k, n_vocab) bytes. scratch == NULL only
 * when cfg.top_k <= 16 (stack fallback, as in the router). */
qw_err qw_sampler_init(qw_sampler *s, const qw_sampler_cfg *cfg,
                       int n_vocab, void *scratch);

/* Mark all but the k largest logits as -inf in mask (mask[i] = 1 keep /
 * 0 drop). logits is read-only; never modified. O(n log k), no full sort. */
qw_err qw_sampler_top_k(qw_sampler *s, const float *logits, int n_vocab,
                        int k, uint8_t *mask);

/* Nucleus: keep the smallest set (lowest index first) whose cumulative
 * probability >= p among indices where mask[i] == 1. Writes the kept
 * indices in ascending order to out (capacity n_vocab). Returns QW_OK and
 * sets *n_kept. p in (0, 1]; p == 1.0 keeps everything mask allows. */
qw_err qw_sampler_top_p(qw_sampler *s, const float *logits,
                        const uint8_t *mask, float p, uint32_t *out,
                        int n_vocab, int *n_kept);

/* Temperature scaling into out (out may alias nothing; distinct from
 * logits). t == 0: greedy — out = one-hot on the argmax (tie: lowest
 * index). t < 0 or non-finite: QW_ERR_RANGE. Max is subtracted before
 * exp() so very small t cannot overflow. */
qw_err qw_sampler_temperature(const float *logits, int n, float t,
                             float *out);

/* Inverse-CDF draw from probs (must sum to ~1). Deterministic given
 * s->rng_state; advances the internal PCG. */
qw_err qw_sampler_sample(const qw_sampler *s, const float *probs, int n,
                         uint32_t *tok_out);

/* Greedy: argmax of logits, tie broken to the lowest index. */
qw_err qw_sampler_greedy(const float *logits, int n, uint32_t *tok_out);

/* Repetition penalty, in place on logits: for each token in prev_tokens,
 * logits[i] /= penalty if logits[i] > 0 else logits[i] *= penalty
 * (penalty > 1 pushes seen tokens down either sign; the sign-split is the
 * standard correct form because dividing a negative logit by > 1 makes it
 * MORE negative, which is wrong). penalty == 1.0: no-op. */
qw_err qw_sampler_repetition_penalty(float *logits, int n,
                                     const uint32_t *prev_tokens, int n_prev,
                                     float penalty);

/* MTP (multi-token prediction) verification lives elsewhere (in the
 * attention/spec-decode layer), not in this module. */

#ifdef __cplusplus
}
#endif

#endif /* QW_SAMPLER_H */
