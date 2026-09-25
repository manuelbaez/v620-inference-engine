/* C API of the qw engine, for the Python server (ctypes) and other hosts.
 * One handle = one engine on the 4 GPUs, holding several sequence slots.
 * Calls must not overlap (one host thread drives the engine). Functions
 * returning int: >= 0 on success, -1 on error (see qw_error). */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw_handle qw_handle;

typedef struct qw_sampling {
    float temperature; /* 0 = greedy */
    float top_p;       /* 1 = off */
    int32_t top_k;     /* 0 = off */
    float min_p;       /* 0 = off */
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty; /* 1 = off */
    uint64_t seed;            /* 0 = random */
} qw_sampling;

/* options: JSON object, e.g. {"model_dir": "...", "ple_dir": "...",
 * "slots": [131072, 65536, 32768, 32768], "prefill_chunk": 8192}. */
qw_handle *qw_open(const char *options_json, char *err, int errlen);
void qw_close(qw_handle *h);
const char *qw_error(qw_handle *h);

int qw_num_slots(qw_handle *h);
int64_t qw_slot_capacity(qw_handle *h, int slot);

/* Picks and reserves a free slot that can hold n + max_new tokens, preferring
 * the one that can reuse the most of `tokens`. Returns the slot, or -1 when
 * none is free (not an error: qw_error is empty). */
int qw_acquire(qw_handle *h, const int32_t *tokens, int64_t n, int64_t max_new);
int qw_release(qw_handle *h, int slot);

/* Makes `slot` hold exactly tokens[0..n). Returns the reused prompt tokens. */
int64_t qw_set_prompt(qw_handle *h, int slot, const int32_t *tokens, int64_t n);
/* Samples the token after the prompt of the last qw_set_prompt. */
int32_t qw_sample_prompt(qw_handle *h, int slot, const qw_sampling *p, float *lp);

/* One batched step: row i appends tokens[i] to slots[i] (rows of a slot
 * consecutive, n <= 16). */
int qw_decode(qw_handle *h, int n, const int32_t *slots, const int32_t *tokens);
/* Samples from row `row` of the last qw_decode. */
int32_t qw_sample_row(qw_handle *h, int row, const qw_sampling *p, float *lp);

/* Top-k (id, logprob) of the raw distribution after the prompt (row < 0) or
 * of a decode row. Returns k. */
int qw_top_logprobs(qw_handle *h, int row, int k, int32_t *ids, float *lps);

#ifdef __cplusplus
}
#endif
