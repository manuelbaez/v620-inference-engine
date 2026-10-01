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
/* Saves the slots' conversations to the disk prefix cache (if configured) and
 * waits for the writes; call before exiting, with no request in flight. */
int qw_persist(qw_handle *h);
const char *qw_error(qw_handle *h);
/* Why the engine can no longer be used (a rank's job failed or a collective timed out; it stays
 * so until the process restarts), copied into buf; 0 while it works, else the length of the
 * message (not counting the final 0). Safe to call from a thread other than the one driving
 * the engine. */
int qw_engine_failure(qw_handle *h, char *buf, int buflen);

/* ---- prefix cache */
/* Token that starts a chat message (<|im_start|>): prefill snapshots the state
 * before such tokens, so prompts sharing a prefix up to a message reuse it. */
int qw_set_boundary_token(qw_handle *h, int32_t id);
typedef struct qw_cache_stats {
    uint64_t hits, tokens_restored, snapshots_saved;
    uint64_t ram_bytes, disk_bytes, blocks, snapshots;
    /* prompt tokens, tokens reused exactly, and tokens after the reuse point in
     * chunks seen before at any position (what non-prefix reuse could save) */
    uint64_t prompt_tokens, reused_tokens, blend_candidate_tokens;
} qw_cache_stats;
int qw_get_cache_stats(qw_handle *h, qw_cache_stats *out);

int qw_num_slots(qw_handle *h);
int64_t qw_slot_capacity(qw_handle *h, int slot);

/* Picks and reserves a free slot that can hold n + max_new tokens, preferring
 * the one that can reuse the most of `tokens`. Returns the slot, or -1 when
 * none is free (not an error: qw_error is empty). */
int qw_acquire(qw_handle *h, const int32_t *tokens, int64_t n, int64_t max_new);
int qw_release(qw_handle *h, int slot);

/* Makes `slot` hold exactly tokens[0..n). Returns the reused prompt tokens. */
int64_t qw_set_prompt(qw_handle *h, int slot, const int32_t *tokens, int64_t n);
/* ---- vision: an image or a video in a prompt (the prompt holds its pad token,
 * <|image_pad|> or <|video_pad|>, at each of its token positions). */
typedef struct qw_media {
    uint64_t hash;          /* content hash: its tokens' identity in the prefix cache */
    int32_t video;          /* 0 image, 1 video */
    int32_t t, h, w;        /* grid in patches: t temporal slices of h x w */
    const float *patches;   /* fp32 [t*h*w][1536], merge-window order */
    const int64_t *starts;  /* first token of each slice (h*w/4 tokens each), t entries */
} qw_media;
/* qw_acquire for a prompt with media (so the slot choice sees the media's identity). */
int qw_acquire_media(qw_handle *h, const int32_t *tokens, int64_t n, int64_t max_new, const qw_media *media,
                     int n_media);
/* 1 if the engine has the vision tower (images and videos accepted). */
int qw_has_vision(qw_handle *h);
/* qw_set_prompt with media; the vision tower runs on the cards for the media
 * tokens that are not already cached. */
int64_t qw_set_prompt_media(qw_handle *h, int slot, const int32_t *tokens, int64_t n, const qw_media *media,
                            int n_media);
/* qw_set_prompt in pieces, so other slots can decode between them: begin
 * restores what the caches hold and returns the reused tokens (media must stay
 * valid until the prompt is in); prefill_some prefills up to max_tokens more
 * and returns 1 once the whole prompt is in, 0 while some is left. */
int64_t qw_begin_prompt(qw_handle *h, int slot, const int32_t *tokens, int64_t n, const qw_media *media, int n_media);
int qw_prefill_some(qw_handle *h, int slot, int64_t max_tokens);
/* Before qw_begin_prompt: 1 while the prompt's prefix-cache entries that are
 * only on disk are being read into RAM in the background (poll), 0 when there
 * is nothing (more) to load; -1 on error. */
int qw_prefetch(qw_handle *h, int slot, const int32_t *tokens, int64_t n, const qw_media *media, int n_media);
/* qw_prefill_some for n slots in one pass when their pieces fit one prefill
 * chunk (else one after the other); done[i] = 1 once slots[i]'s prompt is in. */
int qw_prefill_batch(qw_handle *h, int n, const int32_t *slots, const int64_t *max_tokens, int32_t *done);
/* Samples the token after the slot's prompt. */
int32_t qw_sample_prompt(qw_handle *h, int slot, const qw_sampling *p, float *lp);

/* One batched step: row i appends tokens[i] to slots[i] (rows of a slot
 * consecutive, n <= 16). */
int qw_decode(qw_handle *h, int n, const int32_t *slots, const int32_t *tokens);
/* Samples from row `row` of the last qw_decode. */
int32_t qw_sample_row(qw_handle *h, int row, const qw_sampling *p, float *lp);

/* ---- generation with MTP speculative decoding */
typedef struct qw_step_req {
    int32_t slot;
    int32_t pending; /* sampled token, not decoded yet */
    int32_t budget;  /* tokens the request may still emit (>= 1) */
    int32_t flags;   /* QW_STEP_LOGITS: keep the full logits of its rows (for qw_top_logprobs) */
    qw_sampling sampling;
} qw_step_req;

#define QW_STEP_LOGITS 1

/* 1 if the MTP head is loaded (qw_generate drafts tokens), else 0. */
int qw_has_mtp(qw_handle *h);
/* Tokens that end the request in `slot` (EOS, stop_token_ids). */
int qw_set_stop_tokens(qw_handle *h, int slot, const int32_t *ids, int n);
/* One step of n requests (distinct slots): decodes each pending token plus up
 * to k drafts and emits 1..k+1 tokens per request (at most its budget).
 * Outputs per request i: counts[i] tokens in tokens[i*(k+1) ..] with their
 * logprobs, first_rows[i] (the row of tokens[i*(k+1)] for qw_top_logprobs; the
 * j-th token is row first_rows[i] + j) and stopped[i] (1: the last token is a
 * stop token). k = 0: plain decoding. */
int qw_generate(qw_handle *h, int n, const qw_step_req *reqs, int k, int32_t *tokens, float *logprobs, int32_t *counts,
                int32_t *first_rows, int32_t *stopped);

/* Top-k (id, logprob) of the raw distribution after the prompt (row < 0) or
 * of a decode row. Returns k. */
int qw_top_logprobs(qw_handle *h, int row, int k, int32_t *ids, float *lps);
/* The same after a slot's prompt. */
int qw_top_logprobs_prompt(qw_handle *h, int slot, int k, int32_t *ids, float *lps);

#ifdef __cplusplus
}
#endif
