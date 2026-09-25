/* qw/sched.h — continuous-batching scheduler. Pure C17, libc only.
 *
 * Layer-split pipeline parallelism gives a single sequence NO speedup (the
 * per-GPU times add), so throughput must come from concurrency: several
 * sequences in flight at once, mixing prefill and decode work in every step.
 * This module decides, per step, which sequences run and what each one
 * contributes (a prefill chunk or one decode token).
 *
 * POLICY — chunked prefill (the standard "greedy new-prompt-first up to a
 * token budget, then decode" policy, with the prefill work CHUNKED):
 *
 *   Each step has a fixed token budget B (qw_sched_set_budget; default
 *   512). Within a step, in this priority order:
 *     1. Prefill goes FIRST: waiting prompts are admitted in FIFO order and
 *        prefilled first. A prompt that is longer than the remaining budget
 *        is CHUNKED — it takes as many tokens as fit and the rest is carried
 *        to the next step. Prefilling before decoding keeps
 *        time-to-first-token for every new prompt tight (a new prompt never
 *        waits behind a long decode phase) and, since prefill is
 *        compute-bound while decode is memory-bound, interleaving both in
 *        one step keeps the GPU fed. The cost is a little added per-token
 *        decode latency (a decode step shares the budget with prefill),
 *        which is the deliberate trade for much better TTFT and throughput.
 *     2. Whatever budget is left after prefill is filled with decode,
 *        ONE token per running sequence, in table order.
 *
 *   KV safety: a waiting sequence is admitted only if blocks for its FULL
 *   remaining context (prompt_len + max_tokens, the worst case) fit in the
 *   current free blocks; otherwise admission stops (no overcommit). A
 *   running/decode sequence only ever needs 1 more token, so it is never
 *   blocked while it still has a block.
 *
 *   Eviction (vLLM-style, kept simple and deterministic): if a waiting
 *   sequence cannot be admitted, no running sequence has a pending prefill
 *   chunk, and some running sequence IS SWAPPED, the most recently ADMITTED
 *   running (non-swapped) sequence is evicted back to waiting — its KV
 *   blocks are freed and its prefilled work kept — and admission is retried.
 *
 *   A sequence becomes done (and frees its blocks) when it emits end-of-
 *   stream (qw_sched_finish(seq, eos=true)) or reaches max_tokens
 *   (qw_sched_set_max_tokens).
 *
 * The scheduler is a FIXED-CAPACITY table: no per-step allocation, no
 * randomness, no clocks. The same sequence of calls produces byte-identical
 * plans.
 *
 * Relationship to qw/kvcache.h: this scheduler is an accounting-only
 * decision layer. It tracks free blocks as a running counter
 * (kv_blocks_total at init, minus blocks held by running sequences) and
 * receives a fresh qw_kv_stats each step (used to re-sync that counter). It
 * does NOT own a qw_kvcache; the engine calls qw_kvcache_reserve() /
 * qw_kvcache_release() against the plan it emits.
 */
#ifndef QW_SCHED_H
#define QW_SCHED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/kvcache.h"   /* qw_kv_stats */
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default per-step token budget (see the POLICY note above). */
#define QW_SCHED_DEFAULT_BUDGET 512

/* ----------------------------------------------------------- sequence */
/* Per-sequence scheduler state. id/prompt_len/generated/prefilled are the
 * externally observable fields; active/done/state + the private fields
 * (added_order, admitted_order, max_tokens) drive scheduling. */
typedef struct qw_seq {
    uint32_t id;
    uint64_t prompt_len;    /* prompt tokens to prefill            */
    uint64_t generated;     /* tokens emitted by decode            */
    uint64_t prefilled;     /* prompt tokens prefilled so far      */
    bool     active;        /* admitted into the running set       */
    bool     done;          /* finished (eos or max_tokens)        */
    enum { QW_SEQ_WAITING, QW_SEQ_RUNNING, QW_SEQ_SWAPPED } state;

    uint64_t added_order;     /* FIFO position (qw_sched_add order) */
    uint64_t admitted_order;  /* admission position (evict last)    */
    uint64_t max_tokens;      /* decode cap; 0 = unlimited          */
} qw_seq;

/* ----------------------------------------------------------------- plan */
typedef enum qw_sched_kind {
    QW_SCHED_PREFILL_CHUNK = 0, /* n_tokens = prompt tokens prefilled now */
    QW_SCHED_DECODE        = 1, /* n_tokens == 1                          */
} qw_sched_kind;

/* One work item in a step's plan. */
typedef struct qw_sched_plan_entry {
    uint32_t      seq_id;
    qw_sched_kind kind;
    uint64_t      n_tokens;
} qw_sched_plan_entry;

/* The plan for one step: the scheduled work items plus totals. entries is
 * owned by the scheduler (capacity max_seqs); out->entries aliases it. */
typedef struct qw_step_plan {
    qw_sched_plan_entry *entries;
    int                  n_entries;
    uint64_t             n_prefill_tokens; /* sum of PREFILL_CHUNK n_tokens */
    uint64_t             n_decode_tokens;  /* == number of DECODE entries   */
} qw_step_plan;

/* ------------------------------------------------------------ statistics */
typedef struct qw_sched_statistics {
    int      n_running;       /* sequences in state RUNNING or SWAPPED  */
    int      n_waiting;       /* sequences in state WAITING, not done   */
    double   mean_blocks_running; /* mean kv blocks held per running seq */
    double   prefill_util;    /* (sum prefill tokens) / (steps * budget), 0..1 */
    double   pure_decode_frac;/* fraction of non-empty steps that were
                                 pure decode (no prefill work)          */
} qw_sched_stat;

/* ---------------------------------------------------------------- handle */
/* The scheduler is a plain struct: a fixed-capacity sequence table, a
 * waiting FIFO, a per-step plan, and running stats. Pass it by reference. */
struct qw_sched {
    int      max_seqs;
    uint64_t kv_blocks_total;
    uint64_t free_blocks;      /* running counter, re-synced each step   */
    uint64_t budget;           /* per-step token budget B                */
    uint64_t max_tokens;       /* default decode cap for new sequences   */

    qw_seq    *seqs;           /* fixed table, capacity max_seqs         */
    int        n_seq;
    uint64_t   add_counter;    /* next added_order (FIFO position)       */
    uint64_t   adm_counter;    /* next admitted_order                    */

    qw_step_plan plan;         /* per-step plan (entries owned by us)    */
    uint64_t     n_steps;             /* non-empty steps taken            */
    uint64_t     pure_decode_steps;   /* of those, steps with no prefill  */
    uint64_t     total_prefill_tokens;/* sum of prefill tokens, all steps */
};
typedef struct qw_sched qw_sched;

/* -------------------------------------------------------------- lifecycle */
/* Create a scheduler for up to max_seqs sequences, with kv_blocks_total
 * paged KV blocks to account for. Allocates the fixed sequence table, the
 * waiting FIFO, and the per-step plan array (capacity max_seqs).
 * QW_ERR_NULL on NULL out; QW_ERR_RANGE if max_seqs <= 0 or
 * kv_blocks_total == 0; QW_ERR_ALLOC on allocation failure. */
qw_err qw_sched_init(qw_sched *s, int max_seqs, uint64_t kv_blocks_total);

/* Free all scheduler memory. NULL-safe; idempotent. */
void qw_sched_destroy(qw_sched *s);

/* Set the per-step token budget B (used by qw_sched_step). Default is
 * QW_SCHED_DEFAULT_BUDGET. QW_ERR_NULL on NULL s; QW_ERR_RANGE if b == 0. */
qw_err qw_sched_set_budget(qw_sched *s, uint64_t b);

/* Set the per-sequence decode cap applied to subsequently added sequences.
 * 0 = unlimited (default). QW_ERR_NULL on NULL s. */
qw_err qw_sched_set_max_tokens(qw_sched *s, uint64_t max_tokens);

/* Add a new sequence with the given id and prompt length. It enters the
 * waiting FIFO. id must be unique (QW_ERR_RANGE otherwise) and there must
 * be a free table slot (QW_ERR_RANGE if full). The sequence's max_tokens
 * is set to the current default. QW_ERR_NULL on NULL s. */
qw_err qw_sched_add(qw_sched *s, uint32_t seq_id, uint64_t prompt_len);

/* Run one scheduling step and emit the plan into out.
 *
 *   - out must be non-NULL; out->entries is set to the scheduler's own
 *     plan array (it is not (re)allocated by the scheduler).
 *   - kv_avail (may be NULL) is the current KV-cache snapshot; its
 *     free_blocks re-synchronizes the scheduler's free-block counter so it
 *     stays consistent with the real cache.
 *
 * The step fills out per the POLICY in the header, then applies its
 * consequences: each PREFILL_CHUNK advances the sequence's prefilled count
 * (and frees the newly-covered blocks), each DECODE advances generated (and
 * frees the one block the new token covers), and sequences that reach
 * max_tokens are marked done and freed. out->n_entries and the totals are
 * always set (zero when nothing runs). QW_ERR_NULL on NULL s/out. */
qw_err qw_sched_step(qw_sched *s, const qw_kv_stats *kv_avail,
                     qw_step_plan *out);

/* Mark a sequence done. eos=true means it emitted end-of-stream; either way
 * it is removed from the schedule and its KV blocks are freed.
 * QW_ERR_NULL on NULL s; QW_ERR_RANGE if seq_id is unknown. */
qw_err qw_sched_finish(qw_sched *s, uint32_t seq_id, bool eos);

/* Snapshot statistics into out (see qw_sched_stat). QW_ERR_NULL on NULL
 * s/out. */
qw_err qw_sched_stats(const qw_sched *s, qw_sched_stat *out);

/* One-line human-readable plan summary (counts + totals + budget usage). */
void qw_sched_plan_report(const qw_step_plan *plan, uint64_t budget);

#ifdef __cplusplus
}
#endif

#endif /* QW_SCHED_H */
