/* src/core/sched.c — continuous-batching scheduler (see qw/sched.h for the
 * chunked-prefill policy, the kv-safety rules, and the eviction rule).
 *
 * Accounting model: the scheduler owns a free-block counter
 * (kv_blocks_total at init). A sequence's KV need is modeled at block
 * granularity; holding n tokens occupies blocks_for(n) = ceil(n / T). When a
 * sequence's length grows from a to b tokens it "frees" blocks_for(a) and
 * "takes" blocks_for(b) blocks — the engine performs the matching
 * qw_kvcache_reserve()/qw_kvcache_release() against the real cache, and each
 * step re-syncs the counter from the real cache's free_blocks.
 *
 * No per-step allocation, no randomness, no clocks: the plan is a pure
 * function of the call history. */
#include "qw/sched.h"
#include "qw/macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tokens per paged KV block — mirrors qw/kvcache.h default (256). */
#define SCHED_TPB 256ULL

/* ------------------------------------------------------------------ cfg */
qw_err qw_sched_init(qw_sched *s, int max_seqs, uint64_t kv_blocks_total)
{
    if (s == NULL)
        return QW_ERR_NULL;
    memset(s, 0, sizeof(*s));

    if (max_seqs <= 0 || kv_blocks_total == 0)
        return QW_ERR_RANGE;

    s->max_seqs        = max_seqs;
    s->kv_blocks_total = kv_blocks_total;
    s->free_blocks     = kv_blocks_total;
    s->budget          = QW_SCHED_DEFAULT_BUDGET;
    s->n_seq           = 0;
    s->add_counter     = 0;
    s->adm_counter     = 0;

    s->seqs = calloc((size_t)max_seqs, sizeof(qw_seq));
    s->plan.entries = malloc((size_t)max_seqs * sizeof(qw_sched_plan_entry));
    if (s->seqs == NULL || s->plan.entries == NULL) {
        free(s->seqs);
        free(s->plan.entries);
        s->seqs = NULL;
        s->plan.entries = NULL;
        return QW_ERR_ALLOC;
    }
    s->plan.n_entries = 0;
    s->plan.n_prefill_tokens = 0;
    s->plan.n_decode_tokens  = 0;
    return QW_OK;
}

void qw_sched_destroy(qw_sched *s)
{
    if (s == NULL)
        return;
    free(s->seqs);
    free(s->plan.entries);
    memset(s, 0, sizeof(*s));
}

qw_err qw_sched_set_budget(qw_sched *s, uint64_t b)
{
    if (s == NULL)
        return QW_ERR_NULL;
    if (b == 0)
        return QW_ERR_RANGE;
    s->budget = b;
    return QW_OK;
}

qw_err qw_sched_set_max_tokens(qw_sched *s, uint64_t max_tokens)
{
    if (s == NULL)
        return QW_ERR_NULL;
    s->max_tokens = max_tokens;
    return QW_OK;
}

/* --------------------------------------------------------------- helpers */
static uint64_t blocks_for(uint64_t n_tokens)
{
    return (n_tokens + SCHED_TPB - 1) / SCHED_TPB;
}

static bool seq_done_limit(const qw_seq *q)
{
    return q->max_tokens != 0 && q->generated >= q->max_tokens;
}

/* Find the sequence with id, or NULL. */
static qw_seq *sched_find(qw_sched *s, uint32_t id)
{
    for (int i = 0; i < s->n_seq; i++)
        if (s->seqs[i].id == id)
            return &s->seqs[i];
    return NULL;
}

/* Release a sequence's held blocks and retire it. */
static void sched_free_seq(qw_sched *s, qw_seq *q)
{
    if (q->state == QW_SEQ_RUNNING || q->state == QW_SEQ_SWAPPED)
        s->free_blocks += blocks_for(q->prefilled + q->generated);
    q->active = false;
    q->done   = true;
    q->state  = QW_SEQ_WAITING;
}

/* -------------------------------------------------------------- admission */
/* Admit one waiting sequence if blocks for its FULL remaining context fit;
 * returns true if admitted. */
static bool try_admit(qw_sched *s, qw_seq *q)
{
    uint64_t need = blocks_for(q->prompt_len + q->max_tokens);
    if (need > s->free_blocks)
        return false;
    s->free_blocks -= need;
    q->active    = true;
    q->state     = QW_SEQ_RUNNING;
    q->prefilled = 0;
    q->generated = 0;
    q->admitted_order = s->adm_counter++;
    return true;
}

/* The running sequence most recently admitted that is not SWAPPED. */
static qw_seq *evict_candidate(qw_sched *s)
{
    qw_seq *best = NULL;
    for (int i = 0; i < s->n_seq; i++) {
        qw_seq *q = &s->seqs[i];
        if (q->state != QW_SEQ_RUNNING)
            continue;
        if (best == NULL || q->admitted_order > best->admitted_order)
            best = q;
    }
    return best;
}

/* True if at least one sequence has pending prefill work (running with
 * prefilled < prompt_len, or any not-done waiting). */
static bool any_prefill_pending(const qw_sched *s)
{
    for (int i = 0; i < s->n_seq; i++) {
        const qw_seq *q = &s->seqs[i];
        if (q->done)
            continue;
        if (q->state == QW_SEQ_RUNNING && q->prefilled < q->prompt_len)
            return true;
        if (q->state == QW_SEQ_WAITING)
            return true;
    }
    return false;
}

/* True if some running sequence is SWAPPED. */
static bool any_swapped(const qw_sched *s)
{
    for (int i = 0; i < s->n_seq; i++)
        if (s->seqs[i].state == QW_SEQ_SWAPPED)
            return true;
    return false;
}

/* ------------------------------------------------------------- stepping */
/* One scheduling pass with the CURRENT free-block counter. Emits the plan
 * into out and applies its consequences (advances counters, frees blocks,
 * marks done). Returns QW_OK. */
static void sched_pass(qw_sched *s, qw_step_plan *out)
{
    out->entries = s->plan.entries;
    out->n_entries = 0;
    out->n_prefill_tokens = 0;
    out->n_decode_tokens = 0;

    uint64_t budget = s->budget;

    /* --- 1. Prefill first: chunked, FIFO across waiting prompts. -----
     * Waiting prompts are admitted here (one per pass) only if blocks for
     * their FULL remaining context fit, then chunked to the budget; a
     * prompt longer than the budget carries its remainder to the next
     * step. Already-admitted prompts continue with any remaining budget.
     * Each seq gets at most one prefill entry per pass. */
    int admit_slot = -1; /* table slot of the next waiting prompt (FIFO) */
    for (int i = 0; i < s->n_seq; i++)
        if (!s->seqs[i].done && s->seqs[i].state == QW_SEQ_WAITING) {
            admit_slot = i;
            break;
        }
    bool admitted_this = false;

    for (int i = 0; i < s->n_seq && budget > 0; i++) {
        qw_seq *q = &s->seqs[i];
        if (q->done)
            continue;
        /* Admit the FIFO-head waiting prompt first (resets its prefilled to
        0), so the chunk computed below reflects its true remaining context. */
        if (q->state == QW_SEQ_WAITING) {
            if (i != admit_slot)          /* not the FIFO head */
                continue;
            if (admitted_this)            /* one admit per pass */
                continue;
            if (!try_admit(s, q))         /* no blocks for full context */
                break;                    /* stop admitting (no overcommit) */
            admitted_this = true;
        }
        if (q->state != QW_SEQ_RUNNING || q->prefilled >= q->prompt_len)
            continue;
        uint64_t rem  = q->prompt_len - q->prefilled;
        uint64_t take = rem < budget ? rem : budget;
        if (take == 0)
            continue;
        out->entries[out->n_entries].seq_id   = q->id;
        out->entries[out->n_entries].kind     = QW_SCHED_PREFILL_CHUNK;
        out->entries[out->n_entries].n_tokens = take;
        out->n_entries++;
        out->n_prefill_tokens += take;
        budget -= take;

        /* apply: this chunk of prompt tokens is now prefilled. */
        q->prefilled += take;
    }

    /* --- 2. Decode: 1 token per running seq that finished its prompt,
     * in table order. A seq that just got its prompt completed this pass
     * (prefilled crossed prompt_len) decodes starting NEXT step, so no seq
     * ever appears twice in one plan. */
    for (int i = 0; i < s->n_seq && budget > 0; i++) {
        qw_seq *q = &s->seqs[i];
        if (q->done || q->state != QW_SEQ_RUNNING)
            continue;
        if (q->prefilled < q->prompt_len)  /* prefill not finished */
            continue;
        if (seq_done_limit(q))
            continue;
        out->entries[out->n_entries].seq_id   = q->id;
        out->entries[out->n_entries].kind     = QW_SCHED_DECODE;
        out->entries[out->n_entries].n_tokens = 1;
        out->n_entries++;
        out->n_decode_tokens++;
        budget--;
        q->generated++;
    }

    /* --- 4. Retire sequences that reached max_tokens. ---------------- */
    for (int i = 0; i < s->n_seq; i++) {
        qw_seq *q = &s->seqs[i];
        if (!q->done && q->state == QW_SEQ_RUNNING && seq_done_limit(q))
            sched_free_seq(s, q);
    }
}

/* --------------------------------------------------------------- public */
qw_err qw_sched_add(qw_sched *s, uint32_t seq_id, uint64_t prompt_len)
{
    if (s == NULL)
        return QW_ERR_NULL;
    if (s->n_seq >= s->max_seqs)
        return QW_ERR_RANGE;
    if (sched_find(s, seq_id) != NULL)
        return QW_ERR_RANGE;

    int i = s->n_seq++;
    qw_seq *q = &s->seqs[i];
    memset(q, 0, sizeof(*q));
    q->id          = seq_id;
    q->prompt_len  = prompt_len;
    q->max_tokens  = s->max_tokens;
    q->state       = QW_SEQ_WAITING;
    q->added_order = s->add_counter++;
    return QW_OK;
}

qw_err qw_sched_step(qw_sched *s, const qw_kv_stats *kv_avail,
                     qw_step_plan *out)
{
    if (s == NULL || out == NULL)
        return QW_ERR_NULL;

    /* re-sync free blocks from the real cache if given. */
    if (kv_avail != NULL)
        s->free_blocks = kv_avail->free_blocks;

    /* --- 1+2. Emit and apply the plan (admits + prefill + decode). ---- */
    sched_pass(s, out);

    /* --- 3. Evict (vLLM-style). When a waiting seq cannot be admitted (no
    blocks for its full context), no running seq has a pending prefill chunk,
    and some running seq is SWAPPED, the most recently admitted running
    (non-swapped) seq is evicted back to waiting — its blocks are freed and
    its prefilled work kept — so the next step can re-admit from the FIFO
    head. Deterministic; at most one eviction per step. (Admission itself
    happens inside sched_pass.) */
    if (any_prefill_pending(s) && any_swapped(s)) {
        bool blocked = false;
        for (int i = 0; i < s->n_seq; i++) {
            qw_seq *q = &s->seqs[i];
            if (q->done || q->state != QW_SEQ_WAITING)
                continue;
            if (blocks_for(q->prompt_len + q->max_tokens) > s->free_blocks)
                blocked = true; /* FIFO head cannot fit -> stop admitting */
            break;
        }
        if (blocked) {
            qw_seq *v = evict_candidate(s);
            if (v != NULL) {
                s->free_blocks += blocks_for(v->prefilled + v->generated);
                v->active = false;
                v->state  = QW_SEQ_WAITING;
            }
        }
    }

    /* --- record step stats (accumulating). ---------------------------- */
    if (out->n_entries > 0) {
        s->n_steps++;
        s->total_prefill_tokens += out->n_prefill_tokens;
        if (out->n_prefill_tokens == 0)
            s->pure_decode_steps++;
    }
    return QW_OK;
}

qw_err qw_sched_finish(qw_sched *s, uint32_t seq_id, bool eos)
{
    (void)eos;
    if (s == NULL)
        return QW_ERR_NULL;
    qw_seq *q = sched_find(s, seq_id);
    if (q == NULL)
        return QW_ERR_RANGE;
    sched_free_seq(s, q);
    return QW_OK;
}

qw_err qw_sched_stats(const qw_sched *s, qw_sched_stat *out)
{
    if (s == NULL || out == NULL)
        return QW_ERR_NULL;

    int running = 0, waiting = 0;
    uint64_t total_blocks = 0;
    for (int i = 0; i < s->n_seq; i++) {
        const qw_seq *q = &s->seqs[i];
        if (q->done)
            continue;
        if (q->state == QW_SEQ_RUNNING || q->state == QW_SEQ_SWAPPED) {
            running++;
            total_blocks += blocks_for(q->prefilled + q->generated);
        } else if (q->state == QW_SEQ_WAITING) {
            waiting++;
        }
    }

    double util = 0.0, frac = 0.0;
    if (s->n_steps > 0) {
        util = (double)s->total_prefill_tokens / (double)(s->n_steps * s->budget);
        frac = (double)s->pure_decode_steps / (double)s->n_steps;
    }

    out->n_running          = running;
    out->n_waiting          = waiting;
    out->mean_blocks_running = running ? (double)total_blocks / (double)running
                                       : 0.0;
    out->prefill_util       = util;
    out->pure_decode_frac   = frac;
    return QW_OK;
}

void qw_sched_plan_report(const qw_step_plan *plan, uint64_t budget)
{
    if (plan == NULL)
        return;
    double util = budget ? (double)plan->n_prefill_tokens / (double)budget : 0.0;
    printf("step: %d entries (%llu prefill tok, %llu decode tok) "
           "prefill-util %.1f%% of budget %llu\n",
           plan->n_entries,
           (unsigned long long)plan->n_prefill_tokens,
           (unsigned long long)plan->n_decode_tokens,
           util * 100.0,
           (unsigned long long)budget);
}
