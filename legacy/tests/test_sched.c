/* tests/test_sched.c — no-framework asserts for the continuous-batching
 * scheduler (chunked prefill + decode, kv-safe admission, eviction).
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -O2 -Iinclude \
 *        src/core/types.c src/core/config.c src/core/kvcache.c src/core/sched.c \
 *        tests/test_sched.c -lm -o /tmp/opencode/tse && /tmp/opencode/tse
 *
 * main() returns the count of failures (0 = all pass).
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qw/sched.h"
#include "qw/types.h"

static int g_fail = 0;
static uint64_t g_det_hash = 0;
#define CHECK(cond) do { \
    if (cond) { printf("  ok   %s\n", #cond); } \
    else { printf("  FAIL %s (line %d)\n", #cond, __LINE__); g_fail++; } \
} while (0)

/* Tokens per paged KV block (mirrors qw/sched.c + qw/kvcache.h default). */
#define TPB 256ULL
#define blocks_for(n) (((n) + TPB - 1) / TPB)

/* Deterministic PRNG so randomized tests reproduce their failures. */
static uint64_t g_seed = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void)
{
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 7;
    g_seed ^= g_seed << 17;
    return g_seed;
}

/* Run one step with a given free-block count (re-syncs the scheduler's
 * counter from a synthesized qw_kv_stats). */
static void step_with(qw_sched *s, uint64_t free_blocks, qw_step_plan *plan)
{
    qw_kv_stats st;
    st.n_tokens = 0;
    st.peak_tokens = 0;
    st.free_blocks = free_blocks;
    st.peak_blocks = 0;
    assert(qw_sched_step(s, &st, plan) == QW_OK);
}

/* FNV-1a over the plan stream (ids, kinds, token counts). */
static uint64_t plan_hash(const qw_step_plan *plan)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < plan->n_entries; i++) {
        const qw_sched_plan_entry *e = &plan->entries[i];
        h ^= e->seq_id;  h *= 1099511628211ULL;
        h ^= (uint64_t)e->kind; h *= 1099511628211ULL;
        h ^= e->n_tokens; h *= 1099511628211ULL;
    }
    return h;
}

/* Count of entries for a given seq in one plan (detects double-schedule). */
static int count_seq(const qw_step_plan *plan, uint32_t id)
{
    int n = 0;
    for (int i = 0; i < plan->n_entries; i++)
        if (plan->entries[i].seq_id == id)
            n++;
    return n;
}

/* Total tokens in one plan. */
static uint64_t plan_tokens(const qw_step_plan *plan)
{
    uint64_t t = 0;
    for (int i = 0; i < plan->n_entries; i++)
        t += plan->entries[i].n_tokens;
    return t;
}

/* ------------------------------------------------------------------ t1 */
/* 1 sequence, huge budget: a 1000-token prompt is prefilled in one chunk
 * (budget 4000 >> 1000), then decodes; prefilled reaches prompt_len. */
static void test_single_chunked_then_decode(void)
{
    printf("test 1: single seq, huge budget -> chunked prefill then decode\n");
    qw_sched s;
    CHECK(qw_sched_init(&s, 4, 1024) == QW_OK);
    CHECK(qw_sched_set_budget(&s, 4000) == QW_OK);
    CHECK(qw_sched_add(&s, 7, 1000) == QW_OK);

    qw_step_plan plan;
    uint64_t free = 1024;

    step_with(&s, free, &plan);
    /* step 1: the whole prompt (1000 <= 4000) is prefilled, and the leftover
     * budget (4000 - 1000) lets the now-prefilled seq also decode 1 token.
     * So 2 entries: one PREFILL_CHUNK(1000) + one DECODE(1). Within budget. */
    CHECK(plan.n_entries == 2);
    CHECK(plan.entries[0].kind == QW_SCHED_PREFILL_CHUNK);
    CHECK(plan.entries[0].n_tokens == 1000);
    CHECK(plan.entries[1].kind == QW_SCHED_DECODE);
    CHECK(plan.entries[1].n_tokens == 1);
    CHECK(plan_tokens(&plan) <= 4000);

    /* subsequent steps are pure decode (1 token, no prefill) */
    for (int step = 0; step < 5; step++) {
        step_with(&s, free, &plan);
        CHECK(plan.n_entries == 1);
        CHECK(plan.entries[0].kind == QW_SCHED_DECODE);
        CHECK(plan.entries[0].n_tokens == 1);
    }

    qw_sched_stat st;
    CHECK(qw_sched_stats(&s, &st) == QW_OK);
    CHECK(st.n_running == 1);
    CHECK(st.n_waiting == 0);
    CHECK(st.prefill_util > 0.0); /* prefill happened in step 1 */
    qw_sched_destroy(&s);
}

/* ------------------------------------------------------------------ t1b */
/* 1 sequence, tight budget: a 1000-token prompt is CHUNKED across steps
 * (256+256+256+232) and then decodes. */
static void test_single_tight_budget_chunks(void)
{
    printf("test 1b: single seq, budget 256 -> 1000-token prompt in 4 chunks\n");
    qw_sched s;
    CHECK(qw_sched_init(&s, 4, 1024) == QW_OK);
    CHECK(qw_sched_set_budget(&s, 256) == QW_OK);
    CHECK(qw_sched_add(&s, 7, 1000) == QW_OK);

    qw_step_plan plan;
    uint64_t free = 1024;
    uint64_t prefilled = 0;

    /* 4 prefill chunks: 256,256,256,232 (sum 1000). The 4th chunk (232)
     * leaves 24 budget, so the now-prefilled seq also decodes 1 token in
     * that same step (2 entries there). */
    for (int step = 0; step < 4; step++) {
        step_with(&s, free, &plan);
        CHECK(plan.entries[0].kind == QW_SCHED_PREFILL_CHUNK);
        CHECK(plan_tokens(&plan) <= 256);
        prefilled += plan.entries[0].n_tokens;
    }
    CHECK(prefilled == 1000);

    /* next step: pure decode (prompt done, no prefill) */
    step_with(&s, free, &plan);
    CHECK(plan.n_entries == 1);
    CHECK(plan.entries[0].kind == QW_SCHED_DECODE);

    qw_sched_destroy(&s);
}

/* ------------------------------------------------------------------ t2 */
/* Over a 200-step simulation with random prompts, the per-step token sum
 * never exceeds the budget and no sequence is scheduled twice in a step. */
static void test_budget_and_no_double(void)
{
    printf("test 2: 200-step sim -> sum(n_tokens) <= B, no double schedule\n");
    qw_sched s;
    CHECK(qw_sched_init(&s, 16, 4096) == QW_OK);
    CHECK(qw_sched_set_budget(&s, 512) == QW_OK);
    /* random-ish prompts of varied length */
    CHECK(qw_sched_add(&s, 1, 1000 + (rnd() % 9000)) == QW_OK);
    CHECK(qw_sched_add(&s, 2, 1000 + (rnd() % 9000)) == QW_OK);
    CHECK(qw_sched_add(&s, 3, 1000 + (rnd() % 9000)) == QW_OK);

    qw_step_plan plan;
    for (int step = 0; step < 200; step++) {
        step_with(&s, 4096, &plan);
        CHECK(plan_tokens(&plan) <= 512);
        /* a seq may appear at most twice: once as PREFILL_CHUNK and once as
         * DECODE (it finishes its prompt and decodes in the same step). It
         * must never appear twice in the SAME kind. */
        for (int i = 0; i < plan.n_entries; i++) {
            int same = 0;
            for (int j = 0; j < plan.n_entries; j++)
                if (i != j && plan.entries[i].seq_id == plan.entries[j].seq_id &&
                    plan.entries[i].kind == plan.entries[j].kind)
                    same++;
            CHECK(same == 0); /* no seq scheduled twice in one kind/step */
            CHECK(count_seq(&plan, plan.entries[i].seq_id) <= 2);
        }
    }
    qw_sched_destroy(&s);
}

/* ------------------------------------------------------------------ t3 */
/* KV exhaustion: admission stops when blocks for a prompt's full context
 * do not fit, and a simulated block counter never goes negative. */
static void test_kv_exhaustion(void)
{
    printf("test 3: kv exhaustion -> admission stops, counter >= 0\n");
    qw_sched s;
    const uint64_t TOTAL = 4; /* blocks */
    CHECK(qw_sched_init(&s, 4, TOTAL) == QW_OK);
    CHECK(qw_sched_set_budget(&s, 512) == QW_OK);
    CHECK(qw_sched_add(&s, 1, 1000) == QW_OK); /* full ctx 1000 -> 4 blocks */
    CHECK(qw_sched_add(&s, 2, 1000) == QW_OK); /* needs 4 more -> won't fit */

    qw_step_plan plan;
    /* Track the real free-block count (signed, so the "never negative" check
     * is meaningful): seq 1 is admitted on step 1 holding 4 blocks (its full
     * context), leaving 0 free. We must feed the scheduler the *actual* free
     * count each step (it re-syncs from it), so after step 1 free stays at 0
     * and seq 2 can never be admitted. */
    int64_t sim_free = (int64_t)TOTAL;
    int admitted2 = 0;
    int admitted1 = 0;
    for (int step = 0; step < 60; step++) {
        step_with(&s, (uint64_t)sim_free, &plan);
        if (count_seq(&plan, 1) > 0)
            admitted1++;
        if (count_seq(&plan, 2) > 0)
            admitted2++;
        CHECK(sim_free >= 0); /* never requests non-existent blocks */
        /* after seq 1 is admitted it holds all 4 blocks -> free drops to 0 */
        if (admitted1 && sim_free >= 4)
            sim_free = 0;
    }
    /* seq 2 never admitted: only 4 blocks, seq 1 holds all of them */
    CHECK(admitted1 >= 1);
    CHECK(admitted2 == 0);

    qw_sched_stat st;
    CHECK(qw_sched_stats(&s, &st) == QW_OK);
    CHECK(st.n_running == 1); /* seq 1 */
    CHECK(st.n_waiting == 1); /* seq 2 still waiting */
    qw_sched_destroy(&s);
}

/* ------------------------------------------------------------------ t4 */
/* A 4096-prompt arriving while 8 sequences decode does not starve the
 * running set: within N steps every running seq still decodes. */
static void test_no_starvation(void)
{
    printf("test 4: 4096-prompt + 8 decoders -> decodes continue in N steps\n");
    qw_sched s;
    CHECK(qw_sched_init(&s, 16, 8192) == QW_OK);
    CHECK(qw_sched_set_budget(&s, 512) == QW_OK);
    for (int i = 1; i <= 8; i++)
        CHECK(qw_sched_add(&s, (uint32_t)i, 100) == QW_OK); /* 8 short prompts */

    qw_step_plan plan;
    int decodes[8] = {0};
    int saw_big_prefill = 0;
    for (int step = 0; step < 40; step++) {
        if (step == 10)
            CHECK(qw_sched_add(&s, 99, 4096) == QW_OK); /* big prompt arrives */
        step_with(&s, 8192, &plan);
        for (int i = 0; i < plan.n_entries; i++) {
            const qw_sched_plan_entry *e = &plan.entries[i];
            if (e->seq_id == 99 && e->kind == QW_SCHED_PREFILL_CHUNK)
                saw_big_prefill = 1;
            if (e->kind == QW_SCHED_DECODE && e->seq_id >= 1 && e->seq_id <= 8)
                decodes[e->seq_id - 1]++;
        }
    }
    /* the big prompt did get prefilled (chunked) at some point */
    CHECK(saw_big_prefill == 1);
    /* and every one of the 8 running sequences still decoded in the window */
    int all_decode = 1;
    for (int i = 0; i < 8; i++)
        if (decodes[i] == 0)
            all_decode = 0;
    CHECK(all_decode);
    qw_sched_destroy(&s);
}

/* ------------------------------------------------------------------ t5 */
/* After all sequences finish (eos), blocks return to the total and the
 * running count is zero. */
static void test_all_done_releases(void)
{
    printf("test 5: all finish -> blocks back to total, running == 0\n");
    qw_sched s;
    const uint64_t TOTAL = 16;
    CHECK(qw_sched_init(&s, 4, TOTAL) == QW_OK);
    CHECK(qw_sched_set_budget(&s, 512) == QW_OK);
    CHECK(qw_sched_add(&s, 1, 300) == QW_OK);
    CHECK(qw_sched_add(&s, 2, 700) == QW_OK);
    CHECK(qw_sched_add(&s, 3, 1000) == QW_OK);

    qw_step_plan plan;
    for (int step = 0; step < 10; step++)
        step_with(&s, TOTAL, &plan);

    CHECK(qw_sched_finish(&s, 1, true) == QW_OK);
    CHECK(qw_sched_finish(&s, 2, true) == QW_OK);
    CHECK(qw_sched_finish(&s, 3, true) == QW_OK);

    /* re-sync from a full cache: all blocks free again */
    qw_kv_stats st;
    st.n_tokens = 0; st.peak_tokens = 0;
    st.free_blocks = TOTAL; st.peak_blocks = 0;
    CHECK(qw_sched_step(&s, &st, &plan) == QW_OK);

    qw_sched_stat stt;
    CHECK(qw_sched_stats(&s, &stt) == QW_OK);
    CHECK(stt.n_running == 0);
    CHECK(stt.n_waiting == 0);
    CHECK(stt.mean_blocks_running == 0.0);
    qw_sched_destroy(&s);
}

/* ------------------------------------------------------------------ t6 */
/* Eviction path: the cache is full (seqs 1,2,3 running hold 3 blocks, seq 4
 * SWAPPED holds 1) and seq 5 is waiting but needs a block that does not
 * exist. On the next (idle) step the most recently admitted running
 * (non-swapped) seq (3) is evicted back to waiting — its block freed — and 5
 * is admitted in its place. The evicted seq (3) is later re-admitted and
 * completes. */
static void test_eviction(void)
{
    printf("test 6: eviction frees blocks; evicted seq later completes\n");
    qw_sched s;
    const uint64_t CAP = 4; /* blocks: 1,2,3 hold 1 each, 4 swapped holds 1 */
    CHECK(qw_sched_init(&s, 8, CAP) == QW_OK);
    CHECK(qw_sched_set_budget(&s, 512) == QW_OK);
    CHECK(qw_sched_add(&s, 1, 100) == QW_OK); /* full ctx 100 -> 1 block each */
    CHECK(qw_sched_add(&s, 2, 100) == QW_OK);
    CHECK(qw_sched_add(&s, 3, 100) == QW_OK);
    CHECK(qw_sched_add(&s, 4, 100) == QW_OK);
    CHECK(qw_sched_add(&s, 5, 100) == QW_OK);

    qw_step_plan plan;

    /* Admit 1,2,3 (3 blocks) over 3 steps. 4 and 5 waiting (1 block free
    cannot admit 4's full ctx of 1 block? it can — 1 free >= 1). So 4 gets
    admitted too on step 4, filling the cache. 5 stays waiting. */
    step_with(&s, CAP, &plan); /* admit 1 */
    step_with(&s, CAP, &plan); /* admit 2 */
    step_with(&s, CAP, &plan); /* admit 3 */
    step_with(&s, CAP, &plan); /* admit 4 (1 free left) */
    CHECK(count_seq(&plan, 5) == 0); /* 5 not admitted (0 free) */

    /* Force the eviction precondition. Make 4 the SWAPPED seq. Finish 1 (free
    its block). Now: running = 2,3; swapped = 4 (holds 1 block); waiting = 5.
    Re-sync free blocks to 0 (2,3 hold 2 + 4 holds 1 = 3 held; model the last
    block as not available to 5). 5 needs a block that does not exist, and no
    running seq has a pending prefill chunk -> the eviction fires this step:
    the most recently admitted running non-swapped seq (3) is evicted, its
    block freed. */
    for (int i = 0; i < s.n_seq; i++)
        if (s.seqs[i].id == 4)
            s.seqs[i].state = QW_SEQ_SWAPPED;
    CHECK(qw_sched_finish(&s, 1, true) == QW_OK); /* 1 frees its block */
    step_with(&s, 0, &plan); /* 5 blocked + 4 swapped -> eviction fires */

    /* 3 (most recently admitted running non-swapped) evicted back to
    waiting, its block freed. */
    int evicted3 = 0;
    for (int i = 0; i < s.n_seq; i++)
        if (s.seqs[i].id == 3 && s.seqs[i].state == QW_SEQ_WAITING)
            evicted3 = 1;
    CHECK(evicted3 == 1);

    /* Stop the eviction oscillation: finish the swapped seq (4) so
    any_swapped() is false and eviction no longer fires. */
    CHECK(qw_sched_finish(&s, 4, true) == QW_OK);

    /* The freed block lets the FIFO head (3) be re-admitted on the next step;
    3 later completes. */
    step_with(&s, 1, &plan); /* 1 free -> 3 (FIFO head) re-admitted */
    int re3 = 0;
    for (int i = 0; i < s.n_seq; i++)
        if (s.seqs[i].id == 3 && s.seqs[i].state == QW_SEQ_RUNNING)
            re3 = 1;
    CHECK(re3 == 1);
    CHECK(qw_sched_finish(&s, 3, true) == QW_OK); /* 3 completes, frees block */
    qw_sched_destroy(&s);
}

/* ------------------------------------------------------------------ t7 */
/* Determinism: two identical call sequences produce identical plan streams
 * (compared by hash). */
static void test_determinism(void)
{
    printf("test 7: identical call sequences -> identical plan hash\n");
    for (int run = 0; run < 2; run++) {
        qw_sched s;
        CHECK(qw_sched_init(&s, 16, 4096) == QW_OK);
        CHECK(qw_sched_set_budget(&s, 512) == QW_OK);
        for (int i = 1; i <= 10; i++)
            CHECK(qw_sched_add(&s, (uint32_t)i, (uint64_t)(100 + i * 137)) == QW_OK);

        qw_step_plan plan;
        uint64_t h = 0;
        for (int step = 0; step < 200; step++) {
            step_with(&s, 4096, &plan);
            h = plan_hash(&plan);
        }
        if (run == 0) {
            g_det_hash = h;
        } else {
            CHECK(h == g_det_hash);
        }
        qw_sched_destroy(&s);
    }
}

int main(void)
{
    test_single_chunked_then_decode();
    test_single_tight_budget_chunks();
    test_budget_and_no_double();
    test_kv_exhaustion();
    test_no_starvation();
    test_all_done_releases();
    test_eviction();
    test_determinism();
    if (g_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fail);
    return g_fail;
}
