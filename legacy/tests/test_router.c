/* tests/test_router.c — no-framework asserts for the MoE top-k router.
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -Iinclude \
 *        src/core/types.c src/core/router.c tests/test_router.c
 *
 * main() returns the count of failures (0 = all pass).
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qw/router.h"
#include "qw/types.h"

static int g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { printf("  ok   %s\n", #cond); } \
    else { printf("  FAIL %s (line %d)\n", #cond, __LINE__); g_fail++; } \
} while (0)

static float tfabsf(float x) { return (x < 0.0f) ? -x : x; }

/* Deterministic LCG for test data (fixed seeds per test). */
static uint64_t g_seed;
static uint64_t lcg(void)
{
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_seed >> 32;
}

/* Uniform in [-1,1). */
static float rand_f32(void)
{
    return (float)((double)(lcg() & 0xfffff) / 1048576.0 * 2.0 - 1.0);
}

/* ---------- test 1: known small example, exact expected answer ---------- */
static void test_known(void)
{
    printf("test 1: known 8-expert vector, exact experts + weights\n");
    const int n = 8, k = 3;
    float s[8] = { 5.0f, -1.0f, 7.0f, 2.0f, 0.5f, 3.0f, 0.5f, -3.0f };
    /* top-k set: {2:7, 0:5, 5:3}. Sorted by expert asc: 0, 2, 5.
     * softmax over {7,5,3}: e^4, e^2, 1 (max-subtracted by 7). */
    float e4 = expf(4.0f), e2 = expf(2.0f), e0 = 1.0f;
    float sum = e4 + e2 + e0;

    qw_route out[k];
    qw_err e = qw_router_topk(s, n, k, out, NULL);
    CHECK(e == QW_OK);
    CHECK(out[0].expert == 0);
    CHECK(out[1].expert == 2);
    CHECK(out[2].expert == 5);
    CHECK(tfabsf(out[0].weight - e2 / sum) < 1e-7f);
    CHECK(tfabsf(out[1].weight - e4 / sum) < 1e-7f);
    CHECK(tfabsf(out[2].weight - e0 / sum) < 1e-7f);
    float wsum = out[0].weight + out[1].weight + out[2].weight;
    CHECK(tfabsf(wsum - 1.0f) < 1e-6f);
}

/* ---------------------------- test 2: tie-breaking ----------------------- */
static void test_ties(void)
{
    printf("test 2: tied scores pick lowest expert index\n");
    const int n = 8, k = 3;
    /* experts 1,3,5,7 all score 4.0; expert 0 scores 5.0. k=3: the 3 lowest
     * indices among the tie {1,3,5,7} plus the 5.0 at index 0. */
    float s[8] = { 5.0f, 4.0f, 1.0f, 4.0f, 0.0f, 4.0f, -1.0f, 4.0f };
    qw_route out[k];
    qw_err e = qw_router_topk(s, n, k, out, NULL);
    CHECK(e == QW_OK);
    CHECK(out[0].expert == 0); /* 5.0, unique max */
    CHECK(out[1].expert == 1); /* lowest index of the 4.0 tie */
    CHECK(out[2].expert == 3); /* next lowest index of the tie */
}

/* ---------------------------- test 3: NaN rejected ---------------------- */
static void test_nan_rejected(void)
{
    printf("test 3: NaN input rejected, not ranked\n");
    float s[8] = { 5.0f, -1.0f, 7.0f, 2.0f, 0.5f, 3.0f, 0.5f, -3.0f };
    s[3] = NAN; /* index 3 would rank 4th; must not be selected or crash */
    qw_route out[3];
    qw_err e = qw_router_topk(s, 8, 3, out, NULL);
    CHECK(e == QW_ERR_FORMAT);

    /* out-of-band NaN: even a huge NaN must not win a comparison. */
    float s2[4] = { 1.0f, 2.0f, NAN, 3.0f };
    qw_err e2 = qw_router_topk(s2, 4, 2, out, NULL);
    CHECK(e2 == QW_ERR_FORMAT);

    /* +inf / -inf are also non-finite and rejected. */
    float s3[4] = { 1.0f, 2.0f, INFINITY, 3.0f };
    CHECK(qw_router_topk(s3, 4, 2, out, NULL) == QW_ERR_FORMAT);
    float s4[4] = { 1.0f, 2.0f, -INFINITY, 3.0f };
    CHECK(qw_router_topk(s4, 4, 2, out, NULL) == QW_ERR_FORMAT);
}

/* ---------------------------- test 4: k=1 and k=n ----------------------- */
static void test_edge_k(void)
{
    printf("test 4: k=1 (single argmax) and k=n (full softmax)\n");
    const int n = 512;
    static float s[512];
    for (int i = 0; i < n; i++)
        s[i] = (float)(i % 7) - 3.0f;
    s[42] = 99.0f; /* unique argmax */

    /* k=1: the argmax, weight 1.0 */
    qw_route one[1];
    qw_err e = qw_router_topk(s, n, 1, one, NULL);
    CHECK(e == QW_OK);
    CHECK(one[0].expert == 42);
    CHECK(one[0].weight == 1.0f);

    /* k=n: every expert, weights softmax over all 512, sorted asc by index.
     * k > QW_ROUTER_STACK_K, so scratch is mandatory (caller's arena). */
    qw_route *all = (qw_route *)malloc((size_t)n * sizeof(qw_route));
    void *scratch = malloc((size_t)(n + 1) * 8); /* k+1 route_slots, 8 B each */
    assert(all != NULL && scratch != NULL);
    e = qw_router_topk(s, n, n, all, scratch);
    CHECK(e == QW_OK);
    /* and scratch==NULL with k=512 is rejected, not crash */
    CHECK(qw_router_topk(s, n, n, all, NULL) == QW_ERR_RANGE);
    int sorted = 1;
    for (int i = 1; i < n; i++)
        if (all[i - 1].expert != (uint32_t)(i - 1))
            sorted = 0;
    CHECK(sorted);
    float wsum = 0.0f;
    for (int i = 0; i < n; i++)
        wsum += all[i].weight;
    CHECK(tfabsf(wsum - 1.0f) < 1e-6f);
    free(all);
    free(scratch);
}

/* --------------------- test 5: softmax helper sums to 1 ------------------ */
static void test_softmax_helper(void)
{
    printf("test 5: qw_router_softmax_inplace sums to 1 within 1e-6\n");
    float a[5] = { 0.0f, 1.0f, 2.0f, 100.0f, -50.0f }; /* large max: no overflow */
    qw_err e = qw_router_softmax_inplace(a, 5);
    CHECK(e == QW_OK);
    float sum = 0.0f;
    for (int i = 0; i < 5; i++)
        sum += a[i];
    CHECK(tfabsf(sum - 1.0f) < 1e-6f);
    /* max element (index 3) gets the largest weight */
    CHECK(a[3] > a[0] && a[3] > a[1] && a[3] > a[2] && a[3] > a[4]);

    float b[1] = { 3.14f };
    CHECK(qw_router_softmax_inplace(b, 1) == QW_OK);
    CHECK(b[0] == 1.0f);

    float c[2] = { 1.0f, NAN };
    CHECK(qw_router_softmax_inplace(c, 2) == QW_ERR_FORMAT);
}

/* ------------------- test 6: output sorted ascending by index ----------- */
static void test_sorted_output(void)
{
    printf("test 6: output sorted ascending by expert index\n");
    const int n = 20, k = 6;
    static float s[20];
    /* random scores, high values scattered at non-adjacent indices */
    g_seed = 777;
    for (int i = 0; i < n; i++)
        s[i] = rand_f32();
    s[3] = 10.0f; s[17] = 9.0f; s[5] = 8.0f; s[11] = 7.0f;
    s[0] = 6.0f;  s[19] = 5.0f; /* exactly k=6 top */

    qw_route out[k];
    qw_err e = qw_router_topk(s, n, k, out, NULL);
    CHECK(e == QW_OK);
    int asc = 1;
    for (int i = 1; i < k; i++)
        if (out[i - 1].expert >= out[i].expert)
            asc = 0;
    CHECK(asc);
    /* expected set {0,3,5,11,17,19} in asc order */
    uint32_t exp_idx[6] = { 0, 3, 5, 11, 17, 19 };
    int match = 1;
    for (int i = 0; i < k; i++)
        if (out[i].expert != exp_idx[i])
            match = 0;
    CHECK(match);
}

/* ------------- test 7: top-k set matches naive full sort (1000x512) ------ */
struct ref_entry {
    float    score;
    uint32_t idx;
};

static void test_vs_naive(void)
{
    printf("test 7: top-k set matches naive full-sort reference (1000x512)\n");
    const int n = 512, k = 10, iters = 1000;
    static float s[512];
    static struct ref_entry ref[512];
    qw_route *out = (qw_route *)malloc((size_t)k * sizeof(qw_route));
    assert(out != NULL);
    int bad_set = 0, bad_sort = 0, bad_err = 0;

    g_seed = 12345;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++)
            s[i] = rand_f32();
        /* occasional ties to exercise the tie-break */
        if (it % 50 == 0) { s[100] = s[200] = s[300] = 0.5f; }

        /* naive reference: full sort desc by (score, then -index), take k */
        for (int i = 0; i < n; i++) {
            ref[i].score = s[i];
            ref[i].idx = (uint32_t)i;
        }
        for (int i = 1; i < n; i++) {
            struct ref_entry key = ref[i];
            int j = i - 1;
            /* shift while ref[j] ranks BELOW key: lower score, or equal
             * score with higher index (the deterministic tie-break). */
            while (j >= 0 && (ref[j].score < key.score ||
                              (ref[j].score == key.score &&
                               ref[j].idx > key.idx))) {
                ref[j + 1] = ref[j];
                j--;
            }
            ref[j + 1] = key;
        }

        qw_err e = qw_router_topk(s, n, k, out, NULL);
        if (e != QW_OK) { bad_err++; continue; }

        for (int i = 1; i < k; i++)
            if (out[i - 1].expert >= out[i].expert)
                bad_sort++;
        /* compare the top-k SETS (order-independent): the router outputs
         * ascending by expert index, the reference is descending by score.
         * An expert is in the reference top-k iff it appears at a position
         * < k in the (deterministically ordered) reference. */
        for (int i = 0; i < k; i++) {
            int found = 0;
            for (int r = 0; r < k; r++)
                if (ref[r].idx == out[i].expert)
                    found = 1;
            if (!found)
                bad_set++;
        }
    }
    CHECK(bad_err == 0);
    CHECK(bad_set == 0);
    CHECK(bad_sort == 0);
    printf("  iters=%d bad_err=%d bad_set=%d bad_sort=%d\n",
           iters, bad_err, bad_set, bad_sort);
    free(out);
}

/* ------------------------- test 8: expert_load counter ------------------- */
static void test_expert_load(void)
{
    printf("test 8: qw_router_expert_load counts across a batch\n");
    uint32_t counts[512];
    memset(counts, 0, sizeof(counts));

    /* 3 tokens x 2 routes = 6 routes total (batch of 3, k=2) */
    qw_route routes[6] = {
        { 4, 0.5f }, { 4, 0.5f },
        { 7, 0.5f }, { 4, 0.5f },
        { 0, 0.5f }, { 7, 0.5f },
    };
    qw_err e = qw_router_expert_load(routes, 6, counts);
    CHECK(e == QW_OK);
    CHECK(counts[0] == 1);
    CHECK(counts[4] == 3);
    CHECK(counts[7] == 2);
    uint32_t total = 0;
    for (int i = 0; i < 512; i++)
        total += counts[i];
    CHECK(total == 6);

    CHECK(qw_router_expert_load(NULL, 1, counts) == QW_ERR_NULL);
    CHECK(qw_router_expert_load(routes, 1, NULL) == QW_ERR_NULL);
    CHECK(qw_router_expert_load(routes, -1, counts) == QW_ERR_RANGE);
}

/* ------------------- test 9: scratch path for large k -------------------- */
static void test_arena_path(void)
{
    printf("test 9: scratch path (k > stack limit of %d)\n", QW_ROUTER_STACK_K);
    const int n = 512, k = 64; /* > QW_ROUTER_STACK_K */
    static float s[512];
    g_seed = 99;
    for (int i = 0; i < n; i++)
        s[i] = rand_f32();

    /* scratch must hold k+1 route_slot-sized entries (8 B each). */
    size_t need = (size_t)(k + 1) * 8;
    void *scratch = malloc(need);
    assert(scratch != NULL);

    qw_route *out = (qw_route *)malloc((size_t)k * sizeof(qw_route));
    assert(out != NULL);
    qw_err e = qw_router_topk(s, n, k, out, scratch);
    CHECK(e == QW_OK);
    int asc = 1;
    for (int i = 1; i < k; i++)
        if (out[i - 1].expert >= out[i].expert)
            asc = 0;
    CHECK(asc);
    float wsum = 0.0f;
    for (int i = 0; i < k; i++)
        wsum += out[i].weight;
    CHECK(tfabsf(wsum - 1.0f) < 1e-6f);

    /* scratch == NULL with k > 16 must be rejected, not crash. */
    CHECK(qw_router_topk(s, n, k, out, NULL) == QW_ERR_RANGE);

    free(out);
    free(scratch);
}

/* ---------------------- test 10: bad input rejected ---------------------- */
static void test_bad_input(void)
{
    printf("test 10: bad input rejected\n");
    float s[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    qw_route out[4];
    CHECK(qw_router_topk(NULL, 4, 2, out, NULL) == QW_ERR_NULL);
    CHECK(qw_router_topk(s, 4, 0, out, NULL) == QW_ERR_RANGE);
    CHECK(qw_router_topk(s, 4, 5, out, NULL) == QW_ERR_RANGE);
    CHECK(qw_router_topk(s, 0, 1, out, NULL) == QW_ERR_RANGE);
    CHECK(qw_router_topk(s, 4, 2, NULL, NULL) == QW_ERR_NULL);
}

int main(void)
{
    test_known();
    test_ties();
    test_nan_rejected();
    test_edge_k();
    test_softmax_helper();
    test_sorted_output();
    test_vs_naive();
    test_expert_load();
    test_arena_path();
    test_bad_input();
    if (g_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fail);
    return g_fail;
}
