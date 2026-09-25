/* tests/test_sampler.c — no-framework asserts for logits sampling.
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -Iinclude \
 *        src/core/types.c src/core/sampler.c tests/test_sampler.c -lm
 *
 * main() returns the count of failures (0 = all pass).
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qw/sampler.h"
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
static float rand_f32(void)
{
    return (float)((double)(lcg() & 0xfffff) / 1048576.0 * 2.0 - 1.0);
}

/* ---------------------------- test 1: greedy --------------------------- */
static void test_greedy(void)
{
    printf("test 1: greedy picks the max; ties -> lowest index\n");
    const int n = 8;
    float a[8] = { 1.0f, 5.0f, 5.0f, 2.0f, 0.0f, 5.0f, -1.0f, 3.0f };
    uint32_t tok = 999;
    qw_err e = qw_sampler_greedy(a, n, &tok);
    CHECK(e == QW_OK);
    CHECK(tok == 1); /* 5.0 at 1,2,5 -> lowest index 1 */

    float b[4] = { -2.0f, -7.0f, -7.0f, -4.0f };
    e = qw_sampler_greedy(b, 4, &tok);
    CHECK(e == QW_OK);
    CHECK(tok == 0); /* max is -2.0 */

    float nanv[3] = { 1.0f, 2.0f, NAN };
    e = qw_sampler_greedy(nanv, 3, &tok);
    CHECK(e == QW_ERR_FORMAT);
}

/* ------------------------------ test 2: top_k -------------------------- */
static void test_top_k(void)
{
    printf("test 2: top_k keeps exactly k; -inf (mask==0) count is n-k\n");
    const int n = 64, k = 10;
    float s[64];
    g_seed = 42;
    for (int i = 0; i < n; i++)
        s[i] = rand_f32();

    uint8_t mask[64];
    memset(mask, 0xFF, sizeof mask);
    qw_sampler smp;
    qw_sampler_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.temperature = 1.0f;
    cfg.top_p = 1.0f;
    cfg.top_k = k;
    cfg.repetition_penalty = 1.0f;
    cfg.seed = 1;
    size_t sz = qw_sampler_scratch_size(k, n);
    char *scratch = (char *)malloc(sz);
    CHECK(scratch != NULL);
    qw_err e = qw_sampler_init(&smp, &cfg, n, scratch);
    CHECK(e == QW_OK);
    e = qw_sampler_top_k(&smp, s, n, k, mask);
    CHECK(e == QW_OK);

    int kept = 0;
    for (int i = 0; i < n; i++)
        kept += (mask[i] == 1) ? 1 : 0;
    CHECK(kept == k);
    CHECK((n - kept) == (n - k));

    /* Kept set must be the k largest values. */
    float expect_kth;
    {
        /* Find the k-th largest via selection (small n, test only). */
        float tmp[64];
        memcpy(tmp, s, sizeof s);
        for (int i = 0; i < n - 1; i++)
            for (int j = 0; j < n - 1 - i; j++)
                if (tmp[j] < tmp[j + 1]) {
                    float t = tmp[j]; tmp[j] = tmp[j + 1]; tmp[j + 1] = t;
                }
        expect_kth = tmp[k - 1];
    }
    for (int i = 0; i < n; i++)
        if (s[i] > expect_kth)
            CHECK(mask[i] == 1);
    free(scratch);
}

/* --------------------- test 3: top_k tie determinism ------------------- */
static void test_top_k_ties(void)
{
    printf("test 3: top_k with boundary duplicates -> deterministic set\n");
    const int n = 8, k = 3;
    float s[8] = { 4.0f, 1.0f, 4.0f, 0.0f, 4.0f, -1.0f, 4.0f, 2.0f };
    /* Four 4.0s at {0,2,4,6}, k=3 -> lowest indices {0,2,4} kept. */

    uint8_t m1[8], m2[8];
    qw_sampler smp;
    qw_sampler_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.temperature = 1.0f; cfg.top_p = 1.0f; cfg.top_k = k;
    cfg.repetition_penalty = 1.0f; cfg.seed = 7;
    size_t sz = qw_sampler_scratch_size(k, n);
    char *scratch = (char *)malloc(sz);
    qw_err e = qw_sampler_init(&smp, &cfg, n, scratch);
    CHECK(e == QW_OK);
    e = qw_sampler_top_k(&smp, s, n, k, m1);
    CHECK(e == QW_OK);
    e = qw_sampler_top_k(&smp, s, n, k, m2);
    CHECK(e == QW_OK);
    CHECK(memcmp(m1, m2, sizeof m1) == 0);
    CHECK(m1[0] == 1 && m1[1] == 0);
    CHECK(m1[2] == 1 && m1[3] == 0);
    CHECK(m1[4] == 1 && m1[5] == 0);
    CHECK(m1[6] == 0); /* 4th duplicate dropped (highest index) */
    CHECK(m1[7] == 0);
    free(scratch);
}

/* --------------------------- test 4: temperature ----------------------- */
static void test_temperature(void)
{
    printf("test 4: temperature t=0 -> argmax; t>0 preserves argmax\n");
    const int n = 5;
    float a[5] = { 0.1f, 0.9f, 0.2f, 0.9f, 0.0f };
    float out[5];

    qw_err e = qw_sampler_temperature(a, n, 0.0f, out);
    CHECK(e == QW_OK);
    CHECK(out[1] == 1.0f); /* argmax tie 1,3 -> lowest index 1 */
    float sum0 = out[0] + out[1] + out[2] + out[3] + out[4];
    CHECK(tfabsf(sum0 - 1.0f) < 1e-6f);
    CHECK(out[0] == 0.0f && out[2] == 0.0f && out[3] == 0.0f && out[4] == 0.0f);

    /* t > 0 on a peaked distribution: argmax index preserved. */
    float peak[5] = { -10.0f, -1.0f, -9.0f, -11.0f, -8.0f };
    for (float t = 0.1f; t <= 2.01f; t += 0.5f) {
        e = qw_sampler_temperature(peak, 5, t, out);
        CHECK(e == QW_OK);
        int mx = 0;
        for (int i = 1; i < 5; i++)
            if (out[i] > out[mx])
                mx = i;
        CHECK(mx == 1);
    }

    /* t < 0 and non-finite rejected. */
    e = qw_sampler_temperature(a, n, -1.0f, out);
    CHECK(e == QW_ERR_RANGE);
    e = qw_sampler_temperature(a, n, NAN, out);
    CHECK(e == QW_ERR_RANGE);
    e = qw_sampler_temperature(a, n, INFINITY, out);
    CHECK(e == QW_ERR_RANGE);

    /* Very small t must not overflow. */
    float small[3] = { 0.0f, 1e-4f, 2.0e-4f };
    e = qw_sampler_temperature(small, 3, 1e-30f, out);
    CHECK(e == QW_OK);
    for (int i = 0; i < 3; i++)
        CHECK(isfinite(out[i]));
    CHECK(out[2] == 1.0f || out[2] > 0.9999f);

    /* NaN input rejected. */
    float nanv[3] = { 1.0f, NAN, 2.0f };
    e = qw_sampler_temperature(nanv, 3, 1.0f, out);
    CHECK(e == QW_ERR_FORMAT);
}

/* ------------------------------ test 5: top_p -------------------------- */
static void test_top_p(void)
{
    printf("test 5: top_p p=1 keeps all; p->0 keeps ~1; mass >= p\n");
    const int n = 100;
    float s[100];
    g_seed = 1234;
    for (int i = 0; i < n; i++)
        s[i] = rand_f32();

    uint8_t mask[100];
    for (int i = 0; i < n; i++)
        mask[i] = 1; /* no top_k: all allowed */

    qw_sampler smp;
    qw_sampler_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    /* top_k = n so the pbuf covers every token (top_p over all-allow mask). */
    cfg.temperature = 1.0f; cfg.top_p = 1.0f; cfg.top_k = n;
    cfg.repetition_penalty = 1.0f; cfg.seed = 3;
    size_t sz = qw_sampler_scratch_size(n, n);
    char *scratch = (char *)malloc(sz);
    qw_err e = qw_sampler_init(&smp, &cfg, n, scratch);
    CHECK(e == QW_OK);

    uint32_t out[100];
    int nk = -1;

    /* p = 1.0 keeps everything. */
    e = qw_sampler_top_p(&smp, s, mask, 1.0f, out, n, &nk);
    CHECK(e == QW_OK);
    CHECK(nk == n);

    /* p near 0 keeps ~1 token. */
    e = qw_sampler_top_p(&smp, s, mask, 0.001f, out, n, &nk);
    CHECK(e == QW_OK);
    CHECK(nk >= 1 && nk <= 3);

    /* Probability mass of the kept set is >= p. */
    for (float p = 0.3f; p <= 0.95f; p += 0.15f) {
        e = qw_sampler_top_p(&smp, s, mask, p, out, n, &nk);
        CHECK(e == QW_OK);
        /* Re-softmax over kept set to get the mass. */
        float mx = -INFINITY;
        for (int i = 0; i < nk; i++)
            if (s[out[i]] > mx) mx = s[out[i]];
        float all = 0.0f, kept = 0.0f;
        for (int i = 0; i < n; i++) all += expf(s[i] - mx);
        for (int i = 0; i < nk; i++) kept += expf(s[out[i]] - mx);
        float mass = kept / all;
        CHECK(mass >= p - 1e-4f);
    }

    /* top_p respects the top_k mask: feed a mask from top_k. */
    float big[100];
    for (int i = 0; i < n; i++)
        big[i] = (float)i; /* distinct, ascending */
    uint8_t mk[100];
    qw_sampler smp2;
    qw_sampler_cfg cfg2 = cfg;
    size_t sz2 = qw_sampler_scratch_size(n, n);
    char *scratch2 = (char *)malloc(sz2);
    e = qw_sampler_init(&smp2, &cfg2, n, scratch2);
    CHECK(e == QW_OK);
    e = qw_sampler_top_k(&smp2, big, n, 10, mk);
    CHECK(e == QW_OK);
    e = qw_sampler_top_p(&smp2, big, mk, 1.0f, out, n, &nk);
    CHECK(e == QW_OK);
    CHECK(nk == 10); /* only the 10 mask-allowed tokens */
    for (int i = 0; i < n; i++)
        if (mk[i])
            CHECK(out[i] == (uint32_t)i);
    free(scratch);
    free(scratch2);
}

/* --------------------------- test 6: sampling -------------------------- */
static void test_sampling(void)
{
    printf("test 6: 100000 draws match known 4-token frequencies (1%%)\n");
    const int n = 4;
    float probs[4] = { 0.5f, 0.25f, 0.15f, 0.10f };
    uint32_t tok;

    qw_sampler smp;
    qw_sampler_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.temperature = 1.0f; cfg.top_p = 1.0f; cfg.top_k = 0;
    cfg.repetition_penalty = 1.0f; cfg.seed = 99;
    size_t sz = qw_sampler_scratch_size(0, n);
    char *scratch = (char *)malloc(sz);
    qw_err e = qw_sampler_init(&smp, &cfg, n, scratch);
    CHECK(e == QW_OK);

    uint64_t cnt[4] = { 0, 0, 0, 0 };
    const int N = 100000;
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (qw_sampler_sample(&smp, probs, n, &tok) != QW_OK) {
            ok = 0;
            break;
        }
        cnt[tok]++;
    }
    CHECK(ok);
    for (int i = 0; i < n; i++) {
        float got = (float)cnt[i] / N;
        CHECK(tfabsf(got - probs[i]) < 0.01f);
    }

    /* RNG reproducible from a seed; different seeds -> different streams. */
    qw_sampler a, b, c;
    e = qw_sampler_init(&a, &cfg, n, scratch);
    CHECK(e == QW_OK);
    e = qw_sampler_init(&b, &cfg, n, scratch);
    CHECK(e == QW_OK);
    qw_sampler_cfg cfg_c = cfg;
    cfg_c.seed = 100;
    e = qw_sampler_init(&c, &cfg_c, n, scratch);
    CHECK(e == QW_OK);

    /* Compare the whole 64-draw stream, not one draw: two different seeds
     * can trivially collide on a single token, but not on 64 in a row. */
    uint32_t sa[64], sb[64], sc[64];
    for (int i = 0; i < 64; i++) {
        (void)qw_sampler_sample(&a, probs, n, &sa[i]);
        (void)qw_sampler_sample(&b, probs, n, &sb[i]);
        (void)qw_sampler_sample(&c, probs, n, &sc[i]);
    }
    CHECK(memcmp(sa, sb, sizeof sa) == 0); /* same seed -> identical stream */
    CHECK(memcmp(sa, sc, sizeof sc) != 0); /* diff seed -> different stream */
    free(scratch);
}

/* ------------------------ test 7: repetition penalty ------------------- */
static void test_repetition_penalty(void)
{
    printf("test 7: repetition penalty, sign rule\n");
    float a[5] = { 2.0f, -2.0f, 4.0f, 0.5f, -4.0f };
    float orig[5];
    memcpy(orig, a, sizeof a);
    uint32_t prev[3] = { 0, 1, 4 };
    float pen = 2.0f;

    qw_err e = qw_sampler_repetition_penalty(a, 5, prev, 3, pen);
    CHECK(e == QW_OK);
    CHECK(a[0] == 1.0f);            /* positive: 2/2 */
    CHECK(a[1] == -4.0f);           /* negative: -2*2 (pushed down) */
    CHECK(a[4] == -8.0f);           /* negative: -4*2 */
    CHECK(a[2] == 4.0f && a[3] == 0.5f); /* untouched */

    /* penalty 1.0 is a no-op. */
    float np[5] = { 2.0f, -2.0f, 4.0f, 0.5f, -4.0f };
    qw_err e2 = qw_sampler_repetition_penalty(np, 5, prev, 3, 1.0f);
    CHECK(e2 == QW_OK);
    CHECK(memcmp(np, orig, sizeof np) == 0);

    /* NaN rejected. */
    float nanv[3] = { 1.0f, NAN, 2.0f };
    uint32_t one = 1;
    e = qw_sampler_repetition_penalty(nanv, 3, &one, 1, 1.5f);
    CHECK(e == QW_ERR_FORMAT);

    /* Out-of-range prev token rejected. */
    float ok[3] = { 1.0f, 1.0f, 1.0f };
    uint32_t oob = 10;
    e = qw_sampler_repetition_penalty(ok, 3, &oob, 1, 1.5f);
    CHECK(e == QW_ERR_RANGE);
}

/* --------------------- test 8: NaN rejected, no prop ------------------- */
static void test_nan_reject(void)
{
    printf("test 8: NaN in logits rejected, not propagated\n");
    const int n = 16, k = 4;
    float s[16];
    g_seed = 7;
    for (int i = 0; i < n; i++)
        s[i] = rand_f32();
    s[5] = NAN;

    uint8_t mask[16];
    qw_sampler smp;
    qw_sampler_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.temperature = 1.0f; cfg.top_p = 1.0f; cfg.top_k = k;
    cfg.repetition_penalty = 1.0f; cfg.seed = 1;
    size_t sz = qw_sampler_scratch_size(k, n);
    char *scratch = (char *)malloc(sz);
    qw_err e = qw_sampler_init(&smp, &cfg, n, scratch);
    CHECK(e == QW_OK);
    e = qw_sampler_top_k(&smp, s, n, k, mask);
    CHECK(e == QW_ERR_FORMAT);

    float out[16];
    for (int i = 0; i < n; i++)
        out[i] = 0.0f;
    e = qw_sampler_temperature(s, n, 1.0f, out);
    CHECK(e == QW_ERR_FORMAT);
    for (int i = 0; i < n; i++)
        CHECK(out[i] == 0.0f); /* output untouched on the reject path */
    free(scratch);
}

/* ------------------------- test 9: guard bytes ------------------------- */
static void test_no_ooo_write(void)
{
    printf("test 9: nothing writes outside the output buffer (guards)\n");
    const int n = 8, k = 3, guard = 8;
    float s[8] = { 0.5f, 9.0f, 1.0f, 8.0f, 2.0f, 8.0f, -1.0f, 3.0f };

    uint8_t mask[8 + 2 * guard];
    for (int i = 0; i < 8 + 2 * guard; i++)
        mask[i] = 0xAB;
    uint32_t out[8 + 2 * guard];
    for (int i = 0; i < 8 + 2 * guard; i++)
        out[i] = 0xDEADBEEFu;

    qw_sampler smp;
    qw_sampler_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.temperature = 1.0f; cfg.top_p = 1.0f; cfg.top_k = k;
    cfg.repetition_penalty = 1.0f; cfg.seed = 5;
    size_t sz = qw_sampler_scratch_size(k, n);
    char *scratch = (char *)malloc(sz);
    qw_err e = qw_sampler_init(&smp, &cfg, n, scratch);
    CHECK(e == QW_OK);

    e = qw_sampler_top_k(&smp, s, n, k, mask + guard);
    CHECK(e == QW_OK);
    for (int i = 0; i < guard; i++) {
        CHECK(mask[i] == 0xAB);
        CHECK(mask[n + guard + i] == 0xAB);
        CHECK(out[i] == 0xDEADBEEFu);
        CHECK(out[n + guard + i] == 0xDEADBEEFu);
    }
    /* logits must be unmodified by top_k. */
    float expect[8] = { 0.5f, 9.0f, 1.0f, 8.0f, 2.0f, 8.0f, -1.0f, 3.0f };
    CHECK(memcmp(s, expect, sizeof s) == 0);

    /* top_p writes into out + guard; guards intact. */
    uint8_t allone[8] = { 1,1,1,1,1,1,1,1 };
    int nk = -1;
    e = qw_sampler_top_p(&smp, s, allone, 1.0f, out + guard, n, &nk);
    CHECK(e == QW_OK);
    CHECK(nk == n);
    for (int i = 0; i < guard; i++) {
        CHECK(out[i] == 0xDEADBEEFu);
        CHECK(out[n + guard + i] == 0xDEADBEEFu);
    }
    free(scratch);
}

int main(void)
{
    test_greedy();
    test_top_k();
    test_top_k_ties();
    test_temperature();
    test_top_p();
    test_sampling();
    test_repetition_penalty();
    test_nan_reject();
    test_no_ooo_write();
    printf("\n%s: %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail;
}
