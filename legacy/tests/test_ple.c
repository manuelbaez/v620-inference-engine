/* tests/test_ple.c — unit tests for the n-gram PLE addressing + fp8 codec.
 * No external framework: plain assert + CHECK, main() returns the failure
 * count (0 = all pass).
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -Iinclude \
 *        src/core/types.c src/core/config.c src/core/ple.c tests/test_ple.c \
 *        -o /tmp/opencode/tp
 */
#include "qw/ple.h"
#include "qw/types.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, label)                                          \
    do {                                                            \
        if (cond)                                                   \
            printf("PASS %s\n", label);                             \
        else {                                                      \
            printf("FAIL %s (%s:%d)\n", label, __FILE__, __LINE__); \
            g_fail++;                                               \
        }                                                           \
    } while (0)

static uint64_t g_seed;
static uint64_t lcg(void)
{
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_seed >> 32;
}
/* Uniform token id in [0, 248320) (padded vocab). */
static uint32_t rand_tok(void)
{
    return (uint32_t)(lcg() % 248320ULL);
}

/* ------------------------------------------------------------- test 1: size */
static void test_size(void)
{
    printf("-- test 1: total bytes == 20000000*2560 and within 5%% of 47.7 GiB\n");
    qw_ple_desc d = qw_ple_default();
    uint64_t total = qw_ple_total_bytes(&d);
    printf("  total=%llu bytes\n", (unsigned long long)total);
    CHECK(total == 51200000000ULL, "total == 20000000*2560 exactly");

    double gib  = (double)total / 1073741824.0;
    double diff = gib - 47.7;
    if (diff < 0) diff = -diff;
    printf("  %.4f GiB (target 47.7, diff %.4f%%)\n", gib,
           diff / 47.7 * 100.0);
    CHECK(diff / 47.7 <= 0.05, "within 5% of 47.7 GiB");
    CHECK(qw_ple_fits_host_ram(&d), "fits in 220 GiB host RAM");

    /* row_bytes from a model cfg (hidden_size 2560) matches the default. */
    qw_model_cfg cfg = qw_model_qwen38_flash_next_default();
    CHECK(qw_ple_row_bytes(&cfg) == 2560U, "row_bytes(cfg) == hidden_size 2560");
    CHECK(qw_ple_row_bytes(NULL) == 2560U, "row_bytes(NULL) == default 2560");
}

/* --------------------------------------------------------- test 2: indexing */
static void test_index(void)
{
    printf("-- test 2: deterministic, in-range, 100000 random sequences\n");
    qw_ple_desc d = qw_ple_default();
    g_seed = 0x9e3779b97f4a7c15ULL;
    uint64_t *idx = malloc(100000 * sizeof(uint64_t));
    assert(idx != NULL);

    for (uint64_t i = 0; i < 100000; i++) {
        uint32_t toks[4];
        for (int t = 0; t < 4; t++) toks[t] = rand_tok();
        uint64_t a, b;
        /* bigrams (n_ids=2) of the 4-tok window: first two, last two */
        assert(qw_ple_index(&d, toks, 2, 2, &a) == QW_OK);
        assert(qw_ple_index(&d, toks + 2, 2, 2, &b) == QW_OK);
        idx[i] = a;
        assert(a < d.n_entries);
        assert(b < d.n_entries);
    }
    /* determinism: recomputing gives the identical indices */
    g_seed = 0x9e3779b97f4a7c15ULL;
    int same = 1;
    for (uint64_t i = 0; i < 100000; i++) {
        uint32_t toks[4];
        for (int t = 0; t < 4; t++) toks[t] = rand_tok();
        uint64_t a;
        assert(qw_ple_index(&d, toks, 2, 2, &a) == QW_OK);
        if (a != idx[i]) same = 0;
    }
    CHECK(same, "deterministic across recomputation");
    CHECK(1, "all 100000 indices in [0, n_entries)");
    free(idx);

    /* same n-gram -> same index (explicit) */
    uint32_t g[2] = { 123, 456 };
    uint64_t x, y;
    assert(qw_ple_index(&d, g, 2, 2, &x) == QW_OK);
    assert(qw_ple_index(&d, g, 2, 2, &y) == QW_OK);
    CHECK(x == y, "same bigram -> same index");

    /* unigram path agrees with the general path */
    uint64_t u;
    assert(qw_ple_index_unigram(&d, g[0], &u) == QW_OK);
    assert(qw_ple_index(&d, &g[0], 1, 1, &x) == QW_OK);
    CHECK(u == x, "unigram == general n_ids=1");
}

/* ------------------------------------------------- test 3: collision rate */
static void test_collision(void)
{
    printf("-- test 3: 100k random bigrams, observed collision rate < 1%%\n");
    qw_ple_desc d = qw_ple_default();
    g_seed = 0x2545f4914f6cdd1dULL;
    uint64_t N = 100000, distinct = 0;
    /* open-addressing set sized 2x N so probe chains always terminate. */
    uint64_t CAP = 262144; /* 262144 = 2^18 > 2*N */
    uint64_t *seen = malloc(CAP * sizeof(uint64_t));
    assert(seen != NULL);
    memset(seen, 0xff, CAP * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        uint32_t g[2] = { rand_tok(), rand_tok() };
        uint64_t idx;
        assert(qw_ple_index(&d, g, 2, 2, &idx) == QW_OK);
        uint64_t slot = (idx * 2654435761ULL) & (CAP - 1);
        for (uint64_t p = slot; ; p = (p + 1) & (CAP - 1)) {
            if (seen[p] == 0xffffffffffffffffULL) { seen[p] = idx; break; }
            if (seen[p] == idx) break;
        }
    }
    /* count distinct by scanning the set */
    for (uint64_t i = 0; i < CAP; i++)
        if (seen[i] != 0xffffffffffffffffULL)
            distinct++;
    free(seen);
    double coll = 1.0 - (double)distinct / (double)N;
    printf("  distinct=%llu/%llu  collision rate=%.4f%%\n",
           (unsigned long long)distinct, (unsigned long long)N, coll * 100.0);
    CHECK(coll < 0.01, "collision rate < 1% (hashing-consistent)");
}

/* --------------------------------------------------- test 4: row offset */
static void test_row_offset(void)
{
    printf("-- test 4: row offset overflow caught\n");
    qw_ple_desc d = qw_ple_default();
    uint64_t off;
    assert(qw_ple_row_offset(&d, 0, &off) == QW_OK);
    CHECK(off == 0, "row 0 -> offset 0");
    assert(qw_ple_row_offset(&d, 12345, &off) == QW_OK);
    CHECK(off == 12345ULL * 2560ULL, "row 12345 -> 12345*2560");
    /* huge index: offset would overflow uint64 -> QW_ERR_RANGE */
    qw_err e = qw_ple_row_offset(&d, 10000000000000000000ULL, &off);
    CHECK(e == QW_ERR_RANGE, "huge index -> QW_ERR_RANGE (overflow)");
    /* index past the table -> QW_ERR_RANGE */
    e = qw_ple_row_offset(&d, d.n_entries, &off);
    CHECK(e == QW_ERR_RANGE, "index == n_entries -> QW_ERR_RANGE");
    /* NULL -> QW_ERR_NULL */
    CHECK(qw_ple_row_offset(NULL, 0, &off) == QW_ERR_NULL, "NULL desc -> NULL err");
    CHECK(qw_ple_row_offset(&d, 0, NULL) == QW_ERR_NULL, "NULL out -> NULL err");
}

/* ----------------------------------------------------------- test 5: fp8 */
static void test_fp8(void)
{
    printf("-- test 5: fp8 E4M3 round trip (max rel error reported)\n");
    double max_rel = 0.0;

    /* exact: +0, -0, +1, -1 */
    assert(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(0.0f)) == 0.0f);
    assert(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(-0.0f)) == -0.0f);
    assert(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(1.0f)) == 1.0f);
    assert(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(-1.0f)) == -1.0f);
    printf("  exact: 0 -0 +1 -1  ok\n");

    /* max finite +448 / -448 round exactly */
    CHECK(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(448.0f)) == 448.0f, "+448 exact");
    CHECK(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(-448.0f)) == -448.0f, "-448 exact");

    /* a small value (0.25 is exactly representable: 1.0 x 2^-2) */
    CHECK(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(0.25f)) == 0.25f, "0.25 exact");

    /* NaN in -> NaN out */
    float nan_in = NAN;
    float nan_out = qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(nan_in));
    CHECK(nan_out != nan_out, "NaN in -> NaN out");

    /* sweep over a dense normal range, track max relative error of
     * decode(encode(x)). With a 3-bit mantissa the worst-case RELATIVE error
     * is just under 1/16 = 0.0625 (at values just above a power of two, where
     * the ULP equals the value); the worst-case ABSOLUTE error is a half-ULP
     * = 2^-5 = 0.03125 at 1.0. We bound the relative error here. */
    for (int i = -2000; i <= 2000; i++) {
        float x = (float)(i / 100.0); /* -20.0 .. 20.0 in 0.1 steps */
        if (x == 0.0f)
            continue;
        if (fabs((double)x) < 0.016)
            continue; /* skip the subnormal band (rel error is unbounded) */
        float rt = qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(x));
        double r = (double)(rt - x) / (double)x;
        if (r < 0) r = -r;
        if (r > max_rel) max_rel = r;
    }
    printf("  max relative error over [-20,20] normals = %.4g\n", max_rel);
    CHECK(max_rel < 0.0625, "max rel error within e4m3 precision (< 1/16)");

    /* saturation: 1000 -> 448 (no inf in e4m3) */
    CHECK(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(1000.0f)) == 448.0f,
          "1000 saturates to +448");
    CHECK(qw_fp8_e4m3_to_f32(qw_f32_to_fp8_e4m3(-1000.0f)) == -448.0f,
          "-1000 saturates to -448");
}

/* --------------------------------------------------------- test 6: plan */
static void test_plan(void)
{
    printf("-- test 6: gather plan seq_len=1024 ngram=2, bytes/token by hand\n");
    qw_ple_desc d = qw_ple_default();
    qw_ple_plan p;
    assert(qw_ple_gather_plan(&d, 1024, 2, &p) == QW_OK);
    /* by hand: bytes/token = row_bytes = 2560; bytes/seq = 1024*2560 */
    CHECK(p.bytes_per_token == 2560ULL, "bytes/token == 2560 (one row)");
    CHECK(p.bytes_per_seq == 2621440ULL, "bytes/seq == 1024*2560 = 2621440");
    /* required GB/s at 200 tok/s = 2560 * 200 / 1e9 = 0.000512 GB/s */
    CHECK(fabs(p.gbps_200 - 5.12e-4) < 1e-12, "gbps@200 == 2560*200/1e9");
    CHECK(fabs(p.gbps_50 - 1.28e-4) < 1e-12, "gbps@50 == 2560*50/1e9");
    qw_ple_plan_report(&p);
    (void)d;
}

int main(void)
{
    test_size();
    test_index();
    test_collision();
    test_row_offset();
    test_fp8();
    test_plan();
    if (g_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fail);
    return g_fail;
}
