/* tests/test_quant.c — no-framework asserts for the int8 quant layer.
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -Iinclude \
 *        tests/test_quant.c src/core/quant.c src/core/types.c
 *
 * main() returns the count of failures (0 = all pass).
 */
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qw/quant.h"
#include "qw/types.h"

static int g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { printf("  ok   %s\n", #cond); } \
    else { printf("  FAIL %s (line %d)\n", #cond, __LINE__); g_fail++; } \
} while (0)

/* |x| without libm (fabs/fabsf): clear the sign via a conditional. */
static float tfabsf(float x) { return (x < 0.0f) ? -x : x; }
static double tfabs(double x) { return (x < 0.0) ? -x : x; }

/* Deterministic LCG for test data (documented fixed seeds per test). */
static uint64_t g_seed;
static uint64_t lcg(void)
{
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_seed >> 32;
}

/* Standard-normal via Box-Muller (portable, no libm) for test data. */
static double t_sqrt(double x) {
    if (x <= 0.0) return 0.0;
    double g = x;
    for (int i = 0; i < 8; i++) g = 0.5 * (g + x / g);
    return g;
}
static double t_sin(double x) {
    while (x >  3.14159265358979323846) x -= 6.28318530717958647692;
    while (x < -3.14159265358979323846) x += 6.28318530717958647692;
    double x2 = x * x;
    return x * (1.0 + x2 * (-1.0/6.0 + x2 * (1.0/120.0 - x2 / 5040.0)));
}
static double t_log(double x) {
    if (x <= 0.0) return -1e30;
    int e = 0;
    while (x >= 2.0) { x *= 0.5; e++; }
    while (x < 1.0)  { x *= 2.0; e--; }
    double z = (x - 1.0) / (x + 1.0);
    double z2 = z * z;
    double s = z, term = z;
    for (int i = 1; i < 12; i++) { term *= z2; s += term / (2 * i + 1); }
    return 2.0 * s + (double)e * 0.69314718055994530942;
}
static float gauss_f32(void) {
    double u1 = ((double)(lcg() & 0xffffff) + 1.0) / (1.0 + 16777216.0);
    double u2 = ((double)(lcg() & 0xffffff) + 1.0) / (1.0 + 16777216.0);
    double r  = t_sqrt(-2.0 * t_log(u1));
    double th = 2.0 * 3.14159265358979323846 * u2;
    double g  = r * t_sin(th);
    if (g > 8.0)  g = 8.0;
    if (g < -8.0) g = -8.0;
    return (float)g;
}

/* Uniform in [-1,1). */
static float rand_f32(void)
{
    return (float)((double)(lcg() & 0xfffff) / 1048576.0 * 2.0 - 1.0);
}

/* ------------------------------------------- test 1: pack/unpack roundtrip */
static void test_pack_roundtrip(void)
{
    printf("test 1: qw_q8_pack / qw_q8_dequant_row round trip on known pattern\n");
    int N = 4, K = 8;
    float src[4 * 8] = {
        /* row 0: full-scale pattern, scale = 127/127 = 1. Values are exact
         * multiples of the quantum 1/127: -127, 0, 127, 1, -1, 64, -64, 126. */
        -127.0f, 0.0f, 127.0f, 1.0f, -1.0f, 64.0f, -64.0f, 126.0f,
        /* row 1: all zeros (guard path) */
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        /* row 2: single max-abs outlier forces the scale */
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f,
        /* row 3: tiny values, sub-quantum */
        0.001f, -0.001f, 0.002f, 0.0f, -0.002f, 0.001f, 0.0f, -0.001f,
    };

    qw_q8_block blk;
    qw_err e = qw_q8_pack(src, N, K, &blk);
    CHECK(e == QW_OK);
    CHECK(blk.n == N && blk.k == K);
    CHECK(blk.w_q != NULL && blk.scale != NULL && blk.zp != NULL);

    /* All zero-points are 0 (symmetric). */
    int zp_ok = 1;
    for (int i = 0; i < N; i++)
        if (blk.zp[i] != 0) zp_ok = 0;
    CHECK(zp_ok);

    /* Row 0: scale = 127/127 = 1. Each value dequants to a multiple of 1/127
     * (quantum = 1/127 ~ 0.00787), so worst-case round-trip error is half a
     * quantum ~ 0.00394. 63.5 = 127/2 is exactly representable; 1.0, -1.0,
     * 0, +/-127 are too. */
    float dq[8];
    e = qw_q8_dequant_row(&blk, 0, dq);
    CHECK(e == QW_OK);
    float max_err = 0.0f;
    for (int j = 0; j < K; j++) {
        float d = tfabsf(dq[j] - src[0 * K + j]);
        if (d > max_err) max_err = d;
    }
    printf("  row0 max |dequant - orig| = %g (half-quantum bound 0.00394)\n",
           max_err);
    CHECK(max_err < 0.004f);

    /* Row 1 (all zeros): scale 1.0, dequant exactly 0. */
    CHECK(blk.scale[1] == 1.0f);
    e = qw_q8_dequant_row(&blk, 1, dq);
    CHECK(e == QW_OK);
    int all_zero = 1;
    for (int j = 0; j < K; j++)
        if (dq[j] != 0.0f) all_zero = 0;
    CHECK(all_zero);

    /* Row 2: scale must be 100/127; the outlier dequants to ~100. */
    CHECK(tfabsf(blk.scale[2] - (100.0f / 127.0f)) < 1e-6f);
    e = qw_q8_dequant_row(&blk, 2, dq);
    CHECK(e == QW_OK);
    CHECK(tfabsf(dq[7] - 100.0f) < 1e-3f);

    /* qw_q8_block_nbytes = N*K + N*4 + N*4. */
    CHECK(qw_q8_block_nbytes(&blk) == (size_t)(N * K + N * 4 + N * 4));

    qw_q8_block_free(&blk);
}

/* ------------------------------------- test 2: per-row activation scaling */
static void test_act_q8_row(void)
{
    printf("test 2: qw_act_q8_row — all-zero row and one-huge-outlier row\n");
    int K = 16;

    /* All-zero row: guard path, scale := 1.0, all q := 0. */
    float zrow[16];
    memset(zrow, 0, sizeof(zrow));
    int8_t q[16];
    float scale = -1.0f;
    qw_err e = qw_act_q8_row(zrow, K, q, &scale);
    CHECK(e == QW_OK);
    CHECK(scale == 1.0f);
    int all_zero = 1;
    for (int j = 0; j < K; j++)
        if (q[j] != 0) all_zero = 0;
    CHECK(all_zero);

    /* One huge outlier: the max-abs element must map to exactly +127 and
     * set the scale = amax/127. */
    float orow[16];
    memset(orow, 0, sizeof(orow));
    orow[3] = 999.0f;    /* the outlier */
    orow[9] = 33.0f;
    e = qw_act_q8_row(orow, K, q, &scale);
    CHECK(e == QW_OK);
    CHECK(tfabsf(scale - (999.0f / 127.0f)) < 1e-6f);
    CHECK(q[3] == 127);                 /* max-abs -> +127 */
    /* 33 / (999/127) = 4.194... -> rounds to 4 (half-away-from-zero). */
    CHECK(q[9] == 4);
    /* No value may fall outside [-127,127]. Checked in int: q[j] > 127 is
     * always false for int8 (type-limits), so sign-extend before comparing. */
    int in_range = 1;
    for (int j = 0; j < K; j++) {
        int v = (int)q[j];
        if (v > 127 || v < -127) in_range = 0;
    }
    CHECK(in_range);
    printf("  outlier scale=%g q[3]=%d q[9]=%d\n",
           (double)scale, (int)q[3], (int)q[9]);
}

/* ------------------------------- test 3: naive vs blocked bit-identical acc */
static void test_naive_vs_blocked(void)
{
    printf("test 3: naive vs blocked — bit-identical accumulators (seed 0xC0FFEE0123)\n");
    int M = 20, N = 12, K = 2560; /* K=2560: the real hidden dim */
    assert(qw_q8_overflow_ok(M, N, K));

    size_t a_n = (size_t)M * K, w_n = (size_t)N * K, o_n = (size_t)M * N;
    int8_t  *a_q8 = (int8_t *)malloc(a_n);
    int8_t  *w_q8 = (int8_t *)malloc(w_n);
    float   *a_sc = (float *)malloc((size_t)M * sizeof(float));
    float   *w_sc = (float *)malloc((size_t)N * sizeof(float));
    int32_t *out_naive   = (int32_t *)malloc(o_n * sizeof(int32_t));
    int32_t *out_blocked = (int32_t *)malloc(o_n * sizeof(int32_t));

    if (!a_q8 || !w_q8 || !a_sc || !w_sc || !out_naive || !out_blocked) {
        printf("  FAIL alloc\n");
        g_fail++;
        free(a_q8); free(w_q8); free(a_sc); free(w_sc);
        free(out_naive); free(out_blocked);
        return;
    }

    /* Fixed seed => reproducible random data. Quantize from fp32 so the
     * scales are realistic (not degenerate). */
    g_seed = 0xC0FFEE0123ULL;
    for (int i = 0; i < M; i++) {
        float tmp[2560];
        for (int t = 0; t < K; t++) tmp[t] = rand_f32();
        qw_err e = qw_act_q8_row(tmp, K, a_q8 + (size_t)i * K, &a_sc[i]);
        assert(e == QW_OK);
    }
    {
        float *wsrc = (float *)malloc(w_n * sizeof(float));
        for (size_t i = 0; i < w_n; i++) wsrc[i] = rand_f32();
        qw_q8_block blk;
        qw_err e = qw_q8_pack(wsrc, N, K, &blk);
        assert(e == QW_OK);
        memcpy(w_q8, blk.w_q, w_n);
        memcpy(w_sc, blk.scale, (size_t)N * sizeof(float));
        qw_q8_block_free(&blk);
        free(wsrc);
    }

    /* Compare the raw int32 accumulators: this is the meaningful equality.
     * (The float dequant of the same acc can differ in the last ULP across
     * loop schedules due to FP contraction, so we prove the int32 side.) */
    qw_err en = qw_q8_gemm_int32_naive(a_q8, w_q8, M, N, K, out_naive);
    qw_err eb = qw_q8_gemm_int32_blocked(a_q8, w_q8, M, N, K, out_blocked);
    CHECK(en == QW_OK);
    CHECK(eb == QW_OK);

    int identical = 1;
    for (size_t i = 0; i < o_n; i++)
        if (out_naive[i] != out_blocked[i]) {
            identical = 0;
            printf("  first diff at %zu: naive=%d blocked=%d\n", i,
                   out_naive[i], out_blocked[i]);
            break;
        }
    CHECK(identical);

    /* The float dequant of the same int32 acc can differ across kernels by
     * up to ~1e-4 relative: the blocked kernel accumulates its dequant
     * output with += once per K-strip (k/BK strips, each a float round),
     * while the naive kernel does a single multiply — so the blocked result
     * carries (k/BK) rounding steps. (float)acc is also a double rounding
     * for |acc| > 2^24, and FMA contraction is a compiler choice. 1e-4
     * relative is far below any accuracy concern; the int32 accumulator
     * equality above is the meaningful check. */
    {
        float *f_naive   = (float *)malloc(o_n * sizeof(float));
        float *f_blocked = (float *)malloc(o_n * sizeof(float));
        if (f_naive && f_blocked) {
            qw_q8_gemm_naive(a_q8, a_sc, w_q8, w_sc, M, N, K, f_naive);
            qw_q8_gemm_blocked(a_q8, a_sc, w_q8, w_sc, M, N, K, f_blocked);
            double max_rel = 0.0;
            int rel_ok = 1;
            for (size_t i = 0; i < o_n; i++) {
                double r = f_naive[i], b = f_blocked[i];
                double denom = tfabs(r);
                if (denom < 1e-3) denom = 1.0;
                double rel = tfabs(b - r) / denom;
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-4) {
                    rel_ok = 0;
                    printf("  rel diff >1e-4 at %zu: %g vs %g\n", i,
                           (double)f_naive[i], (double)f_blocked[i]);
                    break;
                }
            }
            printf("  float dequant max rel diff (int32-identical accs): %.2e\n",
                   max_rel);
            CHECK(rel_ok);
        }
        free(f_naive); free(f_blocked);
    }

    free(a_q8); free(w_q8); free(a_sc); free(w_sc);
    free(out_naive); free(out_blocked);
}

/* --------------------------------------- test 4: dequant epilogue vs ref */
static void test_dequant_vs_ref(void)
{
    printf("test 4: blocked dequant epilogue matches int64 ref within 1e-3 rel\n");
    int M = 16, N = 8, K = 640; /* K=640: the real moe_intermediate dim */
    assert(qw_q8_overflow_ok(M, N, K));

    size_t a_n = (size_t)M * K, w_n = (size_t)N * K, o_n = (size_t)M * N;
    int8_t  *a_q8 = (int8_t *)malloc(a_n);
    int8_t  *w_q8 = (int8_t *)malloc(w_n);
    float   *a_sc = (float *)malloc((size_t)M * sizeof(float));
    float   *w_sc = (float *)malloc((size_t)N * sizeof(float));
    float   *out_ref   = (float *)malloc(o_n * sizeof(float));
    float   *out_block = (float *)malloc(o_n * sizeof(float));

    if (!a_q8 || !w_q8 || !a_sc || !w_sc || !out_ref || !out_block) {
        printf("  FAIL alloc\n");
        g_fail++;
        free(a_q8); free(w_q8); free(a_sc); free(w_sc);
        free(out_ref); free(out_block);
        return;
    }

    g_seed = 0xBADC0DE42ULL;
    for (int i = 0; i < M; i++) {
        float tmp[640];
        for (int t = 0; t < K; t++) tmp[t] = rand_f32();
        qw_err e = qw_act_q8_row(tmp, K, a_q8 + (size_t)i * K, &a_sc[i]);
        assert(e == QW_OK);
    }
    {
        float *wsrc = (float *)malloc(w_n * sizeof(float));
        for (size_t i = 0; i < w_n; i++) wsrc[i] = rand_f32();
        qw_q8_block blk;
        qw_err e = qw_q8_pack(wsrc, N, K, &blk);
        assert(e == QW_OK);
        memcpy(w_q8, blk.w_q, w_n);
        memcpy(w_sc, blk.scale, (size_t)N * sizeof(float));
        qw_q8_block_free(&blk);
        free(wsrc);
    }

    qw_err er = qw_q8_gemm_ref(a_q8, a_sc, w_q8, w_sc, M, N, K, out_ref);
    qw_err eb = qw_q8_gemm_blocked(a_q8, a_sc, w_q8, w_sc, M, N, K, out_block);
    CHECK(er == QW_OK);
    CHECK(eb == QW_OK);

    /* Both compute the same int32 acc (ref via int64 then narrow, blocked
     * via int32); the dequant is the same per-element float ops, so results
     * must agree. Check relative error within 1e-3. */
    int ok = 1;
    for (size_t i = 0; i < o_n; i++) {
        float r = out_ref[i], b = out_block[i];
        double denom = tfabs((double)r);
        if (denom < 1e-3) denom = 1.0; /* guard against ~0 outputs */
        double rel = tfabs((double)b - (double)r) / denom;
        if (rel > 1e-3) {
            ok = 0;
            printf("  rel err %g at %zu (ref=%g block=%g)\n", rel, i,
                   (double)r, (double)b);
            break;
        }
    }
    CHECK(ok);

    free(a_q8); free(w_q8); free(a_sc); free(w_sc);
    free(out_ref); free(out_block);
}

/* --------------------------------------------- test 5: overflow check */
static void test_overflow(void)
{
    printf("test 5: qw_q8_overflow_ok accepts K=2560, rejects overflow K\n");
    /* 127*127*K <= INT32_MAX  =>  K <= INT32_MAX/16129 = 133144. */
    CHECK(qw_q8_overflow_ok(1, 1, 2560) == true);    /* hidden dim: safe */
    CHECK(qw_q8_overflow_ok(1, 1, 640) == true);     /* moe_intermediate: safe */
    CHECK(qw_q8_overflow_ok(1, 1, 133144) == true);  /* exactly at the bound */
    CHECK(qw_q8_overflow_ok(1, 1, 133145) == false); /* one past the bound */
    CHECK(qw_q8_overflow_ok(1, 1, 1000000) == false);/* far past: rejects */
    CHECK(qw_q8_overflow_ok(0, 1, 2560) == false);   /* M<=0: invalid */
}

/* --------------------------------------- test 6: gaussian quant error */
static void test_quant_error(void)
{
    printf("test 6: int8 quant error on gaussian data (MSE bound + cos>0.999)\n");
    int M = 64, N = 32, K = 2560;
    assert(qw_q8_overflow_ok(M, N, K));

    size_t a_n = (size_t)M * K, w_n = (size_t)N * K, o_n = (size_t)M * N;
    float *a         = (float *)malloc(a_n * sizeof(float));
    float *w         = (float *)malloc(w_n * sizeof(float));
    float *out_ref   = (float *)malloc(o_n * sizeof(float));
    float *out_q8    = (float *)malloc(o_n * sizeof(float));
    int8_t *a_q8     = (int8_t *)malloc(a_n);
    int8_t *w_q8     = (int8_t *)malloc(w_n);
    float *a_sc      = (float *)malloc((size_t)M * sizeof(float));
    float *w_sc      = (float *)malloc((size_t)N * sizeof(float));

    if (!a || !w || !out_ref || !out_q8 || !a_q8 || !w_q8 || !a_sc || !w_sc) {
        printf("  FAIL alloc\n");
        g_fail++;
        free(a); free(w); free(out_ref); free(out_q8);
        free(a_q8); free(w_q8); free(a_sc); free(w_sc);
        return;
    }

    /* True standard-normal data via Box-Muller (portable, no libm): the
     * spec calls for a gaussian, and int8 quantizes a std-1 gaussian cleanly
     * (a wide non-gaussian does not, which would inflate MSE falsely). */
    g_seed = 0x15A11A5EULL;
    for (size_t i = 0; i < a_n; i++)
        a[i] = gauss_f32();
    for (size_t i = 0; i < w_n; i++)
        w[i] = gauss_f32();

    /* Quantize. */
    for (int i = 0; i < M; i++) {
        qw_err e = qw_act_q8_row(a + (size_t)i * K, K,
                                 a_q8 + (size_t)i * K, &a_sc[i]);
        assert(e == QW_OK);
    }
    {
        qw_q8_block blk;
        qw_err e = qw_q8_pack(w, N, K, &blk);
        assert(e == QW_OK);
        memcpy(w_q8, blk.w_q, w_n);
        memcpy(w_sc, blk.scale, (size_t)N * sizeof(float));
        qw_q8_block_free(&blk);
    }

    /* fp32-accumulate reference vs the int8 blocked kernel. */
    qw_err er = qw_q8_gemm_fp32(a, w, M, N, K, out_ref);
    qw_err eb = qw_q8_gemm_blocked(a_q8, a_sc, w_q8, w_sc, M, N, K, out_q8);
    CHECK(er == QW_OK);
    CHECK(eb == QW_OK);

    double mse = qw_q8_err_mse(out_q8, out_ref, (int)o_n);
    double cos = qw_q8_cosine_sim(out_q8, out_ref, (int)o_n);
    printf("  MSE=%.6e  cosine=%.9f\n", mse, cos);
    /* Per-element int8 quant error ~ 1/127/2 (relative); over a K=2560 dot
     * the independent errors average down (sqrt(K) cancellation), so cosine
     * sim is very high. Stated bounds: cosine > 0.999 and MSE < 0.5. */
    CHECK(cos > 0.999);
    CHECK(mse < 0.5);

    free(a); free(w); free(out_ref); free(out_q8);
    free(a_q8); free(w_q8); free(a_sc); free(w_sc);
}

/* --------------------------------------- test 7: compare_fp16 report */
static void test_compare_fp16(void)
{
    printf("test 7: qw_q8_compare_fp16 prints accuracy report\n");
    double mse = -1.0, cos = -1.0;
    qw_err e = qw_q8_compare_fp16(32, 16, 640, 0x12345678ULL, &mse, &cos);
    CHECK(e == QW_OK);
    CHECK(mse >= 0.0);
    CHECK(cos > 0.99);
    printf("  (report printed above; mse=%.6e cos=%.9f)\n", mse, cos);
}

int main(void)
{
    test_pack_roundtrip();
    test_act_q8_row();
    test_naive_vs_blocked();
    test_dequant_vs_ref();
    test_overflow();
    test_quant_error();
    test_compare_fp16();
    if (g_fail == 0) {
        printf("ALL PASS");
        return 0;
    }
    printf("%d FAILURES\n", g_fail);
    return g_fail;
}
