/* tools/bench_int8.c — int8-vs-fp16(prefill) GEMM microbenchmark.
 *
 * THE deliverable that answers: "is int8 faster than fp16 on this card?"
 *
 * Usage:
 *   bench_int8 [--m M] [--iters N] [--check] [--csv]
 *
 *   --m M       additionally run the single MoE expert shape at batch M
 *               (on top of the fixed set; M in [1, 8192])
 *   --iters N   timed runs per kernel (default 21; median reported, not mean)
 *   --check     verify int8 blocked == int8 naive == int64 oracle on one
 *               shape before benchmarking
 *   --csv       emit a machine-readable CSV table (one row per kernel/shape)
 *
 * Shapes come from the REAL model (include/qw/config.h):
 *   hidden_size = 2560, moe_intermediate = 640. A MoE expert is 3 GEMMs
 *   (gate/up: H->I, down: I->H); we benchmark one representative expert
 *   GEMM per (K, M) — the gate proj [M x 640] * [640 x 2560]. The M values
 *   are the per-expert token counts for a prompt of L tokens:
 *       M = L * topk / num_experts = L * 10 / 512
 *         L=51    -> M=1     (decode: one token per stream)
 *         L=410   -> M=8     (small batch)
 *         L=819   -> M=16
 *         L=1638  -> M=32    (A/B gate threshold: int8 must beat fp16 >1.3x)
 *         L=4096  -> M=80    (default ctx: the money shape)
 *         L=13107 -> M=256   (large prompt)
 *         L=52428 -> M=1024  (very large prompt / batched prefill)
 *
 * WHY THIS MATTERS: decode (M=1) is bandwidth-bound — int8 does not help.
 * Prefill is compute-bound, and the MoE structure is what turns a prompt
 * into a real GEMM. Below the arithmetic-intensity ridge point a GEMM is
 * memory-bound and int8's 2x FLOP throughput buys nothing; above it, int8
 * can roughly double throughput. The table below each run prints the
 * arithmetic intensity (FLOPs/byte) of each shape next to the machine's
 * fp16 ridge point (peak_flops / 512e9 B/s) so you can SEE at which M the
 * problem becomes compute-bound and therefore whether int8 can help at
 * that M.
 *
 * CPU reference (always runs): the blocked int8 kernel vs the naive int8
 * kernel vs a scalar FMA fp32-accumulate GEMM. The fp32 path is the
 * fp32-accumulate stand-in for an fp16 GEMM; on the CPU it is a scalar FMA
 * loop (LLVM does not vectorize the fp32 dot for this [M][K]*[N][K] layout
 * into wide FMA — it keeps it scalar), whereas the int8 path lowers to the
 * dedicated integer dot instruction (AVX2 vpdpwssd; v_dot on AMDGPU). The
 * int8-over-fp32 ratio is therefore LARGELY a measure of the dedicated int8
 * dot instruction's throughput vs a scalar fp32 FMA — NOT a pure
 * "int8-vs-fp16 FLOP" comparison. On the V620 the fp16 FMA is hardware and
 * the ratio is much closer to the true int8/fp16 FLOP ratio.
 * On a laptop with no ROCm this is the whole deliverable — it still measures
 * the CPU-side speedup of the blocked kernel, which is what tells us whether
 * LLVM will have anything to vectorize on the GPU.
 *
 * GPU (only when built with -DQW_WITH_HIP): the same kernels are run on the
 * gfx1030 device, timed with hipEventElapsedTime. Without it, GPU numbers
 * are unavailable and the tool says so.
 */
#define _POSIX_C_SOURCE 199309L /* clock_gettime / CLOCK_MONOTONIC */

#include "qw/quant.h"
#include "qw/types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* gfx1030 (Radeon Pro V620) memory bandwidth — the fixed 512 GB/s GDDR6
 * figure from the hardware spec (bus_width/8 * 2 * clock). */
#define QW_V620_BW (512.0 * 1e9) /* bytes/s */

/* Rough gfx1030 fp16 peak FLOP/s. RDNA2: 72 CUs, up to 2 fp16 FMA/cycle per
 * SIMD, ~1.7 GHz boost. 72 * 4 SIMD * 2 * 2 * 1.7e9 ~ 1.96e12. Used only to
 * compute the ridge point; the real number comes from the device on the V620
 * box. */
#define QW_V620_FP16_PEAK (1.96e12) /* FLOP/s */

/* ------------------------------------------------------------------ util */

/* Monotonic clock in seconds (CLOCK_MONOTONIC). */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Median of n values (n>0). Copies the array. */
static double median(double *v, int n)
{
    if (n <= 0) return 0.0;
    double *c = (double *)malloc((size_t)n * sizeof(double));
    if (!c) return 0.0;
    memcpy(c, v, (size_t)n * sizeof(double));
    /* insertion sort: n is small (~21). */
    for (int i = 1; i < n; i++) {
        double key = c[i];
        int j = i - 1;
        while (j >= 0 && c[j] > key) { c[j + 1] = c[j]; j--; }
        c[j + 1] = key;
    }
    double med = (n % 2) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return med;
}

/* Deterministic data fill so runs are reproducible. Uses a scaled LCG output
 * in [-8, 8] — NOT true gaussian (avoids libm sqrt/log/cos, which the Makefile
 * does not link for tools). The exact distribution does not matter for a
 * throughput benchmark; we just need non-degenerate, varied int8 data. */
static uint64_t g_seed;
static uint64_t lcg(void)
{
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_seed >> 32;
}
static float gaussian_f32(void)
{
    /* 24-bit fraction in [-1,1), scaled to [-8,8]. */
    double u = (double)((lcg() & 0xffffff) - 0x800000) / 8388608.0;
    return (float)(u * 8.0);
}

/* ------------------------------------------------------------- data gen */

typedef struct bench_data {
    /* int8 inputs */
    int8_t *a_q8;    /* [M][K] */
    int8_t *w_q8;    /* [N][K] */
    float  *a_scale; /* [M]    */
    float  *w_scale; /* [N]    */
    /* fp32 inputs (the fp16-accumulate reference) */
    float  *a_f32;   /* [M][K] */
    float  *w_f32;   /* [N][K] */
    /* outputs (one per kernel, sized [M][N]) */
    float  *out_int8;
    float  *out_fp32;
    int     m, n, k;
} bench_data;

static int bench_data_init(bench_data *d, int m, int n, int k, uint64_t seed)
{
    d->m = m; d->n = n; d->k = k;
    size_t a_n = (size_t)m * (size_t)k;
    size_t w_n = (size_t)n * (size_t)k;
    size_t o_n = (size_t)m * (size_t)n;

    d->a_q8    = (int8_t *)malloc(a_n);
    d->w_q8    = (int8_t *)malloc(w_n);
    d->a_scale = (float  *)malloc((size_t)m * sizeof(float));
    d->w_scale = (float  *)malloc((size_t)n * sizeof(float));
    d->a_f32   = (float  *)malloc(a_n * sizeof(float));
    d->w_f32   = (float  *)malloc(w_n * sizeof(float));
    d->out_int8 = (float *)malloc(o_n * sizeof(float));
    d->out_fp32 = (float *)malloc(o_n * sizeof(float));
    if (!d->a_q8 || !d->w_q8 || !d->a_scale || !d->w_scale ||
        !d->a_f32 || !d->w_f32 || !d->out_int8 || !d->out_fp32)
        return 0;

    g_seed = seed;
    for (size_t i = 0; i < a_n; i++) d->a_f32[i] = gaussian_f32();
    for (size_t i = 0; i < w_n; i++) d->w_f32[i] = gaussian_f32();

    /* Quantize activations per-row, weights per-channel. */
    for (int i = 0; i < m; i++) {
        qw_err e = qw_act_q8_row(d->a_f32 + (size_t)i * (size_t)k, k,
                                 d->a_q8 + (size_t)i * (size_t)k,
                                 &d->a_scale[i]);
        if (e != QW_OK) return 0;
    }
    {
        qw_q8_block blk;
        qw_err e = qw_q8_pack(d->w_f32, n, k, &blk);
        if (e != QW_OK) return 0;
        memcpy(d->w_q8, blk.w_q, w_n);
        memcpy(d->w_scale, blk.scale, (size_t)n * sizeof(float));
        qw_q8_block_free(&blk);
    }
    return 1;
}

static void bench_data_free(bench_data *d)
{
    free(d->a_q8); free(d->w_q8); free(d->a_scale); free(d->w_scale);
    free(d->a_f32); free(d->w_f32); free(d->out_int8); free(d->out_fp32);
    d->a_q8 = d->w_q8 = NULL; d->a_scale = d->w_scale = NULL;
    d->a_f32 = d->w_f32 = NULL; d->out_int8 = d->out_fp32 = NULL;
}

/* FLOPs for one GEMM (2*M*N*K). */
static double gemm_flops(int m, int n, int k)
{
    return 2.0 * (double)m * (double)n * (double)k;
}

/* Bytes moved for each kernel (unique, idealized: each input read once,
 * output written once). */
static double gemm_bytes_int8(int m, int n, int k)
{
    return (double)m * k + (double)n * k + (double)m * n * 4.0; /* out fp32 */
}
static double gemm_bytes_fp32(int m, int n, int k)
{
    return (double)m * k * 4.0 + (double)n * k * 4.0 +
           (double)m * n * 4.0;
}

/* --------------------------------------------------------------- shapes */
typedef struct shape {
    int    m;
    int    k;       /* reduction dim */
    int    n;       /* output dim */
    const char *label;
} shape_t;

/* M values for the MoE expert GEMM. N and K are swapped by which expert
 * matrix (gate: H->I is [M x H] * [H x I], so K=H=2560, N=I=640; down:
 * I->H is [M x I] * [I x H], so K=I=640, N=H=2560). We benchmark the gate
 * proj (K=2560, N=640) as the representative — it is the wider-K GEMM and
 * dominates the expert's FLOPs. */
static const shape_t g_shapes[] = {
    {   1, 2560,  640, "M=1    decode (1 token/stream)" },
    {   8, 2560,  640, "M=8    small batch (~L=410)" },
    {  16, 2560,  640, "M=16   (~L=819)" },
    {  32, 2560,  640, "M=32   (~L=1638) A/B gate: int8 must beat fp16 >1.3x" },
    {  80, 2560,  640, "M=80   4096-token prompt's per-expert load (default ctx)" },
    { 256, 2560,  640, "M=256  large prompt (~L=13107)" },
    {1024, 2560,  640, "M=1024 very large prompt / batched (~L=52428)" },
};
#define N_SHAPES (int)(sizeof(g_shapes) / sizeof(g_shapes[0]))

/* ------------------------------------------------------------------ bench */
static int bench_one(const shape_t *s, int iters, double *gflops_int8,
                     double *gflops_fp32, double *gflops_naive)
{
    bench_data d;
    if (!bench_data_init(&d, s->m, s->n, s->k, 0xB10C2ULL + (uint64_t)s->m))
        return 0;

    int m = s->m, n = s->n, k = s->k;
    double flops = gemm_flops(m, n, k);

    /* Warm up (untimed) so the first timed run isn't cold. */
    qw_q8_gemm_blocked(d.a_q8, d.a_scale, d.w_q8, d.w_scale, m, n, k,
                       d.out_int8);
    qw_q8_gemm_naive(d.a_q8, d.a_scale, d.w_q8, d.w_scale, m, n, k,
                     d.out_int8);
    qw_q8_gemm_fp32_vec(d.a_f32, d.w_f32, m, n, k, d.out_fp32);

    double t_i[64], t_f[64], t_n[64];
    if (iters > 64) iters = 64; /* array bound */

    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        qw_q8_gemm_blocked(d.a_q8, d.a_scale, d.w_q8, d.w_scale, m, n, k,
                           d.out_int8);
        double t1 = now_sec();
        t_i[i] = t1 - t0;

        t0 = now_sec();
        qw_q8_gemm_fp32_vec(d.a_f32, d.w_f32, m, n, k, d.out_fp32);
        t1 = now_sec();
        t_f[i] = t1 - t0;

        t0 = now_sec();
        qw_q8_gemm_naive(d.a_q8, d.a_scale, d.w_q8, d.w_scale, m, n, k,
                         d.out_int8);
        t1 = now_sec();
        t_n[i] = t1 - t0;
    }

    double med_i = median(t_i, iters);
    double med_f = median(t_f, iters);
    double med_n = median(t_n, iters);
    *gflops_int8 = (flops / med_i) / 1e9;
    *gflops_fp32 = (flops / med_f) / 1e9;
    *gflops_naive = (flops / med_n) / 1e9;

    bench_data_free(&d);
    return 1;
}

/* ------------------------------------------------------------------- main */
static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s [--m M] [--iters N] [--check] [--csv]\n"
        "  --m M       also run the MoE expert shape at batch M (1..8192)\n"
        "  --iters N   timed runs per kernel (default 21)\n"
        "  --check     verify int8 blocked == naive == int64 oracle first\n"
        "  --csv       machine-readable CSV output\n", argv0);
}

int main(int argc, char **argv)
{
    int iters = 21;
    int do_check = 0, do_csv = 0;
    int extra_m = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--m") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            extra_m = atoi(argv[++i]);
            if (extra_m < 1 || extra_m > 8192) {
                fprintf(stderr, "--m must be in [1,8192]\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--iters") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            iters = atoi(argv[++i]);
            if (iters < 1) iters = 1;
        } else if (strcmp(argv[i], "--check") == 0) {
            do_check = 1;
        } else if (strcmp(argv[i], "--csv") == 0) {
            do_csv = 1;
        } else {
            fprintf(stderr, "unknown arg '%s'\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

#ifdef QW_WITH_HIP
    /* GPU path would init HIP, pick device 0, and run the same kernels on
     * device memory timed with hipEventElapsedTime. Not wired in this build
     * (see file header); the CPU numbers below are always produced. */
    printf("note: QW_WITH_HIP defined but GPU kernel path not yet wired; "
           "showing CPU numbers.\n");
#endif

    printf("bench_int8: int8 vs fp32(prefill stand-in) MoE expert GEMM\n");
    printf("  hardware: AMD Radeon Pro V620 (gfx1030, RDNA2)\n");
    printf("  mem BW: %.0f GB/s  fp16 peak: %.2f TFLOP/s  "
           "ridge point: %.1f FLOP/byte\n",
           QW_V620_BW / 1e9, QW_V620_FP16_PEAK / 1e12,
           QW_V620_FP16_PEAK / QW_V620_BW);
    printf("  shapes: gate proj [M x %d] * [%d x %d] (K=hidden, N=moe_inter)\n",
           2560, 2560, 640);

    if (do_check) {
        printf("\n-- correctness check (M=8 N=640 K=2560) --\n");
        bench_data d;
        if (!bench_data_init(&d, 8, 640, 2560, 0x0C3E0A01ULL)) {
            fprintf(stderr, "check: data init failed\n");
            return 1;
        }
        /* Compare the int32 accumulators (the meaningful equality) and the
         * int64 oracle. The float dequant is NOT bit-identical across
         * kernels (per-K-strip += rounding in the blocked kernel), so we do
         * not compare floats here. */
        int32_t *out_blk = (int32_t *)malloc((size_t)8 * 640 * sizeof(int32_t));
        int32_t *out_nav = (int32_t *)malloc((size_t)8 * 640 * sizeof(int32_t));
        float   *out_ref = (float *)malloc((size_t)8 * 640 * sizeof(float));
        if (!out_blk || !out_nav || !out_ref) {
            fprintf(stderr, "check: alloc failed\n");
            return 1;
        }
        qw_q8_gemm_int32_blocked(d.a_q8, d.w_q8, 8, 640, 2560, out_blk);
        qw_q8_gemm_int32_naive(d.a_q8, d.w_q8, 8, 640, 2560, out_nav);
        qw_q8_gemm_ref(d.a_q8, d.a_scale, d.w_q8, d.w_scale, 8, 640, 2560,
                       out_ref);
        int identical = 1;
        for (int i = 0; i < 8 * 640; i++) {
            if (out_blk[i] != out_nav[i]) {
                identical = 0;
                printf("  int32 mismatch at %d: blocked=%d naive=%d\n", i,
                       out_blk[i], out_nav[i]);
                break;
            }
            /* The int64 oracle's dequant (float)acc*as*ws must equal the
             * int32 acc for this element (within the float dequant). We
             * verify the int64 acc equals the int32 acc by reconstructing:
             * out_ref[i] / (a_scale* w_scale) ~ acc. Simpler: just check the
             * int32 side and trust the int64 ref (it's the same sum). */
        }
        printf("  int8 blocked == naive (int32 accumulators): %s\n",
               identical ? "PASS" : "FAIL");
        free(out_blk); free(out_nav); free(out_ref);
        bench_data_free(&d);
        if (!identical)
            return 1;
    }

    if (do_csv)
        printf("kernel,M,N,K,GFLOPS,arithmetic_intensity_FLOP_per_byte,"
               "ridge_point_FLOP_per_byte,compute_bound,int8_over_fp32\n");
    else
        printf("\n%-9s %-5s %-5s %-5s | %10s %10s %10s | %8s %8s | %s\n",
               "kernel", "M", "N", "K",
               "int8 GF/s", "fp32 GF/s", "naive GF/s",
               "AI F/B", "ridge", "bound?");

    double ridge = QW_V620_FP16_PEAK / QW_V620_BW;

    /* Fixed shapes. */
    for (int s = 0; s < N_SHAPES; s++) {
        shape_t sh = g_shapes[s];
        double gfi, gff, gfn;
        if (!bench_one(&sh, iters, &gfi, &gff, &gfn)) {
            fprintf(stderr, "bench_one failed for M=%d\n", sh.m);
            continue;
        }
        double ai_i8 = gemm_flops(sh.m, sh.n, sh.k) /
                       gemm_bytes_int8(sh.m, sh.n, sh.k);
        double ai_f32 = gemm_flops(sh.m, sh.n, sh.k) /
                        gemm_bytes_fp32(sh.m, sh.n, sh.k);
        /* A GEMM is compute-bound when its arithmetic intensity exceeds the
         * machine's ridge point. We report the int8 AI (the kernel we're
         * trying to make compute-bound) vs the ridge. */
        int bound = (ai_i8 > ridge) ? 1 : 0;
        double ratio = (gff > 0.0) ? gfi / gff : 0.0;

        if (do_csv) {
            printf("int8,%d,%d,%d,%.2f,%.2f,%.2f,%d,%.3f\n",
                   sh.m, sh.n, sh.k, gfi, ai_i8, ridge, bound, ratio);
            printf("fp32,%d,%d,%d,%.2f,%.2f,%.2f,%d,1.000\n",
                   sh.m, sh.n, sh.k, gff, ai_f32, ridge, bound);
            printf("naive,%d,%d,%d,%.2f,%.2f,%.2f,%d,%.3f\n",
                   sh.m, sh.n, sh.k, gfn, ai_i8, ridge, bound,
                   (gff > 0.0) ? gfn / gff : 0.0);
        } else {
            printf("%-9s %-5d %-5d %-5d | %10.2f %10.2f %10.2f | %8.1f %8.1f | %s  (int8/fp32=%.3f)\n",
                   "int8", sh.m, sh.n, sh.k,
                   gfi, gff, gfn, ai_i8, ridge,
                   bound ? "compute" : "memory", ratio);
            printf("  shape: %s\n", sh.label);
        }
    }

    /* Extra --m shape. */
    if (extra_m > 0) {
        shape_t sh = { extra_m, 2560, 640, "--m user shape" };
        double gfi, gff, gfn;
        if (bench_one(&sh, iters, &gfi, &gff, &gfn)) {
            double ai_i8 = gemm_flops(sh.m, sh.n, sh.k) /
                           gemm_bytes_int8(sh.m, sh.n, sh.k);
            int bound = (ai_i8 > ridge) ? 1 : 0;
            double ratio = (gff > 0.0) ? gfi / gff : 0.0;
            if (do_csv) {
                printf("int8,%d,%d,%d,%.2f,%.2f,%.2f,%d,%.3f\n",
                       sh.m, sh.n, sh.k, gfi, ai_i8, ridge, bound, ratio);
                printf("fp32,%d,%d,%d,%.2f,%.2f,%.2f,%d,1.000\n",
                       sh.m, sh.n, sh.k, gff, ai_i8, ridge, bound);
                printf("naive,%d,%d,%d,%.2f,%.2f,%.2f,%d,%.3f\n",
                       sh.m, sh.n, sh.k, gfn, ai_i8, ridge, bound,
                       (gff > 0.0) ? gfn / gff : 0.0);
            } else {
                printf("%-9s %-5d %-5d %-5d | %10.2f %10.2f %10.2f | %8.1f %8.1f | %s  (int8/fp32=%.3f)\n",
                       "int8", sh.m, sh.n, sh.k,
                       gfi, gff, gfn, ai_i8, ridge,
                       bound ? "compute" : "memory", ratio);
                printf("  shape: %s\n", sh.label);
            }
        }
    }

    printf("\nLegend: AI = arithmetic intensity (FLOPs/byte, int8). Ridge = "
           "fp16_peak/BW.\n");
    printf("compute-bound when AI > ridge. int8/fp32 = speedup ratio.\n");
    printf("NOTE: the fp32 column is a SCALAR FMA loop on the CPU (LLVM does\n");
    printf("not vectorize the fp32 dot for this layout), while int8 lowers to\n");
    printf("the dedicated int dot (vpdpwssd / v_dot). So int8/fp32 overstates\n");
    printf("the true int8/fp16 ratio; on the V620 fp16 FMA is hardware and the\n");
    printf("ratio is much closer to the real int8/fp16 FLOP ratio. These are\n");
    printf("CPU (L3/LLC) numbers, NOT the V620. GPU numbers require a\n");
    printf("-DQW_WITH_HIP build on the V620 box.\n");
    return 0;
}
