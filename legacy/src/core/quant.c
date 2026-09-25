/* src/core/quant.c — int8 quantization primitives (W8A8 / W4A8), CPU ref.
 *
 * See include/qw/quant.h for the layout and rationale. Pure C17, libc only.
 *
 * QW_ENABLE_DOT:
 *   When defined AND compiling for AMDGCN (__AMDGCN__), the blocked GEMM's
 *   K inner loop uses the integer dot-product builtin(s). This is DELIBERATELY
 *   the ONE place an unconfirmed intrinsic appears, isolated behind two #ifdefs
 *   so it can never silently break a non-GPU build:
 *     - #ifdef QW_ENABLE_DOT   : off by default; you must opt in.
 *     - #ifdef __AMDGCN__      : only real when hipcc targets a GCN arch.
 *   NOTE: the exact builtin name must be confirmed against the gfx1030 ISA.
 *   hipcc/LLVM exposes `__builtin_amdgcn_sdot4` (signed, 4-lane int8 -> int32)
 *   and `__builtin_amdgcn_u4dot` (unsigned 4-bit dot). gfx1030 (RDNA2) has
 *   v_dot4_vopd / v_dot2cq_vopc integer dot hardware, but whether LLVM emits
 *   these from a C loop on gfx1030 — and under which builtin — is the open
 *   question tools/bench_int8.c settles. Until then the plain int32 loop is
 *   the portable, correct path.
 */
#include "qw/quant.h"
#include "qw/quant_int4.h"
#include "qw/macros.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* QW_ENABLE_DOT is a build-time opt-in; the __AMDGCN__ guard is set by hipcc
 * only. A laptop build (cc, no hip) never reaches the builtin. */

/* ------------------------------------------------------------------ dot */

/* The ONE function in this file that may reference the AMDGPU integer
 * dot-product builtin. Isolated here (and only here) so that a non-GPU build
 * can never break on it: the builtin is compiled only when QW_ENABLE_DOT is
 * defined AND the translation unit is actually targeting AMDGCN (hipcc).
 *
 * CAVEAT: the exact builtin name for gfx1030 (RDNA2) is UNCONFIRMED. LLVM
 * exposes __builtin_amdgcn_sdot4 (signed, 4-lane int8 -> int32); whether it
 * is the right spelling for the RDNA2 v_dot4_vopd instruction — and whether
 * the backend even accepts it on this arch — is the open question
 * tools/bench_int8.c exists to settle. Do not spread this builtin anywhere
 * else; if it turns out to be wrong, only this function changes.
 */
#if defined(QW_ENABLE_DOT) && defined(__AMDGCN__)
static inline int32_t qw_sdot4(int32_t a, int32_t b)
{
    /* 4-lane signed int8 dot: (a.lane_i * b.lane_i) summed over i=0..3.
     * Lane i holds bits [8i, 8i+8) of the packed 32-bit input. */
    return __builtin_amdgcn_sdot4(a, b, 0);
}
#else
static inline int32_t qw_sdot4(int32_t a, int32_t b)
{
    /* Portable fallback: unpack the 4 lanes and multiply-accumulate. Same
     * result as the builtin, one v_dot's worth of work in scalar ALU ops. */
    return (int32_t)((a & 0xff) * (b & 0xff))
         + (int32_t)(((a >> 8)  & 0xff) * ((b >> 8)  & 0xff))
         + (int32_t)(((a >> 16) & 0xff) * ((b >> 16) & 0xff))
         + (int32_t)(((a >> 24) & 0xff) * ((b >> 24) & 0xff));
}
#endif /* QW_ENABLE_DOT && __AMDGCN__ */

/* ------------------------------------------------------------------ alloc */

qw_err qw_q8_block_alloc(qw_q8_block *out, int n, int k)
{
    if (QW_UNLIKELY(out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    size_t qn = (size_t)n * (size_t)k;
    out->n = n;
    out->k = k;
    out->w_q   = (int8_t *)calloc(qn, 1);
    out->scale = (float  *)malloc((size_t)n * sizeof(float));
    out->zp    = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (out->w_q == NULL || out->scale == NULL || out->zp == NULL) {
        qw_q8_block_free(out);
        memset(out, 0, sizeof(*out));
        return QW_ERR_ALLOC;
    }
    for (int i = 0; i < n; i++) {
        out->scale[i] = 1.0f;
        out->zp[i]    = 0;
    }
    return QW_OK;
}

void qw_q8_block_free(qw_q8_block *b)
{
    if (QW_UNLIKELY(b == NULL))
        return;
    free(b->w_q);
    free(b->scale);
    free(b->zp);
    b->w_q = NULL;
    b->scale = NULL;
    b->zp = NULL;
}

qw_err qw_act_q8_alloc(qw_act_q8 *out, int m, int k)
{
    if (QW_UNLIKELY(out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || k <= 0))
        return QW_ERR_RANGE;

    out->m = m;
    out->k = k;
    out->q     = (int8_t *)calloc((size_t)m * (size_t)k, 1);
    out->scale = (float  *)malloc((size_t)m * sizeof(float));
    if (out->q == NULL || out->scale == NULL) {
        free(out->q);
        free(out->scale);
        out->q = NULL;
        out->scale = NULL;
        return QW_ERR_ALLOC;
    }
    for (int i = 0; i < m; i++)
        out->scale[i] = 1.0f;
    return QW_OK;
}

void qw_act_q8_free(qw_act_q8 *a)
{
    if (QW_UNLIKELY(a == NULL))
        return;
    free(a->q);
    free(a->scale);
    a->q = NULL;
    a->scale = NULL;
}

size_t qw_q8_block_nbytes(const qw_q8_block *b)
{
    if (QW_UNLIKELY(b == NULL || b->w_q == NULL))
        return 0;
    /* w_q [N*K] + scale [N] float32 + zp [N] int32.
     * (The layout comment in quant.h mentions an alignment pad; the actual
     * heap arrays are separately allocated here, so "nbytes" is the logical
     * byte count of the three fields, which is what callers budget against.) */
    size_t n = (size_t)b->n, k = (size_t)b->k;
    return n * k + n * sizeof(float) + n * sizeof(int32_t);
}

/* ------------------------------------------------------------- pack/pack */

/* |x| without libm (fabsf): clear the sign bit via a conditional. */
static float qw_fabsf(float x)
{
    return (x < 0.0f) ? -x : x;
}

/* Clamp an int into [-127,127]. */
static int8_t qw_q8_sat(int v)
{
    if (v > 127)  return 127;
    if (v < -127) return -127;
    return (int8_t)v;
}

/* Round a float to the nearest int, half AWAY from zero, WITHOUT libm
 * (lrintf) and WITHOUT implementation-defined float->int conversion.
 *
 * Method: add/subtract 0.5 in the float domain (away from zero), then
 * truncate. The truncation is safe because the quotient is always in
 * [-127.0, 127.0] (guaranteed by the caller: |x| <= amax, and we divide by
 * amax/127), so the result after the +/-0.5 shift is in [-127.5, 127.5],
 * which truncates to an int in [-127, 127] — no overflow, no UB.
 *
 * This is round-half-away-from-zero (documented). It differs from IEEE
 * round-half-to-even only on exact .5 ties, which measure zero for
 * continuous data and only affect a single quantization level. */
static int qw_q8_roundf(float x)
{
    float y;
    if (x >= 0.0f)
        y = x + 0.5f;
    else
        y = x - 0.5f;
    /* y is in [-127.5, 127.5]; truncation toward zero is defined for these. */
    int i = (int)y;
    return i;
}

qw_err qw_q8_pack(const float *src, int n, int k, qw_q8_block *out)
{
    if (QW_UNLIKELY(src == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    qw_err e = qw_q8_block_alloc(out, n, k);
    if (e != QW_OK)
        return e;

    for (int i = 0; i < n; i++) {
        const float *row = src + (size_t)i * (size_t)k;
        float amax = 0.0f;
        for (int j = 0; j < k; j++) {
            float a = qw_fabsf(row[j]);
            if (a > amax) amax = a;
        }
        float scale;
        int8_t *qrow = out->w_q + (size_t)i * (size_t)k;
        if (amax == 0.0f) {
            scale = 1.0f; /* all-zero row: dequant yields exactly 0 */
            for (int j = 0; j < k; j++)
                qrow[j] = 0;
        } else {
            scale = amax / 127.0f;
            float inv = 1.0f / scale;
            for (int j = 0; j < k; j++) {
                /* Quotient is in [-127,127] by construction (|x|<=amax);
                 * round-half-away-from-zero, then clamp as a safety net. */
                int r = qw_q8_roundf(row[j] * inv);
                qrow[j] = qw_q8_sat(r);
            }
        }
        out->scale[i] = scale;
        out->zp[i]    = 0; /* symmetric */
    }
    return QW_OK;
}

qw_err qw_q8_dequant_row(const qw_q8_block *b, int n, float *dst)
{
    if (QW_UNLIKELY(b == NULL || b->w_q == NULL || dst == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n < 0 || n >= b->n))
        return QW_ERR_RANGE;

    const int8_t *row = b->w_q + (size_t)n * (size_t)b->k;
    float scale = b->scale[n];
    int32_t zp  = b->zp[n];
    for (int j = 0; j < b->k; j++)
        dst[j] = ((float)row[j] - (float)zp) * scale;
    return QW_OK;
}

/* ------------------------------------------------------ activation quant */

qw_err qw_act_q8_row(const float *x, int k, int8_t *q, float *scale_out)
{
    if (QW_UNLIKELY(x == NULL || q == NULL || scale_out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(k <= 0))
        return QW_ERR_RANGE;

    float amax = 0.0f;
    for (int j = 0; j < k; j++) {
        float a = qw_fabsf(x[j]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        /* All-zero row guard: scale := 1.0 (neutral), all q := 0.
         * Dequant of 0 with scale 1.0 is exactly 0 — documented behavior. */
        *scale_out = 1.0f;
        for (int j = 0; j < k; j++)
            q[j] = 0;
        return QW_OK;
    }

    float scale = amax / 127.0f;
    float inv = 1.0f / scale;
    for (int j = 0; j < k; j++) {
        /* Quotient in [-127,127] by construction; round-half-away, clamp. */
        int r = qw_q8_roundf(x[j] * inv);
        q[j] = qw_q8_sat(r);
    }
    *scale_out = scale;
    return QW_OK;
}

/* ----------------------------------------------------------- overflow chk */

bool qw_q8_overflow_ok(int m, int n, int k)
{
    if (m <= 0 || n <= 0 || k <= 0)
        return false;
    /* Worst-case |sum| = 127*127*K = 16129*K. INT32_MAX = 2147483647, so
     * K <= 2147483647 / 16129 = 133144 (16129*133145 > INT32_MAX). The
     * real model uses K=2560 (hidden) and K=640 (moe_intermediate) — both
     * far inside. */
    const int64_t prod = (int64_t)127 * (int64_t)127;
    return prod * (int64_t)k <= (int64_t)INT32_MAX;
}

/* ------------------------------------------------------------- GEMM ref */

qw_err qw_q8_gemm_ref(const int8_t *a_q8, const float *a_scale,
                      const int8_t *w_q8, const float *w_scale,
                      int m, int n, int k, float *out)
{
    if (QW_UNLIKELY(a_q8 == NULL || a_scale == NULL || w_q8 == NULL ||
                    w_scale == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    for (int i = 0; i < m; i++) {
        const int8_t *arow = a_q8 + (size_t)i * (size_t)k;
        float as = a_scale[i];
        float *orow = out + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            const int8_t *wrow = w_q8 + (size_t)j * (size_t)k;
            /* int64 local accumulator: proves no int32 overflow en route.
             * The caller is expected to have checked qw_q8_overflow_ok();
             * we still verify the final value fits before narrowing. */
            int64_t acc = 0;
            for (int t = 0; t < k; t++)
                acc += (int64_t)arow[t] * (int64_t)wrow[t];
            if (acc > INT32_MAX || acc < INT32_MIN)
                return QW_ERR_RANGE;
            orow[j] = (float)acc * as * w_scale[j];
        }
    }
    return QW_OK;
}

/* N-block size for the blocked kernel (the float and int32 variants).
 * See qw_q8_gemm_blocked for the K-innermost loop-order rationale. */
#define QW_Q8_BN 4  /* N-tile: rows of W / cols of C reused per A row */

qw_err qw_q8_gemm_int32_naive(const int8_t *a_q8, const int8_t *w_q8,
                              int m, int n, int k, int32_t *out)
{
    if (QW_UNLIKELY(a_q8 == NULL || w_q8 == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    for (int i = 0; i < m; i++) {
        const int8_t *arow = a_q8 + (size_t)i * (size_t)k;
        int32_t *orow = out + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            const int8_t *wrow = w_q8 + (size_t)j * (size_t)k;
            int32_t acc = 0;
            for (int t = 0; t < k; t++)
                acc += (int32_t)arow[t] * (int32_t)wrow[t];
            orow[j] = acc;
        }
    }
    return QW_OK;
}

/* K-outer blocked int32 GEMM.
 *
 * Loop order:
 *   for i (M)
 *     for jb (N-block, outer of the two inner loops)
 *       for ib (M-block, inner)
 *         acc[BR][BN] = 0
 *         for t (K, INNERMOST):  acc[i][jb+j] += a_q8[i][t] * w_q8[jb+j][t]
 *         out[i][jb+j] = acc
 *
 * Why K is innermost: the core `for t: acc[i][j] += a[i][t] * w[j][t]` is a
 * pure dot product over a single K dimension, with `i` and `j` fixed. That
 * is EXACTLY the shape LLVM lowers to `v_dot`-class integer dot products on
 * AMDGCN (and to AVX2 `vpmaddubsw` on x86) — the multiply-accumulate is a
 * straight-line induction over `t` with no other side effects. Keeping `i`
 * and `j` in the OUTER loops (not the innermost) is what lets the vectorizer
 * see the clean dot; the previous K-inner/i-j-inner layout defeated it.
 *
 * The (jb, ib) blocking reuses the A rows (mlen of them) across the whole N
 * block and the W rows (nlen of them) across the M block, so both tiles stay
 * in L1/L2 (and on the GPU, in LDS/registers). 8x4 = 32 accumulators matches
 * the gfx1030 wavefront width (one accumulator per lane).
 */
qw_err qw_q8_gemm_int32_blocked(const int8_t *a_q8, const int8_t *w_q8,
                                int m, int n, int k, int32_t *out)
{
    if (QW_UNLIKELY(a_q8 == NULL || w_q8 == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    for (int i = 0; i < m; i++) {
        const int8_t *arow = a_q8 + (size_t)i * (size_t)k;
        int32_t *orow = out + (size_t)i * (size_t)n;
        /* Split full N-blocks (4 complete W rows) from the tail (<4). The
         * full-block K-loop has NO boundary predicates, so LLVM can
         * vectorize it to vpdpwssd / v_dot. A per-element `if (e_j)` guard
         * inside the K-loop defeats that (the vectorizer cannot hoist a
         * data-independent-but-loop-invariant predicate across the induction
         * when it appears as a conditional on the accumulation). */
        int full_end = (n / QW_Q8_BN) * QW_Q8_BN;
        for (int jb = 0; jb < full_end; jb += QW_Q8_BN) {
            const int8_t *w0 = w_q8 + (size_t)jb * (size_t)k;
            const int8_t *w1 = w_q8 + (size_t)(jb + 1) * (size_t)k;
            const int8_t *w2 = w_q8 + (size_t)(jb + 2) * (size_t)k;
            const int8_t *w3 = w_q8 + (size_t)(jb + 3) * (size_t)k;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            /* 4 parallel int8 dots over arow, K innermost. */
            for (int t = 0; t < k; t++) {
                int32_t av32 = (int32_t)arow[t];
                a0 += av32 * (int32_t)w0[t];
                a1 += av32 * (int32_t)w1[t];
                a2 += av32 * (int32_t)w2[t];
                a3 += av32 * (int32_t)w3[t];
            }
            orow[jb + 0] = a0;
            orow[jb + 1] = a1;
            orow[jb + 2] = a2;
            orow[jb + 3] = a3;
        }
        /* Tail: <4 W rows, scalar. */
        for (int jb = full_end; jb < n; jb++) {
            const int8_t *w0 = w_q8 + (size_t)jb * (size_t)k;
            int32_t a0 = 0;
            for (int t = 0; t < k; t++)
                a0 += (int32_t)arow[t] * (int32_t)w0[t];
            orow[jb] = a0;
        }
    }
    return QW_OK;
}

qw_err qw_q8_gemm_fp32_vec(const float *a, const float *w,
                           int m, int n, int k, float *out)
{
    if (QW_UNLIKELY(a == NULL || w == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    /* K-innermost, 4-way N unroll, full/tail split — the same shape that
     * lets the int8 kernel vectorize (vpdpwssd for int8, FMA for fp32). This
     * is the fp32-accumulate baseline: a fair, fast stand-in for an fp16-input
     * GEMM with fp32 accumulation. */
    for (int i = 0; i < m; i++) {
        const float *arow = a + (size_t)i * (size_t)k;
        float *orow = out + (size_t)i * (size_t)n;
        /* One output element (i,j) per K-loop, K innermost. This is the
         * fp32-accumulate baseline. On the CPU it compiles to a scalar FMA
         * loop (LLVM does not vectorize this fp32 dot for the [M][K]*[N][K]
         * layout into wide FMA — it keeps it scalar), in contrast to the int8
         * path which lowers to the dedicated integer dot (vpdpwssd / v_dot).
         * That asymmetry is real and is reported in the bench: the int8/fp32
         * ratio overstates the true int8/fp16 ratio on a GPU where fp16 FMA
         * is hardware. j is outer so the A row is reused across all N. */
        for (int j = 0; j < n; j++) {
            const float *w0 = w + (size_t)j * (size_t)k;
            float acc = 0.0f;
            for (int t = 0; t < k; t++)
                acc += arow[t] * w0[t];
            orow[j] = acc;
        }
    }
    return QW_OK;
}

qw_err qw_q8_gemm_naive(const int8_t *a_q8, const float *a_scale,
                        const int8_t *w_q8, const float *w_scale,
                        int m, int n, int k, float *out)
{
    if (QW_UNLIKELY(a_q8 == NULL || a_scale == NULL || w_q8 == NULL ||
                    w_scale == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    /* Unoptimized baseline: i, j, k loop order, int32 accumulator. */
    for (int i = 0; i < m; i++) {
        const int8_t *arow = a_q8 + (size_t)i * (size_t)k;
        float as = a_scale[i];
        float *orow = out + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            const int8_t *wrow = w_q8 + (size_t)j * (size_t)k;
            int32_t acc = 0;
            for (int t = 0; t < k; t++)
                acc += (int32_t)arow[t] * (int32_t)wrow[t];
            orow[j] = (float)acc * as * w_scale[j];
        }
    }
    return QW_OK;
}

/* ----------------------------------------------------------- GEMM blocked */

/* Cache-blocked, K-innermost int8 GEMM (the "blocked" benchmark kernel).
 *
 * Loop order:
 *   for i (M, outer)
 *     for jb (N-block, BN=4)
 *       acc[BN] = 0
 *       for t (K, INNERMOST):  acc[j] += a_q8[i][t] * w_q8[jb+j][t]
 *       for j:  out[i][jb+j] = (float)acc[j] * a_scale[i] * w_scale[jb+j]
 *
 * Why K is innermost (the key transformation for LLVM auto-vectorization):
 *   The core `for t: acc[j] += a[i][t] * w[jb+j][t]` is a pure dot product
 *   over a single K dimension with `i` and `j` fixed. That is EXACTLY the
 *   shape the AMDGPU backend lowers to v_dot-class integer dot products on
 *   gfx1030 (and AVX2 vpmaddubsw on x86) — a straight-line multiply-
 *   accumulate induction over `t` with no other side effects. A K-INNER /
 *   i-j-INNERMOST layout (the textbook "blocked GEMM") defeats this: the
 *   vectorizer cannot see a clean dot because the multiply is nested inside
 *   two other loops whose indices it cannot hoist.
 *
 * The N-block (BN=4) reuses the single A row `arow` across all 4 W rows of
 * the block (arow stays in registers/L1), and the W rows are streamed once
 * per (i, jb). On the GPU, the A row maps to LDS and the 4 W rows to
 * registers — the 4 accumulators are one per lane of a 4-wide subgroup,
 * generalizing to the 32-wide wavefront (one accumulator per lane).
 *
 * Dequant is fused in the epilogue: (float)acc * a_scale[i] * w_scale[jb+j]
 * computed once per output element, not inside the K loop.
 *
 * The int32 accumulators are safe by qw_q8_overflow_ok() (K<=133144); the
 * real model uses K=2560/640.
 */
qw_err qw_q8_gemm_blocked(const int8_t *a_q8, const float *a_scale,
                          const int8_t *w_q8, const float *w_scale,
                          int m, int n, int k, float *out)
{
    if (QW_UNLIKELY(a_q8 == NULL || a_scale == NULL || w_q8 == NULL ||
                    w_scale == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0))
        return QW_ERR_RANGE;

    for (int i = 0; i < m; i++) {
        const int8_t *arow = a_q8 + (size_t)i * (size_t)k;
        float as = a_scale[i];
        float *orow = out + (size_t)i * (size_t)n;
        /* Same full/tail split as the int32 variant: no boundary predicates
         * in the full-block K-loop so LLVM vectorizes to vpdpwssd / v_dot. */
        int full_end = (n / QW_Q8_BN) * QW_Q8_BN;
        for (int jb = 0; jb < full_end; jb += QW_Q8_BN) {
            const int8_t *w0 = w_q8 + (size_t)jb * (size_t)k;
            const int8_t *w1 = w_q8 + (size_t)(jb + 1) * (size_t)k;
            const int8_t *w2 = w_q8 + (size_t)(jb + 2) * (size_t)k;
            const int8_t *w3 = w_q8 + (size_t)(jb + 3) * (size_t)k;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            /* ---- K innermost, 4 parallel int8 dots ---- */
            for (int t = 0; t < k; t++) {
                int32_t av32 = (int32_t)arow[t];
                a0 += av32 * (int32_t)w0[t];
                a1 += av32 * (int32_t)w1[t];
                a2 += av32 * (int32_t)w2[t];
                a3 += av32 * (int32_t)w3[t];
            }
            /* ---- fused dequant epilogue ---- */
            orow[jb + 0] = (float)a0 * as * w_scale[jb + 0];
            orow[jb + 1] = (float)a1 * as * w_scale[jb + 1];
            orow[jb + 2] = (float)a2 * as * w_scale[jb + 2];
            orow[jb + 3] = (float)a3 * as * w_scale[jb + 3];
        }
        for (int jb = full_end; jb < n; jb++) {
            const int8_t *w0 = w_q8 + (size_t)jb * (size_t)k;
            int32_t a0 = 0;
            for (int t = 0; t < k; t++)
                a0 += (int32_t)arow[t] * (int32_t)w0[t];
            orow[jb] = (float)a0 * as * w_scale[jb];
        }
    }
    return QW_OK;
}

/* ------------------------------------------------------- accuracy helpers */

double qw_q8_err_mse(const float *a, const float *b, int n)
{
    if (a == NULL || b == NULL || n <= 0)
        return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return s / (double)n;
}

/* Newton sqrt on double, portable, no libm. Used for cosine similarity and
 * Box-Muller test-data generation.
 *
 * Convergence note: the fixed-point iteration g -> (g + x/g)/2 halves the
 * LOG-distance to sqrt(x) only once g is within a factor of ~2 of the answer;
 * from a naive guess (x or 1.0) it takes O(log2(x)) iterations for large x
 * (e.g. x ~ 5e6 needs ~15). So the guess is scaled by the magnitude of x:
 * count the binary exponent of x (log2 floor via successive /2) and start at
 * g = 1.5 * 2^(e/2), which is within a factor of 2 of sqrt(x) for every x.
 * From there 8 Newton iterations give full double precision (~53 bits: each
 * step doubles the correct bits). */
static double qw_isqrt(double x)
{
    if (x <= 0.0)
        return 0.0;
    double t = x;
    int e = 0;
    while (t >= 2.0)  { t *= 0.5; e++; }
    while (t < 1.0)   { t *= 2.0; e--; }
    /* Now t in [1, 2), so x = t * 2^e and sqrt(x) = sqrt(t) * 2^(e/2).
     * sqrt(t) in [1, sqrt(2)); start at 1.5, scale by 2^(e/2). */
    double g = 1.5;
    int half = e >> 1;
    if (half > 0) { for (int i = 0; i < half; i++) g *= 2.0; }
    else          { for (int i = 0; i < -half; i++) g *= 0.5; }
    for (int i = 0; i < 8; i++)
        g = 0.5 * (g + x / g);
    return g;
}

double qw_q8_cosine_sim(const float *a, const float *b, int n)
{
    if (a == NULL || b == NULL || n <= 0)
        return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        double x = (double)a[i], y = (double)b[i];
        dot += x * y;
        na  += x * x;
        nb  += y * y;
    }
    if (na == 0.0 || nb == 0.0)
        return 1.0; /* both (or one) zero => trivially aligned */
    double sim = dot / (qw_isqrt(na) * qw_isqrt(nb));
    /* Cauchy-Schwarz gives |sim| <= 1; clamp rounding drift off the unit
     * interval so callers can use the value as a plain similarity score. */
    if (sim > 1.0)
        return 1.0;
    if (sim < -1.0)
        return -1.0;
    return sim;
}

/* Deterministic LCG (Numerical Recipes constants). Returns next value in
 * [0, 2^32). Pure integer — no libm. */
static uint64_t qw_lcg_next(uint64_t *s)
{
    *s = (*s) * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s >> 32; /* top 32 bits */
}

/* Polynomial sin(x) for |x| <= pi (minimax-style, ~1e-7 accuracy). No libm.
 * Used only to generate test data (Box-Muller), not on the perf path. */
static double qw_sin(double x)
{
    /* Reduce to [-pi, pi]. */
    while (x > 3.14159265358979323846)  x -= 6.28318530717958647692;
    while (x < -3.14159265358979323846) x += 6.28318530717958647692;
    /* Odd polynomial in x on [-pi,pi]: x - x^3/6 + x^5/120 - x^7/5040. */
    double x2 = x * x;
    return x * (1.0 + x2 * (-1.0/6.0 + x2 * (1.0/120.0 - x2 / 5040.0)));
}

/* True standard-normal via Box-Muller, from two LCG uniforms, WITHOUT libm
 * (uses qw_sin + a Newton log). Used only to generate test/compare data so
 * the int8 accuracy figures are meaningful (a gaussian with std 1 quantizes
 * cleanly to int8; a wide non-gaussian does not). */
static double qw_log(double x)
{
    /* Newton refinement of ln around a coarse scaling. For x in (0, inf),
     * scale to [0.5, 2) then use a series. Good enough for test data. */
    if (x <= 0.0) return -1e30;
    int e = 0;
    while (x >= 2.0)      { x *= 0.5; e++; }
    while (x < 1.0)       { x *= 2.0; e--; }
    /* Now x in [1,2): ln(x) via atanh series: ln(x)=2*(z + z^3/3 + z^5/5+..),
     * z=(x-1)/(x+1). x in [1,2) => z in [0, 1/3), converges fast. */
    double z = (x - 1.0) / (x + 1.0);
    double z2 = z * z;
    double s = z, term = z;
    for (int i = 1; i < 12; i++) {
        term *= z2;
        s += term / (2 * i + 1);
    }
    double ln = 2.0 * s + (double)e * 0.69314718055994530942;
    return ln;
}

static float qw_gaussian(uint64_t *s)
{
    double u1 = ((double)(qw_lcg_next(s) & 0xffffff) + 1.0) / (1.0 + 16777216.0);
    double u2 = ((double)(qw_lcg_next(s) & 0xffffff) + 1.0) / (1.0 + 16777216.0);
    double r  = qw_isqrt(-2.0 * qw_log(u1));
    double th = 2.0 * 3.14159265358979323846 * u2;
    double g  = r * qw_sin(th);
    if (g > 8.0)  g = 8.0;
    if (g < -8.0) g = -8.0;
    return (float)g;
}

/* Plain fp32-accumulate GEMM: the numerically honest reference for what an
 * fp16 GEMM with fp32 accumulation would produce (fp16 inputs, fp32 acc). */
qw_err qw_q8_gemm_fp32(const float *a, const float *w,
                       int m, int n, int k, float *out)
{
    if (a == NULL || w == NULL || out == NULL || m <= 0 || n <= 0 || k <= 0)
        return QW_ERR_NULL;
    for (int i = 0; i < m; i++) {
        const float *arow = a + (size_t)i * (size_t)k;
        float *orow = out + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            const float *wrow = w + (size_t)j * (size_t)k;
            float acc = 0.0f;
            for (int t = 0; t < k; t++)
                acc += arow[t] * wrow[t];
            orow[j] = acc;
        }
    }
    return QW_OK;
}

qw_err qw_q8_compare_fp16(int dim_m, int dim_n, int dim_k, uint64_t seed,
                          double *mse_out, double *cos_out)
{
    if (QW_UNLIKELY(dim_m <= 0 || dim_n <= 0 || dim_k <= 0))
        return QW_ERR_RANGE;
    if (!qw_q8_overflow_ok(dim_m, dim_n, dim_k))
        return QW_ERR_RANGE; /* K too large for int32 accumulator */

    size_t a_n = (size_t)dim_m * (size_t)dim_k;
    size_t w_n = (size_t)dim_n * (size_t)dim_k;
    size_t o_n = (size_t)dim_m * (size_t)dim_n;

    float *a   = (float *)malloc(a_n * sizeof(float));
    float *w   = (float *)malloc(w_n * sizeof(float));
    float *out_ref  = (float *)malloc(o_n * sizeof(float));
    float *out_q8   = (float *)malloc(o_n * sizeof(float));
    int8_t *a_q8    = (int8_t *)malloc(a_n);
    int8_t *w_q8    = (int8_t *)malloc(w_n);
    float  *a_scale = (float  *)malloc((size_t)dim_m * sizeof(float));
    float  *w_scale = (float  *)malloc((size_t)dim_n * sizeof(float));

    if (a == NULL || w == NULL || out_ref == NULL || out_q8 == NULL ||
        a_q8 == NULL || w_q8 == NULL || a_scale == NULL || w_scale == NULL) {
        free(a); free(w); free(out_ref); free(out_q8);
        free(a_q8); free(w_q8); free(a_scale); free(w_scale);
        return QW_ERR_ALLOC;
    }

    /* Fill gaussian data (deterministic via seeded LCG). */
    uint64_t s = seed ? seed : 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < a_n; i++) a[i] = qw_gaussian(&s);
    for (size_t i = 0; i < w_n; i++) w[i] = qw_gaussian(&s);

    /* Quantize weights: per-channel (per row of W). */
    {
        qw_q8_block blk;
        qw_err e = qw_q8_pack(w, dim_n, dim_k, &blk);
        if (e != QW_OK) {
            free(a); free(w); free(out_ref); free(out_q8);
            free(a_q8); free(w_q8); free(a_scale); free(w_scale);
            return e;
        }
        memcpy(w_q8, blk.w_q, w_n);
        memcpy(w_scale, blk.scale, (size_t)dim_n * sizeof(float));
        qw_q8_block_free(&blk);
    }
    /* Quantize activations: per-row (per token). */
    for (int i = 0; i < dim_m; i++) {
        qw_err e = qw_act_q8_row(a + (size_t)i * (size_t)dim_k, dim_k,
                                 a_q8 + (size_t)i * (size_t)dim_k,
                                 &a_scale[i]);
        if (e != QW_OK) {
            free(a); free(w); free(out_ref); free(out_q8);
            free(a_q8); free(w_q8); free(a_scale); free(w_scale);
            return e;
        }
    }

    /* fp32-accumulate reference (stand-in for fp16-in/fp32-acc GEMM). */
    qw_err e = qw_q8_gemm_fp32(a, w, dim_m, dim_n, dim_k, out_ref);
    if (e != QW_OK) {
        free(a); free(w); free(out_ref); free(out_q8);
        free(a_q8); free(w_q8); free(a_scale); free(w_scale);
        return e;
    }
    /* int8 GEMM (blocked kernel == naive on correctness). */
    e = qw_q8_gemm_blocked(a_q8, a_scale, w_q8, w_scale, dim_m, dim_n, dim_k,
                           out_q8);
    if (e != QW_OK) {
        free(a); free(w); free(out_ref); free(out_q8);
        free(a_q8); free(w_q8); free(a_scale); free(w_scale);
        return e;
    }

    double mse = qw_q8_err_mse(out_q8, out_ref, (int)o_n);
    double cos = qw_q8_cosine_sim(out_q8, out_ref, (int)o_n);

    fprintf(stderr,
            "qw_q8_compare_fp16: M=%d N=%d K=%d seed=%llu\n"
            "  W8A8 vs fp32-acc ref:  MSE = %.6e   cosine_sim = %.9f\n",
            dim_m, dim_n, dim_k, (unsigned long long)seed, mse, cos);

    if (mse_out) *mse_out = mse;
    if (cos_out) *cos_out = cos;

    free(a); free(w); free(out_ref); free(out_q8);
    free(a_q8); free(w_q8); free(a_scale); free(w_scale);
    return QW_OK;
}

/* ====================================================================
 * int4 weights (W4A8) — see include/qw/quant_int4.h for the layout and
 * the packing convention. Pure C17, libc only, same style as the int8
 * layer above. Every builtin reference is gated exactly like qw_sdot4:
 * #if defined(QW_ENABLE_DOT) && defined(__AMDGCN__) — a non-GPU build
 * never sees an unknown builtin.
 * ==================================================================== */

/* ------------------------------------------------------------------ dot */

/* W4A8 dot note (see qw_q4_gemm_widen below).
 *
 * Confirmed against the LLVM 24 AMDGPU user guide: gfx1030 (RDNA2) has
 * v_dot8_i32_i4 (8 x int4 -> int32) and v_dot4_i32_i8 (4 x int8 -> int32,
 * builtin __builtin_amdgcn_sdot4(a,b,acc,clamp)). There is NO signed int4
 * dot on gfx1030 (and no __builtin_amdgcn_wmma_* / __builtin_amdgcn_sudot4
 * — those are gfx11+), so the int4 side of the W4A8 dot is widened to int8
 * in registers and runs through 2 x sdot4 per 8 values (each sdot4 covers
 * 4 contiguous lanes). The existing qw_sdot4 (above) is reused directly —
 * no separate 8-lane helper is needed, because the 8 values are split into
 * two 4-lane groups that are dotted separately (OR-ing them into one word
 * would merge lanes from different groups into the same byte).
 *
 * The widened-register kernel accumulates the UNBIASED int8 weights
 * (w in [-8,7]); the unsigned v_dot8 bias correction (w -> w+8 in [0,15],
  * then subtract 8*sum_a in the epilogue) is a separate, exactly-invertible
 * transform, kept out of the hot loop so the int32 accumulator stays
 * bit-identical to the naive oracle's sum_k a[k]*w[k].
 */

/* Signed 4-lane int8 dot.
 *
 * NOTE: the existing qw_sdot4 (top of file) extracts its lanes UNSIGNED
 * ((a & 0xff) -> 0..255), which is correct for the W8A8 path where both
 * operands are non-negative after its own handling but WRONG for the W4A8
 * path where the widened int4 weights ([-8,7]) and int8 activations ([-127,
 * 127]) are SIGNED and must sign-extend. The W4A8 kernel therefore uses this
 * separate signed helper (additive; qw_sdot4 is left untouched) so a negative
 * lane (e.g. -8 = 0xF8) multiplies as -8, not 248. */
#if defined(QW_ENABLE_DOT) && defined(__AMDGCN__)
static inline int32_t qw_sdot4_s(int32_t a, int32_t b)
{
    /* The __builtin_amdgcn_sdot4 is already the SIGNED 4-lane int8 dot
     * (its "s" = signed), so on the GPU this maps straight to
     * v_dot4_i32_i8 — no extra sign handling needed. */
    return __builtin_amdgcn_sdot4(a, b, 0);
}
#else
static inline int32_t qw_sdot4_s(int32_t a, int32_t b)
{
    /* Portable fallback: sign-extend each lane, then multiply-accumulate. */
    int32_t s = 0;
    for (int i = 0; i < 4; i++) {
        int32_t av = (int32_t)((a >> (8 * i)) & 0xff);
        int32_t bv = (int32_t)((b >> (8 * i)) & 0xff);
        if (av >= 0x80) av -= 0x100; /* sign-extend int8 */
        if (bv >= 0x80) bv -= 0x100;
        s += av * bv;
    }
    return s;
}
#endif /* QW_ENABLE_DOT && __AMDGCN__ */

/* Sign-extend a 4-bit two's-complement nibble to int ([-8,7]). */
static int qw_q4_nib_to_int(uint8_t nib)
{
    int v = (int)(nib & 0x0F);
    if (v >= 8)
        v -= 16;
    return v;
}

/* Round a float to the nearest int, half AWAY from zero (same method as
 * qw_q8_roundf above; the quotient is in [-7.0,7.0] by construction so the
 * truncation is safe). */
static int qw_q4_roundf(float x)
{
    float y;
    if (x >= 0.0f)
        y = x + 0.5f;
    else
        y = x - 0.5f;
    return (int)y;
}

/* Clamp an int into [-8,7]. */
static int qw_q4_sat(int v)
{
    if (v > 7)
        return 7;
    if (v < -8)
        return -8;
    return v;
}

/* ------------------------------------------------------------------ alloc */

qw_err qw_q4_block_alloc(qw_q4_block *out, int n, int k)
{
    if (QW_UNLIKELY(out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n <= 0 || k <= 0 || (k & 1) != 0))
        return QW_ERR_RANGE;

    size_t pn = (size_t)n * (size_t)(k / 2);
    out->n = n;
    out->k = k;
    out->w_p   = (uint8_t *)calloc(pn, 1);
    out->scale = (float  *)malloc((size_t)n * sizeof(float));
    if (out->w_p == NULL || out->scale == NULL) {
        qw_q4_block_free(out);
        memset(out, 0, sizeof(*out));
        return QW_ERR_ALLOC;
    }
    for (int i = 0; i < n; i++)
        out->scale[i] = 1.0f;
    return QW_OK;
}

void qw_q4_block_free(qw_q4_block *b)
{
    if (QW_UNLIKELY(b == NULL))
        return;
    free(b->w_p);
    free(b->scale);
    b->w_p = NULL;
    b->scale = NULL;
}

size_t qw_q4_block_nbytes(const qw_q4_block *b)
{
    if (QW_UNLIKELY(b == NULL || b->w_p == NULL))
        return 0;
    size_t n = (size_t)b->n, k = (size_t)b->k;
    return n * (k / 2) + n * sizeof(float);
}

/* ------------------------------------------------------------- pack/unpack */

qw_err qw_q4_pack(const float *src, int n, int k, qw_q4_block *out)
{
    if (QW_UNLIKELY(src == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n <= 0 || k <= 0 || (k & 1) != 0))
        return QW_ERR_RANGE;

    qw_err e = qw_q4_block_alloc(out, n, k);
    if (e != QW_OK)
        return e;

    for (int i = 0; i < n; i++) {
        const float *row = src + (size_t)i * (size_t)k;
        float amax = 0.0f;
        for (int j = 0; j < k; j++) {
            float a = qw_fabsf(row[j]);
            if (a > amax) amax = a;
        }
        uint8_t *prow = out->w_p + (size_t)i * (size_t)(k / 2);
        if (amax == 0.0f) {
            out->scale[i] = 1.0f; /* all-zero row: dequant yields exactly 0 */
            for (int j = 0; j < k / 2; j++)
                prow[j] = 0;
        } else {
            out->scale[i] = amax / 7.0f;
            float inv = 1.0f / out->scale[i];
            for (int j = 0; j < k / 2; j++) {
                /* Quotient in [-7,7] by construction (|x|<=amax);
                 * round-half-away-from-zero, then clamp as a safety net. */
                int qe = qw_q4_sat(qw_q4_roundf(row[2 * j]     * inv));
                int qo = qw_q4_sat(qw_q4_roundf(row[2 * j + 1] * inv));
                /* Packing convention (FIXED, see header): low nibble = even
                 * index, high nibble = odd index; each stored 4-bit
                 * two's-complement (qe, qo in [-8,7] => & 0x0F). */
                prow[j] = (uint8_t)((qe & 0x0F) | ((qo & 0x0F) << 4));
            }
        }
    }
    return QW_OK;
}

qw_err qw_q4_unpack_row(const qw_q4_block *b, int n, int *dst)
{
    if (QW_UNLIKELY(b == NULL || b->w_p == NULL || dst == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n < 0 || n >= b->n))
        return QW_ERR_RANGE;
    if (QW_UNLIKELY((b->k & 1) != 0))
        return QW_ERR_RANGE;

    const uint8_t *row = b->w_p + (size_t)n * (size_t)(b->k / 2);
    for (int j = 0; j < b->k / 2; j++) {
        dst[2 * j]     = qw_q4_nib_to_int(row[j] & 0x0F);      /* even idx */
        dst[2 * j + 1] = qw_q4_nib_to_int((row[j] >> 4) & 0x0F); /* odd idx */
    }
    return QW_OK;
}

/* ----------------------------------------------------------- overflow chk */

bool qw_q4_overflow_ok(int k)
{
    if (k <= 0)
        return false;
    /* Worst-case |sum| = 8*127*K = 1016*K. INT32_MAX = 2147483647, so
     * K <= 2147483647 / 1016 = 2113665 (1016*2113666 > INT32_MAX). The
     * real model uses K=2560 (hidden) and K=640 (moe_intermediate) —
     * 1016*2560 = 2600960, ~800x inside the limit. */
    const int64_t prod = (int64_t)8 * (int64_t)127;
    return prod * (int64_t)k <= (int64_t)INT32_MAX;
}

/* ------------------------------------------------------------- GEMM ref */

qw_err qw_q4_gemm_ref(const int8_t *a_q8, const float *a_scale,
                      const int32_t *w4, const float *w_scale,
                      int m, int n, int k, float *out)
{
    if (QW_UNLIKELY(a_q8 == NULL || a_scale == NULL || w4 == NULL ||
                    w_scale == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0 || (k & 1) != 0))
        return QW_ERR_RANGE;

    for (int i = 0; i < m; i++) {
        const int8_t *arow = a_q8 + (size_t)i * (size_t)k;
        float as = a_scale[i];
        float *orow = out + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            const int32_t *wrow = w4 + (size_t)j * (size_t)k;
            /* int32 accumulator; safe by qw_q4_overflow_ok(k). */
            int32_t acc = 0;
            for (int t = 0; t < k; t++)
                acc += (int32_t)arow[t] * wrow[t];
            orow[j] = (float)acc * as * w_scale[j];
        }
    }
    return QW_OK;
}

qw_err qw_q4_gemm_widen(const int8_t *a_q8, const uint8_t *w_p,
                        int m, int n, int k, int32_t *out)
{
    if (QW_UNLIKELY(a_q8 == NULL || w_p == NULL || out == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(m <= 0 || n <= 0 || k <= 0 || (k & 1) != 0))
        return QW_ERR_RANGE;

    for (int i = 0; i < m; i++) {
        const int8_t *arow = a_q8 + (size_t)i * (size_t)k;
        int32_t *orow = out + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            const uint8_t *wrow = w_p + (size_t)j * (size_t)(k / 2);
            int32_t acc = 0;
            /* 8-wide strips: unpack 8 nibbles (4 bytes) to 8 int8 weights,
             * then 2 x 4-lane sdot4 (one per 4 lanes). Lanes are contiguous:
             * strip offsets 0..7 map to weight indices 4s..4s+3 (even) and
             * 4s+4..4s+7 (odd). Sign-extend each 4-bit lane to 8 bits. */
            for (int s = 0; s < k / 8; s++) {
                const uint8_t *wp4 = wrow + s * 4;
                const int8_t  *a8  = arow + s * 8;
                /* Pack each int8 lane into its own byte: mask to the low byte
                 * FIRST (`& 0xff`), THEN shift up. The mask-to-byte-first order
                 * is REQUIRED: a negative int8 sign-extends to 0xFFFFFF.., so
                 * `(int32_t)x & 0xff00` would read the SIGN-EXTENSION byte
                 * (0xFF), not the value — whereas `((int32_t)x & 0xff) << 8`
                 * reads the actual low byte and moves it to slot 8. The same
                 * rule the qw_sdot4 fallback above relies on (its `>>8 & 0xff`
                 * extracts one clean byte at a time). */
                int32_t ae = (((int32_t)(int8_t)a8[0] & 0xff) << 0)
                           | (((int32_t)(int8_t)a8[1] & 0xff) << 8)
                           | (((int32_t)(int8_t)a8[2] & 0xff) << 16)
                           | (((int32_t)(int8_t)a8[3] & 0xff) << 24);
                int32_t ao = (((int32_t)(int8_t)a8[4] & 0xff) << 0)
                           | (((int32_t)(int8_t)a8[5] & 0xff) << 8)
                           | (((int32_t)(int8_t)a8[6] & 0xff) << 16)
                           | (((int32_t)(int8_t)a8[7] & 0xff) << 24);
                int32_t we = (((int32_t)qw_q4_nib_to_int(wp4[0] & 0x0F) & 0xff) << 0)
                           | (((int32_t)qw_q4_nib_to_int((wp4[0] >> 4) & 0x0F) & 0xff) << 8)
                           | (((int32_t)qw_q4_nib_to_int(wp4[1] & 0x0F) & 0xff) << 16)
                           | (((int32_t)qw_q4_nib_to_int((wp4[1] >> 4) & 0x0F) & 0xff) << 24);
                int32_t wo = (((int32_t)qw_q4_nib_to_int(wp4[2] & 0x0F) & 0xff) << 0)
                           | (((int32_t)qw_q4_nib_to_int((wp4[2] >> 4) & 0x0F) & 0xff) << 8)
                           | (((int32_t)qw_q4_nib_to_int(wp4[3] & 0x0F) & 0xff) << 16)
                           | (((int32_t)qw_q4_nib_to_int((wp4[3] >> 4) & 0x0F) & 0xff) << 24);
                 /* TWO separate signed 4-lane sdot4 dots = one 8-lane dot.
                  * ae/ao and we/wo each pack 4 int8 lanes into bytes 0..3, so
                  * they MUST be dotted separately (qw_sdot4_s) — OR-ing ae|ao
                  * would merge a[0] and a[4] into the same byte and corrupt
                  * both. qw_sdot4_s (not qw_sdot4) because the W4A8 lanes are
                  * SIGNED and must sign-extend (see its comment). This is 2 x
                  * v_dot4_i32_i8 per 8 values (gfx1030 has no signed int4
                  * dot). */
                 acc += qw_sdot4_s(ae, we) + qw_sdot4_s(ao, wo);
            }
            /* Scalar tail: the last <8 weights (even k => whole trailing
             * bytes). Same integer math as the oracle. */
            for (int t = (k / 8) * 8; t < k; t++) {
                int nib = (t & 1) ? (int)(wrow[t / 2] >> 4)
                                  : (int)(wrow[t / 2] & 0x0F);
                acc += (int32_t)arow[t] * (int32_t)qw_q4_nib_to_int((uint8_t)nib);
            }
            orow[j] = acc;
        }
    }
    return QW_OK;
}
