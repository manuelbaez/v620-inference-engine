/* qw/quant.h — int8 quantization primitives (W8A8 / W4A8) for gfx1030.
 *
 * Pure C17 CPU reference layer. No HIP, no external deps: libc only.
 *
 * Rationale (see docs/adr-002-int8-prefill.md):
 *   - decode is bandwidth-bound (M=1): int8 does not help there; weights stay
 *     int4, activations fp16 (GEMV path).
 *   - prefill is compute-bound, and because the model is MoE, a prompt of L
 *     tokens gives each active expert M = L*topk/512 tokens — at L=4096 that
 *     is M=80, a real GEMM shape. On gfx1030 there is no WMMA/MFMA and no
 *     warp-matrix API; int8 is the only post-fp16 numeric type with hardware
 *     support, so an int8 GEMM can at best rely on LLVM emitting v_dot-class
 *     integer dot-product instructions from a plain C loop. Whether it does
 *     is the open question tools/bench_int8.c exists to answer.
 *
 * This header is the CPU reference/oracle: it defines the layouts, the
 * quantization math, a naive GEMM (correctness oracle) and a blocked GEMM
 * (the shape we hope LLVM lowers to dot-product instructions on the GPU).
 */
#ifndef QW_QUANT_H
#define QW_QUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ types */

/* Symmetric per-channel (per output row) int8 weight block.
 *
 * Memory layout (GEMM-friendly: W is [N][K] row-major, i.e. each output
 * channel is a contiguous row of K int8 values; A is [M][K] row-major, so
 * C[m][n] = dot(A[m,:], W[n,:]) is two contiguous runs — the best layout
 * for both the CPU blocked kernel and a future GPU kernel):
 *
 *   offsets from the start of qw_q8_block:
 *     [0 .. N*K)            int8  w_q[n*K + k]
 *   + alignment pad to 16   (qw_q8_block_nbytes accounts for it)
 *     float32 scale[n]      one per output channel n
 *     float32 (none)        symmetric => no bias
 *     int32   zp[n]         zero point; 0 for symmetric W8A8, kept for a
 *                           future asymmetric W4A1 path
 *
 * scale[n] maps int8 -> fp32:  w_fp32[n][k] = (w_q - zp[n]) * scale[n].
 * For the symmetric case zp[n] == 0 always.
 *
 * The block owns its own heap allocation (made by qw_q8_pack or
 * qw_q8_block_alloc); free it with qw_q8_block_free().
 */
typedef struct qw_q8_block {
    int      n;   /* output channels (rows of W) */
    int      k;   /* reduction dim (columns) */
    int8_t  *w_q; /* [N][K] int8, row-major; NULL if not allocated */
    float   *scale; /* [N] float32 per-channel scale */
    int32_t *zp;   /* [N] int32 zero point (0 for symmetric) */
} qw_q8_block;

/* Per-token (per-row) int8 activation: one float32 scale per row of K. */
typedef struct qw_act_q8 {
    int      m;   /* rows (tokens) */
    int      k;   /* columns */
    int8_t  *q;   /* [M][K] int8, row-major */
    float   *scale; /* [M] float32 per-row scale */
} qw_act_q8;

/* ------------------------------------------------------------------ alloc */

/* Allocate the arrays of a block (caller supplies n,k; must be > 0).
 * Initializes w_q to 0, scale to 1, zp to 0. QW_ERR_ALLOC on OOM,
 * QW_ERR_NULL on NULL out, QW_ERR_RANGE on n<=0 || k<=0. */
qw_err qw_q8_block_alloc(qw_q8_block *out, int n, int k);

/* Free a block's arrays (NULL-safe). */
void qw_q8_block_free(qw_q8_block *b);

/* Allocate activation arrays. Same error contract as qw_q8_block_alloc. */
qw_err qw_act_q8_alloc(qw_act_q8 *out, int m, int k);

void qw_act_q8_free(qw_act_q8 *a); /* NULL-safe */

/* Total bytes held by a block (w_q + scale + zp + alignment pad).
 * 0 if the block is not allocated. */
size_t qw_q8_block_nbytes(const qw_q8_block *b);

/* ------------------------------------------------------------- pack/pack */

/* Quantize an fp32 row-major weight matrix [N][K] into a symmetric
 * per-channel int8 block. For each row n:
 *   amax = max_k |src[n*K + k]|
 *   scale[n] = amax / 127.0f   (0 if amax == 0; dequant then yields 0)
 *   w_q[n*K + k] = clamp(round(src/scale), -127, 127)
 * zp[n] = 0.
 *
 * src must be a valid pointer to N*K floats. QW_ERR_NULL on NULL src/out,
 * QW_ERR_RANGE on n<=0 || k<=0, QW_ERR_ALLOC on OOM.
 */
qw_err qw_q8_pack(const float *src, int n, int k, qw_q8_block *out);

/* Dequantize one row n of a block back to fp32 (K floats written to dst).
 * dst must hold k floats. QW_ERR_NULL on bad args, QW_ERR_RANGE if n out of
 * [0, block->n). */
qw_err qw_q8_dequant_row(const qw_q8_block *b, int n, float *dst);

/* ------------------------------------------------------ activation quant */

/* Per-row (per-token) symmetric int8 activation quantization.
 *
 *   amax = max_k |x[k]|
 *   If amax == 0: *scale_out = 1.0f and all q = 0 (all-zero row guard;
 *     documented: a zero activation row must dequantize to exactly zero,
 *     and 0 * 1.0 == 0, so scale=1.0 is the neutral choice).
 *   Else: *scale_out = amax / 127.0f
 *         q[k] = clamp(lrintf(x[k] / scale), -127, 127)
 *
 * Rounding is lrintf() — IEEE round-half-to-even, the documented, defined
 * behavior (unlike (int)float casts which truncate toward zero and are only
 * "fine" because we clamp). Clamping to [-127,127] never relies on
 * implementation-defined float->int conversion: we compute a float quotient,
 * round with lrintf (returns int, in range after clamp), then clamp the int.
 *
 * QW_ERR_NULL on NULL x/q/scale_out, QW_ERR_RANGE on k<=0.
 */
qw_err qw_act_q8_row(const float *x, int k, int8_t *q, float *scale_out);

/* ----------------------------------------------------------- overflow chk */

/* Worst-case int32 accumulator check. The int8 GEMM accumulates
 *   C[m][n] = sum_k A_q8[m][k] * W_q8[n][k]
 * in int32. Each product is at most 127*127 = 16129 in magnitude, so the
 * worst-case |sum| over K terms is 16129*K. This returns true iff that fits
 * INT32_MAX, i.e. K <= INT32_MAX/16129 = 133144. For the real model K is
 * 2560 (hidden) or 640 (moe_intermediate) — both far inside. Callers that
 * use a naive int32 accumulator should assert this for their K. */
bool qw_q8_overflow_ok(int m, int n, int k);

/* ------------------------------------------------------------- GEMM ref */

/* NAIVE reference int8 GEMM — the correctness oracle.
 *
 *   For every m in [0,M), n in [0,N):
 *     acc64 = 0 (int64 locally, to prove no int32 overflow on the way)
 *     for t in [0,K): acc64 += (int64)A_q8[i*K+t] * (int64)W_q8[j*K+t]
 *     C_int32[i*N+j] = (int32)acc64    ; checked to fit by caller
 *     out[i*N+j]     = (float)acc64 * a_scale[i] * w_scale[j]   (dequant)
 *
 * The MEANINGFUL equality across kernels is the int32 accumulator (see the
 * raw qw_q8_gemm_int32_* variants). The float dequant of the same acc may
 * differ from another kernel's by up to ~1-2 ulps because (float)acc is a
 * double rounding (acc can exceed 2^24) and FMA contraction is a
 * compilation choice — but never more than a couple of ulps.
 *
 * Written for clarity, not speed. A_q8 is [M][K] row-major, W_q8 is [N][K]
 * row-major (qw_q8_block layout), a_scale is [M] (per-row activation),
 * w_scale is [N] (per-channel weight), out is [M][N] row-major fp32.
 *
 * QW_ERR_NULL on NULL pointers, QW_ERR_RANGE on M/N/K <= 0,
 * QW_ERR_OVERFLOW if any int32 accumulator would not fit (should not happen
 * when qw_q8_overflow_ok(M,N,K) is true).
 */
qw_err qw_q8_gemm_ref(const int8_t *a_q8, const float *a_scale,
                      const int8_t *w_q8, const float *w_scale,
                      int m, int n, int k, float *out);

/* Plain fp32-input, fp32-accumulate GEMM: C[M][N] = A[M][K] * W[N][K] with
 * fp32 a/w row-major and fp32 out. This is the numerically honest stand-in
 * for an fp16-input GEMM with fp32 accumulation (fp16 rounding of the inputs
 * is the only thing it omits), and serves as the accuracy reference for
 * qw_q8_compare_fp16() and the fp32 baseline in tools/bench_int8.c.
 *
 * QW_ERR_NULL on NULL pointers, QW_ERR_RANGE on M/N/K <= 0.
 */
qw_err qw_q8_gemm_fp32(const float *a, const float *w,
                       int m, int n, int k, float *out);

/* Raw int32-accumulator int8 GEMM (no dequant): out[i*N+j] = the int32 sum
 *   sum_k A_q8[i*K+k] * W_q8[n*K+k]  (int32 accumulator, safe when
 *   qw_q8_overflow_ok(M,N,K) is true).
 * Used to PROVE bit-identical int32 accumulators between the naive and
 * blocked kernels (float dequant of the same acc can differ in the last ULP
 * across loop schedules due to FP contraction, so the int32 accumulator is
 * the meaningful equality). QW_ERR_NULL / QW_ERR_RANGE as above.
 */
qw_err qw_q8_gemm_int32_naive(const int8_t *a_q8, const int8_t *w_q8,
                              int m, int n, int k, int32_t *out);

qw_err qw_q8_gemm_int32_blocked(const int8_t *a_q8, const int8_t *w_q8,
                                int m, int n, int k, int32_t *out);

/* Vectorized fp32 GEMM: the fp32-accumulate baseline for the benchmark.
 * K-innermost with 4-way N unrolling (the same shape that lets the int8
 * kernel vectorize), so it is a fair, fast stand-in for an fp16-input GEMM
 * with fp32 accumulation. This is what tools/bench_int8.c reports as the
 * "fp32" column (the fp16-accumulate reference). QW_ERR_NULL / QW_ERR_RANGE
 * as above.
 */
qw_err qw_q8_gemm_fp32_vec(const float *a, const float *w,
                           int m, int n, int k, float *out);

/* NAIVE int8 GEMM with int32 accumulators (NOT the int64 oracle above): the
 * unoptimized baseline used by tools/bench_int8.c to measure how much the
 * blocked kernel gains. Same loop order as qw_q8_gemm_ref but with an int32
 * accumulator (safe when qw_q8_overflow_ok(M,N,K) is true). Same
 * inputs/outputs/error contract as qw_q8_gemm_ref.
 */
qw_err qw_q8_gemm_naive(const int8_t *a_q8, const float *a_scale,
                        const int8_t *w_q8, const float *w_scale,
                        int m, int n, int k, float *out);

/* Cache-blocked, register-blocked (8x4) int8 GEMM, int32 accumulators,
 * dequant fused in the epilogue. Same inputs/outputs as qw_q8_gemm_ref.
 *
 * Loop transformations (see src/core/quant.c for the annotated code):
 *   1. K-loop outermost, then N-block, then M-block (A-block reused across
 *      the whole K strip; W-block streamed once per K strip).
 *   2. Register tiles of BR=8 (M) x BN=4 (N); int32 accumulators held in
 *      registers across the K inner loop.
 *   3. Dequant (multiply by a_scale[m]*w_scale[n]) done once per output
 *      element in the epilogue, not inside the K loop.
 *
 * The point of this kernel: give the AMDGPU/LLVM backend the best possible
 * chance to auto-vectorize the K inner loop into v_dot-class integer dot
 * products. On gfx1030 there is no tensor-core-style tile instruction (no
 * WMMA, no MFMA, no matrix registers), so this is the whole story: integer
 * dot throughput on the ALU, not a matmul unit.
 *
 * QW_ERR_NULL on NULL pointers, QW_ERR_RANGE on M/N/K <= 0.
 */
qw_err qw_q8_gemm_blocked(const int8_t *a_q8, const float *a_scale,
                          const int8_t *w_q8, const float *w_scale,
                          int m, int n, int k, float *out);

/* ------------------------------------------------------- accuracy helpers */

/* Mean squared error between two [n]-element fp32 vectors. */
double qw_q8_err_mse(const float *a, const float *b, int n);

/* Cosine similarity of two [n]-element fp32 vectors (0..1 for aligned). */
double qw_q8_cosine_sim(const float *a, const float *b, int n);

/* Accuracy report for W8A8 on synthetic data.
 *
 * Generates a [N][K] fp32 weight matrix (seeded LCG, deterministic), a
 * [M][K] fp32 activation matrix, quantizes both to int8 (per-channel weight,
 * per-row activation), and compares the int8 GEMM result against a plain
 * fp32-accumulate GEMM (the "fp16-accumulate reference" stand-in: fp32
 * accumulation is the numerically honest reference for what an fp16 GEMM
 * with fp32 accumulation would produce).
 *
 * Prints MSE and cosine similarity to stderr and returns them via the
 * out params (may be NULL). QW_ERR_NULL if any non-NULL input is bad;
 * QW_ERR_RANGE on M/N/K <= 0.
 */
qw_err qw_q8_compare_fp16(int m, int n, int k, uint64_t seed,
                          double *mse_out, double *cos_out);

#ifdef __cplusplus
}
#endif

#endif /* QW_QUANT_H */
