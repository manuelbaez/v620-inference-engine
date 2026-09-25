/* qw/quant_int4.h — symmetric int4 weight block (W4A8) for gfx1030.
 *
 * Pure C17 CPU reference layer, same style and error contract as
 * qw/quant.h (int8 W8A8). No HIP, no external deps: libc only.
 *
 * Why int4 needs no dequantize-to-fp16 on the dot path:
 *   gfx1030 (RDNA2) has v_dot8_i32_i4 — 8 x int4 -> int32 dot-accumulate per
 *   instruction (LLVM 24 AMDGPU user guide). llama.cpp uses exactly this
 *   integer dp path on RDNA2 for its Q4_0/Q8_0 kernels (no float
 *   upconversion). So the int4 weight footprint stays 4 bits AND the dot runs
 *   at integer throughput.
 *
 * NOTE on the hardware:
 *   - v_dot8_i32_i4 on gfx1030 is UNSIGNED: its lanes are [0..15]. Our
 *     symmetric weights live in [-8,7]. The kernel therefore re-biases each
 *     nibble to [0,15] in-register and subtracts a per-row, per-strip bias
 *     correction from the int32 accumulator in the epilogue:
 *         sum a*(w+bias) = sum a*w + bias*sum(a)
 *     Both the widened-register kernel and the naive oracle do EXACTLY the
 *     same integer arithmetic, so their int32 accumulators are bit-identical.
 *   - gfx1030 does NOT have __builtin_amdgcn_wmma_* or
 *     __builtin_amdgcn_sudot4 (those are gfx11+); the plain v_dot spellings
 *     below are the gfx10-specific ISA. RDNA3 renamed the int8 dot to
 *     v_dot4_i32_iu8 — this code targets gfx1030 only.
 *
 * Implementations live in src/core/quant.c (additive section). Every
 * builtin reference is gated `#if defined(QW_ENABLE_DOT) &&
 * defined(__AMDGCN__)` exactly like qw_sdot4 in quant.h/quant.c, so a
 * non-GPU build never sees an unknown builtin.
 */
#ifndef QW_QUANT_INT4_H
#define QW_QUANT_INT4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ types */

/* Symmetric per-output-channel (per row) int4 weight block.
 *
 * Packing convention (FIXED; tests pin it with a hand-built vector so it
 * cannot silently change):
 *   - Two int4 weights per byte, little-nibble-first:
 *         w_packed[n*K/2 + i] = (w[2i] & 0x0F) | (w[2i+1] << 4)
 *     i.e. the LOW nibble of byte i holds the EVEN index (w[2i]); the HIGH
 *     nibble holds the odd index (w[2i+1]).
 *   - Each nibble holds a SIGNED weight in [-8, 7], stored two's-complement
 *     in 4 bits: 0x0=+0 ... 0x7=+7, 0x8=-8 ... 0xF=-1.
 *   - K must be even (two weights per byte); K is a multiple of 8 in the
 *     model (2560, 640).
 *
 * Memory layout (GEMM-friendly, mirroring qw_q8_block: W is [N][K]
 * row-major, each output channel a contiguous run of K/2 packed bytes):
 *
 *   offsets from the start of qw_q4_block:
 *     [0 .. N*K/2)         uint8 w_p[n*K/2 + i]  packed int4, 2 per byte
 *     float32 scale[n]     one per output channel n
 *
 * scale[n] maps int4 -> fp32:  w_fp32[n][k] = (int)w4[n][k] * scale[n],
 * where (int)w4 is the sign-extended 4-bit value. Symmetric => no zero
 * point.
 *
 * The block owns its own heap allocation (made by qw_q4_pack or
 * qw_q4_block_alloc); free it with qw_q4_block_free().
 */
typedef struct qw_q4_block {
    int      n;    /* output channels (rows of W) */
    int      k;    /* reduction dim (columns); must be even */
    uint8_t *w_p;  /* [N][K/2] packed int4 (2 per byte); NULL if unalloc'd */
    float   *scale; /* [N] float32 per-channel scale */
} qw_q4_block;

/* Per-token (per-row) int8 activation: one float32 scale per row of K.
 * Same layout and quantization as qw_act_q8 in qw/quant.h — int4 weights
 * mix with int8 activations (W4A8), so this reuses that exact definition. */
typedef struct qw_act_q8 qw_q4_act;

/* ------------------------------------------------------------------ alloc */

/* Allocate the arrays of a block (caller supplies n, k; must be > 0, k even).
 * Initializes w_p to 0, scale to 1. QW_ERR_ALLOC on OOM, QW_ERR_NULL on NULL
 * out, QW_ERR_RANGE on n<=0 || k<=0 || (k odd). */
qw_err qw_q4_block_alloc(qw_q4_block *out, int n, int k);

/* Free a block's arrays (NULL-safe). */
void qw_q4_block_free(qw_q4_block *b);

/* Total bytes held by a block (w_p + scale). 0 if not allocated. */
size_t qw_q4_block_nbytes(const qw_q4_block *b);

/* ------------------------------------------------------------- pack/unpack */

/* Quantize an fp32 row-major weight matrix [N][K] (k even) into a symmetric
 * per-channel int4 block. For each row n:
 *   amax   = max_k |src[n*K + k]|
 *   scale  = amax / 7.0f     (0 if amax == 0; dequant then yields 0)
 *   q[k]   = clamp(round(src/scale), -8, 7)   (round half away from zero)
 *   w_p[n*K/2 + i] = (q[2i] & 0x0F) | (q[2i+1] << 4)
 *
 * The signed q[k] is stored as a 4-bit two's-complement nibble, so the
 * packing is exact for q in [-8,7] (see the header packing convention).
 *
 * src must be a valid pointer to N*K floats. QW_ERR_NULL on NULL src/out,
 * QW_ERR_RANGE on n<=0 || k<=0 || (k odd), QW_ERR_ALLOC on OOM.
 */
qw_err qw_q4_pack(const float *src, int n, int k, qw_q4_block *out);

/* Round-trip helper for tests: extract row n's K signed int4 weights back
 * from the packed bytes (inverse of the packing convention). dst must hold k
 * ints. QW_ERR_NULL on bad args, QW_ERR_RANGE if n out of [0, block->n) or k
 * odd.
 */
qw_err qw_q4_unpack_row(const qw_q4_block *b, int n, int *dst);

/* ----------------------------------------------------------- overflow chk */

/* Worst-case int32 accumulator check for the W4A8 dot.
 *
 * The dot accumulates sum_k a[k]*w[k] with |a[k]| <= 127 (int8 activation)
 * and |w[k]| <= 8 (int4 weight), so each product is at most 8*127 = 1016 in
 * magnitude and the worst-case |sum| over K terms is 1016*K. This returns
 * true iff that fits INT32_MAX, i.e. K <= INT32_MAX/1016 = 2113665. The real
 * model uses K=2560 (hidden) and K=640 (moe_intermediate) — ~800x inside the
 * limit. Callers using a naive int32 accumulator should assert this for
 * their K (K=2560: 1016*2560 = 2600960 << INT32_MAX). */
bool qw_q4_overflow_ok(int k);

/* ------------------------------------------------------------- GEMM ref */

/* Reference W4A8 GEMM — the naive readability ORACLE.
 *
 *   For every m in [0,M), n in [0,N):
 *     acc = 0 (int32; safe when qw_q4_overflow_ok(k) is true)
 *     for t in [0,K):  acc += (int32)a_q8[m*K+t] * (int32)w4[n*K+t]
 *     out[m*N+n] = (float)acc * a_scale[m] * w_scale[n]   (dequant)
 *
 * w4 is the unpacked signed int4 weight row-major [N][K] (as produced by
 * qw_q4_unpack_row for each row; the bit-identical-accumulator proof
 * compares against the widened kernel that does the same integer math from
 * the packed form). Written for clarity, not speed: a_q8 is [M][K]
 * row-major int8, w4 is [N][K] row-major int32, a_scale is [M] (per-row
 * activation), w_scale is [N] (per-channel weight), out is [M][N] fp32.
 *
 * QW_ERR_NULL on NULL pointers, QW_ERR_RANGE on M/N/K <= 0 or k odd.
 */
qw_err qw_q4_gemm_ref(const int8_t *a_q8, const float *a_scale,
                      const int32_t *w4, const float *w_scale,
                      int m, int n, int k, float *out);

/* WIDENED-REGISTER GEMM: int4 weights x int8 activations, int32
 * accumulators, bit-identical to qw_q4_gemm_ref (same integer math).
 *
 * How it works (8 weights per strip):
 *   1. Unpack 8 packed int4 nibbles (4 bytes) to 8 int8 weights via shifts
 *      + masks + sign-extension, in registers.
 *   2. Re-bias each to u = w + 8 in [0,15] (gfx1030 v_dot8_i32_i4 is
 *      UNSIGNED — see the header note).
 *   3. Accumulate  acc = sum a[t]*u[t]  over the strip, with the 8x2 sdot4
 *      dot helper when the GPU builtin is available (below).
 *   4. Epilogue bias correction:  acc -= 8 * sum_a   (strip), where
 *      sum_a = sum_t a[t] — recovers exactly sum a[t]*w[t].
 *
 * WHY WIDEN-IN-REGISTER (vs requantizing the activation to 4 bits):
 *   gfx1030 has no signed int4 dot (v_dot8_i32_i4 is unsigned-only), so a
 *   "native" int4xint8 dot does not exist. The alternative is requantizing
 *   the int8 activation to 4 bits to match a 4-bit dot — that destroys
 *   activation accuracy (127 -> 15 levels, ~4x coarser) for no benefit.
 *   Widening the 8 nibbles to 8 int8s in registers costs a few shift/mask/
 *   ALU ops per strip but keeps the activation at full 8 bits.
 *   TRADEOFF vs a native udot8 path: a single v_dot8_i32_i4 does 8 MACs in
 *   one instruction; the widen path needs 2 sdot4 per 8 values (each sdot4
 *   covers 4 lanes) plus the unpack ALU and the bias correction. On a real
 *   kernel the unpack/bias runs once per 8-wide strip (amortized over the
 *   M dimension), so the widen path is the accuracy-safe choice and the
 *   udot8 spelling is what LLVM would use for the unsigned accumulation
 *   inside it.
 *
 * w_p is the packed qw_q4_block layout ([N][K/2] bytes), a_q8 is [M][K]
 * int8 row-major, out is [M][N] int32 raw accumulators (NO dequant — the
 * caller dequantizes with a_scale[m]*w_scale[n], exactly as the int8 path
 * exposes qw_q8_gemm_int32_*). K must be a multiple of 8 (the model dims
 * are). QW_ERR_NULL on NULL pointers, QW_ERR_RANGE on M/N/K <= 0 or k%8.
 */
qw_err qw_q4_gemm_widen(const int8_t *a_q8, const uint8_t *w_p,
                        int m, int n, int k, int32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* QW_QUANT_INT4_H */
