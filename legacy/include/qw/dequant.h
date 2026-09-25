/* qw/dequant.h — CPU dequantization oracles + portable half/bfloat16 math.
 *
 * Purpose: these are the CPU correctness oracles for the GPU dequant kernels
 * that do not exist yet. Every function here is pure bit manipulation with
 * no __fp16, no vendor headers, no HIP: it must build and agree bit-for-bit
 * with plain `cc` on any host, and be the ground truth a future GPU kernel
 * is diffed against.
 *
 * Pure C17, libc only.
 *
 * The block byte sizes are the source of truth in include/qw/gguf.h
 * (gguf_type_block_info); the layouts decoded here mirror the GGML GGUF
 * v2/v3 on-disk block formats:
 *
 *   q8_0   32 elts, 48 bytes: { fp16 d; int8 qs[32] }          y[i] = d * qs[i]
 *   q4_0   32 elts, 32 bytes: { fp16 d; uint8 qh[16] }         y[i] = d * (nib[i] - 8)
 *   q4_1   32 elts, 48 bytes: { fp16 d; fp16 m; uint8 qh[16];
 *                               fp16 s[4] }                    y[i] = s[g] + (nib[i]-8) * d
 *                                                               (g = i / 8)
 *   f16 / f32: element passthrough.
 */
#ifndef QW_DEQUANT_H
#define QW_DEQUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/gguf.h"
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Elements per block for the block-quantized layouts. */
#define QW_DQ_BLOCK 32

/* fp16 special-value bit patterns (host bit order). */
#define QW_F16_POS_INF 0x7C00u
#define QW_F16_QNAN    0x7E00u

/* ------------------------------------------------------------ half math */

/* Convert an IEEE 754 binary16 bit pattern to float32. Handles subnormals,
 * both infinities and both NaN payloads, exactly. */
float qw_f16_to_f32(uint16_t h);

/* Convert float32 to an IEEE 754 binary16 bit pattern with
 * round-to-nearest-even (ties to even mantissa), matching the GPU cvt
 * instruction the dequant kernels will emit. Overflow clamps to ±inf
 * (the defined saturation behavior); underflow goes to subnormals, then to
 * zero. The input NaN becomes QW_F16_QNAN. */
uint16_t qw_f32_to_f16(float f);

/* bfloat16 (8 exponent bits, 7 mantissa bits, same sign/exponent layout as
 * fp32): exact widening, and RNE narrowing with overflow to ±inf. RDNA2 has
 * no bf16 unit so the engine avoids it, but the loader can still meet bf16
 * tensors in GGUF files and must not corrupt them. */
float    qw_bf16_to_f32(uint16_t b);
uint16_t qw_f32_to_bf16(float f);

/* ------------------------------------------------------------- dequant */

/* Dequantize one row of a block-quantized row-major tensor.
 *
 * `block` points at the first block of the row, `nblk` blocks are decoded,
 * `out` must hold nblk * QW_DQ_BLOCK floats. Row length n = nblk * 32; for
 * the quant types a row that is not a multiple of 32 is rejected
 * (QW_ERR_RANGE) rather than read past the row.
 *
 * Returns QW_OK, or QW_ERR_NULL / QW_ERR_RANGE / QW_ERR_UNSUPPORTED.
 */
qw_err qw_dequant_q8_0_row(const void *block, float *out, int nblk);
qw_err qw_dequant_q4_0_row(const void *block, float *out, int nblk);
qw_err qw_dequant_q4_1_row(const void *block, float *out, int nblk);
qw_err qw_dequant_f16_row(const void *block, float *out, int n);
qw_err qw_dequant_f32_row(const void *block, float *out, int n);

/* Same dequant, but emitting IEEE 754 binary16 bit patterns so the caller
 * can feed a fp16 GEMV directly (out must hold nblk * QW_DQ_BLOCK
 * uint16_t, or n for the passthroughs). */
qw_err qw_dequant_q8_0_row_to_f16(const void *block, uint16_t *out, int nblk);
qw_err qw_dequant_q4_0_row_to_f16(const void *block, uint16_t *out, int nblk);
qw_err qw_dequant_q4_1_row_to_f16(const void *block, uint16_t *out, int nblk);
qw_err qw_dequant_f32_row_to_f16(const void *block, uint16_t *out, int n);

/* Storage bytes for a row of n_elems elements in a given on-disk type.
 * For the block types, n_elems must be a positive multiple of 32, else the
 * function returns 0 (an error, never a partial size). Overflow-checked;
 * returns 0 on overflow or for any unsupported type. */
size_t qw_dequant_bytes_for(size_t n_elems, gguf_tensor_type t);

#ifdef __cplusplus
}
#endif

#endif /* QW_DEQUANT_H */
