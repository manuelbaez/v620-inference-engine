/* src/core/dequant.c — CPU dequant oracles + portable half/bf16 conversion.
 *
 * See include/qw/dequant.h for purpose and on-disk layouts. Pure C17, libc
 * only: the fp16/bf16 conversion is bit manipulation, not __fp16, so it runs
 * on any host and is the ground truth for the (future) GPU dequant kernels.
 */
#include "qw/dequant.h"
#include "qw/gguf.h"
#include "qw/macros.h"

#include <math.h>
#include <string.h>

/* ============================================================ half math */

/* IEEE 754 binary16 ("half"): 1 sign bit, 5 exponent bits, 10 mantissa bits.
 *
 *   bit 15        sign
 *   bits 14..10   exponent, bias 15 (field = exponent + 15)
 *   bits  9..0    mantissa, implicit leading 1 when 0 < exp field < 30
 *
 * Specials, in the exponent field (ef = h >> 10 & 0x1f):
 *   ef == 0   subnormal, value = +/- 2^-14 * (m / 2^10)
 *   ef == 31  mantissa 0 -> +/-inf, mantissa != 0 -> NaN
 *
 * fp32 has the same shape with a 23-bit mantissa and bias 127. Converting
 * half -> float just re-biases the exponent: E_f32 = (ef - 15) + 127.
 * Converting the other way, RNE half -> float rounds the 13-bit
 * (1 + 10-bit mantissa) value into 24 bits:
 *   13 bits + (15 - 127 + 1) = -100-bit shift... concretely below.
 */

/* h2f: exact widening of binary16 -> binary32. */
float qw_f16_to_f32(uint16_t h)
{
    uint32_t sign   = (uint32_t)(h >> 15) & 1u;
    uint32_t ef     = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant   = (uint32_t)(h & 0x3ffu);
    uint32_t f_exp, f_mant;

    if (ef == 0) {
        if (mant == 0) {
            f_exp = 0;                      /* ±0: exponent field 0 */
            f_mant = 0;
        } else {
            /* Subnormal: value = mant * 2^-24 (= 2^-14 * mant/2^10).
             * Normalize: left-shift mant until its top set bit reaches bit 9
             * (the top of the 10-bit field); `lead` = that shift count. The
             * top bit of mant sits at position (9 - lead), so
             *   value = 2^(9-lead) * 2^-24 * 1.frac = 2^(-15-lead) * 1.frac,
             * and with the fp32 field F defined by value = 2^(F-127) * 1.frac:
             *   f_exp = F = -15 - lead + 127 = 112 - lead.
             * Checked: mant=1 -> lead=9 -> F=103 -> 2^-24 (exact).
             *           mant=0x200 -> lead=0 -> F=112 -> 2^-15 (exact).
             * The 9 fraction bits (m & 0x1ff) left-align into fp32's 23-bit
             * mantissa at bits 22..14 (shift 14). */
            int lead = 0;
            uint32_t m = mant;
            while (!(m & 0x200u)) { m <<= 1; lead++; }
            f_mant   = (m & 0x1ffu) << 14;
            f_exp    = (uint32_t)(112 - lead);
        }
    } else if (ef == 0x1fu) {
        /* Inf or NaN: copy the field shape across (127 - 15 = 112 bias
         * difference). */
        f_exp  = 0xffu;
        f_mant = mant << 13;
    } else {
        /* Normal: E_f32 = (ef - 15) + 127 = ef + 112; mantissa left-shifts
         * from 10 to 23 bits. */
        f_exp  = ef + 112u;
        f_mant = mant << 13;
    }
    uint32_t out = (sign << 31) | (f_exp << 23) | f_mant;
    float r;
    memcpy(&r, &out, sizeof(r));
    return r;
}

/* f2h: narrowing binary32 -> binary16 with round-to-nearest-even.
 *
 * Exponent field arithmetic: the fp32 value's exponent is E = ef32 - 127.
 * The target field is ef16 = E + 15 = ef32 - 112, so the fp32 exponent
 * field shifts down by 112. The 24-bit source mantissa (implicit 1 + 23
 * stored bits) must fit in 10 stored bits: we keep the top 10 bits and
 * round the dropped 13 bits (1 guard + 10 round/sticky bits, sticky
 * accumulated as OR of all discarded bits).
 *
 * RNE: add 1 to the 10-bit field when (guard==1 and
 * (round|sticky|lsb)) — i.e. strictly more than half, or exactly half with
 * an odd low bit. A carry out of the field increments the exponent field,
 * which must then be re-clamped against the inf boundary (overflow).
 */
uint16_t qw_f32_to_f16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 1u;
    uint32_t ef   = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;

    if (ef == 0xffu) {
        /* Inf stays inf; NaN (any payload) becomes the canonical QNaN. */
        return (uint16_t)((sign << 15) | (mant ? QW_F16_QNAN : 0x7c00u));
    }
    /* Unified narrowing. The fp32 value is x = s * 2^(ef - 127), where s is
     * the 24-bit significand (1.mant if ef > 0, 0.mant if ef == 0).
     *
     * Exponent-field arithmetic: the fp16 field is ef16 = (ef - 127) + 15
     * = ef - 112, i.e. the fp32 field shifts down by 112 (the 127 -> 15
     * bias difference). The 24-bit source significand must fit in the 10
     * stored fp16 bits, so we round the significand to (24 - 13) = 11 bits:
     * keep the top 11, round the 13 dropped bits with RNE (1 guard bit +
     * 12 round/sticky bits, sticky = OR of all dropped bits — exact since
     * 12 < 32).
     *
     * RNE rule: round up when the dropped bits are STRICTLY more than half
     * the spacing (guard=1, sticky=1), or EXACTLY half (guard=1, sticky=0)
     * with an odd retained low bit (ties-to-even):
     *   round = guard && (sticky || lsb).
     *
     * Cases by ef:
     *   ef >= 113:      normal output, field ef - 112 in [1, 30]; a carry
     *                   out of the mantissa field bumps the field, which can
     *                   reach 31 -> saturate to ±inf.
     *   103 <= ef<=112: subnormal output on the 2^-24 grid: in grid units
     *                   x = sig * 2^(ef - 103), so shift right by
     *                   (103 - ef) and apply the same RNE rule. At ef == 112
     *                   the rounded grid index can reach 0x400, which is
     *                   exactly the smallest fp16 normal.
     *   ef <= 102:      x < half a grid step of 2^-24 -> ±0.
     */
    uint32_t sig    = (ef > 0) ? ((1u << 23) | mant) : mant;
    uint32_t shift  = 13u;                       /* drop low 13 bits */
    uint32_t lsb    = (sig >> shift) & 1u;       /* retained low bit      */
    uint32_t guard  = (sig >> (shift - 1u)) & 1u;/* half-spacing bit      */
    /* sticky = OR of the bits BELOW the guard bit (the true round/sticky
     * bits). Including the guard bit here would make exact halves look
     * like more-than-halves and break ties-to-even. */
    uint32_t sticky = sig & ((1u << (shift - 1u)) - 2u);
    uint32_t round  = guard && (sticky || lsb);

    uint16_t out16;
    if (ef >= 113u) {
        /* Normal output: ef16 = ef - 112 in [1, 30] (ef == 255 handled
         * above). */
        out16 = (uint16_t)(((sign << 15) | (uint16_t)((ef - 112u) << 10))
                           | (uint16_t)((sig >> shift) & 0x3ffu));
        if (round) {
            uint16_t lo = out16 & 0x03ffu;
            if (++lo == 0x400u)
                out16 = (uint16_t)((out16 & 0xFC00u) + 0x400u); /* exp carry */
            else
                out16 = (uint16_t)((out16 & 0xFC00u) | lo);
        }
    } else {
        /* Subnormal-or-zero output. The fp16 subnormal grid has step 2^-24,
         * so the grid index is h = x / 2^-24. The fp32 significand is
         * sig * 2^(ef-127); in grid units that is sig * 2^(ef-103). Since
         * sig is 24 bits with its implicit 1 in bit 23, right-shifting by
         * (126 - ef) aligns that 1 with the 11-bit grid field (bit 10):
         *   ef = 103 (x = 2^-24):  shift 23 -> 1
         *   ef = 112 (x ~ 2^-14):  shift 14 -> 0x3ff or 0x400
         * 0x400 is exactly the smallest fp16 normal (field 1, mantissa 0),
         * so RNE at the normal/subnormal boundary promotes via the mask.
         * For ef <= 102 the value is under half a grid step -> ±0. */
        if (ef <= 102u)
            out16 = (uint16_t)(sign << 15);
        else {
            shift = 126u - ef;                    /* in [14, 23] */
            lsb    = (sig >> shift) & 1u;
            guard  = (sig >> (shift - 1u)) & 1u;
            /* sticky = OR of bits below the guard (shift >= 2 always here). */
            sticky = sig & ((1u << (shift - 1u)) - 2u);
            round  = guard && (sticky || lsb);
            uint32_t h = (sig >> shift) + round;  /* h in [1, 0x400] */
            out16 = (uint16_t)((sign << 15) | (h & 0x3ffu));
        }
    }
    if ((out16 & 0x7C00u) == 0x7C00u)
        return (uint16_t)((sign << 15) | 0x7C00u);  /* overflow: ±inf */
    return out16;
}

/* bfloat16: same 1+8+7 shape as fp32 (it is literally the top half of an
 * fp32). Widening is exact: low 16 bits zero. Narrowing keeps the top 16
 * bits and RNE-rounds the dropped 16. */
float qw_bf16_to_f32(uint16_t b)
{
    uint32_t bits = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t qw_f32_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 1u;
    uint32_t ef   = (bits >> 23) & 0xffu;

    if (ef == 0xffu)
        return (uint16_t)((sign << 15) | ((bits >> 16) & 0x7fffu));

    uint32_t rem = bits & 0xffffu;                 /* dropped 16 bits */
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounded = (bits >> 16) +
                       (((rem & 0x8000u) != 0) && ((rem & 0x7fffu) != 0 || lsb != 0));
    /* A carry out of the 16-bit field raises the exponent by one, which can
     * reach 0xff (overflow to inf) — the top half already contains the
     * incremented exponent because the field is contiguous with it. */
    if (rounded > 0xffffu)
        return (uint16_t)((sign << 15) | 0x7f80u); /* overflow: ±inf */
    return (uint16_t)rounded;
}

/* =============================================================== dequant */

/* Read a little-endian uint16 from an arbitrary (unaligned) address.
 * GGUF is little-endian; the engine targets LE hosts and the GPU kernel
 * reads the same bytes, so matching the file byte order — not the host
 * scalar type — is what makes this the correct oracle. */
static uint16_t rd16(const void *p)
{
    const unsigned char *b = (const unsigned char *)p;
    return (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
}

static float rd32f(const void *p)
{
    const unsigned char *b = (const unsigned char *)p;
    uint32_t u = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
                 ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* q8_0: { fp16 d; int8 qs[32] }, 48 bytes, y[i] = d * qs[i]. */
qw_err qw_dequant_q8_0_row(const void *block, float *out, int nblk)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (nblk <= 0)
        return QW_ERR_RANGE;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = (const uint8_t *)block + (size_t)b * 48;
        float d = qw_f16_to_f32(rd16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2);
        float *o = out + (size_t)b * QW_DQ_BLOCK;
        for (int i = 0; i < QW_DQ_BLOCK; i++)
            o[i] = d * (float)qs[i];
    }
    return QW_OK;
}

/* q4_0: { fp16 d; uint8 qh[16] }, 32 bytes, y[i] = d * (nib[i] - 8).
 *
 * Nibble order (the convention the future GPU kernel must match): element i
 * lives in byte i/2; even i takes the LOW nibble, odd i the HIGH nibble:
 *   nib[i] = (i & 1) ? qh[i/2] >> 4 : qh[i/2] & 0x0f
 * This is the GGML ggml_dequantize_row_q4_0 convention (qh = "quants high",
 * the packed 4-bit half-byte array). */
qw_err qw_dequant_q4_0_row(const void *block, float *out, int nblk)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (nblk <= 0)
        return QW_ERR_RANGE;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = (const uint8_t *)block + (size_t)b * 32;
        float d = qw_f16_to_f32(rd16(blk));
        const uint8_t *qh = blk + 2;
        float *o = out + (size_t)b * QW_DQ_BLOCK;
        for (int i = 0; i < QW_DQ_BLOCK; i += 2) {
            uint8_t q = qh[i / 2];
            o[i]     = d * ((float)(q & 0x0fu) - 8.0f);
            o[i + 1] = d * ((float)(q >> 4)    - 8.0f);
        }
    }
    return QW_OK;
}

/* q4_1: { fp16 d; fp16 m; uint8 qh[16]; fp16 s[4] }, 48 bytes,
 * y[i] = s[i/8] + (nib[i] - 8) * d, same nibble order as q4_0. */
qw_err qw_dequant_q4_1_row(const void *block, float *out, int nblk)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (nblk <= 0)
        return QW_ERR_RANGE;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = (const uint8_t *)block + (size_t)b * 48;
        float d = qw_f16_to_f32(rd16(blk));
        float m = qw_f16_to_f32(rd16(blk + 2));
        (void)m; /* min: not part of the dequant math, decoded for layout */
        const uint8_t *qh = blk + 4;
        const uint8_t *sp = blk + 20;
        float *o = out + (size_t)b * QW_DQ_BLOCK;
        for (int g = 0; g < 4; g++) {
            float s = qw_f16_to_f32(rd16(sp + (size_t)g * 2));
            for (int i = 0; i < 8; i++) {
                uint8_t q = qh[g * 2 + i / 2];
                uint32_t nib = (i & 1) ? (q >> 4) : (q & 0x0fu);
                o[g * 8 + i] = s + (float)(nib - 8u) * d;
            }
        }
    }
    return QW_OK;
}

/* f16 passthrough: n fp16 elements -> n floats. */
qw_err qw_dequant_f16_row(const void *block, float *out, int n)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (n < 0)
        return QW_ERR_RANGE;
    const uint8_t *blk = (const uint8_t *)block;
    for (int i = 0; i < n; i++)
        out[i] = qw_f16_to_f32(rd16(blk + (size_t)i * 2));
    return QW_OK;
}

/* f32 passthrough: n fp32 elements -> n floats (copy, endianness-matched). */
qw_err qw_dequant_f32_row(const void *block, float *out, int n)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (n < 0)
        return QW_ERR_RANGE;
    const uint8_t *blk = (const uint8_t *)block;
    for (int i = 0; i < n; i++)
        out[i] = rd32f(blk + (size_t)i * 4);
    return QW_OK;
}

/* fp16-emitting variants: same math, qw_f32_to_f16() at the end so the
 * caller can feed a fp16 GEMV. */
qw_err qw_dequant_q8_0_row_to_f16(const void *block, uint16_t *out, int nblk)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (nblk <= 0)
        return QW_ERR_RANGE;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = (const uint8_t *)block + (size_t)b * 48;
        float d = qw_f16_to_f32(rd16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2);
        uint16_t *o = out + (size_t)b * QW_DQ_BLOCK;
        for (int i = 0; i < QW_DQ_BLOCK; i++)
            o[i] = qw_f32_to_f16(d * (float)qs[i]);
    }
    return QW_OK;
}

qw_err qw_dequant_q4_0_row_to_f16(const void *block, uint16_t *out, int nblk)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (nblk <= 0)
        return QW_ERR_RANGE;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = (const uint8_t *)block + (size_t)b * 32;
        float d = qw_f16_to_f32(rd16(blk));
        const uint8_t *qh = blk + 2;
        uint16_t *o = out + (size_t)b * QW_DQ_BLOCK;
        for (int i = 0; i < QW_DQ_BLOCK; i += 2) {
            uint8_t q = qh[i / 2];
            o[i]     = qw_f32_to_f16(d * ((float)(q & 0x0fu) - 8.0f));
            o[i + 1] = qw_f32_to_f16(d * ((float)(q >> 4)    - 8.0f));
        }
    }
    return QW_OK;
}

qw_err qw_dequant_q4_1_row_to_f16(const void *block, uint16_t *out, int nblk)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (nblk <= 0)
        return QW_ERR_RANGE;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = (const uint8_t *)block + (size_t)b * 48;
        float d = qw_f16_to_f32(rd16(blk));
        const uint8_t *qh = blk + 4;
        const uint8_t *sp = blk + 20;
        uint16_t *o = out + (size_t)b * QW_DQ_BLOCK;
        for (int g = 0; g < 4; g++) {
            float s = qw_f16_to_f32(rd16(sp + (size_t)g * 2));
            for (int i = 0; i < 8; i++) {
                uint8_t q = qh[g * 2 + i / 2];
                uint32_t nib = (i & 1) ? (q >> 4) : (q & 0x0fu);
                o[g * 8 + i] = qw_f32_to_f16(s + (float)(nib - 8u) * d);
            }
        }
    }
    return QW_OK;
}

/* f32 -> f16 passthrough (the other direction is the fp16 GEMV input). */
qw_err qw_dequant_f32_row_to_f16(const void *block, uint16_t *out, int n)
{
    if (block == NULL || out == NULL)
        return QW_ERR_NULL;
    if (n < 0)
        return QW_ERR_RANGE;
    const uint8_t *blk = (const uint8_t *)block;
    for (int i = 0; i < n; i++)
        out[i] = qw_f32_to_f16(rd32f(blk + (size_t)i * 4));
    return QW_OK;
}

/* Storage bytes for a row. Block sizes / row granularity come from the
 * gguf_type_block_info() table (the source of truth), so this can never
 * drift from what the loader reports. */
size_t qw_dequant_bytes_for(size_t n_elems, gguf_tensor_type t)
{
    uint32_t bs = 0, row = 0;
    if (!gguf_type_block_info(t, &bs, &row))
        return 0;

    size_t nblk;
    if (row == 1)
        nblk = n_elems;
    else {
        if (n_elems == 0 || (n_elems % row) != 0)
            return 0;                       /* not block-aligned: error */
        nblk = n_elems / (size_t)row;
    }

    /* Overflow-checked: nblk * bs must fit in size_t. */
    if (bs != 0 && nblk > SIZE_MAX / (size_t)bs)
        return 0;
    return nblk * (size_t)bs;
}
