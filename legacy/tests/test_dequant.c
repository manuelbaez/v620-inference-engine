/* tests/test_dequant.c — no-framework asserts for the dequant oracles.
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -Iinclude \
 *        src/core/types.c src/loader/gguf.c src/core/dequant.c \
 *        tests/test_dequant.c -lm
 *
 * main() returns the count of failures (0 = all pass).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qw/dequant.h"
#include "qw/gguf.h"
#include "qw/types.h"

static int g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { printf("  ok   %s\n", #cond); } \
    else { printf("  FAIL %s (line %d)\n", #cond, __LINE__); g_fail++; } \
} while (0)

static uint64_t g_seed;
static uint64_t lcg(void)
{
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_seed >> 32;
}

/* fp16 special bit patterns. */
#define H_ZERO    0x0000u
#define H_NEGZERO 0x8000u
#define H_ONE     0x3C00u
#define H_NEGONE  0xBC00u
#define H_SMIN_N  0x0400u  /* 2^-14, smallest normal */
#define H_MAX     0x7BFFu  /* 65504, largest finite */
#define H_SMIN_S  0x0001u  /* 2^-24, smallest subnormal */
#define H_POSINF  0x7C00u
#define H_NEGINF  0xFC00u
#define H_QNAN    0x7E00u

/* fp32 bit patterns for exact literals. F_TWO_P24 is 2^-24 (the smallest
 * fp16 subnormal) as an fp32 normal: field 103, mantissa 0. */
#define F_TWO_P24       0x33800000u
#define F_NEGZERO       0x80000000u
#define F_INF           0x7F800000u
#define F_NAN           0x7FC00000u

static float bits_f32(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
static uint32_t f32_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}
static int f32_isnan(float f)
{
    return f32_bits(f) > 0x7F800000u;
}
static double fabsd(double x)
{
    return (x < 0.0) ? -x : x;
}

/* ------------------------------------------- test 1: fp16 special values */
static void test_f16_specials(void)
{
    printf("test 1: fp16 special-value round trip\n");

    /* 0.0 and -0.0 (sign preserved) */
    CHECK(qw_f32_to_f16(0.0f) == H_ZERO);
    CHECK(qw_f32_to_f16(-0.0f) == H_NEGZERO);
    CHECK(f32_bits(qw_f16_to_f32(H_ZERO)) == 0x00000000u);
    CHECK(f32_bits(qw_f16_to_f32(H_NEGZERO)) == F_NEGZERO);

    /* 1.0 and -1.0 (exactly representable: exp field 15) */
    CHECK(qw_f32_to_f16(1.0f) == H_ONE);
    CHECK(qw_f32_to_f16(-1.0f) == H_NEGONE);
    CHECK(f32_bits(qw_f16_to_f32(H_ONE)) == 0x3F800000u);
    CHECK(f32_bits(qw_f16_to_f32(H_NEGONE)) == 0xBF800000u);

    /* Smallest normal 2^-14: exp field 1, mantissa 0. */
    CHECK(qw_f32_to_f16(6.103515625e-05f) == H_SMIN_N);
    CHECK(qw_f16_to_f32(H_SMIN_N) == 6.103515625e-05f);

    /* Largest finite 65504 = 2^16: exp field 30, mantissa 0. */
    CHECK(qw_f32_to_f16(65504.0f) == H_MAX);
    CHECK(qw_f16_to_f32(H_MAX) == 65504.0f);

    /* Smallest fp16 subnormal 2^-24: narrows from the fp32 value 2^-24
     * (field 103) to 0x0001, and widens back to exactly 2^-24. */
    CHECK(qw_f32_to_f16(bits_f32(F_TWO_P24)) == H_SMIN_S);
    CHECK(f32_bits(qw_f16_to_f32(H_SMIN_S)) == F_TWO_P24);

    /* fp16 ULP near 1.0 is 2^-10 (10-bit mantissa). Round-to-nearest-even
     * around the 0x3C00 (1.0) / 0x3C01 (1+2^-10) boundary:
     *   1 + 2^-12 = quarter-ulp, closer to 1.0           -> 0x3C00
     *   1 + 2^-11 = exact midpoint (2^-11 = 2^-10/2),
     *     equidistant, ties to even mantissa 0            -> 0x3C00
     *   1 + 3*2^-12 = 0.75 ulp, closer to 1+2^-10        -> 0x3C01 */
    CHECK(qw_f32_to_f16(1.000244140625f) == H_ONE);   /* 1+2^-12, quarter ulp */
    CHECK(qw_f32_to_f16(1.00048828125f) == 0x3C00u);  /* 1+2^-11, tie->even */
    CHECK(qw_f32_to_f16(1.000732421875f) == 0x3C01u); /* 1+3*2^-12, rounds up */

    /* Infinities and NaN preservation. */
    CHECK(qw_f32_to_f16(bits_f32(F_INF)) == H_POSINF);
    CHECK(qw_f32_to_f16(-INFINITY) == H_NEGINF);
    CHECK(qw_f32_to_f16(NAN) == H_QNAN);
    CHECK(qw_f16_to_f32(H_POSINF) == INFINITY);
    CHECK(qw_f16_to_f32(H_NEGINF) == -INFINITY);
    CHECK(f32_isnan(qw_f16_to_f32(H_QNAN)));
}

/* ------------------------------------------- test 2: fp16 random round trip */
static void test_f16_random(void)
{
    printf("test 2: fp16 round trip, 10000 seeded randoms in [0,1)\n");

    g_seed = 0xDECAFBAD1234567ULL;
    double max_err = 0.0;
    for (int i = 0; i < 10000; i++) {
        float x = (float)((double)(lcg() & 0xffffff) / 16777216.0);
        float back = qw_f16_to_f32(qw_f32_to_f16(x));
        /* fp16 has a 10-bit mantissa: RNE guarantees |rel err| <= 2^-11 for
         * all normalized values. The sample itself is a float32 literal
         * (24-bit mantissa), so it sits at most 2^-24 above its true real
         * value; that adds a worst-case 2^-24 to the measured relative
         * error, hence the bound 2^-11 + 2^-24 below. */
        double bound = 0.00048828125 + 5.9604644775390625e-08;
        if (x != 0.0f) {
            double r = fabsd(back - (double)x) / (double)x;
            if (r > max_err)
                max_err = r;
            CHECK(r <= bound);
        }
    }
    printf("  max relative error: %.6e (bound 2^-11 = %.6e)\n",
           max_err, 0.00048828125);
    CHECK(max_err <= 0.00048828125); /* 2^-11, fp16 mantissa bound */
}

/* ------------------------------------------- test 3: bf16 round trip */
static void test_bf16(void)
{
    printf("test 3: bf16 special values + random round trip\n");

    /* bf16 has an 8-bit exponent like fp32, so 1.0 = 0x3F80 (not the fp16
     * 0x3C00). Widening is exact: the top 16 bits of the fp32. */
    CHECK(f32_bits(qw_bf16_to_f32(0x3F80u)) == 0x3F800000u); /* 1.0 */
    CHECK(f32_bits(qw_bf16_to_f32(0xBF80u)) == 0xBF800000u); /* -1.0 */
    CHECK(f32_bits(qw_bf16_to_f32(0x0001u)) == 0x00010000u); /* fp32 subnormal */

    /* 7-bit-mantissa RNE: bf16 ULP near 1.0 is 2^-7. 1+2^-8 is the exact
     * midpoint between 0x3F80 (1.0, mantissa 0, even) and 0x3F81 (1+2^-7,
     * mantissa 1, odd) -> ties to even -> 0x3F80. 1+3*2^-8 is 0.75 ulp,
     * closer to 0x3F81 -> rounds up. */
    CHECK(qw_f32_to_bf16(1.00390625f) == 0x3F80u);  /* 1+2^-8, tie->even */
    CHECK(qw_f32_to_bf16(1.005859375f) == 0x3F81u); /* 1+3*2^-8, rounds up */

    /* Wider exponent: 1e30 must survive (fp32 exp field ~210). */
    CHECK(qw_f32_to_bf16(1e30f) != 0x7F80u);
    /* 1e-30 is far below the fp16 subnormal range but a normal bf16 value. */
    CHECK(qw_f32_to_bf16(1e-30f) != 0u);
    CHECK(f32_isnan(qw_bf16_to_f32(0x7FC0u)));

    g_seed = 0xBEEF0042ULL;
    double max_err = 0.0;
    for (int i = 0; i < 10000; i++) {
        float x = (float)((double)(lcg() & 0xffffff) / 16777216.0);
        float back = qw_bf16_to_f32(qw_f32_to_bf16(x));
        /* bf16 has a 7-bit mantissa: RNE guarantees |rel err| <= 2^-8,
         * plus the same 2^-24 float-literal term as the fp16 test. */
        double bound = 0.00390625 + 5.9604644775390625e-08;
        if (x != 0.0f) {
            double r = fabsd(back - (double)x) / (double)x;
            if (r > max_err)
                max_err = r;
            CHECK(r <= bound);
        }
    }
    printf("  max relative error: %.6e (bound 2^-8 = %.6e)\n",
           max_err, 0.00390625);
    CHECK(max_err <= 0.00390625); /* 2^-8, bf16 mantissa bound */
}

/* ------------------------------------------- test 4: q8_0 hand-built block */
static void test_q8_0(void)
{
    printf("test 4: q8_0 decode of a hand-built block\n");

    /* d = 0.5 -> fp16 bits 0x3800 (2^-1, field 14). qs pattern covers all
     * 32 elements. */
    uint8_t blk[48];
    memset(blk, 0, sizeof(blk));
    blk[0] = 0x00; blk[1] = 0x38; /* d = 0.5, little-endian */
    static const int8_t qs[32] = {
        0, 1, -1, 2, -2, 3, -3, 4,
        5, -5, 6, -6, 7, -7, 8, -8,
        127, -128, 10, -10, 11, -11, 12, -12,
        100, -100, 126, -127, 0, 0, 0, 0
    };
    memcpy(blk + 2, qs, 32);

    float out[32];
    /* y[i] = 0.5f * qs[i]: exact for every int8 * 0.5 in this range (0.5
     * is a power of two, so the product is an exact fp32 scale). */
    static const float expected[32] = {
        0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 2.0f,
        2.5f, -2.5f, 3.0f, -3.0f, 3.5f, -3.5f, 4.0f, -4.0f,
        63.5f, -64.0f, 5.0f, -5.0f, 5.5f, -5.5f, 6.0f, -6.0f,
        50.0f, -50.0f, 63.0f, -63.5f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    CHECK(qw_dequant_q8_0_row(blk, out, 1) == QW_OK);
    for (int i = 0; i < 32; i++)
        CHECK(out[i] == expected[i]);

    /* fp16-emitting variant: exact values, no conversion loss (0.5 * small
     * int8 is always exactly representable in fp16). */
    uint16_t h[32];
    CHECK(qw_dequant_q8_0_row_to_f16(blk, h, 1) == QW_OK);
    for (int i = 0; i < 32; i++)
        CHECK(qw_f16_to_f32(h[i]) == expected[i]);

    /* Multi-block row: second block with d = -2.0 (fp16 0xC000) and all
     * qs = 1 => -2.0 every element. */
    uint8_t blk2[96];
    memcpy(blk2, blk, 48);
    blk2[48] = 0x00; blk2[49] = 0xC0; /* d = -2.0 */
    memset(blk2 + 50, 1, 32);
    float out2[64];
    CHECK(qw_dequant_q8_0_row(blk2, out2, 2) == QW_OK);
    CHECK(out2[0] == 0.0f && out2[15] == -4.0f && out2[32] == -2.0f
          && out2[63] == -2.0f);

    /* Error handling. */
    CHECK(qw_dequant_q8_0_row(NULL, out, 1) == QW_ERR_NULL);
    CHECK(qw_dequant_q8_0_row(blk, NULL, 1) == QW_ERR_NULL);
    CHECK(qw_dequant_q8_0_row(blk, out, 0) == QW_ERR_RANGE);
    CHECK(qw_dequant_q8_0_row(blk, out, -1) == QW_ERR_RANGE);
}

/* ------------------------------------------- test 5: q4_0 nibble orders */
static void test_q4_0(void)
{
    printf("test 5: q4_0 decode, both nibble orders pinned\n");

    /* d = 1.0 -> fp16 0x3C00. Convention: element i in byte i/2, even i =
     * LOW nibble, odd i = HIGH nibble.
     *
     * qh[0] = 0x01 -> byte 0: lo=1 (elem 0), hi=0 (elem 1)
     *    => out[0] = 1*(1-8) = -7, out[1] = 1*(0-8) = -8.
     * qh[1] = 0x42 -> byte 1: lo=2 (elem 2), hi=4 (elem 3)
     *    => out[2] = -6, out[3] = -4.
     * qh[2] = 0xFF -> byte 2: lo=15 (elem 4), hi=15 (elem 5)
     *    => out[4] = 7, out[5] = 7.
     * qh[3] = 0x08 -> byte 3: lo=8 (elem 6), hi=0 (elem 7)
     *    => out[6] = 0, out[7] = -8.
     * If a kernel flipped the convention, out[0] would be -8 and out[1]
     * -7: this test pins the low/high assignment directly. */
    uint8_t blk[32];
    memset(blk, 0, sizeof(blk));
    blk[0] = 0x00; blk[1] = 0x3C; /* d = 1.0 */
    blk[2] = 0x01;
    blk[3] = 0x42;
    blk[4] = 0xFF;
    blk[5] = 0x08;

    float out[32];
    CHECK(qw_dequant_q4_0_row(blk, out, 1) == QW_OK);
    CHECK(out[0] == -7.0f);
    CHECK(out[1] == -8.0f);
    CHECK(out[2] == -6.0f);
    CHECK(out[3] == -4.0f);
    CHECK(out[4] == 7.0f);
    CHECK(out[5] == 7.0f);
    CHECK(out[6] == 0.0f);
    CHECK(out[7] == -8.0f);

    /* fp16 variant agrees after widening. */
    uint16_t h[32];
    CHECK(qw_dequant_q4_0_row_to_f16(blk, h, 1) == QW_OK);
    CHECK(qw_f16_to_f32(h[0]) == -7.0f && qw_f16_to_f32(h[1]) == -8.0f);

    CHECK(qw_dequant_q4_0_row(NULL, out, 1) == QW_ERR_NULL);
    CHECK(qw_dequant_q4_0_row(blk, NULL, 1) == QW_ERR_NULL);
    CHECK(qw_dequant_q4_0_row(blk, out, 0) == QW_ERR_RANGE);
}

/* ------------------------------------------- test 6: passthroughs + bytes */
static void test_passthrough_bytes(void)
{
    printf("test 6: f16/f32 passthrough and qw_dequant_bytes_for\n");

    float src32[4] = { 1.0f, -2.0f, 0.5f, 3.25f };
    float dst[4];
    CHECK(qw_dequant_f32_row(src32, dst, 4) == QW_OK);
    for (int i = 0; i < 4; i++)
        CHECK(dst[i] == src32[i]);

    /* f16 passthrough: write known fp16 bits, read back floats. */
    uint8_t f16row[8];
    f16row[0] = 0x00; f16row[1] = 0x3C; /* 1.0  */
    f16row[2] = 0x00; f16row[3] = 0xBC; /* -1.0 */
    f16row[4] = 0x00; f16row[5] = 0x3C; /* 1.0  */
    f16row[6] = 0xFF; f16row[7] = 0x7B; /* 65504 */
    float d2[4];
    CHECK(qw_dequant_f16_row(f16row, d2, 4) == QW_OK);
    CHECK(d2[0] == 1.0f && d2[1] == -1.0f && d2[2] == 1.0f && d2[3] == 65504.0f);

    /* f32 -> f16 conversion passthrough. */
    uint16_t h[4];
    CHECK(qw_dequant_f32_row_to_f16(src32, h, 4) == QW_OK);
    CHECK(qw_f16_to_f32(h[0]) == 1.0f && qw_f16_to_f32(h[3]) == 3.25f);

    /* Row-length byte counts must match the gguf block table. */
    uint32_t bs, row;
    size_t want[5] = { 0, 0, 0, 0, 0 };
    gguf_tensor_type ts[5] = { GGUF_T_F32, GGUF_T_F16, GGUF_T_Q4_0,
                               GGUF_T_Q4_1, GGUF_T_Q8_0 };
    for (int i = 0; i < 5; i++) {
        gguf_type_block_info(ts[i], &bs, &row);
        size_t ne = (row == 1) ? 4 : 64;
        want[i] = (ne / row) * bs;
        CHECK(qw_dequant_bytes_for(ne, ts[i]) == want[i]);
    }
    CHECK(qw_dequant_bytes_for(4, GGUF_T_F32) == 16);
    CHECK(qw_dequant_bytes_for(4, GGUF_T_F16) == 8);
    CHECK(qw_dequant_bytes_for(32, GGUF_T_Q4_0) == 32);
    CHECK(qw_dequant_bytes_for(64, GGUF_T_Q8_0) == 96);
    CHECK(qw_dequant_bytes_for(64, GGUF_T_Q4_1) == 96);
    CHECK(qw_dequant_bytes_for(0, GGUF_T_Q8_0) == 0);       /* empty row  */
    CHECK(qw_dequant_bytes_for(31, GGUF_T_Q4_0) == 0);      /* not aligned */
    /* 2^61 elements * 4 bytes = 2^63, just under SIZE_MAX (2^64-1). */
    CHECK(qw_dequant_bytes_for(1ULL << 61, GGUF_T_F32) == (size_t)1 << 63);
    /* 2^62 * 4 = 2^64 > SIZE_MAX: overflow -> 0. */
    CHECK(qw_dequant_bytes_for(1ULL << 62, GGUF_T_F32) == 0);
    /* Just inside: (SIZE_MAX / 4) * 4 is representable. */
    CHECK(qw_dequant_bytes_for(4611686018427387903ULL, GGUF_T_F32)
          == 18446744073709551612ULL);
}

/* ------------------------------------------- test 2b: fp16 widen sweep */
static void test_f16_widen_sweep(void)
{
    printf("test 2b: exhaustive fp16 -> f32 sweep (all 65536 patterns)\n");

    int bad = 0, exact = 0;
    double max_rel = 0.0;
    for (uint32_t u = 0; u < 65536; u++) {
        float x = qw_f16_to_f32((uint16_t)u);
        if (x != x)
            continue;                        /* NaN: checked by specials */
        uint16_t h = qw_f32_to_f16(x);
        if (h != (uint16_t)u) {
            if (bad < 5)
                printf("  widen mismatch: h=0x%04X -> f=0x%08X -> h=0x%04X\n",
                       u, f32_bits(x), h);
            bad++;
        }
        float w = qw_f16_to_f32(h);
        if (w == x)
            exact++;
        else if (w != w || x != x)
            continue;
        /* Adjacent fp16 values differ by at most one fp16 ulp, so a widen
         * mismatch is at most 2^-10 relative (subnormals included). */
        double rel = (x != 0.0f) ? fabsd(w - (double)x) / (double)x : 0.0;
        if (rel > max_rel)
            max_rel = rel;
    }
    printf("  roundtrip-exact: %d/65536, max widen mismatch rel: %.6e\n",
           exact, max_rel);
    CHECK(bad == 0);
    CHECK(max_rel <= 0.0009765625 + 1e-12);  /* 2^-10 */
}

int main(void)
{
    test_f16_specials();
    test_f16_random();
    test_f16_widen_sweep();
    test_bf16();
    test_q8_0();
    test_q4_0();
    test_passthrough_bytes();
    if (g_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fail);
    return g_fail;
}
