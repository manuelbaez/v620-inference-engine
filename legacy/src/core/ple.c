/* src/core/ple.c — n-gram PLE table: addressing math, memory fit, PCIe
 * gather bandwidth model, and the fp8 E4M3 codec. Pure C17, libc only.
 *
 * All byte accounting is uint64_t with overflow-checked multiplies (qw_umul),
 * matching the convention in src/core/config.c. No VLA anywhere; the only
 * allocation-free design is the point (the table is huge and host-resident,
 * this module only computes where its rows are and how many bytes move).
 */
#include "qw/ple.h"
#include "qw/macros.h"

/* <math.h> for the NAN macro only (no libm functions called; links without
 * -lm). */
#include <math.h>
#include <stdio.h>
#include <string.h>

/* GiB as a binary unit (1024^3), matching config.c's qw_print_gib. */
static const uint64_t QW_PLE_GIB = 1024ULL * 1024ULL * 1024ULL;

/* Overflow-checked multiply: *r = a*b, QW_ERR_RANGE on overflow. */
static qw_err qw_umul(uint64_t a, uint64_t b, uint64_t *r)
{
    if (a != 0 && b > UINT64_MAX / a)
        return QW_ERR_RANGE;
    *r = a * b;
    return QW_OK;
}

/* Round half to even: distance-to-floor and distance-to-ceiling compared,
 * ties (exactly 0.5) go to the even neighbor. Returns an int >= 0. Used for
 * the E4M3 3-bit mantissa; no floating-point rounding-mode dependence. */
static int rne(int64_t num, int64_t den)
{
    int64_t q = num / den;
    int64_t r = num % den;
    if (r < 0) {
        q--;
        r += den;
    }
    int64_t rem2 = 2 * r;
    if (rem2 > den)
        q++;
    else if (rem2 == den && (q & 1) != 0)
        q++;
    return (int)q;
}

/* ------------------------------------------------------------- defaults */
qw_ple_desc qw_ple_default(void)
{
    qw_ple_desc d;
    d.n_entries  = QW_PLE_DEFAULT_ENTRIES;
    d.row_bytes  = QW_PLE_DEFAULT_ROW_BYTES;
    d.fp8_format = 1; /* E4M3 */
    d.layer      = QW_PLE_DEFAULT_LAYER;
    return d;
}

/* ------------------------------------------------------------------- size */
uint32_t qw_ple_row_bytes(const struct qw_model_cfg *cfg)
{
    /* row = one hidden vector in fp8 (1 B/elem). Fall back to the default
     * width if no cfg (or a non-positive hidden_size) is supplied. */
    if (cfg != NULL && cfg->hidden_size > 0)
        return (uint32_t)cfg->hidden_size;
    return QW_PLE_DEFAULT_ROW_BYTES;
}

uint64_t qw_ple_total_bytes(const qw_ple_desc *desc)
{
    if (desc == NULL)
        return 0;
    uint64_t t;
    if (qw_umul(desc->n_entries, (uint64_t)desc->row_bytes, &t) != QW_OK)
        return 0;
    return t;
}

bool qw_ple_fits_host_ram(const qw_ple_desc *desc)
{
    uint64_t total = qw_ple_total_bytes(desc);
    if (total == 0)
        return false;
    uint64_t budget;
    if (qw_umul(QW_PLE_HOST_RAM_GIB, QW_PLE_GIB, &budget) != QW_OK)
        return false;
    return total <= budget;
}

/* ------------------------------------------------------------------ index */
/* FNV-1a 64-bit over n_ids consecutive token ids, each expanded to 4 little-
 * endian bytes (byte 0 = least significant), then reduced modulo n_entries.
 * Deterministic; collisions alias n-grams to one row BY DESIGN. */
static uint64_t qw_ple_fnv1a(const uint32_t *ids, int n)
{
    uint64_t h = QW_PLE_FNV_OFFSET;
    for (int i = 0; i < n; i++) {
        uint32_t v = ids[i];
        for (int byte = 0; byte < 4; byte++) {
            h ^= (uint64_t)(v & 0xff);
            h *= QW_PLE_FNV_PRIME;
            v >>= 8;
        }
    }
    return h;
}

qw_err qw_ple_index(const qw_ple_desc *desc, const uint32_t *token_ids,
                    int n_ids, int ngram_n, uint64_t *out_index)
{
    if (desc == NULL || token_ids == NULL || out_index == NULL)
        return QW_ERR_NULL;
    if (n_ids < 1 || ngram_n < 1 || n_ids > ngram_n)
        return QW_ERR_RANGE;
    if (desc->n_entries == 0)
        return QW_ERR_RANGE;
    *out_index = qw_ple_fnv1a(token_ids, n_ids) % desc->n_entries;
    return QW_OK;
}

qw_err qw_ple_index_unigram(const qw_ple_desc *desc, uint32_t token_id,
                            uint64_t *out_index)
{
    return qw_ple_index(desc, &token_id, 1, 1, out_index);
}

/* ----------------------------------------------------------------- offset */
qw_err qw_ple_row_offset(const qw_ple_desc *desc, uint64_t index,
                         uint64_t *byte_offset)
{
    if (desc == NULL || byte_offset == NULL)
        return QW_ERR_NULL;
    uint64_t off;
    if (qw_umul(index, (uint64_t)desc->row_bytes, &off) != QW_OK)
        return QW_ERR_RANGE;
    /* Row must lie wholly inside the table: offset + row_bytes <= total.
     * Equivalent (overflow-free) to offset <= total - row_bytes, guarded by
     * the n_entries check below: a valid index < n_entries always satisfies
     * it, and an index >= n_entries overruns the final row. */
    if (index >= desc->n_entries)
        return QW_ERR_RANGE;
    *byte_offset = off;
    return QW_OK;
}

/* ------------------------------------------------------------------- plan */
qw_err qw_ple_gather_plan(const qw_ple_desc *desc, int seq_len, int ngram_n,
                          qw_ple_plan *out)
{
    if (desc == NULL || out == NULL)
        return QW_ERR_NULL;
    if (seq_len < 1 || ngram_n < 1)
        return QW_ERR_RANGE;

    memset(out, 0, sizeof(*out));
    out->seq_len = seq_len;
    out->ngram_n = ngram_n;

    /* Each generated token gathers exactly one row; the n-gram width selects
     * which row but does not change the row width. */
    out->bytes_per_token = (uint64_t)desc->row_bytes;

    uint64_t per_seq;
    if (qw_umul((uint64_t)seq_len, (uint64_t)desc->row_bytes, &per_seq)
        != QW_OK)
        return QW_ERR_RANGE;
    out->bytes_per_seq = per_seq;

    /* Required PCIe GB/s = (bytes_per_token * tokens/s) / 1e9. Using the
     * decimal GiB/s convention of the ~28 GB/s link. */
    double bpt = (double)out->bytes_per_token;
    out->gbps_50  = bpt * 50.0  / 1e9;
    out->gbps_200 = bpt * 200.0 / 1e9;
    return QW_OK;
}

void qw_ple_plan_report(const qw_ple_plan *plan)
{
    if (plan == NULL)
        return;
    uint64_t seq_gib_int = plan->bytes_per_seq / QW_PLE_GIB;
    uint64_t seq_frac =
        (plan->bytes_per_seq % QW_PLE_GIB * 100ULL) / QW_PLE_GIB;
    /* GB/s is tiny here (~0.0005) so use %.4f; also print MiB/s for scale. */
    fprintf(stderr,
            "ple gather plan: seq_len=%d ngram=%d bytes/token=%llu "
            "bytes/seq=%llu (%llu.%02lu GiB) "
            "pcie@50=%.4f GB/s (%.3f MiB/s) pcie@200=%.4f GB/s (%.3f MiB/s) "
            "(link ~%.0f GB/s)\n",
            plan->seq_len, plan->ngram_n,
            (unsigned long long)plan->bytes_per_token,
            (unsigned long long)plan->bytes_per_seq,
            (unsigned long long)seq_gib_int, (unsigned long)seq_frac,
            plan->gbps_50, plan->gbps_50 * 1024.0 * 1024.0 / 1e6,
            plan->gbps_200, plan->gbps_200 * 1024.0 * 1024.0 / 1e6,
            QW_PLE_PCIE_GIBPS);
}

/* --------------------------------------------------------------- fp8 e4m3 */
/* Sign-magnitude bit helpers on the 8-bit E4M3 layout:
 *   bit7 = sign, bits6..3 = exponent (4, bias 7), bits2..0 = mantissa (3). */
static uint8_t e4m3_bits(int sign, uint8_t exp, uint8_t man)
{
    return (uint8_t)(((sign & 1) << 7) | ((exp & 0xf) << 3) | (man & 0x7));
}

float qw_fp8_e4m3_to_f32(uint8_t b)
{
    int     sign  = (b >> 7) & 1;
    uint8_t exp   = (b >> 3) & 0xf;
    uint8_t man   = b & 0x7;

    /* Exponent all-ones: man==7 is the max finite magnitude (1.111 x 2^8 =
     * 448, i.e. the all-ones byte 0x7f); any other mantissa is NaN. There is
     * no infinity in E4M3. */
    if (exp == 0xf)
        return (man == 7) ? (sign ? -448.0f : 448.0f) : NAN;
    if (exp == 0)
        /* subnormal: (sign) man * 2^-9 */
        return (sign ? -1.0f : 1.0f) * (float)man * 0.001953125f;

    /* Normal: (-1)^s (1 + man/8) 2^(exp-7). Integer shift (power of two). */
    float v = (1.0f + (float)man * 0.125f);
    if ((int)exp - 7 >= 0)
        v *= (float)(1U << ((int)exp - 7));
    else
        v /= (float)(1U << (7 - (int)exp));
    return sign ? -v : v;
}

uint8_t qw_f32_to_fp8_e4m3(float x)
{
    /* NaN in -> canonical NaN (exponent all-ones, mantissa 1). */
    if (x != x)
        return e4m3_bits(0, 0xf, 1);

    float ax = (x < 0.0f) ? -x : x;
    int   sign = (x < 0.0f) ? 1 : 0;

    /* Zero (incl. -0.0): keep the sign. */
    if (ax == 0.0f)
        return e4m3_bits(sign, 0, 0);

    /* Saturation: |x| >= 448 -> max finite (the all-ones byte). 448 = 1.111 x
     * 2^8 = exp field 15, mantissa 7. (No infinity in E4M3.) */
    if (ax >= 448.0f)
        return e4m3_bits(sign, 0xf, 7);

    /* Find E with ax = (1 + f) * 2^E, 0 <= f < 1, via integer doublings:
     * scale v into [1,2) and track the power of two. */
    double v = (double)ax;
    int    E = 0;
    while (v >= 2.0) { v *= 0.5; E++; }
    while (v < 1.0)  { v *= 2.0; E--; }
    double f = v - 1.0; /* fractional part in [0, 1) */

    /* Subnormal region: E < -6 (smallest normal is 1.0 x 2^-6). Value =
     * (1+f) * 2^E = ((1+f) * 2^(E+9)) * 2^-9, so the 3-bit field is
     * round-half-even((1+f) * 2^(E+9)). */
    if (E < -6) {
        /* round-half-even((1+f) * 2^(E+9)) with E+9 in [-3..0]: express
         * (1+f)*2^(E+9) = num / 2^(-E-9) and round via rne (no libm). */
        int shift = -(E + 9);                 /* 0,1,2,3 for E = -7..-10 */
        int64_t num = (int64_t)((1.0 + f) * (double)(1 << shift) * 8.0 + 0.5);
        int man = rne(num, 8);
        if (man > 7)
            return e4m3_bits(sign, 1, 0); /* carry promotes to smallest normal */
        if (man == 0)
            return e4m3_bits(sign, 0, 0);
        return e4m3_bits(sign, 0, (uint8_t)man);
    }

    /* Normal region: E in [-6, 7]. Exponent field = E + 7 (in 1..14); the
     * max-finite 448 (exp 15) is only reached by the saturation above. */
    int expf = E + 7;
    /* round-half-even of the 3 mantissa bits: f*2^21 / 2^18, via rne (no
     * libm). (f*2^21 is exact: f < 1 so f*2^21 < 2^21 < 2^53.) */
    int64_t num = (int64_t)(f * (double)(1 << 21) + 0.5);
    int man = rne(num, (1 << 18));           /* round to 3 of 21 bits */
    if (man > 7) {                                  /* carry into exponent */
        man  = 0;
        expf += 1;
        /* expf 15 + man 0 is the NaN zone, not 448: round down to 1.110 x 2^8
         * = 440, the largest finite value that does not encode NaN. */
        if (expf > 14)
            return e4m3_bits(sign, 0xe, 0x6);
    }
    return e4m3_bits(sign, (uint8_t)expf, (uint8_t)man);
}
