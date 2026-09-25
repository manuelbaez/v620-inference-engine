/* qw/ple.h — n-gram PLE (positional layer embedding) table: addressing math,
 * host memory fit, per-token PCIe bandwidth model, and fp8 E4M3 codec.
 * Pure C17, libc only. No HIP.
 *
 * The PLE table is a 20,000,000-row x 2560-byte (hidden_size, fp8 = 1 B/elem)
 * hash-embedding table of Qwen/Qwen3.8-Flash-Next, resident in HOST RAM
 * (~47.7 GiB). The GPU gathers one row per generated token over PCIe, so the
 * per-token byte count and the required GB/s are the numbers that decide
 * whether PCIe (~28 GB/s) becomes the bottleneck. This module computes them.
 *
 * Addressing: the row for an n-gram of `ngram_n` consecutive token ids is the
 * FNV-1a 64-bit hash of those ids, reduced modulo n_entries. This is a hash
 * embedding: two different n-grams that hash to the same row ALIAS onto that
 * one row BY DESIGN. There is no collision resolution, no probe, no second
 * bucket — the table is a fixed-size array indexed by a non-injective hash.
 * A small aliasing rate is the intended, acceptable trade for O(1) gathers.
 *
 * fp8 E4M3: RDNA2 (gfx1030) has NO fp8 hardware, so the table's fp8 bytes are
 * decoded to fp32 on the CPU during loading (qw_fp8_e4m3_to_f32) and re-encoded
 * when writing (qw_f32_to_fp8_e4m3). Both are pure-C bit manipulation from the
 * E4M3 format definition — no vendor headers, no fp8 instructions. On the GPU
 * the table is only ever moved as raw bytes (gathered over PCIe); it is never
 * converted there.
 */
#ifndef QW_PLE_H
#define QW_PLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/config.h"
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- constants */
#define QW_PLE_DEFAULT_ENTRIES   20000000ULL /* 20,000,000 rows            */
#define QW_PLE_DEFAULT_ROW_BYTES 2560U       /* hidden_size, 1 B/elem fp8 */
#define QW_PLE_DEFAULT_LAYER     2           /* PLE sits in layer 2        */

/* fp8 E4M3 format (no fp8 hardware on RDNA2; CPU codec only):
 *   1 sign, 4 exponent (bias 7), 3 mantissa, no infinity. The all-ones
 *   mantissa at exponent 1111 (byte 0x7f) is the max finite 448.0; exponent
 *   1111 with any other mantissa is NaN. Max finite = 448.0. */
#define QW_FP8_E4M3_MAX_FINITE  448.0f

/* Host RAM budget the whole table must fit in (GiB, binary 1024^3). */
#define QW_PLE_HOST_RAM_GIB 220ULL

/* PCIe model for the gather bandwidth estimate (GiB/s, binary; ~28 GB/s). */
#define QW_PLE_PCIE_GIBPS   28.0

/* FNV-1a 64-bit parameters (fixed by definition). */
#define QW_PLE_FNV_OFFSET  1469598103934665603ULL /* 0xcbf29ce484222325  */
#define QW_PLE_FNV_PRIME   1099511628211ULL       /* 0x00000100000001B3  */

/* ------------------------------------------------------------------ types */
/* Descriptor for one PLE table. row_bytes is fixed at the fp8 byte width
 * (hidden_size, one byte per fp8 element); the table is always fp8. */
typedef struct qw_ple_desc {
    uint64_t n_entries; /* rows (20,000,000) */
    uint32_t row_bytes; /* bytes per row (2560 = hidden_size x 1 B) */
    uint8_t  fp8_format;/* reserved: E4M3, always 1 */
    int      layer;     /* which transformer layer (2) */
} qw_ple_desc;

/* Per-token PCIe gather model for one sequence. */
typedef struct qw_ple_plan {
    uint64_t bytes_per_token; /* bytes moved per generated token (1 row)  */
    uint64_t bytes_per_seq;   /* bytes for the whole seq_len-token gather */
    double   gbps_50;         /* required PCIe GB/s at 50 tokens/s        */
    double   gbps_200;        /* required PCIe GB/s at 200 tokens/s       */
    int      seq_len;         /* sequence length the plan was built for   */
    int      ngram_n;         /* n-gram width (1 = unigram, 2 = bigram ..)*/
} qw_ple_plan;

/* -------------------------------------------------------------- defaults */
/* The Qwen3.8-Flash-Next table: 20,000,000 rows x 2560 B, fp8, layer 2. */
qw_ple_desc qw_ple_default(void);

/* ------------------------------------------------------------------- size */
/* Bytes per row of a table, from a model config's hidden_size. The row is
 * one hidden vector in fp8, so row_bytes = hidden_size * 1 byte. The config
 * is only read for hidden_size; a NULL cfg or non-positive hidden_size yields
 * the default row width (2560). */
uint32_t qw_ple_row_bytes(const struct qw_model_cfg *cfg);

/* Total table bytes: n_entries * row_bytes, overflow-checked. Returns 0 on
 * overflow (checked via the uint64 quotient test). For the default table this
 * is exactly 20,000,000 * 2560 = 51,200,000,000 B (~47.7 GiB). */
uint64_t qw_ple_total_bytes(const qw_ple_desc *desc);

/* True iff the table fits in the QW_PLE_HOST_RAM_GIB host budget. */
bool qw_ple_fits_host_ram(const qw_ple_desc *desc);

/* ------------------------------------------------------------------ index */
/* Row index for an n-gram of `n_ids` consecutive token ids: FNV-1a 64-bit
 * over the little-endian bytes of the id sequence, reduced modulo
 * n_entries. Deterministic and always in [0, n_entries). Collisions alias
 * two n-grams to the same row BY DESIGN (hash embedding) — no resolution.
 * QW_ERR_NULL on NULL ids/out, QW_ERR_RANGE on n_ids < 1, n_ids > ngram_n,
 * or n_entries == 0. */
qw_err qw_ple_index(const qw_ple_desc *desc, const uint32_t *token_ids,
                    int n_ids, int ngram_n, uint64_t *out_index);

/* 1-gram (single token) index: FNV-1a of one token id, mod n_entries.
 * Equivalent to qw_ple_index with n_ids = ngram_n = 1. */
qw_err qw_ple_index_unigram(const qw_ple_desc *desc, uint32_t token_id,
                            uint64_t *out_index);

/* ----------------------------------------------------------------- offset */
/* Byte offset of a row within the flat table: index * row_bytes,
 * overflow-checked. QW_ERR_NULL on NULL args, QW_ERR_RANGE if the multiply
 * overflows uint64 or the row would extend past the table's total bytes. */
qw_err qw_ple_row_offset(const qw_ple_desc *desc, uint64_t index,
                         uint64_t *byte_offset);

/* ------------------------------------------------------------------- plan */
/* Per-token gather bandwidth model for a sequence of seq_len tokens at
 * n-gram width ngram_n. Each generated token gathers one row, so
 * bytes_per_token = row_bytes (the n-gram only selects WHICH row; the row
 * width is always row_bytes) and bytes_per_seq = seq_len * row_bytes
 * (overflow-checked). gbps_50 / gbps_200 are the PCIe GB/s required to move
 * that traffic at 50 / 200 tokens/s — the design-deciding number, compared
 * against QW_PLE_PCIE_GIBPS (~28 GB/s). QW_ERR_NULL on NULL out,
 * QW_ERR_RANGE on seq_len < 1, ngram_n < 1, or a bytes_per_seq overflow. */
qw_err qw_ple_gather_plan(const qw_ple_desc *desc, int seq_len, int ngram_n,
                          qw_ple_plan *out);

/* Print the plan: bytes/token, bytes/seq, and required GB/s at 50 and 200
 * tokens/s (the 200 value is the one to compare against ~28 GB/s PCIe). */
void qw_ple_plan_report(const qw_ple_plan *plan);

/* --------------------------------------------------------------- fp8 e4m3 */
/* Decode one fp8 E4M3 byte to fp32 (pure bit manipulation, per the E4M3
 * definition). 0x00 = +0, 0x80 = -0. The all-ones byte 0x7f (exp 1111, man
 * 111) is the max finite magnitude (448.0); exponent 1111 with any other
 * mantissa is NaN (returned as a signaling NaN). No infinity in E4M3. */
float qw_fp8_e4m3_to_f32(uint8_t b);

/* Encode fp32 to one fp8 E4M3 byte (pure bit manipulation). Out-of-range
 * finite values saturate to the max finite magnitude (448.0, signed) rather
 * than becoming NaN. NaN in, NaN out. -0.0 in, -0 out. Rounding of the
 * 3-bit mantissa is round-half-to-even on the discarded bits. */
uint8_t qw_f32_to_fp8_e4m3(float x);

#ifdef __cplusplus
}
#endif

#endif /* QW_PLE_H */
