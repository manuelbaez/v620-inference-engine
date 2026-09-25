/* qw/gguf.h — read-only GGUF v2/v3 file reader. Pure C17, POSIX.
 *
 * The reader mmaps the file READ-ONLY (MAP_PRIVATE) and parses only the
 * header, KV section and tensor info table. Tensor data is never read
 * eagerly, so multi-hundred-GiB files cost ~zero host RAM. If mmap fails
 * (e.g. ENOMEM on 32-bit, or a filesystem that refuses mapping), the reader
 * transparently falls back to FILE* + pread; metadata parsing is identical,
 * only the read primitives differ.
 *
 * All reads are bounds-checked against the mapped length: truncated files
 * are rejected, never read out of bounds. Sizes use uint64_t throughout and
 * every multiplication is checked for overflow. n_dims is capped at 4.
 */
#ifndef QW_GGUF_H
#define QW_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ gguf_type */
/* Mirrors the GGUF value-type enumeration (spec order). */
typedef enum gguf_type {
    GGUF_UINT8   = 0,
    GGUF_INT8    = 1,
    GGUF_UINT16  = 2,
    GGUF_INT16   = 3,
    GGUF_UINT32  = 4,
    GGUF_INT32   = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL    = 7,
    GGUF_STRING  = 8,
    GGUF_ARRAY   = 9,
    GGUF_UINT64  = 10,
    GGUF_INT64   = 11,
    GGUF_FLOAT64 = 12,
} gguf_type;

const char *gguf_type_name(gguf_type t);
/* Bytes per element for scalar types; STRING and ARRAY return 0
 * (they are self-describing, not fixed-size). */
size_t gguf_type_size(gguf_type t);

/* -------------------------------------------------------------- tensors */
/* Tensor value types: GGUF_FLOAT32, GGUF_FLOAT16(2), GGUF_BF16(3),
 * GGUF_Q4_0(6), GGUF_Q4_1(7), GGUF_Q5_0(8), GGUF_Q5_1(9), GGUF_Q8_0(10),
 * GGUF_Q8_1(11), GGUF_Q2_K(12), GGUF_Q3_K(13), GGUF_Q4_K(14), GGUF_Q5_K(15),
 * GGUF_Q6_K(16), GGUF_IQ2_XXS(17), GGUF_IQ2_XS(18), GGUF_IQ3_XS(19),
 * GGUF_IQ1_S(20), GGUF_IQ4_NL(21), GGUF_IQ3_S(22), GGUF_IQ2_S(23),
 * GGUF_IQ4_XS(24), GGUF_F8_E4M3(25), GGUF_F8_E5M2(26), GGUF_F8_E8M0(27),
 * GGUF_F4_E2M1(28). Values 4, 5 and the gap 29+ are reserved. */
typedef enum gguf_tensor_type {
    GGUF_T_F32        = 0,
    GGUF_T_F16        = 1,
    GGUF_T_BF16       = 2,
    GGUF_T_Q4_0       = 3,
    GGUF_T_Q4_1       = 4,
    GGUF_T_Q5_0       = 5,
    GGUF_T_Q5_1       = 6,
    GGUF_T_Q8_0       = 7,
    GGUF_T_Q8_1       = 8,
    GGUF_T_Q2_K       = 9,
    GGUF_T_Q3_K       = 10,
    GGUF_T_Q4_K       = 11,
    GGUF_T_Q5_K       = 12,
    GGUF_T_Q6_K       = 13,
    GGUF_T_IQ2_XXS    = 14,
    GGUF_T_IQ2_XS     = 15,
    GGUF_T_IQ3_XS     = 16,
    GGUF_T_IQ1_S      = 17,
    GGUF_T_IQ4_NL     = 18,
    GGUF_T_IQ3_S      = 19,
    GGUF_T_IQ2_S      = 20,
    GGUF_T_IQ4_XS     = 21,
    GGUF_T_F8_E4M3    = 22,
    GGUF_T_F8_E5M2    = 23,
    GGUF_T_F8_E8M0    = 24,
    GGUF_T_F4_E2M1    = 25,
} gguf_tensor_type;

const char *gguf_tensor_type_name(gguf_tensor_type t);

/* --------------------------------------------------------------- kv */
struct gguf_kv {
    char      *key;
    size_t     key_len; /* exact key bytes; key is also NUL-terminated */
    gguf_type  t;
    size_t     str_len; /* GGUF_STRING only: exact value bytes; the value is
                          * also NUL-terminated (str_len is kept out of the
                          * union so it does not alias str) */
    union {
        uint64_t u64;
        int64_t  i64;
        double   f64;
        bool     bool_;
        char    *str;
        struct {
            gguf_type  et;  /* element type */
            uint64_t   n;   /* element count */
            void      *data; /* et-size * n bytes (scalars),
                                n pointers (strings/nested arrays) */
            size_t   *lens; /* ET_STRING only: exact length of element i */
        } arr;
    };
};

struct gguf_file; /* opaque */

/* ---- scalar accessors: -1 (int/bool) or NULL (string) on type mismatch */
const char *gguf_get_key(const struct gguf_file *f, size_t i);
gguf_type   gguf_get_kv_type(const struct gguf_file *f, size_t i);
int         gguf_get_val_u32(const struct gguf_file *f, size_t i);
int         gguf_get_val_i32(const struct gguf_file *f, size_t i);
float       gguf_get_val_f32(const struct gguf_file *f, size_t i);
bool        gguf_get_val_bool(const struct gguf_file *f, size_t i);
const char *gguf_get_val_str(const struct gguf_file *f, size_t i);
/* Exact byte length of a string KV value (bytes may include 0x00); *s set
 * to the value bytes. On type mismatch: *s = NULL, *len = 0, false. */
bool        gguf_get_val_str_len(const struct gguf_file *f, size_t i,
                                 const char **s, size_t *len);
int64_t     gguf_get_val_i64(const struct gguf_file *f, size_t i);
uint64_t    gguf_get_val_u64(const struct gguf_file *f, size_t i);
double      gguf_get_val_f64(const struct gguf_file *f, size_t i);

/* ---- array accessors (kv i must be GGUF_ARRAY) ---- */
gguf_type gguf_get_arr_type(const struct gguf_file *f, size_t i);
uint64_t  gguf_get_arr_n(const struct gguf_file *f, size_t i);
int64_t   gguf_arr_get_i32(const struct gguf_file *f, size_t i, uint64_t idx);
uint64_t  gguf_arr_get_u32(const struct gguf_file *f, size_t i, uint64_t idx);
float     gguf_arr_get_f32(const struct gguf_file *f, size_t i, uint64_t idx);
bool      gguf_arr_get_bool(const struct gguf_file *f, size_t i, uint64_t idx);
const char *gguf_arr_get_str(const struct gguf_file *f, size_t i, uint64_t idx);
/* Exact bytes + length of a string-array element (may include 0x00);
 * *s = element bytes, *len = element length. On mismatch/out-of-range:
 * *s = NULL, *len = 0, false. */
bool        gguf_get_arr_str_len(const struct gguf_file *f, size_t i,
                                 uint64_t idx, const char **s, size_t *len);
gguf_type gguf_arr_get_arr_type(const struct gguf_file *f, size_t i, uint64_t idx);
uint64_t  gguf_arr_get_arr_n(const struct gguf_file *f, size_t i, uint64_t idx);

/* ----------------------------------------------------------- tensors */
struct gguf_tensor_info {
    char         *name;
    size_t        name_len; /* exact name bytes; name also NUL-terminated */
    uint32_t      n_dims;      /* 0..4 (input capped; >4 rejected at load) */
    uint64_t      dims[4];     /* fast axis first; unused dims = 0 */
    gguf_tensor_type t;
    uint64_t      offset;      /* absolute byte offset of tensor data */
    uint64_t      nbytes;      /* dims * type size, overflow-checked */
};

/* ---- file ---- */
struct gguf_file *gguf_open(const char *path, qw_err *err);
void              gguf_close(struct gguf_file *f);

size_t gguf_n_tensors(const struct gguf_file *f);
size_t gguf_n_kv(const struct gguf_file *f);
uint64_t gguf_alignment(const struct gguf_file *f);
uint32_t gguf_version(const struct gguf_file *f);

/* Case-sensitive O(1) lookup; returns tensor index or -1. */
int gguf_find(const struct gguf_file *f, const char *name);
const struct gguf_tensor_info *gguf_tensor_by_index(const struct gguf_file *f,
                                                    size_t i);
uint64_t gguf_tensor_offset(const struct gguf_file *f, size_t i);
uint64_t gguf_tensor_nbytes(const struct gguf_file *f, size_t i);

/* ---- dtype block table ----
 * For block-quantized types: bytes per block (bs) and elements per block
 * (row). For plain fixed-size types: bs = element bytes, row = 1.
 * Unknown types: *bs = 0, *row = 0, returns false. */
bool gguf_type_block_info(gguf_tensor_type t, uint32_t *bs, uint32_t *row);

/* ---- hardware capability ----
 * false for bf16 and fp8/fp4 tensor types: RDNA2 (gfx1030) has no hardware
 * fp16-acc/bf16/fp8/fp4 units, so those must be dequantized or upcast on
 * host before dispatch. */
bool gguf_tensor_supports_hw(const struct gguf_file *f, size_t i);

/* ---- validation ----
 * Checks version, plausible n_tensors/n_kv, dims, that every tensor's
 * offset+nbytes lies inside the file, offsets monotonically sensible, and
 * fills report (cap bytes, NUL-terminated) with total bytes used vs file
 * size. Returns true if valid. */
bool gguf_validate(const struct gguf_file *f, char *report, size_t cap);

/* ---- metadata helpers ---- */
const char *gguf_hf_arch(const struct gguf_file *f);   /* NULL if absent */
const char *gguf_hf_name(const struct gguf_file *f);   /* NULL if absent */
/* Generic fetchers: try standard GGUF keys first, then
 * qwen3.next.hf_config.* fallback keys. *found set false if absent. */
bool gguf_get_n_layers(const struct gguf_file *f, uint64_t *v);
bool gguf_get_hidden_size(const struct gguf_file *f, uint64_t *v);
bool gguf_get_n_heads(const struct gguf_file *f, uint64_t *v);
bool gguf_get_n_kv_heads(const struct gguf_file *f, uint64_t *v);
bool gguf_get_vocab_size(const struct gguf_file *f, uint64_t *v);

/* ---- summary ----
 * Prints version, n_kv, n_tensors, alignment, total tensor bytes, bytes by
 * dtype, top-20 tensors sorted by size, and per-layer tensor counts
 * (bucketed on ".blk.N." in the name). */
void gguf_print_summary(const struct gguf_file *f);

#ifdef __cplusplus
}
#endif

#endif /* QW_GGUF_H */
