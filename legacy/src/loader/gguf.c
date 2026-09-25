/* src/loader/gguf.c — read-only GGUF v2/v3 file reader (see qw/gguf.h).
 *
 * Implementation notes
 * ---------------------
 * * The file is mmap'd READ-ONLY (MAP_PRIVATE). Only the header, KV section
 *   and tensor info table are parsed; tensor data is never read eagerly, so
 *   a 400 GiB file costs ~zero host RAM. If mmap is unavailable (or the
 *   mapping fails), the reader falls back to FILE* + pread over the same
 *   byte offsets — documented trade-off: slightly slower metadata parsing,
 *   identical behavior.
 * * Every read is bounds-checked against the mapped length: truncated files
 *   are rejected (QW_ERR_FORMAT), never read out of bounds.
 * * All sizes are uint64_t; every multiplication is overflow-checked.
 * * n_dims is capped at 4 (larger => QW_ERR_FORMAT).
 *
 * Tensor data layout convention (llama.cpp / GGUF): tensors are laid out in
 * table order with the data of tensor i starting at an alignment-aligned
 * offset, and the `offset` field in the table is RELATIVE to the start of
 * the data section (the first aligned offset after the KV section). We
 * therefore compute the absolute offset as:
 *
 *     data_start = align_up(kv_end, alignment)
 *     abs_off[i] = data_start + rel_off[i]        (rel_off nondecreasing,
 *                                                   rel_off[0] == 0)
 *
 * This makes "offsets monotonically sensible" checkable and matches what
 * real llama.cpp exporters emit. A file whose relative offsets are not
 * nondecreasing, or whose first is nonzero, is rejected by gguf_validate.
 */
/* POSIX interfaces (mmap, pread, fdopen, off_t) under -std=c17. */
#define _POSIX_C_SOURCE 200809L

#include "qw/gguf.h"
#include "qw/macros.h"

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ----------------------------------------------------------------- cfg */
#define GGUF_MAX_DIMS      4
#define GGUF_MAX_TENSORS   5000000ull
#define GGUF_MAX_KV        5000000ull
#define GGUF_MAX_DIM       (1ull << 30)
#define GGUF_ARR_DEPTH_MAX 32

static const uint8_t kMagic[4] = { 'G', 'G', 'U', 'F' };

/* ----------------------------------------------------------------- read */
/* Byte-level access to either a mapping or a pread() stream. */
struct rw {
    const uint8_t *m; /* mmap base; NULL => pread fallback */
    int            fd; /* pread fallback; -1 in mmap mode */
    uint64_t       size;
};

/* Copy n bytes at offset off into dst. Returns 1 on success, 0 on OOB /
 * IO error (sets *msg). mmap mode: direct memcpy. pread fallback: chunked
 * pread into dst (no intermediate buffer needed). */
static int rd(const struct rw *r, uint64_t off, void *dst, uint64_t n,
              const char **msg)
{
    if (QW_UNLIKELY(off > r->size || n > r->size - off)) {
        *msg = "read past end of file";
        return 0;
    }
    if (r->m != NULL) {
        memcpy(dst, r->m + off, (size_t)n);
        return 1;
    }
    uint8_t *d = (uint8_t *)dst;
    uint64_t got = 0;
    while (got < n) {
        uint64_t chunk = n - got;
        if (chunk > 65536)
            chunk = 65536;
        ssize_t k = pread(r->fd, d + got, (size_t)chunk, (off_t)(off + got));
        if (k <= 0) {
            *msg = "read error (pread fallback)";
            return 0;
        }
        got += (uint64_t)k;
    }
    return 1;
}

static int rd_u8(const struct rw *r, uint64_t off, uint8_t *v,
                 const char **msg)
{
    return rd(r, off, v, 1, msg);
}
static int rd_u32(const struct rw *r, uint64_t off, uint64_t *v,
                  const char **msg)
{
    uint8_t b[4];
    if (!rd(r, off, b, 4, msg))
        return 0;
    *v = (uint64_t)b[0] | ((uint64_t)b[1] << 8) |
         ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24);
    return 1;
}
static int rd_u64(const struct rw *r, uint64_t off, uint64_t *v,
                  const char **msg)
{
    uint8_t b[8];
    if (!rd(r, off, b, 8, msg))
        return 0;
    uint64_t x = 0;
    for (int i = 7; i >= 0; i--)
        x = (x << 8) | b[i];
    *v = x;
    return 1;
}

static float bits_f32(uint32_t u)
{
    union { uint32_t u; float f; } v;
    v.u = u;
    return v.f;
}
static double bits_f64(uint64_t u)
{
    union { uint64_t u; double d; } v;
    v.u = u;
    return v.d;
}

/* --------------------------------------------------------------- alloc */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (p == NULL)
        fprintf(stderr, "qw/gguf: out of memory\n");
    return p;
}
static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (p == NULL)
        fprintf(stderr, "qw/gguf: out of memory\n");
    return p;
}
static char *xstrndup(const void *s, size_t n)
{
    char *p = xmalloc(n + 1);
    if (p == NULL)
        return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

/* ----------------------------------------------------------- gguf type */
const char *gguf_type_name(gguf_type t)
{
    static const char *names[] = {
        "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32",
        "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64",
    };
    if ((unsigned)t >= QW_ARRAY_COUNT(names))
        return "UNKNOWN";
    return names[t];
}

size_t gguf_type_size(gguf_type t)
{
    switch (t) {
    case GGUF_UINT8:   return 1;
    case GGUF_INT8:    return 1;
    case GGUF_UINT16:  return 2;
    case GGUF_INT16:   return 2;
    case GGUF_UINT32:  return 4;
    case GGUF_INT32:   return 4;
    case GGUF_FLOAT32: return 4;
    case GGUF_BOOL:    return 1;
    case GGUF_UINT64:  return 8;
    case GGUF_INT64:   return 8;
    case GGUF_FLOAT64: return 8;
    case GGUF_STRING:  return 0;
    case GGUF_ARRAY:   return 0;
    default:           return 0;
    }
}

const char *gguf_tensor_type_name(gguf_tensor_type t)
{
    switch (t) {
    case GGUF_T_F32:     return "F32";
    case GGUF_T_F16:     return "F16";
    case GGUF_T_BF16:    return "BF16";
    case GGUF_T_Q4_0:    return "Q4_0";
    case GGUF_T_Q4_1:    return "Q4_1";
    case GGUF_T_Q5_0:    return "Q5_0";
    case GGUF_T_Q5_1:    return "Q5_1";
    case GGUF_T_Q8_0:    return "Q8_0";
    case GGUF_T_Q8_1:    return "Q8_1";
    case GGUF_T_Q2_K:    return "Q2_K";
    case GGUF_T_Q3_K:    return "Q3_K";
    case GGUF_T_Q4_K:    return "Q4_K";
    case GGUF_T_Q5_K:    return "Q5_K";
    case GGUF_T_Q6_K:    return "Q6_K";
    case GGUF_T_IQ2_XXS: return "IQ2_XXS";
    case GGUF_T_IQ2_XS:  return "IQ2_XS";
    case GGUF_T_IQ3_XS:  return "IQ3_XS";
    case GGUF_T_IQ1_S:   return "IQ1_S";
    case GGUF_T_IQ4_NL:  return "IQ4_NL";
    case GGUF_T_IQ3_S:   return "IQ3_S";
    case GGUF_T_IQ2_S:   return "IQ2_S";
    case GGUF_T_IQ4_XS:  return "IQ4_XS";
    case GGUF_T_F8_E4M3: return "F8_E4M3";
    case GGUF_T_F8_E5M2: return "F8_E5M2";
    case GGUF_T_F8_E8M0: return "F8_E8M0";
    case GGUF_T_F4_E2M1: return "F4_E2M1";
    default:             return "UNKNOWN";
    }
}

/* Bytes per block (bs) / elements per block (row) for quantized types;
 * for fixed-size types bs = element size, row = 1. */
bool gguf_type_block_info(gguf_tensor_type t, uint32_t *bs, uint32_t *row)
{
    switch (t) {
    case GGUF_T_F32:     *bs = 4;   *row = 1;   return true;
    case GGUF_T_F16:     *bs = 2;   *row = 1;   return true;
    case GGUF_T_BF16:    *bs = 2;   *row = 1;   return true;
    case GGUF_T_Q4_0:    *bs = 32;  *row = 32;  return true;
    case GGUF_T_Q4_1:    *bs = 48;  *row = 32;  return true;
    case GGUF_T_Q5_0:    *bs = 48;  *row = 32;  return true;
    case GGUF_T_Q5_1:    *bs = 64;  *row = 32;  return true;
    case GGUF_T_Q8_0:    *bs = 48;  *row = 32;  return true;
    case GGUF_T_Q8_1:    *bs = 64;  *row = 128; return true;
    case GGUF_T_Q2_K:    *bs = 72;  *row = 256; return true;
    case GGUF_T_Q3_K:    *bs = 112; *row = 256; return true;
    case GGUF_T_Q4_K:    *bs = 144; *row = 256; return true;
    case GGUF_T_Q5_K:    *bs = 176; *row = 256; return true;
    case GGUF_T_Q6_K:    *bs = 152; *row = 32;  return true;
    case GGUF_T_IQ2_XXS: *bs = 88;  *row = 256; return true;
    case GGUF_T_IQ2_XS:  *bs = 88;  *row = 256; return true;
    case GGUF_T_IQ3_XS:  *bs = 132; *row = 256; return true;
    case GGUF_T_IQ1_S:   *bs = 88;  *row = 128; return true;
    case GGUF_T_IQ4_NL:  *bs = 168; *row = 128; return true;
    case GGUF_T_IQ3_S:   *bs = 132; *row = 96;  return true;
    case GGUF_T_IQ2_S:   *bs = 132; *row = 256; return true;
    case GGUF_T_IQ4_XS:  *bs = 176; *row = 256; return true;
    case GGUF_T_F8_E4M3: *bs = 1;   *row = 1;   return true;
    case GGUF_T_F8_E5M2: *bs = 1;   *row = 1;   return true;
    case GGUF_T_F8_E8M0: *bs = 1;   *row = 1;   return true;
    case GGUF_T_F4_E2M1: *bs = 1;   *row = 1;   return true; /* packed
                                                                2/byte;
                                                                nbytes is
                                                                dims/2 */
    default:             *bs = 0;   *row = 0;   return false;
    }
}

/* ------------------------------------------------------------ gguf_file */
struct name_ent {
    char    *name;
    uint32_t idx;
};

struct gguf_file {
    struct rw    rw;
    bool         own_map; /* munmap on close */
    bool         own_fp;  /* close(fd) on close (pread fallback) */
    uint32_t     version;
    uint64_t     n_kv;
    uint64_t     n_tensors;
    uint64_t     align;
    uint64_t     file_size;
    uint64_t     data_start; /* absolute offset of first tensor byte */

    struct gguf_kv        *kv;
    struct gguf_tensor_info *tensors;
    struct name_ent       *names;
    size_t                 name_cap;
};

static void free_kv_tree(struct gguf_kv *kv, uint64_t n, uint64_t depth);

/* Recursively free a KV array: element contents (strings, nested
 * sub-KVs) AND the arr.data pointer itself. Used by gguf_close. */
static void free_kv_tree(struct gguf_kv *kv, uint64_t n, uint64_t depth)
{
    if (kv == NULL)
        return;
    for (uint64_t i = 0; i < n; i++) {
        if (kv[i].t == GGUF_ARRAY) {
            if (kv[i].arr.et == GGUF_STRING && kv[i].arr.data) {
                char **ps = (char **)kv[i].arr.data;
                for (uint64_t j = 0; j < kv[i].arr.n; j++)
                    free(ps[j]);
            } else if (kv[i].arr.et == GGUF_ARRAY && kv[i].arr.data) {
                if (depth + 1 <= GGUF_ARR_DEPTH_MAX)
                    free_kv_tree((struct gguf_kv *)kv[i].arr.data,
                                 kv[i].arr.n, depth + 1);
            }
            free(kv[i].arr.data);
            free(kv[i].arr.lens);
        } else if (kv[i].t == GGUF_STRING) {
            free(kv[i].str);
        }
        free(kv[i].key);
    }
}

/* --------------------------------------------------------- name hash */
static uint64_t fnv1a_len(const void *s, size_t len)
{
    const uint8_t *p = (const uint8_t *)s;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* -------------------------------------------------------- value parse */
/* Parse one KV value at *off (advances *off). depth = nesting depth for
 * arrays (0 = top level). Returns 0 on error (sets *msg, *err). */
static int parse_value(const struct rw *r, struct gguf_kv *kv, uint64_t *off,
                       uint64_t depth, const char **msg, qw_err *err)
{
    if (kv->t == GGUF_STRING) {
        uint64_t slen = 0;
        if (!rd_u64(r, *off, &slen, msg))
            goto efmt;
        if (*off + 8 + slen > r->size) {
            *msg = "string value runs past end of file";
            goto efmt;
        }
        uint8_t buf[256];
        uint8_t *dst = buf;
        if (slen > sizeof(buf)) {
            dst = xmalloc(slen);
            if (dst == NULL)
                goto emem;
        }
        if (!rd(r, *off + 8, dst, slen, msg)) {
            free(dst != buf ? dst : NULL);
            goto efmt;
        }
        kv->str = xstrndup(dst, slen);
        free(dst != buf ? dst : NULL);
        if (kv->str == NULL)
            goto emem;
        kv->str_len = (size_t)slen;
        *off += 8 + slen;
        return 1;
    }

    if (kv->t == GGUF_ARRAY) {
        if (depth >= GGUF_ARR_DEPTH_MAX) {
            *msg = "array nesting depth exceeds 32";
            goto efmt;
        }
        uint8_t etb = 0;
        if (!rd_u8(r, *off, &etb, msg))
            goto efmt;
        *off += 1;
        uint64_t n = 0;
        if (!rd_u64(r, *off, &n, msg))
            goto efmt;
        *off += 8;
        if ((int)etb > (int)GGUF_FLOAT64) {
            *msg = "bad array element type";
            goto efmt;
        }
        if (n > GGUF_MAX_KV) {
            *msg = "implausible array element count";
            goto efmt;
        }
        kv->arr.et = (gguf_type)etb;
        kv->arr.n = n;
        if (n == 0) {
            kv->arr.data = NULL;
            return 1;
        }
        size_t esz = gguf_type_size(kv->arr.et);
        /* esz==0 only for STRING / ARRAY (the recursive cases below). */
        if (esz == 0 && (int)etb != (int)GGUF_STRING &&
            (int)etb != (int)GGUF_ARRAY) {
            *msg = "bad array element type (2)";
            goto efmt;
        }
        if (esz != 0 && n > ULLONG_MAX / esz) {
            *msg = "array byte size overflow";
            goto efmt;
        }

        if ((int)etb == (int)GGUF_STRING) {
            char **ps = xmalloc(n * sizeof(char *));
            size_t *ls = xmalloc(n * sizeof(size_t));
            if (ps == NULL || ls == NULL) {
                free(ps);
                free(ls);
                goto emem;
            }
            kv->arr.data = ps;
            kv->arr.lens = ls;
            for (uint64_t j = 0; j < n; j++) {
                uint64_t slen = 0;
                if (!rd_u64(r, *off, &slen, msg) ||
                    *off + 8 + slen > r->size) {
                    *msg = "array string runs past end of file";
                    goto efmt;
                }
                uint8_t buf[256];
                uint8_t *dst = buf;
                if (slen > sizeof(buf)) {
                    dst = xmalloc(slen);
                    if (dst == NULL) {
                        goto efmt; /* caller frees arr.data/lens + partials */
                    }
                }
                if (!rd(r, *off + 8, dst, slen, msg)) {
                    free(dst != buf ? dst : NULL);
                    goto efmt;
                }
                ps[j] = xstrndup(dst, slen);
                free(dst != buf ? dst : NULL);
                if (ps[j] == NULL)
                    goto emem; /* caller frees arr.data/lens + partials */
                ls[j] = (size_t)slen;
                *off += 8 + slen;
            }
            return 1;
        }

        if ((int)etb == (int)GGUF_ARRAY) {
            /* Nested array: each element is itself a complete array value
             * (vtype byte + et + count + data). Recurse into parse_value
             * for each element. Depth is capped by the `depth` parameter.
             *
             * On error we intentionally do NOT free the partially-parsed
             * sub-elements: the caller (gguf_open) calls gguf_close on
             * failure, which walks the full tree via free_kv_tree. */
            struct gguf_kv *sub = xcalloc(n, sizeof(struct gguf_kv));
            if (sub == NULL)
                goto emem;
            kv->arr.data = sub;
            for (uint64_t j = 0; j < n; j++) {
                /* Each sub-element starts with a vtype byte (always
                 * GGUF_ARRAY for well-formed nested arrays). */
                uint8_t vt2 = 0;
                if (!rd_u8(r, *off, &vt2, msg))
                    goto efmt;
                *off += 1;
                if ((int)vt2 != (int)GGUF_ARRAY) {
                    *msg = "nested array element is not an array";
                    goto efmt;
                }
                sub[j].t = GGUF_ARRAY;
                if (!parse_value(r, &sub[j], off, depth + 1, msg, err))
                    goto efmt;
            }
            return 1;
        }

        /* scalar array */
        {
            uint64_t total = n * esz;
            void *d = xmalloc(total);
            if (d == NULL)
                goto emem;
            kv->arr.data = d;
            if (!rd(r, *off, d, total, msg)) {
                free(d);
                kv->arr.data = NULL;
                goto efmt;
            }
            *off += total;
            return 1;
        }
    }

    /* scalar KV */
    {
        size_t vs = gguf_type_size(kv->t);
        if (vs == 0 || vs > 8) {
            *msg = "bad scalar value type";
            goto efmt;
        }
        uint8_t b[8];
        if (!rd(r, *off, b, vs, msg))
            goto efmt;
        *off += vs;
        /* All scalar values are <= 8 bytes; decode into the widest
         * natural type so the u64/i64/f64 union members are always
         * valid for every scalar KV type. */
        switch (kv->t) {
        case GGUF_UINT8:
        case GGUF_UINT16:
        case GGUF_UINT32:
        case GGUF_UINT64: {
            uint64_t v = 0;
            for (int i = 0; i < (int)vs; i++)
                v |= (uint64_t)b[i] << (8 * i);
            kv->u64 = v;
            kv->i64 = (int64_t)v;
            break;
        }
        case GGUF_INT8:
        case GGUF_INT16:
        case GGUF_INT32:
        case GGUF_INT64: {
            uint64_t raw = 0;
            for (int i = 0; i < (int)vs; i++)
                raw |= (uint64_t)b[i] << (8 * i);
            /* sign-extend from `vs` bytes */
            uint64_t sign_bit = 1ull << (8 * vs - 1);
            uint64_t v = (raw & sign_bit) ? (raw | (~0ull << (8 * vs))) : raw;
            kv->i64 = (int64_t)v;
            kv->u64 = v;
            break;
        }
        case GGUF_FLOAT32:
        case GGUF_FLOAT64: {
            uint64_t raw = 0;
            for (int i = 0; i < (int)vs; i++)
                raw |= (uint64_t)b[i] << (8 * i);
            kv->f64 = (vs == 4) ? (double)bits_f32((uint32_t)raw)
                                 : bits_f64(raw);
            break;
        }
        case GGUF_BOOL:
            kv->bool_ = b[0] != 0;
            kv->u64 = b[0] != 0;
            kv->i64 = b[0] != 0;
            break;
        default:
            *msg = "bad scalar value type (2)";
            goto efmt;
        }
        return 1;
    }

efmt:
    *err = QW_ERR_FORMAT;
    return 0;
emem:
    *err = QW_ERR_ALLOC;
    return 0;
}

/* ---------------------------------------------------------- gguf_open */
struct gguf_file *gguf_open(const char *path, qw_err *err)
{
    qw_err e = QW_OK;
    const char *msg = NULL;
    struct gguf_file *f = NULL;

    if (QW_UNLIKELY(path == NULL)) {
        e = QW_ERR_NULL;
        goto done;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        e = QW_ERR_IO;
        goto done;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        e = QW_ERR_IO;
        goto done;
    }
    uint64_t size = (uint64_t)st.st_size;

    f = xcalloc(1, sizeof(*f));
    if (f == NULL) {
        close(fd);
        e = QW_ERR_ALLOC;
        goto done;
    }
    f->file_size = size;

    if (size > 0) {
        void *m = mmap(NULL, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m != MAP_FAILED) {
            f->rw.m = m;
            f->own_map = true;
        }
    }
    if (f->rw.m == NULL) {
        /* mmap failed: keep the open fd and read via pread().
         * Documented trade-off: slightly slower metadata parsing,
         * identical behavior; tensor data is never read either way. */
        f->rw.fd = fd;
        f->own_fp = true; /* own_fp now means "own the fd" */
    } else {
        close(fd);
    }
    f->rw.size = size;

    {
        const struct rw *r = &f->rw;

        uint8_t magic[4];
        if (!rd(r, 0, magic, 4, &msg) || memcmp(magic, kMagic, 4) != 0) {
            msg = "bad magic (not a GGUF v2/v3 file)";
            e = QW_ERR_FORMAT;
            goto fail;
        }
        uint64_t version = 0, n_tensors = 0, n_kv = 0;
        if (!rd_u32(r, 4, &version, &msg) ||
            !rd_u64(r, 8, &n_tensors, &msg) ||
            !rd_u64(r, 16, &n_kv, &msg)) {
            e = QW_ERR_FORMAT;
            goto fail;
        }
        if (version < 2 || version > 3) {
            e = QW_ERR_FORMAT;
            msg = "unsupported GGUF version (want 2 or 3)";
            goto fail;
        }
        if (n_tensors > GGUF_MAX_TENSORS || n_kv > GGUF_MAX_KV) {
            e = QW_ERR_FORMAT;
            msg = "implausible n_tensors/n_kv (>= 5,000,000)";
            goto fail;
        }
        f->version = (uint32_t)version;
        f->n_tensors = n_tensors;
        f->n_kv = n_kv;

        /* ---- KV section ---- */
        f->kv = xcalloc(n_kv, sizeof(struct gguf_kv));
        if (f->kv == NULL) {
            e = QW_ERR_ALLOC;
            goto fail;
        }
        uint64_t off = 24;
        for (uint64_t i = 0; i < n_kv; i++) {
            struct gguf_kv *kv = &f->kv[i];

            uint64_t klen = 0;
            if (!rd_u64(r, off, &klen, &msg) || off + 8 + klen > r->size) {
                if (msg == NULL)
                    msg = "KV key runs past end of file";
                e = QW_ERR_FORMAT;
                goto fail;
            }
            /* read key bytes */
            uint8_t buf[256];
            uint8_t *dst = buf;
            if (klen > sizeof(buf)) {
                dst = xmalloc(klen);
                if (dst == NULL) {
                    e = QW_ERR_ALLOC;
                    goto fail;
                }
            }
            if (!rd(r, off + 8, dst, klen, &msg)) {
                free(dst != buf ? dst : NULL);
                e = QW_ERR_FORMAT;
                goto fail;
            }
            kv->key = xstrndup(dst, klen);
            free(dst != buf ? dst : NULL);
            if (kv->key == NULL) {
                e = QW_ERR_ALLOC;
                goto fail;
            }
            kv->key_len = (size_t)klen;
            off += 8 + klen;

            uint8_t vt = 0;
            if (!rd_u8(r, off, &vt, &msg)) {
                e = QW_ERR_FORMAT;
                goto fail;
            }
            off += 1;
            if (vt > (uint8_t)GGUF_FLOAT64) {
                e = QW_ERR_FORMAT;
                msg = "bad KV value type";
                goto fail;
            }
            kv->t = (gguf_type)vt;
            if (!parse_value(r, kv, &off, 0, &msg, &e))
                goto fail;
        }

        /* ---- alignment ---- */
        uint64_t align = 32;
        for (uint64_t i = 0; i < n_kv; i++) {
            if (f->kv[i].t == GGUF_UINT32 &&
                f->kv[i].key_len == sizeof("general.alignment") - 1 &&
                memcmp(f->kv[i].key, "general.alignment",
                       sizeof("general.alignment") - 1) == 0) {
                align = f->kv[i].u64;
                break;
            }
        }
        if (align == 0 || (align & (align - 1)) != 0 || align > 4096)
            align = 32;
        f->align = align;
        /* tensor table itself starts at the aligned offset */
        off = (off + align - 1) & ~(align - 1);
        uint64_t data_start = off;
        f->data_start = data_start;

        /* ---- tensor table ---- */
        f->tensors = xcalloc(n_tensors, sizeof(struct gguf_tensor_info));
        if (f->tensors == NULL) {
            e = QW_ERR_ALLOC;
            goto fail;
        }
        uint64_t prev_rel = 0;
        for (uint64_t i = 0; i < n_tensors; i++) {
            struct gguf_tensor_info *ti = &f->tensors[i];

            uint64_t nlen = 0;
            if (!rd_u64(r, off, &nlen, &msg) || off + 8 + nlen > r->size) {
                if (msg == NULL)
                    msg = "tensor name runs past end of file";
                e = QW_ERR_FORMAT;
                goto fail;
            }
            uint8_t buf[256];
            uint8_t *dst = buf;
            if (nlen > sizeof(buf)) {
                dst = xmalloc(nlen);
                if (dst == NULL) {
                    e = QW_ERR_ALLOC;
                    goto fail;
                }
            }
            if (!rd(r, off + 8, dst, nlen, &msg)) {
                free(dst != buf ? dst : NULL);
                e = QW_ERR_FORMAT;
                goto fail;
            }
            ti->name = xstrndup(dst, nlen);
            free(dst != buf ? dst : NULL);
            if (ti->name == NULL) {
                e = QW_ERR_ALLOC;
                goto fail;
            }
            ti->name_len = (size_t)nlen;
            off += 8 + nlen;

            uint64_t nd = 0, vt = 0, rel = 0;
            if (!rd_u32(r, off, &nd, &msg)) {
                e = QW_ERR_FORMAT;
                goto fail;
            }
            off += 4;
            if (nd > GGUF_MAX_DIMS) {
                e = QW_ERR_FORMAT;
                msg = "tensor n_dims > 4";
                goto fail;
            }
            ti->n_dims = (uint32_t)nd;
            for (uint32_t d = 0; d < nd; d++) {
                uint64_t dv = 0;
                if (!rd_u64(r, off, &dv, &msg)) {
                    e = QW_ERR_FORMAT;
                    goto fail;
                }
                off += 8;
                ti->dims[d] = dv;
            }
            if (!rd_u32(r, off, &vt, &msg) ||
                !rd_u64(r, off + 4, &rel, &msg)) {
                e = QW_ERR_FORMAT;
                goto fail;
            }
            off += 12; /* dtype (4) + rel offset (8) */
            if (vt > (uint64_t)GGUF_T_F4_E2M1) {
                e = QW_ERR_FORMAT;
                msg = "bad tensor dtype";
                goto fail;
            }
            ti->t = (gguf_tensor_type)vt;

            uint32_t bs = 0, row = 0;
            if (!gguf_type_block_info(ti->t, &bs, &row)) {
                e = QW_ERR_UNSUPPORTED;
                msg = "unknown tensor dtype";
                goto fail;
            }
            uint64_t nelem = 1;
            for (uint32_t d = 0; d < nd; d++) {
                if (ti->dims[d] == 0 || ti->dims[d] > GGUF_MAX_DIM) {
                    e = QW_ERR_FORMAT;
                    msg = "bad tensor dim (zero or > 2^30)";
                    goto fail;
                }
                if (nelem > ULLONG_MAX / ti->dims[d]) {
                    e = QW_ERR_FORMAT;
                    msg = "tensor dim product overflow";
                    goto fail;
                }
                nelem *= ti->dims[d];
            }
            uint64_t nbytes;
            if (row == 1) {
                if (nelem > ULLONG_MAX / bs) {
                    e = QW_ERR_FORMAT;
                    msg = "tensor byte-size overflow";
                    goto fail;
                }
                nbytes = nelem * bs;
            } else {
                if (nelem % row != 0) {
                    e = QW_ERR_FORMAT;
                    msg = "tensor element count not a multiple of block row";
                    goto fail;
                }
                uint64_t nblk = nelem / row;
                if (nblk > ULLONG_MAX / bs) {
                    e = QW_ERR_FORMAT;
                    msg = "tensor byte-size overflow (quant)";
                    goto fail;
                }
                nbytes = nblk * bs;
            }
            ti->nbytes = nbytes;

            /* Relative offsets must be nondecreasing. Containment within
             * the file is checked by gguf_validate against file_size. */
            if (rel < prev_rel) {
                e = QW_ERR_FORMAT;
                msg = "tensor relative offsets not monotonic";
                goto fail;
            }
            prev_rel = rel;
            ti->offset = data_start + rel;
        }

        /* ---- name -> index hash (open addressing, case-sensitive) ---- */
        size_t cap = 16;
        while (cap < n_tensors * 2)
            cap *= 2;
        if (cap < 16)
            cap = 16;
        f->names = xcalloc(cap, sizeof(struct name_ent));
        if (f->names == NULL) {
            e = QW_ERR_ALLOC;
            goto fail;
        }
        f->name_cap = cap;
        for (uint64_t i = 0; i < n_tensors; i++) {
            uint64_t h =
                fnv1a_len(f->tensors[i].name, f->tensors[i].name_len) &
                (cap - 1);
            while (f->names[h].name != NULL)
                h = (h + 1) & (cap - 1);
            f->names[h].name = f->tensors[i].name;
            f->names[h].idx = (uint32_t)i;
        }
    }

done:
    if (err != NULL)
        *err = e;
    return e == QW_OK ? f : NULL;

fail:
    if (err != NULL)
        *err = e;
    gguf_close(f);
    if (err != NULL)
        *err = e;
    return NULL;
}

void gguf_close(struct gguf_file *f)
{
    if (f == NULL)
        return;
    if (f->kv != NULL)
        free_kv_tree(f->kv, f->n_kv, 0);
    free(f->kv);
    if (f->tensors != NULL) {
        for (uint64_t i = 0; i < f->n_tensors; i++)
            free(f->tensors[i].name);
        free(f->tensors);
    }
    free(f->names);
    if (f->own_map && f->rw.m != NULL)
        munmap((void *)f->rw.m, (size_t)f->rw.size);
    if (f->own_fp && f->rw.fd >= 0)
        close(f->rw.fd);
    free(f);
}

/* --------------------------------------------------------- accessors */
size_t gguf_n_tensors(const struct gguf_file *f)
{
    return f == NULL ? 0 : (size_t)f->n_tensors;
}
size_t gguf_n_kv(const struct gguf_file *f)
{
    return f == NULL ? 0 : (size_t)f->n_kv;
}
uint64_t gguf_alignment(const struct gguf_file *f)
{
    return f == NULL ? 0 : f->align;
}
uint32_t gguf_version(const struct gguf_file *f)
{
    return f == NULL ? 0 : f->version;
}

const char *gguf_get_key(const struct gguf_file *f, size_t i)
{
    return (f != NULL && i < f->n_kv) ? f->kv[i].key : NULL;
}
gguf_type gguf_get_kv_type(const struct gguf_file *f, size_t i)
{
    return (f != NULL && i < f->n_kv) ? f->kv[i].t : (gguf_type)-1;
}
int gguf_get_val_u32(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_UINT32)
        return -1;
    return (int)f->kv[i].u64;
}
int gguf_get_val_i32(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_INT32)
        return -1;
    return (int)f->kv[i].i64;
}
float gguf_get_val_f32(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_FLOAT32)
        return 0.0f;
    return (float)f->kv[i].f64;
}
bool gguf_get_val_bool(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_BOOL)
        return false;
    return f->kv[i].bool_;
}
const char *gguf_get_val_str(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_STRING)
        return NULL;
    return f->kv[i].str;
}
bool gguf_get_val_str_len(const struct gguf_file *f, size_t i,
                          const char **s, size_t *len)
{
    if (s != NULL)
        *s = NULL;
    if (len != NULL)
        *len = 0;
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_STRING)
        return false;
    if (s != NULL)
        *s = f->kv[i].str;
    if (len != NULL)
        *len = f->kv[i].str_len;
    return true;
}
int64_t gguf_get_val_i64(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_INT64)
        return -1;
    return f->kv[i].i64;
}
uint64_t gguf_get_val_u64(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_UINT64)
        return 0;
    return f->kv[i].u64;
}
double gguf_get_val_f64(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_FLOAT64)
        return 0.0;
    return f->kv[i].f64;
}



gguf_type gguf_get_arr_type(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY)
        return (gguf_type)-1;
    return f->kv[i].arr.et;
}
uint64_t gguf_get_arr_n(const struct gguf_file *f, size_t i)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY)
        return 0;
    return f->kv[i].arr.n;
}
int64_t gguf_arr_get_i32(const struct gguf_file *f, size_t i, uint64_t idx)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_INT32)
        return -1;
    if (idx >= f->kv[i].arr.n)
        return -1;
    int32_t v;
    memcpy(&v, (const char *)f->kv[i].arr.data + idx * 4, 4);
    return (int64_t)v;
}
uint64_t gguf_arr_get_u32(const struct gguf_file *f, size_t i, uint64_t idx)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_UINT32)
        return 0;
    if (idx >= f->kv[i].arr.n)
        return 0;
    uint32_t v;
    memcpy(&v, (const char *)f->kv[i].arr.data + idx * 4, 4);
    return v;
}
float gguf_arr_get_f32(const struct gguf_file *f, size_t i, uint64_t idx)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_FLOAT32)
        return 0.0f;
    if (idx >= f->kv[i].arr.n)
        return 0.0f;
    uint32_t bits;
    memcpy(&bits, (const char *)f->kv[i].arr.data + idx * 4, 4);
    return bits_f32(bits);
}
bool gguf_arr_get_bool(const struct gguf_file *f, size_t i, uint64_t idx)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_BOOL)
        return false;
    if (idx >= f->kv[i].arr.n)
        return false;
    return ((const uint8_t *)f->kv[i].arr.data)[idx] != 0;
}
const char *gguf_arr_get_str(const struct gguf_file *f, size_t i,
                             uint64_t idx)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_STRING)
        return NULL;
    if (idx >= f->kv[i].arr.n)
        return NULL;
    return ((const char **)f->kv[i].arr.data)[idx];
}
bool gguf_get_arr_str_len(const struct gguf_file *f, size_t i, uint64_t idx,
                          const char **s, size_t *len)
{
    if (s != NULL)
        *s = NULL;
    if (len != NULL)
        *len = 0;
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_STRING || idx >= f->kv[i].arr.n)
        return false;
    if (s != NULL)
        *s = ((const char **)f->kv[i].arr.data)[idx];
    if (len != NULL)
        *len = ((const size_t *)f->kv[i].arr.lens)[idx];
    return true;
}
gguf_type gguf_arr_get_arr_type(const struct gguf_file *f, size_t i,
                                uint64_t idx)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_ARRAY)
        return (gguf_type)-1;
    if (idx >= f->kv[i].arr.n)
        return (gguf_type)-1;
    return ((const struct gguf_kv *)f->kv[i].arr.data)[idx].arr.et;
}
uint64_t gguf_arr_get_arr_n(const struct gguf_file *f, size_t i,
                            uint64_t idx)
{
    if (f == NULL || i >= f->n_kv || f->kv[i].t != GGUF_ARRAY ||
        f->kv[i].arr.et != GGUF_ARRAY)
        return 0;
    if (idx >= f->kv[i].arr.n)
        return 0;
    return ((const struct gguf_kv *)f->kv[i].arr.data)[idx].arr.n;
}

/* ------------------------------------------------------------ tensors */
int gguf_find(const struct gguf_file *f, const char *name)
{
    if (f == NULL || name == NULL || f->name_cap == 0)
        return -1;
    size_t nlen = strlen(name);
    uint64_t h = fnv1a_len(name, nlen) & (f->name_cap - 1);
    for (;;) {
        const struct name_ent *e = &f->names[h];
        if (e->name == NULL)
            return -1;
        size_t elen = f->tensors[e->idx].name_len;
        if (elen == nlen && memcmp(e->name, name, nlen) == 0)
            return (int)e->idx;
        h = (h + 1) & (f->name_cap - 1);
    }
}
const struct gguf_tensor_info *gguf_tensor_by_index(
    const struct gguf_file *f, size_t i)
{
    return (f != NULL && i < f->n_tensors) ? &f->tensors[i] : NULL;
}
uint64_t gguf_tensor_offset(const struct gguf_file *f, size_t i)
{
    const struct gguf_tensor_info *t = gguf_tensor_by_index(f, i);
    return t != NULL ? t->offset : 0;
}
uint64_t gguf_tensor_nbytes(const struct gguf_file *f, size_t i)
{
    const struct gguf_tensor_info *t = gguf_tensor_by_index(f, i);
    return t != NULL ? t->nbytes : 0;
}

/* ------------------------------------------------------------ hw caps */
bool gguf_tensor_supports_hw(const struct gguf_file *f, size_t i)
{
    const struct gguf_tensor_info *t = gguf_tensor_by_index(f, i);
    if (t == NULL)
        return false;
    /* RDNA2 (gfx1030) has no hardware bf16, fp8 or fp4 units: those
     * tensors must be dequantized / upcast on host before dispatch. */
    switch (t->t) {
    case GGUF_T_BF16:
    case GGUF_T_F8_E4M3:
    case GGUF_T_F8_E5M2:
    case GGUF_T_F8_E8M0:
    case GGUF_T_F4_E2M1:
        return false;
    default:
        return true;
    }
}

/* ----------------------------------------------------------- validate */
bool gguf_validate(const struct gguf_file *f, char *report, size_t cap)
{
    if (report != NULL && cap > 0)
        report[0] = '\0';
    if (f == NULL) {
        if (report != NULL && cap > 1)
            snprintf(report, cap, "null file");
        return false;
    }
    if (f->version < 2 || f->version > 3) {
        if (report != NULL && cap > 1)
            snprintf(report, cap, "bad GGUF version %u", f->version);
        return false;
    }
    if (f->n_tensors >= GGUF_MAX_TENSORS || f->n_kv >= GGUF_MAX_KV) {
        if (report != NULL && cap > 1)
            snprintf(report, cap, "implausible n_tensors/n_kv");
        return false;
    }
    uint64_t total = 0;
    uint64_t max_end = 0;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const struct gguf_tensor_info *t = &f->tensors[i];
        if (t->offset > f->file_size ||
            t->nbytes > f->file_size - t->offset) {
            if (report != NULL && cap > 1)
                snprintf(report, cap,
                         "tensor '%s' (offset %llu + %llu bytes) exceeds "
                         "file size %llu",
                         t->name, (unsigned long long)t->offset,
                         (unsigned long long)t->nbytes,
                         (unsigned long long)f->file_size);
            return false;
        }
        uint64_t end = t->offset + t->nbytes;
        if (end < max_end) { /* overlap => corrupt table */
            if (report != NULL && cap > 1)
                snprintf(report, cap,
                         "tensor '%s' overlaps previous tensor", t->name);
            return false;
        }
        max_end = end;
        if (total > ULLONG_MAX - t->nbytes) {
            if (report != NULL && cap > 1)
                snprintf(report, cap, "total tensor bytes overflow");
            return false;
        }
        total += t->nbytes;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (t->dims[d] == 0 || t->dims[d] > GGUF_MAX_DIM) {
                if (report != NULL && cap > 1)
                    snprintf(report, cap,
                             "tensor '%s' has bad dim %llu",
                             t->name, (unsigned long long)t->dims[d]);
                return false;
            }
        }
    }
    if (report != NULL && cap > 1)
        snprintf(report, cap,
                 "OK: %llu tensors, %llu bytes used of %llu file bytes",
                 (unsigned long long)f->n_tensors,
                 (unsigned long long)total,
                 (unsigned long long)f->file_size);
    return true;
}

/* --------------------------------------------------------- metadata */
static int find_kv(const struct gguf_file *f, const char *key)
{
    if (f == NULL)
        return -1;
    size_t klen = strlen(key);
    for (uint64_t i = 0; i < f->n_kv; i++)
        if (f->kv[i].key_len == klen &&
            memcmp(f->kv[i].key, key, klen) == 0)
            return (int)i;
    return -1;
}

const char *gguf_hf_arch(const struct gguf_file *f)
{
    if (f == NULL)
        return NULL;
    int i = find_kv(f, "qwen3.next.hf_config.architectures");
    if (i < 0)
        return NULL;
    if (f->kv[i].t == GGUF_ARRAY && f->kv[i].arr.n > 0 &&
        f->kv[i].arr.et == GGUF_STRING)
        return ((const char **)f->kv[i].arr.data)[0];
    if (f->kv[i].t == GGUF_STRING)
        return f->kv[i].str;
    return NULL;
}

const char *gguf_hf_name(const struct gguf_file *f)
{
    if (f == NULL)
        return NULL;
    const char *s = gguf_get_val_str(f, (size_t)find_kv(f, "general.name"));
    if (s == NULL)
        s = gguf_get_val_str(f,
                             (size_t)find_kv(f, "qwen3.next.hf_config.name"));
    return s;
}

static bool fetch_u64(const struct gguf_file *f, const char *k1,
                      const char *k2, uint64_t *v)
{
    if (f == NULL)
        return false;
    const char *keys[2] = { k1, k2 };
    for (int k = 0; k < 2; k++) {
        if (keys[k] == NULL)
            continue;
        int i = find_kv(f, keys[k]);
        if (i < 0)
            continue;
        gguf_type t = f->kv[i].t;
        if (t == GGUF_UINT32 || t == GGUF_UINT64) {
            *v = f->kv[i].u64;
            return true;
        }
        if (t == GGUF_INT32 || t == GGUF_INT64) {
            if (f->kv[i].i64 >= 0) {
                *v = (uint64_t)f->kv[i].i64;
                return true;
            }
            /* negative: not a valid count; fall through to next key */
            continue;
        }
    }
    return false;
}

bool gguf_get_n_layers(const struct gguf_file *f, uint64_t *v)
{
    return fetch_u64(f, "llama.block_count",
                     "qwen3.next.hf_config.num_hidden_layers", v);
}
bool gguf_get_hidden_size(const struct gguf_file *f, uint64_t *v)
{
    return fetch_u64(f, "llama.embedding_length",
                     "qwen3.next.hf_config.hidden_size", v);
}
bool gguf_get_n_heads(const struct gguf_file *f, uint64_t *v)
{
    return fetch_u64(f, "llama.attention.head_count",
                     "qwen3.next.hf_config.num_attention_heads", v);
}
bool gguf_get_n_kv_heads(const struct gguf_file *f, uint64_t *v)
{
    return fetch_u64(f, "llama.attention.head_count_kv",
                     "qwen3.next.hf_config.num_key_value_heads", v);
}
bool gguf_get_vocab_size(const struct gguf_file *f, uint64_t *v)
{
    return fetch_u64(f, "llama.vocab_n",
                     "qwen3.next.hf_config.vocab_size", v);
}

/* ---------------------------------------------------------- summary */
struct size_ent {
    uint64_t nbytes;
    const struct gguf_tensor_info *t;
};
static int cmp_size_desc(const void *a, const void *b)
{
    const struct size_ent *x = (const struct size_ent *)a;
    const struct size_ent *y = (const struct size_ent *)b;
    if (x->nbytes < y->nbytes)
        return 1;
    if (x->nbytes > y->nbytes)
        return -1;
    return 0;
}

void gguf_print_summary(const struct gguf_file *f)
{
    if (f == NULL) {
        printf("(null gguf file)\n");
        return;
    }
    printf("GGUF v%u  n_kv=%llu  n_tensors=%llu  align=%llu\n",
           f->version, (unsigned long long)f->n_kv,
           (unsigned long long)f->n_tensors,
           (unsigned long long)f->align);

    static const gguf_tensor_type dts[] = {
        GGUF_T_F32,     GGUF_T_F16,     GGUF_T_BF16,
        GGUF_T_Q4_0,    GGUF_T_Q4_1,    GGUF_T_Q5_0,
        GGUF_T_Q5_1,    GGUF_T_Q8_0,    GGUF_T_Q8_1,
        GGUF_T_Q2_K,    GGUF_T_Q3_K,    GGUF_T_Q4_K,
        GGUF_T_Q5_K,    GGUF_T_Q6_K,    GGUF_T_IQ2_XXS,
        GGUF_T_IQ2_XS,  GGUF_T_IQ3_XS,  GGUF_T_IQ1_S,
        GGUF_T_IQ4_NL,  GGUF_T_IQ3_S,   GGUF_T_IQ2_S,
        GGUF_T_IQ4_XS,  GGUF_T_F8_E4M3, GGUF_T_F8_E5M2,
        GGUF_T_F8_E8M0, GGUF_T_F4_E2M1,
    };
    enum { N_DT = (int)QW_ARRAY_COUNT(dts) };
    uint64_t bydt[N_DT];
    memset(bydt, 0, sizeof(bydt));
    uint64_t total = 0;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        total += f->tensors[i].nbytes;
        for (int d = 0; d < N_DT; d++)
            if (f->tensors[i].t == dts[d]) {
                bydt[d] += f->tensors[i].nbytes;
                break;
            }
    }
    printf("total tensor bytes: %llu\n", (unsigned long long)total);
    printf("bytes by dtype:\n");
    for (int d = 0; d < N_DT; d++)
        if (bydt[d] > 0)
            printf("  %-8s %18llu\n", gguf_tensor_type_name(dts[d]),
                   (unsigned long long)bydt[d]);

    if (f->n_tensors > 0) {
        struct size_ent *ents =
            xmalloc(f->n_tensors * sizeof(struct size_ent));
        if (ents != NULL) {
            for (uint64_t i = 0; i < f->n_tensors; i++) {
                ents[i].nbytes = f->tensors[i].nbytes;
                ents[i].t = &f->tensors[i];
            }
            qsort(ents, f->n_tensors, sizeof(struct size_ent),
                  cmp_size_desc);
            size_t top = (size_t)f->n_tensors < 20
                             ? (size_t)f->n_tensors
                             : 20;
            printf("top %zu tensors by size:\n", top);
            for (size_t i = 0; i < top; i++) {
                const struct gguf_tensor_info *t = ents[i].t;
                printf("  %-8s %14llu  %s\n",
                       gguf_tensor_type_name(t->t),
                       (unsigned long long)ents[i].nbytes, t->name);
            }
            free(ents);
        }
    }

    /* per-layer tensor counts: bucket on ".blk.N." in the name */
    uint64_t *layer_n = NULL;
    uint64_t max_layer = 0;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const char *p = strstr(f->tensors[i].name, ".blk.");
        if (p == NULL)
            continue;
        p += 5;
        uint64_t n = 0;
        while (*p >= '0' && *p <= '9') {
            n = n * 10 + (uint64_t)(*p - '0');
            p++;
        }
        if (n >= max_layer) {
            uint64_t *nn =
                realloc(layer_n, (n + 1) * sizeof(uint64_t));
            if (nn == NULL)
                break;
            layer_n = nn;
            for (uint64_t j = max_layer; j <= n; j++)
                layer_n[j] = 0;
            max_layer = n + 1;
        }
        layer_n[n]++;
    }
    if (layer_n != NULL) {
        uint64_t min_l = 0;
        while (min_l < max_layer && layer_n[min_l] == 0)
            min_l++;
        uint64_t max_l = max_layer - 1;
        while (max_l > min_l && layer_n[max_l] == 0)
            max_l--;
        uint64_t mincnt = UINT64_MAX, maxcnt = 0;
        for (uint64_t n = min_l; n <= max_l; n++) {
            if (layer_n[n] < mincnt)
                mincnt = layer_n[n];
            if (layer_n[n] > maxcnt)
                maxcnt = layer_n[n];
        }
        printf("layers %llu..%llu: tensors/layer min=%llu max=%llu\n",
               (unsigned long long)min_l, (unsigned long long)max_l,
               (unsigned long long)mincnt, (unsigned long long)maxcnt);
        free(layer_n);
    }
}
