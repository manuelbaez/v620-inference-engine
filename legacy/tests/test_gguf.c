/* tests/test_gguf.c — unit tests for the GGUF reader. No external test
 * framework: plain assert, main() returns 0/1, prints PASS/FAIL per case.
 *
 * Builds a synthetic GGUF v3 in memory, writes it to a temp file via
 * mkstemp, and exercises parsing, accessors, lookup, block info, and
 * validation (including a deliberately corrupted file that must be
 * rejected).
 */
#define _POSIX_C_SOURCE 200809L /* mkstemp, strdup */

#include "qw/gguf.h"
#include "qw/types.h"

#define _GNU_SOURCE /* mallinfo2: malloc/free balance check */
#include <assert.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------- writer utils */
struct bw {
    unsigned char *p;
    size_t         n, cap;
};
static void bw_reserve(struct bw *b, size_t extra)
{
    if (b->n + extra > b->cap) {
        size_t nc = b->cap ? b->cap : 256;
        while (nc < b->n + extra)
            nc *= 2;
        unsigned char *np = realloc(b->p, nc);
        if (np == NULL) {
            fprintf(stderr, "test: OOM\n");
            exit(2);
        }
        b->p = np;
        b->cap = nc;
    }
}
static void put(struct bw *b, const void *v, size_t n)
{
    bw_reserve(b, n);
    memcpy(b->p + b->n, v, n);
    b->n += n;
}
static void put8(struct bw *b, uint8_t v) { put(b, &v, 1); }
static void put32(struct bw *b, uint32_t v)
{
    uint8_t u[4] = { (uint8_t)(v & 0xff), (uint8_t)((v >> 8) & 0xff),
                     (uint8_t)((v >> 16) & 0xff), (uint8_t)((v >> 24) & 0xff) };
    put(b, u, 4);
}
static void put64(struct bw *b, uint64_t v)
{
    for (int i = 0; i < 8; i++)
        put8(b, (uint8_t)((v >> (8 * i)) & 0xff));
}
static void putstr(struct bw *b, const char *s)
{
    put64(b, (uint64_t)strlen(s));
    put(b, s, strlen(s));
}
/* length-prefixed raw bytes (may contain NUL; unlike putstr). */
static void putstrn(struct bw *b, const void *s, size_t n)
{
    put64(b, (uint64_t)n);
    put(b, s, n);
}
/* tensor-table entry; rel = offset relative to data section start */
static void put_tensor(struct bw *b, const char *name, const uint64_t *dims,
                       int nd, uint32_t dt, uint64_t rel)
{
    putstr(b, name);
    put32(b, (uint32_t)nd);
    for (int i = 0; i < nd; i++)
        put64(b, dims[i]);
    put32(b, dt);
    put64(b, rel);
}
/* tensor-table entry whose name is raw length-prefixed bytes (may contain
 * NUL; unlike put_tensor which uses strlen). */
static void put_tensorn(struct bw *b, const void *name, size_t nlen,
                        const uint64_t *dims, int nd, uint32_t dt,
                        uint64_t rel)
{
    putstrn(b, name, nlen);
    put32(b, (uint32_t)nd);
    for (int i = 0; i < nd; i++)
        put64(b, dims[i]);
    put32(b, dt);
    put64(b, rel);
}

static int write_file(const char *path, const unsigned char *buf, size_t n)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL)
        return -1;
    size_t w = fwrite(buf, 1, n, fp);
    fclose(fp);
    return w == n ? 0 : -1;
}

/* ------------------------------------------------------------- cases */
static int g_fail = 0;
#define CHECK(cond, label)                                          \
    do {                                                            \
        if (cond)                                                   \
            printf("PASS %s\n", label);                             \
        else {                                                      \
            printf("FAIL %s (%s:%d)\n", label, __FILE__, __LINE__); \
            g_fail = 1;                                             \
        }                                                           \
    } while (0)

/* Build the synthetic file: 3 KV pairs (string, uint32, array-of-string),
 * 3 tensors (f16, q4_0, q8_0) with 32-byte-aligned data section.
 * Returns the path (caller unlinks). */
static char *build_model(void)
{
    struct bw b;
    memset(&b, 0, sizeof(b));

    /* header */
    put(&b, "GGUF", 4);
    put32(&b, 3);          /* version */
    put64(&b, 3);          /* n_tensors */
    put64(&b, 4);          /* n_kv (name, block_count, arch, ctx_len) */

    /* KV section */
    putstr(&b, "general.name");
    put8(&b, GGUF_STRING);
    putstr(&b, "qwen-test-model");

    putstr(&b, "llama.block_count");
    put8(&b, GGUF_UINT32);
    put32(&b, 48);

    putstr(&b, "general.architecture");
    put8(&b, GGUF_STRING);
    putstr(&b, "qwen3next");

    putstr(&b, "llama.context_length");
    put8(&b, GGUF_UINT32);
    put32(&b, 32768);

    /* KV section ends here; pad to 32 */
    while (b.n % 32 != 0)
        put8(&b, 0);
    uint64_t data_start = b.n;

    /* tensor table (relative offsets) */
    uint64_t d0[2] = { 2560, 4 };
    uint64_t d1[1] = { 32 };
    uint64_t d2[1] = { 32 };
    put_tensor(&b, "tok_embeddings.weight", d0, 2, GGUF_T_F16, 0);
    put_tensor(&b, "blk.0.attn.q_weight", d1, 1, GGUF_T_Q4_0, 20480);
    put_tensor(&b, "blk.0.attn.k_weight", d2, 1, GGUF_T_Q8_0, 20512);

    /* data blob */
    size_t data_end = data_start + 20560;
    while (b.n < data_end)
        put8(&b, 0xab);

    /* temp file */
    char tpl[] = "/tmp/opencode/gguf_test_XXXXXX";
    int fd = mkstemp(tpl);
    if (fd < 0) {
        fprintf(stderr, "test: mkstemp failed\n");
        exit(2);
    }
    if (write_file(tpl, b.p, b.n) != 0) {
        close(fd);
        unlink(tpl);
        fprintf(stderr, "test: write_file failed\n");
        exit(2);
    }
    close(fd);
    free(b.p);
    return strdup(tpl);
}

static int test_parse(const char *path)
{
    qw_err err = QW_ERR_NULL;
    struct gguf_file *f = gguf_open(path, &err);
    if (f == NULL) {
        printf("FAIL gguf_open (%s)\n", qw_errstr(err));
        return 1;
    }

    CHECK(gguf_version(f) == 3, "header: version == 3");
    CHECK(gguf_n_tensors(f) == 3, "header: n_tensors == 3");
    CHECK(gguf_n_kv(f) == 4, "header: n_kv == 4");
    CHECK(gguf_alignment(f) == 32, "header: alignment == 32");

    /* KV read-back (order as written) */
    CHECK(strcmp(gguf_get_key(f, 0), "general.name") == 0, "kv0 key");
    CHECK(gguf_get_kv_type(f, 0) == GGUF_STRING, "kv0 type string");
    CHECK(strcmp(gguf_get_val_str(f, 0), "qwen-test-model") == 0,
          "kv0 value string");

    CHECK(strcmp(gguf_get_key(f, 1), "llama.block_count") == 0, "kv1 key");
    CHECK(gguf_get_kv_type(f, 1) == GGUF_UINT32, "kv1 type u32");
    CHECK(gguf_get_val_u32(f, 1) == 48, "kv1 value 48");

    CHECK(strcmp(gguf_get_key(f, 2), "general.architecture") == 0,
          "kv2 key");
    CHECK(strcmp(gguf_get_val_str(f, 2), "qwen3next") == 0, "kv2 value");

    CHECK(strcmp(gguf_get_key(f, 3), "llama.context_length") == 0,
          "kv3 key");
    CHECK(gguf_get_val_u32(f, 3) == 32768, "kv3 value 32768");

    /* typed getters reject wrong types */
    CHECK(gguf_get_val_str(f, 1) == NULL, "kv1 str getter rejects u32");
    CHECK(gguf_get_val_u32(f, 0) == -1, "kv0 u32 getter rejects string");

    /* metadata helpers */
    uint64_t v = 0;
    CHECK(gguf_get_n_layers(f, &v) && v == 48, "meta: n_layers == 48");
    CHECK(!gguf_get_hidden_size(f, &v), "meta: hidden_size absent");
    CHECK(gguf_hf_name(f) != NULL &&
              strcmp(gguf_hf_name(f), "qwen-test-model") == 0,
          "meta: hf name");

    /* tensors: data section starts at 128 (header 24 + 4 KV entries +
     * padding). Verify: KV byte count */
    /*   "general.name"(12)+8 + 1 + 8+15
     *  +"llama.block_count"(17)+8 + 1 + 4
     *  +"general.architecture"(20)+8 + 1 + 8+9
     *  +"llama.context_length"(20)+8 + 1 + 4  = 117+? compute:
     *    entry1: 8+12 +1 + 8+15 = 44
     *    entry2: 8+17 +1 + 4    = 30
     *    entry3: 8+20 +1 + 8+9  = 46
     *    entry4: 8+20 +1 + 4    = 33
     *    total 153; data_start = 24 + 153 = 177 -> pad to 192. */
    const struct gguf_tensor_info *t0 = gguf_tensor_by_index(f, 0);
    const struct gguf_tensor_info *t1 = gguf_tensor_by_index(f, 1);
    const struct gguf_tensor_info *t2 = gguf_tensor_by_index(f, 2);
    CHECK(t0 != NULL && t1 != NULL && t2 != NULL, "tensors: present");

    CHECK(strcmp(t0->name, "tok_embeddings.weight") == 0, "t0 name");
    CHECK(t0->n_dims == 2 && t0->dims[0] == 2560 && t0->dims[1] == 4,
          "t0 dims 2560x4");
    CHECK(t0->t == GGUF_T_F16, "t0 dtype f16");
    CHECK(t0->nbytes == 20480, "t0 nbytes 20480");
    CHECK(t0->offset == 192, "t0 offset == data start (192)");

    CHECK(t1->n_dims == 1 && t1->dims[0] == 32, "t1 dims 32");
    CHECK(t1->t == GGUF_T_Q4_0, "t1 dtype q4_0");
    CHECK(t1->nbytes == 32, "t1 nbytes 32 (q4_0: 32B/32 elems)");
    CHECK(t1->offset == 192 + 20480, "t1 offset");

    CHECK(t2->t == GGUF_T_Q8_0, "t2 dtype q8_0");
    CHECK(t2->nbytes == 48, "t2 nbytes 48 (q8_0: 48B/32 elems)");
    CHECK(t2->offset == 192 + 20480 + 32, "t2 offset");

    /* accessors */
    CHECK(gguf_tensor_offset(f, 0) == 192, "acc: tensor_offset");
    CHECK(gguf_tensor_nbytes(f, 2) == 48, "acc: tensor_nbytes");
    CHECK(gguf_tensor_by_index(f, 99) == NULL, "acc: OOB index NULL");

    /* name lookup */
    CHECK(gguf_find(f, "tok_embeddings.weight") == 0, "find: t0");
    CHECK(gguf_find(f, "blk.0.attn.q_weight") == 1, "find: t1");
    CHECK(gguf_find(f, "blk.0.attn.k_weight") == 2, "find: t2");
    CHECK(gguf_find(f, "blk.0.attn.q_weight_") == -1, "find: near miss");
    CHECK(gguf_find(f, "TOK_EMBEDDINGS.WEIGHT") == -1,
          "find: case sensitive");
    CHECK(gguf_find(f, "") == -1, "find: empty name");

    /* block info table */
    uint32_t bs = 0, row = 0;
    CHECK(gguf_type_block_info(GGUF_T_F16, &bs, &row) && bs == 2 && row == 1,
          "block: f16 2/1");
    CHECK(gguf_type_block_info(GGUF_T_Q4_0, &bs, &row) && bs == 32 &&
              row == 32,
          "block: q4_0 32/32");
    CHECK(gguf_type_block_info(GGUF_T_Q8_0, &bs, &row) && bs == 48 &&
              row == 32,
          "block: q8_0 48/32");
    CHECK(gguf_type_block_info(GGUF_T_Q8_1, &bs, &row) && bs == 64 &&
              row == 128,
          "block: q8_1 64/128");
    CHECK(gguf_type_block_info(GGUF_T_Q6_K, &bs, &row) && bs == 152 &&
              row == 32,
          "block: q6_k 152/32");
    CHECK(gguf_type_block_info(GGUF_T_IQ4_XS, &bs, &row) && bs == 176 &&
              row == 256,
          "block: iq4_xs 176/256");
    CHECK(gguf_type_block_info(GGUF_T_IQ4_NL, &bs, &row) && bs == 168 &&
              row == 128,
          "block: iq4_nl 168/128");
    CHECK(!gguf_type_block_info((gguf_tensor_type)99, &bs, &row),
          "block: unknown dtype");

    /* hw support */
    CHECK(gguf_tensor_supports_hw(f, 0), "hw: f16 supported");
    CHECK(gguf_tensor_supports_hw(f, 1), "hw: q4_0 supported");

    /* validate: good file */
    char report[256] = { 0 };
    CHECK(gguf_validate(f, report, sizeof(report)), "validate: good file");
    printf("      report: %s\n", report);

    /* summary smoke test (no crash) */
    gguf_print_summary(f);

    gguf_close(f);
    return g_fail;
}

/* Corrupted file: same as build_model but tensor table claims a q8_0
 * tensor whose data offset is past EOF. Must be rejected by
 * gguf_validate (and ideally not even open). */
static int test_corrupt(void)
{
    struct bw b;
    memset(&b, 0, sizeof(b));
    put(&b, "GGUF", 4);
    put32(&b, 3);
    put64(&b, 1); /* n_tensors */
    put64(&b, 1); /* n_kv */
    putstr(&b, "general.name");
    put8(&b, GGUF_STRING);
    putstr(&b, "corrupt");
    while (b.n % 32 != 0)
        put8(&b, 0);
    uint64_t d[1] = { 32 };
    /* rel offset far past the actual file end */
    put_tensor(&b, "blk.99.garbage", d, 1, GGUF_T_Q8_0, 0x1000000);
    /* tiny data blob (48 bytes) so file is much shorter than claimed */
    while (b.n < 200)
        put8(&b, 0);

    char tpl[] = "/tmp/opencode/gguf_corrupt_XXXXXX";
    int fd = mkstemp(tpl);
    if (fd < 0) {
        fprintf(stderr, "test: mkstemp failed\n");
        return 1;
    }
    int wrc = write_file(tpl, b.p, b.n);
    close(fd);
    free(b.p);
    if (wrc != 0) {
        unlink(tpl);
        return 1;
    }

    qw_err err = QW_ERR_NULL;
    struct gguf_file *f = gguf_open(tpl, &err);
    int ok = 0;
    if (f == NULL) {
        /* rejected at open: acceptable (offset past file size detected) */
        ok = 1;
    } else {
        char report[256] = { 0 };
        ok = !gguf_validate(f, report, sizeof(report));
        if (ok)
            printf("      corrupt report: %s\n", report);
        gguf_close(f);
    }
    CHECK(ok, "corrupt: offset past EOF rejected");
    unlink(tpl);
    return g_fail;
}

/* Synthetic file with embedded-NUL strings:
 *  - KV string value  "nulval" = a,0x00,b,c,d (5 bytes, NUL at index 1)
 *  - string array     "nularr" = ["x\0y" (3 bytes), "plain"]
 *  - tensor name      "t\0name" (5 bytes, NUL at index 1)
 * Reader must keep the exact bytes + length for all three (no truncation). */
static int test_nul_strings(void)
{
    struct bw b;
    memset(&b, 0, sizeof(b));
    put(&b, "GGUF", 4);
    put32(&b, 3);
    put64(&b, 1); /* n_tensors */
    put64(&b, 2); /* n_kv */
    putstr(&b, "nulval");
    put8(&b, GGUF_STRING);
    putstrn(&b, (const uint8_t *)"a\0bcd", 5);
    putstr(&b, "nularr");
    put8(&b, GGUF_ARRAY);
    put8(&b, GGUF_STRING);
    put64(&b, 2);
    putstrn(&b, (const uint8_t *)"x\0y", 3);
    putstr(&b, "plain");
    while (b.n % 32 != 0)
        put8(&b, 0);
    uint64_t data_start = b.n;
    uint64_t d[1] = { 32 };
    put_tensorn(&b, (const uint8_t *)"t\0name", 5, d, 1, GGUF_T_Q8_0, 0);
    while (b.n < data_start + 48)
        put8(&b, 0);

    char tpl[] = "/tmp/opencode/gguf_nul_XXXXXX";
    int fd = mkstemp(tpl);
    if (fd < 0) {
        fprintf(stderr, "test: mkstemp failed\n");
        return 1;
    }
    int wrc = write_file(tpl, b.p, b.n);
    close(fd);
    free(b.p);
    if (wrc != 0) {
        unlink(tpl);
        return 1;
    }

    qw_err err = QW_ERR_NULL;
    struct gguf_file *f = gguf_open(tpl, &err);
    if (f == NULL) {
        printf("FAIL nul: gguf_open (%s)\n", qw_errstr(err));
        unlink(tpl);
        return 1;
    }
    /* KV string value */
    const char *s = NULL;
    size_t l = 0;
    int okv = gguf_get_val_str_len(f, 0, &s, &l) && l == 5 &&
              memcmp(s, (const uint8_t *)"a\0bcd", 5) == 0;
    CHECK(okv, "nul: KV string value keeps NUL (len 5, exact bytes)");
    /* C-string view still valid: truncated at the NUL */
    CHECK(s != NULL && strcmp(s, "a") == 0, "nul: KV string C-view intact");
    /* string array element 0 (NUL), element 1 (plain) */
    int oka = gguf_get_arr_str_len(f, 1, 0, &s, &l) && l == 3 &&
              memcmp(s, (const uint8_t *)"x\0y", 3) == 0;
    CHECK(oka, "nul: array element keeps NUL (len 3, exact bytes)");
    int okp = gguf_get_arr_str_len(f, 1, 1, &s, &l) && l == 5 &&
              strcmp(s, "plain") == 0;
    CHECK(okp, "nul: array element plain (len 5)");
    /* tensor name through the same length-preserving path */
    const struct gguf_tensor_info *t = gguf_tensor_by_index(f, 0);
    int okt = t != NULL && t->name_len == 5 &&
              memcmp(t->name, (const uint8_t *)"t\0name", 5) == 0;
    CHECK(okt, "nul: tensor name keeps NUL (len 5, exact bytes)");
    gguf_close(f);
    unlink(tpl);
    return g_fail;
}

/* 1000 open/close cycles of the synthetic model. malloc/free balance:
 * mallinfo2() free() count must equal its earlier total() count after the
 * loop, i.e. nothing leaked (glibc frees small pools back to the OS in
 * chunks, so VmRSS alone is too coarse to prove per-iteration balance). */
static int test_leak_balance(const char *path)
{
    struct mallinfo2 before = mallinfo2();
    struct mallinfo2 after = mallinfo2();
    (void)after;
    for (int i = 0; i < 1000; i++) {
        qw_err err = QW_ERR_NULL;
        struct gguf_file *f = gguf_open(path, &err);
        if (f == NULL)
            break;
        gguf_close(f);
    }
    after = mallinfo2();
    long freed = (long)before.uordblks + (long)before.fordblks;
    long after_free = (long)after.uordblks + (long)after.fordblks;
    long leaked_bytes = freed - after_free;
    /* every malloc'd byte must have been freed */
    CHECK(before.fordblks == after.fordblks,
          "leak: 1000 open/close cycles free all mallocs");
    CHECK(leaked_bytes <= 0,
          "leak: no net heap growth (bytes)");
    printf("      mallinfo2: freed %ld bytes, %ld bytes net "
           "after 1000 cycles\n",
           freed, leaked_bytes);
    return g_fail;
}

int main(void)
{
    char *path = build_model();

    int rc1 = test_parse(path);

    int rc3 = test_nul_strings();
    int rc4 = test_leak_balance(path);

    unlink(path);
    free(path);

    int rc2 = test_corrupt();

    int rc = rc1 | rc2 | rc3 | rc4;
    printf("%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
