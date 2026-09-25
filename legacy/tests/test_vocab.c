/* tests/test_vocab.c — unit tests for the Qwen vocab loader (qw/vocab.h).
 * No external test framework: plain assert-style CHECK, PASS/FAIL per case,
 * main() returns the failure count (0 = all pass).
 *
 * Builds a SMALL synthetic GGUF v3 in memory (reusing the byte-writing
 * approach from tests/test_gguf.c), writes it to a temp file via mkstemp,
 * and loads it with qw_vocab_from_gguf. Covers the 12-token happy path,
 * a duplicate-token file (rejected), a missing-tokens file (rejected, key
 * named), and a non-"gpt2" model (rejected).
 *
 * NOTE on tokens: the GGUF reader NUL-terminates every string array element
 * (xstrndup), so a token cannot actually embed a NUL byte through this
 * reader — it would be truncated at the NUL. The spec's "NUL byte" case is
 * therefore represented by a token with an embedded NON-NUL control byte
 * (0x01) plus a raw high byte (0xFF): the same invariant that matters for
 * byte-level BPE (the returned pointer must reproduce the exact original
 * bytes, byte-for-byte, independent of C-string semantics).
 */
#define _POSIX_C_SOURCE 200809L /* mkstemp, strdup */

#include "qw/gguf.h"
#include "qw/types.h"
#include "qw/vocab.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------- writer utils */
/* (byte-writing approach from tests/test_gguf.c) */
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
/* length-prefixed raw bytes (unlike putstr which uses strlen). */
static void putstrn(struct bw *b, const void *s, size_t n)
{
    put64(b, (uint64_t)n);
    put(b, s, n);
}
static void putstr(struct bw *b, const char *s)
{
    putstrn(b, s, strlen(s));
}

/* KV: key, then value type + value. */
static void kv_key(struct bw *b, const char *k) { putstr(b, k); }
static void kv_str(struct bw *b, const char *k, const char *s)
{
    kv_key(b, k);
    put8(b, GGUF_STRING);
    putstr(b, s);
}
static void kv_u32(struct bw *b, const char *k, uint32_t v)
{
    kv_key(b, k);
    put8(b, GGUF_UINT32);
    put32(b, v);
}
static void kv_bool(struct bw *b, const char *k, int v)
{
    kv_key(b, k);
    put8(b, GGUF_BOOL);
    put8(b, (uint8_t)(v ? 1 : 0));
}
/* array of N strings (raw length-prefixed bytes). */
static void kv_strarr(struct bw *b, const char *k,
                      const char *const *s, size_t n)
{
    kv_key(b, k);
    put8(b, GGUF_ARRAY);
    put8(b, GGUF_STRING);
    put64(b, (uint64_t)n);
    for (size_t i = 0; i < n; i++) {
        size_t l = strlen(s[i]);
        putstrn(b, s[i], l);
    }
}
/* array of N int32. */
static void kv_i32arr(struct bw *b, const char *k, const int32_t *v, size_t n)
{
    kv_key(b, k);
    put8(b, GGUF_ARRAY);
    put8(b, GGUF_INT32);
    put64(b, (uint64_t)n);
    for (size_t i = 0; i < n; i++)
        put32(b, (uint32_t)v[i]);
}
/* array of N float32. */
static void kv_f32arr(struct bw *b, const char *k, const float *v, size_t n)
{
    kv_key(b, k);
    put8(b, GGUF_ARRAY);
    put8(b, GGUF_FLOAT32);
    put64(b, (uint64_t)n);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &v[i], 4);
        put32(b, bits);
    }
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
            g_fail++;                                               \
        }                                                           \
    } while (0)

/* The 12 hand-written tokens. id 8 is the special <|eoi|> (type USER_DEFINED);
 * id 9 is CONTROL; id 10 is the single raw byte 0x01 (type BYTE); id 11 is a
 * token with an embedded control byte 0x01 and high byte 0xFF (type BYTE) —
 * the "raw bytes" stand-in for the NUL-byte case (see file header). */
static const char *const k_toks[12] = {
    "",      /* 0  empty */
    "a",     /* 1 */
    "b",     /* 2 */
    "c",     /* 3 */
    "ab",    /* 4 */
    "bc",    /* 5 */
    " ",     /* 6 */
    "x",     /* 7 */
    "<|eoi|>", /* 8 special (USER_DEFINED) */
    "\n",    /* 9 CONTROL */
    "\x01",  /* 10 BYTE: single raw byte 0x01 */
    "a\x01\xff", /* 11 BYTE: raw bytes a,0x01,0xFF */
};
static const int32_t k_ttype[12] = {
    QW_TOK_NORMAL, QW_TOK_NORMAL, QW_TOK_NORMAL, QW_TOK_NORMAL,
    QW_TOK_NORMAL, QW_TOK_NORMAL, QW_TOK_NORMAL, QW_TOK_NORMAL,
    QW_TOK_USER_DEFINED, QW_TOK_CONTROL, QW_TOK_BYTE, QW_TOK_BYTE,
};
static const float k_scores[12] = {
    0.0f, 1.5f, -2.0f, 3.25f, 0.5f, -1.25f, 2.0f,
    -3.5f, 10.0f, -0.5f, 0.125f, -7.75f,
};
static const char *const k_merges[3] = { "a b", "ab c", "ab c" };

/* Build the happy-path model: all tokenizer keys present, 12 unique tokens. */
static void build_ok(struct bw *b)
{
    put(b, "GGUF", 4);
    put32(b, 3);          /* version */
    put64(b, 0);          /* n_tensors */
    put64(b, 12);         /* n_kv */

    kv_str(b, "tokenizer.ggml.model", "gpt2");
    kv_strarr(b, "tokenizer.ggml.tokens", k_toks, 12);
    kv_i32arr(b, "tokenizer.ggml.token_type", k_ttype, 12);
    kv_f32arr(b, "tokenizer.ggml.scores", k_scores, 12);
    kv_bool(b, "tokenizer.ggml.add_bos_token", 1);
    kv_u32(b, "tokenizer.ggml.bos_token_id", 0);
    kv_u32(b, "tokenizer.ggml.eos_token_id", 8);
    kv_u32(b, "tokenizer.ggml.padding_token_id", 1);
    kv_u32(b, "tokenizer.ggml.unk_token_id", 2);
    kv_str(b, "tokenizer.ggml.pre", "qwen2");
    kv_strarr(b, "tokenizer.ggml.merges", k_merges, 3);

    /* no tensors: pad KV section to 32, then nothing else. */
    while (b->n % 32 != 0)
        put8(b, 0);
}

/* Build a file that is like build_ok but `drop_model` omits the model key
 * (or sets it to `model_val` when drop_model is false and model_val set).
 * `drop_tokens` omits the tokens key. `dup` inserts a duplicate of toks[1]. */
static void build_variant(struct bw *b, int drop_tokens, int drop_model,
                          const char *model_val, int dup)
{
    /* count kvs we will emit (11 keys by default; dup keeps the count, it
     * only changes one token's bytes) */
    int nkv = 11;
    if (drop_tokens) nkv--;
    if (drop_model)  nkv--;

    put(b, "GGUF", 4);
    put32(b, 3);
    put64(b, 0);
    put64(b, (uint64_t)nkv);

    if (!drop_model)
        kv_str(b, "tokenizer.ggml.model", model_val != NULL ? model_val : "gpt2");
    if (!drop_tokens) {
        if (dup) {
            /* duplicate: emit toks but with entry 5 == entry 1 ("a") */
            const char *toks_dup[12];
            for (int i = 0; i < 12; i++)
                toks_dup[i] = k_toks[i];
            toks_dup[5] = "a"; /* == toks[1] */
            kv_strarr(b, "tokenizer.ggml.tokens", toks_dup, 12);
        } else {
            kv_strarr(b, "tokenizer.ggml.tokens", k_toks, 12);
        }
    }
    kv_i32arr(b, "tokenizer.ggml.token_type", k_ttype, 12);
    kv_f32arr(b, "tokenizer.ggml.scores", k_scores, 12);
    kv_bool(b, "tokenizer.ggml.add_bos", 1);
    kv_u32(b, "tokenizer.ggml.bos_token_id", 0);
    kv_u32(b, "tokenizer.ggml.eos_token_id", 8);
    kv_u32(b, "tokenizer.ggml.padding_token_id", 1);
    kv_u32(b, "tokenizer.ggml.unk_token_id", 2);
    kv_str(b, "tokenizer.ggml.pre", "qwen2");
    kv_strarr(b, "tokenizer.ggml.merges", k_merges, 3);

    while (b->n % 32 != 0)
        put8(b, 0);
}

static char *to_tmp(struct bw *b, const char *tpl)
{
    char t[64];
    strncpy(t, tpl, sizeof(t) - 1);
    t[sizeof(t) - 1] = '\0';
    int fd = mkstemp(t);
    if (fd < 0) {
        fprintf(stderr, "test: mkstemp failed\n");
        exit(2);
    }
    int rc = write_file(t, b->p, b->n);
    close(fd);
    if (rc != 0) {
        unlink(t);
        fprintf(stderr, "test: write_file failed\n");
        exit(2);
    }
    char *p = strdup(t);
    if (p == NULL) {
        fprintf(stderr, "test: OOM\n");
        exit(2);
    }
    return p;
}

/* ------------------------------------------------------------------ main */
int main(void)
{
    /* ---------- happy path ---------- */
    {
        struct bw b; memset(&b, 0, sizeof(b));
        build_ok(&b);
        char *path = to_tmp(&b, "/tmp/opencode/vocab_ok_XXXXXX");
        free(b.p);

        qw_err err = QW_OK;
        struct gguf_file *f = gguf_open(path, &err);
        CHECK(f != NULL, "ok: gguf_open");
        if (f != NULL) {
            qw_vocab v;
            memset(&v, 0, sizeof(v));
            qw_err ve = qw_vocab_from_gguf(f, &v);
            CHECK(ve == QW_OK, "ok: vocab load");

            if (ve == QW_OK) {
                CHECK(v.n_vocab == 12, "ok: n_vocab == 12");

                /* tokens / scores / token_type round-trip */
                int rt_ok = 1;
                for (uint32_t i = 0; i < 12; i++) {
                    const char *ptr = NULL;
                    uint32_t len = 0;
                    if (qw_vocab_id_to_token(&v, i, &ptr, &len) != QW_OK) {
                        rt_ok = 0;
                        break;
                    }
                    size_t ol = strlen(k_toks[i]);
                    if (len != (uint32_t)ol ||
                        memcmp(ptr, k_toks[i], ol) != 0) {
                        rt_ok = 0;
                        break;
                    }
                    if (v.idx[i].type != (uint8_t)k_ttype[i]) {
                        rt_ok = 0;
                        break;
                    }
                    if (v.scores[i] != k_scores[i]) {
                        rt_ok = 0;
                        break;
                    }
                }
                CHECK(rt_ok, "ok: tokens+scores+type round-trip");

                /* raw-bytes token: id 11 must reproduce a,0x01,0xFF exactly */
                const char *rptr = NULL;
                uint32_t rlen = 0;
                if (qw_vocab_id_to_token(&v, 11, &rptr, &rlen) == QW_OK &&
                    rlen == 3) {
                    uint8_t expect[3] = { 'a', 0x01, 0xFF };
                    CHECK(memcmp(rptr, expect, 3) == 0,
                          "ok: raw-bytes token reproduces exact bytes");
                } else {
                    CHECK(0, "ok: raw-bytes token id 11 len==3");
                }

                /* token_to_id finds every token */
                int found = 1;
                for (uint32_t i = 0; i < 12; i++) {
                    size_t l = strlen(k_toks[i]);
                    uint32_t id = 0;
                    if (qw_vocab_token_to_id(&v, k_toks[i], (uint32_t)l, &id) !=
                            QW_OK || id != i)
                        found = 0;
                }
                CHECK(found, "ok: token_to_id finds every token");

                /* token_to_id misses a non-token */
                uint32_t id = 0;
                CHECK(qw_vocab_token_to_id(&v, "zzz", 3, &id) != QW_OK,
                      "ok: token_to_id misses non-token");
                /* empty string is token 0, so it must be FOUND */
                CHECK(qw_vocab_token_to_id(&v, "", 0, &id) == QW_OK && id == 0,
                      "ok: empty string is token 0");

                /* is_special: special ids are bos=0, eos=8, pad=1, unk=2 */
                bool sp = false;
                qw_err se = qw_vocab_is_special(&v, 8, &sp);
                CHECK(se == QW_OK && sp, "ok: is_special(8/eos) true");
                qw_vocab_is_special(&v, 0, &sp);
                CHECK(sp == true, "ok: is_special(0/bos) true");
                qw_vocab_is_special(&v, 3, &sp);
                CHECK(sp == false, "ok: is_special(3/normal) false");
                qw_vocab_is_special(&v, 9, &sp);
                CHECK(sp == false, "ok: is_special(9/control) false");

                /* special ids */
                qw_special_ids sids;
                CHECK(qw_vocab_special_ids(&v, &sids) == QW_OK &&
                          sids.bos == 0 && sids.eos == 8 && sids.pad == 1 &&
                          sids.unk == 2,
                      "ok: special ids");

                /* byte lookup: 256 entries, filled, no uninitialized */
                uint32_t b2id[256];
                memset(b2id, 0xDE, sizeof(b2id)); /* poison */
                CHECK(qw_vocab_byte_lookup(&v, b2id) == QW_OK,
                      "ok: byte_lookup returns OK");
                int all_written = 1;
                for (int i = 0; i < 256; i++)
                    if (b2id[i] == 0xDEDEDEDEu)
                        all_written = 0;
                CHECK(all_written, "ok: byte_lookup writes all 256");
                CHECK(b2id[0x01] == 10, "ok: byte 0x01 -> id 10");
                CHECK(b2id['a'] == QW_TOK_NONE, "ok: 'a' has no 1-byte token");
                /* only token 10 is a type-BYTE with exactly 1 byte (0x01);
                 * token 11 is type BYTE but len 3, so it maps nothing. */
                int none_count = 0;
                for (int i = 0; i < 256; i++)
                    if (b2id[i] == QW_TOK_NONE)
                        none_count++;
                CHECK(none_count == 255, "ok: 255 bytes unmapped");

                /* merges present, stored raw, not parsed
                 * ("a b"+"ab c"+"ab c" = 3+4+4 = 11 bytes) */
                CHECK(v.merges != NULL && v.merges_len == 11,
                      "ok: merges stored (len 11)");

                qw_vocab_report(&v);
                qw_vocab_free(&v);
            }
            gguf_close(f);
        }
        unlink(path);
        free(path);
    }

    /* ---------- duplicate token rejected ---------- */
    {
        struct bw b; memset(&b, 0, sizeof(b));
        build_variant(&b, 0, 0, NULL, 1);
        char *path = to_tmp(&b, "/tmp/opencode/vocab_dup_XXXXXX");
        free(b.p);

        qw_err err = QW_OK;
        struct gguf_file *f = gguf_open(path, &err);
        int rejected = 0;
        if (f != NULL) {
            qw_vocab v; memset(&v, 0, sizeof(v));
            if (qw_vocab_from_gguf(f, &v) != QW_OK)
                rejected = 1;
            qw_vocab_free(&v);
            gguf_close(f);
        }
        CHECK(rejected, "dup: duplicate token rejected");
        unlink(path);
        free(path);
    }

    /* ---------- missing tokens rejected (key named) ---------- */
    {
        struct bw b; memset(&b, 0, sizeof(b));
        build_variant(&b, 1, 0, NULL, 0);
        char *path = to_tmp(&b, "/tmp/opencode/vocab_notok_XXXXXX");
        free(b.p);

        qw_err err = QW_OK;
        struct gguf_file *f = gguf_open(path, &err);
        int rejected = 0;
        if (f != NULL) {
            qw_vocab v; memset(&v, 0, sizeof(v));
            qw_err ve = qw_vocab_from_gguf(f, &v);
            rejected = (ve != QW_OK);
            qw_vocab_free(&v);
            gguf_close(f);
        }
        CHECK(rejected, "notok: missing tokenizer.ggml.tokens rejected");
        unlink(path);
        free(path);
    }

    /* ---------- non-gpt2 model rejected ---------- */
    {
        struct bw b; memset(&b, 0, sizeof(b));
        build_variant(&b, 0, 0, "llama", 0);
        char *path = to_tmp(&b, "/tmp/opencode/vocab_model_XXXXXX");
        free(b.p);

        qw_err err = QW_OK;
        struct gguf_file *f = gguf_open(path, &err);
        int rejected = 0;
        if (f != NULL) {
            qw_vocab v; memset(&v, 0, sizeof(v));
            qw_err ve = qw_vocab_from_gguf(f, &v);
            rejected = (ve != QW_OK);
            qw_vocab_free(&v);
            gguf_close(f);
        }
        CHECK(rejected, "model: non-gpt2 tokenizer.ggml.model rejected");
        unlink(path);
        free(path);
    }

    printf("%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail;
}
