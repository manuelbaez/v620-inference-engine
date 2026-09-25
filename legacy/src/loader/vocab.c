/* src/loader/vocab.c — Qwen byte-level-BPE vocabulary (see qw/vocab.h).
 *
 * Implementation notes
 * ---------------------
 * * Only tokenizer.ggml.tokens is required; every other key is optional and
 *   handled explicitly (named in the error when mandatory-and-wrong, or
 *   defaulted when truly optional). Nothing is silently guessed.
 * * Token strings are copied into one contiguous pool (no NUL padding); the
 *   index array keeps offset/len/score/type. byte-level BPE tokens may
 *   contain NUL and other raw bytes, so all lengths come from the gguf
 *   reader's length-returning accessors, never strlen.
 * * The id<->token hash is open addressing (linear probe) over the exact
 *   byte range, built once at load so the encode hot path is a single fnv1a
 *   + compare. cap is a power of two >= 2*n_vocab so the probe stays short.
 */
#include "qw/vocab.h"
#include "qw/gguf.h"
#include "qw/macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------- consts */
#define VOCAB_EXPECTED 248320u
#define HASH_EMPTY     0xFFFFFFFFu
#define HASH_INIT_CAP  16u

/* GGUF tokenizer key names. */
#define K_MODEL   "tokenizer.ggml.model"
#define K_TOKENS  "tokenizer.ggml.tokens"
#define K_TTYPE   "tokenizer.ggml.token_type"
#define K_SCORES  "tokenizer.ggml.scores"
#define K_ADD_BOS "tokenizer.ggml.add_bos_token"
#define K_BOS     "tokenizer.ggml.bos_token_id"
#define K_EOS     "tokenizer.ggml.eos_token_id"
#define K_PAD     "tokenizer.ggml.padding_token_id"
#define K_UNK     "tokenizer.ggml.unk_token_id"
#define K_PRE     "tokenizer.ggml.pre"
#define K_MERGES  "tokenizer.ggml.merges"

/* ----------------------------------------------------------------- alloc */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (p == NULL)
        fprintf(stderr, "qw/vocab: out of memory\n");
    return p;
}

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (p == NULL)
        fprintf(stderr, "qw/vocab: out of memory\n");
    return p;
}

static char *xstrdup(const char *s)
{
    if (s == NULL)
        return NULL;
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    if (p == NULL)
        return NULL;
    memcpy(p, s, n + 1);
    return p;
}

/* --------------------------------------------------------------- lookup */
static int find_kv(const struct gguf_file *f, const char *key)
{
    for (uint64_t i = 0; i < gguf_n_kv(f); i++)
        if (strcmp(gguf_get_key(f, i), key) == 0)
            return (int)i;
    return -1;
}

/* fnv1a over the exact byte range [s, s+len). */
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

/* --------------------------------------------------------------- struct */
static void qw_vocab_init(qw_vocab *v)
{
    memset(v, 0, sizeof(*v));
    v->sp.bos = v->sp.eos = v->sp.pad = v->sp.unk = -1;
}

/* Free the vocab's own allocations (does NOT free the gguf file). */
static void qw_vocab_reset(qw_vocab *v)
{
    if (v == NULL)
        return;
    free(v->toks);
    free(v->idx);
    free(v->scores);
    free(v->pre);
    free(v->merges);
    free(v->tok_hash);
    qw_vocab_init(v);
}

/* ------------------------------------------------------------- hash cap */
/* Smallest power of two >= need, and >= HASH_INIT_CAP. */
static uint32_t hash_cap_for(uint32_t need)
{
    uint32_t cap = HASH_INIT_CAP;
    while (cap < need)
        cap <<= 1;
    return cap;
}

/* ---------------------------------------------------------- qwen family */
/* True if the pre-tokenizer name looks like a Qwen variant. */
static bool qwen_pre(const char *s)
{
    return s != NULL && s[0] == 'q' && s[1] == 'w' && s[2] == 'e' &&
           s[3] == 'n';
}

/* ------------------------------------------------------- special id read */
/* Read an optional non-negative integer special-token id. On success sets
 * *present true and *val to the id. Negative values are rejected (they
 * cannot be a token id) and reported by key name. */
static qw_err read_special(const struct gguf_file *f, const char *key,
                           int32_t *val, bool *present)
{
    *present = false;
    int i = find_kv(f, key);
    if (i < 0)
        return QW_OK; /* absent: default -1 */
    if (f != NULL && gguf_get_kv_type(f, (size_t)i) == GGUF_UINT32) {
        int x = gguf_get_val_u32(f, (size_t)i);
        if (x < 0) {
            QW_LOGE("qw/vocab: %s is negative (%d)", key, x);
            return QW_ERR_FORMAT;
        }
        *val = (int32_t)x;
        *present = true;
        return QW_OK;
    }
    if (gguf_get_kv_type(f, (size_t)i) == GGUF_INT32) {
        int x = gguf_get_val_i32(f, (size_t)i);
        if (x < 0) {
            QW_LOGE("qw/vocab: %s is negative (%d)", key, x);
            return QW_ERR_FORMAT;
        }
        *val = (int32_t)x;
        *present = true;
        return QW_OK;
    }
    QW_LOGE("qw/vocab: %s present but not a non-negative integer", key);
    return QW_ERR_FORMAT;
}

/* ---------------------------------------------------------- from gguf */
qw_err qw_vocab_from_gguf(const struct gguf_file *f, qw_vocab *out)
{
    if (out == NULL)
        return QW_ERR_NULL;
    qw_vocab_init(out);
    if (f == NULL)
        return QW_ERR_NULL;

    qw_err err = QW_OK;
    qw_vocab_tok *idx = NULL;
    uint8_t *toks = NULL;
    float *scores = NULL;
    uint8_t *merges = NULL;
    char *pre = NULL;
    uint32_t *hash = NULL;
    uint64_t pool_bytes = 0;
    uint32_t n = 0;

    /* ---- mandatory: tokenizer.ggml.tokens (array of string) ---- */
    int ti = find_kv(f, K_TOKENS);
    if (ti < 0) {
        QW_LOGE("qw/vocab: missing mandatory key %s", K_TOKENS);
        return QW_ERR_FORMAT;
    }
    if (gguf_get_kv_type(f, (size_t)ti) != GGUF_ARRAY ||
        gguf_get_arr_type(f, (size_t)ti) != GGUF_STRING) {
        QW_LOGE("qw/vocab: %s must be an array of string", K_TOKENS);
        return QW_ERR_FORMAT;
    }
    n = (uint32_t)gguf_get_arr_n(f, (size_t)ti);
    if (n == 0) {
        QW_LOGE("qw/vocab: %s is empty", K_TOKENS);
        return QW_ERR_FORMAT;
    }

    /* ---- tokenizer.ggml.model must be the string "gpt2" ---- */
    int mi = find_kv(f, K_MODEL);
    if (mi < 0) {
        QW_LOGE("qw/vocab: missing mandatory key %s (byte-level BPE)", K_MODEL);
        return QW_ERR_FORMAT;
    }
    const char *model = gguf_get_val_str(f, (size_t)mi);
    if (model == NULL || strcmp(model, "gpt2") != 0) {
        QW_LOGE("qw/vocab: %s must be \"gpt2\" (got \"%s\")",
                K_MODEL, model != NULL ? model : "(non-string)");
        return QW_ERR_FORMAT;
    }

    /* ---- pass 1: measure the pool (exact lengths: tokens may embed NUL) ---- */
    for (uint32_t i = 0; i < n; i++) {
        const char *s;
        size_t l;
        if (!gguf_get_arr_str_len(f, (size_t)ti, i, &s, &l)) {
            QW_LOGE("qw/vocab: %s[%u] not a string", K_TOKENS, i);
            return QW_ERR_FORMAT;
        }
        pool_bytes += (uint64_t)l;
    }

    /* ---- allocate pool + index ---- */
    toks = (uint8_t *)xmalloc((size_t)pool_bytes);
    idx = (qw_vocab_tok *)xcalloc(n, sizeof(qw_vocab_tok));
    scores = (float *)xcalloc(n, sizeof(float));
    if (toks == NULL || idx == NULL || scores == NULL) {
        err = QW_ERR_ALLOC;
        goto fail;
    }

    /* ---- pass 2: fill the pool, copy bytes, record offset/len ---- */
    {
        uint32_t off = 0;
        for (uint32_t i = 0; i < n; i++) {
            const char *s;
            size_t l;
            gguf_get_arr_str_len(f, (size_t)ti, i, &s, &l);
            if ((uint32_t)l > UINT32_MAX - off) {
                QW_LOGE("qw/vocab: pool offset overflow");
                err = QW_ERR_FORMAT;
                goto fail;
            }
            memcpy(toks + off, s, l);
            idx[i].off = off;
            idx[i].len = (uint32_t)l;
            off += (uint32_t)l;
        }
    }

    /* ---- token_type (optional; default NORMAL, validate range) ---- */
    {
        int x = find_kv(f, K_TTYPE);
        if (x >= 0) {
            if (gguf_get_kv_type(f, (size_t)x) != GGUF_ARRAY ||
                gguf_get_arr_type(f, (size_t)x) != GGUF_INT32) {
                QW_LOGE("qw/vocab: %s must be an array of int32", K_TTYPE);
                err = QW_ERR_FORMAT;
                goto fail;
            }
            if (gguf_get_arr_n(f, (size_t)x) != (uint64_t)n) {
                QW_LOGE("qw/vocab: %s length %llu != %u tokens",
                        K_TTYPE,
                        (unsigned long long)gguf_get_arr_n(f, (size_t)x), n);
                err = QW_ERR_FORMAT;
                goto fail;
            }
            for (uint32_t i = 0; i < n; i++) {
                int t = gguf_arr_get_i32(f, (size_t)x, i);
                if (t != QW_TOK_NORMAL && t != QW_TOK_UNKNOWN &&
                    t != QW_TOK_CONTROL && t != QW_TOK_USER_DEFINED &&
                    t != QW_TOK_BYTE) {
                    QW_LOGE("qw/vocab: %s[%u] has unknown code %d",
                            K_TTYPE, i, t);
                    err = QW_ERR_FORMAT;
                    goto fail;
                }
                idx[i].type = (uint8_t)t;
            }
        } else {
            for (uint32_t i = 0; i < n; i++)
                idx[i].type = QW_TOK_NORMAL;
        }
    }

    /* ---- scores (optional; must match tokens length when present) ---- */
    {
        int x = find_kv(f, K_SCORES);
        if (x >= 0) {
            if (gguf_get_kv_type(f, (size_t)x) != GGUF_ARRAY ||
                gguf_get_arr_type(f, (size_t)x) != GGUF_FLOAT32) {
                QW_LOGE("qw/vocab: %s must be an array of float32", K_SCORES);
                err = QW_ERR_FORMAT;
                goto fail;
            }
            if (gguf_get_arr_n(f, (size_t)x) != (uint64_t)n) {
                QW_LOGE("qw/vocab: %s length %llu != %u tokens",
                        K_SCORES,
                        (unsigned long long)gguf_get_arr_n(f, (size_t)x), n);
                err = QW_ERR_FORMAT;
                goto fail;
            }
            for (uint32_t i = 0; i < n; i++)
                scores[i] = gguf_arr_get_f32(f, (size_t)x, i);
        }
        /* absent: scores stay 0.0 (xcalloc) */
    }

    /* mirror the final scores into idx[].score for a single canonical copy */
    for (uint32_t i = 0; i < n; i++)
        idx[i].score = scores[i];

    /* ---- pre (optional; default "") ---- */
    {
        int x = find_kv(f, K_PRE);
        if (x >= 0) {
            const char *s = gguf_get_val_str(f, (size_t)x);
            if (s == NULL) {
                QW_LOGE("qw/vocab: %s present but not a string", K_PRE);
                err = QW_ERR_FORMAT;
                goto fail;
            }
            pre = xstrdup(s);
            if (pre == NULL) {
                err = QW_ERR_ALLOC;
                goto fail;
            }
            if (!qwen_pre(s))
                QW_LOGW("qw/vocab: %s \"%s\" is not a qwen-family value",
                        K_PRE, s);
        } else {
            pre = xstrdup("");
            if (pre == NULL) {
                err = QW_ERR_ALLOC;
                goto fail;
            }
        }
    }

    /* ---- merges (optional; store the raw blob, do NOT parse) ---- */
    {
        int x = find_kv(f, K_MERGES);
        if (x >= 0) {
            if (gguf_get_kv_type(f, (size_t)x) != GGUF_ARRAY ||
                gguf_get_arr_type(f, (size_t)x) != GGUF_STRING) {
                QW_LOGE("qw/vocab: %s must be an array of string", K_MERGES);
                err = QW_ERR_FORMAT;
                goto fail;
            }
            uint64_t m = gguf_get_arr_n(f, (size_t)x);
            uint64_t total = 0;
            for (uint64_t i = 0; i < m; i++) {
                const char *s;
                size_t l;
                if (!gguf_get_arr_str_len(f, (size_t)x, i, &s, &l)) {
                    QW_LOGE("qw/vocab: %s[%llu] not a string",
                            K_MERGES, (unsigned long long)i);
                    err = QW_ERR_FORMAT;
                    goto fail;
                }
                total += (uint64_t)l;
            }
            if (total > 0) {
                merges = (uint8_t *)xmalloc((size_t)total);
                if (merges == NULL) {
                    err = QW_ERR_ALLOC;
                    goto fail;
                }
                uint64_t o = 0;
                for (uint64_t i = 0; i < m; i++) {
                    const char *s;
                    size_t l;
                    gguf_get_arr_str_len(f, (size_t)x, i, &s, &l);
                    memcpy(merges + o, s, l);
                    o += (uint64_t)l;
                }
            }
            out->merges_len = total;
        }
    }

    /* ---- special-token ids (optional) ---- */
    {
        bool p;
        err = read_special(f, K_BOS, &out->sp.bos, &p);
        if (err != QW_OK)
            goto fail;
        if (p)
            out->has_special = true;
        err = read_special(f, K_EOS, &out->sp.eos, &p);
        if (err != QW_OK)
            goto fail;
        if (p)
            out->has_special = true;
        err = read_special(f, K_PAD, &out->sp.pad, &p);
        if (err != QW_OK)
            goto fail;
        if (p)
            out->has_special = true;
        err = read_special(f, K_UNK, &out->sp.unk, &p);
        if (err != QW_OK)
            goto fail;
        if (p)
            out->has_special = true;
    }

    /* ---- add_bos (optional; default false) ---- */
    {
        int x = find_kv(f, K_ADD_BOS);
        if (x >= 0) {
            if (gguf_get_kv_type(f, (size_t)x) != GGUF_BOOL) {
                QW_LOGE("qw/vocab: %s must be a bool", K_ADD_BOS);
                err = QW_ERR_FORMAT;
                goto fail;
            }
            out->add_bos = gguf_get_val_bool(f, (size_t)x);
        }
    }

    /* ---- publish what we built ---- */
    out->n_vocab = n;
    out->toks = toks;
    out->idx = idx;
    out->scores = scores;
    out->pre = pre;
    out->merges = merges;

    /* ---- per-type counts ---- */
    for (uint32_t i = 0; i < n; i++)
        out->type_count[idx[i].type]++;

    /* ---- build the id<->token hash ----
     * Slots must be HASH_EMPTY (0xFFFFFFFF), NOT 0: token id 0 is a valid
     * occupant, so a zero-filled table would look full and probes would
     * never terminate. malloc + explicit fill. */
    {
        uint32_t cap = hash_cap_for(n * 2);
        hash = (uint32_t *)xmalloc(cap * sizeof(uint32_t));
        if (hash == NULL) {
            err = QW_ERR_ALLOC;
            goto fail;
        }
        for (uint32_t i = 0; i < cap; i++)
            hash[i] = HASH_EMPTY;
        for (uint32_t i = 0; i < n; i++) {
            const uint8_t *s = toks + idx[i].off;
            uint64_t h = fnv1a_len(s, idx[i].len) & (cap - 1);
            /* probe: an empty slot is claimed; an already-inserted token
             * with identical bytes is a DUPLICATE (error); any other token
             * is just a hash collision, keep probing. cap >= 2*n guarantees
             * an empty slot is always reachable, so this terminates. */
            for (;;) {
                uint32_t j = hash[h];
                if (j == HASH_EMPTY) {
                    hash[h] = i;
                    break;
                }
                if (idx[j].len == idx[i].len &&
                    memcmp(toks + idx[j].off, s, idx[i].len) == 0) {
                    QW_LOGE("qw/vocab: duplicate token at id %u (== id %u)",
                            i, j);
                    err = QW_ERR_FORMAT;
                    goto fail;
                }
                h = (h + 1) & (cap - 1);
            }
        }
        out->tok_hash_cap = cap;
    }
    out->tok_hash = hash;

    /* ---- validate: expected n_vocab (warn only) ---- */
    if (n != VOCAB_EXPECTED)
        QW_LOGW("qw/vocab: n_vocab %u != expected %u (different checkpoint?)",
                n, VOCAB_EXPECTED);

    QW_LOGI("qw/vocab: loaded %u tokens, pre=%s, merges_len=%llu",
            n, pre, (unsigned long long)out->merges_len);
    return QW_OK;

fail:
    /* `out` may already alias one of these locals (the publish step runs
     * before the duplicate check), so clear it before freeing: the caller
     * may then qw_vocab_free a fully-zero struct without double-freeing. */
    qw_vocab_init(out);
    free(toks);
    free(idx);
    free(scores);
    free(merges);
    free(pre);
    free(hash);
    return err;
}

/* ------------------------------------------------------------ id to tok */
qw_err qw_vocab_id_to_token(const qw_vocab *v, uint32_t id, const char **ptr,
                            uint32_t *len)
{
    if (v == NULL || ptr == NULL || len == NULL)
        return QW_ERR_NULL;
    if (id >= v->n_vocab)
        return QW_ERR_RANGE;
    *ptr = (const char *)(v->toks + v->idx[id].off);
    *len = v->idx[id].len;
    return QW_OK;
}

/* ------------------------------------------------------- token to id */
qw_err qw_vocab_token_to_id(const qw_vocab *v, const char *s, uint32_t len,
                            uint32_t *id)
{
    if (v == NULL || s == NULL || id == NULL)
        return QW_ERR_NULL;
    if (v->tok_hash == NULL)
        return QW_ERR_NULL;
    uint32_t cap = v->tok_hash_cap;
    uint64_t h = fnv1a_len(s, len) & (cap - 1);
    for (;;) {
        uint32_t j = v->tok_hash[h];
        if (j == HASH_EMPTY)
            return QW_ERR_RANGE; /* miss */
        if (v->idx[j].len == len &&
            memcmp(v->toks + v->idx[j].off, s, len) == 0) {
            *id = j;
            return QW_OK;
        }
        h = (h + 1) & (cap - 1);
    }
}

/* ---------------------------------------------------------- is special */
qw_err qw_vocab_is_special(const qw_vocab *v, uint32_t id, bool *out)
{
    if (v == NULL || out == NULL)
        return QW_ERR_NULL;
    if (id >= v->n_vocab)
        return QW_ERR_RANGE;
    *out = (v->sp.bos == (int32_t)id || v->sp.eos == (int32_t)id ||
            v->sp.pad == (int32_t)id || v->sp.unk == (int32_t)id);
    return QW_OK;
}

qw_err qw_vocab_special_ids(const qw_vocab *v, qw_special_ids *out)
{
    if (v == NULL || out == NULL)
        return QW_ERR_NULL;
    *out = v->sp;
    return QW_OK;
}

/* ---------------------------------------------------------- byte lookup */
qw_err qw_vocab_byte_lookup(const qw_vocab *v, uint32_t *byte_to_id)
{
    if (v == NULL || byte_to_id == NULL)
        return QW_ERR_NULL;
    for (int i = 0; i < 256; i++)
        byte_to_id[i] = QW_TOK_NONE;
    for (uint32_t i = 0; i < v->n_vocab; i++) {
        if (v->idx[i].type != QW_TOK_BYTE || v->idx[i].len != 1)
            continue;
        uint32_t b = (uint32_t)(v->toks[v->idx[i].off] & 0xFF);
        byte_to_id[b] = i;
    }
    return QW_OK;
}

/* ---------------------------------------------------------------- free */
void qw_vocab_free(qw_vocab *v)
{
    qw_vocab_reset(v);
}

/* --------------------------------------------------------------- report */
void qw_vocab_report(const qw_vocab *v)
{
    if (v == NULL) {
        printf("(null vocab)\n");
        return;
    }
    printf("vocab: n_vocab=%u\n", v->n_vocab);
    printf("  tokens by type:");
    for (int t = 1; t < 7; t++)
        if (v->type_count[t] > 0)
            printf("  %s=%u",
                   t == QW_TOK_NORMAL ? "NORMAL" :
                   t == QW_TOK_UNKNOWN ? "UNKNOWN" :
                   t == QW_TOK_CONTROL ? "CONTROL" :
                   t == QW_TOK_USER_DEFINED ? "USER_DEFINED" : "BYTE",
                   v->type_count[t]);
    printf("\n");
    printf("  add_bos=%s  bos=%d  eos=%d  pad=%d  unk=%d\n",
           v->add_bos ? "true" : "false", v->sp.bos, v->sp.eos,
           v->sp.pad, v->sp.unk);
    printf("  pre=\"%s\"  merges_present=%s (%llu bytes)\n",
           v->pre != NULL ? v->pre : "",
           v->merges != NULL ? "true" : "false",
           (unsigned long long)v->merges_len);
}
