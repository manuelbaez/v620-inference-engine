/* src/core/bpe.c — byte-level-BPE encode/decode (part 2). See qw/bpe.h for
 * the pipeline, the special-token structural rule, and the complexity note.
 *
 * The merge-table index lives in the caller's qw_arena (or a malloc block
 * when no arena is given); per-encode scratch (token id lists) is plain
 * malloc, checked on every call and freed on every path. No VLA. The input
 * is an opaque byte range (len authoritative) — never read past it, embedded
 * NULs included.
 */
#include "qw/bpe.h"
#include "qw/macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- internals */

/* Grow to the next power of two >= need (need >= 1). 0 on overflow. */
static uint32_t pow2_ge(size_t need)
{
    uint32_t c = 1;
    while (c < (uint32_t)need) {
        if (c > (1u << 30))
            return 0;
        c <<= 1;
    }
    return c;
}

/* Token bytes -> id via the vocab's own hash. QW_ERR_RANGE on miss. */
static qw_err tok2id(const qw_vocab *v, const unsigned char *s, uint32_t len,
                     uint32_t *id)
{
    return qw_vocab_token_to_id(v, (const char *)s, len, id);
}

/* Is id a named special-token id? */
static bool is_special(const qw_vocab *v, uint32_t id)
{
    if (id >= v->n_vocab)
        return false;
    const qw_special_ids *sp = &v->sp;
    return (sp->bos >= 0 && (uint32_t)sp->bos == id) ||
           (sp->eos >= 0 && (uint32_t)sp->eos == id) ||
           (sp->pad >= 0 && (uint32_t)sp->pad == id) ||
           (sp->unk >= 0 && (uint32_t)sp->unk == id);
}

/* ----------------------------------------------------------------- init */

/* The merge index is an open-addressing table keyed by the exact (left id,
 * right id) pair. Each slot stores the pair key, the merge RESULT token id,
 * and rank+1 (rank1 == 0 = empty). Lookup probes until it finds the matching
 * pair key (hit) or an empty slot (miss) — exact under collisions because
 * the key is stored and compared.
 *
 * Storing the result id (not just the rank) is what makes the special-token
 * rule structural: the merge step emits the result id straight from the
 * index, and qw_bpe_init only ever stores a result id that passed the
 * is_special() check — so a lookup can never return a special id. */

static uint32_t merge_hash(uint64_t key, uint32_t mask)
{
    return ((uint32_t)key ^ (uint32_t)(key >> 32)) & mask;
}

static uint64_t pair_key(uint32_t l, uint32_t r)
{
    return ((uint64_t)l << 32) | r;
}

/* Return true (and fill *rank, *res) if the pair is a mergeable pair. */
static bool merge_get(const qw_bpe *bpe, uint32_t l, uint32_t r,
                      uint32_t *rank, uint32_t *res)
{
    if (bpe->rank1 == NULL)
        return false;
    uint32_t mask = bpe->cap - 1;
    uint64_t key = pair_key(l, r);
    uint32_t h = merge_hash(key, mask);
    for (;;) {
        if (bpe->rank1[h] == 0)
            return false;
        if (bpe->pair[h] == key) {
            *rank = bpe->rank1[h] - 1;
            *res = bpe->rid[h];
            return true;
        }
        h = (h + 1) & mask;
    }
}

static bool merge_insert(qw_bpe *bpe, uint32_t l, uint32_t r, uint32_t rank,
                         uint32_t res)
{
    uint32_t mask = bpe->cap - 1;
    uint64_t key = pair_key(l, r);
    uint32_t h = merge_hash(key, mask);
    for (;;) {
        if (bpe->rank1[h] == 0) {
            bpe->pair[h] = key;
            bpe->rid[h] = res;
            bpe->rank1[h] = rank + 1;
            return true;
        }
        h = (h + 1) & mask;
    }
}

qw_err qw_bpe_init(qw_bpe *bpe, const qw_vocab *v, const char **merge_lines,
                   int n_merges, qw_arena *scratch)
{
    if (bpe == NULL || v == NULL)
        return QW_ERR_NULL;
    memset(bpe, 0, sizeof(*bpe));

    if (n_merges == 0)
        return QW_OK; /* byte-fallback only; no index */
    if (n_merges < 0 || merge_lines == NULL)
        return QW_ERR_NULL;

    uint32_t cap = pow2_ge((size_t)n_merges * 2 + 16);
    if (cap == 0)
        return QW_ERR_ALLOC;

    size_t bytes = (size_t)cap * (sizeof(uint64_t) + 2 * sizeof(uint32_t));
    void *mem = (scratch != NULL)
                   ? qw_arena_alloc(scratch, bytes, 8)
                   : calloc(1, bytes);
    if (mem == NULL)
        return QW_ERR_ALLOC;
    /* The arena does not zero; the index relies on rank1==0 meaning "empty",
     * so zero the whole block regardless of which allocator supplied it. */
    memset(mem, 0, bytes);

    bpe->arena = scratch;
    uint8_t *p = (uint8_t *)mem;
    bpe->pair = (uint64_t *)p;
    p += (size_t)cap * sizeof(uint64_t);
    bpe->rid = (uint32_t *)p;
    p += (size_t)cap * sizeof(uint32_t);
    bpe->rank1 = (uint32_t *)p;
    bpe->cap = cap;
    bpe->n_merges = (uint32_t)n_merges;

    for (int i = 0; i < n_merges; i++) {
        const char *line = merge_lines[i];
        if (line == NULL)
            return QW_ERR_NULL;

        /* Parse "A B": A = non-empty non-space run, one space, B = non-empty
         * non-space run, NUL. Single-space separator (GGUF blob rule). */
        const unsigned char *a = (const unsigned char *)line;
        size_t alen = 0;
        while (a[alen] != ' ' && a[alen] != '\0')
            alen++;
        if (a[alen] != ' ')
            return QW_ERR_FORMAT;
        const unsigned char *b = a + alen + 1;
        size_t blen = 0;
        while (b[blen] != '\0')
            blen++;
        if (alen == 0 || blen == 0)
            return QW_ERR_FORMAT;
        for (size_t k = 0; k < blen; k++)
            if (b[k] == ' ')
                return QW_ERR_FORMAT;

        uint32_t ia, ib, ir;
        if (tok2id(v, a, (uint32_t)alen, &ia) != QW_OK)
            return QW_ERR_FORMAT;
        if (tok2id(v, b, (uint32_t)blen, &ib) != QW_OK)
            return QW_ERR_FORMAT;

        size_t rlen = (size_t)alen + blen;
        unsigned char *res = malloc(rlen);
        if (res == NULL)
            return QW_ERR_ALLOC;
        memcpy(res, a, alen);
        memcpy(res + alen, b, blen);
        qw_err e = tok2id(v, res, (uint32_t)rlen, &ir);
        free(res);
        if (e != QW_OK)
            return e; /* result token not in vocab */

        /* STRUCTURAL special-token rule: a merge must never produce a
         * special id; reject such a line so the index can never return one. */
        if (is_special(v, ir))
            return QW_ERR_FORMAT;

        if (!merge_insert(bpe, ia, ib, (uint32_t)i, ir))
            return QW_ERR_FORMAT; /* duplicate pair */
    }
    return QW_OK;
}

void qw_bpe_free(qw_bpe *bpe)
{
    if (bpe == NULL)
        return;
    if (bpe->arena == NULL && bpe->pair != NULL)
        free(bpe->pair);
    memset(bpe, 0, sizeof(*bpe));
}

/* -------------------------------------------------------- byte-level map */

qw_err qw_byte_level_encode(const char *in, size_t n, char *out,
                            size_t out_cap, size_t *n_out)
{
    if (in == NULL || out == NULL || n_out == NULL)
        return QW_ERR_NULL;
    if (out_cap < n * 2) {
        *n_out = n * 2;
        return QW_ERR_RANGE;
    }
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c < 0x80) {
            out[o++] = (char)c;
        } else {
            out[o++] = (char)(0xC0 | (c >> 6));
            out[o++] = (char)(0x80 | (c & 0x3F));
        }
    }
    *n_out = o;
    return QW_OK;
}

qw_err qw_byte_level_decode(const char *s, size_t len, char *out,
                            size_t out_cap, size_t *n_out)
{
    if (s == NULL || out == NULL || n_out == NULL)
        return QW_ERR_NULL;
    if (out_cap < len) {
        *n_out = len;
        return QW_ERR_RANGE;
    }
    size_t o = 0;
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            out[o++] = (char)c;
            i++;
        } else if (c >= 0xC0 && c < 0xE0) {
            if (i + 1 >= len)
                return QW_ERR_RANGE;
            unsigned char t = (unsigned char)s[i + 1];
            if (t < 0x80 || t > 0xBF)
                return QW_ERR_RANGE;
            out[o++] = (char)(((c & 0x1F) << 6) | (t & 0x3F));
            i += 2;
        } else {
            return QW_ERR_RANGE;
        }
    }
    *n_out = o;
    return QW_OK;
}

/* ------------------------------------------------------------ tokenizer */

static bool is_word(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9');
}
static bool is_ws(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f';
}
static bool is_punct(unsigned char c)
{
    return c >= 0x21 && c <= 0x7E && !is_word(c) && !is_ws(c);
}

qw_err qw_bpe_pre_tokenizer(const char *text, size_t len, qw_span *out_spans,
                            int max_spans, int *n_spans_out)
{
    if (text == NULL || out_spans == NULL || n_spans_out == NULL)
        return QW_ERR_NULL;
    if (max_spans < 0)
        return QW_ERR_RANGE;

    int n = 0;
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)text[i];
        size_t start = i;
        if (is_ws(c)) {
            while (i < len && is_ws((unsigned char)text[i]))
                i++;
        } else if (is_word(c)) {
            i++;
            while (i < len && is_word((unsigned char)text[i]))
                i++;
            while (i < len && is_punct((unsigned char)text[i]))
                i++;
        } else {
            i++;
        }
        if (n < max_spans) {
            out_spans[n].start = start;
            out_spans[n].end = i;
        }
        n++;
    }
    *n_spans_out = n;
    if (n > max_spans)
        return QW_ERR_RANGE; /* report the total needed in *n_spans_out */
    return QW_OK;
}

/* ------------------------------------------------------------- bpe core */

/* Merge one pre-token (a chunk). tokens[] = (start,end) into mapped.
 * Writes the merged ids to out_ids and returns the count in *n_out
 * (always <= the seed count). The merged result id is taken straight from
 * the index (never recomputed), so it is structurally non-special. */
static qw_err merge_chunk(const qw_bpe *bpe, const qw_vocab *v,
                          const unsigned char *mapped, const qw_span *tokens,
                          int nt, uint32_t *out_ids, int *n_out)
{
    /* Fast path: the whole chunk is one non-special vocab token. */
    if (nt == 1) {
        uint32_t id;
        uint32_t l = (uint32_t)(tokens[0].end - tokens[0].start);
        if (tok2id(v, mapped + tokens[0].start, l, &id) == QW_OK &&
            !is_special(v, id)) {
            out_ids[0] = id;
            *n_out = 1;
            return QW_OK;
        }
    }

    /* Seed: one id per pre-token; a pre-token that is not a (non-special)
     * vocab token is expanded to its byte tokens. The seed count for a chunk
     * is therefore at most the chunk's total byte length (a word span of N
     * bytes expands to N byte ids), so size the buffer by the sum of the
     * token lengths. +1 slack keeps the merge loop's ids[i+1] read in-bounds. */
    size_t seencap = 0;
    for (int i = 0; i < nt; i++)
        seencap += (size_t)(tokens[i].end - tokens[i].start);
    uint32_t *ids = malloc((seencap + 1) * sizeof(uint32_t));
    if (ids == NULL)
        return QW_ERR_ALLOC;
    int m = 0;
    for (int i = 0; i < nt; i++) {
        const unsigned char *p = mapped + tokens[i].start;
        uint32_t l = (uint32_t)(tokens[i].end - tokens[i].start);
        uint32_t id;
        if (tok2id(v, p, l, &id) == QW_OK && !is_special(v, id)) {
            ids[m++] = id;
        } else if (l == 1) {
            if (tok2id(v, p, 1, &id) != QW_OK) {
                free(ids);
                return QW_ERR_RANGE; /* byte token missing */
            }
            ids[m++] = id;
        } else {
            for (uint32_t k = 0; k < l; k++) {
                uint32_t bid;
                if (tok2id(v, p + k, 1, &bid) != QW_OK) {
                    free(ids);
                    return QW_ERR_RANGE;
                }
                ids[m++] = bid;
            }
        }
    }

    /* Apply merges by rank: lowest rank first, leftmost on ties. Each pass
     * scans the list once (O(m)); the list only shrinks. The result id comes
     * from the index, so it is structurally non-special. */
    for (;;) {
        uint32_t best_rank = QW_TOK_NONE;
        int best_i = -1;
        for (int i = 0; i + 1 < m; i++) {
            uint32_t rank, res;
            (void)res;
            if (!merge_get(bpe, ids[i], ids[i + 1], &rank, &res))
                continue;
            if (best_i < 0 || rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0)
            break;

        uint32_t rank, res;
        merge_get(bpe, ids[best_i], ids[best_i + 1], &rank, &res);
        memmove(&ids[best_i + 1], &ids[best_i + 2],
                (size_t)(m - best_i - 2) * sizeof(uint32_t));
        ids[best_i] = res;
        m--;
    }

    for (int i = 0; i < m; i++)
        out_ids[i] = ids[i];
    *n_out = m;
    free(ids);
    return QW_OK;
}

qw_err qw_bpe_encode(const qw_bpe *bpe, const qw_vocab *v, const char *text,
                     size_t len, uint32_t *out_ids, size_t out_cap,
                     size_t *n_out)
{
    if (bpe == NULL || v == NULL || text == NULL || n_out == NULL)
        return QW_ERR_NULL;

    unsigned char *mapped = malloc((len * 2 != 0) ? len * 2 : 1);
    if (mapped == NULL)
        return QW_ERR_ALLOC;
    size_t mlen = 0;
    if (qw_byte_level_encode(text, len, (char *)mapped, len * 2,
                             &mlen) != QW_OK) {
        free(mapped);
        return QW_ERR_RANGE;
    }

    /* Each pre-tokenized span is one merge chunk (no merge crosses a span
     * boundary), and the spans are concatenated directly — whitespace spans
     * are already separate chunks (Qwen2 keeps whitespace as its own tokens),
     * so no extra "join" token is inserted. The output buffer is sized to the
     * sum of the span byte lengths (an upper bound; merges only shrink). */
    size_t upper = 0;
    for (size_t i = 0; i < mlen; ) {
        size_t start = i;
        unsigned char c = mapped[i];
        if (is_ws(c)) {
            while (i < mlen && is_ws(mapped[i]))
                i++;
        } else if (is_word(c)) {
            i++;
            while (i < mlen && is_word(mapped[i]))
                i++;
            while (i < mlen && is_punct(mapped[i]))
                i++;
        } else {
            i++;
        }
        upper += (i - start);
    }

    uint32_t *buf = malloc((upper ? upper : 1) * sizeof(uint32_t));
    if (buf == NULL) {
        free(mapped);
        return QW_ERR_ALLOC;
    }

    size_t total = 0;
    qw_err err = QW_OK;
    size_t i = 0;
    while (i < mlen && err == QW_OK) {
        size_t start = i;
        unsigned char c = mapped[i];
        if (is_ws(c)) {
            while (i < mlen && is_ws(mapped[i]))
                i++;
        } else if (is_word(c)) {
            i++;
            while (i < mlen && is_word(mapped[i]))
                i++;
            while (i < mlen && is_punct(mapped[i]))
                i++;
        } else {
            i++;
        }
        qw_span s = { start, i };
        int cnt = 0;
        if (merge_chunk(bpe, v, mapped, &s, 1, &buf[total], &cnt) != QW_OK) {
            err = QW_ERR_RANGE;
            break;
        }
        total += (size_t)cnt;
    }

    if (err == QW_OK && (out_ids == NULL || out_cap < total)) {
        free(buf);
        free(mapped);
        *n_out = total;
        return QW_ERR_RANGE;
    }
    if (err == QW_OK && out_ids != NULL)
        memcpy(out_ids, buf, total * sizeof(uint32_t));

    free(buf);
    free(mapped);
    *n_out = total;
    return err;
}

qw_err qw_bpe_decode(const qw_bpe *bpe, const qw_vocab *v, const uint32_t *ids,
                     size_t n, char *out, size_t out_cap, size_t *n_out)
{
    (void)bpe;
    if (v == NULL || ids == NULL || out == NULL || n_out == NULL)
        return QW_ERR_NULL;

    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        const char *p;
        uint32_t l;
        if (qw_vocab_id_to_token(v, ids[i], &p, &l) != QW_OK)
            return QW_ERR_RANGE;
        total += l;
    }
    if (total > out_cap) {
        *n_out = total;
        return QW_ERR_RANGE;
    }

    char *mapped = malloc(total ? total : 1);
    if (mapped == NULL)
        return QW_ERR_ALLOC;
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        const char *p;
        uint32_t l;
        if (qw_vocab_id_to_token(v, ids[i], &p, &l) != QW_OK) {
            free(mapped);
            return QW_ERR_RANGE;
        }
        memcpy(mapped + o, p, l);
        o += l;
    }
    size_t decoded = 0;
    qw_err e = qw_byte_level_decode(mapped, o, out, out_cap, &decoded);
    free(mapped);
    if (e != QW_OK) {
        *n_out = total;
        return e;
    }
    *n_out = decoded;
    return QW_OK;
}
