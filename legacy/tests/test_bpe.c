/* tests/test_bpe.c — unit tests for the byte-level-BPE encoder (qw/bpe.h).
 * No external test framework: plain CHECK, PASS/FAIL per case, main() returns
 * the failure count (0 = all pass).
 *
 * The vocabulary is built IN MEMORY by hand (a helper below) — it does NOT go
 * through the GGUF loader, whose string path is being fixed concurrently. The
 * helper fills qw_vocab the exact way src/loader/vocab.c does (pool + idx +
 * fnv1a open-addressing hash, cap a power of two >= 2*n, HASH_EMPTY slots) so
 * qw_vocab_token_to_id / qw_vocab_id_to_token work on it.
 *
 * The merge table is supplied as C strings in rank order, per the module's
 * decoupled contract (the gguf-merges-to-lines conversion is the caller's job).
 *
 * The special-token invariant is verified two ways: (1) structurally, by
 * feeding qw_bpe_init a merge line whose result is a special id and asserting
 * it is REJECTED at load time (so such a pair can never be indexed); and (2)
 * behaviourally, by encoding input text whose literal characters form a
 * special token and asserting the special id never appears in the output.
 */
#include "qw/alloc.h"
#include "qw/bpe.h"
#include "qw/types.h"
#include "qw/vocab.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same FNV-1a as src/loader/vocab.c (64-bit). */
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

/* A token is raw bytes (may contain NUL / >= 128) plus a type. */
typedef struct tokdef {
    const unsigned char *bytes;
    uint32_t len;
    uint8_t type;
} tokdef_t;

/* In-memory vocab builder. Each test builds a fresh, zero-initialized vocab
 * (mirrors the loader's qw_vocab_init); `v` is not assumed to hold prior
 * allocations. toks[i] are raw byte ranges (byte tokens may be any of the 256
 * values, including 0x00 and >= 0x80). */
static void build_vocab(qw_vocab *v, const tokdef_t *toks, int n,
                        const qw_special_ids *sp_in)
{
    memset(v, 0, sizeof(*v));
    v->sp.bos = v->sp.eos = v->sp.pad = v->sp.unk = -1;
    if (sp_in != NULL)
        v->sp = *sp_in;

    v->n_vocab = (uint32_t)n;
    size_t pool = 0;
    for (int i = 0; i < n; i++)
        pool += toks[i].len;

    v->toks = (uint8_t *)malloc(pool ? pool : 1);
    v->idx = (qw_vocab_tok *)malloc((size_t)n * sizeof(qw_vocab_tok));
    v->scores = (float *)malloc((size_t)n * sizeof(float));
    assert(v->toks && v->idx && v->scores);

    uint32_t off = 0;
    for (int i = 0; i < n; i++) {
        memcpy(v->toks + off, toks[i].bytes, toks[i].len);
        v->idx[i].off = off;
        v->idx[i].len = toks[i].len;
        v->idx[i].score = -10.0f;
        v->idx[i].type = toks[i].type;
        v->scores[i] = -10.0f;
        off += toks[i].len;
    }

    /* hash: cap = power of two >= 2*n and >= 16; fnv1a, linear probe. */
    uint32_t cap = 16;
    while (cap < (uint32_t)n * 2)
        cap <<= 1;
    v->tok_hash_cap = cap;
    v->tok_hash = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    assert(v->tok_hash);
    for (uint32_t i = 0; i < cap; i++)
        v->tok_hash[i] = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++) {
        const uint8_t *s = v->toks + v->idx[i].off;
        uint32_t h = (uint32_t)(fnv1a_len(s, v->idx[i].len) & (cap - 1));
        for (;;) {
            if (v->tok_hash[h] == 0xFFFFFFFFu) {
                v->tok_hash[h] = (uint32_t)i;
                break;
            }
            h = (h + 1) & (cap - 1);
        }
    }
}

static void free_vocab(qw_vocab *v)
{
    free(v->toks);
    free(v->idx);
    free(v->scores);
    free(v->pre);
    free(v->merges);
    free(v->tok_hash);
    memset(v, 0, sizeof(*v));
}

/* ------------------------------------------------------------- harness */
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

static const char *const SPECIAL = "<|eoi|>";

/* The main test vocab: the special token, then ALL 256 single-byte tokens
 * (so any input byte is encodable — a real byte-level BPE vocab), then the
 * multi-byte tokens the merge table / tests reference.
 *
 * id layout: 0 = special; 1..256 = byte tokens (byte b has id 1 + b);
 * then "a5"=257, "a50"=258, "space"=259, "a4"=260, "a42"=261.
 */
#define M_SPECIAL 0u
#define M_BYTE(b) (1u + (b)) /* byte token id for raw byte b */
#define M_A5 257u
#define M_A50 258u
#define M_SPCE 259u
#define M_A4 260u
#define M_A42 261u
#define M_N 262u

static void build_main_vocab(qw_vocab *v)
{
    static const unsigned char SP[] = "<|eoi|>";
    static const unsigned char A5[] = "a5";
    static const unsigned char A50[] = "a50";
    static const unsigned char SPCE[] = "space";
    static const unsigned char A4[] = "a4";
    static const unsigned char A42[] = "a42";
    static const qw_special_ids sp = { .bos = -1, .eos = 0, .pad = -1,
                                       .unk = -1 };

    tokdef_t toks[M_N];
    uint8_t rawbytes[256];
    for (unsigned b = 0; b < 256; b++)
        rawbytes[b] = (uint8_t)b;
    int n = 0;
    toks[n].bytes = SP; toks[n].len = (uint32_t)(sizeof(SP) - 1);
    toks[n].type = QW_TOK_USER_DEFINED; n++;
    for (unsigned b = 0; b < 256; b++) {
        toks[n].bytes = rawbytes + b;
        toks[n].len = 1;
        toks[n].type = QW_TOK_BYTE;
        n++;
    }
    toks[n].bytes = A5; toks[n].len = 2; toks[n].type = QW_TOK_NORMAL; n++;
    toks[n].bytes = A50; toks[n].len = 3; toks[n].type = QW_TOK_NORMAL; n++;
    toks[n].bytes = SPCE; toks[n].len = 5; toks[n].type = QW_TOK_NORMAL; n++;
    toks[n].bytes = A4; toks[n].len = 2; toks[n].type = QW_TOK_NORMAL; n++;
    toks[n].bytes = A42; toks[n].len = 3; toks[n].type = QW_TOK_NORMAL; n++;
    assert(n == M_N);
    build_vocab(v, toks, n, &sp);
}

/* ids for the rank/tie/random tests. "abc" is deliberately ABSENT so the
 * input "abc" cannot take the whole-chunk fast path and must be built up by
 * merges. A space byte token is included so the random round-trip test (which
 * sometimes inserts a space) is fully encodable. */
enum { G_A = 0, G_B, G_C, G_SP, G_AB, G_BC, G_N = 6 };

static void build_abc_vocab(qw_vocab *v)
{
    static const unsigned char A[] = "a", B[] = "b", C[] = "c", SPC[] = " ";
    static const unsigned char AB[] = "ab", BC[] = "bc";
    tokdef_t toks[G_N];
    toks[0] = (tokdef_t){ A, 1, QW_TOK_BYTE };
    toks[1] = (tokdef_t){ B, 1, QW_TOK_BYTE };
    toks[2] = (tokdef_t){ C, 1, QW_TOK_BYTE };
    toks[3] = (tokdef_t){ SPC, 1, QW_TOK_BYTE };
    toks[4] = (tokdef_t){ AB, 2, QW_TOK_NORMAL };
    toks[5] = (tokdef_t){ BC, 2, QW_TOK_NORMAL };
    build_vocab(v, toks, G_N, NULL);
}

/* Vocab for the structural-rejection test: the special token is literally
 * "ab" (id 0), and "a"+"b" concatenates to "ab" — so the merge "a b" would
 * produce the special id. qw_bpe_init must REJECT it. */
static void build_special_vocab(qw_vocab *v)
{
    static const unsigned char SPAB[] = "ab";
    static const unsigned char A[] = "a", B[] = "b";
    static const qw_special_ids sp = { .bos = -1, .eos = 0, .pad = -1,
                                       .unk = -1 };
    tokdef_t toks[3];
    toks[0] = (tokdef_t){ SPAB, 2, QW_TOK_USER_DEFINED };
    toks[1] = (tokdef_t){ A, 1, QW_TOK_BYTE };
    toks[2] = (tokdef_t){ B, 1, QW_TOK_BYTE };
    build_vocab(v, toks, 3, &sp);
}

/* ---------------------------------------------------------------- main */
int main(void)
{
    qw_arena *ar = qw_arena_create(1 << 20);
    assert(ar != NULL);

    /* ================= 1. ASCII round trip ================= */
    {
        qw_vocab v;
        build_main_vocab(&v);
        const char *mlines[] = {"a 5", "a5 0", "a 4", "a4 2"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 4, ar) == QW_OK,
              "ascii: bpe_init ok");

        const char *text = "hello world";
        uint32_t ids[64];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, strlen(text), ids, 64, &n);
        CHECK(e == QW_OK, "ascii: encode ok");

        char buf[64];
        size_t dn = 0;
        qw_err d = qw_bpe_decode(&bpe, &v, ids, n, buf, sizeof(buf), &dn);
        CHECK(d == QW_OK, "ascii: decode ok");
        CHECK(dn == strlen(text) && memcmp(buf, text, dn) == 0,
              "ascii: decode == original");

        /* pre-tokenizer: "hello"(5 byte toks) + " "(1) + "world"(5) = 11 ids,
         * the single whitespace token sitting at index 5. */
        CHECK(n == 11, "ascii: 5 + 1 + 5 = 11 ids");
        CHECK(ids[5] == M_BYTE(' '),
              "ascii: index 5 is the whitespace byte token");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 2. bytes >= 128 (UTF-8) round trip ================= */
    {
        qw_vocab v;
        build_main_vocab(&v);
        const char *mlines[] = {"a 5", "a5 0", "a 4", "a4 2"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 4, ar) == QW_OK,
              "utf8: bpe_init ok");

        const char text[] = { 'a', (char)0xC3, (char)0xA9, (char)0xFF,
                              'b' };
        size_t len = sizeof(text);
        uint32_t ids[64];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, len, ids, 64, &n);
        CHECK(e == QW_OK, "utf8: encode ok");

        char buf[64];
        size_t dn = 0;
        qw_err d = qw_bpe_decode(&bpe, &v, ids, n, buf, sizeof(buf), &dn);
        CHECK(d == QW_OK, "utf8: decode ok");
        CHECK(dn == len && memcmp(buf, text, dn) == 0,
              "utf8: decode == original (bytes >= 128)");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 3. embedded NUL, len authoritative ================= */
    {
        qw_vocab v;
        build_main_vocab(&v);
        const char *mlines[] = {"a 5", "a5 0", "a 4", "a4 2"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 4, ar) == QW_OK,
              "nul: bpe_init ok");

        const char text[] = { 'a', '\0', 'b' };
        size_t len = 3;
        uint32_t ids[64];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, len, ids, 64, &n);
        CHECK(e == QW_OK, "nul: encode ok (len authoritative)");
        CHECK(n == 3, "nul: 3 ids (a, 0, b)");

        char buf[64];
        size_t dn = 0;
        qw_err d = qw_bpe_decode(&bpe, &v, ids, n, buf, sizeof(buf), &dn);
        CHECK(d == QW_OK, "nul: decode ok");
        CHECK(dn == 3 && buf[0] == 'a' && buf[1] == '\0' && buf[2] == 'b',
              "nul: decode == original (embedded NUL preserved)");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ============ 4. merges apply in rank order (not greedy-shortest) ====== */
    {
        qw_vocab v;
        build_abc_vocab(&v);
        /* rank 0: "b c" -> bc (the RIGHTMOST adjacent pair);
         * rank 1: "a b" -> ab (the leftmost adjacent pair).
         * Seed [a][b][c]. Rank order merges the lower rank first => (b,c)
         * wins => [a][bc]. A "leftmost-first" (or shortest-first) bug would
         * merge (a,b) first => [ab][c]. */
        const char *mlines[] = {"b c", "a b"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 2, ar) == QW_OK,
              "rank: bpe_init ok");

        const char *text = "abc";
        uint32_t ids[8];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, strlen(text), ids, 8, &n);
        CHECK(e == QW_OK, "rank: encode ok");
        /* rank order => [a][bc] (2 ids). Leftmost/shortest-first bug => [ab][c]. */
        CHECK(n == 2 && ids[0] == G_A && ids[1] == G_BC,
              "rank: rank order beats leftmost-first (got [a][bc])");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 5. tie broken leftmost ================= */
    {
        qw_vocab v;
        build_abc_vocab(&v);
        /* Both pairs at the SAME rank (rank 0). Seed [a][b][c]:
         *   (a,b) and (b,c) are both rank 0. The tie must break LEFTMOST,
         * so (a,b) is merged first => [ab][c] (a "rightmost" bug would give
         * [a][bc]). */
        const char *mlines[] = {"a b", "b c"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 2, ar) == QW_OK,
              "tie: bpe_init ok");

        const char *text = "abc";
        uint32_t ids[8];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, strlen(text), ids, 8, &n);
        CHECK(e == QW_OK, "tie: encode ok");
        /* leftmost (a,b) wins the tie => [ab][c]. Rightmost would give [a][bc]. */
        CHECK(n == 2 && ids[0] == G_AB && ids[1] == G_C,
              "tie: leftmost pair wins (got [ab][c])");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 6. vocab word -> single id ================= */
    {
        qw_vocab v;
        build_main_vocab(&v);
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, NULL, 0, ar) == QW_OK,
              "word: bpe_init (no merges) ok");

        const char *text = "space";
        uint32_t ids[8];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, strlen(text), ids, 8, &n);
        CHECK(e == QW_OK, "word: encode ok");
        CHECK(n == 1 && ids[0] == M_SPCE,
              "word: 'space' -> its single token id");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 7. unknown word -> byte tokens ================= */
    {
        qw_vocab v;
        build_main_vocab(&v);
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, NULL, 0, ar) == QW_OK,
              "unk: bpe_init ok");

        const char *text = "zzz"; /* no "z"/"zz"/"zzz" token in vocab */
        uint32_t ids[8];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, strlen(text), ids, 8, &n);
        CHECK(e == QW_OK, "unk: encode ok");
        CHECK(n == 3 && ids[0] == M_BYTE('z') && ids[1] == M_BYTE('z') &&
                  ids[2] == M_BYTE('z'),
              "unk: 'zzz' -> 3 byte tokens [z][z][z]");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 8. special token never produced ================= */
    {
        qw_vocab v;
        build_main_vocab(&v);
        const char *mlines[] = {"a 5", "a5 0", "a 4", "a4 2"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 4, ar) == QW_OK,
              "special: bpe_init ok");

        /* The literal characters of the special token, as plain input text.
         * They are individual byte tokens in the vocab (none is special), so
         * they are encodable — but the merge table has no pair that yields
         * the special id, so the id can never appear. */
        const char *text = SPECIAL;
        uint32_t ids[64];
        size_t n = 0;
        qw_err e = qw_bpe_encode(&bpe, &v, text, strlen(text), ids, 64, &n);
        CHECK(e == QW_OK, "special: encode ok");

        int found_special = 0;
        for (size_t i = 0; i < n; i++)
            if (ids[i] == (uint32_t)v.sp.eos)
                found_special = 1;
        CHECK(!found_special,
              "special: literal '<|eoi|>' text never yields the special id");

        /* And it round-trips byte-for-byte. */
        char buf[64];
        size_t dn = 0;
        qw_err d = qw_bpe_decode(&bpe, &v, ids, n, buf, sizeof(buf), &dn);
        CHECK(d == QW_OK && dn == strlen(text) &&
                  memcmp(buf, text, dn) == 0,
              "special: special-text round-trips");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ============ 8b. structural: init rejects a merge -> special id ====== */
    {
        qw_vocab v;
        build_special_vocab(&v);
        const char *mlines[] = {"a b"}; /* "a"+"b" = "ab" = special id 0 */
        qw_bpe bpe;
        qw_err e = qw_bpe_init(&bpe, &v, mlines, 1, ar);
        CHECK(e == QW_ERR_FORMAT,
              "special(structural): merge producing a special id rejected");
        (void)bpe;
        free_vocab(&v);
    }

    /* ================= 9. insufficient out_cap -> RANGE + needed ========= */
    {
        qw_vocab v;
        build_main_vocab(&v);
        const char *mlines[] = {"a 5", "a5 0", "a 4", "a4 2"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 4, ar) == QW_OK,
              "cap: bpe_init ok");

        const char *text = "a5";
        uint32_t ids[8];
        size_t need = 0;
        /* encode with no room: must report the exact needed count. */
        qw_err e = qw_bpe_encode(&bpe, &v, text, strlen(text), NULL, 0, &need);
        CHECK(e == QW_ERR_RANGE && need == 1,
              "cap: out_cap=0 -> QW_ERR_RANGE, needed == 1 (merged a50)");

        /* decode with a too-small buffer: reports raw bytes needed. */
        uint32_t one[1] = { M_BYTE('a') };
        size_t dn = 0;
        qw_err d = qw_bpe_decode(&bpe, &v, one, 1, (char *)ids, 0, &dn);
        CHECK(d == QW_ERR_RANGE && dn == 1,
              "cap: decode out_cap=0 -> QW_ERR_RANGE, needed == 1");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 10. 1000 seeded random ASCII round trips ========== */
    {
        qw_vocab v;
        build_abc_vocab(&v);
        /* merges that produce in-vocab tokens: "a b"->ab, "b c"->bc. */
        const char *mlines[] = {"a b", "b c"};
        qw_bpe bpe;
        CHECK(qw_bpe_init(&bpe, &v, mlines, 2, ar) == QW_OK,
              "rand: bpe_init ok");

        int ok = 0;
        srand(12345);
        for (int t = 0; t < 1000; t++) {
            size_t len = (size_t)(rand() % 32);
            char *text = malloc(len ? len : 1);
            assert(text != NULL);
            for (size_t i = 0; i < len; i++)
                text[i] = (char)('a' + (rand() % 3)); /* 'a','b','c' only */
            if (t % 7 == 0 && len > 0)
                text[len / 2] = ' '; /* sometimes a space */

            uint32_t *ids = malloc(256 * sizeof(uint32_t));
            assert(ids != NULL);
            size_t n = 0;
            if (qw_bpe_encode(&bpe, &v, text, len, ids, 256, &n) != QW_OK) {
                free(text);
                free(ids);
                break;
            }
            char *back = malloc(len + 1);
            assert(back != NULL);
            size_t dn = 0;
            if (qw_bpe_decode(&bpe, &v, ids, n, back, len + 1, &dn) !=
                QW_OK ||
                dn != len || memcmp(back, text, len) != 0) {
                free(text);
                free(ids);
                free(back);
                break;
            }
            ok++;
            free(text);
            free(ids);
            free(back);
        }
        CHECK(ok == 1000, "rand: 1000 seeded ASCII round trips");
        qw_bpe_free(&bpe);
        free_vocab(&v);
    }

    /* ================= 11. pre-tokenizer span boundaries ================= */
    {
        /* Documented example. Input (byte-level-mapped, all ASCII):
         *   "ab.c d"
         *   a b . c     d
         * Spans (offsets into the string):
         *   [0,3) "ab."  word "ab" + attached punctuation "."
         *   [3,4) "c"    lone word char (preceded by non-word ".")
         *   [4,5) " "    whitespace run
         *   [5,6) "d"    lone word char
         * => 4 spans. */
        const char *text = "ab.c d";
        size_t len = strlen(text);
        qw_span spans[8];
        int n = 0;
        qw_err e = qw_bpe_pre_tokenizer(text, len, spans, 8, &n);
        CHECK(e == QW_OK && n == 4, "pretok: 'ab.c d' -> 4 spans");
        if (n == 4) {
            CHECK(spans[0].start == 0 && spans[0].end == 3,
                  "pretok: span0 = [0,3) 'ab.'");
            CHECK(spans[1].start == 3 && spans[1].end == 4,
                  "pretok: span1 = [3,4) 'c'");
            CHECK(spans[2].start == 4 && spans[2].end == 5,
                  "pretok: span2 = [4,5) ' '");
            CHECK(spans[3].start == 5 && spans[3].end == 6,
                  "pretok: span3 = [5,6) 'd'");
        }
        /* max_spans too small -> RANGE + total needed. */
        int n2 = 0;
        qw_err e2 = qw_bpe_pre_tokenizer(text, len, spans, 2, &n2);
        CHECK(e2 == QW_ERR_RANGE && n2 == 4,
              "pretok: max_spans=2 -> RANGE, needed == 4");
    }

    /* ================= 12. byte-level map reversibility ================= */
    {
        const char in[] = { 'a', (char)0xFF, (char)0x00, (char)0x80, 'z' };
        size_t n = sizeof(in);
        char mapped[16];
        size_t mlen = 0;
        qw_err e = qw_byte_level_encode(in, n, mapped, sizeof(mapped), &mlen);
        CHECK(e == QW_OK, "blm: encode ok");
        char back[16];
        size_t bn = 0;
        qw_err d = qw_byte_level_decode(mapped, mlen, back, sizeof(back), &bn);
        CHECK(d == QW_OK && bn == n && memcmp(back, in, n) == 0,
              "blm: encode->decode is the identity");
    }

    printf("%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    qw_arena_destroy(ar);
    return g_fail;
}
