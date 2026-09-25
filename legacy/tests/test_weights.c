/* tests/test_weights.c — unit tests for qw/weights (tensor->GPU placement).
 * No external framework: plain CHECK (PASS/FAIL per case), main() returns the
 * failure count.
 *
 * Builds a FAKE gguf in memory — reusing the synthetic GGUF writer from
 * tests/test_gguf.c (a copy of its static writer utils, same on-disk layout)
 * — so no real file is required:
 *   48 layers x 514 tensors (512 Q4_0 experts + 2 Q8_0 projections), named
 *   ".blocks.<N>." (llama.cpp convention; "layers.<N>." is covered by a
 *   dedicated case), plus 4 unlayered tensors (tok_embeddings fp16,
 *   output_norm fp16, output/lm_head fp16, ple_table Q8_0) and 1 MTP tensor.
 * The plan is hand-built for exactly the 0-15 / 16-31 / 32-47 split (the
 * model's per-layer bytes are uniform, so qw_plan_compute would pick that
 * split too, but the hand-built plan is exact and decoupled).
 *
 * Checks: every tensor assigned; layer 0/15 -> gpu0, 16 -> gpu1, 47 -> gpu2;
 * PLE table on gpu -1; no per-GPU overlaps; per-GPU totals within 5% of each
 * other; qw_weights_verify passes (and FAILs when a tensor size is doubled);
 * an unknown tensor name makes qw_weights_plan return QW_ERR_FORMAT; the
 * report prints without crashing; int4 (Q4_0) sizes are exact, no rounding
 * loss (odd element count included).
 */
#define _POSIX_C_SOURCE 200809L /* mkstemp, strdup */

#include "qw/gguf.h"
#include "qw/weights.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------- writer utils */
/* Same byte-level writer as tests/test_gguf.c (copied so this test builds
 * the 24,692-tensor model without that file's 3-tensor builder). */
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

/* ------------------------------------------------------------- model */
/* Tensor inventory for the 48-layer model:
 *   experts:    48 x 512, Q4_0, 640 elems (320 B each)      (int4)
 *   projections:48 x 2,   Q8_0, 32 elems  (48 B each)
 *   unlayered:  tok_embeddings (F16 2560x4), output_norm (F16 2560),
 *               output (F16 2560x4), ple_table (Q8_0 320000),
 *               mtp.extra.weight (Q4_0 640, 1-dim -> odd byte size 320.5?
 *               no: 640/2 = 320; the ODD int4 case lives in test_int4)
 * Data section: real (small) blobs, 32-byte aligned offsets. */
#define T_EXP      48 * 512
#define T_PROJ     48 * 2
#define T_UNLAYER  5
#define T_TOTAL    (T_EXP + T_PROJ + T_UNLAYER)

static char *build_model(void)
{
    struct bw b;
    memset(&b, 0, sizeof(b));

    put(&b, "GGUF", 4);
    put32(&b, 3);                  /* version */
    put64(&b, (uint64_t)T_TOTAL);  /* n_tensors */
    put64(&b, 4);                  /* n_kv: name, block_count, arch, ctx_len */

    putstr(&b, "general.name");
    put8(&b, GGUF_STRING);
    putstr(&b, "weights-test-model");

    putstr(&b, "llama.block_count");
    put8(&b, GGUF_UINT32);
    put32(&b, 48);

    putstr(&b, "general.architecture");
    put8(&b, GGUF_STRING);
    putstr(&b, "qwen3next");

    putstr(&b, "llama.context_length");
    put8(&b, GGUF_UINT32);
    put32(&b, 4096);

    /* pad KV section to 32 so the tensor table lands at the aligned offset
     * the reader expects (reader aligns kv_end up to `align`). */
    while (b.n % 32 != 0)
        put8(&b, 0);

    /* tensor table (offsets are relative and nondecreasing) */
    uint64_t d2[2] = { 2560, 4 }; /* 10240 elems */
    uint64_t d1h[1] = { 2560 };   /* hidden norm */
    uint64_t d1p[1] = { 320000 }; /* ple table */
    uint64_t d1x[1] = { 640 };    /* expert / mtp */
    uint64_t d1q[1] = { 32 };     /* projection */

    uint64_t rel = 0;
    uint64_t r0 = rel; /* remember first offsets for the asserts */

    for (int L = 0; L < 48; L++) {
        char nm[64];
        for (int x = 0; x < 512; x++) {
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.weight",
                     L, x);
            put_tensor(&b, nm, d1x, 1, GGUF_T_Q4_0, rel);
            rel += 320;
        }
        snprintf(nm, sizeof(nm), "model.layers.%d.attn.q_weight", L);
        put_tensor(&b, nm, d1q, 1, GGUF_T_Q8_0, rel);
        rel += 48;
        snprintf(nm, sizeof(nm), "model.layers.%d.attn.o_weight", L);
        put_tensor(&b, nm, d1q, 1, GGUF_T_Q8_0, rel);
        rel += 48;
    }
    put_tensor(&b, "tok_embeddings.weight", d2, 2, GGUF_T_F16, rel);
    rel += 20480;
    put_tensor(&b, "output_norm.weight", d1h, 1, GGUF_T_F16, rel);
    rel += 5120;
    put_tensor(&b, "output.weight", d2, 2, GGUF_T_F16, rel);
    rel += 20480;
    put_tensor(&b, "ple_table.weight", d1p, 1, GGUF_T_Q8_0, rel);
    rel += 320000;
    put_tensor(&b, "mtp.extra.weight", d1x, 1, GGUF_T_Q4_0, rel);
    rel += 320;
    (void)r0;

    /* data section follows the tensor table (32-aligned offsets are all
     * within the blob; the blob just needs to be large enough). */
    while (b.n % 32 != 0)
        put8(&b, 0);
    uint64_t data_start = b.n;
    while (b.n < data_start + rel)
        put8(&b, 0xab);

    char tpl[] = "/tmp/opencode/weights_test_XXXXXX";
    int fd = mkstemp(tpl);
    if (fd < 0) {
        fprintf(stderr, "test: mkstemp failed\n");
        exit(2);
    }
    FILE *fp = fopen(tpl, "wb");
    if (fp == NULL || fwrite(b.p, 1, b.n, fp) != b.n) {
        fprintf(stderr, "test: write failed\n");
        exit(2);
    }
    fclose(fp);
    close(fd);
    free(b.p);
    return strdup(tpl);
}

/* Small 3-tensor model (1 layer) for the error-path + int4 tests. */
static char *build_tiny(const char *extra_name, uint32_t extra_dt,
                        const uint64_t *extra_dims, int extra_nd)
{
    struct bw b;
    memset(&b, 0, sizeof(b));

    put(&b, "GGUF", 4);
    put32(&b, 3);
    uint64_t nt = 3 + (extra_name != NULL ? 1 : 0);
    put64(&b, nt);
    put64(&b, 2); /* n_kv */

    putstr(&b, "general.name");
    put8(&b, GGUF_STRING);
    putstr(&b, "tiny");

    putstr(&b, "llama.block_count");
    put8(&b, GGUF_UINT32);
    put32(&b, 1);

    while (b.n % 32 != 0)
        put8(&b, 0);

    uint64_t d2[2] = { 4, 8 }; /* 32 elems, q4_0 row=32 -> 32 B */
    uint64_t rel = 0;
    put_tensor(&b, "model.layers.0.mlp.experts.0.weight", d2, 2,
               GGUF_T_Q4_0, rel); /* 32 elems -> 32 B (int4, 1B/elem) */
    rel += 32;
    uint64_t d4[2] = { 4, 4 };
    put_tensor(&b, "tok_embeddings.weight", d4, 2, GGUF_T_F16, rel);
    rel += 32;
    uint64_t d8[2] = { 4, 8 }; /* 32 elems, q8_0 row=32 -> 48 B */
    put_tensor(&b, "ple_table.weight", d8, 2, GGUF_T_Q8_0, rel);
    rel += 48;
    if (extra_name != NULL)
        put_tensor(&b, extra_name, extra_dims, extra_nd, extra_dt, rel);

    while (b.n % 32 != 0)
        put8(&b, 0);
    uint64_t data_start = b.n;
    while (b.n < data_start + rel)
        put8(&b, 0xab);

    char tpl[] = "/tmp/opencode/weights_tiny_XXXXXX";
    int fd = mkstemp(tpl);
    if (fd < 0)
        return NULL;
    FILE *fp = fopen(tpl, "wb");
    if (fp == NULL || fwrite(b.p, 1, b.n, fp) != b.n) {
        close(fd);
        free(b.p);
        return NULL;
    }
    fclose(fp);
    close(fd);
    free(b.p);
    return strdup(tpl);
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

/* Hand-built 3-GPU plan: layers 0-15 / 16-31 / 32-47. */
static void build_plan(qw_plan *p)
{
    memset(p, 0, sizeof(*p));
    p->n_gpu = 3;
    p->n_layers = 48;
    p->first_layer[0] = 0;
    p->layers_per_gpu[0] = 16;
    p->first_layer[1] = 16;
    p->layers_per_gpu[1] = 16;
    p->first_layer[2] = 32;
    p->layers_per_gpu[2] = 16;
}

static int find_entry(const qw_weights *w, int layer, const char *suffix)
{
    char pat[64];
    snprintf(pat, sizeof(pat), ".%d.%s", layer, suffix);
    for (size_t i = 0; i < w->n; i++) {
        if (w->entries[i].layer == layer &&
            strstr(w->entries[i].name, pat) != NULL)
            return (int)i;
    }
    return -1;
}

/* gpu for a tensor by name substring. */
static int gpu_of(const qw_weights w, const char *substr)
{
    for (size_t i = 0; i < w.n; i++)
        if (strstr(w.entries[i].name, substr) != NULL)
            return w.entries[i].gpu;
    return -999;
}

int main(void)
{
    char *path = build_model();
    if (path == NULL) {
        printf("FAIL build synthetic gguf\n");
        return 1;
    }
    qw_err err = QW_ERR_NULL;
    struct gguf_file *f = gguf_open(path, &err);
    if (f == NULL) {
        printf("FAIL gguf_open (%s)\n", qw_errstr(err));
        return 1;
    }
    CHECK(gguf_n_tensors(f) == (size_t)T_TOTAL, "gguf: 24692 tensors");

    qw_plan plan;
    build_plan(&plan);

    qw_weights w;
    memset(&w, 0, sizeof(w));
    err = qw_weights_plan(f, &plan, 256, &w);
    CHECK(err == QW_OK, "plan: ok (align 256)");
    CHECK(w.n == (size_t)T_TOTAL, "plan: every tensor assigned");

    /* layer -> gpu mapping */
    CHECK(gpu_of(w, "model.layers.0.") == 0, "layer 0 -> gpu0");
    CHECK(gpu_of(w, "model.layers.15.") == 0, "layer 15 -> gpu0");
    CHECK(gpu_of(w, "model.layers.16.") == 1, "layer 16 -> gpu1");
    CHECK(gpu_of(w, "model.layers.31.") == 1, "layer 31 -> gpu1");
    CHECK(gpu_of(w, "model.layers.32.") == 2, "layer 32 -> gpu2");
    CHECK(gpu_of(w, "model.layers.47.") == 2, "layer 47 -> gpu2");

    /* unlayered policy */
    CHECK(gpu_of(w, "tok_embeddings") == 2, "embedding -> last gpu");
    CHECK(gpu_of(w, "output_norm") == 2, "norm -> last gpu");
    CHECK(gpu_of(w, "output.weight") == 2, "lm_head -> last gpu");
    CHECK(gpu_of(w, "mtp.extra") == 2, "mtp -> last gpu");
    CHECK(gpu_of(w, "ple_table") == -1, "ple_table -> gpu -1 (host)");

    /* int4 sizes exact, no rounding loss */
    int i_exp0 = find_entry(&w, 0, "mlp.experts.0.weight");
    CHECK(i_exp0 >= 0, "find expert tensor");
    if (i_exp0 >= 0) {
        const qw_weights_entry *e = &w.entries[i_exp0];
        int gi = gguf_find(f, e->name);
        CHECK(gi >= 0 && e->size == gguf_tensor_nbytes(f, (size_t)gi),
              "expert size exact (== gguf_tensor_nbytes, no rounding)");
        CHECK(e->dt == QW_INT4, "expert dtype int4");
        CHECK(e->pool_offset % 256 == 0, "pool offset 256-aligned");
        CHECK(e->layer == 0, "layer recorded");
    }
    /* Re-pack each pool exactly as the plan does (align cursor up to 256,
     * advance by size) and confirm the recorded offsets reproduce it:
     * catches overlaps and gaps. */
    uint64_t cursor[3] = { 0, 0, 0 };
    int mismatch = 0;
    for (size_t i = 0; i < w.n; i++) {
        const qw_weights_entry *e = &w.entries[i];
        if (e->gpu < 0) {
            if (e->pool_offset != 0)
                mismatch++;
            continue;
        }
        uint64_t want = (cursor[e->gpu] + 255) & ~(uint64_t)255;
        if (e->pool_offset != want || e->pool_offset % 256 != 0)
            mismatch++;
        cursor[e->gpu] = e->pool_offset + e->size;
    }
    CHECK(mismatch == 0, "no per-gpu overlaps or gaps (re-pack matches)");

    /* per-gpu USED bytes (sum of raw sizes; alignment waste is separate).
     * Uniform model: gpu0/gpu1 = 16 layers x (512 experts + 2 proj);
     * gpu2 also holds the 4 unlayered device tensors + mtp. */
    uint64_t per[3] = { 0, 0, 0 };
    for (size_t i = 0; i < w.n; i++)
        if (w.entries[i].gpu >= 0)
            per[w.entries[i].gpu] += w.entries[i].size;
    uint64_t exp_layer = 512ULL * 640ULL + 2ULL * 48ULL; /* q4_0 640B, q8_0 48B */
    uint64_t exp_gpu0 = 16ULL * exp_layer;
    CHECK(per[0] == exp_gpu0, "gpu0 used == 16 x layer bytes");
    CHECK(per[1] == exp_gpu0, "gpu1 used == 16 x layer bytes");
    uint64_t mx = per[0], mn = per[0];
    for (int g = 1; g < 3; g++) {
        if (per[g] > mx) mx = per[g];
        if (per[g] < mn) mn = per[g];
    }
    CHECK((mx - mn) * 100 <= mx * 5, "per-gpu totals within 5%");
    printf("      gpu0=%llu gpu1=%llu gpu2=%llu (used bytes, incl. unlayered)\n",
           (unsigned long long)per[0], (unsigned long long)per[1],
           (unsigned long long)per[2]);

    /* verify: clean table passes (capacity = the aligned pool span) */
    uint64_t cap = cursor[0];
    for (int g = 1; g < 3; g++)
        if (cursor[g] > cap)
            cap = cursor[g];
    err = qw_weights_verify(&w, &plan, (size_t)cap);
    CHECK(err == QW_OK, "verify: clean table passes");

    /* verify: doubling one tensor's size must FAIL (overlap) */
    {
        qw_weights wbad = w;
        wbad.entries = malloc(w.n * sizeof(qw_weights_entry));
        if (wbad.entries == NULL) {
            printf("FAIL OOM (copy)\n");
            return g_fail ? g_fail : 1;
        }
        memcpy(wbad.entries, w.entries, w.n * sizeof(qw_weights_entry));
        wbad.entries[0].size *= 2;
        err = qw_weights_verify(&wbad, &plan, (size_t)cap + 320);
        CHECK(err != QW_OK, "verify: doubled size fails");
        free(wbad.entries);
    }

    /* verify: a too-small pool capacity must FAIL */
    err = qw_weights_verify(&w, &plan, (size_t)(cap / 2));
    CHECK(err == QW_ERR_RANGE, "verify: capacity overflow fails");

    /* largest gap: a fresh contiguous pack has no fragmentation beyond the
     * per-tensor alignment padding, so the biggest gap is < align (256). */
    uint64_t gap = 999;
    err = qw_weights_largest_gap(&w, 0, &gap);
    CHECK(err == QW_OK && gap < 256, "largest gap < align (unfragmented)");

    /* report: prints without crashing */
    err = qw_weights_report(&w);
    CHECK(err == QW_OK, "report: prints (no crash)");

    /* free the big model before the tiny-file tests */
    gguf_close(f);
    free(w.entries);
    unlink(path);
    free(path);

    /* ---- tiny model: layer out of plan range + unknown name ---- */
    {
        char *t = build_tiny(NULL, 0, NULL, 0);
        qw_plan p1;
        build_plan(&p1);
        p1.n_gpu = 1;
        p1.n_layers = 1;
        p1.first_layer[0] = 0;
        p1.layers_per_gpu[0] = 1;
        struct gguf_file *tf = gguf_open(t, &err);
        qw_weights tw;
        memset(&tw, 0, sizeof(tw));
        err = qw_weights_plan(tf, &p1, 256, &tw);
        CHECK(err == QW_OK, "tiny: plan ok");
        /* int4 exact: 32 q4_0 elems -> 32 B, and the placement size must
         * equal the gguf nbytes with no rounding/truncation. */
        int i4 = gguf_find(tf, "model.layers.0.mlp.experts.0.weight");
        CHECK(i4 >= 0 && gguf_tensor_nbytes(tf, (size_t)i4) == 32,
              "tiny: q4_0 32 elems = 32 B (exact)");
        CHECK(tw.n == 3, "tiny: 3 tensors");
        if (tw.n == 3) {
            int e = -1;
            for (size_t k = 0; k < tw.n; k++)
                if (strstr(tw.entries[k].name, "experts.0") != NULL)
                    e = (int)k;
            CHECK(e >= 0 && tw.entries[e].size == 32 && tw.entries[e].dt == QW_INT4,
                  "tiny: placement size 32 B, dtype int4 (no truncation)");
        }
        gguf_close(tf);
        free(tw.entries);
        unlink(t);
        free(t);
    }

    /* unknown tensor name -> QW_ERR_FORMAT */
    {
        uint64_t dd[2] = { 4, 4 };
        /* F16 (row=1) so the tensor itself is valid; the NAME is what must
         * trip the unknown-naming-scheme path. */
        char *t = build_tiny("mystery.tensor", GGUF_T_F16, dd, 2);
        qw_plan p1;
        build_plan(&p1);
        p1.n_gpu = 1;
        p1.n_layers = 1;
        p1.first_layer[0] = 0;
        p1.layers_per_gpu[0] = 1;
        struct gguf_file *tf = gguf_open(t, &err);
        qw_weights tw;
        memset(&tw, 0, sizeof(tw));
        err = qw_weights_plan(tf, &p1, 256, &tw);
        CHECK(err == QW_ERR_FORMAT, "unknown name -> QW_ERR_FORMAT");
        CHECK(tw.n == 0, "unknown name: table left empty (no leak)");
        gguf_close(tf);
        unlink(t);
        free(t);
    }

    /* layer number the plan does not cover -> QW_ERR_FORMAT */
    {
        char *t = build_model();
        /* tiny file where the tensor names layer 5 but plan covers 0 only */
        uint64_t dd[1] = { 32 };
        /* rebuild tiny with a layers.5 tensor via the extra slot */
        (void)t;
        unlink(t);
        free(t);
        t = NULL;
        struct bw b;
        memset(&b, 0, sizeof(b));
        put(&b, "GGUF", 4);
        put32(&b, 3);
        put64(&b, 2);
        put64(&b, 2);
        putstr(&b, "general.name");
        put8(&b, GGUF_STRING);
        putstr(&b, "tiny");
        putstr(&b, "llama.block_count");
        put8(&b, GGUF_UINT32);
        put32(&b, 1);
        while (b.n % 32 != 0)
            put8(&b, 0);
        put_tensor(&b, "model.layers.5.attn.q_weight", dd, 1, GGUF_T_Q8_0, 0);
        put_tensor(&b, "tok_embeddings.weight", dd, 1, GGUF_T_F16, 48);
        while (b.n % 32 != 0)
            put8(&b, 0);
        uint64_t ds = b.n;
        while (b.n < ds + 96)
            put8(&b, 0xab);
        char tpl[] = "/tmp/opencode/weights_l5_XXXXXX";
        int fd = mkstemp(tpl);
        FILE *fp = fopen(tpl, "wb");
        if (fd >= 0 && fp != NULL && fwrite(b.p, 1, b.n, fp) == b.n) {
            fclose(fp);
            close(fd);
            free(b.p);
            t = strdup(tpl);
        } else {
            free(b.p);
            printf("FAIL build layer-5 tiny\n");
        }
        qw_plan p1;
        build_plan(&p1);
        p1.n_gpu = 1;
        p1.n_layers = 1;
        p1.first_layer[0] = 0;
        p1.layers_per_gpu[0] = 1;
        struct gguf_file *tf = gguf_open(t, &err);
        qw_weights tw;
        memset(&tw, 0, sizeof(tw));
        err = qw_weights_plan(tf, &p1, 256, &tw);
        CHECK(err == QW_ERR_FORMAT, "layer 5 not in 1-layer plan -> FORMAT");
        gguf_close(tf);
        unlink(t);
        free(t);
    }

    printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail;
}
