/* src/loader/weights.c — tensor-to-GPU placement table (see qw/weights.h).
 *
 * Pure C17, libc only, no HIP: this module computes where each tensor goes,
 * it never touches a device. All byte arithmetic is uint64_t; sizes come
 * straight from the GGUF reader, which packs int4 2-per-byte, so sub-byte
 * dtypes are accounted exactly.
 *
 * Allocation model: one realloc-grown entry array (no per-entry mallocs).
 * Every public entry point zero-initializes its output, so a partial build
 * never leaks and a caller sees a well-formed empty table on error.
 */
#include "qw/weights.h"
#include "qw/gguf.h"
#include "qw/macros.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------- table */
static void qw_weights_init(qw_weights *w)
{
    memset(w, 0, sizeof(*w));
}

/* Free the entry array (NULL-safe). Leaves the table in its empty state. */
static void qw_weights_reset(qw_weights *w)
{
    if (w == NULL)
        return;
    free(w->entries);
    qw_weights_init(w);
}

/* Grow the entry array; QW_ERR_ALLOC on OOM (table left as-is). */
static qw_err qw_weights_reserve(qw_weights *w, size_t need)
{
    if (need <= w->cap)
        return QW_OK;
    size_t ncap = w->cap ? w->cap : 64;
    while (ncap < need)
        ncap *= 2;
    qw_weights_entry *ne =
        realloc(w->entries, ncap * sizeof(qw_weights_entry));
    if (ne == NULL) {
        QW_LOGE("qw/weights: out of memory (growing to %zu entries)", ncap);
        return QW_ERR_ALLOC;
    }
    w->entries = ne;
    w->cap = ncap;
    return QW_OK;
}

/* Append one entry. */
static qw_err qw_weights_push(qw_weights *w, const qw_weights_entry *e)
{
    qw_err err = qw_weights_reserve(w, w->n + 1);
    if (err != QW_OK)
        return err;
    w->entries[w->n++] = *e;
    return QW_OK;
}

/* --------------------------------------------------------- gguf -> dtype */
/* Map a GGUF tensor type onto the engine's storage dtype. int4 is the
 * packed-2-per-byte family (Q4_0 and Q4_1; K-quants and IQ quants also
 * pack 2/4 nibbles per byte but report as int8-class here); everything
 * else maps to its natural width. Unknown types return false. */
static bool qw_gguf_dt(gguf_tensor_type t, qw_dtype *out)
{
    switch (t) {
    case GGUF_T_F32:
        *out = QW_F32;
        return true;
    case GGUF_T_F16:
    case GGUF_T_BF16:
        *out = QW_F16;
        return true;
    case GGUF_T_Q4_0:
    case GGUF_T_Q4_1:
        *out = QW_INT4;
        return true;
    case GGUF_T_Q8_0:
    case GGUF_T_Q8_1:
        *out = QW_INT8;
        return true;
    default:
        /* q5/q2-k/q3-k/q4-k/q5-k/q6-k, iq quants, f8/f4 packed types:
         * all sub-byte-or-1B storage classes the planner does not need to
         * distinguish (placement is size-driven, not dtype-driven). */
        *out = QW_INT8;
        return true;
    }
}

/* ------------------------------------------------------------- layer idx */
/* Scan the name for ".blocks.<N>." or "layers.<N>." (N all digits). On a
 * match, *layer_out gets N and true is returned; false if no layer number
 * is present. */
static bool qw_extract_layer(const char *name, int *layer_out)
{
    for (const char *p = name; (p = strstr(p, "layers.")) != NULL; p++) {
        const char *d = p + 7;
        int n = 0;
        bool any = false;
        while (*d >= '0' && *d <= '9') {
            if (n > (INT_MAX - 9) / 10)
                return false; /* absurd layer number: treat as unnumbered
                                  and let the unlayered policy fail loud */
            n = n * 10 + (*d - '0');
            d++;
            any = true;
        }
        if (any) {
            *layer_out = n;
            return true;
        }
    }
    for (const char *p = name; (p = strstr(p, ".blocks.")) != NULL; p++) {
        const char *d = p + 8;
        int n = 0;
        bool any = false;
        while (*d >= '0' && *d <= '9') {
            if (n > (INT_MAX - 9) / 10)
                return false;
            n = n * 10 + (*d - '0');
            d++;
            any = true;
        }
        if (any) {
            *layer_out = n;
            return true;
        }
    }
    return false;
}

/* Which gpu owns layer N in the plan, or -1 if no gpu does. */
static int qw_plan_gpu(const qw_plan *plan, int layer)
{
    for (int g = 0; g < plan->n_gpu; g++) {
        if (layer >= plan->first_layer[g] &&
            layer < plan->first_layer[g] + plan->layers_per_gpu[g])
            return g;
    }
    return -1;
}

/* Case-insensitive ASCII substring test. */
static bool qw_contains_ci(const char *s, const char *needle)
{
    for (; *s != '\0'; s++) {
        const char *a = s, *b = needle;
        while (*a != '\0' && *b != '\0') {
            unsigned char ca = (unsigned char)*a, cb = (unsigned char)*b;
            char ua = (ca >= 'a' && ca <= 'z') ? (char)(ca - 'a' + 'A')
                                               : (char)ca;
            char ub = (cb >= 'a' && cb <= 'z') ? (char)(cb - 'a' + 'A')
                                               : (char)cb;
            if (ua != ub)
                break;
            a++;
            b++;
        }
        if (*b == '\0')
            return true;
    }
    return false;
}

/* --------------------------------------------------------------- verify */
qw_err qw_weights_verify(const qw_weights *w, const qw_plan *plan,
                         size_t pool_capacity)
{
    if (w == NULL || plan == NULL)
        return QW_ERR_NULL;
    if (plan->n_gpu < 1 || plan->n_gpu > QW_MAX_GPUS)
        return QW_ERR_RANGE;

    uint64_t used[QW_MAX_GPUS];
    size_t   cnt[QW_MAX_GPUS];
    for (int g = 0; g < plan->n_gpu; g++) {
        used[g] = 0;
        cnt[g] = 0;
    }

    for (size_t i = 0; i < w->n; i++) {
        const qw_weights_entry *e = &w->entries[i];
        int g = e->gpu;

        if (g == -1) {
            /* host tensor: no pool space, offset pinned at 0 */
            if (e->pool_offset != 0) {
                QW_LOGE("qw/weights: host tensor '%s' has pool_offset %llu "
                        "(expected 0)", e->name,
                        (unsigned long long)e->pool_offset);
                return QW_ERR_FORMAT;
            }
            continue;
        }
        if (g < 0 || g >= plan->n_gpu) {
            QW_LOGE("qw/weights: tensor '%s' on gpu %d (plan has %d gpus)",
                    e->name, g, plan->n_gpu);
            return QW_ERR_FORMAT;
        }
        cnt[g]++;
        used[g] += e->size;
        if (used[g] > pool_capacity) {
            QW_LOGE("qw/weights: gpu%d used %llu bytes > capacity %zu",
                    g, (unsigned long long)used[g], pool_capacity);
            return QW_ERR_RANGE;
        }
    }

    /* Non-overlap: within each pool (entries already in file order, which
     * is also the allocation order), no tensor may start before the previous
     * one ends. Alignment padding between tensors is allowed (and expected);
     * what is NOT allowed is offset_i < prev_offset + prev_size. */
    for (int g = 0; g < plan->n_gpu; g++) {
        uint64_t prev_end = 0;
        bool have_prev = false;
        for (size_t i = 0; i < w->n; i++) {
            const qw_weights_entry *e = &w->entries[i];
            if (e->gpu != g)
                continue;
            if (have_prev && e->pool_offset < prev_end) {
                QW_LOGE("qw/weights: gpu%d tensor '%s' at offset %llu "
                        "overlaps previous (ends %llu)",
                        g, e->name, (unsigned long long)e->pool_offset,
                        (unsigned long long)prev_end);
                return QW_ERR_FORMAT;
            }
            prev_end = e->pool_offset + e->size;
            have_prev = true;
        }
    }

    /* Total bytes accounted: host + per-gpu sums (computed above) must
     * equal the sum of all entry sizes — true iff no entry is counted
     * twice or missed. */
    uint64_t total = 0;
    for (size_t i = 0; i < w->n; i++)
        total += w->entries[i].size;
    uint64_t accounted = 0;
    for (int g = 0; g < plan->n_gpu; g++)
        accounted += used[g];
    for (size_t i = 0; i < w->n; i++)
        if (w->entries[i].gpu == -1)
            accounted += w->entries[i].size;
    if (accounted != total) {
        QW_LOGE("qw/weights: byte accounting mismatch (accounted %llu, "
                "entries %llu)", (unsigned long long)accounted,
                (unsigned long long)total);
        return QW_ERR_FORMAT;
    }
    return QW_OK;
}

/* --------------------------------------------------------------- report */
qw_err qw_weights_report(const qw_weights *w)
{
    if (w == NULL)
        return QW_ERR_NULL;
    if (w->n == 0) {
        printf("weights: (empty table)\n");
        return QW_OK;
    }

    /* Per-GPU aggregates. GPU slots are bounded by QW_MAX_GPUS. */
    uint64_t used[QW_MAX_GPUS];
    size_t   cnt[QW_MAX_GPUS];
    int      lmin[QW_MAX_GPUS], lmax[QW_MAX_GPUS];
    for (int g = 0; g < QW_MAX_GPUS; g++) {
        used[g] = 0;
        cnt[g] = 0;
        lmin[g] = -1;
        lmax[g] = -1;
    }
    uint64_t host_bytes = 0;
    size_t   host_cnt = 0;

    for (size_t i = 0; i < w->n; i++) {
        const qw_weights_entry *e = &w->entries[i];
        if (e->gpu == -1) {
            host_bytes += e->size;
            host_cnt++;
            continue;
        }
        if (e->gpu < 0 || e->gpu >= QW_MAX_GPUS)
            continue; /* malformed table; verify() is the place to catch it */
        used[e->gpu] += e->size;
        cnt[e->gpu]++;
        if (e->layer >= 0) {
            if (lmin[e->gpu] < 0 || e->layer < lmin[e->gpu])
                lmin[e->gpu] = e->layer;
            if (e->layer > lmax[e->gpu])
                lmax[e->gpu] = e->layer;
        }
    }

    int last = 0;
    for (int g = QW_MAX_GPUS - 1; g >= 0; g--)
        if (used[g] > 0 || cnt[g] > 0) {
            last = g;
            break;
        }

    printf("weights: %zu tensors placed\n", w->n);
    for (int g = 0; g <= last; g++) {
        uint64_t gap = 0;
        (void)qw_weights_largest_gap(w, g, &gap);
        if (lmin[g] >= 0)
            printf("  gpu%d: used=%llu bytes free=n/a tensors=%zu "
                   "layers %d..%d largest_gap=%llu\n",
                   g, (unsigned long long)used[g], cnt[g], lmin[g], lmax[g],
                   (unsigned long long)gap);
        else
            printf("  gpu%d: used=%llu bytes free=n/a tensors=%zu "
                   "layers - largest_gap=%llu\n",
                   g, (unsigned long long)used[g], cnt[g],
                   (unsigned long long)gap);
    }
    printf("  host: used=%llu bytes tensors=%zu (gpu -1, pinned host RAM)\n",
           (unsigned long long)host_bytes, host_cnt);
    return QW_OK;
}

/* ------------------------------------------------------------ largest gap */
qw_err qw_weights_largest_gap(const qw_weights *w, int gpu,
                              uint64_t *gap_out)
{
    if (w == NULL || gap_out == NULL)
        return QW_ERR_NULL;
    if (gpu < 0 || gpu >= QW_MAX_GPUS)
        return QW_ERR_RANGE;

    uint64_t prev_end = 0;
    uint64_t max_gap = 0;
    for (size_t i = 0; i < w->n; i++) {
        const qw_weights_entry *e = &w->entries[i];
        if (e->gpu != gpu)
            continue;
        uint64_t gap = e->pool_offset >= prev_end ?
                       e->pool_offset - prev_end : 0;
        if (gap > max_gap)
            max_gap = gap;
        prev_end = e->pool_offset + e->size;
    }
    *gap_out = max_gap;
    return QW_OK;
}

/* ------------------------------------------------------------------ plan */
qw_err qw_weights_plan(const struct gguf_file *f, const qw_plan *plan,
                       uint32_t align, qw_weights *out)
{
    if (f == NULL || plan == NULL || out == NULL)
        return QW_ERR_NULL;
    qw_weights_init(out);
    if (align == 0 || (align & (align - 1)) != 0) {
        QW_LOGE("qw/weights: align %u is not a power of two", align);
        return QW_ERR_RANGE;
    }
    if (plan->n_gpu < 1 || plan->n_gpu > QW_MAX_GPUS) {
        QW_LOGE("qw/weights: plan has %d gpus (want 1..%d)", plan->n_gpu,
                QW_MAX_GPUS);
        return QW_ERR_RANGE;
    }
    for (int g = 0; g < plan->n_gpu; g++) {
        if (plan->layers_per_gpu[g] < 1) {
            QW_LOGE("qw/weights: plan gpu%d has no layers", g);
            return QW_ERR_RANGE;
        }
    }

    size_t n = gguf_n_tensors(f);
    qw_err err = qw_weights_reserve(out, n);
    if (err != QW_OK)
        goto fail;

    /* Per-GPU cursors (bytes placed so far) and a seen-layer bitmap. */
    uint64_t cursor[QW_MAX_GPUS];
    for (int g = 0; g < QW_MAX_GPUS; g++)
        cursor[g] = 0;

    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_info *ti = gguf_tensor_by_index(f, i);
        if (ti == NULL) {
            err = QW_ERR_FORMAT;
            QW_LOGE("qw/weights: tensor %zu missing from gguf table", i);
            goto fail;
        }
        qw_weights_entry e;
        memset(&e, 0, sizeof(e));
        e.name = ti->name;
        e.size = gguf_tensor_nbytes(f, i);
        e.file_offset = gguf_tensor_offset(f, i);
        e.layer = -1;
        if (!qw_gguf_dt(ti->t, &e.dt)) {
            err = QW_ERR_UNSUPPORTED;
            QW_LOGE("qw/weights: unknown gguf dtype %s for tensor '%s'",
                    gguf_tensor_type_name(ti->t), ti->name);
            goto fail;
        }

        int gpu = -1;
        bool is_ple = qw_contains_ci(ti->name, "ple") ||
                      qw_contains_ci(ti->name, "ngram");
        /* Known unlayered device classes (case-insensitive). Anything
         * unlayered that matches NONE of these is an unrecognized naming
         * scheme -> fail loud, per the module policy. */
        bool is_device_unlayered =
            qw_contains_ci(ti->name, "embed") ||
            qw_contains_ci(ti->name, "norm") ||
            qw_contains_ci(ti->name, "head") ||
            qw_contains_ci(ti->name, "output") ||
            qw_contains_ci(ti->name, "mtp");

        int layer = -1;
        if (qw_extract_layer(ti->name, &layer)) {
            e.layer = layer;
            gpu = qw_plan_gpu(plan, layer);
            if (gpu < 0) {
                err = QW_ERR_FORMAT;
                QW_LOGE("qw/weights: tensor '%s' names layer %d which no "
                        "gpu in the plan owns", ti->name, layer);
                goto fail;
            }
            if (is_ple) {
                err = QW_ERR_FORMAT;
                QW_LOGE("qw/weights: tensor '%s' is both PLE/n-gram and "
                        "layer-numbered; PLE tables are unlayered",
                        ti->name);
                goto fail;
            }
        } else if (is_ple) {
            gpu = -1; /* host RAM, never VRAM */
        } else if (is_device_unlayered) {
            gpu = plan->n_gpu - 1; /* embedding, norm, lm_head, MTP */
        } else {
            err = QW_ERR_FORMAT;
            QW_LOGE("qw/weights: tensor '%s' matches no known naming "
                    "scheme (no layer, not PLE/n-gram, not embed/norm/"
                    "head/output/mtp)", ti->name);
            goto fail;
        }

        e.gpu = gpu;
        if (gpu >= 0) {
            uint64_t off = (cursor[gpu] + (uint64_t)align - 1) &
                           ~((uint64_t)align - 1);
            e.pool_offset = off;
            uint64_t end = off + e.size;
            if (end < off) {
                err = QW_ERR_RANGE;
                QW_LOGE("qw/weights: gpu%d pool offset overflow on '%s'",
                        gpu, ti->name);
                goto fail;
            }
            cursor[gpu] = end;
        }

        err = qw_weights_push(out, &e);
        if (err != QW_OK)
            goto fail;
    }

    return QW_OK;

fail:
    qw_weights_reset(out);
    return err;
}
