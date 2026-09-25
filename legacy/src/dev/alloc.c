/* src/dev/alloc.c — arena, device pool, pinned-memory helpers.
 *
 * qw_arena : linear bump arena over a malloc'd block (always available).
 * qw_pool  : one big hipMalloc up front, first-fit over a free-extent
 *            array with coalescing on free. NOT lock-free; a pthread
 *            mutex guards the extent bookkeeping.
 * qw_pinned: hipHostMalloc(hipHostAllocMapped) + hipHostGetDevicePointer;
 *            returns BOTH pointers because the device pointer may differ
 *            from the host pointer.
 *
 * Without -DQW_WITH_HIP the device/pinned paths return
 * QW_ERR_UNSUPPORTED; qw_arena still works (plain malloc).
 */
#include "qw/alloc.h"
#include "qw/device.h"
#include "qw/macros.h"
#include "qw/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef QW_WITH_HIP
#include <hip/hip_runtime_api.h>
#endif

static int is_pow2(size_t x) { return x != 0 && (x & (x - 1)) == 0; }

static size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

/* ================================================================ arena */

qw_arena *qw_arena_create(size_t size) {
    if (size == 0)
        return NULL;
    qw_arena *a = (qw_arena *)calloc(1, sizeof(qw_arena));
    if (!a)
        return NULL;
    a->base = (uint8_t *)malloc(size);
    if (!a->base) {
        free(a);
        return NULL;
    }
    a->size = size;
    a->used = 0;
    a->peak = 0;
    return a;
}

void *qw_arena_alloc(qw_arena *a, size_t size, size_t align) {
    if (!a)
        return NULL;
    if (size == 0 || !is_pow2(align) || align > size)
        return NULL;
    size_t off = align_up(a->used, align);
    if (off + size > a->size)
        return NULL;
    uint8_t *p = a->base + off;
    a->used = off + size;
    if (a->used > a->peak)
        a->peak = a->used;
    return p;
}

void qw_arena_reset(qw_arena *a) {
    if (a)
        a->used = 0;
}

void qw_arena_destroy(qw_arena *a) {
    if (!a)
        return;
    free(a->base);
    free(a);
}

size_t qw_arena_peak(const qw_arena *a) {
    return a ? a->peak : 0;
}

/* ================================================================= pool */

/* Header-free bookkeeping: every live allocation is tracked by its offset
 * into the big block. Free space is tracked separately as a list of
 * extents (f_off/f_len). */
typedef struct {
    size_t off;      /* offset from block base */
    size_t len;      /* reserved (alignment-inclusive) length */
} pool_slot;

struct qw_pool {
    void        *base;
    size_t       size;
    int          dev;
    size_t       allocated;
    size_t       peak;

    /* Free-extent table (dynamically grown). */
    size_t      *f_off;
    size_t      *f_len;
    int          f_count;
    int          f_cap;

    /* Allocation table: maps user pointer -> slot. */
    pool_slot  *slots;
    int         n_slots;
    int         slot_cap;

    pthread_mutex_t lock; /* NOT lock-free */
};

static int pool_extend_f(qw_pool *p) {
    int ncap = p->f_cap ? p->f_cap * 2 : 8;
    size_t *no = (size_t *)realloc(p->f_off, (size_t)ncap * sizeof(size_t));
    if (!no)
        return -1;
    p->f_off = no;
    size_t *nl = (size_t *)realloc(p->f_len, (size_t)ncap * sizeof(size_t));
    if (!nl)
        return -1;
    p->f_len = nl;
    p->f_cap = ncap;
    return 0;
}

/* Append free extent (off,len). Caller must hold the lock. */
static int pool_free_extent_add(qw_pool *p, size_t off, size_t len) {
    if (p->f_count == p->f_cap) {
        if (pool_extend_f(p) != 0)
            return -1;
    }
    p->f_off[p->f_count] = off;
    p->f_len[p->f_count] = len;
    p->f_count++;
    return 0;
}

static qw_pool *pool_create_raw(int dev, size_t size) {
    qw_pool *p = (qw_pool *)calloc(1, sizeof(qw_pool));
    if (!p)
        return NULL;
    pthread_mutex_init(&p->lock, NULL);
    p->dev = dev;
    p->size = size;
    return p;
}

qw_pool *qw_pool_create(int dev, size_t size) {
    if (size == 0)
        return NULL;
    qw_pool *p = pool_create_raw(dev, size);
    if (!p)
        return NULL;

#ifdef QW_WITH_HIP
    if (hipMalloc(&p->base, size) != hipSuccess) {
        QW_LOGE("pool: hipMalloc(%zu) failed on dev %d", size, dev);
        pthread_mutex_destroy(&p->lock);
        free(p);
        return NULL;
    }
    if (pool_free_extent_add(p, 0, size) != 0) {
        hipFree(p->base);
        pthread_mutex_destroy(&p->lock);
        free(p);
        return NULL;
    }
#else
    (void)dev;
    QW_LOGE("pool: hipMalloc unavailable (built without -DQW_WITH_HIP)");
    pthread_mutex_destroy(&p->lock);
    free(p);
    return NULL;
#endif
    return p;
}

void *qw_pool_alloc(qw_pool *pool, size_t size, size_t align) {
    if (!pool || size == 0 || !is_pow2(align) || align > size)
        return NULL;

    pthread_mutex_lock(&pool->lock);

    /* First-fit over free extents; split the best one. */
    int best = -1;
    size_t best_slack = 0;
    for (int i = 0; i < pool->f_count; i++) {
        size_t off = align_up(pool->f_off[i], align);
        size_t end = pool->f_off[i] + pool->f_len[i];
        if (off + size > end)
            continue;
        size_t slack = (end - off) - size;
        if (best == -1 || slack < best_slack) {
            best = i;
            best_slack = slack;
        }
    }
    if (best == -1) {
        pthread_mutex_unlock(&pool->lock);
        return NULL;
    }

    size_t f_off = pool->f_off[best];
    size_t f_len = pool->f_len[best];
    size_t off   = align_up(f_off, align);
    size_t used  = off + size - f_off;

    /* Remove the best extent; re-add head/tail remainders. */
    pool->f_off[best] = pool->f_off[pool->f_count - 1];
    pool->f_len[best] = pool->f_len[pool->f_count - 1];
    pool->f_count--;

    if (off > f_off) {
        if (pool_free_extent_add(pool, f_off, off - f_off) != 0) {
            /* Re-add the whole original extent so state stays sane. */
            pool_free_extent_add(pool, f_off, f_len);
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
    }
    if (off + size < f_off + f_len) {
        if (pool_free_extent_add(pool, off + size,
                                 (f_off + f_len) - (off + size)) != 0) {
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
    }

    /* Record the allocation so qw_pool_free can find it. */
    if (pool->n_slots == pool->slot_cap) {
        int ncap = pool->slot_cap ? pool->slot_cap * 2 : 32;
        pool_slot *ns = (pool_slot *)realloc(pool->slots,
                                             (size_t)ncap * sizeof(pool_slot));
        if (!ns) {
            /* Roll back the extent surgery: re-add what we took. */
            pool_free_extent_add(pool, off, used);
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
        pool->slots = ns;
        pool->slot_cap = ncap;
    }
    pool_slot *s = &pool->slots[pool->n_slots];
    s->off = off;
    s->len = used;

    pool->allocated += used;
    if (pool->allocated > pool->peak)
        pool->peak = pool->allocated;

    pthread_mutex_unlock(&pool->lock);
    return (uint8_t *)pool->base + off;
}

/* Merge all adjacent free extents into one sorted, coalesced list.
 * Caller holds the lock. Complexity O(n^2) — fine for the hundreds of
 * extents expected in a decode pool. */
static void pool_coalesce(qw_pool *p) {
    /* Collect extents. */
    int n = p->f_count;
    if (n == 0)
        return;
    size_t *offs = (size_t *)malloc((size_t)n * sizeof(size_t));
    size_t *lens = (size_t *)malloc((size_t)n * sizeof(size_t));
    if (!offs || !lens) {
        free(offs);
        free(lens);
        return; /* leave list unsorted; still correct, just less coalesced */
    }
    for (int i = 0; i < n; i++) {
        offs[i] = p->f_off[i];
        lens[i] = p->f_len[i];
    }
    /* Insertion sort by offset. */
    for (int i = 1; i < n; i++) {
        size_t o = offs[i], l = lens[i];
        int j = i - 1;
        while (j >= 0 && offs[j] > o) {
            offs[j + 1] = offs[j];
            lens[j + 1] = lens[j];
            j--;
        }
        offs[j + 1] = o;
        lens[j + 1] = l;
    }
    /* Merge adjacent/overlapping in place. */
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m > 0 && offs[i] <= offs[m - 1] + lens[m - 1]) {
            size_t end = offs[m - 1] + lens[m - 1];
            if (offs[i] + lens[i] > end)
                lens[m - 1] = offs[i] + lens[i] - offs[m - 1];
        } else {
            offs[m] = offs[i];
            lens[m] = lens[i];
            m++;
        }
    }
    for (int i = 0; i < m; i++) {
        p->f_off[i] = offs[i];
        p->f_len[i] = lens[i];
    }
    p->f_count = m;
    free(offs);
    free(lens);
}



void qw_pool_free(qw_pool *pool, void *ptr) {
    if (!pool || !ptr)
        return;
    uint8_t *base = (uint8_t *)pool->base;
    if ((uint8_t *)ptr < base || (uint8_t *)ptr >= base + pool->size)
        return; /* not from this pool */

    pthread_mutex_lock(&pool->lock);

    int found = -1;
    for (int i = 0; i < pool->n_slots; i++) {
        if (pool->slots[i].off == (size_t)((uint8_t *)ptr - base)) {
            found = i;
            break;
        }
    }
    if (found == -1) {
        QW_LOGE("pool_free: pointer not from this pool");
        pthread_mutex_unlock(&pool->lock);
        return;
    }

    size_t off = pool->slots[found].off;
    size_t len = pool->slots[found].len;
    pool->slots[found] = pool->slots[pool->n_slots - 1];
    pool->n_slots--;

    pool->allocated -= len;

    if (pool_free_extent_add(pool, off, len) != 0) {
        /* Cannot happen unless OOM on the extent table; allocation stays
         * marked dead. Logged by caller-visible error path is not
         * possible here; best-effort only. */
        QW_LOGE("pool_free: extent table OOM; block leaked");
    }
    pool_coalesce(pool);

    pthread_mutex_unlock(&pool->lock);
}

qw_err qw_pool_stats(qw_pool *pool, qw_pool_stat *out) {
    if (!pool || !out)
        return QW_ERR_NULL;
    pthread_mutex_lock(&pool->lock);
    out->total = pool->size;
    out->allocated = pool->allocated;
    out->peak = pool->peak;
    out->free_bytes = pool->size - pool->allocated;
    pthread_mutex_unlock(&pool->lock);
    return QW_OK;
}

void qw_pool_destroy(qw_pool *pool) {
    if (!pool)
        return;
#ifdef QW_WITH_HIP
    if (pool->base)
        hipFree(pool->base);
#endif
    free(pool->f_off);
    free(pool->f_len);
    free(pool->slots);
    pthread_mutex_destroy(&pool->lock);
    free(pool);
}

/* ============================================================== pinned */

#ifdef QW_WITH_HIP
/* Registry of pinned blocks so we can tell hipFreeHost (hipHostMalloc'd)
 * from hipHostUnregister (hipHostRegister'd). NOT thread-safe; protected
 * by the same conventions as the rest of this file (call from one thread
 * or hold your own lock). */
typedef struct {
    void   *host;
    size_t  size;
    int     registered; /* 1 if via hipHostRegister */
    int     freed;
} pinned_rec;

static pinned_rec *g_pinned     = NULL;
static int         g_pinned_n   = 0;
static int         g_pinned_cap = 0;

static int pinned_track(void *host, size_t size, int registered) {
    if (g_pinned_n == g_pinned_cap) {
        int ncap = g_pinned_cap ? g_pinned_cap * 2 : 16;
        pinned_rec *nr =
            (pinned_rec *)realloc(g_pinned, (size_t)ncap * sizeof(pinned_rec));
        if (!nr)
            return -1;
        g_pinned = nr;
        g_pinned_cap = ncap;
    }
    g_pinned[g_pinned_n].host = host;
    g_pinned[g_pinned_n].size = size;
    g_pinned[g_pinned_n].registered = registered;
    g_pinned[g_pinned_n].freed = 0;
    g_pinned_n++;
    return 0;
}
#endif

qw_err qw_pinned_alloc(void **host_pp, void **dev_pp, size_t size) {
    if (!host_pp || !dev_pp || size == 0)
        return QW_ERR_NULL;
    *host_pp = NULL;
    *dev_pp = NULL;
#ifdef QW_WITH_HIP
    void *host = NULL;
    if (hipHostMalloc(&host, size, hipHostAllocMapped) != hipSuccess) {
        QW_LOGE("hipHostMalloc(%zu) failed", size);
        return QW_ERR_NOMEM;
    }
    void *dev = NULL;
    if (hipHostGetDevicePointer(&dev, host, 0) != hipSuccess) {
        QW_LOGE("hipHostGetDevicePointer failed for %p", host);
        hipFreeHost(host);
        return QW_ERR_DEVICE;
    }
    if (pinned_track(host, size, 0) != 0) {
        hipFreeHost(host);
        return QW_ERR_ALLOC;
    }
    *host_pp = host;
    *dev_pp = dev;
    return QW_OK;
#else
    QW_LOGE("pinned_alloc unavailable (built without -DQW_WITH_HIP)");
    return QW_ERR_UNSUPPORTED;
#endif
}

void qw_pinned_free(void *host_ptr) {
    if (!host_ptr)
        return;
#ifdef QW_WITH_HIP
    int found = -1;
    for (int i = 0; i < g_pinned_n; i++) {
        if (g_pinned[i].host == host_ptr && !g_pinned[i].freed) {
            found = i;
            break;
        }
    }
    if (found != -1) {
        g_pinned[found].freed = 1;
        g_pinned[found] = g_pinned[g_pinned_n - 1];
        g_pinned_n--;
        if (g_pinned[found].registered) {
            /* NOTE: only unregistered when the pointer was obtained from
             * qw_pinned_register on the SAME device context; on a multi-GPU
             * system the caller is responsible for calling this from the
             * device that registered it. For the common single-device case
             * this is correct. */
            hipHostUnregister(host_ptr);
        } else {
            hipFreeHost(host_ptr);
        }
    }
    /* Unknown pointer: best-effort hipFreeHost (safe no-op if not
     * hipHostMalloc'd on this HIP version; on others it may assert).
     * Log and skip to avoid double-free. */
    else {
        QW_LOGE("pinned_free: pointer %p not in registry; skipping", host_ptr);
    }
#endif
}

qw_err qw_pinned_register(void *host_ptr, size_t size, void **dev_pp) {
    if (!host_ptr || !dev_pp || size == 0)
        return QW_ERR_NULL;
    *dev_pp = NULL;
#ifdef QW_WITH_HIP
    /* NOTE: on some systems hipHostRegister does not actually pin memory. */
    if (hipHostRegister(host_ptr, size, hipHostRegisterDefault) != hipSuccess) {
        QW_LOGE("hipHostRegister(%p, %zu) failed", host_ptr, size);
        return QW_ERR_DEVICE;
    }
    void *dev = NULL;
    if (hipHostGetDevicePointer(&dev, host_ptr, 0) != hipSuccess) {
        QW_LOGE("hipHostGetDevicePointer failed for registered %p", host_ptr);
        hipHostUnregister(host_ptr);
        return QW_ERR_DEVICE;
    }
    if (pinned_track(host_ptr, size, 1) != 0) {
        hipHostUnregister(host_ptr);
        return QW_ERR_ALLOC;
    }
    *dev_pp = dev;
    return QW_OK;
#else
    (void)host_ptr;
    (void)size;
    QW_LOGE("pinned_register unavailable (built without -DQW_WITH_HIP)");
    return QW_ERR_UNSUPPORTED;
#endif
}
