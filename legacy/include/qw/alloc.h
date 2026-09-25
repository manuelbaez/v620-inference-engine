/* qw/alloc.h — arena, device pool, pinned-memory helpers. Pure C17.
 *
 * Three allocators:
 *   qw_arena  — linear bump arena over a malloc'd block; for per-token
 *               scratch so the decode loop never calls malloc.
 *   qw_pool   — device-memory pool: ONE hipMalloc up front, first-fit
 *               allocation with coalescing on free. Never hipMallocs
 *               during decode. NOT lock-free; guarded by a mutex.
 *   qw_pinned — pinned host memory mapped into GPU address space
 *               (hipHostMalloc + hipHostGetDevicePointer). The device
 *               pointer MAY DIFFER from the host pointer: both are
 *               returned, host code must use the host pointer and device
 *               code the device pointer.
 *
 * Build with -DQW_WITH_HIP for the real paths; without it every function
 * returns QW_ERR_UNSUPPORTED (qw_arena is pure malloc and always works).
 *
 * Thread-safety: the pool and pinned registry are protected by an explicit
 * pthread mutex. They are NOT lock-free.
 */
#ifndef QW_ALLOC_H
#define QW_ALLOC_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ arena */

typedef struct qw_arena {
    uint8_t    *base;      /* malloc'd block */
    size_t      size;      /* block size */
    size_t      used;      /* current bump position */
    size_t      peak;      /* high-water mark of used */
} qw_arena;

/* size must be > 0. Returns NULL on failure (OOM / size 0). */
qw_arena *qw_arena_create(size_t size);

/* Bump-allocate `size` bytes with `align`. align must be a power of two and
 * <= size. NULL if inputs invalid or the arena is exhausted. */
void *qw_arena_alloc(qw_arena *a, size_t size, size_t align);

/* Reset the bump to the start; memory is NOT released. */
void qw_arena_reset(qw_arena *a);

void qw_arena_destroy(qw_arena *a); /* NULL-safe */

size_t qw_arena_peak(const qw_arena *a);

/* ------------------------------------------------------------------- pool */

typedef struct qw_pool qw_pool;

/* Device-memory stats: bytes reserved / allocated / high-water / free. */
typedef struct qw_pool_stat {
    size_t total;      /* bytes reserved up front */
    size_t allocated;  /* bytes currently allocated (aligned up) */
    size_t peak;       /* high-water mark of allocated */
    size_t free_bytes; /* total - allocated */
} qw_pool_stat;

/* Reserve `size` bytes of device memory on device `dev` (one hipMalloc).
 * size must be > 0. Returns NULL on failure. */
qw_pool *qw_pool_create(int dev, size_t size);

/* First-fit allocation of `size` bytes with `align` from the pool.
 * align must be a power of two and <= size. NULL-safe.
 * Never calls hipMalloc after creation. */
void *qw_pool_alloc(qw_pool *pool, size_t size, size_t align);

/* Free a pointer obtained from qw_pool_alloc. NULL-safe. Coalesces
 * adjacent free extents. */
void qw_pool_free(qw_pool *pool, void *ptr);

void qw_pool_destroy(qw_pool *pool); /* NULL-safe */

qw_err qw_pool_stats(qw_pool *pool, qw_pool_stat *out);

/* ----------------------------------------------------------------- pinned */

/* Pinned host memory mapped into GPU address space.
 *
 * Returns BOTH the host pointer (*host_pp) and the device pointer
 * (*dev_pp). On some systems these differ; device code must use the
 * device pointer, host code the host pointer.
 *
 * size must be > 0. QW_ERR_NULL if host_pp/dev_pp are NULL. */
qw_err qw_pinned_alloc(void **host_pp, void **dev_pp, size_t size);

/* Free a block obtained from qw_pinned_alloc. NULL-safe. */
void qw_pinned_free(void *host_ptr);

/* Register a pre-existing host allocation (e.g. the 47.7 GiB n-gram table
 * the host already owns) with hipHostRegister. Returns the device pointer
 * in *dev_pp. QW_ERR_NULL on bad inputs.
 *
 * NOTE: on some systems hipHostRegister does not actually pin the memory;
 * verify with a bandwidth benchmark before relying on it. */
qw_err qw_pinned_register(void *host_ptr, size_t size, void **dev_pp);

#ifdef __cplusplus
}
#endif

#endif /* QW_ALLOC_H */
