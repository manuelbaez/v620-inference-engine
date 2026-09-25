/* qw/weights.h — tensor-to-GPU placement table. Pure C17, libc only, no HIP.
 *
 * Phase-1 bookkeeping that turns an opened GGUF file plus a layer-split plan
 * (qw_plan_compute) into a flat table of (name, gpu, pool_offset, size) rows:
 * exactly the input phase 2 feeds to hipMemcpy. It computes placements only;
 * it never touches a device, so it runs on any laptop and is fully unit-
 * testable.
 *
 * Placement policy (documented so a rename is a decision, not an accident):
 *   - A tensor whose name contains ".blocks.<N>." or "layers.<N>." (N all
 *     digits) belongs to the GPU that owns layer N in the plan: the first g
 *     with first_layer[g] <= N < first_layer[g] + layers_per_gpu[g]. A
 *     numbered layer the plan does not cover, or a duplicate layer number in
 *     the file, fails the plan.
 *   - Tensors with no layer number are assigned by name class (case-
 *     insensitive substring):
 *       * PLE / n-gram table (name contains "ple" or "ngram"): HOST RAM,
 *         gpu == -1. It is ~47.7 GiB of fp8 and must never occupy a device
 *         pool.
 *       * device unlayered (name contains "embed", "norm", "head", "output",
 *         or "mtp"): the LAST gpu. This covers the token embedding, the
 *         final norm, the lm_head (output), and the tiny MTP module.
 *       * anything else: UNCLASSIFIABLE -> QW_ERR_FORMAT naming the tensor.
 *         This is deliberate: a new or renamed tensor that no rule recognizes
 *         fails loudly instead of being silently dumped on the last gpu.
 *   - pool_offset is assigned per GPU in file order: each tensor's offset is
 *     the current per-GPU cursor, rounded up to `align` (pass 256 for device
 *     pools), and the cursor advances by the ROUNDED-UP size. Offsets within
 *     one pool are therefore monotonic, aligned, and non-overlapping. Host
 *     tensors get gpu == -1, pool_offset == 0; they do not consume pool
 *     space (phase 2 reads them straight from the pinned host copy).
 *
 * Fail-loud: a tensor that matches no rule above returns QW_ERR_FORMAT and
 * names the offender (also via QW_LOGE), so a new tensor naming scheme fails
 * loudly instead of silently misplacing weights.
 *
 * Sizes come from the GGUF reader (gguf_tensor_nbytes), which packs int4
 * 2-per-byte, so sub-byte element dtypes are accounted exactly.
 */
#ifndef QW_WEIGHTS_H
#define QW_WEIGHTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/shard.h"
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gguf_file;

/* One placed tensor. `name` aliases the gguf file's string and is valid
 * while the file is open; `gpu` is -1 for host-resident tensors;
 * `pool_offset` is the byte offset inside the owning GPU's weight pool (0
 * for host tensors); `size` is the tensor's exact storage bytes (gguf
 * tensor nbytes, int4-packed); `file_offset` is the absolute offset of the
 * same bytes in the gguf file (the hipMemcpy source). */
typedef struct qw_weights_entry {
    const char *name;
    int         gpu;
    uint64_t    pool_offset;
    uint64_t    size;
    uint64_t    file_offset;
    qw_dtype    dt;
    int         layer;  /* -1 = unlayered (embedding / norm / head / PLE) */
} qw_weights_entry;

/* The full placement table. Entries are in gguf file order. */
typedef struct qw_weights {
    qw_weights_entry *entries;
    size_t            n;
    size_t            cap;
} qw_weights;

/* Build the placement table for every tensor in f under plan.
 *
 * align must be a power of two (256 for device pools). Per-GPU offsets
 * start at 0 and each tensor is aligned up to `align`; see the module
 * header for the full policy.
 *
 * Errors: QW_ERR_NULL on NULL f/plan/out; QW_ERR_RANGE on a bad align,
 * n_gpu < 1, or a plan whose layer runs are empty / out of order;
 * QW_ERR_FORMAT on a layer number the plan does not cover, a duplicate
 * layer number, or a tensor no rule classifies (the offending name is in
 * the log); QW_ERR_ALLOC on OOM. */
qw_err qw_weights_plan(const struct gguf_file *f, const qw_plan *plan,
                       uint32_t align, qw_weights *out);

/* Verify a placement table:
 *   - every tensor has a sane gpu (host -1, or 0 .. n_gpu-1),
 *   - per GPU no two tensors overlap in [pool_offset, pool_offset + size)
 *     (the per-GPU total doubles as the used-pool size),
 *   - per GPU used bytes <= pool_capacity,
 *   - every host tensor sits at pool_offset 0 (no phantom pool use),
 *   - every tensor is accounted for exactly once (sum of sizes == sum of
 *     sizes; a plan that drops or duplicates a tensor breaks the per-GPU
 *     arithmetic above).
 * pool_capacity is the byte capacity of ONE device pool (all pools are
 * sized identically). Returns QW_OK if the table is sound. */
qw_err qw_weights_verify(const qw_weights *w, const qw_plan *plan,
                         size_t pool_capacity);

/* Print per GPU: bytes used, bytes free (vs pool_capacity, printed only if
 * > 0), tensor count, layer range, largest pool gap; then one line for
 * host-resident tensors. Returns QW_OK. */
qw_err qw_weights_report(const qw_weights *w);

/* Largest single gap in a device pool: the maximum
 * next.pool_offset - (prev.pool_offset + prev.size) over the pool's
 * tensors in file order, including the leading gap from 0 and the trailing
 * gap to the pool end (the per-GPU total used bytes). QW_ERR_NULL on NULL
 * w/gap_out; QW_ERR_RANGE on a bad gpu index. */
qw_err qw_weights_largest_gap(const qw_weights *w, int gpu,
                              uint64_t *gap_out);

#ifdef __cplusplus
}
#endif

#endif /* QW_WEIGHTS_H */
