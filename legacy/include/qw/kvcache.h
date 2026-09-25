/* qw/kvcache.h — paged KV-cache block manager. Pure C17, libc only.
 *
 * Qwen3.8-Flash-Next: 48 layers, only the 12 QSA (full-attention) layers
 * keep KV cache; the 36 GDN layers are linear attention (recurrent state,
 * managed separately — NOT here). Per token, per QSA layer the cache holds
 * K and V, each n_kv_heads x head_dim fp16:
 *
 *   bytes_per_token_per_layer = 2(K,V) * n_kv_heads * head_dim * 2 B(fp16)
 *
 * For the default model (2 kv_heads, head_dim 256) that is 2048 B; across
 * all 12 QSA layers a single token occupies 24576 B — the same figure
 * asserted by qw_model_estimate() (config.h: kv_per_token).
 *
 * Storage is PAGED. The whole cache is n_blocks identical blocks, each
 * tokens_per_block token slots. A block's slot layout is layer-major (the
 * kv_layer axis is the innermost):
 *
 *   byte_off = ((block_id * tokens_per_block + pos_in_block)
 *               * n_kv_layers + kv_layer) * bytes_per_token_per_layer
 *
 * kv_layer runs 0..n_kv_layers-1 as a compact index over the 12 QSA layers
 * (NOT the physical 0..47 layer index; the 36 GDN layers have no slots).
 *
 * Why layer is the innermost stride: for a given token, all 12 layers sit
 * contiguously (12 x 2048 = 24576 B, one cache line per 128 B of it).
 * Decode reads the current token across every QSA layer — one contiguous
 * burst instead of 12 scattered reads. Prefill reads one layer over many
 * tokens and pays the strided access, but prefill is compute-bound.
 *
 * Allocation is a free list of block ids; each sequence owns a growable
 * page table (array of block ids). Pages are handed out one block at a
 * time as a sequence grows, so sequences need no contiguous space and
 * context can grow without reallocation or copying (block contents never
 * move; only the page-table array is realloc'd).
 *
 * A block holds tokens_per_block * n_kv_layers * bytes_per_token_per_layer
 * bytes (6 MiB for the default model); the free list itself is one
 * uint32_t (block id) per block.
 */
#ifndef QW_KVCACHE_H
#define QW_KVCACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- config */
typedef struct qw_kvcfg {
    uint32_t n_kv_layers;            /* QSA layers with KV cache (12)       */
    uint32_t n_kv_heads;             /* KV heads per layer, GQA (2)         */
    uint32_t head_dim;               /* head dimension (256)                */
    uint32_t tokens_per_block;       /* token slots per block (256)         */
    uint64_t n_blocks;               /* total blocks; 0 = derive from budget */
    uint32_t bytes_per_token_per_layer; /* 2*n_kv_heads*head_dim*2; 2048    */
} qw_kvcfg;

/* Default config for Qwen3.8-Flash-Next: 12 KV layers, 2 kv heads,
 * head_dim 256, 256 tokens per block. n_blocks is derived from vram_budget
 * (total cache bytes available): the number of whole blocks that fit,
 * i.e. floor(vram_budget / (tokens_per_block * 24576)). QW_ERR_RANGE if
 * the budget fits zero blocks. bytes_per_token_per_layer is always 2048
 * for these dims. */
qw_err qw_kvcfg_default(uint64_t vram_budget, qw_kvcfg *out);

/* ---------------------------------------------------------- cache handle */
typedef struct qw_kvcache qw_kvcache;

/* Cache statistics. n_tokens = tokens currently reserved (summed over
 * sequences); free_blocks = blocks back on the free list; peak_* are
 * high-water marks since init. */
typedef struct qw_kv_stats {
    uint64_t n_tokens;      /* tokens currently reserved */
    uint64_t peak_tokens;   /* high-water mark of n_tokens */
    uint64_t free_blocks;   /* blocks currently free */
    uint64_t peak_blocks;   /* high-water mark of in-use blocks */
} qw_kv_stats;

/* Create the cache: build the free list (all n_blocks free) and max_seqs
 * empty sequence slots. QW_ERR_NULL on NULL out/cfg; QW_ERR_RANGE if
 * max_seqs <= 0 or the cfg is invalid (zero fields, or n_blocks == 0);
 * QW_ERR_ALLOC if memory allocation fails. After a failed init the handle
 * is fully cleaned up. */
qw_err qw_kvcache_init(qw_kvcache **out, const qw_kvcfg *cfg, int max_seqs);

/* Destroy the cache and its page tables. NULL-safe. */
void qw_kvcache_destroy(qw_kvcache *c);

/* ------------------------------------------------------------- lifecycle */
/* Reserve storage for n_tokens tokens on sequence seq (0..max_seqs-1),
 * extending its page table and pulling blocks from the free list as
 * needed. Never overcommits: if the free list cannot supply
 * ceil(n_tokens / tokens_per_block) blocks it returns QW_ERR_NOMEM and
 * leaves the sequence EXACTLY as it was (no partial allocation).
 *
 * *blocks_taken (may be NULL) receives how many blocks were newly taken
 * by THIS call (0 if the page table already covered n_tokens).
 *
 * QW_ERR_NULL on NULL c/seq out of bounds (seq >= max_seqs); QW_ERR_NOMEM
 * on free-list exhaustion. */
qw_err qw_kvcache_reserve(qw_kvcache *c, int seq, uint64_t n_tokens,
                          uint64_t *blocks_taken);

/* Byte offset of one token's slot at kv_layer, inside sequence seq's
 * reserved blocks. This is what the attention kernel calls — a pure
 * arithmetic lookup, O(1), no search:
 *
 *   block_id    = page_table[seq][token_pos / tokens_per_block]
 *   pos_in_blk  = token_pos % tokens_per_block
 *   byte_off    = ((block_id * tokens_per_block + pos_in_blk)
 *                  * n_kv_layers + kv_layer) * bytes_per_token_per_layer
 *
 * Bounds-checked: seq in [0, max_seqs), token_pos < reserved tokens for
 * seq (QW_ERR_RANGE otherwise — do NOT call before reserving the token),
 * kv_layer in [0, n_kv_layers). */
qw_err qw_kvcache_offset(const qw_kvcache *c, int seq, uint64_t token_pos,
                         int kv_layer, uint64_t *byte_off_out);

/* Release sequence seq entirely: return every one of its blocks to the
 * free list and reset its page table. The sequence may be reserved again
 * afterwards. QW_ERR_NULL on NULL c / seq out of bounds. Releasing an
 * already-empty sequence is a no-op (QW_OK). */
qw_err qw_kvcache_release(qw_kvcache *c, int seq);

/* Speculative-decoding rollback: discard the last n_tokens tokens of
 * sequence seq, freeing exactly the blocks that held ONLY those tokens.
 * Trailing blocks that would become fully empty are returned to the free
 * list; a partially-filled final block (still holding kept tokens) is
 * KEPT even though it has free slots. Frees exactly
 *   ceil((n_prev - n_keep) / tokens_per_block) blocks
 * where n_prev is the pre-rollback reserved count and n_keep = n_prev -
 * n_tokens — which is the number of boundary crossings among the discarded
 * tokens' blocks. n_tokens must be <= reserved tokens (QW_ERR_RANGE
 * otherwise, including n_tokens == 0). */
qw_err qw_kvcache_release_tail(qw_kvcache *c, int seq, uint64_t n_tokens);

/* Snapshot stats into out. QW_ERR_NULL on NULL c/out. */
qw_err qw_kvcache_stats(const qw_kvcache *c, qw_kv_stats *out);

#ifdef __cplusplus
}
#endif

#endif /* QW_KVCACHE_H */
