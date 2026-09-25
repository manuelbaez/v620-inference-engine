/* src/core/kvcache.c — paged KV-cache block manager (index math only).
 *
 * No device pointers here: this module owns the free list and per-sequence
 * page tables and answers byte-offset queries by arithmetic. The actual
 * block memory (device-side) and the pointer that offsets index into it
 * are wired up by the device layer later.
 *
 * Address formula (see kvcache.h for the layout rationale — layer is the
 * innermost stride so one token's 12 QSA layers are contiguous):
 *
 *   byte_off = ((block_id * tokens_per_block + pos_in_block)
 *               * n_kv_layers + kv_layer) * bytes_per_token_per_layer
 */
#include "qw/kvcache.h"
#include "qw/macros.h"

#include <stdlib.h>
#include <string.h>

/* Overflow-checked multiply: *r = a*b, QW_ERR_RANGE on overflow. */
static qw_err kv_umul(uint64_t a, uint64_t b, uint64_t *r)
{
    if (a != 0 && b > UINT64_MAX / a)
        return QW_ERR_RANGE;
    *r = a * b;
    return QW_OK;
}

/* ---------------------------------------------------------------- types */
typedef struct kv_seq {
    uint32_t *blocks;   /* page table: block id per block slot, or NULL */
    uint64_t  n_blocks; /* blocks currently assigned */
    uint64_t  n_tokens; /* tokens reserved (may be < n_blocks*tokens_per_block) */
} kv_seq;

struct qw_kvcache {
    qw_kvcfg cfg;
    int      max_seqs;

    uint32_t *free_list;  /* stack of free block ids */
    uint64_t  free_top;   /* count of free ids (free_list[0..free_top-1]) */

    kv_seq   *seqs;       /* max_seqs sequence slots */

    uint64_t  peak_blocks;  /* high-water in-use blocks */
    uint64_t  peak_tokens;  /* high-water reserved tokens */
};

/* --------------------------------------------------------------- cfg */
qw_err qw_kvcfg_default(uint64_t vram_budget, qw_kvcfg *out)
{
    if (out == NULL)
        return QW_ERR_NULL;

    memset(out, 0, sizeof(*out));
    out->n_kv_layers  = 12;   /* QSA layers: indices 3,7,...,47 */
    out->n_kv_heads   = 2;    /* GQA 12:1 */
    out->head_dim     = 256;
    out->tokens_per_block = 256;

    /* per token per layer: 2(K,V) * n_kv_heads * head_dim * 2 B (fp16) */
    uint64_t bptl;
    if (kv_umul(2ULL * (uint64_t)out->n_kv_heads * (uint64_t)out->head_dim,
                2ULL, &bptl) != QW_OK)
        return QW_ERR_RANGE;
    out->bytes_per_token_per_layer = (uint32_t)bptl; /* 2048 */

    /* per block: tokens_per_block tokens * 12 layers * bytes_per_token_per_layer */
    uint64_t block_bytes;
    if (kv_umul((uint64_t)out->tokens_per_block,
                (uint64_t)out->n_kv_layers * bptl, &block_bytes) != QW_OK)
        return QW_ERR_RANGE;
    if (block_bytes == 0)
        return QW_ERR_RANGE;

    out->n_blocks = vram_budget / block_bytes;
    if (out->n_blocks == 0)
        return QW_ERR_RANGE;
    return QW_OK;
}

/* --------------------------------------------------------------- init */
qw_err qw_kvcache_init(qw_kvcache **out, const qw_kvcfg *cfg, int max_seqs)
{
    if (out == NULL || cfg == NULL)
        return QW_ERR_NULL;
    *out = NULL;

    if (max_seqs <= 0 || cfg->n_kv_layers == 0 || cfg->n_kv_heads == 0 ||
        cfg->head_dim == 0 || cfg->tokens_per_block == 0 ||
        cfg->n_blocks == 0 ||
        cfg->bytes_per_token_per_layer !=
            2u * cfg->n_kv_heads * cfg->head_dim * 2u)
        return QW_ERR_RANGE;

    /* per-token-per-layer bytes must divide evenly into the block slot math
     * (it does for this model: 2048 is a multiple of 32). */
    if (cfg->bytes_per_token_per_layer % 32u != 0)
        return QW_ERR_RANGE;

    qw_kvcache *c = calloc(1, sizeof(*c));
    if (c == NULL)
        return QW_ERR_ALLOC;
    c->cfg      = *cfg;
    c->max_seqs = max_seqs;

    /* free list: all blocks free, id == array index (LIFO top == n_blocks-1) */
    c->free_list = malloc((size_t)cfg->n_blocks * sizeof(uint32_t));
    if (c->free_list == NULL) {
        free(c);
        return QW_ERR_ALLOC;
    }
    for (uint64_t i = 0; i < cfg->n_blocks; i++)
        c->free_list[i] = (uint32_t)i;
    c->free_top = cfg->n_blocks;

    c->seqs = calloc((size_t)max_seqs, sizeof(kv_seq));
    if (c->seqs == NULL) {
        free(c->free_list);
        free(c);
        return QW_ERR_ALLOC;
    }

    *out = c;
    return QW_OK;
}

void qw_kvcache_destroy(qw_kvcache *c)
{
    if (c == NULL)
        return;
    for (int i = 0; i < c->max_seqs; i++)
        free(c->seqs[i].blocks);
    free(c->seqs);
    free(c->free_list);
    free(c);
}

/* -------------------------------------------------------------- helpers */
/* Push a block id onto the free list. */
static void kv_push_free(qw_kvcache *c, uint32_t id)
{
    c->free_list[c->free_top++] = id;
}

/* Pop a block id off the free list. Assumes free_top > 0. */
static uint32_t kv_pop_free(qw_kvcache *c)
{
    return c->free_list[--c->free_top];
}

/* Number of block slots needed to cover n_tokens. */
static uint64_t kv_blocks_for(uint64_t n_tokens, uint32_t tpb)
{
    return (n_tokens + tpb - 1) / tpb;
}

/* Grow seq's page table to hold need_blocks entries (need_blocks >
 * current). Returns QW_OK / QW_ERR_ALLOC. */
static qw_err kv_seq_grow(kv_seq *s, uint64_t need_blocks)
{
    uint32_t *nb = realloc(s->blocks,
                           (size_t)need_blocks * sizeof(uint32_t));
    if (nb == NULL)
        return QW_ERR_ALLOC;
    s->blocks = nb;
    return QW_OK;
}

/* ------------------------------------------------------------- reserve */
qw_err qw_kvcache_reserve(qw_kvcache *c, int seq, uint64_t n_tokens,
                          uint64_t *blocks_taken)
{
    if (c == NULL || blocks_taken == NULL)
        return QW_ERR_NULL;
    *blocks_taken = 0;
    if (seq < 0 || seq >= c->max_seqs)
        return QW_ERR_NULL;

    kv_seq *s = &c->seqs[seq];
    uint64_t need = kv_blocks_for(n_tokens, c->cfg.tokens_per_block);
    if (need <= s->n_blocks) {
        /* page table already covers it; just record the token count */
        if (n_tokens > s->n_tokens)
            s->n_tokens = n_tokens;
        return QW_OK;
    }

    uint64_t extra = need - s->n_blocks;
    if (extra > c->free_top)
        return QW_ERR_NOMEM; /* no partial allocation */

    /* grow page table, then hand out blocks. If realloc failed mid-way we
     * would not have popped any block (pop is after realloc succeeds). */
    if (kv_seq_grow(s, need) != QW_OK)
        return QW_ERR_NOMEM;
    for (uint64_t i = 0; i < extra; i++)
        s->blocks[s->n_blocks + i] = kv_pop_free(c);
    s->n_blocks = need;
    s->n_tokens = n_tokens;
    *blocks_taken = extra;

    /* update high-water marks */
    uint64_t in_use = c->cfg.n_blocks - c->free_top;
    if (in_use > c->peak_blocks)
        c->peak_blocks = in_use;
    uint64_t total_tokens = 0;
    for (int i = 0; i < c->max_seqs; i++)
        total_tokens += c->seqs[i].n_tokens;
    if (total_tokens > c->peak_tokens)
        c->peak_tokens = total_tokens;
    return QW_OK;
}

/* -------------------------------------------------------------- offset */
qw_err qw_kvcache_offset(const qw_kvcache *c, int seq, uint64_t token_pos,
                         int kv_layer, uint64_t *byte_off_out)
{
    if (c == NULL || byte_off_out == NULL)
        return QW_ERR_NULL;
    if (seq < 0 || seq >= c->max_seqs)
        return QW_ERR_NULL;
    if (kv_layer < 0 || kv_layer >= (int)c->cfg.n_kv_layers)
        return QW_ERR_RANGE;

    const kv_seq *s = &c->seqs[seq];
    if (token_pos >= s->n_tokens)
        return QW_ERR_RANGE;

    /* pure arithmetic — O(1), no search. */
    const uint32_t tpb = c->cfg.tokens_per_block;
    uint64_t bidx  = token_pos / tpb;
    uint64_t posib = token_pos % tpb;
    uint64_t block = s->blocks[bidx];

    uint64_t slot;
    if (kv_umul(block * tpb + posib, c->cfg.n_kv_layers, &slot) != QW_OK ||
        kv_umul(slot + (uint64_t)kv_layer, c->cfg.bytes_per_token_per_layer,
                byte_off_out) != QW_OK)
        return QW_ERR_RANGE;
    return QW_OK;
}

/* ------------------------------------------------------------- release */
/* Return all of seq's blocks to the free list. */
static void kv_seq_free_blocks(qw_kvcache *c, kv_seq *s)
{
    for (uint64_t i = 0; i < s->n_blocks; i++)
        kv_push_free(c, s->blocks[i]);
    s->n_blocks = 0;
    s->n_tokens = 0;
}

qw_err qw_kvcache_release(qw_kvcache *c, int seq)
{
    if (c == NULL)
        return QW_ERR_NULL;
    if (seq < 0 || seq >= c->max_seqs)
        return QW_ERR_NULL;

    kv_seq_free_blocks(c, &c->seqs[seq]);
    return QW_OK;
}

qw_err qw_kvcache_release_tail(qw_kvcache *c, int seq, uint64_t n_tokens)
{
    if (c == NULL)
        return QW_ERR_NULL;
    if (seq < 0 || seq >= c->max_seqs)
        return QW_ERR_NULL;

    kv_seq *s = &c->seqs[seq];
    if (n_tokens == 0 || n_tokens > s->n_tokens)
        return QW_ERR_RANGE;

    uint64_t n_keep = s->n_tokens - n_tokens;
    uint64_t keep_blocks = kv_blocks_for(n_keep, c->cfg.tokens_per_block);

    /* free blocks [keep_blocks, n_blocks) — the ones that held only
     * discarded tokens. A partially-filled final kept block stays. */
    for (uint64_t i = keep_blocks; i < s->n_blocks; i++)
        kv_push_free(c, s->blocks[i]);
    s->n_blocks = keep_blocks;
    s->n_tokens = n_keep;
    return QW_OK;
}

/* --------------------------------------------------------------- stats */
qw_err qw_kvcache_stats(const qw_kvcache *c, qw_kv_stats *out)
{
    if (c == NULL || out == NULL)
        return QW_ERR_NULL;

    uint64_t n_tokens = 0;
    for (int i = 0; i < c->max_seqs; i++)
        n_tokens += c->seqs[i].n_tokens;
    out->n_tokens    = n_tokens;
    out->peak_tokens = c->peak_tokens;
    out->free_blocks = c->free_top;
    out->peak_blocks = c->peak_blocks;
    return QW_OK;
}
