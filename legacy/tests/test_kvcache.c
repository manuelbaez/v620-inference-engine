/* tests/test_kvcache.c — no-framework asserts for the paged KV-cache
 * block manager.
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -O2 -Iinclude \
 *        src/core/types.c src/core/config.c src/core/kvcache.c tests/test_kvcache.c \
 *        -lm -o /tmp/opencode/tk && /tmp/opencode/tk
 *
 * main() returns the count of failures (0 = all pass).
 */
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qw/kvcache.h"
#include "qw/types.h"

static int g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { printf("  ok   %s\n", #cond); } \
    else { printf("  FAIL %s (line %d)\n", #cond, __LINE__); g_fail++; } \
} while (0)

/* The per-token KV footprint across all 12 QSA layers, in bytes. Same
 * figure config.c asserts (kv_per_token): 12 * 2048 = 24576. */
#define KV_PER_TOKEN 24576ULL

/* Deterministic PRNG so randomization tests reproduce their failures. */
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void)
{
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 7;
    g_seed ^= g_seed << 17;
    return g_seed;
}

/* ------------------------------------------------------------------ t1 */
static void test_bytes_per_token(void)
{
    printf("test 1: default cfg bytes == 24576/token across 12 layers\n");
    qw_kvcfg cfg;
    CHECK(qw_kvcfg_default(1ULL << 40, &cfg) == QW_OK);
    CHECK(cfg.n_kv_layers == 12);
    CHECK(cfg.n_kv_heads == 2);
    CHECK(cfg.head_dim == 256);
    CHECK(cfg.tokens_per_block == 256);
    CHECK(cfg.bytes_per_token_per_layer == 2048);
    CHECK((uint64_t)cfg.n_kv_layers * cfg.bytes_per_token_per_layer
          == KV_PER_TOKEN);
}

/* ------------------------------------------------------------------ t2 */
/* Cache sized for exactly N tokens: reserve N works, reserve N+1 fails
 * with QW_ERR_NOMEM and leaves the sequence uncorrupted. */
static void test_exact_capacity(void)
{
    printf("test 2: exact-N cache: N reserves, N+1 -> QW_ERR_NOMEM\n");
    const uint64_t tpb = 256;
    const uint64_t nblocks = 5;
    const uint64_t N = nblocks * tpb; /* 1280 tokens */

    qw_kvcfg cfg;
    CHECK(qw_kvcfg_default(N * KV_PER_TOKEN, &cfg) == QW_OK);
    CHECK(cfg.n_blocks == nblocks);

    qw_kvcache *c = NULL;
    CHECK(qw_kvcache_init(&c, &cfg, 1) == QW_OK);
    CHECK(c != NULL);

    uint64_t taken = 99;
    CHECK(qw_kvcache_reserve(c, 0, N, &taken) == QW_OK);
    CHECK(taken == nblocks);

    /* N+1 tokens needs 6 blocks; only 5 exist. */
    taken = 99;
    CHECK(qw_kvcache_reserve(c, 0, N + 1, &taken) == QW_ERR_NOMEM);
    CHECK(taken == 0);

    /* no corruption: token N-1 still addresses at its hand-computed offset,
     * token N does not. */
    uint64_t off;
    CHECK(qw_kvcache_offset(c, 0, N - 1, 11, &off) == QW_OK);
    /* token 1279 = block slot 4 (page_table[4] = 0, the 5th pop), pos 255,
     * layer 11: ((0*256+255)*12+11)*2048 = 3071*2048 = 6,289,408. */
    CHECK(off == 6289408ULL);
    CHECK(qw_kvcache_offset(c, 0, N, 0, &off) == QW_ERR_RANGE);

    /* free list untouched by the failed reserve */
    qw_kv_stats st;
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == 0);
    CHECK(st.n_tokens == N);
    CHECK(st.peak_blocks == nblocks);

    /* re-reserve the same N: no new blocks, still OK */
    CHECK(qw_kvcache_reserve(c, 0, N, &taken) == QW_OK);
    CHECK(taken == 0);

    /* releasing frees every block */
    CHECK(qw_kvcache_release(c, 0) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks);
    CHECK(st.n_tokens == 0);
    qw_kvcache_destroy(c);
}

/* ------------------------------------------------------------------ t3 */
/* Offset injectivity: over 5000 random reservations across 4 sequences,
 * mark every 24576-byte slot touched with the unique tag of the
 * (seq, pos, layer) that owns it. No slot may ever be tagged by two
 * different (seq, pos, layer) triples.
 *
 * The tag space is small by design (16 bits seq-pos + 4 bits layer), so a
 * deliberate collision in that space is impossible to hit accidentally:
 * two distinct triples produce distinct tags, and a slot holding a
 * different tag is proof that two triples mapped to the same bytes. */
static void test_offset_injective(void)
{
    printf("test 3: offset injective over 5000 random reservations\n");
    const uint64_t nblocks = 32;
    const int max_seqs = 4;
    const int n_resv = 5000;
    const uint32_t tpb = 256;

    qw_kvcfg cfg;
    CHECK(qw_kvcfg_default(nblocks * tpb * KV_PER_TOKEN, &cfg) == QW_OK);

    qw_kvcache *c = NULL;
    CHECK(qw_kvcache_init(&c, &cfg, max_seqs) == QW_OK);
    if (c == NULL) return;

    /* tag space: one uint64_t per (token-slot, layer) — each such slot is
     * bytes_per_token_per_layer (2048) bytes, so the slot index is
     * off / 2048, and there are nblocks * tpb * 12 of them. */
    uint64_t nslots = nblocks * (uint64_t)tpb * 12;
    uint64_t *tags = calloc(nslots, sizeof(uint64_t));
    CHECK(tags != NULL);
    if (tags == NULL) { qw_kvcache_destroy(c); return; }

    uint64_t collisions = 0, marked = 0;
    static const uint64_t SEQ_STRIDE = 1ULL << 40; /* per-seq headroom */
    static const uint64_t SENTINEL   = UINT64_MAX;
    /* pre-fill tags with SENTINEL so a 0 tag value is distinguishable */
    for (uint64_t i = 0; i < nslots; i++)
        tags[i] = SENTINEL;

    uint64_t per_seq[max_seqs];
    memset(per_seq, 0, sizeof(per_seq));

    for (int r = 0; r < n_resv; r++) {
        int seq = (int)(rnd() % (uint64_t)max_seqs);
        uint64_t inc = (rnd() % 128ULL) + 1;
        uint64_t want = per_seq[seq] + inc;
        uint64_t taken;
        if (qw_kvcache_reserve(c, seq, want, &taken) != QW_OK) {
            /* block-exhausted: the reserve must have left seq untouched */
            continue;
        }
        per_seq[seq] = want;

        /* tag only the newly reserved tokens [want-inc, want). Tag is a
         * 64-bit bijection over (seq, pos, layer):
         *   tag = seq*SEQ_STRIDE + pos*12 + layer  (+1 to avoid 0)
         * distinct triples -> distinct tags; a slot holding a different
         * non-SENTINEL tag is proof two triples mapped to the same bytes. */
        uint64_t lo = want - inc;
        for (uint64_t pos = lo; pos < want; pos++) {
            for (int layer = 0; layer < 12; layer++) {
                uint64_t off;
                if (qw_kvcache_offset(c, seq, pos, layer, &off) != QW_OK) {
                    CHECK(0);
                    goto done;
                }
                uint64_t slot = off / 2048ULL; /* per (token, layer) slot */
                if (slot >= nslots) { CHECK(0); goto done; }
                uint64_t tag = ((uint64_t)seq * SEQ_STRIDE
                                + pos * 12ULL
                                + (uint64_t)layer) + 1ULL;
                if (tags[slot] != SENTINEL && tags[slot] != tag)
                    collisions++;
                if (tags[slot] == SENTINEL)
                    marked++;
                tags[slot] = tag;
            }
        }
    }
    CHECK(collisions == 0);
    CHECK(marked > 0);

    /* cross-sequence spot check: seq 0 and seq 1 both reserved token 0;
     * their blocks are disjoint (each sequence owns its own blocks), so
     * the two byte offsets must differ and each cover a 24576-byte range
     * that does not overlap the other's. */
    uint64_t a, b;
    CHECK(qw_kvcache_offset(c, 0, 0, 0, &a) == QW_OK);
    CHECK(qw_kvcache_offset(c, 1, 0, 0, &b) == QW_OK);
    CHECK(a != b);
    CHECK((a / KV_PER_TOKEN) != (b / KV_PER_TOKEN));
done:
    free(tags);
    qw_kvcache_destroy(c);
}

/* ------------------------------------------------------------------ t4 */
/* Release returns blocks to the free list and stats return to baseline. */
static void test_release_baseline(void)
{
    printf("test 4: release -> free list and stats back to baseline\n");
    const uint64_t nblocks = 8;
    qw_kvcfg cfg;
    CHECK(qw_kvcfg_default(nblocks * 256 * KV_PER_TOKEN, &cfg) == QW_OK);

    qw_kvcache *c = NULL;
    CHECK(qw_kvcache_init(&c, &cfg, 4) == QW_OK);
    if (c == NULL) return;

    qw_kv_stats st;
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks && st.n_tokens == 0 &&
          st.peak_blocks == 0 && st.peak_tokens == 0);

    uint64_t taken;
    CHECK(qw_kvcache_reserve(c, 2, 600, &taken) == QW_OK); /* 3 blocks */
    CHECK(taken == 3);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks - 3);
    CHECK(st.n_tokens == 600);
    CHECK(st.peak_blocks == 3);
    CHECK(st.peak_tokens == 600);

    /* offset valid while reserved */
    uint64_t off;
    CHECK(qw_kvcache_offset(c, 2, 599, 5, &off) == QW_OK);

    /* release: free list and token count back to baseline; peaks stay
     * (high-water marks, not reset). */
    CHECK(qw_kvcache_release(c, 2) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks);
    CHECK(st.n_tokens == 0);
    CHECK(st.peak_blocks == 3);
    CHECK(st.peak_tokens == 600);

    /* the sequence is reusable after release */
    CHECK(qw_kvcache_reserve(c, 2, 256, &taken) == QW_OK);
    CHECK(taken == 1);
    CHECK(qw_kvcache_release(c, 2) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks);

    qw_kvcache_destroy(c);
}

/* ------------------------------------------------------------------ t5 */
/* release_tail frees exactly the blocks whose boundary the rollback
 * crosses; a partially-filled final block is kept. */
static void test_release_tail(void)
{
    printf("test 5: release_tail frees exactly the boundary-crossed blocks\n");
    const uint64_t nblocks = 16;
    const uint32_t tpb = 256;
    qw_kvcfg cfg;
    CHECK(qw_kvcfg_default(nblocks * tpb * KV_PER_TOKEN, &cfg) == QW_OK);

    qw_kvcache *c = NULL;
    CHECK(qw_kvcache_init(&c, &cfg, 1) == QW_OK);
    if (c == NULL) return;

    uint64_t taken;
    CHECK(qw_kvcache_reserve(c, 0, 700, &taken) == QW_OK); /* 3 blocks */
    CHECK(taken == 3);
    qw_kv_stats st;
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks - 3);

    /* (a) 700->600: ceil(600/256)=3, still 3 blocks -> 0 freed. */
    CHECK(qw_kvcache_release_tail(c, 0, 100) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks - 3);
    CHECK(st.n_tokens == 600);

    /* (b) 600->500: ceil(500/256)=2 -> crosses out of block 3 -> 1 freed. */
    CHECK(qw_kvcache_release_tail(c, 0, 100) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks - 2);
    CHECK(st.n_tokens == 500);

    /* (c) 500->256: ceil(256/256)=1 -> 1 more freed (block now exactly
     *     full in block 0; block 1 was empty of kept tokens). */
    CHECK(qw_kvcache_release_tail(c, 0, 244) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks - 1);
    CHECK(st.n_tokens == 256);

    /* (d) 256->255: still needs 1 block (partial) -> 0 freed. */
    CHECK(qw_kvcache_release_tail(c, 0, 1) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks - 1);
    CHECK(st.n_tokens == 255);

    /* (e) 255->0: frees the last block. */
    CHECK(qw_kvcache_release_tail(c, 0, 255) == QW_OK);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks);
    CHECK(st.n_tokens == 0);

    /* error cases: nothing to roll back / rolling back more than reserved */
    CHECK(qw_kvcache_release_tail(c, 0, 1) == QW_ERR_RANGE);
    CHECK(qw_kvcache_reserve(c, 0, 10, &taken) == QW_OK);
    CHECK(qw_kvcache_release_tail(c, 0, 11) == QW_ERR_RANGE);

    /* the kept partial block still serves its tokens after a tail release:
     * seq is empty (0 tokens). Reserve 10 (positions 0..9), roll back 1 ->
     * 9 kept (positions 0..8): token 8 valid, token 9 not. */
    CHECK(qw_kvcache_reserve(c, 0, 10, &taken) == QW_OK);
    CHECK(qw_kvcache_release_tail(c, 0, 1) == QW_OK);
    uint64_t off;
    CHECK(qw_kvcache_offset(c, 0, 8, 11, &off) == QW_OK);
    CHECK(qw_kvcache_offset(c, 0, 9, 0, &off) == QW_ERR_RANGE);

    qw_kvcache_destroy(c);
}

/* ------------------------------------------------------------------ t6 */
/* Interleaved random growth of 8 sequences, then full release: zero
 * leaked blocks, and the whole pool is re-allocatable. */
static void test_interleaved_no_leak(void)
{
    printf("test 6: interleaved grow of 8 seqs + full release -> no leak\n");
    const uint64_t nblocks = 64;
    const int max_seqs = 8;
    qw_kvcfg cfg;
    CHECK(qw_kvcfg_default(nblocks * 256 * KV_PER_TOKEN, &cfg) == QW_OK);

    qw_kvcache *c = NULL;
    CHECK(qw_kvcache_init(&c, &cfg, max_seqs) == QW_OK);
    if (c == NULL) return;

    uint64_t len[max_seqs];
    memset(len, 0, sizeof(len));

    for (int step = 0; step < 300; step++) {
        int seq = (int)(rnd() % (uint64_t)max_seqs);
        uint64_t inc = (rnd() % 512ULL) + 1;
        len[seq] += inc;
        uint64_t taken;
        qw_err e = qw_kvcache_reserve(c, seq, len[seq], &taken);
        if (e == QW_ERR_NOMEM) {
            len[seq] -= inc; /* not overcommitted: back off to prior len */
        } else if (e != QW_OK) {
            CHECK(0);
            break;
        }
    }

    qw_kv_stats st;
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);

    /* full release of every sequence */
    int rel_ok = 1;
    for (int s = 0; s < max_seqs; s++)
        if (qw_kvcache_release(c, s) != QW_OK)
            rel_ok = 0;
    CHECK(rel_ok);

    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks); /* zero leaked blocks */
    CHECK(st.n_tokens == 0);

    /* whole pool re-allocatable: one full block per sequence */
    int reok = 1;
    for (int s = 0; s < max_seqs; s++) {
        uint64_t taken;
        if (qw_kvcache_reserve(c, s, 256, &taken) != QW_OK || taken != 1)
            reok = 0;
    }
    CHECK(reok);
    CHECK(qw_kvcache_stats(c, &st) == QW_OK);
    CHECK(st.free_blocks == nblocks - max_seqs);

    qw_kvcache_destroy(c);
}

/* ------------------------------------------------------------------ t7 */
/* The offset formula, hand-computed.
 *
 * Reserve 9 blocks in a 16-block cache: pops are LIFO from free_list
 * [0..15], so the page table is [15,14,13,12,11,10,9,8,7]. Block slot 8
 * therefore holds block_id 7.
 *
 *   token_pos = 8*256 + 100 = 2148 -> block_id 7, pos_in_block 100
 *   byte_off  = ((7*256 + 100) * 12 + 5) * 2048
 *             = (1892 * 12 + 5) * 2048
 *             = 22709 * 2048
 *             = 46,508,032
 *
 * Second value, same block (block_id 7), a different pos_in_block:
 *   token_pos = 8*256 + 95 = 2143 -> block_id 7, pos_in_block 95
 *   byte_off  = ((7*256 + 95) * 12 + 11) * 2048
 *             = (1887 * 12 + 11) * 2048
 *             = 22655 * 2048
 *             = 46,397,440
 *
 * Third value: token 0 lives in block slot 0 -> page_table[0] = 15, layer 0:
 *   byte_off  = ((15*256 + 0) * 12 + 0) * 2048
 *             = 3840 * 2048
 *             = 94,371,840
 */
static void test_hand_computed_offset(void)
{
    printf("test 7: offset formula matches hand-computed values\n");
    const uint64_t nblocks = 16;
    const uint32_t tpb = 256;
    qw_kvcfg cfg;
    CHECK(qw_kvcfg_default(nblocks * tpb * KV_PER_TOKEN, &cfg) == QW_OK);

    qw_kvcache *c = NULL;
    CHECK(qw_kvcache_init(&c, &cfg, 1) == QW_OK);
    if (c == NULL) return;

    uint64_t taken;
    CHECK(qw_kvcache_reserve(c, 0, 9 * tpb, &taken) == QW_OK);
    CHECK(taken == 9);

    uint64_t off;
    CHECK(qw_kvcache_offset(c, 0, 2148, 5, &off) == QW_OK);
    CHECK(off == 46508032ULL);

    CHECK(qw_kvcache_offset(c, 0, 2143, 11, &off) == QW_OK);
    CHECK(off == 46397440ULL);

    /* token 0 lives in block slot 0 -> page_table[0] = 15 (first LIFO pop). */
    CHECK(qw_kvcache_offset(c, 0, 0, 0, &off) == QW_OK);
    CHECK(off == 94371840ULL);

    /* bounds checks on the formula inputs */
    CHECK(qw_kvcache_offset(c, 0, 9 * tpb, 0, &off) == QW_ERR_RANGE); /* not reserved */
    CHECK(qw_kvcache_offset(c, 0, 0, 12, &off) == QW_ERR_RANGE);      /* layer oob */
    CHECK(qw_kvcache_offset(c, 1, 0, 0, &off) == QW_ERR_NULL);        /* seq oob */

    qw_kvcache_destroy(c);
}

int main(void)
{
    test_bytes_per_token();
    test_exact_capacity();
    test_offset_injective();
    test_release_baseline();
    test_release_tail();
    test_interleaved_no_leak();
    test_hand_computed_offset();
    if (g_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fail);
    return g_fail;
}
