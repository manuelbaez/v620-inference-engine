/* qw/vocab.h — Qwen byte-level-BPE vocabulary (part 1 of the tokenizer).
 * Pure C17, libc only, no HIP.
 *
 * Loads the GGUF tokenizer metadata out of an opened gguf file: the token
 * table, scores, token types, special-token ids, the pre-tokenizer name and
 * the raw BPE merges blob. This module only EXTRACTS and INDEXES the
 * vocabulary; the merge/encode step is a separate later module, so the
 * merges array is stored as an opaque byte blob and never parsed here.
 *
 * Storage: one contiguous byte pool `toks` holds every token string back to
 * back (no NUL padding, lengths kept separately — byte-level BPE tokens may
 * contain arbitrary bytes including NUL). A parallel `idx` array holds
 * (offset, len, score, type) per token. A separately built open-addressing
 * hash (fnv1a over the exact byte range) backs qw_vocab_token_to_id; both
 * lookups are O(1) and never copy.
 *
 * Only tokenizer.ggml.tokens is mandatory. Every other key is optional:
 * missing keys are either reported by name in the error (tokenizer.ggml.model
 * must be present and equal to "gpt2") or defaulted explicitly (empty pre,
 * absent ids, absent merges), never silently guessed.
 */
#ifndef QW_VOCAB_H
#define QW_VOCAB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gguf_file; /* opaque (qw/gguf.h) */

/* Token type codes, per the GGUF tokenizer.ggml.token_type convention. */
typedef enum qw_token_type {
    QW_TOK_NORMAL      = 1,
    QW_TOK_UNKNOWN     = 2,
    QW_TOK_CONTROL     = 3,
    QW_TOK_USER_DEFINED = 4,
    QW_TOK_BYTE        = 6,
} qw_token_type;

/* The named special-token ids. QW_TOK_NONE (-1) marks "key absent"; an
 * id is always >= 0 when present. */
typedef struct qw_special_ids {
    int32_t bos;   /* tokenizer.ggml.bos_token_id */
    int32_t eos;   /* tokenizer.ggml.eos_token_id */
    int32_t pad;   /* tokenizer.ggml.padding_token_id */
    int32_t unk;   /* tokenizer.ggml.unk_token_id */
} qw_special_ids;

/* One token's row in the index array. `off`/`len` delimit the token bytes
 * in the pool; `score` is the model-logit score; `type` is a qw_token_type. */
typedef struct qw_vocab_tok {
    uint32_t off;
    uint32_t len;
    float    score;
    uint8_t  type;
} qw_vocab_tok;

/* The loaded vocabulary. All pointers alias the struct's own allocations
 * (toks, idx, pre, merges, tok_hash) and are valid until qw_vocab_free. */
typedef struct qw_vocab {
    uint32_t n_vocab;      /* number of tokens */
    uint8_t *toks;         /* contiguous token byte pool (n_vocab strings) */
    qw_vocab_tok *idx;     /* n_vocab index rows, parallel to toks */
    float *scores;         /* n_vocab scores (also in idx[].score) */

    char *pre;             /* tokenizer.ggml.pre ("" if key absent) */
    uint8_t *merges;       /* raw merges blob (tokenizer.ggml.merges) */
    uint64_t merges_len;   /* 0 if key absent */

    bool add_bos;          /* tokenizer.ggml.add_bos_token (false if absent) */
    qw_special_ids sp;     /* special-token ids, -1 when key absent */

    bool has_special;      /* true if any sp member is present */

    uint32_t *tok_hash;    /* open-addressing hash over the exact token
                              bytes: slot -> token id; 0xFFFFFFFF empty */
    uint32_t tok_hash_cap; /* power of two, >= 2 * n_vocab */

    uint32_t type_count[7]; /* per qw_token_type count (index 0 unused) */
} qw_vocab;

/* Load the Qwen tokenizer metadata from f. f must be a non-NULL open gguf
 * file; out must be a non-NULL, zero-initialized-on-entry struct (its
 * fields are reset internally). Returns QW_OK on success (out is fully
 * populated and must later be freed with qw_vocab_free) or a QW_ERR_* code
 * on failure (out is left zero-initialized, safe to free, and the offending
 * key is named in the log).
 *
 * Failure modes:
 *   - tokenizer.ggml.tokens missing, not an array-of-string, or empty
 *     => QW_ERR_FORMAT (key named).
 *   - tokenizer.ggml.model missing or not the string "gpt2"
 *     => QW_ERR_FORMAT (key named / value quoted).
 *   - tokenizer.ggml.scores present but not an array-of-float, or of a
 *     different length than tokens => QW_ERR_FORMAT (key named).
 *   - duplicate token bytes (any length) => QW_ERR_FORMAT (duplicate id
 *     named).
 *   - OOM anywhere => QW_ERR_ALLOC.
 *
 * Non-fatal (warned, not errors): n_vocab != 248320, and a
 * tokenizer.ggml.pre that is not a qwen-family value. */
qw_err qw_vocab_from_gguf(const struct gguf_file *f, qw_vocab *out);

/* O(1) id -> token bytes. *ptr points INTO the pool (no copy); *len is the
 * token's byte length (may be 0, may include NUL bytes). id must be <
 * n_vocab or QW_ERR_RANGE is returned. */
qw_err qw_vocab_id_to_token(const qw_vocab *v, uint32_t id,
                            const char **ptr, uint32_t *len);

/* O(1) token bytes -> id via the load-time hash (hot path). *id receives
 * the token id, or QW_ERR_NOT_FOUND-class miss: returns QW_ERR_RANGE for a
 * token not present. s/len must delimit a valid byte range (len may be 0). */
qw_err qw_vocab_token_to_id(const qw_vocab *v, const char *s, uint32_t len,
                            uint32_t *id);

/* Is id one of the named special tokens? *out set to true iff id equals a
 * present (non -1) special id. id must be < n_vocab or QW_ERR_RANGE. */
qw_err qw_vocab_is_special(const qw_vocab *v, uint32_t id, bool *out);

/* Copy the named special-token ids into *out. */
qw_err qw_vocab_special_ids(const qw_vocab *v, qw_special_ids *out);

/* Fill byte_to_id[256] with the byte -> token-id table for the byte-level
 * vocab: for each token of type BYTE whose string is exactly one byte,
 * map that byte to the token id. Bytes with no such token are mapped to
 * QW_TOK_NONE (defined below, 0xFFFFFFFF). byte_to_id must point at 256
 * uint32_t entries; all 256 are written, none left uninitialized. */
#define QW_TOK_NONE 0xFFFFFFFFu
qw_err qw_vocab_byte_lookup(const qw_vocab *v, uint32_t *byte_to_id);

/* Release every allocation owned by v and zero the struct. Safe on a
 * never-loaded (all-NULL) vocab. */
void qw_vocab_free(qw_vocab *v);

/* Human summary to stdout: n_vocab, per-token-type counts, bos/eos ids,
 * and whether the merges blob was present. */
void qw_vocab_report(const qw_vocab *v);

#ifdef __cplusplus
}
#endif

#endif /* QW_VOCAB_H */
