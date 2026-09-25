/* qw/bpe.h — Qwen byte-level-BPE tokenizer, part 2: the encode/decode side.
 * Pure C17, libc only, no HIP.
 *
 * Consumes a loaded qw_vocab (part 1, qw/vocab.h) and the BPE merge table.
 * The merge table is NOT read from the vocab's raw merges blob: the GGUF
 * string path for that key is being fixed elsewhere, so to stay decoupled the
 * caller supplies the merges as an array of C strings already in rank order.
 * The gguf-blob -> merge-lines conversion therefore happens OUTSIDE this
 * module (split the blob on newlines; each line is "A B").
 *
 * Pipeline (qw_bpe_encode):
 *   1. byte-level map: every input byte b becomes the GPT-2 byte-to-unicode
 *      char U+0000+b (qw_byte_level_encode); bytes >= 128 become two UTF-8
 *      bytes so they are visible printable characters before splitting.
 *   2. pre-tokenizer: a hand-written state machine (no PCRE) applies the
 *      Qwen2 rule set — letter/digit boundaries, whitespace runs, and the
 *      "keep preceding punctuation attached" contraction rule — producing
 *      non-overlapping spans.
 *   3. per span: the whole span if it is one vocab token, else its byte
 *      tokens, then merges applied by RANK (lowest rank first; equal ranks
 *      tie-broken leftmost). Merges never cross a span boundary (the
 *      pre-tokenizer's chunks are the merge units).
 *   4. the span id sequences are concatenated directly — whitespace spans are
 *      already separate chunks (Qwen2 keeps whitespace as its own tokens), so
 *      no extra join token is inserted — giving the final id sequence.
 *
 * Special tokens can NEVER be produced by this encoder. The merge step is
 * structurally incapable of emitting a special id: qw_bpe_init rejects any
 * merge line whose result is a special id (QW_ERR_FORMAT), so the pair->rank
 * index a merge consults can only ever return a non-special result. The
 * whole-span fast path and the byte fallback likewise filter special ids.
 * The invariant is enforced by construction, not by string-matching the
 * special tokens.
 *
 * All outputs are byte-exact: for input whose bytes are all representable
 * (every one of the 256 values, via the total byte-level map),
 * qw_bpe_encode followed by qw_bpe_decode reproduces the original bytes.
 *
 * Truncation is explicit: whenever an output buffer is too small the call
 * returns QW_ERR_RANGE and *n_out receives the number of elements that WOULD
 * have been produced; nothing is silently cut.
 */
#ifndef QW_BPE_H
#define QW_BPE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qw/types.h"
#include "qw/vocab.h"
#include "qw/alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One span of the (byte-level-mapped) text: [start, end) into the buffer
 * produced by qw_byte_level_encode. */
typedef struct qw_span {
    size_t start;
    size_t end;
} qw_span;

/* The loaded BPE state. Owns its merge-table index (pair -> rank); free with
 * qw_bpe_free. The index is the STRUCTURAL enforcement of the "special
 * tokens are never mergeable" rule: a pair whose merge produces a special id
 * is simply not present in it.
 *
 * The open-addressing table stores, per slot, the (left id, right id) pair,
 * the merge result id, and the rank+1 (0 = empty). Storing the key and the
 * result id (not just the rank) makes the lookup exact under collisions and
 * lets the merge step emit the merged token id directly — and because init
 * only ever stores a non-special result id, a lookup can never return one. */
typedef struct qw_bpe {
    qw_arena *arena; /* arena the index came from; NOT owned. NULL if the
                       index was malloc'd (no arena was supplied). */
    uint64_t *pair;  /* slot -> (left id << 32) | right id */
    uint32_t *rid;   /* slot -> merge RESULT token id */
    uint32_t *rank1; /* slot -> rank + 1 (0 = empty) */
    uint32_t cap;    /* power of two, >= 2 * n_merges */
    uint32_t n_merges;
} qw_bpe;

/* Load the merge table. merge_lines[i] is "A B" (two space-separated token
 * strings) with i = the merge rank (0 = applied first). The caller supplies
 * the lines in rank order; the gguf-merges-to-lines conversion happens
 * outside this module. scratch, if non-NULL, must hold
 * >= 16 + 8 * (2 * n_merges + 16) bytes for the index (a safe estimate).
 *
 * A merge line whose result would be a special token is rejected with
 * QW_ERR_FORMAT (the merge table must not reference special ids).
 *
 * Failure modes: NULL bpe/vocab or negative n_merges => QW_ERR_NULL;
 * n_merges > 0 with NULL merge_lines => QW_ERR_NULL; scratch/arena too small
 * or OOM => QW_ERR_ALLOC; malformed line (not exactly "A B") =>
 * QW_ERR_FORMAT; a token on the line not in the vocab => QW_ERR_FORMAT;
 * merge result not in the vocab => QW_ERR_FORMAT; merge result is a special
 * id => QW_ERR_FORMAT; duplicate pair => QW_ERR_FORMAT. */
qw_err qw_bpe_init(qw_bpe *bpe, const qw_vocab *v, const char **merge_lines,
                   int n_merges, qw_arena *scratch);

/* Release the merge-table index. NULL-safe; the arena is NOT destroyed. */
void qw_bpe_free(qw_bpe *bpe);

/* Byte-level (GPT-2) mapping: in[i] -> the UTF-8 encoding of code point
 * U+0000 + in[i]. Bytes < 128 are identity; bytes >= 128 become two UTF-8
 * bytes (lead 0xC0..0xDF, trail 0x80..0xBF) so they are visible printable
 * characters in the mapped text. Reversible via qw_byte_level_decode.
 *
 * out must hold >= n * 2 bytes (the worst case). *n_out receives the number
 * of mapped bytes written; on a QW_ERR_RANGE (out_cap too small) *n_out is
 * set to the number needed (n * 2). QW_ERR_NULL on NULL args. */
qw_err qw_byte_level_encode(const char *in, size_t n, char *out,
                            size_t out_cap, size_t *n_out);

/* Inverse of qw_byte_level_encode: a byte-level-mapped UTF-8 text back to
 * its raw bytes. Each 1-byte char U+0000..U+007F gives its byte; each
 * 2-byte char U+0080..U+00FF gives (lead - 0xC0) * 64 + (trail - 0x80).
 *
 * s/len delimit the mapped text (len authoritative; embedded NULs are fine).
 * out must hold >= len bytes (a raw byte always maps to >= 1 mapped byte).
 * *n_out receives the number of raw bytes written; on a QW_ERR_RANGE
 * (out_cap too small) *n_out is set to the number needed. QW_ERR_NULL on
 * NULL args, QW_ERR_RANGE if out_cap < len or on a malformed mapped char. */
qw_err qw_byte_level_decode(const char *s, size_t len, char *out,
                            size_t out_cap, size_t *n_out);

/* Qwen2-style pre-tokenizer, hand-written state machine (no PCRE), applied
 * to the BYTE-LEVEL-MAPPED text (the output of qw_byte_level_encode).
 *
 * The rule set, in priority order:
 *   1. runs of whitespace are one span each;
 *   2. every run of word characters (ASCII letter/digit) is one span, and a
 *      following run of punctuation (contraction characters) is kept attached
 *      to it — "keep preceding punctuation";
 *   3. every other character (punctuation not kept attached, and any non
 *      ASCII mapped char) is its own span.
 *
 * out_spans receives up to max_spans spans in order; *n_spans_out the count
 * produced. If more spans are produced than fit, returns QW_ERR_RANGE with
 * *n_spans_out = the total needed (nothing truncated). Empty text => zero
 * spans. QW_ERR_NULL on NULL args. */
qw_err qw_bpe_pre_tokenizer(const char *text, size_t len, qw_span *out_spans,
                            int max_spans, int *n_spans_out);

/* Full encode pipeline (see file header for the stages).
 *
 * text/len delimit the raw input (len authoritative; embedded NULs are fine
 * and are encoded as byte token 0). The result id sequence is written to
 * out_ids. Complexity: O(k^2) in the number of pre-tokenized spans k in the
 * worst case (each of the <= total-bytes merge passes scans the current
 * token list once with an O(1) rank lookup), but the constant is small and
 * the list only shrinks, so it is not pathologically slow.
 *
 * Truncation: if out_cap < the number of ids produced, returns QW_ERR_RANGE
 * with *n_out = the count needed; on success *n_out = the count written.
 * QW_ERR_NULL on NULL args (bpe/vocab/text/n_out). A missing space token or
 * a byte with no byte token returns QW_ERR_RANGE.
 *
 * Never emits a special-token id (see file header, structural rule). */
qw_err qw_bpe_encode(const qw_bpe *bpe, const qw_vocab *v, const char *text,
                     size_t len, uint32_t *out_ids, size_t out_cap,
                     size_t *n_out);

/* Inverse of qw_bpe_encode: ids -> original raw bytes, including the
 * byte-level inverse mapping (qw_byte_level_decode).
 *
 * Each id's token bytes are the MAPPED bytes (UTF-8 for bytes >= 128); the
 * whole concatenation is inverse-mapped to raw bytes. A special-token id in
 * the sequence is decoded as the bytes of that token — decode is a pure
 * id -> bytes expansion and does not enforce the encoder's no-special rule.
 *
 * out must hold >= the total raw-byte count. If out_cap is too small,
 * returns QW_ERR_RANGE with *n_out = the raw bytes needed; on success
 * *n_out = the raw bytes written. An id >= v->n_vocab returns QW_ERR_RANGE.
 * QW_ERR_NULL on NULL args. */
qw_err qw_bpe_decode(const qw_bpe *bpe, const qw_vocab *v, const uint32_t *ids,
                     size_t n, char *out, size_t out_cap, size_t *n_out);

#ifdef __cplusplus
}
#endif

#endif /* QW_BPE_H */
