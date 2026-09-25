// The 4-GPU engine: TP4 dense layers, EP4 experts, residual stream sharded
// along H (docs/DESIGN.md). One host thread per GPU enqueues its rank's work;
// ranks meet only in the device-side collectives.
//
// The engine holds several sequences ("slots"), each with its own recurrent
// state and KV region. Prefill runs one slot at a time in chunks. Decode runs
// a batch of rows (up to 16): one token per slot for plain decoding, or several
// consecutive tokens of one slot (speculative verification). Every row count
// has its own captured HIP graph.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "comm/comm.hpp"
#include "core/ple.hpp"
#include "core/safetensors.hpp"

namespace qw {

struct Rank;  // per-GPU state (rank.hpp)

struct EngineOptions {
    std::string model_dir = "/mnt/llms/qwen3.8-flash-next-awq";
    std::string ple_dir = "/mnt/llms/qwen3.8-flash-next-ple/ples_int4";
    std::array<int, RANKS> devices{0, 1, 2, 3};
    // KV capacity (tokens) of each sequence slot; multiples of 256.
    std::vector<int> slot_tokens{262144, 65536, 32768, 32768};
    int prefill_chunk = 8192;  // tokens per prefill step (two micro-batches of half)
    bool warmup = true;        // run a throwaway prefill + decodes at load (loads rocBLAS kernels, captures graphs)
    int load_threads = 12;     // host threads per rank for weight conversion
    bool mtp = true;           // load the MTP head (speculative decoding drafts)
};

class Engine {
public:
    static constexpr int MAX_SLOTS = 8;
    static constexpr int SNAPSHOTS = 8;
    static constexpr int MAX_BATCH_ROWS = 16;  // rows of one decode / draft step

    explicit Engine(const EngineOptions &opt);
    ~Engine();

    int num_slots() const { return int(slots_.size()); }
    int64_t slot_capacity(int slot) const;
    int64_t slot_len(int slot) const;
    int max_slot_tokens() const;
    int prefill_chunk() const { return opt_.prefill_chunk; }
    // Empties a slot (its state is zeroed lazily by the next prefill/decode).
    void slot_reset(int slot);

    // A recurrent-state snapshot taken during prefill at `pos` tokens of the
    // slot (inside a chunk, without splitting it) into VRAM snapshot `snap`.
    struct Capture {
        int64_t pos;
        int snap;
    };
    // Appends tokens to a slot (chunked). Returns the logits after the last one;
    // with all_logits, also the logits after every token ([n][VOCAB]).
    // Captures must lie in (len, len + tokens.size()], at most MAX_CAPTURES per
    // prefill chunk.
    const std::vector<float> &prefill(int slot, const std::vector<int32_t> &tokens,
                                      std::vector<float> *all_logits = nullptr,
                                      const std::function<void(int64_t)> &after_chunk = nullptr,
                                      const std::vector<Capture> &captures = {});
    static constexpr int MAX_CAPTURES = 4;

    struct Row {
        int slot;
        int32_t token;
    };
    // One batched step: each row appends its token to its slot. Rows of a slot
    // must be consecutive. Returns logits [rows][VOCAB].
    const std::vector<float> &decode(const std::vector<Row> &rows);

    // Logits of the last prefill (or single-slot step).
    const std::vector<float> &logits() const { return logits_; }
    // Logits of the last decode, [rows][VOCAB].
    const std::vector<float> &logits_rows() const { return dlogits_; }
    // log-sum-exp of each row of the last decode (computed on the GPUs)
    const std::vector<float> &logits_rows_lse() const { return dlse_; }
    void set_logits(const std::vector<float> &l) { logits_ = l; }

    // Debugging: run only the first n decoder layers (then the final mixer and
    // lm_head) in prefill and decode. Needs QW_NOGRAPH (graphs keep the layer
    // count they were captured with).
    void set_debug_layers(int n) { debug_layers_ = n; }

    // ---- MTP speculative decoding
    bool has_mtp() const { return mtp_; }
    // Keeps the first n rows of `slot`'s run in the last decode (1 <= n <= run
    // length) and drops the rest: the slot's length, n-gram context and
    // recurrent state go back to right after row n-1.
    void accept(int slot, int n);
    // Drafts k tokens per request with the MTP head. The MTP layer runs over
    // the slot's newest committed tokens whose MTP rows have not run yet:
    // next.size() of them (1 after a prefill or a plain decode step, the kept
    // count after accept()), with next[i] the token that follows the i-th (the
    // last one is the pending token, not decoded yet). Requests' slots must be
    // distinct. Returns [requests][k].
    struct DraftReq {
        int slot;
        std::vector<int32_t> next;
    };
    const std::vector<std::vector<int32_t>> &draft(const std::vector<DraftReq> &reqs, int k);
    // Brings the MTP layer's KV up to date after decode steps that drafted
    // nothing: runs its rows for positions [p0, p0 + next.size()) of `slot`,
    // next[i] being the token after position p0 + i, from the hidden states
    // decode kept (at most MTP_HISTORY positions back).
    void mtp_catch_up(int slot, int64_t p0, const std::vector<int32_t> &next);
    static constexpr int MTP_HISTORY = 192;  // < gpu::MTP_HIST minus a verification batch

    // Recurrent-state snapshots (GDN state and conv tails, PLE conv tail) in a
    // VRAM pool. KV is position-indexed and needs no copy: restoring a slot to
    // n tokens is valid while its positions < n were not overwritten since the
    // save (the caller enforces that).
    void snapshot_save(int snap, int slot);
    void snapshot_restore(int snap, int slot, int64_t n, const std::vector<int32_t> &tail);

    // ---- host copies (host_tier.hip), for the block store of the prefix
    // cache. Buffers are per rank, in pinned host memory. Copies are enqueued
    // on the ranks' streams; host_copies_wait() waits for them.
    using RankBufs = std::array<uint8_t *, RANKS>;
    // The KV of positions [p0, p0 + n) of a slot (p0 a multiple of IDX_RATIO):
    // per QSA layer (+ MTP) K and V (fp16 [n][256]), raw indexer keys (fp32
    // [n][128]) and the compressed keys of the groups the range completes
    // (fp16 [n/4][128]).
    size_t kv_rank_bytes(int64_t n) const;
    void export_kv(int slot, int64_t p0, int64_t n, const RankBufs &dst);
    void import_kv(int slot, int64_t p0, int64_t n, const RankBufs &src);
    // The recurrent state (GDN state and conv rings, PLE ring, MTP input) of
    // VRAM snapshot `snap`.
    static size_t recurrent_rank_bytes();
    void export_recurrent(int snap, const RankBufs &dst);
    // Makes `slot` hold n tokens: the recurrent state from src, the KV imported
    // with import_kv (tail: the last tokens, for the n-gram context). Waits
    // for the copies.
    void import_recurrent(int slot, const RankBufs &src, int64_t n, const std::vector<int32_t> &tail);
    void host_copies_wait();
    int rank_device(int r) const;  // HIP device of rank r (the order of RankBufs)
    // Identifies the state layout (shapes, ring sizes, MTP layer): a saved
    // state is only loadable by an engine with the same id.
    uint64_t state_layout_id() const;

    // ---- CacheBlend experiment (blend.hip; docs/DESIGN.md). Reusing a chunk
    // cached after another prefix: its KV is copied to the new positions (keys
    // re-rotated, compressed keys rebuilt) and the GDN state is carried across
    // it with the chunk's affine transfer, S_out = M S_in + U.
    // Between begin and end, prefills also accumulate M (from the identity).
    void blend_transfer_begin();
    void blend_transfer_end();
    // Copies the KV of positions [src_p0, src_p0 + n) of slot src to
    // [dst_p0, dst_p0 + n) of slot dst.
    void blend_splice(int dst, int src, int64_t src_p0, int64_t dst_p0, int64_t n);
    // Carries dst's recurrent state across a spliced chunk: S = S_out + M (S -
    // S_in) with snapshots snap_in / snap_out taken around the chunk where M
    // was accumulated (their positions: old_end = the chunk's end there); the
    // conv rings and MTP input come from snap_out. The slot then holds new_len
    // tokens (tail: its last ones). with_transfer = false: S = S_out (no
    // correction, a baseline).
    void blend_compose(int dst, int snap_in, int snap_out, int64_t old_end, int64_t new_len,
                       const std::vector<int32_t> &tail, bool with_transfer = true);

    // Single-sequence convenience API on slot 0 (tools, tests).
    void reset();
    const std::vector<float> &prefill(const std::vector<int32_t> &tokens, std::vector<float> *all_logits = nullptr,
                                      const std::function<void(int64_t)> &after_chunk = nullptr) {
        return prefill(0, tokens, all_logits, after_chunk);
    }
    const std::vector<float> &step(int32_t token);
    int64_t position() const;
    int64_t max_tokens() const { return slot_capacity(0); }

private:
    void warmup();
    void rank_loop(int r);
    void run_prefill_rank(Rank &rk);
    void run_decode_rank(Rank &rk);
    void run_mtp_rank(Rank &rk);
    uint32_t record_batch(Rank &rk, int M, int kind);
    void stage_rows(const std::vector<Row> &rows, const std::vector<int64_t> &pos);
    void dispatch();  // run the current job on every rank and wait

    EngineOptions opt_;
    SafeTensors st_;               // checkpoint, mapped only while loading
    std::vector<uint16_t> embed_;  // token embeddings, bf16 [VOCAB][H] (host-side lookups)
    std::unique_ptr<PleTable> ple_;
    NgramHasher hasher_;
    std::unique_ptr<Comm> comm_, comm2_;  // comm2_: second prefill micro-batch
    std::vector<std::unique_ptr<Rank>> ranks_;
    std::vector<std::thread> threads_;

    struct SlotHost {
        int64_t len = 0;
        std::vector<int32_t> hist;  // last tokens (n-gram context uses two; more kept for accept())
        bool reset = true;          // device state must be zeroed before use
        int run_first = -1;         // the slot's rows in the last decode batch (MTP inputs), or -1
        int run_len = 0;
        bool pend_valid = false;  // the device MTP input store holds the hidden of token len-1
    };
    std::vector<SlotHost> slots_;

    bool mtp_ = false;
    bool blend_transfer_ = false;
    int debug_layers_ = cfg::N_LAYERS;
    // current jobs (host side, read by every rank thread)
    enum class Job { Prefill, Decode, Mtp } job_ = Job::Decode;
    struct PrefillJob {
        int64_t start = 0;
        int T = 0;
        bool all_logits = false;
        int slot = 0;
        bool reset = false;
        bool mtp_pend = false;  // the MTP pass may start one row early, from the slot's MTP input store
        std::vector<Capture> captures;  // in (start, start + T]
        bool transfer = false;          // accumulate the GDN transfer (CacheBlend experiment)
    } pjob_;
    std::vector<float> pemb_;     // [T][H]
    std::vector<uint16_t> pple_;  // [T][H] fp16
    std::vector<float> plogits_;  // [T][VOCAB] when all_logits
    struct DecodeJob {
        int M = 0;
        std::vector<int32_t> i32;  // [5][MAX_ROWS]: row_slot, row_first, run_start, run_len, run_slot
        std::vector<int64_t> pos;  // [MAX_ROWS]
        std::vector<int> reset_slots;
        bool save = false;  // keep per-row GDN states (for accept())
        // MTP job: per row, the hidden source: >= 0 a row of the batch hidden, < 0 the MTP
        // input store of slot -1 - src, or with src_hist[i] >= 0 that slot's decoded-row
        // hidden at position src_hist[i]
        std::vector<int> src;
        std::vector<int64_t> src_hist;
    } djob_;
    std::vector<float> demb_;                                              // [M][H]
    std::vector<uint16_t> dple_;                                           // [M][H] fp16
    std::vector<float> dlogits_;                                           // [M][VOCAB]
    std::vector<float> dlse_;                                              // [M]
    std::vector<float> dlse_parts_ = std::vector<float>(RANKS * 16 * 2);   // [rank][MAX_ROWS][max, sumexp]
    std::vector<float> damax_parts_ = std::vector<float>(RANKS * 16 * 2);  // [rank][MAX_ROWS][max, index bits]
    std::vector<std::vector<int32_t>> drafts_;
    std::vector<float> logits_;

    // watchdog: per-rank phase of the current job (0 idle, 1 enqueueing, 2 waiting for the GPU)
    std::array<std::atomic<int>, RANKS> phase_{};
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    uint64_t gen_ = 0;
    int done_ = 0;
    bool stop_ = false;
    std::string error_;
};

}  // namespace qw
