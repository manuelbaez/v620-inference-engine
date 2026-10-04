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

#include <rocblas/rocblas.h>

#include "comm/comm.hpp"
#include "core/ple.hpp"
#include "core/safetensors.hpp"
#include "kernels/sampling_types.hpp"

namespace qw {

struct Rank;  // per-GPU state (rank.hpp)
class SpillPool;  // KV beyond a slot's VRAM tokens (spill.hpp)

#define QW_HAVE_SPILL 1  // EngineOptions::slot_spill exists (benchmarks that also build on older trees test for it)
#define QW_HAVE_SPILL_CACHE 1  // Engine::spill_cache_stats exists

struct EngineOptions {
    std::string model_dir = "/mnt/llms/qwen3.8-flash-next-awq";
    std::string ple_dir = "/mnt/llms/qwen3.8-flash-next-ple/ples_int4";
    std::array<int, RANKS> devices{0, 1, 2, 3};
    // KV tokens of each sequence slot held in VRAM; multiples of 256.
    std::vector<int> slot_tokens{262144, 65536, 32768, 32768};
    // Tokens a slot holds beyond its VRAM, in pinned host memory the kernels read directly (multiples of 256; empty or
    // 0: none). A slot's capacity is slot_tokens + slot_spill; the compressed keys stay in VRAM for all of it.
    std::vector<int> slot_spill{};
    // The capacity every slot should have (multiple of 256): each slot spills whatever its VRAM tokens do not cover.
    // 0: off (slots hold what their VRAM holds). Ignored for the slots slot_spill names. QW_SLOT_MAX_TOKENS sets it
    // when this is 0; QW_SPILL=0 turns every form of spill off.
    int slot_max_tokens = 0;
    // Prefill into a slot's spilled part reads that part from VRAM: before each QSA layer the spilled rows the chunk can
    // attend to are copied into a staging buffer (one per card, as large as the largest spill) on a copy stream, while
    // the layers before it run, and the chunk's new rows go back to host memory after. QW_SPILL_STAGE=0: off (the
    // attention reads host memory directly, 2.2-2.6x slower prefill there).
    bool spill_stage = true;
    // Decode-time VRAM cache of spilled K/V rows, shared by all slots (kernels/types.hpp SpillCache): MB per card, 0 off
    // (QW_SPILL_CACHE_MB). Allocated only when some slot spills.
    int spill_cache_mb = 256;
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
    int64_t slot_capacity(int slot) const;      // tokens the slot can hold: VRAM plus spill
    int64_t slot_vram_tokens(int slot) const;   // ... of which in VRAM
    int64_t slot_len(int slot) const;
    int max_slot_tokens() const;
    int prefill_chunk() const { return opt_.prefill_chunk; }
    // QW_PREFILL_PIPELINE (default on) at construction; tests and benchmarks may change it between prefills
    void set_prefill_pipeline(bool on) { pipeline_ = on; }
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
    // prefill chunk. Vision tokens (negative ids, cfg::real_token) take their
    // embeddings from `embeds`, which must cover every one of them.
    struct EmbedSpan {
        int64_t pos;         // slot position of rows[0]
        int64_t n;           // rows
        const float *rows;   // fp32 [n][H]
    };
    const std::vector<float> &prefill(int slot, const std::vector<int32_t> &tokens,
                                      std::vector<float> *all_logits = nullptr,
                                      const std::function<void(int64_t)> &after_chunk = nullptr,
                                      const std::vector<Capture> &captures = {},
                                      const std::vector<EmbedSpan> &embeds = {},
                                      const std::vector<int32_t> *rope3 = nullptr);  // [tokens][3], experiment
    static constexpr int MAX_CAPTURES = 4;

    // Batched prefill: appends tokens to several slots in one pass (one
    // chunk), so the per-pass cost of the layers' collectives and launches is
    // paid once. Slots must be distinct, the tokens together at most
    // prefill_chunk(), each segment's captures at most MAX_CAPTURES. No
    // vision tokens. Returns the logits after each segment's last token.
    struct Segment {
        int slot;
        std::vector<int32_t> tokens;
        std::vector<Capture> captures;
    };
    static constexpr int MAX_SEGMENTS = MAX_SLOTS;
    const std::vector<std::vector<float>> &prefill_batch(const std::vector<Segment> &segs);

    struct Row {
        int slot;
        int32_t token;
    };
    // One batched step: each row appends its token to its slot. Rows of a slot
    // must be consecutive. Returns logits [rows][VOCAB].
    const std::vector<float> &decode(const std::vector<Row> &rows);

    // ---- sampling on the GPUs (gpu_sample.hip, kernels/sampling.hpp)
    // decode() with a spec also runs the sampling kernels on each row
    // (spec->rows[i] for row i; slot, first and token are filled in here) and
    // brings back only their results, sample_out(); the full logits only when
    // spec->logits (else logits_rows() is not valid for this step).
    struct SampleSpec {
        std::vector<gpu::SampleRow> rows;
        bool logits = false;
    };
    const std::vector<float> &decode(const std::vector<Row> &rows, const SampleSpec *spec);
    // Results of the last decode with a spec: row i, rank r at [i * RANKS + r].
    const std::vector<gpu::SampleOut> &sample_out() const { return sout_; }
    bool logits_rows_valid() const { return dlogits_valid_; }
    // The full logits of row `row` of the last decode, read from the GPUs (valid until the next job).
    void row_logits(int row, float *out);
    // The penalties' token counts of a slot: its tokens `hist` (from prompt_end
    // on generated). Only the new tokens are sent while the epoch and
    // prompt_end stay the same and hist only grew; applied by the next decode.
    void penalty_sync(int slot, uint64_t epoch, int64_t prompt_end, const std::vector<int32_t> &hist);

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
    // export_kv skips a rank whose buffer is null, and a rank that shares its buffer with the first rank of its KV
    // group (cfg::kv_primary): the ranks of a group hold identical KV, so a store keeps one copy per group.
    // import_kv reads every rank's buffer (replicas may point at the same one).
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
    // QW_SPILL_LOCALITY experiment: the counters per QSA layer ([QSA_LAYERS_MAX][gpu::LOC_STATS]), then zeroed.
    std::vector<unsigned long long> spill_locality();
    // QW_SPILL_CACHE_STATS: spilled rows decode found in the spill cache, and the ones it read from host memory, since
    // the last call (rank 0).
    std::array<unsigned long long, 2> spill_cache_stats();
    int rank_device(int r) const;  // HIP device of rank r (the order of RankBufs)
    // Identifies the state layout (shapes, ring sizes, MTP layer): a saved
    // state is only loadable by an engine with the same id.
    uint64_t state_layout_id() const;

    // Why the engine can no longer be used, or "" while it works: a rank's job
    // failed or a collective timed out. Both stay set for good (the next job
    // fails with the same message), so only a restart recovers. Safe to call
    // from any thread.
    std::string failure() const;

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

    // ---- vision (vision_job.hip): every card holds the vision tower and
    // encodes whole slices (an image, or one temporal slice of a video); a
    // batch of slices is spread over the cards, longest first.
    bool has_vision() const { return vision_; }
    struct VisionSlice {
        const float *patches;  // fp32 [h*w][1536], merge-window order
        int h, w;              // grid in patches
        float *out;            // fp32 [h*w/4][2560]
    };
    void encode_vision(const std::vector<VisionSlice> &slices);

    // ---- GPTQ calibration (calibrate.hip): while on, prefills accumulate the
    // Hessian X^T X of the input of every int8-able matrix of layers [lo, hi)
    // (N_LAYERS: the final mixer and lm_head; the PLE projection with its
    // layer). calib_dump writes them, r<rank>/<name>.h, and frees them.
    // Run with QW_NO_SPLIT=1 (one stream per rank).
    void calib_begin(int lo, int hi);
    void calib_dump(const std::string &dir);

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
    void run_draft_rank(Rank &rk);
    void run_vision_rank(Rank &rk);
    bool calibrating(int layer) const { return layer >= calib_lo_ && layer < calib_hi_; }
    void calib_acc(Rank &rk, const std::string &name, const uint16_t *X, int ldx, int rows, int K, rocblas_handle blas);
    int calib_lo_ = -1, calib_hi_ = -1;
    uint32_t record_batch(Rank &rk, int M, int kind);
    void stage_rows(const std::vector<Row> &rows, const std::vector<int64_t> &pos, bool embeddings = true);
    void dispatch();  // run the current job on every rank and wait

    // QW_MOE_STATS: how evenly a prefill chunk's routed-expert work spreads over the ranks. Expert parallelism
    // makes every rank wait for the busiest one at the next collective, so the sum over layers of the busiest
    // rank's tiles against the mean's is the factor the MoE phase (and the waiting) grows by.
    // moe_counts_[r]: pinned [micro-batch][layer][EXP_L] tokens routed to each of rank r's experts, filled by the
    // rank's prefill job; collect_moe_balance() sums them after the chunk, log_moe_balance() reports and resets.
    std::array<int32_t *, RANKS> moe_counts_{};
    struct MoeBalance {
        int chunks = 0;
        double max_tiles = 0, mean_tiles = 0, max_pairs = 0, mean_pairs = 0;  // summed over layers and micro-batches
        std::vector<double> layer_max, layer_mean;                            // tiles per layer, summed over chunks
    } moe_bal_;
    bool moe_stats_ = false;
    void collect_moe_balance();
    void log_moe_balance();

    EngineOptions opt_;
    SafeTensors st_;               // checkpoint, mapped only while loading
    // token embeddings, bf16 [VOCAB][H]: host-side lookups, and read by the GPUs
    // for draft tokens (pinned, mapped into every device's address space)
    uint16_t *embed_ = nullptr;
    std::unique_ptr<PleTable> ple_;
    std::unique_ptr<SpillPool> spill_;
    std::vector<uint32_t> cache_epoch_;  // per slot, mirrored on every rank (SpillCache::epoch)
    // A slot's spilled rows are about to change other than through decode: its cached rows become stale.
    void spill_cache_bump(int slot);
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
    enum class Job { Prefill, Decode, Mtp, Draft, Vision } job_ = Job::Decode;
    bool vision_ = false;
    std::vector<std::vector<VisionSlice>> vjob_;  // per rank
    // One prefill chunk: segments of distinct slots, their rows concatenated.
    struct PrefillSeg {
        int slot = 0;
        int64_t start = 0;  // slot position of the segment's first row
        int T = 0;
        int row0 = 0;  // first row in the chunk
        bool reset = false;
        bool mtp_pend = false;          // the MTP pass may start one row early, from the slot's MTP input store
        std::vector<Capture> captures;  // in (start, start + T]
    };
    struct PrefillJob {
        std::vector<PrefillSeg> segs;
        int T = 0;  // rows over all segments
        bool all_logits = false;         // (one segment)
        bool transfer = false;           // accumulate the GDN transfer (CacheBlend experiment; one segment)
        const int32_t *rope3 = nullptr;  // (t, h, w) rotary positions [T][3] of the chunk (multimodal RoPE; one segment)
        const float *emb = nullptr;      // [T][H] the chunk's embeddings and fp16 PLE rows (host; read at the job's start)
        const uint16_t *ple = nullptr;
    } pjob_;
    std::vector<float> pemb_;     // [T][H]
    std::vector<uint16_t> pple_;  // [T][H] fp16
    // QW_PREFILL_PIPELINE (default on): the next chunk's embeddings and PLE rows are prepared into these on a helper thread
    // while the GPUs run the current chunk (they depend on the tokens only), instead of between the two
    std::vector<float> pemb2_;
    std::vector<uint16_t> pple2_;
    bool pipeline_ = true;  // QW_PREFILL_PIPELINE=0 off
    std::vector<float> plogits_;  // [T][VOCAB] when all_logits
    std::vector<std::vector<float>> seg_logits_;  // [segments][VOCAB]: after each segment's last row
    // prefill host side: embeddings and PLE rows of `chunk` into emb / ple at row0. `len` is the slot's length at the
    // chunk's start and `hist` its last tokens (the n-gram context): explicit, so that the next chunk's can be
    // prepared before this one is committed. Reads the tokens, the tables and the hasher only.
    void prefill_inputs(int64_t len, const std::vector<int32_t> &hist, const std::vector<int32_t> &chunk, int row0,
                        const std::vector<EmbedSpan> &embeds, float *emb, uint16_t *ple) const;
    void prefill_inputs(int slot, const std::vector<int32_t> &chunk, int row0, const std::vector<EmbedSpan> &embeds);
    void prefill_commit(int slot, const std::vector<int32_t> &chunk);  // the slot's host state after its rows
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
        bool sample = false, copy_logits = true;  // GPU sampling of the rows; full logits back
        std::vector<gpu::SampleRow> srows;
    } djob_;
    std::vector<gpu::SampleOut> sout_;  // [rows][RANKS]
    // a draft job's steps (chained on the GPUs without host syncs in between)
    struct DraftStep {
        int M = 0;
        std::vector<int32_t> i32;  // row tables as in DecodeJob
        std::vector<int64_t> pos;
        std::vector<int> src;       // hidden source per row, as DecodeJob::src (steps > 0: rows of the batch hidden)
        std::vector<int32_t> last;  // per request: its row whose argmax is its draft
    };
    std::vector<DraftStep> dsteps_;
    std::vector<int32_t> dtok_;  // [steps][MAX_ROWS] drafted tokens (from rank 0)
    bool dlogits_valid_ = false, dlogits_on_gpu_ = false;
    // penalty counts per slot: what the GPUs hold, and updates for the next decode
    struct PenState {
        uint64_t epoch = ~0ull;
        int64_t prompt_end = -1;
        size_t len = 0;
        bool reset = false;          // pending: zero the counts first
        std::vector<int32_t> tok;    // pending tokens
        std::vector<uint8_t> gen;    // pending: generated?
    };
    std::vector<PenState> pen_;
    void apply_penalty_updates(Rank &rk);  // in the decode job, on the rank's stream
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
    mutable std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    uint64_t gen_ = 0;
    int done_ = 0;
    bool stop_ = false;
    std::string error_;
};

}  // namespace qw
