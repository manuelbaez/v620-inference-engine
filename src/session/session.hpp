// Multi-sequence bookkeeping on top of the Engine: which tokens each slot
// holds, recurrent-state snapshots for prompt reuse, slot assignment and
// sampling.
//
// For a new prompt in a slot it:
//   1. restores the deepest prefix of the prompt held by the block store (the
//      host tier shared by every conversation, src/session/block_store.hpp)
//      when that covers more than the slot itself can;
//   2. continues in place when the prompt extends what the slot holds (the
//      usual chat / agent turn: previous prompt + reply + new message);
//   3. else restores the deepest snapshot of that slot that is a prefix of the
//      new prompt (e.g. the reply was re-rendered differently by the template);
//   4. else starts the slot over.
// While prefilling, it snapshots the state at chat message boundaries (inside
// prefill chunks) and at chunk ends, and saves those with their blocks to the
// block store, so later prompts sharing any of those prefixes reuse them.
// KV is position-indexed, so a snapshot at n stays valid only while the slot's
// positions < n are untouched: rewinding a slot to m drops its snapshots past m.
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "core/threadpool.hpp"
#include "engine/engine.hpp"
#include "session/block_store.hpp"
#include "session/sampling.hpp"

namespace qw {

class Session {
public:
    explicit Session(Engine &e);

    int num_slots() const { return e_.num_slots(); }
    // Picks a free slot for `prompt` that can hold prompt + max_new tokens,
    // preferring the one that can reuse the most of it; marks it busy.
    // Returns -1 if none fits right now.
    int acquire(const std::vector<int32_t> &prompt, int64_t max_new);
    void release(int slot);

    // Makes `slot` hold exactly `prompt`, with logits for its last token
    // available to sample_prompt(). Returns the reused prompt tokens.
    int64_t set_prompt(int slot, const std::vector<int32_t> &prompt);

    // An image or a video in a prompt: its patches and where its tokens sit
    // (the prompt holds its pad token there). Its tokens' identity in the
    // prefix cache comes from `hash`; the vision tower only runs for tokens
    // that are prefilled (media/vision_cache.cpp).
    struct Media {
        uint64_t hash;
        bool video;
        int t, h, w;                  // grid in patches; t temporal slices of h*w patches
        const float *patches;         // fp32 [t*h*w][1536], merge-window order
        std::vector<int64_t> starts;  // first token of each slice (h*w/4 tokens each)
    };
    int64_t set_prompt(int slot, const std::vector<int32_t> &prompt, const std::vector<Media> &media);

    // set_prompt in resumable form, so other slots can decode between pieces
    // of a long prefill: begin_prompt restores what the caches hold and returns
    // the reused tokens (media must stay valid until the prompt is in);
    // prefill_some prefills up to max_tokens more and returns true once the
    // whole prompt is in (logits for sample_prompt ready).
    int64_t begin_prompt(int slot, const std::vector<int32_t> &prompt, const std::vector<Media> &media = {});
    // Before begin_prompt: if the prefix cache's hit for `prompt` in `slot` has
    // entries only on disk, loads them into RAM on a background thread. True
    // while that runs (poll until false, then call begin_prompt, which then
    // restores from RAM), so a long disk read never blocks other requests.
    bool prefetch(int slot, const std::vector<int32_t> &prompt, const std::vector<Media> &media = {});
    bool prefill_some(int slot, int64_t max_tokens);
    // prefill_some for several slots (slot, max_tokens) in one engine pass
    // (Engine::prefill_batch) when their pieces fit one prefill chunk and none
    // has media; otherwise one after the other. Returns, per slot, whether its
    // whole prompt is in.
    std::vector<bool> prefill_batch(const std::vector<std::pair<int, int64_t>> &reqs);
    int acquire(const std::vector<int32_t> &prompt, int64_t max_new, const std::vector<Media> &media) {
        return acquire(media.empty() ? prompt : media_keys(prompt, media), max_new);
    }
    // Samples the first token after the slot's prompt is in.
    int32_t sample_prompt(int slot, const SamplingParams &p, float *logprob = nullptr);

    // One batched step; rows of a slot must be consecutive. With a spec, the
    // GPUs also run the sampling kernels (Engine::decode).
    void decode(const std::vector<Engine::Row> &rows, const Engine::SampleSpec *spec = nullptr);
    // Samples from row `row` of the last decode.
    int32_t sample_row(int row, const SamplingParams &p, float *logprob = nullptr);
    void top_logprobs_row(int row, int k, std::vector<int32_t> &ids, std::vector<float> &lps) const;
    void top_logprobs_prompt(int k, std::vector<int32_t> &ids, std::vector<float> &lps) const;  // the last prompt
    void top_logprobs_prompt(int slot, int k, std::vector<int32_t> &ids, std::vector<float> &lps) const;

    // ---- generation with MTP speculative decoding
    // Tokens that end a request in `slot` (EOS, stop ids): verification stops there.
    void set_stop_tokens(int slot, std::vector<int32_t> ids);
    // Saves every slot's newest snapshot to the disk tier and waits for the
    // writes (call at shutdown, with no request running).
    void persist();
    // Blocks until the disk tier's queued writes are on disk (no-op without it).
    void flush_disk() {
        if (store_) store_->flush();
    }
    // Token that starts a chat message (<|im_start|>): prefill snapshots the
    // state before such tokens. -1: no message boundaries.
    void set_boundary_token(int32_t id) { boundary_ = id; }
    BlockStore::Stats cache_stats() const { return store_ ? store_->stats() : BlockStore::Stats{}; }
    // Prompt-level counters: prompt tokens, tokens reused exactly, and tokens
    // after the reuse point in chunks already seen in an earlier prompt at any
    // position (what non-prefix reuse could have saved).
    struct ReuseCounters {
        uint64_t prompt_tokens = 0, reused_tokens = 0, blend_candidate_tokens = 0;
    };
    ReuseCounters reuse_counters() const { return counters_; }
    // GPU-sampled rows whose draw needed the full logits (the candidates did not settle it)
    uint64_t sampling_fallbacks() const { return fallbacks_; }
    struct StepReq {
        int slot;
        int32_t pending;  // sampled, not decoded yet
        int budget;       // tokens the request may still emit (>= 1)
        SamplingParams params;
        bool want_logits = false;  // top_logprobs_row() after the step (the full logits come back)
    };
    struct StepOut {
        std::vector<int32_t> tokens;  // 1..k+1 emitted tokens; tokens[i] sampled from row first_row + i
        std::vector<float> logprobs;
        int first_row = 0;
        bool stopped = false;  // the last token is a stop token
    };
    // One step of every request: decodes its pending token plus up to k MTP
    // drafts in one batch, samples the rows in order while they confirm the
    // drafts, keeps that prefix (rolling back the rest) and drafts the next
    // step. Sampling is exact: every emitted token is drawn from the full
    // model's distribution given the true prefix. k = 0: plain decoding.
    const std::vector<StepOut> &generate(const std::vector<StepReq> &reqs, int k);

    // Single-slot convenience (slot 0), used by tools.
    int64_t set_prompt(const std::vector<int32_t> &prompt) { return set_prompt(0, prompt); }
    void step(int32_t token) { decode({{0, token}}); }
    int32_t sample(const SamplingParams &p, float *logprob = nullptr);

private:
    struct Snap {
        bool valid = false;
        int slot = -1;
        std::vector<int32_t> tokens;  // the state is exactly after these
        std::vector<float> logits;    // logits after tokens.back()
        uint64_t used = 0;
    };
    struct SlotInfo {
        std::vector<int32_t> hist;
        int64_t prompt_end = 0;  // generated tokens start here
        bool busy = false;
        uint64_t used = 0;
        std::vector<int32_t> stop;    // stop tokens of the current request
        std::vector<int32_t> drafts;  // MTP drafts following drafts_for
        int32_t drafts_for = -1;
        float accept = 0.8f;  // running per-draft acceptance (adaptive draft count)
        // plain decoding while drafting does not pay (MTP off): steps left, the
        // next plain stretch's length, and decoded tokens whose MTP rows are missing
        int plain_left = 0, plain_len = 32, mtp_lag = 0;
        std::vector<int32_t> pending;  // prompt being prefilled (begin_prompt .. prefill_some)
        std::vector<Media> pending_media;
        std::vector<float> logits;  // after the prompt's last token (sample_prompt)
        uint64_t epoch = 0;         // bumped when hist is replaced (GPU penalty counts rebuild)
    };
    // prefix cache (prefix_cache.cpp)
    bool restore_from_store(int slot, const std::vector<int32_t> &prompt, size_t slot_reuse, size_t &common);
    // Snapshot points inside the prefill of prompt[from..): message boundaries
    // at least min_gap_ apart, at most Engine::MAX_CAPTURES per prefill chunk.
    // At most max_caps per chunk; append: keep the VRAM snapshots already
    // reserved (for another slot's captures in the same batched pass).
    std::vector<Engine::Capture> plan_captures(const std::vector<int32_t> &prompt, int64_t from, int64_t to,
                                               int max_caps = Engine::MAX_CAPTURES, bool append = false);
    void prefill_range(int slot, const std::vector<int32_t> &prompt, int64_t to);  // [hist.size(), to)
    int64_t begin_keys(int slot, const std::vector<int32_t> &prompt);
    // Returns the VRAM snapshot index; logits: after the slot's last token (default: the engine's)
    int save_snapshot(int slot, const std::vector<float> *logits = nullptr);
    void set_prompt_logits(int slot, const std::vector<float> &l);  // the prompt's logits come from a cache
    void count_reuse(const std::vector<int32_t> &prompt, int64_t reused);
    // vision (vision_cache.cpp): the prompt with its media tokens as content-derived ids, and the
    // embeddings of the media tokens at positions >= from (encoding what the cache lacks)
    std::vector<int32_t> media_keys(const std::vector<int32_t> &prompt, const std::vector<Media> &media) const;
    std::vector<Engine::EmbedSpan> vision_embeds(int64_t from);
    struct VisionEntry {
        std::vector<float> rows;  // [h*w/4][H]
        uint64_t used = 0;
    };
    void drop_snapshots_after(int slot, int64_t n);
    int draft_count(int slot, int k_max) const;
    bool plain_step(int slot, int k_max);  // MTP off for this step?
    int32_t sample_logits(const float *raw, int slot, const SamplingParams &p, float *logprob, float lse_known = NAN,
                          size_t hist_len = SIZE_MAX);
    size_t reusable(int slot, const std::vector<int32_t> &prompt) const;

    Engine &e_;
    std::vector<SlotInfo> slots_;
    std::vector<Snap> snaps_;
    std::vector<int> reserved_;       // VRAM snapshots the running prefill captures into
    std::unique_ptr<BlockStore> store_;
    int32_t boundary_ = -1;
    ReuseCounters counters_;
    const std::vector<Media> *media_ = nullptr;  // of the prompt being set
    std::unordered_map<uint64_t, VisionEntry> vision_cache_;  // (hash, slice) -> embeddings, LRU
    size_t vision_bytes_ = 0, vision_budget_ = 0;
    std::unordered_map<uint64_t, uint64_t> seen_chunks_;  // chunk hash -> last prompt that had it
    int64_t min_gap_ = 1024;          // tokens between snapshots (and the least a saved prefix holds)
    int reserve_ahead_ = 2;           // chunks of saves whose pinned memory is taken ahead; 0: the whole prefill (QW_RESERVE_AHEAD)
    std::vector<int> row_slot_;       // slot of each row of the last decode
    std::vector<StepOut> out_;
    bool single_ = false;  // last op on slot 0 was a single-slot step (sample() reads that row)
    uint64_t clock_ = 0;
    uint64_t fallbacks_ = 0;
    std::mt19937_64 rng_;
    ThreadPool sample_pool_{8};  // samples a step's rows in parallel
};

}  // namespace qw
