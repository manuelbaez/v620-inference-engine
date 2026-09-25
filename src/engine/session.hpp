// Multi-sequence bookkeeping on top of the Engine: which tokens each slot
// holds, recurrent-state snapshots for prompt reuse, slot assignment and
// sampling.
//
// For a new prompt in a slot it:
//   1. continues in place when the prompt extends what the slot holds (the
//      usual chat / agent turn: previous prompt + reply + new message);
//   2. else restores the deepest snapshot of that slot that is a prefix of the
//      new prompt (e.g. the reply was re-rendered differently by the template);
//   3. else starts the slot over.
// KV is position-indexed, so a snapshot at n stays valid only while the slot's
// positions < n are untouched: rewinding a slot to m drops its snapshots past m.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "engine/engine.hpp"

namespace qw {

struct SamplingParams {
    float temperature = 1.f;         // 0 = greedy
    float top_p = 1.f;
    int top_k = 0;                   // 0 = off
    float min_p = 0.f;
    float presence_penalty = 0.f;    // on generated tokens (OpenAI semantics)
    float frequency_penalty = 0.f;
    float repetition_penalty = 1.f;  // on prompt + generated tokens (HF semantics)
    uint64_t seed = 0;               // 0 = nondeterministic
};

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
    // Samples the first token after set_prompt (before any other prefill).
    int32_t sample_prompt(int slot, const SamplingParams &p, float *logprob = nullptr);

    // One batched step; rows of a slot must be consecutive.
    void decode(const std::vector<Engine::Row> &rows);
    // Samples from row `row` of the last decode.
    int32_t sample_row(int row, const SamplingParams &p, float *logprob = nullptr);
    void top_logprobs_row(int row, int k, std::vector<int32_t> &ids, std::vector<float> &lps) const;
    void top_logprobs_prompt(int k, std::vector<int32_t> &ids, std::vector<float> &lps) const;

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
    };
    void save_snapshot(int slot);
    void drop_snapshots_after(int slot, int64_t n);
    int32_t sample_logits(const float *raw, int slot, const SamplingParams &p, float *logprob, float lse_known = NAN);
    size_t reusable(int slot, const std::vector<int32_t> &prompt) const;

    Engine &e_;
    std::vector<SlotInfo> slots_;
    std::vector<Snap> snaps_;
    std::vector<int> row_slot_;  // slot of each row of the last decode
    bool single_ = false;        // last op on slot 0 was a single-slot step (sample() reads that row)
    uint64_t clock_ = 0;
    std::mt19937_64 rng_;
};

}  // namespace qw
