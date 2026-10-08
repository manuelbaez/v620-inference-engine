// CPU fp32 reference implementation of docs/MODEL.md.
//
// It is deliberately simple: one sequence, weights read straight from the
// mmapped checkpoint, every intermediate in fp32. It exists to be the oracle
// for the GPU kernels and to check the spec against vLLM, not to be fast.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/ple.hpp"
#include "core/safetensors.hpp"
#include "core/threadpool.hpp"

namespace qw::ref {

// Per-sequence state: everything a continuation needs.
struct State {
    int64_t n_tokens = 0;         // positions consumed so far
    std::vector<int32_t> tokens;  // full token history (n-gram context)

    // GDN, per GDN layer
    std::vector<float> gdn_conv;  // [N_GDN][GDN_CONV-1][GDN_QKV] last pre-conv rows
    std::vector<float> gdn_S;     // [N_GDN][V_HEADS][DIM k][DIM v]

    // QSA, per QSA layer: growing caches
    struct Qsa {
        std::vector<float> k, v;   // [T][KV_HEADS*HEAD_DIM]
        std::vector<float> raw_k;  // [T][IDX_DIM]
        std::vector<float> ck;     // [T/4][IDX_DIM] compressed, normed, roped
    };
    std::vector<Qsa> qsa;

    // PLE conv input history: [PLE_CONV-1)*DILATION][HCH]
    std::vector<float> ple_conv;

    void reset();
};

class Model {
public:
    Model(const std::string &model_dir, const std::string &ple_dir, int threads = 0);

    // Runs `tokens` as the next positions of `st`. Writes logits for every
    // position into `logits` ([n][VOCAB]) when all_logits, otherwise only for
    // the last one ([VOCAB]).
    void forward(State &st, const std::vector<int32_t> &tokens, std::vector<float> &logits, bool all_logits = false);

    // Optional: record the multi-stream state after each layer (for tests).
    std::vector<std::vector<float>> *dump_layers = nullptr;

    ThreadPool &pool() { return pool_; }
    const SafeTensors &weights() const { return st_; }

private:
    SafeTensors st_;
    std::unique_ptr<PleTable> ple_;
    NgramHasher hasher_;
    ThreadPool pool_;

    std::string lp(int layer) const;  // "model.language_model.layers.<L>."

    void linear(const TensorView &w, const float *x, int T, int ldx, float *y, int ldy);
    void raw_linear(const uint16_t *W, int out, int in, const float *x, int T, int ldx, float *y, int ldy);
    void expert_linear(const TensorView &packed, const TensorView &scale, const float *x, int T, int ldx, float *y,
                       int ldy);

    void hc_mix(const std::string &prefix, bool with_inject, const float *X, int T, float *block_in, float *inj);
    void combine(float *X, const float *block_out, const float *inj, int T);
    void ple(State &st, float *X, const std::vector<int32_t> &tokens);
    void gdn(State &st, int layer, const float *x, int T, float *out);
    void qsa(State &st, int layer, const float *x, int T, float *out);
    void moe(int layer, const float *x, int T, float *out);
};

}  // namespace qw::ref
