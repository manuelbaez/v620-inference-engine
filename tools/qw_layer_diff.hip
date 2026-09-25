// Localizes a divergence between the prefill and the decode paths: for each
// model depth n (the first n decoder layers, then the final mixer and
// lm_head), runs the tokens once through prefill (all logits) and once token
// by token through decode, and prints the mean / max |logprob difference| of
// the actual next tokens. Run with QW_NOGRAPH=1.
// Usage: QW_NOGRAPH=1 qw_layer_diff --tokens 1,2,3 [--layers 1,2,4,8]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> parse_ids(const std::string &s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) v.push_back(int32_t(std::stol(tok)));
    return v;
}

static void log_softmax(const float *l, std::vector<double> &out) {
    const int V = cfg::VOCAB;
    const float mx = *std::max_element(l, l + V);
    double z = 0;
    for (int i = 0; i < V; ++i) z += std::exp(double(l[i] - mx));
    const double lse = mx + std::log(z);
    out.resize(size_t(V));
    for (int i = 0; i < V; ++i) out[size_t(i)] = l[i] - lse;
}

int main(int argc, char **argv) {
    std::vector<int32_t> toks;
    std::vector<int32_t> layers;
    EngineOptions opt;
    opt.slot_tokens = {32768};
    opt.warmup = false;
    opt.mtp = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--tokens")
            toks = parse_ids(next());
        else if (a == "--layers")
            layers = parse_ids(next());
    }
    if (layers.empty())
        for (int n = 1; n <= cfg::N_LAYERS; ++n) layers.push_back(n);
    try {
        QW_CHECK(toks.size() >= 2, "need at least 2 tokens");
        QW_CHECK(std::getenv("QW_NOGRAPH"), "run with QW_NOGRAPH=1");
        Engine e(opt);
        std::vector<double> a, b;
        for (int n : layers) {
            e.set_debug_layers(n);
            e.slot_reset(0);
            std::vector<float> all;
            e.prefill(0, toks, &all);
            e.slot_reset(0);
            double sum = 0, mx = 0, top_max = 0;
            for (size_t t = 0; t + 1 < toks.size(); ++t) {
                const auto &d = e.decode({{0, toks[t]}});
                log_softmax(all.data() + t * cfg::VOCAB, a);
                log_softmax(d.data(), b);
                const double diff = std::fabs(a[size_t(toks[t + 1])] - b[size_t(toks[t + 1])]);
                if (std::getenv("QW_PER_POS")) std::printf("  pos %zu: |dlogprob| %.4f\n", t, diff);
                sum += diff;
                mx = std::max(mx, diff);
                for (int v = 0; v < cfg::VOCAB; ++v)
                    top_max = std::max(top_max, std::fabs(a[size_t(v)] - b[size_t(v)]));
            }
            std::printf("layers %2d: mean |dlogprob| %.4f, max %.4f, max over vocab %.4f\n", n,
                        sum / double(toks.size() - 1), mx, top_max);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
