// Runs the 4-GPU engine on token ids (same interface as qw_ref).
//   qw_gpu --tokens 1,2,3 [--gen N] [--logprobs FILE] [--max-tokens N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(int32_t(std::stol(tok)));
    return v;
}

int main(int argc, char **argv) {
    EngineOptions opt;
    std::string ids_s, lp_file;
    int gen = 8;
    bool tbt = false;
    long prefill_n = -1;  // prefill this many tokens, then feed the rest through decode
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) fail("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--model") opt.model_dir = next();
        else if (a == "--ple") opt.ple_dir = next();
        else if (a == "--tokens") ids_s = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--logprobs") lp_file = next();
        else if (a == "--max-tokens") opt.slot_tokens = {std::stoi(next())};
        else if (a == "--chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--token-by-token") tbt = true;
        else if (a == "--no-warmup") opt.warmup = false;
        else if (a == "--prefill-n") prefill_n = std::stol(next());
        else fail("unknown argument " + a);
    }
    try {
        auto prompt = parse_ids(ids_s);
        QW_CHECK(!prompt.empty(), "--tokens is required");
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        auto t0 = now();
        Engine eng(opt);
        auto t1 = now();
        log("loaded in %.1f s", secs(t0, t1));

        FILE *f = lp_file.empty() ? nullptr : std::fopen(lp_file.c_str(), "w");
        std::vector<float> lp(cfg::VOCAB);
        auto logsoftmax = [&](const std::vector<float> &l) {
            float mx = *std::max_element(l.begin(), l.end());
            double s = 0;
            for (float v : l) s += std::exp(double(v - mx));
            float lse = mx + float(std::log(s));
            for (size_t i = 0; i < l.size(); ++i) lp[i] = l[i] - lse;
        };
        int next = 0;
        auto write_lp = [&](size_t t, const float *logits) {
            std::vector<float> l(logits, logits + cfg::VOCAB);
            logsoftmax(l);
            int top = int(std::max_element(lp.begin(), lp.end()) - lp.begin());
            std::fprintf(f, "%zu %d %.5f %d %.5f\n", t + 1, prompt[t + 1], lp[size_t(prompt[t + 1])], top,
                         lp[size_t(top)]);
        };
        if (prefill_n > 0) {
            std::vector<int32_t> head(prompt.begin(), prompt.begin() + prefill_n);
            std::vector<float> all;
            const auto &l0 = eng.prefill(head, f ? &all : nullptr);
            if (f)
                for (long t = 0; t + 1 < prefill_n; ++t) write_lp(size_t(t), all.data() + t * cfg::VOCAB);
            if (f) write_lp(size_t(prefill_n - 1), l0.data());
            for (size_t t = size_t(prefill_n); t < prompt.size(); ++t) {
                const auto &logits = eng.step(prompt[t]);
                next = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
                if (f && t + 1 < prompt.size()) write_lp(t, logits.data());
            }
        } else if (tbt) {
            for (size_t t = 0; t < prompt.size(); ++t) {
                const auto &logits = eng.step(prompt[t]);
                next = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
                if (f && t + 1 < prompt.size()) write_lp(t, logits.data());
            }
        } else {
            std::vector<float> all;
            const auto &logits = eng.prefill(prompt, f ? &all : nullptr);
            next = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (f)
                for (size_t t = 0; t + 1 < prompt.size(); ++t) write_lp(t, all.data() + t * cfg::VOCAB);
        }
        if (f) std::fclose(f);
        auto t2 = now();
        log("prompt: %zu tokens in %.2f s (%.1f tok/s, %s)", prompt.size(), secs(t1, t2),
            prompt.size() / secs(t1, t2), tbt ? "token-by-token" : "prefill");
        std::printf("generated:");
        auto t3 = now();
        for (int i = 0; i < gen; ++i) {
            std::printf(" %d", next);
            std::fflush(stdout);
            if (i + 1 == gen) break;
            const auto &logits = eng.step(next);
            next = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (std::getenv("QW_TOP")) {  // debug: top-3 logprobs per generated step
                std::vector<float> l(logits.begin(), logits.end());
                logsoftmax(l);
                std::vector<int> idx(l.size());
                for (size_t j = 0; j < idx.size(); ++j) idx[j] = int(j);
                std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(), [&](int a, int b) { return lp[size_t(a)] > lp[size_t(b)]; });
                std::fprintf(stderr, "  step %d top: %d %.3f | %d %.3f | %d %.3f\n", i + 1, idx[0], lp[size_t(idx[0])],
                             idx[1], lp[size_t(idx[1])], idx[2], lp[size_t(idx[2])]);
            }
        }
        auto t4 = now();
        std::printf("\n");
        if (gen > 1) log("decode: %.2f tok/s", (gen - 1) / secs(t3, t4));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
