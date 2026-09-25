// Runs the CPU reference model on token ids.
//
//   qw_ref --model DIR --ple DIR --tokens 1,2,3 [--gen N] [--logprobs FILE] [--threads N]
//
// Prints the greedy continuation. With --logprobs, writes one line per prompt
// position: "<pos> <next_token> <logprob_of_next> <top1_id> <top1_logprob>",
// for comparison against vLLM's prompt_logprobs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "ref/ref_model.hpp"

using namespace qw;

static std::vector<int32_t> parse_ids(const std::string &s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(int32_t(std::stol(tok)));
    return v;
}

static void log_softmax(const float *l, int n, std::vector<float> &out) {
    float mx = *std::max_element(l, l + n);
    double s = 0;
    for (int i = 0; i < n; ++i) s += std::exp(double(l[i] - mx));
    float lse = mx + float(std::log(s));
    out.resize(size_t(n));
    for (int i = 0; i < n; ++i) out[size_t(i)] = l[i] - lse;
}

int main(int argc, char **argv) {
    std::string model = "/mnt/llms/qwen3.8-flash-next-awq", ple = "/mnt/llms/qwen3.8-flash-next-ple/ples_int4";
    std::string ids_s, lp_file;
    int gen = 8, threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) fail("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--model") model = next();
        else if (a == "--ple") ple = next();
        else if (a == "--tokens") ids_s = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--logprobs") lp_file = next();
        else if (a == "--threads") threads = std::stoi(next());
        else fail("unknown argument " + a);
    }
    try {
        auto prompt = parse_ids(ids_s);
        QW_CHECK(!prompt.empty(), "--tokens is required");
        auto t0 = std::chrono::steady_clock::now();
        ref::Model m(model, ple, threads);
        ref::State st;
        st.reset();
        auto t1 = std::chrono::steady_clock::now();
        log("loaded in %.1f s", std::chrono::duration<double>(t1 - t0).count());

        std::vector<float> logits, lp;
        bool all = !lp_file.empty();
        m.forward(st, prompt, logits, all);
        auto t2 = std::chrono::steady_clock::now();
        log("prefill %zu tokens in %.2f s", prompt.size(), std::chrono::duration<double>(t2 - t1).count());
        const float *last = logits.data() + (all ? (prompt.size() - 1) * cfg::VOCAB : 0);

        if (all) {
            FILE *f = std::fopen(lp_file.c_str(), "w");
            QW_CHECK(f, "cannot write " + lp_file);
            for (size_t t = 0; t + 1 < prompt.size(); ++t) {
                log_softmax(logits.data() + t * cfg::VOCAB, cfg::VOCAB, lp);
                int top = int(std::max_element(lp.begin(), lp.end()) - lp.begin());
                std::fprintf(f, "%zu %d %.5f %d %.5f\n", t + 1, prompt[t + 1], lp[size_t(prompt[t + 1])], top,
                             lp[size_t(top)]);
            }
            std::fclose(f);
        }

        std::printf("generated:");
        std::vector<float> step;
        int next = int(std::max_element(last, last + cfg::VOCAB) - last);
        for (int i = 0; i < gen; ++i) {
            std::printf(" %d", next);
            std::fflush(stdout);
            if (i + 1 == gen) break;
            auto s0 = std::chrono::steady_clock::now();
            m.forward(st, {next}, step, false);
            auto s1 = std::chrono::steady_clock::now();
            log("decode step %.2f s", std::chrono::duration<double>(s1 - s0).count());
            next = int(std::max_element(step.begin(), step.end()) - step.begin());
        }
        std::printf("\n");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
