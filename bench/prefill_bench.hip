// Prefill speed by prompt size: each size prefilled R times (default 3) into a slot, after the load-time warm-up;
// best time reported.
//   prefill_bench [--ctx N] [--reps R] [--tokens-file F] sizes...        e.g. prefill_bench 300 2000 8192
// The tokens are a fixed arithmetic sequence of ids, or with --tokens-file the first ones of F (ids separated by
// commas or spaces, e.g. tokenized text: the experts' routing, and so the MoE balance, depends on the tokens).
// QW_PREFILL_CHUNK sets the chunk (default 8192), QW_PLE_DIR the PLE table (default int4; production: ples_fp8). With --ctx N the slot already holds N tokens (prefilled first,
// not timed), so the timed tokens attend to a long context, as the fresh part of an agent turn does.
// The GPUs' energy (kJ and J per token, from the cards' power sensors) is printed with each size.
// QW_PROFILE adds the engine's split of each prefill (host inputs, jobs, after_chunk); QW_MOE_STATS the balance of
// the routed experts over the ranks.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "power_meter.hpp"
#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> soup(int n, int seed) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) v[size_t(i)] = 100 + ((i + seed) * 7919) % 30000;
    return v;
}

int main(int argc, char **argv) {
    try {
        int ctx = 0, reps = 3;
        std::vector<int> sizes;
        std::vector<int32_t> text;  // --tokens-file
        for (int a = 1; a < argc; ++a) {
            const std::string s = argv[a];
            if (s == "--ctx" && a + 1 < argc) ctx = std::atoi(argv[++a]);
            else if (s == "--reps" && a + 1 < argc) reps = std::atoi(argv[++a]);
            else if (s == "--tokens-file" && a + 1 < argc) {
                std::ifstream f(argv[++a]);
                std::string tok;
                for (int c; f && (c = f.get()) != EOF;) {
                    if (c == ',' || c == ' ' || c == '\n') {
                        if (!tok.empty()) text.push_back(int32_t(std::atoi(tok.c_str())));
                        tok.clear();
                    } else {
                        tok += char(c);
                    }
                }
                if (!tok.empty()) text.push_back(int32_t(std::atoi(tok.c_str())));
                QW_CHECK(!text.empty(), "prefill_bench: no token ids in the --tokens-file");
            } else sizes.push_back(std::atoi(argv[a]));
        }
        // n tokens: the file's from `from` on (wrapping), or the sequence
        const auto tokens = [&](int n, int from, int seed) {
            if (text.empty()) return soup(n, seed);
            std::vector<int32_t> v(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) v[size_t(i)] = text[size_t(from + i) % text.size()];
            return v;
        };
        const int biggest = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
        EngineOptions opt;
        opt.slot_tokens = {(std::max(32768, ctx + biggest + 64) + 255) / 256 * 256};  // multiples of 256
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;  // production runs ples_fp8
        if (const char *c = std::getenv("QW_PREFILL_CHUNK")) opt.prefill_chunk = std::atoi(c);
        Engine e(opt);
        const std::vector<int32_t> context = tokens(ctx, 0, 1);
        PowerMeter meter;
        for (int n : sizes) {
            const std::vector<int32_t> toks = tokens(n, ctx, 7);
            double best = 1e9, best_kj = 0;
            for (int rep = 0; rep < reps; ++rep) {
                e.slot_reset(0);
                if (ctx) e.prefill(0, context);
                meter.start();
                const auto t0 = std::chrono::steady_clock::now();
                e.prefill(0, toks);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const PowerMeter::Result en = meter.stop();
                if (secs < best) best = secs, best_kj = en.kilojoules;
            }
            std::printf("prefill %6d tokens after %6d of context, chunk %d: %.3f s, %.0f tok/s, %.2f kJ (%.0f mJ/token)\n", n, ctx,
                        e.prefill_chunk(), best, n / best, best_kj, best_kj * 1e6 / n);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
