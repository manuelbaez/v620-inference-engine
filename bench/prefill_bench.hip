// Prefill speed by prompt size: each size prefilled R times (default 3) into a slot, after the load-time warm-up;
// best time reported.
//   prefill_bench [--ctx N] [--reps R] sizes...        e.g. prefill_bench 300 2000 8192
// QW_PREFILL_CHUNK sets the chunk (default 8192), QW_PLE_DIR the PLE table (default int4; production: ples_fp8). With --ctx N the slot already holds N tokens (prefilled first,
// not timed), so the timed tokens attend to a long context, as the fresh part of an agent turn does.
// The GPUs' energy (kJ and J per token, from the cards' power sensors) is printed with each size.
// QW_PROFILE adds the engine's split of each prefill (host inputs, jobs, after_chunk); QW_MOE_STATS the balance of
// the routed experts over the ranks.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
        for (int a = 1; a < argc; ++a) {
            const std::string s = argv[a];
            if (s == "--ctx" && a + 1 < argc) ctx = std::atoi(argv[++a]);
            else if (s == "--reps" && a + 1 < argc) reps = std::atoi(argv[++a]);
            else sizes.push_back(std::atoi(argv[a]));
        }
        const int biggest = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
        EngineOptions opt;
        opt.slot_tokens = {std::max(32768, ctx + biggest + 64)};
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;  // production runs ples_fp8
        if (const char *c = std::getenv("QW_PREFILL_CHUNK")) opt.prefill_chunk = std::atoi(c);
        Engine e(opt);
        const std::vector<int32_t> context = soup(ctx, 1);
        PowerMeter meter;
        for (int n : sizes) {
            const std::vector<int32_t> toks = soup(n, 7);
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
            std::printf("prefill %6d tokens after %6d of context, chunk %d: %.3f s, %.0f tok/s, %.2f kJ (%.1f J/token)\n", n, ctx,
                        e.prefill_chunk(), best, n / best, best_kj, best_kj * 1e3 / n);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
