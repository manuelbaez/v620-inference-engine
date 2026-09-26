// Prefill speed by prompt size: each size prefilled 3 times into an empty
// slot (after the load-time warm-up), best time reported.
//   prefill_bench 300 2000 8192
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

int main(int argc, char **argv) {
    try {
        EngineOptions opt;
        opt.slot_tokens = {32768};
        Engine e(opt);
        for (int a = 1; a < argc; ++a) {
            const int n = std::atoi(argv[a]);
            std::vector<int32_t> toks(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) toks[size_t(i)] = 100 + (i * 7919) % 30000;
            double best = 1e9;
            for (int rep = 0; rep < 3; ++rep) {
                e.slot_reset(0);
                const auto t0 = std::chrono::steady_clock::now();
                e.prefill(0, toks);
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            std::printf("prefill %6d tokens: %.3f s, %.0f tok/s\n", n, best, n / best);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
