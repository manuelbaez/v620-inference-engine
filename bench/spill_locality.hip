// Experiment: would a VRAM cache of recently used spilled groups pay? Decoding (one row per step) in a slot with
// 8,192 tokens in VRAM and the rest spilled, it counts per QSA layer how many of the groups a step selects are spilled
// and how many of those the same layer selected within the last K steps (K = 1, 2, 4, 8, 16, 32): the hit rate of a
// cache holding the groups of the last K steps. The cache that needs is estimated as the groups a step selects plus
// (K - 1) times the new ones per step, at 4 KiB (K and V of 4 tokens) per group and layer.
//   spill_locality --tokens-file F [--vram 8192] [--ctxs 32768,100000,200000] [--steps 256]
// The prompt is the file's first tokens (real text: tools/text_tokens.py); decoding is greedy from there.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"
#include "kernels/qsa.hpp"

using namespace qw;

int main(int argc, char **argv) {
    try {
        int vram = 8192, steps = 256;
        std::vector<int> ctxs = {32768, 100000, 200000};
        std::vector<int32_t> text;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--vram" && i + 1 < argc) vram = std::atoi(argv[++i]);
            else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
            else if (a == "--ctxs" && i + 1 < argc) {
                ctxs.clear();
                std::stringstream ss(argv[++i]);
                for (std::string t; std::getline(ss, t, ',');) ctxs.push_back(std::atoi(t.c_str()));
            } else if (a == "--tokens-file" && i + 1 < argc) {
                std::ifstream f(argv[++i]);
                std::string t;
                while (f >> t) {
                    std::replace(t.begin(), t.end(), ',', ' ');
                    std::stringstream ss(t);
                    for (std::string u; ss >> u;) text.push_back(int32_t(std::atoi(u.c_str())));
                }
            }
        }
        QW_CHECK(!text.empty(), "spill_locality: --tokens-file with token ids");
        setenv("QW_SPILL_LOCALITY", "1", 1);
        EngineOptions opt;
        opt.prefill_chunk = 4096;
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;
        opt.slot_tokens = {vram};
        opt.slot_spill = {262144 - vram};
        Engine e(opt);
        e.slot_reset(0);
        constexpr int K[gpu::LOC_NK] = {1, 2, 4, 8, 16, 32};
        std::printf("slot: %d tokens in VRAM, the rest spilled; %d decode steps per context; real text, %zu tokens\n",
                    vram, steps, text.size());
        for (int ctx : ctxs) {
            if (ctx > int(text.size())) break;
            const int64_t len = e.slot_len(0);
            if (len < ctx) e.prefill(0, std::vector<int32_t>(text.begin() + len, text.begin() + ctx));
            e.spill_locality();  // zero
            int tok = int(std::max_element(e.logits().begin(), e.logits().end()) - e.logits().begin());
            for (int i = 0; i < steps; ++i) {
                const auto &d = e.decode({{0, tok}});
                tok = int(std::max_element(d.begin(), d.begin() + cfg::VOCAB) - d.begin());
            }
            const auto st = e.spill_locality();
            double all = 0, sp = 0, hit[gpu::LOC_NK] = {}, hit1_min = 1, hit1_max = 0;
            double cache_mb[gpu::LOC_NK] = {};
            for (int l = 0; l < cfg::N_QSA; ++l) {
                const unsigned long long *s = st.data() + size_t(l) * gpu::LOC_STATS;
                all += double(s[0]);
                sp += double(s[1]);
                for (int k = 0; k < gpu::LOC_NK; ++k) hit[k] += double(s[2 + k]);
                if (s[1]) {
                    const double h1 = double(s[2]) / double(s[1]);
                    hit1_min = std::min(hit1_min, h1);
                    hit1_max = std::max(hit1_max, h1);
                    const double sel = double(s[1]) / steps, fresh = double(s[1] - s[2]) / steps;
                    for (int k = 0; k < gpu::LOC_NK; ++k) cache_mb[k] += (sel + (K[k] - 1) * fresh) * 4096.0 / 1048576.0;
                }
            }
            std::printf("\ncontext %d (+%d decoded): %.0f%% of the selected groups spilled\n", ctx, steps,
                        all > 0 ? 100.0 * sp / all : 0.0);
            std::printf("  cache of the last K steps:");
            for (int k = 0; k < gpu::LOC_NK; ++k) std::printf("  K=%d %.1f%%", K[k], sp > 0 ? 100.0 * hit[k] / sp : 0.0);
            std::printf("\n  per layer, K=1: %.1f%% .. %.1f%%\n  cache size per spilled slot (12 layers):", 100 * hit1_min,
                        100 * hit1_max);
            for (int k = 0; k < gpu::LOC_NK; ++k) std::printf("  K=%d %.0f MB", K[k], cache_mb[k]);
            std::printf("\n");
            // runs of 4 rows with the text's next tokens (the shape of speculative verification with drafts that are
            // all accepted): how many distinct spilled groups the 4 rows select, against their sum
            const int64_t at = e.slot_len(0);
            e.spill_locality();
            for (int i = 0; i < steps / 4; ++i) {
                std::vector<Engine::Row> rows;
                for (int k = 0; k < 4; ++k) rows.push_back({0, text[size_t(at + 4 * i + k) % text.size()]});
                e.decode(rows);
            }
            const auto rs = e.spill_locality();
            double sum = 0, uni = 0;
            for (int l = 0; l < cfg::N_QSA; ++l) {
                sum += double(rs[size_t(l) * gpu::LOC_STATS + gpu::LOC_RUN]);
                uni += double(rs[size_t(l) * gpu::LOC_STATS + gpu::LOC_RUN + 1]);
            }
            std::printf("  runs of 4 rows: distinct spilled groups %.2f x one row's (sum over the rows %.2f x): a fetch once "
                        "per run would read %.0f%% of what the rows read now\n",
                        sum > 0 ? 4.0 * uni / sum : 0.0, 4.0, sum > 0 ? 100.0 * uni / sum : 0.0);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "spill_locality: %s\n", ex.what());
        return 1;
    }
}
