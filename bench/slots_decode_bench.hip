// The engine's decode step by how its rows are spread over slots: S slots with k rows each (S * k <= 16), e.g. 8 x 2
// (eight streams with one draft row each) against 16 x 1 (sixteen streams). Engine::decode only: no sampling, no
// session, so what differs between the shapes is on the cards and in the engine's own per-step host work.
//   slots_decode_bench [--tokens-file F] [--ctx 400] [--steps 48]
// Sixteen slots of 8,192 tokens; each slot is prefilled with its own --ctx tokens (of F, or an arithmetic sequence).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

int main(int argc, char **argv) {
    try {
        int ctx = 400, steps = 48;
        std::vector<int32_t> text;
        for (int a = 1; a < argc; ++a) {
            const std::string s = argv[a];
            if (s == "--ctx" && a + 1 < argc)
                ctx = std::atoi(argv[++a]);
            else if (s == "--steps" && a + 1 < argc)
                steps = std::atoi(argv[++a]);
            else if (s == "--tokens-file" && a + 1 < argc) {
                std::ifstream f(argv[++a]);
                std::string tok;
                while (std::getline(f, tok, ','))
                    if (!tok.empty() && tok.find_first_not_of(" \n\r\t") != std::string::npos) text.push_back(int32_t(std::stol(tok)));
            }
        }
        constexpr int S_MAX = Engine::MAX_SLOTS;
        EngineOptions opt;
        opt.slot_tokens.assign(S_MAX, 8192);
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;
        Engine e(opt);
        const auto prompt = [&](int s) {
            std::vector<int32_t> p(static_cast<size_t>(ctx));
            for (int i = 0; i < ctx; ++i)
                p[size_t(i)] = text.size() >= size_t(S_MAX) * size_t(ctx) ? text[size_t(s) * size_t(ctx) + size_t(i)]
                                                                           : 100 + ((i + s * 977) * 7919) % 30000;
            return p;
        };
        const int shapes[][2] = {{1, 1}, {8, 1}, {8, 2}, {16, 1}, {4, 4}, {4, 2}, {2, 8}, {12, 1}, {1, 16}, {16, 1}, {8, 2}};
        for (const auto &sh : shapes) {
            const int S = sh[0], k = sh[1];
            std::vector<int> cur(static_cast<size_t>(S));
            for (int s = 0; s < S; ++s) {
                e.slot_reset(s);
                const auto &l = e.prefill(s, prompt(s));
                cur[size_t(s)] = int(std::max_element(l.begin(), l.end()) - l.begin());
            }
            std::vector<Engine::Row> rows;
            double dt = 0;
            for (int i = 0; i < steps + 4; ++i) {
                rows.clear();
                for (int s = 0; s < S; ++s)
                    for (int j = 0; j < k; ++j) rows.push_back({s, cur[size_t(s)]});
                const auto t0 = std::chrono::steady_clock::now();
                const auto &d = e.decode(rows);
                if (i >= 4) dt += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                for (int s = 0; s < S; ++s) {
                    const float *l = d.data() + size_t(s * k) * cfg::VOCAB;
                    cur[size_t(s)] = int(std::max_element(l, l + cfg::VOCAB) - l);
                }
            }
            std::printf("%2d slots x %2d rows = %2d rows: %.2f ms/step, %.2f ms per row\n", S, k, S * k, dt / steps * 1e3,
                        dt / steps * 1e3 / (S * k));
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "slots_decode_bench: %s\n", ex.what());
        return 1;
    }
}
