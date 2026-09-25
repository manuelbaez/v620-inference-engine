// Prefix-cache costs and wins on the prefill path:
//   1. prefill tok/s of a 16k prompt with no captures vs 4 captures per chunk;
//   2. set_prompt of a 12k-token system prompt + message with the block store
//      off vs on (the eager save of blocks and snapshots);
//   3. time to first token of a second conversation sharing that system prompt.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

using namespace qw;

constexpr int32_t BOUNDARY = 7;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<int32_t> text(int n, int seed) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = uint32_t(seed) * 2654435761u + 1;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}

static std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t> &b) {
    a.push_back(BOUNDARY);
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

int main() {
    try {
        EngineOptions opt;
        opt.slot_tokens = {32768, 32768};
        Engine e(opt);
        const auto P = text(16384, 1);
        const int chunk = e.prefill_chunk();
        for (int with = 0; with < 2; ++with) {
            std::vector<Engine::Capture> caps;
            if (with)
                for (int64_t c0 = 0; c0 < int64_t(P.size()); c0 += chunk)
                    for (int j = 0; j < Engine::MAX_CAPTURES; ++j) caps.push_back({c0 + 1000 + 1500 * j, j});
            double best = 1e9;
            for (int rep = 0; rep < 3; ++rep) {
                e.slot_reset(0);
                const double t0 = now();
                e.prefill(0, P, nullptr, nullptr, caps);
                best = std::min(best, now() - t0);
            }
            std::printf("prefill %zu tokens, %zu captures: %.3f s, %.0f tok/s\n", P.size(), caps.size(), best,
                        double(P.size()) / best);
        }

        const auto sys = text(12000, 2);
        const auto c1 = cat(sys, text(500, 3)), c2 = cat(sys, text(400, 4));
        for (const char *gb : {"0", "16"}) {
            setenv("QW_HOST_CACHE_GB", gb, 1);
            unsetenv("QW_DISK_CACHE_DIR");
            e.slot_reset(0);
            e.slot_reset(1);
            Session s(e);
            s.set_boundary_token(BOUNDARY);
            const int a = s.acquire(c1, 16);
            double t0 = now();
            s.set_prompt(a, c1);
            const double t1 = now() - t0;
            const int b = s.acquire(c2, 16);  // the other slot (a stays busy)
            t0 = now();
            const int64_t reused = s.set_prompt(b, c2);
            const double t2 = now() - t0;
            std::printf("store %s GB: conversation 1 (%zu tokens) %.3f s; conversation 2 (%zu tokens) %.3f s, reused "
                        "%lld\n",
                        gb, c1.size(), t1, c2.size(), t2, (long long)reused);
            s.release(a);
            for (int k = 0; k < 2; ++k) {  // unrelated conversations, with the store's memory already allocated
                const auto c3 = cat(text(12000, 5 + k), text(500, 7 + k));
                const int d = s.acquire(c3, 16);
                t0 = now();
                s.set_prompt(d, c3);
                std::printf("store %s GB: unrelated conversation (%zu tokens) %.3f s\n", gb, c3.size(), now() - t0);
                s.release(d);
            }
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
