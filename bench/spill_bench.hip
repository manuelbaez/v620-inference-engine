// KV spill cost: prefill speed and decode step time at growing context, in a slot that is all in VRAM (A) and in a slot
// whose KV beyond its VRAM tokens is in pinned host memory (B), in the same engine and process so the two are measured
// side by side. Contexts at or below B's VRAM tokens show the cost of the spill path itself; above them, the cost of
// reading and writing host memory.
//   spill_bench [--cap N=262144] [--vram V=32768] [--ctxs 8192,32768,...] [--steps S=32] [--run K=1] [--reps R=3]
//               [--no-b] [--tokens-file F]
// The slots hold the same tokens at every point. Prefill is timed per increment (the tokens from the slot's length to
// the next context, attending to what is already there). Decode steps carry K rows per slot (K=1 plain decoding,
// K=4 the shape of speculative verification) and use GPU sampling without logits, as the server does; reported are the
// median ms per step of A alone, B alone, and A and B in the same step. On a tree without KV spill (no
// QW_HAVE_SPILL) there is no B, so the same binary source gives the baseline of the unmodified engine.
// QW_PREFILL_CHUNK sets the chunk (default 4096, as production), QW_PLE_DIR the PLE table. --tokens-file: the prompt
// is the file's tokens (real text, tools/text_tokens.py) and decode steps feed its continuation; without it a
// pseudo-random token sequence. QW_SPILL_CACHE_STATS=1: the spill cache's hit rate for the decode steps of each context.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<int32_t> soup(int n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = 12345;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}

int main(int argc, char **argv) {
    try {
        int cap = 262144, vram = 32768, steps = 32, run = 1, reps = 3;
        bool want_b = true;
        std::vector<int> ctxs = {8192, 32768, 65536, 131072};
        std::vector<int32_t> file;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--cap" && i + 1 < argc) cap = std::atoi(argv[++i]);
            else if (a == "--vram" && i + 1 < argc) vram = std::atoi(argv[++i]);
            else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
            else if (a == "--run" && i + 1 < argc) run = std::atoi(argv[++i]);
            else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
            else if (a == "--no-b") want_b = false;
            else if (a == "--tokens-file" && i + 1 < argc) {
                std::ifstream f(argv[++i]);
                std::string t;
                while (f >> t) {
                    std::replace(t.begin(), t.end(), ',', ' ');
                    std::stringstream ss(t);
                    for (std::string u; ss >> u;) file.push_back(int32_t(std::atoi(u.c_str())));
                }
            }
            else if (a == "--ctxs" && i + 1 < argc) {
                ctxs.clear();
                std::stringstream ss(argv[++i]);
                for (std::string t; std::getline(ss, t, ',');) ctxs.push_back(std::atoi(t.c_str()));
            }
        }
        EngineOptions opt;
        opt.prefill_chunk = 4096;
        if (const char *c = std::getenv("QW_PREFILL_CHUNK")) opt.prefill_chunk = std::atoi(c);
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;
#ifdef QW_HAVE_SPILL
        const bool have_b = want_b;
        if (have_b) {
            opt.slot_tokens = {cap, vram};
            opt.slot_spill = {0, cap - vram};
        } else {
            opt.slot_tokens = {cap};
        }
#else
        const bool have_b = false;
        opt.slot_tokens = {cap};
#endif
        Engine e(opt);
        const int slots = have_b ? 2 : 1;
        std::vector<int32_t> all = soup(cap);
        if (!file.empty())
            for (size_t i = 0; i < all.size(); ++i) all[i] = file[i % file.size()];
        std::printf("cap %d, B's VRAM tokens %d, chunk %d, %d row(s) per slot, %d steps\n", cap, vram, opt.prefill_chunk,
                    run, steps);
        std::printf("%9s | %-23s | %-34s\n", "context", "prefill tok/s (increment)", "decode ms/step: A | B | A+B");
        for (int s = 0; s < slots; ++s) e.slot_reset(s);

        const auto decode_ms = [&](const std::vector<int> &which) {
            std::vector<Engine::Row> rows;
            Engine::SampleSpec spec;
            for (int s : which)
                for (int k = 0; k < run; ++k) {
                    rows.push_back({s, 0});  // the token is set per step: the text's continuation
                    gpu::SampleRow sr{};
                    sr.inv_t = 0.f;
                    sr.repetition = 1.f;
                    spec.rows.push_back(sr);
                }
            std::vector<double> t;
            for (int i = 0; i < steps + 4; ++i) {
                for (size_t j = 0; j < rows.size(); ++j)
                    rows[j].token = all[size_t(e.slot_len(rows[j].slot) + int64_t(j % size_t(run))) % all.size()];
                const double t0 = now();
                e.decode(rows, &spec);
                if (i >= 4) t.push_back((now() - t0) * 1e3);
                for (int s : which)
                    if (run > 1) e.accept(s, 1);  // keep the context where it is: only the first row is kept
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };

        for (int ctx : ctxs) {
            if (ctx + 2048 > cap) break;  // decode steps drift the length by a few hundred tokens
            double pre[2] = {0, 0};
            for (int s = 0; s < slots; ++s) {
                // a decode phase appends tokens and (with accept) rewinds; continue from the slot's own length
                const int64_t len = e.slot_len(s);
                if (len >= ctx) continue;
                const std::vector<int32_t> inc(all.begin() + len, all.begin() + ctx);
                const double t0 = now();
                e.prefill(s, inc);
                pre[s] = double(inc.size()) / (now() - t0);
            }
#ifdef QW_HAVE_SPILL_CACHE
            e.spill_cache_stats();  // zero (QW_SPILL_CACHE_STATS=1 counts)
#endif
            double best_a = 1e9, best_b = 1e9, best_ab = 1e9;
            for (int r = 0; r < reps; ++r) {
                best_a = std::min(best_a, decode_ms({0}));
                if (have_b) {
                    best_b = std::min(best_b, decode_ms({1}));
                    best_ab = std::min(best_ab, decode_ms({0, 1}));
                }
            }
            std::printf("%9d | A %7.0f  B %7.0f      | %7.2f | %7.2f | %7.2f", ctx, pre[0], pre[1], best_a,
                        have_b ? best_b : 0.0, have_b ? best_ab : 0.0);
#ifdef QW_HAVE_SPILL_CACHE
            const auto cs = e.spill_cache_stats();
            if (cs[0] + cs[1]) std::printf("   spill cache hits %.1f%%", 100.0 * double(cs[0]) / double(cs[0] + cs[1]));
#endif
            std::printf("\n");
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "spill_bench: %s\n", ex.what());
        return 1;
    }
}
