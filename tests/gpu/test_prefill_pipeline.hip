// Preparing the next chunk's embeddings and PLE rows while the GPUs run the current one (QW_PREFILL_PIPELINE)
// only moves host work: a multi-chunk prefill must give bit-identical logits, leave the slot in the same state (the
// next decode steps agree) and call after_chunk at the same positions, with captures in the middle.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> soup(int n, int seed) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = uint32_t(seed) * 2654435761u + 1;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}

int main() {
    try {
        EngineOptions opt;
        opt.slot_tokens = {8192};
        opt.prefill_chunk = 1024;
        opt.warmup = false;
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;
        Engine e(opt);
        int fails = 0;
        auto check = [&](bool ok, const char *what) {
            if (!ok) std::printf("FAIL: %s\n", what);
            fails += !ok;
        };
        const auto prompt = soup(5000, 4);  // 4 full chunks and a partial one
        std::vector<Engine::Capture> caps = {{1500, 0}, {3100, 1}, {4999, 2}};
        struct Run {
            std::vector<float> logits, step1, step2;
            std::vector<int64_t> positions;
            double secs = 0;
        };
        auto run = [&](bool pipeline) {
            e.set_prefill_pipeline(pipeline);
            e.slot_reset(0);
            Run r;
            const auto t0 = std::chrono::steady_clock::now();
            r.logits = e.prefill(0, prompt, nullptr, [&](int64_t pos) { r.positions.push_back(pos); }, caps);
            r.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            int32_t tok = 0;
            for (size_t i = 1; i < r.logits.size(); ++i)
                if (r.logits[i] > r.logits[size_t(tok)]) tok = int32_t(i);
            r.step1 = e.step(tok);
            tok = 0;
            for (size_t i = 1; i < r.step1.size(); ++i)
                if (r.step1[i] > r.step1[size_t(tok)]) tok = int32_t(i);
            r.step2 = e.step(tok);
            return r;
        };
        run(false);  // warm
        const Run a = run(false), b = run(true), c = run(false), d = run(true);
        auto same = [](const std::vector<float> &x, const std::vector<float> &y) {
            return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * 4) == 0;
        };
        check(same(a.logits, c.logits), "plain runs are repeatable (the baseline of the comparison)");
        check(same(a.logits, b.logits) && same(a.logits, d.logits), "pipelined prefill: identical logits");
        check(same(a.step1, b.step1) && same(a.step2, d.step2), "pipelined prefill: the next decode steps agree");
        check(a.positions == b.positions, "after_chunk is called at the same positions");
        std::printf("prefill of %zu tokens: plain %.3f / %.3f s, pipelined %.3f / %.3f s\n", prompt.size(), a.secs, c.secs, b.secs, d.secs);
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
