// Sixteen slots: every slot must behave exactly like slot 0, including the ones numbered 8 to 15 (the spill cache's
// tags and miss list carry the slot number in a bit field, kernels/types.hpp), so every comparison is bit-exact:
//   1. the same prompt prefilled into a resident slot (0) and fifteen slots spilled at 2,048 tokens: identical logits;
//   2. decode steps of 16 rows, one per slot, with the same token: identical logits in every row (after the first
//      steps the spilled slots read their rows from the VRAM cache, each under its own tag);
//   3. runs of 4 rows in slots 0, 9, 12 and 15 (the speculative verification shape, with a rollback): identical;
//   4. slot 15 reset and given other text while the others continue: the others still agree (its cache entries
//      must not be served to them, nor theirs to it).
// Usage: test_slots16 [--n 9000]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> text(int n, int seed) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = uint32_t(seed) * 2654435761u + 1;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}
static int argmax(const float *l) {
    return int(std::max_element(l, l + cfg::VOCAB) - l);
}
static bool same(const float *a, const float *b) {
    return std::memcmp(a, b, size_t(cfg::VOCAB) * sizeof(float)) == 0;
}

int main(int argc, char **argv) {
    int n = 9000;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
    try {
        constexpr int S = Engine::MAX_SLOTS;
        EngineOptions opt;
        opt.slot_tokens.assign(S, 2048);
        opt.slot_spill.assign(S, 14336);
        opt.slot_tokens[0] = 16384;  // resident
        opt.slot_spill[0] = 0;
        opt.warmup = false;
        Engine e(opt);
        int fails = 0;
        const auto check = [&](bool ok, const char *what) {
            std::printf("%-78s %s\n", what, ok ? "ok" : "MISMATCH");
            fails += !ok;
        };
        // ---- 1. prefill
        const auto P = text(n, 1);
        std::vector<float> l0;
        bool ok = true;
        for (int s = 0; s < S; ++s) {
            e.slot_reset(s);
            e.prefill(s, P);
            if (s == 0)
                l0 = e.logits();
            else
                ok = ok && l0 == e.logits();
        }
        check(ok, "prefill: slots 1..15 (spilled at 2048) == slot 0 (resident)");
        // ---- 2. decode, one row per slot
        int tok = argmax(l0.data());
        ok = true;
        int bad_slot = -1;
        for (int i = 0; i < 48 && ok; ++i) {
            std::vector<Engine::Row> rows;
            for (int s = 0; s < S; ++s) rows.push_back({s, tok});
            const auto &d = e.decode(rows);
            for (int s = 1; s < S && ok; ++s)
                if (!same(d.data(), d.data() + size_t(s) * cfg::VOCAB)) ok = false, bad_slot = s;
            tok = argmax(d.data());
        }
        if (!ok) std::printf("  first differing slot: %d\n", bad_slot);
        check(ok, "decode, 48 steps of 16 rows, one per slot: identical logits");
        // ---- 3. runs of 4 rows
        const int run_slots[4] = {0, 9, 12, 15};
        ok = true;
        for (int i = 0; i < 6 && ok; ++i) {
            std::vector<Engine::Row> rows;
            for (int sl : run_slots)
                for (int k = 0; k < 4; ++k) rows.push_back({sl, tok + k});
            const auto &d = e.decode(rows);
            for (int r = 0; r < 4; ++r)
                for (int j = 1; j < 4; ++j)
                    ok = ok && same(d.data() + size_t(r) * cfg::VOCAB, d.data() + size_t(r + 4 * j) * cfg::VOCAB);
            tok = argmax(d.data() + 3 * size_t(cfg::VOCAB));
            for (int sl : run_slots) e.accept(sl, 2);  // roll two rows back: the next run rewrites those positions
        }
        check(ok, "decode, runs of 4 rows in slots 0, 9, 12, 15: identical logits");
        // ---- 4. one slot changes its content; 0, 9 and 12 are in step with each other (and 15 was until now)
        e.slot_reset(15);
        e.prefill(15, text(n, 7));
        ok = true;
        for (int i = 0; i < 16 && ok; ++i) {
            const auto &d = e.decode({{0, tok}, {9, tok}, {12, tok}, {15, tok}});
            ok = same(d.data(), d.data() + cfg::VOCAB) && same(d.data(), d.data() + 2 * size_t(cfg::VOCAB));
            ok = ok && !same(d.data(), d.data() + 3 * size_t(cfg::VOCAB));  // other text: other logits
            tok = argmax(d.data());
        }
        check(ok, "after slot 15 took other text: 0, 9, 12 identical, 15 different");
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
