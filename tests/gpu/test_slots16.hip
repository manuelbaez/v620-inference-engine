// Sixteen slots: slots 8 to 15 must behave exactly like slots 0 to 7 (the spill cache's tags and miss list carry the
// slot number in a bit field, kernels/types.hpp; with 3 bits slot s + 8 would be served slot s's rows). Slots 0-7 hold
// text A, slots 8-15 text B; slots 0 and 8 are resident, the others spilled at 2,048 tokens. Every comparison is
// bit-exact, each spilled slot against the resident slot of its text:
//   1. prefill: identical logits;
//   2. decode steps of 16 rows, one per slot (after the first steps the spilled slots read their rows from the VRAM
//      cache, each under its own tag): identical logits within each group;
//   3. runs of 4 rows in slots 0, 3, 8, 12 and 15 (the speculative verification shape, with a rollback): identical;
//   4. slot 15 reset and given other text while the others continue: the others still agree.
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
        constexpr int S = Engine::MAX_SLOTS, G = S / 2;  // slot s holds text s / G; slots 0 and G are resident
        EngineOptions opt;
        opt.slot_tokens.assign(S, 2048);
        opt.slot_spill.assign(S, 14336);
        for (int r : {0, G}) {
            opt.slot_tokens[size_t(r)] = 16384;
            opt.slot_spill[size_t(r)] = 0;
        }
        opt.warmup = false;
        Engine e(opt);
        int fails = 0;
        const auto check = [&](bool ok, const char *what) {
            std::printf("%-78s %s\n", what, ok ? "ok" : "MISMATCH");
            fails += !ok;
        };
        const auto ref = [&](int s) { return s / G * G; };
        // ---- 1. prefill
        const std::vector<int32_t> P[2] = {text(n, 1), text(n, 2)};
        std::vector<float> l0[2];
        bool ok = true;
        for (int s = 0; s < S; ++s) {
            e.slot_reset(s);
            e.prefill(s, P[s / G]);
            if (s == ref(s))
                l0[s / G] = e.logits();
            else
                ok = ok && l0[s / G] == e.logits();
        }
        check(ok && l0[0] != l0[1], "prefill: spilled slots == the resident slot of their text");
        // ---- 2. decode, one row per slot
        int tok[2] = {argmax(l0[0].data()), argmax(l0[1].data())};
        ok = true;
        int bad_slot = -1;
        for (int i = 0; i < 48 && ok; ++i) {
            std::vector<Engine::Row> rows;
            for (int s = 0; s < S; ++s) rows.push_back({s, tok[s / G]});
            const auto &d = e.decode(rows);
            for (int s = 0; s < S && ok; ++s)
                if (!same(d.data() + size_t(ref(s)) * cfg::VOCAB, d.data() + size_t(s) * cfg::VOCAB)) ok = false, bad_slot = s;
            for (int g = 0; g < 2; ++g) tok[g] = argmax(d.data() + size_t(g * G) * cfg::VOCAB);
        }
        if (!ok) std::printf("  first differing slot: %d\n", bad_slot);
        check(ok, "decode, 48 steps of 16 rows, one per slot: identical logits within each text");
        // ---- 3. runs of 4 rows
        const int run_slots[5] = {0, 3, 8, 12, 15};
        ok = true;
        for (int i = 0; i < 6 && ok; ++i) {
            std::vector<Engine::Row> rows;
            for (int sl : run_slots)
                for (int k = 0; k < 4; ++k) rows.push_back({sl, tok[sl / G] + k});
            const auto &d = e.decode(rows);
            const auto row = [&](int j, int r) { return d.data() + size_t(4 * j + r) * cfg::VOCAB; };
            for (int r = 0; r < 4; ++r) ok = ok && same(row(0, r), row(1, r)) && same(row(2, r), row(3, r)) && same(row(2, r), row(4, r));
            tok[0] = argmax(row(0, 3));
            tok[1] = argmax(row(2, 3));
            for (int sl : run_slots) e.accept(sl, 2);  // roll two rows back: the next run rewrites those positions
        }
        check(ok, "decode, runs of 4 rows in slots 0, 3 and 8, 12, 15: identical logits within each text");
        // ---- 4. one slot changes its content; 8 and 12 stay in step with each other
        e.slot_reset(15);
        e.prefill(15, text(n, 7));
        ok = true;
        for (int i = 0; i < 16 && ok; ++i) {
            const auto &d = e.decode({{7, tok[1]}, {8, tok[1]}, {12, tok[1]}, {15, tok[1]}});
            ok = same(d.data() + cfg::VOCAB, d.data() + 2 * size_t(cfg::VOCAB));
            ok = ok && !same(d.data() + cfg::VOCAB, d.data() + 3 * size_t(cfg::VOCAB));  // other text: other logits
            tok[1] = argmax(d.data() + cfg::VOCAB);
        }
        check(ok, "after slot 15 took other text: 8 and 12 identical, 15 different");
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
