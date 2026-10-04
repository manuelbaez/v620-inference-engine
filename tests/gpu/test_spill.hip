// KV spill: a slot whose KV beyond its VRAM tokens lives in pinned host memory must behave exactly like a slot that is
// all in VRAM (the lists, the arithmetic and the data are the same; only the address of some rows differs), so every
// comparison here is bit-exact:
//   1. prefill (chunks that straddle the VRAM boundary, a second prefill appended in the middle of a chunk's worth)
//      gives identical logits in a resident slot and in slots spilled at different points;
//   2. decode steps (one row per slot) and verification-shaped runs (4 rows of one slot) give identical logits;
//   3. export_kv of ranges inside, across and beyond the boundary gives identical bytes, and import_kv into a
//      spilled slot reproduces a resident slot's bytes.
// Slots: 0 resident, 1 spilled at 4096, 2 spilled at 8192, 3 (different content) spilled at 8192; 24576 tokens each.
// With the prefill staging of the spill on (the default) and off (QW_SPILL_STAGE=0) the results must be the same.
// Usage: test_spill [--n 20000]
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
static float maxdiff(const float *a, const float *b) {
    float m = 0;
    for (int i = 0; i < cfg::VOCAB; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
static bool same(const float *a, const float *b) {
    return std::memcmp(a, b, size_t(cfg::VOCAB) * sizeof(float)) == 0;
}

int main(int argc, char **argv) {
    int n = 20000;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
    try {
        EngineOptions opt;
        opt.slot_tokens = {24576, 4096, 8192, 8192};
        opt.slot_spill = {0, 20480, 16384, 16384};
        opt.warmup = false;
        Engine e(opt);
        int fails = 0;
        const auto check = [&](bool ok, const char *what) {
            std::printf("%-62s %s\n", what, ok ? "ok" : "MISMATCH");
            fails += !ok;
        };
        QW_CHECK(e.slot_capacity(1) == 24576 && e.slot_vram_tokens(1) == 4096, "capacities");

        // ---- 1. prefill
        const auto P = text(n, 1), more = text(777, 2), other = text(10240, 9);
        std::vector<std::vector<float>> lg(3);
        for (int s = 0; s < 3; ++s) {
            e.slot_reset(s);
            e.prefill(s, P);
            lg[size_t(s)] = e.logits();
        }
        check(lg[0] == lg[1], "prefill: slot 1 (spilled at 4096) == resident");
        check(lg[0] == lg[2], "prefill: slot 2 (spilled at 8192) == resident");
        for (int s = 0; s < 3; ++s) {
            e.prefill(s, more);
            lg[size_t(s)] = e.logits();
        }
        check(lg[0] == lg[1] && lg[0] == lg[2], "prefill appended (unaligned start): all three identical");
        e.slot_reset(3);
        e.prefill(3, other);

        // ---- 2. decode
        int tok = argmax(lg[0].data());
        bool ok = true;
        for (int i = 0; i < 40 && ok; ++i) {
            const auto &d = e.decode({{0, tok}, {1, tok}, {2, tok}});
            ok = same(d.data(), d.data() + cfg::VOCAB) && same(d.data(), d.data() + 2 * size_t(cfg::VOCAB));
            tok = argmax(d.data());
        }
        check(ok, "decode, 40 steps, one row per slot: identical logits");
        ok = true;
        for (int i = 0; i < 6 && ok; ++i) {  // runs of 4 rows of one slot: the speculative verification shape
            std::vector<Engine::Row> rows;
            for (int sl = 0; sl < 3; ++sl)
                for (int k = 0; k < 4; ++k) rows.push_back({sl, tok + k});
            const auto &d = e.decode(rows);
            for (int r = 0; r < 4; ++r)
                for (int sl = 1; sl < 3; ++sl)
                    ok = ok && same(d.data() + size_t(r) * cfg::VOCAB, d.data() + size_t(r + 4 * sl) * cfg::VOCAB);
            tok = argmax(d.data() + 3 * size_t(cfg::VOCAB));
            for (int sl = 0; sl < 3; ++sl) e.accept(sl, 2);  // roll two rows back: the next run rewrites those positions
        }
        check(ok, "decode, runs of 4 rows: identical logits");

        // ---- 2b. batched prefill: slot 0 (resident), 1 (spilled; staged when the prefill staging is on) and 2 (spilled;
        // its region no longer fits the staging buffer after slot 1's, so it reads host memory) in one chunk. Segments of
        // one chunk are not bit-identical to each other (a segment cut by the micro-batch split runs its recurrent scan
        // in two pieces), so they are compared within rounding, and a hash of all their logits is printed: the runs with
        // the staging on and off (QW_SPILL_STAGE=0) must print the same hash (2026-10-04: 287027c10cf4e970 both).
        {
            const auto tail = text(1500, 5);
            const auto &seg = e.prefill_batch({{0, tail, {}}, {1, tail, {}}, {2, tail, {}}});
            uint64_t h = 1469598103934665603ull;
            const auto mix = [&](const float *l) {
                for (int i = 0; i < cfg::VOCAB; ++i) {
                    uint32_t u;
                    std::memcpy(&u, l + i, 4);
                    h = (h ^ u) * 1099511628211ull;
                }
            };
            float worst = 0;
            for (int k = 0; k < 3; ++k) {
                mix(seg[size_t(k)].data());
                worst = std::max(worst, maxdiff(seg[0].data(), seg[size_t(k)].data()));
            }
            int t = argmax(seg[0].data());
            float worst_dec = 0;
            for (int i = 0; i < 8; ++i) {
                const auto &d = e.decode({{0, t}, {1, t}, {2, t}});
                for (int k = 0; k < 3; ++k) {
                    mix(d.data() + size_t(k) * cfg::VOCAB);
                    worst_dec = std::max(worst_dec, maxdiff(d.data(), d.data() + size_t(k) * cfg::VOCAB));
                }
                t = argmax(d.data());
            }
            std::printf("batched prefill: max |logit difference| between the slots %.4f, decode after it %.4f; hash %016llx\n",
                        worst, worst_dec, (unsigned long long)h);
            // the noise floor of batched prefill is ~1 in the logits (tests/gpu/test_batch_prefill.hip): the first greedy
            // token must match and the prefill logits stay within it; a wrong staging would differ by far more
            const bool first = argmax(seg[1].data()) == argmax(seg[0].data()) && argmax(seg[2].data()) == argmax(seg[0].data());
            check(first && worst < 1.0f, "batched prefill: resident, staged and unstaged spilled slots agree (rounding)");
        }

        // ---- 3. export / import of KV ranges (slot 0 and slot 1 hold the same positions)
        const int64_t len = std::min(e.slot_len(0), e.slot_len(1));
        struct Range {
            int64_t p0, n;
            const char *name;
        };
        const Range ranges[] = {{0, 2048, "inside VRAM [0,2048)"},
                                {2048, 4096, "across the boundary [2048,6144)"},
                                {12288, 4096, "beyond it [12288,16384)"}};
        const auto dump = [&](int slot, const Range &r) {
            const size_t bytes = e.kv_rank_bytes(r.n);
            std::vector<std::vector<uint8_t>> bufs(RANKS, std::vector<uint8_t>(bytes));
            Engine::RankBufs rb{};
            for (int k = 0; k < RANKS; ++k) rb[size_t(k)] = bufs[size_t(k)].data();
            e.export_kv(slot, r.p0, r.n, rb);
            e.host_copies_wait();
            return bufs;
        };
        for (const Range &r : ranges) {
            QW_CHECK(r.p0 + r.n <= len, "test_spill: --n too small for the export ranges");
            const auto a = dump(0, r), b = dump(1, r);
            bool eq = true;
            for (int k = 0; k < RANKS; ++k) eq = eq && a[size_t(k)] == b[size_t(k)];
            char what[96];
            std::snprintf(what, sizeof what, "export_kv %s: spilled slot == resident", r.name);
            check(eq, what);
        }
        {  // import slot 0's blocks into slot 3 (other content, spilled at 8192) and read them back
            const Range r{6144, 4096, "[6144,10240)"};
            const auto from = dump(0, r);
            const auto before = dump(3, r);
            check(before != from, "import test setup: slot 3 holds different KV before");
            const size_t bytes = e.kv_rank_bytes(r.n);
            Engine::RankBufs rb{};
            std::vector<std::vector<uint8_t>> src = from;
            for (int k = 0; k < RANKS; ++k) rb[size_t(k)] = src[size_t(k)].data();
            e.import_kv(3, r.p0, r.n, rb);
            e.host_copies_wait();
            const auto after = dump(3, r);
            check(after == from && bytes == from[0].size(), "import_kv across the boundary, read back: == source");
        }
        std::printf("%s\n", fails ? "FAILED" : "all spill checks passed");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "test_spill: %s\n", ex.what());
        return 1;
    }
}
