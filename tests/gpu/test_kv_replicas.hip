// The KV of ranks that hold the same KV head is byte-identical, which the prefix cache relies on to keep one
// copy per group (QW_KV_PAIRS): checked on blocks exported from a real prefill, at several positions and a partial
// block. Then the export rules: a buffer shared by a group is written once and equals the group's first rank's
// copy, and a rank without a buffer is skipped.
#include <algorithm>
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
        opt.slot_tokens = {4096};
        opt.warmup = false;
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;
        Engine e(opt);
        int fails = 0;
        auto check = [&](bool ok, const char *what) {
            if (!ok) std::printf("FAIL: %s\n", what);
            fails += !ok;
        };
        e.slot_reset(0);
        e.prefill(0, soup(2000, 3));  // 7 full blocks and a partial one
        struct Range {
            int64_t p0, n;
        };
        for (const Range rg : {Range{0, 256}, Range{256, 256}, Range{1024, 256}, Range{1792, 208}}) {
            const size_t bytes = e.kv_rank_bytes(rg.n);
            std::vector<std::vector<uint8_t>> own(RANKS, std::vector<uint8_t>(bytes, 0xee));
            Engine::RankBufs all;
            for (int r = 0; r < RANKS; ++r) all[size_t(r)] = own[size_t(r)].data();
            e.export_kv(0, rg.p0, rg.n, all);
            e.host_copies_wait();
            bool pairs = true, groups_differ = false;
            for (int r = 0; r < RANKS; ++r) pairs = pairs && own[size_t(r)] == own[size_t(cfg::kv_primary(r))];
            groups_differ = own[0] != own[size_t(cfg::KV_REPLICAS)];
            std::printf("block at %5lld (+%3lld): %s%s\n", (long long)rg.p0, (long long)rg.n,
                        pairs ? "replicas identical" : "REPLICAS DIFFER", groups_differ ? ", groups differ" : "");
            check(pairs, "the ranks of a KV group hold identical bytes");
            check(groups_differ, "different KV heads hold different bytes");

            // shared buffers: written once per group, equal to the first rank's copy
            std::vector<std::vector<uint8_t>> sh(RANKS, std::vector<uint8_t>(bytes, 0xee));
            Engine::RankBufs shared;
            for (int r = 0; r < RANKS; ++r) shared[size_t(r)] = sh[size_t(cfg::kv_primary(r))].data();
            e.export_kv(0, rg.p0, rg.n, shared);
            e.host_copies_wait();
            bool once = true;
            for (int r = 0; r < RANKS; ++r)
                if (cfg::kv_primary(r) == r) once = once && sh[size_t(r)] == own[size_t(r)];
                else once = once && std::all_of(sh[size_t(r)].begin(), sh[size_t(r)].end(), [](uint8_t b) { return b == 0xee; });
            check(once, "a shared buffer is written once and equals the first rank's copy; the replicas' own are untouched");

            // a rank without a buffer is skipped: only the replicas, as BlockStore's pair check exports them
            std::vector<std::vector<uint8_t>> rep(RANKS, std::vector<uint8_t>(bytes, 0xee));
            Engine::RankBufs only_replicas{};
            for (int r = 0; r < RANKS; ++r)
                if (cfg::kv_primary(r) != r) only_replicas[size_t(r)] = rep[size_t(r)].data();
            e.export_kv(0, rg.p0, rg.n, only_replicas);
            e.host_copies_wait();
            bool ok = true;
            for (int r = 0; r < RANKS; ++r)
                if (cfg::kv_primary(r) != r) ok = ok && rep[size_t(r)] == own[size_t(r)];
            check(ok, "a null buffer is skipped and the others are exported");
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
