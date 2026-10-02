// A long prompt prefilled cold through the Session with the prefix cache's block store, under different store
// settings, alternating in one process (each run builds a new Session, so the pinned pools start empty, as after a
// restart; a freed Session gives its arenas back). Measures what the saves cost the prefill: the same prompt with no
// store at all is the floor.
//   cold_prefill_bench [--tokens N=65536] [--reps R=2] [--host-gb G=48] [--configs name,name,...]
// Configs (environment of the run): nostore (QW_HOST_CACHE_GB=0), then settings of the store over today's
// defaults: old, noncoherent, huge, ahead2, ahead4, pairs, pairs+check, huge+ahead2, noncoherent+ahead2,
// all (huge + ahead2 + pairs), nc+pairs (noncoherent + ahead2 + pairs).
// QW_PREFILL_CHUNK (default 4096, as in production) sets the chunk, QW_PLE_DIR the PLE table (production: ples_fp8). Prints per run: seconds, tok/s, the pools'
// arenas pinned and the time that took, saves that waited for an arena and for how long, RAM held by the cache.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

using namespace qw;

constexpr int32_t BOUNDARY = 7;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// a chat-like prompt: segments of 1,200-2,800 tokens of token soup, a boundary token between them
static std::vector<int32_t> conversation(int n, uint32_t seed) {
    std::vector<int32_t> v;
    v.reserve(size_t(n));
    uint32_t x = seed * 2654435761u + 1;
    auto next = [&] {
        x = x * 1664525u + 1013904223u;
        return x >> 8;
    };
    while (int(v.size()) < n) {
        const int seg = 1200 + int(next() % 1600);
        v.push_back(BOUNDARY);
        for (int i = 0; i < seg && int(v.size()) < n; ++i) v.push_back(int32_t(100 + next() % 30000));
    }
    return v;
}

struct Config {
    std::string name;
    std::map<std::string, std::string> env;
};

static const char *KNOBS[] = {"QW_POOL_ARENA", "QW_RESERVE_AHEAD", "QW_KV_PAIRS", "QW_KV_PAIR_CHECK"};

static std::vector<Config> all_configs(double host_gb) {
    const std::string gb = std::to_string(host_gb);
    std::vector<Config> c;
    c.push_back({"nostore", {{"QW_HOST_CACHE_GB", "0"}}});
    auto store = [&](const std::string &name, std::map<std::string, std::string> e) {
        e["QW_HOST_CACHE_GB"] = gb;
        c.push_back({name, e});
    };
    store("old", {});
    store("noncoherent", {{"QW_POOL_ARENA", "noncoherent"}});
    store("huge", {{"QW_POOL_ARENA", "huge"}});
    store("ahead2", {{"QW_RESERVE_AHEAD", "2"}});
    store("ahead4", {{"QW_RESERVE_AHEAD", "4"}});
    store("pairs", {{"QW_KV_PAIRS", "1"}});
    store("pairs+check", {{"QW_KV_PAIRS", "1"}, {"QW_KV_PAIR_CHECK", "1"}});
    store("huge+ahead2", {{"QW_POOL_ARENA", "huge"}, {"QW_RESERVE_AHEAD", "2"}});
    store("noncoherent+ahead2", {{"QW_POOL_ARENA", "noncoherent"}, {"QW_RESERVE_AHEAD", "2"}});
    store("all", {{"QW_POOL_ARENA", "huge"}, {"QW_RESERVE_AHEAD", "2"}, {"QW_KV_PAIRS", "1"}});
    store("nc+pairs", {{"QW_POOL_ARENA", "noncoherent"}, {"QW_RESERVE_AHEAD", "2"}, {"QW_KV_PAIRS", "1"}});
    return c;
}

int main(int argc, char **argv) {
    try {
        int tokens = 65536, reps = 2;
        double host_gb = 48;
        std::string only;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
            else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
            else if (a == "--host-gb" && i + 1 < argc) host_gb = std::atof(argv[++i]);
            else if (a == "--configs" && i + 1 < argc) only = argv[++i];
        }
        EngineOptions opt;
        opt.slot_tokens = {tokens + 512};
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;  // production runs ples_fp8
        opt.prefill_chunk = std::getenv("QW_PREFILL_CHUNK") ? std::atoi(std::getenv("QW_PREFILL_CHUNK")) : 4096;
        Engine e(opt);
        unsetenv("QW_DISK_CACHE_DIR");
        std::vector<Config> cfgs = all_configs(host_gb);
        if (!only.empty()) {
            std::vector<Config> keep;
            for (const Config &c : cfgs)
                if (("," + only + ",").find("," + c.name + ",") != std::string::npos) keep.push_back(c);
            cfgs = keep;
        }
        const auto prompt = conversation(tokens, 5);
        std::printf("prompt %d tokens, chunk %d, cache budget %.0f GB, %d reps (order alternates)\n", tokens,
                    e.prefill_chunk(), host_gb, reps);
        std::printf("%-20s %8s %8s | %6s %8s | %6s %8s | %8s %6s\n", "config", "seconds", "tok/s", "pins", "pin s", "waits",
                    "wait s", "RAM GB", "pairs!");
        std::map<std::string, std::vector<double>> times;
        auto run = [&](const Config &c, bool report) {
            for (const char *k : KNOBS) unsetenv(k);
            unsetenv("QW_HOST_CACHE_GB");
            for (const auto &[k, v] : c.env) setenv(k.c_str(), v.c_str(), 1);
            e.slot_reset(0);
            double secs;
            BlockStore::Stats st;
            {
                Session s(e);
                s.set_boundary_token(BOUNDARY);
                const int slot = s.acquire(prompt, 16);
                QW_CHECK(slot >= 0, "no slot");
                const double t0 = now();
                s.set_prompt(slot, prompt);
                secs = now() - t0;
                st = s.cache_stats();
                s.release(slot);
            }  // the store's arenas are given back here, outside the timing
            if (report) {
                times[c.name].push_back(secs);
                std::printf("%-20s %8.2f %8.0f | %6llu %8.2f | %6llu %8.2f | %8.1f %6llu\n", c.name.c_str(), secs,
                            double(tokens) / secs, (unsigned long long)st.pins, st.pin_s, (unsigned long long)st.pin_waits,
                            st.pin_wait_s, double(st.ram_bytes) / 1e9, (unsigned long long)st.pair_mismatches);
                std::fflush(stdout);
            }
        };
        // a throw-away run: kernels, clocks and the allocator are warm for the first measured one
        {
            Config w{"warm", {{"QW_HOST_CACHE_GB", "0"}}};
            const auto small = conversation(8192, 9);
            for (const char *k : KNOBS) unsetenv(k);
            setenv("QW_HOST_CACHE_GB", "0", 1);
            Session s(e);
            e.slot_reset(0);
            const int slot = s.acquire(small, 16);
            s.set_prompt(slot, small);
            s.release(slot);
        }
        for (int rep = 0; rep < reps; ++rep) {
            std::vector<Config> order = cfgs;
            if (rep % 2) std::reverse(order.begin(), order.end());
            for (const Config &c : order) run(c, true);
        }
        std::printf("\nmean seconds over %d reps (relative to nostore, and to old):\n", reps);
        auto mean = [&](const std::string &n) {
            double s = 0;
            for (double t : times[n]) s += t;
            return times[n].empty() ? 0.0 : s / double(times[n].size());
        };
        const double base = mean("nostore"), old = mean("old");
        for (const Config &c : cfgs)
            std::printf("  %-20s %8.2f s  %+6.1f%% vs nostore  %+6.1f%% vs old\n", c.name.c_str(), mean(c.name),
                        base > 0 ? 100.0 * (mean(c.name) / base - 1) : 0.0, old > 0 ? 100.0 * (mean(c.name) / old - 1) : 0.0);
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
