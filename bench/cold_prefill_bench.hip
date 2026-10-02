// A long prompt prefilled cold through the Session with the prefix cache's block store, under different store
// settings, alternating in one process (each run builds a new Session, so the pinned pools start empty, as after a
// restart; a freed Session gives its arenas back). Measures what the saves cost the prefill: the same prompt with no
// store at all is the floor.
//   cold_prefill_bench [--tokens N=65536] [--reps R=2] [--host-gb G=48] [--hog-gb H=0] [--configs name,name,...]
// --hog-gb H holds H GB of memory during the runs, so that pinning has to find pages in a nearly full host, as on
// the production box. Only with the PLE table mlocked (ulimit -l unlimited): otherwise the table is page cache, the
// hog evicts it, and every token's gather becomes a disk read (the dev box's user has an 8 MB limit: do not use it
// there; the slow pins reproduce there without it).
// Configs (environment of the run): nostore (QW_HOST_CACHE_GB=0), then settings of the store over today's
// defaults (see all_configs() for the list: old, noncoherent, huge, ahead2, ahead4, pairs, pairs+check, nc+ahead2,
// ahead2+pairs, nc+pairs, the reserve and serial ones).
// QW_PREFILL_CHUNK (default 4096, as in production) sets the chunk, QW_PLE_DIR the PLE table (production: ples_fp8). Prints per run: seconds, tok/s, the pools'
// arenas pinned and the time that took, saves that waited for an arena and for how long, RAM held by the cache.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

#include <map>
#include <string>
#include <thread>
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

// locked memory that takes the host's free pages for the length of the runs
struct Hog {
    std::vector<std::pair<void *, size_t>> maps;
    void take(double gb) {
        size_t left = size_t(gb * 1e9);
        while (left) {
            const size_t n = std::min<size_t>(left, size_t(1) << 30);
            void *p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) fail("hog: mmap");
            std::memset(p, 1, n);
            if (mlock(p, n) != 0) log("hog: mlock of %zu MB failed (RLIMIT_MEMLOCK?); the pages are held unlocked", n >> 20);
            maps.push_back({p, n});
            left -= n;
        }
    }
    ~Hog() {
        for (auto &[p, n] : maps) munmap(p, n);
    }
};

struct Config {
    std::string name;
    std::map<std::string, std::string> env;
};

static const char *KNOBS[] = {"QW_POOL_ARENA", "QW_RESERVE_AHEAD", "QW_KV_PAIRS", "QW_KV_PAIR_CHECK", "QW_POOL_RESERVE_GB", "QW_POOL_SERIAL"};

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
    // a standing pinned reserve (QW_POOL_RESERVE_GB): the pools pin it at construction and keep it topped up, so the
    // runs start when it is in place (the bench waits for the pools' pins first, as a server would have long since)
    store("ahead2+pairs", {{"QW_RESERVE_AHEAD", "2"}, {"QW_KV_PAIRS", "1"}});
    store("nc+ahead2", {{"QW_POOL_ARENA", "noncoherent"}, {"QW_RESERVE_AHEAD", "2"}});
    store("reserve10+nc+pairs", {{"QW_POOL_RESERVE_GB", "10"}, {"QW_POOL_ARENA", "noncoherent"}, {"QW_RESERVE_AHEAD", "2"}, {"QW_KV_PAIRS", "1"}});
    store("serial+ahead2+pairs", {{"QW_POOL_SERIAL", "1"}, {"QW_RESERVE_AHEAD", "2"}, {"QW_KV_PAIRS", "1"}});
    store("serial", {{"QW_POOL_SERIAL", "1"}});
    store("serial+ahead2", {{"QW_POOL_SERIAL", "1"}, {"QW_RESERVE_AHEAD", "2"}});
    store("reserve16+serial", {{"QW_POOL_RESERVE_GB", "16"}, {"QW_POOL_SERIAL", "1"}});
    store("reserve16", {{"QW_POOL_RESERVE_GB", "16"}});
    store("reserve16+ahead2", {{"QW_POOL_RESERVE_GB", "16"}, {"QW_RESERVE_AHEAD", "2"}});
    store("reserve10+pairs", {{"QW_POOL_RESERVE_GB", "10"}, {"QW_RESERVE_AHEAD", "2"}, {"QW_KV_PAIRS", "1"}});
    return c;
}

int main(int argc, char **argv) {
    try {
        int tokens = 65536, reps = 2;
        double host_gb = 48, hog_gb = 0;
        std::string only;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
            else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
            else if (a == "--host-gb" && i + 1 < argc) host_gb = std::atof(argv[++i]);
            else if (a == "--hog-gb" && i + 1 < argc) hog_gb = std::atof(argv[++i]);
            else if (a == "--configs" && i + 1 < argc) only = argv[++i];
        }
        EngineOptions opt;
        opt.slot_tokens = {tokens + 512};
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;  // production runs ples_fp8
        opt.prefill_chunk = std::getenv("QW_PREFILL_CHUNK") ? std::atoi(std::getenv("QW_PREFILL_CHUNK")) : 4096;
        Engine e(opt);
        unsetenv("QW_DISK_CACHE_DIR");
        Hog hog;
        if (hog_gb > 0) hog.take(hog_gb);
        std::vector<Config> cfgs = all_configs(host_gb);
        if (!only.empty()) {
            std::vector<Config> keep;
            for (const Config &c : cfgs)
                if (("," + only + ",").find("," + c.name + ",") != std::string::npos) keep.push_back(c);
            cfgs = keep;
        }
        const auto prompt = conversation(tokens, 5);
        std::printf("prompt %d tokens, chunk %d, cache budget %.0f GB, %.0f GB of locked memory held, %d reps (order alternates)\n",
                    tokens, e.prefill_chunk(), host_gb, hog_gb, reps);
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
                // a standing reserve is pinned in the background at construction: a server has had it for hours, so
                // wait until the pools stop pinning before the clock starts
                for (int waited = 0, prev = -1; waited < 180; ++waited) {
                    const int pins = int(s.cache_stats().pins);
                    if (pins == prev) break;
                    prev = pins;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                }
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
        // throw-away runs: kernels and clocks are warm, and every kind of arena has been pinned once (the first pin
        // of a process costs ~0.7 s, which would land on whichever configuration runs first)
        for (const char *kind : {"malloc", "noncoherent", "huge"}) {
            const auto small = conversation(8192, 9);
            for (const char *k : KNOBS) unsetenv(k);
            setenv("QW_HOST_CACHE_GB", "4", 1);
            setenv("QW_POOL_ARENA", kind, 1);
            e.slot_reset(0);
            Session s(e);
            s.set_boundary_token(BOUNDARY);
            const int slot = s.acquire(small, 16);
            s.set_prompt(slot, small);
            s.release(slot);
        }
        for (int rep = 0; rep < reps; ++rep) {
            std::vector<Config> order = cfgs;
            if (rep % 2) std::reverse(order.begin(), order.end());
            for (const Config &c : order) run(c, true);
        }
        std::printf("\nover %d reps: min / median / max seconds, and the mean against nostore and against old:\n", reps);
        auto stat = [&](const std::string &n, int which) {  // 0 min, 1 median, 2 max, 3 mean
            std::vector<double> v = times[n];
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            if (which == 0) return v.front();
            if (which == 1) return v[v.size() / 2];
            if (which == 2) return v.back();
            double s = 0;
            for (double t : v) s += t;
            return s / double(v.size());
        };
        const double base = stat("nostore", 3), old = stat("old", 3);
        for (const Config &c : cfgs)
            std::printf("  %-22s %7.2f / %7.2f / %7.2f s   mean %+6.1f%% vs nostore  %+6.1f%% vs old   (median %+6.1f%% vs nostore)\n",
                        c.name.c_str(), stat(c.name, 0), stat(c.name, 1), stat(c.name, 2),
                        base > 0 ? 100.0 * (stat(c.name, 3) / base - 1) : 0.0, old > 0 ? 100.0 * (stat(c.name, 3) / old - 1) : 0.0,
                        stat("nostore", 1) > 0 ? 100.0 * (stat(c.name, 1) / stat("nostore", 1) - 1) : 0.0);
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
