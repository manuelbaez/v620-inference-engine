// Pinned pool alone (one HIP device): units are distinct and keep what is
// written to them while other threads get and put, emptied arenas are given
// back by the pool's own thread (not inside put()), and the pool grows again
// after that. Run for both kinds of arena (hipHostMalloc, and huge-page
// mappings registered with HIP), whose units must also work as DMA targets.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/pinned_pool.hpp"

using namespace qw;

constexpr size_t UNIT = 4u << 20;  // production: ~5 MB units, 48 per arena
constexpr int PER_ARENA = 16;

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static int run(const char *label, PoolOptions opt) {
    std::printf("--- %s\n", label);
    int fails = 0;
    try {
        auto check = [&](bool ok, const char *what) {
            if (!ok) std::printf("FAIL: %s\n", what);
            fails += !ok;
        };
        PinnedPool pool(UNIT, PER_ARENA, 0, 2, opt);
        const size_t arena = UNIT * PER_ARENA;

        // 1. 100 units are distinct and keep their contents
        std::vector<uint8_t *> held;
        for (int i = 0; i < 100; ++i) {
            held.push_back(pool.get());
            std::fill(held.back(), held.back() + UNIT, uint8_t(i));
        }
        bool intact = true;
        for (int i = 0; i < 100; ++i)
            intact = intact && held[size_t(i)][0] == uint8_t(i) && held[size_t(i)][UNIT - 1] == uint8_t(i);
        check(intact, "units keep what was written (no overlap)");
        check(pool.allocated_bytes() >= 100 * UNIT, "arenas cover the units taken");
        const size_t peak = pool.allocated_bytes();

        // 2. putting everything back: put() must not wait for the arenas to be unpinned, and the
        // pool thread gives all but a spare back
        double worst_put = 0;
        for (uint8_t *p : held) {
            const auto t0 = std::chrono::steady_clock::now();
            pool.put(p);
            worst_put = std::max(worst_put, ms_since(t0));
        }
        held.clear();
        for (int i = 0; i < 500 && pool.allocated_bytes() > 3 * arena; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::printf("peak %zu MB pinned, %zu MB after putting all back; slowest put() %.2f ms\n", peak >> 20,
                    pool.allocated_bytes() >> 20, worst_put);
        check(pool.allocated_bytes() <= 3 * arena, "emptied arenas are given back (a spare stays)");

        // 2b. units are DMA targets: a round trip through a device buffer keeps the bytes
        {
            uint8_t *dev = nullptr;
            CK(hipMalloc(reinterpret_cast<void **>(&dev), UNIT));
            uint8_t *u = pool.get();
            for (size_t i = 0; i < UNIT; ++i) u[i] = uint8_t(i * 131 + 7);
            CK(hipMemcpy(dev, u, UNIT, hipMemcpyHostToDevice));
            std::fill(u, u + UNIT, uint8_t(0));
            CK(hipMemcpy(u, dev, UNIT, hipMemcpyDeviceToHost));
            bool same = true;
            for (size_t i = 0; i < UNIT; ++i) same = same && u[i] == uint8_t(i * 131 + 7);
            check(same, "a unit survives a host -> device -> host copy");
            pool.put(u);
            CK(hipFree(dev));
        }
        {
            const PoolStats st = pool.stats();
            std::printf("stats: %llu arenas pinned in %.3f s, %llu waits (%.3f s)\n", (unsigned long long)st.pins,
                        st.pin_s, (unsigned long long)st.waits, st.wait_s);
            check(st.pins > 0, "the pool counts the arenas it pinned");
        }

        // 2c. a standing reserve: pinned a while after start, used up by a burst without any pinning meanwhile, and
        // topped up again only after a quiet period
        {
            const size_t want_units = 40;
            PinnedPool p2(UNIT, PER_ARENA, 0, want_units, opt);
            for (int i = 0; i < 600 && p2.allocated_bytes() < want_units * UNIT; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            check(p2.allocated_bytes() >= want_units * UNIT, "the standing reserve is pinned after start");
            const uint64_t pins0 = p2.stats().pins;
            std::vector<uint8_t *> burst;
            for (size_t i = 0; i < want_units; ++i) burst.push_back(p2.get());
            check(p2.stats().pins == pins0, "a burst served from the reserve pins nothing meanwhile");
            check(p2.stats().waits == 0, "... and waits for nothing");
            for (int i = 0; i < 1000 && p2.allocated_bytes() < 2 * want_units * UNIT; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            check(p2.allocated_bytes() >= 2 * want_units * UNIT, "the reserve is topped up after the burst, once the pool is quiet");
            for (uint8_t *u : burst) p2.put(u);
        }

        // 3. the pool grows again
        uint8_t *again = pool.get();
        std::fill(again, again + UNIT, uint8_t(7));
        pool.put(again);

        // 4. stress: 8 threads, units checked for corruption by other threads' writes
        std::atomic<int> corrupt{0};
        std::vector<std::thread> th;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < 8; ++t)
            th.emplace_back([&, t] {
                std::mt19937 rng(unsigned(t) + 1);
                std::vector<std::pair<uint8_t *, uint8_t>> mine;
                for (int it = 0; it < 1500; ++it) {
                    if (mine.size() < 24 && (mine.empty() || rng() % 3 != 0)) {
                        uint8_t *p = pool.get();
                        const uint8_t tag = uint8_t(t * 29 + it);
                        p[0] = p[UNIT / 2] = p[UNIT - 1] = tag;
                        mine.push_back({p, tag});
                    } else {
                        const size_t k = rng() % mine.size();
                        auto [p, tag] = mine[k];
                        if (p[0] != tag || p[UNIT / 2] != tag || p[UNIT - 1] != tag) ++corrupt;
                        pool.put(p);
                        mine.erase(mine.begin() + ptrdiff_t(k));
                    }
                }
                for (auto &[p, tag] : mine) {
                    if (p[0] != tag || p[UNIT - 1] != tag) ++corrupt;
                    pool.put(p);
                }
            });
        for (auto &x : th) x.join();
        std::printf("stress: 8 threads x 1500 operations in %.2f s, %zu MB pinned at the end\n", ms_since(t0) / 1e3,
                    pool.allocated_bytes() >> 20);
        check(corrupt == 0, "no unit was overwritten by another thread");
        for (int i = 0; i < 500 && pool.allocated_bytes() > 3 * arena; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        check(pool.allocated_bytes() <= 3 * arena, "everything but a spare is given back after the stress");
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return fails;
}

int main() {
    // a deadlock must end the test, not hang the box: abort() under a debugger shows every thread
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(std::getenv("QW_TEST_TIMEOUT") ? std::atoi(std::getenv("QW_TEST_TIMEOUT")) : 120));
        std::fprintf(stderr, "test_pinned_pool: timed out (a deadlock?)\n");
        std::abort();
    }).detach();
    int fails = 0;
    for (auto kind : {PoolOptions::Arena::HostMalloc, PoolOptions::Arena::NonCoherent, PoolOptions::Arena::Huge}) {
        PoolOptions o;
        o.arena = kind;
        fails += run(PoolOptions::name(kind), o);
    }
    std::printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}
