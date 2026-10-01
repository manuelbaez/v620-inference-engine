// Background disk loads of the block store (Session::prefetch): after a
// restart the RAM tier is empty and a prompt's entries come from the disk tier
// on loader threads.
//   1. a hit that needs nothing from disk is not held up by a load running for
//      another prompt (it used to read as "loading" until that load finished);
//   2. the restored state is the one that was saved (identical logits), with
//      one loader thread and with several (QW_LOAD_THREADS).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

using namespace qw;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<int32_t> text(int n, int seed) {  // deterministic token soup of common ids
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = uint32_t(seed) * 2654435761u + 1;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}

// Polls Session::prefetch until the prompt's entries are in RAM; returns the seconds it took.
static double drain(Session &s, int slot, const std::vector<int32_t> &p) {
    const double t0 = now();
    while (s.prefetch(slot, p)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return now() - t0;
}

int main() {
    try {
        EngineOptions opt;
        opt.slot_tokens = {8192, 8192};
        opt.warmup = false;
        Engine e(opt);
        const std::string dir = "/tmp/qw_disk_load_test";
        std::system(("rm -rf " + dir).c_str());
        setenv("QW_HOST_CACHE_GB", "8", 1);
        setenv("QW_DISK_CACHE_DIR", dir.c_str(), 1);
        int fails = 0;
        auto check = [&](bool ok, const char *what) {
            if (!ok) std::printf("FAIL: %s\n", what);
            fails += !ok;
        };
        const auto A = text(6000, 1), B = text(3000, 2), C = text(3000, 3), D = text(3000, 4);

        // a session fills the store and the disk tier; A's logits are the reference
        std::vector<float> ref;
        {
            Session s(e);
            for (const auto *p : {&A, &B, &C, &D}) {
                const int slot = s.acquire(*p, 16);
                QW_CHECK(slot >= 0, "no slot");
                s.set_prompt(slot, *p);
                if (p == &A) ref = e.logits();
                s.release(slot);
            }
            s.flush_disk();
        }

        // restart with n loader threads: the RAM tier is empty, the disk tier is indexed
        auto restart = [&](int threads) {
            setenv("QW_LOAD_THREADS", std::to_string(threads).c_str(), 1);
            Session s(e);
            // B comes into RAM first, and no slot holds it afterwards: only the store does
            int slot = s.acquire(B, 16);
            drain(s, slot, B);
            check(s.set_prompt(slot, B) == int64_t(B.size()), "B restored whole");
            s.release(slot);
            s.set_prompt(slot, {1});

            // A needs the disk: its load starts. B needs nothing from disk: not held up by it.
            const int slot_a = s.acquire(A, 16);
            const double t0 = now();
            const bool a_loading = s.prefetch(slot_a, A);
            const int slot_b = s.acquire(B, 16);
            const bool b_held = s.prefetch(slot_b, B);
            const bool a_still = s.prefetch(slot_a, A);
            check(a_loading, "A's entries are on disk only: its load starts");
            check(!b_held, "B (all in RAM) is not held up by A's load");
            std::printf("%d thread(s): A loading %s, B %s, A still loading afterwards: %s%s\n", threads,
                        a_loading ? "yes" : "NO", b_held ? "HELD UP" : "not held up", a_still ? "yes" : "no",
                        a_still ? "" : " (inconclusive: the load finished first)");
            const double waited = drain(s, slot_a, A);
            const double load_s = now() - t0;
            const int64_t reused = s.set_prompt(slot_a, A);
            std::vector<float> got = e.logits();
            std::printf("%d thread(s): A's entries loaded in %.3f s (%.3f s after B's check), reused %lld of %zu\n",
                        threads, load_s, waited, (long long)reused, A.size());
            check(reused == int64_t(A.size()), "A restored whole");
            s.release(slot_a);
            s.release(slot_b);
            return got;
        };
        const std::vector<float> one = restart(1), four = restart(4);
        check(one == ref, "restored logits equal the saved ones (1 thread)");
        check(four == ref, "restored logits equal the saved ones (4 threads)");
        std::system(("rm -rf " + dir).c_str());
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
