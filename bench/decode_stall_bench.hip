// Does pinning host memory stall decoding? Decode steps of the real engine timed one by one while background threads
// pin and unpin arenas, as the prefix cache's pools do when it grows or gives memory back. Pinning holds the
// process's memory-map lock (a probe thread doing small mmap/munmap calls waits 45-350 ms on a quiet box), and the
// rank threads that launch the steps map and fault memory too: a pool that pins at a bad moment shows up as a slow
// step. Production saw 1-4 s steps without any load running.
//   decode_stall_bench [--steps N=500] [--threads T=8] [--arena-mb M=256]
// Rows: no pinning (the baseline); hipHostMalloc with default flags (what the pools use); non-coherent flags; and
// each of those with a gate that lets one thread pin at a time (QW_POOL_SERIAL). Per row: median, p99 and worst step
// in ms, and how many arenas were pinned meanwhile.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

#define CK(x)                                                                           \
    do {                                                                                \
        hipError_t err_ = (x);                                                          \
        if (err_ != hipSuccess) fail(std::string(#x) + ": " + hipGetErrorString(err_)); \
    } while (0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Load {
    bool noncoherent = false, serial = false;
};

int main(int argc, char **argv) {
    try {
        int steps = 500, threads = 8;
        size_t arena_mb = 256;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
            else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
            else if (a == "--arena-mb" && i + 1 < argc) arena_mb = size_t(std::atoi(argv[++i]));
        }
        EngineOptions opt;
        opt.slot_tokens = {4096};
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;
        Engine e(opt);
        std::vector<int32_t> prompt(512);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = int32_t(100 + (i * 7919) % 30000);
        e.slot_reset(0);
        e.prefill(prompt);
        int32_t tok = prompt.back();
        auto argmax = [](const std::vector<float> &l) {
            return int32_t(std::max_element(l.begin(), l.end()) - l.begin());
        };
        for (int i = 0; i < 20; ++i) tok = argmax(e.step(tok));  // warm up
        int ndev = 0;
        CK(hipGetDeviceCount(&ndev));
        std::printf("decode steps (single stream, full logits copy), %d pinning threads of %zu MB arenas\n", threads, arena_mb);
        std::printf("%-34s %9s %9s %9s %9s %10s\n", "background", "median ms", "p99 ms", "worst ms", "steps>100ms", "arenas pinned");
        struct Row {
            const char *name;
            bool load;
            Load l;
        };
        const Row rows[] = {{"none", false, {}},
                            {"pinning, default flags", true, {false, false}},
                            {"pinning, non-coherent", true, {true, false}},
                            {"pinning, default, one at a time", true, {false, true}},
                            {"pinning, non-coherent, one at a time", true, {true, true}},
                            {"none (again)", false, {}}};
        for (const Row &row : rows) {
            std::atomic<bool> stop{false};
            std::atomic<int> pinned{0};
            std::mutex gate;
            std::vector<std::thread> th;
            if (row.load)
                for (int t = 0; t < threads; ++t)
                    th.emplace_back([&, t] {
                        CK(hipSetDevice(t % ndev));
                        while (!stop.load()) {
                            void *p = nullptr;
                            {
                                std::unique_lock<std::mutex> g(gate, std::defer_lock);
                                if (row.l.serial) g.lock();
                                CK(hipHostMalloc(&p, arena_mb << 20, row.l.noncoherent ? hipHostMallocNonCoherent : 0u));
                                ++pinned;
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(20));
                            {
                                std::unique_lock<std::mutex> g(gate, std::defer_lock);
                                if (row.l.serial) g.lock();
                                CK(hipHostFree(p));
                            }
                        }
                    });
            std::this_thread::sleep_for(std::chrono::milliseconds(row.load ? 300 : 0));
            std::vector<double> ms;
            ms.reserve(size_t(steps));
            for (int i = 0; i < steps; ++i) {
                const double t0 = now();
                tok = argmax(e.step(tok));
                ms.push_back((now() - t0) * 1e3);
                if (e.position() > 4000) {
                    e.slot_reset(0);
                    e.prefill(prompt);
                }
            }
            stop = true;
            for (auto &t : th) t.join();
            std::sort(ms.begin(), ms.end());
            const double p99 = ms[size_t(double(ms.size() - 1) * 0.99)];
            int slow = 0;
            for (double m : ms) slow += m > 100.0;
            std::printf("%-34s %9.1f %9.1f %9.1f %9d %10d\n", row.name, ms[ms.size() / 2], p99, ms.back(), slow, pinned.load());
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
