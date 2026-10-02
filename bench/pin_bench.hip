// How long it takes to pin host memory for the GPUs, the way the prefix cache's pools do it: arenas of a few
// hundred MB, one allocating thread per pool, every pool on its own device. The pools of one engine (a KV and a
// snapshot pool per rank) pin at the same time, which took 0.7-9 s per 256 MB arena on the production host
// (2026-10-02) where mmap + mlock of the same size takes 0.03-0.26 s.
//   pin_bench [arena_mb=256] [rounds=4]
// For each way to get pinned memory and each number of threads: per-arena latency and aggregate GB/s, and the
// latency of a small mmap + touch + munmap on another thread meanwhile (the probe): if pinning holds the process's
// memory-map lock, every thread that maps or faults memory waits for it, which stalls the engine's other threads.
// Then every way's memory is used from every device (host -> device -> host copy), to check it works there and at
// what speed.
#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/common.hpp"

using namespace qw;

#define CK(x)                                                                           \
    do {                                                                                \
        hipError_t err_ = (x);                                                          \
        if (err_ != hipSuccess) fail(std::string(#x) + ": " + hipGetErrorString(err_)); \
    } while (0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23  // Linux 5.14: fault the range in, writable
#endif

constexpr size_t HUGE_PAGE = size_t(2) << 20;

enum class Way { HostMalloc, HostMallocNonCoherent, Map4K, MapHuge, MapHugePopulate };
static const char *name_of(Way w) {
    switch (w) {
    case Way::HostMalloc: return "hipHostMalloc";
    case Way::HostMallocNonCoherent: return "hipHostMalloc NonCoherent";
    case Way::Map4K: return "mmap 4 KB pages + hipHostRegister";
    case Way::MapHuge: return "mmap MADV_HUGEPAGE + hipHostRegister";
    case Way::MapHugePopulate: return "mmap MADV_HUGEPAGE + MAP_POPULATE + register";
    }
    return "";
}

struct Mem {
    void *p = nullptr;
    size_t map_bytes = 0;  // mapped ways
};

static Mem take(Way w, size_t bytes) {
    Mem m;
    if (w == Way::HostMalloc || w == Way::HostMallocNonCoherent) {
        CK(hipHostMalloc(&m.p, bytes, w == Way::HostMalloc ? 0u : hipHostMallocNonCoherent));
        return m;
    }
    const size_t len = (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
    const int extra = w == Way::MapHugePopulate ? MAP_POPULATE : 0;
    void *raw = mmap(nullptr, len + HUGE_PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) fail("mmap");
    auto *base = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(raw) + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1));
    if (base > raw) munmap(raw, size_t(base - static_cast<uint8_t *>(raw)));
    auto *end = base + len, *raw_end = static_cast<uint8_t *>(raw) + len + HUGE_PAGE;
    if (raw_end > end) munmap(end, size_t(raw_end - end));
    if (w != Way::Map4K) madvise(base, len, MADV_HUGEPAGE);
    if (extra) madvise(base, len, MADV_POPULATE_WRITE);
    CK(hipHostRegister(base, len, hipHostRegisterPortable));
    m.p = base;
    m.map_bytes = len;
    return m;
}

static void give(Way w, Mem &m) {
    if (!m.p) return;
    if (m.map_bytes) {
        (void)hipHostUnregister(m.p);
        munmap(m.p, m.map_bytes);
    } else {
        (void)hipHostFree(m.p);
    }
    m = Mem{};
}

int main(int argc, char **argv) {
    try {
        const size_t mb = argc > 1 ? size_t(std::atoi(argv[1])) : 256;
        const int rounds = argc > 2 ? std::atoi(argv[2]) : 4;
        int ndev = 0;
        CK(hipGetDeviceCount(&ndev));
        const size_t bytes = mb << 20;
        // the first host allocation of a process initializes the runtime's host mappings (0.7 s on the 4-GPU box):
        // pay it before the timings (the engine has paid it long before its first arena)
        for (int d = 0; d < ndev; ++d) {
            CK(hipSetDevice(d));
            void *w = nullptr;
            CK(hipHostMalloc(&w, size_t(2) << 20, 0));
            CK(hipHostFree(w));
        }
        std::printf("%d device(s); arenas of %zu MB, %d rounds per thread\n", ndev, mb, rounds);
        std::printf("%-46s %8s %10s %10s %10s %10s | %9s %9s %9s\n", "way", "threads", "mean s", "max s", "wall s",
                    "GB/s total", "probe max", "probe p99", "probe mean");
        const Way ways[] = {Way::HostMalloc, Way::HostMallocNonCoherent, Way::Map4K, Way::MapHuge, Way::MapHugePopulate};
        std::vector<int> thread_counts{1};
        if (ndev > 1) thread_counts.push_back(ndev);
        thread_counts.push_back(2 * ndev);
        for (Way w : ways)
            for (int threads : thread_counts) {
                std::vector<std::vector<double>> lat;
                lat.resize(size_t(threads));
                // the probe: how long a small map + first touch + unmap takes while the others pin
                std::atomic<bool> stop{false};
                std::vector<double> probe;
                std::thread prober([&] {
                    while (!stop.load()) {
                        const double a = now();
                        void *m = mmap(nullptr, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                        if (m != MAP_FAILED) {
                            *static_cast<volatile char *>(m) = 1;
                            munmap(m, 65536);
                        }
                        probe.push_back(now() - a);
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                });
                const double t0 = now();
                std::vector<std::thread> th;
                for (int t = 0; t < threads; ++t)
                    th.emplace_back([&, t] {
                        CK(hipSetDevice(t % ndev));
                        for (int i = 0; i < rounds; ++i) {
                            const double a = now();
                            Mem m = take(w, bytes);
                            lat[size_t(t)].push_back(now() - a);
                            give(w, m);
                        }
                    });
                for (auto &x : th) x.join();
                stop = true;
                prober.join();
                std::sort(probe.begin(), probe.end());
                const double probe_max = probe.empty() ? 0 : probe.back();
                const double probe_p99 = probe.empty() ? 0 : probe[size_t(double(probe.size() - 1) * 0.99)];
                double probe_sum = 0;
                for (double d : probe) probe_sum += d;
                const double wall = now() - t0;
                double sum = 0, mx = 0;
                size_t n = 0;
                for (auto &v : lat)
                    for (double d : v) sum += d, mx = std::max(mx, d), ++n;
                // wall includes the unpinning too: report the pinning throughput from the latencies instead
                double pin_wall = 0;
                for (auto &v : lat) {
                    double s = 0;
                    for (double d : v) s += d;
                    pin_wall = std::max(pin_wall, s);
                }
                std::printf("%-46s %8d %10.3f %10.3f %10.3f %10.2f | %7.1f ms %7.1f ms %7.3f ms\n", name_of(w), threads,
                            sum / double(n), mx, wall, double(n) * double(bytes) / 1e9 / pin_wall, probe_max * 1e3,
                            probe_p99 * 1e3, probe.empty() ? 0.0 : probe_sum / double(probe.size()) * 1e3);
                std::fflush(stdout);
            }

        std::printf("\neach way's memory: CPU write/read GB/s, then from every device 64 MB host -> device / device -> host GB/s\n");
        for (Way w : ways) {
            std::printf("%-46s", name_of(w));
            Mem m;
            {
                CK(hipSetDevice(0));
                m = take(w, size_t(64) << 20);
            }
            auto *h = static_cast<uint8_t *>(m.p);
            {  // the CPU's view of the memory: a write, then a read (the disk writer and loader threads touch it)
                std::vector<uint8_t> scratch(size_t(64) << 20, 1);
                const double a = now();
                std::memcpy(h, scratch.data(), scratch.size());
                const double b = now();
                std::memcpy(scratch.data(), h, scratch.size());
                const double c = now();
                std::printf("  CPU w/r %5.1f/%5.1f |", 64.0 * 1.048576 / 1e3 / (b - a), 64.0 * 1.048576 / 1e3 / (c - b));
            }
            for (int d = 0; d < ndev; ++d) {
                CK(hipSetDevice(d));
                uint8_t *dev = nullptr;
                CK(hipMalloc(&dev, size_t(64) << 20));
                for (size_t i = 0; i < (size_t(64) << 20); i += 4096) h[i] = uint8_t(i / 4096 + 3 * d);
                CK(hipDeviceSynchronize());
                const double a = now();
                CK(hipMemcpy(dev, h, size_t(64) << 20, hipMemcpyHostToDevice));
                const double b = now();
                std::memset(h, 0, size_t(64) << 20);
                CK(hipMemcpy(h, dev, size_t(64) << 20, hipMemcpyDeviceToHost));
                const double c = now();
                bool ok = true;
                for (size_t i = 0; i < (size_t(64) << 20); i += 4096) ok = ok && h[i] == uint8_t(i / 4096 + 3 * d);
                std::printf("  dev%d %5.1f/%5.1f%s", d, 64.0 * 1.048576 / 1e3 / (b - a), 64.0 * 1.048576 / 1e3 / (c - b),
                            ok ? "" : " WRONG");
                CK(hipFree(dev));
            }
            std::printf("\n");
            give(w, m);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
