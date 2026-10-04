// What reading the KV spill costs on the link, before any engine work: kernels shaped like the decode attention
// (33 blocks per row, 8 waves, one 16-byte load per lane of a 512-byte K row and a V row, 4 consecutive tokens per
// selected group, 512 groups per row) read scattered groups of a SpillPool buffer, i.e. of the pinned, NUMA-placed
// host memory the engine's spill uses.
//   spill_gather_bench [--tokens N=49152] [--rows R=1,4] [--reps 40] [--wide]
// Prints, per GPU alone, per KV pair reading the same buffers, and all four GPUs together: GB/s over the 12 QSA
// layers, split by whether the layer's pages were placed on the GPU's own NUMA node or on its pair's. QW_SPILL_NUMA=0
// leaves placement to the kernel (QW_SPILL_NC=1 maps the host memory non-coherent, i.e. cacheable). --wide adds the
// same bytes over 8x more blocks (the ceiling of the link). Also checks, in both directions, that the device and the
// host see each other's writes through the mapped buffers.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/spill.hpp"

using namespace qw;
using qw::cfg::RANKS;

__global__ __launch_bounds__(256) void gather_k(const uint4 *K, const uint4 *V, const int *list, int n, int chunk,
                                                unsigned *sink) {
    const int row = blockIdx.y, lane = threadIdx.x % 32, w = threadIdx.x / 32;
    const int base = blockIdx.x * chunk;
    unsigned acc = 0;
    for (int i = base + w; i < min(n, base + chunk); i += 8) {
        const int tok = list[size_t(row) * n + i];
        const uint4 k = K[size_t(tok) * 32 + lane], v = V[size_t(tok) * 32 + lane];
        acc ^= k.x ^ k.y ^ k.z ^ k.w ^ v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads
}

__global__ void fill_k(uint32_t *p, size_t n, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        p[i] = uint32_t(i) * 2654435761u + seed;
}
__global__ void verify_k(const uint32_t *p, size_t n, unsigned *bad) {  // the host wrote p[i] = i ^ 0xabcd
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        if (p[i] != (uint32_t(i) ^ 0xabcdu)) atomicAdd(bad, 1u);
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Result {
    double local = 0, remote = 0;  // GB/s, mean over the layers placed on the GPU's node / the pair's node
};

int main(int argc, char **argv) {
    try {
        int tokens = 49152, reps = 40;
        bool wide = false;
        std::vector<int> rows_list = {1, 4};
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
            else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
            else if (a == "--wide") wide = true;
            else if (a == "--rows" && i + 1 < argc) {
                rows_list.clear();
                std::stringstream ss(argv[++i]);
                for (std::string t; std::getline(ss, t, ',');) rows_list.push_back(std::atoi(t.c_str()));
            }
        }
        constexpr int L = 12;  // QSA layers read per decode row
        const std::array<int, RANKS> devs{0, 1, 2, 3};
        for (int r = 0; r < RANKS; ++r) log("GPU %d: NUMA node %d", r, device_numa_node(devs[size_t(r)]));
        SpillPool pool({tokens}, L, devs);

        // ---- the mapped buffers are visible both ways
        for (int r = 0; r < RANKS; ++r) {
            CK(hipSetDevice(devs[size_t(r)]));
            const SpillPool::Buf &b = pool.buf(0, r, 0);
            const size_t words = size_t(tokens) * 512 / 4;
            unsigned *bad;
            CK(hipMalloc(&bad, 4));
            CK(hipMemset(bad, 0, 4));
            fill_k<<<64, 256>>>(reinterpret_cast<uint32_t *>(b.K), words, 100u + uint32_t(r / 2));
            CK(hipDeviceSynchronize());  // device wrote, host reads
            size_t host_bad = 0;
            for (size_t i = 0; i < words; ++i)
                host_bad += reinterpret_cast<uint32_t *>(b.K)[i] != uint32_t(i) * 2654435761u + 100u + uint32_t(r / 2);
            for (size_t i = 0; i < words; ++i) reinterpret_cast<uint32_t *>(b.V)[i] = uint32_t(i) ^ 0xabcdu;  // host wrote
            verify_k<<<64, 256>>>(reinterpret_cast<uint32_t *>(b.V), words, bad);
            CK(hipDeviceSynchronize());
            unsigned dev_bad = 0;
            CK(hipMemcpy(&dev_bad, bad, 4, hipMemcpyDeviceToHost));
            CK(hipFree(bad));
            std::printf("GPU %d: host read of the device's writes: %zu bad words; device read of the host's: %u bad\n", r,
                        host_bad, dev_bad);
        }

        const int groups = tokens / 4;
        std::vector<int *> lists(RANKS, nullptr);
        std::vector<unsigned *> sinks(RANKS, nullptr);
        const int maxrows = *std::max_element(rows_list.begin(), rows_list.end());
        for (int r = 0; r < RANKS; ++r) {
            CK(hipSetDevice(devs[size_t(r)]));
            std::vector<int> h(size_t(maxrows) * 2048);
            uint32_t x = 777u + uint32_t(r);
            for (int row = 0; row < maxrows; ++row) {
                std::vector<int> g(512);
                for (int &v : g) {
                    x = x * 1664525u + 1013904223u;
                    v = int((x >> 8) % uint32_t(groups));
                }
                std::sort(g.begin(), g.end());
                for (int j = 0; j < 512; ++j)
                    for (int q = 0; q < 4; ++q) h[size_t(row) * 2048 + size_t(j) * 4 + size_t(q)] = g[size_t(j)] * 4 + q;
            }
            lists[size_t(r)] = upload(h);
            CK(hipMalloc(&sinks[size_t(r)], 4));
        }

        // one GPU's timing of `reps` launches over each layer; fills res
        const auto run_gpu = [&](int r, int rows, int chunk, int nblk, Result &res, std::atomic<int> *gate, int total) {
            CK(hipSetDevice(devs[size_t(r)]));
            hipStream_t s;
            CK(hipStreamCreate(&s));
            hipEvent_t e0, e1;
            CK(hipEventCreate(&e0));
            CK(hipEventCreate(&e1));
            const auto launch = [&](int l) {
                const SpillPool::Buf &b = pool.buf(0, r, l);
                gather_k<<<dim3(nblk, rows), 256, 0, s>>>(reinterpret_cast<const uint4 *>(b.K),
                                                          reinterpret_cast<const uint4 *>(b.V), lists[size_t(r)], 2048,
                                                          chunk, sinks[size_t(r)]);
            };
            for (int l = 0; l < L; ++l) launch(l);  // warm
            CK(hipStreamSynchronize(s));
            double loc = 0, rem = 0;
            int nl = 0, nr = 0;
            for (int l = 0; l < L; ++l) {
                if (gate) {  // all GPUs start each layer together
                    gate->fetch_add(1);
                    while (gate->load() < total * (l + 1)) {}
                }
                CK(hipEventRecord(e0, s));
                for (int i = 0; i < reps; ++i) launch(l);
                CK(hipEventRecord(e1, s));
                CK(hipEventSynchronize(e1));
                float ms = 0;
                CK(hipEventElapsedTime(&ms, e0, e1));
                const double bytes = double(reps) * rows * 2048 * 1024;
                const double gbs = bytes / (ms * 1e-3) / 1e9;
                // the layer's pages were placed near rank (group * 2 + l % 2)
                const bool local = (r / 2) * 2 + l % 2 == r;
                (local ? loc : rem) += gbs;
                ++(local ? nl : nr);
            }
            res.local = nl ? loc / nl : 0;
            res.remote = nr ? rem / nr : 0;
            CK(hipStreamDestroy(s));
        };

        std::printf("\n%-8s %-6s | %-28s\n", "rows", "mode", "GB/s per GPU: local-node layers / pair-node layers");
        for (int rows : rows_list) {
            for (int variant = 0; variant < (wide ? 2 : 1); ++variant) {
                const int chunk = variant ? 8 : 64, nblk = 2048 / chunk;
                const char *mode = variant ? "wide" : "attend";
                for (int r = 0; r < RANKS; ++r) {  // one GPU alone
                    Result res;
                    run_gpu(r, rows, chunk, nblk, res, nullptr, 1);
                    std::printf("%-8d %-6s | GPU %d alone        %6.1f / %6.1f\n", rows, mode, r, res.local, res.remote);
                }
                for (int pr = 0; pr < 2; ++pr) {  // the two GPUs of a KV pair together, reading the same buffers
                    Result res[2];
                    std::atomic<int> gate{0};
                    std::thread t0([&] { run_gpu(pr * 2, rows, chunk, nblk, res[0], &gate, 2); });
                    std::thread t1([&] { run_gpu(pr * 2 + 1, rows, chunk, nblk, res[1], &gate, 2); });
                    t0.join();
                    t1.join();
                    std::printf("%-8d %-6s | pair %d together    %6.1f / %6.1f   and  %6.1f / %6.1f\n", rows, mode, pr,
                                res[0].local, res[0].remote, res[1].local, res[1].remote);
                }
                {  // all four
                    Result res[RANKS];
                    std::atomic<int> gate{0};
                    std::vector<std::thread> th;
                    for (int r = 0; r < RANKS; ++r) th.emplace_back([&, r] { run_gpu(r, rows, chunk, nblk, res[r], &gate, RANKS); });
                    for (auto &t : th) t.join();
                    double l = 0, m = 0;
                    for (int r = 0; r < RANKS; ++r) l += res[r].local, m += res[r].remote;
                    std::printf("%-8d %-6s | all 4 together     %6.1f / %6.1f   (mean per GPU)\n", rows, mode, l / RANKS, m / RANKS);
                }
            }
        }
        std::printf("\nper decode step and rank at these rates, the 12 layers read %.0f MiB per row\n", 12 * 2.0);
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "spill_gather_bench: %s\n", ex.what());
        return 1;
    }
}
