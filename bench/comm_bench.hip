// How fast the collectives run at the prefill shapes, and how fast the PCIe fabric can go for the same traffic.
//   comm_bench engine [iters]  the engine's Comm at a prefill micro-batch's shapes (512, 1024, 2048 rows): per
//                              collective, ms and the GB/s each rank sends
//   comm_bench raw             plain uint4 push kernels between the GPUs: one pair alone, a rotation step (every GPU
//                              feeds one peer, nobody is fed twice), and all peers at once; message sizes and block
//                              counts swept. This is the ceiling the engine's pushes can be measured against.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "comm/comm.hpp"
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

// ---------------------------------------------------------------- the engine's collectives

// a reusable barrier for the rank threads (C++17 has no std::barrier)
struct Barrier {
    std::mutex m;
    std::condition_variable cv;
    int n, waiting = 0, gen = 0;
    explicit Barrier(int count) : n(count) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        const int g = gen;
        if (++waiting == n) {
            waiting = 0;
            ++gen;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return gen != g; });
        }
    }
};

static void engine_mode(int iters) {
    Comm comm({0, 1, 2, 3}, 32u << 20);
    constexpr int SHARD = cfg::SH;  // 640
    constexpr int H = cfg::H;       // 2560
    constexpr int DPAD = 328;
    struct Op {
        const char *name;
        double bytes_per_row;  // sent per rank per row (to the 3 peers)
    };
    const Op ops[] = {{"all-gather bin (fp16 640 -> 2560)", 3.0 * SHARD * 2},
                      {"reduce-scatter fp16 (2560 -> 640)", 3.0 * SHARD * 2},
                      {"all-reduce HC down fp16 (328)", 3.0 * DPAD * 2},
                      {"all-reduce ssq fp32 (4)", 3.0 * 4 * 4}};
    const int row_sets[] = {512, 1024, 2048};
    struct Variant {
        const char *name;
        Comm::Tuning t;
    };
    const Variant variants[] = {{"plain", {false}}, {"push2d", {true}}};
    const size_t nops = std::size(ops), nrows = std::size(row_sets), nvar = std::size(variants);
    // ms[variant][rank][rows index * nops + op]
    std::vector<std::vector<std::vector<double>>> ms(nvar, std::vector<std::vector<double>>(RANKS, std::vector<double>(nops * nrows, 0.0)));
    Barrier bar(RANKS);
    auto rank_fn = [&](int r) {
        CK(hipSetDevice(comm.device(r)));
        hipStream_t s;
        CK(hipStreamCreate(&s));
        const size_t maxrows = 2048;
        uint16_t *h16;
        float *f32, *dst;
        CK(hipMalloc(&h16, maxrows * H * 2));
        CK(hipMalloc(&f32, maxrows * 1300 * 4));
        CK(hipMalloc(&dst, maxrows * H * 4));
        CK(hipMemset(h16, 0x3c, maxrows * H * 2));
        CK(hipMemset(f32, 0, maxrows * 1300 * 4));
        uint32_t k = 0;
        for (size_t v = 0; v < nvar; ++v) {
            bar.wait();  // every rank is done with the previous variant
            if (r == 0) comm.set_tuning(variants[v].t);
            bar.wait();
            for (size_t ri = 0; ri < nrows; ++ri)
                for (size_t o = 0; o < nops; ++o) {
                    const int rows = row_sets[ri];
                    auto issue = [&](uint32_t kk) {
                        switch (o) {
                        case 0: comm.allgather(r, h16, dst, size_t(SHARD) * 2, kk, s, rows); break;
                        case 1: comm.reduce_scatter(r, h16, dst, SHARD, kk, s, rows, true); break;
                        case 2: comm.allreduce(r, h16, dst, DPAD, kk, s, rows, true); break;
                        default: comm.allreduce(r, f32, dst, 4, kk, s, rows, false); break;
                        }
                    };
                    for (int i = 0; i < 3; ++i) issue(++k);  // warm up
                    CK(hipStreamSynchronize(s));
                    const double t0 = now();
                    for (int i = 0; i < iters; ++i) issue(++k);
                    CK(hipStreamSynchronize(s));
                    ms[v][size_t(r)][ri * nops + o] = (now() - t0) / iters * 1e3;
                }
        }
        CK(hipFree(h16));
        CK(hipFree(f32));
        CK(hipFree(dst));
        CK(hipStreamDestroy(s));
    };
    std::vector<std::thread> th;
    for (int r = 0; r < RANKS; ++r) th.emplace_back(rank_fn, r);
    for (auto &t : th) t.join();
    std::printf("ms per collective (slowest rank), and the GB/s each rank sends; variants as in Comm::Tuning\n");
    std::printf("%-36s %5s %7s", "collective (per micro-batch)", "rows", "MB sent");
    for (const Variant &v : variants) std::printf(" %9s", v.name);
    std::printf("   GB/s plain -> best\n");
    for (size_t ri = 0; ri < nrows; ++ri)
        for (size_t o = 0; o < nops; ++o) {
            std::vector<double> worst(nvar, 0.0);
            for (size_t v = 0; v < nvar; ++v)
                for (int r = 0; r < RANKS; ++r) worst[v] = std::max(worst[v], ms[v][size_t(r)][ri * nops + o]);
            const double mb = ops[o].bytes_per_row * row_sets[ri] / 1e6;
            std::printf("%-36s %5d %7.2f", ops[o].name, row_sets[ri], mb);
            for (size_t v = 0; v < nvar; ++v) std::printf(" %9.3f", worst[v]);
            std::printf("   %5.2f -> %5.2f\n", mb / worst[0], mb / *std::min_element(worst.begin(), worst.end()));
        }
}

// ---------------------------------------------------------------- raw pushes

__global__ void push1(const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n16) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n16; i += size_t(gridDim.x) * blockDim.x) dst[i] = src[i];
}

// every block serves one destination (block % 3), the blocks of a destination stride over its payload
__global__ void push3(const uint4 *__restrict__ src, uint4 *d0, uint4 *d1, uint4 *d2, size_t n16) {
    uint4 *dst = blockIdx.x % 3 == 0 ? d0 : blockIdx.x % 3 == 1 ? d1 : d2;
    const size_t nb = gridDim.x / 3, lb = blockIdx.x / 3;
    if (lb >= nb) return;
    for (size_t i = lb * size_t(blockDim.x) + threadIdx.x; i < n16; i += nb * blockDim.x) dst[i] = src[i];
}

static void raw_mode() {
    struct Gpu {
        uint8_t *src = nullptr, *recv[RANKS] = {};
        hipStream_t s;
        hipEvent_t e0, e1;
    } g[RANKS];
    const size_t MAXB = size_t(16) << 20;
    for (int r = 0; r < RANKS; ++r) {
        CK(hipSetDevice(r));
        for (int q = 0; q < RANKS; ++q)
            if (q != r) {
                const hipError_t e = hipDeviceEnablePeerAccess(q, 0);
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) fail("peer access");
            }
        (void)hipGetLastError();
        CK(hipMalloc(&g[r].src, MAXB));
        CK(hipMemset(g[r].src, 1, MAXB));
        CK(hipStreamCreate(&g[r].s));
        CK(hipEventCreate(&g[r].e0));
        CK(hipEventCreate(&g[r].e1));
    }
    for (int q = 0; q < RANKS; ++q) {  // g[r].recv[q]: the buffer on device q that rank r writes into
        CK(hipSetDevice(q));
        for (int r = 0; r < RANKS; ++r) {
            if (r == q) continue;
            CK(hipMalloc(&g[r].recv[q], MAXB));
        }
    }
    enum Mode { PAIR, ROT, ALL3 };
    // per mode: which ranks send, and to whom. PAIR: only (a -> b). ROT step k: every rank sends to (r + k) % 4.
    // ALL3: every rank sends to its 3 peers at once (one kernel, blocks split by destination).
    auto run = [&](Mode mode, int a, int b, size_t bytes, int blocks, int reps) {
        std::vector<double> ms(RANKS, 0.0);
        std::atomic<int> arrived{0};
        auto go = [&](int r) {
            CK(hipSetDevice(r));
            const bool active = mode != PAIR || r == a;
            auto launch = [&]() {
                if (!active) return;
                const size_t n16 = bytes / 16;
                if (mode == ALL3) {
                    uint8_t *d[3];
                    for (int j = 0, q = 0; q < RANKS; ++q)
                        if (q != r) d[j++] = g[r].recv[q];
                    push3<<<blocks / 3 * 3, 256, 0, g[r].s>>>(reinterpret_cast<const uint4 *>(g[r].src),
                                                              reinterpret_cast<uint4 *>(d[0]), reinterpret_cast<uint4 *>(d[1]),
                                                              reinterpret_cast<uint4 *>(d[2]), n16);
                } else {
                    const int dest = mode == PAIR ? b : (r + a) % RANKS;  // ROT: a is the step
                    push1<<<blocks, 256, 0, g[r].s>>>(reinterpret_cast<const uint4 *>(g[r].src),
                                                      reinterpret_cast<uint4 *>(g[r].recv[dest]), n16);
                }
            };
            launch();  // warm up
            CK(hipStreamSynchronize(g[r].s));
            double best = 1e30;
            for (int i = 0; i < reps; ++i) {
                // all ranks start together: the kernels run for ms, the start skew is microseconds
                const int target = (i + 1) * RANKS + 0;
                arrived.fetch_add(1);
                while (arrived.load() < target) std::this_thread::yield();
                if (active) CK(hipEventRecord(g[r].e0, g[r].s));
                launch();
                if (active) CK(hipEventRecord(g[r].e1, g[r].s));
                CK(hipStreamSynchronize(g[r].s));
                if (active) {
                    float t;
                    CK(hipEventElapsedTime(&t, g[r].e0, g[r].e1));
                    best = std::min(best, double(t));
                }
            }
            ms[size_t(r)] = active ? best : 0.0;
        };
        std::vector<std::thread> th;
        for (int r = 0; r < RANKS; ++r) th.emplace_back(go, r);
        for (auto &t : th) t.join();
        double worst = 0;
        for (double m : ms) worst = std::max(worst, m);
        const double sent = mode == ALL3 ? 3.0 * double(bytes) : double(bytes);
        return sent / 1e9 / (worst / 1e3);  // GB/s of the slowest sender
    };
    // warm barrier counter semantic: arrived counts across calls, so each run starts from a fresh counter
    std::printf("GB/s of the slowest sending GPU (best of 5)\n");
    std::printf("%-26s %8s", "traffic", "MB");
    const int block_sets[] = {16, 32, 64, 128, 256};
    for (int b : block_sets) std::printf(" %7d", b);
    std::printf("   blocks\n");
    struct Case {
        const char *name;
        Mode mode;
        int a, b;
    };
    const Case cases[] = {{"pair 0->1 alone", PAIR, 0, 1}, {"pair 0->2 alone", PAIR, 0, 2}, {"pair 0->3 alone", PAIR, 0, 3},
                          {"pair 1->0 alone", PAIR, 1, 0}, {"pair 2->3 alone", PAIR, 2, 3}, {"rotation +1 (all 4 send)", ROT, 1, 0},
                          {"rotation +2 (all 4 send)", ROT, 2, 0}, {"rotation +3 (all 4 send)", ROT, 3, 0},
                          {"all peers at once (x3)", ALL3, 0, 0}};
    for (double mb : {0.5, 2.6, 8.0})
        for (const Case &c : cases) {
            std::printf("%-26s %8.1f", c.name, mb);
            for (int b : block_sets) std::printf(" %7.1f", run(c.mode, c.a, c.b, size_t(mb * 1e6) / 16 * 16, b, 5));
            std::printf("\n");
        }
}

int main(int argc, char **argv) {
    try {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess || n < RANKS) {
            std::printf("need %d GPUs, found %d; skipping\n", RANKS, n);
            return 0;
        }
        const std::string mode = argc > 1 ? argv[1] : "engine";
        if (mode == "engine")
            engine_mode(argc > 2 ? std::atoi(argv[2]) : 200);
        else if (mode == "raw")
            raw_mode();
        else
            std::printf("usage: comm_bench engine [iters] | raw\n");
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
