// Stress test of the P2P collectives with on-device verification: every
// iteration fills each rank's source with exactly representable values
// (small integers as fp32), runs an all-reduce, a reduce-scatter and an
// all-gather over 1-16 rows (the decode shapes), and counts wrong elements on
// the device. Silent corruption or lost messages on the PCIe fabric show up as
// mismatches or a timeout.
// Usage: stress_comm [iterations, default 20000]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
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

namespace {

constexpr int AR_N = 1300, RS_N = 640, AG_BYTES = 1280, MAXR = 16;

// busy-waits ~cycles clock ticks (per-rank skew between collectives)
__global__ void spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}

__device__ float val(int rank, int it, int i) {
    return float((rank + 1) * ((i + it * 7) % 97));
}

__global__ void fill(float *ar_src, float *rs_src, float *ag_src, int rank, int it, int rows) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * RANKS * RS_N; i += gridDim.x * blockDim.x) {
        if (i < rows * AR_N) ar_src[i] = val(rank, it, i);
        rs_src[i] = val(rank, it, i);
        if (i < rows * AG_BYTES / 4) ag_src[i] = val(rank, it, i);
    }
}

// sum over ranks q of (q + 1) * v = 10 * v
__global__ void check(const float *ar, const float *rs, const float *ag, int rank, int it, int rows,
                      unsigned long long *bad) {
    int wrong = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * RANKS * RS_N; i += gridDim.x * blockDim.x) {
        if (i < rows * AR_N && ar[i] != 10.f * float((i + it * 7) % 97)) ++wrong;
        if (i < rows * RS_N) {  // row t, column j of this rank's part
            const int t = i / RS_N, j = i % RS_N, src = t * RANKS * RS_N + rank * RS_N + j;
            if (rs[i] != 10.f * float((src + it * 7) % 97)) ++wrong;
        }
        const int per = AG_BYTES / 4;
        if (i < rows * RANKS * per) {  // dst[t][q][..] = src_q[t][..]
            const int t = i / (RANKS * per), q = (i / per) % RANKS, j = i % per;
            if (ag[i] != float((q + 1) * ((t * per + j + it * 7) % 97))) ++wrong;
        }
    }
    if (wrong) atomicAdd(bad, (unsigned long long)wrong);
}

}  // namespace

int main(int argc, char **argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20000;
    try {
        Comm comm({0, 1, 2, 3}, 1u << 20);
        std::vector<unsigned long long> bad(RANKS, 0);
        std::vector<double> secs(RANKS, 0);
        auto rank_fn = [&](int r) {
            CK(hipSetDevice(comm.device(r)));
            hipStream_t s;
            CK(hipStreamCreate(&s));
            float *ar_src, *ar_dst, *rs_src, *rs_dst, *ag_src, *ag_dst;
            unsigned long long *d_bad;
            CK(hipMalloc(&ar_src, MAXR * AR_N * 4));
            CK(hipMalloc(&ar_dst, MAXR * AR_N * 4));
            CK(hipMalloc(&rs_src, MAXR * RANKS * RS_N * 4));
            CK(hipMalloc(&rs_dst, MAXR * RS_N * 4));
            CK(hipMalloc(&ag_src, MAXR * AG_BYTES));
            CK(hipMalloc(&ag_dst, MAXR * RANKS * AG_BYTES));
            CK(hipMalloc(&d_bad, 8));
            CK(hipMemset(d_bad, 0, 8));
            comm.set_base(r, 0, s);
            std::mt19937 rng(uint32_t(r) * 7919u + 1);
            const long max_skew = std::getenv("QW_STRESS_SKEW") ? std::atol(std::getenv("QW_STRESS_SKEW")) : 0;
            uint32_t k = 0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it) {
                const int rows = 1 + (it * 5) % MAXR;
                // skewed ranks: random busy time before each collective (QW_STRESS_SKEW=max cycles)
                auto skew = [&] {
                    if (max_skew) spin<<<1, 1, 0, s>>>(static_cast<long long>(rng() % uint32_t(max_skew)));
                };
                fill<<<32, 256, 0, s>>>(ar_src, rs_src, ag_src, r, it, rows);
                skew();
                comm.allreduce(r, ar_src, ar_dst, AR_N, ++k, s, rows);
                skew();
                comm.reduce_scatter(r, rs_src, rs_dst, RS_N, ++k, s, rows);
                skew();
                comm.allgather(r, ag_src, ag_dst, AG_BYTES, ++k, s, rows);
                check<<<32, 256, 0, s>>>(ar_dst, rs_dst, ag_dst, r, it, rows, d_bad);
                if (k > 3000000000u) break;
            }
            CK(hipStreamSynchronize(s));
            secs[size_t(r)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            CK(hipMemcpy(&bad[size_t(r)], d_bad, 8, hipMemcpyDeviceToHost));
        };
        std::vector<std::thread> th;
        for (int r = 0; r < RANKS; ++r) th.emplace_back(rank_fn, r);
        for (auto &t : th) t.join();
        unsigned long long total = 0;
        for (int r = 0; r < RANKS; ++r) {
            std::printf("rank %d: %llu wrong elements, %.1f s (%.1f us per collective)\n", r, bad[size_t(r)],
                        secs[size_t(r)], secs[size_t(r)] / (3.0 * iters) * 1e6);
            total += bad[size_t(r)];
        }
        std::printf("comm error: %s\n", Comm::describe_error(comm.error()).c_str());
        std::printf("%s\n", total || comm.error() ? "FAILED" : "all ok");
        return total || comm.error() ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
