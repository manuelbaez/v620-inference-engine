// Checks the P2P collectives on all 4 GPUs and times them back to back.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
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

int main() {
    try {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess || n < RANKS) {
            std::printf("need %d GPUs, found %d; skipping\n", RANKS, n);
            return 0;
        }
        Comm comm({0, 1, 2, 3}, 16u << 20);
        const int N = 2560;
        constexpr int H4 = 2560;
        std::mt19937 rng(7);
        std::vector<std::vector<float>> in(RANKS, std::vector<float>(size_t(RANKS) * N));
        for (auto &v : in)
            for (auto &x : v) x = std::uniform_real_distribution<float>(-1, 1)(rng);

        std::vector<std::vector<float>> ar(RANKS), rs(RANKS), ag(RANKS);
        std::vector<double> us(RANKS), big_ms(RANKS), flat_ms(RANKS);
        // prefill-shaped: T rows
        const int T = 512, W = 1300;
        std::vector<std::vector<float>> pin(RANKS, std::vector<float>(size_t(T) * W));
        std::vector<std::vector<uint16_t>> hin(RANKS, std::vector<uint16_t>(size_t(T) * H4));
        for (auto &v : pin)
            for (auto &x : v) x = std::uniform_real_distribution<float>(-1, 1)(rng);
        for (auto &v : hin)
            for (auto &x : v) x = f32_to_f16(std::uniform_real_distribution<float>(-1, 1)(rng));
        std::vector<std::vector<float>> par(RANKS), prs(RANKS);
        std::vector<std::vector<uint16_t>> pag(RANKS);
        const int iters = 2000;
        auto rank_fn = [&](int r) {
            CK(hipSetDevice(comm.device(r)));
            hipStream_t s;
            CK(hipStreamCreate(&s));
            float *src, *dst;
            CK(hipMalloc(&src, in[size_t(r)].size() * 4));
            CK(hipMalloc(&dst, size_t(RANKS) * N * 4));
            CK(hipMemcpy(src, in[size_t(r)].data(), in[size_t(r)].size() * 4, hipMemcpyHostToDevice));

            comm.allreduce(r, src, dst, N, 1, s);
            ar[size_t(r)].resize(N);
            CK(hipMemcpyAsync(ar[size_t(r)].data(), dst, N * 4, hipMemcpyDeviceToHost, s));
            CK(hipStreamSynchronize(s));

            comm.reduce_scatter(r, src, dst, N / RANKS, 2, s);  // src holds RANKS parts of N/4
            rs[size_t(r)].resize(N / RANKS);
            CK(hipMemcpyAsync(rs[size_t(r)].data(), dst, N / RANKS * 4, hipMemcpyDeviceToHost, s));
            CK(hipStreamSynchronize(s));

            comm.allgather(r, src, dst, 1280, 3, s);
            ag[size_t(r)].resize(size_t(RANKS) * 320);
            CK(hipMemcpyAsync(ag[size_t(r)].data(), dst, RANKS * 1280, hipMemcpyDeviceToHost, s));
            CK(hipStreamSynchronize(s));

            // multi-row: allreduce T x 1300, fp16 reduce-scatter T x 2560 -> T x 640, allgather T x 640 fp16
            {
                float *ps, *pd;
                uint16_t *hs, *hd;
                CK(hipMalloc(&ps, size_t(T) * W * 4));
                CK(hipMalloc(&pd, size_t(T) * W * 4));
                CK(hipMalloc(&hs, size_t(T) * H4 * 2));
                CK(hipMalloc(&hd, size_t(T) * H4 * 4));
                CK(hipMemcpy(ps, pin[size_t(r)].data(), size_t(T) * W * 4, hipMemcpyHostToDevice));
                CK(hipMemcpy(hs, hin[size_t(r)].data(), size_t(T) * H4 * 2, hipMemcpyHostToDevice));
                comm.allreduce(r, ps, pd, W, 3000, s, T);
                par[size_t(r)].resize(size_t(T) * W);
                CK(hipMemcpyAsync(par[size_t(r)].data(), pd, size_t(T) * W * 4, hipMemcpyDeviceToHost, s));
                comm.reduce_scatter(r, hs, pd, H4 / RANKS, 3001, s, T, true);
                prs[size_t(r)].resize(size_t(T) * H4 / RANKS);
                CK(hipMemcpyAsync(prs[size_t(r)].data(), pd, size_t(T) * H4 / RANKS * 4, hipMemcpyDeviceToHost, s));
                comm.allgather(r, hs + size_t(r) * 0, hd, H4 / RANKS * 2, 3002, s, T);
                pag[size_t(r)].resize(size_t(T) * H4);
                CK(hipMemcpyAsync(pag[size_t(r)].data(), hd, size_t(T) * H4 * 2, hipMemcpyDeviceToHost, s));
                CK(hipStreamSynchronize(s));
                // bandwidth: 2048-row allreduce of 1300 floats (10.6 MB per rank)
                float *bs, *bd;
                CK(hipMalloc(&bs, size_t(2048) * W * 4));
                CK(hipMalloc(&bd, size_t(2048) * W * 4));
                auto b0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 10; ++i) comm.allreduce(r, bs, bd, W, 3003 + uint32_t(i), s, 2048);
                CK(hipStreamSynchronize(s));
                big_ms[size_t(r)] =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count() / 10 * 1e3;
                auto f0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 10; ++i) comm.allreduce(r, bs, bd, W * 2048, 3013 + uint32_t(i), s, 1);
                CK(hipStreamSynchronize(s));
                flat_ms[size_t(r)] =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count() / 10 * 1e3;
            }
            // latency: a chain of small allreduces (5.2 KB, the HC mix size)
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) comm.allreduce(r, src, dst, 1300, 3023 + uint32_t(i), s);
            CK(hipStreamSynchronize(s));
            us[size_t(r)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters * 1e6;
        };
        std::vector<std::thread> th;
        for (int r = 0; r < RANKS; ++r) th.emplace_back(rank_fn, r);
        for (auto &t : th) t.join();
        QW_CHECK(comm.error() == 0, "a receive timed out");

        int bad = 0;
        for (int r = 0; r < RANKS; ++r) {
            for (int i = 0; i < N; ++i) {
                float want = 0;
                for (int q = 0; q < RANKS; ++q) want += in[size_t(q)][size_t(i)];
                if (ar[size_t(r)][size_t(i)] != ar[0][size_t(i)] || std::fabs(ar[size_t(r)][size_t(i)] - want) > 1e-5f)
                    ++bad;
            }
            for (int i = 0; i < N / RANKS; ++i) {
                float want = 0;
                for (int q = 0; q < RANKS; ++q) want += in[size_t(q)][size_t(r * (N / RANKS) + i)];
                if (std::fabs(rs[size_t(r)][size_t(i)] - want) > 1e-5f) ++bad;
            }
            for (int q = 0; q < RANKS; ++q)
                for (int i = 0; i < 320; ++i)
                    if (ag[size_t(r)][size_t(q * 320 + i)] != in[size_t(q)][size_t(i)]) ++bad;
        }
        int pbad = 0;
        for (int r = 0; r < RANKS; ++r) {
            for (size_t i = 0; i < size_t(T) * W; ++i) {
                float want = 0;
                for (int q = 0; q < RANKS; ++q) want += pin[size_t(q)][i];
                if (par[size_t(r)][i] != par[0][i] || std::fabs(par[size_t(r)][i] - want) > 1e-4f) ++pbad;
            }
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < H4 / RANKS; ++i) {
                    float want = 0;
                    for (int q = 0; q < RANKS; ++q)
                        want += f16_to_f32(hin[size_t(q)][size_t(t) * H4 + size_t(r) * (H4 / RANKS) + i]);
                    if (std::fabs(prs[size_t(r)][size_t(t) * (H4 / RANKS) + i] - want) > 1e-3f) ++pbad;
                }
            for (int t = 0; t < T; ++t)
                for (int q = 0; q < RANKS; ++q)
                    for (int i = 0; i < H4 / RANKS; ++i)
                        if (pag[size_t(r)][size_t(t) * H4 + size_t(q) * (H4 / RANKS) + i] !=
                            hin[size_t(q)][size_t(t) * (H4 / RANKS) + i])
                            ++pbad;
        }
        bad += pbad;
        std::printf("multi-row (T=%d) mismatches: %d\n", T, pbad);
        std::printf("2048 x 1300 allreduce (10.6 MB/rank), ms by rank:");
        for (auto m : big_ms) std::printf(" %.2f", m);
        std::printf("\nsame bytes as one flat row, ms by rank:");
        for (auto m : flat_ms) std::printf(" %.2f", m);
        std::printf("\n");
        std::printf("allreduce / reduce_scatter / allgather mismatches: %d\n", bad);
        std::printf("chained 5.2 KB allreduce, us per op by rank:");
        for (auto u : us) std::printf(" %.1f", u);
        std::printf("\n%s\n", bad ? "FAILED" : "all ok");
        return bad ? 1 : 0;
    } catch (const std::exception &e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}
