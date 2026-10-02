// The collectives' kernel variants (Comm::Tuning) give bit-identical results to the plain kernels: the same
// random inputs through all-gather, fp16 reduce-scatter, fp16 and fp32 all-reduce at prefill shapes, under each
// combination of the settings; every output must equal the plain kernels' bit for bit, and every rank's all-reduce
// result must equal rank 0's.
#include <hip/hip_runtime.h>

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
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

int main() {
    try {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess || n < RANKS) {
            std::printf("need %d GPUs, found %d; skipping\n", RANKS, n);
            return 0;
        }
        Comm comm({0, 1, 2, 3}, 32u << 20);
        constexpr int H = cfg::H, SHARD = cfg::SH, DPAD = 328;
        const int row_sets[] = {256, 640, 2048};
        const Comm::Tuning variants[] = {{false, false}, {true, false}, {false, true}, {true, true}};
        const char *vname[] = {"plain", "push2d", "vecrecv", "both"};
        // results[variant][rank]: every output of every op and rows, concatenated as raw bytes
        std::vector<std::vector<std::vector<uint8_t>>> results(4, std::vector<std::vector<uint8_t>>(RANKS));
        Barrier bar(RANKS);
        auto rank_fn = [&](int r) {
            CK(hipSetDevice(comm.device(r)));
            hipStream_t s;
            CK(hipStreamCreate(&s));
            const size_t maxrows = 2048;
            std::mt19937 rng(1000 + unsigned(r));
            std::vector<uint16_t> h(maxrows * H);
            std::vector<float> f(maxrows * 4);
            for (auto &x : h) x = f32_to_f16(std::uniform_real_distribution<float>(-2, 2)(rng));
            for (auto &x : f) x = std::uniform_real_distribution<float>(-1, 1)(rng);
            uint16_t *d16;
            float *d32, *dst;
            CK(hipMalloc(&d16, h.size() * 2));
            CK(hipMalloc(&d32, f.size() * 4));
            CK(hipMalloc(&dst, maxrows * H * 4));
            CK(hipMemcpy(d16, h.data(), h.size() * 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(d32, f.data(), f.size() * 4, hipMemcpyHostToDevice));
            uint32_t k = 0;
            for (int v = 0; v < 4; ++v) {
                bar.wait();
                if (r == 0) comm.set_tuning(variants[v]);
                bar.wait();
                std::vector<uint8_t> &out = results[size_t(v)][size_t(r)];
                for (int rows : row_sets) {
                    auto grab = [&](size_t bytes) {
                        CK(hipStreamSynchronize(s));
                        const size_t o = out.size();
                        out.resize(o + bytes);
                        CK(hipMemcpy(out.data() + o, dst, bytes, hipMemcpyDeviceToHost));
                    };
                    CK(hipMemset(dst, 0xcc, maxrows * H * 4));
                    comm.allgather(r, d16, dst, size_t(SHARD) * 2, ++k, s, rows);
                    grab(size_t(rows) * H * 2);
                    CK(hipMemset(dst, 0xcc, maxrows * H * 4));
                    comm.reduce_scatter(r, d16, dst, SHARD, ++k, s, rows, true);
                    grab(size_t(rows) * SHARD * 4);
                    CK(hipMemset(dst, 0xcc, maxrows * H * 4));
                    comm.allreduce(r, d16, dst, DPAD, ++k, s, rows, true);
                    grab(size_t(rows) * DPAD * 4);
                    CK(hipMemset(dst, 0xcc, maxrows * H * 4));
                    comm.allreduce(r, d32, dst, 4, ++k, s, rows, false);
                    grab(size_t(rows) * 4 * 4);
                    // an fp32 reduce-scatter and all-reduce of wide rows, the fp32 reduction path
                    CK(hipMemset(dst, 0xcc, maxrows * H * 4));
                    comm.reduce_scatter(r, reinterpret_cast<const float *>(d16), dst, 64, ++k, s, rows, false);
                    grab(size_t(rows) * 64 * 4);
                }
            }
            CK(hipStreamDestroy(s));
        };
        std::vector<std::thread> th;
        for (int r = 0; r < RANKS; ++r) th.emplace_back(rank_fn, r);
        for (auto &t : th) t.join();
        QW_CHECK(comm.error() == 0, "a receive timed out");
        int fails = 0;
        for (int v = 1; v < 4; ++v)
            for (int r = 0; r < RANKS; ++r) {
                const bool same = results[size_t(v)][size_t(r)] == results[0][size_t(r)];
                std::printf("%-8s rank %d: %zu bytes %s\n", vname[v], r, results[size_t(v)][size_t(r)].size(),
                            same ? "identical to plain" : "DIFFERENT FROM PLAIN");
                fails += !same;
            }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
