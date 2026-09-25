// gemv_rows and gemv_rows_i8 (int8 weights, per-row scales) for every batch
// row count (1..16) and the model's shapes against a CPU reference, including
// the grouped (rows-per-group) form of the HC down projection and output rows
// with a stride.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/weight_convert.hpp"
#include "kernels/gemv.hpp"

using namespace qw;

#define CK(x)                                                                           \
    do {                                                                                \
        hipError_t err_ = (x);                                                          \
        if (err_ != hipSuccess) fail(std::string(#x) + ": " + hipGetErrorString(err_)); \
    } while (0)

int main() {
    try {
        CK(hipSetDevice(0));
        std::mt19937 rng(3);
        struct Shape {
            int N, K, xs, ys, rpg;
        };
        // sh_gu, router, HC down (grouped: 4 streams of 324 rows over [M][4][640]), o-proj, GDN proj, sh_down
        const Shape shapes[] = {{321, 2560, 2560, 321, 0},   {512, 2560, 2560, 512, 0},   {1296, 640, 2560, 1300, 324},
                                {2560, 1536, 1536, 2560, 0}, {4120, 2560, 2560, 4120, 0}, {2560, 160, 160, 2560, 0}};
        int fails = 0;
        for (const Shape &sh : shapes) {
            std::vector<uint16_t> W(size_t(sh.N) * sh.K), x(size_t(16) * sh.xs);
            for (auto &v : W) v = f32_to_f16(std::uniform_real_distribution<float>(-1, 1)(rng));
            for (auto &v : x) v = f32_to_f16(std::uniform_real_distribution<float>(-1, 1)(rng));
            uint16_t *dW, *dx;
            float *dy;
            const size_t ybytes = (size_t(16) * sh.ys + 4096) * 4;  // + guard
            CK(hipMalloc(&dW, W.size() * 2));
            CK(hipMalloc(&dx, x.size() * 2));
            CK(hipMalloc(&dy, ybytes));
            CK(hipMemcpy(dW, W.data(), W.size() * 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
            for (int M = 1; M <= 16; ++M) {
                std::vector<float> guard(ybytes / 4, 12345.f);
                CK(hipMemcpy(dy, guard.data(), ybytes, hipMemcpyHostToDevice));
                gpu::gemv_rows(dW, dx, sh.xs, dy, sh.ys, sh.N, sh.K, M, sh.rpg, false, nullptr);
                CK(hipDeviceSynchronize());
                std::vector<float> y(ybytes / 4);
                CK(hipMemcpy(y.data(), dy, ybytes, hipMemcpyDeviceToHost));
                double worst = 0;
                int overwritten = 0;
                for (int m = 0; m < M; ++m)
                    for (int n = 0; n < sh.N; ++n) {
                        const size_t xoff = size_t(m) * sh.xs + (sh.rpg ? size_t(n / sh.rpg) * sh.K : 0);
                        double ref = 0;
                        for (int k = 0; k < sh.K; ++k)
                            ref +=
                                double(f16_to_f32(W[size_t(n) * sh.K + size_t(k)])) * f16_to_f32(x[xoff + size_t(k)]);
                        worst = std::max(worst, std::fabs(ref - y[size_t(m) * sh.ys + size_t(n)]));
                    }
                // nothing written outside the M output rows' N columns
                for (size_t i = 0; i < y.size(); ++i) {
                    const size_t m = i / size_t(sh.ys), n = i % size_t(sh.ys);
                    if ((m >= size_t(M) || n >= size_t(sh.N)) && y[i] != 12345.f) ++overwritten;
                }
                const bool ok = worst < 1e-2 && overwritten == 0;
                fails += !ok;
                if (!ok)
                    std::printf("N=%d K=%d M=%d: max err %.4g, stray writes %d  FAIL\n", sh.N, sh.K, M, worst,
                                overwritten);
            }
            std::printf("N=%4d K=%4d rpg=%3d: M=1..16 %s\n", sh.N, sh.K, sh.rpg, fails ? "(see above)" : "ok");
            {  // int8 weights: against the same product with the dequantized weights
                std::vector<int8_t> q;
                std::vector<float> sc;
                quantize_rows_i8(W, sh.N, sh.K, q, sc, gpu::I8_GROUP);
                int8_t *dq;
                float *ds;
                CK(hipMalloc(&dq, q.size()));
                CK(hipMalloc(&ds, sc.size() * 4));
                CK(hipMemcpy(dq, q.data(), q.size(), hipMemcpyHostToDevice));
                CK(hipMemcpy(ds, sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
                double worst = 0, qerr = 0;
                for (int M = 1; M <= 16; ++M) {
                    gpu::gemv_rows_i8(dq, ds, dx, sh.xs, dy, sh.ys, sh.N, sh.K, M, sh.rpg, false, nullptr);
                    CK(hipDeviceSynchronize());
                    std::vector<float> y(ybytes / 4);
                    CK(hipMemcpy(y.data(), dy, ybytes, hipMemcpyDeviceToHost));
                    for (int m = 0; m < M; ++m)
                        for (int n = 0; n < sh.N; ++n) {
                            const size_t xoff = size_t(m) * sh.xs + (sh.rpg ? size_t(n / sh.rpg) * sh.K : 0);
                            double ref = 0, exact = 0;
                            for (int k = 0; k < sh.K; ++k) {
                                const double xv = f16_to_f32(x[xoff + size_t(k)]);
                                ref += double(q[size_t(n) * sh.K + size_t(k)]) *
                                       sc[size_t(n) * (sh.K / gpu::I8_GROUP) + size_t(k / gpu::I8_GROUP)] * xv;
                                exact += double(f16_to_f32(W[size_t(n) * sh.K + size_t(k)])) * xv;
                            }
                            worst = std::max(worst, std::fabs(ref - y[size_t(m) * sh.ys + size_t(n)]));
                            qerr = std::max(qerr, std::fabs(exact - ref));
                        }
                }
                const bool ok = worst < 1e-2;
                fails += !ok;
                std::printf("  int8: max err vs dequantized %.2e %s (quantization itself: %.2e)\n", worst,
                            ok ? "ok" : "FAIL", qerr);
                CK(hipFree(dq));
                CK(hipFree(ds));
            }
            CK(hipFree(dW));
            CK(hipFree(dx));
            CK(hipFree(dy));
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
