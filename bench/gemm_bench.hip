// rocBLAS fp16 GEMM throughput on this model's per-rank prefill shapes:
// Y[T][N] = X[T][K] * W[N][K]^T, fp16 inputs, fp32 accumulate, fp16 or fp32 out.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <chrono>
#include <cstdio>
#include <vector>

#define CK(x)                                                   \
    do {                                                        \
        hipError_t e_ = (x);                                    \
        if (e_ != hipSuccess) {                                 \
            std::printf("%s: %s\n", #x, hipGetErrorString(e_)); \
            return 1;                                           \
        }                                                       \
    } while (0)
#define RB(x)                                     \
    do {                                          \
        rocblas_status s_ = (x);                  \
        if (s_ != rocblas_status_success) {       \
            std::printf("%s: %d\n", #x, int(s_)); \
            return 1;                             \
        }                                         \
    } while (0)

int main() {
    rocblas_handle h;
    RB(rocblas_create_handle(&h));
    struct Shape {
        const char *name;
        int N, K;
    };
    const Shape shapes[] = {{"gdn in_proj", 4120, 2560},    {"gdn out_proj", 2560, 1536}, {"qsa in_proj", 4224, 2560},
                            {"hc up", 2560, 320},           {"router", 512, 2560},        {"shared gu", 321, 2560},
                            {"lm_head shard", 62080, 2560}, {"square 4096", 4096, 4096}};
    const int Ts[] = {256, 512, 1024, 2048, 4096};
    const size_t maxN = 62080, maxK = 4096, maxT = 4096;
    void *W, *X, *Y;
    CK(hipMalloc(&W, maxN * maxK * 2));
    CK(hipMalloc(&X, maxT * maxK * 2));
    CK(hipMalloc(&Y, maxT * maxN * 4));
    CK(hipMemset(W, 0, maxN * maxK * 2));
    CK(hipMemset(X, 0, maxT * maxK * 2));
    const float alpha = 1.f, beta = 0.f;
    std::printf("%-15s %-16s %6s %6s", "variant", "shape", "N", "K");
    for (int T : Ts) std::printf("  T=%-5d", T);
    std::printf("   (TFLOPS, fp32 out)\n");
    // variants: weights [N][K] (op T) or pre-transposed [K][N] (op N); fp32 or fp16 output
    struct Var {
        const char *name;
        rocblas_operation opA;
        bool f16out;
    };
    const Var vars[] = {{"W[N][K] f32out", rocblas_operation_transpose, false},
                        {"W[N][K] f16out", rocblas_operation_transpose, true},
                        {"W[K][N] f32out", rocblas_operation_none, false},
                        {"W[K][N] f16out", rocblas_operation_none, true}};
    for (auto &v : vars)
        for (auto &sh : shapes) {
            std::printf("%-15s %-16s %6d %6d", v.name, sh.name, sh.N, sh.K);
            for (int T : Ts) {
                auto run = [&] {
                    const int lda = v.opA == rocblas_operation_transpose ? sh.K : sh.N;
                    const auto ct = v.f16out ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
                    return rocblas_gemm_ex(h, v.opA, rocblas_operation_none, sh.N, T, sh.K, &alpha, W,
                                           rocblas_datatype_f16_r, lda, X, rocblas_datatype_f16_r, sh.K, &beta, Y, ct,
                                           sh.N, Y, ct, sh.N, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                };
                RB(run());
                CK(hipDeviceSynchronize());
                const int reps = 10;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) RB(run());
                CK(hipDeviceSynchronize());
                double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
                std::printf("  %7.1f", 2.0 * sh.N * sh.K * T / s / 1e12);
            }
            std::printf("\n");
        }
    return 0;
}
