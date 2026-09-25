// Cost of a kernel launch inside a captured HIP graph on this GPU: a graph of
// n dependent tiny kernels, replayed; reports microseconds per kernel. Decode
// steps are ~1,600 kernels, so this bounds what fusing kernels can save.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>

__global__ void tiny(float *x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 0.999f + 1.f;
}

int main() {
    float *x;
    (void)hipMalloc(&x, 1 << 20);
    hipStream_t s;
    (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    for (int blocks : {1, 64}) {
        for (int n : {100, 1000}) {
            hipGraph_t g;
            hipGraphExec_t ge;
            (void)hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal);
            for (int i = 0; i < n; ++i) tiny<<<blocks, 256, 0, s>>>(x, blocks * 256);
            (void)hipStreamEndCapture(s, &g);
            (void)hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
            for (int w = 0; w < 3; ++w) (void)hipGraphLaunch(ge, s);
            (void)hipStreamSynchronize(s);
            const int reps = 20;
            const auto t0 = std::chrono::steady_clock::now();
            for (int w = 0; w < reps; ++w) (void)hipGraphLaunch(ge, s);
            (void)hipStreamSynchronize(s);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%4d kernels of %2d blocks per graph: %.2f us per kernel\n", n, blocks, us / reps / n);
        }
    }
    return 0;
}
