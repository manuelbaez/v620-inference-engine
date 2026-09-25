// Where does large-payload collective time go? Measures, on devices 0 -> 1:
//   push into peer normal VRAM, push into peer uncached VRAM,
//   local read of normal VRAM, local read of uncached VRAM.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>

#define CK(x)                                                             \
    do {                                                                  \
        hipError_t e_ = (x);                                              \
        if (e_ != hipSuccess && e_ != hipErrorPeerAccessAlreadyEnabled) { \
            std::printf("%s: %s\n", #x, hipGetErrorString(e_));           \
            return 1;                                                     \
        }                                                                 \
    } while (0)

__global__ void copy_kernel(const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        dst[i] = src[i];
}

__global__ void read_kernel(const uint4 *__restrict__ src, size_t n, uint32_t *sink) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *sink = acc;
}

#include <thread>
#include <vector>

// All-to-all: every GPU pushes `per_peer` bytes to each other GPU concurrently.
static int all_to_all(size_t per_peer, int blocks_per_peer, bool include_self) {
    int n = 4;
    std::vector<void *> src(4), dst(4);
    for (int d = 0; d < n; ++d) {
        CK(hipSetDevice(d));
        for (int e = 0; e < n; ++e)
            if (e != d) CK(hipDeviceEnablePeerAccess(e, 0));
        (void)hipGetLastError();
        CK(hipMalloc(&src[size_t(d)], per_peer));
        CK(hipExtMallocWithFlags(&dst[size_t(d)], per_peer * 4, hipDeviceMallocUncached));
    }
    std::vector<double> secs(4);
    auto fn = [&](int d) {
        (void)hipSetDevice(d);
        hipStream_t st;
        (void)hipStreamCreate(&st);
        for (int it = 0; it < 6; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            for (int e = 0; e < n; ++e) {
                if (e == d && !include_self) continue;
                copy_kernel<<<blocks_per_peer, 256, 0, st>>>(
                    (uint4 *)src[size_t(d)], (uint4 *)((char *)dst[size_t(e)] + size_t(d) * per_peer), per_peer / 16);
            }
            (void)hipStreamSynchronize(st);
            if (it > 0)
                secs[size_t(d)] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 5;
        }
    };
    std::vector<std::thread> th;
    for (int d = 0; d < n; ++d) th.emplace_back(fn, d);
    for (auto &t : th) t.join();
    std::printf("all-to-all %6zu KB/peer, %3d blocks/peer%s: per-GPU out GB/s:", per_peer >> 10, blocks_per_peer,
                include_self ? " +self" : "");
    for (int d = 0; d < n; ++d) std::printf(" %6.1f", 3.0 * per_peer / secs[size_t(d)] / 1e9);
    std::printf("\n");
    for (int d = 0; d < n; ++d) {
        (void)hipSetDevice(d);
        (void)hipFree(src[size_t(d)]);
        (void)hipFree(dst[size_t(d)]);
    }
    return 0;
}

int main() {
    for (size_t kb : {640, 2560, 10240})
        for (int b : {16, 64, 256}) all_to_all(kb << 10, b, false);
    all_to_all(size_t(10240) << 10, 64, true);
    const size_t BYTES = 32u << 20, n16 = BYTES / 16;
    void *src, *dst_norm, *dst_unc, *loc_unc;
    uint32_t *sink;
    CK(hipSetDevice(1));
    CK(hipMalloc(&dst_norm, BYTES));
    CK(hipExtMallocWithFlags(&dst_unc, BYTES, hipDeviceMallocUncached));
    CK(hipSetDevice(0));
    CK(hipDeviceEnablePeerAccess(1, 0));
    CK(hipMalloc(&src, BYTES));
    CK(hipExtMallocWithFlags(&loc_unc, BYTES, hipDeviceMallocUncached));
    CK(hipMalloc(&sink, 4));
    auto time = [&](const char *name, auto fn) {
        fn();
        (void)hipDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) fn();
        (void)hipDeviceSynchronize();
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 5;
        std::printf("%-32s %7.1f GB/s\n", name, BYTES / s / 1e9);
    };
    for (int blocks : {64, 256, 1024}) {
        std::printf("-- %d blocks\n", blocks);
        time("push -> peer normal", [&] { copy_kernel<<<blocks, 256>>>((uint4 *)src, (uint4 *)dst_norm, n16); });
        time("push -> peer uncached", [&] { copy_kernel<<<blocks, 256>>>((uint4 *)src, (uint4 *)dst_unc, n16); });
        time("local read normal", [&] { read_kernel<<<blocks, 256>>>((uint4 *)src, n16, sink); });
        time("local read uncached", [&] { read_kernel<<<blocks, 256>>>((uint4 *)loc_unc, n16, sink); });
        time("local write uncached", [&] { copy_kernel<<<blocks, 256>>>((uint4 *)src, (uint4 *)loc_unc, n16); });
    }
    return 0;
}
