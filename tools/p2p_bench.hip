// Measures GPU<->GPU transfer paths on this box:
//   - SDMA copy (hipMemcpyPeerAsync) bandwidth, 64 MiB
//   - kernel-initiated peer store (push) and peer load (pull) bandwidth
//   - small-message latency: a push of N bytes plus a flag that the
//     receiver polls in its own uncached VRAM (the collective primitive)
// Usage: p2p_bench [bytes_for_latency=10240]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                      \
    do {                                                                           \
        hipError_t err_ = (x);                                                       \
        if (err_ != hipSuccess && err_ != hipErrorPeerAccessAlreadyEnabled) {            \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(err_)); \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

__global__ void copy_kernel(const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

// Sender: write payload into the peer's buffer, then bump the flag behind it.
__global__ void push_msg(const uint4 *src, uint4 *peer_buf, size_t n, volatile uint32_t *peer_flag,
                         uint32_t seq) {
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) peer_buf[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *peer_flag = seq;
    }
}

// Receiver: poll own uncached flag.
__global__ void wait_flag(volatile uint32_t *flag, uint32_t seq) {
    if (threadIdx.x == 0) {
        long spins = 0;
        while (*flag < seq) {
            __builtin_amdgcn_s_sleep(1);
            if (++spins > (1L << 30)) break;  // bounded: never hang the queue
        }
    }
}

int main(int argc, char **argv) {
    size_t lat_bytes = argc > 1 ? size_t(std::atol(argv[1])) : 10240;
    int n = 0;
    CK(hipGetDeviceCount(&n));
    std::printf("devices: %d\n", n);
    for (int d = 0; d < n; ++d) {
        hipDeviceProp_t p;
        CK(hipGetDeviceProperties(&p, d));
        std::printf("  %d: %s pci %02x:%02x.0\n", d, p.gcnArchName, p.pciBusID, p.pciDeviceID);
    }
    const size_t BYTES = 64ull << 20;
    std::vector<void *> buf(static_cast<size_t>(n)), buf2(static_cast<size_t>(n));
    std::vector<uint32_t *> flag(static_cast<size_t>(n));
    std::vector<hipStream_t> st(static_cast<size_t>(n));
    for (int d = 0; d < n; ++d) {
        CK(hipSetDevice(d));
        for (int e = 0; e < n; ++e)
            if (e != d) CK(hipDeviceEnablePeerAccess(e, 0));
        (void)hipGetLastError();
        CK(hipMalloc(&buf[size_t(d)], BYTES));
        CK(hipExtMallocWithFlags(&buf2[size_t(d)], BYTES, hipDeviceMallocUncached));
        CK(hipExtMallocWithFlags(reinterpret_cast<void **>(&flag[size_t(d)]), 256, hipDeviceMallocUncached));
        CK(hipMemset(flag[size_t(d)], 0, 256));
        CK(hipStreamCreate(&st[size_t(d)]));
    }
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    const char *names[3] = {"SDMA copy", "kernel push", "kernel pull"};
    for (int mode = 0; mode < 3; ++mode) {
        std::printf("\n%s GB/s (row = src, col = dst)\n     ", names[mode]);
        for (int e = 0; e < n; ++e) std::printf("%7d", e);
        std::printf("\n");
        for (int s = 0; s < n; ++s) {
            std::printf("%4d ", s);
            for (int d = 0; d < n; ++d) {
                if (s == d) { std::printf("%7s", "-"); continue; }
                int run_dev = mode == 2 ? d : s;  // pull runs on the receiver
                CK(hipSetDevice(run_dev));
                auto go = [&] {
                    if (mode == 0)
                        CK(hipMemcpyPeerAsync(buf[size_t(d)], d, buf[size_t(s)], s, BYTES, st[size_t(run_dev)]));
                    else
                        copy_kernel<<<288, 256, 0, st[size_t(run_dev)]>>>(
                            static_cast<const uint4 *>(buf[size_t(s)]), static_cast<uint4 *>(buf[size_t(d)]),
                            BYTES / 16);
                };
                go();
                CK(hipStreamSynchronize(st[size_t(run_dev)]));
                auto t0 = now();
                const int reps = 5;
                for (int r = 0; r < reps; ++r) go();
                CK(hipStreamSynchronize(st[size_t(run_dev)]));
                std::printf("%7.1f", reps * BYTES / secs(t0, now()) / 1e9);
            }
            std::printf("\n");
        }
    }

    std::printf("\npush+flag latency, %zu bytes, us per message (row = src, col = dst)\n     ", lat_bytes);
    for (int e = 0; e < n; ++e) std::printf("%7d", e);
    std::printf("\n");
    uint32_t seq = 0;
    for (int s = 0; s < n; ++s) {
        std::printf("%4d ", s);
        for (int d = 0; d < n; ++d) {
            if (s == d) { std::printf("%7s", "-"); continue; }
            const int reps = 200;
            auto t0 = now();
            for (int r = 0; r < reps; ++r) {
                ++seq;
                CK(hipSetDevice(s));
                push_msg<<<1, 256, 0, st[size_t(s)]>>>(static_cast<const uint4 *>(buf[size_t(s)]),
                                                        static_cast<uint4 *>(buf2[size_t(d)]), lat_bytes / 16,
                                                        flag[size_t(d)], seq);
                CK(hipSetDevice(d));
                wait_flag<<<1, 32, 0, st[size_t(d)]>>>(flag[size_t(d)], seq);
                CK(hipStreamSynchronize(st[size_t(d)]));
            }
            std::printf("%7.1f", secs(t0, now()) / reps * 1e6);
        }
        std::printf("\n");
    }
    std::printf("(latency includes two launches and a host sync; the in-graph cost is lower)\n");
    return 0;
}
