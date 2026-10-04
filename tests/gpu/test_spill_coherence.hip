// Can the KV spill's host memory be mapped non-coherent (QW_SPILL_NC=1: the GPUs may cache its lines in L2)? The
// sequences the engine produces, each repeated, on a 1 MiB region (small enough to stay in a V620's 4 MiB L2):
//   1. a GPU reads the region, the CPU rewrites it (an import_kv into a spilled slot), the GPU reads it again;
//   2. a GPU writes the region (decode or prefill storing K/V), the stream is synchronized, the CPU reads it
//      (export_kv, a save to the prefix cache);
//   3. as 1, with the GPU's reads in a captured HIP graph (decode runs as graphs);
//   4. GPU 0 reads, GPU 1 writes and is synchronized, GPU 0 reads again (the two ranks of a KV pair share the buffer).
// Counts stale 32-bit words per sequence, for coherent and non-coherent mappings.
//   test_spill_coherence [--reps 200]
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"

using namespace qw;

constexpr size_t WORDS = (1u << 20) / 4;

__global__ void read_k(const uint32_t *p, uint32_t expect_seed, unsigned *stale) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < WORDS; i += size_t(gridDim.x) * blockDim.x)
        if (p[i] != uint32_t(i) * 2654435761u + expect_seed) atomicAdd(stale, 1u);
}
__global__ void write_k(uint32_t *p, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < WORDS; i += size_t(gridDim.x) * blockDim.x)
        p[i] = uint32_t(i) * 2654435761u + seed;
}
static void host_write(uint32_t *p, uint32_t seed) {
    for (size_t i = 0; i < WORDS; ++i) p[i] = uint32_t(i) * 2654435761u + seed;
}
static size_t host_stale(const uint32_t *p, uint32_t seed) {
    size_t n = 0;
    for (size_t i = 0; i < WORDS; ++i) n += p[i] != uint32_t(i) * 2654435761u + seed;
    return n;
}

int main(int argc, char **argv) {
    int reps = 200;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    int fails = 0;
    for (const bool nc : {false, true}) {
        const unsigned flags = hipHostMallocPortable | hipHostMallocMapped | (nc ? hipHostMallocNonCoherent : 0u);
        uint32_t *buf = nullptr;
        CK(hipHostMalloc(reinterpret_cast<void **>(&buf), WORDS * 4, flags));
        unsigned *stale[2] = {};
        hipStream_t st[2];
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipMalloc(&stale[d], 4));
            CK(hipStreamCreate(&st[d]));
        }
        const auto dev_stale = [&](int d) {
            unsigned v = 0;
            CK(hipSetDevice(d));
            CK(hipMemcpy(&v, stale[d], 4, hipMemcpyDeviceToHost));
            CK(hipMemset(stale[d], 0, 4));
            return size_t(v);
        };
        size_t s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        // sequence 3: two captured graphs, each one read of the region on GPU 0 checking a different seed
        hipGraphExec_t g[2];
        CK(hipSetDevice(0));
        for (int k = 0; k < 2; ++k) {
            hipGraph_t gr;
            CK(hipStreamBeginCapture(st[0], hipStreamCaptureModeThreadLocal));
            read_k<<<64, 256, 0, st[0]>>>(buf, 7000u + uint32_t(k), stale[0]);
            CK(hipStreamEndCapture(st[0], &gr));
            CK(hipGraphInstantiate(&g[k], gr, nullptr, nullptr, 0));
            CK(hipGraphDestroy(gr));
        }
        for (int rep = 0; rep < reps; ++rep) {
            const uint32_t a = 1000u + uint32_t(rep) * 2, b = a + 1;
            // 1. GPU reads, CPU rewrites, GPU reads
            host_write(buf, a);
            CK(hipSetDevice(0));
            read_k<<<64, 256, 0, st[0]>>>(buf, a, stale[0]);
            CK(hipStreamSynchronize(st[0]));
            host_write(buf, b);
            read_k<<<64, 256, 0, st[0]>>>(buf, b, stale[0]);
            CK(hipStreamSynchronize(st[0]));
            s1 += dev_stale(0);
            // 2. GPU writes, sync, CPU reads
            CK(hipSetDevice(0));
            write_k<<<64, 256, 0, st[0]>>>(buf, a + 500000u);
            CK(hipStreamSynchronize(st[0]));
            s2 += host_stale(buf, a + 500000u);
            // 3. in graphs: read (seed 7000), CPU rewrites, read (seed 7001)
            host_write(buf, 7000u);
            CK(hipSetDevice(0));
            CK(hipGraphLaunch(g[0], st[0]));
            CK(hipStreamSynchronize(st[0]));
            host_write(buf, 7001u);
            CK(hipGraphLaunch(g[1], st[0]));
            CK(hipStreamSynchronize(st[0]));
            s3 += dev_stale(0);
            // 4. GPU 0 reads, GPU 1 writes, GPU 0 reads
            host_write(buf, a);
            CK(hipSetDevice(0));
            read_k<<<64, 256, 0, st[0]>>>(buf, a, stale[0]);
            CK(hipStreamSynchronize(st[0]));
            CK(hipSetDevice(1));
            write_k<<<64, 256, 0, st[1]>>>(buf, b + 900000u);
            CK(hipStreamSynchronize(st[1]));
            CK(hipSetDevice(0));
            read_k<<<64, 256, 0, st[0]>>>(buf, b + 900000u, stale[0]);
            CK(hipStreamSynchronize(st[0]));
            s4 += dev_stale(0);
        }
        std::printf("%-13s stale words over %d repetitions: 1. CPU rewrite after GPU read %zu | 2. CPU read after GPU write %zu"
                    " | 3. as 1 in graphs %zu | 4. GPU 1 write after GPU 0 read %zu\n",
                    nc ? "non-coherent" : "coherent", reps, s1, s2, s3, s4);
        fails += (s1 + s2 + s3 + s4) != 0;
        CK(hipHostFree(buf));
    }
    std::printf("%s\n", fails ? "STALE DATA SEEN" : "no stale data");
    return fails ? 1 : 0;
}
