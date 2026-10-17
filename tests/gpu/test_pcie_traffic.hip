// The engine's PCIe byte counts (comm/pcie_traffic.hpp) on all 4 GPUs: collectives count their payload once per peer
// in each direction, host <-> GPU copies count on the rank of the current device (device-to-device copies do not),
// and work captured into a graph counts once per launch, not at capture.
#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "comm/comm.hpp"
#include "comm/pcie_traffic.hpp"
#include "core/common.hpp"

using namespace qw;

#define CK(x)                                                                           \
    do {                                                                                \
        hipError_t err_ = (x);                                                          \
        if (err_ != hipSuccess) fail(std::string(#x) + ": " + hipGetErrorString(err_)); \
    } while (0)

namespace {
int fails = 0;

void check(const std::string &name, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
    fails += !ok;
}

std::array<pcie::Bytes, RANKS> since(const std::array<pcie::Bytes, RANKS> &from) {
    auto to = pcie::totals();
    for (int r = 0; r < RANKS; ++r)
        for (int k = 0; k < pcie::KINDS; ++k) to[size_t(r)][size_t(k)] -= from[size_t(r)][size_t(k)];
    return to;
}
}  // namespace

int main() {
    try {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess || n < RANKS) {
            std::printf("need %d GPUs, found %d; skipping\n", RANKS, n);
            return 0;
        }
        const std::array<int, RANKS> devices{0, 1, 2, 3};
        pcie::set_devices(devices);
        Comm comm(devices, 4u << 20);
        constexpr int ROWS = 64, W = 2560;  // fp32 rows: an allreduce of 640 KB, a reduce-scatter of 160 KB per peer
        constexpr int GRAPH_LAUNCHES = 5;
        constexpr size_t COPY = 3u << 20;

        // eager collectives: allreduce, reduce-scatter, allgather of known sizes
        auto start = pcie::totals();
        auto rank_eager = [&](int r) {
            CK(hipSetDevice(devices[size_t(r)]));
            hipStream_t s;
            CK(hipStreamCreate(&s));
            float *src, *dst;
            CK(hipMalloc(&src, size_t(ROWS) * W * 4));
            CK(hipMalloc(&dst, size_t(ROWS) * W * 4 * RANKS));
            CK(hipMemset(src, 0, size_t(ROWS) * W * 4));
            comm.set_base(r, 0, s);
            comm.allreduce(r, src, dst, W, 1, s, ROWS);
            comm.reduce_scatter(r, src, dst, W / RANKS, 2, s, ROWS);
            comm.allgather(r, src, dst, W, 3, s, ROWS);  // W bytes per row per rank
            CK(hipStreamSynchronize(s));
            CK(hipFree(src));
            CK(hipFree(dst));
            CK(hipStreamDestroy(s));
        };
        {
            std::vector<std::thread> th;
            for (int r = 0; r < RANKS; ++r) th.emplace_back(rank_eager, r);
            for (auto &t : th) t.join();
        }
        const uint64_t per_peer = uint64_t(ROWS) * W * 4 + uint64_t(ROWS) * (W / RANKS) * 4 + uint64_t(ROWS) * W;
        auto d = since(start);
        bool ok = true;
        for (int r = 0; r < RANKS; ++r)
            ok = ok && d[size_t(r)][pcie::PeerOut] == per_peer * (RANKS - 1) &&
                 d[size_t(r)][pcie::PeerIn] == per_peer * (RANKS - 1) && d[size_t(r)][pcie::HostToGpu] == 4 &&
                 d[size_t(r)][pcie::GpuToHost] == 0;
        std::printf("eager: rank 0 to peers %llu B (expected %llu), host->GPU %llu B\n",
                    (unsigned long long)d[0][pcie::PeerOut], (unsigned long long)(per_peer * (RANKS - 1)),
                    (unsigned long long)d[0][pcie::HostToGpu]);
        check("collectives: payload x 3 peers each way; set_base's 4-byte copy is the only host traffic", ok);

        // copies: host -> GPU and back on rank 2's device; device-to-device not counted
        start = pcie::totals();
        {
            CK(hipSetDevice(devices[2]));
            std::vector<uint8_t> host(COPY, 1);
            void *a, *b;
            CK(hipMalloc(&a, COPY));
            CK(hipMalloc(&b, COPY));
            CK(pcie::copy(a, host.data(), COPY, hipMemcpyHostToDevice));
            CK(pcie::copy_async(b, a, COPY, hipMemcpyDeviceToDevice, nullptr));
            CK(pcie::copy_async(host.data(), b, COPY / 2, hipMemcpyDeviceToHost, nullptr));
            // the prefill staging's copies count as their own kind on the rank they are made for
            CK(pcie::copy_async_as(pcie::SpillStageIn, 2, a, host.data(), COPY / 4, hipMemcpyHostToDevice, nullptr));
            CK(pcie::copy_async_as(pcie::SpillStageOut, 2, host.data(), a, COPY / 8, hipMemcpyDeviceToHost, nullptr));
            CK(hipDeviceSynchronize());
            CK(hipFree(a));
            CK(hipFree(b));
        }
        d = since(start);
        check("copies count on the current device's rank, device-to-device not at all",
              d[2][pcie::HostToGpu] == COPY && d[2][pcie::GpuToHost] == COPY / 2 && d[0][pcie::HostToGpu] == 0 &&
                  d[1][pcie::HostToGpu] == 0 && d[3][pcie::HostToGpu] == 0);
        check("copies made as a kind count as that kind only",
              d[2][pcie::SpillStageIn] == COPY / 4 && d[2][pcie::SpillStageOut] == COPY / 8 &&
                  d[0][pcie::SpillStageIn] == 0);

        // graphs: an allgather and a copy captured once, launched GRAPH_LAUNCHES times
        start = pcie::totals();
        std::array<pcie::Bytes, RANKS> tally{};
        std::array<uint64_t, RANKS> at_capture{};
        auto rank_graph = [&](int r) {
            CK(hipSetDevice(devices[size_t(r)]));
            hipStream_t s;
            CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
            void *src, *dst, *host;
            CK(hipMalloc(&src, size_t(ROWS) * W));
            CK(hipMalloc(&dst, size_t(ROWS) * W * RANKS));
            CK(hipHostMalloc(&host, 4096));
            CK(hipMemset(src, 0, size_t(ROWS) * W));
            hipGraph_t graph;
            hipGraphExec_t exec;
            CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
            {
                pcie::Capture cap(tally[size_t(r)]);
                comm.allgather(r, src, dst, W, 1, s, ROWS);
                CK(pcie::copy_async(host, src, 4096, hipMemcpyDeviceToHost, s));
            }
            CK(hipStreamEndCapture(s, &graph));
            CK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            at_capture[size_t(r)] = pcie::totals()[size_t(r)][pcie::PeerOut];
            for (int i = 0; i < GRAPH_LAUNCHES; ++i) {
                comm.set_base(r, uint32_t(3 + i), s);  // sequences 4, 5, ... after the eager 1..3
                CK(hipGraphLaunch(exec, s));
                pcie::add(r, tally[size_t(r)]);
                CK(hipStreamSynchronize(s));
            }
            CK(hipGraphExecDestroy(exec));
            CK(hipGraphDestroy(graph));
            CK(hipFree(src));
            CK(hipFree(dst));
            CK(hipHostFree(host));
            CK(hipStreamDestroy(s));
        };
        {
            std::vector<std::thread> th;
            for (int r = 0; r < RANKS; ++r) th.emplace_back(rank_graph, r);
            for (auto &t : th) t.join();
        }
        d = since(start);
        ok = true;
        for (int r = 0; r < RANKS; ++r)
            ok = ok && at_capture[size_t(r)] == start[size_t(r)][pcie::PeerOut] &&
                 d[size_t(r)][pcie::PeerOut] == uint64_t(GRAPH_LAUNCHES) * ROWS * W * (RANKS - 1) &&
                 d[size_t(r)][pcie::GpuToHost] == uint64_t(GRAPH_LAUNCHES) * 4096 &&
                 d[size_t(r)][pcie::HostToGpu] == uint64_t(GRAPH_LAUNCHES) * 4;
        check("a captured graph counts nothing at capture and its whole tally on each launch", ok);
        if (comm.error()) fail("comm: " + Comm::describe_error(comm.error()));
    } catch (const std::exception &e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
