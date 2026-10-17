#include "comm/pcie_traffic.hpp"

#include <atomic>

namespace qw::pcie {

namespace {
std::array<std::array<std::atomic<uint64_t>, KINDS>, cfg::RANKS> g_bytes{};
std::array<std::atomic<int>, cfg::RANKS> g_device{};  // HIP device + 1 (0: not set)
thread_local Bytes *t_capture = nullptr;

void count_copy(size_t bytes, hipMemcpyKind kind) {
    if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost) return;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) return;
    for (int r = 0; r < cfg::RANKS; ++r)
        if (g_device[size_t(r)].load(std::memory_order_relaxed) == dev + 1) {
            add(r, kind == hipMemcpyHostToDevice ? HostToGpu : GpuToHost, bytes);
            return;
        }
}
}  // namespace

void set_devices(const std::array<int, cfg::RANKS> &devices) {
    for (int r = 0; r < cfg::RANKS; ++r) g_device[size_t(r)].store(devices[size_t(r)] + 1, std::memory_order_relaxed);
}

void add(int r, Kind kind, uint64_t bytes) {
    if (t_capture)
        (*t_capture)[size_t(kind)] += bytes;
    else
        g_bytes[size_t(r)][size_t(kind)].fetch_add(bytes, std::memory_order_relaxed);
}

void add(int r, const Bytes &tally) {
    for (int k = 0; k < KINDS; ++k)
        if (tally[size_t(k)]) add(r, Kind(k), tally[size_t(k)]);
}

std::array<Bytes, cfg::RANKS> totals() {
    std::array<Bytes, cfg::RANKS> out{};
    for (int r = 0; r < cfg::RANKS; ++r)
        for (int k = 0; k < KINDS; ++k)
            out[size_t(r)][size_t(k)] = g_bytes[size_t(r)][size_t(k)].load(std::memory_order_relaxed);
    return out;
}

Capture::Capture(Bytes &tally) : prev_(t_capture) {
    tally = {};
    t_capture = &tally;
}

Capture::~Capture() {
    t_capture = prev_;
}

hipError_t copy(void *dst, const void *src, size_t bytes, hipMemcpyKind kind) {
    count_copy(bytes, kind);
    return hipMemcpy(dst, src, bytes, kind);
}

hipError_t copy_async(void *dst, const void *src, size_t bytes, hipMemcpyKind kind, hipStream_t s) {
    count_copy(bytes, kind);
    return hipMemcpyAsync(dst, src, bytes, kind, s);
}

hipError_t copy_async_as(Kind as, int r, void *dst, const void *src, size_t bytes, hipMemcpyKind kind, hipStream_t s) {
    add(r, as, bytes);
    return hipMemcpyAsync(dst, src, bytes, kind, s);
}

}  // namespace qw::pcie
